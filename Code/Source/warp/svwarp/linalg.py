# SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
# SPDX-License-Identifier: BSD-3-Clause

"""Device resident FSILS matrix (dof = 4, one process), the diagonal
preconditioner, the sub-blocks of the NS solver, vector kernels and the
kernels of the resistance (coupled Neumann) faces."""

import numpy as np
import warp as wp

from .common import DEVICE, F, BD, vec8

wp.set_module_options({"enable_backward": False, "fuse_fp": False})


# ---------------------------------------------------------------------------
# Matrix structure, assembly staging and preconditioner

class Lhs:
    """CSR block structure in the FSILS ordering (identical to the mesh
    ordering with one process), the 4x4 block values and the scaling W."""

    def __init__(self, nNo, offsets, columns):
        self.nNo = int(nNo)
        self.nnz = len(columns)
        offsets = np.asarray(offsets, dtype=np.int32)
        columns = np.asarray(columns, dtype=np.int32)
        self.offsets_np = offsets
        self.columns_np = columns
        rows = np.repeat(np.arange(self.nNo, dtype=np.int32), np.diff(offsets))
        self.rows_np = rows
        diag = np.nonzero(columns == rows)[0]
        if len(diag) != self.nNo:
            raise RuntimeError("svwarp: a row without diagonal entry")
        self.offsets = wp.array(offsets, dtype=wp.int32, device=DEVICE)
        self.columns = wp.array(columns, dtype=wp.int32, device=DEVICE)
        self.rows = wp.array(rows, dtype=wp.int32, device=DEVICE)
        self.diag = wp.array(diag.astype(np.int32), dtype=wp.int32, device=DEVICE)
        self.Val = wp.zeros(self.nnz, dtype=wp.mat44d, device=DEVICE)
        self.W = wp.zeros((self.nNo, 4), dtype=F, device=DEVICE)
        self.Dmul = wp.zeros((self.nNo, 4), dtype=F, device=DEVICE)
        self.stage_cap = 0

    def add_vals(self, ptr, vals, n):
        """Add n staged 4x4 blocks (host arrays) at the block indices ptr."""
        if n == 0:
            return
        if n > self.stage_cap:
            self.stage_cap = 2 * n
            self.stage_ptr = wp.empty(self.stage_cap, dtype=wp.int32, device=DEVICE)
            self.stage_val = wp.empty(self.stage_cap, dtype=wp.mat44d, device=DEVICE)
        wp.copy(self.stage_ptr, ptr, count=n)
        wp.copy(self.stage_val, vals, count=n)
        wp.launch(k_add_vals, dim=n, inputs=[self.stage_ptr, self.stage_val, self.Val], device=DEVICE)

    def precond_diag(self, Dmul, W):
        """FSILS diagonal preconditioner: W = Dmul / sqrt(|diag|), Val scaled
        by W on both sides. Dmul (host, nNo x 4) are the Dirichlet factors of
        precond_diag; W (host, nNo x 4) receives the scaling."""
        wp.copy(self.Dmul, Dmul)
        wp.launch(k_precond_w, dim=self.nNo, inputs=[self.diag, self.Val, self.Dmul, self.W], device=DEVICE)
        wp.launch(k_precond_scale, dim=self.nNo, inputs=[self.offsets, self.columns, self.W, self.Val], device=DEVICE)
        wp.copy(W, self.W)


@wp.kernel
def k_add_vals(ptr: wp.array(dtype=wp.int32), vals: wp.array(dtype=wp.mat44d), Val: wp.array(dtype=wp.mat44d)):
    i = wp.tid()
    wp.atomic_add(Val, ptr[i], vals[i])


@wp.kernel
def k_precond_w(diag: wp.array(dtype=wp.int32), Val: wp.array(dtype=wp.mat44d), Dmul: wp.array2d(dtype=F),
                W: wp.array2d(dtype=F)):
    a = wp.tid()
    d = Val[diag[a]]
    for i in range(4):
        w = d[i, i]
        if w == F(0.0):
            w = F(1.0)
        w = F(1.0) / wp.sqrt(wp.abs(w))
        W[a, i] = w * Dmul[a, i]


@wp.kernel
def k_precond_scale(offsets: wp.array(dtype=wp.int32), columns: wp.array(dtype=wp.int32), W: wp.array2d(dtype=F),
                    Val: wp.array(dtype=wp.mat44d)):
    # pre_mul (rows) then pos_mul (columns), as in precond.cpp
    i = wp.tid()
    for j in range(offsets[i], offsets[i + 1]):
        c = columns[j]
        m = Val[j]
        for r in range(4):
            for cc in range(4):
                m[r, cc] = m[r, cc] * W[i, r] * W[c, cc]
        Val[j] = m


# ---------------------------------------------------------------------------
# Sub-blocks of the NS solver (ns_solver.cpp depart): K (3x3), G (3x1), D (1x3), L

@wp.kernel
def k_depart(Val: wp.array(dtype=wp.mat44d), mK: wp.array(dtype=wp.mat33d), mG: wp.array(dtype=wp.vec3d),
             mD: wp.array(dtype=wp.vec3d), mL: wp.array(dtype=F)):
    j = wp.tid()
    m = Val[j]
    mK[j] = wp.mat33d(m[0, 0], m[0, 1], m[0, 2], m[1, 0], m[1, 1], m[1, 2], m[2, 0], m[2, 1], m[2, 2])
    mG[j] = wp.vec3d(m[0, 3], m[1, 3], m[2, 3])
    mD[j] = wp.vec3d(m[3, 0], m[3, 1], m[3, 2])
    mL[j] = m[3, 3]


# ---------------------------------------------------------------------------
# Vector kernels. Scalars (results of reductions) live in device arrays, so
# that whole solver phases can be captured in CUDA graphs. Kernels with
# block-wide reductions (wp.tile) run with pad(n) threads (see common.py).

@wp.kernel
def k_copy(src: wp.array(dtype=F), dst: wp.array(dtype=F)):
    i = wp.tid()
    dst[i] = src[i]


@wp.kernel
def k_publish8(src: wp.array(dtype=F), dst: wp.array(dtype=vec8), v0: wp.int32):
    # dst[v0 + t] = src[8 (v0 + t) .. +8] (dst: mapped host memory, 64-byte writes)
    t = wp.tid()
    v = vec8()
    for k in range(8):
        v[k] = src[8 * (v0 + t) + k]
    dst[v0 + t] = v


@wp.kernel
def k_fetch8(src: wp.array(dtype=vec8), dst: wp.array(dtype=F)):
    t = wp.tid()
    v = src[t]
    for k in range(8):
        dst[8 * t + k] = v[k]


@wp.kernel
def k_zero(x: wp.array(dtype=F), i0: wp.int32):
    i = wp.tid()
    x[i0 + i] = F(0.0)


@wp.kernel
def k_zero2d(x: wp.array2d(dtype=F)):
    i, j = wp.tid()
    x[i, j] = F(0.0)


@wp.kernel
def k_split4(R: wp.array2d(dtype=F), Rm: wp.array(dtype=F), Rc: wp.array(dtype=F), Rm2: wp.array(dtype=F),
             Rc2: wp.array(dtype=F)):
    a = wp.tid()
    for k in range(3):
        v = R[a, k]
        Rm[3 * a + k] = v
        Rm2[3 * a + k] = v
    Rc[a] = R[a, 3]
    Rc2[a] = R[a, 3]


@wp.kernel
def k_merge4(Rm: wp.array(dtype=F), Rc: wp.array(dtype=F), R: wp.array2d(dtype=F)):
    a = wp.tid()
    R[a, 0] = Rm[3 * a]
    R[a, 1] = Rm[3 * a + 1]
    R[a, 2] = Rm[3 * a + 2]
    R[a, 3] = Rc[a]


@wp.kernel
def k_dot(n: wp.int32, a: wp.array(dtype=F), b: wp.array(dtype=F), sc: wp.array(dtype=F), slot: wp.int32):
    # sc[slot] += a . b (sc[slot] must be zero before the kernel)
    i = wp.tid()
    v = F(0.0)
    if i < n:
        v = a[i] * b[i]
    t = wp.tile(v)
    s = wp.tile_sum(t)
    wp.tile_atomic_add(sc, s, offset=slot)


@wp.kernel
def k_dots(n: wp.int32, m: wp.int32, U: wp.array2d(dtype=F), k: wp.int32, b: wp.array(dtype=F),
           sc: wp.array(dtype=F), slot: wp.int32):
    # sc[slot + j] += U[j] . U[k] for j < m, sc[slot + m] += U[k] . b
    i = wp.tid()
    uk = F(0.0)
    if i < n:
        uk = U[k, i]
    for j in range(m):
        v = F(0.0)
        if i < n:
            v = U[j, i] * uk
        t = wp.tile(v)
        s = wp.tile_sum(t)
        wp.tile_atomic_add(sc, s, offset=slot + j)
    v = F(0.0)
    if i < n:
        v = b[i] * uk
    t = wp.tile(v)
    s = wp.tile_sum(t)
    wp.tile_atomic_add(sc, s, offset=slot + m)


@wp.kernel
def k_lin_comb(coef: wp.array(dtype=F), off: wp.int32, U: wp.array2d(dtype=F), X: wp.array(dtype=F)):
    # X += sum_{j<m} coef[off+1+j] U[j] with m = coef[off]
    i = wp.tid()
    m = wp.int32(coef[off])
    s = X[i]
    for j in range(m):
        s = s + coef[off + 1 + j] * U[j, i]
    X[i] = s


@wp.kernel
def k_lin_comb_from(base: wp.array(dtype=F), coef: wp.array(dtype=F), off: wp.int32, U: wp.array2d(dtype=F),
                    X: wp.array(dtype=F), sc: wp.array(dtype=F), zslot: wp.int32):
    # X = base - sum_{j<m} coef[off+1+j] U[j] with m = coef[off]; sc[zslot] = 0
    i = wp.tid()
    m = wp.int32(coef[off])
    s = F(0.0)
    for j in range(m):
        s = s + coef[off + 1 + j] * U[j, i]
    X[i] = base[i] - s
    if i == 0:
        sc[zslot] = F(0.0)


# ---------------------------------------------------------------------------
# Coupled Neumann (resistance) faces (add_bc_mul of the FSILS solvers):
# ADD: Y += res v (v . X); PRE: Y += -res/(1 + res |v|^2) v (v . Y). The face
# vectors v are kept as dense vectors Vf (nF x 3 nNo) and the solver uses
# the precomputed products G^T v (Schur complement) and K^T v (GMRES).

@wp.kernel
def k_face_scatter(nF: wp.int32, off: wp.array(dtype=wp.int32), glob: wp.array(dtype=wp.int32),
                   fd: wp.array(dtype=F), Vf: wp.array2d(dtype=F)):
    a = wp.tid()
    f = int(0)
    for g in range(nF):
        if a >= off[g]:
            f = g
    Ac = glob[a]
    for k in range(3):
        Vf[f, 3 * Ac + k] = fd[3 * a + k]


@wp.kernel
def k_face_gt_v(nF: wp.int32, off: wp.array(dtype=wp.int32), glob: wp.array(dtype=wp.int32), fd: wp.array(dtype=F),
                offsets: wp.array(dtype=wp.int32), columns: wp.array(dtype=wp.int32), G: wp.array(dtype=wp.vec3d),
                g: wp.array2d(dtype=F)):
    # g[f] += G^T v_f (v_f nonzero on the face nodes only)
    a = wp.tid()
    f = int(0)
    for q in range(nF):
        if a >= off[q]:
            f = q
    Ac = glob[a]
    v = wp.vec3d(fd[3 * a], fd[3 * a + 1], fd[3 * a + 2])
    for j in range(offsets[Ac], offsets[Ac + 1]):
        wp.atomic_add(g, f, columns[j], wp.dot(G[j], v))


@wp.kernel
def k_face_kt_v(nF: wp.int32, off: wp.array(dtype=wp.int32), glob: wp.array(dtype=wp.int32), fd: wp.array(dtype=F),
                offsets: wp.array(dtype=wp.int32), columns: wp.array(dtype=wp.int32), K: wp.array(dtype=wp.mat33d),
                kv: wp.array2d(dtype=F)):
    # kv[f] += K^T v_f
    a = wp.tid()
    f = int(0)
    for q in range(nF):
        if a >= off[q]:
            f = q
    Ac = glob[a]
    v = wp.vec3d(fd[3 * a], fd[3 * a + 1], fd[3 * a + 2])
    for j in range(offsets[Ac], offsets[Ac + 1]):
        c = columns[j]
        r = wp.transpose(K[j]) * v
        for k in range(3):
            wp.atomic_add(kv, f, 3 * c + k, r[k])
