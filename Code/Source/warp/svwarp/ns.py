# SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
# SPDX-License-Identifier: BSD-3-Clause

"""The FSILS NS linear solver (linear_solver/ns_solver.cpp, gmres.cpp,
cgrad.cpp) on the device resident matrix. The algorithm of the bi-partitioned
solver is the same as on the CPU: GMRES on the momentum block (with the
resistance terms of the coupled faces) and CG on the Schur complement
L + G^T G. The Schur complement is formed explicitly and the CG is
preconditioned with an aggregation AMG V-cycle (the CPU CG is not
preconditioned); the convergence criteria are those of the CPU solver.

Every phase of the solver (setup, the Arnoldi steps, chunks of CG
iterations, the outer iteration) is a CUDA graph of fused kernels; the
scalars that the host needs (norms, Hessenberg columns, Gram matrices)
are written by the last kernel of a graph into mapped pinned memory and read
after one stream synchronization, and the coefficients computed on the host
are read by a kernel from mapped memory. The host code only steers the
iterations; this keeps the small-mesh cost at a few launches per phase."""

import os
import sys
import time

import numpy as np
import scipy.sparse
import warp as wp

from . import linalg as la
from . import sell
from . import sell32 as s32
from .amg import Amg
from .common import DEVICE, F, F32, BD, Mapped, pad, pad8, sync, vec8

wp.set_module_options({"enable_backward": False, "fuse_fp": False})

# SVWARP_NOGRAPH=1 launches the kernels one by one (e.g. with SVWARP_DEBUG=1)
USE_GRAPHS = os.environ.get("SVWARP_NOGRAPH") is None
PROFILE = os.environ.get("SVWARP_PROFILE") is not None
MAXF = 8   # maximum number of coupled faces

# Slots of the solver scalars sc (double, device)
RZ, PQ, RR, RES_M, RES_C, EPS_M, EPS_C = 0, 2, 3, 4, 5, 6, 7    # RZ: two slots (parity)
ACC = 8                                                       # MAXF face accumulators
GRAM = 16                                                     # Gram matrix dots (8 iB + 12 slots)
# Slots of the GMRES scalars gs: norms, face dots (two parities), Hessenberg columns
NRM = 0
VW = 2
KW = VW + 2 * MAXF
GH = KW + 2 * MAXF
# Coefficient buffer (host -> device): three segments [count, values...]
# of COEF_SEG = max(sD, 2 iB) + 2 entries each


class SubLs:
    def __init__(self, mItr, sD, absTol, relTol):
        self.mItr, self.sD, self.absTol, self.relTol = int(mItr), int(sD), float(absTol), float(relTol)
        self.itr = 0
        self.success = False
        self.iNorm = self.fNorm = self.dB = 0.0


# ---------------------------------------------------------------------------
# Setup kernels

@wp.kernel
def k_add_L(Lmap: wp.array(dtype=wp.int32), L: wp.array(dtype=F), S: wp.array(dtype=F)):
    q = wp.tid()
    S[Lmap[q]] = L[q]


@wp.kernel
def k_gtg(rows: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32), columns: wp.array(dtype=wp.int32),
          tpos: wp.array(dtype=wp.int32), G: wp.array(dtype=wp.vec3d), S_off: wp.array(dtype=wp.int32),
          S_col: wp.array(dtype=wp.int32), S: wp.array(dtype=F)):
    # S += G^T G: the entry (i, k) of the pattern adds G_ki G_kj to S_ij for
    # the entries (k, j) of row k (the pattern of S is the 2-ring of the
    # pattern of G with sorted columns: binary search of j in row i of S)
    q = wp.tid()
    i = rows[q]
    k = columns[q]
    gki = G[tpos[q]]
    s0 = S_off[i]
    s1 = S_off[i + 1] - 1
    for q2 in range(offsets[k], offsets[k + 1]):
        j = columns[q2]
        v = wp.dot(gki, G[q2])
        lo = s0
        hi = s1
        while lo < hi:
            mid = (lo + hi) // 2
            if S_col[mid] < j:
                lo = mid + 1
            else:
                hi = mid
        wp.atomic_add(S, lo, v)


# ---------------------------------------------------------------------------
# Kernels of the AMG preconditioned CG (cgrad::schur). Scalars: sc[RZ + b]
# with the parity b of the iteration (r.z of the current and the previous
# iteration), sc[PQ], sc[RR], sc[ACC + f] = g_f . p of the faces. Each kernel
# zeroes the accumulators that a later kernel of the iteration fills.

@wp.kernel
def k_cg_init(n: wp.int32, src: wp.array(dtype=F), r: wp.array(dtype=F), X: wp.array(dtype=F), p: wp.array(dtype=F),
              zs: wp.array(dtype=F32), Dinv: wp.array(dtype=F32), wv: wp.array(dtype=F32), sc: wp.array(dtype=F),
              nc1: wp.int32, rc1: wp.array(dtype=F32)):
    # r = src, X = p = 0, sc[RR] += |r|^2, zs = w D^-1 r (pre-smoothing)
    i = wp.tid()
    v = F(0.0)
    if i < n:
        v = src[i]
        r[i] = v
        X[i] = F(0.0)
        p[i] = F(0.0)
        zs[i] = wv[0] * Dinv[i] * F32(v)
    t = wp.tile(v * v)
    s = wp.tile_sum(t)
    wp.tile_atomic_add(sc, s, offset=RR)
    if i == 0:
        sc[RZ + 1] = F(1.0)
        sc[PQ] = F(0.0)
    if i < MAXF:
        sc[ACC + i] = F(0.0)
    if i < nc1:
        rc1[i] = 0.0


@wp.kernel
def k_cg_dir(n: wp.int32, sc: wp.array(dtype=F), b: wp.int32, z: wp.array(dtype=F), p: wp.array(dtype=F),
             nF: wp.int32, g: wp.array2d(dtype=F)):
    # p = z + (rz_new / rz_old) p; sc[ACC + f] += g_f . p
    i = wp.tid()
    beta = sc[RZ + b] / sc[RZ + 1 - b]
    pi = F(0.0)
    if i < n:
        pi = z[i] + beta * p[i]
        p[i] = pi
    for f in range(nF):
        v = F(0.0)
        if i < n:
            v = g[f, i] * pi
        t = wp.tile(v)
        s = wp.tile_sum(t)
        wp.tile_atomic_add(sc, s, offset=ACC + f)
    if i == 0:
        sc[PQ] = F(0.0)


@wp.kernel
def k_cg_q(n: wp.int32, slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32), V: wp.array(dtype=F32),
           p: wp.array(dtype=F), q: wp.array(dtype=F), sc: wp.array(dtype=F), nF: wp.int32, fd: wp.array(dtype=F),
           cpre_off: wp.int32, g: wp.array2d(dtype=F)):
    # q = S p + sum_f cpre_f (g_f . p) g_f; sc[PQ] += p . q
    i = wp.tid()
    v = F(0.0)
    if i < n:
        s = i // sell.C
        q1 = slice_off[s + 1]
        acc = F(0.0)
        for k in range(slice_off[s] + i % sell.C, q1, sell.U * sell.C):
            c0 = cols[k]
            c1 = cols[k + sell.C]
            c2 = cols[k + 2 * sell.C]
            c3 = cols[k + 3 * sell.C]
            acc = acc + (F(V[k]) * p[c0] + F(V[k + sell.C]) * p[c1] + F(V[k + 2 * sell.C]) * p[c2]
                         + F(V[k + 3 * sell.C]) * p[c3])
        for f in range(nF):
            acc = acc + fd[cpre_off + f] * sc[ACC + f] * g[f, i]
        q[i] = acc
        v = p[i] * acc
    t = wp.tile(v)
    s2 = wp.tile_sum(t)
    wp.tile_atomic_add(sc, s2, offset=PQ)
    if i == 0:
        sc[RR] = F(0.0)


@wp.kernel
def k_cg_update(n: wp.int32, sc: wp.array(dtype=F), b: wp.int32, p: wp.array(dtype=F), q: wp.array(dtype=F),
                X: wp.array(dtype=F), r: wp.array(dtype=F), Dinv: wp.array(dtype=F32), wv: wp.array(dtype=F32),
                zs: wp.array(dtype=F32), nc1: wp.int32, rc1: wp.array(dtype=F32)):
    # alpha = rz / pq; X += alpha p; r -= alpha q; sc[RR] += |r|^2; zs = w D^-1 r
    i = wp.tid()
    alpha = sc[RZ + b] / sc[PQ]
    ri = F(0.0)
    if i < n:
        X[i] = X[i] + alpha * p[i]
        ri = r[i] - alpha * q[i]
        r[i] = ri
        zs[i] = wv[0] * Dinv[i] * F32(ri)
    t = wp.tile(ri * ri)
    s = wp.tile_sum(t)
    wp.tile_atomic_add(sc, s, offset=RR)
    if i == 0:
        sc[RZ + 1 - b] = F(0.0)
    if i < nc1:
        rc1[i] = 0.0


@wp.kernel
def k_cg_post(n: wp.int32, slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32),
              V: wp.array(dtype=F32), Dinv: wp.array(dtype=F32), wv: wp.array(dtype=F32), r: wp.array(dtype=F),
              z2: wp.array(dtype=F32), z: wp.array(dtype=F), sc: wp.array(dtype=F), rz_slot: wp.int32):
    # z = z2 + w D^-1 (r - S z2) (post-smoothing of the finest level, z2 the
    # prolongated correction, double output); sc[rz_slot] += r . z
    i = wp.tid()
    v = F(0.0)
    if i < n:
        acc = s32.row_f(i, slice_off, cols, V, z2)
        ri = r[i]
        zi = F(z2[i] + wv[0] * Dinv[i] * (F32(ri) - acc))
        z[i] = zi
        v = ri * zi
    t = wp.tile(v)
    s2 = wp.tile_sum(t)
    wp.tile_atomic_add(sc, s2, offset=rz_slot)
    if i < MAXF:
        sc[ACC + i] = F(0.0)


# ---------------------------------------------------------------------------
# Kernels of GMRES on the momentum block (gmres::gmres with dof = nsd).
# u_i is stored normalized by the kernel of step i from the unnormalized
# vector w of the previous step (norm |w|^2 in gs[NRM + b'], b' the parity
# of the previous step). The face terms need v_f . u_i and (K^T v_f) . u_i,
# which are accumulated as dots with w by the previous step: the coupled
# face products are applied without any extra reduction. Gram-Schmidt is the
# classical variant (all projections from one reduction kernel).

@wp.func
def face_coefs(nF: wp.int32, gs: wp.array(dtype=F), bp: wp.int32, inv: F, fd: wp.array(dtype=F), cadd_off: wp.int32,
               cpre_off: wp.int32, vv_off: wp.int32):
    # Coefficients s_f of the face vectors in w = K u + ADD + PRE (add_bc_mul
    # ADD then PRE, the PRE terms face by face as on the CPU)
    s = vec8()
    t = vec8()
    for f in range(nF):
        d = gs[VW + bp * MAXF + f] * inv          # v_f . u
        s[f] = fd[cadd_off + f] * d               # ADD: res_f v_f (v_f . u)
    for f in range(nF):
        e = gs[KW + bp * MAXF + f] * inv          # v_f . K u
        for g in range(nF):
            e = e + s[g] * fd[vv_off + f * MAXF + g]
        for g in range(f):
            e = e + fd[cpre_off + g] * t[g] * fd[vv_off + f * MAXF + g]
        t[f] = e                                  # v_f . w before the PRE term of face f
    for f in range(nF):
        s[f] = s[f] + fd[cpre_off + f] * t[f]
    return s


@wp.kernel
def k_gm_step1(nNo: wp.int32, i: wp.int32, b: wp.int32, slice_off: wp.array(dtype=wp.int32),
               cols: wp.array(dtype=wp.int32), K: wp.array(dtype=wp.mat33d), win: wp.array(dtype=F),
               u: wp.array2d(dtype=F), wout: wp.array(dtype=F), gs: wp.array(dtype=F), nF: wp.int32,
               Vf: wp.array2d(dtype=F), fd: wp.array(dtype=F), cadd_off: wp.int32, cpre_off: wp.int32,
               vv_off: wp.int32, sD: wp.int32):
    # u_i = win / |win|; wout = K u_i + face terms; gs[GH_b + j] += u_j . wout, j <= i
    a = wp.tid()
    bp = 1 - b
    inv = F(1.0) / wp.sqrt(gs[NRM + bp])
    s = face_coefs(nF, gs, bp, inv, fd, cadd_off, cpre_off, vv_off)
    acc = wp.vec3d()
    ua = wp.vec3d()
    if a < nNo:
        acc = sell.row_vv(a, slice_off, cols, K, win) * inv
        for f in range(nF):
            for k in range(3):
                acc[k] = acc[k] + s[f] * Vf[f, 3 * a + k]
        for k in range(3):
            wout[3 * a + k] = acc[k]
            ua[k] = win[3 * a + k] * inv
            u[i, 3 * a + k] = ua[k]
    for j in range(i):
        v = F(0.0)
        if a < nNo:
            v = u[j, 3 * a] * acc[0] + u[j, 3 * a + 1] * acc[1] + u[j, 3 * a + 2] * acc[2]
        t = wp.tile(v)
        s2 = wp.tile_sum(t)
        wp.tile_atomic_add(gs, s2, offset=GH + b * (sD + 1) + j)
    v = wp.dot(ua, acc)
    t = wp.tile(v)
    s2 = wp.tile_sum(t)
    wp.tile_atomic_add(gs, s2, offset=GH + b * (sD + 1) + i)
    if a == 0:
        gs[NRM + b] = F(0.0)
    if a < MAXF:
        gs[VW + b * MAXF + a] = F(0.0)
        gs[KW + b * MAXF + a] = F(0.0)


@wp.kernel
def k_gm_step2(n3: wp.int32, i: wp.int32, b: wp.int32, u: wp.array2d(dtype=F), w: wp.array(dtype=F),
               gs: wp.array(dtype=F), nF: wp.int32, Vf: wp.array2d(dtype=F), kv: wp.array2d(dtype=F), sD: wp.int32):
    # w -= sum_{j<=i} h_j u_j; gs[NRM + b] += |w|^2; face dots with w; gs[GH_b'] = 0
    k = wp.tid()
    wk = F(0.0)
    if k < n3:
        wk = w[k]
        for j in range(i + 1):
            wk = wk - gs[GH + b * (sD + 1) + j] * u[j, k]
        w[k] = wk
    t = wp.tile(wk * wk)
    s = wp.tile_sum(t)
    wp.tile_atomic_add(gs, s, offset=NRM + b)
    for f in range(nF):
        v = F(0.0)
        v2 = F(0.0)
        if k < n3:
            v = Vf[f, k] * wk
            v2 = kv[f, k] * wk
        t = wp.tile(v)
        s = wp.tile_sum(t)
        wp.tile_atomic_add(gs, s, offset=VW + b * MAXF + f)
        t = wp.tile(v2)
        s = wp.tile_sum(t)
        wp.tile_atomic_add(gs, s, offset=KW + b * MAXF + f)
    if k <= sD:
        gs[GH + (1 - b) * (sD + 1) + k] = F(0.0)


@wp.kernel
def k_gm_init1(n3: wp.int32, R: wp.array(dtype=F), w: wp.array(dtype=F), nF: wp.int32, Vf: wp.array2d(dtype=F),
               gs: wp.array(dtype=F)):
    # w = R; gs[VW + f] += v_f . R
    k = wp.tid()
    wk = F(0.0)
    if k < n3:
        wk = R[k]
        w[k] = wk
    for f in range(nF):
        v = F(0.0)
        if k < n3:
            v = Vf[f, k] * wk
        t = wp.tile(v)
        s = wp.tile_sum(t)
        wp.tile_atomic_add(gs, s, offset=VW + f)


@wp.kernel
def k_gm_init2(n3: wp.int32, w: wp.array(dtype=F), nF: wp.int32, Vf: wp.array2d(dtype=F), kv: wp.array2d(dtype=F),
               fd: wp.array(dtype=F), cpre_off: wp.int32, vv_off: wp.int32, gs: wp.array(dtype=F), sD: wp.int32):
    # PRE face terms of w (dots in gs[VW + f]); then the quantities of the
    # "previous step" of step 0 in the parity-1 slots: |w|^2, v_f . w, kv_f . w
    k = wp.tid()
    t = vec8()
    for f in range(nF):
        e = gs[VW + f]
        for g in range(f):
            e = e + fd[cpre_off + g] * t[g] * fd[vv_off + f * MAXF + g]
        t[f] = e
    wk = F(0.0)
    if k < n3:
        wk = w[k]
        for f in range(nF):
            wk = wk + fd[cpre_off + f] * t[f] * Vf[f, k]
        w[k] = wk
    tt = wp.tile(wk * wk)
    s = wp.tile_sum(tt)
    wp.tile_atomic_add(gs, s, offset=NRM + 1)
    for f in range(nF):
        v = F(0.0)
        v2 = F(0.0)
        if k < n3:
            v = Vf[f, k] * wk
            v2 = kv[f, k] * wk
        tt = wp.tile(v)
        s = wp.tile_sum(tt)
        wp.tile_atomic_add(gs, s, offset=VW + MAXF + f)
        tt = wp.tile(v2)
        s = wp.tile_sum(tt)
        wp.tile_atomic_add(gs, s, offset=KW + MAXF + f)
    if k <= sD:
        gs[GH + k] = F(0.0)


@wp.kernel
def k_face_dots(n3: wp.int32, nF: wp.int32, Vf: wp.array2d(dtype=F), X: wp.array(dtype=F), sc: wp.array(dtype=F),
                slot: wp.int32):
    # sc[slot + f] += v_f . X
    k = wp.tid()
    for f in range(nF):
        v = F(0.0)
        if k < n3:
            v = Vf[f, k] * X[k]
        t = wp.tile(v)
        s = wp.tile_sum(t)
        wp.tile_atomic_add(sc, s, offset=slot + f)


@wp.kernel
def k_K_faces(nNo: wp.int32, slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32),
              K: wp.array(dtype=wp.mat33d), X: wp.array(dtype=F), nF: wp.int32, Vf: wp.array2d(dtype=F),
              fd: wp.array(dtype=F), cadd_off: wp.int32, sc: wp.array(dtype=F), slot: wp.int32, scale: F,
              B: wp.array(dtype=F), Y: wp.array(dtype=F)):
    # Y = scale (K X + sum_f res_f (v_f . X) v_f) + B  (ADD face terms; sc[slot + f] = v_f . X)
    a = wp.tid()
    acc = sell.row_vv(a, slice_off, cols, K, X)
    for f in range(nF):
        cf = fd[cadd_off + f] * sc[slot + f]
        for k in range(3):
            acc[k] = acc[k] + cf * Vf[f, 3 * a + k]
    for k in range(3):
        Y[3 * a + k] = scale * acc[k] + B[3 * a + k]


class NsSolver:
    def __init__(self, lhs):
        self.lhs = lhs
        nNo, nnz = lhs.nNo, lhs.nnz
        self.nNo = nNo
        self.n3 = 3 * nNo
        self.verbose = os.environ.get("SV_GPU_VERBOSE") is not None
        self.mK = wp.zeros(nnz, dtype=wp.mat33d, device=DEVICE)
        self.mG = wp.zeros(nnz, dtype=wp.vec3d, device=DEVICE)
        self.mD = wp.zeros(nnz, dtype=wp.vec3d, device=DEVICE)
        self.mL = wp.zeros(nnz, dtype=F, device=DEVICE)
        # SELL copies of the sub-blocks for the products
        self.sellA = sell.Sell(lhs.offsets_np, lhs.columns_np)
        ta = self.sellA.total
        self.mKs = wp.zeros(ta, dtype=wp.mat33d, device=DEVICE)
        self.mGs = wp.zeros(ta, dtype=wp.vec3d, device=DEVICE)
        self.mDs = wp.zeros(ta, dtype=wp.vec3d, device=DEVICE)
        self.mLs = wp.zeros(ta, dtype=F, device=DEVICE)
        self.R = wp.zeros((nNo, 4), dtype=F, device=DEVICE)
        z3 = lambda: wp.zeros(self.n3, dtype=F, device=DEVICE)
        z1 = lambda: wp.zeros(nNo, dtype=F, device=DEVICE)
        self.Rm, self.Rmi, self.gR, self.gX = z3(), z3(), z3(), z3()
        self.zero3 = z3()
        self.wb = [z3(), z3()]
        self.Rc, self.Rci = z1(), z1()
        self.work = None
        self.face_key = None
        self.graphs = {}
        self.profiled = set()
        # Pattern of the Schur complement S = L + G^T G (scalar CSR, sorted)
        rows, cols = lhs.rows_np.astype(np.int64), lhs.columns_np.astype(np.int64)
        A = scipy.sparse.csr_matrix((np.ones(nnz), cols, lhs.offsets_np.astype(np.int64)), shape=(nNo, nNo))
        S = (A.T @ A).tocsr()
        S.sort_indices()
        S_off = S.indptr.astype(np.int64)
        S_col = S.indices.astype(np.int64)
        self.nnzS = int(S.nnz)
        key_S = np.repeat(np.arange(nNo, dtype=np.int64), np.diff(S_off)) * nNo + S_col
        key_L = rows * nNo + cols
        pos = np.searchsorted(key_S, key_L)
        if not np.array_equal(key_S[pos], key_L):
            raise RuntimeError("svwarp: the pattern of L is not contained in the pattern of G^T G")
        self.Lmap = wp.array(pos.astype(np.int32), dtype=wp.int32, device=DEVICE)
        # Position of the transposed entry (col, row) of every entry
        key_T = cols * nNo + rows
        tpos = np.searchsorted(key_L, key_T)
        if not np.array_equal(key_L[tpos], key_T):
            raise RuntimeError("svwarp: the matrix pattern is not symmetric")
        self.tpos = wp.array(tpos.astype(np.int32), dtype=wp.int32, device=DEVICE)
        self.S_off = wp.array(S_off.astype(np.int32), dtype=wp.int32, device=DEVICE)
        self.S_col = wp.array(S_col.astype(np.int32), dtype=wp.int32, device=DEVICE)
        self.Sval = wp.zeros(self.nnzS, dtype=F, device=DEVICE)
        self.sellS = sell.Sell(S_off, S_col)
        self.Ss = wp.zeros(self.sellS.total, dtype=F32, device=DEVICE)
        if os.environ.get("SVWARP_AMG_GRAPH", "S") == "mesh":
            self.amg = Amg(nNo, S_off, S_col, self.sellS, lhs.offsets_np, lhs.columns_np)
        else:
            self.amg = Amg(nNo, S_off, S_col, self.sellS)
        # CG arrays
        self.cg_X, self.cg_r, self.cg_z, self.cg_p, self.cg_q = z1(), z1(), z1(), z1(), z1()
        self.cg_zs = wp.zeros(nNo, dtype=F32, device=DEVICE)
        self.cg_z2 = wp.zeros(nNo, dtype=F32, device=DEVICE)
        # CG iterations per graph launch (one synchronization per launch; up
        # to CG_CHUNK - 1 iterations after the convergence are computed)
        self.cg_chunk = int(os.environ.get("SVWARP_CG_CHUNK", max(2, min(8, 400000 // nNo // 2 * 2))))
        self.set_faces((0, np.zeros(0, np.int32), np.zeros(1, np.int32), np.zeros(0), np.zeros(0)))
        for m in (sys.modules[__name__], la, sell, s32, sys.modules[Amg.__module__]):
            wp.load_module(module=m, device=DEVICE)
        if self.verbose:
            print(f"svwarp: NS solver nNo {nNo} nnz {nnz} nnz(S) {self.nnzS} AMG levels {self.amg.sizes()} "
                  f"SELL padding {self.sellA.padding:.2f} {self.sellS.padding:.2f} CG chunk {self.cg_chunk}", flush=True)
        self.t_gm = self.t_cg = self.t_all = self.t_setup = 0.0
        self.tt = [0.0, 0.0, 0.0, 0, 0.0, 0.0]
        self.n_call = 0
        self.n_gm = self.n_cg = 0

    # -----------------------------------------------------------------------
    def run(self, key, emit):
        """Launch the CUDA graph of the kernel sequence emit() (captured on
        first use). SVWARP_PROFILE=1 prints the GPU time of every kernel of
        the first launch of every phase."""
        if not USE_GRAPHS:
            emit()
            return
        if PROFILE and key not in self.profiled:
            self.profiled.add(key)
            with wp.ScopedTimer(f"SVWARP profile {key}", cuda_filter=wp.TIMING_KERNEL, synchronize=True):
                emit()
            return
        g = self.graphs.get(key)
        if g is None:
            with wp.ScopedCapture(device=DEVICE) as cap:
                emit()
            g = cap.graph
            self.graphs[key] = g
        wp.capture_launch(g)

    def tlaunch(self, kernel, n, inputs):
        """Launch of a kernel with block-wide reductions over n elements."""
        wp.launch(kernel, dim=pad(n), inputs=inputs, block_dim=BD, device=DEVICE)

    def alloc_work(self, iB, sD):
        w = self.work
        if w is not None and w["iB"] >= iB and w["sD"] >= sD:
            return
        nNo, n3 = self.nNo, self.n3
        ngs = pad8(GH + 2 * (sD + 1))
        self.nsc = pad8(GRAM + 8 * iB + 12)
        self.sc = wp.zeros(self.nsc, dtype=F, device=DEVICE)
        self.hsc = Mapped(self.nsc)
        self.coef_seg = max(sD, 2 * iB) + 2
        self.ncoef = pad8(3 * self.coef_seg)
        self.hcoef = Mapped(self.ncoef)
        self.dcoef = wp.zeros(self.ncoef, dtype=F, device=DEVICE)
        self.work = dict(iB=iB, sD=sD,
                         U=wp.zeros((iB, n3), dtype=F, device=DEVICE),
                         MU=wp.zeros((2 * iB, n3), dtype=F, device=DEVICE),
                         P=wp.zeros((iB, nNo), dtype=F, device=DEVICE),
                         MP=wp.zeros((2 * iB, nNo), dtype=F, device=DEVICE),
                         gu=wp.zeros((sD + 1, n3), dtype=F, device=DEVICE),
                         gs=wp.zeros(ngs, dtype=F, device=DEVICE), hgs=Mapped(ngs))
        self.graphs = {}

    def set_faces(self, face_data):
        """face_data: (nFaces, glob_all, off, valM_all, res) of this solve.
        The face vectors, G^T v and K^T v are computed by the setup graph
        from the staged data in hfd."""
        nF, glob_all, off, valM_all, res = face_data
        nF = int(nF)
        if nF > MAXF:
            raise RuntimeError(f"svwarp: more than {MAXF} coupled faces")
        total = int(off[-1]) if nF else 0
        key = (nF, tuple(int(off[f + 1] - off[f]) for f in range(nF)))
        if key != self.face_key:
            # New face structure: new arrays, the captured graphs are invalid
            self.face_key = key
            self.nF, self.face_total = nF, total
            na = max(nF, 1)
            self.Vf = wp.zeros((na, self.n3), dtype=F, device=DEVICE)
            self.kv = wp.zeros((na, self.n3), dtype=F, device=DEVICE)
            self.g = wp.zeros((na, self.nNo), dtype=F, device=DEVICE)
            self.face_off = wp.array(np.asarray(off[:nF + 1], dtype=np.int32), dtype=wp.int32, device=DEVICE)
            self.face_glob = wp.array(np.asarray(glob_all[:total], dtype=np.int32), dtype=wp.int32, device=DEVICE)
            self.cadd_off = 3 * total
            self.cpre_off = self.cadd_off + MAXF
            self.vv_off = self.cpre_off + MAXF
            nfd = self.vv_off + MAXF * MAXF
            self.hfd = Mapped(nfd)
            self.fd = wp.zeros(self.hfd.n, dtype=F, device=DEVICE)
            self.graphs = {}
        if nF == 0:
            return
        h = self.hfd.np
        v = np.asarray(valM_all[:3 * total], dtype=np.float64)
        h[:3 * total] = v
        h[self.cadd_off:self.cadd_off + nF] = res[:nF]
        dense = np.zeros((nF, self.n3))
        for f in range(nF):
            a0, a1 = int(off[f]), int(off[f + 1])
            idx = (3 * np.asarray(glob_all[a0:a1], dtype=np.int64)[:, None] + np.arange(3)[None, :]).ravel()
            dense[f, idx] += v[3 * a0:3 * a1]
        VV = dense @ dense.T
        for f in range(nF):
            h[self.cpre_off + f] = -res[f] / (1.0 + res[f] * VV[f, f])
            h[self.vv_off + f * MAXF:self.vv_off + f * MAXF + nF] = VV[f]

    # -----------------------------------------------------------------------
    # Kernel sequences of the phases (captured in CUDA graphs)

    def emit_start(self):
        nNo, n3, sc = self.nNo, self.n3, self.sc
        wp.launch(la.k_split4, dim=nNo, inputs=[self.R, self.Rm, self.Rc, self.Rmi, self.Rci], device=DEVICE)
        wp.launch(la.k_zero, dim=self.nsc, inputs=[sc, 0], device=DEVICE)
        self.tlaunch(la.k_dot, n3, [n3, self.Rm, self.Rm, sc, EPS_M])
        self.tlaunch(la.k_dot, nNo, [nNo, self.Rc, self.Rc, sc, EPS_C])
        self.publish()

    def emit_setup(self, nPow):
        lhs, nNo, n3 = self.lhs, self.nNo, self.n3
        wp.launch(la.k_depart, dim=lhs.nnz, inputs=[lhs.Val, self.mK, self.mG, self.mD, self.mL], device=DEVICE)
        self.sellA.gather(self.mK, self.mKs, sell.k_gather_m)
        self.sellA.gather(self.mG, self.mGs, sell.k_gather_v)
        self.sellA.gather(self.mD, self.mDs, sell.k_gather_v)
        self.sellA.gather(self.mL, self.mLs, sell.k_gather_s)
        nF = self.nF
        if nF:
            wp.launch(la.k_zero2d, dim=(nF, n3), inputs=[self.Vf], device=DEVICE)
            wp.launch(la.k_zero2d, dim=(nF, n3), inputs=[self.kv], device=DEVICE)
            wp.launch(la.k_zero2d, dim=(nF, nNo), inputs=[self.g], device=DEVICE)
            wp.launch(la.k_fetch8, dim=self.hfd.nvec, inputs=[self.hfd.dev8, self.fd], device=DEVICE)
            args = [nF, self.face_off, self.face_glob, self.fd]
            wp.launch(la.k_face_scatter, dim=self.face_total, inputs=args + [self.Vf], device=DEVICE)
            wp.launch(la.k_face_gt_v, dim=self.face_total, inputs=args + [lhs.offsets, lhs.columns, self.mG, self.g],
                      device=DEVICE)
            wp.launch(la.k_face_kt_v, dim=self.face_total, inputs=args + [lhs.offsets, lhs.columns, self.mK, self.kv],
                      device=DEVICE)
        wp.launch(la.k_zero, dim=self.nnzS, inputs=[self.Sval, 0], device=DEVICE)
        wp.launch(k_add_L, dim=lhs.nnz, inputs=[self.Lmap, self.mL, self.Sval], device=DEVICE)
        wp.launch(k_gtg, dim=lhs.nnz, inputs=[lhs.rows, lhs.offsets, lhs.columns, self.tpos, self.mG, self.S_off,
                                               self.S_col, self.Sval], device=DEVICE)
        self.sellS.gather(self.Sval, self.Ss, s32.k_gather_sf)
        self.amg.emit_numeric(self.Sval, self.Ss, nPow)
        wp.launch(la.k_copy, dim=n3, inputs=[self.Rm, self.gR], device=DEVICE)

    def pub(self, src, dst, lo, hi):
        """Write src[lo:hi] (rounded to vectors of 8) to the mapped host buffer dst."""
        v0 = lo // 8
        wp.launch(la.k_publish8, dim=(hi - 1) // 8 - v0 + 1, inputs=[src, dst.dev8, v0], device=DEVICE)

    def publish(self, lo=0, hi=16):
        self.pub(self.sc, self.hsc, lo, hi)

    def fetch_coef(self):
        wp.launch(la.k_fetch8, dim=self.hcoef.nvec, inputs=[self.hcoef.dev8, self.dcoef], device=DEVICE)

    def emit_cg_init(self, src):
        """r = src, X = 0, |r|^2 in sc[RR], first V-cycle z = M r, r.z in sc[RZ]."""
        nNo, sc, amg = self.nNo, self.sc, self.amg
        L0, L1 = amg.levels[0], amg.levels[1]
        wp.launch(la.k_zero, dim=4, inputs=[sc, 0], device=DEVICE)
        self.tlaunch(k_cg_init, nNo, [nNo, src, self.cg_r, self.cg_X, self.cg_p, self.cg_zs, L0.Dinv, amg.wv, sc,
                                      L1.n, L1.r])
        zc = amg.emit_down(self.cg_r, self.cg_zs)
        wp.launch(s32.k_prolong_f, dim=nNo, inputs=[self.cg_zs, L0.agg, L0.pb, zc, self.cg_z2], device=DEVICE)
        self.tlaunch(k_cg_post, nNo, [nNo, self.sellS.slice_off, self.sellS.cols, self.Ss, L0.Dinv, amg.wv, self.cg_r,
                                      self.cg_z2, self.cg_z, sc, RZ])

    def emit_cg_chunk(self):
        nNo, sc, amg = self.nNo, self.sc, self.amg
        L0, L1 = amg.levels[0], amg.levels[1]
        X, r, z, p, q, zs = self.cg_X, self.cg_r, self.cg_z, self.cg_p, self.cg_q, self.cg_zs
        for it in range(self.cg_chunk):
            b = it & 1
            self.tlaunch(k_cg_dir, nNo, [nNo, sc, b, z, p, self.nF, self.g])
            self.tlaunch(k_cg_q, nNo, [nNo, self.sellS.slice_off, self.sellS.cols, self.Ss, p, q, sc, self.nF, self.fd,
                                       self.cpre_off, self.g])
            self.tlaunch(k_cg_update, nNo, [nNo, sc, b, p, q, X, r, L0.Dinv, amg.wv, zs, L1.n, L1.r])
            zc = amg.emit_down(r, zs)
            wp.launch(s32.k_prolong_f, dim=nNo, inputs=[zs, L0.agg, L0.pb, zc, self.cg_z2], device=DEVICE)
            self.tlaunch(k_cg_post, nNo, [nNo, self.sellS.slice_off, self.sellS.cols, self.Ss, L0.Dinv, amg.wv, r,
                                          self.cg_z2, z, sc, RZ + 1 - b])
        self.publish()

    def emit_gm_init(self):
        w, n3 = self.work, self.n3
        gs = w["gs"]
        wp.launch(la.k_zero, dim=len(gs), inputs=[gs, 0], device=DEVICE)
        wp.launch(la.k_zero, dim=n3, inputs=[self.gX, 0], device=DEVICE)
        self.tlaunch(k_gm_init1, n3, [n3, self.gR, self.wb[1], self.nF, self.Vf, gs])
        self.emit_gm_init2()

    def emit_gm_init2(self):
        w, n3 = self.work, self.n3
        gs = w["gs"]
        self.tlaunch(k_gm_init2, n3, [n3, self.wb[1], self.nF, self.Vf, self.kv, self.fd, self.cpre_off, self.vv_off,
                                      gs, w["sD"]])
        self.pub(gs, w["hgs"], 0, 8)

    def emit_gm_restart(self):
        """w = R - K X - face terms (restart of GMRES after sD steps)."""
        w, n3, nNo = self.work, self.n3, self.nNo
        gs = w["gs"]
        wp.launch(la.k_zero, dim=len(gs), inputs=[gs, 0], device=DEVICE)
        self.tlaunch(k_face_dots, n3, [n3, self.nF, self.Vf, self.gX, gs, KW])
        wp.launch(k_K_faces, dim=nNo, inputs=[nNo, self.sellA.slice_off, self.sellA.cols, self.mKs, self.gX, self.nF,
                                               self.Vf, self.fd, self.cadd_off, gs, KW, F(-1.0), self.gR, self.wb[1]],
                  device=DEVICE)
        self.tlaunch(k_face_dots, n3, [n3, self.nF, self.Vf, self.wb[1], gs, VW])
        self.emit_gm_init2()

    def emit_gm_step(self, i):
        w, n3, nNo = self.work, self.n3, self.nNo
        gs, u, sD = w["gs"], w["gu"], w["sD"]
        b = i & 1
        win, wout = self.wb[1 - b], self.wb[b]
        self.tlaunch(k_gm_step1, nNo, [nNo, i, b, self.sellA.slice_off, self.sellA.cols, self.mKs, win, u, wout, gs,
                                       self.nF, self.Vf, self.fd, self.cadd_off, self.cpre_off, self.vv_off, sD])
        self.tlaunch(k_gm_step2, n3, [n3, i, b, u, wout, gs, self.nF, self.Vf, self.kv, sD])
        self.pub(gs, w["hgs"], 0, 8)
        g0 = GH + b * (sD + 1)
        self.pub(gs, w["hgs"], g0, g0 + i + 1)

    def emit_gm_fin(self):
        self.fetch_coef()
        wp.launch(la.k_lin_comb, dim=self.n3, inputs=[self.dcoef, 0, self.work["gu"], self.gX], device=DEVICE)

    def emit_A(self, i):
        """U_i = GMRES solution; P_i = Rc - D U_i; CG initialization on P_i."""
        w, nNo, n3 = self.work, self.nNo, self.n3
        wp.launch(la.k_copy, dim=n3, inputs=[self.gX, w["U"][i]], device=DEVICE)
        wp.launch(sell.k_sub_spmv_vs, dim=nNo, inputs=[self.sellA.slice_off, self.sellA.cols, self.mDs, w["U"][i],
                                                        self.Rc, w["P"][i]], device=DEVICE)
        self.emit_cg_init(w["P"][i])
        self.publish()

    def emit_B(self, i):
        """P_i = CG solution; MU_{2i} = G P_i; MU_{2i+1} = Rm - G P_i = next GMRES rhs."""
        w, nNo = self.work, self.nNo
        wp.launch(la.k_copy, dim=nNo, inputs=[self.cg_X, w["P"][i]], device=DEVICE)
        wp.launch(sell.k_spmv_sv_sub, dim=nNo, inputs=[self.sellA.slice_off, self.sellA.cols, self.mGs, w["P"][i],
                                                        self.Rm, w["MU"][2 * i], w["MU"][2 * i + 1]], device=DEVICE)
        wp.launch(la.k_copy, dim=self.n3, inputs=[w["MU"][2 * i + 1], self.gR], device=DEVICE)

    def emit_C(self, i):
        """U_i = GMRES solution; MU_{2i+1} = K U_i (+ faces); MP_{2i} = L P_i;
        MP_{2i+1} = D U_i; dots of the Gram matrix."""
        w, nNo, n3, sc = self.work, self.nNo, self.n3, self.sc
        U, MU, P, MP = w["U"], w["MU"], w["P"], w["MP"]
        iB2, iBB = 2 * i, 2 * i + 1
        wp.launch(la.k_copy, dim=n3, inputs=[self.gX, U[i]], device=DEVICE)
        wp.launch(la.k_zero, dim=self.nsc - ACC, inputs=[sc, ACC], device=DEVICE)
        self.tlaunch(k_face_dots, n3, [n3, self.nF, self.Vf, U[i], sc, ACC])
        wp.launch(k_K_faces, dim=nNo, inputs=[nNo, self.sellA.slice_off, self.sellA.cols, self.mKs, U[i], self.nF,
                                               self.Vf, self.fd, self.cadd_off, sc, ACC, F(1.0), self.zero3, MU[iBB]],
                  device=DEVICE)
        wp.launch(sell.k_spmv_ss, dim=nNo, inputs=[self.sellA.slice_off, self.sellA.cols, self.mLs, P[i], MP[iB2]],
                  device=DEVICE)
        wp.launch(sell.k_spmv_vs, dim=nNo, inputs=[self.sellA.slice_off, self.sellA.cols, self.mDs, U[i], MP[iBB]],
                  device=DEVICE)
        slot = GRAM
        for k in (iB2, iBB):
            self.tlaunch(la.k_dots, n3, [n3, k + 1, MU, k, self.Rmi, sc, slot])
            slot += k + 2
            self.tlaunch(la.k_dots, nNo, [nNo, k + 1, MP, k, self.Rci, sc, slot])
            slot += k + 2
        self.publish(GRAM, slot)

    def emit_D(self, i):
        """Rm = Rmi - sum c_j MU_j, Rc = Rci - sum c_j MP_j, the residual norm; next GMRES rhs = Rm."""
        w, nNo, n3, sc = self.work, self.nNo, self.n3, self.sc
        self.fetch_coef()
        wp.launch(la.k_lin_comb_from, dim=n3, inputs=[self.Rmi, self.dcoef, 0, w["MU"], self.Rm, sc, RES_M], device=DEVICE)
        wp.launch(la.k_lin_comb_from, dim=nNo, inputs=[self.Rci, self.dcoef, 0, w["MP"], self.Rc, sc, RES_C], device=DEVICE)
        self.tlaunch(la.k_dot, n3, [n3, self.Rm, self.Rm, sc, RES_M])
        self.tlaunch(la.k_dot, nNo, [nNo, self.Rc, self.Rc, sc, RES_C])
        wp.launch(la.k_copy, dim=n3, inputs=[self.Rm, self.gR], device=DEVICE)
        self.publish()

    def emit_E(self):
        """Final residual of the continuity part and the solution R = (sum xB_{2i+1} U_i, sum xB_{2i} P_i)."""
        w, nNo, n3, sc = self.work, self.nNo, self.n3, self.sc
        self.fetch_coef()
        wp.launch(la.k_lin_comb_from, dim=nNo, inputs=[self.Rci, self.dcoef, 0, w["MP"], self.Rc, sc, RES_C], device=DEVICE)
        self.tlaunch(la.k_dot, nNo, [nNo, self.Rc, self.Rc, sc, RES_C])
        wp.launch(la.k_zero, dim=n3, inputs=[self.Rmi, 0], device=DEVICE)
        wp.launch(la.k_zero, dim=nNo, inputs=[self.Rci, 0], device=DEVICE)
        cs = self.coef_seg
        wp.launch(la.k_lin_comb, dim=n3, inputs=[self.dcoef, cs, w["U"], self.Rmi], device=DEVICE)
        wp.launch(la.k_lin_comb, dim=nNo, inputs=[self.dcoef, 2 * cs, w["P"], self.Rci], device=DEVICE)
        wp.launch(la.k_merge4, dim=nNo, inputs=[self.Rmi, self.Rci, self.R], device=DEVICE)
        self.publish()

    # -----------------------------------------------------------------------
    def gmres(self, ls):
        """gX = inv(K) gR; ls accumulates itr."""
        w = self.work
        hgs = w["hgs"].np
        sD = ls.sD
        h = np.zeros((sD + 1, sD))
        y = np.zeros(sD)
        c = np.zeros(sD)
        s = np.zeros(sD)
        err = np.zeros(sD + 1)
        ls.success = False
        eps = 0.0
        last_i = 0
        for l in range(ls.mItr):
            if l == 0:
                self.run("gm_init", self.emit_gm_init)
            else:
                self.emit_gm_restart()
                ls.itr += 1
            sync()
            err[0] = np.sqrt(hgs[NRM + 1])
            if l == 0:
                eps = err[0]
                ls.iNorm = eps
                ls.fNorm = eps
                eps = max(ls.absTol, ls.relTol * eps)
            if err[0] == 0.0:
                ls.success = True
                ls.fNorm = ls.dB = 0.0
                return
            ls.dB = ls.fNorm
            for i in range(sD):
                last_i = i
                t0 = time.perf_counter()
                self.run(("gm", i), lambda: self.emit_gm_step(i))
                t1 = time.perf_counter()
                ls.itr += 1
                sync()
                t2 = time.perf_counter()
                self.tt[0] += t1 - t0
                self.tt[1] += t2 - t1
                self.tt[3] += 1
                b = i & 1
                h[:i + 1, i] = hgs[GH + b * (sD + 1):GH + b * (sD + 1) + i + 1]
                h[i + 1, i] = np.sqrt(hgs[NRM + b])
                for j in range(i):
                    tmp = c[j] * h[j, i] + s[j] * h[j + 1, i]
                    h[j + 1, i] = -s[j] * h[j, i] + c[j] * h[j + 1, i]
                    h[j, i] = tmp
                tmp = np.sqrt(h[i, i] ** 2 + h[i + 1, i] ** 2)
                c[i] = h[i, i] / tmp
                s[i] = h[i + 1, i] / tmp
                h[i, i] = tmp
                h[i + 1, i] = 0.0
                err[i + 1] = -s[i] * err[i]
                err[i] = c[i] * err[i]
                self.tt[2] += time.perf_counter() - t2
                if abs(err[i + 1]) < eps:
                    ls.success = True
                    break
            y[:last_i + 1] = err[:last_i + 1]
            for j in range(last_i, -1, -1):
                for k in range(j + 1, last_i + 1):
                    y[j] = y[j] - h[j, k] * y[k]
                y[j] = y[j] / h[j, j]
            hc = self.hcoef.np
            hc[0] = last_i + 1
            hc[1:last_i + 2] = y[:last_i + 1]
            self.run("gm_fin", self.emit_gm_fin)
            ls.fNorm = abs(err[last_i + 1])
            if ls.success:
                break
        ls.dB = 10.0 * np.log(ls.fNorm / ls.dB)

    def cg_schur(self, ls, err):
        """AMG preconditioned CG on the Schur complement, initialized by
        emit_cg_init (err = |r|^2 from it); the solution is in cg_X."""
        hsc = self.hsc.np
        ls.success = False
        iNorm = np.sqrt(err)
        ls.iNorm = iNorm
        eps = max(ls.absTol, ls.relTol * iNorm) ** 2
        errO = err
        nit = 0
        while nit < ls.mItr:
            if err < eps:
                ls.success = True
                break
            errO = err
            self.run("cg_chunk", self.emit_cg_chunk)
            nit += self.cg_chunk
            sync()
            err = float(hsc[RR])
        ls.fNorm = np.sqrt(err)
        ls.itr += nit
        ls.dB = 0.0 if errO < np.finfo(float).eps else 5.0 * np.log(err / errO)
        return nit

    # -----------------------------------------------------------------------
    def solve(self, R_host, faces, RI, GM, CG):
        """R_host (nNo x 4, host): right-hand side in, solution out. Returns
        the statistics (RI_itr, GM_itr, CG_itr, success, iNorm, fNorm, dB, Resm, Resc, time)."""
        t0 = time.perf_counter()
        nNo, n3 = self.nNo, self.n3
        iB = RI.mItr
        nB = 2 * iB
        self.alloc_work(iB, GM.sD)
        w = self.work
        hsc = self.hsc.np
        hc = self.hcoef.np
        wp.copy(self.R, R_host)
        self.run("start", self.emit_start)
        sync()
        rm2, rc2 = float(hsc[EPS_M]), float(hsc[EPS_C])
        eps = np.sqrt(rm2 + rc2)
        RI.iNorm = eps
        RI.fNorm = eps * eps
        RI.itr = GM.itr = CG.itr = 0
        RI.dB = 0.0
        RI.success = False
        Resc = Resm = 0
        eps = max(RI.absTol, RI.relTol * eps)
        finite_norm = np.isfinite(RI.iNorm)
        if not finite_norm or RI.iNorm <= eps:
            R_host.numpy()[:] = 0.0
            RI.success = finite_norm
            Resc = int(100.0 * rc2 / RI.fNorm) if finite_norm and RI.fNorm > 0.0 else 0
            Resm = 100 - Resc if finite_norm and RI.fNorm > 0.0 else 0
            RI.fNorm = RI.iNorm
        else:
            tc = time.perf_counter()
            self.set_faces(faces)
            nPow = self.amg.npow()
            self.run(("setup", nPow), lambda: self.emit_setup(nPow))
            sync()
            self.t_setup += time.perf_counter() - tc
            A = np.zeros((nB, nB))
            B = np.zeros(nB)
            xB = np.zeros(nB)
            oldxB = np.zeros(nB)
            for i in range(RI.mItr):
                iB2, iBB = 2 * i, 2 * i + 1
                RI.dB = RI.fNorm
                # U = inv(K) Rm
                t = time.perf_counter()
                self.gmres(GM)
                self.t_gm += time.perf_counter() - t
                # P = [L + G^t G]^-1 (Rc - D U)
                t = time.perf_counter()
                self.run(("A", i), lambda: self.emit_A(i))
                sync()
                if os.environ.get("SVWARP_DUMP") and not getattr(self, "_dumped", False):
                    self._dumped = True
                    np.savez(os.environ["SVWARP_DUMP"], S_offsets=self.S_off.numpy(), S_columns=self.S_col.numpy(),
                             S_values=self.Sval.numpy(), rhs=self.cg_r.numpy(), A_offsets=self.lhs.offsets_np,
                             A_columns=self.lhs.columns_np, face_g=self.g.numpy()[:self.nF],
                             face_coef=self.hfd.np[self.cpre_off:self.cpre_off + self.nF], relTol=CG.relTol,
                             absTol=CG.absTol)
                self.cg_schur(CG, float(hsc[RR]))
                self.t_cg += time.perf_counter() - t
                # MU1 = G P, MU2 = Rm - G P; U = inv(K) [Rm - G P]
                self.run(("B", i), lambda: self.emit_B(i))
                t = time.perf_counter()
                self.gmres(GM)
                self.t_gm += time.perf_counter() - t
                # MU2 = K U, MP1 = L P, MP2 = D U and the Gram matrix
                self.run(("C", i), lambda: self.emit_C(i))
                sync()
                tmp = hsc[GRAM:GRAM + 4 * i + 6 + 4 * i + 6]
                slot = 0
                for k in (iB2, iBB):
                    m = k + 2
                    vals = tmp[slot:slot + m] + tmp[slot + m:slot + 2 * m]
                    slot += 2 * m
                    for j in range(k + 1):
                        A[j, k] = vals[j]
                        A[k, j] = vals[j]
                    B[k] = vals[k + 1]
                size = iBB + 1
                scale = np.array([1.0 if A[j, j] == 0.0 else np.sqrt(A[j, j]) for j in range(size)])
                projected_rhs = B[:size] / scale
                projection = A[:size, :size] / scale[:, None] / scale[None, :]
                if not (np.all(np.isfinite(projection)) and np.all(np.isfinite(projected_rhs))):
                    break
                coefficients = np.linalg.lstsq(projection, projected_rhs, rcond=None)[0] / scale
                if not np.all(np.isfinite(coefficients)):
                    break
                oldxB = xB.copy()
                xB[:size] = coefficients
                hc[0] = size
                hc[1:size + 1] = xB[:size]
                self.run(("D", i), lambda: self.emit_D(i))
                sync()
                residual2 = float(hsc[RES_M] + hsc[RES_C])
                if not np.isfinite(residual2) or residual2 >= RI.fNorm:
                    xB = oldxB
                    break
                RI.fNorm = residual2
                RI.itr = i + 1
                if RI.fNorm <= eps * eps:
                    RI.success = True
                    break
            m = 2 * RI.itr
            hc[0] = m
            hc[1:m + 1] = xB[:m]
            cs = self.coef_seg
            hc[cs] = RI.itr
            hc[cs + 1:cs + 1 + RI.itr] = xB[1:m:2]
            hc[2 * cs] = RI.itr
            hc[2 * cs + 1:2 * cs + 1 + RI.itr] = xB[0:m:2]
            self.run("E", self.emit_E)
            sync()
            rc2 = float(hsc[RES_C])
            Resc = int(100.0 * rc2 / RI.fNorm) if RI.fNorm > 0.0 else 0
            Resm = 100 - Resc if RI.fNorm > 0.0 else 0
            RI.dB = 5.0 * np.log(RI.fNorm / RI.dB) if RI.fNorm > 0.0 and RI.dB > 0.0 else 0.0
            RI.fNorm = np.sqrt(RI.fNorm)
            wp.copy(R_host, self.R)
            sync()
        dt = time.perf_counter() - t0
        self.t_all += dt
        self.n_call += 1
        self.n_gm += GM.itr
        self.n_cg += CG.itr
        if os.environ.get("SVWARP_TIMING"):
            n = self.n_call
            print(f"SVWARP calls={n} per call: all={self.t_all / n:.4f} schur setup={self.t_setup / n:.4f} "
                  f"gmres={self.t_gm / n:.4f} ({1e3 * self.t_gm / max(1, self.n_gm):.3f} ms/it) "
                  f"cg={self.t_cg / n:.4f} ({1e3 * self.t_cg / max(1, self.n_cg):.3f} ms/it)", flush=True)
            k = max(1, self.tt[3])
            print(f"SVWARP gmres step: launch {1e6 * self.tt[0] / k:.1f} us, sync {1e6 * self.tt[1] / k:.1f} us, "
                  f"host {1e6 * self.tt[2] / k:.1f} us", flush=True)
        return (RI.itr, GM.itr, CG.itr, int(RI.success), RI.iNorm, RI.fNorm, RI.dB, Resm, Resc, dt)
