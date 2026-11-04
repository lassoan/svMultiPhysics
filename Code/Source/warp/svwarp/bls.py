# SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
# SPDX-License-Identifier: BSD-3-Clause

"""FSILS GMRES (gmres_v) and CG (cgrad_v) for the dof x dof block matrices
of the FSI (dof = 4) and mesh motion (dof = 3) equations on the device
resident matrix, preconditioned with an aggregation AMG V-cycle on the block
matrix (right preconditioner of GMRES; SVWARP_AMG=0 selects the plain FSILS
algorithms, with the iteration counts of the CPU solver).

AMG (svgpu_fsi_amg.inc of the CUDA experiment): unsmoothed aggregation of the
nodes, piecewise constant prolongation for every dof, Galerkin coarse
matrices (block sums), the pseudo-inverse of the coarsest matrix (at most
CMAX nodes, computed on the host). One damped block Jacobi pre- and
post-smoothing step on every level, damping omega / rho(Dinv A) from a power
iteration. The hierarchy is in float.

The kernels are generated per block size by make(DOF) (closure kernels with
the Warp matrix types of the blocks); the block matrices of the products are
arrays of matrices in the SELL layout (sell.py). As in ns.py, the phases are
CUDA graphs, the scalars live on the device and reach the host through
mapped memory."""

import os
import time

import numpy as np
import warp as wp

from . import linalg as la
from . import sell
from .amg import aggregate
from .common import DEVICE, F, F32, BD, Mapped, pad, pad8, sync

wp.set_module_options({"enable_backward": False, "fuse_fp": False})

USE_GRAPHS = os.environ.get("SVWARP_NOGRAPH") is None
C = sell.C
RED_BLOCKS = 512   # blocks of the Gram-Schmidt reduction kernels (grid-stride loop beyond)

# Slots of the solver scalars gs (double, device)
NRM = 0    # two slots (parity): |w|^2 of the current and the previous Arnoldi step
RZ = 2     # two slots (parity): r.z of the CG
PQ = 4
RR = 5
GH = 8     # Hessenberg columns (two parities of sD + 1)


def amg_enabled():
    return os.environ.get("SVWARP_AMG", "1") != "0"


_kernels = {}


def kernels(DOF):
    if DOF not in _kernels:
        _kernels[DOF] = make(DOF)
    return _kernels[DOF]


def make(DOF):
    """Kernels of the block size DOF (a module of closure kernels)."""
    DD = DOF * DOF
    MAT = wp.types.matrix(shape=(DOF, DOF), dtype=wp.float64)
    VEC = wp.types.vector(DOF, wp.float64)
    MATF = wp.types.matrix(shape=(DOF, DOF), dtype=wp.float32)
    VECF = wp.types.vector(DOF, wp.float32)
    MAT2 = wp.types.matrix(shape=(DOF, 2 * DOF), dtype=wp.float64)
    VECI = wp.types.vector(DOF, wp.int32)

    # -- matrix staging and preconditioner ---------------------------------

    @wp.kernel
    def k_add_vals(ptr: wp.array(dtype=wp.int32), vals: wp.array(dtype=MAT), Val: wp.array(dtype=MAT)):
        i = wp.tid()
        wp.atomic_add(Val, ptr[i], vals[i])

    @wp.kernel
    def k_precond_w(diag: wp.array(dtype=wp.int32), Val: wp.array(dtype=MAT), Dmul: wp.array2d(dtype=F),
                    W: wp.array2d(dtype=F)):
        a = wp.tid()
        d = Val[diag[a]]
        for i in range(DOF):
            w = d[i, i]
            if w == F(0.0):
                w = F(1.0)
            w = F(1.0) / wp.sqrt(wp.abs(w))
            W[a, i] = w * Dmul[a, i]

    @wp.kernel
    def k_precond_scale(offsets: wp.array(dtype=wp.int32), columns: wp.array(dtype=wp.int32),
                        W: wp.array2d(dtype=F), Val: wp.array(dtype=MAT)):
        i = wp.tid()
        for j in range(offsets[i], offsets[i + 1]):
            c = columns[j]
            m = Val[j]
            for r in range(DOF):
                for cc in range(DOF):
                    m[r, cc] = m[r, cc] * W[i, r] * W[c, cc]
            Val[j] = m

    @wp.kernel
    def k_zero_mat(Val: wp.array(dtype=MAT)):
        i = wp.tid()
        Val[i] = MAT()

    # -- SELL block products ------------------------------------------------

    @wp.kernel
    def k_gather(smap: wp.array(dtype=wp.int32), V: wp.array(dtype=MAT), Vs: wp.array(dtype=MAT)):
        q = wp.tid()
        j = smap[q]
        if j >= 0:
            Vs[q] = V[j]
        else:
            Vs[q] = MAT()

    @wp.kernel
    def k_gather_f(smap: wp.array(dtype=wp.int32), V: wp.array(dtype=MAT), Vs: wp.array(dtype=MATF)):
        q = wp.tid()
        j = smap[q]
        m = MATF()
        if j >= 0:
            v = V[j]
            for r in range(DOF):
                for c in range(DOF):
                    m[r, c] = wp.float32(v[r, c])
        Vs[q] = m

    @wp.func
    def brow(i: wp.int32, slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32),
             K: wp.array(dtype=MAT), x: wp.array(dtype=F)) -> VEC:
        s = i // C
        q1 = slice_off[s + 1]
        acc = VEC()
        for q in range(slice_off[s] + i % C, q1, 2 * C):
            c0 = cols[q]
            c1 = cols[q + C]
            x0 = VEC()
            x1 = VEC()
            for k in range(DOF):
                x0[k] = x[DOF * c0 + k]
                x1[k] = x[DOF * c1 + k]
            acc = acc + K[q] * x0 + K[q + C] * x1
        return acc

    @wp.func
    def browf(i: wp.int32, slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32),
              K: wp.array(dtype=MATF), x: wp.array(dtype=F32)) -> VECF:
        s = i // C
        q1 = slice_off[s + 1]
        acc = VECF()
        for q in range(slice_off[s] + i % C, q1, 2 * C):
            c0 = cols[q]
            c1 = cols[q + C]
            x0 = VECF()
            x1 = VECF()
            for k in range(DOF):
                x0[k] = x[DOF * c0 + k]
                x1[k] = x[DOF * c1 + k]
            acc = acc + K[q] * x0 + K[q + C] * x1
        return acc

    @wp.kernel
    def k_spmv(slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32), K: wp.array(dtype=MAT),
               x: wp.array(dtype=F), y: wp.array(dtype=F)):
        a = wp.tid()
        acc = brow(a, slice_off, cols, K, x)
        for k in range(DOF):
            y[DOF * a + k] = acc[k]

    # -- AMG ----------------------------------------------------------------

    @wp.kernel
    def k_dinv(dslot: wp.array(dtype=wp.int32), Vs: wp.array(dtype=MATF), Dinv: wp.array(dtype=MATF)):
        # Inverse of the diagonal blocks. The dofs with a zero row and column
        # in the block (Dirichlet dofs after the FSILS scaling) are left out
        # (zero rows and columns of the result, like the pseudo-inverse);
        # identity for a singular remaining block.
        a = wp.tid()
        d = Vs[dslot[a]]
        act = VECI()
        for k in range(DOF):
            act[k] = 0
            for q in range(DOF):
                if d[k, q] != 0.0 or d[q, k] != 0.0:
                    act[k] = 1
        m = MAT2()
        for r in range(DOF):
            for c in range(DOF):
                if act[r] != 0 and act[c] != 0:
                    m[r, c] = F(d[r, c])
                elif r == c:
                    m[r, c] = F(1.0)
                else:
                    m[r, c] = F(0.0)
                if r == c:
                    m[r, DOF + c] = F(1.0)
                else:
                    m[r, DOF + c] = F(0.0)
        ok = int(1)
        for c in range(DOF):
            if ok != 0:
                p = c
                for r in range(c + 1, DOF):
                    if wp.abs(m[r, c]) > wp.abs(m[p, c]):
                        p = r
                if m[p, c] == F(0.0):
                    ok = 0
                else:
                    if p != c:
                        for k in range(2 * DOF):
                            t = m[c, k]
                            m[c, k] = m[p, k]
                            m[p, k] = t
                    inv = F(1.0) / m[c, c]
                    for k in range(2 * DOF):
                        m[c, k] = m[c, k] * inv
                    for r in range(DOF):
                        if r != c:
                            f = m[r, c]
                            if f != F(0.0):
                                for k in range(2 * DOF):
                                    m[r, k] = m[r, k] - f * m[c, k]
        o = MATF()
        for r in range(DOF):
            for c in range(DOF):
                v = wp.float32(0.0)
                if act[r] != 0 and act[c] != 0:
                    if ok != 0:
                        v = wp.float32(m[r, DOF + c])
                    elif r == c:
                        v = wp.float32(1.0)
                o[r, c] = v
        Dinv[a] = o

    @wp.kernel
    def k_galerkin(gPtr: wp.array(dtype=wp.int32), gList: wp.array(dtype=wp.int32), cslot: wp.array(dtype=wp.int32),
                   Vf: wp.array(dtype=MATF), Vc: wp.array(dtype=MATF)):
        # Coarse block entries: sums of the fine block entries
        cb = wp.tid()
        s = MATF()
        for k in range(gPtr[cb], gPtr[cb + 1]):
            s = s + Vf[gList[k]]
        Vc[cslot[cb]] = s

    @wp.kernel
    def k_pow_init(n: wp.int32, x: wp.array(dtype=F32), ns: wp.array(dtype=F), slot: wp.int32):
        i = wp.tid()
        v = F(0.0)
        if i < n:
            h = wp.uint32(i) * wp.uint32(2654435761)
            h = h ^ (h >> wp.uint32(15))
            xi = wp.float32(1.0) + wp.float32(h % wp.uint32(1000)) * wp.float32(1e-3)
            x[i] = xi
            v = F(xi) * F(xi)
        t = wp.tile(v)
        s = wp.tile_sum(t)
        wp.tile_atomic_add(ns, s, offset=slot)

    @wp.kernel
    def k_pow_norm(n: wp.int32, x: wp.array(dtype=F32), ns: wp.array(dtype=F), slot: wp.int32):
        i = wp.tid()
        v = F(0.0)
        if i < n:
            xi = x[i]
            v = F(xi) * F(xi)
        t = wp.tile(v)
        s = wp.tile_sum(t)
        wp.tile_atomic_add(ns, s, offset=slot)

    @wp.kernel
    def k_pow_step(nNo: wp.int32, slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32),
                   Vs: wp.array(dtype=MATF), Dinv: wp.array(dtype=MATF), x: wp.array(dtype=F32),
                   ns: wp.array(dtype=F), slot_in: wp.int32, y: wp.array(dtype=F32), slot_out: wp.int32):
        # y = Dinv A x / |x|; ns[slot_out] += |y|^2
        a = wp.tid()
        v = F(0.0)
        if a < nNo:
            inv = wp.float32(F(1.0) / wp.sqrt(ns[slot_in]))
            ax = browf(a, slice_off, cols, Vs, x)
            yv = Dinv[a] * ax * inv
            for k in range(DOF):
                y[DOF * a + k] = yv[k]
                v = v + F(yv[k]) * F(yv[k])
        t = wp.tile(v)
        s = wp.tile_sum(t)
        wp.tile_atomic_add(ns, s, offset=slot_out)

    @wp.kernel
    def k_set_w(omega: F, ns: wp.array(dtype=F), slot: wp.int32, wv: wp.array(dtype=F32), l: wp.int32):
        rho = wp.sqrt(ns[slot])
        if rho < F(1e-12):
            rho = F(1e-12)
        wv[l] = wp.float32(omega / rho)

    @wp.kernel
    def k_dense(slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32), smap: wp.array(dtype=wp.int32),
                Vs: wp.array(dtype=MATF), nc: wp.int32, Ad: wp.array2d(dtype=F)):
        I = wp.tid()
        for r in range(DOF):
            for c in range(nc):
                Ad[DOF * I + r, c] = F(0.0)
        s = I // C
        q1 = slice_off[s + 1]
        for q in range(slice_off[s] + I % C, q1, C):
            if smap[q] >= 0:
                J = cols[q]
                m = Vs[q]
                for r in range(DOF):
                    for c in range(DOF):
                        Ad[DOF * I + r, DOF * J + c] = F(m[r, c])

    @wp.kernel
    def k_smooth0_d(Dinv: wp.array(dtype=MATF), wv: wp.array(dtype=F32), l: wp.int32, r: wp.array(dtype=F),
                    z: wp.array(dtype=F32)):
        # z = w Dinv r (finest level: r double)
        a = wp.tid()
        rv = VECF()
        for k in range(DOF):
            rv[k] = wp.float32(r[DOF * a + k])
        zv = Dinv[a] * rv * wv[l]
        for k in range(DOF):
            z[DOF * a + k] = zv[k]

    @wp.kernel
    def k_smooth0_f(Dinv: wp.array(dtype=MATF), wv: wp.array(dtype=F32), l: wp.int32, r: wp.array(dtype=F32),
                    z: wp.array(dtype=F32)):
        a = wp.tid()
        rv = VECF()
        for k in range(DOF):
            rv[k] = r[DOF * a + k]
        zv = Dinv[a] * rv * wv[l]
        for k in range(DOF):
            z[DOF * a + k] = zv[k]

    @wp.kernel
    def k_res_d(slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32), Vs: wp.array(dtype=MATF),
                r: wp.array(dtype=F), z: wp.array(dtype=F32), t: wp.array(dtype=F32)):
        # t = r - A z
        a = wp.tid()
        y = browf(a, slice_off, cols, Vs, z)
        for k in range(DOF):
            t[DOF * a + k] = wp.float32(r[DOF * a + k]) - y[k]

    @wp.kernel
    def k_res_f(slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32), Vs: wp.array(dtype=MATF),
                r: wp.array(dtype=F32), z: wp.array(dtype=F32), t: wp.array(dtype=F32)):
        a = wp.tid()
        y = browf(a, slice_off, cols, Vs, z)
        for k in range(DOF):
            t[DOF * a + k] = r[DOF * a + k] - y[k]

    @wp.kernel
    def k_restrict(aPtr: wp.array(dtype=wp.int32), aList: wp.array(dtype=wp.int32), t: wp.array(dtype=F32),
                   rc: wp.array(dtype=F32)):
        # rc(I) = sum of t over the nodes of aggregate I
        q = wp.tid()
        I = q // DOF
        k = q % DOF
        s = wp.float32(0.0)
        for p in range(aPtr[I], aPtr[I + 1]):
            s = s + t[DOF * aList[p] + k]
        rc[q] = s

    @wp.kernel
    def k_coarse(nc: wp.int32, Ainv: wp.array2d(dtype=F32), r: wp.array(dtype=F32), z: wp.array(dtype=F32)):
        i = wp.tid()
        s = wp.float32(0.0)
        for k in range(nc):
            s = s + Ainv[i, k] * r[k]
        z[i] = s

    @wp.kernel
    def k_prolong(agg: wp.array(dtype=wp.int32), z: wp.array(dtype=F32), ec: wp.array(dtype=F32),
                  z2: wp.array(dtype=F32)):
        # z2 = z + P ec
        q = wp.tid()
        z2[q] = z[q] + ec[DOF * agg[q // DOF] + q % DOF]

    @wp.kernel
    def k_post_f(slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32), Vs: wp.array(dtype=MATF),
                 Dinv: wp.array(dtype=MATF), wv: wp.array(dtype=F32), l: wp.int32, r: wp.array(dtype=F32),
                 z2: wp.array(dtype=F32), z: wp.array(dtype=F32)):
        # z = z2 + w Dinv (r - A z2)
        a = wp.tid()
        y = browf(a, slice_off, cols, Vs, z2)
        rv = VECF()
        for k in range(DOF):
            rv[k] = r[DOF * a + k] - y[k]
        zv = Dinv[a] * rv * wv[l]
        for k in range(DOF):
            z[DOF * a + k] = z2[DOF * a + k] + zv[k]

    @wp.kernel
    def k_post_0(slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32), Vs: wp.array(dtype=MATF),
                 Dinv: wp.array(dtype=MATF), wv: wp.array(dtype=F32), r: wp.array(dtype=F), z2: wp.array(dtype=F32),
                 z: wp.array(dtype=F)):
        # Finest level: double output; no component in the dofs without
        # equation (zero row of the block, e.g. Dirichlet dofs)
        a = wp.tid()
        y = browf(a, slice_off, cols, Vs, z2)
        rv = VECF()
        for k in range(DOF):
            rv[k] = wp.float32(r[DOF * a + k]) - y[k]
        d = Dinv[a]
        zv = d * rv * wv[0]
        for i in range(DOF):
            act = int(0)
            for k in range(DOF):
                if d[i, k] != 0.0:
                    act = 1
            if act != 0:
                z[DOF * a + i] = F(z2[DOF * a + i] + zv[i])
            else:
                z[DOF * a + i] = F(0.0)

    # -- GMRES --------------------------------------------------------------

    @wp.kernel
    def k_gm_init(n: wp.int32, R: wp.array(dtype=F), w: wp.array(dtype=F), gs: wp.array(dtype=F)):
        # w = R; gs[NRM + 1] += |w|^2
        k = wp.tid()
        wk = F(0.0)
        if k < n:
            wk = R[k]
            w[k] = wk
        t = wp.tile(wk * wk)
        s = wp.tile_sum(t)
        wp.tile_atomic_add(gs, s, offset=NRM + 1)

    @wp.kernel
    def k_gm_restart(nNo: wp.int32, slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32),
                     K: wp.array(dtype=MAT), X: wp.array(dtype=F), R: wp.array(dtype=F), w: wp.array(dtype=F),
                     gs: wp.array(dtype=F)):
        # w = R - K X; gs[NRM + 1] += |w|^2
        a = wp.tid()
        v = F(0.0)
        if a < nNo:
            acc = brow(a, slice_off, cols, K, X)
            for k in range(DOF):
                wk = R[DOF * a + k] - acc[k]
                w[DOF * a + k] = wk
                v = v + wk * wk
        t = wp.tile(v)
        s = wp.tile_sum(t)
        wp.tile_atomic_add(gs, s, offset=NRM + 1)

    @wp.kernel
    def k_gm_scale(n: wp.int32, win: wp.array(dtype=F), gs: wp.array(dtype=F), bp: wp.int32, i: wp.int32,
                   u: wp.array2d(dtype=F)):
        # u_i = win / |win|
        k = wp.tid()
        inv = F(1.0) / wp.sqrt(gs[NRM + bp])
        u[i, k] = win[k] * inv

    @wp.kernel
    def k_gm_spmv(nNo: wp.int32, slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32),
                  K: wp.array(dtype=MAT), x: wp.array(dtype=F), u: wp.array2d(dtype=F), wout: wp.array(dtype=F),
                  gs: wp.array(dtype=F), g0: wp.int32, b: wp.int32):
        # wout = K x; gs[g0] += u_0 . wout; gs[NRM + b] = 0
        a = wp.tid()
        v = F(0.0)
        if a < nNo:
            acc = brow(a, slice_off, cols, K, x)
            for k in range(DOF):
                wout[DOF * a + k] = acc[k]
                v = v + u[0, DOF * a + k] * acc[k]
        t = wp.tile(v)
        s = wp.tile_sum(t)
        wp.tile_atomic_add(gs, s, offset=g0)
        if a == 0:
            gs[NRM + b] = F(0.0)

    @wp.kernel
    def k_gm_spmv_noamg(nNo: wp.int32, slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32),
                        K: wp.array(dtype=MAT), win: wp.array(dtype=F), i: wp.int32, u: wp.array2d(dtype=F),
                        wout: wp.array(dtype=F), gs: wp.array(dtype=F), g0: wp.int32, b: wp.int32):
        # u_i = win / |win|; wout = K u_i; gs[g0] += u_0 . wout; gs[NRM + b] = 0
        a = wp.tid()
        v = F(0.0)
        if a < nNo:
            inv = F(1.0) / wp.sqrt(gs[NRM + 1 - b])
            acc = brow(a, slice_off, cols, K, win) * inv
            for k in range(DOF):
                wout[DOF * a + k] = acc[k]
                u[i, DOF * a + k] = win[DOF * a + k] * inv
                v = v + u[0, DOF * a + k] * acc[k]
        t = wp.tile(v)
        s = wp.tile_sum(t)
        wp.tile_atomic_add(gs, s, offset=g0)
        if a == 0:
            gs[NRM + b] = F(0.0)

    @wp.kernel
    def k_gm_mgs(n: wp.int32, stride: wp.int32, j: wp.int32, last: wp.int32, g0: wp.int32, g1: wp.int32,
                 sD: wp.int32, b: wp.int32, u: wp.array2d(dtype=F), w: wp.array(dtype=F), gs: wp.array(dtype=F)):
        # w -= h_j u_j (h_j = gs[g0 + j]); then gs[g0 + j + 1] += u_{j+1} . w
        # (j < i) or gs[NRM + b] += |w|^2 (last); the last pass clears
        # gs[g1 + k] for the next step. Grid-stride loop (several elements
        # per thread amortize the block reduction).
        k = wp.tid()
        v = F(0.0)
        h = gs[g0 + j]
        q = k
        while q < n:
            wk = w[q] - h * u[j, q]
            w[q] = wk
            if last != 0:
                v = v + wk * wk
            else:
                v = v + u[j + 1, q] * wk
            q += stride
        t = wp.tile(v)
        s = wp.tile_sum(t)
        if last != 0:
            wp.tile_atomic_add(gs, s, offset=NRM + b)
        else:
            wp.tile_atomic_add(gs, s, offset=g0 + j + 1)
        if last != 0 and k <= sD:
            gs[g1 + k] = F(0.0)

    @wp.kernel
    def k_lin_comb_to(n: wp.int32, coef: wp.array(dtype=F), u: wp.array2d(dtype=F), T: wp.array(dtype=F)):
        # T = sum_{j<m} coef[1+j] u[j] with m = coef[0]
        k = wp.tid()
        m = wp.int32(coef[0])
        s = F(0.0)
        for j in range(m):
            s = s + coef[1 + j] * u[j, k]
        T[k] = s

    @wp.kernel
    def k_add(Z: wp.array(dtype=F), X: wp.array(dtype=F)):
        k = wp.tid()
        X[k] = X[k] + Z[k]

    # -- CG -----------------------------------------------------------------

    @wp.kernel
    def k_cg_init(n: wp.int32, R: wp.array(dtype=F), r: wp.array(dtype=F), X: wp.array(dtype=F),
                  p: wp.array(dtype=F), gs: wp.array(dtype=F)):
        # r = R, X = p = 0, gs[RR] += |r|^2, gs[RZ + 1] = 1
        k = wp.tid()
        rk = F(0.0)
        if k < n:
            rk = R[k]
            r[k] = rk
            X[k] = F(0.0)
            p[k] = F(0.0)
        t = wp.tile(rk * rk)
        s = wp.tile_sum(t)
        wp.tile_atomic_add(gs, s, offset=RR)
        if k == 0:
            gs[RZ + 1] = F(1.0)

    @wp.kernel
    def k_cg_rz(n: wp.int32, r: wp.array(dtype=F), z: wp.array(dtype=F), gs: wp.array(dtype=F), slot: wp.int32):
        k = wp.tid()
        v = F(0.0)
        if k < n:
            v = r[k] * z[k]
        t = wp.tile(v)
        s = wp.tile_sum(t)
        wp.tile_atomic_add(gs, s, offset=slot)

    @wp.kernel
    def k_cg_dir(gs: wp.array(dtype=F), b: wp.int32, z: wp.array(dtype=F), p: wp.array(dtype=F)):
        # p = z + (rz_new / rz_old) p; gs[PQ] = 0
        k = wp.tid()
        beta = gs[RZ + b] / gs[RZ + 1 - b]
        p[k] = z[k] + beta * p[k]
        if k == 0:
            gs[PQ] = F(0.0)

    @wp.kernel
    def k_cg_q(nNo: wp.int32, slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32),
               K: wp.array(dtype=MAT), p: wp.array(dtype=F), q: wp.array(dtype=F), gs: wp.array(dtype=F)):
        # q = K p; gs[PQ] += p . q; gs[RR] = 0
        a = wp.tid()
        v = F(0.0)
        if a < nNo:
            acc = brow(a, slice_off, cols, K, p)
            for k in range(DOF):
                q[DOF * a + k] = acc[k]
                v = v + p[DOF * a + k] * acc[k]
        t = wp.tile(v)
        s = wp.tile_sum(t)
        wp.tile_atomic_add(gs, s, offset=PQ)
        if a == 0:
            gs[RR] = F(0.0)

    @wp.kernel
    def k_cg_update(n: wp.int32, gs: wp.array(dtype=F), b: wp.int32, p: wp.array(dtype=F), q: wp.array(dtype=F),
                    X: wp.array(dtype=F), r: wp.array(dtype=F)):
        # alpha = rz / pq; X += alpha p; r -= alpha q; gs[RR] += |r|^2; gs[RZ_b'] = 0
        k = wp.tid()
        alpha = gs[RZ + b] / gs[PQ]
        rk = F(0.0)
        if k < n:
            X[k] = X[k] + alpha * p[k]
            rk = r[k] - alpha * q[k]
            r[k] = rk
        t = wp.tile(rk * rk)
        s = wp.tile_sum(t)
        wp.tile_atomic_add(gs, s, offset=RR)
        if k == 0:
            gs[RZ + 1 - b] = F(0.0)

    ns = dict(locals())
    ns["DOF"], ns["DD"] = DOF, DD
    return ns


class Level:
    pass


class Bamg:
    """Aggregation AMG for the block matrix with the node pattern (offsets,
    columns) whose SELL layout is sell0 (shared with the caller)."""

    def __init__(self, K, sell0, n, offsets, columns):
        t0 = time.perf_counter()
        self.K = K
        DOF = K["DOF"]
        self.DOF = DOF
        self.omega = float(os.environ.get("SVWARP_AMG_W", 0.9))
        self.cmax = int(os.environ.get("SVWARP_AMG_CMAX", 128))
        self.levels = []
        self.nSolve = 0
        offsets = np.asarray(offsets, dtype=np.int64)
        columns = np.asarray(columns, dtype=np.int64)
        i32 = lambda a: wp.array(np.asarray(a).astype(np.int32), dtype=wp.int32, device=DEVICE)
        zf = lambda m: wp.zeros(m, dtype=F32, device=DEVICE)
        while True:
            L = Level()
            L.n = n
            L.nnz = len(columns)
            lengths = np.diff(offsets)
            rows = np.repeat(np.arange(n, dtype=np.int64), lengths)
            kk = np.arange(L.nnz, dtype=np.int64) - np.repeat(offsets[:-1], lengths)
            if self.levels:
                L.sell = sell.Sell(offsets, columns)
                # Coarse SELL slots of the Galerkin sums of the previous level
                self.levels[-1].cslot = i32(L.sell.slot(rows, kk))
            else:
                L.sell = sell0
            L.Vs = wp.zeros(L.sell.total, dtype=K["MATF"], device=DEVICE)
            diag = np.nonzero(columns == rows)[0]
            if len(diag) != n:
                raise RuntimeError("svwarp: AMG: a row without diagonal entry")
            L.dslot = i32(L.sell.slot(rows[diag], kk[diag]))
            L.Dinv = wp.zeros(n, dtype=K["MATF"], device=DEVICE)
            L.r, L.z, L.t, L.z2 = zf(DOF * n), zf(DOF * n), zf(DOF * n), zf(DOF * n)
            L.ev = [zf(DOF * n), zf(DOF * n)]
            self.levels.append(L)
            if n <= self.cmax:
                break
            agg, na = aggregate(n, offsets, columns)
            if na >= n:
                break
            # Coarse pattern; coarse entry of every fine entry; the fine SELL
            # slots of every coarse entry (Galerkin sums); the nodes of every
            # aggregate (restriction)
            key = agg[rows] * na + agg[columns]
            order = np.argsort(key, kind="stable")
            ks = key[order]
            first = np.concatenate(([True], ks[1:] != ks[:-1]))
            ukeys = ks[first]
            cmap = np.empty(L.nnz, dtype=np.int64)
            cmap[order] = np.cumsum(first) - 1
            cI = ukeys // na
            cJ = ukeys % na
            c_offsets = np.concatenate(([0], np.cumsum(np.bincount(cI, minlength=na))))
            gorder = np.argsort(cmap, kind="stable")
            gPtr = np.concatenate(([0], np.cumsum(np.bincount(cmap, minlength=len(ukeys)))))
            gList = L.sell.slot(rows, kk)[gorder]
            aorder = np.argsort(agg, kind="stable")
            aPtr = np.concatenate(([0], np.cumsum(np.bincount(agg, minlength=na))))
            L.agg = i32(agg)
            L.aPtr, L.aList = i32(aPtr), i32(aorder)
            L.gPtr, L.gList = i32(gPtr), i32(gList)
            n = na
            offsets = c_offsets
            columns = cJ
        nl = len(self.levels)
        self.nc = DOF * self.levels[-1].n
        self.Ad = wp.zeros((self.nc, self.nc), dtype=F, device=DEVICE)
        self.Ainv = wp.zeros((self.nc, self.nc), dtype=F32, device=DEVICE)
        self.wv = zf(nl)
        self.ns = wp.zeros(16 * nl, dtype=F, device=DEVICE)
        if os.environ.get("SV_GPU_VERBOSE"):
            print(f"svwarp: AMG dof {DOF} levels {self.sizes()} omega {self.omega} in {time.perf_counter() - t0:.2f} s",
                  flush=True)

    def sizes(self):
        return [L.n for L in self.levels]

    def numeric(self, Val):
        """Coarse matrices, smoothers and the coarsest inverse for the
        current matrix Val (CSR blocks); synchronizes (the pseudo-inverse of
        the coarsest matrix is computed on the host)."""
        K = self.K
        DOF = self.DOF
        nPow = 15 if self.nSolve == 0 else 4
        first = self.nSolve == 0
        self.nSolve += 1
        lv = self.levels
        nl = len(lv)
        L0 = lv[0]
        L0.sell.gather(Val, L0.Vs, K["k_gather_f"])
        for l, L in enumerate(lv):
            wp.launch(K["k_dinv"], dim=L.n, inputs=[L.dslot, L.Vs, L.Dinv], device=DEVICE)
            if l + 1 == nl:
                break
            C_ = lv[l + 1]
            wp.launch(K["k_galerkin"], dim=C_.nnz, inputs=[L.gPtr, L.gList, L.cslot, L.Vs, C_.Vs], device=DEVICE)
            # Damping with the spectral radius of Dinv A (power iteration)
            s0 = 16 * l
            nv = DOF * L.n
            wp.launch(la.k_zero, dim=16, inputs=[self.ns, s0], device=DEVICE)
            x, y = L.ev
            if first:
                wp.launch(K["k_pow_init"], dim=pad(nv), inputs=[nv, x, self.ns, s0], block_dim=BD, device=DEVICE)
            else:
                wp.launch(K["k_pow_norm"], dim=pad(nv), inputs=[nv, x, self.ns, s0], block_dim=BD, device=DEVICE)
            for k in range(nPow):
                wp.launch(K["k_pow_step"], dim=pad(L.n),
                          inputs=[L.n, L.sell.slice_off, L.sell.cols, L.Vs, L.Dinv, x, self.ns, s0 + k, y, s0 + k + 1],
                          block_dim=BD, device=DEVICE)
                x, y = y, x
            wp.launch(K["k_set_w"], dim=1, inputs=[F(self.omega), self.ns, s0 + nPow, self.wv, l], device=DEVICE)
            if x is not L.ev[0]:
                wp.copy(L.ev[0], x)
        # Dense coarsest matrix and its pseudo-inverse on the host (the matrix
        # is not symmetric and has zero rows and columns)
        Lc = lv[-1]
        wp.launch(K["k_dense"], dim=Lc.n, inputs=[Lc.sell.slice_off, Lc.sell.cols, Lc.sell.map, Lc.Vs, self.nc, self.Ad],
                  device=DEVICE)
        A = self.Ad.numpy()
        Ai = np.linalg.pinv(A).astype(np.float32)
        self.Ainv.assign(Ai)
        if os.environ.get("SV_GPU_VERBOSE") and self.nSolve <= 2:
            print(f"svwarp: AMG dof {DOF} damping {self.wv.numpy()[:nl - 1]}", flush=True)

    def emit_vcycle(self, r, z):
        """Kernel launches of z = M r (double in and out, finest level)."""
        K = self.K
        DOF = self.DOF
        lv = self.levels
        nl = len(lv)
        for l in range(nl - 1):
            L, C_ = lv[l], lv[l + 1]
            A = [L.sell.slice_off, L.sell.cols, L.Vs]
            if l == 0:
                wp.launch(K["k_smooth0_d"], dim=L.n, inputs=[L.Dinv, self.wv, 0, r, L.z], device=DEVICE)
                wp.launch(K["k_res_d"], dim=L.n, inputs=A + [r, L.z, L.t], device=DEVICE)
            else:
                wp.launch(K["k_smooth0_f"], dim=L.n, inputs=[L.Dinv, self.wv, l, L.r, L.z], device=DEVICE)
                wp.launch(K["k_res_f"], dim=L.n, inputs=A + [L.r, L.z, L.t], device=DEVICE)
            wp.launch(K["k_restrict"], dim=DOF * C_.n, inputs=[L.aPtr, L.aList, L.t, C_.r], device=DEVICE)
        Lc = lv[-1]
        wp.launch(K["k_coarse"], dim=self.nc, inputs=[self.nc, self.Ainv, Lc.r, Lc.z], device=DEVICE)
        for l in range(nl - 2, -1, -1):
            L, C_ = lv[l], lv[l + 1]
            A = [L.sell.slice_off, L.sell.cols, L.Vs]
            wp.launch(K["k_prolong"], dim=DOF * L.n, inputs=[L.agg, L.z, C_.z, L.z2], device=DEVICE)
            if l > 0:
                wp.launch(K["k_post_f"], dim=L.n, inputs=A + [L.Dinv, self.wv, l, L.r, L.z2, L.z], device=DEVICE)
            else:
                wp.launch(K["k_post_0"], dim=L.n, inputs=A + [L.Dinv, self.wv, r, L.z2, z], device=DEVICE)


class LsSolver:
    """GMRES and CG for one block size on the device matrix Val (CSR blocks
    of the FSILS lhs, dof = 3 or 4)."""

    def __init__(self, DOF, lhs, sell0, Val):
        self.K = K = kernels(DOF)
        self.DOF = DOF
        self.lhs = lhs
        self.Val = Val
        self.sell = sell0
        self.nNo = lhs.nNo
        self.n = n = DOF * lhs.nNo
        z = lambda: wp.zeros(n, dtype=F, device=DEVICE)
        self.Ks = wp.zeros(sell0.total, dtype=K["MAT"], device=DEVICE)
        self.R, self.X, self.P, self.Q, self.Z, self.T = z(), z(), z(), z(), z(), z()
        self.wb = [z(), z()]
        self.use_amg = amg_enabled()
        self.amg = Bamg(K, sell0, lhs.nNo, lhs.offsets_np, lhs.columns_np) if self.use_amg else None
        self.sD = 0
        self.graphs = {}
        self.cg_chunk = int(os.environ.get("SVWARP_CG_CHUNK", max(2, min(8, 400000 // n // 2 * 2))))
        self.t_setup = self.t_solve = 0.0
        self.n_call = self.n_it = 0
        if os.environ.get("SV_GPU_VERBOSE"):
            print(f"svwarp: FSI solver dof {DOF} nNo {lhs.nNo} nnz {lhs.nnz} SELL padding {sell0.padding:.2f} "
                  f"AMG {int(self.use_amg)} CG chunk {self.cg_chunk}", flush=True)

    def run(self, key, emit):
        if not USE_GRAPHS:
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
        wp.launch(kernel, dim=pad(n), inputs=inputs, block_dim=BD, device=DEVICE)

    def alloc_work(self, sD):
        if self.sD >= sD:
            return
        self.sD = sD
        self.ngs = pad8(GH + 2 * (sD + 1))
        self.gs = wp.zeros(self.ngs, dtype=F, device=DEVICE)
        self.hgs = Mapped(self.ngs)
        self.hcoef = Mapped(sD + 2)
        self.dcoef = wp.zeros(self.hcoef.n, dtype=F, device=DEVICE)
        self.gu = wp.zeros((sD + 1, self.n), dtype=F, device=DEVICE)
        self.graphs = {}

    def setup(self):
        """Matrix of this solve: the SELL copy and the AMG hierarchy."""
        t0 = time.perf_counter()
        self.sell.gather(self.Val, self.Ks, self.K["k_gather"])
        if self.use_amg:
            self.amg.numeric(self.Val)
        sync()
        self.t_setup += time.perf_counter() - t0

    def publish(self, lo, hi):
        v0 = lo // 8
        wp.launch(la.k_publish8, dim=(hi - 1) // 8 - v0 + 1, inputs=[self.gs, self.hgs.dev8, v0], device=DEVICE)

    # -- GMRES --------------------------------------------------------------

    def emit_gm_init(self):
        n, gs = self.n, self.gs
        wp.launch(la.k_zero, dim=self.ngs, inputs=[gs, 0], device=DEVICE)
        wp.launch(la.k_zero, dim=n, inputs=[self.X, 0], device=DEVICE)
        self.tlaunch(self.K["k_gm_init"], n, [n, self.R, self.wb[1], gs])
        self.publish(0, 8)

    def emit_gm_restart(self):
        nNo, gs = self.nNo, self.gs
        wp.launch(la.k_zero, dim=self.ngs, inputs=[gs, 0], device=DEVICE)
        self.tlaunch(self.K["k_gm_restart"], nNo, [nNo, self.sell.slice_off, self.sell.cols, self.Ks, self.X, self.R,
                                                   self.wb[1], gs])
        self.publish(0, 8)

    def emit_gm_step(self, i):
        K, n, nNo, gs, u, sD = self.K, self.n, self.nNo, self.gs, self.gu, self.sD
        b = i & 1
        bp = 1 - b
        win, wout = self.wb[1 - b], self.wb[b]
        g0 = GH + b * (sD + 1)
        g1 = GH + bp * (sD + 1)
        A = [self.sell.slice_off, self.sell.cols, self.Ks]
        if self.use_amg:
            wp.launch(K["k_gm_scale"], dim=n, inputs=[n, win, gs, bp, i, u], device=DEVICE)
            self.amg.emit_vcycle(u[i], self.Z)
            self.tlaunch(K["k_gm_spmv"], nNo, [nNo] + A + [self.Z, u, wout, gs, g0, b])
        else:
            self.tlaunch(K["k_gm_spmv_noamg"], nNo, [nNo] + A + [win, i, u, wout, gs, g0, b])
        stride = min(pad(n), RED_BLOCKS * BD)
        for j in range(i + 1):
            wp.launch(K["k_gm_mgs"], dim=stride, inputs=[n, stride, j, int(j == i), g0, g1, sD, b, u, wout, gs],
                      block_dim=BD, device=DEVICE)
        self.publish(0, 8)
        self.publish(g0, g0 + i + 1)

    def emit_gm_fin(self):
        K, n = self.K, self.n
        wp.launch(la.k_fetch8, dim=self.hcoef.nvec, inputs=[self.hcoef.dev8, self.dcoef], device=DEVICE)
        if self.use_amg:
            wp.launch(K["k_lin_comb_to"], dim=n, inputs=[n, self.dcoef, self.gu, self.T], device=DEVICE)
            self.amg.emit_vcycle(self.T, self.Z)
            wp.launch(K["k_add"], dim=n, inputs=[self.Z, self.X], device=DEVICE)
        else:
            wp.launch(la.k_lin_comb, dim=n, inputs=[self.dcoef, 0, self.gu, self.X], device=DEVICE)

    def gmres(self, mItr, sD, absTol, relTol):
        """gmres::gmres_v without coupled faces, with the AMG right
        preconditioner. R in, X out (device). Returns the statistics."""
        self.alloc_work(sD)
        sD = self.sD
        hgs = self.hgs.np
        h = np.zeros((sD + 1, sD))
        y = np.zeros(sD)
        c = np.zeros(sD)
        s = np.zeros(sD)
        err = np.zeros(sD + 1)
        eps = iNorm = fNorm = dB = 0.0
        itr = last_i = 0
        success = False
        for l in range(mItr):
            if l == 0:
                self.run("gm_init", self.emit_gm_init)
            else:
                self.run("gm_restart", self.emit_gm_restart)
                itr += 1
            sync()
            err[0] = np.sqrt(hgs[NRM + 1])
            if l == 0:
                iNorm = fNorm = err[0]
                eps = max(absTol, relTol * err[0])
                if iNorm <= absTol:
                    success = True
                    break
            if err[0] == 0.0:
                raise RuntimeError("svwarp: a zero matrix norm was computed in GMRES")
            dB = fNorm
            for i in range(sD):
                last_i = i
                self.run(("gm", i), lambda: self.emit_gm_step(i))
                itr += 1
                sync()
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
                if abs(err[i + 1]) < eps:
                    success = True
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
            fNorm = abs(err[last_i + 1])
            if success:
                break
        sync()
        self.n_it += itr
        dB = 10.0 * np.log(fNorm / dB) if dB > 0.0 and fNorm > 0.0 else 0.0
        return itr, int(success), iNorm, fNorm, dB

    # -- CG -----------------------------------------------------------------

    def emit_cg_init(self):
        K, n, gs = self.K, self.n, self.gs
        r = self.Q
        wp.launch(la.k_zero, dim=8, inputs=[gs, 0], device=DEVICE)
        self.tlaunch(K["k_cg_init"], n, [n, self.R, r, self.X, self.P, gs])
        if self.use_amg:
            self.amg.emit_vcycle(r, self.Z)
        else:
            wp.launch(la.k_copy, dim=n, inputs=[r, self.Z], device=DEVICE)
        self.tlaunch(K["k_cg_rz"], n, [n, r, self.Z, gs, RZ])
        self.publish(0, 8)

    def emit_cg_chunk(self, parity):
        K, n, nNo, gs = self.K, self.n, self.nNo, self.gs
        r, z, p, q = self.Q, self.Z, self.P, self.T
        A = [self.sell.slice_off, self.sell.cols, self.Ks]
        for it in range(self.cg_chunk):
            b = (parity + it) & 1
            wp.launch(K["k_cg_dir"], dim=n, inputs=[gs, b, z, p], device=DEVICE)
            self.tlaunch(K["k_cg_q"], nNo, [nNo] + A + [p, q, gs])
            self.tlaunch(K["k_cg_update"], n, [n, gs, b, p, q, self.X, r])
            if self.use_amg:
                self.amg.emit_vcycle(r, z)
            else:
                wp.launch(la.k_copy, dim=n, inputs=[r, z], device=DEVICE)
            self.tlaunch(K["k_cg_rz"], n, [n, r, z, gs, RZ + 1 - b])
        self.publish(0, 8)

    def cg(self, mItr, absTol, relTol):
        """cgrad::cgrad_v with the AMG preconditioner. R in, X out (device)."""
        self.alloc_work(1)
        hgs = self.hgs.np
        self.run("cg_init", self.emit_cg_init)
        sync()
        err = errO = float(hgs[RR])
        iNorm = np.sqrt(err)
        eps = max(absTol, relTol * iNorm) ** 2
        nit = parity = 0
        success = False
        while nit < mItr:
            if err < eps:
                success = True
                break
            errO = err
            self.run(("cg_chunk", parity), lambda: self.emit_cg_chunk(parity))
            nit += self.cg_chunk
            parity = (parity + self.cg_chunk) & 1
            sync()
            err = float(hgs[RR])
        if err < eps:
            success = True
        self.n_it += nit
        dB = 0.0 if errO < np.finfo(float).eps else 5.0 * np.log(err / errO)
        return nit, int(success), iNorm, np.sqrt(err), dB

    def solve(self, type_, mItr, sD, absTol, relTol, R_host):
        """R_host (host, dof x nNo): right-hand side in, solution out. The
        device matrix is cleared afterwards. Returns (itr, success, iNorm,
        fNorm, dB, time)."""
        t0 = time.perf_counter()
        wp.copy(self.R, R_host)
        self.setup()
        if type_ == 0:
            st = self.gmres(mItr, sD, absTol, relTol)
        else:
            st = self.cg(mItr, absTol, relTol)
        wp.copy(R_host, self.X)
        wp.launch(self.K["k_zero_mat"], dim=self.lhs.nnz, inputs=[self.Val], device=DEVICE)
        sync()
        dt = time.perf_counter() - t0
        self.t_solve += dt
        self.n_call += 1
        if os.environ.get("SVWARP_TIMING") and self.n_call % 10 == 0:
            n = self.n_call
            print(f"SVWARP ls dof {self.DOF} calls={n} per call: all={self.t_solve / n:.4f} setup={self.t_setup / n:.4f} "
                  f"({1e3 * self.t_solve / max(1, self.n_it):.3f} ms/it)", flush=True)
        return st + (dt,)
