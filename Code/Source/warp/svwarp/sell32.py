# SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
# SPDX-License-Identifier: BSD-3-Clause

"""Single precision kernels of the AMG hierarchy on SELL matrices (see
sell.py). The hierarchy is stored in float: the FP64 rate of GeForce GPUs is
1/64 of the FP32 rate and the preconditioner needs no double precision. The
kernels of the V-cycle fuse the smoothing, the residual, the restriction
(atomic additions into the aggregates) and the prolongation, so that one
level costs two or three kernels."""

import warp as wp

from .common import F, F32
from .sell import C, U

wp.set_module_options({"enable_backward": False, "fuse_fp": False})

NPART_C = 8   # threads per row of the dense coarsest solve


@wp.func
def row_f(i: wp.int32, slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32), V: wp.array(dtype=F32),
          x: wp.array(dtype=F32)):
    s = i // C
    q1 = slice_off[s + 1]
    acc = F32(0.0)
    for q in range(slice_off[s] + i % C, q1, U * C):
        c0 = cols[q]
        c1 = cols[q + C]
        c2 = cols[q + 2 * C]
        c3 = cols[q + 3 * C]
        acc = acc + V[q] * x[c0] + V[q + C] * x[c1] + V[q + 2 * C] * x[c2] + V[q + 3 * C] * x[c3]
    return acc


@wp.kernel
def k_copy_f(src: wp.array(dtype=F32), dst: wp.array(dtype=F32)):
    i = wp.tid()
    dst[i] = src[i]


@wp.kernel
def k_zero_f(x: wp.array(dtype=F32)):
    i = wp.tid()
    x[i] = 0.0


@wp.kernel
def k_fill_f(x: wp.array(dtype=F32), v: F32):
    i = wp.tid()
    x[i] = v


@wp.kernel
def k_gather_sf(smap: wp.array(dtype=wp.int32), V: wp.array(dtype=F), Vs: wp.array(dtype=F32)):
    q = wp.tid()
    j = smap[q]
    if j >= 0:
        Vs[q] = F32(V[j])
    else:
        Vs[q] = 0.0


@wp.kernel
def k_dinv_f(diag: wp.array(dtype=wp.int32), V: wp.array(dtype=F), Dinv: wp.array(dtype=F32)):
    i = wp.tid()
    d = V[diag[i]]
    if d != F(0.0):
        Dinv[i] = F32(F(1.0) / d)
    else:
        Dinv[i] = 0.0


# Galerkin coarse matrix A_c = P^T A P with P_{i,agg(i)} = pb_i: one thread
# per fine entry, atomic additions into the coarse entry cmap[q]

@wp.kernel
def k_galerkin(cmap: wp.array(dtype=wp.int32), frow: wp.array(dtype=wp.int32), fcol: wp.array(dtype=wp.int32),
               pb: wp.array(dtype=F32), Vf: wp.array(dtype=F), Vc: wp.array(dtype=F)):
    q = wp.tid()
    wp.atomic_add(Vc, cmap[q], F(pb[frow[q]] * pb[fcol[q]]) * Vf[q])


@wp.kernel
def k_dense(slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32), V: wp.array(dtype=F32), nc: wp.int32,
            Ad: wp.array2d(dtype=F)):
    i = wp.tid()
    for c in range(nc):
        Ad[i, c] = F(0.0)
    s = i // C
    q1 = slice_off[s + 1]
    for q in range(slice_off[s] + i % C, q1, C):
        Ad[i, cols[q]] = Ad[i, cols[q]] + F(V[q])


# Inverse of the dense coarsest matrix by Gauss-Jordan steps without
# pivoting (the matrix is symmetric positive definite), in double precision,
# ping-pong between two buffers; nc kernel launches inside the setup graph

@wp.kernel
def k_gj_step(nc: wp.int32, k: wp.int32, A: wp.array2d(dtype=F), B: wp.array2d(dtype=F)):
    t = wp.tid()
    i = t // nc
    j = t - i * nc
    p = A[k, k]
    if i == k:
        if j == k:
            B[i, j] = F(1.0) / p
        else:
            B[i, j] = A[k, j] / p
    elif j == k:
        B[i, j] = -A[i, k] / p
    else:
        B[i, j] = A[i, j] - A[i, k] * A[k, j] / p


@wp.kernel
def k_cast_f(nc: wp.int32, A: wp.array2d(dtype=F), Af: wp.array2d(dtype=F32)):
    t = wp.tid()
    i = t // nc
    j = t - i * nc
    Af[i, j] = F32(A[i, j])


# Power iteration for the spectral radius of D^-1 A (damping of the Jacobi
# smoother): x_{k+1} = D^-1 A x_k / |x_k|, with |x_k|^2 accumulated in
# ns[slot] by the kernel that produces x_k.

@wp.kernel
def k_init_vec_f(n: wp.int32, x: wp.array(dtype=F32), ns: wp.array(dtype=F), slot: wp.int32):
    i = wp.tid()
    v = F32(0.0)
    if i < n:
        h = wp.uint32(i) * wp.uint32(2654435761)
        h = h ^ (h >> wp.uint32(15))
        v = 1.0 + F32(h % wp.uint32(1000)) * 1e-3
        x[i] = v
    t = wp.tile(F(v * v))
    s = wp.tile_sum(t)
    wp.tile_atomic_add(ns, s, offset=slot)


@wp.kernel
def k_norm_f(n: wp.int32, x: wp.array(dtype=F32), ns: wp.array(dtype=F), slot: wp.int32):
    i = wp.tid()
    v = F32(0.0)
    if i < n:
        v = x[i]
    t = wp.tile(F(v * v))
    s = wp.tile_sum(t)
    wp.tile_atomic_add(ns, s, offset=slot)


@wp.kernel
def k_dax_f(n: wp.int32, slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32), V: wp.array(dtype=F32),
            Dinv: wp.array(dtype=F32), x: wp.array(dtype=F32), ns: wp.array(dtype=F), slot_in: wp.int32,
            y: wp.array(dtype=F32), slot_out: wp.int32):
    # y = D^-1 A x / sqrt(ns[slot_in]); ns[slot_out] += |y|^2
    i = wp.tid()
    yi = F32(0.0)
    if i < n:
        inv = F32(F(1.0) / wp.sqrt(ns[slot_in]))
        yi = Dinv[i] * row_f(i, slice_off, cols, V, x) * inv
        y[i] = yi
    t = wp.tile(F(yi * yi))
    s2 = wp.tile_sum(t)
    wp.tile_atomic_add(ns, s2, offset=slot_out)


@wp.kernel
def k_set_w(omega: F, ns: wp.array(dtype=F), slot: wp.int32, wv: wp.array(dtype=F32), lvl: wp.int32):
    # damping omega / rho with rho = |x_last| (the input of the last step was normalized)
    rho = wp.sqrt(ns[slot])
    if rho < F(1e-12):
        rho = F(1e-12)
    wv[lvl] = F32(omega / rho)


# Prolongator weights: the candidate b (relaxation sweeps on A b = 0 from
# b = 1, as PyAMG's improve_candidates) normalized within every aggregate

@wp.kernel
def k_cand_f(slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32), V: wp.array(dtype=F32),
             Dinv: wp.array(dtype=F32), wc: wp.array(dtype=F32), lvl: wp.int32, x: wp.array(dtype=F32),
             y: wp.array(dtype=F32)):
    # y = x - wc D^-1 A x
    i = wp.tid()
    y[i] = x[i] - wc[lvl] * Dinv[i] * row_f(i, slice_off, cols, V, x)


@wp.kernel
def k_agg_norm_f(agg: wp.array(dtype=wp.int32), b: wp.array(dtype=F32), nrm: wp.array(dtype=F32)):
    i = wp.tid()
    wp.atomic_add(nrm, agg[i], b[i] * b[i])


@wp.kernel
def k_pb_f(agg: wp.array(dtype=wp.int32), b: wp.array(dtype=F32), nrm: wp.array(dtype=F32), pb: wp.array(dtype=F32)):
    i = wp.tid()
    s = nrm[agg[i]]
    if s > 0.0:
        pb[i] = b[i] / wp.sqrt(s)
    else:
        pb[i] = 0.0


# V-cycle kernels. Level l: r (restricted residual), z (smoothed correction),
# A (SELL float), Dinv, w = wv[l], agg (aggregate of each node = node of the
# next level), pb (prolongator weights). The finest level has double r (the
# CG residual).

@wp.kernel
def k_restrict_d(slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32), V: wp.array(dtype=F32),
                 r: wp.array(dtype=F), zs: wp.array(dtype=F32), agg: wp.array(dtype=wp.int32), pb: wp.array(dtype=F32),
                 rc: wp.array(dtype=F32), nzero: wp.int32, zarr: wp.array(dtype=F32)):
    # rc[agg[i]] += pb_i (r_i - (A zs)_i) (finest level, double r); zarr[0..nzero) = 0
    i = wp.tid()
    t = F32(r[i]) - row_f(i, slice_off, cols, V, zs)
    wp.atomic_add(rc, agg[i], pb[i] * t)
    if i < nzero:
        zarr[i] = 0.0


@wp.kernel
def k_lvl_mid_f(slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32), V: wp.array(dtype=F32),
                Dinv: wp.array(dtype=F32), wv: wp.array(dtype=F32), lvl: wp.int32, r: wp.array(dtype=F32),
                z: wp.array(dtype=F32), agg: wp.array(dtype=wp.int32), pb: wp.array(dtype=F32),
                rc: wp.array(dtype=F32), nzero: wp.int32, zarr: wp.array(dtype=F32)):
    # z = w D^-1 r; rc[agg[i]] += pb_i (r_i - (A z)_i); zarr[0..nzero) = 0
    i = wp.tid()
    w = wv[lvl]
    s = i // C
    q1 = slice_off[s + 1]
    acc = F32(0.0)
    for q in range(slice_off[s] + i % C, q1, U * C):
        c0 = cols[q]
        c1 = cols[q + C]
        c2 = cols[q + 2 * C]
        c3 = cols[q + 3 * C]
        acc = acc + V[q] * (Dinv[c0] * r[c0]) + V[q + C] * (Dinv[c1] * r[c1]) + V[q + 2 * C] * (Dinv[c2] * r[c2]) \
            + V[q + 3 * C] * (Dinv[c3] * r[c3])
    z[i] = w * Dinv[i] * r[i]
    wp.atomic_add(rc, agg[i], pb[i] * (r[i] - w * acc))
    if i < nzero:
        zarr[i] = 0.0


@wp.kernel
def k_coarse_f(nc: wp.int32, Ainv: wp.array2d(dtype=F32), r: wp.array(dtype=F32), z: wp.array(dtype=F32)):
    # z += Ainv r with NPART_C threads per row (z zero before)
    t = wp.tid()
    i = t // NPART_C
    s = F32(0.0)
    for k in range(t % NPART_C, nc, NPART_C):
        s = s + Ainv[i, k] * r[k]
    wp.atomic_add(z, i, s)


@wp.kernel
def k_prolong_f(z: wp.array(dtype=F32), agg: wp.array(dtype=wp.int32), pb: wp.array(dtype=F32),
                zc: wp.array(dtype=F32), z2: wp.array(dtype=F32)):
    i = wp.tid()
    z2[i] = z[i] + pb[i] * zc[agg[i]]


@wp.kernel
def k_lvl_post_f(slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32), V: wp.array(dtype=F32),
                 Dinv: wp.array(dtype=F32), wv: wp.array(dtype=F32), lvl: wp.int32, r: wp.array(dtype=F32),
                 z2: wp.array(dtype=F32), zout: wp.array(dtype=F32)):
    # zout = z2 + w D^-1 (r - A z2)
    i = wp.tid()
    zout[i] = z2[i] + wv[lvl] * Dinv[i] * (r[i] - row_f(i, slice_off, cols, V, z2))
