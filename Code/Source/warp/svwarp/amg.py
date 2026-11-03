# SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
# SPDX-License-Identifier: BSD-3-Clause

"""Aggregation AMG preconditioner for a scalar sparse matrix (the Schur
complement of the NS solver).

Hierarchy: standard aggregation of the matrix graph (a root with its free
neighbors, remaining nodes join a neighboring aggregate), a prolongator with
one weighted column per aggregate, Galerkin coarse matrices, the inverse of
the coarsest matrix (at most CMAX nodes). The aggregation depends only on
the matrix graph and is computed once on the host; the weights and the
coarse values are computed on the device in every solve.

Prolongator weights: the FSILS diagonal scaling of the matrix makes the
near-kernel of the Schur complement far from constant, so a piecewise
constant prolongator is poor (about 100 CG iterations on the iliac artery
case against 40). The weights are the "improved candidate" of PyAMG: a few
damped Jacobi sweeps on A b = 0 from b = 1, normalized in every aggregate.

V-cycle: one damped Jacobi pre- and post-smoothing step on every level, with
the damping omega / rho(D^-1 A) (rho from a power iteration on the device).
The V-cycle is a fixed symmetric operator for a symmetric matrix, so that it
can precondition CG. The matrices and vectors of the hierarchy are in single
precision (sell32.py); the input and the output are double.

All device work is emitted as kernel launches without host synchronization,
so that the caller can capture it in CUDA graphs: emit_numeric() (weights,
coarse matrices, damping, the dense coarsest matrix and its inverse) and
emit_down() (the V-cycle below the finest level; the finest level is fused
into the CG kernels of ns.py)."""

import os
import time

import numpy as np
import warp as wp

from . import sell
from . import sell32 as s32
from .common import DEVICE, F, F32, BD, pad
from .linalg import k_zero

wp.set_module_options({"enable_backward": False, "fuse_fp": False})

# Size of the coarsest level (dense inverse by Gauss-Jordan steps on the
# device: 1 ms for 200 nodes, 9 ms for 512). Two levels with an exact coarse
# solve need fewer CG iterations than three.
CMAX = int(os.environ.get("SVWARP_AMG_CMAX", 512))
NPOW0 = 15   # power iteration steps in the first solve, then NPOW
NPOW = 4
NCAND = int(os.environ.get("SVWARP_AMG_CAND", 16))  # candidate relaxation sweeps
OMEGA_C = float(os.environ.get("SVWARP_AMG_WC", 1.0))


def aggregate(n, offsets, columns):
    """Standard aggregation (as in PyAMG) of the graph; returns agg (n) and
    the number of aggregates. PyAMG's C++ implementation is used when the
    package is installed (the Python loops take seconds on large meshes)."""
    try:
        import scipy.sparse
        from pyamg.aggregation.aggregate import standard_aggregation
        G = scipy.sparse.csr_matrix((np.ones(len(columns)), columns, offsets), shape=(n, n))
        AggOp, _ = standard_aggregation(G)
        AggOp = AggOp.tocsr()
        agg = np.full(n, -1, dtype=np.int64)
        has = np.diff(AggOp.indptr) > 0
        agg[has] = AggOp.indices
        na = AggOp.shape[1]
        if np.any(agg < 0):
            # Isolated nodes get their own aggregates
            iso = np.nonzero(agg < 0)[0]
            agg[iso] = na + np.arange(len(iso))
            na += len(iso)
        return agg, na
    except ImportError:
        pass
    agg = np.full(n, -1, dtype=np.int64)
    nbr = [columns[offsets[i]:offsets[i + 1]] for i in range(n)]
    na = 0
    # Pass 1: roots whose neighbors are all free
    for i in range(n):
        if agg[i] >= 0:
            continue
        nb = nbr[i]
        if np.all((agg[nb] < 0) | (nb == i)):
            agg[i] = na
            agg[nb] = na
            na += 1
    # Pass 2: join a neighboring aggregate of pass 1
    a2 = agg.copy()
    for i in range(n):
        if agg[i] >= 0:
            continue
        nb = nbr[i]
        k = np.nonzero(agg[nb] >= 0)[0]
        if len(k):
            a2[i] = agg[nb[k[0]]]
    agg = a2
    # Pass 3: new aggregates of the remaining nodes and their free neighbors
    for i in range(n):
        if agg[i] >= 0:
            continue
        agg[i] = na
        nb = nbr[i]
        free = nb[agg[nb] < 0]
        agg[free] = na
        na += 1
    return agg, na


class Level:
    pass


class Amg:
    def __init__(self, n, offsets, columns, sell0, graph_offsets=None, graph_columns=None, omega=4.0 / 3.0):
        """Structure of the hierarchy for a matrix with the CSR pattern
        (offsets, columns) whose SELL layout is sell0 (shared with the
        caller, which also owns the values of the finest level). The
        aggregation of the finest level uses the graph (graph_offsets,
        graph_columns) if given, otherwise the pattern."""
        t_setup = time.perf_counter()
        self.omega = float(os.environ.get("SVWARP_AMG_W", omega))
        self.levels = []
        self.nSolve = 0
        offsets = np.asarray(offsets, dtype=np.int64)
        columns = np.asarray(columns, dtype=np.int64)
        go = offsets if graph_offsets is None else np.asarray(graph_offsets, dtype=np.int64)
        gc = columns if graph_columns is None else np.asarray(graph_columns, dtype=np.int64)
        i32 = lambda a: wp.array(np.asarray(a).astype(np.int32), dtype=wp.int32, device=DEVICE)
        zf = lambda m: wp.zeros(m, dtype=F32, device=DEVICE)
        while True:
            L = Level()
            L.n = n
            L.nnz = len(columns)
            rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(offsets))
            L.diag = i32(np.nonzero(columns == rows)[0])
            if self.levels:
                L.sell = sell.Sell(offsets, columns)
                L.V = wp.zeros(L.nnz, dtype=F, device=DEVICE)       # CSR values (double)
                L.Vs = zf(L.sell.total)
                L.r, L.z, L.z2 = zf(n), zf(n), zf(n)
            else:
                L.sell = sell0
                L.V = None
                L.Vs = None
            L.Dinv = zf(n)
            self.levels.append(L)
            if n <= CMAX:
                break
            agg, na = aggregate(n, go, gc)
            if na >= n:
                break
            # Coarse pattern; coarse entry of every fine entry (Galerkin sums)
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
            L.agg = i32(agg)
            L.cmap, L.frow, L.fcol = i32(cmap), i32(rows), i32(columns)
            L.pb, L.bb, L.anrm = zf(n), [zf(n), zf(n)], zf(na)
            L.ev = [zf(n), zf(n)]
            n = na
            offsets = c_offsets
            columns = cJ
            go, gc = offsets, columns
        if len(self.levels) < 2:
            raise RuntimeError("svwarp: the AMG needs at least two levels")
        if os.environ.get("SV_GPU_VERBOSE"):
            print(f"svwarp: AMG setup levels {self.sizes()} in {time.perf_counter() - t_setup:.2f} s", flush=True)
        nl = len(self.levels)
        Lc = self.levels[-1]
        self.nc = Lc.n
        self.Ad = [wp.zeros((self.nc, self.nc), dtype=F, device=DEVICE), wp.zeros((self.nc, self.nc), dtype=F, device=DEVICE)]
        self.Ainv = wp.zeros((self.nc, self.nc), dtype=F32, device=DEVICE)
        self.wv = zf(nl)                                       # smoother damping per level
        self.wc = zf(nl)                                       # candidate relaxation damping
        self.ns = wp.zeros(16 * nl, dtype=F, device=DEVICE)    # norms of the power iterations
        self.n1 = self.levels[1].n

    def sizes(self):
        return [L.n for L in self.levels]

    def npow(self):
        """Power iteration steps of the next emit_numeric (the graphs of the
        caller are keyed by it)."""
        return NPOW0 if self.nSolve == 0 else NPOW

    def emit_numeric(self, V0, Vs0, nPow):
        """Kernel launches (no synchronization) of the weights, the coarse
        matrices, the smoothers and the dense coarsest matrix for the finest
        level values V0 (device CSR, double) and Vs0 (their SELL float copy)."""
        self.nSolve += 1
        first = nPow == NPOW0
        L0 = self.levels[0]
        L0.V, L0.Vs = V0, Vs0
        nl = len(self.levels)
        for l, L in enumerate(self.levels):
            wp.launch(s32.k_dinv_f, dim=L.n, inputs=[L.diag, L.V, L.Dinv], device=DEVICE)
            if l + 1 == nl:
                break
            A = [L.sell.slice_off, L.sell.cols, L.Vs]
            # Damping of the smoother and of the candidate relaxation with
            # the spectral radius of D^-1 A
            s0 = 16 * l
            wp.launch(k_zero, dim=16, inputs=[self.ns, s0], device=DEVICE)
            x, y = L.ev
            if first:
                wp.launch(s32.k_init_vec_f, dim=pad(L.n), inputs=[L.n, x, self.ns, s0], block_dim=BD, device=DEVICE)
            else:
                wp.launch(s32.k_norm_f, dim=pad(L.n), inputs=[L.n, x, self.ns, s0], block_dim=BD, device=DEVICE)
            for k in range(nPow):
                wp.launch(s32.k_dax_f, dim=pad(L.n), inputs=[L.n] + A + [L.Dinv, x, self.ns, s0 + k, y, s0 + k + 1],
                          block_dim=BD, device=DEVICE)
                x, y = y, x
            wp.launch(s32.k_set_w, dim=1, inputs=[F(self.omega), self.ns, s0 + nPow, self.wv, l], device=DEVICE)
            wp.launch(s32.k_set_w, dim=1, inputs=[F(OMEGA_C), self.ns, s0 + nPow, self.wc, l], device=DEVICE)
            if x is not L.ev[0]:
                wp.launch(s32.k_copy_f, dim=L.n, inputs=[x, L.ev[0]], device=DEVICE)
            # Prolongator weights: relaxed candidate, normalized per aggregate
            b0, b1 = L.bb
            wp.launch(s32.k_fill_f, dim=L.n, inputs=[b0, F32(1.0)], device=DEVICE)
            for k in range(NCAND):
                wp.launch(s32.k_cand_f, dim=L.n, inputs=A + [L.Dinv, self.wc, l, b0, b1], device=DEVICE)
                b0, b1 = b1, b0
            C = self.levels[l + 1]
            wp.launch(s32.k_zero_f, dim=C.n, inputs=[L.anrm], device=DEVICE)
            wp.launch(s32.k_agg_norm_f, dim=L.n, inputs=[L.agg, b0, L.anrm], device=DEVICE)
            wp.launch(s32.k_pb_f, dim=L.n, inputs=[L.agg, b0, L.anrm, L.pb], device=DEVICE)
            # Galerkin coarse matrix
            wp.launch(k_zero, dim=C.nnz, inputs=[C.V, 0], device=DEVICE)
            wp.launch(s32.k_galerkin, dim=L.nnz, inputs=[L.cmap, L.frow, L.fcol, L.pb, L.V, C.V], device=DEVICE)
            C.sell.gather(C.V, C.Vs, s32.k_gather_sf)
        Lc = self.levels[-1]
        nc = self.nc
        a, b = self.Ad
        wp.launch(s32.k_dense, dim=nc, inputs=[Lc.sell.slice_off, Lc.sell.cols, Lc.Vs, nc, a], device=DEVICE)
        for k in range(nc):
            wp.launch(s32.k_gj_step, dim=nc * nc, inputs=[nc, k, a, b], device=DEVICE)
            a, b = b, a
        wp.launch(s32.k_cast_f, dim=nc * nc, inputs=[nc, a, self.Ainv], device=DEVICE)

    def emit_down(self, r, zs):
        """V-cycle below the finest level: restriction of the finest residual
        r - A zs (r double, zs the pre-smoothed float correction) and the
        coarse levels; returns the float array with the correction of level
        1, to be prolongated and post-smoothed by the caller. The residual
        accumulator of level 1 must be zero."""
        lv = self.levels
        nl = len(lv)
        L0, L1, Lc = lv[0], lv[1], lv[-1]
        z = lv[2].r if nl > 2 else Lc.z
        wp.launch(s32.k_restrict_d, dim=L0.n,
                  inputs=[L0.sell.slice_off, L0.sell.cols, L0.Vs, r, zs, L0.agg, L0.pb, L1.r, len(z), z], device=DEVICE)
        for l in range(1, nl - 1):
            L, C = lv[l], lv[l + 1]
            z = lv[l + 2].r if l + 2 < nl else Lc.z
            wp.launch(s32.k_lvl_mid_f, dim=L.n,
                      inputs=[L.sell.slice_off, L.sell.cols, L.Vs, L.Dinv, self.wv, l, L.r, L.z, L.agg, L.pb, C.r,
                              len(z), z], device=DEVICE)
        wp.launch(s32.k_coarse_f, dim=self.nc * s32.NPART_C, inputs=[self.nc, self.Ainv, Lc.r, Lc.z], device=DEVICE)
        zc = Lc.z
        for l in range(nl - 2, 0, -1):
            L = lv[l]
            wp.launch(s32.k_prolong_f, dim=L.n, inputs=[L.z, L.agg, L.pb, zc, L.z2], device=DEVICE)
            wp.launch(s32.k_lvl_post_f, dim=L.n,
                      inputs=[L.sell.slice_off, L.sell.cols, L.Vs, L.Dinv, self.wv, l, L.r, L.z2, L.z], device=DEVICE)
            zc = L.z
        return zc
