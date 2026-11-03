# SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
# SPDX-License-Identifier: BSD-3-Clause

"""Sliced ELLPACK (SELL-C) storage of a sparse pattern for coalesced sparse
matrix-vector products with one thread per row: the rows are grouped in
slices of C rows, every slice is padded to its longest row (rounded up to a
multiple of U) and stored entry by entry (entry k of all rows of the slice
are contiguous), so that the threads of a slice read consecutive memory. The
layout depends only on the pattern; the values are gathered from the CSR
order in every solve. The product loops are unrolled U times (independent
loads), which matters when a mesh is too small to hide the memory latency
with parallelism."""

import numpy as np
import warp as wp

from .common import DEVICE, F

wp.set_module_options({"enable_backward": False, "fuse_fp": False})

C = 32
U = 4


class Sell:
    def __init__(self, offsets, columns):
        offsets = np.asarray(offsets, dtype=np.int64)
        columns = np.asarray(columns, dtype=np.int64)
        n = len(offsets) - 1
        nnz = len(columns)
        self.n = n
        nsl = (n + C - 1) // C
        lengths = np.zeros(nsl * C, dtype=np.int64)
        lengths[:n] = np.diff(offsets)
        slice_len = (lengths.reshape(nsl, C).max(axis=1) + U - 1) // U * U
        slice_off = np.concatenate(([0], np.cumsum(slice_len * C)))
        total = int(slice_off[-1])
        rows = np.repeat(np.arange(n, dtype=np.int64), lengths[:n])
        k = np.arange(nnz, dtype=np.int64) - np.repeat(offsets[:-1], lengths[:n])
        q = slice_off[rows // C] + k * C + rows % C
        smap = np.full(total, -1, dtype=np.int64)
        smap[q] = np.arange(nnz)
        cols = np.zeros(total, dtype=np.int64)
        cols[q] = columns
        self.total = total
        self.slice_off = wp.array(slice_off.astype(np.int32), dtype=wp.int32, device=DEVICE)
        self.cols = wp.array(cols.astype(np.int32), dtype=wp.int32, device=DEVICE)
        self.map = wp.array(smap.astype(np.int32), dtype=wp.int32, device=DEVICE)
        self.padding = total / max(nnz, 1)

    def gather(self, V, Vs, kernel):
        """Vs[q] = V[map[q]] (0 for the padding)."""
        wp.launch(kernel, dim=self.total, inputs=[self.map, V, Vs], device=DEVICE)


@wp.kernel
def k_gather_s(smap: wp.array(dtype=wp.int32), V: wp.array(dtype=F), Vs: wp.array(dtype=F)):
    q = wp.tid()
    j = smap[q]
    if j >= 0:
        Vs[q] = V[j]
    else:
        Vs[q] = F(0.0)


@wp.kernel
def k_gather_v(smap: wp.array(dtype=wp.int32), V: wp.array(dtype=wp.vec3d), Vs: wp.array(dtype=wp.vec3d)):
    q = wp.tid()
    j = smap[q]
    if j >= 0:
        Vs[q] = V[j]
    else:
        Vs[q] = wp.vec3d()


@wp.kernel
def k_gather_m(smap: wp.array(dtype=wp.int32), V: wp.array(dtype=wp.mat33d), Vs: wp.array(dtype=wp.mat33d)):
    q = wp.tid()
    j = smap[q]
    if j >= 0:
        Vs[q] = V[j]
    else:
        Vs[q] = wp.mat33d()


# Products y = M x. Vectors with 3 components per node are flat: x[3*node+k]

@wp.func
def row_ss(i: wp.int32, slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32), V: wp.array(dtype=F),
           x: wp.array(dtype=F)):
    s = i // C
    q1 = slice_off[s + 1]
    acc = F(0.0)
    for q in range(slice_off[s] + i % C, q1, U * C):
        c0 = cols[q]
        c1 = cols[q + C]
        c2 = cols[q + 2 * C]
        c3 = cols[q + 3 * C]
        acc = acc + V[q] * x[c0] + V[q + C] * x[c1] + V[q + 2 * C] * x[c2] + V[q + 3 * C] * x[c3]
    return acc


@wp.func
def row_vv(i: wp.int32, slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32),
           K: wp.array(dtype=wp.mat33d), x: wp.array(dtype=F)):
    s = i // C
    q1 = slice_off[s + 1]
    acc = wp.vec3d()
    for q in range(slice_off[s] + i % C, q1, U * C):
        c0 = cols[q]
        c1 = cols[q + C]
        c2 = cols[q + 2 * C]
        c3 = cols[q + 3 * C]
        x0 = wp.vec3d(x[3 * c0], x[3 * c0 + 1], x[3 * c0 + 2])
        x1 = wp.vec3d(x[3 * c1], x[3 * c1 + 1], x[3 * c1 + 2])
        x2 = wp.vec3d(x[3 * c2], x[3 * c2 + 1], x[3 * c2 + 2])
        x3 = wp.vec3d(x[3 * c3], x[3 * c3 + 1], x[3 * c3 + 2])
        acc = acc + K[q] * x0 + K[q + C] * x1 + K[q + 2 * C] * x2 + K[q + 3 * C] * x3
    return acc


@wp.func
def row_vs(i: wp.int32, slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32),
           D: wp.array(dtype=wp.vec3d), x: wp.array(dtype=F)):
    s = i // C
    q1 = slice_off[s + 1]
    acc = F(0.0)
    for q in range(slice_off[s] + i % C, q1, U * C):
        c0 = cols[q]
        c1 = cols[q + C]
        c2 = cols[q + 2 * C]
        c3 = cols[q + 3 * C]
        x0 = wp.vec3d(x[3 * c0], x[3 * c0 + 1], x[3 * c0 + 2])
        x1 = wp.vec3d(x[3 * c1], x[3 * c1 + 1], x[3 * c1 + 2])
        x2 = wp.vec3d(x[3 * c2], x[3 * c2 + 1], x[3 * c2 + 2])
        x3 = wp.vec3d(x[3 * c3], x[3 * c3 + 1], x[3 * c3 + 2])
        acc = acc + wp.dot(D[q], x0) + wp.dot(D[q + C], x1) + wp.dot(D[q + 2 * C], x2) + wp.dot(D[q + 3 * C], x3)
    return acc


@wp.func
def row_sv(i: wp.int32, slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32),
           G: wp.array(dtype=wp.vec3d), x: wp.array(dtype=F)):
    s = i // C
    q1 = slice_off[s + 1]
    acc = wp.vec3d()
    for q in range(slice_off[s] + i % C, q1, U * C):
        c0 = cols[q]
        c1 = cols[q + C]
        c2 = cols[q + 2 * C]
        c3 = cols[q + 3 * C]
        acc = acc + G[q] * x[c0] + G[q + C] * x[c1] + G[q + 2 * C] * x[c2] + G[q + 3 * C] * x[c3]
    return acc


@wp.kernel
def k_spmv_ss(slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32), V: wp.array(dtype=F),
              x: wp.array(dtype=F), y: wp.array(dtype=F)):
    i = wp.tid()
    y[i] = row_ss(i, slice_off, cols, V, x)


@wp.kernel
def k_spmv_vv(slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32), K: wp.array(dtype=wp.mat33d),
              x: wp.array(dtype=F), y: wp.array(dtype=F)):
    i = wp.tid()
    acc = row_vv(i, slice_off, cols, K, x)
    y[3 * i] = acc[0]
    y[3 * i + 1] = acc[1]
    y[3 * i + 2] = acc[2]


@wp.kernel
def k_spmv_vs(slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32), D: wp.array(dtype=wp.vec3d),
              x: wp.array(dtype=F), y: wp.array(dtype=F)):
    i = wp.tid()
    y[i] = row_vs(i, slice_off, cols, D, x)


@wp.kernel
def k_sub_spmv_vs(slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32), D: wp.array(dtype=wp.vec3d),
                  x: wp.array(dtype=F), b: wp.array(dtype=F), y: wp.array(dtype=F)):
    # y = b - D x
    i = wp.tid()
    y[i] = b[i] - row_vs(i, slice_off, cols, D, x)


@wp.kernel
def k_spmv_sv_sub(slice_off: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32), G: wp.array(dtype=wp.vec3d),
                  x: wp.array(dtype=F), b: wp.array(dtype=F), y: wp.array(dtype=F), y2: wp.array(dtype=F)):
    # y = G x, y2 = b - G x
    i = wp.tid()
    acc = row_sv(i, slice_off, cols, G, x)
    for k in range(3):
        y[3 * i + k] = acc[k]
        y2[3 * i + k] = b[3 * i + k] - acc[k]
