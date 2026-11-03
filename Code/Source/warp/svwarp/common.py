# SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
# SPDX-License-Identifier: BSD-3-Clause

"""Types and helpers shared by the svwarp modules."""

import ctypes

import numpy as np
import warp as wp

F = wp.float64
F32 = wp.float32
vec4 = wp.vec4d
vec8 = wp.types.vector(8, wp.float64)
mat34 = wp.types.matrix(shape=(3, 4), dtype=wp.float64)
mat44 = wp.mat44d
mat16x16 = wp.types.matrix(shape=(16, 16), dtype=wp.float64)
mat31 = wp.types.matrix(shape=(3, 1), dtype=wp.float64)

DEVICE = "cuda:0"

# Block size of the kernels with block-wide (tile) reductions. Such kernels
# are launched with pad(n) threads so that every block is complete; threads
# with index >= n take part in the reductions with zero contributions.
BD = 256


def pad(n):
    return (int(n) + BD - 1) // BD * BD


def pad8(n):
    return (int(n) + 7) // 8 * 8


def sync():
    wp.synchronize_stream(wp.get_stream(DEVICE))


def host_array(ptr, dtype, shape):
    """Warp CPU array over host memory owned by the caller (no copy)."""
    return wp.array(ptr=int(ptr), dtype=dtype, shape=shape, device="cpu", copy=False)


def host_numpy(ptr, ctype, shape):
    """NumPy view over host memory owned by the caller (no copy)."""
    n = int(np.prod(shape))
    buf = (ctype * n).from_address(int(ptr))
    return np.frombuffer(buf, dtype=np.ctypeslib.as_ctypes_type(ctype)).reshape(shape)


def to_device(ptr, dtype, shape):
    """Copy host memory to a new device array."""
    d = wp.empty(shape, dtype=dtype, device=DEVICE)
    wp.copy(d, host_array(ptr, dtype, shape))
    return d


class Mapped:
    """Pinned host memory of n doubles (rounded up to a multiple of 8) that
    kernels read and write directly through the device alias `dev8` (mapped
    memory, as vectors of 8 doubles: 64-byte transactions). Small values
    cross between the host and the device this way inside CUDA graphs: a
    memcpy node costs 40-80 us on Windows, a one-block kernel 2 us. `np` is a
    NumPy view of the host memory; the host reads it after a stream
    synchronization."""

    def __init__(self, n):
        self.n = pad8(n)
        self.nvec = self.n // 8
        self.host = wp.zeros(self.n, dtype=F, device="cpu", pinned=True)
        self.dev8 = wp.array(ptr=self.host.ptr, dtype=vec8, shape=(self.nvec,), device=DEVICE, copy=False)
        self.np = self.host.numpy()
