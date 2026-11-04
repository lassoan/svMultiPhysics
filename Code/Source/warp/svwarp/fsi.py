# SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
# SPDX-License-Identifier: BSD-3-Clause

"""GPU assembly of the FSI equation (fsi.cpp: ALE fluid in the current
configuration with the mesh velocity, and the neo-Hookean wall with the M94
volumetric penalty, dof = 4) and of the mesh motion equation (mesh.cpp:
linear elasticity, dof = 3) on linear tetrahedra, with NVIDIA Warp. The
element routines are ports of fluid_3d_m/fluid_3d_c (fluid.py, with the
mesh velocity), struct_3d (neo-Hookean, M94) and l_elas_3d."""

import ctypes
import os
import time

import numpy as np
import warp as wp

from . import fluid as fl
from . import linalg as la
from . import sell
from .common import DEVICE, F, host_numpy, mat34, mat44, mat16x16, to_device, vec4

wp.set_module_options({"enable_backward": False, "fuse_fp": os.environ.get("SVWARP_FMA") is not None, "max_unroll": 16})

mat84 = wp.types.matrix(shape=(8, 4), dtype=wp.float64)     # nodal values, tDof (<= 8) x 4
mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
mat63 = wp.types.matrix(shape=(6, 3), dtype=wp.float64)
mat243 = wp.types.matrix(shape=(24, 3), dtype=wp.float64)   # Bm of the 4 nodes (6 x 3 each)
mat916 = wp.types.matrix(shape=(9, 16), dtype=wp.float64)   # lK of the mesh equation: 9 x (4a + b)
vec6i = wp.types.vector(6, wp.int32)
vec6 = wp.types.vector(6, wp.float64)


@wp.struct
class StructParams:
    rho: wp.float64
    dmp: wp.float64
    fb: wp.vec3d
    C10: wp.float64
    Kp: wp.float64
    dt: wp.float64
    af: wp.float64
    am: wp.float64
    beta: wp.float64
    gam: wp.float64


@wp.struct
class MeshParams:
    rho: wp.float64
    E: wp.float64
    nu: wp.float64
    f: wp.vec3d
    dt: wp.float64
    af: wp.float64
    am: wp.float64
    beta: wp.float64


@wp.func
def struct_3d(p: StructParams, w: wp.float64, N: vec4, Nx: mat34, al: mat84, yl: mat84, dl: mat84, bfl: mat34,
              lR: mat44, lK: mat16x16):
    """struct_ns::struct_3d for the neo-Hookean model with the M94 volumetric
    penalty, no viscosity, prestress or active stress (dof = 4, eq.s = 0)."""
    afu = p.af * p.beta * p.dt * p.dt
    amd = p.am * p.rho + p.af * p.gam * p.dt * p.dmp

    # Inertia, damping and body force
    ud = wp.vec3d()
    for i in range(3):
        s = F(0.0)
        for a in range(4):
            s = s + (p.rho * (al[i, a] - bfl[i, a]) + p.dmp * yl[i, a]) * N[a]
        ud[i] = s - p.rho * p.fb[i]

    # F = I + Grad(u)
    Fm = wp.mat33d()
    for i in range(3):
        for j in range(3):
            s = F(0.0)
            if i == j:
                s = F(1.0)
            for a in range(4):
                s = s + dl[i, a] * Nx[j, a]
            Fm[i, j] = s
    J = (Fm[0, 0] * (Fm[1, 1] * Fm[2, 2] - Fm[1, 2] * Fm[2, 1]) - Fm[0, 1] * (Fm[1, 0] * Fm[2, 2] - Fm[1, 2] * Fm[2, 0])
         + Fm[0, 2] * (Fm[1, 0] * Fm[2, 1] - Fm[1, 1] * Fm[2, 0]))
    J2d = wp.pow(J, F(-2.0 / 3.0))
    C = wp.mat33d()
    for i in range(3):
        for j in range(3):
            s = F(0.0)
            for k in range(3):
                s = s + Fm[k, i] * Fm[k, j]
            C[i, j] = s
    detC = (C[0, 0] * (C[1, 1] * C[2, 2] - C[1, 2] * C[2, 1]) - C[0, 1] * (C[1, 0] * C[2, 2] - C[1, 2] * C[2, 0])
            + C[0, 2] * (C[1, 0] * C[2, 1] - C[1, 1] * C[2, 0]))
    Ci = wp.mat33d()
    Ci[0, 0] = (C[1, 1] * C[2, 2] - C[1, 2] * C[2, 1]) / detC
    Ci[0, 1] = (C[0, 2] * C[2, 1] - C[0, 1] * C[2, 2]) / detC
    Ci[0, 2] = (C[0, 1] * C[1, 2] - C[0, 2] * C[1, 1]) / detC
    Ci[1, 0] = (C[1, 2] * C[2, 0] - C[1, 0] * C[2, 2]) / detC
    Ci[1, 1] = (C[0, 0] * C[2, 2] - C[0, 2] * C[2, 0]) / detC
    Ci[1, 2] = (C[0, 2] * C[1, 0] - C[0, 0] * C[1, 2]) / detC
    Ci[2, 0] = (C[1, 0] * C[2, 1] - C[1, 1] * C[2, 0]) / detC
    Ci[2, 1] = (C[0, 1] * C[2, 0] - C[0, 0] * C[2, 1]) / detC
    Ci[2, 2] = (C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]) / detC

    # Volumetric penalty (M94): p = Kp (1 - 1/J), pl = Kp
    pv = p.Kp * (F(1.0) - F(1.0) / J)
    pl = p.Kp
    # Isochoric neo-Hookean: S_bar = 2 C10 I, CC_bar = 0
    trC = C[0, 0] + C[1, 1] + C[2, 2]
    r1 = J2d * F(2.0) * p.C10 * trC / F(3.0)
    S = wp.mat33d()
    Siso = wp.mat33d()
    for i in range(3):
        for j in range(3):
            d = F(0.0)
            if i == j:
                d = F(2.0) * p.C10
            Siso[i, j] = J2d * d - r1 * Ci[i, j]
            S[i, j] = pv * J * Ci[i, j] + Siso[i, j]
    # CC = -2 p J sym(Ci,Ci) + pl J Ci x Ci - 2/3 (Ci x Siso + Siso x Ci) + 2 r1 sym(Ci,Ci) - 2 r1/3 Ci x Ci
    # in Voigt notation [11, 22, 33, 12, 23, 31]
    vi = vec6i(0, 1, 2, 0, 1, 2)
    vj = vec6i(0, 1, 2, 1, 2, 0)
    cs = F(-2.0) * pv * J + F(2.0) * r1
    cd = pl * J - F(2.0) * r1 / F(3.0)
    Dm = mat66()
    for I in range(6):
        i = vi[I]
        j = vj[I]
        for K in range(6):
            k = vi[K]
            l = vj[K]
            sym = F(0.5) * (Ci[i, k] * Ci[j, l] + Ci[i, l] * Ci[j, k])
            Dm[I, K] = (cs * sym + cd * Ci[i, j] * Ci[k, l]
                        - F(2.0) / F(3.0) * (Ci[i, j] * Siso[k, l] + Siso[i, j] * Ci[k, l]))

    # P = F S, local residual
    P = wp.mat33d()
    for i in range(3):
        for j in range(3):
            s = F(0.0)
            for k in range(3):
                s = s + Fm[i, k] * S[k, j]
            P[i, j] = s
    for a in range(4):
        for i in range(3):
            lR[i, a] = lR[i, a] + w * (ud[i] * N[a] + P[i, 0] * Nx[0, a] + P[i, 1] * Nx[1, a] + P[i, 2] * Nx[2, a])

    # Strain-displacement matrices Bm[a] (6 x 3), rows 6a..6a+5
    Bm = mat243()
    for a in range(4):
        g0 = Nx[0, a]
        g1 = Nx[1, a]
        g2 = Nx[2, a]
        for c in range(3):
            Bm[6 * a + 0, c] = g0 * Fm[c, 0]
            Bm[6 * a + 1, c] = g1 * Fm[c, 1]
            Bm[6 * a + 2, c] = g2 * Fm[c, 2]
            Bm[6 * a + 3, c] = g0 * Fm[c, 1] + g1 * Fm[c, 0]
            Bm[6 * a + 4, c] = g1 * Fm[c, 2] + g2 * Fm[c, 1]
            Bm[6 * a + 5, c] = g2 * Fm[c, 0] + g0 * Fm[c, 2]
    for b in range(4):
        DBm = mat63()
        for I in range(6):
            for c in range(3):
                s = F(0.0)
                for K in range(6):
                    s = s + Dm[I, K] * Bm[6 * b + K, c]
                DBm[I, c] = s
        SNx = wp.vec3d()
        for i in range(3):
            SNx[i] = S[i, 0] * Nx[0, b] + S[i, 1] * Nx[1, b] + S[i, 2] * Nx[2, b]
        for a in range(4):
            NxSNx = Nx[0, a] * SNx[0] + Nx[1, a] * SNx[1] + Nx[2, a] * SNx[2]
            T1 = amd * N[a] * N[b] + afu * NxSNx
            for r in range(3):
                for c in range(3):
                    BDB = F(0.0)
                    for I in range(6):
                        BDB = BDB + Bm[6 * a + I, r] * DBm[I, c]
                    d = F(0.0)
                    if r == c:
                        d = T1
                    lK[4 * r + c, 4 * a + b] = lK[4 * r + c, 4 * a + b] + w * (d + afu * BDB)
    return lR, lK


@wp.func
def l_elas_3d(p: MeshParams, w: wp.float64, N: vec4, Nx: mat34, al: mat84, dl: mat84, bfl: mat34, lR: mat34,
              lK: mat916):
    """l_elas::l_elas_3d for the mesh motion equation (dof = 3, eq.s = 4)."""
    i0 = 4
    lam = p.E * p.nu / (F(1.0) + p.nu) / (F(1.0) - F(2.0) * p.nu)
    mu = p.E * F(0.5) / (F(1.0) + p.nu)
    lDm = lam / mu
    T1 = p.af * p.beta * p.dt * p.dt
    amd = p.am / T1 * p.rho
    wl = w * T1 * mu
    ud = wp.vec3d(-p.f[0], -p.f[1], -p.f[2])
    ed = vec6()
    for a in range(4):
        ud[0] = ud[0] + N[a] * (al[i0, a] - bfl[0, a])
        ud[1] = ud[1] + N[a] * (al[i0 + 1, a] - bfl[1, a])
        ud[2] = ud[2] + N[a] * (al[i0 + 2, a] - bfl[2, a])
        ed[0] = ed[0] + Nx[0, a] * dl[i0, a]
        ed[1] = ed[1] + Nx[1, a] * dl[i0 + 1, a]
        ed[2] = ed[2] + Nx[2, a] * dl[i0 + 2, a]
        ed[3] = ed[3] + Nx[1, a] * dl[i0, a] + Nx[0, a] * dl[i0 + 1, a]
        ed[4] = ed[4] + Nx[2, a] * dl[i0 + 1, a] + Nx[1, a] * dl[i0 + 2, a]
        ed[5] = ed[5] + Nx[0, a] * dl[i0 + 2, a] + Nx[2, a] * dl[i0, a]
    divD = lam * (ed[0] + ed[1] + ed[2])
    S = vec6()
    S[0] = divD + F(2.0) * mu * ed[0]
    S[1] = divD + F(2.0) * mu * ed[1]
    S[2] = divD + F(2.0) * mu * ed[2]
    S[3] = mu * ed[3]
    S[4] = mu * ed[4]
    S[5] = mu * ed[5]
    for a in range(4):
        lR[0, a] = lR[0, a] + w * (p.rho * N[a] * ud[0] + Nx[0, a] * S[0] + Nx[1, a] * S[3] + Nx[2, a] * S[5])
        lR[1, a] = lR[1, a] + w * (p.rho * N[a] * ud[1] + Nx[0, a] * S[3] + Nx[1, a] * S[1] + Nx[2, a] * S[4])
        lR[2, a] = lR[2, a] + w * (p.rho * N[a] * ud[2] + Nx[0, a] * S[5] + Nx[1, a] * S[4] + Nx[2, a] * S[2])
        for b in range(4):
            ab = 4 * a + b
            NxdNx = Nx[0, a] * Nx[0, b] + Nx[1, a] * Nx[1, b] + Nx[2, a] * Nx[2, b]
            T = amd * N[a] * N[b] / mu + NxdNx
            lK[0, ab] = lK[0, ab] + wl * (T + (F(1.0) + lDm) * Nx[0, a] * Nx[0, b])
            lK[1, ab] = lK[1, ab] + wl * (lDm * Nx[0, a] * Nx[1, b] + Nx[1, a] * Nx[0, b])
            lK[2, ab] = lK[2, ab] + wl * (lDm * Nx[0, a] * Nx[2, b] + Nx[2, a] * Nx[0, b])
            lK[3, ab] = lK[3, ab] + wl * (lDm * Nx[1, a] * Nx[0, b] + Nx[0, a] * Nx[1, b])
            lK[4, ab] = lK[4, ab] + wl * (T + (F(1.0) + lDm) * Nx[1, a] * Nx[1, b])
            lK[5, ab] = lK[5, ab] + wl * (lDm * Nx[1, a] * Nx[2, b] + Nx[2, a] * Nx[1, b])
            lK[6, ab] = lK[6, ab] + wl * (lDm * Nx[2, a] * Nx[0, b] + Nx[0, a] * Nx[2, b])
            lK[7, ab] = lK[7, ab] + wl * (lDm * Nx[2, a] * Nx[1, b] + Nx[1, a] * Nx[2, b])
            lK[8, ab] = lK[8, ab] + wl * (T + (F(1.0) + lDm) * Nx[2, a] * Nx[2, b])
    return lR, lK


@wp.func
def load_element(e: wp.int32, IEN: wp.array2d(dtype=wp.int32), x: wp.array2d(dtype=F), tDof: wp.int32,
                 Ag: wp.array2d(dtype=F), Yg: wp.array2d(dtype=F), Dg: wp.array2d(dtype=F), Bf: wp.array2d(dtype=F)):
    xl = mat34()
    al = mat84()
    yl = mat84()
    dl = mat84()
    bfl = mat34()
    for a in range(4):
        Ac = IEN[e, a]
        for i in range(3):
            xl[i, a] = x[Ac, i]
            bfl[i, a] = Bf[Ac, i]
        for i in range(tDof):
            al[i, a] = Ag[Ac, i]
            yl[i, a] = Yg[Ac, i]
            dl[i, a] = Dg[Ac, i]
    return xl, al, yl, dl, bfl


@wp.kernel
def k_fsi_fluid(IEN: wp.array2d(dtype=wp.int32), x: wp.array2d(dtype=F), tDof: wp.int32, Ag: wp.array2d(dtype=F),
                Yg: wp.array2d(dtype=F), Dg: wp.array2d(dtype=F), Bf: wp.array2d(dtype=F), prm: fl.FluidParams,
                nG: wp.int32, gw: vec4, gN: mat44, Nxi: mat34, ePtr: wp.array2d(dtype=wp.int32),
                R: wp.array2d(dtype=F), Val: wp.array(dtype=wp.mat44d)):
    """FSI fluid domain: current configuration x + d(4..6), mesh velocity y(4..6)."""
    e = wp.tid()
    xl, al, yl, dl, bfl = load_element(e, IEN, x, tDof, Ag, Yg, Dg, Bf)
    al4 = mat44()
    yl4 = mat44()
    ylm = mat34()
    for a in range(4):
        for i in range(4):
            al4[i, a] = al[i, a]
            yl4[i, a] = yl[i, a]
        for i in range(3):
            xl[i, a] = xl[i, a] + dl[4 + i, a]
            ylm[i, a] = yl[4 + i, a]
    lR = mat44()
    lK = mat16x16()
    Nx, Jac, ks = fl.gnn(Nxi, xl)
    for g in range(nG):
        w = gw[g] * Jac
        N = vec4(gN[g, 0], gN[g, 1], gN[g, 2], gN[g, 3])
        lR, lK = fl.fluid_3d_m(prm, w, ks, N, Nx, al4, yl4, ylm, bfl, lR, lK)
    for g in range(nG):
        w = gw[g] * Jac
        N = vec4(gN[g, 0], gN[g, 1], gN[g, 2], gN[g, 3])
        lR, lK = fl.fluid_3d_c(prm, w, ks, N, Nx, al4, yl4, ylm, bfl, lR, lK)
    for a in range(4):
        rowN = IEN[e, a]
        for i in range(4):
            wp.atomic_add(R, rowN, i, lR[i, a])
        for b in range(4):
            ab = 4 * a + b
            blk = wp.mat44d()
            for r in range(4):
                for c in range(4):
                    blk[r, c] = lK[4 * r + c, ab]
            wp.atomic_add(Val, ePtr[e, ab], blk)


@wp.kernel
def k_fsi_struct(IEN: wp.array2d(dtype=wp.int32), x: wp.array2d(dtype=F), tDof: wp.int32, Ag: wp.array2d(dtype=F),
                 Yg: wp.array2d(dtype=F), Dg: wp.array2d(dtype=F), Bf: wp.array2d(dtype=F), prm: StructParams,
                 nG: wp.int32, gw: vec4, gN: mat44, Nxi: mat34, ePtr: wp.array2d(dtype=wp.int32),
                 R: wp.array2d(dtype=F), Val: wp.array(dtype=wp.mat44d)):
    """FSI struct domain (reference configuration)."""
    e = wp.tid()
    xl, al, yl, dl, bfl = load_element(e, IEN, x, tDof, Ag, Yg, Dg, Bf)
    lR = mat44()
    lK = mat16x16()
    Nx, Jac, ks = fl.gnn(Nxi, xl)
    for g in range(nG):
        N = vec4(gN[g, 0], gN[g, 1], gN[g, 2], gN[g, 3])
        lR, lK = struct_3d(prm, gw[g] * Jac, N, Nx, al, yl, dl, bfl, lR, lK)
    for a in range(4):
        rowN = IEN[e, a]
        for i in range(4):
            wp.atomic_add(R, rowN, i, lR[i, a])
        for b in range(4):
            ab = 4 * a + b
            blk = wp.mat44d()
            for r in range(4):
                for c in range(4):
                    blk[r, c] = lK[4 * r + c, ab]
            wp.atomic_add(Val, ePtr[e, ab], blk)


@wp.kernel
def k_mesh(IEN: wp.array2d(dtype=wp.int32), x: wp.array2d(dtype=F), tDof: wp.int32, Ag: wp.array2d(dtype=F),
           Yg: wp.array2d(dtype=F), Dg: wp.array2d(dtype=F), Do: wp.array2d(dtype=F), Bf: wp.array2d(dtype=F),
           prm: MeshParams, nG: wp.int32, gw: vec4, gN: mat44, Nxi: mat34, ePtr: wp.array2d(dtype=wp.int32),
           R: wp.array2d(dtype=F), Val: wp.array(dtype=wp.mat33d)):
    """Mesh motion: reference configuration x + Do(4..6), displacement d - Do."""
    e = wp.tid()
    xl, al, yl, dl, bfl = load_element(e, IEN, x, tDof, Ag, Yg, Dg, Bf)
    for a in range(4):
        Ac = IEN[e, a]
        for i in range(3):
            dol = Do[Ac, 4 + i]
            xl[i, a] = xl[i, a] + dol
            dl[4 + i, a] = dl[4 + i, a] - dol
            bfl[i, a] = F(0.0)
    lR = mat34()
    lK = mat916()
    Nx, Jac, ks = fl.gnn(Nxi, xl)
    for g in range(nG):
        N = vec4(gN[g, 0], gN[g, 1], gN[g, 2], gN[g, 3])
        lR, lK = l_elas_3d(prm, gw[g], N, Nx, al, dl, bfl, lR, lK)
    for a in range(4):
        rowN = IEN[e, a]
        for i in range(3):
            wp.atomic_add(R, rowN, i, lR[i, a])
        for b in range(4):
            ab = 4 * a + b
            blk = wp.mat33d()
            for r in range(3):
                for c in range(3):
                    blk[r, c] = lK[3 * r + c, ab]
            wp.atomic_add(Val, ePtr[e, ab], blk)


class Mesh:
    pass


class Fsi:
    """Node coordinates, device resident matrix (dof = 4: Val of the FSILS
    lhs, dof = 3: Val3), the meshes and the state vectors of the assembly."""

    def __init__(self, tnNo, x, nnz, rowPtr, colPtr):
        t0 = time.perf_counter()
        self.tnNo, self.nnz = int(tnNo), int(nnz)
        self.x = to_device(x, F, (tnNo, 3))
        offsets = host_numpy(rowPtr, ctypes.c_int, (tnNo + 1,)).copy()
        columns = host_numpy(colPtr, ctypes.c_int, (nnz,)).copy()
        self.lhs = la.Lhs(tnNo, offsets, columns)
        self.Val3 = wp.zeros(self.nnz, dtype=wp.mat33d, device=DEVICE)
        self.sell = sell.Sell(offsets, columns)
        self.meshes = []
        self.tDof = 0
        self.R = {4: wp.zeros((tnNo, 4), dtype=F, device=DEVICE), 3: wp.zeros((tnNo, 3), dtype=F, device=DEVICE)}
        self.Rh = {4: wp.zeros((tnNo, 4), dtype=F, device="cpu", pinned=True),
                   3: wp.zeros((tnNo, 3), dtype=F, device="cpu", pinned=True)}
        self.Bf = wp.zeros((tnNo, 3), dtype=F, device=DEVICE)
        self.Bfh = wp.zeros((tnNo, 3), dtype=F, device="cpu", pinned=True)
        self.Bfn = self.Bfh.numpy()
        self.solvers = {}
        self.stage_cap = 0
        self.t_assem = 0.0
        self.n_assem = 0
        wp.synchronize_device(DEVICE)
        if os.environ.get("SV_GPU_VERBOSE"):
            print(f"svwarp: FSI setup tnNo {tnNo} nnz {nnz} SELL padding {self.sell.padding:.2f} in "
                  f"{time.perf_counter() - t0:.2f} s", flush=True)

    def val(self, dof):
        return self.lhs.Val if dof == 4 else self.Val3

    def add_mesh(self, nEl, IEN, nG, w, N, Nxi):
        m = Mesh()
        m.nEl = int(nEl)
        m.IEN = to_device(IEN, wp.int32, (nEl, 4))
        m.ePtr = wp.zeros((nEl, 16), dtype=wp.int32, device=DEVICE)
        wp.launch(fl.k_elem_ptr, dim=nEl, inputs=[m.IEN, self.lhs.offsets, self.lhs.columns, m.ePtr], device=DEVICE)
        m.nG = int(nG)
        wn = host_numpy(w, ctypes.c_double, (nG,))
        Nn = host_numpy(N, ctypes.c_double, (nG, 4))
        Nxin = host_numpy(Nxi, ctypes.c_double, (4, 3))
        gw = [0.0] * 4
        for g in range(nG):
            gw[g] = float(wn[g])
        m.gw = vec4(*gw)
        m.gN = mat44()
        for g in range(nG):
            for a in range(4):
                m.gN[g, a] = float(Nn[g, a])
        m.Nxi = mat34()
        for a in range(4):
            for i in range(3):
                m.Nxi[i, a] = float(Nxin[a, i])
        self.meshes.append(m)
        wp.synchronize_device(DEVICE)
        return len(self.meshes) - 1

    def _state(self, tDof):
        if self.tDof == tDof:
            return
        for name in ("Ag", "Yg", "Dg", "Do"):
            setattr(self, name, wp.zeros((self.tnNo, tDof), dtype=F, device=DEVICE))
            setattr(self, name + "h", wp.zeros((self.tnNo, tDof), dtype=F, device="cpu", pinned=True))
            setattr(self, name + "n", getattr(self, name + "h").numpy())
        self.tDof = tDof

    def assemble(self, mesh, type_, tDof, Ag, Yg, Dg, Do, Bf, params, R):
        """Assemble one mesh into the device matrix (added) and add the
        residual to R (host, dof x tnNo). type: 0 = FSI fluid (ALE), 1 = FSI
        struct, 2 = mesh motion (params as svgpu_fsi_assemble)."""
        t0 = time.perf_counter()
        tnNo = self.tnNo
        tDof = int(tDof)
        if tDof > 8:
            raise RuntimeError("svwarp: tDof > 8 is not supported")
        self._state(tDof)
        for name, ptr in (("Ag", Ag), ("Yg", Yg), ("Dg", Dg)):
            getattr(self, name + "n")[:] = host_numpy(ptr, ctypes.c_double, (tnNo, tDof))
            wp.copy(getattr(self, name), getattr(self, name + "h"))
        if Do:
            self.Don[:] = host_numpy(Do, ctypes.c_double, (tnNo, tDof))
            wp.copy(self.Do, self.Doh)
        self.Bfn[:] = host_numpy(Bf, ctypes.c_double, (tnNo, 3))
        wp.copy(self.Bf, self.Bfh)
        p = host_numpy(params, ctypes.c_double, (12,))
        m = self.meshes[mesh]
        dof = 3 if type_ == 2 else 4
        Rd = self.R[dof]
        Rd.zero_()
        common = [m.IEN, self.x, tDof, self.Ag, self.Yg, self.Dg]
        if type_ == 0:
            prm = fl.FluidParams()
            prm.rho, prm.mu = F(p[0]), F(p[1])
            prm.f = wp.vec3d(F(p[2]), F(p[3]), F(p[4]))
            prm.dt, prm.af, prm.am, prm.gam, prm.brink = F(p[5]), F(p[6]), F(p[7]), F(p[8]), F(0.0)
            prm.mvMsh = 1
            wp.launch(k_fsi_fluid, dim=m.nEl, inputs=common + [self.Bf, prm, m.nG, m.gw, m.gN, m.Nxi, m.ePtr, Rd,
                                                                self.lhs.Val], device=DEVICE)
        elif type_ == 1:
            prm = StructParams()
            prm.rho, prm.dmp = F(p[0]), F(p[1])
            prm.fb = wp.vec3d(F(p[2]), F(p[3]), F(p[4]))
            prm.C10, prm.Kp, prm.dt, prm.af, prm.am, prm.beta, prm.gam = (F(p[5]), F(p[6]), F(p[7]), F(p[8]), F(p[9]),
                                                                           F(p[10]), F(p[11]))
            wp.launch(k_fsi_struct, dim=m.nEl, inputs=common + [self.Bf, prm, m.nG, m.gw, m.gN, m.Nxi, m.ePtr, Rd,
                                                                 self.lhs.Val], device=DEVICE)
        else:
            prm = MeshParams()
            prm.rho, prm.E, prm.nu = F(p[0]), F(p[1]), F(p[2])
            prm.f = wp.vec3d(F(p[3]), F(p[4]), F(p[5]))
            prm.dt, prm.af, prm.am, prm.beta = F(p[6]), F(p[7]), F(p[8]), F(p[9])
            wp.launch(k_mesh, dim=m.nEl, inputs=common + [self.Do, self.Bf, prm, m.nG, m.gw, m.gN, m.Nxi, m.ePtr, Rd,
                                                           self.Val3], device=DEVICE)
        wp.copy(self.Rh[dof], Rd)
        wp.synchronize_device(DEVICE)
        Rn = host_numpy(R, ctypes.c_double, (tnNo, dof))
        Rn += self.Rh[dof].numpy()
        self.t_assem += time.perf_counter() - t0
        self.n_assem += 1
        if os.environ.get("SVWARP_TIMING") and self.n_assem % 30 == 0:
            print(f"SVWARP fsi assemble per call {self.t_assem / self.n_assem:.4f} s", flush=True)
