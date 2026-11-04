# SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
# SPDX-License-Identifier: BSD-3-Clause

"""GPU assembly of the fluid equation (construct_fluid, fluid_3d_m and
fluid_3d_c of physics/fluid/src/fluid.cpp) with NVIDIA Warp, for linear
tetrahedra with VMS stabilization and a constant viscosity. For these elements
the second derivatives of the shape functions are zero and the viscosity does
not depend on the shear rate, so the terms with Nxx, mu_x and mu_g of the CPU
routines are left out; the remaining operations are the same, in the same
order, so that the results match the CPU assembly to rounding."""

import warp as wp

from .common import F, mat34, mat44, mat16x16, vec4

import os

# SVWARP_FMA=1: fused multiply-add (faster; the results differ from the CPU
# assembly in the last digits)
wp.set_module_options({"enable_backward": False, "fuse_fp": os.environ.get("SVWARP_FMA") is not None, "max_unroll": 16})


@wp.struct
class FluidParams:
    rho: wp.float64
    mu: wp.float64
    f: wp.vec3d
    dt: wp.float64
    af: wp.float64
    am: wp.float64
    gam: wp.float64
    brink: wp.float64     # Brinkman inverse permeability
    mvMsh: wp.int32       # moving mesh (ALE): the convection velocity is relative to the mesh velocity ylm


@wp.func
def is_zero(value1: wp.float64) -> wp.bool:
    # utils::is_zero(value, 0.0)
    a = wp.abs(value1)
    nrm = wp.max(a, F(2.220446049250313e-16))
    return (a - F(0.0)) / nrm < F(10.0) * F(2.220446049250313e-16)


@wp.func
def gnn(Nxi: mat34, xl: mat34):
    """nn::gnn for linear tetrahedra: shape function derivatives Nx (3 x 4),
    the Jacobian and ks = xiX^T xiX (the Kxi of fluid_3d)."""
    xXi = wp.mat33d()
    for a in range(4):
        for i in range(3):
            xXi[i, 0] = xXi[i, 0] + xl[i, a] * Nxi[0, a]
            xXi[i, 1] = xXi[i, 1] + xl[i, a] * Nxi[1, a]
            xXi[i, 2] = xXi[i, 2] + xl[i, a] * Nxi[2, a]
    Jac = (xXi[0, 0] * xXi[1, 1] * xXi[2, 2] + xXi[0, 1] * xXi[1, 2] * xXi[2, 0] + xXi[0, 2] * xXi[1, 0] * xXi[2, 1]
           - xXi[0, 0] * xXi[1, 2] * xXi[2, 1] - xXi[0, 1] * xXi[1, 0] * xXi[2, 2] - xXi[0, 2] * xXi[1, 1] * xXi[2, 0])
    xiX = wp.mat33d()
    xiX[0, 0] = (xXi[1, 1] * xXi[2, 2] - xXi[1, 2] * xXi[2, 1]) / Jac
    xiX[0, 1] = (xXi[2, 1] * xXi[0, 2] - xXi[2, 2] * xXi[0, 1]) / Jac
    xiX[0, 2] = (xXi[0, 1] * xXi[1, 2] - xXi[0, 2] * xXi[1, 1]) / Jac
    xiX[1, 0] = (xXi[1, 2] * xXi[2, 0] - xXi[1, 0] * xXi[2, 2]) / Jac
    xiX[1, 1] = (xXi[2, 2] * xXi[0, 0] - xXi[2, 0] * xXi[0, 2]) / Jac
    xiX[1, 2] = (xXi[0, 2] * xXi[1, 0] - xXi[0, 0] * xXi[1, 2]) / Jac
    xiX[2, 0] = (xXi[1, 0] * xXi[2, 1] - xXi[1, 1] * xXi[2, 0]) / Jac
    xiX[2, 1] = (xXi[2, 0] * xXi[0, 1] - xXi[2, 1] * xXi[0, 0]) / Jac
    xiX[2, 2] = (xXi[0, 0] * xXi[1, 1] - xXi[0, 1] * xXi[1, 0]) / Jac
    ks = wp.mat33d()
    ks[0, 0] = xiX[0, 0] * xiX[0, 0] + xiX[1, 0] * xiX[1, 0] + xiX[2, 0] * xiX[2, 0]
    ks[0, 1] = xiX[0, 1] * xiX[0, 0] + xiX[1, 1] * xiX[1, 0] + xiX[2, 1] * xiX[2, 0]
    ks[0, 2] = xiX[0, 2] * xiX[0, 0] + xiX[1, 2] * xiX[1, 0] + xiX[2, 2] * xiX[2, 0]
    ks[1, 1] = xiX[0, 1] * xiX[0, 1] + xiX[1, 1] * xiX[1, 1] + xiX[2, 1] * xiX[2, 1]
    ks[1, 2] = xiX[0, 1] * xiX[0, 2] + xiX[1, 1] * xiX[1, 2] + xiX[2, 1] * xiX[2, 2]
    ks[2, 2] = xiX[0, 2] * xiX[0, 2] + xiX[1, 2] * xiX[1, 2] + xiX[2, 2] * xiX[2, 2]
    ks[1, 0] = ks[0, 1]
    ks[2, 0] = ks[0, 2]
    ks[2, 1] = ks[1, 2]
    Nx = mat34()
    for a in range(4):
        Nx[0, a] = F(0.0) + Nxi[0, a] * xiX[0, 0] + Nxi[1, a] * xiX[1, 0] + Nxi[2, a] * xiX[2, 0]
        Nx[1, a] = F(0.0) + Nxi[0, a] * xiX[0, 1] + Nxi[1, a] * xiX[1, 1] + Nxi[2, a] * xiX[2, 1]
        Nx[2, a] = F(0.0) + Nxi[0, a] * xiX[0, 2] + Nxi[1, a] * xiX[1, 2] + Nxi[2, a] * xiX[2, 2]
    return Nx, Jac, ks


@wp.func
def fluid_3d_m(prm: FluidParams, w: wp.float64, Kxi: wp.mat33d, N: vec4, Nx: mat34, al: mat44, yl: mat44,
               ylm: mat34, bfl: mat34, lR: mat44, lK: mat16x16):
    """Momentum equation at one Gauss point (fluid_3d_m). lR(i,a) and
    lK(k, 4a+b) are accumulated. Linear elements: Nxx = 0, mu_g = mu_x = 0."""
    dt = prm.dt
    ctM = F(1.0)
    ctC = F(36.0)
    rho = prm.rho
    mu = prm.mu
    brink = prm.brink
    T1 = prm.af * prm.gam * dt
    amd = prm.am / T1
    wl = w * T1
    wr = w * rho

    ud = wp.vec3d(-prm.f[0], -prm.f[1], -prm.f[2])
    u = wp.vec3d()
    ux = wp.mat33d()
    for a in range(4):
        ud[0] = ud[0] + N[a] * (al[0, a] - bfl[0, a])
        ud[1] = ud[1] + N[a] * (al[1, a] - bfl[1, a])
        ud[2] = ud[2] + N[a] * (al[2, a] - bfl[2, a])
        u[0] = u[0] + N[a] * yl[0, a]
        u[1] = u[1] + N[a] * yl[1, a]
        u[2] = u[2] + N[a] * yl[2, a]
        ux[0, 0] = ux[0, 0] + Nx[0, a] * yl[0, a]
        ux[1, 0] = ux[1, 0] + Nx[1, a] * yl[0, a]
        ux[2, 0] = ux[2, 0] + Nx[2, a] * yl[0, a]
        ux[0, 1] = ux[0, 1] + Nx[0, a] * yl[1, a]
        ux[1, 1] = ux[1, 1] + Nx[1, a] * yl[1, a]
        ux[2, 1] = ux[2, 1] + Nx[2, a] * yl[1, a]
        ux[0, 2] = ux[0, 2] + Nx[0, a] * yl[2, a]
        ux[1, 2] = ux[1, 2] + Nx[1, a] * yl[2, a]
        ux[2, 2] = ux[2, 2] + Nx[2, a] * yl[2, a]
    divU = ux[0, 0] + ux[1, 1] + ux[2, 2]

    # Pressure and its gradient
    p = F(0.0)
    px = wp.vec3d()
    for a in range(4):
        p = p + N[a] * yl[3, a]
        px[0] = px[0] + Nx[0, a] * yl[3, a]
        px[1] = px[1] + Nx[1, a] * yl[3, a]
        px[2] = px[2] + Nx[2, a] * yl[3, a]

    # Convection velocity relative to the mesh velocity
    if prm.mvMsh != 0:
        for a in range(4):
            u[0] = u[0] - N[a] * ylm[0, a]
            u[1] = u[1] - N[a] * ylm[1, a]
            u[2] = u[2] - N[a] * ylm[2, a]

    # Strain rate tensor 2*e_ij := (u_i,j + u_j,i)
    es = wp.mat33d()
    es[0, 0] = ux[0, 0] + ux[0, 0]
    es[1, 1] = ux[1, 1] + ux[1, 1]
    es[2, 2] = ux[2, 2] + ux[2, 2]
    es[1, 0] = ux[1, 0] + ux[0, 1]
    es[2, 1] = ux[2, 1] + ux[1, 2]
    es[0, 2] = ux[0, 2] + ux[2, 0]
    es[0, 1] = es[1, 0]
    es[1, 2] = es[2, 1]
    es[2, 0] = es[0, 2]

    # Stabilization parameters
    kT = F(4.0) * ((ctM / dt) * (ctM / dt))
    kT = kT + (brink * mu / rho) * (brink * mu / rho)
    kU = (u[0] * u[0] * Kxi[0, 0] + u[1] * u[0] * Kxi[1, 0] + u[2] * u[0] * Kxi[2, 0]
          + u[0] * u[1] * Kxi[0, 1] + u[1] * u[1] * Kxi[1, 1] + u[2] * u[1] * Kxi[2, 1]
          + u[0] * u[2] * Kxi[0, 2] + u[1] * u[2] * Kxi[1, 2] + u[2] * u[2] * Kxi[2, 2])
    kS = (Kxi[0, 0] * Kxi[0, 0] + Kxi[1, 0] * Kxi[1, 0] + Kxi[2, 0] * Kxi[2, 0]
          + Kxi[0, 1] * Kxi[0, 1] + Kxi[1, 1] * Kxi[1, 1] + Kxi[2, 1] * Kxi[2, 1]
          + Kxi[0, 2] * Kxi[0, 2] + Kxi[1, 2] * Kxi[1, 2] + Kxi[2, 2] * Kxi[2, 2])
    kS = ctC * kS * ((mu / rho) * (mu / rho))
    tauM = F(1.0) / (rho * wp.sqrt(kT + kU + kS))

    rV = wp.vec3d()
    rV[0] = ud[0] + u[0] * ux[0, 0] + u[1] * ux[1, 0] + u[2] * ux[2, 0]
    rV[1] = ud[1] + u[0] * ux[0, 1] + u[1] * ux[1, 1] + u[2] * ux[2, 1]
    rV[2] = ud[2] + u[0] * ux[0, 2] + u[1] * ux[1, 2] + u[2] * ux[2, 2]

    # rS = mu_x.es + mu*d2u2 = 0 for linear elements
    up = wp.vec3d()
    up[0] = -tauM * (rho * rV[0] + px[0] + mu * brink * u[0])
    up[1] = -tauM * (rho * rV[1] + px[1] + mu * brink * u[1])
    up[2] = -tauM * (rho * rV[2] + px[2] + mu * brink * u[2])

    tauC = F(1.0) / (tauM * (Kxi[0, 0] + Kxi[1, 1] + Kxi[2, 2]))
    tauB = (up[0] * up[0] * Kxi[0, 0] + up[1] * up[0] * Kxi[1, 0]
            + up[2] * up[0] * Kxi[2, 0] + up[0] * up[1] * Kxi[0, 1]
            + up[1] * up[1] * Kxi[1, 1] + up[2] * up[1] * Kxi[2, 1]
            + up[0] * up[2] * Kxi[0, 2] + up[1] * up[2] * Kxi[1, 2]
            + up[2] * up[2] * Kxi[2, 2])
    if is_zero(tauB):
        tauB = F(2.220446049250313e-16)
    tauB = rho / wp.sqrt(tauB)

    ua = wp.vec3d(u[0] + up[0], u[1] + up[1], u[2] + up[2])
    pa = p - tauC * divU

    rV[0] = tauB * (up[0] * ux[0, 0] + up[1] * ux[1, 0] + up[2] * ux[2, 0])
    rV[1] = tauB * (up[0] * ux[0, 1] + up[1] * ux[1, 1] + up[2] * ux[2, 1])
    rV[2] = tauB * (up[0] * ux[0, 2] + up[1] * ux[1, 2] + up[2] * ux[2, 2])

    rM = wp.mat33d()
    rM[0, 0] = mu * es[0, 0] - rho * up[0] * ua[0] + rV[0] * up[0] - pa
    rM[1, 0] = mu * es[1, 0] - rho * up[0] * ua[1] + rV[0] * up[1]
    rM[2, 0] = mu * es[2, 0] - rho * up[0] * ua[2] + rV[0] * up[2]
    rM[0, 1] = mu * es[0, 1] - rho * up[1] * ua[0] + rV[1] * up[0]
    rM[1, 1] = mu * es[1, 1] - rho * up[1] * ua[1] + rV[1] * up[1] - pa
    rM[2, 1] = mu * es[2, 1] - rho * up[1] * ua[2] + rV[1] * up[2]
    rM[0, 2] = mu * es[0, 2] - rho * up[2] * ua[0] + rV[2] * up[0]
    rM[1, 2] = mu * es[1, 2] - rho * up[2] * ua[1] + rV[2] * up[1]
    rM[2, 2] = mu * es[2, 2] - rho * up[2] * ua[2] + rV[2] * up[2] - pa

    rV[0] = ud[0] + ua[0] * ux[0, 0] + ua[1] * ux[1, 0] + ua[2] * ux[2, 0]
    rV[1] = ud[1] + ua[0] * ux[0, 1] + ua[1] * ux[1, 1] + ua[2] * ux[2, 1]
    rV[2] = ud[2] + ua[0] * ux[0, 2] + ua[1] * ux[1, 2] + ua[2] * ux[2, 2]

    # Local residual and the quantities of the stiffness matrix
    # updu(i,j,a): for linear elements, with mu_x = mu_g = 0, only the
    # diagonal T1(a) remains
    uNx = vec4()
    upNx = vec4()
    uaNx = vec4()
    T1a = vec4()
    for a in range(4):
        lR[0, a] = lR[0, a] + wr * N[a] * rV[0] + w * (Nx[0, a] * rM[0, 0] + Nx[1, a] * rM[1, 0] + Nx[2, a] * rM[2, 0])
        lR[1, a] = lR[1, a] + wr * N[a] * rV[1] + w * (Nx[0, a] * rM[0, 1] + Nx[1, a] * rM[1, 1] + Nx[2, a] * rM[2, 1])
        lR[2, a] = lR[2, a] + wr * N[a] * rV[2] + w * (Nx[0, a] * rM[0, 2] + Nx[1, a] * rM[1, 2] + Nx[2, a] * rM[2, 2])
        uNx[a] = u[0] * Nx[0, a] + u[1] * Nx[1, a] + u[2] * Nx[2, a]
        upNx[a] = up[0] * Nx[0, a] + up[1] * Nx[1, a] + up[2] * Nx[2, a]
        uaNx[a] = uNx[a] + upNx[a]
        T1a[a] = -rho * uNx[a] - mu * brink * N[a]

    # Tangent (stiffness) matrices
    for b in range(4):
        for a in range(4):
            ab = 4 * a + b
            NxNx = Nx[0, a] * Nx[0, b] + Nx[1, a] * Nx[1, b] + Nx[2, a] * Nx[2, b]
            T1 = (mu * NxNx + rho * amd * N[b] * (N[a] + rho * tauM * uaNx[a]) + rho * N[a] * (uNx[b] + upNx[b])
                  + tauB * upNx[a] * upNx[b])
            T3 = rho * tauM * uaNx[a] * T1a[b]
            # dRm_a1/du_b1
            T2 = (mu + tauC) * (Nx[0, a] * Nx[0, b]) - T3
            lK[0, ab] = lK[0, ab] + wl * (T2 + T1)
            lK[0, ab] = lK[0, ab] + mu * brink * wl * N[b] * N[a]
            # dRm_a1/du_b2
            T2 = mu * (Nx[1, a] * Nx[0, b]) + tauC * (Nx[0, a] * Nx[1, b])
            lK[1, ab] = lK[1, ab] + wl * T2
            # dRm_a1/du_b3
            T2 = mu * (Nx[2, a] * Nx[0, b]) + tauC * (Nx[0, a] * Nx[2, b])
            lK[2, ab] = lK[2, ab] + wl * T2
            # dRm_a2/du_b1
            T2 = mu * (Nx[0, a] * Nx[1, b]) + tauC * (Nx[1, a] * Nx[0, b])
            lK[4, ab] = lK[4, ab] + wl * T2
            # dRm_a2/du_b2
            T2 = (mu + tauC) * (Nx[1, a] * Nx[1, b]) - T3
            lK[5, ab] = lK[5, ab] + wl * (T2 + T1)
            lK[5, ab] = lK[5, ab] + mu * brink * wl * N[b] * N[a]
            # dRm_a2/du_b3
            T2 = mu * (Nx[2, a] * Nx[1, b]) + tauC * (Nx[1, a] * Nx[2, b])
            lK[6, ab] = lK[6, ab] + wl * T2
            # dRm_a3/du_b1
            T2 = mu * (Nx[0, a] * Nx[2, b]) + tauC * (Nx[2, a] * Nx[0, b])
            lK[8, ab] = lK[8, ab] + wl * T2
            # dRm_a3/du_b2
            T2 = mu * (Nx[1, a] * Nx[2, b]) + tauC * (Nx[2, a] * Nx[1, b])
            lK[9, ab] = lK[9, ab] + wl * T2
            # dRm_a3/du_b3
            T2 = (mu + tauC) * (Nx[2, a] * Nx[2, b]) - T3
            lK[10, ab] = lK[10, ab] + wl * (T2 + T1)
            lK[10, ab] = lK[10, ab] + mu * brink * wl * N[b] * N[a]

    for b in range(4):
        for a in range(4):
            ab = 4 * a + b
            T1 = rho * tauM * uaNx[a]
            lK[3, ab] = lK[3, ab] - wl * (Nx[0, a] * N[b] - Nx[0, b] * T1)
            lK[7, ab] = lK[7, ab] - wl * (Nx[1, a] * N[b] - Nx[1, b] * T1)
            lK[11, ab] = lK[11, ab] - wl * (Nx[2, a] * N[b] - Nx[2, b] * T1)

    # Residual contribution of the Brinkman term
    for a in range(4):
        lR[0, a] = lR[0, a] + mu * brink * w * N[a] * (u[0] + up[0])
        lR[1, a] = lR[1, a] + mu * brink * w * N[a] * (u[1] + up[1])
        lR[2, a] = lR[2, a] + mu * brink * w * N[a] * (u[2] + up[2])
    return lR, lK


@wp.func
def fluid_3d_c(prm: FluidParams, w: wp.float64, Kxi: wp.mat33d, N: vec4, Nx: mat34, al: mat44, yl: mat44,
               ylm: mat34, bfl: mat34, lR: mat44, lK: mat16x16):
    """Continuity equation at one Gauss point (fluid_3d_c)."""
    dt = prm.dt
    ctM = F(1.0)
    ctC = F(36.0)
    rho = prm.rho
    mu = prm.mu
    brink = prm.brink
    T1 = prm.af * prm.gam * dt
    amd = prm.am / T1
    wl = w * T1

    ud = wp.vec3d(-prm.f[0], -prm.f[1], -prm.f[2])
    u = wp.vec3d()
    ux = wp.mat33d()
    for a in range(4):
        ud[0] = ud[0] + N[a] * (al[0, a] - bfl[0, a])
        ud[1] = ud[1] + N[a] * (al[1, a] - bfl[1, a])
        ud[2] = ud[2] + N[a] * (al[2, a] - bfl[2, a])
        u[0] = u[0] + N[a] * yl[0, a]
        u[1] = u[1] + N[a] * yl[1, a]
        u[2] = u[2] + N[a] * yl[2, a]
        ux[0, 0] = ux[0, 0] + Nx[0, a] * yl[0, a]
        ux[1, 0] = ux[1, 0] + Nx[1, a] * yl[0, a]
        ux[2, 0] = ux[2, 0] + Nx[2, a] * yl[0, a]
        ux[0, 1] = ux[0, 1] + Nx[0, a] * yl[1, a]
        ux[1, 1] = ux[1, 1] + Nx[1, a] * yl[1, a]
        ux[2, 1] = ux[2, 1] + Nx[2, a] * yl[1, a]
        ux[0, 2] = ux[0, 2] + Nx[0, a] * yl[2, a]
        ux[1, 2] = ux[1, 2] + Nx[1, a] * yl[2, a]
        ux[2, 2] = ux[2, 2] + Nx[2, a] * yl[2, a]
    divU = ux[0, 0] + ux[1, 1] + ux[2, 2]

    px = wp.vec3d()
    for a in range(4):
        px[0] = px[0] + Nx[0, a] * yl[3, a]
        px[1] = px[1] + Nx[1, a] * yl[3, a]
        px[2] = px[2] + Nx[2, a] * yl[3, a]

    if prm.mvMsh != 0:
        for a in range(4):
            u[0] = u[0] - N[a] * ylm[0, a]
            u[1] = u[1] - N[a] * ylm[1, a]
            u[2] = u[2] - N[a] * ylm[2, a]

    kT = F(4.0) * ((ctM / dt) * (ctM / dt))
    kT = kT + (brink * mu / rho) * (brink * mu / rho)
    kU = (u[0] * u[0] * Kxi[0, 0] + u[1] * u[0] * Kxi[1, 0] + u[2] * u[0] * Kxi[2, 0]
          + u[0] * u[1] * Kxi[0, 1] + u[1] * u[1] * Kxi[1, 1] + u[2] * u[1] * Kxi[2, 1]
          + u[0] * u[2] * Kxi[0, 2] + u[1] * u[2] * Kxi[1, 2] + u[2] * u[2] * Kxi[2, 2])
    kS = (Kxi[0, 0] * Kxi[0, 0] + Kxi[1, 0] * Kxi[1, 0] + Kxi[2, 0] * Kxi[2, 0]
          + Kxi[0, 1] * Kxi[0, 1] + Kxi[1, 1] * Kxi[1, 1] + Kxi[2, 1] * Kxi[2, 1]
          + Kxi[0, 2] * Kxi[0, 2] + Kxi[1, 2] * Kxi[1, 2] + Kxi[2, 2] * Kxi[2, 2])
    kS = ctC * kS * ((mu / rho) * (mu / rho))
    tauM = F(1.0) / (rho * wp.sqrt(kT + kU + kS))

    rV = wp.vec3d()
    rV[0] = ud[0] + u[0] * ux[0, 0] + u[1] * ux[1, 0] + u[2] * ux[2, 0]
    rV[1] = ud[1] + u[0] * ux[0, 1] + u[1] * ux[1, 1] + u[2] * ux[2, 1]
    rV[2] = ud[2] + u[0] * ux[0, 2] + u[1] * ux[1, 2] + u[2] * ux[2, 2]
    up = wp.vec3d()
    up[0] = -tauM * (rho * rV[0] + px[0] + mu * brink * u[0])
    up[1] = -tauM * (rho * rV[1] + px[1] + mu * brink * u[1])
    up[2] = -tauM * (rho * rV[2] + px[2] + mu * brink * u[2])

    # updu(i,i,a) = T1(a) = -rho*uNx - mu*brink*N(a) (other entries zero)
    T1a = vec4()
    for a in range(4):
        uNx = u[0] * Nx[0, a] + u[1] * Nx[1, a] + u[2] * Nx[2, a]
        T1a[a] = -rho * uNx - mu * brink * N[a]

    # Local residual
    for a in range(4):
        upNx = up[0] * Nx[0, a] + up[1] * Nx[1, a] + up[2] * Nx[2, a]
        lR[3, a] = lR[3, a] + w * (N[a] * divU - upNx)

    # Tangent (stiffness) matrices
    for b in range(4):
        T1 = rho * amd * N[b]
        for a in range(4):
            ab = 4 * a + b
            T2 = Nx[0, a] * (T1a[b] - T1)
            lK[12, ab] = lK[12, ab] + wl * (N[a] * Nx[0, b] - tauM * T2)
            T2 = Nx[1, a] * (T1a[b] - T1)
            lK[13, ab] = lK[13, ab] + wl * (N[a] * Nx[1, b] - tauM * T2)
            T2 = Nx[2, a] * (T1a[b] - T1)
            lK[14, ab] = lK[14, ab] + wl * (N[a] * Nx[2, b] - tauM * T2)

    for b in range(4):
        for a in range(4):
            ab = 4 * a + b
            NxNx = Nx[0, a] * Nx[0, b] + Nx[1, a] * Nx[1, b] + Nx[2, a] * Nx[2, b]
            lK[15, ab] = lK[15, ab] + wl * tauM * NxNx
    return lR, lK


@wp.kernel
def k_fluid(IEN: wp.array2d(dtype=wp.int32), x: wp.array2d(dtype=wp.float64), Ag: wp.array2d(dtype=wp.float64),
            Yg: wp.array2d(dtype=wp.float64), Bf: wp.array2d(dtype=wp.float64), prm: FluidParams, nG: wp.int32,
            gw: vec4, gN: mat44, Nxi: mat34, ePtr: wp.array2d(dtype=wp.int32), R: wp.array2d(dtype=wp.float64),
            Val: wp.array(dtype=wp.mat44d)):
    """One element per thread: residual into R (tnNo x 4) and the element
    matrix into the blocks Val[ePtr[e, 4a+b]] (4 x 4, row i = residual
    component, column j = dof), with atomic additions."""
    e = wp.tid()
    xl = mat34()
    al = mat44()
    yl = mat44()
    bfl = mat34()
    for a in range(4):
        Ac = IEN[e, a]
        for i in range(3):
            xl[i, a] = x[Ac, i]
            bfl[i, a] = Bf[Ac, i]
        for i in range(4):
            al[i, a] = Ag[Ac, i]
            yl[i, a] = Yg[Ac, i]
    lR = mat44()
    lK = mat16x16()
    ylm = mat34()
    Nx, Jac, ks = gnn(Nxi, xl)
    for g in range(nG):
        w = gw[g] * Jac
        N = vec4(gN[g, 0], gN[g, 1], gN[g, 2], gN[g, 3])
        lR, lK = fluid_3d_m(prm, w, ks, N, Nx, al, yl, ylm, bfl, lR, lK)
    for g in range(nG):
        w = gw[g] * Jac
        N = vec4(gN[g, 0], gN[g, 1], gN[g, 2], gN[g, 3])
        lR, lK = fluid_3d_c(prm, w, ks, N, Nx, al, yl, ylm, bfl, lR, lK)
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
def k_elem_ptr(IEN: wp.array2d(dtype=wp.int32), offsets: wp.array(dtype=wp.int32), columns: wp.array(dtype=wp.int32),
               ePtr: wp.array2d(dtype=wp.int32)):
    """Index of the matrix block (IEN[e,a], IEN[e,b]) for every element node
    pair (binary search in the sorted columns of the row)."""
    e = wp.tid()
    for a in range(4):
        rowN = IEN[e, a]
        for b in range(4):
            colN = IEN[e, b]
            left = offsets[rowN]
            right = offsets[rowN + 1]
            ptr = (right + left) / 2
            while colN != columns[ptr]:
                if colN > columns[ptr]:
                    left = ptr
                else:
                    right = ptr
                ptr = (right + left) / 2
            ePtr[e, 4 * a + b] = ptr
