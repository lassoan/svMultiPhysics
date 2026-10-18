// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// GPU assembly and linear solvers for FSI (fsi.cpp, mesh.cpp):
//   - FSI equation (dof = 4): fluid domain in the current configuration with
//     the mesh velocity (ALE), and struct domain (neo-Hookean with the M94
//     volumetric penalty), on linear tetrahedra
//   - mesh motion equation (dof = 3): linear elasticity (l_elas_3d)
//   - FSILS GMRES (gmres_v) and CG (cgrad_v) for dof = 3 and 4, and the FSILS
//     diagonal preconditioner, on the device resident matrix.

#include "svgpu.h"
#include "svgpu_internal.h"

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } } while (0)

extern double* svgpu_dev_val_ptr;

namespace {

namespace cg = cooperative_groups;

constexpr int TPB = 256;

//===========================================================================
// Element routines

struct FluidParams {
  double rho, mu, f[3], dt, af, am, gam, brink;
  int mvMsh;
};

__device__ inline bool is_zero(double value1, double value2 = 0.0)
{
  double a = fabs(value1), b = fabs(value2);
  if (b > a) { double t = a; a = b; b = t; }
  double nrm = fmax(a, DBL_EPSILON);
  return (a - b) / nrm < 10.0 * DBL_EPSILON;
}

#define maxNoN 4
#define Nw(a) Nw_[a]
#define Nq(a) Nq_[a]
#define Nwx(i,a) Nwx_[i][a]
#define Nqx(i,a) Nqx_[i][a]
#define Nwxx(i,a) Nwxx_[i][a]
#define al(i,a) al_[i][a]
#define yl(i,a) yl_[i][a]
#define bfl(i,a) bfl_[i][a]
#define lR(i,a) lR_[i][a]
#define lK(k,a,b) lK_[k][a][b]
#define Kxi(i,j) Kxi_[i][j]
#include "fluid_element.inc"
#undef Nw
#undef Nq
#undef Nwx
#undef Nqx
#undef Nwxx
#undef al
#undef yl
#undef bfl
#undef lR
#undef lK
#undef Kxi

struct GaussData {
  int nG;
  double w[4];
  double N[4][4];     // N[g][a]
  double Nxi[3][4];
};

// nn::gnn for nsd = insd = 3
__device__ void d_gnn(const double Nxi[3][4], const double x[3][4], double Nx[3][4], double& Jac, double ks[3][3])
{
  double xXi[3][3] = {};
  for (int a = 0; a < 4; a++) {
    for (int i = 0; i < 3; i++) {
      xXi[i][0] = xXi[i][0] + x[i][a] * Nxi[0][a];
      xXi[i][1] = xXi[i][1] + x[i][a] * Nxi[1][a];
      xXi[i][2] = xXi[i][2] + x[i][a] * Nxi[2][a];
    }
  }
  Jac = xXi[0][0]*xXi[1][1]*xXi[2][2] + xXi[0][1]*xXi[1][2]*xXi[2][0] + xXi[0][2]*xXi[1][0]*xXi[2][1] -
        xXi[0][0]*xXi[1][2]*xXi[2][1] - xXi[0][1]*xXi[1][0]*xXi[2][2] - xXi[0][2]*xXi[1][1]*xXi[2][0];
  double xiX[3][3];
  xiX[0][0] = (xXi[1][1]*xXi[2][2] - xXi[1][2]*xXi[2][1])/Jac;
  xiX[0][1] = (xXi[2][1]*xXi[0][2] - xXi[2][2]*xXi[0][1])/Jac;
  xiX[0][2] = (xXi[0][1]*xXi[1][2] - xXi[0][2]*xXi[1][1])/Jac;
  xiX[1][0] = (xXi[1][2]*xXi[2][0] - xXi[1][0]*xXi[2][2])/Jac;
  xiX[1][1] = (xXi[2][2]*xXi[0][0] - xXi[2][0]*xXi[0][2])/Jac;
  xiX[1][2] = (xXi[0][2]*xXi[1][0] - xXi[0][0]*xXi[1][2])/Jac;
  xiX[2][0] = (xXi[1][0]*xXi[2][1] - xXi[1][1]*xXi[2][0])/Jac;
  xiX[2][1] = (xXi[2][0]*xXi[0][1] - xXi[2][1]*xXi[0][0])/Jac;
  xiX[2][2] = (xXi[0][0]*xXi[1][1] - xXi[0][1]*xXi[1][0])/Jac;
  ks[0][0] = xiX[0][0]*xiX[0][0]+xiX[1][0]*xiX[1][0]+xiX[2][0]*xiX[2][0];
  ks[0][1] = xiX[0][1]*xiX[0][0]+xiX[1][1]*xiX[1][0]+xiX[2][1]*xiX[2][0];
  ks[0][2] = xiX[0][2]*xiX[0][0]+xiX[1][2]*xiX[1][0]+xiX[2][2]*xiX[2][0];
  ks[1][1] = xiX[0][1]*xiX[0][1]+xiX[1][1]*xiX[1][1]+xiX[2][1]*xiX[2][1];
  ks[1][2] = xiX[0][1]*xiX[0][2]+xiX[1][1]*xiX[1][2]+xiX[2][1]*xiX[2][2];
  ks[2][2] = xiX[0][2]*xiX[0][2]+xiX[1][2]*xiX[1][2]+xiX[2][2]*xiX[2][2];
  ks[1][0] = ks[0][1];
  ks[2][0] = ks[0][2];
  ks[2][1] = ks[1][2];
  for (int a = 0; a < 4; a++) {
    Nx[0][a] = 0.0 + Nxi[0][a]*xiX[0][0] + Nxi[1][a]*xiX[1][0] + Nxi[2][a]*xiX[2][0];
    Nx[1][a] = 0.0 + Nxi[0][a]*xiX[0][1] + Nxi[1][a]*xiX[1][1] + Nxi[2][a]*xiX[2][1];
    Nx[2][a] = 0.0 + Nxi[0][a]*xiX[0][2] + Nxi[1][a]*xiX[1][2] + Nxi[2][a]*xiX[2][2];
  }
}

struct StructParams {
  double rho, dmp, fb[3], C10, Kp, dt, af, am, beta, gam;
};

// struct_ns::struct_3d for the neo-Hookean model with the M94 volumetric
// penalty, no viscosity, prestress or active stress (dof = 4, eq.s = 0)
__device__ void d_struct_3d(const StructParams& p, const double w, const double* N, const double Nx[3][4],
    const double al[8][4], const double yl[8][4], const double dl[8][4], const double bfl[3][4],
    double lR[4][4], double lK[16][4][4])
{
  const double afu = p.af * p.beta * p.dt * p.dt;
  const double amd = p.am * p.rho + p.af * p.gam * p.dt * p.dmp;

  // Inertia, damping and body force
  double ud[3];
  for (int i = 0; i < 3; i++) {
    double s = 0.0;
    for (int a = 0; a < 4; a++) s += (p.rho * (al[i][a] - bfl[i][a]) + p.dmp * yl[i][a]) * N[a];
    ud[i] = s - p.rho * p.fb[i];
  }

  // F = I + Grad(u)
  double F[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      double s = (i == j) ? 1.0 : 0.0;
      for (int a = 0; a < 4; a++) s += dl[i][a] * Nx[j][a];
      F[i][j] = s;
    }
  }
  const double J = F[0][0]*(F[1][1]*F[2][2]-F[1][2]*F[2][1]) - F[0][1]*(F[1][0]*F[2][2]-F[1][2]*F[2][0])
                 + F[0][2]*(F[1][0]*F[2][1]-F[1][1]*F[2][0]);
  const double J2d = pow(J, -2.0 / 3.0);
  double C[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      double s = 0.0;
      for (int k = 0; k < 3; k++) s += F[k][i] * F[k][j];
      C[i][j] = s;
    }
  }
  const double detC = C[0][0]*(C[1][1]*C[2][2]-C[1][2]*C[2][1]) - C[0][1]*(C[1][0]*C[2][2]-C[1][2]*C[2][0])
                    + C[0][2]*(C[1][0]*C[2][1]-C[1][1]*C[2][0]);
  double Ci[3][3];
  Ci[0][0] = (C[1][1]*C[2][2]-C[1][2]*C[2][1]) / detC;
  Ci[0][1] = (C[0][2]*C[2][1]-C[0][1]*C[2][2]) / detC;
  Ci[0][2] = (C[0][1]*C[1][2]-C[0][2]*C[1][1]) / detC;
  Ci[1][0] = (C[1][2]*C[2][0]-C[1][0]*C[2][2]) / detC;
  Ci[1][1] = (C[0][0]*C[2][2]-C[0][2]*C[2][0]) / detC;
  Ci[1][2] = (C[0][2]*C[1][0]-C[0][0]*C[1][2]) / detC;
  Ci[2][0] = (C[1][0]*C[2][1]-C[1][1]*C[2][0]) / detC;
  Ci[2][1] = (C[0][1]*C[2][0]-C[0][0]*C[2][1]) / detC;
  Ci[2][2] = (C[0][0]*C[1][1]-C[0][1]*C[1][0]) / detC;

  // Volumetric penalty (M94): p = Kp (1 - 1/J), pl = Kp
  const double pv = p.Kp * (1.0 - 1.0 / J), pl = p.Kp;
  // Isochoric neo-Hookean: S_bar = 2 C10 I, CC_bar = 0
  const double trC = C[0][0] + C[1][1] + C[2][2];
  const double r1 = J2d * 2.0 * p.C10 * trC / 3.0;
  double S[3][3], Siso[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Siso[i][j] = J2d * ((i == j) ? 2.0 * p.C10 : 0.0) - r1 * Ci[i][j];
      S[i][j] = pv * J * Ci[i][j] + Siso[i][j];
    }
  }
  // CC = -2 p J sym(Ci,Ci) + pl J Ci x Ci - 2/3 (Ci x Siso + Siso x Ci) + 2 r1 sym(Ci,Ci) - 2 r1/3 Ci x Ci
  // in Voigt notation [11, 22, 33, 12, 23, 31]
  const int vi[6] = {0, 1, 2, 0, 1, 2}, vj[6] = {0, 1, 2, 1, 2, 0};
  const double cs = -2.0 * pv * J + 2.0 * r1, cd = pl * J - 2.0 * r1 / 3.0;
  double Dm[6][6];
  for (int I = 0; I < 6; I++) {
    const int i = vi[I], j = vj[I];
    for (int K = 0; K < 6; K++) {
      const int k = vi[K], l = vj[K];
      const double sym = 0.5 * (Ci[i][k] * Ci[j][l] + Ci[i][l] * Ci[j][k]);
      Dm[I][K] = cs * sym + cd * Ci[i][j] * Ci[k][l] - 2.0 / 3.0 * (Ci[i][j] * Siso[k][l] + Siso[i][j] * Ci[k][l]);
    }
  }

  // P = F S, local residual
  double P[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      double s = 0.0;
      for (int k = 0; k < 3; k++) s += F[i][k] * S[k][j];
      P[i][j] = s;
    }
  }
  for (int a = 0; a < 4; a++) {
    for (int i = 0; i < 3; i++) {
      lR[i][a] += w * (ud[i] * N[a] + P[i][0] * Nx[0][a] + P[i][1] * Nx[1][a] + P[i][2] * Nx[2][a]);
    }
  }

  // Strain-displacement matrices Bm[a] (6 x 3)
  double Bm[4][6][3];
  for (int a = 0; a < 4; a++) {
    const double g0 = Nx[0][a], g1 = Nx[1][a], g2 = Nx[2][a];
    for (int c = 0; c < 3; c++) {
      Bm[a][0][c] = g0 * F[c][0];
      Bm[a][1][c] = g1 * F[c][1];
      Bm[a][2][c] = g2 * F[c][2];
      Bm[a][3][c] = g0 * F[c][1] + g1 * F[c][0];
      Bm[a][4][c] = g1 * F[c][2] + g2 * F[c][1];
      Bm[a][5][c] = g2 * F[c][0] + g0 * F[c][2];
    }
  }
  for (int b = 0; b < 4; b++) {
    double DBm[6][3];
    for (int I = 0; I < 6; I++) {
      for (int c = 0; c < 3; c++) {
        double s = 0.0;
        for (int K = 0; K < 6; K++) s += Dm[I][K] * Bm[b][K][c];
        DBm[I][c] = s;
      }
    }
    double SNx[3];
    for (int i = 0; i < 3; i++) SNx[i] = S[i][0] * Nx[0][b] + S[i][1] * Nx[1][b] + S[i][2] * Nx[2][b];
    for (int a = 0; a < 4; a++) {
      const double NxSNx = Nx[0][a] * SNx[0] + Nx[1][a] * SNx[1] + Nx[2][a] * SNx[2];
      const double T1 = amd * N[a] * N[b] + afu * NxSNx;
      for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
          double BDB = 0.0;
          for (int I = 0; I < 6; I++) BDB += Bm[a][I][r] * DBm[I][c];
          lK[4 * r + c][a][b] += w * (((r == c) ? T1 : 0.0) + afu * BDB);
        }
      }
    }
  }
}

struct MeshParams {
  double rho, E, nu, f[3], dt, af, am, beta;
};

// l_elas::l_elas_3d for the mesh motion equation (dof = 3, eq.s = 4)
__device__ void d_l_elas_3d(const MeshParams& p, const double w, const double* N, const double Nx[3][4],
    const double al[8][4], const double dl[8][4], const double bfl[3][4], double lR[3][4], double lK[9][4][4])
{
  const int i0 = 4;
  const double lambda = p.E * p.nu / (1.0 + p.nu) / (1.0 - 2.0 * p.nu);
  const double mu = p.E * 0.5 / (1.0 + p.nu);
  const double lDm = lambda / mu;
  const double T1 = p.af * p.beta * p.dt * p.dt;
  const double amd = p.am / T1 * p.rho;
  const double wl = w * T1 * mu;
  double ud[3] = {-p.f[0], -p.f[1], -p.f[2]}, ed[6] = {};
  for (int a = 0; a < 4; a++) {
    ud[0] = ud[0] + N[a] * (al[i0][a] - bfl[0][a]);
    ud[1] = ud[1] + N[a] * (al[i0 + 1][a] - bfl[1][a]);
    ud[2] = ud[2] + N[a] * (al[i0 + 2][a] - bfl[2][a]);
    ed[0] = ed[0] + Nx[0][a] * dl[i0][a];
    ed[1] = ed[1] + Nx[1][a] * dl[i0 + 1][a];
    ed[2] = ed[2] + Nx[2][a] * dl[i0 + 2][a];
    ed[3] = ed[3] + Nx[1][a] * dl[i0][a] + Nx[0][a] * dl[i0 + 1][a];
    ed[4] = ed[4] + Nx[2][a] * dl[i0 + 1][a] + Nx[1][a] * dl[i0 + 2][a];
    ed[5] = ed[5] + Nx[0][a] * dl[i0 + 2][a] + Nx[2][a] * dl[i0][a];
  }
  const double divD = lambda * (ed[0] + ed[1] + ed[2]);
  double S[6];
  S[0] = divD + 2.0 * mu * ed[0];
  S[1] = divD + 2.0 * mu * ed[1];
  S[2] = divD + 2.0 * mu * ed[2];
  S[3] = mu * ed[3];
  S[4] = mu * ed[4];
  S[5] = mu * ed[5];
  for (int a = 0; a < 4; a++) {
    lR[0][a] = lR[0][a] + w * (p.rho * N[a] * ud[0] + Nx[0][a] * S[0] + Nx[1][a] * S[3] + Nx[2][a] * S[5]);
    lR[1][a] = lR[1][a] + w * (p.rho * N[a] * ud[1] + Nx[0][a] * S[3] + Nx[1][a] * S[1] + Nx[2][a] * S[4]);
    lR[2][a] = lR[2][a] + w * (p.rho * N[a] * ud[2] + Nx[0][a] * S[5] + Nx[1][a] * S[4] + Nx[2][a] * S[2]);
    for (int b = 0; b < 4; b++) {
      const double NxdNx = Nx[0][a] * Nx[0][b] + Nx[1][a] * Nx[1][b] + Nx[2][a] * Nx[2][b];
      const double T = amd * N[a] * N[b] / mu + NxdNx;
      lK[0][a][b] = lK[0][a][b] + wl * (T + (1.0 + lDm) * Nx[0][a] * Nx[0][b]);
      lK[1][a][b] = lK[1][a][b] + wl * (lDm * Nx[0][a] * Nx[1][b] + Nx[1][a] * Nx[0][b]);
      lK[2][a][b] = lK[2][a][b] + wl * (lDm * Nx[0][a] * Nx[2][b] + Nx[2][a] * Nx[0][b]);
      lK[3][a][b] = lK[3][a][b] + wl * (lDm * Nx[1][a] * Nx[0][b] + Nx[0][a] * Nx[1][b]);
      lK[4][a][b] = lK[4][a][b] + wl * (T + (1.0 + lDm) * Nx[1][a] * Nx[1][b]);
      lK[5][a][b] = lK[5][a][b] + wl * (lDm * Nx[1][a] * Nx[2][b] + Nx[2][a] * Nx[1][b]);
      lK[6][a][b] = lK[6][a][b] + wl * (lDm * Nx[2][a] * Nx[0][b] + Nx[0][a] * Nx[2][b]);
      lK[7][a][b] = lK[7][a][b] + wl * (lDm * Nx[2][a] * Nx[1][b] + Nx[1][a] * Nx[2][b]);
      lK[8][a][b] = lK[8][a][b] + wl * (T + (1.0 + lDm) * Nx[2][a] * Nx[2][b]);
    }
  }
}

//===========================================================================
// Assembly kernels (one thread per element)

struct MeshDev {
  int nEl = 0;
  int *IEN = nullptr, *ePtr = nullptr;
  GaussData gd{};
};

struct State {
  int tnNo = 0, nnz = 0, tDof = 0;
  double *x = nullptr, *Ag = nullptr, *Yg = nullptr, *Dg = nullptr, *Do = nullptr, *Bf = nullptr;
  double *R = nullptr, *Val = nullptr, *hR = nullptr;
  int* dRow = nullptr;
  int* dCol = nullptr;
};

State st;
std::vector<MeshDev> meshes;

__device__ inline void load_element(int e, const int* IEN, const double* x, int tDof, const double* Ag,
    const double* Yg, const double* Dg, const double* Bf, double xl[3][4], double al[8][4], double yl[8][4],
    double dl[8][4], double bfl[3][4])
{
  for (int a = 0; a < 4; a++) {
    int Ac = IEN[4 * e + a];
    for (int i = 0; i < 3; i++) {
      xl[i][a] = x[3 * Ac + i];
      bfl[i][a] = Bf[3 * Ac + i];
    }
    for (int i = 0; i < tDof && i < 8; i++) {
      al[i][a] = Ag[(size_t)tDof * Ac + i];
      yl[i][a] = Yg[(size_t)tDof * Ac + i];
      dl[i][a] = Dg[(size_t)tDof * Ac + i];
    }
  }
}

template <int DOF>
__device__ inline void scatter(int e, const int* IEN, const int* ePtr, const double lRv[][4],
    const double lKv[][4][4], double* R, double* Val)
{
  for (int a = 0; a < 4; a++) {
    int rowN = IEN[4 * e + a];
    for (int i = 0; i < DOF; i++) atomicAdd(&R[DOF * (size_t)rowN + i], lRv[i][a]);
    for (int b = 0; b < 4; b++) {
      int p = ePtr[16 * e + 4 * a + b];
      for (int k = 0; k < DOF * DOF; k++) atomicAdd(&Val[DOF * DOF * (size_t)p + k], lKv[k][a][b]);
    }
  }
}

// FSI fluid domain: current configuration x + d(4..6), mesh velocity y(4..6)
__global__ void k_fsi_fluid(int nEl, const int* IEN, const int* ePtr, State s, FluidParams prm, GaussData gd)
{
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= nEl) return;
  double xl[3][4], al[8][4] = {}, yl[8][4] = {}, dl[8][4] = {}, bfl[3][4];
  load_element(e, IEN, s.x, s.tDof, s.Ag, s.Yg, s.Dg, s.Bf, xl, al, yl, dl, bfl);
  for (int i = 0; i < 3; i++) {
    for (int a = 0; a < 4; a++) xl[i][a] = xl[i][a] + dl[4 + i][a];
  }
  double lRv[4][4] = {};
  double lKv[16][4][4] = {};
  double Nx[3][4], ks[3][3], Jac;
  const double Nxx[6][4] = {};
  const double uris[3] = {};
  d_gnn(gd.Nxi, xl, Nx, Jac, ks);
  for (int g = 0; g < gd.nG; g++) {
    d_fluid_3d_m(prm, 1, 4, 4, gd.w[g] * Jac, ks, gd.N[g], gd.N[g], Nx, Nx, Nxx, al, yl, bfl, lRv, lKv, 0.0, 0.0, uris);
  }
  for (int g = 0; g < gd.nG; g++) {
    d_fluid_3d_c(prm, 1, 4, 4, gd.w[g] * Jac, ks, gd.N[g], gd.N[g], Nx, Nx, Nxx, al, yl, bfl, lRv, lKv, 0.0, 0.0, uris);
  }
  scatter<4>(e, IEN, ePtr, lRv, lKv, s.R, s.Val);
}

// FSI struct domain (reference configuration)
__global__ void k_fsi_struct(int nEl, const int* IEN, const int* ePtr, State s, StructParams prm, GaussData gd)
{
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= nEl) return;
  double xl[3][4], al[8][4] = {}, yl[8][4] = {}, dl[8][4] = {}, bfl[3][4];
  load_element(e, IEN, s.x, s.tDof, s.Ag, s.Yg, s.Dg, s.Bf, xl, al, yl, dl, bfl);
  double lRv[4][4] = {};
  double lKv[16][4][4] = {};
  double Nx[3][4], ks[3][3], Jac;
  d_gnn(gd.Nxi, xl, Nx, Jac, ks);
  for (int g = 0; g < gd.nG; g++) {
    d_struct_3d(prm, gd.w[g] * Jac, gd.N[g], Nx, al, yl, dl, bfl, lRv, lKv);
  }
  scatter<4>(e, IEN, ePtr, lRv, lKv, s.R, s.Val);
}

// Mesh motion: reference configuration x + Do(4..6), displacement d - Do
__global__ void k_mesh(int nEl, const int* IEN, const int* ePtr, State s, MeshParams prm, GaussData gd)
{
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= nEl) return;
  double xl[3][4], al[8][4] = {}, yl[8][4] = {}, dl[8][4] = {}, bfl[3][4];
  load_element(e, IEN, s.x, s.tDof, s.Ag, s.Yg, s.Dg, s.Bf, xl, al, yl, dl, bfl);
  for (int a = 0; a < 4; a++) {
    int Ac = IEN[4 * e + a];
    for (int i = 0; i < 3; i++) {
      double dol = s.Do[(size_t)s.tDof * Ac + 4 + i];
      xl[i][a] = xl[i][a] + dol;
      dl[4 + i][a] = dl[4 + i][a] - dol;
      bfl[i][a] = 0.0;
    }
  }
  double lRv[3][4] = {};
  double lKv[9][4][4] = {};
  double Nx[3][4], ks[3][3], Jac;
  d_gnn(gd.Nxi, xl, Nx, Jac, ks);
  for (int g = 0; g < gd.nG; g++) {
    d_l_elas_3d(prm, gd.w[g], gd.N[g], Nx, al, dl, bfl, lRv, lKv);
  }
  scatter<3>(e, IEN, ePtr, lRv, lKv, s.R, s.Val);
}

__global__ void k_elem_ptr(int nEl, const int* IEN, const int* rowPtr, const int* colPtr, int* ePtr)
{
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= nEl) return;
  for (int a = 0; a < 4; a++) {
    int rowN = IEN[4 * e + a];
    for (int b = 0; b < 4; b++) {
      int colN = IEN[4 * e + b];
      int left = rowPtr[rowN], right = rowPtr[rowN + 1];
      int ptr = (right + left) / 2;
      while (colN != colPtr[ptr]) {
        if (colN > colPtr[ptr]) left = ptr; else right = ptr;
        ptr = (right + left) / 2;
      }
      ePtr[16 * e + 4 * a + b] = ptr;
    }
  }
}

template <typename T> T* dev_upload(const T* h, size_t n)
{
  T* d;
  CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(n, 1) * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

//===========================================================================
// Linear algebra on the device resident matrix (FSILS lhs ordering)

struct Lhs {
  int nNo = 0, nnz = 0;
  int *rowS = nullptr, *rowE = nullptr, *col = nullptr, *rowOf = nullptr, *diag = nullptr;
  double *W = nullptr, *Dmul = nullptr, *stageVal = nullptr;
  int* stagePtr = nullptr;
  int stageCap = 0;
  // Krylov basis and work vectors (dof * nNo each)
  double *u = nullptr, *X = nullptr, *R = nullptr, *P = nullptr, *Q = nullptr, *acc = nullptr, *hDev = nullptr;
  int sDcap = 0;
  int nbCoop = 0;
};
Lhs lhs;

void lhs_setup(int nNo, int nnz, const int* rowPtr, const int* colPtr, const int* diagPtr)
{
  if (lhs.nNo == nNo && lhs.nnz == nnz) return;
  lhs.nNo = nNo;
  lhs.nnz = nnz;
  std::vector<int> rs(nNo), re(nNo), rowOf(nnz);
  for (int a = 0; a < nNo; a++) {
    rs[a] = rowPtr[2 * a];
    re[a] = rowPtr[2 * a + 1];
    for (int j = rs[a]; j <= re[a]; j++) rowOf[j] = a;
  }
  lhs.rowS = dev_upload(rs.data(), nNo);
  lhs.rowE = dev_upload(re.data(), nNo);
  lhs.col = dev_upload(colPtr, nnz);
  lhs.rowOf = dev_upload(rowOf.data(), nnz);
  lhs.diag = dev_upload(diagPtr, nNo);
  CUDA_CHECK(cudaMalloc(&lhs.W, 4 * (size_t)nNo * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lhs.Dmul, 4 * (size_t)nNo * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lhs.X, 4 * (size_t)nNo * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lhs.R, 4 * (size_t)nNo * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lhs.P, 4 * (size_t)nNo * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lhs.Q, 4 * (size_t)nNo * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lhs.acc, 4 * 64 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lhs.hDev, 4096 * sizeof(double)));
}

__global__ void k_add_val(int n, int dd, const int* ptr, const double* vals, double* Val)
{
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= dd * n) return;
  int i = q / dd, k = q % dd;
  atomicAdd(&Val[dd * (size_t)ptr[i] + k], vals[q]);
}

__global__ void k_precond_w(int nNo, int dof, const int* diag, const double* Val, const double* Dmul, double* W)
{
  int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= nNo) return;
  for (int i = 0; i < dof; i++) {
    double w = Val[(size_t)dof * dof * diag[a] + (dof + 1) * i];
    if (w == 0.0) w = 1.0;
    w = 1.0 / sqrt(fabs(w));
    W[dof * a + i] = w * Dmul[dof * a + i];
  }
}

__global__ void k_precond_scale(int nnz, int dof, const int* rowOf, const int* col, const double* W, double* Val)
{
  size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int dd = dof * dof;
  if (q >= (size_t)dd * nnz) return;
  int j = (int)(q / dd), k = (int)(q % dd);
  int r = k / dof, c = k % dof;
  Val[q] = Val[q] * W[dof * rowOf[j] + r] * W[dof * col[j] + c];
}

__device__ inline double block_sum(double v)
{
  __shared__ double sh[TPB / 32];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  double x = 0.0;
  for (int w = 0; w < TPB / 32; w++) x += sh[w];
  __syncthreads();
  return x;
}

// y_i = sum_j K_j x_col(j) for the dof x dof blocks (LANES threads per row)
template <int DOF, int LANES>
__device__ inline void spmv_rows(int nNo, const int* rowS, const int* rowE, const int* col, const double* K,
    const double* x, double* y, int tid, int nth)
{
  const int lane = tid % LANES;
  const int ngroups = nth / LANES;
  const int nIt = ((nNo + ngroups - 1) / ngroups) * ngroups;
  for (int i = tid / LANES; i < nIt; i += ngroups) {
    double s[DOF] = {};
    if (i < nNo) {
      for (int j = rowS[i] + lane; j <= rowE[i]; j += LANES) {
        const double* k = K + DOF * DOF * (size_t)j;
        const double* xc = x + DOF * (size_t)col[j];
        double xv[DOF];
        for (int c = 0; c < DOF; c++) xv[c] = xc[c];
        for (int r = 0; r < DOF; r++) {
          double t = 0.0;
          for (int c = 0; c < DOF; c++) t += k[DOF * r + c] * xv[c];
          s[r] += t;
        }
      }
    }
    for (int r = 0; r < DOF; r++) {
      for (int o = LANES / 2; o > 0; o >>= 1) s[r] += __shfl_down_sync(0xffffffff, s[r], o, LANES);
    }
    if (lane == 0 && i < nNo) {
      for (int r = 0; r < DOF; r++) y[DOF * (size_t)i + r] = s[r];
    }
  }
}

// Accumulator slots (rotating, see k_cg_amg2)
#define NEXT_ACC() (accp = a.acc + (slot % 4) * 64, slot++, \
    (blockIdx.x == 0 && threadIdx.x < 64) ? (a.acc[(slot % 4) * 64 + threadIdx.x] = 0.0) : 0.0, accp)

struct ArnArgs {
  int nNo, i;
  const int *rowS, *rowE, *col;
  const double* K;
  double *u, *h, *acc;
};

// One Arnoldi step: w = K u_i, modified Gram-Schmidt, normalization
template <int DOF>
__global__ void k_arnoldi(ArnArgs a)
{
  cg::grid_group grid = cg::this_grid();
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nth = gridDim.x * blockDim.x;
  const size_t n = DOF * (size_t)a.nNo;
  const double* ui = a.u + a.i * n;
  double* w = a.u + (a.i + 1) * n;
  int slot = 0;
  double* accp;

  NEXT_ACC();
  spmv_rows<DOF, 8>(a.nNo, a.rowS, a.rowE, a.col, a.K, ui, w, tid, nth);
  grid.sync();
  {
    double s = 0.0;
    for (size_t q = tid; q < n; q += nth) s += a.u[q] * w[q];
    s = block_sum(s);
    if (threadIdx.x == 0) atomicAdd(&accp[0], s);
  }
  grid.sync();
  for (int j = 0; j <= a.i; j++) {
    double hj = __ldcg(&accp[0]);
    if (tid == 0) a.h[j] = hj;
    const double* uj = a.u + j * n;
    const double* un = (j < a.i) ? a.u + (j + 1) * n : w;
    NEXT_ACC();
    double s = 0.0;
    for (size_t q = tid; q < n; q += nth) {
      double wq = w[q] - hj * uj[q];
      w[q] = wq;
      s += un[q] * wq;
    }
    s = block_sum(s);
    if (threadIdx.x == 0) atomicAdd(&accp[0], s);
    grid.sync();
  }
  double nrm = sqrt(__ldcg(&accp[0]));
  if (tid == 0) a.h[a.i + 1] = nrm;
  for (size_t q = tid; q < n; q += nth) w[q] = w[q] * (1.0 / nrm);
}

// u0 = R - K X, and |u0|^2 into acc
template <int DOF>
__global__ void k_residual(int nNo, const int* rowS, const int* rowE, const int* col, const double* K,
    const double* R, const double* X, double* u0)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nth = gridDim.x * blockDim.x;
  spmv_rows<DOF, 8>(nNo, rowS, rowE, col, K, X, u0, tid, nth);
}

__global__ void k_sub_vec(size_t n, const double* a, double* b)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] = a[i] - b[i];
}

__global__ void k_dot2(size_t n, const double* a, const double* b, double* acc)
{
  double s = 0.0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    s += a[i] * b[i];
  }
  s = block_sum(s);
  if (threadIdx.x == 0) atomicAdd(acc, s);
}

__global__ void k_scale_vec(size_t n, double a, double* x)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = a * x[i];
}

struct PtrList { const double* p[64]; double c[64]; };

// X = X + sum_j c_j u_j (in order)
__global__ void k_update_x(size_t n, int m, PtrList pl, double* X)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double v = X[i];
  for (int j = 0; j < m; j++) v = v + pl.c[j] * pl.p[j][i];
  X[i] = v;
}

struct CgArgs {
  int nNo, mItr;
  const int *rowS, *rowE, *col;
  const double* K;
  double *X, *R, *P, *Q, *acc;
  double err, eps;
  double* out;
};

// cgrad_v: all iterations in one cooperative kernel
template <int DOF>
__global__ void k_cg(CgArgs a)
{
  cg::grid_group grid = cg::this_grid();
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nth = gridDim.x * blockDim.x;
  const size_t n = DOF * (size_t)a.nNo;
  double err = a.err, errO = err;
  int last_i = 0, success = 0;
  int slot = 0;
  double* accp;
  for (int it = 0; it < a.mItr; it++) {
    last_i = it;
    if (err < a.eps) { success = 1; break; }
    errO = err;
    NEXT_ACC();
    spmv_rows<DOF, 8>(a.nNo, a.rowS, a.rowE, a.col, a.K, a.P, a.Q, tid, nth);
    grid.sync();
    double s = 0.0;
    for (size_t q = tid; q < n; q += nth) s += a.P[q] * a.Q[q];
    s = block_sum(s);
    if (threadIdx.x == 0) atomicAdd(&accp[0], s);
    grid.sync();
    const double alpha = errO / __ldcg(&accp[0]);
    NEXT_ACC();
    double rr = 0.0;
    for (size_t q = tid; q < n; q += nth) {
      a.X[q] = a.X[q] + alpha * a.P[q];
      double r = a.R[q] + (-alpha) * a.Q[q];
      a.R[q] = r;
      rr += r * r;
    }
    rr = block_sum(rr);
    if (threadIdx.x == 0) atomicAdd(&accp[0], rr);
    grid.sync();
    err = __ldcg(&accp[0]);
    const double c1 = errO / err, c2 = err / errO;
    for (size_t q = tid; q < n; q += nth) {
      double t = a.P[q] + c1 * a.R[q];
      a.P[q] = c2 * t;
    }
    grid.sync();
  }
  if (tid == 0) {
    a.out[0] = err;
    a.out[1] = errO;
    a.out[2] = last_i;
    a.out[3] = success;
  }
}

template <typename Kern> int coop_blocks(Kern k)
{
  int dev, nSM, perSM;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, dev));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, k, TPB, 0));
  int maxPer = getenv("SVGPU_PERSM") ? atoi(getenv("SVGPU_PERSM")) : 2;
  return nSM * std::max(1, std::min(perSM, maxPer));
}

double dot_dev(const double* a, const double* b, size_t n)
{
  CUDA_CHECK(cudaMemsetAsync(lhs.acc, 0, sizeof(double)));
  k_dot2<<<160, TPB>>>(n, a, b, lhs.acc);
  CUDA_CHECK(cudaGetLastError());
  double r;
  CUDA_CHECK(cudaMemcpy(&r, lhs.acc, sizeof(double), cudaMemcpyDeviceToHost));
  return r;
}

double now()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// gmres::gmres_v without coupled faces. R (device, dof*nNo) in, X (device) out
template <int DOF>
void gmres_v(int mItr, int sD, double absTol, double relTol, svgpu_stats* stats)
{
  const int nNo = lhs.nNo;
  const size_t n = DOF * (size_t)nNo;
  if (sD + 1 > lhs.sDcap) {
    if (lhs.u) cudaFree(lhs.u);
    CUDA_CHECK(cudaMalloc(&lhs.u, n * (sD + 1) * sizeof(double)));
    lhs.sDcap = sD + 1;
  }
  static int nbArn3 = 0, nbArn4 = 0;
  int& nbArn = (DOF == 3) ? nbArn3 : nbArn4;
  if (!nbArn) nbArn = coop_blocks(k_arnoldi<DOF>);
  double* u = lhs.u;
  auto uj = [&](int j) { return u + (size_t)j * n; };
  const double* K = svgpu_dev_val_ptr;
  const int nbV = (int)((n + TPB - 1) / TPB);

  double eps = std::sqrt(dot_dev(lhs.R, lhs.R, n));
  double iNorm = eps, fNorm = eps, dB = 0.0;
  eps = std::max(absTol, relTol * eps);
  int itr = 0, last_i = 0;
  bool success = false;
  CUDA_CHECK(cudaMemsetAsync(lhs.X, 0, n * sizeof(double)));
  if (iNorm <= absTol) {
    success = true;
  } else {
    std::vector<double> h((size_t)(sD + 1) * sD), y(sD), c(sD), s(sD), err(sD + 1);
    auto H = [&](int i, int j) -> double& { return h[i + (size_t)j * (sD + 1)]; };
    for (int l = 0; l < mItr; l++) {
      dB = fNorm;
      itr++;
      k_residual<DOF><<<nbV / 8 + 1, TPB>>>(nNo, lhs.rowS, lhs.rowE, lhs.col, K, lhs.R, lhs.X, uj(0));
      k_sub_vec<<<nbV, TPB>>>(n, lhs.R, uj(0));
      CUDA_CHECK(cudaGetLastError());
      err[0] = std::sqrt(dot_dev(uj(0), uj(0), n));
      if (err[0] == 0.0) throw std::runtime_error("svgpu: a zero matrix norm was computed in GMRES");
      k_scale_vec<<<nbV, TPB>>>(n, 1.0 / err[0], uj(0));
      for (int i = 0; i < sD; i++) {
        itr++;
        last_i = i;
        CUDA_CHECK(cudaMemsetAsync(lhs.acc, 0, 64 * sizeof(double)));
        ArnArgs a{nNo, i, lhs.rowS, lhs.rowE, lhs.col, K, u, lhs.hDev, lhs.acc};
        void* args[] = {&a};
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_arnoldi<DOF>, nbArn, TPB, args, 0, 0));
        std::vector<double> hc(i + 2);
        CUDA_CHECK(cudaMemcpy(hc.data(), lhs.hDev, (i + 2) * sizeof(double), cudaMemcpyDeviceToHost));
        for (int j = 0; j <= i + 1; j++) H(j, i) = hc[j];
        for (int j = 0; j <= i - 1; j++) {
          double tmp = c[j] * H(j, i) + s[j] * H(j + 1, i);
          H(j + 1, i) = -s[j] * H(j, i) + c[j] * H(j + 1, i);
          H(j, i) = tmp;
        }
        double tmp = std::sqrt(H(i, i) * H(i, i) + H(i + 1, i) * H(i + 1, i));
        c[i] = H(i, i) / tmp;
        s[i] = H(i + 1, i) / tmp;
        H(i, i) = tmp;
        H(i + 1, i) = 0.0;
        err[i + 1] = -s[i] * err[i];
        err[i] = c[i] * err[i];
        if (std::fabs(err[i + 1]) < eps) {
          success = true;
          break;
        }
      }
      if (last_i >= sD) last_i = sD - 1;
      for (int i = 0; i <= last_i; i++) y[i] = err[i];
      for (int j = last_i; j >= 0; j--) {
        for (int k = j + 1; k <= last_i; k++) y[j] = y[j] - H(j, k) * y[k];
        y[j] = y[j] / H(j, j);
      }
      for (int j0 = 0; j0 <= last_i; j0 += 64) {
        PtrList pl;
        int m = std::min(64, last_i + 1 - j0);
        for (int j = 0; j < m; j++) { pl.p[j] = uj(j0 + j); pl.c[j] = y[j0 + j]; }
        k_update_x<<<nbV, TPB>>>(n, m, pl, lhs.X);
      }
      CUDA_CHECK(cudaGetLastError());
      fNorm = std::fabs(err[last_i + 1]);
      if (success) break;
    }
  }
  if (stats) {
    stats->GM_itr = itr;
    stats->success = success ? 1 : 0;
    stats->iNorm = iNorm;
    stats->fNorm = fNorm;
    stats->dB = (dB > 0.0 && fNorm > 0.0) ? 10.0 * std::log(fNorm / dB) : 0.0;
  }
}

// cgrad::cgrad_v. R (device) in, X (device) out
template <int DOF>
void cgrad_v(int mItr, double absTol, double relTol, svgpu_stats* stats)
{
  const int nNo = lhs.nNo;
  const size_t n = DOF * (size_t)nNo;
  static int nb3 = 0, nb4 = 0;
  int& nb = (DOF == 3) ? nb3 : nb4;
  if (!nb) nb = coop_blocks(k_cg<DOF>);
  double iNorm = std::sqrt(dot_dev(lhs.R, lhs.R, n));
  double eps = std::pow(std::max(absTol, relTol * iNorm), 2.0);
  CUDA_CHECK(cudaMemsetAsync(lhs.X, 0, n * sizeof(double)));
  CUDA_CHECK(cudaMemcpyAsync(lhs.P, lhs.R, n * sizeof(double), cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemsetAsync(lhs.acc, 0, 4 * 64 * sizeof(double)));
  CgArgs a{nNo, mItr, lhs.rowS, lhs.rowE, lhs.col, svgpu_dev_val_ptr, lhs.X, lhs.R, lhs.P, lhs.Q, lhs.acc,
      iNorm * iNorm, eps, lhs.hDev};
  void* args[] = {&a};
  CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_cg<DOF>, nb, TPB, args, 0, 0));
  double h[4];
  CUDA_CHECK(cudaMemcpy(h, lhs.hDev, sizeof(h), cudaMemcpyDeviceToHost));
  if (stats) {
    stats->CG_itr = (int)h[2];
    stats->success = h[3] != 0.0;
    stats->iNorm = iNorm;
    stats->fNorm = std::sqrt(h[0]);
    stats->dB = (h[1] < std::numeric_limits<double>::epsilon()) ? 0.0 : 5.0 * std::log(h[0] / h[1]);
  }
}

}  // namespace

//===========================================================================
// C interface

extern "C" SVGPU_API int svgpu_fsi_setup(int tnNo, const double* x, int nnz, const int* rowPtr, const int* colPtr)
{
  try {
    st.tnNo = tnNo;
    st.nnz = nnz;
    st.x = dev_upload(x, 3 * (size_t)tnNo);
    st.dRow = dev_upload(rowPtr, (size_t)tnNo + 1);
    st.dCol = dev_upload(colPtr, (size_t)nnz);
    CUDA_CHECK(cudaMalloc(&st.R, 4 * (size_t)tnNo * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&st.Val, 16 * (size_t)nnz * sizeof(double)));
    CUDA_CHECK(cudaMemset(st.Val, 0, 16 * (size_t)nnz * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&st.hR, 4 * (size_t)tnNo * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&st.Bf, 3 * (size_t)tnNo * sizeof(double)));
    svgpu_dev_val_ptr = st.Val;
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "svgpu_fsi_setup: %s\n", e.what());
    return 1;
  }
}

extern "C" SVGPU_API int svgpu_fsi_add_mesh(int nEl, const int* IEN, int nG, const double* w, const double* N,
    const double* Nxi)
{
  try {
    if (nG > 4) throw std::runtime_error("svgpu: at most 4 Gauss points are supported");
    MeshDev m;
    m.nEl = nEl;
    m.IEN = dev_upload(IEN, 4 * (size_t)nEl);
    CUDA_CHECK(cudaMalloc(&m.ePtr, 16 * (size_t)nEl * sizeof(int)));
    k_elem_ptr<<<(nEl + 127) / 128, 128>>>(nEl, m.IEN, st.dRow, st.dCol, m.ePtr);
    CUDA_CHECK(cudaGetLastError());
    m.gd.nG = nG;
    for (int g = 0; g < nG; g++) {
      m.gd.w[g] = w[g];
      for (int a = 0; a < 4; a++) m.gd.N[g][a] = N[a + 4 * g];
    }
    for (int a = 0; a < 4; a++) {
      for (int i = 0; i < 3; i++) m.gd.Nxi[i][a] = Nxi[i + 3 * a];
    }
    meshes.push_back(m);
    return (int)meshes.size() - 1;
  } catch (const std::exception& e) {
    fprintf(stderr, "svgpu_fsi_add_mesh: %s\n", e.what());
    return -1;
  }
}

/// Assemble one mesh into the device matrix and add the residual to R
/// (dof x tnNo). type: 0 = FSI fluid (ALE), 1 = FSI struct, 2 = mesh motion.
/// params: fluid {rho, mu, fx, fy, fz, dt, af, am, gam}, struct {rho, dmp,
/// fx, fy, fz, C10, Kpen, dt, af, am, beta, gam}, mesh {rho, E, nu, fx, fy,
/// fz, dt, af, am, beta}
extern "C" SVGPU_API int svgpu_fsi_assemble(int mesh, int type, int tDof, const double* Ag, const double* Yg,
    const double* Dg, const double* Do, const double* Bf, const double* params, double* R)
{
  try {
    if (mesh < 0 || mesh >= (int)meshes.size()) throw std::runtime_error("svgpu: unknown mesh");
    if (tDof > 8) throw std::runtime_error("svgpu: tDof > 8 is not supported");
    const size_t nA = (size_t)tDof * st.tnNo;
    if (st.tDof != tDof) {
      if (st.Ag) { cudaFree(st.Ag); cudaFree(st.Yg); cudaFree(st.Dg); cudaFree(st.Do); }
      CUDA_CHECK(cudaMalloc(&st.Ag, nA * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&st.Yg, nA * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&st.Dg, nA * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&st.Do, nA * sizeof(double)));
      st.tDof = tDof;
    }
    CUDA_CHECK(cudaMemcpy(st.Ag, Ag, nA * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(st.Yg, Yg, nA * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(st.Dg, Dg, nA * sizeof(double), cudaMemcpyHostToDevice));
    if (Do) CUDA_CHECK(cudaMemcpy(st.Do, Do, nA * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(st.Bf, Bf, 3 * (size_t)st.tnNo * sizeof(double), cudaMemcpyHostToDevice));
    const int dof = (type == 2) ? 3 : 4;
    CUDA_CHECK(cudaMemset(st.R, 0, dof * (size_t)st.tnNo * sizeof(double)));
    const MeshDev& m = meshes[mesh];
    const int nb = (m.nEl + 127) / 128;
    const double* p = params;
    if (type == 0) {
      FluidParams prm{p[0], p[1], {p[2], p[3], p[4]}, p[5], p[6], p[7], p[8], 0.0, 1};
      k_fsi_fluid<<<nb, 128>>>(m.nEl, m.IEN, m.ePtr, st, prm, m.gd);
    } else if (type == 1) {
      StructParams prm{p[0], p[1], {p[2], p[3], p[4]}, p[5], p[6], p[7], p[8], p[9], p[10], p[11]};
      k_fsi_struct<<<nb, 128>>>(m.nEl, m.IEN, m.ePtr, st, prm, m.gd);
    } else {
      MeshParams prm{p[0], p[1], p[2], {p[3], p[4], p[5]}, p[6], p[7], p[8], p[9]};
      k_mesh<<<nb, 128>>>(m.nEl, m.IEN, m.ePtr, st, prm, m.gd);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(st.hR, st.R, dof * (size_t)st.tnNo * sizeof(double), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < dof * (size_t)st.tnNo; i++) R[i] += st.hR[i];
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "svgpu_fsi_assemble: %s\n", e.what());
    return 1;
  }
}

/// Add staged matrix entries (dof*dof values each) to the device matrix
extern "C" SVGPU_API int svgpu_add_val_dof(int n, int dof, const int* ptr, const double* vals)
{
  try {
    if (n == 0) return 0;
    const int dd = dof * dof;
    if (n > lhs.stageCap) {
      if (lhs.stagePtr) { cudaFree(lhs.stagePtr); cudaFree(lhs.stageVal); }
      lhs.stageCap = 2 * n;
      CUDA_CHECK(cudaMalloc(&lhs.stagePtr, lhs.stageCap * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&lhs.stageVal, 16 * (size_t)lhs.stageCap * sizeof(double)));
    }
    CUDA_CHECK(cudaMemcpy(lhs.stagePtr, ptr, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lhs.stageVal, vals, dd * (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    k_add_val<<<(dd * n + 255) / 256, 256>>>(n, dd, lhs.stagePtr, lhs.stageVal, svgpu_dev_val_ptr);
    CUDA_CHECK(cudaGetLastError());
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "svgpu_add_val_dof: %s\n", e.what());
    return 1;
  }
}

/// FSILS diagonal preconditioner of the device matrix (dof = 3 or 4)
extern "C" SVGPU_API int svgpu_precond_diag_dof(int nNo, int nnz, int dof, const int* rowPtr, const int* colPtr,
    const int* diagPtr, const double* Dmul, double* W)
{
  try {
    lhs_setup(nNo, nnz, rowPtr, colPtr, diagPtr);
    CUDA_CHECK(cudaMemcpy(lhs.Dmul, Dmul, dof * (size_t)nNo * sizeof(double), cudaMemcpyHostToDevice));
    k_precond_w<<<(nNo + 255) / 256, 256>>>(nNo, dof, lhs.diag, svgpu_dev_val_ptr, lhs.Dmul, lhs.W);
    CUDA_CHECK(cudaGetLastError());
    const size_t nq = (size_t)dof * dof * nnz;
    k_precond_scale<<<(unsigned)((nq + 255) / 256), 256>>>(nnz, dof, lhs.rowOf, lhs.col, lhs.W, svgpu_dev_val_ptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(W, lhs.W, dof * (size_t)nNo * sizeof(double), cudaMemcpyDeviceToHost));
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "svgpu_precond_diag_dof: %s\n", e.what());
    return 1;
  }
}

/// FSILS GMRES (type 0) or CG (type 1) on the device matrix. R: dof x nNo
/// right-hand side in, solution out. The device matrix is cleared afterwards.
extern "C" SVGPU_API int svgpu_ls_solve(int type, int nNo, int nnz, int dof, int mItr, int sD, double absTol,
    double relTol, double* R, svgpu_stats* stats)
{
  try {
    double t0 = now();
    const size_t n = dof * (size_t)nNo;
    CUDA_CHECK(cudaMemcpy(lhs.R, R, n * sizeof(double), cudaMemcpyHostToDevice));
    if (type == 0) {
      if (dof == 4) gmres_v<4>(mItr, sD, absTol, relTol, stats);
      else if (dof == 3) gmres_v<3>(mItr, sD, absTol, relTol, stats);
      else throw std::runtime_error("svgpu: GMRES only for dof = 3 or 4");
    } else {
      if (dof == 3) cgrad_v<3>(mItr, absTol, relTol, stats);
      else if (dof == 4) cgrad_v<4>(mItr, absTol, relTol, stats);
      else throw std::runtime_error("svgpu: CG only for dof = 3 or 4");
    }
    CUDA_CHECK(cudaMemcpy(R, lhs.X, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(svgpu_dev_val_ptr, 0, (size_t)dof * dof * nnz * sizeof(double)));
    if (stats) stats->time = now() - t0;
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "svgpu_ls_solve: %s\n", e.what());
    return 1;
  }
}
