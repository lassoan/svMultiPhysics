// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// GPU assembly and linear solvers for FSI (fsi.cpp, mesh.cpp) as mfem::forall
// kernels:
//   - FSI equation (dof = 4): fluid domain in the current configuration with
//     the mesh velocity (ALE), and struct domain (neo-Hookean with the M94
//     volumetric penalty), on linear tetrahedra
//   - mesh motion equation (dof = 3): linear elasticity (l_elas_3d)
//   - FSILS GMRES (gmres_v) and CG (cgrad_v) for dof = 3 and 4 and the FSILS
//     diagonal preconditioner on the device resident matrix. The Krylov
//     solvers are preconditioned with an aggregation AMG V-cycle on the
//     dof x dof block matrix (right preconditioner of GMRES; SVMFEM_AMG=0
//     selects the plain FSILS algorithms, with the iteration counts of the
//     CPU solver).
//
// The block matrices of the products are in the SELL layout of
// svmfem_common.hpp with the values stored entry-major (value e of slot q at
// e * total + q), so that one thread per row reads coalesced memory. As in
// the NS solver, the phases are kernel sequences captured in CUDA graphs,
// the scalars live on the device and reach the host through mapped memory.

#include "svmfem.h"
#include "svmfem_amg.hpp"
#include "svmfem_internal.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>

namespace svmfem {

/// Pseudo-inverse of a dense row-major matrix (svmfem_lstsq.cpp)
void pinv(int n, const double* A, float* Ainv);

namespace {

//===========================================================================
// Element routines

struct FluidParams {
  double rho, mu, f[3], dt, af, am, gam, brink;
  int mvMsh;
};

MFEM_HOST_DEVICE inline bool is_zero(double value1, double value2 = 0.0)
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
MFEM_HOST_DEVICE void d_gnn(const double Nxi[3][4], const double x[3][4], double Nx[3][4], double& Jac,
    double ks[3][3])
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
MFEM_HOST_DEVICE void d_struct_3d(const StructParams& p, const double w, const double* N, const double Nx[3][4],
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
MFEM_HOST_DEVICE void d_l_elas_3d(const MeshParams& p, const double w, const double* N, const double Nx[3][4],
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

MFEM_HOST_DEVICE inline void load_element(int e, const int* IEN, const double* x, int tDof, const double* Ag,
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
MFEM_HOST_DEVICE inline void scatter(int e, const int* IEN, const int* ePtr, const double lRv[][4],
    const double lKv[][4][4], double* R, double* Val)
{
  for (int a = 0; a < 4; a++) {
    int rowN = IEN[4 * e + a];
    for (int i = 0; i < DOF; i++) AtomicAdd(R[DOF * (size_t)rowN + i], lRv[i][a]);
    for (int b = 0; b < 4; b++) {
      int p = ePtr[16 * e + 4 * a + b];
      for (int k = 0; k < DOF * DOF; k++) AtomicAdd(Val[DOF * DOF * (size_t)p + k], lKv[k][a][b]);
    }
  }
}

int env_int(const char* name, int def)
{
  const char* v = getenv(name);
  return v ? atoi(v) : def;
}

double env_double(const char* name, double def)
{
  const char* v = getenv(name);
  return v ? atof(v) : def;
}

// AMG preconditioner of the FSI solvers unless SVMFEM_AMG=0
bool amg_enabled()
{
  const char* e = getenv("SVMFEM_AMG");
  return !(e && atoi(e) == 0);
}

}  // namespace

// Named namespace: nvcc on Windows requires external linkage for the
// functions that define the device lambdas
namespace fsi {

struct MeshDev {
  int nEl = 0;
  Array<int> IEN, ePtr;   // 4 x nEl, 16 x nEl
  GaussData gd{};
};

/// Mesh and state of the assembly, shared by the solvers
struct FsiState {
  int tnNo = 0, nnz = 0, tDof = 0;
  Vector x, Ag, Yg, Dg, Do, Bf, R;
  std::unique_ptr<Mapped> hAg, hYg, hDg, hDo, hBf, hR;
  std::vector<std::unique_ptr<MeshDev>> meshes;
  Sell sell;                     // SELL layout of the node pattern
  Array<int>* stage_ptr = nullptr;
  Vector* stage_val = nullptr;
  int stage_cap = 0;
  double t_assem = 0.0;
  int n_assem = 0;
  ~FsiState()
  {
    delete stage_ptr;
    delete stage_val;
  }
};

FsiState* st = nullptr;

//===========================================================================
// Assembly kernels (one thread per element)

// FSI fluid domain: current configuration x + d(4..6), mesh velocity y(4..6)
void assemble_fluid(const MeshDev& m, const FluidParams& prm)
{
  const GaussData gd = m.gd;
  const int* IEN = m.IEN.Read();
  const int* ePtr = m.ePtr.Read();
  const int tDof = st->tDof;
  const double* x = st->x.Read();
  const double* Ag = st->Ag.Read();
  const double* Yg = st->Yg.Read();
  const double* Dg = st->Dg.Read();
  const double* Bf = st->Bf.Read();
  double* R = st->R.ReadWrite();
  double* Val = lhs()->Val.ReadWrite();
  launch(m.nEl, [=] MFEM_HOST_DEVICE (int e) {
    double xl[3][4], al[8][4] = {}, yl[8][4] = {}, dl[8][4] = {}, bfl[3][4];
    load_element(e, IEN, x, tDof, Ag, Yg, Dg, Bf, xl, al, yl, dl, bfl);
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
    scatter<4>(e, IEN, ePtr, lRv, lKv, R, Val);
  });
}

// FSI struct domain (reference configuration)
void assemble_struct(const MeshDev& m, const StructParams& prm)
{
  const GaussData gd = m.gd;
  const int* IEN = m.IEN.Read();
  const int* ePtr = m.ePtr.Read();
  const int tDof = st->tDof;
  const double* x = st->x.Read();
  const double* Ag = st->Ag.Read();
  const double* Yg = st->Yg.Read();
  const double* Dg = st->Dg.Read();
  const double* Bf = st->Bf.Read();
  double* R = st->R.ReadWrite();
  double* Val = lhs()->Val.ReadWrite();
  launch(m.nEl, [=] MFEM_HOST_DEVICE (int e) {
    double xl[3][4], al[8][4] = {}, yl[8][4] = {}, dl[8][4] = {}, bfl[3][4];
    load_element(e, IEN, x, tDof, Ag, Yg, Dg, Bf, xl, al, yl, dl, bfl);
    double lRv[4][4] = {};
    double lKv[16][4][4] = {};
    double Nx[3][4], ks[3][3], Jac;
    d_gnn(gd.Nxi, xl, Nx, Jac, ks);
    for (int g = 0; g < gd.nG; g++) {
      d_struct_3d(prm, gd.w[g] * Jac, gd.N[g], Nx, al, yl, dl, bfl, lRv, lKv);
    }
    scatter<4>(e, IEN, ePtr, lRv, lKv, R, Val);
  });
}

// Mesh motion: reference configuration x + Do(4..6), displacement d - Do
void assemble_mesh(const MeshDev& m, const MeshParams& prm)
{
  const GaussData gd = m.gd;
  const int* IEN = m.IEN.Read();
  const int* ePtr = m.ePtr.Read();
  const int tDof = st->tDof;
  const double* x = st->x.Read();
  const double* Ag = st->Ag.Read();
  const double* Yg = st->Yg.Read();
  const double* Dg = st->Dg.Read();
  const double* Do = st->Do.Read();
  const double* Bf = st->Bf.Read();
  double* R = st->R.ReadWrite();
  double* Val = lhs()->Val.ReadWrite();
  launch(m.nEl, [=] MFEM_HOST_DEVICE (int e) {
    double xl[3][4], al[8][4] = {}, yl[8][4] = {}, dl[8][4] = {}, bfl[3][4];
    load_element(e, IEN, x, tDof, Ag, Yg, Dg, Bf, xl, al, yl, dl, bfl);
    for (int a = 0; a < 4; a++) {
      int Ac = IEN[4 * e + a];
      for (int i = 0; i < 3; i++) {
        double dol = Do[(size_t)tDof * Ac + 4 + i];
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
    scatter<3>(e, IEN, ePtr, lRv, lKv, R, Val);
  });
}

void compute_elem_ptr(MeshDev& m)
{
  const Lhs* L = lhs();
  const int* ien = m.IEN.Read();
  const int* off = L->offsets.Read();
  const int* col = L->columns.Read();
  int* ePtr = m.ePtr.Write();
  launch(m.nEl, [=] MFEM_HOST_DEVICE (int e) {
    for (int a = 0; a < 4; a++) {
      int rowN = ien[4 * e + a];
      for (int b = 0; b < 4; b++) {
        int colN = ien[4 * e + b];
        int lo = off[rowN], hi = off[rowN + 1] - 1;
        while (lo < hi) {
          int mid = (lo + hi) / 2;
          if (col[mid] < colN) lo = mid + 1; else hi = mid;
        }
        ePtr[16 * e + 4 * a + b] = lo;
      }
    }
  });
}

void add_val(int n, int dd, const int* dptr, const double* dv)
{
  double* Val = lhs()->Val.ReadWrite();
  launch(dd * n, [=] MFEM_HOST_DEVICE (int q) {
    int i = q / dd, k = q % dd;
    AtomicAdd(Val[(size_t)dd * dptr[i] + k], dv[q]);
  });
}

void precond_diag(int dof)
{
  Lhs* L = lhs();
  const int nNo = L->nNo, nnz = L->nnz, dd = dof * dof;
  const int* diag = L->diag.Read();
  const int* rows = L->rows.Read();
  const int* col = L->columns.Read();
  const double* dDmul = L->Dmul.Read();
  double* Val = L->Val.ReadWrite();
  double* dW = L->W.Write();
  // W = 1/sqrt(|diag|) (1 for a zero diagonal), times the Dirichlet factors
  launch(nNo, [=] MFEM_HOST_DEVICE (int a) {
    for (int i = 0; i < dof; i++) {
      double w = Val[(size_t)dd * diag[a] + (dof + 1) * i];
      if (w == 0.0) w = 1.0;
      w = 1.0 / sqrt(fabs(w));
      dW[dof * a + i] = w * dDmul[dof * a + i];
    }
  });
  // Val(dof r + c, j) *= W(r, row(j)) * W(c, col(j)) (pre_mul then pos_mul)
  launch(dd * nnz, [=] MFEM_HOST_DEVICE (int q) {
    int j = q / dd, k = q % dd;
    int r = k / dof, c = k % dof;
    Val[q] = Val[q] * dW[dof * rows[j] + r] * dW[dof * col[j] + c];
  });
}

//===========================================================================
// Block SELL products: value e of slot q at V[e * total + q]

/// Row i of the DOF x DOF block matrix V times x, accumulated in TA
template <int DOF, typename TV, typename TX, typename TA>
MFEM_HOST_DEVICE inline void brow(int i, const int* soff, const int* cols, const TV* V, int total, const TX* x,
    TA acc[DOF])
{
  const int s = i / 32, r = i % 32;
  for (int k = 0; k < DOF; k++) acc[k] = (TA)0;
  const int q1 = soff[s + 1];
  for (int q = soff[s] + r; q < q1; q += 64) {
    const int c0 = cols[q], c1 = cols[q + 32];
    TA x0[DOF], x1[DOF];
    for (int k = 0; k < DOF; k++) {
      x0[k] = (TA)x[DOF * c0 + k];
      x1[k] = (TA)x[DOF * c1 + k];
    }
    for (int rr = 0; rr < DOF; rr++) {
      TA t = (TA)0;
      for (int cc = 0; cc < DOF; cc++) {
        const size_t o = (size_t)(rr * DOF + cc) * total;
        t += (TA)V[o + q] * x0[cc] + (TA)V[o + q + 32] * x1[cc];
      }
      acc[rr] += t;
    }
  }
}

/// Vs[e * total + q] = (TO)Val[DD smap[q] + e] (0 for the padding)
template <int DD, typename TO>
void bsell_gather(const Sell& s, const double* Val, TO* Vs)
{
  const int* smap = s.smap.Read();
  const int total = s.total;
  launch(total, [=] MFEM_HOST_DEVICE (int q) {
    const int j = smap[q];
    for (int e = 0; e < DD; e++) Vs[(size_t)e * total + q] = j >= 0 ? (TO)Val[(size_t)DD * j + e] : (TO)0;
  });
}

//===========================================================================
// Aggregation AMG for the dof x dof block matrix (svgpu_fsi_amg.inc of the
// CUDA experiment): unsmoothed aggregation of the nodes, piecewise constant
// prolongation for every dof, Galerkin coarse matrices (block sums), the
// pseudo-inverse of the coarsest matrix (at most CMAX nodes). One damped
// block Jacobi pre- and post-smoothing step on every level, damping
// omega / rho(Dinv A) from a power iteration. The hierarchy is in float.

template <int DOF>
class Bamg {
 public:
  static constexpr int DD = DOF * DOF;
  struct Level {
    int n = 0, nnz = 0;
    Sell sellL;
    const Sell* sell = nullptr;
    Array<float> Vs;                // DD x total, entry-major
    Array<int> dslot;               // SELL slot of the diagonal entry of each node
    Array<float> Dinv;              // DD x n (pseudo-inverse of the diagonal blocks)
    Array<float> r, z, t, z2, ev;   // DOF x n
    // To the next level
    Array<int> agg, aPtr, aList;    // aggregate of each node, nodes of each aggregate
    Array<int> gPtr, gList, cslot;  // fine SELL slots summed into each coarse entry, its coarse SELL slot
  };

  Bamg(const Sell& sell0, int n0, const std::vector<int>& off0, const std::vector<int>& col0);
  std::vector<int> sizes() const;
  int npow() const { return nSolve == 0 ? 15 : 4; }
  /// Coarse matrices, smoothers and the coarsest inverse for the current
  /// matrix Val (synchronizes: the pseudo-inverse is computed on the host)
  void numeric(const double* Val);
  /// Kernel launches of z = M r (double in and out, finest level)
  void emit_vcycle(const double* r, double* z);

  std::vector<Level> levels;
  int nc = 0;
  Vector Ad;             // dense coarsest matrix
  Array<float> Ainv;
  Array<float> wv;       // damping per level
  Vector ns;             // norms of the power iterations (16 per level)
  std::vector<double> hA;
  std::vector<float> hI;
  double omega;
  int cmax;
  int nSolve = 0;
};

template <int DOF>
Bamg<DOF>::Bamg(const Sell& sell0, int n0, const std::vector<int>& off0, const std::vector<int>& col0)
{
  double t0 = now();
  omega = env_double("SVMFEM_AMG_W", 0.9);
  cmax = env_int("SVMFEM_AMG_CMAX", 128);
  levels.reserve(32);
  std::vector<int> off = off0, col = col0;
  int n = n0;
  std::vector<int> prev_cmap;   // coarse CSR entry of the fine entries (level - 1)
  while (true) {
    levels.emplace_back();
    Level& L = levels.back();
    const int l = (int)levels.size() - 1;
    L.n = n;
    L.nnz = (int)col.size();
    if (l > 0) {
      L.sellL.build(n, off, col);
      L.sell = &L.sellL;
      // The coarse slots of the Galerkin sums of the previous level
      Level& F = levels[l - 1];
      std::vector<int> cslot(L.nnz);
      for (int I = 0; I < n; I++) {
        for (int j = off[I]; j < off[I + 1]; j++) cslot[j] = L.sell->slot(I, j - off[I]);
      }
      dupload(F.cslot, cslot);
    } else {
      L.sell = &sell0;
    }
    dalloc(L.Vs, DD * L.sell->total);
    fill(L.Vs.Write(), DD * L.sell->total, 0.0f);
    std::vector<int> dslot(n, -1);
    for (int i = 0; i < n; i++) {
      for (int j = off[i]; j < off[i + 1]; j++) {
        if (col[j] == i) dslot[i] = L.sell->slot(i, j - off[i]);
      }
      if (dslot[i] < 0) fail("AMG: a row without diagonal entry");
    }
    dupload(L.dslot, dslot);
    dalloc(L.Dinv, DD * n);
    for (Array<float>* v : {&L.r, &L.z, &L.t, &L.z2, &L.ev}) dalloc(*v, DOF * n);
    if (n <= cmax) break;
    std::vector<int> agg;
    const int na = aggregate(n, off, col, agg);
    if (na >= n) break;
    // Coarse pattern: the aggregate pairs of the entries, sorted in every row
    std::vector<std::vector<int>> rows(na);
    for (int i = 0; i < n; i++) {
      for (int j = off[i]; j < off[i + 1]; j++) rows[agg[i]].push_back(agg[col[j]]);
    }
    std::vector<int> coff(na + 1, 0), ccol;
    for (int I = 0; I < na; I++) {
      auto& r = rows[I];
      std::sort(r.begin(), r.end());
      r.erase(std::unique(r.begin(), r.end()), r.end());
      ccol.insert(ccol.end(), r.begin(), r.end());
      coff[I + 1] = (int)ccol.size();
    }
    // Coarse entry of every fine entry, then the fine SELL slots of every coarse entry
    const int cnnz = (int)ccol.size();
    std::vector<int> gCnt(cnnz + 1, 0), fb2cb(L.nnz);
    for (int i = 0; i < n; i++) {
      const int I = agg[i];
      for (int j = off[i]; j < off[i + 1]; j++) {
        const int J = agg[col[j]];
        const int cb = (int)(std::lower_bound(ccol.begin() + coff[I], ccol.begin() + coff[I + 1], J) - ccol.begin());
        fb2cb[j] = cb;
        gCnt[cb + 1]++;
      }
    }
    for (int k = 0; k < cnnz; k++) gCnt[k + 1] += gCnt[k];
    std::vector<int> gList(L.nnz), pos(gCnt.begin(), gCnt.end() - 1);
    for (int i = 0; i < n; i++) {
      for (int j = off[i]; j < off[i + 1]; j++) gList[pos[fb2cb[j]]++] = L.sell->slot(i, j - off[i]);
    }
    std::vector<int> aCnt(na + 1, 0);
    for (int i = 0; i < n; i++) aCnt[agg[i] + 1]++;
    for (int I = 0; I < na; I++) aCnt[I + 1] += aCnt[I];
    std::vector<int> aList(n), apos(aCnt.begin(), aCnt.end() - 1);
    for (int i = 0; i < n; i++) aList[apos[agg[i]]++] = i;
    dupload(L.agg, agg);
    dupload(L.aPtr, aCnt);
    dupload(L.aList, aList);
    dupload(L.gPtr, gCnt);
    dupload(L.gList, gList);
    n = na;
    off = coff;
    col = ccol;
  }
  const int nl = (int)levels.size();
  nc = DOF * levels.back().n;
  dalloc(Ad, nc * nc);
  dalloc(Ainv, nc * nc);
  dalloc(wv, nl);
  fill(wv.Write(), nl, 0.0f);
  dalloc(ns, 16 * nl);
  hA.resize((size_t)nc * nc);
  hI.resize((size_t)nc * nc);
  if (verbose()) {
    printf("svmfem: AMG dof %d levels", DOF);
    for (int s : sizes()) printf(" %d", s);
    printf(" omega %g in %.2f s\n", omega, now() - t0);
  }
}

template <int DOF>
std::vector<int> Bamg<DOF>::sizes() const
{
  std::vector<int> s;
  for (const auto& L : levels) s.push_back(L.n);
  return s;
}

template <int DOF>
void Bamg<DOF>::numeric(const double* Val)
{
  const int nPow = npow();
  const bool first = nSolve == 0;
  nSolve++;
  const int nl = (int)levels.size();
  bsell_gather<DD, float>(*levels[0].sell, Val, levels[0].Vs.Write());
  double* dns = ns.ReadWrite();
  float* dwv = wv.ReadWrite();
  for (int l = 0; l < nl; l++) {
    Level& L = levels[l];
    const int n = L.n;
    const int total = L.sell->total;
    const int* dslot = L.dslot.Read();
    const float* Vs = L.Vs.Read();
    float* Dinv = L.Dinv.Write();
    // Inverse of the diagonal blocks. The dofs with a zero row and column in
    // the block (Dirichlet dofs after the FSILS scaling) are left out (zero
    // rows and columns of the result, like the pseudo-inverse); identity
    // for a singular remaining block.
    launch(n, [=] MFEM_HOST_DEVICE (int a) {
      double m[DOF][2 * DOF];
      bool act[DOF];
      float d[DD];
      for (int e = 0; e < DD; e++) d[e] = Vs[(size_t)e * total + dslot[a]];
      for (int k = 0; k < DOF; k++) {
        act[k] = false;
        for (int q = 0; q < DOF; q++) act[k] = act[k] || d[DOF * k + q] != 0.0f || d[DOF * q + k] != 0.0f;
      }
      for (int r = 0; r < DOF; r++) {
        for (int c = 0; c < DOF; c++) {
          m[r][c] = (act[r] && act[c]) ? d[DOF * r + c] : (r == c ? 1.0 : 0.0);
          m[r][DOF + c] = (r == c) ? 1.0 : 0.0;
        }
      }
      bool ok = true;
      for (int c = 0; c < DOF && ok; c++) {
        int p = c;
        for (int r = c + 1; r < DOF; r++) if (fabs(m[r][c]) > fabs(m[p][c])) p = r;
        if (m[p][c] == 0.0) { ok = false; break; }
        if (p != c) for (int k = 0; k < 2 * DOF; k++) { double t = m[c][k]; m[c][k] = m[p][k]; m[p][k] = t; }
        const double inv = 1.0 / m[c][c];
        for (int k = 0; k < 2 * DOF; k++) m[c][k] *= inv;
        for (int r = 0; r < DOF; r++) {
          if (r == c) continue;
          const double f = m[r][c];
          if (f != 0.0) for (int k = 0; k < 2 * DOF; k++) m[r][k] -= f * m[c][k];
        }
      }
      float* o = Dinv + DD * (size_t)a;
      for (int r = 0; r < DOF; r++) {
        for (int c = 0; c < DOF; c++) {
          o[DOF * r + c] = !(act[r] && act[c]) ? 0.0f : ok ? (float)m[r][DOF + c] : (r == c ? 1.0f : 0.0f);
        }
      }
    });
    if (l + 1 == nl) break;
    Level& C = levels[l + 1];
    // Galerkin coarse entries: sums of the fine entries
    {
      const int* gPtr = L.gPtr.Read();
      const int* gList = L.gList.Read();
      const int* cslot = L.cslot.Read();
      const int totalC = C.sell->total;
      float* Vc = C.Vs.ReadWrite();
      launch(C.nnz, [=] MFEM_HOST_DEVICE (int cb) {
        float s[DD];
        for (int e = 0; e < DD; e++) s[e] = 0.0f;
        for (int k = gPtr[cb]; k < gPtr[cb + 1]; k++) {
          const int q = gList[k];
          for (int e = 0; e < DD; e++) s[e] += Vs[(size_t)e * total + q];
        }
        const int qc = cslot[cb];
        for (int e = 0; e < DD; e++) Vc[(size_t)e * totalC + qc] = s[e];
      });
    }
    // Damping with the spectral radius of Dinv A (power iteration; the
    // norms accumulate in ns[s0 + k] on the device)
    const int* soff = L.sell->slice_off.Read();
    const int* cols = L.sell->cols.Read();
    const int s0 = 16 * l;
    const int nv = DOF * n;
    fill(dns + s0, 16, 0.0);
    float* x = L.ev.ReadWrite();
    float* y = L.z2.ReadWrite();
    if (first) {
      launch_red<1>(nv, dns, slots(s0), [=] MFEM_HOST_DEVICE (int i, double* v) {
        unsigned h = (unsigned)i * 2654435761u;
        h = h ^ (h >> 15);
        float xi = 1.0f + (float)(h % 1000u) * 1e-3f;
        x[i] = xi;
        v[0] = (double)xi * xi;
      });
    } else {
      launch_red<1>(nv, dns, slots(s0), [=] MFEM_HOST_DEVICE (int i, double* v) {
        float xi = x[i];
        v[0] = (double)xi * xi;
      });
    }
    for (int k = 0; k < nPow; k++) {
      const int slot_in = s0 + k, slot_out = s0 + k + 1;
      const float* xin = x;
      float* yout = y;
      launch_red<1>(n, dns, slots(slot_out), [=] MFEM_HOST_DEVICE (int a, double* v) {
        const float inv = (float)(1.0 / sqrt(dns[slot_in]));
        float ax[DOF];
        brow<DOF, float, float, float>(a, soff, cols, Vs, total, xin, ax);
        const float* d = Dinv + DD * (size_t)a;
        double s = 0.0;
        for (int r = 0; r < DOF; r++) {
          float t = 0.0f;
          for (int c = 0; c < DOF; c++) t += d[DOF * r + c] * ax[c];
          t *= inv;
          yout[DOF * a + r] = t;
          s += (double)t * t;
        }
        v[0] = s;
      });
      std::swap(x, y);
    }
    {
      const int slot = s0 + nPow;
      const double om = omega;
      launch(1, [=] MFEM_HOST_DEVICE (int) {
        double rho = sqrt(dns[slot]);
        if (rho < 1e-12) rho = 1e-12;
        dwv[l] = (float)(om / rho);
      });
    }
    if (x != L.ev.ReadWrite()) copy(x, L.ev.ReadWrite(), nv);
  }
  // Dense coarsest matrix and its pseudo-inverse (on the host: the matrix is
  // not symmetric and has zero rows and columns)
  {
    Level& Lc = levels.back();
    const int n = Lc.n, ncc = nc, total = Lc.sell->total;
    const int* soff = Lc.sell->slice_off.Read();
    const int* cols = Lc.sell->cols.Read();
    const int* smap = Lc.sell->smap.Read();
    const float* Vs = Lc.Vs.Read();
    double* A = Ad.Write();
    launch(n, [=] MFEM_HOST_DEVICE (int I) {
      for (int r = 0; r < DOF; r++) {
        for (int c = 0; c < ncc; c++) A[(size_t)(DOF * I + r) * ncc + c] = 0.0;
      }
      const int s = I / 32, rr = I % 32;
      for (int q = soff[s] + rr; q < soff[s + 1]; q += 32) {
        if (smap[q] < 0) continue;
        const int J = cols[q];
        for (int r = 0; r < DOF; r++) {
          for (int c = 0; c < DOF; c++) A[(size_t)(DOF * I + r) * ncc + DOF * J + c] = Vs[(size_t)(DOF * r + c) * total + q];
        }
      }
    });
    ddownload(Ad, hA.data(), nc * nc);
    pinv(nc, hA.data(), hI.data());
    dupload(Ainv, hI.data(), nc * nc);
  }
  static int nPrint = 0;
  if (verbose() && nPrint++ < 2) {
    const float* w = wv.HostRead();
    printf("svmfem: AMG dof %d damping", DOF);
    for (int l = 0; l + 1 < nl; l++) printf(" %g", w[l]);
    printf("\n");
  }
}

template <int DOF>
void Bamg<DOF>::emit_vcycle(const double* r0, double* z0)
{
  const int nl = (int)levels.size();
  const float* dwv = wv.Read();
  // Down: zs = w Dinv r; t = r - A zs; r_c = restriction of t
  for (int l = 0; l + 1 < nl; l++) {
    Level& L = levels[l];
    Level& C = levels[l + 1];
    const int n = L.n, total = L.sell->total;
    const int* soff = L.sell->slice_off.Read();
    const int* cols = L.sell->cols.Read();
    const float* Vs = L.Vs.Read();
    const float* Dinv = L.Dinv.Read();
    const double* rd = (l == 0) ? r0 : nullptr;
    const float* rf = (l == 0) ? nullptr : L.r.Read();
    float* zs = L.z.Write();
    float* t = L.t.Write();
    launch(n, [=] MFEM_HOST_DEVICE (int a) {
      const float w = dwv[l];
      const float* d = Dinv + DD * (size_t)a;
      float rv[DOF];
      for (int k = 0; k < DOF; k++) rv[k] = rd ? (float)rd[DOF * a + k] : rf[DOF * a + k];
      for (int i = 0; i < DOF; i++) {
        float s = 0.0f;
        for (int k = 0; k < DOF; k++) s += d[DOF * i + k] * rv[k];
        zs[DOF * a + i] = w * s;
      }
    });
    launch(n, [=] MFEM_HOST_DEVICE (int a) {
      float y[DOF];
      brow<DOF, float, float, float>(a, soff, cols, Vs, total, zs, y);
      for (int k = 0; k < DOF; k++) {
        const float rk = rd ? (float)rd[DOF * a + k] : rf[DOF * a + k];
        t[DOF * a + k] = rk - y[k];
      }
    });
    const int* aPtr = L.aPtr.Read();
    const int* aList = L.aList.Read();
    float* rc = C.r.Write();
    launch(DOF * C.n, [=] MFEM_HOST_DEVICE (int q) {
      const int I = q / DOF, k = q % DOF;
      float s = 0.0f;
      for (int p = aPtr[I]; p < aPtr[I + 1]; p++) s += t[DOF * (size_t)aList[p] + k];
      rc[q] = s;
    });
  }
  // Coarsest level: z = Ainv r
  {
    Level& Lc = levels.back();
    const int ncc = nc;
    const float* Ai = Ainv.Read();
    const float* rc = Lc.r.Read();
    float* zc = Lc.z.Write();
    launch(ncc, [=] MFEM_HOST_DEVICE (int i) {
      float s = 0.0f;
      for (int k = 0; k < ncc; k++) s += Ai[(size_t)i * ncc + k] * rc[k];
      zc[i] = s;
    });
  }
  // Up: z2 = zs + P z_c; z = z2 + w Dinv (r - A z2)
  for (int l = nl - 2; l >= 0; l--) {
    Level& L = levels[l];
    Level& C = levels[l + 1];
    const int n = L.n, total = L.sell->total;
    const int* soff = L.sell->slice_off.Read();
    const int* cols = L.sell->cols.Read();
    const float* Vs = L.Vs.Read();
    const float* Dinv = L.Dinv.Read();
    const int* agg = L.agg.Read();
    const float* zs = L.z.Read();
    const float* ec = C.z.Read();
    float* z2 = L.z2.Write();
    launch(DOF * n, [=] MFEM_HOST_DEVICE (int q) {
      z2[q] = zs[q] + ec[DOF * (size_t)agg[q / DOF] + q % DOF];
    });
    if (l > 0) {
      const float* rf = L.r.Read();
      float* z = L.z.Write();
      launch(n, [=] MFEM_HOST_DEVICE (int a) {
        float y[DOF];
        brow<DOF, float, float, float>(a, soff, cols, Vs, total, z2, y);
        const float* d = Dinv + DD * (size_t)a;
        const float w = dwv[l];
        for (int i = 0; i < DOF; i++) {
          float s = 0.0f;
          for (int k = 0; k < DOF; k++) s += d[DOF * i + k] * (rf[DOF * a + k] - y[k]);
          z[DOF * a + i] = z2[DOF * a + i] + w * s;
        }
      });
    } else {
      // Finest level: double output; no component in the dofs without
      // equation (zero row of the block, e.g. Dirichlet dofs), like the
      // unpreconditioned solver
      launch(n, [=] MFEM_HOST_DEVICE (int a) {
        float y[DOF];
        brow<DOF, float, float, float>(a, soff, cols, Vs, total, z2, y);
        const float* d = Dinv + DD * (size_t)a;
        const float w = dwv[0];
        for (int i = 0; i < DOF; i++) {
          float s = 0.0f;
          bool act = false;
          for (int k = 0; k < DOF; k++) {
            s += d[DOF * i + k] * ((float)r0[DOF * a + k] - y[k]);
            act = act || d[DOF * i + k] != 0.0f;
          }
          z0[DOF * a + i] = act ? (double)(z2[DOF * a + i] + w * s) : 0.0;
        }
      });
    }
  }
}

//===========================================================================
// FSILS GMRES and CG on the device matrix, with the AMG preconditioner

// Slots of the solver scalars gs (double, device)
constexpr int NRM = 0;   // two slots (parity): |w|^2 of the current and the previous Arnoldi step
constexpr int RZ = 2;    // two slots (parity): r.z of the CG
constexpr int PQ = 4, RR = 5;
constexpr int GH = 8;    // Hessenberg columns (two parities of sD + 1)

template <int DOF>
class LsSolver {
 public:
  static constexpr int DD = DOF * DOF;
  explicit LsSolver(Lhs& lhs);
  void gmres(int mItr, int sD, double absTol, double relTol, svmfem_stats* stats);
  void cg(int mItr, double absTol, double relTol, svmfem_stats* stats);

  Lhs& lhs;
  int nNo, nnz, n, total;
  const Sell& sell;
  Vector Ks;                 // DD x total double
  Vector R, X, P, Q, Z, T, wb0, wb1, gu, gs, dcoef;
  std::unique_ptr<Mapped> hgs, hcoef;
  std::unique_ptr<Bamg<DOF>> amg;
  bool use_amg;
  int sD = 0, ngs = 0, ncoef = 0, cg_chunk = 2;
  GraphCache graphs;
  double t_setup = 0, t_solve = 0;
  int n_call = 0, n_it = 0;

  void alloc_work(int sD);
  void setup();
  void emit_gm_init();
  void emit_gm_restart();
  void emit_gm_step(int i);
  void emit_gm_fin();
  void emit_cg_init();
  void emit_cg_chunk(int parity);
  template <typename F> void run(const std::string& key, F&& emit)
  {
    graphs.run(key, std::function<void()>(emit));
  }
};

template <int DOF>
LsSolver<DOF>::LsSolver(Lhs& lhs_) : lhs(lhs_), sell(st->sell)
{
  nNo = lhs.nNo;
  nnz = lhs.nnz;
  n = DOF * nNo;
  total = sell.total;
  dalloc(Ks, (size_t)DD * total);
  for (Vector* v : {&R, &X, &P, &Q, &Z, &T, &wb0, &wb1}) dalloc(*v, n);
  use_amg = amg_enabled();
  if (use_amg) amg.reset(new Bamg<DOF>(sell, nNo, lhs.h_off, lhs.h_col));
  cg_chunk = env_int("SVMFEM_CG_CHUNK", std::max(2, std::min(8, 400000 / n / 2 * 2)));
  if (verbose()) printf("svmfem: FSI solver dof %d nNo %d nnz %d SELL padding %.2f AMG %d CG chunk %d\n", DOF, nNo, nnz,
      sell.padding, (int)use_amg, cg_chunk);
}

template <int DOF>
void LsSolver<DOF>::alloc_work(int sD_)
{
  if (sD >= sD_) return;
  sD = sD_;
  ngs = pad8(GH + 2 * (sD + 1));
  dalloc(gs, ngs);
  fill(gs.Write(), ngs, 0.0);
  hgs.reset(new Mapped(ngs));
  ncoef = pad8(sD + 2);
  hcoef.reset(new Mapped(ncoef));
  dalloc(dcoef, ncoef);
  dalloc(gu, (size_t)(sD + 1) * n);
  graphs.clear();
}

/// Matrix of this solve: the SELL copy and the AMG hierarchy
template <int DOF>
void LsSolver<DOF>::setup()
{
  const double t0 = now();
  const double* Val = lhs.Val.Read();
  bsell_gather<DD, double>(sell, Val, Ks.Write());
  if (use_amg) amg->numeric(Val);
  sync();
  t_setup += now() - t0;
}

/// gs = 0, X = 0, w = R, gs[NRM + 1] = |w|^2
template <int DOF>
void LsSolver<DOF>::emit_gm_init()
{
  double* dgs = gs.ReadWrite();
  fill(dgs, ngs, 0.0);
  fill(X.Write(), n, 0.0);
  const double* dR = R.Read();
  double* w = wb1.Write();
  launch_red<1>(n, dgs, slots(NRM + 1), [=] MFEM_HOST_DEVICE (int k, double* v) {
    const double wk = dR[k];
    w[k] = wk;
    v[0] = wk * wk;
  });
  publish8(dgs, hgs->dev, 0, 1);
}

/// w = R - K X (restart after sD steps), gs[NRM + 1] = |w|^2
template <int DOF>
void LsSolver<DOF>::emit_gm_restart()
{
  double* dgs = gs.ReadWrite();
  fill(dgs, ngs, 0.0);
  const double* dR = R.Read();
  const double* dX = X.Read();
  double* w = wb1.Write();
  const int* soff = sell.slice_off.Read();
  const int* cols = sell.cols.Read();
  const double* K = Ks.Read();
  const int tot = total;
  launch_red<1>(nNo, dgs, slots(NRM + 1), [=] MFEM_HOST_DEVICE (int a, double* v) {
    double acc[DOF];
    brow<DOF, double, double, double>(a, soff, cols, K, tot, dX, acc);
    double s = 0.0;
    for (int k = 0; k < DOF; k++) {
      const double wk = dR[DOF * a + k] - acc[k];
      w[DOF * a + k] = wk;
      s += wk * wk;
    }
    v[0] = s;
  });
  publish8(dgs, hgs->dev, 0, 1);
}

/// Arnoldi step i: u_i = w / |w|; w = K M^-1 u_i; modified Gram-Schmidt
/// against u_0..u_i (h_j in gs[GH_b + j]; the update w -= h_j u_j and the
/// next dot product u_{j+1} . w in one pass, as gmres::gmres_v); |w|^2 in
/// gs[NRM + b]
template <int DOF>
void LsSolver<DOF>::emit_gm_step(int i)
{
  const int b = i & 1, bp = 1 - b;
  double* dgs = gs.ReadWrite();
  double* u = gu.ReadWrite();
  const double* win = (b == 0 ? wb1 : wb0).Read();
  double* wout = (b == 0 ? wb0 : wb1).Write();
  const int* soff = sell.slice_off.Read();
  const int* cols = sell.cols.Read();
  const double* K = Ks.Read();
  const int tot = total, nn = n, sd = sD;
  const int g0 = GH + b * (sd + 1), g1 = GH + bp * (sd + 1);
  double* ui = u + (size_t)i * n;
  // u_i = win / |win|; wout = K M^-1 u_i; gs[GH_b] += u_0 . wout; gs[NRM + b] = 0
  if (use_amg) {
    launch(nn, [=] MFEM_HOST_DEVICE (int k) {
      const double inv = 1.0 / sqrt(dgs[NRM + bp]);
      ui[k] = win[k] * inv;
    });
    double* dZ = Z.Write();
    amg->emit_vcycle(ui, dZ);
    launch_red<1>(nNo, dgs, slots(g0), [=] MFEM_HOST_DEVICE (int a, double* v) {
      double acc[DOF];
      brow<DOF, double, double, double>(a, soff, cols, K, tot, dZ, acc);
      double s = 0.0;
      for (int k = 0; k < DOF; k++) {
        wout[DOF * a + k] = acc[k];
        s += u[DOF * a + k] * acc[k];
      }
      v[0] = s;
      if (a == 0) dgs[NRM + b] = 0.0;
    });
  } else {
    launch_red<1>(nNo, dgs, slots(g0), [=] MFEM_HOST_DEVICE (int a, double* v) {
      const double inv = 1.0 / sqrt(dgs[NRM + bp]);
      double acc[DOF];
      brow<DOF, double, double, double>(a, soff, cols, K, tot, win, acc);
      double s = 0.0;
      for (int k = 0; k < DOF; k++) {
        const double wk = acc[k] * inv;
        wout[DOF * a + k] = wk;
        ui[DOF * a + k] = win[DOF * a + k] * inv;
        s += u[DOF * a + k] * wk;
      }
      v[0] = s;
      if (a == 0) dgs[NRM + b] = 0.0;
    });
  }
  // w -= h_j u_j; then u_{j+1} . w (j < i) or |w|^2 (j = i); the last pass
  // clears gs[GH_b'] for the next step
  for (int j = 0; j <= i; j++) {
    const double* uj = u + (size_t)j * n;
    const double* un = u + (size_t)(j + 1) * n;
    const bool last = (j == i);
    const int slot = last ? NRM + b : g0 + j + 1;
    launch_red<1>(nn, dgs, slots(slot), [=] MFEM_HOST_DEVICE (int k, double* v) {
      const double wk = wout[k] - dgs[g0 + j] * uj[k];
      wout[k] = wk;
      v[0] = (last ? wk : un[k]) * wk;
      if (last && k <= sd) dgs[g1 + k] = 0.0;
    });
  }
  publish8(dgs, hgs->dev, 0, 1);
  publish8(dgs, hgs->dev, g0 / 8, (g0 + i) / 8 - g0 / 8 + 1);
}

/// X += M^-1 sum_j y_j u_j with the coefficients from the host
template <int DOF>
void LsSolver<DOF>::emit_gm_fin()
{
  fetch8(hcoef->dev, dcoef.Write(), hcoef->nvec);
  const double* coef = dcoef.Read();
  const double* u = gu.Read();
  double* dX = X.ReadWrite();
  const int nn = n;
  if (use_amg) {
    double* dT = T.Write();
    launch(nn, [=] MFEM_HOST_DEVICE (int k) {
      const int m = (int)coef[0];
      double s = 0.0;
      for (int j = 0; j < m; j++) s = s + coef[1 + j] * u[(size_t)j * nn + k];
      dT[k] = s;
    });
    double* dZ = Z.Write();
    amg->emit_vcycle(dT, dZ);
    launch(nn, [=] MFEM_HOST_DEVICE (int k) { dX[k] = dX[k] + dZ[k]; });
  } else {
    launch(nn, [=] MFEM_HOST_DEVICE (int k) {
      const int m = (int)coef[0];
      double s = dX[k];
      for (int j = 0; j < m; j++) s = s + coef[1 + j] * u[(size_t)j * nn + k];
      dX[k] = s;
    });
  }
}

// gmres::gmres_v without coupled faces, with the AMG right preconditioner
// (K M^-1 y = R, X = M^-1 y: the convergence check is on the residual of
// the original system). R (device) in, X (device) out.
template <int DOF>
void LsSolver<DOF>::gmres(int mItr, int sD_, double absTol, double relTol, svmfem_stats* stats)
{
  alloc_work(sD_);
  const int sd = sD;
  double* h_gs = hgs->host;
  std::vector<double> h((size_t)(sd + 1) * sd, 0.0), y(sd), c(sd), s(sd), err(sd + 1);
  auto H = [&](int i, int j) -> double& { return h[i + (size_t)j * (sd + 1)]; };
  double eps = 0.0, iNorm = 0.0, fNorm = 0.0, dB = 0.0;
  int itr = 0, last_i = 0;
  bool success = false;
  for (int l = 0; l < mItr; l++) {
    if (l == 0) {
      run("gm_init", [&] { emit_gm_init(); });
    } else {
      run("gm_restart", [&] { emit_gm_restart(); });
      itr++;
    }
    sync();
    err[0] = std::sqrt(h_gs[NRM + 1]);
    if (l == 0) {
      iNorm = fNorm = err[0];
      eps = std::max(absTol, relTol * err[0]);
      if (iNorm <= absTol) {
        success = true;
        break;
      }
    }
    if (err[0] == 0.0) fail("a zero matrix norm was computed in GMRES");
    dB = fNorm;
    for (int i = 0; i < sd; i++) {
      last_i = i;
      run("gm" + std::to_string(i), [&] { emit_gm_step(i); });
      itr++;
      sync();
      const int b = i & 1;
      for (int j = 0; j <= i; j++) H(j, i) = h_gs[GH + b * (sd + 1) + j];
      H(i + 1, i) = std::sqrt(h_gs[NRM + b]);
      for (int j = 0; j < i; j++) {
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
    for (int i = 0; i <= last_i; i++) y[i] = err[i];
    for (int j = last_i; j >= 0; j--) {
      for (int k = j + 1; k <= last_i; k++) y[j] = y[j] - H(j, k) * y[k];
      y[j] = y[j] / H(j, j);
    }
    double* hc = hcoef->host;
    hc[0] = last_i + 1;
    for (int j = 0; j <= last_i; j++) hc[1 + j] = y[j];
    run("gm_fin", [&] { emit_gm_fin(); });
    fNorm = std::fabs(err[last_i + 1]);
    if (success) break;
  }
  sync();
  n_it += itr;
  if (stats) {
    stats->GM_itr = itr;
    stats->success = success ? 1 : 0;
    stats->iNorm = iNorm;
    stats->fNorm = fNorm;
    stats->dB = (dB > 0.0 && fNorm > 0.0) ? 10.0 * std::log(fNorm / dB) : 0.0;
  }
}

/// r = R, X = p = 0, gs[RR] = |r|^2, z = M r, gs[RZ] = r.z
template <int DOF>
void LsSolver<DOF>::emit_cg_init()
{
  double* dgs = gs.ReadWrite();
  const double* dR = R.Read();
  double* r = Q.Write();      // Q holds r (q is recomputed every iteration into T)
  double* dX = X.Write();
  double* p = P.Write();
  fill(dgs, 8, 0.0);
  launch_red<1>(n, dgs, slots(RR), [=] MFEM_HOST_DEVICE (int k, double* v) {
    const double rk = dR[k];
    r[k] = rk;
    dX[k] = 0.0;
    p[k] = 0.0;
    v[0] = rk * rk;
    if (k == 0) dgs[RZ + 1] = 1.0;
  });
  double* z = Z.Write();
  if (use_amg) amg->emit_vcycle(r, z);
  else copy(r, z, n);
  launch_red<1>(n, dgs, slots(RZ), [=] MFEM_HOST_DEVICE (int k, double* v) { v[0] = r[k] * z[k]; });
  publish8(dgs, hgs->dev, 0, 1);
}

/// cg_chunk iterations of the preconditioned CG (cgrad::cgrad_v with the
/// V-cycle): the parity b of the iteration (parity of the first one given)
/// selects the r.z slot
template <int DOF>
void LsSolver<DOF>::emit_cg_chunk(int parity)
{
  double* dgs = gs.ReadWrite();
  double* dX = X.ReadWrite();
  double* r = Q.ReadWrite();
  double* z = Z.ReadWrite();
  double* p = P.ReadWrite();
  double* q = T.ReadWrite();
  const int* soff = sell.slice_off.Read();
  const int* cols = sell.cols.Read();
  const double* K = Ks.Read();
  const int tot = total, nn = n;
  for (int it = 0; it < cg_chunk; it++) {
    const int b = (parity + it) & 1;
    // p = z + (rz_new / rz_old) p
    launch(nn, [=] MFEM_HOST_DEVICE (int k) {
      const double beta = dgs[RZ + b] / dgs[RZ + 1 - b];
      p[k] = z[k] + beta * p[k];
      if (k == 0) dgs[PQ] = 0.0;
    });
    // q = K p; gs[PQ] += p . q
    launch_red<1>(nNo, dgs, slots(PQ), [=] MFEM_HOST_DEVICE (int a, double* v) {
      double acc[DOF];
      brow<DOF, double, double, double>(a, soff, cols, K, tot, p, acc);
      double s = 0.0;
      for (int k = 0; k < DOF; k++) {
        q[DOF * a + k] = acc[k];
        s += p[DOF * a + k] * acc[k];
      }
      v[0] = s;
      if (a == 0) dgs[RR] = 0.0;
    });
    // alpha = rz / pq; X += alpha p; r -= alpha q; gs[RR] += |r|^2
    launch_red<1>(nn, dgs, slots(RR), [=] MFEM_HOST_DEVICE (int k, double* v) {
      const double alpha = dgs[RZ + b] / dgs[PQ];
      dX[k] = dX[k] + alpha * p[k];
      const double rk = r[k] - alpha * q[k];
      r[k] = rk;
      v[0] = rk * rk;
      if (k == 0) dgs[RZ + 1 - b] = 0.0;
    });
    // z = M r; gs[RZ_b'] += r . z
    if (use_amg) amg->emit_vcycle(r, z);
    else copy(r, z, nn);
    launch_red<1>(nn, dgs, slots(RZ + 1 - b), [=] MFEM_HOST_DEVICE (int k, double* v) { v[0] = r[k] * z[k]; });
  }
  publish8(dgs, hgs->dev, 0, 1);
}

// cgrad::cgrad_v with the AMG preconditioner. R (device) in, X (device) out
template <int DOF>
void LsSolver<DOF>::cg(int mItr, double absTol, double relTol, svmfem_stats* stats)
{
  alloc_work(1);
  double* h_gs = hgs->host;
  run("cg_init", [&] { emit_cg_init(); });
  sync();
  double err = h_gs[RR], errO = err;
  const double iNorm = std::sqrt(err);
  const double eps = std::pow(std::max(absTol, relTol * iNorm), 2.0);
  int nit = 0, parity = 0;
  bool success = false;
  while (nit < mItr) {
    if (err < eps) {
      success = true;
      break;
    }
    errO = err;
    run("cg_chunk" + std::to_string(parity), [&] { emit_cg_chunk(parity); });
    nit += cg_chunk;
    parity = (parity + cg_chunk) & 1;
    sync();
    err = h_gs[RR];
  }
  if (err < eps) success = true;
  n_it += nit;
  if (stats) {
    stats->CG_itr = nit;
    stats->success = success ? 1 : 0;
    stats->iNorm = iNorm;
    stats->fNorm = std::sqrt(err);
    stats->dB = (errO < std::numeric_limits<double>::epsilon()) ? 0.0 : 5.0 * std::log(err / errO);
  }
}

LsSolver<3>* ls3 = nullptr;
LsSolver<4>* ls4 = nullptr;

template <int DOF>
int solve(LsSolver<DOF>*& ls, int type, int mItr, int sD, double absTol, double relTol, double* R, svmfem_stats* stats)
{
  const double t0 = now();
  Lhs* L = lhs();
  if (!ls) ls = new LsSolver<DOF>(*L);
  dcopy_in(ls->R, R, ls->n);
  ls->setup();
  if (type == 0) ls->gmres(mItr, sD, absTol, relTol, stats);
  else ls->cg(mItr, absTol, relTol, stats);
  ddownload(ls->X, R, ls->n);
  fill(L->Val.Write(), 16 * L->nnz, 0.0);
  const double dt = now() - t0;
  ls->t_solve += dt;
  ls->n_call++;
  if (stats) stats->time = dt;
  if (getenv("SVMFEM_TIMING") && ls->n_call % 10 == 0) {
    printf("SVMFEM ls dof %d calls=%d per call: all=%.4f setup=%.4f (%.3f ms/it)\n", DOF, ls->n_call,
        ls->t_solve / ls->n_call, ls->t_setup / ls->n_call, 1e3 * ls->t_solve / std::max(1, ls->n_it));
  }
  return 0;
}

}  // namespace fsi

void destroy_fsi()
{
  delete fsi::ls3;
  delete fsi::ls4;
  fsi::ls3 = nullptr;
  fsi::ls4 = nullptr;
  delete fsi::st;
  fsi::st = nullptr;
}

}  // namespace svmfem

//===========================================================================
// C interface

using namespace svmfem;
using namespace svmfem::fsi;

extern "C" SVMFEM_API int svmfem_fsi_setup(int tnNo, const double* x, int nnz, const int* rowPtr, const int* colPtr)
{
  try {
    init_device(nullptr);
    double t0 = now();
    delete st;
    st = new FsiState();
    st->tnNo = tnNo;
    st->nnz = nnz;
    dupload(st->x, x, 3 * tnNo);
    Lhs* L = make_lhs(tnNo, nnz, rowPtr, colPtr);
    st->sell.build(tnNo, L->h_off, L->h_col);
    dalloc(st->R, 4 * tnNo);
    dalloc(st->Bf, 3 * tnNo);
    st->hR.reset(new Mapped(4 * tnNo));
    st->hBf.reset(new Mapped(3 * tnNo));
    sync();
    if (verbose()) printf("svmfem: FSI setup tnNo %d nnz %d SELL padding %.2f in %.2f s\n", tnNo, nnz, st->sell.padding, now() - t0);
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "svmfem_fsi_setup: %s\n", e.what());
    return 1;
  }
}

extern "C" SVMFEM_API int svmfem_fsi_add_mesh(int nEl, const int* IEN, int nG, const double* w, const double* N,
    const double* Nxi)
{
  try {
    if (!st) fail("svmfem_fsi_setup was not called");
    if (nG > 4) fail("at most 4 Gauss points are supported");
    std::unique_ptr<MeshDev> m(new MeshDev());
    m->nEl = nEl;
    dupload(m->IEN, IEN, 4 * nEl);
    dalloc(m->ePtr, 16 * nEl);
    compute_elem_ptr(*m);
    m->gd.nG = nG;
    for (int g = 0; g < nG; g++) {
      m->gd.w[g] = w[g];
      for (int a = 0; a < 4; a++) m->gd.N[g][a] = N[a + 4 * g];
    }
    for (int a = 0; a < 4; a++) {
      for (int i = 0; i < 3; i++) m->gd.Nxi[i][a] = Nxi[i + 3 * a];
    }
    st->meshes.push_back(std::move(m));
    sync();
    return (int)st->meshes.size() - 1;
  } catch (const std::exception& e) {
    fprintf(stderr, "svmfem_fsi_add_mesh: %s\n", e.what());
    return -1;
  }
}

extern "C" SVMFEM_API int svmfem_fsi_assemble(int mesh, int type, int tDof, const double* Ag, const double* Yg,
    const double* Dg, const double* Do, const double* Bf, const double* params, double* R)
{
  try {
    if (!st) fail("svmfem_fsi_setup was not called");
    if (mesh < 0 || mesh >= (int)st->meshes.size()) fail("unknown mesh");
    if (tDof > 8) fail("tDof > 8 is not supported");
    trace("fsi assemble");
    const double t0 = now();
    const int tnNo = st->tnNo;
    const int nA = tDof * tnNo;
    if (st->tDof != tDof) {
      for (Vector* v : {&st->Ag, &st->Yg, &st->Dg, &st->Do}) dalloc(*v, nA);
      for (auto* m : {&st->hAg, &st->hYg, &st->hDg, &st->hDo}) m->reset(new Mapped(nA));
      fill(st->Do.Write(), nA, 0.0);
      st->tDof = tDof;
    }
    auto upload = [&](Vector& d, Mapped& h, const double* src, int m) {
      std::memcpy(h.host, src, (size_t)m * sizeof(double));
      dcopy_in(d, h.host, m);
    };
    upload(st->Ag, *st->hAg, Ag, nA);
    upload(st->Yg, *st->hYg, Yg, nA);
    upload(st->Dg, *st->hDg, Dg, nA);
    if (Do) upload(st->Do, *st->hDo, Do, nA);
    upload(st->Bf, *st->hBf, Bf, 3 * tnNo);
    const int dof = (type == 2) ? 3 : 4;
    fill(st->R.Write(), dof * tnNo, 0.0);
    const MeshDev& m = *st->meshes[mesh];
    const double* p = params;
    if (type == 0) {
      FluidParams prm{p[0], p[1], {p[2], p[3], p[4]}, p[5], p[6], p[7], p[8], 0.0, 1};
      assemble_fluid(m, prm);
    } else if (type == 1) {
      StructParams prm{p[0], p[1], {p[2], p[3], p[4]}, p[5], p[6], p[7], p[8], p[9], p[10], p[11]};
      assemble_struct(m, prm);
    } else {
      MeshParams prm{p[0], p[1], p[2], {p[3], p[4], p[5]}, p[6], p[7], p[8], p[9]};
      assemble_mesh(m, prm);
    }
    ddownload(st->R, st->hR->host, dof * tnNo);
    for (int i = 0; i < dof * tnNo; i++) R[i] += st->hR->host[i];
    st->t_assem += now() - t0;
    if (++st->n_assem % 30 == 0 && getenv("SVMFEM_TIMING")) {
      printf("SVMFEM fsi assemble per call %.4f s\n", st->t_assem / st->n_assem);
    }
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "svmfem_fsi_assemble: %s\n", e.what());
    return 1;
  }
}

extern "C" SVMFEM_API int svmfem_add_val_dof(int n, int dof, const int* ptr, const double* vals)
{
  try {
    if (n == 0) return 0;
    if (!st) fail("svmfem_fsi_setup was not called");
    const int dd = dof * dof;
    if (n > st->stage_cap) {
      st->stage_cap = 2 * n;
      delete st->stage_ptr;
      delete st->stage_val;
      st->stage_ptr = new Array<int>();
      st->stage_val = new Vector();
      dalloc(*st->stage_ptr, st->stage_cap);
      dalloc(*st->stage_val, 16 * st->stage_cap);
    }
    {
      int* p = st->stage_ptr->HostWrite();
      std::memcpy(p, ptr, n * sizeof(int));
      double* v = st->stage_val->HostWrite();
      std::memcpy(v, vals, (size_t)dd * n * sizeof(double));
    }
    add_val(n, dd, st->stage_ptr->Read(), st->stage_val->Read());
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "svmfem_add_val_dof: %s\n", e.what());
    return 1;
  }
}

extern "C" SVMFEM_API int svmfem_precond_diag_dof(int dof, const double* Dmul, double* W)
{
  try {
    Lhs* L = lhs();
    if (!L) fail("svmfem_fsi_setup was not called");
    if (dof != 3 && dof != 4) fail("dof must be 3 or 4");
    trace("fsi precond");
    dcopy_in(L->Dmul, Dmul, dof * L->nNo);
    precond_diag(dof);
    ddownload(L->W, W, dof * L->nNo);
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "svmfem_precond_diag_dof: %s\n", e.what());
    return 1;
  }
}

extern "C" SVMFEM_API int svmfem_ls_solve(int type, int dof, int mItr, int sD, double absTol, double relTol,
    double* R, svmfem_stats* stats)
{
  try {
    if (!st || !lhs()) fail("svmfem_fsi_setup was not called");
    trace("fsi solve");
    if (dof == 4) return solve<4>(ls4, type, mItr, sD, absTol, relTol, R, stats);
    if (dof == 3) return solve<3>(ls3, type, mItr, sD, absTol, relTol, R, stats);
    fail("GMRES and CG only for dof = 3 or 4");
  } catch (const std::exception& e) {
    fprintf(stderr, "svmfem_ls_solve: %s\n", e.what());
    return 1;
  }
  return 1;
}
