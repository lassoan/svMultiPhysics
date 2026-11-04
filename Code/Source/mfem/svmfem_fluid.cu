// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// Assembly of the fluid equation (construct_fluid) for linear tetrahedra
// with VMS stabilization and constant viscosity, the device resident matrix
// and the FSILS diagonal preconditioner, as mfem::forall kernels. The
// element routines (fluid_element.inc) are generated from fluid.cpp by
// gen_fluid.py with MFEM_HOST_DEVICE, so the same code runs on the CPU
// device.

#include "svmfem.h"
#include "svmfem_internal.hpp"

#include <cfloat>
#include <cmath>
#include <cstring>

namespace svmfem {

namespace {

struct FluidParams {
  double rho, mu, f[3], dt, af, am, gam, brink;
  int mvMsh;
};

struct GaussData {
  int nG;
  double w[4];
  double N[4][4];     // N[g][a]
  double Nxi[3][4];   // parametric derivatives (constant for linear tetrahedra)
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

// nn::gnn for insd = nsd = 3
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

/// Mesh and state of the assembly
struct Fluid {
  int nEl = 0, tnNo = 0, tDof = 0;
  Array<int> IEN, ePtr;        // 4 x nEl, 16 x nEl (matrix entry of each node pair)
  Vector x;                    // 3 x tnNo
  Vector Ag, Yg, Bf, R;        // device state (tDof x tnNo, 3 x tnNo, 4 x tnNo)
  Mapped *hAg = nullptr, *hYg = nullptr, *hBf = nullptr, *hR = nullptr;   // pinned staging
  GaussData gd{};
  double t_assem = 0.0;
  int n_assem = 0;
};

Fluid* fl = nullptr;
Lhs* the_lhs = nullptr;

// Staging of host matrix entries (face terms)
Array<int>* stage_ptr = nullptr;
Vector* stage_val = nullptr;
int stage_cap = 0;

}  // namespace

Lhs* lhs()
{
  return the_lhs;
}

Lhs* make_lhs(int nNo, int nnz, const int* rowPtr, const int* colPtr)
{
  // Matrix structure
  Lhs* L = new Lhs();
  L->nNo = nNo;
  L->nnz = nnz;
  L->h_off.assign(rowPtr, rowPtr + nNo + 1);
  L->h_col.assign(colPtr, colPtr + nnz);
  std::vector<int> rows(nnz), diag(nNo, -1), tpos(nnz, -1);
  for (int a = 0; a < nNo; a++) {
    for (int j = rowPtr[a]; j < rowPtr[a + 1]; j++) {
      rows[j] = a;
      if (colPtr[j] == a) diag[a] = j;
    }
  }
  for (int a = 0; a < nNo; a++) {
    if (diag[a] < 0) fail("a row without diagonal entry");
    for (int j = rowPtr[a]; j < rowPtr[a + 1]; j++) {
      // Binary search of a in row colPtr[j] (sorted columns)
      int c = colPtr[j];
      int lo = rowPtr[c], hi = rowPtr[c + 1] - 1;
      while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (colPtr[mid] < a) lo = mid + 1; else hi = mid;
      }
      if (colPtr[lo] != a) fail("the matrix pattern is not symmetric");
      tpos[j] = lo;
    }
  }
  dupload(L->offsets, L->h_off);
  dupload(L->columns, L->h_col);
  dupload(L->rows, rows);
  dupload(L->diag, diag);
  dupload(L->tpos, tpos);
  dalloc(L->Val, 16 * nnz);
  fill(L->Val.Write(), 16 * nnz, 0.0);
  dalloc(L->W, 4 * nNo);
  dalloc(L->Dmul, 4 * nNo);
  delete the_lhs;
  the_lhs = L;
  return L;
}


void destroy_fluid()
{
  if (fl) {
    delete fl->hAg;
    delete fl->hYg;
    delete fl->hBf;
    delete fl->hR;
  }
  delete fl;
  fl = nullptr;
  delete the_lhs;
  the_lhs = nullptr;
  delete stage_ptr;
  delete stage_val;
  stage_ptr = nullptr;
  stage_val = nullptr;
  stage_cap = 0;
}

}  // namespace svmfem

using namespace svmfem;

extern "C" SVMFEM_API int svmfem_fluid_setup(int nEl, const int* IEN, int tnNo, const double* x, int nnz,
    const int* rowPtr, const int* colPtr, int nG, const double* w, const double* N, const double* Nxi)
{
  try {
    init_device(nullptr);
    if (nG > 4) fail("at most 4 Gauss points are supported");
    double t0 = now();
    fl = new Fluid();
    fl->nEl = nEl;
    fl->tnNo = tnNo;
    dupload(fl->IEN, IEN, 4 * nEl);
    dupload(fl->x, x, 3 * tnNo);
    Lhs* L = make_lhs(tnNo, nnz, rowPtr, colPtr);
    // Matrix entry of every node pair of every element
    dalloc(fl->ePtr, 16 * nEl);
    {
      const int* ien = fl->IEN.Read();
      const int* off = L->offsets.Read();
      const int* col = L->columns.Read();
      int* ePtr = fl->ePtr.Write();
      launch(nEl, [=] MFEM_HOST_DEVICE (int e) {
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
    dalloc(fl->R, 4 * tnNo);
    dalloc(fl->Bf, 3 * tnNo);
    fl->hR = new Mapped(4 * tnNo);
    fl->hBf = new Mapped(3 * tnNo);
    fl->gd.nG = nG;
    for (int g = 0; g < nG; g++) {
      fl->gd.w[g] = w[g];
      for (int a = 0; a < 4; a++) fl->gd.N[g][a] = N[a + 4 * g];
    }
    for (int a = 0; a < 4; a++) {
      for (int i = 0; i < 3; i++) fl->gd.Nxi[i][a] = Nxi[i + 3 * a];
    }
    sync();
    if (verbose()) printf("svmfem: fluid setup nEl %d tnNo %d nnz %d nG %d in %.2f s\n", nEl, tnNo, nnz, nG, now() - t0);
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "svmfem_fluid_setup: %s\n", e.what());
    return 1;
  }
}

extern "C" SVMFEM_API int svmfem_fluid_assemble(int tDof, const double* Ag, const double* Yg, const double* Bf,
    double rho, double mu, const double* f, double dt, double af, double am, double gam, double brink, double* R)
{
  try {
    if (!fl) fail("svmfem_fluid_setup was not called");
    trace("assemble");
    double t0 = now();
    const int tnNo = fl->tnNo, nEl = fl->nEl;
    const int nA = tDof * tnNo;
    if (fl->tDof != tDof) {
      dalloc(fl->Ag, nA);
      dalloc(fl->Yg, nA);
      delete fl->hAg;
      delete fl->hYg;
      fl->hAg = new Mapped(nA);
      fl->hYg = new Mapped(nA);
      fl->tDof = tDof;
    }
    // Inputs through pinned staging buffers
    std::memcpy(fl->hAg->host, Ag, nA * sizeof(double));
    std::memcpy(fl->hYg->host, Yg, nA * sizeof(double));
    std::memcpy(fl->hBf->host, Bf, 3 * (size_t)tnNo * sizeof(double));
    dcopy_in(fl->Ag, fl->hAg->host, nA);
    dcopy_in(fl->Yg, fl->hYg->host, nA);
    dcopy_in(fl->Bf, fl->hBf->host, 3 * tnNo);
    Lhs* L = the_lhs;
    fill(L->Val.Write(), 16 * L->nnz, 0.0);
    fill(fl->R.Write(), 4 * tnNo, 0.0);
    const FluidParams prm{rho, mu, {f[0], f[1], f[2]}, dt, af, am, gam, brink, 0};
    const GaussData gd = fl->gd;
    const int* ien = fl->IEN.Read();
    const int* ePtr = fl->ePtr.Read();
    const double* x = fl->x.Read();
    const double* dAg = fl->Ag.Read();
    const double* dYg = fl->Yg.Read();
    const double* dBf = fl->Bf.Read();
    double* dR = fl->R.ReadWrite();
    double* Val = L->Val.ReadWrite();
    launch(nEl, [=] MFEM_HOST_DEVICE (int e) {
      double xl[3][4], al[8][4] = {}, yl[8][4] = {}, bfl[3][4];
      for (int a = 0; a < 4; a++) {
        int Ac = ien[4 * e + a];
        for (int i = 0; i < 3; i++) {
          xl[i][a] = x[3 * Ac + i];
          bfl[i][a] = dBf[3 * Ac + i];
        }
        for (int i = 0; i < 4; i++) {
          al[i][a] = dAg[tDof * Ac + i];
          yl[i][a] = dYg[tDof * Ac + i];
        }
      }
      double lR[4][4] = {};
      double lK[16][4][4] = {};
      double Nx[3][4], ks[3][3], Jac;
      const double Nxx[6][4] = {};
      const double uris[3] = {};
      d_gnn(gd.Nxi, xl, Nx, Jac, ks);
      for (int g = 0; g < gd.nG; g++) {
        double w = gd.w[g] * Jac;
        d_fluid_3d_m(prm, 1, 4, 4, w, ks, gd.N[g], gd.N[g], Nx, Nx, Nxx, al, yl, bfl, lR, lK, prm.brink, 0.0, uris);
      }
      for (int g = 0; g < gd.nG; g++) {
        double w = gd.w[g] * Jac;
        d_fluid_3d_c(prm, 1, 4, 4, w, ks, gd.N[g], gd.N[g], Nx, Nx, Nxx, al, yl, bfl, lR, lK, prm.brink, 0.0, uris);
      }
      for (int a = 0; a < 4; a++) {
        int rowN = ien[4 * e + a];
        for (int i = 0; i < 4; i++) AtomicAdd(dR[4 * rowN + i], lR[i][a]);
        for (int b = 0; b < 4; b++) {
          int p = ePtr[16 * e + 4 * a + b];
          for (int k = 0; k < 16; k++) AtomicAdd(Val[16 * (size_t)p + k], lK[k][a][b]);
        }
      }
    });
    ddownload(fl->R, fl->hR->host, 4 * tnNo);
    for (int i = 0; i < 4 * tnNo; i++) R[i] += fl->hR->host[i];
    trace("assemble done");
    fl->t_assem += now() - t0;
    if (++fl->n_assem % 10 == 0 && getenv("SVMFEM_TIMING")) {
      printf("SVMFEM assemble per call %.4f s\n", fl->t_assem / fl->n_assem);
    }
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "svmfem_fluid_assemble: %s\n", e.what());
    return 1;
  }
}

extern "C" SVMFEM_API int svmfem_add_val(int n, const int* ptr, const double* vals)
{
  try {
    if (n == 0) return 0;
    if (!the_lhs) fail("svmfem_fluid_setup was not called");
    if (n > stage_cap) {
      stage_cap = 2 * n;
      delete stage_ptr;
      delete stage_val;
      stage_ptr = new Array<int>();
      stage_val = new Vector();
      dalloc(*stage_ptr, stage_cap);
      dalloc(*stage_val, 16 * stage_cap);
    }
    // Host -> device through the (pageable) host side of the MFEM arrays
    {
      int* p = stage_ptr->HostWrite();
      std::memcpy(p, ptr, n * sizeof(int));
      double* v = stage_val->HostWrite();
      std::memcpy(v, vals, 16 * (size_t)n * sizeof(double));
    }
    const int* dptr = stage_ptr->Read();
    const double* dv = stage_val->Read();
    double* Val = the_lhs->Val.ReadWrite();
    launch(16 * n, [=] MFEM_HOST_DEVICE (int q) {
      int i = q / 16, k = q % 16;
      AtomicAdd(Val[16 * (size_t)dptr[i] + k], dv[q]);
    });
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "svmfem_add_val: %s\n", e.what());
    return 1;
  }
}

extern "C" SVMFEM_API int svmfem_precond_diag(const double* Dmul, double* W)
{
  try {
    Lhs* L = the_lhs;
    if (!L) fail("svmfem_fluid_setup was not called");
    trace("precond");
    const int nNo = L->nNo, nnz = L->nnz;
    dcopy_in(L->Dmul, Dmul, 4 * nNo);
    const int* diag = L->diag.Read();
    const int* rows = L->rows.Read();
    const int* col = L->columns.Read();
    const double* dDmul = L->Dmul.Read();
    double* Val = L->Val.ReadWrite();
    double* dW = L->W.Write();
    // W = 1/sqrt(|diag|) (1 for a zero diagonal), times the Dirichlet factors
    launch(nNo, [=] MFEM_HOST_DEVICE (int a) {
      for (int i = 0; i < 4; i++) {
        double w = Val[16 * (size_t)diag[a] + 5 * i];
        if (w == 0.0) w = 1.0;
        w = 1.0 / sqrt(fabs(w));
        dW[4 * a + i] = w * dDmul[4 * a + i];
      }
    });
    // Val(4r+c, j) *= W(r, row(j)) * W(c, col(j)) (pre_mul then pos_mul)
    launch(16 * nnz, [=] MFEM_HOST_DEVICE (int q) {
      int j = q / 16, k = q % 16;
      int r = k / 4, c = k % 4;
      Val[q] = Val[q] * dW[4 * rows[j] + r] * dW[4 * col[j] + c];
    });
    ddownload(L->W, W, 4 * nNo);
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "svmfem_precond_diag: %s\n", e.what());
    return 1;
  }
}
