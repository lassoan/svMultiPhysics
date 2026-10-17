// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// GPU assembly of the fluid equation (construct_fluid) for linear tetrahedra
// with VMS stabilization and constant viscosity. The element routines
// (fluid_element.inc) are generated from fluid.cpp by gen_fluid.py.

#include "svgpu.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } } while (0)

double* svgpu_dev_val_ptr = nullptr;

namespace {

struct FluidParams {
  double rho, mu, f[3], dt, af, am, gam, brink;
};

struct GaussData {
  int nG;
  double w[4];
  double N[4][4];     // N[g][a]
  double Nxi[3][4];   // parametric derivatives (constant for linear tetrahedra)
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

// nn::gnn for insd = nsd = 3
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

__global__ void k_fluid(int nEl, const int* IEN, const double* x, int tDof, const double* Ag, const double* Yg,
    const double* Bf, FluidParams prm, GaussData gd, const int* ePtr, double* R, double* Val)
{
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= nEl) return;

  double xl[3][4], al[4][4], yl[4][4], bfl[3][4];
  for (int a = 0; a < 4; a++) {
    int Ac = IEN[4 * e + a];
    for (int i = 0; i < 3; i++) {
      xl[i][a] = x[3 * Ac + i];
      bfl[i][a] = Bf[3 * Ac + i];
    }
    for (int i = 0; i < 4; i++) {
      al[i][a] = Ag[tDof * Ac + i];
      yl[i][a] = Yg[tDof * Ac + i];
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
    int rowN = IEN[4 * e + a];
    for (int i = 0; i < 4; i++) atomicAdd(&R[4 * rowN + i], lR[i][a]);
    for (int b = 0; b < 4; b++) {
      int p = ePtr[16 * e + 4 * a + b];
      for (int k = 0; k < 16; k++) atomicAdd(&Val[16 * (size_t)p + k], lK[k][a][b]);
    }
  }
}

struct Fluid {
  int nEl = 0, tnNo = 0, nnz = 0, tDof = 0;
  int *IEN = nullptr, *ePtr = nullptr;
  double *x = nullptr, *Ag = nullptr, *Yg = nullptr, *Bf = nullptr, *R = nullptr, *Val = nullptr;
  double *hR = nullptr, *hVal = nullptr;   // pinned host buffers
  GaussData gd{};
};

Fluid* fl = nullptr;

// Device resident matrix and preconditioner data
int *dRowOf = nullptr, *dColL = nullptr, *dDiag = nullptr;
double *dW = nullptr, *dDmul = nullptr, *dStageVal = nullptr;
int *dStagePtr = nullptr;
int stageCap = 0;

__global__ void k_add_val(int n, const int* ptr, const double* vals, double* Val)
{
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= 16 * n) return;
  int i = q / 16, k = q % 16;
  atomicAdd(&Val[16 * (size_t)ptr[i] + k], vals[q]);
}

// W = 1/sqrt(|diag|) (1 for a zero diagonal), times the Dirichlet factors
__global__ void k_precond_w(int nNo, const int* diag, const double* Val, const double* Dmul, double* W)
{
  int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= nNo) return;
  for (int i = 0; i < 4; i++) {
    double w = Val[16 * (size_t)diag[a] + 5 * i];
    if (w == 0.0) w = 1.0;
    w = 1.0 / sqrt(fabs(w));
    W[4 * a + i] = w * Dmul[4 * a + i];
  }
}

// Val(4r+c, j) *= W(r, row(j)) * W(c, col(j)) (pre_mul then pos_mul)
__global__ void k_precond_scale(int nnz, const int* rowOf, const int* col, const double* W, double* Val)
{
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= 16 * nnz) return;
  int j = q / 16, k = q % 16;
  int r = k / 4, c = k % 4;
  Val[q] = Val[q] * W[4 * rowOf[j] + r] * W[4 * col[j] + c];
}

template <typename T> T* dev_upload(const T* h, size_t n)
{
  T* d;
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

}  // namespace

extern "C" SVGPU_API int svgpu_fluid_setup(int nEl, const int* IEN, int tnNo, const double* x, int nnz,
    const int* rowPtr, const int* colPtr, int nG, const double* w, const double* N, const double* Nxi)
{
  try {
    if (nG > 4) throw std::runtime_error("svgpu: at most 4 Gauss points are supported");
    fl = new Fluid();
    fl->nEl = nEl;
    fl->tnNo = tnNo;
    fl->nnz = nnz;
    fl->IEN = dev_upload(IEN, 4 * (size_t)nEl);
    fl->x = dev_upload(x, 3 * (size_t)tnNo);
    int* dRow = dev_upload(rowPtr, (size_t)tnNo + 1);
    int* dCol = dev_upload(colPtr, (size_t)nnz);
    CUDA_CHECK(cudaMalloc(&fl->ePtr, 16 * (size_t)nEl * sizeof(int)));
    k_elem_ptr<<<(nEl + 127) / 128, 128>>>(nEl, fl->IEN, dRow, dCol, fl->ePtr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dRow);
    cudaFree(dCol);
    CUDA_CHECK(cudaMalloc(&fl->R, 4 * (size_t)tnNo * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&fl->Val, 16 * (size_t)nnz * sizeof(double)));
    svgpu_dev_val_ptr = fl->Val;
    CUDA_CHECK(cudaMallocHost(&fl->hR, 4 * (size_t)tnNo * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&fl->hVal, 16 * (size_t)nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&fl->Bf, 3 * (size_t)tnNo * sizeof(double)));
    fl->gd.nG = nG;
    for (int g = 0; g < nG; g++) {
      fl->gd.w[g] = w[g];
      for (int a = 0; a < 4; a++) fl->gd.N[g][a] = N[a + 4 * g];
    }
    for (int a = 0; a < 4; a++) {
      for (int i = 0; i < 3; i++) fl->gd.Nxi[i][a] = Nxi[i + 3 * a];
    }
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "svgpu_fluid_setup: %s\n", e.what());
    return 1;
  }
}

/// Assemble the fluid equation and add it to R (4 x tnNo) and Val (16 x nnz)
extern "C" SVGPU_API int svgpu_fluid_assemble(int tDof, const double* Ag, const double* Yg, const double* Bf,
    double rho, double mu, const double* f, double dt, double af, double am, double gam, double brink,
    double* R, double* Val)
{
  try {
    if (!fl) throw std::runtime_error("svgpu_fluid_setup was not called");
    const size_t nA = (size_t)tDof * fl->tnNo;
    if (fl->tDof != tDof) {
      if (fl->Ag) { cudaFree(fl->Ag); cudaFree(fl->Yg); }
      CUDA_CHECK(cudaMalloc(&fl->Ag, nA * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&fl->Yg, nA * sizeof(double)));
      fl->tDof = tDof;
    }
    CUDA_CHECK(cudaMemcpy(fl->Ag, Ag, nA * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(fl->Yg, Yg, nA * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(fl->Bf, Bf, 3 * (size_t)fl->tnNo * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(fl->R, 0, 4 * (size_t)fl->tnNo * sizeof(double)));
    CUDA_CHECK(cudaMemset(fl->Val, 0, 16 * (size_t)fl->nnz * sizeof(double)));
    FluidParams prm{rho, mu, {f[0], f[1], f[2]}, dt, af, am, gam, brink};
    k_fluid<<<(fl->nEl + 127) / 128, 128>>>(fl->nEl, fl->IEN, fl->x, tDof, fl->Ag, fl->Yg, fl->Bf, prm, fl->gd,
        fl->ePtr, fl->R, fl->Val);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(fl->hR, fl->R, 4 * (size_t)fl->tnNo * sizeof(double), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < 4 * (size_t)fl->tnNo; i++) R[i] += fl->hR[i];
    // Val stays on the device when it is null
    if (Val) {
      CUDA_CHECK(cudaMemcpy(fl->hVal, fl->Val, 16 * (size_t)fl->nnz * sizeof(double), cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < 16 * (size_t)fl->nnz; i++) Val[i] += fl->hVal[i];
    }
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "svgpu_fluid_assemble: %s\n", e.what());
    return 1;
  }
}

/// Add matrix entries (16 values per entry, at the given entry positions) to
/// the device resident matrix
extern "C" SVGPU_API int svgpu_add_val(int n, const int* ptr, const double* vals)
{
  try {
    if (n == 0) return 0;
    if (n > stageCap) {
      if (dStagePtr) { cudaFree(dStagePtr); cudaFree(dStageVal); }
      stageCap = 2 * n;
      CUDA_CHECK(cudaMalloc(&dStagePtr, stageCap * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&dStageVal, 16 * (size_t)stageCap * sizeof(double)));
    }
    CUDA_CHECK(cudaMemcpy(dStagePtr, ptr, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dStageVal, vals, 16 * (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    k_add_val<<<(16 * n + 255) / 256, 256>>>(n, dStagePtr, dStageVal, fl->Val);
    CUDA_CHECK(cudaGetLastError());
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "svgpu_add_val: %s\n", e.what());
    return 1;
  }
}

/// FSILS diagonal preconditioner (precond_diag, dof = 4) of the device
/// resident matrix. rowPtr: 2 x nNo, colPtr: nnz, diagPtr: nNo (FSILS lhs
/// ordering), Dmul: 4 x nNo Dirichlet factors. Returns W (4 x nNo).
extern "C" SVGPU_API int svgpu_precond_diag(int nNo, int nnz, const int* rowPtr, const int* colPtr,
    const int* diagPtr, const double* Dmul, double* W)
{
  try {
    if (!dRowOf) {
      std::vector<int> rowOf(nnz);
      for (int a = 0; a < nNo; a++) {
        for (int j = rowPtr[2 * a]; j <= rowPtr[2 * a + 1]; j++) rowOf[j] = a;
      }
      dRowOf = dev_upload(rowOf.data(), (size_t)nnz);
      dColL = dev_upload(colPtr, (size_t)nnz);
      dDiag = dev_upload(diagPtr, (size_t)nNo);
      CUDA_CHECK(cudaMalloc(&dW, 4 * (size_t)nNo * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&dDmul, 4 * (size_t)nNo * sizeof(double)));
    }
    CUDA_CHECK(cudaMemcpy(dDmul, Dmul, 4 * (size_t)nNo * sizeof(double), cudaMemcpyHostToDevice));
    k_precond_w<<<(nNo + 255) / 256, 256>>>(nNo, dDiag, fl->Val, dDmul, dW);
    CUDA_CHECK(cudaGetLastError());
    k_precond_scale<<<(int)((16 * (size_t)nnz + 255) / 256), 256>>>(nnz, dRowOf, dColL, dW, fl->Val);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(W, dW, 4 * (size_t)nNo * sizeof(double), cudaMemcpyDeviceToHost));
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "svgpu_precond_diag: %s\n", e.what());
    return 1;
  }
}
