// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// GPU data and operations used by the GPU implementation of the FSILS NS
// solver (svgpu_ns.cpp). All vectors are device pointers, with the FSILS
// (column-major, dof x nNo) layout.

#ifndef SVGPU_INTERNAL_H
#define SVGPU_INTERNAL_H

#include <vector>

namespace svgpu {

struct CgState;

class Gpu {
 public:
  struct CgResult { bool success; int itr; double iNorm, fNorm, dB; };
  // Matrix structure (constant)
  int nNo = 0, nnz = 0;
  int *rowS = nullptr, *rowE = nullptr, *col = nullptr, *tpos = nullptr;

  // Matrix blocks (nsd = 3, dof = 4)
  double *Val = nullptr, *mK = nullptr, *mG = nullptr, *mD = nullptr, *mL = nullptr, *Gt = nullptr;

  // Coupled faces: concatenated nodes and values, offsets, preconditioner and
  // multiplication coefficients
  int nCoupled = 0;
  int *fGlob = nullptr, *fOff = nullptr;
  double *fValM = nullptr, *fCoefAdd = nullptr, *fCoefPre = nullptr;
  int fCapacity = 0;

  // Reduction workspace and device scalars
  double* partial = nullptr;
  double* scal = nullptr;     // general purpose device scalars
  int nScal = 0;

  // Schur complement CG
  double *cgX = nullptr, *cgP = nullptr, *cgSP = nullptr, *cgGP = nullptr, *cgR = nullptr;
  CgState* cgState = nullptr;
  void* cgGraph = nullptr;     // cudaGraphExec_t
  int cgGraphItr = -1;
  bool cgGraphCoupled = false;

  // Explicit Schur complement S = L - Gt*G (CSR) and its rank-1 corrections
  // of the resistance boundary conditions, for the Schur complement CG
  std::vector<int> hRowS, hRowE, hCol;
  int nnzS = 0;
  int *sRow = nullptr, *sCol = nullptr, *sLidx = nullptr, *sPairPtr = nullptr, *sPairJ = nullptr, *sPairL = nullptr;
  float *sVal = nullptr;
  double *fU = nullptr, *fW = nullptr, *fVd = nullptr;   // nCoupled x nNo (fVd: nCoupled x 3*nNo)
  double *cgSPp = nullptr, *cgPart = nullptr;
  // One Arnoldi step of gmres (u_{i+1} from u_i), Hessenberg column to hcol (host)
  void arnoldi(double* u, int i, double* hcol);
  void schur_setup();
  void schur_numeric();
  CgResult cg_schur_s(double* R, int mItr, double absTol, double relTol);

  ~Gpu();

  void setup(int nNo, int nnz, const int* rowPtr, const int* colPtr);
  void set_faces(const std::vector<int>& glob, const std::vector<int>& off, const std::vector<double>& valM,
      const std::vector<double>& coefAdd, const std::vector<double>& coefPre);

  double* alloc(size_t n);
  void zero(double* x, size_t n);
  void copy(double* dst, const double* src, size_t n);
  void upload(double* dst, const double* src, size_t n);
  void download(double* dst, const double* src, size_t n);
  void sync();

  // Split Val into the blocks of the NS system
  void depart();

  // y = K*x for the different block types (nsd = 3)
  void spmv_vv(const double* K, const double* x, double* y);
  void spmv_sv(const double* K, const double* x, double* y);
  void spmv_vs(const double* K, const double* x, double* y);
  void spmv_ss(const double* K, const double* x, double* y);

  // Resistance boundary conditions (nsd = 3): y += valM*coef*(valM . x)
  void bc_add(const double* x, double* y);
  // In-place preconditioner of the resistance boundary conditions
  void bc_pre(double* y);

  // Dot product of length n vectors, result to the device scalar dst
  void dot_dev(const double* a, const double* b, int n, double* dst);
  double dot(const double* a, const double* b, int n);

  // y = y + alpha*x, alpha on the host or on the device (with sign)
  void axpy(double alpha, const double* x, double* y, int n);
  void axpy_dev(const double* alpha, double sign, const double* x, double* y, int n);
  // x = x / s (s on the device)
  void scale_inv_dev(const double* s, double* x, int n);
  void scale(double a, double* x, int n);
  // z = x - y
  void sub(const double* x, const double* y, double* z, int n);
  // y = x0 - sum_j c[j]*xs[j] (sequential order), x0 may be null (zero)
  void lin_comb(const double* x0, double sign, const double* c, const std::vector<const double*>& xs,
      double* y, int n);
  // Copy rows 0..2 of a dof=4 array to a dof=3 array, and row 3 to a scalar array
  void split4(const double* R, double* Rm, double* Rc);
  void merge4(const double* Rm, const double* Rc, double* R);

  // Schur complement CG (in place on R of length nNo), returns iterations
  CgResult cg_schur(double* R, int mItr, double absTol, double relTol);
};

// Two-level AMG preconditioned Schur complement CG (svgpu_amg.cu)
void amg_setup_impl(Gpu& g, const std::vector<int>& sRowH, const std::vector<int>& sColH);
void amg_numeric_impl(Gpu& g, const int* sRow, const int* sCol, const float* sVal, int nnzS);
Gpu::CgResult cg_schur_amg_impl(Gpu& g, double* R, int mItr, double absTol, double relTol);

}

#endif
