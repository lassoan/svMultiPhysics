// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// CUDA kernels of the GPU implementation of the FSILS NS solver.

#include "svgpu_internal.h"

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <cstdlib>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } } while (0)
#define KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())

namespace svgpu {

namespace {

constexpr int TPB = 256;      // threads per block
constexpr int NRED = 120;     // blocks of reductions
constexpr int NPART = 1024;   // capacity of the partial sums
constexpr int LANES = 8;      // threads per matrix row

inline int blocks(int n) { return (n + TPB - 1) / TPB; }

__device__ inline double block_sum(double v)
{
  __shared__ double sh[TPB];
  sh[threadIdx.x] = v;
  __syncthreads();
  for (int s = TPB / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  double r = sh[0];
  __syncthreads();
  return r;
}


// Sum over the LANES threads of a matrix row (consecutive lanes of a warp)
__device__ inline double row_sum(double v)
{
  for (int o = LANES / 2; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o, LANES);
  return v;
}

//---------------------------------------------------------------------------
// Sparse matrix-vector products (thread per row). 'done' (optional) skips the
// kernel when the CG iteration converged.

__global__ void k_spmv_vv(int nNo, const int* rowS, const int* rowE, const int* col, const double* K,
    const double* U, double* KU)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nNo) return;
  double s0 = 0.0, s1 = 0.0, s2 = 0.0;
  for (int j = rowS[i]; j <= rowE[i]; j++) {
    int c = col[j];
    const double* k = K + 9 * (size_t)j;
    double u0 = U[3 * c], u1 = U[3 * c + 1], u2 = U[3 * c + 2];
    s0 = s0 + k[0] * u0 + k[1] * u1 + k[2] * u2;
    s1 = s1 + k[3] * u0 + k[4] * u1 + k[5] * u2;
    s2 = s2 + k[6] * u0 + k[7] * u1 + k[8] * u2;
  }
  KU[3 * i] = s0; KU[3 * i + 1] = s1; KU[3 * i + 2] = s2;
}

__global__ void k_spmv_sv(int nNo, const int* rowS, const int* rowE, const int* col, const double* K,
    const double* U, double* KU, const int* done)
{
  if (done && *done) return;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nNo) return;
  double s0 = 0.0, s1 = 0.0, s2 = 0.0;
  for (int j = rowS[i]; j <= rowE[i]; j++) {
    double u = U[col[j]];
    const double* k = K + 3 * (size_t)j;
    s0 += k[0] * u;
    s1 += k[1] * u;
    s2 += k[2] * u;
  }
  KU[3 * i] = s0; KU[3 * i + 1] = s1; KU[3 * i + 2] = s2;
}

__global__ void k_spmv_vs(int nNo, const int* rowS, const int* rowE, const int* col, const double* K,
    const double* U, double* KU)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nNo) return;
  double s = 0.0;
  for (int j = rowS[i]; j <= rowE[i]; j++) {
    int c = col[j];
    const double* k = K + 3 * (size_t)j;
    s = s + k[0] * U[3 * c] + k[1] * U[3 * c + 1] + k[2] * U[3 * c + 2];
  }
  KU[i] = s;
}

__global__ void k_spmv_ss(int nNo, const int* rowS, const int* rowE, const int* col, const double* K,
    const double* U, double* KU)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nNo) return;
  double s = 0.0;
  for (int j = rowS[i]; j <= rowE[i]; j++) {
    s = s + K[j] * U[col[j]];
  }
  KU[i] = s;
}

//---------------------------------------------------------------------------
// Split Val (16 x nnz) into the blocks of the NS system

__global__ void k_depart(int nnz, const double* Val, double* mK, double* mG, double* mD, double* mL)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nnz) return;
  const double* t = Val + 16 * (size_t)j;
  double* k = mK + 9 * (size_t)j;
  k[0] = t[0]; k[1] = t[1]; k[2] = t[2];
  k[3] = t[4]; k[4] = t[5]; k[5] = t[6];
  k[6] = t[8]; k[7] = t[9]; k[8] = t[10];
  mG[3 * j] = t[3]; mG[3 * j + 1] = t[7]; mG[3 * j + 2] = t[11];
  mD[3 * j] = t[12]; mD[3 * j + 1] = t[13]; mD[3 * j + 2] = t[14];
  mL[j] = t[15];
}

__global__ void k_transpose_g(int nnz, const int* tpos, const double* mG, double* Gt)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nnz) return;
  int l = tpos[j];
  if (l < 0) return;
  Gt[3 * l] = -mG[3 * j];
  Gt[3 * l + 1] = -mG[3 * j + 1];
  Gt[3 * l + 2] = -mG[3 * j + 2];
}

//---------------------------------------------------------------------------
// Resistance boundary conditions, one block per coupled face

__global__ void k_bc(const int* fOff, const int* fGlob, const double* fValM, const double* coef,
    const double* X, double* Y, const int* done)
{
  if (done && *done) return;
  int f = blockIdx.x;
  int a0 = fOff[f], a1 = fOff[f + 1];
  double s = 0.0;
  for (int a = a0 + threadIdx.x; a < a1; a += blockDim.x) {
    int Ac = fGlob[a];
    s += fValM[3 * a] * X[3 * Ac] + fValM[3 * a + 1] * X[3 * Ac + 1] + fValM[3 * a + 2] * X[3 * Ac + 2];
  }
  double S = coef[f] * block_sum(s);
  for (int a = a0 + threadIdx.x; a < a1; a += blockDim.x) {
    int Ac = fGlob[a];
    Y[3 * Ac] += fValM[3 * a] * S;
    Y[3 * Ac + 1] += fValM[3 * a + 1] * S;
    Y[3 * Ac + 2] += fValM[3 * a + 2] * S;
  }
}

//---------------------------------------------------------------------------
// Reductions

__global__ void k_dot_partial(int n, const double* a, const double* b, double* partial)
{
  double s = 0.0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    s += a[i] * b[i];
  }
  s = block_sum(s);
  if (threadIdx.x == 0) partial[blockIdx.x] = s;
}

__global__ void k_finish(int np, const double* partial, double* dst, int op)
{
  double s = 0.0;
  for (int i = threadIdx.x; i < np; i += blockDim.x) s += partial[i];
  s = block_sum(s);
  if (threadIdx.x == 0) *dst = (op == 1) ? sqrt(s) : s;
}

//---------------------------------------------------------------------------
// Vector operations

__global__ void k_axpy(int n, double alpha, const double* x, double* y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = y[i] + alpha * x[i];
}

__global__ void k_axpy_dev(int n, const double* alpha, double sign, const double* x, double* y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = y[i] + (sign * *alpha) * x[i];
}

__global__ void k_scale_inv_dev(int n, const double* s, double* x)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = x[i] * (1.0 / *s);
}

__global__ void k_scale(int n, double a, double* x)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = a * x[i];
}

__global__ void k_sub(int n, const double* x, const double* y, double* z)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) z[i] = x[i] - y[i];
}

struct PtrList { const double* p[64]; };

__global__ void k_lin_comb(int n, const double* x0, double sign, const double* c, int m, PtrList xs, double* y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double v = x0 ? x0[i] : 0.0;
  for (int j = 0; j < m; j++) {
    v = v + sign * (c[j] * xs.p[j][i]);
  }
  y[i] = v;
}

__global__ void k_lin_dinv(int n, const double* d, const double* r, double* z)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) z[i] = d[i] * r[i];
}

__global__ void k_split4(int nNo, const double* R, double* Rm, double* Rc)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nNo) return;
  Rm[3 * i] = R[4 * i]; Rm[3 * i + 1] = R[4 * i + 1]; Rm[3 * i + 2] = R[4 * i + 2];
  Rc[i] = R[4 * i + 3];
}

__global__ void k_merge4(int nNo, const double* Rm, const double* Rc, double* R)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nNo) return;
  R[4 * i] = Rm[3 * i]; R[4 * i + 1] = Rm[3 * i + 1]; R[4 * i + 2] = Rm[3 * i + 2];
  R[4 * i + 3] = Rc[i];
}

}  // namespace

//---------------------------------------------------------------------------
// Schur complement CG, executed as one CUDA graph without host
// synchronization. The CG state is kept on the device.

struct CgState {
  double err, errO, eps, alpha, c1, c2;
  int done, success, last_i;
};

namespace {

__global__ void k_cg_top(CgState* st, int it)
{
  if (st->done) return;
  st->last_i = it;
  if (st->err < st->eps) {
    st->done = 1;
    st->success = 1;
    return;
  }
  st->errO = st->err;
}

// SP = L*P - D*GP, and partial sums of P.SP
__global__ void k_cg_sp(int nNo, const int* rowS, const int* rowE, const int* col, const double* L,
    const double* D, const double* P, const double* GP, double* SP, double* partial, const CgState* st)
{
  if (st->done) return;
  double prod = 0.0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNo; i += gridDim.x * blockDim.x) {
    double ls = 0.0, ds = 0.0;
    for (int j = rowS[i]; j <= rowE[i]; j++) {
      int c = col[j];
      ls = ls + L[j] * P[c];
      const double* k = D + 3 * (size_t)j;
      ds = ds + k[0] * GP[3 * c] + k[1] * GP[3 * c + 1] + k[2] * GP[3 * c + 2];
    }
    double sp = ls - ds;
    SP[i] = sp;
    prod += P[i] * sp;
  }
  prod = block_sum(prod);
  if (threadIdx.x == 0) partial[blockIdx.x] = prod;
}

__global__ void k_cg_alpha(int np, const double* partial, CgState* st)
{
  if (st->done) return;
  double s = 0.0;
  for (int i = threadIdx.x; i < np; i += blockDim.x) s += partial[i];
  s = block_sum(s);
  if (threadIdx.x == 0) st->alpha = st->errO / s;
}

// X += alpha*P, R -= alpha*SP, partial sums of R.R
__global__ void k_cg_update(int nNo, double* X, const double* P, double* R, const double* SP, double* partial,
    const CgState* st)
{
  if (st->done) return;
  double alpha = st->alpha;
  double s = 0.0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNo; i += gridDim.x * blockDim.x) {
    X[i] = X[i] + alpha * P[i];
    double r = R[i] + (-alpha) * SP[i];
    R[i] = r;
    s += r * r;
  }
  s = block_sum(s);
  if (threadIdx.x == 0) partial[blockIdx.x] = s;
}

__global__ void k_cg_err(int np, const double* partial, CgState* st)
{
  if (st->done) return;
  double s = 0.0;
  for (int i = threadIdx.x; i < np; i += blockDim.x) s += partial[i];
  s = block_sum(s);
  if (threadIdx.x == 0) {
    st->err = s;
    st->c1 = st->errO / s;
    st->c2 = s / st->errO;
  }
}

__global__ void k_cg_p(int nNo, double* P, const double* R, const CgState* st)
{
  if (st->done) return;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nNo) return;
  double t = P[i] + st->c1 * R[i];
  P[i] = st->c2 * t;
}


// Inverse of the diagonal of S = L - D*G (D = Gt), for the Jacobi
// preconditioner of the Schur complement CG
__global__ void k_schur_dinv(int nNo, const int* rowS, const int* rowE, const int* col, const int* tpos,
    const double* L, const double* D, const double* G, double* dinv)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nNo) return;
  double d = 0.0;
  for (int j = rowS[i]; j <= rowE[i]; j++) {
    if (col[j] == i) d += L[j];
    int l = tpos[j];
    if (l >= 0) d -= D[3 * j] * G[3 * l] + D[3 * j + 1] * G[3 * l + 1] + D[3 * j + 2] * G[3 * l + 2];
  }
  dinv[i] = (d > 0.0) ? 1.0 / d : 1.0;
}

//---------------------------------------------------------------------------
// Schur complement CG as one cooperative kernel: all iterations in one
// launch, with grid synchronizations between the phases. The reductions are
// finished redundantly by every block (same order, so the same result).

namespace cg = cooperative_groups;

struct CoopArgs {
  int nNo, mItr, nCoupled;
  const int *rowS, *rowE, *col;
  const double *G, *D, *L;
  double *X, *P, *SP, *GP, *R;
  const int *fOff, *fGlob;
  const double *fValM, *fCoef;
  double *partA, *partB, *faceS;
  CgState* st;
  const double* dinv;   // Jacobi preconditioner (null: no preconditioning)
  double *partC;
  double rz0;
  unsigned long long* tph;   // optional phase timers (6 accumulators)
};

__device__ inline unsigned long long gtimer()
{
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define PHASE(k) if (a.tph && blockIdx.x == 0 && threadIdx.x == 0) { unsigned long long t_ = gtimer(); a.tph[k] += t_ - tlast; tlast = t_; }

__device__ inline double sum_partials(const double* p, int n)
{
  double s = 0.0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) s += p[i];
  return block_sum(s);
}

__global__ void k_cg_coop(CoopArgs a)
{
  cg::grid_group grid = cg::this_grid();
  const int nb = gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nth = nb * blockDim.x;
  const int lane = threadIdx.x % LANES;
  const int gid0 = tid / LANES;
  const int ngroups = nth / LANES;
  // Row loop bound rounded up to a multiple of the number of groups, so that
  // all lanes of a warp do the same number of iterations
  const int nRowIt = ((a.nNo + ngroups - 1) / ngroups) * ngroups;
  double err = a.st->err, errO = a.st->errO;
  const double eps = a.st->eps;
  const bool prec = a.dinv != nullptr;
  double rz = a.rz0;
  int last_i = 0, success = 0;
  unsigned long long tlast = (a.tph && blockIdx.x == 0 && threadIdx.x == 0) ? gtimer() : 0;

  for (int it = 0; it < a.mItr; it++) {
    last_i = it;
    if (err < eps) { success = 1; break; }
    errO = err;

    // GP = G*P (LANES threads per row; the row loop has the same trip count
    // in all lanes of a warp, so that the shuffles are not divergent)
    for (int i0 = gid0; i0 < nRowIt; i0 += ngroups) {
      int i = i0;
      double s0 = 0.0, s1 = 0.0, s2 = 0.0;
      if (i < a.nNo) {
        for (int j = a.rowS[i] + lane; j <= a.rowE[i]; j += LANES) {
          double u = a.P[a.col[j]];
          const double* k = a.G + 3 * (size_t)j;
          s0 += k[0] * u; s1 += k[1] * u; s2 += k[2] * u;
        }
      }
      s0 = row_sum(s0); s1 = row_sum(s1); s2 = row_sum(s2);
      if (lane == 0 && i < a.nNo) { a.GP[3 * i] = s0; a.GP[3 * i + 1] = s1; a.GP[3 * i + 2] = s2; }
    }
    grid.sync();
    PHASE(0)

    // Preconditioner of the resistance boundary conditions
    if (a.nCoupled > 0) {
      if (blockIdx.x < a.nCoupled) {
        int f = blockIdx.x;
        double s = 0.0;
        for (int q = a.fOff[f] + threadIdx.x; q < a.fOff[f + 1]; q += blockDim.x) {
          int Ac = a.fGlob[q];
          s += a.fValM[3 * q] * a.GP[3 * Ac] + a.fValM[3 * q + 1] * a.GP[3 * Ac + 1] + a.fValM[3 * q + 2] * a.GP[3 * Ac + 2];
        }
        s = block_sum(s);
        if (threadIdx.x == 0) a.faceS[f] = a.fCoef[f] * s;
      }
      grid.sync();
      for (int f = 0; f < a.nCoupled; f++) {
        double S = a.faceS[f];
        for (int q = a.fOff[f] + tid; q < a.fOff[f + 1]; q += nth) {
          int Ac = a.fGlob[q];
          a.GP[3 * Ac] += a.fValM[3 * q] * S;
          a.GP[3 * Ac + 1] += a.fValM[3 * q + 1] * S;
          a.GP[3 * Ac + 2] += a.fValM[3 * q + 2] * S;
        }
      }
      grid.sync();
    }
    PHASE(1)

    // SP = L*P - D*GP, partial sums of P.SP
    double prod = 0.0;
    for (int i0 = gid0; i0 < nRowIt; i0 += ngroups) {
      int i = i0;
      double ls = 0.0, ds = 0.0;
      if (i < a.nNo) {
        for (int j = a.rowS[i] + lane; j <= a.rowE[i]; j += LANES) {
          int c = a.col[j];
          ls = ls + a.L[j] * a.P[c];
          const double* k = a.D + 3 * (size_t)j;
          ds = ds + k[0] * a.GP[3 * c] + k[1] * a.GP[3 * c + 1] + k[2] * a.GP[3 * c + 2];
        }
      }
      ls = row_sum(ls); ds = row_sum(ds);
      if (lane == 0 && i < a.nNo) {
        double sp = ls - ds;
        a.SP[i] = sp;
        prod += a.P[i] * sp;
      }
    }
    prod = block_sum(prod);
    if (threadIdx.x == 0) a.partA[blockIdx.x] = prod;
    grid.sync();
    PHASE(2)

    if (prec) {
      // Preconditioned CG: alpha = r.z / p.q, p = z + beta*p
      double alpha = rz / sum_partials(a.partA, nb);
      double rr = 0.0, rzn = 0.0;
      for (int i = tid; i < a.nNo; i += nth) {
        a.X[i] = a.X[i] + alpha * a.P[i];
        double r = a.R[i] - alpha * a.SP[i];
        a.R[i] = r;
        rr += r * r;
        rzn += r * r * a.dinv[i];
      }
      rr = block_sum(rr);
      rzn = block_sum(rzn);
      if (threadIdx.x == 0) { a.partB[blockIdx.x] = rr; a.partC[blockIdx.x] = rzn; }
      grid.sync();
      err = sum_partials(a.partB, nb);
      double rzNew = sum_partials(a.partC, nb);
      double beta = rzNew / rz;
      rz = rzNew;
      for (int i = tid; i < a.nNo; i += nth) {
        a.P[i] = a.dinv[i] * a.R[i] + beta * a.P[i];
      }
      grid.sync();
      continue;
    }

    // alpha, X += alpha*P, R -= alpha*SP, partial sums of R.R
    double alpha = errO / sum_partials(a.partA, nb);
    double rr = 0.0;
    for (int i = tid; i < a.nNo; i += nth) {
      a.X[i] = a.X[i] + alpha * a.P[i];
      double r = a.R[i] + (-alpha) * a.SP[i];
      a.R[i] = r;
      rr += r * r;
    }
    rr = block_sum(rr);
    if (threadIdx.x == 0) a.partB[blockIdx.x] = rr;
    grid.sync();
    PHASE(3)

    err = sum_partials(a.partB, nb);
    double c1 = errO / err, c2 = err / errO;
    for (int i = tid; i < a.nNo; i += nth) {
      double t = a.P[i] + c1 * a.R[i];
      a.P[i] = c2 * t;
    }
    grid.sync();
    PHASE(4)
  }
  if (tid == 0) {
    a.st->err = err;
    a.st->errO = errO;
    a.st->last_i = last_i;
    a.st->success = success;
  }
}

}  // namespace

//---------------------------------------------------------------------------

Gpu::~Gpu()
{
  // The process exits after the simulation, so device memory is not freed
}

double* Gpu::alloc(size_t n)
{
  double* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(double)));
  CUDA_CHECK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(double)));
  return p;
}

void Gpu::zero(double* x, size_t n) { CUDA_CHECK(cudaMemsetAsync(x, 0, n * sizeof(double))); }
void Gpu::copy(double* dst, const double* src, size_t n)
{
  CUDA_CHECK(cudaMemcpyAsync(dst, src, n * sizeof(double), cudaMemcpyDeviceToDevice));
}
void Gpu::upload(double* dst, const double* src, size_t n)
{
  CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(double), cudaMemcpyHostToDevice));
}
void Gpu::download(double* dst, const double* src, size_t n)
{
  CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(double), cudaMemcpyDeviceToHost));
}
void Gpu::sync() { CUDA_CHECK(cudaDeviceSynchronize()); }

void Gpu::setup(int nNo_, int nnz_, const int* rowPtr, const int* colPtr)
{
  nNo = nNo_;
  nnz = nnz_;
  std::vector<int> rs(nNo), re(nNo), tp(nnz, -1);
  for (int i = 0; i < nNo; i++) {
    rs[i] = rowPtr[2 * i];
    re[i] = rowPtr[2 * i + 1];
  }
  // Position of the transposed entry of each entry (for Gt)
  for (int i = 0; i < nNo; i++) {
    for (int j = rs[i]; j <= re[i]; j++) {
      int k = colPtr[j];
      for (int l = rs[k]; l <= re[k]; l++) {
        if (colPtr[l] == i) {
          tp[j] = l;
          break;
        }
      }
    }
  }
  auto upload_int = [](int*& d, const int* h, size_t n) {
    CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(n, 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(int), cudaMemcpyHostToDevice));
  };
  upload_int(rowS, rs.data(), nNo);
  upload_int(rowE, re.data(), nNo);
  upload_int(col, colPtr, nnz);
  upload_int(tpos, tp.data(), nnz);

  Val = alloc(16 * (size_t)nnz);
  mK = alloc(9 * (size_t)nnz);
  mG = alloc(3 * (size_t)nnz);
  mD = alloc(3 * (size_t)nnz);
  mL = alloc(nnz);
  Gt = alloc(3 * (size_t)nnz);
  partial = alloc(NPART);
  nScal = 1024;
  scal = alloc(nScal);

  cgX = alloc(nNo);
  cgP = alloc(nNo);
  cgSP = alloc(nNo);
  cgGP = alloc(3 * (size_t)nNo);
  cgR = alloc(nNo);
  CUDA_CHECK(cudaMalloc(&cgState, sizeof(CgState)));

  hRowS = rs;
  hRowE = re;
  hCol.assign(colPtr, colPtr + nnz);
  schur_setup();
}

void Gpu::set_faces(const std::vector<int>& glob, const std::vector<int>& off, const std::vector<double>& valM,
    const std::vector<double>& coefAdd, const std::vector<double>& coefPre)
{
  nCoupled = (int)off.size() - 1;
  if ((int)glob.size() > fCapacity || !fOff) {
    fCapacity = std::max<int>((int)glob.size(), 1);
    CUDA_CHECK(cudaMalloc(&fGlob, fCapacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&fValM, 3 * (size_t)fCapacity * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&fOff, 65 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&fCoefAdd, 64 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&fCoefPre, 64 * sizeof(double)));
  }
  if (nCoupled > 64) throw std::runtime_error("svgpu: too many coupled faces");
  if (!glob.empty()) {
    CUDA_CHECK(cudaMemcpy(fGlob, glob.data(), glob.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(fValM, valM.data(), valM.size() * sizeof(double), cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemcpy(fOff, off.data(), off.size() * sizeof(int), cudaMemcpyHostToDevice));
  if (nCoupled > 0) {
    CUDA_CHECK(cudaMemcpy(fCoefAdd, coefAdd.data(), nCoupled * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(fCoefPre, coefPre.data(), nCoupled * sizeof(double), cudaMemcpyHostToDevice));
  }
}

void Gpu::depart()
{
  k_depart<<<blocks(nnz), TPB>>>(nnz, Val, mK, mG, mD, mL);
  KERNEL_CHECK();
  k_transpose_g<<<blocks(nnz), TPB>>>(nnz, tpos, mG, Gt);
  KERNEL_CHECK();
  static const bool explicitS = !getenv("SVGPU_CG");
  if (explicitS) schur_numeric();
}

void Gpu::spmv_vv(const double* K, const double* x, double* y)
{
  k_spmv_vv<<<blocks(nNo), TPB>>>(nNo, rowS, rowE, col, K, x, y);
  KERNEL_CHECK();
}
void Gpu::spmv_sv(const double* K, const double* x, double* y)
{
  k_spmv_sv<<<blocks(nNo), TPB>>>(nNo, rowS, rowE, col, K, x, y, nullptr);
  KERNEL_CHECK();
}
void Gpu::spmv_vs(const double* K, const double* x, double* y)
{
  k_spmv_vs<<<blocks(nNo), TPB>>>(nNo, rowS, rowE, col, K, x, y);
  KERNEL_CHECK();
}
void Gpu::spmv_ss(const double* K, const double* x, double* y)
{
  k_spmv_ss<<<blocks(nNo), TPB>>>(nNo, rowS, rowE, col, K, x, y);
  KERNEL_CHECK();
}

void Gpu::bc_add(const double* x, double* y)
{
  if (nCoupled == 0) return;
  k_bc<<<nCoupled, TPB>>>(fOff, fGlob, fValM, fCoefAdd, x, y, nullptr);
  KERNEL_CHECK();
}

void Gpu::bc_pre(double* y)
{
  if (nCoupled == 0) return;
  k_bc<<<nCoupled, TPB>>>(fOff, fGlob, fValM, fCoefPre, y, y, nullptr);
  KERNEL_CHECK();
}

void Gpu::dot_dev(const double* a, const double* b, int n, double* dst)
{
  k_dot_partial<<<NRED, TPB>>>(n, a, b, partial);
  KERNEL_CHECK();
  k_finish<<<1, TPB>>>(NRED, partial, dst, 0);
  KERNEL_CHECK();
}

double Gpu::dot(const double* a, const double* b, int n)
{
  dot_dev(a, b, n, scal);
  double r;
  download(&r, scal, 1);
  return r;
}

void Gpu::axpy(double alpha, const double* x, double* y, int n)
{
  k_axpy<<<blocks(n), TPB>>>(n, alpha, x, y);
  KERNEL_CHECK();
}
void Gpu::axpy_dev(const double* alpha, double sign, const double* x, double* y, int n)
{
  k_axpy_dev<<<blocks(n), TPB>>>(n, alpha, sign, x, y);
  KERNEL_CHECK();
}
void Gpu::scale_inv_dev(const double* s, double* x, int n)
{
  k_scale_inv_dev<<<blocks(n), TPB>>>(n, s, x);
  KERNEL_CHECK();
}
void Gpu::scale(double a, double* x, int n)
{
  k_scale<<<blocks(n), TPB>>>(n, a, x);
  KERNEL_CHECK();
}
void Gpu::sub(const double* x, const double* y, double* z, int n)
{
  k_sub<<<blocks(n), TPB>>>(n, x, y, z);
  KERNEL_CHECK();
}

void Gpu::lin_comb(const double* x0, double sign, const double* c, const std::vector<const double*>& xs,
    double* y, int n)
{
  if (xs.size() > 64) throw std::runtime_error("svgpu: too many vectors in lin_comb");
  PtrList pl;
  for (size_t j = 0; j < xs.size(); j++) pl.p[j] = xs[j];
  double* cd = scal + 512;
  CUDA_CHECK(cudaMemcpy(cd, c, xs.size() * sizeof(double), cudaMemcpyHostToDevice));
  k_lin_comb<<<blocks(n), TPB>>>(n, x0, sign, cd, (int)xs.size(), pl, y);
  KERNEL_CHECK();
}

void Gpu::split4(const double* R, double* Rm, double* Rc)
{
  k_split4<<<blocks(nNo), TPB>>>(nNo, R, Rm, Rc);
  KERNEL_CHECK();
}
void Gpu::merge4(const double* Rm, const double* Rc, double* R)
{
  k_merge4<<<blocks(nNo), TPB>>>(nNo, Rm, Rc, R);
  KERNEL_CHECK();
}

Gpu::CgResult Gpu::cg_schur(double* R, int mItr, double absTol, double relTol)
{
  // Default: CG with the explicit Schur complement (svgpu_schur.cu)
  if (!getenv("SVGPU_CG")) return cg_schur_s(R, mItr, absTol, relTol);

  CgResult res{};
  double iNorm = std::sqrt(dot(R, R, nNo));
  res.iNorm = iNorm;
  double eps = std::pow(std::max(absTol, relTol * iNorm), 2.0);
  CgState st{};
  st.err = st.errO = iNorm * iNorm;
  st.eps = eps;
  CUDA_CHECK(cudaMemcpy(cgState, &st, sizeof(CgState), cudaMemcpyHostToDevice));
  copy(cgR, R, nNo);
  copy(cgP, R, nNo);
  zero(cgX, nNo);

  static const bool use_graph = getenv("SVGPU_CG") && std::string(getenv("SVGPU_CG")) == "graph";
  if (!use_graph) {
    static int nbCoop = 0;
    static double *partB = nullptr, *partC = nullptr, *faceS = nullptr, *dinv = nullptr;
    static const bool prec = getenv("SVGPU_CG_PREC") && std::string(getenv("SVGPU_CG_PREC")) == "jacobi";
    if (!nbCoop) {
      int dev, nSM, perSM;
      CUDA_CHECK(cudaGetDevice(&dev));
      CUDA_CHECK(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, dev));
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, k_cg_coop, TPB, 0));
      nbCoop = std::min(nSM * perSM, NPART);
      partB = alloc(NPART);
      partC = alloc(NPART);
      faceS = alloc(64);
      dinv = alloc(nNo);
    }
    double rz0 = 0.0;
    if (prec) {
      // The matrices change in every solve, the preconditioner is computed in each CG call
      k_schur_dinv<<<blocks(nNo), TPB>>>(nNo, rowS, rowE, col, tpos, mL, Gt, mG, dinv);
      KERNEL_CHECK();
      // p = z = M^-1 r
      k_lin_dinv<<<blocks(nNo), TPB>>>(nNo, dinv, cgR, cgP);
      KERNEL_CHECK();
      rz0 = dot(cgR, cgP, nNo);
    }
    CoopArgs a{nNo, mItr, nCoupled, rowS, rowE, col, mG, Gt, mL, cgX, cgP, cgSP, cgGP, cgR,
        fOff, fGlob, fValM, fCoefPre, partial, partB, faceS, cgState, prec ? dinv : nullptr, partC, rz0, nullptr};
    static unsigned long long* tph = nullptr;
    static int nTph = 0;
    if (getenv("SVGPU_PHASES")) {
      if (!tph) { CUDA_CHECK(cudaMalloc(&tph, 8 * sizeof(unsigned long long))); CUDA_CHECK(cudaMemset(tph, 0, 64)); }
      a.tph = tph;
    }
    void* args[] = {&a};
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_cg_coop, nbCoop, TPB, args, 0, 0));
    CUDA_CHECK(cudaMemcpy(&st, cgState, sizeof(CgState), cudaMemcpyDeviceToHost));
    if (a.tph) {
      unsigned long long h[8];
      CUDA_CHECK(cudaMemcpy(h, tph, sizeof(h), cudaMemcpyDeviceToHost));
      static long long nit = 0;
      nit += st.last_i;
      if (++nTph % 8 == 0) {
        printf("CG phases (us/it, %d blocks): GP %.1f bc %.1f SP %.1f upd %.1f P %.1f\n", nbCoop,
            1e-3 * h[0] / nit, 1e-3 * h[1] / nit, 1e-3 * h[2] / nit, 1e-3 * h[3] / nit, 1e-3 * h[4] / nit);
      }
    }
    copy(R, cgX, nNo);
    res.success = st.success != 0;
    res.itr = st.last_i;
    res.fNorm = std::sqrt(st.err);
    res.dB = (st.errO < std::numeric_limits<double>::epsilon()) ? 0.0 : 5.0 * std::log(st.err / st.errO);
    return res;
  }

  // Record all iterations once into a graph; the kernels of the iterations
  // after convergence return immediately
  bool coupled = nCoupled > 0;
  if (!cgGraph || cgGraphItr != mItr || cgGraphCoupled != coupled) {
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    cudaGraph_t g;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    int nb = std::min(NRED, blocks(nNo));
    for (int it = 0; it < mItr; it++) {
      k_cg_top<<<1, 1, 0, s>>>(cgState, it);
      k_spmv_sv<<<blocks(nNo), TPB, 0, s>>>(nNo, rowS, rowE, col, mG, cgP, cgGP, &cgState->done);
      if (coupled) {
        k_bc<<<nCoupled, TPB, 0, s>>>(fOff, fGlob, fValM, fCoefPre, cgGP, cgGP, &cgState->done);
      }
      k_cg_sp<<<nb, TPB, 0, s>>>(nNo, rowS, rowE, col, mL, Gt, cgP, cgGP, cgSP, partial, cgState);
      k_cg_alpha<<<1, TPB, 0, s>>>(nb, partial, cgState);
      k_cg_update<<<nb, TPB, 0, s>>>(nNo, cgX, cgP, cgR, cgSP, partial, cgState);
      k_cg_err<<<1, TPB, 0, s>>>(nb, partial, cgState);
      k_cg_p<<<blocks(nNo), TPB, 0, s>>>(nNo, cgP, cgR, cgState);
    }
    CUDA_CHECK(cudaStreamEndCapture(s, &g));
    cudaGraphExec_t ge;
    CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
    CUDA_CHECK(cudaGraphDestroy(g));
    if (cgGraph) CUDA_CHECK(cudaGraphExecDestroy((cudaGraphExec_t)cgGraph));
    cgGraph = ge;
    cgGraphItr = mItr;
    cgGraphCoupled = coupled;
    CUDA_CHECK(cudaStreamDestroy(s));
  }
  CUDA_CHECK(cudaGraphLaunch((cudaGraphExec_t)cgGraph, 0));
  CUDA_CHECK(cudaMemcpy(&st, cgState, sizeof(CgState), cudaMemcpyDeviceToHost));
  copy(R, cgX, nNo);

  res.success = st.success != 0;
  res.itr = st.last_i;
  res.fNorm = std::sqrt(st.err);
  res.dB = (st.errO < std::numeric_limits<double>::epsilon()) ? 0.0 : 5.0 * std::log(st.err / st.errO);
  return res;
}

}  // namespace svgpu
