// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// Schur complement CG preconditioned with a two-level aggregation AMG
// V-cycle (Jacobi pre- and post-smoothing, exact coarse solve with the
// inverse of the Galerkin coarse matrix). Everything runs on the GPU: the
// coarse matrix is assembled with atomics and inverted with a cooperative
// Gauss-Jordan kernel in every solve, and all CG iterations are executed in
// one cooperative kernel.

#include "svgpu_internal.h"

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } } while (0)

namespace svgpu {

namespace {

namespace cg = cooperative_groups;

constexpr int TPB = 256;
constexpr int LANES = 16;
constexpr int MAXF = 4;

inline int blocks(int n) { return (n + TPB - 1) / TPB; }

struct Amg {
  int nc = 0;
  int *agg = nullptr, *cIdx = nullptr;
  // S*P (nNo x nc) in CSR, and the position in it of each entry of S
  int *spRow = nullptr, *spCol = nullptr, *spMap = nullptr;
  double* spVal = nullptr;
  int nnzSP = 0;
  double *T = nullptr, *PB = nullptr;
  double *Ac = nullptr, *dinv = nullptr;
  double *Z = nullptr, *Z2 = nullptr, *Q = nullptr, *rc = nullptr, *acc = nullptr;
  int nb = 0, nbGJ = 0;
};
Amg amg;

__device__ inline double row_sum(double v)
{
  for (int o = LANES / 2; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o, LANES);
  return v;
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

__global__ void k_coarse_matrix(int nnzS, const int* cIdx, const float* S, double* Ac)
{
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e < nnzS) atomicAdd(&Ac[cIdx[e]], (double)S[e]);
}

__global__ void k_sp_values(int nnzS, const int* spMap, const float* S, double* SP)
{
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e < nnzS) atomicAdd(&SP[spMap[e]], (double)S[e]);
}

// y = dinv1 .* (S*x) (dinv1 = 1/diag(S)), for the power iteration
__global__ void k_dinv_s(int nNo, const int* sRow, const int* sCol, const float* S, const double* dinv1,
    const double* x, double* y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nNo) return;
  double s = 0.0;
  for (int j = sRow[i]; j < sRow[i + 1]; j++) s += (double)S[j] * x[sCol[j]];
  y[i] = dinv1[i] * s;
}

__global__ void k_jacobi_diag(int nNo, const int* sRow, const int* sCol, const float* S, double omega, double* dinv)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nNo) return;
  double d = 0.0;
  for (int j = sRow[i]; j < sRow[i + 1]; j++) {
    if (sCol[j] == i) d = S[j];
  }
  dinv[i] = (d != 0.0) ? omega / d : 0.0;
}

// In-place Gauss-Jordan inversion of the n x n (row-major) matrix A
__global__ void k_gauss_jordan(int n, double* A)
{
  cg::grid_group grid = cg::this_grid();
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nth = gridDim.x * blockDim.x;
  for (int k = 0; k < n; k++) {
    double p = A[(size_t)k * n + k];
    for (int q = tid; q < n * n; q += nth) {
      int i = q / n, j = q % n;
      if (i != k && j != k) A[q] -= A[(size_t)i * n + k] * (A[(size_t)k * n + j] / p);
    }
    grid.sync();
    for (int q = tid; q < n; q += nth) {
      if (q != k) {
        A[(size_t)k * n + q] /= p;
        A[(size_t)q * n + k] = -A[(size_t)q * n + k] / p;
      }
    }
    if (tid == 0) A[(size_t)k * n + k] = 1.0 / p;
    grid.sync();
  }
}

struct AArgs {
  int nNo, mItr, nF, nc;
  const int *sRow, *sCol, *agg;
  const float* sVal;
  const double *U, *W, *coef, *Ainv, *dinv;
  double *X, *P, *R, *Z, *Z2, *Q, *rc, *acc;
  double err, eps;
  double* out;
};

// z = M^-1 r: two-level V-cycle with Jacobi smoothing. Uses Z (result), Z2
// and rc. All threads of the grid must call it.
__device__ void vcycle(const AArgs& a, cg::grid_group& grid, int tid, int nth, int lane, int gid0, int ngroups,
    int nRowIt)
{
  // Pre-smoothing z1 = dinv*r
  for (int i = tid; i < a.nNo; i += nth) a.Z2[i] = a.dinv[i] * a.R[i];
  for (int c = tid; c < a.nc; c += nth) a.rc[c] = 0.0;
  grid.sync();
  // Restriction of r - S*z1
  for (int i0 = gid0; i0 < nRowIt; i0 += ngroups) {
    int i = i0;
    double s = 0.0;
    if (i < a.nNo) {
      for (int j = a.sRow[i] + lane; j < a.sRow[i + 1]; j += LANES) s += (double)a.sVal[j] * a.Z2[a.sCol[j]];
    }
    s = row_sum(s);
    if (lane == 0 && i < a.nNo) atomicAdd(&a.rc[a.agg[i]], a.R[i] - s);
  }
  grid.sync();
  // Coarse correction z2 = z1 + P*Ainv*rc (one warp per coarse row; the
  // result is first stored in the coarse vector part of Z)
  {
    const int warp = tid / 32, wl = tid % 32, nw = nth / 32;
    for (int c = warp; c < a.nc; c += nw) {
      double s = 0.0;
      for (int d = wl; d < a.nc; d += 32) s += a.Ainv[(size_t)c * a.nc + d] * a.rc[d];
      for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
      if (wl == 0) a.Q[c] = s;   // Q is free here; it is recomputed in the next iteration
    }
  }
  grid.sync();
  for (int i = tid; i < a.nNo; i += nth) a.Z2[i] = a.Z2[i] + a.Q[a.agg[i]];
  grid.sync();
  // Post-smoothing z = z2 + dinv*(r - S*z2)
  for (int i0 = gid0; i0 < nRowIt; i0 += ngroups) {
    int i = i0;
    double s = 0.0;
    if (i < a.nNo) {
      for (int j = a.sRow[i] + lane; j < a.sRow[i + 1]; j += LANES) s += (double)a.sVal[j] * a.Z2[a.sCol[j]];
    }
    s = row_sum(s);
    if (lane == 0 && i < a.nNo) a.Z[i] = a.Z2[i] + a.dinv[i] * (a.R[i] - s);
  }
  grid.sync();
}

__global__ void k_cg_amg(AArgs a)
{
  cg::grid_group grid = cg::this_grid();
  const int nb = gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nth = nb * blockDim.x;
  const int lane = threadIdx.x % LANES;
  const int gid0 = tid / LANES;
  const int ngroups = nth / LANES;
  const int nRowIt = ((a.nNo + ngroups - 1) / ngroups) * ngroups;
  double err = a.err;
  int last_i = 0, success = 0;
  // Accumulators: acc[slot*16 + k], slot alternates every phase
  int slot = 0;
  auto next_acc = [&]() {
    // The slot of the next phase was last read two phases ago, so it can be
    // cleared now (before the synchronization of this phase)
    double* cur = a.acc + slot * 16;
    slot = (slot + 1) % 3;
    if (blockIdx.x == 0 && threadIdx.x < 16) a.acc[slot * 16 + threadIdx.x] = 0.0;
    return cur;
  };

  // z = M^-1 r, p = z, rz = r.z
  vcycle(a, grid, tid, nth, lane, gid0, ngroups, nRowIt);
  double rz;
  {
    double* acc = next_acc();
    double v = 0.0;
    for (int i = tid; i < a.nNo; i += nth) {
      a.P[i] = a.Z[i];
      v += a.R[i] * a.Z[i];
    }
    v = block_sum(v);
    if (threadIdx.x == 0) atomicAdd(&acc[0], v);
    grid.sync();
    rz = __ldcg(&acc[0]);
  }

  for (int it = 0; it < a.mItr; it++) {
    last_i = it;
    if (err < a.eps) { success = 1; break; }

    // Q = S_eff*P, p.Q, u_f.P, w_f.P
    double* acc = next_acc();
    double v0 = 0.0;
    for (int i0 = gid0; i0 < nRowIt; i0 += ngroups) {
      int i = i0;
      double s = 0.0;
      if (i < a.nNo) {
        for (int j = a.sRow[i] + lane; j < a.sRow[i + 1]; j += LANES) s += (double)a.sVal[j] * a.P[a.sCol[j]];
      }
      s = row_sum(s);
      if (lane == 0 && i < a.nNo) {
        a.Q[i] = s;
        v0 += a.P[i] * s;
      }
    }
    v0 = block_sum(v0);
    if (threadIdx.x == 0) atomicAdd(&acc[0], v0);
    for (int f = 0; f < a.nF; f++) {
      double vu = 0.0, vw = 0.0;
      for (int i = tid; i < a.nNo; i += nth) {
        vu += a.U[(size_t)f * a.nNo + i] * a.P[i];
        vw += a.W[(size_t)f * a.nNo + i] * a.P[i];
      }
      vu = block_sum(vu);
      vw = block_sum(vw);
      if (threadIdx.x == 0) { atomicAdd(&acc[1 + 2 * f], vu); atomicAdd(&acc[2 + 2 * f], vw); }
    }
    grid.sync();
    double pQ = __ldcg(&acc[0]);
    double Uf[MAXF];
    for (int f = 0; f < a.nF; f++) {
      Uf[f] = __ldcg(&acc[1 + 2 * f]);
      pQ -= a.coef[f] * __ldcg(&acc[2 + 2 * f]) * Uf[f];
    }
    double alpha = rz / pQ;

    // X += alpha*P, R -= alpha*Q_eff, R.R
    acc = next_acc();
    double rr = 0.0;
    for (int i = tid; i < a.nNo; i += nth) {
      double q = a.Q[i];
      for (int f = 0; f < a.nF; f++) q -= a.coef[f] * a.W[(size_t)f * a.nNo + i] * Uf[f];
      a.X[i] = a.X[i] + alpha * a.P[i];
      double r = a.R[i] - alpha * q;
      a.R[i] = r;
      rr += r * r;
    }
    rr = block_sum(rr);
    if (threadIdx.x == 0) atomicAdd(&acc[0], rr);
    grid.sync();
    err = __ldcg(&acc[0]);
    if (err < a.eps) { last_i = it + 1; success = 1; break; }

    // z = M^-1 r, beta, P = z + beta*P
    vcycle(a, grid, tid, nth, lane, gid0, ngroups, nRowIt);
    acc = next_acc();
    double v = 0.0;
    for (int i = tid; i < a.nNo; i += nth) v += a.R[i] * a.Z[i];
    v = block_sum(v);
    if (threadIdx.x == 0) atomicAdd(&acc[0], v);
    grid.sync();
    double rzNew = __ldcg(&acc[0]);
    double beta = rzNew / rz;
    rz = rzNew;
    for (int i = tid; i < a.nNo; i += nth) a.P[i] = a.Z[i] + beta * a.P[i];
    grid.sync();
  }
  if (tid == 0) {
    a.out[0] = err;
    a.out[1] = a.err;
    a.out[2] = last_i;
    a.out[3] = success;
  }
}



constexpr int NR = 4;   // rows processed at the same time by a group of lanes

// For all rows i: s_i = sum_j val(j) * x(col(j)) over the entries of row i,
// then fin(i, s_i) by lane 0. Rows are distributed over the groups of LANES
// threads, NR rows at a time; all lanes of a warp do the same number of
// iterations of the outer loop (shuffles in row_sum).
template <typename V, typename X, typename F>
__device__ inline void rows_nr(int nNo, const int* rowPtr, const int* col, V val, X x, F fin, int lane, int gid0,
    int ngroups)
{
  const int step = ngroups * NR;
  const int nIt = ((nNo + step - 1) / step) * step;
  for (int base = gid0 * NR; base < nIt; base += step) {
    double s[NR];
    int b[NR], e[NR];
    int maxLen = 0;
    for (int r = 0; r < NR; r++) {
      int i = base + r;
      s[r] = 0.0;
      if (i < nNo) { b[r] = rowPtr[i]; e[r] = rowPtr[i + 1]; } else { b[r] = 0; e[r] = 0; }
      maxLen = max(maxLen, e[r] - b[r]);
    }
    for (int k = lane; k < maxLen; k += LANES) {
      for (int r = 0; r < NR; r++) {
        int j = b[r] + k;
        if (j < e[r]) s[r] += val(j) * x(col[j]);
      }
    }
    for (int r = 0; r < NR; r++) {
      double t = row_sum(s[r]);
      int i = base + r;
      if (lane == 0 && i < nNo) fin(i, t);
    }
  }
}

struct BArgs {
  int nNo, mItr, nF, nc;
  const int *sRow, *sCol, *agg, *spRow, *spCol;
  const float* sVal;
  const double *spVal, *U, *W, *coef, *Ainv, *dinv;
  double *X, *PA, *PB, *R, *Z, *T, *Q, *rc, *acc;
  double err, eps;
  double* out;
  unsigned long long* tph;
};

__device__ inline unsigned long long gtimer()
{
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ unsigned long long g_tlast;
#define PHASE(k) if (a.tph && blockIdx.x == 0 && threadIdx.x == 0) { unsigned long long t_ = gtimer(); a.tph[k] += t_ - g_tlast; g_tlast = t_; }

// Preconditioned CG with the two-level V-cycle, 5 grid synchronizations
// per iteration:
//   1. p = z + beta*p (on the fly, double buffered), Q = S_eff*p, dots
//   2. x, r update, r.r
//   3. t = S*z1 with z1 = dinv*r (on the fly), restriction of r - t
//   4. coarse correction ec = Ainv*rc
//   5. z = z1 + P*ec + dinv*(r - t - (S*P)*ec), r.z
__device__ void precondition2(const BArgs& a, cg::grid_group& grid, double* acc, int tid, int nth, int lane,
    int gid0, int ngroups, int nRowIt)
{
  // 3. restriction
  rows_nr(a.nNo, a.sRow, a.sCol, [&](int j) { return (double)a.sVal[j]; },
      [&](int c) { return a.dinv[c] * a.R[c]; },
      [&](int i, double s) {
        a.T[i] = s;
        atomicAdd(&a.rc[a.agg[i]], a.R[i] - s);
      }, lane, gid0, ngroups);
  grid.sync();
  PHASE(2)
  // 4. coarse correction (stored in Q[0..nc-1])
  {
    const int warp = tid / 32, wl = tid % 32, nw = nth / 32;
    for (int c = warp; c < a.nc; c += nw) {
      double s = 0.0;
      for (int d = wl; d < a.nc; d += 32) s += a.Ainv[(size_t)c * a.nc + d] * a.rc[d];
      for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
      if (wl == 0) a.Q[c] = s;
    }
  }
  grid.sync();
  PHASE(3)
  // 5. post-smoothing, r.z
  double rz = 0.0;
  rows_nr(a.nNo, a.spRow, a.spCol, [&](int j) { return a.spVal[j]; }, [&](int c) { return a.Q[c]; },
      [&](int i, double s) {
        double z2 = a.dinv[i] * a.R[i] + a.Q[a.agg[i]];
        double z = z2 + a.dinv[i] * (a.R[i] - a.T[i] - s);
        a.Z[i] = z;
        rz += a.R[i] * z;
      }, lane, gid0, ngroups);
  rz = block_sum(rz);
  if (threadIdx.x == 0) atomicAdd(&acc[0], rz);
  grid.sync();
  PHASE(4)
}

__global__ void k_cg_amg2(BArgs a)
{
  cg::grid_group grid = cg::this_grid();
  const int nb = gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nth = nb * blockDim.x;
  const int lane = threadIdx.x % LANES;
  const int gid0 = tid / LANES;
  const int ngroups = nth / LANES;
  const int nRowIt = ((a.nNo + ngroups - 1) / ngroups) * ngroups;
  double err = a.err;
  int last_i = 0, success = 0;
  if (a.tph && blockIdx.x == 0 && threadIdx.x == 0) g_tlast = gtimer();
  // Accumulator slots: the slot of the next phase is cleared by block 0
  // (it was last read three phases ago)
  int slot = 0;
  double* acc;
#define NEXT_ACC() (acc = a.acc + (slot % 4) * 16, slot++, \
    (blockIdx.x == 0 && threadIdx.x < 16) ? (a.acc[(slot % 4) * 16 + threadIdx.x] = 0.0) : 0.0, acc)

  for (int c = tid; c < a.nc; c += nth) a.rc[c] = 0.0;
  NEXT_ACC();
  grid.sync();
  NEXT_ACC();
  precondition2(a, grid, acc, tid, nth, lane, gid0, ngroups, nRowIt);
  double rz = __ldcg(&acc[0]);
  if (a.tph && blockIdx.x == 0 && threadIdx.x == 0) { a.tph[5] += 1; g_tlast = gtimer(); }
  double beta = 0.0;
  double* Pold = a.PA;
  double* Pnew = a.PB;

  for (int it = 0; it < a.mItr; it++) {
    last_i = it;
    if (err < a.eps) { success = 1; break; }

    // 1. p = z + beta*p, Q = S_eff*p and the dots
    NEXT_ACC();
    double v0 = 0.0;
    rows_nr(a.nNo, a.sRow, a.sCol, [&](int j) { return (double)a.sVal[j]; },
        [&](int c) { return a.Z[c] + beta * Pold[c]; },
        [&](int i, double s) {
          double p = a.Z[i] + beta * Pold[i];
          Pnew[i] = p;
          a.Q[i] = s;
          v0 += p * s;
        }, lane, gid0, ngroups);
    v0 = block_sum(v0);
    if (threadIdx.x == 0) atomicAdd(&acc[0], v0);
    for (int f = 0; f < a.nF; f++) {
      double vu = 0.0, vw = 0.0;
      for (int i = tid; i < a.nNo; i += nth) {
        double p = a.Z[i] + beta * Pold[i];
        vu += a.U[(size_t)f * a.nNo + i] * p;
        vw += a.W[(size_t)f * a.nNo + i] * p;
      }
      vu = block_sum(vu);
      vw = block_sum(vw);
      if (threadIdx.x == 0) { atomicAdd(&acc[1 + 2 * f], vu); atomicAdd(&acc[2 + 2 * f], vw); }
    }
    grid.sync();
    PHASE(0)
    double pQ = __ldcg(&acc[0]);
    double Uf[MAXF];
    for (int f = 0; f < a.nF; f++) {
      Uf[f] = __ldcg(&acc[1 + 2 * f]);
      pQ -= a.coef[f] * __ldcg(&acc[2 + 2 * f]) * Uf[f];
    }
    double alpha = rz / pQ;

    // 2. x += alpha*p, r -= alpha*Q_eff, r.r (and rc = 0 for the restriction)
    NEXT_ACC();
    double rr = 0.0;
    for (int i = tid; i < a.nNo; i += nth) {
      double q = a.Q[i];
      for (int f = 0; f < a.nF; f++) q -= a.coef[f] * a.W[(size_t)f * a.nNo + i] * Uf[f];
      a.X[i] = a.X[i] + alpha * Pnew[i];
      double r = a.R[i] - alpha * q;
      a.R[i] = r;
      rr += r * r;
    }
    for (int c = tid; c < a.nc; c += nth) a.rc[c] = 0.0;
    rr = block_sum(rr);
    if (threadIdx.x == 0) atomicAdd(&acc[0], rr);
    grid.sync();
    PHASE(1)
    err = __ldcg(&acc[0]);
    if (err < a.eps) { last_i = it + 1; success = 1; break; }

    // 3-5. z = M^-1 r
    NEXT_ACC();
    precondition2(a, grid, acc, tid, nth, lane, gid0, ngroups, nRowIt);
    double rzNew = __ldcg(&acc[0]);
    beta = rzNew / rz;
    rz = rzNew;
    double* t = Pold; Pold = Pnew; Pnew = t;
  }
#undef NEXT_ACC
  if (tid == 0) {
    a.out[0] = err;
    a.out[1] = a.err;
    a.out[2] = last_i;
    a.out[3] = success;
  }
}
}  // namespace

// Aggregates of the graph of S: a root node with its unaggregated neighbors,
// remaining nodes join a neighboring aggregate
void amg_setup_impl(Gpu& g, const std::vector<int>& sRowH, const std::vector<int>& sColH)
{
  const int n = g.nNo;
  // Aggregate radius in the graph of S: 1 (a node and its neighbors), or 2
  // (also the unaggregated neighbors of the neighbors) for large meshes, so
  // that the dense coarse matrix stays small
  int rad = getenv("SVGPU_AMG_RAD") ? atoi(getenv("SVGPU_AMG_RAD")) : (n > 50000 ? 2 : 1);
  std::vector<int> agg(n, -1);
  int nc = 0;
  for (int i = 0; i < n; i++) {
    if (agg[i] >= 0) continue;
    bool free = true;
    for (int j = sRowH[i]; j < sRowH[i + 1]; j++) {
      if (agg[sColH[j]] >= 0) { free = false; break; }
    }
    if (!free) continue;
    for (int j = sRowH[i]; j < sRowH[i + 1]; j++) agg[sColH[j]] = nc;
    if (rad >= 2) {
      for (int j = sRowH[i]; j < sRowH[i + 1]; j++) {
        int k = sColH[j];
        for (int l = sRowH[k]; l < sRowH[k + 1]; l++) {
          if (agg[sColH[l]] < 0) agg[sColH[l]] = nc;
        }
      }
    }
    agg[i] = nc++;
  }
  for (int i = 0; i < n; i++) {
    if (agg[i] >= 0) continue;
    for (int j = sRowH[i]; j < sRowH[i + 1]; j++) {
      if (agg[sColH[j]] >= 0) { agg[i] = agg[sColH[j]]; break; }
    }
    if (agg[i] < 0) agg[i] = nc++;
  }
  std::vector<int> cIdx(sColH.size());
  for (int i = 0; i < n; i++) {
    for (int j = sRowH[i]; j < sRowH[i + 1]; j++) cIdx[j] = agg[i] * nc + agg[sColH[j]];
  }
  amg.nc = nc;
  CUDA_CHECK(cudaMalloc(&amg.agg, n * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(amg.agg, agg.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&amg.cIdx, cIdx.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(amg.cIdx, cIdx.data(), cIdx.size() * sizeof(int), cudaMemcpyHostToDevice));
  amg.Ac = g.alloc((size_t)nc * nc);
  amg.dinv = g.alloc(n);
  amg.Z = g.alloc(n);
  amg.Z2 = g.alloc(n);
  amg.Q = g.alloc(std::max(n, nc));
  amg.rc = g.alloc(nc);
  amg.acc = g.alloc(64);
  amg.T = g.alloc(n);
  amg.PB = g.alloc(n);
  // Structure of S*P: the aggregates of the neighbors of each node
  std::vector<int> spRow(n + 1, 0), spCol, spMap(sColH.size());
  std::vector<int> pos(nc, -1), cols;
  for (int i = 0; i < n; i++) {
    cols.clear();
    for (int j = sRowH[i]; j < sRowH[i + 1]; j++) {
      int c = agg[sColH[j]];
      if (pos[c] < 0) { pos[c] = 1; cols.push_back(c); }
    }
    std::sort(cols.begin(), cols.end());
    for (size_t q = 0; q < cols.size(); q++) pos[cols[q]] = spRow[i] + (int)q;
    for (int j = sRowH[i]; j < sRowH[i + 1]; j++) spMap[j] = pos[agg[sColH[j]]];
    for (int c : cols) { spCol.push_back(c); pos[c] = -1; }
    spRow[i + 1] = (int)spCol.size();
  }
  amg.nnzSP = (int)spCol.size();
  auto up = [](int*& d, const std::vector<int>& h) {
    CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice));
  };
  up(amg.spRow, spRow);
  up(amg.spCol, spCol);
  up(amg.spMap, spMap);
  amg.spVal = g.alloc(amg.nnzSP);
  printf("svgpu: AMG with %d aggregates for %d unknowns, nnz(S*P) = %d\n", nc, n, amg.nnzSP);
}

void amg_numeric_impl(Gpu& g, const int* sRow, const int* sCol, const float* sVal, int nnzS)
{
  const int nc = amg.nc;
  CUDA_CHECK(cudaMemset(amg.Ac, 0, (size_t)nc * nc * sizeof(double)));
  k_coarse_matrix<<<blocks(nnzS), TPB>>>(nnzS, amg.cIdx, sVal, amg.Ac);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemset(amg.spVal, 0, (size_t)amg.nnzSP * sizeof(double)));
  k_sp_values<<<blocks(nnzS), TPB>>>(nnzS, amg.spMap, sVal, amg.spVal);
  CUDA_CHECK(cudaGetLastError());
  // Jacobi damping omega = 1.8/lambda_max(D^-1 S), lambda_max from a power
  // iteration in the first solve
  static double omega = 0.0;
  if (omega == 0.0) {
    if (getenv("SVGPU_AMG_OMEGA")) {
      omega = atof(getenv("SVGPU_AMG_OMEGA"));
    } else {
      const int n = g.nNo;
      double* x = g.alloc(n);
      double* y = g.alloc(n);
      k_jacobi_diag<<<blocks(n), TPB>>>(n, sRow, sCol, sVal, 1.0, amg.dinv);
      std::vector<double> h(n);
      for (int i = 0; i < n; i++) h[i] = 1.0 + 0.5 * std::sin(0.37 * i);
      g.upload(x, h.data(), n);
      double lambda = 1.0;
      for (int k = 0; k < 30; k++) {
        k_dinv_s<<<blocks(n), TPB>>>(n, sRow, sCol, sVal, amg.dinv, x, y);
        double nx = std::sqrt(g.dot(x, x, n)), ny = std::sqrt(g.dot(y, y, n));
        lambda = ny / nx;
        g.scale(0.0, x, n);
        g.axpy(1.0 / ny, y, x, n);
      }
      omega = 1.8 / (1.1 * lambda);
      printf("svgpu: lambda_max(D^-1 S) ~ %.3f, Jacobi omega = %.3f\n", lambda, omega);
      cudaFree(x);
      cudaFree(y);
    }
  }
  k_jacobi_diag<<<blocks(g.nNo), TPB>>>(g.nNo, sRow, sCol, sVal, omega, amg.dinv);
  CUDA_CHECK(cudaGetLastError());
  if (!amg.nbGJ) {
    int dev, nSM, perSM;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, k_gauss_jordan, TPB, 0));
    amg.nbGJ = nSM * std::min(perSM, 2);
  }
  int n = nc;
  double* A = amg.Ac;
  void* args[] = {&n, &A};
  CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_gauss_jordan, amg.nbGJ, TPB, args, 0, 0));
}

struct AmgView {
  int nc;
  const int *agg, *spRow, *spCol;
  const double *spVal, *Ainv, *dinv;
  double *Z, *T, *Q, *rc;
};
AmgView amg_view()
{
  return AmgView{amg.nc, amg.agg, amg.spRow, amg.spCol, amg.spVal, amg.Ac, amg.dinv, amg.Z, amg.T, amg.Q, amg.rc};
}

Gpu::CgResult cg_schur_amg_graph(Gpu& g, double* R, int mItr, double absTol, double relTol);

Gpu::CgResult cg_schur_amg_impl(Gpu& g, double* R, int mItr, double absTol, double relTol)
{
  static const bool useGraph = getenv("SVGPU_AMG_GRAPH") != nullptr;
  if (useGraph) return cg_schur_amg_graph(g, R, mItr, absTol, relTol);
  Gpu::CgResult res{};
  double iNorm = std::sqrt(g.dot(R, R, g.nNo));
  res.iNorm = iNorm;
  double eps = std::pow(std::max(absTol, relTol * iNorm), 2.0);
  g.copy(g.cgR, R, g.nNo);
  g.zero(g.cgX, g.nNo);
  if (!amg.nb) {
    int dev, nSM, perSM;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, k_cg_amg, TPB, 0));
    int maxPer = getenv("SVGPU_PERSM") ? atoi(getenv("SVGPU_PERSM")) : 1;
    amg.nb = nSM * std::min(perSM, maxPer);
  }
  CUDA_CHECK(cudaMemset(amg.acc, 0, 64 * sizeof(double)));
  double* out = g.scal + 900;
  static const bool fused = !getenv("SVGPU_AMG_V1");
  if (fused) {
    static int nb2 = 0;
    if (!nb2) {
      int dev, nSM, perSM;
      CUDA_CHECK(cudaGetDevice(&dev));
      CUDA_CHECK(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, dev));
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, k_cg_amg2, TPB, 0));
      int maxPer = getenv("SVGPU_PERSM") ? atoi(getenv("SVGPU_PERSM")) : 1;
      nb2 = nSM * std::min(perSM, maxPer);
    }
    BArgs b{g.nNo, mItr, g.nCoupled, amg.nc, g.sRow, g.sCol, amg.agg, amg.spRow, amg.spCol, g.sVal, amg.spVal,
        g.fU, g.fW, g.fCoefPre, amg.Ac, amg.dinv, g.cgX, g.cgP, amg.PB, g.cgR, amg.Z, amg.T, amg.Q, amg.rc, amg.acc,
        iNorm * iNorm, eps, out, nullptr};
    static unsigned long long* tph = nullptr;
    static long long nit = 0;
    static int ncall = 0;
    if (getenv("SVGPU_PHASES")) {
      if (!tph) { CUDA_CHECK(cudaMalloc(&tph, 64)); CUDA_CHECK(cudaMemset(tph, 0, 64)); }
      b.tph = tph;
    }
    void* args2[] = {&b};
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_cg_amg2, nb2, TPB, args2, 0, 0));
    double h[4];
    g.download(h, out, 4);
    if (tph) {
      nit += (long long)h[2];
      if (++ncall % 8 == 0) {
        unsigned long long t[6];
        CUDA_CHECK(cudaMemcpy(t, tph, sizeof(t), cudaMemcpyDeviceToHost));
        printf("AMG CG phases (us/it): Sp %.1f update %.1f restrict %.1f coarse %.1f smooth %.1f\n",
            1e-3 * t[0] / nit, 1e-3 * t[1] / nit, 1e-3 * t[2] / nit, 1e-3 * t[3] / nit, 1e-3 * t[4] / nit);
      }
    }
    g.copy(R, g.cgX, g.nNo);
    res.success = h[3] != 0.0;
    res.itr = (int)h[2];
    res.fNorm = std::sqrt(h[0]);
    res.dB = (h[1] < std::numeric_limits<double>::epsilon()) ? 0.0 : 5.0 * std::log(h[0] / h[1]);
    return res;
  }
  AArgs a{g.nNo, mItr, g.nCoupled, amg.nc, g.sRow, g.sCol, amg.agg, g.sVal, g.fU, g.fW, g.fCoefPre, amg.Ac,
      amg.dinv, g.cgX, g.cgP, g.cgR, amg.Z, amg.Z2, amg.Q, amg.rc, amg.acc, iNorm * iNorm, eps, out};
  void* args[] = {&a};
  CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_cg_amg, amg.nb, TPB, args, 0, 0));
  double h[4];
  g.download(h, out, 4);
  g.copy(R, g.cgX, g.nNo);
  res.success = h[3] != 0.0;
  res.itr = (int)h[2];
  res.fNorm = std::sqrt(h[0]);
  res.dB = (h[1] < std::numeric_limits<double>::epsilon()) ? 0.0 : 5.0 * std::log(h[0] / h[1]);
  return res;
}

}  // namespace svgpu
