// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// AMG preconditioned Schur complement CG (same algorithm as k_cg_amg2 in
// svgpu_amg.cu) with ordinary full-occupancy kernels in CUDA graphs: one graph
// executes 8 iterations, the host checks the device convergence flag after
// each graph. The kernels of the iterations after convergence return
// immediately.

#include "svgpu_internal.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } } while (0)

namespace svgpu {

namespace {

constexpr int TPB = 256;
constexpr int LANES = 16;
constexpr int MAXF = 4;
constexpr int CHUNK = 8;   // iterations per graph (even, P buffers alternate)

// Device state of the CG
struct GState {
  double rz, err, eps, beta, alpha;
  double acc[4][16];   // accumulators of the reductions of the phases
  int done, success, itr, mItr;
};

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

struct GArgs {
  int nNo, nF, nc;
  const int *sRow, *sCol, *agg, *spRow, *spCol;
  const float* sVal;
  const double *spVal, *U, *W, *coef, *Ainv, *dinv;
  double *X, *R, *Z, *T, *Q, *Ec, *rc;
  GState* st;
};

// Row index and lane of the thread (LANES threads per row)
__device__ inline int my_row(int& lane)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  lane = t % LANES;
  return t / LANES;
}

// 1. p = z + beta*p (Pold -> Pnew), Q = S*p, dots p.Q, u_f.p, w_f.p (acc[0])
__global__ void g_sp(GArgs a, const double* Pold, double* Pnew)
{
  GState* st = a.st;
  if (st->done) return;
  int lane;
  int i = my_row(lane);
  const double beta = st->beta;
  double s = 0.0;
  if (i < a.nNo) {
    for (int j = a.sRow[i] + lane; j < a.sRow[i + 1]; j += LANES) {
      int c = a.sCol[j];
      s += (double)a.sVal[j] * (a.Z[c] + beta * Pold[c]);
    }
  }
  s = row_sum(s);
  double v[1 + 2 * MAXF] = {};
  if (lane == 0 && i < a.nNo) {
    double p = a.Z[i] + beta * Pold[i];
    Pnew[i] = p;
    a.Q[i] = s;
    v[0] = p * s;
    for (int f = 0; f < a.nF; f++) {
      v[1 + 2 * f] = a.U[(size_t)f * a.nNo + i] * p;
      v[2 + 2 * f] = a.W[(size_t)f * a.nNo + i] * p;
    }
  }
  for (int k = 0; k < 1 + 2 * a.nF; k++) {
    double x = block_sum(v[k]);
    if (threadIdx.x == 0) atomicAdd(&st->acc[0][k], x);
  }
  if (blockIdx.x == 0 && threadIdx.x < 16) st->acc[1][threadIdx.x] = 0.0;
}

// 2. alpha, x += alpha*p, r -= alpha*Q_eff, r.r (acc[1]); rc = 0
__global__ void g_update(GArgs a, const double* Pnew)
{
  GState* st = a.st;
  if (st->done) return;
  double pQ = st->acc[0][0];
  double Uf[MAXF];
  for (int f = 0; f < a.nF; f++) {
    Uf[f] = st->acc[0][1 + 2 * f];
    pQ -= a.coef[f] * st->acc[0][2 + 2 * f] * Uf[f];
  }
  double alpha = st->rz / pQ;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double rr = 0.0;
  if (i < a.nNo) {
    double q = a.Q[i];
    for (int f = 0; f < a.nF; f++) q -= a.coef[f] * a.W[(size_t)f * a.nNo + i] * Uf[f];
    a.X[i] = a.X[i] + alpha * Pnew[i];
    double r = a.R[i] - alpha * q;
    a.R[i] = r;
    rr = r * r;
  }
  if (i < a.nc) a.rc[i] = 0.0;
  rr = block_sum(rr);
  if (threadIdx.x == 0) atomicAdd(&st->acc[1][0], rr);
  if (blockIdx.x == 0 && threadIdx.x < 16) st->acc[2][threadIdx.x] = 0.0;
}

// Convergence check after the update (one thread)
__global__ void g_check(GState* st)
{
  if (st->done) return;
  st->err = st->acc[1][0];
  st->itr++;
  if (st->err < st->eps) {
    st->done = 1;
    st->success = 1;
  } else if (st->itr >= st->mItr) {
    st->done = 1;
  }
  st->acc[0][0] = 0.0;
  for (int k = 0; k < 16; k++) st->acc[0][k] = 0.0;
}

// 3. t = S*z1 (z1 = dinv*r), rc += r - t
__global__ void g_restrict(GArgs a, int force)
{
  if (!force && a.st->done) return;
  int lane;
  int i = my_row(lane);
  double s = 0.0;
  if (i < a.nNo) {
    for (int j = a.sRow[i] + lane; j < a.sRow[i + 1]; j += LANES) {
      int c = a.sCol[j];
      s += (double)a.sVal[j] * (a.dinv[c] * a.R[c]);
    }
  }
  s = row_sum(s);
  if (lane == 0 && i < a.nNo) {
    a.T[i] = s;
    atomicAdd(&a.rc[a.agg[i]], a.R[i] - s);
  }
}

// 4. ec = Ainv*rc (warp per coarse row)
__global__ void g_coarse(GArgs a, int force)
{
  if (!force && a.st->done) return;
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int c = t / 32, wl = t % 32;
  if (c >= a.nc) return;
  double s = 0.0;
  for (int d = wl; d < a.nc; d += 32) s += a.Ainv[(size_t)c * a.nc + d] * a.rc[d];
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
  if (wl == 0) a.Ec[c] = s;
}

// 5. z = z1 + P*ec + dinv*(r - t - (S*P)*ec), r.z (acc[2])
__global__ void g_smooth(GArgs a, int force)
{
  GState* st = a.st;
  if (!force && st->done) return;
  int lane;
  int i = my_row(lane);
  double s = 0.0;
  if (i < a.nNo) {
    for (int j = a.spRow[i] + lane; j < a.spRow[i + 1]; j += LANES) s += a.spVal[j] * a.Ec[a.spCol[j]];
  }
  s = row_sum(s);
  double rz = 0.0;
  if (lane == 0 && i < a.nNo) {
    double z2 = a.dinv[i] * a.R[i] + a.Ec[a.agg[i]];
    double z = z2 + a.dinv[i] * (a.R[i] - a.T[i] - s);
    a.Z[i] = z;
    rz = a.R[i] * z;
  }
  rz = block_sum(rz);
  if (threadIdx.x == 0) atomicAdd(&st->acc[2][0], rz);
}

// beta = rz_new / rz (one thread)
__global__ void g_beta(GState* st, int first)
{
  if (!first && st->done) return;
  double rzNew = st->acc[2][0];
  st->beta = first ? 0.0 : rzNew / st->rz;
  st->rz = rzNew;
  for (int k = 0; k < 16; k++) st->acc[2][k] = 0.0;
}

struct Graph {
  cudaGraphExec_t exec = nullptr;
  int nNo = 0;
};
Graph graph;
GState* dState = nullptr;
double *dEc = nullptr, *dPB = nullptr;

}  // namespace

// Arrays of the AMG of svgpu_amg.cu
struct AmgView {
  int nc;
  const int *agg, *spRow, *spCol;
  const double *spVal, *Ainv, *dinv;
  double *Z, *T, *Q, *rc;
};
AmgView amg_view();

Gpu::CgResult cg_schur_amg_graph(Gpu& g, double* R, int mItr, double absTol, double relTol)
{
  Gpu::CgResult res{};
  double iNorm = std::sqrt(g.dot(R, R, g.nNo));
  res.iNorm = iNorm;
  double eps = std::pow(std::max(absTol, relTol * iNorm), 2.0);
  g.copy(g.cgR, R, g.nNo);
  g.zero(g.cgX, g.nNo);
  AmgView av = amg_view();
  if (!dState) {
    CUDA_CHECK(cudaMalloc(&dState, sizeof(GState)));
    dEc = g.alloc(av.nc);
    dPB = g.alloc(g.nNo);
  }
  GState st{};
  st.eps = eps;
  st.err = iNorm * iNorm;
  st.mItr = mItr;
  st.done = (st.err < eps) ? 1 : 0;
  st.success = st.done;
  CUDA_CHECK(cudaMemcpy(dState, &st, sizeof(GState), cudaMemcpyHostToDevice));

  GArgs a{g.nNo, g.nCoupled, av.nc, g.sRow, g.sCol, av.agg, av.spRow, av.spCol, g.sVal, av.spVal, g.fU, g.fW,
      g.fCoefPre, av.Ainv, av.dinv, g.cgX, g.cgR, av.Z, av.T, av.Q, dEc, av.rc, dState};
  const int nbRows = (g.nNo * LANES + TPB - 1) / TPB;
  const int nbVec = (std::max(g.nNo, av.nc) + TPB - 1) / TPB;
  const int nbCoarse = (av.nc * 32 + TPB - 1) / TPB;

  // z = M^-1 r, p = z (Pold = P with beta = 0)
  CUDA_CHECK(cudaMemsetAsync(av.rc, 0, av.nc * sizeof(double)));
  g_restrict<<<nbRows, TPB>>>(a, 1);
  g_coarse<<<nbCoarse, TPB>>>(a, 1);
  g_smooth<<<nbRows, TPB>>>(a, 1);
  g_beta<<<1, 1>>>(dState, 1);
  CUDA_CHECK(cudaGetLastError());
  double* PA = g.cgP;
  double* PB = dPB;

  if (!graph.exec || graph.nNo != g.nNo) {
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    cudaGraph_t gr;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int k = 0; k < CHUNK; k++) {
      const double* Pold = (k % 2 == 0) ? PA : PB;
      double* Pnew = (k % 2 == 0) ? PB : PA;
      g_sp<<<nbRows, TPB, 0, s>>>(a, Pold, Pnew);
      g_update<<<nbVec, TPB, 0, s>>>(a, Pnew);
      g_check<<<1, 1, 0, s>>>(dState);
      g_restrict<<<nbRows, TPB, 0, s>>>(a, 0);
      g_coarse<<<nbCoarse, TPB, 0, s>>>(a, 0);
      g_smooth<<<nbRows, TPB, 0, s>>>(a, 0);
      g_beta<<<1, 1, 0, s>>>(dState, 0);
    }
    CUDA_CHECK(cudaStreamEndCapture(s, &gr));
    CUDA_CHECK(cudaGraphInstantiate(&graph.exec, gr, 0));
    CUDA_CHECK(cudaGraphDestroy(gr));
    CUDA_CHECK(cudaStreamDestroy(s));
    graph.nNo = g.nNo;
  }
  // The first Pold must hold z with beta = 0: Pold is not used then (beta*Pold = 0)
  CUDA_CHECK(cudaMemsetAsync(PA, 0, g.nNo * sizeof(double)));
  int done = 0;
  while (!done) {
    CUDA_CHECK(cudaGraphLaunch(graph.exec, 0));
    CUDA_CHECK(cudaMemcpy(&done, &dState->done, sizeof(int), cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaMemcpy(&st, dState, sizeof(GState), cudaMemcpyDeviceToHost));
  g.copy(R, g.cgX, g.nNo);
  res.success = st.success != 0;
  res.itr = st.itr;
  res.fNorm = std::sqrt(st.err);
  res.dB = 5.0 * std::log(st.err / (iNorm * iNorm));
  return res;
}

}  // namespace svgpu
