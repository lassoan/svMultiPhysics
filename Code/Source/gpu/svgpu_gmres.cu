// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// One Arnoldi step of gmres::gmres (dof = nsd = 3) as one cooperative kernel:
//   w = K*u_i + resistance terms, preconditioner of the resistance terms,
//   modified Gram-Schmidt against u_0..u_i, normalization.
// The update w -= h_j u_j and the partial sums of the next dot product are
// computed in the same pass, so each Gram-Schmidt step needs one grid sync.

#include "svgpu_internal.h"

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } } while (0)

namespace svgpu {

namespace {

namespace cg = cooperative_groups;

constexpr int TPB = 256;
constexpr int LANES = 8;
constexpr int MAXF = 4;

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

__device__ inline double row_sum(double v)
{
  for (int o = LANES / 2; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o, LANES);
  return v;
}

struct ArnArgs {
  int nNo, i, nF, ld;            // ld: leading dimension of h (sD + 1)
  const int *rowS, *rowE, *col;
  const double* K;
  const int *fOff, *fGlob;
  const double *fValM, *coefAdd, *coefPre;
  double* u;                     // Krylov basis, 3*nNo per vector
  double* h;                     // column i of the Hessenberg matrix (device)
  double* acc;                   // accumulators, 64 x 4 slots
};

__global__ void k_arnoldi(ArnArgs a)
{
  cg::grid_group grid = cg::this_grid();
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nth = gridDim.x * blockDim.x;
  const int lane = threadIdx.x % LANES;
  const int gid0 = tid / LANES;
  const int ngroups = nth / LANES;
  const int nRowIt = ((a.nNo + ngroups - 1) / ngroups) * ngroups;
  const size_t n3 = 3 * (size_t)a.nNo;
  const double* ui = a.u + a.i * n3;
  double* w = a.u + (a.i + 1) * n3;
  // Accumulator slots: slot s is used by phase s; the next slot is cleared
  // by block 0 in the current phase (it was last read two phases ago)
  int slot = 0;
  auto next_acc = [&]() {
    double* cur = a.acc + (slot % 4) * 64;
    slot++;
    if (blockIdx.x == 0 && threadIdx.x < 64) a.acc[(slot % 4) * 64 + threadIdx.x] = 0.0;
    return cur;
  };

  // w = K*u_i, and the face sums valM.u_i of the resistance terms
  double* acc = next_acc();
  for (int i0 = gid0; i0 < nRowIt; i0 += ngroups) {
    int r = i0;
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    if (r < a.nNo) {
      for (int j = a.rowS[r] + lane; j <= a.rowE[r]; j += LANES) {
        int c = a.col[j];
        const double* k = a.K + 9 * (size_t)j;
        double u0 = ui[3 * c], u1 = ui[3 * c + 1], u2 = ui[3 * c + 2];
        s0 += k[0] * u0 + k[1] * u1 + k[2] * u2;
        s1 += k[3] * u0 + k[4] * u1 + k[5] * u2;
        s2 += k[6] * u0 + k[7] * u1 + k[8] * u2;
      }
    }
    s0 = row_sum(s0); s1 = row_sum(s1); s2 = row_sum(s2);
    if (lane == 0 && r < a.nNo) { w[3 * r] = s0; w[3 * r + 1] = s1; w[3 * r + 2] = s2; }
  }
  for (int f = 0; f < a.nF; f++) {
    double s = 0.0;
    for (int q = a.fOff[f] + tid; q < a.fOff[f + 1]; q += nth) {
      int Ac = a.fGlob[q];
      s += a.fValM[3 * q] * ui[3 * Ac] + a.fValM[3 * q + 1] * ui[3 * Ac + 1] + a.fValM[3 * q + 2] * ui[3 * Ac + 2];
    }
    s = block_sum(s);
    if (threadIdx.x == 0) atomicAdd(&acc[f], s);
  }
  grid.sync();

  if (a.nF > 0) {
    // w += valM*S (resistance), then the face sums valM.w of the preconditioner
    double S[MAXF];
    for (int f = 0; f < a.nF; f++) S[f] = a.coefAdd[f] * __ldcg(&acc[f]);
    acc = next_acc();
    for (int f = 0; f < a.nF; f++) {
      for (int q = a.fOff[f] + tid; q < a.fOff[f + 1]; q += nth) {
        int Ac = a.fGlob[q];
        w[3 * Ac] += a.fValM[3 * q] * S[f];
        w[3 * Ac + 1] += a.fValM[3 * q + 1] * S[f];
        w[3 * Ac + 2] += a.fValM[3 * q + 2] * S[f];
      }
    }
    grid.sync();
    for (int f = 0; f < a.nF; f++) {
      double s = 0.0;
      for (int q = a.fOff[f] + tid; q < a.fOff[f + 1]; q += nth) {
        int Ac = a.fGlob[q];
        s += a.fValM[3 * q] * w[3 * Ac] + a.fValM[3 * q + 1] * w[3 * Ac + 1] + a.fValM[3 * q + 2] * w[3 * Ac + 2];
      }
      s = block_sum(s);
      if (threadIdx.x == 0) atomicAdd(&acc[f], s);
    }
    grid.sync();
    for (int f = 0; f < a.nF; f++) S[f] = a.coefPre[f] * __ldcg(&acc[f]);
    for (int f = 0; f < a.nF; f++) {
      for (int q = a.fOff[f] + tid; q < a.fOff[f + 1]; q += nth) {
        int Ac = a.fGlob[q];
        w[3 * Ac] += a.fValM[3 * q] * S[f];
        w[3 * Ac + 1] += a.fValM[3 * q + 1] * S[f];
        w[3 * Ac + 2] += a.fValM[3 * q + 2] * S[f];
      }
    }
    grid.sync();
  }

  // Modified Gram-Schmidt: h_0 = u_0.w; then for each j: w -= h_j u_j and
  // the partial sums of u_{j+1}.w (or of w.w after the last vector)
  acc = next_acc();
  {
    double s = 0.0;
    for (int q = tid; q < (int)n3; q += nth) s += a.u[q] * w[q];
    s = block_sum(s);
    if (threadIdx.x == 0) atomicAdd(&acc[0], s);
  }
  grid.sync();
  for (int j = 0; j <= a.i; j++) {
    double hj = __ldcg(&acc[0]);
    if (tid == 0) a.h[j] = hj;
    const double* uj = a.u + j * n3;
    const double* un = (j < a.i) ? a.u + (j + 1) * n3 : w;
    acc = next_acc();
    double s = 0.0;
    for (int q = tid; q < (int)n3; q += nth) {
      double wq = w[q] - hj * uj[q];
      w[q] = wq;
      s += un[q] * wq;
    }
    s = block_sum(s);
    if (threadIdx.x == 0) atomicAdd(&acc[0], s);
    grid.sync();
  }
  // Normalization (each thread scales the entries it updated)
  double nrm = sqrt(__ldcg(&acc[0]));
  if (tid == 0) a.h[a.i + 1] = nrm;
  for (int q = tid; q < (int)n3; q += nth) w[q] = w[q] * (1.0 / nrm);
}

double* accBuf = nullptr;
double* hDev = nullptr;
int nbArn = 0;

}  // namespace

void Gpu::arnoldi(double* u, int i, double* hcol)
{
  if (!accBuf) {
    accBuf = alloc(4 * 64);
    hDev = alloc(4096);
    int dev, nSM, perSM;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, k_arnoldi, TPB, 0));
    nbArn = nSM * std::min(perSM, 1);
  }
  if (i + 2 > 4096) throw std::runtime_error("svgpu: Krylov space too large");
  if (nCoupled > MAXF) throw std::runtime_error("svgpu: too many coupled faces");
  CUDA_CHECK(cudaMemsetAsync(accBuf, 0, 64 * sizeof(double)));
  ArnArgs a{nNo, i, nCoupled, 0, rowS, rowE, col, mK, fOff, fGlob, fValM, fCoefAdd, fCoefPre, u, hDev, accBuf};
  void* args[] = {&a};
  CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_arnoldi, nbArn, TPB, args, 0, 0));
  download(hcol, hDev, i + 2);
}

}  // namespace svgpu
