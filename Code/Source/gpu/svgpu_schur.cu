// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// Schur complement CG of the GPU NS solver with the explicit Schur complement
// S = L - Gt*G. The preconditioner of the resistance boundary conditions,
// applied to G*P in cgrad::schur, is a sum of rank-1 terms:
//   S_eff*P = S*P - sum_f c_f w_f (u_f . P),  u_f = G^T v_f,  w_f = Gt v_f,
// with v_f the valM vectors of face f. One sparse product and three grid
// synchronizations per iteration.

#include "svgpu_internal.h"

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } } while (0)

namespace svgpu {

namespace {

namespace cg = cooperative_groups;

constexpr int TPB = 256;
constexpr int LANES = 16;
constexpr int MAXF = 4;          // coupled faces
constexpr int NV = 1 + 2 * MAXF; // reduced values per iteration
constexpr int NPART = 1024;

inline int blocks(int n) { return (n + TPB - 1) / TPB; }

template <typename T> T* upload_vec(const std::vector<T>& h)
{
  T* d;
  CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T)));
  if (!h.empty()) CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

// Block sum of n values (n <= NV), result in all threads
__device__ void block_sum_n(double* v, int n)
{
  __shared__ double sh[NV][TPB / 32];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  for (int k = 0; k < n; k++) {
    double x = v[k];
    for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffff, x, o);
    if (lane == 0) sh[k][warp] = x;
  }
  __syncthreads();
  for (int k = 0; k < n; k++) {
    double x = 0.0;
    for (int w = 0; w < TPB / 32; w++) x += sh[k][w];
    v[k] = x;
  }
  __syncthreads();
}

// Sum over all blocks of nv partial sums (part[k*NPART + block]), computed by
// warp 0 of every block and broadcast to the block
__device__ void grid_sums(const double* part, int nb, int nv, double* out)
{
  __shared__ double sh[NV];
  if (threadIdx.x < 32) {
    for (int k = 0; k < nv; k++) {
      double x = 0.0;
      for (int b = threadIdx.x; b < nb; b += 32) x += part[k * NPART + b];
      for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffff, x, o);
      if (threadIdx.x == 0) sh[k] = x;
    }
  }
  __syncthreads();
  for (int k = 0; k < nv; k++) out[k] = sh[k];
  __syncthreads();
}

__device__ inline double row_sum(double v)
{
  for (int o = LANES / 2; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o, LANES);
  return v;
}

__global__ void k_schur_values(int nnzS, const int* pairPtr, const int* pairJ, const int* pairL, const int* lIdx,
    const double* L, const double* Gt, const double* G, float* S)
{
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= nnzS) return;
  double s = 0.0;
  for (int q = pairPtr[e]; q < pairPtr[e + 1]; q++) {
    int j = pairJ[q], l = pairL[q];
    s += Gt[3 * j] * G[3 * l] + Gt[3 * j + 1] * G[3 * l + 1] + Gt[3 * j + 2] * G[3 * l + 2];
  }
  int li = lIdx[e];
  S[e] = (float)((li >= 0 ? L[li] : 0.0) - s);
}

// u_f = G^T v_f (atomics over the rows of the face nodes)
__global__ void k_face_u(int nNo, int nF, const int* fOff, const int* fGlob, const double* fValM, const int* rowS,
    const int* rowE, const int* col, const double* G, double* U)
{
  int f = blockIdx.y;
  int q = fOff[f] + blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= fOff[f + 1]) return;
  int Ac = fGlob[q];
  const double* v = fValM + 3 * q;
  for (int l = rowS[Ac]; l <= rowE[Ac]; l++) {
    atomicAdd(&U[(size_t)f * nNo + col[l]], v[0] * G[3 * l] + v[1] * G[3 * l + 1] + v[2] * G[3 * l + 2]);
  }
}

// w_f = Gt v_f
__global__ void k_face_w(int nNo, int nF, const int* rowS, const int* rowE, const int* col, const double* Gt,
    const double* Vd, double* W)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nNo) return;
  for (int f = 0; f < nF; f++) {
    const double* v = Vd + (size_t)f * 3 * nNo;
    double s = 0.0;
    for (int j = rowS[i]; j <= rowE[i]; j++) {
      int c = col[j];
      s += Gt[3 * j] * v[3 * c] + Gt[3 * j + 1] * v[3 * c + 1] + Gt[3 * j + 2] * v[3 * c + 2];
    }
    W[(size_t)f * nNo + i] = s;
  }
}

struct SArgs {
  int nNo, mItr, nF;
  const int *sRow, *sCol;
  const float* sVal;
  const double *U, *W, *coef;
  double *X, *P, *R, *SPp, *part;
  double err, eps;
  double* out;   // err, errO, last_i, success
  unsigned long long* tph;
};

__device__ inline unsigned long long gtimer()
{
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define PHASE(k) if (a.tph && blockIdx.x == 0 && threadIdx.x == 0) { unsigned long long t_ = gtimer(); a.tph[k] += t_ - tlast; tlast = t_; }

__global__ void k_cg_s(SArgs a)
{
  cg::grid_group grid = cg::this_grid();
  const int nb = gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nth = nb * blockDim.x;
  const int lane = threadIdx.x % LANES;
  const int gid0 = tid / LANES;
  const int ngroups = nth / LANES;
  const int nRowIt = ((a.nNo + ngroups - 1) / ngroups) * ngroups;
  const int nv = 1 + 2 * a.nF;
  double err = a.err, errO = err;
  int last_i = 0, success = 0;
  double v[NV];
  unsigned long long tlast = (a.tph && blockIdx.x == 0 && threadIdx.x == 0) ? gtimer() : 0;

  for (int it = 0; it < a.mItr; it++) {
    last_i = it;
    if (err < a.eps) { success = 1; break; }
    errO = err;

    // SPp = S*P, partial sums of P.SPp, u_f.P, w_f.P
    for (int k = 0; k < nv; k++) v[k] = 0.0;
    for (int i0 = gid0; i0 < nRowIt; i0 += ngroups) {
      int i = i0;
      double s = 0.0;
      if (i < a.nNo) {
        for (int j = a.sRow[i] + lane; j < a.sRow[i + 1]; j += LANES) s += (double)a.sVal[j] * a.P[a.sCol[j]];
      }
      s = row_sum(s);
      if (lane == 0 && i < a.nNo) {
        a.SPp[i] = s;
        v[0] += a.P[i] * s;
      }
    }
    for (int f = 0; f < a.nF; f++) {
      for (int i = tid; i < a.nNo; i += nth) {
        double p = a.P[i];
        v[1 + 2 * f] += a.U[(size_t)f * a.nNo + i] * p;
        v[2 + 2 * f] += a.W[(size_t)f * a.nNo + i] * p;
      }
    }
    block_sum_n(v, nv);
    // Accumulators alternate between two slots; the slot of the next
    // iteration is cleared by block 0 after the first sync
    double* acc = a.part + (it % 2) * 16;
    double* accNext = a.part + ((it + 1) % 2) * 16;
    if (threadIdx.x < nv) atomicAdd(&acc[threadIdx.x], v[threadIdx.x]);
    grid.sync();
    PHASE(0)
    if (blockIdx.x == 0 && threadIdx.x < 16) accNext[threadIdx.x] = 0.0;

    // Sums, alpha
    for (int k = 0; k < nv; k++) v[k] = __ldcg(&acc[k]);
    double pSP = v[0];
    double Uf[MAXF];
    for (int f = 0; f < a.nF; f++) {
      Uf[f] = v[1 + 2 * f];
      pSP -= a.coef[f] * v[2 + 2 * f] * Uf[f];
    }
    double alpha = errO / pSP;

    // X += alpha*P, R -= alpha*SP, partial sums of R.R
    double rr = 0.0;
    for (int i = tid; i < a.nNo; i += nth) {
      double sp = a.SPp[i];
      for (int f = 0; f < a.nF; f++) sp -= a.coef[f] * a.W[(size_t)f * a.nNo + i] * Uf[f];
      a.X[i] = a.X[i] + alpha * a.P[i];
      double r = a.R[i] + (-alpha) * sp;
      a.R[i] = r;
      rr += r * r;
    }
    v[0] = rr;
    block_sum_n(v, 1);
    if (threadIdx.x == 0) atomicAdd(&acc[NV], v[0]);
    grid.sync();
    PHASE(1)

    err = __ldcg(&acc[NV]);
    double c1 = errO / err, c2 = err / errO;
    for (int i = tid; i < a.nNo; i += nth) {
      double t = a.P[i] + c1 * a.R[i];
      a.P[i] = c2 * t;
    }
    grid.sync();
    PHASE(2)
  }
  if (tid == 0) {
    a.out[0] = err;
    a.out[1] = errO;
    a.out[2] = last_i;
    a.out[3] = success;
  }
}

template <int LN>
__global__ void k_spmv_s(int nNo, const int* sRow, const int* sCol, const float* sVal, const double* P, double* Y)
{
  int g = (blockIdx.x * blockDim.x + threadIdx.x) / LN;
  int lane = threadIdx.x % LN;
  double s = 0.0;
  if (g < nNo) {
    for (int j = sRow[g] + lane; j < sRow[g + 1]; j += LN) s += (double)sVal[j] * P[sCol[j]];
  }
  for (int o = LN / 2; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o, LN);
  if (lane == 0 && g < nNo) Y[g] = s;
}

__global__ void k_axpy_s(int n, double a, const double* x, double* y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = y[i] + a * x[i];
}

}  // namespace

void Gpu::schur_setup()
{
  // Structure of S = L - Gt*G: the nodes connected through a common neighbor
  std::vector<int> sRowH(nNo + 1, 0), sColH, lIdxH, pairPtrH{0}, pairJH, pairLH;
  std::vector<int> pos(nNo, -1);
  std::vector<int> cols;
  std::vector<std::vector<std::pair<int, int>>> pairs;
  for (int i = 0; i < nNo; i++) {
    cols.clear();
    for (int j = hRowS[i]; j <= hRowE[i]; j++) {
      int m = hCol[j];
      for (int l = hRowS[m]; l <= hRowE[m]; l++) {
        int k = hCol[l];
        if (pos[k] < 0) {
          pos[k] = 1;
          cols.push_back(k);
        }
      }
    }
    std::sort(cols.begin(), cols.end());
    for (size_t q = 0; q < cols.size(); q++) pos[cols[q]] = (int)q;
    pairs.assign(cols.size(), {});
    for (int j = hRowS[i]; j <= hRowE[i]; j++) {
      int m = hCol[j];
      for (int l = hRowS[m]; l <= hRowE[m]; l++) pairs[pos[hCol[l]]].push_back({j, l});
    }
    std::vector<int> li(cols.size(), -1);
    for (int j = hRowS[i]; j <= hRowE[i]; j++) li[pos[hCol[j]]] = j;
    for (size_t q = 0; q < cols.size(); q++) {
      sColH.push_back(cols[q]);
      lIdxH.push_back(li[q]);
      for (auto& pr : pairs[q]) {
        pairJH.push_back(pr.first);
        pairLH.push_back(pr.second);
      }
      pairPtrH.push_back((int)pairJH.size());
    }
    sRowH[i + 1] = (int)sColH.size();
    for (int k : cols) pos[k] = -1;
  }
  nnzS = (int)sColH.size();
  sRow = upload_vec(sRowH);
  sCol = upload_vec(sColH);
  sLidx = upload_vec(lIdxH);
  sPairPtr = upload_vec(pairPtrH);
  sPairJ = upload_vec(pairJH);
  sPairL = upload_vec(pairLH);
  CUDA_CHECK(cudaMalloc(&sVal, std::max(nnzS, 1) * sizeof(float)));
  cgSPp = alloc(nNo);
  cgPart = alloc((NV + 1) * (size_t)NPART);
  fU = alloc((size_t)MAXF * nNo);
  fW = alloc((size_t)MAXF * nNo);
  fVd = alloc((size_t)MAXF * 3 * nNo);
  if (!(getenv("SVGPU_SCHUR_PREC") && std::string(getenv("SVGPU_SCHUR_PREC")) == "none")) {
    amg_setup_impl(*this, sRowH, sColH);
  }
}

void Gpu::schur_numeric()
{
  k_schur_values<<<blocks(nnzS), TPB>>>(nnzS, sPairPtr, sPairJ, sPairL, sLidx, mL, Gt, mG, sVal);
  CUDA_CHECK(cudaGetLastError());
  if (!(getenv("SVGPU_SCHUR_PREC") && std::string(getenv("SVGPU_SCHUR_PREC")) == "none")) {
    amg_numeric_impl(*this, sRow, sCol, sVal, nnzS);
  }
  if (nCoupled > MAXF) throw std::runtime_error("svgpu: too many coupled faces");
  if (nCoupled > 0) {
    // Dense valM vectors of the faces
    std::vector<int> off(nCoupled + 1), glob;
    std::vector<double> valM;
    CUDA_CHECK(cudaMemcpy(off.data(), fOff, (nCoupled + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    glob.resize(off[nCoupled]);
    valM.resize(3 * (size_t)off[nCoupled]);
    CUDA_CHECK(cudaMemcpy(glob.data(), fGlob, glob.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(valM.data(), fValM, valM.size() * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> vd((size_t)nCoupled * 3 * nNo, 0.0);
    for (int f = 0; f < nCoupled; f++) {
      for (int q = off[f]; q < off[f + 1]; q++) {
        for (int i = 0; i < 3; i++) vd[(size_t)f * 3 * nNo + 3 * glob[q] + i] = valM[3 * q + i];
      }
    }
    CUDA_CHECK(cudaMemcpy(fVd, vd.data(), vd.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(fU, 0, (size_t)nCoupled * nNo * sizeof(double)));
    int maxN = 0;
    for (int f = 0; f < nCoupled; f++) maxN = std::max(maxN, off[f + 1] - off[f]);
    dim3 grid(std::max(1, blocks(maxN)), nCoupled);
    k_face_u<<<grid, TPB>>>(nNo, nCoupled, fOff, fGlob, fValM, rowS, rowE, col, mG, fU);
    CUDA_CHECK(cudaGetLastError());
    k_face_w<<<blocks(nNo), TPB>>>(nNo, nCoupled, rowS, rowE, col, Gt, fVd, fW);
    CUDA_CHECK(cudaGetLastError());
  }
}

Gpu::CgResult Gpu::cg_schur_s(double* R, int mItr, double absTol, double relTol)
{
  if (!(getenv("SVGPU_SCHUR_PREC") && std::string(getenv("SVGPU_SCHUR_PREC")) == "none")) {
    return cg_schur_amg_impl(*this, R, mItr, absTol, relTol);
  }
  CgResult res{};
  double iNorm = std::sqrt(dot(R, R, nNo));
  res.iNorm = iNorm;
  double eps = std::pow(std::max(absTol, relTol * iNorm), 2.0);
  copy(cgR, R, nNo);
  copy(cgP, R, nNo);
  zero(cgX, nNo);

  static int nb = 0;
  if (!nb) {
    int dev, nSM, perSM;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, k_cg_s, TPB, 0));
    int maxPer = getenv("SVGPU_PERSM") ? atoi(getenv("SVGPU_PERSM")) : 6;
    nb = std::min(nSM * std::min(perSM, maxPer), NPART);
    printf("svgpu: Schur CG with %d blocks (%d per SM possible), nnz(S) = %d\n", nb, perSM, nnzS);
  }
  static unsigned long long* tph = nullptr;
  static long long nit = 0;
  static int ncall = 0;
  if (getenv("SVGPU_PHASES") && !tph) {
    CUDA_CHECK(cudaMalloc(&tph, 64));
    CUDA_CHECK(cudaMemset(tph, 0, 64));
  }
  if (getenv("SVGPU_BENCH")) {
    static bool done = false;
    if (!done) {
      done = true;
      cudaEvent_t e0, e1;
      cudaEventCreate(&e0); cudaEventCreate(&e1);
      auto bench = [&](const char* name, auto launch) {
        launch();
        cudaEventRecord(e0);
        for (int r = 0; r < 200; r++) launch();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        printf("bench %s: %.2f us\n", name, 1000.0 * ms / 200);
      };
      bench("spmv S lanes=4", [&]() { k_spmv_s<4><<<(nNo * 4 + 255) / 256, 256>>>(nNo, sRow, sCol, sVal, cgP, cgSPp); });
      bench("spmv S lanes=8", [&]() { k_spmv_s<8><<<(nNo * 8 + 255) / 256, 256>>>(nNo, sRow, sCol, sVal, cgP, cgSPp); });
      bench("spmv S lanes=16", [&]() { k_spmv_s<16><<<(nNo * 16 + 255) / 256, 256>>>(nNo, sRow, sCol, sVal, cgP, cgSPp); });
      bench("spmv S lanes=32", [&]() { k_spmv_s<32><<<(nNo * 32 + 255) / 256, 256>>>(nNo, sRow, sCol, sVal, cgP, cgSPp); });
      bench("axpy n=nNo", [&]() { k_axpy_s<<<(nNo + 255) / 256, 256>>>(nNo, 1e-9, cgP, cgSPp); });
      bench("dot n=nNo", [&]() { dot_dev(cgP, cgP, nNo, scal + 800); });
    }
  }
  if (getenv("SVGPU_DUMP")) {
    static int ndump = 0;
    if (ndump++ == 12) {
      std::vector<int> r(nNo + 1), c(nnzS);
      std::vector<float> v(nnzS);
      std::vector<double> b(nNo), u((size_t)nCoupled * nNo), w((size_t)nCoupled * nNo), cf(nCoupled);
      CUDA_CHECK(cudaMemcpy(r.data(), sRow, r.size() * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(c.data(), sCol, c.size() * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(v.data(), sVal, v.size() * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(b.data(), R, b.size() * 8, cudaMemcpyDeviceToHost));
      if (nCoupled) {
        CUDA_CHECK(cudaMemcpy(u.data(), fU, u.size() * 8, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(w.data(), fW, w.size() * 8, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cf.data(), fCoefPre, cf.size() * 8, cudaMemcpyDeviceToHost));
      }
      FILE* fp = fopen("schur_dump.bin", "wb");
      int hdr[3] = {nNo, nnzS, nCoupled};
      fwrite(hdr, 4, 3, fp);
      fwrite(r.data(), 4, r.size(), fp); fwrite(c.data(), 4, c.size(), fp); fwrite(v.data(), 4, v.size(), fp);
      fwrite(b.data(), 8, b.size(), fp); fwrite(u.data(), 8, u.size(), fp); fwrite(w.data(), 8, w.size(), fp);
      fwrite(cf.data(), 8, cf.size(), fp);
      fclose(fp);
      printf("svgpu: dumped the Schur complement system\n");
    }
  }
  CUDA_CHECK(cudaMemset(cgPart, 0, 32 * sizeof(double)));
  double* out = scal + 900;
  SArgs a{nNo, mItr, nCoupled, sRow, sCol, sVal, fU, fW, fCoefPre, cgX, cgP, cgR, cgSPp, cgPart,
      iNorm * iNorm, eps, out, tph};
  void* args[] = {&a};
  CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_cg_s, nb, TPB, args, 0, 0));
  double h[4];
  download(h, out, 4);
  copy(R, cgX, nNo);
  if (tph) {
    nit += (long long)h[2];
    if (++ncall % 8 == 0) {
      unsigned long long t[3];
      CUDA_CHECK(cudaMemcpy(t, tph, sizeof(t), cudaMemcpyDeviceToHost));
      printf("Schur CG phases (us/it): SpMV+dots %.1f update %.1f P %.1f\n", 1e-3 * t[0] / nit, 1e-3 * t[1] / nit,
          1e-3 * t[2] / nit);
    }
  }
  res.success = h[3] != 0.0;
  res.itr = (int)h[2];
  res.fNorm = std::sqrt(h[0]);
  res.dB = (h[1] < std::numeric_limits<double>::epsilon()) ? 0.0 : 5.0 * std::log(h[0] / h[1]);
  return res;
}

}  // namespace svgpu
