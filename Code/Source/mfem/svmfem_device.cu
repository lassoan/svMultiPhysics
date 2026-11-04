// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// Device configuration, mapped host memory and CUDA graphs of the MFEM GPU
// experiment (svmfem_common.hpp).

#include "svmfem_common.hpp"
#include "svmfem.h"

#ifdef MFEM_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstring>

namespace svmfem {

namespace {

mfem::Device* device = nullptr;
bool gpu = false;
bool use_graphs = true;

#ifdef MFEM_USE_CUDA
#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fail(std::string("CUDA error: ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } } while (0)
#endif

}  // namespace

void init_device(const char* dev)
{
  if (device) return;
  std::string name = dev ? dev : "";
  if (name.empty()) name = getenv("SVMFEM_DEVICE") ? getenv("SVMFEM_DEVICE") : "cuda";
#ifndef MFEM_USE_CUDA
#ifdef MFEM_USE_OPENMP
  const bool known = name == "cpu" || name == "omp";
#else
  const bool known = name == "cpu";
#endif
  if (!known) {
    fprintf(stderr, "svmfem: built without CUDA, running on the CPU\n");
    name = "cpu";
  }
#endif
  device = new mfem::Device(name.c_str());
  gpu = mfem::Device::Allows(mfem::Backend::CUDA_MASK);
  // Registered after the CUDA runtime initialization, so that it runs before
  // the runtime's own exit handler
  atexit(svmfem_finalize);
  use_graphs = gpu && getenv("SVMFEM_NOGRAPH") == nullptr;
  if (verbose()) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    device->Print();
    printf("svmfem: MFEM %s, device %s, graphs %d\n", MFEM_VERSION_STRING, name.c_str(), (int)use_graphs);
  }
}

bool on_gpu()
{
  return gpu;
}

bool verbose()
{
  static const bool v = getenv("SV_GPU_VERBOSE") != nullptr;
  return v;
}

void trace(const char* what)
{
  static const bool on = getenv("SVMFEM_DEBUG") != nullptr;
  if (on) fprintf(stderr, "svmfem: %s\n", what);
}

double now()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void sync()
{
#ifdef MFEM_USE_CUDA
  if (gpu) CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
#endif
}

void ddownload(const Vector& v, double* h, int n)
{
#ifdef MFEM_USE_CUDA
  if (gpu) {
    CUDA_CHECK(cudaMemcpyAsync(h, v.Read(), n * sizeof(double), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    sync();
    return;
  }
#endif
  std::memcpy(h, v.Read(), n * sizeof(double));
}

void dcopy_in(Vector& v, const double* h, int n)
{
#ifdef MFEM_USE_CUDA
  if (gpu) {
    CUDA_CHECK(cudaMemcpyAsync(v.Write(), h, n * sizeof(double), cudaMemcpyHostToDevice, cudaStreamPerThread));
    sync();
    return;
  }
#endif
  std::memcpy(v.Write(), h, n * sizeof(double));
}

// ---------------------------------------------------------------------------
// Simple kernels

void fill(double* x, int n, double v)
{
  launch(n, [=] MFEM_HOST_DEVICE (int i) { x[i] = v; });
}

void fill(float* x, int n, float v)
{
  launch(n, [=] MFEM_HOST_DEVICE (int i) { x[i] = v; });
}

void copy(const double* src, double* dst, int n)
{
  launch(n, [=] MFEM_HOST_DEVICE (int i) { dst[i] = src[i]; });
}

void copy(const float* src, float* dst, int n)
{
  launch(n, [=] MFEM_HOST_DEVICE (int i) { dst[i] = src[i]; });
}

void dot(int n, const double* a, const double* b, double* sc, int slot)
{
  launch_red<1>(n, sc, slots(slot), [=] MFEM_HOST_DEVICE (int i, double* v) { v[0] = a[i] * b[i]; });
}

void publish8(const double* src, Vec8* dst, int v0, int nvec)
{
  launch(nvec, [=] MFEM_HOST_DEVICE (int t) {
    Vec8 v;
    for (int k = 0; k < 8; k++) v.v[k] = src[8 * (v0 + t) + k];
    dst[v0 + t] = v;
  });
}

void fetch8(const Vec8* src, double* dst, int nvec)
{
  launch(nvec, [=] MFEM_HOST_DEVICE (int t) {
    Vec8 v = src[t];
    for (int k = 0; k < 8; k++) dst[8 * t + k] = v.v[k];
  });
}

// ---------------------------------------------------------------------------
// SELL layout

void Sell::build(int n_, const std::vector<int>& off, const std::vector<int>& col)
{
  n = n_;
  nslices = (n + C - 1) / C;
  std::vector<int> h_soff(nslices + 1, 0);
  for (int s = 0; s < nslices; s++) {
    int w = 0;
    for (int i = s * C; i < std::min((s + 1) * C, n); i++) w = std::max(w, off[i + 1] - off[i]);
    w = (w + U - 1) / U * U;
    h_soff[s + 1] = h_soff[s] + w * C;
  }
  total = h_soff[nslices];
  std::vector<int> h_cols(total, 0), h_smap(total, -1);
  for (int i = 0; i < n; i++) {
    const int s = i / C, r = i % C;
    for (int k = 0; k < off[i + 1] - off[i]; k++) {
      const int q = h_soff[s] + k * C + r;
      h_cols[q] = col[off[i] + k];
      h_smap[q] = off[i] + k;
    }
  }
  padding = off[n] > 0 ? (double)total / off[n] : 1.0;
  dupload(slice_off, h_soff);
  dupload(cols, h_cols);
  dupload(smap, h_smap);
}

void sell_gather(const Sell& s, const double* V, int m, double* Vs)
{
  const int* smap = s.smap.Read();
  launch(s.total, [=] MFEM_HOST_DEVICE (int q) {
    const int j = smap[q];
    for (int k = 0; k < m; k++) Vs[(size_t)m * q + k] = j >= 0 ? V[(size_t)m * j + k] : 0.0;
  });
}

void sell_gather_f(const Sell& s, const double* V, float* Vs)
{
  const int* smap = s.smap.Read();
  launch(s.total, [=] MFEM_HOST_DEVICE (int q) {
    const int j = smap[q];
    Vs[q] = j >= 0 ? (float)V[j] : 0.0f;
  });
}

// ---------------------------------------------------------------------------

Mapped::Mapped(int n_) : n(pad8(n_)), nvec(pad8(n_) / 8)
{
#ifdef MFEM_USE_CUDA
  if (gpu) {
    CUDA_CHECK(cudaHostAlloc(&host, n * sizeof(double), cudaHostAllocMapped));
    void* d;
    CUDA_CHECK(cudaHostGetDevicePointer(&d, host, 0));
    dev = (Vec8*)d;
  } else
#endif
  {
    host = new double[n];
    dev = (Vec8*)host;
  }
  std::memset(host, 0, n * sizeof(double));
}

Mapped::~Mapped()
{
#ifdef MFEM_USE_CUDA
  if (gpu) {
    cudaFreeHost(host);
    return;
  }
#endif
  delete[] host;
}

// ---------------------------------------------------------------------------

GraphCache::~GraphCache()
{
  clear();
}

void GraphCache::clear()
{
#ifdef MFEM_USE_CUDA
  for (auto& kv : graphs) cudaGraphExecDestroy((cudaGraphExec_t)kv.second);
#endif
  graphs.clear();
}

void GraphCache::run(const std::string& key, const std::function<void()>& emit)
{
#ifdef MFEM_USE_CUDA
  if (use_graphs) {
    auto it = graphs.find(key);
    if (it == graphs.end()) {
      // The kernels are launched on the per-thread default stream (the
      // library is compiled with --default-stream per-thread), which can be
      // captured; mfem::forall launches nothing else.
      cudaGraph_t graph;
      CUDA_CHECK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
      emit();
      CUDA_CHECK(cudaStreamEndCapture(cudaStreamPerThread, &graph));
      cudaGraphExec_t exec;
      CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
      CUDA_CHECK(cudaGraphDestroy(graph));
      it = graphs.emplace(key, (void*)exec).first;
    }
    CUDA_CHECK(cudaGraphLaunch((cudaGraphExec_t)it->second, cudaStreamPerThread));
    return;
  }
#endif
  emit();
}

}  // namespace svmfem

extern "C" SVMFEM_API void svmfem_finalize(void)
{
  using namespace svmfem;
  if (!device) return;
  destroy_solver();
  destroy_fluid();
  delete device;   // destroys the MFEM memory manager (device allocations)
  device = nullptr;
}

extern "C" SVMFEM_API int svmfem_init(const char* device)
{
  try {
    svmfem::init_device(device);
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "svmfem_init: %s\n", e.what());
    return 1;
  }
}
