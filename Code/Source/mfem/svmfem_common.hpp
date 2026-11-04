// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// Common definitions of the MFEM GPU experiment: the device, kernel launch
// helpers built on mfem::forall, block reductions into device scalars,
// mapped host memory and CUDA graphs of kernel sequences.
//
// All kernels are written as mfem::forall / mfem::forall_2D bodies with
// MFEM_HOST_DEVICE lambdas, so that the same source runs on the GPU (CUDA)
// and on the CPU (SVMFEM_DEVICE=cpu). Vectors are mfem::Vector and
// mfem::Array<T>; the kernels use their device pointers (Read/Write).

#ifndef SVMFEM_COMMON_HPP
#define SVMFEM_COMMON_HPP

#include "mfem.hpp"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace svmfem {

using mfem::Array;
using mfem::Vector;

constexpr int TPB = 256;   // threads per block of the reduction kernels

/// Configure the MFEM device once ("cuda" or "cpu")
void init_device(const char* device);
/// True when the kernels run on the GPU
bool on_gpu();
/// Wait for the device work of this thread's stream
void sync();
/// Verbose messages (SV_GPU_VERBOSE)
bool verbose();
/// Release the objects of the assembly and of the solver (svmfem_finalize)
void destroy_fluid();
void destroy_solver();
/// Wall-clock time in seconds
double now();
/// Trace message to stderr (SVMFEM_DEBUG=1)
void trace(const char* what);

inline int blocks(int n) { return (n + TPB - 1) / TPB; }

[[noreturn]] inline void fail(const std::string& what)
{
  throw std::runtime_error("svmfem: " + what);
}

// ---------------------------------------------------------------------------
// Device arrays: mfem::Vector and mfem::Array<T> used only through their
// device pointers (the host side is never read, except by explicit copies)

/// Allocate a device vector of n doubles (host memory stays unused). The
/// device memory is allocated here, so that no allocation happens inside a
/// graph capture.
inline void dalloc(Vector& v, int n)
{
  v.SetSize(n);
  v.UseDevice(true);
  v.Write();
}

template <typename T> inline void dalloc(Array<T>& a, int n)
{
  a.SetSize(n);
  a.UseDevice(true);
  a.Write();
}

/// Device vector of n doubles with the values of a host array (copied once)
inline void dupload(Vector& v, const double* h, int n)
{
  v.SetSize(n);
  v.UseDevice(true);
  double* p = v.HostWrite();
  for (int i = 0; i < n; i++) p[i] = h[i];
  v.Read();   // host -> device
}

template <typename T> inline void dupload(Array<T>& a, const T* h, int n)
{
  a.SetSize(n);
  a.UseDevice(true);
  T* p = a.HostWrite();
  for (int i = 0; i < n; i++) p[i] = h[i];
  a.Read();
}

template <typename T> inline void dupload(Array<T>& a, const std::vector<T>& h)
{
  dupload(a, h.data(), (int)h.size());
}

/// Copy n values of a device vector to a host array (synchronizes)
void ddownload(const Vector& v, double* h, int n);
/// Copy n host values to a device vector (synchronous)
void dcopy_in(Vector& v, const double* h, int n);

// ---------------------------------------------------------------------------
// Kernels

/// Element-wise kernel: body(i) for i < n
template <typename BODY> inline void launch(int n, BODY&& body)
{
  if (n > 0) mfem::forall(n, body);
}

// Simple kernels (defined in svmfem_device.cu: with nvcc and MSVC, the host
// wrapper of an extended lambda defined in an inline header function calls
// a null pointer)

/// Set n values
void fill(double* x, int n, double v);
void fill(float* x, int n, float v);
void copy(const double* src, double* dst, int n);
void copy(const float* src, float* dst, int n);
/// sc[slot] += a . b
void dot(int n, const double* a, const double* b, double* sc, int slot);

/// Slots of the device scalars that a block reduction accumulates into
template <int NR> struct Slots {
  int s[NR];
};

inline Slots<1> slots(int a) { return {{a}}; }
inline Slots<2> slots(int a, int b) { return {{a, b}}; }

/// Block-reduced kernel: body(i, v) sets v[0..NR) for i < n; the sums over
/// each block of TPB threads are added atomically to sc[slots.s[k]] (which
/// must be zero before the first kernel accumulating into them). This is
/// the device-scalar replacement of the host-synchronizing dot products.
template <int NR, typename BODY> void launch_red(int n, double* sc, Slots<NR> sl, BODY&& body)
{
  const int nb = blocks(n);
  if (nb == 0) return;
  mfem::forall_2D(nb, TPB, 1, [=] MFEM_HOST_DEVICE (int blk) {
    MFEM_SHARED double sh[NR][TPB];
    MFEM_FOREACH_THREAD(t, x, TPB) {
      const int i = blk * TPB + t;
      double v[NR];
      for (int k = 0; k < NR; k++) v[k] = 0.0;
      if (i < n) body(i, v);
      for (int k = 0; k < NR; k++) sh[k][t] = v[k];
    }
    MFEM_SYNC_THREAD;
    for (int s = TPB / 2; s > 0; s >>= 1) {
      MFEM_FOREACH_THREAD(t, x, TPB) {
        if (t < s) {
          for (int k = 0; k < NR; k++) sh[k][t] += sh[k][t + s];
        }
      }
      MFEM_SYNC_THREAD;
    }
    MFEM_FOREACH_THREAD(t, x, TPB) {
      if (t < NR) AtomicAdd(sc[sl.s[t]], sh[t][0]);
    }
  });
}

// ---------------------------------------------------------------------------
// Mapped host memory: pinned host memory that kernels write (results of
// reductions read by the host after a synchronization) or read (coefficients
// computed by the host). With the CPU device it is plain host memory.

struct Vec8 {
  double v[8];
};

class Mapped {
 public:
  explicit Mapped(int n);
  ~Mapped();
  Mapped(const Mapped&) = delete;
  Mapped& operator=(const Mapped&) = delete;
  int n = 0, nvec = 0;
  double* host = nullptr;   // host pointer (the values)
  Vec8* dev = nullptr;      // the same memory as seen by the kernels, in 64-byte vectors
  double& operator[](int i) { return host[i]; }
};

/// dst[v0 + t] = src[8 (v0 + t) .. +8] for t < nvec (64-byte writes to mapped memory)
void publish8(const double* src, Vec8* dst, int v0, int nvec);
/// dst[8 t .. +8] = src[t] for t < nvec (mapped memory to a device vector)
void fetch8(const Vec8* src, double* dst, int nvec);

inline int pad8(int n) { return (n + 7) / 8 * 8; }

// ---------------------------------------------------------------------------
// Sliced ELLPACK layout (SELL-C with C = 32 rows per slice, slice widths
// rounded up to multiples of U = 4): the entries of a slice are stored
// column-major, so that one thread per row reads coalesced memory, and the
// row loops are unrolled 4 times. smap is the CSR entry of each slot (-1 for
// the padding, which has column 0 and value 0). The matrices of the solver
// are gathered from their CSR values into this layout in every solve.

struct Sell {
  static constexpr int C = 32, U = 4;
  int n = 0, nslices = 0, total = 0;
  Array<int> slice_off, cols, smap;   // device
  double padding = 0.0;               // total / nnz
  void build(int n, const std::vector<int>& off, const std::vector<int>& col);
};

/// Vs[m q + k] = V[m smap[q] + k] (0 for the padding), k < m
void sell_gather(const Sell& s, const double* V, int m, double* Vs);
/// Vs[q] = (float)V[smap[q]]
void sell_gather_f(const Sell& s, const double* V, float* Vs);

#define SELL_ROW_LOOP(q0, q1, BODY)   for (int q = (q0); q < (q1); q += 4 * 32) {     const int c0 = cols[q], c1 = cols[q + 32], c2 = cols[q + 64], c3 = cols[q + 96];     BODY   }

/// Row i of a float SELL matrix times a float vector
MFEM_HOST_DEVICE inline float sell_row_ff(int i, const int* soff, const int* cols, const float* V, const float* x)
{
  const int s = i / 32, r = i % 32;
  float acc = 0.0f;
  SELL_ROW_LOOP(soff[s] + r, soff[s + 1],
    acc += V[q] * x[c0] + V[q + 32] * x[c1] + V[q + 64] * x[c2] + V[q + 96] * x[c3];)
  return acc;
}

/// Row i of a float SELL matrix times a double vector (double accumulation)
MFEM_HOST_DEVICE inline double sell_row_fd(int i, const int* soff, const int* cols, const float* V, const double* x)
{
  const int s = i / 32, r = i % 32;
  double acc = 0.0;
  SELL_ROW_LOOP(soff[s] + r, soff[s + 1],
    acc = acc + ((double)V[q] * x[c0] + (double)V[q + 32] * x[c1] + (double)V[q + 64] * x[c2] + (double)V[q + 96] * x[c3]);)
  return acc;
}

/// Row i of a float SELL matrix times (d .* x) (Jacobi smoothing)
MFEM_HOST_DEVICE inline float sell_row_f_scaled(int i, const int* soff, const int* cols, const float* V, const float* d,
    const float* x)
{
  const int s = i / 32, r = i % 32;
  float acc = 0.0f;
  SELL_ROW_LOOP(soff[s] + r, soff[s + 1],
    acc += V[q] * (d[c0] * x[c0]) + V[q + 32] * (d[c1] * x[c1]) + V[q + 64] * (d[c2] * x[c2]) + V[q + 96] * (d[c3] * x[c3]);)
  return acc;
}

/// Row a of the 3x3 block matrix K (9 values per entry) times the 3-vector x
MFEM_HOST_DEVICE inline void sell_row_vv(int a, const int* soff, const int* cols, const double* K, const double* x,
    double acc[3])
{
  const int s = a / 32, r = a % 32;
  acc[0] = acc[1] = acc[2] = 0.0;
  for (int q = soff[s] + r; q < soff[s + 1]; q += 32) {
    const int c = cols[q];
    const double* k = K + 9 * (size_t)q;
    const double u0 = x[3 * c], u1 = x[3 * c + 1], u2 = x[3 * c + 2];
    acc[0] = acc[0] + k[0] * u0 + k[1] * u1 + k[2] * u2;
    acc[1] = acc[1] + k[3] * u0 + k[4] * u1 + k[5] * u2;
    acc[2] = acc[2] + k[6] * u0 + k[7] * u1 + k[8] * u2;
  }
}

/// Row a of the 1x3 block matrix D times the 3-vector x
MFEM_HOST_DEVICE inline double sell_row_vs(int a, const int* soff, const int* cols, const double* D, const double* x)
{
  const int s = a / 32, r = a % 32;
  double acc = 0.0;
  for (int q = soff[s] + r; q < soff[s + 1]; q += 32) {
    const int c = cols[q];
    acc = acc + D[3 * q] * x[3 * c] + D[3 * q + 1] * x[3 * c + 1] + D[3 * q + 2] * x[3 * c + 2];
  }
  return acc;
}

/// Row a of the 3x1 block matrix G times the scalar vector x
MFEM_HOST_DEVICE inline void sell_row_sv(int a, const int* soff, const int* cols, const double* G, const double* x,
    double acc[3])
{
  const int s = a / 32, r = a % 32;
  acc[0] = acc[1] = acc[2] = 0.0;
  for (int q = soff[s] + r; q < soff[s + 1]; q += 32) {
    const double u = x[cols[q]];
    acc[0] += G[3 * q] * u;
    acc[1] += G[3 * q + 1] * u;
    acc[2] += G[3 * q + 2] * u;
  }
}

/// Row a of the scalar matrix L times x
MFEM_HOST_DEVICE inline double sell_row_ss(int a, const int* soff, const int* cols, const double* L, const double* x)
{
  const int s = a / 32, r = a % 32;
  double acc = 0.0;
  SELL_ROW_LOOP(soff[s] + r, soff[s + 1],
    acc = acc + (L[q] * x[c0] + L[q + 32] * x[c1] + L[q + 64] * x[c2] + L[q + 96] * x[c3]);)
  return acc;
}

// ---------------------------------------------------------------------------
// CUDA graphs: a kernel sequence (emit) is captured on first use and
// replayed afterwards (SVMFEM_NOGRAPH=1 launches the kernels one by one;
// with the CPU device the kernels always run directly).

class GraphCache {
 public:
  ~GraphCache();
  /// Launch the kernel sequence emit() as the graph with this key
  void run(const std::string& key, const std::function<void()>& emit);
  /// Forget the captured graphs (after a change of the kernel arguments)
  void clear();

 private:
  std::map<std::string, void*> graphs;   // cudaGraphExec_t
};

}  // namespace svmfem

#endif
