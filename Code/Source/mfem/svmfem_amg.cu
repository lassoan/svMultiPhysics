// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// Aggregation AMG preconditioner for a scalar sparse matrix (the Schur
// complement of the NS solver).
//
// Hierarchy: standard aggregation of the matrix graph (a root with its free
// neighbors, remaining nodes join a neighboring aggregate), a prolongator
// with one weighted column per aggregate, Galerkin coarse matrices, the
// inverse of the coarsest matrix (at most CMAX nodes). The aggregation
// depends only on the matrix graph and is computed once on the host; the
// weights and the coarse values are computed on the device in every solve.
//
// Prolongator weights: the FSILS diagonal scaling of the matrix makes the
// near-kernel of the Schur complement far from constant, so a piecewise
// constant prolongator is poor (about 100 CG iterations on the iliac artery
// case against 40). The weights are the "improved candidate" of PyAMG: a few
// damped Jacobi sweeps on A b = 0 from b = 1, normalized in every aggregate.
//
// V-cycle: one damped Jacobi pre- and post-smoothing step on every level,
// with the damping omega / rho(D^-1 A) (rho from a power iteration on the
// device). The V-cycle is a fixed symmetric operator for a symmetric matrix,
// so that it can precondition CG. The matrices and vectors of the hierarchy
// are in single precision (the FP64 rate of GeForce GPUs is 1/64 of the FP32
// rate) in the SELL layout; the input and the output are double.
//
// All device work is emitted as kernel launches without host
// synchronization, so that the caller can capture it in CUDA graphs.

#include "svmfem_amg.hpp"

#include <algorithm>
#include <cmath>

namespace svmfem {

namespace {

int env_int(const char* name, int def)
{
  const char* v = getenv(name);
  return v ? atoi(v) : def;
}

double env_double(const char* name, double def)
{
  const char* v = getenv(name);
  return v ? atof(v) : def;
}

// Size of the coarsest level (dense inverse by Gauss-Jordan steps on the
// device). Two levels with an exact coarse solve need fewer CG iterations
// than three.
const int CMAX = env_int("SVMFEM_AMG_CMAX", 512);
const int NPOW0 = 15;   // power iteration steps in the first solve, then NPOW
const int NPOW = 4;
const int NCAND = env_int("SVMFEM_AMG_CAND", 16);   // candidate relaxation sweeps
const double OMEGA_C = env_double("SVMFEM_AMG_WC", 1.0);

}  // namespace

int aggregate(int n, const std::vector<int>& off, const std::vector<int>& col, std::vector<int>& agg)
{
  agg.assign(n, -1);
  int na = 0;
  // Pass 1: roots whose neighbors are all free
  for (int i = 0; i < n; i++) {
    if (agg[i] >= 0) continue;
    bool free = true;
    for (int j = off[i]; j < off[i + 1] && free; j++) free = (col[j] == i || agg[col[j]] < 0);
    if (!free) continue;
    agg[i] = na;
    for (int j = off[i]; j < off[i + 1]; j++) agg[col[j]] = na;
    na++;
  }
  // Pass 2: join a neighboring aggregate of pass 1
  std::vector<int> a2 = agg;
  for (int i = 0; i < n; i++) {
    if (agg[i] >= 0) continue;
    for (int j = off[i]; j < off[i + 1]; j++) {
      if (agg[col[j]] >= 0) {
        a2[i] = agg[col[j]];
        break;
      }
    }
  }
  agg = a2;
  // Pass 3: new aggregates of the remaining nodes and their free neighbors
  for (int i = 0; i < n; i++) {
    if (agg[i] >= 0) continue;
    agg[i] = na;
    for (int j = off[i]; j < off[i + 1]; j++) {
      if (agg[col[j]] < 0) agg[col[j]] = na;
    }
    na++;
  }
  return na;
}

Amg::Amg(int n, const std::vector<int>& off, const std::vector<int>& col, const Sell* sell0, double omega_)
{
  double t0 = now();
  omega = env_double("SVMFEM_AMG_W", omega_);
  levels.reserve(32);   // the device pointers of the levels must stay valid
  std::vector<int> h_off = off, h_col = col;
  while (true) {
    levels.emplace_back();
    AmgLevel& L = levels.back();
    const int l = (int)levels.size() - 1;
    L.n = n;
    L.nnz = (int)h_col.size();
    L.h_off = h_off;
    L.h_col = h_col;
    std::vector<int> diag(n, -1);
    for (int i = 0; i < n; i++) {
      for (int j = h_off[i]; j < h_off[i + 1]; j++) {
        if (h_col[j] == i) diag[i] = j;
      }
      if (diag[i] < 0) fail("AMG: a row without diagonal entry");
    }
    dupload(L.diag, diag);
    if (l > 0) {
      dupload(L.offsets, h_off);
      dupload(L.columns, h_col);
      L.sellL.build(n, h_off, h_col);
      L.sell = &L.sellL;
      dalloc(L.V, L.nnz);
      dalloc(L.Vs, L.sellL.total);
      L.pV = L.V.Read();
      L.pVs = L.Vs.Read();
      dalloc(L.r, n);
      dalloc(L.z, n);
      dalloc(L.z2, n);
      fill(L.r.Write(), n, 0.0f);
    } else {
      L.sell = sell0;
    }
    dalloc(L.Dinv, n);
    if (n <= CMAX) break;
    std::vector<int> agg;
    const int na = aggregate(n, h_off, h_col, agg);
    if (na >= n) break;
    // Coarse pattern: the aggregate pairs of the entries, sorted in every
    // row; the coarse entry of every fine entry (Galerkin sums)
    std::vector<int> rows_of(n);   // fine rows grouped by aggregate
    std::vector<int> cnt(na + 1, 0);
    for (int i = 0; i < n; i++) cnt[agg[i] + 1]++;
    for (int I = 0; I < na; I++) cnt[I + 1] += cnt[I];
    {
      std::vector<int> pos(cnt.begin(), cnt.end() - 1);
      for (int i = 0; i < n; i++) rows_of[pos[agg[i]]++] = i;
    }
    std::vector<int> c_off(na + 1, 0), c_col;
    std::vector<int> cmap(L.nnz), frow(L.nnz);
    std::vector<int> tmp;
    for (int I = 0; I < na; I++) {
      tmp.clear();
      for (int q = cnt[I]; q < cnt[I + 1]; q++) {
        int i = rows_of[q];
        for (int j = h_off[i]; j < h_off[i + 1]; j++) tmp.push_back(agg[h_col[j]]);
      }
      std::sort(tmp.begin(), tmp.end());
      tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
      const int base = (int)c_col.size();
      c_col.insert(c_col.end(), tmp.begin(), tmp.end());
      c_off[I + 1] = (int)c_col.size();
      for (int q = cnt[I]; q < cnt[I + 1]; q++) {
        int i = rows_of[q];
        for (int j = h_off[i]; j < h_off[i + 1]; j++) {
          frow[j] = i;
          int J = agg[h_col[j]];
          cmap[j] = base + (int)(std::lower_bound(tmp.begin(), tmp.end(), J) - tmp.begin());
        }
      }
    }
    dupload(L.agg, agg);
    dupload(L.cmap, cmap);
    dupload(L.frow, frow);
    dupload(L.fcol, h_col);
    dalloc(L.pb, n);
    dalloc(L.bb0, n);
    dalloc(L.bb1, n);
    dalloc(L.anrm, na);
    dalloc(L.ev0, n);
    dalloc(L.ev1, n);
    n = na;
    h_off = c_off;
    h_col = c_col;
  }
  if (levels.size() < 2) fail("the AMG needs at least two levels");
  const int nl = (int)levels.size();
  nc = levels.back().n;
  dalloc(Ad0, nc * nc);
  dalloc(Ad1, nc * nc);
  dalloc(Ainv, nc * nc);
  dalloc(wv, nl);
  dalloc(wc, nl);
  dalloc(ns, 16 * nl);
  fill(wv.Write(), nl, 0.0f);
  fill(wc.Write(), nl, 0.0f);
  if (verbose()) {
    printf("svmfem: AMG setup levels");
    for (int s : sizes()) printf(" %d", s);
    printf(" in %.2f s\n", now() - t0);
  }
}

std::vector<int> Amg::sizes() const
{
  std::vector<int> s;
  for (const auto& L : levels) s.push_back(L.n);
  return s;
}

int Amg::npow() const
{
  return nSolve == 0 ? NPOW0 : NPOW;
}

void Amg::emit_numeric(const double* V0, const float* Vs0, int nPow)
{
  nSolve++;
  const bool first = nPow == NPOW0;
  levels[0].pV = V0;
  levels[0].pVs = Vs0;
  const int nl = (int)levels.size();
  double* dns = ns.ReadWrite();
  float* dwv = wv.ReadWrite();
  float* dwc = wc.ReadWrite();
  for (int l = 0; l < nl; l++) {
    AmgLevel& L = levels[l];
    const int n = L.n;
    const int* diag = L.diag.Read();
    const double* V = L.pV;
    float* Dinv = L.Dinv.Write();
    launch(n, [=] MFEM_HOST_DEVICE (int i) {
      double d = V[diag[i]];
      Dinv[i] = (d != 0.0) ? (float)(1.0 / d) : 0.0f;
    });
    if (l + 1 == nl) break;
    const int* soff = L.sell->slice_off.Read();
    const int* cols = L.sell->cols.Read();
    const float* Vs = L.pVs;
    // Damping of the smoother and of the candidate relaxation with the
    // spectral radius of D^-1 A: power iteration x_{k+1} = D^-1 A x_k / |x_k|,
    // with |x_k|^2 accumulated in ns[s0 + k] by the kernel that produces x_k
    const int s0 = 16 * l;
    fill(dns + s0, 16, 0.0);
    float* x = L.ev0.ReadWrite();
    float* y = L.ev1.ReadWrite();
    if (first) {
      launch_red<1>(n, dns, slots(s0), [=] MFEM_HOST_DEVICE (int i, double* v) {
        unsigned h = (unsigned)i * 2654435761u;
        h = h ^ (h >> 15);
        float xi = 1.0f + (float)(h % 1000u) * 1e-3f;
        x[i] = xi;
        v[0] = (double)xi * xi;
      });
    } else {
      launch_red<1>(n, dns, slots(s0), [=] MFEM_HOST_DEVICE (int i, double* v) {
        float xi = x[i];
        v[0] = (double)xi * xi;
      });
    }
    for (int k = 0; k < nPow; k++) {
      const int slot_in = s0 + k, slot_out = s0 + k + 1;
      const float* xin = x;
      float* yout = y;
      launch_red<1>(n, dns, slots(slot_out), [=] MFEM_HOST_DEVICE (int i, double* v) {
        float inv = (float)(1.0 / sqrt(dns[slot_in]));
        float yi = Dinv[i] * sell_row_ff(i, soff, cols, Vs, xin) * inv;
        yout[i] = yi;
        v[0] = (double)yi * yi;
      });
      std::swap(x, y);
    }
    {
      const int slot = s0 + nPow;
      const double om = omega, omc = OMEGA_C;
      launch(1, [=] MFEM_HOST_DEVICE (int) {
        double rho = sqrt(dns[slot]);
        if (rho < 1e-12) rho = 1e-12;
        dwv[l] = (float)(om / rho);
        dwc[l] = (float)(omc / rho);
      });
    }
    if (x != L.ev0.ReadWrite()) copy(x, L.ev0.ReadWrite(), n);
    // Prolongator weights: relaxed candidate, normalized per aggregate
    float* b0 = L.bb0.ReadWrite();
    float* b1 = L.bb1.ReadWrite();
    fill(b0, n, 1.0f);
    for (int k = 0; k < NCAND; k++) {
      const float* bin = b0;
      float* bout = b1;
      launch(n, [=] MFEM_HOST_DEVICE (int i) {
        bout[i] = bin[i] - dwc[l] * Dinv[i] * sell_row_ff(i, soff, cols, Vs, bin);
      });
      std::swap(b0, b1);
    }
    AmgLevel& C = levels[l + 1];
    const int* agg = L.agg.Read();
    float* anrm = L.anrm.ReadWrite();
    float* pb = L.pb.Write();
    const float* b = b0;
    fill(anrm, C.n, 0.0f);
    launch(n, [=] MFEM_HOST_DEVICE (int i) { AtomicAdd(anrm[agg[i]], b[i] * b[i]); });
    launch(n, [=] MFEM_HOST_DEVICE (int i) {
      float s = anrm[agg[i]];
      pb[i] = (s > 0.0f) ? b[i] / sqrtf(s) : 0.0f;
    });
    // Galerkin coarse matrix A_c = P^T A P with P_{i,agg(i)} = pb_i: one
    // thread per fine entry, atomic additions into the coarse entry cmap[q];
    // then its float SELL copy
    double* Vc = C.V.ReadWrite();
    const int* cmap = L.cmap.Read();
    const int* frow = L.frow.Read();
    const int* fcol = L.fcol.Read();
    fill(Vc, C.nnz, 0.0);
    launch(L.nnz, [=] MFEM_HOST_DEVICE (int q) {
      AtomicAdd(Vc[cmap[q]], (double)(pb[frow[q]] * pb[fcol[q]]) * V[q]);
    });
    sell_gather_f(C.sellL, Vc, C.Vs.ReadWrite());
  }
  // Dense coarsest matrix (from its double CSR values) and its inverse by
  // Gauss-Jordan steps without pivoting (the matrix is symmetric positive
  // definite), in double precision, ping-pong between two buffers (nc
  // kernel launches)
  AmgLevel& Lc = levels.back();
  {
    const int n = nc;
    const int* off = Lc.offsets.Read();
    const int* col = Lc.columns.Read();
    const double* V = Lc.pV;
    double* a = Ad0.ReadWrite();
    double* bm = Ad1.ReadWrite();
    launch(n, [=] MFEM_HOST_DEVICE (int i) {
      for (int c = 0; c < n; c++) a[i * n + c] = 0.0;
      for (int q = off[i]; q < off[i + 1]; q++) a[i * n + col[q]] += V[q];
    });
    for (int k = 0; k < n; k++) {
      const double* A = a;
      double* B = bm;
      launch(n * n, [=] MFEM_HOST_DEVICE (int t) {
        int i = t / n, j = t - i * n;
        double p = A[k * n + k];
        if (i == k) {
          B[t] = (j == k) ? 1.0 / p : A[k * n + j] / p;
        } else if (j == k) {
          B[t] = -A[i * n + k] / p;
        } else {
          B[t] = A[t] - A[i * n + k] * A[k * n + j] / p;
        }
      });
      std::swap(a, bm);
    }
    const double* A = a;
    float* Af = Ainv.Write();
    launch(n * n, [=] MFEM_HOST_DEVICE (int t) { Af[t] = (float)A[t]; });
  }
}

const float* Amg::emit_down(const double* r, const float* zs)
{
  const int nl = (int)levels.size();
  AmgLevel& L0 = levels[0];
  AmgLevel& L1 = levels[1];
  AmgLevel& Lc = levels.back();
  const float* dwv = wv.Read();
  {
    // rc[agg[i]] += pb_i (r_i - (A zs)_i) (finest level, double r); the
    // residual accumulator of level 2 is zeroed for the next level
    const int n = L0.n;
    const int* soff = L0.sell->slice_off.Read();
    const int* cols = L0.sell->cols.Read();
    const float* Vs = L0.pVs;
    const int* agg = L0.agg.Read();
    const float* pb = L0.pb.Read();
    float* rc = L1.r.ReadWrite();
    const int nzero = nl > 2 ? levels[2].n : 0;
    float* zarr = nl > 2 ? levels[2].r.ReadWrite() : nullptr;
    launch(n, [=] MFEM_HOST_DEVICE (int i) {
      float t = (float)r[i] - sell_row_ff(i, soff, cols, Vs, zs);
      AtomicAdd(rc[agg[i]], pb[i] * t);
      if (i < nzero) zarr[i] = 0.0f;
    });
  }
  for (int l = 1; l < nl - 1; l++) {
    // z = w D^-1 r; rc[agg[i]] += pb_i (r_i - (A z)_i)
    AmgLevel& L = levels[l];
    AmgLevel& C = levels[l + 1];
    const int n = L.n;
    const int* soff = L.sell->slice_off.Read();
    const int* cols = L.sell->cols.Read();
    const float* Vs = L.pVs;
    const float* Dinv = L.Dinv.Read();
    const float* rl = L.r.Read();
    float* z = L.z.Write();
    const int* agg = L.agg.Read();
    const float* pb = L.pb.Read();
    float* rc = C.r.ReadWrite();
    const int nzero = l + 2 < nl ? levels[l + 2].n : 0;
    float* zarr = l + 2 < nl ? levels[l + 2].r.ReadWrite() : nullptr;
    launch(n, [=] MFEM_HOST_DEVICE (int i) {
      float w = dwv[l];
      float acc = sell_row_f_scaled(i, soff, cols, Vs, Dinv, rl);
      z[i] = w * Dinv[i] * rl[i];
      AtomicAdd(rc[agg[i]], pb[i] * (rl[i] - w * acc));
      if (i < nzero) zarr[i] = 0.0f;
    });
  }
  {
    // Coarsest level: z = Ainv r (one thread per row)
    const int n = nc;
    const float* Ai = Ainv.Read();
    const float* rc = Lc.r.Read();
    float* zc = Lc.z.Write();
    launch(n, [=] MFEM_HOST_DEVICE (int i) {
      float s = 0.0f;
      for (int k = 0; k < n; k++) s += Ai[i * n + k] * rc[k];
      zc[i] = s;
    });
  }
  const float* zc = Lc.z.Read();
  for (int l = nl - 2; l >= 1; l--) {
    AmgLevel& L = levels[l];
    const int n = L.n;
    const int* soff = L.sell->slice_off.Read();
    const int* cols = L.sell->cols.Read();
    const float* Vs = L.pVs;
    const float* Dinv = L.Dinv.Read();
    const float* rl = L.r.Read();
    const int* agg = L.agg.Read();
    const float* pb = L.pb.Read();
    float* z = L.z.ReadWrite();
    float* z2 = L.z2.ReadWrite();
    const float* zcl = zc;
    launch(n, [=] MFEM_HOST_DEVICE (int i) { z2[i] = z[i] + pb[i] * zcl[agg[i]]; });
    launch(n, [=] MFEM_HOST_DEVICE (int i) {
      z[i] = z2[i] + dwv[l] * Dinv[i] * (rl[i] - sell_row_ff(i, soff, cols, Vs, z2));
    });
    zc = z;
  }
  return zc;
}

}  // namespace svmfem
