// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// Aggregation AMG preconditioner for the dof x dof block matrices of
// svgpu_fsi.cu (included there, inside its anonymous namespace).
//
// Hierarchy: unsmoothed aggregation of the nodes (standard aggregation of the
// matrix graph as in PyAMG: a root with all its free neighbors, then the
// remaining nodes join a neighboring aggregate), piecewise constant
// prolongation for every dof, Galerkin coarse matrices (block sums), the
// pseudo-inverse of the coarsest matrix (at most AMG_CMAX nodes). The
// aggregation depends only on the matrix graph and is done once on the host;
// the coarse matrices are computed on the device for every solve.
//
// V-cycle: one damped block Jacobi pre- and post-smoothing step on every
// level, damping omega / rho(Dinv A) as in PyAMG (rho from a power iteration).
// It is a fixed linear operator, so it can be used as a right preconditioner
// of GMRES, and it is symmetric for a symmetric matrix (CG). The matrices and
// vectors of the hierarchy are in float: the FP64 rate of GeForce GPUs is 1/64
// of the FP32 rate. Only the input and output of the V-cycle are double.

int AMG_CMAX = 128;   // nodes of the coarsest level (at most), SVGPU_AMG_CMAX

// AMG preconditioner of the GPU FSI solvers unless SVGPU_AMG=0
bool amg_enabled()
{
  const char* e = getenv("SVGPU_AMG");
  return !(e && atoi(e) == 0);
}

struct AmgLevel {
  int n = 0, nnz = 0;
  int *rowS = nullptr, *rowE = nullptr, *col = nullptr, *diag = nullptr;
  float *V = nullptr, *Dinv = nullptr;
  // Next level: aggregate of each node, nodes of each aggregate, blocks of
  // this level summed into each block of the next level
  int* agg = nullptr;
  int *aPtr = nullptr, *aList = nullptr;
  int *gPtr = nullptr, *gList = nullptr;
  float *r = nullptr, *z = nullptr, *t = nullptr, *z2 = nullptr;   // V-cycle vectors
  float w = 0.0f;   // damping: omega / rho(Dinv A) (as in PyAMG)
  float* ev = nullptr;   // power iteration vector (warm start in the next solve)
};

struct Amg {
  int dof = 0;
  std::vector<AmgLevel> lv;
  int nc = 0;              // unknowns of the coarsest level
  double* Ad = nullptr;    // dense coarsest matrix, inverted in place
  float* Ainv = nullptr;
  double* Z = nullptr;     // V-cycle output on the finest level
  double* T = nullptr;     // GMRES: U y before the preconditioner
  float omega = 0.9f;
  int nSolve = 0;
};

// Host side standard aggregation of the graph (rs, re, cl), as in PyAMG
int amg_aggregate(int n, const std::vector<int>& rs, const std::vector<int>& re, const std::vector<int>& cl,
    std::vector<int>& agg)
{
  agg.assign(n, -1);
  int na = 0;
  // Pass 1: roots whose neighbors are all free
  for (int i = 0; i < n; i++) {
    if (agg[i] >= 0) continue;
    bool free = true;
    for (int j = rs[i]; j <= re[i] && free; j++) free = (cl[j] == i || agg[cl[j]] < 0);
    if (!free) continue;
    agg[i] = na;
    for (int j = rs[i]; j <= re[i]; j++) agg[cl[j]] = na;
    na++;
  }
  // Pass 2: join a neighboring aggregate of pass 1
  std::vector<int> a2 = agg;
  for (int i = 0; i < n; i++) {
    if (agg[i] >= 0) continue;
    for (int j = rs[i]; j <= re[i]; j++) {
      if (agg[cl[j]] >= 0) {
        a2[i] = agg[cl[j]];
        break;
      }
    }
  }
  agg = a2;
  // Pass 3: new aggregates of the remaining nodes and their free neighbors
  for (int i = 0; i < n; i++) {
    if (agg[i] >= 0) continue;
    agg[i] = na;
    for (int j = rs[i]; j <= re[i]; j++) {
      if (agg[cl[j]] < 0) agg[cl[j]] = na;
    }
    na++;
  }
  return na;
}

template <typename T> T* amg_upload(const std::vector<T>& h)
{
  return dev_upload(h.data(), h.size());
}

// Structure of the hierarchy from the finest graph (device arrays of lhs)
void amg_setup(Amg& m, int dof)
{
  m.dof = dof;
  m.lv.clear();
  const int nNo = lhs.nNo, nnz = lhs.nnz;
  std::vector<int> rs(nNo), re(nNo), cl(nnz), dg(nNo);
  CUDA_CHECK(cudaMemcpy(rs.data(), lhs.rowS, nNo * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(re.data(), lhs.rowE, nNo * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(cl.data(), lhs.col, nnz * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(dg.data(), lhs.diag, nNo * sizeof(int), cudaMemcpyDeviceToHost));
  AmgLevel L0;
  L0.n = nNo;
  L0.nnz = nnz;
  L0.rowS = lhs.rowS;
  L0.rowE = lhs.rowE;
  L0.col = lhs.col;
  L0.diag = lhs.diag;
  m.lv.push_back(L0);
  int n = nNo;
  if (getenv("SVGPU_AMG_CMAX")) AMG_CMAX = atoi(getenv("SVGPU_AMG_CMAX"));
  while (n > AMG_CMAX) {
    std::vector<int> agg;
    const int na = amg_aggregate(n, rs, re, cl, agg);
    if (na >= n) break;
    // Coarse graph: the aggregate pairs of the blocks, sorted in every row
    std::vector<std::vector<int>> rows(na);
    for (int i = 0; i < n; i++) {
      for (int j = rs[i]; j <= re[i]; j++) rows[agg[i]].push_back(agg[cl[j]]);
    }
    std::vector<int> crs(na), cre(na), ccl, cdg(na);
    for (int I = 0; I < na; I++) {
      auto& r = rows[I];
      std::sort(r.begin(), r.end());
      r.erase(std::unique(r.begin(), r.end()), r.end());
      crs[I] = (int)ccl.size();
      for (int J : r) {
        if (J == I) cdg[I] = (int)ccl.size();
        ccl.push_back(J);
      }
      cre[I] = (int)ccl.size() - 1;
    }
    // Coarse block of every fine block, then the fine blocks of every coarse block
    const int cnnz = (int)ccl.size();
    std::vector<int> gCnt(cnnz + 1, 0), fb2cb(rs.empty() ? 0 : re[n - 1] + 1);
    for (int i = 0; i < n; i++) {
      const int I = agg[i];
      for (int j = rs[i]; j <= re[i]; j++) {
        const int J = agg[cl[j]];
        const int cb = (int)(std::lower_bound(ccl.begin() + crs[I], ccl.begin() + cre[I] + 1, J) - ccl.begin());
        fb2cb[j] = cb;
        gCnt[cb + 1]++;
      }
    }
    for (int k = 0; k < cnnz; k++) gCnt[k + 1] += gCnt[k];
    std::vector<int> gList(fb2cb.size()), pos(gCnt.begin(), gCnt.end() - 1);
    for (int j = 0; j < (int)fb2cb.size(); j++) gList[pos[fb2cb[j]]++] = j;
    std::vector<int> aCnt(na + 1, 0);
    for (int i = 0; i < n; i++) aCnt[agg[i] + 1]++;
    for (int I = 0; I < na; I++) aCnt[I + 1] += aCnt[I];
    std::vector<int> aList(n), apos(aCnt.begin(), aCnt.end() - 1);
    for (int i = 0; i < n; i++) aList[apos[agg[i]]++] = i;

    AmgLevel& F = m.lv.back();
    F.agg = amg_upload(agg);
    F.aPtr = amg_upload(aCnt);
    F.aList = amg_upload(aList);
    F.gPtr = amg_upload(gCnt);
    F.gList = amg_upload(gList);
    AmgLevel C;
    C.n = na;
    C.nnz = cnnz;
    C.rowS = amg_upload(crs);
    C.rowE = amg_upload(cre);
    C.col = amg_upload(ccl);
    C.diag = amg_upload(cdg);
    m.lv.push_back(C);
    rs = crs;
    re = cre;
    cl = ccl;
    n = na;
  }
  const size_t dd = (size_t)dof * dof;
  for (size_t l = 0; l < m.lv.size(); l++) {
    AmgLevel& L = m.lv[l];
    CUDA_CHECK(cudaMalloc(&L.V, dd * L.nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&L.Dinv, dd * L.n * sizeof(float)));
    for (float** v : {&L.r, &L.z, &L.t, &L.z2, &L.ev}) CUDA_CHECK(cudaMalloc(v, (size_t)dof * L.n * sizeof(float)));
  }
  m.nc = dof * m.lv.back().n;
  CUDA_CHECK(cudaMalloc(&m.Ad, (size_t)m.nc * m.nc * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m.Ainv, (size_t)m.nc * m.nc * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&m.Z, (size_t)dof * nNo * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&m.T, (size_t)dof * nNo * sizeof(double)));
  if (getenv("SVGPU_AMG_W")) m.omega = (float)atof(getenv("SVGPU_AMG_W"));
  if (getenv("SV_GPU_VERBOSE")) {
    printf("svgpu: AMG dof %d levels", dof);
    for (auto& L : m.lv) printf(" %d", L.n);
    printf(" omega %g\n", m.omega);
  }
}

__global__ void k_amg_to_float(size_t n, const double* a, float* b)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] = (float)a[i];
}

// Inverse of the diagonal blocks. The dofs with a zero row and column in the
// block (Dirichlet dofs after the FSILS scaling) are left out (zero rows and
// columns of the result, like the pseudo-inverse); identity for a singular
// remaining block.
template <int DOF>
__global__ void k_amg_dinv(int n, const int* diag, const float* V, float* Dinv)
{
  int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= n) return;
  double m[DOF][2 * DOF];
  bool act[DOF];
  const float* d = V + DOF * DOF * (size_t)diag[a];
  for (int k = 0; k < DOF; k++) {
    act[k] = false;
    for (int q = 0; q < DOF; q++) act[k] = act[k] || d[DOF * k + q] != 0.0f || d[DOF * q + k] != 0.0f;
  }
  for (int r = 0; r < DOF; r++) {
    for (int c = 0; c < DOF; c++) {
      m[r][c] = (act[r] && act[c]) ? d[DOF * r + c] : (r == c ? 1.0 : 0.0);
      m[r][DOF + c] = (r == c) ? 1.0 : 0.0;
    }
  }
  bool ok = true;
  for (int c = 0; c < DOF && ok; c++) {
    int p = c;
    for (int r = c + 1; r < DOF; r++) if (fabs(m[r][c]) > fabs(m[p][c])) p = r;
    if (m[p][c] == 0.0) { ok = false; break; }
    if (p != c) for (int k = 0; k < 2 * DOF; k++) { double t = m[c][k]; m[c][k] = m[p][k]; m[p][k] = t; }
    const double inv = 1.0 / m[c][c];
    for (int k = 0; k < 2 * DOF; k++) m[c][k] *= inv;
    for (int r = 0; r < DOF; r++) {
      if (r == c) continue;
      const double f = m[r][c];
      if (f != 0.0) for (int k = 0; k < 2 * DOF; k++) m[r][k] -= f * m[c][k];
    }
  }
  float* o = Dinv + DOF * DOF * (size_t)a;
  for (int r = 0; r < DOF; r++) {
    for (int c = 0; c < DOF; c++) {
      o[DOF * r + c] = !(act[r] && act[c]) ? 0.0f : ok ? (float)m[r][DOF + c] : (r == c ? 1.0f : 0.0f);
    }
  }
}

// Coarse block entries: sums of the fine block entries
template <int DOF>
__global__ void k_amg_galerkin(int nnzC, const int* gPtr, const int* gList, const float* Vf, float* Vc)
{
  constexpr int DD = DOF * DOF;
  size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (size_t)DD * nnzC) return;
  const int cb = (int)(q / DD), e = (int)(q % DD);
  double s = 0.0;
  for (int k = gPtr[cb]; k < gPtr[cb + 1]; k++) s += Vf[DD * (size_t)gList[k] + e];
  Vc[q] = (float)s;
}

// Dense coarsest matrix
template <int DOF>
__global__ void k_amg_dense(int n, const int* rowS, const int* rowE, const int* col, const float* V, int nc,
    double* Ad)
{
  int I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= n) return;
  for (int c = 0; c < DOF * n; c++) {
    for (int r = 0; r < DOF; r++) Ad[(size_t)(DOF * I + r) * nc + c] = 0.0;
  }
  for (int j = rowS[I]; j <= rowE[I]; j++) {
    for (int r = 0; r < DOF; r++) {
      for (int c = 0; c < DOF; c++) Ad[(size_t)(DOF * I + r) * nc + DOF * col[j] + c] = V[DOF * DOF * (size_t)j + DOF * r + c];
    }
  }
}

// Row products y_i = sum_j K_j x_col(j) of a warp in float: lane k < DOF
// receives (y_i)_k in out. Generic: lane l < NL handles entry l % (DOF*DOF)
// of the blocks j = rowS + l / (DOF*DOF) + BPP m.
template <int DOF>
__device__ inline void warp_rowf(int i, const int* rowS, const int* rowE, const int* col, const float* K,
    const float* x, int lane, float out[DOF])
{
  constexpr int DD = DOF * DOF;
  constexpr int BPP = 32 / DD;
  constexpr int NL = BPP * DD;
  const int e = lane % DD, b0 = lane / DD, c = e % DOF;
  float v = 0.0f;
  if (lane < NL) {
    const int je = rowE[i];
    for (int j = rowS[i] + b0; j <= je; j += BPP) v += K[DD * (size_t)j + e] * x[DOF * (size_t)col[j] + c];
  }
  for (int k = 1; k < BPP; k++) {
    float t = __shfl_down_sync(0xffffffff, v, k * DD);
    if (b0 == 0) v += t;
  }
  float r = v;
  for (int k = 1; k < DOF; k++) {
    float t = __shfl_down_sync(0xffffffff, v, k);
    if (c == 0) r += t;
  }
  for (int k = 0; k < DOF; k++) out[k] = __shfl_sync(0xffffffff, r, k * DOF);
}

// 4 x 4 blocks: lane l handles row l % 4 of the blocks j = rowS + l / 4 + 8 m
// with float4 loads
template <>
__device__ inline void warp_rowf<4>(int i, const int* rowS, const int* rowE, const int* col, const float* K,
    const float* x, int lane, float out[4])
{
  const int b = lane / 4, r = lane % 4;
  const int je = rowE[i];
  float s = 0.0f;
  for (int j = rowS[i] + b; j <= je; j += 8) {
    const float4 k0 = reinterpret_cast<const float4*>(K + 16 * (size_t)j)[r];
    const float4 x0 = *reinterpret_cast<const float4*>(x + 4 * (size_t)col[j]);
    s += k0.x * x0.x + k0.y * x0.y + k0.z * x0.z + k0.w * x0.w;
  }
  for (int o = 4; o < 32; o <<= 1) s += __shfl_xor_sync(0xffffffff, s, o);
  for (int k = 0; k < 4; k++) out[k] = __shfl_sync(0xffffffff, s, k);
}

__device__ inline bool amg_done(const int* done)
{
  return done && __ldcg(done);
}

// z = w Dinv r (r: double on the finest level)
template <int DOF, typename TR>
__global__ void k_amg_smooth0(int n, const float* Dinv, float w, const TR* r, float* z, const int* done)
{
  if (amg_done(done)) return;
  int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= n) return;
  const float* d = Dinv + DOF * DOF * (size_t)a;
  float rv[DOF];
  for (int k = 0; k < DOF; k++) rv[k] = (float)r[DOF * (size_t)a + k];
  for (int i = 0; i < DOF; i++) {
    float s = 0.0f;
    for (int k = 0; k < DOF; k++) s += d[DOF * i + k] * rv[k];
    z[DOF * (size_t)a + i] = w * s;
  }
}

// t = r - A z (warp per row)
template <int DOF, typename TR>
__global__ void k_amg_res(int n, const int* rowS, const int* rowE, const int* col, const float* V, const TR* r,
    const float* z, float* t, const int* done)
{
  if (amg_done(done)) return;
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (i >= n) return;
  float y[DOF];
  warp_rowf<DOF>(i, rowS, rowE, col, V, z, lane, y);
#pragma unroll
  for (int k = 0; k < DOF; k++) {
    if (lane == k) t[DOF * (size_t)i + k] = (float)r[DOF * (size_t)i + k] - y[k];
  }
}

// rc(I) = sum of t over the nodes of aggregate I
template <int DOF>
__global__ void k_amg_restrict(int nA, const int* aPtr, const int* aList, const float* t, float* rc, const int* done)
{
  if (amg_done(done)) return;
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= DOF * nA) return;
  const int I = q / DOF, k = q % DOF;
  float s = 0.0f;
  for (int p = aPtr[I]; p < aPtr[I + 1]; p++) s += t[DOF * (size_t)aList[p] + k];
  rc[q] = s;
}

// z2 = z + P ec
template <int DOF>
__global__ void k_amg_prolong(int n, const int* agg, const float* z, const float* ec, float* z2, const int* done)
{
  if (amg_done(done)) return;
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= DOF * n) return;
  z2[q] = z[q] + ec[DOF * (size_t)agg[q / DOF] + q % DOF];
}

// z = z2 + w Dinv (r - A z2) (warp per row; z: double on the finest level)
template <int DOF, typename TR, typename TZ>
__global__ void k_amg_post(int n, const int* rowS, const int* rowE, const int* col, const float* V,
    const float* Dinv, float w, const TR* r, const float* z2, TZ* z, const int* done)
{
  if (amg_done(done)) return;
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (i >= n) return;
  float y[DOF];
  warp_rowf<DOF>(i, rowS, rowE, col, V, z2, lane, y);
  if (lane < DOF) {
    const float* d = Dinv + DOF * DOF * (size_t)i;
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < DOF; k++) s += d[DOF * lane + k] * ((float)r[DOF * (size_t)i + k] - y[k]);
    TZ zv = (TZ)(z2[DOF * (size_t)i + lane] + w * s);
    if (sizeof(TZ) == sizeof(double)) {
      // Finest level: no component in the dofs without equation (zero row
      // of the block, e.g. Dirichlet dofs), like the unpreconditioned solver
      bool act = false;
#pragma unroll
      for (int k = 0; k < DOF; k++) act = act || d[DOF * lane + k] != 0.0f;
      if (!act) zv = 0;
    }
    z[DOF * (size_t)i + lane] = zv;
  }
}

// z = Ainv r (warp per row)
__global__ void k_amg_coarse(int nc, const float* Ainv, const float* r, float* z, const int* done)
{
  if (amg_done(done)) return;
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (i >= nc) return;
  float s = 0.0f;
  for (int k = lane; k < nc; k += 32) s += Ainv[(size_t)i * nc + k] * r[k];
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
  if (lane == 0) z[i] = s;
}

// y = Dinv A x (warp per row)
template <int DOF>
__global__ void k_amg_dax(int n, const int* rowS, const int* rowE, const int* col, const float* V, const float* Dinv,
    const float* x, float* y)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (i >= n) return;
  float ax[DOF];
  warp_rowf<DOF>(i, rowS, rowE, col, V, x, lane, ax);
  if (lane < DOF) {
    const float* d = Dinv + DOF * DOF * (size_t)i;
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < DOF; k++) s += d[DOF * lane + k] * ax[k];
    y[DOF * (size_t)i + lane] = s;
  }
}

__global__ void k_amg_init_vec(size_t n, float* x)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned h = (unsigned)i * 2654435761u;
  h ^= h >> 15;
  x[i] = 1.0f + (h % 1000) * 1e-3f;
}

// x = a x; acc += |x|^2 (after the scaling)
__global__ void k_amg_scale_norm(size_t n, float a, float* x, double* acc)
{
  double s = 0.0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float v = a * x[i];
    x[i] = v;
    s += (double)v * v;
  }
  s = block_sum(s);
  if (threadIdx.x == 0) atomicAdd(acc, s);
}

inline int nb_rows(int n) { return (int)(((size_t)n * 32 + TPB - 1) / TPB); }
inline int nb_n(size_t n) { return (int)((n + TPB - 1) / TPB); }

double amg_norm2(float* x, size_t n, float a)
{
  CUDA_CHECK(cudaMemsetAsync(lhs.acc, 0, sizeof(double)));
  k_amg_scale_norm<<<160, TPB>>>(n, a, x, lhs.acc);
  double r;
  CUDA_CHECK(cudaMemcpy(&r, lhs.acc, sizeof(double), cudaMemcpyDeviceToHost));
  return r;
}

// Coarse matrices and smoother of the current device matrix
template <int DOF>
void amg_numeric(Amg& m)
{
  const size_t dd = DOF * DOF;
  static const bool tim = getenv("SVGPU_TIMING") != nullptr;
  double tt[4] = {};
  auto tick = [&](int k) { if (tim) { CUDA_CHECK(cudaDeviceSynchronize()); tt[k] = now(); } };
  tick(0);
  AmgLevel& L0 = m.lv[0];
  k_amg_to_float<<<nb_n(dd * L0.nnz), TPB>>>(dd * L0.nnz, svgpu_dev_val_ptr, L0.V);
  for (size_t l = 0; l < m.lv.size(); l++) {
    AmgLevel& L = m.lv[l];
    k_amg_dinv<DOF><<<nb_n(L.n), TPB>>>(L.n, L.diag, L.V, L.Dinv);
    if (l + 1 < m.lv.size()) {
      AmgLevel& C = m.lv[l + 1];
      k_amg_galerkin<DOF><<<nb_n(dd * C.nnz), TPB>>>(C.nnz, L.gPtr, L.gList, L.V, C.V);
    }
  }
  tick(1);
  // Damping of the smoother with the spectral radius of Dinv A (power iteration)
  // (15 iterations in the first solve, then 4 from the previous vector)
  static const int nPow0 = getenv("SVGPU_AMG_POW") ? atoi(getenv("SVGPU_AMG_POW")) : 15;
  const int nPow = m.nSolve++ == 0 ? nPow0 : std::min(nPow0, 4);
  for (size_t l = 0; l + 1 < m.lv.size(); l++) {
    AmgLevel& L = m.lv[l];
    const size_t nv = (size_t)DOF * L.n;
    float* x = L.ev;
    float* y = L.z2;
    if (nPow == nPow0) k_amg_init_vec<<<nb_n(nv), TPB>>>(nv, x);
    double nx = std::sqrt(amg_norm2(x, nv, 1.0f)), rho = 1.0;
    for (int k = 0; k < nPow; k++) {
      k_amg_dax<DOF><<<nb_rows(L.n), TPB>>>(L.n, L.rowS, L.rowE, L.col, L.V, L.Dinv, x, y);
      const double ny = std::sqrt(amg_norm2(y, nv, 1.0f));
      rho = ny / nx;
      if (ny == 0.0) break;
      amg_norm2(y, nv, (float)(1.0 / ny));
      std::swap(x, y);
      nx = 1.0;
    }
    if (x != L.ev) CUDA_CHECK(cudaMemcpyAsync(L.ev, x, nv * sizeof(float), cudaMemcpyDeviceToDevice));
    L.w = (float)(m.omega / std::max(rho, 1e-12));
  }
  static int nPrint = 0;
  if (getenv("SV_GPU_VERBOSE") && nPrint++ < 2) {
    printf("svgpu: AMG damping");
    for (size_t l = 0; l + 1 < m.lv.size(); l++) printf(" %g", m.lv[l].w);
    printf("\n");
  }
  tick(2);
  AmgLevel& Lc = m.lv.back();
  k_amg_dense<DOF><<<nb_n(Lc.n), TPB>>>(Lc.n, Lc.rowS, Lc.rowE, Lc.col, Lc.V, m.nc, m.Ad);
  CUDA_CHECK(cudaGetLastError());
  // Pseudo-inverse of the small coarsest matrix on the host (zero rows and
  // columns of the Dirichlet dofs)
  static std::vector<double> hA;
  static std::vector<float> hI;
  hA.resize((size_t)m.nc * m.nc);
  hI.resize(hA.size());
  CUDA_CHECK(cudaMemcpy(hA.data(), m.Ad, hA.size() * sizeof(double), cudaMemcpyDeviceToHost));
  using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<Mat> Am(hA.data(), m.nc, m.nc);
  Mat Ai = Am.completeOrthogonalDecomposition().pseudoInverse();
  for (size_t q = 0; q < hI.size(); q++) hI[q] = (float)Ai.data()[q];
  CUDA_CHECK(cudaMemcpy(m.Ainv, hI.data(), hI.size() * sizeof(float), cudaMemcpyHostToDevice));
  tick(3);
  if (tim) printf("SVGPU-TIMING amg numeric: galerkin %.2f ms, rho %.2f ms, coarse %.2f ms\n",
      1e3 * (tt[1] - tt[0]), 1e3 * (tt[2] - tt[1]), 1e3 * (tt[3] - tt[2]));
}

// Kernel timing of one V-cycle (SVGPU_AMG_PROF)
struct AmgProf {
  bool on = false;
  cudaEvent_t e0;
  void start() { if (on) { cudaEventCreate(&e0); cudaEventRecord(e0); } }
  void mark(const char* what, size_t l) {
    if (!on) return;
    cudaEvent_t e1;
    cudaEventCreate(&e1);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    printf("SVGPU-AMG-PROF level %d %s %.3f ms\n", (int)l, what, ms);
    e0 = e1;
  }
};
AmgProf amgProf;

// One level of the V-cycle: z = M r (TR, TZ: double on the finest level)
template <int DOF, typename TR, typename TZ>
void amg_level(Amg& m, size_t l, const TR* r, TZ* z, const int* done);

template <int DOF>
void amg_coarse_or_level(Amg& m, size_t l, const float* r, float* z, const int* done)
{
  if (l + 1 == m.lv.size()) {
    k_amg_coarse<<<nb_rows(m.nc), TPB>>>(m.nc, m.Ainv, r, z, done);
    amgProf.mark("coarse", l);
  } else {
    amg_level<DOF, float, float>(m, l, r, z, done);
  }
}

template <int DOF, typename TR, typename TZ>
void amg_level(Amg& m, size_t l, const TR* r, TZ* z, const int* done)
{
  AmgLevel& L = m.lv[l];
  AmgLevel& C = m.lv[l + 1];
  float* zs = L.z;   // pre-smoothed z
  k_amg_smooth0<DOF, TR><<<nb_n(L.n), TPB>>>(L.n, L.Dinv, L.w, r, zs, done);
  amgProf.mark("smooth0", l);
  k_amg_res<DOF, TR><<<nb_rows(L.n), TPB>>>(L.n, L.rowS, L.rowE, L.col, L.V, r, zs, L.t, done);
  amgProf.mark("res", l);
  k_amg_restrict<DOF><<<nb_n((size_t)DOF * C.n), TPB>>>(C.n, L.aPtr, L.aList, L.t, C.r, done);
  amgProf.mark("restrict", l);
  amg_coarse_or_level<DOF>(m, l + 1, C.r, C.z, done);
  k_amg_prolong<DOF><<<nb_n((size_t)DOF * L.n), TPB>>>(L.n, L.agg, zs, C.z, L.z2, done);
  amgProf.mark("prolong", l);
  k_amg_post<DOF, TR, TZ><<<nb_rows(L.n), TPB>>>(L.n, L.rowS, L.rowE, L.col, L.V, L.Dinv, L.w, r, L.z2, z, done);
  amgProf.mark("post", l);
}

// z = M r on the finest level
template <int DOF>
void amg_vcycle(Amg& m, size_t l, const double* r, double* z, const int* done)
{
  amgProf.start();
  amg_level<DOF, double, double>(m, 0, r, z, done);
}

template <int DOF> Amg& amg_get()
{
  static Amg m;
  if (m.lv.empty() || m.lv[0].n != lhs.nNo || m.lv[0].nnz != lhs.nnz) amg_setup(m, DOF);
  return m;
}

__global__ void k_add_vec(size_t n, const double* a, double* b)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] += a[i];
}
