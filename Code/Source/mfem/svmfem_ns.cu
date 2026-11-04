// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// The FSILS NS linear solver (linear_solver/ns_solver.cpp, gmres.cpp,
// cgrad.cpp) on the device resident matrix, with mfem::forall kernels. The
// algorithm of the bi-partitioned solver is the same as on the CPU: GMRES on
// the momentum block (with the resistance terms of the coupled faces) and CG
// on the Schur complement L + G^T G. The Schur complement is formed
// explicitly (its pattern with mfem::SparseMatrix) and the CG is
// preconditioned with an aggregation AMG V-cycle (svmfem_amg.cu); the
// convergence criteria are those of the CPU solver.
//
// Every phase of the solver (setup, the Arnoldi steps, chunks of CG
// iterations, the outer iteration) is a sequence of kernels that can be
// captured in a CUDA graph; the scalars that the host needs (norms,
// Hessenberg columns, Gram matrices) are written by the last kernel of a
// phase into mapped pinned memory and read after one stream
// synchronization, and the coefficients computed on the host are read by a
// kernel from mapped memory. The host code only steers the iterations.

#include "svmfem.h"
#include "svmfem_amg.hpp"
#include "svmfem_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>

namespace svmfem {

/// Least squares solution x of the n x n system A x = b (column-major A),
/// with Eigen's complete orthogonal decomposition (svmfem_lstsq.cpp)
void lstsq_solve(int n, const double* A, const double* b, double* x);

// Named namespace: nvcc on Windows requires external linkage for the
// functions that define the device lambdas
namespace ns {

constexpr int MAXF = 4;        // maximum number of coupled faces

// Slots of the solver scalars sc (double, device)
constexpr int RZ = 0;          // two slots (parity): r.z of the current and the previous iteration
constexpr int PQ = 2, RR = 3, RES_M = 4, RES_C = 5, EPS_M = 6, EPS_C = 7;
constexpr int ACC = 8;         // MAXF face accumulators
constexpr int GRAM = 16;       // Gram matrix dots (8 iB + 12 slots)
// Slots of the GMRES scalars gs: norms, face dots (two parities), Hessenberg columns
constexpr int NRM = 0;
constexpr int VW = 2;
constexpr int KW = VW + 2 * MAXF;
constexpr int GH = KW + 2 * MAXF;

struct SubLs {
  int mItr, sD;
  double absTol, relTol;
  int itr = 0;
  bool success = false;
  double iNorm = 0, fNorm = 0, dB = 0;
};

// The products of the block matrices (K 3x3 with 9 values per entry,
// row-major, G 3x1, D 1x3, L 1x1) and of the float Schur complement use the
// SELL layout (sell_row_* in svmfem_common.hpp).

/// Coefficients s_f of the face vectors in w = K u + ADD + PRE (add_bc_mul
/// ADD then PRE, the PRE terms face by face as on the CPU), from the dots
/// v_f . u (gs[VW + bp MAXF + f]) and v_f . K u (gs[KW + ...]) scaled by inv
MFEM_HOST_DEVICE inline void face_coefs(int nF, const double* gs, int bp, double inv, const double* fd, int cadd_off,
    int cpre_off, int vv_off, double s[MAXF])
{
  double t[MAXF];
  for (int f = 0; f < nF; f++) {
    double d = gs[VW + bp * MAXF + f] * inv;      // v_f . u
    s[f] = fd[cadd_off + f] * d;                  // ADD: res_f v_f (v_f . u)
  }
  for (int f = 0; f < nF; f++) {
    double e = gs[KW + bp * MAXF + f] * inv;      // v_f . K u
    for (int g = 0; g < nF; g++) e = e + s[g] * fd[vv_off + f * MAXF + g];
    for (int g = 0; g < f; g++) e = e + fd[cpre_off + g] * t[g] * fd[vv_off + f * MAXF + g];
    t[f] = e;                                     // v_f . w before the PRE term of face f
  }
  for (int f = 0; f < nF; f++) s[f] = s[f] + fd[cpre_off + f] * t[f];
}

double env_double(const char* name, double def)
{
  const char* v = getenv(name);
  return v ? atof(v) : def;
}

int env_int(const char* name, int def)
{
  const char* v = getenv(name);
  return v ? atoi(v) : def;
}

// ---------------------------------------------------------------------------

class NsSolver {
 public:
  explicit NsSolver(Lhs& lhs);
  void solve(double* R_host, int nFaces, const svmfem_face* faces, const svmfem_params& p, svmfem_stats* stats);

 public:   // (nvcc requires public access for the functions that define extended lambdas)
  Lhs& lhs;
  int nNo, n3, nnz;
  // Sub-blocks of the matrix (CSR) and their SELL copies for the products
  Vector mK, mG, mD, mL;
  Sell sellA;
  Vector mKs, mGs, mDs, mLs;
  // Vectors
  Vector R, Rm, Rmi, gR, gX, zero3, wb0, wb1, Rc, Rci;
  // Schur complement S = L + G^T G: pattern (sorted CSR), position of each
  // entry of L in S, values (double) and their float SELL copy
  int nnzS = 0;
  Array<int> S_off, S_col, Lmap;
  Vector Sval;
  Sell sellS;
  Array<float> Ss;
  std::unique_ptr<Amg> amg;
  // CG arrays
  Vector cg_X, cg_r, cg_z, cg_p, cg_q;
  Array<float> cg_zs, cg_z2;
  int cg_chunk;
  // Work arrays of the outer iteration and GMRES
  int iB = 0, sD = 0, nsc = 0, coef_seg = 0, ncoef = 0, ngs = 0;
  Vector sc, dcoef, U, MU, P, MP, gu, gs;
  std::unique_ptr<Mapped> hsc, hcoef, hgs;
  // Coupled faces
  int nF = 0, face_total = 0;
  std::vector<int> face_sizes;
  Vector Vf, kv, g, fd;
  Array<int> face_off, face_glob;
  int cadd_off = 0, cpre_off = 0, vv_off = 0;
  std::unique_ptr<Mapped> hfd;
  GraphCache graphs;
  // Timing
  double t_gm = 0, t_cg = 0, t_all = 0, t_setup = 0;
  int n_call = 0, n_gm = 0, n_cg = 0;

  void alloc_work(int iB, int sD);
  void set_faces(int nFaces, const svmfem_face* faces);
  void publish(int lo = 0, int hi = 16);
  void pub(const Vector& src, Mapped& dst, int lo, int hi);
  void fetch_coef();
  // Kernel sequences of the phases
  void emit_start();
  void emit_setup(int nPow);
  void emit_cg_init(const double* src);
  void emit_cg_chunk();
  void emit_gm_init();
  void emit_gm_init2();
  void emit_gm_restart();
  void emit_gm_step(int i);
  void emit_gm_fin();
  void emit_A(int i);
  void emit_B(int i);
  void emit_C(int i);
  void emit_D(int i);
  void emit_E();
  // Drivers
  void gmres(SubLs& ls);
  int cg_schur(SubLs& ls, double err);
  template <typename F> void run(const std::string& key, F&& emit)
  {
    graphs.run(key, std::function<void()>(emit));
  }
};

NsSolver::NsSolver(Lhs& lhs_) : lhs(lhs_)
{
  trace("NsSolver setup");
  double t0 = now();
  nNo = lhs.nNo;
  nnz = lhs.nnz;
  n3 = 3 * nNo;
  dalloc(mK, 9 * nnz);
  dalloc(mG, 3 * nnz);
  dalloc(mD, 3 * nnz);
  dalloc(mL, nnz);
  sellA.build(nNo, lhs.h_off, lhs.h_col);
  dalloc(mKs, 9 * sellA.total);
  dalloc(mGs, 3 * sellA.total);
  dalloc(mDs, 3 * sellA.total);
  dalloc(mLs, sellA.total);
  for (Vector* v : {&R}) dalloc(*v, 4 * nNo);
  for (Vector* v : {&Rm, &Rmi, &gR, &gX, &zero3, &wb0, &wb1}) dalloc(*v, n3);
  for (Vector* v : {&Rc, &Rci, &cg_X, &cg_r, &cg_z, &cg_p, &cg_q}) dalloc(*v, nNo);
  fill(zero3.Write(), n3, 0.0);
  dalloc(cg_zs, nNo);
  dalloc(cg_z2, nNo);
  // Pattern of the Schur complement S = L + G^T G (scalar CSR, sorted
  // columns), with mfem::SparseMatrix: the pattern of A^T A
  {
    std::vector<double> ones(nnz, 1.0);
    mfem::SparseMatrix A(lhs.h_off.data(), lhs.h_col.data(), ones.data(), nNo, nNo, false, false, true);
    std::unique_ptr<mfem::SparseMatrix> At(mfem::Transpose(A));
    std::unique_ptr<mfem::SparseMatrix> S(mfem::Mult(*At, A));
    trace("Schur pattern");
    S->SortColumnIndices();   // (on the device when a GPU is configured)
    nnzS = S->NumNonZeroElems();
    const int* si = S->HostReadI();
    const int* sj = S->HostReadJ();
    dupload(S_off, si, nNo + 1);
    dupload(S_col, sj, nnzS);
    // Position of every entry of the pattern of A in S
    std::vector<int> lmap(nnz);
    for (int i = 0; i < nNo; i++) {
      for (int q = lhs.h_off[i]; q < lhs.h_off[i + 1]; q++) {
        int c = lhs.h_col[q];
        const int* p = std::lower_bound(sj + si[i], sj + si[i + 1], c);
        if (p == sj + si[i + 1] || *p != c) fail("the pattern of L is not contained in the pattern of G^T G");
        lmap[q] = (int)(p - sj);
      }
    }
    dupload(Lmap, lmap);
    dalloc(Sval, nnzS);
    std::vector<int> h_soff(si, si + nNo + 1), h_scol(sj, sj + nnzS);
    sellS.build(nNo, h_soff, h_scol);
    dalloc(Ss, sellS.total);
    trace("AMG setup");
    amg.reset(new Amg(nNo, h_soff, h_scol, &sellS, 4.0 / 3.0));
  }
  // CG iterations per graph launch (one synchronization per launch; up to
  // cg_chunk - 1 iterations after the convergence are computed)
  cg_chunk = env_int("SVMFEM_CG_CHUNK", std::max(2, std::min(8, 400000 / nNo / 2 * 2)));
  set_faces(0, nullptr);
  if (verbose()) {
    printf("svmfem: NS solver nNo %d nnz %d nnz(S) %d AMG levels", nNo, nnz, nnzS);
    for (int s : amg->sizes()) printf(" %d", s);
    printf(" SELL padding %.2f %.2f CG chunk %d, setup %.2f s\n", sellA.padding, sellS.padding, cg_chunk,
        now() - t0);
  }
}

void NsSolver::alloc_work(int iB_, int sD_)
{
  if (iB >= iB_ && sD >= sD_) return;
  iB = iB_;
  sD = sD_;
  ngs = pad8(GH + 2 * (sD + 1));
  nsc = pad8(GRAM + 8 * iB + 12);
  dalloc(sc, nsc);
  fill(sc.Write(), nsc, 0.0);
  hsc.reset(new Mapped(nsc));
  coef_seg = std::max(sD, 2 * iB) + 2;
  ncoef = pad8(3 * coef_seg);
  hcoef.reset(new Mapped(ncoef));
  dalloc(dcoef, ncoef);
  dalloc(U, (size_t)iB * n3);
  dalloc(MU, (size_t)2 * iB * n3);
  dalloc(P, (size_t)iB * nNo);
  dalloc(MP, (size_t)2 * iB * nNo);
  dalloc(gu, (size_t)(sD + 1) * n3);
  dalloc(gs, ngs);
  fill(gs.Write(), ngs, 0.0);
  hgs.reset(new Mapped(ngs));
  graphs.clear();
}

/// Faces of this solve: node indices, face vectors, resistances. The face
/// vectors, G^T v and K^T v are computed by the setup graph from the staged
/// data in hfd.
void NsSolver::set_faces(int nFaces, const svmfem_face* faces)
{
  if (nFaces > MAXF) fail("more than " + std::to_string(MAXF) + " coupled faces");
  std::vector<int> sizes(nFaces);
  int total = 0;
  for (int f = 0; f < nFaces; f++) {
    sizes[f] = faces[f].nNo;
    total += faces[f].nNo;
  }
  if (nFaces != nF || sizes != face_sizes || !hfd) {
    // New face structure: new arrays, the captured graphs are invalid
    nF = nFaces;
    face_sizes = sizes;
    face_total = total;
    const int na = std::max(nF, 1);
    dalloc(Vf, (size_t)na * n3);
    dalloc(kv, (size_t)na * n3);
    dalloc(g, (size_t)na * nNo);
    std::vector<int> off(nF + 1, 0), glob;
    for (int f = 0; f < nF; f++) {
      off[f + 1] = off[f] + faces[f].nNo;
      glob.insert(glob.end(), faces[f].glob, faces[f].glob + faces[f].nNo);
    }
    dupload(face_off, off);
    if (glob.empty()) glob.push_back(0);
    dupload(face_glob, glob);
    cadd_off = 3 * total;
    cpre_off = cadd_off + MAXF;
    vv_off = cpre_off + MAXF;
    const int nfd = vv_off + MAXF * MAXF;
    hfd.reset(new Mapped(nfd));
    dalloc(fd, hfd->n);
    fill(fd.Write(), hfd->n, 0.0);
    graphs.clear();
  }
  if (nF == 0) return;
  double* h = hfd->host;
  std::vector<double> dense((size_t)nF * n3, 0.0);
  int a0 = 0;
  for (int f = 0; f < nF; f++) {
    const svmfem_face& fa = faces[f];
    for (int a = 0; a < fa.nNo; a++) {
      for (int k = 0; k < 3; k++) {
        double v = fa.valM[3 * a + k];
        h[3 * (a0 + a) + k] = v;
        dense[(size_t)f * n3 + 3 * fa.glob[a] + k] += v;
      }
    }
    a0 += fa.nNo;
    h[cadd_off + f] = fa.res;
  }
  for (int f = 0; f < nF; f++) {
    for (int q = 0; q < nF; q++) {
      double s = 0.0;
      for (int k = 0; k < n3; k++) s += dense[(size_t)f * n3 + k] * dense[(size_t)q * n3 + k];
      h[vv_off + f * MAXF + q] = s;
    }
    h[cpre_off + f] = -faces[f].res / (1.0 + faces[f].res * h[vv_off + f * MAXF + f]);
  }
}

void NsSolver::pub(const Vector& src, Mapped& dst, int lo, int hi)
{
  const int v0 = lo / 8;
  publish8(src.Read(), dst.dev, v0, (hi - 1) / 8 - v0 + 1);
}

void NsSolver::publish(int lo, int hi)
{
  pub(sc, *hsc, lo, hi);
}

void NsSolver::fetch_coef()
{
  fetch8(hcoef->dev, dcoef.Write(), hcoef->nvec);
}

// ---------------------------------------------------------------------------
// Kernel sequences of the phases

void NsSolver::emit_start()
{
  const double* dR = R.Read();
  double* dRm = Rm.Write();
  double* dRc = Rc.Write();
  double* dRmi = Rmi.Write();
  double* dRci = Rci.Write();
  double* dsc = sc.ReadWrite();
  const int n = nNo;
  launch(n, [=] MFEM_HOST_DEVICE (int a) {
    for (int k = 0; k < 3; k++) {
      double v = dR[4 * a + k];
      dRm[3 * a + k] = v;
      dRmi[3 * a + k] = v;
    }
    dRc[a] = dR[4 * a + 3];
    dRci[a] = dR[4 * a + 3];
  });
  fill(dsc, nsc, 0.0);
  dot(n3, dRm, dRm, dsc, EPS_M);
  dot(nNo, dRc, dRc, dsc, EPS_C);
  publish();
}

void NsSolver::emit_setup(int nPow)
{
  const int* off = lhs.offsets.Read();
  const int* col = lhs.columns.Read();
  const int* rows = lhs.rows.Read();
  const int* tpos = lhs.tpos.Read();
  const double* Val = lhs.Val.Read();
  double* K = mK.Write();
  double* G = mG.Write();
  double* D = mD.Write();
  double* L = mL.Write();
  // Split Val into the blocks of the NS system (ns_solver.cpp depart)
  launch(nnz, [=] MFEM_HOST_DEVICE (int j) {
    const double* t = Val + 16 * (size_t)j;
    double* k = K + 9 * (size_t)j;
    k[0] = t[0]; k[1] = t[1]; k[2] = t[2];
    k[3] = t[4]; k[4] = t[5]; k[5] = t[6];
    k[6] = t[8]; k[7] = t[9]; k[8] = t[10];
    G[3 * j] = t[3]; G[3 * j + 1] = t[7]; G[3 * j + 2] = t[11];
    D[3 * j] = t[12]; D[3 * j + 1] = t[13]; D[3 * j + 2] = t[14];
    L[j] = t[15];
  });
  sell_gather(sellA, K, 9, mKs.Write());
  sell_gather(sellA, G, 3, mGs.Write());
  sell_gather(sellA, D, 3, mDs.Write());
  sell_gather(sellA, L, 1, mLs.Write());
  if (nF > 0) {
    // Dense face vectors v_f, g_f = G^T v_f (Schur complement) and
    // kv_f = K^T v_f (GMRES) from the staged face data
    double* dVf = Vf.Write();
    double* dkv = kv.Write();
    double* dg = g.Write();
    double* dfd = fd.Write();
    fill(dVf, nF * n3, 0.0);
    fill(dkv, nF * n3, 0.0);
    fill(dg, nF * nNo, 0.0);
    fetch8(hfd->dev, dfd, hfd->nvec);
    const int* foff = face_off.Read();
    const int* fglob = face_glob.Read();
    const int nf = nF, nn3 = n3, nn = nNo;
    launch(face_total, [=] MFEM_HOST_DEVICE (int a) {
      int f = 0;
      for (int q = 0; q < nf; q++) {
        if (a >= foff[q]) f = q;
      }
      const int Ac = fglob[a];
      const double v0 = dfd[3 * a], v1 = dfd[3 * a + 1], v2 = dfd[3 * a + 2];
      for (int k = 0; k < 3; k++) dVf[(size_t)f * nn3 + 3 * Ac + k] = dfd[3 * a + k];
      for (int j = off[Ac]; j < off[Ac + 1]; j++) {
        const int c = col[j];
        // g[f] += G^T v_f, kv[f] += K^T v_f (v_f nonzero on the face nodes only)
        AtomicAdd(dg[(size_t)f * nn + c], G[3 * j] * v0 + G[3 * j + 1] * v1 + G[3 * j + 2] * v2);
        const double* k = K + 9 * (size_t)j;
        AtomicAdd(dkv[(size_t)f * nn3 + 3 * c], k[0] * v0 + k[3] * v1 + k[6] * v2);
        AtomicAdd(dkv[(size_t)f * nn3 + 3 * c + 1], k[1] * v0 + k[4] * v1 + k[7] * v2);
        AtomicAdd(dkv[(size_t)f * nn3 + 3 * c + 2], k[2] * v0 + k[5] * v1 + k[8] * v2);
      }
    });
  }
  // S = L + G^T G: the entry (i, k) of the pattern adds G_ki G_kj to S_ij for
  // the entries (k, j) of row k (binary search of j in row i of S)
  double* S = Sval.Write();
  float* dSs = Ss.Write();
  const int* lmap = Lmap.Read();
  const int* soff = S_off.Read();
  const int* scol = S_col.Read();
  fill(S, nnzS, 0.0);
  launch(nnz, [=] MFEM_HOST_DEVICE (int q) { S[lmap[q]] = L[q]; });
  launch(nnz, [=] MFEM_HOST_DEVICE (int q) {
    const int i = rows[q];
    const int k = col[q];
    const double* gki = G + 3 * (size_t)tpos[q];
    const int s0 = soff[i];
    const int s1 = soff[i + 1] - 1;
    for (int q2 = off[k]; q2 < off[k + 1]; q2++) {
      const int j = col[q2];
      const double v = gki[0] * G[3 * q2] + gki[1] * G[3 * q2 + 1] + gki[2] * G[3 * q2 + 2];
      int lo = s0, hi = s1;
      while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (scol[mid] < j) lo = mid + 1; else hi = mid;
      }
      AtomicAdd(S[lo], v);
    }
  });
  sell_gather_f(sellS, S, dSs);
  amg->emit_numeric(S, dSs, nPow);
  copy(Rm.Read(), gR.Write(), n3);
}

/// r = src, X = 0, |r|^2 in sc[RR], first V-cycle z = M r, r.z in sc[RZ]
void NsSolver::emit_cg_init(const double* src)
{
  AmgLevel& L0 = amg->level(0);
  AmgLevel& L1 = amg->level(1);
  double* dsc = sc.ReadWrite();
  double* r = cg_r.Write();
  double* X = cg_X.Write();
  double* p = cg_p.Write();
  float* zs = cg_zs.Write();
  const float* Dinv = L0.Dinv.Read();
  const float* wv = amg->damping();
  const int nc1 = L1.n;
  float* rc1 = L1.r.Write();
  const int n = nNo;
  fill(dsc, 4, 0.0);
  launch_red<1>(n, dsc, slots(RR), [=] MFEM_HOST_DEVICE (int i, double* v) {
    // r = src, X = p = 0, sc[RR] += |r|^2, zs = w D^-1 r (pre-smoothing)
    double ri = src[i];
    r[i] = ri;
    X[i] = 0.0;
    p[i] = 0.0;
    zs[i] = wv[0] * Dinv[i] * (float)ri;
    v[0] = ri * ri;
    if (i == 0) {
      dsc[RZ + 1] = 1.0;
      dsc[PQ] = 0.0;
    }
    if (i < MAXF) dsc[ACC + i] = 0.0;
    if (i < nc1) rc1[i] = 0.0f;
  });
  const float* zc = amg->emit_down(r, zs);
  const int* agg = L0.agg.Read();
  const float* pb = L0.pb.Read();
  float* z2 = cg_z2.Write();
  launch(n, [=] MFEM_HOST_DEVICE (int i) { z2[i] = zs[i] + pb[i] * zc[agg[i]]; });
  const int* soff = sellS.slice_off.Read();
  const int* scol = sellS.cols.Read();
  const float* dSs = Ss.Read();
  double* z = cg_z.Write();
  launch_red<1>(n, dsc, slots(RZ), [=] MFEM_HOST_DEVICE (int i, double* v) {
    // z = z2 + w D^-1 (r - S z2) (post-smoothing of the finest level); sc[RZ] += r . z
    float acc = sell_row_ff(i, soff, scol, dSs, z2);
    double ri = r[i];
    double zi = (double)(z2[i] + wv[0] * Dinv[i] * ((float)ri - acc));
    z[i] = zi;
    v[0] = ri * zi;
    if (i < MAXF) dsc[ACC + i] = 0.0;
  });
}

/// cg_chunk iterations of the AMG preconditioned CG (cgrad::schur). Scalars:
/// sc[RZ + b] with the parity b of the iteration (r.z of the current and the
/// previous iteration), sc[PQ], sc[RR], sc[ACC + f] = g_f . p of the faces.
/// Each kernel zeroes the accumulators that a later kernel fills.
void NsSolver::emit_cg_chunk()
{
  AmgLevel& L0 = amg->level(0);
  AmgLevel& L1 = amg->level(1);
  double* dsc = sc.ReadWrite();
  double* X = cg_X.ReadWrite();
  double* r = cg_r.ReadWrite();
  double* z = cg_z.ReadWrite();
  double* p = cg_p.ReadWrite();
  double* q = cg_q.ReadWrite();
  float* zs = cg_zs.ReadWrite();
  float* z2 = cg_z2.ReadWrite();
  const float* Dinv = L0.Dinv.Read();
  const float* wv = amg->damping();
  const int nc1 = L1.n;
  float* rc1 = L1.r.ReadWrite();
  const int* agg = L0.agg.Read();
  const float* pb = L0.pb.Read();
  const int* soff = sellS.slice_off.Read();
  const int* scol = sellS.cols.Read();
  const float* dSs = Ss.Read();
  const int n = nNo, nf = nF;
  const double* dg = g.Read();
  const double* dfd = fd.Read();
  const int cpre = cpre_off;
  for (int it = 0; it < cg_chunk; it++) {
    const int b = it & 1;
    // p = z + (rz_new / rz_old) p; sc[ACC + f] += g_f . p
    if (nf == 0) {
      launch(n, [=] MFEM_HOST_DEVICE (int i) {
        double beta = dsc[RZ + b] / dsc[RZ + 1 - b];
        p[i] = z[i] + beta * p[i];
        if (i == 0) dsc[PQ] = 0.0;
      });
    } else if (nf == 1) {
      launch_red<1>(n, dsc, slots(ACC), [=] MFEM_HOST_DEVICE (int i, double* v) {
        double beta = dsc[RZ + b] / dsc[RZ + 1 - b];
        double pi = z[i] + beta * p[i];
        p[i] = pi;
        v[0] = dg[i] * pi;
        if (i == 0) dsc[PQ] = 0.0;
      });
    } else {
      launch(n, [=] MFEM_HOST_DEVICE (int i) {
        double beta = dsc[RZ + b] / dsc[RZ + 1 - b];
        p[i] = z[i] + beta * p[i];
        if (i == 0) dsc[PQ] = 0.0;
      });
      launch_red_multi<MAXF>(n, nf, dsc, range_slots(ACC, nf), [=] MFEM_HOST_DEVICE (int i, int f) {
        return dg[(size_t)f * n + i] * p[i];
      });
    }
    // q = S p + sum_f cpre_f (g_f . p) g_f; sc[PQ] += p . q
    launch_red<1>(n, dsc, slots(PQ), [=] MFEM_HOST_DEVICE (int i, double* v) {
      double acc = sell_row_fd(i, soff, scol, dSs, p);
      for (int f = 0; f < nf; f++) acc = acc + dfd[cpre + f] * dsc[ACC + f] * dg[(size_t)f * n + i];
      q[i] = acc;
      v[0] = p[i] * acc;
      if (i == 0) dsc[RR] = 0.0;
    });
    // alpha = rz / pq; X += alpha p; r -= alpha q; sc[RR] += |r|^2; zs = w D^-1 r
    launch_red<1>(n, dsc, slots(RR), [=] MFEM_HOST_DEVICE (int i, double* v) {
      double alpha = dsc[RZ + b] / dsc[PQ];
      X[i] = X[i] + alpha * p[i];
      double ri = r[i] - alpha * q[i];
      r[i] = ri;
      zs[i] = wv[0] * Dinv[i] * (float)ri;
      v[0] = ri * ri;
      if (i == 0) dsc[RZ + 1 - b] = 0.0;
      if (i < nc1) rc1[i] = 0.0f;
    });
    const float* zc = amg->emit_down(r, zs);
    launch(n, [=] MFEM_HOST_DEVICE (int i) { z2[i] = zs[i] + pb[i] * zc[agg[i]]; });
    launch_red<1>(n, dsc, slots(RZ + 1 - b), [=] MFEM_HOST_DEVICE (int i, double* v) {
      float acc = sell_row_ff(i, soff, scol, dSs, z2);
      double ri = r[i];
      double zi = (double)(z2[i] + wv[0] * Dinv[i] * ((float)ri - acc));
      z[i] = zi;
      v[0] = ri * zi;
      if (i < MAXF) dsc[ACC + i] = 0.0;
    });
  }
  publish();
}

// GMRES on the momentum block (gmres::gmres with dof = nsd). u_i is stored
// normalized by the kernel of step i from the unnormalized vector w of the
// previous step (norm |w|^2 in gs[NRM + b'], b' the parity of the previous
// step). The face terms need v_f . u_i and (K^T v_f) . u_i, which are
// accumulated as dots with w by the previous step: the coupled face products
// are applied without any extra reduction. Gram-Schmidt is the classical
// variant (all projections from one reduction kernel).

void NsSolver::emit_gm_init()
{
  double* dgs = gs.ReadWrite();
  fill(dgs, ngs, 0.0);
  fill(gX.Write(), n3, 0.0);
  const double* dgR = gR.Read();
  double* w = wb1.Write();
  const double* dVf = Vf.Read();
  const int n = n3, nf = nF;
  // w = R; gs[VW + f] += v_f . R
  copy(dgR, w, n);
  if (nf > 0) {
    launch_red_multi<MAXF>(n, nf, dgs, range_slots(VW, nf), [=] MFEM_HOST_DEVICE (int k, int f) {
      return dVf[(size_t)f * n + k] * w[k];
    });
  }
  emit_gm_init2();
}

/// PRE face terms of w (dots in gs[VW + f]); then the quantities of the
/// "previous step" of step 0 in the parity-1 slots: |w|^2, v_f . w, kv_f . w
void NsSolver::emit_gm_init2()
{
  double* dgs = gs.ReadWrite();
  double* w = wb1.ReadWrite();
  const double* dVf = Vf.Read();
  const double* dkv = kv.Read();
  const double* dfd = fd.Read();
  const int n = n3, nf = nF, cpre = cpre_off, vvo = vv_off, sd = sD;
  if (nf > 0) {
    launch(n, [=] MFEM_HOST_DEVICE (int k) {
      double t[MAXF];
      for (int f = 0; f < nf; f++) {
        double e = dgs[VW + f];
        for (int q = 0; q < f; q++) e = e + dfd[cpre + q] * t[q] * dfd[vvo + f * MAXF + q];
        t[f] = e;
      }
      double wk = w[k];
      for (int f = 0; f < nf; f++) wk = wk + dfd[cpre + f] * t[f] * dVf[(size_t)f * n + k];
      w[k] = wk;
    });
  }
  SlotList sl{};
  sl.n = 1 + 2 * nf;
  sl.s[0] = NRM + 1;
  for (int f = 0; f < nf; f++) {
    sl.s[1 + 2 * f] = VW + MAXF + f;
    sl.s[2 + 2 * f] = KW + MAXF + f;
  }
  launch_red_multi<8>(n, 1 + 2 * nf, dgs, sl, [=] MFEM_HOST_DEVICE (int k, int j) {
    double wk = w[k];
    if (j == 0) {
      if (k <= sd) dgs[GH + k] = 0.0;
      return wk * wk;
    }
    int f = (j - 1) / 2;
    return ((j - 1) % 2 == 0 ? dVf[(size_t)f * n + k] : dkv[(size_t)f * n + k]) * wk;
  });
  pub(gs, *hgs, 0, 8);
}

/// w = R - K X - face terms (restart of GMRES after sD steps)
void NsSolver::emit_gm_restart()
{
  double* dgs = gs.ReadWrite();
  fill(dgs, ngs, 0.0);
  const double* X = gX.Read();
  const double* dVf = Vf.Read();
  const double* dfd = fd.Read();
  const double* dgR = gR.Read();
  double* w = wb1.Write();
  const int* off = sellA.slice_off.Read();
  const int* col = sellA.cols.Read();
  const double* K = mKs.Read();
  const int n = n3, nf = nF, cadd = cadd_off, nn = nNo;
  if (nf > 0) {
    launch_red_multi<MAXF>(n, nf, dgs, range_slots(KW, nf), [=] MFEM_HOST_DEVICE (int k, int f) {
      return dVf[(size_t)f * n + k] * X[k];
    });
  }
  // w = R - (K X + sum_f res_f (v_f . X) v_f)
  launch(nn, [=] MFEM_HOST_DEVICE (int a) {
    double acc[3];
    sell_row_vv(a, off, col, K, X, acc);
    for (int f = 0; f < nf; f++) {
      double cf = dfd[cadd + f] * dgs[KW + f];
      for (int k = 0; k < 3; k++) acc[k] = acc[k] + cf * dVf[(size_t)f * n + 3 * a + k];
    }
    for (int k = 0; k < 3; k++) w[3 * a + k] = dgR[3 * a + k] - acc[k];
  });
  if (nf > 0) {
    launch_red_multi<MAXF>(n, nf, dgs, range_slots(VW, nf), [=] MFEM_HOST_DEVICE (int k, int f) {
      return dVf[(size_t)f * n + k] * w[k];
    });
  }
  emit_gm_init2();
}

void NsSolver::emit_gm_step(int i)
{
  const int b = i & 1, bp = 1 - b;
  double* dgs = gs.ReadWrite();
  double* u = gu.ReadWrite();
  // The unnormalized w of the previous step is in wb1 for even i (the
  // initialization writes wb1), the output goes to the other buffer
  const double* win = (b == 0 ? wb1 : wb0).Read();
  double* wout = (b == 0 ? wb0 : wb1).Write();
  const int* off = sellA.slice_off.Read();
  const int* col = sellA.cols.Read();
  const double* K = mKs.Read();
  const double* dVf = Vf.Read();
  const double* dkv = kv.Read();
  const double* dfd = fd.Read();
  const int n = n3, nf = nF, nn = nNo, sd = sD;
  const int cadd = cadd_off, cpre = cpre_off, vvo = vv_off;
  const int g0 = GH + b * (sd + 1);
  double* ui = u + (size_t)i * n3;
  // u_i = win / |win|; wout = K u_i + face terms
  launch(nn, [=] MFEM_HOST_DEVICE (int a) {
    double inv = 1.0 / sqrt(dgs[NRM + bp]);
    double s[MAXF];
    face_coefs(nf, dgs, bp, inv, dfd, cadd, cpre, vvo, s);
    double acc[3];
    sell_row_vv(a, off, col, K, win, acc);
    for (int k = 0; k < 3; k++) {
      acc[k] = acc[k] * inv;
      for (int f = 0; f < nf; f++) acc[k] = acc[k] + s[f] * dVf[(size_t)f * n + 3 * a + k];
      wout[3 * a + k] = acc[k];
      ui[3 * a + k] = win[3 * a + k] * inv;
    }
  });
  // gs[GH_b + j] += u_j . wout, j <= i (classical Gram-Schmidt); clear the
  // accumulators of the next step
  launch_red_multi<8>(n, i + 1, dgs, range_slots(g0, i + 1), [=] MFEM_HOST_DEVICE (int k, int j) {
    if (j == 0) {
      if (k == 0) dgs[NRM + b] = 0.0;
      if (k < MAXF) {
        dgs[VW + b * MAXF + k] = 0.0;
        dgs[KW + b * MAXF + k] = 0.0;
      }
    }
    return u[(size_t)j * n + k] * wout[k];
  });
  // w -= sum_{j<=i} h_j u_j
  launch(n, [=] MFEM_HOST_DEVICE (int k) {
    double wk = wout[k];
    for (int j = 0; j <= i; j++) wk = wk - dgs[g0 + j] * u[(size_t)j * n + k];
    wout[k] = wk;
  });
  // gs[NRM + b] += |w|^2; face dots with w; gs[GH_b'] = 0
  SlotList sl{};
  sl.n = 1 + 2 * nf;
  sl.s[0] = NRM + b;
  for (int f = 0; f < nf; f++) {
    sl.s[1 + 2 * f] = VW + b * MAXF + f;
    sl.s[2 + 2 * f] = KW + b * MAXF + f;
  }
  const int g1 = GH + bp * (sd + 1);
  launch_red_multi<8>(n, 1 + 2 * nf, dgs, sl, [=] MFEM_HOST_DEVICE (int k, int j) {
    double wk = wout[k];
    if (j == 0) {
      if (k <= sd) dgs[g1 + k] = 0.0;
      return wk * wk;
    }
    int f = (j - 1) / 2;
    return ((j - 1) % 2 == 0 ? dVf[(size_t)f * n + k] : dkv[(size_t)f * n + k]) * wk;
  });
  pub(gs, *hgs, 0, 8);
  pub(gs, *hgs, g0, g0 + i + 1);
}

/// X += sum_j y_j u_j with the coefficients from the host
void NsSolver::emit_gm_fin()
{
  fetch_coef();
  const double* coef = dcoef.Read();
  const double* u = gu.Read();
  double* X = gX.ReadWrite();
  const int n = n3;
  launch(n, [=] MFEM_HOST_DEVICE (int k) {
    int m = (int)coef[0];
    double s = X[k];
    for (int j = 0; j < m; j++) s = s + coef[1 + j] * u[(size_t)j * n + k];
    X[k] = s;
  });
}

/// U_i = GMRES solution; P_i = Rc - D U_i; CG initialization on P_i
void NsSolver::emit_A(int i)
{
  double* Ui = U.ReadWrite() + (size_t)i * n3;
  double* Pi = P.ReadWrite() + (size_t)i * nNo;
  copy(gX.Read(), Ui, n3);
  const int* off = sellA.slice_off.Read();
  const int* col = sellA.cols.Read();
  const double* D = mDs.Read();
  const double* dRc = Rc.Read();
  launch(nNo, [=] MFEM_HOST_DEVICE (int a) { Pi[a] = dRc[a] - sell_row_vs(a, off, col, D, Ui); });
  emit_cg_init(Pi);
  publish();
}

/// P_i = CG solution; MU_{2i} = G P_i; MU_{2i+1} = Rm - G P_i = next GMRES rhs
void NsSolver::emit_B(int i)
{
  double* Pi = P.ReadWrite() + (size_t)i * nNo;
  double* MU0 = MU.ReadWrite() + (size_t)(2 * i) * n3;
  double* MU1 = MU.ReadWrite() + (size_t)(2 * i + 1) * n3;
  copy(cg_X.Read(), Pi, nNo);
  const int* off = sellA.slice_off.Read();
  const int* col = sellA.cols.Read();
  const double* G = mGs.Read();
  const double* dRm = Rm.Read();
  launch(nNo, [=] MFEM_HOST_DEVICE (int a) {
    double acc[3];
    sell_row_sv(a, off, col, G, Pi, acc);
    for (int k = 0; k < 3; k++) {
      MU0[3 * a + k] = acc[k];
      MU1[3 * a + k] = dRm[3 * a + k] - acc[k];
    }
  });
  copy(MU1, gR.Write(), n3);
}

/// U_i = GMRES solution; MU_{2i+1} = K U_i (+ faces); MP_{2i} = L P_i;
/// MP_{2i+1} = D U_i; dots of the Gram matrix
void NsSolver::emit_C(int i)
{
  const int iB2 = 2 * i, iBB = 2 * i + 1;
  double* dsc = sc.ReadWrite();
  double* Ui = U.ReadWrite() + (size_t)i * n3;
  const double* Pi = P.Read() + (size_t)i * nNo;
  double* MUb = MU.ReadWrite() + (size_t)iBB * n3;
  double* MP0 = MP.ReadWrite() + (size_t)iB2 * nNo;
  double* MP1 = MP.ReadWrite() + (size_t)iBB * nNo;
  copy(gX.Read(), Ui, n3);
  fill(dsc + ACC, nsc - ACC, 0.0);
  const double* dVf = Vf.Read();
  const double* dfd = fd.Read();
  const int* off = sellA.slice_off.Read();
  const int* col = sellA.cols.Read();
  const double* K = mKs.Read();
  const double* D = mDs.Read();
  const double* L = mLs.Read();
  const int n = n3, nf = nF, nn = nNo, cadd = cadd_off;
  if (nf > 0) {
    launch_red_multi<MAXF>(n, nf, dsc, range_slots(ACC, nf), [=] MFEM_HOST_DEVICE (int k, int f) {
      return dVf[(size_t)f * n + k] * Ui[k];
    });
  }
  launch(nn, [=] MFEM_HOST_DEVICE (int a) {
    double acc[3];
    sell_row_vv(a, off, col, K, Ui, acc);
    for (int f = 0; f < nf; f++) {
      double cf = dfd[cadd + f] * dsc[ACC + f];
      for (int k = 0; k < 3; k++) acc[k] = acc[k] + cf * dVf[(size_t)f * n + 3 * a + k];
    }
    for (int k = 0; k < 3; k++) MUb[3 * a + k] = acc[k];
    MP0[a] = sell_row_ss(a, off, col, L, Pi);
    MP1[a] = sell_row_vs(a, off, col, D, Ui);
  });
  // Gram matrix: sc[slot + j] += MU_j . MU_k (j <= k), sc[slot + k + 1] += MU_k . Rmi, the same for MP
  const double* dMU = MU.Read();
  const double* dMP = MP.Read();
  const double* dRmi = Rmi.Read();
  const double* dRci = Rci.Read();
  int slot = GRAM;
  for (int k : {iB2, iBB}) {
    const double* MUk = dMU + (size_t)k * n3;
    launch_red_multi<8>(n, k + 2, dsc, range_slots(slot, k + 2), [=] MFEM_HOST_DEVICE (int q, int j) {
      return (j <= k ? dMU[(size_t)j * n + q] : dRmi[q]) * MUk[q];
    });
    slot += k + 2;
    const double* MPk = dMP + (size_t)k * nNo;
    launch_red_multi<8>(nn, k + 2, dsc, range_slots(slot, k + 2), [=] MFEM_HOST_DEVICE (int q, int j) {
      return (j <= k ? dMP[(size_t)j * nn + q] : dRci[q]) * MPk[q];
    });
    slot += k + 2;
  }
  publish(GRAM, slot);
}

/// Rm = Rmi - sum c_j MU_j, Rc = Rci - sum c_j MP_j, the residual norm; next GMRES rhs = Rm
void NsSolver::emit_D(int)
{
  fetch_coef();
  double* dsc = sc.ReadWrite();
  const double* coef = dcoef.Read();
  const double* dMU = MU.Read();
  const double* dMP = MP.Read();
  const double* dRmi = Rmi.Read();
  const double* dRci = Rci.Read();
  double* dRm = Rm.Write();
  double* dRc = Rc.Write();
  const int n = n3, nn = nNo;
  launch(n, [=] MFEM_HOST_DEVICE (int k) {
    int m = (int)coef[0];
    double s = 0.0;
    for (int j = 0; j < m; j++) s = s + coef[1 + j] * dMU[(size_t)j * n + k];
    dRm[k] = dRmi[k] - s;
    if (k == 0) dsc[RES_M] = 0.0;
  });
  launch(nn, [=] MFEM_HOST_DEVICE (int k) {
    int m = (int)coef[0];
    double s = 0.0;
    for (int j = 0; j < m; j++) s = s + coef[1 + j] * dMP[(size_t)j * nn + k];
    dRc[k] = dRci[k] - s;
    if (k == 0) dsc[RES_C] = 0.0;
  });
  dot(n3, dRm, dRm, dsc, RES_M);
  dot(nNo, dRc, dRc, dsc, RES_C);
  copy(dRm, gR.Write(), n3);
  publish();
}

/// Final residual of the continuity part and the solution R = (sum xB_{2i+1} U_i, sum xB_{2i} P_i)
void NsSolver::emit_E()
{
  fetch_coef();
  double* dsc = sc.ReadWrite();
  const double* coef = dcoef.Read();
  const double* dMP = MP.Read();
  const double* dU = U.Read();
  const double* dP = P.Read();
  double* dRmi = Rmi.ReadWrite();
  double* dRci = Rci.ReadWrite();
  double* dRc = Rc.Write();
  double* dR = R.Write();
  const int n = n3, nn = nNo, cs = coef_seg;
  launch(nn, [=] MFEM_HOST_DEVICE (int k) {
    int m = (int)coef[0];
    double s = 0.0;
    for (int j = 0; j < m; j++) s = s + coef[1 + j] * dMP[(size_t)j * nn + k];
    dRc[k] = dRci[k] - s;
    if (k == 0) dsc[RES_C] = 0.0;
  });
  dot(nNo, dRc, dRc, dsc, RES_C);
  launch(n, [=] MFEM_HOST_DEVICE (int k) {
    int m = (int)coef[cs];
    double s = 0.0;
    for (int j = 0; j < m; j++) s = s + coef[cs + 1 + j] * dU[(size_t)j * n + k];
    dRmi[k] = s;
  });
  launch(nn, [=] MFEM_HOST_DEVICE (int k) {
    int m = (int)coef[2 * cs];
    double s = 0.0;
    for (int j = 0; j < m; j++) s = s + coef[2 * cs + 1 + j] * dP[(size_t)j * nn + k];
    dRci[k] = s;
  });
  launch(nn, [=] MFEM_HOST_DEVICE (int a) {
    for (int k = 0; k < 3; k++) dR[4 * a + k] = dRmi[3 * a + k];
    dR[4 * a + 3] = dRci[a];
  });
  publish();
}

// ---------------------------------------------------------------------------
// Drivers

/// gX = inv(K) gR; ls accumulates itr
void NsSolver::gmres(SubLs& ls)
{
  double* h_gs = hgs->host;
  const int sd = ls.sD;
  std::vector<double> h((size_t)(sd + 1) * sd, 0.0), y(sd), c(sd), s(sd), err(sd + 1);
  auto H = [&](int i, int j) -> double& { return h[i + (size_t)j * (sd + 1)]; };
  ls.success = false;
  double eps = 0.0;
  int last_i = 0;
  for (int l = 0; l < ls.mItr; l++) {
    if (l == 0) {
      run("gm_init", [&] { emit_gm_init(); });
    } else {
      run("gm_restart", [&] { emit_gm_restart(); });
      ls.itr++;
    }
    sync();
    err[0] = std::sqrt(h_gs[NRM + 1]);
    if (l == 0) {
      eps = err[0];
      ls.iNorm = eps;
      ls.fNorm = eps;
      eps = std::max(ls.absTol, ls.relTol * eps);
    }
    if (err[0] == 0.0) {
      ls.success = true;
      ls.fNorm = ls.dB = 0.0;
      return;
    }
    ls.dB = ls.fNorm;
    for (int i = 0; i < sd; i++) {
      last_i = i;
      run("gm" + std::to_string(i), [&] { emit_gm_step(i); });
      ls.itr++;
      sync();
      const int b = i & 1;
      for (int j = 0; j <= i; j++) H(j, i) = h_gs[GH + b * (sd + 1) + j];
      H(i + 1, i) = std::sqrt(h_gs[NRM + b]);
      for (int j = 0; j < i; j++) {
        double tmp = c[j] * H(j, i) + s[j] * H(j + 1, i);
        H(j + 1, i) = -s[j] * H(j, i) + c[j] * H(j + 1, i);
        H(j, i) = tmp;
      }
      double tmp = std::sqrt(H(i, i) * H(i, i) + H(i + 1, i) * H(i + 1, i));
      c[i] = H(i, i) / tmp;
      s[i] = H(i + 1, i) / tmp;
      H(i, i) = tmp;
      H(i + 1, i) = 0.0;
      err[i + 1] = -s[i] * err[i];
      err[i] = c[i] * err[i];
      if (std::fabs(err[i + 1]) < eps) {
        ls.success = true;
        break;
      }
    }
    for (int i = 0; i <= last_i; i++) y[i] = err[i];
    for (int j = last_i; j >= 0; j--) {
      for (int k = j + 1; k <= last_i; k++) y[j] = y[j] - H(j, k) * y[k];
      y[j] = y[j] / H(j, j);
    }
    double* hc = hcoef->host;
    hc[0] = last_i + 1;
    for (int j = 0; j <= last_i; j++) hc[1 + j] = y[j];
    run("gm_fin", [&] { emit_gm_fin(); });
    ls.fNorm = std::fabs(err[last_i + 1]);
    if (ls.success) break;
  }
  ls.dB = 10.0 * std::log(ls.fNorm / ls.dB);
}

/// AMG preconditioned CG on the Schur complement, initialized by
/// emit_cg_init (err = |r|^2 from it); the solution is in cg_X
int NsSolver::cg_schur(SubLs& ls, double err)
{
  double* h_sc = hsc->host;
  ls.success = false;
  const double iNorm = std::sqrt(err);
  ls.iNorm = iNorm;
  const double eps = std::pow(std::max(ls.absTol, ls.relTol * iNorm), 2.0);
  double errO = err;
  int nit = 0;
  while (nit < ls.mItr) {
    if (err < eps) {
      ls.success = true;
      break;
    }
    errO = err;
    run("cg_chunk", [&] { emit_cg_chunk(); });
    nit += cg_chunk;
    sync();
    err = h_sc[RR];
  }
  ls.fNorm = std::sqrt(err);
  ls.itr += nit;
  ls.dB = errO < std::numeric_limits<double>::epsilon() ? 0.0 : 5.0 * std::log(err / errO);
  return nit;
}

void NsSolver::solve(double* R_host, int nFaces, const svmfem_face* faces, const svmfem_params& p,
    svmfem_stats* stats)
{
  trace("solve");
  const double t0 = now();
  const int iB_ = p.RI.mItr;
  const int nB = 2 * iB_;
  alloc_work(iB_, p.GM.sD);
  SubLs RI{p.RI.mItr, p.RI.sD, p.RI.absTol, p.RI.relTol};
  SubLs GM{p.GM.mItr, p.GM.sD, p.GM.absTol, p.GM.relTol};
  SubLs CG{p.CG.mItr, p.CG.sD, p.CG.absTol, p.CG.relTol};
  double* h_sc = hsc->host;
  double* hc = hcoef->host;
  dcopy_in(R, R_host, 4 * nNo);
  trace("start");
  run("start", [&] { emit_start(); });
  sync();
  const double rm2 = h_sc[EPS_M], rc2_0 = h_sc[EPS_C];
  double eps = std::sqrt(rm2 + rc2_0);
  RI.iNorm = eps;
  RI.fNorm = eps * eps;
  RI.itr = GM.itr = CG.itr = 0;
  RI.dB = 0.0;
  RI.success = false;
  int Resc = 0, Resm = 0;
  eps = std::max(RI.absTol, RI.relTol * eps);
  const bool finite_norm = std::isfinite(RI.iNorm);
  if (!finite_norm || RI.iNorm <= eps) {
    for (int i = 0; i < 4 * nNo; i++) R_host[i] = 0.0;
    RI.success = finite_norm;
    Resc = finite_norm && RI.fNorm > 0.0 ? (int)(100.0 * rc2_0 / RI.fNorm) : 0;
    Resm = finite_norm && RI.fNorm > 0.0 ? 100 - Resc : 0;
    RI.fNorm = RI.iNorm;
  } else {
    double tc = now();
    set_faces(nFaces, faces);
    const int nPow = amg->npow();
    trace("setup");
    run("setup" + std::to_string(nPow), [&] { emit_setup(nPow); });
    sync();
    trace("setup done");
    t_setup += now() - tc;
    std::vector<double> A((size_t)nB * nB, 0.0), B(nB, 0.0), xB(nB, 0.0), oldxB(nB, 0.0);
    auto Am = [&](int i, int j) -> double& { return A[i + (size_t)j * nB]; };
    for (int i = 0; i < RI.mItr; i++) {
      const int iB2 = 2 * i, iBB = 2 * i + 1;
      RI.dB = RI.fNorm;
      // U = inv(K) Rm
      trace("gmres 1");
      double t = now();
      gmres(GM);
      t_gm += now() - t;
      // P = [L + G^t G]^-1 (Rc - D U)
      t = now();
      trace("A");
      run("A" + std::to_string(i), [&] { emit_A(i); });
      sync();
      trace("cg");
      cg_schur(CG, h_sc[RR]);
      t_cg += now() - t;
      // MU1 = G P, MU2 = Rm - G P; U = inv(K) [Rm - G P]
      trace("B");
      run("B" + std::to_string(i), [&] { emit_B(i); });
      t = now();
      trace("gmres 2");
      gmres(GM);
      t_gm += now() - t;
      // MU2 = K U, MP1 = L P, MP2 = D U and the Gram matrix
      trace("C");
      run("C" + std::to_string(i), [&] { emit_C(i); });
      sync();
      {
        int slot = GRAM;
        for (int k : {iB2, iBB}) {
          const int m = k + 2;
          for (int j = 0; j <= k; j++) {
            double v = h_sc[slot + j] + h_sc[slot + m + j];
            Am(j, k) = v;
            Am(k, j) = v;
          }
          B[k] = h_sc[slot + k + 1] + h_sc[slot + m + k + 1];
          slot += 2 * m;
        }
      }
      const int size = iBB + 1;
      std::vector<double> projection((size_t)size * size), scale(size), projected_rhs(size), coefficients(size);
      bool finite = true;
      for (int j = 0; j < size; j++) {
        scale[j] = Am(j, j) == 0.0 ? 1.0 : std::sqrt(Am(j, j));
        projected_rhs[j] = B[j] / scale[j];
        finite = finite && std::isfinite(projected_rhs[j]);
      }
      for (int k = 0; k < size; k++) {
        for (int j = 0; j < size; j++) {
          projection[j + (size_t)k * size] = Am(j, k) / scale[j] / scale[k];
          finite = finite && std::isfinite(projection[j + (size_t)k * size]);
        }
      }
      if (!finite) break;
      // Least squares solution of the projected system (as ns_solver.cpp)
      lstsq_solve(size, projection.data(), projected_rhs.data(), coefficients.data());
      for (int j = 0; j < size; j++) {
        coefficients[j] /= scale[j];
        finite = finite && std::isfinite(coefficients[j]);
      }
      if (!finite) break;
      oldxB = xB;
      for (int j = 0; j < size; j++) xB[j] = coefficients[j];
      hc[0] = size;
      for (int j = 0; j < size; j++) hc[1 + j] = xB[j];
      trace("D");
      run("D" + std::to_string(i), [&] { emit_D(i); });
      sync();
      const double residual2 = h_sc[RES_M] + h_sc[RES_C];
      if (!std::isfinite(residual2) || residual2 >= RI.fNorm) {
        xB = oldxB;
        break;
      }
      RI.fNorm = residual2;
      RI.itr = i + 1;
      if (RI.fNorm <= eps * eps) {
        RI.success = true;
        break;
      }
    }
    const int m = 2 * RI.itr;
    hc[0] = m;
    for (int j = 0; j < m; j++) hc[1 + j] = xB[j];
    const int cs = coef_seg;
    hc[cs] = RI.itr;
    hc[2 * cs] = RI.itr;
    for (int j = 0; j < RI.itr; j++) {
      hc[cs + 1 + j] = xB[2 * j + 1];
      hc[2 * cs + 1 + j] = xB[2 * j];
    }
    trace("E");
    run("E", [&] { emit_E(); });
    sync();
    const double rc2 = h_sc[RES_C];
    Resc = RI.fNorm > 0.0 ? (int)(100.0 * rc2 / RI.fNorm) : 0;
    Resm = RI.fNorm > 0.0 ? 100 - Resc : 0;
    RI.dB = RI.fNorm > 0.0 && RI.dB > 0.0 ? 5.0 * std::log(RI.fNorm / RI.dB) : 0.0;
    RI.fNorm = std::sqrt(RI.fNorm);
    ddownload(R, R_host, 4 * nNo);
  }
  const double dt = now() - t0;
  t_all += dt;
  n_call++;
  n_gm += GM.itr;
  n_cg += CG.itr;
  if (getenv("SVMFEM_TIMING")) {
    const int n = n_call;
    printf("SVMFEM calls=%d per call: all=%.4f schur setup=%.4f gmres=%.4f (%.3f ms/it) cg=%.4f (%.3f ms/it)\n", n,
        t_all / n, t_setup / n, t_gm / n, 1e3 * t_gm / std::max(1, n_gm), t_cg / n, 1e3 * t_cg / std::max(1, n_cg));
  }
  if (stats) {
    stats->RI_itr = RI.itr;
    stats->GM_itr = GM.itr;
    stats->CG_itr = CG.itr;
    stats->success = RI.success ? 1 : 0;
    stats->iNorm = RI.iNorm;
    stats->fNorm = RI.fNorm;
    stats->dB = RI.dB;
    stats->Resm = Resm;
    stats->Resc = Resc;
    stats->time = dt;
  }
}

NsSolver* solver = nullptr;

}  // namespace ns

void destroy_solver()
{
  delete ns::solver;
  ns::solver = nullptr;
}

}  // namespace svmfem

using namespace svmfem;
using namespace svmfem::ns;

extern "C" SVMFEM_API int svmfem_ns_solve(double* R, int nFaces, const svmfem_face* faces,
    const svmfem_params* params, svmfem_stats* stats)
{
  try {
    if (!lhs()) fail("svmfem_fluid_setup was not called");
    if (!solver) solver = new NsSolver(*lhs());
    solver->solve(R, nFaces, faces, *params, stats);
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "svmfem_ns_solve: %s\n", e.what());
    return 1;
  }
}
