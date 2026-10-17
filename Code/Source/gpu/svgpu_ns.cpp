// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// GPU implementation of the FSILS NS solver (linear_solver/ns_solver.cpp,
// gmres.cpp, cgrad.cpp). The algorithm is the same; all vectors are on the GPU
// and only scalars are transferred to the host.

#include "svgpu.h"
#include "svgpu_internal.h"

#include <Eigen/QR>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <algorithm>

extern double* svgpu_dev_val_ptr;

namespace svgpu {

namespace {

constexpr int nsd = 3;

struct Work {
  int nNo = 0, iB = 0, sD = 0;
  double *R = nullptr, *Rm = nullptr, *Rmi = nullptr, *Rc = nullptr, *Rci = nullptr;
  double *U = nullptr, *MU = nullptr, *P = nullptr, *MP = nullptr;
  double *gu = nullptr, *gX = nullptr, *gh = nullptr;
};

Gpu* gpu = nullptr;
Work work;
double tGM = 0, tCG = 0, tAll = 0, tDep = 0; int nCall = 0, nCGit = 0;

double now()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void alloc_work(int nNo, int iB, int sD)
{
  if (work.nNo == nNo && work.iB >= iB && work.sD >= sD) return;
  work.nNo = nNo;
  work.iB = iB;
  work.sD = sD;
  size_t n3 = 3 * (size_t)nNo;
  work.R = gpu->alloc(4 * (size_t)nNo);
  work.Rm = gpu->alloc(n3);
  work.Rmi = gpu->alloc(n3);
  work.Rc = gpu->alloc(nNo);
  work.Rci = gpu->alloc(nNo);
  work.U = gpu->alloc(n3 * iB);
  work.MU = gpu->alloc(n3 * 2 * iB);
  work.P = gpu->alloc((size_t)nNo * iB);
  work.MP = gpu->alloc((size_t)nNo * 2 * iB);
  work.gu = gpu->alloc(n3 * (sD + 1));
  work.gX = gpu->alloc(n3);
  work.gh = gpu->alloc((size_t)(sD + 1) * sD);
}

struct SubLs {
  int mItr, sD;
  double absTol, relTol;
  bool success = false;
  int itr = 0;
  double iNorm = 0, fNorm = 0, dB = 0;
};

// gmres::gmres (dof = nsd): X = inv(K)*R, K = mK plus resistance terms
void gmres(SubLs& ls, const double* R, double* X)
{
  const int nNo = gpu->nNo;
  const int n = nsd * nNo;
  const bool coupled = gpu->nCoupled > 0;
  double* u = work.gu;
  double* hd = work.gh;
  auto uj = [&](int j) { return u + (size_t)j * n; };

  std::vector<double> h((size_t)(ls.sD + 1) * ls.sD), y(ls.sD), c(ls.sD), s(ls.sD), err(ls.sD + 1);
  auto H = [&](int i, int j) -> double& { return h[i + (size_t)j * (ls.sD + 1)]; };

  ls.success = false;
  double eps = 0.0;
  int last_i = 0;
  gpu->zero(X, n);

  for (int l = 0; l < ls.mItr; l++) {
    if (l == 0) {
      gpu->copy(uj(0), R, n);
    } else {
      gpu->spmv_vv(gpu->mK, X, uj(0));
      gpu->bc_add(X, uj(0));
      ls.itr = ls.itr + 1;
      gpu->sub(R, uj(0), uj(0), n);
    }
    if (coupled) gpu->bc_pre(uj(0));

    err[0] = std::sqrt(gpu->dot(uj(0), uj(0), n));
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
    gpu->scale(1.0 / err[0], uj(0), n);

    for (int i = 0; i < ls.sD; i++) {
      last_i = i;
      static const bool coopArnoldi = !getenv("SVGPU_GMRES_KERNELS");
      std::vector<double> hc(i + 2);
      if (coopArnoldi) {
        gpu->arnoldi(u, i, hc.data());
        ls.itr = ls.itr + 1;
      } else {
        gpu->spmv_vv(gpu->mK, uj(i), uj(i + 1));
        gpu->bc_add(uj(i), uj(i + 1));
        ls.itr = ls.itr + 1;
        if (coupled) gpu->bc_pre(uj(i + 1));

        // Modified Gram-Schmidt, h(:,i) on the device
        double* w = uj(i + 1);
        double* hcol = hd + (size_t)i * (ls.sD + 1);
        for (int j = 0; j <= i; j++) {
          gpu->dot_dev(uj(j), w, n, hcol + j);
          gpu->axpy_dev(hcol + j, -1.0, uj(j), w, n);
        }
        gpu->dot_dev(w, w, n, hcol + i + 1);
        // h(i+1,i) = sqrt(w.w) is applied on the host after the download; the
        // device value is squared, so normalize with its square root
        gpu->download(hc.data(), hcol, i + 2);
        hc[i + 1] = std::sqrt(hc[i + 1]);
        gpu->scale(1.0 / hc[i + 1], w, n);
      }
      for (int j = 0; j <= i + 1; j++) H(j, i) = hc[j];

      for (int j = 0; j <= i - 1; j++) {
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

    if (last_i >= ls.sD) last_i = ls.sD - 1;
    for (int i = 0; i <= last_i; i++) y[i] = err[i];
    for (int j = last_i; j >= 0; j--) {
      for (int k = j + 1; k <= last_i; k++) y[j] = y[j] - H(j, k) * y[k];
      y[j] = y[j] / H(j, j);
    }
    std::vector<const double*> us;
    for (int j = 0; j <= last_i; j++) us.push_back(uj(j));
    gpu->lin_comb(X, 1.0, y.data(), us, X, n);

    ls.fNorm = std::fabs(err[last_i + 1]);
    if (ls.success) break;
  }
  ls.dB = 10.0 * std::log(ls.fNorm / ls.dB);
}

}  // namespace

}  // namespace svgpu

double* svgpu_dev_val_ptr_get();
using namespace svgpu;

extern "C" SVGPU_API int svgpu_ns_solve(int nNo, int nnz, int dof, const int* rowPtr, const int* colPtr,
    const double* Val, double* Rh, int nFaces, const svgpu_face* faces, const svgpu_params* params,
    svgpu_stats* stats)
{
  try {
    if (dof != 4) throw std::runtime_error("svgpu: only nsd = 3 is supported");
    double t0 = now();

    if (!gpu) {
      gpu = new Gpu();
      gpu->setup(nNo, nnz, rowPtr, colPtr);
    } else if (gpu->nNo != nNo || gpu->nnz != nnz) {
      throw std::runtime_error("svgpu: the matrix structure changed");
    }

    // Faces: nS, and coefficients of the resistance terms (gmres.cpp, add_bc_mul.cpp)
    std::vector<int> glob, off{0};
    std::vector<double> valM, coefAdd, coefPre;
    for (int f = 0; f < nFaces; f++) {
      const auto& fa = faces[f];
      double nS = 0.0;
      for (int a = 0; a < fa.nNo; a++) {
        for (int i = 0; i < nsd; i++) nS += fa.valM[i + nsd * a] * fa.valM[i + nsd * a];
        glob.push_back(fa.glob[a]);
        for (int i = 0; i < nsd; i++) valM.push_back(fa.valM[i + nsd * a]);
      }
      off.push_back((int)glob.size());
      coefAdd.push_back(fa.res);
      coefPre.push_back(-fa.res / (1.0 + fa.res * nS));
    }
    gpu->set_faces(glob, off, valM, coefAdd, coefPre);

    const svgpu_params& p = *params;
    const int iB = p.RI.mItr;
    const int nB = 2 * iB;
    alloc_work(nNo, iB, p.GM.sD);

    SubLs RI{p.RI.mItr, p.RI.sD, p.RI.absTol, p.RI.relTol};
    SubLs GM{p.GM.mItr, p.GM.sD, p.GM.absTol, p.GM.relTol};
    SubLs CG{p.CG.mItr, p.CG.sD, p.CG.absTol, p.CG.relTol};

    const size_t n3 = 3 * (size_t)nNo;
    gpu->upload(work.R, Rh, 4 * (size_t)nNo);
    gpu->split4(work.R, work.Rmi, work.Rci);
    gpu->copy(work.Rm, work.Rmi, n3);
    gpu->copy(work.Rc, work.Rci, nNo);
    auto Ui = [&](int i) { return work.U + i * n3; };
    auto MUi = [&](int i) { return work.MU + i * n3; };
    auto Pi = [&](int i) { return work.P + (size_t)i * nNo; };
    auto MPi = [&](int i) { return work.MP + (size_t)i * nNo; };

    double eps = std::sqrt(gpu->dot(work.Rm, work.Rm, (int)n3) + gpu->dot(work.Rc, work.Rc, nNo));
    RI.iNorm = eps;
    RI.fNorm = eps * eps;
    int Resc = 0, Resm = 0;
    RI.itr = 0;
    RI.dB = 0.0;
    RI.success = false;
    eps = std::max(RI.absTol, RI.relTol * eps);
    const bool finite_norm = std::isfinite(RI.iNorm);
    if (!finite_norm || RI.iNorm <= eps) {
      for (size_t i = 0; i < 4 * (size_t)nNo; i++) Rh[i] = 0.0;
      RI.success = finite_norm;
      double rc2 = gpu->dot(work.Rc, work.Rc, nNo);
      Resc = finite_norm && RI.fNorm > 0.0 ? (int)(100.0 * rc2 / RI.fNorm) : 0;
      Resm = finite_norm && RI.fNorm > 0.0 ? 100 - Resc : 0;
      RI.fNorm = RI.iNorm;
    } else {
      if (Val) {
        gpu->upload(gpu->Val, Val, 16 * (size_t)nnz);
      } else {
        gpu->Val = svgpu_dev_val_ptr;
      }
      { double t = now(); gpu->depart(); gpu->sync(); tDep += now() - t; }

      std::vector<double> A((size_t)nB * nB), B(nB), xB(nB, 0.0), oldxB(nB, 0.0);
      auto Am = [&](int i, int j) -> double& { return A[i + (size_t)j * nB]; };

      for (int i = 0; i < RI.mItr; i++) {
        int iB2 = 2 * i, iBB = 2 * i + 1;
        RI.dB = RI.fNorm;

        // U = inv(mK)*Rm
        { double t = now(); gmres(GM, work.Rm, Ui(i)); gpu->sync(); tGM += now() - t; }
        // P = Rc - D*U
        gpu->spmv_vs(gpu->mD, Ui(i), Pi(i));
        gpu->sub(work.Rc, Pi(i), Pi(i), nNo);
        // P = [L + G^t*G]^-1*P
        double tc = now();
        auto cg = gpu->cg_schur(Pi(i), CG.mItr, CG.absTol, CG.relTol);
        gpu->sync(); tCG += now() - tc; nCGit += cg.itr;
        if (getenv("SVGPU_CGDBG")) printf("CG itr=%d success=%d iNorm=%.3e fNorm=%.3e ratio=%.3e\n", cg.itr, (int)cg.success, cg.iNorm, cg.fNorm, cg.fNorm/cg.iNorm);
        CG.itr += cg.itr;
        // MU1 = G*P, MU2 = Rm - G*P
        gpu->spmv_sv(gpu->mG, Pi(i), MUi(iB2));
        gpu->sub(work.Rm, MUi(iB2), MUi(iBB), (int)n3);
        // U = inv(K)*[Rm - G*P]
        { double t = now(); gmres(GM, MUi(iBB), Ui(i)); gpu->sync(); tGM += now() - t; }
        // MU2 = K*U
        gpu->spmv_vv(gpu->mK, Ui(i), MUi(iBB));
        gpu->bc_add(Ui(i), MUi(iBB));
        // MP1 = L*P, MP2 = D*U
        gpu->spmv_ss(gpu->mL, Pi(i), MPi(iB2));
        gpu->spmv_vs(gpu->mD, Ui(i), MPi(iBB));

        // Dot products for the Gram matrix (on the device, one download)
        double* td = gpu->scal;
        int c = 0;
        for (int k = iB2; k <= iBB; k++) {
          for (int j = 0; j <= k; j++) {
            gpu->dot_dev(MUi(j), MUi(k), (int)n3, td + 2 * c);
            gpu->dot_dev(MPi(j), MPi(k), nNo, td + 2 * c + 1);
            c++;
          }
          gpu->dot_dev(MUi(k), work.Rmi, (int)n3, td + 2 * c);
          gpu->dot_dev(MPi(k), work.Rci, nNo, td + 2 * c + 1);
          c++;
        }
        std::vector<double> t2(2 * c), tmp(c);
        gpu->download(t2.data(), td, 2 * c);
        for (int k = 0; k < c; k++) tmp[k] = t2[2 * k] + t2[2 * k + 1];

        c = 0;
        for (int k = iB2; k <= iBB; k++) {
          for (int j = 0; j <= k; j++) {
            Am(j, k) = tmp[c];
            Am(k, j) = tmp[c];
            c++;
          }
          B[k] = tmp[c];
          c++;
        }

        const int size = iBB + 1;
        Eigen::MatrixXd projection(size, size);
        Eigen::VectorXd scale(size), projected_rhs(size);
        for (int j = 0; j < size; ++j) {
          scale(j) = Am(j, j) == 0.0 ? 1.0 : std::sqrt(Am(j, j));
          projected_rhs(j) = B[j] / scale(j);
        }
        for (int k = 0; k < size; ++k) {
          for (int j = 0; j < size; ++j) projection(j, k) = Am(j, k) / scale(j) / scale(k);
        }
        if (!projection.allFinite() || !projected_rhs.allFinite()) break;
        const Eigen::VectorXd coefficients =
            projection.completeOrthogonalDecomposition().solve(projected_rhs).cwiseQuotient(scale);
        if (!coefficients.allFinite()) break;
        oldxB = xB;
        for (int j = 0; j < size; ++j) xB[j] = coefficients(j);

        std::vector<const double*> mus, mps;
        for (int j = 0; j <= iBB; j++) {
          mus.push_back(MUi(j));
          mps.push_back(MPi(j));
        }
        gpu->lin_comb(work.Rmi, -1.0, xB.data(), mus, work.Rm, (int)n3);
        gpu->lin_comb(work.Rci, -1.0, xB.data(), mps, work.Rc, nNo);

        const double residual2 = gpu->dot(work.Rm, work.Rm, (int)n3) + gpu->dot(work.Rc, work.Rc, nNo);
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

      std::vector<const double*> mps, us, ps;
      std::vector<double> cR, cU, cP;
      for (int j = 0; j < 2 * RI.itr; ++j) {
        mps.push_back(MPi(j));
        cR.push_back(xB[j]);
      }
      if (!mps.empty()) {
        gpu->lin_comb(work.Rci, -1.0, cR.data(), mps, work.Rc, nNo);
      } else {
        gpu->copy(work.Rc, work.Rci, nNo);
      }
      double rc2 = gpu->dot(work.Rc, work.Rc, nNo);
      Resc = RI.fNorm > 0.0 ? (int)(100.0 * rc2 / RI.fNorm) : 0;
      Resm = RI.fNorm > 0.0 ? 100 - Resc : 0;

      for (int i = 0; i < RI.itr; i++) {
        us.push_back(Ui(i));
        cU.push_back(xB[2 * i + 1]);
        ps.push_back(Pi(i));
        cP.push_back(xB[2 * i]);
      }
      if (!us.empty()) {
        gpu->lin_comb(nullptr, 1.0, cU.data(), us, work.Rmi, (int)n3);
        gpu->lin_comb(nullptr, 1.0, cP.data(), ps, work.Rci, nNo);
      } else {
        gpu->zero(work.Rmi, n3);
        gpu->zero(work.Rci, nNo);
      }
      RI.dB = RI.fNorm > 0.0 && RI.dB > 0.0 ? 5.0 * std::log(RI.fNorm / RI.dB) : 0.0;
      RI.fNorm = std::sqrt(RI.fNorm);
      gpu->merge4(work.Rmi, work.Rci, work.R);
      gpu->download(Rh, work.R, 4 * (size_t)nNo);
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
      stats->time = now() - t0;
      tAll += stats->time; nCall++;
      if (getenv("SVGPU_TIMING")) printf("SVGPU calls=%d all=%.4f gmres=%.4f cg=%.4f depart=%.4f (per CG it %.1f us)\n", nCall, tAll/nCall, tGM/nCall, tCG/nCall, tDep/nCall, 1e6*tCG/std::max(1,nCGit));
    }
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "svgpu_ns_solve: %s\n", e.what());
    return 1;
  }
}
