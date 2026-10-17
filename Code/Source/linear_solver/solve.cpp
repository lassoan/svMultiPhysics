// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "lhs.h"
#include "CmMod.h"
#include "bicgs.h"
#include "cgrad.h"
#include "gmres.h"
#include "ns_solver.h"
#include "precond.h"

#ifdef SV_USE_GPU
#include "svgpu.h"
#include <chrono>
#include <cstdlib>
double svgpu_t_pre = 0.0, svgpu_t_solve = 0.0, svgpu_t_post = 0.0, svgpu_t_assem = 0.0;
// The matrix is assembled on the GPU and stays there; matrix entries assembled
// on the host (face terms) are staged and added to the device matrix
bool svgpu_val_on_device = false;
std::vector<int> svgpu_stage_ptr;
std::vector<double> svgpu_stage_val;
int svgpu_n_solve = 0;
static double svgpu_now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#endif

namespace fsi_linear_solver {

/// @brief In this routine, the appropriate LS algorithm is called and
/// the solution is returned.
/// Modifies: Val, Ri
///
/// Ri(dof,lhs.nNo): Residual
/// Val(dof*dof,lhs.nnz): LHS
///
/// Reproduces 'SUBROUTINE FSILS_SOLVE (lhs, ls, dof, Ri, Val, prec, incL, res)'.
//
void fsils_solve(FSILS_lhsType& lhs, FSILS_lsType& ls, const int dof, Array<double>& Ri, Array<double>& Val, 
    const consts::PreconditionerType prec, const Vector<int>& incL, const Vector<double>& res)
{
  using namespace consts;

  #define n_debug_fsils_solve
  #ifdef debug_fsils_solve
  DebugMsg dmsg(__func__,  lhs.commu.task);
  dmsg.banner();
  #endif

  const int nNo = lhs.nNo;
  const int nnz = lhs.nnz;
  const int nFaces = lhs.nFaces;
#ifdef SV_USE_GPU
  double t_start = svgpu_now();
#endif
  #ifdef debug_fsils_solve
  dmsg << "nNo: " << nNo;
  dmsg << "nnz: " << nnz;
  dmsg << "nFaces: " << nFaces;
  dmsg << "dof: " << dof;
  dmsg << "ls.LS_type: " << ls.LS_type;
  #endif

  if (lhs.nFaces != 0) {
    for (auto& face : lhs.face) { 
      face.incFlag = true;
    }

    if (incL.size() != 0) {
      for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
        #ifdef debug_fsils_solve
        dmsg << "incL[" << faIn << "]: " << incL[faIn];
        #endif
        if (incL(faIn) == 0) {
          lhs.face[faIn].incFlag = false;
        }
      }
    }

    bool flag = false;
    for (auto& face : lhs.face) { 
      if (face.bGrp == BcType::BC_TYPE_Neu) {
        flag = true;
        break;
      }
    }

    if (res.size() == 0 && flag) {
      throw std::runtime_error("[fsils_solve] res is required for Neu surfaces");
    }

    for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
      auto& face = lhs.face[faIn];
      face.coupledFlag = false;
      if (!face.incFlag) {
        continue;
      }
      bool flag = (face.bGrp == BcType::BC_TYPE_Neu);
      if (flag && res(faIn) != 0.0) {
        face.res = res(faIn);
        face.coupledFlag = true;
      }
    }
  }

  Array<double> R(dof,nNo), Wr(dof,nNo), Wc(dof,nNo);

  for (int a = 0; a < nNo; a++) {
    for (int i = 0; i < dof; i++) {
      R(i,lhs.map(a)) = Ri(i,a);
    }
  }

  // Apply preconditioner.
  //
  // Modifies Val and R.
  //

#ifdef SV_USE_GPU
  if (svgpu_val_on_device) {
    if (prec != PreconditionerType::PREC_FSILS || dof != 4) {
      throw std::runtime_error("svgpu: the device resident matrix requires the FSILS preconditioner");
    }
    if (svgpu_add_val((int)svgpu_stage_ptr.size(), svgpu_stage_ptr.data(), svgpu_stage_val.data()) != 0) {
      throw std::runtime_error("svgpu_add_val failed");
    }
    svgpu_stage_ptr.clear();
    svgpu_stage_val.clear();
    // Dirichlet factors of precond_diag
    Array<double> Dmul(dof, nNo);
    Dmul = 1.0;
    for (auto& face : lhs.face) {
      if (!face.incFlag || face.bGrp != BcType::BC_TYPE_Dir) continue;
      int n = std::min(face.dof, dof);
      for (int a = 0; a < face.nNo; a++) {
        for (int i = 0; i < n; i++) Dmul(i, face.glob(a)) *= face.val(i, a);
      }
    }
    if (svgpu_precond_diag(nNo, nnz, lhs.rowPtr.data(), lhs.colPtr.data(), lhs.diagPtr.data(), Dmul.data(),
        Wc.data()) != 0) {
      throw std::runtime_error("svgpu_precond_diag failed");
    }
    for (int i = 0; i < R.size(); i++) R(i) = Wc(i) * R(i);
    for (auto& face : lhs.face) {
      if (!face.coupledFlag) continue;
      if (face.has_cap) throw std::runtime_error("svgpu: faces with caps are not supported");
      for (int a = 0; a < face.nNo; a++) {
        for (int i = 0; i < std::min(face.dof, dof); i++) face.valM(i, a) = face.val(i, a) * Wc(i, face.glob(a));
      }
    }
  } else
#endif
  if (prec == PreconditionerType::PREC_FSILS) {
    precond::precond_diag(lhs, lhs.rowPtr, lhs.colPtr, lhs.diagPtr, dof, Val, R, Wc);
  } else if (prec == PreconditionerType::PREC_RCS) {
    precond::precond_rcs(lhs, lhs.rowPtr, lhs.colPtr, lhs.diagPtr, dof, Val, R, Wr, Wc);
  } else {
    //PRINT *, "This linear solver and preconditioner combination is not supported."
  }

  // Solve for 'R'.
  //
#ifdef SV_USE_GPU
  double t_pre = svgpu_now();
  svgpu_t_pre += t_pre - t_start;
#endif
  switch (ls.LS_type) {
    case LinearSolverType::LS_TYPE_NS: {
#ifdef SV_USE_GPU
      if (getenv("SV_GPU") && lhs.commu.nTasks == 1) {
        // GPU NS solver (experimental)
        std::vector<svgpu_face> faces;
        for (auto& face : lhs.face) {
          if (face.coupledFlag) {
            if (face.has_cap) throw std::runtime_error("svgpu: faces with caps are not supported");
            faces.push_back({face.nNo, face.glob.data(), face.valM.data(), face.res});
          }
        }
        auto sub = [](const FSILS_subLsType& s) { return svgpu_sub_params{s.mItr, s.sD, s.absTol, s.relTol}; };
        svgpu_params params{sub(ls.RI), sub(ls.GM), sub(ls.CG)};
        svgpu_stats st{};
        if (svgpu_ns_solve(nNo, nnz, dof, lhs.rowPtr.data(), lhs.colPtr.data(),
            svgpu_val_on_device ? nullptr : Val.data(), R.data(),
            (int)faces.size(), faces.data(), &params, &st) != 0) {
          throw std::runtime_error("svgpu_ns_solve failed");
        }
        ls.RI.itr = st.RI_itr; ls.GM.itr = st.GM_itr; ls.CG.itr = st.CG_itr;
        ls.RI.success = st.success != 0; ls.RI.iNorm = st.iNorm; ls.RI.fNorm = st.fNorm;
        ls.RI.dB = st.dB; ls.Resm = st.Resm; ls.Resc = st.Resc; ls.RI.callD = st.time;
        if (getenv("SV_LS_STATS")) {
          printf("LSSTATS-GPU RI.itr=%d GM.itr=%d CG.itr=%d time=%g\n", st.RI_itr, st.GM_itr, st.CG_itr, st.time);
        }
        break;
      }
#endif
      ns_solver::ns_solver(lhs, ls, dof, Val, R);
    } break;

    case LinearSolverType::LS_TYPE_GMRES:
      if (dof == 1) {
        auto Valv = Val.row(0);
        auto Rv = R.row(0);
        gmres::gmres_s(lhs, ls.RI, dof, Valv, Rv);
        Val.set_row(0,Valv);
        R.set_row(0,Rv);
      } else {
        gmres::gmres_v(lhs, ls.RI, dof, Val, R);
      }
    break;

    case LinearSolverType::LS_TYPE_CG:
      if (dof == 1) {
        auto Valv = Val.row(0);
        auto Rv = R.row(0);
        cgrad::cgrad_s(lhs, ls.RI, Valv, Rv);
        Val.set_row(0,Valv);
        R.set_row(0,Rv);
      } else {
        cgrad::cgrad_v(lhs, ls.RI, dof, Val, R);
      }
    break;

    case LinearSolverType::LS_TYPE_BICGS:
      if (dof == 1) {
        auto Valv = Val.row(0);
        auto Rv = R.row(0);
        bicgs::bicgss(lhs, ls.RI, Valv, Rv);
        Val.set_row(0,Valv);
        R.set_row(0,Rv);
      } else {
        bicgs::bicgsv(lhs, ls.RI, dof, Val, R);
      }
    break;

    default:
      throw std::runtime_error("FSILS: LS_type not defined");
  }

  // Element-wise multiplication.
  //
#ifdef SV_USE_GPU
  double t_solved = svgpu_now();
  svgpu_t_solve += t_solved - t_pre;
#endif
  for (int i = 0; i < Wc.size(); i++) {
    R(i) = Wc(i) * R(i);
  }

  for (int a = 0; a < nNo; a++) {
    for (int i = 0; i < R.nrows(); i++) {
      Ri(i,a) = R(i,lhs.map(a));
    }
  }
#ifdef SV_USE_GPU
  svgpu_t_post += svgpu_now() - t_solved;
  if (getenv("SV_GPU_TIMING") && ++svgpu_n_solve % 10 == 0) {
    printf("SV_GPU_TIMING per solve: assembly %.4f precond+copy %.4f solve %.4f post %.4f\n", svgpu_t_assem / svgpu_n_solve,
        svgpu_t_pre / svgpu_n_solve, svgpu_t_solve / svgpu_n_solve, svgpu_t_post / svgpu_n_solve);
  }
#endif
}

};
