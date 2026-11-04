// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifdef SV_USE_MFEM

#include "svmfem_host.h"

#include "consts.h"
#include "svmfem.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace svmfem_host {

namespace {

bool on_device = false;
std::vector<int> stage_ptr;
std::vector<double> stage_val;

void check(int status, const char* what)
{
  if (status != 0) throw std::runtime_error(std::string("svmfem: ") + what + " failed");
}

}  // namespace

bool val_on_device()
{
  return on_device;
}

void stage(int ptr, const double* vals)
{
  stage_ptr.push_back(ptr);
  stage_val.insert(stage_val.end(), vals, vals + 16);
}

bool construct_fluid(ComMod& com_mod, const mshType& lM, const Array<double>& Ag, const Array<double>& Yg)
{
  using namespace consts;
  const auto& eq = com_mod.eq[com_mod.cEq];
  const auto& fs0 = lM.fs[0];
  const bool supported = getenv("SV_MFEM") && getenv("SV_MFEM_ASSEM") && com_mod.cm.np() == 1 &&
      com_mod.nsd == 3 && lM.eNoN == 4 && lM.nFs == 1 && fs0.lShpF && fs0.nG <= 4 && eq.nDmn == 1 &&
      eq.dmn[0].phys == EquationType::phys_fluid &&
      eq.dmn[0].fluid_visc.viscType == FluidViscosityModelType::viscType_Const && !com_mod.mvMsh &&
      !com_mod.risFlag && !com_mod.urisFlag && com_mod.dof == 4 && com_mod.nMsh == 1;
  if (!supported) return false;
  static bool setup = false;
  if (!setup) {
    check(svmfem_init(nullptr), "init");
    check(svmfem_fluid_setup(lM.nEl, lM.IEN.data(), com_mod.tnNo, com_mod.x.data(), (int)com_mod.colPtr.size(),
        com_mod.rowPtr.data(), com_mod.colPtr.data(), fs0.nG, fs0.w.data(), fs0.N.data(), fs0.Nx.data()),
        "fluid_setup");
    setup = true;
  }
  const auto& dmn = eq.dmn[0];
  com_mod.cDmn = 0;
  auto prop = dmn.prop;
  on_device = true;
  const double f[3] = {prop[PhysicalPropertyType::f_x], prop[PhysicalPropertyType::f_y],
      prop[PhysicalPropertyType::f_z]};
  check(svmfem_fluid_assemble(com_mod.tDof, Ag.data(), Yg.data(), com_mod.Bf.data(),
      prop[PhysicalPropertyType::fluid_density], dmn.fluid_visc.mu_i, f, com_mod.dt, eq.af, eq.am, eq.gam,
      prop.at(PhysicalPropertyType::brinkman_inverse_permeability), com_mod.R.data()), "fluid_assemble");
  return true;
}

void fsils_solve(fsi_linear_solver::FSILS_lhsType& lhs, fsi_linear_solver::FSILS_lsType& ls, int dof,
    Array<double>& R, Array<double>& Wc)
{
  using namespace fsi_linear_solver;
  const int nNo = lhs.nNo;
  if (dof != 4) throw std::runtime_error("svmfem: the NS solver requires dof = 4");
  // Face terms assembled on the host
  check(svmfem_add_val((int)stage_ptr.size(), stage_ptr.data(), stage_val.data()), "add_val");
  stage_ptr.clear();
  stage_val.clear();
  // Preconditioner: Dirichlet factors of precond_diag, W from the device
  Array<double> Dmul(dof, nNo);
  Dmul = 1.0;
  for (auto& face : lhs.face) {
    if (!face.incFlag || face.bGrp != BcType::BC_TYPE_Dir) continue;
    int n = std::min(face.dof, dof);
    for (int a = 0; a < face.nNo; a++) {
      for (int i = 0; i < n; i++) Dmul(i, face.glob(a)) *= face.val(i, a);
    }
  }
  check(svmfem_precond_diag(Dmul.data(), Wc.data()), "precond_diag");
  for (int i = 0; i < R.size(); i++) R(i) = Wc(i) * R(i);
  // Coupled (resistance) faces
  std::vector<svmfem_face> faces;
  for (auto& face : lhs.face) {
    if (!face.coupledFlag) continue;
    if (face.has_cap) throw std::runtime_error("svmfem: faces with caps are not supported");
    for (int a = 0; a < face.nNo; a++) {
      for (int i = 0; i < 3; i++) face.valM(i, a) = face.val(i, a) * Wc(i, face.glob(a));
    }
    faces.push_back({face.nNo, face.glob.data(), face.valM.data(), face.res});
  }
  svmfem_params p{{ls.RI.mItr, ls.RI.sD, ls.RI.absTol, ls.RI.relTol}, {ls.GM.mItr, ls.GM.sD, ls.GM.absTol, ls.GM.relTol},
      {ls.CG.mItr, ls.CG.sD, ls.CG.absTol, ls.CG.relTol}};
  svmfem_stats st{};
  check(svmfem_ns_solve(R.data(), (int)faces.size(), faces.data(), &p, &st), "ns_solve");
  ls.RI.itr = st.RI_itr;
  ls.GM.itr = st.GM_itr;
  ls.CG.itr = st.CG_itr;
  ls.RI.success = st.success != 0;
  ls.RI.iNorm = st.iNorm;
  ls.RI.fNorm = st.fNorm;
  ls.RI.dB = st.dB;
  ls.Resm = st.Resm;
  ls.Resc = st.Resc;
  ls.RI.callD = st.time;
  if (getenv("SV_LS_STATS")) {
    printf("LSSTATS-MFEM RI.itr=%d GM.itr=%d CG.itr=%d time=%g\n", ls.RI.itr, ls.GM.itr, ls.CG.itr, st.time);
  }
}

void finalize()
{
  if (on_device) svmfem_finalize();
}

}  // namespace svmfem_host

#endif
