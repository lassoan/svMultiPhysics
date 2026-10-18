// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// Host side of the experimental GPU assembly of FSI (construct_fsi) and mesh
// motion (construct_mesh) equations (Code/Source/gpu/svgpu_fsi.cu).

#ifdef SV_USE_GPU

#include "svgpu_fsi_host.h"

#include "all_fun.h"
#include "consts.h"
#include "svgpu.h"

#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>

extern bool svgpu_val_on_device;

namespace svgpu_host {

namespace {

struct MeshInfo {
  int id = -1;            // svgpu mesh index
  int cDmn = -1;          // domain of all elements, -1 if mixed
  bool checked = false;
};

std::map<const mshType*, MeshInfo> meshes;
bool setup = false;

// Domain of all elements of the mesh for equation cEq, or -1 if mixed
int mesh_domain(const ComMod& com_mod, const mshType& lM, int cEq)
{
  int d0 = all_fun::domain(com_mod, lM, cEq, 0);
  for (int e = 1; e < lM.nEl; e++) {
    if (all_fun::domain(com_mod, lM, cEq, e) != d0) return -1;
  }
  return d0;
}

bool common_ok(const ComMod& com_mod, const mshType& lM)
{
  const auto& fs0 = lM.fs[0];
  return getenv("SV_GPU") && getenv("SV_GPU_ASSEM") && com_mod.cm.np() == 1 && com_mod.nsd == 3 &&
      lM.eNoN == 4 && lM.nFs == 1 && fs0.lShpF && fs0.nG <= 4 && !com_mod.risFlag && !com_mod.urisFlag &&
      com_mod.tDof <= 8;
}

MeshInfo& info(ComMod& com_mod, const mshType& lM, int cEq)
{
  if (!setup) {
    if (svgpu_fsi_setup(com_mod.tnNo, com_mod.x.data(), com_mod.colPtr.size(), com_mod.rowPtr.data(),
        com_mod.colPtr.data()) != 0) {
      throw std::runtime_error("svgpu_fsi_setup failed");
    }
    setup = true;
  }
  auto& mi = meshes[&lM];
  if (mi.id < 0) {
    const auto& fs0 = lM.fs[0];
    mi.id = svgpu_fsi_add_mesh(lM.nEl, lM.IEN.data(), fs0.nG, fs0.w.data(), fs0.N.data(), fs0.Nx.data());
    if (mi.id < 0) throw std::runtime_error("svgpu_fsi_add_mesh failed");
  }
  return mi;
}

}  // namespace

bool construct_fsi(ComMod& com_mod, const mshType& lM, const SolutionStates& solutions)
{
  using namespace consts;
  const int cEq = com_mod.cEq;
  const auto& eq = com_mod.eq[cEq];
  int cDmn = mesh_domain(com_mod, lM, cEq);
  static std::map<const mshType*, bool> reported;
  if (getenv("SV_GPU_VERBOSE") && !reported[&lM]) {
    reported[&lM] = true;
    printf("svgpu: mesh %s: common %d dof %d domain %d phys %d nFs %d lShpF %d nG %d eNoN %d tDof %d\n",
        lM.name.c_str(), (int)common_ok(com_mod, lM), com_mod.dof, cDmn,
        cDmn >= 0 ? (int)eq.dmn[cDmn].phys : -1, lM.nFs, (int)lM.fs[0].lShpF, lM.fs[0].nG, lM.eNoN, com_mod.tDof);
  }
  static int nFalse = 0, nTrue = 0;
  auto no = [&](const char* why) {
    if (getenv("SV_GPU_VERBOSE") && nFalse++ < 20) printf("svgpu: %s on the CPU: %s\n", lM.name.c_str(), why);
    return false;
  };
  if (!common_ok(com_mod, lM) || com_mod.dof != 4) return no("common");
  if (cDmn < 0) return no("domain");
  const auto& dmn = eq.dmn[cDmn];
  double prm[12] = {};
  int type;
  if (dmn.phys == EquationType::phys_fluid) {
    if (dmn.fluid_visc.viscType != FluidViscosityModelType::viscType_Const || !com_mod.mvMsh) return no("fluid");
    auto prop = dmn.prop;
    double v[9] = {prop[PhysicalPropertyType::fluid_density], dmn.fluid_visc.mu_i, prop[PhysicalPropertyType::f_x],
        prop[PhysicalPropertyType::f_y], prop[PhysicalPropertyType::f_z], com_mod.dt, eq.af, eq.am, eq.gam};
    std::copy(v, v + 9, prm);
    type = 0;
  } else if (dmn.phys == EquationType::phys_struct) {
    const auto& stM = dmn.stM;
    static bool reported = false;
    auto reject = [&](const char* why) {
      if (!reported && getenv("SV_GPU_VERBOSE")) printf("svgpu: struct domain on the CPU: %s\n", why);
      reported = true;
      return false;
    };
    if (stM.isoType != ConstitutiveModelType::stIso_nHook) return reject("material model");
    if (stM.volType != ConstitutiveModelType::stVol_M94) return reject("volumetric model");
    if (dmn.solid_visc.viscType != SolidViscosityModelType::viscType_NA && dmn.solid_visc.mu != 0.0) {
      return reject("solid viscosity");
    }
    if (dmn.active_stress != nullptr) return reject("active stress");
    if (com_mod.pS0.size() != 0) return reject("prestress");
    if (eq.s != 0) return reject("equation dof offset");
    const auto& prop = dmn.prop;
    double v[12] = {prop.at(PhysicalPropertyType::solid_density), prop.at(PhysicalPropertyType::damping),
        prop.at(PhysicalPropertyType::f_x), prop.at(PhysicalPropertyType::f_y), prop.at(PhysicalPropertyType::f_z),
        stM.C10, stM.Kpen, com_mod.dt, eq.af, eq.am, eq.beta, eq.gam};
    std::copy(v, v + 12, prm);
    type = 1;
  } else {
    return false;
  }
  auto& mi = info(com_mod, lM, cEq);
  com_mod.cDmn = cDmn;
  const auto& Ag = solutions.intermediate.get_acceleration();
  const auto& Yg = solutions.intermediate.get_velocity();
  const auto& Dg = solutions.intermediate.get_displacement();
  if (svgpu_fsi_assemble(mi.id, type, com_mod.tDof, Ag.data(), Yg.data(), Dg.data(), nullptr, com_mod.Bf.data(),
      prm, com_mod.R.data()) != 0) {
    throw std::runtime_error("svgpu_fsi_assemble failed");
  }
  svgpu_val_on_device = getenv("SV_GPU") != nullptr;
  if (getenv("SV_GPU_VERBOSE") && nTrue++ < 4) printf("svgpu: %s assembled on the GPU (type %d)\n", lM.name.c_str(), type);
  return true;
}

bool construct_mesh(ComMod& com_mod, const mshType& lM, const SolutionStates& solutions)
{
  using namespace consts;
  if (!common_ok(com_mod, lM) || com_mod.dof != 3) return false;
  const int cEq = com_mod.cEq;
  const auto& eq = com_mod.eq[cEq];
  int cDmn = mesh_domain(com_mod, lM, cEq);
  if (cDmn < 0) return false;
  const auto& dmn = eq.dmn[cDmn];
  if (dmn.phys != EquationType::phys_mesh) {
    // No mesh motion elements in this mesh
    return true;
  }
  if (eq.s != com_mod.nsd + 1 || com_mod.pstEq) return false;
  const auto& prop = dmn.prop;
  double prm[10] = {prop.at(PhysicalPropertyType::solid_density), prop.at(PhysicalPropertyType::elasticity_modulus),
      prop.at(PhysicalPropertyType::poisson_ratio), prop.at(PhysicalPropertyType::f_x),
      prop.at(PhysicalPropertyType::f_y), prop.at(PhysicalPropertyType::f_z), com_mod.dt, eq.af, eq.am, eq.beta};
  auto& mi = info(com_mod, lM, cEq);
  com_mod.cDmn = cDmn;
  const auto& Ag = solutions.intermediate.get_acceleration();
  const auto& Yg = solutions.intermediate.get_velocity();
  const auto& Dg = solutions.intermediate.get_displacement();
  const auto& Do = solutions.old.get_displacement();
  if (svgpu_fsi_assemble(mi.id, 2, com_mod.tDof, Ag.data(), Yg.data(), Dg.data(), Do.data(), com_mod.Bf.data(),
      prm, com_mod.R.data()) != 0) {
    throw std::runtime_error("svgpu_fsi_assemble failed");
  }
  svgpu_val_on_device = getenv("SV_GPU") != nullptr;
  return true;
}

}  // namespace svgpu_host

#endif
