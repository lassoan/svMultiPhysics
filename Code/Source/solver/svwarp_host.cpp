// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifdef SV_USE_WARP

#include "svwarp_host.h"

#include "consts.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace svwarp_host {

namespace {

PyObject* module = nullptr;
bool on_device = false;
std::vector<int> stage_ptr;
std::vector<double> stage_val;

std::string env_or(const char* name, const char* def)
{
  const char* v = getenv(name);
  return v ? std::string(v) : std::string(def);
}

void fail(const char* what)
{
  if (PyErr_Occurred()) PyErr_Print();
  throw std::runtime_error(std::string("svwarp: ") + what + " failed");
}

/// Start the interpreter and import svwarp. The Python installation and the
/// site-packages with warp are set at configure time (SV_WARP_PYTHON_HOME,
/// SV_WARP_SITE_PACKAGES, SV_WARP_SOURCE_DIR) and can be overridden with
/// environment variables of the same names.
void init()
{
  if (module) return;
  PyConfig config;
  PyConfig_InitIsolatedConfig(&config);
  std::string home = env_or("SV_WARP_PYTHON_HOME", SV_WARP_PYTHON_HOME);
  PyConfig_SetBytesString(&config, &config.home, home.c_str());
  config.site_import = 1;
  PyStatus status = Py_InitializeFromConfig(&config);
  PyConfig_Clear(&config);
  if (PyStatus_Exception(status)) throw std::runtime_error("svwarp: Py_InitializeFromConfig failed");
  std::string site = env_or("SV_WARP_SITE_PACKAGES", SV_WARP_SITE_PACKAGES);
  std::string src = env_or("SV_WARP_SOURCE_DIR", SV_WARP_SOURCE_DIR);
  PyObject* path = PySys_GetObject("path");
  for (const std::string& p : {site, src}) {
    PyObject* s = PyUnicode_FromString(p.c_str());
    PyList_Insert(path, 0, s);
    Py_DECREF(s);
  }
  module = PyImport_ImportModule("svwarp");
  if (!module) fail("import svwarp");
  PyObject* r = PyObject_CallMethod(module, "init", nullptr);
  if (!r) fail("svwarp.init");
  Py_DECREF(r);
}

PyObject* arg(long long v) { return PyLong_FromLongLong(v); }
PyObject* arg(int v) { return PyLong_FromLong(v); }
PyObject* arg(double v) { return PyFloat_FromDouble(v); }
template <typename T> PyObject* arg(const T* p) { return PyLong_FromUnsignedLongLong((unsigned long long)(uintptr_t)p); }
template <typename T> PyObject* arg(T* p) { return PyLong_FromUnsignedLongLong((unsigned long long)(uintptr_t)p); }

/// Call svwarp.<name>(args...) and return the result (new reference)
template <typename... A>
PyObject* call(const char* name, A... a)
{
  init();
  PyObject* fn = PyObject_GetAttrString(module, name);
  if (!fn) fail(name);
  PyObject* args = PyTuple_New(sizeof...(A));
  PyObject* items[] = {arg(a)..., nullptr};
  for (size_t i = 0; i < sizeof...(A); i++) PyTuple_SetItem(args, i, items[i]);
  PyObject* r = PyObject_CallObject(fn, args);
  Py_DECREF(args);
  Py_DECREF(fn);
  if (!r) fail(name);
  return r;
}

template <typename... A>
void call0(const char* name, A... a)
{
  Py_DECREF(call(name, a...));
}

}  // namespace

bool val_on_device()
{
  return on_device;
}

void set_on_device(bool on)
{
  on_device = on;
}

void stage(int ptr, const double* vals, int n)
{
  stage_ptr.push_back(ptr);
  stage_val.insert(stage_val.end(), vals, vals + n);
}

void fsi_setup(int tnNo, const double* x, int nnz, const int* rowPtr, const int* colPtr)
{
  call0("fsi_setup", tnNo, x, nnz, rowPtr, colPtr);
}

int fsi_add_mesh(int nEl, const int* IEN, int nG, const double* w, const double* N, const double* Nxi)
{
  PyObject* r = call("fsi_add_mesh", nEl, IEN, nG, w, N, Nxi);
  int id = (int)PyLong_AsLong(r);
  Py_DECREF(r);
  return id;
}

void fsi_assemble(int mesh, int type, int tDof, const double* Ag, const double* Yg, const double* Dg,
    const double* Do, const double* Bf, const double* params, double* R)
{
  call0("fsi_assemble", mesh, type, tDof, Ag, Yg, Dg, Do, Bf, params, R);
}

bool construct_fluid(ComMod& com_mod, const mshType& lM, const Array<double>& Ag, const Array<double>& Yg)
{
  using namespace consts;
  const auto& eq = com_mod.eq[com_mod.cEq];
  const auto& fs0 = lM.fs[0];
  const bool supported = getenv("SV_WARP_ASSEM") && com_mod.cm.np() == 1 && com_mod.nsd == 3 && lM.eNoN == 4 &&
      lM.nFs == 1 && fs0.lShpF && fs0.nG <= 4 && eq.nDmn == 1 && eq.dmn[0].phys == EquationType::phys_fluid &&
      eq.dmn[0].fluid_visc.viscType == FluidViscosityModelType::viscType_Const && !com_mod.mvMsh &&
      !com_mod.risFlag && !com_mod.urisFlag && com_mod.dof == 4 && com_mod.nMsh == 1;
  if (!supported) return false;
  static bool setup = false;
  if (!setup) {
    call0("fluid_setup", lM.nEl, lM.IEN.data(), com_mod.tnNo, com_mod.x.data(), (int)com_mod.colPtr.size(),
        com_mod.rowPtr.data(), com_mod.colPtr.data(), fs0.nG, fs0.w.data(), fs0.N.data(), fs0.Nx.data());
    setup = true;
  }
  const auto& dmn = eq.dmn[0];
  com_mod.cDmn = 0;
  auto prop = dmn.prop;
  on_device = true;
  call0("fluid_assemble", com_mod.tDof, Ag.data(), Yg.data(), com_mod.Bf.data(),
      prop[PhysicalPropertyType::fluid_density], dmn.fluid_visc.mu_i, prop[PhysicalPropertyType::f_x],
      prop[PhysicalPropertyType::f_y], prop[PhysicalPropertyType::f_z], com_mod.dt, eq.af, eq.am, eq.gam,
      prop.at(PhysicalPropertyType::brinkman_inverse_permeability), com_mod.R.data());
  return true;
}

void fsils_solve(fsi_linear_solver::FSILS_lhsType& lhs, fsi_linear_solver::FSILS_lsType& ls, int dof,
    Array<double>& R, Array<double>& Wc)
{
  using namespace fsi_linear_solver;
  const int nNo = lhs.nNo;
  const bool ns = ls.LS_type == LinearSolverType::LS_TYPE_NS;
  if (ns && dof != 4) throw std::runtime_error("svwarp: the NS solver requires dof = 4");
  if (!ns && dof != 3 && dof != 4) throw std::runtime_error("svwarp: GMRES and CG require dof = 3 or 4");
  // Face terms assembled on the host
  if (ns) {
    call0("add_vals", (int)stage_ptr.size(), stage_ptr.data(), stage_val.data());
  } else {
    call0("add_vals_dof", (int)stage_ptr.size(), dof, stage_ptr.data(), stage_val.data());
  }
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
  if (ns) {
    call0("precond_diag", Dmul.data(), Wc.data());
  } else {
    call0("precond_diag_dof", dof, Dmul.data(), Wc.data());
  }
  for (int i = 0; i < R.size(); i++) R(i) = Wc(i) * R(i);
  if (!ns) {
    // GMRES and CG on the device matrix (no coupled faces)
    for (auto& face : lhs.face) {
      if (face.coupledFlag) throw std::runtime_error("svwarp: coupled faces are not supported with GMRES and CG");
    }
    const int type = (ls.LS_type == LinearSolverType::LS_TYPE_GMRES) ? 0 : 1;
    PyObject* st = call("ls_solve", type, dof, ls.RI.mItr, ls.RI.sD, ls.RI.absTol, ls.RI.relTol, R.data());
    double v[6];
    for (int i = 0; i < 6; i++) v[i] = PyFloat_AsDouble(PyTuple_GetItem(st, i));
    Py_DECREF(st);
    ls.RI.itr = (int)v[0];
    ls.RI.success = v[1] != 0.0;
    ls.RI.iNorm = v[2];
    ls.RI.fNorm = v[3];
    ls.RI.dB = v[4];
    ls.RI.callD = v[5];
    if (getenv("SV_LS_STATS")) {
      printf("LSSTATS-WARP type=%d dof=%d itr=%d time=%g" "\n", type, dof, ls.RI.itr, v[5]);
    }
    return;
  }
  // Coupled (resistance) faces
  std::vector<int> glob, off{0};
  std::vector<double> valM, res;
  for (auto& face : lhs.face) {
    if (!face.coupledFlag) continue;
    if (face.has_cap) throw std::runtime_error("svwarp: faces with caps are not supported");
    for (int a = 0; a < face.nNo; a++) {
      glob.push_back(face.glob(a));
      for (int i = 0; i < 3; i++) {
        face.valM(i, a) = face.val(i, a) * Wc(i, face.glob(a));
        valM.push_back(face.valM(i, a));
      }
    }
    off.push_back((int)glob.size());
    res.push_back(face.res);
  }
  PyObject* st = call("ns_solve", R.data(), (int)res.size(), glob.data(), off.data(), valM.data(), res.data(),
      ls.RI.mItr, ls.RI.sD, ls.RI.absTol, ls.RI.relTol, ls.GM.mItr, ls.GM.sD, ls.GM.absTol, ls.GM.relTol,
      ls.CG.mItr, ls.CG.sD, ls.CG.absTol, ls.CG.relTol);
  double v[10];
  for (int i = 0; i < 10; i++) v[i] = PyFloat_AsDouble(PyTuple_GetItem(st, i));
  Py_DECREF(st);
  ls.RI.itr = (int)v[0];
  ls.GM.itr = (int)v[1];
  ls.CG.itr = (int)v[2];
  ls.RI.success = v[3] != 0.0;
  ls.RI.iNorm = v[4];
  ls.RI.fNorm = v[5];
  ls.RI.dB = v[6];
  ls.Resm = (int)v[7];
  ls.Resc = (int)v[8];
  ls.RI.callD = v[9];
  if (getenv("SV_LS_STATS")) {
    printf("LSSTATS-WARP RI.itr=%d GM.itr=%d CG.itr=%d time=%g\n", ls.RI.itr, ls.GM.itr, ls.CG.itr, v[9]);
  }
}

}  // namespace svwarp_host

#endif
