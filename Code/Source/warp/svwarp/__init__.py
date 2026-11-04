# SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
# SPDX-License-Identifier: BSD-3-Clause

"""GPU acceleration of svMultiPhysics with NVIDIA Warp (experimental): the
assembly of the fluid equation and the FSILS NS linear solver run on the GPU.

The functions of this module are called from solver/svwarp_host.cpp through
the embedded Python interpreter. Arrays are passed as host pointers (integers)
with the layout of the svMultiPhysics Array class (column-major, i.e. the
first index is contiguous)."""

import ctypes
import os
import time

import numpy as np
import warp as wp

from .common import DEVICE, F, host_array, host_numpy, mat34, mat44, to_device, vec4
from . import bls
from . import fluid as fl
from . import fsi as fsimod
from . import linalg as la
from . import ns as nsmod

_state = {}
_timers = {}


def _timed(name, t0):
    t = _timers.setdefault(name, [0.0, 0])
    t[0] += time.perf_counter() - t0
    t[1] += 1
    if os.environ.get("SVWARP_TIMING") and t[1] % 10 == 0:
        print("SVWARP api per call: " + " ".join(f"{k}={v[0] / max(v[1], 1):.4f}" for k, v in _timers.items()), flush=True)


def init():
    wp.config.enable_backward = False
    wp.config.quiet = True
    if os.environ.get("SVWARP_DEBUG"):
        wp.config.verify_cuda = True
    wp.init()
    if os.environ.get("SV_GPU_VERBOSE"):
        print(f"svwarp: Warp {wp.config.version}, device {wp.get_device(DEVICE).name}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Assembly of the fluid equation

def fluid_setup(nEl, IEN, tnNo, x, nnz, rowPtr, colPtr, nG, w, N, Nxi):
    """Mesh (IEN: 4 x nEl, x: 3 x tnNo), matrix structure (CSR, tnNo+1 and
    nnz) and the Gauss data (w: nG, N: 4 x nG, Nxi: 3 x 4)."""
    t0 = time.perf_counter()
    s = _state
    s["nEl"], s["tnNo"], s["nnz"] = int(nEl), int(tnNo), int(nnz)
    s["IEN"] = to_device(IEN, wp.int32, (nEl, 4))
    s["x"] = to_device(x, F, (tnNo, 3))
    offsets = host_numpy(rowPtr, ctypes.c_int, (tnNo + 1,)).copy()
    columns = host_numpy(colPtr, ctypes.c_int, (nnz,)).copy()
    lhs = la.Lhs(tnNo, offsets, columns)
    s["lhs"] = lhs
    s["ePtr"] = wp.zeros((nEl, 4 * 4), dtype=wp.int32, device=DEVICE)
    wp.launch(fl.k_elem_ptr, dim=nEl, inputs=[s["IEN"], lhs.offsets, lhs.columns, s["ePtr"]], device=DEVICE)
    s["nG"] = int(nG)
    wn = host_numpy(w, ctypes.c_double, (nG,))
    Nn = host_numpy(N, ctypes.c_double, (nG, 4))     # N(a,g) at a + 4g
    Nxin = host_numpy(Nxi, ctypes.c_double, (4, 3))  # Nxi(i,a) at i + 3a
    gw = [0.0] * 4
    for g in range(nG):
        gw[g] = float(wn[g])
    s["gw"] = vec4(*gw)
    gN = mat44()
    for g in range(nG):
        for a in range(4):
            gN[g, a] = float(Nn[g, a])
    s["gN"] = gN
    nxi = mat34()
    for a in range(4):
        for i in range(3):
            nxi[i, a] = float(Nxin[a, i])
    s["Nxi"] = nxi
    s["R"] = wp.zeros((tnNo, 4), dtype=F, device=DEVICE)
    s["Rh"] = wp.zeros((tnNo, 4), dtype=F, device="cpu", pinned=True)
    s["tDof"] = 0
    s["ns"] = None
    s["t_assem"] = 0.0
    s["n_assem"] = 0
    wp.synchronize_device(DEVICE)
    if os.environ.get("SV_GPU_VERBOSE"):
        print(f"svwarp: fluid setup nEl {nEl} tnNo {tnNo} nnz {nnz} nG {nG} in {time.perf_counter() - t0:.2f} s", flush=True)
    return 0


def fluid_assemble(tDof, Ag, Yg, Bf, rho, mu, fx, fy, fz, dt, af, am, gam, brink, R):
    """Assemble the fluid equation into the device matrix (cleared first) and
    add the residual to R (host, 4 x tnNo)."""
    t0 = time.perf_counter()
    s = _state
    tnNo = s["tnNo"]
    tDof = int(tDof)
    if s["tDof"] != tDof:
        # Device state and pinned staging buffers (copies from pinned memory
        # are asynchronous; from pageable memory they cost a staging copy)
        for name, m in (("Ag", tDof), ("Yg", tDof), ("Bf", 3)):
            s[name] = wp.zeros((tnNo, m), dtype=F, device=DEVICE)
            s[name + "h"] = wp.zeros((tnNo, m), dtype=F, device="cpu", pinned=True)
            s[name + "n"] = s[name + "h"].numpy()
        s["tDof"] = tDof
    for name, ptr, m in (("Ag", Ag, tDof), ("Yg", Yg, tDof), ("Bf", Bf, 3)):
        s[name + "n"][:] = host_numpy(ptr, ctypes.c_double, (tnNo, m))
        wp.copy(s[name], s[name + "h"])
    prm = fl.FluidParams()
    prm.rho, prm.mu = F(rho), F(mu)
    prm.f = wp.vec3d(F(fx), F(fy), F(fz))
    prm.dt, prm.af, prm.am, prm.gam, prm.brink = F(dt), F(af), F(am), F(gam), F(brink)
    prm.mvMsh = 0
    lhs = s["lhs"]
    lhs.Val.zero_()
    s["R"].zero_()
    wp.launch(fl.k_fluid, dim=s["nEl"],
              inputs=[s["IEN"], s["x"], s["Ag"], s["Yg"], s["Bf"], prm, s["nG"], s["gw"], s["gN"], s["Nxi"], s["ePtr"],
                      s["R"], lhs.Val], device=DEVICE)
    wp.copy(s["Rh"], s["R"])
    wp.synchronize_device(DEVICE)
    Rn = host_numpy(R, ctypes.c_double, (tnNo, 4))
    Rn += s["Rh"].numpy()
    _timed("assemble", t0)
    return 0


def add_vals(n, ptr, vals):
    """Add n staged 4x4 blocks (vals: 16 x n, host) at the block indices ptr (host)."""
    t0 = time.perf_counter()
    n = int(n)
    if n:
        _state["lhs"].add_vals(host_array(ptr, wp.int32, (n,)), host_array(vals, mat44, (n,)), n)
    _timed("add_vals", t0)
    return 0


def precond_diag(Dmul, W):
    """FSILS diagonal preconditioner of the device matrix; Dmul, W: 4 x nNo (host)."""
    t0 = time.perf_counter()
    tnNo = _state["tnNo"]
    _state["lhs"].precond_diag(host_array(Dmul, F, (tnNo, 4)), host_array(W, F, (tnNo, 4)))
    wp.synchronize_device(DEVICE)
    _timed("precond", t0)
    return 0


# ---------------------------------------------------------------------------
# FSI: assembly of the FSI and mesh motion equations, GMRES and CG for
# dof = 3 and 4 (fsi.py, bls.py)

def fsi_setup(tnNo, x, nnz, rowPtr, colPtr):
    """Node coordinates (3 x tnNo) and the matrix structure (CSR)."""
    _state["fsi"] = fsimod.Fsi(int(tnNo), x, int(nnz), rowPtr, colPtr)
    return 0


def fsi_add_mesh(nEl, IEN, nG, w, N, Nxi):
    """Add a mesh (IEN: 4 x nEl, Gauss data as fluid_setup); returns its index."""
    return _state["fsi"].add_mesh(int(nEl), IEN, int(nG), w, N, Nxi)


def fsi_assemble(mesh, type_, tDof, Ag, Yg, Dg, Do, Bf, params, R):
    """Assemble one mesh into the device matrix (added) and add the residual
    to R (host, dof x tnNo). type: 0 = FSI fluid, 1 = FSI struct, 2 = mesh
    motion; params: 12 doubles (see fsi.Fsi.assemble)."""
    _state["fsi"].assemble(int(mesh), int(type_), int(tDof), Ag, Yg, Dg, Do, Bf, params, R)
    return 0


def add_vals_dof(n, dof, ptr, vals):
    """Add n staged dof x dof blocks (host) at the block indices ptr (host)."""
    t0 = time.perf_counter()
    n, dof = int(n), int(dof)
    if n:
        f = _state["fsi"]
        K = bls.kernels(dof)
        if n > f.stage_cap:
            f.stage_cap = 2 * n
            f.stage_ptr = wp.empty(f.stage_cap, dtype=wp.int32, device=DEVICE)
            f.stage_val = {d: wp.empty(f.stage_cap, dtype=bls.kernels(d)["MAT"], device=DEVICE) for d in (3, 4)}
        wp.copy(f.stage_ptr, host_array(ptr, wp.int32, (n,)), count=n)
        wp.copy(f.stage_val[dof], host_array(vals, K["MAT"], (n,)), count=n)
        wp.launch(K["k_add_vals"], dim=n, inputs=[f.stage_ptr, f.stage_val[dof], f.val(dof)], device=DEVICE)
    _timed("add_vals", t0)
    return 0


def precond_diag_dof(dof, Dmul, W):
    """FSILS diagonal preconditioner of the device matrix for dof = 3 or 4;
    Dmul, W: dof x nNo (host)."""
    t0 = time.perf_counter()
    dof = int(dof)
    f = _state["fsi"]
    K = bls.kernels(dof)
    tnNo = f.tnNo
    if not hasattr(f, "Dmul"):
        f.Dmul = {d: wp.zeros((tnNo, d), dtype=F, device=DEVICE) for d in (3, 4)}
        f.W = {d: wp.zeros((tnNo, d), dtype=F, device=DEVICE) for d in (3, 4)}
    wp.copy(f.Dmul[dof], host_array(Dmul, F, (tnNo, dof)))
    wp.launch(K["k_precond_w"], dim=tnNo, inputs=[f.lhs.diag, f.val(dof), f.Dmul[dof], f.W[dof]], device=DEVICE)
    wp.launch(K["k_precond_scale"], dim=tnNo, inputs=[f.lhs.offsets, f.lhs.columns, f.W[dof], f.val(dof)],
              device=DEVICE)
    wp.copy(host_array(W, F, (tnNo, dof)), f.W[dof])
    wp.synchronize_device(DEVICE)
    _timed("precond", t0)
    return 0


def ls_solve(type_, dof, mItr, sD, absTol, relTol, R):
    """FSILS GMRES (type 0) or CG (type 1) on the device matrix. R: dof x nNo
    (host) right-hand side in, solution out. Returns (itr, success, iNorm,
    fNorm, dB, time)."""
    dof = int(dof)
    f = _state["fsi"]
    if dof not in f.solvers:
        f.solvers[dof] = bls.LsSolver(dof, f.lhs, f.sell, f.val(dof))
    st = f.solvers[dof].solve(int(type_), int(mItr), int(sD), float(absTol), float(relTol),
                              host_array(R, F, (dof * f.tnNo,)))
    return tuple(float(v) for v in st)


def ns_solve(R, nFaces, glob, off, valM, res, RI_mItr, RI_sD, RI_absTol, RI_relTol, GM_mItr, GM_sD, GM_absTol,
             GM_relTol, CG_mItr, CG_sD, CG_absTol, CG_relTol):
    """FSILS NS solver on the device matrix. R: 4 x nNo (host) right-hand
    side in, solution out. Faces: concatenated node indices glob (int) with
    offsets off (nFaces+1), face vectors valM (3 per node) and resistances
    res (nFaces). Returns the statistics tuple (see NsSolver.solve)."""
    s = _state
    tnNo = s["tnNo"]
    if s["ns"] is None:
        s["ns"] = nsmod.NsSolver(s["lhs"])
    nFaces = int(nFaces)
    offn = host_numpy(off, ctypes.c_int, (nFaces + 1,)) if nFaces else np.zeros(1, dtype=np.int32)
    total = int(offn[-1]) if nFaces else 0
    faces = (nFaces, host_numpy(glob, ctypes.c_int, (total,)) if total else np.zeros(0, np.int32), offn,
             host_numpy(valM, ctypes.c_double, (3 * total,)) if total else np.zeros(0),
             host_numpy(res, ctypes.c_double, (nFaces,)) if nFaces else np.zeros(0))
    RI = nsmod.SubLs(RI_mItr, RI_sD, RI_absTol, RI_relTol)
    GM = nsmod.SubLs(GM_mItr, GM_sD, GM_absTol, GM_relTol)
    CG = nsmod.SubLs(CG_mItr, CG_sD, CG_absTol, CG_relTol)
    t0 = time.perf_counter()
    stats = s["ns"].solve(host_array(R, F, (tnNo, 4)), faces, RI, GM, CG)
    _timed("ns_solve", t0)
    return tuple(float(v) for v in stats)
