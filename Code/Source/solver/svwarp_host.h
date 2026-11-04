// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// Host side of the experimental GPU acceleration with NVIDIA Warp
// (Code/Source/warp/svwarp): an embedded Python interpreter runs the Warp
// kernels of the fluid assembly and of the FSILS NS linear solver. Enabled at
// run time with the environment variable SV_WARP=1 (and SV_WARP_ASSEM=1 for
// the GPU assembly); otherwise the CPU code runs.

#ifndef SVWARP_HOST_H
#define SVWARP_HOST_H

#ifdef SV_USE_WARP

#include "ComMod.h"
#include "fils_struct.hpp"

#include <vector>

namespace svwarp_host {

/// GPU assembly of the fluid equation into the device matrix; false if the
/// case is not supported (the CPU assembly then runs).
bool construct_fluid(ComMod& com_mod, const mshType& lM, const Array<double>& Ag, const Array<double>& Yg);

/// The matrix of the current equation is on the device (host assembly of
/// element contributions is staged, see stage()).
bool val_on_device();
void set_on_device(bool on);

/// Stage matrix entries assembled on the host (face terms): ptr is the
/// position in the FSILS matrix, vals the dof*dof block (n values).
void stage(int ptr, const double* vals, int n);

/// FSILS NS, GMRES or CG solver on the device matrix: preconditioner and
/// solve (fsils_solve with the FSILS preconditioner, one process).
void fsils_solve(fsi_linear_solver::FSILS_lhsType& lhs, fsi_linear_solver::FSILS_lsType& ls, int dof,
    Array<double>& R, Array<double>& Wc);

/// FSI (svwarp.fsi_setup, fsi_add_mesh, fsi_assemble), see svwarp_fsi_host.cpp
void fsi_setup(int tnNo, const double* x, int nnz, const int* rowPtr, const int* colPtr);
int fsi_add_mesh(int nEl, const int* IEN, int nG, const double* w, const double* N, const double* Nxi);
void fsi_assemble(int mesh, int type, int tDof, const double* Ag, const double* Yg, const double* Dg,
    const double* Do, const double* Bf, const double* params, double* R);

}  // namespace svwarp_host

#endif
#endif
