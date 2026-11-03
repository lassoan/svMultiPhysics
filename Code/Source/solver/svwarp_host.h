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

/// Stage matrix entries assembled on the host (face terms): ptr is the
/// position in the FSILS matrix, vals the dof*dof block.
void stage(int ptr, const double* vals);

/// FSILS NS solver on the device matrix: preconditioner and solve
/// (fsils_solve with the FSILS preconditioner, one process).
void fsils_solve(fsi_linear_solver::FSILS_lhsType& lhs, fsi_linear_solver::FSILS_lsType& ls, int dof,
    Array<double>& R, Array<double>& Wc);

}  // namespace svwarp_host

#endif
#endif
