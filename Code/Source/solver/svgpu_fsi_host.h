// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SVGPU_FSI_HOST_H
#define SVGPU_FSI_HOST_H

#include "ComMod.h"
#include "SolutionStates.h"

namespace svgpu_host {

/// GPU assembly of a mesh of the FSI equation; false if not supported
bool construct_fsi(ComMod& com_mod, const mshType& lM, const SolutionStates& solutions);

/// GPU assembly of a mesh of the mesh motion equation; false if not supported
bool construct_mesh(ComMod& com_mod, const mshType& lM, const SolutionStates& solutions);

}

#endif
