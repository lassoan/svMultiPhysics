// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// Device resident FSILS matrix shared by the assembly (svmfem_fluid.cu) and
// the NS solver (svmfem_ns.cu).

#ifndef SVMFEM_INTERNAL_HPP
#define SVMFEM_INTERNAL_HPP

#include "svmfem_common.hpp"

namespace svmfem {

/// CSR block structure in the FSILS ordering (identical to the mesh ordering
/// with one process), the 4x4 block values (Val[16 j + 4 r + c]) and the
/// diagonal scaling W of the preconditioner.
struct Lhs {
  int nNo = 0, nnz = 0;
  std::vector<int> h_off, h_col;             // host CSR (nNo + 1, nnz)
  Array<int> offsets, columns, rows, diag;   // device CSR, row of each entry, diagonal entries
  Array<int> tpos;                           // position of the transposed entry of each entry
  Vector Val;                                // 16 x nnz
  Vector W, Dmul;                            // 4 x nNo
};

/// The device matrix (set up by svmfem_fluid_setup), null before
Lhs* lhs();

}  // namespace svmfem

#endif
