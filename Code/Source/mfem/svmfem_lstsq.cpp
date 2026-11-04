// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// Least squares solution of the projected system of the outer NS iteration
// (as ns_solver.cpp), host only.

#include <Eigen/QR>

namespace svmfem {

void lstsq_solve(int n, const double* A, const double* b, double* x)
{
  Eigen::Map<const Eigen::MatrixXd> M(A, n, n);
  Eigen::Map<const Eigen::VectorXd> rhs(b, n);
  Eigen::VectorXd sol = M.completeOrthogonalDecomposition().solve(rhs);
  for (int i = 0; i < n; i++) x[i] = sol(i);
}

}  // namespace svmfem
