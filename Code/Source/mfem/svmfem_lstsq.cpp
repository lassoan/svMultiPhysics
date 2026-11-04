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

/// Pseudo-inverse of the n x n row-major matrix A (the coarsest matrix of the
/// FSI AMG, with zero rows and columns of the Dirichlet dofs), in float
void pinv(int n, const double* A, float* Ainv)
{
  using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<const Mat> M(A, n, n);
  Mat I = M.completeOrthogonalDecomposition().pseudoInverse();
  for (int q = 0; q < n * n; q++) Ainv[q] = (float)I.data()[q];
}

}  // namespace svmfem
