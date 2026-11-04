// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// Aggregation AMG preconditioner for a scalar sparse matrix (the Schur
// complement of the NS solver), see svmfem_amg.cu.

#ifndef SVMFEM_AMG_HPP
#define SVMFEM_AMG_HPP

#include "svmfem_common.hpp"

namespace svmfem {

struct AmgLevel {
  int n = 0, nnz = 0;
  std::vector<int> h_off, h_col;            // host CSR pattern
  Array<int> offsets, columns;              // device CSR pattern (levels >= 1)
  Array<int> diag;                          // diagonal entry of every row (CSR)
  Vector V;                                 // CSR values, double (levels >= 1)
  const double* pV = nullptr;               // CSR values used by the kernels (level 0: the caller's)
  Sell sellL;                               // SELL layout (levels >= 1)
  const Sell* sell = nullptr;               // SELL layout used by the kernels (level 0: the caller's)
  Array<float> Vs;                          // SELL values, float (levels >= 1)
  const float* pVs = nullptr;               // SELL values used by the kernels (level 0: the caller's)
  Array<float> Dinv, r, z, z2;              // inverse diagonal, residual, corrections
  // Aggregation to the next level
  Array<int> agg, cmap, frow, fcol;         // aggregate of each node; coarse entry, row and column of each entry
  Array<float> pb, bb0, bb1, anrm, ev0, ev1;   // prolongator weights, candidate, aggregate norms, power iteration
};

class Amg {
 public:
  /// Structure of the hierarchy for a matrix with the CSR pattern (off, col)
  /// on the host, whose SELL layout is sell0 (shared with the caller, which
  /// also owns the values of the finest level); omega: Jacobi damping
  /// relative to 1/rho(D^-1 A).
  Amg(int n, const std::vector<int>& off, const std::vector<int>& col, const Sell* sell0, double omega);

  std::vector<int> sizes() const;
  int nlevels() const { return (int)levels.size(); }
  /// Power iteration steps of the next emit_numeric
  int npow() const;
  AmgLevel& level(int l) { return levels[l]; }
  /// Smoother damping per level (device, float)
  const float* damping() { return wv.Read(); }

  /// Kernel launches of the weights, the coarse matrices, the smoothers and
  /// the dense coarsest matrix for the finest level values V0 (device CSR,
  /// double) and Vs0 (their SELL float copy).
  void emit_numeric(const double* V0, const float* Vs0, int nPow);

  /// V-cycle below the finest level: restriction of the finest residual
  /// r - A zs (r double, zs the pre-smoothed float correction) and the coarse
  /// levels; returns the float correction of level 1, to be prolongated and
  /// post-smoothed by the caller. The residual accumulator of level 1 must
  /// be zero.
  const float* emit_down(const double* r, const float* zs);

 private:
  std::vector<AmgLevel> levels;
  double omega;
  int nSolve = 0;
  int nc = 0;                    // coarsest level size
  Vector Ad0, Ad1;               // dense coarsest matrix (Gauss-Jordan ping-pong)
  Array<float> Ainv;             // its inverse
  Array<float> wv, wc;           // smoother and candidate damping per level
  Vector ns;                     // norms of the power iterations (16 per level)
};

/// Standard aggregation (as in PyAMG) of the graph (CSR); returns the number
/// of aggregates and the aggregate of each node in agg
int aggregate(int n, const std::vector<int>& off, const std::vector<int>& col, std::vector<int>& agg);

}  // namespace svmfem

#endif
