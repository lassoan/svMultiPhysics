// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// Experimental GPU (CUDA) implementation of the FSILS NS linear solver
// (single process). Plain C interface, so that the library can be built with
// nvcc independently of the compiler used for svMultiPhysics.

#ifndef SVGPU_H
#define SVGPU_H

#ifdef _WIN32
#  ifdef SVGPU_EXPORTS
#    define SVGPU_API __declspec(dllexport)
#  else
#    define SVGPU_API __declspec(dllimport)
#  endif
#else
#  define SVGPU_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// Coupled (resistance) Neumann face of the FSILS lhs.
typedef struct {
  int nNo;              ///< number of face nodes
  const int* glob;      ///< node indices (in FSILS ordering)
  const double* valM;   ///< nsd x nNo, column-major
  double res;           ///< resistance
} svgpu_face;

typedef struct {
  int mItr;
  int sD;
  double absTol;
  double relTol;
} svgpu_sub_params;

typedef struct {
  svgpu_sub_params RI, GM, CG;
} svgpu_params;

typedef struct {
  int RI_itr, GM_itr, CG_itr;
  int success;
  double iNorm, fNorm, dB;
  int Resm, Resc;
  double time;
} svgpu_stats;

/// Solve the (preconditioned) NS system A*x = R with the FSILS NS algorithm.
///
/// rowPtr: 2 x nNo (first and last entry index of each row), colPtr: nnz,
/// Val: dof*dof x nnz, R: dof x nNo (input right-hand side, output solution).
/// The matrix structure must be the same in all calls. With Val = null, the
/// device resident matrix of the GPU assembly is used.
SVGPU_API int svgpu_ns_solve(int nNo, int nnz, int dof, const int* rowPtr, const int* colPtr,
    const double* Val, double* R, int nFaces, const svgpu_face* faces, const svgpu_params* params,
    svgpu_stats* stats);

/// Set up the GPU assembly of the fluid equation (linear tetrahedra). IEN:
/// 4 x nEl, x: 3 x tnNo, rowPtr: tnNo+1 (CSR), colPtr: nnz, w: nG,
/// N: 4 x nG, Nxi: 3 x 4 (parametric shape function derivatives).
SVGPU_API int svgpu_fluid_setup(int nEl, const int* IEN, int tnNo, const double* x, int nnz,
    const int* rowPtr, const int* colPtr, int nG, const double* w, const double* N, const double* Nxi);

/// Assemble the fluid equation and add it to R (4 x tnNo) and Val (16 x nnz).
/// With Val = null, the matrix stays on the device (svgpu_ns_solve with
/// Val = null uses it).
SVGPU_API int svgpu_fluid_assemble(int tDof, const double* Ag, const double* Yg, const double* Bf,
    double rho, double mu, const double* f, double dt, double af, double am, double gam, double brink,
    double* R, double* Val);

/// Add matrix entries (16 values each) to the device resident matrix.
SVGPU_API int svgpu_add_val(int n, const int* ptr, const double* vals);

/// FSILS diagonal preconditioner of the device resident matrix (dof = 4).
SVGPU_API int svgpu_precond_diag(int nNo, int nnz, const int* rowPtr, const int* colPtr,
    const int* diagPtr, const double* Dmul, double* W);

#ifdef __cplusplus
}
#endif

#endif
