// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// Experimental GPU acceleration of the fluid equation with MFEM (assembly,
// FSILS diagonal preconditioner and FSILS NS linear solver on the device
// resident matrix, one process). Plain C interface, so that the library can
// be built with nvcc and MSVC independently of the compiler used for
// svMultiPhysics (clang-cl).

#ifndef SVMFEM_H
#define SVMFEM_H

#ifdef _WIN32
#  ifdef SVMFEM_EXPORTS
#    define SVMFEM_API __declspec(dllexport)
#  else
#    define SVMFEM_API __declspec(dllimport)
#  endif
#else
#  define SVMFEM_API
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
} svmfem_face;

typedef struct {
  int mItr;
  int sD;
  double absTol;
  double relTol;
} svmfem_sub_params;

typedef struct {
  svmfem_sub_params RI, GM, CG;
} svmfem_params;

typedef struct {
  int RI_itr, GM_itr, CG_itr;
  int success;
  double iNorm, fNorm, dB;
  int Resm, Resc;
  double time;
} svmfem_stats;

/// Configure the MFEM device: "cuda" (default) or "cpu" (the same kernels on
/// the host), from the environment variable SVMFEM_DEVICE when device is null.
SVMFEM_API int svmfem_init(const char* device);

/// Release the device objects (also registered with atexit by svmfem_init:
/// the MFEM memory manager must release the device memory before the CUDA
/// runtime shuts down).
SVMFEM_API void svmfem_finalize(void);

/// Set up the GPU assembly of the fluid equation (linear tetrahedra) and the
/// device resident matrix. IEN: 4 x nEl, x: 3 x tnNo, rowPtr: tnNo+1 (CSR),
/// colPtr: nnz, w: nG, N: 4 x nG, Nxi: 3 x 4 (parametric shape function
/// derivatives).
SVMFEM_API int svmfem_fluid_setup(int nEl, const int* IEN, int tnNo, const double* x, int nnz,
    const int* rowPtr, const int* colPtr, int nG, const double* w, const double* N, const double* Nxi);

/// Assemble the fluid equation into the device matrix (cleared first) and
/// add the residual to R (4 x tnNo, host).
SVMFEM_API int svmfem_fluid_assemble(int tDof, const double* Ag, const double* Yg, const double* Bf,
    double rho, double mu, const double* f, double dt, double af, double am, double gam, double brink,
    double* R);

/// Add n matrix entries (16 values each, at the given entry positions) to the
/// device resident matrix.
SVMFEM_API int svmfem_add_val(int n, const int* ptr, const double* vals);

/// FSILS diagonal preconditioner of the device resident matrix (dof = 4):
/// Dmul (4 x nNo) are the Dirichlet factors of precond_diag, W (4 x nNo)
/// receives the scaling.
SVMFEM_API int svmfem_precond_diag(const double* Dmul, double* W);

/// Solve the preconditioned NS system on the device matrix with the FSILS NS
/// algorithm. R: 4 x nNo (input right-hand side, output solution).
SVMFEM_API int svmfem_ns_solve(double* R, int nFaces, const svmfem_face* faces, const svmfem_params* params,
    svmfem_stats* stats);

#ifdef __cplusplus
}
#endif

#endif
