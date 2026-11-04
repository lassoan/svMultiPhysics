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

// ---------------------------------------------------------------------------
// FSI (fsi.cpp, mesh.cpp): assembly of the FSI equation (ALE fluid and
// neo-Hookean wall, dof = 4) and of the mesh motion equation (linear
// elasticity, dof = 3) on linear tetrahedra, and the FSILS GMRES and CG
// solvers for dof = 3 and 4 on the device resident matrix, preconditioned
// with an aggregation AMG V-cycle (unless SVMFEM_AMG=0).

/// Node coordinates (3 x tnNo) and the matrix structure (CSR, tnNo+1 and
/// nnz) of the device resident matrix (16 x nnz values for dof = 4, the
/// first 9 x nnz for dof = 3).
SVMFEM_API int svmfem_fsi_setup(int tnNo, const double* x, int nnz, const int* rowPtr, const int* colPtr);

/// Add a mesh (IEN: 4 x nEl, Gauss data as svmfem_fluid_setup); returns its
/// index, or -1.
SVMFEM_API int svmfem_fsi_add_mesh(int nEl, const int* IEN, int nG, const double* w, const double* N,
    const double* Nxi);

/// Assemble one mesh into the device matrix (added, not cleared: the matrix
/// is cleared by svmfem_ls_solve) and add the residual to R (dof x tnNo,
/// host). type: 0 = FSI fluid (ALE), 1 = FSI struct, 2 = mesh motion. Ag,
/// Yg, Dg, Do: tDof x tnNo (Do only for type 2), Bf: 3 x tnNo. params: fluid
/// {rho, mu, fx, fy, fz, dt, af, am, gam}, struct {rho, dmp, fx, fy, fz,
/// C10, Kpen, dt, af, am, beta, gam}, mesh {rho, E, nu, fx, fy, fz, dt, af,
/// am, beta}.
SVMFEM_API int svmfem_fsi_assemble(int mesh, int type, int tDof, const double* Ag, const double* Yg,
    const double* Dg, const double* Do, const double* Bf, const double* params, double* R);

/// Add n matrix entries (dof*dof values each) to the device matrix
SVMFEM_API int svmfem_add_val_dof(int n, int dof, const int* ptr, const double* vals);

/// FSILS diagonal preconditioner of the device matrix for dof = 3 or 4
/// (Dmul, W: dof x nNo)
SVMFEM_API int svmfem_precond_diag_dof(int dof, const double* Dmul, double* W);

/// FSILS GMRES (type 0) or CG (type 1) on the device matrix. R: dof x nNo
/// right-hand side in, solution out. The device matrix is cleared afterwards.
SVMFEM_API int svmfem_ls_solve(int type, int dof, int mItr, int sD, double absTol, double relTol, double* R,
    svmfem_stats* stats);

#ifdef __cplusplus
}
#endif

#endif
