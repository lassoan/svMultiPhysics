# GPU acceleration with NVIDIA Warp (experiment)

This directory contains an experimental GPU implementation of the per-time-step computation of the fluid equation with rigid walls (CFD) and of FSI (see below): the element assembly and the FSILS linear solvers, written in Python with [NVIDIA Warp](https://github.com/NVIDIA/warp) and called from svMultiPhysics through an embedded Python interpreter. It is the Warp counterpart of the hand-written CUDA experiment on the `gpu-exp` branch (`Code/Source/gpu`), with the same hooks into svMultiPhysics, so that the two approaches can be compared. It is not a production feature.

## Summary

| Case | Mesh | CPU, 10 MPI processes | Warp (this branch) | CUDA (`gpu-exp`) |
|---|---|---|---|---|
| `tests/cases/fluid/iliac_artery`, 20 time steps | 75,819 tetrahedra, 15,334 nodes | 106.3 s | 9.1 s (**12×**) | 11.7 s (9×) |
| Pipe (`Code/Source/gpu/make_pipe.py`), 10 time steps | 1,002,240 tetrahedra, 181,875 nodes | 1,631 s | 33.0 s (**49×**) | 43.9 s (37×) |
| FSI pipe (`Code/Source/gpu/make_fsi_pipe.py`), 5 time steps | 1,244,160 tetrahedra, 220,409 nodes | 885 s | 31.7 s (**28×**) | 17.0 s (52×) |

Hardware: Intel Core i9-10900X (10 cores) and NVIDIA GeForce RTX 3090, Windows 11, Warp 1.19. Times are the solver times reported by svMultiPhysics (the CPU runs use the default Windows build: clang-cl, mimalloc). The results agree with the CPU results within 7·10⁻¹⁴ (iliac artery) and 2·10⁻¹⁰ (pipe) relative difference; the Newton iteration counts are the same as on the CPU (86 for the pipe, 184 for the iliac artery).

Main conclusions:

- **Warp reaches and passes the hand-written CUDA version on both meshes.** On the 1M-element pipe the solve is bandwidth-bound and the two implementations use the same algorithm; the Warp version is faster because of a better Schur complement preconditioner (weighted aggregation, see below) and a cheaper per-solve setup. On the 76k-element iliac artery every kernel is launch- and latency-bound; the first Warp version was 2.5× slower than CUDA (28.5 s), the current one is 20% faster, with the measures described under *Small meshes*.
- **About half of the code of the CUDA version.** The Warp implementation is about 2,700 lines of Python plus 240 lines of C++ (the embedded interpreter and the hooks), against about 4,500 lines of CUDA/C++ on `gpu-exp` for the same functionality.
- **Same CPU code reused.** As with the CUDA version, the time integration, boundary conditions, output and the FSILS matrix structure are the existing C++ code; only the element integrand and the linear solver are replaced (ported line by line, so that the element matrices match the CPU ones to rounding).

## Building and running

1. A Python 3.12 installation with the development headers and a virtual environment with `warp-lang`, `numpy`, `scipy` (Schur complement pattern) and `pyamg` (C++ aggregation; a Python fallback exists):
   ```
   python -m venv venv
   venv/Scripts/pip install warp-lang numpy scipy pyamg
   ```
2. Configure svMultiPhysics with
   ```
   -DSV_USE_WARP=ON -DPython3_ROOT_DIR=<Python installation> -DSV_WARP_SITE_PACKAGES=<venv>/Lib/site-packages
   ```
   The Python DLL is copied next to the executable. The paths of the Python installation, the site-packages and the `svwarp` sources are compiled in and can be overridden at run time with the environment variables `SV_WARP_PYTHON_HOME`, `SV_WARP_SITE_PACKAGES` and `SV_WARP_SOURCE_DIR`.
3. Run with one process and the environment variables `SV_WARP=1` and `SV_WARP_ASSEM=1`. Without them the CPU code runs. The first run compiles the kernels (about 10 s); Warp caches them.

Other environment variables: `SV_LS_STATS=1` (iteration counts of every solve), `SVWARP_TIMING=1` (time per solve: Schur setup, GMRES, CG, assembly), `SV_GPU_VERBOSE=1` (setup information), `SVWARP_FMA=1` (fused multiply-add in the assembly kernel: 10% faster assembly, results differ from the CPU in the last digits), `SVWARP_CG_CHUNK` (CG iterations per convergence check, default 8 for small meshes down to 2 for large ones), `SVWARP_AMG_W` (Jacobi damping, default 4/3), `SVWARP_AMG_CAND` and `SVWARP_AMG_WC` (relaxation sweeps and damping of the prolongator weights, default 16 and 1), `SVWARP_AMG_CMAX` (size of the coarsest level, default 512), `SVWARP_AMG_GRAPH=mesh` (aggregation on the mesh graph instead of the Schur complement graph), `SVWARP_NOGRAPH=1` and `SVWARP_DEBUG=1` (no CUDA graphs, CUDA error checking after every kernel), `SVWARP_PROFILE=1` (GPU time of every kernel of the first launch of every solver phase; the times of small kernels include launch gaps), `SVWARP_DUMP=<file>` (dump of a Schur system for offline tests).

Supported cases: one process; fluid equation with linear tetrahedra, VMS stabilization (`nFs = 1`), one mesh and one domain with constant viscosity, no moving mesh, no RIS valves, up to 8 resistance (coupled Neumann) faces without caps, FSILS preconditioner and `NS` linear solver; FSI with linear tetrahedra, one fluid and one wall domain (one domain per mesh), constant fluid viscosity, neo-Hookean wall with the M94 volumetric model, no solid viscosity, active stress or prestress, mesh motion with linear elasticity, GMRES and CG with the FSILS preconditioner, no coupled faces (`SVWARP_AMG=0` for the plain FSILS algorithms, `SVWARP_AMG_CMAX` for the size of the coarsest AMG level, default 128 nodes). Other cases use the CPU code or stop with an error.

## Files

| File | Contents |
|---|---|
| `svwarp/__init__.py` | Functions called from C++ (setup, assembly, staged face entries, preconditioner, solve) |
| `svwarp/fluid.py` | Element assembly kernel: `fluid_3d_m` and `fluid_3d_c` of `physics/fluid/src/fluid.cpp` for linear tetrahedra |
| `svwarp/linalg.py` | Device matrix (FSILS layout), diagonal preconditioner, vector and reduction kernels, resistance face kernels |
| `svwarp/sell.py`, `svwarp/sell32.py` | Sliced ELLPACK storage with coalesced sparse products (double) and the kernels of the AMG hierarchy (single precision) |
| `svwarp/ns.py` | FSILS NS solver: GMRES on the momentum block, AMG-preconditioned CG on the Schur complement, as CUDA graphs |
| `svwarp/amg.py` | Aggregation AMG preconditioner |
| `svwarp/common.py` | Types, mapped host memory |
| `svwarp/fsi.py` | FSI: assembly kernels of the ALE fluid, neo-Hookean wall and mesh motion equations, device matrices for dof = 3 and 4 |
| `svwarp/bls.py` | FSI: block SELL products, aggregation AMG for the dof × dof block matrix, GMRES and CG for dof = 3 and 4 (closure kernels generated per block size) |
| `../solver/svwarp_host.cpp`, `../solver/svwarp_fsi_host.cpp` | Embedded Python interpreter, the hooks of `construct_fluid`, `construct_fsi`, `construct_mesh`, `FsilsLinearAlgebra::assemble` and `fsils_solve` |

Changes outside this directory: `solver/physics/fluid/src/fluid.cpp` (one call at the start of `construct_fluid`), `solver/FsilsLinearAlgebra.cpp` (face terms staged for the device matrix, no reallocation of the host matrix), `linear_solver/solve.cpp` (GPU path of `fsils_solve`), and the CMake files (`SV_USE_WARP`).

## Design

**Data flow.** The C++ hooks pass host pointers of the svMultiPhysics arrays to Python (as integers), where they are wrapped without copies. Per Newton iteration: the state arrays (`Ag`, `Yg`, `Bf`, about 30 MB for the pipe) are copied to the GPU, the element kernel assembles the matrix into the device-resident FSILS block structure with atomic additions and the residual is copied back and added to `R`; the boundary-condition integrals stay on the CPU and their matrix entries are staged and added to the device matrix in the solve; the solve returns the solution in `R`.

**Assembly.** One thread per element computes the 4×4 element nodes × 4 dof element matrix (`fluid_3d_m`, `fluid_3d_c`) for all Gauss points, as in the CUDA version. The terms that vanish for linear elements with constant viscosity (second derivatives of the shape functions, the derivative of the viscosity) are left out; the remaining operations are in the CPU order, so that without FMA the element matrices equal the CPU ones to rounding. Warp's fixed-size matrix types hold the local arrays (a 16×16 matrix for the element matrix). The kernel is bound by the FP64 arithmetic of the element matrix (60 ns per element, 5.8 ms for the iliac artery, of which the atomic additions are 1.3 ms).

**Linear solver.** The FSILS NS algorithm is ported with the vectors on the GPU and only scalars on the host (restart logic, Givens rotations, the Gram matrix of the outer iteration):
- *GMRES* on the momentum block: one Arnoldi step is two kernels. The first computes the product with the 3×3 blocks and the resistance terms and the projections on the previous basis vectors (classical Gram-Schmidt with block-wide reductions); the second subtracts the projections and computes the norm. The coupled-face terms (`add_bc_mul` ADD and PRE, applied face by face as on the CPU) need the dot products of the face vectors with the basis vector and with its product, which the previous step accumulates together with the norm, with precomputed Kᵀv; so no extra reduction kernels are needed. The host downloads the new Hessenberg column after every step.
- *Schur complement CG*: the Schur complement L + Gᵀ G is formed explicitly by one kernel (an atomic addition per pair of entries of G, binary search in the sorted pattern), the resistance faces contribute symmetric rank-1 terms, and the CG is preconditioned with a 2- or 3-level aggregation AMG in single precision; the CPU CG is not preconditioned. One CG iteration is 6 kernels (direction and face dots; product and p·q; update, |r|² and pre-smoothing; residual and restriction; coarse solve; prolongation and post-smoothing with r·z). A CUDA graph runs 2–8 iterations; the host checks the convergence after every graph. The convergence criteria are those of the CPU solver.
- *Outer iteration*: the products, the dots of the Gram matrix and the residual update of every outer iteration are one graph each; the host solves the small least-squares problem.
- *Sparse products* use a sliced ELLPACK layout (slices of 32 rows padded to the longest row, entries stored entry-major, loops unrolled 4×) so that one thread per row reads coalesced memory: 2–3× faster than thread-per-row CSR on these matrices.

**Preconditioner.** Standard aggregation (PyAMG's algorithm) of the Schur complement graph gives aggregates of about 60–75 nodes; the coarsest level (at most 512 nodes, 202 for the iliac artery, 80 at the third level of the pipe) is inverted by Gauss-Jordan steps on the device. The FSILS diagonal scaling of the matrix makes the near-kernel of the Schur complement far from constant, so a piecewise constant prolongator needs about 100 CG iterations per Schur solve on the iliac artery; the prolongator weights are therefore PyAMG's "improved candidate" (16 damped Jacobi sweeps on A b = 0 from b = 1, normalized per aggregate), which brings it to 28–37 iterations (the Galerkin products weight the entries accordingly). One damped Jacobi sweep with ω = 4/3 / ρ(D⁻¹A) before and after the coarse correction, ρ from a power iteration on the device. The parameters were chosen offline with PyAMG and a NumPy emulation of the hierarchy on Schur systems dumped from the solver (`SVWARP_DUMP`): smoothed aggregation or Chebyshev smoothing need fewer iterations but more sparse products per iteration, which does not pay on either mesh.

**Small meshes: launch and latency.** On the 15k-node iliac artery a solve has about 1,000 kernels, so that the fixed costs dominate. Measurements on this machine (Windows, WDDM): a kernel launched from Python costs 28 µs of host time and about 50 µs of latency; a kernel inside a CUDA graph costs about 1 µs; a host synchronization after a graph about 10 µs; a memcpy node inside a graph 40–80 µs; a kernel writing 64-byte vectors to mapped pinned host memory 2 µs. The solver is therefore organized as follows:
- Every phase (per-solve setup, Arnoldi step, chunk of CG iterations, outer iteration phases) is one CUDA graph of fused kernels, captured on first use (about 60 graphs for the iliac artery). The host only steers the iterations.
- Scalars for the host (norms, Hessenberg columns, Gram matrices) are written by the last kernel of a graph into mapped pinned memory (`Mapped` in `common.py`) and read after one stream synchronization; the coefficients computed on the host are read by the first kernel of the next graph from mapped memory. No memcpy nodes.
- Reductions are block-wide tile reductions (`wp.tile`, `wp.tile_sum`, `wp.tile_atomic_add`) fused into the kernels that produce the vectors, with the accumulators zeroed by an earlier kernel of the same graph. Kernels run with a padded number of threads so that every block is complete.
- The per-solve setup (sub-block gathers, Schur complement, AMG weights, coarse matrices, power iterations, dense inverse) is one graph without any host synchronization.
- What remains per Newton iteration on the iliac artery: assembly 8 ms (kernel 5.8), Schur setup 3 ms, GMRES 8 ms (52 steps at 0.15 ms: 0.09 ms of kernels, the rest launch and synchronization), CG 11 ms (about 115 iterations at 0.1 ms: three products with the Schur complement of 884k entries at about 20 µs each, which is latency-bound at 15k rows), outer iteration 3 ms. The CUDA version runs whole solvers in single cooperative kernels; Warp has no grid-wide synchronization, so the kernel count per iteration is its floor.

## Where the time goes (pipe, per Newton iteration, 0.17 s)

| Part | Time | Notes |
|---|---|---|
| Schur CG | 0.096 s | about 45 iterations per Schur solve, 4–5 Schur solves; 0.5 ms per iteration: 3 sparse products with the Schur complement, 3 reductions |
| GMRES | 0.051 s | about 77 Arnoldi steps per solve at 0.66 ms: product with the 3×3 blocks 0.42 ms (450 GB/s) |
| Assembly | 0.07 s | kernel 60 ms (45 ms with FMA), state upload and residual download 10 ms |
| Schur setup | 0.011 s | Gᵀ G (2.6M entries of G, 10.9M of S), AMG weights and coarse matrices, power iterations, 80×80 inverse |
| Preconditioner, copies, host | 0.01 s | |

## FSI

The FSI case is `tests/cases/fsi/pipe_3d` on larger meshes (`Code/Source/gpu/make_fsi_pipe.py` of the CUDA branch: lumen radius 1, wall thickness 0.2, length 10, conforming tetrahedral meshes): ALE fluid in the current configuration with the mesh velocity, neo-Hookean wall with the M94 volumetric penalty, linear elastic mesh motion, 5 time steps. The FSI equation is solved with GMRES (dof = 4, Krylov space 50), the mesh motion with CG (dof = 3), both with the FSILS diagonal preconditioner on the CPU and with an additional aggregation AMG V-cycle on the GPU (right preconditioner of GMRES, preconditioner of CG; `SVWARP_AMG=0` selects the plain FSILS algorithms, which then reproduce the CPU iteration counts of the CG exactly and the fields within 10⁻¹⁴).

| Mesh | svMultiPhysics, 10 MPI processes | CUDA (`gpu-exp`) | Warp (`warp-exp`) | MFEM (`mfem-exp`) |
|---|---|---|---|---|
| 192k tetrahedra (`mesh16`), 35,721 nodes | 63.1 s | 4.0 s | 7.0 s | 5.5 s |
| 1.24M tetrahedra (`mesh32`, 983k fluid and 261k wall), 220,409 nodes | 885 s | 17.0 s | 31.7 s | 24.4 s |

The CPU time is the current clang-cl build with AVX2 (`msvc-build`, 10 MPI processes); the CUDA branch measured 745 s for the same run with the build without AVX2. All versions take 35 Newton iterations of the FSI equation and 35 of the mesh equation on the 1.24M mesh with the same Newton residuals; the AMG brings the GMRES iterations of the run from 26,486 (CPU) to about 3,850 and the CG iterations from 9,896 to about 2,120. The velocity, pressure, displacement and stress fields agree with the CPU within 7·10⁻¹⁴ (MFEM) and 5·10⁻¹¹ (Warp) relative difference.

The implementation (`fsi.py`, `bls.py`, about 1,500 lines, one day of work alongside the MFEM port) follows the CUDA branch and the MFEM version:

- **Assembly**: `fluid_3d_m`/`fluid_3d_c` of `fluid.py` extended with the mesh velocity (`mvMsh`), `struct_3d` and `l_elas_3d` ported by hand from the CUDA kernels with Warp matrix types (a 24×3 matrix holds the strain-displacement matrices of the four nodes, a 9×16 matrix the mesh element matrix), one kernel per mesh and equation with atomic additions of 4×4 or 3×3 blocks.
- **Closure kernels**: the solver and AMG kernels are generated per block size by `bls.make(DOF)`, a function whose Warp kernels capture `DOF` and the matrix types (`mat44d`, `mat33f`, ...) as constants; Warp compiles one module per block size (about 40 s the first time, cached afterwards).
- **Block products** on arrays of Warp matrices in the SELL layout, two slots per loop iteration; the AMG hierarchy in `float32` matrices.
- **AMG**: as in the MFEM version (aggregation of the node graph, piecewise constant prolongation, block Jacobi with the pseudo-inverse of the diagonal blocks, damping from a power iteration on the device, Galerkin sums by gather lists); the pseudo-inverse of the coarsest matrix (up to 512 unknowns) is `numpy.linalg.pinv` on the host, about 20 ms per solve.
- **GMRES and CG** as in the MFEM version: one graph per Arnoldi step (normalization, V-cycle, product, modified Gram-Schmidt as one reduction kernel per basis vector, with grid-stride loops), the Hessenberg column through mapped memory; CG in chunks of two iterations per graph.

Per Newton iteration on the 1.24M mesh: assembly 41 ms, GMRES 4.3 ms per Arnoldi step, CG 1.4 ms per iteration. The Warp version is 30% slower than the MFEM version on this mesh: the Arnoldi step costs 4.3 ms against 3.3 ms, mostly in the V-cycle (1.2 ms against 0.6 ms), whose float block matrices are arrays of 4×4 matrices (64-byte loads per thread) rather than the entry-major layout of the MFEM version; without the AMG the step costs 3.1 ms in both. The Python launch overhead is hidden by the graphs.

## Warp specifics learned

- Python float literals are `float32` in kernels; double precision code needs `wp.float64(...)` everywhere.
- Kernels compile in seconds and are cached; the first run of a new kernel signature costs about 1–4 s. Modules must be loaded (`wp.load_module`) before a graph capture.
- CUDA graph capture works with Warp launches, but everything captured must be allocation-free and must refer to persistent arrays: `array_inner` (scratch allocation), per-solve objects and kernel constants baked into a graph all produced wrong results or illegal memory accesses until replaced by persistent device arrays. Scalars that change per solve (damping factors, face coefficients) are read from device arrays, never passed as kernel arguments.
- Memcpy and memset nodes inside graphs are slow on Windows (40–80 µs); kernels reading and writing mapped pinned memory in 64-byte vectors replace them. Writing single doubles to mapped memory from many threads costs 40 µs per 144 values.
- `wp.tile` turns per-thread values into a block-wide tile, `wp.tile_sum` reduces it, `wp.tile_atomic_add` accumulates into a device array with a runtime offset; this works inside runtime loops. Every thread of the block must reach the tile operations (padded launch sizes, guarded memory accesses).
- `bsr_set_from_triplets` prunes numerical zeros by default; `bsr_mm` with `reuse_topology` needs its `work_arrays` (both replaced by own kernels now).
- There is no grid-wide synchronization or warp-level shuffle in kernels (apart from the tile API), so solver loops cannot be fused into one kernel as in the CUDA version; kernels with one thread per row of a 15k-row matrix cannot hide the memory latency (a 58-entry row costs 20–40 µs even with unrolled loops; 4 threads per row with tile reductions gains little).
- `wp.config.verify_cuda` cannot be used with graph capture (`SVWARP_NOGRAPH=1` for debugging).
- The OpenBLAS of the pip NumPy takes 2 ms for a 200×200 Cholesky and inverse on this machine (hence the Gauss-Jordan inverse on the device).

## Possible further work

- Small meshes: the momentum products in single precision (halves the 17 MB of the 3×3 blocks of the iliac artery per Arnoldi step), two Arnoldi steps per graph, a faster element kernel (one warp per element so that the 16×16 element matrix stays in registers).
- Multi-GPU through MPI would need halo exchanges in the solver.
