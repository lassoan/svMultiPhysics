# GPU acceleration experiment

This directory contains an experimental CUDA implementation of the per-time-step computation of the fluid equation: element assembly, FSILS diagonal preconditioner and FSILS NS linear solver. The goal of the experiment was to find out how much faster svMultiPhysics can be on a GPU, and what it would take to get there. It is not a production feature.

## Summary

| Case | Mesh | CPU, 10 MPI processes | GPU | Speedup |
|---|---|---|---|---|
| `tests/cases/fluid/iliac_artery`, 20 time steps | 75,819 tetrahedra, 15,334 nodes | 105.1 s | 11.7 s | **9×** |
| Pipe (`make_pipe.py`), 10 time steps | 1,002,240 tetrahedra, 181,875 nodes | 1,631 s | 43.9 s | **37×** |

Hardware: Intel Core i9-10900X (10 cores) and NVIDIA GeForce RTX 3090, Windows 11. CPU times are with the default Windows build (clang-cl, mimalloc, no AVX2). GPU and CPU results agree within 5·10⁻¹³ (iliac artery) and 6·10⁻⁹ (pipe) relative difference.

Main conclusions:

- The speedup grows with mesh size, because fixed costs per operation (kernel launch latency, synchronization) matter less on large meshes.
- About half of the gain comes from an algorithmic change: the Schur complement CG of the NS solver is preconditioned with a two-level algebraic multigrid (AMG) V-cycle. The CPU solver's unpreconditioned CG never converges within its 300-iteration limit, so this change alone would also make the CPU solver faster and more robust.
- Both the assembly and the linear solver must run on the GPU. With the GPU solver and single-process CPU assembly, the simulation is not faster than the CPU on small meshes.
- The GPU linear solver does not depend on the physics and can be maintained as one component. GPU assembly should only be extended to other physics after a refactor that lets the same element code compile for the CPU and the GPU, not as hand-maintained CUDA copies of the physics code.

## Building and running

The library is built separately with nvcc, then linked into svMultiPhysics.

1. Install the CUDA toolkit (12.x). For example, without administrator rights, using [pixi](https://pixi.sh) and conda-forge:
   ```
   pixi init cuda
   cd cuda
   pixi add "cuda-nvcc=12.9" "cuda-cudart-dev=12.9" "cuda-cccl=12.9" "cuda-version=12.9"
   ```
2. Build `svgpu.dll` from an *x64 Native Tools Command Prompt for VS 2022*, with nvcc in the `PATH`:
   ```
   cmake -S Code/Source/gpu -B svgpu-build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
   cmake --build svgpu-build
   ```
3. Configure svMultiPhysics with `-DSV_GPU_DIR=<path of svgpu-build>`, build it, and copy `svgpu.dll` next to `svmultiphysics.exe`.
4. Run with one process and the environment variables `SV_GPU=1` (GPU linear solver) and `SV_GPU_ASSEM=1` (GPU assembly). Without these variables, the CPU code is used.

Other environment variables, for experiments: `SV_LS_STATS`, `SVGPU_TIMING`, `SV_GPU_TIMING`, `SVGPU_PHASES` (timing output), `SVGPU_SCHUR_PREC=none` (CG without AMG), `SVGPU_AMG_OMEGA`, `SVGPU_AMG_RAD` (AMG parameters), `SVGPU_GMRES_KERNELS` and `SVGPU_AMG_GRAPH` (earlier implementation variants).

Supported cases: one process; fluid equation with linear tetrahedra, VMS stabilization (`nFs = 1`), one domain with constant viscosity, no moving mesh, no RIS valves, resistance (coupled Neumann) faces without caps, FSILS preconditioner and `NS` linear solver. Other cases use the CPU code or stop with an error.

## Files

| File | Contents |
|---|---|
| `svgpu.h` | C interface of the library |
| `svgpu_fluid.cu` | Element assembly, device-resident matrix, FSILS diagonal preconditioner |
| `fluid_element.inc`, `gen_fluid.py` | Element routines, generated from `physics/fluid/src/fluid.cpp` |
| `svgpu_ns.cpp` | FSILS NS solver algorithm (outer iteration, GMRES control) |
| `svgpu_kernels.cu` | Matrix setup, sparse products and vector operations, earlier CG variants |
| `svgpu_gmres.cu` | Arnoldi step of GMRES as one cooperative kernel |
| `svgpu_schur.cu` | Explicit Schur complement, CG without AMG |
| `svgpu_amg.cu` | Two-level AMG and the AMG-preconditioned CG kernel |
| `svgpu_amg_graph.cu` | CUDA graph variant of the AMG CG (slower, not used by default) |
| `make_pipe.py` | Generator of the 1M-tetrahedra pipe test mesh |

Changes outside this directory: `linear_solver/solve.cpp` (GPU path of `fsils_solve`), `solver/FsilsLinearAlgebra.cpp` (face terms staged for the device-resident matrix), `solver/physics/fluid/src/fluid.cpp` (GPU path of `construct_fluid`), and the CMake files (`SV_GPU_DIR`).

## Starting point: where the CPU time goes

Profile of iliac_artery on the CPU (10 processes), per Newton iteration:

| Part | Share of time |
|---|---|
| FSILS sparse matrix-vector products | ~45% |
| Element assembly (`construct_fluid`) | ~15-25% |
| MPI communication and synchronization | ~12% |

Iteration counts per Newton iteration: 3-5 outer NS solver iterations, ~70 GMRES iterations and ~1,500 Schur complement CG iterations. Every CG call reaches its 300-iteration limit without converging: 300 iterations reduce the residual only by a factor of 10-100, because the Schur complement is ill-conditioned and the CG is not preconditioned. This is the main waste in the CPU solver.

## Implementation

### Assembly

One thread per tetrahedron computes the element residual and tangent matrix, and adds them to the global matrix and residual with atomic adds, through an element-to-matrix-entry map that is computed once. The element routines (`fluid_3d_m`, `fluid_3d_c`) are generated from `fluid.cpp` by `gen_fluid.py`, which only replaces the `ComMod` accesses with parameters, so the physics is not rewritten by hand. This works because these routines happen to use only fixed-size arithmetic.

### Device-resident matrix

The matrix stays on the GPU between assembly and solve. The face terms that are assembled on the host (backflow stabilization) are collected in `FsilsLinearAlgebra::assemble` and added on the GPU. The FSILS diagonal preconditioner (row and column scaling, Dirichlet faces) runs on the GPU, and only the scaling vector is copied back for the face data. Only vectors are transferred between host and GPU in a Newton iteration.

### NS linear solver

The algorithm of the FSILS NS solver is kept, with all vectors on the GPU and only scalars transferred to the host.

- **GMRES.** Each Arnoldi step is one cooperative kernel: block matrix-vector product, resistance terms, modified Gram-Schmidt and normalization. The update `w -= h_j u_j` and the partial sums of the next dot product are computed in the same pass, so each Gram-Schmidt step needs one grid synchronization. Only the Hessenberg column is copied to the host, for the Givens rotations.
- **Schur complement CG.**
  - **Explicit Schur complement:** `S = L - Gᵀ·G` is assembled explicitly in single precision. Its sparsity pattern is computed once, and its values in every solve. Single precision is enough because the inner CG is an approximate solve: the outer NS iteration uses the double precision blocks.
  - **Resistance boundary conditions:** their preconditioner, which FSILS applies to `G·P`, becomes rank-1 corrections of `S`.
  - **AMG preconditioner:** the CG is preconditioned with a two-level aggregation AMG V-cycle.
    - Aggregates are built once from the graph of `S`.
    - Smoothing is Jacobi before and after the coarse correction. The damping is `1.8/λmax(D⁻¹S)`, with λmax from a power iteration.
    - The coarse solve is exact: the Galerkin coarse matrix is assembled with atomics and inverted on the GPU by a cooperative Gauss-Jordan kernel in every solve.
  - **One kernel for the whole solve:** all CG iterations run in one cooperative kernel, with 5 grid synchronizations per iteration. The reductions use atomic adds into rotating accumulators, and the smoothed vectors are computed on the fly to avoid extra synchronizations.

## How the speedup evolved (iliac_artery, 20 time steps)

| Step | Time | Speedup |
|---|---|---|
| CPU, 10 processes | 105.1 s | 1× |
| GPU NS solver (same algorithm), CPU assembly on one core | slower than CPU (0.76 s vs 0.58 s per Newton iteration) | <1× |
| + GPU assembly | ~23 s | ~4.5× |
| + AMG-preconditioned Schur CG | 17.1 s | 6.1× |
| + device-resident matrix and GPU preconditioner | 12.8 s | 8.2× |
| + one cooperative kernel per Arnoldi step | **11.7 s** | **9.0×** |

With AMG, each Schur CG solve converges to its 10⁻³ tolerance in about 36 iterations instead of stopping at 300 iterations. The outer NS solver and Newton iteration counts did not change.

Final time per Newton iteration (about 62 ms):

| Part | Time |
|---|---|
| Schur complement CG | ~32 ms |
| GMRES | ~7.5 ms |
| Assembly | ~6.6 ms |
| Splitting the matrix into blocks, Schur complement, AMG setup | ~2.6 ms |
| Preconditioner and transfers | ~1 ms |
| Rest (outer NS iteration, host side) | ~12 ms |

## Why the small mesh is limited to about 9×

On the 15k-node mesh one preconditioned CG iteration takes about 110 µs, although its memory traffic would take less than 20 µs at the GPU's bandwidth. The limits:

- **Windows display driver (WDDM).** Each ordinary kernel launch costs about 7 µs. Cooperative kernels avoid most launches (a grid synchronization costs about 1-2 µs), but only about one block per SM can be used efficiently. Too few threads are then active to hide memory latency on a small problem.
- **Problem size.** A sparse product with 15k rows cannot keep the GPU busy, so its cost is dominated by latency, not bandwidth.
- **Double precision on a consumer GPU.** The RTX 3090 delivers about 0.56 TFLOPS in FP64, about the same as the CPU. This mainly affects assembly.

Alternatives that were tried and not kept:

| Alternative | Result |
|---|---|
| CUDA graph of the CG iterations (unpreconditioned) | Same time as the cooperative kernel (~57 µs per iteration) |
| CUDA graph of full-occupancy kernels for the AMG CG | Slower (182 µs vs 110 µs per iteration) |
| More blocks per SM, launch bounds | Slower (more synchronization cost, register spills) |
| Several threads per row, interleaving rows, processing 4 rows at a time | No significant change |
| Jacobi preconditioning of the CG alone | Residual reduction only 3-10× better, still no convergence |
| Additive two-level preconditioner | ~2× more iterations than the multiplicative V-cycle |
| Chebyshev smoothers (degree 2-4) | Fewer iterations, but more sparse products in total |
| Finer aggregates (more coarse unknowns) | Iterations 97 → 71 with 2,300 aggregates; would need more levels |

## Large mesh (1M tetrahedra)

`make_pipe.py` generates a pipe with a radius of 1 cm and a length of 24 cm. The cross-section is a structured grid mapped to a disk, and each hexahedron is split into 6 tetrahedra. The boundary conditions are the same types as in iliac_artery: parabolic inflow, a resistance outlet with backflow stabilization, and a no-slip wall.

| | CPU, 10 processes | GPU |
|---|---|---|
| Solver time, 10 time steps | 1,631 s | 43.9 s |
| Wall-clock time (including reading the 58 MB text mesh) | 1,643 s | 56 s |
| Time per Newton iteration | 17-25 s | ~0.5 s |
| Newton iterations | 94 | 86 |
| Linear solves that did not converge | 14 | 0 |

The unpreconditioned CPU Schur CG gets worse as the mesh grows, while the AMG-preconditioned GPU CG converged in every solve.

Two problems that this mesh revealed were fixed:

- **Smoother damping.** A fixed Jacobi damping of 0.8 diverged (λmax(D⁻¹S) = 2.65 on this mesh, so 0.8·λmax > 2), and every linear solve failed. The damping is now `1.8/λmax`.
- **Coarse matrix size.** With aggregates of a node and its neighbors, the dense coarse matrix would have had about 2,400 unknowns. On meshes with more than 50,000 nodes the aggregates also include the neighbors of the neighbors (672 coarse unknowns here).

## GPU linear solver without GPU assembly

Because the GPU path supports one process, assembly on the CPU runs on one core:

| Mesh | Assembly on one core | GPU solve | Newton iteration | CPU, 10 processes | Speedup |
|---|---|---|---|---|---|
| iliac_artery | ~0.6-0.9 s | ~0.05 s | ~0.7-1 s | 0.58 s | <1× |
| 1M pipe (estimate) | ~7-11 s | ~0.3 s | ~7-11 s | 17-25 s | ~2× |

Offloading only the linear solver is therefore useful only if the assembly stays parallel on the CPU (MPI processes or threads) and feeds one GPU solver. This is not implemented. The matrix transfer per Newton iteration (about 30 MB for iliac_artery, 350 MB for the 1M mesh) would cost 10-30 ms. Estimated speedups: about 4× on iliac_artery and 10-15× on the 1M mesh.

## Applicability to svMultiPhysics in general

Size of the experiment code: about 4,500 lines:

| Part | Lines (approx.) | Physics-specific |
|---|---|---|
| Linear solver (GMRES, Schur CG, AMG, sparse operations, preconditioner, interface) | ~2,500 | No |
| Fluid assembly (generated element routines, kernel, driver) | ~1,000 | Yes |
| Variants that are not used and instrumentation | ~1,000 | Could be removed |

- **Linear solver.** The GPU FSILS solver, preconditioner and device-resident matrix do not depend on the physics: any equation assembled into the FSILS matrix format can use them. It is one component of about 2,000-3,000 lines, which would need MPI halo exchanges for multiple GPUs. The current cooperative-kernel design synchronizes the whole GPU inside a kernel, so it would have to be restructured for that.
- **Assembly.** The element code of svMultiPhysics is large: fluid ~3,100 lines, solid ~6,200 lines plus the material models, electrophysiology ~1,800 lines, plus FSI/ALE, CMM, porous media, heat transfer and the face integrals of the boundary conditions.
  - **Problem:** most of it uses heap-allocated `Array`/`Vector` temporaries, material class hierarchies and `ComMod` lookups, which cannot run on a GPU.
  - **Cost of copies:** porting it as separate CUDA code would mean about 10,000-15,000 lines of a second implementation of the physics. Every change would have to be made twice, with a constant risk of the two versions diverging.
- **Recommended structure for GPU support.** Refactor the element routines to fixed-size, stack-allocated small matrices (for example fixed-size Eigen types), with a host/device annotation, so that the same source compiles for the CPU and for CUDA. Kokkos, already a dependency through Trilinos, is an alternative. Then:
  - **Shared physics code:** a generic GPU assembly driver per element type calls the shared routines. The physics code then exists once, and the GPU-specific code stays at about 3,000-5,000 lines (solver, drivers, halo exchanges).
  - **CPU benefit:** removing the heap allocations per Gauss point would also make the CPU code faster.
  - **Gradual migration:** each physics can be migrated separately, while the others keep using the CPU code.

## Recommendations

In order of value for effort:

1. **AMG-preconditioned Schur CG in the CPU FSILS solver.** It is a small change without code duplication, it helps every user, and it fixes the non-converging Schur CG. It would also show how much of the GPU speedup comes from the algorithm and how much from the hardware.
2. **A GPU linear solver backend**, physics-independent, with multi-GPU support.
3. **GPU assembly physics by physics**, after the single-source refactor of the element routines.
4. **Measurements on Linux and a data-center GPU** (A100/H100): much lower kernel launch latency and high FP64 throughput should improve the results, especially on small meshes and in assembly.
