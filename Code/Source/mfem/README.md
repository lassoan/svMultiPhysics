# GPU acceleration experiment with MFEM

This directory contains an experimental implementation of the per-time-step computation of the rigid-wall fluid equation with [MFEM](https://mfem.org): element assembly, FSILS diagonal preconditioner and FSILS NS linear solver, written as MFEM kernels (`mfem::forall` with host/device lambdas) on MFEM vectors and arrays. It is the third implementation of the same experiment, after the hand-written CUDA version (branch `gpu-exp`, `Code/Source/gpu`) and the NVIDIA Warp version (branch `warp-exp`, `Code/Source/warp`). The goal was to find out what a portable, library-based implementation costs in performance and in effort compared with the two others. It is not a production feature.

## Summary

| Case | svMultiPhysics, 10 MPI processes | CUDA (`gpu-exp`) | Warp (`warp-exp`) | MFEM, GPU (this branch) | Speedup |
|---|---|---|---|---|---|
| `tests/cases/fluid/iliac_artery`, 75,819 tetrahedra, 20 time steps | 106 s | 11.7 s | 9.1 s | **7.2 s** | **15×** |
| Pipe (`Code/Source/gpu/make_pipe.py`), 1,002,240 tetrahedra, 10 time steps | 1,631 s | 43.9 s | 33.0 s | **32.5 s** | **50×** |

The same MFEM code on the CPU only (MFEM built without CUDA, the library compiled as plain C++; MFEM's `cpu` backend on one thread and `omp` backend on 10 threads), against the svMultiPhysics CPU code of the same build:

| Case | svMultiPhysics, 1 process | svMultiPhysics, 10 MPI processes | MFEM, 1 thread | MFEM, 10 OpenMP threads | MFEM, GPU |
|---|---|---|---|---|---|
| iliac artery, 20 time steps | 592 s | 95 s | 301 s | **76 s** | 7.2 s |
| Pipe, 10 time steps | (not run) | 1,631 s | 2,031 s | **386 s** | 32.5 s |

Hardware: Intel Core i9-10900X (10 cores) and NVIDIA GeForce RTX 3090, Windows 11. CPU times are with the clang-cl build with mimalloc and AVX2 (branch `msvc-build`; the 106 s of the first table is the build without AVX2, used by the two other experiments). The MFEM results agree with the CPU results within 7·10⁻¹⁴ (iliac artery) and 2·10⁻¹⁰ (pipe) relative difference, with the same Newton and linear iteration counts as the Warp version; the CPU backends give identical residuals to the GPU.

Main conclusions:

- **MFEM's kernel model costs nothing in performance once the kernels are designed for it.** The solver has the same structure as the Warp version: fused kernels, device-resident scalars, mapped host memory for the few values the host needs, every phase captured in a CUDA graph. `mfem::forall` kernels can be captured in CUDA graphs (the library is compiled with the per-thread default stream), and `mfem::forall_2D` gives the block-level cooperation that fused reductions need. The MFEM version is the fastest of the three on both meshes, mostly because it combines the small-mesh design of the Warp version with C++ host code (no interpreter overhead) and the SELL layout of the Warp version.
- **MFEM's generic pieces are used for the setup, not for the iterations.** `mfem::SparseMatrix` computes the pattern of the Schur complement (`Aᵀ·A`) and MFEM's memory manager owns every array; the Krylov solvers, the sparse products and the preconditioner are custom kernels, because MFEM's `CGSolver`/`GMRESSolver` synchronize with the host at every dot product (about 100 µs each on Windows), which would cost a factor of 3-5 on the small mesh, and because the FSILS algorithm (block structure, resistance faces, convergence criteria) has no counterpart in MFEM. Without hypre (MPI), MFEM has no AMG; the aggregation AMG of the Warp version was ported (about 450 lines).
- **The same source runs on the CPU, and is faster there than the current CPU code.** With MFEM built without CUDA the library compiles as plain C++ (`SVMFEM_CUDA=OFF`), and `SVMFEM_DEVICE=cpu` or `omp` runs every kernel on the host through MFEM's CPU or OpenMP backend with identical residuals. On 10 threads it beats svMultiPhysics on 10 MPI processes by 1.25× on the iliac artery and 4.2× on the pipe, on one thread it beats one MPI process by 2× on the iliac artery and comes within 25% of 10 MPI processes on the pipe. Most of this is the algorithm (the AMG-preconditioned Schur CG converges in about 120-190 iterations where the unpreconditioned FSILS CG stops at its 300-iteration limit) rather than the kernels: the OpenMP speedup of the solver kernels is only 3.4× on 10 cores (memory bound, atomic reductions), and the assembly with atomic additions gains 6.4×. This is the property that the CUDA experiment recommended for GPU support of the physics code: the element routines exist once, with `MFEM_HOST_DEVICE`, and a CPU-only build needs no CUDA toolkit at all.
- **MFEM on Windows with CUDA is unsupported territory.** MFEM 4.10 does not build with nvcc and MSVC without patches (see below), and nvcc's extended lambdas on Windows have restrictions that the Linux builds never hit (external linkage and public access of the enclosing function, no lambdas in inline header functions). None of this is a problem on Linux.

## Building and running

MFEM must be built with CUDA and without MPI (hypre is not needed). On Windows, MFEM 4.10 has never been built with nvcc: `mfem-msvc-cuda.patch` contains the four kinds of changes that were needed:

- `/wd4819` is passed to nvcc as a bare option, which nvcc takes for an input file (`CMakeLists.txt`);
- the MSVC branch of the reducers uses a non-existent intrinsic (`__lzclz`);
- a namespace-scope `const` variable is used in device code (`kernels.hpp`), and a host lambda uses a device-only `asm` (`bilininteg_hcurl_kernels.hpp`);
- about 90 `static` functions and anonymous namespaces define extended lambdas: nvcc on Windows requires the enclosing function to have external linkage.

Steps (from an *x64 Native Tools Command Prompt for VS 2022*, with the CUDA toolkit in the `PATH`; the toolkit can be installed without administrator rights with pixi, see `Code/Source/gpu/README.md`):

1. Build and install MFEM:
   ```
   git clone --depth 1 --branch v4.10 https://github.com/mfem/mfem.git mfem-src
   git -C mfem-src apply <this directory>/mfem-msvc-cuda.patch
   cmake -S mfem-src -B mfem-build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=mfem-install ^
     -DMFEM_USE_CUDA=YES -DCMAKE_CUDA_ARCHITECTURES=86 -DMFEM_USE_MPI=NO "-DCMAKE_CUDA_FLAGS=-D_USE_MATH_DEFINES" ^
     "-DCMAKE_CXX_FLAGS=-D_USE_MATH_DEFINES" -DMFEM_ENABLE_EXAMPLES=OFF -DMFEM_ENABLE_MINIAPPS=OFF -DMFEM_ENABLE_TESTING=OFF
   python <this directory>/mfem-parallel-compile.py mfem-build 10
   cmake --build mfem-build --target install
   ```
   NMake builds serially and nvcc takes about 25 s per file (250 files); `mfem-parallel-compile.py` runs the compile commands of the generated makefile in parallel (8 minutes with 10 jobs), after which the NMake build only links and installs. (The CUDA Visual Studio integration, which would allow MSBuild to build in parallel, is not part of the conda-forge toolkit.)
2. Build `svmfem.dll` from this directory:
   ```
   cmake -S Code/Source/mfem -B svmfem-build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DMFEM_DIR=<mfem-install>/lib/cmake/mfem
   cmake --build svmfem-build
   ```
3. Configure svMultiPhysics with `-DSV_MFEM_DIR=<path of svmfem-build>` and build it (`svmfem.dll` is copied next to `svmultiphysics.exe`; the CUDA runtime, cuSPARSE and cuBLAS DLLs of the toolkit must be in the `PATH`).
4. Run with one process and the environment variables `SV_MFEM=1` (GPU linear solver) and `SV_MFEM_ASSEM=1` (GPU assembly). Without these variables, the CPU code is used.

CPU-only build (no CUDA toolkit; MFEM with the OpenMP backend, built in parallel with MSBuild since no nvcc is involved; the LLVM OpenMP runtime is needed for MFEM's `min`/`max` reductions with MSVC):
```
cmake -S mfem-src -B mfem-build-cpu -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=mfem-install-cpu -DMFEM_USE_CUDA=NO ^
  -DMFEM_USE_OPENMP=YES "-DOpenMP_CXX_FLAGS=/openmp:llvm" -DOpenMP_CXX_LIB_NAMES=libomp -DOpenMP_libomp_LIBRARY=libomp.lib ^
  -DMFEM_USE_MPI=NO "-DCMAKE_CXX_FLAGS=-D_USE_MATH_DEFINES /EHsc /DWIN32 /D_WINDOWS" -DMFEM_ENABLE_EXAMPLES=OFF -DMFEM_ENABLE_MINIAPPS=OFF -DMFEM_ENABLE_TESTING=OFF
cmake --build mfem-build-cpu --config Release --target install --parallel 10
cmake -S Code/Source/mfem -B svmfem-cpu-build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DSVMFEM_CUDA=OFF ^
  "-DOpenMP_CXX_FLAGS=/openmp:llvm" -DOpenMP_CXX_LIB_NAMES=libomp -DOpenMP_libomp_LIBRARY=libomp.lib -DMFEM_DIR=<mfem-install-cpu>/lib/cmake/mfem
cmake --build svmfem-cpu-build
```
The resulting `svmfem.dll` has the same interface and can replace the CUDA one next to `svmultiphysics.exe`; run with `SVMFEM_DEVICE=omp` and `OMP_NUM_THREADS=<cores>` (or `SVMFEM_DEVICE=cpu` for one thread).

Other environment variables: `SVMFEM_DEVICE=cpu` (or `omp` with the CPU-only build) runs the same kernels on the host through MFEM's CPU or OpenMP backend, `SVMFEM_NOGRAPH=1` launches the kernels one by one instead of CUDA graphs, `SVMFEM_TIMING=1` and `SV_LS_STATS=1` print timings and iteration counts, `SV_GPU_VERBOSE=1` prints the setup, `SVMFEM_DEBUG=1` traces the phases, `SVMFEM_AMG_W`, `SVMFEM_AMG_WC`, `SVMFEM_AMG_CAND`, `SVMFEM_AMG_CMAX` and `SVMFEM_CG_CHUNK` are the AMG and CG parameters of the Warp version.

Supported cases are those of the other two experiments: one process; fluid equation with linear tetrahedra, VMS stabilization, one domain with constant viscosity, no moving mesh, no RIS valves, resistance (coupled Neumann) faces without caps, FSILS preconditioner and `NS` linear solver. Other cases use the CPU code or stop with an error.

## Files

| File | Contents |
|---|---|
| `svmfem.h` | C interface of the library (the same functions as `svgpu.h`) |
| `svmfem_common.hpp`, `svmfem_device.cu` | MFEM device setup, kernel launch helpers, block reductions into device scalars, SELL matrix layout, mapped host memory, CUDA graphs |
| `svmfem_fluid.cu` | Element assembly, device resident matrix, FSILS diagonal preconditioner |
| `fluid_element.inc`, `gen_fluid.py` | Element routines, generated from `physics/fluid/src/fluid.cpp` with `MFEM_HOST_DEVICE` |
| `svmfem_ns.cu` | FSILS NS solver: outer iteration, GMRES, Schur complement, AMG preconditioned CG |
| `svmfem_amg.hpp`, `svmfem_amg.cu` | Aggregation AMG (host setup, device numeric phase and V-cycle) |
| `svmfem_lstsq.cpp` | Least squares solve of the outer iteration (Eigen, host) |
| `mfem-msvc-cuda.patch`, `mfem-parallel-compile.py` | MFEM build on Windows with nvcc |

Changes outside this directory: `solver/svmfem_host.cpp` (host side of the hooks), `linear_solver/solve.cpp`, `solver/FsilsLinearAlgebra.cpp`, `solver/physics/fluid/src/fluid.cpp`, `solver/main.cpp` and the CMake files (`SV_MFEM_DIR`): the same hook points as the two other experiments.

## Implementation

The algorithms are those of the Warp version (see `Code/Source/warp/README.md` for the measurements behind the design): GPU assembly with one thread per element and atomic additions into the device resident FSILS matrix; the FSILS NS solver with GMRES on the momentum block (resistance faces through precomputed `Kᵀv` and face Gram matrices, classical Gram-Schmidt), the explicit Schur complement `L + GᵀG` in single precision, CG preconditioned with a two-level aggregation AMG (improved-candidate prolongator weights, damped Jacobi smoothing with the damping from a power iteration, dense coarse inverse by Gauss-Jordan steps on the device); every phase a CUDA graph; scalars through mapped pinned memory.

How the pieces map to MFEM:

- **Kernels** are `mfem::forall(n, [=] MFEM_HOST_DEVICE (int i) {...})`. The lambdas capture raw device pointers obtained from `mfem::Vector::Read()/Write()` and `mfem::Array<T>`; MFEM's memory manager allocates and tracks the host and device copies.
- **Reductions into device scalars** (dot products, norms, face dots, the Gram matrices) use `mfem::forall_2D(nblocks, 256, 1, ...)` with `MFEM_SHARED`, `MFEM_FOREACH_THREAD` and `MFEM_SYNC_THREAD`: a tree reduction in shared memory per block and one `AtomicAdd` per block into a slot of the device scalar array (`launch_red` in `svmfem_common.hpp`, `launch_red_multi` for several values). On the CPU device the same code runs as loops. The kernels of the CG iteration and of the Arnoldi step read these scalars directly (`alpha = sc[RZ]/sc[PQ]`), so no value crosses to the host inside a phase.
- **CUDA graphs**: the library is compiled with `--default-stream=per-thread`, so `mfem::forall` launches go to a stream that can be captured (`cudaStreamBeginCapture(cudaStreamPerThread)`); the legacy default stream cannot be captured. Everything inside a captured phase must be an `mfem::forall` from this library: MFEM's own vector operations (`Vector::operator*`, `operator=`) synchronize or copy and would break the capture, and every device allocation must happen before (`dalloc` allocates eagerly).
- **Setup with MFEM's sparse algebra**: the pattern of the Schur complement is `Mult(Transpose(A), A)` on an `mfem::SparseMatrix` with the pattern of the FSILS matrix, sorted with `SortColumnIndices()` (which runs on the device with cuSPARSE: the result must be read with `HostReadJ()`). The aggregation and the Galerkin patterns are host code.
- **Matrix layouts**: the block matrices (K, G, D, L) and the Schur complement are gathered from their CSR values into a sliced ELLPACK layout (SELL-32, widths rounded to multiples of 4) in every solve, so that one thread per row reads coalesced memory. With plain CSR and one thread per row the pipe case took 71 s instead of 32.5 s (CG iteration 2.3 ms instead of 0.6 ms).
- **Host-side scalars**: the last kernel of a phase writes the values the host needs (norms, Hessenberg column, Gram matrix) into mapped pinned memory (`cudaHostAlloc` with `cudaHostAllocMapped`) in 64-byte vectors; the host reads them after one `cudaStreamSynchronize`. Coefficients computed by the host (Givens solution, outer iteration) are read by a kernel from mapped memory. Both have no MFEM counterpart and are the only CUDA calls outside the device setup.
- **Exit**: the device objects are released explicitly at the end of `main` (`svmfem_finalize`): MFEM's memory manager otherwise frees device memory from a static destructor after the CUDA runtime has shut down, which MFEM reports as a fatal error.

Per Newton iteration (solve and assembly, from `SVMFEM_TIMING=1`):

| | iliac artery (15,334 nodes) | pipe (181,875 nodes) |
|---|---|---|
| Assembly | 7.2 ms | 77 ms |
| Setup (blocks, SELL gathers, Schur complement, AMG, coarse inverse) | 3.0 ms | 17 ms |
| GMRES (52 and 77 steps) | 5.6 ms (0.11 ms/step) | 61 ms (0.79 ms/step) |
| Schur complement CG (118 and 191 iterations) | 11.8 ms (0.10 ms/iteration) | 116 ms (0.61 ms/iteration) |
| Total solve | 21.7 ms | 203 ms |
| Warp version, total solve | 24.3 ms | 171 ms |

Without CUDA graphs (`SVMFEM_NOGRAPH=1`) the iliac case takes 8.7 s instead of 7.2 s (29 ms instead of 22 ms per solve): a kernel launched directly costs about 9 µs on this machine (WDDM driver), a kernel inside a graph about 3 µs, and the solve of the small mesh runs about 2,000 kernels per Newton iteration.

The same kernels on the CPU (per Newton iteration; the svMultiPhysics CPU code takes 3.2 s on one process and 0.52 s on 10 processes on the iliac artery):

| | iliac artery, 1 thread | iliac artery, 10 OpenMP threads | pipe, 10 OpenMP threads |
|---|---|---|---|
| Assembly | 527 ms | 82 ms | 904 ms |
| Solve | 1,089 ms | 316 ms | 3,450 ms |
| of which Schur complement CG | 494 ms (4.2 ms/iteration) | 169 ms (1.4 ms/iteration) | 2,029 ms (10.6 ms/iteration) |
| of which GMRES | 398 ms (7.7 ms/step) | 103 ms (2.0 ms/step) | 1,077 ms (13.9 ms/step) |

The GPU kernels were designed for the GPU (one thread per row, SELL layout, atomic reductions per 256-element block, float AMG): on the CPU they are correct but not tuned, which is the price of single-source code; an OpenMP-aware layout (rows per thread, no atomics in the reductions) would help the CPU without changing the algorithm.

## nvcc and MSVC pitfalls

These cost most of the debugging time; none of them exists with gcc or clang on Linux.

- The enclosing function of an extended (`__host__ __device__`) lambda must have external linkage and, for a member function, public access: no `static` functions, no anonymous namespaces, no private members. MFEM itself violates this in about 90 places (hence the patch).
- An extended lambda defined in an `inline` function of a header compiles, but its host-side wrapper calls a null function pointer at run time (access violation). The small kernels of `svmfem_common.hpp` are therefore defined in `svmfem_device.cu`.
- A bare `/option` in the nvcc command line is taken for an input file; `_USE_MATH_DEFINES` must be defined for `M_PI`; `asm("trap;")` in host code is rejected by the MSVC front end.
- NMake builds serially; MSBuild would need the CUDA Visual Studio integration, which the conda-forge toolkit does not include.

## Comparison of the three implementations

| | CUDA (`gpu-exp`, fluid part) | Warp (`warp-exp`) | MFEM (this branch) |
|---|---|---|---|
| Language | C++/CUDA, cooperative kernels | Python, Warp kernels, CUDA graphs | C++, MFEM kernels, CUDA graphs |
| Lines (solver, assembly, interface) | ~3,300 | ~2,700 (Python) | ~2,800 + 270 host hooks |
| iliac artery, 20 steps | 11.7 s | 9.1 s | 7.2 s |
| Pipe, 10 steps | 43.9 s | 33.0 s | 32.5 s |
| Runs on the CPU from the same source | no | no (Warp has a CPU backend, not used) | yes (`SVMFEM_DEVICE=cpu` or `omp`; 76 s on 10 cores for the iliac case) |
| Portability | NVIDIA only | NVIDIA only (Warp) | CUDA, HIP, OpenMP, CPU (MFEM backends) |
| Dependencies | CUDA toolkit | Python, warp-lang, NumPy, SciPy, PyAMG (optional) | CUDA toolkit, MFEM (patched on Windows) |
| Build on Windows | nvcc, 1 minute | none (JIT at run time, 1 s startup) | MFEM 8 minutes (parallel) + nvcc 2 minutes |
| Element routines | generated, `__device__` | hand-ported to Warp kernels | generated, `MFEM_HOST_DEVICE` |
| What the library provided | nothing beyond the runtime | kernel language, graphs, tiles, interop with NumPy/SciPy/PyAMG | memory manager, forall, sparse matrices, device abstraction |

The three implementations use the same algorithms and reach the same iteration counts, so the differences are implementation overheads: the Python host code of the Warp version (about 30 µs per launch, 0.5-1 ms per Newton iteration of host work), the host synchronizations of the cooperative-kernel CUDA version on the small mesh, and the memory layouts on the large mesh.

What MFEM would add for a production implementation: with MPI and hypre, `HypreParMatrix` and `HypreBoomerAMG` replace the custom AMG and give multi-GPU parallelism through MFEM's partitioned matrices; the HIP backend runs the same code on AMD GPUs; and the physics could use MFEM's finite element classes (`Mesh`, `FiniteElementSpace`, integrators with partial assembly) instead of hand-written element loops, which would be a new discretization rather than a port of svMultiPhysics's.
