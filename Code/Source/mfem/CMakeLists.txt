# Experimental GPU acceleration of the fluid equation with MFEM: assembly,
# preconditioner and FSILS NS linear solver as mfem::forall kernels on the
# device resident matrix. Built separately with nvcc and MSVC (MFEM built
# with MFEM_USE_CUDA), then linked into svMultiPhysics with
# -DSV_MFEM_DIR=<build directory of this project>.
cmake_minimum_required(VERSION 3.20)
# SVMFEM_CUDA=OFF: CPU-only build (against an MFEM built without CUDA, for
# the CPU and OpenMP backends); the .cu sources are then compiled as C++
option(SVMFEM_CUDA "Build with CUDA (nvcc)" ON)
if(SVMFEM_CUDA)
  project(svmfem LANGUAGES CXX CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
  endif()
  set(CMAKE_CUDA_STANDARD 17)
else()
  project(svmfem LANGUAGES CXX)
endif()
set(CMAKE_CXX_STANDARD 17)

# MFEM_DIR: the lib/cmake/mfem directory of the MFEM installation
find_package(mfem REQUIRED NAMES MFEM HINTS "${MFEM_DIR}")
message(STATUS "MFEM ${MFEM_VERSION} from ${MFEM_DIR} (CUDA ${MFEM_USE_CUDA})")

set(SVMFEM_SOURCES svmfem_device.cu svmfem_fluid.cu svmfem_amg.cu svmfem_ns.cu svmfem_fsi.cu)
if(NOT SVMFEM_CUDA)
  set_source_files_properties(${SVMFEM_SOURCES} PROPERTIES LANGUAGE CXX)
  if(MSVC)
    set_source_files_properties(${SVMFEM_SOURCES} PROPERTIES COMPILE_OPTIONS "/TP")
  endif()
  find_package(OpenMP)
endif()
add_library(svmfem SHARED ${SVMFEM_SOURCES} svmfem_lstsq.cpp)
if(NOT SVMFEM_CUDA AND OpenMP_CXX_FOUND)
  target_link_libraries(svmfem PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_definitions(svmfem PRIVATE SVMFEM_EXPORTS _USE_MATH_DEFINES)
target_include_directories(svmfem PRIVATE ${MFEM_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../../ThirdParty/eigen/include/eigen3)
target_link_libraries(svmfem PRIVATE mfem)
# No fused multiply-add, so that the results are close to the CPU solver;
# per-thread default stream, so that mfem::forall launches can be captured
# in CUDA graphs
# Symbols for crash analysis (SVMFEM_SYMBOLS=ON)
option(SVMFEM_SYMBOLS "Debug symbols (PDB) of the release build" OFF)
if(SVMFEM_SYMBOLS)
  target_compile_options(svmfem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zi> $<$<COMPILE_LANGUAGE:CXX>:/Zi>)
  target_link_options(svmfem PRIVATE /DEBUG)
endif()
target_compile_options(svmfem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false --extended-lambda --expt-relaxed-constexpr --default-stream=per-thread -Xcompiler=/wd4819>)
