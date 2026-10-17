# Experimental GPU (CUDA) implementation of the FSILS NS linear solver.
# Built separately with nvcc (e.g. NMake generator), then linked into
# svMultiPhysics with -DSV_GPU_DIR=<build directory of this project>.
cmake_minimum_required(VERSION 3.20)
project(svgpu LANGUAGES CXX CUDA)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_library(svgpu SHARED svgpu_kernels.cu svgpu_ns.cpp svgpu_fluid.cu svgpu_schur.cu svgpu_amg.cu svgpu_gmres.cu svgpu_amg_graph.cu)
target_compile_definitions(svgpu PRIVATE SVGPU_EXPORTS)
target_include_directories(svgpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../ThirdParty/eigen/include/eigen3)
target_link_libraries(svgpu PRIVATE CUDA::cudart_static)
# No fused multiply-add, so that the results are close to the CPU solver
target_compile_options(svgpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false --extended-lambda>)
