#-----------------------------------------------------------------------------
# Dependencies that the superbuild can download and build.
#
# This file is included by the top-level CMakeLists.txt (to pass the install
# locations of the built dependencies to svMultiPhysics) and by
# Externals/CMakeLists.txt (to build them). Each dependency <name> is built
# when SV_EXTERNALS_ENABLE_<name> is ON, by Externals/CMakeExternals/<name>.cmake,
# and installed into ${SV_EXTERNALS_INSTALL_DIR}/<name>.
#
# Versions and options follow Docker/ubuntu/dockerfile where possible.
#-----------------------------------------------------------------------------

# Dependencies in build order
set(SV_EXTERNALS_DEPENDENCIES
  MSMPI       # MS-MPI SDK and runtime (Windows only)
  OpenBLAS    # BLAS and LAPACK (prebuilt package on Windows)
  Flang       # LLVM Flang Fortran compiler (Windows only, used for ScaLAPACK, MUMPS and genBC)
  Cygwin      # Unix tools for building PETSc (Windows only)
  VTK
  Boost
  HDF5
  HYPRE
  Trilinos
  METIS       # PETSc forks of METIS and ParMETIS (used by PETSc, MUMPS and SuperLU_DIST)
  ParMETIS
  SuiteSparse
  SuperLU_DIST
  ScaLAPACK
  Eigen
  MUMPS       # source package, built by PETSc
  PETSc
  svZeroDSolver
  mimalloc    # memory allocator (faster than the default allocator on Windows)
  )

set(SV_EXTERNALS_INSTALL_DIR "${CMAKE_BINARY_DIR}/Externals-install" CACHE PATH
  "Directory where the dependencies built by the superbuild are installed")
set(SV_EXTERNALS_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/Externals-download" CACHE PATH
  "Directory where the source packages of the dependencies are downloaded")

foreach(proj ${SV_EXTERNALS_DEPENDENCIES})
  set(SV_EXTERNALS_${proj}_INSTALL_DIR "${SV_EXTERNALS_INSTALL_DIR}/${proj}")
endforeach()

#-----------------------------------------------------------------------------
# Versions, download locations and SHA256 hashes

set(SV_EXTERNALS_MSMPI_SDK_MSI "${CMAKE_CURRENT_LIST_DIR}/../../Code/ThirdParty/msmpi/Installers/msmpisdk.msi")
set(SV_EXTERNALS_MSMPI_RUNTIME_URL "https://conda.anaconda.org/conda-forge/win-64/msmpi-10.1.2-h3502643_2.tar.bz2")
set(SV_EXTERNALS_MSMPI_RUNTIME_SHA256 "0d1c78b7560fe13416726acf82737646e08de8c2f2d1f9942ab8b2faa30f93d8")

set(SV_EXTERNALS_OpenBLAS_VERSION "0.3.34")
set(SV_EXTERNALS_OpenBLAS_WINDOWS_URL "https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.34/OpenBLAS-0.3.34-x64.zip")
set(SV_EXTERNALS_OpenBLAS_WINDOWS_SHA256 "e9cb6134541f36c27346d5fc5995652f060fba227cebbbabcbda5a5a44d7c76b")
set(SV_EXTERNALS_OpenBLAS_URL "https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.34/OpenBLAS-0.3.34.tar.gz")

# LLVM Flang 20 from conda-forge (the version 23 package lacks the Fortran intrinsic module files)
set(SV_EXTERNALS_Flang_PACKAGES
  "https://conda.anaconda.org/conda-forge/win-64/flang-20.1.8-h1b5488d_0.conda|40cd9f5b6080da3f2a158ad0c7fbb7577299a4f6436e73a4d0166af1f9c4ac11"
  "https://conda.anaconda.org/conda-forge/win-64/libflang-20.1.8-hac47afa_0.conda|7c0305d656bc015d01ab5ebf99dbe273fddb8ce452e45a03aa4d25aafa972c44"
  "https://conda.anaconda.org/conda-forge/noarch/compiler-rt_win-64-20.1.8-h49e36cd_1.conda|6685e190d417106c4bc723571f85bf7d1579333e86a12fc16b4b29ddb8500ca0"
  "https://conda.anaconda.org/conda-forge/win-64/libzlib-1.3.2-hfd05255_3.conda|0629c2cc0404d3bb29d6baa7b4ba62da80797015e86de050db81ea5a07050527"
  "https://conda.anaconda.org/conda-forge/win-64/zstd-1.5.7-h534d264_7.conda|ca7daae4f218a11fab82cc2857f0ea518ec3f46acec60490485347a4c22c6b3e"
  )

set(SV_EXTERNALS_Cygwin_SETUP_URL "https://cygwin.com/setup-x86_64.exe")
set(SV_EXTERNALS_Cygwin_MIRROR "https://mirrors.kernel.org/sourceware/cygwin/" CACHE STRING "Cygwin package mirror")
set(SV_EXTERNALS_Cygwin_PACKAGES "python3,make,diffutils,cmake,git,patch")

set(SV_EXTERNALS_VTK_VERSION "9.3.0")
set(SV_EXTERNALS_VTK_URL "https://www.vtk.org/files/release/9.3/VTK-9.3.0.tar.gz")
set(SV_EXTERNALS_VTK_SHA256 "fdc7b9295225b34e4fdddc49cd06e66e94260cb00efee456e0f66568c9681be9")

set(SV_EXTERNALS_Boost_VERSION "1.89.0")
set(SV_EXTERNALS_Boost_URL "https://archives.boost.io/release/1.89.0/source/boost_1_89_0.tar.gz")
set(SV_EXTERNALS_Boost_SHA256 "9de758db755e8330a01d995b0a24d09798048400ac25c03fc5ea9be364b13c93")

set(SV_EXTERNALS_HDF5_VERSION "2.2.0")
set(SV_EXTERNALS_HDF5_URL "https://github.com/HDFGroup/hdf5/archive/refs/tags/2.2.0.tar.gz")
set(SV_EXTERNALS_HDF5_SHA256 "5b8d75125ae7b4fef55d3b39e8d3dbfd238cdf63b6518afd67fa69ac80f06542")

# hypre 2.x: Trilinos 16 (Ifpack2) expects the hypre 2 headers
set(SV_EXTERNALS_HYPRE_VERSION "2.33.0")
set(SV_EXTERNALS_HYPRE_URL "https://github.com/hypre-space/hypre/archive/refs/tags/v2.33.0.tar.gz")
set(SV_EXTERNALS_HYPRE_SHA256 "0f9103c34bce7a5dcbdb79a502720fc8aab4db9fd0146e0791cde7ec878f27da")

# Trilinos 16: Trilinos 17 requires C++20, svMultiPhysics uses C++17
set(SV_EXTERNALS_Trilinos_VERSION "16.2.2")
set(SV_EXTERNALS_Trilinos_URL "https://github.com/trilinos/Trilinos/archive/refs/tags/trilinos-release-16-2-2.tar.gz")
set(SV_EXTERNALS_Trilinos_SHA256 "211810d0f642736ac93ca500c3e28c72a9e8fa9991a757d60c0ecc85108e35d4")

# The versions that PETSc 3.26 downloads
set(SV_EXTERNALS_METIS_URL "https://bitbucket.org/petsc/pkg-metis/get/v5.1.0-p12.tar.gz")
set(SV_EXTERNALS_METIS_SHA256 "2f16abe87394d4fd1430d66a8995076c352af40f7a4d1c5300b3b1cc9d545663")
set(SV_EXTERNALS_ParMETIS_URL "https://bitbucket.org/petsc/pkg-parmetis/get/v4.0.3-p9.tar.gz")
set(SV_EXTERNALS_ParMETIS_SHA256 "612717e85992c984f09b0f5670be421bbb90a4c04145ab5b9a3358b92765d891")

set(SV_EXTERNALS_SuiteSparse_URL "https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v7.14.1.tar.gz")
set(SV_EXTERNALS_SuiteSparse_SHA256 "81e560e1f74546df139edb765b3f5bc865866da23062312ffe8fd821063c8397")

set(SV_EXTERNALS_SuperLU_DIST_URL "https://github.com/xiaoyeli/superlu_dist/archive/refs/tags/v9.2.1.tar.gz")
set(SV_EXTERNALS_SuperLU_DIST_SHA256 "c80a1c2edaaa451ee9a54e005e5f3f56dc55cabe2b0a8d7acf5a1447a648157a")

set(SV_EXTERNALS_ScaLAPACK_URL "https://github.com/Reference-ScaLAPACK/scalapack/archive/refs/tags/v2.2.3.tar.gz")
set(SV_EXTERNALS_ScaLAPACK_SHA256 "5d93701eca663925e98010dd8d0f45fd79b2191d74e5afa5711d587370a8b9dd")

set(SV_EXTERNALS_Eigen_URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.tar.gz")
set(SV_EXTERNALS_Eigen_SHA256 "b93c667d1b69265cdb4d9f30ec21f8facbbe8b307cf34c0b9942834c6d4fdbe2")

set(SV_EXTERNALS_MUMPS_VERSION "5.9.1")
set(SV_EXTERNALS_MUMPS_URL "https://mumps-solver.org/MUMPS_5.9.1.tar.gz")
set(SV_EXTERNALS_MUMPS_SHA256 "659c9b57646b5a003ac618baa1faf9dd2044e46c732b3daaccbc7158003e1b46")

set(SV_EXTERNALS_PETSc_VERSION "3.26.0")
set(SV_EXTERNALS_PETSc_URL "https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.26.0.tar.gz")
set(SV_EXTERNALS_PETSc_SHA256 "f5230023e6e22ee607802a13c82bc25f3d81e71654ad386a5b9bdff17fed93df")

set(SV_EXTERNALS_svZeroDSolver_GIT_URL "https://github.com/SimVascular/svZeroDSolver.git")
set(SV_EXTERNALS_svZeroDSolver_GIT_TAG "v3.2")
# The test cases expect svZeroDSolver in the source tree (as in the CI)
set(SV_EXTERNALS_svZeroDSolver_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../svZeroDSolver")

set(SV_EXTERNALS_mimalloc_VERSION "2.2.4")
set(SV_EXTERNALS_mimalloc_URL "https://github.com/microsoft/mimalloc/archive/refs/tags/v2.2.4.tar.gz")
set(SV_EXTERNALS_mimalloc_SHA256 "754a98de5e2912fddbeaf24830f982b4540992f1bab4a0a8796ee118e0752bda")

#-----------------------------------------------------------------------------
# Locations of the built dependencies

set(SV_EXTERNALS_MSMPI_SDK_DIR "${SV_EXTERNALS_MSMPI_INSTALL_DIR}/sdk/PFiles/Microsoft SDKs/MPI")
set(SV_EXTERNALS_MSMPI_INCLUDE_DIR "${SV_EXTERNALS_MSMPI_SDK_DIR}/Include")
# mpif.h (Fortran) includes mpifptr.h, which is in Include/x64
set(SV_EXTERNALS_MSMPI_INCLUDE_X64_DIR "${SV_EXTERNALS_MSMPI_SDK_DIR}/Include/x64")
set(SV_EXTERNALS_MSMPI_LIB_DIR "${SV_EXTERNALS_MSMPI_SDK_DIR}/Lib/x64")
set(SV_EXTERNALS_MSMPI_BIN_DIR "${SV_EXTERNALS_MSMPI_INSTALL_DIR}/runtime/Library/bin")
set(SV_EXTERNALS_Flang_BIN_DIR "${SV_EXTERNALS_Flang_INSTALL_DIR}/Library/bin")
set(SV_EXTERNALS_Flang_LIB_DIR "${SV_EXTERNALS_Flang_INSTALL_DIR}/Library/lib")
if(WIN32)
  set(SV_EXTERNALS_OpenBLAS_LIBRARY "${SV_EXTERNALS_OpenBLAS_INSTALL_DIR}/lib/libopenblas.lib")
else()
  set(SV_EXTERNALS_OpenBLAS_LIBRARY "${SV_EXTERNALS_OpenBLAS_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}openblas${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

# CMake command for building dependencies that use MPI. With MS-MPI, CMake's
# FindMPI uses the MSMPI_INC and MSMPI_LIB64 environment variables.
set(SV_EXTERNALS_CMAKE_COMMAND "${CMAKE_COMMAND}")
if(WIN32 AND SV_EXTERNALS_ENABLE_MSMPI)
  set(SV_EXTERNALS_CMAKE_COMMAND "${CMAKE_COMMAND}" -E env
    "MSMPI_INC=${SV_EXTERNALS_MSMPI_INCLUDE_DIR}/"
    "MSMPI_LIB64=${SV_EXTERNALS_MSMPI_LIB_DIR}/"
    "${CMAKE_COMMAND}")
endif()

# BLAS/LAPACK used by the dependencies and by svMultiPhysics
if(SV_EXTERNALS_ENABLE_OpenBLAS)
  set(SV_EXTERNALS_BLAS_LAPACK_LIBRARIES "${SV_EXTERNALS_OpenBLAS_LIBRARY}")
else()
  set(SV_EXTERNALS_BLAS_LAPACK_LIBRARIES "" CACHE STRING "BLAS/LAPACK libraries to use for the dependencies")
endif()

set(SV_EXTERNALS_PATCH_DIR "${CMAKE_CURRENT_LIST_DIR}/../Patches")

#-----------------------------------------------------------------------------
# Helpers for the Externals/CMakeExternals/<name>.cmake files

# Set var to the enabled dependencies among the given ones (for the DEPENDS
# option of ExternalProject_Add)
function(sv_externals_enabled_depends var)
  set(deps)
  foreach(dep ${ARGN})
    if(SV_EXTERNALS_ENABLE_${dep})
      list(APPEND deps ${dep})
    endif()
  endforeach()
  set(${var} ${deps} PARENT_SCOPE)
endfunction()

if(NOT SV_EXTERNALS_BUILD_TYPE)
  set(SV_EXTERNALS_BUILD_TYPE Release)
endif()

# CMake cache arguments common to all dependencies built with CMake
set(SV_EXTERNALS_COMMON_CMAKE_CACHE_ARGS
  -DCMAKE_BUILD_TYPE:STRING=${SV_EXTERNALS_BUILD_TYPE}
  -DBUILD_SHARED_LIBS:BOOL=OFF
  -DBUILD_TESTING:BOOL=OFF
  -DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.5
  )
if(MSVC)
  # Same C runtime (/MD) as svMultiPhysics
  list(APPEND SV_EXTERNALS_COMMON_CMAKE_CACHE_ARGS
    -DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=MultiThreadedDLL)
endif()

# Patch tool: git apply works on Windows, Linux and macOS
find_package(Git QUIET)
