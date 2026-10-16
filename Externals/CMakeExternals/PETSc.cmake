# PETSc with the external packages of Docker/ubuntu/dockerfile.
#
# On Linux and macOS, PETSc downloads and builds the external packages itself.
# On Windows, PETSc's build system runs in Cygwin with the MSVC compilers; the
# packages that PETSc cannot download and build with Microsoft compilers are
# built by the superbuild and passed in (see petsc-windows.sh.in).
set(proj PETSc)

include(ProcessorCount)
ProcessorCount(_ncpu)
if(_ncpu EQUAL 0)
  set(_ncpu 1)
endif()

if(WIN32)
  sv_externals_enabled_depends(_deps MSMPI OpenBLAS Flang Cygwin METIS ParMETIS
    SuiteSparse SuperLU_DIST ScaLAPACK HYPRE Eigen MUMPS)
  foreach(dep MSMPI OpenBLAS Flang Cygwin METIS ParMETIS SuiteSparse SuperLU_DIST ScaLAPACK HYPRE Eigen MUMPS)
    if(NOT SV_EXTERNALS_ENABLE_${dep})
      message(FATAL_ERROR "Building PETSc on Windows requires SV_EXTERNALS_ENABLE_${dep}=ON")
    endif()
  endforeach()

  # Paths in Cygwin notation (/cygdrive/<drive>/...)
  function(sv_externals_cygwin_path var path)
    string(SUBSTRING "${path}" 0 1 drive)
    string(TOLOWER "${drive}" drive)
    string(SUBSTRING "${path}" 2 -1 rest)
    set(${var} "/cygdrive/${drive}${rest}" PARENT_SCOPE)
  endfunction()
  set(_src ${SV_EXTERNALS_TOPLEVEL_BLD_DIR}/${proj}/src/${proj})
  sv_externals_cygwin_path(CYG_PETSC_SRC "${_src}")
  sv_externals_cygwin_path(CYG_PETSC_PREFIX "${SV_EXTERNALS_PETSc_INSTALL_DIR}")
  sv_externals_cygwin_path(CYG_MSMPI_SDK "${SV_EXTERNALS_MSMPI_SDK_DIR}")
  sv_externals_cygwin_path(CYG_MSMPI_BIN "${SV_EXTERNALS_MSMPI_BIN_DIR}")
  sv_externals_cygwin_path(CYG_FLANG_LIB "${SV_EXTERNALS_Flang_LIB_DIR}")
  sv_externals_cygwin_path(CYG_MUMPS_PACKAGE
    "${SV_EXTERNALS_MUMPS_INSTALL_DIR}/MUMPS_${SV_EXTERNALS_MUMPS_VERSION}.tar.gz")
  foreach(dep OpenBLAS METIS ParMETIS SuiteSparse SuperLU_DIST ScaLAPACK HYPRE Eigen)
    sv_externals_cygwin_path(CYG_${dep} "${SV_EXTERNALS_${dep}_INSTALL_DIR}")
  endforeach()
  configure_file(${SV_EXTERNALS_CMAKE_DIR}/petsc-windows.sh.in
    ${CMAKE_BINARY_DIR}/petsc-windows.sh @ONLY NEWLINE_STYLE UNIX)
  sv_externals_cygwin_path(_script "${CMAKE_BINARY_DIR}/petsc-windows.sh")

  ExternalProject_Add(${proj}
    PREFIX ${SV_EXTERNALS_TOPLEVEL_BLD_DIR}/${proj}
    URL ${SV_EXTERNALS_PETSc_URL}
    URL_HASH SHA256=${SV_EXTERNALS_PETSc_SHA256}
    DOWNLOAD_DIR ${SV_EXTERNALS_DOWNLOAD_DIR}
    SOURCE_DIR ${_src}
    BUILD_IN_SOURCE TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${SV_EXTERNALS_Cygwin_INSTALL_DIR}/bin/bash.exe --login -c ${_script}
    INSTALL_COMMAND ""
    DEPENDS ${_deps}
    )
else()
  sv_externals_enabled_depends(_deps OpenBLAS)
  set(_blas_lapack)
  if(SV_EXTERNALS_BLAS_LAPACK_LIBRARIES)
    set(_blas_lapack --with-blaslapack-lib=${SV_EXTERNALS_BLAS_LAPACK_LIBRARIES})
  endif()

  ExternalProject_Add(${proj}
    PREFIX ${SV_EXTERNALS_TOPLEVEL_BLD_DIR}/${proj}
    URL ${SV_EXTERNALS_PETSc_URL}
    URL_HASH SHA256=${SV_EXTERNALS_PETSc_SHA256}
    DOWNLOAD_DIR ${SV_EXTERNALS_DOWNLOAD_DIR}
    BUILD_IN_SOURCE TRUE
    CONFIGURE_COMMAND ./configure
      --prefix=${SV_EXTERNALS_PETSc_INSTALL_DIR}
      --with-debugging=0
      --with-precision=double
      --with-strict-petscerrorcode
      ${_blas_lapack}
      --download-suitesparse
      --download-mumps
      --download-superlu
      --download-superlu_dist
      --download-ml
      --download-eigen
      --download-hypre
      --download-scalapack
      --download-metis
      --download-parmetis
    BUILD_COMMAND make PETSC_DIR=<SOURCE_DIR> PETSC_ARCH=arch-opt -j${_ncpu} all
    INSTALL_COMMAND make PETSC_DIR=<SOURCE_DIR> PETSC_ARCH=arch-opt install
    DEPENDS ${_deps}
    )
endif()
