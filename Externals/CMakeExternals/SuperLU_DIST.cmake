# SuperLU_DIST with ParMETIS, with the options that PETSc uses (PETSc cannot
# download it when using Microsoft compilers)
set(proj SuperLU_DIST)

sv_externals_enabled_depends(_deps MSMPI OpenBLAS METIS ParMETIS)

set(_patch "")
if(MSVC)
  # Fixes for MSVC (file paths, pointer arithmetic, unistd.h)
  set(_patch ${GIT_EXECUTABLE} apply --whitespace=nowarn ${SV_EXTERNALS_PATCH_DIR}/superlu_dist-9.2.1-msvc.patch)
endif()

if(WIN32)
  set(_parmetis_libs "${SV_EXTERNALS_ParMETIS_INSTALL_DIR}/lib/parmetis.lib;${SV_EXTERNALS_METIS_INSTALL_DIR}/lib/metis.lib")
else()
  set(_parmetis_libs "${SV_EXTERNALS_ParMETIS_INSTALL_DIR}/lib/libparmetis.a;${SV_EXTERNALS_METIS_INSTALL_DIR}/lib/libmetis.a")
endif()

ExternalProject_Add(${proj}
  PREFIX ${SV_EXTERNALS_TOPLEVEL_BLD_DIR}/${proj}
  URL ${SV_EXTERNALS_SuperLU_DIST_URL}
  URL_HASH SHA256=${SV_EXTERNALS_SuperLU_DIST_SHA256}
  DOWNLOAD_DIR ${SV_EXTERNALS_DOWNLOAD_DIR}
  PATCH_COMMAND ${_patch}
  CMAKE_COMMAND ${SV_EXTERNALS_CMAKE_COMMAND}
  CMAKE_CACHE_ARGS
    ${SV_EXTERNALS_COMMON_CMAKE_CACHE_ARGS}
    -DCMAKE_INSTALL_PREFIX:PATH=${SV_EXTERNALS_SuperLU_DIST_INSTALL_DIR}
    -DCMAKE_INSTALL_LIBDIR:STRING=lib
    -DUSE_XSDK_DEFAULTS:STRING=YES
    -DXSDK_ENABLE_Fortran:BOOL=OFF
    -Denable_tests:BOOL=OFF
    -Denable_examples:BOOL=OFF
    -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP:BOOL=TRUE
    -DTPL_BLAS_LIBRARIES:STRING=${SV_EXTERNALS_BLAS_LAPACK_LIBRARIES}
    -DTPL_LAPACK_LIBRARIES:STRING=${SV_EXTERNALS_BLAS_LAPACK_LIBRARIES}
    -DTPL_ENABLE_PARMETISLIB:BOOL=TRUE
    "-DTPL_PARMETIS_INCLUDE_DIRS:STRING=${SV_EXTERNALS_ParMETIS_INSTALL_DIR}/include$<SEMICOLON>${SV_EXTERNALS_METIS_INSTALL_DIR}/include"
    "-DTPL_PARMETIS_LIBRARIES:STRING=${_parmetis_libs}"
  DEPENDS ${_deps}
  )
