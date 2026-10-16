# ScaLAPACK (Fortran). On Windows it is built with LLVM Flang (the MSVC C
# compiler for the C parts) and the NMake Makefiles generator, because the Visual
# Studio generator does not support Flang. This requires running the superbuild
# from a Visual Studio developer command prompt.
set(proj ScaLAPACK)

sv_externals_enabled_depends(_deps MSMPI OpenBLAS Flang)

set(_generator)
set(_args)
if(WIN32)
  set(_generator CMAKE_GENERATOR "NMake Makefiles")
  list(APPEND _args
    -DCMAKE_C_COMPILER:STRING=cl
    -DCMAKE_Fortran_COMPILER:FILEPATH=${SV_EXTERNALS_Flang_BIN_DIR}/flang.exe
    -DCMAKE_USER_MAKE_RULES_OVERRIDE_Fortran:FILEPATH=${SV_EXTERNALS_CMAKE_DIR}/FlangNMakeFix.cmake
    -DCMAKE_TRY_COMPILE_PLATFORM_VARIABLES:STRING=CMAKE_USER_MAKE_RULES_OVERRIDE_Fortran)
  if(SV_EXTERNALS_ENABLE_MSMPI)
    # mpif.h includes mpifptr.h, which is in Include/x64
    list(APPEND _args
      "-DMPI_Fortran_ADDITIONAL_INCLUDE_DIRS:STRING=${SV_EXTERNALS_MSMPI_SDK_DIR}/Include$<SEMICOLON>${SV_EXTERNALS_MSMPI_SDK_DIR}/Include/x64")
  endif()
endif()

ExternalProject_Add(${proj}
  PREFIX ${SV_EXTERNALS_TOPLEVEL_BLD_DIR}/${proj}
  URL ${SV_EXTERNALS_ScaLAPACK_URL}
  URL_HASH SHA256=${SV_EXTERNALS_ScaLAPACK_SHA256}
  DOWNLOAD_DIR ${SV_EXTERNALS_DOWNLOAD_DIR}
  ${_generator}
  CMAKE_COMMAND ${SV_EXTERNALS_CMAKE_COMMAND}
  CMAKE_CACHE_ARGS
    ${SV_EXTERNALS_COMMON_CMAKE_CACHE_ARGS}
    -DCMAKE_INSTALL_PREFIX:PATH=${SV_EXTERNALS_ScaLAPACK_INSTALL_DIR}
    -DCMAKE_INSTALL_LIBDIR:STRING=lib
    -DSCALAPACK_BUILD_TESTS:BOOL=OFF
    -DBLAS_LIBRARIES:STRING=${SV_EXTERNALS_BLAS_LAPACK_LIBRARIES}
    -DLAPACK_LIBRARIES:STRING=${SV_EXTERNALS_BLAS_LAPACK_LIBRARIES}
    ${_args}
  DEPENDS ${_deps}
  )
