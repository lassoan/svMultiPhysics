# OpenBLAS (BLAS and LAPACK)
#
# On Windows the prebuilt package is used (building the LAPACK part requires a
# Fortran compiler). It provides an import library that can be used with MSVC.
set(proj OpenBLAS)

if(WIN32)
  ExternalProject_Add(${proj}
    PREFIX ${SV_EXTERNALS_TOPLEVEL_BLD_DIR}/${proj}
    URL ${SV_EXTERNALS_OpenBLAS_WINDOWS_URL}
    URL_HASH SHA256=${SV_EXTERNALS_OpenBLAS_WINDOWS_SHA256}
    DOWNLOAD_DIR ${SV_EXTERNALS_DOWNLOAD_DIR}
    SOURCE_DIR ${SV_EXTERNALS_OpenBLAS_INSTALL_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    )
else()
  ExternalProject_Add(${proj}
    PREFIX ${SV_EXTERNALS_TOPLEVEL_BLD_DIR}/${proj}
    URL ${SV_EXTERNALS_OpenBLAS_URL}
    DOWNLOAD_DIR ${SV_EXTERNALS_DOWNLOAD_DIR}
    CMAKE_CACHE_ARGS
      ${SV_EXTERNALS_COMMON_CMAKE_CACHE_ARGS}
      -DBUILD_SHARED_LIBS:BOOL=ON
      -DCMAKE_INSTALL_PREFIX:PATH=${SV_EXTERNALS_OpenBLAS_INSTALL_DIR}
      -DCMAKE_INSTALL_LIBDIR:STRING=lib
    )
endif()
