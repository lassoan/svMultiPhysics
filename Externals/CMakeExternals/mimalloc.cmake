# mimalloc memory allocator. svMultiPhysics allocates and frees many small
# arrays, which is faster with mimalloc than with the default allocator of
# Windows. A shared library is built, which replaces malloc/free of the
# whole program (on Windows using mimalloc-redirect.dll).
set(proj mimalloc)

ExternalProject_Add(${proj}
  PREFIX ${SV_EXTERNALS_TOPLEVEL_BLD_DIR}/${proj}
  URL ${SV_EXTERNALS_mimalloc_URL}
  URL_HASH SHA256=${SV_EXTERNALS_mimalloc_SHA256}
  DOWNLOAD_DIR ${SV_EXTERNALS_DOWNLOAD_DIR}
  CMAKE_CACHE_ARGS
    ${SV_EXTERNALS_COMMON_CMAKE_CACHE_ARGS}
    -DCMAKE_INSTALL_PREFIX:PATH=${SV_EXTERNALS_mimalloc_INSTALL_DIR}
    -DMI_BUILD_SHARED:BOOL=ON
    -DMI_BUILD_STATIC:BOOL=OFF
    -DMI_BUILD_OBJECT:BOOL=OFF
    -DMI_BUILD_TESTS:BOOL=OFF
  )
