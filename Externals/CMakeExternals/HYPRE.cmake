# hypre (used by Trilinos and PETSc)
set(proj HYPRE)

sv_externals_enabled_depends(_deps MSMPI)

set(_args)
if(MSVC)
  # Some hypre 2.33 C sources use true/false without including stdbool.h
  list(APPEND _args "-DCMAKE_C_FLAGS:STRING=/DWIN32 /D_WINDOWS /FIstdbool.h")
endif()

ExternalProject_Add(${proj}
  PREFIX ${SV_EXTERNALS_TOPLEVEL_BLD_DIR}/${proj}
  URL ${SV_EXTERNALS_HYPRE_URL}
  URL_HASH SHA256=${SV_EXTERNALS_HYPRE_SHA256}
  DOWNLOAD_DIR ${SV_EXTERNALS_DOWNLOAD_DIR}
  SOURCE_SUBDIR src
  CMAKE_COMMAND ${SV_EXTERNALS_CMAKE_COMMAND}
  CMAKE_CACHE_ARGS
    ${SV_EXTERNALS_COMMON_CMAKE_CACHE_ARGS}
    -DCMAKE_INSTALL_PREFIX:PATH=${SV_EXTERNALS_HYPRE_INSTALL_DIR}
    -DHYPRE_ENABLE_MPI:BOOL=ON
    ${_args}
  DEPENDS ${_deps}
  )
