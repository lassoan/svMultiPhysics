# Trilinos: packages and third-party libraries as in Docker/ubuntu/dockerfile,
# plus NOX with its Tpetra (Thyra) adapters, which svMultiPhysics uses.
set(proj Trilinos)

sv_externals_enabled_depends(_deps MSMPI OpenBLAS Boost HDF5 HYPRE)

set(_patch "")
set(_args)
if(MSVC)
  # Fixes for MSVC (template definitions and gettimeofday)
  set(_patch ${GIT_EXECUTABLE} apply --whitespace=nowarn ${SV_EXTERNALS_PATCH_DIR}/trilinos-16.2.2-msvc.patch)
  list(APPEND _args "-DCMAKE_CXX_FLAGS:STRING=/EHsc /bigobj /permissive- /DNOMINMAX")
endif()

set(_tpl_args)
if(SV_EXTERNALS_BLAS_LAPACK_LIBRARIES)
  list(APPEND _tpl_args
    -DTPL_BLAS_LIBRARIES:STRING=${SV_EXTERNALS_BLAS_LAPACK_LIBRARIES}
    -DTPL_LAPACK_LIBRARIES:STRING=${SV_EXTERNALS_BLAS_LAPACK_LIBRARIES})
endif()
if(SV_EXTERNALS_ENABLE_Boost)
  if(WIN32)
    set(_boost_include ${SV_EXTERNALS_Boost_INSTALL_DIR}/include/boost-1_89)
  else()
    set(_boost_include ${SV_EXTERNALS_Boost_INSTALL_DIR}/include)
  endif()
  list(APPEND _tpl_args
    -DTPL_Boost_INCLUDE_DIRS:PATH=${_boost_include}
    -DBoost_LIBRARY_DIRS:PATH=${SV_EXTERNALS_Boost_INSTALL_DIR}/lib)
endif()
if(SV_EXTERNALS_ENABLE_HDF5)
  if(WIN32)
    set(_hdf5_libs "${SV_EXTERNALS_HDF5_INSTALL_DIR}/lib/libhdf5_hl.lib;${SV_EXTERNALS_HDF5_INSTALL_DIR}/lib/libhdf5.lib")
  else()
    set(_hdf5_libs "${SV_EXTERNALS_HDF5_INSTALL_DIR}/lib/libhdf5_hl.a;${SV_EXTERNALS_HDF5_INSTALL_DIR}/lib/libhdf5.a")
  endif()
  list(APPEND _tpl_args
    -DTPL_HDF5_INCLUDE_DIRS:PATH=${SV_EXTERNALS_HDF5_INSTALL_DIR}/include
    -DTPL_HDF5_LIBRARIES:STRING=${_hdf5_libs})
endif()
if(SV_EXTERNALS_ENABLE_HYPRE)
  if(WIN32)
    set(_hypre_lib ${SV_EXTERNALS_HYPRE_INSTALL_DIR}/lib/HYPRE.lib)
  else()
    set(_hypre_lib ${SV_EXTERNALS_HYPRE_INSTALL_DIR}/lib/libHYPRE.a)
  endif()
  # Ifpack2 parses the hypre headers in HYPRE_INCLUDE_DIRS
  list(APPEND _tpl_args
    -DTPL_HYPRE_INCLUDE_DIRS:PATH=${SV_EXTERNALS_HYPRE_INSTALL_DIR}/include
    -DHYPRE_INCLUDE_DIRS:PATH=${SV_EXTERNALS_HYPRE_INSTALL_DIR}/include
    -DTPL_HYPRE_LIBRARIES:STRING=${_hypre_lib})
endif()
if(WIN32)
  list(APPEND _tpl_args -DMPI_USE_COMPILER_WRAPPERS:BOOL=OFF)
  if(SV_EXTERNALS_ENABLE_MSMPI)
    list(APPEND _tpl_args -DMPI_EXEC:FILEPATH=${SV_EXTERNALS_MSMPI_BIN_DIR}/mpiexec.exe)
  endif()
endif()

ExternalProject_Add(${proj}
  PREFIX ${SV_EXTERNALS_TOPLEVEL_BLD_DIR}/${proj}
  URL ${SV_EXTERNALS_Trilinos_URL}
  URL_HASH SHA256=${SV_EXTERNALS_Trilinos_SHA256}
  DOWNLOAD_DIR ${SV_EXTERNALS_DOWNLOAD_DIR}
  PATCH_COMMAND ${_patch}
  CMAKE_COMMAND ${SV_EXTERNALS_CMAKE_COMMAND}
  CMAKE_CACHE_ARGS
    ${SV_EXTERNALS_COMMON_CMAKE_CACHE_ARGS}
    -DCMAKE_INSTALL_PREFIX:PATH=${SV_EXTERNALS_Trilinos_INSTALL_DIR}
    ${_args}
    -DTrilinos_ENABLE_Fortran:BOOL=OFF
    -DTrilinos_ENABLE_TESTS:BOOL=OFF
    -DTrilinos_ENABLE_EXAMPLES:BOOL=OFF
    -DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF
    -DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON
    -DTrilinos_ENABLE_Teuchos:BOOL=ON
    -DTrilinos_ENABLE_Kokkos:BOOL=ON
    -DTrilinos_ENABLE_KokkosKernels:BOOL=ON
    -DTrilinos_ENABLE_Tpetra:BOOL=ON
    -DTrilinos_ENABLE_Belos:BOOL=ON
    -DTrilinos_ENABLE_Ifpack2:BOOL=ON
    -DTrilinos_ENABLE_Amesos2:BOOL=ON
    -DTrilinos_ENABLE_Xpetra:BOOL=ON
    -DTrilinos_ENABLE_MueLu:BOOL=ON
    -DTrilinos_ENABLE_ROL:BOOL=ON
    -DTrilinos_ENABLE_Sacado:BOOL=ON
    -DTrilinos_ENABLE_Zoltan:BOOL=ON
    -DTrilinos_ENABLE_Zoltan2:BOOL=ON
    -DTrilinos_ENABLE_NOX:BOOL=ON
    -DTrilinos_ENABLE_ThyraCore:BOOL=ON
    -DTrilinos_ENABLE_ThyraTpetraAdapters:BOOL=ON
    -DXpetra_ENABLE_Kokkos_compat:BOOL=ON
    -DMueLu_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON
    -DKokkos_ENABLE_SERIAL:BOOL=ON
    -DTpetra_INST_SERIAL:BOOL=ON
    -DTpetra_INST_DOUBLE:BOOL=ON
    -DTpetra_INST_INT_INT:BOOL=ON
    -DTPL_ENABLE_MPI:BOOL=ON
    -DTPL_ENABLE_BLAS:BOOL=ON
    -DTPL_ENABLE_LAPACK:BOOL=ON
    -DTPL_ENABLE_Boost:BOOL=${SV_EXTERNALS_ENABLE_Boost}
    -DTPL_ENABLE_HDF5:BOOL=${SV_EXTERNALS_ENABLE_HDF5}
    -DTPL_ENABLE_HYPRE:BOOL=${SV_EXTERNALS_ENABLE_HYPRE}
    ${_tpl_args}
  DEPENDS ${_deps}
  )
