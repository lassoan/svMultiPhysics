# svZeroDSolver, used by the svZeroD coupled test cases. It is built in the
# svZeroDSolver directory of the source tree, where the test cases expect it
# (as in the CI). The directory is ignored by git.
set(proj svZeroDSolver)

set(_generator)
if(CMAKE_GENERATOR MATCHES "Visual Studio")
  # A single-configuration generator, so that the interface library is created
  # in build/src/interface (and not in a configuration subdirectory)
  set(_generator CMAKE_GENERATOR "NMake Makefiles")
endif()

ExternalProject_Add(${proj}
  PREFIX ${SV_EXTERNALS_TOPLEVEL_BLD_DIR}/${proj}
  GIT_REPOSITORY ${SV_EXTERNALS_svZeroDSolver_GIT_URL}
  GIT_TAG ${SV_EXTERNALS_svZeroDSolver_GIT_TAG}
  GIT_SHALLOW TRUE
  SOURCE_DIR ${SV_EXTERNALS_svZeroDSolver_SOURCE_DIR}
  BINARY_DIR ${SV_EXTERNALS_svZeroDSolver_SOURCE_DIR}/build
  ${_generator}
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${SV_EXTERNALS_BUILD_TYPE}
    -DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.5
  INSTALL_COMMAND ""
  )
