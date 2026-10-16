# MUMPS source package, built by PETSc (--download-mumps=<package>).
#
# On Windows the package is patched so that MUMPS uses the Fortran name mangling
# of LLVM Flang instead of always assuming that of Intel Fortran.
set(proj MUMPS)

set(_package ${SV_EXTERNALS_MUMPS_INSTALL_DIR}/MUMPS_${SV_EXTERNALS_MUMPS_VERSION}.tar.gz)
set(_patch "")
if(WIN32)
  set(_patch ${GIT_EXECUTABLE} apply --whitespace=nowarn ${SV_EXTERNALS_PATCH_DIR}/mumps-5.9.1-flang-windows.patch)
endif()

ExternalProject_Add(${proj}
  PREFIX ${SV_EXTERNALS_TOPLEVEL_BLD_DIR}/${proj}
  URL ${SV_EXTERNALS_MUMPS_URL}
  URL_HASH SHA256=${SV_EXTERNALS_MUMPS_SHA256}
  DOWNLOAD_DIR ${SV_EXTERNALS_DOWNLOAD_DIR}
  SOURCE_DIR ${SV_EXTERNALS_TOPLEVEL_BLD_DIR}/${proj}/MUMPS_${SV_EXTERNALS_MUMPS_VERSION}
  PATCH_COMMAND ${_patch}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  # Package the (patched) sources for PETSc
  INSTALL_COMMAND ${CMAKE_COMMAND} -E make_directory ${SV_EXTERNALS_MUMPS_INSTALL_DIR}
    COMMAND ${CMAKE_COMMAND} -E chdir ${SV_EXTERNALS_TOPLEVEL_BLD_DIR}/${proj}
      ${CMAKE_COMMAND} -E tar czf ${_package} MUMPS_${SV_EXTERNALS_MUMPS_VERSION}
  )
