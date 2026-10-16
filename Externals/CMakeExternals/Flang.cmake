# LLVM Flang Fortran compiler (Windows only)
#
# Used to build the Fortran dependencies (ScaLAPACK, MUMPS) and the genBC
# programs of the tests with the MSVC toolchain. The conda-forge packages of
# Flang, its runtime and the libraries it needs are extracted without installing.
if(NOT WIN32)
  message(FATAL_ERROR "SV_EXTERNALS_ENABLE_Flang is only supported on Windows. Use the system Fortran compiler on other platforms.")
endif()

set(proj Flang)

set(_packages "")
foreach(package ${SV_EXTERNALS_Flang_PACKAGES})
  string(APPEND _packages "\"${package}\"\n")
endforeach()
file(WRITE "${CMAKE_BINARY_DIR}/${proj}-packages.cmake" "set(PACKAGES\n${_packages})\n")

ExternalProject_Add(${proj}
  PREFIX ${SV_EXTERNALS_TOPLEVEL_BLD_DIR}/${proj}
  DOWNLOAD_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ${CMAKE_COMMAND}
    -DPACKAGES_FILE=${CMAKE_BINARY_DIR}/${proj}-packages.cmake
    -DDOWNLOAD_DIR=${SV_EXTERNALS_DOWNLOAD_DIR}
    -DDEST=${SV_EXTERNALS_Flang_INSTALL_DIR}
    -P ${SV_EXTERNALS_CMAKE_DIR}/SvExtCondaInstall.cmake
  )
