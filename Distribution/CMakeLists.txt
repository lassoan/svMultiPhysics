#-----------------------------------------------------------------------------
# Installation packages (enabled with SV_ENABLE_DISTRIBUTION)
#
# The package contains the svmultiphysics executable and the shared libraries it
# depends on (e.g. MPI, BLAS/LAPACK, VTK), so it can be used on computers where
# the dependencies are not installed. Create the package with
#   cpack -C Release
# (or by building the "package" target) in the svMultiPhysics build directory.
#
# Package types (SV_INSTALLER_TYPE):
#   Windows: NSIS (installer, default), ZIP, WIX
#   Linux:   TGZ (default), DEB
#   macOS:   TGZ (default), productbuild
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Runtime libraries
#-----------------------------------------------------------------------------

# Microsoft Visual C++ runtime
set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)
include(InstallRequiredSystemLibraries)
if(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS)
  install(PROGRAMS ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
    DESTINATION ${SV_INSTALL_RUNTIME_DIR}
    COMPONENT CoreExecutables)
endif()

# Shared libraries that svmultiphysics depends on. They are searched for in the
# directories of SV_DISTRIBUTION_DEPENDENCY_DIRS and, on Windows, in the PATH.
set(SV_DISTRIBUTION_DEPENDENCY_DIRS "" CACHE STRING
  "Directories where the shared libraries that svmultiphysics depends on (e.g. MPI, BLAS/LAPACK, VTK) are searched for when creating an installation package")
if(WIN32)
  set(_dependency_destination ${SV_INSTALL_RUNTIME_DIR})
  set(_dependency_dirs ${SV_DISTRIBUTION_DEPENDENCY_DIRS} $ENV{PATH})
  set(_system_libraries_regex "[/\\\\][Ss][Yy][Ss][Tt][Ee][Mm]32[/\\\\]")
else()
  set(_dependency_destination ${SV_INSTALL_LIBRARY_DIR})
  set(_dependency_dirs ${SV_DISTRIBUTION_DEPENDENCY_DIRS})
  # Do not include the libraries of the operating system
  set(_system_libraries_regex "^/(lib|lib64|usr/lib|usr/lib64|System|usr/lib/system)/")
  # Find the installed libraries relative to the executable
  if(APPLE)
    set_target_properties(${SV_MULTIPHYSICS_EXE} PROPERTIES INSTALL_RPATH "@executable_path/../${SV_INSTALL_LIBRARY_DIR}")
  else()
    set_target_properties(${SV_MULTIPHYSICS_EXE} PROPERTIES INSTALL_RPATH "$ORIGIN/../${SV_INSTALL_LIBRARY_DIR}")
  endif()
endif()
string(REPLACE "\\" "/" _dependency_dirs "${_dependency_dirs}")

install(CODE "set(_sv_dependency_dirs \"${_dependency_dirs}\")
set(_sv_dependency_destination \"${_dependency_destination}\")
set(_sv_system_libraries_regex \"${_system_libraries_regex}\")"
  COMPONENT CoreExecutables)
install(CODE [[
  file(GET_RUNTIME_DEPENDENCIES
    EXECUTABLES $<TARGET_FILE:svmultiphysics>
    RESOLVED_DEPENDENCIES_VAR _sv_dependencies
    UNRESOLVED_DEPENDENCIES_VAR _sv_unresolved_dependencies
    DIRECTORIES ${_sv_dependency_dirs}
    PRE_EXCLUDE_REGEXES "^api-ms-" "^ext-ms-"
    POST_EXCLUDE_REGEXES "${_sv_system_libraries_regex}"
    )
  foreach(_dependency ${_sv_dependencies})
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/${_sv_dependency_destination}"
      TYPE SHARED_LIBRARY FOLLOW_SYMLINK_CHAIN FILES "${_dependency}")
  endforeach()
  foreach(_dependency ${_sv_unresolved_dependencies})
    message(WARNING "Shared library ${_dependency} required by svmultiphysics was not found; it is not included in the package")
  endforeach()
  ]]
  COMPONENT CoreExecutables)

# MS-MPI: the executables needed to run in parallel (mpiexec, smpd) are not
# linked, so they are installed explicitly
if(WIN32 AND MPIEXEC_EXECUTABLE)
  get_filename_component(_mpi_bin_dir "${MPIEXEC_EXECUTABLE}" DIRECTORY)
  foreach(_file mpiexec.exe smpd.exe msmpires.dll)
    if(EXISTS "${_mpi_bin_dir}/${_file}")
      install(PROGRAMS "${_mpi_bin_dir}/${_file}"
        DESTINATION ${SV_INSTALL_RUNTIME_DIR}
        COMPONENT CoreExecutables)
    endif()
  endforeach()
endif()

install(FILES ${SV_SOURCE_HOME}/LICENSE.txt ${SV_SOURCE_HOME}/README.md
  DESTINATION ${SV_INSTALL_DOC_DIR}
  COMPONENT CoreExecutables)

#-----------------------------------------------------------------------------
# Platform independent options
#-----------------------------------------------------------------------------
set(CPACK_PACKAGE_NAME "svMultiPhysics")
set(CPACK_PACKAGE_VENDOR "SimVascular.org")
set(CPACK_PACKAGE_CONTACT "SimVascular<sv@simvascular.org>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "svMultiPhysics multi-physics finite element solver")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/SimVascular/svMultiPhysics")
set(CPACK_RESOURCE_FILE_LICENSE "${SV_SOURCE_HOME}/LICENSE.txt")
set(CPACK_PACKAGE_VERSION_MAJOR ${SV_MAJOR_VERSION})
set(CPACK_PACKAGE_VERSION_MINOR ${SV_MINOR_VERSION})
set(CPACK_PACKAGE_VERSION_PATCH ${SV_PATCH_VERSION})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "svMultiPhysics")
set(CPACK_MONOLITHIC_INSTALL 1)
string(REPLACE "." "-" SV_INSTALL_VERSION_DIR ${SV_FULL_VERSION})

#-----------------------------------------------------------------------------
# WINDOWS
#-----------------------------------------------------------------------------
if(WIN32)
  set(SV_INSTALLER_TYPE NSIS CACHE STRING "Installation package type")
  set_property(CACHE SV_INSTALLER_TYPE PROPERTY STRINGS NSIS ZIP WIX)
  set(CPACK_GENERATOR ${SV_INSTALLER_TYPE})
  set(CPACK_PACKAGE_FILE_NAME "svMultiPhysics-windows-x64-${SV_FULL_VERSION}")

  # NSIS installer
  set(CPACK_NSIS_PACKAGE_NAME "svMultiPhysics")
  set(CPACK_NSIS_DISPLAY_NAME "svMultiPhysics ${SV_FULL_VERSION}")
  set(CPACK_NSIS_MUI_ICON "${CMAKE_CURRENT_SOURCE_DIR}/windows/simvascular.ico")
  set(CPACK_NSIS_MUI_UNIICON "${CMAKE_CURRENT_SOURCE_DIR}/windows/simvascular.ico")
  set(CPACK_NSIS_INSTALLED_ICON_NAME "${SV_INSTALL_RUNTIME_DIR}\\\\${SV_MULTIPHYSICS_EXE}.exe")
  set(CPACK_NSIS_URL_INFO_ABOUT "${CPACK_PACKAGE_HOMEPAGE_URL}")
  # Offer to add the bin directory (svmultiphysics, mpiexec) to the PATH
  set(CPACK_NSIS_MODIFY_PATH ON)
  set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
  set(CPACK_NSIS_EXECUTABLES_DIRECTORY ${SV_INSTALL_RUNTIME_DIR})
  # Location of makensis, if it is not in the PATH
  set(SV_NSIS_EXECUTABLE "" CACHE FILEPATH "NSIS makensis executable (if not in the PATH)")
  if(SV_NSIS_EXECUTABLE)
    set(CPACK_NSIS_EXECUTABLE "${SV_NSIS_EXECUTABLE}")
  endif()

  # WiX installer
  set(CPACK_WIX_UI_BANNER ${CMAKE_CURRENT_SOURCE_DIR}/windows/msi-banner.jpg)
  set(CPACK_WIX_UI_DIALOG ${CMAKE_CURRENT_SOURCE_DIR}/windows/msi-dialog.jpg)
  set(CPACK_WIX_PRODUCT_ICON "${CMAKE_CURRENT_SOURCE_DIR}/windows/simvascular.ico")
  set(CPACK_WIX_UPGRADE_GUID "2359EF69-C03F-401C-9BC3-C19B10297E3B")
endif()

#-----------------------------------------------------------------------------
# LINUX
#-----------------------------------------------------------------------------
if(UNIX AND NOT APPLE)
  set(SV_INSTALLER_TYPE TGZ CACHE STRING "Installation package type")
  set_property(CACHE SV_INSTALLER_TYPE PROPERTY STRINGS TGZ DEB)
  set(CPACK_GENERATOR ${SV_INSTALLER_TYPE})
  set(CPACK_PACKAGE_FILE_NAME "svMultiPhysics-linux-x64-${SV_FULL_VERSION}")
  if(SV_INSTALLER_TYPE STREQUAL DEB)
    set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local/sv/svMultiPhysics/${SV_INSTALL_VERSION_DIR}")
  else()
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)
  endif()
endif()

#-----------------------------------------------------------------------------
# APPLE
#-----------------------------------------------------------------------------
if(APPLE)
  set(SV_INSTALLER_TYPE TGZ CACHE STRING "Installation package type")
  set_property(CACHE SV_INSTALLER_TYPE PROPERTY STRINGS TGZ productbuild)
  set(CPACK_GENERATOR ${SV_INSTALLER_TYPE})
  set(CPACK_PACKAGE_FILE_NAME "svMultiPhysics-darwin-${CMAKE_SYSTEM_PROCESSOR}-${SV_FULL_VERSION}")
  if(SV_INSTALLER_TYPE STREQUAL productbuild)
    set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local/sv/svMultiPhysics/${SV_INSTALL_VERSION_DIR}")
  else()
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)
  endif()
endif()

if(CPACK_PACKAGE_FILENAME_TAG)
  set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}-${CPACK_PACKAGE_FILENAME_TAG}")
endif()

message(STATUS "Package name: ${CPACK_PACKAGE_FILE_NAME}")
include(CPack)
