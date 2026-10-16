# Microsoft MPI (Windows only)
#
# The SDK (headers and import libraries) is extracted from the installer in
# Code/ThirdParty/msmpi/Installers without installing it. The runtime (msmpi.dll,
# mpiexec, smpd) is taken from the conda-forge msmpi package. No administrator
# rights are needed. (The headers of the conda-forge package are not used: they
# include stdint.h, which conflicts with the stdint definitions of GKlib.)
if(NOT WIN32)
  message(FATAL_ERROR "SV_EXTERNALS_ENABLE_MSMPI is only supported on Windows. Use a system MPI on other platforms.")
endif()

set(proj MSMPI)

# msiexec does not extract anything if the installer path contains ".."
get_filename_component(_msi "${SV_EXTERNALS_MSMPI_SDK_MSI}" ABSOLUTE)
file(TO_NATIVE_PATH "${_msi}" _msi)
file(TO_NATIVE_PATH "${SV_EXTERNALS_MSMPI_INSTALL_DIR}/sdk" _sdk_dir)
file(WRITE "${CMAKE_BINARY_DIR}/${proj}-runtime-packages.cmake"
  "set(PACKAGES \"${SV_EXTERNALS_MSMPI_RUNTIME_URL}|${SV_EXTERNALS_MSMPI_RUNTIME_SHA256}\")\n")

ExternalProject_Add(${proj}
  PREFIX ${SV_EXTERNALS_TOPLEVEL_BLD_DIR}/${proj}
  DOWNLOAD_COMMAND ""
  CONFIGURE_COMMAND ""
  # Administrative install: extracts the files of the installer
  BUILD_COMMAND cmd /c start /wait msiexec /a ${_msi} /qn TARGETDIR=${_sdk_dir}
  INSTALL_COMMAND ${CMAKE_COMMAND}
    -DPACKAGES_FILE=${CMAKE_BINARY_DIR}/${proj}-runtime-packages.cmake
    -DDOWNLOAD_DIR=${SV_EXTERNALS_DOWNLOAD_DIR}
    -DDEST=${SV_EXTERNALS_MSMPI_INSTALL_DIR}/runtime
    -P ${SV_EXTERNALS_CMAKE_DIR}/SvExtCondaInstall.cmake
  )
