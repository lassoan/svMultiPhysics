# Cygwin (Windows only): Unix tools that PETSc's build system needs on Windows.
#
# A minimal Cygwin is installed without administrator rights. The flang-cygwin
# wrapper, which lets PETSc use LLVM Flang from Cygwin, is installed with it.
if(NOT WIN32)
  message(FATAL_ERROR "SV_EXTERNALS_ENABLE_Cygwin is only supported on Windows.")
endif()

set(proj Cygwin)

file(TO_NATIVE_PATH "${SV_EXTERNALS_Cygwin_INSTALL_DIR}" _root)
file(TO_NATIVE_PATH "${SV_EXTERNALS_DOWNLOAD_DIR}/cygwin-packages" _package_dir)

# Path of flang.exe in Cygwin notation (/cygdrive/<drive>/...)
set(_flang "${SV_EXTERNALS_Flang_BIN_DIR}/flang.exe")
string(SUBSTRING "${_flang}" 0 1 _drive)
string(TOLOWER "${_drive}" _drive)
string(SUBSTRING "${_flang}" 2 -1 _path)
set(SV_EXTERNALS_FLANG_EXE_CYGWIN "/cygdrive/${_drive}${_path}")
configure_file(${SV_EXTERNALS_CMAKE_DIR}/flang-cygwin.in
  ${CMAKE_BINARY_DIR}/flang-cygwin @ONLY NEWLINE_STYLE UNIX)

ExternalProject_Add(${proj}
  PREFIX ${SV_EXTERNALS_TOPLEVEL_BLD_DIR}/${proj}
  URL ${SV_EXTERNALS_Cygwin_SETUP_URL}
  DOWNLOAD_NO_EXTRACT TRUE
  DOWNLOAD_DIR ${SV_EXTERNALS_DOWNLOAD_DIR}
  DOWNLOAD_NAME cygwin-setup-x86_64.exe
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ${SV_EXTERNALS_DOWNLOAD_DIR}/cygwin-setup-x86_64.exe
    --quiet-mode --no-admin --no-shortcuts --no-startmenu --no-desktop --no-write-registry
    --root ${_root} --local-package-dir ${_package_dir}
    --site ${SV_EXTERNALS_Cygwin_MIRROR}
    --packages ${SV_EXTERNALS_Cygwin_PACKAGES}
  INSTALL_COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/flang-cygwin
    ${SV_EXTERNALS_Cygwin_INSTALL_DIR}/usr/local/bin/flang-cygwin
  )
