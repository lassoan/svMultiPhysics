# Eigen headers (used by PETSc; PETSc cannot download it when using Microsoft compilers)
set(proj Eigen)

ExternalProject_Add(${proj}
  PREFIX ${SV_EXTERNALS_TOPLEVEL_BLD_DIR}/${proj}
  URL ${SV_EXTERNALS_Eigen_URL}
  URL_HASH SHA256=${SV_EXTERNALS_Eigen_SHA256}
  DOWNLOAD_DIR ${SV_EXTERNALS_DOWNLOAD_DIR}
  SOURCE_DIR ${SV_EXTERNALS_Eigen_INSTALL_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  )
