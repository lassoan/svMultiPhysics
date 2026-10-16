# VTK: only the modules used by svMultiPhysics, without rendering
# (options as in Docker/ubuntu/dockerfile)
set(proj VTK)

set(_system_libs ON)
if(WIN32)
  # Use the copies of expat and zlib that come with VTK
  set(_system_libs OFF)
endif()

ExternalProject_Add(${proj}
  PREFIX ${SV_EXTERNALS_TOPLEVEL_BLD_DIR}/${proj}
  URL ${SV_EXTERNALS_VTK_URL}
  URL_HASH SHA256=${SV_EXTERNALS_VTK_SHA256}
  DOWNLOAD_DIR ${SV_EXTERNALS_DOWNLOAD_DIR}
  CMAKE_CACHE_ARGS
    ${SV_EXTERNALS_COMMON_CMAKE_CACHE_ARGS}
    -DCMAKE_INSTALL_PREFIX:PATH=${SV_EXTERNALS_VTK_INSTALL_DIR}
    -DBUILD_EXAMPLES:BOOL=OFF
    -DVTK_USE_SYSTEM_EXPAT:BOOL=${_system_libs}
    -DVTK_USE_SYSTEM_ZLIB:BOOL=${_system_libs}
    -DVTK_LEGACY_REMOVE:BOOL=ON
    -DVTK_Group_Rendering:BOOL=OFF
    -DVTK_Group_StandAlone:BOOL=OFF
    -DVTK_RENDERING_BACKEND:STRING=None
    -DVTK_WRAP_PYTHON:BOOL=OFF
    -DModule_vtkChartsCore:BOOL=ON
    -DModule_vtkCommonCore:BOOL=ON
    -DModule_vtkCommonDataModel:BOOL=ON
    -DModule_vtkCommonExecutionModel:BOOL=ON
    -DModule_vtkFiltersCore:BOOL=ON
    -DModule_vtkFiltersFlowPaths:BOOL=ON
    -DModule_vtkFiltersModeling:BOOL=ON
    -DModule_vtkIOLegacy:BOOL=ON
    -DModule_vtkIOXML:BOOL=ON
    -DVTK_GROUP_ENABLE_Views:STRING=NO
    -DVTK_GROUP_ENABLE_Web:STRING=NO
    -DVTK_GROUP_ENABLE_Imaging:STRING=NO
    -DVTK_GROUP_ENABLE_Qt:STRING=DONT_WANT
    -DVTK_GROUP_ENABLE_Rendering:STRING=DONT_WANT
  )
