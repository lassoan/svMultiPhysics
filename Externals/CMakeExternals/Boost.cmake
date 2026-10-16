# Boost (built with its own b2 build system)
set(proj Boost)

include(ProcessorCount)
ProcessorCount(_ncpu)
if(_ncpu EQUAL 0)
  set(_ncpu 1)
endif()

if(WIN32)
  # The bootstrap scripts run helper scripts from the current directory, which
  # does not work if NoDefaultCurrentDirectoryInExePath is set.
  set(_configure ${CMAKE_COMMAND} -E env --unset=NoDefaultCurrentDirectoryInExePath
    cmd /c bootstrap.bat vc143)
  set(_build ${CMAKE_COMMAND} -E env --unset=NoDefaultCurrentDirectoryInExePath
    <SOURCE_DIR>/b2.exe install --prefix=${SV_EXTERNALS_Boost_INSTALL_DIR} --build-dir=<BINARY_DIR>
    toolset=msvc address-model=64 link=static runtime-link=shared variant=release threading=multi
    -j${_ncpu})
else()
  set(_configure ./bootstrap.sh --prefix=${SV_EXTERNALS_Boost_INSTALL_DIR})
  set(_build ./b2 install --prefix=${SV_EXTERNALS_Boost_INSTALL_DIR} --build-dir=<BINARY_DIR> -j${_ncpu})
endif()

ExternalProject_Add(${proj}
  PREFIX ${SV_EXTERNALS_TOPLEVEL_BLD_DIR}/${proj}
  URL ${SV_EXTERNALS_Boost_URL}
  URL_HASH SHA256=${SV_EXTERNALS_Boost_SHA256}
  DOWNLOAD_DIR ${SV_EXTERNALS_DOWNLOAD_DIR}
  BUILD_IN_SOURCE TRUE
  CONFIGURE_COMMAND ${_configure}
  BUILD_COMMAND ${_build}
  INSTALL_COMMAND ""
  )
