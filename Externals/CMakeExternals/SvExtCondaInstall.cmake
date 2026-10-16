# Download and extract conda-forge packages (.conda or .tar.bz2) into a directory.
#
# Usage:
#   cmake -DPACKAGES_FILE=<file> -DDOWNLOAD_DIR=<dir> -DDEST=<dir> -P SvExtCondaInstall.cmake
#
# PACKAGES_FILE is a CMake file that sets PACKAGES to a list of "<url>|<sha256>" items.
# A .conda package is a zip archive containing the package files in pkg-*.tar.zst.

cmake_minimum_required(VERSION 3.20)

include("${PACKAGES_FILE}")
file(MAKE_DIRECTORY "${DOWNLOAD_DIR}" "${DEST}")

foreach(package ${PACKAGES})
  string(REPLACE "|" ";" package "${package}")
  list(GET package 0 url)
  list(GET package 1 sha256)
  get_filename_component(name "${url}" NAME)
  set(file "${DOWNLOAD_DIR}/${name}")
  if(NOT EXISTS "${file}")
    message(STATUS "Downloading ${url}")
    file(DOWNLOAD "${url}" "${file}" EXPECTED_HASH SHA256=${sha256} STATUS status)
    list(GET status 0 status_code)
    if(NOT status_code EQUAL 0)
      file(REMOVE "${file}")
      message(FATAL_ERROR "Failed to download ${url}: ${status}")
    endif()
  endif()

  message(STATUS "Extracting ${name}")
  if(name MATCHES "\\.conda$")
    set(tmp "${DOWNLOAD_DIR}/${name}-extracted")
    file(REMOVE_RECURSE "${tmp}")
    file(ARCHIVE_EXTRACT INPUT "${file}" DESTINATION "${tmp}")
    file(GLOB pkg_archive "${tmp}/pkg-*.tar.zst")
    file(ARCHIVE_EXTRACT INPUT "${pkg_archive}" DESTINATION "${DEST}")
    file(REMOVE_RECURSE "${tmp}")
  else()
    file(ARCHIVE_EXTRACT INPUT "${file}" DESTINATION "${DEST}")
  endif()
endforeach()

# Flang looks for the compiler-rt builtins library as clang_rt.builtins.lib,
# while the compiler-rt package provides clang_rt.builtins-x86_64.lib
if(EXISTS "${DEST}/lib/clang/20/lib/windows/clang_rt.builtins-x86_64.lib")
  file(COPY_FILE "${DEST}/lib/clang/20/lib/windows/clang_rt.builtins-x86_64.lib"
    "${DEST}/Library/lib/clang_rt.builtins.lib")
endif()
