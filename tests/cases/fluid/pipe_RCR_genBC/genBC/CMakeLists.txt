# Builds the genBC executable used by the test case.
#
# The executable is created as genBC.exe in this directory (on all platforms),
# where the solver input file expects it.

cmake_minimum_required(VERSION 3.20)

# Build optimized code by default (must be set before project())
if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()

if(CMAKE_GENERATOR MATCHES "NMake")
  # Workaround for a CMake bug with LLVMFlang and NMake Makefiles: CMake appends
  # the runtime link flags after the end marker of the inline response file,
  # which NMake cannot parse. Skip linking in the compiler checks.
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
endif()

project(genBC Fortran)

# Second part of the workaround: move the end marker of the inline response
# file after the appended link flags.
if(CMAKE_GENERATOR MATCHES "NMake" AND CMAKE_Fortran_COMPILER_ID STREQUAL "LLVMFlang")
  string(REPLACE "${CMAKE_END_TEMP_FILE}" "" CMAKE_Fortran_LINK_EXECUTABLE "${CMAKE_Fortran_LINK_EXECUTABLE}")
  string(APPEND CMAKE_Fortran_LINK_EXECUTABLE "${CMAKE_END_TEMP_FILE}")
endif()

add_executable(genBC
  src/Modules.f
  src/USER.f
  src/GenBC.f
)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_include_directories(genBC PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

set_target_properties(genBC PROPERTIES
  Fortran_PREPROCESS ON
  SUFFIX ".exe"
  # The generator expression prevents multi-config generators from adding a
  # per-configuration subdirectory.
  RUNTIME_OUTPUT_DIRECTORY $<1:${CMAKE_CURRENT_SOURCE_DIR}>
)
