# Copyright (c) 2014-2015 The Regents of the University of California.
# All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject
# to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#-----------------------------------------------------------------------------
# This toplevel CMakeLists.txt should be used as the first attempt to build
# SimVascular.
# This CMakeLists.txt uses ExternalProject_Add to do two main things:
# 1. ExternalProject_Add(Externals ---> Gets all of SimVascular's externals
# 2. ExternalProject_Add(SimVascular ---> Builds SimVascular
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)

#-----------------------------------------------------------------------------
Project(SV_TOP)
message(STATUS "SV_TOP: ${SV_TOP}")
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# CMake module path inside of true simvascular source
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Code/CMake"
	"${CMAKE_MODULE_PATH}")

# Add -march=native to CMAKE_C_FLAGS and CMAKE_CXX_FLAGS for Eigen performance
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------

# CMake Includes
include(CheckLibraryExists)
include(GetPrerequisites)
include(GenerateExportHeader)
include(ExternalProject)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# SimVascular Includes
include(SimVascularMacros)
include(SimVascularFunctions)
include(SimVascularSystemSetup)
include(SimVascularFunctionCheckCompilerFlags)
#-----------------------------------------------------------------------------

# Add additional variables that can be set from the cmake gui or from the command line.
#
# These variables must later be add to 'ExternalProject_Add(svMultiPhysics' as -D options. 
#
set(SV_USE_TRILINOS OFF CACHE BOOL "Build with the Trilinos linear algebra package")
#set(SV_USE_PETSC OFF CACHE BOOL "Build with the PETSc linear algebra package")
set(SV_PETSC_DIR "" CACHE STRING "Path to a local install of the PETSc linear algebra package")
set(ENABLE_COVERAGE OFF CACHE BOOL "Enable code coverage")
set(ENABLE_ARRAY_INDEX_CHECKING OFF CACHE BOOL "Enable Array index checking")
set(SV_LOCAL_VTK_PATH "" CACHE STRING "Path to a local build of VTK.")
set(ENABLE_UNIT_TEST OFF CACHE BOOL "Enable Unit Test by Google Test")

#-----------------------------------------------------------------------------
# RPATH handling
# No objects built directly with project. Not needed!
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Dependencies that the superbuild can download and build (all OFF by
# default). See Externals/CMakeExternals/SvExtDependencies.cmake.
include(${CMAKE_CURRENT_SOURCE_DIR}/Externals/CMakeExternals/SvExtDependencies.cmake)
set(SV_EXTERNALS_ENABLE_CMAKE_ARGS)
foreach(proj ${SV_EXTERNALS_DEPENDENCIES})
  option(SV_EXTERNALS_ENABLE_${proj} "Download and build ${proj}" OFF)
  list(APPEND SV_EXTERNALS_ENABLE_CMAKE_ARGS -DSV_EXTERNALS_ENABLE_${proj}:BOOL=${SV_EXTERNALS_ENABLE_${proj}})
endforeach()
# SvExtDependencies.cmake was included before the options were defined
include(${CMAKE_CURRENT_SOURCE_DIR}/Externals/CMakeExternals/SvExtDependencies.cmake)
if(CMAKE_BUILD_TYPE)
  set(SV_EXTERNALS_BUILD_TYPE ${CMAKE_BUILD_TYPE})
endif()

#-----------------------------------------------------------------------------
# Externals!
set(SV_EXTERNALS_ADDITIONAL_CMAKE_ARGS "" CACHE STRING "If more options want to be provided to the sv_externals build, they can be with this string")
set(SV_EXTERNALS_APPLE_CMAKE_ARGS)
ExternalProject_Add(Externals
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Externals
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/Externals-build
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
  CMAKE_CACHE_ARGS
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
    -DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_MACOSX_RPATH:BOOL=ON
    -DBUILD_TESTING:BOOL=OFF
    -DBUILD_EXAMPLES:BOOL=OFF
    ${SV_EXTERNALS_ENABLE_CMAKE_ARGS}
    -DSV_EXTERNALS_INSTALL_DIR:PATH=${SV_EXTERNALS_INSTALL_DIR}
    -DSV_EXTERNALS_DOWNLOAD_DIR:PATH=${SV_EXTERNALS_DOWNLOAD_DIR}
    -DSV_EXTERNALS_BUILD_TYPE:STRING=${SV_EXTERNALS_BUILD_TYPE}
    -DSV_EXTERNALS_Cygwin_MIRROR:STRING=${SV_EXTERNALS_Cygwin_MIRROR}
    -DSV_EXTERNALS_BLAS_LAPACK_LIBRARIES:STRING=${SV_EXTERNALS_BLAS_LAPACK_LIBRARIES}
    ${SV_EXTERNALS_APPLE_CMAKE_ARGS}
    ${SV_EXTERNALS_ADDITIONAL_CMAKE_ARGS}
    )

set(SV_EXTERNALS_TOPLEVEL_DIR "${CMAKE_BINARY_DIR}/Externals-build/sv_externals")

#-----------------------------------------------------------------------------
# Locations of the dependencies built by the superbuild, for svMultiPhysics
set(SV_DEPENDENCIES_CMAKE_ARGS)
set(SV_PREFIX_PATH ${CMAKE_PREFIX_PATH})
set(SV_USE_TRILINOS_VALUE ${SV_USE_TRILINOS})
set(SV_PETSC_DIR_VALUE ${SV_PETSC_DIR})
set(SV_LOCAL_VTK_PATH_VALUE ${SV_LOCAL_VTK_PATH})
if(SV_EXTERNALS_ENABLE_VTK)
  set(SV_LOCAL_VTK_PATH_VALUE ${SV_EXTERNALS_VTK_INSTALL_DIR}/lib/cmake/vtk-9.3)
endif()
if(SV_EXTERNALS_ENABLE_Trilinos)
  set(SV_USE_TRILINOS_VALUE ON)
  list(APPEND SV_PREFIX_PATH ${SV_EXTERNALS_Trilinos_INSTALL_DIR})
endif()
if(SV_EXTERNALS_ENABLE_PETSc)
  set(SV_PETSC_DIR_VALUE ${SV_EXTERNALS_PETSc_INSTALL_DIR})
endif()
if(SV_EXTERNALS_BLAS_LAPACK_LIBRARIES)
  list(APPEND SV_DEPENDENCIES_CMAKE_ARGS
    -DBLAS_LIBRARIES:STRING=${SV_EXTERNALS_BLAS_LAPACK_LIBRARIES}
    -DLAPACK_LIBRARIES:STRING=${SV_EXTERNALS_BLAS_LAPACK_LIBRARIES})
endif()
string(REPLACE ";" "$<SEMICOLON>" SV_PREFIX_PATH "${SV_PREFIX_PATH}")

# svMultiPhysics 
#
set(SV_ADDITIONAL_CMAKE_ARGS "" CACHE STRING "If more options want to be provided to the sv_externals build, they can be with this string")
set(SV_APPLE_CMAKE_ARGS)

# Build options shared by the svMultiPhysics builds
set(SV_COMMON_CMAKE_CACHE_ARGS
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
    -DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
    -DCMAKE_PREFIX_PATH:STRING=${SV_PREFIX_PATH}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_MACOSX_RPATH:BOOL=ON
    -DBUILD_TESTING:BOOL=OFF
    -DBUILD_EXAMPLES:BOOL=OFF
    -DBUILD_SHARED_LIBS:BOOL=OFF
    -DSV_EXTERNALS_USE_TOPLEVEL_DIR:BOOL=ON
    -DSV_EXTERNALS_TOPLEVEL_DIR:PATH=${SV_EXTERNALS_TOPLEVEL_DIR}
    -DENABLE_COVERAGE:BOOL=${ENABLE_COVERAGE}
    -DENABLE_UNIT_TEST:BOOL=${ENABLE_UNIT_TEST}
    -DENABLE_ARRAY_INDEX_CHECKING:BOOL=${ENABLE_ARRAY_INDEX_CHECKING}
    -DSV_LOCAL_VTK_PATH:STRING=${SV_LOCAL_VTK_PATH_VALUE}
    ${SV_DEPENDENCIES_CMAKE_ARGS}
    ${SV_APPLE_CMAKE_ARGS}
    ${SV_ADDITIONAL_CMAKE_ARGS}
    )

if(SV_EXTERNALS_ENABLE_PETSc)
  # As in the CI, svMultiPhysics is built separately with Trilinos (or without
  # external linear algebra package) and with PETSc
  set(SV_MAIN_PETSC_DIR "")
else()
  set(SV_MAIN_PETSC_DIR ${SV_PETSC_DIR_VALUE})
endif()

ExternalProject_Add(svMultiPhysics
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Code
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/svMultiPhysics-build
  DEPENDS Externals
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
  CMAKE_COMMAND ${SV_EXTERNALS_CMAKE_COMMAND}
  CMAKE_CACHE_ARGS
    ${SV_COMMON_CMAKE_CACHE_ARGS}
    -DSV_USE_TRILINOS:BOOL=${SV_USE_TRILINOS_VALUE}
    #-DSV_USE_PETSC:BOOL=${SV_USE_PETSC}
    -DSV_PETSC_DIR:STRING=${SV_MAIN_PETSC_DIR}
    )

if(SV_EXTERNALS_ENABLE_PETSc)
  ExternalProject_Add(svMultiPhysics-petsc
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Code
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/svMultiPhysics-petsc-build
    DEPENDS Externals
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
    CMAKE_COMMAND ${SV_EXTERNALS_CMAKE_COMMAND}
    CMAKE_CACHE_ARGS
      ${SV_COMMON_CMAKE_CACHE_ARGS}
      -DSV_USE_TRILINOS:BOOL=OFF
      -DSV_PETSC_DIR:STRING=${SV_PETSC_DIR_VALUE}
      )
endif()
#-----------------------------------------------------------------------------
