# Workaround for a CMake issue with LLVM Flang and the NMake Makefiles generator:
# CMake appends the Fortran runtime link flags after the end marker of the
# inline response file of the link rule, which NMake cannot parse.
# Move the end marker to the end of the rule.
# Use with -DCMAKE_USER_MAKE_RULES_OVERRIDE_Fortran=<this file>
# and -DCMAKE_TRY_COMPILE_PLATFORM_VARIABLES=CMAKE_USER_MAKE_RULES_OVERRIDE_Fortran
if(CMAKE_GENERATOR MATCHES "NMake" AND CMAKE_Fortran_COMPILER_ID STREQUAL "LLVMFlang")
  foreach(v CMAKE_Fortran_LINK_EXECUTABLE CMAKE_Fortran_CREATE_SHARED_LIBRARY CMAKE_Fortran_CREATE_SHARED_MODULE)
    if(DEFINED ${v})
      string(REPLACE "${CMAKE_END_TEMP_FILE}" "" ${v} "${${v}}")
      string(APPEND ${v} "${CMAKE_END_TEMP_FILE}")
    endif()
  endforeach()
endif()
