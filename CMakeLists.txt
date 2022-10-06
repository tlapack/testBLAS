# Copyright (c) 2021, University of Colorado Denver. All rights reserved.
#
# This file is part of testBLAS.
# testBLAS is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.1)

#-------------------------------------------------------------------------------
# Read project version

file(READ version.txt version_txt)

#-------------------------------------------------------------------------------
# Test BLAS project

project(testBLAS
    VERSION ${version_txt}
    LANGUAGES CXX)
message(STATUS "testBLAS version " ${PROJECT_VERSION})

#-------------------------------------------------------------------------------
# Options

set( cpp_blas_lib blaspp tlapack )

set( TESTBLAS_BLAS blaspp CACHE STRING "C++ package used to access BLAS routines." )
set_property( CACHE TESTBLAS_BLAS PROPERTY STRINGS ${cpp_blas_lib} )

#-------------------------------------------------------------------------------
# Modules

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#-------------------------------------------------------------------------------
# Packages

# Load C++ BLAS wrappers
if( NOT TARGET ${TESTBLAS_BLAS} )
  find_package( ${TESTBLAS_BLAS} REQUIRED )
endif()

# Load Catch2
include( "${testBLAS_SOURCE_DIR}/cmake/FetchPackage.cmake" )
FetchPackage( "Catch2" "https://github.com/catchorg/Catch2.git" "v3.0.1" )
if( EXISTS "${Catch2_SOURCE_DIR}" )
  list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
else()
  list(APPEND CMAKE_MODULE_PATH "${Catch2_DIR}")
endif()

#-------------------------------------------------------------------------------
# Targets

# corner-cases: Generate C++ file for tests with corner cases
add_custom_target( corner-cases
COMMAND
  cd "${CMAKE_CURRENT_SOURCE_DIR}/scripts" &&
      "${CMAKE_CURRENT_SOURCE_DIR}/scripts/wrapper_tests.py"
    > "${CMAKE_CURRENT_SOURCE_DIR}/src/test_corner_cases.cpp" )

# Test sources
file( GLOB test_sources
  "${CMAKE_CURRENT_SOURCE_DIR}/src/test_*.cpp" )

#-------------------------------------------------------------------------------
# tester: Program for tests

add_executable( tester ${test_sources} )

target_include_directories( tester PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include" )
target_link_libraries( tester PRIVATE Catch2::Catch2WithMain ${TESTBLAS_BLAS} )
if( TESTBLAS_BLAS STREQUAL "tlapack" )
  target_compile_definitions( tester PRIVATE TESTBLAS_USE_TLAPACK )
else()
  target_compile_definitions( tester PRIVATE TESTBLAS_USE_BLASPP )
endif()

# Add tests to CTest
include(CTest)
include(Catch)
catch_discover_tests(tester)
