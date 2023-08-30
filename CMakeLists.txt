# Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
#
# This file is part of testBLAS.
# testBLAS is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

cmake_minimum_required(VERSION 3.14)
# VERSION 3.11: FetchContent_Declare
# VERSION 3.14: FetchContent_MakeAvailable

#-------------------------------------------------------------------------------
# Dependencies on other projects
include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        6e79e682b726f524310d55dec8ddac4e9c52fb5f # v3.4.0
)

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
find_package( Catch2 3 QUIET )
if( NOT Catch2_FOUND )
  message( STATUS "Making Catch2 available..." )
  FetchContent_MakeAvailable( Catch2 )
  list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
  message( STATUS "  Populated content: ${catch2_SOURCE_DIR}" )
else()
  list(APPEND CMAKE_MODULE_PATH "${Catch2_DIR}")
  message( STATUS "Found Catch2: ${Catch2_DIR}" )
endif()

#-------------------------------------------------------------------------------
# Targets

if( PROJECT_IS_TOP_LEVEL )

  # corner-cases: Generate C++ file for tests with corner cases
  add_custom_target( corner-cases
  COMMAND
    cd "${CMAKE_CURRENT_SOURCE_DIR}/scripts" &&
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/wrapper_tests.py"
      > "${CMAKE_CURRENT_SOURCE_DIR}/src/test_corner_cases.cpp" )

endif()

# Test sources
file( GLOB test_sources
  "${CMAKE_CURRENT_SOURCE_DIR}/src/test_*.cpp" )

#-------------------------------------------------------------------------------
# tester: Program for tests

add_executable( testBLAS ${test_sources} )

target_include_directories( testBLAS PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include" )
target_link_libraries( testBLAS PRIVATE Catch2::Catch2WithMain ${TESTBLAS_BLAS} )
if( TESTBLAS_BLAS STREQUAL "tlapack" )
  target_compile_definitions( testBLAS PRIVATE TESTBLAS_USE_TLAPACK )
else()
  target_compile_definitions( testBLAS PRIVATE TESTBLAS_USE_BLASPP )
endif()

# Add tests to CTest
include(CTest)
include(Catch)
catch_discover_tests(testBLAS)
