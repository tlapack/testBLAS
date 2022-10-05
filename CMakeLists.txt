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

option( TEST_MPFR "Use mpreal from MPFR C++ library (http://www.holoborodko.com/pavel/mpfr/) for testing." OFF )

#-------------------------------------------------------------------------------
# Modules

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#-------------------------------------------------------------------------------
# Packages

# Load C++ BLAS wrappers
if( NOT TARGET blaspp )
  find_package( blaspp REQUIRED )
endif()

# Load Catch2
include( "${testBLAS_SOURCE_DIR}/cmake/FetchPackage.cmake" )
FetchPackage( "Catch2" "https://github.com/catchorg/Catch2.git" "v3.0.1" )
if( EXISTS "${Catch2_SOURCE_DIR}" )
  list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
else()
  list(APPEND CMAKE_MODULE_PATH "${Catch2_DIR}")
endif()

# Search for MPFR library if TEST_MPFR = ON
if( TEST_MPFR )
  
  find_package( GMP  4.2.1 REQUIRED ) # Load GMP
  find_package( MPFR 2.3.1 REQUIRED ) # Load MPFR
  
  # Verify if mpreal.h is installed
  find_path( MPREAL_PATH
    NAMES mpreal.h
    PATHS ${MPFR_INCLUDES} ${GMP_INCLUDES} )
  
  # Abort if mpreal.h is not installed
  if( MPREAL_PATH-NOTFOUND )
    message( FATAL_ERROR "MPFR C++ (MPREAL) not found. Try installing mpreal.h from
    (1) https://github.com/advanpix/mpreal,
 or (2) http://www.holoborodko.com/pavel/mpfr." )
  endif()

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
target_link_libraries( tester PRIVATE Catch2::Catch2WithMain blaspp )

if( TEST_MPFR )
  target_compile_definitions( tester PRIVATE USE_MPFR )
  target_include_directories( tester PRIVATE ${MPFR_INCLUDES} ${GMP_INCLUDES} )
  target_link_libraries( tester PRIVATE ${MPFR_LIBRARIES} ${GMP_LIBRARIES} )
endif()

# Add tests to CTest
include(CTest)
include(Catch)
catch_discover_tests(tester)
