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

# Load <T>LAPACK
if( NOT TARGET tlapack )
  find_package( tlapack REQUIRED HINTS ${tlapack_DIR} )
endif()

# Load Catch2
find_package( Catch2 QUIET HINTS ${Catch2_DIR} ) # Try to load Catch2 from the system
if( NOT Catch2_FOUND )

  if  ( EXISTS "${Catch2_DIR}/include/catch.hpp" )

    add_subdirectory( ${Catch2_DIR} ${CMAKE_CURRENT_BINARY_DIR}/Catch2 )
    message( STATUS "Using Catch2 from ${Catch2_DIR}" )

    # Add folder with Catch.cmake to the CMAKE_MODULE_PATH
    set( CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${Catch2_DIR}/extras;${Catch2_DIR}/contrib" )

  elseif( EXISTS "$ENV{Catch2_DIR}/include/catch.hpp" )

    get_property( docString CACHE Catch2_DIR PROPERTY HELPSTRING )
    set( Catch2_DIR $ENV{Catch2_DIR} CACHE STRING "${docString}" FORCE )
    add_subdirectory( ${Catch2_DIR} ${CMAKE_CURRENT_BINARY_DIR}/Catch2 )
    message( STATUS "Using Catch2 from ${Catch2_DIR}" )

    # Add folder with Catch.cmake to the CMAKE_MODULE_PATH
    set( CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${Catch2_DIR}/extras;${Catch2_DIR}/contrib" )

  elseif( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.14" )

    message( STATUS "Catch2 not found. Trying to fetch from https://github.com/catchorg/Catch2.git. "
                    "It may take a while." )

    include(FetchContent)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v2.13.1 )

    FetchContent_MakeAvailable(Catch2)

    # Add folder with Catch.cmake to the CMAKE_MODULE_PATH
    set( CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${Catch2_SOURCE_DIR}/extras;${Catch2_SOURCE_DIR}/contrib" )

    # Test if the fetch was successful
    if( EXISTS "${Catch2_SOURCE_DIR}/include/catch.hpp" )
      message( STATUS "Using Catch2 from https://github.com/catchorg/Catch2.git." )
    else()
      message( FATAL_ERROR "Failed in fetching Catch2 from https://github.com/catchorg/Catch2.git." )
    endif()

    # Hide Catch2_DIR, CATCH_ and FETCHCONTENT_ options
    mark_as_advanced( FORCE Catch2_DIR )
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
      if( "${_variableName}" MATCHES "^CATCH_" OR
          "${_variableName}" MATCHES "^FETCHCONTENT_" )
        mark_as_advanced( FORCE ${_variableName} )
      endif()
    endforeach()

  else()
    message( FATAL_ERROR "Catch2 not found. Set \"Catch2_DIR\" to a directory containing either
    (1) \"include/catch.hpp\",
 or (2) one of the files: Catch2Config.cmake, catch2-config.cmake." )
  endif()

endif( NOT Catch2_FOUND )

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

if( NOT TARGET tester )

  add_executable( tester tests_main.cpp ${test_sources} )

  target_include_directories( tester PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include" )
  target_link_libraries( tester PRIVATE Catch2::Catch2 tblas )

  if( TEST_MPFR )
    target_compile_definitions( tester PRIVATE USE_MPFR )
    target_include_directories( tester PRIVATE ${MPFR_INCLUDES} ${GMP_INCLUDES} )
    target_link_libraries( tester PRIVATE ${MPFR_LIBRARIES} ${GMP_LIBRARIES} )
  endif()

  set_target_properties( tester
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )

  # Add tests to CTest
  include(Catch)
  catch_discover_tests(tester)

else()

  if( TARGET tester )
    target_sources( tester PRIVATE ${test_sources} )
    target_include_directories( tester PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include" )

    if( TEST_MPFR )
      target_compile_definitions( tester PRIVATE USE_MPFR )
      target_include_directories( tester PRIVATE ${MPFR_INCLUDES} ${GMP_INCLUDES} )
    endif()
  endif()

endif()