# testBLAS
C++ tester for BLAS

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![GitHub Workflow Status](https://github.com/tlapack/testBLAS/actions/workflows/cmake.yml/badge.svg)](https://github.com/tlapack/testBLAS/actions/workflows/cmake.yml)

## About

testBLAS
+ is based on the [Catch2](https://github.com/catchorg/Catch2) unit testing framework for C++.
+ uses the C++ BLAS interface of [\<T\>LAPACK](https://github.com/tlapack/tlapack).
+ uses [BLAS++](https://bitbucket.org/icl/blaspp) wrappers as the interface to vendor BLAS.

*Supported in part by [NSF ACI 2004850](http://www.nsf.gov/awardsearch/showAward?AWD_ID=2004850).*

## How to build testBLAS

testBLAS is built and installed with [CMake](https://cmake.org/).

### Getting CMake

You can either download binaries for the [latest stable](https://cmake.org/download/#latest) or [previous](https://cmake.org/download/#previous) release of CMake, or build the [current development distribution](https://github.com/Kitware/CMake) from source. CMake is also available in the APT repository on Ubuntu 16.04 or higher.

### Building testBLAS

testBLAS can be built using the standard CMake recipe

```sh
mkdir build
cmake -B build      # configuration step
cmake --build build # build step
```

### CMake options

Standard environment variables affect CMake. Some examples are

    CXX                 C++ compiler
    CXXFLAGS            C++ compiler flags
    LDFLAGS             linker flags

* [This page](https://cmake.org/cmake/help/latest/manual/cmake-env-variables.7.html) lists the environment variables that have special meaning to CMake.

It is also possible to pass variables to CMake during the configuration step using the `-D` flag. The following example builds testBLAS in debug mode inside the directory `build`.

```sh
mkdir build
cmake -B build -D CMAKE_BUILD_TYPE=Debug .
cmake --build build
```

* [This page](https://cmake.org/cmake/help/latest/manual/cmake-variables.7.html) documents variables that are provided by CMake or have meaning to CMake when set by project code.

### testBLAS options

Here are the testBLAS specific options and their default values

    # Option                         # Default

    BUILD_EXAMPLES                   ON
        
        Build examples
    
    TEST_MPFR                        OFF

        Use mpreal from MPFR C++ library
        (http://www.holoborodko.com/pavel/mpfr/) for testing.

## License

BSD 3-Clause License

Copyright (c) 2021, University of Colorado Denver. All rights reserved.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.