# testBLAS
C++ tester for BLAS

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![GitHub Workflow Status](https://github.com/tlapack/testBLAS/actions/workflows/cmake.yml/badge.svg)](https://github.com/tlapack/testBLAS/actions/workflows/cmake.yml)

## About

testBLAS
+ is based on the [Catch2](https://github.com/catchorg/Catch2) unit testing framework for C++.
+ uses the C++ BLAS interface of [\<T\>LAPACK](https://github.com/tlapack/tlapack).
+ uses [BLAS++](https://bitbucket.org/icl/blaspp) wrappers as the interface to vendor BLAS.

Some additional information:

1. The test files are in the [src](src) directory. To add a new test case, you just need to create a file `test_<something>.cpp` under the [src](src) directory and build testBLAS.

2. The test file [test_corner_cases.cpp](src/test_corner_cases.cpp) has corner cases tests based on the BLAS specifications from:
   - [Lawson CL, Hanson RJ, Kincaid DR, Krogh FT (1979) Basic Linear Algebra Subprograms for Fortran Usage. ACM Trans Math Softw 5:308–323.](https://doi.org/10.1145/355841.355847)
   - [Dongarra JJ, Du Croz J, Hammarling S, Hanson RJ (1988) An Extended Set of FORTRAN Basic Linear Algebra Subprograms. ACM Trans Math Softw 14:1–17.](https://doi.org/10.1145/42288.42291)
   - [Dongarra JJ, Du Croz J, Hammarling S, Duff IS (1990) A set of level 3 basic linear algebra subprograms. ACM Trans Math Softw 16:1–17.](https://doi.org/10.1145/77626.79170)

3. Currently, testBLAS uses the [default error handler from <T>LAPACK](https://github.com/tlapack/tlapack/blob/master/src/xerbla.cpp).

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

## Testing

There are 2 workflows that run automatically:

- the first one uses testBLAS to test the [BLAS templates in \<T\>LAPACK](https://github.com/tlapack/tlapack/include/blas).

- the second one uses testBLAS to test the [BLAS++ wrappers](https://bitbucket.org/icl/blaspp/src/master/include/blas/wrappers.hh) to the [Netlib Reference BLAS](https://github.com/Reference-LAPACK/lapack/tree/master/BLAS/SRC).

See https://github.com/tlapack/testBLAS/actions.

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