// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TESTBLAS_TYPES_HH__
#define __TESTBLAS_TYPES_HH__

#include <complex>

#define TEST_REAL_STD_TYPES \
    float, \
    double, \
    long double

#define TEST_CPLX_STD_TYPES \
    std::complex<float>, \
    std::complex<double>, \
    std::complex<long double>

//-----------------------------------------------------------------------------
#ifdef USE_MPFR
    #include <mpreal.h>
    #define TEST_REAL_TYPES TEST_REAL_STD_TYPES, mpfr::mpreal
    #define TEST_CPLX_TYPES TEST_CPLX_STD_TYPES, std::complex<mpfr::mpreal>
#else
    #define TEST_REAL_TYPES TEST_REAL_STD_TYPES
    #define TEST_CPLX_TYPES TEST_CPLX_STD_TYPES
#endif

//-----------------------------------------------------------------------------
#define TEST_TYPES TEST_REAL_TYPES, TEST_CPLX_TYPES

//-----------------------------------------------------------------------------
#define TEST_CPLX_NAN { \
    Complex( nan, 0.0), \
    Complex( 0.0, nan), \
    Complex( inf, nan), \
    Complex( nan, inf), \
    Complex(-inf, nan), \
    Complex( nan,-inf) }
#define TEST_CPLX_INF { \
    Complex( inf, 0.0), \
    Complex( 0.0, inf), \
    Complex(-inf, 0.0), \
    Complex( 0.0,-inf) }

#endif // __TESTBLAS_TYPES_HH__