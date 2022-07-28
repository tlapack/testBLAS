// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TESTBLAS_DEFINES_HH
#define TESTBLAS_DEFINES_HH

#include <complex>

#ifndef TEST_REAL_STD_TYPES
    #define TEST_REAL_STD_TYPES \
        float, \
        double, \
        long double
#endif

#ifndef TEST_CPLX_STD_TYPES
    #define TEST_CPLX_STD_TYPES \
        std::complex<float>, \
        std::complex<double>, \
        std::complex<long double>
#endif

//-----------------------------------------------------------------------------
#ifdef USE_MPFR
    #ifndef TEST_REAL_TYPES
        #define TEST_REAL_TYPES TEST_REAL_STD_TYPES, mpfr::mpreal
    #endif
    #ifndef TEST_CPLX_TYPES
        #define TEST_CPLX_TYPES TEST_CPLX_STD_TYPES, std::complex<mpfr::mpreal>
    #endif
#else
    #ifndef TEST_REAL_TYPES
        #define TEST_REAL_TYPES TEST_REAL_STD_TYPES
    #endif
    #ifndef TEST_CPLX_TYPES
        #define TEST_CPLX_TYPES TEST_CPLX_STD_TYPES
    #endif
#endif

//-----------------------------------------------------------------------------
#ifndef TEST_TYPES
    #define TEST_TYPES TEST_REAL_TYPES, TEST_CPLX_TYPES
#endif
#ifndef TEST_STD_TYPES
    #define TEST_STD_TYPES TEST_REAL_STD_TYPES, TEST_CPLX_STD_TYPES
#endif

#endif // TESTBLAS_DEFINES_HH
