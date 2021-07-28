// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TESTBLAS_DEFINES_HH__
#define __TESTBLAS_DEFINES_HH__

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
    #include <mpreal.h>
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

//-----------------------------------------------------------------------------

// The following complex values have unspecified behavior in the C++ standard.
//
// Complex( inf, nan),
// Complex( nan, inf),
// Complex(-inf, nan),
// Complex( nan,-inf)
//
// For instance, std::abs applied to std::complex<T>( Inf, NaN ) returns Inf
// if T is either float, double, or long double. However, the same function
// call returns a NaN if T is the multiprecision type mpfr::mpreal from
// https://github.com/advanpix/mpreal. We obtain the same pattern for
// (-Inf + NaN*i), (NaN + Inf*i) and (NaN - Inf*i).
// Moreover, if T = mpfr::mpreal, std::abs returns a NaN for any of the inputs
// (Inf + Inf*i), (-Inf + Inf*i), (Inf + Inf*i), (Inf - Inf*i), (Inf + 0*i),
// and (0 - Inf*i). std::abs returns 0 for the input (0 + NaN*i).
// 
// Another curious operation is the complex division. For each of the standard
// types, float, double and long double, the divisions ( 0 + 0*i )/( Inf + NaN*i ),
// ( 0 + 0*i )/( -Inf + NaN*i ), ( 0 + 0*i )/( NaN + Inf*i ) and
// ( 0 + 0*i )/( NaN - Inf*i ) all return ( 0 + 0*i ).
//
#ifndef TEST_CPLX_NAN
    #define TEST_CPLX_NAN { \
        Complex( nan, nan), \
        Complex( nan, 0.0), \
        Complex( 0.0, nan) }
#endif
    
#ifndef TEST_CPLX_INF
    #define TEST_CPLX_INF { \
        Complex( inf, inf), \
        Complex( inf,-inf), \
        Complex(-inf, inf), \
        Complex(-inf,-inf), \
        Complex( inf, 0.0), \
        Complex( 0.0, inf), \
        Complex(-inf, 0.0), \
        Complex( 0.0,-inf) }
#endif

#endif // __TESTBLAS_DEFINES_HH__
