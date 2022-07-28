// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TESTBLAS_UTILS_HH
#define TESTBLAS_UTILS_HH

#include "defines.hpp"
#include <vector>
#include <limits>

namespace testBLAS {

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

/**
 * @brief Set the vector nanVec with NaNs for test
 */
template< typename real_t >
inline void set_nan_vector(
    std::vector<real_t>& nanVec )
{
    const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
    nanVec = std::vector<real_t>({ nan });
}

/**
 * @brief Set the vector nanVec with NaNs for test
 */
template< typename real_t >
inline void set_nan_vector(
    std::vector< std::complex<real_t> >& nanVec )
{
    using Complex = std::complex<real_t>;
    const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
    nanVec = std::vector< Complex >( {
        Complex( nan, nan),
        Complex( nan,  0 ),
        Complex(  0 , nan) } );
}

/**
 * @brief Set the vector nanVec with Infs for test
 */
template< typename real_t >
inline void set_inf_vector( 
    std::vector<real_t>& infVec )
{
    const real_t inf = std::numeric_limits<real_t>::infinity();
    infVec = std::vector<real_t>({ inf, -inf });
}

/**
 * @brief Set the vector nanVec with Infs for test
 */
template< typename real_t >
inline void set_inf_vector( 
    std::vector< std::complex<real_t> >& infVec )
{
    using Complex = std::complex<real_t>;
    const real_t inf = std::numeric_limits<real_t>::infinity();
    infVec = std::vector< Complex >( {
        Complex( inf, inf),
        Complex( inf,-inf),
        Complex(-inf, inf),
        Complex(-inf,-inf),
        Complex( inf,  0 ),
        Complex(  0 , inf),
        Complex(-inf,  0 ),
        Complex(  0 ,-inf) } );
}

/**
 * @brief Set the array indexes for tests
 * 
 * Locations: 0;  1; n/16; n/2;  n-1.
 * 
 */
template< typename int_t >
inline void set_array_locations(
    const int_t n,
    std::vector<int_t>& locVec ) 
{
    const int_t arr[] = { 0, 1, n-1, n/2, n/16 };
    const int_t numVals = 
        ( n <= 1 )  ? 1 : (
        ( n == 2 )  ? 2 : (
        ( n == 3 )  ? 3 : (
        ( n < 32 )  ? 4
                    : 5 )));
    locVec = std::vector<int_t>( arr, arr + numVals );
}

/**
 * @brief Set the pairs of indexes for tests
 * 
 * Locations: 0,1; 0,n/16;    0,n/2;    0,n-1;   
 *                 1,n/16;    1,n/2;    1,n-1;
 *                         n/16,n/2; n/16,n-1;
 *                                    n/2,n-1.
 */
template< typename int_t >
inline void set_array_pairLocations(
    const int_t n,
    std::vector<int_t>& locVec ) 
{
    const int_t arr[]
        = { 0, 1,
            0, n-1,
            1, n-1,
            0, n/2,
            1, n/2,
            n/2, n-1,
            0, n/16,
            1, n/16,
            n/16, n/2,
            n/16, n-1 };
    const int_t numVals = 
        ( n <= 1 )  ? 0 : (
        ( n == 2 )  ? 2 : (
        ( n == 3 )  ? 6 : (
        ( n < 32 )  ? 12
                    : 20 )));
    locVec = std::vector<int_t>( arr, arr + numVals );
}

/**
 * @brief Set the trios of indexes for tests
 * 
 * Locations: 0,1,n/16;    0,1,n/2;    0,1,n-1;
 *                      0,n/16,n/2; 0,n/16,n-1;
 *                                   0,n/2,n-1;
 *                      1,n/16,n/2; 1,n/16,n-1;
 *                                  1,n/16,n-1;
 *                                n/16,n/2,n-1.
 */
template< typename int_t >
inline void set_array_trioLocations(
    const int_t n,
    std::vector<int_t>& locVec ) 
{
    const int_t arr[]
        = { 0, 1, n-1,
            0, 1, n/2,
            0, n/2, n-1,
            1, n/2, n-1,
            0, 1, n/16,
            0, n/16, n/2,
            0, n/16, n-1,
            1, n/16, n/2,
            1, n/16, n-1,
            n/16, n/2, n-1 };
    const int_t numVals = 
        ( n <= 2 )  ? 0 : (
        ( n == 3 )  ? 3 : (
        ( n < 32 )  ? 12
                    : 30 ));
    locVec = std::vector<int_t>( arr, arr + numVals );
}

}

#endif // TESTBLAS_UTILS_HH