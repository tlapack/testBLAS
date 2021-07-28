// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TESTBLAS_UTILS_HH__
#define __TESTBLAS_UTILS_HH__

#include "defines.hpp"
#include <vector>
#include <limits>

namespace testBLAS {

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
    const real_t inf = std::numeric_limits<real_t>::infinity();
    const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
    nanVec = std::vector< Complex >( TEST_CPLX_NAN );
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
    infVec = std::vector< Complex >( TEST_CPLX_INF );
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

#endif // __TESTBLAS_UTILS_HH__