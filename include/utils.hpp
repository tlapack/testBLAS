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

    using std::isnan;
    using std::isinf;
    using std::abs;
    using std::ceil;
    using std::floor;
    using std::pow;

    // ------------------------------------------------------------------------
    /// isnan for complex numbers
    template< typename real_t >
    inline bool isnan( const std::complex<real_t>& x )
    {
        return isnan( real(x) ) || isnan( imag(x) );
    }

    // ------------------------------------------------------------------------
    /// isinf for complex numbers
    template< typename real_t >
    inline bool isinf( const std::complex<real_t>& x )
    {
        return isinf( real(x) ) || isinf( imag(x) );
    }

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

    /** Blue's min constant b for the sum of squares
     * @see https://doi.org/10.1145/355769.355771
     * @ingroup utils
     */
    template <typename real_t>
    inline constexpr real_t blue_min()
    {
        const real_t half( 0.5 );
        const int fradix = std::numeric_limits<real_t>::radix;
        const int expm   = std::numeric_limits<real_t>::min_exponent;

        return pow( fradix, ceil( half*(expm-1) ) );
    }

    /** Blue's max constant B for the sum of squares
     * @see https://doi.org/10.1145/355769.355771
     * @ingroup utils
     */
    template <typename real_t>
    inline constexpr real_t blue_max()
    {
        const real_t half( 0.5 );
        const int fradix = std::numeric_limits<real_t>::radix;
        const int expM   = std::numeric_limits<real_t>::max_exponent;
        const int t      = std::numeric_limits< real_t >::digits;

        return pow( fradix, floor( half*( expM - t + 1 ) ) );
    }

}

#endif // TESTBLAS_UTILS_HH
