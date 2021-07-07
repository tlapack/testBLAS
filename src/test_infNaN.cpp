/// @file test_infNaN.cpp
/// @brief Test cases for iamax with NaNs, Infs and the overflow threshold (OV).
//
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch.hpp>
#include <tblas.hpp>
#include "test_types.hpp"

#include <limits>
#include <vector>
#include <complex>

using namespace blas;

template< typename real_t >
inline void set_nan_vec(
    std::vector<real_t>& nan_vec )
{
    static_assert( ! is_complex<real_t>::value,
                    "real_t must be a Real type." );
    const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
    nan_vec = std::vector<real_t>({ 
        nan
    });
}

template< typename real_t >
inline void set_nan_vec(
    std::vector< std::complex<real_t> >& nan_vec )
{
    using T = std::complex<real_t>;
    const real_t inf = std::numeric_limits<real_t>::infinity();
    const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
    nan_vec = std::vector< T >({
        nan,
        T(0.0,nan),
        T(inf,nan),
        T(nan,inf),
        T(-inf,nan),
        T(nan,-inf)
    });
}

template< typename real_t >
inline void set_inf_vec( 
    std::vector<real_t>& inf_vec )
{
    static_assert( ! is_complex<real_t>::value,
                    "real_t must be a Real type." );
    const real_t inf = std::numeric_limits<real_t>::infinity();
    inf_vec = std::vector<real_t>({ inf });
}

template< typename real_t >
inline void set_inf_vec( 
    std::vector< std::complex<real_t> >& inf_vec )
{
    using T = std::complex<real_t>;
    const real_t inf = std::numeric_limits<real_t>::infinity();
    inf_vec = std::vector< T >({
        inf,
        T(0.0,inf),
        T(-inf,0.0),
        T(0.0,-inf)
    });
}

template< typename real_t >
inline void set_complex1( real_t& Ak, blas::size_t k ){
    static_assert( ! is_complex<real_t>::value,
                    "real_t must be a Real type." );
}

template< typename real_t >
inline void set_complex1( std::complex<real_t>& Ak, blas::size_t k ){
    Ak = std::complex<real_t>( -k, k );
}

template< typename real_t >
inline void set_complex2( real_t& Ak, blas::size_t k ){
    static_assert( ! is_complex<real_t>::value,
                    "real_t must be a Real type." );
}

template< typename real_t >
inline void set_complex2( std::complex<real_t>& Ak, blas::size_t k ){
    const real_t OV = std::numeric_limits<real_t>::max();
    Ak = OV * (real_t)((k+2.)/(k+3.)) * std::complex<real_t>( 1, 1 );
}

template< typename TestType >
void check_1nan(
    const blas::size_t n,
    TestType A[],
    const bool checkWithInf = true )
{
    using real_t = real_type<TestType>;
    
    std::vector<TestType> nan_vec;
    set_nan_vec( nan_vec );
        
    // Indexes for test
    const blas::size_t k_arr[] = { 0, 1, n-1, n/2 };
    const unsigned k_values = 
        ( n <= 1 ) ? 1 : (
        ( n == 2 ) ? 2 : (
        ( n == 3 ) ? 3
                   : 4 ));
    
    // Tests
    for (unsigned i = 0; i < k_values; i += 1) {
        
        const auto& k = k_arr[i];
        const TestType Ak = A[k];
        
        const blas::size_t infIdx1 = (k > 0) ? 0 : 1;
        const blas::size_t infIdx2 = (k < n-1) ? n-1 : n-2;

        for (const auto& aNAN : nan_vec) {

            // NaN in A[k]
            A[k] = aNAN;
            
            // No Infs
            CHECK( iamax( n, A, 1 ) == k );

            if( checkWithInf && n > 1 ) {
                const real_t inf = std::numeric_limits<real_t>::infinity();
                const TestType AinfIdx1 = A[ infIdx1 ];

                // Inf in first non-NaN location
                A[ infIdx1 ] = inf;
                CHECK( iamax( n, A, 1 ) == k );
                
                // -Inf in first non-NaN location
                A[ infIdx1 ] = -inf;
                CHECK( iamax( n, A, 1 ) == k );

                if( n > 2 ) {
                    const TestType AinfIdx2 = A[ infIdx2 ];

                    // Inf in last non-NaN location
                    A[ infIdx2 ] = inf;
                    CHECK( iamax( n, A, 1 ) == k );
                    
                    // -Inf in last non-NaN location
                    A[ infIdx2 ] = -inf;
                    CHECK( iamax( n, A, 1 ) == k );
                    
                    // Inf in first and last non-NaN location
                    A[ infIdx1 ] = A[ infIdx2 ] = inf;
                    CHECK( iamax( n, A, 1 ) == k );
                    
                    // -Inf in first and last non-NaN location
                    A[ infIdx1 ] = A[ infIdx2 ] = -inf;
                    CHECK( iamax( n, A, 1 ) == k );
                
                    // Reset value
                    A[ infIdx2 ] = AinfIdx2;
                }
                
                // Reset value
                A[ infIdx1 ] = AinfIdx1;
            }

            // Reset value
            A[k] = Ak;
        }
    }
}

template< typename TestType >
void check_2nan(
    const blas::size_t n,
    TestType A[],
    const bool checkWithInf = true )
{
    using real_t = real_type<TestType>;
    
    std::vector<TestType> nan_vec;
    set_nan_vec( nan_vec );
            
    // Indexes for test
    const blas::size_t k_arr[]
        = { 0, 1,
            0, n-1,
            1, n-1,
            0, n/2,
            1, n/2,
            n/2, n-1 };
    const unsigned k_values = 
        ( n <= 1 ) ? 0 : (
        ( n == 2 ) ? 2 : (
        ( n == 3 ) ? 6
                    : 12 ));
    
    // Tests
    for (unsigned i = 0; i < k_values; i += 2) {
        
        const auto& k1 = k_arr[i];
        const auto& k2 = k_arr[i+1];
        const TestType Ak1 = A[k1];
        const TestType Ak2 = A[k2];
        
        const blas::size_t infIdx1 =
            (k1 > 0) ? 0 : (
            (k2 > 1) ? 1
                        : 2 );
        const blas::size_t infIdx2 =
            (k2 < n-1) ? n-1 : (
            (k1 < n-2) ? n-2
                        : n-3 );

        for (const auto& aNAN : nan_vec) {

            // NaNs in A[k1] and A[k2]
            A[k1] = A[k2] = aNAN;
            
            // No Infs
            CHECK( iamax( n, A, 1 ) == k1 );

            if( checkWithInf && n > 2 ) {
                const real_t inf = std::numeric_limits<real_t>::infinity();
                const TestType AinfIdx1 = A[ infIdx1 ];

                // Inf in first non-NaN location
                A[ infIdx1 ] = inf;
                CHECK( iamax( n, A, 1 ) == k1 );
                
                // -Inf in first non-NaN location
                A[ infIdx1 ] = -inf;
                CHECK( iamax( n, A, 1 ) == k1 );

                if( n > 3 ) {
                    const TestType AinfIdx2 = A[ infIdx2 ];

                    // Inf in last non-NaN location
                    A[ infIdx2 ] = inf;
                    CHECK( iamax( n, A, 1 ) == k1 );
                    
                    // -Inf in last non-NaN location
                    A[ infIdx2 ] = -inf;
                    CHECK( iamax( n, A, 1 ) == k1 );
                    
                    // Inf in first and last non-NaN location
                    A[ infIdx1 ] = A[ infIdx2 ] = inf;
                    CHECK( iamax( n, A, 1 ) == k1 );
                    
                    // -Inf in first and last non-NaN location
                    A[ infIdx1 ] = A[ infIdx2 ] = -inf;
                    CHECK( iamax( n, A, 1 ) == k1 );
                
                    // Reset value
                    A[ infIdx2 ] = AinfIdx2;
                }
                
                // Reset value
                A[ infIdx1 ] = AinfIdx1;
            }

            // Reset values
            A[k1] = Ak1;
            A[k2] = Ak2;
        }
    }
}

template< typename TestType >
void check_3nan(
    const blas::size_t n,
    TestType A[],
    const bool checkWithInf = true )
{
    using real_t = real_type<TestType>;
    
    std::vector<TestType> nan_vec;
    set_nan_vec( nan_vec );
            
    // Indexes for test
    const blas::size_t k_arr[]
        = { 0, 1, n-1,
            0, 1, n/2,
            0, n/2, n-1,
            1, n/2, n-1 };
    const unsigned k_values = 
        ( n <= 2 ) ? 0 : (
        ( n == 3 ) ? 3
                    : 12 );
    
    // Tests
    for (unsigned i = 0; i < k_values; i += 3) {
        
        const auto& k1 = k_arr[i];
        const auto& k2 = k_arr[i+1];
        const auto& k3 = k_arr[i+2];
        const TestType Ak1 = A[k1];
        const TestType Ak2 = A[k2];
        const TestType Ak3 = A[k3];
        
        const blas::size_t infIdx1 =
            (k1 > 0) ? 0 : (
            (k2 > 1) ? 1
                        : 2 );
        const blas::size_t infIdx2 =
            (k2 < n-1) ? n-1 : (
            (k1 < n-2) ? n-2
                        : n-3 );

        for (const auto& aNAN : nan_vec) {

            // NaNs in A[k1], A[k2] and A[k3]
            A[k1] = A[k2] = A[k3] = aNAN;
        
            // No Infs
            CHECK( iamax( n, A, 1 ) == k1 );

            if( checkWithInf && n > 3 ) {
                const real_t inf = std::numeric_limits<real_t>::infinity();
                const TestType AinfIdx1 = A[ infIdx1 ];

                // Inf in first non-NaN location
                A[ infIdx1 ] = inf;
                CHECK( iamax( n, A, 1 ) == k1 );
                
                // -Inf in first non-NaN location
                A[ infIdx1 ] = -inf;
                CHECK( iamax( n, A, 1 ) == k1 );

                if( n > 4 ) {
                    const TestType AinfIdx2 = A[ infIdx2 ];

                    // Inf in last non-NaN location
                    A[ infIdx2 ] = inf;
                    CHECK( iamax( n, A, 1 ) == k1 );
                    
                    // -Inf in last non-NaN location
                    A[ infIdx2 ] = -inf;
                    CHECK( iamax( n, A, 1 ) == k1 );
                    
                    // Inf in first and last non-NaN location
                    A[ infIdx1 ] = A[ infIdx2 ] = inf;
                    CHECK( iamax( n, A, 1 ) == k1 );
                    
                    // -Inf in first and last non-NaN location
                    A[ infIdx1 ] = A[ infIdx2 ] = -inf;
                    CHECK( iamax( n, A, 1 ) == k1 );
                
                    // Reset value
                    A[ infIdx2 ] = AinfIdx2;
                }
                
                // Reset value
                A[ infIdx1 ] = AinfIdx1;
            }

            // Reset values
            A[k1] = Ak1;
            A[k2] = Ak2;
            A[k3] = Ak3;
        }
    }
}

template< typename TestType >
void check_1inf(
    const blas::size_t n,
    TestType A[] )
{    
    std::vector<TestType> inf_vec;
    set_inf_vec( inf_vec );
        
    // Indexes for test
    const blas::size_t k_arr[] = { 0, 1, n-1, n/2 };
    const unsigned k_values = 
        ( n <= 1 ) ? 1 : (
        ( n == 2 ) ? 2 : (
        ( n == 3 ) ? 3
                    : 4 ));
    
    // Tests
    for (unsigned i = 0; i < k_values; i += 1) {
        
        const auto& k = k_arr[i];
        const TestType Ak = A[k];

        for (const auto& aInf : inf_vec) {

            // (-1)^k*Inf in A[k]
            A[k] = ( k % 2 == 0 ) ? aInf : -aInf;
            
            // No Infs
            CHECK( iamax( n, A, 1 ) == k );

            // Reset value
            A[k] = Ak;
        }
    }
}

template< typename TestType >
void check_2inf(
    const blas::size_t n,
    TestType A[] )
{    
    std::vector<TestType> inf_vec;
    set_inf_vec( inf_vec );
            
    // Indexes for test
    const blas::size_t k_arr[]
        = { 0, 1,
            0, n-1,
            1, n-1,
            0, n/2,
            1, n/2,
            n/2, n-1 };
    const unsigned k_values = 
        ( n <= 1 ) ? 0 : (
        ( n == 2 ) ? 2 : (
        ( n == 3 ) ? 6
                    : 12 ));
    
    // Tests
    for (unsigned i = 0; i < k_values; i += 2) {
        
        const auto& k1 = k_arr[i];
        const auto& k2 = k_arr[i+1];
        const TestType Ak1 = A[k1];
        const TestType Ak2 = A[k2];

        for (const auto& aInf : inf_vec) {

            // (-1)^k*Inf in A[k]
            A[k1] = ( k1 % 2 == 0 ) ? aInf : -aInf;
            A[k2] = ( k2 % 2 == 0 ) ? aInf : -aInf;
            
            CHECK( iamax( n, A, 1 ) == k1 );

            // Reset values
            A[k1] = Ak1;
            A[k2] = Ak2;
        }
    }
}

template< typename TestType >
void check_3inf(
    const blas::size_t n,
    TestType A[] )
{    
    std::vector<TestType> inf_vec;
    set_inf_vec( inf_vec );
            
    // Indexes for test
    const blas::size_t k_arr[]
        = { 0, 1, n-1,
            0, 1, n/2,
            0, n/2, n-1,
            1, n/2, n-1 };
    const unsigned k_values = 
        ( n <= 2 ) ? 0 : (
        ( n == 3 ) ? 3
                    : 12 );
    
    // Tests
    for (unsigned i = 0; i < k_values; i += 3) {
        
        const auto& k1 = k_arr[i];
        const auto& k2 = k_arr[i+1];
        const auto& k3 = k_arr[i+2];
        const TestType Ak1 = A[k1];
        const TestType Ak2 = A[k2];
        const TestType Ak3 = A[k3];

        for (const auto& aInf : inf_vec) {

            // (-1)^k*Inf in A[k]
            A[k1] = ( k1 % 2 == 0 ) ? aInf : -aInf;
            A[k2] = ( k2 % 2 == 0 ) ? aInf : -aInf;
            A[k3] = ( k3 % 2 == 0 ) ? aInf : -aInf;
            
            CHECK( iamax( n, A, 1 ) == k1 );

            // Reset values
            A[k1] = Ak1;
            A[k2] = Ak2;
            A[k3] = Ak3;
        }
    }
}

/**
 * @brief Test case for iamax with arrays with at least 1 NaN
 * 
 * Default entries:
 *  (1) A[k] = (-1)^k*k
 *  (2) A[k] = (-1)^k*Inf
 * and, for complex data type: 
 *  (3) A[k] = -k + i*k for k even, and A[k] = OV*((k+2)/(k+3))*(1+i) for k odd
 *  (4) A[k] = OV*((k+2)/(k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd
 *  (5) A[k] = -k + i*k for k even, and A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k odd
 *  (6) A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd
 * 
 * Test cases:
 * 
 * (1) At least 1 NaN, no Infs (<=14 cases)
 *       1 NaN, at location:
 *              1;  2; n/2;  n
 *       2 NaNs (if possible, i.e. n>1, ditto later)
 *              1,2;   1,n/2;  1,n;   2,n/2;  2,n;  n/2,n
 *       3 NaNs
 *              1,2,n/2;  1,2,n;  1,n/2,n;  2,n/2,n
 * 
 * (2) At least 1 NaN and at least 1 Inf
 *       For each example above (<=7*14 cases):
 *              Insert Inf in first non-NaN location
 *              Insert -Inf in first non-NaN location
 *              Ditto for last non-NaN location
 *              Ditto for first and last non-NaN locations
 * 
 * (3) All NaNs
 */
TEMPLATE_TEST_CASE( "iamax returns the first NaN for real arrays with at least 1 NaN",
                    "[iamax][BLASlv1][NaN]", TEST_TYPES ) {
    using real_t = real_type<TestType>;

    // Constants
    const blas::size_t N = 128;       // N > 0
    const TestType inf = std::numeric_limits<real_t>::infinity();

    // Arrays
    const std::vector<blas::size_t> n_vec
        = { 1, 2, 3, 10, N }; // n_vec[i] > 0
    TestType A[N];

    SECTION( "At least 1 NaN in the array A" ) {

        WHEN( "A[k] = (-1)^k*k" ) {
            for (blas::size_t k = 0; k < N; ++k)
                A[k] = ( k % 2 == 0 ) ? k : -k;
            for (const auto& n : n_vec) {
                check_1nan( n, A );
                check_2nan( n, A );
                check_3nan( n, A );
            }
        }

        WHEN( "A[k] = (-1)^k*Inf" ) {
            for (blas::size_t k = 0; k < N; ++k)
                A[k] = ( k % 2 == 0 ) ? inf : -inf;
            for (const auto& n : n_vec) {
                check_1nan( n, A, false );
                check_2nan( n, A, false );
                check_3nan( n, A, false );
            }
        }

        if (is_complex<TestType>::value) {

            WHEN( "A[k] = -k + i*k for k even, and A[k] = OV*((k+2)/(k+3))*(1+i) for k odd" ) {
                for (blas::size_t k = 0; k < N; ++k) {
                    if ( k % 2 == 0 ) set_complex1( A[k], k );
                    else              set_complex2( A[k], k );
                }
                for (const auto& n : n_vec) {
                    check_1nan( n, A );
                    check_2nan( n, A );
                    check_3nan( n, A );
                }
            }

            WHEN( "A[k] = OV*((k+2)/(k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
                for (blas::size_t k = 0; k < N; ++k) {
                    if ( k % 2 == 0 ) set_complex2( A[k], k );
                    else              set_complex1( A[k], k );
                }
                for (const auto& n : n_vec) {
                    check_1nan( n, A );
                    check_2nan( n, A );
                    check_3nan( n, A );
                }
            }

            WHEN( "A[k] = -k + i*k for k even, and A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k odd" ) {
                for (const auto& n : n_vec) {
                    for (blas::size_t k = 0; k < N; ++k) {
                        if ( k % 2 == 0 ) set_complex1( A[k], k );
                        else              set_complex2( A[k], n-k );
                    }
                    check_1nan( n, A );
                    check_2nan( n, A );
                    check_3nan( n, A );
                }
            }

            WHEN( "A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
                for (const auto& n : n_vec) {
                    for (blas::size_t k = 0; k < N; ++k) {
                        if ( k % 2 == 0 ) set_complex2( A[k], n-k );
                        else              set_complex1( A[k], k );
                    }
                    check_1nan( n, A );
                    check_2nan( n, A );
                    check_3nan( n, A );
                }
            }
        }
    }

    SECTION( "All NaNs" ) {
        for (blas::size_t k = 0; k < N; ++k)
            A[k] = NAN;
        for (const auto& n : n_vec)
            CHECK( iamax( n, A, 1 ) == 0 );
    }
}

/**
 * @brief Test case for iamax with arrays with at least 1 Inf and no NaNs
 * 
 * Default entries:
 *  (1) A[k] = (-1)^k*k
 *  (2) A[k] = (-1)^k*Inf
 * and, for complex data type: 
 *  (3) A[k] = -k + i*k for k even, and A[k] = OV*((k+2)/(k+3))*(1+i) for k odd
 *  (4) A[k] = OV*((k+2)/(k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd
 *  (5) A[k] = -k + i*k for k even, and A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k odd
 *  (6) A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd
 * 
 * Test cases:
 * 
 * (1) At least 1 Inf (<=14 cases)
 *       1 Inf, at location:
 *              1;  2; n/2;  n
 *       2 Infs (if possible, i.e. n>1, ditto later)
 *              1,2;   1,n/2;  1,n;   2,n/2;  2,n;  n/2,n
 *       3 Infs
 *              1,2,n/2;  1,2,n;  1,n/2,n;  2,n/2,n
 * 
 * (3) All Infs
 */
TEMPLATE_TEST_CASE( "iamax returns the first Inf for real arrays with at least 1 Inf and no NaNs",
                    "[iamax][BLASlv1][Inf]", TEST_TYPES ) {
    using real_t = real_type<TestType>;

    // Constants
    const blas::size_t N = 128;       // N > 0
    const TestType inf = std::numeric_limits<real_t>::infinity();

    // Arrays
    const std::vector<blas::size_t> n_vec
        = { 1, 2, 3, 10, N }; // n_vec[i] > 0
    TestType A[N];

    SECTION( "At least 1 Inf in the array A" ) {

        WHEN( "A[k] = (-1)^k*k" ) {
            for (blas::size_t k = 0; k < N; ++k)
                A[k] = ( k % 2 == 0 ) ? k : -k;
            for (const auto& n : n_vec) {
                check_1inf( n, A );
                check_2inf( n, A );
                check_3inf( n, A );
            }
        }

        if (is_complex<TestType>::value) {

            WHEN( "A[k] = -k + i*k for k even, and A[k] = OV*((k+2)/(k+3))*(1+i) for k odd" ) {
                for (blas::size_t k = 0; k < N; ++k) {
                    if ( k % 2 == 0 ) set_complex1( A[k], k );
                    else              set_complex2( A[k], k );
                }
                for (const auto& n : n_vec) {
                    check_1inf( n, A );
                    check_2inf( n, A );
                    check_3inf( n, A );
                }
            }

            WHEN( "A[k] = OV*((k+2)/(k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
                for (blas::size_t k = 0; k < N; ++k) {
                    if ( k % 2 == 0 ) set_complex2( A[k], k );
                    else              set_complex1( A[k], k );
                }
                for (const auto& n : n_vec) {
                    check_1inf( n, A );
                    check_2inf( n, A );
                    check_3inf( n, A );
                }
            }

            WHEN( "A[k] = -k + i*k for k even, and A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k odd" ) {
                for (const auto& n : n_vec) {
                    for (blas::size_t k = 0; k < N; ++k) {
                        if ( k % 2 == 0 ) set_complex1( A[k], k );
                        else              set_complex2( A[k], n-k );
                    }
                    check_1inf( n, A );
                    check_2inf( n, A );
                    check_3inf( n, A );
                }
            }

            WHEN( "A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
                for (const auto& n : n_vec) {
                    for (blas::size_t k = 0; k < N; ++k) {
                        if ( k % 2 == 0 ) set_complex2( A[k], n-k );
                        else              set_complex1( A[k], k );
                    }
                    check_1inf( n, A );
                    check_2inf( n, A );
                    check_3inf( n, A );
                }
            }
        }
    }

    SECTION( "All Infs" ) {
        for (blas::size_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? inf : -inf;
        for (const auto& n : n_vec)
            CHECK( iamax( n, A, 1 ) == 0 );
    }
}

/**
 * @brief Test case for iamax where A(k) are finite but abs(real(A(k)))+abs(imag(A(k))) can overflow.
 * 
 * 4 cases:
 *     A(k) = -k + i*k for k even, A(k) = OV*((k+2)/(k+3)) + i*OV*((k+2)/(k+3)) for k odd.
 *              (Correct answer = last odd k)
 *     Swap odd and even. (Correct answer = last even k)
 *     A(k) = -k + i*k for k even, A(k) = OV*((n-k+2)/(n-k+3)) + i*OV*((n-k+2)/(n-k+3)) for k odd.
 *              (Correct answer = 1)
 *     Swap odd and even (Correct answer = 2).
 */
TEMPLATE_TEST_CASE( "iamax works for complex data A when abs(real(A(k)))+abs(imag(A(k))) can overflow",
                    "[iamax][BLASlv1]", TEST_CPLX_TYPES ) {
    // Constants
    const blas::size_t N = 128;       // N > 0

    // Arrays
    const std::vector<blas::size_t> n_vec
        = { 1, 2, 3, 10, N }; // n_vec[i] > 0
    TestType A[N];

    WHEN( "A[k] = -k + i*k for k even, and A[k] = OV*((k+2)/(k+3))*(1+i) for k odd" ) {
        for (blas::size_t k = 0; k < N; ++k) {
            if ( k % 2 == 0 ) set_complex1( A[k], k );
            else              set_complex2( A[k], k );
        }
        for (const auto& n : n_vec) {
            if ( n == 1 )          CHECK( iamax( n, A, 1 ) == 0 );
            else if ( n % 2 == 0 ) CHECK( iamax( n, A, 1 ) == n-1 );
            else                   CHECK( iamax( n, A, 1 ) == n-2 );
        }
    }

    WHEN( "A[k] = OV*((k+2)/(k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
        for (blas::size_t k = 0; k < N; ++k) {
            if ( k % 2 == 0 ) set_complex2( A[k], k );
            else              set_complex1( A[k], k );
        }
        for (const auto& n : n_vec) {
            if ( n == 1 )          CHECK( iamax( n, A, 1 ) == 0 );
            else if ( n % 2 == 0 ) CHECK( iamax( n, A, 1 ) == n-2 );
            else                   CHECK( iamax( n, A, 1 ) == n-1 );
        }
    }

    WHEN( "A[k] = -k + i*k for k even, and A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k odd" ) {
        for (const auto& n : n_vec) {
            for (blas::size_t k = 0; k < N; ++k) {
                if ( k % 2 == 0 ) set_complex1( A[k], k );
                else              set_complex2( A[k], n-k );
            }
            if ( n == 1 ) CHECK( iamax( n, A, 1 ) == 0 );
            else          CHECK( iamax( n, A, 1 ) == 1 );
        }
    }

    WHEN( "A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
        for (const auto& n : n_vec) {
            for (blas::size_t k = 0; k < N; ++k) {
                if ( k % 2 == 0 ) set_complex2( A[k], n-k );
                else              set_complex1( A[k], k );
            }
            CHECK( iamax( n, A, 1 ) == 0 );
        }
    }
}
