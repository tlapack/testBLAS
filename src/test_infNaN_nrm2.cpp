/// @file test_infNaN_nrm2.cpp
/// @brief Test cases for nrm2 with NaNs, Infs and the overflow threshold (OV).
//
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <tblas.hpp>
#include "test_types.hpp"
#include "utils.hpp"

#include <catch2/catch.hpp>
#include <limits>
#include <vector>
#include <complex>

using namespace blas;

// -----------------------------------------------------------------------------
// Auxiliary routines

/**
 * @brief Set Ak = -k + i*k
 */
template< typename real_t >
inline void set_complexk( std::complex<real_t>& Ak, blas::size_t k ){
    Ak = std::complex<real_t>( -k, k );
}
template< typename real_t >
inline void set_complexk( real_t& Ak, blas::size_t k ){
    static_assert( ! is_complex<real_t>::value, "real_t must be a Real type." );
}

/**
 * @brief Set Ak = OV * ((k+2)/(k+3)) * (1+i)
 */
template< typename real_t >
inline void set_complexOV( std::complex<real_t>& Ak, blas::size_t k ){
    const real_t OV = std::numeric_limits<real_t>::max();
    Ak = OV * (real_t)((k+2.)/(k+3.)) * std::complex<real_t>( 1, 1 );
}
template< typename real_t >
inline void set_complexOV( real_t& Ak, blas::size_t k ){
    static_assert( ! is_complex<real_t>::value, "real_t must be a Real type." );
}

// -----------------------------------------------------------------------------
// Test cases for nrm2 with Infs and NaNs at specific positions

/**
 * @brief Check if nrm2( n, A, 1 ) works as expected using exactly 1 NaN
 * 
 * NaN locations: @see testBLAS::set_array_locations
 * 
 * If checkWithInf == true:
 *       For NaN location above:
 *              Insert Inf in first non-NaN location
 *              Insert -Inf in first non-NaN location
 *              Ditto for last non-NaN location
 *              Ditto for first and last non-NaN locations
 * 
 * @param[in] n
 *      Size of A
 * @param[in] A
 *      Array with non-NAN data
 * @param[in] checkWithInf
 *      If true, run the test cases with Infs.
 */
template< typename TestType >
void check_nrm2_1nan(
    const blas::size_t n,
    TestType A[],
    const bool checkWithInf = true )
{
    using real_t = real_type<TestType>;
    
    std::vector<TestType> nan_vec;
    testBLAS::set_nan_vector( nan_vec );
        
    // Indexes for test
    std::vector<blas::size_t> k_vec;
    testBLAS::set_array_locations( n, k_vec );
    
    // Tests
    for (const auto& k : k_vec) {
        const TestType Ak = A[k];
        
        const blas::size_t infIdx1 = (k > 0) ? 0 : 1;
        const blas::size_t infIdx2 = (k < n-1) ? n-1 : n-2;

        for (const auto& aNAN : nan_vec) {

            // NaN in A[k]
            A[k] = aNAN;
            
            // No Infs
            CHECK( isnan( nrm2( n, A, 1 ) ) );

            if( checkWithInf && n > 1 ) {
                const real_t inf = std::numeric_limits<real_t>::infinity();
                const TestType AinfIdx1 = A[ infIdx1 ];

                // Inf in first non-NaN location
                A[ infIdx1 ] = inf;
                CHECK( isnan( nrm2( n, A, 1 ) ) );
                
                // -Inf in first non-NaN location
                A[ infIdx1 ] = -inf;
                CHECK( isnan( nrm2( n, A, 1 ) ) );

                if( n > 2 ) {
                    const TestType AinfIdx2 = A[ infIdx2 ];

                    // Inf in last non-NaN location
                    A[ infIdx2 ] = inf;
                    CHECK( isnan( nrm2( n, A, 1 ) ) );
                    
                    // -Inf in last non-NaN location
                    A[ infIdx2 ] = -inf;
                    CHECK( isnan( nrm2( n, A, 1 ) ) );
                    
                    // Inf in first and last non-NaN location
                    A[ infIdx1 ] = A[ infIdx2 ] = inf;
                    CHECK( isnan( nrm2( n, A, 1 ) ) );
                    
                    // -Inf in first and last non-NaN location
                    A[ infIdx1 ] = A[ infIdx2 ] = -inf;
                    CHECK( isnan( nrm2( n, A, 1 ) ) );
                
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

/**
 * @brief Check if nrm2( n, A, 1 ) works as expected using exactly 2 NaNs
 * 
 * NaN locations: @see testBLAS::set_array_pairLocations
 * 
 * If checkWithInf == true:
 *       For NaN location above:
 *              Insert Inf in first non-NaN location
 *              Insert -Inf in first non-NaN location
 *              Ditto for last non-NaN location
 *              Ditto for first and last non-NaN locations
 * 
 * @param[in] n
 *      Size of A
 * @param[in] A
 *      Array with non-NAN data
 * @param[in] checkWithInf
 *      If true, run the test cases with Infs.
 */
template< typename TestType >
void check_nrm2_2nans(
    const blas::size_t n,
    TestType A[],
    const bool checkWithInf = true )
{
    using real_t = real_type<TestType>;
    
    std::vector<TestType> nan_vec;
    testBLAS::set_nan_vector( nan_vec );
        
    // Indexes for test
    std::vector<blas::size_t> k_vec;
    testBLAS::set_array_pairLocations( n, k_vec );
    
    // Tests
    for (unsigned i = 0; i < k_vec.size(); i += 2) {
        
        const auto& k1 = k_vec[i];
        const auto& k2 = k_vec[i+1];
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
            CHECK( isnan( nrm2( n, A, 1 ) ) );

            if( checkWithInf && n > 2 ) {
                const real_t inf = std::numeric_limits<real_t>::infinity();
                const TestType AinfIdx1 = A[ infIdx1 ];

                // Inf in first non-NaN location
                A[ infIdx1 ] = inf;
                CHECK( isnan( nrm2( n, A, 1 ) ) );
                
                // -Inf in first non-NaN location
                A[ infIdx1 ] = -inf;
                CHECK( isnan( nrm2( n, A, 1 ) ) );

                if( n > 3 ) {
                    const TestType AinfIdx2 = A[ infIdx2 ];

                    // Inf in last non-NaN location
                    A[ infIdx2 ] = inf;
                    CHECK( isnan( nrm2( n, A, 1 ) ) );
                    
                    // -Inf in last non-NaN location
                    A[ infIdx2 ] = -inf;
                    CHECK( isnan( nrm2( n, A, 1 ) ) );
                    
                    // Inf in first and last non-NaN location
                    A[ infIdx1 ] = A[ infIdx2 ] = inf;
                    CHECK( isnan( nrm2( n, A, 1 ) ) );
                    
                    // -Inf in first and last non-NaN location
                    A[ infIdx1 ] = A[ infIdx2 ] = -inf;
                    CHECK( isnan( nrm2( n, A, 1 ) ) );
                
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

/**
 * @brief Check if nrm2( n, A, 1 ) works as expected using exactly 3 NaNs
 * 
 * NaN locations: @see testBLAS::set_array_trioLocations
 * 
 * If checkWithInf == true:
 *       For NaN location above:
 *              Insert Inf in first non-NaN location
 *              Insert -Inf in first non-NaN location
 *              Ditto for last non-NaN location
 *              Ditto for first and last non-NaN locations
 * 
 * @param[in] n
 *      Size of A
 * @param[in] A
 *      Array with non-NAN data
 * @param[in] checkWithInf
 *      If true, run the test cases with Infs.
 */
template< typename TestType >
void check_nrm2_3nans(
    const blas::size_t n,
    TestType A[],
    const bool checkWithInf = true )
{
    using real_t = real_type<TestType>;
    
    std::vector<TestType> nan_vec;
    testBLAS::set_nan_vector( nan_vec );
        
    // Indexes for test
    std::vector<blas::size_t> k_vec;
    testBLAS::set_array_trioLocations( n, k_vec );
    
    // Tests
    for (unsigned i = 0; i < k_vec.size(); i += 3) {
        
        const auto& k1 = k_vec[i];
        const auto& k2 = k_vec[i+1];
        const auto& k3 = k_vec[i+2];
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
            CHECK( isnan( nrm2( n, A, 1 ) ) );

            if( checkWithInf && n > 3 ) {
                const real_t inf = std::numeric_limits<real_t>::infinity();
                const TestType AinfIdx1 = A[ infIdx1 ];

                // Inf in first non-NaN location
                A[ infIdx1 ] = inf;
                CHECK( isnan( nrm2( n, A, 1 ) ) );
                
                // -Inf in first non-NaN location
                A[ infIdx1 ] = -inf;
                CHECK( isnan( nrm2( n, A, 1 ) ) );

                if( n > 4 ) {
                    const TestType AinfIdx2 = A[ infIdx2 ];

                    // Inf in last non-NaN location
                    A[ infIdx2 ] = inf;
                    CHECK( isnan( nrm2( n, A, 1 ) ) );
                    
                    // -Inf in last non-NaN location
                    A[ infIdx2 ] = -inf;
                    CHECK( isnan( nrm2( n, A, 1 ) ) );
                    
                    // Inf in first and last non-NaN location
                    A[ infIdx1 ] = A[ infIdx2 ] = inf;
                    CHECK( isnan( nrm2( n, A, 1 ) ) );
                    
                    // -Inf in first and last non-NaN location
                    A[ infIdx1 ] = A[ infIdx2 ] = -inf;
                    CHECK( isnan( nrm2( n, A, 1 ) ) );
                
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

/**
 * @brief Check if nrm2( n, A, 1 ) works as expected using exactly 1 Inf
 * 
 * Inf locations: @see testBLAS::set_array_locations
 * 
 * @param[in] n
 *      Size of A.
 * @param[in] A
 *      Array with finite values.
 */

template< typename TestType >
void check_nrm2_1inf(
    const blas::size_t n,
    TestType A[] )
{    
    std::vector<TestType> inf_vec;
    testBLAS::set_inf_vector( inf_vec );
        
    // Indexes for test
    std::vector<blas::size_t> k_vec;
    testBLAS::set_array_locations( n, k_vec );
    
    // Tests
    for (const auto& k : k_vec) {
        const TestType Ak = A[k];

        for (const auto& aInf : inf_vec) {

            // (-1)^k*Inf in A[k]
            A[k] = ( k % 2 == 0 ) ? aInf : -aInf;
            
            // No Infs
            CHECK( isinf( nrm2( n, A, 1 ) ) );
            if( !isinf( nrm2( n, A, 1 ) ) )
                WARN( nrm2( n, A, 1 ) );

            // Reset value
            A[k] = Ak;
        }
    }
}

/**
 * @brief Check if nrm2( n, A, 1 ) works as expected using exactly 2 Infs
 * 
 * Inf locations: @see testBLAS::set_array_pairLocations
 * 
 * @param[in] n
 *      Size of A.
 * @param[in] A
 *      Array with finite values.
 */
template< typename TestType >
void check_nrm2_2infs(
    const blas::size_t n,
    TestType A[] )
{    
    std::vector<TestType> inf_vec;
    testBLAS::set_inf_vector( inf_vec );
        
    // Indexes for test
    std::vector<blas::size_t> k_vec;
    testBLAS::set_array_pairLocations( n, k_vec );
    
    // Tests
    for (unsigned i = 0; i < k_vec.size(); i += 2) {
        
        const auto& k1 = k_vec[i];
        const auto& k2 = k_vec[i+1];
        const TestType Ak1 = A[k1];
        const TestType Ak2 = A[k2];

        for (const auto& aInf : inf_vec) {

            // (-1)^k*Inf in A[k]
            A[k1] = ( k1 % 2 == 0 ) ? aInf : -aInf;
            A[k2] = ( k2 % 2 == 0 ) ? aInf : -aInf;
            
            CHECK( isinf( nrm2( n, A, 1 ) ) );

            // Reset values
            A[k1] = Ak1;
            A[k2] = Ak2;
        }
    }
}

/**
 * @brief Check if nrm2( n, A, 1 ) works as expected using exactly 3 Infs
 * 
 * Inf locations: @see testBLAS::set_array_trioLocations
 * 
 * @param[in] n
 *      Size of A.
 * @param[in] A
 *      Array with finite values.
 */
template< typename TestType >
void check_nrm2_3infs(
    const blas::size_t n,
    TestType A[] )
{    
    std::vector<TestType> inf_vec;
    testBLAS::set_inf_vector( inf_vec );
        
    // Indexes for test
    std::vector<blas::size_t> k_vec;
    testBLAS::set_array_trioLocations( n, k_vec );
    
    // Tests
    for (unsigned i = 0; i < k_vec.size(); i += 3) {
        
        const auto& k1 = k_vec[i];
        const auto& k2 = k_vec[i+1];
        const auto& k3 = k_vec[i+2];
        const TestType Ak1 = A[k1];
        const TestType Ak2 = A[k2];
        const TestType Ak3 = A[k3];

        for (const auto& aInf : inf_vec) {

            // (-1)^k*Inf in A[k]
            A[k1] = ( k1 % 2 == 0 ) ? aInf : -aInf;
            A[k2] = ( k2 % 2 == 0 ) ? aInf : -aInf;
            A[k3] = ( k3 % 2 == 0 ) ? aInf : -aInf;
            
            CHECK( isinf( nrm2( n, A, 1 ) ) );

            // Reset values
            A[k1] = Ak1;
            A[k2] = Ak2;
            A[k3] = Ak3;
        }
    }
}

// -----------------------------------------------------------------------------
// Main Test Cases

/**
 * @brief Test case for nrm2 with arrays containing at least 1 NaN
 * 
 * Default entries:
 *  (1) A[k] = (-1)^k*k
 *  (2) A[k] = (-1)^k*Inf
 * and, for complex data type: 
 *  (3) A[k] = -k + i*k for k even, and A[k] = OV*((k+2)/(k+3))*(1+i) for k odd
 *  (4) A[k] = OV*((k+2)/(k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd
 *  (5) A[k] = -k + i*k for k even, and A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k odd
 *  (6) A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd
 */
TEMPLATE_TEST_CASE( "nrm2 returns NaN for real arrays with at least 1 NaN",
                    "[nrm2][BLASlv1][NaN]", TEST_TYPES ) {
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
                check_nrm2_1nan( n, A );
                check_nrm2_2nans( n, A );
                check_nrm2_3nans( n, A );
            }
        }

        WHEN( "A[k] = (-1)^k*Inf" ) {
            for (blas::size_t k = 0; k < N; ++k)
                A[k] = ( k % 2 == 0 ) ? inf : -inf;
            for (const auto& n : n_vec) {
                check_nrm2_1nan( n, A, false );
                check_nrm2_2nans( n, A, false );
                check_nrm2_3nans( n, A, false );
            }
        }

        if (is_complex<TestType>::value) {

            WHEN( "A[k] = -k + i*k for k even, and A[k] = OV*((k+2)/(k+3))*(1+i) for k odd" ) {
                for (blas::size_t k = 0; k < N; ++k) {
                    if ( k % 2 == 0 ) set_complexk( A[k], k );
                    else              set_complexOV( A[k], k );
                }
                for (const auto& n : n_vec) {
                    check_nrm2_1nan( n, A );
                    check_nrm2_2nans( n, A );
                    check_nrm2_3nans( n, A );
                }
            }

            WHEN( "A[k] = OV*((k+2)/(k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
                for (blas::size_t k = 0; k < N; ++k) {
                    if ( k % 2 == 0 ) set_complexOV( A[k], k );
                    else              set_complexk( A[k], k );
                }
                for (const auto& n : n_vec) {
                    check_nrm2_1nan( n, A );
                    check_nrm2_2nans( n, A );
                    check_nrm2_3nans( n, A );
                }
            }

            WHEN( "A[k] = -k + i*k for k even, and A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k odd" ) {
                for (const auto& n : n_vec) {
                    for (blas::size_t k = 0; k < N; ++k) {
                        if ( k % 2 == 0 ) set_complexk( A[k], k );
                        else              set_complexOV( A[k], n-k );
                    }
                    check_nrm2_1nan( n, A );
                    check_nrm2_2nans( n, A );
                    check_nrm2_3nans( n, A );
                }
            }

            WHEN( "A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
                for (const auto& n : n_vec) {
                    for (blas::size_t k = 0; k < N; ++k) {
                        if ( k % 2 == 0 ) set_complexOV( A[k], n-k );
                        else              set_complexk( A[k], k );
                    }
                    check_nrm2_1nan( n, A );
                    check_nrm2_2nans( n, A );
                    check_nrm2_3nans( n, A );
                }
            }
        }
    }

    SECTION( "All NaNs" ) {
        for (blas::size_t k = 0; k < N; ++k)
            A[k] = NAN;
        for (const auto& n : n_vec)
            CHECK( isnan( nrm2( n, A, 1 ) ) );
    }
}

/**
 * @brief Test case for nrm2 with arrays containing at least 1 Inf and no NaNs
 * 
 * Default entries:
 *  (1) A[k] = (-1)^k*k
 * and, for complex data type: 
 *  (3) A[k] = -k + i*k for k even, and A[k] = OV*((k+2)/(k+3))*(1+i) for k odd
 *  (4) A[k] = OV*((k+2)/(k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd
 *  (5) A[k] = -k + i*k for k even, and A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k odd
 *  (6) A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd
 */
TEMPLATE_TEST_CASE( "nrm2 returns Inf for real arrays with at least 1 Inf and no NaNs",
                    "[nrm2][BLASlv1][Inf]", TEST_TYPES ) {
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
                check_nrm2_1inf( n, A );
                check_nrm2_2infs( n, A );
                check_nrm2_3infs( n, A );
            }
        }

        if (is_complex<TestType>::value) {

            WHEN( "A[k] = -k + i*k for k even, and A[k] = OV*((k+2)/(k+3))*(1+i) for k odd" ) {
                for (blas::size_t k = 0; k < N; ++k) {
                    if ( k % 2 == 0 ) set_complexk( A[k], k );
                    else              set_complexOV( A[k], k );
                }
                for (const auto& n : n_vec) {
                    check_nrm2_1inf( n, A );
                    check_nrm2_2infs( n, A );
                    check_nrm2_3infs( n, A );
                }
            }

            WHEN( "A[k] = OV*((k+2)/(k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
                for (blas::size_t k = 0; k < N; ++k) {
                    if ( k % 2 == 0 ) set_complexOV( A[k], k );
                    else              set_complexk( A[k], k );
                }
                for (const auto& n : n_vec) {
                    check_nrm2_1inf( n, A );
                    check_nrm2_2infs( n, A );
                    check_nrm2_3infs( n, A );
                }
            }

            WHEN( "A[k] = -k + i*k for k even, and A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k odd" ) {
                for (const auto& n : n_vec) {
                    for (blas::size_t k = 0; k < N; ++k) {
                        if ( k % 2 == 0 ) set_complexk( A[k], k );
                        else              set_complexOV( A[k], n-k );
                    }
                    check_nrm2_1inf( n, A );
                    check_nrm2_2infs( n, A );
                    check_nrm2_3infs( n, A );
                }
            }

            WHEN( "A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
                for (const auto& n : n_vec) {
                    for (blas::size_t k = 0; k < N; ++k) {
                        if ( k % 2 == 0 ) set_complexOV( A[k], n-k );
                        else              set_complexk( A[k], k );
                    }
                    check_nrm2_1inf( n, A );
                    check_nrm2_2infs( n, A );
                    check_nrm2_3infs( n, A );
                }
            }
        }
    }

    SECTION( "All Infs" ) {
        for (blas::size_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? inf : -inf;
        for (const auto& n : n_vec)
            CHECK( isinf( nrm2( n, A, 1 ) ) );
    }
}
