/// @file test_infNaN_iamax.cpp
/// @brief Test cases for iamax with NaNs, Infs and the overflow threshold (OV).
//
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <legacy_api/blas.hpp>
#include "defines.hpp"
#include "utils.hpp"
#ifdef USE_MPFR
    #include <plugins/tlapack_mpreal.hpp>
#endif

#include <limits>
#include <vector>
#include <complex>

using namespace tlapack;

// -----------------------------------------------------------------------------
// Auxiliary routines

/**
 * @brief Set Ak = -k + i*k
 */
template< typename real_t >
inline void set_complexk( std::complex<real_t>& Ak, idx_t k ){
    Ak = std::complex<real_t>( -k, k );
}
template< typename real_t >
inline void set_complexk( real_t& Ak, idx_t k ){
    static_assert( ! is_complex<real_t>::value, "real_t must be a Real type." );
}

/**
 * @brief Set Ak = OV * ((k+2)/(k+3)) * (1+i)
 */
template< typename real_t >
inline void set_complexOV( std::complex<real_t>& Ak, idx_t k ){
    const real_t OV = std::numeric_limits<real_t>::max();
    Ak = OV * (real_t)((k+2.)/(k+3.)) * std::complex<real_t>( 1, 1 );
}
template< typename real_t >
inline void set_complexOV( real_t& Ak, idx_t k ){
    static_assert( ! is_complex<real_t>::value, "real_t must be a Real type." );
}

// -----------------------------------------------------------------------------
// Test cases for iamax with Infs and NaNs at specific positions

/**
 * @brief Check if iamax( n, A, 1 ) works as expected using exactly 1 NaN
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
void check_iamax_1nan(
    const idx_t n,
    TestType A[],
    const bool checkWithInf = true )
{
    using real_t = real_type<TestType>;
    
    std::vector<TestType> nan_vec;
    testBLAS::set_nan_vector( nan_vec );
        
    // Indexes for test
    std::vector<idx_t> k_vec;
    testBLAS::set_array_locations( n, k_vec );
    
    // Tests
    for (const auto& k : k_vec) {
        const TestType Ak = A[k];
        
        const idx_t infIdx1 = (k > 0) ? 0 : 1;
        const idx_t infIdx2 = (k < n-1) ? n-1 : n-2;

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

/**
 * @brief Check if iamax( n, A, 1 ) works as expected using exactly 2 NaNs
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
void check_iamax_2nans(
    const idx_t n,
    TestType A[],
    const bool checkWithInf = true )
{
    using real_t = real_type<TestType>;
    
    std::vector<TestType> nan_vec;
    testBLAS::set_nan_vector( nan_vec );
        
    // Indexes for test
    std::vector<idx_t> k_vec;
    testBLAS::set_array_pairLocations( n, k_vec );
    
    // Tests
    for (unsigned i = 0; i < k_vec.size(); i += 2) {
        
        const auto& k1 = k_vec[i];
        const auto& k2 = k_vec[i+1];
        const TestType Ak1 = A[k1];
        const TestType Ak2 = A[k2];
        
        const idx_t infIdx1 =
            (k1 > 0) ? 0 : (
            (k2 > 1) ? 1
                       : 2 );
        const idx_t infIdx2 =
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

/**
 * @brief Check if iamax( n, A, 1 ) works as expected using exactly 3 NaNs
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
void check_iamax_3nans(
    const idx_t n,
    TestType A[],
    const bool checkWithInf = true )
{
    using real_t = real_type<TestType>;
    
    std::vector<TestType> nan_vec;
    testBLAS::set_nan_vector( nan_vec );
        
    // Indexes for test
    std::vector<idx_t> k_vec;
    testBLAS::set_array_trioLocations( n, k_vec );
    
    // Tests
    for (unsigned i = 0; i < k_vec.size(); i += 3) {
        
        const auto& k1 = k_vec[i];
        const auto& k2 = k_vec[i+1];
        const auto& k3 = k_vec[i+2];
        const TestType Ak1 = A[k1];
        const TestType Ak2 = A[k2];
        const TestType Ak3 = A[k3];
        
        const idx_t infIdx1 =
            (k1 > 0) ? 0 : (
            (k2 > 1) ? 1
                        : 2 );
        const idx_t infIdx2 =
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

/**
 * @brief Check if iamax( n, A, 1 ) works as expected using exactly 1 Inf
 * 
 * Inf locations: @see testBLAS::set_array_locations
 * 
 * @param[in] n
 *      Size of A.
 * @param[in] A
 *      Array with finite values.
 */

template< typename TestType >
void check_iamax_1inf(
    const idx_t n,
    TestType A[] )
{    
    std::vector<TestType> inf_vec;
    testBLAS::set_inf_vector( inf_vec );
        
    // Indexes for test
    std::vector<idx_t> k_vec;
    testBLAS::set_array_locations( n, k_vec );
    
    // Tests
    for (const auto& k : k_vec) {
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

/**
 * @brief Check if iamax( n, A, 1 ) works as expected using exactly 2 Infs
 * 
 * Inf locations: @see testBLAS::set_array_pairLocations
 * 
 * @param[in] n
 *      Size of A.
 * @param[in] A
 *      Array with finite values.
 */
template< typename TestType >
void check_iamax_2infs(
    const idx_t n,
    TestType A[] )
{    
    std::vector<TestType> inf_vec;
    testBLAS::set_inf_vector( inf_vec );
        
    // Indexes for test
    std::vector<idx_t> k_vec;
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
            
            CHECK( iamax( n, A, 1 ) == k1 );

            // Reset values
            A[k1] = Ak1;
            A[k2] = Ak2;
        }
    }
}

/**
 * @brief Check if iamax( n, A, 1 ) works as expected using exactly 3 Infs
 * 
 * Inf locations: @see testBLAS::set_array_trioLocations
 * 
 * @param[in] n
 *      Size of A.
 * @param[in] A
 *      Array with finite values.
 */
template< typename TestType >
void check_iamax_3infs(
    const idx_t n,
    TestType A[] )
{    
    std::vector<TestType> inf_vec;
    testBLAS::set_inf_vector( inf_vec );
        
    // Indexes for test
    std::vector<idx_t> k_vec;
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
            
            CHECK( iamax( n, A, 1 ) == k1 );

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
 * @brief Test case for iamax with arrays containing at least 1 NaN
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
TEMPLATE_TEST_CASE( "iamax returns the first NaN for arrays with at least 1 NaN",
                    "[iamax][BLASlv1][NaN]", TEST_TYPES ) {
    using real_t = real_type<TestType>;

    // Constants
    const idx_t N = 128;       // N > 0
    const TestType inf = std::numeric_limits<real_t>::infinity();

    // Arrays
    const std::vector<idx_t> n_vec
        = { 1, 2, 3, 10, N }; // n_vec[i] > 0
    TestType A[N];

    SECTION( "At least 1 NaN in the array A" ) {

        WHEN( "A[k] = (-1)^k*k" ) {
            for (idx_t k = 0; k < N; ++k)
                A[k] = ( k % 2 == 0 ) ? k : -k;
            for (const auto& n : n_vec) {
                check_iamax_1nan( n, A );
                check_iamax_2nans( n, A );
                check_iamax_3nans( n, A );
            }
        }

        WHEN( "A[k] = (-1)^k*Inf" ) {
            for (idx_t k = 0; k < N; ++k)
                A[k] = ( k % 2 == 0 ) ? inf : -inf;
            for (const auto& n : n_vec) {
                check_iamax_1nan( n, A, false );
                check_iamax_2nans( n, A, false );
                check_iamax_3nans( n, A, false );
            }
        }

        if (is_complex<TestType>::value) {

            WHEN( "A[k] = -k + i*k for k even, and A[k] = OV*((k+2)/(k+3))*(1+i) for k odd" ) {
                for (idx_t k = 0; k < N; ++k) {
                    if ( k % 2 == 0 ) set_complexk( A[k], k );
                    else              set_complexOV( A[k], k );
                }
                for (const auto& n : n_vec) {
                    check_iamax_1nan( n, A );
                    check_iamax_2nans( n, A );
                    check_iamax_3nans( n, A );
                }
            }

            WHEN( "A[k] = OV*((k+2)/(k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
                for (idx_t k = 0; k < N; ++k) {
                    if ( k % 2 == 0 ) set_complexOV( A[k], k );
                    else              set_complexk( A[k], k );
                }
                for (const auto& n : n_vec) {
                    check_iamax_1nan( n, A );
                    check_iamax_2nans( n, A );
                    check_iamax_3nans( n, A );
                }
            }

            WHEN( "A[k] = -k + i*k for k even, and A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k odd" ) {
                for (const auto& n : n_vec) {
                    for (idx_t k = 0; k < n; ++k) {
                        if ( k % 2 == 0 ) set_complexk( A[k], k );
                        else              set_complexOV( A[k], n-k );
                    }
                    check_iamax_1nan( n, A );
                    check_iamax_2nans( n, A );
                    check_iamax_3nans( n, A );
                }
            }

            WHEN( "A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
                for (const auto& n : n_vec) {
                    for (idx_t k = 0; k < n; ++k) {
                        if ( k % 2 == 0 ) set_complexOV( A[k], n-k );
                        else              set_complexk( A[k], k );
                    }
                    check_iamax_1nan( n, A );
                    check_iamax_2nans( n, A );
                    check_iamax_3nans( n, A );
                }
            }
        }
    }

    SECTION( "All NaNs" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = NAN;
        for (const auto& n : n_vec)
            CHECK( iamax( n, A, 1 ) == 0 );
    }
}

/**
 * @brief Test case for iamax with arrays containing at least 1 Inf and no NaNs
 * 
 * Default entries:
 *  (1) A[k] = (-1)^k*k
 * and, for complex data type: 
 *  (3) A[k] = -k + i*k for k even, and A[k] = OV*((k+2)/(k+3))*(1+i) for k odd
 *  (4) A[k] = OV*((k+2)/(k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd
 *  (5) A[k] = -k + i*k for k even, and A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k odd
 *  (6) A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd
 */
TEMPLATE_TEST_CASE( "iamax returns the first Inf for arrays with at least 1 Inf and no NaNs",
                    "[iamax][BLASlv1][Inf]", TEST_TYPES ) {
    using real_t = real_type<TestType>;

    // Constants
    const idx_t N = 128;       // N > 0
    const TestType inf = std::numeric_limits<real_t>::infinity();

    // Arrays
    const std::vector<idx_t> n_vec
        = { 1, 2, 3, 10, N }; // n_vec[i] > 0
    TestType A[N];

    SECTION( "At least 1 Inf in the array A" ) {

        WHEN( "A[k] = (-1)^k*k" ) {
            for (idx_t k = 0; k < N; ++k)
                A[k] = ( k % 2 == 0 ) ? k : -k;
            for (const auto& n : n_vec) {
                check_iamax_1inf( n, A );
                check_iamax_2infs( n, A );
                check_iamax_3infs( n, A );
            }
        }

        if (is_complex<TestType>::value) {

            WHEN( "A[k] = -k + i*k for k even, and A[k] = OV*((k+2)/(k+3))*(1+i) for k odd" ) {
                for (idx_t k = 0; k < N; ++k) {
                    if ( k % 2 == 0 ) set_complexk( A[k], k );
                    else              set_complexOV( A[k], k );
                }
                for (const auto& n : n_vec) {
                    check_iamax_1inf( n, A );
                    check_iamax_2infs( n, A );
                    check_iamax_3infs( n, A );
                }
            }

            WHEN( "A[k] = OV*((k+2)/(k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
                for (idx_t k = 0; k < N; ++k) {
                    if ( k % 2 == 0 ) set_complexOV( A[k], k );
                    else              set_complexk( A[k], k );
                }
                for (const auto& n : n_vec) {
                    check_iamax_1inf( n, A );
                    check_iamax_2infs( n, A );
                    check_iamax_3infs( n, A );
                }
            }

            WHEN( "A[k] = -k + i*k for k even, and A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k odd" ) {
                for (const auto& n : n_vec) {
                    for (idx_t k = 0; k < n; ++k) {
                        if ( k % 2 == 0 ) set_complexk( A[k], k );
                        else              set_complexOV( A[k], n-k );
                    }
                    check_iamax_1inf( n, A );
                    check_iamax_2infs( n, A );
                    check_iamax_3infs( n, A );
                }
            }

            WHEN( "A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
                for (const auto& n : n_vec) {
                    for (idx_t k = 0; k < n; ++k) {
                        if ( k % 2 == 0 ) set_complexOV( A[k], n-k );
                        else              set_complexk( A[k], k );
                    }
                    check_iamax_1inf( n, A );
                    check_iamax_2infs( n, A );
                    check_iamax_3infs( n, A );
                }
            }
        }
    }

    SECTION( "All Infs" ) {
        for (idx_t k = 0; k < N; ++k)
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
    const idx_t N = 128;       // N > 0

    // Arrays
    const std::vector<idx_t> n_vec
        = { 1, 2, 3, 10, N }; // n_vec[i] > 0
    TestType A[N];

    WHEN( "A[k] = -k + i*k for k even, and A[k] = OV*((k+2)/(k+3))*(1+i) for k odd" ) {
        for (idx_t k = 0; k < N; ++k) {
            if ( k % 2 == 0 ) set_complexk( A[k], k );
            else              set_complexOV( A[k], k );
        }
        for (const auto& n : n_vec) {
            if ( n == 1 )          CHECK( iamax( n, A, 1 ) == 0 );
            else if ( n % 2 == 0 ) CHECK( iamax( n, A, 1 ) == n-1 );
            else                   CHECK( iamax( n, A, 1 ) == n-2 );
        }
    }

    WHEN( "A[k] = OV*((k+2)/(k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
        for (idx_t k = 0; k < N; ++k) {
            if ( k % 2 == 0 ) set_complexOV( A[k], k );
            else              set_complexk( A[k], k );
        }
        for (const auto& n : n_vec) {
            if ( n == 1 )          CHECK( iamax( n, A, 1 ) == 0 );
            else if ( n % 2 == 0 ) CHECK( iamax( n, A, 1 ) == n-2 );
            else                   CHECK( iamax( n, A, 1 ) == n-1 );
        }
    }

    WHEN( "A[k] = -k + i*k for k even, and A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k odd" ) {
        for (const auto& n : n_vec) {
            for (idx_t k = 0; k < n; ++k) {
                if ( k % 2 == 0 ) set_complexk( A[k], k );
                else              set_complexOV( A[k], n-k );
            }
            if ( n == 1 ) CHECK( iamax( n, A, 1 ) == 0 );
            else          CHECK( iamax( n, A, 1 ) == 1 );
        }
    }

    WHEN( "A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
        for (const auto& n : n_vec) {
            for (idx_t k = 0; k < n; ++k) {
                if ( k % 2 == 0 ) set_complexOV( A[k], n-k );
                else              set_complexk( A[k], k );
            }
            CHECK( iamax( n, A, 1 ) == 0 );
        }
    }
}
