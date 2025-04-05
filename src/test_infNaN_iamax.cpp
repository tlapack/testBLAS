/// @file test_infNaN_iamax.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @brief Test cases for iamax with NaNs, Infs and the overflow threshold (OV).
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <limits>
#include <vector>
#include <complex>

#include "utils.hpp"

using namespace testBLAS;

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

template< typename TestType >
void check_iamax_nans_and_infs(
    const idx_t n,
    TestType A[],
    const idx_t firstNaN,
    const idx_t nNaNs,
    const idx_t infLocs[] )
{
    using real_t = real_type<TestType>;
    
    const real_t Inf = std::numeric_limits<real_t>::infinity();
    const std::vector<real_t> Infs = { -Inf, Inf };

    if( n > nNaNs ) {

        // 1 Inf, note that (i < n-nNaNs) prevents duplicates when n == nNaNs+1
        for(idx_t i=0; (i < 2) && (i < n-nNaNs); ++i) {
            
            const idx_t infLoc = infLocs[i];
            const TestType Ai = A[ infLoc ];
            
            for( const real_t& inf : Infs ) {

                // Inf in A[infLoc]
                A[ infLoc ] = inf;
                INFO( "A[" << infLoc << "] = " << A[infLoc] );
                CHECK( iamax( n, A, 1 ) == firstNaN );
                                            
            }

            // Reset value
            A[ infLoc ] = Ai;
        }

        // 2 Infs
        if( n > nNaNs+1 ) {

            const TestType A0 = A[ infLocs[0] ];
            const TestType A1 = A[ infLocs[1] ];

            for( const real_t& inf0 : Infs )
            for( const real_t& inf1 : Infs ) {

                // Inf in A[infLocs[0]]
                A[ infLocs[0] ] = inf0;
                INFO( "A[" << infLocs[0] << "] = " << A[infLocs[0]] );

                // Inf in A[infLocs[1]]
                A[ infLocs[1] ] = inf1;
                INFO( "A[" << infLocs[1] << "] = " << A[infLocs[1]] );

                CHECK( iamax( n, A, 1 ) == firstNaN );
            }

            // Reset value
            A[ infLocs[0] ] = A0;
            A[ infLocs[1] ] = A1;
        }
    }
}

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
    std::vector<TestType> nan_vec;
    testBLAS::set_nan_vector( nan_vec );
        
    // Indexes for test
    std::vector<idx_t> k_vec;
    testBLAS::set_array_locations( n, k_vec );
    
    // Tests
    for (const auto& k : k_vec) {

        const TestType Ak = A[k];

        const idx_t infLocs[] = {
            idx_t( (k > 0) ? 0 : 1 ),
            idx_t( (k < n-1) ? n-1 : n-2 )
        };

        for (const auto& aNAN : nan_vec) {

            // NaN in A[k]
            A[k] = aNAN;
            INFO( "A[" << k << "] = " << A[k] );
            
            UNSCOPED_INFO( "No Infs" );
            CHECK( iamax( n, A, 1 ) == k );

            if( checkWithInf )
                check_iamax_nans_and_infs( n, A, k, 1, infLocs );
        }

        // Reset value
        A[k] = Ak;
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

        const idx_t infLocs[] = {
            idx_t( (k1 > 0) ? 0 : ((k2 > 1) ? 1 : 2 ) ),
            idx_t( (k2 < n-1) ? n-1 : ((k1 < n-2) ? n-2 : n-3 ) )
        };

        for (const auto& aNAN : nan_vec) {

            // NaNs in A[k1] and A[k2]
            A[k1] = A[k2] = aNAN;
            INFO( "A[" << k1 << "] = " << A[k1] );
            INFO( "A[" << k2 << "] = " << A[k2] );

            UNSCOPED_INFO( "No Infs" );
            CHECK( iamax( n, A, 1 ) == k1 );

            if( checkWithInf )
                check_iamax_nans_and_infs( n, A, k1, 2, infLocs );
        }

        // Reset values
        A[k1] = Ak1;
        A[k2] = Ak2;
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

        const idx_t infLocs[] = {
            idx_t(  (k1 > 0)    ? 0 : (
                    (k2 > 1)    ? 1
                                : 2 ) ),
            idx_t(  (k2 < n-1)  ? n-1 : (
                    (k1 < n-2)  ? n-2
                                : n-3 ) )
        };

        for (const auto& aNAN : nan_vec) {

            // NaNs in A[k1], A[k2] and A[k3]
            A[k1] = A[k2] = A[k3] = aNAN;
            INFO( "A[" << k1 << "] = " << A[k1] );
            INFO( "A[" << k2 << "] = " << A[k2] );
            INFO( "A[" << k3 << "] = " << A[k3] );
        
            UNSCOPED_INFO( "No Infs" );
            CHECK( iamax( n, A, 1 ) == k1 );

            if( checkWithInf ) 
                check_iamax_nans_and_infs( n, A, k1, 3, infLocs );
        }

        // Reset values
        A[k1] = Ak1;
        A[k2] = Ak2;
        A[k3] = Ak3;
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
            INFO( "A[" << k << "] = " << A[k] );
            
            CHECK( iamax( n, A, 1 ) == k );
        }

        // Reset value
        A[k] = Ak;
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
            INFO( "A[" << k1 << "] = " << A[k1] );
            INFO( "A[" << k2 << "] = " << A[k2] );
            
            CHECK( iamax( n, A, 1 ) == k1 );
        }

        // Reset values
        A[k1] = Ak1;
        A[k2] = Ak2;
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
            INFO( "A[" << k1 << "] = " << A[k1] );
            INFO( "A[" << k2 << "] = " << A[k2] );
            INFO( "A[" << k3 << "] = " << A[k3] );
            
            CHECK( iamax( n, A, 1 ) == k1 );
        }

        // Reset values
        A[k1] = Ak1;
        A[k2] = Ak2;
        A[k3] = Ak3;
    }
}

// -----------------------------------------------------------------------------
// Main Test Cases

/**
 * @brief Test case for iamax with arrays containing at least 1 NaN
 * 
 * Default entries:
 *  (a) A[k] = (-1)^k*k
 *  (b) A[k] = (-1)^k*Inf
 * and, for complex data type: 
 *  (c) A[k] = -k + i*k for k even, and A[k] = OV*((k+2)/(k+3))*(1+i) for k odd
 *  (d) A[k] = OV*((k+2)/(k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd
 *  (e) A[k] = -k + i*k for k even, and A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k odd
 *  (f) A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd
 *  (g) A[k] = NaN
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

    SECTION( "(a) A[k] = (-1)^k*k" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? k : -k;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            check_iamax_1nan( n, A );
            check_iamax_2nans( n, A );
            check_iamax_3nans( n, A );
        }
    }

    SECTION( "(b) A[k] = (-1)^k*Inf" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? inf : -inf;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            check_iamax_1nan( n, A, false );
            check_iamax_2nans( n, A, false );
            check_iamax_3nans( n, A, false );
        }
    }

    if (is_complex<TestType>::value) {

        SECTION( "(c) A[k] = -k + i*k for k even, and A[k] = OV*((k+2)/(k+3))*(1+i) for k odd" ) {
            for (idx_t k = 0; k < N; ++k) {
                if ( k % 2 == 0 ) set_complexk( A[k], k );
                else              set_complexOV( A[k], k );
            }
            for (const auto& n : n_vec) {
                INFO( "n = " << n );
                check_iamax_1nan( n, A );
                check_iamax_2nans( n, A );
                check_iamax_3nans( n, A );
            }
        }

        SECTION( "(d) A[k] = OV*((k+2)/(k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
            for (idx_t k = 0; k < N; ++k) {
                if ( k % 2 == 0 ) set_complexOV( A[k], k );
                else              set_complexk( A[k], k );
            }
            for (const auto& n : n_vec) {
                INFO( "n = " << n );
                check_iamax_1nan( n, A );
                check_iamax_2nans( n, A );
                check_iamax_3nans( n, A );
            }
        }

        SECTION( "(e) A[k] = -k + i*k for k even, and A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k odd" ) {
            for (const auto& n : n_vec) {
                INFO( "n = " << n );
                for (idx_t k = 0; k < n; ++k) {
                    if ( k % 2 == 0 ) set_complexk( A[k], k );
                    else              set_complexOV( A[k], n-k );
                }
                check_iamax_1nan( n, A );
                check_iamax_2nans( n, A );
                check_iamax_3nans( n, A );
            }
        }

        SECTION( "(f) A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
            for (const auto& n : n_vec) {
                INFO( "n = " << n );
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

    SECTION( "(g) A[k] = NaN" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = NAN;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            CHECK( iamax( n, A, 1 ) == 0 );
        }
    }
}

/**
 * @brief Test case for iamax with arrays containing at least 1 Inf and no NaNs
 * 
 * Default entries:
 *  (a) A[k] = (-1)^k*k
 *  (b) A[k] = (-1)^k*Inf
 * and, for complex data type: 
 *  (c) A[k] = -k + i*k for k even, and A[k] = OV*((k+2)/(k+3))*(1+i) for k odd
 *  (d) A[k] = OV*((k+2)/(k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd
 *  (e) A[k] = -k + i*k for k even, and A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k odd
 *  (f) A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd
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

    SECTION( "(a) A[k] = (-1)^k*k" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? k : -k;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            check_iamax_1inf( n, A );
            check_iamax_2infs( n, A );
            check_iamax_3infs( n, A );
        }
    }

    SECTION( "(b) A[k] = (-1)^k*Inf" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? inf : -inf;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            CHECK( iamax( n, A, 1 ) == 0 );
        }
    }

    if (is_complex<TestType>::value) {

        SECTION( "(c) A[k] = -k + i*k for k even, and A[k] = OV*((k+2)/(k+3))*(1+i) for k odd" ) {
            for (idx_t k = 0; k < N; ++k) {
                if ( k % 2 == 0 ) set_complexk( A[k], k );
                else              set_complexOV( A[k], k );
            }
            for (const auto& n : n_vec) {
                INFO( "n = " << n );
                check_iamax_1inf( n, A );
                check_iamax_2infs( n, A );
                check_iamax_3infs( n, A );
            }
        }

        SECTION( "(d) A[k] = OV*((k+2)/(k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
            for (idx_t k = 0; k < N; ++k) {
                if ( k % 2 == 0 ) set_complexOV( A[k], k );
                else              set_complexk( A[k], k );
            }
            for (const auto& n : n_vec) {
                INFO( "n = " << n );
                check_iamax_1inf( n, A );
                check_iamax_2infs( n, A );
                check_iamax_3infs( n, A );
            }
        }

        SECTION( "(e) A[k] = -k + i*k for k even, and A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k odd" ) {
            for (const auto& n : n_vec) {
                INFO( "n = " << n );
                for (idx_t k = 0; k < n; ++k) {
                    if ( k % 2 == 0 ) set_complexk( A[k], k );
                    else              set_complexOV( A[k], n-k );
                }
                check_iamax_1inf( n, A );
                check_iamax_2infs( n, A );
                check_iamax_3infs( n, A );
            }
        }

        SECTION( "(f) A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
            for (const auto& n : n_vec) {
                INFO( "n = " << n );
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

/**
 * @brief Test case for iamax where A(k) are finite but abs(real(A(k)))+abs(imag(A(k))) can overflow.
 * 
 * 4 cases:
 *  (a) A[k] = -k + i*k for k even, and A[k] = OV*((k+2)/(k+3))*(1+i) for k odd. Correct answer = last odd k.
 *  (b) A[k] = OV*((k+2)/(k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd. Correct answer = last even k.
 *  (c) A[k] = -k + i*k for k even, and A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k odd. Correct answer = 0.
 *  (d) A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd. Correct answer = 1.
 */
TEMPLATE_TEST_CASE( "iamax works for complex data A when abs(real(A(k)))+abs(imag(A(k))) can overflow",
                    "[iamax][BLASlv1]", TEST_CPLX_TYPES ) {
    // Constants
    const idx_t N = 128;       // N > 0

    // Arrays
    const std::vector<idx_t> n_vec
        = { 1, 2, 3, 10, N }; // n_vec[i] > 0
    TestType A[N];

    SECTION( "(a) A[k] = -k + i*k for k even, and A[k] = OV*((k+2)/(k+3))*(1+i) for k odd" ) {
        for (idx_t k = 0; k < N; ++k) {
            if ( k % 2 == 0 ) set_complexk( A[k], k );
            else              set_complexOV( A[k], k );
        }
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            if ( n == 1 )          CHECK( iamax( n, A, 1 ) == 0 );
            else if ( n % 2 == 0 ) CHECK( iamax( n, A, 1 ) == n-1 );
            else                   CHECK( iamax( n, A, 1 ) == n-2 );
        }
    }

    SECTION( "(b) A[k] = OV*((k+2)/(k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
        for (idx_t k = 0; k < N; ++k) {
            if ( k % 2 == 0 ) set_complexOV( A[k], k );
            else              set_complexk( A[k], k );
        }
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            if ( n % 2 == 0 ) CHECK( iamax( n, A, 1 ) == n-2 );
            else              CHECK( iamax( n, A, 1 ) == n-1 );
        }
    }

    SECTION( "(c) A[k] = -k + i*k for k even, and A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k odd" ) {
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            for (idx_t k = 0; k < n; ++k) {
                if ( k % 2 == 0 ) set_complexk( A[k], k );
                else              set_complexOV( A[k], n-k );
            }
            if ( n == 1 ) CHECK( iamax( n, A, 1 ) == 0 );
            else          CHECK( iamax( n, A, 1 ) == 1 );
        }
    }

    SECTION( "(d) A[k] = OV*((n-k+2)/(n-k+3))*(1+i) for k even, and A[k] = -k + i*k for k odd" ) {
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            for (idx_t k = 0; k < n; ++k) {
                if ( k % 2 == 0 ) set_complexOV( A[k], n-k );
                else              set_complexk( A[k], k );
            }
            CHECK( iamax( n, A, 1 ) == 0 );
        }
    }
}
