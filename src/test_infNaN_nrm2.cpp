/// @file test_infNaN_nrm2.cpp
/// @brief Test cases for nrm2 with NaNs, Infs and values whose under- or overflows when squared.
//
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <legacy_api/blas.hpp>
#include "defines.hpp"
#include "utils.hpp"
#ifdef USE_MPFR
    #include <plugins/tlapack_mpreal.hpp>
#endif

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <limits>
#include <vector>
#include <complex>

using namespace tlapack;

// -----------------------------------------------------------------------------
// Test cases for nrm2 with Infs and NaNs at specific positions

template< typename TestType >
void check_nrm2_nans_and_infs(
    const idx_t n,
    TestType A[],
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
                CHECK( isnan( nrm2( n, A, 1 ) ) );
                                            
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

                CHECK( isnan( nrm2( n, A, 1 ) ) );
            }

            // Reset value
            A[ infLocs[0] ] = A0;
            A[ infLocs[1] ] = A1;
        }
    }
}

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
            CHECK( isnan( nrm2( n, A, 1 ) ) );

            if( checkWithInf )
                check_nrm2_nans_and_infs( n, A, 1, infLocs );
        }

        // Reset value
        A[k] = Ak;
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
            CHECK( isnan( nrm2( n, A, 1 ) ) );

            if( checkWithInf )
                check_nrm2_nans_and_infs( n, A, 2, infLocs );
        }

        // Reset values
        A[k1] = Ak1;
        A[k2] = Ak2;
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
            CHECK( isnan( nrm2( n, A, 1 ) ) );

            if( checkWithInf )
                check_nrm2_nans_and_infs( n, A, 3, infLocs );
        }

        // Reset values
        A[k1] = Ak1;
        A[k2] = Ak2;
        A[k3] = Ak3;
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

            auto nrm2ofA = nrm2( n, A, 1 );
            INFO( "nrm2ofA = " << nrm2ofA );
            CHECK( isinf( nrm2ofA ) );
        }

        // Reset value
        A[k] = Ak;
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

            auto nrm2ofA = nrm2( n, A, 1 );
            INFO( "nrm2ofA = " << nrm2ofA );
            CHECK( isinf( nrm2ofA ) );
        }

        // Reset values
        A[k1] = Ak1;
        A[k2] = Ak2;
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

            auto nrm2ofA = nrm2( n, A, 1 );
            INFO( "nrm2ofA = " << nrm2ofA );
            CHECK( isinf( nrm2ofA ) );
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
 * @brief Test case for nrm2 with arrays containing at least 1 NaN
 * 
 * Default entries:
 *  (1) A[k] = (-1)^k*b/2, where b is the Blue's min constant. (b/2)^2 underflows but the norm is (b/2)*sqrt(n)
 *  (2) A[k] = (-1)^k*x, where x is the underflow threshold. x^2 underflows but the norm is x*sqrt(n)
 *  (3) A[k] = (-1)^k*x, where x is the smallest subnormal number. x^2 underflows but the norm is x*sqrt(n)
 *      Mind that not all platforms might implement subnormal numbers.
 *  (4) A[k] = (-1)^k*2*B/n, where B is the Blue's max constant, n > 1. (2*B/n)^2 and the norm are finite but sum_k A[k]^2 overflows
 *  (5) A[k] = (-1)^k*2*B, where B is the Blue's max constant. (2*B)^2 overflows but the norm is (2*B)*sqrt(n)
 *  (6) A[k] = b for k even, and A[k] = -7*b for k odd, where b is the Blue's min constant. The norm is 5*b*sqrt(n)
 *  (7) A[k] = B for k even, and A[k] = -7*B for k odd, where B is the Blue's max constant. The norm is 5*B*sqrt(n)
 *  (8) A[k] = (-1)^k*2*OV/sqrt(n), n > 1. 2*OV/sqrt(n) is finite but the norm overflows
 *  (9) A[k] = (-1)^k*k
 *  (10) A[k]= (-1)^k*Inf
 */
TEMPLATE_TEST_CASE( "nrm2 returns NaN for arrays with at least 1 NaN",
                    "[nrm2][BLASlv1][NaN][Inf]", TEST_TYPES ) {
    using real_t = real_type<TestType>;

    // Constants
    const idx_t N = 128;       // N > 0
    const real_t inf = std::numeric_limits<real_t>::infinity();
    const real_t b = blue_min<real_t>();
    const real_t B = blue_max<real_t>();
    const real_t tinyNum    = std::numeric_limits<real_t>::min();
    const real_t tiniestNum = std::numeric_limits<real_t>::denorm_min();
    const real_t smallNum   = b / 2;
    const real_t bigNum     = B * 2;
    const real_t hugeNum    = pow(
        std::numeric_limits<real_t>::radix,
        real_t(std::numeric_limits<real_t>::max_exponent-1)
    );

    // Arrays
    const std::vector<idx_t> n_vec
        = { 1, 2, 3, 10, N }; // n_vec[i] > 0
    TestType A[N];

    SECTION( "A[k] = (-1)^k*b/2, where b is the Blue's min constant. (b/2)^2 underflows but the norm is (b/2)*sqrt(n)" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? smallNum : -smallNum;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            check_nrm2_1nan( n, A );
            check_nrm2_2nans( n, A );
            check_nrm2_3nans( n, A );
        }
    }

    SECTION( "A[k] = (-1)^k*x, where x is the underflow threshold. x^2 underflows but the norm is positive" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? tinyNum : -tinyNum;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            check_nrm2_1nan( n, A );
            check_nrm2_2nans( n, A );
            check_nrm2_3nans( n, A );
        }
    }

    SECTION( "A[k] = (-1)^k*x, where x is the smallest subnormal number. x^2 underflows but the norm is positive" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? tiniestNum : -tiniestNum;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            check_nrm2_1nan( n, A );
            check_nrm2_2nans( n, A );
            check_nrm2_3nans( n, A );
        }
    }

    SECTION( "A[k] = (-1)^k*2*B/n, where B is the Blue's max constant, n > 1. (2*B/n)^2 and the norm are finite but sum_k A[k]^2 overflows" ) {
        for (const auto& n : n_vec) {
            if( n <= 1 ) continue;
            INFO( "n = " << n );
            const real_t Ak = bigNum / n;
            for (idx_t k = 0; k < n; ++k)
                A[k] = ( k % 2 == 0 ) ? Ak : -Ak;
            check_nrm2_1nan( n, A );
            check_nrm2_2nans( n, A );
            check_nrm2_3nans( n, A );
        }
    }

    SECTION( "A[k] = (-1)^k*2*B, where B is the Blue's max constant. (2*B)^2 overflows but the norm is (2*B)*sqrt(n)" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? bigNum : -bigNum;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            check_nrm2_1nan( n, A );
            check_nrm2_2nans( n, A );
            check_nrm2_3nans( n, A );
        }
    }

    SECTION( "A[k] = b for k even, and A[k] = -7*b for k odd, where b is the Blue's min constant. The norm is 5*b*sqrt(n)" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? b : -7*b;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            check_nrm2_1nan( n, A );
            check_nrm2_2nans( n, A );
            check_nrm2_3nans( n, A );
        }
    }

    SECTION( "A[k] = B for k even, and A[k] = -7*B for k odd, where B is the Blue's max constant. The norm is 5*B*sqrt(n)" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? B : -7*B;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            check_nrm2_1nan( n, A );
            check_nrm2_2nans( n, A );
            check_nrm2_3nans( n, A );
        }
    }

    SECTION( "A[k] = (-1)^k*2*OV/sqrt(n), n > 1. 2*OV/sqrt(n) is finite but the norm overflows" ) {
        for (const auto& n : n_vec) {
            if( n <= 1 ) continue;
            INFO( "n = " << n );
            for (idx_t k = 0; k < n; ++k)
                A[k] = ( k % 2 == 0 ) ? 2*hugeNum/sqrt(n) : -2*hugeNum/sqrt(n);
            check_nrm2_1nan( n, A );
            check_nrm2_2nans( n, A );
            check_nrm2_3nans( n, A );
        }
    }

    SECTION( "A[k] = (-1)^k*k" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? k : -k;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            check_nrm2_1nan( n, A );
            check_nrm2_2nans( n, A );
            check_nrm2_3nans( n, A );
        }
    }

    SECTION( "A[k] = (-1)^k*Inf" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? inf : -inf;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            check_nrm2_1nan( n, A, false );
            check_nrm2_2nans( n, A, false );
            check_nrm2_3nans( n, A, false );
        }
    }

    SECTION( "All NaNs" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = NAN;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            CHECK( isnan( nrm2( n, A, 1 ) ) );
        }
    }
}

/**
 * @brief Test case for nrm2 with arrays containing at least 1 Inf and no NaNs
 * 
 * Default entries:
 *  (1) A[k] = (-1)^k*b/2, where b is the Blue's min constant. (b/2)^2 underflows but the norm is (b/2)*sqrt(n)
 *  (2) A[k] = (-1)^k*x, where x is the underflow threshold. x^2 underflows but the norm is x*sqrt(n)
 *  (3) A[k] = (-1)^k*x, where x is the smallest subnormal number. x^2 underflows but the norm is x*sqrt(n)
 *      Mind that not all platforms might implement subnormal numbers.
 *  (4) A[k] = (-1)^k*2*B/n, where B is the Blue's max constant, n > 1. (2*B/n)^2 and the norm are finite but sum_k A[k]^2 overflows
 *  (5) A[k] = (-1)^k*2*B, where B is the Blue's max constant. (2*B)^2 overflows but the norm is (2*B)*sqrt(n)
 *  (6) A[k] = b for k even, and A[k] = -7*b for k odd, where b is the Blue's min constant. The norm is 5*b*sqrt(n)
 *  (7) A[k] = B for k even, and A[k] = -7*B for k odd, where B is the Blue's max constant. The norm is 5*B*sqrt(n)
 *  (8) A[k] = (-1)^k*2*OV/sqrt(n), n > 1. 2*OV/sqrt(n) is finite but the norm overflows
 *  (9) A[k] = (-1)^k*k
 */
TEMPLATE_TEST_CASE( "nrm2 returns Inf for arrays with at least 1 Inf and no NaNs",
                    "[nrm2][BLASlv1][Inf]", TEST_TYPES ) {
    using real_t = real_type<TestType>;

    // Constants
    const idx_t N = 128;       // N > 0
    const real_t inf = std::numeric_limits<real_t>::infinity();
    const real_t b = blue_min<real_t>();
    const real_t B = blue_max<real_t>();
    const real_t tinyNum    = std::numeric_limits<real_t>::min();
    const real_t tiniestNum = std::numeric_limits<real_t>::denorm_min();
    const real_t smallNum   = b / 2;
    const real_t bigNum     = B * 2;
    const real_t hugeNum    = pow(
        std::numeric_limits<real_t>::radix,
        real_t(std::numeric_limits<real_t>::max_exponent-1)
    );

    // Arrays
    const std::vector<idx_t> n_vec
        = { 1, 2, 3, 10, N }; // n_vec[i] > 0
    TestType A[N];

    SECTION( "A[k] = (-1)^k*b/2, where b is the Blue's min constant. (b/2)^2 underflows but the norm is (b/2)*sqrt(n)" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? smallNum : -smallNum;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            check_nrm2_1inf( n, A );
            check_nrm2_2infs( n, A );
            check_nrm2_3infs( n, A );
        }
    }

    SECTION( "A[k] = (-1)^k*x, where x is the underflow threshold. x^2 underflows but the norm is positive" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? tinyNum : -tinyNum;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            check_nrm2_1inf( n, A );
            check_nrm2_2infs( n, A );
            check_nrm2_3infs( n, A );
        }
    }

    SECTION( "A[k] = (-1)^k*x, where x is the smallest subnormal number. x^2 underflows but the norm is positive" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? tiniestNum : -tiniestNum;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            check_nrm2_1inf( n, A );
            check_nrm2_2infs( n, A );
            check_nrm2_3infs( n, A );
        }
    }

    SECTION( "A[k] = (-1)^k*2*B/n, where B is the Blue's max constant, n > 1. (2*B/n)^2 and the norm are finite but sum_k A[k]^2 overflows" ) {
        for (const auto& n : n_vec) {
            if( n <= 1 ) continue;
            INFO( "n = " << n );
            const real_t Ak = bigNum / n;
            for (idx_t k = 0; k < n; ++k)
                A[k] = ( k % 2 == 0 ) ? Ak : -Ak;
            check_nrm2_1inf( n, A );
            check_nrm2_2infs( n, A );
            check_nrm2_3infs( n, A );
        }
    }

    SECTION( "A[k] = (-1)^k*2*B, where B is the Blue's max constant. (2*B)^2 overflows but the norm is (2*B)*sqrt(n)" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? bigNum : -bigNum;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            check_nrm2_1inf( n, A );
            check_nrm2_2infs( n, A );
            check_nrm2_3infs( n, A );
        }
    }

    SECTION( "A[k] = b for k even, and A[k] = -7*b for k odd, where b is the Blue's min constant. The norm is 5*b*sqrt(n)" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? b : -7*b;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            check_nrm2_1inf( n, A );
            check_nrm2_2infs( n, A );
            check_nrm2_3infs( n, A );
        }
    }

    SECTION( "A[k] = B for k even, and A[k] = -7*B for k odd, where B is the Blue's max constant. The norm is 5*B*sqrt(n)" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? B : -7*B;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            check_nrm2_1inf( n, A );
            check_nrm2_2infs( n, A );
            check_nrm2_3infs( n, A );
        }
    }

    SECTION( "A[k] = (-1)^k*2*OV/sqrt(n), n > 1. 2*OV/sqrt(n) is finite but the norm overflows" ) {
        for (const auto& n : n_vec) {
            if( n <= 1 ) continue;
            INFO( "n = " << n );
            for (idx_t k = 0; k < n; ++k)
                A[k] = ( k % 2 == 0 ) ? 2*hugeNum/sqrt(n) : -2*hugeNum/sqrt(n);
            check_nrm2_1inf( n, A );
            check_nrm2_2infs( n, A );
            check_nrm2_3infs( n, A );
        }
    }

    SECTION( "A[k] = (-1)^k*k" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? k : -k;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            check_nrm2_1inf( n, A );
            check_nrm2_2infs( n, A );
            check_nrm2_3infs( n, A );
        }
    }

    SECTION( "All Infs" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? inf : -inf;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            CHECK( isinf( nrm2( n, A, 1 ) ) );
        }
    }
}

/**
 * @brief Test case for nrm2 with finite input which expects an exact output
 * 
 * Default entries:
 *  (1) A[k] = (-1)^k*b/2, where b is the Blue's min constant. (b/2)^2 underflows but the norm is (b/2)*sqrt(n)
 *  (2) A[k] = (-1)^k*x, where x is the underflow threshold. x^2 underflows but the norm is x*sqrt(n)
 *  (3) A[k] = (-1)^k*x, where x is the smallest subnormal number. x^2 underflows but the norm is x*sqrt(n)
 *      Mind that not all platforms might implement subnormal numbers.
 *  (4) A[k] = (-1)^k*2*B/n, where B is the Blue's max constant, n > 1. (2*B/n)^2 and the norm are finite but sum_k A[k]^2 overflows
 *  (5) A[k] = (-1)^k*2*B, where B is the Blue's max constant. (2*B)^2 overflows but the norm is (2*B)*sqrt(n)
 *  (6) A[k] = b for k even, and A[k] = -7*b for k odd, where b is the Blue's min constant. The norm is 5*b*sqrt(n)
 *  (7) A[k] = B for k even, and A[k] = -7*B for k odd, where B is the Blue's max constant. The norm is 5*B*sqrt(n)
 *  (8) A[k] = (-1)^k*2*OV/sqrt(n), n > 1. 2*OV/sqrt(n) is finite but the norm overflows
 */
TEMPLATE_TEST_CASE( "nrm2 with finite input which expects an exact output",
                    "[nrm2][BLASlv1]", TEST_TYPES ) {
    using real_t = real_type<TestType>;

    // Constants
    const idx_t N = 256;       // N > 0
    const real_t b = blue_min<real_t>();
    const real_t B = blue_max<real_t>();
    const real_t tinyNum    = std::numeric_limits<real_t>::min();
    const real_t tiniestNum = std::numeric_limits<real_t>::denorm_min();
    const real_t smallNum   = b / 2;
    const real_t bigNum     = B * 2;
    const real_t hugeNum    = pow(
        std::numeric_limits<real_t>::radix,
        real_t(std::numeric_limits<real_t>::max_exponent-1)
    );

    // Arrays
    const std::vector<idx_t> n_vec
        = { 1, 4, 16, 64, N }; // n_vec[i] > 0
    TestType A[N];

    SECTION( "A[k] = (-1)^k*b/2, where b is the Blue's min constant. (b/2)^2 underflows but the norm is positive" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? smallNum : -smallNum;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );

            auto nrm2ofA = nrm2( n, A, 1 );
            INFO( "Rel. error = " << std::scientific << tlapack::abs( nrm2ofA / (smallNum*sqrt(n)) - 1 ) );

            CHECK( nrm2ofA / (smallNum*sqrt(n)) == 1 );
        }
    }

    SECTION( "A[k] = (-1)^k*x, where x is the underflow threshold. x^2 underflows but the norm is positive" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? tinyNum : -tinyNum;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );

            auto nrm2ofA = nrm2( n, A, 1 );
            INFO( "Rel. error = " << std::scientific << tlapack::abs( nrm2ofA / (tinyNum*sqrt(n)) - 1 ) );

            CHECK( nrm2ofA / (tinyNum*sqrt(n)) == 1 );
        }
    }

    SECTION( "A[k] = (-1)^k*x, where x is the smallest subnormal number. x^2 underflows but the norm is positive" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? tiniestNum : -tiniestNum;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );

            auto nrm2ofA = nrm2( n, A, 1 );
            INFO( "Rel. error = " << std::scientific << tlapack::abs( nrm2ofA / (tiniestNum*sqrt(n)) - 1 ) );

            CHECK( nrm2ofA / (tiniestNum*sqrt(n)) == 1 );
        }
    }

    SECTION( "A[k] = (-1)^k*2*B/n, where B is the Blue's max constant, n > 1. (2*B/n)^2 and the norm are finite but sum_k A[k]^2 overflows" ) {
        for (const auto& n : n_vec) {
            if( n <= 1 ) continue;
            INFO( "n = " << n );

            const real_t Ak = bigNum / n;
            for (idx_t k = 0; k < n; ++k)
                A[k] = ( k % 2 == 0 ) ? Ak : -Ak;

            auto nrm2ofA = nrm2( n, A, 1 );
            INFO( "Rel. error = " << std::scientific << tlapack::abs( nrm2ofA / (bigNum/sqrt(n)) - 1 ) );

            CHECK( nrm2( n, A, 1 ) / (bigNum/sqrt(n)) == 1 );
        }
    }

    SECTION( "A[k] = (-1)^k*2*B, where B is the Blue's max constant. (2*B)^2 overflows but the norm is (2*B)*sqrt(n)" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? bigNum : -bigNum;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );

            auto nrm2ofA = nrm2( n, A, 1 );
            INFO( "Rel. error = " << std::scientific << tlapack::abs( nrm2ofA / (bigNum*sqrt(n)) - 1 ) );

            CHECK( nrm2( n, A, 1 ) / (bigNum*sqrt(n)) == 1 );
        }
    }

    SECTION( "A[k] = b for k even, and A[k] = -7*b for k odd, where b is the Blue's min constant. The norm is 5*b*sqrt(n)" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? b : -7*b;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            
            auto nrm2ofA = nrm2( n, A, 1 );
            
            if ( n == 1 ) {
                INFO( "Rel. error = " << std::scientific << tlapack::abs( (nrm2ofA-b)/b ) );
                CHECK( nrm2ofA / b == 1 );
            } else {
                INFO( "Rel. error = " << std::scientific << tlapack::abs( (nrm2ofA/(b*sqrt(n))-5)/5 ) );
                CHECK( nrm2ofA / (b*sqrt(n)) == 5 );
            }
        }
    }

    SECTION( "A[k] = B for k even, and A[k] = -7*B for k odd, where B is the Blue's max constant. The norm is 5*B*sqrt(n)" ) {
        for (idx_t k = 0; k < N; ++k)
            A[k] = ( k % 2 == 0 ) ? B : -7*B;
        for (const auto& n : n_vec) {
            INFO( "n = " << n );
            
            auto nrm2ofA = nrm2( n, A, 1 );
            
            if ( n == 1 ) {
                INFO( "Rel. error = " << std::scientific << tlapack::abs( (nrm2ofA-B)/B ) );
                CHECK( nrm2ofA / B == 1 );
            } else {
                INFO( "Rel. error = " << std::scientific << tlapack::abs( (nrm2ofA/(B*sqrt(n))-5)/5 ) );
                CHECK( nrm2ofA / (B*sqrt(n)) == 5 );
            }
        }
    }

    SECTION( "A[k] = (-1)^k*2*OV/sqrt(n), n > 1. 2*OV/sqrt(n) is finite but the norm overflows" ) {
        for (const auto& n : n_vec) {
            if( n <= 1 ) continue;
            INFO( "n = " << n );

            for (idx_t k = 0; k < n; ++k)
                A[k] = ( k % 2 == 0 ) ? 2*hugeNum/sqrt(n) : -2*hugeNum/sqrt(n);

            auto nrm2ofA = nrm2( n, A, 1 );
            INFO( "nrm2ofA = " << std::scientific << nrm2ofA );
            
            CHECK( isinf( nrm2ofA ) );
        }
    }
}
