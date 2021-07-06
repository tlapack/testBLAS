/// @brief Test case for iamax with real vectors
///
/// Total #cases < 5*(15 + 7*15 + 15) = 615
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

using namespace blas;

/**
 * @brief Test case for iamax with real arrays with at least 1 NaN
 * 
 * ISAMAX test cases: Default entries A(k) = (-1)^k*k
 * 
 * (1) At least 1 NaN, no Infs (<=15 cases)
 *       1 NaN, at location:
 *              1;  2; n/2;  n
 *       2 NaNs (if possible, i.e. n>1, ditto later)
 *              1,2;   1,n/2;  1,n;   2,n/2;  2,n;  n/2,n
 *       3 NaNs
 *              1,2,n/2;  1,2,n;  1,n/2,n;  2,n/2,n
 *       All NaNs
 * 
 * (2) At least 1 NaN and at least 1 Inf
 *       For each example above (<=7*15 cases):
 *              Insert Inf in first non-NaN location
 *              Insert -Inf in first non-NaN location
 *              Ditto for last non-NaN location
 *              Ditto for first and last non-NaN locations
 *              Insert (-1)^k*Inf in all non-NaN locations
 */
TEMPLATE_TEST_CASE( "iamax returns the first NaN for real arrays with at least 1 NaN",
                    "[iamax][BLASlv1][NaN]", TEST_REAL_TYPES ) {
    using real_t = real_type<TestType>;

    // Constants
    const blas::size_t N = 128;       // N > 1
    const TestType inf = std::numeric_limits<real_t>::infinity();

    // Arrays
    const std::vector<blas::size_t> n_vec
        = { 1, 2, 3, 10, N }; // n_vec[i] > 0
    TestType A[N];

    // Default entries
    #define _A(k) (( k % 2 == 0 ) ? k : -k)
    for (blas::size_t k = 0; k < N; ++k)
        A[k] = _A(k);

    SECTION( "1 NaN, at location: 0; 1; n/2; n-1" ) {
        for (const auto& n : n_vec) {
            
            // Indexes for test
            const blas::size_t k_arr[] = { 0, 1, n-1, n/2 };
            const blas::size_t k_values = 
                ( n <= 1 ) ? 1 : (
                ( n == 2 ) ? 2 : (
                ( n == 3 ) ? 3
                           : 4 ));
            
            // Tests
            for (blas::size_t i = 0; i < k_values; i += 1) {
                const auto& k = k_arr[i];
                const blas::size_t infIdx1 = (k > 0) ? 0 : 1;
                const blas::size_t infIdx2 = (k < n-1) ? n-1 : n-2;

                // NaN in A[k]
                A[k] = NAN;
                
                // No Infs
                CHECK( iamax( n, A, 1 ) == k );

                if( n <= 1 ) {
                    A[k] = _A(k);
                    continue;
                }
                
                // Inf in first non-NaN location
                A[ infIdx1 ] = inf;
                CHECK( iamax( n, A, 1 ) == k );
                
                // -Inf in first non-NaN location
                A[ infIdx1 ] = -inf;
                CHECK( iamax( n, A, 1 ) == k );
                
                // Reset value
                A[ infIdx1 ] = _A( infIdx1 );

                if( n <= 2 ) {
                    A[k] = _A(k);
                    continue;
                }
                
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

                // Insert (-1)^i*Inf in all non-NaN locations
                for (blas::size_t i = 0; i < n; ++i)
                    A[i] = ( i % 2 == 0 ) ? inf : -inf;
                A[k] = NAN;
                CHECK( iamax( n, A, 1 ) == k );

                // Reset values
                for (blas::size_t i = 0; i < n; ++i)
                    A[i] = _A(i);
            }
        }
    }

    SECTION( "2 NaNs if n > 1" ) {
        for (const auto& n : n_vec) {
            
            // Indexes for test
            const blas::size_t k_arr[]
                = { 0, 1,
                    0, n-1,
                    1, n-1,
                    0, n/2,
                    1, n/2,
                    n/2, n-1 };
            const blas::size_t k_values = 
                ( n <= 1 ) ? 0 : (
                ( n == 2 ) ? 2 : (
                ( n == 3 ) ? 6
                           : 12 ));
            
            // Tests
            for (blas::size_t i = 0; i < k_values; i += 2) {
                const auto& k1 = k_arr[i];
                const auto& k2 = k_arr[i+1];
                const blas::size_t infIdx1 =
                    (k1 > 0) ? 0 : (
                    (k2 > 1) ? 1
                             : 2 );
                const blas::size_t infIdx2 =
                    (k2 < n-1) ? n-1 : (
                    (k1 < n-2) ? n-2
                               : n-3 );

                // NaNs in A[k1] and A[k2]
                A[k1] = A[k2] = NAN;
                
                // No Infs
                CHECK( iamax( n, A, 1 ) == k1 );

                if( n <= 2 ) {
                    A[k1] = _A(k1);
                    A[k2] = _A(k2);
                    continue;
                }
                
                // Inf in first non-NaN location
                A[ infIdx1 ] = inf;
                CHECK( iamax( n, A, 1 ) == k1 );
                
                // -Inf in first non-NaN location
                A[ infIdx1 ] = -inf;
                CHECK( iamax( n, A, 1 ) == k1 );
                
                // Reset value
                A[ infIdx1 ] = _A( infIdx1 );

                if( n <= 3 ) {
                    A[k1] = _A(k1);
                    A[k2] = _A(k2);
                    continue;
                }
                
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

                // Insert (-1)^i*Inf in all non-NaN locations
                for (blas::size_t i = 0; i < n; ++i)
                    A[i] = ( i % 2 == 0 ) ? inf : -inf;
                A[k1] = A[k2] = NAN;
                CHECK( iamax( n, A, 1 ) == k1 );

                // Reset values
                for (blas::size_t i = 0; i < n; ++i)
                    A[i] = _A(i);
            }
        }
    }

    SECTION( "3 NaNs if n > 2" ) {
        for (const auto& n : n_vec) {
            
            // Indexes for test
            const blas::size_t k_arr[]
                = { 0, 1, n-1,
                    0, 1, n/2,
                    0, n/2, n-1,
                    1, n/2, n-1 };
            const blas::size_t k_values = 
                ( n <= 2 ) ? 0 : (
                ( n == 3 ) ? 3
                           : 12 );
            
            // Tests
            for (blas::size_t i = 0; i < k_values; i += 3) {
                const auto& k1 = k_arr[i];
                const auto& k2 = k_arr[i+1];
                const auto& k3 = k_arr[i+2];
                const blas::size_t infIdx1 =
                    (k1 > 0) ? 0 : (
                    (k2 > 1) ? 1 : (
                    (k3 > 2) ? 2
                             : 3 ));
                const blas::size_t infIdx2 =
                    (k3 < n-1) ? n-1 : (
                    (k2 < n-2) ? n-2 : (
                    (k1 < n-3) ? n-3
                               : n-4 ));

                // NaNs in A[k1], A[k2] and A[k3]
                A[k1] = A[k2] = A[k3] = NAN;
                
                // No Infs
                CHECK( iamax( n, A, 1 ) == k1 );

                if( n <= 3 ) {
                    A[k1] = _A(k1);
                    A[k2] = _A(k2);
                    A[k3] = _A(k3);
                    continue;
                }
                
                // Inf in first non-NaN location
                A[ infIdx1 ] = inf;
                CHECK( iamax( n, A, 1 ) == k1 );
                
                // -Inf in first non-NaN location
                A[ infIdx1 ] = -inf;
                CHECK( iamax( n, A, 1 ) == k1 );
                
                // Reset value
                A[ infIdx1 ] = _A( infIdx1 );

                if( n <= 4 ) {
                    A[k1] = _A(k1);
                    A[k2] = _A(k2);
                    A[k3] = _A(k3);
                    continue;
                }
                
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

                // Insert (-1)^i*Inf in all non-NaN locations
                for (blas::size_t i = 0; i < n; ++i)
                    A[i] = ( i % 2 == 0 ) ? inf : -inf;
                A[k1] = A[k2] = A[k3] = NAN;
                CHECK( iamax( n, A, 1 ) == k1 );

                // Reset values
                for (blas::size_t i = 0; i < n; ++i)
                    A[i] = _A(i);
            }
        }
    }

    SECTION( "All NaNs" ) {
        for (const auto& n : n_vec) {
            for (blas::size_t k = 0; k < n; ++k)
                A[k] = NAN;
            
            CHECK( iamax( n, A, 1 ) == 0 );
            
            for (blas::size_t k = 0; k < n; ++k)
                A[k] = _A(k);
        }
    }

    #undef _A
}



/**
 * @brief Test case for iamax with real arrays with at least 1 Inf and no NaNs
 * 
 * ISAMAX test cases: Default entries A(k) = (-1)^k*k
 * 
 * At least 1 Inf (<=15 cases)
 *       1 Inf, at location:
 *              1;  2; n/2;  n
 *       2 Infs (if possible, i.e. n>1, ditto later)
 *              1,2;   1,n/2;  1,n;   2,n/2;  2,n;  n/2,n
 *       3 Infs
 *              1,2,n/2;  1,2,n;  1,n/2,n;  2,n/2,n
 *       All Infs
 */
TEMPLATE_TEST_CASE( "iamax returns the first Inf for real arrays with at least 1 Inf and no NaNs",
                    "[iamax][BLASlv1][Inf]", TEST_REAL_TYPES ) {
    using real_t = real_type<TestType>;

    // Constants
    const blas::size_t N = 128;       // N > 1
    const TestType inf = std::numeric_limits<real_t>::infinity();

    // Arrays
    const std::vector<blas::size_t> n_vec
        = { 1, 2, 3, 10, N }; // n_vec[i] > 0
    TestType A[N];

    // Default entries
    #define _A(k) (( k % 2 == 0 ) ? k : -k)
    for (blas::size_t k = 0; k < N; ++k)
        A[k] = _A(k);

    SECTION( "1 Inf, at location: 0; 1; n/2; n-1" ) {
        for (const auto& n : n_vec) {
            
            // Indexes for test
            const blas::size_t k_arr[] = { 0, 1, n-1, n/2 };
            const blas::size_t k_values = 
                ( n <= 1 ) ? 1 : (
                ( n == 2 ) ? 2 : (
                ( n == 3 ) ? 3
                           : 4 ));
            
            // Tests
            for (blas::size_t i = 0; i < k_values; i += 1) {
                const auto& k = k_arr[i];

                // (-1)^k*Inf in A[k]
                A[k] = ( k % 2 == 0 ) ? inf : -inf;
                
                CHECK( iamax( n, A, 1 ) == k );

                // Default values
                A[k] = _A(k);
            }
        }
    }

    SECTION( "2 Infs if n > 1" ) {
        for (const auto& n : n_vec) {
            
            // Indexes for test
            const blas::size_t k_arr[]
                = { 0, 1,
                    0, n-1,
                    1, n-1,
                    0, n/2,
                    1, n/2,
                    n/2, n-1 };
            const blas::size_t k_values = 
                ( n <= 1 ) ? 0 : (
                ( n == 2 ) ? 2 : (
                ( n == 3 ) ? 6
                           : 12 ));
            
            // Tests
            for (blas::size_t i = 0; i < k_values; i += 2) {
                const auto& k1 = k_arr[i];
                const auto& k2 = k_arr[i+1];

                // (-1)^k*Inf in A[k]
                A[k1] = ( k1 % 2 == 0 ) ? inf : -inf;
                A[k2] = ( k2 % 2 == 0 ) ? inf : -inf;
                
                CHECK( iamax( n, A, 1 ) == k1 );

                // Default values
                A[k1] = _A(k1);
                A[k2] = _A(k2);
            }
        }
    }

    SECTION( "3 Infs if n > 2" ) {
        for (const auto& n : n_vec) {
            
            // Indexes for test
            const blas::size_t k_arr[]
                = { 0, 1, n-1,
                    0, 1, n/2,
                    0, n/2, n-1,
                    1, n/2, n-1 };
            const blas::size_t k_values = 
                ( n <= 2 ) ? 0 : (
                ( n == 3 ) ? 3
                           : 12 );
            
            // Tests
            for (blas::size_t i = 0; i < k_values; i += 3) {
                const auto& k1 = k_arr[i];
                const auto& k2 = k_arr[i+1];
                const auto& k3 = k_arr[i+2];

                // (-1)^k*Inf in A[k]
                A[k1] = ( k1 % 2 == 0 ) ? inf : -inf;
                A[k2] = ( k2 % 2 == 0 ) ? inf : -inf;
                A[k3] = ( k3 % 2 == 0 ) ? inf : -inf;
                
                CHECK( iamax( n, A, 1 ) == k1 );

                // Default values
                A[k1] = _A(k1);
                A[k2] = _A(k2);
                A[k3] = _A(k3);
            }
        }
    }

    SECTION( "All Infs" ) {
        for (const auto& n : n_vec) {
            for (blas::size_t k = 0; k < n; ++k)
                A[k] = ( k % 2 == 0 ) ? inf : -inf;
            
            CHECK( iamax( n, A, 1 ) == 0 );
            
            for (blas::size_t k = 0; k < n; ++k)
                A[k] = _A(k);
        }
    }

    #undef _A
}
