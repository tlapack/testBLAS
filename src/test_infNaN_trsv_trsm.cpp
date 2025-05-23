/// @file test_infNaN_trsv_trsm.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @brief Test cases for trsv and trsm with NaNs and Infs.
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "utils.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <limits>
#include <vector>
#include <complex>
#include <sstream>

using namespace testBLAS;
    
// Constants
const idx_t N        = 128; // Number of rows
const idx_t P        = 3;  // Number of columns
const idx_t max_int  = 10;  // Values in the system range from 1 to (max_int-1)
const idx_t max_exp  = (const idx_t) ceil( log2(max_int) );
const float sparsity = .75;  // Approximate percentage of zeros in the data

// -----------------------------------------------------------------------------
// Main Test Cases

/**
 * @brief Test if trsv propagates Infs and NaNs from the triangular matrix to the solution
 * 
 *  1) generate random T (upper or lower triangular) with small integer entries,
        and with 1, and perhaps 2, 4, ... along the diagonal (to avoid roundoff)
    2) generate a random sparse x also with small integer entries
    3) let b = T*x (no roundoff)
    4) modify T by inserting (some) Infs and NaNs in the columns of T corresponding
        to zero entries in x
    5) solve Tx = b for x
    6) test whether NaNs appear in x in the same rows that NaNs and Infs appear in T.
 */
TEMPLATE_TEST_CASE( "trsv propagates Infs and NaNs from the triangular matrix to the solution",
                    "[trsv][BLASlv2][NaN][Inf]", TEST_TYPES ) {
    using real_t = real_type<TestType>;

    // Constants
    const TestType zero( 0 );
    const real_t one( 1 );

    // Arrays
    const std::vector<idx_t> n_vec = { 2, 3, 10, N };
    TestType T[N*N];
    TestType x[N];
    TestType b[N];
    bool T_nanRow[N];
    std::vector<TestType> nan_vec;
    std::vector<TestType> inf_vec;

    // Views
    #define T(i,j) T[ j*N + i ]

    // Initialize vectors with the different Infs and NaNs
    testBLAS::set_nan_vector( nan_vec );
    testBLAS::set_inf_vector( inf_vec );

    // Init random seed
    srand( 3 );

    // Initialize the lower part of T with junk
    for (idx_t j = 0; j < N; ++j)
        for (idx_t i = j+1; i < N; ++i)
            T(i,j) = static_cast<float>( 0xDEADBEEF );
    
    for (const auto& n : n_vec) {
        INFO( "n = " << n );

        // Initialize the upper part of T with random integers from 0 to 9
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < j; ++i)
                T(i,j) = rand() % max_int;

        // Initialize the diagonal of T with random integers powers of 2
        for (idx_t i = 0; i < n; ++i)
            T(i,i) = pow( 2, rand() % max_exp );

        // Initialize x and b with 0s and set T_nanRow to false
        for (idx_t i = 0; i < n; ++i) {
            x[i] = zero;
            b[i] = zero;
            T_nanRow[i] = false;
        }

        // Put some ints in x
        for (idx_t i = 0; i < (idx_t) floor(n*(1-sparsity)); ++i)
            x[ rand() % n ] = max( one, real_t(rand() % max_int) );

        // let b = T*x (no roundoff)
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i <= j; ++i)
                b[i] += T(i,j) * x[j];

        // Put Infs and NaNs in the columns of T respective to the 0s in x
        // Do not put Infs in the diagonal 
        for (idx_t j = 0; j < n; ++j) {
            if( x[j] == zero ) {
                idx_t i = rand() % (j+1); // 0 <= i <= j
                if( i == j )
                    T( i, j ) = nan_vec[ rand() % nan_vec.size() ];
                else {
                    T( i, j ) = ( rand() % 2 == 0 )
                            ? inf_vec[ rand() % inf_vec.size() ]
                            : nan_vec[ rand() % nan_vec.size() ];
                }
                T_nanRow[i] = true;
            }
        }

        // Copy b to x
        for (idx_t i = 0; i < n; ++i)
            x[i] = b[i];

        // Solve for x
        trsv(   Layout::ColMajor, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                n, T, N, x, 1 );

        // Test if NaNs have propagated
        for (idx_t i = 0; i < n; ++i) {
            if( T_nanRow[i] ) {

                std::stringstream lineT;
                lineT << T(i,i);
                for (idx_t col = i+1; col < n; ++col)
                    lineT << ", " << T(i,col);
                UNSCOPED_INFO( "T[" << i << "," << i << ":] = " << lineT.str() );

                std::stringstream lineX;
                lineX << x[i];
                for (idx_t row = i+1; row < n; ++row)
                    lineX << ", " << x[row];
                UNSCOPED_INFO( "x[" << i << ":] = " << lineX.str() );

                UNSCOPED_INFO( "b[" << i << "] = " << b[i] );

                CHECK( isnan(x[i]) );
            }
        }
    }
    
    #undef T
}

/**
 * @brief Test if trsm propagates Infs and NaNs from the triangular matrix to the solution
 * 
 *  1) generate random T (upper or lower triangular) with small integer entries,
        and with 1, and perhaps 2, 4, ... along the diagonal (to avoid roundoff)
    2) generate a random sparse X also with small integer entries
    3) let B = T*X (no roundoff)
    4) modify T by inserting (some) Infs and NaNs in the columns of T corresponding
        to zero entries in X
    5) solve TX = B for X
    6) test whether NaNs appear in X in the same rows that NaNs and Infs appear in T.
 */
TEMPLATE_TEST_CASE( "trsm propagates Infs and NaNs from the triangular matrix to the solution",
                    "[trsm][BLASlv3][NaN][Inf]", TEST_TYPES ) {
    using real_t = real_type<TestType>;

    // Constants
    const TestType zero( 0 );
    const real_t one( 1 );

    // Arrays
    const std::vector<idx_t> n_vec = { 2, 3, 10, N };
    const std::vector<idx_t> p_vec = { 1, P };
    TestType T[N*N];
    TestType x[N*P];
    TestType b[N*P];
    bool T_nanRow[N];
    std::vector<TestType> nan_vec;
    std::vector<TestType> inf_vec;

    // Views
    #define T(i,j)          T[ j*N + i ]
    #define X(i,j)          x[ j*N + i ]
    #define B(i,j)          b[ j*N + i ]

    // Initialize vectors with the different Infs and NaNs
    testBLAS::set_nan_vector( nan_vec );
    testBLAS::set_inf_vector( inf_vec );

    // Init random seed
    srand( 3 );

    // Initialize the lower part of T with junk
    for (idx_t j = 0; j < N; ++j)
        for (idx_t i = j+1; i < N; ++i)
            T(i,j) = static_cast<float>( 0xDEADBEEF );
    
    for (const auto& p : p_vec) {
    for (const auto& n : n_vec) {
        INFO( "n: " << n );

        // Initialize the upper part of T with random integers from 0 to 9
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < j; ++i)
                T(i,j) = rand() % max_int;

        // Initialize the diagonal of T with random integers powers of 2
        for (idx_t i = 0; i < n; ++i)
            T(i,i) = pow( 2, rand() % max_exp );

        // Initialize X and B with 0s and set T_nanRow to false
        for (idx_t j = 0; j < p; ++j) {
            for (idx_t i = 0; i < n; ++i) {
                X(i,j) = zero;
                B(i,j) = zero;
                T_nanRow[i] = false;
            }
        }

        // Put some ints in x
        for (idx_t i = 0; i < (idx_t) floor((n*p)*(1-sparsity)); ++i)
            X( rand() % n, rand() % p ) = max( one, real_t(rand() % max_int) );

        // let B = T*X (no roundoff)
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i <= j; ++i)
                for (idx_t k = 0; k < p; ++k)
                    B(i,k) += T(i,j) * X(j,k);

        // Put Infs and NaNs in the columns of T respective to the 0s in x
        // Do not put Infs in the diagonal 
        for (idx_t j = 0; j < p; ++j) {
            for (idx_t i = 0; i < n; ++i) {
                if( X(i,j) == zero ) {
                    idx_t idx = rand() % (i+1); // 0 <= idx <= i
                    if( idx == i )
                        T( idx, i ) = nan_vec[ rand() % nan_vec.size() ];
                    else {
                        T( idx, i ) = ( rand() % 2 == 0 )
                                    ? inf_vec[ rand() % inf_vec.size() ]
                                    : nan_vec[ rand() % nan_vec.size() ];
                    }
                    T_nanRow[idx] = true;
                    break;
                }
            }
        }

        // Copy B to X
        for (idx_t j = 0; j < p; ++j)
            for (idx_t i = 0; i < n; ++i)
                X(i,j) = B(i,j);

        // Solve for X
        trsm(   Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                n, p, real_t(1.0), T, N, x, N );

        // Test if NaNs have propagated
        for (idx_t j = 0; j < p; ++j) {
            for (idx_t i = 0; i < n; ++i) {
                if( T_nanRow[i] ) {

                    std::stringstream lineT;
                    lineT << T(i,i);
                    for (idx_t col = i+1; col < n; ++col)
                        lineT << ", " << T(i,col);
                    UNSCOPED_INFO( "T[" << i << "," << i << ":] = " << lineT.str() );

                    std::stringstream lineX;
                    lineX << X(i,j);
                    for (idx_t row = i+1; row < n; ++row)
                        lineX << ", " << X(row,j);
                    UNSCOPED_INFO( "X[" << i << ":," << j << "] = " << lineX.str() );

                    UNSCOPED_INFO( "B[" << i << "," << j << "] = " << B(i,j) );

                    bool foundNaNinX = false;
                    for (idx_t k = 0; k < p; k++) {
                        if( isnan(X(i,k)) ) {
                            foundNaNinX = true;
                            break;
                        }
                    }
                    
                    CHECK( foundNaNinX );
                }
            }
        }
    }}

    #undef T
    #undef X
    #undef B
}
