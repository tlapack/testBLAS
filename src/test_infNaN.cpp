// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <iostream>
#include <tblas.hpp>
#include "test_types.hpp"
#include "utils.hpp"

#include <catch2/catch.hpp>
#include <limits>

/*
#define BLAS_CHECK( cond ) do { \
    bool cond_check = ( cond ); \
    CHECK( cond_check ); \
    if( !cond_check ) \
        std::cout << x << std::endl; \
} while(false)
*/

#ifdef TESTBLAS_PRINT_INPUT
    #define BLAS_CHECK( cond ) do { \
        bool cond_check = ( cond ); \
        CHECK( cond_check ); \
        if( !cond_check ) \
            std::cout << x << std::endl; \
    } while(false)
#else
    #define BLAS_CHECK( cond ) CHECK( cond )
#endif

using namespace blas;

TEMPLATE_TEST_CASE( "NANs work as expected", "[NaN]", TEST_TYPES ) {
    
    std::vector<TestType> nan_vec;
    testBLAS::set_nan_vector( nan_vec );

    SECTION( "NAN != NAN" ) {
        for (const auto& x : nan_vec)
            BLAS_CHECK( x != x );
    }

    SECTION( "isnan(NAN) == true" ) {
        for (const auto& x : nan_vec)
            BLAS_CHECK( isnan(x) );
    }
}

TEMPLATE_TEST_CASE( "Infs work as expected", "[Inf]", TEST_TYPES ) {
    
    std::vector<TestType> inf_vec;
    testBLAS::set_inf_vector( inf_vec );

    SECTION( "Inf is not a NaN" ) {
        for (const auto& x : inf_vec)
            BLAS_CHECK( !isnan(x) );
    }

    SECTION( "Inf is Infinity" ) {
        for (const auto& x : inf_vec)
            BLAS_CHECK( isinf(x) );
    }
}

TEMPLATE_TEST_CASE( "blas::abs works as expected", "[NaN][Inf]", TEST_STD_TYPES ) {
    
    std::vector<TestType> nan_vec;
    testBLAS::set_nan_vector( nan_vec );
    
    std::vector<TestType> inf_vec;
    testBLAS::set_inf_vector( inf_vec );

    SECTION( "isnan(blas::abs(NAN)) == true" ) {
        for (const auto& x : nan_vec)
            BLAS_CHECK( isnan(blas::abs(x)) );
    }

    SECTION( "isinf(blas::abs(Inf)) == true" ) {
        for (const auto& x : inf_vec)
            BLAS_CHECK( isinf(blas::abs(x)) );
    }

    SECTION( "isinf(blas::abs(NAN)) == false" ) {
        for (const auto& x : nan_vec)
            BLAS_CHECK( !isinf(blas::abs(x)) );
    }
}

TEMPLATE_TEST_CASE( "std::abs works as expected", "[NaN][Inf]", TEST_STD_TYPES ) {
    
    std::vector<TestType> nan_vec;
    testBLAS::set_nan_vector( nan_vec );
    
    std::vector<TestType> inf_vec;
    testBLAS::set_inf_vector( inf_vec );

    // SECTION( "isnan(std::abs(NAN)) == true" ) {
    //     for (const auto& x : nan_vec)
    //         BLAS_CHECK( isnan(std::abs(x)) );
    // }

    SECTION( "isinf(std::abs(Inf)) == true" ) {
        for (const auto& x : inf_vec)
            BLAS_CHECK( isinf(std::abs(x)) );
    }

    // SECTION( "isinf(std::abs(NAN)) == false" ) {
    //     for (const auto& x : nan_vec)
    //         BLAS_CHECK( !isinf(std::abs(x)) );
    // }

    SECTION( "isnan(std::abs(Inf)) == false" ) {
        for (const auto& x : inf_vec)
            BLAS_CHECK( !isnan(std::abs(x)) );
    }

    // SECTION( "isnan(abs(NAN)) == true" ) {
    //     for (const auto& x : nan_vec)
    //         BLAS_CHECK( isnan(abs(x)) );
    // }

    SECTION( "isinf(abs(Inf)) == true" ) {
        for (const auto& x : inf_vec)
            BLAS_CHECK( isinf(abs(x)) );
    }

    // SECTION( "isinf(abs(NAN)) == false" ) {
    //     for (const auto& x : nan_vec)
    //         BLAS_CHECK( !isinf(abs(x)) );
    // }

    SECTION( "isnan(abs(Inf)) == false" ) {
        for (const auto& x : inf_vec)
            BLAS_CHECK( !isnan(abs(x)) );
    }
}

#ifdef USE_MPFR
TEMPLATE_TEST_CASE( "mpfr::abs works as expected", "[NaN][Inf]", mpfr::mpreal ) {
    
    std::vector<TestType> nan_vec;
    testBLAS::set_nan_vector( nan_vec );
    
    std::vector<TestType> inf_vec;
    testBLAS::set_inf_vector( inf_vec );

    SECTION( "isnan(mpfr::abs(NAN)) == true" ) {
        for (const auto& x : nan_vec)
            BLAS_CHECK( isnan(mpfr::abs(x)) );
    }

    SECTION( "isinf(mpfr::abs(Inf)) == true" ) {
        for (const auto& x : inf_vec)
            BLAS_CHECK( isinf(mpfr::abs(x)) );
    }

    SECTION( "isinf(mpfr::abs(NAN)) == false" ) {
        for (const auto& x : nan_vec)
            BLAS_CHECK( !isinf(mpfr::abs(x)) );
    }
}
TEST_CASE( "std::abs(std::complex<mpfr::mpreal>) works as expected", "[NaN][Inf]") {
    using TestType = std::complex<mpfr::mpreal>;
    
    std::vector<TestType> nan_vec;
    testBLAS::set_nan_vector( nan_vec );
    
    std::vector<TestType> inf_vec;
    testBLAS::set_inf_vector( inf_vec );

    // std::abs( std::complex<mpfr::mpreal>(0 + NaN*i) ) == 0
    // SECTION( "isnan(std::abs(NAN)) == true" ) {
    //     for (const auto& x : nan_vec)
    //         BLAS_CHECK( isnan(std::abs(x)) );
    // }

    // SECTION( "isinf(std::abs(Inf)) == true" ) {
    //     for (const auto& x : inf_vec)
    //         BLAS_CHECK( isinf(std::abs(x)) );
    // }

    SECTION( "isinf(std::abs(NAN)) == false" ) {
        for (const auto& x : nan_vec)
            BLAS_CHECK( !isinf(std::abs(x)) );
    }

    // SECTION( "isnan(std::abs(Inf)) == false" ) {
    //     for (const auto& x : inf_vec)
    //         BLAS_CHECK( !isnan(std::abs(x)) );
    // }

    // abs( std::complex<mpfr::mpreal>(0 + NaN*i) ) == 0
    // SECTION( "isnan(abs(NAN)) == true" ) {
    //     for (const auto& x : nan_vec)
    //         BLAS_CHECK( isnan(abs(x)) );
    // }

    // SECTION( "isinf(abs(Inf)) == true" ) {
    //     for (const auto& x : inf_vec)
    //         BLAS_CHECK( isinf(abs(x)) );
    // }

    SECTION( "isinf(abs(NAN)) == false" ) {
        for (const auto& x : nan_vec)
            BLAS_CHECK( !isinf(abs(x)) );
    }

    // SECTION( "isnan(abs(Inf)) == false" ) {
    //     for (const auto& x : inf_vec)
    //         BLAS_CHECK( !isnan(abs(x)) );
    // }
}
#endif
