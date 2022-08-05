// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "defines.hpp"
#include "utils.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <limits>
#include <iostream>

#include <tlapack/legacy_api/blas.hpp>
#ifdef USE_MPFR
    #include <tlapack/plugins/mpreal.hpp>
#endif

using namespace tlapack;

TEMPLATE_TEST_CASE( "NANs work as expected", "[NaN]", TEST_TYPES ) {
    
    std::vector<TestType> nan_vec;
    testBLAS::set_nan_vector( nan_vec );

    SECTION( "NAN != NAN" ) {
        for (const auto& x : nan_vec)
            CHECK( x != x );
    }

    SECTION( "isnan(NAN) == true" ) {
        for (const auto& x : nan_vec)
            CHECK( isnan(x) );
    }
}

TEMPLATE_TEST_CASE( "Infs work as expected", "[Inf]", TEST_TYPES ) {
    
    std::vector<TestType> inf_vec;
    testBLAS::set_inf_vector( inf_vec );

    SECTION( "Inf is not a NaN" ) {
        for (const auto& x : inf_vec)
            CHECK( !isnan(x) );
    }

    SECTION( "Inf is Infinity" ) {
        for (const auto& x : inf_vec)
            CHECK( isinf(x) );
    }
}

TEMPLATE_TEST_CASE( "tlapack::abs works as expected", "[NaN][Inf]", TEST_TYPES ) {
    
    std::vector<TestType> nan_vec;
    testBLAS::set_nan_vector( nan_vec );
    
    std::vector<TestType> inf_vec;
    testBLAS::set_inf_vector( inf_vec );

    SECTION( "isnan(tlapack::abs(NAN)) == true" ) {
        for (const auto& x : nan_vec)
            CHECK( isnan(tlapack::abs(x)) );
    }

    SECTION( "isinf(tlapack::abs(Inf)) == true" ) {
        for (const auto& x : inf_vec)
            CHECK( isinf(tlapack::abs(x)) );
    }

    SECTION( "isinf(tlapack::abs(NAN)) == false" ) {
        for (const auto& x : nan_vec)
            CHECK( !isinf(tlapack::abs(x)) );
    }
}

TEMPLATE_TEST_CASE( "std::abs works as expected", "[NaN][Inf]", TEST_STD_TYPES ) {
    typedef real_type<TestType> real_t;
    
    std::vector<TestType> nan_vec;
    testBLAS::set_nan_vector( nan_vec );
    
    std::vector<TestType> inf_vec;
    testBLAS::set_inf_vector( inf_vec );

    SECTION( "isnan(std::abs(NAN)) == true" ) {
        for (const auto& x : nan_vec) {
            real_t y = std::abs(x);
            INFO( "std::abs( " << x << " ) = " << y );
            CHECK( isnan(y) );
        }
    }

    SECTION( "isinf(std::abs(Inf)) == true" ) {
        for (const auto& x : inf_vec) {
            real_t y = std::abs(x);
            INFO( "std::abs( " << x << " ) = " << y );
            CHECK( isinf(y) );
        }
    }

    SECTION( "isinf(std::abs(NAN)) == false" ) {
        for (const auto& x : nan_vec) {
            real_t y = std::abs(x);
            INFO( "std::abs( " << x << " ) = " << y );
            CHECK( !isinf(y) );
        }
    }

    SECTION( "isnan(std::abs(Inf)) == false" ) {
        for (const auto& x : inf_vec) {
            real_t y = std::abs(x);
            INFO( "std::abs( " << x << " ) = " << y );
            CHECK( !isnan(y) );
        }
    }

    SECTION( "isnan(abs(NAN)) == true" ) {
        for (const auto& x : nan_vec) {
            real_t y = abs(x);
            INFO( "abs( " << x << " ) = " << y );
            CHECK( isnan(y) );
        }
    }

    SECTION( "isinf(abs(Inf)) == true" ) {
        for (const auto& x : inf_vec) {
            real_t y = abs(x);
            INFO( "abs( " << x << " ) = " << y );
            CHECK( isinf(y) );
        }
    }

    SECTION( "isinf(abs(NAN)) == false" ) {
        for (const auto& x : nan_vec) {
            real_t y = abs(x);
            INFO( "abs( " << x << " ) = " << y );
            CHECK( !isinf(y) );
        }
    }

    SECTION( "isnan(abs(Inf)) == false" ) {
        for (const auto& x : inf_vec) {
            real_t y = abs(x);
            INFO( "abs( " << x << " ) = " << y );
            CHECK( !isnan(y) );
        }
    }
}

TEMPLATE_TEST_CASE( "Complex division works as expected", "[NaN][Inf]", TEST_TYPES ) {
    
    const TestType zero( 0.0 );
    const TestType one ( 1.0 );

    std::vector<TestType> nan_vec;
    testBLAS::set_nan_vector( nan_vec );
    
    std::vector<TestType> inf_vec;
    testBLAS::set_inf_vector( inf_vec );

    SECTION( "isnan( 1.0 / NaN )" ) {
        for (const auto& x : nan_vec) {
            TestType y = one / x;
            INFO( "1.0 / " << x << " = " << y );
            CHECK( isnan(y) );
        }
    }

    SECTION( "isnan( 0.0 / NaN )" ) {
        for (const auto& x : nan_vec) {
            TestType y = zero / x;
            INFO( "0.0 / " << x << " = " << y );
            CHECK( isnan(y) );
        }
    }

    SECTION( "1.0 / Inf == 0.0" ) {
        for (const auto& x : inf_vec) {
            TestType y = one / x;
            INFO( "1.0 / " << x << " = " << y );
            CHECK( y == zero );
        }
    }

    SECTION( "0.0 / Inf == 0.0" ) {
        for (const auto& x : inf_vec) {
            TestType y = zero / x;
            INFO( "0.0 / " << x << " = " << y );
            CHECK( y == zero );
        }
    }

    if( is_complex<TestType>::value ) {
    SECTION( "(1.0 + 1.0*I) / Inf == 0.0" ) {
        for (const auto& x : inf_vec) {
            complex_type<TestType> y = complex_type<TestType>(1.0,1.0) / x;
            INFO( "(1.0 + 1.0*I) / " << x << " = " << y );
            CHECK( y == zero );
        }
    }}

    SECTION( "Inf / Inf == NaN" ) {
        for (const auto& x : inf_vec) {
            TestType y = x / x;
            INFO( x << " / " << x << " = " << y );
            CHECK( isnan(y) );
        }
    }
}

#ifdef USE_MPFR
TEST_CASE( "mpfr::abs works as expected", "[NaN][Inf]" ) {
    typedef mpfr::mpreal TestType;
    
    std::vector<TestType> nan_vec;
    testBLAS::set_nan_vector( nan_vec );
    
    std::vector<TestType> inf_vec;
    testBLAS::set_inf_vector( inf_vec );

    SECTION( "isnan(mpfr::abs(NAN)) == true" ) {
        for (const auto& x : nan_vec)
            CHECK( isnan(mpfr::abs(x)) );
    }

    SECTION( "isinf(mpfr::abs(Inf)) == true" ) {
        for (const auto& x : inf_vec)
            CHECK( isinf(mpfr::abs(x)) );
    }

    SECTION( "isinf(mpfr::abs(NAN)) == false" ) {
        for (const auto& x : nan_vec)
            CHECK( !isinf(mpfr::abs(x)) );
    }
}
TEST_CASE( "std::abs(std::complex<mpfr::mpreal>) works as expected", "[NaN][Inf]") {
    typedef std::complex<mpfr::mpreal> TestType;
    typedef mpfr::mpreal real_t;
    
    std::vector<TestType> nan_vec;
    testBLAS::set_nan_vector( nan_vec );
    
    std::vector<TestType> inf_vec;
    testBLAS::set_inf_vector( inf_vec );

    SECTION( "isnan(std::abs(NAN)) == true" ) {
        for (const auto& x : nan_vec) {
            real_t y = std::abs(x);
            INFO( "std::abs( " << x << " ) = " << y );
            CHECK( isnan(y) );
        }
    }

    SECTION( "isinf(std::abs(Inf)) == true" ) {
        for (const auto& x : inf_vec) {
            real_t y = std::abs(x);
            INFO( "std::abs( " << x << " ) = " << y );
            CHECK( isinf(y) );
        }
    }

    SECTION( "isinf(std::abs(NAN)) == false" ) {
        for (const auto& x : nan_vec) {
            real_t y = std::abs(x);
            INFO( "std::abs( " << x << " ) = " << y );
            CHECK( !isinf(y) );
        }
    }

    SECTION( "isnan(std::abs(Inf)) == false" ) {
        for (const auto& x : inf_vec) {
            real_t y = std::abs(x);
            INFO( "std::abs( " << x << " ) = " << y );
            CHECK( !isnan(y) );
        }
    }
}
#endif
