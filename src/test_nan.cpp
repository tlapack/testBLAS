// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch.hpp>
#include <tblas.hpp>
#include "test_types.hpp"
#include <limits>

using namespace blas;

TEST_CASE( "NANs work as expected", "[NaN]" ) {

    SECTION( "NAN != NAN" ) {
        CHECK( NAN != NAN );
        CHECK( std::numeric_limits<float>::quiet_NaN() != std::numeric_limits<float>::quiet_NaN() );
        CHECK( std::numeric_limits<float>::signaling_NaN() != std::numeric_limits<float>::signaling_NaN() );
    }

    SECTION( "INF is not NAN" ) {
        const float inf = std::numeric_limits<float>::infinity();
        CHECK_FALSE( isnan(inf) );
        CHECK_FALSE( isnan(-inf) );
    }

    SECTION( "isnan() identifies complex NAN" ) {
        CHECK( isnan( complex_type<float>( 0 ,NAN) ) );
        CHECK( isnan( complex_type<float>(NAN, 0 ) ) );
        CHECK( isnan( complex_type<float>(NAN,NAN) ) );
    }
}

TEMPLATE_TEST_CASE( "iamax returns the first NaN",
                    "[iamax][BLASlv1][NaN]", TEST_TYPES ) {
    using real_t = real_type<TestType>;

    // Constants:
    const real_t huge = std::numeric_limits<real_t>::max();
    const real_t inf = ( std::numeric_limits<real_t>::has_infinity )
      ? std::numeric_limits<real_t>::infinity()
      : real_t(1.0)/real_t(0.0);
    const real_t aNaN = NAN;
    
    // Tests:
    { TestType const x[] = {aNaN, inf, aNaN};
      CHECK( iamax( 3, x, 1 ) == 0 ); }
    { TestType const x[] = {huge, aNaN, aNaN};
      CHECK( iamax( 3, x, 1 ) == 1 ); }
    { TestType const x[] = {inf, huge, aNaN};
      CHECK( iamax( 3, x, 1 ) == 2 ); }
}