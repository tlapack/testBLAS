// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "utils.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <regex>

using namespace testBLAS;

template<typename T>
constexpr const char* get_func_name() { return __PRETTY_FUNCTION__; }

template<typename T>
const std::string get_type_name()
{
    const std::string s = get_func_name<T>();
    const std::regex rgx(".*T = (.*)\\]");
    
    std::smatch match;
    std::regex_search(s.begin(), s.end(), match, rgx);

    return match[1];
}

TEMPLATE_TEST_CASE( "gemm does not propagate NaNs in C if beta = 0", "[NaN]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    const Layout layout = Layout::ColMajor;
    const Op transA = Op::NoTrans;
    const Op transB = Op::NoTrans;
    const idx_t m = 3;
    const idx_t n = 7;
    const idx_t k = 11;
    const idx_t lda = m;
    const idx_t ldb = k;
    const idx_t ldc = m;
    const TestType alpha = real_t( GENERATE(1,-5) );

    // Se A and B with numbers such that A*B does not overflow
    TestType const A[m*k] = {TestType(1)};
    TestType const B[k*n] = {TestType(2)};
    TestType C[m*n] = {TestType(NAN)};

    gemm( layout, transA, transB, m, n, k, alpha, A, lda, B, ldb, real_t(0), C, ldc );
    for (idx_t i = 0; i < m*n; ++i) {
        INFO( "beta = ("<< get_type_name<real_t>() << ") 0" );
        REQUIRE( !isnan(C[i]) );
    }
}

TEMPLATE_TEST_CASE( "gemm does not propagate NaNs in A or B if alpha = 0", "[NaN]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    const Layout layout = Layout::ColMajor;
    const Op transA = Op::NoTrans;
    const Op transB = Op::NoTrans;
    const idx_t m = 3;
    const idx_t n = 7;
    const idx_t k = 11;
    const idx_t lda = m;
    const idx_t ldb = k;
    const idx_t ldc = m;
    const TestType beta = real_t( GENERATE(1,-5) );

    // Se A and B with numbers such that A*B does not overflow
    TestType const A[m*k] = {TestType(NAN)};
    TestType const B[k*n] = {TestType(NAN)};
    TestType C[m*n] = {TestType(1)};

    gemm( layout, transA, transB, m, n, k, real_t(0), A, lda, B, ldb, beta, C, ldc );
    for (idx_t i = 0; i < m*n; ++i) {
        INFO( "alpha = ("<< get_type_name<real_t>() << ") 0" );
        REQUIRE( !isnan(C[i]) );
    }
}
