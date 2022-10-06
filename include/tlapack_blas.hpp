// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TESTBLAS_BLAS_HH
#define TESTBLAS_BLAS_HH

#define TLAPACK_SIZE_T std::size_t
#define TLAPACK_INT_T std::int64_t

#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/legacy_api/blas.hpp>

using idx_t = TLAPACK_SIZE_T;
using int_t = TLAPACK_INT_T;

namespace testBLAS {

    using std::min;
    using std::max;

    using tlapack::real_type;
    using tlapack::complex_type;
    using tlapack::scalar_type;
    using tlapack::is_complex;

    using tlapack::Layout;
    using tlapack::Op;
    using tlapack::Uplo;
    using tlapack::Diag;
    using tlapack::Side;

    // =============================================================================
    // Level 1 BLAS template implementations

    using tlapack::asum;
    using tlapack::axpy;
    using tlapack::copy;
    using tlapack::dot;
    using tlapack::dotu;
    using tlapack::iamax;
    using tlapack::nrm2;
    using tlapack::rot;
    using tlapack::rotg;
    using tlapack::rotm;
    using tlapack::rotmg;
    using tlapack::scal;
    using tlapack::swap;

    // =============================================================================
    // Level 2 BLAS template implementations

    using tlapack::gemv;
    using tlapack::ger;
    using tlapack::geru;
    using tlapack::hemv;
    using tlapack::her;
    using tlapack::her2;
    using tlapack::symv;
    using tlapack::syr;
    using tlapack::syr2;
    using tlapack::trmv;
    using tlapack::trsv;

    // =============================================================================
    // Level 3 BLAS template implementations

    using tlapack::gemm;
    using tlapack::hemm;
    using tlapack::herk;
    using tlapack::her2k;
    using tlapack::symm;
    using tlapack::syrk;
    using tlapack::syr2k;
    using tlapack::trmm;
    using tlapack::trsm;

}

#endif // TESTBLAS_BLAS_HH
