/// @file tlapack_blas.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
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

    using tlapack::Layout;
    using tlapack::Op;
    using tlapack::Uplo;
    using tlapack::Diag;
    using tlapack::Side;

    template<typename T>
    struct is_complex {
        static constexpr bool value = tlapack::is_complex<T>;
    };

    // =============================================================================
    // Level 1 BLAS template implementations

    using tlapack::legacy::asum;
    using tlapack::legacy::axpy;
    using tlapack::legacy::copy;
    using tlapack::legacy::dot;
    using tlapack::legacy::dotu;
    using tlapack::legacy::iamax;
    using tlapack::legacy::nrm2;
    using tlapack::legacy::rot;
    using tlapack::legacy::rotg;
    using tlapack::legacy::rotm;
    using tlapack::legacy::rotmg;
    using tlapack::legacy::scal;
    using tlapack::legacy::swap;

    // =============================================================================
    // Level 2 BLAS template implementations

    using tlapack::legacy::gemv;
    using tlapack::legacy::ger;
    using tlapack::legacy::geru;
    using tlapack::legacy::hemv;
    using tlapack::legacy::her;
    using tlapack::legacy::her2;
    using tlapack::legacy::symv;
    using tlapack::legacy::syr;
    using tlapack::legacy::syr2;
    using tlapack::legacy::trmv;
    using tlapack::legacy::trsv;

    // =============================================================================
    // Level 3 BLAS template implementations

    using tlapack::legacy::gemm;
    using tlapack::legacy::hemm;
    using tlapack::legacy::herk;
    using tlapack::legacy::her2k;
    using tlapack::legacy::symm;
    using tlapack::legacy::syrk;
    using tlapack::legacy::syr2k;
    using tlapack::legacy::trmm;
    using tlapack::legacy::trsm;

}

#endif // TESTBLAS_BLAS_HH
