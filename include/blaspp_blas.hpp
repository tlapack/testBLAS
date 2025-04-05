/// @file blaspp_blas.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TESTBLAS_BLAS_HH
#define TESTBLAS_BLAS_HH

#include <blas.hh> // from BLAS++

using idx_t = std::int64_t;
using int_t = std::int64_t;

namespace testBLAS {

    using std::min;
    using std::max;

    using blas::real_type;
    using blas::complex_type;
    using blas::scalar_type;
    using blas::is_complex;

    using blas::Layout;
    using blas::Op;
    using blas::Uplo;
    using blas::Diag;
    using blas::Side;

    // =============================================================================
    // Level 1 BLAS template implementations

    using blas::rotg;
    using blas::rotmg;
    
    // Return if n <= 0 like in Standard BLAS
    template< typename T >
    inline real_type<T> asum( idx_t n, T const *x, int_t incx )
    {
        if( n <= 0 ) return 0;
        return ::blas::asum( n, x, incx );
    }
    
    // Return if n <= 0 like in Standard BLAS
    template< typename TX, typename TY >
    inline void axpy(
        idx_t n,
        scalar_type<TX, TY> alpha,
        TX const *x, int_t incx,
        TY       *y, int_t incy )
    {
        if( n <= 0 ) return;
        ::blas::axpy( n, alpha, x, incx, y, incy );
    }
    
    // Return if n <= 0 like in Standard BLAS
    template< typename TX, typename TY >
    inline void copy(
        idx_t n,
        TX const *x, int_t incx,
        TY       *y, int_t incy )
    {
        if( n <= 0 ) return;
        ::blas::copy( n, x, incx, y, incy );
    }
    
    // Return if n <= 0 like in Standard BLAS
    template< typename TX, typename TY >
    inline scalar_type<TX,TY> dot(
        idx_t n,
        TX const *x, int_t incx,
        TY const *y, int_t incy )
    {
        if( n <= 0 ) return 0;
        return ::blas::dot( n, x, incx, y, incy );
    }
    
    // Return if n <= 0 like in Standard BLAS
    template< typename TX, typename TY >
    inline scalar_type<TX,TY> dotu(
        idx_t n,
        TX const *x, int_t incx,
        TY const *y, int_t incy )
    {
        if( n <= 0 ) return 0;
        return ::blas::dotu( n, x, incx, y, incy );
    }
    
    // Return if n <= 0 like in Standard BLAS
    template< typename T >
    inline idx_t iamax( idx_t n, T const *x, int_t incx )
    {
        if( n <= 0 ) return 0;
        return ::blas::iamax( n, x, incx );
    }
    
    // Return if n <= 0 like in Standard BLAS
    template< typename T >
    inline real_type<T> nrm2( idx_t n, T const *x, int_t incx )
    {
        if( n <= 0 ) return 0;
        return ::blas::nrm2( n, x, incx );
    }

    // Return if n <= 0 like in Standard BLAS
    template< typename TX, typename TY >
    void rot(
        idx_t n,
        TX *x, int_t incx,
        TY *y, int_t incy,
        real_type<TX, TY> c,
        real_type<TX, TY> s )
    {
        if( n <= 0 ) return;
        ::blas::rot( n, x, incx, y, incy, c, s );
    }

    // Return if n <= 0 like in Standard BLAS
    template< typename TX, typename TY >
    void rotm(
        idx_t n,
        TX *x, int_t incx,
        TY *y, int_t incy,
        scalar_type<TX, TY> const param[5] )
    {
        if( n <= 0 ) return;
        ::blas::rotm( n, x, incx, y, incy, param );
    }
    
    // Return if n <= 0 like in Standard BLAS
    template< typename T >
    inline void scal( idx_t n, T alpha, T *x, int_t incx )
    {
        if( n <= 0 ) return;
        ::blas::scal( n, alpha, x, incx );
    }
    
    // Return if n <= 0 like in Standard BLAS
    template< typename TX, typename TY >
    inline void swap(
        idx_t n,
        TX *x, int_t incx,
        TY *y, int_t incy )
    {
        if( n <= 0 ) return;
        ::blas::swap( n, x, incx, y, incy );
    }

    // =============================================================================
    // Level 2 BLAS template implementations

    using blas::gemv;
    using blas::ger;
    using blas::geru;
    using blas::hemv;
    using blas::her;
    using blas::her2;
    using blas::symv;
    using blas::syr;
    using blas::syr2;
    using blas::trmv;
    using blas::trsv;

    // =============================================================================
    // Level 3 BLAS template implementations

    using blas::gemm;
    using blas::hemm;
    using blas::herk;
    using blas::her2k;
    using blas::symm;
    using blas::syrk;
    using blas::syr2k;
    using blas::trmm;
    using blas::trsm;

}

#endif // TESTBLAS_BLAS_HH
