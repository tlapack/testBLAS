// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TESTBLAS_UTILS_HH__
#define __TESTBLAS_UTILS_HH__

#include "test_types.hpp"

namespace testBLAS {

/**
 * @brief Set the vector nanVec with NaNs for test
 */
template< typename real_t >
inline void set_nan_vector(
    std::vector<real_t>& nanVec )
{
    const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
    nanVec = std::vector<real_t>({ nan });
}

/**
 * @brief Set the vector nanVec with NaNs for test
 */
template< typename real_t >
inline void set_nan_vector(
    std::vector< std::complex<real_t> >& nanVec )
{
    using Complex = std::complex<real_t>;
    const real_t inf = std::numeric_limits<real_t>::infinity();
    const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
    nanVec = std::vector< Complex >( TEST_CPLX_NAN );
}

/**
 * @brief Set the vector nanVec with Infs for test
 */
template< typename real_t >
inline void set_inf_vector( 
    std::vector<real_t>& infVec )
{
    const real_t inf = std::numeric_limits<real_t>::infinity();
    infVec = std::vector<real_t>({ inf });
}

/**
 * @brief Set the vector nanVec with Infs for test
 */
template< typename real_t >
inline void set_inf_vector( 
    std::vector< std::complex<real_t> >& infVec )
{
    using Complex = std::complex<real_t>;
    const real_t inf = std::numeric_limits<real_t>::infinity();
    infVec = std::vector< Complex >( TEST_CPLX_INF );
}

}

#endif // __TESTBLAS_UTILS_HH__