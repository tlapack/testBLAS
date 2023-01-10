/// @file defines.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TESTBLAS_DEFINES_HH
#define TESTBLAS_DEFINES_HH

#include <complex>

//-----------------------------------------------------------------------------
#ifndef TEST_REAL_TYPES
    #define TEST_REAL_TYPES \
        float, \
        double
#endif

#ifndef TEST_CPLX_TYPES
    #define TEST_CPLX_TYPES \
        std::complex<float>, \
        std::complex<double>
#endif

#ifndef TEST_TYPES
    #define TEST_TYPES TEST_REAL_TYPES, TEST_CPLX_TYPES
#endif

//-----------------------------------------------------------------------------
#if defined(TESTBLAS_USE_BLASPP)
    #include "blaspp_blas.hpp"
#elif defined(TESTBLAS_USE_TLAPACK)
    #include "tlapack_blas.hpp"
#endif

#endif // TESTBLAS_DEFINES_HH
