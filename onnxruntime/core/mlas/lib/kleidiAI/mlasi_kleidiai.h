/*++

Copyright (c) Microsoft Corporation.  All rights reserved.

Licensed under the MIT License.

Module Name:

    mlasi.h

Abstract:

    This module contains the private data structures and procedure prototypes
    for the ARM KleidiAI implementation of the Microsoft Machine Learning algebra subprogram library.

--*/

#pragma once

#include "mlasi.h"

namespace ARMKleidiAI {

class NotSupported : public std::exception{};

#if defined(USE_KLEIDIAI) && !defined(_MSVC_LANG)

#define kai_check_if_supported(check)            \
    do {                                         \
        try {                                    \
            check;                               \
        } catch (ARMKleidiAI::NotSupported) {    \
            /*intentionally fall through and     \
              fallback*/                         \
            /*Gather usage stats for integration \
              coverage*/                         \
        }                                        \
    }                                            \
    while (0)
#else

#define kai_check_if_supported(check)

#endif

//
// Buffer packing routines.
//

size_t
MLASCALL
MlasGemmPackBSize(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K
    );

void
MLASCALL
MlasGemmPackB(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K,
    const float* B,
    size_t ldb,
    void* PackedB
    );

void
MLASCALL
MlasGemmBatch(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    const MLAS_SGEMM_DATA_PARAMS* Data,
    size_t BatchSize,
    MLAS_THREADPOOL* ThreadPool
    );
}