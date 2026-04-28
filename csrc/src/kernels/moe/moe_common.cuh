// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#ifndef SUROGATE_SRC_KERNELS_MOE_MOE_COMMON_CUH
#define SUROGATE_SRC_KERNELS_MOE_MOE_COMMON_CUH

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "kernels/kernels.h"
#include "kernels/kernel_utils.cuh"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

template <typename T>
constexpr cudaDataType_t cublas_dtype() {
    if constexpr (std::is_same_v<T, float>)
        return CUDA_R_32F;
    else if constexpr (std::is_same_v<T, nv_bfloat16>)
        return CUDA_R_16BF;
    else if constexpr (std::is_same_v<T, half>)
        return CUDA_R_16F;
    else
        static_assert(!sizeof(T), "Unsupported type for cuBLAS");
}

#endif  // SUROGATE_SRC_KERNELS_MOE_MOE_COMMON_CUH
