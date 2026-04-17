// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file convert.cu
 * @brief CUDA kernels for data type conversion between floating-point formats.
 *
 * Provides element-wise conversion between FP32, BF16, and FP16 data types.
 */

#include <cassert>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "utilities/utils.h"
#include "utilities/vec.cuh"

/**
 * @brief CUDA kernel for element-wise data type conversion.
 *
 * Converts each element from source type to destination type using static_cast.
 * Each thread processes one element.
 *
 * @tparam Src Source data type.
 * @tparam Dst Destination data type.
 * @param[out] target Destination array of size elements.
 * @param[in] source Source array of size elements.
 * @param size Number of elements to convert.
 *
 * @note TODO: This kernel could be optimized with vectorized loads/stores.
 */
template <typename Src, typename Dst>
__global__ void convert_dtype_kernel(Dst* target, const Src* source, std::size_t size) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) {
        return;
    }
    target[tid] = static_cast<Dst>(source[tid]);
    // TODO vectorize
}

/**
 * @brief Template launcher for data type conversion kernel.
 *
 * Launches the convert_dtype_kernel with 128 threads per block.
 * Runs on the default stream.
 *
 * @tparam Src Source data type.
 * @tparam Dst Destination data type.
 * @param[out] target Destination array.
 * @param[in] source Source array.
 * @param size Number of elements to convert.
 */
template <typename Src, typename Dst>
void convert_dtype_launcher(Dst* target, const Src* source, std::size_t size, cudaStream_t stream) {
    unsigned long n_blocks = div_ceil(size, 128ul);
    convert_dtype_kernel<Src, Dst><<<n_blocks, 128, 0, stream>>>(target, source, size);
}

/**
 * @brief Converts BF16 array to FP32.
 *
 * @param[out] target Destination array in FP32.
 * @param[in] source Source array in BF16.
 * @param size Number of elements to convert.
 */
void convert_dtype(float* target, const nv_bfloat16* source, std::size_t size) {
    convert_dtype_launcher(target, source, size, /*stream=*/0);
}

/**
 * @brief Converts FP32 array to BF16.
 *
 * @param[out] target Destination array in BF16.
 * @param[in] source Source array in FP32.
 * @param size Number of elements to convert.
 */
void convert_dtype(nv_bfloat16* target, const float* source, std::size_t size) {
    convert_dtype_launcher(target, source, size, /*stream=*/0);
}

/**
 * @brief Converts FP16 (half) array to BF16.
 *
 * @param[out] target Destination array in BF16.
 * @param[in] source Source array in FP16.
 * @param size Number of elements to convert.
 */
void convert_dtype(nv_bfloat16* target, const half* source, std::size_t size) {
    convert_dtype_launcher(target, source, size, /*stream=*/0);
}

void convert_dtype(float* target, const nv_bfloat16* source, std::size_t size, cudaStream_t stream) {
    convert_dtype_launcher(target, source, size, stream);
}

void convert_dtype(nv_bfloat16* target, const float* source, std::size_t size, cudaStream_t stream) {
    convert_dtype_launcher(target, source, size, stream);
}

void convert_dtype(nv_bfloat16* target, const half* source, std::size_t size, cudaStream_t stream) {
    convert_dtype_launcher(target, source, size, stream);
}

void convert_dtype(nv_bfloat16* target, const __nv_fp8_e4m3* source, std::size_t size) {
    convert_dtype_launcher(target, source, size, /*stream=*/0);
}

void convert_dtype(nv_bfloat16* target, const __nv_fp8_e4m3* source, std::size_t size, cudaStream_t stream) {
    convert_dtype_launcher(target, source, size, stream);
}
