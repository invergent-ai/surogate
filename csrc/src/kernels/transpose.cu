// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file transpose.cu
 * @brief CUDA kernels for matrix transpose operations.
 *
 * Provides GPU-accelerated matrix transpose with automatic vectorization
 * selection based on data type and alignment. Supports FP32, BF16, and FP8 formats.
 */

#include "transpose_template.cuh"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

/**
 * @brief CUDA kernel for matrix transpose using tiled approach.
 *
 * Uses apply_and_transpose_helper with identity transformation to perform
 * pure transpose. Each thread handles a VEC_SIZE x VEC_SIZE tile.
 *
 * @tparam VEC_SIZE Tile dimension for vectorized access.
 * @tparam T Data type.
 * @param[out] dest Destination array of shape (cols, rows).
 * @param[in] src Source array of shape (rows, cols).
 * @param rows Number of rows in source.
 * @param cols Number of columns in source.
 */
template <int VEC_SIZE, typename T>
__global__ void transpose_kernel(T* dest, const T* src, int rows, int cols) {
    apply_and_transpose_helper<VEC_SIZE>([](auto&& a) { return a; }, dest, src, rows, cols);
}

/**
 * @brief Template launcher for matrix transpose with automatic vectorization.
 *
 * Selects optimal VEC_SIZE based on data type and alignment:
 * - VEC_SIZE=16: For 1-byte types (FP8) when dimensions are multiples of 16
 * - VEC_SIZE=8: For 1-2 byte types (FP8, BF16) when dimensions are multiples of 8
 * - VEC_SIZE=1: Fallback for unaligned cases
 *
 * @tparam T Data type.
 * @param[out] dst Destination array of shape (cols, rows).
 * @param[in] src Source array of shape (rows, cols).
 * @param rows Number of rows in source.
 * @param cols Number of columns in source.
 * @param stream CUDA stream for asynchronous execution.
 */
template <typename T>
void transpose_imp(T* dst, const T* src, int rows, int cols, cudaStream_t stream) {
    if (rows % 16 == 0 && cols % 16 == 0 && sizeof(T) == 1) {
        dim3 block_size = {8, 8};
        dim3 grid_size = {(unsigned)div_ceil(rows, 16 * (int)block_size.x),
                          (unsigned)div_ceil(cols, 16 * (int)block_size.y)};
        transpose_kernel<16><<<grid_size, block_size, 0, stream>>>(dst, src, rows, cols);
        CUDA_CHECK(cudaGetLastError());
    } else if (rows % 8 == 0 && cols % 8 == 0 && sizeof(T) <= 2) {
        dim3 block_size = {8, 8};
        dim3 grid_size = {(unsigned)div_ceil(rows, 8 * (int)block_size.x),
                          (unsigned)div_ceil(cols, 8 * (int)block_size.y)};
        transpose_kernel<8><<<grid_size, block_size, 0, stream>>>(dst, src, rows, cols);
        CUDA_CHECK(cudaGetLastError());
    } else {
        dim3 block_size = {8, 8};
        dim3 grid_size = {(unsigned)div_ceil(rows, (int)block_size.x), (unsigned)div_ceil(cols, (int)block_size.y)};
        transpose_kernel<1><<<grid_size, block_size, 0, stream>>>(dst, src, rows, cols);
        CUDA_CHECK(cudaGetLastError());
    }
}

/// @brief Transposes an FP32 matrix.
void transpose(float* dst, const float* src, int rows, int cols, cudaStream_t stream) {
    transpose_imp(dst, src, rows, cols, stream);
}

/// @brief Transposes an FP8 E4M3 matrix.
void transpose(__nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src, int rows, int cols, cudaStream_t stream) {
    transpose_imp(dst, src, rows, cols, stream);
}

/// @brief Transposes an FP8 E5M2 matrix.
void transpose(__nv_fp8_e5m2* dst, const __nv_fp8_e5m2* src, int rows, int cols, cudaStream_t stream) {
    transpose_imp(dst, src, rows, cols, stream);
}

/// @brief Transposes a BF16 matrix.
void transpose(nv_bfloat16* dst, const nv_bfloat16* src, int rows, int cols, cudaStream_t stream) {
    transpose_imp(dst, src, rows, cols, stream);
}
