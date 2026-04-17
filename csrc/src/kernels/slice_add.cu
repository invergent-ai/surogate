// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file slice_add.cu
 * @brief CUDA kernels for adding a 2D slice to a destination tensor.
 *
 * Provides GPU-accelerated element-wise addition of a source tensor to a
 * column slice of a destination tensor. Useful for operations like adding
 * gradients to a specific column range of a larger gradient buffer.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "utilities/utils.h"

/**
 * @brief Device helper to cast float result back to target type.
 * @tparam T Target data type.
 * @param v Float value to cast.
 * @return Value cast to type T.
 */
template <typename T>
__device__ inline T add_cast(float v);

/// @brief Specialization for FP32 (identity cast).
template <>
__device__ inline float add_cast<float>(float v) {
    return v;
}

/// @brief Specialization for BF16 (converts from float).
template <>
__device__ inline nv_bfloat16 add_cast<nv_bfloat16>(float v) {
    return (nv_bfloat16)v;
}

/**
 * @brief CUDA kernel to add a source tensor to a column slice of destination.
 *
 * Each thread handles one element. Computes the row/column from linear index,
 * applies the column offset to find the destination location, and performs
 * element-wise addition: dst[r, dst_col_offset + c] += src[r, c]
 *
 * @tparam T Data type (float or nv_bfloat16).
 * @param[in,out] dst Destination tensor of shape (rows, dst_cols).
 * @param[in] src Source tensor of shape (rows, src_cols).
 * @param rows Number of rows.
 * @param dst_cols Number of columns in destination.
 * @param src_cols Number of columns in source.
 * @param dst_col_offset Starting column offset in destination.
 */
template <typename T>
__global__ void
add_2d_slice_kernel(T* dst, const T* src, long rows, long dst_cols, long src_cols, long dst_col_offset) {
    long idx = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
    long total = rows * src_cols;
    if (idx >= total) return;
    long r = idx / src_cols;
    long c = idx - r * src_cols;
    long dst_idx = r * dst_cols + dst_col_offset + c;

    float a = (float)dst[dst_idx];
    float b = (float)src[idx];
    dst[dst_idx] = add_cast<T>(a + b);
}

/**
 * @brief Template launcher for 2D slice addition kernel.
 *
 * Validates bounds and launches add_2d_slice_kernel with 256 threads per block.
 *
 * @tparam T Data type (float or nv_bfloat16).
 * @param[in,out] dst Destination tensor.
 * @param[in] src Source tensor to add.
 * @param rows Number of rows.
 * @param dst_cols Number of columns in destination.
 * @param src_cols Number of columns in source.
 * @param dst_col_offset Starting column offset in destination.
 * @param stream CUDA stream for asynchronous execution.
 * @throws std::logic_error If dst_col_offset is out of bounds.
 */
template <typename T>
static void add_2d_slice_imp(T* dst,
                             const T* src,
                             long rows,
                             long dst_cols,
                             long src_cols,
                             long dst_col_offset,
                             cudaStream_t stream) {
    if (rows <= 0 || dst_cols <= 0 || src_cols <= 0) return;
    if (dst_col_offset < 0 || dst_col_offset + src_cols > dst_cols) {
        throw std::logic_error("add_2d_slice: dst_col_offset out of bounds");
    }

    constexpr int block = 256;
    long total = rows * src_cols;
    long grid = div_ceil(total, (long)block);
    add_2d_slice_kernel<T><<<grid, block, 0, stream>>>(dst, src, rows, dst_cols, src_cols, dst_col_offset);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Adds a source FP32 tensor to a column slice of destination.
 *
 * Performs: dst[:, dst_col_offset : dst_col_offset + src_cols] += src
 *
 * @param[in,out] dst Destination FP32 tensor of shape (rows, dst_cols).
 * @param[in] src Source FP32 tensor of shape (rows, src_cols).
 * @param rows Number of rows.
 * @param dst_cols Number of columns in destination.
 * @param src_cols Number of columns in source.
 * @param dst_col_offset Starting column offset in destination.
 * @param stream CUDA stream.
 */
void add_2d_slice(float* dst,
                  const float* src,
                  long rows,
                  long dst_cols,
                  long src_cols,
                  long dst_col_offset,
                  cudaStream_t stream) {
    add_2d_slice_imp(dst, src, rows, dst_cols, src_cols, dst_col_offset, stream);
}

/**
 * @brief Adds a source BF16 tensor to a column slice of destination.
 *
 * Performs: dst[:, dst_col_offset : dst_col_offset + src_cols] += src
 *
 * @param[in,out] dst Destination BF16 tensor of shape (rows, dst_cols).
 * @param[in] src Source BF16 tensor of shape (rows, src_cols).
 * @param rows Number of rows.
 * @param dst_cols Number of columns in destination.
 * @param src_cols Number of columns in source.
 * @param dst_col_offset Starting column offset in destination.
 * @param stream CUDA stream.
 */
void add_2d_slice(nv_bfloat16* dst,
                  const nv_bfloat16* src,
                  long rows,
                  long dst_cols,
                  long src_cols,
                  long dst_col_offset,
                  cudaStream_t stream) {
    add_2d_slice_imp(dst, src, rows, dst_cols, src_cols, dst_col_offset, stream);
}
