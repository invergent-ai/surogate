// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file fill.cu
 * @brief CUDA kernels for filling arrays with constant values.
 *
 * Provides GPU-accelerated memory initialization with a specified value.
 */

#include <cassert>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "utilities/tensor.h"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

/**
 * @brief CUDA kernel for filling an array with a constant value.
 *
 * Each thread writes one element. Simple implementation without vectorization.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] dst Destination array to fill.
 * @param value Constant value to write to all elements.
 * @param count Number of elements to fill.
 *
 * @note TODO: This kernel could be optimized with vectorized stores.
 */
template <typename floatX>
__global__ void fill_kernel(floatX* dst, floatX value, std::size_t count) {
    long id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= count) return;
    // TODO vectorize
    dst[id] = value;
}

/**
 * @brief Template launcher for the fill kernel.
 *
 * Launches the fill_kernel with 256 threads per block.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] dst Destination array to fill.
 * @param value Constant value to write.
 * @param count Number of elements to fill.
 * @param stream CUDA stream for asynchronous execution.
 */
template <typename floatX>
void fill_imp(floatX* dst, floatX value, std::size_t count, cudaStream_t stream) {
    fill_kernel<<<div_ceil(count, static_cast<std::size_t>(256)), 256, 0, stream>>>(dst, value, count);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Fills an FP32 array with a constant value.
 *
 * @param[out] dst Destination array in FP32.
 * @param value Constant value to write to all elements.
 * @param count Number of elements to fill.
 * @param stream CUDA stream for asynchronous execution.
 */
void fill_constant(float* dst, float value, std::size_t count, cudaStream_t stream) {
    fill_imp(dst, value, count, stream);
}

/**
 * @brief Fills a BF16 array with a constant value.
 *
 * @param[out] dst Destination array in BF16.
 * @param value Constant value to write to all elements.
 * @param count Number of elements to fill.
 * @param stream CUDA stream for asynchronous execution.
 */
void fill_constant(nv_bfloat16* dst, nv_bfloat16 value, std::size_t count, cudaStream_t stream) {
    fill_imp(dst, value, count, stream);
}

namespace {

__global__ void dense_cu_seqlens_kernel(int32_t* cu_seqlens, int num_docs, int max_doc_seqlen) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > num_docs) return;
    cu_seqlens[i] = i * max_doc_seqlen;
}

}  // namespace

void fill_dense_cu_seqlens(int32_t* cu_seqlens, int num_docs, int max_doc_seqlen, cudaStream_t stream) {
    if (cu_seqlens == nullptr || num_docs < 0 || max_doc_seqlen < 0) {
        throw std::invalid_argument("fill_dense_cu_seqlens: invalid arguments");
    }
    constexpr int kBlockSize = 256;
    const int grid =
        static_cast<int>(div_ceil(static_cast<std::size_t>(num_docs + 1), static_cast<std::size_t>(kBlockSize)));
    dense_cu_seqlens_kernel<<<grid, kBlockSize, 0, stream>>>(cu_seqlens, num_docs, max_doc_seqlen);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Bulk zeroing (non-contiguous segments)
// ============================================================================

namespace {

__global__ void zero_segments_kernel(const std::uint64_t* ptrs, const std::uint64_t* sizes, int n) {
    const int seg = static_cast<int>(blockIdx.x);
    if (seg >= n) return;

    const std::uint64_t raw_ptr = ptrs[seg];
    const std::uint64_t raw_size = sizes[seg];
    if (raw_ptr == 0 || raw_size == 0) return;

    auto* p = reinterpret_cast<std::byte*>(static_cast<std::uintptr_t>(raw_ptr));
    const std::size_t bytes = static_cast<std::size_t>(raw_size);

    // Vectorized stores for the bulk (16 bytes at a time).
    const std::size_t vec_bytes = bytes & ~static_cast<std::size_t>(15);
    const std::uintptr_t p_u = reinterpret_cast<std::uintptr_t>(p);
    // p comes from cudaMalloc/allocator, should be well-aligned; keep it defensive.
    if ((p_u & 0xF) == 0) {
        auto* v = reinterpret_cast<uint4*>(p);
        const std::size_t n_vec = vec_bytes / 16;
        for (std::size_t i = static_cast<std::size_t>(threadIdx.x); i < n_vec;
             i += static_cast<std::size_t>(blockDim.x)) {
            v[i] = make_uint4(0, 0, 0, 0);
        }
    } else {
        for (std::size_t i = static_cast<std::size_t>(threadIdx.x); i < vec_bytes;
             i += static_cast<std::size_t>(blockDim.x)) {
            p[i] = std::byte{0};
        }
    }

    // Tail.
    for (std::size_t i = vec_bytes + static_cast<std::size_t>(threadIdx.x); i < bytes;
         i += static_cast<std::size_t>(blockDim.x)) {
        p[i] = std::byte{0};
    }
}

}  // namespace

void zero_device_segments(const std::uint64_t* ptrs, const std::uint64_t* sizes, int n, cudaStream_t stream) {
    if (n <= 0 || ptrs == nullptr || sizes == nullptr) return;
    // One block per segment; segments are large enough that per-block looping is fine.
    zero_segments_kernel<<<n, 256, 0, stream>>>(ptrs, sizes, n);
    CUDA_CHECK(cudaGetLastError());
}

namespace {

template <typename T>
__global__ void zero_matrix_columns_kernel(T* data, long rows, long cols, long col_start, long col_end) {
    const long width = col_end - col_start;
    const long total = rows * width;
    for (long id = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; id < total;
         id += static_cast<long>(blockDim.x) * gridDim.x) {
        const long row = id / width;
        const long col = col_start + (id - row * width);
        data[row * cols + col] = T{};
    }
}

template <typename T>
void zero_matrix_columns_impl(T* data, long rows, long cols, long col_start, long col_end, cudaStream_t stream) {
    const long width = col_end - col_start;
    if (rows <= 0 || width <= 0) return;
    constexpr int block = 256;
    const long total = rows * width;
    const int grid = static_cast<int>(
        std::min<long>(div_ceil(static_cast<std::size_t>(total), static_cast<std::size_t>(block)), 65535));
    zero_matrix_columns_kernel<<<grid, block, 0, stream>>>(data, rows, cols, col_start, col_end);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void zero_matrix_columns(Tensor& dest, long col_start, long col_end, cudaStream_t stream) {
    if (!dest.Data) return;
    if (dest.Rank != 2) {
        throw std::invalid_argument("zero_matrix_columns: expected rank-2 tensor");
    }
    const long rows = dest.Sizes[0];
    const long cols = dest.Sizes[1];
    if (col_start < 0 || col_end < col_start || col_end > cols) {
        throw std::invalid_argument("zero_matrix_columns: invalid column range");
    }
    if (col_start == col_end || rows <= 0) {
        return;
    }

    if (dest.DType == ETensorDType::FP32) {
        zero_matrix_columns_impl(dest.get<float>(), rows, cols, col_start, col_end, stream);
    } else if (dest.DType == ETensorDType::BF16) {
        zero_matrix_columns_impl(dest.get<nv_bfloat16>(), rows, cols, col_start, col_end, stream);
    } else if (dest.DType == ETensorDType::FP16) {
        zero_matrix_columns_impl(dest.get<half>(), rows, cols, col_start, col_end, stream);
    } else {
        throw std::invalid_argument("zero_matrix_columns: unsupported tensor dtype");
    }
}
