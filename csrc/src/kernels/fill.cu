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
#include <cstddef>
#include <cstdint>

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
template<typename floatX>
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
template<typename floatX>
void fill_imp(floatX* dst, floatX value, std::size_t count, cudaStream_t stream) {
    fill_kernel<<<div_ceil(count, static_cast<std::size_t>(256)), 256, 0, stream>>> (dst, value, count);
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

// ============================================================================
// Bulk zeroing (non-contiguous segments)
// ============================================================================

namespace {

__global__ void zero_segments_kernel(const std::uint64_t* ptrs,
                                     const std::uint64_t* sizes,
                                     int n) {
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
        for (std::size_t i = static_cast<std::size_t>(threadIdx.x);
             i < n_vec;
             i += static_cast<std::size_t>(blockDim.x)) {
            v[i] = make_uint4(0, 0, 0, 0);
        }
    } else {
        for (std::size_t i = static_cast<std::size_t>(threadIdx.x);
             i < vec_bytes;
             i += static_cast<std::size_t>(blockDim.x)) {
            p[i] = std::byte{0};
        }
    }

    // Tail.
    for (std::size_t i = vec_bytes + static_cast<std::size_t>(threadIdx.x);
         i < bytes;
         i += static_cast<std::size_t>(blockDim.x)) {
        p[i] = std::byte{0};
    }
}

}  // namespace

void zero_device_segments(const std::uint64_t* ptrs,
                          const std::uint64_t* sizes,
                          int n,
                          cudaStream_t stream) {
    if (n <= 0 || ptrs == nullptr || sizes == nullptr) return;
    // One block per segment; segments are large enough that per-block looping is fine.
    zero_segments_kernel<<<n, 256, 0, stream>>>(ptrs, sizes, n);
    CUDA_CHECK(cudaGetLastError());
}
