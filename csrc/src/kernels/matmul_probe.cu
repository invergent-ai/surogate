// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file matmul_probe.cu
 * @brief Synthetic operands and a bitwise output comparison for the matmul autotune.
 *
 * The autotune keeps only cuBLASLt algorithms whose output is bit-identical to the default
 * algorithm's, so tuning changes speed but never results. It compares them on synthetic operands
 * rather than live ones: live operands can hide a different summation order (a padded batch
 * leaves whole K ranges zero, so a split-K variant matches by accident).
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "kernels.h"
#include "kernels/squirrel_noise.cuh"
#include "utilities/utils.h"

namespace {

constexpr int kBlock = 256;

int grid_for(std::size_t count) {
    return static_cast<int>(std::min<std::size_t>(div_ceil(count, static_cast<std::size_t>(kBlock)), 4096));
}

// Uniform in [-1, 1): no zeros to speak of, so every K range adds to every output.
template <typename T>
__global__ void fill_matmul_probe_kernel(T* dst, std::size_t count, unsigned int seed) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {
        const unsigned int bits =
            squirrel_noise_5(static_cast<unsigned int>(i), seed + static_cast<unsigned int>(i >> 32));
        const float u = static_cast<float>(bits >> 8) * (2.0f / 16777216.0f) - 1.0f;
        dst[i] = static_cast<T>(u);
    }
}

template <typename T>
void fill_matmul_probe_impl(T* dst, std::size_t count, unsigned int seed, cudaStream_t stream) {
    if (count == 0) return;
    fill_matmul_probe_kernel<<<grid_for(count), kBlock, 0, stream>>>(dst, count, seed);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void count_word_mismatches_kernel(const std::uint32_t* a,
                                             const std::uint32_t* b,
                                             std::size_t words,
                                             std::uint32_t value_mask,
                                             unsigned long long* counts) {
    unsigned long long mismatched = 0;
    unsigned long long nonzero = 0;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < words; i += stride) {
        const std::uint32_t x = a[i];
        mismatched += x != b[i];
        nonzero += (x & value_mask) != 0;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        mismatched += __shfl_down_sync(0xffffffffu, mismatched, offset);
        nonzero += __shfl_down_sync(0xffffffffu, nonzero, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        if (mismatched) atomicAdd(&counts[0], mismatched);
        if (nonzero) atomicAdd(&counts[1], nonzero);
    }
}

}  // namespace

void fill_matmul_probe(float* dst, std::size_t count, unsigned int seed, cudaStream_t stream) {
    fill_matmul_probe_impl(dst, count, seed, stream);
}
void fill_matmul_probe(nv_bfloat16* dst, std::size_t count, unsigned int seed, cudaStream_t stream) {
    fill_matmul_probe_impl(dst, count, seed, stream);
}
void fill_matmul_probe(half* dst, std::size_t count, unsigned int seed, cudaStream_t stream) {
    fill_matmul_probe_impl(dst, count, seed, stream);
}
void fill_matmul_probe(__nv_fp8_e4m3* dst, std::size_t count, unsigned int seed, cudaStream_t stream) {
    fill_matmul_probe_impl(dst, count, seed, stream);
}
void fill_matmul_probe(__nv_fp8_e5m2* dst, std::size_t count, unsigned int seed, cudaStream_t stream) {
    fill_matmul_probe_impl(dst, count, seed, stream);
}

void count_word_mismatches(const void* a,
                           const void* b,
                           std::size_t bytes,
                           int elem_bytes,
                           unsigned long long* counts,
                           cudaStream_t stream) {
    if (bytes % 4 != 0 || (reinterpret_cast<std::uintptr_t>(a) | reinterpret_cast<std::uintptr_t>(b)) % 4 != 0) {
        throw std::invalid_argument("count_word_mismatches: buffers must be 4-byte aligned whole words");
    }
    // Clear each element's sign bit so -0 counts as zero.
    std::uint32_t value_mask = 0;
    switch (elem_bytes) {
        case 1: value_mask = 0x7F7F7F7Fu; break;
        case 2: value_mask = 0x7FFF7FFFu; break;
        case 4: value_mask = 0x7FFFFFFFu; break;
        default: throw std::invalid_argument("count_word_mismatches: elem_bytes must be 1, 2 or 4");
    }
    const std::size_t words = bytes / 4;
    if (words == 0) return;
    count_word_mismatches_kernel<<<grid_for(words), kBlock, 0, stream>>>(static_cast<const std::uint32_t*>(a),
                                                                         static_cast<const std::uint32_t*>(b),
                                                                         words,
                                                                         value_mask,
                                                                         counts);
    CUDA_CHECK(cudaGetLastError());
}
