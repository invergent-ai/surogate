// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file random.cu
 * @brief CUDA kernels for random number generation.
 *
 * Provides GPU-accelerated generation of normally distributed random numbers
 * using the Philox4x32 PRNG via cuRAND. Generates 4 values per thread for
 * efficiency using Box-Muller transform.
 */

#include <cassert>

#include <curand_kernel.h>

#include "utilities/utils.h"
#include "utilities/vec.cuh"

/**
 * @brief CUDA kernel to fill an array with normally distributed random values.
 *
 * Uses Philox4x32 PRNG with curand_normal4 to generate 4 normal samples per thread.
 * Each thread initializes its own RNG state based on seed and subsequence,
 * then transforms the output to the desired mean and standard deviation.
 *
 * @tparam floatX Output data type (float or nv_bfloat16).
 * @param[out] dst Destination array to fill with random values.
 * @param count Number of elements (must be multiple of 4).
 * @param mean Mean of the normal distribution.
 * @param std Standard deviation of the normal distribution.
 * @param seed Random seed for reproducibility.
 * @param subsequence Subsequence offset for parallel streams.
 */
template <typename floatX>
__global__ void rng_normal_kernel(floatX* dst,
                                  std::size_t count,
                                  float mean,
                                  float std,
                                  unsigned long long seed,
                                  unsigned long long subsequence) {
    curandStatePhilox4_32_10_t state;
    long id = 4 * (threadIdx.x + blockIdx.x * blockDim.x);
    if (id >= count) return;

    curand_init(seed, subsequence, id, &state);
    float4 normal = curand_normal4(&state);
    GenericVector<floatX, 4> cvt;
    cvt[0] = static_cast<floatX>(normal.x * std + mean);
    cvt[1] = static_cast<floatX>(normal.y * std + mean);
    cvt[2] = static_cast<floatX>(normal.z * std + mean);
    cvt[3] = static_cast<floatX>(normal.w * std + mean);
    cvt.store(dst + id);
}

/**
 * @brief Template launcher for normal random number generation kernel.
 *
 * Launches rng_normal_kernel with 256 threads per block, processing
 * 4 elements per thread for efficient vectorized stores.
 *
 * @tparam floatX Output data type (float or nv_bfloat16).
 * @param[out] dst Destination array to fill.
 * @param count Number of elements (must be multiple of 4).
 * @param mean Mean of the normal distribution.
 * @param std Standard deviation of the normal distribution.
 * @param seed Random seed for reproducibility.
 * @param subsequence Subsequence offset for parallel streams.
 * @param stream CUDA stream for asynchronous execution.
 */
template <typename floatX>
void rng_normal_imp(floatX* dst,
                    std::size_t count,
                    float mean,
                    float std,
                    unsigned long long seed,
                    unsigned long long subsequence,
                    cudaStream_t stream) {
    assert(count % 4 == 0);
    rng_normal_kernel<<<div_ceil(count, static_cast<std::size_t>(4 * 256)), 256, 0, stream>>>(dst,
                                                                                              count,
                                                                                              mean,
                                                                                              std,
                                                                                              seed,
                                                                                              subsequence);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Fills an FP32 array with normally distributed random values.
 *
 * @param[out] dst Destination FP32 array.
 * @param count Number of elements (must be multiple of 4).
 * @param mean Mean of the normal distribution.
 * @param std Standard deviation of the normal distribution.
 * @param seed Random seed for reproducibility.
 * @param subsequence Subsequence offset for parallel streams.
 * @param stream CUDA stream for asynchronous execution.
 */
void fill_normal(float* dst,
                 std::size_t count,
                 float mean,
                 float std,
                 unsigned long long seed,
                 unsigned long long subsequence,
                 cudaStream_t stream) {
    rng_normal_imp<float>(dst, count, mean, std, seed, subsequence, stream);
}

/**
 * @brief Fills a BF16 array with normally distributed random values.
 *
 * Values are generated in FP32, transformed, then converted to BF16.
 *
 * @param[out] dst Destination BF16 array.
 * @param count Number of elements (must be multiple of 4).
 * @param mean Mean of the normal distribution.
 * @param std Standard deviation of the normal distribution.
 * @param seed Random seed for reproducibility.
 * @param subsequence Subsequence offset for parallel streams.
 * @param stream CUDA stream for asynchronous execution.
 */
void fill_normal(nv_bfloat16* dst,
                 std::size_t count,
                 float mean,
                 float std,
                 unsigned long long seed,
                 unsigned long long subsequence,
                 cudaStream_t stream) {
    rng_normal_imp<nv_bfloat16>(dst, count, mean, std, seed, subsequence, stream);
}
