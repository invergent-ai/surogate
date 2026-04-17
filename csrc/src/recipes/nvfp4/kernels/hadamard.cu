// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file hadamard.cu
 * @brief CUDA kernels for Random Hadamard Transform (RHT) for FP4 training.
 *
 * The Random Hadamard Transform smooths tensor value distributions before FP4
 * quantization, reducing the impact of outliers on quantization error.
 *
 * Algorithm:
 * 1. Apply random diagonal sign matrix D (from random seed)
 * 2. Apply 16x16 block-wise Hadamard transform H16
 * 3. Scale by 1/4 (= 1/sqrt(16))
 *
 * Result: y = (1/4) * H16 @ D @ x.reshape(-1, 16).T
 *
 * The inverse transform is: x = D^T @ H16^T @ y * 4 = D @ H16 @ y * 4
 * (Since H16 is symmetric and D^T = D for diagonal sign matrices)
 *
 * References:
 * - TransformerEngine: transformer_engine/common/hadamard_transform/
 * - "FP8-LM: Training FP8 Large Language Models" (https://arxiv.org/abs/2310.18313)
 */

#include "kernels/kernel_utils.cuh"
#include "utilities/tensor.h"
#include "utilities/utils.h"

#include <cuda_bf16.h>
#include <curand_kernel.h>

namespace {

// Hadamard transform block size (16x16)
constexpr int HADAMARD_SIZE = 16;
constexpr float HADAMARD_SCALE = 0.25f;  // 1/sqrt(16) = 1/4

/**
 * @brief Generate a random sign (+1 or -1) based on coordinate and seed.
 *
 * Uses a simple hash function to generate deterministic random signs
 * based on the element position and seed. This allows reproducible
 * transforms for the same seed.
 *
 * @param idx Coordinate index along the transformed dimension (0..K-1)
 * @param seed Random seed
 * @return +1.0f or -1.0f
 */
__device__ __forceinline__ float random_sign(long idx, unsigned int seed) {
    // Simple hash combining position and seed
    unsigned int hash = idx ^ seed;
    hash = hash * 2654435761u;  // Knuth's multiplicative hash
    hash ^= (hash >> 16);
    hash *= 0x85ebca6bu;
    hash ^= (hash >> 13);
    // Return +1 or -1 based on LSB
    return (hash & 1) ? 1.0f : -1.0f;
}

/**
 * @brief In-place 16-point Hadamard transform using butterfly operations.
 *
 * Implements the Fast Walsh-Hadamard Transform (FWHT) for 16 elements.
 * Uses log2(16) = 4 stages of butterfly operations.
 *
 * @param[in,out] x Array of 16 float values to transform
 */
__device__ __forceinline__ void hadamard16_inplace(float x[16]) {
// Stage 1: pairs (0,1), (2,3), ... (14,15)
#pragma unroll
    for (int i = 0; i < 16; i += 2) {
        float a = x[i];
        float b = x[i + 1];
        x[i] = a + b;
        x[i + 1] = a - b;
    }

// Stage 2: pairs with stride 2
#pragma unroll
    for (int i = 0; i < 16; i += 4) {
        float a0 = x[i];
        float a1 = x[i + 1];
        float b0 = x[i + 2];
        float b1 = x[i + 3];
        x[i] = a0 + b0;
        x[i + 1] = a1 + b1;
        x[i + 2] = a0 - b0;
        x[i + 3] = a1 - b1;
    }

// Stage 3: pairs with stride 4
#pragma unroll
    for (int i = 0; i < 16; i += 8) {
        float a0 = x[i];
        float a1 = x[i + 1];
        float a2 = x[i + 2];
        float a3 = x[i + 3];
        float b0 = x[i + 4];
        float b1 = x[i + 5];
        float b2 = x[i + 6];
        float b3 = x[i + 7];
        x[i] = a0 + b0;
        x[i + 1] = a1 + b1;
        x[i + 2] = a2 + b2;
        x[i + 3] = a3 + b3;
        x[i + 4] = a0 - b0;
        x[i + 5] = a1 - b1;
        x[i + 6] = a2 - b2;
        x[i + 7] = a3 - b3;
    }

    // Stage 4: pairs with stride 8
    float a[8], b[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        a[i] = x[i];
        b[i] = x[i + 8];
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        x[i] = a[i] + b[i];
        x[i + 8] = a[i] - b[i];
    }
}

/**
 * @brief Random Hadamard Transform forward kernel.
 *
 * Applies RHT to transform tensor distributions before FP4 quantization.
 * Each thread processes one 16-element block.
 *
 * @param[out] out Output BF16 tensor (same shape as input)
 * @param[in] in Input BF16 tensor
 * @param[out] amax_out Optional: track absolute maximum for subsequent quantization
 * @param total_elements Total number of elements (must be multiple of 16)
 * @param seed Random seed for sign matrix generation
 */
__global__ void hadamard_transform_forward_kernel(nv_bfloat16* __restrict__ out,
                                                  const nv_bfloat16* __restrict__ in,
                                                  float* __restrict__ amax_out,
                                                  long total_elements,
                                                  int K,
                                                  unsigned int seed) {
    // Shared memory for block-level amax reduction
    __shared__ float s_block_amax;
    if (threadIdx.x == 0) {
        s_block_amax = 0.0f;
    }
    __syncthreads();

    const long num_blocks = total_elements / HADAMARD_SIZE;
    float thread_amax = 0.0f;

    // Grid-stride loop over 16-element blocks
    for (long block_idx = blockIdx.x * blockDim.x + threadIdx.x; block_idx < num_blocks;
         block_idx += gridDim.x * blockDim.x) {
        const long base_idx = block_idx * HADAMARD_SIZE;
        const long col0 = base_idx % K;
        float values[HADAMARD_SIZE];

// Load and apply random diagonal sign matrix
#pragma unroll
        for (int i = 0; i < HADAMARD_SIZE; ++i) {
            float val = (float)in[base_idx + i];
            float sign = random_sign(col0 + i, seed);
            values[i] = val * sign;
        }

        // Apply Hadamard transform
        hadamard16_inplace(values);

// Scale and store, track amax
#pragma unroll
        for (int i = 0; i < HADAMARD_SIZE; ++i) {
            float scaled = values[i] * HADAMARD_SCALE;
            thread_amax = fmaxf(thread_amax, fabsf(scaled));
            out[base_idx + i] = (nv_bfloat16)scaled;
        }
    }

    // Reduce amax if output pointer provided
    if (amax_out != nullptr) {
        // Warp-level reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            thread_amax = fmaxf(thread_amax, __shfl_xor_sync(0xFFFFFFFF, thread_amax, offset));
        }

        if (threadIdx.x % 32 == 0) {
            atomicMax(reinterpret_cast<unsigned int*>(&s_block_amax), __float_as_uint(thread_amax));
        }
        __syncthreads();

        if (threadIdx.x == 0 && s_block_amax > 0.0f) {
            atomicMax(reinterpret_cast<unsigned int*>(amax_out), __float_as_uint(s_block_amax));
        }
    }
}

/**
 * @brief Random Hadamard Transform inverse kernel.
 *
 * Applies inverse RHT to restore original distribution after FP4 operations.
 * Since H16 is symmetric and D^T = D, the inverse is: x = D @ H16 @ y * 4
 *
 * @param[out] out Output BF16 tensor
 * @param[in] in Input BF16 tensor (post-transform)
 * @param total_elements Total number of elements (must be multiple of 16)
 * @param seed Random seed (must match forward transform)
 */
__global__ void hadamard_transform_inverse_kernel(nv_bfloat16* __restrict__ out,
                                                  const nv_bfloat16* __restrict__ in,
                                                  long total_elements,
                                                  int K,
                                                  unsigned int seed) {
    const long num_blocks = total_elements / HADAMARD_SIZE;

    // Grid-stride loop over 16-element blocks
    for (long block_idx = blockIdx.x * blockDim.x + threadIdx.x; block_idx < num_blocks;
         block_idx += gridDim.x * blockDim.x) {
        const long base_idx = block_idx * HADAMARD_SIZE;
        const long col0 = base_idx % K;
        float values[HADAMARD_SIZE];

// Load values
#pragma unroll
        for (int i = 0; i < HADAMARD_SIZE; ++i) {
            values[i] = (float)in[base_idx + i];
        }

        // Apply Hadamard transform (same as forward since H16 is symmetric)
        hadamard16_inplace(values);

// Apply inverse scaling (multiply by 4) and random sign matrix
// Note: inverse scale is 1/HADAMARD_SCALE = 4, but we need to account
// for the forward scale that was already applied, so total inverse scale
// is HADAMARD_SCALE (the Hadamard transform is self-inverse when properly scaled)
#pragma unroll
        for (int i = 0; i < HADAMARD_SIZE; ++i) {
            float sign = random_sign(col0 + i, seed);
            float scaled = values[i] * HADAMARD_SCALE * sign;
            out[base_idx + i] = (nv_bfloat16)scaled;
        }
    }
}

/**
 * @brief Random Hadamard Transform forward kernel (row-wise).
 *
 * Applies a 16-point RHT down the M dimension for each column.
 * (M must be multiple of 16; K can be any positive value.)
 */
__global__ void hadamard_transform_forward_rows_kernel(nv_bfloat16* __restrict__ out,
                                                       const nv_bfloat16* __restrict__ in,
                                                       float* __restrict__ amax_out,
                                                       int M,
                                                       int K,
                                                       unsigned int seed) {
    __shared__ float s_block_amax;
    if (threadIdx.x == 0) {
        s_block_amax = 0.0f;
    }
    __syncthreads();

    const int row_blocks = M / HADAMARD_SIZE;
    const long num_blocks = (long)row_blocks * (long)K;

    float thread_amax = 0.0f;

    for (long block_idx = blockIdx.x * blockDim.x + threadIdx.x; block_idx < num_blocks;
         block_idx += gridDim.x * blockDim.x) {
        const int col = (int)(block_idx % K);
        const int rb = (int)(block_idx / K);
        const int row0 = rb * HADAMARD_SIZE;

        float values[HADAMARD_SIZE];
#pragma unroll
        for (int i = 0; i < HADAMARD_SIZE; ++i) {
            float val = (float)in[(row0 + i) * (long)K + col];
            float sign = random_sign((long)row0 + i, seed);
            values[i] = val * sign;
        }

        hadamard16_inplace(values);

#pragma unroll
        for (int i = 0; i < HADAMARD_SIZE; ++i) {
            float scaled = values[i] * HADAMARD_SCALE;
            thread_amax = fmaxf(thread_amax, fabsf(scaled));
            out[(row0 + i) * (long)K + col] = (nv_bfloat16)scaled;
        }
    }

    if (amax_out != nullptr) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            thread_amax = fmaxf(thread_amax, __shfl_xor_sync(0xFFFFFFFF, thread_amax, offset));
        }

        if (threadIdx.x % 32 == 0) {
            atomicMax(reinterpret_cast<unsigned int*>(&s_block_amax), __float_as_uint(thread_amax));
        }
        __syncthreads();

        if (threadIdx.x == 0 && s_block_amax > 0.0f) {
            atomicMax(reinterpret_cast<unsigned int*>(amax_out), __float_as_uint(s_block_amax));
        }
    }
}

/**
 * @brief Random Hadamard Transform inverse kernel (row-wise).
 *
 * Inverse of forward_rows: apply H16 then random sign matrix D, with scaling.
 */
__global__ void hadamard_transform_inverse_rows_kernel(nv_bfloat16* __restrict__ out,
                                                       const nv_bfloat16* __restrict__ in,
                                                       int M,
                                                       int K,
                                                       unsigned int seed) {
    const int row_blocks = M / HADAMARD_SIZE;
    const long num_blocks = (long)row_blocks * (long)K;

    for (long block_idx = blockIdx.x * blockDim.x + threadIdx.x; block_idx < num_blocks;
         block_idx += gridDim.x * blockDim.x) {
        const int col = (int)(block_idx % K);
        const int rb = (int)(block_idx / K);
        const int row0 = rb * HADAMARD_SIZE;

        float values[HADAMARD_SIZE];
#pragma unroll
        for (int i = 0; i < HADAMARD_SIZE; ++i) {
            values[i] = (float)in[(row0 + i) * (long)K + col];
        }

        hadamard16_inplace(values);

#pragma unroll
        for (int i = 0; i < HADAMARD_SIZE; ++i) {
            float sign = random_sign((long)row0 + i, seed);
            float scaled = values[i] * HADAMARD_SCALE * sign;
            out[(row0 + i) * (long)K + col] = (nv_bfloat16)scaled;
        }
    }
}

/**
 * @brief FP32 version of Hadamard transform forward kernel.
 */
__global__ void hadamard_transform_forward_f32_kernel(float* __restrict__ out,
                                                      const float* __restrict__ in,
                                                      float* __restrict__ amax_out,
                                                      long total_elements,
                                                      int K,
                                                      unsigned int seed) {
    __shared__ float s_block_amax;
    if (threadIdx.x == 0) {
        s_block_amax = 0.0f;
    }
    __syncthreads();

    const long num_blocks = total_elements / HADAMARD_SIZE;
    float thread_amax = 0.0f;

    for (long block_idx = blockIdx.x * blockDim.x + threadIdx.x; block_idx < num_blocks;
         block_idx += gridDim.x * blockDim.x) {
        const long base_idx = block_idx * HADAMARD_SIZE;
        const long col0 = base_idx % K;
        float values[HADAMARD_SIZE];

#pragma unroll
        for (int i = 0; i < HADAMARD_SIZE; ++i) {
            float val = in[base_idx + i];
            float sign = random_sign(col0 + i, seed);
            values[i] = val * sign;
        }

        hadamard16_inplace(values);

#pragma unroll
        for (int i = 0; i < HADAMARD_SIZE; ++i) {
            float scaled = values[i] * HADAMARD_SCALE;
            thread_amax = fmaxf(thread_amax, fabsf(scaled));
            out[base_idx + i] = scaled;
        }
    }

    if (amax_out != nullptr) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            thread_amax = fmaxf(thread_amax, __shfl_xor_sync(0xFFFFFFFF, thread_amax, offset));
        }

        if (threadIdx.x % 32 == 0) {
            atomicMax(reinterpret_cast<unsigned int*>(&s_block_amax), __float_as_uint(thread_amax));
        }
        __syncthreads();

        if (threadIdx.x == 0 && s_block_amax > 0.0f) {
            atomicMax(reinterpret_cast<unsigned int*>(amax_out), __float_as_uint(s_block_amax));
        }
    }
}

/**
 * @brief FP32 version of Hadamard transform inverse kernel.
 */
__global__ void hadamard_transform_inverse_f32_kernel(float* __restrict__ out,
                                                      const float* __restrict__ in,
                                                      long total_elements,
                                                      int K,
                                                      unsigned int seed) {
    const long num_blocks = total_elements / HADAMARD_SIZE;

    for (long block_idx = blockIdx.x * blockDim.x + threadIdx.x; block_idx < num_blocks;
         block_idx += gridDim.x * blockDim.x) {
        const long base_idx = block_idx * HADAMARD_SIZE;
        const long col0 = base_idx % K;
        float values[HADAMARD_SIZE];

#pragma unroll
        for (int i = 0; i < HADAMARD_SIZE; ++i) {
            values[i] = in[base_idx + i];
        }

        hadamard16_inplace(values);

#pragma unroll
        for (int i = 0; i < HADAMARD_SIZE; ++i) {
            float sign = random_sign(col0 + i, seed);
            float scaled = values[i] * HADAMARD_SCALE * sign;
            out[base_idx + i] = scaled;
        }
    }
}

}  // anonymous namespace

// ============================================================================
// Host Launcher Functions
// ============================================================================

/**
 * @brief Apply Random Hadamard Transform (forward) to BF16 tensor.
 *
 * Transforms tensor distributions before FP4 quantization.
 *
 * @param[out] out Output BF16 tensor (same shape as input)
 * @param[in] in Input BF16 tensor
 * @param[out] amax_out Optional output for tracking absolute maximum (can be nullptr)
 * @param M Number of rows
 * @param K Number of columns (must be multiple of 16)
 * @param seed Random seed for sign matrix generation
 * @param stream CUDA stream
 */
void hadamard_transform_forward(nv_bfloat16* out,
                                const nv_bfloat16* in,
                                float* amax_out,
                                int M,
                                int K,
                                unsigned int seed,
                                cudaStream_t stream) {
    if (K % HADAMARD_SIZE != 0) {
        throw std::runtime_error("hadamard_transform_forward: K must be multiple of 16");
    }

    const long total_elements = (long)M * K;
    const long num_blocks_to_process = total_elements / HADAMARD_SIZE;

    const int threads_per_block = 256;
    const int num_cuda_blocks = std::min((int)div_ceil(num_blocks_to_process, (long)threads_per_block),
                                         2048  // Cap grid size
    );

    if (amax_out != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(amax_out, 0, sizeof(float), stream));
    }

    hadamard_transform_forward_kernel<<<num_cuda_blocks, threads_per_block, 0, stream>>>(out,
                                                                                         in,
                                                                                         amax_out,
                                                                                         total_elements,
                                                                                         K,
                                                                                         seed);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Apply Random Hadamard Transform (inverse) to BF16 tensor.
 *
 * Restores original distribution after FP4 operations.
 *
 * @param[out] out Output BF16 tensor
 * @param[in] in Input BF16 tensor (post-transform)
 * @param M Number of rows
 * @param K Number of columns (must be multiple of 16)
 * @param seed Random seed (must match forward transform)
 * @param stream CUDA stream
 */
void hadamard_transform_inverse(nv_bfloat16* out,
                                const nv_bfloat16* in,
                                int M,
                                int K,
                                unsigned int seed,
                                cudaStream_t stream) {
    if (K % HADAMARD_SIZE != 0) {
        throw std::runtime_error("hadamard_transform_inverse: K must be multiple of 16");
    }

    const long total_elements = (long)M * K;
    const long num_blocks_to_process = total_elements / HADAMARD_SIZE;

    const int threads_per_block = 256;
    const int num_cuda_blocks = std::min((int)div_ceil(num_blocks_to_process, (long)threads_per_block), 2048);

    hadamard_transform_inverse_kernel<<<num_cuda_blocks, threads_per_block, 0, stream>>>(out,
                                                                                         in,
                                                                                         total_elements,
                                                                                         K,
                                                                                         seed);
    CUDA_CHECK(cudaGetLastError());
}

void hadamard_transform_forward_rows(nv_bfloat16* out,
                                     const nv_bfloat16* in,
                                     float* amax_out,
                                     int M,
                                     int K,
                                     unsigned int seed,
                                     cudaStream_t stream) {
    if (M % HADAMARD_SIZE != 0) {
        throw std::runtime_error("hadamard_transform_forward_rows: M must be multiple of 16");
    }

    const long num_blocks_to_process = (long)(M / HADAMARD_SIZE) * (long)K;

    const int threads_per_block = 256;
    const int num_cuda_blocks = std::min((int)div_ceil(num_blocks_to_process, (long)threads_per_block), 2048);

    if (amax_out != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(amax_out, 0, sizeof(float), stream));
    }

    hadamard_transform_forward_rows_kernel<<<num_cuda_blocks, threads_per_block, 0, stream>>>(out,
                                                                                              in,
                                                                                              amax_out,
                                                                                              M,
                                                                                              K,
                                                                                              seed);
    CUDA_CHECK(cudaGetLastError());
}

void hadamard_transform_inverse_rows(nv_bfloat16* out,
                                     const nv_bfloat16* in,
                                     int M,
                                     int K,
                                     unsigned int seed,
                                     cudaStream_t stream) {
    if (M % HADAMARD_SIZE != 0) {
        throw std::runtime_error("hadamard_transform_inverse_rows: M must be multiple of 16");
    }

    const long num_blocks_to_process = (long)(M / HADAMARD_SIZE) * (long)K;

    const int threads_per_block = 256;
    const int num_cuda_blocks = std::min((int)div_ceil(num_blocks_to_process, (long)threads_per_block), 2048);

    hadamard_transform_inverse_rows_kernel<<<num_cuda_blocks, threads_per_block, 0, stream>>>(out, in, M, K, seed);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Apply Random Hadamard Transform (forward) to FP32 tensor.
 */
void hadamard_transform_forward(float* out,
                                const float* in,
                                float* amax_out,
                                int M,
                                int K,
                                unsigned int seed,
                                cudaStream_t stream) {
    if (K % HADAMARD_SIZE != 0) {
        throw std::runtime_error("hadamard_transform_forward: K must be multiple of 16");
    }

    const long total_elements = (long)M * K;
    const long num_blocks_to_process = total_elements / HADAMARD_SIZE;

    const int threads_per_block = 256;
    const int num_cuda_blocks = std::min((int)div_ceil(num_blocks_to_process, (long)threads_per_block), 2048);

    if (amax_out != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(amax_out, 0, sizeof(float), stream));
    }

    hadamard_transform_forward_f32_kernel<<<num_cuda_blocks, threads_per_block, 0, stream>>>(out,
                                                                                             in,
                                                                                             amax_out,
                                                                                             total_elements,
                                                                                             K,
                                                                                             seed);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Apply Random Hadamard Transform (inverse) to FP32 tensor.
 */
void hadamard_transform_inverse(float* out, const float* in, int M, int K, unsigned int seed, cudaStream_t stream) {
    if (K % HADAMARD_SIZE != 0) {
        throw std::runtime_error("hadamard_transform_inverse: K must be multiple of 16");
    }

    const long total_elements = (long)M * K;
    const long num_blocks_to_process = total_elements / HADAMARD_SIZE;

    const int threads_per_block = 256;
    const int num_cuda_blocks = std::min((int)div_ceil(num_blocks_to_process, (long)threads_per_block), 2048);

    hadamard_transform_inverse_f32_kernel<<<num_cuda_blocks, threads_per_block, 0, stream>>>(out,
                                                                                             in,
                                                                                             total_elements,
                                                                                             K,
                                                                                             seed);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Tensor-based Wrapper Functions
// ============================================================================

/**
 * @brief Tensor-based wrapper for Hadamard transform forward.
 */
void hadamard_transform_forward(Tensor& out,
                                const Tensor& in,
                                float* amax_out,
                                int M,
                                int K,
                                unsigned int seed,
                                cudaStream_t stream) {
    if (in.DType == ETensorDType::BF16) {
        hadamard_transform_forward(out.get<nv_bfloat16>(), in.get<nv_bfloat16>(), amax_out, M, K, seed, stream);
    } else if (in.DType == ETensorDType::FP32) {
        hadamard_transform_forward(out.get<float>(), in.get<float>(), amax_out, M, K, seed, stream);
    } else {
        throw std::runtime_error("hadamard_transform_forward: unsupported dtype (must be BF16 or FP32)");
    }
}

/**
 * @brief Tensor-based wrapper for Hadamard transform inverse.
 */
void hadamard_transform_inverse(Tensor& out, const Tensor& in, int M, int K, unsigned int seed, cudaStream_t stream) {
    if (in.DType == ETensorDType::BF16) {
        hadamard_transform_inverse(out.get<nv_bfloat16>(), in.get<nv_bfloat16>(), M, K, seed, stream);
    } else if (in.DType == ETensorDType::FP32) {
        hadamard_transform_inverse(out.get<float>(), in.get<float>(), M, K, seed, stream);
    } else {
        throw std::runtime_error("hadamard_transform_inverse: unsupported dtype (must be BF16 or FP32)");
    }
}
