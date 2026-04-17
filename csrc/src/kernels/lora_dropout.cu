// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file lora_dropout.cu
 * @brief LoRA dropout kernel with inverted dropout scaling.
 *
 * Applies dropout to LoRA intermediate activations (after A @ input, before B @ intermediate).
 * Uses Squirrel Noise 5 for deterministic per-element randomness, allowing the same
 * dropout mask to be regenerated in both forward and backward passes.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "squirrel_noise.cuh"
#include "utilities/dtype.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

/**
 * @brief CUDA kernel for in-place LoRA dropout with inverted scaling.
 *
 * Applies inverted dropout: each element is either zeroed (with probability dropout_prob)
 * or scaled by 1/(1-dropout_prob) to maintain expected values during training.
 *
 * Uses Squirrel Noise 5 for deterministic dropout masks - the same seed produces
 * identical masks, allowing forward and backward passes to use the same dropout pattern.
 *
 * @tparam T Data type (float, nv_bfloat16, or half)
 * @param[in,out] data Array to apply dropout to (modified in place)
 * @param[in] dropout_prob Probability of dropping each element (0.0 to 1.0)
 * @param[in] scale Inverted dropout scale factor (= 1.0 / (1.0 - dropout_prob))
 * @param[in] seed Random seed for deterministic dropout mask generation
 * @param[in] nelem Total number of elements
 */
template <typename T>
__global__ void
lora_dropout_scale_kernel(T* __restrict__ data, float dropout_prob, float scale, unsigned int seed, long nelem) {
    using vec_t = GenericVector<T, 16 / sizeof(T)>;
    long idx = (blockIdx.x * blockDim.x + threadIdx.x) * vec_t::size;

    if (idx + vec_t::size <= nelem) {
        // Vectorized path: full vector fits within bounds
        vec_t data_vec = vec_t::load(data + idx);
        vec_t result;
        for (int j = 0; j < vec_t::size; ++j) {
            // Deterministic per-element dropout using Squirrel Noise 5
            unsigned int random = squirrel_noise_5(static_cast<unsigned int>(idx + j), seed);
            // Use upper 24 bits for better distribution
            float threshold = static_cast<float>(random >> 8) / static_cast<float>(0xFFFFFF);

            if (threshold < dropout_prob) {
                result[j] = static_cast<T>(0);  // Dropped
            } else {
                result[j] = static_cast<T>(static_cast<float>(data_vec[j]) * scale);  // Inverted dropout
            }
        }
        result.store(data + idx);
    } else if (idx < nelem) {
        // Scalar tail path: handle remaining elements one by one
        for (long j = idx; j < nelem; ++j) {
            unsigned int random = squirrel_noise_5(static_cast<unsigned int>(j), seed);
            float threshold = static_cast<float>(random >> 8) / static_cast<float>(0xFFFFFF);
            float val = static_cast<float>(data[j]);
            data[j] = (threshold < dropout_prob) ? static_cast<T>(0) : static_cast<T>(val * scale);
        }
    }
}

/**
 * @brief Apply in-place dropout scaling to a tensor.
 *
 * Applies inverted dropout to LoRA intermediate activations. If dropout_prob <= 0,
 * this is a no-op. The same seed will produce identical dropout masks, enabling
 * correct gradient computation in the backward pass.
 *
 * @param intermediate Tensor to apply dropout to (modified in place)
 * @param dropout_prob Dropout probability (0.0 = no dropout, 1.0 = drop all)
 * @param seed Random seed for deterministic mask generation
 * @param stream CUDA stream for asynchronous execution
 */
void lora_dropout_scale(Tensor& intermediate, float dropout_prob, unsigned int seed, cudaStream_t stream) {
    // No-op if dropout is disabled
    if (dropout_prob <= 0.0f) {
        return;
    }

    // Clamp dropout probability to valid range
    if (dropout_prob >= 1.0f) {
        // Edge case: drop everything - zero the tensor
        CUDA_CHECK(
            cudaMemsetAsync(intermediate.Data, 0, intermediate.nelem() * get_dtype_size(intermediate.DType), stream));
        return;
    }

    const float scale = 1.0f / (1.0f - dropout_prob);
    const long nelem = intermediate.nelem();

    constexpr int threads_per_block = 512;
    // Account for vectorization: each thread processes 16/sizeof(T) elements
    const int elements_per_thread = 16 / get_dtype_size(intermediate.DType);
    const int num_blocks =
        (nelem + threads_per_block * elements_per_thread - 1) / (threads_per_block * elements_per_thread);

    switch (intermediate.DType) {
        case ETensorDType::FP32:
            lora_dropout_scale_kernel<float>
                <<<num_blocks, threads_per_block, 0, stream>>>(reinterpret_cast<float*>(intermediate.Data),
                                                               dropout_prob,
                                                               scale,
                                                               seed,
                                                               nelem);
            break;
        case ETensorDType::BF16:
            lora_dropout_scale_kernel<nv_bfloat16>
                <<<num_blocks, threads_per_block, 0, stream>>>(reinterpret_cast<nv_bfloat16*>(intermediate.Data),
                                                               dropout_prob,
                                                               scale,
                                                               seed,
                                                               nelem);
            break;
        case ETensorDType::FP16:
            lora_dropout_scale_kernel<half>
                <<<num_blocks, threads_per_block, 0, stream>>>(reinterpret_cast<half*>(intermediate.Data),
                                                               dropout_prob,
                                                               scale,
                                                               seed,
                                                               nelem);
            break;
        default: throw std::runtime_error("lora_dropout_scale: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
}
