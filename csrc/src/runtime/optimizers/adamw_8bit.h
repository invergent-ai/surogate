// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// AdamW 8-bit optimizer with block-wise quantization
// Based on bitsandbytes implementation: https://github.com/TimDettmers/bitsandbytes

#ifndef SUROGATE_SRC_MODULES_OPTIMIZERS_ADAMW_8BIT_H
#define SUROGATE_SRC_MODULES_OPTIMIZERS_ADAMW_8BIT_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstddef>

namespace optimizers {

// Block size for 8-bit optimizer (number of elements processed per block)
// This determines the granularity of quantization - each block has its own absmax.
// Match bitsandbytes block-wise optimizer (256 elements per block).
constexpr int ADAMW8BIT_BLOCK_SIZE = 256;
constexpr int ADAMW_GRAPH_PARAM_COUNT = 5;

/**
 * @brief Creates a dynamic quantization map for 8-bit optimizer states.
 *
 * @param[out] code Output array of 256 float values representing the quantization map.
 * @param signed_map If true, creates a signed map for [-1, 1]; otherwise [0, 1].
 */
void create_dynamic_quantization_map(float* code, bool signed_map);

/**
 * @brief Creates the default signed quantization map for first moment (m).
 * @param[out] code Output array of 256 float values.
 */
void create_adamw8bit_quantiles1(float* code);

/**
 * @brief Creates the default unsigned quantization map for second moment (v).
 * @param[out] code Output array of 256 float values.
 */
void create_adamw8bit_quantiles2(float* code);

/**
 * @brief Initializes the 8-bit optimizer state tensors.
 */
void init_adamw8bit_state(
    unsigned char* state1,
    unsigned char* state2,
    float* absmax1,
    float* absmax2,
    size_t n,
    cudaStream_t stream
);

// ----------------------------------------------------------------------------
// Single-tensor update functions
// ----------------------------------------------------------------------------

/**
 * @brief Launch the 8-bit AdamW optimizer kernel for float parameters.
 */
void adamw_update_8bit(
    float* p,
    const float* g,
    unsigned char* state1,
    unsigned char* state2,
    size_t n,
    float lr,
    float beta1,
    float beta2,
    int step,
    float eps,
    float weight_decay,
    const float* gnorm_scale,
    const float* quantiles1,
    const float* quantiles2,
    float* absmax1,
    float* absmax2,
    const float* opt_params,
    const int* opt_step,
    cudaStream_t stream
);

/**
 * @brief Launch the 8-bit AdamW optimizer kernel for BF16 parameters.
 */
void adamw_update_8bit(
    nv_bfloat16* p,
    const nv_bfloat16* g,
    unsigned char* state1,
    unsigned char* state2,
    size_t n,
    float lr,
    float beta1,
    float beta2,
    int step,
    float eps,
    float weight_decay,
    const float* gnorm_scale,
    const float* quantiles1,
    const float* quantiles2,
    float* absmax1,
    float* absmax2,
    const float* opt_params,
    const int* opt_step,
    cudaStream_t stream
);

/**
 * @brief Launch the 8-bit AdamW optimizer kernel for mixed precision (FP32 params, BF16 grads).
 */
void adamw_update_8bit(
    float* p,
    const nv_bfloat16* g,
    unsigned char* state1,
    unsigned char* state2,
    size_t n,
    float lr,
    float beta1,
    float beta2,
    int step,
    float eps,
    float weight_decay,
    const float* gnorm_scale,
    const float* quantiles1,
    const float* quantiles2,
    float* absmax1,
    float* absmax2,
    const float* opt_params,
    const int* opt_step,
    cudaStream_t stream
);

/**
 * @brief Launch the 8-bit AdamW optimizer kernel for FP16 parameters.
 */
void adamw_update_8bit(
    half* p,
    const half* g,
    unsigned char* state1,
    unsigned char* state2,
    size_t n,
    float lr,
    float beta1,
    float beta2,
    int step,
    float eps,
    float weight_decay,
    const float* gnorm_scale,
    const float* quantiles1,
    const float* quantiles2,
    float* absmax1,
    float* absmax2,
    const float* opt_params,
    const int* opt_step,
    cudaStream_t stream
);

// ----------------------------------------------------------------------------
// Multi-tensor update functions (for LoRA with many small tensors)
// ----------------------------------------------------------------------------

/**
 * @brief Launch multi-tensor 8-bit AdamW optimizer for float parameters.
 */
void adamw_update_8bit_multi_tensor(
    float** params,
    float** grads,
    const int* sizes,
    int num_tensors,
    unsigned char* state1,
    unsigned char* state2,
    float* absmax1,
    float* absmax2,
    const int* state_offsets,
    size_t total_params,
    float lr,
    float beta1,
    float beta2,
    int step,
    float eps,
    float weight_decay,
    const float* gnorm_scale,
    const float* quantiles1,
    const float* quantiles2,
    const float* opt_params,
    const int* opt_step,
    cudaStream_t stream
);

/**
 * @brief Launch multi-tensor 8-bit AdamW optimizer for BF16 parameters.
 */
void adamw_update_8bit_multi_tensor(
    nv_bfloat16** params,
    nv_bfloat16** grads,
    const int* sizes,
    int num_tensors,
    unsigned char* state1,
    unsigned char* state2,
    float* absmax1,
    float* absmax2,
    const int* state_offsets,
    size_t total_params,
    float lr,
    float beta1,
    float beta2,
    int step,
    float eps,
    float weight_decay,
    const float* gnorm_scale,
    const float* quantiles1,
    const float* quantiles2,
    const float* opt_params,
    const int* opt_step,
    cudaStream_t stream
);

} // namespace optimizers

#endif // SUROGATE_SRC_MODULES_OPTIMIZERS_ADAMW_8BIT_H
