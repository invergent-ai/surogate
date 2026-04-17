// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Flash AdamW 8-bit optimizer with softsign/sqrt quantization
// Based on Databricks FlashOptim: https://github.com/databricks/flashoptim
//
// Key differences from adamw_8bit (bitsandbytes-style):
// - Uses softsign transform for momentum (better distribution utilization)
// - Uses sqrt transform for variance (compresses dynamic range)
// - Smaller group size (32) with FP16 scales (vs 256+ blocks with FP32 absmax)
// - Simple linear quantization (no quantile maps or binary search needed)

#ifndef SUROGATE_SRC_MODULES_OPTIMIZERS_FLASH_ADAMW_8BIT_H
#define SUROGATE_SRC_MODULES_OPTIMIZERS_FLASH_ADAMW_8BIT_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstddef>

namespace optimizers {

// Group size for flash 8-bit optimizer quantization.
// Each group of elements shares a single FP16 scale factor.
// Smaller groups = better accuracy, slightly more scale overhead.
constexpr int FLASH_ADAMW8BIT_GROUP_SIZE = 32;

// Number of optimizer hyperparams stored on device for graph capture
constexpr int FLASH_ADAMW_GRAPH_PARAM_COUNT = 5;

/**
 * @brief Initializes the flash 8-bit optimizer state tensors.
 *
 * Sets state1 (first moment) to 0 (int8 zero) and state2 (second moment)
 * to 0 (uint8 zero). Initializes FP16 scales to small positive values.
 */
void init_flash_adamw8bit_state(signed char* state1,    // int8 for signed momentum
                                unsigned char* state2,  // uint8 for unsigned variance
                                half* scales1,          // FP16 per-group scales for momentum
                                half* scales2,          // FP16 per-group scales for variance
                                size_t n,
                                cudaStream_t stream);

// ----------------------------------------------------------------------------
// Single-tensor update functions
// ----------------------------------------------------------------------------

void flash_adamw_update_8bit(float* p,
                             const float* g,
                             signed char* state1,
                             unsigned char* state2,
                             half* scales1,
                             half* scales2,
                             size_t n,
                             float lr,
                             float beta1,
                             float beta2,
                             int step,
                             float eps,
                             float weight_decay,
                             const float* gnorm_scale,
                             const float* opt_params,
                             const int* opt_step,
                             cudaStream_t stream);

void flash_adamw_update_8bit(nv_bfloat16* p,
                             const nv_bfloat16* g,
                             signed char* state1,
                             unsigned char* state2,
                             half* scales1,
                             half* scales2,
                             size_t n,
                             float lr,
                             float beta1,
                             float beta2,
                             int step,
                             float eps,
                             float weight_decay,
                             const float* gnorm_scale,
                             const float* opt_params,
                             const int* opt_step,
                             cudaStream_t stream);

void flash_adamw_update_8bit(half* p,
                             const half* g,
                             signed char* state1,
                             unsigned char* state2,
                             half* scales1,
                             half* scales2,
                             size_t n,
                             float lr,
                             float beta1,
                             float beta2,
                             int step,
                             float eps,
                             float weight_decay,
                             const float* gnorm_scale,
                             const float* opt_params,
                             const int* opt_step,
                             cudaStream_t stream);

// ----------------------------------------------------------------------------
// Multi-tensor update functions (for LoRA with many small tensors)
// ----------------------------------------------------------------------------

void flash_adamw_update_8bit_multi_tensor(float** params,
                                          float** grads,
                                          const int* sizes,
                                          int num_tensors,
                                          signed char* state1,
                                          unsigned char* state2,
                                          half* scales1,
                                          half* scales2,
                                          const int* state_offsets,
                                          size_t total_params,
                                          float lr,
                                          float beta1,
                                          float beta2,
                                          int step,
                                          float eps,
                                          float weight_decay,
                                          const float* gnorm_scale,
                                          const float* opt_params,
                                          const int* opt_step,
                                          cudaStream_t stream);

void flash_adamw_update_8bit_multi_tensor(nv_bfloat16** params,
                                          nv_bfloat16** grads,
                                          const int* sizes,
                                          int num_tensors,
                                          signed char* state1,
                                          unsigned char* state2,
                                          half* scales1,
                                          half* scales2,
                                          const int* state_offsets,
                                          size_t total_params,
                                          float lr,
                                          float beta1,
                                          float beta2,
                                          int step,
                                          float eps,
                                          float weight_decay,
                                          const float* gnorm_scale,
                                          const float* opt_params,
                                          const int* opt_step,
                                          cudaStream_t stream);

/**
 * @brief Returns the number of scale elements needed for a given parameter count.
 */
inline size_t flash_adamw8bit_num_scales(size_t n) {
    return (n + FLASH_ADAMW8BIT_GROUP_SIZE - 1) / FLASH_ADAMW8BIT_GROUP_SIZE;
}

}  // namespace optimizers

#endif  // SUROGATE_SRC_MODULES_OPTIMIZERS_FLASH_ADAMW_8BIT_H
