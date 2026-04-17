// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Full-precision AdamW optimizer kernel (FP32 state).
// Supports single-tensor and multi-tensor (LoRA) updates,
// CUDA graph capture via device-side hyperparameters.

#ifndef SUROGATE_SRC_MODULES_OPTIMIZERS_ADAMW_H
#define SUROGATE_SRC_MODULES_OPTIMIZERS_ADAMW_H

#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace optimizers {

// Number of optimizer hyperparams stored on device for graph capture
// Layout: [lr, beta1, beta2, eps, weight_decay_scale]
constexpr int ADAMW_FP32_GRAPH_PARAM_COUNT = 5;

// ----------------------------------------------------------------------------
// State initialization
// ----------------------------------------------------------------------------

/**
 * @brief Initializes full-precision AdamW state (FP32 momentum and variance to zero).
 */
void init_adamw_state(float* m, float* v, size_t n, cudaStream_t stream);

// ----------------------------------------------------------------------------
// Single-tensor update functions
// ----------------------------------------------------------------------------

void adamw_update(float* param,
                  const float* grad,
                  float* m,
                  float* v,
                  std::size_t n,
                  float lr,
                  float beta1,
                  float beta2,
                  int step,
                  float epsilon,
                  float weight_decay,
                  const float* gnorm_scale,
                  const float* opt_params,
                  const int* opt_step,
                  cudaStream_t stream);

void adamw_update(float* param,
                  const nv_bfloat16* grad,
                  float* m,
                  float* v,
                  std::size_t n,
                  float lr,
                  float beta1,
                  float beta2,
                  int step,
                  float epsilon,
                  float weight_decay,
                  const float* gnorm_scale,
                  const float* opt_params,
                  const int* opt_step,
                  cudaStream_t stream);

void adamw_update(nv_bfloat16* param,
                  const nv_bfloat16* grad,
                  float* m,
                  float* v,
                  std::size_t n,
                  float lr,
                  float beta1,
                  float beta2,
                  int step,
                  float epsilon,
                  float weight_decay,
                  const float* gnorm_scale,
                  const float* opt_params,
                  const int* opt_step,
                  cudaStream_t stream);

void adamw_update(nv_bfloat16* param,
                  const float* grad,
                  float* m,
                  float* v,
                  std::size_t n,
                  float lr,
                  float beta1,
                  float beta2,
                  int step,
                  float epsilon,
                  float weight_decay,
                  const float* gnorm_scale,
                  const float* opt_params,
                  const int* opt_step,
                  cudaStream_t stream);

void adamw_update(half* param,
                  const half* grad,
                  float* m,
                  float* v,
                  std::size_t n,
                  float lr,
                  float beta1,
                  float beta2,
                  int step,
                  float epsilon,
                  float weight_decay,
                  const float* gnorm_scale,
                  const float* opt_params,
                  const int* opt_step,
                  cudaStream_t stream);

void adamw_update(half* param,
                  const float* grad,
                  float* m,
                  float* v,
                  std::size_t n,
                  float lr,
                  float beta1,
                  float beta2,
                  int step,
                  float epsilon,
                  float weight_decay,
                  const float* gnorm_scale,
                  const float* opt_params,
                  const int* opt_step,
                  cudaStream_t stream);

// ----------------------------------------------------------------------------
// Multi-tensor update functions (for LoRA with many small tensors)
// ----------------------------------------------------------------------------

void adamw_update_multi_tensor(float** params,
                               float** grads,
                               const int* sizes,
                               int num_tensors,
                               float* m,
                               float* v,
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

void adamw_update_multi_tensor(nv_bfloat16** params,
                               nv_bfloat16** grads,
                               const int* sizes,
                               int num_tensors,
                               float* m,
                               float* v,
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

}  // namespace optimizers

#endif  // SUROGATE_SRC_MODULES_OPTIMIZERS_ADAMW_H
