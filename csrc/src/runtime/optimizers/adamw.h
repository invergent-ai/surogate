// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Full-precision AdamW optimizer kernel (FP32 state).

#ifndef SUROGATE_SRC_MODULES_OPTIMIZERS_ADAMW_H
#define SUROGATE_SRC_MODULES_OPTIMIZERS_ADAMW_H

#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace optimizers {

void adamw_update(float* param, const float* grad, float* m, float* v, std::size_t n,
                  float lr, float beta1, float beta2, float beta1_correction, float beta2_correction,
                  float epsilon, float weight_decay, const float* grad_scale,
                  cudaStream_t stream);

void adamw_update(float* param, const nv_bfloat16* grad, float* m, float* v, std::size_t n,
                  float lr, float beta1, float beta2, float beta1_correction, float beta2_correction,
                  float epsilon, float weight_decay, const float* grad_scale,
                  cudaStream_t stream);

void adamw_update(nv_bfloat16* param, const nv_bfloat16* grad, float* m, float* v, std::size_t n,
                  float lr, float beta1, float beta2, float beta1_correction, float beta2_correction,
                  float epsilon, float weight_decay, const float* grad_scale,
                  cudaStream_t stream);

void adamw_update(nv_bfloat16* param, const float* grad, float* m, float* v, std::size_t n,
                  float lr, float beta1, float beta2, float beta1_correction, float beta2_correction,
                  float epsilon, float weight_decay, const float* grad_scale,
                  cudaStream_t stream);

void adamw_update(half* param, const half* grad, float* m, float* v, std::size_t n,
                  float lr, float beta1, float beta2, float beta1_correction, float beta2_correction,
                  float epsilon, float weight_decay, const float* grad_scale,
                  cudaStream_t stream);

void adamw_update(half* param, const float* grad, float* m, float* v, std::size_t n,
                  float lr, float beta1, float beta2, float beta1_correction, float beta2_correction,
                  float epsilon, float weight_decay, const float* grad_scale,
                  cudaStream_t stream);

}  // namespace optimizers

#endif  // SUROGATE_SRC_MODULES_OPTIMIZERS_ADAMW_H
