// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// CPU-side FP32 AdamW optimizer for CPU-RAM centric training.
// No quantization — CPU RAM is abundant, so full FP32 state is used.

#ifndef SUROGATE_SRC_MODULES_OPTIMIZERS_CPU_ADAMW_H
#define SUROGATE_SRC_MODULES_OPTIMIZERS_CPU_ADAMW_H

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "utilities/tensor.h"

namespace optimizers {

struct CPUAdamWState {
    std::vector<float> m;  // FP32 first moment (contiguous across all params)
    std::vector<float> v;  // FP32 second moment
    std::size_t total_params = 0;
    bool initialized = false;
};

/// FP32 parameter update with FP32 gradients.
void cpu_adamw_step(float* param,
                    const float* grad,
                    float* m,
                    float* v,
                    std::size_t n,
                    float lr,
                    float beta1,
                    float beta2,
                    int step,
                    float eps,
                    float weight_decay,
                    float grad_scale);

/// FP32 parameter update with BF16 gradients (promoted to FP32 internally).
void cpu_adamw_step_bf16(float* param,
                         const void* grad_bf16,
                         float* m,
                         float* v,
                         std::size_t n,
                         float lr,
                         float beta1,
                         float beta2,
                         int step,
                         float eps,
                         float weight_decay,
                         float grad_scale);

/// BF16 parameter update with BF16 gradients (reads BF16, computes FP32, writes BF16).
void cpu_adamw_step_bf16_param(void* param_bf16,
                               const void* grad_bf16,
                               float* m,
                               float* v,
                               std::size_t n,
                               float lr,
                               float beta1,
                               float beta2,
                               int step,
                               float eps,
                               float weight_decay,
                               float grad_scale);

/// Compute sum of squared values across all CPU gradients (for gradient norm).
/// Returns sum(g_i^2) for all parameters. Thread-safe via reduction.
double cpu_gradient_norm_squared(const std::unordered_map<std::string, Tensor>& cpu_grads,
                                 const std::vector<std::string>& param_names);

/// CPU-side vector add: dst[i] += src[i] for n elements.
/// Used for micro-step gradient accumulation on CPU.
void cpu_vector_add_f32(float* dst, const float* src, std::size_t n);
void cpu_vector_add_bf16_to_f32(float* dst, const void* src_bf16, std::size_t n);

}  // namespace optimizers

#endif  // SUROGATE_SRC_MODULES_OPTIMIZERS_CPU_ADAMW_H
