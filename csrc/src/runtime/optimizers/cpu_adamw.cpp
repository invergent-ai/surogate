// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// CPU-side FP32 AdamW optimizer implementation.
// OpenMP-parallelized for multi-core throughput.

#include "runtime/optimizers/cpu_adamw.h"

#include <cmath>
#include <cstring>
#include <unordered_set>
#include <cuda_bf16.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace optimizers {

void cpu_adamw_step(
    float* __restrict__ p,
    const float* __restrict__ g,
    float* __restrict__ m,
    float* __restrict__ v,
    std::size_t n,
    float lr,
    float beta1,
    float beta2,
    int step,
    float eps,
    float weight_decay,
    float grad_scale)
{
    const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));
    const float wd_factor = 1.0f - lr * weight_decay;

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        const float gi = g[i] * grad_scale;
        m[i] = beta1 * m[i] + (1.0f - beta1) * gi;
        v[i] = beta2 * v[i] + (1.0f - beta2) * gi * gi;
        const float m_hat = m[i] / bc1;
        const float v_hat = v[i] / bc2;
        p[i] = p[i] * wd_factor - lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

void cpu_adamw_step_bf16(
    float* __restrict__ p,
    const void* grad_bf16,
    float* __restrict__ m,
    float* __restrict__ v,
    std::size_t n,
    float lr,
    float beta1,
    float beta2,
    int step,
    float eps,
    float weight_decay,
    float grad_scale)
{
    const auto* g = reinterpret_cast<const nv_bfloat16*>(grad_bf16);
    const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));
    const float wd_factor = 1.0f - lr * weight_decay;

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        const float gi = static_cast<float>(g[i]) * grad_scale;
        m[i] = beta1 * m[i] + (1.0f - beta1) * gi;
        v[i] = beta2 * v[i] + (1.0f - beta2) * gi * gi;
        const float m_hat = m[i] / bc1;
        const float v_hat = v[i] / bc2;
        p[i] = p[i] * wd_factor - lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

void cpu_adamw_step_bf16_param(
    void* param_bf16,
    const void* grad_bf16,
    float* __restrict__ m,
    float* __restrict__ v,
    std::size_t n,
    float lr,
    float beta1,
    float beta2,
    int step,
    float eps,
    float weight_decay,
    float grad_scale)
{
    auto* p = reinterpret_cast<nv_bfloat16*>(param_bf16);
    const auto* g = reinterpret_cast<const nv_bfloat16*>(grad_bf16);
    const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));
    const float wd_factor = 1.0f - lr * weight_decay;

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        float pi = static_cast<float>(p[i]);
        const float gi = static_cast<float>(g[i]) * grad_scale;
        m[i] = beta1 * m[i] + (1.0f - beta1) * gi;
        v[i] = beta2 * v[i] + (1.0f - beta2) * gi * gi;
        const float m_hat = m[i] / bc1;
        const float v_hat = v[i] / bc2;
        pi = pi * wd_factor - lr * m_hat / (std::sqrt(v_hat) + eps);
        p[i] = static_cast<nv_bfloat16>(pi);
    }
}

double cpu_gradient_norm_squared(
    const std::unordered_map<std::string, Tensor>& cpu_grads,
    const std::vector<std::string>& param_names)
{
    double total = 0.0;

    // Collect all (data, size, dtype) tuples first so we can iterate in parallel
    struct GradInfo {
        const void* data;
        std::size_t nelem;
        ETensorDType dtype;
    };
    std::vector<GradInfo> infos;
    infos.reserve(param_names.size());

    // Track seen pointers to skip tied weights (embedding == lm_head)
    std::unordered_set<const void*> seen;
    for (const auto& name : param_names) {
        auto it = cpu_grads.find(name);
        if (it == cpu_grads.end() || !it->second.Data) continue;
        if (seen.count(it->second.Data)) continue;
        seen.insert(it->second.Data);
        infos.push_back({it->second.Data, static_cast<std::size_t>(it->second.nelem()), it->second.DType});
    }

    // Per-parameter norm in parallel, reduce sequentially
    for (const auto& info : infos) {
        double param_norm = 0.0;
        if (info.dtype == ETensorDType::FP32) {
            const auto* d = static_cast<const float*>(info.data);
            #pragma omp parallel for reduction(+:param_norm) schedule(static)
            for (std::size_t i = 0; i < info.nelem; ++i) {
                param_norm += static_cast<double>(d[i]) * static_cast<double>(d[i]);
            }
        } else if (info.dtype == ETensorDType::BF16) {
            const auto* d = reinterpret_cast<const nv_bfloat16*>(info.data);
            #pragma omp parallel for reduction(+:param_norm) schedule(static)
            for (std::size_t i = 0; i < info.nelem; ++i) {
                const double val = static_cast<double>(static_cast<float>(d[i]));
                param_norm += val * val;
            }
        }
        total += param_norm;
    }
    return total;
}

void cpu_vector_add_f32(float* __restrict__ dst, const float* __restrict__ src, std::size_t n) {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
}

void cpu_vector_add_bf16_to_f32(float* __restrict__ dst, const void* src_bf16, std::size_t n) {
    const auto* src = reinterpret_cast<const nv_bfloat16*>(src_bf16);
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] += static_cast<float>(src[i]);
    }
}

} // namespace optimizers
