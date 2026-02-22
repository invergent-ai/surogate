// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Full-precision AdamW optimizer kernel (FP32 state).

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "utilities/utils.h"
#include "adamw.h"

namespace optimizers {

namespace {

template <typename T>
__device__ inline float to_float(T v) {
    return static_cast<float>(v);
}

template <>
__device__ inline float to_float<nv_bfloat16>(nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <>
__device__ inline float to_float<half>(half v) {
    return __half2float(v);
}

template <typename T>
__device__ inline T from_float(float v) {
    return static_cast<T>(v);
}

template <>
__device__ inline nv_bfloat16 from_float<nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <>
__device__ inline half from_float<half>(float v) {
    return __float2half_rn(v);
}

template <typename TParam, typename TGrad>
__global__ void adamw_update_kernel(
    TParam* param,
    const TGrad* grad,
    float* m,
    float* v,
    std::size_t n,
    float lr,
    float beta1,
    float beta2,
    float beta1_correction,
    float beta2_correction,
    float epsilon,
    float weight_decay,
    const float* grad_scale)
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = to_float(grad[idx]);
    if (grad_scale) {
        g *= *grad_scale;
    }

    float m_i = beta1 * m[idx] + (1.0f - beta1) * g;
    float v_i = beta2 * v[idx] + (1.0f - beta2) * g * g;
    m[idx] = m_i;
    v[idx] = v_i;

    float m_hat = m_i / beta1_correction;
    float v_hat = v_i / beta2_correction;
    float denom = sqrtf(v_hat) + epsilon;

    float p = to_float(param[idx]);
    if (weight_decay > 0.0f) {
        p *= (1.0f - lr * weight_decay);
    }
    p -= lr * (m_hat / denom);
    param[idx] = from_float<TParam>(p);
}

template <typename TParam, typename TGrad>
void launch_adamw_update(
    TParam* param,
    const TGrad* grad,
    float* m,
    float* v,
    std::size_t n,
    float lr,
    float beta1,
    float beta2,
    float beta1_correction,
    float beta2_correction,
    float epsilon,
    float weight_decay,
    const float* grad_scale,
    cudaStream_t stream)
{
    if (n == 0) return;
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    adamw_update_kernel<<<blocks, threads, 0, stream>>>(
        param, grad, m, v, n, lr, beta1, beta2, beta1_correction, beta2_correction,
        epsilon, weight_decay, grad_scale);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void adamw_update(float* param, const float* grad, float* m, float* v, std::size_t n,
                  float lr, float beta1, float beta2, float beta1_correction, float beta2_correction,
                  float epsilon, float weight_decay, const float* grad_scale,
                  cudaStream_t stream) {
    launch_adamw_update(param, grad, m, v, n, lr, beta1, beta2, beta1_correction, beta2_correction,
                        epsilon, weight_decay, grad_scale, stream);
}

void adamw_update(float* param, const nv_bfloat16* grad, float* m, float* v, std::size_t n,
                  float lr, float beta1, float beta2, float beta1_correction, float beta2_correction,
                  float epsilon, float weight_decay, const float* grad_scale,
                  cudaStream_t stream) {
    launch_adamw_update(param, grad, m, v, n, lr, beta1, beta2, beta1_correction, beta2_correction,
                        epsilon, weight_decay, grad_scale, stream);
}

void adamw_update(nv_bfloat16* param, const nv_bfloat16* grad, float* m, float* v, std::size_t n,
                  float lr, float beta1, float beta2, float beta1_correction, float beta2_correction,
                  float epsilon, float weight_decay, const float* grad_scale,
                  cudaStream_t stream) {
    launch_adamw_update(param, grad, m, v, n, lr, beta1, beta2, beta1_correction, beta2_correction,
                        epsilon, weight_decay, grad_scale, stream);
}

void adamw_update(nv_bfloat16* param, const float* grad, float* m, float* v, std::size_t n,
                  float lr, float beta1, float beta2, float beta1_correction, float beta2_correction,
                  float epsilon, float weight_decay, const float* grad_scale,
                  cudaStream_t stream) {
    launch_adamw_update(param, grad, m, v, n, lr, beta1, beta2, beta1_correction, beta2_correction,
                        epsilon, weight_decay, grad_scale, stream);
}

void adamw_update(half* param, const half* grad, float* m, float* v, std::size_t n,
                  float lr, float beta1, float beta2, float beta1_correction, float beta2_correction,
                  float epsilon, float weight_decay, const float* grad_scale,
                  cudaStream_t stream) {
    launch_adamw_update(param, grad, m, v, n, lr, beta1, beta2, beta1_correction, beta2_correction,
                        epsilon, weight_decay, grad_scale, stream);
}

void adamw_update(half* param, const float* grad, float* m, float* v, std::size_t n,
                  float lr, float beta1, float beta2, float beta1_correction, float beta2_correction,
                  float epsilon, float weight_decay, const float* grad_scale,
                  cudaStream_t stream) {
    launch_adamw_update(param, grad, m, v, n, lr, beta1, beta2, beta1_correction, beta2_correction,
                        epsilon, weight_decay, grad_scale, stream);
}

}  // namespace optimizers
