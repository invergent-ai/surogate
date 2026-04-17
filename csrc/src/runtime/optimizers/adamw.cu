// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Full-precision AdamW optimizer kernel (FP32 state).
// Grid-stride loop with multi-element-per-thread processing.
// Supports CUDA graph capture via device-side opt_params/opt_step.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "utilities/utils.h"
#include "adamw.h"

namespace optimizers {

// ----------------------------------------------------------------------------
// Constants

constexpr int ADAMW_BLOCK_THREADS = 256;
constexpr int ADAMW_N_PER_THREAD = 4;
constexpr int ADAMW_BLOCK_SIZE_N = ADAMW_BLOCK_THREADS * ADAMW_N_PER_THREAD;  // 1024 elements per block

// ----------------------------------------------------------------------------
// Device helpers

template <typename T>
__device__ __forceinline__ float to_float(T v) {
    return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <>
__device__ __forceinline__ float to_float<half>(half v) {
    return __half2float(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v) {
    return static_cast<T>(v);
}

template <>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <>
__device__ __forceinline__ half from_float<half>(float v) {
    return __float2half_rn(v);
}

// ----------------------------------------------------------------------------
// Single-tensor kernel

template <typename TParam, typename TGrad>
__launch_bounds__(ADAMW_BLOCK_THREADS, 4) __global__ void kAdamWKernel(TParam* p,
                                                                       const TGrad* __restrict__ g,
                                                                       float* m,
                                                                       float* v,
                                                                       const float beta1,
                                                                       const float beta2,
                                                                       const float eps,
                                                                       const int step,
                                                                       const float lr,
                                                                       float weight_decay,
                                                                       const float* __restrict__ opt_params,
                                                                       const int* __restrict__ opt_step,
                                                                       const float* __restrict__ gnorm_scale_ptr,
                                                                       const int n) {
    // Read hyperparams from device memory if available (CUDA graph capture)
    const float gnorm_scale = gnorm_scale_ptr ? *gnorm_scale_ptr : 1.0f;
    float beta1_val = beta1;
    float beta2_val = beta2;
    float eps_val = eps;
    int step_val = step;
    float lr_val = lr;
    float weight_decay_val = weight_decay;
    if (opt_params) {
        lr_val = opt_params[0];
        beta1_val = opt_params[1];
        beta2_val = opt_params[2];
        eps_val = opt_params[3];
        weight_decay_val = opt_params[4] * weight_decay;
        if (opt_step) {
            step_val = opt_step[0];
        }
    }

    // Bias correction factors computed in-kernel
    const float correction1 = 1.0f - __powf(beta1_val, step_val);
    const float correction2 = 1.0f - __powf(beta2_val, step_val);

    const int total_blocks = (n + ADAMW_BLOCK_SIZE_N - 1) / ADAMW_BLOCK_SIZE_N;

    // Grid-stride loop over blocks of ADAMW_BLOCK_SIZE_N elements
    for (int block_idx = blockIdx.x; block_idx < total_blocks; block_idx += gridDim.x) {
        const int base_idx = block_idx * ADAMW_BLOCK_SIZE_N;
        const int thread_offset = base_idx + threadIdx.x * ADAMW_N_PER_THREAD;

#pragma unroll
        for (int j = 0; j < ADAMW_N_PER_THREAD; j++) {
            const int idx = thread_offset + j;
            if (idx >= n) continue;

            float g_val = to_float(g[idx]);

            if (!isnan(g_val) && !isinf(g_val)) {
                g_val *= gnorm_scale;

                // Update moments
                float m_i = beta1_val * m[idx] + (1.0f - beta1_val) * g_val;
                float v_i = beta2_val * v[idx] + (1.0f - beta2_val) * g_val * g_val;
                m[idx] = m_i;
                v[idx] = v_i;

                // Bias-corrected estimates
                float m_hat = m_i / correction1;
                float v_hat = v_i / correction2;

                // Adam update
                float param = to_float(p[idx]);
                param -= lr_val * m_hat / (sqrtf(v_hat) + eps_val);

                // Decoupled weight decay
                if (weight_decay_val > 0.0f) {
                    param *= (1.0f - lr_val * weight_decay_val);
                }

                p[idx] = from_float<TParam>(param);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Single-tensor launch

template <typename TParam, typename TGrad>
static void launch_adamw_update(TParam* param,
                                const TGrad* grad,
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
                                cudaStream_t stream) {
    if (n == 0) return;

    int total_blocks = (int)div_ceil(n, (size_t)ADAMW_BLOCK_SIZE_N);
    int grid_size = std::min(total_blocks, 1024);

    kAdamWKernel<TParam, TGrad><<<grid_size, ADAMW_BLOCK_THREADS, 0, stream>>>(param,
                                                                               grad,
                                                                               m,
                                                                               v,
                                                                               beta1,
                                                                               beta2,
                                                                               epsilon,
                                                                               step,
                                                                               lr,
                                                                               weight_decay,
                                                                               opt_params,
                                                                               opt_step,
                                                                               gnorm_scale,
                                                                               (int)n);
    CUDA_CHECK(cudaGetLastError());
}

// Explicit overloads

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
                  cudaStream_t stream) {
    launch_adamw_update(param,
                        grad,
                        m,
                        v,
                        n,
                        lr,
                        beta1,
                        beta2,
                        step,
                        epsilon,
                        weight_decay,
                        gnorm_scale,
                        opt_params,
                        opt_step,
                        stream);
}

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
                  cudaStream_t stream) {
    launch_adamw_update(param,
                        grad,
                        m,
                        v,
                        n,
                        lr,
                        beta1,
                        beta2,
                        step,
                        epsilon,
                        weight_decay,
                        gnorm_scale,
                        opt_params,
                        opt_step,
                        stream);
}

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
                  cudaStream_t stream) {
    launch_adamw_update(param,
                        grad,
                        m,
                        v,
                        n,
                        lr,
                        beta1,
                        beta2,
                        step,
                        epsilon,
                        weight_decay,
                        gnorm_scale,
                        opt_params,
                        opt_step,
                        stream);
}

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
                  cudaStream_t stream) {
    launch_adamw_update(param,
                        grad,
                        m,
                        v,
                        n,
                        lr,
                        beta1,
                        beta2,
                        step,
                        epsilon,
                        weight_decay,
                        gnorm_scale,
                        opt_params,
                        opt_step,
                        stream);
}

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
                  cudaStream_t stream) {
    launch_adamw_update(param,
                        grad,
                        m,
                        v,
                        n,
                        lr,
                        beta1,
                        beta2,
                        step,
                        epsilon,
                        weight_decay,
                        gnorm_scale,
                        opt_params,
                        opt_step,
                        stream);
}

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
                  cudaStream_t stream) {
    launch_adamw_update(param,
                        grad,
                        m,
                        v,
                        n,
                        lr,
                        beta1,
                        beta2,
                        step,
                        epsilon,
                        weight_decay,
                        gnorm_scale,
                        opt_params,
                        opt_step,
                        stream);
}

// ----------------------------------------------------------------------------
// Multi-tensor kernel (for LoRA with many small tensors)
//
// Processes multiple parameter tensors in a single kernel launch.
// state_offsets[i] gives the element offset into the contiguous m/v buffers.

template <typename T>
__launch_bounds__(ADAMW_BLOCK_THREADS, 4) __global__
    void kAdamWMultiTensorKernel(T** params,
                                 T** __restrict__ const grads,
                                 const int* sizes,
                                 int num_tensors,
                                 float* m_buf,
                                 float* v_buf,
                                 const int* state_offsets,
                                 const float beta1,
                                 const float beta2,
                                 const float eps,
                                 const int step,
                                 const float lr,
                                 float weight_decay,
                                 const float* __restrict__ opt_params,
                                 const int* __restrict__ opt_step,
                                 const float* __restrict__ gnorm_scale_ptr) {
    const float gnorm_scale = gnorm_scale_ptr ? *gnorm_scale_ptr : 1.0f;
    float beta1_val = beta1;
    float beta2_val = beta2;
    float eps_val = eps;
    int step_val = step;
    float lr_val = lr;
    float weight_decay_val = weight_decay;
    if (opt_params) {
        lr_val = opt_params[0];
        beta1_val = opt_params[1];
        beta2_val = opt_params[2];
        eps_val = opt_params[3];
        weight_decay_val = opt_params[4] * weight_decay;
        if (opt_step) {
            step_val = opt_step[0];
        }
    }

    const float correction1 = 1.0f - __powf(beta1_val, step_val);
    const float correction2 = 1.0f - __powf(beta2_val, step_val);

    // Process each tensor
    for (int tensor_idx = 0; tensor_idx < num_tensors; tensor_idx++) {
        T* p = params[tensor_idx];
        const T* g = grads[tensor_idx];
        const int n = sizes[tensor_idx];
        const int s_offset = state_offsets[tensor_idx];

        float* m = m_buf + s_offset;
        float* v = v_buf + s_offset;

        const int total_blocks = (n + ADAMW_BLOCK_SIZE_N - 1) / ADAMW_BLOCK_SIZE_N;

        for (int block_idx = blockIdx.x; block_idx < total_blocks; block_idx += gridDim.x) {
            const int base_idx = block_idx * ADAMW_BLOCK_SIZE_N;
            const int thread_offset = base_idx + threadIdx.x * ADAMW_N_PER_THREAD;

#pragma unroll
            for (int j = 0; j < ADAMW_N_PER_THREAD; j++) {
                const int idx = thread_offset + j;
                if (idx >= n) continue;

                float g_val = to_float(g[idx]);

                if (!isnan(g_val) && !isinf(g_val)) {
                    g_val *= gnorm_scale;

                    float m_i = beta1_val * m[idx] + (1.0f - beta1_val) * g_val;
                    float v_i = beta2_val * v[idx] + (1.0f - beta2_val) * g_val * g_val;
                    m[idx] = m_i;
                    v[idx] = v_i;

                    float m_hat = m_i / correction1;
                    float v_hat = v_i / correction2;

                    float param = to_float(p[idx]);
                    param -= lr_val * m_hat / (sqrtf(v_hat) + eps_val);

                    if (weight_decay_val > 0.0f) {
                        param *= (1.0f - lr_val * weight_decay_val);
                    }

                    p[idx] = from_float<T>(param);
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Multi-tensor launch

template <typename T>
static void launch_adamw_update_multi_tensor(T** params,
                                             T** grads,
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
                                             cudaStream_t stream) {
    if (num_tensors == 0 || total_params == 0) return;

    int total_blocks = (int)div_ceil(total_params, (size_t)ADAMW_BLOCK_SIZE_N);
    int grid_size = std::min(total_blocks, 256);

    kAdamWMultiTensorKernel<T><<<grid_size, ADAMW_BLOCK_THREADS, 0, stream>>>(params,
                                                                              const_cast<T**>(grads),
                                                                              sizes,
                                                                              num_tensors,
                                                                              m,
                                                                              v,
                                                                              state_offsets,
                                                                              beta1,
                                                                              beta2,
                                                                              eps,
                                                                              step,
                                                                              lr,
                                                                              weight_decay,
                                                                              opt_params,
                                                                              opt_step,
                                                                              gnorm_scale);
    CUDA_CHECK(cudaGetLastError());
}

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
                               cudaStream_t stream) {
    launch_adamw_update_multi_tensor(params,
                                     grads,
                                     sizes,
                                     num_tensors,
                                     m,
                                     v,
                                     state_offsets,
                                     total_params,
                                     lr,
                                     beta1,
                                     beta2,
                                     step,
                                     eps,
                                     weight_decay,
                                     gnorm_scale,
                                     opt_params,
                                     opt_step,
                                     stream);
}

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
                               cudaStream_t stream) {
    launch_adamw_update_multi_tensor(params,
                                     grads,
                                     sizes,
                                     num_tensors,
                                     m,
                                     v,
                                     state_offsets,
                                     total_params,
                                     lr,
                                     beta1,
                                     beta2,
                                     step,
                                     eps,
                                     weight_decay,
                                     gnorm_scale,
                                     opt_params,
                                     opt_step,
                                     stream);
}

// ----------------------------------------------------------------------------
// State initialization kernel

__global__ void kInitAdamWState(float* m, float* v, size_t n) {
    const size_t stride = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
    for (size_t idx =
             static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
         idx < n;
         idx += stride) {
        m[idx] = 0.0f;
        v[idx] = 0.0f;
    }
}

void init_adamw_state(float* m, float* v, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    int threads = 256;
    const size_t blocks_ideal = div_ceil(n, (size_t)threads);
    const int blocks = static_cast<int>(std::min(blocks_ideal, (size_t)65535));

    kInitAdamWState<<<blocks, threads, 0, stream>>>(m, v, n);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace optimizers
