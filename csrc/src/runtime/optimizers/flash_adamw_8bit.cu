// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Flash AdamW 8-bit optimizer kernel based on Databricks FlashOptim
// Reference: https://github.com/databricks/flashoptim
//
// Quantization scheme:
// - Momentum (m): signed int8 with softsign transform
//   Encode: normalize by absmax -> softsign(x) = 2x/(1+|x|) -> scale to [-127, 127]
//   Decode: scale by 1/127 -> inverse_softsign(y) = y/(2-|y|) -> multiply by scale
//
// - Variance (v): unsigned uint8 with sqrt transform
//   Encode: sqrt(v) -> normalize by absmax -> scale to [0, 255]
//   Decode: scale by 1/255 -> multiply by scale -> square to recover v
//
// Both use per-group (32 elements) FP16 scale factors.

#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "utilities/utils.h"
#include "flash_adamw_8bit.h"

namespace optimizers {

// ----------------------------------------------------------------------------
// Constants

constexpr int FLASH_GROUP_SIZE = FLASH_ADAMW8BIT_GROUP_SIZE;  // 32
constexpr int FLASH_BLOCK_THREADS = 256;
// Elements per thread block = threads * (GROUP_SIZE / warp_size) but we process
// one group per warp, so BLOCK_SIZE_N = FLASH_BLOCK_THREADS elements.
// Each thread block processes BLOCK_SIZE_N elements in a grid-stride loop.
constexpr int FLASH_BLOCK_SIZE_N = FLASH_BLOCK_THREADS * 4;                    // 1024 elements per block iteration
constexpr int FLASH_GROUPS_PER_BLOCK = FLASH_BLOCK_SIZE_N / FLASH_GROUP_SIZE;  // 32 groups

static_assert(FLASH_BLOCK_SIZE_N % FLASH_GROUP_SIZE == 0, "BLOCK_SIZE_N must be a multiple of GROUP_SIZE");
static_assert(FLASH_GROUPS_PER_BLOCK > 0, "Must have at least one group per block");

// Elements per thread
constexpr int FLASH_N_PER_THREAD = FLASH_BLOCK_SIZE_N / FLASH_BLOCK_THREADS;  // 4

// ----------------------------------------------------------------------------
// Device helpers for softsign quantization

// Softsign transform: y = 2x / (1 + |x|)
// Maps [-1, 1] -> [-1, 1] but with better distribution near zero
__device__ __forceinline__ float softsign_transform(float x) {
    return 2.0f * x / (1.0f + fabsf(x));
}

// Inverse softsign: x = y / (2 - |y|)
// Recovers original normalized value from softsign-transformed value
__device__ __forceinline__ float inverse_softsign(float y) {
    return y / (2.0f - fabsf(y));
}

// ----------------------------------------------------------------------------
// Flash AdamW 8-bit kernel
//
// This is significantly simpler than the bitsandbytes-style kernel:
// - No quantile maps (256-entry lookup tables) in shared memory
// - No quadrant pivots or binary search for quantization
// - No sign consistency correction
// - Just: absmax -> normalize -> softsign/linear -> round to int8

template <typename T>
__launch_bounds__(FLASH_BLOCK_THREADS, 3) __global__
    void kFlashAdamW8bitKernel(T* p,
                               T* __restrict__ const g,
                               signed char* state1,    // momentum as int8
                               unsigned char* state2,  // variance as uint8
                               half* scales1,          // FP16 scales for momentum
                               half* scales2,          // FP16 scales for variance
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
    // Read gnorm_scale from device memory (allows CUDA graph capture)
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

    // Bias correction factors
    const float correction1 = 1.0f - __powf(beta1_val, step_val);
    const float correction2 = 1.0f - __powf(beta2_val, step_val);

    // Total blocks needed
    const int total_blocks = (n + FLASH_BLOCK_SIZE_N - 1) / FLASH_BLOCK_SIZE_N;

    // Per-thread storage
    float g_vals[FLASH_N_PER_THREAD];
    float m_vals[FLASH_N_PER_THREAD];
    float v_vals[FLASH_N_PER_THREAD];

    // Shared memory for per-group absmax reduction
    // Each warp handles some groups; we use shared memory for cross-warp reduction
    __shared__ float smem_absmax1[FLASH_GROUPS_PER_BLOCK];
    __shared__ float smem_absmax2[FLASH_GROUPS_PER_BLOCK];

    // Grid-stride loop over blocks of FLASH_BLOCK_SIZE_N elements
    for (int block_idx = blockIdx.x; block_idx < total_blocks; block_idx += gridDim.x) {
        const int base_idx = block_idx * FLASH_BLOCK_SIZE_N;

        // Each thread loads FLASH_N_PER_THREAD consecutive elements
        const int thread_offset = base_idx + threadIdx.x * FLASH_N_PER_THREAD;

// --- Phase 1: Load gradients and dequantize states ---
#pragma unroll
        for (int j = 0; j < FLASH_N_PER_THREAD; j++) {
            const int idx = thread_offset + j;
            if (idx < n) {
                float g_val = float(g[idx]);

                if (!isnan(g_val) && !isinf(g_val)) {
                    g_val *= gnorm_scale;
                    g_vals[j] = g_val;

                    // Dequantize momentum: int8 -> softsign_inv -> scale
                    const int group_idx = idx / FLASH_GROUP_SIZE;
                    float scale1 = __half2float(scales1[group_idx]);
                    float scale2 = __half2float(scales2[group_idx]);

                    float m_quantized = float(state1[idx]) / 127.0f;
                    m_vals[j] = inverse_softsign(m_quantized) * scale1;

                    // Dequantize variance: uint8 -> scale -> square
                    float v_sqrt_quantized = float(state2[idx]) / 255.0f;
                    float v_sqrt = v_sqrt_quantized * scale2;
                    v_vals[j] = v_sqrt * v_sqrt;

                    // Update moments
                    m_vals[j] = beta1_val * m_vals[j] + (1.0f - beta1_val) * g_val;
                    v_vals[j] = beta2_val * v_vals[j] + (1.0f - beta2_val) * g_val * g_val;
                } else {
                    g_vals[j] = 0.0f;
                    m_vals[j] = 0.0f;
                    v_vals[j] = 0.0f;
                }
            } else {
                g_vals[j] = 0.0f;
                m_vals[j] = 0.0f;
                v_vals[j] = 0.0f;
            }
        }

        // --- Phase 2: Compute per-group absmax for requantization ---
        // Each thread contributes to group absmax via shared memory
        // Initialize shared memory
        if (threadIdx.x < FLASH_GROUPS_PER_BLOCK) {
            smem_absmax1[threadIdx.x] = 0.0f;
            smem_absmax2[threadIdx.x] = 0.0f;
        }
        __syncthreads();

// Each thread atomically updates the absmax for its groups
#pragma unroll
        for (int j = 0; j < FLASH_N_PER_THREAD; j++) {
            const int idx = thread_offset + j;
            if (idx < n) {
                const int local_group = (idx - base_idx) / FLASH_GROUP_SIZE;
                atomicMax(reinterpret_cast<int*>(&smem_absmax1[local_group]), __float_as_int(fabsf(m_vals[j])));
                // For variance, we store sqrt(v), so compute absmax of sqrt(v)
                float v_sqrt = sqrtf(v_vals[j]);
                atomicMax(reinterpret_cast<int*>(&smem_absmax2[local_group]), __float_as_int(v_sqrt));
            }
        }
        __syncthreads();

// --- Phase 3: Update parameters ---
#pragma unroll
        for (int j = 0; j < FLASH_N_PER_THREAD; j++) {
            const int idx = thread_offset + j;
            if (idx < n) {
                float param = float(p[idx]);

                if (g_vals[j] != 0.0f || m_vals[j] != 0.0f) {
                    // Bias-corrected estimates
                    float m_hat = m_vals[j] / correction1;
                    float v_hat = v_vals[j] / correction2;

                    // Adam update
                    param -= lr_val * m_hat / (sqrtf(v_hat) + eps_val);

                    // Decoupled weight decay
                    if (weight_decay_val > 0.0f) {
                        param *= (1.0f - lr_val * weight_decay_val);
                    }
                }

                p[idx] = T(param);
            }
        }

// --- Phase 4: Quantize and store states ---
#pragma unroll
        for (int j = 0; j < FLASH_N_PER_THREAD; j++) {
            const int idx = thread_offset + j;
            if (idx < n) {
                const int local_group = (idx - base_idx) / FLASH_GROUP_SIZE;
                const int global_group = idx / FLASH_GROUP_SIZE;

                // Quantize momentum with softsign
                float abs1 = smem_absmax1[local_group];
                abs1 = fmaxf(abs1, 1e-12f);  // avoid div by zero
                float m_normalized = m_vals[j] / abs1;
                float m_transformed = softsign_transform(m_normalized);
                float m_scaled = m_transformed * 127.0f;
                // Round to nearest int8
                int m_int = __float2int_rn(m_scaled);
                m_int = max(-127, min(127, m_int));
                state1[idx] = static_cast<signed char>(m_int);

                // Quantize variance (store sqrt(v) as uint8)
                float abs2 = smem_absmax2[local_group];
                abs2 = fmaxf(abs2, 1e-12f);
                float v_sqrt = sqrtf(v_vals[j]);
                float v_normalized = v_sqrt / abs2;
                float v_scaled = v_normalized * 255.0f;
                int v_int = __float2int_rn(v_scaled);
                v_int = max(0, min(255, v_int));
                state2[idx] = static_cast<unsigned char>(v_int);

                // Store per-group scales as FP16
                // Only the first thread in each group writes the scale
                if (idx % FLASH_GROUP_SIZE == 0) {
                    scales1[global_group] = __float2half(abs1);
                    scales2[global_group] = __float2half(abs2);
                }
            }
        }

        __syncthreads();  // Ensure shared memory is safe for next iteration
    }
}

// ----------------------------------------------------------------------------
// Host-side launch functions

template <typename T>
static void launch_flash_adamw_8bit(T* p,
                                    const T* g,
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
                                    cudaStream_t stream) {
    if (n == 0) return;

    int total_blocks = (int)div_ceil(n, (size_t)FLASH_BLOCK_SIZE_N);
    // Cap grid size to avoid excessive blocks
    int grid_size = std::min(total_blocks, 1024);

    kFlashAdamW8bitKernel<T><<<grid_size, FLASH_BLOCK_THREADS, 0, stream>>>(p,
                                                                            const_cast<T*>(g),
                                                                            state1,
                                                                            state2,
                                                                            scales1,
                                                                            scales2,
                                                                            beta1,
                                                                            beta2,
                                                                            eps,
                                                                            step,
                                                                            lr,
                                                                            weight_decay,
                                                                            opt_params,
                                                                            opt_step,
                                                                            gnorm_scale,
                                                                            (int)n);
    CUDA_CHECK(cudaGetLastError());
}

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
                             cudaStream_t stream) {
    launch_flash_adamw_8bit(p,
                            g,
                            state1,
                            state2,
                            scales1,
                            scales2,
                            n,
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
                             cudaStream_t stream) {
    launch_flash_adamw_8bit(p,
                            g,
                            state1,
                            state2,
                            scales1,
                            scales2,
                            n,
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
                             cudaStream_t stream) {
    launch_flash_adamw_8bit(p,
                            g,
                            state1,
                            state2,
                            scales1,
                            scales2,
                            n,
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
// Multi-tensor kernel (for LoRA with many small tensors)
//
// Processes multiple parameter tensors in a single kernel launch.
// Each tensor is processed with the same flash quantization scheme.
// state_offsets[i] gives the element offset into the contiguous state buffers
// for tensor i (must be GROUP_SIZE-aligned).

template <typename T>
__launch_bounds__(FLASH_BLOCK_THREADS, 3) __global__
    void kFlashAdamW8bitMultiTensorKernel(T** params,
                                          T** __restrict__ const grads,
                                          const int* sizes,
                                          int num_tensors,
                                          signed char* state1,
                                          unsigned char* state2,
                                          half* scales1,
                                          half* scales2,
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

    float g_vals[FLASH_N_PER_THREAD];
    float m_vals[FLASH_N_PER_THREAD];
    float v_vals[FLASH_N_PER_THREAD];

    __shared__ float smem_absmax1[FLASH_GROUPS_PER_BLOCK];
    __shared__ float smem_absmax2[FLASH_GROUPS_PER_BLOCK];

    // Process each tensor
    for (int tensor_idx = 0; tensor_idx < num_tensors; tensor_idx++) {
        T* p = params[tensor_idx];
        const T* g = grads[tensor_idx];
        const int n = sizes[tensor_idx];
        const int s_offset = state_offsets[tensor_idx];

        signed char* t_state1 = state1 + s_offset;
        unsigned char* t_state2 = state2 + s_offset;
        const int group_offset = s_offset / FLASH_GROUP_SIZE;
        half* t_scales1 = scales1 + group_offset;
        half* t_scales2 = scales2 + group_offset;

        const int total_blocks = (n + FLASH_BLOCK_SIZE_N - 1) / FLASH_BLOCK_SIZE_N;

        for (int block_idx = blockIdx.x; block_idx < total_blocks; block_idx += gridDim.x) {
            const int base_idx = block_idx * FLASH_BLOCK_SIZE_N;
            const int thread_offset = base_idx + threadIdx.x * FLASH_N_PER_THREAD;

// Phase 1: Load and dequantize
#pragma unroll
            for (int j = 0; j < FLASH_N_PER_THREAD; j++) {
                const int idx = thread_offset + j;
                if (idx < n) {
                    float g_val = float(g[idx]);
                    if (!isnan(g_val) && !isinf(g_val)) {
                        g_val *= gnorm_scale;
                        g_vals[j] = g_val;

                        const int local_group_idx = idx / FLASH_GROUP_SIZE;
                        float scale1 = __half2float(t_scales1[local_group_idx]);
                        float scale2 = __half2float(t_scales2[local_group_idx]);

                        float m_quantized = float(t_state1[idx]) / 127.0f;
                        m_vals[j] = inverse_softsign(m_quantized) * scale1;

                        float v_sqrt_quantized = float(t_state2[idx]) / 255.0f;
                        float v_sqrt = v_sqrt_quantized * scale2;
                        v_vals[j] = v_sqrt * v_sqrt;

                        m_vals[j] = beta1_val * m_vals[j] + (1.0f - beta1_val) * g_val;
                        v_vals[j] = beta2_val * v_vals[j] + (1.0f - beta2_val) * g_val * g_val;
                    } else {
                        g_vals[j] = 0.0f;
                        m_vals[j] = 0.0f;
                        v_vals[j] = 0.0f;
                    }
                } else {
                    g_vals[j] = 0.0f;
                    m_vals[j] = 0.0f;
                    v_vals[j] = 0.0f;
                }
            }

            // Phase 2: Per-group absmax
            if (threadIdx.x < FLASH_GROUPS_PER_BLOCK) {
                smem_absmax1[threadIdx.x] = 0.0f;
                smem_absmax2[threadIdx.x] = 0.0f;
            }
            __syncthreads();

#pragma unroll
            for (int j = 0; j < FLASH_N_PER_THREAD; j++) {
                const int idx = thread_offset + j;
                if (idx < n) {
                    const int local_group = (idx - base_idx) / FLASH_GROUP_SIZE;
                    atomicMax(reinterpret_cast<int*>(&smem_absmax1[local_group]), __float_as_int(fabsf(m_vals[j])));
                    float v_sqrt = sqrtf(v_vals[j]);
                    atomicMax(reinterpret_cast<int*>(&smem_absmax2[local_group]), __float_as_int(v_sqrt));
                }
            }
            __syncthreads();

// Phase 3: Update parameters
#pragma unroll
            for (int j = 0; j < FLASH_N_PER_THREAD; j++) {
                const int idx = thread_offset + j;
                if (idx < n) {
                    float param = float(p[idx]);
                    if (g_vals[j] != 0.0f || m_vals[j] != 0.0f) {
                        float m_hat = m_vals[j] / correction1;
                        float v_hat = v_vals[j] / correction2;
                        param -= lr_val * m_hat / (sqrtf(v_hat) + eps_val);
                        if (weight_decay_val > 0.0f) {
                            param *= (1.0f - lr_val * weight_decay_val);
                        }
                    }
                    p[idx] = T(param);
                }
            }

// Phase 4: Quantize and store
#pragma unroll
            for (int j = 0; j < FLASH_N_PER_THREAD; j++) {
                const int idx = thread_offset + j;
                if (idx < n) {
                    const int local_group = (idx - base_idx) / FLASH_GROUP_SIZE;
                    const int global_group = idx / FLASH_GROUP_SIZE;

                    float abs1 = smem_absmax1[local_group];
                    abs1 = fmaxf(abs1, 1e-12f);
                    float m_normalized = m_vals[j] / abs1;
                    float m_transformed = softsign_transform(m_normalized);
                    float m_scaled = m_transformed * 127.0f;
                    int m_int = __float2int_rn(m_scaled);
                    m_int = max(-127, min(127, m_int));
                    t_state1[idx] = static_cast<signed char>(m_int);

                    float abs2 = smem_absmax2[local_group];
                    abs2 = fmaxf(abs2, 1e-12f);
                    float v_sqrt = sqrtf(v_vals[j]);
                    float v_normalized = v_sqrt / abs2;
                    float v_scaled = v_normalized * 255.0f;
                    int v_int = __float2int_rn(v_scaled);
                    v_int = max(0, min(255, v_int));
                    t_state2[idx] = static_cast<unsigned char>(v_int);

                    if (idx % FLASH_GROUP_SIZE == 0) {
                        t_scales1[global_group] = __float2half(abs1);
                        t_scales2[global_group] = __float2half(abs2);
                    }
                }
            }

            __syncthreads();
        }

        __syncthreads();  // Ensure all blocks finish this tensor before moving to next
    }
}

// ----------------------------------------------------------------------------
// Multi-tensor host-side launch functions

template <typename T>
static void launch_flash_adamw_8bit_multi_tensor(T** params,
                                                 T** grads,
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
                                                 cudaStream_t stream) {
    if (num_tensors == 0 || total_params == 0) return;

    int total_blocks = (int)div_ceil(total_params, (size_t)FLASH_BLOCK_SIZE_N);
    int grid_size = std::min(total_blocks, 256);

    kFlashAdamW8bitMultiTensorKernel<T><<<grid_size, FLASH_BLOCK_THREADS, 0, stream>>>(params,
                                                                                       const_cast<T**>(grads),
                                                                                       sizes,
                                                                                       num_tensors,
                                                                                       state1,
                                                                                       state2,
                                                                                       scales1,
                                                                                       scales2,
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
                                          cudaStream_t stream) {
    launch_flash_adamw_8bit_multi_tensor(params,
                                         grads,
                                         sizes,
                                         num_tensors,
                                         state1,
                                         state2,
                                         scales1,
                                         scales2,
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
                                          cudaStream_t stream) {
    launch_flash_adamw_8bit_multi_tensor(params,
                                         grads,
                                         sizes,
                                         num_tensors,
                                         state1,
                                         state2,
                                         scales1,
                                         scales2,
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

__global__ void kInitFlashAdamW8bitState(signed char* state1,
                                         unsigned char* state2,
                                         half* scales1,
                                         half* scales2,
                                         size_t n,
                                         size_t num_groups) {
    const size_t total = n > num_groups ? n : num_groups;
    const size_t stride = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
    for (size_t idx =
             static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
         idx < total;
         idx += stride) {
        if (idx < n) {
            state1[idx] = 0;  // Zero momentum (int8)
            state2[idx] = 0;  // Zero variance (uint8)
        }
        if (idx < num_groups) {
            scales1[idx] = __float2half(1e-7f);  // Small positive scale
            scales2[idx] = __float2half(1e-7f);
        }
    }
}

void init_flash_adamw8bit_state(signed char* state1,
                                unsigned char* state2,
                                half* scales1,
                                half* scales2,
                                size_t n,
                                cudaStream_t stream) {
    if (n == 0) return;

    const size_t num_groups = flash_adamw8bit_num_scales(n);
    const size_t total_elements = std::max(n, num_groups);

    int threads = 256;
    const size_t blocks_ideal = div_ceil(total_elements, (size_t)threads);
    const int blocks = static_cast<int>(std::min(blocks_ideal, (size_t)65535));

    kInitFlashAdamW8bitState<<<blocks, threads, 0, stream>>>(state1, state2, scales1, scales2, n, num_groups);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace optimizers
