// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// AdamW 8-bit optimizer kernel based on bitsandbytes implementation
// Reference: https://github.com/TimDettmers/bitsandbytes

#include <algorithm>
#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "kernels/squirrel_noise.cuh"
#include "kernels/kernel_utils.cuh"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

#include "adamw_8bit.h"

namespace optimizers {

// ----------------------------------------------------------------------------
// Constants and configuration

// Block size for 8-bit optimizer (number of elements processed per block)
// This determines the granularity of quantization - each block has its own absmax.
constexpr int ADAMW8BIT_BLOCK_SIZE_INTERNAL = ADAMW8BIT_BLOCK_SIZE;
static_assert(ADAMW8BIT_BLOCK_SIZE_INTERNAL % 256 == 0, "ADAMW8BIT block size must be a multiple of 256");

// Number of elements processed per thread. Keep 256 threads per block.
constexpr int ADAMW8BIT_NUM_PER_THREAD = ADAMW8BIT_BLOCK_SIZE_INTERNAL / 256;

// Number of lanes for quantile caching in shared memory
constexpr int ADAMW8BIT_LANES = 2;

// Number of quadrant pivots for fast quantization search
constexpr int ADAMW8BIT_QUAD = 3;

// ----------------------------------------------------------------------------
// Quantization helper functions

/**
 * @brief Fast 2D quantization using quadrant pivots and binary search.
 *
 * Quantizes a normalized value (in [-1, 1] or [0, 1]) to an 8-bit index
 * using a pre-computed quantization map. Uses quadrant pivots for
 * initial narrowing, then binary search for final determination.
 *
 * @tparam SIGNED If 1, maps values in [-1, 1]; if 0, maps values in [0, 1].
 * @param quadrants Pre-computed quadrant pivot values for fast initial search.
 * @param smem_code Shared memory containing the 256-entry quantization map.
 * @param x Normalized input value to quantize.
 * @return 8-bit quantization index (0-255).
 */
template <int SIGNED>
__device__ __forceinline__ unsigned char
quantize_2D(float* __restrict__ quadrants, float* __restrict__ const smem_code, float x) {
    int pivot = 127;
    int upper_pivot = 255;
    int lower_pivot = 0;

    float lower = SIGNED ? -1.0f : 0.0f;
    float upper = 1.0f;
    float midpoint;
    float val = quadrants[1];
    int local_pivot = 1;
    int offset = 1;

    // Binary search through the quantization map
    // i >>= 1 gives steps of {64, 32, 16, 8, 4, 2, 1}
    for (int i = 64; i > 0; i >>= 1) {
        if (x > val) {
            lower_pivot = pivot;
            lower = val;
            pivot += i;
            local_pivot += offset;
        } else {
            upper_pivot = pivot;
            upper = val;
            pivot -= i;
            local_pivot -= offset;
        }
        val = i >= 64 ? quadrants[local_pivot] : smem_code[pivot];
        offset -= 1;
    }

    // Final decision based on midpoint
    if (x > val) {
        midpoint = (upper + val) * 0.5f;
        if (x > midpoint)
            return upper_pivot;
        else
            return pivot;
    } else {
        midpoint = (lower + val) * 0.5f;
        if (x < midpoint)
            return lower_pivot;
        else
            return pivot;
    }
}

// ----------------------------------------------------------------------------
// 8-bit AdamW blockwise kernel

/**
 * @brief 8-bit AdamW optimizer kernel with block-wise quantization.
 *
 * Implements AdamW with 8-bit quantized optimizer states (m and v).
 * Each block of BLOCK_SIZE elements shares a single absmax scaling factor,
 * which enables efficient compression of optimizer states to 8 bits.
 *
 * The algorithm:
 * 1. Dequantize m and v states using per-block absmax values
 * 2. Update m: m = beta1 * m + (1 - beta1) * g
 * 3. Update v: v = beta2 * v + (1 - beta2) * g^2
 * 4. Update params: p = p - lr * (m_hat / (sqrt(v_hat) + eps))
 * 5. Apply weight decay: p = p * (1 - lr * weight_decay)
 * 6. Quantize updated m and v back to 8-bit
 *
 * @tparam T Parameter and gradient type (float, half, or nv_bfloat16).
 * @tparam BLOCK_SIZE Number of elements per quantization block.
 * @tparam N_PER_TH Number of elements processed by each thread.
 * @param p Parameter tensor (read/write).
 * @param g Gradient tensor (read-only).
 * @param state1 First moment (m) in 8-bit quantized form (read/write).
 * @param state2 Second moment (v) in 8-bit quantized form (read/write).
 * @param beta1 Exponential decay rate for first moment (typically 0.9).
 * @param beta2 Exponential decay rate for second moment (typically 0.999).
 * @param eps Small constant for numerical stability (typically 1e-8).
 * @param step Current optimization step (1-indexed, for bias correction).
 * @param lr Learning rate.
 * @param quantiles1 256-entry quantization map for first moment.
 * @param quantiles2 256-entry quantization map for second moment.
 * @param absmax1 Per-block absolute max values for first moment.
 * @param absmax2 Per-block absolute max values for second moment.
 * @param weight_decay Weight decay coefficient (L2 regularization).
 * @param gnorm_scale Gradient scaling factor (for gradient clipping).
 * @param skip_zeros Whether to skip zero gradients.
 * @param n Total number of parameters.
 */
template <typename T, int BLOCK_SIZE, int N_PER_TH>
__launch_bounds__(256, 3)
__global__ void kAdamW8bitBlockwise(
    T* p,
    T* __restrict__ const g,
    unsigned char* state1,
    unsigned char* state2,
    const float beta1,
    const float beta2,
    const float eps,
    const int step,
    const float lr,
    float* __restrict__ const quantiles1,
    float* __restrict__ const quantiles2,
    float* absmax1,
    float* absmax2,
    float weight_decay,
    const float* __restrict__ opt_params,
    const int* __restrict__ opt_step,
    const float* __restrict__ gnorm_scale_ptr,  // Device pointer for graph-capture compatibility
    const bool skip_zeros,
    const int n
) {
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
    const int n_full = gridDim.x * BLOCK_SIZE;
    const int base_idx = blockIdx.x * BLOCK_SIZE;
    int valid_items = 0;
    float g_val = 0.0f;
    float s1_vals[N_PER_TH];
    float s2_vals[N_PER_TH];

    // Bias correction factors
    const float correction1 = 1.0f - __powf(beta1_val, step_val);
    const float correction2 = sqrtf(1.0f - __powf(beta2_val, step_val));
    const float step_size = __fdividef(-lr_val * correction2, correction1);
    
    const int lane_id = threadIdx.x % ADAMW8BIT_LANES;
    float new_local_abs_max1 = -FLT_MAX;
    float new_local_abs_max2 = -FLT_MAX;
    float quadrants1[ADAMW8BIT_QUAD];
    float quadrants2[ADAMW8BIT_QUAD];

    unsigned char c1s[N_PER_TH];
    unsigned char c2s[N_PER_TH];
    T g_vals[N_PER_TH];
    T p_vals[N_PER_TH];

    // Shared memory for quantization maps (replicated across lanes for coalesced access)
    __shared__ float smem_quantiles1[ADAMW8BIT_LANES][257];
    __shared__ float smem_quantiles2[ADAMW8BIT_LANES][257];
    
    // Block-level reduction storage
    __shared__ float smem_exchange1[1];
    __shared__ float smem_exchange2[1];

    // Load quantization maps into shared memory
    if (threadIdx.x < 256) {
        smem_quantiles1[0][threadIdx.x] = quantiles1[threadIdx.x];
        smem_quantiles2[0][threadIdx.x] = quantiles2[threadIdx.x];
        #pragma unroll
        for (unsigned int j = 1; j < ADAMW8BIT_LANES; j++) {
            smem_quantiles1[j][threadIdx.x] = smem_quantiles1[0][threadIdx.x];
            smem_quantiles2[j][threadIdx.x] = smem_quantiles2[0][threadIdx.x];
        }
    }
    // smem_exchange is updated via atomicMax (bitwise) and must be initialized before first use.
    // Without this, the first iteration can pick up garbage values and corrupt absmax state,
    // leading to unstable or divergent training (especially at higher learning rates).
    if (threadIdx.x == 0) {
        smem_exchange1[0] = -FLT_MAX;
        smem_exchange2[0] = -FLT_MAX;
    }
    __syncthreads();

    // Compute quadrant pivots for fast quantization
    #pragma unroll
    for (int k = 0; k < ADAMW8BIT_QUAD; k++) {
        quadrants1[k] = smem_quantiles1[lane_id][(k * 256 / (ADAMW8BIT_QUAD + 1)) + (256 / (ADAMW8BIT_QUAD + 1) - 1)];
        quadrants2[k] = smem_quantiles2[lane_id][(k * 256 / (ADAMW8BIT_QUAD + 1)) + (256 / (ADAMW8BIT_QUAD + 1) - 1)];
    }

    // Process blocks
    for (unsigned int i = base_idx; i < n_full; i += gridDim.x * BLOCK_SIZE) {
        valid_items = n - i >= BLOCK_SIZE ? BLOCK_SIZE : n - i;
        
        // Load data
        const int thread_offset = i + threadIdx.x * N_PER_TH;
        #pragma unroll
        for (int j = 0; j < N_PER_TH; j++) {
            const int idx = thread_offset + j;
            if (idx < n) {
                g_vals[j] = g[idx];
                c1s[j] = state1[idx];
                c2s[j] = state2[idx];
            } else {
                g_vals[j] = T(0);
                c1s[j] = 128;  // Represents 0 in signed quantization
                c2s[j] = 0;    // Represents 0 in unsigned quantization
            }
        }

        new_local_abs_max1 = -FLT_MAX;
        new_local_abs_max2 = -FLT_MAX;

        // Update moments
        #pragma unroll
        for (unsigned int j = 0; j < N_PER_TH; j++) {
            g_val = float(g_vals[j]);
            
            if (!isnan(g_val) && !isinf(g_val)) {
                // Apply gradient scaling
                g_val *= gnorm_scale;

                // Dequantize and update second moment (v)
                // v is stored unsigned (non-negative values)
                s2_vals[j] = smem_quantiles2[lane_id][c2s[j]] * absmax2[i / BLOCK_SIZE];
                s2_vals[j] = s2_vals[j] * beta2_val + (1.0f - beta2_val) * g_val * g_val;

                // Dequantize and update first moment (m)
                // m is stored signed (can be negative)
                s1_vals[j] = smem_quantiles1[lane_id][c1s[j]] * absmax1[i / BLOCK_SIZE];
                s1_vals[j] = s1_vals[j] * beta1_val + (1.0f - beta1_val) * g_val;
            } else {
                s1_vals[j] = 0.0f;
                s2_vals[j] = 0.0f;
            }

            new_local_abs_max1 = fmaxf(new_local_abs_max1, fabsf(s1_vals[j]));
            new_local_abs_max2 = fmaxf(new_local_abs_max2, fabsf(s2_vals[j]));
        }

        // Block-level reduction to find max absmax
        // Use warp reduction first
        for (int offset = 16; offset > 0; offset >>= 1) {
            new_local_abs_max1 = fmaxf(new_local_abs_max1, __shfl_xor_sync(0xFFFFFFFF, new_local_abs_max1, offset));
            new_local_abs_max2 = fmaxf(new_local_abs_max2, __shfl_xor_sync(0xFFFFFFFF, new_local_abs_max2, offset));
        }
        
        // First thread of each warp writes to shared memory
        if (threadIdx.x % 32 == 0) {
            atomicMax(reinterpret_cast<int*>(&smem_exchange1[0]), __float_as_int(new_local_abs_max1));
            atomicMax(reinterpret_cast<int*>(&smem_exchange2[0]), __float_as_int(new_local_abs_max2));
        }
        __syncthreads();

        // Thread 0 writes absmax to global memory
        if (threadIdx.x == 0) {
            absmax1[i / BLOCK_SIZE] = smem_exchange1[0];
            absmax2[i / BLOCK_SIZE] = smem_exchange2[0];
        }
        
        // All threads read the final absmax
        new_local_abs_max1 = smem_exchange1[0];
        new_local_abs_max2 = smem_exchange2[0];
        
        // Reset for next iteration
        if (threadIdx.x == 0) {
            smem_exchange1[0] = -FLT_MAX;
            smem_exchange2[0] = -FLT_MAX;
        }
        
        __syncthreads();

        // Load parameters
        #pragma unroll
        for (int j = 0; j < N_PER_TH; j++) {
            const int idx = thread_offset + j;
            if (idx < n) {
                p_vals[j] = p[idx];
            } else {
                p_vals[j] = T(0);
            }
        }

        // Update parameters
        #pragma unroll
        for (unsigned int j = 0; j < N_PER_TH; j++) {
            if (!isnan(float(g_vals[j])) && !isinf(float(g_vals[j]))) {
                // AdamW update: p = p - lr * (m_hat / (sqrt(v_hat) + eps))
                float param = float(p_vals[j]);
                param += step_size * __fdividef(s1_vals[j], sqrtf(s2_vals[j]) + correction2 * eps_val);

                // Weight decay: p = p * (1 - lr * weight_decay)
                if (weight_decay_val > 0.0f) {
                    param *= (1.0f - lr_val * weight_decay_val);
                }
                p_vals[j] = T(param);
            }
        }

        // Store updated parameters
        #pragma unroll
        for (int j = 0; j < N_PER_TH; j++) {
            const int idx = thread_offset + j;
            if (idx < n) {
                p[idx] = p_vals[j];
            }
        }

        // Quantize and store states
        #pragma unroll
        for (unsigned int j = 0; j < N_PER_TH; j++) {
            // Quantize first moment (signed)
            float normalized1 = new_local_abs_max1 > 0.0f ? 
                __fdividef(s1_vals[j], new_local_abs_max1) : 0.0f;
            c1s[j] = quantize_2D<1>(quadrants1, smem_quantiles1[lane_id], normalized1);
            
            // Ensure sign consistency after quantization
            if (signbit(smem_quantiles1[lane_id][c1s[j]]) != signbit(s1_vals[j])) {
                if (s1_vals[j] > 0.0f)
                    c1s[j] += 1;
                else
                    c1s[j] -= 1;
            }

            // Quantize second moment (unsigned, always positive)
            float normalized2 = new_local_abs_max2 > 0.0f ?
                __fdividef(s2_vals[j], new_local_abs_max2) : 0.0f;
            c2s[j] = quantize_2D<0>(quadrants2, smem_quantiles2[lane_id], normalized2);
        }

        // Store quantized states
        #pragma unroll
        for (int j = 0; j < N_PER_TH; j++) {
            const int idx = thread_offset + j;
            if (idx < n) {
                state1[idx] = c1s[j];
                state2[idx] = c2s[j];
            }
        }
    }
}

/**
 * @brief Mixed-precision 8-bit AdamW optimizer kernel (TParam params, TGrad gradients).
 * 
 * Similar to kAdamW8bitBlockwise but supports different types for parameters and gradients.
 * Gradients are converted to float for the update computation.
 */
template <typename TParam, typename TGrad, int BLOCK_SIZE, int N_PER_TH>
__launch_bounds__(256, 3)
__global__ void kAdamW8bitBlockwiseMixed(
    TParam* p,
    TGrad* __restrict__ const g,
    unsigned char* state1,
    unsigned char* state2,
    const float beta1,
    const float beta2,
    const float eps,
    const int step,
    const float lr,
    float* __restrict__ const quantiles1,
    float* __restrict__ const quantiles2,
    float* absmax1,
    float* absmax2,
    float weight_decay,
    const float* __restrict__ opt_params,
    const int* __restrict__ opt_step,
    const float* __restrict__ gnorm_scale_ptr,  // Device pointer for graph-capture compatibility
    const bool skip_zeros,
    const int n
) {
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
    const int n_full = gridDim.x * BLOCK_SIZE;
    const int base_idx = blockIdx.x * BLOCK_SIZE;
    float s1_vals[N_PER_TH];
    float s2_vals[N_PER_TH];

    // Bias correction factors
    const float correction1 = 1.0f - __powf(beta1_val, step_val);
    const float correction2 = sqrtf(1.0f - __powf(beta2_val, step_val));
    const float step_size = __fdividef(-lr_val * correction2, correction1);

    const int lane_id = threadIdx.x % ADAMW8BIT_LANES;
    float new_local_abs_max1 = -FLT_MAX;
    float new_local_abs_max2 = -FLT_MAX;
    float quadrants1[ADAMW8BIT_QUAD];
    float quadrants2[ADAMW8BIT_QUAD];

    unsigned char c1s[N_PER_TH];
    unsigned char c2s[N_PER_TH];
    TParam p_vals[N_PER_TH];
    TGrad g_vals[N_PER_TH];

    // Shared memory for quantization maps (replicated across lanes for coalesced access)
    __shared__ float smem_quantiles1[ADAMW8BIT_LANES][257];
    __shared__ float smem_quantiles2[ADAMW8BIT_LANES][257];
    __shared__ float smem_exchange1[1];
    __shared__ float smem_exchange2[1];

    // Load quantization maps into shared memory
    if (threadIdx.x < 256) {
        smem_quantiles1[0][threadIdx.x] = quantiles1[threadIdx.x];
        smem_quantiles2[0][threadIdx.x] = quantiles2[threadIdx.x];
        #pragma unroll
        for (unsigned int j = 1; j < ADAMW8BIT_LANES; j++) {
            smem_quantiles1[j][threadIdx.x] = smem_quantiles1[0][threadIdx.x];
            smem_quantiles2[j][threadIdx.x] = smem_quantiles2[0][threadIdx.x];
        }
    }
    if (threadIdx.x == 0) {
        smem_exchange1[0] = -FLT_MAX;
        smem_exchange2[0] = -FLT_MAX;
    }
    __syncthreads();

    // Compute quadrant pivots for fast quantization
    #pragma unroll
    for (int k = 0; k < ADAMW8BIT_QUAD; k++) {
        quadrants1[k] = smem_quantiles1[lane_id][(k * 256 / (ADAMW8BIT_QUAD + 1)) + (256 / (ADAMW8BIT_QUAD + 1) - 1)];
        quadrants2[k] = smem_quantiles2[lane_id][(k * 256 / (ADAMW8BIT_QUAD + 1)) + (256 / (ADAMW8BIT_QUAD + 1) - 1)];
    }

    for (unsigned int i = base_idx; i < n_full; i += gridDim.x * BLOCK_SIZE) {
        new_local_abs_max1 = -FLT_MAX;
        new_local_abs_max2 = -FLT_MAX;

        // Load gradients and state codes
        const int thread_offset = i + threadIdx.x * N_PER_TH;
        #pragma unroll
        for (int j = 0; j < N_PER_TH; j++) {
            const int idx = thread_offset + j;
            if (idx < n) {
                g_vals[j] = g[idx];
                c1s[j] = state1[idx];
                c2s[j] = state2[idx];
            } else {
                g_vals[j] = TGrad(0);
                c1s[j] = 128;  // Represents 0 in signed quantization
                c2s[j] = 0;    // Represents 0 in unsigned quantization
            }
        }

        // Update moments
        #pragma unroll
        for (unsigned int j = 0; j < N_PER_TH; j++) {
            float g_val = float(g_vals[j]);
            if (!isnan(g_val) && !isinf(g_val)) {
                g_val *= gnorm_scale;
                s2_vals[j] = smem_quantiles2[lane_id][c2s[j]] * absmax2[i / BLOCK_SIZE];
                s2_vals[j] = s2_vals[j] * beta2_val + (1.0f - beta2_val) * g_val * g_val;
                s1_vals[j] = smem_quantiles1[lane_id][c1s[j]] * absmax1[i / BLOCK_SIZE];
                s1_vals[j] = s1_vals[j] * beta1_val + (1.0f - beta1_val) * g_val;
            } else {
                s1_vals[j] = 0.0f;
                s2_vals[j] = 0.0f;
            }
            new_local_abs_max1 = fmaxf(new_local_abs_max1, fabsf(s1_vals[j]));
            new_local_abs_max2 = fmaxf(new_local_abs_max2, fabsf(s2_vals[j]));
        }

        // Block-level reduction to find max absmax (warp reduce + atomic across warps)
        for (int offset = 16; offset > 0; offset >>= 1) {
            new_local_abs_max1 = fmaxf(new_local_abs_max1, __shfl_xor_sync(0xFFFFFFFF, new_local_abs_max1, offset));
            new_local_abs_max2 = fmaxf(new_local_abs_max2, __shfl_xor_sync(0xFFFFFFFF, new_local_abs_max2, offset));
        }
        if (threadIdx.x % 32 == 0) {
            atomicMax(reinterpret_cast<int*>(&smem_exchange1[0]), __float_as_int(new_local_abs_max1));
            atomicMax(reinterpret_cast<int*>(&smem_exchange2[0]), __float_as_int(new_local_abs_max2));
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            absmax1[i / BLOCK_SIZE] = smem_exchange1[0];
            absmax2[i / BLOCK_SIZE] = smem_exchange2[0];
        }
        new_local_abs_max1 = smem_exchange1[0];
        new_local_abs_max2 = smem_exchange2[0];
        if (threadIdx.x == 0) {
            smem_exchange1[0] = -FLT_MAX;
            smem_exchange2[0] = -FLT_MAX;
        }
        __syncthreads();

        // Load parameters
        #pragma unroll
        for (int j = 0; j < N_PER_TH; j++) {
            const int idx = thread_offset + j;
            if (idx < n) {
                p_vals[j] = p[idx];
            } else {
                p_vals[j] = TParam(0);
            }
        }

        // Update parameters
        #pragma unroll
        for (unsigned int j = 0; j < N_PER_TH; j++) {
            float g_val = float(g_vals[j]);
            if (!isnan(g_val) && !isinf(g_val)) {
                float param = float(p_vals[j]);
                param += step_size * __fdividef(s1_vals[j], sqrtf(s2_vals[j]) + correction2 * eps_val);
                if (weight_decay_val > 0.0f) {
                    param *= (1.0f - lr_val * weight_decay_val);
                }
                p_vals[j] = TParam(param);
            }
        }

        // Store updated parameters
        #pragma unroll
        for (int j = 0; j < N_PER_TH; j++) {
            const int idx = thread_offset + j;
            if (idx < n) {
                p[idx] = p_vals[j];
            }
        }

        // Quantize and store states
        #pragma unroll
        for (unsigned int j = 0; j < N_PER_TH; j++) {
            float normalized1 = new_local_abs_max1 > 0.0f ?
                __fdividef(s1_vals[j], new_local_abs_max1) : 0.0f;
            c1s[j] = quantize_2D<1>(quadrants1, smem_quantiles1[lane_id], normalized1);

            if (signbit(smem_quantiles1[lane_id][c1s[j]]) != signbit(s1_vals[j])) {
                if (s1_vals[j] > 0.0f) c1s[j] += 1;
                else c1s[j] -= 1;
            }

            float normalized2 = new_local_abs_max2 > 0.0f ?
                __fdividef(s2_vals[j], new_local_abs_max2) : 0.0f;
            c2s[j] = quantize_2D<0>(quadrants2, smem_quantiles2[lane_id], normalized2);
        }

        #pragma unroll
        for (int j = 0; j < N_PER_TH; j++) {
            const int idx = thread_offset + j;
            if (idx < n) {
                state1[idx] = c1s[j];
                state2[idx] = c2s[j];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Host-side launch functions

/**
 * @brief Launch the 8-bit AdamW optimizer kernel for float parameters.
 *
 * @param p Parameter tensor.
 * @param g Gradient tensor.
 * @param state1 First moment (m) in 8-bit quantized form.
 * @param state2 Second moment (v) in 8-bit quantized form.
 * @param n Number of parameters.
 * @param lr Learning rate.
 * @param beta1 First moment decay rate.
 * @param beta2 Second moment decay rate.
 * @param step Current optimization step (1-indexed).
 * @param eps Numerical stability constant.
 * @param weight_decay Weight decay coefficient.
 * @param gnorm_scale Device pointer to gradient scaling factor (enables CUDA graph capture).
 * @param quantiles1 Quantization map for first moment (256 floats).
 * @param quantiles2 Quantization map for second moment (256 floats).
 * @param absmax1 Per-block absmax for first moment.
 * @param absmax2 Per-block absmax for second moment.
 * @param stream CUDA stream.
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
) {
    constexpr int BLOCK_SIZE = ADAMW8BIT_BLOCK_SIZE_INTERNAL;
    constexpr int N_PER_TH = ADAMW8BIT_NUM_PER_THREAD;

    int num_blocks = div_ceil(n, (size_t)BLOCK_SIZE);
    int threads_per_block = BLOCK_SIZE / N_PER_TH;

    kAdamW8bitBlockwise<float, BLOCK_SIZE, N_PER_TH>
        <<<num_blocks, threads_per_block, 0, stream>>>(
            p, const_cast<float*>(g), state1, state2,
            beta1, beta2, eps, step, lr,
            const_cast<float*>(quantiles1), const_cast<float*>(quantiles2),
            absmax1, absmax2,
            weight_decay, opt_params, opt_step, gnorm_scale, false, (int)n
        );
    CUDA_CHECK(cudaGetLastError());
}

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
) {
    constexpr int BLOCK_SIZE = ADAMW8BIT_BLOCK_SIZE_INTERNAL;
    constexpr int N_PER_TH = ADAMW8BIT_NUM_PER_THREAD;

    int num_blocks = div_ceil(n, (size_t)BLOCK_SIZE);
    int threads_per_block = BLOCK_SIZE / N_PER_TH;

    kAdamW8bitBlockwise<nv_bfloat16, BLOCK_SIZE, N_PER_TH>
        <<<num_blocks, threads_per_block, 0, stream>>>(
            p, const_cast<nv_bfloat16*>(g), state1, state2,
            beta1, beta2, eps, step, lr,
            const_cast<float*>(quantiles1), const_cast<float*>(quantiles2),
            absmax1, absmax2,
            weight_decay, opt_params, opt_step, gnorm_scale, false, (int)n
        );
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Launch the 8-bit AdamW optimizer kernel for mixed precision (FP32 params, BF16 grads).
 *
 * This variant converts BF16 gradients to FP32 on-the-fly for the update computation,
 * keeping parameters in FP32 for full precision master weights with LoRA.
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
) {
    constexpr int BLOCK_SIZE = ADAMW8BIT_BLOCK_SIZE_INTERNAL;
    constexpr int N_PER_TH = ADAMW8BIT_NUM_PER_THREAD;

    int num_blocks = div_ceil(n, (size_t)BLOCK_SIZE);
    int threads_per_block = BLOCK_SIZE / N_PER_TH;

    // Use the mixed-precision kernel
    kAdamW8bitBlockwiseMixed<float, nv_bfloat16, BLOCK_SIZE, N_PER_TH>
        <<<num_blocks, threads_per_block, 0, stream>>>(
            p, const_cast<nv_bfloat16*>(g), state1, state2,
            beta1, beta2, eps, step, lr,
            const_cast<float*>(quantiles1), const_cast<float*>(quantiles2),
            absmax1, absmax2,
            weight_decay, opt_params, opt_step, gnorm_scale, false, (int)n
        );
    CUDA_CHECK(cudaGetLastError());
}

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
) {
    constexpr int BLOCK_SIZE = ADAMW8BIT_BLOCK_SIZE_INTERNAL;
    constexpr int N_PER_TH = ADAMW8BIT_NUM_PER_THREAD;

    int num_blocks = div_ceil(n, (size_t)BLOCK_SIZE);
    int threads_per_block = BLOCK_SIZE / N_PER_TH;

    kAdamW8bitBlockwise<half, BLOCK_SIZE, N_PER_TH>
        <<<num_blocks, threads_per_block, 0, stream>>>(
            p, const_cast<half*>(g), state1, state2,
            beta1, beta2, eps, step, lr,
            const_cast<float*>(quantiles1), const_cast<float*>(quantiles2),
            absmax1, absmax2,
            weight_decay, opt_params, opt_step, gnorm_scale, false, (int)n
        );
    CUDA_CHECK(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Quantization map initialization

/**
 * @brief Creates a dynamic quantization map for 8-bit optimizer states.
 *
 * The dynamic data type uses a dynamic exponent and fraction. As the exponent
 * increases from 0 to -7, the number of bits available for the fraction shrinks.
 * This provides better precision near zero while still covering a wide dynamic range.
 *
 * For signed values (first moment), the map covers [-1, 1].
 * For unsigned values (second moment), the map covers [0, 1].
 *
 * Reference: "8-Bit Approximations for Parallelism in Deep Learning"
 * https://arxiv.org/abs/1511.04561
 *
 * @param[out] code Output array of 256 float values representing the quantization map.
 * @param signed_map If true, creates a signed map for [-1, 1]; otherwise [0, 1].
 */
__host__ void create_dynamic_quantization_map(float* code, bool signed_map) {
    const int total_bits = 8;
    const int max_exponent_bits = 7;
    const bool is_signed = signed_map;
    
    std::vector<float> data;
    
    // Non-sign bits available
    int non_sign_bits = total_bits - 1;
    
    // Additional items from the case where all exponent bits are zero
    int additional_items = (1 << (non_sign_bits - max_exponent_bits)) - 1;
    
    for (int i = 0; i < max_exponent_bits; i++) {
        int fraction_items = is_signed ?
            (1 << (i + non_sign_bits - max_exponent_bits)) + 1 :
            (1 << (i + non_sign_bits - max_exponent_bits + 1)) + 1;
        
        float scale = powf(10.0f, -(max_exponent_bits - 1) + i);
        
        for (int j = 0; j < fraction_items - 1; j++) {
            float boundary_low = 0.1f + (0.9f * j) / (fraction_items - 1);
            float boundary_high = 0.1f + (0.9f * (j + 1)) / (fraction_items - 1);
            float mean = (boundary_low + boundary_high) / 2.0f;
            
            data.push_back(scale * mean);
            if (is_signed) {
                data.push_back(-scale * mean);
            }
        }
    }
    
    // Handle additional items
    if (additional_items > 0) {
        float scale = powf(10.0f, -(max_exponent_bits - 1) + max_exponent_bits - 1);
        for (int j = 0; j < additional_items; j++) {
            float boundary_low = 0.1f + (0.9f * j) / additional_items;
            float boundary_high = 0.1f + (0.9f * (j + 1)) / additional_items;
            float mean = (boundary_low + boundary_high) / 2.0f;
            
            data.push_back(scale * mean);
            if (is_signed) {
                data.push_back(-scale * mean);
            }
        }
    }
    
    // Add zero and one
    data.push_back(0.0f);
    data.push_back(1.0f);
    
    // Sort the data
    std::sort(data.begin(), data.end());
    
    // Pad to 256 elements if needed
    while (data.size() < 256) {
        data.push_back(0.0f);
    }
    std::sort(data.begin(), data.end());
    
    // Copy to output
    for (int i = 0; i < 256; i++) {
        code[i] = data[i];
    }
}

/**
 * @brief Creates the default signed quantization map for first moment (m).
 *
 * @param[out] code Output array of 256 float values.
 */
__host__ void create_adamw8bit_quantiles1(float* code) {
    create_dynamic_quantization_map(code, true);
}

/**
 * @brief Creates the default unsigned quantization map for second moment (v).
 *
 * @param[out] code Output array of 256 float values.
 */
__host__ void create_adamw8bit_quantiles2(float* code) {
    create_dynamic_quantization_map(code, false);
}

// ----------------------------------------------------------------------------
// State initialization

/**
 * @brief Initializes the 8-bit optimizer state tensors.
 *
 * Sets state1 (first moment) to 128 (representing 0 in signed quantization)
 * and state2 (second moment) to 0 (representing 0 in unsigned quantization).
 * Also initializes absmax arrays to small positive values.
 *
 * @param state1 First moment state tensor.
 * @param state2 Second moment state tensor.
 * @param absmax1 Per-block absmax for first moment.
 * @param absmax2 Per-block absmax for second moment.
 * @param n Number of parameters.
 * @param stream CUDA stream.
 */
__global__ void kInitAdamW8bitState(
    unsigned char* state1,
    unsigned char* state2,
    float* absmax1,
    float* absmax2,
    size_t n,
    size_t num_blocks
) {
    const size_t total = n > num_blocks ? n : num_blocks;
    const size_t stride = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
    for (size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
         idx < total;
         idx += stride) {
        if (idx < n) {
            state1[idx] = 128;  // 0 in signed quantization (middle of the range)
            state2[idx] = 0;    // 0 in unsigned quantization (start of the range)
        }
        if (idx < num_blocks) {
            absmax1[idx] = 1e-7f;  // Small positive value to avoid division by zero
            absmax2[idx] = 1e-7f;
        }
    }
}

void init_adamw8bit_state(
    unsigned char* state1,
    unsigned char* state2,
    float* absmax1,
    float* absmax2,
    size_t n,
    cudaStream_t stream
) {
    // Guard against empty tensors - nothing to initialize
    if (n == 0) {
        return;
    }

    const size_t num_blocks = div_ceil(n, (size_t)ADAMW8BIT_BLOCK_SIZE_INTERNAL);
    const size_t total_elements = std::max(n, num_blocks);

    int threads = 256;
    const size_t blocks_ideal = div_ceil(total_elements, (size_t)threads);
    const int blocks = static_cast<int>(std::min(blocks_ideal, (size_t)65535));

    kInitAdamW8bitState<<<blocks, threads, 0, stream>>>(
        state1, state2, absmax1, absmax2, n, num_blocks
    );
    CUDA_CHECK(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Multi-tensor AdamW 8-bit kernel
//
// This kernel processes multiple tensors in a single launch, reducing kernel
// launch overhead from O(num_tensors) to O(1). Critical for LoRA training
// where we have hundreds of small adapter tensors.
// ----------------------------------------------------------------------------

/**
 * @brief Multi-tensor 8-bit AdamW kernel.
 *
 * Processes all tensors sequentially within a single kernel launch.
 * Uses grid-stride loop over all elements across all tensors.
 *
 * Memory layout:
 * - All tensors are stored contiguously in the combined state buffers
 * - tensor_offsets[i] gives the starting element index for tensor i
 * - Each tensor's optimizer state blocks are packed starting at (tensor_offsets[i] / BLOCK_SIZE)
 *
 * @tparam T Parameter type (float or nv_bfloat16).
 */
template <typename T, int BLOCK_SIZE, int N_PER_TH>
__launch_bounds__(256, 3)
__global__ void kAdamW8bitMultiTensor(
    // Tensor metadata (device arrays)
    T** __restrict__ params,           // Array of param pointers
    T** __restrict__ grads,            // Array of grad pointers
    const int* __restrict__ sizes,     // Array of tensor sizes
    const int num_tensors,             // Number of tensors
    // Combined state buffers
    unsigned char* __restrict__ state1,
    unsigned char* __restrict__ state2,
    float* __restrict__ absmax1,
    float* __restrict__ absmax2,
    const int* __restrict__ state_offsets,  // Element offset for each tensor in state buffers
    // Optimizer hyperparameters
    const float beta1,
    const float beta2,
    const float eps,
    const int step,
    const float lr,
    float* __restrict__ const quantiles1,
    float* __restrict__ const quantiles2,
    float weight_decay,
    const float* __restrict__ opt_params,
    const int* __restrict__ opt_step,
    const float* __restrict__ gnorm_scale_ptr
) {
    // Read gnorm_scale from device memory
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
    const float correction2 = sqrtf(1.0f - __powf(beta2_val, step_val));
    const float step_size = __fdividef(-lr_val * correction2, correction1);

    const int lane_id = threadIdx.x % ADAMW8BIT_LANES;

    // Shared memory for quantization maps
    __shared__ float smem_quantiles1[ADAMW8BIT_LANES][257];
    __shared__ float smem_quantiles2[ADAMW8BIT_LANES][257];
    __shared__ float smem_exchange1[1];
    __shared__ float smem_exchange2[1];

    // Load quantization maps
    if (threadIdx.x < 256) {
        smem_quantiles1[0][threadIdx.x] = quantiles1[threadIdx.x];
        smem_quantiles2[0][threadIdx.x] = quantiles2[threadIdx.x];
        #pragma unroll
        for (unsigned int j = 1; j < ADAMW8BIT_LANES; j++) {
            smem_quantiles1[j][threadIdx.x] = smem_quantiles1[0][threadIdx.x];
            smem_quantiles2[j][threadIdx.x] = smem_quantiles2[0][threadIdx.x];
        }
    }

    if (threadIdx.x == 0) {
        smem_exchange1[0] = -FLT_MAX;
        smem_exchange2[0] = -FLT_MAX;
    }
    __syncthreads();

    // Compute quadrant pivots
    float quadrants1[ADAMW8BIT_QUAD];
    float quadrants2[ADAMW8BIT_QUAD];
    #pragma unroll
    for (int k = 0; k < ADAMW8BIT_QUAD; k++) {
        quadrants1[k] = smem_quantiles1[lane_id][(k * 256 / (ADAMW8BIT_QUAD + 1)) + (256 / (ADAMW8BIT_QUAD + 1) - 1)];
        quadrants2[k] = smem_quantiles2[lane_id][(k * 256 / (ADAMW8BIT_QUAD + 1)) + (256 / (ADAMW8BIT_QUAD + 1) - 1)];
    }

    // Thread-local storage
    float s1_vals[N_PER_TH];
    float s2_vals[N_PER_TH];
    unsigned char c1s[N_PER_TH];
    unsigned char c2s[N_PER_TH];
    T g_vals[N_PER_TH];
    T p_vals[N_PER_TH];

    // Process each tensor
    for (int tensor_idx = 0; tensor_idx < num_tensors; tensor_idx++) {
        T* p = params[tensor_idx];
        const T* g = grads[tensor_idx];
        const int n = sizes[tensor_idx];
        const int state_offset = state_offsets[tensor_idx];

        // Pointers into combined state buffers for this tensor
        unsigned char* t_state1 = state1 + state_offset;
        unsigned char* t_state2 = state2 + state_offset;
        const int block_offset_base = state_offset / BLOCK_SIZE;
        float* t_absmax1 = absmax1 + block_offset_base;
        float* t_absmax2 = absmax2 + block_offset_base;

        // Number of quantization blocks for this tensor
        const int num_blocks_for_tensor = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Process this tensor using grid-stride loop over its blocks
        for (int block_id = blockIdx.x; block_id < num_blocks_for_tensor; block_id += gridDim.x) {
            const int base_idx = block_id * BLOCK_SIZE;
            const int valid_items = min(BLOCK_SIZE, n - base_idx);

            float new_local_abs_max1 = -FLT_MAX;
            float new_local_abs_max2 = -FLT_MAX;

            // Load data
            const int thread_offset = base_idx + threadIdx.x * N_PER_TH;
            #pragma unroll
            for (int j = 0; j < N_PER_TH; j++) {
                const int idx = thread_offset + j;
                if (idx < n) {
                    g_vals[j] = g[idx];
                    c1s[j] = t_state1[idx];
                    c2s[j] = t_state2[idx];
                } else {
                    g_vals[j] = T(0);
                    c1s[j] = 128;
                    c2s[j] = 0;
                }
            }

            // Update moments
            #pragma unroll
            for (unsigned int j = 0; j < N_PER_TH; j++) {
                float g_val = float(g_vals[j]);

                if (!isnan(g_val) && !isinf(g_val)) {
                    g_val *= gnorm_scale;

                    s2_vals[j] = smem_quantiles2[lane_id][c2s[j]] * t_absmax2[block_id];
                    s2_vals[j] = s2_vals[j] * beta2_val + (1.0f - beta2_val) * g_val * g_val;

                    s1_vals[j] = smem_quantiles1[lane_id][c1s[j]] * t_absmax1[block_id];
                    s1_vals[j] = s1_vals[j] * beta1_val + (1.0f - beta1_val) * g_val;
                } else {
                    s1_vals[j] = 0.0f;
                    s2_vals[j] = 0.0f;
                }

                new_local_abs_max1 = fmaxf(new_local_abs_max1, fabsf(s1_vals[j]));
                new_local_abs_max2 = fmaxf(new_local_abs_max2, fabsf(s2_vals[j]));
            }

            // Block-level reduction for absmax
            for (int offset = 16; offset > 0; offset >>= 1) {
                new_local_abs_max1 = fmaxf(new_local_abs_max1, __shfl_xor_sync(0xFFFFFFFF, new_local_abs_max1, offset));
                new_local_abs_max2 = fmaxf(new_local_abs_max2, __shfl_xor_sync(0xFFFFFFFF, new_local_abs_max2, offset));
            }

            if (threadIdx.x % 32 == 0) {
                atomicMax(reinterpret_cast<int*>(&smem_exchange1[0]), __float_as_int(new_local_abs_max1));
                atomicMax(reinterpret_cast<int*>(&smem_exchange2[0]), __float_as_int(new_local_abs_max2));
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                t_absmax1[block_id] = smem_exchange1[0];
                t_absmax2[block_id] = smem_exchange2[0];
            }

            new_local_abs_max1 = smem_exchange1[0];
            new_local_abs_max2 = smem_exchange2[0];

            if (threadIdx.x == 0) {
                smem_exchange1[0] = -FLT_MAX;
                smem_exchange2[0] = -FLT_MAX;
            }
            __syncthreads();

            // Load parameters
            #pragma unroll
            for (int j = 0; j < N_PER_TH; j++) {
                const int idx = thread_offset + j;
                if (idx < n) {
                    p_vals[j] = p[idx];
                } else {
                    p_vals[j] = T(0);
                }
            }

            // Update parameters
            #pragma unroll
            for (unsigned int j = 0; j < N_PER_TH; j++) {
                if (!isnan(float(g_vals[j])) && !isinf(float(g_vals[j]))) {
                    float param = float(p_vals[j]);
                    param += step_size * __fdividef(s1_vals[j], sqrtf(s2_vals[j]) + correction2 * eps_val);

                    if (weight_decay_val > 0.0f) {
                        param *= (1.0f - lr_val * weight_decay_val);
                    }
                    p_vals[j] = T(param);
                }
            }

            // Store updated parameters
            #pragma unroll
            for (int j = 0; j < N_PER_TH; j++) {
                const int idx = thread_offset + j;
                if (idx < n) {
                    p[idx] = p_vals[j];
                }
            }

            // Quantize and store states
            #pragma unroll
            for (unsigned int j = 0; j < N_PER_TH; j++) {
                float normalized1 = new_local_abs_max1 > 0.0f ?
                    __fdividef(s1_vals[j], new_local_abs_max1) : 0.0f;
                c1s[j] = quantize_2D<1>(quadrants1, smem_quantiles1[lane_id], normalized1);

                if (signbit(smem_quantiles1[lane_id][c1s[j]]) != signbit(s1_vals[j])) {
                    if (s1_vals[j] > 0.0f)
                        c1s[j] += 1;
                    else
                        c1s[j] -= 1;
                }

                float normalized2 = new_local_abs_max2 > 0.0f ?
                    __fdividef(s2_vals[j], new_local_abs_max2) : 0.0f;
                c2s[j] = quantize_2D<0>(quadrants2, smem_quantiles2[lane_id], normalized2);
            }

            // Store quantized states
            #pragma unroll
            for (int j = 0; j < N_PER_TH; j++) {
                const int idx = thread_offset + j;
                if (idx < n) {
                    t_state1[idx] = c1s[j];
                    t_state2[idx] = c2s[j];
                }
            }
        }

        // Sync before processing next tensor to ensure all blocks are done with shared memory
        __syncthreads();
    }
}

/**
 * @brief Launch multi-tensor 8-bit AdamW optimizer.
 *
 * @param params Array of parameter pointers (device memory).
 * @param grads Array of gradient pointers (device memory).
 * @param sizes Array of tensor sizes (device memory).
 * @param num_tensors Number of tensors.
 * @param state1 Combined first moment state buffer.
 * @param state2 Combined second moment state buffer.
 * @param absmax1 Combined per-block absmax for first moment.
 * @param absmax2 Combined per-block absmax for second moment.
 * @param state_offsets Element offset for each tensor in state buffers (device memory).
 * @param total_params Total number of parameters across all tensors.
 * @param lr Learning rate.
 * @param beta1 First moment decay rate.
 * @param beta2 Second moment decay rate.
 * @param step Current optimization step (1-indexed).
 * @param eps Numerical stability constant.
 * @param weight_decay Weight decay coefficient.
 * @param gnorm_scale Device pointer to gradient scaling factor.
 * @param quantiles1 Quantization map for first moment.
 * @param quantiles2 Quantization map for second moment.
 * @param stream CUDA stream.
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
) {
    // Guard against empty tensors
    if (num_tensors == 0 || total_params == 0) {
        return;
    }

    constexpr int BLOCK_SIZE = ADAMW8BIT_BLOCK_SIZE_INTERNAL;
    constexpr int N_PER_TH = ADAMW8BIT_NUM_PER_THREAD;

    // Calculate total blocks needed across all tensors
    int total_blocks = (int)div_ceil(total_params, (size_t)BLOCK_SIZE);
    // Cap grid size to avoid excessive blocks
    int grid_size = std::min(total_blocks, 256);
    int threads_per_block = BLOCK_SIZE / N_PER_TH;

    kAdamW8bitMultiTensor<float, BLOCK_SIZE, N_PER_TH>
        <<<grid_size, threads_per_block, 0, stream>>>(
            params, grads, sizes, num_tensors,
            state1, state2, absmax1, absmax2, state_offsets,
            beta1, beta2, eps, step, lr,
            const_cast<float*>(quantiles1), const_cast<float*>(quantiles2),
            weight_decay, opt_params, opt_step, gnorm_scale
        );
    CUDA_CHECK(cudaGetLastError());
}

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
) {
    // Guard against empty tensors
    if (num_tensors == 0 || total_params == 0) {
        return;
    }

    constexpr int BLOCK_SIZE = ADAMW8BIT_BLOCK_SIZE_INTERNAL;
    constexpr int N_PER_TH = ADAMW8BIT_NUM_PER_THREAD;

    int total_blocks = (int)div_ceil(total_params, (size_t)BLOCK_SIZE);
    int grid_size = std::min(total_blocks, 256);
    int threads_per_block = BLOCK_SIZE / N_PER_TH;

    kAdamW8bitMultiTensor<nv_bfloat16, BLOCK_SIZE, N_PER_TH>
        <<<grid_size, threads_per_block, 0, stream>>>(
            params, grads, sizes, num_tensors,
            state1, state2, absmax1, absmax2, state_offsets,
            beta1, beta2, eps, step, lr,
            const_cast<float*>(quantiles1), const_cast<float*>(quantiles2),
            weight_decay, opt_params, opt_step, gnorm_scale
        );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace optimizers
