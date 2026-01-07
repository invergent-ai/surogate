// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// NorMuon optimizer CUDA kernels

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

#include "normuon.h"
#include "polar_express.h"
#include "kernels/kernel_utils.cuh"

namespace optimizers {

// ----------------------------------------------------------------------------
// Constants

constexpr int NORMUON_THREADS = 256;
constexpr int NORMUON_NUM_PER_THREAD = 8;

// ----------------------------------------------------------------------------
// Quantization (reusing AdamW pattern)

/**
 * @brief Fast 2D quantization using quadrant pivots and binary search (signed)
 */
__device__ __forceinline__ unsigned char
quantize_signed(float* __restrict__ quadrants, float* __restrict__ const smem_code, float x) {
    int pivot = 127;
    int upper_pivot = 255;
    int lower_pivot = 0;

    float lower = -1.0f;
    float upper = 1.0f;
    float midpoint;
    float val = quadrants[1];
    int local_pivot = 1;
    int offset = 1;

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
// Quantiles creation (same as AdamW signed quantiles)

void create_normuon_quantiles(float* code) {
    // Create signed dynamic quantization map for [-1, 1]
    // Using normal quantile function (matches bitsandbytes)
    for (int i = 0; i < 256; ++i) {
        // Map index to quantile value
        float p = (i + 0.5f) / 256.0f;
        // Simple linear mapping for now (could use erfinv for normal quantiles)
        code[i] = 2.0f * p - 1.0f;
    }
}

// ----------------------------------------------------------------------------
// Momentum state initialization

__global__ void kInitNormuonState(
    unsigned char* __restrict__ momentum_state,
    float* __restrict__ momentum_absmax,
    size_t n,
    size_t num_blocks
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize momentum state to middle value (representing 0)
    if (idx < n) {
        momentum_state[idx] = 127;  // Middle of signed range
    }

    // Initialize absmax to small positive value
    if (idx < num_blocks) {
        momentum_absmax[idx] = 1e-7f;
    }
}

void init_normuon_momentum_state(
    unsigned char* momentum_state,
    float* momentum_absmax,
    size_t n,
    cudaStream_t stream
) {
    constexpr size_t BLOCK_SIZE = NORMUON_BLOCK_SIZE;
    size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int threads = NORMUON_THREADS;
    int blocks = (max(n, num_blocks) + threads - 1) / threads;

    kInitNormuonState<<<blocks, threads, 0, stream>>>(
        momentum_state, momentum_absmax, n, num_blocks
    );
}

// ----------------------------------------------------------------------------
// 8-bit Momentum Update Kernel

template <typename T, int BLOCK_SIZE, int N_PER_TH>
__launch_bounds__(256, 3)
__global__ void kNormuonMomentum8bit(
    const T* __restrict__ gradient,
    unsigned char* __restrict__ momentum_state,
    T* __restrict__ momentum_out,
    const float beta1,
    const float* __restrict__ quantiles,
    float* __restrict__ absmax,
    const size_t n
) {
    const int block_idx = blockIdx.x;
    const int base = block_idx * BLOCK_SIZE;
    const int thread_offset = threadIdx.x * N_PER_TH;

    // Load quantiles into shared memory
    __shared__ float smem_quantiles[256];
    __shared__ float smem_quadrants[3];
    __shared__ float smem_absmax_new;

    if (threadIdx.x < 256) {
        smem_quantiles[threadIdx.x] = quantiles[threadIdx.x];
    }
    if (threadIdx.x == 0) {
        smem_quadrants[0] = quantiles[64];   // 1/4 point
        smem_quadrants[1] = quantiles[128];  // 1/2 point
        smem_quadrants[2] = quantiles[192];  // 3/4 point
        smem_absmax_new = 0.0f;
    }
    __syncthreads();

    float local_absmax = 0.0f;
    float current_absmax = absmax[block_idx];

    // Process N_PER_TH elements per thread
    float m_vals[N_PER_TH];
    float g_vals[N_PER_TH];

    #pragma unroll
    for (int j = 0; j < N_PER_TH; ++j) {
        size_t idx = base + thread_offset + j;
        if (idx < n) {
            // Load and dequantize momentum
            unsigned char code = momentum_state[idx];
            m_vals[j] = smem_quantiles[code] * current_absmax;

            // Load gradient
            g_vals[j] = static_cast<float>(gradient[idx]);
        } else {
            m_vals[j] = 0.0f;
            g_vals[j] = 0.0f;
        }
    }

    // Update momentum: m = beta1 * m + (1 - beta1) * g
    #pragma unroll
    for (int j = 0; j < N_PER_TH; ++j) {
        m_vals[j] = beta1 * m_vals[j] + (1.0f - beta1) * g_vals[j];
        local_absmax = fmaxf(local_absmax, fabsf(m_vals[j]));
    }

    // Reduce absmax across warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_absmax = fmaxf(local_absmax, __shfl_xor_sync(0xFFFFFFFF, local_absmax, offset));
    }

    // Reduce across block
    if (threadIdx.x % 32 == 0) {
        atomicMax(reinterpret_cast<int*>(&smem_absmax_new), __float_as_int(local_absmax));
    }
    __syncthreads();

    float new_absmax = fmaxf(smem_absmax_new, 1e-7f);

    // Quantize and store
    #pragma unroll
    for (int j = 0; j < N_PER_TH; ++j) {
        size_t idx = base + thread_offset + j;
        if (idx < n) {
            // Normalize and quantize
            float normalized = m_vals[j] / new_absmax;
            unsigned char code = quantize_signed(smem_quadrants, smem_quantiles, normalized);
            momentum_state[idx] = code;

            // Output dequantized momentum
            momentum_out[idx] = static_cast<T>(m_vals[j]);
        }
    }

    // Store new absmax
    if (threadIdx.x == 0) {
        absmax[block_idx] = new_absmax;
    }
}

void normuon_momentum_update_8bit(
    const nv_bfloat16* gradient,
    unsigned char* momentum_state,
    nv_bfloat16* momentum_out,
    size_t n,
    float beta1,
    const float* quantiles,
    float* absmax,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = NORMUON_BLOCK_SIZE;
    constexpr int N_PER_TH = NORMUON_NUM_PER_THREAD;
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    kNormuonMomentum8bit<nv_bfloat16, BLOCK_SIZE, N_PER_TH>
        <<<num_blocks, BLOCK_SIZE / N_PER_TH, 0, stream>>>(
            gradient, momentum_state, momentum_out, beta1, quantiles, absmax, n
        );
}

void normuon_momentum_update_8bit(
    const float* gradient,
    unsigned char* momentum_state,
    float* momentum_out,
    size_t n,
    float beta1,
    const float* quantiles,
    float* absmax,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = NORMUON_BLOCK_SIZE;
    constexpr int N_PER_TH = NORMUON_NUM_PER_THREAD;
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    kNormuonMomentum8bit<float, BLOCK_SIZE, N_PER_TH>
        <<<num_blocks, BLOCK_SIZE / N_PER_TH, 0, stream>>>(
            gradient, momentum_state, momentum_out, beta1, quantiles, absmax, n
        );
}

// ----------------------------------------------------------------------------
// Variance Reduction Kernels

/**
 * @brief Compute row means of squared values
 * Output shape: (batch, M, 1)
 */
__global__ void kVarianceMeanRows(
    const nv_bfloat16* __restrict__ v,
    float* __restrict__ v_mean,
    int M,
    int N
) {
    const int batch_idx = blockIdx.z;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        const nv_bfloat16* v_row = v + batch_idx * M * N + row * N;
        float sum = 0.0f;

        for (int j = 0; j < N; ++j) {
            float val = __bfloat162float(v_row[j]);
            sum += val * val;
        }

        v_mean[batch_idx * M + row] = sum / N;
    }
}

/**
 * @brief Compute column means of squared values
 * Output shape: (batch, 1, N)
 */
__global__ void kVarianceMeanCols(
    const nv_bfloat16* __restrict__ v,
    float* __restrict__ v_mean,
    int M,
    int N
) {
    const int batch_idx = blockIdx.z;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < N) {
        const nv_bfloat16* v_batch = v + batch_idx * M * N;
        float sum = 0.0f;

        for (int i = 0; i < M; ++i) {
            float val = __bfloat162float(v_batch[i * N + col]);
            sum += val * val;
        }

        v_mean[batch_idx * N + col] = sum / M;
    }
}

void compute_variance_mean(
    const nv_bfloat16* v,
    float* v_mean,
    int batch,
    int M,
    int N,
    bool reduce_over_cols,
    cudaStream_t stream
) {
    if (reduce_over_cols) {
        // Reduce over columns -> output (batch, M, 1)
        int blocks = (M + NORMUON_THREADS - 1) / NORMUON_THREADS;
        dim3 grid(blocks, 1, batch);
        kVarianceMeanRows<<<grid, NORMUON_THREADS, 0, stream>>>(v, v_mean, M, N);
    } else {
        // Reduce over rows -> output (batch, 1, N)
        int blocks = (N + NORMUON_THREADS - 1) / NORMUON_THREADS;
        dim3 grid(blocks, 1, batch);
        kVarianceMeanCols<<<grid, NORMUON_THREADS, 0, stream>>>(v, v_mean, M, N);
    }
}

/**
 * @brief Apply variance reduction with EMA update
 *
 * This kernel fuses the Adafactor-style variance normalization:
 * 1. Update EMA: buf = beta2 * buf + (1 - beta2) * mean
 * 2. Compute per-element scale
 * 3. Apply to v
 */
__global__ void kApplyVarianceReductionRows(
    nv_bfloat16* __restrict__ v,
    float* __restrict__ variance_buffer,
    const float* __restrict__ v_mean,
    int M,
    int N,
    float beta2
) {
    const int batch_idx = blockIdx.z;
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    // Get pointer to this batch's variance buffer
    float* buf = variance_buffer + batch_idx * M;
    const float* mean = v_mean + batch_idx * M;
    nv_bfloat16* v_batch = v + batch_idx * M * N;

    // Only the first thread per row updates the EMA
    __shared__ float row_scale;
    if (threadIdx.x == 0) {
        float old_buf = buf[row];
        float new_mean = mean[row];

        // EMA update
        float new_buf = beta2 * old_buf + (1.0f - beta2) * new_mean;
        buf[row] = new_buf;

        // Compute scale: rsqrt(buf) with clamp
        row_scale = rsqrtf(fmaxf(new_buf, 1e-10f));
    }
    __syncthreads();

    // Apply scale to this element
    float val = __bfloat162float(v_batch[row * N + col]);
    v_batch[row * N + col] = __float2bfloat16(val * row_scale);
}

__global__ void kApplyVarianceReductionCols(
    nv_bfloat16* __restrict__ v,
    float* __restrict__ variance_buffer,
    const float* __restrict__ v_mean,
    int M,
    int N,
    float beta2
) {
    const int batch_idx = blockIdx.z;
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float* buf = variance_buffer + batch_idx * N;
    const float* mean = v_mean + batch_idx * N;
    nv_bfloat16* v_batch = v + batch_idx * M * N;

    // All threads in first row update the EMA for their column
    float col_scale;
    if (row == 0) {
        float old_buf = buf[col];
        float new_mean = mean[col];

        float new_buf = beta2 * old_buf + (1.0f - beta2) * new_mean;
        buf[col] = new_buf;

        col_scale = rsqrtf(fmaxf(new_buf, 1e-10f));
    }
    // Need to sync and broadcast - use a simpler approach with per-element read
    col_scale = rsqrtf(fmaxf(buf[col], 1e-10f));

    float val = __bfloat162float(v_batch[row * N + col]);
    v_batch[row * N + col] = __float2bfloat16(val * col_scale);
}

void apply_variance_reduction(
    nv_bfloat16* v,
    float* variance_buffer,
    int batch,
    int M,
    int N,
    float beta2,
    bool reduce_over_cols,
    cudaStream_t stream
) {
    // First compute means
    // Allocate temporary buffer for means (reuse variance_buffer space or separate)
    // For simplicity, we'll compute means in a separate kernel then apply

    // This is a simplified version - in production you'd want to fuse more

    if (reduce_over_cols) {
        // Need M floats for means per batch
        // Use variance_buffer as both EMA and temp mean storage (since they're same size)

        // Compute means into variance_buffer temporarily, then fuse EMA update
        compute_variance_mean(v, variance_buffer, batch, M, N, true, stream);

        // Apply with EMA update
        int blocks_x = (N + NORMUON_THREADS - 1) / NORMUON_THREADS;
        dim3 grid(blocks_x, M, batch);
        kApplyVarianceReductionRows<<<grid, NORMUON_THREADS, 0, stream>>>(
            v, variance_buffer, variance_buffer, M, N, beta2
        );
    } else {
        compute_variance_mean(v, variance_buffer, batch, M, N, false, stream);

        int blocks_x = (N + NORMUON_THREADS - 1) / NORMUON_THREADS;
        dim3 grid(blocks_x, M, batch);
        kApplyVarianceReductionCols<<<grid, NORMUON_THREADS, 0, stream>>>(
            v, variance_buffer, variance_buffer, M, N, beta2
        );
    }
}

// ----------------------------------------------------------------------------
// Cautious Weight Decay Update

template <typename TParam, typename TUpdate>
__global__ void kCautiousWeightDecayUpdate(
    TParam* __restrict__ p,
    const TUpdate* __restrict__ v,
    size_t n,
    float lr,
    float weight_decay
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float p_val = static_cast<float>(p[idx]);
    float v_val = static_cast<float>(v[idx]);

    // Cautious mask: only decay if update and param have same sign
    float mask = (v_val * p_val >= 0.0f) ? 1.0f : 0.0f;

    // p = p - (p * mask * wd * lr) - (v * lr)
    float new_p = p_val - (p_val * mask * weight_decay * lr) - (v_val * lr);

    p[idx] = static_cast<TParam>(new_p);
}

void cautious_weight_decay_update(
    nv_bfloat16* p,
    const nv_bfloat16* v,
    size_t n,
    float lr,
    float weight_decay,
    cudaStream_t stream
) {
    int blocks = (n + NORMUON_THREADS - 1) / NORMUON_THREADS;
    kCautiousWeightDecayUpdate<nv_bfloat16, nv_bfloat16>
        <<<blocks, NORMUON_THREADS, 0, stream>>>(p, v, n, lr, weight_decay);
}

void cautious_weight_decay_update(
    float* p,
    const nv_bfloat16* v,
    size_t n,
    float lr,
    float weight_decay,
    cudaStream_t stream
) {
    int blocks = (n + NORMUON_THREADS - 1) / NORMUON_THREADS;
    kCautiousWeightDecayUpdate<float, nv_bfloat16>
        <<<blocks, NORMUON_THREADS, 0, stream>>>(p, v, n, lr, weight_decay);
}

// ----------------------------------------------------------------------------
// Full NorMuon Update for 2D Weights

void normuon_update_2d(
    cublasHandle_t handle,
    nv_bfloat16* param,
    const nv_bfloat16* gradient,
    unsigned char* momentum_state,
    float* variance_buffer,
    nv_bfloat16* polar_workspace,
    int M,
    int N,
    float lr,
    float beta1,
    float beta2,
    float weight_decay,
    const float* quantiles,
    float* absmax,
    cudaStream_t stream
) {
    size_t n = static_cast<size_t>(M) * N;

    // Step 1: Momentum update (gradient -> momentum smoothed)
    // Use polar_workspace as temporary for momentum output
    nv_bfloat16* momentum_out = polar_workspace;

    normuon_momentum_update_8bit(
        gradient,
        momentum_state,
        momentum_out,
        n,
        beta1,
        quantiles,
        absmax,
        stream
    );

    // Step 2: Polar Express orthogonalization (in-place on momentum_out)
    // Need additional workspace beyond momentum_out
    size_t ws_size = polar_express_workspace_size(1, M, N);
    nv_bfloat16* pe_workspace = momentum_out + n;

    polar_express(
        handle,
        momentum_out,
        pe_workspace,
        1,  // batch = 1
        M,
        N,
        stream
    );

    // Step 3: Variance reduction
    bool reduce_over_cols = (M >= N);
    apply_variance_reduction(
        momentum_out,
        variance_buffer,
        1,  // batch = 1
        M,
        N,
        beta2,
        reduce_over_cols,
        stream
    );

    // Step 4: Cautious weight decay + parameter update
    // Apply learning rate multiplier
    float lr_mult = normuon_lr_multiplier(M, N);
    float effective_lr = lr * lr_mult;

    cautious_weight_decay_update(
        param,
        momentum_out,
        n,
        effective_lr,
        weight_decay,
        stream
    );
}

} // namespace optimizers
