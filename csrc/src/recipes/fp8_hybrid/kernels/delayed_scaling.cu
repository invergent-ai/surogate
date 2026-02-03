// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file delayed_scaling.cu
 * @brief CUDA kernels for FP8 delayed scaling recipe
 *
 * Implements TransformerEngine-style delayed scaling where:
 * 1. Scale factors are computed from previous iteration(s) abs_max values
 * 2. Amax history is maintained in a rolling buffer
 * 3. New scales are computed by reducing over the history window
 *
 * Reference: TransformerEngine delayed_scaling.cu
 */

#include <cfloat>
#include <cmath>

#include <cuda_runtime.h>

#include "kernels/kernel_utils.cuh"
#include "runtime/core/fp8_scaling_config.h"
#include "runtime/core/fp8_scaling_state.h"

namespace {

constexpr int BLOCK_SIZE = 256;

/**
 * @brief CUDA kernel to update amax history and compute new FP8 scaling factors
 *
 * TransformerEngine-style delayed scaling: scales are computed from PREVIOUS iterations.
 *
 * Each block handles one quantizer. The kernel:
 * 1. Reads the current history (slot [0] contains last iteration's amax)
 * 2. Computes effective amax using the specified algorithm (max or most_recent)
 * 3. Computes new scale = scaled_max / effective_amax (or keeps previous scale if invalid)
 * 4. Rolls the history buffer forward (hist[i] = hist[i+1])
 * 5. Zeros out slot [0] for the next iteration to record into
 *
 * The next iteration will record its amax into slot [0] during forward/backward,
 * and this kernel will use it in the FOLLOWING update.
 *
 * @param amax_history History buffer [history_len, num_quantizers] (in-place update)
 * @param scales Scale factors [num_quantizers] (input/output - may keep previous)
 * @param recorded_amaxes Unused - kept for API compatibility
 * @param history_len Length of history window
 * @param num_quantizers Number of quantizers (grid size)
 * @param amax_compute_algo 0 = MAX, 1 = MOST_RECENT
 * @param scaled_max_e4m3 = fp8_max * 2^(-margin) for E4M3 (448 * 2^(-margin))
 * @param scaled_max_e5m2 = fp8_max * 2^(-margin) for E5M2 (57344 * 2^(-margin))
 * @param amax_epsilon Optional floor for amax values (0.0 = no floor, not used by TE)
 */
__global__ void delayed_scaling_update_kernel(
    float* __restrict__ amax_history,
    float* __restrict__ scales,
    const float* __restrict__ recorded_amaxes,
    int history_len,
    int num_quantizers,
    int amax_compute_algo,  // 0 = MAX, 1 = MOST_RECENT
    float scaled_max_e4m3,
    float scaled_max_e5m2,
    float amax_epsilon
) {
    const int qid = blockIdx.x;  // Quantizer index
    if (qid >= num_quantizers) return;

    const int tid = threadIdx.x;

    // Column stride in history buffer (column-major for coalesced access per quantizer)
    // History layout: [history_len, num_quantizers] row-major
    // So each quantizer's history is strided by num_quantizers
    float* hist = amax_history + qid;
    const int stride = num_quantizers;

    // Get the newly recorded amax from this iteration's forward pass
    const float new_amax = recorded_amaxes[qid];

    // Compute max across history while rolling (TransformerEngine approach)
    __shared__ float shared_max[BLOCK_SIZE];
    float thread_max = 0.0f;

    // Roll history forward: hist[i] = hist[i+1], and insert new_amax at slot [0]
    // Process forward in chunks to avoid overwriting values we still need
    for (int off = 0; off < history_len; off += blockDim.x) {
        const int i = off + tid;
        float val = 0.0f;
        if (i < history_len) {
            // Read value: shift from next slot, or use new_amax for the last slot
            val = (i < history_len - 1) ? hist[(i + 1) * stride] : new_amax;
            thread_max = fmaxf(thread_max, val);
        }
        __syncthreads();  // Ensure reads complete before writes

        if (i < history_len) {
            // Write rolled value: slot[0] gets new_amax, others shift down
            hist[i * stride] = (i == 0) ? new_amax : val;
        }
    }

    // Compute effective amax based on algorithm
    float effective_amax;
    if (amax_compute_algo == 1) {  // MOST_RECENT
        effective_amax = new_amax;
    } else {  // MAX (default)
        // Reduce thread_max across block
        shared_max[tid] = thread_max;
        __syncthreads();

        // Tree reduction
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
            }
            __syncthreads();
        }
        effective_amax = shared_max[0];
    }

    // Compute new scale (only thread 0)
    if (tid == 0) {
        // Select scaled_max based on quantizer type
        // Forward quantizers (0-3 within each layer) use E4M3, backward quantizers (4-7) use E5M2
        // With per-layer quantizers, check the quantizer type within the layer (qid % 8)
        const int qtype = qid % 8;  // 0-7: type within layer
        float scaled_max = (qtype < 4) ? scaled_max_e4m3 : scaled_max_e5m2;

        float scale;
        // TransformerEngine behavior: keep previous scale if amax is invalid
        if (isfinite(effective_amax) && effective_amax > 0.0f) {
            // Apply epsilon floor if configured (optional feature, not in TE)
            float clamped_amax = effective_amax;
            if (amax_epsilon > 0.0f && clamped_amax < amax_epsilon) {
                clamped_amax = amax_epsilon;
            }

            // Compute scale with regular division (TE doesn't use __fdiv_rn)
            scale = scaled_max / clamped_amax;

            // Clamp to FLT_MAX if scale becomes infinite (amax too small)
            if (isinf(scale)) {
                scale = FLT_MAX;
            }
        } else {
            // Invalid amax: keep previous scale (TransformerEngine behavior)
            scale = scales[qid];
        }

        scales[qid] = scale;
    }
}

/**
 * @brief Kernel to initialize scales to 1.0 and zero history/recorded amaxes
 */
__global__ void reset_fp8_scaling_state_kernel(
    float* __restrict__ amax_history,
    float* __restrict__ scales,
    float* __restrict__ recorded_amaxes,
    int history_len,
    int num_quantizers
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Zero amax history
    const int history_size = history_len * num_quantizers;
    if (idx < history_size) {
        amax_history[idx] = 0.0f;
    }

    // Initialize scales to 1.0 and zero recorded amaxes
    if (idx < num_quantizers) {
        scales[idx] = 1.0f;
        recorded_amaxes[idx] = 0.0f;
    }
}

} // anonymous namespace

namespace modules {

// FP8ScalingState implementation

FP8ScalingState::FP8ScalingState(
    const FP8ScalingConfig& config,
    const std::shared_ptr<TensorAllocator>& allocator,
    int device_id,
    int num_layers
) : mConfig(config), mDeviceId(device_id), mNumLayers(num_layers),
    mNumQuantizers(get_total_quantizers(num_layers)) {
    const long history_len = config.amax_history_len;
    const long num_quantizers = mNumQuantizers;

    // Allocate tensors
    mAmaxHistory = allocator->allocate(
        ETensorDType::FP32, "fp8_amax_history",
        EAllocationType::ON_DEVICE,
        {history_len, num_quantizers});

    mScales = allocator->allocate(
        ETensorDType::FP32, "fp8_scales",
        EAllocationType::ON_DEVICE,
        {num_quantizers});

    mRecordedAmaxes = allocator->allocate(
        ETensorDType::FP32, "fp8_recorded_amaxes",
        EAllocationType::ON_DEVICE,
        {num_quantizers});

    // Initialize to default values (will be done on first use)
}

void FP8ScalingState::reset(cudaStream_t stream) {
    const int history_len = mConfig.amax_history_len;
    const int num_quantizers = mNumQuantizers;
    const int total_elems = history_len * num_quantizers + 2 * num_quantizers;

    const int block_size = 256;
    const int grid_size = (total_elems + block_size - 1) / block_size;

    reset_fp8_scaling_state_kernel<<<grid_size, block_size, 0, stream>>>(
        mAmaxHistory.get<float>(),
        mScales.get<float>(),
        mRecordedAmaxes.get<float>(),
        history_len,
        num_quantizers
    );
    CUDA_CHECK(cudaGetLastError());
}

void FP8ScalingState::zero_recorded_amaxes(cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(
        mRecordedAmaxes.Data,
        0,
        mNumQuantizers * sizeof(float),
        stream));
}

void delayed_scaling_update(FP8ScalingState& state, cudaStream_t stream) {
    const auto& config = state.config();
    const int history_len = config.amax_history_len;
    const int num_quantizers = state.num_quantizers();

    // Compute scaled_max values
    const float scaled_max_e4m3 = config.get_scaled_max(ETensorDType::FP8_E4M3);
    const float scaled_max_e5m2 = config.get_scaled_max(ETensorDType::FP8_E5M2);

    // Convert algorithm enum to int for kernel
    const int algo = (config.amax_compute_algo == AmaxComputeAlgo::MOST_RECENT) ? 1 : 0;

    // Launch kernel: one block per quantizer
    const int grid_size = num_quantizers;
    const int block_size = BLOCK_SIZE;

    delayed_scaling_update_kernel<<<grid_size, block_size, 0, stream>>>(
        state.amax_history().get<float>(),
        state.scales().get<float>(),
        state.recorded_amaxes().get<float>(),
        history_len,
        num_quantizers,
        algo,
        scaled_max_e4m3,
        scaled_max_e5m2,
        config.amax_epsilon
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace modules
