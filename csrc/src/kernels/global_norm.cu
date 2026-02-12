// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Based on llm.c https://github.com/karpathy/llm.c

/**
 * @file global_norm.cu
 * @brief CUDA kernels for computing global gradient norm and gradient clipping.
 *
 * Implements a two-phase global norm computation:
 * 1. Per-block squared sum reduction (global_norm_squared)
 * 2. Final reduction + sqrt + optional clipping scale (global_norm_sqrt)
 *
 * This split allows accumulating norms across multiple gradient tensors
 * before computing the final norm, which is essential for gradient clipping.
 */

#include <cassert>
#include <cmath>
#include <cstddef>

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>


#include "utilities/utils.h"
#include "kernel_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

/**
 * @brief Device function to compute squared sum for a range of elements.
 *
 * Each thread accumulates squared values using a grid-stride loop, then performs
 * warp-level and block-level reductions using cooperative groups.
 *
 * @tparam T Data type (float or nv_bfloat16).
 * @param[in] data Input array.
 * @param count Number of elements.
 * @return Block-wide sum of squared elements.
 */
template<class T>
__device__ float global_norm_squared_for_range(const T* data, size_t count) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_width = blockDim.x * gridDim.x;
    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];
    }

    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    auto warp = cooperative_groups::tiled_partition<32>(block);
    accumulator = reduce_group_add(warp, accumulator);
    __shared__ float shared_accumulator[32];
    if(warp.thread_rank() == 0) {
        shared_accumulator[warp.meta_group_rank()] = accumulator;
    }
    __syncthreads();
    // block-level reduce
    float total = warp.thread_rank() < warp.meta_group_size() ? shared_accumulator[warp.thread_rank()] : 0.f;
    total = reduce_group_add(warp, total);
    return total;
}

/**
 * @brief CUDA kernel to compute partial squared norm per block.
 *
 * Each block computes a partial sum of squared elements and accumulates it
 * to out[blockIdx.x]. This allows calling the kernel multiple times for
 * different tensors to build up the total squared norm.
 *
 * @note Avoids atomic operations by using per-block output slots, requiring
 *       a follow-up deterministic_sum or global_norm_sqrt call to combine results.
 *
 * @tparam T Data type (float or nv_bfloat16).
 * @param[in,out] out Output array of size grid_size, accumulated in-place.
 * @param[in] data Input array.
 * @param count Number of elements.
 */
template<class T>
__global__ void global_norm_squared_kernel(float* out, const T* data, size_t count) {
    float block_sum = global_norm_squared_for_range(data, count);
    // each block accumulates its partial sum to out[blockIdx]
    // we want to avoid using atomic addition here, so we combine this kernel with another kernel call
    // that sums up the partial block sums
    if(threadIdx.x == 0) {
        out[blockIdx.x] = out[blockIdx.x] + block_sum;
    }
}

/**
 * @brief CUDA kernel for deterministic summation using a single block.
 *
 * Sums all elements deterministically by using a single block, avoiding
 * non-deterministic cross-block reduction. Uses warp shuffles and shared
 * memory for efficient intra-block reduction.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] out Output scalar (single float).
 * @param[in] data Input array.
 * @param count Number of elements to sum.
 */
template<class floatX>
__global__ void deterministic_sum_kernel(float* out, const floatX* data, std::size_t count) {
    assert(gridDim.x == 1);     // only a single block!
    float thread_sum = 0;
    for(size_t index = threadIdx.x; index < count; index += blockDim.x) {
        thread_sum += (float)data[index];
    }

    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    auto warp = cooperative_groups::tiled_partition<32>(block);
    float warp_sum = reduce_group_add(warp, thread_sum);
    __shared__ float shared_accumulator[32];
    if(warp.thread_rank() == 0) {
        shared_accumulator[warp.meta_group_rank()] = warp_sum;
    }
    __syncthreads();
    // block-level reduce
    if(warp.meta_group_rank() == 0) {
        float total = warp.thread_rank() < warp.meta_group_size() ? shared_accumulator[warp.thread_rank()] : 0.f;
        total = reduce_group_add(warp, total);
        if (threadIdx.x == 0) {
            *out = total;
        }
    }
}

/**
 * @brief CUDA kernel to compute final gradient norm, apply token normalization, and gradient clipping.
 *
 * Implements HuggingFace-style loss normalization: normalizes gradients by the total valid
 * (non-masked) token count across the entire gradient accumulation cycle. This ensures each
 * TOKEN contributes equally to the gradient update, not each accumulation step.
 *
 * Takes the accumulated squared norm from out[0], computes:
 * 1. Token scale: 1.0 / valid_token_count (HuggingFace-style normalization)
 * 2. Gradient norm after token scaling
 * 3. Clipping scale if norm exceeds grad_clip
 *
 * Results are written to:
 * - out[0]: The actual gradient norm (after token scaling, for logging)
 * - out[1]: Total scale to apply to gradients (token_scale * clip_scale)
 * - out_cpu: The actual norm value (for logging/monitoring)
 *
 * @param[in,out] out GPU buffer where out[0] = squared norm, out[1] = scale output.
 * @param[out] out_cpu Pinned CPU memory for the computed norm value.
 * @param grad_clip Maximum allowed gradient norm (0 or negative to disable clipping).
 * @param valid_token_count Accumulated count of non-masked tokens across all micro-batches.
 * @param total_tokens Unused (kept for API compatibility).
 */
__global__ void global_norm_sqrt_kernel(float* out, float* out_cpu, float grad_clip,
                                        const int* valid_token_count, float total_tokens) {
    (void)total_tokens;  // Unused in HuggingFace-style normalization

    float n_squared = out[0];
    float norm = std::sqrt(n_squared);

    // HuggingFace-style normalization: scale gradients by 1 / valid_token_count
    // This ensures each TOKEN contributes equally, not each accumulation step.
    float token_scale = 1.0f;
    if (valid_token_count) {
        int valid = *valid_token_count;
        if (valid > 0) {
            token_scale = 1.0f / static_cast<float>(valid);
        }
    }

    // The norm we report is the scaled norm (what the actual gradient magnitude will be)
    float scaled_norm = norm * token_scale;

    // Apply gradient clipping against the scaled norm
    float clip_scale = 1.0f;
    if (grad_clip > 0.f && scaled_norm > 0.f && scaled_norm > grad_clip) {
        clip_scale = grad_clip / scaled_norm;
    }

    // Total scale combines token normalization and clipping
    float total_scale = token_scale * clip_scale;

    out[0] = scaled_norm;  // Report the scaled norm (actual gradient magnitude)
    out[1] = total_scale;  // Total scale to apply to gradients
    if (out_cpu) {
        *out_cpu = scaled_norm;
    }
}


// ----------------------------------------------------------------------------
// Prescaled norm kernels (overflow-safe for large BF16 gradients)
//
// When BF16 gradients have very large values (e.g., ~1e18 from Mamba backward pass),
// sum(g^2) overflows FP32. The prescaled approach uses:
//   1. Find amax = max(|g_i|) across all gradients
//   2. Compute sum((g_i / amax)^2) — each term <= 1.0, no overflow
//   3. norm = amax * sqrt(sum)

/**
 * @brief Atomic max for non-negative floats using integer CAS.
 *
 * For non-negative IEEE 754 floats, the integer representation is
 * monotonically increasing, so integer atomicMax gives float max.
 */
__device__ void atomicMaxFloat(float* addr, float val) {
    if (val <= 0.f) return;
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int val_as_int = __float_as_int(val);
    int old = *addr_as_int;
    while (val_as_int > old) {
        old = atomicCAS(addr_as_int, old, val_as_int);
    }
}

/**
 * @brief Kernel to find max absolute value across a tensor.
 *
 * Uses warp/block reduction + atomicMax. Output must be zeroed before first call.
 */
template<class T>
__global__ void global_amax_kernel(float* out, const T* data, size_t count) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_width = blockDim.x * gridDim.x;
    float local_max = 0.f;
    for (size_t i = index; i < count; i += grid_width) {
        float val = fabsf((float)data[i]);
        if (val > local_max) local_max = val;
    }

    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    auto warp = cooperative_groups::tiled_partition<32>(block);
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        float other = warp.shfl_xor(local_max, offset);
        if (other > local_max) local_max = other;
    }

    __shared__ float shared_max[32];
    if (warp.thread_rank() == 0) {
        shared_max[warp.meta_group_rank()] = local_max;
    }
    __syncthreads();

    if (warp.meta_group_rank() == 0) {
        float val = warp.thread_rank() < warp.meta_group_size() ? shared_max[warp.thread_rank()] : 0.f;
        for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
            float other = warp.shfl_xor(val, offset);
            if (other > val) val = other;
        }
        if (threadIdx.x == 0) {
            atomicMaxFloat(out, val);
        }
    }
}

/**
 * @brief Prescaled squared norm kernel.
 *
 * Computes sum((x * prescale)^2) where prescale is read from device memory.
 * Each prescaled element has |value| <= 1.0, preventing FP32 overflow.
 */
template<class T>
__global__ void global_norm_squared_prescaled_kernel(float* out, const T* data, size_t count,
                                                      const float* prescale_device) {
    const float prescale = *prescale_device;
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_width = blockDim.x * gridDim.x;
    float accumulator = 0.f;
    for (size_t i = index; i < count; i += grid_width) {
        float val = (float)data[i] * prescale;
        accumulator += val * val;
    }

    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    auto warp = cooperative_groups::tiled_partition<32>(block);
    accumulator = reduce_group_add(warp, accumulator);
    __shared__ float shared_accumulator[32];
    if (warp.thread_rank() == 0) {
        shared_accumulator[warp.meta_group_rank()] = accumulator;
    }
    __syncthreads();
    float total = warp.thread_rank() < warp.meta_group_size() ? shared_accumulator[warp.thread_rank()] : 0.f;
    total = reduce_group_add(warp, total);
    if (threadIdx.x == 0) {
        out[blockIdx.x] = out[blockIdx.x] + total;
    }
}

/**
 * @brief Computes prescale = 1/amax on device.
 */
__global__ void compute_prescale_kernel(float* prescale_out, const float* amax_in) {
    float amax = *amax_in;
    *prescale_out = (amax > 1e-30f) ? (1.0f / amax) : 1.0f;
}

/**
 * @brief Final norm computation with prescale correction.
 *
 * The accumulated squared norm was prescaled: sum((x/amax)^2).
 * True norm = amax * sqrt(sum).
 */
__global__ void global_norm_sqrt_prescaled_kernel(float* out, float* out_cpu, float grad_clip,
                                                   const int* valid_token_count, float total_tokens,
                                                   const float* amax_device) {
    (void)total_tokens;

    float n_squared_prescaled = out[0];
    float amax = *amax_device;
    float norm = amax * std::sqrt(n_squared_prescaled);

    float token_scale = 1.0f;
    if (valid_token_count) {
        int valid = *valid_token_count;
        if (valid > 0) {
            token_scale = 1.0f / static_cast<float>(valid);
        }
    }

    float scaled_norm = norm * token_scale;

    float clip_scale = 1.0f;
    if (grad_clip > 0.f && scaled_norm > 0.f && scaled_norm > grad_clip) {
        clip_scale = grad_clip / scaled_norm;
    }

    float total_scale = token_scale * clip_scale;

    out[0] = scaled_norm;
    out[1] = total_scale;
    if (out_cpu) {
        *out_cpu = scaled_norm;
    }
}


// ----------------------------------------------------------------------------
// kernel launcher

/**
 * @brief Determines the maximum number of partial block sums needed.
 *
 * Calculates the grid size used by global_norm_squared kernels, which equals
 * the required size of the output buffer for partial sums.
 *
 * @note Must be kept in sync with global_norm_squared kernel launch parameters.
 *
 * @param dp CUDA device properties.
 * @return Maximum number of blocks (and thus partial sums).
 */
int get_max_num_block_sums(const cudaDeviceProp& dp) {
    // NOTE: this needs to be kept in sync with `global_norm_squared` below.
    const int block_size = 512;
    // launch just enough blocks to fill the grid. deliberately no DIV_CEIL.
    // having one block less than possible is a tiny performance hit, having
    // one block too many is catastrophic, since it only can start once all the other
    // blocks finish. anyway, I think cuda_threads_per_SM should be a multiple of 512
    // on all gpus, so the division really is going to be exact.
    const int grid_size = dp.maxThreadsPerMultiProcessor * dp.multiProcessorCount / block_size;

    return grid_size;
}

/**
 * @brief Template implementation for computing partial squared norms.
 *
 * Launches the global_norm_squared_kernel with adaptive grid size based on
 * tensor size and device capabilities. Results are accumulated into out[blockIdx].
 *
 * @tparam T Data type (float or nv_bfloat16).
 * @param[in,out] out Output buffer of size get_max_num_block_sums(), accumulated.
 * @param[in] values Input tensor.
 * @param count Number of elements.
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
template<typename T>
void global_norm_squared_imp(float* out, const T* values, size_t count, const cudaDeviceProp& dp, cudaStream_t stream) {
    // out points to an array of get_max_num_block_sums elements
    const int block_size = 512;
    const int max_grid_size = get_max_num_block_sums(dp);

    // for tiny tensors, using a device-wide grid is a waste of resources.
    const int max_useful_blocks = div_ceil(count, (size_t)block_size);
    const int grid_size = std::min(max_grid_size, max_useful_blocks);
    assert(grid_size > 0);      // gives a better error than letting the call below fail

    global_norm_squared_kernel<<<grid_size, block_size, 0, stream>>>(out, values, count);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Accumulates squared norm of FP32 values into output buffer.
 *
 * @param[in,out] out Output buffer, accumulated in-place.
 * @param[in] values Input FP32 tensor.
 * @param count Number of elements.
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void global_norm_squared(float* out, const float* values, size_t count, const cudaDeviceProp& dp, cudaStream_t stream) {
    global_norm_squared_imp(out, values, count, dp, stream);
}

/**
 * @brief Accumulates squared norm of BF16 values into output buffer.
 *
 * @param[in,out] out Output buffer, accumulated in-place.
 * @param[in] values Input BF16 tensor.
 * @param count Number of elements.
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void global_norm_squared(float* out, const nv_bfloat16* values, size_t count, const cudaDeviceProp& dp, cudaStream_t stream) {
    global_norm_squared_imp(out, values, count, dp, stream);
}

/**
 * @brief Computes final gradient norm and clipping scale from accumulated squared sums.
 *
 * Call this after all global_norm_squared calls have accumulated partial sums.
 * A deterministic_sum is typically called first to reduce partial sums to out[0].
 *
 * @param[in,out] out GPU buffer: out[0] = squared norm input, out[1] = scale output.
 * @param[out] out_cpu Pinned CPU memory for the computed norm value.
 * @param grad_clip Maximum gradient norm for clipping (0 to disable).
 * @param dp CUDA device properties (unused but kept for API consistency).
 * @param stream CUDA stream.
 */
void global_norm_sqrt(float* out, float* out_cpu, float grad_clip,
                      const int* valid_token_count, float total_tokens,
                      const cudaDeviceProp& dp, cudaStream_t stream) {
    global_norm_sqrt_kernel<<<1,  1, 0, stream>>>(out, out_cpu, grad_clip, valid_token_count, total_tokens);
}

/**
 * @brief Deterministically sums FP32 values into a single output.
 *
 * Uses a single block for deterministic reduction order.
 *
 * @param[out] out Output scalar.
 * @param[in] values Input FP32 array.
 * @param count Number of elements.
 * @param stream CUDA stream.
 */
void deterministic_sum(float* out, const float* values, std::size_t count, cudaStream_t stream) {
    deterministic_sum_kernel<<<1, 512, 0, stream>>>(out, values, count);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Deterministically sums BF16 values into a single FP32 output.
 *
 * Uses a single block for deterministic reduction order.
 * Values are converted to FP32 during accumulation.
 *
 * @param[out] out Output scalar in FP32.
 * @param[in] values Input BF16 array.
 * @param count Number of elements.
 * @param stream CUDA stream.
 */
void deterministic_sum(float* out, const nv_bfloat16* values, std::size_t count, cudaStream_t stream) {
    deterministic_sum_kernel<<<1, 512, 0, stream>>>(out, values, count);
    CUDA_CHECK(cudaGetLastError());
}

// --- Prescaled norm wrappers ---

template<typename T>
void global_amax_imp(float* out, const T* values, size_t count, const cudaDeviceProp& dp, cudaStream_t stream) {
    const int block_size = 512;
    const int max_grid_size = get_max_num_block_sums(dp);
    const int max_useful_blocks = div_ceil(count, (size_t)block_size);
    const int grid_size = std::min(max_grid_size, max_useful_blocks);
    if (grid_size <= 0) return;
    global_amax_kernel<<<grid_size, block_size, 0, stream>>>(out, values, count);
    CUDA_CHECK(cudaGetLastError());
}

void global_amax(float* out, const float* values, size_t count, const cudaDeviceProp& dp, cudaStream_t stream) {
    global_amax_imp(out, values, count, dp, stream);
}

void global_amax(float* out, const nv_bfloat16* values, size_t count, const cudaDeviceProp& dp, cudaStream_t stream) {
    global_amax_imp(out, values, count, dp, stream);
}

void compute_prescale(float* prescale_out, const float* amax_in, cudaStream_t stream) {
    compute_prescale_kernel<<<1, 1, 0, stream>>>(prescale_out, amax_in);
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void global_norm_squared_prescaled_imp(float* out, const T* values, size_t count, const float* prescale_device,
                                        const cudaDeviceProp& dp, cudaStream_t stream) {
    const int block_size = 512;
    const int max_grid_size = get_max_num_block_sums(dp);
    const int max_useful_blocks = div_ceil(count, (size_t)block_size);
    const int grid_size = std::min(max_grid_size, max_useful_blocks);
    if (grid_size <= 0) return;
    global_norm_squared_prescaled_kernel<<<grid_size, block_size, 0, stream>>>(out, values, count, prescale_device);
    CUDA_CHECK(cudaGetLastError());
}

void global_norm_squared_prescaled(float* out, const float* values, size_t count, const float* prescale_device,
                                    const cudaDeviceProp& dp, cudaStream_t stream) {
    global_norm_squared_prescaled_imp(out, values, count, prescale_device, dp, stream);
}

void global_norm_squared_prescaled(float* out, const nv_bfloat16* values, size_t count, const float* prescale_device,
                                    const cudaDeviceProp& dp, cudaStream_t stream) {
    global_norm_squared_prescaled_imp(out, values, count, prescale_device, dp, stream);
}

void global_norm_sqrt_prescaled(float* out, float* out_cpu, float grad_clip,
                                 const int* valid_token_count, float total_tokens,
                                 const float* amax_device,
                                 const cudaDeviceProp& dp, cudaStream_t stream) {
    (void)dp;
    global_norm_sqrt_prescaled_kernel<<<1, 1, 0, stream>>>(out, out_cpu, grad_clip, valid_token_count, total_tokens, amax_device);
    CUDA_CHECK(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Multi-tensor fused kernels
//
// Iterate an array of tensor pointers inside a single kernel, reducing kernel
// launch overhead from O(N_tensors) to O(1). Each tensor has a dtype flag
// (0=FP32, 1=BF16) to handle mixed-precision gradient tensors.

/**
 * @brief Fused amax across multiple tensors.
 *
 * Iterates all tensors in a single kernel. Each block processes elements
 * across all tensors via grid-stride loop, then atomicMax to a single output.
 */
template<int BLOCK_SIZE>
__global__ void global_amax_multi_tensor_kernel(
    float* __restrict__ amax_out,
    const void* const* __restrict__ data_ptrs,
    const size_t* __restrict__ sizes,
    const int* __restrict__ dtype_flags,
    int num_tensors)
{
    const size_t grid_width = static_cast<size_t>(blockDim.x) * gridDim.x;
    float local_max = 0.f;

    for (int t = 0; t < num_tensors; ++t) {
        const size_t count = sizes[t];
        if (dtype_flags[t] == 1) {
            const auto* data = static_cast<const nv_bfloat16*>(data_ptrs[t]);
            for (size_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < count; i += grid_width) {
                float val = fabsf((float)data[i]);
                if (val > local_max) local_max = val;
            }
        } else {
            const auto* data = static_cast<const float*>(data_ptrs[t]);
            for (size_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < count; i += grid_width) {
                float val = fabsf(data[i]);
                if (val > local_max) local_max = val;
            }
        }
    }

    // Warp reduction (max)
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    auto warp = cooperative_groups::tiled_partition<32>(block);
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        float other = warp.shfl_xor(local_max, offset);
        if (other > local_max) local_max = other;
    }

    __shared__ float shared_max[32];
    if (warp.thread_rank() == 0) {
        shared_max[warp.meta_group_rank()] = local_max;
    }
    __syncthreads();

    if (warp.meta_group_rank() == 0) {
        float val = warp.thread_rank() < warp.meta_group_size() ? shared_max[warp.thread_rank()] : 0.f;
        for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
            float other = warp.shfl_xor(val, offset);
            if (other > val) val = other;
        }
        if (threadIdx.x == 0) {
            atomicMaxFloat(amax_out, val);
        }
    }
}

/**
 * @brief Fused prescaled squared norm across multiple tensors.
 *
 * Computes sum((x * prescale)^2) across ALL tensors, writing per-block partial
 * sums to out[blockIdx.x]. Downstream deterministic_sum + global_norm_sqrt_prescaled
 * are unchanged.
 */
template<int BLOCK_SIZE>
__global__ void global_norm_squared_prescaled_multi_tensor_kernel(
    float* __restrict__ out,
    const void* const* __restrict__ data_ptrs,
    const size_t* __restrict__ sizes,
    const int* __restrict__ dtype_flags,
    int num_tensors,
    const float* __restrict__ prescale_device)
{
    const float prescale = *prescale_device;
    const size_t grid_width = static_cast<size_t>(blockDim.x) * gridDim.x;
    float accumulator = 0.f;

    for (int t = 0; t < num_tensors; ++t) {
        const size_t count = sizes[t];
        if (dtype_flags[t] == 1) {
            const auto* data = static_cast<const nv_bfloat16*>(data_ptrs[t]);
            for (size_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < count; i += grid_width) {
                float val = (float)data[i] * prescale;
                accumulator += val * val;
            }
        } else {
            const auto* data = static_cast<const float*>(data_ptrs[t]);
            for (size_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < count; i += grid_width) {
                float val = data[i] * prescale;
                accumulator += val * val;
            }
        }
    }

    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    auto warp = cooperative_groups::tiled_partition<32>(block);
    accumulator = reduce_group_add(warp, accumulator);
    __shared__ float shared_accumulator[32];
    if (warp.thread_rank() == 0) {
        shared_accumulator[warp.meta_group_rank()] = accumulator;
    }
    __syncthreads();
    float total = warp.thread_rank() < warp.meta_group_size() ? shared_accumulator[warp.thread_rank()] : 0.f;
    total = reduce_group_add(warp, total);
    if (threadIdx.x == 0) {
        out[blockIdx.x] = out[blockIdx.x] + total;
    }
}

/**
 * @brief Fused squared norm across multiple tensors (no prescale).
 *
 * For dense gradient norm computation. Accumulates sum(x^2) across all tensors.
 */
template<int BLOCK_SIZE>
__global__ void global_norm_squared_multi_tensor_kernel(
    float* __restrict__ out,
    const void* const* __restrict__ data_ptrs,
    const size_t* __restrict__ sizes,
    const int* __restrict__ dtype_flags,
    int num_tensors)
{
    const size_t grid_width = static_cast<size_t>(blockDim.x) * gridDim.x;
    float accumulator = 0.f;

    for (int t = 0; t < num_tensors; ++t) {
        const size_t count = sizes[t];
        if (dtype_flags[t] == 1) {
            const auto* data = static_cast<const nv_bfloat16*>(data_ptrs[t]);
            for (size_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < count; i += grid_width) {
                accumulator += (float)data[i] * (float)data[i];
            }
        } else {
            const auto* data = static_cast<const float*>(data_ptrs[t]);
            for (size_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < count; i += grid_width) {
                accumulator += data[i] * data[i];
            }
        }
    }

    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    auto warp = cooperative_groups::tiled_partition<32>(block);
    accumulator = reduce_group_add(warp, accumulator);
    __shared__ float shared_accumulator[32];
    if (warp.thread_rank() == 0) {
        shared_accumulator[warp.meta_group_rank()] = accumulator;
    }
    __syncthreads();
    float total = warp.thread_rank() < warp.meta_group_size() ? shared_accumulator[warp.thread_rank()] : 0.f;
    total = reduce_group_add(warp, total);
    if (threadIdx.x == 0) {
        out[blockIdx.x] = out[blockIdx.x] + total;
    }
}

// --- Multi-tensor launchers ---

void global_amax_multi_tensor(float* amax_out, const void* const* data_ptrs,
                               const size_t* sizes, const int* dtype_flags,
                               int num_tensors, const cudaDeviceProp& dp, cudaStream_t stream) {
    if (num_tensors == 0) return;
    const int grid_size = get_max_num_block_sums(dp);
    global_amax_multi_tensor_kernel<512><<<grid_size, 512, 0, stream>>>(
        amax_out, data_ptrs, sizes, dtype_flags, num_tensors);
    CUDA_CHECK(cudaGetLastError());
}

void global_norm_squared_prescaled_multi_tensor(float* out, const void* const* data_ptrs,
                                                 const size_t* sizes, const int* dtype_flags,
                                                 int num_tensors, const float* prescale_device,
                                                 const cudaDeviceProp& dp, cudaStream_t stream) {
    if (num_tensors == 0) return;
    const int grid_size = get_max_num_block_sums(dp);
    global_norm_squared_prescaled_multi_tensor_kernel<512><<<grid_size, 512, 0, stream>>>(
        out, data_ptrs, sizes, dtype_flags, num_tensors, prescale_device);
    CUDA_CHECK(cudaGetLastError());
}

void global_norm_squared_multi_tensor(float* out, const void* const* data_ptrs,
                                       const size_t* sizes, const int* dtype_flags,
                                       int num_tensors, const cudaDeviceProp& dp, cudaStream_t stream) {
    if (num_tensors == 0) return;
    const int grid_size = get_max_num_block_sums(dp);
    global_norm_squared_multi_tensor_kernel<512><<<grid_size, 512, 0, stream>>>(
        out, data_ptrs, sizes, dtype_flags, num_tensors);
    CUDA_CHECK(cudaGetLastError());
}
