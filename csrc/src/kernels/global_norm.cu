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
 * @brief CUDA kernel to compute final gradient norm and clipping scale.
 *
 * Takes the accumulated squared norm from out[0], computes the square root,
 * and determines the gradient clipping scale factor. Results are written to:
 * - out[1]: Clipping scale (grad_clip/norm if norm > grad_clip, else 1.0)
 * - out_cpu: The actual norm value (for logging/monitoring)
 *
 * @param[in,out] out GPU buffer where out[0] = squared norm, out[1] = scale output.
 * @param[out] out_cpu Pinned CPU memory for the computed norm value.
 * @param grad_clip Maximum allowed gradient norm (0 or negative to disable clipping).
 */
__global__ void global_norm_sqrt_kernel(float* out, float* out_cpu, float grad_clip,
                                        const int* valid_token_count, float total_tokens) {
    float n_squared = out[0];
    float norm = std::sqrt(n_squared);
    // Token-scale to convert gradients from "mean over (B*T*grad_accum)" to "mean over valid tokens".
    // If masks are not present, valid_token_count ~= total_tokens and token_scale ~= 1.
    float token_scale = 1.f;
    if (valid_token_count && total_tokens > 0.f) {
        int valid = *valid_token_count;
        if (valid > 0) {
            token_scale = total_tokens / static_cast<float>(valid);
        }
    }

    // We conceptually want to apply: g' = token_scale * g, then clip so that ||g'|| <= grad_clip.
    // The combined multiplier is:
    //   if (token_scale * norm > grad_clip)  =>  grad_clip / norm
    //   else                                =>  token_scale
    float scaled_norm = norm * token_scale;
    float total_scale = token_scale;
    if (grad_clip > 0.f && norm > 0.f && scaled_norm > grad_clip) {
        total_scale = grad_clip / norm;
    }
    out[0] = norm;
    out[1] = total_scale;
    // Log the raw (unscaled) norm so masks don't create large log spikes.
    if (out_cpu) {
        *out_cpu = norm;
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
