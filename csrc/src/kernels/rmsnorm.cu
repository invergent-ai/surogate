// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Based on llm.c https://github.com/karpathy/llm.c

/**
 * @file rmsnorm.cu
 * @brief CUDA kernels for RMS normalization forward and backward passes.
 *
 * Implements Root Mean Square Layer Normalization used in LLaMA-style models:
 * - Forward: out = (x / sqrt(mean(x^2) + eps)) * weight
 * - Fused variant combines residual addition with normalization
 * - Backward: computes gradients for input and weight parameters
 *
 * Uses warp-level reductions and shared memory for efficiency.
 * Supports optional absolute maximum tracking for quantization.
 */

#include <cassert>

#include "kernel_utils.cuh"
#include "utilities/utils.h"
#include "utilities/vec.cuh"
#include "utilities/tensor.h"

// ----------------------------------------------------------------------------
// CUDA kernels

constexpr const int WARP_SIZE = 32;

/**
 * @brief Device function for RMS normalization forward pass.
 *
 * Computes RMS normalization: out = (x * rsqrt(mean(x^2) + eps)) * weight
 * Uses shared memory to cache weights and inputs for efficient access.
 * Each warp processes one token, with vectorized 128-bit loads/stores.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] out Normalized output tensor of shape (N, C).
 * @param[out] rms Optional output for 1/sqrt(mean(x^2)+eps) per token, may be NULL.
 * @param[in] inp Input tensor of shape (N, C).
 * @param[in] weight Scale weights of shape (C,).
 * @param[in,out] abs_max_ptr Optional global absolute maximum tracker, may be NULL.
 * @param epsilon Small constant for numerical stability.
 * @param N Number of tokens (B*T).
 * @param C Hidden dimension.
 */
template<class floatX>
__device__ void rmsnorm_forward_kernel(floatX* __restrict__ out, float* __restrict__ rms,
                                       const floatX*  __restrict__ inp, const floatX*  __restrict__ weight,
                                       float* __restrict__ abs_max_ptr, float epsilon, int N, int C) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;

    // this kernel is a simplified version of layernorm_forward_kernel6
    assert(blockDim.x == WARP_SIZE);

    // load weights into shared memory
    // do this before we allow any threads to exit!
    extern __shared__ char* params[];
    __shared__ float block_abs_max;
    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_in = reinterpret_cast<x128*>(params) + ((1 + threadIdx.y) * C / x128::size);

    int sidx = (threadIdx.x + WARP_SIZE * threadIdx.y) * x128::size;
    for(int i = sidx; i < C; i += blockDim.y * WARP_SIZE * x128::size) {
        s_weight[i/x128::size] = x128::load(weight + i);
    }
    if (abs_max_ptr && threadIdx.x == 0) {
        block_abs_max =0.f;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if(idx >= N) { return; } // guard

    // adjust pointers to current token
    inp += idx * C;
    out += idx * C;

    float acc = 0.f;

    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = x128::load_cs(inp + c);
        s_in[c / x128::size] = in_data;
        for(int k = 0; k < x128::size; ++k) {
            float data_k = (float)in_data[k];
            acc += data_k * data_k;
        }
    }

    acc = warpReduceSum(acc) / C;
    float s = rsqrtf(acc + epsilon);
    float thread_abs_max = 0.f;

    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];
        const x128 w = s_weight[c / x128::size];
        x128 out_data;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * (float)in_data[k]; // normalized output
            // Note: would make sense to do this in fp32, but transformers uses bf16 here,
            // so we try to match
            out_data[k] = (floatX)n * (floatX)w[k]; // scale
            if (abs_max_ptr) {
                thread_abs_max = fmaxf(thread_abs_max, fabsf(out_data[k]));
            }
        }

        out_data.store(out + c);    // TODO cs
    }

    handle_absmax_reduction(abs_max_ptr, &block_abs_max, thread_abs_max);

    // store the rms, no need to cache it
    if(threadIdx.x == 0 && rms != nullptr) {
        __stcs(rms + idx, s);
    }
}

/**
 * @brief Device function for fused residual addition and RMS normalization.
 *
 * Combines two operations in a single kernel pass:
 * 1. residual = inp1 + inp2 (residual connection)
 * 2. normed = rmsnorm(residual) * weight
 *
 * This fusion reduces memory bandwidth by avoiding materialization of
 * the intermediate residual sum before normalization.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] residual Output residual sum of shape (N, C).
 * @param[out] normed Normalized output tensor of shape (N, C).
 * @param[out] rrms Output for 1/sqrt(mean(x^2)+eps) per token for backward pass.
 * @param[in] inp1 First input tensor of shape (N, C).
 * @param[in] inp2 Second input tensor of shape (N, C).
 * @param[in] weight Scale weights of shape (C,).
 * @param[in,out] abs_max_ptr Optional global absolute maximum tracker, may be NULL.
 * @param epsilon Small constant for numerical stability.
 * @param N Number of tokens (B*T).
 * @param C Hidden dimension.
 */
template<typename floatX>
__device__ void fused_residual_rmsnorm_forward_kernel(floatX* residual, floatX* normed, float* rrms,
                                                      const floatX* inp1, const floatX* inp2,
                                                      const floatX* weight, float* __restrict__ abs_max_ptr, float epsilon,
                                                      int N, int C) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    assert(blockDim.x == WARP_SIZE);

    // load weights and biases into shared memory
    // do this before we allow any threads to exit!
    extern __shared__ char* params[];
    __shared__ float block_abs_max;
    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_res = reinterpret_cast<x128*>(params) + ((1 + threadIdx.y) * C / x128::size);

    int sidx = (threadIdx.x + WARP_SIZE * threadIdx.y) * x128::size;
    for(int i = sidx; i < C; i += blockDim.y * WARP_SIZE * x128::size) {
        s_weight[i/x128::size] = x128::load(weight + i);
    }
    if (abs_max_ptr && threadIdx.x == 0) {
        block_abs_max = 0.f;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if(idx >= N) return;

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    float sum_squared = 0.f;

    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in1 = x128::load_cs(inp1 + c);
        const x128 in2 = x128::load_cs(inp2 + c);
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            out[k] = static_cast<floatX>((float)in1[k] + (float)in2[k]);
            sum_squared += (float)out[k] * (float)out[k];
        }
        out.store(residual + c);   // TODO cs
        s_res[c / x128::size] = out;
    }

    sum_squared = warpReduceSum(sum_squared) / C;
    float s = rsqrtf(sum_squared + epsilon);
    float thread_abs_max = 0.f;

    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 res = s_res[c / x128::size];
        const x128 w = s_weight[c / x128::size];
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * (float)res[k]; // normalized output
            out[k] = (floatX)n * (floatX)w[k]; // scale
            if (abs_max_ptr) {
                thread_abs_max = fmaxf(thread_abs_max, fabsf(out[k]));
            }
        }

        out.store(normed + c);
    }

    handle_absmax_reduction(abs_max_ptr, &block_abs_max, thread_abs_max);

    // cache the rrms for the backward pass later
    if(threadIdx.x == 0) {
        rrms[idx] = s;
    }
}

/**
 * @brief Unified CUDA kernel for RMS normalization with optional residual fusion.
 *
 * Dispatches to either pure rmsnorm or fused residual+rmsnorm based on whether
 * residual pointer is NULL. Having dispatch inside kernel enables CUDA graph capture
 * of the full transformer block.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] residual Output residual sum (NULL for pure rmsnorm).
 * @param[out] normed Normalized output tensor of shape (N, C).
 * @param[out] rrms Output for reciprocal RMS per token.
 * @param[in] inp1 Input tensor (or first input for fused variant).
 * @param[in] inp2 Second input for fused variant (ignored if residual is NULL).
 * @param[in] weight Scale weights of shape (C,).
 * @param[in,out] abs_max_ptr Optional global absolute maximum tracker.
 * @param epsilon Small constant for numerical stability.
 * @param N Number of tokens.
 * @param C Hidden dimension.
 */
template<typename floatX>
__global__ void rmsnorm_forward_unified_kernel(floatX* residual, floatX* normed, float* rrms,
                                               const floatX* inp1, const floatX* inp2,
                                               const floatX* weight, float* __restrict__ abs_max_ptr, float epsilon,
                                               int N, int C) {
    if(residual == nullptr) {
        rmsnorm_forward_kernel(normed, rrms, inp1, weight, abs_max_ptr, epsilon, N, C);
    } else {
        fused_residual_rmsnorm_forward_kernel(residual, normed, rrms, inp1, inp2, weight, abs_max_ptr, epsilon, N, C);
    }
}

/**
 * @brief CUDA kernel for RMS normalization with fused FP8 quantization.
 *
 * Computes RMS normalization and quantizes output to FP8 E4M3 in a single pass,
 * eliminating the need for a separate quantization kernel and intermediate buffer.
 * Uses pre-computed abs-max from forward pass for scaling.
 *
 * @tparam floatX Input data type (float or nv_bfloat16).
 * @param[out] out Quantized FP8 output tensor of shape (N, C).
 * @param[out] scale_ptr Inverse scale factor for dequantization.
 * @param[out] rms Optional output for 1/sqrt(mean(x^2)+eps) per token.
 * @param[in] inp Input tensor of shape (N, C).
 * @param[in] weight Scale weights of shape (C,).
 * @param[in] abs_max_ptr Pre-computed absolute maximum for quantization scaling.
 * @param epsilon Small constant for numerical stability.
 * @param N Number of tokens (B*T).
 * @param C Hidden dimension.
 */
template<typename floatX>
__device__ void rmsnorm_forward_quant_kernel(__nv_fp8_e4m3* __restrict__ out, float* __restrict__ scale_ptr,
                                             float* __restrict__ rms, const floatX* __restrict__ inp,
                                             const floatX* __restrict__ weight, const float* __restrict__ abs_max_ptr,
                                             float epsilon, int N, int C) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    using f8v_t = GenericVector<__nv_fp8_e4m3, 16 / sizeof(floatX)>;

    assert(blockDim.x == WARP_SIZE);

    // Compute quantization scale from pre-computed abs-max
    float scale = 448.f / fmaxf(*abs_max_ptr, 1e-10f);
    if(threadIdx.x == 0 && blockIdx.x == 0 && scale_ptr) {
        *scale_ptr = 1.f / scale;
    }

    // Load weights into shared memory
    extern __shared__ char* params[];
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_in = reinterpret_cast<x128*>(params) + ((1 + threadIdx.y) * C / x128::size);

    int sidx = (threadIdx.x + WARP_SIZE * threadIdx.y) * x128::size;
    for(int i = sidx; i < C; i += blockDim.y * WARP_SIZE * x128::size) {
        s_weight[i/x128::size] = x128::load(weight + i);
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if(idx >= N) { return; }

    // Adjust pointers to current token
    inp += idx * C;
    out += idx * C;

    // Compute RMS
    float acc = 0.f;
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = x128::load_cs(inp + c);
        s_in[c / x128::size] = in_data;
        for(int k = 0; k < x128::size; ++k) {
            float data_k = (float)in_data[k];
            acc += data_k * data_k;
        }
    }

    acc = warpReduceSum(acc) / C;
    float s = rsqrtf(acc + epsilon);

    // Normalize, scale, and quantize in one pass
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];
        const x128 w = s_weight[c / x128::size];
        f8v_t packed_out;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * (float)in_data[k];  // normalize
            float scaled = (float)((floatX)n * (floatX)w[k]);  // scale by weight
            __nv_fp8_e4m3 quant;
            quant.__x = __nv_cvt_float_to_fp8(scale * scaled, __nv_saturation_t::__NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E4M3);
            packed_out[k] = quant;
        }
        packed_out.store(out + c);
    }

    // Store RMS for backward pass
    if(threadIdx.x == 0 && rms != nullptr) {
        __stcs(rms + idx, s);
    }
}

/**
 * @brief Wrapper kernel for rmsnorm_forward_quant_kernel.
 *
 * CUDA kernel wrapper that calls the device function for RMS normalization
 * with fused FP8 quantization.
 */
template<typename floatX>
__global__ void rmsnorm_forward_quant_unified_kernel(__nv_fp8_e4m3* out, float* scale_ptr, float* rms,
                                                     const floatX* inp, const floatX* weight,
                                                     const float* abs_max_ptr, float epsilon, int N, int C) {
    rmsnorm_forward_quant_kernel(out, scale_ptr, rms, inp, weight, abs_max_ptr, epsilon, N, C);
}

// ----------------------------------------------------------------------------
// Kernel launchers

/**
 * @brief Template launcher for unified RMS normalization kernel.
 *
 * Configures and launches rmsnorm_forward_unified_kernel with optimal
 * block size (256 threads = 8 warps) and dynamic shared memory for
 * weights and per-warp input caching.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] residual Output residual sum (NULL for pure rmsnorm).
 * @param[out] normed Normalized output tensor.
 * @param[out] rrms Reciprocal RMS values per token.
 * @param[in] inp1 Input tensor.
 * @param[in] inp2 Second input for fused variant.
 * @param[in] weight Scale weights.
 * @param[in,out] abs_max_ptr Optional absolute maximum tracker.
 * @param epsilon Numerical stability constant.
 * @param N Number of tokens.
 * @param C Hidden dimension.
 * @param stream CUDA stream.
 */
template<typename floatX>
void rmsnorm_forward_unified_imp(floatX* residual, floatX* normed, float* rrms,
                                 const floatX* inp1, const floatX* inp2,
                                 const floatX* weight, float* abs_max_ptr,
                                 float epsilon, int N, int C, cudaStream_t stream) {
    const int block_size = 256;
    int block_y = block_size / WARP_SIZE;
    const int grid_size = div_ceil(N, block_y);
    size_t smem = (1 + block_y) * C * sizeof(floatX);

    if (abs_max_ptr)
        CUDA_CHECK(cudaMemsetAsync(abs_max_ptr, 0, sizeof(float), stream));

    // in order to use more than 48 KiB of smem, need to call cudaFuncSetAttribute
    // this may fail, in which case we fall back to the smem free implementation.
    CUDA_CHECK(cudaGetLastError());
    auto status = cudaFuncSetAttribute(rmsnorm_forward_unified_kernel<floatX>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    CUDA_CHECK(cudaGetLastError());
    if(status == cudaSuccess) {
        rmsnorm_forward_unified_kernel<<<grid_size, dim3(WARP_SIZE, block_y), smem, stream>>>(residual, normed,
                                                                                              rrms, inp1, inp2,
                                                                                              weight, abs_max_ptr, epsilon, N, C);
    } else {
        assert(false);
    }
    CUDA_CHECK(cudaGetLastError());
}

/// @brief RMS normalization forward pass for FP32 tensors.
void rmsnorm_forward(float* out, float* rms, const float* inp, const float* weight, float* abs_max_ptr, float epsilon, int B, int T, int C, cudaStream_t stream) {
    rmsnorm_forward_unified_imp<float>(nullptr, out, rms, inp, nullptr, weight, abs_max_ptr, epsilon, B*T, C, stream);
}

/// @brief RMS normalization forward pass for BF16 tensors.
void rmsnorm_forward(nv_bfloat16* out, float* rms, const nv_bfloat16* inp, const nv_bfloat16* weight, float* abs_max_ptr, float epsilon, int B, int T, int C, cudaStream_t stream)  {
    rmsnorm_forward_unified_imp<nv_bfloat16>(nullptr, out, rms, inp, nullptr, weight, abs_max_ptr, epsilon, B*T, C, stream);
}

// ============================================================================
// RMSNorm apply with pre-computed rstd (for deterministic recomputation)
// ============================================================================
// This kernel applies out = inp * rstd * weight using a saved rstd value,
// avoiding the need to recompute rstd which can differ due to FP non-associativity.

template<typename floatX>
__global__ void rmsnorm_apply_saved_kernel(floatX* __restrict__ out,
                                           const floatX* __restrict__ inp,
                                           const floatX* __restrict__ weight,
                                           const float* __restrict__ rstd,
                                           int N, int C) {
    // Each block handles one token (row), threads handle C elements
    int row = blockIdx.x;
    if (row >= N) return;

    float s = rstd[row];
    inp += row * C;
    out += row * C;

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float inp_val = (float)inp[c];
        float weight_val = (float)weight[c];
        out[c] = (floatX)(inp_val * s * weight_val);
    }
}

template<typename floatX>
void rmsnorm_apply_saved_imp(floatX* out, const floatX* inp, const floatX* weight,
                             const float* rstd, int N, int C, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = N;
    rmsnorm_apply_saved_kernel<<<grid_size, block_size, 0, stream>>>(out, inp, weight, rstd, N, C);
    CUDA_CHECK(cudaGetLastError());
}

void rmsnorm_apply_saved(float* out, const float* inp, const float* weight, const float* rstd,
                         int B, int T, int C, cudaStream_t stream) {
    rmsnorm_apply_saved_imp<float>(out, inp, weight, rstd, B * T, C, stream);
}

void rmsnorm_apply_saved(nv_bfloat16* out, const nv_bfloat16* inp, const nv_bfloat16* weight, const float* rstd,
                         int B, int T, int C, cudaStream_t stream) {
    rmsnorm_apply_saved_imp<nv_bfloat16>(out, inp, weight, rstd, B * T, C, stream);
}

// ============================================================================
// Fused residual add + RMSNorm apply with pre-computed rstd
// ============================================================================
// Computes: residual = inp1 + inp2, normed = residual * rstd * weight
// Uses saved rstd for deterministic recomputation.

template<typename floatX>
__global__ void fused_residual_rmsnorm_apply_saved_kernel(floatX* __restrict__ residual,
                                                          floatX* __restrict__ normed,
                                                          const floatX* __restrict__ inp1,
                                                          const floatX* __restrict__ inp2,
                                                          const floatX* __restrict__ weight,
                                                          const float* __restrict__ rstd,
                                                          int N, int C) {
    int row = blockIdx.x;
    if (row >= N) return;

    float s = rstd[row];
    inp1 += row * C;
    inp2 += row * C;
    residual += row * C;
    normed += row * C;

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float r = (float)inp1[c] + (float)inp2[c];
        residual[c] = (floatX)r;
        normed[c] = (floatX)(r * s * (float)weight[c]);
    }
}

template<typename floatX>
void fused_residual_rmsnorm_apply_saved_imp(floatX* residual, floatX* normed,
                                             const floatX* inp1, const floatX* inp2,
                                             const floatX* weight, const float* rstd,
                                             int N, int C, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = N;
    fused_residual_rmsnorm_apply_saved_kernel<<<grid_size, block_size, 0, stream>>>(
        residual, normed, inp1, inp2, weight, rstd, N, C);
    CUDA_CHECK(cudaGetLastError());
}

void fused_residual_rmsnorm_apply_saved(float* residual, float* normed,
                                        const float* inp1, const float* inp2,
                                        const float* weight, const float* rstd,
                                        int N, int C, cudaStream_t stream) {
    fused_residual_rmsnorm_apply_saved_imp<float>(residual, normed, inp1, inp2, weight, rstd, N, C, stream);
}

void fused_residual_rmsnorm_apply_saved(nv_bfloat16* residual, nv_bfloat16* normed,
                                        const nv_bfloat16* inp1, const nv_bfloat16* inp2,
                                        const nv_bfloat16* weight, const float* rstd,
                                        int N, int C, cudaStream_t stream) {
    fused_residual_rmsnorm_apply_saved_imp<nv_bfloat16>(residual, normed, inp1, inp2, weight, rstd, N, C, stream);
}

/// @brief Fused residual addition + RMS normalization forward for FP32 tensors.
void fused_residual_rmsnorm_forward(float* residual, float* normed, float* rrms, const float* inp1, const float* inp2, const float* weight, float* abs_max_ptr,
                                    float epsilon, int N, int C, cudaStream_t stream) {
    rmsnorm_forward_unified_imp(residual, normed, rrms, inp1, inp2, weight, abs_max_ptr, epsilon, N, C, stream);
}

/// @brief Fused residual addition + RMS normalization forward for BF16 tensors.
void fused_residual_rmsnorm_forward(nv_bfloat16* residual, nv_bfloat16* normed, float* rrms, const nv_bfloat16* inp1, const nv_bfloat16* inp2, const nv_bfloat16* weight, float* abs_max_ptr,
                                    float epsilon, int N, int C, cudaStream_t stream) {
    rmsnorm_forward_unified_imp(residual, normed, rrms, inp1, inp2, weight, abs_max_ptr, epsilon, N, C, stream);
}

/**
 * @brief CUDA kernel for RMS normalization backward pass.
 *
 * Computes gradients for both input and weight parameters. Uses a two-phase
 * approach: first accumulates weight gradients per-block in shared memory,
 * then the last block to finish reduces all partial sums.
 *
 * Key optimizations:
 * - Vectorized 128-bit loads/stores
 * - Warp-level reductions for per-token statistics
 * - Atomic block counting for deterministic final reduction
 * - Shared memory for weight gradient accumulation
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[in,out] dinp Input gradient tensor, accumulated in-place.
 * @param[in,out] dweight Weight gradient tensor, accumulated in-place.
 * @param[in,out] scratch Global scratch memory for cross-block reduction.
 * @param[in] dout Upstream gradient tensor of shape (B*T, C).
 * @param[in] inp Original input tensor of shape (B*T, C).
 * @param[in] weight Scale weights of shape (C,).
 * @param[in] rstd Reciprocal standard deviation from forward pass.
 * @param[in,out] abs_max_ptr Optional absolute maximum tracker.
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Hidden dimension.
 */
template<class floatX>
__global__ void __launch_bounds__(512, 2)
rmsnorm_backward_kernel10(floatX* dinp, floatX* dweight, std::byte* scratch,
                          const floatX* dout, const floatX* inp, const floatX* weight,
                          const float* rstd, float* abs_max_ptr, int B, int T, int C,
                          bool skip_weight_grad) {
    // size of scratch: sizeof(float) * C + 128
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    using f128 = GenericVector<float, 16/sizeof(float)>;

    int BLOCK_SIZE = blockDim.x;
    int warpsInBlock = BLOCK_SIZE / WARP_SIZE; //number of warps in block
    extern __shared__ float shared[];
    __shared__ float block_abs_max;
    float thread_abs_max = 0.f;

    int warpId = threadIdx.x / WARP_SIZE; // warp index within a block
    int baseIdx = blockIdx.x * warpsInBlock + warpId;
    int warpThreadIdx = threadIdx.x % WARP_SIZE; // Thread index within the warp
    int warpsInGrid = gridDim.x * warpsInBlock;
    int C_per_iteration = WARP_SIZE * x128::size;
    int iterations_C = div_ceil(C, C_per_iteration); // + 2;

    // the first half of shared memory is bias, second is weight
    size_t rounded_C = div_ceil(C, (int)(32 * x128::size)) * (32 * x128::size);
    float* dweight_shared = shared + rounded_C;
    // warp zero doesn't actually write to the _tmp_shared memory locations, so we don't need to reserve memory
    // the obvious solution is to change the addressing below to use (threadId.x-32) as offset, but that causes
    // register spills, so instead we mess with the base pointer here, which doesn't increase register usage.
    float* dweight_tmp_shared = shared + 2 * rounded_C + f128::size * BLOCK_SIZE - 2 * WARP_SIZE * f128::size;

    // init shared memory to zero
    for(int i = threadIdx.x * f128::size; i < rounded_C; i += BLOCK_SIZE * f128::size) {
        f128::zeros().store(dweight_shared + i);
    }
    if (abs_max_ptr && threadIdx.x == 0) {
        block_abs_max = 0.f;
    }
    __syncthreads();

    if(baseIdx >= B * T) {
        // make sure we're not reading uninitialized memory below
        f128::zeros().store(dweight_tmp_shared + threadIdx.x * f128::size);
    }

    for (int bt = baseIdx; bt < B * T; bt += warpsInGrid) {
        const floatX* dout_bt = dout + bt * C;
        const floatX* inp_bt = inp +bt * C;
        floatX* dinp_bt = dinp + bt * C;

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warpThreadIdx * x128::size; i < C; i += WARP_SIZE * x128::size) {
            x128 dout128_i   = x128::load(dout_bt + i);
            x128 inp128_i    = x128::load(inp_bt  + i);
            x128 weight128_i = x128::load(weight  + i);
            for (int k = 0; k < x128::size; k++) {
                float dnorm_i = (float)weight128_i[k] * (float)dout128_i[k];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * (float)inp128_i[k];
            }
        }

        const float rstd_bt = rstd[bt];
        dnorm_norm_mean = warpReduceSum(dnorm_norm_mean) / C * rstd_bt;

        for (int c = 0; c < iterations_C; c++) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);

            x128 dout128   = x128::zeros();
            x128 inp128    = x128::zeros();
            x128 dinp128   = x128::zeros();
            x128 weight128 = x128::zeros();

            if(global_index < C) {
                dout128 = x128::load_cs(dout_bt + global_index);
                inp128 = x128::load_cs(inp_bt + global_index);
                dinp128 = x128::load(dinp_bt + global_index);
                weight128 = x128::load(weight + global_index);
            }

            for(int o = 0; o < x128::size / f128::size; ++o) {
                f128 dweight_f;
                for(int i = 0; i < f128::size; ++i) {
                    int x = o * f128::size + i;
                    float dout_i = (float)dout128[x];
                    float norm_bti = ((float)inp128[x]) * rstd_bt;
                    dweight_f[i] = norm_bti * dout_i;

                    float dval = 0.0f;
                    dval += (float) weight128[x] * (float)dout128[x]; // term 1
                    dval -= norm_bti * dnorm_norm_mean; // term 2
                    dval *= rstd_bt; // final scale
                    dinp128[x] = (floatX) ((float) dinp128[x] + dval);
                }

                if (warpId != 0) {
                    // this seems to generate a 64-bit store, instead of 128-bit.
                    // however, forcing 128-bit (e.g., using inline ptx), results in register
                    // spilling and much worse performance, so we'll keep it like this for now
                    // but ideally, we could reduce the register pressure a little.
                    dweight_f.store(dweight_tmp_shared + threadIdx.x * f128::size);
                }
                __syncthreads();
                if (warpId == 0) {
                    for (int j = 1; j < warpsInBlock; j++) {
                        f128 dweight_tmp = f128::load(dweight_tmp_shared + f128::size * (threadIdx.x + j * WARP_SIZE));
                        for(int i = 0; i < f128::size; ++i) {
                            dweight_f[i] += dweight_tmp[i];
                        }
                    }
                }
                __syncthreads();
                if (warpId == 0) {
                    f128 dw_old = f128::load(dweight_shared + global_index + f128::size * o);
                    for(int i = 0; i < f128::size; ++i) {
                        dweight_f[i] += dw_old[i];
                    }
                    dweight_f.store(dweight_shared + global_index + f128::size * o);
                }
            }
            if(global_index < C) {
                // cache in L2 as this is read by the next kernel, but bypass L1 to minimise thrashing
                // TODO cache hint
                dinp128.store(dinp_bt + global_index);

                for(int i = 0; i < x128::size; ++i) {
                    thread_abs_max = fmaxf(thread_abs_max, fabsf(dinp128[i]));
                }
            }
        }
    }
    __syncthreads();
    handle_absmax_reduction(abs_max_ptr, &block_abs_max, thread_abs_max);

    // Skip the entire dweight reduction when skip_weight_grad is true
    // This saves memory bandwidth and compute for LoRA-only mode where base weight gradients are not needed
    if (skip_weight_grad) {
        return;
    }

    // Each block writes its partial sum to global memory
    // The last block to finish becomes responsible for summing up all the partial sums
    // This is done by atomically incrementing a flag (cleared to 0 before launching the kernel)
    unsigned int* scratchFlag = reinterpret_cast<unsigned int*>(scratch);
    // Increment scratch pointer by a full cacheline so that everything remains cacheline aligned
    float* scratch_dweight = reinterpret_cast<float*>(scratch + 128);
    for(int i = threadIdx.x * f128::size; i < C; i += BLOCK_SIZE * f128::size) {
        // Write to global memory in the same "shared memory banking friendly" order
        f128::load(dweight_shared + i).store(scratch_dweight + i + C*blockIdx.x);
    }
    __syncthreads();
    // that portion of shared memory is no longer used, so we can repurpose it for the scratch flag.
    unsigned int *tmp_flag = (unsigned int*)(shared + 2*rounded_C);
    if (threadIdx.x == 0) {
        *tmp_flag = atomicInc(scratchFlag, gridDim.x);
    }
    __syncthreads();
    if (*tmp_flag == gridDim.x-1) {
        // Reduction of the partial sums by the final block
        // todo - there isn't enough parallelism even inside that single SM...
        // ==> so could maybe split into another kernel with YET ANOTHER level of reduction?!
        for(int i = threadIdx.x * f128::size; i < C; i += BLOCK_SIZE * f128::size) {
            f128 dweight_accum = f128::zeros();

            for (int read_block_idx = 0; read_block_idx < gridDim.x; read_block_idx++) {
                int offset = i + C*read_block_idx;
                f128 dweight128 = f128::load(scratch_dweight + offset);
                for(int k = 0; k < f128::size; k++) {
                    dweight_accum[k] += dweight128[k];
                }
            }
            dweight_accum.store(dweight_shared + i);
        }
        __syncthreads();

        // convert from float/FP32 to floatX/BF16 for the final write
        // this is separate because it cannot use as many warps as the above (f128 vs x128)
        // todo - if we split this code into another kernel, we could maybe do it at the same time?
        for (int c = warpId; c < iterations_C; c += warpsInBlock) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);
            if (global_index >= C) {
                break;
            }
            x128 dweight128 = x128::load(dweight + global_index);
            for(int o = 0; o < x128::size / f128::size; ++o) {
                f128 s_dw = f128::load(dweight_shared + global_index + o * f128::size);
                for(int i = 0; i < f128::size; ++i) {
                    int x = o * f128::size + i;
                    dweight128[x] = (floatX)(s_dw[i] + (float)dweight128[x]);
                }
            }
            dweight128.store(dweight + global_index);
        }
    }
}

/**
 * @brief Calculates required scratch memory size for RMS normalization backward.
 *
 * Scratch memory is used for cross-block reduction of weight gradients.
 * Each block writes its partial sum, and the last block reduces them all.
 *
 * @param C Hidden dimension.
 * @param dp CUDA device properties (for multiprocessor count).
 * @return Required scratch buffer size in bytes.
 */
int get_rmsnorm_backward_scratch_size(int C, const cudaDeviceProp& dp) {
    int per_block = 128 + C * sizeof(float);
    return per_block * dp.multiProcessorCount * 2;
}

/**
 * @brief Template launcher for RMS normalization backward kernel.
 *
 * Configures shared memory, copies dresidual to dinp if needed, and launches
 * the backward kernel with 512 threads per block and 2 blocks per SM.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] dinp Input gradient output, accumulated with dresidual.
 * @param[in,out] dweight Weight gradient, accumulated in-place.
 * @param[in,out] scratch Global scratch buffer for cross-block reduction.
 * @param[in] dresidual Upstream gradient from residual connection.
 * @param[in] dout Upstream gradient from normalization output.
 * @param[in] inp Original input tensor.
 * @param[in] weight Scale weights.
 * @param[in] rstd Reciprocal standard deviation from forward.
 * @param[in,out] abs_max_ptr Optional absolute maximum tracker.
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Hidden dimension.
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
template<class floatX>
void rmsnorm_backward_imp(floatX* dinp, floatX* dweight, std::byte* scratch,
                          const floatX* dresidual, const floatX* dout, const floatX* inp, const floatX* weight, const float* rstd,
                          float* abs_max_ptr,
                          int B, int T, int C, const cudaDeviceProp& dp, cudaStream_t stream, bool skip_weight_grad) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    using f128 = GenericVector<float, 16/sizeof(float)>;
    const int block_size = 512;
    const int blocks_per_sm = 2; // supported on every architecture and less cache thrashing than 3
    const int grid_size = blocks_per_sm * dp.multiProcessorCount;
    size_t rounded_C = div_ceil(C, (int)(32 * x128::size)) * (32 * x128::size);
    size_t shared_mem_size = (2 * rounded_C + 2 * (block_size - 32) * f128::size) * sizeof(float);
    CUDA_CHECK(cudaFuncSetAttribute(rmsnorm_backward_kernel10<floatX>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
    if(dresidual != dinp) {
        CUDA_CHECK(cudaMemcpyAsync(dinp, dresidual, B*T*C * sizeof(floatX), cudaMemcpyDeviceToDevice, stream));
    }
    if (abs_max_ptr) {
        CUDA_CHECK(cudaMemsetAsync(abs_max_ptr, 0, sizeof(float), stream));
    }
    // Only need to reset the scratch flag if computing weight gradients
    if (!skip_weight_grad) {
        CUDA_CHECK(cudaMemsetAsync(scratch, 0, 1 * sizeof(float), stream));
    }
    rmsnorm_backward_kernel10<<<grid_size, block_size, shared_mem_size, stream>>>(dinp, dweight, scratch, dout, inp, weight, rstd, abs_max_ptr, B, T, C, skip_weight_grad);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief RMS normalization backward pass for FP32 tensors.
 *
 * Computes gradients for input and weight parameters given upstream gradients.
 *
 * @param[out] dinp Input gradient output.
 * @param[in,out] dweight Weight gradient, accumulated in-place.
 * @param[in,out] scratch Scratch buffer from get_rmsnorm_backward_scratch_size().
 * @param[in] dresidual Upstream gradient from residual connection.
 * @param[in] dout Upstream gradient from normalization output.
 * @param[in] inp Original input tensor.
 * @param[in] weight Scale weights.
 * @param[in] rstd Reciprocal standard deviation from forward.
 * @param[in,out] abs_max_ptr Optional absolute maximum tracker.
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Hidden dimension.
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void rmsnorm_backward(float* dinp, float* dweight, std::byte* scratch,
                      const float* dresidual, const float* dout, const float* inp, const float* weight, const float* rstd, float* abs_max_ptr,
                      int B, int T, int C, const cudaDeviceProp& dp, cudaStream_t stream, bool skip_weight_grad) {
    rmsnorm_backward_imp(dinp, dweight, scratch, dresidual, dout, inp, weight, rstd, abs_max_ptr, B, T, C, dp, stream, skip_weight_grad);
}

/**
 * @brief RMS normalization backward pass for BF16 tensors.
 *
 * Computes gradients for input and weight parameters given upstream gradients.
 *
 * @param[out] dinp Input gradient output.
 * @param[in,out] dweight Weight gradient, accumulated in-place.
 * @param[in,out] scratch Scratch buffer from get_rmsnorm_backward_scratch_size().
 * @param[in] dresidual Upstream gradient from residual connection.
 * @param[in] dout Upstream gradient from normalization output.
 * @param[in] inp Original input tensor.
 * @param[in] weight Scale weights.
 * @param[in] rstd Reciprocal standard deviation from forward.
 * @param[in,out] abs_max_ptr Optional absolute maximum tracker.
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Hidden dimension.
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void rmsnorm_backward(nv_bfloat16* dinp, nv_bfloat16* dweight, std::byte* scratch,
                      const nv_bfloat16* dresidual, const nv_bfloat16* dout, const nv_bfloat16* inp, const nv_bfloat16* weight, const float* rstd, float* abs_max_ptr,
                      int B, int T, int C, const cudaDeviceProp& dp, cudaStream_t stream, bool skip_weight_grad) {
    rmsnorm_backward_imp(dinp, dweight, scratch, dresidual, dout, inp, weight, rstd, abs_max_ptr, B, T, C, dp, stream, skip_weight_grad);
}

/**
 * @brief RMS normalization forward pass with fused FP8 quantization.
 *
 * Computes RMS normalization and quantizes output to FP8 E4M3 in a single kernel,
 * using pre-computed abs-max from the initial forward pass. This eliminates redundant
 * quantization overhead during recomputation.
 *
 * @param[out] out Quantized FP8 output tensor.
 * @param[out] scale_ptr Inverse scale factor for dequantization.
 * @param[out] rms Reciprocal RMS per token for backward pass.
 * @param[in] inp Input BF16 tensor of shape (B, T, C).
 * @param[in] weight Scale weights of shape (C,).
 * @param[in] abs_max_ptr Pre-computed absolute maximum for quantization.
 * @param epsilon Small constant for numerical stability.
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Hidden dimension.
 * @param stream CUDA stream.
 */
void rmsnorm_forward_quant(__nv_fp8_e4m3* out, float* scale_ptr, float* rms,
                           const nv_bfloat16* inp, const nv_bfloat16* weight,
                           const float* abs_max_ptr, float epsilon,
                           int B, int T, int C, cudaStream_t stream) {
    const int block_size = 256;
    const int warps_per_block = block_size / WARP_SIZE;
    using x128 = GenericVector<nv_bfloat16, 16/sizeof(nv_bfloat16)>;
    assert(C % x128::size == 0);
    
    const int N = B * T;
    const int grid_size = div_ceil(N, warps_per_block);
    size_t smem = (1 + warps_per_block) * C * sizeof(nv_bfloat16);
    
    rmsnorm_forward_quant_unified_kernel<<<grid_size, dim3(WARP_SIZE, warps_per_block), smem, stream>>>(
        out, scale_ptr, rms, inp, weight, abs_max_ptr, epsilon, N, C
    );
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Tensor wrapper for rmsnorm_forward_quant.
 *
 * Type-safe wrapper that dispatches to the appropriate implementation based on tensor dtypes.
 */
void rmsnorm_forward_quant(Tensor& out, float* scale_ptr, Tensor& rms,
                           const Tensor& inp, const Tensor& weight,
                           const float* abs_max_ptr, float epsilon,
                           int B, int T, int C, cudaStream_t stream) {
    if (inp.DType != ETensorDType::BF16) {
        throw std::runtime_error("rmsnorm_forward_quant: only BF16 input supported");
    }
    if (out.DType != ETensorDType::FP8_E4M3) {
        throw std::runtime_error("rmsnorm_forward_quant: output must be FP8_E4M3");
    }
    rmsnorm_forward_quant(out.get<__nv_fp8_e4m3>(), scale_ptr, rms.get<float>(),
                          inp.get<nv_bfloat16>(), weight.get<nv_bfloat16>(),
                          abs_max_ptr, epsilon, B, T, C, stream);
}
