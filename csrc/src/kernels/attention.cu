// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file attention.cu
 * @brief Memory-efficient attention implementation for FP32 precision.
 *
 * This is a relatively simple baseline implementation of memory-efficient attention.
 * Its main purpose is to allow running in *32-bit* precision, which is not supported
 * by cuDNN. Uses online softmax (FlashAttention-style) to avoid materializing the
 * full attention matrix.
 */

#include <cmath>
#include <cstdio>
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include "kernels/kernels.h"
#include "utilities/tensor.h"
#include "utilities/vec.cuh"
#include "kernel_utils.cuh"

namespace cg = cooperative_groups;

/**
 * @brief CUDA kernel for memory-efficient forward attention with online softmax.
 *
 * Implements FlashAttention-style online softmax to compute attention without
 * materializing the full B*T*T attention matrix. Uses sub-warp parallelism
 * for efficient reduction and split-K accumulation across sequence positions.
 *
 * Grid: (Hq, B, T) - one block per query head, batch, and query position.
 * Block: 512 threads organized into sub-warps of 16 threads each.
 *
 * @tparam E Head dimension (must be 64 or 128).
 * @tparam scalar_t Data type (float or nv_bfloat16).
 * @param[out] out Output tensor of shape (B, T, Hq, E).
 * @param[out] stats Log-sum-exp statistics for backward pass, shape (B, Hq, T).
 * @param scale Attention scale factor (typically 1/sqrt(E)).
 * @param qkv Input QKV tensor of shape (B, T, Hq + 2*Hkv, E).
 * @param B Batch size.
 * @param T Sequence length.
 * @param Hq Number of query heads.
 * @param Hkv Number of key/value heads.
 */
template <int E, class scalar_t>
__global__ void __launch_bounds__(512) attention_forward_gpu_kernel(scalar_t* out,
                                                                    float* stats,
                                                                    float scale,
                                                                    const scalar_t* qkv,
                                                                    int B,
                                                                    int T,
                                                                    int Hq,
                                                                    int Hkv,
                                                                    int window_size) {
    constexpr const int SubWarpSize = 16;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    auto sub_warp = cg::tiled_partition<SubWarpSize>(block);

    extern __shared__ float scratch[];

    int h = blockIdx.x;
    int b = blockIdx.y;
    int t = blockIdx.z;

    int hkv = h * Hkv / Hq;
    int TH = Hq + 2 * Hkv;
    ptrdiff_t batch_offset = b * T * TH * E;
    qkv += batch_offset;
    const scalar_t* query = qkv + t * TH * E + h * E;
    const scalar_t* keys = qkv + (Hq + hkv) * E;
    const scalar_t* values = qkv + (Hq + Hkv + hkv) * E;

    using vec_t = GenericVector<scalar_t, 4>;
    using fvec_t = GenericVector<float, 4>;
    using q_cache_t = GenericVector<float, E / SubWarpSize>;
    q_cache_t q_cache;

    // combine values
    using v_cache_t = GenericVector<float, E / SubWarpSize>;
    v_cache_t v_cache = v_cache_t::zeros();

    // determine maximum and online logsumexp
    float maximum = std::numeric_limits<float>::lowest();
    float lse = 0;

    for (int ee = 0; ee < E / (SubWarpSize * vec_t::size); ++ee) {
        int e = (ee * SubWarpSize + sub_warp.thread_rank()) * vec_t::size;
        vec_t qv = vec_t::load(query + e);
        for (int j = 0; j < vec_t::size; ++j) {
            q_cache[ee * vec_t::size + j] = (float)qv[j];
        }
    }

    int start = 0;
    if (window_size > 0) {
        start = t - window_size + 1;
        if (start < 0) start = 0;
    }
    for (int l = start + sub_warp.meta_group_rank(); l <= t; l += sub_warp.meta_group_size()) {
        ptrdiff_t kv_offset = l * TH * E;
        float qk = 0;
        for (int ee = 0; ee < E / (SubWarpSize * vec_t::size); ++ee) {
            int e = (ee * SubWarpSize + sub_warp.thread_rank()) * vec_t::size;
            vec_t kv = vec_t::load(keys + kv_offset + e);
            for (int j = 0; j < vec_t::size; ++j) {
                qk += q_cache[ee * vec_t::size + j] * (float)kv[j];
            }
        }
        qk = reduce_group_add(sub_warp, qk);
        if (qk > maximum) {
            float rescale = std::exp(scale * (maximum - qk));
            for (int j = 0; j < v_cache_t::size; ++j) {
                v_cache[j] *= rescale;
            }
            lse *= rescale;
            maximum = qk;
        }
        float att = std::exp(scale * (qk - maximum));
        lse += std::exp(scale * (qk - maximum));

        for (int ee = 0; ee < E / (SubWarpSize * vec_t::size); ++ee) {
            int e = (ee * SubWarpSize + sub_warp.thread_rank()) * vec_t::size;
            vec_t vv = vec_t::load(values + kv_offset + e);
            for (int j = 0; j < vec_t::size; ++j) {
                v_cache[ee * vec_t::size + j] += att * (float)vv[j];
            }
        }
    }

    // combine split-k results
    if (sub_warp.thread_rank() == 0) {
        scratch[sub_warp.meta_group_rank()] = maximum;
        scratch[sub_warp.meta_group_rank() + sub_warp.meta_group_size()] = lse;
    }

    __syncthreads();
    float r_max = maximum;
    float l_max = maximum;
    float r_lse = 0;
    if (warp.thread_rank() < sub_warp.meta_group_size()) {
        r_max = scratch[warp.thread_rank()];
        r_lse = scratch[warp.thread_rank() + sub_warp.meta_group_size()];
    }

    maximum = reduce_group_max(warp, r_max);
    r_lse *= std::exp(scale * (r_max - maximum));
    lse = reduce_group_add(warp, r_lse);
    float rescale = std::exp(scale * (l_max - maximum)) / lse;
    for (int j = 0; j < v_cache_t::size; ++j) {
        v_cache[j] *= rescale;
    }
    if (threadIdx.x == 0) {
        stats[b * Hq * T + h * T + t] = scale * maximum + std::log(lse);
    }
    __syncthreads();

    for (int ee = 0; ee < E / (SubWarpSize * vec_t::size); ++ee) {
        int e = (ee * SubWarpSize + sub_warp.thread_rank()) * vec_t::size;
        fvec_t store;
        for (int j = 0; j < vec_t::size; ++j) {
            store[j] = v_cache[ee * vec_t::size + j];
        }
        store.store(scratch + e + E * sub_warp.meta_group_rank());
    }

    if (warp.meta_group_rank() != 0) return;
    __syncthreads();
    // write result
    for (int e = vec_t::size * warp.thread_rank(); e < E; e += vec_t::size * warp.size()) {
        fvec_t res = fvec_t::zeros();
        for (int j = 0; j < sub_warp.meta_group_size(); ++j) {
            fvec_t sv = fvec_t::load(scratch + e + E * j);
            for (int jj = 0; jj < vec_t::size; ++jj) {
                res[jj] += sv[jj];
            }
        }
        vec_t cv;
        for (int j = 0; j < vec_t::size; ++j) {
            cv[j] = (scalar_t)res[j];
        }
        cv.store(out + ((b * T + t) * Hq + h) * E + e);
    }
}

/**
 * @brief CUDA kernel for attention backward pass.
 *
 * Computes gradients for Q, K, and V tensors. Each thread processes one
 * key-value position and accumulates gradients using atomic operations.
 * Uses the log-sum-exp statistics from the forward pass for numerical stability.
 *
 * Grid: (Hq, B, T) - one block per query head, batch, and query position.
 * Block: 512 threads, each processing different key positions.
 *
 * @tparam E Head dimension (must be 64 or 128).
 * @tparam scalar_t Data type (float or nv_bfloat16).
 * @param[out] dqkv Output gradient tensor of shape (B, T, Hq + 2*Hkv, E).
 * @param[in] stats Log-sum-exp statistics from forward pass, shape (B, Hq, T).
 * @param scale Attention scale factor (typically 1/sqrt(E)).
 * @param[in] out Forward pass output tensor of shape (B, T, Hq, E).
 * @param[in] dout Upstream gradient tensor of shape (B, T, Hq, E).
 * @param[in] qkv Input QKV tensor from forward pass, shape (B, T, Hq + 2*Hkv, E).
 * @param B Batch size.
 * @param T Sequence length.
 * @param Hq Number of query heads.
 * @param Hkv Number of key/value heads.
 */
template <int E, class scalar_t>
__global__ void __launch_bounds__(512) attention_backward_gpu_kernel(scalar_t* dqkv,
                                                                     const float* stats,
                                                                     float scale,
                                                                     const scalar_t* out,
                                                                     const scalar_t* dout,
                                                                     const scalar_t* qkv,
                                                                     int B,
                                                                     int T,
                                                                     int Hq,
                                                                     int Hkv,
                                                                     int window_size) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int t = blockIdx.z;

    const int hkv = h * Hkv / Hq;
    const int TH = Hq + 2 * Hkv;

    qkv += b * T * TH * E;
    dqkv += b * T * TH * E;
    out += b * T * Hq * E + h * E;
    dout += b * T * Hq * E + h * E;

    const scalar_t* query = qkv + h * E;
    const scalar_t* keys = qkv + (Hq + hkv) * E;
    const scalar_t* values = qkv + (Hq + Hkv + hkv) * E;

    scalar_t* dquery = dqkv + h * E;
    scalar_t* dkeys = dqkv + (Hq + hkv) * E;
    scalar_t* dvalues = dqkv + (Hq + Hkv + hkv) * E;

    float lse = stats[b * Hq * T + h * T + t];
    float D = 0.0;
    for (int i = 0; i < E; ++i) {
        D += dout[t * Hq * E + i] * out[t * Hq * E + i];
    }

    ptrdiff_t q_offset = t * TH * E;
    int start = 0;
    if (window_size > 0) {
        start = t - window_size + 1;
        if (start < 0) start = 0;
    }
    for (int l = start + threadIdx.x; l <= t; l += blockDim.x) {
        ptrdiff_t kv_offset = l * TH * E;
        float qk = 0;
        for (int i = 0; i < E; ++i) {
            qk += (float)query[q_offset + i] * (float)keys[kv_offset + i];
        }

        float att = std::exp(scale * qk - lse);
        float datt = 0.0;

        // Update V gradient and calculate attention gradient
        for (int i = 0; i < E; ++i) {
            float do_t = dout[t * Hq * E + i];
            atomicAdd(dvalues + kv_offset + i, att * do_t);
            datt += do_t * values[kv_offset + i];
        }

        float dqk = scale * att * (datt - D);

        // Update QK gradients
        for (int i = 0; i < E; ++i) {
            atomicAdd(dquery + q_offset + i, dqk * keys[kv_offset + i]);
            atomicAdd(dkeys + kv_offset + i, dqk * query[q_offset + i]);
        }
    }
}

/**
 * @brief Launches the forward attention kernel with appropriate head size specialization.
 *
 * Dispatches to the correct kernel instantiation based on head size (64 or 128).
 * Uses a grid of (Hq, B, T) blocks with 512 threads per block.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] out Output tensor of shape (B, T, Hq, Hs).
 * @param[out] stats Log-sum-exp statistics for backward pass, shape (B, Hq, T).
 * @param scale Attention scale factor (typically 1/sqrt(Hs)).
 * @param[in] qkv Input QKV tensor of shape (B, T, Hq + 2*Hkv, Hs).
 * @param B Batch size.
 * @param T Sequence length.
 * @param Hq Number of query heads.
 * @param Hkv Number of key/value heads.
 * @param Hs Head size (must be 64 or 128).
 * @param stream CUDA stream for asynchronous execution.
 * @return cudaError_t CUDA error status.
 */
template <class floatX>
cudaError_t attention_gpu_forward(floatX* out,
                                  float* stats,
                                  float scale,
                                  const floatX* qkv,
                                  int B,
                                  int T,
                                  int Hq,
                                  int Hkv,
                                  int Hs,
                                  int window_size,
                                  cudaStream_t stream) {
    dim3 grid_dim{(unsigned)Hq, (unsigned)B, (unsigned)T};
    dim3 block_dim{512, 1, 1};
    size_t smem = Hs * sizeof(float) * block_dim.x / 16;

    if (Hs == 512) {
        attention_forward_gpu_kernel<512>
            <<<grid_dim, block_dim, smem, stream>>>(out, stats, scale, qkv, B, T, Hq, Hkv, window_size);
    } else if (Hs == 256) {
        attention_forward_gpu_kernel<256>
            <<<grid_dim, block_dim, smem, stream>>>(out, stats, scale, qkv, B, T, Hq, Hkv, window_size);
    } else if (Hs == 128) {
        attention_forward_gpu_kernel<128>
            <<<grid_dim, block_dim, smem, stream>>>(out, stats, scale, qkv, B, T, Hq, Hkv, window_size);
    } else if (Hs == 64) {
        attention_forward_gpu_kernel<64>
            <<<grid_dim, block_dim, smem, stream>>>(out, stats, scale, qkv, B, T, Hq, Hkv, window_size);
    } else {
        printf("Unsupported head dimension %d\n", Hs);
        return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

/**
 * @brief Forward attention for FP32 tensors (cuDNN-compatible interface).
 *
 * Wrapper that matches the cuDNN attention interface but uses the custom GPU kernel
 * for FP32 support. The workspace and handle parameters are ignored since this
 * implementation doesn't use cuDNN.
 *
 * @param[out] out Output tensor of shape (B, T, Hq, HS).
 * @param[out] stats Log-sum-exp statistics for backward pass, shape (B, Hq, T).
 * @param[in] inp Input QKV tensor of shape (B, T, Hq + 2*Hkv, HS).
 * @param workspace Unused (for cuDNN interface compatibility).
 * @param handle Unused (for cuDNN interface compatibility).
 * @param B Batch size.
 * @param T Sequence length.
 * @param Hq Number of query heads.
 * @param Hkv Number of key/value heads.
 * @param HS Head size.
 * @param stream CUDA stream for asynchronous execution.
 */
void attention_forward_cudnn(float* out,        // output: (B, T, Nq, HS)
                             float* stats,      // output for backward pass: (B, Hq, T)
                             const float* inp,  // input: (B, T, Hq + 2Hkv, HS) QKV
                             std::byte* workspace,
                             cudnnHandle_t handle,
                             int B,
                             int T,
                             int Hq,
                             int Hkv,
                             int HS,
                             cudaStream_t stream) {
    attention_gpu_forward(out, stats, 1.f / sqrtf(HS), inp, B, T, Hq, Hkv, HS, /*window_size=*/0, stream);
}

/**
 * @brief Launches the backward attention kernel with appropriate head size specialization.
 *
 * Dispatches to the correct kernel instantiation based on head size (64 or 128).
 * Zeros the output gradient tensor before accumulating gradients with atomic operations.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] dqkv Output gradient tensor of shape (B, T, Hq + 2*Hkv, Hs).
 * @param[in] stats Log-sum-exp statistics from forward pass, shape (B, Hq, T).
 * @param scale Attention scale factor (typically 1/sqrt(Hs)).
 * @param[in] out Forward pass output tensor of shape (B, T, Hq, Hs).
 * @param[in] dout Upstream gradient tensor of shape (B, T, Hq, Hs).
 * @param[in] qkv Input QKV tensor from forward pass, shape (B, T, Hq + 2*Hkv, Hs).
 * @param B Batch size.
 * @param T Sequence length.
 * @param Hq Number of query heads.
 * @param Hkv Number of key/value heads.
 * @param Hs Head size (must be 64 or 128).
 * @param stream CUDA stream for asynchronous execution.
 * @return cudaError_t CUDA error status.
 */
template <class floatX>
cudaError_t attention_gpu_backward(floatX* dqkv,
                                   const float* stats,
                                   float scale,
                                   const floatX* out,
                                   const floatX* dout,
                                   const floatX* qkv,
                                   int B,
                                   int T,
                                   int Hq,
                                   int Hkv,
                                   int Hs,
                                   int window_size,
                                   cudaStream_t stream) {
    dim3 grid_dim{(unsigned)Hq, (unsigned)B, (unsigned)T};
    dim3 block_dim{512, 1, 1};
    size_t smem = Hs * sizeof(float) * block_dim.x / 16;
    const size_t dqkv_bytes = static_cast<size_t>(B) * static_cast<size_t>(T) * static_cast<size_t>(Hq + 2 * Hkv) *
                              static_cast<size_t>(Hs) * sizeof(floatX);
    cudaMemsetAsync(dqkv, 0, dqkv_bytes, stream);
    if (Hs == 128) {
        attention_backward_gpu_kernel<128>
            <<<grid_dim, block_dim, smem, stream>>>(dqkv, stats, scale, out, dout, qkv, B, T, Hq, Hkv, window_size);
    } else if (Hs == 64) {
        attention_backward_gpu_kernel<64>
            <<<grid_dim, block_dim, smem, stream>>>(dqkv, stats, scale, out, dout, qkv, B, T, Hq, Hkv, window_size);
    } else if (Hs == 256) {
        attention_backward_gpu_kernel<256>
            <<<grid_dim, block_dim, smem, stream>>>(dqkv, stats, scale, out, dout, qkv, B, T, Hq, Hkv, window_size);
    } else {
        printf("Unsupported head dimension %d\n", Hs);
        return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

/**
 * @brief Backward attention for FP32 tensors (cuDNN-compatible interface).
 *
 * Wrapper that matches the cuDNN attention interface but uses the custom GPU kernel
 * for FP32 support. The workspace parameter is ignored since this implementation
 * doesn't use cuDNN.
 *
 * @param[out] dqkv Output gradient tensor of shape (B, T, Hq + 2*Hkv, HS).
 * @param[in] stats Log-sum-exp statistics from forward pass, shape (B, Hq, T).
 * @param[in] out Forward pass output tensor of shape (B, T, Hq, HS).
 * @param[in] dout Upstream gradient tensor of shape (B, T, Hq, HS).
 * @param[in] qkv Input QKV tensor from forward pass, shape (B, T, Hq + 2*Hkv, HS).
 * @param workspace Unused (for cuDNN interface compatibility).
 * @param B Batch size.
 * @param T Sequence length.
 * @param Hq Number of query heads.
 * @param Hkv Number of key/value heads.
 * @param HS Head size.
 * @param stream CUDA stream for asynchronous execution.
 */
void attention_backward_cudnn(float* dqkv,
                              const float* stats,
                              const float* out,
                              const float* dout,
                              const float* qkv,
                              std::byte* workspace,
                              int B,
                              int T,
                              int Hq,
                              int Hkv,
                              int HS,
                              cudaStream_t stream) {
    attention_gpu_backward(dqkv, stats, 1.f / sqrtf(HS), out, dout, qkv, B, T, Hq, Hkv, HS, /*window_size=*/0, stream);
}

/**
 * @brief Forward attention with Tensor wrapper (dtype dispatch).
 *
 * Dispatches to the appropriate typed implementation based on the output tensor's
 * data type. Supports FP32 and BF16 tensors.
 *
 * @param[out] out Output Tensor of shape (B, T, Hq, HS).
 * @param[out] stats Log-sum-exp statistics Tensor for backward pass, shape (B, Hq, T).
 * @param[in] inp Input QKV Tensor of shape (B, T, Hq + 2*Hkv, HS).
 * @param workspace Workspace Tensor (may be unused for FP32).
 * @param handle cuDNN handle (used for BF16 path).
 * @param B Batch size.
 * @param T Sequence length.
 * @param Hq Number of query heads.
 * @param Hkv Number of key/value heads.
 * @param HS Head size.
 * @param stream CUDA stream for asynchronous execution.
 * @throws std::logic_error If tensor dtype is not FP32 or BF16.
 */
void attention_forward_cudnn(Tensor& out,        // output: (B, T, Hq, HS)
                             Tensor& stats,      // output for backward pass: (B, Hq, T)
                             const Tensor& inp,  // input: (B, T, Hq + Hk + Hv, HS) QKV
                             Tensor& workspace,
                             cudnnHandle_t handle,
                             int B,
                             int T,
                             int Hq,
                             int Hkv,
                             int HS,
                             cudaStream_t stream) {
    std::byte* ws = workspace.get<std::byte>();
    if (out.DType == ETensorDType::FP32) {
        attention_forward_cudnn(out.get<float>(),
                                stats.get<float>(),
                                inp.get<float>(),
                                ws,
                                handle,
                                B,
                                T,
                                Hq,
                                Hkv,
                                HS,
                                stream);
    } else if (out.DType == ETensorDType::BF16) {
        attention_forward_cudnn(out.get<nv_bfloat16>(),
                                stats.get<float>(),
                                inp.get<nv_bfloat16>(),
                                ws,
                                handle,
                                B,
                                T,
                                Hq,
                                Hkv,
                                HS,
                                stream);
    } else {
        throw std::logic_error("attention_forward: unsupported dtype");
    }
}

void attention_forward_custom(Tensor& out,        // output: (B, T, Hq, HS)
                              Tensor& stats,      // output for backward pass: (B, Hq, T)
                              const Tensor& inp,  // input: (B, T, Hq + Hk + Hv, HS) QKV
                              int B,
                              int T,
                              int Hq,
                              int Hkv,
                              int HS,
                              int window_size,
                              cudaStream_t stream,
                              float scale_override) {
    const float scale = (scale_override != 0.0f) ? scale_override : 1.f / sqrtf(static_cast<float>(HS));
    if (out.DType == ETensorDType::FP32) {
        attention_gpu_forward(out.get<float>(),
                              stats.get<float>(),
                              scale,
                              inp.get<float>(),
                              B,
                              T,
                              Hq,
                              Hkv,
                              HS,
                              window_size,
                              stream);
    } else if (out.DType == ETensorDType::BF16) {
        attention_gpu_forward(out.get<nv_bfloat16>(),
                              stats.get<float>(),
                              scale,
                              inp.get<nv_bfloat16>(),
                              B,
                              T,
                              Hq,
                              Hkv,
                              HS,
                              window_size,
                              stream);
    } else {
        throw std::logic_error("attention_forward_custom: unsupported dtype");
    }
}

void attention_backward_custom(Tensor& dqkv,
                               const Tensor& stats,
                               const Tensor& out,
                               const Tensor& dout,
                               const Tensor& qkv,
                               int B,
                               int T,
                               int Hq,
                               int Hkv,
                               int HS,
                               int window_size,
                               cudaStream_t stream,
                               float scale_override) {
    const float scale = (scale_override != 0.0f) ? scale_override : 1.f / sqrtf(static_cast<float>(HS));
    if (out.DType == ETensorDType::FP32) {
        attention_gpu_backward(dqkv.get<float>(),
                               stats.get<float>(),
                               scale,
                               out.get<float>(),
                               dout.get<float>(),
                               qkv.get<float>(),
                               B,
                               T,
                               Hq,
                               Hkv,
                               HS,
                               window_size,
                               stream);
    } else {
        throw std::logic_error("attention_backward_custom: unsupported dtype");
    }
}

/**
 * @brief Backward attention with Tensor wrapper (dtype dispatch).
 *
 * Dispatches to the appropriate typed implementation based on the output tensor's
 * data type. Supports FP32 and BF16 tensors.
 *
 * @param[out] dqkv Output gradient Tensor of shape (B, T, Hq + 2*Hkv, HS).
 * @param[in] stats Log-sum-exp statistics Tensor from forward pass, shape (B, Hq, T).
 * @param[in] out Forward pass output Tensor of shape (B, T, Hq, HS).
 * @param[in] dout Upstream gradient Tensor of shape (B, T, Hq, HS).
 * @param[in] qkv Input QKV Tensor from forward pass, shape (B, T, Hq + 2*Hkv, HS).
 * @param workspace Workspace Tensor (may be unused for FP32).
 * @param handle cuDNN handle (used for BF16 path).
 * @param B Batch size.
 * @param T Sequence length.
 * @param Hq Number of query heads.
 * @param Hkv Number of key/value heads.
 * @param HS Head size.
 * @param stream CUDA stream for asynchronous execution.
 * @throws std::logic_error If tensor dtype is not FP32 or BF16.
 */
void attention_backward_cudnn(Tensor& dqkv,
                              const Tensor& stats,
                              const Tensor& out,
                              const Tensor& dout,
                              const Tensor& qkv,
                              Tensor& workspace,
                              cudnnHandle_t handle,
                              int B,
                              int T,
                              int Hq,
                              int Hkv,
                              int HS,
                              cudaStream_t stream) {
    std::byte* ws = workspace.get<std::byte>();
    if (out.DType == ETensorDType::FP32) {
        attention_backward_cudnn(dqkv.get<float>(),
                                 stats.get<float>(),
                                 out.get<float>(),
                                 dout.get<float>(),
                                 qkv.get<float>(),
                                 ws,
                                 B,
                                 T,
                                 Hq,
                                 Hkv,
                                 HS,
                                 stream);
    } else if (out.DType == ETensorDType::BF16) {
        // Argument order is now consistent: out, dout, qkv (matching header declaration)
        attention_backward_cudnn(dqkv.get<nv_bfloat16>(),
                                 stats.get<float>(),
                                 out.get<nv_bfloat16>(),
                                 dout.get<nv_bfloat16>(),
                                 qkv.get<nv_bfloat16>(),
                                 ws,
                                 handle,
                                 B,
                                 T,
                                 Hq,
                                 Hkv,
                                 HS,
                                 stream);
    } else {
        throw std::logic_error("attention_backward: unsupported dtype");
    }
}

// ============================================================================
// Attention sinks (GPT-OSS): adjust output + LSE after forward, and compute sink grads.
// ============================================================================
template <typename T>
__global__ void attention_apply_sinks_kernel(T* __restrict__ out,
                                             float* __restrict__ lse,
                                             const T* __restrict__ sinks,
                                             int B,
                                             int Tseq,
                                             int Hq,
                                             int Hs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * Tseq * Hq;
    if (idx >= total) return;
    int b = idx / (Hq * Tseq);
    int rem = idx % (Hq * Tseq);
    int h = rem / Tseq;
    int t = rem % Tseq;

    float l = lse[(b * Hq + h) * Tseq + t];
    float s = static_cast<float>(sinks[h]);
    float maxv = fmaxf(l, s);
    float lse_new = maxv + logf(expf(l - maxv) + expf(s - maxv));
    float scale = expf(l - lse_new);

    lse[(b * Hq + h) * Tseq + t] = lse_new;

    T* out_ptr = out + ((b * Tseq + t) * Hq + h) * Hs;
    for (int i = 0; i < Hs; ++i) {
        float v = static_cast<float>(out_ptr[i]) * scale;
        out_ptr[i] = static_cast<T>(v);
    }
}

template <typename T>
__global__ void attention_sinks_backward_kernel(float* __restrict__ d_sinks,
                                                const T* __restrict__ out,
                                                const T* __restrict__ dout,
                                                const float* __restrict__ lse,
                                                const T* __restrict__ sinks,
                                                int B,
                                                int Tseq,
                                                int Hq,
                                                int Hs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * Tseq * Hq;
    if (idx >= total) return;
    int b = idx / (Hq * Tseq);
    int rem = idx % (Hq * Tseq);
    int h = rem / Tseq;
    int t = rem % Tseq;

    const T* out_ptr = out + ((b * Tseq + t) * Hq + h) * Hs;
    const T* dout_ptr = dout + ((b * Tseq + t) * Hq + h) * Hs;
    float D = 0.0f;
    for (int i = 0; i < Hs; ++i) {
        D += static_cast<float>(out_ptr[i]) * static_cast<float>(dout_ptr[i]);
    }

    float l = lse[(b * Hq + h) * Tseq + t];
    float s = static_cast<float>(sinks[h]);
    float p_sink = expf(s - l);
    float grad = -p_sink * D;
    atomicAdd(&d_sinks[h], grad);
}

void attention_apply_sinks(nv_bfloat16* out,
                           float* lse,
                           const nv_bfloat16* sinks,
                           int B,
                           int T,
                           int Hq,
                           int Hs,
                           cudaStream_t stream) {
    int total = B * T * Hq;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    attention_apply_sinks_kernel<<<grid_size, block_size, 0, stream>>>(out, lse, sinks, B, T, Hq, Hs);
}

void attention_apply_sinks(float* out,
                           float* lse,
                           const float* sinks,
                           int B,
                           int T,
                           int Hq,
                           int Hs,
                           cudaStream_t stream) {
    int total = B * T * Hq;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    attention_apply_sinks_kernel<<<grid_size, block_size, 0, stream>>>(out, lse, sinks, B, T, Hq, Hs);
}

void attention_sinks_backward(float* d_sinks,
                              const nv_bfloat16* out,
                              const nv_bfloat16* dout,
                              const float* lse,
                              const nv_bfloat16* sinks,
                              int B,
                              int T,
                              int Hq,
                              int Hs,
                              cudaStream_t stream) {
    int total = B * T * Hq;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    attention_sinks_backward_kernel<<<grid_size, block_size, 0, stream>>>(d_sinks, out, dout, lse, sinks, B, T, Hq, Hs);
}

void attention_sinks_backward(float* d_sinks,
                              const float* out,
                              const float* dout,
                              const float* lse,
                              const float* sinks,
                              int B,
                              int T,
                              int Hq,
                              int Hs,
                              cudaStream_t stream) {
    int total = B * T * Hq;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    attention_sinks_backward_kernel<<<grid_size, block_size, 0, stream>>>(d_sinks, out, dout, lse, sinks, B, T, Hq, Hs);
}

// ============================================================================
// cuBLAS matmul-based attention (SDPA math equivalent)
// No head_dim limit. Used for head_dim > 256 where FA2/cuDNN are unsupported.
// ============================================================================

__global__ void causal_softmax_lse_kernel(float* scores, float* lse_out, int T, float scale) {
    int bh = blockIdx.x, t = blockIdx.y;
    float* row = scores + (static_cast<long long>(bh) * T + t) * T;
    __shared__ float sdata[32];
    // Scale + max
    float mx = -1e30f;
    for (int j = threadIdx.x; j <= t; j += blockDim.x) {
        row[j] *= scale;
        mx = fmaxf(mx, row[j]);
    }
    for (int o = 16; o > 0; o >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, o));
    if (threadIdx.x % 32 == 0) sdata[threadIdx.x / 32] = mx;
    __syncthreads();
    if (threadIdx.x < blockDim.x / 32) {
        mx = sdata[threadIdx.x];
        for (int o = 16; o > 0; o >>= 1)
            mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, o));
        if (threadIdx.x == 0) sdata[0] = mx;
    }
    __syncthreads();
    mx = sdata[0];
    // Exp + sum
    float se = 0.0f;
    for (int j = threadIdx.x; j <= t; j += blockDim.x) {
        float v = expf(row[j] - mx);
        row[j] = v;
        se += v;
    }
    for (int j = t + 1 + threadIdx.x; j < T; j += blockDim.x)
        row[j] = 0.0f;
    for (int o = 16; o > 0; o >>= 1)
        se += __shfl_down_sync(0xffffffff, se, o);
    if (threadIdx.x % 32 == 0) sdata[threadIdx.x / 32] = se;
    __syncthreads();
    if (threadIdx.x < blockDim.x / 32) {
        se = sdata[threadIdx.x];
        for (int o = 16; o > 0; o >>= 1)
            se += __shfl_down_sync(0xffffffff, se, o);
        if (threadIdx.x == 0) sdata[0] = se;
    }
    __syncthreads();
    se = sdata[0];
    float inv = (se > 0.f) ? 1.f / se : 0.f;
    for (int j = threadIdx.x; j < T; j += blockDim.x)
        row[j] *= inv;
    if (threadIdx.x == 0 && lse_out) lse_out[static_cast<long long>(bh) * T + t] = mx + logf(fmaxf(se, 1e-20f));
}

// Gather BF16 rows with stride into contiguous BF16 buffer (no dtype conversion).
__global__ void bf16_gather_rows_k(nv_bfloat16* dst, const nv_bfloat16* src, int T, int HS, int stride) {
    int t = blockIdx.x, e = blockIdx.y * blockDim.x + threadIdx.x;
    if (t < T && e < HS) dst[t * HS + e] = src[static_cast<long long>(t) * stride + e];
}

// Scatter BF16 rows from contiguous buffer back into strided layout.
__global__ void bf16_scatter_rows_k(nv_bfloat16* dst, const nv_bfloat16* src, int T, int HS, int stride) {
    int t = blockIdx.x, e = blockIdx.y * blockDim.x + threadIdx.x;
    if (t < T && e < HS) dst[static_cast<long long>(t) * stride + e] = src[t * HS + e];
}

// Causal softmax matching HF eager precision:
//   BF16 scores * BF16 scale → BF16 (truncated), then FP32 softmax → BF16 output
// HF: `scores = matmul(Q,K^T) * scaling` in BF16, then `softmax(scores, dtype=float32).to(bf16)`
__global__ void causal_softmax_bf16_kernel(nv_bfloat16* scores_bf16, float* lse_out, int T, nv_bfloat16 scale_bf16) {
    int bh = blockIdx.x, t = blockIdx.y;
    nv_bfloat16* row = scores_bf16 + (static_cast<long long>(bh) * T + t) * T;
    __shared__ float sdata[32];

    // Apply BF16 scaling in-place (matches HF's `* scaling` in BF16 space)
    for (int j = threadIdx.x; j <= t; j += blockDim.x)
        row[j] = __float2bfloat16(__bfloat162float(row[j]) * __bfloat162float(scale_bf16));
    for (int j = t + 1 + threadIdx.x; j < T; j += blockDim.x)
        row[j] = __float2bfloat16(-1e30f);  // causal mask: -inf for future positions
    __syncthreads();

    // Block-wide max reduction. Stage 2 is skipped for single-warp blocks
    // (blockDim.x <= 32) because stage 1 already produced the full answer
    // in thread 0. Stage 2 used `__shfl_down_sync(0xffffffff, ...)` which
    // deadlocks when fewer than 32 threads enter the divergent branch
    // (e.g., T=32 → blockDim.x=32 → num_warps=1 → only thread 0 enters).
    const int num_warps_local = (blockDim.x + 31) / 32;

    // Softmax in FP32 (matches HF's `softmax(scores, dtype=torch.float32)`)
    float mx = -1e30f;
    for (int j = threadIdx.x; j < T; j += blockDim.x)
        mx = fmaxf(mx, __bfloat162float(row[j]));
    for (int o = 16; o > 0; o >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, o));
    if (threadIdx.x % 32 == 0) sdata[threadIdx.x / 32] = mx;
    __syncthreads();
    if (num_warps_local > 1) {
        if (threadIdx.x < num_warps_local) {
            mx = sdata[threadIdx.x];
            const unsigned mask = (num_warps_local == 32) ? 0xffffffffu : ((1u << num_warps_local) - 1u);
            for (int o = num_warps_local >> 1; o > 0; o >>= 1)
                mx = fmaxf(mx, __shfl_down_sync(mask, mx, o));
            if (threadIdx.x == 0) sdata[0] = mx;
        }
        __syncthreads();
    }
    mx = sdata[0];

    // Exp + sum
    float se = 0.0f;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        float v = expf(__bfloat162float(row[j]) - mx);
        row[j] = __float2bfloat16(v);
        se += v;
    }
    for (int o = 16; o > 0; o >>= 1)
        se += __shfl_down_sync(0xffffffff, se, o);
    if (threadIdx.x % 32 == 0) sdata[threadIdx.x / 32] = se;
    __syncthreads();
    if (num_warps_local > 1) {
        if (threadIdx.x < num_warps_local) {
            se = sdata[threadIdx.x];
            const unsigned mask = (num_warps_local == 32) ? 0xffffffffu : ((1u << num_warps_local) - 1u);
            for (int o = num_warps_local >> 1; o > 0; o >>= 1)
                se += __shfl_down_sync(mask, se, o);
            if (threadIdx.x == 0) sdata[0] = se;
        }
        __syncthreads();
    }
    se = sdata[0];

    // Normalize → BF16 (matches HF's `.to(query.dtype)`)
    float inv = (se > 0.f) ? 1.f / se : 0.f;
    for (int j = threadIdx.x; j < T; j += blockDim.x)
        row[j] = __float2bfloat16(__bfloat162float(row[j]) * inv);

    // LSE for backward
    if (threadIdx.x == 0 && lse_out) lse_out[static_cast<long long>(bh) * T + t] = mx + logf(fmaxf(se, 1e-20f));
}

void attention_forward_matmul(Tensor& out,
                              Tensor& stats,
                              const Tensor& qkv,
                              int B,
                              int T,
                              int Hq,
                              int Hkv,
                              int HS,
                              cublasHandle_t cublas,
                              cudaStream_t stream,
                              float scale_override) {
    const int H = Hq + 2 * Hkv;
    const float scale = (scale_override != 0.0f) ? scale_override : 1.f / sqrtf(static_cast<float>(HS));
    cublasSetStream(cublas, stream);

    nv_bfloat16 *d_scores, *d_q, *d_k, *d_v;
    CUDA_CHECK(cudaMallocAsync(&d_scores, sizeof(nv_bfloat16) * B * Hq * T * T, stream));
    CUDA_CHECK(cudaMallocAsync(&d_q, sizeof(nv_bfloat16) * T * HS, stream));
    CUDA_CHECK(cudaMallocAsync(&d_k, sizeof(nv_bfloat16) * T * HS, stream));
    CUDA_CHECK(cudaMallocAsync(&d_v, sizeof(nv_bfloat16) * T * HS, stream));

    const auto* p = qkv.get<nv_bfloat16>();
    const int st = H * HS;
    dim3 gg(T, (HS + 255) / 256);
    float alpha = 1.f, beta = 0.f;

    // Phase 1: Q@K^T → BF16 scores (per head, per batch)
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < Hq; ++h) {
            int hkv = h * Hkv / Hq;
            bf16_gather_rows_k<<<gg, 256, 0, stream>>>(d_q, p + (long)b * T * st + h * HS, T, HS, st);
            bf16_gather_rows_k<<<gg, 256, 0, stream>>>(d_k, p + (long)b * T * st + (Hq + hkv) * HS, T, HS, st);
            nv_bfloat16* s = d_scores + ((long)b * Hq + h) * T * T;
            // BF16 GEMM: scores = K^T @ Q, BF16 in/out with FP32 accumulation
            cublasGemmEx(cublas,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         T,
                         T,
                         HS,
                         &alpha,
                         d_k,
                         CUDA_R_16BF,
                         HS,
                         d_q,
                         CUDA_R_16BF,
                         HS,
                         &beta,
                         s,
                         CUDA_R_16BF,
                         T,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);
        }
    }

    // Phase 2: Causal softmax (BF16 scale → FP32 softmax → BF16, matching HF eager)
    nv_bfloat16 scale_bf16 = __float2bfloat16(scale);
    causal_softmax_bf16_kernel<<<dim3(B * Hq, T), std::min(256, T), 0, stream>>>(d_scores,
                                                                                 stats.get<float>(),
                                                                                 T,
                                                                                 scale_bf16);

    // Phase 3: attn@V → BF16 output (per head, per batch)
    auto* out_p = out.get<nv_bfloat16>();
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < Hq; ++h) {
            int hkv = h * Hkv / Hq;
            bf16_gather_rows_k<<<gg, 256, 0, stream>>>(d_v, p + (long)b * T * st + (Hq + Hkv + hkv) * HS, T, HS, st);
            nv_bfloat16* s = d_scores + ((long)b * Hq + h) * T * T;
            // BF16 GEMM: output = V @ softmax_weights
            // Output goes to a temp buffer (d_q reused), then scattered to strided layout.
            cublasGemmEx(cublas,
                         CUBLAS_OP_N,
                         CUBLAS_OP_N,
                         HS,
                         T,
                         T,
                         &alpha,
                         d_v,
                         CUDA_R_16BF,
                         HS,
                         s,
                         CUDA_R_16BF,
                         T,
                         &beta,
                         d_q,
                         CUDA_R_16BF,
                         HS,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);
            bf16_scatter_rows_k<<<gg, 256, 0, stream>>>(out_p + (long)b * T * Hq * HS + h * HS, d_q, T, HS, Hq * HS);
        }
    }

    CUDA_CHECK(cudaFreeAsync(d_scores, stream));
    CUDA_CHECK(cudaFreeAsync(d_q, stream));
    CUDA_CHECK(cudaFreeAsync(d_k, stream));
    CUDA_CHECK(cudaFreeAsync(d_v, stream));
}

// Softmax backward kernel: given d_out and P (softmax output),
// compute d_S = P * (d_P - rowsum(d_P * P)) where d_P = d_out @ V^T.
// This kernel takes pre-computed d_P (in BF16) and P (in BF16),
// produces d_S in BF16 with FP32 intermediates.
// Also applies the 1/sqrt(d) scaling to d_S for dQ/dK computation.
__global__ void causal_softmax_backward_bf16_kernel(nv_bfloat16* d_scores,   // [BH, T, T] output: d_S * scale
                                                    const nv_bfloat16* d_P,  // [BH, T, T] input: d_out @ V^T
                                                    const nv_bfloat16* P,    // [BH, T, T] input: softmax output
                                                    int T,
                                                    float scale) {
    int bh = blockIdx.x, t = blockIdx.y;
    const nv_bfloat16* dp_row = d_P + (static_cast<long long>(bh) * T + t) * T;
    const nv_bfloat16* p_row = P + (static_cast<long long>(bh) * T + t) * T;
    nv_bfloat16* ds_row = d_scores + (static_cast<long long>(bh) * T + t) * T;
    __shared__ float sdata[32];

    // Compute rowsum(d_P * P) in FP32. Two-stage reduction:
    //   (1) each warp reduces via __shfl_down_sync → lane 0 holds warp sum
    //   (2) one warp reduces the warp sums from sdata → thread 0 holds block sum
    // NOTE: __shfl_down_sync with mask 0xffffffff requires ALL 32 threads in
    // the warp to be active. If fewer than 32 threads enter the second stage,
    // the warp deadlocks. We therefore:
    //   - Skip stage 2 when blockDim.x <= 32 (single warp — stage 1 already
    //     has the block sum in thread 0).
    //   - In stage 2, use a mask reflecting the actual active threads.
    float dot = 0.f;
    for (int j = threadIdx.x; j <= t; j += blockDim.x)
        dot += __bfloat162float(dp_row[j]) * __bfloat162float(p_row[j]);
    for (int o = 16; o > 0; o >>= 1)
        dot += __shfl_down_sync(0xffffffff, dot, o);
    if (threadIdx.x % 32 == 0) sdata[threadIdx.x / 32] = dot;
    __syncthreads();
    const int num_warps = (blockDim.x + 31) / 32;
    if (num_warps > 1) {
        if (threadIdx.x < num_warps) {
            dot = sdata[threadIdx.x];
            const unsigned active_mask = (num_warps == 32) ? 0xffffffffu : ((1u << num_warps) - 1u);
            for (int o = num_warps >> 1; o > 0; o >>= 1)
                dot += __shfl_down_sync(active_mask, dot, o);
            if (threadIdx.x == 0) sdata[0] = dot;
        }
        __syncthreads();
    }
    // Single warp: sdata[0] already holds the full block sum (written by
    // thread 0 in stage 1). Multi-warp: thread 0 above just wrote it. Either
    // way, every thread can now load the block sum from sdata[0].
    dot = sdata[0];

    // d_S = P * (d_P - dot) * scale; zero for future positions (causal)
    for (int j = threadIdx.x; j <= t; j += blockDim.x) {
        float p = __bfloat162float(p_row[j]);
        float dp = __bfloat162float(dp_row[j]);
        ds_row[j] = __float2bfloat16(p * (dp - dot) * scale);
    }
    for (int j = t + 1 + threadIdx.x; j < T; j += blockDim.x)
        ds_row[j] = __float2bfloat16(0.f);
}

void attention_backward_matmul(Tensor& d_qkv,
                               const Tensor& lse,
                               const Tensor& out,
                               const Tensor& d_out,
                               const Tensor& qkv,
                               int B,
                               int T,
                               int Hq,
                               int Hkv,
                               int HS,
                               cublasHandle_t cublas,
                               cudaStream_t stream,
                               float scale_override) {
    // SDPA backward using cuBLAS GEMMs, matching the forward's BF16 precision.
    //
    // Per-(b, h) we compute:
    //   dV_h = dO_h @ P_h^T
    //   dP_h = dO_h @ V_h^T
    //   dS_h = P_h * (dP_h - rowsum(dP_h * P_h)) * scale
    //   dQ_h = dS_h @ K_h
    //   dK_h = dS_h^T @ Q_h
    //
    // GQA layout (Hq > Hkv): multiple Q heads share one (K, V) head, so
    // dK and dV must be SUMMED across Q heads that map to the same K/V slot.
    // Earlier versions scattered each head's dK/dV directly into d_qkv's
    // single K/V slot, overwriting prior heads — for Hkv=1 only the last Q
    // head's contribution survived. We now write per-Q-head dK/dV into
    // expanded buffers [B, T, Hq, HS] and finalize with reduce_scatter_dkv,
    // mirroring the Flash-Attn varlen backward path.
    const int H = Hq + 2 * Hkv;
    const float scale = (scale_override != 0.0f) ? scale_override : 1.f / sqrtf(static_cast<float>(HS));
    cublasSetStream(cublas, stream);

    const bool is_gqa = (Hq != Hkv);

    nv_bfloat16 *w_q, *w_k, *w_v, *w_do, *w_scores, *w_dp;
    CUDA_CHECK(cudaMallocAsync(&w_q, sizeof(nv_bfloat16) * T * HS, stream));
    CUDA_CHECK(cudaMallocAsync(&w_k, sizeof(nv_bfloat16) * T * HS, stream));
    CUDA_CHECK(cudaMallocAsync(&w_v, sizeof(nv_bfloat16) * T * HS, stream));
    CUDA_CHECK(cudaMallocAsync(&w_do, sizeof(nv_bfloat16) * T * HS, stream));
    CUDA_CHECK(cudaMallocAsync(&w_scores, sizeof(nv_bfloat16) * T * T, stream));
    CUDA_CHECK(cudaMallocAsync(&w_dp, sizeof(nv_bfloat16) * T * T, stream));

    // Expanded per-Q-head dK/dV buffers (GQA only). For MHA they're unused.
    nv_bfloat16* dk_expanded = nullptr;
    nv_bfloat16* dv_expanded = nullptr;
    if (is_gqa) {
        const std::size_t expanded_bytes = sizeof(nv_bfloat16) * static_cast<std::size_t>(B) *
                                           static_cast<std::size_t>(T) * static_cast<std::size_t>(Hq) *
                                           static_cast<std::size_t>(HS);
        CUDA_CHECK(cudaMallocAsync(&dk_expanded, expanded_bytes, stream));
        CUDA_CHECK(cudaMallocAsync(&dv_expanded, expanded_bytes, stream));
    }

    const auto* p_qkv = qkv.get<nv_bfloat16>();
    const auto* p_do = d_out.get<nv_bfloat16>();
    auto* p_dqkv = d_qkv.get<nv_bfloat16>();
    const int st = H * HS;
    const int st_exp = Hq * HS;  // stride for expanded dK/dV buffers
    dim3 gg(T, (HS + 255) / 256);
    float alpha = 1.f, beta = 0.f;
    nv_bfloat16 scale_bf16 = __float2bfloat16(scale);

    // Zero d_qkv — we write Q directly per head; K/V come from
    // reduce_scatter_dkv (GQA) or direct per-head scatter (MHA).
    CUDA_CHECK(cudaMemsetAsync(d_qkv.Data, 0, d_qkv.bytes(), stream));

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < Hq; ++h) {
            const int hkv = h * Hkv / Hq;

            // Gather Q, K, V, d_out for this head
            bf16_gather_rows_k<<<gg, 256, 0, stream>>>(w_q, p_qkv + (long)b * T * st + h * HS, T, HS, st);
            bf16_gather_rows_k<<<gg, 256, 0, stream>>>(w_k, p_qkv + (long)b * T * st + (Hq + hkv) * HS, T, HS, st);
            bf16_gather_rows_k<<<gg, 256, 0, stream>>>(w_v,
                                                       p_qkv + (long)b * T * st + (Hq + Hkv + hkv) * HS,
                                                       T,
                                                       HS,
                                                       st);
            bf16_gather_rows_k<<<gg, 256, 0, stream>>>(w_do, p_do + (long)b * T * Hq * HS + h * HS, T, HS, Hq * HS);

            // 1. Recompute scores: S = K^T @ Q   [T, T] col-major
            cublasGemmEx(cublas,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         T,
                         T,
                         HS,
                         &alpha,
                         w_k,
                         CUDA_R_16BF,
                         HS,
                         w_q,
                         CUDA_R_16BF,
                         HS,
                         &beta,
                         w_scores,
                         CUDA_R_16BF,
                         T,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            // 2. Recompute softmax: P = causal_softmax(S * scale). w_scores := P.
            causal_softmax_bf16_kernel<<<dim3(1, T), std::min(256, T), 0, stream>>>(w_scores, nullptr, T, scale_bf16);

            // 3. dV_h = dO @ P^T   (col-major HS×T = dO(HS,T) @ P(T,T)^T).
            //    Write into w_k (K_h is no longer needed until step 6,
            //    where we re-gather it).
            cublasGemmEx(cublas,
                         CUBLAS_OP_N,
                         CUBLAS_OP_T,
                         HS,
                         T,
                         T,
                         &alpha,
                         w_do,
                         CUDA_R_16BF,
                         HS,
                         w_scores,
                         CUDA_R_16BF,
                         T,
                         &beta,
                         w_k,
                         CUDA_R_16BF,
                         HS,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);
            if (is_gqa) {
                // Scatter per-Q-head into expanded dV buffer at stride Hq*HS.
                bf16_scatter_rows_k<<<gg, 256, 0, stream>>>(dv_expanded + (long)b * T * st_exp + h * HS,
                                                            w_k,
                                                            T,
                                                            HS,
                                                            st_exp);
            } else {
                // MHA: single K/V slot per Q head — direct write is correct.
                bf16_scatter_rows_k<<<gg, 256, 0, stream>>>(p_dqkv + (long)b * T * st + (Hq + Hkv + hkv) * HS,
                                                            w_k,
                                                            T,
                                                            HS,
                                                            st);
            }

            // 4. dP = dO @ V^T   [T, T]. w_v still holds V_h.
            cublasGemmEx(cublas,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         T,
                         T,
                         HS,
                         &alpha,
                         w_v,
                         CUDA_R_16BF,
                         HS,
                         w_do,
                         CUDA_R_16BF,
                         HS,
                         &beta,
                         w_dp,
                         CUDA_R_16BF,
                         T,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);

            // 5. dS = P * (dP - rowsum(dP * P)) * scale. w_dp := dS.
            causal_softmax_backward_bf16_kernel<<<dim3(1, T), std::min(256, T), 0, stream>>>(w_dp,
                                                                                             w_dp,
                                                                                             w_scores,
                                                                                             T,
                                                                                             scale);

            // 6. dQ_h = K @ dS^T   (col-major HS×T = K(HS,T) @ dS(T,T)^T).
            //    Re-gather K because w_k was reused for dV_h in step 3.
            bf16_gather_rows_k<<<gg, 256, 0, stream>>>(w_k, p_qkv + (long)b * T * st + (Hq + hkv) * HS, T, HS, st);
            cublasGemmEx(cublas,
                         CUBLAS_OP_N,
                         CUBLAS_OP_T,
                         HS,
                         T,
                         T,
                         &alpha,
                         w_k,
                         CUDA_R_16BF,
                         HS,
                         w_dp,
                         CUDA_R_16BF,
                         T,
                         &beta,
                         w_q,
                         CUDA_R_16BF,
                         HS,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);
            // dQ is unique per Q head — direct write into d_qkv.
            bf16_scatter_rows_k<<<gg, 256, 0, stream>>>(p_dqkv + (long)b * T * st + h * HS, w_q, T, HS, st);

            // 7. dK_h = Q @ dS   (col-major HS×T = Q(HS,T) @ dS(T,T)).
            //    Re-gather Q because w_q was reused for dQ_h in step 6.
            bf16_gather_rows_k<<<gg, 256, 0, stream>>>(w_q, p_qkv + (long)b * T * st + h * HS, T, HS, st);
            cublasGemmEx(cublas,
                         CUBLAS_OP_N,
                         CUBLAS_OP_N,
                         HS,
                         T,
                         T,
                         &alpha,
                         w_q,
                         CUDA_R_16BF,
                         HS,
                         w_dp,
                         CUDA_R_16BF,
                         T,
                         &beta,
                         w_v,
                         CUDA_R_16BF,
                         HS,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);
            if (is_gqa) {
                bf16_scatter_rows_k<<<gg, 256, 0, stream>>>(dk_expanded + (long)b * T * st_exp + h * HS,
                                                            w_v,
                                                            T,
                                                            HS,
                                                            st_exp);
            } else {
                bf16_scatter_rows_k<<<gg, 256, 0, stream>>>(p_dqkv + (long)b * T * st + (Hq + hkv) * HS,
                                                            w_v,
                                                            T,
                                                            HS,
                                                            st);
            }
        }
    }

    // GQA finalize: reduce Hq→Hkv and scatter into d_qkv's K and V sections.
    // reduce_scatter_dkv treats the first B*T rows as total_q and sums
    // h_ratio = Hq/Hkv expanded heads into each KV slot.
    if (is_gqa) {
        reduce_scatter_dkv(p_dqkv, dk_expanded, dv_expanded, B * T, Hq, Hkv, HS, stream);
        CUDA_CHECK(cudaFreeAsync(dk_expanded, stream));
        CUDA_CHECK(cudaFreeAsync(dv_expanded, stream));
    }

    CUDA_CHECK(cudaFreeAsync(w_q, stream));
    CUDA_CHECK(cudaFreeAsync(w_k, stream));
    CUDA_CHECK(cudaFreeAsync(w_v, stream));
    CUDA_CHECK(cudaFreeAsync(w_do, stream));
    CUDA_CHECK(cudaFreeAsync(w_scores, stream));
    CUDA_CHECK(cudaFreeAsync(w_dp, stream));
}
