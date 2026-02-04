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
template<int E, class scalar_t>
__global__ void __launch_bounds__(512) attention_forward_gpu_kernel(
    scalar_t* out, float* stats, float scale,
    const scalar_t* qkv,
    int B, int T, int Hq, int Hkv) {
    constexpr const int SubWarpSize = 16;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    auto sub_warp = cg::tiled_partition<SubWarpSize>(block);

    extern __shared__ float scratch[];

    int h = blockIdx.x;
    int b = blockIdx.y;
    int t = blockIdx.z;

    int hkv = h * Hkv / Hq;
    int TH = Hq + 2*Hkv;
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

    for (int l = sub_warp.meta_group_rank(); l <= t; l += sub_warp.meta_group_size()) {
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
    if(threadIdx.x == 0) {
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
template<int E, class scalar_t>
__global__ void __launch_bounds__(512) attention_backward_gpu_kernel(
        scalar_t* dqkv, const float* stats, float scale,
        const scalar_t* out, const scalar_t* dout, const scalar_t* qkv,
        int B, int T, int Hq, int Hkv) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int t = blockIdx.z;

    const int hkv = h * Hkv / Hq;
    const int TH = Hq + 2*Hkv;

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
    for(int i = 0; i < E; ++i) {
        D += dout[t * Hq * E + i] * out[t * Hq * E + i];
    }

    ptrdiff_t q_offset = t * TH * E;
    for (int l = threadIdx.x; l <= t; l += blockDim.x) {
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
template<class floatX>
cudaError_t attention_gpu_forward(floatX* out, float* stats, float scale,
                          const floatX* qkv,
                          int B, int T, int Hq, int Hkv, int Hs, cudaStream_t stream) {
    dim3 grid_dim{(unsigned)Hq, (unsigned)B, (unsigned)T};
    dim3 block_dim{512, 1, 1};
    size_t smem = Hs * sizeof(float) * block_dim.x / 16;

    if (Hs == 128) {
        attention_forward_gpu_kernel<128><<<grid_dim, block_dim, smem, stream>>>(
            out, stats, scale, qkv, B, T, Hq, Hkv);
    } else if (Hs == 64) {
        attention_forward_gpu_kernel<64><<<grid_dim, block_dim, smem, stream>>>(
            out, stats, scale, qkv,  B, T, Hq, Hkv);
    } else {
        printf("Unsupported head dimension");
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
void attention_forward_cudnn(float* out,  // output: (B, T, Nq, HS)
                             float* stats, // output for backward pass: (B, Hq, T)
                             const float* inp,  // input: (B, T, Hq + 2Hkv, HS) QKV
                             std::byte* workspace, cudnnHandle_t handle,
                             int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream) {
    attention_gpu_forward(out, stats, 1.f / sqrtf(HS), inp, B, T, Hq, Hkv, HS, stream);
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
template<class floatX>
cudaError_t attention_gpu_backward(floatX* dqkv, const float* stats, float scale,
                                   const floatX* out, const floatX* dout, const floatX* qkv,
                                   int B, int T, int Hq, int Hkv, int Hs, cudaStream_t stream) {
    dim3 grid_dim{(unsigned)Hq, (unsigned)B, (unsigned)T};
    dim3 block_dim{512, 1, 1};
    size_t smem = Hs * sizeof(float) * block_dim.x / 16;
    const size_t dqkv_bytes =
        static_cast<size_t>(B) * static_cast<size_t>(T) *
        static_cast<size_t>(Hq + 2 * Hkv) * static_cast<size_t>(Hs) * sizeof(floatX);
    cudaMemsetAsync(dqkv, 0, dqkv_bytes, stream);
    if (Hs == 128) {
        attention_backward_gpu_kernel<128><<<grid_dim, block_dim, smem, stream>>>(
            dqkv, stats, scale, out, dout, qkv, B, T, Hq, Hkv);
    } else if (Hs == 64) {
        attention_backward_gpu_kernel<64><<<grid_dim, block_dim, smem, stream>>>(
            dqkv, stats, scale, out, dout, qkv, B, T, Hq, Hkv);
    } else {
        printf("Unsupported head dimension");
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
void attention_backward_cudnn(float* dqkv, const float* stats,
                              const float* out, const float* dout, const float* qkv, std::byte* workspace,
                              int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream) {
    attention_gpu_backward(dqkv, stats, 1.f / sqrtf(HS), out, dout, qkv, B, T, Hq, Hkv, HS, stream);
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
void attention_forward_cudnn(Tensor& out,  // output: (B, T, Hq, HS)
                             Tensor& stats, // output for backward pass: (B, Hq, T)
                             const Tensor& inp,  // input: (B, T, Hq + Hk + Hv, HS) QKV
                             Tensor& workspace, cudnnHandle_t handle,
                             int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream) {
    std::byte* ws = workspace.get<std::byte>();
    if(out.DType == ETensorDType::FP32) {
        attention_forward_cudnn(out.get<float>(), stats.get<float>(), inp.get<float>(), ws, handle, B, T, Hq, Hkv, HS, stream);
    } else if(out.DType == ETensorDType::BF16) {
        attention_forward_cudnn(out.get<nv_bfloat16>(), stats.get<float>(), inp.get<nv_bfloat16>(), ws, handle, B, T, Hq, Hkv, HS, stream);
    } else {
        throw std::logic_error("attention_forward: unsupported dtype");
    }
}

void attention_forward_custom(Tensor& out,  // output: (B, T, Hq, HS)
                              Tensor& stats, // output for backward pass: (B, Hq, T)
                              const Tensor& inp,  // input: (B, T, Hq + Hk + Hv, HS) QKV
                              int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream) {
    if (out.DType == ETensorDType::FP32) {
        attention_gpu_forward(out.get<float>(), stats.get<float>(), 1.f / sqrtf(HS),
                              inp.get<float>(), B, T, Hq, Hkv, HS, stream);
    } else if (out.DType == ETensorDType::BF16) {
        attention_gpu_forward(out.get<nv_bfloat16>(), stats.get<float>(), 1.f / sqrtf(HS),
                              inp.get<nv_bfloat16>(), B, T, Hq, Hkv, HS, stream);
    } else {
        throw std::logic_error("attention_forward_custom: unsupported dtype");
    }
}

void attention_backward_custom(Tensor& dqkv, const Tensor& stats,
                               const Tensor& out, const Tensor& dout, const Tensor& qkv,
                               int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream) {
    if (out.DType == ETensorDType::FP32) {
        attention_gpu_backward(dqkv.get<float>(), stats.get<float>(), 1.f / sqrtf(HS),
                               out.get<float>(), dout.get<float>(), qkv.get<float>(),
                               B, T, Hq, Hkv, HS, stream);
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
void attention_backward_cudnn(Tensor& dqkv, const Tensor& stats,
                              const Tensor& out, const Tensor& dout, const Tensor& qkv,
                              Tensor& workspace, cudnnHandle_t handle,
                              int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream) {
    std::byte* ws = workspace.get<std::byte>();
    if(out.DType == ETensorDType::FP32) {
        attention_backward_cudnn(dqkv.get<float>(), stats.get<float>(), out.get<float>(), dout.get<float>(), qkv.get<float>(), ws, B, T, Hq, Hkv, HS, stream);
    } else if(out.DType == ETensorDType::BF16) {
        // Argument order is now consistent: out, dout, qkv (matching header declaration)
        attention_backward_cudnn(dqkv.get<nv_bfloat16>(), stats.get<float>(), out.get<nv_bfloat16>(), dout.get<nv_bfloat16>(), qkv.get<nv_bfloat16>(), ws, handle, B, T, Hq, Hkv, HS, stream);
    } else {
        throw std::logic_error("attention_backward: unsupported dtype");
    }
}
