// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file qk_norm.cu
 * @brief Head-wise RMSNorm on packed QKV buffers (Qwen3-style Q/K norm).
 *
 * Implements RMSNorm over vectors of length head_size for each (token, head) inside a
 * packed per-token QKV buffer of shape (B, T, qkv_channels) where each token stores:
 *   [Q (Hq*Hs), K (Hkv*Hs), V (Hkv*Hs)].
 *
 * This is used to apply Qwen3's q_norm/k_norm without changing the packed layout expected
 * by RoPE and cuDNN FlashAttention.
 */

#include <algorithm>
#include <cassert>

#include "kernels.h"
#include "kernel_utils.cuh"
#include "utilities/tensor.h"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

namespace {

constexpr int WARP_SIZE = 32;

template <typename floatX>
__device__ __forceinline__ float to_float(floatX v) {
    return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename floatX>
__device__ __forceinline__ floatX from_float(float v) {
    return static_cast<floatX>(v);
}

template <>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <typename floatX>
__device__ __forceinline__ const floatX* head_ptr(const floatX* qkv, int token_idx, int head_idx,
                                                   int qkv_channels, int head_size, int num_heads,
                                                   int channel_offset) {
    (void)num_heads;
    const int base = token_idx * qkv_channels + channel_offset + head_idx * head_size;
    return qkv + base;
}

template <typename floatX>
__device__ __forceinline__ floatX* head_ptr(floatX* qkv, int token_idx, int head_idx,
                                            int qkv_channels, int head_size, int num_heads,
                                            int channel_offset) {
    (void)num_heads;
    const int base = token_idx * qkv_channels + channel_offset + head_idx * head_size;
    return qkv + base;
}

template<typename floatX>
__global__ void qkv_head_rmsnorm_forward_kernel(floatX* qkv, float* rstd, const floatX* weight,
                                                float epsilon, int tokens, int qkv_channels,
                                                int num_heads, int head_size, int channel_offset) {
    using x128 = GenericVector<floatX, 16 / sizeof(floatX)>;
    static_assert(x128::size > 0);

    assert(blockDim.x == WARP_SIZE);

    // shared layout: [weight][per-warp input cache]
    extern __shared__ char smem_raw[];
    x128* s_weight = reinterpret_cast<x128*>(smem_raw);
    x128* s_in = reinterpret_cast<x128*>(smem_raw) + ((1 + threadIdx.y) * head_size / x128::size);

    // Load weight once per block.
    const int vecs = head_size / x128::size;
    for (int i = (threadIdx.x + WARP_SIZE * threadIdx.y); i < vecs; i += blockDim.y * WARP_SIZE) {
        s_weight[i] = x128::load(weight + i * x128::size);
    }
    __syncthreads();

    const int vec_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int total_vecs = tokens * num_heads;
    if (vec_idx >= total_vecs) return;

    const int token_idx = vec_idx / num_heads;
    const int head_idx = vec_idx - token_idx * num_heads;
    floatX* x = head_ptr(qkv, token_idx, head_idx, qkv_channels, head_size, num_heads, channel_offset);

    float acc = 0.f;
    for (int c = threadIdx.x; c < vecs; c += WARP_SIZE) {
        const x128 in = x128::load_cs(x + c * x128::size);
        s_in[c] = in;
        #pragma unroll
        for (int k = 0; k < x128::size; ++k) {
            float v = to_float(in[k]);
            acc += v * v;
        }
    }

    acc = warpReduceSum(acc) / static_cast<float>(head_size);
    float s = rsqrtf(acc + epsilon);

    for (int c = threadIdx.x; c < vecs; c += WARP_SIZE) {
        const x128 in = s_in[c];
        const x128 w = s_weight[c];
        x128 out;
        #pragma unroll
        for (int k = 0; k < x128::size; ++k) {
            const float n = s * to_float(in[k]);
            out[k] = from_float<floatX>(n * to_float(w[k]));
        }
        out.store(x + c * x128::size);
    }

    if (threadIdx.x == 0) {
        rstd[vec_idx] = s;
    }
}

template<typename floatX>
__global__ void qkv_head_rmsnorm_backward_dx_kernel(floatX* d_qkv, const floatX* qkv_out, const floatX* weight, const float* rstd,
                                                    int tokens, int qkv_channels,
                                                    int num_heads, int head_size, int channel_offset) {
    using x128 = GenericVector<floatX, 16 / sizeof(floatX)>;
    static_assert(x128::size > 0);

    assert(blockDim.x == WARP_SIZE);

    const int vec_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int total_vecs = tokens * num_heads;
    if (vec_idx >= total_vecs) return;

    const int token_idx = vec_idx / num_heads;
    const int head_idx = vec_idx - token_idx * num_heads;

    const floatX* out_ptr = head_ptr(qkv_out, token_idx, head_idx, qkv_channels, head_size, num_heads, channel_offset);
    floatX* dy_ptr = head_ptr(d_qkv, token_idx, head_idx, qkv_channels, head_size, num_heads, channel_offset);

    const int vecs = head_size / x128::size;

    // dot = sum(dy * out) (no division needed).
    float dot = 0.f;
    for (int c = threadIdx.x; c < vecs; c += WARP_SIZE) {
        const x128 out = x128::load_cs(out_ptr + c * x128::size);
        const x128 dy = x128::load_cs(dy_ptr + c * x128::size);
        #pragma unroll
        for (int k = 0; k < x128::size; ++k) {
            dot += to_float(dy[k]) * to_float(out[k]);
        }
    }
    dot = warpReduceSum(dot);

    const float s = rstd[vec_idx];
    const float coeff = s / static_cast<float>(head_size);

    for (int c = threadIdx.x; c < vecs; c += WARP_SIZE) {
        const x128 out = x128::load_cs(out_ptr + c * x128::size);
        const x128 dy = x128::load_cs(dy_ptr + c * x128::size);
        const x128 w = x128::load_cs(weight + c * x128::size);
        x128 dx;
        #pragma unroll
        for (int k = 0; k < x128::size; ++k) {
            const float wf = to_float(w[k]);
            const float dyf = to_float(dy[k]);
            const float out_f = to_float(out[k]);
            const float xhat = (wf != 0.f) ? (out_f / wf) : 0.f;
            const float wdy = dyf * wf;
            const float dx_f = wdy * s - xhat * coeff * dot;
            dx[k] = from_float<floatX>(dx_f);
        }
        dx.store(dy_ptr + c * x128::size);
    }
}

template<typename floatX>
__global__ void qkv_head_rmsnorm_backward_dweight_kernel(floatX* d_weight, const floatX* d_qkv, const floatX* qkv_out, const floatX* weight,
                                                         int tokens, int qkv_channels,
                                                         int num_heads, int head_size, int channel_offset,
                                                         bool accumulate) {
    const int c = static_cast<int>(blockIdx.x);
    if (c >= head_size) return;

    const float wf = to_float(weight[c]);
    const float inv_w = (wf != 0.f) ? (1.0f / wf) : 0.f;

    const int total_vecs = tokens * num_heads;
    float sum = 0.f;
    for (int vec_idx = threadIdx.x; vec_idx < total_vecs; vec_idx += blockDim.x) {
        const int token_idx = vec_idx / num_heads;
        const int head_idx = vec_idx - token_idx * num_heads;
        const int base = token_idx * qkv_channels + channel_offset + head_idx * head_size + c;
        const float out_f = to_float(qkv_out[base]);
        const float dy_f = to_float(d_qkv[base]);
        sum += dy_f * out_f;
    }

    // Reduce within block.
    __shared__ float smem[256];
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float grad = smem[0] * inv_w;  // dy * (out / w)
        const float prev = accumulate ? to_float(d_weight[c]) : 0.f;
        d_weight[c] = from_float<floatX>(prev + grad);
    }
}

template<typename floatX>
void launch_forward(floatX* qkv, float* rstd, const floatX* weight,
                    float epsilon, int tokens, int qkv_channels,
                    int num_heads, int head_size, int channel_offset,
                    cudaStream_t stream) {
    const int vecs = tokens * num_heads;
    constexpr int block_y = 4;
    dim3 block(WARP_SIZE, block_y);
    dim3 grid(div_ceil(vecs, block_y));

    using x128 = GenericVector<floatX, 16 / sizeof(floatX)>;
    if (head_size % x128::size != 0) {
        throw std::runtime_error("qkv_head_rmsnorm_forward: head_size must be a multiple of 16B vector width");
    }
    const std::size_t shmem = sizeof(floatX) * static_cast<std::size_t>(head_size) * static_cast<std::size_t>(1 + block_y);
    CUDA_CHECK(cudaFuncSetAttribute(qkv_head_rmsnorm_forward_kernel<floatX>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));
    qkv_head_rmsnorm_forward_kernel<floatX><<<grid, block, shmem, stream>>>(
        qkv, rstd, weight, epsilon, tokens, qkv_channels, num_heads, head_size, channel_offset);
    CUDA_CHECK(cudaGetLastError());
}

template<typename floatX>
void launch_backward_dx(floatX* d_qkv, const floatX* qkv_out, const floatX* weight, const float* rstd,
                        int tokens, int qkv_channels,
                        int num_heads, int head_size, int channel_offset,
                        cudaStream_t stream) {
    const int vecs = tokens * num_heads;
    constexpr int block_y = 4;
    dim3 block(WARP_SIZE, block_y);
    dim3 grid(div_ceil(vecs, block_y));

    using x128 = GenericVector<floatX, 16 / sizeof(floatX)>;
    if (head_size % x128::size != 0) {
        throw std::runtime_error("qkv_head_rmsnorm_backward_dx: head_size must be a multiple of 16B vector width");
    }

    qkv_head_rmsnorm_backward_dx_kernel<floatX><<<grid, block, 0, stream>>>(
        d_qkv, qkv_out, weight, rstd, tokens, qkv_channels, num_heads, head_size, channel_offset);
    CUDA_CHECK(cudaGetLastError());
}

template<typename floatX>
void launch_backward_dweight(floatX* d_weight, const floatX* d_qkv, const floatX* qkv_out, const floatX* weight,
                             int tokens, int qkv_channels,
                             int num_heads, int head_size, int channel_offset,
                             bool accumulate,
                             cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(head_size);
    qkv_head_rmsnorm_backward_dweight_kernel<floatX><<<grid, block, 0, stream>>>(
        d_weight, d_qkv, qkv_out, weight, tokens, qkv_channels, num_heads, head_size, channel_offset, accumulate);
    CUDA_CHECK(cudaGetLastError());
}

template <typename floatX>
__device__ __forceinline__ void load_cos_sin(const floatX* freqs_cis, int pos, int head_size, int pair_idx, float& c, float& s) {
    // freqs_cis stores interleaved cos/sin pairs: [cos0, sin0, cos1, sin1, ...]
    // indexed by pair_idx in [0, head_size/2).
    const int base = pos * head_size + 2 * pair_idx;
    c = to_float(freqs_cis[base + 0]);
    s = to_float(freqs_cis[base + 1]);
}

template<typename floatX>
__global__ void qkv_qk_norm_rope_forward_kernel(
    floatX* qkv,
    float* q_rstd, float* k_rstd,
    const floatX* q_weight, const floatX* k_weight,
    const floatX* freqs_cis, const int* position_ids,
    float epsilon,
    int tokens, int T,
    int Hq, int Hkv,
    int head_size, int qkv_channels
) {
    // One warp per (token, head) vector. Supports head_size/2 multiple of 32.
    constexpr int WARP = 32;
    static_assert(WARP_SIZE == WARP);
    const int vecs_per_block = static_cast<int>(blockDim.y);
    const int vec_idx = static_cast<int>(blockIdx.x) * vecs_per_block + static_cast<int>(threadIdx.y);
    const int total_vecs = tokens * (Hq + Hkv);
    if (vec_idx >= total_vecs) return;

    const int token_idx = vec_idx / (Hq + Hkv);
    const int head_qk = vec_idx - token_idx * (Hq + Hkv);
    const bool is_q = head_qk < Hq;
    const int head_idx = is_q ? head_qk : (head_qk - Hq);
    const int q_rows = Hq * head_size;
    const int channel_offset = is_q ? 0 : q_rows;
    floatX* x = qkv + token_idx * qkv_channels + channel_offset + head_idx * head_size;
    const floatX* w = is_q ? q_weight : k_weight;

    const int b = token_idx / T;
    const int t = token_idx - b * T;
    const int pos = position_ids ? position_ids[token_idx] : t;

    const int half = head_size / 2;
    if ((half % WARP) != 0) return; // unsupported shape

    // Compute mean square over head_size.
    float acc = 0.0f;
    for (int i = threadIdx.x; i < half; i += WARP) {
        const float v0 = to_float(x[i]);
        const float v1 = to_float(x[i + half]);
        acc += v0 * v0 + v1 * v1;
    }
    acc = warpReduceSum(acc) / static_cast<float>(head_size);
    const float s = rsqrtf(acc + epsilon);

    if (threadIdx.x == 0) {
        float* rstd = is_q ? q_rstd : k_rstd;
        const int stride = is_q ? Hq : Hkv;
        rstd[token_idx * stride + head_idx] = s;
    }

    // Apply RMSNorm(weight) then RoPE in one pass; write post-RoPE outputs back in-place.
    for (int i = threadIdx.x; i < half; i += WARP) {
        const float v0 = to_float(x[i]);
        const float v1 = to_float(x[i + half]);

        const float w0 = to_float(w[i]);
        const float w1 = to_float(w[i + half]);

        const float y0 = (v0 * s) * w0;
        const float y1 = (v1 * s) * w1;

        float c, sin;
        load_cos_sin(freqs_cis, pos, head_size, i, c, sin);

        const float o0 = y0 * c - y1 * sin;
        const float o1 = y1 * c + y0 * sin;

        x[i] = from_float<floatX>(o0);
        x[i + half] = from_float<floatX>(o1);
    }
}

template<typename floatX>
__global__ void qkv_head_rmsnorm_rope_backward_dx_kernel(
    floatX* d_qkv,
    const floatX* qkv_rope,
    const floatX* weight,
    const float* rstd,
    const floatX* freqs_cis,
    const int* position_ids,
    int tokens, int T, int qkv_channels,
    int num_heads, int head_size, int channel_offset,
    float* abs_max_ptr
) {
    constexpr int WARP = 32;
    static_assert(WARP_SIZE == WARP);
    const int vecs_per_block = static_cast<int>(blockDim.y);
    const int vec_idx = static_cast<int>(blockIdx.x) * vecs_per_block + static_cast<int>(threadIdx.y);
    const int total_vecs = tokens * num_heads;
    const bool record_abs = abs_max_ptr != nullptr;
    const bool active = vec_idx < total_vecs;
    if (!record_abs) {
        if (!active) return;
    }

    const int token_idx = active ? (vec_idx / num_heads) : 0;
    const int head_idx = active ? (vec_idx - token_idx * num_heads) : 0;
    const int b = active ? (token_idx / T) : 0;
    const int t = active ? (token_idx - b * T) : 0;
    const int pos = active ? (position_ids ? position_ids[token_idx] : t) : 0;

    const floatX* outp = (!active)
        ? nullptr
        : qkv_rope + token_idx * qkv_channels + channel_offset + head_idx * head_size;
    floatX* dptr = (!active)
        ? nullptr
        : d_qkv + token_idx * qkv_channels + channel_offset + head_idx * head_size;

    const int half = head_size / 2;
    if ((half % WARP) != 0) return;

    // dot = sum(dy_pre * out_pre) over head_size.
    float dot = 0.0f;
    if (!active) {
        dot = 0.0f;
    } else {
        for (int i = threadIdx.x; i < half; i += WARP) {
            const float o0 = to_float(outp[i]);
            const float o1 = to_float(outp[i + half]);
            const float dy0 = to_float(dptr[i]);
            const float dy1 = to_float(dptr[i + half]);

            float c, sin;
            load_cos_sin(freqs_cis, pos, head_size, i, c, sin);

            // Inverse RoPE: a0 = b0*c + b1*s; a1 = b1*c - b0*s
            const float out0 = o0 * c + o1 * sin;
            const float out1 = o1 * c - o0 * sin;
            const float d0 = dy0 * c + dy1 * sin;
            const float d1 = dy1 * c - dy0 * sin;

            dot += d0 * out0 + d1 * out1;
        }
    }
    dot = warpReduceSum(dot);

    float s = 0.0f;
    if (!active) {
        s = 0.0f;
    } else {
        s = rstd[token_idx * num_heads + head_idx];
    }
    const float coeff = s / static_cast<float>(head_size);

    // Transform dy_post -> dx_pre in-place for Q/K channels.
    float thread_abs_max = 0.f;
    if (!active) {
        thread_abs_max = 0.f;
    } else {
        for (int i = threadIdx.x; i < half; i += WARP) {
            const float o0 = to_float(outp[i]);
            const float o1 = to_float(outp[i + half]);
            const float dy0 = to_float(dptr[i]);
            const float dy1 = to_float(dptr[i + half]);

            float c, sin;
            load_cos_sin(freqs_cis, pos, head_size, i, c, sin);

            const float out0 = o0 * c + o1 * sin;
            const float out1 = o1 * c - o0 * sin;
            const float d0 = dy0 * c + dy1 * sin;
            const float d1 = dy1 * c - dy0 * sin;

            const float w0 = to_float(weight[i]);
            const float w1 = to_float(weight[i + half]);
            const float inv_w0 = (w0 != 0.f) ? (1.0f / w0) : 0.0f;
            const float inv_w1 = (w1 != 0.f) ? (1.0f / w1) : 0.0f;

            const float xhat0 = out0 * inv_w0;
            const float xhat1 = out1 * inv_w1;
            const float wdy0 = d0 * w0;
            const float wdy1 = d1 * w1;

            const float dx0 = wdy0 * s - xhat0 * coeff * dot;
            const float dx1 = wdy1 * s - xhat1 * coeff * dot;

            dptr[i] = from_float<floatX>(dx0);
            dptr[i + half] = from_float<floatX>(dx1);
            if (record_abs) {
                thread_abs_max = fmaxf(thread_abs_max, fabsf(dx0));
                thread_abs_max = fmaxf(thread_abs_max, fabsf(dx1));
            }
        }
    }

    if (record_abs) {
        __shared__ float block_abs_max;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            block_abs_max = 0.f;
        }
        __syncthreads();
        handle_absmax_reduction(abs_max_ptr, &block_abs_max, thread_abs_max);
    }
}

template<typename floatX>
__global__ void qkv_head_rmsnorm_rope_backward_dweight_kernel(
    floatX* d_weight,
    const floatX* d_qkv,
    const floatX* qkv_rope,
    const floatX* weight,
    const floatX* freqs_cis,
    const int* position_ids,
    int tokens, int T, int qkv_channels,
    int num_heads, int head_size, int channel_offset,
    bool accumulate
) {
    const int c = static_cast<int>(blockIdx.x);
    if (c >= head_size) return;

    const float wf = to_float(weight[c]);
    const float inv_w = (wf != 0.f) ? (1.0f / wf) : 0.0f;

    const int half = head_size / 2;
    float sum = 0.f;

    const int total_vecs = tokens * num_heads;
    for (int vec_idx = static_cast<int>(threadIdx.x); vec_idx < total_vecs; vec_idx += static_cast<int>(blockDim.x)) {
        const int token_idx = vec_idx / num_heads;
        const int head_idx = vec_idx - token_idx * num_heads;
        const int b = token_idx / T;
        const int t = token_idx - b * T;
        const int pos = position_ids ? position_ids[token_idx] : t;

        const floatX* outp = qkv_rope + token_idx * qkv_channels + channel_offset + head_idx * head_size;
        const floatX* dptr = d_qkv + token_idx * qkv_channels + channel_offset + head_idx * head_size;

        float out_c, dy_c;
        if (c < half) {
            const float o0 = to_float(outp[c]);
            const float o1 = to_float(outp[c + half]);
            const float dy0 = to_float(dptr[c]);
            const float dy1 = to_float(dptr[c + half]);

            float cosv, sinv;
            load_cos_sin(freqs_cis, pos, head_size, c, cosv, sinv);
            out_c = o0 * cosv + o1 * sinv;
            dy_c = dy0 * cosv + dy1 * sinv;
        } else {
            const int p = c - half;
            const float o0 = to_float(outp[p]);
            const float o1 = to_float(outp[p + half]);
            const float dy0 = to_float(dptr[p]);
            const float dy1 = to_float(dptr[p + half]);

            float cosv, sinv;
            load_cos_sin(freqs_cis, pos, head_size, p, cosv, sinv);
            out_c = o1 * cosv - o0 * sinv;
            dy_c = dy1 * cosv - dy0 * sinv;
        }

        sum += dy_c * out_c;
    }

    __shared__ float smem[256];
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride >= 1; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float grad = smem[0] * inv_w;
        const float prev = accumulate ? to_float(d_weight[c]) : 0.f;
        d_weight[c] = from_float<floatX>(prev + grad);
    }
}

template<typename floatX>
__global__ void qkv_abs_max_slice_kernel(
    const floatX* qkv,
    int tokens, int qkv_channels,
    int channel_offset, int channel_count,
    float* abs_max_ptr
) {
    float thread_max = 0.f;
    const int total = tokens * channel_count;
    for (int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
         idx < total;
         idx += static_cast<int>(blockDim.x) * static_cast<int>(gridDim.x)) {
        const int token_idx = idx / channel_count;
        const int channel_idx = idx - token_idx * channel_count;
        const int offset = token_idx * qkv_channels + channel_offset + channel_idx;
        const float v = to_float(qkv[offset]);
        thread_max = fmaxf(thread_max, fabsf(v));
    }

    __shared__ float block_abs_max;
    if (threadIdx.x == 0) {
        block_abs_max = 0.f;
    }
    __syncthreads();
    handle_absmax_reduction(abs_max_ptr, &block_abs_max, thread_max);
}

template<typename floatX>
void launch_qk_norm_rope_forward(floatX* qkv, float* q_rstd, float* k_rstd,
                                 const floatX* q_weight, const floatX* k_weight,
                                 const floatX* freqs_cis, const int* position_ids,
                                 float epsilon, int tokens, int T, int Hq, int Hkv, int head_size, int qkv_channels,
                                 cudaStream_t stream) {
    constexpr int block_y = 4;
    dim3 block(WARP_SIZE, block_y);
    dim3 grid(div_ceil(tokens * (Hq + Hkv), block_y));
    qkv_qk_norm_rope_forward_kernel<floatX><<<grid, block, 0, stream>>>(
        qkv, q_rstd, k_rstd, q_weight, k_weight, freqs_cis, position_ids, epsilon,
        tokens, T, Hq, Hkv, head_size, qkv_channels);
    CUDA_CHECK(cudaGetLastError());
}

template<typename floatX>
void launch_rmsnorm_rope_backward_dx(floatX* d_qkv, const floatX* qkv_rope, const floatX* weight, const float* rstd,
                                    const floatX* freqs_cis, const int* position_ids,
                                    int tokens, int T, int qkv_channels,
                                    int num_heads, int head_size, int channel_offset,
                                    cudaStream_t stream, float* abs_max_ptr) {
    constexpr int block_y = 4;
    dim3 block(WARP_SIZE, block_y);
    dim3 grid(div_ceil(tokens * num_heads, block_y));
    qkv_head_rmsnorm_rope_backward_dx_kernel<floatX><<<grid, block, 0, stream>>>(
        d_qkv, qkv_rope, weight, rstd, freqs_cis, position_ids,
        tokens, T, qkv_channels, num_heads, head_size, channel_offset, abs_max_ptr);
    CUDA_CHECK(cudaGetLastError());
}

template<typename floatX>
void launch_rmsnorm_rope_backward_dweight(floatX* d_weight, const floatX* d_qkv, const floatX* qkv_rope, const floatX* weight,
                                         const floatX* freqs_cis, const int* position_ids,
                                         int tokens, int T, int qkv_channels,
                                         int num_heads, int head_size, int channel_offset,
                                         bool accumulate,
                                         cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(head_size);
    qkv_head_rmsnorm_rope_backward_dweight_kernel<floatX><<<grid, block, 0, stream>>>(
        d_weight, d_qkv, qkv_rope, weight, freqs_cis, position_ids,
        tokens, T, qkv_channels, num_heads, head_size, channel_offset, accumulate);
    CUDA_CHECK(cudaGetLastError());
}

template<typename floatX>
void launch_qkv_abs_max_slice(const floatX* qkv, int tokens, int qkv_channels,
                              int channel_offset, int channel_count,
                              float* abs_max_ptr, cudaStream_t stream) {
    if (!abs_max_ptr) return;
    constexpr int block = 256;
    const int total = tokens * channel_count;
    if (total <= 0) return;
    const int grid = std::min(1024, div_ceil(total, block));
    qkv_abs_max_slice_kernel<floatX><<<grid, block, 0, stream>>>(
        qkv, tokens, qkv_channels, channel_offset, channel_count, abs_max_ptr);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void qkv_head_rmsnorm_forward(Tensor& qkv, Tensor& rstd, const Tensor& weight,
                              float epsilon, int B, int T, int qkv_channels,
                              int num_heads, int head_size, int channel_offset,
                              cudaStream_t stream) {
    if (rstd.DType != ETensorDType::FP32) {
        throw std::logic_error("qkv_head_rmsnorm_forward: rstd must be FP32");
    }
    if (qkv.DType != weight.DType) {
        fprintf(stderr, "[qk_norm] qkv/weight dtype mismatch: received=%d, wanted=%d\n",
                (int)qkv.DType, (int)weight.DType);
        throw std::logic_error("qkv_head_rmsnorm_forward: qkv/weight dtype mismatch");
    }
    const int tokens = B * T;
    if (qkv.Rank != 3 || qkv.Sizes[0] != B || qkv.Sizes[1] != T || qkv.Sizes[2] != qkv_channels) {
        fprintf(stderr, "[qk_norm] qkv shape mismatch: qkv.Sizes=[%ld,%ld,%ld], expected=[%d,%d,%d]\n",
                qkv.Sizes[0], qkv.Sizes[1], qkv.Sizes[2], B, T, qkv_channels);
        throw std::logic_error("qkv_head_rmsnorm_forward: unexpected qkv shape");
    }
    if (weight.Rank != 1 || weight.Sizes[0] != head_size) {
        fprintf(stderr, "[qk_norm] weight shape mismatch: weight.Rank=%d (expected 1), weight.Sizes[0]=%ld (expected %d)\n",
                weight.Rank, weight.Sizes[0], head_size);
        throw std::logic_error("qkv_head_rmsnorm_forward: unexpected weight shape");
    }
    if (rstd.Rank != 3 || rstd.Sizes[0] != B || rstd.Sizes[1] != T || rstd.Sizes[2] != num_heads) {
        fprintf(stderr, "[qk_norm] rstd shape mismatch: rstd.Sizes=[%ld,%ld,%ld], expected=[%d,%d,%d]\n",
                rstd.Sizes[0], rstd.Sizes[1], rstd.Sizes[2], B, T, num_heads);
        throw std::logic_error("qkv_head_rmsnorm_forward: unexpected rstd shape");
    }

    if (qkv.DType == ETensorDType::BF16) {
        launch_forward(qkv.get<nv_bfloat16>(), rstd.get<float>(), weight.get<nv_bfloat16>(),
                       epsilon, tokens, qkv_channels, num_heads, head_size, channel_offset, stream);
    } else if (qkv.DType == ETensorDType::FP32) {
        launch_forward(qkv.get<float>(), rstd.get<float>(), weight.get<float>(),
                       epsilon, tokens, qkv_channels, num_heads, head_size, channel_offset, stream);
    } else {
        throw std::logic_error("qkv_head_rmsnorm_forward: unsupported dtype");
    }
}

void qkv_head_rmsnorm_backward_dx(Tensor& d_qkv, const Tensor& qkv_out, const Tensor& weight, const Tensor& rstd,
                                  int B, int T, int qkv_channels,
                                  int num_heads, int head_size, int channel_offset,
                                  cudaStream_t stream) {
    if (rstd.DType != ETensorDType::FP32) {
        throw std::logic_error("qkv_head_rmsnorm_backward_dx: rstd must be FP32");
    }
    if (d_qkv.DType != qkv_out.DType || d_qkv.DType != weight.DType) {
        throw std::logic_error("qkv_head_rmsnorm_backward_dx: dtype mismatch");
    }
    const int tokens = B * T;

    if (d_qkv.DType == ETensorDType::BF16) {
        launch_backward_dx(d_qkv.get<nv_bfloat16>(), qkv_out.get<nv_bfloat16>(), weight.get<nv_bfloat16>(), rstd.get<float>(),
                           tokens, qkv_channels, num_heads, head_size, channel_offset, stream);
    } else if (d_qkv.DType == ETensorDType::FP32) {
        launch_backward_dx(d_qkv.get<float>(), qkv_out.get<float>(), weight.get<float>(), rstd.get<float>(),
                           tokens, qkv_channels, num_heads, head_size, channel_offset, stream);
    } else {
        throw std::logic_error("qkv_head_rmsnorm_backward_dx: unsupported dtype");
    }
}

void qkv_head_rmsnorm_backward_dweight(Tensor& d_weight, const Tensor& d_qkv, const Tensor& qkv_out, const Tensor& weight,
                                       int B, int T, int qkv_channels,
                                       int num_heads, int head_size, int channel_offset,
                                       bool accumulate, cudaStream_t stream) {
    if (d_weight.DType != d_qkv.DType || d_weight.DType != qkv_out.DType || d_weight.DType != weight.DType) {
        throw std::logic_error("qkv_head_rmsnorm_backward_dweight: dtype mismatch");
    }
    if (d_weight.Rank != 1 || d_weight.Sizes[0] != head_size) {
        throw std::logic_error("qkv_head_rmsnorm_backward_dweight: unexpected d_weight shape");
    }
    const int tokens = B * T;

    if (d_weight.DType == ETensorDType::BF16) {
        launch_backward_dweight(d_weight.get<nv_bfloat16>(), d_qkv.get<nv_bfloat16>(), qkv_out.get<nv_bfloat16>(), weight.get<nv_bfloat16>(),
                                tokens, qkv_channels, num_heads, head_size, channel_offset, accumulate, stream);
    } else if (d_weight.DType == ETensorDType::FP32) {
        launch_backward_dweight(d_weight.get<float>(), d_qkv.get<float>(), qkv_out.get<float>(), weight.get<float>(),
                                tokens, qkv_channels, num_heads, head_size, channel_offset, accumulate, stream);
    } else {
        throw std::logic_error("qkv_head_rmsnorm_backward_dweight: unsupported dtype");
    }
}

void qkv_qk_norm_rope_forward(Tensor& qkv,
                              Tensor& q_rstd, Tensor& k_rstd,
                              const Tensor& q_weight, const Tensor& k_weight,
                              const Tensor& freqs_cis, const int* position_ids,
                              float epsilon, int B, int T, int Hq, int Hkv, int head_size,
                              cudaStream_t stream) {
    if (qkv.DType != q_weight.DType || qkv.DType != k_weight.DType || qkv.DType != freqs_cis.DType) {
        throw std::logic_error("qkv_qk_norm_rope_forward: dtype mismatch");
    }
    if (q_rstd.DType != ETensorDType::FP32 || k_rstd.DType != ETensorDType::FP32) {
        throw std::logic_error("qkv_qk_norm_rope_forward: rstd must be FP32");
    }
    const int tokens = B * T;
    const int qkv_channels = (Hq + 2 * Hkv) * head_size;
    if (qkv.Rank != 3 || qkv.Sizes[0] != B || qkv.Sizes[1] != T || qkv.Sizes[2] != qkv_channels) {
        throw std::logic_error("qkv_qk_norm_rope_forward: unexpected qkv shape");
    }
    if (q_weight.Rank != 1 || q_weight.Sizes[0] != head_size || k_weight.Rank != 1 || k_weight.Sizes[0] != head_size) {
        throw std::logic_error("qkv_qk_norm_rope_forward: unexpected weight shape");
    }
    if (q_rstd.Rank != 3 || q_rstd.Sizes[0] != B || q_rstd.Sizes[1] != T || q_rstd.Sizes[2] != Hq) {
        throw std::logic_error("qkv_qk_norm_rope_forward: unexpected q_rstd shape");
    }
    if (k_rstd.Rank != 3 || k_rstd.Sizes[0] != B || k_rstd.Sizes[1] != T || k_rstd.Sizes[2] != Hkv) {
        throw std::logic_error("qkv_qk_norm_rope_forward: unexpected k_rstd shape");
    }
    if ((head_size % 2) != 0) {
        throw std::logic_error("qkv_qk_norm_rope_forward: head_size must be even");
    }
    if (((head_size / 2) % WARP_SIZE) != 0) {
        throw std::logic_error("qkv_qk_norm_rope_forward: head_size/2 must be a multiple of 32");
    }
    // freqs_cis is expected to be at least (max_pos, head_size) with interleaved cos/sin pairs.
    if (freqs_cis.Rank < 2 || freqs_cis.Sizes[1] < head_size) {
        throw std::logic_error("qkv_qk_norm_rope_forward: unexpected freqs_cis shape");
    }

    if (qkv.DType == ETensorDType::BF16) {
        launch_qk_norm_rope_forward(qkv.get<nv_bfloat16>(), q_rstd.get<float>(), k_rstd.get<float>(),
                                    q_weight.get<nv_bfloat16>(), k_weight.get<nv_bfloat16>(),
                                    freqs_cis.get<nv_bfloat16>(), position_ids,
                                    epsilon, tokens, T, Hq, Hkv, head_size, qkv_channels, stream);
    } else if (qkv.DType == ETensorDType::FP32) {
        launch_qk_norm_rope_forward(qkv.get<float>(), q_rstd.get<float>(), k_rstd.get<float>(),
                                    q_weight.get<float>(), k_weight.get<float>(),
                                    freqs_cis.get<float>(), position_ids,
                                    epsilon, tokens, T, Hq, Hkv, head_size, qkv_channels, stream);
    } else {
        throw std::logic_error("qkv_qk_norm_rope_forward: unsupported dtype");
    }
}

void qkv_head_rmsnorm_rope_backward_dx(Tensor& d_qkv, const Tensor& qkv_rope, const Tensor& weight, const Tensor& rstd,
                                       const Tensor& freqs_cis, const int* position_ids,
                                       int B, int T, int qkv_channels,
                                       int num_heads, int head_size, int channel_offset,
                                       cudaStream_t stream, float* abs_max_ptr) {
    if (rstd.DType != ETensorDType::FP32) {
        throw std::logic_error("qkv_head_rmsnorm_rope_backward_dx: rstd must be FP32");
    }
    if (d_qkv.DType != qkv_rope.DType || d_qkv.DType != weight.DType || d_qkv.DType != freqs_cis.DType) {
        throw std::logic_error("qkv_head_rmsnorm_rope_backward_dx: dtype mismatch");
    }
    const int tokens = B * T;
    if ((head_size % 2) != 0) {
        throw std::logic_error("qkv_head_rmsnorm_rope_backward_dx: head_size must be even");
    }
    if (((head_size / 2) % WARP_SIZE) != 0) {
        throw std::logic_error("qkv_head_rmsnorm_rope_backward_dx: head_size/2 must be a multiple of 32");
    }

    if (d_qkv.DType == ETensorDType::BF16) {
        launch_rmsnorm_rope_backward_dx(d_qkv.get<nv_bfloat16>(), qkv_rope.get<nv_bfloat16>(), weight.get<nv_bfloat16>(), rstd.get<float>(),
                                        freqs_cis.get<nv_bfloat16>(), position_ids,
                                        tokens, T, qkv_channels, num_heads, head_size, channel_offset, stream, abs_max_ptr);
    } else if (d_qkv.DType == ETensorDType::FP32) {
        launch_rmsnorm_rope_backward_dx(d_qkv.get<float>(), qkv_rope.get<float>(), weight.get<float>(), rstd.get<float>(),
                                        freqs_cis.get<float>(), position_ids,
                                        tokens, T, qkv_channels, num_heads, head_size, channel_offset, stream, abs_max_ptr);
    } else {
        throw std::logic_error("qkv_head_rmsnorm_rope_backward_dx: unsupported dtype");
    }
}

void qkv_head_rmsnorm_rope_backward_dweight(Tensor& d_weight, const Tensor& d_qkv, const Tensor& qkv_rope, const Tensor& weight,
                                            const Tensor& freqs_cis, const int* position_ids,
                                            int B, int T, int qkv_channels,
                                            int num_heads, int head_size, int channel_offset,
                                            bool accumulate, cudaStream_t stream) {
    if (d_weight.DType != d_qkv.DType || d_weight.DType != qkv_rope.DType || d_weight.DType != weight.DType || d_weight.DType != freqs_cis.DType) {
        throw std::logic_error("qkv_head_rmsnorm_rope_backward_dweight: dtype mismatch");
    }
    if (d_weight.Rank != 1 || d_weight.Sizes[0] != head_size) {
        throw std::logic_error("qkv_head_rmsnorm_rope_backward_dweight: unexpected d_weight shape");
    }
    const int tokens = B * T;
    if ((head_size % 2) != 0) {
        throw std::logic_error("qkv_head_rmsnorm_rope_backward_dweight: head_size must be even");
    }
    if (((head_size / 2) % WARP_SIZE) != 0) {
        throw std::logic_error("qkv_head_rmsnorm_rope_backward_dweight: head_size/2 must be a multiple of 32");
    }

    if (d_weight.DType == ETensorDType::BF16) {
        launch_rmsnorm_rope_backward_dweight(d_weight.get<nv_bfloat16>(), d_qkv.get<nv_bfloat16>(), qkv_rope.get<nv_bfloat16>(), weight.get<nv_bfloat16>(),
                                             freqs_cis.get<nv_bfloat16>(), position_ids,
                                             tokens, T, qkv_channels, num_heads, head_size, channel_offset, accumulate, stream);
    } else if (d_weight.DType == ETensorDType::FP32) {
        launch_rmsnorm_rope_backward_dweight(d_weight.get<float>(), d_qkv.get<float>(), qkv_rope.get<float>(), weight.get<float>(),
                                             freqs_cis.get<float>(), position_ids,
                                             tokens, T, qkv_channels, num_heads, head_size, channel_offset, accumulate, stream);
    } else {
        throw std::logic_error("qkv_head_rmsnorm_rope_backward_dweight: unsupported dtype");
    }
}

void qkv_abs_max_slice(const Tensor& qkv, int B, int T, int qkv_channels,
                       int channel_offset, int channel_count,
                       float* abs_max_ptr, cudaStream_t stream) {
    if (!abs_max_ptr) return;
    if (qkv.DType != ETensorDType::BF16 && qkv.DType != ETensorDType::FP32) {
        throw std::logic_error("qkv_abs_max_slice: unsupported dtype");
    }
    const int tokens = B * T;
    if (qkv.Rank != 3 || qkv.Sizes[0] != B || qkv.Sizes[1] != T || qkv.Sizes[2] != qkv_channels) {
        throw std::logic_error("qkv_abs_max_slice: unexpected qkv shape");
    }
    if (channel_offset < 0 || channel_count < 0 || channel_offset + channel_count > qkv_channels) {
        throw std::logic_error("qkv_abs_max_slice: invalid channel slice");
    }

    if (qkv.DType == ETensorDType::BF16) {
        launch_qkv_abs_max_slice(qkv.get<nv_bfloat16>(), tokens, qkv_channels,
                                 channel_offset, channel_count, abs_max_ptr, stream);
    } else {
        launch_qkv_abs_max_slice(qkv.get<float>(), tokens, qkv_channels,
                                 channel_offset, channel_count, abs_max_ptr, stream);
    }
}
