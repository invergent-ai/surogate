// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Hashed n-gram memory kernels (see ngram_memory.h). The rows and the IQ4_NL gather follow the
// engine's rows_kernel / gather_kernel (csrc/src/serve/ops/ngram_ple/ngram_ple.cu) operation for
// operation; the gate is the engine's gate_kernel algebra for one stream.

#include "kernels/ngram_memory.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "kernels/kernel_utils.cuh"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

namespace ngram_memory {
namespace {

constexpr int kThreads = 256;
constexpr int WARP_SIZE = 32;  // as kernels/rmsnorm.cu
constexpr float kGateClamp = 1e-6F;

__constant__ std::int8_t kIq4nlValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                             1,    13,   25,  38,  53,  69,  89,  113};

__device__ __forceinline__ float ld(const float* p, long i) { return p[i]; }
__device__ __forceinline__ float ld(const nv_bfloat16* p, long i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ void st(float* p, long i, float v) { p[i] = v; }
__device__ __forceinline__ void st(nv_bfloat16* p, long i, float v) { p[i] = __float2bfloat16_rn(v); }

__device__ __forceinline__ float sigmoid(float x) { return 1.0F / (1.0F + expf(-x)); }

__device__ __forceinline__ bool is_zero_token(int id, const ZeroTokens& zero) {
    for (int i = 0; i < zero.count; ++i) {
        if (id == zero.ids[i]) return true;
    }
    return false;
}

// Sum of two values over a block of kThreads threads; every thread gets the totals.
__device__ __forceinline__ void block_sum2(float& a, float& b, float* scratch /* 2 * kThreads/32 */) {
    a = warpReduceSum(a);
    b = warpReduceSum(b);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    __syncthreads();
    if (lane == 0) {
        scratch[warp] = a;
        scratch[kThreads / 32 + warp] = b;
    }
    __syncthreads();
    a = 0.0F;
    b = 0.0F;
    for (int w = 0; w < kThreads / 32; ++w) {
        a += scratch[w];
        b += scratch[kThreads / 32 + w];
    }
}

// ------------------------------------------------------------------------------------------------
// Rows: one thread per token computes all heads (the engine: one block per token, a thread per head;
// the arithmetic is the same).
__global__ void rows_kernel(int* __restrict__ rows, const int* __restrict__ ids, const int* __restrict__ positions,
                            const int* __restrict__ carry, int B, int T, Hash hash) {
    const long t = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<long>(B) * T) return;
    const int b = static_cast<int>(t / T);
    const int j = static_cast<int>(t - static_cast<long>(b) * T);
    const int position = positions[t];
    std::int64_t ctx[kMaxNgram];
    ctx[0] = ids[t];
    bool cut = false;
    for (int k = 1; k < hash.ngram; ++k) {
        int token = -1;
        if (!cut && position >= k) {
            if (j - k >= 0) {
                token = ids[t - k];
            } else if (carry != nullptr) {
                const int slot = 2 + (j - k);  // carry holds the chunk's two predecessors, oldest first
                token = slot >= 0 ? carry[b * 2 + slot] : -1;
            }
        }
        cut = cut || token < 0 || token == hash.eos_token;
        ctx[k] = cut ? hash.eos_token : token;
    }
    const int per_gram = hash.heads / (hash.ngram - 1);
    for (int h = 0; h < hash.heads; ++h) {
        const int order = 2 + h / per_gram;
        std::uint64_t mixed = static_cast<std::uint64_t>(ctx[0]) * hash.multipliers[0];
        for (int q = 1; q < order; ++q) {
            mixed ^= static_cast<std::uint64_t>(ctx[q]) * hash.multipliers[q];
        }
        rows[t * hash.heads + h] =
            static_cast<int>(mixed % static_cast<std::uint64_t>(hash.head_vocab_sizes[h]) + hash.head_offsets[h]);
    }
}

// ------------------------------------------------------------------------------------------------
// Gather: the engine's gather_kernel.
__global__ void gather_kernel(nv_bfloat16* __restrict__ embedding, const int* __restrict__ rows,
                              const std::uint8_t* __restrict__ table, int row_bytes, int head_dim, int heads,
                              RowShuffle shuffle) {
    const long t = blockIdx.x;
    const int values = heads * head_dim;
    for (int v = static_cast<int>(threadIdx.x); v < values; v += kThreads) {
        const int head = v / head_dim;
        const int within = v - head * head_dim;
        const int block = within / 32;
        const int lane = within - block * 32;
        std::int64_t row = rows[t * heads + head];
        if (shuffle.a != 0) {  // (r * a + b) mod R; r < 2^31 and a, b < 2^32 keep it inside uint64
            row = static_cast<std::int64_t>((static_cast<std::uint64_t>(row) * shuffle.a + shuffle.b) %
                                            static_cast<std::uint64_t>(shuffle.row_count));
        }
        const std::uint8_t* data = table + row * row_bytes + static_cast<std::int64_t>(block) * 18;
        const float scale = __half2float(
            __ushort_as_half(static_cast<unsigned short>(data[0] | (static_cast<unsigned>(data[1]) << 8))));
        const std::uint8_t packed = data[2 + (lane & 15)];
        const int nibble = lane < 16 ? (packed & 0x0F) : (packed >> 4);
        embedding[t * values + v] = __float2bfloat16_rn(scale * static_cast<float>(kIq4nlValues[nibble]));
    }
}

// Gather of the trainable FP32 table: one block per token, each value rounded once to bf16.
__global__ void gather_fp32_kernel(nv_bfloat16* __restrict__ embedding, const int* __restrict__ rows,
                                   const float* __restrict__ table, int head_dim, int heads) {
    const long t = blockIdx.x;
    const int values = heads * head_dim;
    for (int v = static_cast<int>(threadIdx.x); v < values; v += kThreads) {
        const int head = v / head_dim;
        const int within = v - head * head_dim;
        const std::int64_t row = rows[t * heads + head];
        embedding[t * values + v] = __float2bfloat16_rn(table[row * head_dim + within]);
    }
}

__global__ void residual_add_kernel(nv_bfloat16* __restrict__ out, const nv_bfloat16* __restrict__ a,
                                    const nv_bfloat16* __restrict__ b, long n) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16_rn(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

// ------------------------------------------------------------------------------------------------
// Gate forward: one block per token.
template <typename T>
__global__ void gate_forward_kernel(T* __restrict__ u, float* __restrict__ stats, float* __restrict__ global_stats,
                                    const T* __restrict__ r, const T* __restrict__ k, const T* __restrict__ v,
                                    const float* __restrict__ gk, const float* __restrict__ gq,
                                    const int* __restrict__ ids, ZeroTokens zero, int C, float eps) {
    __shared__ float scratch[2 * kThreads / 32];
    const long t = blockIdx.x;
    const long base = t * C;
    float sum_k = 0.0F;
    float sum_r = 0.0F;
    for (int c = static_cast<int>(threadIdx.x); c < C; c += kThreads) {
        const float kv = ld(k, base + c);
        const float rv = ld(r, base + c);
        sum_k = fmaf(kv, kv, sum_k);
        sum_r = fmaf(rv, rv, sum_r);
    }
    block_sum2(sum_k, sum_r, scratch);
    const float inv_k = rsqrtf(sum_k / static_cast<float>(C) + eps);
    const float inv_q = rsqrtf(sum_r / static_cast<float>(C) + eps);
    float dot = 0.0F;
    float unused = 0.0F;
    for (int c = static_cast<int>(threadIdx.x); c < C; c += kThreads) {
        const float kn = ld(k, base + c) * inv_k * gk[c];
        const float qn = ld(r, base + c) * inv_q * gq[c];
        dot = fmaf(kn, qn, dot);
    }
    block_sum2(dot, unused, scratch);
    const float z = dot * rsqrtf(static_cast<float>(C));
    const float g = sigmoid(copysignf(sqrtf(fmaxf(fabsf(z), kGateClamp)), z));
    const bool zeroed = is_zero_token(ids[t], zero);
    const float ge = zeroed ? 0.0F : g;
    float sum_u = 0.0F;
    for (int c = static_cast<int>(threadIdx.x); c < C; c += kThreads) {
        const float value = ge * ld(v, base + c);
        st(u, base + c, value);
        const float rounded = ld(u, base + c);
        sum_u = fmaf(rounded, rounded, sum_u);
    }
    if (global_stats != nullptr) {
        float unused2 = 0.0F;
        block_sum2(sum_u, unused2, scratch);
    }
    if (threadIdx.x == 0) {
        stats[t * kTokenStats + 0] = inv_k;
        stats[t * kTokenStats + 1] = inv_q;
        stats[t * kTokenStats + 2] = z;
        stats[t * kTokenStats + 3] = g;
        if (global_stats != nullptr) {
            atomicAdd(global_stats + 0, 1.0F);
            if (zeroed) {
                atomicAdd(global_stats + 7, 1.0F);
            } else {
                atomicAdd(global_stats + 1, g);
                atomicAdd(global_stats + 2, g * g);
                if (g > 0.9F) atomicAdd(global_stats + 3, 1.0F);
                if (g < 0.1F) atomicAdd(global_stats + 4, 1.0F);
            }
            atomicAdd(global_stats + 5, sum_u);
            atomicAdd(global_stats + 6, sum_r);
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Gate backward: one block per token (see ngram_memory.h for the algebra).
//   dv_c = g ds_c;  dg = sum_c ds_c v_c;  dz = dg g (1-g) * (|z| > 1e-6 ? 0.5 / sqrt|z| : 0)
//   a = dz / sqrt(C) * inv_k * inv_q;  P = sum_c gk_c gq_c r_c k_c
//   dk_c = a (gk_c gq_c r_c - inv_k^2 k_c P / C);  dr_c = a (gk_c gq_c k_c - inv_q^2 r_c P / C)
template <typename T>
__global__ void gate_backward_kernel(T* __restrict__ dk, T* __restrict__ dv, float* __restrict__ w,
                                     T* __restrict__ d_res, float* __restrict__ abs_max, const T* __restrict__ r,
                                     const T* __restrict__ k, const T* __restrict__ v, const float* __restrict__ stats,
                                     const float* __restrict__ gk, const float* __restrict__ gq,
                                     const int* __restrict__ ids, ZeroTokens zero, int C) {
    __shared__ float scratch[2 * kThreads / 32];
    __shared__ float block_max;
    const long t = blockIdx.x;
    const long base = t * C;
    if (abs_max != nullptr && threadIdx.x == 0) block_max = 0.0F;
    const bool zeroed = is_zero_token(ids[t], zero);
    if (zeroed) {
        for (int c = static_cast<int>(threadIdx.x); c < C; c += kThreads) {
            st(dk, base + c, 0.0F);
            st(dv, base + c, 0.0F);
        }
        if (threadIdx.x == 0) w[t] = 0.0F;
        return;  // d_res passes through unchanged
    }
    const float inv_k = stats[t * kTokenStats + 0];
    const float inv_q = stats[t * kTokenStats + 1];
    const float z = stats[t * kTokenStats + 2];
    const float g = stats[t * kTokenStats + 3];
    float dg = 0.0F;
    float P = 0.0F;
    for (int c = static_cast<int>(threadIdx.x); c < C; c += kThreads) {
        dg = fmaf(ld(d_res, base + c), ld(v, base + c), dg);
        P = fmaf(gk[c] * gq[c] * ld(r, base + c), ld(k, base + c), P);
    }
    block_sum2(dg, P, scratch);
    const float az = fabsf(z);
    const float dz = az > kGateClamp ? dg * g * (1.0F - g) * 0.5F * rsqrtf(az) : 0.0F;
    const float a = dz * rsqrtf(static_cast<float>(C)) * inv_k * inv_q;
    const float pk = inv_k * inv_k * P / static_cast<float>(C);
    const float pq = inv_q * inv_q * P / static_cast<float>(C);
    float thread_max = 0.0F;
    for (int c = static_cast<int>(threadIdx.x); c < C; c += kThreads) {
        const float ds = ld(d_res, base + c);
        const float rc = ld(r, base + c);
        const float kc = ld(k, base + c);
        const float gg = gk[c] * gq[c];
        st(dv, base + c, g * ds);
        st(dk, base + c, a * (gg * rc - pk * kc));
        // a == 0 (e.g. W_v = 0: no gate gradient) leaves the residual gradient untouched bit for bit.
        if (a != 0.0F) st(d_res, base + c, ds + a * (gg * kc - pq * rc));
        thread_max = fmaxf(thread_max, fabsf(ld(d_res, base + c)));
    }
    if (threadIdx.x == 0) w[t] = a;
    if (abs_max != nullptr) {
        thread_max = warpReduceMax(thread_max);
        __syncthreads();
        if ((threadIdx.x & 31) == 0) atomicMax(reinterpret_cast<unsigned*>(&block_max), __float_as_uint(thread_max));
        __syncthreads();
        if (threadIdx.x == 0) atomicMax(reinterpret_cast<unsigned*>(abs_max), __float_as_uint(block_max));
    }
}

// ------------------------------------------------------------------------------------------------
// Gain gradients: partials[p, c] = sum over the p-th token range of w_t r_tc k_tc, then a column sum.
template <typename T>
__global__ void gain_partials_kernel(float* __restrict__ partials, const float* __restrict__ w,
                                     const T* __restrict__ r, const T* __restrict__ k, int N, int C, int per_part) {
    const int c = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    if (c >= C) return;
    const int p = static_cast<int>(blockIdx.y);
    const int t0 = p * per_part;
    const int t1 = min(N, t0 + per_part);
    float acc = 0.0F;
    for (int t = t0; t < t1; ++t) {
        const float wt = w[t];
        if (wt != 0.0F) acc = fmaf(wt, ld(r, static_cast<long>(t) * C + c) * ld(k, static_cast<long>(t) * C + c), acc);
    }
    partials[static_cast<long>(p) * C + c] = acc;
}

template <typename G>
__global__ void gain_finalize_kernel(G* __restrict__ dgk, G* __restrict__ dgq, const float* __restrict__ partials,
                                     int parts, const float* __restrict__ gk, const float* __restrict__ gq, int C) {
    const int c = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    if (c >= C) return;
    float s = 0.0F;
    for (int p = 0; p < parts; ++p) s += partials[static_cast<long>(p) * C + c];
    st(dgk, c, ld(dgk, c) + gq[c] * s);
    st(dgq, c, ld(dgq, c) + gk[c] * s);
}

// ------------------------------------------------------------------------------------------------
// Fused residual + RMSNorm with a third addend: kernels/rmsnorm.cu's
// fused_residual_rmsnorm_forward_kernel with `res_f = a + b (+ u where u != 0)`.
__global__ void residual3_rmsnorm_kernel(nv_bfloat16* residual, nv_bfloat16* normed, float* rrms,
                                         const nv_bfloat16* inp1, const nv_bfloat16* inp2, const nv_bfloat16* inp3,
                                         const nv_bfloat16* weight, float epsilon, int N, int C) {
    using floatX = nv_bfloat16;
    using x128 = GenericVector<floatX, 16 / sizeof(floatX)>;
    assert(blockDim.x == WARP_SIZE);

    extern __shared__ __align__(16) unsigned char params[];
    x128* s_weight = reinterpret_cast<x128*>(params);

    int sidx = (threadIdx.x + WARP_SIZE * threadIdx.y) * x128::size;
    for (int i = sidx; i < C; i += blockDim.y * WARP_SIZE * x128::size) {
        s_weight[i / x128::size] = x128::load(weight + i);
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx >= N) return;

    residual += static_cast<long>(C) * idx;
    normed += static_cast<long>(C) * idx;
    inp1 += static_cast<long>(C) * idx;
    inp2 += static_cast<long>(C) * idx;
    inp3 += static_cast<long>(C) * idx;

    float sum_squared = 0.f;
    for (int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in1 = x128::load_cs(inp1 + c);
        const x128 in2 = x128::load_cs(inp2 + c);
        const x128 in3 = x128::load_cs(inp3 + c);
        x128 out;
        for (int k = 0; k < x128::size; ++k) {
            float res_f = (float)in1[k] + (float)in2[k];
            const float u = (float)in3[k];
            if (u != 0.f) res_f += u;
            out[k] = static_cast<floatX>(res_f);
            sum_squared += res_f * res_f;
        }
        out.store(residual + c);
    }

    sum_squared = warpReduceSum(sum_squared) / C;
    float s = rsqrtf(sum_squared + epsilon);

    for (int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in1 = x128::load_cs(inp1 + c);
        const x128 in2 = x128::load_cs(inp2 + c);
        const x128 in3 = x128::load_cs(inp3 + c);
        const x128 w = s_weight[c / x128::size];
        x128 out;
        for (int k = 0; k < x128::size; ++k) {
            float res_f = (float)in1[k] + (float)in2[k];
            const float u = (float)in3[k];
            if (u != 0.f) res_f += u;
            float n = s * res_f;
            float w_val = (float)w[k];
            out[k] = (floatX)(n * w_val);
        }
        out.store(normed + c);
    }

    if (threadIdx.x == 0) {
        rrms[idx] = s;
    }
}

__global__ void bf16_to_fp32_kernel(float* dst, const nv_bfloat16* src, long n) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

unsigned blocks_for(long n, int threads) { return static_cast<unsigned>((n + threads - 1) / threads); }

}  // namespace

void rows(int* rows, const int* ids, const int* positions, const int* carry, int B, int T, const Hash& hash,
          cudaStream_t stream) {
    const long n = static_cast<long>(B) * T;
    if (n <= 0) return;
    rows_kernel<<<blocks_for(n, 128), 128, 0, stream>>>(rows, ids, positions, carry, B, T, hash);
    CUDA_CHECK(cudaGetLastError());
}

void gather_iq4nl(nv_bfloat16* emb, const int* rows, const std::uint8_t* table, int row_bytes, int head_dim,
                  int heads, int N, cudaStream_t stream, RowShuffle shuffle) {
    if (N <= 0) return;
    gather_kernel<<<static_cast<unsigned>(N), kThreads, 0, stream>>>(emb, rows, table, row_bytes, head_dim, heads,
                                                                      shuffle);
    CUDA_CHECK(cudaGetLastError());
}

void gather_fp32(nv_bfloat16* emb, const int* rows, const float* table, int head_dim, int heads, int N,
                 cudaStream_t stream) {
    if (N <= 0) return;
    gather_fp32_kernel<<<static_cast<unsigned>(N), kThreads, 0, stream>>>(emb, rows, table, head_dim, heads);
    CUDA_CHECK(cudaGetLastError());
}

void residual_add(nv_bfloat16* out, const nv_bfloat16* a, const nv_bfloat16* b, long n, cudaStream_t stream) {
    if (n <= 0) return;
    residual_add_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(out, a, b, n);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void gate_forward(T* u, float* stats, float* global_stats, const T* r, const T* k, const T* v, const float* gk,
                  const float* gq, const int* ids, ZeroTokens zero, int N, int C, float eps, cudaStream_t stream) {
    if (N <= 0) return;
    gate_forward_kernel<T><<<static_cast<unsigned>(N), kThreads, 0, stream>>>(u, stats, global_stats, r, k, v, gk, gq,
                                                                               ids, zero, C, eps);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void gate_backward(T* dk, T* dv, float* w, T* d_res, float* abs_max, const T* r, const T* k, const T* v,
                   const float* stats, const float* gk, const float* gq, const int* ids, ZeroTokens zero, int N,
                   int C, cudaStream_t stream) {
    if (N <= 0) return;
    gate_backward_kernel<T><<<static_cast<unsigned>(N), kThreads, 0, stream>>>(dk, dv, w, d_res, abs_max, r, k, v,
                                                                                stats, gk, gq, ids, zero, C);
    CUDA_CHECK(cudaGetLastError());
}

int gain_grad_parts(int N) { return std::max(1, std::min(256, (N + 63) / 64)); }

template <typename T, typename G>
void gain_grads(G* dgk, G* dgq, float* partials, const float* w, const T* r, const T* k, const float* gk,
                const float* gq, int N, int C, cudaStream_t stream) {
    if (N <= 0) return;
    const int parts = gain_grad_parts(N);
    const int per_part = (N + parts - 1) / parts;
    const dim3 grid(blocks_for(C, kThreads), static_cast<unsigned>(parts));
    gain_partials_kernel<T><<<grid, kThreads, 0, stream>>>(partials, w, r, k, N, C, per_part);
    CUDA_CHECK(cudaGetLastError());
    gain_finalize_kernel<G><<<blocks_for(C, kThreads), kThreads, 0, stream>>>(dgk, dgq, partials, parts, gk, gq, C);
    CUDA_CHECK(cudaGetLastError());
}

void residual3_rmsnorm_forward(nv_bfloat16* residual, nv_bfloat16* normed, float* rrms, const nv_bfloat16* a,
                               const nv_bfloat16* b, const nv_bfloat16* u, const nv_bfloat16* weight, float eps,
                               int N, int C, cudaStream_t stream) {
    // The launch geometry of rmsnorm_forward_unified_imp: one warp per token, 8 tokens per block.
    const int block_size = 256;
    const int block_y = block_size / WARP_SIZE;
    const int grid_size = div_ceil(N, block_y);
    const size_t smem = (1 + block_y) * C * sizeof(nv_bfloat16);
    CUDA_CHECK(cudaFuncSetAttribute(residual3_rmsnorm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(smem)));
    residual3_rmsnorm_kernel<<<grid_size, dim3(WARP_SIZE, block_y), smem, stream>>>(residual, normed, rrms, a, b, u,
                                                                                    weight, eps, N, C);
    CUDA_CHECK(cudaGetLastError());
}

void to_fp32(float* dst, const void* src, bool bf16, long n, cudaStream_t stream) {
    if (n <= 0) return;
    if (!bf16) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        return;
    }
    bf16_to_fp32_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(dst, static_cast<const nv_bfloat16*>(src), n);
    CUDA_CHECK(cudaGetLastError());
}

template void gate_forward<float>(float*, float*, float*, const float*, const float*, const float*, const float*,
                                  const float*, const int*, ZeroTokens, int, int, float, cudaStream_t);
template void gate_forward<nv_bfloat16>(nv_bfloat16*, float*, float*, const nv_bfloat16*, const nv_bfloat16*,
                                        const nv_bfloat16*, const float*, const float*, const int*, ZeroTokens, int,
                                        int, float, cudaStream_t);
template void gate_backward<float>(float*, float*, float*, float*, float*, const float*, const float*, const float*,
                                   const float*, const float*, const float*, const int*, ZeroTokens, int, int,
                                   cudaStream_t);
template void gate_backward<nv_bfloat16>(nv_bfloat16*, nv_bfloat16*, float*, nv_bfloat16*, float*, const nv_bfloat16*,
                                         const nv_bfloat16*, const nv_bfloat16*, const float*, const float*,
                                         const float*, const int*, ZeroTokens, int, int, cudaStream_t);
template void gain_grads<float, float>(float*, float*, float*, const float*, const float*, const float*, const float*,
                                       const float*, int, int, cudaStream_t);
template void gain_grads<nv_bfloat16, float>(float*, float*, float*, const float*, const nv_bfloat16*,
                                             const nv_bfloat16*, const float*, const float*, int, int, cudaStream_t);
template void gain_grads<nv_bfloat16, nv_bfloat16>(nv_bfloat16*, nv_bfloat16*, float*, const float*,
                                                   const nv_bfloat16*, const nv_bfloat16*, const float*, const float*,
                                                   int, int, cudaStream_t);
template void gain_grads<float, nv_bfloat16>(nv_bfloat16*, nv_bfloat16*, float*, const float*, const float*,
                                             const float*, const float*, const float*, int, int, cudaStream_t);

}  // namespace ngram_memory
