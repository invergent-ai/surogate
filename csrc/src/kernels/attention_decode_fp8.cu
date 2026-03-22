// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// FP8 E4M3 KV-cache kernels: quantize BF16→FP8 on append, dequant FP8→BF16 for attention.
// Per-head quantization: one scale per (batch, position, head).

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
#include <cmath>

#include "kernels/attention_decode.h"

namespace {

constexpr float kFP8E4M3Max = 448.0f;

__device__ __forceinline__ float fp8_e4m3_to_float(__nv_fp8_e4m3 x) {
    return __half2float(
        __nv_cvt_fp8_to_halfraw(
            x.__x,
            __nv_fp8_interpretation_t::__NV_E4M3));
}

// ============================================================================
// Contiguous FP8 append
// ============================================================================

/// Grid: (batch_size, Hkv, 2)  — dim z: 0=K, 1=V
/// Block: (Hs)
/// For each (batch, head), find absmax across Hs dims, compute scale,
/// quantize each element to FP8 E4M3, write to cache.
__global__ void kv_cache_append_fp8_kernel(
        __nv_fp8_e4m3* __restrict__ k_cache,
        __nv_fp8_e4m3* __restrict__ v_cache,
        float* __restrict__ k_scales,
        float* __restrict__ v_scales,
        const nv_bfloat16* __restrict__ qkv_rope,
        const int* __restrict__ seq_lens_gpu,
        int max_seq_len, int Hq, int Hkv, int Hs) {

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int kv_idx = blockIdx.z;  // 0=K, 1=V
    const int dim_idx = threadIdx.x;
    if (dim_idx >= Hs) return;

    const int seq_pos = seq_lens_gpu[batch_idx];
    const int H_total = Hq + 2 * Hkv;

    // Source offset in interleaved QKV
    const int src_head = (kv_idx == 0) ? (Hq + head_idx) : (Hq + Hkv + head_idx);
    const int src_offset = batch_idx * H_total * Hs + src_head * Hs + dim_idx;
    const float val = __bfloat162float(qkv_rope[src_offset]);

    // Shared memory for per-head absmax reduction
    extern __shared__ float smem[];
    smem[dim_idx] = fabsf(val);
    __syncthreads();

    // Warp-level reduction for absmax
    for (int stride = Hs / 2; stride > 0; stride >>= 1) {
        if (dim_idx < stride) {
            smem[dim_idx] = fmaxf(smem[dim_idx], smem[dim_idx + stride]);
        }
        __syncthreads();
    }
    const float absmax = fmaxf(smem[0], 1e-12f);
    const float scale = kFP8E4M3Max / absmax;
    const float inv_scale = absmax / kFP8E4M3Max;

    // Quantize and write
    __nv_fp8_e4m3 quantized;
    quantized.__x = __nv_cvt_float_to_fp8(
        val * scale,
        __nv_saturation_t::__NV_SATFINITE,
        __nv_fp8_interpretation_t::__NV_E4M3);

    auto* cache = (kv_idx == 0) ? k_cache : v_cache;
    const long cache_offset = static_cast<long>(batch_idx) * max_seq_len * Hkv * Hs
                            + static_cast<long>(seq_pos) * Hkv * Hs
                            + head_idx * Hs
                            + dim_idx;
    cache[cache_offset] = quantized;

    // Write inv_scale (one per head per position)
    if (dim_idx == 0) {
        auto* scales = (kv_idx == 0) ? k_scales : v_scales;
        const long scale_offset = static_cast<long>(batch_idx) * max_seq_len * Hkv
                                + static_cast<long>(seq_pos) * Hkv
                                + head_idx;
        scales[scale_offset] = inv_scale;
    }
}

// ============================================================================
// Paged FP8 append
// ============================================================================

__global__ void kv_cache_store_paged_fp8_kernel(
        __nv_fp8_e4m3* __restrict__ k_pages,
        __nv_fp8_e4m3* __restrict__ v_pages,
        float* __restrict__ k_scales,
        float* __restrict__ v_scales,
        const nv_bfloat16* __restrict__ qkv_rope,
        const int* __restrict__ block_table,
        int block_table_stride,
        int page_block_size,
        int T, int Hq, int Hkv, int Hs, int start_pos) {

    const int bt_idx = blockIdx.x;   // batch_idx * T + t
    const int head_idx = blockIdx.y;
    const int kv_idx = blockIdx.z;   // 0=K, 1=V
    const int dim_idx = threadIdx.x;
    if (dim_idx >= Hs) return;

    const int batch_idx = bt_idx / T;
    const int t = bt_idx % T;
    const int t_abs = start_pos + t;
    const int H_total = Hq + 2 * Hkv;
    const int page_elems = page_block_size * Hkv * Hs;
    const int page_scale_elems = page_block_size * Hkv;

    const int src_head = (kv_idx == 0) ? (Hq + head_idx) : (Hq + Hkv + head_idx);
    const long src_base = static_cast<long>(bt_idx) * H_total * Hs;
    const long src = src_base + src_head * Hs + dim_idx;
    const float val = __bfloat162float(qkv_rope[src]);

    extern __shared__ float smem[];
    smem[dim_idx] = fabsf(val);
    __syncthreads();
    for (int stride = Hs / 2; stride > 0; stride >>= 1) {
        if (dim_idx < stride) smem[dim_idx] = fmaxf(smem[dim_idx], smem[dim_idx + stride]);
        __syncthreads();
    }
    const float absmax = fmaxf(smem[0], 1e-12f);
    const float scale = kFP8E4M3Max / absmax;
    const float inv_scale = absmax / kFP8E4M3Max;
    __nv_fp8_e4m3 quantized;
    quantized.__x = __nv_cvt_float_to_fp8(
        val * scale,
        __nv_saturation_t::__NV_SATFINITE,
        __nv_fp8_interpretation_t::__NV_E4M3);

    const int virtual_page = t_abs / page_block_size;
    const int page_offset = t_abs % page_block_size;
    const int physical_page = block_table[batch_idx * block_table_stride + virtual_page];

    auto* pages = (kv_idx == 0) ? k_pages : v_pages;
    const long dst = static_cast<long>(physical_page) * page_elems
                   + page_offset * Hkv * Hs + head_idx * Hs + dim_idx;
    pages[dst] = quantized;

    if (dim_idx == 0) {
        auto* scales = (kv_idx == 0) ? k_scales : v_scales;
        const long scale_dst = static_cast<long>(physical_page) * page_scale_elems
                             + page_offset * Hkv + head_idx;
        scales[scale_dst] = inv_scale;
    }
}

__global__ void kv_cache_append_paged_fp8_kernel(
        __nv_fp8_e4m3* __restrict__ k_pages,
        __nv_fp8_e4m3* __restrict__ v_pages,
        float* __restrict__ k_scales,
        float* __restrict__ v_scales,
        const nv_bfloat16* __restrict__ qkv_rope,
        const int* __restrict__ seq_lens_gpu,
        const int* __restrict__ block_table,
        int block_table_stride,
        int page_block_size,
        int Hq, int Hkv, int Hs) {

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int kv_idx = blockIdx.z;
    const int dim_idx = threadIdx.x;
    if (dim_idx >= Hs) return;

    const int seq_pos = seq_lens_gpu[batch_idx];
    const int H_total = Hq + 2 * Hkv;
    const int page_elems = page_block_size * Hkv * Hs;
    const int page_scale_elems = page_block_size * Hkv;

    const int src_head = (kv_idx == 0) ? (Hq + head_idx) : (Hq + Hkv + head_idx);
    const int src_offset = batch_idx * H_total * Hs + src_head * Hs + dim_idx;
    const float val = __bfloat162float(qkv_rope[src_offset]);

    extern __shared__ float smem[];
    smem[dim_idx] = fabsf(val);
    __syncthreads();
    for (int stride = Hs / 2; stride > 0; stride >>= 1) {
        if (dim_idx < stride) smem[dim_idx] = fmaxf(smem[dim_idx], smem[dim_idx + stride]);
        __syncthreads();
    }
    const float absmax = fmaxf(smem[0], 1e-12f);
    const float scale = kFP8E4M3Max / absmax;
    const float inv_scale = absmax / kFP8E4M3Max;

    __nv_fp8_e4m3 quantized;
    quantized.__x = __nv_cvt_float_to_fp8(
        val * scale,
        __nv_saturation_t::__NV_SATFINITE,
        __nv_fp8_interpretation_t::__NV_E4M3);

    const int virtual_page = seq_pos / page_block_size;
    const int page_offset = seq_pos % page_block_size;
    const int physical_page = block_table[batch_idx * block_table_stride + virtual_page];

    auto* pages = (kv_idx == 0) ? k_pages : v_pages;
    const long dest = static_cast<long>(physical_page) * page_elems
                    + page_offset * Hkv * Hs + head_idx * Hs + dim_idx;
    pages[dest] = quantized;

    if (dim_idx == 0) {
        auto* scales = (kv_idx == 0) ? k_scales : v_scales;
        const long scale_dest = static_cast<long>(physical_page) * page_scale_elems
                              + page_offset * Hkv + head_idx;
        scales[scale_dest] = inv_scale;
    }
}

// ============================================================================
// Contiguous FP8→BF16 dequant
// ============================================================================

/// Grid: (batch_size, Hkv)  Block: (Hs)
/// Processes all valid positions [0, k_len) for each batch item.
/// We use a grid-stride loop over positions.
__global__ void kv_cache_dequant_fp8_kernel(
        nv_bfloat16* __restrict__ k_out,
        nv_bfloat16* __restrict__ v_out,
        const __nv_fp8_e4m3* __restrict__ k_fp8,
        const __nv_fp8_e4m3* __restrict__ v_fp8,
        const float* __restrict__ k_scales,
        const float* __restrict__ v_scales,
        const int* __restrict__ seqused_k_gpu,
        int max_seq_len, int Hkv, int Hs) {

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int dim_idx = threadIdx.x;
    if (dim_idx >= Hs) return;

    const int seq_len = seqused_k_gpu[batch_idx];

    for (int pos = 0; pos < seq_len; ++pos) {
        const long offset = static_cast<long>(batch_idx) * max_seq_len * Hkv * Hs
                          + static_cast<long>(pos) * Hkv * Hs
                          + head_idx * Hs + dim_idx;
        const long scale_offset = static_cast<long>(batch_idx) * max_seq_len * Hkv
                                + static_cast<long>(pos) * Hkv + head_idx;

        const float k_raw = __half2float(
            __nv_cvt_fp8_to_halfraw(
                k_fp8[offset].__x,
                __nv_fp8_interpretation_t::__NV_E4M3));
        const float v_raw = __half2float(
            __nv_cvt_fp8_to_halfraw(
                v_fp8[offset].__x,
                __nv_fp8_interpretation_t::__NV_E4M3));
        float k_val = k_raw * k_scales[scale_offset];
        float v_val = v_raw * v_scales[scale_offset];
        k_out[offset] = __float2bfloat16(k_val);
        v_out[offset] = __float2bfloat16(v_val);
    }
}

// ============================================================================
// Paged FP8→BF16 dequant
// ============================================================================

__global__ void kv_cache_dequant_paged_fp8_kernel(
        nv_bfloat16* __restrict__ k_out,
        nv_bfloat16* __restrict__ v_out,
        const __nv_fp8_e4m3* __restrict__ k_pages_fp8,
        const __nv_fp8_e4m3* __restrict__ v_pages_fp8,
        const float* __restrict__ k_scales,
        const float* __restrict__ v_scales,
        const int* __restrict__ seqused_k_gpu,
        const int* __restrict__ block_table,
        int block_table_stride,
        int page_block_size,
        int max_seq_len, int Hkv, int Hs) {

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int dim_idx = threadIdx.x;
    if (dim_idx >= Hs) return;

    const int seq_len = seqused_k_gpu[batch_idx];
    const int page_elems = page_block_size * Hkv * Hs;
    const int page_scale_elems = page_block_size * Hkv;

    for (int pos = 0; pos < seq_len; ++pos) {
        const int vp = pos / page_block_size;
        const int po = pos % page_block_size;
        const int pp = block_table[batch_idx * block_table_stride + vp];

        const long src = static_cast<long>(pp) * page_elems + po * Hkv * Hs + head_idx * Hs + dim_idx;
        const long scale_src = static_cast<long>(pp) * page_scale_elems + po * Hkv + head_idx;

        const float k_raw = __half2float(
            __nv_cvt_fp8_to_halfraw(
                k_pages_fp8[src].__x,
                __nv_fp8_interpretation_t::__NV_E4M3));
        const float v_raw = __half2float(
            __nv_cvt_fp8_to_halfraw(
                v_pages_fp8[src].__x,
                __nv_fp8_interpretation_t::__NV_E4M3));
        float k_val = k_raw * k_scales[scale_src];
        float v_val = v_raw * v_scales[scale_src];

        // Output to contiguous BF16 buffer for Flash Attention
        const long dst = static_cast<long>(batch_idx) * max_seq_len * Hkv * Hs
                       + static_cast<long>(pos) * Hkv * Hs + head_idx * Hs + dim_idx;
        k_out[dst] = __float2bfloat16(k_val);
        v_out[dst] = __float2bfloat16(v_val);
    }
}

// ============================================================================
// Paged FP8 direct decode (on-the-fly dequant + online softmax)
// ============================================================================

/// Grid: (batch_size, Hq)  Block: (next_pow2(Hs), <=256)
/// One CTA computes one (batch, q_head) decode output vector.
__global__ void attention_decode_paged_fp8_direct_kernel(
        nv_bfloat16* __restrict__ out,
        float* __restrict__ lse,
        const nv_bfloat16* __restrict__ q,
        const __nv_fp8_e4m3* __restrict__ k_pages_fp8,
        const __nv_fp8_e4m3* __restrict__ v_pages_fp8,
        const float* __restrict__ k_scales,
        const float* __restrict__ v_scales,
        const int32_t* __restrict__ seqused_k,
        const int* __restrict__ block_table,
        int block_table_stride,
        int page_block_size,
        int batch_size, int Hq, int Hkv, int Hs) {

    const int batch_idx = blockIdx.x;
    const int q_head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (batch_idx >= batch_size || q_head_idx >= Hq) return;

    constexpr int kMaxThreads = 256;
    __shared__ float s_partials[kMaxThreads];
    __shared__ float s_alpha;
    __shared__ float s_beta;
    __shared__ float s_m;
    __shared__ float s_d;

    const int q_per_kv = Hq / Hkv;
    const int kv_head_idx = q_head_idx / q_per_kv;
    const int seq_len = seqused_k[batch_idx];
    const int page_elems = page_block_size * Hkv * Hs;
    const int page_scale_elems = page_block_size * Hkv;
    const float sm_scale = rsqrtf(static_cast<float>(Hs));

    float q_val = 0.0f;
    if (tid < Hs) {
        const long q_offset = (static_cast<long>(batch_idx) * Hq + q_head_idx) * Hs + tid;
        q_val = __bfloat162float(q[q_offset]);
    }

    if (tid == 0) {
        s_m = -INFINITY;
        s_d = 0.0f;
        s_alpha = 0.0f;
        s_beta = 0.0f;
    }
    __syncthreads();

    float out_acc = 0.0f;

    for (int pos = 0; pos < seq_len; ++pos) {
        const int vp = pos / page_block_size;
        const int po = pos % page_block_size;
        const int pp = block_table[batch_idx * block_table_stride + vp];

        const long base = static_cast<long>(pp) * page_elems
                        + static_cast<long>(po) * Hkv * Hs
                        + static_cast<long>(kv_head_idx) * Hs;
        const long scale_base = static_cast<long>(pp) * page_scale_elems
                              + static_cast<long>(po) * Hkv
                              + kv_head_idx;

        float k_val = 0.0f;
        float v_val = 0.0f;
        if (tid < Hs) {
            k_val = fp8_e4m3_to_float(k_pages_fp8[base + tid]) * k_scales[scale_base];
            v_val = fp8_e4m3_to_float(v_pages_fp8[base + tid]) * v_scales[scale_base];
        }

        s_partials[tid] = (tid < Hs) ? (q_val * k_val) : 0.0f;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_partials[tid] += s_partials[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const float score = s_partials[0] * sm_scale;
            const float m_prev = s_m;
            const float d_prev = s_d;
            const float m_new = fmaxf(m_prev, score);
            const float alpha = isfinite(m_prev) ? expf(m_prev - m_new) : 0.0f;
            const float beta = expf(score - m_new);
            s_alpha = alpha;
            s_beta = beta;
            s_m = m_new;
            s_d = d_prev * alpha + beta;
        }
        __syncthreads();

        if (tid < Hs) {
            out_acc = out_acc * s_alpha + s_beta * v_val;
        }
        __syncthreads();
    }

    if (tid < Hs) {
        const float denom = s_d;
        const float out_val = (denom > 0.0f) ? (out_acc / denom) : 0.0f;
        const long out_offset = (static_cast<long>(batch_idx) * Hq + q_head_idx) * Hs + tid;
        out[out_offset] = __float2bfloat16(out_val);
    }

    if (tid == 0 && lse) {
        const float lse_val = (s_d > 0.0f && isfinite(s_m)) ? (s_m + logf(s_d)) : -INFINITY;
        lse[static_cast<long>(q_head_idx) * batch_size + batch_idx] = lse_val;
    }
}

}  // anonymous namespace

// ============================================================================
// Host wrappers
// ============================================================================

void kv_cache_append_fp8(
        __nv_fp8_e4m3* k_cache_fp8, __nv_fp8_e4m3* v_cache_fp8,
        float* k_scales, float* v_scales,
        const nv_bfloat16* qkv_rope,
        const int* seq_lens_gpu,
        int batch_size, int max_seq_len,
        int Hq, int Hkv, int Hs,
        cudaStream_t stream) {

    dim3 grid(batch_size, Hkv, 2);  // z: 0=K, 1=V
    dim3 block(Hs);
    const int smem_bytes = Hs * sizeof(float);
    kv_cache_append_fp8_kernel<<<grid, block, smem_bytes, stream>>>(
        k_cache_fp8, v_cache_fp8, k_scales, v_scales,
        qkv_rope, seq_lens_gpu, max_seq_len, Hq, Hkv, Hs);
}

void kv_cache_append_paged_fp8(
        __nv_fp8_e4m3* k_pages_fp8, __nv_fp8_e4m3* v_pages_fp8,
        float* k_scales, float* v_scales,
        const nv_bfloat16* qkv_rope,
        const int* seq_lens_gpu,
        const int* block_table, int block_table_stride,
        int page_block_size,
        int batch_size, int Hq, int Hkv, int Hs,
        cudaStream_t stream) {

    dim3 grid(batch_size, Hkv, 2);
    dim3 block(Hs);
    const int smem_bytes = Hs * sizeof(float);
    kv_cache_append_paged_fp8_kernel<<<grid, block, smem_bytes, stream>>>(
        k_pages_fp8, v_pages_fp8, k_scales, v_scales,
        qkv_rope, seq_lens_gpu,
        block_table, block_table_stride, page_block_size,
        Hq, Hkv, Hs);
}

void kv_cache_store_paged_fp8(
        __nv_fp8_e4m3* k_pages_fp8, __nv_fp8_e4m3* v_pages_fp8,
        float* k_scales, float* v_scales,
        const nv_bfloat16* qkv_rope,
        const int* block_table, int block_table_stride,
        int page_block_size,
        int batch_size, int T,
        int Hq, int Hkv, int Hs,
        int start_pos,
        cudaStream_t stream) {

    dim3 grid(batch_size * T, Hkv, 2);
    dim3 block(Hs);
    const int smem_bytes = Hs * sizeof(float);
    kv_cache_store_paged_fp8_kernel<<<grid, block, smem_bytes, stream>>>(
        k_pages_fp8, v_pages_fp8, k_scales, v_scales,
        qkv_rope, block_table, block_table_stride, page_block_size,
        T, Hq, Hkv, Hs, start_pos);
}

void kv_cache_dequant_fp8_to_bf16(
        nv_bfloat16* k_out_bf16, nv_bfloat16* v_out_bf16,
        const __nv_fp8_e4m3* k_cache_fp8, const __nv_fp8_e4m3* v_cache_fp8,
        const float* k_scales, const float* v_scales,
        const int* seqused_k_gpu,
        int batch_size, int max_seq_len,
        int Hkv, int Hs,
        cudaStream_t stream) {

    dim3 grid(batch_size, Hkv);
    dim3 block(Hs);
    kv_cache_dequant_fp8_kernel<<<grid, block, 0, stream>>>(
        k_out_bf16, v_out_bf16,
        k_cache_fp8, v_cache_fp8,
        k_scales, v_scales,
        seqused_k_gpu, max_seq_len, Hkv, Hs);
}

void kv_cache_dequant_paged_fp8_to_bf16(
        nv_bfloat16* k_out_bf16, nv_bfloat16* v_out_bf16,
        const __nv_fp8_e4m3* k_pages_fp8, const __nv_fp8_e4m3* v_pages_fp8,
        const float* k_scales, const float* v_scales,
        const int* seqused_k_gpu,
        const int* block_table, int block_table_stride,
        int page_block_size,
        int batch_size, int max_seq_len,
        int Hkv, int Hs,
        cudaStream_t stream) {

    dim3 grid(batch_size, Hkv);
    dim3 block(Hs);
    kv_cache_dequant_paged_fp8_kernel<<<grid, block, 0, stream>>>(
        k_out_bf16, v_out_bf16,
        k_pages_fp8, v_pages_fp8,
        k_scales, v_scales,
        seqused_k_gpu, block_table, block_table_stride,
        page_block_size, max_seq_len, Hkv, Hs);
}

bool attention_decode_paged_fp8_direct(
        nv_bfloat16* out, float* lse,
        const nv_bfloat16* q,
        const __nv_fp8_e4m3* k_pages_fp8, const __nv_fp8_e4m3* v_pages_fp8,
        const float* k_scales, const float* v_scales,
        const int32_t* seqused_k,
        const int* block_table, int block_table_stride,
        int page_block_size,
        int batch_size, int Hq, int Hkv, int Hs,
        cudaStream_t stream) {

    if (!out || !q || !k_pages_fp8 || !v_pages_fp8 || !k_scales || !v_scales ||
        !seqused_k || !block_table) {
        return false;
    }
    if (batch_size <= 0 || Hq <= 0 || Hkv <= 0 || Hs <= 0 ||
        page_block_size <= 0 || block_table_stride <= 0) {
        return false;
    }
    if (Hq % Hkv != 0) {
        return false;
    }
    if (Hs > 256) {
        return false;
    }

    int block_threads = 1;
    while (block_threads < Hs) block_threads <<= 1;
    if (block_threads < 32) block_threads = 32;
    if (block_threads > 256) {
        return false;
    }

    dim3 grid(batch_size, Hq);
    dim3 block(block_threads);
    attention_decode_paged_fp8_direct_kernel<<<grid, block, 0, stream>>>(
        out, lse, q,
        k_pages_fp8, v_pages_fp8,
        k_scales, v_scales,
        seqused_k,
        block_table, block_table_stride,
        page_block_size,
        batch_size, Hq, Hkv, Hs);

    return cudaGetLastError() == cudaSuccess;
}
