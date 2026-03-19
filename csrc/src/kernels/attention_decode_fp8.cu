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

#include "kernels/attention_decode.h"

namespace {

constexpr float kFP8E4M3Max = 448.0f;

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
    const __nv_fp8_e4m3 quantized = __nv_fp8_e4m3(val * scale);

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

    const __nv_fp8_e4m3 quantized = __nv_fp8_e4m3(val * scale);

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
/// Processes ALL valid positions [0, seq_len) for each batch item.
/// We use a grid-stride loop over positions.
__global__ void kv_cache_dequant_fp8_kernel(
        nv_bfloat16* __restrict__ k_out,
        nv_bfloat16* __restrict__ v_out,
        const __nv_fp8_e4m3* __restrict__ k_fp8,
        const __nv_fp8_e4m3* __restrict__ v_fp8,
        const float* __restrict__ k_scales,
        const float* __restrict__ v_scales,
        const int* __restrict__ seq_lens_gpu,
        int max_seq_len, int Hkv, int Hs) {

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int dim_idx = threadIdx.x;
    if (dim_idx >= Hs) return;

    const int seq_len = seq_lens_gpu[batch_idx];

    for (int pos = 0; pos < seq_len; ++pos) {
        const long offset = static_cast<long>(batch_idx) * max_seq_len * Hkv * Hs
                          + static_cast<long>(pos) * Hkv * Hs
                          + head_idx * Hs + dim_idx;
        const long scale_offset = static_cast<long>(batch_idx) * max_seq_len * Hkv
                                + static_cast<long>(pos) * Hkv + head_idx;

        float k_val = static_cast<float>(k_fp8[offset]) * k_scales[scale_offset];
        float v_val = static_cast<float>(v_fp8[offset]) * v_scales[scale_offset];
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
        const int* __restrict__ seq_lens_gpu,
        const int* __restrict__ block_table,
        int block_table_stride,
        int page_block_size,
        int max_seq_len, int Hkv, int Hs) {

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int dim_idx = threadIdx.x;
    if (dim_idx >= Hs) return;

    const int seq_len = seq_lens_gpu[batch_idx];
    const int page_elems = page_block_size * Hkv * Hs;
    const int page_scale_elems = page_block_size * Hkv;

    for (int pos = 0; pos < seq_len; ++pos) {
        const int vp = pos / page_block_size;
        const int po = pos % page_block_size;
        const int pp = block_table[batch_idx * block_table_stride + vp];

        const long src = static_cast<long>(pp) * page_elems + po * Hkv * Hs + head_idx * Hs + dim_idx;
        const long scale_src = static_cast<long>(pp) * page_scale_elems + po * Hkv + head_idx;

        float k_val = static_cast<float>(k_pages_fp8[src]) * k_scales[scale_src];
        float v_val = static_cast<float>(v_pages_fp8[src]) * v_scales[scale_src];

        // Output to contiguous BF16 buffer for Flash Attention
        const long dst = static_cast<long>(batch_idx) * max_seq_len * Hkv * Hs
                       + static_cast<long>(pos) * Hkv * Hs + head_idx * Hs + dim_idx;
        k_out[dst] = __float2bfloat16(k_val);
        v_out[dst] = __float2bfloat16(v_val);
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

void kv_cache_dequant_fp8_to_bf16(
        nv_bfloat16* k_out_bf16, nv_bfloat16* v_out_bf16,
        const __nv_fp8_e4m3* k_cache_fp8, const __nv_fp8_e4m3* v_cache_fp8,
        const float* k_scales, const float* v_scales,
        const int* seq_lens_gpu,
        int batch_size, int max_seq_len,
        int Hkv, int Hs,
        cudaStream_t stream) {

    dim3 grid(batch_size, Hkv);
    dim3 block(Hs);
    kv_cache_dequant_fp8_kernel<<<grid, block, 0, stream>>>(
        k_out_bf16, v_out_bf16,
        k_cache_fp8, v_cache_fp8,
        k_scales, v_scales,
        seq_lens_gpu, max_seq_len, Hkv, Hs);
}

void kv_cache_dequant_paged_fp8_to_bf16(
        nv_bfloat16* k_out_bf16, nv_bfloat16* v_out_bf16,
        const __nv_fp8_e4m3* k_pages_fp8, const __nv_fp8_e4m3* v_pages_fp8,
        const float* k_scales, const float* v_scales,
        const int* seq_lens_gpu,
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
        seq_lens_gpu, block_table, block_table_stride,
        page_block_size, max_seq_len, Hkv, Hs);
}
