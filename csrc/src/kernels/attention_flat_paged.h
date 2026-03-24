// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Flat-token attention with paged KV-cache.
//
// Two-phase API:
//   1. flat_attention_plan() — run ONCE per flat_step (CPU-side, one sync)
//   2. attention_flat_paged_flashinfer() — run per layer (GPU-only, no sync)

#ifndef SUROGATE_SRC_KERNELS_ATTENTION_FLAT_PAGED_H
#define SUROGATE_SRC_KERNELS_ATTENTION_FLAT_PAGED_H

#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

/// Run FlashInfer PrefillPlan ONCE per flat_step.
/// Computes tile indices, request indices, KV chunk sizes on CPU.
/// Uploads results to GPU workspace buffers.
/// Must be called before any attention_flat_paged_flashinfer() calls.
void flat_attention_plan(
    const int32_t* h_q_indptr,
    const int32_t* h_seq_lens_k,
    const int* h_block_table,
    int block_table_stride,
    int page_block_size,
    int batch_size,
    int total_q_tokens,
    int Hq, int Hkv, int Hs,
    int32_t* d_page_indptr,
    int32_t* d_page_indices,
    int32_t* d_last_page_len,
    void* d_int_ws,
    void* d_float_ws,
    std::size_t int_ws_size,
    std::size_t float_ws_size,
    void* plan_info,   // opaque: flashinfer::PrefillPlanInfo*
    cudaStream_t stream);

/// Per-layer flat-token attention (GPU-only, no sync).
/// Uses pre-computed plan from flat_attention_plan().
void attention_flat_paged_flashinfer(
    nv_bfloat16* out, float* lse,
    const nv_bfloat16* q,
    const nv_bfloat16* k_pages, const nv_bfloat16* v_pages,
    const int32_t* q_indptr_gpu,
    const int32_t* seq_lens_k,
    const int* block_table, int block_table_stride,
    int page_block_size,
    int batch_size, int total_q_tokens,
    int padded_batch_size_hint,
    int Hq, int Hkv, int Hs,
    int32_t* page_indptr_gpu,
    int32_t* page_indices_gpu,
    int32_t* last_page_len_gpu,
    void* int_ws_gpu,
    void* float_ws_gpu,
    const void* plan_info,   // opaque: const flashinfer::PrefillPlanInfo*
    int32_t* scratch_request_indices,
    int32_t* scratch_qo_tile_indices,
    int32_t* scratch_kv_chunk_size,
    cudaStream_t stream);

/// Per-layer flat-token attention with native FP8 paged KV (FlashInfer).
/// Requires KV pages to be stored in FlashInfer-compatible unit-scale FP8 format.
void attention_flat_paged_flashinfer_fp8(
    nv_bfloat16* out, float* lse,
    const nv_bfloat16* q,
    const __nv_fp8_e4m3* k_pages, const __nv_fp8_e4m3* v_pages,
    const int32_t* q_indptr_gpu,
    const int32_t* seq_lens_k,
    const int* block_table, int block_table_stride,
    int page_block_size,
    int batch_size, int total_q_tokens,
    int padded_batch_size_hint,
    int Hq, int Hkv, int Hs,
    int32_t* page_indptr_gpu,
    int32_t* page_indices_gpu,
    int32_t* last_page_len_gpu,
    void* int_ws_gpu,
    void* float_ws_gpu,
    const void* plan_info,
    int32_t* scratch_request_indices,
    int32_t* scratch_qo_tile_indices,
    int32_t* scratch_kv_chunk_size,
    cudaStream_t stream);

/// Write KV from flat RoPE'd QKV to paged KV-cache.
void kv_cache_store_flat_paged_bf16(
    nv_bfloat16* k_pages, nv_bfloat16* v_pages,
    const nv_bfloat16* qkv_rope,
    const int32_t* token_to_req,
    const int32_t* kv_write_pos,
    const int* block_table, int block_table_stride,
    int page_block_size,
    int total_tokens, int Hq, int Hkv, int Hs,
    cudaStream_t stream);

/// Write KV from flat RoPE'd QKV to paged FP8 KV-cache
/// in FlashInfer-compatible unit-scale format.
void kv_cache_store_flat_paged_fp8(
    __nv_fp8_e4m3* k_pages_fp8, __nv_fp8_e4m3* v_pages_fp8,
    float* k_scales, float* v_scales,
    const nv_bfloat16* qkv_rope,
    const int32_t* token_to_req,
    const int32_t* kv_write_pos,
    const int* block_table, int block_table_stride,
    int page_block_size,
    int total_tokens, int Hq, int Hkv, int Hs,
    cudaStream_t stream);

/// Gather rows from src[total_rows, cols] at indices[batch] → dst[batch, cols].
/// Used by LM head in flat-token mode to extract last-token hidden states.
void gather_rows_bf16(
    void* dst, const void* src,
    const int32_t* indices,
    int batch, int cols,
    cudaStream_t stream);

#endif  // SUROGATE_SRC_KERNELS_ATTENTION_FLAT_PAGED_H
