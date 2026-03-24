// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Decode attention kernels for autoregressive generation with KV-cache.
//
// These wrappers call the FlashAttention varlen kernel with:
//   Q: [batch_size, 1, Hq, Hs] (single new query token per sequence)
//   K: [batch_size, seq_len, Hkv, Hs] (from KV-cache, seq_len varies per sequence)
//   V: [batch_size, seq_len, Hkv, Hs] (from KV-cache)
// Each sequence can have a different KV length via cu_seqlens_k.

#ifndef SUROGATE_SRC_KERNELS_ATTENTION_DECODE_H
#define SUROGATE_SRC_KERNELS_ATTENTION_DECODE_H

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

/// Decode attention: Q attends to KV-cache.
///
/// @param out          Output [batch_size, 1, Hq, Hs] (BF16, contiguous)
/// @param lse          Log-sum-exp [Hq, batch_size] (FP32, for potential future backward)
/// @param q            Query [batch_size, 1, Hq, Hs] (BF16, from QKV projection after RoPE)
///                     Stride: q_row_stride = Hq * Hs (Q heads only, not interleaved QKV)
/// @param k_cache      K cache [batch_size, max_seq_len, Hkv, Hs] (BF16)
/// @param v_cache      V cache [batch_size, max_seq_len, Hkv, Hs] (BF16)
/// @param cu_seqlens_q Cumulative Q lengths [batch_size + 1] (each sequence has 1 query token)
///                     Values: [0, 1, 2, ..., batch_size]
/// @param seqused_k    Per-sequence K lengths [batch_size] (NOT cumulative).
///                     Passed to Flash Attention as non-cumulative seqlens_k.
/// @param max_seqlen_k Maximum KV sequence length across the batch
/// @param kv_stride_seqlen Physical KV stride (sequence capacity) per batch row.
///                         For contiguous caches this is the cache capacity
///                         max_seq_len, not max_seqlen_k.
/// @param batch_size   Number of sequences
/// @param Hq           Number of query heads
/// @param Hkv          Number of key/value heads (GQA: Hkv < Hq)
/// @param Hs           Head dimension
/// @param stream       CUDA stream
void attention_decode_flash(
    nv_bfloat16* out, float* lse,
    const nv_bfloat16* q,
    const nv_bfloat16* k_cache, const nv_bfloat16* v_cache,
    const int32_t* cu_seqlens_q, const int32_t* seqused_k,
    int max_seqlen_k, int kv_stride_seqlen,
    int batch_size, int Hq, int Hkv, int Hs,
    cudaStream_t stream,
    int q_stride_n = -1);

/// Append new K/V tokens to the KV-cache from a RoPE'd QKV buffer.
///
/// Extracts K and V heads from interleaved QKV output [batch_size, 1, (Hq+2*Hkv), Hs],
/// and writes them to the appropriate positions in the KV-cache.
///
/// @param k_cache      K cache [batch_size, max_seq_len, Hkv, Hs] (BF16, writable)
/// @param v_cache      V cache [batch_size, max_seq_len, Hkv, Hs] (BF16, writable)
/// @param qkv_rope     QKV after RoPE [batch_size, 1, (Hq+2*Hkv), Hs] (BF16)
/// @param seq_lens     Current sequence length per batch item [batch_size] (host)
///                     K/V are written at position seq_lens[i] for each batch item i.
/// @param batch_size   Number of sequences
/// @param max_seq_len  Maximum sequence length capacity of the KV-cache
/// @param Hq           Number of query heads
/// @param Hkv          Number of key/value heads
/// @param Hs           Head dimension
/// @param stream       CUDA stream
void kv_cache_append_bf16(
    nv_bfloat16* k_cache, nv_bfloat16* v_cache,
    const nv_bfloat16* qkv_rope,
    const int* seq_lens_gpu,
    int batch_size, int max_seq_len,
    int Hq, int Hkv, int Hs,
    cudaStream_t stream);

/// Fill cu_seqlens_q and seqused_k for decode attention.
/// cu_seqlens_q: [0, 1, 2, ..., batch_size]  (each seq has 1 query token)
/// seqused_k:    [s0+1, s1+1, ...]            (per-sequence K lengths after append)
void fill_decode_cu_seqlens(
    int32_t* cu_seqlens_q,
    int32_t* seqused_k,
    const int* seq_lens_gpu,
    int batch_size,
    cudaStream_t stream);

/// Fill int32 tensor with iota values [0, 1, ..., n-1].
void fill_iota_i32(
    int32_t* out,
    int n,
    cudaStream_t stream);

/// Append K/V to paged KV-cache from interleaved QKV (paged mode).
///
/// Like kv_cache_append_bf16, but writes to the correct page via block_table lookup.
///
/// @param k_pages       K page pool for this layer: [total_pages, page_block_size, Hkv, Hs]
/// @param v_pages       V page pool for this layer
/// @param qkv_rope      QKV after RoPE [batch_size, 1, (Hq+2*Hkv), Hs]
/// @param seq_lens_gpu  Current seq length per batch item [batch_size] (position to write at)
/// @param block_table   Block table [batch_size, max_pages_per_seq] (GPU)
/// @param block_table_stride  = max_pages_per_seq
/// @param page_block_size     Tokens per page
/// @param batch_size    Number of sequences
/// @param Hq, Hkv, Hs  Head counts and dimension
/// @param stream        CUDA stream
void kv_cache_append_paged_bf16(
    nv_bfloat16* k_pages, nv_bfloat16* v_pages,
    const nv_bfloat16* qkv_rope,
    const int* seq_lens_gpu,
    const int* block_table, int block_table_stride,
    int page_block_size,
    int batch_size, int Hq, int Hkv, int Hs,
    cudaStream_t stream);

/// Decode attention with paged KV-cache (Flash Attention paged mode).
///
/// @param out           Output [batch_size, Hq, Hs]
/// @param lse           Log-sum-exp [Hq, batch_size]
/// @param q             Query [batch_size, Hq, Hs]
/// @param k_pages       K page pool for this layer: [total_pages, page_block_size, Hkv, Hs]
/// @param v_pages       V page pool for this layer
/// @param cu_seqlens_q  Cumulative Q lengths [batch_size + 1]
/// @param seqused_k     Per-sequence K lengths [batch_size]
/// @param block_table   Block table [batch_size, max_pages_per_seq] (GPU)
/// @param block_table_stride  = max_pages_per_seq
/// @param page_block_size     Tokens per page
/// @param max_seqlen_k  Maximum KV length across the batch
/// @param batch_size, Hq, Hkv, Hs
/// @param stream        CUDA stream
void attention_decode_flash_paged(
    nv_bfloat16* out, float* lse,
    const nv_bfloat16* q,
    const nv_bfloat16* k_pages, const nv_bfloat16* v_pages,
    const int32_t* cu_seqlens_q, const int32_t* seqused_k,
    const int* block_table, int block_table_stride,
    int page_block_size,
    int max_seqlen_k,
    int batch_size, int Hq, int Hkv, int Hs,
    cudaStream_t stream,
    int q_stride_n = -1);

/// Decode attention with paged KV-cache using FlashInfer kernels.
///
/// This path expects BF16 paged KV and dynamically builds compact FlashInfer
/// metadata (indptr/indices/last_page_len + request indices) from:
/// - seqused_k (current per-sequence K lengths),
/// - dense block_table [batch_size, block_table_stride].
///
/// scratch_* buffers must be preallocated on device:
/// - scratch_page_counts: [batch_size]
/// - scratch_indptr:      [batch_size + 1]
/// - scratch_last_page_len:[batch_size]
/// - scratch_indices:     [batch_size * block_table_stride] (max possible)
/// - scratch_request_indices: [batch_size]
/// - scratch_kv_tile_indices: [batch_size]
/// - scratch_kv_chunk_size: [1]
void attention_decode_flashinfer_paged(
    nv_bfloat16* out, float* lse,
    const nv_bfloat16* q,
    const nv_bfloat16* k_pages, const nv_bfloat16* v_pages,
    const int32_t* seqused_k,
    const int* block_table, int block_table_stride,
    int page_block_size,
    int batch_size, int Hq, int Hkv, int Hs,
    int32_t* scratch_page_counts,
    int32_t* scratch_indptr,
    int32_t* scratch_last_page_len,
    int32_t* scratch_indices,
    int32_t* scratch_request_indices,
    int32_t* scratch_kv_tile_indices,
    int32_t* scratch_kv_chunk_size,
    bool enable_pdl,
    cudaStream_t stream,
    int q_stride_n = -1);

/// Decode attention with paged FP8 KV-cache using FlashInfer (native FP8).
///
/// Unlike the BF16 FlashInfer path, this reads FP8 E4M3 KV pages directly —
/// no intermediate BF16 dequant buffers.  FlashInfer dequants on-the-fly in
/// registers during the attention computation.
///
/// scratch_* buffers: same as attention_decode_flashinfer_paged.
void attention_decode_flashinfer_paged_fp8(
    nv_bfloat16* out, float* lse,
    const nv_bfloat16* q,
    const __nv_fp8_e4m3* k_pages, const __nv_fp8_e4m3* v_pages,
    const int32_t* seqused_k,
    const int* block_table, int block_table_stride,
    int page_block_size,
    int batch_size, int Hq, int Hkv, int Hs,
    int32_t* scratch_page_counts,
    int32_t* scratch_indptr,
    int32_t* scratch_last_page_len,
    int32_t* scratch_indices,
    int32_t* scratch_request_indices,
    int32_t* scratch_kv_tile_indices,
    int32_t* scratch_kv_chunk_size,
    bool enable_pdl,
    cudaStream_t stream,
    int q_stride_n = -1);

/// Mask finished sequences: replace token IDs with 0 for finished sequences.
/// finished_gpu[i] != 0 → token_ids[i] = 0.
/// This prevents finished sequences from generating meaningful embeddings.
void mask_finished_tokens(
    int32_t* token_ids,
    const int* finished_gpu,
    int batch_size,
    cudaStream_t stream);

/// Update decode sequence state after sampling one token.
///
/// For each active sequence (finished_gpu[i] == 0):
/// - seq_lens_gpu[i] += 1
/// - completion_lens_gpu[i] += 1
/// - if sampled_tokens[i] == eos_token_id, mark finished_gpu[i] = 1
///
/// Finished sequences are left unchanged.
void update_generation_state(
    const int32_t* sampled_tokens,
    int* finished_gpu,
    int* seq_lens_gpu,
    int32_t* completion_lens_gpu,
    int32_t eos_token_id,
    int batch_size,
    cudaStream_t stream);

/// Count active (unfinished) sequences.
/// Writes the count to active_count_gpu[0].
void count_active_sequences(
    const int* finished_gpu,
    int* active_count_gpu,
    int batch_size,
    cudaStream_t stream);

/// Bulk-store K/V from interleaved QKV [B, T, (Hq+2*Hkv), Hs] into contiguous KV-cache.
/// Used during prefill to populate the entire prompt's KV in one shot.
///
/// @param k_cache      K cache for this layer: [B, max_seq_len, Hkv, Hs] (BF16)
/// @param v_cache      V cache for this layer
/// @param qkv_rope     QKV after RoPE [B, T, (Hq+2*Hkv), Hs] (BF16)
/// @param B, T         Batch size and sequence length of the QKV
/// @param max_seq_len  Max sequence length capacity of the cache
/// @param Hq, Hkv, Hs Head dimensions
/// @param start_pos    Absolute token offset for chunked prefill writes
/// @param stream       CUDA stream
void kv_cache_store_bf16(
    nv_bfloat16* k_cache, nv_bfloat16* v_cache,
    const nv_bfloat16* qkv_rope,
    int B, int T, int max_seq_len,
    int Hq, int Hkv, int Hs,
    int start_pos,
    cudaStream_t stream);

/// Bulk-store K/V into paged KV-cache during prefill.
void kv_cache_store_paged_bf16(
    nv_bfloat16* k_pages, nv_bfloat16* v_pages,
    const nv_bfloat16* qkv_rope,
    const int* block_table, int block_table_stride,
    int page_block_size,
    int B, int T,
    int Hq, int Hkv, int Hs,
    int start_pos,
    cudaStream_t stream);

// ============================================================================
// FP8 E4M3 KV-cache kernels (SM89+)
// ============================================================================

/// Bulk-store all T positions of K/V from BF16 QKV into paged FP8 KV-cache.
/// Stores FlashInfer-compatible unit-scale FP8 pages and writes scale=1.0f.
void kv_cache_store_paged_fp8(
    __nv_fp8_e4m3* k_pages_fp8, __nv_fp8_e4m3* v_pages_fp8,
    float* k_scales, float* v_scales,
    const nv_bfloat16* qkv_rope,
    const int* block_table, int block_table_stride,
    int page_block_size,
    int batch_size, int T,
    int Hq, int Hkv, int Hs,
    int start_pos,
    cudaStream_t stream);

/// Append K/V to FP8 KV-cache from BF16 QKV, quantizing on-the-fly.
///
/// Extracts K and V from interleaved BF16 QKV [B, 1, (Hq+2*Hkv), Hs],
/// quantizes each head vector to FP8 E4M3 with a per-head scale, and writes
/// to the contiguous FP8 KV-cache.
///
/// @param k_cache_fp8   K cache [B, max_seq_len, Hkv, Hs] (FP8 E4M3)
/// @param v_cache_fp8   V cache [B, max_seq_len, Hkv, Hs] (FP8 E4M3)
/// @param k_scales      K per-head scales [B, max_seq_len, Hkv] (FP32, inv scale for dequant)
/// @param v_scales      V per-head scales [B, max_seq_len, Hkv] (FP32)
/// @param qkv_rope      BF16 QKV after RoPE [B, 1, (Hq+2*Hkv), Hs]
/// @param seq_lens_gpu  Current seq length per batch [B] (position to write at)
/// @param batch_size, max_seq_len, Hq, Hkv, Hs
/// @param stream        CUDA stream
void kv_cache_append_fp8(
    __nv_fp8_e4m3* k_cache_fp8, __nv_fp8_e4m3* v_cache_fp8,
    float* k_scales, float* v_scales,
    const nv_bfloat16* qkv_rope,
    const int* seq_lens_gpu,
    int batch_size, int max_seq_len,
    int Hq, int Hkv, int Hs,
    cudaStream_t stream);

/// Append K/V to paged FP8 KV-cache in FlashInfer-compatible unit-scale format.
void kv_cache_append_paged_fp8(
    __nv_fp8_e4m3* k_pages_fp8, __nv_fp8_e4m3* v_pages_fp8,
    float* k_scales, float* v_scales,
    const nv_bfloat16* qkv_rope,
    const int* seq_lens_gpu,
    const int* block_table, int block_table_stride,
    int page_block_size,
    int batch_size, int Hq, int Hkv, int Hs,
    cudaStream_t stream);

/// Dequantize a layer's KV-cache from FP8 E4M3 → BF16.
///
/// Used before Flash Attention (which only accepts BF16).
/// Only dequantizes positions [0, seq_len) per batch item.
///
/// @param k_out_bf16    Output K [B, max_seq_len, Hkv, Hs] (BF16)
/// @param v_out_bf16    Output V [B, max_seq_len, Hkv, Hs] (BF16)
/// @param k_cache_fp8   Input K [B, max_seq_len, Hkv, Hs] (FP8 E4M3)
/// @param v_cache_fp8   Input V [B, max_seq_len, Hkv, Hs] (FP8 E4M3)
/// @param k_scales      Per-head scales [B, max_seq_len, Hkv] (FP32)
/// @param v_scales      Per-head scales [B, max_seq_len, Hkv] (FP32)
/// @param seqused_k_gpu Per-batch K lengths [B] for the current decode step
///                      (typically seq_lens + 1 after KV append).
/// @param batch_size, max_seq_len, Hkv, Hs
/// @param stream        CUDA stream
void kv_cache_dequant_fp8_to_bf16(
    nv_bfloat16* k_out_bf16, nv_bfloat16* v_out_bf16,
    const __nv_fp8_e4m3* k_cache_fp8, const __nv_fp8_e4m3* v_cache_fp8,
    const float* k_scales, const float* v_scales,
    const int* seqused_k_gpu,
    int batch_size, int max_seq_len,
    int Hkv, int Hs,
    cudaStream_t stream);

/// Dequantize a paged layer's KV-cache from FP8 E4M3 → BF16.
void kv_cache_dequant_paged_fp8_to_bf16(
    nv_bfloat16* k_out_bf16, nv_bfloat16* v_out_bf16,
    const __nv_fp8_e4m3* k_pages_fp8, const __nv_fp8_e4m3* v_pages_fp8,
    const float* k_scales, const float* v_scales,
    const int* seqused_k_gpu,
    const int* block_table, int block_table_stride,
    int page_block_size,
    int batch_size, int max_seq_len,
    int Hkv, int Hs,
    cudaStream_t stream);

/// Broadcast prefix KV from a source batch slot to N-1 destination slots.
///
/// After prefilling one sequence slot for a prompt, this copies the KV-cache
/// entries [0, prefix_len) from the source slot to each destination slot.
/// Used for GRPO multi-completion: prefill once, broadcast to N sequence slots.
///
/// @param k_cache       K cache for one layer: [batch_size, max_seq_len, Hkv, Hs] (BF16)
/// @param v_cache       V cache for one layer: [batch_size, max_seq_len, Hkv, Hs] (BF16)
/// @param src_slot      Source batch index (the slot that was prefilled)
/// @param dst_slots     Array of destination batch indices [num_copies]
/// @param num_copies    Number of destination slots (N-1)
/// @param prefix_len    Number of KV positions to copy [0, prefix_len)
/// @param max_seq_len   Maximum sequence length capacity of the KV-cache
/// @param Hkv           Number of KV heads
/// @param Hs            Head dimension
/// @param stream        CUDA stream
void kv_cache_broadcast_prefix(
    nv_bfloat16* k_cache, nv_bfloat16* v_cache,
    int src_slot, const int* dst_slots, int num_copies,
    int prefix_len, int max_seq_len,
    int Hkv, int Hs,
    cudaStream_t stream);

#endif  // SUROGATE_SRC_KERNELS_ATTENTION_DECODE_H
