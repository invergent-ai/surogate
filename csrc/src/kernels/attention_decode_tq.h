// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file attention_decode_tq.h
 * @brief Public API for TurboQuant 3.5-bit fused decode attention + KV cache ops.
 *
 * Fused kernel: dequantizes TQ packed KV in registers, computes attention,
 * never materializes BF16. 2.13x less HBM traffic than FP8 → faster decode.
 *
 * KV signs layout: [num_kv_heads, 4, head_dim/32] uint32s
 *   Slot 0: d1 for K (main rotation signs)
 *   Slot 1: d2 for K (QJL rotation signs)
 *   Slot 2: d1 for V
 *   Slot 3: d2 for V
 *
 * Page pool layout: [total_pages, page_block_size, num_kv_heads, TQ_PACKED]
 *   where TQ_PACKED = 7*head_dim/16 + 4 bytes per vector.
 */

#ifndef SUROGATE_SRC_KERNELS_ATTENTION_DECODE_TQ_H
#define SUROGATE_SRC_KERNELS_ATTENTION_DECODE_TQ_H

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

/// Packed bytes per TQ vector.
constexpr int tq_packed_size(int head_dim) { return 7 * head_dim / 16 + 4; }

/// Fused paged decode attention from TurboQuant 3.5-bit KV cache.
///
/// Performs Q @ K^T → softmax → O in a single kernel, dequantizing TQ packed
/// KV data in registers. One warp per (batch, kv_head), GQA handled via threadIdx.y.
///
/// @param out            [batch, Hq, Hs] BF16 output
/// @param q              [batch, Hq, Hs] BF16 queries (after RoPE)
/// @param k_pages        TQ packed K page pool for this layer
/// @param v_pages        TQ packed V page pool for this layer
/// @param kv_signs       [Hkv, 4, Hs/32] random rotation signs
/// @param seq_lens       [batch] current K sequence lengths
/// @param block_table    [batch, block_table_stride] virtual→physical page mapping
/// @param q_stride_n     Stride between Q vectors (-1 = Hq*Hs)
/// @param scratch_O     Pre-allocated [B, Hq, splits, Hs] float (nullptr = auto-alloc)
/// @param scratch_ml    Pre-allocated [B, Hq, splits, 2] float  (nullptr = auto-alloc)
/// @param num_sms       SM count for split heuristic (0 = auto-detect)
void attention_decode_paged_tq(
    nv_bfloat16* out,
    const nv_bfloat16* q,
    const uint8_t* k_pages,
    const uint8_t* v_pages,
    const uint32_t* kv_signs,
    const int32_t* seq_lens,
    const int* block_table, int block_table_stride,
    int page_block_size,
    int batch_size, int Hq, int Hkv, int Hs,
    cudaStream_t stream, int q_stride_n = -1,
    float* scratch_O = nullptr, float* scratch_ml = nullptr, int num_sms = 0);

/// Append single-token K/V to TQ paged cache (decode step).
///
/// Extracts K and V from interleaved QKV after RoPE, TurboQuant-compresses,
/// and writes to the page pool at the current sequence position.
///
/// @param k_pages        TQ packed K page pool for this layer
/// @param v_pages        TQ packed V page pool for this layer
/// @param qkv_rope       [batch, 1, Hq+2*Hkv, Hs] BF16 after RoPE
/// @param kv_signs       [Hkv, 4, Hs/32] random rotation signs
/// @param seq_lens_gpu   [batch] current lengths (write position)
/// @param block_table    [batch, block_table_stride]
void kv_cache_append_paged_tq(
    uint8_t* k_pages, uint8_t* v_pages,
    const nv_bfloat16* qkv_rope,
    const uint32_t* kv_signs,
    const int* seq_lens_gpu,
    const int* block_table, int block_table_stride,
    int page_block_size,
    int batch_size, int Hq, int Hkv, int Hs,
    cudaStream_t stream);

/// Generate random signs for all KV heads on host, caller copies to device.
/// Output: [num_kv_heads, 4, head_dim/32] uint32s
///   (d1_k, d2_k, d1_v, d2_v per head, each head_dim/32 words)
void tq_generate_kv_signs_host(
    uint32_t* signs_out,
    int num_kv_heads, int head_dim,
    uint64_t seed);

#endif  // SUROGATE_SRC_KERNELS_ATTENTION_DECODE_TQ_H
