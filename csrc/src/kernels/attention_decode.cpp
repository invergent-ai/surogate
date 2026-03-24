// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Decode attention using FlashAttention with separate Q / K-cache / V-cache.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cutlass/bfloat16.h>

#define FLASH_NAMESPACE surogate_flash
#include "flash.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "utilities/utils.h"

namespace {

template <int HeadDim>
void run_fwd_causal(surogate_flash::Flash_fwd_params& params, cudaStream_t stream) {
    surogate_flash::run_mha_fwd_<cutlass::bfloat16_t, HeadDim, /*Is_causal=*/true>(params, stream);
}

/// Set common decode params shared by contiguous and paged paths.
/// Q: [B, 1, Hq, Hs] in batch-stride mode (NOT varlen).
/// K/V addressing set separately by caller.
void set_decode_common(surogate_flash::Flash_fwd_params& params,
                       const nv_bfloat16* q, nv_bfloat16* out, float* lse,
                       const int32_t* cu_seqlens_q, const int32_t* seqused_k,
                       int max_seqlen_k, int batch_size,
                       int Hq, int Hkv, int Hs,
                       int q_stride_n) {
    if (q_stride_n <= 0) {
        q_stride_n = Hq * Hs;
    }
    // Q: batch-stride mode [B, Hq, Hs]
    params.q_ptr = const_cast<void*>(static_cast<const void*>(q));
    params.q_batch_stride = q_stride_n;
    params.q_row_stride = q_stride_n;
    params.q_head_stride = Hs;

    // Output: batch-stride mode [B, Hq, Hs]
    params.o_ptr = out;
    params.o_batch_stride = Hq * Hs;
    params.o_row_stride = Hq * Hs;
    params.o_head_stride = Hs;

    params.h = Hq;
    params.h_k = Hkv;
    params.h_h_k_ratio = Hq / Hkv;
    params.b = batch_size;
    params.seqlen_q = 1;
    params.seqlen_k = max_seqlen_k;
    params.d = Hs;
    params.d_rounded = Hs <= 128 ? ((Hs + 31) / 32) * 32 : ((Hs + 63) / 64) * 64;
    params.seqlen_q_rounded = 128;
    params.seqlen_k_rounded = ((max_seqlen_k + 127) / 128) * 128;
    params.total_q = batch_size;

    params.scale_softmax = 1.0f / std::sqrt(static_cast<float>(Hs));
    params.scale_softmax_log2 = params.scale_softmax * static_cast<float>(M_LOG2E);

    // Decode uses seqlen_q=1 per batch item, but FlashAttention forward still
    // goes through varlen kernels when M/N are not tile-aligned. Supplying
    // cu_seqlens_q keeps batch row mapping correct for B > 1 in decode mode.
    params.cu_seqlens_q = const_cast<int*>(reinterpret_cast<const int*>(cu_seqlens_q));
    params.cu_seqlens_k = nullptr;
    params.seqused_k = const_cast<int*>(reinterpret_cast<const int*>(seqused_k));
    params.is_seqlens_k_cumulative = false;

    params.softmax_lse_ptr = lse;
    params.is_bf16 = true;
    params.is_causal = true;
    params.p_dropout = 1.0f;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;
    params.p_dropout_in_uint8_t = 255;
    params.window_size_left = -1;
    params.window_size_right = 0;
    params.softcap = 0.0f;
    params.unpadded_lse = false;
    params.num_splits = 0;
}

void dispatch_fwd(surogate_flash::Flash_fwd_params& params, int Hs, cudaStream_t stream) {
    if      (Hs <= 64)  run_fwd_causal<64>(params, stream);
    else if (Hs <= 96)  run_fwd_causal<96>(params, stream);
    else if (Hs <= 128) run_fwd_causal<128>(params, stream);
    else if (Hs <= 256) run_fwd_causal<256>(params, stream);
    else throw std::runtime_error("attention_decode: head_size > 256 not supported");
    CUDA_CHECK(cudaGetLastError());
}

}  // anonymous namespace

// ============================================================================
// Contiguous KV-cache decode attention
// ============================================================================
void attention_decode_flash(
        nv_bfloat16* out, float* lse,
        const nv_bfloat16* q,
        const nv_bfloat16* k_cache, const nv_bfloat16* v_cache,
        const int32_t* cu_seqlens_q, const int32_t* seqused_k,
        int max_seqlen_k, int kv_stride_seqlen,
        int batch_size, int Hq, int Hkv, int Hs,
        cudaStream_t stream,
        int q_stride_n) {

    surogate_flash::Flash_fwd_params params;
    std::memset(&params, 0, sizeof(params));

    set_decode_common(params, q, out, lse, cu_seqlens_q, seqused_k,
                      max_seqlen_k, batch_size, Hq, Hkv, Hs, q_stride_n);

    // K/V: padded batch mode [B, max_seq_len_k, Hkv, Hs]
    params.k_ptr = const_cast<void*>(static_cast<const void*>(k_cache));
    params.k_batch_stride = static_cast<int64_t>(kv_stride_seqlen) * Hkv * Hs;
    params.k_row_stride = Hkv * Hs;
    params.k_head_stride = Hs;

    params.v_ptr = const_cast<void*>(static_cast<const void*>(v_cache));
    params.v_batch_stride = static_cast<int64_t>(kv_stride_seqlen) * Hkv * Hs;
    params.v_row_stride = Hkv * Hs;
    params.v_head_stride = Hs;

    dispatch_fwd(params, Hs, stream);
}

// ============================================================================
// Paged KV-cache decode attention
// ============================================================================
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
        int q_stride_n) {

    surogate_flash::Flash_fwd_params params;
    std::memset(&params, 0, sizeof(params));

    set_decode_common(params, q, out, lse, cu_seqlens_q, seqused_k,
                      max_seqlen_k, batch_size, Hq, Hkv, Hs, q_stride_n);

    // K/V: paged mode — batch_stride = page stride
    const int page_elems = page_block_size * Hkv * Hs;
    params.k_ptr = const_cast<void*>(static_cast<const void*>(k_pages));
    params.k_batch_stride = static_cast<int64_t>(page_elems);
    params.k_row_stride = Hkv * Hs;
    params.k_head_stride = Hs;

    params.v_ptr = const_cast<void*>(static_cast<const void*>(v_pages));
    params.v_batch_stride = static_cast<int64_t>(page_elems);
    params.v_row_stride = Hkv * Hs;
    params.v_head_stride = Hs;

    params.block_table = const_cast<int*>(block_table);
    params.block_table_batch_stride = block_table_stride;
    params.page_block_size = page_block_size;

    dispatch_fwd(params, Hs, stream);
}
