// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Thin wrapper around DAO-AiLab Flash Attention varlen kernels for
// document-level attention masking in packed sequences.

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

void set_common_params(surogate_flash::Flash_fwd_params& params,
                       const nv_bfloat16* qkv_ptr, nv_bfloat16* out_ptr, float* lse_ptr,
                       int B_ragged, int max_seqlen, int total_q,
                       int Hq, int Hkv, int HS,
                       const int32_t* cu_seqlens_gpu) {
    const int H = Hq + 2 * Hkv;
    std::memset(&params, 0, sizeof(params));

    // Q/K/V from interleaved (B, T, H, HS) buffer — use strides, no copy
    params.q_ptr = const_cast<void*>(static_cast<const void*>(qkv_ptr));
    params.k_ptr = const_cast<void*>(static_cast<const void*>(qkv_ptr + Hq * HS));
    params.v_ptr = const_cast<void*>(static_cast<const void*>(qkv_ptr + (Hq + Hkv) * HS));
    params.o_ptr = out_ptr;

    // Strides: interleaved QKV has row_stride = H*HS, head_stride = HS
    params.q_row_stride = H * HS;
    params.k_row_stride = H * HS;
    params.v_row_stride = H * HS;
    params.q_head_stride = HS;
    params.k_head_stride = HS;
    params.v_head_stride = HS;
    params.q_batch_stride = 0;  // varlen mode — not used
    params.k_batch_stride = 0;
    params.v_batch_stride = 0;

    // Output: (total_q, Hq, HS) contiguous
    params.o_row_stride = Hq * HS;
    params.o_head_stride = HS;
    params.o_batch_stride = 0;

    params.h = Hq;
    params.h_k = Hkv;
    params.h_h_k_ratio = Hq / Hkv;
    params.b = B_ragged;
    params.seqlen_q = max_seqlen;
    params.seqlen_k = max_seqlen;
    params.d = HS;
    params.d_rounded = HS <= 128 ? ((HS + 31) / 32) * 32 : ((HS + 63) / 64) * 64;
    params.seqlen_q_rounded = ((max_seqlen + 127) / 128) * 128;
    params.seqlen_k_rounded = ((max_seqlen + 127) / 128) * 128;
    params.total_q = total_q;

    params.scale_softmax = 1.0f / std::sqrt(static_cast<float>(HS));
    params.scale_softmax_log2 = params.scale_softmax * static_cast<float>(M_LOG2E);

    // Variable-length sequence support
    params.cu_seqlens_q = const_cast<int*>(reinterpret_cast<const int*>(cu_seqlens_gpu));
    params.cu_seqlens_k = const_cast<int*>(reinterpret_cast<const int*>(cu_seqlens_gpu));
    params.is_seqlens_k_cumulative = true;

    params.softmax_lse_ptr = lse_ptr;
    params.is_bf16 = true;
    params.is_causal = true;
    // Flash-attn internal convention: p_dropout = probability of KEEPING (not dropping).
    // p_dropout = 1.0 → keep everything → DROPOUT_SWITCH(1.0 < 1.0) = false → non-dropout kernel.
    // p_dropout < 1.0 triggers the dropout kernel which writes to params.rng_state (nullptr for us).
    params.p_dropout = 1.0f;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;
    params.p_dropout_in_uint8_t = 255;  // floor(1.0 * 255) — not used when Is_dropout=false
    params.window_size_left = -1;   // no window limit left
    params.window_size_right = 0;   // causal: no right context
    params.softcap = 0.0f;
    params.unpadded_lse = true;     // LSE in (Hq, total_q) format
    params.num_splits = 0;          // unused — run_mha_fwd_ uses standard (non-split) kernel
}

template <int HeadDim>
void run_fwd(surogate_flash::Flash_fwd_params& params, cudaStream_t stream) {
    surogate_flash::run_mha_fwd_<cutlass::bfloat16_t, HeadDim, /*Is_causal=*/true>(params, stream);
}

template <int HeadDim>
void run_bwd(surogate_flash::Flash_bwd_params& params, cudaStream_t stream) {
    surogate_flash::run_mha_bwd_<cutlass::bfloat16_t, HeadDim, /*Is_causal=*/true>(params, stream);
}

}  // anonymous namespace

void attention_forward_flash_varlen(
        nv_bfloat16* out, float* lse, const nv_bfloat16* qkv,
        const int32_t* cu_seqlens_gpu,
        int B_ragged, int max_seqlen, int total_q,
        int Hq, int Hkv, int HS, cudaStream_t stream) {
    surogate_flash::Flash_fwd_params params;
    set_common_params(params, qkv, out, lse, B_ragged, max_seqlen, total_q,
                      Hq, Hkv, HS, cu_seqlens_gpu);

    if      (HS <= 64)  run_fwd<64>(params, stream);
    else if (HS <= 96)  run_fwd<96>(params, stream);
    else if (HS <= 128) run_fwd<128>(params, stream);
    else if (HS <= 256) run_fwd<256>(params, stream);
    else throw std::runtime_error("Flash Attention varlen: head_size > 256 not supported");

    CUDA_CHECK(cudaGetLastError());
}

// Declared in flash_attn_scatter.cu
void reduce_scatter_dkv(
        nv_bfloat16* dqkv,
        const nv_bfloat16* dk_expanded,
        const nv_bfloat16* dv_expanded,
        int total_q, int Hq, int Hkv, int HS,
        cudaStream_t stream);

void attention_backward_flash_varlen(
        nv_bfloat16* dqkv, const float* lse,
        const nv_bfloat16* out, const nv_bfloat16* dout, const nv_bfloat16* qkv,
        const int32_t* cu_seqlens_gpu,
        float* dq_accum, float* dsoftmax_sum,
        nv_bfloat16* dk_expanded, nv_bfloat16* dv_expanded,
        int B_ragged, int max_seqlen, int total_q,
        int Hq, int Hkv, int HS, bool deterministic, cudaStream_t stream) {
    surogate_flash::Flash_bwd_params params;
    // IMPORTANT: zero ALL fields first. set_common_params takes Flash_fwd_params&,
    // so its memset only zeros sizeof(Flash_fwd_params) bytes, leaving backward-
    // specific fields (dk_accum_ptr, dv_accum_ptr, etc.) as stack garbage.
    std::memset(&params, 0, sizeof(params));
    // Set forward-compatible common params (redundantly zeros fwd portion, harmless)
    set_common_params(params, qkv, const_cast<nv_bfloat16*>(out),
                      const_cast<float*>(lse),
                      B_ragged, max_seqlen, total_q, Hq, Hkv, HS, cu_seqlens_gpu);

    const int H = Hq + 2 * Hkv;
    const bool is_gqa = (Hq != Hkv);

    // Gradient of output: (total_q, Hq, HS) contiguous (same layout as forward output)
    params.do_ptr = const_cast<void*>(static_cast<const void*>(dout));
    params.do_row_stride = Hq * HS;
    params.do_head_stride = HS;
    params.do_batch_stride = 0;

    // dQ: always write directly to Q section of interleaved dqkv.
    // convert_dQ kernel respects strides, and Q has Hq head slots — fits fine.
    params.dq_ptr = dqkv;
    params.dq_row_stride = H * HS;
    params.dq_head_stride = HS;
    params.dq_batch_stride = 0;

    if (is_gqa) {
        // GQA: flash backward writes dK/dV with Hq head indices (bidh = 0..Hq-1),
        // but the interleaved buffer only has Hkv head slots for K and V.
        // Use separate expanded buffers: (total_q, Hq, HS) contiguous.
        // After backward, reduce Hq→Hkv and scatter into interleaved K/V sections.
        params.dk_ptr = dk_expanded;
        params.dv_ptr = dv_expanded;
        params.dk_row_stride = Hq * HS;
        params.dv_row_stride = Hq * HS;
        params.dk_head_stride = HS;
        params.dv_head_stride = HS;
        params.dk_batch_stride = 0;
        params.dv_batch_stride = 0;
    } else {
        // MHA (Hq == Hkv): write dK/dV directly to interleaved buffer.
        // The kernel writes with bidh = 0..Hkv-1, exactly fitting the Hkv slots.
        params.dk_ptr = dqkv + Hq * HS;
        params.dv_ptr = dqkv + (Hq + Hkv) * HS;
        params.dk_row_stride = H * HS;
        params.dv_row_stride = H * HS;
        params.dk_head_stride = HS;
        params.dv_head_stride = HS;
        params.dk_batch_stride = 0;
        params.dv_batch_stride = 0;
    }

    // Temporary buffers for backward
    params.dq_accum_ptr = dq_accum;
    params.dsoftmax_sum = dsoftmax_sum;
    params.deterministic = deterministic;

    const int HS_rounded = HS <= 128 ? ((HS + 31) / 32) * 32 : ((HS + 63) / 64) * 64;
    if (deterministic) {
        params.dq_accum_split_stride =
            static_cast<int64_t>(total_q + 128 * B_ragged) * Hq * HS_rounded;
    }

    if      (HS <= 64)  run_bwd<64>(params, stream);
    else if (HS <= 96)  run_bwd<96>(params, stream);
    else if (HS <= 128) run_bwd<128>(params, stream);
    else if (HS <= 256) run_bwd<256>(params, stream);
    else throw std::runtime_error("Flash Attention varlen backward: head_size > 256 not supported");

    CUDA_CHECK(cudaGetLastError());

    // GQA post-processing: reduce expanded dK/dV (Hq heads) → Hkv heads
    // and scatter into K/V sections of interleaved dqkv.
    if (is_gqa) {
        reduce_scatter_dkv(dqkv, dk_expanded, dv_expanded,
                           total_q, Hq, Hkv, HS, stream);
        CUDA_CHECK(cudaGetLastError());
    }
}
