#include "runtime/dsl/compiled_ops.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "kernels/attention_decode.h"
#include "kernels/attention_flat_paged.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace {

bool env_flag_enabled(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) {
        return false;
    }
    return std::strcmp(v, "0") != 0 &&
           std::strcmp(v, "false") != 0 &&
           std::strcmp(v, "False") != 0 &&
           std::strcmp(v, "FALSE") != 0;
}

bool flashinfer_decode_pdl_enabled() {
    static int cached = -1;
    if (cached >= 0) {
        return cached != 0;
    }
    // PDL can be enabled explicitly; default is off for stability.
    cached = env_flag_enabled("SUROGATE_FLASHINFER_DECODE_PDL") ? 1 : 0;
    return cached != 0;
}

bool flashinfer_decode_head_supported(const int head_dim) {
    return head_dim == 64 || head_dim == 96 || head_dim == 128 || head_dim == 256;
}

bool flashinfer_decode_device_supported(const cudaDeviceProp& prop) {
    const int sm = prop.major * 10 + prop.minor;
    return sm >= 80;
}

void log_flashinfer_decode_enabled_once(const cudaDeviceProp& prop, int head_dim) {
    static std::atomic<bool> printed{false};
    bool expected = false;
    if (printed.compare_exchange_strong(expected, true)) {
        const int sm = prop.major * 10 + prop.minor;
        std::fprintf(
            stderr,
            "FlashInfer paged decode enabled (SM%d, head_dim=%d).\n",
            sm,
            head_dim);
        std::fflush(stderr);
    }
}

void log_flashinfer_decode_head_fallback_once(const int head_dim) {
    static std::atomic<bool> printed{false};
    bool expected = false;
    if (printed.compare_exchange_strong(expected, true)) {
        std::fprintf(
            stderr,
            "[surogate] FlashInfer decode requested, but head_dim=%d is unsupported; using FlashAttention decode.\n",
            head_dim);
        std::fflush(stderr);
    }
}

}  // namespace

namespace dsl {

void CompiledExecutor::dispatch_flash_attention(const CompiledOp& op) {
    Tensor& qkv = resolve_tensor(op.inputs[0]);
    Tensor* sinks = nullptr;
    if (op.inputs.size() > 1 && !op.inputs[1].name.empty()) {
        sinks = &resolve_tensor(op.inputs[1]);
    }
    Tensor& out = ensure_output_tensor(op.outputs[0]);
    Tensor& lse = ensure_output_tensor(op.outputs[1]);

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    const int H  = Hq + 2 * Hkv;

    int layer_idx = op.attrs.layer_idx;
    if (layer_idx < 0 && !op.inputs.empty() && op.inputs[0].layer_idx >= 0) {
        layer_idx = op.inputs[0].layer_idx;
    }
    if (layer_idx < 0 && !op.inputs.empty()) {
        std::string field;
        parse_block_param(op.inputs[0].name, layer_idx, field);
    }

    // -----------------------------------------------------------------------
    // Flat-token mode: mixed prefill+decode via FlashInfer paged prefill.
    // Plan was pre-computed in flat_step() — no CPU sync here.
    // For decode-only (total_q == batch_size, all q_lens=1), fall through
    // to the regular decode path which is proven to work.
    // -----------------------------------------------------------------------
    // Flat-token with prefill: use FlashInfer BatchPrefillWithPagedKV.
    // For decode-only (total_q == batch_size), fall through to regular decode path.
    if (mDecodeState && mDecodeState->flat_token_mode && mDecodeState->paged && layer_idx >= 0
        && mDecodeState->flat_total_tokens > mDecodeState->flat_batch_size) {
        const int ds_Hkv = mDecodeState->num_kv_heads;
        const int ds_Hs = mDecodeState->head_dim;
        const int ds_batch = mDecodeState->flat_batch_size;
        const int total_q = mDecodeState->flat_total_tokens;

        // Extract Q from interleaved QKV [total_tokens, (Hq+2Hkv), Hs] → [total_tokens, Hq, Hs]
        Tensor q_buf = mRunState.temp_alloc(ETensorDType::BF16,
            {static_cast<long>(total_q), static_cast<long>(Hq), static_cast<long>(Hs)}, "flat_q");
        mTemps.push_back(q_buf);
        const auto* qkv_bf16 = qkv.get<nv_bfloat16>();
        auto* q_bf16 = q_buf.get<nv_bfloat16>();
        CUDA_CHECK(cudaMemcpy2DAsync(
            q_bf16,
            static_cast<std::size_t>(Hq) * Hs * sizeof(nv_bfloat16),
            qkv_bf16,
            static_cast<std::size_t>(H) * Hs * sizeof(nv_bfloat16),
            static_cast<std::size_t>(Hq) * Hs * sizeof(nv_bfloat16),
            static_cast<std::size_t>(total_q),
            cudaMemcpyDeviceToDevice, mRunState.MainStream));

        const int elem_size = mDecodeState->fp8 ? 1 : static_cast<int>(sizeof(nv_bfloat16));
        const std::size_t layer_pool_bytes =
            static_cast<std::size_t>(mDecodeState->total_pages)
            * mDecodeState->page_block_size * ds_Hkv * ds_Hs * elem_size;
        const std::size_t layer_offset = static_cast<std::size_t>(layer_idx) * layer_pool_bytes;

        Tensor flat_lse = mRunState.temp_alloc(ETensorDType::FP32,
            {static_cast<long>(total_q) * Hq}, "flat_lse");
        mTemps.push_back(flat_lse);

        // Use pre-computed plan from DecodeState (no CPU sync needed).
        if (mDecodeState->fp8) {
            const auto* k_pool_fp8 = reinterpret_cast<const __nv_fp8_e4m3*>(
                reinterpret_cast<const std::byte*>(mDecodeState->k_pages) + layer_offset);
            const auto* v_pool_fp8 = reinterpret_cast<const __nv_fp8_e4m3*>(
                reinterpret_cast<const std::byte*>(mDecodeState->v_pages) + layer_offset);
            attention_flat_paged_flashinfer_fp8(
                out.get<nv_bfloat16>(), flat_lse.get<float>(),
                q_bf16, k_pool_fp8, v_pool_fp8,
                mDecodeState->q_indptr_gpu,
                mDecodeState->seq_lens_k_gpu,
                mDecodeState->block_table_gpu, mDecodeState->block_table_stride,
                mDecodeState->page_block_size,
                ds_batch, total_q, mDecodeState->flat_padded_batch_size,
                Hq, ds_Hkv, ds_Hs,
                mDecodeState->flat_page_indptr_gpu,
                mDecodeState->flat_page_indices_gpu,
                mDecodeState->flat_last_page_len_gpu,
                mDecodeState->flat_plan_int_ws_gpu,
                mDecodeState->flat_plan_float_ws_gpu,
                static_cast<const void*>(mDecodeState->flat_plan_info_storage),
                nullptr, nullptr, nullptr,
                mRunState.MainStream);
        } else {
            auto* k_pool = reinterpret_cast<nv_bfloat16*>(
                reinterpret_cast<std::byte*>(mDecodeState->k_pages) + layer_offset);
            auto* v_pool = reinterpret_cast<nv_bfloat16*>(
                reinterpret_cast<std::byte*>(mDecodeState->v_pages) + layer_offset);
            attention_flat_paged_flashinfer(
                out.get<nv_bfloat16>(), flat_lse.get<float>(),
                q_bf16, k_pool, v_pool,
                mDecodeState->q_indptr_gpu,
                mDecodeState->seq_lens_k_gpu,
                mDecodeState->block_table_gpu, mDecodeState->block_table_stride,
                mDecodeState->page_block_size,
                ds_batch, total_q, mDecodeState->flat_padded_batch_size,
                Hq, ds_Hkv, ds_Hs,
                mDecodeState->flat_page_indptr_gpu,
                mDecodeState->flat_page_indices_gpu,
                mDecodeState->flat_last_page_len_gpu,
                mDecodeState->flat_plan_int_ws_gpu,
                mDecodeState->flat_plan_float_ws_gpu,
                static_cast<const void*>(mDecodeState->flat_plan_info_storage),
                nullptr, nullptr, nullptr,
                mRunState.MainStream);
        }
            
        return;
    }

    // -----------------------------------------------------------------------
    // Decode path: Q attends to KV-cache (autoregressive generation)
    // -----------------------------------------------------------------------
    // Prefill mode: normal self-attention (no decode interception).
    // KV-cache is populated by dispatch_rope; attention computes over full prompt.

    // Decode mode: intercept attention to use KV-cache
    if (mDecodeState && !mDecodeState->prefill_mode && mT == 1 && layer_idx >= 0) {
        const int ds_batch = static_cast<int>(mB);

        // Extract Q from interleaved QKV [B, 1, (Hq+2Hkv), Hs] → [B, Hq, Hs]
        Tensor q_buf = mRunState.temp_alloc(ETensorDType::BF16,
            {static_cast<long>(ds_batch), static_cast<long>(Hq), static_cast<long>(Hs)}, "decode_q");
        mTemps.push_back(q_buf);
        const auto* qkv_bf16 = qkv.get<nv_bfloat16>();
        auto* q_bf16 = q_buf.get<nv_bfloat16>();
        // Single strided copy: extract Q heads from interleaved QKV.
        CUDA_CHECK(cudaMemcpy2DAsync(
            q_bf16,
            static_cast<std::size_t>(Hq) * Hs * sizeof(nv_bfloat16),     // dst pitch
            qkv_bf16,
            static_cast<std::size_t>(H) * Hs * sizeof(nv_bfloat16),      // src pitch
            static_cast<std::size_t>(Hq) * Hs * sizeof(nv_bfloat16),     // width
            static_cast<std::size_t>(ds_batch),                           // height
            cudaMemcpyDeviceToDevice,
            mRunState.MainStream));

        const bool flashinfer_device_supported = flashinfer_decode_device_supported(mRunState.DeviceProp);
        const bool flashinfer_device_ok = flashinfer_device_supported;
        const int q_stride_n = Hq * Hs;

        if (mDecodeState->paged) {
            const int elem_size = mDecodeState->fp8 ? 1 : static_cast<int>(sizeof(nv_bfloat16));
            const std::size_t layer_pool_bytes =
                static_cast<std::size_t>(mDecodeState->total_pages)
                * mDecodeState->page_block_size * Hkv * Hs * elem_size;
            const std::size_t layer_offset = static_cast<std::size_t>(layer_idx) * layer_pool_bytes;

            if (mDecodeState->fp8) {
                // Use effective K lengths for this step (seq_lens + 1 after KV append).
                const int32_t* k_lens_gpu_i32 = mDecodeState->cu_seqlens_k_gpu
                    ? mDecodeState->cu_seqlens_k_gpu
                    : reinterpret_cast<const int32_t*>(mDecodeState->seq_lens_gpu);

                const auto* k_pages_layer = reinterpret_cast<const __nv_fp8_e4m3*>(
                    reinterpret_cast<const std::byte*>(mDecodeState->k_pages) + layer_offset);
                const auto* v_pages_layer = reinterpret_cast<const __nv_fp8_e4m3*>(
                    reinterpret_cast<const std::byte*>(mDecodeState->v_pages) + layer_offset);

                const bool flashinfer_head_ok = flashinfer_decode_head_supported(Hs);
                if (!flashinfer_device_supported) {
                    throw std::runtime_error(
                        "flash_attention: FP8 paged decode requires FlashInfer on SM80+ device");
                }
                if (!flashinfer_head_ok) {
                    throw std::runtime_error(
                        "flash_attention: FP8 paged decode unsupported head_dim; supported: 64/96/128/256");
                }

                Tensor fi_page_counts = mRunState.temp_alloc(
                    ETensorDType::INT32,
                    {static_cast<long>(ds_batch)},
                    "flashinfer_page_counts");
                Tensor fi_indptr = mRunState.temp_alloc(
                    ETensorDType::INT32,
                    {static_cast<long>(ds_batch + 1)},
                    "flashinfer_indptr");
                Tensor fi_last_page_len = mRunState.temp_alloc(
                    ETensorDType::INT32,
                    {static_cast<long>(ds_batch)},
                    "flashinfer_last_page_len");
                Tensor fi_indices = mRunState.temp_alloc(
                    ETensorDType::INT32,
                    {static_cast<long>(ds_batch) * mDecodeState->block_table_stride},
                    "flashinfer_indices");
                Tensor fi_request_indices = mRunState.temp_alloc(
                    ETensorDType::INT32,
                    {static_cast<long>(ds_batch)},
                    "flashinfer_request_indices");
                Tensor fi_kv_tile_indices = mRunState.temp_alloc(
                    ETensorDType::INT32,
                    {static_cast<long>(ds_batch)},
                    "flashinfer_kv_tile_indices");
                Tensor fi_kv_chunk_size = mRunState.temp_alloc(
                    ETensorDType::INT32,
                    {1},
                    "flashinfer_kv_chunk_size");
                mTemps.push_back(fi_page_counts);
                mTemps.push_back(fi_indptr);
                mTemps.push_back(fi_last_page_len);
                mTemps.push_back(fi_indices);
                mTemps.push_back(fi_request_indices);
                mTemps.push_back(fi_kv_tile_indices);
                mTemps.push_back(fi_kv_chunk_size);

                attention_decode_flashinfer_paged_fp8(
                    out.get<nv_bfloat16>(),
                    lse.get<float>(),
                    q_bf16,
                    k_pages_layer,
                    v_pages_layer,
                    k_lens_gpu_i32,
                    mDecodeState->block_table_gpu,
                    mDecodeState->block_table_stride,
                    mDecodeState->page_block_size,
                    ds_batch,
                    Hq,
                    Hkv,
                    Hs,
                    fi_page_counts.get<int32_t>(),
                    fi_indptr.get<int32_t>(),
                    fi_last_page_len.get<int32_t>(),
                    fi_indices.get<int32_t>(),
                    fi_request_indices.get<int32_t>(),
                    fi_kv_tile_indices.get<int32_t>(),
                    fi_kv_chunk_size.get<int32_t>(),
                    flashinfer_decode_pdl_enabled(),
                    mRunState.MainStream,
                    q_stride_n);
                log_flashinfer_decode_enabled_once(mRunState.DeviceProp, Hs);
            } else {
                // BF16 paged: direct paged attention
                auto* k_pool_layer = reinterpret_cast<nv_bfloat16*>(
                    reinterpret_cast<std::byte*>(mDecodeState->k_pages) + layer_offset);
                auto* v_pool_layer = reinterpret_cast<nv_bfloat16*>(
                    reinterpret_cast<std::byte*>(mDecodeState->v_pages) + layer_offset);
                const bool flashinfer_head_ok = flashinfer_decode_head_supported(Hs);
                if (flashinfer_device_ok && flashinfer_head_ok) {
                    const int32_t* k_lens_gpu = mDecodeState->cu_seqlens_k_gpu
                        ? mDecodeState->cu_seqlens_k_gpu
                        : reinterpret_cast<const int32_t*>(mDecodeState->seq_lens_gpu);

                    Tensor fi_page_counts = mRunState.temp_alloc(
                        ETensorDType::INT32,
                        {static_cast<long>(ds_batch)},
                        "flashinfer_page_counts");
                    Tensor fi_indptr = mRunState.temp_alloc(
                        ETensorDType::INT32,
                        {static_cast<long>(ds_batch + 1)},
                        "flashinfer_indptr");
                    Tensor fi_last_page_len = mRunState.temp_alloc(
                        ETensorDType::INT32,
                        {static_cast<long>(ds_batch)},
                        "flashinfer_last_page_len");
                    Tensor fi_indices = mRunState.temp_alloc(
                        ETensorDType::INT32,
                        {static_cast<long>(ds_batch) * mDecodeState->block_table_stride},
                        "flashinfer_indices");
                    Tensor fi_request_indices = mRunState.temp_alloc(
                        ETensorDType::INT32,
                        {static_cast<long>(ds_batch)},
                        "flashinfer_request_indices");
                    Tensor fi_kv_tile_indices = mRunState.temp_alloc(
                        ETensorDType::INT32,
                        {static_cast<long>(ds_batch)},
                        "flashinfer_kv_tile_indices");
                    Tensor fi_kv_chunk_size = mRunState.temp_alloc(
                        ETensorDType::INT32,
                        {1},
                        "flashinfer_kv_chunk_size");
                    mTemps.push_back(fi_page_counts);
                    mTemps.push_back(fi_indptr);
                    mTemps.push_back(fi_last_page_len);
                    mTemps.push_back(fi_indices);
                    mTemps.push_back(fi_request_indices);
                    mTemps.push_back(fi_kv_tile_indices);
                    mTemps.push_back(fi_kv_chunk_size);

                    attention_decode_flashinfer_paged(
                        out.get<nv_bfloat16>(),
                        lse.get<float>(),
                        q_bf16,
                        k_pool_layer,
                        v_pool_layer,
                        k_lens_gpu,
                        mDecodeState->block_table_gpu,
                        mDecodeState->block_table_stride,
                        mDecodeState->page_block_size,
                        ds_batch,
                        Hq,
                        Hkv,
                        Hs,
                        fi_page_counts.get<int32_t>(),
                        fi_indptr.get<int32_t>(),
                        fi_last_page_len.get<int32_t>(),
                        fi_indices.get<int32_t>(),
                        fi_request_indices.get<int32_t>(),
                        fi_kv_tile_indices.get<int32_t>(),
                        fi_kv_chunk_size.get<int32_t>(),
                        flashinfer_decode_pdl_enabled(),
                        mRunState.MainStream,
                        q_stride_n);
                    log_flashinfer_decode_enabled_once(mRunState.DeviceProp, Hs);
                } else {
                    if (flashinfer_device_ok && !flashinfer_head_ok) {
                        log_flashinfer_decode_head_fallback_once(Hs);
                    }
                    attention_decode_flash_paged(
                        out.get<nv_bfloat16>(), lse.get<float>(),
                        q_bf16, k_pool_layer, v_pool_layer,
                        mDecodeState->cu_seqlens_q_gpu,
                        mDecodeState->cu_seqlens_k_gpu,
                        mDecodeState->block_table_gpu, mDecodeState->block_table_stride,
                        mDecodeState->page_block_size,
                        mDecodeState->max_seqlen_k,
                        ds_batch, Hq, Hkv, Hs,
                        mRunState.MainStream, q_stride_n);
                }
            }
        } else {
            // Contiguous KV-cache
            const int ds_max_seq = mDecodeState->max_seq_len;

            if (mDecodeState->fp8) {
                // FP8 contiguous: dequant to temp BF16 buffer, then contiguous attention
                const int msk = mDecodeState->max_seqlen_k;
                Tensor k_tmp = mRunState.temp_alloc(ETensorDType::BF16,
                    {static_cast<long>(ds_batch) * msk * Hkv * Hs}, "fp8_dequant_k");
                Tensor v_tmp = mRunState.temp_alloc(ETensorDType::BF16,
                    {static_cast<long>(ds_batch) * msk * Hkv * Hs}, "fp8_dequant_v");
                mTemps.push_back(k_tmp);
                mTemps.push_back(v_tmp);

                const std::size_t layer_stride_fp8 =
                    static_cast<std::size_t>(ds_batch) * ds_max_seq * Hkv * Hs * 1;  // 1 byte per FP8
                const std::size_t layer_offset_fp8 = static_cast<std::size_t>(layer_idx) * layer_stride_fp8;
                const std::size_t scale_layer_elems =
                    static_cast<std::size_t>(ds_batch) * ds_max_seq * Hkv;
                const std::size_t scale_layer_offset = static_cast<std::size_t>(layer_idx) * scale_layer_elems;

                // Dequant must use effective K lengths for this step
                // (seq_lens + 1 after KV append).
                const int* k_lens_gpu = mDecodeState->cu_seqlens_k_gpu
                    ? reinterpret_cast<const int*>(mDecodeState->cu_seqlens_k_gpu)
                    : mDecodeState->seq_lens_gpu;

                kv_cache_dequant_fp8_to_bf16(
                    k_tmp.get<nv_bfloat16>(), v_tmp.get<nv_bfloat16>(),
                    reinterpret_cast<const __nv_fp8_e4m3*>(
                        reinterpret_cast<const std::byte*>(mDecodeState->k_data) + layer_offset_fp8),
                    reinterpret_cast<const __nv_fp8_e4m3*>(
                        reinterpret_cast<const std::byte*>(mDecodeState->v_data) + layer_offset_fp8),
                    mDecodeState->k_scales_fp8 + scale_layer_offset,
                    mDecodeState->v_scales_fp8 + scale_layer_offset,
                    k_lens_gpu,
                    ds_batch, ds_max_seq, Hkv, Hs,
                    mRunState.MainStream);

                attention_decode_flash(
                    out.get<nv_bfloat16>(), lse.get<float>(),
                    q_bf16, k_tmp.get<nv_bfloat16>(), v_tmp.get<nv_bfloat16>(),
                    mDecodeState->cu_seqlens_q_gpu,
                    mDecodeState->cu_seqlens_k_gpu,
                    msk, msk, ds_batch, Hq, Hkv, Hs,
                    mRunState.MainStream, q_stride_n);
            } else {
                const std::size_t layer_stride_bytes =
                    static_cast<std::size_t>(ds_batch) * ds_max_seq * Hkv * Hs * sizeof(nv_bfloat16);
                const std::size_t layer_offset = static_cast<std::size_t>(layer_idx) * layer_stride_bytes;

                auto* k_layer = reinterpret_cast<nv_bfloat16*>(
                    reinterpret_cast<std::byte*>(mDecodeState->k_data) + layer_offset);
                auto* v_layer = reinterpret_cast<nv_bfloat16*>(
                    reinterpret_cast<std::byte*>(mDecodeState->v_data) + layer_offset);

                attention_decode_flash(
                    out.get<nv_bfloat16>(), lse.get<float>(),
                    q_bf16, k_layer, v_layer,
                    mDecodeState->cu_seqlens_q_gpu,
                    mDecodeState->cu_seqlens_k_gpu,
                    mDecodeState->max_seqlen_k, ds_max_seq,
                    ds_batch, Hq, Hkv, Hs,
                    mRunState.MainStream, q_stride_n);
            }
        }
        return;
    }

    // -----------------------------------------------------------------------
    // Training path
    // -----------------------------------------------------------------------
    int window_size = op.attrs.window_size;
    if (window_size <= 0 && mConfig.use_sliding_window && mConfig.is_sliding_layer(layer_idx)) {
        window_size = mConfig.sliding_window_size;
    }

    const bool cudnn_supported = (window_size <= 0) &&
                                 (Hs > 0) &&
                                 (Hs % 8 == 0) &&
                                 (Hs <= 128) &&
                                 (mRunState.scratch().cudnn_workspace.Data != nullptr);

    // Use FlashAttention varlen when:
    // - document masking is enabled, or
    // - cuDNN full-attention is unavailable (e.g. head_dim > 128).
    //
    // For the latter, synthesize dense cu_seqlens for (B, T) packed as B documents.
    const bool use_varlen = (mCuSeqlensGpu != nullptr) || (window_size <= 0 && !cudnn_supported);
    const int32_t* cu_seqlens_ptr = mCuSeqlensGpu;
    int num_docs = mNumDocs;
    int max_doc_seqlen = mMaxDocSeqlen;
    int total_doc_tokens = mTotalDocTokens;
    Tensor generated_cu_seqlens;
    if (use_varlen && cu_seqlens_ptr == nullptr) {
        num_docs = static_cast<int>(mB);
        max_doc_seqlen = static_cast<int>(mT);
        total_doc_tokens = num_docs * max_doc_seqlen;
        generated_cu_seqlens = mRunState.temp_alloc(ETensorDType::INT32, {static_cast<long>(num_docs + 1)}, "generated_cu_seqlens");
        mTemps.push_back(generated_cu_seqlens);
        fill_dense_cu_seqlens(generated_cu_seqlens.get<int32_t>(), num_docs, max_doc_seqlen, mRunState.MainStream);
        cu_seqlens_ptr = generated_cu_seqlens.get<int32_t>();
    }

    if (use_varlen) {
        if (out.DType != ETensorDType::BF16 || qkv.DType != ETensorDType::BF16) {
            throw std::logic_error("flash_attention: varlen path currently requires BF16 tensors");
        }
        // Document-level masking: Flash Attention varlen path.
        // Write LSE directly into the pre-allocated output tensor (persists to backward).
        // For B>1, LSE is still consumed only by matching backward kernels and sinks path.
        attention_forward_flash_varlen(
            out.get<nv_bfloat16>(), lse.get<float>(), qkv.get<nv_bfloat16>(),
            cu_seqlens_ptr, num_docs, max_doc_seqlen, total_doc_tokens,
            Hq, Hkv, Hs, mRunState.MainStream);
    } else if (window_size > 0) {
        attention_forward_custom(out, lse, qkv,
                                 static_cast<int>(mB), static_cast<int>(mT),
                                 Hq, Hkv, Hs, window_size, mRunState.MainStream);
    } else {
        if (!mRunState.scratch().cudnn_workspace.Data) {
            mRunState.temp_acquire(mRunState.scratch().cudnn_workspace);
            mTemps.push_back(mRunState.scratch().cudnn_workspace);
        }
        attention_forward_cudnn(out, lse, qkv, mRunState.scratch().cudnn_workspace,
                                mRunState.CudnnHandle, static_cast<int>(mB), static_cast<int>(mT),
                                Hq, Hkv, Hs, mRunState.MainStream);
    }

    if (sinks) {
        Tensor* sinks_use = sinks;
        Tensor sinks_cast;
        if (sinks->DType != out.DType) {
            sinks_cast = mRunState.temp_alloc(out.DType, {static_cast<long>(Hq)}, "flash_attention_sinks_cast");
            mTemps.push_back(sinks_cast);
            if (out.DType == ETensorDType::BF16) {
                convert_dtype(sinks_cast.get<nv_bfloat16>(), sinks->get<float>(),
                              sinks->nelem(), mRunState.MainStream);
            } else if (out.DType == ETensorDType::FP32) {
                convert_dtype(sinks_cast.get<float>(), sinks->get<nv_bfloat16>(),
                              sinks->nelem(), mRunState.MainStream);
            } else {
                throw std::logic_error("flash_attention: unsupported sinks dtype conversion");
            }
            sinks_use = &sinks_cast;
        }
        if (out.DType == ETensorDType::BF16) {
            attention_apply_sinks(out.get<nv_bfloat16>(), lse.get<float>(),
                                  sinks_use->get<nv_bfloat16>(),
                                  static_cast<int>(mB), static_cast<int>(mT),
                                  Hq, Hs, mRunState.MainStream);
        } else if (out.DType == ETensorDType::FP32) {
            attention_apply_sinks(out.get<float>(), lse.get<float>(),
                                  sinks_use->get<float>(),
                                  static_cast<int>(mB), static_cast<int>(mT),
                                  Hq, Hs, mRunState.MainStream);
        } else {
            throw std::logic_error("flash_attention: unsupported output dtype");
        }
    }
}

void CompiledExecutor::dispatch_flash_attention_backward(const CompiledOp& op) {
    // inputs (from autodiff): d_out, out (attention output), lse, qkv
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& out = resolve_tensor(op.inputs[1]);
    Tensor& lse = resolve_tensor(op.inputs[2]);
    Tensor& qkv = resolve_tensor(op.inputs[3]);
    Tensor* sinks = nullptr;
    if (op.inputs.size() > 4 && !op.inputs[4].name.empty()) {
        sinks = &resolve_tensor(op.inputs[4]);
    }
    Tensor* d_qkv_ptr = &ensure_output_tensor(op.outputs[0]);
    const long qkv_nelem = static_cast<long>(qkv.nelem());
    if (d_qkv_ptr->Rank == 0 || d_qkv_ptr->nelem() != qkv_nelem || d_qkv_ptr->DType != d_out.DType) {
        std::vector<long> shape(qkv.Sizes.begin(), qkv.Sizes.begin() + qkv.Rank);
        Tensor tmp = mRunState.temp_alloc(d_out.DType, shape, "flash_attention_d_qkv");
        mTemps.push_back(tmp);
        d_qkv_ptr = &mTemps.back();
    }
    Tensor& d_qkv = *d_qkv_ptr;

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    int layer_idx = op.attrs.layer_idx;
    if (layer_idx < 0 && !op.inputs.empty() && op.inputs[0].layer_idx >= 0) {
        layer_idx = op.inputs[0].layer_idx;
    }
    if (layer_idx < 0 && op.inputs.size() > 3) {
        std::string field;
        parse_block_param(op.inputs[3].name, layer_idx, field);
    }

    int window_size = op.attrs.window_size;
    if (window_size <= 0 && mConfig.use_sliding_window && mConfig.is_sliding_layer(layer_idx)) {
        window_size = mConfig.sliding_window_size;
    }

    const bool cudnn_supported = (window_size <= 0) &&
                                 (Hs > 0) &&
                                 (Hs % 8 == 0) &&
                                 (Hs <= 128) &&
                                 (mRunState.scratch().cudnn_workspace.Data != nullptr);

    const bool use_varlen = (mCuSeqlensGpu != nullptr) || (window_size <= 0 && !cudnn_supported);
    const int32_t* cu_seqlens_ptr = mCuSeqlensGpu;
    int num_docs = mNumDocs;
    int max_doc_seqlen = mMaxDocSeqlen;
    int total_doc_tokens = mTotalDocTokens;
    Tensor generated_cu_seqlens;
    if (use_varlen && cu_seqlens_ptr == nullptr) {
        num_docs = static_cast<int>(mB);
        max_doc_seqlen = static_cast<int>(mT);
        total_doc_tokens = num_docs * max_doc_seqlen;
        generated_cu_seqlens = mRunState.temp_alloc(ETensorDType::INT32, {static_cast<long>(num_docs + 1)}, "generated_cu_seqlens");
        mTemps.push_back(generated_cu_seqlens);
        fill_dense_cu_seqlens(generated_cu_seqlens.get<int32_t>(), num_docs, max_doc_seqlen, mRunState.MainStream);
        cu_seqlens_ptr = generated_cu_seqlens.get<int32_t>();
    }

    if (use_varlen) {
        if (out.DType != ETensorDType::BF16 ||
            d_out.DType != ETensorDType::BF16 ||
            qkv.DType != ETensorDType::BF16 ||
            d_qkv.DType != ETensorDType::BF16) {
            throw std::logic_error("flash_attention_backward: varlen path currently requires BF16 tensors");
        }
        // Document-level masking: Flash Attention varlen backward
        const int Hs_rounded = Hs <= 128 ? ((Hs + 31) / 32) * 32 : ((Hs + 63) / 64) * 64;
        const long padded_total = static_cast<long>(total_doc_tokens) + 128L * static_cast<long>(num_docs);
        const long dq_accum_elems = padded_total * static_cast<long>(Hq) * static_cast<long>(Hs_rounded);
        const long dsoftmax_elems = static_cast<long>(Hq) * padded_total;
        Tensor dq_accum = mRunState.temp_alloc(ETensorDType::FP32, {dq_accum_elems}, "flash_attention_dq_accum");
        Tensor dsoftmax = mRunState.temp_alloc(ETensorDType::FP32, {dsoftmax_elems}, "flash_attention_dsoftmax");
        mTemps.push_back(dq_accum);
        mTemps.push_back(dsoftmax);

        // GQA expanded dk/dv buffers: flash backward writes dK/dV with Hq head
        // indices, but interleaved buffer only has Hkv slots. Allocate separate
        // (total_q, Hq, HS) buffers when Hq != Hkv.
        nv_bfloat16* dk_exp_ptr = nullptr;
        nv_bfloat16* dv_exp_ptr = nullptr;
        Tensor dk_expanded, dv_expanded;
        if (Hq != Hkv) {
            const long exp_elems = static_cast<long>(total_doc_tokens) * static_cast<long>(Hq) * static_cast<long>(Hs);
            dk_expanded = mRunState.temp_alloc(ETensorDType::BF16, {exp_elems}, "flash_attention_dk_expanded");
            dv_expanded = mRunState.temp_alloc(ETensorDType::BF16, {exp_elems}, "flash_attention_dv_expanded");
            mTemps.push_back(dk_expanded);
            mTemps.push_back(dv_expanded);
            dk_exp_ptr = dk_expanded.get<nv_bfloat16>();
            dv_exp_ptr = dv_expanded.get<nv_bfloat16>();
        }

        // Zero dqkv — convert_dQ writes all Q elements, but K/V sections need
        // zeroing for MHA path (or are overwritten by reduce_scatter for GQA).
        fill_zero(d_qkv, mRunState.MainStream);

        attention_backward_flash_varlen(
            d_qkv.get<nv_bfloat16>(), lse.get<float>(),
            out.get<nv_bfloat16>(), d_out.get<nv_bfloat16>(), qkv.get<nv_bfloat16>(),
            cu_seqlens_ptr, dq_accum.get<float>(), dsoftmax.get<float>(),
            dk_exp_ptr, dv_exp_ptr,
            num_docs, max_doc_seqlen, total_doc_tokens,
            Hq, Hkv, Hs, /*deterministic=*/false, mRunState.MainStream);
    } else if (window_size > 0) {
        if (out.DType == ETensorDType::FP32) {
            attention_backward_custom(d_qkv, lse, out, d_out, qkv,
                                      static_cast<int>(mB), static_cast<int>(mT),
                                      Hq, Hkv, Hs, window_size, mRunState.MainStream);
        } else if (out.DType == ETensorDType::BF16) {
            auto shape_vec = [](const Tensor& t) {
                return std::vector<long>(t.Sizes.begin(), t.Sizes.begin() + t.Rank);
            };
            Tensor out_f32 = mRunState.temp_alloc(ETensorDType::FP32, shape_vec(out), "flash_attention_out_f32");
            Tensor d_out_f32 = mRunState.temp_alloc(ETensorDType::FP32, shape_vec(d_out), "flash_attention_d_out_f32");
            Tensor qkv_f32 = mRunState.temp_alloc(ETensorDType::FP32, shape_vec(qkv), "flash_attention_qkv_f32");
            Tensor d_qkv_f32 = mRunState.temp_alloc(ETensorDType::FP32, shape_vec(d_qkv), "flash_attention_d_qkv_f32");
            mTemps.push_back(out_f32);
            mTemps.push_back(d_out_f32);
            mTemps.push_back(qkv_f32);
            mTemps.push_back(d_qkv_f32);

            convert_dtype(out_f32.get<float>(), out.get<nv_bfloat16>(), out.nelem(), mRunState.MainStream);
            convert_dtype(d_out_f32.get<float>(), d_out.get<nv_bfloat16>(), d_out.nelem(), mRunState.MainStream);
            convert_dtype(qkv_f32.get<float>(), qkv.get<nv_bfloat16>(), qkv.nelem(), mRunState.MainStream);

            attention_backward_custom(d_qkv_f32, lse, out_f32, d_out_f32, qkv_f32,
                                      static_cast<int>(mB), static_cast<int>(mT),
                                      Hq, Hkv, Hs, window_size, mRunState.MainStream);
            convert_dtype(d_qkv.get<nv_bfloat16>(), d_qkv_f32.get<float>(),
                          d_qkv.nelem(), mRunState.MainStream);
        } else {
            throw std::logic_error("flash_attention_backward: unsupported dtype for custom path");
        }
    } else {
        if (!mRunState.scratch().cudnn_workspace.Data) {
            mRunState.temp_acquire(mRunState.scratch().cudnn_workspace);
            mTemps.push_back(mRunState.scratch().cudnn_workspace);
        }

        const int attn_chunks = mOptions.AttBwdChunks;
        if (attn_chunks < 1) {
            throw std::runtime_error("attn_bwd_chunks must be >= 1");
        }
        const int chunk_B = (attn_chunks == 1)
            ? static_cast<int>(mB)
            : static_cast<int>(div_exact(mB, static_cast<long>(attn_chunks)));

        if (attn_chunks == 1) {
            attention_backward_cudnn(d_qkv, lse, out, d_out, qkv,
                                     mRunState.scratch().cudnn_workspace,
                                     mRunState.CudnnHandle,
                                     static_cast<int>(mB), static_cast<int>(mT),
                                     Hq, Hkv, Hs, mRunState.MainStream);
        } else {
            for (int chunk = 0; chunk < attn_chunks; ++chunk) {
                const long start = static_cast<long>(chunk) * static_cast<long>(chunk_B);
                const long end = start + static_cast<long>(chunk_B);
                Tensor d_out_chunk = slice(d_out, 0, start, end);
                Tensor out_chunk = slice(out, 0, start, end);
                Tensor lse_chunk = slice(lse, 0, start, end);
                Tensor qkv_chunk = slice(qkv, 0, start, end);
                Tensor d_qkv_chunk = slice(d_qkv, 0, start, end);

                attention_backward_cudnn(d_qkv_chunk, lse_chunk, out_chunk, d_out_chunk, qkv_chunk,
                                         mRunState.scratch().cudnn_workspace,
                                         mRunState.CudnnHandle,
                                         static_cast<int>(chunk_B), static_cast<int>(mT),
                                         Hq, Hkv, Hs, mRunState.MainStream);
            }
        }
    }

    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        if (!sinks || !sinks->Data) {
            // Sinks parameter not available (e.g., offloaded in QLoRA mode or not a LoRA target).
            // Skip sinks gradient computation — it's unused when sinks isn't being trained.
            goto skip_sinks;
        }
        Tensor& d_sinks_out = ensure_output_tensor(op.outputs[1]);

        bool accumulate = mAccumulateTensors.count(op.outputs[1].name) > 0;
        if (!accumulate && !op.outputs[1].name.empty()) {
            if (auto base = base_param_from_grad(op.outputs[1].name)) {
                accumulate = mAccumulateTensors.count("d_" + *base) > 0;
            }
        }

        Tensor d_sinks_f32 = mRunState.temp_alloc(ETensorDType::FP32, {static_cast<long>(Hq)}, "flash_attention_d_sinks_f32");
        mTemps.push_back(d_sinks_f32);
        fill_zero(d_sinks_f32, mRunState.MainStream);

        Tensor* sinks_use = sinks;
        Tensor sinks_cast;
        if (sinks->DType != out.DType) {
            sinks_cast = mRunState.temp_alloc(out.DType, {static_cast<long>(Hq)}, "flash_attention_sinks_cast");
            mTemps.push_back(sinks_cast);
            if (out.DType == ETensorDType::BF16) {
                convert_dtype(sinks_cast.get<nv_bfloat16>(), sinks->get<float>(),
                              sinks->nelem(), mRunState.MainStream);
            } else if (out.DType == ETensorDType::FP32) {
                convert_dtype(sinks_cast.get<float>(), sinks->get<nv_bfloat16>(),
                              sinks->nelem(), mRunState.MainStream);
            } else {
                throw std::logic_error("flash_attention_backward: unsupported sinks dtype conversion");
            }
            sinks_use = &sinks_cast;
        }

        if (out.DType == ETensorDType::BF16) {
            attention_sinks_backward(d_sinks_f32.get<float>(),
                                     out.get<nv_bfloat16>(), d_out.get<nv_bfloat16>(), lse.get<float>(),
                                     sinks_use->get<nv_bfloat16>(),
                                     static_cast<int>(mB), static_cast<int>(mT),
                                     Hq, Hs, mRunState.MainStream);
        } else if (out.DType == ETensorDType::FP32) {
            attention_sinks_backward(d_sinks_f32.get<float>(),
                                     out.get<float>(), d_out.get<float>(), lse.get<float>(),
                                     sinks_use->get<float>(),
                                     static_cast<int>(mB), static_cast<int>(mT),
                                     Hq, Hs, mRunState.MainStream);
        } else {
            throw std::logic_error("flash_attention_backward: unsupported output dtype for sinks grad");
        }

        if (d_sinks_out.DType == ETensorDType::FP32) {
            if (accumulate) {
                vector_add_sr(d_sinks_out, d_sinks_out, d_sinks_f32, 1.0f,
                              static_cast<long>(d_sinks_out.nelem()), 0, mRunState.MainStream);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(d_sinks_out.Data, d_sinks_f32.Data, d_sinks_out.bytes(),
                                           cudaMemcpyDeviceToDevice, mRunState.MainStream));
            }
        } else if (d_sinks_out.DType == ETensorDType::BF16) {
            Tensor d_sinks_bf16 = mRunState.temp_alloc(ETensorDType::BF16, {static_cast<long>(Hq)}, "flash_attention_d_sinks_bf16");
            mTemps.push_back(d_sinks_bf16);
            convert_dtype(d_sinks_bf16.get<nv_bfloat16>(), d_sinks_f32.get<float>(),
                          d_sinks_f32.nelem(), mRunState.MainStream);
            if (accumulate) {
                vector_add_sr(d_sinks_out, d_sinks_out, d_sinks_bf16, 1.0f,
                              static_cast<long>(d_sinks_out.nelem()), 0, mRunState.MainStream);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(d_sinks_out.Data, d_sinks_bf16.Data, d_sinks_out.bytes(),
                                           cudaMemcpyDeviceToDevice, mRunState.MainStream));
            }
        } else {
            throw std::logic_error("flash_attention_backward: unsupported d_sinks dtype");
        }
    }
    skip_sinks:

    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        store_tensor(op.outputs[0], d_qkv);
    }
}


}  // namespace dsl
