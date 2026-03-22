#include "runtime/dsl/compiled_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "kernels/attention_decode.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_rope(const CompiledOp& op) {
    Tensor& qkv = resolve_tensor(op.inputs[0]);
    Tensor& freqs = resolve_tensor(op.inputs[1]);
    Tensor& pos_ids = resolve_tensor(op.inputs[2]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());

    if (mForwardPlan) {
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(op.inputs[0].name, layer_idx, field) &&
            layer_idx >= 0 && static_cast<std::size_t>(layer_idx) < mForwardPlan->size()) {
            AttnForwardPlan plan{};
            plan.valid = true;
            plan.use_qk_norm = false;
            plan.rope_fused = false;
            plan.use_cudnn = true;
            plan.rotary_dim = op.attrs.rotary_dim;
            (*mForwardPlan)[static_cast<std::size_t>(layer_idx)].attn = plan;
        }
    }

    rope_forward(out, qkv, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
                 static_cast<int>(mB), static_cast<int>(mT), Hq, Hkv, Hs,
                 op.attrs.rotary_dim, mRunState.MainStream);

    // ========================================================================
    // Prefill mode: bulk-store ALL positions of K/V to KV-cache
    // Normal self-attention proceeds (dispatch_flash_attention is NOT intercepted)
    // ========================================================================
    if (mDecodeState && mDecodeState->prefill_mode && mT > 1) {
        int layer_idx = -1;
        std::string field;
        parse_block_param(op.inputs[0].name, layer_idx, field);

        if (layer_idx >= 0) {
            const int ds_batch = static_cast<int>(mB);
            const int ds_T = static_cast<int>(mT);
            const int ds_Hkv = mDecodeState->num_kv_heads;
            const int ds_Hs = mDecodeState->head_dim;

            if (mDecodeState->paged) {
                const int elem_size = mDecodeState->fp8 ? 1 : static_cast<int>(sizeof(nv_bfloat16));
                const std::size_t layer_pool_bytes =
                    static_cast<std::size_t>(mDecodeState->total_pages)
                    * mDecodeState->page_block_size * ds_Hkv * ds_Hs * elem_size;
                const std::size_t layer_offset = static_cast<std::size_t>(layer_idx) * layer_pool_bytes;
                if (mDecodeState->fp8) {
                    const std::size_t scale_layer_elems =
                        static_cast<std::size_t>(mDecodeState->total_pages)
                        * mDecodeState->page_block_size * ds_Hkv;
                    const std::size_t scale_layer_offset =
                        static_cast<std::size_t>(layer_idx) * scale_layer_elems;

                    kv_cache_store_paged_fp8(
                        reinterpret_cast<__nv_fp8_e4m3*>(
                            reinterpret_cast<std::byte*>(mDecodeState->k_pages) + layer_offset),
                        reinterpret_cast<__nv_fp8_e4m3*>(
                            reinterpret_cast<std::byte*>(mDecodeState->v_pages) + layer_offset),
                        mDecodeState->k_scales_paged_fp8 + scale_layer_offset,
                        mDecodeState->v_scales_paged_fp8 + scale_layer_offset,
                        out.get<nv_bfloat16>(),
                        mDecodeState->block_table_gpu, mDecodeState->block_table_stride,
                        mDecodeState->page_block_size,
                        ds_batch, ds_T, Hq, ds_Hkv, ds_Hs,
                        mDecodeState->prefill_pos_offset,
                        mRunState.MainStream);
                } else {
                    auto* k_pool = reinterpret_cast<nv_bfloat16*>(
                        reinterpret_cast<std::byte*>(mDecodeState->k_pages) + layer_offset);
                    auto* v_pool = reinterpret_cast<nv_bfloat16*>(
                        reinterpret_cast<std::byte*>(mDecodeState->v_pages) + layer_offset);

                    kv_cache_store_paged_bf16(
                        k_pool, v_pool, out.get<nv_bfloat16>(),
                        mDecodeState->block_table_gpu, mDecodeState->block_table_stride,
                        mDecodeState->page_block_size,
                        ds_batch, ds_T, Hq, ds_Hkv, ds_Hs,
                        mDecodeState->prefill_pos_offset,
                        mRunState.MainStream);
                }
            } else {
                const int ds_max_seq = mDecodeState->max_seq_len;
                const std::size_t layer_stride_bytes =
                    static_cast<std::size_t>(ds_batch) * ds_max_seq * ds_Hkv * ds_Hs * sizeof(nv_bfloat16);
                const std::size_t layer_offset = static_cast<std::size_t>(layer_idx) * layer_stride_bytes;

                auto* k_layer = reinterpret_cast<nv_bfloat16*>(
                    reinterpret_cast<std::byte*>(mDecodeState->k_data) + layer_offset);
                auto* v_layer = reinterpret_cast<nv_bfloat16*>(
                    reinterpret_cast<std::byte*>(mDecodeState->v_data) + layer_offset);

                kv_cache_store_bf16(
                    k_layer, v_layer, out.get<nv_bfloat16>(),
                    ds_batch, ds_T, ds_max_seq, Hq, ds_Hkv, ds_Hs,
                    mDecodeState->prefill_pos_offset,
                    mRunState.MainStream);
            }
        }
        // Don't return — let the normal forward proceed (no attention interception)
    }

    // ========================================================================
    // Decode mode: append K/V to KV-cache after RoPE (T=1)
    // ========================================================================
    if (mDecodeState && !mDecodeState->prefill_mode && mT == 1) {
        int layer_idx = -1;
        std::string field;
        parse_block_param(op.inputs[0].name, layer_idx, field);

        if (layer_idx >= 0) {
            const int ds_batch = static_cast<int>(mB);
            const int ds_Hkv = mDecodeState->num_kv_heads;
            const int ds_Hs = mDecodeState->head_dim;

            if (mDecodeState->paged) {
                // Paged KV-cache
                const int elem_size = mDecodeState->fp8 ? 1 : static_cast<int>(sizeof(nv_bfloat16));
                const std::size_t layer_pool_bytes =
                    static_cast<std::size_t>(mDecodeState->total_pages)
                    * mDecodeState->page_block_size * ds_Hkv * ds_Hs * elem_size;
                const std::size_t layer_offset = static_cast<std::size_t>(layer_idx) * layer_pool_bytes;

                if (mDecodeState->fp8) {
                    const std::size_t scale_layer_elems =
                        static_cast<std::size_t>(mDecodeState->total_pages)
                        * mDecodeState->page_block_size * ds_Hkv;
                    const std::size_t scale_layer_offset = static_cast<std::size_t>(layer_idx) * scale_layer_elems;

                    kv_cache_append_paged_fp8(
                        reinterpret_cast<__nv_fp8_e4m3*>(
                            reinterpret_cast<std::byte*>(mDecodeState->k_pages) + layer_offset),
                        reinterpret_cast<__nv_fp8_e4m3*>(
                            reinterpret_cast<std::byte*>(mDecodeState->v_pages) + layer_offset),
                        mDecodeState->k_scales_paged_fp8 + scale_layer_offset,
                        mDecodeState->v_scales_paged_fp8 + scale_layer_offset,
                        out.get<nv_bfloat16>(),
                        mDecodeState->seq_lens_gpu,
                        mDecodeState->block_table_gpu, mDecodeState->block_table_stride,
                        mDecodeState->page_block_size,
                        ds_batch, Hq, ds_Hkv, ds_Hs,
                        mRunState.MainStream);
                } else {
                    auto* k_pool_layer = reinterpret_cast<nv_bfloat16*>(
                        reinterpret_cast<std::byte*>(mDecodeState->k_pages) + layer_offset);
                    auto* v_pool_layer = reinterpret_cast<nv_bfloat16*>(
                        reinterpret_cast<std::byte*>(mDecodeState->v_pages) + layer_offset);

                    kv_cache_append_paged_bf16(
                        k_pool_layer, v_pool_layer,
                        out.get<nv_bfloat16>(),
                        mDecodeState->seq_lens_gpu,
                        mDecodeState->block_table_gpu, mDecodeState->block_table_stride,
                        mDecodeState->page_block_size,
                        ds_batch, Hq, ds_Hkv, ds_Hs,
                        mRunState.MainStream);
                }
            } else {
                // Contiguous KV-cache
                const int ds_max_seq = mDecodeState->max_seq_len;

                if (mDecodeState->fp8) {
                    const std::size_t layer_stride_bytes =
                        static_cast<std::size_t>(ds_batch) * ds_max_seq * ds_Hkv * ds_Hs * 1;  // FP8=1 byte
                    const std::size_t layer_offset = static_cast<std::size_t>(layer_idx) * layer_stride_bytes;
                    const std::size_t scale_layer_elems =
                        static_cast<std::size_t>(ds_batch) * ds_max_seq * ds_Hkv;
                    const std::size_t scale_layer_offset = static_cast<std::size_t>(layer_idx) * scale_layer_elems;

                    kv_cache_append_fp8(
                        reinterpret_cast<__nv_fp8_e4m3*>(
                            reinterpret_cast<std::byte*>(mDecodeState->k_data) + layer_offset),
                        reinterpret_cast<__nv_fp8_e4m3*>(
                            reinterpret_cast<std::byte*>(mDecodeState->v_data) + layer_offset),
                        mDecodeState->k_scales_fp8 + scale_layer_offset,
                        mDecodeState->v_scales_fp8 + scale_layer_offset,
                        out.get<nv_bfloat16>(),
                        mDecodeState->seq_lens_gpu,
                        ds_batch, ds_max_seq,
                        Hq, ds_Hkv, ds_Hs,
                        mRunState.MainStream);
                } else {
                    const std::size_t layer_stride_bytes =
                        static_cast<std::size_t>(ds_batch) * ds_max_seq * ds_Hkv * ds_Hs * sizeof(nv_bfloat16);
                    const std::size_t layer_offset = static_cast<std::size_t>(layer_idx) * layer_stride_bytes;

                    auto* k_layer = reinterpret_cast<nv_bfloat16*>(
                        reinterpret_cast<std::byte*>(mDecodeState->k_data) + layer_offset);
                    auto* v_layer = reinterpret_cast<nv_bfloat16*>(
                        reinterpret_cast<std::byte*>(mDecodeState->v_data) + layer_offset);

                    kv_cache_append_bf16(
                        k_layer, v_layer,
                        out.get<nv_bfloat16>(),
                        mDecodeState->seq_lens_gpu,
                        ds_batch, ds_max_seq,
                        Hq, ds_Hkv, ds_Hs,
                        mRunState.MainStream);
                }
            }
        }
    }
}

void CompiledExecutor::dispatch_rope_backward(const CompiledOp& op) {
    // inputs: d_qkv_rope, freq_cis, position_ids
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& freqs = resolve_tensor(op.inputs[1]);
    Tensor& pos_ids = resolve_tensor(op.inputs[2]);
    Tensor& d_qkv = ensure_output_tensor(op.outputs[0]);

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());

    // For FP8 hybrid backward, record abs_max of d_qkv for subsequent quantization
    float* abs_max_ptr = mRunState.has_fp8_hybrid_backward()
        ? mRunState.simplified_quant_grads().d_qkv.abs_max()
        : nullptr;

    rope_backward(d_qkv, d_out, freqs, reinterpret_cast<int*>(pos_ids.Data), abs_max_ptr,
                  static_cast<int>(mB), static_cast<int>(mT),
                  Hq, Hkv, Hs, op.attrs.rotary_dim, mRunState.MainStream);
}

}  // namespace dsl
