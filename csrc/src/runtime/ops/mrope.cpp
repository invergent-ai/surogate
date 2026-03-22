#include "runtime/dsl/compiled_ops.h"

#include "kernels/attention_decode.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace dsl {

void CompiledExecutor::dispatch_mrope(const CompiledOp& op) {
    Tensor& qkv_in = resolve_tensor(op.inputs[0]);
    Tensor& freqs = resolve_tensor(op.inputs[1]);
    Tensor& pos_ids = resolve_tensor(op.inputs[2]);

    Tensor* qkv_out_ptr = &ensure_output_tensor(op.outputs[0]);
    if (qkv_out_ptr->DType != qkv_in.DType ||
        qkv_out_ptr->Rank != qkv_in.Rank ||
        qkv_out_ptr->nelem() != qkv_in.nelem()) {
        std::vector<long> shape(qkv_in.Sizes.begin(), qkv_in.Sizes.begin() + qkv_in.Rank);
        Tensor tmp = mRunState.temp_alloc(qkv_in.DType, shape, "mrope_out");
        mTemps.push_back(tmp);
        qkv_out_ptr = &mTemps.back();
    }
    Tensor& qkv_out = *qkv_out_ptr;

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    const int qkv_channels = Hs * (Hq + 2 * Hkv);

    if (qkv_in.Data != qkv_out.Data) {
        CUDA_CHECK(cudaMemcpyAsync(qkv_out.Data, qkv_in.Data,
                                   qkv_in.bytes(), cudaMemcpyDeviceToDevice, mRunState.MainStream));
    }

    Tensor qkv_view = qkv_out;
    const long needed = static_cast<long>(mB) * static_cast<long>(mT) * qkv_channels;
    if ((qkv_out.Rank == 4 || (qkv_out.Rank == 3 && qkv_out.Sizes[2] != qkv_channels)) &&
        static_cast<long>(qkv_out.nelem()) >= needed) {
        qkv_view = view_tensor(qkv_out, {mB, mT, qkv_channels});
    }
    int rotary_dim = op.attrs.rotary_dim;

    const int* pos_ptr = reinterpret_cast<int*>(pos_ids.Data);
    int pos_planes = 1;
    if (pos_ids.Rank == 3) {
        pos_planes = static_cast<int>(pos_ids.Sizes[0]);
        if (pos_planes == 4) {
            pos_ptr += static_cast<int>(mB * mT);
            pos_planes = 3;
        }
    }

    mrope_forward(qkv_view, qkv_view, freqs, pos_ptr, pos_planes,
                 op.attrs.mrope_section[0], op.attrs.mrope_section[1], op.attrs.mrope_section[2],
                 nullptr, static_cast<int>(mB), static_cast<int>(mT), Hq, Hkv, Hs, rotary_dim,
                 mRunState.MainStream);

    store_tensor(op.outputs[0], qkv_out);

    // KV-cache intercept (same as dispatch_rope / dispatch_qkv_qk_norm_rope)
    if (mDecodeState) {
        int layer_idx = -1;
        std::string field;
        parse_block_param(op.inputs[0].name, layer_idx, field);

        if (layer_idx >= 0) {
            const int ds_batch = static_cast<int>(mB);
            const int ds_Hkv = mDecodeState->num_kv_heads;
            const int ds_Hs = mDecodeState->head_dim;

            if (mDecodeState->prefill_mode && mT > 1) {
                if (!mDecodeState->paged) {
                    const std::size_t ls = static_cast<std::size_t>(ds_batch) * mDecodeState->max_seq_len * ds_Hkv * ds_Hs * sizeof(nv_bfloat16);
                    auto* kl = reinterpret_cast<nv_bfloat16*>(reinterpret_cast<std::byte*>(mDecodeState->k_data) + layer_idx * ls);
                    auto* vl = reinterpret_cast<nv_bfloat16*>(reinterpret_cast<std::byte*>(mDecodeState->v_data) + layer_idx * ls);
                    kv_cache_store_bf16(kl, vl, qkv_out.get<nv_bfloat16>(),
                        ds_batch, static_cast<int>(mT), mDecodeState->max_seq_len, Hq, ds_Hkv, ds_Hs,
                        mDecodeState->prefill_pos_offset,
                        mRunState.MainStream);
                } else {
                    const int es = mDecodeState->fp8 ? 1 : static_cast<int>(sizeof(nv_bfloat16));
                    const std::size_t lp = static_cast<std::size_t>(mDecodeState->total_pages)
                        * mDecodeState->page_block_size * ds_Hkv * ds_Hs * es;
                    const std::size_t layer_offset = static_cast<std::size_t>(layer_idx) * lp;
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
                            qkv_out.get<nv_bfloat16>(),
                            mDecodeState->block_table_gpu, mDecodeState->block_table_stride,
                            mDecodeState->page_block_size, ds_batch, static_cast<int>(mT), Hq, ds_Hkv, ds_Hs,
                            mDecodeState->prefill_pos_offset,
                            mRunState.MainStream);
                    } else {
                        auto* kp = reinterpret_cast<nv_bfloat16*>(
                            reinterpret_cast<std::byte*>(mDecodeState->k_pages) + layer_offset);
                        auto* vp = reinterpret_cast<nv_bfloat16*>(
                            reinterpret_cast<std::byte*>(mDecodeState->v_pages) + layer_offset);
                        kv_cache_store_paged_bf16(kp, vp, qkv_out.get<nv_bfloat16>(),
                            mDecodeState->block_table_gpu, mDecodeState->block_table_stride,
                            mDecodeState->page_block_size, ds_batch, static_cast<int>(mT), Hq, ds_Hkv, ds_Hs,
                            mDecodeState->prefill_pos_offset,
                            mRunState.MainStream);
                    }
                }
            } else if (!mDecodeState->prefill_mode && mT == 1) {
                if (!mDecodeState->paged) {
                    const std::size_t ls = static_cast<std::size_t>(ds_batch) * mDecodeState->max_seq_len * ds_Hkv * ds_Hs * sizeof(nv_bfloat16);
                    auto* kl = reinterpret_cast<nv_bfloat16*>(reinterpret_cast<std::byte*>(mDecodeState->k_data) + layer_idx * ls);
                    auto* vl = reinterpret_cast<nv_bfloat16*>(reinterpret_cast<std::byte*>(mDecodeState->v_data) + layer_idx * ls);
                    kv_cache_append_bf16(kl, vl, qkv_out.get<nv_bfloat16>(),
                        mDecodeState->seq_lens_gpu, ds_batch, mDecodeState->max_seq_len,
                        Hq, ds_Hkv, ds_Hs, mRunState.MainStream);
                } else {
                    const int es = mDecodeState->fp8 ? 1 : static_cast<int>(sizeof(nv_bfloat16));
                    const std::size_t lp = static_cast<std::size_t>(mDecodeState->total_pages)
                        * mDecodeState->page_block_size * ds_Hkv * ds_Hs * es;
                    const std::size_t layer_offset = static_cast<std::size_t>(layer_idx) * lp;
                    if (mDecodeState->fp8) {
                        const std::size_t scale_layer_elems =
                            static_cast<std::size_t>(mDecodeState->total_pages)
                            * mDecodeState->page_block_size * ds_Hkv;
                        const std::size_t scale_layer_offset =
                            static_cast<std::size_t>(layer_idx) * scale_layer_elems;
                        kv_cache_append_paged_fp8(
                            reinterpret_cast<__nv_fp8_e4m3*>(
                                reinterpret_cast<std::byte*>(mDecodeState->k_pages) + layer_offset),
                            reinterpret_cast<__nv_fp8_e4m3*>(
                                reinterpret_cast<std::byte*>(mDecodeState->v_pages) + layer_offset),
                            mDecodeState->k_scales_paged_fp8 + scale_layer_offset,
                            mDecodeState->v_scales_paged_fp8 + scale_layer_offset,
                            qkv_out.get<nv_bfloat16>(),
                            mDecodeState->seq_lens_gpu,
                            mDecodeState->block_table_gpu, mDecodeState->block_table_stride,
                            mDecodeState->page_block_size, ds_batch, Hq, ds_Hkv, ds_Hs, mRunState.MainStream);
                    } else {
                        auto* kp = reinterpret_cast<nv_bfloat16*>(
                            reinterpret_cast<std::byte*>(mDecodeState->k_pages) + layer_offset);
                        auto* vp = reinterpret_cast<nv_bfloat16*>(
                            reinterpret_cast<std::byte*>(mDecodeState->v_pages) + layer_offset);
                        kv_cache_append_paged_bf16(kp, vp, qkv_out.get<nv_bfloat16>(),
                            mDecodeState->seq_lens_gpu, mDecodeState->block_table_gpu, mDecodeState->block_table_stride,
                            mDecodeState->page_block_size, ds_batch, Hq, ds_Hkv, ds_Hs, mRunState.MainStream);
                    }
                }
            }
        }
    }
}

void CompiledExecutor::dispatch_mrope_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);

    // Allow inputs: [d_out, freqs, position_ids] or legacy [d_out, qkv, freqs, position_ids]
    const bool has_qkv = op.inputs.size() == 4;
    Tensor& freqs = resolve_tensor(op.inputs[has_qkv ? 2 : 1]);
    Tensor& pos_ids = resolve_tensor(op.inputs[has_qkv ? 3 : 2]);

    Tensor* d_qkv_ptr = &ensure_output_tensor(op.outputs[0]);
    if (d_qkv_ptr->Rank == 0 || d_qkv_ptr->nelem() != d_out.nelem() || d_qkv_ptr->DType != d_out.DType) {
        std::vector<long> shape(d_out.Sizes.begin(), d_out.Sizes.begin() + d_out.Rank);
        Tensor tmp = mRunState.temp_alloc(d_out.DType, shape, "mrope_backward_d_qkv");
        mTemps.push_back(tmp);
        d_qkv_ptr = &mTemps.back();
    }
    Tensor& d_qkv = *d_qkv_ptr;

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    const int qkv_channels = Hs * (Hq + 2 * Hkv);

    Tensor d_out_view = (d_out.Rank == 4) ? view_tensor(d_out, {mB, mT, static_cast<long>(qkv_channels)}) : d_out;
    Tensor d_qkv_view = (d_qkv.Rank == 4) ? view_tensor(d_qkv, {mB, mT, static_cast<long>(qkv_channels)}) : d_qkv;

    if (d_qkv_view.Data != d_out_view.Data) {
        const std::size_t bytes = static_cast<std::size_t>(d_out_view.nelem()) * get_dtype_size(d_out_view.DType);
        CUDA_CHECK(cudaMemcpyAsync(d_qkv_view.Data, d_out_view.Data, bytes,
                                   cudaMemcpyDeviceToDevice, mRunState.MainStream));
    }

    int rotary_dim = op.attrs.rotary_dim;
    const int* pos_ptr = reinterpret_cast<int*>(pos_ids.Data);
    int pos_planes = 1;
    if (pos_ids.Rank == 3) {
        pos_planes = static_cast<int>(pos_ids.Sizes[0]);
        if (pos_planes == 4) {
            pos_ptr += static_cast<int>(mB * mT);
            pos_planes = 3;
        }
    }

    mrope_backward(d_qkv_view, d_qkv_view, freqs, pos_ptr, pos_planes,
                   op.attrs.mrope_section[0], op.attrs.mrope_section[1], op.attrs.mrope_section[2],
                   nullptr, static_cast<int>(mB), static_cast<int>(mT), Hq, Hkv, Hs, rotary_dim,
                   mRunState.MainStream);

    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        store_tensor(op.outputs[0], d_qkv);
    }
}

}  // namespace dsl
