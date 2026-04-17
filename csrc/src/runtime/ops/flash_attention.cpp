#include "runtime/executor/compiled_ops.h"

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

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/executor/graph_executor_helpers.h"
#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace dsl {
void CompiledExecutor::dispatch_flash_attention(const CompiledOp& op) {
    Tensor& qkv = resolve_tensor(op.inputs[0]);
    Tensor* sinks = nullptr;
    if (op.inputs.size() > 1 && !op.inputs[1].name.empty()) {
        sinks = &resolve_tensor(op.inputs[1]);
    }

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = derive_head_size(qkv, Hq, Hkv, static_cast<int>(mConfig.head_size()));
    const int H  = Hq + 2 * Hkv;
    const Tensor& out_candidate = ensure_output_tensor(op.outputs[0]);
    const Tensor& lse_candidate = ensure_output_tensor(op.outputs[1]);
    // Always derive output shape from actual QKV dims (activation layout may
    // have wrong shape for hybrid models with per-block-type head dimensions).
    const std::vector<long> out_shape = {mB, mT, Hq, Hs};
    const std::vector<long> lse_shape =
        !op.outputs[1].shape.empty()
            ? op.outputs[1].shape
            : std::vector<long>(lse_candidate.Sizes.begin(), lse_candidate.Sizes.begin() + lse_candidate.Rank);
    Tensor out = ensure_output_tensor_or_persistent(
        out_candidate,
        mRunState, mMoeSavedBuffers, mMoeSavedSizes,
        op.op_id + "." + op.outputs[0].name + ".out",
        qkv.DType, out_shape, "flash_attention");
    Tensor lse = ensure_output_tensor_or_persistent(
        lse_candidate,
        mRunState, mMoeSavedBuffers, mMoeSavedSizes,
        op.op_id + "." + op.outputs[1].name + ".lse",
        ETensorDType::FP32, lse_shape, "flash_attention");

    int layer_idx = op.attrs.layer_idx;
    if (layer_idx < 0 && !op.inputs.empty() && op.inputs[0].layer_idx >= 0) {
        layer_idx = op.inputs[0].layer_idx;
    }
    if (layer_idx < 0 && !op.inputs.empty()) {
        std::string field;
        parse_block_param(op.inputs[0].name, layer_idx, field);
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

    if (use_varlen && Hs <= 256) {
        if (out.DType != ETensorDType::BF16 || qkv.DType != ETensorDType::BF16) {
            throw std::logic_error("flash_attention: varlen path currently requires BF16 tensors");
        }
        // Document-level masking: Flash Attention varlen path (max head_dim=256).
        // Pass window_size so sliding-window layers (e.g. Gemma4 sliding blocks)
        // get local attention instead of silently falling through to full causal.
        attention_forward_flash_varlen(
            out.get<nv_bfloat16>(), lse.get<float>(), qkv.get<nv_bfloat16>(),
            cu_seqlens_ptr, num_docs, max_doc_seqlen, total_doc_tokens,
            Hq, Hkv, Hs, mRunState.MainStream, op.attrs.softmax_scale,
            window_size > 0 ? window_size : 0);
    } else if (window_size > 0) {
        // Sliding window attention: custom kernel (supports head_dim <= 128)
        attention_forward_custom(out, lse, qkv,
                                 static_cast<int>(mB), static_cast<int>(mT),
                                 Hq, Hkv, Hs, window_size, mRunState.MainStream,
                                 op.attrs.softmax_scale);
    } else if (!cudnn_supported) {
        // head_dim > 128: Use cuBLAS matmul-based attention (SDPA math equivalent).
        attention_forward_matmul(out, lse, qkv,
                                 static_cast<int>(mB), static_cast<int>(mT),
                                 Hq, Hkv, Hs, mRunState.cublas_handle(), mRunState.MainStream,
                                 op.attrs.softmax_scale);
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

    store_tensor(op.outputs[0], out);
    store_tensor(op.outputs[1], lse);
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
    const long qkv_nelem = static_cast<long>(qkv.nelem());
    const std::vector<long> d_qkv_shape(
        qkv.Sizes.begin(), qkv.Sizes.begin() + qkv.Rank);
    Tensor d_qkv = ensure_output_tensor_or_persistent(
        ensure_output_tensor(op.outputs[0]),
        mRunState, mMoeSavedBuffers, mMoeSavedSizes,
        op.op_id + "." + op.outputs[0].name + ".d_qkv",
        d_out.DType, d_qkv_shape, "flash_attention_backward");
    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = derive_head_size(qkv, Hq, Hkv, static_cast<int>(mConfig.head_size()));
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
    Tensor dk_expanded, dv_expanded;
    nv_bfloat16* dk_exp_ptr = nullptr;
    nv_bfloat16* dv_exp_ptr = nullptr;
    if (use_varlen && cu_seqlens_ptr == nullptr) {
        num_docs = static_cast<int>(mB);
        max_doc_seqlen = static_cast<int>(mT);
        total_doc_tokens = num_docs * max_doc_seqlen;
        generated_cu_seqlens = mRunState.temp_alloc(ETensorDType::INT32, {static_cast<long>(num_docs + 1)}, "generated_cu_seqlens");
        mTemps.push_back(generated_cu_seqlens);
        fill_dense_cu_seqlens(generated_cu_seqlens.get<int32_t>(), num_docs, max_doc_seqlen, mRunState.MainStream);
        cu_seqlens_ptr = generated_cu_seqlens.get<int32_t>();
    }

    // head_dim > 256 without sliding window: use matmul backward (matches forward_matmul)
    if (use_varlen && Hs > 256) {
        attention_backward_matmul(d_qkv, lse, out, d_out, qkv,
                                  static_cast<int>(mB), static_cast<int>(mT),
                                  Hq, Hkv, Hs, mRunState.cublas_handle(), mRunState.MainStream,
                                  op.attrs.softmax_scale);
    } else if (use_varlen) {
        if (out.DType != ETensorDType::BF16 ||
            d_out.DType != ETensorDType::BF16 ||
            qkv.DType != ETensorDType::BF16 ||
            d_qkv.DType != ETensorDType::BF16) {
            throw std::logic_error("flash_attention_backward: varlen path currently requires BF16 tensors");
        }
        const bool deterministic_bwd =
            (std::getenv("SUROGATE_FLASH_ATTN_VARLEN_BWD_DETERMINISTIC") != nullptr);
        // Document-level masking: Flash Attention varlen backward
        const int Hs_rounded = Hs <= 128 ? ((Hs + 31) / 32) * 32 : ((Hs + 63) / 64) * 64;
        const long padded_total = static_cast<long>(total_doc_tokens) + 128L * static_cast<long>(num_docs);
        const long dq_accum_stride = padded_total * static_cast<long>(Hq) * static_cast<long>(Hs_rounded);
        const int split_den = std::max(1, num_docs * Hq);
        const int dq_accum_splits =
            deterministic_bwd
                ? std::max(1, (mRunState.DeviceProp.multiProcessorCount + split_den - 1) / split_den)
                : 1;
        const long dq_accum_elems = dq_accum_stride * static_cast<long>(dq_accum_splits);
        const long dsoftmax_elems = static_cast<long>(Hq) * padded_total;
        Tensor dq_accum = mRunState.temp_alloc(ETensorDType::FP32, {dq_accum_elems}, "flash_attention_dq_accum");
        Tensor dsoftmax = mRunState.temp_alloc(ETensorDType::FP32, {dsoftmax_elems}, "flash_attention_dsoftmax");
        mTemps.push_back(dq_accum);
        mTemps.push_back(dsoftmax);
        // Upstream FlashAttention zeroes softmax_d for varlen backward regardless of
        // deterministic mode. The kernel treats it as an accumulation/work buffer.
        fill_zero(dsoftmax, mRunState.MainStream);
        if (deterministic_bwd) {
            // FlashAttention deterministic backward uses one dQ accumulation buffer per split.
            // The upstream API allocates nsplits * stride and zeros it before launch.
            CUDA_CHECK(cudaMemsetAsync(
                dq_accum.Data, 0, static_cast<size_t>(dq_accum_elems) * sizeof(float),
                mRunState.MainStream));
        }

        // GQA expanded dk/dv buffers: flash backward writes dK/dV with Hq head
        // indices, but interleaved buffer only has Hkv slots. Allocate separate
        // (total_q, Hq, HS) buffers when Hq != Hkv.
        if (Hq != Hkv) {
            const long exp_elems = static_cast<long>(total_doc_tokens) * static_cast<long>(Hq) * static_cast<long>(Hs);
            dk_expanded = mRunState.temp_alloc(ETensorDType::BF16, {exp_elems}, "flash_attention_dk_expanded");
            dv_expanded = mRunState.temp_alloc(ETensorDType::BF16, {exp_elems}, "flash_attention_dv_expanded");
            mTemps.push_back(dk_expanded);
            mTemps.push_back(dv_expanded);
            dk_exp_ptr = dk_expanded.get<nv_bfloat16>();
            dv_exp_ptr = dv_expanded.get<nv_bfloat16>();
            // Flash varlen GQA backward may leave portions of the expanded KV grads untouched
            // before the explicit Hq->Hkv reduce/scatter. Under EP, rank-local temp allocator
            // history differs, so stale BF16 contents here become rank-specific K/V noise.
            fill_zero(dk_expanded, mRunState.MainStream);
            fill_zero(dv_expanded, mRunState.MainStream);
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
            Hq, Hkv, Hs, deterministic_bwd, mRunState.MainStream,
            op.attrs.softmax_scale,
            window_size > 0 ? window_size : 0);
    } else if (window_size > 0) {
        if (out.DType == ETensorDType::FP32) {
            attention_backward_custom(d_qkv, lse, out, d_out, qkv,
                                      static_cast<int>(mB), static_cast<int>(mT),
                                      Hq, Hkv, Hs, window_size, mRunState.MainStream,
                                      op.attrs.softmax_scale);
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
                                      Hq, Hkv, Hs, window_size, mRunState.MainStream,
                                      op.attrs.softmax_scale);
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

    const bool sync_flash_attn_bwd_out =
        mComm && mComm->ep_enabled() && (mComm->dp_size() == 1) &&
        use_varlen && (Hq != Hkv);
    if (sync_flash_attn_bwd_out) {
        // Under EP with dp_size=1, these attention activations are replicated across ranks.
        // Flash-attention backward on the shared path must therefore produce identical dQKV.
        // Varlen GQA backward currently emits rank-divergent K/V gradients even with identical
        // inputs, so average the replicated gradient here to restore the intended shared value.
        mComm->all_reduce_avg(d_qkv, mRunState.MainStream);
    }

    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        if (!sinks || !sinks->Data) {
            // Sinks parameter not available (e.g., offloaded in QLoRA mode or not a LoRA target).
            // Skip sinks gradient computation — it's unused when sinks isn't being trained.
            goto skip_sinks;
        }
        const std::vector<long> d_sinks_shape =
            !op.outputs[1].shape.empty() ? op.outputs[1].shape : std::vector<long>{static_cast<long>(Hq)};
        Tensor d_sinks_out = ensure_output_tensor_or_persistent(
            ensure_output_tensor(op.outputs[1]),
            mRunState, mMoeSavedBuffers, mMoeSavedSizes,
            op.op_id + "." + op.outputs[1].name + ".d_sinks",
            op.outputs[1].dtype, d_sinks_shape, "flash_attention_backward");

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
        store_tensor(op.outputs[1], d_sinks_out);
    }
    skip_sinks:

    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        store_tensor(op.outputs[0], d_qkv);
    }
}


}  // namespace dsl
