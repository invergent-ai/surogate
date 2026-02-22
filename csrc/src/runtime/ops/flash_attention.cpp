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
#include "utilities/dtype.h"
#include "utilities/utils.h"

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
    // Training path
    // -----------------------------------------------------------------------
    int window_size = op.attrs.window_size;
    if (window_size <= 0 && mConfig.use_sliding_window && mConfig.is_sliding_layer(layer_idx)) {
        window_size = mConfig.sliding_window_size;
    }

    if (mCuSeqlensGpu) {
        // Document-level masking: Flash Attention varlen path.
        // Write LSE directly into the pre-allocated output tensor (persists to backward).
        // For B=1 (GRPO packed sequences), (B, Hq, T) and (Hq, total_q) have identical layout.
        attention_forward_flash_varlen(
            out.get<nv_bfloat16>(), lse.get<float>(), qkv.get<nv_bfloat16>(),
            mCuSeqlensGpu, mNumDocs, mMaxDocSeqlen, mTotalDocTokens,
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
            sinks_cast = mRunState.temp_alloc(out.DType, {static_cast<long>(Hq)});
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
    Tensor& d_qkv = ensure_output_tensor(op.outputs[0]);

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

    if (mCuSeqlensGpu) {
        // Document-level masking: Flash Attention varlen backward
        const int Hs_rounded = Hs <= 128 ? ((Hs + 31) / 32) * 32 : ((Hs + 63) / 64) * 64;
        const long padded_total = static_cast<long>(mTotalDocTokens) + 128L * static_cast<long>(mNumDocs);
        const long dq_accum_elems = padded_total * static_cast<long>(Hq) * static_cast<long>(Hs_rounded);
        const long dsoftmax_elems = static_cast<long>(Hq) * padded_total;
        Tensor dq_accum = mRunState.temp_alloc(ETensorDType::FP32, {dq_accum_elems});
        Tensor dsoftmax = mRunState.temp_alloc(ETensorDType::FP32, {dsoftmax_elems});
        mTemps.push_back(dq_accum);
        mTemps.push_back(dsoftmax);

        // GQA expanded dk/dv buffers: flash backward writes dK/dV with Hq head
        // indices, but interleaved buffer only has Hkv slots. Allocate separate
        // (total_q, Hq, HS) buffers when Hq != Hkv.
        nv_bfloat16* dk_exp_ptr = nullptr;
        nv_bfloat16* dv_exp_ptr = nullptr;
        Tensor dk_expanded, dv_expanded;
        if (Hq != Hkv) {
            const long exp_elems = static_cast<long>(mTotalDocTokens) * static_cast<long>(Hq) * static_cast<long>(Hs);
            dk_expanded = mRunState.temp_alloc(ETensorDType::BF16, {exp_elems});
            dv_expanded = mRunState.temp_alloc(ETensorDType::BF16, {exp_elems});
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
            mCuSeqlensGpu, dq_accum.get<float>(), dsoftmax.get<float>(),
            dk_exp_ptr, dv_exp_ptr,
            mNumDocs, mMaxDocSeqlen, mTotalDocTokens,
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
            Tensor out_f32 = mRunState.temp_alloc(ETensorDType::FP32, shape_vec(out));
            Tensor d_out_f32 = mRunState.temp_alloc(ETensorDType::FP32, shape_vec(d_out));
            Tensor qkv_f32 = mRunState.temp_alloc(ETensorDType::FP32, shape_vec(qkv));
            Tensor d_qkv_f32 = mRunState.temp_alloc(ETensorDType::FP32, shape_vec(d_qkv));
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
        if (!sinks) {
            throw std::runtime_error("flash_attention_backward: d_sinks requested but sinks input missing");
        }
        Tensor& d_sinks_out = ensure_output_tensor(op.outputs[1]);

        bool accumulate = mAccumulateTensors.count(op.outputs[1].name) > 0;
        if (!accumulate && !op.outputs[1].name.empty()) {
            if (auto base = base_param_from_grad(op.outputs[1].name)) {
                accumulate = mAccumulateTensors.count("d_" + *base) > 0;
            }
        }

        Tensor d_sinks_f32 = mRunState.temp_alloc(ETensorDType::FP32, {static_cast<long>(Hq)});
        mTemps.push_back(d_sinks_f32);
        fill_zero(d_sinks_f32, mRunState.MainStream);

        Tensor* sinks_use = sinks;
        Tensor sinks_cast;
        if (sinks->DType != out.DType) {
            sinks_cast = mRunState.temp_alloc(out.DType, {static_cast<long>(Hq)});
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
            Tensor d_sinks_bf16 = mRunState.temp_alloc(ETensorDType::BF16, {static_cast<long>(Hq)});
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
}


}  // namespace dsl
