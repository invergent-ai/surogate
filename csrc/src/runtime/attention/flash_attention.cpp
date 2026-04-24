// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

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

#include "kernels/kernels.h"
#include "runtime/attention/attention_backend.h"
#include "runtime/attention/attention_kernels.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/buffer_plan.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/executor/graph_executor_helpers.h"
#include "runtime/executor/graph_executor_utils.h"
#include "runtime/executor/op_registry.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"
#include "utilities/stack.h"
#include "utilities/utils.h"

namespace dsl {
namespace {

// ---------------------------------------------------------------------------
// Helpers shared by forward and backward dispatch.
// ---------------------------------------------------------------------------

int resolve_layer_idx(const CompiledOp& op, int explicit_idx_input) {
    int layer_idx = op.attrs.layer_idx;
    if (layer_idx < 0 && !op.inputs.empty() && op.inputs[0].layer_idx >= 0) {
        layer_idx = op.inputs[0].layer_idx;
    }
    if (layer_idx < 0 && explicit_idx_input >= 0 && static_cast<int>(op.inputs.size()) > explicit_idx_input) {
        std::string field;
        parse_block_param(op.inputs[explicit_idx_input].name, layer_idx, field);
    }
    return layer_idx;
}

/// Cast a 1-D sinks tensor to ``target_dtype`` if it doesn't already
/// match. Returns a reference to either the original sinks tensor or a
/// newly-allocated temp (tracked via ``temps`` for op-end release).
Tensor& cast_sinks_to(Tensor& sinks,
                      ETensorDType target_dtype,
                      int Hq,
                      std::vector<Tensor>& temps,
                      DslRunState& run_state,
                      cudaStream_t stream,
                      Tensor& scratch /* out */) {
    if (sinks.DType == target_dtype) {
        return sinks;
    }
    scratch = run_state.temp_alloc(target_dtype, {static_cast<long>(Hq)}, "flash_attention_sinks_cast");
    temps.push_back(scratch);
    if (target_dtype == ETensorDType::BF16) {
        convert_dtype(scratch.get<nv_bfloat16>(), sinks.get<float>(), sinks.nelem(), stream);
    } else if (target_dtype == ETensorDType::FP32) {
        convert_dtype(scratch.get<float>(), sinks.get<nv_bfloat16>(), sinks.nelem(), stream);
    } else {
        throw std::logic_error("flash_attention: unsupported sinks dtype conversion");
    }
    return scratch;
}

void apply_sinks_forward(Tensor& out,
                         Tensor& lse,
                         Tensor& sinks,
                         int B,
                         int T,
                         int Hq,
                         int Hs,
                         std::vector<Tensor>& temps,
                         DslRunState& run_state,
                         cudaStream_t stream) {
    Tensor scratch;
    Tensor& sinks_use = cast_sinks_to(sinks, out.DType, Hq, temps, run_state, stream, scratch);
    if (out.DType == ETensorDType::BF16) {
        attention_apply_sinks(out.get<nv_bfloat16>(),
                              lse.get<float>(),
                              sinks_use.get<nv_bfloat16>(),
                              B,
                              T,
                              Hq,
                              Hs,
                              stream);
    } else if (out.DType == ETensorDType::FP32) {
        attention_apply_sinks(out.get<float>(), lse.get<float>(), sinks_use.get<float>(), B, T, Hq, Hs, stream);
    } else {
        throw std::logic_error("flash_attention: unsupported output dtype");
    }
}

std::string tensor_shape_debug(const Tensor& t) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < t.Rank; ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << t.Sizes[i];
    }
    oss << "]";
    return oss.str();
}

void debug_dump_runtime_tensor(const std::string& name, const Tensor& t, const char* dump_dir) {
    if (!dump_dir || !*dump_dir || !t.Data || t.nelem() <= 0) {
        return;
    }
    std::string safe;
    safe.reserve(name.size());
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.') {
            safe += c;
        } else {
            safe += '_';
        }
    }
    const std::size_t nelem = static_cast<std::size_t>(t.nelem());
    std::vector<float> host_data(nelem);
    if (t.DType == ETensorDType::FP32) {
        CUDA_CHECK(cudaMemcpy(host_data.data(), t.Data, nelem * sizeof(float), cudaMemcpyDeviceToHost));
    } else if (t.DType == ETensorDType::BF16) {
        std::vector<uint16_t> bf16_data(nelem);
        CUDA_CHECK(cudaMemcpy(bf16_data.data(), t.Data, nelem * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < nelem; ++i) {
            uint32_t bits = static_cast<uint32_t>(bf16_data[i]) << 16;
            float val;
            std::memcpy(&val, &bits, sizeof(float));
            host_data[i] = val;
        }
    } else {
        return;
    }
    const std::string bin_path = std::string(dump_dir) + "/" + safe + ".bin";
    FILE* bin_f = std::fopen(bin_path.c_str(), "wb");
    if (bin_f) {
        std::fwrite(host_data.data(), sizeof(float), nelem, bin_f);
        std::fclose(bin_f);
    }
    const std::string json_path = std::string(dump_dir) + "/" + safe + ".json";
    FILE* json_f = std::fopen(json_path.c_str(), "w");
    if (json_f) {
        std::fprintf(json_f, "{\"name\": \"%s\", \"dtype\": \"float32\", \"shape\": [", name.c_str());
        for (int i = 0; i < t.Rank; ++i) {
            std::fprintf(json_f, "%ld%s", t.Sizes[i], (i + 1 < t.Rank) ? ", " : "");
        }
        std::fprintf(json_f, "]}\n");
        std::fclose(json_f);
    }
}

bool should_dump_attention_layer(int layer_idx) {
    const char* layer_env = std::getenv("SUROGATE_DEBUG_ATTENTION_DUMP_LAYER");
    if (!layer_env || !*layer_env) {
        return false;
    }
    char* end = nullptr;
    const long requested = std::strtol(layer_env, &end, 10);
    if (end == layer_env) {
        return false;
    }
    return layer_idx == static_cast<int>(requested);
}

}  // namespace

void CompiledExecutor::dispatch_flash_attention(const CompiledOp& op) {
    Tensor& qkv = resolve_tensor(op.inputs[0]);
    Tensor* sinks = nullptr;
    if (op.inputs.size() > 1 && !op.inputs[1].name.empty()) {
        sinks = &resolve_tensor(op.inputs[1]);
    }

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = derive_head_size(qkv, Hq, Hkv, static_cast<int>(mConfig.head_size()));

    const Tensor& out_candidate = ensure_output_tensor(op.outputs[0]);
    const Tensor& lse_candidate = ensure_output_tensor(op.outputs[1]);
    // Always derive output shape from actual QKV dims (activation layout may
    // have wrong shape for hybrid models with per-block-type head dimensions).
    const std::vector<long> out_shape = {mB, mT, Hq, Hs};
    const std::vector<long> lse_shape =
        !op.outputs[1].shape.empty()
            ? op.outputs[1].shape
            : std::vector<long>(lse_candidate.Sizes.begin(), lse_candidate.Sizes.begin() + lse_candidate.Rank);
    Tensor out = ensure_output_tensor_or_persistent(out_candidate,
                                                    mRunState,
                                                    mMoeSavedBuffers,
                                                    mMoeSavedSizes,
                                                    op.op_id + "." + op.outputs[0].name + ".out",
                                                    qkv.DType,
                                                    out_shape,
                                                    "flash_attention");
    Tensor lse = ensure_output_tensor_or_persistent(lse_candidate,
                                                    mRunState,
                                                    mMoeSavedBuffers,
                                                    mMoeSavedSizes,
                                                    op.op_id + "." + op.outputs[1].name + ".lse",
                                                    ETensorDType::FP32,
                                                    lse_shape,
                                                    "flash_attention");

    const int layer_idx = resolve_layer_idx(op, /*explicit_idx_input=*/0);

    int window_size = op.attrs.window_size;
    if (window_size <= 0 && mConfig.use_sliding_window && mConfig.is_sliding_layer(layer_idx)) {
        window_size = mConfig.sliding_window_size;
    }

    AttentionParams params;
    params.B = static_cast<int>(mB);
    params.T = static_cast<int>(mT);
    params.Hq = Hq;
    params.Hkv = Hkv;
    params.Hs = Hs;
    params.window_size = window_size;
    params.softmax_scale = op.attrs.softmax_scale;
    params.causal = true;
    params.dtype = qkv.DType;
    params.qkv = qkv;
    params.out = out;
    params.lse = lse;
    params.sinks = sinks;
    params.cu_seqlens = mCuSeqlensGpu;
    params.cu_seqlens_cpu = mCuSeqlensCpu;
    params.num_docs = mNumDocs;
    params.max_doc_seqlen = mMaxDocSeqlen;
    params.total_doc_tokens = mTotalDocTokens;
    params.stream = mRunState.MainStream;
    params.cudnn_handle = mRunState.CudnnHandle;
    params.cublas_handle = mRunState.cublas_handle();
    params.cudnn_workspace = mRunState.scratch().cudnn_workspace;
    params.run_state = &mRunState;
    params.temps = &mTemps;
    params.sm_version = mRunState.DeviceProp.major * 10 + mRunState.DeviceProp.minor;

    AttentionBackend& backend = AttentionBackendRegistry::instance().select(params);
    if (env_enabled("SUROGATE_DEBUG_ATTENTION_RUNTIME")) {
        std::cerr << "[ATTN] dir=fwd"
                  << " layer=" << layer_idx << " backend=" << backend.name() << " window=" << window_size
                  << " Hq=" << Hq << " Hkv=" << Hkv << " Hs=" << Hs << " qkv_shape=" << tensor_shape_debug(qkv)
                  << " out_shape=" << tensor_shape_debug(out) << " lse_shape=" << tensor_shape_debug(lse)
                  << " packed=" << (mCuSeqlensGpu ? 1 : 0) << std::endl;
    }
    backend.forward(params);
    if (should_dump_attention_layer(layer_idx)) {
        const char* dump_dir = std::getenv("SUROGATE_DEBUG_DUMP_DIR");
        debug_dump_runtime_tensor(fmt::format("attn_live.layer{}.qkv", layer_idx), qkv, dump_dir);
        debug_dump_runtime_tensor(fmt::format("attn_live.layer{}.out", layer_idx), out, dump_dir);
        debug_dump_runtime_tensor(fmt::format("attn_live.layer{}.lse", layer_idx), lse, dump_dir);
    }

    if (sinks) {
        apply_sinks_forward(out,
                            lse,
                            *sinks,
                            static_cast<int>(mB),
                            static_cast<int>(mT),
                            Hq,
                            Hs,
                            mTemps,
                            mRunState,
                            mRunState.MainStream);
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

    const std::vector<long> d_qkv_shape(qkv.Sizes.begin(), qkv.Sizes.begin() + qkv.Rank);
    Tensor d_qkv = ensure_output_tensor_or_persistent(ensure_output_tensor(op.outputs[0]),
                                                      mRunState,
                                                      mMoeSavedBuffers,
                                                      mMoeSavedSizes,
                                                      op.op_id + "." + op.outputs[0].name + ".d_qkv",
                                                      d_out.DType,
                                                      d_qkv_shape,
                                                      "flash_attention_backward");
    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = derive_head_size(qkv, Hq, Hkv, static_cast<int>(mConfig.head_size()));

    const int layer_idx = resolve_layer_idx(op, /*explicit_idx_input=*/3);

    int window_size = op.attrs.window_size;
    if (window_size <= 0 && mConfig.use_sliding_window && mConfig.is_sliding_layer(layer_idx)) {
        window_size = mConfig.sliding_window_size;
    }

    AttentionParams params;
    params.B = static_cast<int>(mB);
    params.T = static_cast<int>(mT);
    params.Hq = Hq;
    params.Hkv = Hkv;
    params.Hs = Hs;
    params.window_size = window_size;
    params.softmax_scale = op.attrs.softmax_scale;
    params.causal = true;
    params.dtype = qkv.DType;
    params.qkv = qkv;
    params.out = out;
    params.lse = lse;
    params.sinks = sinks;
    params.d_out = d_out;
    params.d_qkv = d_qkv;
    params.cu_seqlens = mCuSeqlensGpu;
    params.cu_seqlens_cpu = mCuSeqlensCpu;
    params.num_docs = mNumDocs;
    params.max_doc_seqlen = mMaxDocSeqlen;
    params.total_doc_tokens = mTotalDocTokens;
    params.stream = mRunState.MainStream;
    params.cudnn_handle = mRunState.CudnnHandle;
    params.cublas_handle = mRunState.cublas_handle();
    params.cudnn_workspace = mRunState.scratch().cudnn_workspace;
    params.run_state = &mRunState;
    params.temps = &mTemps;
    // Packed sliding-window attention has shown unstable backward norms on the
    // non-deterministic FlashAttention varlen path. Route that case through
    // the deterministic kernel by default; the env flag still forces
    // deterministic mode for all varlen backward calls when desired.
    params.deterministic_bwd = (std::getenv("SUROGATE_FLASH_ATTN_VARLEN_BWD_DETERMINISTIC") != nullptr) ||
                               (mCuSeqlensGpu != nullptr && window_size > 0);
    params.attn_bwd_chunks = mOptions.AttBwdChunks;
    params.sm_version = mRunState.DeviceProp.major * 10 + mRunState.DeviceProp.minor;

    AttentionBackend& backend = AttentionBackendRegistry::instance().select(params);
    if (env_enabled("SUROGATE_DEBUG_ATTENTION_RUNTIME")) {
        std::cerr << "[ATTN] dir=bwd"
                  << " layer=" << layer_idx << " backend=" << backend.name() << " window=" << window_size
                  << " Hq=" << Hq << " Hkv=" << Hkv << " Hs=" << Hs << " qkv_shape=" << tensor_shape_debug(qkv)
                  << " out_shape=" << tensor_shape_debug(out) << " d_out_shape=" << tensor_shape_debug(d_out)
                  << " lse_shape=" << tensor_shape_debug(lse) << " packed=" << (mCuSeqlensGpu ? 1 : 0) << std::endl;
    }
    if (should_dump_attention_layer(layer_idx)) {
        const char* dump_dir = std::getenv("SUROGATE_DEBUG_DUMP_DIR");
        debug_dump_runtime_tensor(fmt::format("attn_live.layer{}.d_out", layer_idx), d_out, dump_dir);
        debug_dump_runtime_tensor(fmt::format("attn_live.layer{}.out_bwd", layer_idx), out, dump_dir);
        debug_dump_runtime_tensor(fmt::format("attn_live.layer{}.lse_bwd", layer_idx), lse, dump_dir);
        debug_dump_runtime_tensor(fmt::format("attn_live.layer{}.qkv_bwd", layer_idx), qkv, dump_dir);
    }
    backend.backward(params);

    // Under EP with dp_size=1, attention activations are replicated
    // across ranks, so the backward must produce identical dQKV per rank.
    // Flash-attention varlen GQA backward emits rank-divergent K/V
    // gradients with identical inputs; the backend advertises that via
    // ``gqa_backward_is_rank_divergent()`` and we recover here.
    if (mComm && mComm->ep_enabled() && (mComm->dp_size() == 1) && (Hq != Hkv) &&
        backend.gqa_backward_is_rank_divergent()) {
        mComm->all_reduce_avg(d_qkv, mRunState.MainStream);
    }

    // Compute ∂L/∂sinks only when the sinks param is live. In QLoRA mode
    // sinks may be offloaded, and on non-sinks-LoRA configs it isn't
    // trained — skip the gradient computation in both cases.
    const bool wants_d_sinks = op.outputs.size() > 1 && !op.outputs[1].name.empty() && sinks && sinks->Data;
    if (wants_d_sinks) {
        const std::vector<long> d_sinks_shape =
            !op.outputs[1].shape.empty() ? op.outputs[1].shape : std::vector<long>{static_cast<long>(Hq)};
        Tensor d_sinks_out = ensure_output_tensor_or_persistent(ensure_output_tensor(op.outputs[1]),
                                                                mRunState,
                                                                mMoeSavedBuffers,
                                                                mMoeSavedSizes,
                                                                op.op_id + "." + op.outputs[1].name + ".d_sinks",
                                                                op.outputs[1].dtype,
                                                                d_sinks_shape,
                                                                "flash_attention_backward");

        bool accumulate = mAccumulateTensors.count(op.outputs[1].name) > 0;
        if (!accumulate && mCurrentGraph) {
            if (auto base = base_param_from_grad_kind(op.outputs[1].tensor_id, *mCurrentGraph)) {
                accumulate = mAccumulateTensors.count("d_" + *base) > 0;
            }
        }

        Tensor d_sinks_f32 =
            mRunState.temp_alloc(ETensorDType::FP32, {static_cast<long>(Hq)}, "flash_attention_d_sinks_f32");
        mTemps.push_back(d_sinks_f32);
        fill_zero(d_sinks_f32, mRunState.MainStream);

        Tensor sinks_scratch;
        Tensor& sinks_use =
            cast_sinks_to(*sinks, out.DType, Hq, mTemps, mRunState, mRunState.MainStream, sinks_scratch);

        if (out.DType == ETensorDType::BF16) {
            attention_sinks_backward(d_sinks_f32.get<float>(),
                                     out.get<nv_bfloat16>(),
                                     d_out.get<nv_bfloat16>(),
                                     lse.get<float>(),
                                     sinks_use.get<nv_bfloat16>(),
                                     static_cast<int>(mB),
                                     static_cast<int>(mT),
                                     Hq,
                                     Hs,
                                     mRunState.MainStream);
        } else if (out.DType == ETensorDType::FP32) {
            attention_sinks_backward(d_sinks_f32.get<float>(),
                                     out.get<float>(),
                                     d_out.get<float>(),
                                     lse.get<float>(),
                                     sinks_use.get<float>(),
                                     static_cast<int>(mB),
                                     static_cast<int>(mT),
                                     Hq,
                                     Hs,
                                     mRunState.MainStream);
        } else {
            throw std::logic_error("flash_attention_backward: unsupported output dtype for sinks grad");
        }

        if (d_sinks_out.DType == ETensorDType::FP32) {
            if (accumulate) {
                vector_add_sr(d_sinks_out,
                              d_sinks_out,
                              d_sinks_f32,
                              1.0f,
                              static_cast<long>(d_sinks_out.nelem()),
                              0,
                              mRunState.MainStream);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(d_sinks_out.Data,
                                           d_sinks_f32.Data,
                                           d_sinks_out.bytes(),
                                           cudaMemcpyDeviceToDevice,
                                           mRunState.MainStream));
            }
        } else if (d_sinks_out.DType == ETensorDType::BF16) {
            Tensor d_sinks_bf16 =
                mRunState.temp_alloc(ETensorDType::BF16, {static_cast<long>(Hq)}, "flash_attention_d_sinks_bf16");
            mTemps.push_back(d_sinks_bf16);
            convert_dtype(d_sinks_bf16.get<nv_bfloat16>(),
                          d_sinks_f32.get<float>(),
                          d_sinks_f32.nelem(),
                          mRunState.MainStream);
            if (accumulate) {
                vector_add_sr(d_sinks_out,
                              d_sinks_out,
                              d_sinks_bf16,
                              1.0f,
                              static_cast<long>(d_sinks_out.nelem()),
                              0,
                              mRunState.MainStream);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(d_sinks_out.Data,
                                           d_sinks_bf16.Data,
                                           d_sinks_out.bytes(),
                                           cudaMemcpyDeviceToDevice,
                                           mRunState.MainStream));
            }
        } else {
            throw std::logic_error("flash_attention_backward: unsupported d_sinks dtype");
        }
        store_tensor(op.outputs[1], d_sinks_out);
    }

    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        store_tensor(op.outputs[0], d_qkv);
    }
}

namespace {

// -----------------------------------------------------------------------------
// FlashAttention backward rule
// Forward: out, lse = flash_attention(qkv)
// Backward: d_qkv = flash_attention_backward(d_out, out, lse, qkv)
// -----------------------------------------------------------------------------
std::vector<Operation> flash_attention_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    std::string out = fwd.outputs.empty() ? "out" : fwd.outputs[0];
    std::string lse = fwd.outputs.size() > 1 ? fwd.outputs[1] : out + "_lse";
    std::string qkv = fwd.inputs.empty() ? "qkv" : fwd.inputs[0];

    AttrMap attrs = copy_attrs(fwd.attrs, {"causal", "softmax_scale", "window_size"});

    std::vector<std::string> inputs = {ctx.d_output, saved_ref(out), saved_ref(lse), saved_ref(qkv)};
    bool has_sinks = (fwd.inputs.size() > 1 && !fwd.inputs[1].empty());
    if (has_sinks) {
        inputs.push_back(saved_ref(fwd.inputs[1]));
    }

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");
    if (has_sinks) {
        outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");
    }

    ops.push_back(make_operation("flash_attention_backward_" + std::to_string(ctx.op_counter++),
                                 "flash_attention_backward",
                                 "flash_attention_backward",
                                 inputs,
                                 outputs,
                                 attrs));

    return ops;
}

}  // namespace

// Upper bound on stack bytes for the FlashAttention varlen backward backend
// (the most memory-hungry of the available backends — cuDNN uses a pre-
// allocated persistent workspace, custom/sdpa are smaller). If dispatch
// picks a cheaper backend at runtime, this over-provisions — accepted
// because the bound must be safe for every backend selection.
//
// Per-layer temps allocated by FlashVarlenAttention::backward:
//   dq_accum   FP32  padded_total * Hq * Hs_rounded  (× splits, assumed 1)
//   dsoftmax   FP32  padded_total * Hq
//   dk_exp     BF16  total_q * Hq * Hs               (GQA only, Hq != Hkv)
//   dv_exp     BF16  total_q * Hq * Hs               (GQA only, Hq != Hkv)
// where padded_total = total_q + 128 * num_docs, Hs_rounded rounds Hs up
// to 32 (Hs <= 128) or 64 (Hs > 128). Without doc-masking at plan time we
// assume num_docs = 1; the 128-token padding is small relative to B*T.
//
// Inputs from autodiff: [d_out, out, lse, qkv, sinks?]
//   qkv shape = [B, T, Hq + 2*Hkv, Hs]
/// Forward stack bound covering the mem_eff backend's temps
/// (flash-varlen's forward has no appreciable stack temps). Called at
/// compile time regardless of which backend will be selected at run
/// time — the sizing must fit the mem_eff case so the stack arena is
/// large enough.
// Max head_dim across layers. BufferPlan's ``AttnDim`` is the max of
// per-layer ``Hq * head_size``; dividing by ``Hq`` recovers the max
// head_size.
static long max_head_size(const BufferPlan& plan) {
    if (plan.Hq <= 0) return 0;
    return plan.AttnDim / plan.Hq;
}

long flash_attention_forward_stack_bound(const CompiledOp& op, const BufferPlan& plan) {
    (void)op;
    const long Hq = plan.Hq;
    const long Hs = max_head_size(plan);
    if (Hq <= 0 || Hs <= 0) return 0;
    constexpr long FP32 = 4;
    const long total_q = plan.B * plan.T;

    long bytes = 0;
    bytes += align_stack_bytes(total_q * Hq * Hs * FP32);  // mem_eff output_accum
    bytes += align_stack_bytes(total_q * Hq * FP32);       // mem_eff lse scratch (upper bound: lse_dim<=T)
    return bytes;
}

long flash_attention_backward_stack_bound(const CompiledOp& op, const BufferPlan& plan) {
    (void)op;
    // Fall back to plan dims: at plan time, op.inputs[3].shape is often
    // empty (shape env not yet resolved). We want the worst-case size
    // across all attention layers anyway since the arena gets sized to
    // the peak.
    const long Hq = plan.Hq;
    const long Hkv = plan.Hkv;
    const long Hs = max_head_size(plan);
    if (Hq <= 0 || Hs <= 0) return 0;

    constexpr long BF16 = 2, FP32 = 4;
    // Conservative: num_docs = 1 (pre-packing). +128 padding per doc.
    const long total_q = plan.B * plan.T;
    const long padded_total = total_q + 128;
    const long Hs_rounded = Hs <= 128 ? ((Hs + 31) / 32) * 32 : ((Hs + 63) / 64) * 64;

    // Flash-varlen's temp footprint.
    long flash_varlen_bytes = 0;
    flash_varlen_bytes += align_stack_bytes(padded_total * Hq * Hs_rounded * FP32);  // dq_accum
    flash_varlen_bytes += align_stack_bytes(padded_total * Hq * FP32);               // dsoftmax
    if (Hq != Hkv) {
        flash_varlen_bytes += align_stack_bytes(total_q * Hq * Hs * BF16);  // dk_expanded
        flash_varlen_bytes += align_stack_bytes(total_q * Hq * Hs * BF16);  // dv_expanded
    }

    // mem_eff backward's temp footprint — fires for Hs > 256 layers
    // where flash-varlen's Hs<=256 cap rejects (and, after priority
    // upgrade, for many MHA/MQA varlen cases too). The mem_eff backend
    // allocates via rs.temp_alloc so these end up in the same stack
    // arena flash-varlen targets.
    //
    // Fwd: lse_scratch + output_accum.
    // Bwd: lse_scratch + delta + delta_dense + workspace; MQA also
    //      allocates 3 x [B*T, Hq, Hs] BF16 partial-grad scratches.
    long mem_eff_bytes = 0;
    const long lse_scratch_bytes = total_q * Hq * FP32;            // pessimistic: lse_dim <= T
    mem_eff_bytes += align_stack_bytes(lse_scratch_bytes);         // lse scratch
    mem_eff_bytes += align_stack_bytes(lse_scratch_bytes);         // delta (kernel layout)
    mem_eff_bytes += align_stack_bytes(total_q * Hq * FP32);       // delta_dense
    mem_eff_bytes += align_stack_bytes(total_q * Hq * Hs * FP32);  // kernel workspace upper bound
    const bool mem_eff_might_be_mqa = (Hkv != Hq);
    if (mem_eff_might_be_mqa) {
        mem_eff_bytes += 3 * align_stack_bytes(total_q * Hq * Hs * BF16);  // dq/dk/dv partials
    }

    // The backend is selected at runtime; compile-time sizing must
    // accommodate whichever fires. Take the max.
    return std::max(flash_varlen_bytes, mem_eff_bytes);
}

}  // namespace dsl

REGISTER_AUTODIFF("flash_attention", ::dsl::flash_attention_backward);
REGISTER_AUTODIFF("flash_attention_qkv", ::dsl::flash_attention_backward);
REGISTER_STACK_BOUND("flash_attention", FlashAttention, ::dsl::flash_attention_forward_stack_bound);
REGISTER_STACK_BOUND("flash_attention_backward", FlashAttentionBackward, ::dsl::flash_attention_backward_stack_bound);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// FlashAttention
// ------------------------------------------------------------------------
const int _flash_attention_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "flash_attention";
    sig.min_inputs = 1;
    sig.max_inputs = 2;
    sig.min_outputs = 2;
    sig.max_outputs = 2;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        const auto& qkv = inputs[0];
        const auto& out = outputs[0];

        if (qkv.size() < 3 || qkv.size() > 4) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "flash_attention: qkv has rank " << qkv.size() << " but expected 3 or 4";
            err.message = oss.str();
            return std::make_optional(err);
        }

        if (inputs.size() > 1) {
            const auto& sinks = inputs[1];
            if (!sinks.empty() && sinks.size() != 1) {
                ShapeValidationError err;
                err.message = "flash_attention: sinks must be 1D [Hq]";
                return std::make_optional(err);
            }
        }

        if (out.empty()) {
            return std::optional<ShapeValidationError>();
        }

        if (out.size() != qkv.size()) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "flash_attention: out has rank " << out.size() << " but expected " << qkv.size() << " (same as qkv)";
            err.message = oss.str();
            return std::make_optional(err);
        }

        if (qkv.size() >= 2 && out.size() >= 2) {
            if (qkv[0] != out[0] || qkv[1] != out[1]) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "flash_attention: out batch dims [" << out[0] << "," << out[1] << "] don't match qkv [" << qkv[0]
                    << "," << qkv[1] << "]";
                err.message = oss.str();
                return std::make_optional(err);
            }
        }

        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

// ------------------------------------------------------------------------
// FlashAttentionBackward
// ------------------------------------------------------------------------
const int _flash_attention_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "flash_attention_backward";
    sig.min_inputs = 4;
    sig.max_inputs = 5;
    sig.min_outputs = 1;
    sig.max_outputs = 2;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        const auto& d_out = inputs[0];
        const auto& qkv = inputs[3];
        const auto& d_qkv = outputs[0];

        if (!d_qkv.empty()) {
            if (auto err = validators::check_same_numel(d_qkv, qkv, "d_qkv", "qkv", "flash_attention_backward")) {
                return err;
            }
        }

        if (!qkv.empty() && (qkv.size() < 3 || qkv.size() > 4)) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "flash_attention_backward: qkv has rank " << qkv.size() << " but expected 3 or 4";
            err.message = oss.str();
            return std::make_optional(err);
        }

        if (inputs.size() > 4 && outputs.size() > 1) {
            const auto& d_sinks = outputs[1];
            if (!d_sinks.empty() && d_sinks.size() != 1) {
                ShapeValidationError err;
                err.message = "flash_attention_backward: d_sinks must be 1D [Hq]";
                return std::make_optional(err);
            }
        }

        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

}  // namespace
}  // namespace shape_checker
}  // namespace dsl
