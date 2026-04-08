// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL model execution functions (forward, backward, validation, run state allocation).

#include "runtime/dsl/dsl_model.h"
#include "runtime/dsl/dsl_model_internal.h"
#include "runtime/dsl/dsl_runtime.h"
#include "runtime/dsl/graph_executor.h"
#include "runtime/dsl/graph_executor_helpers.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string_view>

#include "kernels/kernels.h"
#include "runtime/core/forward_hooks.h"
#include "runtime/core/backward_hooks.h"
#include "runtime/core/fp8_scaling_state.h"
#include "runtime/lora/lora_utils.h"
#include "runtime/lora/lora_model_utils.h"
#include "runtime/optimizers/flash_adamw_8bit.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"

#include <iostream>
#include <optional>
#include <vector>

namespace dsl {

namespace {

bool contains_ci(std::string_view haystack, std::string_view needle) {
    std::string h(haystack);
    std::string n(needle);
    std::transform(h.begin(), h.end(), h.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::transform(n.begin(), n.end(), n.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return h.find(n) != std::string::npos;
}

bool is_qwen3_5_model(const modules::ModelConfig& cfg) {
    return contains_ci(cfg.ModelTypeName, "qwen3_5") ||
           contains_ci(cfg.ModelTypeName, "qwen3.5") ||
           contains_ci(cfg.ArchitectureName, "qwen3_5") ||
           contains_ci(cfg.ArchitectureName, "qwen3.5");
}

} // namespace

// ============================================================================
// Document masking: detect document boundaries from position_ids resets
// ============================================================================

struct DocMaskingInfo {
    std::vector<std::int32_t> cu_seqlens;  // (num_docs + 1,) cumulative offsets
    int num_docs;
    int max_seqlen;
    int total_q;
};

/// Scan position_ids for non-consecutive transitions to detect document
/// boundaries in packed sequences. Returns nullopt if no boundaries found (i.e.
/// single contiguous sequence per batch element — standard SFT/PT).
///
/// When \p mrope is true the position_ids come from a multimodal RoPE model
/// (e.g. Qwen3-VL / Qwen3.5-VL).  In that case image/video tokens share the
/// same temporal position (the value stays constant across visual tokens within
/// one image), so `curr - prev != 1` would create hundreds of false document
/// boundaries.  We detect boundaries only by strict *decreases* in position
/// (resets), which correctly identifies sample-packing boundaries while
/// ignoring the flat temporal positions inside image regions.
static std::optional<DocMaskingInfo> compute_doc_masking(
        const std::int32_t* position_ids, int B, int T, bool mrope = false) {
    if (!position_ids) return std::nullopt;

    std::vector<std::int32_t> cu_seqlens;
    cu_seqlens.push_back(0);
    int max_seqlen = 0;
    bool has_boundaries = false;

    for (int b = 0; b < B; ++b) {
        int doc_start = b * T;
        for (int t = 1; t < T; ++t) {
            int idx = b * T + t;
            const int prev = position_ids[idx - 1];
            const int curr = position_ids[idx];
            const bool is_boundary = mrope ? (curr < prev) : (curr - prev != 1);
            if (is_boundary) {
                // Document boundary: position_ids are not strictly consecutive.
                // Mirrors HF packed-sequence detection (diff != 1).
                int doc_len = (b * T + t) - doc_start;
                if (doc_len > 0) {
                    cu_seqlens.push_back(cu_seqlens.back() + doc_len);
                    max_seqlen = std::max(max_seqlen, doc_len);
                }
                doc_start = b * T + t;
                has_boundaries = true;
            }
        }
        // Last document in this batch element
        int last_len = (b + 1) * T - doc_start;
        if (last_len > 0) {
            cu_seqlens.push_back(cu_seqlens.back() + last_len);
            max_seqlen = std::max(max_seqlen, last_len);
        }
    }

    if (!has_boundaries) return std::nullopt;

    int num_docs = static_cast<int>(cu_seqlens.size()) - 1;
    int total_q = cu_seqlens.back();
    return DocMaskingInfo{std::move(cu_seqlens), num_docs, max_seqlen, total_q};
}

void DslModel::build_lora_name_tables() {
    const int num_layers = static_cast<int>(mModelConfig.NumLayers);
    mLoRALn1Names.resize(num_layers);
    mLoRALn2Names.resize(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        const std::string idx = std::to_string(i);
        std::string ln1 = "blocks[" + idx + "].ln1_weight";
        if (!mParams->has(ln1)) {
            ln1 = "blocks[" + idx + "].norm_weight";
        }
        mLoRALn1Names[i] = std::move(ln1);

        std::string ln2 = "blocks[" + idx + "].ln2_weight";
        if (!mParams->has(ln2)) {
            ln2 = "blocks[" + idx + "].norm_weight";
        }
        mLoRALn2Names[i] = std::move(ln2);
    }
}

void DslModel::forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::forward called before allocate_run_state()");
    }

    // Detect document boundaries for packed sequence masking.
    // Position_ids with per-document resets (e.g. [0,1,2, 0,1, 0,1,2,3])
    // trigger Flash Attention varlen with cu_seqlens instead of cuDNN full-attention.
    mDocMaskingActive = false;
    if (mOptions.DocMasking && position_ids.Data && position_ids.Device == -1) {
        const auto* pos_ptr = reinterpret_cast<const std::int32_t*>(position_ids.Data);
        const int B = static_cast<int>(inputs.Sizes[0]);
        const int T = static_cast<int>(inputs.Sizes[1]);
        const bool mrope = mModelConfig.Rope.is_multimodal();
        auto doc_info = compute_doc_masking(pos_ptr, B, T, mrope);
        if (doc_info) {
            mExecutor->set_doc_masking(doc_info->cu_seqlens.data(),
                                        doc_info->num_docs,
                                        doc_info->max_seqlen,
                                        doc_info->total_q);
            mDocMaskingActive = true;
        }
    }

    if (!lora_enabled()) {
        mExecutor->forward(inputs, position_ids, comm, micro_step);
        return;
    }

    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);
    if (qlora_enabled() && micro_step == 0 && mQLoRAProvider) {
        mQLoRAProvider->invalidate_cache();
    }

    // Store micro_step for dropout seed computation (needed by backward pass)
    mLoRARunState->micro_step = micro_step;
    mLoRARunState->is_training = true;

    auto hook = [this, micro_step](int layer_idx, cudaStream_t stream, modules::ForwardHookPoint point, void* context) {
        (void)context;
        const auto& cfg = mModelConfig;
        auto& rs = *mRunState;
        const int B = (int)rs.B;
        const int T = (int)rs.T;
        const int C = (int)cfg.HiddenSize;
        const int D = cfg.get_intermediate_size(layer_idx);
        const int Hq = (int)cfg.NumQueryHeads;
        const int Hkv = (int)cfg.NumKeyValHeads;
        const int Hs = (int)cfg.head_size();
        const int rank = mLoRAConfig->rank;
        const float scaling = mLoRAConfig->scaling();
        const float dropout = mLoRAConfig->dropout;
        const bool is_training = mLoRARunState->is_training;
        const bool gated_mlp = modules::is_gated_activation(cfg.activation_type);
        const bool use_qwen35_attention_lora = is_qwen3_5_model(cfg);

        // Helper to compute unique dropout seed per layer and projection type
        auto get_dropout_seed = [&](int proj_type) -> unsigned int {
            // seed = base_seed + layer_idx * 1000000 + proj_type * 100000 + micro_step * 10000
            return mLoRARunState->dropout_base_seed
                   + static_cast<unsigned int>(layer_idx) * 1000000u
                   + static_cast<unsigned int>(proj_type) * 100000u
                   + static_cast<unsigned int>(micro_step) * 10000u;
        };

        auto& acts = rs.simplified_acts(layer_idx);
        auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);

        switch (point) {
            case modules::ForwardHookPoint::AfterQKVProjection: {
                if (use_qwen35_attention_lora) break;
                // Projection types: 0=Q, 1=K, 2=V, 3=O, 4=Up, 5=Gate, 6=Down
                if (lora_block.attention.q.has_value()) {
                    modules::detail::apply_lora_contribution(acts.qkv, 0, acts.ln1, lora_block.attention.q.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(0), is_training,
                                                    B * T, C, Hq * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.attention.k.has_value()) {                   
                    modules::detail::apply_lora_contribution(acts.qkv, Hq * Hs, acts.ln1, lora_block.attention.k.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(1), is_training,
                                                    B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.attention.v.has_value()) {
                    modules::detail::apply_lora_contribution(acts.qkv, (Hq + Hkv) * Hs, acts.ln1, lora_block.attention.v.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(2), is_training,
                                                    B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterAttnOutProjection: {
                if (use_qwen35_attention_lora) break;
                if (lora_block.attention.o.has_value()) {
                    modules::detail::apply_lora_contribution(acts.att_out, 0, acts.att, lora_block.attention.o.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(3), is_training,
                                                    B * T, Hq * Hs, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterMLPUpProjection: {
                Tensor& ln2_input = acts.ln2.Data ? acts.ln2 : acts.ln1;
                if (lora_block.mlp.up.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_up, 0, ln2_input, lora_block.mlp.up.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(4), is_training,
                                                    B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (gated_mlp && lora_block.mlp.gate.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_up, D, ln2_input, lora_block.mlp.gate.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(5), is_training,
                                                    B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterMLPDownProjection: {
                if (lora_block.mlp.down.has_value()) {
                    Tensor down_input = acts.swiglu;
                    Tensor down_input_tmp{};
                    bool free_down_input_tmp = false;
                    if (!down_input.Data && acts.mlp_up.Data) {
                        down_input_tmp = rs.temp_alloc(acts.mlp_down.DType, {B * T, D}, "down_input_tmp");
                        Tensor down_input_view = down_input_tmp;
                        down_input_view.Rank = 3;
                        down_input_view.Sizes[0] = B;
                        down_input_view.Sizes[1] = T;
                        down_input_view.Sizes[2] = D;
                        for (int i = 3; i < MAX_TENSOR_DIM; ++i) down_input_view.Sizes[i] = 1;
                        switch (cfg.activation_type) {
                            case modules::ActivationType::SwiGLU:
                            case modules::ActivationType::GeGLU:
                                swiglu_forward(down_input_view, acts.mlp_up, nullptr, B, T, D, stream);
                                break;
                            case modules::ActivationType::ReLU2: {
                                const long N = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(D);
                                relu2_forward(down_input_view, acts.mlp_up, N, stream);
                            } break;
                            case modules::ActivationType::SiLU: {
                                const long N = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(D);
                                silu_forward(down_input_view, acts.mlp_up, N, stream);
                            } break;
                            case modules::ActivationType::GeLU: {
                                const long N = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(D);
                                gelu_forward(down_input_view, acts.mlp_up, N, stream);
                            } break;
                            default:
                                throw std::runtime_error("unsupported activation type for LoRA MLPDown replay");
                        }
                        down_input = down_input_view;
                        free_down_input_tmp = true;
                    }
                    if (down_input.Data) {
                        try {
                            modules::detail::apply_lora_contribution(acts.mlp_down, 0, down_input, lora_block.mlp.down.value(),
                                                            mLoRARunState->intermediate, mLoRARunState->slice,
                                                            scaling, dropout, get_dropout_seed(6), is_training,
                                                            B * T, D, C, rank,
                                                            rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                        } catch (...) {
                            if (free_down_input_tmp) rs.temp_free(down_input_tmp);
                            throw;
                        }
                    }
                    if (free_down_input_tmp) rs.temp_free(down_input_tmp);
                }
            } break;
            default:
                break;
        }
    };

    mExecutor->forward_with_hook(inputs, position_ids, comm, micro_step, hook);
}

float DslModel::validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::validate called before allocate_run_state()");
    }

    if (!lora_enabled()) {
        return mExecutor->validate(inputs, position_ids, targets, comm, micro_step);
    }

    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

    auto hook = [this](int layer_idx, cudaStream_t stream, modules::ForwardHookPoint point, void* context) {
        (void)context;
        const auto& cfg = mModelConfig;
        auto& rs = *mRunState;
        const int B = (int)rs.B;
        const int T = (int)rs.T;
        const int C = (int)cfg.HiddenSize;
        const int D = cfg.get_intermediate_size(layer_idx);
        const int Hq = (int)cfg.NumQueryHeads;
        const int Hkv = (int)cfg.NumKeyValHeads;
        const int Hs = (int)cfg.head_size();
        const int rank = mLoRAConfig->rank;
        const float scaling = mLoRAConfig->scaling();
        const bool gated_mlp = modules::is_gated_activation(cfg.activation_type);
        const bool use_qwen35_attention_lora = is_qwen3_5_model(cfg);

        auto& acts = rs.simplified_acts(layer_idx);
        auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);

        switch (point) {
            case modules::ForwardHookPoint::AfterQKVProjection: {
                if (use_qwen35_attention_lora) break;
                // Validation: no dropout (is_training=false)
                if (lora_block.attention.q.has_value()) {
                    modules::detail::apply_lora_contribution(acts.qkv, 0, acts.ln1, lora_block.attention.q.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, Hq * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.attention.k.has_value()) {
                    modules::detail::apply_lora_contribution(acts.qkv, Hq * Hs, acts.ln1, lora_block.attention.k.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.attention.v.has_value()) {
                    modules::detail::apply_lora_contribution(acts.qkv, (Hq + Hkv) * Hs, acts.ln1, lora_block.attention.v.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterAttnOutProjection: {
                if (use_qwen35_attention_lora) break;
                if (lora_block.attention.o.has_value()) {
                    modules::detail::apply_lora_contribution(acts.att_out, 0, acts.att, lora_block.attention.o.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, Hq * Hs, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterMLPUpProjection: {
                Tensor& ln2_input = acts.ln2.Data ? acts.ln2 : acts.ln1;
                if (lora_block.mlp.up.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_up, 0, ln2_input, lora_block.mlp.up.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (gated_mlp && lora_block.mlp.gate.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_up, D, ln2_input, lora_block.mlp.gate.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterMLPDownProjection: {
                if (lora_block.mlp.down.has_value()) {
                    Tensor down_input = acts.swiglu;
                    Tensor down_input_tmp{};
                    bool free_down_input_tmp = false;
                    if (!down_input.Data && acts.mlp_up.Data) {
                        down_input_tmp = rs.temp_alloc(acts.mlp_down.DType, {B * T, D}, "down_input_tmp");
                        Tensor down_input_view = down_input_tmp;
                        down_input_view.Rank = 3;
                        down_input_view.Sizes[0] = B;
                        down_input_view.Sizes[1] = T;
                        down_input_view.Sizes[2] = D;
                        for (int i = 3; i < MAX_TENSOR_DIM; ++i) down_input_view.Sizes[i] = 1;
                        switch (cfg.activation_type) {
                            case modules::ActivationType::SwiGLU:
                            case modules::ActivationType::GeGLU:
                                swiglu_forward(down_input_view, acts.mlp_up, nullptr, B, T, D, stream);
                                break;
                            case modules::ActivationType::ReLU2: {
                                const long N = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(D);
                                relu2_forward(down_input_view, acts.mlp_up, N, stream);
                            } break;
                            case modules::ActivationType::SiLU: {
                                const long N = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(D);
                                silu_forward(down_input_view, acts.mlp_up, N, stream);
                            } break;
                            case modules::ActivationType::GeLU: {
                                const long N = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(D);
                                gelu_forward(down_input_view, acts.mlp_up, N, stream);
                            } break;
                            default:
                                throw std::runtime_error("unsupported activation type for LoRA MLPDown replay");
                        }
                        down_input = down_input_view;
                        free_down_input_tmp = true;
                    }
                    if (down_input.Data) {
                        try {
                            modules::detail::apply_lora_contribution(acts.mlp_down, 0, down_input, lora_block.mlp.down.value(),
                                                            mLoRARunState->intermediate, mLoRARunState->slice,
                                                            scaling, 0.0f, 0, false,
                                                            B * T, D, C, rank,
                                                            rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                        } catch (...) {
                            if (free_down_input_tmp) rs.temp_free(down_input_tmp);
                            throw;
                        }
                    }
                    if (free_down_input_tmp) rs.temp_free(down_input_tmp);
                }
            } break;
            default:
                break;
        }
    };

    return mExecutor->validate_with_hook(inputs, position_ids, targets, comm, micro_step, hook);
}

void DslModel::backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::backward called before allocate_run_state()");
    }
    mUseTokenScale = true;

    if (!lora_enabled()) {
        mExecutor->backward(inputs, targets, comm, grad_accum_steps, micro_step);
        if (mDocMaskingActive) {
            mExecutor->clear_doc_masking();
            mDocMaskingActive = false;
        }
        return;
    }

    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;

    // Lazily build pre-formatted weight name tables (once, avoids per-layer string construction).
    if (mLoRALn1Names.empty()) {
        build_lora_name_tables();
    }

    mLoRAGrads->start_micro_step(main_stream, micro_step, grad_accum_steps);

    auto hook = [this, &comm](int layer_idx, bool accumulate, cudaStream_t stream, modules::BackwardHookPoint point, void* context) {
        (void)context;
        const auto& cfg = mModelConfig;
        auto& rs = *mRunState;
        const int B = (int)rs.B;
        const int T = (int)rs.T;
        const int C = (int)cfg.HiddenSize;
        const int D = cfg.get_intermediate_size(layer_idx);
        const int Hq = (int)cfg.NumQueryHeads;
        const int Hkv = (int)cfg.NumKeyValHeads;
        const int Hs = (int)cfg.head_size();
        const int rank = mLoRAConfig->rank;
        const float dropout = mLoRAConfig->dropout;
        const bool is_training = mLoRARunState->is_training;
        const int micro_step = mLoRARunState->micro_step;
        const bool gated_mlp = modules::is_gated_activation(cfg.activation_type);
        const bool use_qwen35_attention_lora = is_qwen3_5_model(cfg);

        // Helper to compute unique dropout seed per layer and projection type
        auto get_dropout_seed = [&](int proj_type) -> unsigned int {
            return mLoRARunState->dropout_base_seed
                   + static_cast<unsigned int>(layer_idx) * 1000000u
                   + static_cast<unsigned int>(proj_type) * 100000u
                   + static_cast<unsigned int>(micro_step) * 10000u;
        };

        auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);

        switch (point) {
            case modules::BackwardHookPoint::AfterMLPDownBackward: {
                if (!lora_block.mlp.down.has_value()) break;

                bool lora_accum = false;
                auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                lora_accum = lora_accum || accumulate;
                if (!lora_grads.mlp.down.has_value()) break;

                auto& a = rs.simplified_acts(layer_idx);
                auto& da = rs.simplified_grads(layer_idx);
                modules::detail::backward_lora_layer(
                    lora_grads.mlp.down->A, lora_grads.mlp.down->B,
                    da.d_swiglu,
                    da.d_res_ffn, 0,
                    a.swiglu,
                    lora_block.mlp.down->A, lora_block.mlp.down->B,
                    mLoRAConfig->scaling(),
                    dropout, get_dropout_seed(6), is_training,
                    mLoRARunState->intermediate, mLoRARunState->slice,
                    B * T, D, C, rank, lora_accum,
                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
            } break;
            case modules::BackwardHookPoint::AfterMLPUpBackward: {
                bool lora_accum = false;
                auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                lora_accum = lora_accum || accumulate;

                auto& a = rs.simplified_acts(layer_idx);
                auto& da = rs.simplified_grads(layer_idx);
                Tensor& d_ln2 = da.d_ln2.Data ? da.d_ln2 : da.d_ln1;

                // Get ln2 input: either from stored activation or recompute from residual stream
                // LN2 input is residual_att = res_ffn[L-1] + att_out[L]
                Tensor ln2_input;
                if (mOptions.recompute_enabled()) {
                    if (mLoRARunState && mLoRARunState->recompute_ln.Data) {
                        // Use pre-built name table (avoids per-layer string construction).
                        Tensor& ln2_weight = mParams->get(mLoRALn2Names[layer_idx]);
                        // Prefer using recomputed residual_att from simplified_acts.
                        // Fallback: Standard blocks use res_ffn[L-1]; hybrid blocks use res_in[L].
                        Tensor ln2_residual;
                        if (a.residual_att.Data) {
                            ln2_residual = a.residual_att;
                        } else if (mModelConfig.architecture == modules::ArchitectureType::Hybrid) {
                            if (rs.has_residual_offloading()) {
                                rs.fetch_residual(layer_idx, rs.side_stream());
                            }
                            ln2_residual = rs.get_residual(layer_idx, stream);
                        } else if (layer_idx == 0) {
                            ln2_residual = rs.non_block_activations().encoded;
                        } else {
                            // Ensure residual is fetched when offloading is enabled
                            if (rs.has_residual_offloading()) {
                                rs.fetch_residual(layer_idx - 1, rs.side_stream());
                            }
                            ln2_residual = rs.get_residual(layer_idx - 1, stream);
                        }
                        ln2_input = recompute_lora_rmsnorm(*mLoRARunState, ln2_residual, ln2_weight,
                                                          mModelConfig.RmsNormEps, B, T, C, stream);
                    } else {
                        ln2_input = a.ln2.Data ? a.ln2 : a.ln1;
                    }
                } else {
                    ln2_input = a.ln2.Data ? a.ln2 : a.ln1;
                }

                // Prepare gradient tensors (use empty tensor if projection not enabled)
                Tensor dA_up{}, dB_up{}, dA_gate{}, dB_gate{};
                modules::LoRALayerWeights<Tensor> lora_up{}, lora_gate{};

                if (gated_mlp) {
                    if (lora_block.mlp.up.has_value() && lora_grads.mlp.up.has_value()) {
                        dA_up = lora_grads.mlp.up->A;
                        dB_up = lora_grads.mlp.up->B;
                        lora_up = *lora_block.mlp.up;
                    }
                    if (lora_block.mlp.gate.has_value() && lora_grads.mlp.gate.has_value()) {
                        dA_gate = lora_grads.mlp.gate->A;
                        dB_gate = lora_grads.mlp.gate->B;
                        lora_gate = *lora_block.mlp.gate;
                    }

                    if (!dA_up.Data && !dA_gate.Data) break;

                    // Projection types: 4=Up, 5=Gate
                    modules::detail::backward_lora_mlp_up_gate_fused(
                        dA_up, dB_up,
                        dA_gate, dB_gate,
                        d_ln2,
                        da.d_mlp_up,
                        ln2_input,
                        lora_up, lora_gate,
                        mLoRAConfig->scaling(),
                        dropout, get_dropout_seed(4), get_dropout_seed(5), is_training,
                        B * T,
                        C,
                        D,
                        rank,
                        lora_accum,
                        mLoRARunState->intermediate,
                        mLoRARunState->intermediate2,
                        mLoRARunState->slice,
                        rs.CublasLtHandle,
                        rs.CuBlasWorkspace,
                        stream);
                } else {
                    if (!lora_block.mlp.up.has_value() || !lora_grads.mlp.up.has_value()) break;
                    dA_up = lora_grads.mlp.up->A;
                    dB_up = lora_grads.mlp.up->B;
                    lora_up = *lora_block.mlp.up;

                    // Projection type: 4=Up
                    const unsigned int dropout_seed = get_dropout_seed(4);
                    modules::detail::backward_lora_layer(
                        dA_up, dB_up,
                        d_ln2,
                        da.d_mlp_up, 0,
                        ln2_input,
                        lora_up.A, lora_up.B,
                        mLoRAConfig->scaling(),
                        dropout, dropout_seed, is_training,
                        mLoRARunState->intermediate, mLoRARunState->slice,
                        B * T, C, D, rank, lora_accum,
                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::BackwardHookPoint::AfterAttnOutBackward: {
                if (use_qwen35_attention_lora) break;
                if (!lora_block.attention.o.has_value()) break;

                bool lora_accum = false;
                auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                lora_accum = lora_accum || accumulate;
                if (!lora_grads.attention.o.has_value()) break;

                auto& a = rs.simplified_acts(layer_idx);
                auto& da = rs.simplified_grads(layer_idx);

                // Projection type 3 = O
                const unsigned int dropout_seed = get_dropout_seed(3);

                modules::detail::backward_lora_layer(
                    lora_grads.attention.o->A, lora_grads.attention.o->B,
                    da.d_att,
                    da.d_att_out, 0,
                    a.att,
                    lora_block.attention.o->A, lora_block.attention.o->B,
                    mLoRAConfig->scaling(),
                    dropout, dropout_seed, is_training,
                    mLoRARunState->intermediate, mLoRARunState->slice,
                    B * T, Hq * Hs, C, rank, lora_accum,
                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
            } break;
            case modules::BackwardHookPoint::AfterQKVBackward: {
                if (use_qwen35_attention_lora) break;
                bool lora_accum = false;
                auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                lora_accum = lora_accum || accumulate;

                auto& a = rs.simplified_acts(layer_idx);
                auto& da = rs.simplified_grads(layer_idx);

                // Get ln1 input: either from stored activation or recompute from residual
                // Standard blocks: LN1 input is res_ffn[L-1] (output of previous layer)
                // Hybrid blocks: LN1 input is res_in[L] (stored at index L by the block itself)
                Tensor ln1_input;
                if (mOptions.recompute_enabled()) {
                    if (mLoRARunState && mLoRARunState->recompute_ln.Data) {
                        // Use pre-built name table (avoids per-layer string construction).
                        Tensor& ln1_weight = mParams->get(mLoRALn1Names[layer_idx]);
                        Tensor ln1_residual;
                        if (mModelConfig.architecture == modules::ArchitectureType::Hybrid) {
                            // Hybrid blocks store their own LN input as res_in[L] at index L
                            if (rs.has_residual_offloading()) {
                                rs.fetch_residual(layer_idx, rs.side_stream());
                            }
                            ln1_residual = rs.get_residual(layer_idx, stream);
                        } else if (layer_idx == 0) {
                            ln1_residual = rs.non_block_activations().encoded;
                        } else {
                            // Ensure residual is fetched when offloading is enabled
                            if (rs.has_residual_offloading()) {
                                rs.fetch_residual(layer_idx - 1, rs.side_stream());
                            }
                            ln1_residual = rs.get_residual(layer_idx - 1, stream);
                        }
                        ln1_input = recompute_lora_rmsnorm(*mLoRARunState, ln1_residual, ln1_weight,
                                                          mModelConfig.RmsNormEps, B, T, C, stream);
                    } else {
                        ln1_input = a.ln1;
                    }
                } else {
                    ln1_input = a.ln1;
                }

                // Prepare gradient tensors (use empty tensor if projection not enabled)
                Tensor dA_q{}, dB_q{}, dA_k{}, dB_k{}, dA_v{}, dB_v{};
                modules::LoRALayerWeights<Tensor> lora_q{}, lora_k{}, lora_v{};

                if (lora_block.attention.q.has_value() && lora_grads.attention.q.has_value()) {
                    dA_q = lora_grads.attention.q->A;
                    dB_q = lora_grads.attention.q->B;
                    lora_q = *lora_block.attention.q;
                }
                if (lora_block.attention.k.has_value() && lora_grads.attention.k.has_value()) {
                    dA_k = lora_grads.attention.k->A;
                    dB_k = lora_grads.attention.k->B;
                    lora_k = *lora_block.attention.k;
                }
                if (lora_block.attention.v.has_value() && lora_grads.attention.v.has_value()) {
                    dA_v = lora_grads.attention.v->A;
                    dB_v = lora_grads.attention.v->B;
                    lora_v = *lora_block.attention.v;
                }

                if (!dA_q.Data && !dA_k.Data && !dA_v.Data) break;

                // Projection types: 0=Q, 1=K, 2=V
                modules::detail::backward_lora_qkv_fused(
                    dA_q, dB_q,
                    dA_k, dB_k,
                    dA_v, dB_v,
                    da.d_ln1,
                    da.d_qkv,
                    ln1_input,
                    lora_q, lora_k, lora_v,
                    mLoRAConfig->scaling(),
                    dropout, get_dropout_seed(0), get_dropout_seed(1), get_dropout_seed(2), is_training,
                    B * T,
                    C,
                    Hq * Hs,
                    Hkv * Hs,
                    rank,
                    lora_accum,
                    mLoRARunState->intermediate,
                    mLoRARunState->intermediate2,
                    mLoRARunState->slice,
                    rs.CublasLtHandle,
                    rs.CuBlasWorkspace,
                    stream);

                mLoRAGrads->notify_block(layer_idx, stream, comm);
            } break;
            default:
                break;
        }
    };

    mExecutor->backward_with_hook(inputs, targets, comm, grad_accum_steps, micro_step, hook);

    if (mDocMaskingActive) {
        mExecutor->clear_doc_masking();
        mDocMaskingActive = false;
    }

    mLoRAGrads->end_micro_step(main_stream, comm);
    // Extend the base-model BackwardDone event to include LoRA gradient reductions.
    internal::record_event_if_not_capturing(rs.BackwardDone, main_stream);
}

void DslModel::allocate_run_state(const RuntimeOptions& options, NCCLCommunicator& comm, int B, int T,
                                  bool allocate_optimizer) {
    if (!mAllocator) {
        mAllocator = std::make_shared<TensorAllocator>();
    }
    mOptions = options;
    if (qlora_enabled() && mQLoRAConfig.is_fp4()) {
        mOptions.UseCudaGraphs = false;
    }
    const std::size_t dummy_stack_bytes = 1024ULL * 1024ULL * 1024ULL * 1024ULL;  // 1TB dummy stack
    const ActivationLayoutIR* layout = mModule->activation_layout.has_value()
                                           ? &*mModule->activation_layout
                                           : nullptr;
    mRunState = std::make_unique<DslRunState>(mModelConfig, mRuntimeConfig, mOptions, B, T, mAllocator,
                                              lora_enabled(), mQLoRAConfig.is_prequantized(),
                                              dummy_stack_bytes, /*allocate_stack=*/false, layout);
    mRunState->WorldSize = comm.world_size();
    if (mParams) {
        mParams->set_default_stream(mRunState->MainStream);
        if (mQLoRAProvider) {
            mParams->set_qlora_provider(mQLoRAProvider.get());
        }
    }

    const long base_size = static_cast<long>(mRunState->Stack.max_utilization());
    long moe_extra = 0;
    if (mModelConfig.NumExperts > 0) {
        const long moe_intermediate = (mModelConfig.MoeIntermediateSize > 0)
                                          ? mModelConfig.MoeIntermediateSize
                                          : mModelConfig.IntermediateSize;
        const long hidden = mModelConfig.HiddenSize;
        const long num_experts = mModelConfig.NumExperts;
        const long top_k = std::max(1, mModelConfig.NumExpertsPerTok);
        const long dtype_bytes = 2;  // BF16 bytes (matches modular sizing heuristic)
        const long up_factor = mModelConfig.mlp_up_factor();
        const long expert_gate_up_tp = num_experts * up_factor * moe_intermediate * hidden * dtype_bytes;
        const long expert_down_tp = num_experts * moe_intermediate * hidden * dtype_bytes;
        const long permuted_tokens = 2L * B * T * top_k * hidden * dtype_bytes;
        // MoE activation backward (gpt_oss_moe_act_backward) allocates d_inp buffers
        // in intermediate dimension ({N, up_factor * intermediate}), not hidden dimension.
        // Account for this larger BT-proportional backward buffer.
        const long moe_bwd_act = 2L * B * T * top_k * up_factor * moe_intermediate * dtype_bytes;
        moe_extra = expert_gate_up_tp + expert_down_tp + permuted_tokens + moe_bwd_act;
    }
    ETensorDType act_dtype = mOptions.ModelType.value_or(mConfig->DType);
    if (is_fp8_dtype(act_dtype)) {
        act_dtype = ETensorDType::BF16;
    }
    const long dtype_bytes = static_cast<long>(get_dtype_size(act_dtype));
    const long BT = static_cast<long>(B) * static_cast<long>(T);
    const long C = mModelConfig.HiddenSize;
    const long QKV = mModelConfig.head_size() * (mModelConfig.NumQueryHeads + 2 * mModelConfig.NumKeyValHeads);
    const long MUp = static_cast<long>(mModelConfig.mlp_up_rows());
    const long extra_tmp = std::max({BT * C, BT * QKV, BT * MUp}) * dtype_bytes;
    long attn_fallback_bytes = 0;
    const bool lora_stack_tight = lora_enabled();
    const long safety_floor = lora_stack_tight ? (32L * 1024 * 1024) : (64L * 1024 * 1024);
    const long safety_bytes = std::max(safety_floor, base_size / 8);
    // The sizing simulation captures ~55% of actual backward peak (flash attention
    // backward workspace and accumulated temps are not fully modeled), so we use
    // base_multiplier=2 for both LoRA and full fine-tune.
    // CPU-RAM centric: tighter multiplier since stack resets at each layer boundary.
    const long base_multiplier = mOptions.CpuTraining ? 1L : 2L;
    long required_size = std::max(1024L * 1024,
                                  base_size * base_multiplier + moe_extra + safety_bytes + extra_tmp + attn_fallback_bytes);
    const long slack_bytes = mOptions.CpuTraining ? (128L * 1024 * 1024)
                           : lora_stack_tight     ? (256L * 1024 * 1024)
                           :                        (512L * 1024 * 1024);
    required_size += slack_bytes;  // extra slack for unmodeled temps
    const bool is_qwen3_hybrid_lora =
        lora_stack_tight &&
        (mModelConfig.architecture == modules::ArchitectureType::Hybrid) &&
        (mModelConfig.Architecture == PretrainedConfig::QWEN3);
    if (is_qwen3_hybrid_lora) {
        // Qwen3.5 hybrid blocks can hit an additional transient peak around SwiGLU
        // backward + LoRA hook temps that is not fully captured by the sizing simulation.
        // Reserve one extra BT x MUp buffer plus fixed margin.
        const long qwen35_swiglu_peak = BT * static_cast<long>(mModelConfig.mlp_up_rows()) * dtype_bytes;
        const long qwen35_extra_slack = std::max(128L * 1024 * 1024, qwen35_swiglu_peak + 64L * 1024 * 1024);
        required_size += qwen35_extra_slack;
    }
    long moe_stack_slack = 0;
    if (mModelConfig.NumExperts > 0) {
        moe_stack_slack = 2048L * 1024 * 1024;  // MoE backward temps can spike beyond simulated high-water mark
    }
    if (const char* env = std::getenv("SUROGATE_STACK_SLACK_MB")) {
        const long mb = std::max(0L, std::atol(env));
        moe_stack_slack = std::max(moe_stack_slack, mb * 1024 * 1024);
    }
    required_size += moe_stack_slack;
    long min_stack_base = mOptions.CpuTraining  ? (512L * 1024 * 1024)
                        : lora_stack_tight     ? (512L * 1024 * 1024)
                        :                        (3L * 1024 * 1024 * 1024);
    if (is_qwen3_hybrid_lora) {
        min_stack_base = std::max(min_stack_base, 1024L * 1024 * 1024);
    }
    if (mOptions.UseCudaGraphs) {
        // CUDA graph capture/replay requires stable temp addresses and tends to keep
        // additional transient allocations live during capture. Reserve extra headroom
        // to avoid capture-only stack OOMs.
        const long graph_extra_slack = lora_stack_tight ? (512L * 1024 * 1024) : (1024L * 1024 * 1024);
        required_size += graph_extra_slack;
        min_stack_base = std::max(min_stack_base, lora_stack_tight ? (1024L * 1024 * 1024)
                                                                    : (4L * 1024 * 1024 * 1024));
        if (is_qwen3_hybrid_lora) {
            // Qwen3.5 hybrid + LoRA + CUDA graphs retains additional transient tensors
            // during capture/replay that are not represented by simulation. Keep a
            // dedicated margin so mamba_gated_rmsnorm/swiglu backward peaks fit.
            required_size += 512L * 1024 * 1024;
            min_stack_base = std::max(min_stack_base, 1536L * 1024 * 1024);
        }
    }
    if (const char* env = std::getenv("SUROGATE_MIN_STACK_MB")) {
        const long mb = std::max(64L, std::atol(env));
        min_stack_base = mb * 1024 * 1024;
    }
    const long min_stack_bytes = min_stack_base + attn_fallback_bytes + moe_stack_slack;
    required_size = std::max(required_size, min_stack_bytes);  // Full fine-tune keeps 3GB+fallback; LoRA can use tighter floor.
    const auto high_mark = mRunState->Stack.get_high_mark();
    // DEBUG: Stack allocation size
    if (options.DebugMemoryBreakdown && comm.rank() == 0) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cerr << "[DEBUG-STACK] base_size=" << base_size/(1024*1024) << " MiB"
                  << ", required_size=" << required_size/(1024*1024) << " MiB"
                  << ", GPU used=" << (total_mem - free_mem)/(1024*1024) << " MiB"
                  << ", free=" << free_mem/(1024*1024) << " MiB" << std::endl;
    }
    Tensor stack_buffer = mAllocator->allocate(ETensorDType::BYTE, "dsl_stack", EAllocationType::ON_DEVICE, {required_size});
    mRunState->set_stack_buffer(std::move(stack_buffer), high_mark);
    comm.barrier();

    // Configure gradient manager for multi-GPU overlapped reduction
    if (mGrads && comm.world_size() > 1) {
        DslGradStoreConfig grad_config;
        grad_config.num_shards = comm.world_size();
        grad_config.shard_idx = comm.rank();
        grad_config.shard_gradients = mOptions.ShardGradients;  // ZeRO-2
        grad_config.use_all_to_all_reduce = mOptions.UseAllToAllReduce;
        grad_config.num_layers = mModelConfig.NumLayers;
        mGrads->configure(grad_config);
    }

    // CPU-RAM centric training: enable per-layer gradient streaming only for full fine-tune.
    // LoRA adapter gradients already live in the dedicated LoRA grad manager, so streaming
    // frozen base-model grads here is unnecessary and can interfere with the LoRA path.
    if (mGrads && mOptions.CpuTraining && !lora_enabled() && !mGrads->param_names().empty()) {
        // For single-GPU, configure() may not have been called yet (it requires world_size > 1).
        // Ensure the layer map is built by calling configure with minimal config.
        if (comm.world_size() == 1) {
            DslGradStoreConfig grad_config;
            grad_config.num_shards = 1;
            grad_config.shard_idx = 0;
            grad_config.num_layers = mModelConfig.NumLayers;
            mGrads->configure(grad_config);
        }
        mGrads->enable_streaming(*mParams);
    }

    GraphExecutorOptions exec_opts;
    exec_opts.auto_backward = true;
    exec_opts.debug_print_backward = false;
    mExecutor = std::make_unique<GraphExecutor>(*mModule, *mRunState, *mParams, *mGrads, mModelConfig, mOptions, exec_opts);
    if (!mRngState.empty()) {
        mExecutor->set_rng_state(mRngState);
    }

    // Estimate actual backward stack peak from the compiled graph and resize if needed.
    // The heuristic sizing above may underestimate for architectures with heavy backward
    // ops (e.g. Qwen3.5 gated delta rule) that allocate many internal temps on the stack.
    if (auto* exec = dynamic_cast<GraphExecutor*>(mExecutor.get())) {
        const long bwd_peak = exec->estimate_backward_stack_peak(B, T);
        if (options.DebugMemoryBreakdown && comm.rank() == 0) {
            std::cerr << "[DEBUG-STACK] Backward peak estimate=" << bwd_peak / (1024 * 1024) << " MiB"
                      << ", heuristic=" << required_size / (1024 * 1024) << " MiB" << std::endl;
        }
        if (bwd_peak > 0) {
            // Safety margin for dispatch-internal temps not in the graph.
            // cpu_training: tighter margin since stack resets at each layer boundary.
            const long safety = mOptions.CpuTraining
                ? std::max(64L * 1024 * 1024, bwd_peak / 8)
                : std::max(128L * 1024 * 1024, bwd_peak / 3);
            const long needed = bwd_peak + safety;
            if (needed > required_size) {
                if (options.DebugMemoryBreakdown && comm.rank() == 0) {
                    std::cerr << "[DEBUG-STACK] Resizing stack: " << required_size / (1024 * 1024) << " MiB"
                              << " -> " << needed / (1024 * 1024) << " MiB" << std::endl;
                }
                Tensor new_stack = mAllocator->allocate(ETensorDType::BYTE, "dsl_stack",
                                                        EAllocationType::ON_DEVICE, {needed});
                mRunState->set_stack_buffer(std::move(new_stack), high_mark);
                required_size = needed;
            }
        }
    }

    // Enable MoE routing stats tracking
    if (mModelConfig.NumExperts > 0) {
        float aux_coef = mModelConfig.moe_config.has_value()
                         ? mModelConfig.moe_config->router_aux_loss_coef
                         : 0.01f;
        mRunState->set_moe_config(mModelConfig.NumExperts, aux_coef);
    }

    // Wire weight manager for streaming/sharding
    if (mWeightManager) {
        if (auto* exec = dynamic_cast<GraphExecutor*>(mExecutor.get())) {
            exec->set_weight_manager(mWeightManager.get());
        }
    }

    if (lora_enabled()) {
        ensure_lora_run_state(comm, B, T);
        mExecutor->set_lora_state(mLoRAConfig ? &*mLoRAConfig : nullptr,
                                  mLoRAWeights.get(), mLoRAGrads.get(), mLoRARunState.get());
    }

    if (allocate_optimizer) {
        if (lora_enabled()) {
            if (!mLoRAAdamW8BitState) {
                mLoRAAdamW8BitState = std::make_unique<modules::LoRAAdamW8BitState>();
            }
        } else {
            if (!mAdamW8BitState) {
                mAdamW8BitState = std::make_unique<AdamW8BitState>();
            }
        }
    }
}

void DslModel::zero_grads(cudaStream_t stream) {
    if (mGrads) {
        mGrads->zero_all(stream);
    }
}

void DslModel::set_internal_graphs_enabled(bool enabled) {
    if (mExecutor) {
        mExecutor->set_internal_graphs_enabled(enabled);
    }
}

bool DslModel::internal_graphs_enabled() const {
    return mExecutor ? mExecutor->internal_graphs_enabled() : false;
}

bool DslModel::has_capture_unsafe_ops() const {
    return mExecutor ? mExecutor->has_capture_unsafe_ops() : false;
}

std::vector<float> DslModel::compute_logprobs(const std::int32_t* input_ids,
                                               const std::int32_t* targets,
                                               int B, int T, bool use_lora,
                                               NCCLCommunicator& comm,
                                               const std::int32_t* position_ids,
                                               const float* temperatures) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::compute_logprobs called before allocate_run_state()");
    }

    auto* graph_exec = dynamic_cast<GraphExecutor*>(mExecutor.get());
    if (!graph_exec) {
        throw std::runtime_error("DslModel::compute_logprobs: executor is not a GraphExecutor");
    }

    const int BT = B * T;
    std::vector<float> result(static_cast<std::size_t>(BT), 0.0f);

    const modules::ForwardHook* hook_ptr = nullptr;
    modules::ForwardHook hook;

    if (use_lora && lora_enabled()) {
        ensure_lora_run_state(comm, B, T);

        hook = [this](int layer_idx, cudaStream_t stream, modules::ForwardHookPoint point, void* context) {
            (void)context;
            const auto& cfg = mModelConfig;
            auto& rs = *mRunState;
            const int B_ = (int)rs.B;
            const int T_ = (int)rs.T;
            const int C = (int)cfg.HiddenSize;
            const int D = cfg.get_intermediate_size(layer_idx);
            const int Hq = (int)cfg.NumQueryHeads;
            const int Hkv = (int)cfg.NumKeyValHeads;
            const int Hs = (int)cfg.head_size();
            const int rank = mLoRAConfig->rank;
            const float scaling = mLoRAConfig->scaling();
            const bool gated_mlp = modules::is_gated_activation(cfg.activation_type);
            const bool use_qwen35_attention_lora = is_qwen3_5_model(cfg);

            auto& acts = rs.simplified_acts(layer_idx);
            auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);

            switch (point) {
                case modules::ForwardHookPoint::AfterQKVProjection: {
                    if (use_qwen35_attention_lora) break;
                    if (lora_block.attention.q.has_value()) {
                        modules::detail::apply_lora_contribution(acts.qkv, 0, acts.ln1, lora_block.attention.q.value(),
                                                        mLoRARunState->intermediate, mLoRARunState->slice,
                                                        scaling, 0.0f, 0, false,
                                                        B_ * T_, C, Hq * Hs, rank,
                                                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    }
                    if (lora_block.attention.k.has_value()) {
                        modules::detail::apply_lora_contribution(acts.qkv, Hq * Hs, acts.ln1, lora_block.attention.k.value(),
                                                        mLoRARunState->intermediate, mLoRARunState->slice,
                                                        scaling, 0.0f, 0, false,
                                                        B_ * T_, C, Hkv * Hs, rank,
                                                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    }
                    if (lora_block.attention.v.has_value()) {
                        modules::detail::apply_lora_contribution(acts.qkv, (Hq + Hkv) * Hs, acts.ln1, lora_block.attention.v.value(),
                                                        mLoRARunState->intermediate, mLoRARunState->slice,
                                                        scaling, 0.0f, 0, false,
                                                        B_ * T_, C, Hkv * Hs, rank,
                                                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    }
                } break;
                case modules::ForwardHookPoint::AfterAttnOutProjection: {
                    if (use_qwen35_attention_lora) break;
                    if (lora_block.attention.o.has_value()) {
                        modules::detail::apply_lora_contribution(acts.att_out, 0, acts.att, lora_block.attention.o.value(),
                                                        mLoRARunState->intermediate, mLoRARunState->slice,
                                                        scaling, 0.0f, 0, false,
                                                        B_ * T_, Hq * Hs, C, rank,
                                                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    }
                } break;
                case modules::ForwardHookPoint::AfterMLPUpProjection: {
                    Tensor& ln2_input = acts.ln2.Data ? acts.ln2 : acts.ln1;
                    if (lora_block.mlp.up.has_value()) {
                        modules::detail::apply_lora_contribution(acts.mlp_up, 0, ln2_input, lora_block.mlp.up.value(),
                                                        mLoRARunState->intermediate, mLoRARunState->slice,
                                                        scaling, 0.0f, 0, false,
                                                        B_ * T_, C, D, rank,
                                                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    }
                    if (gated_mlp && lora_block.mlp.gate.has_value()) {
                        modules::detail::apply_lora_contribution(acts.mlp_up, D, ln2_input, lora_block.mlp.gate.value(),
                                                        mLoRARunState->intermediate, mLoRARunState->slice,
                                                        scaling, 0.0f, 0, false,
                                                        B_ * T_, C, D, rank,
                                                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    }
                } break;
                case modules::ForwardHookPoint::AfterMLPDownProjection: {
                    if (lora_block.mlp.down.has_value()) {
                        Tensor down_input = acts.swiglu;
                        Tensor down_input_tmp{};
                        bool free_down_input_tmp = false;
                        if (!down_input.Data && acts.mlp_up.Data) {
                            down_input_tmp = rs.temp_alloc(acts.mlp_down.DType, {B_ * T_, D}, "down_input_tmp");
                            Tensor down_input_view = down_input_tmp;
                            down_input_view.Rank = 3;
                            down_input_view.Sizes[0] = B_;
                            down_input_view.Sizes[1] = T_;
                            down_input_view.Sizes[2] = D;
                            for (int i = 3; i < MAX_TENSOR_DIM; ++i) down_input_view.Sizes[i] = 1;
                            switch (cfg.activation_type) {
                                case modules::ActivationType::SwiGLU:
                                case modules::ActivationType::GeGLU:
                                    swiglu_forward(down_input_view, acts.mlp_up, nullptr, B_, T_, D, stream);
                                    break;
                                case modules::ActivationType::ReLU2: {
                                    const long N = static_cast<long>(B_) * static_cast<long>(T_) * static_cast<long>(D);
                                    relu2_forward(down_input_view, acts.mlp_up, N, stream);
                                } break;
                                case modules::ActivationType::SiLU: {
                                    const long N = static_cast<long>(B_) * static_cast<long>(T_) * static_cast<long>(D);
                                    silu_forward(down_input_view, acts.mlp_up, N, stream);
                                } break;
                                case modules::ActivationType::GeLU: {
                                    const long N = static_cast<long>(B_) * static_cast<long>(T_) * static_cast<long>(D);
                                    gelu_forward(down_input_view, acts.mlp_up, N, stream);
                                } break;
                                default:
                                    throw std::runtime_error("unsupported activation type for LoRA MLPDown replay");
                            }
                            down_input = down_input_view;
                            free_down_input_tmp = true;
                        }
                        if (down_input.Data) {
                            try {
                                modules::detail::apply_lora_contribution(acts.mlp_down, 0, down_input, lora_block.mlp.down.value(),
                                                                mLoRARunState->intermediate, mLoRARunState->slice,
                                                                scaling, 0.0f, 0, false,
                                                                B_ * T_, D, C, rank,
                                                                rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                            } catch (...) {
                                if (free_down_input_tmp) rs.temp_free(down_input_tmp);
                                throw;
                            }
                        }
                        if (free_down_input_tmp) rs.temp_free(down_input_tmp);
                    }
                } break;
                default:
                    break;
            }
        };
        hook_ptr = &hook;
    }

    // Detect document boundaries and enable flash varlen masking if needed.
    const bool mrope = mModelConfig.Rope.is_multimodal();
    std::optional<DocMaskingInfo> doc_info;
    if (mOptions.DocMasking) {
        doc_info = compute_doc_masking(position_ids, B, T, mrope);
    }
    if (doc_info) {
        graph_exec->set_doc_masking(doc_info->cu_seqlens.data(),
                                     doc_info->num_docs,
                                     doc_info->max_seqlen,
                                     doc_info->total_q);
    }

    graph_exec->execute_logprobs_forward((long)B, (long)T, input_ids, targets,
                                          result.data(), hook_ptr, comm, position_ids, temperatures);

    if (doc_info) {
        graph_exec->clear_doc_masking();
    }

    return result;
}

void DslModel::step_with_custom_loss(Tensor inputs, Tensor position_ids, Tensor targets,
                                      const float* per_token_grads_cpu,
                                      int grad_accum_steps, int micro_step,
                                      NCCLCommunicator& comm,
                                      const float* temperatures) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::step_with_custom_loss called before allocate_run_state()");
    }
    mUseTokenScale = false;

    auto* graph_exec = dynamic_cast<GraphExecutor*>(mExecutor.get());
    if (!graph_exec) {
        throw std::runtime_error("DslModel::step_with_custom_loss: executor is not a GraphExecutor");
    }

    // Detect document boundaries and enable flash varlen masking if needed.
    const int B_val = static_cast<int>(inputs.Sizes[0]);
    const int T_val = static_cast<int>(inputs.Sizes[1]);
    const std::int32_t* position_ids_ptr =
        (position_ids.Data && position_ids.Device == -1)
            ? reinterpret_cast<const std::int32_t*>(position_ids.Data)
            : nullptr;
    const bool mrope = mModelConfig.Rope.is_multimodal();
    std::optional<DocMaskingInfo> doc_info;
    if (mOptions.DocMasking) {
        doc_info = compute_doc_masking(position_ids_ptr, B_val, T_val, mrope);
    }
    if (doc_info) {
        graph_exec->set_doc_masking(doc_info->cu_seqlens.data(),
                                     doc_info->num_docs,
                                     doc_info->max_seqlen,
                                     doc_info->total_q);
    }

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;
    float* inv_temperature_gpu = nullptr;
    if (temperatures) {
        const std::size_t bt = static_cast<std::size_t>(B_val) * static_cast<std::size_t>(T_val);
        std::vector<float> inv_temp(bt);
        for (std::size_t i = 0; i < bt; ++i) {
            inv_temp[i] = 1.0f / temperatures[i];
        }
        CUDA_CHECK(cudaMalloc(&inv_temperature_gpu, bt * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(inv_temperature_gpu, inv_temp.data(),
                                   bt * sizeof(float), cudaMemcpyHostToDevice, main_stream));
        graph_exec->set_inv_temperature_context(inv_temperature_gpu);
    }

    // Forward pass (with LoRA hooks if enabled) — saves activations for backward.
    forward(inputs, position_ids, comm, micro_step);

    if (!lora_enabled()) {
        // No LoRA: plain backward with custom d_loss.
        graph_exec->backward_with_custom_dloss(inputs, targets, per_token_grads_cpu,
                                               comm, grad_accum_steps, micro_step, nullptr,
                                               nullptr);
        if (doc_info) graph_exec->clear_doc_masking();
        if (inv_temperature_gpu) {
            graph_exec->set_inv_temperature_context(nullptr);
            CUDA_CHECK(cudaFree(inv_temperature_gpu));
        }
        return;
    }

    // LoRA backward: mirror DslModel::backward() exactly, but use backward_with_custom_dloss.
    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

    if (mLoRALn1Names.empty()) {
        build_lora_name_tables();
    }

    mLoRAGrads->start_micro_step(main_stream, micro_step, grad_accum_steps);

    modules::BackwardHook hook = [this, &comm](int layer_idx, bool accumulate, cudaStream_t stream,
                               modules::BackwardHookPoint point, void* context) {
        (void)context;
        const auto& cfg = mModelConfig;
        auto& rs = *mRunState;
        const int B = (int)rs.B;
        const int T = (int)rs.T;
        const int C = (int)cfg.HiddenSize;
        const int D = cfg.get_intermediate_size(layer_idx);
        const int Hq = (int)cfg.NumQueryHeads;
        const int Hkv = (int)cfg.NumKeyValHeads;
        const int Hs = (int)cfg.head_size();
        const int rank = mLoRAConfig->rank;
        const float dropout = mLoRAConfig->dropout;
        const bool is_training = mLoRARunState->is_training;
        const int micro_step = mLoRARunState->micro_step;
        const bool gated_mlp = modules::is_gated_activation(cfg.activation_type);
        const bool use_qwen35_attention_lora = is_qwen3_5_model(cfg);

        auto get_dropout_seed = [&](int proj_type) -> unsigned int {
            return mLoRARunState->dropout_base_seed
                   + static_cast<unsigned int>(layer_idx) * 1000000u
                   + static_cast<unsigned int>(proj_type) * 100000u
                   + static_cast<unsigned int>(micro_step) * 10000u;
        };

        auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);

        switch (point) {
            case modules::BackwardHookPoint::AfterMLPDownBackward: {
                if (!lora_block.mlp.down.has_value()) {
                    break;
                }
                bool lora_accum = false;
                auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                lora_accum = lora_accum || accumulate;
                if (!lora_grads.mlp.down.has_value()) {
                    break;
                }
                auto& a = rs.simplified_acts(layer_idx);
                auto& da = rs.simplified_grads(layer_idx);
                modules::detail::backward_lora_layer(
                    lora_grads.mlp.down->A, lora_grads.mlp.down->B,
                    da.d_swiglu, da.d_res_ffn, 0, a.swiglu,
                    lora_block.mlp.down->A, lora_block.mlp.down->B,
                    mLoRAConfig->scaling(), dropout, get_dropout_seed(6), is_training,
                    mLoRARunState->intermediate, mLoRARunState->slice,
                    B * T, D, C, rank, lora_accum,
                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
            } break;
            case modules::BackwardHookPoint::AfterMLPUpBackward: {
                bool lora_accum = false;
                auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                lora_accum = lora_accum || accumulate;
                auto& a = rs.simplified_acts(layer_idx);
                auto& da = rs.simplified_grads(layer_idx);
                Tensor& d_ln2 = da.d_ln2.Data ? da.d_ln2 : da.d_ln1;
                Tensor ln2_input;
                if (mOptions.recompute_enabled()) {
                    if (mLoRARunState && mLoRARunState->recompute_ln.Data) {
                        Tensor& ln2_weight = mParams->get(mLoRALn2Names[layer_idx]);
                        Tensor ln2_residual;
                        if (a.residual_att.Data) {
                            ln2_residual = a.residual_att;
                        } else if (mModelConfig.architecture == modules::ArchitectureType::Hybrid) {
                            if (rs.has_residual_offloading()) rs.fetch_residual(layer_idx, rs.side_stream());
                            ln2_residual = rs.get_residual(layer_idx, stream);
                        } else if (layer_idx == 0) {
                            ln2_residual = rs.non_block_activations().encoded;
                        } else {
                            if (rs.has_residual_offloading()) rs.fetch_residual(layer_idx - 1, rs.side_stream());
                            ln2_residual = rs.get_residual(layer_idx - 1, stream);
                        }
                        ln2_input = recompute_lora_rmsnorm(*mLoRARunState, ln2_residual, ln2_weight,
                                                           mModelConfig.RmsNormEps, B, T, C, stream);
                    } else {
                        ln2_input = a.ln2.Data ? a.ln2 : a.ln1;
                    }
                } else {
                    ln2_input = a.ln2.Data ? a.ln2 : a.ln1;
                }
                Tensor dA_up{}, dB_up{}, dA_gate{}, dB_gate{};
                modules::LoRALayerWeights<Tensor> lora_up{}, lora_gate{};
                if (gated_mlp) {
                    if (lora_block.mlp.up.has_value() && lora_grads.mlp.up.has_value()) {
                        dA_up = lora_grads.mlp.up->A; dB_up = lora_grads.mlp.up->B;
                        lora_up = *lora_block.mlp.up;
                    }
                    if (lora_block.mlp.gate.has_value() && lora_grads.mlp.gate.has_value()) {
                        dA_gate = lora_grads.mlp.gate->A; dB_gate = lora_grads.mlp.gate->B;
                        lora_gate = *lora_block.mlp.gate;
                    }
                    if (!dA_up.Data && !dA_gate.Data) break;
                    modules::detail::backward_lora_mlp_up_gate_fused(
                        dA_up, dB_up, dA_gate, dB_gate, d_ln2, da.d_mlp_up, ln2_input,
                        lora_up, lora_gate, mLoRAConfig->scaling(),
                        dropout, get_dropout_seed(4), get_dropout_seed(5), is_training,
                        B * T, C, D, rank, lora_accum,
                        mLoRARunState->intermediate, mLoRARunState->intermediate2,
                        mLoRARunState->slice, rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                } else {
                    if (!lora_block.mlp.up.has_value() || !lora_grads.mlp.up.has_value()) break;
                    dA_up = lora_grads.mlp.up->A; dB_up = lora_grads.mlp.up->B;
                    lora_up = *lora_block.mlp.up;
                    modules::detail::backward_lora_layer(
                        dA_up, dB_up, d_ln2, da.d_mlp_up, 0, ln2_input,
                        lora_up.A, lora_up.B, mLoRAConfig->scaling(),
                        dropout, get_dropout_seed(4), is_training,
                        mLoRARunState->intermediate, mLoRARunState->slice,
                        B * T, C, D, rank, lora_accum,
                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::BackwardHookPoint::AfterAttnOutBackward: {
                if (use_qwen35_attention_lora) break;
                if (!lora_block.attention.o.has_value()) break;
                bool lora_accum = false;
                auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                lora_accum = lora_accum || accumulate;
                if (!lora_grads.attention.o.has_value()) break;
                auto& a = rs.simplified_acts(layer_idx);
                auto& da = rs.simplified_grads(layer_idx);
                modules::detail::backward_lora_layer(
                    lora_grads.attention.o->A, lora_grads.attention.o->B,
                    da.d_att, da.d_att_out, 0, a.att,
                    lora_block.attention.o->A, lora_block.attention.o->B,
                    mLoRAConfig->scaling(), dropout, get_dropout_seed(3), is_training,
                    mLoRARunState->intermediate, mLoRARunState->slice,
                    B * T, Hq * Hs, C, rank, lora_accum,
                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
            } break;
            case modules::BackwardHookPoint::AfterQKVBackward: {
                if (use_qwen35_attention_lora) break;
                bool lora_accum = false;
                auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                lora_accum = lora_accum || accumulate;
                auto& a = rs.simplified_acts(layer_idx);
                auto& da = rs.simplified_grads(layer_idx);
                Tensor ln1_input;
                if (mOptions.recompute_enabled()) {
                    if (mLoRARunState && mLoRARunState->recompute_ln.Data) {
                        Tensor& ln1_weight = mParams->get(mLoRALn1Names[layer_idx]);
                        Tensor ln1_residual;
                        if (mModelConfig.architecture == modules::ArchitectureType::Hybrid) {
                            if (rs.has_residual_offloading()) rs.fetch_residual(layer_idx, rs.side_stream());
                            ln1_residual = rs.get_residual(layer_idx, stream);
                        } else if (layer_idx == 0) {
                            ln1_residual = rs.non_block_activations().encoded;
                        } else {
                            if (rs.has_residual_offloading()) rs.fetch_residual(layer_idx - 1, rs.side_stream());
                            ln1_residual = rs.get_residual(layer_idx - 1, stream);
                        }
                        ln1_input = recompute_lora_rmsnorm(*mLoRARunState, ln1_residual, ln1_weight,
                                                           mModelConfig.RmsNormEps, B, T, C, stream);
                    } else {
                        ln1_input = a.ln1;
                    }
                } else {
                    ln1_input = a.ln1;
                }
                Tensor dA_q{}, dB_q{}, dA_k{}, dB_k{}, dA_v{}, dB_v{};
                modules::LoRALayerWeights<Tensor> lora_q{}, lora_k{}, lora_v{};
                if (lora_block.attention.q.has_value() && lora_grads.attention.q.has_value()) {
                    dA_q = lora_grads.attention.q->A; dB_q = lora_grads.attention.q->B;
                    lora_q = *lora_block.attention.q;
                }
                if (lora_block.attention.k.has_value() && lora_grads.attention.k.has_value()) {
                    dA_k = lora_grads.attention.k->A; dB_k = lora_grads.attention.k->B;
                    lora_k = *lora_block.attention.k;
                }
                if (lora_block.attention.v.has_value() && lora_grads.attention.v.has_value()) {
                    dA_v = lora_grads.attention.v->A; dB_v = lora_grads.attention.v->B;
                    lora_v = *lora_block.attention.v;
                }
                if (!dA_q.Data && !dA_k.Data && !dA_v.Data) break;
                modules::detail::backward_lora_qkv_fused(
                    dA_q, dB_q, dA_k, dB_k, dA_v, dB_v,
                    da.d_ln1, da.d_qkv, ln1_input, lora_q, lora_k, lora_v,
                    mLoRAConfig->scaling(),
                    dropout, get_dropout_seed(0), get_dropout_seed(1), get_dropout_seed(2), is_training,
                    B * T, C, Hq * Hs, Hkv * Hs, rank, lora_accum,
                    mLoRARunState->intermediate, mLoRARunState->intermediate2,
                    mLoRARunState->slice, rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                mLoRAGrads->notify_block(layer_idx, stream, comm);
            } break;
            default:
                break;
        }
    };

    graph_exec->backward_with_custom_dloss(inputs, targets, per_token_grads_cpu,
                                            comm, grad_accum_steps, micro_step, &hook,
                                            nullptr);

    if (doc_info) graph_exec->clear_doc_masking();
    if (inv_temperature_gpu) {
        graph_exec->set_inv_temperature_context(nullptr);
        CUDA_CHECK(cudaFree(inv_temperature_gpu));
    }

    mLoRAGrads->end_micro_step(main_stream, comm);
    // Extend the base-model BackwardDone event to include LoRA gradient reductions.
    internal::record_event_if_not_capturing(rs.BackwardDone, main_stream);
}

// ============================================================================
// GRPO single-pass: forward (saves activations) + logprobs extraction
// ============================================================================

std::vector<float> DslModel::forward_for_grpo(Tensor inputs, Tensor position_ids, Tensor targets,
                                               int grad_accum_steps, int micro_step,
                                               NCCLCommunicator& comm,
                                               const float* temperatures) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::forward_for_grpo called before allocate_run_state()");
    }
    mUseTokenScale = false;

    auto* graph_exec = dynamic_cast<GraphExecutor*>(mExecutor.get());
    if (!graph_exec) {
        throw std::runtime_error("DslModel::forward_for_grpo: executor is not a GraphExecutor");
    }

    // Detect document boundaries and enable flash varlen masking if needed.
    const int B_val = static_cast<int>(inputs.Sizes[0]);
    const int T_val = static_cast<int>(inputs.Sizes[1]);
    const std::size_t BT = static_cast<std::size_t>(B_val) * static_cast<std::size_t>(T_val);
    const std::int32_t* position_ids_ptr =
        (position_ids.Data && position_ids.Device == -1)
            ? reinterpret_cast<const std::int32_t*>(position_ids.Data)
            : nullptr;
    const bool mrope = mModelConfig.Rope.is_multimodal();
    std::optional<DocMaskingInfo> doc_info;
    if (mOptions.DocMasking) {
        doc_info = compute_doc_masking(position_ids_ptr, B_val, T_val, mrope);
    }
    if (doc_info) {
        graph_exec->set_doc_masking(doc_info->cu_seqlens.data(),
                                     doc_info->num_docs,
                                     doc_info->max_seqlen,
                                     doc_info->total_q);
    }

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;

    // Set up per-token inverse temperatures (persists for backward_grpo).
    if (mGrpoInvTemperatureGpu) {
        CUDA_CHECK(cudaFree(mGrpoInvTemperatureGpu));
        mGrpoInvTemperatureGpu = nullptr;
    }
    if (temperatures) {
        std::vector<float> inv_temp(BT);
        for (std::size_t i = 0; i < BT; ++i) {
            inv_temp[i] = 1.0f / temperatures[i];
        }
        CUDA_CHECK(cudaMalloc(&mGrpoInvTemperatureGpu, BT * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(mGrpoInvTemperatureGpu, inv_temp.data(),
                                   BT * sizeof(float), cudaMemcpyHostToDevice, main_stream));
        graph_exec->set_inv_temperature_context(mGrpoInvTemperatureGpu);
    }

    // Always zero the losses buffer so we get per-micro-batch losses
    // (not accumulated from previous micro-steps).
    fill_zero(rs.Losses, main_stream);
    fill_zero(rs.ValidTokenCount, main_stream);
    fill_zero(rs.CorrectCount, main_stream);

    // Forward pass (with LoRA hooks if enabled) — saves activations for backward.
    forward(inputs, position_ids, comm, micro_step);

    // Extract logprobs from the Losses buffer.
    // cross_entropy_forward writes: losses[t] = logsumexp - logit[target[t]] = -logprob[t].
    // Masked positions (target == -100) have losses[t] = 0, so logprob[t] = 0.
    std::vector<float> logprobs(BT, 0.0f);
    CUDA_CHECK(cudaMemcpyAsync(logprobs.data(), rs.Losses.Data,
                               BT * sizeof(float), cudaMemcpyDeviceToHost, main_stream));
    CUDA_CHECK(cudaStreamSynchronize(main_stream));
    for (std::size_t i = 0; i < BT; ++i) {
        logprobs[i] = -logprobs[i];
    }

    // Doc masking and temperature context persist for backward_grpo().
    return logprobs;
}

// ============================================================================
// GRPO backward pass (uses activations saved by forward_for_grpo)
// ============================================================================

void DslModel::backward_grpo(Tensor inputs, Tensor targets,
                               const float* per_token_grads_cpu,
                               int grad_accum_steps, int micro_step,
                               NCCLCommunicator& comm) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::backward_grpo called before allocate_run_state()");
    }

    auto* graph_exec = dynamic_cast<GraphExecutor*>(mExecutor.get());
    if (!graph_exec) {
        throw std::runtime_error("DslModel::backward_grpo: executor is not a GraphExecutor");
    }

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;

    if (!lora_enabled()) {
        // No LoRA: plain backward with custom d_loss.
        // Temperature context is already set from forward_for_grpo (pass nullptr to skip re-allocation).
        graph_exec->backward_with_custom_dloss(inputs, targets, per_token_grads_cpu,
                                               comm, grad_accum_steps, micro_step, nullptr,
                                               nullptr);
    } else {
        // LoRA backward: mirror step_with_custom_loss LoRA backward path exactly.
        ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

        if (mLoRALn1Names.empty()) {
            build_lora_name_tables();
        }

        mLoRAGrads->start_micro_step(main_stream, micro_step, grad_accum_steps);

        modules::BackwardHook hook = [this, &comm](int layer_idx, bool accumulate, cudaStream_t stream,
                                   modules::BackwardHookPoint point, void* context) {
            (void)context;
            const auto& cfg = mModelConfig;
            auto& rs = *mRunState;
            const int B = (int)rs.B;
            const int T = (int)rs.T;
            const int C = (int)cfg.HiddenSize;
            const int D = cfg.get_intermediate_size(layer_idx);
            const int Hq = (int)cfg.NumQueryHeads;
            const int Hkv = (int)cfg.NumKeyValHeads;
            const int Hs = (int)cfg.head_size();
            const int rank = mLoRAConfig->rank;
            const float dropout = mLoRAConfig->dropout;
            const bool is_training = mLoRARunState->is_training;
            const int micro_step = mLoRARunState->micro_step;
            const bool gated_mlp = modules::is_gated_activation(cfg.activation_type);
            const bool use_qwen35_attention_lora = is_qwen3_5_model(cfg);

            auto get_dropout_seed = [&](int proj_type) -> unsigned int {
                return mLoRARunState->dropout_base_seed
                       + static_cast<unsigned int>(layer_idx) * 1000000u
                       + static_cast<unsigned int>(proj_type) * 100000u
                       + static_cast<unsigned int>(micro_step) * 10000u;
            };

            auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);

            switch (point) {
                case modules::BackwardHookPoint::AfterMLPDownBackward: {
                    if (!lora_block.mlp.down.has_value()) {
                        break;
                    }
                    bool lora_accum = false;
                    auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                    lora_accum = lora_accum || accumulate;
                    if (!lora_grads.mlp.down.has_value()) {
                        break;
                    }
                    auto& a = rs.simplified_acts(layer_idx);
                    auto& da = rs.simplified_grads(layer_idx);
                    modules::detail::backward_lora_layer(
                        lora_grads.mlp.down->A, lora_grads.mlp.down->B,
                        da.d_swiglu, da.d_res_ffn, 0, a.swiglu,
                        lora_block.mlp.down->A, lora_block.mlp.down->B,
                        mLoRAConfig->scaling(), dropout, get_dropout_seed(6), is_training,
                        mLoRARunState->intermediate, mLoRARunState->slice,
                        B * T, D, C, rank, lora_accum,
                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                } break;
                case modules::BackwardHookPoint::AfterMLPUpBackward: {
                    bool lora_accum = false;
                    auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                    lora_accum = lora_accum || accumulate;
                    auto& a = rs.simplified_acts(layer_idx);
                    auto& da = rs.simplified_grads(layer_idx);
                    Tensor& d_ln2 = da.d_ln2.Data ? da.d_ln2 : da.d_ln1;
                    Tensor ln2_input;
                    if (mOptions.recompute_enabled()) {
                        if (mLoRARunState && mLoRARunState->recompute_ln.Data) {
                            Tensor& ln2_weight = mParams->get(mLoRALn2Names[layer_idx]);
                            Tensor ln2_residual;
                            if (a.residual_att.Data) {
                                ln2_residual = a.residual_att;
                            } else if (mModelConfig.architecture == modules::ArchitectureType::Hybrid) {
                                if (rs.has_residual_offloading()) rs.fetch_residual(layer_idx, rs.side_stream());
                                ln2_residual = rs.get_residual(layer_idx, stream);
                            } else if (layer_idx == 0) {
                                ln2_residual = rs.non_block_activations().encoded;
                            } else {
                                if (rs.has_residual_offloading()) rs.fetch_residual(layer_idx - 1, rs.side_stream());
                                ln2_residual = rs.get_residual(layer_idx - 1, stream);
                            }
                            ln2_input = recompute_lora_rmsnorm(*mLoRARunState, ln2_residual, ln2_weight,
                                                               mModelConfig.RmsNormEps, B, T, C, stream);
                        } else {
                            ln2_input = a.ln2.Data ? a.ln2 : a.ln1;
                        }
                    } else {
                        ln2_input = a.ln2.Data ? a.ln2 : a.ln1;
                    }
                    Tensor dA_up{}, dB_up{}, dA_gate{}, dB_gate{};
                    modules::LoRALayerWeights<Tensor> lora_up{}, lora_gate{};
                    if (gated_mlp) {
                        if (lora_block.mlp.up.has_value() && lora_grads.mlp.up.has_value()) {
                            dA_up = lora_grads.mlp.up->A; dB_up = lora_grads.mlp.up->B;
                            lora_up = *lora_block.mlp.up;
                        }
                        if (lora_block.mlp.gate.has_value() && lora_grads.mlp.gate.has_value()) {
                            dA_gate = lora_grads.mlp.gate->A; dB_gate = lora_grads.mlp.gate->B;
                            lora_gate = *lora_block.mlp.gate;
                        }
                        if (!dA_up.Data && !dA_gate.Data) break;
                        modules::detail::backward_lora_mlp_up_gate_fused(
                            dA_up, dB_up, dA_gate, dB_gate, d_ln2, da.d_mlp_up, ln2_input,
                            lora_up, lora_gate, mLoRAConfig->scaling(),
                            dropout, get_dropout_seed(4), get_dropout_seed(5), is_training,
                            B * T, C, D, rank, lora_accum,
                            mLoRARunState->intermediate, mLoRARunState->intermediate2,
                            mLoRARunState->slice, rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    } else {
                        if (!lora_block.mlp.up.has_value() || !lora_grads.mlp.up.has_value()) break;
                        dA_up = lora_grads.mlp.up->A; dB_up = lora_grads.mlp.up->B;
                        lora_up = *lora_block.mlp.up;
                        modules::detail::backward_lora_layer(
                            dA_up, dB_up, d_ln2, da.d_mlp_up, 0, ln2_input,
                            lora_up.A, lora_up.B, mLoRAConfig->scaling(),
                            dropout, get_dropout_seed(4), is_training,
                            mLoRARunState->intermediate, mLoRARunState->slice,
                            B * T, C, D, rank, lora_accum,
                            rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    }
                } break;
                case modules::BackwardHookPoint::AfterAttnOutBackward: {
                    if (use_qwen35_attention_lora) break;
                    if (!lora_block.attention.o.has_value()) break;
                    bool lora_accum = false;
                    auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                    lora_accum = lora_accum || accumulate;
                    if (!lora_grads.attention.o.has_value()) break;
                    auto& a = rs.simplified_acts(layer_idx);
                    auto& da = rs.simplified_grads(layer_idx);
                    modules::detail::backward_lora_layer(
                        lora_grads.attention.o->A, lora_grads.attention.o->B,
                        da.d_att, da.d_att_out, 0, a.att,
                        lora_block.attention.o->A, lora_block.attention.o->B,
                        mLoRAConfig->scaling(), dropout, get_dropout_seed(3), is_training,
                        mLoRARunState->intermediate, mLoRARunState->slice,
                        B * T, Hq * Hs, C, rank, lora_accum,
                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                } break;
                case modules::BackwardHookPoint::AfterQKVBackward: {
                    if (use_qwen35_attention_lora) break;
                    bool lora_accum = false;
                    auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                    lora_accum = lora_accum || accumulate;
                    auto& a = rs.simplified_acts(layer_idx);
                    auto& da = rs.simplified_grads(layer_idx);
                    Tensor ln1_input;
                    if (mOptions.recompute_enabled()) {
                        if (mLoRARunState && mLoRARunState->recompute_ln.Data) {
                            Tensor& ln1_weight = mParams->get(mLoRALn1Names[layer_idx]);
                            Tensor ln1_residual;
                            if (mModelConfig.architecture == modules::ArchitectureType::Hybrid) {
                                if (rs.has_residual_offloading()) rs.fetch_residual(layer_idx, rs.side_stream());
                                ln1_residual = rs.get_residual(layer_idx, stream);
                            } else if (layer_idx == 0) {
                                ln1_residual = rs.non_block_activations().encoded;
                            } else {
                                if (rs.has_residual_offloading()) rs.fetch_residual(layer_idx - 1, rs.side_stream());
                                ln1_residual = rs.get_residual(layer_idx - 1, stream);
                            }
                            ln1_input = recompute_lora_rmsnorm(*mLoRARunState, ln1_residual, ln1_weight,
                                                               mModelConfig.RmsNormEps, B, T, C, stream);
                        } else {
                            ln1_input = a.ln1;
                        }
                    } else {
                        ln1_input = a.ln1;
                    }
                    Tensor dA_q{}, dB_q{}, dA_k{}, dB_k{}, dA_v{}, dB_v{};
                    modules::LoRALayerWeights<Tensor> lora_q{}, lora_k{}, lora_v{};
                    if (lora_block.attention.q.has_value() && lora_grads.attention.q.has_value()) {
                        dA_q = lora_grads.attention.q->A; dB_q = lora_grads.attention.q->B;
                        lora_q = *lora_block.attention.q;
                    }
                    if (lora_block.attention.k.has_value() && lora_grads.attention.k.has_value()) {
                        dA_k = lora_grads.attention.k->A; dB_k = lora_grads.attention.k->B;
                        lora_k = *lora_block.attention.k;
                    }
                    if (lora_block.attention.v.has_value() && lora_grads.attention.v.has_value()) {
                        dA_v = lora_grads.attention.v->A; dB_v = lora_grads.attention.v->B;
                        lora_v = *lora_block.attention.v;
                    }
                    if (!dA_q.Data && !dA_k.Data && !dA_v.Data) break;
                    modules::detail::backward_lora_qkv_fused(
                        dA_q, dB_q, dA_k, dB_k, dA_v, dB_v,
                        da.d_ln1, da.d_qkv, ln1_input, lora_q, lora_k, lora_v,
                        mLoRAConfig->scaling(),
                        dropout, get_dropout_seed(0), get_dropout_seed(1), get_dropout_seed(2), is_training,
                        B * T, C, Hq * Hs, Hkv * Hs, rank, lora_accum,
                        mLoRARunState->intermediate, mLoRARunState->intermediate2,
                        mLoRARunState->slice, rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    mLoRAGrads->notify_block(layer_idx, stream, comm);
                } break;
                default:
                    break;
            }
        };

        graph_exec->backward_with_custom_dloss(inputs, targets, per_token_grads_cpu,
                                                comm, grad_accum_steps, micro_step, &hook,
                                                nullptr);

        mLoRAGrads->end_micro_step(main_stream, comm);
        internal::record_event_if_not_capturing(rs.BackwardDone, main_stream);
    }

    // Clean up state that was set by forward_for_grpo.
    if (mDocMaskingActive) {
        graph_exec->clear_doc_masking();
        mDocMaskingActive = false;
    }
    if (mGrpoInvTemperatureGpu) {
        graph_exec->set_inv_temperature_context(nullptr);
        CUDA_CHECK(cudaFree(mGrpoInvTemperatureGpu));
        mGrpoInvTemperatureGpu = nullptr;
    }
}

}  // namespace dsl
