// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_MODEL_FORWARD_H
#define SUROGATE_SRC_MODULES_LORA_LORA_MODEL_FORWARD_H

#include "lora_model_core.h"
#include "lora_model_utils.h"
#include "lora_utils.h"
#include "modules/lora/fast_expert_lora.h"
#include "modules/moe/moe_types.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <vector>

#include <cuda_runtime_api.h>

namespace modules {
namespace detail {
inline bool lora_fwd_nan_trace_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("SUROGATE_LORA_FWD_NAN_TRACE");
        enabled = (env && std::atoi(env) != 0) ? 1 : 0;
    }
    return enabled == 1;
}

inline int lora_fwd_nan_trace_layer() {
    static int layer = -2;
    if (layer == -2) {
        const char* env = std::getenv("SUROGATE_LORA_FWD_TRACE_LAYER");
        layer = env ? std::atoi(env) : -1;
    }
    return layer;
}

inline int lora_fwd_nan_trace_limit() {
    static int limit = -1;
    if (limit < 0) {
        const char* env = std::getenv("SUROGATE_LORA_FWD_TRACE_LIMIT");
        limit = env ? std::atoi(env) : 8;
    }
    return limit;
}

inline int lora_fwd_nan_trace_samples() {
    static int samples = -1;
    if (samples < 0) {
        const char* env = std::getenv("SUROGATE_LORA_FWD_TRACE_SAMPLES");
        samples = env ? std::atoi(env) : 8;
        if (samples < 1) samples = 1;
    }
    return samples;
}

inline bool lora_fwd_nan_trace_should_log(int layer_idx) {
    if (!lora_fwd_nan_trace_enabled()) return false;
    const int target = lora_fwd_nan_trace_layer();
    if (target >= 0 && target != layer_idx) return false;
    static std::atomic<int> counter{0};
    const int limit = lora_fwd_nan_trace_limit();
    if (limit <= 0) return false;
    const int idx = counter.fetch_add(1);
    return idx < limit;
}

inline bool lora_fwd_nan_trace_can_copy(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) {
        return false;
    }
    return status == cudaStreamCaptureStatusNone;
}

inline float lora_to_float(float v) { return v; }
inline float lora_to_float(nv_bfloat16 v) { return __bfloat162float(v); }
inline float lora_to_float(half v) { return __half2float(v); }

template <typename T>
inline void lora_trace_row(const char* tag,
                           int layer_idx,
                           const Tensor& t,
                           long row_idx,
                           long col_offset,
                           int samples,
                           cudaStream_t stream) {
    if (!t.Data || samples <= 0) return;
    if (!lora_fwd_nan_trace_can_copy(stream)) return;

    const long stride = t.Sizes[t.Rank - 1];
    const long rows = [&]() {
        long r = 1;
        for (int i = 0; i < t.Rank - 1; ++i) r *= t.Sizes[i];
        return r;
    }();
    if (row_idx < 0 || row_idx >= rows) return;
    if (col_offset < 0 || col_offset >= stride) return;
    const int count = std::min<long>(samples, stride - col_offset);
    const long elem_offset = row_idx * stride + col_offset;

    std::vector<T> host(count);
    const std::size_t bytes = (std::size_t)count * sizeof(T);
    CUDA_CHECK(cudaMemcpyAsync(host.data(),
                               t.Data + elem_offset * sizeof(T),
                               bytes,
                               cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int nan_count = 0;
    int inf_count = 0;
    float min_val = 0.0f;
    float max_val = 0.0f;
    if (!host.empty()) {
        float v0 = lora_to_float(host[0]);
        min_val = v0;
        max_val = v0;
        for (int i = 0; i < count; ++i) {
            float v = lora_to_float(host[i]);
            if (std::isnan(v)) nan_count++;
            if (std::isinf(v)) inf_count++;
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
    }

    fprintf(stderr,
            "[LORA_FWD_NAN] layer=%d tag=%s dtype=%s row=%ld offset=%ld samples=%d nan=%d inf=%d min=%g max=%g vals=%g,%g,%g,%g\n",
            layer_idx,
            tag,
            dtype_to_str(t.DType),
            row_idx,
            col_offset,
            count,
            nan_count,
            inf_count,
            min_val,
            max_val,
            count > 0 ? lora_to_float(host[0]) : 0.0f,
            count > 1 ? lora_to_float(host[1]) : 0.0f,
            count > 2 ? lora_to_float(host[2]) : 0.0f,
            count > 3 ? lora_to_float(host[3]) : 0.0f);
}

inline void lora_trace_tensor(const char* tag,
                              int layer_idx,
                              const Tensor& t,
                              long col_offset,
                              int samples,
                              cudaStream_t stream) {
    if (t.DType == ETensorDType::BF16) {
        lora_trace_row<nv_bfloat16>(tag, layer_idx, t, 0, col_offset, samples, stream);
        const long rows = [&]() {
            long r = 1;
            for (int i = 0; i < t.Rank - 1; ++i) r *= t.Sizes[i];
            return r;
        }();
        if (rows > 1) {
            lora_trace_row<nv_bfloat16>(tag, layer_idx, t, rows / 2, col_offset, samples, stream);
        }
        return;
    }
    if (t.DType == ETensorDType::FP16) {
        lora_trace_row<half>(tag, layer_idx, t, 0, col_offset, samples, stream);
        const long rows = [&]() {
            long r = 1;
            for (int i = 0; i < t.Rank - 1; ++i) r *= t.Sizes[i];
            return r;
        }();
        if (rows > 1) {
            lora_trace_row<half>(tag, layer_idx, t, rows / 2, col_offset, samples, stream);
        }
        return;
    }
    if (t.DType == ETensorDType::FP32) {
        lora_trace_row<float>(tag, layer_idx, t, 0, col_offset, samples, stream);
        const long rows = [&]() {
            long r = 1;
            for (int i = 0; i < t.Rank - 1; ++i) r *= t.Sizes[i];
            return r;
        }();
        if (rows > 1) {
            lora_trace_row<float>(tag, layer_idx, t, rows / 2, col_offset, samples, stream);
        }
        return;
    }
}
} // namespace detail

template<typename Block>
void ModularLoRAModel<Block>::forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) {
    if (!lora_enabled()) {
        mBaseModel->forward(inputs, position_ids, comm, micro_step);
        return;
    }

    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

    if (qlora_enabled() && micro_step == 0) {
        if (mFP8WeightProvider) mFP8WeightProvider->invalidate_cache();
        if (mFP4WeightProvider) mFP4WeightProvider->invalidate_cache();
        if (mBnBWeightProvider) mBnBWeightProvider->invalidate_cache();
    }

    // Store micro_step for dropout seed computation (needed by backward pass)
    mLoRARunState->micro_step = micro_step;
    mLoRARunState->is_training = true;

    auto hook = [this, micro_step](int layer_idx, cudaStream_t stream, ForwardHookPoint point, void* context) {
        const auto& cfg = mBaseModel->config();
        auto& rs = mBaseModel->run_state();
        const int B = (int)rs.B;
        const int T = (int)rs.T;
        const int C = (int)cfg.HiddenSize;
        const int D = (int)cfg.IntermediateSize;
        // For MoE models, experts have a different intermediate size
        const int moe_D = (cfg.moe_config && cfg.moe_config->moe_intermediate_size > 0) ? cfg.moe_config->moe_intermediate_size : D;
        const bool moe_gated = is_gated_activation(cfg.activation_type);
        const int Hq = (int)cfg.NumQueryHeads;
        const int Hkv = (int)cfg.NumKeyValHeads;
        const int Hs = (int)cfg.head_size();
        const int rank = mLoRAConfig.rank;
        const float scaling = mLoRAConfig.scaling();
        const float dropout = mLoRAConfig.dropout;
        const bool is_training = mLoRARunState->is_training;

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
            case ForwardHookPoint::MoEExpertGroupManual: {
               if (lora_block.moe.use_grouped && lora_block.moe.has_any() && context) {
                    static int moe_lora_entry_trace = 0;
                    if (std::getenv("SUROGATE_MOE_LORA_SPLIT_TRACE") && moe_lora_entry_trace < 8) {
                        fprintf(stderr,
                                "[MOE_LORA_SPLIT] enter layer=%d gated=%d use_grouped=%d has_any=%d ctx=%p\n",
                                layer_idx,
                                moe_gated ? 1 : 0,
                                lora_block.moe.use_grouped ? 1 : 0,
                                lora_block.moe.has_any() ? 1 : 0,
                                context);
                        moe_lora_entry_trace++;
                    }
                    auto* moe_ctx = static_cast<MoEGroupedContext*>(context);

                    // Selective expert dequantization: only dequantize router-selected experts
                    // This saves ~1.1GB of dequant buffer memory for 128-expert MoE models
                    SelectiveExpertInfo selection_info;
                    if (mBnBWeightProvider && mBnBWeightProvider->use_selective_dequant()) {
                        if (moe_ctx->host_offsets) {
                            selection_info.reset();
                            selection_info.num_total = moe_ctx->num_experts;
                            selection_info.expert_to_compact.assign(moe_ctx->num_experts, -1);
                            for (int e = 0; e < moe_ctx->num_experts; ++e) {
                                int tokens_e = moe_ctx->host_offsets[e + 1] - moe_ctx->host_offsets[e];
                                if (tokens_e <= 0) continue;
                                selection_info.expert_to_compact[e] = static_cast<int>(selection_info.active_experts.size());
                                selection_info.active_experts.push_back(e);
                            }
                            selection_info.num_active = static_cast<int>(selection_info.active_experts.size());
                            selection_info.enabled = selection_info.num_active > 0;
                        } else if (moe_ctx->expert_indices) {
                            selection_info.build_from_router_output(*moe_ctx->expert_indices, moe_ctx->num_experts, stream);
                        }

                        if (selection_info.enabled) {
                            mBnBWeightProvider->dequantize_selected_experts(layer_idx, selection_info, stream);
                        }
                    }

                    auto& base_weights = mBaseModel->weights_manager().get_block(layer_idx, stream);

                    using WeightsType = std::remove_reference_t<decltype(base_weights)>;
                    if constexpr (has_experts<WeightsType>::value) {
                        if (base_weights.experts.use_batched) {
                            // When using selective dequant with global indexing, we must still
                            // iterate over ALL experts. The GEMM will skip experts with 0 tokens,
                            // but we need to use global expert indices to match the weight buffer layout.
                            // The weights are dequantized at global positions, so num_experts must
                            // always be the full count (not the active count).
                            const int gemm_num_experts = moe_ctx->num_experts;

                            // DEBUG: Disabled for now
                            // static int fwd_hook_calls = 0;
                            // if (fwd_hook_calls < 3) { ... }

                            if (moe_gated) {
                                detail::grouped_fast_expert_lora_forward(
                                    const_cast<Tensor&>(*moe_ctx->expert_outputs),
                                    *moe_ctx->permuted_input,
                                    base_weights.experts.gate_up_proj,
                                    base_weights.experts.down_proj,
                                    lora_block.moe.grouped,
                                    mLoRARunState->moe_lora_gate,   // contiguous buffer for gate output
                                    mLoRARunState->moe_lora_up,     // contiguous buffer for up output
                                    *moe_ctx->expert_offsets,
                                    scaling,
                                    gemm_num_experts, C, moe_D, rank,
                                    mLoRARunState->moe_lora_intermediate1,
                                    mLoRARunState->moe_lora_intermediate2,
                                    mLoRARunState->moe_lora_gate_up,
                                    rs.CublasHandle,
                                    stream,
                                    moe_ctx->host_offsets,
                                    selection_info.enabled ? &selection_info : nullptr,
                                    layer_idx
                                );
                            } else {
                                detail::grouped_fast_expert_lora_forward_nongated(
                                    const_cast<Tensor&>(*moe_ctx->expert_outputs),
                                    *moe_ctx->permuted_input,
                                    base_weights.experts.gate_up_proj,
                                    base_weights.experts.down_proj,
                                    lora_block.moe.grouped,
                                    mLoRARunState->moe_lora_up,     // contiguous buffer for up output
                                    *moe_ctx->expert_offsets,
                                    scaling,
                                    gemm_num_experts, C, moe_D, rank,
                                    mLoRARunState->moe_lora_intermediate1,
                                    mLoRARunState->moe_lora_intermediate2, // activation buffer
                                    rs.CublasHandle,
                                    stream,
                                    moe_ctx->host_offsets,
                                    selection_info.enabled ? &selection_info : nullptr,
                                    cfg.activation_type
                                );
                            }

                            // Copy base gate_up to context for backward pass reconstruction.
                            // grouped_fast_expert_lora_forward writes the base projection (before LoRA)
                            // to moe_lora_gate_up. The backward pass needs this to reconstruct
                            // the forward activations for correct gradient computation.
                            if (moe_ctx->expert_gate_up && moe_ctx->expert_gate_up->Data) {
                                const Tensor& base_gate_up = moe_gated ? mLoRARunState->moe_lora_gate_up
                                                                      : mLoRARunState->moe_lora_up;
                                CUDA_CHECK(cudaMemcpyAsync(
                                    moe_ctx->expert_gate_up->Data,
                                    base_gate_up.Data,
                                    base_gate_up.bytes(),
                                    cudaMemcpyDeviceToDevice,
                                    stream));
                            }

                            moe_ctx->handled = true;
                        }
                    }
                }
            } break;
            case ForwardHookPoint::AfterQKVProjection: {
                // Projection types: 0=Q, 1=K, 2=V, 3=O, 4=Up, 5=Gate, 6=Down, 7=Router
                if (lora_block.attention.q.has_value()) {
                    const bool trace = detail::lora_fwd_nan_trace_should_log(layer_idx);
                    if (trace) {
                        const int samples = detail::lora_fwd_nan_trace_samples();
                        detail::lora_trace_tensor("LORA_QKV_PRE_Q", layer_idx, acts.qkv, 0, samples, stream);
                        detail::lora_trace_tensor("LORA_QKV_IN_Q", layer_idx, acts.ln1, 0, samples, stream);
                        detail::lora_trace_tensor("LORA_QKV_A_Q", layer_idx, lora_block.attention.q->A, 0, samples, stream);
                        detail::lora_trace_tensor("LORA_QKV_B_Q", layer_idx, lora_block.attention.q->B, 0, samples, stream);
                    }
                    detail::apply_lora_contribution(acts.qkv, 0, acts.ln1, lora_block.attention.q.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(0), is_training,
                                                    B * T, C, Hq * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    if (trace) {
                        const int samples = detail::lora_fwd_nan_trace_samples();
                        detail::lora_trace_tensor("LORA_QKV_POST_Q", layer_idx, acts.qkv, 0, samples, stream);
                    }
                }
                if (lora_block.attention.k.has_value()) {
                    const bool trace = detail::lora_fwd_nan_trace_should_log(layer_idx);
                    const long k_offset = (long)Hq * Hs;
                    if (trace) {
                        const int samples = detail::lora_fwd_nan_trace_samples();
                        detail::lora_trace_tensor("LORA_QKV_PRE_K", layer_idx, acts.qkv, k_offset, samples, stream);
                        detail::lora_trace_tensor("LORA_QKV_IN_K", layer_idx, acts.ln1, 0, samples, stream);
                        detail::lora_trace_tensor("LORA_QKV_A_K", layer_idx, lora_block.attention.k->A, 0, samples, stream);
                        detail::lora_trace_tensor("LORA_QKV_B_K", layer_idx, lora_block.attention.k->B, 0, samples, stream);
                    }
                    detail::apply_lora_contribution(acts.qkv, Hq * Hs, acts.ln1, lora_block.attention.k.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(1), is_training,
                                                    B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    if (trace) {
                        const int samples = detail::lora_fwd_nan_trace_samples();
                        detail::lora_trace_tensor("LORA_QKV_POST_K", layer_idx, acts.qkv, k_offset, samples, stream);
                    }
                }
                if (lora_block.attention.v.has_value()) {
                    const bool trace = detail::lora_fwd_nan_trace_should_log(layer_idx);
                    const long v_offset = (long)(Hq + Hkv) * Hs;
                    if (trace) {
                        const int samples = detail::lora_fwd_nan_trace_samples();
                        detail::lora_trace_tensor("LORA_QKV_PRE_V", layer_idx, acts.qkv, v_offset, samples, stream);
                        detail::lora_trace_tensor("LORA_QKV_IN_V", layer_idx, acts.ln1, 0, samples, stream);
                        detail::lora_trace_tensor("LORA_QKV_A_V", layer_idx, lora_block.attention.v->A, 0, samples, stream);
                        detail::lora_trace_tensor("LORA_QKV_B_V", layer_idx, lora_block.attention.v->B, 0, samples, stream);
                    }
                    detail::apply_lora_contribution(acts.qkv, (Hq + Hkv) * Hs, acts.ln1, lora_block.attention.v.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(2), is_training,
                                                    B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    if (trace) {
                        const int samples = detail::lora_fwd_nan_trace_samples();
                        detail::lora_trace_tensor("LORA_QKV_POST_V", layer_idx, acts.qkv, v_offset, samples, stream);
                    }
                }
            } break;
            case ForwardHookPoint::AfterAttnOutProjection: {
                if (lora_block.attention.o.has_value()) {
                    detail::apply_lora_contribution(acts.att_out, 0, acts.att, lora_block.attention.o.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(3), is_training,
                                                    B * T, Hq * Hs, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case ForwardHookPoint::AfterMLPUpProjection: {
                if (lora_block.mlp.up.has_value()) {
                    detail::apply_lora_contribution(acts.mlp_up, 0, acts.ln2, lora_block.mlp.up.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(4), is_training,
                                                    B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.mlp.gate.has_value()) {
                    detail::apply_lora_contribution(acts.mlp_up, D, acts.ln2, lora_block.mlp.gate.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(5), is_training,
                                                    B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case ForwardHookPoint::AfterMLPDownProjection: {
                if (lora_block.mlp.down.has_value()) {
                    detail::apply_lora_contribution(acts.mlp_down, 0, acts.swiglu, lora_block.mlp.down.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(6), is_training,
                                                    B * T, D, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case ForwardHookPoint::AfterRouterProjection: {
                // Apply router LoRA: logits += scaling * (input @ A^T @ B^T)
                if (lora_block.router.has_value() && lora_block.router->has_value() && context) {
                    auto* router_ctx = static_cast<MoERouterContext*>(context);
                    const int E = router_ctx->num_experts;
                    // Router logits are in FP32 for numerical stability, but we compute LoRA in work dtype
                    // and add to the FP32 logits
                    detail::apply_lora_contribution_fp32(
                        *router_ctx->logits, 0, *router_ctx->input, *lora_block.router,
                        mLoRARunState->intermediate, mLoRARunState->slice,
                        scaling, dropout, get_dropout_seed(7), is_training,
                        B * T, C, E, rank,
                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
        }
    };

    mBaseModel->forward_with_hook(inputs, position_ids, comm, micro_step, hook);
}

template<typename Block>
float ModularLoRAModel<Block>::validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) {
    if (!lora_enabled()) {
        return mBaseModel->validate(inputs, position_ids, targets, comm, micro_step);
    }

    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

    auto full_hook = [this](int layer_idx, cudaStream_t stream, ForwardHookPoint point, void* context) {
        const auto& cfg = mBaseModel->config();
        auto& rs = mBaseModel->run_state();
        const int B = (int)rs.B;
        const int T = (int)rs.T;
        const int C = (int)cfg.HiddenSize;
        const int D = (int)cfg.IntermediateSize;
        // For MoE models, experts have a different intermediate size
        const int moe_D = (cfg.moe_config && cfg.moe_config->moe_intermediate_size > 0) ? cfg.moe_config->moe_intermediate_size : D;
        const bool moe_gated = is_gated_activation(cfg.activation_type);
        const int Hq = (int)cfg.NumQueryHeads;
        const int Hkv = (int)cfg.NumKeyValHeads;
        const int Hs = (int)cfg.head_size();
        const int rank = mLoRAConfig.rank;
        const float scaling = mLoRAConfig.scaling();

        auto& acts = rs.simplified_acts(layer_idx);
        auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);

        switch (point) {
            case ForwardHookPoint::MoEExpertGroupManual: {
                if (lora_block.moe.use_grouped && lora_block.moe.has_any() && context) {
                    auto* moe_ctx = static_cast<MoEGroupedContext*>(context);

                    // Selective expert dequantization: only dequantize router-selected experts
                    SelectiveExpertInfo selection_info;
                    if (mBnBWeightProvider && mBnBWeightProvider->use_selective_dequant()) {
                        if (moe_ctx->host_offsets) {
                            selection_info.reset();
                            selection_info.num_total = moe_ctx->num_experts;
                            selection_info.expert_to_compact.assign(moe_ctx->num_experts, -1);
                            for (int e = 0; e < moe_ctx->num_experts; ++e) {
                                int tokens_e = moe_ctx->host_offsets[e + 1] - moe_ctx->host_offsets[e];
                                if (tokens_e <= 0) continue;
                                selection_info.expert_to_compact[e] = static_cast<int>(selection_info.active_experts.size());
                                selection_info.active_experts.push_back(e);
                            }
                            selection_info.num_active = static_cast<int>(selection_info.active_experts.size());
                            selection_info.enabled = selection_info.num_active > 0;
                        } else if (moe_ctx->expert_indices) {
                            selection_info.build_from_router_output(*moe_ctx->expert_indices, moe_ctx->num_experts, stream);
                        }

                        if (selection_info.enabled) {
                            mBnBWeightProvider->dequantize_selected_experts(layer_idx, selection_info, stream);
                        }
                    }

                    auto& base_weights = mBaseModel->weights_manager().get_block(layer_idx, stream);

                    using WeightsType = std::remove_reference_t<decltype(base_weights)>;
                    if constexpr (has_experts<WeightsType>::value) {
                        if (base_weights.experts.use_batched) {
                            // When using selective dequant with global indexing, we must still
                            // iterate over ALL experts. The GEMM will skip experts with 0 tokens.
                            const int gemm_num_experts = moe_ctx->num_experts;

                            if (moe_gated) {
                                detail::grouped_fast_expert_lora_forward(
                                    const_cast<Tensor&>(*moe_ctx->expert_outputs),
                                    *moe_ctx->permuted_input,
                                    base_weights.experts.gate_up_proj,
                                    base_weights.experts.down_proj,
                                    lora_block.moe.grouped,
                                    mLoRARunState->moe_lora_gate,   // contiguous buffer for gate output
                                    mLoRARunState->moe_lora_up,     // contiguous buffer for up output
                                    *moe_ctx->expert_offsets,
                                    scaling,
                                    gemm_num_experts, C, moe_D, rank,
                                    mLoRARunState->moe_lora_intermediate1,
                                    mLoRARunState->moe_lora_intermediate2,
                                    mLoRARunState->moe_lora_gate_up,
                                    rs.CublasHandle,
                                    stream,
                                    moe_ctx->host_offsets,
                                    selection_info.enabled ? &selection_info : nullptr,
                                    layer_idx
                                );
                            } else {
                                detail::grouped_fast_expert_lora_forward_nongated(
                                    const_cast<Tensor&>(*moe_ctx->expert_outputs),
                                    *moe_ctx->permuted_input,
                                    base_weights.experts.gate_up_proj,
                                    base_weights.experts.down_proj,
                                    lora_block.moe.grouped,
                                    mLoRARunState->moe_lora_up,     // contiguous buffer for up output
                                    *moe_ctx->expert_offsets,
                                    scaling,
                                    gemm_num_experts, C, moe_D, rank,
                                    mLoRARunState->moe_lora_intermediate1,
                                    mLoRARunState->moe_lora_intermediate2, // activation buffer
                                    rs.CublasHandle,
                                    stream,
                                    moe_ctx->host_offsets,
                                    selection_info.enabled ? &selection_info : nullptr,
                                    cfg.activation_type
                                );
                            }
                            moe_ctx->handled = true;
                        }
                    }
                }
            } break;
            case ForwardHookPoint::AfterQKVProjection: {
                // Validation: no dropout (is_training=false)
                if (lora_block.attention.q.has_value()) {
                    const bool trace = detail::lora_fwd_nan_trace_should_log(layer_idx);
                    if (trace) {
                        const int samples = detail::lora_fwd_nan_trace_samples();
                        detail::lora_trace_tensor("LORA_QKV_PRE_Q", layer_idx, acts.qkv, 0, samples, stream);
                        detail::lora_trace_tensor("LORA_QKV_IN_Q", layer_idx, acts.ln1, 0, samples, stream);
                        detail::lora_trace_tensor("LORA_QKV_A_Q", layer_idx, lora_block.attention.q->A, 0, samples, stream);
                        detail::lora_trace_tensor("LORA_QKV_B_Q", layer_idx, lora_block.attention.q->B, 0, samples, stream);
                    }
                    detail::apply_lora_contribution(acts.qkv, 0, acts.ln1, lora_block.attention.q.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, Hq * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    if (trace) {
                        const int samples = detail::lora_fwd_nan_trace_samples();
                        detail::lora_trace_tensor("LORA_QKV_POST_Q", layer_idx, acts.qkv, 0, samples, stream);
                    }
                }
                if (lora_block.attention.k.has_value()) {
                    const bool trace = detail::lora_fwd_nan_trace_should_log(layer_idx);
                    const long k_offset = (long)Hq * Hs;
                    if (trace) {
                        const int samples = detail::lora_fwd_nan_trace_samples();
                        detail::lora_trace_tensor("LORA_QKV_PRE_K", layer_idx, acts.qkv, k_offset, samples, stream);
                        detail::lora_trace_tensor("LORA_QKV_IN_K", layer_idx, acts.ln1, 0, samples, stream);
                        detail::lora_trace_tensor("LORA_QKV_A_K", layer_idx, lora_block.attention.k->A, 0, samples, stream);
                        detail::lora_trace_tensor("LORA_QKV_B_K", layer_idx, lora_block.attention.k->B, 0, samples, stream);
                    }
                    detail::apply_lora_contribution(acts.qkv, Hq * Hs, acts.ln1, lora_block.attention.k.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    if (trace) {
                        const int samples = detail::lora_fwd_nan_trace_samples();
                        detail::lora_trace_tensor("LORA_QKV_POST_K", layer_idx, acts.qkv, k_offset, samples, stream);
                    }
                }
                if (lora_block.attention.v.has_value()) {
                    const bool trace = detail::lora_fwd_nan_trace_should_log(layer_idx);
                    const long v_offset = (long)(Hq + Hkv) * Hs;
                    if (trace) {
                        const int samples = detail::lora_fwd_nan_trace_samples();
                        detail::lora_trace_tensor("LORA_QKV_PRE_V", layer_idx, acts.qkv, v_offset, samples, stream);
                        detail::lora_trace_tensor("LORA_QKV_IN_V", layer_idx, acts.ln1, 0, samples, stream);
                        detail::lora_trace_tensor("LORA_QKV_A_V", layer_idx, lora_block.attention.v->A, 0, samples, stream);
                        detail::lora_trace_tensor("LORA_QKV_B_V", layer_idx, lora_block.attention.v->B, 0, samples, stream);
                    }
                    detail::apply_lora_contribution(acts.qkv, (Hq + Hkv) * Hs, acts.ln1, lora_block.attention.v.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    if (trace) {
                        const int samples = detail::lora_fwd_nan_trace_samples();
                        detail::lora_trace_tensor("LORA_QKV_POST_V", layer_idx, acts.qkv, v_offset, samples, stream);
                    }
                }
            } break;
            case ForwardHookPoint::AfterAttnOutProjection: {
                if (lora_block.attention.o.has_value()) {
                    detail::apply_lora_contribution(acts.att_out, 0, acts.att, lora_block.attention.o.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, Hq * Hs, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case ForwardHookPoint::AfterMLPUpProjection: {
                if (lora_block.mlp.up.has_value()) {
                    detail::apply_lora_contribution(acts.mlp_up, 0, acts.ln2, lora_block.mlp.up.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.mlp.gate.has_value()) {
                    detail::apply_lora_contribution(acts.mlp_up, D, acts.ln2, lora_block.mlp.gate.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case ForwardHookPoint::AfterMLPDownProjection: {
                if (lora_block.mlp.down.has_value()) {
                    detail::apply_lora_contribution(acts.mlp_down, 0, acts.swiglu, lora_block.mlp.down.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, D, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
        }
    };

    return mBaseModel->validate_with_hook(inputs, position_ids, targets, comm, micro_step, full_hook);
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_MODEL_FORWARD_H
