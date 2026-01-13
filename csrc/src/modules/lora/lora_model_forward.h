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

namespace modules {

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

    auto hook = [this](int layer_idx, cudaStream_t stream, ForwardHookPoint point, void* context) {
        const auto& cfg = mBaseModel->config();
        auto& rs = mBaseModel->run_state();
        const int B = (int)rs.B;
        const int T = (int)rs.T;
        const int C = (int)cfg.HiddenSize;
        const int D = (int)cfg.IntermediateSize;
        // For MoE models, experts have a different intermediate size
        const int moe_D = (cfg.moe_config && cfg.moe_config->moe_intermediate_size > 0) ? cfg.moe_config->moe_intermediate_size : D;
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
                                selection_info.enabled ? &selection_info : nullptr
                            );
                            moe_ctx->handled = true;
                        }
                    }
                }
            } break;
            case ForwardHookPoint::AfterQKVProjection: {
                if (lora_block.attention.q.has_value()) {
                    detail::apply_lora_contribution(acts.qkv, 0, acts.ln1, lora_block.attention.q.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, C, Hq * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.attention.k.has_value()) {
                    detail::apply_lora_contribution(acts.qkv, Hq * Hs, acts.ln1, lora_block.attention.k.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.attention.v.has_value()) {
                    detail::apply_lora_contribution(acts.qkv, (Hq + Hkv) * Hs, acts.ln1, lora_block.attention.v.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case ForwardHookPoint::AfterAttnOutProjection: {
                if (lora_block.attention.o.has_value()) {
                    detail::apply_lora_contribution(acts.att_out, 0, acts.att, lora_block.attention.o.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, Hq * Hs, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case ForwardHookPoint::AfterMLPUpProjection: {
                if (lora_block.mlp.up.has_value()) {
                    detail::apply_lora_contribution(acts.mlp_up, 0, acts.ln2, lora_block.mlp.up.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.mlp.gate.has_value()) {
                    detail::apply_lora_contribution(acts.mlp_up, D, acts.ln2, lora_block.mlp.gate.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case ForwardHookPoint::AfterMLPDownProjection: {
                if (lora_block.mlp.down.has_value()) {
                    detail::apply_lora_contribution(acts.mlp_down, 0, acts.swiglu, lora_block.mlp.down.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, D, C, rank,
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
                                selection_info.enabled ? &selection_info : nullptr
                            );
                            moe_ctx->handled = true;
                        }
                    }
                }
            } break;
            case ForwardHookPoint::AfterQKVProjection: {
                if (lora_block.attention.q.has_value()) {
                    detail::apply_lora_contribution(acts.qkv, 0, acts.ln1, lora_block.attention.q.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, C, Hq * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.attention.k.has_value()) {
                    detail::apply_lora_contribution(acts.qkv, Hq * Hs, acts.ln1, lora_block.attention.k.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.attention.v.has_value()) {
                    detail::apply_lora_contribution(acts.qkv, (Hq + Hkv) * Hs, acts.ln1, lora_block.attention.v.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case ForwardHookPoint::AfterAttnOutProjection: {
                if (lora_block.attention.o.has_value()) {
                    detail::apply_lora_contribution(acts.att_out, 0, acts.att, lora_block.attention.o.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, Hq * Hs, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case ForwardHookPoint::AfterMLPUpProjection: {
                if (lora_block.mlp.up.has_value()) {
                    detail::apply_lora_contribution(acts.mlp_up, 0, acts.ln2, lora_block.mlp.up.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.mlp.gate.has_value()) {
                    detail::apply_lora_contribution(acts.mlp_up, D, acts.ln2, lora_block.mlp.gate.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case ForwardHookPoint::AfterMLPDownProjection: {
                if (lora_block.mlp.down.has_value()) {
                    detail::apply_lora_contribution(acts.mlp_down, 0, acts.swiglu, lora_block.mlp.down.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, D, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
        }
    };

    return mBaseModel->validate_with_hook(inputs, position_ids, targets, comm, micro_step, full_hook);
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_MODEL_FORWARD_H
