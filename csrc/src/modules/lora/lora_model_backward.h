// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_MODEL_BACKWARD_H
#define SUROGATE_SRC_MODULES_LORA_LORA_MODEL_BACKWARD_H

#include "lora_model_core.h"
#include "lora_model_utils.h"
#include "lora_utils.h"
#include "modules/lora/fast_expert_lora.h"
#include "modules/moe/moe_types.h"

namespace modules {

template<typename Block>
void ModularLoRAModel<Block>::backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) {
    if (!lora_enabled()) {
        mBaseModel->backward(inputs, targets, comm, grad_accum_steps, micro_step);
        return;
    }

    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

    auto& rs = mBaseModel->run_state();
    cudaStream_t main_stream = rs.MainStream;

    mLoRAGrads->start_micro_step(main_stream, micro_step, grad_accum_steps);

    auto hook = [this, &comm](int layer_idx, bool accumulate, cudaStream_t stream, BackwardHookPoint point, void* context) {
        const int B = (int)mBaseModel->run_state().B;
        const int T = (int)mBaseModel->run_state().T;
        auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);

        switch (point) {
            case BackwardHookPoint::MoEExpertGroupManual: {
                if (lora_block.moe.use_grouped && lora_block.moe.has_any() && context) {
                    auto* moe_ctx = static_cast<MoEGroupedContext*>(context);
                    bool lora_accum = false;
                    auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                    lora_accum = lora_accum || accumulate;

                    auto& base_weights = mBaseModel->weights_manager().get_block(layer_idx, stream);
                    // Selective expert dequantization (QLoRA MoE):
                    // In forward, we dequantize only router-selected experts into compact buffers.
                    // Backward must dequantize the same selection for the current layer; otherwise
                    // grouped GEMMs will skip tokens for experts not present in the compact buffers,
                    // leaving gradients uninitialized and causing extreme/unstable norms.
                    SelectiveExpertInfo selection_info;
                    const SelectiveExpertInfo* selection_info_ptr = nullptr;
                    if (mBnBWeightProvider && mBnBWeightProvider->use_selective_dequant()) {
                        if (moe_ctx->host_offsets) {
                            // Fast path: build selection from host-side expert offsets (no D2H sync).
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
                            // Slow fallback: copy (BT, top_k) expert indices to host and dedupe.
                            selection_info.build_from_router_output(*moe_ctx->expert_indices, moe_ctx->num_experts, stream);
                        }

                        if (selection_info.enabled) {
                            mBnBWeightProvider->dequantize_selected_experts(layer_idx, selection_info, stream);
                            selection_info_ptr = &selection_info;
                        }
                    }
                    const auto& cfg = mBaseModel->config();
                    const int C = (int)cfg.HiddenSize;
                    const int D = (int)cfg.IntermediateSize;
                    // For MoE models, experts have a different intermediate size
                    const int moe_D = (cfg.moe_config && cfg.moe_config->moe_intermediate_size > 0) ? cfg.moe_config->moe_intermediate_size : D;
                    const int rank = mLoRAConfig.rank;

                    using WeightsType = std::remove_reference_t<decltype(base_weights)>;
                    if constexpr (has_experts<WeightsType>::value) {
                        if (base_weights.experts.use_batched && lora_grads.moe.use_grouped) {
                            // Recompute (gate, up) activations for THIS layer into the shared LoRA buffers.
                            //
                            // grouped_fast_expert_lora_backward mutates (total_gate, total_up) in-place via
                            // silu_mul_backward_inplace(). These buffers are shared across layers, so we must
                            // reconstruct them per layer; otherwise earlier layers would see stale values
                            // (often from the last processed layer), leading to incorrect gradients and
                            // extreme/unstable global norms.
                            //
                            // Base expert_gate_up for this layer is available in moe_ctx->expert_gate_up.
                            // We split it, then add the current LoRA contributions for gate/up exactly as in forward.
                            if (moe_ctx->expert_gate_up) {
                                const int total_tokens = moe_ctx->total_tokens;
                                split_gate_up(*moe_ctx->expert_gate_up,
                                              mLoRARunState->moe_lora_up,
                                              mLoRARunState->moe_lora_gate,
                                              total_tokens, moe_D, stream);

                                const bool use_selective = selection_info_ptr && selection_info_ptr->enabled;
                                const int* active_indices = use_selective ? selection_info_ptr->active_experts.data() : nullptr;
                                const int gemm_num_experts = use_selective ? selection_info_ptr->num_active : moe_ctx->num_experts;

                                auto add_grouped_lora = [&](const std::optional<LoRAGroupedLayerWeights<Tensor>>& lora_layer,
                                                           Tensor& out) {
                                    if (!lora_layer.has_value() || !lora_layer->has_value()) return;

                                    if (moe_ctx->permuted_input->DType != lora_layer->A.DType ||
                                        moe_ctx->permuted_input->DType != lora_layer->B.DType ||
                                        moe_ctx->permuted_input->DType != out.DType ||
                                        moe_ctx->permuted_input->DType != mLoRARunState->moe_lora_intermediate1.DType) {
                                        throw std::runtime_error("MoE LoRA backward: dtype mismatch while reconstructing gate/up. "
                                                                 "Set lora_dtype='bf16' to match activation dtype.");
                                    }

                                    // lora_intermediate1: (tokens, rank) = x @ A^T
                                    if (moe_ctx->permuted_input->DType == ETensorDType::BF16) {
                                        moe_grouped_gemm(
                                            mLoRARunState->moe_lora_intermediate1.get<nv_bfloat16>(),
                                            moe_ctx->permuted_input->get<nv_bfloat16>(),
                                            lora_layer->A.get<nv_bfloat16>(),
                                            moe_ctx->expert_offsets->get<int>(),
                                            moe_ctx->num_experts,
                                            /*M=*/rank,
                                            /*K=*/C,
                                            mBaseModel->run_state().CublasHandle,
                                            stream,
                                            moe_ctx->host_offsets,
                                            /*alpha=*/1.0f,
                                            /*beta=*/0.0f,
                                            EMMTranspose::TN,
                                            active_indices,
                                            /*weight_is_compact=*/false,
                                            gemm_num_experts
                                        );

                                        // out += scaling * (lora_intermediate1 @ B^T)
                                        moe_grouped_gemm(
                                            out.get<nv_bfloat16>(),
                                            mLoRARunState->moe_lora_intermediate1.get<nv_bfloat16>(),
                                            lora_layer->B.get<nv_bfloat16>(),
                                            moe_ctx->expert_offsets->get<int>(),
                                            moe_ctx->num_experts,
                                            /*M=*/moe_D,
                                            /*K=*/rank,
                                            mBaseModel->run_state().CublasHandle,
                                            stream,
                                            moe_ctx->host_offsets,
                                            /*alpha=*/mLoRAConfig.scaling(),
                                            /*beta=*/1.0f,
                                            EMMTranspose::TN,
                                            active_indices,
                                            /*weight_is_compact=*/false,
                                            gemm_num_experts
                                        );
                                    } else {
                                        moe_grouped_gemm(
                                            mLoRARunState->moe_lora_intermediate1.get<float>(),
                                            moe_ctx->permuted_input->get<float>(),
                                            lora_layer->A.get<float>(),
                                            moe_ctx->expert_offsets->get<int>(),
                                            moe_ctx->num_experts,
                                            /*M=*/rank,
                                            /*K=*/C,
                                            mBaseModel->run_state().CublasHandle,
                                            stream,
                                            moe_ctx->host_offsets,
                                            /*alpha=*/1.0f,
                                            /*beta=*/0.0f,
                                            EMMTranspose::TN,
                                            active_indices,
                                            /*weight_is_compact=*/false,
                                            gemm_num_experts
                                        );

                                        moe_grouped_gemm(
                                            out.get<float>(),
                                            mLoRARunState->moe_lora_intermediate1.get<float>(),
                                            lora_layer->B.get<float>(),
                                            moe_ctx->expert_offsets->get<int>(),
                                            moe_ctx->num_experts,
                                            /*M=*/moe_D,
                                            /*K=*/rank,
                                            mBaseModel->run_state().CublasHandle,
                                            stream,
                                            moe_ctx->host_offsets,
                                            /*alpha=*/mLoRAConfig.scaling(),
                                            /*beta=*/1.0f,
                                            EMMTranspose::TN,
                                            active_indices,
                                            /*weight_is_compact=*/false,
                                            gemm_num_experts
                                        );
                                    }
                                };

                                add_grouped_lora(lora_block.moe.grouped.gate, mLoRARunState->moe_lora_gate);
                                add_grouped_lora(lora_block.moe.grouped.up, mLoRARunState->moe_lora_up);
                            }

                            detail::grouped_fast_expert_lora_backward(
                                lora_grads.moe.grouped,
                                const_cast<Tensor&>(*moe_ctx->d_permuted_input),
                                *moe_ctx->d_expert_outputs,
                                base_weights.experts.gate_up_proj,
                                base_weights.experts.down_proj,
                                lora_block.moe.grouped,
                                mLoRARunState->moe_lora_gate,   // contiguous buffer for gate output
                                mLoRARunState->moe_lora_up,     // contiguous buffer for up output
                                *moe_ctx->permuted_input,
                                *moe_ctx->expert_offsets,
                                mLoRAConfig.scaling(),
                                moe_ctx->num_experts, C, moe_D, rank,
                                lora_accum,
                                mLoRARunState->moe_lora_intermediate1,
                                mLoRARunState->moe_lora_intermediate2,
                                mLoRARunState->moe_lora_gate_up,
                                mBaseModel->run_state().CublasHandle,
                                stream,
                                moe_ctx->host_offsets,
                                selection_info_ptr
                            );
                            moe_ctx->handled = true;
                        }
                    }
                }
            } break;
            case BackwardHookPoint::AfterMLPDownBackward:
                backward_lora_mlp_down(layer_idx, B, T, accumulate, comm, stream);
                break;
            case BackwardHookPoint::AfterMLPUpBackward:
                backward_lora_mlp_up(layer_idx, B, T, accumulate, comm, stream);
                break;
            case BackwardHookPoint::AfterAttnOutBackward:
                backward_lora_attn_out(layer_idx, B, T, accumulate, comm, stream);
                break;
            case BackwardHookPoint::AfterQKVBackward:
                backward_lora_qkv(layer_idx, B, T, accumulate, comm, stream);
                mLoRAGrads->notify_block(layer_idx, stream, comm);
                break;
            default:
                break;
        }
    };

    mBaseModel->backward_with_hook(inputs, targets, comm, grad_accum_steps, micro_step, hook);
    mLoRAGrads->end_micro_step(main_stream, comm);
    // Extend the base-model BackwardDone event to include LoRA gradient reductions.
    CUDA_CHECK(cudaEventRecord(rs.BackwardDone, main_stream));
}

template<typename Block>
void ModularLoRAModel<Block>::backward_lora_qkv(int layer_idx, int B, int T, bool accumulate, NCCLCommunicator& comm, cudaStream_t stream) {
    const auto& cfg = mBaseModel->config();
    const int C = (int)cfg.HiddenSize;
    const int Hq = (int)cfg.NumQueryHeads;
    const int Hkv = (int)cfg.NumKeyValHeads;
    const int Hs = (int)cfg.head_size();
    const int rank = mLoRAConfig.rank;

    auto& rs = mBaseModel->run_state();
    auto& a = rs.simplified_acts(layer_idx);
    auto& da = rs.simplified_grads(layer_idx);

    auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);
    bool lora_accum = false;
    auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
    lora_accum = lora_accum || accumulate;

    // Get ln1 input: either from stored activation or recompute from residual
    Tensor ln1_input;
    if (rs.config().recompute_lora) {
        // Recompute ln1 from the residual input
        Tensor& residual = rs.get_residual(layer_idx, stream);
        auto& block_weights = mBaseModel->weights_manager().get_block(layer_idx, stream);
        ln1_input = recompute_rmsnorm(residual, block_weights.ln1.weight,
                                      cfg.RmsNormEps, B, T, C, stream);
    } else {
        ln1_input = a.ln1;
    }

    // Prepare gradient tensors (use empty tensor if projection not enabled)
    Tensor dA_q{}, dB_q{}, dA_k{}, dB_k{}, dA_v{}, dB_v{};
    LoRALayerWeights<Tensor> lora_q{}, lora_k{}, lora_v{};

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

    detail::backward_lora_qkv_fused(
        dA_q, dB_q,
        dA_k, dB_k,
        dA_v, dB_v,
        da.d_ln1,
        da.d_qkv,
        ln1_input,
        lora_q, lora_k, lora_v,
        mLoRAConfig.scaling(),
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
}

template<typename Block>
void ModularLoRAModel<Block>::backward_lora_attn_out(int layer_idx, int B, int T, bool accumulate, NCCLCommunicator& comm, cudaStream_t stream) {
    const auto& cfg = mBaseModel->config();
    const int C = (int)cfg.HiddenSize;
    const int Hq = (int)cfg.NumQueryHeads;
    const int Hs = (int)cfg.head_size();
    const int rank = mLoRAConfig.rank;

    auto& rs = mBaseModel->run_state();
    auto& a = rs.simplified_acts(layer_idx);
    auto& da = rs.simplified_grads(layer_idx);

    auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);
    if (!lora_block.attention.o.has_value()) return;

    bool lora_accum = false;
    auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
    lora_accum = lora_accum || accumulate;
    if (!lora_grads.attention.o.has_value()) return;

    Tensor x = a.att;
    Tensor dL_dy = da.d_res_att;

    detail::backward_lora_layer(lora_grads.attention.o->A, lora_grads.attention.o->B,
                               da.d_att,
                               dL_dy, 0,
                               x,
                               lora_block.attention.o->A, lora_block.attention.o->B,
                               mLoRAConfig.scaling(),
                               mLoRARunState->intermediate, mLoRARunState->slice,
                               B * T, Hq * Hs, C, rank, lora_accum,
                               rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
}

template<typename Block>
void ModularLoRAModel<Block>::backward_lora_mlp_up(int layer_idx, int B, int T, bool accumulate, NCCLCommunicator& comm, cudaStream_t stream) {
    const auto& cfg = mBaseModel->config();
    const int C = (int)cfg.HiddenSize;
    const int D = (int)cfg.IntermediateSize;
    const int rank = mLoRAConfig.rank;

    auto& rs = mBaseModel->run_state();
    auto& a = rs.simplified_acts(layer_idx);
    auto& da = rs.simplified_grads(layer_idx);

    auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);
    bool lora_accum = false;
    auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
    lora_accum = lora_accum || accumulate;

    // Get ln2 input: either from stored activation or recompute from residual_att
    Tensor ln2_input;
    if (rs.config().recompute_lora) {
        // Recompute ln2 from residual_att (which is residual + att_out)
        auto& block_weights = mBaseModel->weights_manager().get_block(layer_idx, stream);
        ln2_input = recompute_rmsnorm(a.residual_att, block_weights.ln2.weight,
                                      cfg.RmsNormEps, B, T, C, stream);
    } else {
        ln2_input = a.ln2;
    }

    // Prepare gradient tensors (use empty tensor if projection not enabled)
    Tensor dA_up{}, dB_up{}, dA_gate{}, dB_gate{};
    LoRALayerWeights<Tensor> lora_up{}, lora_gate{};

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

    detail::backward_lora_mlp_up_gate_fused(
        dA_up, dB_up,
        dA_gate, dB_gate,
        da.d_ln2,
        da.d_mlp_up,
        ln2_input,
        lora_up, lora_gate,
        mLoRAConfig.scaling(),
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
}

template<typename Block>
void ModularLoRAModel<Block>::backward_lora_mlp_down(int layer_idx, int B, int T, bool accumulate, NCCLCommunicator& comm, cudaStream_t stream) {
    const auto& cfg = mBaseModel->config();
    const int C = (int)cfg.HiddenSize;
    const int D = (int)cfg.IntermediateSize;
    const int rank = mLoRAConfig.rank;

    auto& rs = mBaseModel->run_state();
    auto& a = rs.simplified_acts(layer_idx);
    auto& da = rs.simplified_grads(layer_idx);

    auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);
    if (!lora_block.mlp.down.has_value()) return;

    bool lora_accum = false;
    auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
    lora_accum = lora_accum || accumulate;
    if (!lora_grads.mlp.down.has_value()) return;

    Tensor x = a.swiglu;
    Tensor dL_dy = da.d_res_ffn;

    detail::backward_lora_layer(lora_grads.mlp.down->A, lora_grads.mlp.down->B,
                               da.d_swiglu,
                               dL_dy, 0,
                               x,
                               lora_block.mlp.down->A, lora_block.mlp.down->B,
                               mLoRAConfig.scaling(),
                               mLoRARunState->intermediate, mLoRARunState->slice,
                               B * T, D, C, rank, lora_accum,
                               rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_MODEL_BACKWARD_H
