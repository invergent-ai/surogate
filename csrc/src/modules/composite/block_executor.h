// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Block execution helpers for composable transformer blocks.

#ifndef SUROGATE_SRC_MODULES_COMPOSITE_BLOCK_EXECUTOR_H
#define SUROGATE_SRC_MODULES_COMPOSITE_BLOCK_EXECUTOR_H

#include <cmath>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <vector>

#include "block_spec.h"
#include "modules/model_config.h"
#include "modules/forward_hooks.h"
#include "modules/backward_hooks.h"
#include "modules/moe/moe_types.h"
#include "modules/primitives/recipe_ops.h"
#include "modules/run_state_types.h"
#include "modules/fp8_run_state.h"
#include "modules/fp4_run_state.h"
#include "modules/fp8_scaling_config.h"
#include "modules/weights/weight_manager_types.h"
#include "kernels/kernels.h"

namespace modules {

template<typename Block>
class ModularRunState;

template<typename Block>
class ModularWeightManager;

struct BlockExecutor {
    template<typename Gradients>
    static auto& attention_grads(Gradients& g) {
        if constexpr (requires { g.attention_grads; }) {
            return g.attention_grads;
        } else {
            return g.attention;
        }
    }

    template<typename Gradients>
    static Tensor& ln2_weight_grad(Gradients& g) {
        if constexpr (requires { g.ln2_grads.d_weight; }) {
            return g.ln2_grads.d_weight;
        } else {
            return g.ln2.d_weight;
        }
    }

    template<typename Block>
    static void forward(
        const BlockSpec& spec,
        const ::recipes::Recipe& recipe,
        ModularRunState<Block>& rs,
        typename Block::Weights& weights,
        SimplifiedLayerActivations& acts,
        SimplifiedLayerQuantActivations& quant_acts,
        Tensor& residual,
        int layer_idx,
        const ModelConfig& config,
        const ModelOptions& options,
        cudaStream_t stream,
        FP8ForwardQuantActivations* fp8_fwd_quants,
        FP4ForwardQuantActivations* fp4_fwd_quants,
        ModularWeightManager<Block>* weight_manager,
        bool allow_quant_layer,
        const ForwardHook* hook)
    {
        const int B = (int)rs.B;
        const int T = (int)rs.T;
        const int C = config.HiddenSize;
        const int D = config.IntermediateSize;
        const int M = config.mlp_up_rows();
        const int Hq = config.NumQueryHeads;
        const int Hkv = config.NumKeyValHeads;
        const int Hs = config.head_size();
        const int qkv_channels = config.qkv_channels();
        const int mamba_heads = config.MambaNumHeads;
        const int mamba_head_dim = config.MambaHeadDim;
        const int mamba_state = config.MambaSsmStateSize;
        const int mamba_groups = config.MambaNGroups > 0 ? config.MambaNGroups : 1;
        const int mamba_dim = config.mamba_dim();
        const int mamba_conv_dim = mamba_dim + 2 * mamba_groups * mamba_state;
        const int mamba_proj_size = mamba_dim + mamba_conv_dim + mamba_heads;
        const int mamba_chunk_size = config.MambaChunkSize > 0 ? config.MambaChunkSize : 2048;
        const int mamba_chunks = (T + mamba_chunk_size - 1) / mamba_chunk_size;

        using WeightsType = std::decay_t<decltype(weights)>;
        constexpr bool kHasMoE = has_moe_weights<WeightsType>::value;
        constexpr bool kHasMLP = has_mlp_weights<WeightsType>::value;

        recipe_ops::ForwardQuantView qv{fp8_fwd_quants, fp4_fwd_quants, &quant_acts};

        auto has_op = [&](BlockOp op) {
            for (int i = 0; i < spec.forward_ops_count; ++i) {
                if (spec.forward_ops[i] == op) return true;
            }
            return false;
        };

        const bool has_attention = has_op(BlockOp::Attention) || has_op(BlockOp::AttnOut) || has_op(BlockOp::QKV);
        const bool has_ln2 = has_op(BlockOp::LN2) || has_op(BlockOp::ResidualLN2);
        const bool mlp_uses_ln1 = !has_ln2;
        const bool has_mlp_output = has_op(BlockOp::MLPDown) || has_op(BlockOp::Mamba);

        auto run_ln1 = [&]() {
            if (layer_idx == 0) {
                rmsnorm_forward(acts.ln1, acts.ln1_rstd, residual, weights.ln1.weight,
                                qv.ln1_abs_max(), config.RmsNormEps, B, T, C, stream);
            } else {
                auto& prev = rs.simplified_acts(layer_idx - 1);
                fused_residual_rmsnorm_forward(
                    residual, acts.ln1, acts.ln1_rstd,
                    prev.residual_att, prev.mlp_down, weights.ln1.weight,
                    qv.ln1_abs_max(), config.RmsNormEps, B * T, C, stream);
                if (options.offload_residuals) {
                    rs.mark_residual_ready(layer_idx - 1, stream);
                }
            }
        };

        auto run_qkv = [&]() {
            const int qidx = allow_quant_layer ?
                get_quantizer_index(layer_idx, QuantizerIndex::FWD_LN1) : -1;
            const auto cache = recipe_ops::forward_cached_weights(weight_manager, MatmulOp::QKV, allow_quant_layer);
            recipe_ops::forward_matmul(
                recipe, rs,
                acts.qkv, acts.ln1, weights.attention.qkv_weight,
                weights.attention.qkv_bias.has_value() ? &weights.attention.qkv_bias.value() : nullptr,
                B, T, C, qkv_channels,
                layer_idx, MatmulOp::QKV,
                qv.ln1_inp_quant(), qidx, cache, stream, allow_quant_layer);
        };

        auto run_qk_norm = [&]() {
            recipe_ops::qk_norm_forward(
                acts.qkv, acts.q_rstd, acts.k_rstd, weights.attention,
                config.RmsNormEps, B, T, qkv_channels, Hq, Hkv, Hs, stream);
        };

        auto run_rope = [&]() {
            int* pos_ids_ptr = rs.PositionIDs.template get<int>();
            auto& freq_cis = rs.non_block_activations().freq_cis;
            const bool use_qk_norm = recipe_ops::has_qk_norm(weights.attention);
            Tensor& qkv_for_attn = (use_qk_norm && acts.qkv_rope.Data != nullptr) ? acts.qkv_rope : acts.qkv;
            recipe_ops::rope_forward_dispatch(
                qkv_for_attn, acts.qkv, freq_cis, pos_ids_ptr,
                config, options, B, T, Hq, Hkv, Hs, stream);
        };

        auto run_attention = [&]() {
            const bool use_qk_norm = recipe_ops::has_qk_norm(weights.attention);
            Tensor& qkv_for_attn = (use_qk_norm && acts.qkv_rope.Data != nullptr) ? acts.qkv_rope : acts.qkv;
            recipe_ops::attention_forward(acts.att, acts.lse, qkv_for_attn, rs.CuBlasWorkspace,
                                          rs.CudnnHandle, B, T, Hq, Hkv, Hs, stream);
            recipe_ops::record_abs_max(qv.att_abs_max(), acts.att, rs.DeviceProp, stream);
        };

        auto run_attn_out = [&]() {
            const int qidx = allow_quant_layer ?
                get_quantizer_index(layer_idx, QuantizerIndex::FWD_ATT) : -1;
            const auto cache = recipe_ops::forward_cached_weights(weight_manager, MatmulOp::AttnOut, allow_quant_layer);
            recipe_ops::forward_matmul(
                recipe, rs,
                acts.att_out, acts.att, weights.attention.out_weight,
                nullptr,
                B, T, Hq * Hs, C,
                layer_idx, MatmulOp::AttnOut,
                qv.att_inp_quant(), qidx, cache, stream, allow_quant_layer);
        };

        auto run_residual_ln2 = [&]() {
            fused_residual_rmsnorm_forward(acts.residual_att, acts.ln2, acts.ln2_rstd,
                                           residual, acts.att_out, weights.ln2.weight,
                                           qv.ln2_abs_max(),
                                           config.RmsNormEps, B * T, C, stream);
        };

        auto run_residual_add = [&]() {
            const long N = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(C);
            if (has_attention) {
                vector_add_sr(acts.residual_att, residual, acts.att_out, 1.0f, N, /*seed=*/0, stream);
            } else {
                // No attention path: residual_att should mirror residual.
                vector_add_sr(acts.residual_att, residual, residual, 0.0f, N, /*seed=*/0, stream);
            }
            if (!has_mlp_output) {
                fill_zero(acts.mlp_down, stream);
            }
        };

        auto run_ln2 = [&]() {
            Tensor& ln2_input = spec.ln2_on_residual_att ? acts.residual_att : residual;
            rmsnorm_forward(acts.ln2, acts.ln2_rstd, ln2_input, weights.ln2.weight,
                            qv.ln2_abs_max(), config.RmsNormEps, B, T, C, stream);
        };

        const bool moe_gated = is_gated_activation(config.activation_type);

        struct MoEForwardState {
            Tensor router_logits;
            Tensor router_probs;
            Tensor routing_weights_fp32;
            Tensor expert_indices;
            Tensor expert_counts;
            Tensor expert_offsets;
            Tensor expert_positions;
            Tensor gather_indices;
            Tensor scatter_indices;
            Tensor expert_outputs;
            int num_experts = 0;
            int top_k = 0;
            int expert_D = 0;
            int total_expert_tokens = 0;
        };

        std::optional<MoEForwardState> moe_state;

        auto run_router = [&]() {
            if constexpr (!kHasMoE) {
                throw std::logic_error("BlockExecutor::forward: Router op requires MoE weights");
            } else {
                if (!config.moe_config.has_value()) {
                    throw std::logic_error("BlockExecutor::forward: MoE config missing");
                }
                const auto& moe_cfg = *config.moe_config;
                if (moe_cfg.n_group > 1 || moe_cfg.topk_group > 1) {
                    throw std::logic_error("BlockExecutor::forward: grouped MoE routing is not supported yet");
                }
                const int BT = B * T;
                const int num_experts = moe_cfg.num_experts;
                const int top_k = moe_cfg.top_k;
                const int expert_D = moe_cfg.moe_intermediate_size > 0 ? moe_cfg.moe_intermediate_size : D;
                const int total_expert_tokens = BT * top_k;
                const int dev = rs.DeviceId;

                moe_state.emplace();
                auto& ms = *moe_state;
                ms.num_experts = num_experts;
                ms.top_k = top_k;
                ms.expert_D = expert_D;
                ms.total_expert_tokens = total_expert_tokens;

                // Create flat view of ln2 for routing: (B, T, C) -> (BT, C)
                Tensor flat_ln2;
                flat_ln2.Data = acts.ln2.Data;
                flat_ln2.DType = acts.ln2.DType;
                flat_ln2.Sizes[0] = BT;
                flat_ln2.Sizes[1] = C;
                flat_ln2.Rank = 2;
                flat_ln2.Device = dev;

                // Allocate router temporaries
                ms.router_logits = Tensor{ETensorDType::FP32, {BT, num_experts}, nullptr, nullptr, 2, dev};
                ms.router_probs = Tensor{ETensorDType::FP32, {BT, num_experts}, nullptr, nullptr, 2, dev};
                ms.routing_weights_fp32 = Tensor{ETensorDType::FP32, {BT, top_k}, nullptr, nullptr, 2, dev};
                ms.expert_indices = Tensor{ETensorDType::INT32, {BT, top_k}, nullptr, nullptr, 2, dev};
                ms.expert_counts = Tensor{ETensorDType::INT32, {num_experts}, nullptr, nullptr, 1, dev};
                ms.expert_offsets = Tensor{ETensorDType::INT32, {num_experts + 1}, nullptr, nullptr, 1, dev};
                ms.expert_positions = Tensor{ETensorDType::INT32, {num_experts}, nullptr, nullptr, 1, dev};
                ms.gather_indices = Tensor{ETensorDType::INT32, {total_expert_tokens}, nullptr, nullptr, 1, dev};
                ms.scatter_indices = Tensor{ETensorDType::INT32, {total_expert_tokens}, nullptr, nullptr, 1, dev};

                rs.temp_acquire(ms.router_logits);
                rs.temp_acquire(ms.router_probs);
                rs.temp_acquire(ms.routing_weights_fp32);
                rs.temp_acquire(ms.expert_indices);
                rs.temp_acquire(ms.expert_counts);
                rs.temp_acquire(ms.expert_offsets);
                rs.temp_acquire(ms.expert_positions);
                rs.temp_acquire(ms.gather_indices);
                rs.temp_acquire(ms.scatter_indices);

                fill_zero(ms.expert_counts, stream);
                fill_zero(ms.expert_positions, stream);

                // Router projection: logits = ln2 @ gate^T
                matmul(
                    ms.router_logits, weights.router.gate, flat_ln2, std::nullopt,
                    nullptr, nullptr,
                    rs.CublasLtHandle, rs.CuBlasWorkspace,
                    num_experts, BT, C, EMMTranspose::TN, false,
                    stream
                );

                if (hook) {
                    MoERouterContext router_ctx;
                    router_ctx.logits = &ms.router_logits;
                    router_ctx.input = &flat_ln2;
                    router_ctx.num_experts = num_experts;
                    router_ctx.hidden_size = C;
                    (*hook)(layer_idx, stream, ForwardHookPoint::AfterRouterProjection, &router_ctx);
                }

                // Router activation (softmax or sigmoid)
                if (moe_cfg.use_sigmoid) {
                    moe_sigmoid_forward(
                        ms.router_probs.template get<float>(),
                        ms.router_logits.template get<float>(),
                        BT * num_experts, stream
                    );
                } else {
                    moe_softmax_forward(
                        ms.router_probs.template get<float>(),
                        ms.router_logits.template get<float>(),
                        BT, num_experts, stream
                    );
                }

                // Top-K selection (optionally normalized)
                moe_topk_forward(
                    ms.expert_indices.template get<int>(),
                    ms.routing_weights_fp32.template get<float>(),
                    ms.router_probs.template get<float>(),
                    BT, num_experts, top_k, moe_cfg.norm_topk_prob, stream
                );

                if (moe_cfg.routed_scaling_factor != 1.0f) {
                    moe_scale_forward(
                        ms.routing_weights_fp32.template get<float>(),
                        ms.routing_weights_fp32.template get<float>(),
                        moe_cfg.routed_scaling_factor,
                        BT * top_k,
                        stream
                    );
                }

                // Expert counts
                moe_compute_expert_counts(
                    ms.expert_counts.template get<int>(),
                    ms.expert_indices.template get<int>(),
                    BT, top_k, num_experts, stream
                );

                // Aux and z losses (training stats)
                float* d_aux_loss = nullptr;
                float* d_z_loss = nullptr;
                CUDA_CHECK(cudaMallocAsync(&d_aux_loss, sizeof(float), stream));
                CUDA_CHECK(cudaMallocAsync(&d_z_loss, sizeof(float), stream));
                CUDA_CHECK(cudaMemsetAsync(d_aux_loss, 0, sizeof(float), stream));
                CUDA_CHECK(cudaMemsetAsync(d_z_loss, 0, sizeof(float), stream));

                const float aux_loss_coef = (options.router_aux_loss_coef >= 0.0f)
                    ? options.router_aux_loss_coef : moe_cfg.router_aux_loss_coef;
                moe_compute_aux_loss(
                    d_aux_loss,
                    ms.router_probs.template get<float>(),
                    ms.expert_indices.template get<int>(),
                    BT, num_experts, top_k, aux_loss_coef, stream
                );

                const float z_loss_coef = (options.router_z_loss_coef >= 0.0f)
                    ? options.router_z_loss_coef : moe_cfg.router_z_loss_coef;
                moe_router_z_loss_forward(
                    d_z_loss,
                    ms.router_logits.template get<float>(),
                    BT, num_experts, z_loss_coef, stream
                );

                // Expert offsets and gather/scatter indices
                moe_compute_expert_offsets(
                    ms.expert_offsets.template get<int>(),
                    ms.expert_counts.template get<int>(),
                    num_experts, stream
                );

                // Cache expert offsets + stats on host
                rs.MoeHostExpertOffsets.resize(num_experts + 1);
                float h_aux_loss = 0.0f, h_z_loss = 0.0f;
                CUDA_CHECK(cudaMemcpyAsync(rs.MoeHostExpertOffsets.data(), ms.expert_offsets.template get<int>(),
                                           (num_experts + 1) * sizeof(int),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(&h_aux_loss, d_aux_loss, sizeof(float),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(&h_z_loss, d_z_loss, sizeof(float),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                rs.MoeHostOffsetsValid = true;

                rs.accumulate_moe_stats(h_aux_loss, h_z_loss, num_experts,
                                        ms.expert_counts.template get<int>(), stream);

                CUDA_CHECK(cudaFreeAsync(d_aux_loss, stream));
                CUDA_CHECK(cudaFreeAsync(d_z_loss, stream));

                moe_build_indices(
                    ms.gather_indices.template get<int>(),
                    ms.scatter_indices.template get<int>(),
                    ms.expert_indices.template get<int>(),
                    ms.expert_offsets.template get<int>(),
                    ms.expert_positions.template get<int>(),
                    BT, top_k, num_experts, stream
                );
            }
        };

        auto run_experts = [&]() {
            if constexpr (!kHasMoE) {
                throw std::logic_error("BlockExecutor::forward: Experts op requires MoE weights");
            } else {
                if (!moe_state.has_value()) {
                    throw std::logic_error("BlockExecutor::forward: MoE router state missing");
                }
                auto& ms = *moe_state;
                const int BT = B * T;
                const int dev = rs.DeviceId;

                // Allocate expert outputs first so it can stay on stack for combine (LIFO-friendly).
                Tensor expert_outputs{acts.ln2.DType, {ms.total_expert_tokens, C}, nullptr, nullptr, 2, dev};
                rs.temp_acquire(expert_outputs);

                // Permute tokens to expert-grouped order
                Tensor permuted_input{acts.ln2.DType, {ms.total_expert_tokens, C}, nullptr, nullptr, 2, dev};
                rs.temp_acquire(permuted_input);

                Tensor flat_ln2;
                flat_ln2.Data = acts.ln2.Data;
                flat_ln2.DType = acts.ln2.DType;
                flat_ln2.Sizes[0] = BT;
                flat_ln2.Sizes[1] = C;
                flat_ln2.Rank = 2;
                flat_ln2.Device = dev;

                if (acts.ln2.DType == ETensorDType::BF16) {
                    moe_permute_tokens(
                        permuted_input.template get<nv_bfloat16>(),
                        flat_ln2.template get<nv_bfloat16>(),
                        ms.gather_indices.template get<int>(),
                        ms.total_expert_tokens, BT, C, ms.top_k, stream
                    );
                } else {
                    moe_permute_tokens(
                        permuted_input.template get<float>(),
                        flat_ln2.template get<float>(),
                        ms.gather_indices.template get<int>(),
                        ms.total_expert_tokens, BT, C, ms.top_k, stream
                    );
                }

                // Allocate gate+up buffer after permuted_input for correct stack order
                const int moe_M = moe_gated ? (2 * ms.expert_D) : ms.expert_D;
                Tensor expert_gate_up{acts.ln2.DType, {ms.total_expert_tokens, moe_M}, nullptr, nullptr, 2, dev};
                rs.temp_acquire(expert_gate_up);
                fill_zero(expert_outputs, stream);

                bool hook_handled = false;
                if (hook) {
                    MoEGroupedContext moe_ctx;
                    moe_ctx.expert_offsets = &ms.expert_offsets;
                    moe_ctx.permuted_input = &permuted_input;
                    moe_ctx.expert_gate_up = &expert_gate_up;
                    moe_ctx.expert_outputs = &expert_outputs;
                    moe_ctx.expert_indices = &ms.expert_indices;
                    moe_ctx.host_offsets = rs.MoeHostOffsetsValid ? rs.MoeHostExpertOffsets.data() : nullptr;
                    moe_ctx.num_experts = ms.num_experts;
                    moe_ctx.top_k = ms.top_k;
                    moe_ctx.total_tokens = ms.total_expert_tokens;
                    (*hook)(layer_idx, stream, ForwardHookPoint::MoEExpertGroupManual, &moe_ctx);
                    hook_handled = moe_ctx.handled;
                }

                if (!hook_handled) {
                    const int* host_offsets = rs.MoeHostOffsetsValid ? rs.MoeHostExpertOffsets.data() : nullptr;

                    Tensor expert_act{acts.ln2.DType, {ms.total_expert_tokens, ms.expert_D}, nullptr, nullptr, 2, dev};
                    rs.temp_acquire(expert_act);

                    if (moe_gated) {
                        if (acts.ln2.DType == ETensorDType::BF16) {
                            moe_grouped_gemm_gate_up(
                                expert_gate_up.template get<nv_bfloat16>(),
                                permuted_input.template get<nv_bfloat16>(),
                                weights.experts.gate_up_proj.template get<nv_bfloat16>(),
                                ms.expert_offsets.template get<int>(),
                                ms.num_experts, C, ms.expert_D,
                                rs.CublasHandle, stream, host_offsets
                            );
                        } else {
                            moe_grouped_gemm_gate_up(
                                expert_gate_up.template get<float>(),
                                permuted_input.template get<float>(),
                                weights.experts.gate_up_proj.template get<float>(),
                                ms.expert_offsets.template get<int>(),
                                ms.num_experts, C, ms.expert_D,
                                rs.CublasHandle, stream, host_offsets
                            );
                        }

                        Tensor expert_up{acts.ln2.DType, {ms.total_expert_tokens, ms.expert_D}, nullptr, nullptr, 2, dev};
                        Tensor expert_gate{acts.ln2.DType, {ms.total_expert_tokens, ms.expert_D}, nullptr, nullptr, 2, dev};
                        rs.temp_acquire(expert_up);
                        rs.temp_acquire(expert_gate);

                        split_gate_up(expert_gate_up, expert_up, expert_gate, ms.total_expert_tokens, ms.expert_D, stream);
                        silu_mul_forward(expert_act, expert_gate, expert_up, ms.total_expert_tokens, ms.expert_D, stream);

                        rs.temp_free(expert_gate);
                        rs.temp_free(expert_up);
                    } else {
                        if (acts.ln2.DType == ETensorDType::BF16) {
                            moe_grouped_gemm(
                                expert_gate_up.template get<nv_bfloat16>(),
                                permuted_input.template get<nv_bfloat16>(),
                                weights.experts.gate_up_proj.template get<nv_bfloat16>(),
                                ms.expert_offsets.template get<int>(),
                                ms.num_experts,
                                /*M=*/ms.expert_D,
                                /*K=*/C,
                                rs.CublasHandle,
                                stream,
                                host_offsets,
                                /*alpha=*/1.0f,
                                /*beta=*/0.0f,
                                EMMTranspose::TN
                            );
                        } else {
                            moe_grouped_gemm(
                                expert_gate_up.template get<float>(),
                                permuted_input.template get<float>(),
                                weights.experts.gate_up_proj.template get<float>(),
                                ms.expert_offsets.template get<int>(),
                                ms.num_experts,
                                /*M=*/ms.expert_D,
                                /*K=*/C,
                                rs.CublasHandle,
                                stream,
                                host_offsets,
                                /*alpha=*/1.0f,
                                /*beta=*/0.0f,
                                EMMTranspose::TN
                            );
                        }

                        const long N = static_cast<long>(ms.total_expert_tokens) * static_cast<long>(ms.expert_D);
                        switch (config.activation_type) {
                            case ActivationType::SiLU:
                                silu_forward(expert_act, expert_gate_up, N, stream);
                                break;
                            case ActivationType::ReLU2:
                                relu2_forward(expert_act, expert_gate_up, N, stream);
                                break;
                            default:
                                throw std::logic_error("BlockExecutor::forward: unsupported MoE activation");
                        }
                    }

                    if (acts.ln2.DType == ETensorDType::BF16) {
                        moe_grouped_gemm_down(
                            expert_outputs.template get<nv_bfloat16>(),
                            expert_act.template get<nv_bfloat16>(),
                            weights.experts.down_proj.template get<nv_bfloat16>(),
                            ms.expert_offsets.template get<int>(),
                            ms.num_experts, C, ms.expert_D,
                            rs.CublasHandle, stream, host_offsets
                        );
                    } else {
                        moe_grouped_gemm_down(
                            expert_outputs.template get<float>(),
                            expert_act.template get<float>(),
                            weights.experts.down_proj.template get<float>(),
                            ms.expert_offsets.template get<int>(),
                            ms.num_experts, C, ms.expert_D,
                            rs.CublasHandle, stream, host_offsets
                        );
                    }

                    rs.temp_free(expert_act);
                }

                // Stash expert_outputs for combine
                ms.expert_outputs = expert_outputs;

                rs.temp_free(expert_gate_up);
                rs.temp_free(permuted_input);
            }
        };

        auto run_combine = [&]() {
            if constexpr (!kHasMoE) {
                throw std::logic_error("BlockExecutor::forward: Combine op requires MoE weights");
            } else {
                if (!moe_state.has_value()) {
                    throw std::logic_error("BlockExecutor::forward: MoE expert state missing");
                }
                auto& ms = *moe_state;
                const int BT = B * T;
                const int dev = rs.DeviceId;

                if (acts.ln2.DType == ETensorDType::BF16) {
                    Tensor routing_weights_bf16{ETensorDType::BF16, {BT, ms.top_k}, nullptr, nullptr, 2, dev};
                    rs.temp_acquire(routing_weights_bf16);
                    convert_dtype(routing_weights_bf16.template get<nv_bfloat16>(),
                                  ms.routing_weights_fp32.template get<float>(),
                                  BT * ms.top_k, stream);

                    moe_unpermute_and_combine(
                        acts.mlp_down.template get<nv_bfloat16>(),
                        ms.expert_outputs.template get<nv_bfloat16>(),
                        routing_weights_bf16.template get<nv_bfloat16>(),
                        ms.scatter_indices.template get<int>(),
                        BT, ms.total_expert_tokens, C, ms.top_k, stream
                    );

                    rs.temp_free(routing_weights_bf16);
                } else {
                    moe_unpermute_and_combine(
                        acts.mlp_down.template get<float>(),
                        ms.expert_outputs.template get<float>(),
                        ms.routing_weights_fp32.template get<float>(),
                        ms.scatter_indices.template get<int>(),
                        BT, ms.total_expert_tokens, C, ms.top_k, stream
                    );
                }

                // Optional shared expert (Nemotron/DeepSeek-style)
                if (config.moe_config.has_value()) {
                    const auto& moe_cfg = *config.moe_config;
                    if (moe_cfg.use_shared_expert && weights.shared_expert.has_value()) {
                        const int shared_D = (moe_cfg.shared_expert_size > 0)
                            ? moe_cfg.shared_expert_size
                            : ms.expert_D;

                        Tensor flat_ln2;
                        flat_ln2.Data = acts.ln2.Data;
                        flat_ln2.DType = acts.ln2.DType;
                        flat_ln2.Sizes[0] = BT;
                        flat_ln2.Sizes[1] = C;
                        flat_ln2.Rank = 2;
                        flat_ln2.Device = dev;

                        Tensor shared_act{acts.ln2.DType, {BT, shared_D}, nullptr, nullptr, 2, dev};
                        Tensor shared_out{acts.ln2.DType, {BT, C}, nullptr, nullptr, 2, dev};
                        rs.temp_acquire(shared_act);
                        rs.temp_acquire(shared_out);

                        if (moe_gated) {
                            Tensor shared_gate_up{acts.ln2.DType, {BT, 2 * shared_D}, nullptr, nullptr, 2, dev};
                            Tensor shared_gate{acts.ln2.DType, {BT, shared_D}, nullptr, nullptr, 2, dev};
                            Tensor shared_up{acts.ln2.DType, {BT, shared_D}, nullptr, nullptr, 2, dev};
                            rs.temp_acquire(shared_gate_up);
                            rs.temp_acquire(shared_gate);
                            rs.temp_acquire(shared_up);

                            matmul(shared_gate_up, weights.shared_expert->gate_proj, flat_ln2, std::nullopt,
                                   nullptr, nullptr,
                                   rs.CublasLtHandle, rs.CuBlasWorkspace,
                                   2 * shared_D, BT, C, EMMTranspose::TN, false,
                                   stream);

                            split_gate_up(shared_gate_up, shared_up, shared_gate, BT, shared_D, stream);
                            silu_mul_forward(shared_act, shared_gate, shared_up, BT, shared_D, stream);

                            rs.temp_free(shared_up);
                            rs.temp_free(shared_gate);
                            rs.temp_free(shared_gate_up);
                        } else {
                            matmul(shared_act, weights.shared_expert->gate_proj, flat_ln2, std::nullopt,
                                   nullptr, nullptr,
                                   rs.CublasLtHandle, rs.CuBlasWorkspace,
                                   shared_D, BT, C, EMMTranspose::TN, false,
                                   stream);
                            const long N = static_cast<long>(BT) * static_cast<long>(shared_D);
                            switch (config.activation_type) {
                                case ActivationType::SiLU:
                                    silu_forward(shared_act, shared_act, N, stream);
                                    break;
                                case ActivationType::ReLU2:
                                    relu2_forward(shared_act, shared_act, N, stream);
                                    break;
                                default:
                                    throw std::logic_error("BlockExecutor::forward: unsupported shared expert activation");
                            }
                        }

                        matmul(shared_out, weights.shared_expert->down_proj, shared_act, std::nullopt,
                               nullptr, nullptr,
                               rs.CublasLtHandle, rs.CuBlasWorkspace,
                               C, BT, shared_D, EMMTranspose::TN, false,
                               stream);

                        const long N = static_cast<long>(BT) * static_cast<long>(C);
                        vector_add_sr(acts.mlp_down, acts.mlp_down, shared_out, 1.0f, N, /*seed=*/0, stream);

                        rs.temp_free(shared_out);
                        rs.temp_free(shared_act);
                    }
                }

                // Free temporaries (reverse order where possible)
                rs.temp_free(ms.expert_outputs);
                rs.temp_free(ms.scatter_indices);
                rs.temp_free(ms.gather_indices);
                rs.temp_free(ms.expert_positions);
                rs.temp_free(ms.expert_offsets);
                rs.temp_free(ms.expert_counts);
                rs.temp_free(ms.expert_indices);
                rs.temp_free(ms.routing_weights_fp32);
                rs.temp_free(ms.router_probs);
                rs.temp_free(ms.router_logits);

                moe_state.reset();
            }
        };

        auto run_mlp_up = [&]() {
            if constexpr (!kHasMLP) {
                throw std::logic_error("BlockExecutor::forward: MLPUp op requires dense weights");
            } else {
                Tensor& mlp_in = mlp_uses_ln1 ? acts.ln1 : acts.ln2;
                Tensor* mlp_in_quant = mlp_uses_ln1 ? qv.ln1_inp_quant() : qv.ln2_inp_quant();
                const int qidx = allow_quant_layer ?
                    get_quantizer_index(layer_idx, mlp_uses_ln1 ? QuantizerIndex::FWD_LN1 : QuantizerIndex::FWD_LN2) : -1;
                const auto cache = recipe_ops::forward_cached_weights(weight_manager, MatmulOp::MLPUp, allow_quant_layer);
                recipe_ops::forward_matmul(
                    recipe, rs,
                    acts.mlp_up, mlp_in, weights.mlp_up_weight,
                    nullptr,
                    B, T, C, M,
                    layer_idx, MatmulOp::MLPUp,
                    mlp_in_quant, qidx, cache, stream, allow_quant_layer);
            }
        };

        auto run_swiglu = [&]() {
            if constexpr (!kHasMLP) {
                throw std::logic_error("BlockExecutor::forward: SwiGLU op requires dense weights");
            } else {
                recipe_ops::swiglu_forward(
                    recipe,
                    acts.swiglu,
                    nullptr,
                    acts.mlp_up,
                    qv.swiglu_abs_max(),
                    B, T, D,
                    stream);
            }
        };

        auto run_mlp_act = [&]() {
            if constexpr (!kHasMLP) {
                throw std::logic_error("BlockExecutor::forward: MLPAct op requires dense weights");
            } else {
                switch (config.activation_type) {
                    case ActivationType::SwiGLU:
                        run_swiglu();
                        break;
                    case ActivationType::GeGLU:
                        throw std::logic_error("BlockExecutor::forward: GeGLU activation not implemented");
                    case ActivationType::SiLU: {
                        const long N = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(M);
                        silu_forward(acts.swiglu, acts.mlp_up, N, stream);
                        recipe_ops::record_abs_max(qv.swiglu_abs_max(), acts.swiglu, rs.DeviceProp, stream);
                        break;
                    }
                    case ActivationType::ReLU2: {
                        const long N = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(M);
                        relu2_forward(acts.swiglu, acts.mlp_up, N, stream);
                        recipe_ops::record_abs_max(qv.swiglu_abs_max(), acts.swiglu, rs.DeviceProp, stream);
                        break;
                    }
                    case ActivationType::ReLU:
                    case ActivationType::GeLU:
                        throw std::logic_error("BlockExecutor::forward: Activation type not supported yet");
                    default:
                        throw std::logic_error("BlockExecutor::forward: Unknown activation type");
                }
            }
        };

        auto run_mlp_down = [&]() {
            if constexpr (!kHasMLP) {
                throw std::logic_error("BlockExecutor::forward: MLPDown op requires dense weights");
            } else {
                const int qidx = allow_quant_layer ?
                    get_quantizer_index(layer_idx, QuantizerIndex::FWD_SWIGLU) : -1;
                const auto cache = recipe_ops::forward_cached_weights(weight_manager, MatmulOp::MLPDown, allow_quant_layer);
                recipe_ops::forward_matmul(
                    recipe, rs,
                    acts.mlp_down, acts.swiglu, weights.mlp_down_weight,
                    nullptr,
                    B, T, D, C,
                    layer_idx, MatmulOp::MLPDown,
                    qv.swiglu_inp_quant(), qidx, cache, stream, allow_quant_layer);
            }
        };

        auto run_mamba = [&]() {
            if constexpr (!kHasMLP) {
                throw std::logic_error("BlockExecutor::forward: Mamba op requires dense weights");
            } else {
                if (!weights.mamba.has_value()) {
                    throw std::logic_error("BlockExecutor::forward: Mamba weights missing");
                }
                auto& mw = *weights.mamba;
                const int dev = rs.DeviceId;

                // 1) Input projection
                Tensor proj{acts.ln1.DType, {B, T, mamba_proj_size}, nullptr, nullptr, 3, dev};
                rs.temp_acquire(proj);
                recipe_ops::CachedWeights dummy_cache{};
                recipe_ops::forward_matmul(
                    recipe, rs,
                    proj, acts.ln1, mw.in_proj_weight,
                    mw.in_proj_bias.has_value() ? &mw.in_proj_bias.value() : nullptr,
                    B, T, C, mamba_proj_size,
                    layer_idx, MatmulOp::MLPUp,
                    nullptr, -1, dummy_cache, stream, /*allow_quant=*/false);

                // Split projection: gate, conv_in, delta
                mamba_split_proj(acts.mamba_gate, acts.mamba_conv_in, acts.mamba_delta,
                                 proj, B, T, mamba_dim, mamba_conv_dim, mamba_heads, mamba_head_dim, stream);
                rs.temp_free(proj);

                // 2) Causal conv1d (depthwise) + activation
                Tensor conv_out{acts.ln1.DType, {B, mamba_conv_dim, T}, nullptr, nullptr, 3, dev};
                rs.temp_acquire(conv_out);
                mamba_causal_conv1d_forward(conv_out, acts.mamba_conv_in, mw.conv1d_weight,
                                            mw.conv1d_bias.has_value() ? &mw.conv1d_bias.value() : nullptr,
                                            B, T, mamba_conv_dim, config.MambaConvKernel,
                                            /*silu=*/true, stream);

                // Split conv output: u, B, C
                mamba_split_conv_out(acts.mamba_u, acts.mamba_B, acts.mamba_C,
                                     conv_out, B, T, mamba_dim, mamba_groups, mamba_state, stream);

                // 3) Selective scan
                Tensor A_exp{ETensorDType::FP32, {mamba_dim, mamba_state}, nullptr, nullptr, 2, dev};
                Tensor D_exp{ETensorDType::FP32, {mamba_dim}, nullptr, nullptr, 1, dev};
                Tensor dt_bias_exp{ETensorDType::FP32, {mamba_dim}, nullptr, nullptr, 1, dev};
                Tensor scan_out_bdt{acts.ln1.DType, {B, mamba_dim, T}, nullptr, nullptr, 3, dev};
                rs.temp_acquire(A_exp);
                rs.temp_acquire(D_exp);
                rs.temp_acquire(dt_bias_exp);
                rs.temp_acquire(scan_out_bdt);

                mamba_expand_A(A_exp, mw.A_log, mamba_heads, mamba_head_dim, mamba_state, stream);
                mamba_expand_head_param(D_exp, mw.D, mamba_heads, mamba_head_dim, stream);
                mamba_expand_head_param(dt_bias_exp, mw.dt_bias, mamba_heads, mamba_head_dim, stream);

                mamba_selective_scan_forward(scan_out_bdt,
                                             acts.mamba_u, acts.mamba_delta,
                                             A_exp, acts.mamba_B, acts.mamba_C,
                                             D_exp, dt_bias_exp,
                                             acts.mamba_x,
                                             B, T, mamba_dim, mamba_state, mamba_groups, mamba_chunks, stream);

                mamba_transpose_bdt_to_btd(acts.mamba_scan_out, scan_out_bdt, B, T, mamba_dim, stream);

                // Free scan temporaries (LIFO)
                rs.temp_free(scan_out_bdt);
                rs.temp_free(dt_bias_exp);
                rs.temp_free(D_exp);
                rs.temp_free(A_exp);
                rs.temp_free(conv_out);

                // 4) Gated RMSNorm
                silu_mul_forward(acts.mamba_gated, acts.mamba_gate, acts.mamba_scan_out, B * T, mamba_dim, stream);
                mamba_group_rmsnorm_forward(acts.mamba_normed, acts.mamba_rstd,
                                            acts.mamba_gated, mw.norm_weight,
                                            config.RmsNormEps, B, T, mamba_dim, mamba_groups, stream);

                // 5) Output projection
                recipe_ops::forward_matmul(
                    recipe, rs,
                    acts.mlp_down, acts.mamba_normed, mw.out_proj_weight,
                    mw.out_proj_bias.has_value() ? &mw.out_proj_bias.value() : nullptr,
                    B, T, mamba_dim, C,
                    layer_idx, MatmulOp::MLPDown,
                    nullptr, -1, dummy_cache, stream, /*allow_quant=*/false);
            }
        };

        bool free_ffn_temporaries = false;

        for (int op_idx = 0; op_idx < spec.forward_ops_count; ++op_idx) {
            auto op = spec.forward_ops[op_idx];
            switch (op) {
                case BlockOp::LN1: run_ln1(); break;
                case BlockOp::QKV:
                    run_qkv();
                    if (hook) (*hook)(layer_idx, stream, ForwardHookPoint::AfterQKVProjection, nullptr);
                    break;
                case BlockOp::QKNorm: run_qk_norm(); break;
                case BlockOp::RoPE: run_rope(); break;
                case BlockOp::Attention: run_attention(); break;
                case BlockOp::AttnOut:
                    run_attn_out();
                    if (hook) (*hook)(layer_idx, stream, ForwardHookPoint::AfterAttnOutProjection, nullptr);
                    break;
                case BlockOp::ResidualLN2: run_residual_ln2(); break;
                case BlockOp::ResidualAdd: run_residual_add(); break;
                case BlockOp::LN2: run_ln2(); break;
                case BlockOp::MLPUp:
                    if constexpr (kHasMLP) {
                        if (rs.ffn_temps_on_stack()) {
                            // Use stack-backed temps for recompute-block mode (matches legacy path).
                            if (acts.mlp_up.Data == nullptr) rs.temp_acquire(acts.mlp_up);
                            if (acts.swiglu.Data == nullptr) rs.temp_acquire(acts.swiglu);
                            free_ffn_temporaries = true;
                        }
                    }
                    run_mlp_up();
                    if (hook) (*hook)(layer_idx, stream, ForwardHookPoint::AfterMLPUpProjection, nullptr);
                    break;
                case BlockOp::SwiGLU: run_swiglu(); break;
                case BlockOp::MLPAct: run_mlp_act(); break;
                case BlockOp::MLPDown:
                    run_mlp_down();
                    if (hook) (*hook)(layer_idx, stream, ForwardHookPoint::AfterMLPDownProjection, nullptr);
                    if (free_ffn_temporaries) {
                        rs.temp_free(acts.swiglu);
                        rs.temp_free(acts.mlp_up);
                        free_ffn_temporaries = false;
                    }
                    break;
                case BlockOp::Router:
                    run_router();
                    break;
                case BlockOp::Experts:
                    run_experts();
                    break;
                case BlockOp::Combine:
                    run_combine();
                    break;
                case BlockOp::Mamba:
                    run_mamba();
                    break;
                default: break;
            }
        }
    }

    template<typename Block>
    static void backward(
        const BlockSpec& spec,
        const ::recipes::Recipe& recipe,
        ModularRunState<Block>& rs,
        typename Block::Weights& weights,
        typename Block::Gradients& grads,
        SimplifiedLayerActivations& acts,
        SimplifiedLayerGradients& d_acts,
        SimplifiedLayerQuantActivations& quant_acts,
        SimplifiedQuantGradients& quant_grads,
        int layer_idx,
        const ModelConfig& config,
        const ModelOptions& options,
        bool accumulate,
        cudaStream_t stream,
        bool allow_quant_layer,
        const BackwardHook* hook)
    {
        const int B = (int)rs.B;
        const int T = (int)rs.T;
        const int C = config.HiddenSize;
        const int D = config.IntermediateSize;
        const int M = config.mlp_up_rows();
        const int Hq = config.NumQueryHeads;
        const int Hkv = config.NumKeyValHeads;
        const int Hs = config.head_size();
        const int qkv_channels = config.qkv_channels();
        const int mamba_heads = config.MambaNumHeads;
        const int mamba_head_dim = config.MambaHeadDim;
        const int mamba_state = config.MambaSsmStateSize;
        const int mamba_groups = config.MambaNGroups > 0 ? config.MambaNGroups : 1;
        const int mamba_dim = config.mamba_dim();
        const int mamba_conv_dim = mamba_dim + 2 * mamba_groups * mamba_state;
        const int mamba_proj_size = mamba_dim + mamba_conv_dim + mamba_heads;
        const int mamba_chunk_size = config.MambaChunkSize > 0 ? config.MambaChunkSize : 2048;
        const int mamba_chunks = (T + mamba_chunk_size - 1) / mamba_chunk_size;

        const bool skip_weight_grad = rs.is_lora_only_mode();
        const bool debug_grads = std::getenv("SUROGATE_DEBUG_MODULAR_GRADS") != nullptr;
        const bool debug_matmul = std::getenv("SUROGATE_DEBUG_MODULAR_MATMUL_GRADS") != nullptr;
        const int debug_mid_layer = config.NumLayers / 2;

        auto dump_row95 = [&](const char* tag, const Tensor& t) {
            if (!t.Data) {
                fprintf(stderr, "[MOD GRADS] %s ptr=null\n", tag);
                return;
            }
            long rows = (t.Rank >= 3) ? (t.Sizes[0] * t.Sizes[1]) : t.Sizes[0];
            long hidden = (t.Rank >= 3) ? t.Sizes[2] : (t.Rank >= 2 ? t.Sizes[1] : 0);
            if (rows <= 0 || hidden <= 0) {
                fprintf(stderr, "[MOD GRADS] %s ptr=%p\n", tag, (void*)t.Data);
                return;
            }
            long row = rows > 95 ? 95 : (rows - 1);
            const long N = hidden < 64 ? hidden : 64;
            float sum_row = 0.0f;
            if (t.DType == ETensorDType::BF16) {
                std::vector<nv_bfloat16> buf(N);
                CUDA_CHECK(cudaMemcpy(buf.data(), (nv_bfloat16*)t.Data + row * hidden, N * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
                for (long i = 0; i < N; i++) sum_row += fabsf(__bfloat162float(buf[i]));
            } else if (t.DType == ETensorDType::FP32) {
                std::vector<float> buf(N);
                CUDA_CHECK(cudaMemcpy(buf.data(), (float*)t.Data + row * hidden, N * sizeof(float), cudaMemcpyDeviceToHost));
                for (long i = 0; i < N; i++) sum_row += fabsf(buf[i]);
            } else {
                fprintf(stderr, "[MOD GRADS] %s ptr=%p dtype=%d\n", tag, (void*)t.Data, static_cast<int>(t.DType));
                return;
            }
            fprintf(stderr, "[MOD GRADS] %s layer=%d row%ld_abssum=%.6f\n", tag, layer_idx, row, sum_row);
        };

        recipe_ops::BackwardQuantView qg{&quant_acts, rs.has_grad_quants() ? &quant_grads : nullptr};

        using WeightsType = std::decay_t<decltype(weights)>;
        constexpr bool kHasMoE = has_moe_weights<WeightsType>::value;
        constexpr bool kHasMLP = has_mlp_weights<WeightsType>::value;
        auto& att_grads = attention_grads(grads);
        auto has_op = [&](BlockOp op) {
            for (int i = 0; i < spec.forward_ops_count; ++i) {
                if (spec.forward_ops[i] == op) return true;
            }
            return false;
        };

        const bool has_attention = has_op(BlockOp::Attention) || has_op(BlockOp::AttnOut) || has_op(BlockOp::QKV);
        const bool has_mlp = has_op(BlockOp::MLPUp) || has_op(BlockOp::MLPDown) ||
                             has_op(BlockOp::SwiGLU) || has_op(BlockOp::MLPAct) ||
                             has_op(BlockOp::Mamba);
        const bool has_mamba = has_op(BlockOp::Mamba);
        const bool has_ln2 = has_op(BlockOp::LN2) || has_op(BlockOp::ResidualLN2);
        const bool mlp_uses_ln1 = !has_ln2;

        auto bwd_mlp_down = [&]() {
            if constexpr (!kHasMLP) {
                throw std::logic_error("BlockExecutor::backward: MLPDown op requires dense weights");
            } else {
                recipe_ops::CachedWeights dgrad_cache{};
                recipe_ops::backward_matmul(
                    recipe, rs,
                    d_acts.d_swiglu, grads.d_mlp_down_weight, nullptr,
                    d_acts.d_res_ffn, acts.swiglu, weights.mlp_down_weight,
                    B, T, D, C,
                    layer_idx, MatmulOp::MLPDown,
                    accumulate, skip_weight_grad,
                    qg.inp_swiglu(), qg.dout_d_res_ffn(),
                    nullptr, dgrad_cache, stream, allow_quant_layer);
                if (debug_matmul &&
                    (layer_idx <= 1 || layer_idx == debug_mid_layer || layer_idx == config.NumLayers - 1)) {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    dump_row95("matmul mlp_down d_out", d_acts.d_res_ffn);
                    dump_row95("matmul mlp_down d_in", d_acts.d_swiglu);
                }
            }
        };

        auto bwd_swiglu = [&]() {
            if constexpr (!kHasMLP) {
                throw std::logic_error("BlockExecutor::backward: SwiGLU op requires dense weights");
            } else {
                recipe_ops::swiglu_backward(
                    recipe,
                    d_acts.d_mlp_up,
                    d_acts.d_swiglu,
                    acts.mlp_up,
                    nullptr,
                    qg.d_mlp_up_abs_max(),
                    B, T, D,
                    stream);
            }
        };

        auto bwd_mlp_act = [&]() {
            if constexpr (!kHasMLP) {
                throw std::logic_error("BlockExecutor::backward: MLPAct op requires dense weights");
            } else {
                switch (config.activation_type) {
                    case ActivationType::SwiGLU:
                        bwd_swiglu();
                        break;
                    case ActivationType::GeGLU:
                        throw std::logic_error("BlockExecutor::backward: GeGLU activation not implemented");
                    case ActivationType::SiLU: {
                        const long N = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(M);
                        silu_backward(d_acts.d_mlp_up, acts.mlp_up, d_acts.d_swiglu, N, stream);
                        recipe_ops::record_abs_max(qg.d_mlp_up_abs_max(), d_acts.d_mlp_up, rs.DeviceProp, stream);
                        break;
                    }
                    case ActivationType::ReLU2: {
                        const long N = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(M);
                        relu2_backward(d_acts.d_mlp_up, acts.mlp_up, d_acts.d_swiglu, N, stream);
                        recipe_ops::record_abs_max(qg.d_mlp_up_abs_max(), d_acts.d_mlp_up, rs.DeviceProp, stream);
                        break;
                    }
                    case ActivationType::ReLU:
                    case ActivationType::GeLU:
                        throw std::logic_error("BlockExecutor::backward: Activation type not supported yet");
                    default:
                        throw std::logic_error("BlockExecutor::backward: Unknown activation type");
                }
            }
        };

        auto bwd_mlp_up = [&]() {
            if constexpr (!kHasMLP) {
                throw std::logic_error("BlockExecutor::backward: MLPUp op requires dense weights");
            } else {
                Tensor& mlp_in = mlp_uses_ln1 ? acts.ln1 : acts.ln2;
                Tensor& d_mlp_in = mlp_uses_ln1 ? d_acts.d_ln1 : d_acts.d_ln2;
                Tensor* q_in = mlp_uses_ln1 ? qg.inp_ln1() : qg.inp_ln2();
                recipe_ops::CachedWeights dgrad_cache{};
                recipe_ops::backward_matmul(
                    recipe, rs,
                    d_mlp_in, grads.d_mlp_up_weight, nullptr,
                    d_acts.d_mlp_up, mlp_in, weights.mlp_up_weight,
                    B, T, C, M,
                    layer_idx, MatmulOp::MLPUp,
                    accumulate, skip_weight_grad,
                    q_in, qg.dout_d_mlp_up(),
                    nullptr, dgrad_cache, stream, allow_quant_layer);
                if (debug_matmul &&
                    (layer_idx <= 1 || layer_idx == debug_mid_layer || layer_idx == config.NumLayers - 1)) {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    dump_row95("matmul mlp_up d_out", d_acts.d_mlp_up);
                    dump_row95("matmul mlp_up d_in", d_mlp_in);
                }
            }
        };

        auto bwd_mamba = [&]() {
            if constexpr (!kHasMLP) {
                throw std::logic_error("BlockExecutor::backward: Mamba op requires dense weights");
            } else {
                if (!weights.mamba.has_value() || !grads.mamba.has_value()) {
                    throw std::logic_error("BlockExecutor::backward: Mamba weights/gradients missing");
                }
                auto& mw = *weights.mamba;
                auto& mg = *grads.mamba;
                const int dev = rs.DeviceId;

                recipe_ops::CachedWeights dummy_cache{};

                // 1) Output projection backward: d_normed + d_out_proj_weight
                recipe_ops::backward_matmul(
                    recipe, rs,
                    d_acts.d_mamba_normed, mg.d_out_proj_weight,
                    mg.d_out_proj_bias.has_value() ? &mg.d_out_proj_bias.value() : nullptr,
                    d_acts.d_res_ffn, acts.mamba_normed, mw.out_proj_weight,
                    B, T, mamba_dim, C,
                    layer_idx, MatmulOp::MLPDown,
                    accumulate, skip_weight_grad,
                    nullptr, nullptr,
                    nullptr, dummy_cache, stream, /*allow_quant=*/false);

                if (!skip_weight_grad && mg.d_out_proj_bias.has_value()) {
                    const long scratch_bytes = get_bias_backward_scratch_size(mg.d_out_proj_bias->DType, C, rs.DeviceProp);
                    Tensor bias_scratch = rs.temp_alloc(ETensorDType::FP32, {scratch_bytes / (long)sizeof(float)});
                    backward_bias(mg.d_out_proj_bias.value(), d_acts.d_res_ffn,
                                  nullptr, nullptr, bias_scratch,
                                  B, T, C, rs.DeviceProp, stream);
                    rs.temp_free(bias_scratch);
                }

                // 2) Group RMSNorm backward
                Tensor d_norm_weight_fp32{ETensorDType::FP32, {mamba_dim}, nullptr, nullptr, 1, dev};
                rs.temp_acquire(d_norm_weight_fp32);
                mamba_group_rmsnorm_backward_dx(d_acts.d_mamba_gated, d_acts.d_mamba_normed,
                                                acts.mamba_gated, mw.norm_weight,
                                                acts.mamba_rstd, B, T, mamba_dim, mamba_groups, stream);
                if (!skip_weight_grad) {
                    mamba_group_rmsnorm_backward_dweight_fp32(d_norm_weight_fp32, d_acts.d_mamba_normed,
                                                              acts.mamba_gated, acts.mamba_rstd,
                                                              B, T, mamba_dim, mamba_groups, stream);
                    if (accumulate) {
                        Tensor d_norm_bf16{ETensorDType::BF16, {mamba_dim}, nullptr, nullptr, 1, dev};
                        rs.temp_acquire(d_norm_bf16);
                        convert_dtype(d_norm_bf16.get<nv_bfloat16>(), d_norm_weight_fp32.get<float>(), mamba_dim, stream);
                        vector_add_sr(mg.d_norm_weight.template get<nv_bfloat16>(),
                                      mg.d_norm_weight.template get<nv_bfloat16>(),
                                      d_norm_bf16.get<nv_bfloat16>(),
                                      1.0f, mamba_dim, 0, stream);
                        rs.temp_free(d_norm_bf16);
                    } else {
                        convert_dtype(mg.d_norm_weight.template get<nv_bfloat16>(), d_norm_weight_fp32.get<float>(), mamba_dim, stream);
                    }
                }
                rs.temp_free(d_norm_weight_fp32);

                // 3) Gate backward (in-place): d_gate + d_scan_out
                silu_mul_backward_inplace(acts.mamba_gate, acts.mamba_scan_out, d_acts.d_mamba_gated,
                                          nullptr, B * T, mamba_dim, stream);

                // 4) Transpose d_scan_out to (B, D, T)
                Tensor d_scan_out_bdt{acts.mamba_scan_out.DType, {B, mamba_dim, T}, nullptr, nullptr, 3, dev};
                rs.temp_acquire(d_scan_out_bdt);
                mamba_transpose_btd_to_bdt(d_scan_out_bdt, acts.mamba_scan_out, B, T, mamba_dim, stream);

                // 5) Selective scan backward
                Tensor A_exp{ETensorDType::FP32, {mamba_dim, mamba_state}, nullptr, nullptr, 2, dev};
                Tensor D_exp{ETensorDType::FP32, {mamba_dim}, nullptr, nullptr, 1, dev};
                Tensor dt_bias_exp{ETensorDType::FP32, {mamba_dim}, nullptr, nullptr, 1, dev};
                Tensor dA{ETensorDType::FP32, {mamba_dim, mamba_state}, nullptr, nullptr, 2, dev};
                Tensor dD_exp{ETensorDType::FP32, {mamba_dim}, nullptr, nullptr, 1, dev};
                Tensor d_dt_bias_exp{ETensorDType::FP32, {mamba_dim}, nullptr, nullptr, 1, dev};
                rs.temp_acquire(A_exp);
                rs.temp_acquire(D_exp);
                rs.temp_acquire(dt_bias_exp);
                rs.temp_acquire(dA);
                if (!skip_weight_grad) {
                    rs.temp_acquire(dD_exp);
                    rs.temp_acquire(d_dt_bias_exp);
                }

                mamba_expand_A(A_exp, mw.A_log, mamba_heads, mamba_head_dim, mamba_state, stream);
                mamba_expand_head_param(D_exp, mw.D, mamba_heads, mamba_head_dim, stream);
                mamba_expand_head_param(dt_bias_exp, mw.dt_bias, mamba_heads, mamba_head_dim, stream);

                mamba_selective_scan_backward(d_acts.d_mamba_u, d_acts.d_mamba_delta,
                                              dA, d_acts.d_mamba_B, d_acts.d_mamba_C,
                                              skip_weight_grad ? nullptr : &dD_exp,
                                              skip_weight_grad ? nullptr : &d_dt_bias_exp,
                                              acts.mamba_u, acts.mamba_delta,
                                              A_exp, acts.mamba_B, acts.mamba_C,
                                              D_exp, dt_bias_exp,
                                              d_scan_out_bdt, acts.mamba_x,
                                              B, T, mamba_dim, mamba_state, mamba_groups, mamba_chunks, stream);

                if (!skip_weight_grad) {
                    mamba_reduce_dA_log(mg.d_A_log, dA, A_exp,
                                        mamba_heads, mamba_head_dim, mamba_state, accumulate, stream);
                    mamba_reduce_head_param(mg.d_D, dD_exp, mamba_heads, mamba_head_dim, accumulate, stream);
                    mamba_reduce_head_param(mg.d_dt_bias, d_dt_bias_exp, mamba_heads, mamba_head_dim, accumulate, stream);
                }

                if (!skip_weight_grad) {
                    rs.temp_free(d_dt_bias_exp);
                    rs.temp_free(dD_exp);
                }
                rs.temp_free(dA);
                rs.temp_free(dt_bias_exp);
                rs.temp_free(D_exp);
                rs.temp_free(A_exp);
                rs.temp_free(d_scan_out_bdt);

                // 6) Pack conv1d output gradients (BF16)
                mamba_pack_conv_out(d_acts.d_mamba_conv_out, d_acts.d_mamba_u,
                                    d_acts.d_mamba_B, d_acts.d_mamba_C,
                                    B, T, mamba_dim, mamba_groups, mamba_state, stream);

                // 7) Conv1d backward: dx + d_conv_weight/bias
                Tensor d_conv_in{acts.mamba_conv_in.DType, {B, mamba_conv_dim, T}, nullptr, nullptr, 3, dev};
                rs.temp_acquire(d_conv_in);
                Tensor d_conv_w_fp32{ETensorDType::FP32, {mamba_conv_dim, config.MambaConvKernel}, nullptr, nullptr, 2, dev};
                Tensor d_conv_b_fp32{ETensorDType::FP32, {mamba_conv_dim}, nullptr, nullptr, 1, dev};
                rs.temp_acquire(d_conv_w_fp32);
                Tensor* d_conv_b_ptr = nullptr;
                if (mw.conv1d_bias.has_value()) {
                    rs.temp_acquire(d_conv_b_fp32);
                    d_conv_b_ptr = &d_conv_b_fp32;
                }

                mamba_causal_conv1d_backward(d_conv_in, d_conv_w_fp32, d_conv_b_ptr,
                                             acts.mamba_conv_in, mw.conv1d_weight,
                                             d_acts.d_mamba_conv_out,
                                             B, T, mamba_conv_dim, config.MambaConvKernel,
                                             /*silu=*/true, stream);

                if (!skip_weight_grad) {
                    if (accumulate) {
                        Tensor d_conv_w_bf16{ETensorDType::BF16, {mamba_conv_dim, config.MambaConvKernel}, nullptr, nullptr, 2, dev};
                        rs.temp_acquire(d_conv_w_bf16);
                        convert_dtype(d_conv_w_bf16.get<nv_bfloat16>(), d_conv_w_fp32.get<float>(),
                                      d_conv_w_fp32.nelem(), stream);
                        vector_add_sr(mg.d_conv1d_weight.template get<nv_bfloat16>(),
                                      mg.d_conv1d_weight.template get<nv_bfloat16>(),
                                      d_conv_w_bf16.get<nv_bfloat16>(),
                                      1.0f, d_conv_w_fp32.nelem(), 0, stream);
                        rs.temp_free(d_conv_w_bf16);
                    } else {
                        convert_dtype(mg.d_conv1d_weight.template get<nv_bfloat16>(), d_conv_w_fp32.get<float>(),
                                      d_conv_w_fp32.nelem(), stream);
                    }

                    if (mw.conv1d_bias.has_value() && mg.d_conv1d_bias.has_value() && d_conv_b_ptr) {
                        if (accumulate) {
                            Tensor d_conv_b_bf16{ETensorDType::BF16, {mamba_conv_dim}, nullptr, nullptr, 1, dev};
                            rs.temp_acquire(d_conv_b_bf16);
                            convert_dtype(d_conv_b_bf16.get<nv_bfloat16>(), d_conv_b_fp32.get<float>(),
                                          d_conv_b_fp32.nelem(), stream);
                            vector_add_sr(mg.d_conv1d_bias->template get<nv_bfloat16>(),
                                          mg.d_conv1d_bias->template get<nv_bfloat16>(),
                                          d_conv_b_bf16.get<nv_bfloat16>(),
                                          1.0f, d_conv_b_fp32.nelem(), 0, stream);
                            rs.temp_free(d_conv_b_bf16);
                        } else {
                            convert_dtype(mg.d_conv1d_bias->template get<nv_bfloat16>(), d_conv_b_fp32.get<float>(),
                                          d_conv_b_fp32.nelem(), stream);
                        }
                    }
                }

                if (d_conv_b_ptr) {
                    rs.temp_free(d_conv_b_fp32);
                }
                rs.temp_free(d_conv_w_fp32);

                // 8) Reduce d_delta -> d_dt
                Tensor d_dt{acts.mamba_gate.DType, {B, T, mamba_heads}, nullptr, nullptr, 3, dev};
                rs.temp_acquire(d_dt);
                mamba_reduce_delta_to_dt(d_dt, d_acts.d_mamba_delta,
                                         B, T, mamba_heads, mamba_head_dim, stream);

                // 9) Pack d_proj and run in_proj backward
                Tensor d_proj{acts.ln1.DType, {B, T, mamba_proj_size}, nullptr, nullptr, 3, dev};
                rs.temp_acquire(d_proj);
                mamba_pack_dproj(d_proj, acts.mamba_gate, d_conv_in, d_dt,
                                 B, T, mamba_dim, mamba_conv_dim, mamba_heads, stream);

                recipe_ops::backward_matmul(
                    recipe, rs,
                    d_acts.d_ln1, mg.d_in_proj_weight,
                    mg.d_in_proj_bias.has_value() ? &mg.d_in_proj_bias.value() : nullptr,
                    d_proj, acts.ln1, mw.in_proj_weight,
                    B, T, C, mamba_proj_size,
                    layer_idx, MatmulOp::MLPUp,
                    accumulate, skip_weight_grad,
                    nullptr, nullptr,
                    nullptr, dummy_cache, stream, /*allow_quant=*/false);

                if (!skip_weight_grad && mg.d_in_proj_bias.has_value()) {
                    const long scratch_bytes = get_bias_backward_scratch_size(mg.d_in_proj_bias->DType, mamba_proj_size, rs.DeviceProp);
                    Tensor bias_scratch = rs.temp_alloc(ETensorDType::FP32, {scratch_bytes / (long)sizeof(float)});
                    backward_bias(mg.d_in_proj_bias.value(), d_proj,
                                  nullptr, nullptr, bias_scratch,
                                  B, T, mamba_proj_size, rs.DeviceProp, stream);
                    rs.temp_free(bias_scratch);
                }

                rs.temp_free(d_proj);
                rs.temp_free(d_dt);
                rs.temp_free(d_conv_in);
            }
        };

        auto bwd_ln2_dense = [&]() {
            Tensor& ln2_d_weight = ln2_weight_grad(grads);
            static bool ln2_dbg_top = false;
            static bool ln2_dbg_mid = false;
            static bool ln2_dbg_l1 = false;
            static bool ln2_dbg_l0 = false;
            const int mid_layer = config.NumLayers / 2;
            const bool ln2_dbg_layer =
                (layer_idx == config.NumLayers - 1 && !ln2_dbg_top) ||
                (layer_idx == mid_layer && !ln2_dbg_mid) ||
                (layer_idx == 1 && !ln2_dbg_l1) ||
                (layer_idx == 0 && !ln2_dbg_l0);
            if (debug_grads && ln2_dbg_layer) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                dump_row95("ln2 pre d_y", d_acts.d_ln2);
                dump_row95("ln2 pre d_residual_next", d_acts.d_res_ffn);
            }
            rmsnorm_backward(d_acts.d_res_att, ln2_d_weight, rs.scratch().rmsnorm_scratch,
                             d_acts.d_res_ffn, d_acts.d_ln2,
                             acts.residual_att, weights.ln2.weight, acts.ln2_rstd,
                             qg.d_res_att_abs_max(),
                             B, T, C, rs.DeviceProp, stream,
                             skip_weight_grad);
            if (debug_grads && ln2_dbg_layer) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                dump_row95("ln2 post d_input", d_acts.d_res_att);
                if (layer_idx == config.NumLayers - 1) ln2_dbg_top = true;
                if (layer_idx == mid_layer) ln2_dbg_mid = true;
                if (layer_idx == 1) ln2_dbg_l1 = true;
                if (layer_idx == 0) ln2_dbg_l0 = true;
            }
        };

        auto bwd_attn_out = [&]() {
            recipe_ops::CachedWeights dgrad_cache{};
            recipe_ops::backward_matmul(
                recipe, rs,
                d_acts.d_att, att_grads.d_out_weight, nullptr,
                d_acts.d_res_att, acts.att, weights.attention.out_weight,
                B, T, Hq * Hs, C,
                layer_idx, MatmulOp::AttnOut,
                accumulate, skip_weight_grad,
                qg.inp_att(), qg.dout_d_res_att(),
                nullptr, dgrad_cache, stream, allow_quant_layer);
            if (debug_matmul &&
                (layer_idx <= 1 || layer_idx == debug_mid_layer || layer_idx == config.NumLayers - 1)) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                dump_row95("matmul attn_out d_out", d_acts.d_res_att);
                dump_row95("matmul attn_out d_in", d_acts.d_att);
            }
        };

        auto bwd_attention = [&]() {
            const bool use_qk_norm = recipe_ops::has_qk_norm(weights.attention);
            Tensor& qkv_for_attn = (use_qk_norm && acts.qkv_rope.Data != nullptr) ? acts.qkv_rope : acts.qkv;
            recipe_ops::attention_backward(d_acts.d_qkv, acts.lse, acts.att, d_acts.d_att,
                                           qkv_for_attn, rs.CuBlasWorkspace,
                                           rs.CudnnHandle, B, T, Hq, Hkv, Hs, stream);
        };

        auto bwd_rope = [&]() {
            int* pos_ids_ptr = rs.PositionIDs.template get<int>();
            auto& freq_cis = rs.non_block_activations().freq_cis;
            recipe_ops::rope_backward_dispatch(
                d_acts.d_qkv, d_acts.d_qkv, freq_cis, pos_ids_ptr,
                config, options, qg.d_qkv_abs_max(),
                B, T, Hq, Hkv, Hs, stream);
        };

        auto bwd_qk_norm = [&]() {
            const bool use_qk_norm = recipe_ops::has_qk_norm(weights.attention);
            if (use_qk_norm && acts.qkv_rope.Data == nullptr) {
                int* pos_ids_ptr = rs.PositionIDs.template get<int>();
                auto& freq_cis = rs.non_block_activations().freq_cis;
                recipe_ops::rope_backward_dispatch(
                    acts.qkv, acts.qkv, freq_cis, pos_ids_ptr,
                    config, options, nullptr,
                    B, T, Hq, Hkv, Hs, stream);
            }

            recipe_ops::qk_norm_backward(
                d_acts.d_qkv, acts.qkv, acts.q_rstd, acts.k_rstd,
                weights.attention, att_grads,
                config.RmsNormEps, B, T, qkv_channels, Hq, Hkv, Hs,
                accumulate, skip_weight_grad, stream);
        };

        auto bwd_qkv = [&]() {
            Tensor* dbias = att_grads.d_qkv_bias.has_value() ?
                            &att_grads.d_qkv_bias.value() : nullptr;
            recipe_ops::CachedWeights dgrad_cache{};
            recipe_ops::backward_matmul(
                recipe, rs,
                d_acts.d_ln1, att_grads.d_qkv_weight, dbias,
                d_acts.d_qkv, acts.ln1, weights.attention.qkv_weight,
                B, T, C, qkv_channels,
                layer_idx, MatmulOp::QKV,
                accumulate, skip_weight_grad,
                qg.inp_ln1(), qg.dout_d_qkv(),
                nullptr, dgrad_cache, stream, allow_quant_layer);
            if (debug_matmul &&
                (layer_idx <= 1 || layer_idx == debug_mid_layer || layer_idx == config.NumLayers - 1)) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                dump_row95("matmul qkv d_out", d_acts.d_qkv);
                dump_row95("matmul qkv d_in", d_acts.d_ln1);
            }
        };

        // Hooks: layer start
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeLayerBackward, nullptr);

        // In full-block recompute mode, keep large backward intermediates stack-backed (matches legacy path).
        const bool stack_large_bwd_temps = rs.large_bwd_temps_on_stack();
        bool acquired_d_qkv = false;
        const bool moe_gated = is_gated_activation(config.activation_type);

        if (spec.variant == BlockVariant::MoE) {
            if constexpr (!kHasMoE) {
                throw std::logic_error("BlockExecutor::backward: MoE variant requires MoE weights");
            } else {
                if (!config.moe_config.has_value()) {
                    throw std::logic_error("BlockExecutor::backward: MoE config missing");
                }

                const int BT = B * T;
                const auto& moe_cfg = *config.moe_config;
                if (moe_cfg.n_group > 1 || moe_cfg.topk_group > 1) {
                    throw std::logic_error("BlockExecutor::backward: grouped MoE routing is not supported yet");
                }
                const int num_experts = moe_cfg.num_experts;
                const int top_k = moe_cfg.top_k;
                const int expert_D = moe_cfg.moe_intermediate_size > 0 ? moe_cfg.moe_intermediate_size : D;
                const int total_expert_tokens = BT * top_k;
                const int dev = rs.DeviceId;

                const bool lora_only = rs.is_lora_only_mode();

                if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeMLPDownBackward, nullptr);

                // Allocate routing temporaries
                Tensor router_logits{ETensorDType::FP32, {BT, num_experts}, nullptr, nullptr, 2, dev};
                Tensor router_probs{ETensorDType::FP32, {BT, num_experts}, nullptr, nullptr, 2, dev};
                Tensor routing_weights_fp32{ETensorDType::FP32, {BT, top_k}, nullptr, nullptr, 2, dev};
                Tensor expert_indices{ETensorDType::INT32, {BT, top_k}, nullptr, nullptr, 2, dev};
                Tensor expert_counts{ETensorDType::INT32, {num_experts}, nullptr, nullptr, 1, dev};
                Tensor expert_offsets{ETensorDType::INT32, {num_experts + 1}, nullptr, nullptr, 1, dev};
                Tensor expert_positions{ETensorDType::INT32, {num_experts}, nullptr, nullptr, 1, dev};
                Tensor gather_indices{ETensorDType::INT32, {total_expert_tokens}, nullptr, nullptr, 1, dev};
                Tensor scatter_indices{ETensorDType::INT32, {total_expert_tokens}, nullptr, nullptr, 1, dev};

                rs.temp_acquire(router_logits);
                rs.temp_acquire(router_probs);
                rs.temp_acquire(routing_weights_fp32);
                rs.temp_acquire(expert_indices);
                rs.temp_acquire(expert_counts);
                rs.temp_acquire(expert_offsets);
                rs.temp_acquire(expert_positions);
                rs.temp_acquire(gather_indices);
                rs.temp_acquire(scatter_indices);

                fill_zero(expert_counts, stream);
                fill_zero(expert_positions, stream);

                // Flat ln2 view
                Tensor flat_ln2;
                flat_ln2.Data = acts.ln2.Data;
                flat_ln2.DType = acts.ln2.DType;
                flat_ln2.Sizes[0] = BT;
                flat_ln2.Sizes[1] = C;
                flat_ln2.Rank = 2;
                flat_ln2.Device = dev;

                // Router recompute (no hooks)
                matmul(
                    router_logits, weights.router.gate, flat_ln2, std::nullopt,
                    nullptr, nullptr,
                    rs.CublasLtHandle, rs.CuBlasWorkspace,
                    num_experts, BT, C, EMMTranspose::TN, false,
                    stream
                );

                if (moe_cfg.use_sigmoid) {
                    moe_sigmoid_forward(
                        router_probs.template get<float>(),
                        router_logits.template get<float>(),
                        BT * num_experts, stream
                    );
                } else {
                    moe_softmax_forward(
                        router_probs.template get<float>(),
                        router_logits.template get<float>(),
                        BT, num_experts, stream
                    );
                }

                moe_topk_forward(
                    expert_indices.template get<int>(),
                    routing_weights_fp32.template get<float>(),
                    router_probs.template get<float>(),
                    BT, num_experts, top_k, moe_cfg.norm_topk_prob, stream
                );

                if (moe_cfg.routed_scaling_factor != 1.0f) {
                    moe_scale_forward(
                        routing_weights_fp32.template get<float>(),
                        routing_weights_fp32.template get<float>(),
                        moe_cfg.routed_scaling_factor,
                        BT * top_k,
                        stream
                    );
                }

                moe_compute_expert_counts(
                    expert_counts.template get<int>(),
                    expert_indices.template get<int>(),
                    BT, top_k, num_experts, stream
                );

                moe_compute_expert_offsets(
                    expert_offsets.template get<int>(),
                    expert_counts.template get<int>(),
                    num_experts, stream
                );

                moe_build_indices(
                    gather_indices.template get<int>(),
                    scatter_indices.template get<int>(),
                    expert_indices.template get<int>(),
                    expert_offsets.template get<int>(),
                    expert_positions.template get<int>(),
                    BT, top_k, num_experts, stream
                );

                // Cache host offsets for grouped GEMM
                rs.MoeHostExpertOffsets.resize(num_experts + 1);
                CUDA_CHECK(cudaMemcpyAsync(rs.MoeHostExpertOffsets.data(), expert_offsets.template get<int>(),
                                           (num_experts + 1) * sizeof(int),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                rs.MoeHostOffsetsValid = true;

                // Expert temporaries
                const int moe_M = moe_gated ? (2 * expert_D) : expert_D;
                Tensor permuted_input{acts.ln2.DType, {total_expert_tokens, C}, nullptr, nullptr, 2, dev};
                Tensor expert_gate_up{acts.ln2.DType, {total_expert_tokens, moe_M}, nullptr, nullptr, 2, dev};
                Tensor expert_outputs{acts.ln2.DType, {total_expert_tokens, C}, nullptr, nullptr, 2, dev};
                Tensor d_expert_outputs{acts.ln2.DType, {total_expert_tokens, C}, nullptr, nullptr, 2, dev};
                Tensor d_expert_gate_up{acts.ln2.DType, {total_expert_tokens, moe_M}, nullptr, nullptr, 2, dev};
                Tensor d_permuted_input{acts.ln2.DType, {total_expert_tokens, C}, nullptr, nullptr, 2, dev};

                rs.temp_acquire(permuted_input);
                rs.temp_acquire(expert_gate_up);
                rs.temp_acquire(expert_outputs);
                rs.temp_acquire(d_expert_outputs);
                rs.temp_acquire(d_expert_gate_up);
                rs.temp_acquire(d_permuted_input);

                fill_zero(d_permuted_input, stream);

                // Recompute expert forward for backward
                if (acts.ln2.DType == ETensorDType::BF16) {
                    moe_permute_tokens(
                        permuted_input.template get<nv_bfloat16>(),
                        flat_ln2.template get<nv_bfloat16>(),
                        gather_indices.template get<int>(),
                        total_expert_tokens, BT, C, top_k, stream
                    );
                } else {
                    moe_permute_tokens(
                        permuted_input.template get<float>(),
                        flat_ln2.template get<float>(),
                        gather_indices.template get<int>(),
                        total_expert_tokens, BT, C, top_k, stream
                    );
                }

                {
                    const int* host_offsets = rs.MoeHostOffsetsValid ? rs.MoeHostExpertOffsets.data() : nullptr;
                    Tensor expert_act{acts.ln2.DType, {total_expert_tokens, expert_D}, nullptr, nullptr, 2, dev};
                    rs.temp_acquire(expert_act);

                    if (moe_gated) {
                        if (acts.ln2.DType == ETensorDType::BF16) {
                            moe_grouped_gemm_gate_up(
                                expert_gate_up.template get<nv_bfloat16>(),
                                permuted_input.template get<nv_bfloat16>(),
                                weights.experts.gate_up_proj.template get<nv_bfloat16>(),
                                expert_offsets.template get<int>(),
                                num_experts, C, expert_D,
                                rs.CublasHandle, stream, host_offsets
                            );
                        } else {
                            moe_grouped_gemm_gate_up(
                                expert_gate_up.template get<float>(),
                                permuted_input.template get<float>(),
                                weights.experts.gate_up_proj.template get<float>(),
                                expert_offsets.template get<int>(),
                                num_experts, C, expert_D,
                                rs.CublasHandle, stream, host_offsets
                            );
                        }

                        swiglu_forward(expert_act, expert_gate_up, nullptr, 1, total_expert_tokens, expert_D, stream);
                    } else {
                        if (acts.ln2.DType == ETensorDType::BF16) {
                            moe_grouped_gemm(
                                expert_gate_up.template get<nv_bfloat16>(),
                                permuted_input.template get<nv_bfloat16>(),
                                weights.experts.gate_up_proj.template get<nv_bfloat16>(),
                                expert_offsets.template get<int>(),
                                num_experts,
                                /*M=*/expert_D,
                                /*K=*/C,
                                rs.CublasHandle,
                                stream,
                                host_offsets,
                                /*alpha=*/1.0f,
                                /*beta=*/0.0f,
                                EMMTranspose::TN
                            );
                        } else {
                            moe_grouped_gemm(
                                expert_gate_up.template get<float>(),
                                permuted_input.template get<float>(),
                                weights.experts.gate_up_proj.template get<float>(),
                                expert_offsets.template get<int>(),
                                num_experts,
                                /*M=*/expert_D,
                                /*K=*/C,
                                rs.CublasHandle,
                                stream,
                                host_offsets,
                                /*alpha=*/1.0f,
                                /*beta=*/0.0f,
                                EMMTranspose::TN
                            );
                        }

                        const long N = static_cast<long>(total_expert_tokens) * static_cast<long>(expert_D);
                        switch (config.activation_type) {
                            case ActivationType::SiLU:
                                silu_forward(expert_act, expert_gate_up, N, stream);
                                break;
                            case ActivationType::ReLU2:
                                relu2_forward(expert_act, expert_gate_up, N, stream);
                                break;
                            default:
                                throw std::logic_error("BlockExecutor::backward: unsupported MoE activation");
                        }
                    }

                    if (acts.ln2.DType == ETensorDType::BF16) {
                        moe_grouped_gemm_down(
                            expert_outputs.template get<nv_bfloat16>(),
                            expert_act.template get<nv_bfloat16>(),
                            weights.experts.down_proj.template get<nv_bfloat16>(),
                            expert_offsets.template get<int>(),
                            num_experts, C, expert_D,
                            rs.CublasHandle, stream, host_offsets
                        );
                    } else {
                        moe_grouped_gemm_down(
                            expert_outputs.template get<float>(),
                            expert_act.template get<float>(),
                            weights.experts.down_proj.template get<float>(),
                            expert_offsets.template get<int>(),
                            num_experts, C, expert_D,
                            rs.CublasHandle, stream, host_offsets
                        );
                    }
                    rs.temp_free(expert_act);
                }

                // Combine backward
                Tensor d_routing_weights_fp32{ETensorDType::FP32, {BT, top_k}, nullptr, nullptr, 2, dev};
                rs.temp_acquire(d_routing_weights_fp32);

                if (d_acts.d_res_ffn.DType == ETensorDType::BF16) {
                    Tensor d_routing_weights{ETensorDType::BF16, {BT, top_k}, nullptr, nullptr, 2, dev};
                    Tensor routing_weights_bf16{ETensorDType::BF16, {BT, top_k}, nullptr, nullptr, 2, dev};
                    rs.temp_acquire(d_routing_weights);
                    rs.temp_acquire(routing_weights_bf16);

                    convert_dtype(routing_weights_bf16.template get<nv_bfloat16>(),
                                  routing_weights_fp32.template get<float>(),
                                  BT * top_k, stream);

                    Tensor d_res_ffn_bf16 = d_acts.d_res_ffn;
                    d_res_ffn_bf16.Sizes[0] = BT;
                    d_res_ffn_bf16.Sizes[1] = C;
                    d_res_ffn_bf16.Rank = 2;

                    moe_combine_backward(
                        d_expert_outputs.template get<nv_bfloat16>(),
                        d_routing_weights.template get<nv_bfloat16>(),
                        d_res_ffn_bf16.template get<nv_bfloat16>(),
                        expert_outputs.template get<nv_bfloat16>(),
                        routing_weights_bf16.template get<nv_bfloat16>(),
                        scatter_indices.template get<int>(),
                        BT, total_expert_tokens, C, top_k, stream
                    );

                    convert_dtype(
                        d_routing_weights_fp32.template get<float>(),
                        d_routing_weights.template get<nv_bfloat16>(),
                        BT * top_k,
                        stream
                    );

                    rs.temp_free(routing_weights_bf16);
                    rs.temp_free(d_routing_weights);
                } else {
                    Tensor d_res_ffn_fp32 = d_acts.d_res_ffn;
                    d_res_ffn_fp32.Sizes[0] = BT;
                    d_res_ffn_fp32.Sizes[1] = C;
                    d_res_ffn_fp32.Rank = 2;

                    moe_combine_backward(
                        d_expert_outputs.template get<float>(),
                        d_routing_weights_fp32.template get<float>(),
                        d_res_ffn_fp32.template get<float>(),
                        expert_outputs.template get<float>(),
                        routing_weights_fp32.template get<float>(),
                        scatter_indices.template get<int>(),
                        BT, total_expert_tokens, C, top_k, stream
                    );
                }

                if (moe_cfg.routed_scaling_factor != 1.0f) {
                    moe_scale_forward(
                        d_routing_weights_fp32.template get<float>(),
                        d_routing_weights_fp32.template get<float>(),
                        moe_cfg.routed_scaling_factor,
                        BT * top_k,
                        stream
                    );
                }

                if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterMLPDownBackward, nullptr);

                bool hook_handled = false;
                if (hook) {
                    MoEGroupedContext moe_ctx;
                    moe_ctx.expert_offsets = &expert_offsets;
                    moe_ctx.permuted_input = &permuted_input;
                    moe_ctx.expert_gate_up = &expert_gate_up;
                    moe_ctx.expert_outputs = &expert_outputs;
                    moe_ctx.d_expert_outputs = &d_expert_outputs;
                    moe_ctx.d_expert_gate_up = &d_expert_gate_up;
                    moe_ctx.d_permuted_input = &d_permuted_input;
                    moe_ctx.host_offsets = rs.MoeHostOffsetsValid ? rs.MoeHostExpertOffsets.data() : nullptr;
                    moe_ctx.num_experts = num_experts;
                    moe_ctx.top_k = top_k;
                    moe_ctx.total_tokens = total_expert_tokens;
                    (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::MoEExpertGroupManual, &moe_ctx);
                    hook_handled = moe_ctx.handled;
                }

                if (!hook_handled) {
                    const int* host_offsets = rs.MoeHostOffsetsValid ? rs.MoeHostExpertOffsets.data() : nullptr;

                    if (!lora_only) {
                        if constexpr (requires { grads.experts.d_down_proj; }) {
                            if (grads.experts.d_down_proj.Data && grads.experts.d_down_proj.nelem() > 0) {
                                Tensor expert_swiglu{acts.ln2.DType, {total_expert_tokens, expert_D}, nullptr, nullptr, 2, dev};
                                rs.temp_acquire(expert_swiglu);
                                swiglu_forward(expert_swiglu, expert_gate_up, nullptr, 1, total_expert_tokens, expert_D, stream);

                                const float beta = accumulate ? 1.0f : 0.0f;
                                if (acts.ln2.DType == ETensorDType::BF16) {
                                    moe_grouped_gemm_weight_grad(
                                        grads.experts.d_down_proj.template get<nv_bfloat16>(),
                                        d_expert_outputs.template get<nv_bfloat16>(),
                                        expert_swiglu.template get<nv_bfloat16>(),
                                        expert_offsets.template get<int>(),
                                        num_experts,
                                        C,
                                        expert_D,
                                        rs.CublasHandle,
                                        stream,
                                        host_offsets,
                                        1.0f,
                                        beta
                                    );
                                } else {
                                    moe_grouped_gemm_weight_grad(
                                        grads.experts.d_down_proj.template get<float>(),
                                        d_expert_outputs.template get<float>(),
                                        expert_swiglu.template get<float>(),
                                        expert_offsets.template get<int>(),
                                        num_experts,
                                        C,
                                        expert_D,
                                        rs.CublasHandle,
                                        stream,
                                        host_offsets,
                                        1.0f,
                                        beta
                                    );
                                }

                                rs.temp_free(expert_swiglu);
                            }
                        }
                    }

                    Tensor d_expert_act{acts.ln2.DType, {total_expert_tokens, expert_D}, nullptr, nullptr, 2, dev};
                    rs.temp_acquire(d_expert_act);

                    if (acts.ln2.DType == ETensorDType::BF16) {
                        moe_grouped_gemm_down_backward(
                            d_expert_act.template get<nv_bfloat16>(),
                            d_expert_outputs.template get<nv_bfloat16>(),
                            weights.experts.down_proj.template get<nv_bfloat16>(),
                            expert_offsets.template get<int>(),
                            num_experts, C, expert_D,
                            rs.CublasHandle, stream, host_offsets
                        );
                    } else {
                        moe_grouped_gemm_down_backward(
                            d_expert_act.template get<float>(),
                            d_expert_outputs.template get<float>(),
                            weights.experts.down_proj.template get<float>(),
                            expert_offsets.template get<int>(),
                            num_experts, C, expert_D,
                            rs.CublasHandle, stream, host_offsets
                        );
                    }

                    if (moe_gated) {
                        Tensor expert_up{acts.ln2.DType, {total_expert_tokens, expert_D}, nullptr, nullptr, 2, dev};
                        Tensor expert_gate{acts.ln2.DType, {total_expert_tokens, expert_D}, nullptr, nullptr, 2, dev};
                        rs.temp_acquire(expert_up);
                        rs.temp_acquire(expert_gate);

                        split_gate_up(expert_gate_up, expert_up, expert_gate, total_expert_tokens, expert_D, stream);

                        Tensor expert_h;
                        expert_h.Data = d_expert_gate_up.Data;
                        expert_h.DType = acts.ln2.DType;
                        expert_h.Sizes = {total_expert_tokens, expert_D};
                        silu_mul_backward_inplace(expert_gate, expert_up, d_expert_act, &expert_h, total_expert_tokens, expert_D, stream);

                        concat_d_gate_up(expert_up, expert_gate, d_expert_gate_up, total_expert_tokens, expert_D, stream);

                        rs.temp_free(expert_gate);
                        rs.temp_free(expert_up);
                    } else {
                        const long N = static_cast<long>(total_expert_tokens) * static_cast<long>(expert_D);
                        switch (config.activation_type) {
                            case ActivationType::SiLU:
                                silu_backward(d_expert_gate_up, expert_gate_up, d_expert_act, N, stream);
                                break;
                            case ActivationType::ReLU2:
                                relu2_backward(d_expert_gate_up, expert_gate_up, d_expert_act, N, stream);
                                break;
                            default:
                                throw std::logic_error("BlockExecutor::backward: unsupported MoE activation");
                        }
                    }

                    rs.temp_free(d_expert_act);

                    if (!lora_only) {
                        if constexpr (requires { grads.experts.d_gate_up_proj; }) {
                            if (grads.experts.d_gate_up_proj.Data && grads.experts.d_gate_up_proj.nelem() > 0) {
                                const float beta = accumulate ? 1.0f : 0.0f;
                                const int gate_rows = moe_gated ? (2 * expert_D) : expert_D;
                                if (acts.ln2.DType == ETensorDType::BF16) {
                                    moe_grouped_gemm_weight_grad(
                                        grads.experts.d_gate_up_proj.template get<nv_bfloat16>(),
                                        d_expert_gate_up.template get<nv_bfloat16>(),
                                        permuted_input.template get<nv_bfloat16>(),
                                        expert_offsets.template get<int>(),
                                        num_experts,
                                        gate_rows,
                                        C,
                                        rs.CublasHandle,
                                        stream,
                                        host_offsets,
                                        1.0f,
                                        beta
                                    );
                                } else {
                                    moe_grouped_gemm_weight_grad(
                                        grads.experts.d_gate_up_proj.template get<float>(),
                                        d_expert_gate_up.template get<float>(),
                                        permuted_input.template get<float>(),
                                        expert_offsets.template get<int>(),
                                        num_experts,
                                        gate_rows,
                                        C,
                                        rs.CublasHandle,
                                        stream,
                                        host_offsets,
                                        1.0f,
                                        beta
                                    );
                                }
                            }
                        }
                    }

                    if (acts.ln2.DType == ETensorDType::BF16) {
                        if (moe_gated) {
                            moe_grouped_gemm_gate_up_backward(
                                d_permuted_input.template get<nv_bfloat16>(),
                                d_expert_gate_up.template get<nv_bfloat16>(),
                                weights.experts.gate_up_proj.template get<nv_bfloat16>(),
                                expert_offsets.template get<int>(),
                                num_experts, C, expert_D,
                                rs.CublasHandle, stream, host_offsets
                            );
                        } else {
                            moe_grouped_gemm_up_backward(
                                d_permuted_input.template get<nv_bfloat16>(),
                                d_expert_gate_up.template get<nv_bfloat16>(),
                                weights.experts.gate_up_proj.template get<nv_bfloat16>(),
                                expert_offsets.template get<int>(),
                                num_experts, C, expert_D,
                                rs.CublasHandle, stream, host_offsets
                            );
                        }
                    } else {
                        if (moe_gated) {
                            moe_grouped_gemm_gate_up_backward(
                                d_permuted_input.template get<float>(),
                                d_expert_gate_up.template get<float>(),
                                weights.experts.gate_up_proj.template get<float>(),
                                expert_offsets.template get<int>(),
                                num_experts, C, expert_D,
                                rs.CublasHandle, stream, host_offsets
                            );
                        } else {
                            moe_grouped_gemm_up_backward(
                                d_permuted_input.template get<float>(),
                                d_expert_gate_up.template get<float>(),
                                weights.experts.gate_up_proj.template get<float>(),
                                expert_offsets.template get<int>(),
                                num_experts, C, expert_D,
                                rs.CublasHandle, stream, host_offsets
                            );
                        }
                    }
                }

                if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterMLPUpBackward, nullptr);

                // Permute backward: scatter d_permuted_input to d_ln2
                fill_zero(d_acts.d_ln2, stream);
                if (d_acts.d_ln2.DType == ETensorDType::BF16) {
                    Tensor d_ln2_flat = d_acts.d_ln2;
                    d_ln2_flat.Sizes[0] = BT;
                    d_ln2_flat.Sizes[1] = C;
                    d_ln2_flat.Rank = 2;
                    moe_permute_backward(
                        d_ln2_flat.template get<nv_bfloat16>(),
                        d_permuted_input.template get<nv_bfloat16>(),
                        gather_indices.template get<int>(),
                        total_expert_tokens, BT, C, top_k, stream
                    );
                } else {
                    Tensor d_ln2_flat = d_acts.d_ln2;
                    d_ln2_flat.Sizes[0] = BT;
                    d_ln2_flat.Sizes[1] = C;
                    d_ln2_flat.Rank = 2;
                    moe_permute_backward(
                        d_ln2_flat.template get<float>(),
                        d_permuted_input.template get<float>(),
                        gather_indices.template get<int>(),
                        total_expert_tokens, BT, C, top_k, stream
                    );
                }

                // Shared expert backward (adds into d_ln2)
                if (moe_cfg.use_shared_expert && weights.shared_expert.has_value()) {
                    const int shared_D = (moe_cfg.shared_expert_size > 0)
                        ? moe_cfg.shared_expert_size
                        : expert_D;

                    Tensor flat_ln2;
                    flat_ln2.Data = acts.ln2.Data;
                    flat_ln2.DType = acts.ln2.DType;
                    flat_ln2.Sizes[0] = BT;
                    flat_ln2.Sizes[1] = C;
                    flat_ln2.Rank = 2;
                    flat_ln2.Device = dev;

                    Tensor d_res_ffn_flat = d_acts.d_res_ffn;
                    d_res_ffn_flat.Sizes[0] = BT;
                    d_res_ffn_flat.Sizes[1] = C;
                    d_res_ffn_flat.Rank = 2;

                    Tensor shared_act{acts.ln2.DType, {BT, shared_D}, nullptr, nullptr, 2, dev};
                    Tensor shared_up{acts.ln2.DType, {BT, shared_D}, nullptr, nullptr, 2, dev};
                    rs.temp_acquire(shared_act);
                    rs.temp_acquire(shared_up);

                    Tensor shared_gate_up;
                    Tensor shared_gate;
                    if (moe_gated) {
                        shared_gate_up = Tensor{acts.ln2.DType, {BT, 2 * shared_D}, nullptr, nullptr, 2, dev};
                        shared_gate = Tensor{acts.ln2.DType, {BT, shared_D}, nullptr, nullptr, 2, dev};
                        rs.temp_acquire(shared_gate_up);
                        rs.temp_acquire(shared_gate);

                        matmul(shared_gate_up, weights.shared_expert->gate_proj, flat_ln2, std::nullopt,
                               nullptr, nullptr,
                               rs.CublasLtHandle, rs.CuBlasWorkspace,
                               2 * shared_D, BT, C, EMMTranspose::TN, false,
                               stream);

                        split_gate_up(shared_gate_up, shared_up, shared_gate, BT, shared_D, stream);
                        silu_mul_forward(shared_act, shared_gate, shared_up, BT, shared_D, stream);
                    } else {
                        matmul(shared_up, weights.shared_expert->gate_proj, flat_ln2, std::nullopt,
                               nullptr, nullptr,
                               rs.CublasLtHandle, rs.CuBlasWorkspace,
                               shared_D, BT, C, EMMTranspose::TN, false,
                               stream);
                        const long N = static_cast<long>(BT) * static_cast<long>(shared_D);
                        switch (config.activation_type) {
                            case ActivationType::SiLU:
                                silu_forward(shared_act, shared_up, N, stream);
                                break;
                            case ActivationType::ReLU2:
                                relu2_forward(shared_act, shared_up, N, stream);
                                break;
                            default:
                                throw std::logic_error("BlockExecutor::backward: unsupported shared expert activation");
                        }
                    }

                    Tensor d_shared_act{acts.ln2.DType, {BT, shared_D}, nullptr, nullptr, 2, dev};
                    rs.temp_acquire(d_shared_act);
                    matmul(d_shared_act, weights.shared_expert->down_proj, d_res_ffn_flat, std::nullopt,
                           nullptr, nullptr,
                           rs.CublasLtHandle, rs.CuBlasWorkspace,
                           shared_D, BT, C, EMMTranspose::NN, false,
                           stream);

                    if (!lora_only && grads.shared_expert.has_value()) {
                        matmul(grads.shared_expert->d_down_proj, d_res_ffn_flat, shared_act, std::nullopt,
                               nullptr, nullptr,
                               rs.CublasLtHandle, rs.CuBlasWorkspace,
                               C, shared_D, BT, EMMTranspose::TN, accumulate,
                               stream);
                    }

                    Tensor d_ln2_shared{acts.ln2.DType, {BT, C}, nullptr, nullptr, 2, dev};
                    rs.temp_acquire(d_ln2_shared);

                    if (moe_gated) {
                        Tensor d_shared_gate_up{acts.ln2.DType, {BT, 2 * shared_D}, nullptr, nullptr, 2, dev};
                        rs.temp_acquire(d_shared_gate_up);

                        Tensor shared_h;
                        shared_h.Data = d_shared_gate_up.Data;
                        shared_h.DType = acts.ln2.DType;
                        shared_h.Sizes = {BT, shared_D};
                        silu_mul_backward_inplace(shared_gate, shared_up, d_shared_act, &shared_h, BT, shared_D, stream);

                        concat_d_gate_up(shared_up, shared_gate, d_shared_gate_up, BT, shared_D, stream);

                        if (!lora_only && grads.shared_expert.has_value()) {
                            matmul(grads.shared_expert->d_gate_proj, d_shared_gate_up, flat_ln2, std::nullopt,
                                   nullptr, nullptr,
                                   rs.CublasLtHandle, rs.CuBlasWorkspace,
                                   2 * shared_D, C, BT, EMMTranspose::TN, accumulate,
                                   stream);
                        }

                        matmul(d_ln2_shared, weights.shared_expert->gate_proj, d_shared_gate_up, std::nullopt,
                               nullptr, nullptr,
                               rs.CublasLtHandle, rs.CuBlasWorkspace,
                               C, BT, 2 * shared_D, EMMTranspose::NN, false,
                               stream);

                        rs.temp_free(d_shared_gate_up);
                        rs.temp_free(shared_gate);
                        rs.temp_free(shared_gate_up);
                    } else {
                        const long N = static_cast<long>(BT) * static_cast<long>(shared_D);
                        switch (config.activation_type) {
                            case ActivationType::SiLU:
                                silu_backward(d_shared_act, shared_up, d_shared_act, N, stream);
                                break;
                            case ActivationType::ReLU2:
                                relu2_backward(d_shared_act, shared_up, d_shared_act, N, stream);
                                break;
                            default:
                                throw std::logic_error("BlockExecutor::backward: unsupported shared expert activation");
                        }

                        if (!lora_only && grads.shared_expert.has_value()) {
                            matmul(grads.shared_expert->d_gate_proj, d_shared_act, flat_ln2, std::nullopt,
                                   nullptr, nullptr,
                                   rs.CublasLtHandle, rs.CuBlasWorkspace,
                                   shared_D, C, BT, EMMTranspose::TN, accumulate,
                                   stream);
                        }

                        matmul(d_ln2_shared, weights.shared_expert->gate_proj, d_shared_act, std::nullopt,
                               nullptr, nullptr,
                               rs.CublasLtHandle, rs.CuBlasWorkspace,
                               C, BT, shared_D, EMMTranspose::NN, false,
                               stream);
                    }

                    const long N = static_cast<long>(BT) * static_cast<long>(C);
                    vector_add_sr(d_acts.d_ln2, d_acts.d_ln2, d_ln2_shared, 1.0f, N, /*seed=*/0, stream);

                    rs.temp_free(d_ln2_shared);
                    rs.temp_free(d_shared_act);
                    rs.temp_free(shared_up);
                    rs.temp_free(shared_act);
                }

                // Router backward
                Tensor d_probs{ETensorDType::FP32, {BT, num_experts}, nullptr, nullptr, 2, dev};
                Tensor d_logits{ETensorDType::FP32, {BT, num_experts}, nullptr, nullptr, 2, dev};
                rs.temp_acquire(d_probs);
                rs.temp_acquire(d_logits);
                fill_zero(d_probs, stream);

                moe_topk_backward(
                    d_probs.template get<float>(),
                    d_routing_weights_fp32.template get<float>(),
                    router_probs.template get<float>(),
                    expert_indices.template get<int>(),
                    BT, num_experts, top_k,
                    moe_cfg.norm_topk_prob,
                    stream
                );

                if (moe_cfg.use_sigmoid) {
                    moe_sigmoid_backward(
                        d_logits.template get<float>(),
                        d_probs.template get<float>(),
                        router_probs.template get<float>(),
                        BT * num_experts,
                        stream
                    );
                } else {
                    moe_softmax_backward(
                        d_logits.template get<float>(),
                        d_probs.template get<float>(),
                        router_probs.template get<float>(),
                        BT, num_experts,
                        stream
                    );
                }

                Tensor d_logits_bf16{ETensorDType::BF16, {BT, num_experts}, nullptr, nullptr, 2, dev};
                rs.temp_acquire(d_logits_bf16);
                convert_dtype(
                    d_logits_bf16.template get<nv_bfloat16>(),
                    d_logits.template get<float>(),
                    BT * num_experts,
                    stream
                );

                Tensor d_ln2_flat = d_acts.d_ln2;
                d_ln2_flat.Sizes[0] = BT;
                d_ln2_flat.Sizes[1] = C;
                d_ln2_flat.Rank = 2;
                matmul(
                    d_ln2_flat,
                    weights.router.gate,
                    d_logits_bf16,
                    std::nullopt,
                    nullptr, nullptr,
                    rs.CublasLtHandle, rs.CuBlasWorkspace,
                    C, BT, num_experts,
                    EMMTranspose::NN,
                    true,
                    stream
                );

                if (!lora_only || rs.is_train_router()) {
                    if constexpr (requires { grads.router.d_gate; }) {
                        if (grads.router.d_gate.Data && grads.router.d_gate.nelem() > 0) {
                            matmul(
                                grads.router.d_gate,
                                flat_ln2,
                                d_logits_bf16,
                                std::nullopt,
                                nullptr, nullptr,
                                rs.CublasLtHandle, rs.CuBlasWorkspace,
                                C, num_experts, BT,
                                EMMTranspose::NT,
                                accumulate,
                                stream
                            );
                        }
                    }
                }

                if (hook) {
                    MoERouterBackwardContext router_bwd_ctx;
                    router_bwd_ctx.d_logits = &d_logits;
                    router_bwd_ctx.input = &flat_ln2;
                    router_bwd_ctx.num_experts = num_experts;
                    router_bwd_ctx.hidden_size = C;
                    router_bwd_ctx.BT = BT;
                    (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterRouterBackward, &router_bwd_ctx);
                }

                rs.temp_free(d_logits_bf16);
                rs.temp_free(d_logits);
                rs.temp_free(d_probs);

                // Free MoE temporaries
                rs.temp_free(d_routing_weights_fp32);
                rs.temp_free(d_permuted_input);
                rs.temp_free(d_expert_gate_up);
                rs.temp_free(d_expert_outputs);
                rs.temp_free(expert_outputs);
                rs.temp_free(expert_gate_up);
                rs.temp_free(permuted_input);
                rs.temp_free(scatter_indices);
                rs.temp_free(gather_indices);
                rs.temp_free(expert_positions);
                rs.temp_free(expert_offsets);
                rs.temp_free(expert_counts);
                rs.temp_free(expert_indices);
                rs.temp_free(routing_weights_fp32);
                rs.temp_free(router_probs);
                rs.temp_free(router_logits);

                // LN2 backward
                Tensor& ln2_d_weight = ln2_weight_grad(grads);
                rmsnorm_backward(d_acts.d_res_att, ln2_d_weight, rs.scratch().rmsnorm_scratch,
                                 d_acts.d_res_ffn, d_acts.d_ln2,
                                 acts.residual_att, weights.ln2.weight, acts.ln2_rstd,
                                 nullptr,
                                 B, T, C, rs.DeviceProp, stream,
                                 lora_only);

                if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeAttnOutBackward, nullptr);
                bwd_attn_out();
                if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterAttnOutBackward, nullptr);
                if (stack_large_bwd_temps && d_acts.d_qkv.Data == nullptr) {
                    rs.temp_acquire(d_acts.d_qkv);
                    acquired_d_qkv = true;
                }
                bwd_attention();

                bwd_rope();
                bwd_qk_norm();
                if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeQKVBackward, nullptr);
                bwd_qkv();
                if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterQKVBackward, nullptr);

                if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterLayerBackward, nullptr);

                if (acquired_d_qkv) {
                    rs.temp_free(d_acts.d_qkv);
                }
            }
            return;
        }

        // -------------------- Mamba backward --------------------
        if (has_mamba) {
            bwd_mamba();
            if (has_attention) {
                throw std::logic_error("BlockExecutor::backward: Mamba blocks cannot include attention yet");
            }
            const long N = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(C);
            vector_add_sr(d_acts.d_res_att, d_acts.d_res_ffn, d_acts.d_res_ffn, 0.0f, N, /*seed=*/0, stream);
            if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterLayerBackward, nullptr);
            return;
        }

        // -------------------- MLP backward --------------------
        Tensor saved_d_mlp_up{};
        bool restore_mlp_up = false;
        if (has_mlp) {
            if constexpr (kHasMLP) {
                if (stack_large_bwd_temps) {
                    if (d_acts.d_swiglu.Data == nullptr) {
                        rs.temp_acquire(d_acts.d_swiglu);
                    }
                    // Reuse the (recomputed) mlp_up buffer in-place for d_mlp_up.
                    saved_d_mlp_up = d_acts.d_mlp_up;
                    d_acts.d_mlp_up = acts.mlp_up;
                    restore_mlp_up = true;
                }
            }

            if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeMLPDownBackward, nullptr);
            bwd_mlp_down();
            if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterMLPDownBackward, nullptr);

            bwd_mlp_act();
            if constexpr (kHasMLP) {
                if (stack_large_bwd_temps) {
                    rs.temp_free(d_acts.d_swiglu);
                    // We no longer need activation output after backward.
                    if (rs.ffn_temps_on_stack()) {
                        rs.temp_free(acts.swiglu);
                    }
                }
            }
            if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeMLPUpBackward, nullptr);
            bwd_mlp_up();
            if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterMLPUpBackward, nullptr);
            if constexpr (kHasMLP) {
                if (restore_mlp_up) {
                    d_acts.d_mlp_up = saved_d_mlp_up;
                    if (rs.ffn_temps_on_stack()) {
                        rs.temp_free(acts.mlp_up);
                    }
                }
            }
        }

        if (!has_attention) {
            const long N = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(C);
            vector_add_sr(d_acts.d_res_att, d_acts.d_res_ffn, d_acts.d_res_ffn, 0.0f, N, /*seed=*/0, stream);
            if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterLayerBackward, nullptr);
            return;
        }

        if (spec.variant == BlockVariant::Parallel && has_ln2) {
            const long N = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(C);
            // Preserve residual gradient for attention path.
            vector_add_sr(d_acts.d_res_att, d_acts.d_res_ffn, d_acts.d_res_ffn, 0.0f, N, /*seed=*/0, stream);

            // LN2 backward on residual (standalone)
            fill_zero(d_acts.d_res_ffn, stream);
            Tensor& residual = (layer_idx == 0)
                ? rs.non_block_activations().encoded
                : rs.get_residual(layer_idx - 1, stream);
            Tensor& ln2_input = spec.ln2_on_residual_att ? acts.residual_att : residual;
            Tensor& ln2_d_weight = ln2_weight_grad(grads);
            rmsnorm_backward(d_acts.d_res_ffn, ln2_d_weight, rs.scratch().rmsnorm_scratch,
                             d_acts.d_res_ffn, d_acts.d_ln2,
                             ln2_input, weights.ln2.weight, acts.ln2_rstd,
                             nullptr,
                             B, T, C, rs.DeviceProp, stream,
                             skip_weight_grad);

            if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeAttnOutBackward, nullptr);
            bwd_attn_out();
            if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterAttnOutBackward, nullptr);
            if (stack_large_bwd_temps && d_acts.d_qkv.Data == nullptr) {
                rs.temp_acquire(d_acts.d_qkv);
                acquired_d_qkv = true;
            }
            bwd_attention();

            // Merge MLP residual grad into d_res_att for LN1 backward.
            vector_add_sr(d_acts.d_res_att, d_acts.d_res_att, d_acts.d_res_ffn, 1.0f, N, /*seed=*/0, stream);
        } else if (has_ln2) {
            bwd_ln2_dense();
            if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeAttnOutBackward, nullptr);
            bwd_attn_out();
            if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterAttnOutBackward, nullptr);
            if (stack_large_bwd_temps && d_acts.d_qkv.Data == nullptr) {
                rs.temp_acquire(d_acts.d_qkv);
                acquired_d_qkv = true;
            }
            bwd_attention();
        } else {
            // Attention-only block: d_res_att is directly the upstream gradient.
            const long N = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(C);
            vector_add_sr(d_acts.d_res_att, d_acts.d_res_ffn, d_acts.d_res_ffn, 0.0f, N, /*seed=*/0, stream);
            if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeAttnOutBackward, nullptr);
            bwd_attn_out();
            if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterAttnOutBackward, nullptr);
            if (stack_large_bwd_temps && d_acts.d_qkv.Data == nullptr) {
                rs.temp_acquire(d_acts.d_qkv);
                acquired_d_qkv = true;
            }
            bwd_attention();
        }

        // RoPE + QK norm + QKV
        bwd_rope();
        bwd_qk_norm();
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeQKVBackward, nullptr);
        bwd_qkv();
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterQKVBackward, nullptr);

        if (acquired_d_qkv) {
            rs.temp_free(d_acts.d_qkv);
        }

        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterLayerBackward, nullptr);
    }

    template<typename Block>
    static void recompute(
        const BlockSpec& spec,
        const ::recipes::Recipe& recipe,
        ModularRunState<Block>& rs,
        typename Block::Weights& weights,
        SimplifiedLayerActivations& acts,
        SimplifiedLayerQuantActivations& quant_acts,
        Tensor& residual,
        int layer_idx,
        const ModelConfig& config,
        const ModelOptions& options,
        cudaStream_t stream,
        FP8ForwardQuantActivations* fp8_fwd_quants,
        FP4ForwardQuantActivations* fp4_fwd_quants,
        ModularWeightManager<Block>* weight_manager,
        bool allow_quant_layer)
    {
        if (!options.recompute_rmsnorm &&
            !options.recompute_qkv &&
            !options.recompute_attention &&
            !options.recompute_ffn &&
            !options.recompute_swiglu &&
            !options.recompute_block) {
            return;
        }

        const int B = (int)rs.B;
        const int T = (int)rs.T;
        const int C = config.HiddenSize;
        const int D = config.IntermediateSize;
        const int M = config.mlp_up_rows();
        const int Hq = config.NumQueryHeads;
        const int Hkv = config.NumKeyValHeads;
        const int Hs = config.head_size();
        const int qkv_channels = config.qkv_channels();

        using WeightsType = std::decay_t<decltype(weights)>;
        constexpr bool kHasMLP = has_mlp_weights<WeightsType>::value;

        const bool recompute_ln1 = options.recompute_rmsnorm || options.recompute_attention || options.recompute_block;
        const bool recompute_ln2 = options.recompute_rmsnorm || options.recompute_ffn || options.recompute_block;
        const bool recompute_qkv = options.recompute_qkv || options.recompute_attention || options.recompute_block;
        const bool recompute_att = options.recompute_attention || options.recompute_block;
        const bool recompute_mlp_up = options.recompute_ffn || options.recompute_block;
        const bool recompute_swiglu = options.recompute_swiglu || options.recompute_ffn || options.recompute_block;

        recipe_ops::ForwardQuantView qv{fp8_fwd_quants, fp4_fwd_quants, &quant_acts};

        auto has_op = [&](BlockOp op) {
            for (int i = 0; i < spec.forward_ops_count; ++i) {
                if (spec.forward_ops[i] == op) return true;
            }
            return false;
        };

        const bool has_ln2 = has_op(BlockOp::LN2) || has_op(BlockOp::ResidualLN2);
        const bool mlp_uses_ln1 = !has_ln2;

        if (recompute_ln1) {
            rmsnorm_forward(acts.ln1, acts.ln1_rstd, residual, weights.ln1.weight,
                           qv.ln1_abs_max(), config.RmsNormEps, B, T, C, stream);
        }

        if (recompute_qkv) {
            const auto cache = recipe_ops::forward_cached_weights(weight_manager, MatmulOp::QKV, allow_quant_layer);
            recipe_ops::forward_matmul(
                recipe, rs,
                acts.qkv, acts.ln1, weights.attention.qkv_weight,
                weights.attention.qkv_bias.has_value() ? &weights.attention.qkv_bias.value() : nullptr,
                B, T, C, qkv_channels,
                layer_idx, MatmulOp::QKV,
                qv.ln1_inp_quant(), /*delayed_quantizer_idx=*/-1,
                cache, stream, allow_quant_layer);

            recipe_ops::qk_norm_forward(
                acts.qkv, acts.q_rstd, acts.k_rstd, weights.attention,
                config.RmsNormEps, B, T, qkv_channels, Hq, Hkv, Hs, stream);

            int* pos_ids_ptr = rs.PositionIDs.template get<int>();
            auto& freq_cis = rs.non_block_activations().freq_cis;
            const bool use_qk_norm = recipe_ops::has_qk_norm(weights.attention);
            Tensor& qkv_for_attn = (use_qk_norm && acts.qkv_rope.Data != nullptr) ? acts.qkv_rope : acts.qkv;
            recipe_ops::rope_forward_dispatch(
                qkv_for_attn, acts.qkv, freq_cis, pos_ids_ptr,
                config, options, B, T, Hq, Hkv, Hs, stream);
        }

        if (recompute_att) {
            const bool use_qk_norm = recipe_ops::has_qk_norm(weights.attention);
            Tensor& qkv_for_attn = (use_qk_norm && acts.qkv_rope.Data != nullptr) ? acts.qkv_rope : acts.qkv;
            recipe_ops::attention_forward(acts.att, acts.lse, qkv_for_attn, rs.CuBlasWorkspace,
                                          rs.CudnnHandle, B, T, Hq, Hkv, Hs, stream);
            recipe_ops::record_abs_max(qv.att_abs_max(), acts.att, rs.DeviceProp, stream);

            if (options.recompute_block) {
                const auto cache = recipe_ops::forward_cached_weights(weight_manager, MatmulOp::AttnOut, allow_quant_layer);
                recipe_ops::forward_matmul(
                    recipe, rs,
                    acts.att_out, acts.att, weights.attention.out_weight,
                    nullptr,
                    B, T, Hq * Hs, C,
                    layer_idx, MatmulOp::AttnOut,
                    qv.att_inp_quant(), /*delayed_quantizer_idx=*/-1,
                    cache, stream, allow_quant_layer);
            }
        }

        if (recompute_ln2) {
            if (spec.variant == BlockVariant::Dense && options.recompute_block) {
                fused_residual_rmsnorm_forward(
                    acts.residual_att, acts.ln2, acts.ln2_rstd,
                    residual, acts.att_out, weights.ln2.weight,
                    qv.ln2_abs_max(), config.RmsNormEps, B * T, C, stream
                );
            } else {
                if (spec.variant == BlockVariant::Parallel && options.recompute_block) {
                    const long N = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(C);
                    vector_add_sr(acts.residual_att, residual, acts.att_out, 1.0f, N, /*seed=*/0, stream);
                }
                Tensor& ln2_input = spec.ln2_on_residual_att ? acts.residual_att : residual;
                rmsnorm_forward(acts.ln2, acts.ln2_rstd, ln2_input, weights.ln2.weight,
                               qv.ln2_abs_max(), config.RmsNormEps, B, T, C, stream);
            }
        }

        if (recompute_mlp_up) {
            if constexpr (kHasMLP) {
                if (rs.ffn_temps_on_stack()) {
                    if (acts.mlp_up.Data == nullptr) rs.temp_acquire(acts.mlp_up);
                    if (acts.swiglu.Data == nullptr) rs.temp_acquire(acts.swiglu);
                }

                Tensor& mlp_in = mlp_uses_ln1 ? acts.ln1 : acts.ln2;
                Tensor* mlp_in_quant = mlp_uses_ln1 ? qv.ln1_inp_quant() : qv.ln2_inp_quant();
                const auto cache = recipe_ops::forward_cached_weights(weight_manager, MatmulOp::MLPUp, allow_quant_layer);
                recipe_ops::forward_matmul(
                    recipe, rs,
                    acts.mlp_up, mlp_in, weights.mlp_up_weight,
                    nullptr,
                    B, T, C, M,
                    layer_idx, MatmulOp::MLPUp,
                    mlp_in_quant, /*delayed_quantizer_idx=*/-1,
                    cache, stream, allow_quant_layer);
            }
        }

        if (recompute_swiglu) {
            if constexpr (kHasMLP) {
                if (rs.ffn_temps_on_stack()) {
                    if (acts.swiglu.Data == nullptr) rs.temp_acquire(acts.swiglu);
                }
                switch (config.activation_type) {
                    case ActivationType::SwiGLU:
                        recipe_ops::swiglu_forward(
                            recipe,
                            acts.swiglu,
                            nullptr,
                            acts.mlp_up,
                            qv.swiglu_abs_max(),
                            B, T, D,
                            stream);
                        break;
                    case ActivationType::GeGLU:
                        throw std::logic_error("BlockExecutor::recompute: GeGLU activation not implemented");
                    case ActivationType::SiLU: {
                        const long N = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(M);
                        silu_forward(acts.swiglu, acts.mlp_up, N, stream);
                        recipe_ops::record_abs_max(qv.swiglu_abs_max(), acts.swiglu, rs.DeviceProp, stream);
                        break;
                    }
                    case ActivationType::ReLU2: {
                        const long N = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(M);
                        relu2_forward(acts.swiglu, acts.mlp_up, N, stream);
                        recipe_ops::record_abs_max(qv.swiglu_abs_max(), acts.swiglu, rs.DeviceProp, stream);
                        break;
                    }
                    case ActivationType::ReLU:
                    case ActivationType::GeLU:
                        throw std::logic_error("BlockExecutor::recompute: Activation type not supported yet");
                    default:
                        throw std::logic_error("BlockExecutor::recompute: Unknown activation type");
                }
            }
        }
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_COMPOSITE_BLOCK_EXECUTOR_H
