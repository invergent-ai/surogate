// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL model execution functions (forward, backward, validation, run state allocation).

#include "dsl/dsl_model.h"
#include "dsl/dsl_model_internal.h"
#include "dsl/dsl_runtime.h"
#include "dsl/graph_executor.h"
#include "dsl/graph_executor_helpers.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <iostream>

#include "kernels/kernels.h"
#include "modules/forward_hooks.h"
#include "modules/backward_hooks.h"
#include "modules/fp8_scaling_state.h"
#include "modules/lora/lora_utils.h"
#include "modules/lora/lora_model_utils.h"
#include "modules/optimizers/adamw_8bit.h"
#include "utilities/comm.h"

namespace dsl {

void DslModel::forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::forward called before allocate_run_state()");
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
        const int D = (int)cfg.IntermediateSize;
        const int Hq = (int)cfg.NumQueryHeads;
        const int Hkv = (int)cfg.NumKeyValHeads;
        const int Hs = (int)cfg.head_size();
        const int rank = mLoRAConfig->rank;
        const float scaling = mLoRAConfig->scaling();
        const float dropout = mLoRAConfig->dropout;
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
            case modules::ForwardHookPoint::AfterQKVProjection: {
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
                if (lora_block.attention.o.has_value()) {
                    modules::detail::apply_lora_contribution(acts.att_out, 0, acts.att, lora_block.attention.o.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(3), is_training,
                                                    B * T, Hq * Hs, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterMLPUpProjection: {
                if (lora_block.mlp.up.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_up, 0, acts.ln2, lora_block.mlp.up.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(4), is_training,
                                                    B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.mlp.gate.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_up, D, acts.ln2, lora_block.mlp.gate.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(5), is_training,
                                                    B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterMLPDownProjection: {
                if (lora_block.mlp.down.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_down, 0, acts.swiglu, lora_block.mlp.down.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(6), is_training,
                                                    B * T, D, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
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
        const int D = (int)cfg.IntermediateSize;
        const int Hq = (int)cfg.NumQueryHeads;
        const int Hkv = (int)cfg.NumKeyValHeads;
        const int Hs = (int)cfg.head_size();
        const int rank = mLoRAConfig->rank;
        const float scaling = mLoRAConfig->scaling();

        auto& acts = rs.simplified_acts(layer_idx);
        auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);

        switch (point) {
            case modules::ForwardHookPoint::AfterQKVProjection: {
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
                if (lora_block.attention.o.has_value()) {
                    modules::detail::apply_lora_contribution(acts.att_out, 0, acts.att, lora_block.attention.o.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, Hq * Hs, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterMLPUpProjection: {
                if (lora_block.mlp.up.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_up, 0, acts.ln2, lora_block.mlp.up.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.mlp.gate.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_up, D, acts.ln2, lora_block.mlp.gate.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterMLPDownProjection: {
                if (lora_block.mlp.down.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_down, 0, acts.swiglu, lora_block.mlp.down.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, D, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
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

    if (!lora_enabled()) {
        mExecutor->backward(inputs, targets, comm, grad_accum_steps, micro_step);
        return;
    }

    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;

    mLoRAGrads->start_micro_step(main_stream, micro_step, grad_accum_steps);

    auto hook = [this, &comm](int layer_idx, bool accumulate, cudaStream_t stream, modules::BackwardHookPoint point, void* context) {
        (void)context;
        static int hook_call_count = 0;
        if (hook_call_count < 10) {
            std::cerr << "[LoRA hook] called, layer=" << layer_idx << " point=" << static_cast<int>(point) << std::endl;
            hook_call_count++;
        }
        const auto& cfg = mModelConfig;
        auto& rs = *mRunState;
        const int B = (int)rs.B;
        const int T = (int)rs.T;
        const int C = (int)cfg.HiddenSize;
        const int D = (int)cfg.IntermediateSize;
        const int Hq = (int)cfg.NumQueryHeads;
        const int Hkv = (int)cfg.NumKeyValHeads;
        const int Hs = (int)cfg.head_size();
        const int rank = mLoRAConfig->rank;
        const float dropout = mLoRAConfig->dropout;
        const bool is_training = mLoRARunState->is_training;
        const int micro_step = mLoRARunState->micro_step;

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

                // Projection type 6 = Down
                const unsigned int dropout_seed = get_dropout_seed(6);

                modules::detail::backward_lora_layer(
                    lora_grads.mlp.down->A, lora_grads.mlp.down->B,
                    da.d_swiglu,
                    da.d_res_ffn, 0,
                    a.swiglu,
                    lora_block.mlp.down->A, lora_block.mlp.down->B,
                    mLoRAConfig->scaling(),
                    dropout, dropout_seed, is_training,
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

                // Get ln2 input: either from stored activation or recompute from residual stream
                // LN2 input is residual_att = res_ffn[L-1] + att_out[L]
                Tensor ln2_input;
                if (mOptions.recompute_enabled()) {
                    if (mLoRARunState && mLoRARunState->recompute_ln.Data) {
                        const std::string ln2_name = "blocks[" + std::to_string(layer_idx) + "].ln2_weight";
                        Tensor& ln2_weight = mParams->get(ln2_name);
                        // Prefer using recomputed residual_att from simplified_acts.
                        // Fallback uses res_ffn[L-1], but this path shouldn't be hit
                        // when recompute is working correctly since residual_att should be populated.
                        Tensor ln2_residual;
                        if (a.residual_att.Data) {
                            ln2_residual = a.residual_att;
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
                        ln2_input = a.ln2;
                    }
                } else {
                    ln2_input = a.ln2;
                }

                // Prepare gradient tensors (use empty tensor if projection not enabled)
                Tensor dA_up{}, dB_up{}, dA_gate{}, dB_gate{};
                modules::LoRALayerWeights<Tensor> lora_up{}, lora_gate{};

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
                    da.d_ln2,
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
            } break;
            case modules::BackwardHookPoint::AfterAttnOutBackward: {
                if (!lora_block.attention.o.has_value()) break;

                bool lora_accum = false;
                auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                lora_accum = lora_accum || accumulate;
                if (!lora_grads.attention.o.has_value()) break;

                auto& a = rs.simplified_acts(layer_idx);
                auto& da = rs.simplified_grads(layer_idx);

                // Debug: check if tensors are valid
                static int attn_debug_count = 0;
                if (attn_debug_count < 5) {
                    cudaStreamSynchronize(stream);
                    float d_att_val = 0.0f, att_val = 0.0f;
                    if (da.d_att.Data) {
                        cudaMemcpy(&d_att_val, da.d_att.Data, sizeof(float), cudaMemcpyDeviceToHost);
                    }
                    if (a.att.Data) {
                        cudaMemcpy(&att_val, a.att.Data, sizeof(float), cudaMemcpyDeviceToHost);
                    }
                    std::cerr << "[AttnO hook L" << layer_idx << "] da.d_att.Data=" << da.d_att.Data
                              << " d_att_val=" << d_att_val << " a.att.Data=" << a.att.Data
                              << " att_val=" << att_val << std::endl;
                    attn_debug_count++;
                }

                // Projection type 3 = O
                const unsigned int dropout_seed = get_dropout_seed(3);

                modules::detail::backward_lora_layer(
                    lora_grads.attention.o->A, lora_grads.attention.o->B,
                    da.d_att,
                    da.d_res_att, 0,
                    a.att,
                    lora_block.attention.o->A, lora_block.attention.o->B,
                    mLoRAConfig->scaling(),
                    dropout, dropout_seed, is_training,
                    mLoRARunState->intermediate, mLoRARunState->slice,
                    B * T, Hq * Hs, C, rank, lora_accum,
                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
            } break;
            case modules::BackwardHookPoint::AfterQKVBackward: {
                bool lora_accum = false;
                auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                lora_accum = lora_accum || accumulate;

                auto& a = rs.simplified_acts(layer_idx);
                auto& da = rs.simplified_grads(layer_idx);

                // Get ln1 input: either from stored activation or recompute from residual
                // LN1 input is res_ffn[L-1] (output of previous layer) for layer L > 0
                Tensor ln1_input;
                if (mOptions.recompute_enabled()) {
                    if (mLoRARunState && mLoRARunState->recompute_ln.Data) {
                        const std::string ln1_name = "blocks[" + std::to_string(layer_idx) + "].ln1_weight";
                        Tensor& ln1_weight = mParams->get(ln1_name);
                        Tensor ln1_residual;
                        if (layer_idx == 0) {
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

                // Debug: check if tensors are valid
                static int qkv_debug_count = 0;
                if (qkv_debug_count < 5) {
                    cudaStreamSynchronize(stream);
                    float d_qkv_val = 0.0f, ln1_val = 0.0f;
                    if (da.d_qkv.Data) {
                        cudaMemcpy(&d_qkv_val, da.d_qkv.Data, sizeof(float), cudaMemcpyDeviceToHost);
                    }
                    if (ln1_input.Data) {
                        cudaMemcpy(&ln1_val, ln1_input.Data, sizeof(float), cudaMemcpyDeviceToHost);
                    }
                    std::cerr << "[QKV hook L" << layer_idx << "] da.d_qkv.Data=" << da.d_qkv.Data
                              << " d_qkv_val=" << d_qkv_val << " ln1_input.Data=" << ln1_input.Data
                              << " ln1_val=" << ln1_val << std::endl;
                    qkv_debug_count++;
                }

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

    static int bwd_hook_count = 0;
    if (bwd_hook_count < 5) {
        std::cerr << "[DslModel::backward] calling backward_with_hook, bwd_hook_count=" << bwd_hook_count << std::endl;
        bwd_hook_count++;
    }

    mExecutor->backward_with_hook(inputs, targets, comm, grad_accum_steps, micro_step, hook);

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
    mRunState = std::make_unique<DslRunState>(*mConfig, mOptions, B, T, mAllocator, lora_enabled(),
                                              dummy_stack_bytes, /*allocate_stack=*/false);
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
        moe_extra = expert_gate_up_tp + expert_down_tp + permuted_tokens;
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
    const long safety_bytes = std::max(64L * 1024 * 1024, base_size / 8);
    long required_size = std::max(1024L * 1024, base_size + base_size + moe_extra + safety_bytes + extra_tmp);
    required_size += 512L * 1024 * 1024;  // extra slack for unmodeled temps
    required_size = std::max(required_size, 3L * 1024 * 1024 * 1024);  // 3GB minimum for full fine-tune stability
    const auto high_mark = mRunState->Stack.get_high_mark();
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

    GraphExecutorOptions exec_opts;
    exec_opts.auto_backward = true;
    exec_opts.debug_print_backward = false;
    mExecutor = std::make_unique<GraphExecutor>(*mModule, *mRunState, *mParams, *mGrads, mModelConfig, mOptions, exec_opts);
    if (!mRngState.empty()) {
        mExecutor->set_rng_state(mRngState);
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
            if (!mLoRAAdamW8BitState->quantiles1.Data) {
                mLoRAAdamW8BitState->quantiles1 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_quantiles1", {256});
                mLoRAAdamW8BitState->quantiles2 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_quantiles2", {256});
                std::vector<float> h_q1(256), h_q2(256);
                create_adamw8bit_quantiles1(h_q1.data());
                create_adamw8bit_quantiles2(h_q2.data());
                CUDA_CHECK(cudaMemcpy(mLoRAAdamW8BitState->quantiles1.Data, h_q1.data(), h_q1.size() * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(mLoRAAdamW8BitState->quantiles2.Data, h_q2.data(), h_q2.size() * sizeof(float), cudaMemcpyHostToDevice));
            }
        } else {
            if (!mAdamW8BitState) {
                mAdamW8BitState = std::make_unique<AdamW8BitState>();
            }
            if (!mAdamW8BitState->quantiles1.Data) {
                mAdamW8BitState->quantiles1 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_quantiles1", {256});
                mAdamW8BitState->quantiles2 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_quantiles2", {256});
                std::vector<float> h_q1(256), h_q2(256);
                create_adamw8bit_quantiles1(h_q1.data());
                create_adamw8bit_quantiles2(h_q2.data());
                CUDA_CHECK(cudaMemcpy(mAdamW8BitState->quantiles1.Data, h_q1.data(), h_q1.size() * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(mAdamW8BitState->quantiles2.Data, h_q2.data(), h_q2.size() * sizeof(float), cudaMemcpyHostToDevice));
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

}  // namespace dsl
