// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Segment-based backward recomputation support for DSL Graph executor.
//
// This file implements a clean, segment-based recomputation system that:
// 1. Organizes recomputation into atomic segments (attention path, FFN path)
// 2. Ensures correct dependency ordering within each segment
// 3. Guarantees numerical consistency with the forward pass
//
// Recompute Levels:
//   - None: All activations saved, no recomputation
//   - Standard: Recompute attention and FFN intermediates from checkpoints
//   - Aggressive: Recompute everything except residuals and LSE

#include "dsl/graph_executor.h"

#include <string>

#include "dsl/dsl_run_state.h"
#include "dsl/dsl_runtime.h"
#include "dsl/forward_plan.h"
#include "dsl/graph_executor_helpers.h"
#include "dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "modules/lora/lora_model_utils.h"
#include "modules/lora/lora_run_state.h"
#include "modules/matmul_context.h"
#include "modules/model_config.h"
#include "training/runtime_options.h"
#include "utilities/stack.h"
#include "utilities/tensor.h"

namespace dsl {

// ============================================================================
// Segment-based recomputation implementation
// ============================================================================
//
// The transformer block is divided into two recompute segments:
//
// Attention Segment:
//   Inputs (checkpoints): residual[l], ln1_rstd[l]
//   Computes: ln1 -> qkv -> rope -> attention -> out_proj -> att_out
//   Outputs: ln1, qkv (post-RoPE), att, att_out
//
// FFN Segment:
//   Inputs (checkpoints): residual_att[l] (= residual + att_out), ln2_rstd[l]
//   Computes: ln2 -> mlp_up -> swiglu
//   Outputs: ln2, mlp_up, swiglu
//
// What is ALWAYS saved (never recomputed):
//   - residual[l]: Input to each layer, required for gradient accumulation
//   - lse[l]: Log-sum-exp from attention (buffered; recompute overwrites)
//   - ln1_rstd[l], ln2_rstd[l]: RMSNorm statistics, small and needed for backward
//   - q_rstd[l], k_rstd[l]: QK-norm statistics if enabled
//
// Standard level may save additional intermediates depending on runtime options.
// Aggressive level only saves the minimum checkpoint set above.
//

void GraphExecutor::recompute_attention_segment(int layer_idx, long B, long T) {
    if (!mOptions.recompute_enabled()) return;
    if (layer_idx < 0 || layer_idx >= mConfig.NumLayers) return;

    auto& rs = mRunState;
    const auto& config = mConfig;
    const int Bv = static_cast<int>(B);
    const int Tv = static_cast<int>(T);
    const int C = static_cast<int>(config.HiddenSize);

    auto& acts = rs.simplified_acts(layer_idx);

    // Helper to ensure activation buffer is allocated
    auto ensure_act = [&](Tensor& t) {
        if (!t.Data) {
            rs.temp_acquire(t);
        }
    };

    auto block_name = [&](const char* field) {
        return "blocks[" + std::to_string(layer_idx) + "]." + field;
    };

    // When residual offloading is enabled, we need to fetch the residual back from CPU
    // before recomputation can use it. This is critical for the combination of
    // offload_residual + recompute + LoRA which needs residuals during backward.
    // NOTE: The residual INPUT to layer L is res_ffn[L-1] (output of previous layer).
    // So we fetch layer_idx - 1, not layer_idx.
    if (layer_idx > 0 && rs.has_residual_offloading()) {
        rs.fetch_residual(layer_idx - 1, rs.side_stream());
    }

    // Get residual input to this layer (res_ffn[layer_idx - 1] for layer > 0)
    Tensor& res_in = (layer_idx == 0)
        ? rs.non_block_activations().encoded
        : rs.get_residual(layer_idx - 1, rs.MainStream);

    // In FFT mode (not LoRA-only), we save qkv/att/att_out/lse per-layer from the forward pass.
    // We only need to recompute LN1 (bit-exact from saved rstd) since ln1 buffer is shared.
    // Skip everything else to preserve bit-exact gradients and avoid cuDNN non-determinism.
    const bool skip_attention_recompute = !rs.is_lora_only_mode();
    if (skip_attention_recompute) {
        // Recompute LN1 using saved rstd for bit-exact results
        // This is needed because ln1 buffer is shared across layers
        ensure_act(acts.ln1);
        Tensor& ln1_weight = mWeights.get(block_name("ln1_weight"));
        rmsnorm_apply_saved(acts.ln1, res_in, ln1_weight, acts.ln1_rstd, Bv, Tv, C, rs.MainStream);
        return;
    }

    const int Hq = static_cast<int>(config.NumQueryHeads);
    const int Hkv = static_cast<int>(config.NumKeyValHeads);
    const int Hs = static_cast<int>(config.head_size());
    const int qkv_channels = Hs * (Hq + 2 * Hkv);
    const int BT = Bv * Tv;
    const bool lora_enabled = mLoRAConfig && mLoRAWeights && mLoRARunState &&
                              mLoRAConfig->enabled() && mLoRAWeights->enabled();

    const LayerForwardPlan* fwd_plan = forward_plan(layer_idx);
    const bool allow_quant = allow_quant_layer(mOptions, config, layer_idx);
    const bool have_recipe = mOptions.TrainingRecipe != nullptr;

    // ========================================================================
    // Step 1: Recompute LN1 using saved rstd for bit-exact results (LoRA mode)
    // ========================================================================
    ensure_act(acts.ln1);
    Tensor& ln1_weight = mWeights.get(block_name("ln1_weight"));
    rmsnorm_apply_saved(acts.ln1, res_in, ln1_weight, acts.ln1_rstd, Bv, Tv, C, rs.MainStream);

    // ========================================================================
    // Step 2: Recompute QKV projection
    // ========================================================================
    // Helper to get forward plan for a matmul op (used by multiple steps)
    auto plan_for = [&](modules::MatmulOp op) -> const MatmulForwardPlan* {
        if (!fwd_plan) return nullptr;
        switch (op) {
            case modules::MatmulOp::QKV: return fwd_plan->qkv.valid ? &fwd_plan->qkv : nullptr;
            case modules::MatmulOp::AttnOut: return fwd_plan->out_proj.valid ? &fwd_plan->out_proj : nullptr;
            case modules::MatmulOp::MLPUp: return fwd_plan->mlp_up.valid ? &fwd_plan->mlp_up : nullptr;
            case modules::MatmulOp::MLPDown: return fwd_plan->mlp_down.valid ? &fwd_plan->mlp_down : nullptr;
            default: return nullptr;
        }
    };

    ensure_act(acts.qkv);
    const std::string qkv_weight_name = block_name("qkv_weight");
    Tensor& qkv_weight = mWeights.get(qkv_weight_name);
    Tensor ln1_flat = view_tensor(acts.ln1, {B * T, C});
    Tensor qkv_flat = view_tensor(acts.qkv, {B * T, qkv_channels});

    // Check for bias
    std::string bias_name = block_name("qkv_bias");
    std::optional<Tensor> qkv_bias;

    const MatmulForwardPlan* qkv_plan = plan_for(modules::MatmulOp::QKV);
    const bool use_bias = qkv_plan ? qkv_plan->has_bias : mWeights.has(bias_name);
    if (use_bias && mWeights.has(bias_name)) {
        qkv_bias = mWeights.get(bias_name);
    }

    const bool use_recipe_qkv = qkv_plan ? (qkv_plan->use_recipe && have_recipe) : have_recipe;
    if (use_recipe_qkv) {
        const recipes::Recipe& recipe = *mOptions.TrainingRecipe;
        modules::MatmulContext ctx;
        ctx.out = &qkv_flat;
        ctx.inp = &ln1_flat;
        ctx.weight = &qkv_weight;
        ctx.bias = qkv_bias ? &*qkv_bias : nullptr;
        ctx.B = Bv;
        ctx.T = Tv;
        ctx.C_in = C;
        ctx.C_out = qkv_channels;
        ctx.run_state = &rs;
        ctx.stream = rs.MainStream;
        ctx.layer_idx = layer_idx;
        ctx.op = modules::MatmulOp::QKV;
        ctx.allow_fp8 = qkv_plan ? qkv_plan->allow_fp8 : allow_quant;
        ctx.allow_fp4 = qkv_plan ? qkv_plan->allow_fp4 : (allow_quant && mOptions.fp4_enabled());
        if (ctx.allow_fp8) {
            ctx.inp_quant = fp8_forward_buffer(rs, ctx.op);
            ctx.delayed_quantizer_idx = qkv_plan ? qkv_plan->delayed_quantizer_idx
                                                 : fp8_quantizer_index(rs, ctx.op, layer_idx);
            if (!qkv_plan || qkv_plan->use_fp8_cache) {
                ctx.cached_weight = get_fp8_cached_weight(qkv_weight_name, qkv_weight, rs.MainStream);
            }
        }
        if (ctx.allow_fp4 && (qkv_plan == nullptr || qkv_plan->use_fp4_cache)) {
            if (const auto* fp4_cache = get_fp4_cached_weight(qkv_weight_name, qkv_weight, rs.MainStream)) {
                ctx.cached_fp4_data = &fp4_cache->data;
                ctx.cached_fp4_scales = &fp4_cache->scales;
                ctx.cached_fp4_amax = fp4_cache->amax.get<float>();
            }
        }
        recipe.forward_matmul(ctx);
    } else {
        matmul(qkv_flat, qkv_weight, ln1_flat, qkv_bias, nullptr, nullptr,
               rs.CublasLtHandle, rs.CuBlasWorkspace,
               qkv_channels, static_cast<int>(B * T), C, EMMTranspose::TN, false, rs.MainStream);
    }

    // Re-apply LoRA contributions
    if (lora_enabled) {
        auto& lora_block = mLoRAWeights->get_block(layer_idx, rs.MainStream);
        const int rank = mLoRAConfig->rank;
        const float scaling = mLoRAConfig->scaling();
        const float dropout = mLoRAConfig->dropout;
        const bool training = mLoRARunState->is_training;
        auto get_dropout_seed = [&](int proj_type) -> unsigned int {
            return mLoRARunState->dropout_base_seed
                   + static_cast<unsigned int>(layer_idx) * 1000000u
                   + static_cast<unsigned int>(proj_type) * 100000u
                   + static_cast<unsigned int>(mLoRARunState->micro_step) * 10000u;
        };
        if (lora_block.attention.q.has_value()) {
            modules::detail::apply_lora_contribution(acts.qkv, 0, acts.ln1, lora_block.attention.q.value(),
                                                     mLoRARunState->intermediate, mLoRARunState->slice,
                                                     scaling, dropout, get_dropout_seed(0), training,
                                                     BT, C, Hq * Hs, rank,
                                                     rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
        }
        if (lora_block.attention.k.has_value()) {
            modules::detail::apply_lora_contribution(acts.qkv, Hq * Hs, acts.ln1, lora_block.attention.k.value(),
                                                     mLoRARunState->intermediate, mLoRARunState->slice,
                                                     scaling, dropout, get_dropout_seed(1), training,
                                                     BT, C, Hkv * Hs, rank,
                                                     rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
        }
        if (lora_block.attention.v.has_value()) {
            modules::detail::apply_lora_contribution(acts.qkv, (Hq + Hkv) * Hs, acts.ln1, lora_block.attention.v.value(),
                                                     mLoRARunState->intermediate, mLoRARunState->slice,
                                                     scaling, dropout, get_dropout_seed(2), training,
                                                     BT, C, Hkv * Hs, rank,
                                                     rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
        }
    }

    // ========================================================================
    // Step 3: Recompute QK-norm + RoPE
    // ========================================================================
    const bool use_qk_norm = (fwd_plan && fwd_plan->attn.valid) ? fwd_plan->attn.use_qk_norm : config.UseQKNorm;
    if (use_qk_norm) {
        // q_rstd and k_rstd are always saved
        Tensor qkv_view = view_tensor(acts.qkv, {B, T, qkv_channels});
        const bool rope_fused_plan = (fwd_plan && fwd_plan->attn.valid) ? fwd_plan->attn.rope_fused : false;
        if (rope_fused_plan) {
            qkv_qk_norm_rope_forward(qkv_view, acts.q_rstd, acts.k_rstd,
                                     mWeights.get(block_name("q_norm_weight")),
                                     mWeights.get(block_name("k_norm_weight")),
                                     rs.non_block_activations().freq_cis,
                                     reinterpret_cast<int*>(rs.PositionIDs.Data),
                                     static_cast<float>(config.RmsNormEps),
                                     Bv, Tv, Hq, Hkv, Hs, rs.MainStream);
        } else {
            qkv_head_rmsnorm_forward(qkv_view, acts.q_rstd,
                                     mWeights.get(block_name("q_norm_weight")),
                                     static_cast<float>(config.RmsNormEps),
                                     Bv, Tv, qkv_channels, Hq, Hs, 0, rs.MainStream);
            qkv_head_rmsnorm_forward(qkv_view, acts.k_rstd,
                                     mWeights.get(block_name("k_norm_weight")),
                                     static_cast<float>(config.RmsNormEps),
                                     Bv, Tv, qkv_channels, Hkv, Hs, Hq * Hs, rs.MainStream);
            Tensor qkv_rope = (acts.qkv.Rank == 4)
                ? acts.qkv
                : view_tensor(acts.qkv, {B, T, Hq + 2 * Hkv, Hs});
            const int rotary_dim = (fwd_plan && fwd_plan->attn.valid && fwd_plan->attn.rotary_dim > 0)
                ? fwd_plan->attn.rotary_dim
                : Hs;
            rope_forward(qkv_rope, qkv_rope, rs.non_block_activations().freq_cis,
                         reinterpret_cast<int*>(rs.PositionIDs.Data), nullptr,
                         Bv, Tv, Hq, Hkv, Hs, rotary_dim, rs.MainStream);
        }
    } else {
        Tensor qkv_rope = (acts.qkv.Rank == 4)
            ? acts.qkv
            : view_tensor(acts.qkv, {B, T, Hq + 2 * Hkv, Hs});
        const int rotary_dim = (fwd_plan && fwd_plan->attn.valid && fwd_plan->attn.rotary_dim > 0)
            ? fwd_plan->attn.rotary_dim
            : Hs;
        rope_forward(qkv_rope, qkv_rope, rs.non_block_activations().freq_cis,
                     reinterpret_cast<int*>(rs.PositionIDs.Data), nullptr,
                     Bv, Tv, Hq, Hkv, Hs, rotary_dim, rs.MainStream);
    }

    // ========================================================================
    // Step 4 & 5: Recompute Attention + Out Projection
    // ========================================================================
    // Note: We only reach here in LoRA mode (FFT mode returns early above).
    // In LoRA mode, att/att_out/lse are shared buffers and must be recomputed.
    ensure_act(acts.att);

    Tensor qkv_view = (acts.qkv.Rank == 4)
        ? acts.qkv
        : view_tensor(acts.qkv, {B, T, Hq + 2 * Hkv, Hs});
    if (!rs.scratch().cudnn_workspace.Data) {
        rs.temp_acquire(rs.scratch().cudnn_workspace);
    }
    Tensor att_out = view_tensor(acts.att, {B, T, Hq, Hs});
    Tensor lse_view = view_tensor(acts.lse, {B, Hq, T});
    attention_forward_cudnn(att_out, lse_view, qkv_view, rs.scratch().cudnn_workspace,
                            rs.CudnnHandle, Bv, Tv, Hq, Hkv, Hs, rs.MainStream);

    // Recompute out_proj
    ensure_act(acts.att_out);
    const std::string out_weight_name = block_name("out_weight");
    Tensor& out_weight = mWeights.get(out_weight_name);
    Tensor att_flat = view_tensor(acts.att, {B * T, Hq * Hs});
    Tensor att_out_flat = view_tensor(acts.att_out, {B * T, C});

    const MatmulForwardPlan* out_plan = plan_for(modules::MatmulOp::AttnOut);
    const bool use_recipe_out = out_plan ? (out_plan->use_recipe && have_recipe) : have_recipe;
    if (use_recipe_out) {
        const recipes::Recipe& recipe = *mOptions.TrainingRecipe;
        modules::MatmulContext ctx;
        ctx.out = &att_out_flat;
        ctx.inp = &att_flat;
        ctx.weight = &out_weight;
        ctx.bias = nullptr;
        ctx.B = Bv;
        ctx.T = Tv;
        ctx.C_in = Hq * Hs;
        ctx.C_out = C;
        ctx.run_state = &rs;
        ctx.stream = rs.MainStream;
        ctx.layer_idx = layer_idx;
        ctx.op = modules::MatmulOp::AttnOut;
        ctx.allow_fp8 = out_plan ? out_plan->allow_fp8 : allow_quant;
        ctx.allow_fp4 = out_plan ? out_plan->allow_fp4 : (allow_quant && mOptions.fp4_enabled());
        if (ctx.allow_fp8) {
            ctx.inp_quant = fp8_forward_buffer(rs, ctx.op);
            ctx.delayed_quantizer_idx = out_plan ? out_plan->delayed_quantizer_idx
                                                 : fp8_quantizer_index(rs, ctx.op, layer_idx);
            if (!out_plan || out_plan->use_fp8_cache) {
                ctx.cached_weight = get_fp8_cached_weight(out_weight_name, out_weight, rs.MainStream);
            }
        }
        if (ctx.allow_fp4 && (out_plan == nullptr || out_plan->use_fp4_cache)) {
            if (const auto* fp4_cache = get_fp4_cached_weight(out_weight_name, out_weight, rs.MainStream)) {
                ctx.cached_fp4_data = &fp4_cache->data;
                ctx.cached_fp4_scales = &fp4_cache->scales;
                ctx.cached_fp4_amax = fp4_cache->amax.get<float>();
            }
        }
        recipe.forward_matmul(ctx);
    } else {
        matmul(att_out_flat, out_weight, att_flat, std::nullopt, nullptr, nullptr,
               rs.CublasLtHandle, rs.CuBlasWorkspace,
               C, static_cast<int>(B * T), Hq * Hs, EMMTranspose::TN, false, rs.MainStream);
    }

    // Re-apply LoRA o_proj contribution
    if (lora_enabled) {
        auto& lora_block = mLoRAWeights->get_block(layer_idx, rs.MainStream);
        if (lora_block.attention.o.has_value()) {
            const int rank = mLoRAConfig->rank;
            const float scaling = mLoRAConfig->scaling();
            const float dropout = mLoRAConfig->dropout;
            const bool training = mLoRARunState->is_training;
            auto get_dropout_seed = [&](int proj_type) -> unsigned int {
                return mLoRARunState->dropout_base_seed
                       + static_cast<unsigned int>(layer_idx) * 1000000u
                       + static_cast<unsigned int>(proj_type) * 100000u
                       + static_cast<unsigned int>(mLoRARunState->micro_step) * 10000u;
            };
            modules::detail::apply_lora_contribution(acts.att_out, 0, acts.att, lora_block.attention.o.value(),
                                                     mLoRARunState->intermediate, mLoRARunState->slice,
                                                     scaling, dropout, get_dropout_seed(3), training,
                                                     BT, Hq * Hs, C, rank,
                                                     rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
        }
    }
}


void GraphExecutor::recompute_ffn_segment(int layer_idx, long B, long T) {
    if (!mOptions.recompute_enabled()) return;
    if (layer_idx < 0 || layer_idx >= mConfig.NumLayers) return;

    auto& rs = mRunState;
    const auto& config = mConfig;
    const int Bv = static_cast<int>(B);
    const int Tv = static_cast<int>(T);
    const int C = static_cast<int>(config.HiddenSize);
    const int BT = Bv * Tv;

    auto& acts = rs.simplified_acts(layer_idx);

    auto ensure_act = [&](Tensor& t) {
        if (!t.Data) {
            rs.temp_acquire(t);
        }
    };

    auto block_name = [&](const char* field) {
        return "blocks[" + std::to_string(layer_idx) + "]." + field;
    };

    // When residual offloading is enabled, we need to fetch the residual back from CPU
    // before recomputation can use it. This is critical for the combination of
    // offload_residual + recompute + LoRA which needs residuals during backward.
    // NOTE: The residual INPUT to layer L is res_ffn[L-1] (output of previous layer).
    // So we fetch layer_idx - 1, not layer_idx.
    if (layer_idx > 0 && rs.has_residual_offloading()) {
        rs.fetch_residual(layer_idx - 1, rs.side_stream());
    }

    // Get residual input: for FFN, this is residual + att_out (res_ffn[layer_idx - 1] for layer > 0)
    Tensor& res_in = (layer_idx == 0)
        ? rs.non_block_activations().encoded
        : rs.get_residual(layer_idx - 1, rs.MainStream);

    // In FFT mode (not LoRA-only), we save mlp_up/swiglu per-layer from the forward pass.
    // We only need to recompute LN2 and residual_att (bit-exact from saved inputs).
    // Skip mlp_up/swiglu recompute to avoid matmul non-determinism.
    const bool skip_ffn_recompute = !rs.is_lora_only_mode();
    if (skip_ffn_recompute) {
        // Recompute LN2 and residual_att using saved inputs for bit-exact results
        // ln2 buffer is shared, but we can recompute it bit-exactly from saved ln2_rstd
        ensure_act(acts.ln2);
        ensure_act(acts.residual_att);
        Tensor& ln2_weight = mWeights.get(block_name("ln2_weight"));
        fused_residual_rmsnorm_apply_saved(
            acts.residual_att, acts.ln2,
            res_in, acts.att_out, ln2_weight, acts.ln2_rstd, BT, C, rs.MainStream);
        return;
    }

    // MoE models don't have dense MLP weights (mlp_up_weight/mlp_down_weight).
    // Skip MLP recompute for MoE - the MoE ops save their activations differently.
    // Only recompute LN2 and residual_att for MoE models.
    const bool is_moe = config.NumExperts > 0;
    if (is_moe) {
        // For MoE, only recompute LN2 (already done below)
        ensure_act(acts.ln2);
        ensure_act(acts.residual_att);
        Tensor& ln2_weight = mWeights.get(block_name("ln2_weight"));
        fused_residual_rmsnorm_apply_saved(
            acts.residual_att, acts.ln2,
            res_in, acts.att_out, ln2_weight, acts.ln2_rstd, BT, C, rs.MainStream);
        return;
    }

    // LoRA mode: full FFN segment recompute (dense MLP only)
    const int D = static_cast<int>(config.IntermediateSize);
    const int MUp = 2 * D;
    const bool lora_enabled = mLoRAConfig && mLoRAWeights && mLoRARunState &&
                              mLoRAConfig->enabled() && mLoRAWeights->enabled();

    const LayerForwardPlan* fwd_plan = forward_plan(layer_idx);
    const bool allow_quant = allow_quant_layer(mOptions, config, layer_idx);
    const bool have_recipe = mOptions.TrainingRecipe != nullptr;

    auto plan_for = [&](modules::MatmulOp op) -> const MatmulForwardPlan* {
        if (!fwd_plan) return nullptr;
        switch (op) {
            case modules::MatmulOp::MLPUp: return fwd_plan->mlp_up.valid ? &fwd_plan->mlp_up : nullptr;
            case modules::MatmulOp::MLPDown: return fwd_plan->mlp_down.valid ? &fwd_plan->mlp_down : nullptr;
            default: return nullptr;
        }
    };

    // ========================================================================
    // Step 1: Recompute LN2 using saved rstd for bit-exact results (LoRA mode)
    // ========================================================================
    ensure_act(acts.ln2);
    ensure_act(acts.residual_att);
    Tensor& ln2_weight = mWeights.get(block_name("ln2_weight"));

    // LN2 takes residual_att = res_in + att_out as input
    // att_out is recomputed during backward when recompute is enabled
    fused_residual_rmsnorm_apply_saved(
        acts.residual_att, acts.ln2,
        res_in, acts.att_out, ln2_weight, acts.ln2_rstd, BT, C, rs.MainStream);

    // ========================================================================
    // Step 2: Recompute MLP Up projection
    // ========================================================================
    ensure_act(acts.mlp_up);
    const std::string mlp_up_weight_name = block_name("mlp_up_weight");
    Tensor& mlp_up_weight = mWeights.get(mlp_up_weight_name);
    Tensor ln2_flat = view_tensor(acts.ln2, {B * T, C});
    Tensor mlp_up_flat = view_tensor(acts.mlp_up, {B * T, MUp});

    const MatmulForwardPlan* mlp_plan = plan_for(modules::MatmulOp::MLPUp);
    const bool use_recipe_mlp = mlp_plan ? (mlp_plan->use_recipe && have_recipe) : have_recipe;
    if (use_recipe_mlp) {
        const recipes::Recipe& recipe = *mOptions.TrainingRecipe;
        modules::MatmulContext ctx;
        ctx.out = &mlp_up_flat;
        ctx.inp = &ln2_flat;
        ctx.weight = &mlp_up_weight;
        ctx.bias = nullptr;
        ctx.B = Bv;
        ctx.T = Tv;
        ctx.C_in = C;
        ctx.C_out = MUp;
        ctx.run_state = &rs;
        ctx.stream = rs.MainStream;
        ctx.layer_idx = layer_idx;
        ctx.op = modules::MatmulOp::MLPUp;
        ctx.allow_fp8 = mlp_plan ? mlp_plan->allow_fp8 : allow_quant;
        ctx.allow_fp4 = mlp_plan ? mlp_plan->allow_fp4 : (allow_quant && mOptions.fp4_enabled());
        if (ctx.allow_fp8) {
            ctx.inp_quant = fp8_forward_buffer(rs, ctx.op);
            ctx.delayed_quantizer_idx = mlp_plan ? mlp_plan->delayed_quantizer_idx
                                                 : fp8_quantizer_index(rs, ctx.op, layer_idx);
            if (!mlp_plan || mlp_plan->use_fp8_cache) {
                ctx.cached_weight = get_fp8_cached_weight(mlp_up_weight_name, mlp_up_weight, rs.MainStream);
            }
        }
        if (ctx.allow_fp4 && (mlp_plan == nullptr || mlp_plan->use_fp4_cache)) {
            if (const auto* fp4_cache = get_fp4_cached_weight(mlp_up_weight_name, mlp_up_weight, rs.MainStream)) {
                ctx.cached_fp4_data = &fp4_cache->data;
                ctx.cached_fp4_scales = &fp4_cache->scales;
                ctx.cached_fp4_amax = fp4_cache->amax.get<float>();
            }
        }
        recipe.forward_matmul(ctx);
    } else {
        matmul(mlp_up_flat, mlp_up_weight, ln2_flat, std::nullopt, nullptr, nullptr,
               rs.CublasLtHandle, rs.CuBlasWorkspace,
               MUp, BT, C, EMMTranspose::TN, false, rs.MainStream);
    }

    // Re-apply LoRA MLP contributions
    if (lora_enabled) {
        auto& lora_block = mLoRAWeights->get_block(layer_idx, rs.MainStream);
        const int rank = mLoRAConfig->rank;
        const float scaling = mLoRAConfig->scaling();
        const float dropout = mLoRAConfig->dropout;
        const bool training = mLoRARunState->is_training;
        auto get_dropout_seed = [&](int proj_type) -> unsigned int {
            return mLoRARunState->dropout_base_seed
                   + static_cast<unsigned int>(layer_idx) * 1000000u
                   + static_cast<unsigned int>(proj_type) * 100000u
                   + static_cast<unsigned int>(mLoRARunState->micro_step) * 10000u;
        };
        if (lora_block.mlp.up.has_value()) {
            modules::detail::apply_lora_contribution(acts.mlp_up, 0, acts.ln2, lora_block.mlp.up.value(),
                                                     mLoRARunState->intermediate, mLoRARunState->slice,
                                                     scaling, dropout, get_dropout_seed(4), training,
                                                     BT, C, D, rank,
                                                     rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
        }
        if (lora_block.mlp.gate.has_value()) {
            modules::detail::apply_lora_contribution(acts.mlp_up, D, acts.ln2, lora_block.mlp.gate.value(),
                                                     mLoRARunState->intermediate, mLoRARunState->slice,
                                                     scaling, dropout, get_dropout_seed(5), training,
                                                     BT, C, D, rank,
                                                     rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
        }
    }

    // ========================================================================
    // Step 3: Recompute SwiGLU activation
    // ========================================================================
    ensure_act(acts.swiglu);
    swiglu_forward(acts.swiglu, acts.mlp_up, nullptr, Bv, Tv, D, rs.MainStream);
}


void GraphExecutor::recompute_block(int layer_idx, long B, long T) {
    if (!mOptions.recompute_enabled()) return;

    // Execute both segments in order
    recompute_attention_segment(layer_idx, B, T);
    recompute_ffn_segment(layer_idx, B, T);
}

}  // namespace dsl
