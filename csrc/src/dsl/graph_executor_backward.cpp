// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Backward recomputation support for DSL Graph executor.

#include "dsl/graph_executor.h"

#include <string>

#include "dsl/dsl_runtime.h"
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
// Optimized backward recomputation support
// ============================================================================

void GraphExecutor::init_recompute_flags() {
    const bool disable_recompute_block = mRunState.is_lora_only_mode() && !mOptions.RecomputeLoRA;
    const bool recompute_block = mOptions.RecomputeBlock && !disable_recompute_block;

    // Check if LoRA is enabled and needs activations for backward
    const bool lora_needs_activations = mLoRAConfig && mLoRAConfig->enabled() &&
                                        mLoRAWeights && mLoRAWeights->enabled();

    // Attention path recomputation
    mRecomputeFlags.att = mOptions.RecomputeAtt || recompute_block;
    mRecomputeFlags.qkv = mOptions.RecomputeQKV || mRecomputeFlags.att;
    mRecomputeFlags.qk_norm = mOptions.RecomputeQKNorm || mRecomputeFlags.qkv;
    mRecomputeFlags.rope = mOptions.RecomputeRoPE || mRecomputeFlags.qkv;
    mRecomputeFlags.out_proj = mOptions.RecomputeOutProj || recompute_block;

    // FFN/MLP path recomputation
    mRecomputeFlags.ffn = mOptions.RecomputeFFN || recompute_block;
    mRecomputeFlags.swiglu = mOptions.RecomputeSwiGLu || mRecomputeFlags.ffn;
    mRecomputeFlags.mlp_down = mOptions.RecomputeMLPDown || recompute_block;

    // CRITICAL FIX: LoRA backward hooks require specific activations as inputs.
    // If any recomputation is enabled but not full block recompute, ensure LoRA dependencies are available.
    // Hook dependencies (see dsl_model.cpp:817-1090):
    //   - AfterMLPDownBackward: needs a.swiglu
    //   - AfterMLPUpBackward: needs a.ln2
    //   - AfterAttnOutBackward: needs a.att
    //   - AfterQKVBackward: needs a.ln1
    if (lora_needs_activations && !recompute_block) {
        // If recomputing any part of the transformer block, we must recompute the full FFN path
        // to ensure swiglu is available for MLPDown backward hook.
        // This also ensures ln2 is recomputed (via line 1447) for MLPUp backward hook.
        if (mRecomputeFlags.att || mRecomputeFlags.qkv || mRecomputeFlags.out_proj) {
            // CRITICAL: We must recompute out_proj (att_out) because ln2 recompute uses att_out
            // as input to fused_residual_rmsnorm_forward. If att_out is not recomputed,
            // ln2 will read stale/invalid data, causing NaN gradients.
            mRecomputeFlags.out_proj = true;
            mRecomputeFlags.ffn = true;
            mRecomputeFlags.swiglu = true;
        }
    }

    // Normalization recomputation
    const bool recompute_rmsnorm = mOptions.RecomputeRMSNorm || recompute_block;
    mRecomputeFlags.ln1 = recompute_rmsnorm || mRecomputeFlags.qkv || recompute_block;
    mRecomputeFlags.ln2 = recompute_rmsnorm || mRecomputeFlags.out_proj || mRecomputeFlags.ffn || recompute_block;

    // Overall flag
    mRecomputeFlags.any = mRecomputeFlags.ln1 || mRecomputeFlags.qkv || mRecomputeFlags.att ||
                          mRecomputeFlags.ln2 || mRecomputeFlags.ffn || mRecomputeFlags.swiglu ||
                          mRecomputeFlags.out_proj || mRecomputeFlags.mlp_down;
}


void GraphExecutor::reset_recompute_graphs() {
    for (auto& graph : mRecomputeGraphs) {
        if (graph != nullptr) {
            cudaGraphExecDestroy(graph);
            graph = nullptr;
        }
    }
    mRecomputeGraphsInitialized = false;
}

void GraphExecutor::recompute_layer_optimized(int layer_idx, long B, long T, bool use_graph) {
    if (!mRecomputeFlags.any) return;
    if (layer_idx < 0 || layer_idx >= mConfig.NumLayers) return;

    auto& rs = mRunState;
    const auto& config = mConfig;
    const auto& flags = mRecomputeFlags;

    // Check if we can use cached CUDA graph
    if (use_graph && mRecomputeGraphsInitialized && layer_idx < static_cast<int>(mRecomputeGraphs.size())) {
        cudaGraphExec_t& graph = mRecomputeGraphs[layer_idx];
        if (graph != nullptr) {
            rs.Stack.restore(mRecomputeCheckpoints[layer_idx]);
            CUDA_CHECK(cudaGraphLaunch(graph, rs.MainStream));
            return;
        }
    }

    // Initialize graph arrays if needed
    if (use_graph && !mRecomputeGraphsInitialized) {
        mRecomputeGraphs.resize(static_cast<std::size_t>(config.NumLayers), nullptr);
        mRecomputeCheckpoints.resize(static_cast<std::size_t>(config.NumLayers));
        mRecomputeGraphsInitialized = true;
    }

    // Capture checkpoint for potential graph capture
    DeviceMemoryStack::Checkpoint checkpoint;
    if (use_graph && layer_idx < static_cast<int>(mRecomputeGraphs.size())) {
        checkpoint = rs.Stack.checkpoint();
        mRecomputeCheckpoints[layer_idx] = checkpoint;
    }

    // Start graph capture if enabled
    cudaGraph_t captured_graph = nullptr;
    if (use_graph && layer_idx < static_cast<int>(mRecomputeGraphs.size()) && mRecomputeGraphs[layer_idx] == nullptr) {
        CUDA_CHECK(cudaStreamBeginCapture(rs.MainStream, cudaStreamCaptureModeThreadLocal));
    }

    // ========================================================================
    // Recomputation logic (streamlined, no conditionals in hot path)
    // ========================================================================
    const int Bv = static_cast<int>(B);
    const int Tv = static_cast<int>(T);
    const int C = static_cast<int>(config.HiddenSize);
    const int D = static_cast<int>(config.IntermediateSize);
    const int Hq = static_cast<int>(config.NumQueryHeads);
    const int Hkv = static_cast<int>(config.NumKeyValHeads);
    const int Hs = static_cast<int>(config.head_size());
    const int qkv_channels = Hs * (Hq + 2 * Hkv);
    const int MUp = 2 * D;
    const int BT = Bv * Tv;
    const bool lora_enabled = mLoRAConfig && mLoRAWeights && mLoRARunState &&
                              mLoRAConfig->enabled() && mLoRAWeights->enabled();

    auto& acts = rs.simplified_acts(layer_idx);
    const LayerForwardPlan* fwd_plan = forward_plan(layer_idx);
    auto plan_for = [&](modules::MatmulOp op) -> const MatmulForwardPlan* {
        if (!fwd_plan) return nullptr;
        switch (op) {
            case modules::MatmulOp::QKV:
                return fwd_plan->qkv.valid ? &fwd_plan->qkv : nullptr;
            case modules::MatmulOp::AttnOut:
                return fwd_plan->out_proj.valid ? &fwd_plan->out_proj : nullptr;
            case modules::MatmulOp::MLPUp:
                return fwd_plan->mlp_up.valid ? &fwd_plan->mlp_up : nullptr;
            case modules::MatmulOp::MLPDown:
                return fwd_plan->mlp_down.valid ? &fwd_plan->mlp_down : nullptr;
            default:
                return nullptr;
        }
    };

    auto block_name = [&](const char* field) {
        return "blocks[" + std::to_string(layer_idx) + "]." + field;
    };

    // Helper to ensure activation buffer is allocated
    auto ensure_act = [&](Tensor& t) {
        if (!t.Data) {
            rs.temp_acquire(t);
        }
    };

    // Residual input to this layer:
    // - Layer 0: input is the encoded embeddings (equivalent to res_ffn[0])
    // - Layer L>0: input is the residual stream for THIS layer (res_ffn[L])
    Tensor& res_in = (layer_idx == 0)
        ? rs.non_block_activations().encoded
        : rs.get_residual(layer_idx, rs.MainStream);

    // LN1 recomputation
    if (flags.ln1) {
        ensure_act(acts.ln1);
        ensure_act(acts.ln1_rstd);
        Tensor& ln1_weight = mWeights.get("blocks[" + std::to_string(layer_idx) + "].ln1_weight");
        rmsnorm_forward(acts.ln1, acts.ln1_rstd, res_in, ln1_weight, nullptr,
                        config.RmsNormEps, Bv, Tv, C, rs.MainStream);
    }

    // QKV + RoPE recomputation
    const bool allow_quant = allow_quant_layer(mOptions, config, layer_idx);
    const bool have_recipe = mOptions.TrainingRecipe != nullptr;

    if (flags.qkv) {
        ensure_act(acts.qkv);
        const std::string qkv_weight_name = "blocks[" + std::to_string(layer_idx) + "].qkv_weight";
        Tensor& qkv_weight = mWeights.get(qkv_weight_name);
        Tensor ln1_flat = view_tensor(acts.ln1, {B * T, C});
        Tensor qkv_flat = view_tensor(acts.qkv, {B * T, qkv_channels});

        // Check for bias
        std::string bias_name = "blocks[" + std::to_string(layer_idx) + "].qkv_bias";
        std::optional<Tensor> qkv_bias;
        const MatmulForwardPlan* qkv_plan = plan_for(modules::MatmulOp::QKV);
        const bool use_bias = qkv_plan ? qkv_plan->has_bias : mWeights.has(bias_name);
        if (use_bias && mWeights.has(bias_name)) {
            qkv_bias = mWeights.get(bias_name);
        }

        const bool use_recipe_qkv = qkv_plan ? (qkv_plan->use_recipe && have_recipe) : have_recipe;
        if (use_recipe_qkv) {
            // Use recipe dispatch (match forward quantization plan)
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
            // BF16 fallback when recipe not available
            int M = static_cast<int>(B * T);
            int N = qkv_channels;
            int K_dim = C;
            matmul(qkv_flat, qkv_weight, ln1_flat, qkv_bias, nullptr, nullptr,
                   rs.CublasLtHandle, rs.CuBlasWorkspace,
                   N, M, K_dim, EMMTranspose::TN, false, rs.MainStream);
        }

        // Re-apply LoRA contributions so recomputed activations match forward.
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

        // QK normalization + RoPE (fused when possible)
        const bool use_qk_norm = (fwd_plan && fwd_plan->attn.valid) ? fwd_plan->attn.use_qk_norm : config.UseQKNorm;
        if (use_qk_norm) {
            ensure_act(acts.q_rstd);
            ensure_act(acts.k_rstd);
            Tensor qkv_view = view_tensor(acts.qkv, {B, T, qkv_channels});
            const bool rope_fused_plan = (fwd_plan && fwd_plan->attn.valid) ? fwd_plan->attn.rope_fused : false;
            if (rope_fused_plan) {
                qkv_qk_norm_rope_forward(qkv_view, acts.q_rstd, acts.k_rstd,
                                         mWeights.get("blocks[" + std::to_string(layer_idx) + "].q_norm_weight"),
                                         mWeights.get("blocks[" + std::to_string(layer_idx) + "].k_norm_weight"),
                                         rs.non_block_activations().freq_cis,
                                         reinterpret_cast<int*>(rs.PositionIDs.Data),
                                         static_cast<float>(config.RmsNormEps),
                                         Bv, Tv, Hq, Hkv, Hs, rs.MainStream);
            } else {
                qkv_head_rmsnorm_forward(qkv_view, acts.q_rstd,
                                         mWeights.get("blocks[" + std::to_string(layer_idx) + "].q_norm_weight"),
                                         static_cast<float>(config.RmsNormEps),
                                         Bv, Tv, qkv_channels, Hq, Hs, 0, rs.MainStream);
                qkv_head_rmsnorm_forward(qkv_view, acts.k_rstd,
                                         mWeights.get("blocks[" + std::to_string(layer_idx) + "].k_norm_weight"),
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
    }

    // Attention recomputation
    if (flags.att) {
        ensure_act(acts.att);
        ensure_act(acts.lse);

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

        // Attention output projection recomputation
        if (flags.out_proj) {
            ensure_act(acts.att_out);
            const std::string out_weight_name = "blocks[" + std::to_string(layer_idx) + "].out_weight";
            Tensor& out_weight = mWeights.get(out_weight_name);
            Tensor att_flat = view_tensor(acts.att, {B * T, Hq * Hs});
            Tensor att_out_flat = view_tensor(acts.att_out, {B * T, C});
            const MatmulForwardPlan* out_plan = plan_for(modules::MatmulOp::AttnOut);
            const bool use_recipe_out = out_plan ? (out_plan->use_recipe && have_recipe) : have_recipe;
            if (use_recipe_out) {
                // Use recipe dispatch (match forward quantization plan)
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
                // BF16 fallback
                int att_dim = Hq * Hs;
                matmul(att_out_flat, out_weight, att_flat, std::nullopt, nullptr, nullptr,
                       rs.CublasLtHandle, rs.CuBlasWorkspace,
                       C, static_cast<int>(B * T), att_dim, EMMTranspose::TN, false, rs.MainStream);
            }
            // Re-apply LoRA o_proj contribution so recomputed att_out matches forward.
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
    }

    // LN2 + residual recomputation
    if (flags.ln2) {
        ensure_act(acts.ln2);
        ensure_act(acts.ln2_rstd);
        Tensor& ln2_weight = mWeights.get("blocks[" + std::to_string(layer_idx) + "].ln2_weight");
        fused_residual_rmsnorm_forward(
            acts.residual_att, acts.ln2, acts.ln2_rstd,
            res_in, acts.att_out, ln2_weight, nullptr,
            config.RmsNormEps, static_cast<int>(B * T), C, rs.MainStream);
    }

    // FFN up projection recomputation
    if (flags.ffn) {
        ensure_act(acts.mlp_up);
        const std::string mlp_up_weight_name = "blocks[" + std::to_string(layer_idx) + "].mlp_up_weight";
        Tensor& mlp_up_weight = mWeights.get(mlp_up_weight_name);
        Tensor ln2_flat = view_tensor(acts.ln2, {B * T, C});
        Tensor mlp_up_flat = view_tensor(acts.mlp_up, {B * T, MUp});
        const MatmulForwardPlan* mlp_plan = plan_for(modules::MatmulOp::MLPUp);
        const bool use_recipe_mlp = mlp_plan ? (mlp_plan->use_recipe && have_recipe) : have_recipe;
        if (use_recipe_mlp) {
            // Use recipe dispatch (match forward quantization plan)
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
            // BF16 fallback
            matmul(mlp_up_flat, mlp_up_weight, ln2_flat, std::nullopt, nullptr, nullptr,
                   rs.CublasLtHandle, rs.CuBlasWorkspace,
                   MUp, static_cast<int>(B * T), C, EMMTranspose::TN, false, rs.MainStream);
        }
        // Re-apply LoRA MLP contributions so recomputed activations match forward.
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
    }

    // SwiGLU recomputation
    if (flags.swiglu) {
        ensure_act(acts.swiglu);
        swiglu_forward(acts.swiglu, acts.mlp_up, nullptr,
                       Bv, Tv, D, rs.MainStream);
    }

    // End graph capture if we were capturing
    if (use_graph && layer_idx < static_cast<int>(mRecomputeGraphs.size()) && mRecomputeGraphs[layer_idx] == nullptr) {
        CUDA_CHECK(cudaStreamEndCapture(rs.MainStream, &captured_graph));
        CUDA_CHECK(cudaGraphInstantiate(&mRecomputeGraphs[layer_idx], captured_graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(captured_graph));
        CUDA_CHECK(cudaGraphLaunch(mRecomputeGraphs[layer_idx], rs.MainStream));
    }
}

}  // namespace dsl
