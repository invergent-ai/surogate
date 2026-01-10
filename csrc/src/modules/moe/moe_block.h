// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_MOE_MOE_BLOCK_H
#define SUROGATE_SRC_MODULES_MOE_MOE_BLOCK_H

#include "expert.h"
#include "router.h"
#include "config/rope_config.h"
#include "modules/module_base.h"
#include "modules/primitives/attention.h"
#include "modules/primitives/rmsnorm.h"

namespace modules {

/**
 * @brief Mixture of Experts Transformer Block
 *
 * A transformer block that replaces the dense FFN with a Mixture of Experts layer.
 * Structure:
 *   input -> LN1 -> Attention -> residual -> LN2 -> Router -> Experts -> residual -> output
 *
 * Supports various MoE configurations:
 * - Standard top-k routing (Mixtral, GShard)
 * - Switch routing (top-1)
 * - Expert choice routing
 * - Shared expert (Nemotron, DeepSeek)
 *
 * @tparam AttentionType The attention module type (default: AttentionModule)
 * @tparam RouterType The router module type (default: RouterModule)
 * @tparam NormType The normalization module type (default: FusedResidualRMSNormModule)
 */
template<typename AttentionType = AttentionModule,
         typename RouterType = RouterModule,
         typename NormType = FusedResidualRMSNormModule>
class MoETransformerBlock : public ModuleBase<MoETransformerBlock<AttentionType, RouterType, NormType>> {
public:
    /**
     * @brief Configuration for MoE transformer block
     */
    struct Config {
        // Attention configuration
        int hidden_size;
        int num_query_heads;
        int num_kv_heads;
        int head_size;
        float rms_norm_eps = 1e-5f;
        RoPEConfig rope;  // Flexible RoPE configuration
        int max_seq_len = 2048;
        bool use_qkv_bias = false;

        // MoE configuration
        int num_experts = 8;
        int top_k = 2;
        int intermediate_size;          ///< Per-expert intermediate size
        float aux_loss_coef = 0.01f;
        float capacity_factor = 1.25f;
        bool use_shared_expert = false; ///< Add shared expert (Nemotron/DeepSeek style)
        int shared_expert_intermediate = 0;  ///< Shared expert size (0 = same as regular)
    };

    /**
     * @brief Weight tensors for the MoE block
     */
    struct Weights {
        // Normalization
        typename NormType::Weights ln1;
        typename NormType::Weights ln2;

        // Attention
        typename AttentionType::Weights attention;

        // Router
        typename RouterType::Weights router;

        // Experts
        ExpertGroupModule::Weights experts;

        // Shared expert (optional)
        std::optional<ExpertModule::Weights> shared_expert;
    };

    /**
     * @brief Saved activations for backward pass
     */
    struct Activations {
        // Normalization
        typename NormType::Activations ln1;
        typename NormType::Activations ln2;

        // Attention
        typename AttentionType::Activations attention;

        // Router
        typename RouterType::Activations router;
        typename RouterType::RouterOutput routing;

        // Experts
        ExpertGroupModule::Activations experts;

        // Shared expert (optional)
        std::optional<ExpertModule::Activations> shared_expert;

        // Residual connections
        Tensor residual_att;        ///< After attention residual
        Tensor moe_output;          ///< Combined expert output
    };

    /**
     * @brief Weight gradients
     */
    struct Gradients {
        typename NormType::Gradients ln1;
        typename NormType::Gradients ln2;
        typename AttentionType::Gradients attention;
        typename RouterType::Gradients router;
        ExpertGroupModule::Gradients experts;
        std::optional<ExpertModule::Gradients> shared_expert;
    };

    explicit MoETransformerBlock(Config config);

    /**
     * @brief Forward pass through MoE block
     *
     * @param ctx Module context
     * @param w Block weights
     * @param input Input tensor (B, T, hidden_size)
     * @param acts Activation storage
     * @return Output tensor (B, T, hidden_size)
     */
    Tensor forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts);

    /**
     * @brief Backward pass through MoE block
     */
    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         Tensor& grad_output, Gradients& grads, bool accumulate = false);

    /**
     * @brief Get auxiliary loss from router (for training)
     */
    [[nodiscard]] float get_aux_loss(const Activations& acts) const {
        return acts.routing.aux_loss * mConfig.aux_loss_coef;
    }

    [[nodiscard]] const Config& config() const { return mConfig; }

private:
    Config mConfig;

    // Sub-modules
    NormType mLN1;
    NormType mLN2;
    AttentionType mAttention;
    RouterType mRouter;
    ExpertGroupModule mExperts;
    std::optional<SharedExpertModule> mSharedExpert;
};

// ============================================================================
// Implementation
// ============================================================================

template<typename AttentionType, typename RouterType, typename NormType>
MoETransformerBlock<AttentionType, RouterType, NormType>::MoETransformerBlock(Config config)
    : mConfig(config)
    , mLN1({.hidden_size = config.hidden_size, .epsilon = config.rms_norm_eps})
    , mLN2({.hidden_size = config.hidden_size, .epsilon = config.rms_norm_eps})
    , mAttention({
        .hidden_size = config.hidden_size,
        .num_query_heads = config.num_query_heads,
        .num_kv_heads = config.num_kv_heads,
        .rope = config.rope,
        .use_qkv_bias = config.use_qkv_bias
      })
    , mRouter({
        .hidden_size = config.hidden_size,
        .num_experts = config.num_experts,
        .top_k = config.top_k,
        .aux_loss_coef = config.aux_loss_coef,
        .capacity_factor = config.capacity_factor
      })
    , mExperts({
        .num_experts = config.num_experts,
        .hidden_size = config.hidden_size,
        .intermediate_size = config.intermediate_size,
        .top_k = config.top_k,
        .capacity_factor = static_cast<int>(config.capacity_factor),
        .use_gated = true
      }) {

    // Initialize shared expert if configured
    if (config.use_shared_expert) {
        int shared_size = config.shared_expert_intermediate > 0 ?
                          config.shared_expert_intermediate : config.intermediate_size;
        SharedExpertModule::Config shared_config;
        shared_config.hidden_size = config.hidden_size;
        shared_config.intermediate_size = shared_size;
        shared_config.use_gated = true;
        mSharedExpert.emplace(shared_config);
    }
}

template<typename AttentionType, typename RouterType, typename NormType>
Tensor MoETransformerBlock<AttentionType, RouterType, NormType>::forward_impl(
    ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {

    const int B = ctx.B;
    const int T = ctx.T;
    const int C = mConfig.hidden_size;
    const long N = static_cast<long>(B) * T * C;

    // ========================================================================
    // Attention block: LN1 -> Attention -> Residual
    // ========================================================================

    // Layer norm 1 (standalone for first application)
    Tensor ln1_out;
    if constexpr (std::is_same_v<NormType, FusedResidualRMSNormModule>) {
        RMSNormModule standalone_ln1({mConfig.hidden_size, mConfig.rms_norm_eps});
        RMSNormModule::Activations standalone_acts;
        RMSNormModule::Weights standalone_weights{w.ln1.weight};
        ln1_out = standalone_ln1.forward(ctx, standalone_weights, input, standalone_acts);
        acts.ln1.output.Value = ln1_out;
        acts.ln1.rstd = standalone_acts.rstd;
    } else {
        ln1_out = mLN1.forward(ctx, w.ln1, input, acts.ln1);
    }

    // Attention
    Tensor att_out = mAttention.forward(ctx, w.attention, ln1_out, acts.attention);

    // ========================================================================
    // MoE block: LN2 (with fused residual) -> Router -> Experts -> Residual
    // ========================================================================

    // Pre-MoE LayerNorm with fused residual add:
    // residual_att = input + att_out
    // ln2_out = RMSNorm(residual_att)
    Tensor ln2_out;
    if constexpr (std::is_same_v<NormType, FusedResidualRMSNormModule>) {
        FusedResidualRMSNormModule ln2({mConfig.hidden_size, mConfig.rms_norm_eps});
        ln2.forward_with_residual(ctx, w.ln2, att_out, input, acts.ln2);
        acts.residual_att = acts.ln2.residual_out;
        ln2_out = acts.ln2.output.Value;
    } else {
        // Manual residual add + norm (seed=0 for deterministic stochastic rounding)
        vector_add_sr(acts.residual_att, input, att_out, 1.0f, N, /*seed=*/0, ctx.stream);
        ln2_out = mLN2.forward(ctx, w.ln2, acts.residual_att, acts.ln2);
    }

    // Router: compute routing decisions
    // Reshape ln2_out to (B*T, C) for routing
    Tensor flat_input = ln2_out;
    flat_input.Sizes[0] = B * T;
    flat_input.Sizes[1] = C;
    flat_input.Rank = 2;

    acts.routing = mRouter.forward(ctx, w.router, flat_input, acts.router);

    // Expert group: dispatch, compute, combine
    Tensor expert_out = mExperts.forward(ctx, w.experts, flat_input, acts.routing, acts.experts);

    // Add shared expert output if configured
    if (mSharedExpert && w.shared_expert.has_value()) {
        Tensor shared_out = mSharedExpert->forward_all(ctx, *w.shared_expert, flat_input,
                                                        *acts.shared_expert);
        // expert_out += shared_out (in-place add, seed=0 for deterministic)
        vector_add_sr(expert_out, expert_out, shared_out, 1.0f, B * T * C, /*seed=*/0, ctx.stream);
    }

    // Reshape back to (B, T, C)
    acts.moe_output = expert_out;
    acts.moe_output.Sizes[0] = B;
    acts.moe_output.Sizes[1] = T;
    acts.moe_output.Sizes[2] = C;
    acts.moe_output.Rank = 3;

    // Second residual connection: output = residual_att + moe_output
    // This will be handled by the next block's fused LN1 (or done explicitly for last block)
    // Return the MoE output - caller handles final residual
    return acts.moe_output;
}

template<typename AttentionType, typename RouterType, typename NormType>
Tensor MoETransformerBlock<AttentionType, RouterType, NormType>::backward_impl(
    ModuleContext& ctx, Weights& w, Activations& acts,
    Tensor& grad_output, Gradients& grads, bool accumulate) {

    const int B = ctx.B;
    const int T = ctx.T;
    const int C = mConfig.hidden_size;

    // ========================================================================
    // Backward through MoE block
    // ========================================================================

    // Backward through second residual
    Tensor d_residual_att = grad_output;  // Gradient flows to both branches
    Tensor d_moe_output = grad_output;

    // Reshape for expert backward
    d_moe_output.Sizes[0] = B * T;
    d_moe_output.Sizes[1] = C;
    d_moe_output.Rank = 2;

    // Backward through shared expert if present
    Tensor d_flat_input_shared;
    if (mSharedExpert && w.shared_expert.has_value()) {
        d_flat_input_shared = mSharedExpert->backward(ctx, *w.shared_expert, *acts.shared_expert,
                                                       d_moe_output, *grads.shared_expert, accumulate);
    }

    // Backward through expert group
    Tensor d_flat_input = mExperts.backward(ctx, w.experts, acts.experts, acts.routing,
                                            d_moe_output, grads.experts, accumulate);

    // Add gradient from shared expert
    if (d_flat_input_shared.Data) {
        // vector_add(d_flat_input, d_flat_input, d_flat_input_shared, B * T * C, ctx.stream);
    }

    // Backward through router
    // Note: Router gradient affects the routing weights, which affects how gradients
    // are distributed to experts. This is complex and requires careful handling.
    Tensor d_router_input = mRouter.backward(ctx, w.router, acts.router,
                                              d_flat_input, grads.router, accumulate);

    // Backward through LN2
    Tensor d_ln2_input = mLN2.backward(ctx, w.ln2, acts.ln2, d_router_input, grads.ln2, accumulate);

    // Add gradient from residual path
    // d_residual_att += d_ln2_input
    // vector_add(d_residual_att, d_residual_att, d_ln2_input, B * T * C, ctx.stream);

    // ========================================================================
    // Backward through Attention block
    // ========================================================================

    // Backward through attention
    Tensor d_att_out = d_residual_att;
    Tensor d_ln1_out = mAttention.backward(ctx, w.attention, acts.attention,
                                            d_att_out, grads.attention, accumulate);

    // Backward through LN1
    Tensor d_input = mLN1.backward(ctx, w.ln1, acts.ln1, d_ln1_out, grads.ln1, accumulate);

    // Add gradient from first residual path
    // d_input += d_residual_att (portion that bypassed attention)

    return d_input;
}

/**
 * @brief Type alias for standard MoE block
 */
using StandardMoEBlock = MoETransformerBlock<AttentionModule, RouterModule, FusedResidualRMSNormModule>;

/**
 * @brief Type alias for Switch Transformer block (top-1 routing)
 */
using SwitchTransformerBlock = MoETransformerBlock<AttentionModule, SwitchRouterModule, FusedResidualRMSNormModule>;

/**
 * @brief MoE block configuration builder
 */
class MoEBlockConfigBuilder {
public:
    MoEBlockConfigBuilder() = default;

    MoEBlockConfigBuilder& hidden_size(int size) { mConfig.hidden_size = size; return *this; }
    MoEBlockConfigBuilder& num_query_heads(int heads) { mConfig.num_query_heads = heads; return *this; }
    MoEBlockConfigBuilder& num_kv_heads(int heads) { mConfig.num_kv_heads = heads; return *this; }
    MoEBlockConfigBuilder& head_size(int size) { mConfig.head_size = size; return *this; }
    MoEBlockConfigBuilder& rms_norm_eps(float eps) { mConfig.rms_norm_eps = eps; return *this; }
    MoEBlockConfigBuilder& rope_config(const RoPEConfig& cfg) { mConfig.rope = cfg; return *this; }
    MoEBlockConfigBuilder& rope_theta(float theta) { mConfig.rope.theta = theta; return *this; }  // Convenience
    MoEBlockConfigBuilder& max_seq_len(int len) { mConfig.max_seq_len = len; return *this; }
    MoEBlockConfigBuilder& use_qkv_bias(bool bias) { mConfig.use_qkv_bias = bias; return *this; }

    MoEBlockConfigBuilder& num_experts(int n) { mConfig.num_experts = n; return *this; }
    MoEBlockConfigBuilder& top_k(int k) { mConfig.top_k = k; return *this; }
    MoEBlockConfigBuilder& intermediate_size(int size) { mConfig.intermediate_size = size; return *this; }
    MoEBlockConfigBuilder& aux_loss_coef(float coef) { mConfig.aux_loss_coef = coef; return *this; }
    MoEBlockConfigBuilder& capacity_factor(float factor) { mConfig.capacity_factor = factor; return *this; }

    MoEBlockConfigBuilder& with_shared_expert(int intermediate_size = 0) {
        mConfig.use_shared_expert = true;
        mConfig.shared_expert_intermediate = intermediate_size;
        return *this;
    }

    typename StandardMoEBlock::Config build() const { return mConfig; }

private:
    typename StandardMoEBlock::Config mConfig;
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MOE_MOE_BLOCK_H
