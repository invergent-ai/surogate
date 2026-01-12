// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_COMPOSITE_TRANSFORMER_BLOCK_H
#define SUROGATE_SRC_MODULES_COMPOSITE_TRANSFORMER_BLOCK_H

#include "config/rope_config.h"
#include "modules/module_base.h"
#include "modules/primitives/attention.h"
#include "modules/primitives/rmsnorm.h"
#include "modules/primitives/linear.h"
#include "modules/primitives/swiglu.h"

namespace modules {

/**
 * @brief Dense Transformer Block
 *
 * Implements a standard pre-norm transformer block:
 *   x = x + Attention(RMSNorm(x))
 *   x = x + MLP(RMSNorm(x))
 *
 * Where MLP is: Linear (gate+up) -> SwiGLU -> Linear (down)
 *
 * Template parameters allow swapping components at compile time for
 * different architectures without runtime overhead:
 * - AttentionType: Different attention implementations (MHA, GQA, etc.)
 * - ActivationType: Different activations (SwiGLU, GeGLU, etc.)
 * - NormType: Different normalizations (RMSNorm, LayerNorm, etc.)
 */
template<
    typename AttentionType = AttentionModule,
    typename ActivationType = SwiGLUModule,
    typename NormType = FusedResidualRMSNormModule
>
class DenseTransformerBlock : public ModuleBase<DenseTransformerBlock<AttentionType, ActivationType, NormType>> {
public:
    /**
     * @brief Configuration for transformer block
     */
    struct Config {
        // Attention config
        int hidden_size;
        int num_query_heads;
        int num_kv_heads;
        int head_size;          // Usually hidden_size / num_query_heads
        RoPEConfig rope;        // Flexible RoPE configuration
        int max_seq_len;        // Maximum sequence length for RoPE
        bool use_qkv_bias = false;
        bool use_qk_norm = false;

        // MLP config
        int intermediate_size;

        // Norm config
        float rms_norm_eps = 1e-5f;

        // Derived configs for sub-modules
        [[nodiscard]] typename AttentionType::Config attention_config() const {
            typename AttentionType::Config cfg;
            cfg.hidden_size = hidden_size;
            cfg.num_query_heads = num_query_heads;
            cfg.num_kv_heads = num_kv_heads;
            cfg.rope = rope;
            cfg.use_qkv_bias = use_qkv_bias;
            cfg.use_qk_norm = use_qk_norm;
            cfg.qk_norm_eps = rms_norm_eps;  // Use same epsilon as layer norm
            cfg.head_size = head_size;
            return cfg;
        }

        [[nodiscard]] typename NormType::Config norm_config() const {
            return {
                .hidden_size = hidden_size,
                .epsilon = rms_norm_eps
            };
        }

        [[nodiscard]] typename ActivationType::Config activation_config() const {
            return {.intermediate_size = intermediate_size};
        }
    };

    /**
     * @brief All weights for the transformer block
     */
    struct Weights {
        // Pre-attention norm
        typename NormType::Weights ln1;

        // Attention
        typename AttentionType::Weights attention;

        // Pre-MLP norm
        typename NormType::Weights ln2;

        // MLP
        Tensor mlp_up_weight;       ///< (2 * intermediate_size, hidden_size) gate+up fused
        Tensor mlp_down_weight;     ///< (hidden_size, intermediate_size)
    };

    /**
     * @brief Saved activations for backward pass
     */
    struct Activations {
        // Norm 1
        typename NormType::Activations ln1_acts;

        // Attention
        typename AttentionType::Activations attention_acts;
        Tensor attention_output;    ///< Output of attention (before residual add)

        // Residual after attention
        Tensor residual_att;        ///< residual + attention_output

        // Norm 2
        typename NormType::Activations ln2_acts;

        // MLP
        QuantizableTensor mlp_up_input;     ///< LN2 output
        Tensor mlp_up_output;               ///< Gate+up projection output
        typename ActivationType::Activations activation_acts;
        QuantizableTensor mlp_down_input;   ///< SwiGLU output
        Tensor mlp_down_output;             ///< MLP output (before residual add)
    };

    /**
     * @brief Weight gradients
     */
    struct Gradients {
        typename NormType::Gradients ln1_grads;
        typename AttentionType::Gradients attention_grads;
        typename NormType::Gradients ln2_grads;
        Tensor d_mlp_up_weight;
        Tensor d_mlp_down_weight;
    };

    explicit DenseTransformerBlock(Config config)
        : mConfig(config),
          mAttention(config.attention_config()),
          mActivation(config.activation_config()) {}

    /**
     * @brief Forward pass through the transformer block
     *
     * @param ctx Module context
     * @param w Block weights
     * @param residual Input residual stream (modified in-place)
     * @param acts Activation storage
     * @return Updated residual stream
     */
    Tensor forward_impl(ModuleContext& ctx, Weights& w, Tensor& residual, Activations& acts);

    /**
     * @brief Backward pass through the transformer block
     *
     * @param ctx Module context
     * @param w Block weights
     * @param acts Saved activations
     * @param grad_residual Gradient w.r.t. output residual
     * @param grads Gradient storage
     * @param accumulate If true, accumulate gradients
     * @return Gradient w.r.t. input residual
     */
    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         Tensor& grad_residual, Gradients& grads, bool accumulate = false);

    /**
     * @brief Recompute activations for gradient checkpointing
     */
    void recompute_impl(ModuleContext& ctx, Weights& w, Tensor& residual, Activations& acts);

    // Accessors
    [[nodiscard]] const Config& config() const { return mConfig; }

private:
    Config mConfig;
    AttentionType mAttention;
    ActivationType mActivation;

    // Helper for MLP forward
    void forward_mlp(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts);

    // Helper for MLP backward
    Tensor backward_mlp(ModuleContext& ctx, Weights& w, Activations& acts,
                        Tensor& grad_output, Gradients& grads, bool accumulate);
};

// ============================================================================
// Implementation
// ============================================================================

template<typename Att, typename Act, typename Norm>
Tensor DenseTransformerBlock<Att, Act, Norm>::forward_impl(
    ModuleContext& ctx, Weights& w, Tensor& residual, Activations& acts) {

    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;

    // ========== Attention Sub-Block ==========
    // x = x + Attention(RMSNorm(x))

    // 1) Pre-attention LayerNorm (fused with residual for second block onwards)
    //    For first application: just normalize the input
    //    Output: acts.ln1_acts.output
    Tensor ln1_output;
    if constexpr (std::is_same_v<Norm, FusedResidualRMSNormModule>) {
        // First block doesn't have previous residual to fuse
        // Use standalone norm or handle in caller
        RMSNormModule standalone_ln1({mConfig.hidden_size, mConfig.rms_norm_eps});
        RMSNormModule::Activations standalone_acts;
        RMSNormModule::Weights standalone_weights{w.ln1.weight};
        ln1_output = standalone_ln1.forward(ctx, standalone_weights, residual, standalone_acts);
        acts.ln1_acts.output.Value = ln1_output;
        acts.ln1_acts.rstd = standalone_acts.rstd;
    } else {
        Norm norm_module(typename Norm::Config{mConfig.hidden_size, mConfig.rms_norm_eps});
        ln1_output = norm_module.forward(ctx, w.ln1, residual, acts.ln1_acts);
    }

    // 2) Attention
    acts.attention_output = mAttention.forward(ctx, w.attention, ln1_output, acts.attention_acts);

    // 3) Residual connection (residual = residual + attention_output)
    //    This is handled by the fused norm in the next step

    // ========== MLP Sub-Block ==========
    // x = x + MLP(RMSNorm(x))

    // 4) Pre-MLP LayerNorm with fused residual add
    //    residual_att = residual + attention_output
    //    ln2_output = RMSNorm(residual_att)
    if constexpr (std::is_same_v<Norm, FusedResidualRMSNormModule>) {
        FusedResidualRMSNormModule ln2({mConfig.hidden_size, mConfig.rms_norm_eps});
        ln2.forward_with_residual(ctx, w.ln2, acts.attention_output, residual, acts.ln2_acts);
        acts.residual_att = acts.ln2_acts.residual_out;
    } else {
        // Manual residual add + norm
        // residual += attention_output
        // ln2_output = norm(residual)
        acts.residual_att = residual;  // Would need actual add kernel
    }

    // 5) MLP: gate+up -> activation -> down
    forward_mlp(ctx, w, acts.ln2_acts.output.Value, acts);

    // 6) Final residual connection
    //    output_residual = residual_att + mlp_down_output
    //    This will be fused with the next block's LN1
    //    For the last block, it's done explicitly

    return acts.mlp_down_output;  // Caller adds to residual
}

template<typename Att, typename Act, typename Norm>
void DenseTransformerBlock<Att, Act, Norm>::forward_mlp(
    ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {

    const int BT = ctx.B * ctx.T;
    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;

    // Save input for backward
    acts.mlp_up_input.Value = input;

    // Gate + Up projection (fused)
    bool needs_quant = w.mlp_up_weight.DType != input.DType;
    if (needs_quant && acts.mlp_up_input.Quant.has_value()) {
        quantize_with_abs_max(
            acts.mlp_up_input.Quant.value(),
            acts.mlp_up_input.Quant->scale(),
            input,
            acts.mlp_up_input.Quant->abs_max(),
            BT * C,
            *ctx.device_prop,
            ctx.stream
        );
    }

    const Tensor& inp = acts.mlp_up_input.for_matmul();
    matmul(
        acts.mlp_up_output, w.mlp_up_weight, inp, std::nullopt,
        w.mlp_up_weight.scale(), acts.mlp_up_input.scale(),
        ctx.cublas_handle, *ctx.workspace,
        2 * D, BT, C, EMMTranspose::TN, false,
        ctx.stream
    );

    // SwiGLU activation
    typename Act::Weights act_weights{};
    mActivation.forward(ctx, act_weights, acts.mlp_up_output, acts.activation_acts);

    // Save activation output for backward
    acts.mlp_down_input.Value = acts.activation_acts.output.Value;

    // Down projection
    needs_quant = w.mlp_down_weight.DType != acts.mlp_down_input.Value.DType;
    if (needs_quant && acts.mlp_down_input.Quant.has_value()) {
        quantize_with_abs_max(
            acts.mlp_down_input.Quant.value(),
            acts.mlp_down_input.Quant->scale(),
            acts.mlp_down_input.Value,
            acts.mlp_down_input.Quant->abs_max(),
            BT * D,
            *ctx.device_prop,
            ctx.stream
        );
    }

    const Tensor& down_inp = acts.mlp_down_input.for_matmul();
    matmul(
        acts.mlp_down_output, w.mlp_down_weight, down_inp, std::nullopt,
        w.mlp_down_weight.scale(), acts.mlp_down_input.scale(),
        ctx.cublas_handle, *ctx.workspace,
        C, BT, D, EMMTranspose::TN, false,
        ctx.stream
    );
}

template<typename Att, typename Act, typename Norm>
Tensor DenseTransformerBlock<Att, Act, Norm>::backward_impl(
    ModuleContext& ctx, Weights& w, Activations& acts,
    Tensor& grad_residual, Gradients& grads, bool accumulate) {

    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;

    // ========== MLP Backward ==========
    // grad_residual contains gradient from next block

    // Backward through MLP down projection
    Tensor d_swiglu = backward_mlp(ctx, w, acts, grad_residual, grads, accumulate);

    // Backward through LN2 (accumulates into grad_residual_att)
    Tensor grad_residual_att;
    if constexpr (std::is_same_v<Norm, FusedResidualRMSNormModule>) {
        FusedResidualRMSNormModule ln2({mConfig.hidden_size, mConfig.rms_norm_eps});
        // d_ln2_input is returned, grad_residual_att is accumulated
        Tensor d_ln2_input = ln2.backward_with_residual(
            ctx, w.ln2, acts.ln2_acts,
            d_swiglu, grad_residual_att, grads.ln2_grads
        );
    }

    // ========== Attention Backward ==========

    // Backward through attention output projection and attention mechanism
    Tensor d_ln1 = mAttention.backward(ctx, w.attention, acts.attention_acts,
                                        grad_residual_att, grads.attention_grads, accumulate);

    // Backward through LN1
    // This produces gradient w.r.t. input residual
    Tensor grad_input_residual;
    if constexpr (std::is_same_v<Norm, FusedResidualRMSNormModule>) {
        FusedResidualRMSNormModule ln1({mConfig.hidden_size, mConfig.rms_norm_eps});
        grad_input_residual = ln1.backward_with_residual(
            ctx, w.ln1, acts.ln1_acts,
            d_ln1, grad_residual_att, grads.ln1_grads
        );
    } else {
        RMSNormModule ln1({mConfig.hidden_size, mConfig.rms_norm_eps});
        RMSNormModule::Activations ln1_acts_simple;
        ln1_acts_simple.rstd = acts.ln1_acts.rstd;
        ln1_acts_simple.input = acts.ln1_acts.residual_out;
        ln1_acts_simple.output = acts.ln1_acts.output;
        RMSNormModule::Weights ln1_weights{w.ln1.weight};
        RMSNormModule::Gradients ln1_grads{grads.ln1_grads.d_weight, grads.ln1_grads.scratch};
        grad_input_residual = ln1.backward(ctx, ln1_weights, ln1_acts_simple, d_ln1, ln1_grads, accumulate);
    }

    return grad_input_residual;
}

template<typename Att, typename Act, typename Norm>
Tensor DenseTransformerBlock<Att, Act, Norm>::backward_mlp(
    ModuleContext& ctx, Weights& w, Activations& acts,
    Tensor& grad_output, Gradients& grads, bool accumulate) {

    const int BT = ctx.B * ctx.T;
    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;

    // Backward through down projection
    Tensor d_swiglu;
    d_swiglu.DType = acts.mlp_down_input.Value.DType;

    // d_swiglu = grad_output @ mlp_down_weight (NN transpose)
    matmul(
        d_swiglu, w.mlp_down_weight, grad_output, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        D, BT, C, EMMTranspose::NN, false,
        ctx.stream
    );

    // d_mlp_down_weight = swiglu_output^T @ grad_output (NT transpose)
    matmul(
        grads.d_mlp_down_weight, acts.mlp_down_input.Value, grad_output, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        D, C, BT, EMMTranspose::NT, accumulate,
        ctx.stream
    );

    // Backward through SwiGLU
    Tensor d_mlp_up;
    d_mlp_up.DType = acts.mlp_up_output.DType;

    typename Act::Weights act_weights{};
    typename Act::Gradients act_grads{};
    d_mlp_up = mActivation.backward(ctx, act_weights, acts.activation_acts,
                                     d_swiglu, act_grads, false);

    // Backward through up projection
    Tensor d_ln2;
    d_ln2.DType = acts.mlp_up_input.Value.DType;

    // d_ln2 = d_mlp_up @ mlp_up_weight (NN transpose)
    matmul(
        d_ln2, w.mlp_up_weight, d_mlp_up, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        C, BT, 2 * D, EMMTranspose::NN, false,
        ctx.stream
    );

    // d_mlp_up_weight = ln2_output^T @ d_mlp_up (NT transpose)
    matmul(
        grads.d_mlp_up_weight, acts.mlp_up_input.Value, d_mlp_up, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        C, 2 * D, BT, EMMTranspose::NT, accumulate,
        ctx.stream
    );

    return d_ln2;
}

template<typename Att, typename Act, typename Norm>
void DenseTransformerBlock<Att, Act, Norm>::recompute_impl(
    ModuleContext& ctx, Weights& w, Tensor& residual, Activations& acts) {

    // Recompute LN1
    RMSNormModule ln1({mConfig.hidden_size, mConfig.rms_norm_eps});
    RMSNormModule::Activations ln1_simple;
    RMSNormModule::Weights ln1_weights{w.ln1.weight};
    Tensor ln1_output = ln1.forward(ctx, ln1_weights, residual, ln1_simple);
    acts.ln1_acts.rstd = ln1_simple.rstd;

    // Recompute attention (preserves LSE from forward)
    mAttention.recompute(ctx, w.attention, ln1_output, acts.attention_acts);

    // Recompute LN2 with residual
    if constexpr (std::is_same_v<Norm, FusedResidualRMSNormModule>) {
        FusedResidualRMSNormModule ln2({mConfig.hidden_size, mConfig.rms_norm_eps});
        ln2.forward_with_residual(ctx, w.ln2, acts.attention_output, residual, acts.ln2_acts);
    }

    // Recompute MLP (optional - depends on recomputation policy)
    forward_mlp(ctx, w, acts.ln2_acts.output.Value, acts);
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_COMPOSITE_TRANSFORMER_BLOCK_H
