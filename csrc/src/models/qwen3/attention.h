// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3AttentionModule - Attention module for Qwen3 family models
//
// Extends attention with QK normalization (per-head RMSNorm applied to Q and K
// before RoPE). This improves training stability for large models.

#ifndef SUROGATE_SRC_MODELS_QWEN3_ATTENTION_H
#define SUROGATE_SRC_MODELS_QWEN3_ATTENTION_H

#include "modules/primitives/base_attention.h"

namespace modules {

/**
 * @brief Attention module for Qwen3 family models
 *
 * Extends base attention with QK normalization:
 * - Per-head RMSNorm is applied to Q and K heads before RoPE
 * - This stabilizes training and allows for larger head dimensions
 *
 * The forward pass is: QKV proj -> QK norm -> RoPE -> Attention -> Out proj
 *
 * QK normalization is applied per-head, normalizing over the head dimension.
 * This is different from standard LayerNorm which normalizes over all features.
 */
class Qwen3AttentionModule : public BaseAttentionModule<Qwen3AttentionModule> {
public:
    using Base = BaseAttentionModule<Qwen3AttentionModule>;

    /**
     * @brief Configuration for Qwen3Attention
     *
     * Uses BaseConfig fields but defaults use_qk_norm to true for Qwen3.
     */
    struct Config : public Base::BaseConfig {
        Config() {
            use_qk_norm = true;  // Qwen3 uses QK norm by default
        }

        [[nodiscard]] bool has_qk_norm() const { return use_qk_norm; }
    };

    /**
     * @brief Weight tensors for Qwen3Attention
     *
     * Extends BaseWeights with QK normalization weights.
     */
    struct Weights : public Base::BaseWeights {
        std::optional<Tensor> q_norm_weight; ///< (head_dim,) per-head RMSNorm weight for Q
        std::optional<Tensor> k_norm_weight; ///< (head_dim,) per-head RMSNorm weight for K
    };

    /**
     * @brief Saved state for backward pass
     *
     * Extends BaseActivations with QK normalization state.
     */
    struct Activations : public Base::BaseActivations {
        Tensor q_rstd;  ///< (B, T, Hq) reciprocal std for Q heads
        Tensor k_rstd;  ///< (B, T, Hkv) reciprocal std for K heads
    };

    /**
     * @brief Weight gradients
     *
     * Extends BaseGradients with QK normalization gradients.
     */
    struct Gradients : public Base::BaseGradients {
        std::optional<Tensor> d_q_norm_weight; ///< (head_dim,)
        std::optional<Tensor> d_k_norm_weight; ///< (head_dim,)
    };

    explicit Qwen3AttentionModule(Config config) : mConfig(std::move(config)) {}

    /**
     * @brief Forward pass: QKV proj -> QK norm -> RoPE -> Attention -> Out proj
     *
     * @param ctx Module context with CUDA resources
     * @param w Weight tensors
     * @param input Normalized input (B*T, hidden_size)
     * @param acts Activation storage for backward
     * @return Output (B*T, hidden_size)
     */
    Tensor forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
        // 1) QKV projection
        Base::forward_qkv_proj(ctx, mConfig, w.qkv_weight, w.qkv_bias, input, acts.qkv_input, acts.qkv_output);

        // 2) Apply QK normalization + RoPE (fused when both are enabled)
        if (mConfig.use_qk_norm) {
            if (!w.q_norm_weight.has_value() || !w.k_norm_weight.has_value()) {
                throw std::runtime_error("Qwen3Attention: QK norm enabled but q_norm_weight/k_norm_weight not provided");
            }
            Base::QKNormConfig qk_config{.use_qk_norm = true, .qk_norm_eps = mConfig.qk_norm_eps};
            Base::apply_qk_norm_rope(ctx, mConfig, qk_config, acts.qkv_output,
                                      w.q_norm_weight.value(), w.k_norm_weight.value(),
                                      w.rope_freqs, acts.q_rstd, acts.k_rstd);
        } else {
            // Fallback to just RoPE (if QK norm disabled for some reason)
            Base::apply_rope(ctx, mConfig, acts.qkv_output, w.rope_freqs);
        }

        // 3) FlashAttention via cuDNN
        Base::forward_attention_core(ctx, mConfig, acts.qkv_output, acts.attention_output, acts.lse);

        // 4) Output projection
        return Base::forward_out_proj(ctx, mConfig, w.out_weight, acts.attention_output, acts.att_for_out);
    }

    /**
     * @brief Backward pass through attention block with QK norm
     */
    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         Tensor& grad_output, Gradients& grads, bool accumulate = false) {
        // Backward through output projection
        Tensor d_att = Base::backward_out_proj(ctx, mConfig, w.out_weight, acts.att_for_out,
                                                grad_output, grads.d_out_weight, accumulate);

        // Backward through FlashAttention
        Tensor d_qkv;
        d_qkv.DType = acts.qkv_output.DType;
        Base::backward_attention_core(ctx, mConfig, d_att, acts.qkv_output,
                                       acts.attention_output, acts.lse, d_qkv);

        // Backward through QK norm + RoPE
        if (mConfig.use_qk_norm) {
            Base::QKNormConfig qk_config{.use_qk_norm = true, .qk_norm_eps = mConfig.qk_norm_eps};
            Base::apply_qk_norm_rope_backward(
                ctx, mConfig, qk_config, d_qkv, acts.qkv_output,
                w.q_norm_weight.value(), w.k_norm_weight.value(),
                w.rope_freqs, acts.q_rstd, acts.k_rstd,
                grads.d_q_norm_weight, grads.d_k_norm_weight, accumulate
            );
        } else {
            // Fallback to just RoPE backward
            Base::apply_rope_backward(ctx, mConfig, d_qkv, w.rope_freqs);
        }

        // Backward through QKV projection
        return Base::backward_qkv_proj(ctx, mConfig, w.qkv_weight, acts.qkv_input, d_qkv,
                                        grads.d_qkv_weight, grads.d_qkv_bias, grads.bias_scratch, accumulate);
    }

    /**
     * @brief Recompute attention activations (for gradient checkpointing)
     */
    void recompute_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
        // Recompute QKV projection
        acts.qkv_input.Value = input;
        Base::forward_qkv_proj(ctx, mConfig, w.qkv_weight, w.qkv_bias, input, acts.qkv_input, acts.qkv_output);

        // Recompute QK norm + RoPE (also recomputes q_rstd/k_rstd needed for backward)
        if (mConfig.use_qk_norm) {
            Base::QKNormConfig qk_config{.use_qk_norm = true, .qk_norm_eps = mConfig.qk_norm_eps};
            Base::apply_qk_norm_rope(ctx, mConfig, qk_config, acts.qkv_output,
                                      w.q_norm_weight.value(), w.k_norm_weight.value(),
                                      w.rope_freqs, acts.q_rstd, acts.k_rstd);
        } else {
            Base::apply_rope(ctx, mConfig, acts.qkv_output, w.rope_freqs);
        }

        // Note: LSE must be saved during forward - too expensive to recompute
    }

    // Accessors
    [[nodiscard]] const Config& config() const { return mConfig; }
    [[nodiscard]] int hidden_size() const { return mConfig.hidden_size; }
    [[nodiscard]] int num_query_heads() const { return mConfig.num_query_heads; }
    [[nodiscard]] int num_kv_heads() const { return mConfig.num_kv_heads; }
    [[nodiscard]] bool use_qk_norm() const { return mConfig.use_qk_norm; }

private:
    Config mConfig;
};

} // namespace modules

#endif // SUROGATE_SRC_MODELS_QWEN3_ATTENTION_H
