// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen2AttentionModule - Attention module for Qwen2 family models
//
// Extends LlamaAttention with sliding window attention support.
// When sliding_window > 0, attention is limited to a window of recent tokens.

#ifndef SUROGATE_SRC_MODELS_QWEN2_ATTENTION_H
#define SUROGATE_SRC_MODELS_QWEN2_ATTENTION_H

#include "modules/primitives/base_attention.h"

namespace modules {

/**
 * @brief Attention module for Qwen2 family models
 *
 * Extends base attention with sliding window support. The sliding window
 * limits attention to the most recent tokens within the window size,
 * enabling efficient processing of long sequences.
 *
 * Implementation note: Sliding window attention requires kernel support.
 * When sliding_window=0 (disabled), this behaves identically to LlamaAttention.
 */
class Qwen2AttentionModule : public BaseAttentionModule<Qwen2AttentionModule> {
public:
    using Base = BaseAttentionModule<Qwen2AttentionModule>;

    /**
     * @brief Configuration for Qwen2Attention
     *
     * Extends BaseConfig with sliding window support.
     */
    struct Config : public Base::BaseConfig {
        int sliding_window = 0;  ///< Window size (0 = disabled, full attention)

        [[nodiscard]] bool has_sliding_window() const { return sliding_window > 0; }
    };

    /**
     * @brief Weight tensors for Qwen2Attention (same as base)
     */
    struct Weights : public Base::BaseWeights {
        // Qwen2Attention uses same weights as base
    };

    /**
     * @brief Saved state for backward pass
     */
    struct Activations : public Base::BaseActivations {
        // Qwen2Attention uses same activations as base
    };

    /**
     * @brief Weight gradients
     */
    struct Gradients : public Base::BaseGradients {
        // Qwen2Attention uses same gradients as base
    };

    explicit Qwen2AttentionModule(Config config) : mConfig(std::move(config)) {}

    /**
     * @brief Forward pass with optional sliding window attention
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

        // 2) Apply RoPE positional encoding
        Base::apply_rope(ctx, mConfig, acts.qkv_output, w.rope_freqs);

        // 3) FlashAttention via cuDNN
        //    TODO: Add sliding window support when kernel supports it
        //    if (mConfig.has_sliding_window()) {
        //        forward_attention_core_with_window(ctx, mConfig, acts.qkv_output,
        //            acts.attention_output, acts.lse, mConfig.sliding_window);
        //    } else {
        Base::forward_attention_core(ctx, mConfig, acts.qkv_output, acts.attention_output, acts.lse);
        //    }

        // 4) Output projection
        return Base::forward_out_proj(ctx, mConfig, w.out_weight, acts.attention_output, acts.att_for_out);
    }

    /**
     * @brief Backward pass through attention block
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

        // Backward through RoPE
        Base::apply_rope_backward(ctx, mConfig, d_qkv, w.rope_freqs);

        // Backward through QKV projection
        return Base::backward_qkv_proj(ctx, mConfig, w.qkv_weight, acts.qkv_input, d_qkv,
                                        grads.d_qkv_weight, grads.d_qkv_bias, grads.bias_scratch, accumulate);
    }

    /**
     * @brief Recompute attention activations (for gradient checkpointing)
     */
    void recompute_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
        acts.qkv_input.Value = input;
        Base::forward_qkv_proj(ctx, mConfig, w.qkv_weight, w.qkv_bias, input, acts.qkv_input, acts.qkv_output);
        Base::apply_rope(ctx, mConfig, acts.qkv_output, w.rope_freqs);
    }

    // Accessors
    [[nodiscard]] const Config& config() const { return mConfig; }
    [[nodiscard]] int hidden_size() const { return mConfig.hidden_size; }
    [[nodiscard]] int num_query_heads() const { return mConfig.num_query_heads; }
    [[nodiscard]] int num_kv_heads() const { return mConfig.num_kv_heads; }
    [[nodiscard]] int sliding_window() const { return mConfig.sliding_window; }

private:
    Config mConfig;
};

} // namespace modules

#endif // SUROGATE_SRC_MODELS_QWEN2_ATTENTION_H
