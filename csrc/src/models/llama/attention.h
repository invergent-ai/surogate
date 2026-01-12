// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// LlamaAttentionModule - Attention module for LLaMA family models
//
// Standard Multi-Head/Grouped-Query Attention with RoPE positional encoding.
// This is the base attention implementation that other models extend.

#ifndef SUROGATE_SRC_MODELS_LLAMA_ATTENTION_H
#define SUROGATE_SRC_MODELS_LLAMA_ATTENTION_H

#include "modules/primitives/base_attention.h"

namespace modules {

/**
 * @brief Attention module for LLaMA family models
 *
 * Implements standard multi-head/grouped-query attention with:
 * - Fused QKV projection
 * - RoPE positional encoding
 * - FlashAttention via cuDNN
 * - Output projection
 *
 * This serves as the foundation for other attention variants:
 * - Qwen2AttentionModule adds sliding window
 * - Qwen3AttentionModule adds QK normalization
 */
class LlamaAttentionModule : public BaseAttentionModule<LlamaAttentionModule> {
public:
    using Base = BaseAttentionModule<LlamaAttentionModule>;

    /**
     * @brief Configuration for LlamaAttention
     *
     * Inherits all fields from BaseConfig.
     */
    struct Config : public Base::BaseConfig {
        // LlamaAttention uses all base fields without additions
    };

    /**
     * @brief Weight tensors for LlamaAttention
     */
    struct Weights : public Base::BaseWeights {
        // LlamaAttention uses all base weights without additions
    };

    /**
     * @brief Saved state for backward pass
     */
    struct Activations : public Base::BaseActivations {
        // LlamaAttention uses all base activations without additions
    };

    /**
     * @brief Weight gradients
     */
    struct Gradients : public Base::BaseGradients {
        // LlamaAttention uses all base gradients without additions
    };

    explicit LlamaAttentionModule(Config config) : mConfig(std::move(config)) {}

    /**
     * @brief Forward pass: QKV projection -> RoPE -> Attention -> Output projection
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
        Base::forward_attention_core(ctx, mConfig, acts.qkv_output, acts.attention_output, acts.lse);

        // 4) Output projection
        return Base::forward_out_proj(ctx, mConfig, w.out_weight, acts.attention_output, acts.att_for_out);
    }

    /**
     * @brief Backward pass through full attention block
     *
     * @param ctx Module context with CUDA resources
     * @param w Weight tensors
     * @param acts Saved activations from forward
     * @param grad_output Gradient w.r.t. output (B*T, hidden_size)
     * @param grads Gradient storage
     * @param accumulate If true, accumulate into existing gradients
     * @return Gradient w.r.t. input (B*T, hidden_size)
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
     *
     * Recomputes QKV projection and RoPE. FlashAttention stats (LSE) must
     * be saved during forward as they're expensive to recompute.
     */
    void recompute_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
        // Recompute QKV projection
        acts.qkv_input.Value = input;
        Base::forward_qkv_proj(ctx, mConfig, w.qkv_weight, w.qkv_bias, input, acts.qkv_input, acts.qkv_output);

        // Recompute RoPE
        Base::apply_rope(ctx, mConfig, acts.qkv_output, w.rope_freqs);

        // Note: LSE must be saved during forward - too expensive to recompute
    }

    // Accessors
    [[nodiscard]] const Config& config() const { return mConfig; }
    [[nodiscard]] int hidden_size() const { return mConfig.hidden_size; }
    [[nodiscard]] int num_query_heads() const { return mConfig.num_query_heads; }
    [[nodiscard]] int num_kv_heads() const { return mConfig.num_kv_heads; }

private:
    Config mConfig;
};

} // namespace modules

#endif // SUROGATE_SRC_MODELS_LLAMA_ATTENTION_H
