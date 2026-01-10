// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_PRIMITIVES_ATTENTION_H
#define SUROGATE_SRC_MODULES_PRIMITIVES_ATTENTION_H

#include "modules/module_base.h"
#include "modules/primitives/linear.h"
#include "config/rope_config.h"
#include "kernels/kernels.h"

namespace modules {

/**
 * @brief Multi-Head/Grouped-Query Attention module
 *
 * Implements the attention mechanism with:
 * - QKV projection (fused)
 * - RoPE positional encoding
 * - FlashAttention (via cuDNN)
 * - Output projection
 *
 * Supports both Multi-Head Attention (MHA) where num_kv_heads == num_query_heads,
 * and Grouped-Query Attention (GQA) where num_kv_heads < num_query_heads.
 *
 * The module is parameterized by the projection types to allow customization
 * (e.g., different quantization strategies for QKV vs output projection).
 */
class AttentionModule : public ModuleBase<AttentionModule> {
public:
    /**
     * @brief Configuration for attention
     */
    struct Config {
        int hidden_size;            ///< Model dimension
        int num_query_heads;        ///< Number of query heads
        int num_kv_heads;           ///< Number of key/value heads (< num_query_heads for GQA)
        RoPEConfig rope;            ///< Flexible RoPE configuration
        bool use_qkv_bias = false;  ///< Whether QKV projection has bias (Qwen uses this)
        int head_size = 0;          ///< Optional explicit head dim (0 => hidden_size / num_query_heads)

        // Derived dimensions
        [[nodiscard]] int head_dim() const { return head_size > 0 ? head_size : (hidden_size / num_query_heads); }
        [[nodiscard]] int attn_out_channels() const { return head_dim() * num_query_heads; }
        [[nodiscard]] int qkv_channels() const {
            return head_dim() * (num_query_heads + 2 * num_kv_heads);
        }
        [[nodiscard]] bool is_gqa() const { return num_kv_heads < num_query_heads; }

        /// Get the number of dimensions that will have RoPE applied
        [[nodiscard]] int rotary_dim() const { return rope.rotary_dim(head_dim()); }

        /// Backwards compatibility: get rope_theta from RoPEConfig
        [[nodiscard]] float rope_theta() const { return rope.theta; }
    };

    /**
     * @brief Weight tensors for attention
     */
    struct Weights {
        Tensor qkv_weight;                  ///< (qkv_channels, hidden_size) fused QKV
        std::optional<Tensor> qkv_bias;     ///< (qkv_channels,) optional QKV bias
        Tensor out_weight;                  ///< (hidden_size, attn_out_channels) output projection
        std::optional<Tensor> q_norm_weight; ///< (head_dim,) optional per-head RMSNorm weight (Qwen3)
        std::optional<Tensor> k_norm_weight; ///< (head_dim,) optional per-head RMSNorm weight (Qwen3)
        Tensor rope_freqs;                  ///< Precomputed RoPE frequencies
    };

    /**
     * @brief Saved state for backward pass
     */
    struct Activations {
        // QKV projection
        QuantizableTensor qkv_input;        ///< Input to QKV projection (LN output)
        Tensor qkv_output;                  ///< QKV after RoPE (B, T, qkv_channels)

        // Attention
        Tensor attention_output;            ///< (B, T, attn_out_channels) attention output
        Tensor lse;                         ///< (B, num_heads, T) log-sum-exp from FlashAttention
        QuantizableTensor att_for_out;      ///< Attention output (possibly quantized for out proj)

        // Output projection saved by LinearModule if needed
        QuantizableTensor out_input;        ///< Input to output projection
    };

    /**
     * @brief Weight gradients
     */
    struct Gradients {
        Tensor d_qkv_weight;                ///< (qkv_channels, hidden_size)
        std::optional<Tensor> d_qkv_bias;   ///< (qkv_channels,) optional
        Tensor d_out_weight;                ///< (hidden_size, attn_out_channels)
        std::optional<Tensor> d_q_norm_weight; ///< (head_dim,) optional
        std::optional<Tensor> d_k_norm_weight; ///< (head_dim,) optional
        std::optional<Tensor> bias_scratch; ///< For bias backward
    };

    explicit AttentionModule(Config config) : mConfig(config) {}

    /**
     * @brief Forward pass: QKV projection -> RoPE -> Attention -> Output projection
     *
     * @param ctx Module context with CUDA resources
     * @param w Weight tensors
     * @param input Normalized input (B*T, hidden_size)
     * @param acts Activation storage for backward
     * @return Output (B*T, hidden_size)
     */
    Tensor forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts);

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
                         Tensor& grad_output, Gradients& grads, bool accumulate = false);

    /**
     * @brief Recompute attention activations (for gradient checkpointing)
     *
     * Recomputes QKV projection and RoPE. FlashAttention stats (LSE) must
     * be saved during forward as they're expensive to recompute.
     */
    void recompute_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts);

    // Accessors
    [[nodiscard]] const Config& config() const { return mConfig; }
    [[nodiscard]] int hidden_size() const { return mConfig.hidden_size; }
    [[nodiscard]] int num_query_heads() const { return mConfig.num_query_heads; }
    [[nodiscard]] int num_kv_heads() const { return mConfig.num_kv_heads; }

    /**
     * @brief Get required workspace size for cuDNN attention
     */
    static std::size_t get_workspace_size(int B, int T, int Hq, int Hkv, int Hs, cudnnHandle_t handle) {
        return cudnn_get_workspace_size(B, T, Hq, Hkv, Hs, handle);
    }

private:
    Config mConfig;

    // Helper for QKV projection forward
    void forward_qkv(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts);

    // Helper for attention output projection forward
    Tensor forward_out_projection(ModuleContext& ctx, Weights& w, Activations& acts);

    // Helper for QKV backward
    Tensor backward_qkv(ModuleContext& ctx, Weights& w, Activations& acts,
                        Tensor& d_qkv, Gradients& grads, bool accumulate);
};

inline Tensor AttentionModule::forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
    const int B = ctx.B;
    const int T = ctx.T;
    const int C = mConfig.hidden_size;
    const int Hq = mConfig.num_query_heads;
    const int Hkv = mConfig.num_kv_heads;
    const int Hs = mConfig.head_dim();

    // Save input for backward
    acts.qkv_input.Value = input;

    // 1) QKV projection
    forward_qkv(ctx, w, input, acts);

    // 2) Apply RoPE to Q and K (if enabled)
    const int rotary_dim = mConfig.rotary_dim();
    if (rotary_dim > 0) {
        rope_forward(
            acts.qkv_output, acts.qkv_output,
            w.rope_freqs,
            ctx.position_ids,
            nullptr,  // abs_max for quantization
            B, T, Hq, Hkv, Hs,
            rotary_dim,  // partial RoPE support
            ctx.stream
        );
    }

    // 3) FlashAttention via cuDNN
    attention_forward_cudnn(
        acts.attention_output,
        acts.lse,
        acts.qkv_output,
        *ctx.workspace,
        ctx.cudnn_handle,
        B, T, Hq, Hkv, Hs,
        ctx.stream
    );

    // Save attention output for backward (possibly quantized for out projection)
    acts.att_for_out.Value = acts.attention_output;

    // 4) Output projection
    return forward_out_projection(ctx, w, acts);
}

inline void AttentionModule::forward_qkv(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
    const int BT = ctx.B * ctx.T;
    const int C = mConfig.hidden_size;
    const int OC = mConfig.qkv_channels();

    // Determine if quantization is needed
    bool needs_quant = w.qkv_weight.DType != input.DType;

    if (needs_quant && acts.qkv_input.Quant.has_value()) {
        quantize_with_abs_max(
            acts.qkv_input.Quant.value(),
            acts.qkv_input.Quant->scale(),
            input,
            acts.qkv_input.Quant->abs_max(),
            BT * C,
            *ctx.device_prop,
            ctx.stream
        );
    }

    const Tensor& inp = acts.qkv_input.for_matmul();
    float* scale_a = w.qkv_weight.scale();
    float* scale_b = acts.qkv_input.scale();

    matmul(
        acts.qkv_output, w.qkv_weight, inp, w.qkv_bias,
        scale_a, scale_b,
        ctx.cublas_handle, *ctx.workspace,
        OC, BT, C, EMMTranspose::TN, false,
        ctx.stream
    );
}

inline Tensor AttentionModule::forward_out_projection(ModuleContext& ctx, Weights& w, Activations& acts) {
    const int BT = ctx.B * ctx.T;
    const int C = mConfig.hidden_size;
    const int InC = mConfig.attn_out_channels();

    Tensor output;
    output.DType = acts.attention_output.DType;

    // Determine if quantization is needed
    bool needs_quant = w.out_weight.DType != acts.attention_output.DType;

    if (needs_quant && acts.att_for_out.Quant.has_value()) {
        quantize_with_abs_max(
            acts.att_for_out.Quant.value(),
            acts.att_for_out.Quant->scale(),
            acts.attention_output,
            acts.att_for_out.Quant->abs_max(),
            BT * InC,
            *ctx.device_prop,
            ctx.stream
        );
    }

    const Tensor& inp = acts.att_for_out.for_matmul();
    float* scale_a = w.out_weight.scale();
    float* scale_b = acts.att_for_out.scale();

    matmul(
        output, w.out_weight, inp, std::nullopt,
        scale_a, scale_b,
        ctx.cublas_handle, *ctx.workspace,
        C, BT, InC, EMMTranspose::TN, false,
        ctx.stream
    );

    return output;
}

inline Tensor AttentionModule::backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                                             Tensor& grad_output, Gradients& grads, bool accumulate) {
    const int B = ctx.B;
    const int T = ctx.T;
    const int C = mConfig.hidden_size;
    const int Hq = mConfig.num_query_heads;
    const int Hkv = mConfig.num_kv_heads;
    const int Hs = mConfig.head_dim();
    const int InC = mConfig.attn_out_channels();

    // Backward through output projection
    Tensor d_att;  // Gradient w.r.t. attention output
    d_att.DType = acts.attention_output.DType;

    // d_att = d_output @ out_weight (NN transpose)
    matmul(
        d_att, w.out_weight, grad_output, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        InC, B*T, C, EMMTranspose::NN, false,
        ctx.stream
    );

    // d_out_weight = att^T @ d_output (NT transpose)
    matmul(
        grads.d_out_weight, acts.att_for_out.Value, grad_output, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        InC, C, B*T, EMMTranspose::NT, accumulate,
        ctx.stream
    );

    // Backward through FlashAttention
    Tensor d_qkv;
    d_qkv.DType = acts.qkv_output.DType;

    attention_backward_cudnn(
        d_qkv,
        acts.lse,
        acts.attention_output,
        d_att,
        acts.qkv_output,
        *ctx.workspace,
        ctx.cudnn_handle,
        B, T, Hq, Hkv, Hs,
        ctx.stream
    );

    // Backward through RoPE (if enabled)
    const int rotary_dim = mConfig.rotary_dim();
    if (rotary_dim > 0) {
        rope_backward(
            d_qkv, d_qkv,
            w.rope_freqs,
            ctx.position_ids,
            nullptr,  // abs_max
            B, T, Hq, Hkv, Hs,
            rotary_dim,  // partial RoPE support
            ctx.stream
        );
    }

    // Backward through QKV projection
    return backward_qkv(ctx, w, acts, d_qkv, grads, accumulate);
}

inline Tensor AttentionModule::backward_qkv(ModuleContext& ctx, Weights& w, Activations& acts,
                                             Tensor& d_qkv, Gradients& grads, bool accumulate) {
    const int BT = ctx.B * ctx.T;
    const int C = mConfig.hidden_size;
    const int OC = mConfig.qkv_channels();

    Tensor d_input;
    d_input.DType = acts.qkv_input.Value.DType;

    // d_input = d_qkv @ qkv_weight (NN transpose)
    matmul(
        d_input, w.qkv_weight, d_qkv, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        C, BT, OC, EMMTranspose::NN, false,
        ctx.stream
    );

    // d_qkv_weight = input^T @ d_qkv (NT transpose)
    matmul(
        grads.d_qkv_weight, acts.qkv_input.Value, d_qkv, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        C, OC, BT, EMMTranspose::NT, accumulate,
        ctx.stream
    );

    // d_qkv_bias if needed
    if (grads.d_qkv_bias.has_value() && grads.bias_scratch.has_value()) {
        backward_bias(
            grads.d_qkv_bias.value(), d_qkv, nullptr, nullptr,
            grads.bias_scratch.value(), ctx.B, ctx.T, OC,
            *ctx.device_prop, ctx.stream
        );
    }

    return d_input;
}

inline void AttentionModule::recompute_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
    const int B = ctx.B;
    const int T = ctx.T;
    const int Hq = mConfig.num_query_heads;
    const int Hkv = mConfig.num_kv_heads;
    const int Hs = mConfig.head_dim();

    // Recompute QKV projection
    acts.qkv_input.Value = input;
    forward_qkv(ctx, w, input, acts);

    // Recompute RoPE (if enabled)
    const int rotary_dim = mConfig.rotary_dim();
    if (rotary_dim > 0) {
        rope_forward(
            acts.qkv_output, acts.qkv_output,
            w.rope_freqs,
            ctx.position_ids,
            nullptr,
            B, T, Hq, Hkv, Hs,
            rotary_dim,  // partial RoPE support
            ctx.stream
        );
    }

    // Note: LSE must be saved during forward - too expensive to recompute
    // attention_output is recomputed but we need LSE from the forward pass
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_PRIMITIVES_ATTENTION_H
