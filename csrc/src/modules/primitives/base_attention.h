// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// BaseAttentionModule - CRTP base class for attention module variants
//
// Provides common attention functionality with protected helper methods that
// derived classes can compose differently:
//   - LlamaAttentionModule: Standard attention with RoPE
//   - Qwen2AttentionModule: Adds sliding window attention
//   - Qwen3AttentionModule: Adds QK normalization

#ifndef SUROGATE_SRC_MODULES_PRIMITIVES_BASE_ATTENTION_H
#define SUROGATE_SRC_MODULES_PRIMITIVES_BASE_ATTENTION_H

#include "modules/module_base.h"
#include "modules/primitives/linear.h"
#include "config/rope_config.h"
#include "kernels/kernels.h"

namespace modules {

/**
 * @brief CRTP base class for attention module variants
 *
 * This class provides the common implementation for all attention variants
 * (LlamaAttention, Qwen2Attention, Qwen3Attention) through protected helper
 * methods. Derived classes implement forward_impl/backward_impl by composing
 * these helpers.
 *
 * The CRTP pattern ensures zero-overhead dispatch while allowing code reuse.
 *
 * @tparam Derived The concrete attention module type (CRTP)
 */
template<typename Derived>
class BaseAttentionModule : public ModuleBase<Derived> {
public:
    /**
     * @brief Base configuration for attention
     *
     * Contains all fields used by any attention variant. Derived modules
     * can ignore fields they don't use. This provides a uniform interface
     * for transformer block config creation.
     */
    struct BaseConfig {
        int hidden_size;            ///< Model dimension
        int num_query_heads;        ///< Number of query heads
        int num_kv_heads;           ///< Number of key/value heads (< num_query_heads for GQA)
        RoPEConfig rope;            ///< Flexible RoPE configuration
        bool use_qkv_bias = false;  ///< Whether QKV projection has bias
        bool use_qk_norm = false;   ///< Whether to apply QK normalization (Qwen3)
        float qk_norm_eps = 1e-6f;  ///< Epsilon for QK RMSNorm
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
     * @brief Base weight tensors for attention
     *
     * Derived modules can extend this with model-specific weights.
     */
    struct BaseWeights {
        Tensor qkv_weight;                  ///< (qkv_channels, hidden_size) fused QKV
        std::optional<Tensor> qkv_bias;     ///< (qkv_channels,) optional QKV bias
        Tensor out_weight;                  ///< (hidden_size, attn_out_channels) output projection
        Tensor rope_freqs;                  ///< Precomputed RoPE frequencies
    };

    /**
     * @brief Base saved state for backward pass
     */
    struct BaseActivations {
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
     * @brief Base weight gradients
     */
    struct BaseGradients {
        Tensor d_qkv_weight;                ///< (qkv_channels, hidden_size)
        std::optional<Tensor> d_qkv_bias;   ///< (qkv_channels,) optional
        Tensor d_out_weight;                ///< (hidden_size, attn_out_channels)
        std::optional<Tensor> bias_scratch; ///< For bias backward
    };

    /**
     * @brief Get required workspace size for cuDNN attention
     */
    static std::size_t get_workspace_size(int B, int T, int Hq, int Hkv, int Hs, cudnnHandle_t handle) {
        return cudnn_get_workspace_size(B, T, Hq, Hkv, Hs, handle);
    }

protected:
    // ========================================================================
    // Protected helper methods for derived classes to compose
    // ========================================================================

    /**
     * @brief Forward QKV projection
     *
     * Computes fused QKV projection: qkv = input @ qkv_weight^T + qkv_bias
     * Handles optional quantization for FP8 training.
     *
     * @param ctx Module context with CUDA resources
     * @param config Attention configuration
     * @param qkv_weight QKV projection weight (qkv_channels, hidden_size)
     * @param qkv_bias Optional QKV bias
     * @param input Input tensor (B*T, hidden_size)
     * @param qkv_input Quantizable tensor to save input for backward
     * @param qkv_output Output tensor to fill (B*T, qkv_channels)
     */
    static void forward_qkv_proj(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& qkv_weight,
        const std::optional<Tensor>& qkv_bias,
        Tensor& input,
        QuantizableTensor& qkv_input,
        Tensor& qkv_output
    ) {
        const int BT = ctx.B * ctx.T;
        const int C = config.hidden_size;
        const int OC = config.qkv_channels();

        // Save input for backward
        qkv_input.Value = input;

        // Determine if quantization is needed
        bool needs_quant = qkv_weight.DType != input.DType;

        if (needs_quant && qkv_input.Quant.has_value()) {
            quantize_with_abs_max(
                qkv_input.Quant.value(),
                qkv_input.Quant->scale(),
                input,
                qkv_input.Quant->abs_max(),
                BT * C,
                *ctx.device_prop,
                ctx.stream
            );
        }

        const Tensor& inp = qkv_input.for_matmul();
        const float* scale_a = qkv_weight.scale();
        const float* scale_b = qkv_input.scale();

        matmul(
            qkv_output, qkv_weight, inp, qkv_bias,
            scale_a, scale_b,
            ctx.cublas_handle, *ctx.workspace,
            OC, BT, C, EMMTranspose::TN, false,
            ctx.stream
        );
    }

    /**
     * @brief Apply RoPE positional encoding to QKV tensor
     *
     * @param ctx Module context
     * @param config Attention configuration
     * @param qkv QKV tensor to apply RoPE to (in-place)
     * @param rope_freqs Precomputed RoPE frequencies
     */
    static void apply_rope(
        ModuleContext& ctx,
        const BaseConfig& config,
        Tensor& qkv,
        const Tensor& rope_freqs
    ) {
        const int rotary_dim = config.rotary_dim();
        if (rotary_dim <= 0) return;

        rope_forward(
            qkv, qkv,
            rope_freqs,
            ctx.position_ids,
            nullptr,  // abs_max for quantization
            ctx.B, ctx.T,
            config.num_query_heads,
            config.num_kv_heads,
            config.head_dim(),
            rotary_dim,
            ctx.stream
        );
    }

    /**
     * @brief RoPE backward pass
     */
    static void apply_rope_backward(
        ModuleContext& ctx,
        const BaseConfig& config,
        Tensor& d_qkv,
        const Tensor& rope_freqs
    ) {
        const int rotary_dim = config.rotary_dim();
        if (rotary_dim <= 0) return;

        rope_backward(
            d_qkv, d_qkv,
            rope_freqs,
            ctx.position_ids,
            nullptr,  // abs_max
            ctx.B, ctx.T,
            config.num_query_heads,
            config.num_kv_heads,
            config.head_dim(),
            rotary_dim,
            ctx.stream
        );
    }

    /**
     * @brief Forward attention core using cuDNN FlashAttention
     *
     * @param ctx Module context
     * @param config Attention configuration
     * @param qkv QKV tensor after RoPE/norm (B, T, qkv_channels)
     * @param attention_output Output tensor to fill
     * @param lse Log-sum-exp tensor to fill (for backward)
     */
    static void forward_attention_core(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& qkv,
        Tensor& attention_output,
        Tensor& lse
    ) {
        attention_forward_cudnn(
            attention_output,
            lse,
            qkv,
            *ctx.workspace,
            ctx.cudnn_handle,
            ctx.B, ctx.T,
            config.num_query_heads,
            config.num_kv_heads,
            config.head_dim(),
            ctx.stream
        );
    }

    /**
     * @brief Backward attention core
     *
     * @param ctx Module context
     * @param config Attention configuration
     * @param d_att Gradient w.r.t. attention output
     * @param qkv QKV tensor from forward
     * @param attention_output Attention output from forward
     * @param lse Log-sum-exp from forward
     * @param d_qkv Output: gradient w.r.t. QKV
     */
    static void backward_attention_core(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& d_att,
        const Tensor& qkv,
        const Tensor& attention_output,
        const Tensor& lse,
        Tensor& d_qkv
    ) {
        attention_backward_cudnn(
            d_qkv,
            lse,
            attention_output,
            d_att,
            qkv,
            *ctx.workspace,
            ctx.cudnn_handle,
            ctx.B, ctx.T,
            config.num_query_heads,
            config.num_kv_heads,
            config.head_dim(),
            ctx.stream
        );
    }

    /**
     * @brief Forward output projection
     *
     * @param ctx Module context
     * @param config Attention configuration
     * @param out_weight Output projection weight (hidden_size, attn_out_channels)
     * @param attention_output Attention output (B*T, attn_out_channels)
     * @param att_for_out Quantizable tensor for backward
     * @return Output tensor (B*T, hidden_size)
     */
    static Tensor forward_out_proj(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& out_weight,
        Tensor& attention_output,
        QuantizableTensor& att_for_out
    ) {
        const int BT = ctx.B * ctx.T;
        const int C = config.hidden_size;
        const int InC = config.attn_out_channels();

        Tensor output;
        output.DType = attention_output.DType;

        // Save attention output for backward
        att_for_out.Value = attention_output;

        // Determine if quantization is needed
        bool needs_quant = out_weight.DType != attention_output.DType;

        if (needs_quant && att_for_out.Quant.has_value()) {
            quantize_with_abs_max(
                att_for_out.Quant.value(),
                att_for_out.Quant->scale(),
                attention_output,
                att_for_out.Quant->abs_max(),
                BT * InC,
                *ctx.device_prop,
                ctx.stream
            );
        }

        const Tensor& inp = att_for_out.for_matmul();
        const float* scale_a = out_weight.scale();
        const float* scale_b = att_for_out.scale();

        matmul(
            output, out_weight, inp, std::nullopt,
            scale_a, scale_b,
            ctx.cublas_handle, *ctx.workspace,
            C, BT, InC, EMMTranspose::TN, false,
            ctx.stream
        );

        return output;
    }

    /**
     * @brief Backward through output projection
     *
     * @param ctx Module context
     * @param config Attention configuration
     * @param out_weight Output projection weight
     * @param att_for_out Saved attention output
     * @param grad_output Gradient w.r.t. output
     * @param d_out_weight Gradient w.r.t. output weight (accumulated)
     * @param accumulate Whether to accumulate gradients
     * @return Gradient w.r.t. attention output
     */
    static Tensor backward_out_proj(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& out_weight,
        const QuantizableTensor& att_for_out,
        Tensor& grad_output,
        Tensor& d_out_weight,
        bool accumulate
    ) {
        const int BT = ctx.B * ctx.T;
        const int C = config.hidden_size;
        const int InC = config.attn_out_channels();

        Tensor d_att;
        d_att.DType = att_for_out.Value.DType;

        // d_att = grad_output @ out_weight (NN transpose)
        matmul(
            d_att, out_weight, grad_output, std::nullopt,
            nullptr, nullptr,
            ctx.cublas_handle, *ctx.workspace,
            InC, BT, C, EMMTranspose::NN, false,
            ctx.stream
        );

        // d_out_weight = att^T @ grad_output (NT transpose)
        matmul(
            d_out_weight, att_for_out.Value, grad_output, std::nullopt,
            nullptr, nullptr,
            ctx.cublas_handle, *ctx.workspace,
            InC, C, BT, EMMTranspose::NT, accumulate,
            ctx.stream
        );

        return d_att;
    }

    // ========================================================================
    // QK Normalization helpers (for Qwen3 and similar models)
    // ========================================================================

    /**
     * @brief Configuration extension for QK normalization
     */
    struct QKNormConfig {
        bool use_qk_norm = false;   ///< Whether to apply QK normalization
        float qk_norm_eps = 1e-6f;  ///< Epsilon for QK RMSNorm
    };

    /**
     * @brief Additional weights for QK normalization
     */
    struct QKNormWeights {
        std::optional<Tensor> q_norm_weight; ///< (head_dim,) per-head RMSNorm weight for Q
        std::optional<Tensor> k_norm_weight; ///< (head_dim,) per-head RMSNorm weight for K
    };

    /**
     * @brief Additional activations for QK normalization backward
     */
    struct QKNormActivations {
        Tensor q_rstd;  ///< (B, T, Hq) reciprocal std for Q heads
        Tensor k_rstd;  ///< (B, T, Hkv) reciprocal std for K heads
    };

    /**
     * @brief Additional gradients for QK normalization
     */
    struct QKNormGradients {
        std::optional<Tensor> d_q_norm_weight; ///< (head_dim,)
        std::optional<Tensor> d_k_norm_weight; ///< (head_dim,)
    };

    /**
     * @brief Apply fused QK normalization + RoPE (forward)
     *
     * Applies per-head RMSNorm to Q and K before RoPE, then applies RoPE.
     * This is more efficient than separate operations.
     *
     * @param ctx Module context
     * @param config Base attention config
     * @param qk_config QK norm config
     * @param qkv QKV tensor to apply norm+RoPE to (in-place)
     * @param q_norm_weight Q normalization weight
     * @param k_norm_weight K normalization weight
     * @param rope_freqs RoPE frequencies
     * @param q_rstd Output: reciprocal std for Q (for backward)
     * @param k_rstd Output: reciprocal std for K (for backward)
     */
    static void apply_qk_norm_rope(
        ModuleContext& ctx,
        const BaseConfig& config,
        const QKNormConfig& qk_config,
        Tensor& qkv,
        const Tensor& q_norm_weight,
        const Tensor& k_norm_weight,
        const Tensor& rope_freqs,
        Tensor& q_rstd,
        Tensor& k_rstd
    ) {
        const int B = ctx.B;
        const int T = ctx.T;
        const int Hq = config.num_query_heads;
        const int Hkv = config.num_kv_heads;
        const int Hs = config.head_dim();
        const int rotary_dim = config.rotary_dim();

        if (rotary_dim > 0) {
            // Fused QK norm + RoPE kernel (most efficient)
            qkv_qk_norm_rope_forward(
                qkv,
                q_rstd, k_rstd,
                q_norm_weight, k_norm_weight,
                rope_freqs, ctx.position_ids,
                qk_config.qk_norm_eps, B, T, Hq, Hkv, Hs,
                ctx.stream
            );
        } else {
            // QK norm without RoPE (rare case)
            const int qkv_channels = config.qkv_channels();
            // Apply Q norm
            qkv_head_rmsnorm_forward(
                qkv, q_rstd, q_norm_weight,
                qk_config.qk_norm_eps, B, T, qkv_channels,
                Hq, Hs, 0,  // Q starts at channel offset 0
                ctx.stream
            );
            // Apply K norm
            qkv_head_rmsnorm_forward(
                qkv, k_rstd, k_norm_weight,
                qk_config.qk_norm_eps, B, T, qkv_channels,
                Hkv, Hs, Hq * Hs,  // K starts after Q
                ctx.stream
            );
        }
    }

    /**
     * @brief Backward through QK normalization + RoPE
     *
     * @param ctx Module context
     * @param config Base attention config
     * @param qk_config QK norm config
     * @param d_qkv Gradient w.r.t. QKV (modified in place)
     * @param qkv QKV tensor from forward
     * @param q_norm_weight Q normalization weight
     * @param k_norm_weight K normalization weight
     * @param rope_freqs RoPE frequencies
     * @param q_rstd Saved q_rstd from forward
     * @param k_rstd Saved k_rstd from forward
     * @param d_q_norm_weight Output: gradient w.r.t. Q norm weight
     * @param d_k_norm_weight Output: gradient w.r.t. K norm weight
     * @param accumulate Whether to accumulate weight gradients
     */
    static void apply_qk_norm_rope_backward(
        ModuleContext& ctx,
        const BaseConfig& config,
        const QKNormConfig& qk_config,
        Tensor& d_qkv,
        const Tensor& qkv,
        const Tensor& q_norm_weight,
        const Tensor& k_norm_weight,
        const Tensor& rope_freqs,
        const Tensor& q_rstd,
        const Tensor& k_rstd,
        std::optional<Tensor>& d_q_norm_weight,
        std::optional<Tensor>& d_k_norm_weight,
        bool accumulate
    ) {
        const int B = ctx.B;
        const int T = ctx.T;
        const int Hq = config.num_query_heads;
        const int Hkv = config.num_kv_heads;
        const int Hs = config.head_dim();
        const int qkv_channels = config.qkv_channels();
        const int rotary_dim = config.rotary_dim();

        if (rotary_dim > 0) {
            // Fused QK norm + RoPE backward
            // Backward through Q norm + RoPE
            qkv_head_rmsnorm_rope_backward_dx(
                d_qkv, qkv, q_norm_weight, q_rstd,
                rope_freqs, ctx.position_ids,
                B, T, qkv_channels, Hq, Hs, 0,
                ctx.stream
            );
            // Backward through K norm + RoPE
            qkv_head_rmsnorm_rope_backward_dx(
                d_qkv, qkv, k_norm_weight, k_rstd,
                rope_freqs, ctx.position_ids,
                B, T, qkv_channels, Hkv, Hs, Hq * Hs,
                ctx.stream
            );
            // Weight gradients
            if (d_q_norm_weight.has_value()) {
                qkv_head_rmsnorm_rope_backward_dweight(
                    d_q_norm_weight.value(), d_qkv, qkv, q_norm_weight,
                    rope_freqs, ctx.position_ids,
                    B, T, qkv_channels, Hq, Hs, 0,
                    accumulate, ctx.stream
                );
            }
            if (d_k_norm_weight.has_value()) {
                qkv_head_rmsnorm_rope_backward_dweight(
                    d_k_norm_weight.value(), d_qkv, qkv, k_norm_weight,
                    rope_freqs, ctx.position_ids,
                    B, T, qkv_channels, Hkv, Hs, Hq * Hs,
                    accumulate, ctx.stream
                );
            }
        } else {
            // QK norm backward without RoPE
            qkv_head_rmsnorm_backward_dx(
                d_qkv, qkv, q_norm_weight, q_rstd,
                B, T, qkv_channels, Hq, Hs, 0,
                ctx.stream
            );
            qkv_head_rmsnorm_backward_dx(
                d_qkv, qkv, k_norm_weight, k_rstd,
                B, T, qkv_channels, Hkv, Hs, Hq * Hs,
                ctx.stream
            );
            // Weight gradients
            if (d_q_norm_weight.has_value()) {
                qkv_head_rmsnorm_backward_dweight(
                    d_q_norm_weight.value(), d_qkv, qkv, q_norm_weight,
                    B, T, qkv_channels, Hq, Hs, 0,
                    accumulate, ctx.stream
                );
            }
            if (d_k_norm_weight.has_value()) {
                qkv_head_rmsnorm_backward_dweight(
                    d_k_norm_weight.value(), d_qkv, qkv, k_norm_weight,
                    B, T, qkv_channels, Hkv, Hs, Hq * Hs,
                    accumulate, ctx.stream
                );
            }
        }
    }

    /**
     * @brief Backward through QKV projection
     *
     * @param ctx Module context
     * @param config Attention configuration
     * @param qkv_weight QKV projection weight
     * @param qkv_input Saved input from forward
     * @param d_qkv Gradient w.r.t. QKV output
     * @param d_qkv_weight Gradient w.r.t. QKV weight (accumulated)
     * @param d_qkv_bias Optional gradient w.r.t. QKV bias
     * @param bias_scratch Scratch buffer for bias backward
     * @param accumulate Whether to accumulate gradients
     * @return Gradient w.r.t. input
     */
    static Tensor backward_qkv_proj(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& qkv_weight,
        const QuantizableTensor& qkv_input,
        Tensor& d_qkv,
        Tensor& d_qkv_weight,
        std::optional<Tensor>& d_qkv_bias,
        std::optional<Tensor>& bias_scratch,
        bool accumulate
    ) {
        const int BT = ctx.B * ctx.T;
        const int C = config.hidden_size;
        const int OC = config.qkv_channels();

        Tensor d_input;
        d_input.DType = qkv_input.Value.DType;

        // d_input = d_qkv @ qkv_weight (NN transpose)
        matmul(
            d_input, qkv_weight, d_qkv, std::nullopt,
            nullptr, nullptr,
            ctx.cublas_handle, *ctx.workspace,
            C, BT, OC, EMMTranspose::NN, false,
            ctx.stream
        );

        // d_qkv_weight = input^T @ d_qkv (NT transpose)
        matmul(
            d_qkv_weight, qkv_input.Value, d_qkv, std::nullopt,
            nullptr, nullptr,
            ctx.cublas_handle, *ctx.workspace,
            C, OC, BT, EMMTranspose::NT, accumulate,
            ctx.stream
        );

        // d_qkv_bias if needed
        if (d_qkv_bias.has_value() && bias_scratch.has_value()) {
            backward_bias(
                d_qkv_bias.value(), d_qkv, nullptr, nullptr,
                bias_scratch.value(), ctx.B, ctx.T, OC,
                *ctx.device_prop, ctx.stream
            );
        }

        return d_input;
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_PRIMITIVES_BASE_ATTENTION_H
