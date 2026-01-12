// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_PRIMITIVES_LINEAR_H
#define SUROGATE_SRC_MODULES_PRIMITIVES_LINEAR_H

#include "modules/module_base.h"
#include "kernels/kernels.h"

namespace modules {

/**
 * @brief Linear projection module: y = x @ W^T + b
 *
 * Supports:
 * - Optional bias
 * - FP8 quantization for activations/weights
 * - Fused output for QKV or gate+up projections (via output_offset/total_out_features)
 *
 * Weight layout:
 * - weight: (out_features, in_features) - transposed for TN matmul
 * - bias: (out_features,) optional
 *
 * The module wraps the forward_qmm/backward_qmm patterns,
 * providing a clean interface while maintaining the same performance characteristics.
 */
class LinearModule : public ModuleBase<LinearModule> {
public:
    /**
     * @brief Configuration for linear projection
     */
    struct Config {
        int in_features;            ///< Input dimension
        int out_features;           ///< Output dimension
        bool has_bias = false;      ///< Whether to add bias after matmul

        // For fused projections (QKV, gate+up) - write to a slice of larger output
        int output_offset = 0;      ///< Column offset in fused output tensor
        int total_out_features = 0; ///< Total fused output size (0 = not fused)

        // Helper to check if this is a fused projection
        [[nodiscard]] bool is_fused() const {
            return total_out_features > 0;
        }
    };

    /**
     * @brief Weight tensors for linear projection
     */
    struct Weights {
        Tensor weight;                      ///< (out_features, in_features)
        std::optional<Tensor> bias;         ///< (out_features,) optional
    };

    /**
     * @brief Saved state for backward pass
     */
    struct Activations {
        QuantizableTensor input_cached;     ///< Input tensor (may be quantized)
        bool input_was_quantized = false;   ///< Whether input was already quantized
    };

    /**
     * @brief Weight gradients
     */
    struct Gradients {
        Tensor d_weight;                    ///< (out_features, in_features)
        std::optional<Tensor> d_bias;       ///< (out_features,) optional
        std::optional<Tensor> bias_scratch; ///< Scratch buffer for bias backward
    };

    explicit LinearModule(Config config) : mConfig(config) {}

    /**
     * @brief Forward pass: y = x @ W^T + b
     *
     * If quantization is enabled (weight dtype != input dtype), the input
     * is quantized before matmul and the quantized version is cached.
     *
     * @param ctx Module context with CUDA resources
     * @param w Weight tensors
     * @param input Input tensor (B*T, in_features)
     * @param acts Activation storage for backward
     * @return Output tensor (B*T, out_features)
     */
    Tensor forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts);

    /**
     * @brief Backward pass: compute gradients w.r.t. input and weights
     *
     * Computes:
     * - d_input = d_output @ W
     * - d_weight += input^T @ d_output (or = if !accumulate)
     * - d_bias += sum(d_output, dim=0) if has_bias
     *
     * @param ctx Module context with CUDA resources
     * @param w Weight tensors
     * @param acts Saved activations from forward
     * @param grad_output Gradient w.r.t. output (B*T, out_features)
     * @param grads Gradient storage
     * @param accumulate If true, accumulate into existing gradients
     * @return Gradient w.r.t. input (B*T, in_features)
     */
    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         Tensor& grad_output, Gradients& grads, bool accumulate = false);

    // Accessors
    [[nodiscard]] const Config& config() const { return mConfig; }
    [[nodiscard]] int in_features() const { return mConfig.in_features; }
    [[nodiscard]] int out_features() const { return mConfig.out_features; }

private:
    Config mConfig;
};

// Implementation inline for header-only or in separate .cpp

inline Tensor LinearModule::forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
    const int BT = ctx.B * ctx.T;
    const int C = mConfig.in_features;
    const int OC = mConfig.out_features;

    // Cache input for backward
    acts.input_cached.Value = input;

    // Determine if we need quantization
    bool needs_quant = w.weight.DType != input.DType;

    if (needs_quant && acts.input_cached.Quant.has_value()) {
        // Quantize input
        quantize_with_abs_max(
            acts.input_cached.Quant.value(),
            acts.input_cached.Quant->scale(),
            input,
            acts.input_cached.Quant->abs_max(),
            BT * C,
            *ctx.device_prop,
            ctx.stream
        );
        acts.input_was_quantized = true;
    } else {
        acts.input_was_quantized = false;
    }

    // Get the tensor to use for matmul
    const Tensor& inp_for_mm = acts.input_cached.for_matmul();

    // Determine scales for FP8 matmul
    float* scale_a = w.weight.scale();
    float* scale_b = acts.input_cached.scale();

    // Allocate output - for fused case, output is pre-allocated by caller
    // Here we assume output is passed in or pre-allocated
    // In practice, the caller provides the output buffer

    // For now, create a placeholder - actual allocation handled by run state
    Tensor output;
    output.DType = w.weight.DType == ETensorDType::FP8_E4M3 ? ctx.matmul_dtype : w.weight.DType;
    // Caller must set output.Data appropriately

    if (mConfig.is_fused()) {
        // Write to slice of larger output tensor
        matmul_strided_c(
            output, w.weight, inp_for_mm, w.bias,
            scale_a, scale_b,
            ctx.cublas_handle, *ctx.workspace,
            OC, BT, C, EMMTranspose::TN, false,
            mConfig.total_out_features,
            ctx.stream
        );
    } else {
        matmul(
            output, w.weight, inp_for_mm, w.bias,
            scale_a, scale_b,
            ctx.cublas_handle, *ctx.workspace,
            OC, BT, C, EMMTranspose::TN, false,
            ctx.stream
        );
    }

    return output;
}

inline Tensor LinearModule::backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                                          Tensor& grad_output, Gradients& grads, bool accumulate) {
    const int BT = ctx.B * ctx.T;
    const int C = mConfig.in_features;
    const int OC = mConfig.out_features;

    // Prepare output gradient for matmul (may need quantization)
    QuantizableTensor d_out;
    d_out.Value = grad_output;

    // Allocate gradient input tensor
    Tensor grad_input;
    grad_input.DType = acts.input_cached.Value.DType;
    // Caller must set grad_input.Data appropriately

    if (w.weight.DType == acts.input_cached.Value.DType) {
        // No quantization path
        // d_input = d_output @ W (NN transpose)
        matmul(grad_input, w.weight, d_out.Value, std::nullopt,
               nullptr, nullptr,
               ctx.cublas_handle, *ctx.workspace,
               C, BT, OC, EMMTranspose::NN, false, ctx.stream);

        // d_weight = input^T @ d_output (NT transpose)
        matmul(grads.d_weight, acts.input_cached.Value, d_out.Value, std::nullopt,
               nullptr, nullptr,
               ctx.cublas_handle, *ctx.workspace,
               C, OC, BT, EMMTranspose::NT, accumulate, ctx.stream);

        // d_bias if needed
        if (grads.d_bias.has_value() && grads.bias_scratch.has_value()) {
            backward_bias(grads.d_bias.value(), d_out.Value, nullptr, nullptr,
                         grads.bias_scratch.value(), ctx.B, ctx.T, OC,
                         *ctx.device_prop, ctx.stream);
        }
    } else if (w.weight.DType == ETensorDType::BF16) {
        // BF16 weight, quantized input path
        if (d_out.Quant.has_value()) {
            quantize_with_abs_max(d_out.Quant.value(), d_out.Quant->scale(),
                                  d_out.Value, nullptr, BT * OC,
                                  *ctx.device_prop, ctx.stream);
        }

        if (!acts.input_was_quantized && acts.input_cached.Quant.has_value()) {
            quantize_with_abs_max(acts.input_cached.Quant.value(),
                                  acts.input_cached.Quant->scale(),
                                  acts.input_cached.Value, nullptr, BT * C,
                                  *ctx.device_prop, ctx.stream);
        }

        const Tensor& d_out_q = d_out.Quant.has_value() ? d_out.Quant.value() : d_out.Value;
        const Tensor& inp_q = acts.input_cached.for_matmul();

        matmul(grad_input, w.weight, d_out_q, std::nullopt,
               nullptr, nullptr,
               ctx.cublas_handle, *ctx.workspace,
               C, BT, OC, EMMTranspose::NN, false, ctx.stream);

        matmul(grads.d_weight, inp_q, d_out_q, std::nullopt,
               nullptr, nullptr,
               ctx.cublas_handle, *ctx.workspace,
               C, OC, BT, EMMTranspose::NT, accumulate, ctx.stream);

        if (grads.d_bias.has_value() && grads.bias_scratch.has_value()) {
            backward_bias(grads.d_bias.value(), d_out.Value, nullptr, nullptr,
                         grads.bias_scratch.value(), ctx.B, ctx.T, OC,
                         *ctx.device_prop, ctx.stream);
        }
    } else {
        // FP8 path - needs transpose for weight
        if (d_out.Quant.has_value()) {
            quantize_with_abs_max(d_out.Quant.value(), d_out.Quant->scale(),
                                  d_out.Value, d_out.Quant->abs_max(), BT * OC,
                                  *ctx.device_prop, ctx.stream);
        }

        Tensor& inp_q = acts.input_cached.for_matmul();

        // Note: FP8 backward may need weight transpose - see backward_qmm
        // This is a simplified version; full implementation would handle temp buffers
        matmul(grad_input, w.weight, d_out.Quant.value_or(d_out.Value), std::nullopt,
               w.weight.scale(), d_out.Quant ? d_out.Quant->scale() : nullptr,
               ctx.cublas_handle, *ctx.workspace,
               C, BT, OC, EMMTranspose::NN, false, ctx.stream);

        matmul(grads.d_weight, inp_q, d_out.Quant.value_or(d_out.Value), std::nullopt,
               inp_q.scale(), d_out.Quant ? d_out.Quant->scale() : nullptr,
               ctx.cublas_handle, *ctx.workspace,
               C, OC, BT, EMMTranspose::NT, accumulate, ctx.stream);

        if (grads.d_bias.has_value() && grads.bias_scratch.has_value()) {
            backward_bias(grads.d_bias.value(), d_out.Value, nullptr, nullptr,
                         grads.bias_scratch.value(), ctx.B, ctx.T, OC,
                         *ctx.device_prop, ctx.stream);
        }
    }

    return grad_input;
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_PRIMITIVES_LINEAR_H
