// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_PRIMITIVES_RMSNORM_H
#define SUROGATE_SRC_MODULES_PRIMITIVES_RMSNORM_H

#include "modules/module_base.h"
#include "kernels/kernels.h"

namespace modules {

/**
 * @brief RMS Normalization module: y = x / RMS(x) * weight
 *
 * RMS(x) = sqrt(mean(x^2) + epsilon)
 *
 * This is used in LLaMA/Qwen style transformers instead of LayerNorm.
 * No centering (mean subtraction) is performed.
 *
 * Supports:
 * - Optional quantization of output (for subsequent FP8 matmul)
 * - Fused residual+RMSNorm variant (x = residual + input; y = RMSNorm(x))
 */
class RMSNormModule : public ModuleBase<RMSNormModule> {
public:
    /**
     * @brief Configuration for RMSNorm
     */
    struct Config {
        int hidden_size;            ///< Dimension to normalize over
        float epsilon = 1e-5f;      ///< Epsilon for numerical stability
    };

    /**
     * @brief Weight tensors for RMSNorm
     */
    struct Weights {
        Tensor weight;              ///< (hidden_size,) learnable scale
    };

    /**
     * @brief Saved state for backward pass
     */
    struct Activations {
        Tensor rstd;                        ///< (B*T,) reciprocal std for backward
        QuantizableTensor output;           ///< Normalized output (may be quantized)
        Tensor input;                       ///< Input (needed for backward)
    };

    /**
     * @brief Weight gradients
     */
    struct Gradients {
        Tensor d_weight;            ///< (hidden_size,)
        Tensor scratch;             ///< Scratch buffer for backward reduction
    };

    explicit RMSNormModule(Config config) : mConfig(config) {}

    /**
     * @brief Forward pass: y = x / RMS(x) * weight
     *
     * @param ctx Module context with CUDA resources
     * @param w Weight tensors
     * @param input Input tensor (B*T, hidden_size)
     * @param acts Activation storage for backward
     * @return Output tensor (B*T, hidden_size)
     */
    Tensor forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts);

    /**
     * @brief Backward pass: compute gradients w.r.t. input and weight
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
     * @brief Get required scratch size for backward pass
     */
    static int get_backward_scratch_size(int hidden_size, const cudaDeviceProp& dp) {
        return get_rmsnorm_backward_scratch_size(hidden_size, dp);
    }

    // Accessors
    [[nodiscard]] const Config& config() const { return mConfig; }
    [[nodiscard]] int hidden_size() const { return mConfig.hidden_size; }
    [[nodiscard]] float epsilon() const { return mConfig.epsilon; }

private:
    Config mConfig;
};

inline Tensor RMSNormModule::forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
    const int C = mConfig.hidden_size;

    // Save input for backward
    acts.input = input;

    // Get abs_max pointer for quantization (if output quant buffer is set up)
    float* abs_max_ptr = acts.output.Quant.has_value() ? acts.output.Quant->abs_max() : nullptr;

    // Run RMSNorm forward
    rmsnorm_forward(
        acts.output.Value,
        acts.rstd,
        input,
        w.weight,
        abs_max_ptr,
        mConfig.epsilon,
        ctx.B, ctx.T, C,
        ctx.stream
    );

    return acts.output.Value;
}

inline Tensor RMSNormModule::backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                                           Tensor& grad_output, Gradients& grads, bool accumulate) {
    const int C = mConfig.hidden_size;

    // Allocate gradient input - caller provides the buffer
    Tensor grad_input;
    grad_input.DType = acts.input.DType;
    // Caller must set grad_input.Data

    // Get abs_max pointer for quantized grad_input
    float* abs_max_ptr = nullptr;  // Could be set if quantizing d_input

    // Note: rmsnorm_backward expects dresidual as a separate input for fused residual+norm
    // For standalone RMSNorm, we pass the same buffer for dresidual (it does +=)
    // In practice, the caller handles residual gradient accumulation

    rmsnorm_backward(
        grad_input,
        grads.d_weight,
        grads.scratch,
        grad_input,     // dresidual - same as dinp for standalone norm
        grad_output,
        acts.input,
        w.weight,
        acts.rstd,
        abs_max_ptr,
        ctx.B, ctx.T, C,
        *ctx.device_prop,
        ctx.stream
    );

    return grad_input;
}

/**
 * @brief Fused Residual + RMSNorm module
 *
 * Computes:
 *   residual = residual + input
 *   output = RMSNorm(residual)
 *
 * This is more efficient than separate add + norm operations.
 */
class FusedResidualRMSNormModule : public ModuleBase<FusedResidualRMSNormModule> {
public:
    struct Config {
        int hidden_size;
        float epsilon = 1e-5f;
    };

    struct Weights {
        Tensor weight;
    };

    struct Activations {
        Tensor rstd;
        Tensor residual_out;        ///< Updated residual (residual + input)
        QuantizableTensor output;   ///< Normalized output
    };

    struct Gradients {
        Tensor d_weight;
        Tensor scratch;
    };

    explicit FusedResidualRMSNormModule(Config config) : mConfig(config) {}

    /**
     * @brief Forward pass: residual += input; output = RMSNorm(residual)
     *
     * @param ctx Module context
     * @param w Weights
     * @param inputs Two inputs: [input, residual] where residual is updated in-place
     * @param acts Activation storage
     * @return Normalized output
     */
    Tensor forward_with_residual(ModuleContext& ctx, Weights& w,
                                  Tensor& input, Tensor& residual, Activations& acts);

    /**
     * @brief Backward pass with residual gradient
     *
     * @param ctx Module context
     * @param w Weights
     * @param acts Saved activations
     * @param grad_output Gradient w.r.t. normalized output
     * @param grad_residual Gradient w.r.t. residual (accumulated)
     * @param grads Weight gradients
     * @return Gradient w.r.t. input (before residual add)
     */
    Tensor backward_with_residual(ModuleContext& ctx, Weights& w, Activations& acts,
                                   Tensor& grad_output, Tensor& grad_residual,
                                   Gradients& grads);

    // Standard interface - not used for fused version but required by concept
    Tensor forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
        // Fused version requires separate residual - this shouldn't be called
        return forward_with_residual(ctx, w, input, input, acts);
    }

    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         Tensor& grad_output, Gradients& grads, bool = false) {
        Tensor grad_res;
        return backward_with_residual(ctx, w, acts, grad_output, grad_res, grads);
    }

private:
    Config mConfig;
};

inline Tensor FusedResidualRMSNormModule::forward_with_residual(
    ModuleContext& ctx, Weights& w, Tensor& input, Tensor& residual, Activations& acts) {

    const int N = ctx.B * ctx.T;
    const int C = mConfig.hidden_size;

    float* abs_max_ptr = acts.output.Quant.has_value() ? acts.output.Quant->abs_max() : nullptr;

    fused_residual_rmsnorm_forward(
        residual,           // Updated in-place
        acts.output.Value,
        acts.rstd,
        residual,           // inp1
        input,              // inp2
        w.weight,
        abs_max_ptr,
        mConfig.epsilon,
        N, C,
        ctx.stream
    );

    acts.residual_out = residual;
    return acts.output.Value;
}

inline Tensor FusedResidualRMSNormModule::backward_with_residual(
    ModuleContext& ctx, Weights& w, Activations& acts,
    Tensor& grad_output, Tensor& grad_residual, Gradients& grads) {

    const int C = mConfig.hidden_size;

    Tensor grad_input;
    grad_input.DType = acts.residual_out.DType;

    // rmsnorm_backward accumulates into dresidual
    rmsnorm_backward(
        grad_input,
        grads.d_weight,
        grads.scratch,
        grad_residual,      // dresidual - gets gradient accumulated
        grad_output,
        acts.residual_out,
        w.weight,
        acts.rstd,
        nullptr,
        ctx.B, ctx.T, C,
        *ctx.device_prop,
        ctx.stream
    );

    return grad_input;
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_PRIMITIVES_RMSNORM_H
