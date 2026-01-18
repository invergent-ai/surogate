// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_PRIMITIVES_MLP_H
#define SUROGATE_SRC_MODULES_PRIMITIVES_MLP_H

#include "modules/module_base.h"
#include "modules/primitives/swiglu.h"
#include "modules/primitives/linear.h"
#include "kernels/kernels.h"

namespace modules {

/**
 * @brief MLP (Feed-Forward Network) module for transformer blocks
 *
 * Implements the standard LLaMA/Qwen-style MLP:
 *   hidden = swiglu(Linear_up(x))
 *   output = Linear_down(hidden)
 *
 * Where Linear_up produces fused [gate, up] projections:
 *   gate = x @ W_gate^T
 *   up = x @ W_up^T
 *   hidden = swish(gate) * up
 *
 * Shapes:
 * - Input: (B*T, hidden_size)
 * - Gate+Up output: (B*T, 2 * intermediate_size)
 * - SwiGLU output: (B*T, intermediate_size)
 * - Final output: (B*T, hidden_size)
 *
 * Supports:
 * - FP8 activation quantization between projections
 * - Fused gate+up weights for memory efficiency
 * - Gradient checkpointing (recomputation)
 */
template<typename ActivationType = SwiGLUModule>
class MLPModule : public ModuleBase<MLPModule<ActivationType>> {
public:
    /**
     * @brief Configuration for MLP
     */
    struct Config {
        int hidden_size;            ///< Input/output dimension
        int intermediate_size;      ///< Hidden dimension (typically 4x or 8/3x hidden_size)

        /// Get activation config
        [[nodiscard]] typename ActivationType::Config activation_config() const {
            return {.intermediate_size = intermediate_size};
        }
    };

    /**
     * @brief Weight tensors for MLP
     *
     * The up/gate weights are fused into a single tensor for efficiency.
     * Weight layout: [gate_weight; up_weight] where each is (intermediate_size, hidden_size)
     * Combined shape: (2 * intermediate_size, hidden_size)
     */
    struct Weights {
        Tensor up_weight;           ///< (2 * intermediate_size, hidden_size) fused gate+up
        Tensor down_weight;         ///< (hidden_size, intermediate_size) down projection
    };

    /**
     * @brief Saved state for backward pass
     */
    struct Activations {
        // Input to MLP
        QuantizableTensor input;            ///< Input from LN2 (may be quantized for matmul)

        // Gate+Up projection output
        Tensor up_output;                   ///< (B*T, 2 * intermediate_size)

        // Activation output
        typename ActivationType::Activations activation_acts;

        // Input to down projection
        QuantizableTensor down_input;       ///< SwiGLU output (may be quantized)

        // Final output
        Tensor output;                      ///< (B*T, hidden_size)
    };

    /**
     * @brief Weight gradients
     */
    struct Gradients {
        Tensor d_up_weight;         ///< (2 * intermediate_size, hidden_size)
        Tensor d_down_weight;       ///< (hidden_size, intermediate_size)
    };

    explicit MLPModule(Config config)
        : mConfig(config),
          mActivation(config.activation_config()) {}

    /**
     * @brief Forward pass: up -> activation -> down
     *
     * @param ctx Module context with CUDA resources
     * @param w Weight tensors
     * @param input Input tensor (B*T, hidden_size)
     * @param acts Activation storage for backward
     * @return Output tensor (B*T, hidden_size)
     */
    Tensor forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts);

    /**
     * @brief Backward pass: compute gradients w.r.t. input and weights
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
     * @brief Recompute activations for gradient checkpointing
     *
     * Recomputes up projection and activation, skipping down projection
     * (which isn't needed for backward through the activation).
     */
    void recompute_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts);

    // Accessors
    [[nodiscard]] const Config& config() const { return mConfig; }
    [[nodiscard]] int hidden_size() const { return mConfig.hidden_size; }
    [[nodiscard]] int intermediate_size() const { return mConfig.intermediate_size; }

private:
    Config mConfig;
    ActivationType mActivation;

    // Forward helpers
    void forward_up_projection(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts);
    void forward_down_projection(ModuleContext& ctx, Weights& w, Activations& acts);

    // Backward helpers
    Tensor backward_down_projection(ModuleContext& ctx, Weights& w, Activations& acts,
                                     Tensor& grad_output, Gradients& grads, bool accumulate);
    Tensor backward_up_projection(ModuleContext& ctx, Weights& w, Activations& acts,
                                   Tensor& d_swiglu, Gradients& grads, bool accumulate);
};

// ============================================================================
// Implementation
// ============================================================================

template<typename Act>
Tensor MLPModule<Act>::forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
    // Save input for backward
    acts.input.Value = input;

    // 1) Gate + Up projection (fused)
    forward_up_projection(ctx, w, input, acts);

    // 2) SwiGLU activation
    typename Act::Weights act_weights{};
    mActivation.forward(ctx, act_weights, acts.up_output, acts.activation_acts);

    // Save activation output for down projection backward
    acts.down_input.Value = acts.activation_acts.output.Value;

    // 3) Down projection
    forward_down_projection(ctx, w, acts);

    return acts.output;
}

template<typename Act>
void MLPModule<Act>::forward_up_projection(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
    const int BT = ctx.B * ctx.T;
    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;

    // Check if quantization is needed
    const bool needs_quant = w.up_weight.DType != input.DType;

    if (needs_quant && acts.input.Quant.has_value()) {
        // Quantize input for FP8 matmul
        quantize_with_abs_max(
            acts.input.Quant.value(),
            acts.input.Quant->scale(),
            input,
            acts.input.Quant->abs_max(),
            BT * C,
            *ctx.device_prop,
            ctx.stream
        );
    }

    const Tensor& inp = acts.input.for_matmul();
    float* scale_a = w.up_weight.scale();
    float* scale_b = acts.input.scale();

    // Gate+Up: output = input @ up_weight^T
    // output shape: (BT, 2*D)
    matmul(
        acts.up_output, w.up_weight, inp, std::nullopt,
        scale_a, scale_b,
        ctx.cublas_handle, *ctx.workspace,
        2 * D, BT, C, EMMTranspose::TN, false,
        ctx.stream
    );
}

template<typename Act>
void MLPModule<Act>::forward_down_projection(ModuleContext& ctx, Weights& w, Activations& acts) {
    const int BT = ctx.B * ctx.T;
    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;

    // Check if quantization is needed
    const bool needs_quant = w.down_weight.DType != acts.down_input.Value.DType;

    if (needs_quant && acts.down_input.Quant.has_value()) {
        // Quantize swiglu output for FP8 matmul
        quantize_with_abs_max(
            acts.down_input.Quant.value(),
            acts.down_input.Quant->scale(),
            acts.down_input.Value,
            acts.down_input.Quant->abs_max(),
            BT * D,
            *ctx.device_prop,
            ctx.stream
        );
    }

    const Tensor& inp = acts.down_input.for_matmul();
    float* scale_a = w.down_weight.scale();
    float* scale_b = acts.down_input.scale();

    // Down: output = swiglu_output @ down_weight^T
    // output shape: (BT, C)
    matmul(
        acts.output, w.down_weight, inp, std::nullopt,
        scale_a, scale_b,
        ctx.cublas_handle, *ctx.workspace,
        C, BT, D, EMMTranspose::TN, false,
        ctx.stream
    );
}

template<typename Act>
Tensor MLPModule<Act>::backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                                      Tensor& grad_output, Gradients& grads, bool accumulate) {
    // Backward through down projection
    Tensor d_swiglu = backward_down_projection(ctx, w, acts, grad_output, grads, accumulate);

    // Backward through activation
    typename Act::Weights act_weights{};
    typename Act::Gradients act_grads{};
    Tensor d_up = mActivation.backward(ctx, act_weights, acts.activation_acts, d_swiglu, act_grads, false);

    // Backward through up projection
    return backward_up_projection(ctx, w, acts, d_up, grads, accumulate);
}

template<typename Act>
Tensor MLPModule<Act>::backward_down_projection(ModuleContext& ctx, Weights& w, Activations& acts,
                                                  Tensor& grad_output, Gradients& grads, bool accumulate) {
    const int BT = ctx.B * ctx.T;
    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;

    // d_swiglu = grad_output @ down_weight (NN transpose)
    Tensor d_swiglu;
    d_swiglu.DType = acts.down_input.Value.DType;

    matmul(
        d_swiglu, w.down_weight, grad_output, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        D, BT, C, EMMTranspose::NN, false,
        ctx.stream
    );

    // d_down_weight = swiglu_output^T @ grad_output (NT transpose)
    matmul(
        grads.d_down_weight, acts.down_input.Value, grad_output, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        D, C, BT, EMMTranspose::NT, accumulate,
        ctx.stream
    );

    return d_swiglu;
}

template<typename Act>
Tensor MLPModule<Act>::backward_up_projection(ModuleContext& ctx, Weights& w, Activations& acts,
                                                Tensor& d_up, Gradients& grads, bool accumulate) {
    const int BT = ctx.B * ctx.T;
    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;

    // d_input = d_up @ up_weight (NN transpose)
    Tensor d_input;
    d_input.DType = acts.input.Value.DType;

    matmul(
        d_input, w.up_weight, d_up, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        C, BT, 2 * D, EMMTranspose::NN, false,
        ctx.stream
    );

    // d_up_weight = input^T @ d_up (NT transpose)
    matmul(
        grads.d_up_weight, acts.input.Value, d_up, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        C, 2 * D, BT, EMMTranspose::NT, accumulate,
        ctx.stream
    );

    return d_input;
}

template<typename Act>
void MLPModule<Act>::recompute_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
    // Recompute up projection
    acts.input.Value = input;
    forward_up_projection(ctx, w, input, acts);

    // Recompute activation
    typename Act::Weights act_weights{};
    mActivation.forward(ctx, act_weights, acts.up_output, acts.activation_acts);

    // Save for down projection backward
    acts.down_input.Value = acts.activation_acts.output.Value;

    // Note: down projection output isn't needed for backward, skip recomputing it
}

// Type aliases for common configurations
using SwiGLUMLP = MLPModule<SwiGLUModule>;
using GeGLUMLP = MLPModule<GeGLUModule>;

} // namespace modules

#endif // SUROGATE_SRC_MODULES_PRIMITIVES_MLP_H
