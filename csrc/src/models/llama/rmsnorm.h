// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// LlamaRMSNormModule - RMSNorm module for LLaMA family models

#ifndef SUROGATE_SRC_MODELS_LLAMA_RMSNORM_H
#define SUROGATE_SRC_MODELS_LLAMA_RMSNORM_H

#include "modules/primitives/base_rmsnorm.h"

namespace modules {

/**
 * @brief RMSNorm module for LLaMA family models
 *
 * Implements y = x / RMS(x) * weight with epsilon typically 1e-5.
 * This is the standard RMSNorm used across most LLaMA-style models.
 */
class LlamaRMSNormModule : public BaseRMSNormModule<LlamaRMSNormModule> {
public:
    using Base = BaseRMSNormModule<LlamaRMSNormModule>;

    using Config = Base::BaseConfig;
    using Weights = Base::BaseWeights;
    using Activations = Base::BaseActivations;
    using Gradients = Base::BaseGradients;

    explicit LlamaRMSNormModule(Config config) : mConfig(std::move(config)) {}

    /**
     * @brief Forward pass: y = x / RMS(x) * weight
     */
    Tensor forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
        acts.input = input;
        float* abs_max_ptr = acts.output.Quant.has_value() ? acts.output.Quant->abs_max() : nullptr;
        Base::forward_rmsnorm(ctx, mConfig, w.weight, input, acts.rstd, acts.output.Value, abs_max_ptr);
        return acts.output.Value;
    }

    /**
     * @brief Backward pass
     */
    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         Tensor& grad_output, Gradients& grads, bool /*accumulate*/ = false) {
        Tensor grad_input;
        grad_input.DType = acts.input.DType;
        Base::backward_rmsnorm(ctx, mConfig, w.weight, acts.input, acts.rstd,
                               grad_output, grad_input, grads.d_weight, grads.scratch);
        return grad_input;
    }

    [[nodiscard]] const Config& config() const { return mConfig; }
    [[nodiscard]] int hidden_size() const { return mConfig.hidden_size; }
    [[nodiscard]] float epsilon() const { return mConfig.epsilon; }

private:
    Config mConfig;
};

/**
 * @brief Fused Residual + RMSNorm module for LLaMA family models
 *
 * Computes: residual += input; output = RMSNorm(residual)
 * More efficient than separate add + norm operations.
 */
class LlamaFusedResidualRMSNormModule : public BaseRMSNormModule<LlamaFusedResidualRMSNormModule> {
public:
    using Base = BaseRMSNormModule<LlamaFusedResidualRMSNormModule>;

    using Config = Base::BaseConfig;
    using Weights = Base::BaseWeights;

    struct Activations {
        Tensor rstd;
        Tensor residual_out;        ///< Updated residual (residual + input)
        QuantizableTensor output;   ///< Normalized output
    };

    using Gradients = Base::BaseGradients;

    explicit LlamaFusedResidualRMSNormModule(Config config) : mConfig(std::move(config)) {}

    /**
     * @brief Forward pass: residual += input; output = RMSNorm(residual)
     */
    Tensor forward_with_residual(ModuleContext& ctx, Weights& w,
                                  Tensor& input, Tensor& residual, Activations& acts) {
        float* abs_max_ptr = acts.output.Quant.has_value() ? acts.output.Quant->abs_max() : nullptr;
        Base::forward_fused_residual_rmsnorm(ctx, mConfig, w.weight, input, residual,
                                              acts.rstd, acts.output.Value, abs_max_ptr);
        acts.residual_out = residual;
        return acts.output.Value;
    }

    /**
     * @brief Backward pass with residual gradient
     */
    Tensor backward_with_residual(ModuleContext& ctx, Weights& w, Activations& acts,
                                   Tensor& grad_output, Tensor& grad_residual,
                                   Gradients& grads) {
        Tensor grad_input;
        grad_input.DType = acts.residual_out.DType;
        Base::backward_fused_residual_rmsnorm(ctx, mConfig, w.weight, acts.residual_out, acts.rstd,
                                               grad_output, grad_residual, grad_input,
                                               grads.d_weight, grads.scratch);
        return grad_input;
    }

    // Standard interface (for compatibility with concept)
    Tensor forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
        return forward_with_residual(ctx, w, input, input, acts);
    }

    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         Tensor& grad_output, Gradients& grads, bool = false) {
        Tensor grad_res;
        return backward_with_residual(ctx, w, acts, grad_output, grad_res, grads);
    }

    [[nodiscard]] const Config& config() const { return mConfig; }

private:
    Config mConfig;
};

} // namespace modules

#endif // SUROGATE_SRC_MODELS_LLAMA_RMSNORM_H
