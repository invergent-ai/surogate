// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// BaseRMSNormModule - CRTP base class for RMSNorm module variants
//
// Provides common RMSNorm functionality with protected helper methods.
// All modern LLM architectures (Llama, Qwen2, Qwen3) use the same RMSNorm
// formula: y = x / RMS(x) * weight, where RMS(x) = sqrt(mean(x^2) + eps).

#ifndef SUROGATE_SRC_MODULES_PRIMITIVES_BASE_RMSNORM_H
#define SUROGATE_SRC_MODULES_PRIMITIVES_BASE_RMSNORM_H

#include "modules/module_base.h"
#include "kernels/kernels.h"

namespace modules {

/**
 * @brief CRTP base class for RMSNorm module variants
 *
 * Provides the common implementation for RMSNorm operations.
 * The RMSNorm formula is: y = x / RMS(x) * weight
 * where RMS(x) = sqrt(mean(x^2) + epsilon)
 *
 * Unlike LayerNorm, RMSNorm does not center (subtract mean), making it
 * simpler and slightly faster while achieving similar results.
 *
 * @tparam Derived The concrete RMSNorm module type (CRTP)
 */
template<typename Derived>
class BaseRMSNormModule : public ModuleBase<Derived> {
public:
    /**
     * @brief Base configuration for RMSNorm
     */
    struct BaseConfig {
        int hidden_size;            ///< Dimension to normalize over
        float epsilon = 1e-5f;      ///< Epsilon for numerical stability
    };

    /**
     * @brief Base weight tensors for RMSNorm
     */
    struct BaseWeights {
        Tensor weight;              ///< (hidden_size,) learnable scale
    };

    /**
     * @brief Base saved state for backward pass
     */
    struct BaseActivations {
        Tensor rstd;                        ///< (B*T,) reciprocal std for backward
        QuantizableTensor output;           ///< Normalized output (may be quantized)
        Tensor input;                       ///< Input (needed for backward)
    };

    /**
     * @brief Base weight gradients
     */
    struct BaseGradients {
        Tensor d_weight;            ///< (hidden_size,)
        Tensor scratch;             ///< Scratch buffer for backward reduction
    };

    /**
     * @brief Get required scratch size for backward pass
     */
    static int get_backward_scratch_size(int hidden_size, const cudaDeviceProp& dp) {
        return get_rmsnorm_backward_scratch_size(hidden_size, dp);
    }

protected:
    // ========================================================================
    // Protected helper methods for derived classes
    // ========================================================================

    /**
     * @brief Forward RMSNorm: y = x / RMS(x) * weight
     *
     * @param ctx Module context with CUDA resources
     * @param config RMSNorm configuration
     * @param weight Scale weight tensor
     * @param input Input tensor (B*T, hidden_size)
     * @param rstd Output: reciprocal std for backward (B*T,)
     * @param output Output tensor for result
     * @param abs_max_ptr Optional pointer for quantization tracking
     */
    static void forward_rmsnorm(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& weight,
        const Tensor& input,
        Tensor& rstd,
        Tensor& output,
        float* abs_max_ptr = nullptr
    ) {
        rmsnorm_forward(
            output,
            rstd,
            input,
            weight,
            abs_max_ptr,
            config.epsilon,
            ctx.B, ctx.T, config.hidden_size,
            ctx.stream
        );
    }

    /**
     * @brief Backward through RMSNorm (standalone, no residual)
     *
     * @param ctx Module context
     * @param config RMSNorm configuration
     * @param weight Scale weight tensor
     * @param input Input from forward
     * @param rstd Reciprocal std from forward
     * @param grad_output Gradient w.r.t. output
     * @param grad_input Output: gradient w.r.t. input
     * @param d_weight Output: gradient w.r.t. weight
     * @param scratch Scratch buffer for reduction
     */
    static void backward_rmsnorm(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& weight,
        const Tensor& input,
        const Tensor& rstd,
        const Tensor& grad_output,
        Tensor& grad_input,
        Tensor& d_weight,
        Tensor& scratch
    ) {
        rmsnorm_backward(
            grad_input,
            d_weight,
            scratch,
            grad_input,     // dresidual - same as dinp for standalone norm
            grad_output,
            input,
            weight,
            rstd,
            nullptr,        // abs_max for quantization
            ctx.B, ctx.T, config.hidden_size,
            *ctx.device_prop,
            ctx.stream
        );
    }

    /**
     * @brief Forward fused residual + RMSNorm
     *
     * Computes: residual = residual + input; output = RMSNorm(residual)
     * More efficient than separate add + norm.
     *
     * @param ctx Module context
     * @param config RMSNorm configuration
     * @param weight Scale weight tensor
     * @param input Input to add to residual
     * @param residual Residual stream (modified in-place)
     * @param rstd Output: reciprocal std
     * @param output Output: normalized result
     * @param abs_max_ptr Optional pointer for quantization
     */
    static void forward_fused_residual_rmsnorm(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& weight,
        const Tensor& input,
        Tensor& residual,
        Tensor& rstd,
        Tensor& output,
        float* abs_max_ptr = nullptr
    ) {
        const int N = ctx.B * ctx.T;
        const int C = config.hidden_size;

        fused_residual_rmsnorm_forward(
            residual,           // Updated in-place
            output,
            rstd,
            residual,           // inp1
            input,              // inp2
            weight,
            abs_max_ptr,
            config.epsilon,
            N, C,
            ctx.stream
        );
    }

    /**
     * @brief Backward through fused residual + RMSNorm
     *
     * @param ctx Module context
     * @param config RMSNorm configuration
     * @param weight Scale weight tensor
     * @param residual_out Residual from forward (residual + input)
     * @param rstd Reciprocal std from forward
     * @param grad_output Gradient w.r.t. normalized output
     * @param grad_residual Gradient w.r.t. residual (accumulated)
     * @param grad_input Output: gradient w.r.t. input
     * @param d_weight Output: gradient w.r.t. weight
     * @param scratch Scratch buffer
     */
    static void backward_fused_residual_rmsnorm(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& weight,
        const Tensor& residual_out,
        const Tensor& rstd,
        const Tensor& grad_output,
        Tensor& grad_residual,
        Tensor& grad_input,
        Tensor& d_weight,
        Tensor& scratch
    ) {
        rmsnorm_backward(
            grad_input,
            d_weight,
            scratch,
            grad_residual,      // dresidual - gets gradient accumulated
            grad_output,
            residual_out,
            weight,
            rstd,
            nullptr,
            ctx.B, ctx.T, config.hidden_size,
            *ctx.device_prop,
            ctx.stream
        );
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_PRIMITIVES_BASE_RMSNORM_H
