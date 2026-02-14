// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_RECIPES_RECIPE_H
#define SUROGATE_SRC_RECIPES_RECIPE_H

#include <memory>
#include <string_view>
#include <cuda_runtime.h>

#include "runtime/core/matmul_context.h"
#include "runtime/training/matmul_backend.h"  // For EMatmulBackend
#include "utilities/tensor.h"

namespace recipes {

/**
 * @brief Quantization parameters for a specific tensor role.
 *
 * These parameters control how tensors are quantized during forward and backward passes.
 * Mirrors TransformerEngine's QParams structure.
 */
struct QuantParams {
    bool random_hadamard_transform = false;  ///< Apply RHT before quantization to spread outliers
    bool stochastic_rounding = false;        ///< Use stochastic rounding (for gradients)
    bool block_2d_quantization = false;      ///< Use 2D block scaling (16x16 blocks for weights)
    bool power_2_scale = false;              ///< Constrain scales to powers of 2
    float amax_epsilon = 0.0f;               ///< Minimum amax value to prevent division by zero
};

/**
 * @brief Matmul configuration parameters.
 */
struct MatmulParams {
    bool use_split_accumulator = true;  ///< Use FP8 split accumulator for precision
};

/**
 * @brief FP8/FP4 format specification.
 */
enum class Format {
    BF16,    ///< BFloat16 (no quantization)
    E4M3,    ///< FP8 E4M3 (max = 448)
    E5M2,    ///< FP8 E5M2 (max = 57344)
    E2M1,    ///< FP4 E2M1 (max = 6.0)
    HYBRID   ///< E4M3 for forward, E5M2 for backward
};

/**
 * @brief Abstract base class for training recipes.
 *
 * A recipe defines the quantization strategy and matmul behavior for training.
 * Each recipe encapsulates:
 * - Format specification (BF16, FP8, FP4)
 * - Quantization parameters for inputs, weights, and gradients
 * - Forward and backward matmul implementations
 * - Recipe-specific state allocation
 *
 * Recipes are immutable configuration objects created at model initialization.
 */
class Recipe {
public:
    virtual ~Recipe() = default;

    // =========================================================================
    // Type checking methods (TransformerEngine pattern)
    // =========================================================================

    [[nodiscard]] virtual bool is_bf16() const { return false; }
    [[nodiscard]] virtual bool is_fp8_hybrid() const { return false; }
    [[nodiscard]] virtual bool is_nvfp4() const { return false; }
    [[nodiscard]] virtual bool is_nvfp4_cutlass() const { return false; }

    // =========================================================================
    // Format specification
    // =========================================================================

    /// @brief Format used for forward pass activations/weights
    [[nodiscard]] virtual Format forward_format() const = 0;

    /// @brief Format used for backward pass gradients
    [[nodiscard]] virtual Format backward_format() const = 0;

    // =========================================================================
    // Quantization parameters for different tensor roles
    // =========================================================================

    /// @brief Parameters for quantizing forward pass inputs
    [[nodiscard]] virtual QuantParams quant_fwd_input() const = 0;

    /// @brief Parameters for quantizing forward pass weights
    [[nodiscard]] virtual QuantParams quant_fwd_weight() const = 0;

    /// @brief Parameters for quantizing backward pass gradients
    [[nodiscard]] virtual QuantParams quant_bwd_grad() const = 0;

    // =========================================================================
    // Matmul configuration
    // =========================================================================

    /// @brief Configuration for forward pass matmul
    [[nodiscard]] virtual MatmulParams gemm_fprop() const = 0;

    /// @brief Configuration for gradient w.r.t. input matmul
    [[nodiscard]] virtual MatmulParams gemm_dgrad() const = 0;

    /// @brief Configuration for gradient w.r.t. weight matmul
    [[nodiscard]] virtual MatmulParams gemm_wgrad() const = 0;

    // =========================================================================
    // State requirements
    // =========================================================================

    /// @brief Whether this recipe requires amax history for delayed scaling
    [[nodiscard]] virtual bool requires_amax_history() const { return false; }

    /// @brief Length of amax history buffer (if requires_amax_history() is true)
    [[nodiscard]] virtual int amax_history_len() const { return 0; }

    /// @brief Whether this recipe requires per-block scales
    [[nodiscard]] virtual bool requires_block_scales() const { return false; }

    /// @brief Whether this recipe requires Hadamard workspace
    [[nodiscard]] virtual bool requires_hadamard_workspace() const { return false; }

    /// @brief Whether this recipe skips quantization for embedding and lm_head layers
    [[nodiscard]] virtual bool skip_embedding_lmhead_quant() const { return false; }

    // =========================================================================
    // Backend selection
    // =========================================================================

    /// @brief Which matmul backend this recipe uses (AUTO, CUBLASLT, CUTLASS)
    [[nodiscard]] virtual EMatmulBackend matmul_backend() const { return EMatmulBackend::AUTO; }

    // =========================================================================
    // Recipe metadata
    // =========================================================================

    /// @brief Human-readable name for logging/debugging
    [[nodiscard]] virtual std::string_view name() const = 0;

    // =========================================================================
    // Run state configuration (for Phase 6 simplification)
    // =========================================================================

    /**
     * @brief Whether this recipe uses FP8 forward pass quantization
     *
     * When true, the run state should allocate FP8ForwardQuantActivations buffers.
     * Used by FP8HybridRecipe and similar recipes.
     */
    [[nodiscard]] virtual bool uses_fp8_forward() const {
        return forward_format() == Format::E4M3 || forward_format() == Format::HYBRID;
    }

    /**
     * @brief Whether this recipe uses FP4 forward pass quantization
     *
     * When true, the run state should allocate FP4ForwardQuantActivations buffers.
     * Used by NVFP4SimpleRecipe and similar recipes.
     */
    [[nodiscard]] virtual bool uses_fp4_forward() const {
        return forward_format() == Format::E2M1;
    }

    /**
     * @brief Whether this recipe uses FP8 E5M2 backward pass
     *
     * When true, the run state should allocate E5M2 gradient buffers for backward.
     * Used by FP8HybridRecipe (E4M3 weights × E5M2 gradients).
     */
    [[nodiscard]] virtual bool uses_fp8_hybrid_backward() const {
        return backward_format() == Format::E5M2 || forward_format() == Format::HYBRID;
    }

    /**
     * @brief Whether this recipe handles forward matmul dispatch
     *
     * When true, the model should use recipe->forward_matmul() for all projections
     * instead of the conditional logic. This allows recipes to fully control
     * the matmul path.
     *
     * Default: false (use code path for backward compatibility)
     */
    [[nodiscard]] virtual bool handles_forward_matmul() const { return false; }

    // =========================================================================
    // Active matmul dispatch (new recipe-driven approach)
    // =========================================================================

    /**
     * @brief Execute forward matmul for a projection
     *
     * This method dispatches the forward pass matmul based on the recipe's
     * format and backend configuration. The default implementation uses
     * the recipe's format/backend settings to call the appropriate detail::
     * function.
     *
     * @param ctx Matmul context with all tensors and dimensions
     *
     * Derived classes can override for custom behavior (e.g., FP8 with
     * delayed scaling, FP4 with Hadamard transform).
     */
    virtual void forward_matmul(modules::MatmulContext& ctx) const;

    /**
     * @brief Execute backward matmul for a projection
     *
     * Computes both dinp (gradient w.r.t. input) and dweight (gradient w.r.t.
     * weight), unless skip_weight_grad is set in the context.
     *
     * @param ctx Matmul context with gradients and input tensors
     */
    virtual void backward_matmul(modules::MatmulContext& ctx) const;

    /**
     * @brief Execute SwiGLU activation (forward)
     *
     * Some recipes (e.g., nvfp4-simple) use scaled SwiGLU for numerical
     * stability. This method dispatches to the appropriate implementation.
     *
     * @param ctx SwiGLU context with tensors and dimensions
     */
    virtual void swiglu_forward(modules::SwiGLUContext& ctx) const;

    /**
     * @brief Execute SwiGLU activation (backward)
     *
     * @param ctx SwiGLU context with gradient tensors
     */
    virtual void swiglu_backward(modules::SwiGLUContext& ctx) const;

    // =========================================================================
    // MoE grouped matmul dispatch
    // =========================================================================

    /**
     * @brief Execute forward MoE grouped matmul
     *
     * Dispatches the MoE grouped GEMM based on the recipe's precision.
     * BF16 recipe uses BF16 cuDNN FE, FP8 recipe uses FP8 dequant + MoE GEMM, etc.
     *
     * @param ctx MoE matmul context with all tensors and dimensions
     */
    virtual void forward_moe_matmul(modules::MoeMatmulContext& ctx) const;

    /**
     * @brief Execute backward MoE grouped matmul
     *
     * Computes gradients w.r.t. input (dinp) and optionally weights (dweight).
     * FP8 hybrid recipe quantizes gradients to E5M2 and uses E4M3 weights × E5M2 grads.
     *
     * @param ctx MoE matmul context with gradient tensors (dout, dinp, dweight)
     */
    virtual void backward_moe_matmul(modules::MoeMatmulContext& ctx) const;

    // =========================================================================
    // Run state configuration
    // =========================================================================

    /**
     * @brief Check if this recipe needs FP8 forward quantization buffers
     */
    [[nodiscard]] virtual bool needs_fp8_forward_buffers() const {
        return forward_format() == Format::E4M3 || forward_format() == Format::HYBRID;
    }

    /**
     * @brief Check if this recipe needs FP4 forward quantization buffers
     */
    [[nodiscard]] virtual bool needs_fp4_forward_buffers() const {
        return forward_format() == Format::E2M1;
    }

    /**
     * @brief Check if this recipe needs E5M2 backward gradient buffers
     */
    [[nodiscard]] virtual bool needs_e5m2_backward_buffers() const {
        return backward_format() == Format::E5M2 || forward_format() == Format::HYBRID;
    }

    /**
     * @brief Check if a layer should skip quantization (for FP8/FP4 stability)
     *
     * @param layer_idx Layer index (0-based)
     * @param num_layers Total number of layers
     * @param skip_first Number of first layers to skip
     * @param skip_last Number of last layers to skip
     */
    [[nodiscard]] virtual bool should_skip_layer_quant(
        int layer_idx, int num_layers, int skip_first, int skip_last) const {
        return (layer_idx < skip_first) || (layer_idx >= num_layers - skip_last);
    }
};

}  // namespace recipes

#endif  // SUROGATE_SRC_RECIPES_RECIPE_H
