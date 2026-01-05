// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_RECIPES_NVFP4_NVFP4_RECIPE_H
#define SUROGATE_SRC_RECIPES_NVFP4_NVFP4_RECIPE_H

#include "recipes/recipe.h"

namespace recipes {

/**
 * @brief Error metric for Four Over Six adaptive block scaling.
 *
 * Determines how quantization error is measured when selecting between
 * scaling to 4.0 vs 6.0 for each block.
 */
enum class FourOverSixErrorMetric {
    MSE,      ///< Mean squared error (default, best for training)
    L1,       ///< L1 norm (sum of absolute errors)
    AbsMax    ///< Maximum absolute error
};

/**
 * @brief NVFP4 block-scaled recipe for FP4 training.
 *
 * Uses FP4 E2M1 with two-level block scaling for extreme memory efficiency
 * on Blackwell GPUs (SM100+). Implements TransformerEngine's NVFP4BlockScaling recipe.
 *
 * Two-level scaling:
 * - Level 1: FP8 E4M3 scale per 16 consecutive values
 * - Level 2: FP32 global per-tensor scale (amax)
 *
 * The recipe implements three key techniques for narrow-format training:
 * 1. 2D block scaling for weights (16x16 blocks)
 * 2. Stochastic rounding for gradients (avoids quantization bias)
 * 3. Random Hadamard Transform (RHT) for inputs/gradients (spreads outliers)
 *
 * Four Over Six (4/6) Adaptive Block Scaling:
 * Based on "Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling"
 * (arXiv:2512.02010). For each block, evaluates both scaling to max=6.0 and max=4.0,
 * selecting the option with lower quantization error. This improves representation
 * of near-maximal values where FP4's large quantization step (4→6) causes high error.
 *
 * FP4 E2M1 format:
 * - Values: ±{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
 * - Maximum representable value: 6.0
 * - Storage: 2 values packed per byte
 *
 * Configuration:
 * - Forward: FP4 E2M1 activations and weights
 * - Backward: FP4 E2M1 with stochastic rounding
 * - Backend: cuDNN (default) or CUTLASS (via --fp4-backend=cutlass)
 */
class NVFP4Recipe final : public Recipe {
public:
    /**
     * @brief Configuration for NVFP4 recipe.
     */
    struct Config {
        bool disable_rht = false;                ///< Disable Random Hadamard Transform
        bool disable_stochastic_rounding = false; ///< Disable stochastic rounding for gradients
        bool disable_2d_quantization = false;    ///< Use 1D instead of 2D block scaling for weights
        int skip_quant_first_layers = 0;         ///< Skip quantization for first N layers (embedding)
        int skip_quant_last_layers = 0;          ///< Skip quantization for last N layers (lm_head)
        EMatmulBackend backend = EMatmulBackend::CUBLASLT;  ///< cuDNN (CUBLASLT) or CUTLASS

        // Four Over Six (4/6) Adaptive Block Scaling options
        bool enable_four_over_six = true;       ///< Enable 4/6 adaptive block scaling
        FourOverSixErrorMetric four_over_six_metric = FourOverSixErrorMetric::MSE;  ///< Error metric for 4/6 selection
    };

    NVFP4Recipe() : mConfig{} {}
    explicit NVFP4Recipe(Config config) : mConfig(std::move(config)) {}

    [[nodiscard]] bool is_nvfp4() const override { return mConfig.backend != EMatmulBackend::CUTLASS; }
    [[nodiscard]] bool is_nvfp4_cutlass() const override { return mConfig.backend == EMatmulBackend::CUTLASS; }

    [[nodiscard]] Format forward_format() const override { return Format::E2M1; }
    [[nodiscard]] Format backward_format() const override { return Format::E2M1; }

    [[nodiscard]] QuantParams quant_fwd_input() const override {
        return {
            .random_hadamard_transform = !mConfig.disable_rht,
            .stochastic_rounding = false,
            .block_2d_quantization = false,
            .power_2_scale = false,
            .amax_epsilon = 0.0f
        };
    }

    [[nodiscard]] QuantParams quant_fwd_weight() const override {
        return {
            .random_hadamard_transform = false,  // RHT not applied to weights per TE recipe
            .stochastic_rounding = false,
            .block_2d_quantization = !mConfig.disable_2d_quantization,
            .power_2_scale = false,
            .amax_epsilon = 0.0f
        };
    }

    [[nodiscard]] QuantParams quant_bwd_grad() const override {
        return {
            .random_hadamard_transform = !mConfig.disable_rht,
            .stochastic_rounding = !mConfig.disable_stochastic_rounding,
            .block_2d_quantization = false,
            .power_2_scale = false,
            .amax_epsilon = 0.0f
        };
    }

    [[nodiscard]] MatmulParams gemm_fprop() const override {
        return {.use_split_accumulator = true};
    }
    [[nodiscard]] MatmulParams gemm_dgrad() const override {
        return {.use_split_accumulator = true};
    }
    [[nodiscard]] MatmulParams gemm_wgrad() const override {
        return {.use_split_accumulator = true};
    }

    [[nodiscard]] bool requires_block_scales() const override { return true; }
    [[nodiscard]] bool requires_hadamard_workspace() const override { return !mConfig.disable_rht; }

    /// @brief Use scaled SwiGLU when RHT is disabled for FP4 numerical stability
    [[nodiscard]] bool requires_scaled_swiglu() const override { return mConfig.disable_rht; }

    [[nodiscard]] EMatmulBackend matmul_backend() const override { return mConfig.backend; }

    [[nodiscard]] std::string_view name() const override {
        if (mConfig.enable_four_over_six) {
            return mConfig.backend == EMatmulBackend::CUTLASS ? "nvfp4-4o6-cutlass" : "nvfp4-4o6";
        }
        return mConfig.backend == EMatmulBackend::CUTLASS ? "nvfp4-cutlass" : "nvfp4";
    }

    [[nodiscard]] const Config& config() const { return mConfig; }

    /// @brief Check if Four Over Six adaptive block scaling is enabled
    [[nodiscard]] bool uses_four_over_six() const { return mConfig.enable_four_over_six; }

    /// @brief Get the error metric for Four Over Six selection
    [[nodiscard]] FourOverSixErrorMetric four_over_six_metric() const { return mConfig.four_over_six_metric; }

    // =========================================================================
    // Recipe-driven dispatch (Phase 7)
    // =========================================================================

    /// @brief NVFP4Recipe handles forward matmul dispatch for both cuDNN and CUTLASS paths
    [[nodiscard]] bool handles_forward_matmul() const override {
        return true;  // Handles both CUBLASLT (cuDNN) and CUTLASS backends
    }

    /// @brief NVFP4Recipe handles backward matmul with FP4 quantization for both backends
    [[nodiscard]] bool handles_backward_matmul() const override {
        return true;  // Handles both CUBLASLT (cuDNN) and CUTLASS backends
    }

    // =========================================================================
    // Matmul dispatch overrides
    // =========================================================================

    /**
     * @brief FP4 cuDNN forward matmul with Hadamard transform support
     *
     * Quantizes input to NVFP4 E2M1 with cuDNN scale layout and optional RHT,
     * then performs FP4 x FP4 GEMM via cuDNN.
     */
    void forward_matmul(modules::MatmulContext& ctx) const override;

    /**
     * @brief FP4 cuDNN backward matmul with stochastic rounding
     *
     * Computes gradients using NVFP4 E2M1 with stochastic rounding for gradients
     * and optional Random Hadamard Transform for weight gradient computation.
     * - dinp = dout @ W (FP4 quantized)
     * - dweight = inp^T @ dout (FP4 quantized with RHT for wgrad)
     */
    void backward_matmul(modules::MatmulContext& ctx) const override;

    // =========================================================================
    // SwiGLU overrides for scaled variant (when RHT is disabled)
    // =========================================================================

    /**
     * @brief Scaled SwiGLU forward for FP4 numerical stability (when RHT disabled)
     *
     * Normalizes gate values to [-1, 1] range and saves scale for later use.
     * Only active when disable_rht is true.
     */
    void swiglu_forward(modules::SwiGLUContext& ctx) const override;

    /**
     * @brief Scaled SwiGLU backward (when RHT disabled)
     */
    void swiglu_backward(modules::SwiGLUContext& ctx) const override;

private:
    Config mConfig;

    // CUTLASS backend implementations
    void forward_matmul_cutlass(modules::MatmulContext& ctx) const;
    void backward_matmul_cutlass(modules::MatmulContext& ctx) const;
};

}  // namespace recipes

#endif  // SUROGATE_SRC_RECIPES_NVFP4_NVFP4_RECIPE_H
