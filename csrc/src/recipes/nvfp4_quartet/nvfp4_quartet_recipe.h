// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Based on Quartet-II by IST Austria (Erik Schultheis et al.)
// SPDX-License-Identifier: Apache-2.0

#ifndef SUROGATE_SRC_RECIPES_NVFP4_QUARTET_NVFP4_QUARTET_RECIPE_H
#define SUROGATE_SRC_RECIPES_NVFP4_QUARTET_NVFP4_QUARTET_RECIPE_H

#include "recipes/recipe.h"

namespace recipes {

/**
 * @brief NVFP4 Quartet-II recipe with EDEN quantization for unbiased FP4 training.
 *
 * Implements the Quartet-II algorithm from "Quartet II: Accurate LLM Pre-Training
 * in NVFP4 with Improved Unbiased Gradient Estimation" (arXiv:2601.22813).
 *
 * Key innovations over standard NVFP4:
 *
 * 1. **EDEN (Efficient Dequantization-based Error Normalization)**:
 *    - Computes correction factor per Hadamard group: correction = Σ(x²) / Σ(x·q)
 *    - Applies correction to scales, making gradients approximately unbiased
 *    - Achieves 2x lower quantization error than standard stochastic rounding
 *
 * 2. **Stochastic Rounding on Scales Only**:
 *    - FP4 values use RTN (round-to-nearest), NOT stochastic rounding
 *    - Stochastic rounding applied to E4M3 block scales only
 *    - Maintains unbiasedness while improving FP4 precision
 *
 * 3. **Hadamard Re-randomization**:
 *    - Per-backward-pass sign flips on Hadamard matrix columns
 *    - Same rerotated Hadamard reused within backward (EW and EtX)
 *    - Prevents systematic quantization bias across training steps
 *
 * 4. **128-element Hadamard Groups**:
 *    - Uses 128×128 Hadamard transform (vs 16 in standard NVFP4)
 *    - Better decorrelation of values for FP4 representability
 *
 * Configuration:
 * - Forward: RTN quantization with optional 4/6 adaptive scaling
 * - Backward: EDEN quantization with SR on scales
 * - Backend: CUTLASS optimized for Blackwell GPUs (SM100+)
 */
class NVFP4QuartetRecipe final : public Recipe {
public:
    /**
     * @brief Configuration for NVFP4 Quartet recipe.
     */
    struct Config {
        /// @brief Backend for FP4 matmul (CUTLASS recommended for Quartet-II)
        EMatmulBackend backend = EMatmulBackend::CUTLASS;

        /// @brief Forward pass scale override (1.0 = standard, no modification)
        float forward_scale_override = 1.0f;

        /// @brief Backward pass scale override ((17/16) * 0.93 ≈ 1.054 per paper)
        float backward_scale_override = (17.0f / 16.0f) * 0.93f;

        /// @brief Maximum scale value for EDEN (255.99 leaves room for correction)
        float scales_max = 255.99f;

        /// @brief Hadamard transform dimension (128 for Quartet-II)
        int hadamard_dim = 128;

        /// @brief Skip quantization for first N layers (embedding)
        int skip_quant_first_layers = 0;

        /// @brief Skip quantization for last N layers (lm_head)
        int skip_quant_last_layers = 0;
    };

    NVFP4QuartetRecipe() : mConfig{} {}
    explicit NVFP4QuartetRecipe(Config config) : mConfig(std::move(config)) {}

    // =========================================================================
    // Type checking
    // =========================================================================

    [[nodiscard]] bool is_nvfp4() const override { return false; }  // Different from standard NVFP4
    [[nodiscard]] bool is_nvfp4_cutlass() const override { return true; }  // Uses CUTLASS backend
    [[nodiscard]] bool is_nvfp4_quartet() const { return true; }

    // =========================================================================
    // Format specification
    // =========================================================================

    [[nodiscard]] Format forward_format() const override { return Format::E2M1; }
    [[nodiscard]] Format backward_format() const override { return Format::E2M1; }

    // =========================================================================
    // Quantization parameters
    // =========================================================================

    [[nodiscard]] QuantParams quant_fwd_input() const override {
        return {
            .random_hadamard_transform = false,   // Forward uses standard RTN quantization (no Hadamard/EDEN)
            .stochastic_rounding = false,         // RTN for FP4 values
            .block_2d_quantization = false,
            .power_2_scale = false,
            .amax_epsilon = 0.0f
        };
    }

    [[nodiscard]] QuantParams quant_fwd_weight() const override {
        return {
            .random_hadamard_transform = false,   // No RHT for weights (per TE recipe)
            .stochastic_rounding = false,         // RTN for weights
            .block_2d_quantization = false,
            .power_2_scale = false,
            .amax_epsilon = 0.0f
        };
    }

    [[nodiscard]] QuantParams quant_bwd_grad() const override {
        return {
            .random_hadamard_transform = true,    // Re-randomized Hadamard per backward pass
            .stochastic_rounding = false,         // KEY: RTN for FP4 values, SR on scales only
            .block_2d_quantization = false,
            .power_2_scale = false,
            .amax_epsilon = 0.0f
        };
    }

    // =========================================================================
    // Matmul configuration
    // =========================================================================

    [[nodiscard]] MatmulParams gemm_fprop() const override {
        return {.use_split_accumulator = true};
    }
    [[nodiscard]] MatmulParams gemm_dgrad() const override {
        return {.use_split_accumulator = true};
    }
    [[nodiscard]] MatmulParams gemm_wgrad() const override {
        return {.use_split_accumulator = true};
    }

    // =========================================================================
    // State requirements
    // =========================================================================

    [[nodiscard]] bool requires_block_scales() const override { return true; }
    [[nodiscard]] bool requires_hadamard_workspace() const override { return true; }

    [[nodiscard]] EMatmulBackend matmul_backend() const override { return mConfig.backend; }

    // =========================================================================
    // Recipe metadata
    // =========================================================================

    [[nodiscard]] std::string_view name() const override { return "nvfp4-quartet"; }

    [[nodiscard]] const Config& config() const { return mConfig; }

    /// @brief Get Hadamard dimension (128 for Quartet-II)
    [[nodiscard]] int hadamard_dim() const { return mConfig.hadamard_dim; }

    /// @brief Get forward scale override
    [[nodiscard]] float forward_scale_override() const { return mConfig.forward_scale_override; }

    /// @brief Get backward scale override (~1.054 for EDEN)
    [[nodiscard]] float backward_scale_override() const { return mConfig.backward_scale_override; }

    /// @brief Get maximum scale value for EDEN
    [[nodiscard]] float scales_max() const { return mConfig.scales_max; }

    // =========================================================================
    // Matmul dispatch overrides
    // =========================================================================

    /**
     * @brief Forward matmul with Quartet-II quantization
     *
     * Uses RTN quantization for FP4 values (not stochastic rounding).
     * Forward uses standard NVFP4-style quantization (no Hadamard/EDEN).
     */
    void forward_matmul(modules::MatmulContext& ctx) const override;

    /**
     * @brief Backward matmul with EDEN quantization
     *
     * Key EDEN features:
     * - RTN for FP4 values
     * - EDEN correction factor per 128-element group
     * - Stochastic rounding on E4M3 scales only
     * - Re-randomized Hadamard for each backward pass
     */
    void backward_matmul(modules::MatmulContext& ctx) const override;

private:
    Config mConfig;
};

}  // namespace recipes

#endif  // SUROGATE_SRC_RECIPES_NVFP4_QUARTET_NVFP4_QUARTET_RECIPE_H
