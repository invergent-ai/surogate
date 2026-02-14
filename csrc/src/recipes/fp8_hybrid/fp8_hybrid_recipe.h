// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_RECIPES_FP8_HYBRID_FP8_HYBRID_RECIPE_H
#define SUROGATE_SRC_RECIPES_FP8_HYBRID_FP8_HYBRID_RECIPE_H

#include "recipes/recipe.h"

namespace recipes {

/**
 * @brief Amax computation algorithm for delayed scaling.
 */
enum class AmaxComputeAlgo {
    MAX,          ///< Use maximum amax from history window
    MOST_RECENT   ///< Use most recent amax value
};

/**
 * @brief FP8 HYBRID recipe with delayed scaling.
 *
 * Uses E4M3 for forward pass activations/weights and E5M2 for backward pass gradients.
 * Implements TransformerEngine's DelayedScaling strategy.
 *
 * Delayed scaling uses scale factors from the previous iteration for stability,
 * and records amax history for computing scales for the next iteration.
 *
 * Configuration:
 * - Forward: FP8 E4M3 with per-tensor scaling
 * - Backward: FP8 E5M2 with per-tensor scaling
 * - Scale factor: Computed from amax history using margin
 */
class FP8HybridRecipe final : public Recipe {
public:
    /**
     * @brief Configuration for FP8 hybrid recipe.
     */
    struct Config {
        int margin = 0;                              ///< Margin for scale factor computation
        int amax_history_len = 1024;                 ///< Length of amax history window
        AmaxComputeAlgo amax_compute_algo = AmaxComputeAlgo::MAX;  ///< Algorithm for amax selection
        bool reduce_amax = true;                     ///< Reduce amax across distributed group
    };

    FP8HybridRecipe() : mConfig{} {}
    explicit FP8HybridRecipe(Config config) : mConfig(std::move(config)) {}

    [[nodiscard]] bool is_fp8_hybrid() const override { return true; }

    [[nodiscard]] Format forward_format() const override { return Format::E4M3; }
    [[nodiscard]] Format backward_format() const override { return Format::E5M2; }

    [[nodiscard]] QuantParams quant_fwd_input() const override {
        return {
            .random_hadamard_transform = false,
            .stochastic_rounding = false,
            .block_2d_quantization = false,
            .power_2_scale = false,
            .amax_epsilon = 0.0f
        };
    }

    [[nodiscard]] QuantParams quant_fwd_weight() const override {
        return {
            .random_hadamard_transform = false,
            .stochastic_rounding = false,
            .block_2d_quantization = false,
            .power_2_scale = false,
            .amax_epsilon = 0.0f
        };
    }

    [[nodiscard]] QuantParams quant_bwd_grad() const override {
        return {
            .random_hadamard_transform = false,
            .stochastic_rounding = false,
            .block_2d_quantization = false,
            .power_2_scale = false,
            .amax_epsilon = 0.0f
        };
    }

    [[nodiscard]] MatmulParams gemm_fprop() const override {
        return {.use_split_accumulator = false};
    }
    [[nodiscard]] MatmulParams gemm_dgrad() const override {
        return {.use_split_accumulator = true};
    }
    [[nodiscard]] MatmulParams gemm_wgrad() const override {
        return {.use_split_accumulator = true};
    }

    [[nodiscard]] bool requires_amax_history() const override { return true; }
    [[nodiscard]] int amax_history_len() const override { return mConfig.amax_history_len; }

    [[nodiscard]] std::string_view name() const override { return "fp8-hybrid"; }

    [[nodiscard]] const Config& config() const { return mConfig; }

    // =========================================================================
    // Matmul dispatch overrides
    // =========================================================================

    /**
     * @brief FP8 forward matmul with delayed scaling support
     *
     * Quantizes input to E4M3, handles cached or on-the-fly weight quantization,
     * and performs FP8 x FP8 matmul via cuBLASLt.
     */
    void forward_matmul(modules::MatmulContext& ctx) const override;

    /**
     * @brief FP8 backward matmul with E5M2 gradient quantization
     *
     * Computes:
     * - dinp = W^T @ dout (E4M3 weight × E5M2 gradient)
     * - dweight = inp^T @ dout (E4M3 activation × E5M2 gradient)
     * - dbias (optional)
     */
    void backward_matmul(modules::MatmulContext& ctx) const override;

    /**
     * @brief FP8 MoE grouped matmul (forward)
     *
     * Implements full FP8 training support for MoE:
     * 1. Pre-quantized FP8 weights (WoQ via cuDNN FE) - most efficient
     * 2. Full FP8 training (quantize activations to E4M3)
     * 3. BF16 fallback
     */
    void forward_moe_matmul(modules::MoeMatmulContext& ctx) const override;

    /**
     * @brief FP8 MoE grouped matmul (backward)
     *
     * Quantizes upstream gradients to E5M2 and computes:
     * - dinp = weights^T @ dout (E4M3 weights × E5M2 gradients)
     *
     * Weight gradients computed separately by DSL op dispatchers.
     */
    void backward_moe_matmul(modules::MoeMatmulContext& ctx) const override;

private:
    Config mConfig;
};

}  // namespace recipes

#endif  // SUROGATE_SRC_RECIPES_FP8_HYBRID_FP8_HYBRID_RECIPE_H
