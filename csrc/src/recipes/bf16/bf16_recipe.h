// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_RECIPES_BF16_BF16_RECIPE_H
#define SUROGATE_SRC_RECIPES_BF16_BF16_RECIPE_H

#include "recipes/recipe.h"

namespace recipes {

/**
 * @brief BF16 baseline recipe - no quantization.
 *
 * Uses BF16 for all matmul operations. This is the default recipe
 * for maximum accuracy when memory/compute are not constrained.
 *
 * Configuration:
 * - Forward: BF16 activations and weights
 * - Backward: BF16 gradients
 * - No quantization or special scaling
 */
class BF16Recipe final : public Recipe {
public:
    [[nodiscard]] bool is_bf16() const override { return true; }

    [[nodiscard]] Format forward_format() const override { return Format::BF16; }
    [[nodiscard]] Format backward_format() const override { return Format::BF16; }

    [[nodiscard]] QuantParams quant_fwd_input() const override { return {}; }
    [[nodiscard]] QuantParams quant_fwd_weight() const override { return {}; }
    [[nodiscard]] QuantParams quant_bwd_grad() const override { return {}; }

    [[nodiscard]] MatmulParams gemm_fprop() const override {
        return {.use_split_accumulator = false};
    }
    [[nodiscard]] MatmulParams gemm_dgrad() const override {
        return {.use_split_accumulator = false};
    }
    [[nodiscard]] MatmulParams gemm_wgrad() const override {
        return {.use_split_accumulator = false};
    }

    [[nodiscard]] std::string_view name() const override { return "bf16"; }
};

}  // namespace recipes

#endif  // SUROGATE_SRC_RECIPES_BF16_BF16_RECIPE_H
