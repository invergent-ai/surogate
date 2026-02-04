// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_RECIPES_RECIPE_FACTORY_H
#define SUROGATE_SRC_RECIPES_RECIPE_FACTORY_H

#include <memory>
#include <string>
#include <vector>

#include "runtime/training/matmul_backend.h"  // For EMatmulBackend
#include "recipes/recipe.h"

namespace recipes {

/**
 * @brief Generic recipe configuration parsed from CLI arguments.
 *
 * Contains all possible recipe options. Each recipe only uses the fields
 * relevant to its configuration.
 */
struct RecipeConfig {
    // FP8 options (for fp8-hybrid)
    int fp8_amax_history_len = 1024;
    int fp8_margin = 0;

    // FP4 options (for nvfp4)
    bool fp4_disable_2d_quantization = false;
    int skip_quant_first_layers = 0;  ///< Skip quantization for first N layers (embedding layers)
    int skip_quant_last_layers = 0;   ///< Skip quantization for last N layers (lm_head layers)
    EMatmulBackend fp4_backend = EMatmulBackend::CUBLASLT;  ///< FP4 matmul backend (CUBLASLT=cuDNN, CUTLASS)
};

/**
 * @brief Factory for creating training recipes.
 *
 * Creates recipe instances from string names (for CLI) or with explicit configuration.
 */
class RecipeFactory {
public:
    /**
     * @brief Create a recipe from its name with default configuration.
     *
     * @param name Recipe name: "bf16", "fp8-hybrid", "nvfp4"
     * @return Unique pointer to the created recipe
     * @throws std::invalid_argument if name is not recognized
     */
    static std::unique_ptr<Recipe> create(const std::string& name);

    /**
     * @brief Create a recipe from its name with custom configuration.
     *
     * @param name Recipe name
     * @param config Generic configuration (recipe extracts relevant fields)
     * @return Unique pointer to the created recipe
     * @throws std::invalid_argument if name is not recognized
     */
    static std::unique_ptr<Recipe> create(const std::string& name, const RecipeConfig& config);

    /**
     * @brief Get list of all available recipe names.
     *
     * @return Vector of recipe name strings
     */
    static std::vector<std::string> available_recipes();
};

}  // namespace recipes

#endif  // SUROGATE_SRC_RECIPES_RECIPE_FACTORY_H
