// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "recipes/recipe_factory.h"

#include <stdexcept>

#include "recipes/bf16/bf16_recipe.h"
#include "recipes/fp8_hybrid/fp8_hybrid_recipe.h"
#include "recipes/nvfp4/nvfp4_recipe.h"

namespace recipes {

std::unique_ptr<Recipe> RecipeFactory::create(const std::string& name) {
    return create(name, RecipeConfig{});
}

std::unique_ptr<Recipe> RecipeFactory::create(const std::string& name, const RecipeConfig& config) {
    if (name == "bf16") {
        return std::make_unique<BF16Recipe>();
    }

    if (name == "fp8-hybrid" || name == "fp8_hybrid") {
        FP8HybridRecipe::Config fp8_config{.margin = config.fp8_margin,
                                           .amax_history_len = config.fp8_amax_history_len,
                                           .amax_compute_algo = AmaxComputeAlgo::MAX,
                                           .reduce_amax = true};
        return std::make_unique<FP8HybridRecipe>(fp8_config);
    }

    // The single NVFP4 recipe. (The Quartet-II variant was removed: measured slower than
    // this one on the same replay even before weight streaming, 32% slower once NVFP4
    // weight streaming is on -- which it structurally cannot use, needing the BF16 weight
    // to dequantize/transpose/re-Hadamard every backward -- with no fidelity advantage.)
    if (name == "nvfp4") {
        NVFP4Recipe::Config nvfp4_config{.disable_2d_quantization = config.fp4_disable_2d_quantization,
                                         .skip_quant_first_layers = config.skip_quant_first_layers,
                                         .skip_quant_last_layers = config.skip_quant_last_layers,
                                         .backend = config.fp4_backend,
                                         .enable_four_over_six = config.fp4_four_over_six};
        return std::make_unique<NVFP4Recipe>(nvfp4_config);
    }

    throw std::invalid_argument("Unknown recipe: " + name + ". Available recipes: bf16, fp8-hybrid, nvfp4");
}

std::vector<std::string> RecipeFactory::available_recipes() {
    return {"bf16", "fp8-hybrid", "nvfp4"};
}

}  // namespace recipes
