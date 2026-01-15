// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3MoEConfig - Configuration for Qwen3 Mixture of Experts models.

#ifndef SUROGATE_SRC_MODELS_QWEN3MOE_CONFIG_H
#define SUROGATE_SRC_MODELS_QWEN3MOE_CONFIG_H

#include <algorithm>
#include <vector>

#include "models/qwen3/config.h"

/**
 * @brief Qwen3MoEConfig - Configuration for Qwen3 Mixture of Experts models.
 *
 * Inherits from Qwen3Config and adds MoE-specific fields.
 * Supports hybrid dense/MoE layers via decoder_sparse_step and mlp_only_layers.
 */
struct Qwen3MoEConfig : public Qwen3Config {
    // MoE-specific configuration
    int NumExperts = 0;             ///< Number of routed experts (0 = not MoE)
    int NumExpertsPerTok = 0;       ///< Top-K experts selected per token
    int MoeIntermediateSize = 0;    ///< Per-expert MLP hidden dim (0 = use IntermediateSize)
    int DecoderSparseStep = 1;      ///< MoE layer frequency: MoE every N layers (1 = all MoE)
    std::vector<int> MlpOnlyLayers; ///< Explicit list of layer indices using dense MLP instead of MoE
    bool NormTopkProb = false;      ///< Normalize top-k routing weights to sum to 1
    float RouterAuxLossCoef = 0.001f; ///< Load balancing auxiliary loss coefficient
    float RouterZLossCoef = 0.001f;   ///< Router z-loss (logit regularization) coefficient

    Qwen3MoEConfig() {
        Architecture = QWEN3_MOE;
    }

    [[nodiscard]] std::unique_ptr<PretrainedConfig> clone() const override {
        return std::make_unique<Qwen3MoEConfig>(*this);
    }

    [[nodiscard]] bool is_moe() const override { return NumExperts > 0; }

    /**
     * @brief Check if a specific layer uses MoE routing.
     *
     * A layer is MoE if:
     * 1. NumExperts > 0 (model is MoE)
     * 2. Layer is NOT in mlp_only_layers
     * 3. (layer_idx + 1) % decoder_sparse_step == 0
     */
    [[nodiscard]] bool is_moe_layer(int layer_idx) const override {
        if (NumExperts == 0) return false;

        // Check if this layer is explicitly marked as dense (mlp_only_layers)
        if (std::find(MlpOnlyLayers.begin(), MlpOnlyLayers.end(), layer_idx) != MlpOnlyLayers.end()) {
            return false;
        }

        // Check decoder_sparse_step pattern: MoE if (layer_idx + 1) % step == 0
        // With step=1, all layers are MoE (default behavior)
        return (layer_idx + 1) % DecoderSparseStep == 0;
    }

    /**
     * @brief Get the intermediate size for a specific layer.
     *
     * MoE layers use MoeIntermediateSize, dense layers use IntermediateSize.
     */
    [[nodiscard]] int get_intermediate_size(int layer_idx) const {
        if (is_moe_layer(layer_idx) && MoeIntermediateSize > 0) {
            return MoeIntermediateSize;
        }
        return IntermediateSize;
    }
};

#endif // SUROGATE_SRC_MODELS_QWEN3MOE_CONFIG_H
