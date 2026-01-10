// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_TYPES_H
#define SUROGATE_SRC_MODULES_LORA_LORA_TYPES_H

#include <optional>
#include <vector>

#include "lora_config.h"

namespace modules {

/**
 * @brief LoRA weights for a single linear layer: W' = W + scaling * B @ A
 *
 * A is (rank, in_features) - initialized with Kaiming uniform
 * B is (out_features, rank) - initialized with zeros
 */
template<typename TTensor>
struct LoRALayerWeights {
    TTensor A;  ///< (rank, in_features)
    TTensor B;  ///< (out_features, rank)

    [[nodiscard]] bool has_value() const { return A.Data != nullptr; }
};

/**
 * @brief LoRA weights for attention projections
 */
template<typename TTensor>
struct LoRAAttentionWeights {
    std::optional<LoRALayerWeights<TTensor>> q;  ///< Query projection
    std::optional<LoRALayerWeights<TTensor>> k;  ///< Key projection
    std::optional<LoRALayerWeights<TTensor>> v;  ///< Value projection
    std::optional<LoRALayerWeights<TTensor>> o;  ///< Output projection
};

/**
 * @brief LoRA weights for MLP projections
 */
template<typename TTensor>
struct LoRAMLPWeights {
    std::optional<LoRALayerWeights<TTensor>> gate;  ///< Gate projection
    std::optional<LoRALayerWeights<TTensor>> up;    ///< Up projection
    std::optional<LoRALayerWeights<TTensor>> down;  ///< Down projection
};

/**
 * @brief LoRA weights for a single MoE expert
 *
 * Each expert has its own independent LoRA adapters for gate, up, and down projections.
 * This enables per-expert fine-tuning in MoE models.
 */
template<typename TTensor>
struct LoRAExpertWeights {
    std::optional<LoRALayerWeights<TTensor>> gate;  ///< Gate projection LoRA
    std::optional<LoRALayerWeights<TTensor>> up;    ///< Up projection LoRA
    std::optional<LoRALayerWeights<TTensor>> down;  ///< Down projection LoRA

    [[nodiscard]] bool has_any() const {
        return (gate.has_value() && gate->has_value()) ||
               (up.has_value() && up->has_value()) ||
               (down.has_value() && down->has_value());
    }
};

/**
 * @brief LoRA weights for all experts in a MoE block
 *
 * Manages per-expert LoRA adapters for MoE transformer blocks.
 * Each expert can have independent LoRA weights.
 */
template<typename TTensor>
struct LoRAMoEWeights {
    std::vector<LoRAExpertWeights<TTensor>> experts;  ///< Per-expert LoRA weights

    [[nodiscard]] bool has_any() const {
        for (const auto& expert : experts) {
            if (expert.has_any()) return true;
        }
        return false;
    }

    [[nodiscard]] int num_experts() const {
        return static_cast<int>(experts.size());
    }
};

/**
 * @brief LoRA weights for a transformer block
 */
template<typename TTensor>
struct LoRABlockWeights {
    LoRAAttentionWeights<TTensor> attention;
    LoRAMLPWeights<TTensor> mlp;       ///< For dense models
    LoRAMoEWeights<TTensor> moe;       ///< For MoE models (per-expert LoRA)
};

/**
 * @brief Complete LoRA adapter weights
 */
template<typename TTensor>
struct LoRAWeightsSet {
    std::vector<LoRABlockWeights<TTensor>> blocks;
    ModularLoRAConfig config;
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_TYPES_H
