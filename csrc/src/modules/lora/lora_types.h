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
 * @brief Type trait to detect if a weights struct has experts
 */
template<typename T>
struct has_experts {
    template<typename U> static auto test(U* p) -> decltype(p->experts, std::true_type());
    template<typename U> static auto test(...) -> std::false_type;
    static constexpr bool value = decltype(test<T>(nullptr))::value;
};

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
 * Supports two layouts:
 * 1. Separate: std::vector<LoRAExpertWeights> - used for sequential expert execution
 * 2. Grouped: single tensors with expert dimension - used for high-performance grouped GEMM
 */
template<typename TTensor>
struct LoRAGroupedLayerWeights {
    TTensor A;  ///< (num_experts, rank, in_features)
    TTensor B;  ///< (num_experts, out_features, rank)

    [[nodiscard]] bool has_value() const { return A.Data != nullptr; }
};

/**
 * @brief Grouped LoRA weights for MoE experts
 */
template<typename TTensor>
struct LoRAGroupedExpertWeights {
    std::optional<LoRAGroupedLayerWeights<TTensor>> gate;  ///< Gate projection LoRA
    std::optional<LoRAGroupedLayerWeights<TTensor>> up;    ///< Up projection LoRA
    std::optional<LoRAGroupedLayerWeights<TTensor>> down;  ///< Down projection LoRA

    [[nodiscard]] bool has_any() const {
        return (gate.has_value() && gate->has_value()) ||
               (up.has_value() && up->has_value()) ||
               (down.has_value() && down->has_value());
    }
};

template<typename TTensor>
struct LoRAMoEWeights {
    // Layout 1: Separate (Sequential)
    std::vector<LoRAExpertWeights<TTensor>> experts;

    // Layout 2: Grouped (Batched)
    LoRAGroupedExpertWeights<TTensor> grouped;

    bool use_grouped = false;  ///< Whether to use the grouped layout

    [[nodiscard]] bool has_any() const {
        if (use_grouped) {
            return grouped.has_any();
        }
        for (const auto& expert : experts) {
            if (expert.has_any()) return true;
        }
        return false;
    }

    [[nodiscard]] int num_experts() const {
        if (use_grouped) {
            if (grouped.gate.has_value()) return grouped.gate->A.Sizes[0];
            if (grouped.up.has_value()) return grouped.up->A.Sizes[0];
            if (grouped.down.has_value()) return grouped.down->A.Sizes[0];
        }
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
