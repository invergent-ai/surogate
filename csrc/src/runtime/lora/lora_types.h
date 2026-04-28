// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_TYPES_H
#define SUROGATE_SRC_MODULES_LORA_LORA_TYPES_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

#include "lora_config.h"

namespace modules {

/**
 * @brief Type trait to detect if a weights struct has experts
 */
template <typename T>
struct has_experts {
    template <typename U>
    static auto test(U* p) -> decltype(p->experts, std::true_type());
    template <typename U>
    static auto test(...) -> std::false_type;
    static constexpr bool value = decltype(test<T>(nullptr))::value;
};

/**
 * @brief LoRA weights for a single linear layer: W' = W + scaling * B @ A
 *
 * A is (rank, in_features) - initialized with Kaiming uniform
 * B is (out_features, rank) - initialized with zeros
 */
template <typename TTensor>
struct LoRALayerWeights {
    TTensor A;  ///< (rank, in_features)
    TTensor B;  ///< (out_features, rank)

    [[nodiscard]] bool has_value() const {
        return A.Data != nullptr;
    }
};

/**
 * @brief LoRA weights for attention projections
 */
template <typename TTensor>
struct LoRAAttentionWeights {
    std::optional<LoRALayerWeights<TTensor>> q;  ///< Query projection
    std::optional<LoRALayerWeights<TTensor>> k;  ///< Key projection
    std::optional<LoRALayerWeights<TTensor>> v;  ///< Value projection
    std::optional<LoRALayerWeights<TTensor>> o;  ///< Output projection
};

/**
 * @brief LoRA weights for MLP projections
 */
template <typename TTensor>
struct LoRAMLPWeights {
    std::optional<LoRALayerWeights<TTensor>> gate;     ///< Gate projection
    std::optional<LoRALayerWeights<TTensor>> gate_up;  ///< Fused gate+up projection
    std::optional<LoRALayerWeights<TTensor>> up;       ///< Up projection
    std::optional<LoRALayerWeights<TTensor>> down;     ///< Down projection

    [[nodiscard]] bool has_any() const {
        return (gate.has_value() && gate->has_value()) || (gate_up.has_value() && gate_up->has_value()) ||
               (up.has_value() && up->has_value()) || (down.has_value() && down->has_value());
    }
};

/**
 * @brief LoRA weights for a single MoE expert
 *
 * Each expert has its own independent LoRA adapters for gate, up, and down projections.
 * This enables per-expert fine-tuning in MoE models.
 */
template <typename TTensor>
struct LoRAExpertWeights {
    std::optional<LoRALayerWeights<TTensor>> gate;     ///< Gate projection LoRA
    std::optional<LoRALayerWeights<TTensor>> gate_up;  ///< Fused gate+up projection LoRA
    std::optional<LoRALayerWeights<TTensor>> up;       ///< Up projection LoRA
    std::optional<LoRALayerWeights<TTensor>> down;     ///< Down projection LoRA

    [[nodiscard]] bool has_any() const {
        return (gate.has_value() && gate->has_value()) || (gate_up.has_value() && gate_up->has_value()) ||
               (up.has_value() && up->has_value()) || (down.has_value() && down->has_value());
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
template <typename TTensor>
struct LoRAGroupedLayerWeights {
    TTensor A;  ///< (num_experts, rank, in_features)
    TTensor B;  ///< (num_experts, out_features, rank)

    [[nodiscard]] bool has_value() const {
        return A.Data != nullptr;
    }
};

/**
 * @brief Grouped LoRA weights for MoE experts
 */
template <typename TTensor>
struct LoRAGroupedExpertWeights {
    std::optional<LoRAGroupedLayerWeights<TTensor>> gate;     ///< Gate projection LoRA
    std::optional<LoRAGroupedLayerWeights<TTensor>> gate_up;  ///< Fused gate+up projection LoRA
    std::optional<LoRAGroupedLayerWeights<TTensor>> up;       ///< Up projection LoRA
    std::optional<LoRAGroupedLayerWeights<TTensor>> down;     ///< Down projection LoRA

    [[nodiscard]] bool has_any() const {
        return (gate.has_value() && gate->has_value()) || (gate_up.has_value() && gate_up->has_value()) ||
               (up.has_value() && up->has_value()) || (down.has_value() && down->has_value());
    }
};

template <typename TTensor>
struct LoRAMoEWeights {
    // Optional shared expert (Nemotron/DeepSeek)
    std::optional<LoRAMLPWeights<TTensor>> shared;

    // Layout 1: Separate (Sequential)
    std::vector<LoRAExpertWeights<TTensor>> experts;

    // Layout 2: Grouped (Batched)
    LoRAGroupedExpertWeights<TTensor> grouped;

    bool use_grouped = false;  ///< Whether to use the grouped layout

    [[nodiscard]] bool has_any() const {
        if (shared.has_value() && shared->has_any()) {
            return true;
        }
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
template <typename TTensor>
struct LoRABlockWeights {
    LoRAAttentionWeights<TTensor> attention;
    LoRAMLPWeights<TTensor> mlp;  ///< For dense models
    LoRAMoEWeights<TTensor> moe;  ///< For MoE models (per-expert LoRA)
    std::optional<LoRALayerWeights<TTensor>>
        router;  ///< Router gate LoRA for MoE (when train_router enabled) - PEFT-compatible
};

/**
 * @brief Complete LoRA adapter weights
 */
template <typename TTensor>
struct LoRAWeightsSet {
    std::vector<LoRABlockWeights<TTensor>> blocks;
    ModularLoRAConfig config;
};

/// @brief Canonical ID for each DSL-declared LoRA target.
///
/// The Python DSL declares LoRA targets by semantic name. The graph
/// compiler resolves the name to a ``LoRATargetId`` once (at IR load
/// time), so hot-path dispatch indexes the block storage by enum value
/// rather than comparing 15 strings per slice per matmul.
///
/// ``Unknown`` is the sentinel for future names not yet listed here. It
/// is still stored (with its raw name) on the slice so dropout-seeding
/// stays deterministic, but it won't resolve to any current adapter
/// storage slot.
enum class LoRATargetId : std::uint8_t {
    Q = 0,
    K,
    V,
    O,
    Up,
    Gate,
    Down,
    SharedUp,
    SharedDown,
    SharedGate,
    Router,
    GateUp,
    ExpertGate,
    ExpertUp,
    ExpertGateUp,
    ExpertDown,
    Unknown = 255,
};

inline LoRATargetId lora_target_from_name(std::string_view name) {
    if (name == "q") return LoRATargetId::Q;
    if (name == "k") return LoRATargetId::K;
    if (name == "v") return LoRATargetId::V;
    if (name == "o") return LoRATargetId::O;
    if (name == "up") return LoRATargetId::Up;
    if (name == "gate") return LoRATargetId::Gate;
    if (name == "down") return LoRATargetId::Down;
    if (name == "shared_up") return LoRATargetId::SharedUp;
    if (name == "shared_down") return LoRATargetId::SharedDown;
    if (name == "shared_gate") return LoRATargetId::SharedGate;
    if (name == "router") return LoRATargetId::Router;
    if (name == "gate_up") return LoRATargetId::GateUp;
    if (name == "expert_gate") return LoRATargetId::ExpertGate;
    if (name == "expert_up") return LoRATargetId::ExpertUp;
    if (name == "expert_gate_up") return LoRATargetId::ExpertGateUp;
    if (name == "expert_down") return LoRATargetId::ExpertDown;
    return LoRATargetId::Unknown;
}

template <typename TTensor>
inline LoRALayerWeights<TTensor>* get_layer_weight_by_target(LoRABlockWeights<TTensor>& block, LoRATargetId id) {
    switch (id) {
        case LoRATargetId::Q: return block.attention.q.has_value() ? &*block.attention.q : nullptr;
        case LoRATargetId::K: return block.attention.k.has_value() ? &*block.attention.k : nullptr;
        case LoRATargetId::V: return block.attention.v.has_value() ? &*block.attention.v : nullptr;
        case LoRATargetId::O: return block.attention.o.has_value() ? &*block.attention.o : nullptr;
        case LoRATargetId::Gate: return block.mlp.gate.has_value() ? &*block.mlp.gate : nullptr;
        case LoRATargetId::Up: return block.mlp.up.has_value() ? &*block.mlp.up : nullptr;
        case LoRATargetId::Down: return block.mlp.down.has_value() ? &*block.mlp.down : nullptr;
        case LoRATargetId::GateUp: return block.mlp.gate_up.has_value() ? &*block.mlp.gate_up : nullptr;
        case LoRATargetId::Router: return block.router.has_value() ? &*block.router : nullptr;
        case LoRATargetId::SharedUp:
            return (block.moe.shared.has_value() && block.moe.shared->up.has_value()) ? &*block.moe.shared->up
                                                                                      : nullptr;
        case LoRATargetId::SharedDown:
            return (block.moe.shared.has_value() && block.moe.shared->down.has_value()) ? &*block.moe.shared->down
                                                                                        : nullptr;
        case LoRATargetId::SharedGate:
            return (block.moe.shared.has_value() && block.moe.shared->gate.has_value()) ? &*block.moe.shared->gate
                                                                                        : nullptr;
        default: return nullptr;
    }
}

template <typename TTensor>
inline LoRAGroupedLayerWeights<TTensor>* get_grouped_weight_by_target(LoRABlockWeights<TTensor>& block,
                                                                      LoRATargetId id) {
    if (!block.moe.use_grouped) return nullptr;
    auto& g = block.moe.grouped;
    switch (id) {
        case LoRATargetId::ExpertGate: return g.gate.has_value() ? &*g.gate : nullptr;
        case LoRATargetId::ExpertUp: return g.up.has_value() ? &*g.up : nullptr;
        case LoRATargetId::ExpertGateUp: return g.gate_up.has_value() ? &*g.gate_up : nullptr;
        case LoRATargetId::ExpertDown: return g.down.has_value() ? &*g.down : nullptr;
        default: return nullptr;
    }
}

template <typename TTensor>
inline LoRALayerWeights<TTensor>* get_expert_weight_by_target(LoRAExpertWeights<TTensor>& expert, LoRATargetId id) {
    switch (id) {
        case LoRATargetId::ExpertGate: return expert.gate.has_value() ? &*expert.gate : nullptr;
        case LoRATargetId::ExpertUp: return expert.up.has_value() ? &*expert.up : nullptr;
        case LoRATargetId::ExpertGateUp: return expert.gate_up.has_value() ? &*expert.gate_up : nullptr;
        case LoRATargetId::ExpertDown: return expert.down.has_value() ? &*expert.down : nullptr;
        default: return nullptr;
    }
}

inline constexpr std::array<LoRATargetId, 8> kBaseLoRALayerTargets = {
    LoRATargetId::Q,
    LoRATargetId::K,
    LoRATargetId::V,
    LoRATargetId::O,
    LoRATargetId::Gate,
    LoRATargetId::GateUp,
    LoRATargetId::Up,
    LoRATargetId::Down,
};

inline constexpr std::array<LoRATargetId, 4> kExpertLoRALayerTargets = {
    LoRATargetId::ExpertGate,
    LoRATargetId::ExpertGateUp,
    LoRATargetId::ExpertUp,
    LoRATargetId::ExpertDown,
};

inline constexpr std::array<LoRATargetId, 2> kSharedLoRALayerTargets = {
    LoRATargetId::SharedUp,
    LoRATargetId::SharedDown,
};

inline constexpr bool lora_target_is_expert(LoRATargetId id) {
    return id == LoRATargetId::ExpertGate || id == LoRATargetId::ExpertGateUp || id == LoRATargetId::ExpertUp ||
           id == LoRATargetId::ExpertDown;
}

template <typename TTensor, typename Fn>
inline void for_each_lora_layer_weight(LoRABlockWeights<TTensor>& block, Fn&& fn) {
    for (LoRATargetId id : kBaseLoRALayerTargets) {
        if (auto* layer = get_layer_weight_by_target(block, id)) {
            fn(id, *layer);
        }
    }

    if (block.moe.use_grouped) {
        for (LoRATargetId id : kExpertLoRALayerTargets) {
            if (auto* layer = get_grouped_weight_by_target(block, id)) {
                fn(id, *layer);
            }
        }
    } else {
        for (auto& expert : block.moe.experts) {
            for (LoRATargetId id : kExpertLoRALayerTargets) {
                if (auto* layer = get_expert_weight_by_target(expert, id)) {
                    fn(id, *layer);
                }
            }
        }
    }

    for (LoRATargetId id : kSharedLoRALayerTargets) {
        if (auto* layer = get_layer_weight_by_target(block, id)) {
            fn(id, *layer);
        }
    }

    if (auto* layer = get_layer_weight_by_target(block, LoRATargetId::Router)) {
        fn(LoRATargetId::Router, *layer);
    }
}

template <typename TWeightTensor, typename TGradTensor, typename Fn>
inline void for_each_lora_layer_weight_pair(LoRABlockWeights<TWeightTensor>& weights,
                                            LoRABlockWeights<TGradTensor>& grads,
                                            Fn&& fn) {
    auto visit_pair = [&](LoRATargetId id) {
        auto* w = get_layer_weight_by_target(weights, id);
        auto* g = get_layer_weight_by_target(grads, id);
        if (w && g) {
            fn(id, *w, *g);
        }
    };

    for (LoRATargetId id : kBaseLoRALayerTargets) {
        visit_pair(id);
    }

    if (weights.moe.use_grouped) {
        for (LoRATargetId id : kExpertLoRALayerTargets) {
            auto* w = get_grouped_weight_by_target(weights, id);
            auto* g = get_grouped_weight_by_target(grads, id);
            if (w && g) {
                fn(id, *w, *g);
            }
        }
    } else {
        const std::size_t n = std::min(weights.moe.experts.size(), grads.moe.experts.size());
        for (std::size_t e = 0; e < n; ++e) {
            for (LoRATargetId id : kExpertLoRALayerTargets) {
                auto* w = get_expert_weight_by_target(weights.moe.experts[e], id);
                auto* g = get_expert_weight_by_target(grads.moe.experts[e], id);
                if (w && g) {
                    fn(id, *w, *g);
                }
            }
        }
    }

    for (LoRATargetId id : kSharedLoRALayerTargets) {
        visit_pair(id);
    }
    visit_pair(LoRATargetId::Router);
}

/// Stable integer key for a LoRA target, used to seed per-projection
/// dropout streams so identical targets reproduce across re-runs. Canonical
/// IDs use their enum value directly. ``Unknown`` targets fall back to a
/// 32-bit FNV-1a hash of the raw name with the high bit set, so they land
/// in ``[2^31, 2^32)`` — disjoint from canonical IDs, ~2^-31 pairwise
/// collision probability between distinct unknown names.
inline unsigned lora_target_seed_key(LoRATargetId id, std::string_view name_for_unknown) {
    if (id != LoRATargetId::Unknown) {
        return static_cast<unsigned>(id);
    }
    unsigned h = 2166136261u;
    for (char c : name_for_unknown) {
        h ^= static_cast<unsigned char>(c);
        h *= 16777619u;
    }
    return h | 0x80000000u;
}

}  // namespace modules

#endif  // SUROGATE_SRC_MODULES_LORA_LORA_TYPES_H
