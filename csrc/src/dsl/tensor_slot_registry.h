// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Tensor Slot Registry - Maps tensor names to pre-resolved slots.
//
// This module maps DSL-defined tensor names to TensorSlot enum values for fast
// dispatch in the runtime. The mappings are loaded from the DSL ActivationLayoutIR,
// with the Python DSL being the single source of truth for slot declarations.
//
// The TensorSlot enum values correspond to specific struct fields in RunState,
// enabling O(1) tensor lookups during forward/backward passes.

#ifndef SUROGATE_SRC_DSL_TENSOR_SLOT_REGISTRY_H
#define SUROGATE_SRC_DSL_TENSOR_SLOT_REGISTRY_H

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "dsl/ir.h"
#include "dsl/tensor_slot.h"

namespace dsl {

/// @brief Registry for mapping tensor names to pre-resolved slots
///
/// The registry is initialized from the DSL ActivationLayoutIR and provides
/// fast O(1) lookups for tensor name -> slot mappings. Known slot names are
/// mapped to TensorSlot enum values for fast runtime dispatch.
class TensorSlotRegistry {
public:
    /// @brief Entry in the slot registry
    struct SlotEntry {
        TensorSlot slot = TensorSlot::Mapped;
        std::string canonical_name;          ///< Canonical name (if alias)
        ActivationScope scope = ActivationScope::Block;
        std::vector<Dim> shape;              ///< Shape expression
        std::optional<ETensorDType> dtype;   ///< Override dtype
        bool save_for_backward = false;
        bool recompute_in_backward = false;  ///< Can be recomputed instead of saved
        std::vector<std::string> recompute_from;
        std::string recompute_op;
        AttrMap recompute_attrs;
        std::string recompute_policy;
        std::string recompute_group;
        std::vector<std::string> recompute_outputs;
        std::vector<std::string> lora_targets;
        SharePolicy share_policy = SharePolicy::WhenRecomputed;  ///< Cross-layer sharing policy
        ActivationMemoryHint memory_hint = ActivationMemoryHint::Persistent;
        std::string shares_with;             ///< Slot to share memory with (if hint == Shared)
        std::string gradient_of;             ///< For gradient slots
        std::string alias_of;                ///< Optional alias target (reuse existing buffer)
        std::string condition;               ///< Condition expression
    };

    TensorSlotRegistry() = default;

    /// @brief Initialize from DSL activation layout (required - no fallback)
    void init_from_layout(const ActivationLayoutIR& layout);

    /// @brief Look up a tensor by name, returning its slot info
    /// @param name Tensor name (may be canonical or alias)
    /// @return SlotEntry if found, nullopt otherwise
    std::optional<SlotEntry> lookup(const std::string& name) const;

    /// @brief Check if a name is a block-scoped activation
    bool is_block_activation(const std::string& name) const;

    /// @brief Check if a name is a global-scoped activation
    bool is_global_activation(const std::string& name) const;

    /// @brief Check if a name is a gradient slot
    bool is_gradient(const std::string& name) const;

    /// @brief Get the canonical name for an alias
    std::string get_canonical_name(const std::string& name) const;

    /// @brief Get the save list (tensors to save for backward)
    const std::vector<std::string>& get_save_list() const { return mSaveList; }

    /// @brief Get the recompute list (tensors that can be recomputed in backward)
    const std::vector<std::string>& get_recompute_list() const { return mRecomputeList; }

    /// @brief Check if a slot can be recomputed in backward
    bool can_recompute(const std::string& name) const;

    /// @brief Check if a slot will actually be recomputed given the current mode
    /// @param name Tensor name
    /// @param lora_only_mode True if in LoRA-only mode (not FFT mode)
    /// @return True if the tensor will be recomputed in the given mode
    ///
    /// This differs from can_recompute() by also checking the recompute_policy:
    /// - "always": will recompute in any mode
    /// - "lora_only": will only recompute in LoRA mode, not in FFT mode
    /// - "never": will never recompute
    bool will_recompute(const std::string& name, bool lora_only_mode) const;

    /// @brief Check if a slot shares memory with another slot
    bool is_shared(const std::string& name) const;

    /// @brief Get the slot this slot shares memory with (empty if not shared)
    std::string get_shares_with(const std::string& name) const;

    /// @brief Get the memory hint for a slot
    ActivationMemoryHint get_memory_hint(const std::string& name) const;

    /// @brief Get the share policy for a slot
    SharePolicy get_share_policy(const std::string& name) const;

    /// @brief Determine if a slot should be shared across layers given the current mode
    /// @param name Tensor name
    /// @param lora_only_mode True if in LoRA-only mode (not FFT mode)
    /// @param recompute_enabled True if recompute is enabled
    /// @return True if the slot should use shared allocation across layers
    ///
    /// This method evaluates the slot's share_policy to determine if sharing is safe:
    /// - PerLayer: Never share (return false)
    /// - WhenRecomputed: Share only if will_recompute() returns true
    /// - AlwaysShare: Always share (return true)
    /// - FFTShare: Share only in FFT mode (when lora_only_mode is false)
    /// - LoRAShare: Share only in LoRA mode (when lora_only_mode is true)
    bool should_share(const std::string& name, bool lora_only_mode, bool recompute_enabled) const;

    /// @brief Check if the registry has been initialized from a DSL layout
    bool has_dsl_layout() const { return mHasDslLayout; }

    /// @brief Iterate over all registered slots
    /// @param func Callable with signature (const std::string& name, const SlotEntry& entry)
    template<typename Func>
    void for_each(Func&& func) const {
        for (const auto& [name, entry] : mRegistry) {
            func(name, entry);
        }
    }

private:
    std::unordered_map<std::string, SlotEntry> mRegistry;
    std::vector<std::string> mSaveList;
    std::vector<std::string> mRecomputeList;
    bool mHasDslLayout = false;
};

/// @brief Map a TensorSlot enum to its canonical name (for debugging)
const char* builtin_slot_name(TensorSlot slot);

/// @brief Map a slot name to TensorSlot enum for fast runtime dispatch
/// @return TensorSlot::Mapped if not a known slot (will use dynamic lookup)
TensorSlot builtin_slot_from_name(const std::string& name);

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_TENSOR_SLOT_REGISTRY_H
