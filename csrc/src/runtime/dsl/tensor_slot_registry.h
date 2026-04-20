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

#include "runtime/dsl/ir.h"
#include "runtime/dsl/tensor_slot.h"

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
        std::string canonical_name;  ///< Canonical name (if alias)
        ActivationScope scope = ActivationScope::Block;
        std::vector<Dim> shape;             ///< Shape expression
        std::optional<ETensorDType> dtype;  ///< Override dtype
        bool save_for_backward = false;
        bool recompute_in_backward = false;                ///< Can be recomputed instead of saved
        std::string recompute_policy;                      // Derived from share_policy in init_from_layout
        SharePolicy share_policy = SharePolicy::PerLayer;  ///< Cross-layer sharing policy
        ActivationMemoryHint memory_hint = ActivationMemoryHint::Persistent;
        std::string shares_with;  ///< Slot to share memory with (if hint == Shared)
        std::string gradient_of;  ///< For gradient slots
        std::string alias_of;     ///< Optional alias target (reuse existing buffer)
        std::string condition;    ///< Condition expression
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
    const std::vector<std::string>& get_save_list() const {
        return mSaveList;
    }

    /// @brief Get the recompute list (tensors that can be recomputed in backward)
    const std::vector<std::string>& get_recompute_list() const {
        return mRecomputeList;
    }

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

    // should_share removed (Phase 4 M5): no callers after
    // BufferPlan::build dropped its share_for lambda.

    /// @brief Check if the registry has been initialized from a DSL layout
    bool has_dsl_layout() const {
        return mHasDslLayout;
    }

    /// @brief Iterate over all registered slots
    /// @param func Callable with signature (const std::string& name, const SlotEntry& entry)
    template <typename Func>
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

/// Resolve `name` to a TensorSlot, treating a `_flat` suffix as an alias of
/// the underlying (non-flat) slot when the suffixed name isn't registered
/// directly. Names like `xF_flat`/`ln_final_flat` carry their own 2D shape in
/// the DSL layout and aren't in the static name→slot table; this helper makes
/// them route to the parent slot (LNFinal) while telling the caller via
/// `*out_is_flat_view` that a 2D view is required.
///
/// Returns `TensorSlot::Mapped` if neither the suffixed nor the base name
/// resolves.
TensorSlot resolve_slot_with_flat(const std::string& name, bool* out_is_flat_view = nullptr);

/// Block-scope variant: parses `blocks[N].<field>` (or `blocks.N.field`,
/// `layerN.field`), strips the SSA numeric suffix, and returns the resulting
/// slot. Block `_flat` aliases (`ln1_flat`, `qkv_flat`, ...) are in the
/// name→slot table directly and resolve to the same slot as their 3D base;
/// `*out_is_flat_view` is set to true whenever the stripped field name ends
/// in `_flat`, so callers that need a 2D view of the underlying 3D buffer
/// can flatten accordingly.
///
/// Returns `TensorSlot::Mapped` (and leaves outputs untouched) when `name`
/// is not block-qualified.
TensorSlot resolve_block_slot(const std::string& name,
                              int* out_layer_idx = nullptr,
                              bool* out_is_flat_view = nullptr,
                              std::string* out_base_field = nullptr);

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_TENSOR_SLOT_REGISTRY_H
