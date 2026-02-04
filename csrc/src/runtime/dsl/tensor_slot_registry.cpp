// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/dsl/tensor_slot_registry.h"

namespace dsl {

// ============================================================================
// Name-to-TensorSlot mappings for fast runtime dispatch
// ============================================================================
// These mappings connect DSL slot names to TensorSlot enum values, which
// correspond to specific struct fields in RunState. The Python DSL is the
// source of truth for slot declarations; these mappings enable O(1) dispatch.

namespace {

// Combined name-to-slot mapping for all known slots
const std::unordered_map<std::string, TensorSlot> kSlotMappings = {
    // Block activations
    {"ln1", TensorSlot::BlockLN1},
    {"ln1_flat", TensorSlot::BlockLN1},
    {"ln1_rstd", TensorSlot::BlockLN1RSTD},
    {"ln2", TensorSlot::BlockLN2},
    {"ln2_flat", TensorSlot::BlockLN2},
    {"ln2_rstd", TensorSlot::BlockLN2RSTD},
    {"q_rstd", TensorSlot::BlockQRSTD},
    {"k_rstd", TensorSlot::BlockKRSTD},
    {"qkv", TensorSlot::BlockQKV},
    {"qkv_flat", TensorSlot::BlockQKV},
    {"qkv_biased", TensorSlot::BlockQKV},
    {"qkv_rope", TensorSlot::BlockQKVRoPE},
    {"lse", TensorSlot::BlockLSE},
    {"att", TensorSlot::BlockAtt},
    {"att_flat", TensorSlot::BlockAtt},
    {"attn", TensorSlot::BlockAtt},
    {"att_out", TensorSlot::BlockAttOut},
    {"att_out_flat", TensorSlot::BlockAttOut},
    {"res_att", TensorSlot::BlockResidualAtt},
    {"residual_att", TensorSlot::BlockResidualAtt},
    {"mlp_up", TensorSlot::BlockMLPUp},
    {"mlp_up_flat", TensorSlot::BlockMLPUp},
    {"swiglu", TensorSlot::BlockSwiGLU},
    {"swiglu_flat", TensorSlot::BlockSwiGLU},
    {"mlp_down", TensorSlot::BlockMLPDown},
    {"mlp_down_flat", TensorSlot::BlockMLPDown},
    {"res_ffn", TensorSlot::BlockResidualFFN},
    {"residual_ffn", TensorSlot::BlockResidualFFN},
    // Block gradients
    {"d_ln1", TensorSlot::BlockDLN1},
    {"d_qkv", TensorSlot::BlockDQKV},
    {"d_qkv_rope", TensorSlot::BlockDQKV},
    {"d_qkv_rope_flat", TensorSlot::BlockDQKV},
    {"d_att", TensorSlot::BlockDAtt},
    {"d_swiglu", TensorSlot::BlockDSwiGLU},
    {"d_mlp_up", TensorSlot::BlockDMLPUp},
    {"d_mlp_down", TensorSlot::BlockDMLPDown},
    {"d_ln2", TensorSlot::BlockDLN2},
    {"d_res_att", TensorSlot::BlockDResAtt},
    {"d_res_ffn", TensorSlot::BlockDResFFN},
    // Global activations
    {"encoded", TensorSlot::Encoded},
    {"x0", TensorSlot::Encoded},
    {"ln_final", TensorSlot::LNFinal},
    {"xF", TensorSlot::LNFinal},
    {"ln_final_rstd", TensorSlot::LNFinalRSTD},
    {"freq_cis", TensorSlot::FreqCis},
    {"rope_freqs", TensorSlot::FreqCis},
    {"final_residual", TensorSlot::FinalResidual},
    {"residual_final", TensorSlot::FinalResidual},
    // IO slots
    {"token_ids", TensorSlot::TokenIDs},
    {"position_ids", TensorSlot::PositionIDs},
    {"targets", TensorSlot::Targets},
    {"loss", TensorSlot::Losses},
    {"losses", TensorSlot::Losses},
    {"d_loss", TensorSlot::DLoss},
};

// Reverse mapping: TensorSlot enum to canonical name (for debugging)
const char* slot_to_name(TensorSlot slot) {
    switch (slot) {
        case TensorSlot::BlockLN1: return "ln1";
        case TensorSlot::BlockLN1RSTD: return "ln1_rstd";
        case TensorSlot::BlockLN2: return "ln2";
        case TensorSlot::BlockLN2RSTD: return "ln2_rstd";
        case TensorSlot::BlockQRSTD: return "q_rstd";
        case TensorSlot::BlockKRSTD: return "k_rstd";
        case TensorSlot::BlockQKV: return "qkv";
        case TensorSlot::BlockQKVRoPE: return "qkv_rope";
        case TensorSlot::BlockLSE: return "lse";
        case TensorSlot::BlockAtt: return "att";
        case TensorSlot::BlockAttOut: return "att_out";
        case TensorSlot::BlockResidualAtt: return "res_att";
        case TensorSlot::BlockMLPUp: return "mlp_up";
        case TensorSlot::BlockSwiGLU: return "swiglu";
        case TensorSlot::BlockMLPDown: return "mlp_down";
        case TensorSlot::BlockResidualFFN: return "res_ffn";
        case TensorSlot::BlockDLN1: return "d_ln1";
        case TensorSlot::BlockDQKV: return "d_qkv";
        case TensorSlot::BlockDAtt: return "d_att";
        case TensorSlot::BlockDSwiGLU: return "d_swiglu";
        case TensorSlot::BlockDMLPUp: return "d_mlp_up";
        case TensorSlot::BlockDMLPDown: return "d_mlp_down";
        case TensorSlot::BlockDLN2: return "d_ln2";
        case TensorSlot::BlockDResAtt: return "d_res_att";
        case TensorSlot::BlockDResFFN: return "d_res_ffn";
        case TensorSlot::Encoded: return "encoded";
        case TensorSlot::LNFinal: return "ln_final";
        case TensorSlot::LNFinalRSTD: return "ln_final_rstd";
        case TensorSlot::FinalResidual: return "final_residual";
        case TensorSlot::FreqCis: return "freq_cis";
        case TensorSlot::TokenIDs: return "token_ids";
        case TensorSlot::PositionIDs: return "position_ids";
        case TensorSlot::Targets: return "targets";
        case TensorSlot::Losses: return "loss";
        case TensorSlot::DLoss: return "d_loss";
        case TensorSlot::Parameter: return "parameter";
        case TensorSlot::Temporary: return "temporary";
        case TensorSlot::Saved: return "saved";
        case TensorSlot::Mapped: return "mapped";
    }
    return "unknown";
}

}  // namespace

// ============================================================================
// TensorSlotRegistry Implementation
// ============================================================================

void TensorSlotRegistry::init_from_layout(const ActivationLayoutIR& layout) {
    mHasDslLayout = true;

    // Process forward activation slots
    for (const auto& slot_ir : layout.slots) {
        SlotEntry entry;
        entry.canonical_name = slot_ir.name;
        entry.scope = slot_ir.scope;
        entry.shape = slot_ir.shape;
        entry.dtype = slot_ir.dtype;
        entry.save_for_backward = slot_ir.save_for_backward;
        entry.recompute_in_backward = slot_ir.recompute_in_backward;
        entry.recompute_from = slot_ir.recompute_from;
        entry.recompute_op = slot_ir.recompute_op;
        entry.recompute_attrs = slot_ir.recompute_attrs;
        entry.recompute_policy = slot_ir.recompute_policy;
        entry.recompute_group = slot_ir.recompute_group;
        entry.recompute_outputs = slot_ir.recompute_outputs;
        entry.lora_targets = slot_ir.lora_targets;
        entry.share_policy = slot_ir.share_policy;
        entry.memory_hint = slot_ir.memory_hint;
        entry.shares_with = slot_ir.shares_with;
        entry.alias_of = slot_ir.alias_of;
        entry.condition = slot_ir.condition;

        // Map to TensorSlot enum for fast runtime dispatch
        entry.slot = builtin_slot_from_name(slot_ir.name);

        // Register canonical name
        mRegistry[slot_ir.name] = entry;

        // Register aliases
        for (const auto& alias : slot_ir.aliases) {
            SlotEntry alias_entry = entry;
            alias_entry.canonical_name = slot_ir.name;
            // Aliases may also have TensorSlot mappings
            TensorSlot alias_slot = builtin_slot_from_name(alias);
            if (alias_slot != TensorSlot::Mapped) {
                alias_entry.slot = alias_slot;
            }
            mRegistry[alias] = alias_entry;
        }

        // Build save list
        if (slot_ir.save_for_backward) {
            mSaveList.push_back(slot_ir.name);
        }

        // Build recompute list
        if (slot_ir.recompute_in_backward) {
            mRecomputeList.push_back(slot_ir.name);
        }
    }

    // Process gradient slots
    for (const auto& slot_ir : layout.gradient_slots) {
        SlotEntry entry;
        entry.canonical_name = slot_ir.name;
        entry.scope = slot_ir.scope;
        entry.shape = slot_ir.shape;
        entry.dtype = slot_ir.dtype;
        entry.gradient_of = slot_ir.gradient_of;
        entry.memory_hint = slot_ir.memory_hint;
        entry.shares_with = slot_ir.shares_with;
        entry.alias_of = slot_ir.alias_of;
        entry.condition = slot_ir.condition;

        // Map to TensorSlot enum for fast runtime dispatch
        entry.slot = builtin_slot_from_name(slot_ir.name);

        mRegistry[slot_ir.name] = entry;

        // Register aliases
        for (const auto& alias : slot_ir.aliases) {
            SlotEntry alias_entry = entry;
            alias_entry.canonical_name = slot_ir.name;
            mRegistry[alias] = alias_entry;
        }
    }
}

std::optional<TensorSlotRegistry::SlotEntry> TensorSlotRegistry::lookup(const std::string& name) const {
    auto it = mRegistry.find(name);
    if (it != mRegistry.end()) {
        return it->second;
    }
    return std::nullopt;
}

bool TensorSlotRegistry::is_block_activation(const std::string& name) const {
    auto entry = lookup(name);
    return entry && entry->scope == ActivationScope::Block;
}

bool TensorSlotRegistry::is_global_activation(const std::string& name) const {
    auto entry = lookup(name);
    return entry && entry->scope == ActivationScope::Global;
}

bool TensorSlotRegistry::is_gradient(const std::string& name) const {
    auto entry = lookup(name);
    return entry && (entry->scope == ActivationScope::Gradient ||
                     entry->scope == ActivationScope::GlobalGradient);
}

std::string TensorSlotRegistry::get_canonical_name(const std::string& name) const {
    auto entry = lookup(name);
    if (entry && !entry->canonical_name.empty()) {
        return entry->canonical_name;
    }
    return name;
}

bool TensorSlotRegistry::can_recompute(const std::string& name) const {
    auto entry = lookup(name);
    return entry && (entry->recompute_in_backward ||
                     entry->memory_hint == ActivationMemoryHint::Recompute);
}

bool TensorSlotRegistry::will_recompute(const std::string& name, bool lora_only_mode) const {
    auto entry = lookup(name);
    if (!entry) {
        return false;
    }

    // Must be marked as recomputable
    if (!entry->recompute_in_backward &&
        entry->memory_hint != ActivationMemoryHint::Recompute) {
        return false;
    }

    // Check policy
    const std::string& policy = entry->recompute_policy;
    if (policy == "never") {
        return false;
    }
    if (policy == "lora_only") {
        // Only recompute in LoRA mode, not in FFT mode
        // In FFT mode, use saved tensors (requires gradient_accumulation_steps=1 to avoid corruption)
        return lora_only_mode;
    }
    if (policy == "fft_only") {
        // Only recompute in FFT mode, not in LoRA mode
        return !lora_only_mode;
    }
    // "always" or empty (default) - always recompute
    return true;
}

bool TensorSlotRegistry::is_shared(const std::string& name) const {
    auto entry = lookup(name);
    return entry && entry->memory_hint == ActivationMemoryHint::Shared;
}

std::string TensorSlotRegistry::get_shares_with(const std::string& name) const {
    auto entry = lookup(name);
    if (entry && entry->memory_hint == ActivationMemoryHint::Shared) {
        return entry->shares_with;
    }
    return "";
}

ActivationMemoryHint TensorSlotRegistry::get_memory_hint(const std::string& name) const {
    auto entry = lookup(name);
    if (entry) {
        return entry->memory_hint;
    }
    return ActivationMemoryHint::Persistent;  // Default
}

SharePolicy TensorSlotRegistry::get_share_policy(const std::string& name) const {
    auto entry = lookup(name);
    if (entry) {
        return entry->share_policy;
    }
    return SharePolicy::WhenRecomputed;  // Default
}

bool TensorSlotRegistry::should_share(const std::string& name, bool lora_only_mode, bool recompute_enabled) const {
    auto entry = lookup(name);
    if (!entry) {
        // Default: share when recompute is enabled and will_recompute says so
        return recompute_enabled && will_recompute(name, lora_only_mode);
    }

    switch (entry->share_policy) {
        case SharePolicy::PerLayer:
            // Never share - always allocate per-layer
            return false;

        case SharePolicy::WhenRecomputed:
            // Share only if recompute is enabled AND will_recompute returns true
            // This ensures we don't share when the tensor needs to be saved per-layer
            return recompute_enabled && will_recompute(name, lora_only_mode);

        case SharePolicy::AlwaysShare:
            // Always share regardless of mode (use with caution)
            // The caller must ensure this is safe for their use case
            return true;

        case SharePolicy::FFTShare:
            // Share only in FFT mode (not LoRA mode) when recompute is enabled
            // Useful for tensors that:
            // - Are safe to share in FFT (no backward hooks need per-layer values)
            // - Need per-layer in LoRA (LoRA hooks need original values per-layer)
            // Example: attention output (att, lse) - LoRA O-proj hook needs per-layer att
            return recompute_enabled && !lora_only_mode;

        case SharePolicy::LoRAShare:
            // Share only in LoRA mode (not FFT mode) when recompute is enabled
            // Useful for tensors that:
            // - LoRA can recompute during backward (hooks don't need saved values)
            // - FFT needs saved per-layer for bit-exact gradients
            return recompute_enabled && lora_only_mode;
    }

    // Fallback (should not reach here)
    return recompute_enabled && will_recompute(name, lora_only_mode);
}

// ============================================================================
// Utility Functions
// ============================================================================

const char* builtin_slot_name(TensorSlot slot) {
    return slot_to_name(slot);
}

TensorSlot builtin_slot_from_name(const std::string& name) {
    auto it = kSlotMappings.find(name);
    if (it != kSlotMappings.end()) {
        return it->second;
    }
    return TensorSlot::Mapped;
}

}  // namespace dsl
