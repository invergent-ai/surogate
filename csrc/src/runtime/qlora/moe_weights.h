// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Generic MoE weight structures for QLoRA quantization systems.
// Works with any quantization format (BnB NF4, FP8, FP4).

#ifndef SUROGATE_SRC_MODULES_QLORA_MOE_WEIGHTS_H
#define SUROGATE_SRC_MODULES_QLORA_MOE_WEIGHTS_H

#include <optional>
#include <vector>

#include "utilities/tensor.h"

namespace modules {

/**
 * @brief MoE configuration for weight managers
 *
 * Shared configuration for MoE models across all quantization formats.
 * This is separate from QLoRAConfig to keep concerns separate.
 */
struct MoEWeightConfig {
    int num_experts = 0;           ///< Number of experts (0 = dense model)
    int num_experts_per_tok = 8;   ///< Top-k experts selected per token
    int moe_intermediate_size = 0; ///< Per-expert MLP intermediate size (0 = use regular intermediate_size)

    /**
     * @brief Check if this is an MoE model
     */
    [[nodiscard]] bool is_moe() const { return num_experts > 0; }

    /**
     * @brief Get effective intermediate size for MoE experts
     * @param default_intermediate The regular intermediate_size to use if moe_intermediate_size is 0
     */
    [[nodiscard]] int effective_intermediate_size(int default_intermediate) const {
        return moe_intermediate_size > 0 ? moe_intermediate_size : default_intermediate;
    }
};

/**
 * @brief Generic MoE expert weights - works with any quantized weight type
 *
 * Represents the MLP weights for a single expert in an MoE block.
 * The quantization format depends on the template parameter.
 *
 * Weight dimensions:
 * - gate_up_proj: (mlp_up_factor * moe_intermediate_size, hidden_size) - fused gate+up (or up-only) projection
 * - down_proj: (hidden_size, moe_intermediate_size) - down projection
 *
 * @tparam QuantizedWeight The quantized weight type (BnBBlockQuantizedWeight, FP8BlockWeight, FP4BlockWeight)
 */
template<typename QuantizedWeight>
struct MoEExpertWeights {
    QuantizedWeight gate_up_proj;  ///< Fused gate+up projection (mlp_up_factor * intermediate, hidden)
    QuantizedWeight down_proj;     ///< Down projection (hidden, intermediate)

    /**
     * @brief Get total memory footprint in bytes
     */
    [[nodiscard]] std::size_t bytes() const {
        return gate_up_proj.bytes() + down_proj.bytes();
    }

    /**
     * @brief Check if weights are properly initialized
     */
    [[nodiscard]] bool is_valid() const {
        return gate_up_proj.is_valid() && down_proj.is_valid();
    }
};

/**
 * @brief Generic MoE block weights - extends base block with expert weights
 *
 * Represents all weights for a single MoE transformer block layer.
 * Inherits attention and layer norm weights from BaseBlockWeights,
 * and adds MoE-specific weights (experts + router).
 *
 * Note: The base class should NOT have gate_up_proj and down_proj
 * members when using MoE, as those are replaced by expert-specific weights.
 *
 * @tparam BaseBlockWeights The base block weights type (with qkv_proj, out_proj, ln weights)
 * @tparam QuantizedWeight The quantized weight type
 */
template<typename BaseBlockWeights, typename QuantizedWeight>
struct MoEBlockWeights {
    // =========================================================================
    // Attention weights (same as dense blocks)
    // =========================================================================

    /// Fused Q/K/V projection (3 * num_heads * head_size, hidden_size)
    QuantizedWeight qkv_proj;

    /// Attention output projection (hidden_size, num_heads * head_size)
    QuantizedWeight out_proj;

    // =========================================================================
    // Layer normalization weights (BF16, not quantized)
    // =========================================================================

    /// Pre-attention RMSNorm weight (hidden_size,)
    Tensor ln1_weight;

    /// Post-attention / Pre-MoE RMSNorm weight (hidden_size,)
    Tensor ln2_weight;

    /// QK-norm weights for models like Qwen3 (optional)
    std::optional<Tensor> q_norm_weight;
    std::optional<Tensor> k_norm_weight;

    // =========================================================================
    // MoE-specific weights
    // =========================================================================

    /// Expert weights (num_experts sets of gate_up + down projections)
    std::vector<MoEExpertWeights<QuantizedWeight>> experts;

    /// Router gate weight (num_experts, hidden_size) - kept in BF16, not quantized
    /// Small tensor, quantization provides negligible benefit
    Tensor router_gate;

    /// Optional shared expert weights (single expert applied to all tokens)
    std::optional<MoEExpertWeights<QuantizedWeight>> shared_expert;

    /**
     * @brief Get total memory footprint in bytes
     */
    [[nodiscard]] std::size_t bytes() const {
        std::size_t total = qkv_proj.bytes() + out_proj.bytes();
        total += ln1_weight.bytes() + ln2_weight.bytes();
        if (q_norm_weight.has_value()) total += q_norm_weight->bytes();
        if (k_norm_weight.has_value()) total += k_norm_weight->bytes();

        for (const auto& expert : experts) {
            total += expert.bytes();
        }
        if (shared_expert.has_value()) {
            total += shared_expert->bytes();
        }
        total += router_gate.bytes();

        return total;
    }

    /**
     * @brief Check if all weights are properly initialized
     */
    [[nodiscard]] bool is_valid() const {
        if (!qkv_proj.is_valid() || !out_proj.is_valid()) {
            return false;
        }
        if (ln1_weight.Data == nullptr || ln2_weight.Data == nullptr) {
            return false;
        }
        if (router_gate.Data == nullptr) {
            return false;
        }
        for (const auto& expert : experts) {
            if (!expert.is_valid()) {
                return false;
            }
        }
        if (shared_expert.has_value() && !shared_expert->is_valid()) {
            return false;
        }
        return true;
    }

    /**
     * @brief Get number of experts
     */
    [[nodiscard]] int num_experts() const {
        return static_cast<int>(experts.size());
    }
};

/**
 * @brief Dequantized expert weights for forward pass
 *
 * BF16 weights ready for matmul, used by the weight provider
 * after dequantizing from the quantized format.
 */
struct DequantizedExpertWeights {
    Tensor gate_up_proj;  ///< (2 * intermediate, hidden) BF16
    Tensor down_proj;     ///< (hidden, intermediate) BF16

    /**
     * @brief Get total memory footprint in bytes
     */
    [[nodiscard]] std::size_t bytes() const {
        return gate_up_proj.bytes() + down_proj.bytes();
    }
};

/**
 * @brief Expert cache entry for selective dequantization
 *
 * Tracks which expert is currently loaded in a dequantization buffer slot.
 * Used by weight providers to avoid redundant dequantization.
 */
struct ExpertCacheEntry {
    int layer_idx = -1;      ///< Layer index (-1 = empty)
    int expert_idx = -1;     ///< Expert index (-1 = empty)
    uint64_t step_version = 0; ///< Training step when this was loaded

    /**
     * @brief Check if this entry is valid for the given request
     */
    [[nodiscard]] bool matches(int layer, int expert, uint64_t step) const {
        return layer_idx == layer && expert_idx == expert && step_version == step;
    }

    /**
     * @brief Update entry with new expert
     */
    void update(int layer, int expert, uint64_t step) {
        layer_idx = layer;
        expert_idx = expert;
        step_version = step;
    }

    /**
     * @brief Clear the entry
     */
    void clear() {
        layer_idx = -1;
        expert_idx = -1;
        step_version = 0;
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_QLORA_MOE_WEIGHTS_H
