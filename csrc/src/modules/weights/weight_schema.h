// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Weight Schema - Module weight requirements declaration
//
// This file defines the ModuleWeightSchema trait that each module type uses
// to declare its weight requirements. The schema provides:
// 1. Required and optional weights for each module
// 2. Expected shapes based on configuration
// 3. Validation before weight loading
//
// This enables the registry-based weight loading system in Phase 7 of the
// Transformers-like inheritance pattern implementation.

#ifndef SUROGATE_SRC_MODULES_WEIGHTS_WEIGHT_SCHEMA_H
#define SUROGATE_SRC_MODULES_WEIGHTS_WEIGHT_SCHEMA_H

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "config/pretrained_config.h"
#include "models/qwen3moe/config.h"
#include "utilities/tensor.h"

namespace modules {

/**
 * @brief Requirement level for a weight tensor
 */
enum class WeightRequirement {
    Required,       ///< Weight must be present
    Optional,       ///< Weight may be present (e.g., bias, QK norm)
    Conditional     ///< Weight required based on config (e.g., QK norm weights when UseQKNorm=true)
};

/**
 * @brief Specification for a single weight tensor
 */
struct TensorSpec {
    std::string name;                   ///< Internal field name (e.g., "qkv_weight")
    std::vector<long> shape;            ///< Expected shape
    WeightRequirement requirement;      ///< Whether this weight is required
    std::string description;            ///< Human-readable description for error messages

    TensorSpec() = default;
    TensorSpec(std::string n, std::vector<long> s, WeightRequirement r, std::string desc = "")
        : name(std::move(n)), shape(std::move(s)), requirement(r), description(std::move(desc)) {}

    [[nodiscard]] long nelem() const {
        long n = 1;
        for (long dim : shape) n *= dim;
        return n;
    }
};

/**
 * @brief Schema describing all weights for a module type
 */
struct ModuleSchema {
    std::string module_name;            ///< Module type name for error messages
    std::vector<TensorSpec> weights;    ///< All weight specifications

    /**
     * @brief Get a specific weight spec by name
     */
    [[nodiscard]] const TensorSpec* find(const std::string& name) const {
        for (const auto& w : weights) {
            if (w.name == name) return &w;
        }
        return nullptr;
    }

    /**
     * @brief Get all required weights
     */
    [[nodiscard]] std::vector<const TensorSpec*> required_weights() const {
        std::vector<const TensorSpec*> result;
        for (const auto& w : weights) {
            if (w.requirement == WeightRequirement::Required) {
                result.push_back(&w);
            }
        }
        return result;
    }
};

// ============================================================================
// Module Weight Schema Trait
// ============================================================================

/**
 * @brief Primary template for ModuleWeightSchema trait
 *
 * Specializations define the schema for specific module types.
 * Each specialization provides a static `describe()` method that returns
 * a ModuleSchema based on the model configuration.
 */
template<typename Module>
struct ModuleWeightSchema {
    // Default: no weights (for parameter-free modules like SwiGLU activation)
    static ModuleSchema describe(const PretrainedConfig& /*cfg*/) {
        return ModuleSchema{};
    }
};

// ============================================================================
// Attention Module Schema
// ============================================================================

// Forward declarations
class AttentionModule;

template<>
struct ModuleWeightSchema<AttentionModule> {
    static ModuleSchema describe(const PretrainedConfig& cfg) {
        const int hs = cfg.head_size();
        const int hq = cfg.NumQueryHeads;
        const int hkv = cfg.NumKeyValHeads;
        const int c = cfg.HiddenSize;
        const int q_rows = hs * hq;
        const int kv_rows = hs * hkv;
        const int qkv_rows = q_rows + 2 * kv_rows;

        ModuleSchema schema;
        schema.module_name = "AttentionModule";
        schema.weights = {
            {"qkv_weight", {qkv_rows, c}, WeightRequirement::Required,
             "Fused QKV projection weight"},
            {"qkv_bias", {qkv_rows}, cfg.UseQKVBias ? WeightRequirement::Required : WeightRequirement::Optional,
             "Fused QKV projection bias"},
            {"out_weight", {c, q_rows}, WeightRequirement::Required,
             "Output projection weight"},
            {"q_norm_weight", {hs}, cfg.UseQKNorm ? WeightRequirement::Conditional : WeightRequirement::Optional,
             "Q normalization weight (Qwen3-style)"},
            {"k_norm_weight", {hs}, cfg.UseQKNorm ? WeightRequirement::Conditional : WeightRequirement::Optional,
             "K normalization weight (Qwen3-style)"},
        };
        return schema;
    }
};

// ============================================================================
// RMSNorm Module Schema
// ============================================================================

class RMSNormModule;

template<>
struct ModuleWeightSchema<RMSNormModule> {
    static ModuleSchema describe(const PretrainedConfig& cfg) {
        ModuleSchema schema;
        schema.module_name = "RMSNormModule";
        schema.weights = {
            {"weight", {cfg.HiddenSize}, WeightRequirement::Required,
             "RMSNorm scale weight"},
        };
        return schema;
    }
};

// ============================================================================
// MLP/SwiGLU Module Schema
// ============================================================================

class SwiGLUMLPModule;

template<>
struct ModuleWeightSchema<SwiGLUMLPModule> {
    static ModuleSchema describe(const PretrainedConfig& cfg) {
        const int c = cfg.HiddenSize;
        const int d = cfg.IntermediateSize;

        ModuleSchema schema;
        schema.module_name = "SwiGLUMLPModule";
        schema.weights = {
            {"up_weight", {2 * d, c}, WeightRequirement::Required,
             "Fused gate+up projection weight"},
            {"down_weight", {c, d}, WeightRequirement::Required,
             "Down projection weight"},
        };
        return schema;
    }
};

// ============================================================================
// Router Module Schema (MoE)
// ============================================================================

class RouterModule;

template<>
struct ModuleWeightSchema<RouterModule> {
    static ModuleSchema describe(const PretrainedConfig& cfg) {
        // Only MoE configs have router weights
        const auto* moe_cfg = dynamic_cast<const Qwen3MoEConfig*>(&cfg);
        if (!moe_cfg || moe_cfg->NumExperts == 0) {
            return ModuleSchema{"RouterModule", {}};
        }

        ModuleSchema schema;
        schema.module_name = "RouterModule";
        schema.weights = {
            {"gate_weight", {moe_cfg->NumExperts, cfg.HiddenSize}, WeightRequirement::Required,
             "Router gate projection weight"},
        };
        return schema;
    }
};

// ============================================================================
// Expert Group Module Schema (MoE)
// ============================================================================

class ExpertGroupModule;

template<>
struct ModuleWeightSchema<ExpertGroupModule> {
    static ModuleSchema describe(const PretrainedConfig& cfg) {
        const auto* moe_cfg = dynamic_cast<const Qwen3MoEConfig*>(&cfg);
        if (!moe_cfg || moe_cfg->NumExperts == 0) {
            return ModuleSchema{"ExpertGroupModule", {}};
        }

        const int c = cfg.HiddenSize;
        const int d = moe_cfg->MoeIntermediateSize > 0 ? moe_cfg->MoeIntermediateSize : cfg.IntermediateSize;
        const int num_experts = moe_cfg->NumExperts;

        ModuleSchema schema;
        schema.module_name = "ExpertGroupModule";
        schema.weights = {
            // Batched layout for grouped GEMM efficiency
            {"gate_up_weight", {num_experts, 2 * d, c}, WeightRequirement::Required,
             "Batched expert gate+up projection weights"},
            {"down_weight", {num_experts, c, d}, WeightRequirement::Required,
             "Batched expert down projection weights"},
        };
        return schema;
    }
};

// ============================================================================
// Shared Expert Module Schema (MoE)
// ============================================================================

class SharedExpertModule;

template<>
struct ModuleWeightSchema<SharedExpertModule> {
    static ModuleSchema describe(const PretrainedConfig& cfg) {
        const auto* moe_cfg = dynamic_cast<const Qwen3MoEConfig*>(&cfg);
        // Shared expert is optional even in MoE models
        if (!moe_cfg || moe_cfg->NumExperts == 0) {
            return ModuleSchema{"SharedExpertModule", {}};
        }

        const int c = cfg.HiddenSize;
        // Shared expert may have different intermediate size
        const int d = moe_cfg->MoeIntermediateSize > 0 ? moe_cfg->MoeIntermediateSize : cfg.IntermediateSize;

        ModuleSchema schema;
        schema.module_name = "SharedExpertModule";
        schema.weights = {
            {"gate_weight", {d, c}, WeightRequirement::Optional,
             "Shared expert gate projection weight"},
            {"up_weight", {d, c}, WeightRequirement::Optional,
             "Shared expert up projection weight"},
            {"down_weight", {c, d}, WeightRequirement::Optional,
             "Shared expert down projection weight"},
        };
        return schema;
    }
};

// ============================================================================
// Embedding Module Schema
// ============================================================================

class EmbeddingModule;

template<>
struct ModuleWeightSchema<EmbeddingModule> {
    static ModuleSchema describe(const PretrainedConfig& cfg) {
        ModuleSchema schema;
        schema.module_name = "EmbeddingModule";
        schema.weights = {
            {"weight", {cfg.VocabSize, cfg.HiddenSize}, WeightRequirement::Required,
             "Token embedding matrix"},
        };
        return schema;
    }
};

// ============================================================================
// LM Head Module Schema
// ============================================================================

class LMHeadModule;

template<>
struct ModuleWeightSchema<LMHeadModule> {
    static ModuleSchema describe(const PretrainedConfig& cfg) {
        ModuleSchema schema;
        schema.module_name = "LMHeadModule";
        schema.weights = {
            {"weight", {cfg.VocabSize, cfg.HiddenSize},
             cfg.TiedWordEmbeddings ? WeightRequirement::Optional : WeightRequirement::Required,
             "Language model head weight (may alias embeddings if tied)"},
        };
        return schema;
    }
};

// ============================================================================
// Schema Validation
// ============================================================================

/**
 * @brief Result of schema validation
 */
struct ValidationResult {
    bool success = true;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

    void add_error(const std::string& msg) {
        success = false;
        errors.push_back(msg);
    }

    void add_warning(const std::string& msg) {
        warnings.push_back(msg);
    }

    [[nodiscard]] std::string format() const {
        std::string result;
        for (const auto& e : errors) {
            result += "ERROR: " + e + "\n";
        }
        for (const auto& w : warnings) {
            result += "WARNING: " + w + "\n";
        }
        return result;
    }
};

/**
 * @brief Validate a tensor against its specification
 *
 * @param tensor The tensor to validate (or nullptr if not present)
 * @param spec The expected specification
 * @param module_name Module name for error messages
 * @param layer_idx Layer index for error messages (-1 for non-block weights)
 * @return ValidationResult with any errors or warnings
 */
inline ValidationResult validate_tensor(
    const Tensor* tensor,
    const TensorSpec& spec,
    const std::string& module_name,
    int layer_idx = -1
) {
    ValidationResult result;
    std::string location = layer_idx >= 0
        ? module_name + " (layer " + std::to_string(layer_idx) + ")." + spec.name
        : module_name + "." + spec.name;

    if (!tensor || tensor->Data == nullptr) {
        if (spec.requirement == WeightRequirement::Required ||
            spec.requirement == WeightRequirement::Conditional) {
            result.add_error(location + " is missing but required. " + spec.description);
        }
        return result;
    }

    // Check shape
    if (tensor->Rank != static_cast<int>(spec.shape.size())) {
        result.add_error(location + " has wrong rank: expected " +
                        std::to_string(spec.shape.size()) + ", got " +
                        std::to_string(tensor->Rank));
        return result;
    }

    for (int i = 0; i < tensor->Rank; ++i) {
        if (tensor->Sizes[i] != spec.shape[i]) {
            result.add_error(location + " has wrong shape at dimension " +
                            std::to_string(i) + ": expected " +
                            std::to_string(spec.shape[i]) + ", got " +
                            std::to_string(tensor->Sizes[i]));
        }
    }

    return result;
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_WEIGHTS_WEIGHT_SCHEMA_H
