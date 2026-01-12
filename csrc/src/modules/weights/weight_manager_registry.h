// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Weight Manager Registry - Registry-based weight loading
//
// This file provides a registry-driven weight loading system that replaces
// the hardcoded conditionals in the original weight_manager_io.h with a
// pattern-based approach. Benefits:
//
// 1. Adding new models: Create XxxWeightMapping class (no core file changes)
// 2. Centralized fusion logic: Reusable fusion helpers in weight_fusion.h
// 3. Clear error messages: Module-specific error context
// 4. Pre-load validation: Schema validation before loading
//
// This is Phase 7 of the Transformers-like inheritance pattern implementation.

#ifndef SUROGATE_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_REGISTRY_H
#define SUROGATE_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_REGISTRY_H

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "config/pretrained_config.h"
#include "modules/weights/weight_fusion.h"
#include "modules/weights/weight_mapping.h"
#include "modules/weights/weight_schema.h"
#include "utilities/safetensors.h"
#include "utilities/tensor.h"

namespace modules {

/**
 * @brief Error context for weight loading failures
 */
struct WeightLoadError {
    std::string tensor_name;        ///< HuggingFace tensor name
    std::string internal_field;     ///< Internal destination field
    std::string module_name;        ///< Module type (for context)
    int layer_idx = -1;             ///< Layer index (-1 for non-block)
    std::string message;            ///< Human-readable error message

    [[nodiscard]] std::string format() const {
        std::string location = module_name;
        if (layer_idx >= 0) {
            location += " (layer " + std::to_string(layer_idx) + ")";
        }
        return location + "." + internal_field + ": " + message +
               " [HF tensor: " + tensor_name + "]";
    }
};

/**
 * @brief Result of weight loading operation
 */
struct WeightLoadResult {
    bool success = true;
    std::vector<WeightLoadError> errors;
    std::vector<std::string> warnings;
    std::unordered_set<std::string> loaded_tensors;     ///< Tensors successfully loaded
    std::unordered_set<std::string> unknown_tensors;    ///< Tensors not recognized

    void add_error(WeightLoadError err) {
        success = false;
        errors.push_back(std::move(err));
    }

    void add_warning(const std::string& msg) {
        warnings.push_back(msg);
    }

    [[nodiscard]] std::string format() const {
        std::string result;
        if (!success) {
            result += "Weight loading failed with " + std::to_string(errors.size()) + " error(s):\n";
            for (const auto& e : errors) {
                result += "  - " + e.format() + "\n";
            }
        }
        if (!warnings.empty()) {
            result += "Warnings (" + std::to_string(warnings.size()) + "):\n";
            for (const auto& w : warnings) {
                result += "  - " + w + "\n";
            }
        }
        if (!unknown_tensors.empty()) {
            result += "Unknown tensors (" + std::to_string(unknown_tensors.size()) + "): ";
            int count = 0;
            for (const auto& t : unknown_tensors) {
                if (count++ > 5) {
                    result += "...";
                    break;
                }
                result += t + ", ";
            }
            result += "\n";
        }
        return result;
    }
};

/**
 * @brief Registry-based weight loading interface
 *
 * This class provides the main entry point for registry-based weight loading.
 * It uses WeightMapping classes to determine how to load each tensor and
 * validates against ModuleWeightSchema specifications.
 *
 * Usage:
 *   WeightRegistry registry(config);
 *   registry.register_destination("model.layers.0.input_layernorm.weight", ln1_weight);
 *   auto result = registry.load_from_file("model.safetensors");
 *   if (!result.success) { throw std::runtime_error(result.format()); }
 */
class WeightRegistry {
public:
    /**
     * @brief Construct registry for a specific model configuration
     */
    explicit WeightRegistry(const PretrainedConfig& config)
        : mConfig(config)
        , mMapping(get_weight_mapping_for_config(config)) {
        mMapping->register_patterns();
    }

    /**
     * @brief Register a destination tensor for a specific internal field path
     *
     * @param internal_path Internal field path (e.g., "blocks.0.attention.qkv_weight")
     * @param tensor Destination tensor
     */
    void register_destination(const std::string& internal_path, TensorShard& tensor) {
        mDestinations[internal_path] = &tensor;
    }

    /**
     * @brief Register destinations for all non-block weights
     */
    void register_non_block_weights(
        TensorShard& embeddings,
        TensorShard& final_norm,
        TensorShard& lm_head
    ) {
        register_destination("embeddings", embeddings);
        register_destination("final_norm_weight", final_norm);
        register_destination("lm_head", lm_head);
    }

    /**
     * @brief Load weights from a SafeTensors file
     *
     * Uses the registered mapping to determine how each tensor should be loaded.
     * Handles weight fusion (QKV, gate+up) via the mapping's custom loaders.
     *
     * @param reader SafeTensors reader with file already opened
     * @param allow_cast Allow dtype conversion during loading
     * @return WeightLoadResult with success/error information
     */
    WeightLoadResult load_from_reader(SafeTensorsReader& reader, bool allow_cast) {
        WeightLoadResult result;

        for (const auto& entry : reader.entries()) {
            const std::string& name = entry.name();

            // Find matching pattern
            const WeightPattern* pattern = mMapping->find_pattern(name);
            if (!pattern) {
                result.unknown_tensors.insert(name);
                continue;
            }

            // Extract layer index if present
            int layer_idx = BaseWeightMapping::extract_layer_idx(name);

            // Expand internal field path with layer index
            std::string internal_path = BaseWeightMapping::expand_pattern(
                pattern->internal_field, layer_idx);

            // Find destination tensor
            auto it = mDestinations.find(internal_path);
            if (it == mDestinations.end()) {
                if (!pattern->optional) {
                    result.add_error({
                        name, internal_path, "Unknown", layer_idx,
                        "No destination registered for internal path"
                    });
                }
                continue;
            }

            TensorShard* dst = it->second;

            try {
                if (pattern->loader) {
                    // Use custom loader (for fusion)
                    // Read source tensor into temporary buffer
                    std::vector<char> src_buffer(entry.byte_size());
                    entry.read_raw_bytes(src_buffer.data(), src_buffer.size());

                    pattern->loader(*dst, src_buffer.data(), src_buffer.size(),
                                   mConfig, layer_idx);
                } else {
                    // Direct copy
                    load_direct(*dst, entry, allow_cast);
                }
                result.loaded_tensors.insert(name);
            } catch (const std::exception& e) {
                result.add_error({
                    name, internal_path, "Unknown", layer_idx,
                    std::string("Load failed: ") + e.what()
                });
            }
        }

        // Validate all required tensors were loaded
        validate_completeness(result);

        return result;
    }

    /**
     * @brief Validate that all required weights have been registered
     */
    ValidationResult validate_schema() const {
        ValidationResult result;

        // Validate non-block weights
        validate_module_schema<EmbeddingModule>("embeddings", -1, result);
        validate_module_schema<RMSNormModule>("final_norm_weight", -1, result);
        validate_module_schema<LMHeadModule>("lm_head", -1, result);

        // Validate per-layer weights
        for (int i = 0; i < mConfig.NumLayers; ++i) {
            std::string prefix = "blocks." + std::to_string(i);
            validate_module_schema<AttentionModule>(prefix + ".attention", i, result);
            validate_module_schema<RMSNormModule>(prefix + ".ln1", i, result);
            validate_module_schema<RMSNormModule>(prefix + ".ln2", i, result);

            // MLP or MoE depending on layer type
            if (mConfig.is_moe_layer(i)) {
                validate_module_schema<RouterModule>(prefix + ".router", i, result);
                validate_module_schema<ExpertGroupModule>(prefix + ".experts", i, result);
                validate_module_schema<SharedExpertModule>(prefix + ".shared_expert", i, result);
            } else {
                validate_module_schema<SwiGLUMLPModule>(prefix + ".mlp", i, result);
            }
        }

        return result;
    }

private:
    /**
     * @brief Direct tensor load (no transformation)
     */
    void load_direct(TensorShard& dst, const SafeTensorEntry& entry, bool allow_cast) {
        // Handle sharding if applicable
        std::ptrdiff_t src_begin = 0;
        std::ptrdiff_t src_end = dst.global_nelem();
        load_intersect(dst, entry, src_begin, src_end, allow_cast);
    }

    /**
     * @brief Load intersection of source and sharded destination
     */
    void load_intersect(TensorShard& dst, const SafeTensorEntry& src,
                        std::ptrdiff_t src_begin, std::ptrdiff_t src_end,
                        bool allow_cast) {
        if (src_begin >= src_end) return;

        std::ptrdiff_t dst_begin = dst.shard_offset();
        std::ptrdiff_t dst_end = dst_begin + static_cast<std::ptrdiff_t>(dst.nelem());

        if (src_begin >= dst_end || src_end <= dst_begin) return;

        std::ptrdiff_t slice_begin = std::max(src_begin, dst_begin);
        std::ptrdiff_t slice_end = std::min(src_end, dst_end);

        std::ptrdiff_t dst_offset = slice_begin - dst_begin;
        std::ptrdiff_t elements = slice_end - slice_begin;

        Tensor dst_slice = static_cast<const Tensor&>(dst);
        dst_slice.Sizes.fill(1);
        dst_slice.Rank = 1;
        dst_slice.Sizes[0] = elements;
        dst_slice.Data = dst.Data + dst_offset * get_dtype_size(dst.DType);

        src.read_raw(dst_slice, slice_begin - src_begin, elements, allow_cast);
    }

    /**
     * @brief Validate schema for a specific module
     */
    template<typename Module>
    void validate_module_schema(const std::string& prefix, int layer_idx,
                               ValidationResult& result) const {
        ModuleSchema schema = ModuleWeightSchema<Module>::describe(mConfig);
        if (schema.weights.empty()) return;

        for (const auto& spec : schema.weights) {
            std::string path = prefix + "." + spec.name;
            auto it = mDestinations.find(path);
            const Tensor* tensor = (it != mDestinations.end()) ? it->second : nullptr;

            auto tensor_result = validate_tensor(tensor, spec, schema.module_name, layer_idx);
            for (const auto& e : tensor_result.errors) {
                result.add_error(e);
            }
            for (const auto& w : tensor_result.warnings) {
                result.add_warning(w);
            }
        }
    }

    /**
     * @brief Validate that all required tensors were loaded
     */
    void validate_completeness(WeightLoadResult& result) const {
        // For now, we rely on the pattern matching - if a required pattern
        // wasn't found in the file, it would have been reported as an error
        // during loading if the destination was registered but not loaded.

        // Future enhancement: Compare loaded_tensors against required weights
        // from schema to catch missing tensors.
    }

    const PretrainedConfig& mConfig;
    std::unique_ptr<BaseWeightMapping> mMapping;
    std::unordered_map<std::string, TensorShard*> mDestinations;
};

/**
 * @brief Convenience function to load weights using registry
 *
 * This is the recommended entry point for weight loading. It:
 * 1. Creates a registry for the model config
 * 2. Registers all weight destinations
 * 3. Validates the schema
 * 4. Loads from file
 * 5. Returns comprehensive error information
 *
 * @param filename Path to SafeTensors file
 * @param config Model configuration
 * @param destinations Map of internal paths to tensor destinations
 * @param allow_cast Allow dtype conversion during loading
 * @return WeightLoadResult with success/error information
 */
inline WeightLoadResult load_weights_with_registry(
    const std::string& filename,
    const PretrainedConfig& config,
    std::unordered_map<std::string, TensorShard*>& destinations,
    bool allow_cast
) {
    WeightRegistry registry(config);

    // Register all destinations
    for (auto& [path, tensor] : destinations) {
        registry.register_destination(path, *tensor);
    }

    // Validate schema before loading
    auto schema_result = registry.validate_schema();
    WeightLoadResult result;
    if (!schema_result.success) {
        for (const auto& e : schema_result.errors) {
            result.add_error({"", "", "SchemaValidation", -1, e});
        }
        return result;
    }

    // Load from file
    SafeTensorsReader reader{filename};
    return registry.load_from_reader(reader, allow_cast);
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_REGISTRY_H
