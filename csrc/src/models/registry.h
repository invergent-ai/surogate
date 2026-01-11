// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "config/pretrained_config.h"

namespace modules {
class BaseWeightMapping;
}

namespace models {

/**
 * @brief Operations for a specific model architecture.
 *
 * Each architecture (LLaMA, Qwen2, Qwen3, Qwen3MoE) provides these operations
 * for loading/saving configs and creating presets. The load functions return
 * the correct derived config type (LlamaConfig, Qwen3Config, etc.).
 */
struct ArchitectureOps {
    std::string_view hf_architecture_name;
    PretrainedConfig::ArchitectureId id;

    // Load config from HuggingFace config.json, returning the correct derived type
    std::unique_ptr<PretrainedConfig> (*load_from_hf_config_json)(const nlohmann::json& config_json, ETensorDType dtype);

    // Save config to HuggingFace config.json format
    void (*save_to_hf_config_json)(const PretrainedConfig& config, nlohmann::json& config_json);

    // Optional: create a pretrained config from a from-scratch preset name
    std::unique_ptr<PretrainedConfig> (*create_from_preset_name)(std::string_view name, ETensorDType dtype);

    // Create weight mapping for this architecture (for HuggingFace import/export)
    std::unique_ptr<modules::BaseWeightMapping> (*create_weight_mapping)();
};

const ArchitectureOps& architecture_from_hf_name(std::string_view hf_architecture_name);
const ArchitectureOps& architecture_from_id(PretrainedConfig::ArchitectureId id);

// Try all registered architectures for a from-scratch preset name.
std::unique_ptr<PretrainedConfig> create_from_preset_name(std::string_view name, ETensorDType dtype);

std::vector<std::string_view> supported_hf_architectures();

// Get all registered architecture IDs
std::vector<PretrainedConfig::ArchitectureId> supported_architecture_ids();

// Check if an architecture is registered by HF name
bool is_supported_hf_architecture(std::string_view hf_architecture_name);

// Check if an architecture is registered by ID
bool is_supported_architecture_id(PretrainedConfig::ArchitectureId id);

// Get all registered preset names across all architectures
std::vector<std::string_view> all_preset_names();

// Create weight mapping for an architecture by ID
std::unique_ptr<modules::BaseWeightMapping> create_weight_mapping(PretrainedConfig::ArchitectureId id);

} // namespace models

