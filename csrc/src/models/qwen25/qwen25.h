// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string_view>

#include <nlohmann/json_fwd.hpp>

#include "config/pretrained_config.h"
#include "models/registry.h"
#include "models/qwen25/weight_mapping.h"

namespace models {

/**
 * @brief Architecture handler for Qwen2/Qwen2.5 family models.
 *
 * Returns Qwen2Config instances from load operations.
 */
class Qwen25Architecture {
public:
    static constexpr std::string_view kHfArchitectureName = "Qwen2ForCausalLM";

    static std::unique_ptr<PretrainedConfig> load_from_hf_config_json(const nlohmann::json& config_json, ETensorDType dtype);
    static void save_to_hf_config_json(const PretrainedConfig& config, nlohmann::json& config_json);
    static std::unique_ptr<PretrainedConfig> create_from_preset_name(std::string_view name, ETensorDType dtype);
    static std::unique_ptr<modules::BaseWeightMapping> create_weight_mapping() {
        return std::make_unique<modules::Qwen2WeightMapping>();
    }

    static ArchitectureOps ops() {
        return {
            .hf_architecture_name = kHfArchitectureName,
            .id = PretrainedConfig::QWEN2,
            .load_from_hf_config_json = &Qwen25Architecture::load_from_hf_config_json,
            .save_to_hf_config_json = &Qwen25Architecture::save_to_hf_config_json,
            .create_from_preset_name = &Qwen25Architecture::create_from_preset_name,
            .create_weight_mapping = &Qwen25Architecture::create_weight_mapping,
        };
    }
};

} // namespace models

