// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// NemotronH architecture registry entry.

#pragma once

#include <memory>
#include <string_view>

#include <nlohmann/json_fwd.hpp>

#include "config/pretrained_config.h"
#include "models/registry.h"
#include "models/nemotron_h/weight_mapping.h"

namespace models {

class NemotronHArchitecture {
public:
    static constexpr std::string_view kHfArchitectureName = "NemotronHForCausalLM";

    static std::unique_ptr<PretrainedConfig> load_from_hf_config_json(const nlohmann::json& config_json, ETensorDType dtype);
    static void save_to_hf_config_json(const PretrainedConfig& config, nlohmann::json& config_json);
    static std::unique_ptr<PretrainedConfig> create_from_preset_name(std::string_view, ETensorDType) {
        return nullptr;
    }
    static std::unique_ptr<modules::BaseWeightMapping> create_weight_mapping() {
        return std::make_unique<modules::NemotronHWeightMapping>();
    }

    static ArchitectureOps ops() {
        return {
            .hf_architecture_name = kHfArchitectureName,
            .id = PretrainedConfig::NEMOTRON_H,
            .load_from_hf_config_json = &NemotronHArchitecture::load_from_hf_config_json,
            .save_to_hf_config_json = &NemotronHArchitecture::save_to_hf_config_json,
            .create_from_preset_name = &NemotronHArchitecture::create_from_preset_name,
            .create_weight_mapping = &NemotronHArchitecture::create_weight_mapping,
        };
    }
};

} // namespace models
