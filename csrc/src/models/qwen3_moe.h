// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string_view>

#include <nlohmann/json_fwd.hpp>

#include "config/pretrained_config.h"
#include "models/registry.h"

namespace models {

/**
 * @brief Architecture handler for Qwen3 MoE models.
 *
 * Returns Qwen3MoEConfig instances from load operations.
 */
class Qwen3MoEArchitecture {
public:
    static constexpr std::string_view kHfArchitectureName = "Qwen3MoeForCausalLM";

    static std::unique_ptr<PretrainedConfig> load_from_hf_config_json(const nlohmann::json& config_json, ETensorDType dtype);
    static void save_to_hf_config_json(const PretrainedConfig& config, nlohmann::json& config_json);
    static std::unique_ptr<PretrainedConfig> create_from_preset_name(std::string_view name, ETensorDType dtype);

    static ArchitectureOps ops() {
        return {
            .hf_architecture_name = kHfArchitectureName,
            .id = PretrainedConfig::QWEN3_MOE,
            .load_from_hf_config_json = &Qwen3MoEArchitecture::load_from_hf_config_json,
            .save_to_hf_config_json = &Qwen3MoEArchitecture::save_to_hf_config_json,
            .create_from_preset_name = &Qwen3MoEArchitecture::create_from_preset_name,
        };
    }
};

} // namespace models
