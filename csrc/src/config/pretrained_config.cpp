// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "config/pretrained_config.h"

#include "models/registry.h"
#include "utilities/utils.h"

#include <fstream>

#include <fmt/core.h>
#include <nlohmann/json.hpp>

std::unique_ptr<PretrainedConfig> load_pretrained_config(const char* file_name, ETensorDType dtype) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        throw std::runtime_error(fmt::format("could not open config file {}", file_name));
    }

    auto config_json = nlohmann::json::parse(file);

    auto archs = config_json.at("architectures").get<std::vector<std::string>>();
    if (archs.size() != 1) {
        throw std::runtime_error("got multiple values for architecture");
    }
    const auto& ops = models::architecture_from_hf_name(archs.front());
    return ops.load_from_hf_config_json(config_json, dtype);
}

// Legacy compatibility function - returns a copy
PretrainedConfig load_pretrained_config_legacy(const char* file_name, ETensorDType dtype) {
    auto cfg = load_pretrained_config(file_name, dtype);
    return *cfg;  // Return a copy of the config
}

[[nodiscard]] std::string_view PretrainedConfig::model_name() const {
    switch (Architecture) {
        case PretrainedConfig::QWEN2:
            return "Qwen2";
        case PretrainedConfig::QWEN3:
            return "Qwen3";
        case PretrainedConfig::QWEN3_MOE:
            return "Qwen3-MoE";
        case PretrainedConfig::LLAMA:
            return "LLaMA";
        default:
            throw std::logic_error("Unknown architecture");
    }
}

void save_pretrained_config(const PretrainedConfig& config, const char* file_name) {
    std::ofstream file(file_name);
    if (!file.is_open()) {
        throw std::runtime_error(fmt::format("could not open file for writing {}", file_name));
    }

    nlohmann::json config_json;
    const auto& ops = models::architecture_from_id(config.Architecture);
    ops.save_to_hf_config_json(config, config_json);

    file << config_json.dump(4);
}

std::unique_ptr<PretrainedConfig> create_pretrained_config_from_name(std::string_view name, ETensorDType dtype) {
    if (auto cfg = models::create_from_preset_name(name, dtype)) return cfg;
    throw std::runtime_error(fmt::format("unknown model name {}", name));
}

// Legacy compatibility function - returns a copy
PretrainedConfig create_pretrained_config_from_name_legacy(std::string_view name, ETensorDType dtype) {
    auto cfg = create_pretrained_config_from_name(name, dtype);
    return *cfg;  // Return a copy of the config
}
