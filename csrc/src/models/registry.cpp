// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "models/registry.h"

#include <stdexcept>

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include "models/llama.h"
#include "models/qwen25.h"
#include "models/qwen3.h"
#include "models/qwen3_moe.h"

namespace models {

static const std::vector<ArchitectureOps>& registry() {
    static const std::vector<ArchitectureOps> kRegistry = {
        LlamaArchitecture::ops(),
        Qwen25Architecture::ops(),
        Qwen3Architecture::ops(),
        Qwen3MoEArchitecture::ops(),
    };
    return kRegistry;
}

const ArchitectureOps& architecture_from_hf_name(std::string_view hf_architecture_name) {
    for (const auto& ops : registry()) {
        if (ops.hf_architecture_name == hf_architecture_name) return ops;
    }
    throw std::runtime_error(fmt::format("unknown architecture {}", hf_architecture_name));
}

const ArchitectureOps& architecture_from_id(PretrainedConfig::ArchitectureId id) {
    for (const auto& ops : registry()) {
        if (ops.id == id) return ops;
    }
    throw std::logic_error("unknown ArchitectureId");
}

std::unique_ptr<PretrainedConfig> create_from_preset_name(std::string_view name, ETensorDType dtype) {
    for (const auto& ops : registry()) {
        if (!ops.create_from_preset_name) continue;
        if (auto cfg = ops.create_from_preset_name(name, dtype)) return cfg;
    }
    return nullptr;
}

std::vector<std::string_view> supported_hf_architectures() {
    std::vector<std::string_view> out;
    out.reserve(registry().size());
    for (const auto& ops : registry()) out.push_back(ops.hf_architecture_name);
    return out;
}

} // namespace models

