// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "models/registry.h"

#include <stdexcept>

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include "models/llama/llama.h"
#include "models/qwen25/qwen25.h"
#include "models/qwen3/qwen3.h"
#include "models/qwen3moe/qwen3_moe.h"
#include "models/nemotron_h/nemotron_h.h"

namespace models {

static const std::vector<ArchitectureOps>& registry() {
    static const std::vector<ArchitectureOps> kRegistry = {
        LlamaArchitecture::ops(),
        Qwen25Architecture::ops(),
        Qwen3Architecture::ops(),
        Qwen3MoEArchitecture::ops(),
        NemotronHArchitecture::ops(),
    };
    return kRegistry;
}

const ArchitectureOps& architecture_from_hf_name(std::string_view hf_architecture_name) {
    for (const auto& ops : registry()) {
        if (ops.hf_architecture_name == hf_architecture_name) return ops;
    }

    // Build helpful error message listing supported architectures
    std::string supported;
    for (const auto& ops : registry()) {
        if (!supported.empty()) supported += ", ";
        supported += ops.hf_architecture_name;
    }
    throw std::runtime_error(fmt::format(
        "Unknown architecture '{}'. Supported architectures: {}",
        hf_architecture_name, supported));
}

const ArchitectureOps& architecture_from_id(PretrainedConfig::ArchitectureId id) {
    for (const auto& ops : registry()) {
        if (ops.id == id) return ops;
    }
    throw std::logic_error(fmt::format(
        "Unknown ArchitectureId {}. This is a programming error - all architecture IDs should be registered.",
        static_cast<int>(id)));
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

std::vector<PretrainedConfig::ArchitectureId> supported_architecture_ids() {
    std::vector<PretrainedConfig::ArchitectureId> out;
    out.reserve(registry().size());
    for (const auto& ops : registry()) out.push_back(ops.id);
    return out;
}

bool is_supported_hf_architecture(std::string_view hf_architecture_name) {
    for (const auto& ops : registry()) {
        if (ops.hf_architecture_name == hf_architecture_name) return true;
    }
    return false;
}

bool is_supported_architecture_id(PretrainedConfig::ArchitectureId id) {
    for (const auto& ops : registry()) {
        if (ops.id == id) return true;
    }
    return false;
}

std::vector<std::string_view> all_preset_names() {
    // Return well-known preset names (could be expanded by querying architectures)
    // For now, return a static list of documented presets
    return {
        // LLaMA presets
        "llama-2-7b", "llama-2-13b", "llama-3-8b",
        // Qwen2.5 presets
        "Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B",
        "Qwen2.5-14B", "Qwen2.5-32B", "Qwen2.5-72B",
        // Qwen3 presets
        "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B",
        "Qwen3-14B", "Qwen3-32B",
        // Qwen3 MoE presets
        "Qwen3-MoE-30B-A3B",
    };
}

std::unique_ptr<modules::BaseWeightMapping> create_weight_mapping(PretrainedConfig::ArchitectureId id) {
    const auto& ops = architecture_from_id(id);
    auto mapping = ops.create_weight_mapping();
    mapping->register_patterns();
    mapping->register_export_patterns();
    return mapping;
}

} // namespace models
