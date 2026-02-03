// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "config/pretrained_config.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace {

std::optional<int> as_int(const nlohmann::json& value) {
    if (value.is_number_integer()) return value.get<int>();
    if (value.is_number_unsigned()) return static_cast<int>(value.get<std::uint64_t>());
    if (value.is_number_float()) return static_cast<int>(value.get<double>());
    if (value.is_string()) {
        try {
            return std::stoi(value.get<std::string>());
        } catch (...) {
            return std::nullopt;
        }
    }
    if (value.is_array() && !value.empty()) {
        return as_int(value.front());
    }
    return std::nullopt;
}

std::optional<float> as_float(const nlohmann::json& value) {
    if (value.is_number_float() || value.is_number_integer() || value.is_number_unsigned()) {
        return static_cast<float>(value.get<double>());
    }
    if (value.is_string()) {
        try {
            return std::stof(value.get<std::string>());
        } catch (...) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

std::optional<bool> as_bool(const nlohmann::json& value) {
    if (value.is_boolean()) return value.get<bool>();
    if (value.is_number_integer()) return value.get<int>() != 0;
    if (value.is_string()) {
        const std::string v = value.get<std::string>();
        if (iequals(v, "true") || v == "1") return true;
        if (iequals(v, "false") || v == "0") return false;
    }
    return std::nullopt;
}

template<typename T>
std::optional<T> get_opt(const nlohmann::json& obj, const char* key) {
    auto it = obj.find(key);
    if (it == obj.end()) return std::nullopt;
    if constexpr (std::is_same_v<T, int>) {
        return as_int(*it);
    } else if constexpr (std::is_same_v<T, float>) {
        return as_float(*it);
    } else if constexpr (std::is_same_v<T, bool>) {
        return as_bool(*it);
    } else if constexpr (std::is_same_v<T, std::string>) {
        if (it->is_string()) return it->get<std::string>();
    }
    return std::nullopt;
}

}  // namespace

std::unique_ptr<PretrainedConfig> load_pretrained_config(const char* file_name, ETensorDType dtype) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        throw std::runtime_error(fmt::format("could not open config file {}", file_name));
    }

    const auto config_json = nlohmann::json::parse(file);

    auto cfg = std::make_unique<PretrainedConfig>();
    cfg->DType = dtype;

    // Architecture/model type (best-effort, stored as raw strings)
    if (config_json.contains("architectures") && config_json["architectures"].is_array() &&
        !config_json["architectures"].empty()) {
        if (auto arch = config_json["architectures"].front().get<std::string>(); !arch.empty()) {
            cfg->ArchitectureName = std::move(arch);
        }
    }
    if (auto model_type = get_opt<std::string>(config_json, "model_type")) {
        cfg->ModelTypeName = *model_type;
    }

    // Token IDs
    if (auto bos = get_opt<int>(config_json, "bos_token_id")) cfg->BosTokenId = *bos;
    if (auto eos = get_opt<int>(config_json, "eos_token_id")) cfg->EosTokenId = *eos;
    if (auto pad = get_opt<int>(config_json, "pad_token_id")) {
        cfg->PadTokenId = *pad;
    }

    // Dimensions
    if (auto v = get_opt<int>(config_json, "hidden_size")) cfg->HiddenSize = *v;
    if (auto v = get_opt<int>(config_json, "intermediate_size")) cfg->IntermediateSize = *v;
    if (auto v = get_opt<int>(config_json, "vocab_size")) cfg->VocabSize = *v;
    if (auto v = get_opt<int>(config_json, "num_attention_heads")) cfg->NumQueryHeads = *v;
    if (auto v = get_opt<int>(config_json, "num_heads")) cfg->NumQueryHeads = *v;
    if (auto v = get_opt<int>(config_json, "num_key_value_heads")) {
        cfg->NumKeyValHeads = *v;
    } else {
        cfg->NumKeyValHeads = cfg->NumQueryHeads;
    }
    if (auto v = get_opt<int>(config_json, "num_hidden_layers")) cfg->NumLayers = *v;
    if (auto v = get_opt<int>(config_json, "num_layers")) cfg->NumLayers = *v;
    if (auto v = get_opt<int>(config_json, "head_dim")) cfg->HeadDim = *v;

    // Position + RoPE
    if (auto v = get_opt<int>(config_json, "max_position_embeddings")) cfg->MaxPositionEmbeddings = *v;
    if (auto v = get_opt<float>(config_json, "rope_theta")) cfg->RopeTheta = *v;
    cfg->Rope = RoPEConfig::full(cfg->RopeTheta);

    if (auto partial = get_opt<float>(config_json, "partial_rotary_factor")) {
        if (*partial > 0.0f && *partial < 1.0f) {
            cfg->Rope = RoPEConfig::partial(*partial, cfg->RopeTheta);
        }
    }

    if (config_json.contains("mrope_section") && config_json["mrope_section"].is_array()) {
        const auto& arr = config_json["mrope_section"];
        if (arr.size() >= 3) {
            const int t = as_int(arr[0]).value_or(0);
            const int h = as_int(arr[1]).value_or(0);
            const int w = as_int(arr[2]).value_or(0);
            cfg->Rope = RoPEConfig::multimodal(t, h, w, cfg->RopeTheta);
        }
    }

    if (config_json.contains("rope_scaling") && config_json["rope_scaling"].is_object()) {
        const auto& scaling = config_json["rope_scaling"];
        if (auto factor = get_opt<float>(scaling, "factor")) {
            cfg->Rope.scaling_factor = *factor;
        }
    } else if (auto factor = get_opt<float>(config_json, "rope_scaling_factor")) {
        cfg->Rope.scaling_factor = *factor;
    }

    // Norm + tying
    if (auto v = get_opt<float>(config_json, "rms_norm_eps")) cfg->RmsNormEps = *v;
    if (auto v = get_opt<float>(config_json, "layer_norm_eps")) cfg->RmsNormEps = *v;
    if (auto v = get_opt<bool>(config_json, "tie_word_embeddings")) cfg->TiedWordEmbeddings = *v;
    if (auto v = get_opt<bool>(config_json, "tie_embeddings")) cfg->TiedWordEmbeddings = *v;

    // Attention flags
    if (auto v = get_opt<bool>(config_json, "attention_bias")) {
        cfg->UseQKVBias = *v;
    }
    if (auto v = get_opt<bool>(config_json, "qkv_bias")) {
        cfg->UseQKVBias = *v;
    }
    if (auto v = get_opt<bool>(config_json, "use_qkv_bias")) {
        cfg->UseQKVBias = *v;
    }

    if (auto v = get_opt<bool>(config_json, "use_qk_norm")) {
        cfg->UseQKNorm = *v;
    }
    if (auto v = get_opt<bool>(config_json, "qk_norm")) {
        cfg->UseQKNorm = *v;
    }

    return cfg;
}

[[nodiscard]] std::string_view PretrainedConfig::model_name() const {
    if (!ArchitectureName.empty()) {
        return ArchitectureName;
    }
    if (!ModelTypeName.empty()) {
        return ModelTypeName;
    }
    return "custom";
}

void save_pretrained_config(const PretrainedConfig& config, const char* file_name) {
    std::ofstream file(file_name);
    if (!file.is_open()) {
        throw std::runtime_error(fmt::format("could not open file for writing {}", file_name));
    }

    nlohmann::json config_json;
    if (!config.ArchitectureName.empty()) {
        config_json["architectures"] = {config.ArchitectureName};
    }
    if (!config.ModelTypeName.empty()) {
        config_json["model_type"] = config.ModelTypeName;
    }
    config_json["bos_token_id"] = config.BosTokenId;
    config_json["eos_token_id"] = config.EosTokenId;
    config_json["pad_token_id"] = config.PadTokenId;
    config_json["hidden_size"] = config.HiddenSize;
    config_json["intermediate_size"] = config.IntermediateSize;
    config_json["vocab_size"] = config.VocabSize;
    config_json["num_attention_heads"] = config.NumQueryHeads;
    config_json["num_key_value_heads"] = config.NumKeyValHeads;
    config_json["num_hidden_layers"] = config.NumLayers;
    if (config.HeadDim > 0) {
        config_json["head_dim"] = config.HeadDim;
    }
    config_json["max_position_embeddings"] = config.MaxPositionEmbeddings;
    config_json["rope_theta"] = config.RopeTheta;
    config_json["rms_norm_eps"] = config.RmsNormEps;
    config_json["tie_word_embeddings"] = config.TiedWordEmbeddings;
    config_json["attention_bias"] = config.UseQKVBias;
    config_json["torch_dtype"] = dtype_to_torch_str(config.DType);

    if (config.Rope.scaling_factor != 1.0f) {
        config_json["rope_scaling"] = {{"factor", config.Rope.scaling_factor}};
    }
    if (config.Rope.is_partial()) {
        config_json["partial_rotary_factor"] = config.Rope.partial_factor;
    }
    if (config.Rope.is_multimodal()) {
        config_json["mrope_section"] = {config.Rope.mrope_section[0],
                                        config.Rope.mrope_section[1],
                                        config.Rope.mrope_section[2]};
    }

    file << config_json.dump(4);
}

std::unique_ptr<PretrainedConfig> create_pretrained_config_from_name(std::string_view name, ETensorDType dtype) {
    if (name.empty()) {
        throw std::runtime_error("create_pretrained_config_from_name: empty name");
    }
    std::filesystem::path path{std::string(name)};
    if (std::filesystem::exists(path)) {
        if (std::filesystem::is_directory(path)) {
            path /= "config.json";
        }
        if (std::filesystem::exists(path)) {
            return load_pretrained_config(path.string().c_str(), dtype);
        }
    }
    throw std::runtime_error(
        "create_pretrained_config_from_name: presets removed; pass a config.json path or use from_pretrained");
}
