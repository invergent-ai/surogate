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
#include <vector>

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

std::optional<std::vector<float>> as_float_array(const nlohmann::json& value) {
    if (!value.is_array()) return std::nullopt;
    std::vector<float> out;
    out.reserve(value.size());
    for (const auto& item : value) {
        if (auto v = as_float(item)) {
            out.push_back(*v);
        }
    }
    if (out.empty()) return std::nullopt;
    return out;
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

    const nlohmann::json* text_cfg = nullptr;
    if (config_json.contains("text_config") && config_json["text_config"].is_object()) {
        text_cfg = &config_json["text_config"];
    }
    const nlohmann::json* vision_cfg = nullptr;
    if (config_json.contains("vision_config") && config_json["vision_config"].is_object()) {
        vision_cfg = &config_json["vision_config"];
    }

    if (vision_cfg) {
        cfg->UseVisualInputs = true;
        if (vision_cfg->contains("deepstack_visual_indexes") &&
            (*vision_cfg)["deepstack_visual_indexes"].is_array()) {
            cfg->DeepstackVisualLayers =
                static_cast<int>((*vision_cfg)["deepstack_visual_indexes"].size());
        }
    }
    if (!cfg->UseVisualInputs && !cfg->ModelTypeName.empty()) {
        // Best-effort heuristic for vision-language models without explicit vision_config
        if (cfg->ModelTypeName.find("vl") != std::string::npos ||
            cfg->ModelTypeName.find("vision") != std::string::npos) {
            cfg->UseVisualInputs = true;
        }
    }
    if (cfg->UseVisualInputs && cfg->DeepstackVisualLayers == 0) {
        // Default to Qwen3-VL deepstack count when not specified.
        cfg->DeepstackVisualLayers = 3;
    }

    auto get_opt_int = [&](const char* key) -> std::optional<int> {
        if (auto v = get_opt<int>(config_json, key)) return v;
        if (text_cfg) return get_opt<int>(*text_cfg, key);
        return std::nullopt;
    };
    auto get_opt_float = [&](const char* key) -> std::optional<float> {
        if (auto v = get_opt<float>(config_json, key)) return v;
        if (text_cfg) return get_opt<float>(*text_cfg, key);
        return std::nullopt;
    };
    auto get_opt_bool = [&](const char* key) -> std::optional<bool> {
        if (auto v = get_opt<bool>(config_json, key)) return v;
        if (text_cfg) return get_opt<bool>(*text_cfg, key);
        return std::nullopt;
    };

    // Dimensions
    if (auto v = get_opt_int("hidden_size")) cfg->HiddenSize = *v;
    if (auto v = get_opt_int("intermediate_size")) cfg->IntermediateSize = *v;
    if (auto v = get_opt_int("vocab_size")) cfg->VocabSize = *v;
    if (auto v = get_opt_int("num_attention_heads")) cfg->NumQueryHeads = *v;
    if (auto v = get_opt_int("num_heads")) cfg->NumQueryHeads = *v;
    if (auto v = get_opt_int("num_key_value_heads")) {
        cfg->NumKeyValHeads = *v;
    } else {
        cfg->NumKeyValHeads = cfg->NumQueryHeads;
    }
    if (auto v = get_opt_int("num_hidden_layers")) cfg->NumLayers = *v;
    if (auto v = get_opt_int("num_layers")) cfg->NumLayers = *v;
    if (auto v = get_opt_int("head_dim")) cfg->HeadDim = *v;

    // Position + RoPE
    if (auto v = get_opt_int("max_position_embeddings")) cfg->MaxPositionEmbeddings = *v;
    if (auto v = get_opt_float("rope_theta")) cfg->RopeTheta = *v;
    cfg->Rope = RoPEConfig::full(cfg->RopeTheta);

    if (auto partial = get_opt_float("partial_rotary_factor")) {
        if (*partial > 0.0f && *partial < 1.0f) {
            cfg->Rope = RoPEConfig::partial(*partial, cfg->RopeTheta);
        }
    }

    const nlohmann::json* mrope_arr = nullptr;
    if (config_json.contains("mrope_section") && config_json["mrope_section"].is_array()) {
        mrope_arr = &config_json["mrope_section"];
    } else if (text_cfg && text_cfg->contains("mrope_section") && (*text_cfg)["mrope_section"].is_array()) {
        mrope_arr = &(*text_cfg)["mrope_section"];
    }
    if (mrope_arr) {
        const auto& arr = *mrope_arr;
        if (arr.size() >= 3) {
            const int t = as_int(arr[0]).value_or(0);
            const int h = as_int(arr[1]).value_or(0);
            const int w = as_int(arr[2]).value_or(0);
            cfg->Rope = RoPEConfig::multimodal(t, h, w, cfg->RopeTheta);
        }
    }

    const nlohmann::json* rope_scaling = nullptr;
    if (config_json.contains("rope_scaling") && config_json["rope_scaling"].is_object()) {
        rope_scaling = &config_json["rope_scaling"];
    } else if (text_cfg && (*text_cfg).contains("rope_scaling") && (*text_cfg)["rope_scaling"].is_object()) {
        rope_scaling = &(*text_cfg)["rope_scaling"];
    }

    if (rope_scaling) {
        const auto& scaling = *rope_scaling;
        if (auto rope_type = get_opt<std::string>(scaling, "rope_type")) {
            cfg->Rope.rope_type = *rope_type;
        } else if (auto rope_type = get_opt<std::string>(scaling, "type")) {
            cfg->Rope.rope_type = *rope_type;
        }
        if (auto factor = get_opt<float>(scaling, "factor")) {
            cfg->Rope.scaling_factor = *factor;
        }
        if (auto attention_factor = get_opt<float>(scaling, "attention_factor")) {
            cfg->Rope.attention_factor = *attention_factor;
        }
        if (auto beta_fast = get_opt<float>(scaling, "beta_fast")) {
            cfg->Rope.beta_fast = *beta_fast;
        }
        if (auto beta_slow = get_opt<float>(scaling, "beta_slow")) {
            cfg->Rope.beta_slow = *beta_slow;
        }
        if (auto mscale = get_opt<float>(scaling, "mscale")) {
            cfg->Rope.mscale = *mscale;
        }
        if (auto mscale_all_dim = get_opt<float>(scaling, "mscale_all_dim")) {
            cfg->Rope.mscale_all_dim = *mscale_all_dim;
        }
        if (auto orig_max = get_opt<int>(scaling, "original_max_position_embeddings")) {
            cfg->Rope.original_max_position_embeddings = *orig_max;
        }
        if (auto long_factor = as_float_array(scaling.value("long_factor", nlohmann::json()))) {
            cfg->Rope.long_factor = std::move(*long_factor);
        }
        if (auto short_factor = as_float_array(scaling.value("short_factor", nlohmann::json()))) {
            cfg->Rope.short_factor = std::move(*short_factor);
        }
        if (auto low_freq_factor = get_opt<float>(scaling, "low_freq_factor")) {
            cfg->Rope.low_freq_factor = *low_freq_factor;
        }
        if (auto high_freq_factor = get_opt<float>(scaling, "high_freq_factor")) {
            cfg->Rope.high_freq_factor = *high_freq_factor;
        }
        if (auto truncate = get_opt<bool>(scaling, "truncate")) {
            cfg->Rope.truncate = *truncate;
        }
        if (scaling.contains("mrope_section") && scaling["mrope_section"].is_array()) {
            const auto& arr = scaling["mrope_section"];
            if (arr.size() >= 3) {
                const int t = as_int(arr[0]).value_or(0);
                const int h = as_int(arr[1]).value_or(0);
                const int w = as_int(arr[2]).value_or(0);
                cfg->Rope = RoPEConfig::multimodal(t, h, w, cfg->RopeTheta);
            }
        }
    } else if (auto factor = get_opt<float>(config_json, "rope_scaling_factor")) {
        cfg->Rope.scaling_factor = *factor;
    }

    if (auto orig_max = get_opt<int>(config_json, "original_max_position_embeddings")) {
        cfg->Rope.original_max_position_embeddings_config = *orig_max;
    } else if (text_cfg) {
        if (auto orig_max = get_opt<int>(*text_cfg, "original_max_position_embeddings")) {
            cfg->Rope.original_max_position_embeddings_config = *orig_max;
        }
    }

    // Norm + tying
    if (auto v = get_opt_float("rms_norm_eps")) cfg->RmsNormEps = *v;
    if (auto v = get_opt_float("layer_norm_eps")) cfg->RmsNormEps = *v;
    if (auto v = get_opt<bool>(config_json, "tie_word_embeddings")) cfg->TiedWordEmbeddings = *v;
    if (auto v = get_opt<bool>(config_json, "tie_embeddings")) cfg->TiedWordEmbeddings = *v;

    // Attention flags
    if (auto v = get_opt_bool("attention_bias")) {
        cfg->UseQKVBias = *v;
    }
    if (auto v = get_opt_bool("qkv_bias")) {
        cfg->UseQKVBias = *v;
    }
    if (auto v = get_opt_bool("use_qkv_bias")) {
        cfg->UseQKVBias = *v;
    }

    if (auto v = get_opt_bool("use_qk_norm")) {
        cfg->UseQKNorm = *v;
    }
    if (auto v = get_opt_bool("qk_norm")) {
        cfg->UseQKNorm = *v;
    }

    // Sliding window attention (GPT-OSS)
    if (auto v = get_opt_int("sliding_window")) {
        cfg->SlidingWindow = *v;
    } else if (text_cfg) {
        if (auto v = get_opt<int>(*text_cfg, "sliding_window")) {
            cfg->SlidingWindow = *v;
        }
    }

    const nlohmann::json* layer_types = nullptr;
    if (config_json.contains("layer_types") && config_json["layer_types"].is_array()) {
        layer_types = &config_json["layer_types"];
    } else if (text_cfg && (*text_cfg).contains("layer_types") && (*text_cfg)["layer_types"].is_array()) {
        layer_types = &(*text_cfg)["layer_types"];
    }
    if (layer_types) {
        cfg->LayerTypes.clear();
        cfg->LayerTypes.reserve(layer_types->size());
        for (const auto& item : *layer_types) {
            if (!item.is_string()) {
                cfg->LayerTypes.push_back(0);
                continue;
            }
            const std::string v = item.get<std::string>();
            const bool sliding = (v.find("sliding") != std::string::npos);
            cfg->LayerTypes.push_back(sliding ? 1 : 0);
        }
    } else if (cfg->SlidingWindow > 0 && cfg->ModelTypeName == "gpt_oss" && cfg->NumLayers > 0) {
        // GPT-OSS default: alternating sliding/full attention starting with sliding.
        cfg->LayerTypes.resize(static_cast<std::size_t>(cfg->NumLayers));
        for (int i = 0; i < cfg->NumLayers; ++i) {
            cfg->LayerTypes[static_cast<std::size_t>(i)] = ((i + 1) % 2) ? 1 : 0;
        }
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
    if (config.SlidingWindow > 0) {
        config_json["sliding_window"] = config.SlidingWindow;
    }
    if (!config.LayerTypes.empty()) {
        nlohmann::json layer_types = nlohmann::json::array();
        for (int t : config.LayerTypes) {
            layer_types.push_back(t ? "sliding_attention" : "full_attention");
        }
        config_json["layer_types"] = std::move(layer_types);
    }

    nlohmann::json rope_scaling;
    bool has_rope_scaling = false;
    if (config.Rope.scaling_factor != 1.0f) {
        rope_scaling["factor"] = config.Rope.scaling_factor;
        has_rope_scaling = true;
    }
    if (!config.Rope.rope_type.empty() && config.Rope.rope_type != "default") {
        rope_scaling["rope_type"] = config.Rope.rope_type;
        has_rope_scaling = true;
    }
    if (config.Rope.attention_factor) {
        rope_scaling["attention_factor"] = *config.Rope.attention_factor;
        has_rope_scaling = true;
    }
    if (config.Rope.beta_fast) {
        rope_scaling["beta_fast"] = *config.Rope.beta_fast;
        has_rope_scaling = true;
    }
    if (config.Rope.beta_slow) {
        rope_scaling["beta_slow"] = *config.Rope.beta_slow;
        has_rope_scaling = true;
    }
    if (config.Rope.mscale) {
        rope_scaling["mscale"] = *config.Rope.mscale;
        has_rope_scaling = true;
    }
    if (config.Rope.mscale_all_dim) {
        rope_scaling["mscale_all_dim"] = *config.Rope.mscale_all_dim;
        has_rope_scaling = true;
    }
    if (config.Rope.original_max_position_embeddings) {
        rope_scaling["original_max_position_embeddings"] = *config.Rope.original_max_position_embeddings;
        has_rope_scaling = true;
    }
    if (!config.Rope.long_factor.empty()) {
        rope_scaling["long_factor"] = config.Rope.long_factor;
        has_rope_scaling = true;
    }
    if (!config.Rope.short_factor.empty()) {
        rope_scaling["short_factor"] = config.Rope.short_factor;
        has_rope_scaling = true;
    }
    if (config.Rope.low_freq_factor) {
        rope_scaling["low_freq_factor"] = *config.Rope.low_freq_factor;
        has_rope_scaling = true;
    }
    if (config.Rope.high_freq_factor) {
        rope_scaling["high_freq_factor"] = *config.Rope.high_freq_factor;
        has_rope_scaling = true;
    }
    if (config.Rope.truncate) {
        rope_scaling["truncate"] = *config.Rope.truncate;
        has_rope_scaling = true;
    }
    if (has_rope_scaling) {
        config_json["rope_scaling"] = rope_scaling;
    }
    if (config.Rope.is_partial()) {
        config_json["partial_rotary_factor"] = config.Rope.partial_factor;
    }
    if (config.Rope.is_multimodal()) {
        config_json["mrope_section"] = {config.Rope.mrope_section[0],
                                        config.Rope.mrope_section[1],
                                        config.Rope.mrope_section[2]};
    }
    if (config.Rope.original_max_position_embeddings_config) {
        config_json["original_max_position_embeddings"] = *config.Rope.original_max_position_embeddings_config;
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
