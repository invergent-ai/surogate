// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "models/qwen3.h"

#include <nlohmann/json.hpp>

#include "utilities/utils.h"

namespace models {

static std::unique_ptr<Qwen3Config> create_qwen3_config(
    int head_dim, int hidden_size, int intermediate_size, int max_position_embeddings,
    int num_attention_heads, int num_hidden_layers, int num_key_value_heads,
    float rms_norm_eps, float rope_theta, bool tie_word_embeddings, int vocab_size,
    ETensorDType dtype
) {
    auto config = std::make_unique<Qwen3Config>();
    config->BosTokenId = 151643;
    config->EosTokenId = 151645;
    config->PadTokenId = 151643;
    config->HiddenSize = hidden_size;
    config->IntermediateSize = intermediate_size;
    config->VocabSize = vocab_size;
    config->NumQueryHeads = num_attention_heads;
    config->NumKeyValHeads = num_key_value_heads;
    config->NumLayers = num_hidden_layers;
    config->HeadDim = head_dim;
    config->MaxPositionEmbeddings = max_position_embeddings;
    config->RopeTheta = rope_theta;
    config->RmsNormEps = rms_norm_eps;
    config->TiedWordEmbeddings = tie_word_embeddings;
    config->UseQKVBias = true;
    config->UseQKNorm = true;  // Qwen3 uses QK norm by default
    config->DType = dtype;
    return config;
}

std::unique_ptr<PretrainedConfig> Qwen3Architecture::load_from_hf_config_json(const nlohmann::json& config_json, ETensorDType dtype) {
    auto result = std::make_unique<Qwen3Config>();
    result->DType = dtype;

    result->BosTokenId = config_json.at("bos_token_id").get<int>();
    result->EosTokenId = config_json.at("eos_token_id").get<int>();
    result->PadTokenId = config_json.value("pad_token_id", result->BosTokenId);

    result->HiddenSize = config_json.at("hidden_size").get<int>();
    result->IntermediateSize = config_json.at("intermediate_size").get<int>();
    result->VocabSize = config_json.at("vocab_size").get<int>();
    result->NumQueryHeads = config_json.at("num_attention_heads").get<int>();
    result->NumKeyValHeads = config_json.at("num_key_value_heads").get<int>();
    result->NumLayers = config_json.at("num_hidden_layers").get<int>();

    // Qwen3 uses explicit head_dim, and q_proj output dim is num_attention_heads * head_dim.
    result->HeadDim = config_json.at("head_dim").get<int>();

    result->MaxPositionEmbeddings = config_json.at("max_position_embeddings").get<int>();
    result->RopeTheta = config_json.at("rope_theta").get<float>();
    result->TiedWordEmbeddings = config_json.at("tie_word_embeddings").get<bool>();
    result->RmsNormEps = config_json.value("rms_norm_eps", 1e-6f);

    // Qwen3 uses no attention bias by default.
    result->UseQKVBias = config_json.value("attention_bias", false);
    result->UseQKNorm = true;  // Qwen3 uses QK norm

    return result;
}

void Qwen3Architecture::save_to_hf_config_json(const PretrainedConfig& config, nlohmann::json& config_json) {
    config_json["architectures"] = {std::string(kHfArchitectureName)};
    config_json["model_type"] = "qwen3";
    config_json["bos_token_id"] = config.BosTokenId;
    config_json["eos_token_id"] = config.EosTokenId;
    config_json["pad_token_id"] = config.PadTokenId;
    config_json["hidden_size"] = config.HiddenSize;
    config_json["intermediate_size"] = config.IntermediateSize;
    config_json["vocab_size"] = config.VocabSize;
    config_json["num_attention_heads"] = config.NumQueryHeads;
    config_json["num_key_value_heads"] = config.NumKeyValHeads;
    config_json["num_hidden_layers"] = config.NumLayers;
    config_json["head_dim"] = config.HeadDim > 0 ? config.HeadDim : config.head_size();
    config_json["max_position_embeddings"] = config.MaxPositionEmbeddings;
    config_json["rope_theta"] = config.RopeTheta;
    config_json["rms_norm_eps"] = config.RmsNormEps;
    config_json["tie_word_embeddings"] = config.TiedWordEmbeddings;
    config_json["attention_bias"] = false;
    config_json["torch_dtype"] = dtype_to_torch_str(config.DType);
}

std::unique_ptr<PretrainedConfig> Qwen3Architecture::create_from_preset_name(std::string_view name, ETensorDType dtype) {
    if (iequals(name, "Qwen3-0.6B")) {
        return create_qwen3_config(128, 1024, 3072, 40960, 16, 28, 8, 1e-6f, 1000000.0f, true, 151936, dtype);
    }
    if (iequals(name, "Qwen3-1.7B")) {
        return create_qwen3_config(128, 2048, 6144, 40960, 16, 28, 8, 1e-6f, 1000000.0f, true, 151936, dtype);
    }
    if (iequals(name, "Qwen3-4B")) {
        return create_qwen3_config(128, 2560, 9728, 40960, 32, 36, 8, 1e-6f, 1000000.0f, true, 151936, dtype);
    }
    if (iequals(name, "Qwen3-8B")) {
        return create_qwen3_config(128, 4096, 12288, 40960, 32, 36, 8, 1e-6f, 1000000.0f, true, 151936, dtype);
    }
    if (iequals(name, "Qwen3-14B")) {
        return create_qwen3_config(128, 5120, 17408, 40960, 40, 40, 8, 1e-6f, 1000000.0f, true, 151936, dtype);
    }
    if (iequals(name, "Qwen3-32B")) {
        return create_qwen3_config(128, 5120, 25600, 40960, 64, 64, 8, 1e-6f, 1000000.0f, true, 151936, dtype);
    }

    return nullptr;
}

} // namespace models

