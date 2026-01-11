// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "models/qwen25.h"

#include <nlohmann/json.hpp>

#include "utilities/utils.h"

namespace models {

static std::unique_ptr<Qwen2Config> create_qwen25_config(int hidden_size, int intermediate_size, int q_heads, int kv_heads, int depth,
                                                          float rms, bool tied, ETensorDType dtype) {
    auto config = std::make_unique<Qwen2Config>();
    config->BosTokenId = 151643;
    config->EosTokenId = 151645;
    config->PadTokenId = 151643;
    config->HiddenSize = hidden_size;
    config->IntermediateSize = intermediate_size;
    config->VocabSize = 151936;
    config->NumQueryHeads = q_heads;
    config->NumKeyValHeads = kv_heads;
    config->NumLayers = depth;
    config->HeadDim = 0;
    config->MaxPositionEmbeddings = 32768;
    config->RopeTheta = 1'000'000.0f;
    config->RmsNormEps = rms;
    config->TiedWordEmbeddings = tied;
    config->UseQKVBias = true;
    config->UseQKNorm = false;
    config->SlidingWindow = 0;  // Qwen2-specific field
    config->DType = dtype;
    return config;
}

std::unique_ptr<PretrainedConfig> Qwen25Architecture::load_from_hf_config_json(const nlohmann::json& config_json, ETensorDType dtype) {
    auto result = std::make_unique<Qwen2Config>();
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
    result->HeadDim = config_json.value("head_dim", 0);

    result->MaxPositionEmbeddings = config_json.at("max_position_embeddings").get<int>();
    result->RopeTheta = config_json.at("rope_theta").get<float>();
    result->TiedWordEmbeddings = config_json.at("tie_word_embeddings").get<bool>();
    result->RmsNormEps = config_json.value("rms_norm_eps", 1e-6f);

    // Qwen2/Qwen2.5 commonly uses bias in Q/K/V projections; HF configs frequently omit the flag.
    result->UseQKVBias = config_json.value("attention_bias", true);
    result->UseQKNorm = false;

    // Qwen2-specific: sliding window
    result->SlidingWindow = config_json.value("sliding_window", 0);

    return result;
}

void Qwen25Architecture::save_to_hf_config_json(const PretrainedConfig& config, nlohmann::json& config_json) {
    config_json["architectures"] = {std::string(kHfArchitectureName)};
    config_json["model_type"] = "qwen2";
    config_json["bos_token_id"] = config.BosTokenId;
    config_json["eos_token_id"] = config.EosTokenId;
    config_json["pad_token_id"] = config.PadTokenId;
    config_json["hidden_size"] = config.HiddenSize;
    config_json["intermediate_size"] = config.IntermediateSize;
    config_json["vocab_size"] = config.VocabSize;
    config_json["num_attention_heads"] = config.NumQueryHeads;
    config_json["num_key_value_heads"] = config.NumKeyValHeads;
    config_json["num_hidden_layers"] = config.NumLayers;
    config_json["max_position_embeddings"] = config.MaxPositionEmbeddings;
    config_json["rope_theta"] = config.RopeTheta;
    config_json["rms_norm_eps"] = config.RmsNormEps;
    config_json["tie_word_embeddings"] = config.TiedWordEmbeddings;
    config_json["torch_dtype"] = dtype_to_torch_str(config.DType);

    // Keep a few HF fields for convenience / parity with other exporters.
    config_json["attention_dropout"] = 0.f;
    config_json["initializer_range"] = 0.02f;
    config_json["hidden_act"] = "silu";
    config_json["use_cache"] = true;
    config_json["max_window_layers"] = config.NumLayers;
    config_json["sliding_window"] = config.MaxPositionEmbeddings;
    config_json["use_sliding_window"] = false;
    config_json["use_mrope"] = false;
}

std::unique_ptr<PretrainedConfig> Qwen25Architecture::create_from_preset_name(std::string_view name, ETensorDType dtype) {
    if (iequals(name, "Qwen2.5-0.5B")) {
        return create_qwen25_config(896, 4864, 14, 2, 24, 1e-06f, true, dtype);
    }
    if (iequals(name, "Qwen2.5-1.5B")) {
        return create_qwen25_config(1536, 8960, 12, 2, 28, 1e-06f, true, dtype);
    }
    if (iequals(name, "Qwen2.5-3B")) {
        return create_qwen25_config(2048, 11008, 16, 2, 36, 1e-06f, true, dtype);
    }
    if (iequals(name, "Qwen2.5-7B")) {
        return create_qwen25_config(3584, 18944, 28, 4, 28, 1e-06f, false, dtype);
    }
    if (iequals(name, "Qwen2.5-14B")) {
        return create_qwen25_config(5120, 13824, 40, 8, 48, 1e-05f, false, dtype);
    }
    if (iequals(name, "Qwen2.5-32B")) {
        return create_qwen25_config(5120, 27648, 40, 8, 64, 1e-05f, false, dtype);
    }
    if (iequals(name, "Qwen2.5-72B")) {
        return create_qwen25_config(8192, 29568, 64, 8, 80, 1e-05f, false, dtype);
    }
    return nullptr;
}

} // namespace models

