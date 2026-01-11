// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "models/llama.h"

#include <nlohmann/json.hpp>

#include <fmt/core.h>

#include "utilities/utils.h"

namespace models {

static std::unique_ptr<LlamaConfig> create_llama2_config(int hidden_size, int intermediate_size, int heads, int depth, ETensorDType dtype) {
    auto config = std::make_unique<LlamaConfig>();
    config->BosTokenId = 1;
    config->EosTokenId = 2;
    config->PadTokenId = 0;
    config->HiddenSize = hidden_size;
    config->IntermediateSize = intermediate_size;
    config->VocabSize = 32000;
    config->NumQueryHeads = heads;
    config->NumKeyValHeads = heads;
    config->NumLayers = depth;
    config->HeadDim = 0;
    config->MaxPositionEmbeddings = 4096;
    config->RopeTheta = 10000.f;
    config->RmsNormEps = 1e-05f;
    config->TiedWordEmbeddings = false;
    config->UseQKVBias = false;
    config->UseQKNorm = false;
    config->DType = dtype;
    return config;
}

static std::unique_ptr<LlamaConfig> create_llama3_config(int hidden_size, int intermediate_size, int q_heads, int kv_heads, int depth,
                                                          ETensorDType dtype) {
    auto config = std::make_unique<LlamaConfig>();
    config->BosTokenId = 128000;
    config->EosTokenId = 128001;
    config->PadTokenId = 128255;
    config->HiddenSize = hidden_size;
    config->IntermediateSize = intermediate_size;
    config->VocabSize = 128256;
    config->NumQueryHeads = q_heads;
    config->NumKeyValHeads = kv_heads;
    config->NumLayers = depth;
    config->HeadDim = 0;
    config->MaxPositionEmbeddings = 4096;
    config->RopeTheta = 500000.f;
    config->RmsNormEps = 1e-05f;
    config->TiedWordEmbeddings = false;
    config->UseQKVBias = false;
    config->UseQKNorm = false;
    config->DType = dtype;
    return config;
}

std::unique_ptr<PretrainedConfig> LlamaArchitecture::load_from_hf_config_json(const nlohmann::json& config_json, ETensorDType dtype) {
    auto result = std::make_unique<LlamaConfig>();
    result->DType = dtype;

    result->BosTokenId = config_json.at("bos_token_id").get<int>();
    result->EosTokenId = config_json.at("eos_token_id").get<int>();
    result->PadTokenId = config_json.value("pad_token_id", 0);

    result->HiddenSize = config_json.at("hidden_size").get<int>();
    result->IntermediateSize = config_json.at("intermediate_size").get<int>();
    result->VocabSize = config_json.at("vocab_size").get<int>();
    result->NumQueryHeads = config_json.at("num_attention_heads").get<int>();
    result->NumKeyValHeads = config_json.at("num_key_value_heads").get<int>();
    result->NumLayers = config_json.at("num_hidden_layers").get<int>();
    result->HeadDim = config_json.value("head_dim", 0);

    result->MaxPositionEmbeddings = config_json.at("max_position_embeddings").get<int>();
    result->RopeTheta = config_json.value("rope_theta", 10000.0f);
    result->TiedWordEmbeddings = config_json.value("tie_word_embeddings", false);
    result->RmsNormEps = config_json.value("rms_norm_eps", 1e-05f);

    result->UseQKVBias = config_json.value("attention_bias", false);
    result->UseQKNorm = false;

    return result;
}

void LlamaArchitecture::save_to_hf_config_json(const PretrainedConfig& config, nlohmann::json& config_json) {
    config_json["architectures"] = {std::string(kHfArchitectureName)};
    config_json["model_type"] = "llama";
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
    config_json["attention_bias"] = false;
    config_json["mlp_bias"] = false;
    config_json["torch_dtype"] = dtype_to_torch_str(config.DType);
}

std::unique_ptr<PretrainedConfig> LlamaArchitecture::create_from_preset_name(std::string_view name, ETensorDType dtype) {
    if (iequals(name, "llama-2-7b")) {
        return create_llama2_config(4096, 11008, 32, 32, dtype);
    }
    if (iequals(name, "llama-2-13b")) {
        return create_llama2_config(5120, 13824, 40, 40, dtype);
    }
    if (iequals(name, "llama-3-8b")) {
        return create_llama3_config(4096, 14336, 32, 8, 32, dtype);
    }
    return nullptr;
}

} // namespace models

