// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "models/qwen3moe/qwen3_moe.h"
#include "models/qwen3moe/config.h"

#include <nlohmann/json.hpp>

#include "utilities/utils.h"

namespace models {

static std::unique_ptr<Qwen3MoEConfig> create_qwen3_moe_config(
    int head_dim, int hidden_size, int intermediate_size, int moe_intermediate_size,
    int max_position_embeddings, int num_attention_heads, int num_hidden_layers,
    int num_key_value_heads, int num_experts, int num_experts_per_tok,
    int decoder_sparse_step, const std::vector<int>& mlp_only_layers,
    bool norm_topk_prob, float router_aux_loss_coef,
    float rms_norm_eps, float rope_theta, bool tie_word_embeddings, int vocab_size,
    ETensorDType dtype
) {
    auto config = std::make_unique<Qwen3MoEConfig>();
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
    config->UseQKVBias = false;
    config->UseQKNorm = true;  // Qwen3 uses QK norm
    config->DType = dtype;

    // MoE-specific fields (in Qwen3MoEConfig)
    config->NumExperts = num_experts;
    config->NumExpertsPerTok = num_experts_per_tok;
    config->MoeIntermediateSize = moe_intermediate_size;
    config->DecoderSparseStep = decoder_sparse_step;
    config->MlpOnlyLayers = mlp_only_layers;
    config->NormTopkProb = norm_topk_prob;
    config->RouterAuxLossCoef = router_aux_loss_coef;

    return config;
}

std::unique_ptr<PretrainedConfig> Qwen3MoEArchitecture::load_from_hf_config_json(const nlohmann::json& config_json, ETensorDType dtype) {
    auto result = std::make_unique<Qwen3MoEConfig>();
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

    // Qwen3 MoE uses explicit head_dim
    result->HeadDim = config_json.at("head_dim").get<int>();

    result->MaxPositionEmbeddings = config_json.at("max_position_embeddings").get<int>();
    result->RopeTheta = config_json.at("rope_theta").get<float>();
    result->TiedWordEmbeddings = config_json.at("tie_word_embeddings").get<bool>();
    result->RmsNormEps = config_json.value("rms_norm_eps", 1e-6f);

    // Qwen3 MoE uses no attention bias by default
    result->UseQKVBias = config_json.value("attention_bias", false);
    // Qwen3 MoE uses QK normalization
    result->UseQKNorm = true;

    // MoE-specific configuration (in Qwen3MoEConfig)
    result->NumExperts = config_json.at("num_experts").get<int>();
    result->NumExpertsPerTok = config_json.at("num_experts_per_tok").get<int>();
    result->MoeIntermediateSize = config_json.value("moe_intermediate_size", result->IntermediateSize);
    result->DecoderSparseStep = config_json.value("decoder_sparse_step", 1);
    result->NormTopkProb = config_json.value("norm_topk_prob", false);
    result->RouterAuxLossCoef = config_json.value("router_aux_loss_coef", 0.001f);
    result->RouterZLossCoef = config_json.value("router_z_loss_coef", 0.001f);

    // Parse mlp_only_layers array if present
    if (config_json.contains("mlp_only_layers") && config_json["mlp_only_layers"].is_array()) {
        result->MlpOnlyLayers = config_json["mlp_only_layers"].get<std::vector<int>>();
    }

    return result;
}

void Qwen3MoEArchitecture::save_to_hf_config_json(const PretrainedConfig& config, nlohmann::json& config_json) {
    // Cast to Qwen3MoEConfig to access MoE-specific fields
    const auto* moe_config = dynamic_cast<const Qwen3MoEConfig*>(&config);

    config_json["architectures"] = {std::string(kHfArchitectureName)};
    config_json["model_type"] = "qwen3_moe";
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

    // MoE-specific fields (from Qwen3MoEConfig)
    if (moe_config) {
        config_json["num_experts"] = moe_config->NumExperts;
        config_json["num_experts_per_tok"] = moe_config->NumExpertsPerTok;
        config_json["moe_intermediate_size"] = moe_config->MoeIntermediateSize;
        config_json["decoder_sparse_step"] = moe_config->DecoderSparseStep;
        config_json["norm_topk_prob"] = moe_config->NormTopkProb;
        config_json["router_aux_loss_coef"] = moe_config->RouterAuxLossCoef;
        config_json["router_z_loss_coef"] = moe_config->RouterZLossCoef;

        if (!moe_config->MlpOnlyLayers.empty()) {
            config_json["mlp_only_layers"] = moe_config->MlpOnlyLayers;
        }
    }
}

std::unique_ptr<PretrainedConfig> Qwen3MoEArchitecture::create_from_preset_name(std::string_view name, ETensorDType dtype) {
    // Qwen3-MoE-30B-A3B: 128 experts, 8 active per token
    // Total params: ~30B, Active params: ~3B per token
    if (iequals(name, "Qwen3-MoE-30B-A3B") || iequals(name, "qwen3_moe_30b_a3b")) {
        return create_qwen3_moe_config(
            /*head_dim=*/128,
            /*hidden_size=*/2048,
            /*intermediate_size=*/6144,  // Dense MLP intermediate size
            /*moe_intermediate_size=*/768, // Expert intermediate size
            /*max_position_embeddings=*/262144,
            /*num_attention_heads=*/32,
            /*num_hidden_layers=*/48,
            /*num_key_value_heads=*/4,
            /*num_experts=*/128,
            /*num_experts_per_tok=*/8,
            /*decoder_sparse_step=*/1,
            /*mlp_only_layers=*/{},  // Empty - all layers are MoE
            /*norm_topk_prob=*/true,
            /*router_aux_loss_coef=*/0.001f,
            /*rms_norm_eps=*/1e-6f,
            /*rope_theta=*/10000000.0f,
            /*tie_word_embeddings=*/false,
            /*vocab_size=*/151936,
            dtype
        );
    }

    return nullptr;
}

} // namespace models
