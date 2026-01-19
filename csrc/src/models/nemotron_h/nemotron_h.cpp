// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Nemotron-H config loader/saver.

#include "models/nemotron_h/nemotron_h.h"
#include "models/nemotron_h/config.h"

#include <nlohmann/json.hpp>

#include "utilities/utils.h"

namespace models {

static void maybe_fill_layers_from_pattern(NemotronHConfig& cfg) {
    if (!cfg.LayersBlockType.empty()) return;
    if (cfg.HybridOverridePattern.empty()) return;

    cfg.LayersBlockType.reserve(cfg.HybridOverridePattern.size());
    for (char c : cfg.HybridOverridePattern) {
        switch (c) {
            case 'M':
                cfg.LayersBlockType.emplace_back("mamba");
                break;
            case 'E':
                cfg.LayersBlockType.emplace_back("moe");
                break;
            case '*':
                cfg.LayersBlockType.emplace_back("attention");
                break;
            case '-':
                cfg.LayersBlockType.emplace_back("mlp");
                break;
            default:
                cfg.LayersBlockType.emplace_back("attention");
                break;
        }
    }
}

std::unique_ptr<PretrainedConfig> NemotronHArchitecture::load_from_hf_config_json(const nlohmann::json& config_json, ETensorDType dtype) {
    auto cfg = std::make_unique<NemotronHConfig>();
    cfg->DType = dtype;

    cfg->BosTokenId = config_json.value("bos_token_id", 1);
    cfg->EosTokenId = config_json.value("eos_token_id", 2);
    cfg->PadTokenId = config_json.value("pad_token_id", cfg->BosTokenId);

    cfg->HiddenSize = config_json.at("hidden_size").get<int>();
    cfg->IntermediateSize = config_json.at("intermediate_size").get<int>();
    cfg->VocabSize = config_json.at("vocab_size").get<int>();
    cfg->NumQueryHeads = config_json.at("num_attention_heads").get<int>();
    cfg->NumKeyValHeads = config_json.value("num_key_value_heads", cfg->NumQueryHeads);
    cfg->NumLayers = config_json.at("num_hidden_layers").get<int>();

    cfg->HeadDim = config_json.value("head_dim", 0);
    cfg->MaxPositionEmbeddings = config_json.value("max_position_embeddings", 2048);

    cfg->RmsNormEps = config_json.value("layer_norm_epsilon", config_json.value("rms_norm_eps", 1e-6f));
    cfg->UseQKVBias = config_json.value("attention_bias", false);
    cfg->AttentionBias = config_json.value("attention_bias", false);
    cfg->MlpBias = config_json.value("mlp_bias", false);
    cfg->AttentionDropout = config_json.value("attention_dropout", 0.0f);
    cfg->ResidualInFp32 = config_json.value("residual_in_fp32", false);

    cfg->RopeTheta = config_json.value("rope_theta", 10000.0f);
    cfg->Rope = RoPEConfig::none();

    // Mamba-specific settings
    cfg->MambaNumHeads = config_json.value("mamba_num_heads", 0);
    cfg->MambaHeadDim = config_json.value("mamba_head_dim", 0);
    cfg->SsmStateSize = config_json.value("ssm_state_size", 0);
    cfg->ConvKernel = config_json.value("conv_kernel", 0);
    cfg->NGroups = config_json.value("n_groups", 1);
    cfg->ChunkSize = config_json.value("chunk_size", 0);
    cfg->TimeStepMin = config_json.value("time_step_min", 0.0f);
    cfg->TimeStepMax = config_json.value("time_step_max", 0.0f);
    cfg->TimeStepFloor = config_json.value("time_step_floor", 0.0f);
    cfg->TimeStepLimit = config_json.value("time_step_limit", 0.0f);
    cfg->UseBias = config_json.value("use_bias", false);
    cfg->UseConvBias = config_json.value("use_conv_bias", false);
    cfg->MambaHiddenAct = config_json.value("mamba_hidden_act", std::string("silu"));
    cfg->MlpHiddenAct = config_json.value("mlp_hidden_act", std::string("silu"));

    // MoE-specific settings (Nemotron-H hybrid MoE)
    cfg->NRoutedExperts = config_json.value("n_routed_experts", 0);
    cfg->NumExpertsPerTok = config_json.value("num_experts_per_tok", 0);
    cfg->MoeIntermediateSize = config_json.value("moe_intermediate_size", 0);
    cfg->MoeSharedExpertIntermediateSize = config_json.value("moe_shared_expert_intermediate_size", 0);
    cfg->NSharedExperts = config_json.value("n_shared_experts", 0);
    cfg->NormTopkProb = config_json.value("norm_topk_prob", false);
    cfg->RoutedScalingFactor = config_json.value("routed_scaling_factor", 1.0f);
    cfg->TopkGroup = config_json.value("topk_group", 1);
    cfg->NGroup = config_json.value("n_group", 1);
    cfg->RouterAuxLossCoef = config_json.value("router_aux_loss_coef", 0.01f);
    cfg->RouterZLossCoef = config_json.value("router_z_loss_coef", 0.001f);

    if (config_json.contains("layers_block_type")) {
        for (const auto& entry : config_json.at("layers_block_type")) {
            cfg->LayersBlockType.push_back(entry.get<std::string>());
        }
    }

    if (config_json.contains("hybrid_override_pattern")) {
        cfg->HybridOverridePattern = config_json.at("hybrid_override_pattern").get<std::string>();
    }

    maybe_fill_layers_from_pattern(*cfg);

    if (cfg->LayersBlockType.empty() && cfg->NumLayers > 0) {
        cfg->LayersBlockType.assign(cfg->NumLayers, "attention");
    }

    return cfg;
}

void NemotronHArchitecture::save_to_hf_config_json(const PretrainedConfig& config, nlohmann::json& config_json) {
    const auto& cfg = dynamic_cast<const NemotronHConfig&>(config);

    config_json["architectures"] = {std::string(kHfArchitectureName)};
    config_json["model_type"] = "nemotron_h";
    config_json["bos_token_id"] = cfg.BosTokenId;
    config_json["eos_token_id"] = cfg.EosTokenId;
    config_json["pad_token_id"] = cfg.PadTokenId;
    config_json["hidden_size"] = cfg.HiddenSize;
    config_json["intermediate_size"] = cfg.IntermediateSize;
    config_json["vocab_size"] = cfg.VocabSize;
    config_json["num_attention_heads"] = cfg.NumQueryHeads;
    config_json["num_key_value_heads"] = cfg.NumKeyValHeads;
    config_json["num_hidden_layers"] = cfg.NumLayers;
    config_json["head_dim"] = cfg.HeadDim > 0 ? cfg.HeadDim : cfg.head_size();
    config_json["max_position_embeddings"] = cfg.MaxPositionEmbeddings;
    config_json["layer_norm_epsilon"] = cfg.RmsNormEps;
    config_json["attention_bias"] = cfg.AttentionBias;
    config_json["mlp_bias"] = cfg.MlpBias;
    config_json["attention_dropout"] = cfg.AttentionDropout;
    config_json["residual_in_fp32"] = cfg.ResidualInFp32;
    config_json["rope_theta"] = cfg.RopeTheta;

    config_json["mamba_num_heads"] = cfg.MambaNumHeads;
    config_json["mamba_head_dim"] = cfg.MambaHeadDim;
    config_json["ssm_state_size"] = cfg.SsmStateSize;
    config_json["conv_kernel"] = cfg.ConvKernel;
    config_json["n_groups"] = cfg.NGroups;
    config_json["chunk_size"] = cfg.ChunkSize;
    config_json["time_step_min"] = cfg.TimeStepMin;
    config_json["time_step_max"] = cfg.TimeStepMax;
    config_json["time_step_floor"] = cfg.TimeStepFloor;
    config_json["time_step_limit"] = cfg.TimeStepLimit;
    config_json["use_bias"] = cfg.UseBias;
    config_json["use_conv_bias"] = cfg.UseConvBias;
    config_json["mamba_hidden_act"] = cfg.MambaHiddenAct;
    config_json["mlp_hidden_act"] = cfg.MlpHiddenAct;

    // MoE-specific fields (Nemotron-H hybrid MoE)
    config_json["n_routed_experts"] = cfg.NRoutedExperts;
    config_json["num_experts_per_tok"] = cfg.NumExpertsPerTok;
    config_json["moe_intermediate_size"] = cfg.MoeIntermediateSize;
    config_json["moe_shared_expert_intermediate_size"] = cfg.MoeSharedExpertIntermediateSize;
    config_json["n_shared_experts"] = cfg.NSharedExperts;
    config_json["norm_topk_prob"] = cfg.NormTopkProb;
    config_json["routed_scaling_factor"] = cfg.RoutedScalingFactor;
    config_json["topk_group"] = cfg.TopkGroup;
    config_json["n_group"] = cfg.NGroup;
    config_json["router_aux_loss_coef"] = cfg.RouterAuxLossCoef;
    config_json["router_z_loss_coef"] = cfg.RouterZLossCoef;

    if (!cfg.LayersBlockType.empty()) {
        config_json["layers_block_type"] = cfg.LayersBlockType;
    } else if (!cfg.HybridOverridePattern.empty()) {
        config_json["hybrid_override_pattern"] = cfg.HybridOverridePattern;
    }

    config_json["torch_dtype"] = dtype_to_torch_str(cfg.DType);
}

} // namespace models
