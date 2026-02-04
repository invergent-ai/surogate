// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "config/pretrained_config.h"

namespace {

std::filesystem::path write_temp_json(const nlohmann::json& j, const std::string& name) {
    auto dir = std::filesystem::temp_directory_path();
    auto path = dir / name;
    std::ofstream f(path);
    REQUIRE(f.is_open());
    f << j.dump(2);
    return path;
}

} // namespace

TEST_CASE("load_pretrained_config: parses architecture/model_type strings", "[models][config]") {
    nlohmann::json j;
    j["architectures"] = {"Qwen2ForCausalLM"};
    j["model_type"] = "qwen2";
    j["bos_token_id"] = 151643;
    j["eos_token_id"] = 151643;
    j["pad_token_id"] = 151643;
    j["hidden_size"] = 896;
    j["intermediate_size"] = 4864;
    j["vocab_size"] = 151936;
    j["num_attention_heads"] = 14;
    j["num_key_value_heads"] = 2;
    j["num_hidden_layers"] = 24;
    j["max_position_embeddings"] = 32768;
    j["rope_theta"] = 1000000.0;
    j["rms_norm_eps"] = 1e-6;
    j["tie_word_embeddings"] = true;
    j["attention_bias"] = true;

    auto path = write_temp_json(j, "surogate_test_qwen2_config.json");

    auto cfg = load_pretrained_config(path.c_str(), ETensorDType::BF16);
    REQUIRE(cfg != nullptr);
    REQUIRE(cfg->ArchitectureName == "Qwen2ForCausalLM");
    REQUIRE(cfg->ModelTypeName == "qwen2");
    REQUIRE(cfg->HiddenSize == 896);
    REQUIRE(cfg->NumQueryHeads == 14);
    REQUIRE(cfg->NumKeyValHeads == 2);
    REQUIRE(cfg->HeadDim == 0);
    REQUIRE(cfg->UseQKNorm == false);
    REQUIRE(cfg->UseQKVBias == true);
    REQUIRE(cfg->PadTokenId == cfg->BosTokenId);
    REQUIRE(cfg->head_size() == 896 / 14);
}

TEST_CASE("load_pretrained_config: parses head_dim + qk_norm", "[models][config]") {
    nlohmann::json j;
    j["architectures"] = {"Qwen3ForCausalLM"};
    j["model_type"] = "qwen3";
    j["bos_token_id"] = 151643;
    j["eos_token_id"] = 151643;
    j["pad_token_id"] = 151643;
    j["hidden_size"] = 1024;
    j["intermediate_size"] = 3072;
    j["vocab_size"] = 151936;
    j["num_attention_heads"] = 16;
    j["num_key_value_heads"] = 8;
    j["num_hidden_layers"] = 28;
    j["head_dim"] = 128;
    j["max_position_embeddings"] = 32768;
    j["rope_theta"] = 1000000.0;
    j["rms_norm_eps"] = 1e-6;
    j["tie_word_embeddings"] = true;
    j["attention_bias"] = false;
    j["use_qk_norm"] = true;

    auto path = write_temp_json(j, "surogate_test_qwen3_config.json");

    auto cfg = load_pretrained_config(path.c_str(), ETensorDType::BF16);
    REQUIRE(cfg != nullptr);
    REQUIRE(cfg->ArchitectureName == "Qwen3ForCausalLM");
    REQUIRE(cfg->ModelTypeName == "qwen3");
    REQUIRE(cfg->HiddenSize == 1024);
    REQUIRE(cfg->NumQueryHeads == 16);
    REQUIRE(cfg->HeadDim == 128);
    REQUIRE(cfg->head_size() == 128);
    REQUIRE(cfg->attn_out_channels() == 16 * 128);
    REQUIRE(cfg->UseQKNorm == true);
    REQUIRE(cfg->UseQKVBias == false);
    REQUIRE(cfg->PadTokenId == cfg->BosTokenId);
}

TEST_CASE("create_pretrained_config_from_name: loads config.json by path", "[models][config]") {
    nlohmann::json j;
    j["architectures"] = {"TestForCausalLM"};
    j["model_type"] = "test";
    j["bos_token_id"] = 1;
    j["eos_token_id"] = 2;
    j["pad_token_id"] = 3;
    j["hidden_size"] = 128;
    j["intermediate_size"] = 512;
    j["vocab_size"] = 32000;
    j["num_attention_heads"] = 8;
    j["num_key_value_heads"] = 8;
    j["num_hidden_layers"] = 4;
    auto path = write_temp_json(j, "surogate_test_generic_config.json");

    auto cfg = create_pretrained_config_from_name(path.string(), ETensorDType::BF16);
    REQUIRE(cfg != nullptr);
    REQUIRE(cfg->ArchitectureName == "TestForCausalLM");
    REQUIRE(cfg->ModelTypeName == "test");
    REQUIRE(cfg->HiddenSize == 128);
    REQUIRE(cfg->NumLayers == 4);
}
