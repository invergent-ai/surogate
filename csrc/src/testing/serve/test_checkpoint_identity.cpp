#include "artifact/reader.h"
#include <api/targets/llama/package.h>
#include <api/targets/qwen3/package.h>
#include <api/targets/qwen3_5/package.h>
#include <api/targets/qwen3_5_moe/package.h>
#include <api/targets/qwen4exp/package.h>
#include <api/targets/glm5_next/package.h>
#include "artifact_fixture.h"
#include "encoder/gemma_embedding.h"

#include <cassert>
#include <stdexcept>

template<class Package>
void check_family(std::string weights = "groupwise-int") {
    sinfer::artifact::ArtifactIdentity first{"a-checkpoint", std::move(weights),
                                              std::string(Package::target_key)};
    auto renamed = first;
    renamed.model_id = "entirely-unrelated-label";
    assert(Package::resolve_weights(first) == Package::resolve_weights(renamed));
    renamed.architecture = "unknown-architecture";
    bool refused = false;
    try {
        (void)Package::resolve_weights(renamed);
    } catch (const std::runtime_error&) {
        refused = true;
    }
    assert(refused);
}

void check_encoder() {
    using Json = nlohmann::json;
    const Json directory{
        {"identity", {{"model_id", "renamed-encoder"}, {"weights_id", "w8"},
                       {"architecture", "gemma_embedding"}}},
        {"geometry", {{"hidden", 384}, {"residual", 384}, {"layers", 3}, {"intermediate", 768},
                       {"output_rows", 512}, {"token_domain", 512}, {"query_heads", 4},
                       {"kv_heads", 1}, {"head_dim", 64}, {"rotary_dim", 64},
                       {"max_context", 4096}, {"rms_epsilon", 1e-5}, {"rope_theta", 123456.0},
                       {"attention_scale", 0.125}, {"embedding_scale", 19.595918},
                       {"sliding_window", 128}, {"sliding_rope_theta", 1234.0}}},
        {"layer_types", {"sliding_attention", "full_attention", "sliding_attention"}},
        {"objects", Json::array({{{"name", "fixture"}, {"kind", "resource"},
                                  {"encoding", "raw-bytes-v1"}, {"offset", 0}, {"bytes", 1}}})},
    };
    auto fixture = sinfer::test::artifact_fixture::write_fixture(directory, "checkpoint_encoder");
    const sinfer::artifact::Reader reader(fixture.path);
    const auto config = sinfer::encoder::GemmaEmbeddingConfig::from_artifact(reader);
    assert(config.hidden == 384 && config.layers == 3 && config.head_dim == 64);
    assert(config.query_size() == 256 && config.intermediate == 768);
    assert(config.max_tokens == 4096 && config.rms_epsilon == 1e-5F);
    assert(config.attention_scale == 0.125F && config.rope_theta_local == 1234.0F);
    assert(!config.is_global(0) && config.is_global(1) && !config.is_global(2));
}

void check_glm() {
    using Json = nlohmann::json;
    using Glm = sinfer::targets::glm5_next::Package;
    const Json directory{
        {"identity", {{"model_id", "renamed-checkpoint"}, {"weights_id", "w8-mhc-v1"},
                       {"architecture", "glm5_next"}}},
        {"geometry", {{"hidden", 256}, {"residual", 512}, {"layers", 4}, {"intermediate", 128},
                       {"output_rows", 512}, {"token_domain", 512}, {"query_heads", 4},
                       {"kv_heads", 1}, {"head_dim", 64}, {"rotary_dim", 0}, {"rope_theta", 0},
                       {"max_context", 135}, {"rms_epsilon", 1e-4}, {"attention_scale", 0.125},
                       {"gdn_scale", 0.176776695}, {"hc_streams", 2}, {"hc_epsilon", 2e-5},
                       {"hc_sinkhorn_iterations", 12}, {"q_lora_rank", 64}, {"kv_lora_rank", 64},
                       {"qk_head_dim", 128}, {"v_head_dim", 128}, {"gdn_conv_kernel", 5},
                       {"gdn_key_heads", 4}, {"gdn_value_heads", 4}, {"gdn_key_head_dim", 32},
                       {"gdn_value_head_dim", 32}, {"kda_gate_rank", 32}, {"kda_gate_bound", 3},
                       {"dense_intermediate", 512}, {"leading_dense_layers", 2}, {"experts", 8},
                       {"experts_per_token", 2}, {"shared_intermediate", 128}, {"routed_scale", 1.5},
                       {"swiglu_limit", 7}, {"mtp_layers", 0}}},
        {"layer_types", {"linear_attention", "full_attention", "linear_attention", "full_attention"}},
        {"objects", Json::array({{{"name", "fixture"}, {"kind", "resource"},
                                  {"encoding", "raw-bytes-v1"}, {"offset", 0}, {"bytes", 1}}})},
    };
    auto fixture = sinfer::test::artifact_fixture::write_fixture(directory, "checkpoint_glm");
    const sinfer::artifact::Reader reader(fixture.path);
    const auto geometry = Glm::declared_geometry(reader);
    assert(geometry.qk_head_dim == 128 && geometry.v_head_dim == 128 && geometry.experts == 8);
    assert(geometry.hc_streams == 2 && geometry.hc_sinkhorn_iterations == 12);
    assert(geometry.max_context == 135 && geometry.kda_gate_bound == 3.0F);
    assert(!geometry.layer_attends(0) && geometry.layer_attends(1));
    for (const auto& [name, unused] : directory["geometry"].items()) {
        auto incomplete = directory;
        incomplete["geometry"].erase(name);
        auto missing = sinfer::test::artifact_fixture::write_fixture(incomplete, "checkpoint_glm_missing");
        bool refused = false;
        try {
            const sinfer::artifact::Reader bad(missing.path);
            (void)Glm::declared_geometry(bad);
        } catch (const std::exception&) {
            refused = true;
        }
        assert(refused);
    }
}

template<class Hybrid>
void check_hybrid(bool mixture = false, bool hyper_connected = false) {
    using Json = nlohmann::json;
    Json directory{
        {"identity", {{"model_id", "renamed-hybrid"}, {"weights_id", "groupwise-int"},
                       {"architecture", "qwen3_5"}}},
        {"geometry", {{"hidden", 384}, {"residual", 384}, {"layers", 4}, {"intermediate", 768},
                       {"output_rows", 512}, {"token_domain", 500}, {"query_heads", 4},
                       {"kv_heads", 1}, {"head_dim", 128}, {"rotary_dim", 64}, {"rope_theta", 543210},
                       {"max_context", 8192}, {"rms_epsilon", 2e-5}, {"attention_scale", .08838835},
                       {"gdn_scale", .176776695}, {"gdn_key_heads", 2}, {"gdn_key_head_dim", 32},
                       {"gdn_value_heads", 4}, {"gdn_value_head_dim", 64}, {"gdn_conv_kernel", 4},
                       {"draft_vocab", 500}, {"mtp_layers", 0}}},
        {"layer_types", {"full_attention", "linear_attention", "full_attention", "linear_attention"}},
        {"objects", Json::array({{{"name", "fixture"}, {"kind", "resource"},
                                  {"encoding", "raw-bytes-v1"}, {"offset", 0}, {"bytes", 1}}})},
    };
    directory["identity"]["architecture"] = Hybrid::target_key;
    if (mixture) {
        directory["geometry"].update({{"experts", 8}, {"experts_per_token", 2},
                                      {"shared_intermediate", 256}, {"routed_scale", 1.0}});
        if (!hyper_connected) {
            directory["dflash_geometry"] = {{"hidden", 384}, {"layers", 3}, {"local_layers", 2},
                {"intermediate", 192}, {"query_heads", 4}, {"kv_heads", 2}, {"head_dim", 64},
                {"local_capacity", 128}, {"mask_token", 500}, {"block_size", 8}, {"max_context", 4096},
                {"feature_layers", 2}, {"feature_rows", 768}, {"rms_epsilon", 3e-5},
                {"rope_theta", 234567.}, {"attention_scale", .125}};
            directory["dflash_target_layers"] = {0, 2};
            for (const auto& [name, unused] : directory["dflash_geometry"].items()) {
                auto incomplete = directory;
                incomplete["dflash_geometry"].erase(name);
                auto missing = sinfer::test::artifact_fixture::write_fixture(incomplete, "checkpoint_dflash_missing");
                bool refused = false;
                try { const sinfer::artifact::Reader bad(missing.path); }
                catch (const std::exception&) { refused = true; }
                assert(refused);
            }
        }
    }
    if (hyper_connected) {
        directory["geometry"].update({{"residual", 1152}, {"hc_streams", 3}, {"hc_low_rank", 32},
            {"indexer_heads", 8}, {"indexer_head_dim", 128}, {"indexer_top_k", 64}, {"indexer_block", 4},
            {"ple_ngram", 2}, {"ple_layer", 1}, {"ple_heads_per_ngram", 2}, {"ple_head_dim", 32},
            {"ple_conv_kernel", 3}, {"ple_table_rows", 128}, {"ple_eos_token", 499},
            {"ple_image_token", 498}, {"draft_vocab", 0}});
    }
    directory["objects"].push_back({{"name", "text/layers/2/mlp/down"}, {"kind", "tensor"},
        {"shape", {384, 768}}, {"format", "BF16"}, {"layout", "contiguous-le-v1"},
        {"offset", 4096}, {"bytes", 384 * 768 * 2}});
    auto fixture = sinfer::test::artifact_fixture::write_fixture(directory, "checkpoint_hybrid");
    const sinfer::artifact::Reader reader(fixture.path);
    const auto geometry = Hybrid::declared_geometry(reader);
    assert(geometry.hidden == 384 && geometry.head_dim == 128 && geometry.draft_vocab == (hyper_connected ? 0 : 500));
    assert(geometry.query_size() == 512 && geometry.value_dim() == 256);
    if (!hyper_connected) {
        const auto& stored = geometry.linear_storage.at("text/layers/2/mlp/down");
        assert(stored.rows == 384 && stored.columns == 768 && stored.formats.size() == 1);
    }
    assert(geometry.layer_attends(0) && !geometry.layer_attends(1));
    if (mixture) {
        assert(geometry.experts == 8 && geometry.shared_intermediate == 256);
        if (!hyper_connected) {
            assert(geometry.dflash.layers == 3 && geometry.dflash.feature_rows == 768);
            assert(geometry.dflash.query_size() == 256 && geometry.dflash.kv_size() == 128);
            assert(geometry.dflash.target_layers()[1] == 2 && geometry.dflash.block_size == 8);
        } else {
            assert(geometry.residual == 1152 && geometry.hc_streams == 3);
            assert(geometry.ple_heads() == 2 && geometry.ple_embed() == 64);
            assert(geometry.indexer_heads == 8 && geometry.indexer_top_k == 64);
        }
    }
    for (const auto& [name, unused] : directory["geometry"].items()) {
        auto incomplete = directory;
        incomplete["geometry"].erase(name);
        auto missing = sinfer::test::artifact_fixture::write_fixture(incomplete, "checkpoint_hybrid_missing");
        bool refused = false;
        try {
            const sinfer::artifact::Reader bad(missing.path);
            (void)Hybrid::declared_geometry(bad);
        } catch (const std::exception&) {
            refused = true;
        }
        assert(refused);
    }
}

int main() {
    check_encoder();
    check_glm();
    check_hybrid<sinfer::targets::qwen3_5::Package>();
    check_hybrid<sinfer::targets::qwen3_5_moe::Package>(true);
    check_hybrid<sinfer::targets::qwen4exp::Package>(true, true);
    check_family<sinfer::targets::qwen3::Package>();
    check_family<sinfer::targets::llama::Package>();
    check_family<sinfer::targets::qwen3_5::Package>();
    check_family<sinfer::targets::qwen3_5_moe::Package>();
    check_family<sinfer::targets::qwen4exp::Package>("w8-hc-v1");
    using Hybrid = sinfer::targets::qwen3_5::Package;
    const auto mixed = Hybrid::resolve_weights({"custom", "nvfp4-mixed-bf16", "qwen3_5"});
    const auto mlp = Hybrid::resolve_weights({"custom", "nvfp4-mlp-only", "qwen3_5"});
    assert(mixed == Hybrid::WeightsProfile::Nvfp4MixedBf16);
    assert(mlp == Hybrid::WeightsProfile::Nvfp4MlpOnly);
    bool refused = false;
    try {
        (void)Hybrid::resolve_weights({"qwen3.8-27b", "nvfp4", "qwen3_5"});
    } catch (const std::runtime_error&) {
        refused = true;
    }
    assert(refused);
}
