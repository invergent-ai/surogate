#include "serve/lora_registry.h"
#include <nlohmann/json.hpp>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>

using Json = nlohmann::json;
using sinfer::serve::LoraRegistry;

struct Fixture {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("surogate-lora-registry-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Fixture() {
        std::filesystem::create_directory(path);
    }
    ~Fixture() {
        std::filesystem::remove_all(path);
    }
    void write(const std::string& module, bool wrong_rank = false, bool wrong_bytes = false, bool extra = false) {
        std::ofstream(path / "adapter_config.json") << R"({"peft_type":"LORA","r":1,"lora_alpha":1})";
        Json header = {
            {module + ".lora_A.weight",
             {{"dtype", "F16"}, {"shape", {1, 2}}, {"data_offsets", {0, wrong_bytes ? 6 : 4}}}},
            {module + ".lora_B.weight",
             {{"dtype", "F16"}, {"shape", {1, wrong_rank ? 2 : 1}}, {"data_offsets", {4, wrong_rank ? 8 : 6}}}}};
        if (extra) {
            header[module + ".bias"] = {{"dtype", "F16"}, {"shape", {1}}, {"data_offsets", {6, 8}}};
        }
        const std::string encoded = header.dump();
        const std::uint64_t bytes = encoded.size();
        const std::uint16_t data[]{1, 0x3C00, 0x3C00, 0};
        std::ofstream file(path / "adapter_model.safetensors", std::ios::binary);
        file.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
        file.write(encoded.data(), encoded.size());
        file.write(reinterpret_cast<const char*>(data), sizeof(data));
    }
    void custom(const Json& config, const std::vector<std::pair<std::string, std::vector<int>>>& tensors) {
        std::ofstream(path / "adapter_config.json") << config;
        Json header;
        std::vector<float> values;
        for (const auto& [name, shape] : tensors) {
            std::size_t count = 1;
            for (int dim : shape) { count *= dim; }
            const auto begin = values.size() * sizeof(float);
            values.resize(values.size() + count, 0.5F);
            header[name] = {{"dtype", "F32"}, {"shape", shape}, {"data_offsets", {begin, values.size() * sizeof(float)}}};
        }
        const auto encoded = header.dump();
        const std::uint64_t bytes = encoded.size();
        std::ofstream file(path / "adapter_model.safetensors", std::ios::binary);
        file.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
        file.write(encoded.data(), encoded.size());
        file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    }
    auto read(std::vector<std::string>& skipped) {
        LoraRegistry registry;
        registry.load({{"policy", path.string()}}, 8);
        return LoraRegistry::read_payloads(*registry.find("policy"), skipped);
    }
};

int main() {
    Fixture f;
    std::vector<std::string> skipped;
    f.write("base_model.model.model.layers.2.mlp.experts.7.gate_proj");
    auto payloads = f.read(skipped);
    assert(skipped.empty() && payloads.size() == 1);
    assert(payloads[0].layer == 2 && payloads[0].module == "mlp.experts.7.gate_proj");
    assert(payloads[0].a[0] == 0x3380);  // the smallest F16 subnormal, 2^-24
    f.write("base_model.model.layers.2.self_attn.q_proj");
    payloads = f.read(skipped);
    assert(skipped.empty() && payloads[0].module == "self_attn.q_proj");
    f.write("base_model.model.model.vision_model.encoder.layers.2.self_attn.q_proj");
    payloads = f.read(skipped);
    assert(skipped.empty() && payloads[0].layer == -2 && payloads[0].module == "vision.encoder.layers.2.self_attn.q_proj");
    for (int malformed = 0; malformed < 3; ++malformed) {
        f.write("model.layers.0.mlp.up_proj", malformed == 0, malformed == 1, malformed == 2);
        bool rejected = false;
        try {
            (void)f.read(skipped);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        assert(rejected);
    }
    Json config = {{"r", 2}, {"lora_alpha", 8}, {"use_rslora", true}, {"use_dora", true},
        {"bias", "all"}, {"lora_bias", true}, {"rank_pattern", {{"self_attn.q_proj", 1}}},
        {"alpha_pattern", {{"self_attn.q_proj", 3}}}};
    const std::string text = "base_model.model.model.layers.0.self_attn.q_proj";
    const std::string embed = "base_model.model.model.embed_tokens";
    f.custom(config, {{text + ".lora_A.weight", {1, 4}}, {text + ".lora_B.weight", {4, 1}},
        {text + ".lora_magnitude_vector", {4}}, {text + ".base_layer.bias", {4}}, {text + ".lora_B.bias", {4}},
        {embed + ".lora_embedding_A", {2, 8}}, {embed + ".lora_embedding_B", {4, 2}},
        {embed + ".lora_magnitude_vector", {1, 4}},
        {"base_model.model.model.visual.blocks.0.norm1.bias", {4}},
        {"base_model.model.model.layers.1.linear_attn.dt_bias", {2}}});
    payloads = f.read(skipped);
    assert(skipped.empty() && payloads.size() == 4);
    for (const auto& payload : payloads) {
        if (payload.layer == 0) {
            assert(payload.rank == 1 && payload.scale == 3 && payload.magnitude.size() == 4 &&
                   payload.bias.size() == 4 && payload.lora_bias.size() == 4);
        } else if (payload.layer == -1) {
            assert(payload.module == "embed_tokens" && payload.rank == 2 && payload.in_dim == 8 && payload.out_dim == 4);
            assert(std::abs(payload.scale - 8 / std::sqrt(2.0)) < 1e-6);
        } else if (payload.layer == 1) {
            assert(payload.module == "linear_attn.dt_bias.bias" && payload.bias.size() == 2);
        } else {
            assert(payload.layer == -2 && payload.module == "vision.blocks.0.norm1.bias" && payload.bias.size() == 4);
        }
    }
    config = {{"r", 2}, {"lora_alpha", 2}};
    f.custom(config, {{"model.visual.patch_embed.proj.lora_A.weight", {2, 3, 2, 2, 2}},
                     {"model.visual.patch_embed.proj.lora_B.weight", {8, 2, 1, 1, 1}}});
    payloads = f.read(skipped);
    assert(payloads[0].in_dim == 24 && payloads[0].out_dim == 8 && payloads[0].layer == -2);
    config = {{"r", 2}, {"lora_alpha", 2}, {"use_dora", true}};
    f.custom(config, {{embed + ".lora_embedding_A", {2, 8}}, {embed + ".lora_embedding_B", {4, 2}},
                     {embed + ".lora_magnitude_vector", {4}}, {embed + ".base_layer.weight", {8, 4}}});
    payloads = f.read(skipped);
    assert(skipped.empty() && payloads.size() == 1 && payloads[0].base_weight.size() == 32);
    assert(payloads[0].in_dim == 8 && payloads[0].out_dim == 4 && payloads[0].a.size() == 16);
    config = {{"r", 2}, {"lora_alpha", 2}, {"modules_to_save", {"lm_head"}}};
    for (const std::string module : {"base_model.model.lm_head", "base_model.model.lm_head.modules_to_save.default"}) {
        f.custom(config, {{module + ".weight", {8, 4}}, {module + ".bias", {8}}});
        payloads = f.read(skipped);
        assert(payloads.size() == 1 && payloads[0].module == "lm_head" && payloads[0].base_weight.size() == 32);
        assert(payloads[0].in_dim == 4 && payloads[0].out_dim == 8 && payloads[0].rank == 1);
        assert(payloads[0].a == std::vector<std::uint16_t>(4) && payloads[0].b == std::vector<std::uint16_t>(8));
    }
    for (bool unsupported : {false, true}) {
        config = {{"r", 2}, {"lora_alpha", 2}};
        const auto module = unsupported ? text : embed;
        f.custom(config, {{module + ".lora_A.weight", {2, 8}}, {module + ".lora_B.weight", {4, 2}},
                         {module + ".base_layer.weight", {7, 4}}});
        bool rejected = false;
        try { (void)f.read(skipped); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    // A valid adapter plus one unsupported weight must be rejected with an actionable diagnostic.
    const auto expect_unsupported = [&](const Json& settings, const std::string& tensor,
                                        const std::string& named) {
        f.custom(settings, {{text + ".lora_A.weight", {2, 4}}, {text + ".lora_B.weight", {4, 2}},
                            {tensor, {4, 4}}});
        bool rejected = false;
        try { (void)f.read(skipped); }
        catch (const std::invalid_argument& error) {
            rejected = true;
            const std::string message = error.what();
            assert(message.find("policy") != std::string::npos);
            assert(message.find(named) != std::string::npos);
            assert(message.find("unsupported") != std::string::npos);
            assert(message.find("surogate merge") != std::string::npos);
        }
        assert(rejected);
    };
    const std::string full = "base_model.model.model.layers.0.mlp.down_proj.weight";
    expect_unsupported({{"r", 2}, {"lora_alpha", 2}, {"modules_to_save", {"mlp.down_proj"}}}, full, "mlp.down_proj");
    expect_unsupported({{"r", 2}, {"lora_alpha", 2}}, full, full);
    const std::string unknown = "base_model.model.model.layers.0.mlp.down_proj.adapter_scale";
    expect_unsupported({{"r", 2}, {"lora_alpha", 2}}, unknown, unknown);
    std::cout << "Adapter namespaces, tensor bounds, rank agreement and F16 conversion passed\n";
}
