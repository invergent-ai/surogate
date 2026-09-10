#include "serve/lora_registry.h"
#include <nlohmann/json.hpp>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

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
    assert(f.read(skipped).empty() && skipped.size() == 1);
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
    std::cout << "Adapter namespaces, tensor bounds, rank agreement and F16 conversion passed\n";
}
