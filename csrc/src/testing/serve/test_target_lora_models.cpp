// Opt-in adapter directory and GPU upload checks against a converted checkpoint.
#include "api/engine.h"
#include "api/ops/lora_store.h"
#include "artifact/reader.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>

using namespace sinfer;

int main() {
    const char* path = std::getenv("SUROGATE_TARGET_LORA_TEST_ARTIFACT");
    if (!path) { return 77; }
    artifact::Reader reader(path);
    EngineOptions options;
    options.artifact_path = path;
    options.max_context = 256;
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(512);
    options.max_concurrency = 2;
    options.prefill_chunk = 128;
    options.use_cuda_graph = false;
    options.lora_enable = true;
    options.lora_slots = 1;
    options.lora_max_rank = 2;
    if (const auto* devices = std::getenv("SUROGATE_TARGET_LORA_TEST_DEVICES")) {
        std::istringstream selected(devices);
        for (std::string item; std::getline(selected, item, ':');) { options.devices.push_back(std::stoi(item)); }
        options.device = options.devices.front();
    }
    Engine engine(options);
    const int layers = reader.geometry().at("layers");
    auto& stores = engine.lora_stores();
    const auto upload = [&](ops::LoraStore& store, int layer, const std::string& module,
                            const std::string& object) {
        const auto* weight = reader.find(object);
        assert(weight && std::holds_alternative<artifact::TensorDescriptor>(*weight));
        const auto& shape = std::get<artifact::TensorDescriptor>(*weight).shape;
        assert(shape.size() == 2);
        const int out = shape[0], in = shape[1];
        std::vector<std::uint16_t> a(in, 0), b(out, 0);
        store.validate_module(layer, module, a, b, 1, in, out, 1);
        store.set_module_slot(layer, module, 0, a, b, 1, in, out, 1);
    };
    if (reader.identity().architecture == "glm5_next") {
        for (int layer = 0; layer < layers; ++layer) {
            ops::LoraStore* owner = nullptr;
            for (int device : stores.devices()) {
                auto* candidate = stores.peek(device);
                if (!candidate || !candidate->covers_layer(layer)) { continue; }
                assert(!owner && "an adapter layer must belong to only one physical stage");
                owner = candidate;
                assert(cudaSetDevice(device) == cudaSuccess);
            }
            assert(owner && "every layer must have a resident adapter owner");
            const auto prefix = "text/layers/" + std::to_string(layer) + "/";
            if (reader.find(prefix + "mlp/down")) { upload(*owner, layer, "down_proj", prefix + "mlp/down"); }
            if (reader.find(prefix + "mla/output")) { upload(*owner, layer, "o_proj", prefix + "mla/output"); }
        }
    } else {
        assert(reader.identity().architecture == "gemma4_e");
        const int first_shared = layers - static_cast<int>(reader.geometry().at("kv_shared_layers"));
        assert(first_shared > 0 && first_shared < layers);
        auto& store = *stores.peek(engine.device());
        assert(cudaSetDevice(engine.device()) == cudaSuccess);
        for (int layer = 0; layer < layers; ++layer) {
            const auto prefix = "text/layers/" + std::to_string(layer) + "/";
            for (const auto* module : {"gate", "up", "down"}) {
                upload(store, layer, std::string(module) + "_proj", prefix + "mlp/" + module);
            }
            upload(store, layer, "q_proj", prefix + "attention/query");
            upload(store, layer, "o_proj", prefix + "attention/output");
            for (const auto* module : {"k_proj", "v_proj"}) {
                if (layer < first_shared) {
                    const auto object = prefix + (std::string_view(module) == "k_proj" ? "attention/key" : "attention/value");
                    if (reader.find(object)) { upload(store, layer, module, object); }
                } else {
                    try {
                        store.validate_module(layer, module, {0}, {0}, 1, 1, 1, 1);
                        assert(false && "shared-KV projection cannot have its own adapter");
                    } catch (const std::invalid_argument& error) {
                        assert(std::string(error.what()).find("earlier layer") != std::string::npos);
                    }
                }
            }
        }
    }
    RequestOptions request;
    request.execution.requested_output_tokens = 4;
    request.execution.sampling.temperature = 0;
    request.execution.top_logprobs = 0;
    const auto base = engine.generate(engine.prepare_tokens({100, 101, 102}), request);
    request.execution.allow_prefix_reuse = false;
    request.execution.lora_slot = 0;
    const auto adapted = engine.generate(engine.prepare_tokens({100, 101, 102}), request);
    assert(!base.generated_token_ids.empty() && adapted.generated_token_ids == base.generated_token_ids);
    assert(base.completion_logprobs.size() == base.generated_token_ids.size());
    assert(adapted.completion_logprobs.size() == base.completion_logprobs.size());
    for (std::size_t i = 0; i < base.completion_logprobs.size(); ++i) {
        const float original = base.completion_logprobs[i].selected.logprob;
        const float selected = adapted.completion_logprobs[i].selected.logprob;
        assert(std::isfinite(original) && std::isfinite(selected));
        assert(std::abs(selected - original) < 0.005F);
    }
    std::cout << "Target adapter shapes, uploads, absent-module refusals and zero-adapter generation passed\n";
}
