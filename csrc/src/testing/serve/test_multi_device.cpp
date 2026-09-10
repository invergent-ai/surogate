// Opt-in: SUROGATE_MULTI_DEVICE_TEST_ARTIFACT. DEVICES is a colon-separated list;
// the default 0:0 exercises serial pipeline stages on one physical test GPU.
#include "serve/generation_service.h"

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <vector>

using namespace sinfer::serve;
using namespace std::chrono_literals;

int main() {
    const auto* artifact = std::getenv("SUROGATE_MULTI_DEVICE_TEST_ARTIFACT");
    if (!artifact) {
        return 77;
    }
    setenv("SUROGATE_SERVE_PIPELINE_SERIAL_CONSTRUCT", "1", 1);
    setenv("SUROGATE_SERVE_PIPELINE_GROUPS", "1", 1);
    ServeOptions options;
    options.artifact_path = artifact;
    options.max_context = 512;
    options.kv_capacity = sinfer::KvCapacityPolicy::explicit_capacity(2048);
    options.max_concurrency = 4;
    options.enable_sleep_mode = true;
    if (const auto* adapter = std::getenv("SUROGATE_MULTI_DEVICE_TEST_ADAPTER")) {
        options.enable_lora = true;
        options.lora_modules.push_back({"test-adapter", adapter});
    }
    options.use_cuda_graph = std::getenv("SUROGATE_MULTI_DEVICE_TEST_EAGER") == nullptr;
    if (std::getenv("SUROGATE_MULTI_DEVICE_TEST_MTP")) {
        options.speculative.backend = sinfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
    }
    std::istringstream selected(
        std::getenv("SUROGATE_MULTI_DEVICE_TEST_DEVICES") ? std::getenv("SUROGATE_MULTI_DEVICE_TEST_DEVICES") : "0:0");
    std::vector<int> devices;
    for (std::string item; std::getline(selected, item, ':');) {
        devices.push_back(std::stoi(item));
    }
    assert(devices.size() >= 2);
    options.device = devices.front();
    GenerationRequest request;
    if (options.enable_lora) {
        request.lora_adapter = "test-adapter";
    }
    request.raw_prompt = "Continue: 1, 2, 3, 4,";
    request.max_tokens = 128;
    request.max_tokens_set = true;
    request.ignore_eos = true;
    request.sampling.temperature = 0;
    request.return_token_ids = true;
    const auto generate = [&](GenerationService& service) {
        auto prepared = service.prepare(request);
        return service.run(prepared, nullptr).completion_token_ids;
    };
    std::vector<sinfer::TokenId> expected;
    {
        GenerationService baseline(options);
        expected = generate(baseline);
        assert(expected.size() == 128);
    }
    options.devices = devices;
    GenerationService pipeline(options);
    assert(generate(pipeline) == expected);
    const auto footprint = pipeline.resident_bytes();
    std::size_t sum = 0;
    for (int device : pipeline.devices()) {
        sum += pipeline.resident_bytes(device);
    }
    assert(footprint > 0 && footprint == sum);
    pipeline.sleep();
    assert(pipeline.is_sleeping());
    assert(pipeline.resident_bytes() == footprint);  // parked KV still costs memory on wake
    pipeline.wake_up();
    assert(generate(pipeline) == expected);

    // An independent replica on the same test device must retain its own graphs and weights
    // when the pipeline sleeps. Real multi-GPU runs use the DEVICES override above.
    auto replica_options = options;
    replica_options.devices.clear();
    GenerationService replica(replica_options);
    pipeline.sleep();
    assert(generate(replica) == expected);
    pipeline.wake_up();
    replica.sleep();
    assert(generate(pipeline) == expected);
    replica.wake_up();

    pipeline.shrink_kv();
    std::vector<std::unique_ptr<PreparedRequest>> pending;
    for (int i = 0; i < 8; ++i) {
        pending.push_back(std::make_unique<PreparedRequest>(pipeline.prepare(request)));
    }
    // Compare preemption with the same concurrent workload: quantized kernels can use
    // different arithmetic for one request and a batch, especially on short-conv models.
    std::vector<std::vector<sinfer::TokenId>> batched;
    for (auto& prepared : pending) {
        batched.push_back(pipeline.run(*prepared, nullptr).completion_token_ids);
        assert(batched.back().size() == expected.size());
    }
    pending.clear();
    pipeline.shrink_kv();
    for (int i = 0; i < 8; ++i) {
        pending.push_back(std::make_unique<PreparedRequest>(pipeline.prepare(request)));
    }
    std::this_thread::sleep_for(5ms);
    pipeline.sleep(true);
    assert(pipeline.is_sleeping());
    assert(generate(replica) == expected);
    pipeline.wake_up();
    std::size_t lane = 0;
    for (auto& prepared : pending) {
        const auto result = pipeline.run(*prepared, nullptr);
        assert(result.completion_tokens == request.max_tokens);
        if (result.completion_token_ids != batched[lane]) {
            std::cerr << "preempted lane " << lane << " differs from its uninterrupted batch\n";
        }
        assert(result.completion_token_ids == batched[lane++]);
    }
    pending.clear();
    // Graceful sleep closes admission while existing requests finish; it must not park
    // their worker before draining them.
    std::vector<std::future<std::vector<sinfer::TokenId>>> draining;
    for (int i = 0; i < 8; ++i) {
        pending.push_back(std::make_unique<PreparedRequest>(pipeline.prepare(request)));
    }
    for (auto& prepared : pending) {
        draining.push_back(std::async(std::launch::async, [&pipeline, ready = std::move(prepared)]() mutable {
            auto result = pipeline.run(*ready, nullptr).completion_token_ids;
            ready.reset();
            return result;
        }));
    }
    pipeline.sleep();
    lane = 0;
    for (auto& result : draining) {
        assert(result.get() == batched[lane++]);
    }
    pipeline.wake_up();
    std::cout << "pipeline parity, per-device footprints, sleep isolation and active resume passed\n";
}
