// Opt-in real-model regression. Supply a prepared artifact and two distinct
// compatible adapters through SUROGATE_LORA_TEST_{ARTIFACT,OLD,NEW}; BAD names
// an adapter with an incompatible module. Reserve the device with CUDA_VISIBLE_DEVICES.
#include "serve/generation_service.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <future>
#include <iostream>
#include <thread>

using namespace std::chrono_literals;
using sinfer::serve::GenerationOutcome;
using sinfer::serve::GenerationRequest;
using sinfer::serve::GenerationService;

static GenerationOutcome generate(GenerationService& service, const GenerationRequest& request) {
    auto prepared = service.prepare(request);
    return service.run(prepared, nullptr);
}

static void same_policy(const GenerationOutcome& actual, const GenerationOutcome& expected) {
    assert(actual.completion_token_ids == expected.completion_token_ids);
    assert(actual.token_logprobs.size() == expected.token_logprobs.size());
    for (std::size_t i = 0; i < actual.token_logprobs.size(); ++i) {
        assert(std::isfinite(actual.token_logprobs[i]));
        assert(std::abs(actual.token_logprobs[i] - expected.token_logprobs[i]) < 0.005F);
    }
}

static void await_decode(GenerationService& service) {
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    while (service.runtime_stats().decode_ready_requests == 0) {
        assert(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(1ms);
    }
}

int main() {
    const auto artifact = std::getenv("SUROGATE_LORA_TEST_ARTIFACT");
    const auto old_adapter = std::getenv("SUROGATE_LORA_TEST_OLD");
    const auto new_adapter = std::getenv("SUROGATE_LORA_TEST_NEW");
    const auto bad_adapter = std::getenv("SUROGATE_LORA_TEST_BAD");
    if (!artifact || !old_adapter || !new_adapter || !bad_adapter) { return 77; }

    sinfer::serve::ServeOptions options;
    options.artifact_path = artifact;
    options.max_context = 2048;
    options.kv_capacity = sinfer::KvCapacityPolicy::explicit_capacity(4096);
    options.max_concurrency = 2;
    options.enable_lora = true;
    options.max_loras = 1; // Updates must also work with no spare adapter slot.
    options.max_lora_rank = 8;
    options.use_cuda_graph = std::getenv("SUROGATE_LORA_TEST_EAGER") == nullptr;
    GenerationService service(options);
    service.load_lora_adapter("policy", old_adapter);

    GenerationRequest request;
    request.lora_adapter = "policy";
    request.raw_prompt = "Continue the sequence of integers, separated by commas:\n1, 2, 3, 4,";
    request.max_tokens = 512;
    request.max_tokens_set = true;
    request.ignore_eos = true;
    request.sampling.temperature = 0;
    request.want_logprobs = true;
    request.return_token_ids = true;

    auto warmup = request;
    warmup.max_tokens = 4;
    (void)generate(service, warmup);
    service.shrink_kv();
    const auto original = generate(service, request);
    assert(original.completion_tokens == request.max_tokens);
    {
        // Reuse the complete evaluated prefix: the cached hidden state must still
        // select the adapter's head when generating its last sampled token again.
        auto cached = request;
        cached.raw_prompt.reset();
        cached.prompt_token_ids = original.prompt_token_ids;
        cached.prompt_token_ids.insert(cached.prompt_token_ids.end(),
            original.completion_token_ids.begin(), original.completion_token_ids.end() - 1);
        cached.max_tokens = 1;
        const auto replay = generate(service, cached);
        assert(replay.metrics.prefix_cache_hit_tokens == static_cast<int>(cached.prompt_token_ids.size()));
        assert(replay.completion_token_ids == std::vector<sinfer::TokenId>{original.completion_token_ids.back()});
        assert(std::abs(replay.token_logprobs[0] - original.token_logprobs.back()) < .005F);
    }
    service.shrink_kv();
    {
        auto running = service.prepare(request);
        await_decode(service);
        auto replace = std::async(std::launch::async, [&] { service.load_lora_adapter("policy", new_adapter); });
        assert(replace.wait_for(10ms) == std::future_status::timeout);
        same_policy(service.run(running, nullptr), original);
        replace.get();
    }
    const auto replacement = generate(service, request);
    assert(replacement.metrics.prefix_cache_hit_tokens == 0);
    // The test adapters must exercise a real policy change.
    bool differs = replacement.completion_token_ids != original.completion_token_ids;
    for (std::size_t i = 0; !differs && i < replacement.token_logprobs.size(); ++i) {
        differs = std::abs(replacement.token_logprobs[i] - original.token_logprobs[i]) > 0.01F;
    }
    assert(differs);
    try {
        service.load_lora_adapter("policy", bad_adapter);
        assert(false && "incompatible replacement was accepted");
    } catch (const std::invalid_argument&) {}
    service.shrink_kv();
    same_policy(generate(service, request), replacement);

    service.shrink_kv();
    {
        auto running = service.prepare(request);
        await_decode(service);
        auto unload = std::async(std::launch::async, [&] { service.unload_lora_adapter("policy"); });
        assert(unload.wait_for(10ms) == std::future_status::timeout);
        same_policy(service.run(running, nullptr), replacement);
        unload.get();
    }
    assert(service.lora_slot("policy") == -1);
    service.load_lora_adapter("recycled", old_adapter);
    request.lora_adapter = "recycled";
    const auto recycled = generate(service, request);
    assert(recycled.metrics.prefix_cache_hit_tokens == 0);
    same_policy(recycled, original);
    {
        auto abandoned = service.prepare(request);
        await_decode(service);
        // Destruction cancels asynchronously. The worker must retain the slot
        // until it consumes the cancellation, not just until this scope ends.
    }
    service.unload_lora_adapter("recycled");
    service.load_lora_adapter("after-cancel", new_adapter);
    request.lora_adapter = "after-cancel";
    same_policy(generate(service, request), replacement);
    std::cout << "GPU adapter replacement, unload, reuse, validation and cancellation passed\n";
}
