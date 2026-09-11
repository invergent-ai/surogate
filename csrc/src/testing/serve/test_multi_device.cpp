// Opt-in: SUROGATE_MULTI_DEVICE_TEST_ARTIFACT. DEVICES is a colon-separated list;
// the default 0:0 exercises serial pipeline stages on one physical test GPU.
#include "serve/generation_service.h"

#include <cassert>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <set>
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
    const auto* groups = std::getenv("SUROGATE_MULTI_DEVICE_TEST_GROUPS");
    setenv("SUROGATE_SERVE_PIPELINE_GROUPS", groups ? groups : "1", 1);
    if (groups) {
        setenv("SUROGATE_SERVE_PIPELINE_MIN_LANES", "1", 1);
    }
    ServeOptions options;
    options.artifact_path = artifact;
    options.max_context = 512;
    if (const auto* context = std::getenv("SUROGATE_MULTI_DEVICE_TEST_CONTEXT")) {
        options.max_context = std::stoul(context);
    }
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
    if (std::getenv("SUROGATE_MULTI_DEVICE_TEST_DFLASH")) {
        options.speculative.backend = sinfer::SpeculativeBackend::DFlash;
        options.speculative.draft_tokens = 3;
        options.kv_cache = sinfer::KvCacheStorage::BFloat16;
    }
    options.speculative.adaptive = std::getenv("SUROGATE_MULTI_DEVICE_TEST_ADAPTIVE") != nullptr;
    if (const auto* count = std::getenv("SUROGATE_MULTI_DEVICE_TEST_DRAFT_TOKENS")) {
        options.speculative.draft_tokens = std::stoul(count);
    }
    if (const auto* dtype = std::getenv("SUROGATE_MULTI_DEVICE_TEST_KV_DTYPE")) {
        assert(std::string(dtype) == "bf16" || std::string(dtype) == "fp8");
        options.kv_cache =
            std::string(dtype) == "fp8" ? sinfer::KvCacheStorage::Fp8E4M3 : sinfer::KvCacheStorage::BFloat16;
    }
    const bool vision = std::getenv("SUROGATE_MULTI_DEVICE_TEST_VISION") != nullptr;
    options.enable_vision = vision;
    const bool cache_turn = std::getenv("SUROGATE_MULTI_DEVICE_TEST_CACHE") != nullptr;
    if (std::getenv("SUROGATE_MULTI_DEVICE_TEST_LONG_PROMPT")) {
        options.prefill_chunk = 128;
    }
    std::istringstream selected(
        std::getenv("SUROGATE_MULTI_DEVICE_TEST_DEVICES") ? std::getenv("SUROGATE_MULTI_DEVICE_TEST_DEVICES") : "0:0");
    std::vector<int> devices;
    for (std::string item; std::getline(selected, item, ':');) {
        devices.push_back(std::stoi(item));
    }
    assert(devices.size() >= 2);
    const bool shared_device = std::set<int>(devices.begin(), devices.end()).size() != devices.size();
    setenv("SUROGATE_SERVE_PIPELINE_SERIAL_CONSTRUCT", shared_device ? "1" : "0", 0);
    std::cout << "pipeline devices:";
    for (int device : devices) {
        std::cout << ' ' << device;
    }
    std::cout << "; serial construction=" << std::getenv("SUROGATE_SERVE_PIPELINE_SERIAL_CONSTRUCT") << std::endl;
    options.device = devices.front();
    GenerationRequest request;
    if (options.enable_lora) {
        request.lora_adapter = "test-adapter";
    }
    request.raw_prompt = "Continue: 1, 2, 3, 4,";
    if (std::getenv("SUROGATE_MULTI_DEVICE_TEST_LONG_PROMPT")) {
        for (int i = 0; i < 200; ++i) {
            *request.raw_prompt += " sample";
        }
        *request.raw_prompt += "\nContinue: 1, 2, 3, 4,";
    }
    request.max_tokens = vision ? 16 : 128;
    if (vision) {
        if (const auto* tokens = std::getenv("SUROGATE_MULTI_DEVICE_TEST_VISION_TOKENS")) {
            request.max_tokens = std::stoul(tokens);
        }
    }
    if (vision) {
        request.raw_prompt.reset();
        request.enable_thinking = false;
        ContentPart image;
        image.kind = ContentKind::Image;
        image.source.kind = sinfer::product::media_acquire::SourceKind::Bytes;
        image.source.media_type = "image/x-portable-pixmap";
        const std::string header = "P6\n512 512\n255\n";
        image.source.bytes.assign(header.begin(), header.end());
        for (int i = 0; i < 512 * 512; ++i) {
            image.source.bytes.insert(image.source.bytes.end(), {255, 0, 0});
        }
        request.messages.push_back({.role = sinfer::ChatRole::User,
                                    .content = {std::move(image),
                                                {.kind = ContentKind::Text,
                                                 .text = "What is the background color? Reply with one color only."}}});
    }
    request.max_tokens_set = true;
    // The image fixture asks for a color. Compare completed answers, rather than
    // forcing arbitrary special-token continuations after the model has stopped.
    request.ignore_eos = !vision;
    request.sampling.temperature = 0;
    request.return_token_ids = true;
    const bool score_tokens = std::getenv("SUROGATE_MULTI_DEVICE_TEST_LOGPROBS") != nullptr;
    if (score_tokens) {
        request.want_logprobs = true;
        request.top_logprobs = 5;
        request.prompt_logprobs = cache_turn ? -1 : 0;
    }
    std::vector<float> expected_scores;

    std::string completed_text;
    const auto generate = [&](GenerationService& service) {
        auto prepared = service.prepare(request);
        auto result = service.run(prepared, nullptr);
        if (options.speculative.backend == sinfer::SpeculativeBackend::DFlash) {
            std::cout << "DFlash rounds=" << result.metrics.speculative_rounds
                      << " accepted=" << result.metrics.speculative_accepted_tokens << '\n';
            assert(result.metrics.speculative_rounds + result.metrics.speculative_fallback_steps > 0);
        }
        if (score_tokens) {
            assert(result.token_logprobs.size() == result.completion_token_ids.size());
            assert(result.completion_scores.size() == result.completion_token_ids.size());
            for (const auto& score : result.completion_scores) {
                assert(score.top.size() == 5);
                assert(std::isfinite(score.selected.logprob));
            }
            if (request.prompt_logprobs >= 0) {
                assert(result.prompt_scores.size() == result.prompt_tokens);
                assert(result.prompt_scores.front().selected.token_id == -1);
                for (std::size_t i = 1; i < result.prompt_scores.size(); ++i) {
                    assert(std::isfinite(result.prompt_scores[i].selected.logprob));
                    assert(result.prompt_scores[i].selected.token_id == result.prompt_token_ids[i]);
                }
            }
            if (expected_scores.empty()) expected_scores = result.token_logprobs;
            else {
                assert(expected_scores.size() == result.token_logprobs.size());
                for (std::size_t i = 0; i < expected_scores.size(); ++i) {
                    assert(std::abs(expected_scores[i] - result.token_logprobs[i]) < 0.02f);
                }
            }
        }
        completed_text = result.text;
        if (vision) {
            assert(result.text.find("red") != std::string::npos || result.text.find("Red") != std::string::npos);
        }
        return result.completion_token_ids;
    };
    GenerationRequest continued = request;
    std::vector<sinfer::TokenId> expected, expected_turn;
    const auto continue_turn = [&](GenerationService& service) {
        auto prepared = service.prepare(continued);
        auto result = service.run(prepared, nullptr);
        std::cout << "continued turn cached tokens=" << result.metrics.prefix_cache_hit_tokens << '\n';
        assert(result.metrics.prefix_cache_hit_tokens > 0);
        assert(!result.completion_token_ids.empty() && result.completion_token_ids.size() <= continued.max_tokens);
        return result.completion_token_ids;
    };
    {
        GenerationService baseline(options);
        expected = generate(baseline);
        assert(!expected.empty() && expected.size() <= request.max_tokens);
        if (!vision) { assert(expected.size() == 128); }
        if (cache_turn) {
            if (vision) {
                continued.messages.push_back({.role = sinfer::ChatRole::Assistant,
                                              .content = {{.kind = ContentKind::Text, .text = completed_text}}});
                continued.messages.push_back(
                    {.role = sinfer::ChatRole::User, .content = {{.kind = ContentKind::Text,
                        .text = "Name the color again. Reply with one color only."}}});
            } else {
                continued.raw_prompt = *request.raw_prompt + completed_text + "\nContinue.";
            }
            expected_turn = continue_turn(baseline);
        }
    }
    if (std::getenv("SUROGATE_MULTI_DEVICE_TEST_SINGLE_ONLY")) {
        return 0;
    }
    options.devices = devices;
    GenerationService pipeline(options);
    assert(generate(pipeline) == expected);
    if (cache_turn) {
        assert(continue_turn(pipeline) == expected_turn);
    }
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

    if (vision) {
        // Submit just one cold image so sleep can catch its encoder, before
        // preparing a batch gives the worker time to finish that image.
        pipeline.shrink_kv();
        auto partial = pipeline.prepare(request);
        const auto deadline = std::chrono::steady_clock::now() + 60s;
        while (pipeline.runtime_stats().prefilling_requests == 0) {
            assert(std::chrono::steady_clock::now() < deadline);
            std::this_thread::sleep_for(100us);
        }
        std::cerr << "partial image: requesting preemptive sleep\n";
        pipeline.sleep(true);
        assert(pipeline.is_sleeping());
        assert(pipeline.runtime_stats().prefilling_requests == 1);
        pipeline.wake_up();
        assert(pipeline.run(partial, nullptr).completion_token_ids == expected);
        std::cerr << "partial image: resumed with matching output\n";
    }

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
        assert(result.completion_tokens == expected.size());
        if (result.completion_token_ids != batched[lane]) {
            std::cerr << "preempted lane " << lane << " differs from its uninterrupted batch\n";
            for (std::size_t i = 0; i < result.completion_token_ids.size(); ++i) {
                if (result.completion_token_ids[i] != batched[lane][i]) {
                    std::cerr << "first difference at " << i << ": " << result.completion_token_ids[i]
                              << " vs " << batched[lane][i] << '\n';
                    break;
                }
            }
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
    if (std::getenv("SUROGATE_MULTI_DEVICE_TEST_MIXED")) {
        pipeline.shrink_kv();
        GenerationRequest active;
        active.raw_prompt = "Continue: 1, 2, 3, 4,";
        active.max_tokens = 192;
        active.max_tokens_set = true;
        active.ignore_eos = true;
        active.sampling.temperature = 0;
        active.want_logprobs = true;
        active.return_token_ids = true;
        active.lora_adapter = request.lora_adapter;
        auto running = pipeline.prepare(active);
        const auto deadline = std::chrono::steady_clock::now() + 60s;
        while (pipeline.runtime_stats().decode_ready_requests == 0) {
            assert(std::chrono::steady_clock::now() < deadline);
            std::this_thread::sleep_for(1ms);
        }
        std::vector<GenerationRequest> inputs{active};
        std::vector<std::unique_ptr<PreparedRequest>> incoming;
        for (int i = 0; i < 3; ++i) {
            auto prompt = active;
            prompt.raw_prompt.reset();
            prompt.max_tokens = 24;
            for (int j = 0; j < 90; ++j) {
                for (int token : {100 + i, 110 + i, 120 + i}) { prompt.prompt_token_ids.push_back(token); }
            }
            inputs.push_back(prompt);
            incoming.push_back(std::make_unique<PreparedRequest>(pipeline.prepare(prompt)));
        }
        std::vector<GenerationOutcome> outcomes;
        outcomes.push_back(pipeline.run(running, nullptr));
        for (auto& item : incoming) { outcomes.push_back(pipeline.run(*item, nullptr)); }
        for (std::size_t i = 0; i < outcomes.size(); ++i) {
            const auto& actual = outcomes[i];
            auto replay = inputs[i];
            replay.raw_prompt.reset();
            replay.prompt_token_ids = actual.prompt_token_ids;
            replay.prompt_token_ids.insert(replay.prompt_token_ids.end(), actual.completion_token_ids.begin(), actual.completion_token_ids.end());
            replay.max_tokens = 1;
            replay.prompt_logprobs = 0;
            pipeline.shrink_kv();
            auto prepared = pipeline.prepare(replay);
            auto reference = pipeline.run(prepared, nullptr);
            assert(actual.token_logprobs.size() == actual.completion_token_ids.size());
            float peak = 0, sum = 0;
            for (std::size_t j = 0; j < actual.token_logprobs.size(); ++j) {
                const auto& expected = reference.prompt_scores.at(actual.prompt_token_ids.size() + j).selected;
                assert(expected.token_id == actual.completion_token_ids[j]);
                const float error = std::abs(expected.logprob - actual.token_logprobs[j]);
                peak = std::max(peak, error);
                sum += error;
            }
            std::cout << "mixed pipeline score error: max=" << peak << " mean=" << sum / actual.token_logprobs.size() << '\n';
            assert(peak <= (options.speculative.adaptive ? .25F : .12F));
            if (options.speculative.adaptive) { assert(sum / actual.token_logprobs.size() <= .035F); }
        }
    }
    std::cout << "pipeline parity, per-device footprints, sleep isolation and active resume passed\n";
}
