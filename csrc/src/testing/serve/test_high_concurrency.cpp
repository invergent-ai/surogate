// Opt-in integration: SUROGATE_CONCURRENCY_TEST_ARTIFACT supplies a checkpoint
// with >=2048 context (or CONTEXT overrides it). Optional OLD/NEW adapter paths
// exercise per-request policies. SPEC selects mtp/dflash, FP8 selects FP8 KV,
// and EAGER disables CUDA graphs.
#include "serve/generation_service.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

using namespace sinfer::serve;
using namespace std::chrono_literals;

int main() {
    const auto* artifact = std::getenv("SUROGATE_CONCURRENCY_TEST_ARTIFACT");
    if (!artifact) {
        return 77;
    }
    const auto* old_adapter = std::getenv("SUROGATE_CONCURRENCY_TEST_OLD");
    const auto* new_adapter = std::getenv("SUROGATE_CONCURRENCY_TEST_NEW");
    constexpr std::uint32_t lanes = 257;
    ServeOptions options;
    options.artifact_path = artifact;
    options.max_context = 2048;
    if (const auto* context = std::getenv("SUROGATE_CONCURRENCY_TEST_CONTEXT")) {
        options.max_context = static_cast<std::uint32_t>(std::stoul(context));
    }
    if (std::getenv("SUROGATE_CONCURRENCY_TEST_FP8")) {
        options.kv_cache = sinfer::KvCacheStorage::Fp8E4M3;
    }
    options.kv_capacity = sinfer::KvCapacityPolicy::explicit_capacity(options.max_context * lanes);
    options.max_concurrency = lanes;
    options.max_pending_requests = 32;
    options.pending_timeout_ms = 120000;
    options.use_cuda_graph = std::getenv("SUROGATE_CONCURRENCY_TEST_EAGER") == nullptr;
    options.enable_lora = old_adapter && new_adapter;
    options.max_loras = 2;
    options.max_lora_rank = 8;
    if (const auto* backend = std::getenv("SUROGATE_CONCURRENCY_TEST_SPEC")) {
        options.speculative.backend =
            std::string_view(backend) == "mtp" ? sinfer::SpeculativeBackend::Mtp : sinfer::SpeculativeBackend::DFlash;
        options.speculative.draft_tokens = 3;
        options.speculative.max_lanes = sinfer::kSpeculateAtAnyWidth;
    }
    GenerationService service(options);
    if (options.enable_lora) {
        service.load_lora_adapter("old", old_adapter);
        service.load_lora_adapter("new", new_adapter);
    }
    GenerationRequest request;
    request.raw_prompt = "Continue: 1, 2, 3, 4,";
    request.max_tokens = options.max_context / 2;
    request.max_tokens_set = true;
    request.ignore_eos = true;
    request.sampling.temperature = 0;
    request.return_token_ids = true;
    // Speculative rounds currently expose token decisions without token logprobs.
    request.want_logprobs = options.speculative.backend == sinfer::SpeculativeBackend::None;
    std::vector<std::vector<sinfer::TokenId>> expected;
    for (int policy = 0; policy < (options.enable_lora ? 2 : 1); ++policy) {
        auto baseline = request;
        baseline.max_tokens = 16;
        if (options.enable_lora) {
            baseline.lora_adapter = policy ? "new" : "old";
        }
        auto prepared = service.prepare(baseline);
        expected.push_back(service.run(prepared, nullptr).completion_token_ids);
        assert(expected.back().size() == baseline.max_tokens);
    }
    service.shrink_kv();
    std::vector<std::unique_ptr<PreparedRequest>> pending;
    const auto make_request = [&](int policy) {
        auto input = request;
        if (options.enable_lora) {
            input.lora_adapter = policy ? "new" : "old";
        }
        return std::make_unique<PreparedRequest>(service.prepare(input));
    };
    for (std::uint32_t lane = 0; lane < lanes; ++lane) {
        pending.push_back(make_request(lane % expected.size()));
    }
    std::uint32_t peak = 0;
    const auto deadline = std::chrono::steady_clock::now() + 90s;
    while (peak < lanes && std::chrono::steady_clock::now() < deadline) {
        peak = std::max(peak, service.runtime_stats().decode_ready_requests);
        std::this_thread::sleep_for(1ms);
    }
    if (peak != lanes) {
        std::cerr << "only " << peak << " of " << lanes << " requests became decode-ready together\n";
    }
    assert(peak == lanes);
    std::cout << peak << " requests decode-ready together; checking completed generations\n" << std::flush;
    // Cancellation and lane recycling must also work above the old ceiling.
    pending[129].reset();
    pending[256].reset();
    pending[129] = make_request(129 % expected.size());
    pending[256] = make_request(256 % expected.size());
    for (std::size_t lane = 0; lane < pending.size(); ++lane) {
        const auto outcome = service.run(*pending[lane], nullptr);
        assert(outcome.completion_tokens == request.max_tokens);
        const auto& prefix = expected[lane % expected.size()];
        assert(outcome.completion_token_ids.size() >= prefix.size());
        assert(std::equal(prefix.begin(), prefix.end(), outcome.completion_token_ids.begin()));
        if (request.want_logprobs) {
            assert(outcome.token_logprobs.size() == outcome.completion_token_ids.size());
            for (float value : outcome.token_logprobs) {
                assert(std::isfinite(value));
            }
        }
        pending[lane].reset();
    }
    const auto stats = service.runtime_stats();
    assert(stats.running_requests == 0 && stats.waiting_requests == 0);
    std::cout << peak
              << " simultaneous decode-ready requests: generation, policy isolation, cancellation and reuse passed\n";
}
