// Opt-in: SUROGATE_BURST_CACHE_TEST_ARTIFACT. REWRITE additionally checks a
// conversation continuation (e.g. Qwen3.5); FP8 selects FP8 KV.
#include "serve/generation_service.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>

using namespace sinfer::serve;

int main() {
    const auto* artifact = std::getenv("SUROGATE_BURST_CACHE_TEST_ARTIFACT");
    if (!artifact) { return 77; }
    ServeOptions options;
    options.artifact_path = artifact;
    options.max_context = 512;
    options.kv_capacity = sinfer::KvCapacityPolicy::explicit_capacity(2048);
    options.max_concurrency = 4;
    options.max_pending_requests = 64;
    options.enable_sleep_mode = true;
    options.kv_cache = std::getenv("SUROGATE_BURST_CACHE_TEST_FP8")
        ? sinfer::KvCacheStorage::Fp8E4M3 : sinfer::KvCacheStorage::BFloat16;
    GenerationService cached(options);
    options.allow_prefix_reuse = false;
    GenerationService cold(options);
    const auto cached_sequence_bytes = cached.memory_summary().sequence.capacity_bytes;
    const auto cold_sequence_bytes = cold.memory_summary().sequence.capacity_bytes;
    const auto run = [](GenerationService& service, const GenerationRequest& request) {
        auto prepared = service.prepare(request);
        return service.run(prepared, nullptr);
    };
    GenerationRequest initial;
    initial.messages = {{.role = sinfer::ChatRole::User, .content = {
        {.kind = ContentKind::Text, .text = "Reply with exactly these three words: red blue green"}}}};
    initial.enable_thinking = false;
    initial.sampling.temperature = 0;
    initial.max_tokens_set = true;
    initial.max_tokens = 3;
    initial.ignore_eos = true;
    initial.return_token_ids = true;
    initial.want_logprobs = true;
    initial.top_logprobs = 5;
    initial.prompt_logprobs = 5;
    auto seed = run(cold, initial);
    assert(cold.memory_summary().sequence.capacity_bytes == cold_sequence_bytes);
    assert(seed.completion_token_ids.size() == 3 && seed.token_texts.size() == 3);
    assert(!seed.token_texts.front().empty());
    initial.max_tokens = 16;
    initial.stop_strings = {seed.token_texts[0] + seed.token_texts[1] + seed.token_texts[2]};

    for (int scenario = 0; scenario < 5; ++scenario) {
        if (scenario == 3 && !std::getenv("SUROGATE_BURST_CACHE_TEST_REWRITE")) { continue; }
        cached.shrink_kv();
        assert(cached.memory_summary().sequence.capacity_bytes == cached_sequence_bytes);
        auto stopped = run(cached, initial);
        assert(stopped.finish_reason == sinfer::FinishReason::StopString);
        assert(stopped.completion_token_ids.size() == 3);
        if (std::getenv("SUROGATE_BURST_CACHE_TEST_REWRITE")) {
            assert(cached.memory_summary().sequence.capacity_bytes > cached_sequence_bytes);
        }
        // The first token comes from prefill. The stop accepts just two of
        // the next burst's eight tokens, so its tail/state must not be reused.
        auto followup = initial;
        followup.stop_strings.clear();
        followup.max_tokens = 4;
        if (scenario == 1 || scenario == 2) {
            followup.prompt_token_ids = stopped.prompt_token_ids;
            followup.prompt_token_ids.insert(followup.prompt_token_ids.end(),
                stopped.completion_token_ids.begin(), stopped.completion_token_ids.end());
        }
        if (scenario == 2) {
            cached.sleep();
            cached.wake_up();
        }
        if (scenario == 3) {
            followup.messages.push_back({.role = sinfer::ChatRole::Assistant,
                .content = {{.kind = ContentKind::Text, .text = initial.stop_strings.front()}}});
            followup.messages.push_back({.role = sinfer::ChatRole::User,
                .content = {{.kind = ContentKind::Text, .text = "Continue briefly."}}});
        }
        if (scenario == 4) { followup.prompt_logprobs = 20; }
        auto actual = run(cached, followup);
        auto expected = run(cold, followup);
        assert(cold.memory_summary().sequence.capacity_bytes == cold_sequence_bytes);
        std::cerr << "scenario " << scenario << ": reused " << actual.metrics.prefix_cache_hit_tokens << '\n';
        assert(actual.completion_token_ids == expected.completion_token_ids);
        assert(actual.prompt_token_ids == expected.prompt_token_ids);
        assert(actual.token_logprobs.size() == expected.token_logprobs.size());
        assert(actual.prompt_scores.size() == expected.prompt_scores.size());
        for (std::size_t i = 0; i < actual.token_logprobs.size(); ++i) {
            if (std::abs(actual.token_logprobs[i] - expected.token_logprobs[i]) > .12F) {
                std::cerr << "completion " << i << ": " << actual.token_logprobs[i] << " vs " << expected.token_logprobs[i] << '\n';
            }
            assert(std::abs(actual.token_logprobs[i] - expected.token_logprobs[i]) <= .12F);
        }
        for (std::size_t i = 1; i < actual.prompt_scores.size(); ++i) {
            if (std::abs(actual.prompt_scores[i].selected.logprob - expected.prompt_scores[i].selected.logprob) > .12F) {
                std::cerr << "prompt " << i << ": " << actual.prompt_scores[i].selected.logprob << " vs " << expected.prompt_scores[i].selected.logprob << '\n';
            }
            assert(actual.prompt_scores[i].selected.token_id == expected.prompt_scores[i].selected.token_id);
            assert(std::abs(actual.prompt_scores[i].selected.logprob -
                            expected.prompt_scores[i].selected.logprob) <= .12F);
        }
        if (scenario == 4) { assert(actual.metrics.prefix_cache_hit_tokens == 0); }
        else { assert(actual.metrics.prefix_cache_hit_tokens > 0); }
    }
    if (std::getenv("SUROGATE_BURST_CACHE_TEST_REWRITE")) {
        // More queued conversations than live lanes or snapshot slots: admission
        // plans must be refreshed when a different request evicts their snapshot.
        std::vector<GenerationRequest> prompts{initial, initial};
        prompts[1].messages[0].content[0].text = "Please output exactly these three words: blue green red";
        std::vector<GenerationOutcome> references;
        for (auto& prompt : prompts) {
            prompt.stop_strings.clear();
            prompt.max_tokens = 3;
            references.push_back(run(cold, prompt));
            const auto& texts = references.back().token_texts;
            assert(texts.size() == 3);
            prompt.stop_strings = {texts[0] + texts[1] + texts[2]};
            prompt.max_tokens = 16;
        }
        cached.shrink_kv();
        std::vector<std::unique_ptr<PreparedRequest>> pending;
        for (int i = 0; i < 32; ++i) {
            pending.push_back(std::make_unique<PreparedRequest>(cached.prepare(prompts[i % 2])));
        }
        for (std::size_t i = 0; i < pending.size(); ++i) {
            const auto result = cached.run(*pending[i], nullptr);
            assert(result.finish_reason == sinfer::FinishReason::StopString);
            assert(result.completion_token_ids == references[i % 2].completion_token_ids);
        }
        cached.shrink_kv();
        assert(cached.memory_summary().sequence.capacity_bytes == cached_sequence_bytes);
    }
    std::cout << "Early-stop prefix reuse, scores and sleep/wake passed\n";
}
