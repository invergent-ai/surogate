#include "api/engine.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

sinfer::EngineOptions ordinary_engine_options(const char* artifact) {
    sinfer::EngineOptions options;
    options.artifact_path  = artifact;
    options.max_context    = 128;
    options.kv_capacity    = sinfer::KvCapacityPolicy::explicit_capacity(128);
    options.prefill_chunk  = 128;
    options.kv_cache       = sinfer::KvCacheStorage::BFloat16;
    options.use_cuda_graph = false;
    options.enable_vision  = false;
    return options;
}

sinfer::EngineOptions dflash_engine_options(const char* artifact, sinfer::ProposalHead proposal,
                                            std::uint32_t max_context) {
    sinfer::EngineOptions options     = ordinary_engine_options(artifact);
    options.max_context               = max_context;
    options.kv_capacity               = sinfer::KvCapacityPolicy::explicit_capacity(max_context);
    options.speculative.backend       = sinfer::SpeculativeBackend::DFlash;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = proposal;
    options.use_cuda_graph            = true;
    return options;
}

sinfer::RequestOptions greedy_options(std::uint32_t outputs, bool reuse) {
    sinfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

sinfer::PromptInput initial_conversation() {
    sinfer::PromptInput input;
    input.options.enable_thinking = false;

    sinfer::ChatMessage user;
    user.role = sinfer::ChatRole::User;
    user.parts.push_back(sinfer::MessagePart{
        .kind = sinfer::MessagePartKind::Text, .text = "Name one prime number.", .media = {}});
    input.messages.push_back(std::move(user));
    return input;
}

sinfer::PromptInput followup_conversation(const sinfer::GenerationResult& first,
                                          std::string followup) {
    sinfer::PromptInput input = initial_conversation();
    sinfer::ChatMessage assistant;
    assistant.role              = sinfer::ChatRole::Assistant;
    assistant.reasoning_content = first.reasoning;
    assistant.parts.push_back(sinfer::MessagePart{
        .kind = sinfer::MessagePartKind::Text, .text = first.content, .media = {}});
    input.messages.push_back(std::move(assistant));

    sinfer::ChatMessage next;
    next.role = sinfer::ChatRole::User;
    next.parts.push_back(sinfer::MessagePart{
        .kind = sinfer::MessagePartKind::Text, .text = std::move(followup), .media = {}});
    input.messages.push_back(std::move(next));
    return input;
}

sinfer::PromptInput altered_history_after_boundary() {
    sinfer::PromptInput input = initial_conversation();
    sinfer::ChatMessage assistant;
    assistant.role = sinfer::ChatRole::Assistant;
    assistant.parts.push_back(sinfer::MessagePart{
        .kind = sinfer::MessagePartKind::Text, .text = "This history was changed.", .media = {}});
    input.messages.push_back(std::move(assistant));

    sinfer::ChatMessage next;
    next.role = sinfer::ChatRole::User;
    next.parts.push_back(sinfer::MessagePart{
        .kind = sinfer::MessagePartKind::Text, .text = "Continue briefly.", .media = {}});
    input.messages.push_back(std::move(next));
    return input;
}

int verify_dflash_load(const sinfer::Engine& engine) {
    const sinfer::LoadSummary load = engine.load_summary();
    if (load.target != "qwen3_6_moe" || load.weights_id != "groupwise-int" ||
        load.host_to_device_bytes == 0 || load.artifact_bytes_read < load.host_to_device_bytes) {
        std::cerr << "DFlash Engine materialized an invalid artifact payload: target="
                  << load.target << " weights=" << load.weights_id << '\n';
        return 1;
    }
    const sinfer::MemorySummary memory = engine.memory_summary();
    if (memory.max_context != 4352 || memory.kv_cache != sinfer::KvCacheStorage::BFloat16 ||
        memory.kv_payload_bytes == 0 || memory.weights.capacity_bytes == 0 ||
        memory.weights.used_bytes == 0 ||
        memory.weights.used_bytes > memory.weights.capacity_bytes ||
        memory.sequence.capacity_bytes == 0 || memory.sequence.used_bytes == 0 ||
        memory.sequence.used_bytes > memory.sequence.capacity_bytes ||
        memory.workspace.capacity_bytes == 0 || memory.request_transient.capacity_bytes != 0 ||
        memory.workspace_logical_peak_bytes != 0 || memory.cuda_graph_allowance_bytes == 0) {
        std::cerr << "DFlash Engine has an invalid frozen memory layout\n";
        return 1;
    }
    return 0;
}

int exercise_partial_terminal(sinfer::Engine& engine, const std::vector<sinfer::TokenId>& prompt,
                              const std::vector<sinfer::TokenId>& baseline) {
    for (std::size_t stop_index = 1; stop_index < baseline.size(); ++stop_index) {
        const sinfer::TokenId stop = baseline[stop_index];
        if (std::find(baseline.begin(), baseline.begin() + static_cast<std::ptrdiff_t>(stop_index),
                      stop) != baseline.begin() + static_cast<std::ptrdiff_t>(stop_index)) {
            continue;
        }
        sinfer::RequestOptions options = greedy_options(24, false);
        options.stop.token_ids.push_back(stop);
        const sinfer::GenerationResult stopped =
            engine.generate(engine.prepare_tokens(prompt), options);
        if (stopped.finish_reason != sinfer::FinishReason::StopToken) { continue; }
        if (stopped.generated_token_ids.size() != stop_index + 1 ||
            !std::equal(stopped.generated_token_ids.begin(), stopped.generated_token_ids.end(),
                        baseline.begin())) {
            std::cerr << "partial DFlash terminal diverged from the target baseline prefix\n";
            return 1;
        }

        const std::uint64_t fully_licensed = 1 + stopped.speculative.rounds +
                                             stopped.speculative.accepted_tokens +
                                             stopped.speculative.fallback_steps;
        if (stopped.generated_token_ids.size() >= fully_licensed) { continue; }

        std::vector<sinfer::TokenId> continuation = prompt;
        continuation.insert(continuation.end(), stopped.generated_token_ids.begin(),
                            stopped.generated_token_ids.end());
        continuation.push_back(198);
        const sinfer::GenerationResult reused = engine.generate(
            engine.prepare_tokens(std::move(continuation)), greedy_options(2, true));
        const std::uint32_t expected_reuse =
            static_cast<std::uint32_t>(prompt.size() + stopped.generated_token_ids.size() - 1);
        if (reused.reused_prompt_tokens != expected_reuse ||
            reused.generated_token_ids.size() != 2) {
            std::cerr << "partial DFlash terminal did not publish its exact context frontier: "
                      << "reused=" << reused.reused_prompt_tokens << " expected=" << expected_reuse
                      << '\n';
            return 1;
        }
        return 0;
    }
    std::cerr << "fixed DFlash fixture exposed no terminal stop inside a licensed batch\n";
    return 1;
}

int exercise_boundary_restore(sinfer::Engine& engine) {
    const sinfer::GenerationResult first =
        engine.generate(engine.prepare(initial_conversation()), greedy_options(2, false));
    if (first.generated_token_ids.size() != 2) {
        std::cerr << "DFlash boundary fixture did not establish resident state\n";
        return 1;
    }

    const sinfer::GenerationResult restored =
        engine.generate(engine.prepare(followup_conversation(first, "Answer with one digit.")),
                        greedy_options(4, true));
    if (restored.reused_prompt_tokens == 0 || restored.generated_token_ids.size() != 4) {
        std::cerr << "DFlash assistant-boundary restore did not reuse its cache snapshot: reused="
                  << restored.reused_prompt_tokens
                  << " outputs=" << restored.generated_token_ids.size() << '\n';
        return 1;
    }
    const sinfer::GenerationResult baseline =
        engine.generate(engine.prepare(followup_conversation(first, "Answer with one digit.")),
                        greedy_options(4, false));
    if (baseline.generated_token_ids != restored.generated_token_ids) {
        std::cerr << "DFlash boundary restore changed greedy target output\n";
        return 1;
    }
    return 0;
}

int exercise_long_boundary_restore(sinfer::Engine& engine) {
    constexpr std::uint32_t generated_tokens = 4100;
    const sinfer::GenerationResult long_run  = engine.generate(
        engine.prepare(initial_conversation()), greedy_options(generated_tokens, false));
    if (long_run.generated_token_ids.size() != generated_tokens) {
        std::cerr << "DFlash long-restore fixture did not cross the cyclic-cache window\n";
        return 1;
    }
    const std::uint32_t resident_frontier = long_run.prompt.prompt_tokens + generated_tokens - 1;
    const sinfer::GenerationResult restored =
        engine.generate(engine.prepare(altered_history_after_boundary()), greedy_options(2, true));
    if (restored.reused_prompt_tokens == 0 ||
        resident_frontier - restored.reused_prompt_tokens <= 4096 ||
        restored.generated_token_ids.size() != 2) {
        std::cerr << "DFlash long-distance restore did not use the saved cyclic cache: resident="
                  << resident_frontier << " restored=" << restored.reused_prompt_tokens << '\n';
        return 1;
    }
    const sinfer::GenerationResult baseline =
        engine.generate(engine.prepare(altered_history_after_boundary()), greedy_options(2, false));
    if (baseline.generated_token_ids != restored.generated_token_ids) {
        std::cerr << "DFlash long-distance boundary restore changed greedy target output\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("SINFER_QWEN3_6_35B_A3B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: SINFER_QWEN3_6_35B_A3B_WEIGHTS is not set\n";
        return 77;
    }

    const std::vector<sinfer::TokenId> prompt{
        248045, 846,    198, 109266, 3709,  96220, 117443, 97913,
        1710,   248046, 198, 248045, 74455, 198,   248068, 198,
    };
    std::vector<sinfer::TokenId> target_output;
    {
        sinfer::Engine ordinary(ordinary_engine_options(artifact));
        target_output =
            ordinary.generate(ordinary.prepare_tokens(prompt), greedy_options(24, false))
                .generated_token_ids;
        if (target_output.size() != 24) {
            std::cerr << "ordinary target baseline did not generate 24 tokens\n";
            return 1;
        }
    }

    {
        sinfer::EngineOptions options =
            dflash_engine_options(artifact, sinfer::ProposalHead::Full, 128);
        options.max_concurrency = 2;
        sinfer::Engine full(std::move(options));
        auto first  = full.submit(full.prepare_tokens(prompt), greedy_options(17, false));
        auto second = full.submit(full.prepare_tokens(prompt), greedy_options(9, false));
        const sinfer::GenerationResult first_result  = first.wait();
        const sinfer::GenerationResult second_result = second.wait();
        const auto valid = [&](const sinfer::GenerationResult& result, std::size_t count) {
            return result.generated_token_ids.size() == count &&
                   std::equal(result.generated_token_ids.begin(), result.generated_token_ids.end(),
                              target_output.begin(),
                              target_output.begin() + static_cast<std::ptrdiff_t>(count)) &&
                   result.speculative.backend == sinfer::SpeculativeBackend::DFlash &&
                   result.speculative.rounds != 0;
        };
        if (!valid(first_result, 17) || !valid(second_result, 9)) {
            std::cerr << "concurrent full-head DFlash Graph route diverged from ordinary target "
                         "output\n";
            return 1;
        }
    }

    sinfer::Engine engine(dflash_engine_options(artifact, sinfer::ProposalHead::Optimized, 4352));
    if (const int result = verify_dflash_load(engine); result != 0) { return result; }
    engine.reset_memory_peaks();
    const sinfer::GenerationResult dflash =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(24, false));
    if (dflash.generated_token_ids != target_output) {
        const auto mismatch =
            std::mismatch(dflash.generated_token_ids.begin(), dflash.generated_token_ids.end(),
                          target_output.begin(), target_output.end());
        std::cerr << "DFlash Graph route diverged from ordinary greedy target output at "
                  << static_cast<std::size_t>(mismatch.first - dflash.generated_token_ids.begin())
                  << ": dflash="
                  << (mismatch.first == dflash.generated_token_ids.end() ? -1 : *mismatch.first)
                  << " target=" << (mismatch.second == target_output.end() ? -1 : *mismatch.second)
                  << '\n';
        return 1;
    }
    const sinfer::MemorySummary memory = engine.memory_summary();
    if (memory.workspace_logical_peak_bytes == 0 ||
        memory.workspace_logical_peak_bytes > memory.workspace.capacity_bytes) {
        std::cerr << "DFlash request did not report a valid planned workspace phase\n";
        return 1;
    }
    if (dflash.speculative.backend != sinfer::SpeculativeBackend::DFlash ||
        dflash.speculative.rounds == 0) {
        std::cerr << "DFlash fixture did not execute speculative decode\n";
        return 1;
    }
    if (const int result = exercise_partial_terminal(engine, prompt, target_output); result != 0) {
        return result;
    }
    if (const int result = exercise_boundary_restore(engine); result != 0) { return result; }
    if (const int result = exercise_long_boundary_restore(engine); result != 0) { return result; }

    std::cout << "ok\n";
    return 0;
}
