// Opt-in: SUROGATE_COMPLETED_KV_TEST_ARTIFACT, a Qwen3.5, Gemma4 E, or Llama artifact.
// Drive the program directly so a stop inside a burst is deterministic regardless
// of token contents, scheduling, or how quickly the model reaches EOS.
#include "core/device.h"
#include "targets/registry.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {
void require(bool condition, std::string_view message) {
    if (!condition) { throw std::runtime_error(std::string(message)); }
}

template <class Instance>
void exercise(Instance& instance, bool truncated_cache_reusable) {
    auto& program = *instance.program;
    auto& frontend = instance.loaded->frontend;
    auto start = [&](std::uint32_t lane, std::uint32_t prompt_tokens, bool terminal = false) {
        auto prompt = frontend.prepare_tokens(std::vector<sinfer::TokenId>(prompt_tokens, 1));
        sinfer::runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens = 16;
        options.top_logprobs = 0;
        auto base = program.plan_request_base(prompt, options);
        auto plan = program.plan_request_for_lane(lane, prompt, base);
        require(program.can_admit_lane(lane, plan), "request must fit before prefill");
        const auto summary = plan.summary();
        instance.request_memory.activate(summary.transient_bytes, summary.transient_alignment);
        auto step = program.start_prefill_lane(lane, std::move(prompt), std::move(plan),
                                               instance.request_memory.region());
        while (!step.complete) { step = program.advance_prefill_lane(lane); }
        require(step.round.tokens.size() == 1, "prefill must produce one token");
        const auto first = step.round.tokens.front();
        program.resolve_prefill_lane(lane, terminal);
        instance.request_memory.deactivate();
        return first;
    };
    auto head_fits = [&](std::uint32_t prompt_tokens, bool after_eviction) {
        auto prompt = frontend.prepare_tokens(std::vector<sinfer::TokenId>(prompt_tokens, 2));
        sinfer::runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens = 16;
        auto base = program.plan_request_base(prompt, options);
        // Reservations also cover a padded prefill window starting at any cursor.
        require(base.summary().admission.main_kv_pages == (prompt_tokens + 127 + 63) / 64,
                "head reservation must match the tight KV budget");
        for (std::uint32_t lane = 0; lane < 2; ++lane) {
            auto plan = program.plan_request_for_lane(lane, prompt, base);
            if (after_eviction ? program.can_admit_lane_after_retained_eviction(lane, plan)
                               : program.can_admit_lane(lane, plan)) { return true; }
        }
        return false;
    };

    require(program.admission_capacity().main_kv_pages == 9, "test needs nine KV pages");
    const std::array<sinfer::TokenId, 3> first{start(0, 65), start(1, 65), start(2, 65)};
    require(program.kv_occupancy().entitled_pages == 9, "three requests must fill the pool");

    const std::array<std::uint32_t, 3> lanes{0, 1, 2};
    const std::array<sinfer::runtime::RoundBudget, 3> budgets{{{15}, {15}, {15}}};
    program.set_round_burst_limit(8);
    auto round = program.decode_batch(lanes, budgets);
    for (const auto count : round.row_counts) { require(count == 8, "test needs an eight-token burst"); }
    // Stop two requests after one token; leave the third as a live incumbent.
    const std::array<sinfer::TokenId, 2> stopped{round.tokens[0], round.tokens[round.row_stride]};
    const std::array<std::uint32_t, 3> accepted{1, 1, 8};
    const std::array<std::uint8_t, 3> terminal{1, 1, 0}, cancelled{};
    program.resolve_pending_batch(lanes, accepted, terminal, cancelled);
    for (std::uint32_t lane = 0; lane < 2; ++lane) {
        require(program.has_retained_lane(lane) == truncated_cache_reusable,
                "truncated cache must only be retained when it can be reused");
        sinfer::GenerationResult result;
        result.generated_token_ids = {first[lane], stopped[lane]};
        program.collect_logprobs(lane, result);
        require(result.completion_logprobs.size() == 2, "completion scores must survive cleanup");
        require(result.completion_logprobs.back().selected.token_id == stopped[lane],
                "scores must stay aligned with the accepted prefix");
    }
    require(head_fits(257, true),
            "six-page head must fit beside the three-page incumbent after cache eviction");
    if (!truncated_cache_reusable) {
        require(program.kv_occupancy().entitled_pages == 3,
                "unretained completed lanes must release their entitlements");
        require(program.kv_occupancy().pages_in_use == 2,
                "unretained completed lanes must release their physical pages");
        require(head_fits(257, false), "head must enter immediately after dead cache cleanup");
    }
    program.abort_lane(2);
    require(head_fits(449, true), "an idle engine must admit an exclusive-feasible head");
    for (std::uint32_t lane = 0; lane < 2; ++lane) { program.evict_retained_lane(lane); }
    require(program.kv_occupancy().entitled_pages == 0, "all old entitlements must be released");
    require(head_fits(449, false), "the entire pool must be available to the next request");

    // Complete a whole round on a reused lane: an intact cache remains reusable.
    (void)start(0, 449);
    const std::array<std::uint32_t, 1> single{0};
    const std::array<sinfer::runtime::RoundBudget, 1> one_budget{{{15}}};
    round = program.decode_batch(single, one_budget);
    const std::array<std::uint32_t, 1> full{static_cast<std::uint32_t>(round.row_counts[0])};
    const std::array<std::uint8_t, 1> done{1}, live{};
    program.resolve_pending_batch(single, full, done, live);
    require(program.has_retained_lane(0), "untruncated completion must retain its cache");
    require(program.kv_occupancy().entitled_pages == 8, "retained cache must still own its pages");
    program.evict_retained_lane(0);
    require(program.kv_occupancy().entitled_pages == 0, "retained eviction must release its pages");

    // Archive A while B uses its execution lane. Invalidation for an adapter
    // replacement must remove A even though the lane remains actively decoding B.
    (void)start(0, 193, true);
    (void)start(0, 65);
    const auto reusable_archive = [&] {
        auto prompt = frontend.prepare_tokens(std::vector<sinfer::TokenId>(194, 1));
        sinfer::runtime::ResolvedExecutionOptions execution;
        execution.requested_output_tokens = 1;
        auto base = program.plan_request_base(prompt, execution);
        return program.plan_request_for_lane(1, prompt, base).summary().reusable_prompt_tokens;
    };
    require(reusable_archive() == 193, "an unrelated active request must not discard archived A");
    program.evict_archived_prefixes();
    require(reusable_archive() == 0, "adapter invalidation must clear archives while B is active");
    program.abort_lane(0);
}
} // namespace

int main() {
    const auto* artifact = std::getenv("SUROGATE_COMPLETED_KV_TEST_ARTIFACT");
    if (!artifact) {
        std::cout << "SKIP: set SUROGATE_COMPLETED_KV_TEST_ARTIFACT\n";
        return 77;
    }
    try {
        sinfer::DeviceContext device(0);
        sinfer::EngineOptions options;
        options.artifact_path = artifact;
        options.max_context = 576;
        options.kv_capacity = sinfer::KvCapacityPolicy::explicit_capacity(576);
        options.max_concurrency = 3;
        options.prefill_chunk = 128;
        options.use_cuda_graph = !std::getenv("SUROGATE_COMPLETED_KV_TEST_EAGER");
        if (std::getenv("SUROGATE_COMPLETED_KV_TEST_BF16")) {
            options.kv_cache = sinfer::KvCacheStorage::BFloat16;
        }
        auto target = sinfer::targets::construct_target(options, device);
        if (auto* instance = std::get_if<std::unique_ptr<sinfer::targets::Qwen3_5Instance>>(&target.active)) {
            exercise(**instance, false);
        } else if (auto* instance = std::get_if<std::unique_ptr<sinfer::targets::Gemma4EInstance>>(&target.active)) {
            exercise(**instance, false);
        } else if (auto* instance = std::get_if<std::unique_ptr<sinfer::targets::LlamaInstance>>(&target.active)) {
            exercise(**instance, true);
        } else {
            throw std::runtime_error("use a Qwen3.5, Gemma4 E, or Llama artifact");
        }
        std::cout << "completed KV ownership and admission: OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
