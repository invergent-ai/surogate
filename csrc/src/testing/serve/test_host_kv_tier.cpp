// Opt-in: SUROGATE_HOST_KV_TEST_ARTIFACT, a hybrid Qwen3.5/3.6 artifact (Gated DeltaNet state +
// attention KV). SUROGATE_HOST_KV_TEST_MTP=1 serves it with its MTP head (two drafts);
// SUROGATE_HOST_KV_TEST_LEGACY=1 runs without the host tier: a spill drops the conversation, and
// a conversation archived when another request took its lane survives a later eviction (it used
// to be wiped by every one).
//
// The host prefix tier (--prefix-cache-host-gb), driven through the program so every step is
// deterministic. A conversation's first turn is prefilled and decoded greedily on a lane; the
// next turn (that prompt, the answer, and new tokens) then runs three ways:
//   gpu   the lane stayed retained on the GPU (append at its frontier);
//   host  the KV pool took the lane back (spill_retained_lane) and the turn resumes from the host
//         snapshot: same reuse frontier, and tokens, sampled log-probabilities and top-5 scores
//         bit-identical to `gpu` -- the snapshot holds the KV bytes and recurrent state exactly;
//   fresh nothing cached: the whole prompt is prefilled. Generated tokens' KV then comes from
//         the prefill kernels instead of the decode rounds, so log-probabilities move by
//         rounding; reported, and checked against a loose bound.
#include "core/device.h"
#include "targets/registry.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
void require(bool condition, std::string_view message) {
    if (!condition) { throw std::runtime_error(std::string(message)); }
}

std::uint32_t env_or(const char* name, std::uint32_t fallback) {
    const char* raw = std::getenv(name);
    return raw ? static_cast<std::uint32_t>(std::strtoul(raw, nullptr, 10)) : fallback;
}

struct Turn {
    std::vector<sinfer::TokenId> tokens;
    std::vector<float> logprobs;
    std::vector<sinfer::TokenScore> scores;
    std::uint32_t reused = 0;
};

std::vector<sinfer::TokenId> pseudo_text(std::size_t count, std::uint32_t seed) {
    // Ordinary vocabulary ids, away from the special tokens at the top of the table.
    std::vector<sinfer::TokenId> out(count);
    std::uint64_t x = 0x9e3779b97f4a7c15ULL * (seed + 1);
    for (auto& token : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        token = static_cast<sinfer::TokenId>(1000 + x % 120000);
    }
    return out;
}

template <class Instance>
void exercise(Instance& instance, bool tier, bool mtp) {
    auto& program  = *instance.program;
    auto& frontend = instance.loaded->frontend;
    const std::uint32_t generate = env_or("SUROGATE_HOST_KV_TEST_TOKENS", 48);

    const auto run_turn = [&](std::uint32_t lane, const std::vector<sinfer::TokenId>& ids, bool reuse) {
        auto prompt = frontend.prepare_tokens(std::vector<sinfer::TokenId>(ids));
        sinfer::runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens = generate;
        options.top_logprobs            = 5;
        options.allow_prefix_reuse      = reuse;
        auto base = program.plan_request_base(prompt, options);
        auto plan = program.plan_request_for_lane(lane, prompt, base);
        require(program.can_admit_lane(lane, plan), "turn must fit before prefill");
        Turn turn;
        turn.reused        = plan.summary().reusable_prompt_tokens;
        const auto summary = plan.summary();
        instance.request_memory.activate(summary.transient_bytes, summary.transient_alignment);
        auto step = program.start_prefill_lane(lane, std::move(prompt), std::move(plan),
                                               instance.request_memory.region());
        while (!step.complete) { step = program.advance_prefill_lane(lane); }
        require(step.round.tokens.size() == 1, "prefill must produce one token");
        turn.tokens.push_back(step.round.tokens.front());
        turn.logprobs.push_back(step.round.logprobs.empty() ? NAN : step.round.logprobs.front());
        program.resolve_prefill_lane(lane, generate == 1);
        instance.request_memory.deactivate();
        const std::array<std::uint32_t, 1> lanes{lane};
        while (turn.tokens.size() < generate) {
            const std::uint32_t left = generate - static_cast<std::uint32_t>(turn.tokens.size());
            const std::array<sinfer::runtime::RoundBudget, 1> budgets{{{left}}};
            auto round = program.decode_batch(lanes, budgets);
            const std::uint32_t count = std::min<std::uint32_t>(static_cast<std::uint32_t>(round.row_counts[0]), left);
            require(count > 0, "a round must produce a token");
            for (std::uint32_t i = 0; i < count; ++i) {
                turn.tokens.push_back(round.tokens[i]);
                turn.logprobs.push_back(round.logprobs.empty() ? NAN : round.logprobs[i]);
            }
            const std::array<std::uint32_t, 1> accepted{count};
            const std::array<std::uint8_t, 1> terminal{static_cast<std::uint8_t>(turn.tokens.size() >= generate)}, cancelled{};
            program.resolve_pending_batch(lanes, accepted, terminal, cancelled);
        }
        sinfer::GenerationResult result;
        result.generated_token_ids = turn.tokens;
        program.collect_logprobs(lane, result);
        turn.scores = result.completion_logprobs;
        return turn;
    };
    const auto same_bits = [](float a, float b) {
        return std::memcmp(&a, &b, sizeof(float)) == 0 || (std::isnan(a) && std::isnan(b));
    };
    const auto identical = [&](const Turn& a, const Turn& b) {
        if (a.tokens != b.tokens || a.logprobs.size() != b.logprobs.size() || a.scores.size() != b.scores.size()) {
            return false;
        }
        for (std::size_t i = 0; i < a.logprobs.size(); ++i) {
            if (!same_bits(a.logprobs[i], b.logprobs[i])) { return false; }
        }
        for (std::size_t i = 0; i < a.scores.size(); ++i) {
            const auto& x = a.scores[i];
            const auto& y = b.scores[i];
            if (x.selected.token_id != y.selected.token_id || !same_bits(x.selected.logprob, y.selected.logprob) ||
                x.top.size() != y.top.size()) {
                return false;
            }
            for (std::size_t k = 0; k < x.top.size(); ++k) {
                if (x.top[k].token_id != y.top[k].token_id || !same_bits(x.top[k].logprob, y.top[k].logprob)) {
                    return false;
                }
            }
        }
        return true;
    };
    const auto reset = [&] {
        for (std::uint32_t lane = 0; lane < 2; ++lane) { program.evict_retained_lane(lane); }
        program.evict_archived_prefixes();
    };

    const std::uint32_t first_tokens = env_or("SUROGATE_HOST_KV_TEST_PROMPT", 3000);
    const auto first = pseudo_text(first_tokens, 1);
    const auto suffix = pseudo_text(env_or("SUROGATE_HOST_KV_TEST_SUFFIX", 700), 2);

    // gpu: the second turn appends to the lane the first one left retained.
    const Turn turn1 = run_turn(0, first, true);
    require(program.has_retained_lane(0), "a finished turn must stay retained on its lane");
    std::vector<sinfer::TokenId> second = first;
    second.insert(second.end(), turn1.tokens.begin(), turn1.tokens.end());
    second.insert(second.end(), suffix.begin(), suffix.end());
    const Turn gpu = run_turn(0, second, true);
    std::cout << "gpu:   reused " << gpu.reused << " of " << second.size() << " prompt tokens\n";
    require(gpu.reused >= first.size(), "the retained lane must serve the second turn");
    reset();

    // host: the same first turn, then the KV pool takes the lane back.
    const auto before = program.host_kv_stats();
    const Turn again = run_turn(0, first, true);
    require(identical(again, turn1), "greedy decoding must repeat exactly for the comparison to mean anything");
    if (!tier) {
        program.spill_retained_lane(0);
        require(!program.has_retained_lane(0), "a spilled lane must give its pages back");
        const auto spilled = program.host_kv_stats();
        require(spilled.dropped_lanes == before.dropped_lanes + 1, "without the tier a spill drops the conversation");
        const Turn none = run_turn(0, second, true);
        require(none.reused == 0, "without the tier nothing is left to resume from");
        std::cout << "legacy: spill dropped the conversation; the next turn prefilled all " << second.size()
                  << " tokens\n";
        reset();
        // Regression: the 512 MiB archive keeps a conversation another request displaced, and an
        // eviction of some other retained lane must not wipe it.
        (void)run_turn(0, first, true);
        (void)run_turn(0, pseudo_text(600, 9), true); // takes lane 0: the first turn goes to the archive
        require(program.host_kv_stats().entries == 1, "a displaced conversation must be archived");
        program.evict_retained_lane(0);
        require(program.host_kv_stats().entries == 1, "evicting a retained lane must not wipe the archive");
        const Turn archived = run_turn(1, second, true);
        require(archived.reused >= first.size(), "the archived conversation must still serve its next turn");
        require(identical(archived, gpu), "an archive restore must reproduce the GPU hit");
        std::cout << "legacy: archive survived an eviction; restored turn bit-identical to the GPU hit\n";
        reset();
        return;
    }
    // A finished turn goes to the host in the background: evicting it later copies nothing.
    program.background_offload();
    const auto offloaded = program.host_kv_stats();
    require(offloaded.offloads == before.offloads + 1 && offloaded.entries == 1,
            "a retained conversation must be copied to the host in the background");
    program.spill_retained_lane(0);
    require(!program.has_retained_lane(0), "a spilled lane must give its pages back");
    require(program.kv_occupancy().entitled_pages == 0, "a spilled lane must release its entitlement");
    const auto spilled = program.host_kv_stats();
    require(spilled.spills_free == before.spills_free + 1 && spilled.spills == before.spills,
            "a lane copied in the background must be evicted without another copy");
    require(spilled.entries == 1, "the spill must keep one host snapshot");
    std::cout << "spill: " << spilled.used_bytes / (1 << 20) << " MiB of the pool for " << first.size() + generate
              << " tokens (" << spilled.offload_bytes / (1 << 20) << " MiB copied)\n";
    const Turn host = run_turn(0, second, true);
    const auto restored = program.host_kv_stats();
    std::cout << "host:  reused " << host.reused << " of " << second.size() << " prompt tokens\n";
    require(restored.restores == spilled.restores + 1, "the second turn must resume from the host snapshot");
    require(host.reused == gpu.reused, "host and GPU hits must resume at the same frontier");
    require(identical(host, gpu), "a host restore must reproduce the GPU hit bit for bit");
    std::cout << "host == gpu: " << host.tokens.size() << " tokens, logprobs and top-5 scores bit-identical\n";

    // A longer snapshot of the same conversation replaces the shorter one (copied at eviction
    // this time: no background copy ran), and reuses its whole chunks.
    program.spill_retained_lane(0);
    const auto twice = program.host_kv_stats();
    require(twice.spills == restored.spills + 1, "a lane with no background copy is copied when it is evicted");
    require(twice.entries == 1 && twice.superseded == restored.superseded + 1,
            "the second snapshot must supersede the first");
    require(twice.shared_objects > restored.shared_objects || first.size() < 1024,
            "the longer snapshot must find the shorter one's chunks stored");
    std::vector<sinfer::TokenId> third = second;
    third.insert(third.end(), host.tokens.begin(), host.tokens.end());
    const auto more = pseudo_text(300, 3);
    third.insert(third.end(), more.begin(), more.end());
    const Turn next = run_turn(0, third, true);
    require(next.reused >= second.size(), "the longer snapshot must serve the conversation's next turn");
    std::cout << "host restores: " << program.host_kv_stats().restores << ", restore copy p50 bucket <= "
              << [&] {
                     const auto h = program.host_kv_stats().restore_ms;
                     std::uint64_t seen = 0;
                     for (std::size_t i = 0; i < h.counts.size(); ++i) {
                         seen += h.counts[i];
                         if (seen * 2 >= h.count) {
                             return i < sinfer::LatencyHistogram::kBoundsMs.size() ? sinfer::LatencyHistogram::kBoundsMs[i] : 1e9;
                         }
                     }
                     return 0.0;
                 }()
              << " ms\n";
    // Adapter updates and idle shrink invalidate the tier.
    program.evict_archived_prefixes();
    require(program.host_kv_stats().entries == 0, "invalidation must empty the host index");
    reset();

    // fresh: the whole second prompt prefilled.
    const Turn fresh = run_turn(0, second, false);
    require(fresh.reused == 0, "a turn without reuse must prefill everything");
    std::size_t agree = 0;
    while (agree < fresh.tokens.size() && fresh.tokens[agree] == gpu.tokens[agree]) { ++agree; }
    double max_delta = 0.0, sum_delta = 0.0;
    std::size_t counted = 0;
    for (std::size_t i = 0; i < std::min(agree + 1, fresh.logprobs.size()); ++i) {
        if (std::isnan(fresh.logprobs[i]) || std::isnan(gpu.logprobs[i])) { continue; }
        const double delta = std::abs(static_cast<double>(fresh.logprobs[i]) - gpu.logprobs[i]);
        max_delta = std::max(max_delta, delta);
        sum_delta += delta;
        ++counted;
    }
    std::cout << "fresh vs gpu: first " << agree << " of " << gpu.tokens.size()
              << " tokens agree; sampled-token logprob |delta| mean "
              << (counted ? sum_delta / counted : 0.0) << ", max " << max_delta << " over " << counted
              << " positions" << (mtp ? " (MTP)" : "") << "\n";
    // Not a bit-for-bit comparison: the answer's KV comes from the prefill kernels here and from the
    // decode rounds in a cached turn, and pseudo-random prompts leave near-ties everywhere. The
    // bound only catches a restore that is wrong, not rounding.
    require(counted == 0 || sum_delta / counted < 2.0, "a fresh prefill must stay close to a cached turn");
    reset();
}
} // namespace

int main() {
    const auto* artifact = std::getenv("SUROGATE_HOST_KV_TEST_ARTIFACT");
    if (!artifact) {
        std::cout << "SKIP: set SUROGATE_HOST_KV_TEST_ARTIFACT\n";
        return 77;
    }
    try {
        // A shared-prefix capture cuts the second prefill of a prompt at the shared boundary, which
        // changes its arithmetic; the comparisons below need every run of a prompt to be the same.
        ::setenv("SUROGATE_SERVE_SHARED_PREFIX_SLOTS", "0", 1);
        const bool tier = std::getenv("SUROGATE_HOST_KV_TEST_LEGACY") == nullptr;
        const bool mtp  = std::getenv("SUROGATE_HOST_KV_TEST_MTP") != nullptr;
        sinfer::DeviceContext device(0);
        sinfer::EngineOptions options;
        options.artifact_path   = artifact;
        options.max_context     = 8192;
        options.kv_capacity     = sinfer::KvCapacityPolicy::explicit_capacity(16384);
        options.max_concurrency = 2;
        options.prefill_chunk   = 2048;
        options.kv_cache        = sinfer::KvCacheStorage::Fp8E4M3;
        options.use_cuda_graph  = !std::getenv("SUROGATE_HOST_KV_TEST_EAGER");
        if (mtp) {
            options.speculative.backend      = sinfer::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens = 2;
            options.speculative.max_lanes    = sinfer::kSpeculateAtAnyWidth;
        }
        if (tier) { options.prefix_cache_host_bytes = std::size_t{2} << 30; }
        auto target = sinfer::targets::construct_target(options, device);
        if (auto* moe = std::get_if<std::unique_ptr<sinfer::targets::Qwen3_5MoeInstance>>(&target.active)) {
            exercise(**moe, tier, mtp);
        } else if (auto* dense = std::get_if<std::unique_ptr<sinfer::targets::Qwen3_5Instance>>(&target.active)) {
            exercise(**dense, tier, mtp);
        } else {
            throw std::runtime_error("use a Qwen3.5/3.6 hybrid artifact");
        }
        std::cout << "host KV tier: OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
