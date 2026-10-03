// Opt-in real-model regression for base rounds (#262). Supply a prepared artifact and a
// compatible adapter through SUROGATE_LORA_TEST_{ARTIFACT,OLD} -- one that changes the policy
// (other greedy tokens on the prompt below, or scores half a nat away), since the test tells
// the two policies apart; reserve the device with CUDA_VISIBLE_DEVICES. SUROGATE_LORA_TEST_SPEC=mtp|dflash serves with
// that draft head, SUROGATE_LORA_TEST_RANK overrides the rank bound (8), and
// SUROGATE_LORA_TEST_EAGER disables CUDA graphs.
//
// An engine that carries adapters runs a round none of whose rows selects one as an engine
// without adapters runs it (ops::ScopedLoraBaseRound). That is a second set of kernels and a
// second set of graphs behind every request, chosen per round, so on real weights:
//
//   - a base request alone answers as the same engine without adapters answers -- the same
//     kernels, so the same scores, not merely close ones;
//   - a base request whose rounds are shared with an adapter request -- which are therefore
//     not base rounds -- still answers as the base model: the same policy through other
//     kernels (the adapter-capable routes, a wider round), so the tokens must agree and the
//     scores only roughly. Its neighbour still answers as the adapter and not as the base
//     model; how closely it repeats its own answer alone is printed and not asserted, because
//     that is the engine's width-dependence acting on whatever adapter the test was given
//     (a random one diverges within a few greedy tokens, with or without base rounds);
//   - neither flavor leaves anything behind for the other: alone again, each request scores
//     what it scored alone before;
//   - and the adapter is a policy change at all, or none of the above shows anything.
#include "serve/generation_service.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <thread>

using namespace std::chrono_literals;
using sinfer::serve::GenerationOutcome;
using sinfer::serve::GenerationRequest;
using sinfer::serve::GenerationService;

static GenerationOutcome generate(GenerationService& service, const GenerationRequest& request) {
    auto prepared = service.prepare(request);
    return service.run(prepared, nullptr);
}

// How far two answers agree: the tokens they share from the start, and the widest gap between
// their scores over those tokens and the one after (the token where they part, if they do).
struct Agreement {
    std::size_t common = 0;
    std::size_t tokens = 0;
    float gap          = 0.0F;
    [[nodiscard]] bool same_tokens() const noexcept { return common == tokens; }
};
static Agreement agreement(const GenerationOutcome& actual, const GenerationOutcome& expected) {
    Agreement out;
    out.tokens = std::max(actual.completion_token_ids.size(), expected.completion_token_ids.size());
    const std::size_t shared =
        std::min(actual.completion_token_ids.size(), expected.completion_token_ids.size());
    while (out.common < shared &&
           actual.completion_token_ids[out.common] == expected.completion_token_ids[out.common]) {
        ++out.common;
    }
    const std::size_t scored = std::min({out.common + 1, actual.token_logprobs.size(),
                                         expected.token_logprobs.size()});
    for (std::size_t i = 0; i < scored; ++i) {
        assert(std::isfinite(actual.token_logprobs[i]));
        out.gap = std::max(out.gap, std::abs(actual.token_logprobs[i] - expected.token_logprobs[i]));
    }
    return out;
}
static std::ostream& operator<<(std::ostream& out, const Agreement& a) {
    return out << a.common << " of " << a.tokens << " tokens in common, widest score gap " << a.gap;
}

// The same policy: the same tokens, and scores within `tolerance`. Two runs through the same
// kernels at the same width agree to rounding (kSameKernels); the same model through another
// kernel family -- the adapter-capable routes instead of the fused ones, the int8-activation
// prefill experts instead of the decode ones -- moves a confident token's score by a tenth of
// a nat on quantised weights, which is the engine's width-dependence and not this test's
// subject (kOtherKernels).
constexpr float kSameKernels  = 0.005F;
constexpr float kOtherKernels = 0.5F;
static bool failed = false;
static void same_policy(const char* what, const GenerationOutcome& actual,
                        const GenerationOutcome& expected, float tolerance) {
    const Agreement a = agreement(actual, expected);
    const bool ok     = a.same_tokens() && a.gap < tolerance;
    std::cout << (ok ? "ok    " : "FAILED") << "  " << what << ": " << a << std::endl;
    failed |= !ok;
}
// The adapter's own row beside a base row: still the adapter's policy, not the base model's.
static void still_adapted(const char* what, const GenerationOutcome& shared,
                          const GenerationOutcome& adapted_alone,
                          const GenerationOutcome& base_alone) {
    const bool ok = !agreement(shared, base_alone).same_tokens();
    std::cout << (ok ? "ok    " : "FAILED") << "  " << what << ": differs from the base answer; against "
              << "its answer alone, " << agreement(shared, adapted_alone) << std::endl;
    failed |= !ok;
}

static void await_decode(GenerationService& service) {
    const auto deadline = std::chrono::steady_clock::now() + 60s;
    while (service.runtime_stats().decode_ready_requests == 0) {
        assert(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(1ms);
    }
}

int main() {
    const auto artifact = std::getenv("SUROGATE_LORA_TEST_ARTIFACT");
    const auto adapter  = std::getenv("SUROGATE_LORA_TEST_OLD");
    if (!artifact || !adapter) { return 77; }

    sinfer::serve::ServeOptions options;
    options.artifact_path   = artifact;
    options.max_context     = 2048;
    options.kv_capacity     = sinfer::KvCapacityPolicy::explicit_capacity(4 * 2048);
    options.max_concurrency = 4;
    options.max_loras       = 1;
    options.max_lora_rank   = 8;
    if (const auto rank = std::getenv("SUROGATE_LORA_TEST_RANK")) {
        options.max_lora_rank = static_cast<std::uint32_t>(std::stoul(rank));
    }
    options.use_cuda_graph = std::getenv("SUROGATE_LORA_TEST_EAGER") == nullptr;
    if (const auto backend = std::getenv("SUROGATE_LORA_TEST_SPEC")) {
        options.speculative.backend = std::string_view(backend) == "mtp"
                                          ? sinfer::SpeculativeBackend::Mtp
                                          : sinfer::SpeculativeBackend::DFlash;
        options.speculative.draft_tokens = 2;
    }

    GenerationRequest request;
    request.raw_prompt = "Continue the sequence of integers, separated by commas:\n1, 2, 3, 4,";
    request.max_tokens          = 256;
    request.max_tokens_set      = true;
    request.ignore_eos          = true;
    request.sampling.temperature = 0;
    request.want_logprobs       = true;
    request.return_token_ids    = true;
    auto warmup       = request;
    warmup.max_tokens = 4;

    // The reference: this engine as it serves without adapters.
    GenerationOutcome plain;
    {
        options.enable_lora = false;
        GenerationService service(options);
        (void)generate(service, warmup);
        service.shrink_kv();
        plain = generate(service, request);
        assert(plain.completion_tokens == request.max_tokens);
    }

    options.enable_lora = true;
    GenerationService service(options);
    service.load_lora_adapter("policy", adapter);
    auto adapted_request         = request;
    adapted_request.lora_adapter = "policy";
    auto adapted_warmup          = warmup;
    adapted_warmup.lora_adapter  = "policy";

    (void)generate(service, warmup);
    service.shrink_kv();
    const auto base_alone = generate(service, request);
    assert(base_alone.metrics.prefix_cache_hit_tokens == 0);
    same_policy("base request alone, against the engine without adapters", base_alone, plain,
                kSameKernels);

    (void)generate(service, adapted_warmup);
    service.shrink_kv();
    const auto adapted_alone = generate(service, adapted_request);
    assert(adapted_alone.completion_tokens == request.max_tokens);
    {
        // The test adapter must exercise a real policy change: other tokens, or scores further
        // from the base model's than another kernel family puts them.
        const Agreement a = agreement(adapted_alone, base_alone);
        const bool ok     = !a.same_tokens() || a.gap > kOtherKernels;
        std::cout << (ok ? "ok    " : "FAILED") << "  adapter against base (must differ): " << a
                  << std::endl;
        failed |= !ok;
    }

    service.shrink_kv();
    {
        // The adapter request decodes first, so every round the base request joins carries an
        // adapter row: its prefill rides a mixed round and its tokens come off adapter rounds.
        auto adapted = service.prepare(adapted_request);
        await_decode(service);
        auto base = service.prepare(request);
        const auto base_shared    = service.run(base, nullptr);
        const auto adapted_shared = service.run(adapted, nullptr);
        same_policy("base request sharing rounds with the adapter", base_shared, base_alone,
                    kOtherKernels);
        still_adapted("adapter request sharing rounds with the base", adapted_shared,
                      adapted_alone, base_alone);
    }
    service.shrink_kv();
    {
        // The other order: the base request's first rounds are base rounds, and the adapter
        // request then joins it.
        auto base = service.prepare(request);
        await_decode(service);
        auto adapted = service.prepare(adapted_request);
        const auto adapted_shared = service.run(adapted, nullptr);
        const auto base_shared    = service.run(base, nullptr);
        same_policy("base request joined by the adapter", base_shared, base_alone, kOtherKernels);
        still_adapted("adapter request joining the base", adapted_shared, adapted_alone,
                      base_alone);
    }

    service.shrink_kv();
    same_policy("base request alone again", generate(service, request), base_alone, kSameKernels);
    service.shrink_kv();
    same_policy("adapter request alone again", generate(service, adapted_request), adapted_alone,
                kSameKernels);
    if (failed) {
        std::cout << "base rounds: FAILED\n";
        return 1;
    }
    std::cout << "base rounds: base and adapter policies hold alone, shared and after each other\n";
}
