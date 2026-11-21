#pragma once

// --kv-admission ondemand (DESIGN-kv-admission.md): the decisions the executor makes when a
// running sequence needs more KV pages than the pool has free. Kept free of the executor and the
// program so the order of growth, eviction and preemption can be tested on its own.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <span>
#include <vector>

namespace sinfer::runtime {

/// A request holding a lane, as the growth pass sees it.
struct KvGrowthLane {
    std::uint32_t lane = 0;
    /// Arrival rank (the request id): lower arrived earlier. Kept across preemptions, so a
    /// resumed request keeps its place ahead of every request that arrived after it.
    std::uint64_t order = 0;
    /// Admitted on demand and recomputable; a request admitted with its whole reservation never
    /// needs to grow and is never a victim.
    bool preemptable = false;
    /// Decode-ready: its next round may write past its entitlement.
    bool needs_growth = false;
};

struct KvGrowthHooks {
    /// Grows the lane's entitlement to cover its next round; false when the pool refused.
    std::function<bool(std::uint32_t lane)> grow;
    /// Gives back some idle KV (a retained prefix-cache lane, the shared prefixes); false when
    /// nothing idle is left.
    std::function<bool()> release_idle;
    /// Preempts the lane: its pages go back and its request waits to be recomputed.
    std::function<void(std::uint32_t lane)> preempt;
};

struct KvGrowthOutcome {
    /// Lanes preempted, in order.
    std::vector<std::uint32_t> preempted;
    /// Lanes that could not grow although nothing else is left to give way: alone on the
    /// engine, or only behind older requests that cannot be preempted. Their pages are held by
    /// something no lane owns (a GPU prefix image) or the device gate refused; the caller
    /// decides whether they wait or fail.
    std::vector<std::uint32_t> blocked;
};

/// The most one unit of work writes past a lane's frontier, in tokens: an ordinary burst of
/// `burst` rounds, a verify round's anchor and drafts (`draft_window + 1`), or a DSpark draft
/// block, whose K / V covers [frontier, frontier + draft_block) whatever k is (0 for any other
/// model); plus one. A lane's entitlement covers this before each round.
[[nodiscard]] constexpr std::uint32_t kv_round_reach(std::uint32_t burst, std::uint32_t draft_window,
                                                     std::uint32_t draft_block) noexcept {
    return std::max({burst, draft_window + 1U, draft_block}) + 1U;
}

/// Whether an engine admits on demand. DFlash reserves; a DSpark drafter (draft KV in the text pool) may opt in, only with an
/// explicit --kv-admission ondemand (lane f1). Without it this is the rule every engine had before the opt-in existed.
[[nodiscard]] constexpr bool kv_engine_on_demand(bool on_demand_mode, bool pipeline_stage, bool dflash, bool dspark,
                                                 bool explicit_on_demand) noexcept {
    return on_demand_mode && !pipeline_stage && (!dflash || (dspark && explicit_on_demand));
}

/// Whether a DSpark engine verifies untruncated sampled rows by the coupled rule: SUROGATE_SPEC_COUPLED and
/// SUROGATE_SPEC_EXACT_SPLIT both on (unset, empty or a nonzero integer). The coupled verify is a split route, so the second
/// switch off leaves such rows on the one-hot rule even with the first on (review R8 minor 1).
[[nodiscard]] inline bool kv_dspark_verify_coupled(const char* coupled, const char* exact_split) noexcept {
    const auto on = [](const char* raw) { return raw == nullptr || raw[0] == '\0' || std::atoi(raw) != 0; };
    return on(coupled) && on(exact_split);
}

/// Whether a DSpark row's committed tokens are independent of its drafts, so a recompute after a preemption goes on with
/// the tokens it would have produced: a greedy row, or an untruncated sampled row under coupled verification with no
/// min_p floor (whose coupled draw can differ from ordinary decoding by 1 ulp, review R5 minor 2). A row with top_p < 1 or
/// top_k > 0 keeps the one-hot rejection rule: after a resume its continuation is drawn with other keys. Such rows keep
/// their whole reservation unless SUROGATE_SERVE_KV_TRUNCATED_ONDEMAND=1 (lane f1, agent-EN's merge condition).
[[nodiscard]] constexpr bool kv_dspark_row_recomputes_exactly(float temperature, std::int32_t top_k, float top_p,
                                                              float min_p, bool coupled) noexcept {
    return temperature <= 0.0F || (coupled && top_k <= 0 && top_p >= 1.0F && min_p <= 0.0F);
}

/// Grows every lane that needs it, oldest first. When the pool refuses, idle KV goes first, then
/// the request that arrived last is preempted -- which may be the lane that is growing -- and the
/// lane tries again. An older request is never preempted for a newer one.
[[nodiscard]] inline KvGrowthOutcome ensure_kv_growth(std::vector<KvGrowthLane> lanes,
                                                      const KvGrowthHooks& hooks) {
    KvGrowthOutcome outcome;
    std::stable_sort(lanes.begin(), lanes.end(),
                     [](const KvGrowthLane& a, const KvGrowthLane& b) { return a.order < b.order; });
    std::vector<bool> gone(lanes.size(), false);
    for (std::size_t i = 0; i < lanes.size(); ++i) {
        if (gone[i] || !lanes[i].needs_growth) { continue; }
        while (!hooks.grow(lanes[i].lane)) {
            if (hooks.release_idle && hooks.release_idle()) { continue; }
            std::size_t victim = lanes.size();
            for (std::size_t j = lanes.size(); j-- > i;) {
                if (!gone[j] && lanes[j].preemptable) {
                    victim = j;
                    break;
                }
            }
            bool others = false;
            for (std::size_t j = 0; j < lanes.size() && !others; ++j) { others = j != i && !gone[j]; }
            if (victim == lanes.size() || (victim == i && !others)) {
                outcome.blocked.push_back(lanes[i].lane);
                break;
            }
            hooks.preempt(lanes[victim].lane);
            gone[victim] = true;
            outcome.preempted.push_back(lanes[victim].lane);
            if (victim == i) { break; }
        }
    }
    return outcome;
}

/// Where a preempted request goes back into the waiting queue: in arrival order, so it resumes
/// before every request that arrived after it -- the front, when admission is first come, first
/// served -- and after an older preempted request.
[[nodiscard]] inline std::size_t kv_requeue_position(std::span<const std::uint64_t> waiting_orders,
                                                     std::uint64_t order) noexcept {
    for (std::size_t i = 0; i < waiting_orders.size(); ++i) {
        if (waiting_orders[i] > order) { return i; }
    }
    return waiting_orders.size();
}

/// SUROGATE_SERVE_KV_FORCE_PREEMPT=N[xK] (tests): preempt every recomputable decoding request
/// once it has generated N tokens, then every N more, K times in all (default once).
struct KvForcedPreemption {
    std::uint32_t every = 0;
    std::uint32_t times = 1;

    [[nodiscard]] bool due(std::size_t generated, std::uint32_t preempted) const noexcept {
        return every != 0 && preempted < times &&
               generated >= static_cast<std::size_t>(every) * (preempted + 1U);
    }
};

[[nodiscard]] inline KvForcedPreemption parse_kv_forced_preemption(const char* raw) noexcept {
    KvForcedPreemption out;
    if (raw == nullptr || *raw == '\0') { return out; }
    char* end = nullptr;
    const unsigned long every = std::strtoul(raw, &end, 10);
    out.every = static_cast<std::uint32_t>(std::min<unsigned long>(every, 0xFFFFFFFFUL));
    if (end != nullptr && (*end == 'x' || *end == 'X')) {
        const unsigned long times = std::strtoul(end + 1, nullptr, 10);
        out.times = static_cast<std::uint32_t>(std::max<unsigned long>(1, std::min<unsigned long>(times, 1U << 20)));
    }
    return out;
}

} // namespace sinfer::runtime
