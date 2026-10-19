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
