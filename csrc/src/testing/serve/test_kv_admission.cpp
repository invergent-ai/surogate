// --kv-admission ondemand (DESIGN-kv-admission.md): the growth / preemption policy, the queue
// position of a preempted request, a scheduler simulation of many long prompts with no
// max_tokens, and the KV pool's entitlement accounting through growth, preemption and shared
// prefix pages. CPU only: the pool runs on host memory and is never bound to a block-table row,
// so nothing here reaches the device (run it with CUDA_VISIBLE_DEVICES= to be sure).

#ifndef KV_ADMISSION_POLICY_ONLY // (a host-only build of the policy checks, for development)
#include "core/paged_kv_cache.h"
#endif
#include "runtime/engine/kv_admission_policy.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

using sinfer::runtime::KvGrowthHooks;
using sinfer::runtime::KvGrowthLane;
using sinfer::runtime::KvGrowthOutcome;

/// A pool of pages and the lanes holding them, for the policy alone.
struct FakePool {
    std::uint32_t capacity = 0;
    std::uint32_t idle     = 0; // pages held by idle cache (retained lanes, shared prefixes)
    std::map<std::uint32_t, std::uint32_t> held;
    std::map<std::uint32_t, std::uint32_t> wants; // pages each lane needs to grow by
    std::vector<std::uint32_t> preempted;
    std::uint32_t idle_releases = 0;

    [[nodiscard]] std::uint32_t used() const {
        std::uint32_t total = idle;
        for (const auto& [lane, pages] : held) { total += pages; }
        return total;
    }
    KvGrowthHooks hooks() {
        return KvGrowthHooks{
            .grow =
                [this](std::uint32_t lane) {
                    const std::uint32_t need = wants[lane];
                    if (used() + need > capacity) { return false; }
                    held[lane] += need;
                    wants[lane] = 0;
                    return true;
                },
            .release_idle =
                [this] {
                    if (idle == 0) { return false; }
                    idle = idle > 4 ? idle - 4 : 0; // one retained lane at a time
                    ++idle_releases;
                    return true;
                },
            .preempt =
                [this](std::uint32_t lane) {
                    held.erase(lane);
                    wants.erase(lane);
                    preempted.push_back(lane);
                },
        };
    }
};

void test_growth_policy() {
    // Room for everyone: nothing is preempted.
    {
        FakePool pool{.capacity = 100};
        pool.held  = {{0, 10}, {1, 10}};
        pool.wants = {{0, 1}, {1, 1}};
        const auto outcome = sinfer::runtime::ensure_kv_growth(
            {{0, 1, true, true}, {1, 2, true, true}}, pool.hooks());
        check(outcome.preempted.empty() && outcome.blocked.empty() && pool.used() == 22,
              "growth that fits preempts nothing");
    }
    // Short: idle cache goes before any running request.
    {
        FakePool pool{.capacity = 30, .idle = 8};
        pool.held  = {{0, 10}, {1, 11}};
        pool.wants = {{0, 4}, {1, 1}};
        const auto outcome = sinfer::runtime::ensure_kv_growth(
            {{0, 1, true, true}, {1, 2, true, true}}, pool.hooks());
        check(outcome.preempted.empty() && pool.idle_releases >= 1,
              "idle KV is released before a running request is preempted");
    }
    // Short with no idle cache: the request that arrived last goes, not the one growing.
    {
        FakePool pool{.capacity = 40};
        pool.held  = {{0, 15}, {1, 12}, {2, 13}};
        pool.wants = {{0, 2}, {1, 0}, {2, 0}};
        // Lane numbers do not follow arrival: lane 1 arrived last.
        const auto outcome = sinfer::runtime::ensure_kv_growth(
            {{0, 10, true, true}, {1, 30, true, false}, {2, 20, true, true}}, pool.hooks());
        check(outcome.preempted.size() == 1 && outcome.preempted[0] == 1,
              "the newest arrival is preempted first, whatever its lane");
        check(pool.held.count(0) == 1 && pool.held[0] == 17, "the oldest request grew");
    }
    // The growing lane is itself the newest: it preempts itself, the older ones keep running.
    {
        FakePool pool{.capacity = 30};
        pool.held  = {{0, 15}, {1, 15}};
        pool.wants = {{0, 0}, {1, 1}};
        const auto outcome = sinfer::runtime::ensure_kv_growth(
            {{0, 1, true, true}, {1, 2, true, true}}, pool.hooks());
        check(outcome.preempted.size() == 1 && outcome.preempted[0] == 1 && pool.held.count(0) == 1,
              "a lane that arrived last and cannot grow preempts itself");
        check(outcome.blocked.empty(), "self-preemption is not a block");
    }
    // An older request is never preempted for a newer one, even when that would free the most.
    {
        FakePool pool{.capacity = 50};
        pool.held  = {{0, 40}, {1, 10}};
        pool.wants = {{0, 0}, {1, 5}};
        const auto outcome = sinfer::runtime::ensure_kv_growth(
            {{0, 1, true, true}, {1, 2, true, true}}, pool.hooks());
        check(outcome.preempted == std::vector<std::uint32_t>{1}, "an older request is never a victim of a newer one");
    }
    // A newer request admitted with its whole reservation is skipped as a victim.
    {
        FakePool pool{.capacity = 30};
        pool.held  = {{0, 10}, {1, 10}, {2, 10}};
        pool.wants = {{0, 2}};
        const auto outcome = sinfer::runtime::ensure_kv_growth(
            {{0, 1, true, true}, {1, 2, true, true}, {2, 3, false, true}}, pool.hooks());
        check(outcome.preempted == std::vector<std::uint32_t>{1},
              "a request that cannot be recomputed is never preempted");
    }
    // Alone and still short (pages no lane owns): blocked, for the caller to fail or wait.
    {
        FakePool pool{.capacity = 20, .idle = 0};
        pool.held  = {{0, 20}};
        pool.wants = {{0, 1}};
        const auto outcome =
            sinfer::runtime::ensure_kv_growth({{0, 1, true, true}}, pool.hooks());
        check(outcome.preempted.empty() && outcome.blocked == std::vector<std::uint32_t>{0},
              "a lone request that cannot grow is reported blocked, not preempted");
    }
    // Behind older requests that cannot be preempted, the newest preempts itself and waits.
    {
        FakePool pool{.capacity = 20};
        pool.held  = {{0, 10}, {1, 10}};
        pool.wants = {{1, 1}};
        const auto outcome = sinfer::runtime::ensure_kv_growth(
            {{0, 1, false, true}, {1, 2, true, true}}, pool.hooks());
        check(outcome.preempted == std::vector<std::uint32_t>{1} && outcome.blocked.empty(),
              "the newest request gives way to older reserved ones");
    }
    // A staged prompt (not decoding) that arrived last is a victim too.
    {
        FakePool pool{.capacity = 30};
        pool.held  = {{0, 10}, {5, 20}};
        pool.wants = {{0, 3}};
        const auto outcome = sinfer::runtime::ensure_kv_growth(
            {{0, 1, true, true}, {5, 7, true, false}}, pool.hooks());
        check(outcome.preempted == std::vector<std::uint32_t>{5}, "a newer staged prompt is preempted");
    }
}

void test_requeue_and_forced() {
    const std::vector<std::uint64_t> waiting{12, 15, 40};
    check(sinfer::runtime::kv_requeue_position(waiting, 7) == 0, "a preempted request goes to the front");
    check(sinfer::runtime::kv_requeue_position(waiting, 13) == 1,
          "after an older preempted request, before newer ones");
    check(sinfer::runtime::kv_requeue_position({}, 3) == 0, "empty queue");
    const auto once = sinfer::runtime::parse_kv_forced_preemption("64");
    check(once.every == 64 && once.times == 1 && !once.due(63, 0) && once.due(64, 0) && !once.due(500, 1),
          "forced preemption N fires once at N tokens");
    const auto thrice = sinfer::runtime::parse_kv_forced_preemption("16x3");
    check(thrice.due(32, 1) && !thrice.due(31, 1) && thrice.due(48, 2) && !thrice.due(1000, 3),
          "forced preemption NxK fires every N tokens, K times");
    check(sinfer::runtime::parse_kv_forced_preemption(nullptr).every == 0, "unset is off");
}

/// A whole engine's scheduling in pages and tokens: first come, first served admission of prompt +
/// headroom (with a watermark while others run), growth by rounds, preemption by recompute.
struct SimRequest {
    std::uint64_t id = 0;
    std::uint32_t prompt = 0;
    std::uint32_t output = 0;    // what it will generate before it stops
    std::uint32_t generated = 0;
    std::uint32_t preemptions = 0;
    std::uint32_t entitled = 0;  // pages
    std::uint32_t computed = 0;  // tokens of KV it holds (prefill progress, then decode)
    bool decoding = false;
    std::uint32_t limit_pages = 0;
};

struct SimShape {
    std::uint32_t prompt_min = 0, prompt_span = 1, output_min = 0, output_span = 1;
};

void test_simulation(std::uint32_t seed, std::uint32_t requests, std::uint32_t lanes,
                     std::uint32_t capacity_tokens, SimShape shape) {
    constexpr std::uint32_t kPage = 64, kContext = 262144, kHeadroom = 256, kReach = 9 + 1;
    const std::uint32_t capacity = capacity_tokens / kPage;
    const std::uint32_t watermark = capacity / 100;
    std::mt19937 rng(seed);
    auto pages_for = [&](std::uint64_t tokens) { return static_cast<std::uint32_t>((tokens + kPage - 1) / kPage); };
    std::deque<std::shared_ptr<SimRequest>> waiting;
    for (std::uint32_t i = 0; i < requests; ++i) {
        auto request = std::make_shared<SimRequest>();
        request->id = i + 1;
        request->prompt = shape.prompt_min + rng() % shape.prompt_span;
        // No max_tokens: the reservation is the whole context; the answer stops on its own.
        request->output = shape.output_min + rng() % shape.output_span;
        request->limit_pages = pages_for(kContext);
        waiting.push_back(request);
    }
    std::vector<std::shared_ptr<SimRequest>> slots(lanes);
    std::vector<std::uint64_t> first_admissions;
    std::uint32_t entitled = 0, completed = 0, preemptions = 0;
    std::uint64_t steps = 0;
    bool fcfs_violated = false, older_preempted = false, over_capacity = false;
    const auto running = [&] {
        for (const auto& s : slots) { if (s) { return true; } }
        return false;
    };
    while (completed < requests && steps < 2000000) {
        ++steps;
        // Admission: strict FCFS, the head waits for its prompt + headroom (+ watermark).
        while (!waiting.empty()) {
            auto& head = waiting.front();
            auto lane = std::find(slots.begin(), slots.end(), nullptr);
            if (lane == slots.end()) { break; }
            const std::uint32_t context = head->prompt + head->generated;
            const std::uint32_t want = std::min(head->limit_pages, pages_for(context + kHeadroom));
            const std::uint32_t reserve = running() ? watermark : 0;
            if (entitled + want + reserve > capacity) { break; }
            if (head->preemptions == 0) {
                if (!first_admissions.empty() && head->id < first_admissions.back()) { fcfs_violated = true; }
                first_admissions.push_back(head->id);
            }
            head->entitled = want;
            head->computed = 0;
            head->decoding = false;
            entitled += want;
            *lane = head;
            waiting.pop_front();
        }
        // Growth before the round, through the real policy.
        std::vector<KvGrowthLane> view;
        for (std::uint32_t lane = 0; lane < lanes; ++lane) {
            if (slots[lane]) {
                view.push_back({lane, slots[lane]->id, true, slots[lane]->decoding});
            }
        }
        const KvGrowthHooks hooks{
            .grow =
                [&](std::uint32_t lane) {
                    SimRequest& r = *slots[lane];
                    const std::uint32_t want = std::max(
                        r.entitled, std::min(r.limit_pages, pages_for(r.computed + kReach)));
                    if (entitled - r.entitled + want > capacity) { return false; }
                    entitled = entitled - r.entitled + want;
                    r.entitled = want;
                    return true;
                },
            .release_idle = [] { return false; },
            .preempt =
                [&](std::uint32_t lane) {
                    auto victim = slots[lane];
                    for (const auto& other : slots) {
                        if (other && other->id > victim->id) { older_preempted = true; }
                    }
                    entitled -= victim->entitled;
                    victim->entitled = 0;
                    ++victim->preemptions;
                    ++preemptions;
                    slots[lane].reset();
                    std::vector<std::uint64_t> ids;
                    for (const auto& w : waiting) { ids.push_back(w->id); }
                    waiting.insert(waiting.begin() + static_cast<std::ptrdiff_t>(
                                                         sinfer::runtime::kv_requeue_position(ids, victim->id)),
                                   victim);
                },
        };
        const KvGrowthOutcome outcome = sinfer::runtime::ensure_kv_growth(view, hooks);
        if (!outcome.blocked.empty()) {
            check(false, "a request was blocked although every admitted one fits the pool alone");
            return;
        }
        if (entitled > capacity) { over_capacity = true; }
        // One round: prefill advances by 8k tokens a lane, decode by one token.
        for (std::uint32_t lane = 0; lane < lanes; ++lane) {
            if (!slots[lane]) { continue; }
            SimRequest& r = *slots[lane];
            const std::uint32_t context = r.prompt + r.generated;
            if (!r.decoding) {
                r.computed = std::min(context, r.computed + 8192);
                if (r.computed == context) { r.decoding = true; }
                continue;
            }
            if (pages_for(r.computed + 1) > r.entitled) {
                check(false, "a decode step wrote past its entitlement");
                return;
            }
            ++r.computed;
            ++r.generated;
            if (r.generated >= r.output) {
                entitled -= r.entitled;
                slots[lane].reset();
                ++completed;
            }
        }
    }
    const std::string label = "simulation seed " + std::to_string(seed) + ": ";
    check(completed == requests, label + "every request completes (" + std::to_string(completed) + "/" +
                                     std::to_string(requests) + " in " + std::to_string(steps) + " rounds)");
    check(!fcfs_violated, label + "first admissions are in arrival order");
    check(!older_preempted, label + "no request is preempted while a newer one runs");
    check(!over_capacity, label + "entitlement never exceeds the pool");
    check(entitled == 0, label + "every page comes back");
    std::cout << label << completed << " requests, " << steps << " rounds, " << preemptions
              << " preemptions\n";
}

#ifndef KV_ADMISSION_POLICY_ONLY
// ---- The KV pool itself, on host memory -------------------------------------------------------

struct HostPool {
    sinfer::PagedKVPoolLayout layout;
    std::unique_ptr<std::byte[], decltype(&std::free)> memory{nullptr, &std::free};
    std::unique_ptr<sinfer::PagedKVPool> pool;
};

HostPool make_pool(std::uint32_t pages) {
    HostPool out;
    sinfer::LayoutBuilder builder;
    out.layout = sinfer::plan_paged_kv_pool(
        builder, {.page_group_count      = pages,
                  .logical_page_capacity = pages,
                  .table_rows            = 4,
                  .planes = {sinfer::PagedKVPlaneSpec{.dtype = sinfer::DType::BF16,
                                                       .leading_extent = 1,
                                                       .head_extent    = 8}}});
    const std::size_t bytes = builder.finish(256);
    out.memory.reset(static_cast<std::byte*>(std::aligned_alloc(256, (bytes + 255) / 256 * 256)));
    out.pool = std::make_unique<sinfer::PagedKVPool>(sinfer::DeviceSpan{out.memory.get(), bytes}, out.layout);
    return out;
}

bool grow(sinfer::PagedKVAllocation& allocation, std::uint32_t pages) {
    sinfer::PagedKVResize change{.allocation       = &allocation,
                                 .mapped_pages     = allocation.mapped_page_count(),
                                 .page_entitlement = pages};
    try {
        sinfer::resize_paged_kv_bundle(std::span<const sinfer::PagedKVResize>(&change, 1));
        return true;
    } catch (const std::bad_alloc&) {
        return false;
    }
}

void test_pool_accounting() {
    // Growth by rounds within the pool, refusal at the edge, and pages back on preemption.
    {
        HostPool host = make_pool(32);
        auto& pool = *host.pool;
        auto a = pool.reserve(10); // prompt + headroom
        auto b = pool.reserve(10);
        a.materialize_pages(9);
        b.materialize_pages(9);
        check(pool.entitled_pages() == 20 && pool.mapped_pages() == 18, "admission entitles prompt pages only");
        check(grow(a, 16) && pool.entitled_pages() == 26, "a sequence grows its entitlement");
        a.materialize_pages(16);
        check(!grow(b, 17) && pool.entitled_pages() == 26 && b.page_entitlement() == 10,
              "a growth the pool cannot hold is refused and changes nothing");
        b.release(); // preempted: every page it owned comes back
        check(pool.entitled_pages() == 16 && pool.mapped_pages() == 16, "a preempted sequence returns its pages");
        check(grow(a, 32) && pool.entitled_pages() == 32, "the survivor grows into them");
        a.materialize_pages(32);
        a.cancel_unmapped_entitlement();
        a.trim_pages(20);
        a.cancel_unmapped_entitlement();
        check(pool.entitled_pages() == 20, "finishing drops entitlement past the mapped pages");
    }
    // Shared prefix pages: a fork borrows them, grows only what it owns, and preempting it returns
    // only its own pages (PR #322's accounting).
    {
        HostPool host = make_pool(64);
        auto& pool = *host.pool;
        auto owner = pool.reserve(12);
        owner.materialize_pages(10);
        auto prefix = pool.share_prefix(owner, 8); // 512 tokens shared at a page boundary
        check(owner.borrowed_pages() == 8 && owner.owned_entitlement() == 4 && pool.entitled_pages() == 12,
              "sharing moves ownership, not totals");
        auto fork = pool.fork_prefix(prefix, 8 * 64, 11, nullptr); // page-aligned: no device copy
        check(fork.borrowed_pages() == 8 && fork.owned_entitlement() == 3 && pool.entitled_pages() == 15,
              "a fork is entitled only its own pages");
        check(grow(fork, 20) && fork.owned_entitlement() == 12 && pool.entitled_pages() == 24,
              "a fork's growth counts only owned pages");
        fork.materialize_pages(20);
        const std::uint32_t mapped_before = pool.mapped_pages();
        fork.release(); // preempted
        check(pool.entitled_pages() == 12 && pool.mapped_pages() == mapped_before - 12,
              "preempting a fork returns its own pages, the borrowed ones stay");
        check(prefix->mapped_page_count() == 8, "the shared prefix is intact");
        owner.release();
        check(pool.entitled_pages() == 8, "the shared prefix keeps its pages while held");
        prefix.reset();
        check(pool.entitled_pages() == 0 && pool.mapped_pages() == 0 && pool.free_pages() == 64,
              "the last holder returns the shared pages");
    }
}

#endif // KV_ADMISSION_POLICY_ONLY

} // namespace

int main() {
    test_growth_policy();
    test_requeue_and_forced();
    // 483k-token RTX 5090 replica, 16 lanes, BFCL-like short requests at the agentic default.
    test_simulation(1, 400, 16, 483328, {500, 4000, 20, 600});
    // Many 100k+ prompts, no max_tokens, on the same pool and on a small one.
    test_simulation(2, 120, 16, 483328, {100000, 120000, 200, 4000});
    test_simulation(3, 60, 8, 262144 + 4096, {100000, 120000, 200, 4000});
    test_simulation(4, 200, 64, 483328, {100000, 120000, 200, 4000});
    // Long answers on a full pool: growth runs it out over and over (thrash), every request
    // still finishes, in order, and no older request gives way to a newer one.
    test_simulation(5, 80, 16, 262144 + 4096, {20000, 60000, 10000, 60000});
    test_simulation(6, 300, 64, 483328, {1000, 30000, 1000, 40000});
#ifndef KV_ADMISSION_POLICY_ONLY
    test_pool_accounting();
#endif
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "kv admission: growth, preemption, requeue, simulation and pool accounting pass\n";
    return 0;
}
