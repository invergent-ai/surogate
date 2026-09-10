#pragma once

// The working-set manager for multi-model overcommit: more models than fit,
// juggled by sleeping and waking whole engines.
//
// One constraint governs everything: the sum of awake models' footprints must
// stay under the device budget. A request for a sleeping model queues inside
// ensure_awake() rather than failing; the planner frees the deficit by
// sleeping idle models -- least recently used first, minimal covering subset,
// never one inside its keep-warm window -- then wakes the target. Transitions
// are whole-engine sleep/wake at ~53 GB/s of PCIe copy, so the working set
// changes in tenths of a second, and models that fit together stay awake
// together and share compute on their own streams.
//
// Everything is serialized under one mutex; waiters park on a condition
// variable and are woken after every transition. Deliberately not here in v1:
// preemptive eviction of busy models (the engine supports it -- see
// SUROGATE_SLEEP_PREEMPT -- but idle-only eviction plus queueing covers the
// common patterns), priorities, and predictive pre-wake.

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace sinfer::serve {

class GenerationService;

class ModelScheduler {
public:
    struct Entry {
        std::string name;
        GenerationService* service = nullptr;
        int priority               = 1; ///< 0 low, 1 normal, 2 high
    };

    /// `device_budgets` records the VRAM on each GPU the resident set may use (measured free at
    /// startup plus what the already-awake models occupy). The constructor also
    /// pre-pins every model's host backup -- first-time pinning runs at
    /// ~2 GiB/s and must never land on some other model's requester -- and
    /// starts the re-wake tick that resumes preempted in-flight work.
    ModelScheduler(std::vector<Entry> entries, std::map<int, std::size_t> device_budgets);
    ~ModelScheduler();

    /// Blocks until `service` is awake and fits, waking and evicting as
    /// needed. Throws on timeout. Also stamps the model's last-use time, so
    /// callers invoke it on every routed request.
    void ensure_awake(GenerationService* service);

    /// Idle-eviction support during startup construction: sleep every awake,
    /// idle model to make room (used when constructing a later engine OOMs).
    void sleep_all_idle();

private:
    struct State {
        Entry entry;
        std::chrono::steady_clock::time_point last_used;
        std::chrono::steady_clock::time_point woke_at;
    };
    bool try_make_room_locked(State& target, bool allow_preempt);
    void tick_loop();

    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<State> models_;
    std::map<int, std::size_t> device_budgets_;
    std::chrono::milliseconds keep_warm_{3000};
    std::chrono::milliseconds preempt_after_{5000};
    std::chrono::milliseconds min_dwell_{2000};
    std::chrono::seconds wait_timeout_{180};
    bool stopping_ = false;
    std::thread ticker_;
};

} // namespace sinfer::serve
