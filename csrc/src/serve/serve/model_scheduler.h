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
// A worker serializes memory transitions, including priority-aware preemption.
// Request waiters use a separate lock so deadlines and disconnects remain
// observable while a GPU transfer is in progress.

#include "api/types.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <memory>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace sinfer::serve {

// Residency operations used by the planner, independent of HTTP and generation.
class ScheduledModel {
public:
    virtual ~ScheduledModel() = default;
    virtual void prepare_sleep_backup() = 0;
    virtual void sleep(bool preempt = false) = 0;
    virtual void wake_up() = 0;
    virtual bool is_sleeping() const = 0;
    virtual void shrink_kv() = 0;
    virtual std::size_t active_requests() const = 0;
    // Work past initial wake admission, including preempted generations.
    virtual std::size_t resumable_requests() const = 0;
    virtual std::size_t resident_bytes(int device = -1) const = 0;
    virtual std::vector<int> devices() const = 0;
};

class ModelScheduler {
public:
    struct Entry {
        std::string name;
        ScheduledModel* service = nullptr;
        int priority               = 1; ///< 0 low, 1 normal, 2 high
    };

    /// `device_budgets` records the VRAM on each GPU the resident set may use (measured free at
    /// startup plus what the already-awake models occupy). The constructor also
    /// pre-pins every model's host backup -- first-time pinning runs at
    /// ~2 GiB/s and must never land on some other model's requester -- and
    /// starts the re-wake tick that resumes preempted in-flight work.
    ModelScheduler(std::vector<Entry> entries, std::map<int, std::size_t> device_budgets);
    ~ModelScheduler();

    /// Blocks until `service` is awake and fits. Uses the caller's admission
    /// deadline; cancellation and timeout release the waiter even during a copy.
    /// An already-started transfer finishes safely on the worker. Also stamps
    /// the model's last-use time, so callers invoke it on every routed request.
    void ensure_awake(ScheduledModel* service, const PreparationControl& control);

    /// Idle-eviction support during startup construction: sleep every awake,
    /// idle model to make room (used when constructing a later engine OOMs).
    void sleep_all_idle();

private:
    struct State {
        Entry entry;
        std::chrono::steady_clock::time_point last_used;
        std::chrono::steady_clock::time_point woke_at;
    };
    struct Waiter;
    bool try_make_room_locked(State& target, bool allow_preempt,
                              const PreparationControl& control);
    void tick_loop();

    // Never held by a request waiter: a transfer may take longer than its deadline.
    std::mutex mutex_;
    std::mutex wait_mutex_;
    std::condition_variable cv_;
    std::vector<std::shared_ptr<Waiter>> waiters_;
    std::vector<State> models_;
    std::map<int, std::size_t> device_budgets_;
    std::chrono::milliseconds keep_warm_{3000};
    std::chrono::milliseconds preempt_after_{5000};
    std::chrono::milliseconds min_dwell_{2000};
    std::atomic<bool> stopping_{false};
    std::thread ticker_;
};

} // namespace sinfer::serve
