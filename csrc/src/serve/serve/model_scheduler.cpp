#include "serve/model_scheduler.h"
#include "serve/model_device_budget.h"

#include "serve/console_log.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace sinfer::serve {

using Clock = std::chrono::steady_clock;

namespace {
void check_wake_control(const PreparationControl& control) {
    if (control.cancellation.requested()) {
        throw RequestError(RequestErrorKind::Cancelled, "request cancelled while waiting for model wake");
    }
    if (control.deadline != Clock::time_point{} && Clock::now() >= control.deadline) {
        throw RequestError(RequestErrorKind::QueueTimeout, "request expired while waiting for model wake");
    }
}

std::chrono::milliseconds env_ms(const char* name, std::chrono::milliseconds fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) { return fallback; }
    return std::chrono::milliseconds(std::atol(raw));
}
} // namespace

ModelScheduler::ModelScheduler(std::vector<Entry> entries, std::map<int, std::size_t> device_budgets)
    : device_budgets_(std::move(device_budgets)) {
    keep_warm_     = env_ms("SUROGATE_MM_KEEPWARM_MS", keep_warm_);
    preempt_after_ = env_ms("SUROGATE_MM_PREEMPT_AFTER_MS", preempt_after_);
    min_dwell_     = env_ms("SUROGATE_MM_MIN_DWELL_MS", min_dwell_);
    const auto now = Clock::now();
    for (auto& entry : entries) {
        // Pre-pin every model's host backup now: the parked ones already have
        // theirs (construction slept them); the awake ones must not pay the
        // one-time pinning on the first eviction, which would stall whichever
        // OTHER model's requester triggered it.
        entry.service->prepare_sleep_backup();
        models_.push_back(State{std::move(entry), now, now});
    }
    ticker_ = std::thread([this] { tick_loop(); });
}

struct ModelScheduler::Waiter {
    ScheduledModel* service;
    Clock::time_point arrived;
    Clock::time_point deadline;
    std::atomic<bool> abandoned{false};
    bool done = false; // guarded by wait_mutex_
    std::exception_ptr error;
};

ModelScheduler::~ModelScheduler() {
    stopping_.store(true);
    cv_.notify_all();
    if (ticker_.joinable()) { ticker_.join(); }
}

void ModelScheduler::ensure_awake(ScheduledModel* service, const PreparationControl& control) {
    check_wake_control(control);
    // The common awake path must not pay a worker handoff on every request.
    if (std::unique_lock lock(mutex_, std::try_to_lock); lock.owns_lock() && !stopping_.load()) {
        const auto target = std::find_if(models_.begin(), models_.end(), [&](const State& state) {
            return state.entry.service == service;
        });
        if (target == models_.end()) { return; }
        if (!service->is_sleeping()) {
            target->last_used = Clock::now();
            return;
        }
    }
    auto waiter = std::make_shared<Waiter>();
    waiter->service = service;
    waiter->arrived = Clock::now();
    waiter->deadline = control.deadline;
    std::unique_lock lock(wait_mutex_);
    waiters_.push_back(waiter);
    cv_.notify_all();
    const auto remove = [&] {
        waiter->abandoned.store(true);
        std::erase(waiters_, waiter);
        cv_.notify_all();
    };
    try {
        while (true) {
            check_wake_control(control);
            if (stopping_.load()) {
                throw RequestError(RequestErrorKind::Unavailable, "model scheduler is stopping");
            }
            if (waiter->done) {
                if (waiter->error) { std::rethrow_exception(waiter->error); }
                break;
            }
            auto next = Clock::now() + std::chrono::milliseconds(10);
            if (control.deadline != Clock::time_point{}) { next = std::min(next, control.deadline); }
            cv_.wait_until(lock, next);
        }
    } catch (...) {
        remove();
        throw;
    }
    remove();
}

bool ModelScheduler::try_make_room_locked(State& target, bool allow_preempt,
                                           const PreparationControl& control) {
    check_wake_control(control);
    const auto target_devices = target.entry.service->devices();
    const auto short_devices = [&] {
        DeviceBytes needed, resident;
        for (int device : target_devices) {
            needed[device] = target.entry.service->resident_bytes(device);
            for (const auto& state : models_) {
                if (!state.entry.service->is_sleeping() && state.entry.service != target.entry.service) {
                    resident[device] += state.entry.service->resident_bytes(device);
                }
            }
        }
        return model_memory_shortfall(needed, resident, device_budgets_);
    };
    const auto occupies = [](const State& state, const std::vector<int>& devices) {
        return std::any_of(devices.begin(), devices.end(), [&](int device) {
            return state.entry.service->resident_bytes(device) != 0;
        });
    };
    // The cheap move first: an idle awake model's KV is mostly prefix cache, and an elastic
    // pool gives those granules back without the model leaving the device. Shrink every idle
    // neighbour once and re-check before any model is put to sleep.
    if (const auto shortfall = short_devices(); !shortfall.empty()) {
        for (auto& state : models_) {
            if (state.entry.service == target.entry.service || state.entry.service->is_sleeping() ||
                state.entry.service->active_requests() != 0 || !occupies(state, shortfall)) {
                continue;
            }
            check_wake_control(control);
            state.entry.service->shrink_kv();
        }
    }
    for (auto shortfall = short_devices(); !shortfall.empty(); shortfall = short_devices()) {
        State* victim  = nullptr;
        bool busy_pick = false;
        const auto now = Clock::now();
        for (auto& state : models_) {
            if (state.entry.service == target.entry.service) { continue; }
            if (state.entry.service->is_sleeping() || !occupies(state, shortfall)) { continue; }
            const bool busy = state.entry.service->active_requests() != 0;
            if (busy && !allow_preempt) { continue; }
            // The preemption fence: a busy model yields only to an equal or
            // higher priority target. A lower-priority requester waits for the
            // natural drain instead -- that is what priority means here.
            if (busy && state.entry.priority > target.entry.priority) { continue; }
            // Idle models of any tier stay evictable (nothing pins VRAM by
            // being idle), but higher tiers keep their warmth longer.
            const auto warmth = state.entry.priority == 2   ? keep_warm_ * 2
                                : state.entry.priority == 0 ? keep_warm_ / 2
                                                            : keep_warm_;
            if (!busy && now - state.last_used < warmth) { continue; }
            if (busy && now - state.woke_at < min_dwell_) { continue; }
            // Idle victims strictly before busy ones; within a class, lower
            // priority first, then LRU.
            const bool better =
                victim == nullptr || (busy_pick && !busy) ||
                (busy_pick == busy &&
                 (state.entry.priority < victim->entry.priority ||
                  (state.entry.priority == victim->entry.priority &&
                   state.last_used < victim->last_used)));
            if (better) {
                victim    = &state;
                busy_pick = busy;
            }
        }
        if (victim == nullptr) { return false; }
        if (busy_pick) {
            write_console_log(ConsoleLogLevel::Info,
                              "scheduler: preempting busy model '" + victim->entry.name +
                                  "' at a round boundary; its generations resume after re-wake");
        }
        check_wake_control(control);
        victim->entry.service->sleep(/*preempt=*/busy_pick);
    }
    return true;
}

void ModelScheduler::tick_loop() {
    auto next_resume = Clock::now() + std::chrono::milliseconds(500);
    while (!stopping_.load()) {
        std::vector<std::shared_ptr<Waiter>> pending;
        {
            std::unique_lock lock(wait_mutex_);
            // New requests notify immediately; blocked requests are reconsidered every
            // 10 ms without tying their HTTP worker to an eviction or restore operation.
            cv_.wait_for(lock, waiters_.empty() ? std::chrono::milliseconds(500)
                                               : std::chrono::milliseconds(10));
            if (stopping_.load()) { return; }
            pending = waiters_;
        }
        std::lock_guard model_lock(mutex_);
        for (const auto& waiter : pending) {
            if (waiter->abandoned.load() || stopping_.load()) { continue; }
            const PreparationControl control{
                .deadline = waiter->deadline,
                // Never retain a callback referring to an HTTP request on this worker.
                .cancellation = CancellationView([this, waiter] {
                    return stopping_.load() || waiter->abandoned.load();
                }),
            };
            std::exception_ptr error;
            bool done = false;
            try {
                check_wake_control(control);
                const auto target = std::find_if(models_.begin(), models_.end(), [&](const State& state) {
                    return state.entry.service == waiter->service;
                });
                if (target == models_.end()) {
                    done = true;
                } else {
                    target->last_used = Clock::now();
                    if (!target->entry.service->is_sleeping()) {
                        done = true;
                    } else if (try_make_room_locked(*target,
                                   Clock::now() - waiter->arrived >= preempt_after_, control)) {
                        check_wake_control(control);
                        target->entry.service->wake_up();
                        target->woke_at = target->last_used = Clock::now();
                        done = true;
                    }
                }
            } catch (...) {
                done = true;
                error = std::current_exception();
            }
            if (done) {
                std::lock_guard lock(wait_mutex_);
                waiter->done = true;
                waiter->error = error;
                std::erase(waiters_, waiter);
                cv_.notify_all();
            }
        }
        if (Clock::now() < next_resume || stopping_.load()) { continue; }
        next_resume = Clock::now() + std::chrono::milliseconds(500);
        const PreparationControl resume_control{
            .cancellation = CancellationView([this] { return stopping_.load(); }),
        };
        // Preempted generations never re-enter HTTP. Initial wake waiters must not
        // count here: a cancelled request must not be revived by the resume ticker.
        for (auto& state : models_) {
            ScheduledModel* service = state.entry.service;
            if (!service->is_sleeping() || service->resumable_requests() == 0) { continue; }
            try {
                if (try_make_room_locked(state, false, resume_control)) {
                    check_wake_control(resume_control);
                    write_console_log(ConsoleLogLevel::Info,
                                      "scheduler: re-waking '" + state.entry.name +
                                          "' to finish its parked generations");
                    service->wake_up();
                    state.woke_at = Clock::now();
                }
            } catch (const std::exception& error) {
                write_console_log(ConsoleLogLevel::Error,
                    "scheduler: could not wake '" + state.entry.name + "': " + error.what());
            }
        }
    }
}

void ModelScheduler::sleep_all_idle() {
    const std::lock_guard<std::mutex> lock(mutex_);
    for (auto& state : models_) {
        if (!state.entry.service->is_sleeping() &&
            state.entry.service->active_requests() == 0) {
            state.entry.service->sleep();
        }
    }
    cv_.notify_all();
}

} // namespace sinfer::serve
