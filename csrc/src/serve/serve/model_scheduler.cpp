#include "serve/model_scheduler.h"

#include "serve/console_log.h"
#include "serve/generation_service.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace sinfer::serve {

using Clock = std::chrono::steady_clock;

namespace {
std::chrono::milliseconds env_ms(const char* name, std::chrono::milliseconds fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) { return fallback; }
    return std::chrono::milliseconds(std::atol(raw));
}
} // namespace

ModelScheduler::ModelScheduler(std::vector<Entry> entries, std::size_t budget_bytes)
    : budget_bytes_(budget_bytes) {
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

ModelScheduler::~ModelScheduler() {
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    cv_.notify_all();
    if (ticker_.joinable()) { ticker_.join(); }
}

void ModelScheduler::ensure_awake(GenerationService* service) {
    std::unique_lock<std::mutex> lock(mutex_);
    State* target = nullptr;
    for (auto& state : models_) {
        if (state.entry.service == service) { target = &state; break; }
    }
    if (target == nullptr) { return; } // unmanaged service: nothing to do
    target->last_used = Clock::now();
    if (!service->is_sleeping()) { return; }

    const auto arrived  = Clock::now();
    const auto deadline = arrived + wait_timeout_;
    while (true) {
        if (!service->is_sleeping()) { return; }
        // Idle victims first; after the grace period, a busy model may be
        // preempted at a round boundary -- its generations park and resume.
        const bool preempt_ok = Clock::now() - arrived >= preempt_after_;
        if (try_make_room_locked(*target, preempt_ok)) {
            service->wake_up();
            target->woke_at   = Clock::now();
            target->last_used = target->woke_at;
            cv_.notify_all();
            return;
        }
        if (cv_.wait_until(lock, std::min(deadline, Clock::now() + preempt_after_)) ==
                std::cv_status::timeout &&
            Clock::now() > deadline) {
            throw std::runtime_error(
                "the model is asleep and no room could be freed within the queue timeout -- "
                "other models are busy; retry, or lower their load");
        }
    }
}

bool ModelScheduler::try_make_room_locked(State& target, bool allow_preempt) {
    const std::size_t needed = target.entry.service->resident_bytes();
    auto awake_bytes         = [&] {
        std::size_t total = 0;
        for (const auto& state : models_) {
            if (!state.entry.service->is_sleeping()) {
                total += state.entry.service->resident_bytes();
            }
        }
        return total;
    };
    // The cheap move first: an idle awake model's KV is mostly prefix cache, and an elastic
    // pool gives those granules back without the model leaving the device. Shrink every idle
    // neighbour once and re-check before any model is put to sleep.
    if (awake_bytes() + needed > budget_bytes_) {
        for (auto& state : models_) {
            if (state.entry.service == target.entry.service || state.entry.service->is_sleeping() ||
                state.entry.service->active_requests() != 0) {
                continue;
            }
            state.entry.service->shrink_kv();
        }
    }
    while (awake_bytes() + needed > budget_bytes_) {
        State* victim  = nullptr;
        bool busy_pick = false;
        const auto now = Clock::now();
        for (auto& state : models_) {
            if (state.entry.service == target.entry.service) { continue; }
            if (state.entry.service->is_sleeping()) { continue; }
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
        victim->entry.service->sleep(/*preempt=*/busy_pick);
    }
    return true;
}

void ModelScheduler::tick_loop() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stopping_) {
        cv_.wait_for(lock, std::chrono::milliseconds(500));
        if (stopping_) { return; }
        // A preempted model holds parked generations that never re-enter the
        // HTTP path, so nothing calls ensure_awake for them: wake it here as
        // soon as idle room suffices. Never preempt on its behalf -- that
        // would ping-pong two busy models forever.
        for (auto& state : models_) {
            GenerationService* service = state.entry.service;
            if (!service->is_sleeping() || service->active_requests() == 0) { continue; }
            if (try_make_room_locked(state, /*allow_preempt=*/false)) {
                write_console_log(ConsoleLogLevel::Info,
                                  "scheduler: re-waking '" + state.entry.name +
                                      "' to finish its parked generations");
                service->wake_up();
                state.woke_at = Clock::now();
                cv_.notify_all();
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
