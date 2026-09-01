#include "serve/model_scheduler.h"

#include "serve/generation_service.h"

#include <algorithm>
#include <stdexcept>

namespace sinfer::serve {

using Clock = std::chrono::steady_clock;

ModelScheduler::ModelScheduler(std::vector<Entry> entries, std::size_t budget_bytes)
    : budget_bytes_(budget_bytes) {
    const auto now = Clock::now();
    for (auto& entry : entries) {
        models_.push_back(State{std::move(entry), now, now});
    }
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

    const auto deadline = Clock::now() + wait_timeout_;
    while (true) {
        if (!service->is_sleeping()) { return; }
        if (try_make_room_locked(*target)) {
            service->wake_up();
            target->woke_at   = Clock::now();
            target->last_used = target->woke_at;
            cv_.notify_all();
            return;
        }
        if (cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
            throw std::runtime_error(
                "the model is asleep and no room could be freed within the queue timeout -- "
                "other models are busy; retry, or lower their load");
        }
    }
}

bool ModelScheduler::try_make_room_locked(State& target) {
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
    while (awake_bytes() + needed > budget_bytes_) {
        // Evict the least recently used awake model that is idle and past its
        // keep-warm window. Busy models are never evicted here -- requests for
        // the target wait instead (preemptive eviction is a later, gated step).
        State* victim  = nullptr;
        const auto now = Clock::now();
        for (auto& state : models_) {
            if (state.entry.service == target.entry.service) { continue; }
            if (state.entry.service->is_sleeping()) { continue; }
            if (state.entry.service->active_requests() != 0) { continue; }
            if (now - state.last_used < keep_warm_) { continue; }
            if (victim == nullptr || state.last_used < victim->last_used) { victim = &state; }
        }
        if (victim == nullptr) { return false; }
        victim->entry.service->sleep();
    }
    return true;
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
