#include "core/engine_context.h"

#include <atomic>
#include <cstdio>
#include <stdexcept>

namespace sinfer::ops {
namespace {
thread_local EngineOpsContext* t_context = nullptr;

EngineOpsContext& default_context() {
    static EngineOpsContext instance;
    return instance;
}
} // namespace

namespace detail {
int next_ops_slot_index() {
    static std::atomic<int> counter{0};
    const int index = counter.fetch_add(1);
    if (index >= EngineOpsContext::kMaxSlots) {
        throw std::logic_error("EngineOpsContext: raise kMaxSlots");
    }
    return index;
}
} // namespace detail

EngineOpsContext::~EngineOpsContext() {
    for (auto it = slots_.rbegin(); it != slots_.rend(); ++it) {
        if (void* value = it->value.load(std::memory_order_relaxed); value != nullptr) { it->destroy(value); }
    }
}

void* EngineOpsContext::get_or_create(int index, void* (*create)(), void (*destroy)(void*)) {
    Slot& slot = slots_[static_cast<std::size_t>(index)];
    // Pipeline stages may create the same typed slot concurrently on different devices.
    if (void* value = slot.value.load(std::memory_order_acquire)) { return value; }
    const std::lock_guard<std::mutex> lock(mutex_);
    if (void* value = slot.value.load(std::memory_order_relaxed)) { return value; }
    void* value = create();
    slot.destroy = destroy;
    slot.value.store(value, std::memory_order_release);
    return value;
}

EngineOpsContext& current_ops_context() noexcept {
    return t_context != nullptr ? *t_context : default_context();
}

void bind_ops_context(EngineOpsContext* context) noexcept { t_context = context; }

const void* current_ops_owner() noexcept { return t_context; }

} // namespace sinfer::ops
