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
        if (it->value != nullptr && it->destroy != nullptr) { it->destroy(it->value); }
    }
}

void* EngineOpsContext::get_or_create(int index, void* (*create)(), void (*destroy)(void*)) {
    Slot& slot = slots_[static_cast<std::size_t>(index)];
    // Fast path without the lock: the pointer is written once, under the lock,
    // and read many times afterwards from the two threads that bind this
    // context (construction, then the worker); both synchronize through the
    // executor-thread handoff before any concurrent read.
    if (slot.value != nullptr) { return slot.value; }
    const std::lock_guard<std::mutex> lock(mutex_);
    if (slot.value == nullptr) {
        slot.value   = create();
        slot.destroy = destroy;
    }
    return slot.value;
}

EngineOpsContext& current_ops_context() noexcept {
    return t_context != nullptr ? *t_context : default_context();
}

void bind_ops_context(EngineOpsContext* context) noexcept { t_context = context; }

const void* current_ops_owner() noexcept { return t_context; }

} // namespace sinfer::ops
