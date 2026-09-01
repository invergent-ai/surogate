#pragma once

// Per-engine home for op-layer state that used to be process-global.
//
// The op planes and the adapter store were written when a process held one
// engine: Marlin's staging scratch, its band width and adoption flags, the
// LoRA banks -- all file-scope. Two engines in one process (multi-model
// serving) would race the scratch contents from concurrent rounds on two
// streams, and no host-side lock can serialize device-side consumers.
//
// The context is bound per thread. The Engine constructor binds it while the
// target is constructed and its graphs are captured; the executor's worker
// thread binds the same object before its first round. That pairing is the
// point: addresses baked into captured graphs on the construction thread and
// the addresses eager rounds use on the worker thread come from the same
// slots. Threads with nothing bound (tools, tests, single-model paths) fall
// back to a shared process-default context, which keeps single-engine
// behavior byte-identical.
//
// State joins the context through typed slots: `engine_slot<T>()` returns the
// bound context's lazily-constructed T. A plane migrates by replacing its
// file-scope globals with one struct and one accessor line.

#include <array>
#include <mutex>

namespace sinfer::ops {

class EngineOpsContext {
public:
    static constexpr int kMaxSlots = 16;

    EngineOpsContext() = default;
    ~EngineOpsContext();
    EngineOpsContext(const EngineOpsContext&)            = delete;
    EngineOpsContext& operator=(const EngineOpsContext&) = delete;

    template <class T> T& slot();

private:
    struct Slot {
        void* value = nullptr;
        void (*destroy)(void*) = nullptr;
    };
    void* get_or_create(int index, void* (*create)(), void (*destroy)(void*));

    std::array<Slot, kMaxSlots> slots_{};
    std::mutex mutex_;
};

namespace detail {
int next_ops_slot_index();
template <class T> int ops_slot_index() {
    static const int index = next_ops_slot_index();
    return index;
}
} // namespace detail

template <class T> T& EngineOpsContext::slot() {
    void* value = get_or_create(
        detail::ops_slot_index<T>(), []() -> void* { return new T(); },
        [](void* p) { delete static_cast<T*>(p); });
    return *static_cast<T*>(value);
}

/// The context bound to this thread, or the process default.
[[nodiscard]] EngineOpsContext& current_ops_context() noexcept;
/// Bind (nullptr restores the process default). The caller keeps ownership.
void bind_ops_context(EngineOpsContext* context) noexcept;
/// The bound context as an opaque owner token (nullptr when on the default).
[[nodiscard]] const void* current_ops_owner() noexcept;

/// Shorthand: the bound context's T.
template <class T> T& engine_slot() { return current_ops_context().slot<T>(); }

} // namespace sinfer::ops
