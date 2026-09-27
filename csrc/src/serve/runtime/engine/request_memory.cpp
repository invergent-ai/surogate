#include "runtime/engine/request_memory.h"
#include "runtime/engine/transient_slots.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/sleep.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>

namespace sinfer::runtime {
class RequestMemory::Impl {
public:
    Impl(DeviceContext& context, std::size_t capacity) : device(context.device), slots(capacity) {
        if (capacity != 0) {
            CUDA_CHECK(cudaSetDevice(device));
            arena = std::make_unique<DeviceArena>(capacity);
            // Image features and a partially encoded image survive across
            // scheduler rounds. Preemptive sleep must restore them as well.
            sleep_tag_region(arena->base(), SleepTag::Offload);
        }
    }

    ~Impl() {
        if (arena != nullptr) {
            (void)cudaSetDevice(device);
            arena.reset();
        }
    }

    int device = 0;
    std::unique_ptr<DeviceArena> arena;
    TransientSlots slots;
};

RequestMemory::RequestMemory(DeviceContext& device, std::size_t frozen_capacity_bytes)
    : impl_(std::make_unique<Impl>(device, frozen_capacity_bytes)) {}

RequestMemory::~RequestMemory() = default;

void RequestMemory::activate(std::size_t bytes, std::size_t alignment) {
    activate_lane(0, bytes, alignment);
}

bool RequestMemory::can_activate_lane(std::uint32_t lane, std::size_t bytes, std::size_t alignment) const noexcept {
    return impl_->slots.fit(lane, bytes, alignment).has_value();
}

void RequestMemory::activate_lane(std::uint32_t lane, std::size_t bytes, std::size_t alignment) {
    impl_->slots.activate(lane, bytes, alignment);
}

void RequestMemory::deactivate_lane(std::uint32_t lane) noexcept { impl_->slots.release(lane); }
void RequestMemory::deactivate() noexcept { impl_->slots.clear(); }

TransientRegion RequestMemory::region(std::uint32_t lane) const noexcept {
    const auto region = impl_->slots.region(lane);
    if (!region) { return {}; }
    return {static_cast<std::byte*>(impl_->arena->base()) + region->offset, region->bytes, region->alignment};
}

ArenaMemorySummary RequestMemory::summary() const noexcept {
    return ArenaMemorySummary{impl_->arena != nullptr ? impl_->arena->capacity() : 0,
                              impl_->slots.used(), impl_->slots.peak()};
}

void RequestMemory::reset_peak() noexcept { impl_->slots.reset_peak(); }

} // namespace sinfer::runtime
