#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/sleep.h"

#include <map>
#include <memory>
#include <mutex>
#include <utility>

namespace sinfer::ops::detail {

// A derived plane follows the engine's allocation policy. Offload-tagged arenas
// retain both addresses and contents through sleep, including graph references.
struct PlaneStorage {
    std::unique_ptr<DeviceArena> storage;
    cudaEvent_t ready = nullptr;
    PlaneStorage() = default;
    PlaneStorage(PlaneStorage&& other) noexcept
        : storage(std::move(other.storage)), ready(std::exchange(other.ready, nullptr)) {}
    ~PlaneStorage() {
        if (ready) {
            (void)cudaEventSynchronize(ready);
            (void)cudaEventDestroy(ready);
        }
    }
};

inline std::unique_ptr<DeviceArena> try_plane_storage(std::size_t bytes) {
    try { return std::make_unique<DeviceArena>(bytes); }
    catch (const std::exception&) { return {}; } // optional acceleration can decline an allocation
}

inline std::size_t plane_storage_bytes(const DeviceArena* storage) noexcept {
    if (!storage) { return 0; }
    const auto mapped = sleep_allocation_bytes(storage->base());
    return mapped ? mapped : storage->capacity();
}

// Device states are never erased while an engine is running, so references stay
// valid after releasing the map lock. Destruction happens after executor shutdown.
template <class State> class DevicePlaneStates {
public:
    State& current() {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        const std::lock_guard<std::mutex> lock(mutex_);
        return devices_[device];
    }
    ~DevicePlaneStates() {
        int previous = 0;
        const bool restore = cudaGetDevice(&previous) == cudaSuccess;
        while (!devices_.empty()) {
            const auto first = devices_.begin();
            (void)cudaSetDevice(first->first);
            // A failed construction may have launched work before the executor
            // could take ownership. VMM teardown must wait just as cudaFree did.
            (void)cudaDeviceSynchronize();
            devices_.erase(first);
        }
        if (restore) { (void)cudaSetDevice(previous); }
    }
private:
    std::mutex mutex_;
    std::map<int, State> devices_;
};

} // namespace sinfer::ops::detail
