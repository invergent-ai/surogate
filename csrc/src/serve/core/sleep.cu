// Sleep-mode memory registry (core/sleep.h).

#include "core/sleep.h"

#include "core/device.h"
#include "core/engine_context.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

namespace sinfer {
namespace {

std::atomic<bool> g_sleepable{false};

void driver_check(CUresult result, const char* what) {
    if (result == CUDA_SUCCESS) { return; }
    const char* text = nullptr;
    cuGetErrorString(result, &text);
    throw std::runtime_error(std::string(what) + " failed: " +
                             (text != nullptr ? text : "unknown CUDA driver error"));
}

struct Region {
    const void* owner                   = nullptr;
    int device                          = 0;
    CUdeviceptr va                      = 0;
    std::size_t bytes                   = 0; ///< granularity-rounded mapped size
    CUmemGenericAllocationHandle handle = {};
    SleepTag tag                        = SleepTag::Offload;
    void* backup                        = nullptr; ///< pinned host, kept across cycles
    bool asleep                         = false;
};

struct Registry {
    std::mutex mutex;
    std::map<void*, Region> regions;
};

Registry& registry() {
    static Registry instance;
    return instance;
}

CUmemAllocationProp allocation_prop(int device) {
    CUmemAllocationProp prop = {};
    prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id         = device;
    return prop;
}

void map_region(Region& region) {
    const CUmemAllocationProp prop = allocation_prop(region.device);
    driver_check(cuMemCreate(&region.handle, region.bytes, &prop, 0), "cuMemCreate");
    CUresult mapped = cuMemMap(region.va, region.bytes, 0, region.handle, 0);
    if (mapped != CUDA_SUCCESS) {
        (void)cuMemRelease(region.handle);
        driver_check(mapped, "cuMemMap");
    }
    CUmemAccessDesc access = {};
    access.location        = prop.location;
    access.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    const CUresult set     = cuMemSetAccess(region.va, region.bytes, &access, 1);
    if (set != CUDA_SUCCESS) {
        (void)cuMemUnmap(region.va, region.bytes);
        (void)cuMemRelease(region.handle);
        driver_check(set, "cuMemSetAccess");
    }
}

void unmap_region(Region& region) {
    driver_check(cuMemUnmap(region.va, region.bytes), "cuMemUnmap");
    driver_check(cuMemRelease(region.handle), "cuMemRelease");
}

} // namespace

bool sleepable_allocations_enabled() noexcept { return g_sleepable.load(); }
void set_sleepable_allocations(bool enabled) noexcept { g_sleepable.store(enabled); }

void* sleep_alloc(std::size_t bytes, int device) {
    // The driver API needs an initialized context; the runtime call ensures one.
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));
    const CUmemAllocationProp prop = allocation_prop(device);
    std::size_t granularity        = 0;
    driver_check(
        cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        "cuMemGetAllocationGranularity");
    Region region;
    region.owner  = ops::current_ops_owner();
    region.device = device;
    region.bytes  = (bytes + granularity - 1) / granularity * granularity;
    driver_check(cuMemAddressReserve(&region.va, region.bytes, 0, 0, 0), "cuMemAddressReserve");
    try {
        map_region(region);
    } catch (...) {
        (void)cuMemAddressFree(region.va, region.bytes);
        throw;
    }
    void* base = reinterpret_cast<void*>(region.va);
    const std::lock_guard<std::mutex> lock(registry().mutex);
    registry().regions.emplace(base, region);
    return base;
}

bool sleep_free(void* base) noexcept {
    Region region;
    {
        const std::lock_guard<std::mutex> lock(registry().mutex);
        auto found = registry().regions.find(base);
        if (found == registry().regions.end()) { return false; }
        region = found->second;
        registry().regions.erase(found);
    }
    try {
        if (!region.asleep) { unmap_region(region); }
        driver_check(cuMemAddressFree(region.va, region.bytes), "cuMemAddressFree");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "sleep_free: %s\n", error.what());
    }
    if (region.backup != nullptr) { (void)cudaFreeHost(region.backup); }
    return true;
}

void sleep_tag_region(const void* base, SleepTag tag) {
    const std::lock_guard<std::mutex> lock(registry().mutex);
    auto found = registry().regions.find(const_cast<void*>(base));
    if (found == registry().regions.end()) {
        // Sleep mode off: arenas are ordinary cudaMalloc and there is nothing
        // to tag. The call sites run unconditionally, so this is the norm.
        return;
    }
    found->second.tag = tag;
}

std::size_t sleep_device(int device, const void* owner) {
    using Clock = std::chrono::steady_clock;
    const std::lock_guard<std::mutex> lock(registry().mutex);
    CUDA_CHECK(cudaSetDevice(device));
    std::size_t released = 0;
    double pin_ms = 0, copy_ms = 0, unmap_ms = 0;
    for (auto& [base, region] : registry().regions) {
        if (region.device != device || region.asleep) { continue; }
        if (owner != nullptr && region.owner != owner) { continue; }
        if (region.tag == SleepTag::Offload) {
            if (region.backup == nullptr) {
                const auto t0 = Clock::now();
                CUDA_CHECK(cudaMallocHost(&region.backup, region.bytes));
                pin_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            }
            const auto t0 = Clock::now();
            CUDA_CHECK(cudaMemcpy(region.backup, base, region.bytes, cudaMemcpyDeviceToHost));
            copy_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        }
        const auto t0 = Clock::now();
        unmap_region(region);
        unmap_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        region.asleep = true;
        released += region.bytes;
    }
    if (std::getenv("SUROGATE_SLEEP_DEBUG") != nullptr) {
        std::fprintf(stderr, "sleep_device: pin %.0fms copy %.0fms unmap %.0fms\n", pin_ms,
                     copy_ms, unmap_ms);
    }
    return released;
}

std::size_t wake_device(int device, const void* owner) {
    using Clock = std::chrono::steady_clock;
    const std::lock_guard<std::mutex> lock(registry().mutex);
    CUDA_CHECK(cudaSetDevice(device));
    std::size_t mapped = 0;
    double map_ms = 0, copy_ms = 0;
    for (auto& [base, region] : registry().regions) {
        if (region.device != device || !region.asleep) { continue; }
        if (owner != nullptr && region.owner != owner) { continue; }
        const auto t0 = Clock::now();
        map_region(region); // throws on OOM; earlier regions stay woken for retry
        map_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        region.asleep = false;
        if (region.tag == SleepTag::Offload && region.backup != nullptr) {
            const auto t1 = Clock::now();
            CUDA_CHECK(cudaMemcpy(base, region.backup, region.bytes, cudaMemcpyHostToDevice));
            copy_ms += std::chrono::duration<double, std::milli>(Clock::now() - t1).count();
        }
        mapped += region.bytes;
    }
    if (std::getenv("SUROGATE_SLEEP_DEBUG") != nullptr) {
        std::fprintf(stderr, "wake_device: map %.0fms copy %.0fms\n", map_ms, copy_ms);
    }
    return mapped;
}

std::size_t sleep_owned_bytes(const void* owner) noexcept {
    const std::lock_guard<std::mutex> lock(registry().mutex);
    std::size_t bytes = 0;
    for (const auto& [base, region] : registry().regions) {
        if (owner == nullptr || region.owner == owner) { bytes += region.bytes; }
    }
    return bytes;
}

std::size_t device_free_bytes(int device) noexcept {
    if (cudaSetDevice(device) != cudaSuccess) { return 0; }
    std::size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) { return 0; }
    return free_bytes;
}

bool device_asleep(int device, const void* owner) noexcept {
    const std::lock_guard<std::mutex> lock(registry().mutex);
    for (const auto& [base, region] : registry().regions) {
        if (region.device == device && region.asleep &&
            (owner == nullptr || region.owner == owner)) { return true; }
    }
    return false;
}

std::size_t sleep_backup_bytes(int device) noexcept {
    const std::lock_guard<std::mutex> lock(registry().mutex);
    std::size_t bytes = 0;
    for (const auto& [base, region] : registry().regions) {
        if (region.device == device && region.backup != nullptr) { bytes += region.bytes; }
    }
    return bytes;
}

} // namespace sinfer
