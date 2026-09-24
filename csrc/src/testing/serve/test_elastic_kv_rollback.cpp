#include "core/device.h"
#include "core/device_memory_error.h"
#include "core/elastic_kv_region.h"

#include <cuda.h>
#include <dlfcn.h>

#include <atomic>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
std::atomic<int> fail_create_after{0};
std::atomic<int> live_handles{0};
std::atomic<int> fail_copy_after{0};
std::atomic<int> live_backups{0};

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}
}

// Inject OOM at each plane without consuming the rest of the GPU's memory.
// Successful driver calls are real, so the retry also detects stale VA mappings.
extern "C" CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle* handle, std::size_t bytes,
                                         const CUmemAllocationProp* prop, unsigned long long flags) {
    static const auto real = reinterpret_cast<decltype(&cuMemCreate)>(dlsym(RTLD_NEXT, "cuMemCreate"));
    if (fail_create_after > 0 && --fail_create_after == 0) { return CUDA_ERROR_OUT_OF_MEMORY; }
    const auto result = real(handle, bytes, prop, flags);
    if (result == CUDA_SUCCESS) { ++live_handles; }
    return result;
}

extern "C" CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle handle) {
    static const auto real = reinterpret_cast<decltype(&cuMemRelease)>(dlsym(RTLD_NEXT, "cuMemRelease"));
    const auto result = real(handle);
    if (result == CUDA_SUCCESS) { --live_handles; }
    return result;
}

extern "C" cudaError_t CUDARTAPI cudaMallocHost(void** pointer, std::size_t bytes) {
    using Allocate = cudaError_t(CUDARTAPI*)(void**, std::size_t);
    static const auto real = reinterpret_cast<Allocate>(dlsym(RTLD_NEXT, "cudaMallocHost"));
    const auto result = real(pointer, bytes);
    if (result == cudaSuccess) { ++live_backups; }
    return result;
}
extern "C" cudaError_t CUDARTAPI cudaFreeHost(void* pointer) {
    static const auto real = reinterpret_cast<decltype(&cudaFreeHost)>(dlsym(RTLD_NEXT, "cudaFreeHost"));
    const auto result = real(pointer);
    if (pointer && result == cudaSuccess) { --live_backups; }
    return result;
}
extern "C" cudaError_t CUDARTAPI cudaMemcpy(void* target, const void* source, std::size_t bytes, cudaMemcpyKind kind) {
    static const auto real = reinterpret_cast<decltype(&cudaMemcpy)>(dlsym(RTLD_NEXT, "cudaMemcpy"));
    if (kind == cudaMemcpyHostToDevice && fail_copy_after > 0 && --fail_copy_after == 0) {
        // A failed restore must leave this plane pending, even though its granule
        // is already mapped. Poison it so accidentally skipping it cannot pass.
        (void)cudaMemset(target, 0xee, bytes);
        return cudaErrorMemoryAllocation;
    }
    return real(target, source, bytes, kind);
}

static void test_wake_retry(sinfer::DeviceContext& device) {
    constexpr std::size_t quantum = 2ULL << 20;
    constexpr int pages = 3, planes = 3;
    constexpr std::size_t bytes = pages * planes * quantum;
    for (const bool copy_failure : {false, true}) {
        const int baseline = live_handles, backup_baseline = live_backups;
        {
            sinfer::ElasticKvRegion region({
                .device = device.device, .fence_stream = device.stream, .bytes = bytes,
                .page_count = pages, .granule_pages = 1, .reserve_granules = 0,
                .planes = {{0, quantum}, {pages * quantum, quantum}, {2 * pages * quantum, quantum}}});
            for (int page = 0; page < pages; ++page) { region.acquire_page(page); }
            region.wait_idle();
            for (int plane = 0; plane < planes; ++plane) {
                for (int page = 0; page < pages; ++page) {
                    auto* at = static_cast<std::byte*>(region.base()) + (plane * pages + page) * quantum;
                    CUDA_CHECK(cudaMemsetAsync(at, 0x20 + plane * pages + page, quantum, device.stream));
                }
            }
            device.synchronize();
            require(region.sleep() == bytes && region.mapped_bytes() == 0, "sleep did not release the populated granules");
            require(live_backups == backup_baseline + pages * planes, "sleep did not retain every live plane");
            if (copy_failure) { fail_copy_after = planes + 2; }
            else { fail_create_after = planes + 2; }
            bool failed = false;
            try { (void)region.wake(); } catch (const std::runtime_error&) { failed = true; }
            require(failed && fail_copy_after == 0 && fail_create_after == 0, "wake fault was not injected");
            const auto before = region.mapped_bytes();
            require(before == (copy_failure ? 2 : 1) * planes * quantum, "partial wake mapping state is incorrect");
            require(region.wake() == bytes - before, "wake counted mappings restored before the retry");
            region.wait_idle();
            require(region.mapped_bytes() == bytes, "retry did not restore every granule");
            require(live_backups == backup_baseline, "retry leaked a pinned backup");
            std::vector<unsigned char> actual(quantum);
            for (int plane = 0; plane < planes; ++plane) {
                for (int page = 0; page < pages; ++page) {
                    const auto* at = static_cast<const std::byte*>(region.base()) + (plane * pages + page) * quantum;
                    CUDA_CHECK(cudaMemcpy(actual.data(), at, quantum, cudaMemcpyDeviceToHost));
                    const auto expected = 0x20 + plane * pages + page;
                    require(std::all_of(actual.begin(), actual.end(), [&](auto value) { return value == expected; }),
                            "wake retry lost a live cache plane");
                }
            }
            require(region.wake() == 0, "completed wake was not idempotent");
            require(region.sleep() == bytes && region.wake() == bytes, "region cannot sleep/wake after recovery");
            for (int page = 0; page < pages; ++page) { region.release_page(page); }
            region.wait_idle();
        }
        require(live_handles == baseline && live_backups == backup_baseline, "wake retry leaked backing allocations");
    }
}

// What the region reads as free device memory in the cases below.
std::size_t g_probe_free = std::size_t{1} << 40;
std::size_t probe_free() { return g_probe_free; }

// Out of memory on demand (SUROGATE-CHANGES #16): the region gives back the granules no page
// uses and tries again; with nothing to give back the failure is a DeviceOutOfMemory, which the
// executor recovers from, and nothing leaks.
static void test_out_of_memory_on_demand(sinfer::DeviceContext& device) {
    constexpr std::size_t quantum = 2ULL << 20;
    const int baseline = live_handles;
    {
        sinfer::ElasticKvRegion region({
            .device = device.device, .fence_stream = device.stream, .bytes = 4 * quantum,
            .page_count = 4, .granule_pages = 1, .reserve_granules = 2,
            .planes = {{0, quantum}}});
        region.wait_idle();
        region.acquire_page(0);
        region.wait_idle();
        require(region.mapped_granules() == 3, "the reserve keeps two free granules mapped");
        // Page 3 lies past the reserve. Its first cuMemCreate fails; the free granules go back
        // and the retry maps it.
        fail_create_after = 1;
        region.acquire_page(3);
        require(fail_create_after == 0, "on-demand fault was not injected");
        require(region.mapped_bytes() >= 2 * quantum, "page 3 was not mapped after the retry");
        region.wait_idle();
        region.release_page(3);
        region.release_page(0);
        region.wait_idle();
    }
    require(live_handles == baseline, "reclaim and retry leaked an allocation handle");
    {
        sinfer::ElasticKvRegion region({
            .device = device.device, .fence_stream = device.stream, .bytes = 2 * quantum,
            .page_count = 2, .granule_pages = 1, .reserve_granules = 0,
            .planes = {{0, quantum}}});
        region.wait_idle();
        region.acquire_page(0);
        fail_create_after = 1;
        bool typed = false;
        try { region.acquire_page(1); } catch (const sinfer::DeviceOutOfMemory&) { typed = true; }
        require(typed && fail_create_after == 0, "out of memory with nothing to reclaim was not a DeviceOutOfMemory");
        require(region.mapped_granules() == 1, "the failed map changed the mapping");
        region.acquire_page(1);
        region.wait_idle();
        require(region.mapped_granules() == 2, "the page could not be mapped once memory returned");
        region.release_page(1);
        region.release_page(0);
        region.wait_idle();
    }
    require(live_handles == baseline, "a refused map leaked an allocation handle");
}

// The mapped footprint stays within the cap that automatic sizing planned for: the reserve does
// not map past it, a free granule is traded for the one demand needs, and with none to trade the
// map goes past the cap only while the device keeps the headroom free.
static void test_mapping_stays_within_the_cap(sinfer::DeviceContext& device) {
    constexpr std::size_t quantum = 2ULL << 20;
    const int baseline = live_handles;
    g_probe_free = std::size_t{1} << 40;
    {
        sinfer::ElasticKvRegion region({
            .device = device.device, .fence_stream = device.stream, .bytes = 4 * quantum,
            .page_count = 4, .granule_pages = 1, .reserve_granules = 4, .cap_pages = 2,
            .headroom_bytes = 8 * quantum, .free_bytes_probe = &probe_free,
            .planes = {{0, quantum}}});
        region.wait_idle();
        require(region.mapped_granules() == 2, "the reserve mapped past the cap");
        region.acquire_page(0);
        region.acquire_page(1);
        region.wait_idle();
        require(region.mapped_granules() == 2, "two pages in use map two granules");
        region.release_page(1); // kept mapped: it is inside the reserve
        region.wait_idle();
        require(region.mapped_granules() == 2, "the emptied granule stays mapped for reuse");
        region.acquire_page(3); // at the cap: granule 1 is traded for granule 3
        region.wait_idle();
        require(region.mapped_granules() == 2, "demand at the cap went past it with a free granule to trade");
        // Pages 0 and 3 in use, nothing free to trade. Short of the headroom: refused.
        g_probe_free = 8 * quantum;
        bool refused = false;
        try { region.acquire_page(1); } catch (const sinfer::DeviceOutOfMemory&) { refused = true; }
        require(refused, "a map past the cap was not refused with the headroom at stake");
        require(region.mapped_granules() == 2, "the refused map changed the mapping");
        // With the headroom free it goes ahead.
        g_probe_free = 9 * quantum;
        region.acquire_page(1);
        region.wait_idle();
        require(region.mapped_granules() == 3, "a map past the cap was refused with memory to spare");
        for (const int page : {0, 1, 3}) { region.release_page(page); }
        region.wait_idle();
    }
    g_probe_free = std::size_t{1} << 40;
    require(live_handles == baseline, "the cap cases leaked an allocation handle");
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { return 77; }
    try {
        sinfer::DeviceContext device(0);
        test_wake_retry(device);
        test_out_of_memory_on_demand(device);
        test_mapping_stays_within_the_cap(device);
        constexpr std::size_t quantum = 2ULL << 20;
        for (const bool retry : {false, true}) {
            for (int plane = 0; plane < 3; ++plane) {
                const int baseline = live_handles;
                {
                    sinfer::ElasticKvRegion region({
                        .device = device.device, .fence_stream = device.stream,
                        .bytes = 3 * quantum, .page_count = 1, .granule_pages = 1,
                        .reserve_granules = 0,
                        .planes = {{0, quantum}, {quantum, quantum}, {2 * quantum, quantum}}});
                    region.wait_idle();
                    fail_create_after = plane + 1;
                    bool failed = false;
                    try { region.acquire_page(0); }
                    catch (const std::runtime_error&) { failed = true; }
                    require(failed && fail_create_after == 0, "plane allocation fault was not injected");
                    require(region.mapped_bytes() == 0 && region.mapped_granules() == 0,
                            "failed mapping changed physical accounting");
                    require(live_handles == baseline, "failed mapping leaked an allocation handle");
                    if (retry) {
                        region.acquire_page(0);
                        region.wait_idle();
                        require(region.mapped_bytes() == 3 * quantum, "mapping could not be retried");
                        CUDA_CHECK(cudaMemsetAsync(region.base(), 0x5a, 3 * quantum, device.stream));
                        device.synchronize();
                        region.release_page(0);
                        region.wait_idle();
                        require(region.mapped_bytes() == 0, "retried page did not retire");
                    }
                }
                require(live_handles == baseline, "region destruction leaked an allocation handle");
            }
        }
        std::cout << "Partial KV allocation failures release every plane and remain retryable\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
