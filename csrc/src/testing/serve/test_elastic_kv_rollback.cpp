#include "core/device.h"
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

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { return 77; }
    try {
        sinfer::DeviceContext device(0);
        test_wake_retry(device);
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
