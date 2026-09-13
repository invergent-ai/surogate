#include "core/device.h"
#include "core/elastic_kv_region.h"

#include <cuda.h>
#include <dlfcn.h>

#include <atomic>
#include <iostream>
#include <stdexcept>

namespace {
std::atomic<int> fail_create_after{0};
std::atomic<int> live_handles{0};

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

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { return 77; }
    try {
        sinfer::DeviceContext device(0);
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
