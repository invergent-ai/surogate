#include "core/device.h"
#include "core/elastic_kv_region.h"

#include <dlfcn.h>

#include <array>
#include <atomic>
#include <iostream>

namespace {
constexpr std::size_t MiB = 1ULL << 20;
std::atomic<bool> inject{false};
std::array<std::size_t, 2> available{};
std::atomic<int> queried_device{-1};
}

// Exercise the default CUDA probe with deterministic, asymmetric device budgets.
extern "C" cudaError_t CUDARTAPI cudaMemGetInfo(std::size_t* free, std::size_t* total) {
    static const auto real = reinterpret_cast<decltype(&cudaMemGetInfo)>(dlsym(RTLD_NEXT, "cudaMemGetInfo"));
    if (!inject) { return real(free, total); }
    int device = -1;
    const auto result = cudaGetDevice(&device);
    if (result != cudaSuccess) { return result; }
    queried_device = device;
    *free = available.at(device);
    *total = 128 * MiB;
    return cudaSuccess;
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) { return 77; }
    try {
        for (int owner : {0, 1}) {
            sinfer::DeviceContext device(owner);
            sinfer::ElasticKvRegion region({
                .device = owner, .fence_stream = device.stream,
                .bytes = 8 * MiB, .page_count = 2, .granule_pages = 1,
                .reserve_granules = 0, .cap_pages = 1, .overcommit = true,
                .planes = {{0, 2 * MiB}, {4 * MiB, 2 * MiB}}});
            region.wait_idle();
            for (bool fits : {false, true}) {
                available[owner] = (fits ? 64 : 4) * MiB;
                available[1 - owner] = (fits ? 4 : 64) * MiB;
                CUDA_CHECK(cudaSetDevice(1 - owner));
                inject = true;
                const bool allowed = region.try_entitle(2);
                inject = false;
                int after = -1;
                CUDA_CHECK(cudaGetDevice(&after));
                if (allowed != fits || queried_device != owner || after != 1 - owner) {
                    std::cerr << "gate queried the wrong device or changed the caller's device\n";
                    return 1;
                }
                region.set_entitled_pages(0);
            }
        }
        std::cout << "KV overcommit uses its own device budget and preserves caller binding\n";
        return 0;
    } catch (const std::exception& error) {
        inject = false;
        std::cerr << error.what() << '\n';
        return 1;
    }
}
