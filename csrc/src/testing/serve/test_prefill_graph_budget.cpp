// Prefill and mixed-round graphs captured while serving share a byte budget
// (#228). Mixed graphs are keyed by exact decode width, chunk bucket and band,
// so a live server meets new keys for as long as it runs; before the budget
// each one was captured and kept until a capture found the device full, and
// that failure stops the engine. The family must refuse a capture while the
// device still has room for the rest of the round -- the caller then runs the
// eager body -- evict graphs that have gone idle before refusing, and never
// evict what the startup precapture pinned.
#include "core/device.h"
#include "family/impl/runtime/prefill_graph.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <iostream>

namespace {

using sinfer::family::detail::PrefillGraphFamily;

std::size_t fake_free_bytes = 0;
std::size_t fake_free() { return fake_free_bytes; }

constexpr std::size_t kRoomy = 64ULL << 30;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

int budget_cases(sinfer::DeviceContext& device) {
    int failures = 0;
    constexpr std::size_t kScratchBytes = 1 << 20;
    void* scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&scratch, kScratchBytes));
    const auto body = [&] {
        CUDA_CHECK(cudaMemsetAsync(scratch, 0, kScratchBytes, device.stream));
    };
    const auto mixed = [](std::int32_t batch) {
        return PrefillGraphFamily::mixed_key(128, batch, 0);
    };

    {
        PrefillGraphFamily family(device, 256, 4096, 0, 64ULL << 20);
        family.set_free_bytes_probe(&fake_free);

        // The startup precapture is the plan's own: it captures whatever the device says.
        fake_free_bytes = 0;
        sinfer::DecodeGraphExecutable* const bucket_128 = family.ensure(128, body);
        sinfer::DecodeGraphExecutable* const bucket_256 = family.ensure(256, body);
        failures += check(bucket_128 != nullptr && bucket_256 != nullptr,
                          "startup captures must not be refused");
        family.finish_startup();
        failures += check(family.lazy_graph_bytes() == 0,
                          "startup graphs must be pinned, not charged to the budget");

        fake_free_bytes = kRoomy;
        sinfer::DecodeGraphExecutable* const width_1 = family.ensure(mixed(1), body);
        failures += check(width_1 != nullptr, "a capture with room must succeed");
        failures += check(family.ensure(mixed(1), body) == width_1,
                          "a captured key must replay its graph");
        failures += check(family.graph_count() == 3, "one graph per key");

        // Short of the floor: refuse without capturing, and keep serving what exists.
        fake_free_bytes = PrefillGraphFamily::kLazyCaptureFloorBytes / 2;
        failures += check(family.ensure(mixed(2), body) == nullptr,
                          "a capture the device cannot spare must fall back to the eager body");
        failures += check(family.refusals() == 1, "the refusal must be counted");
        failures += check(family.graph_count() == 3, "a refused key must not be captured");
        failures += check(family.ensure(mixed(1), body) == width_1,
                          "existing graphs must keep replaying while the device is short");
        failures += check(family.evictions() == 0,
                          "a graph used within the idle window must not be evicted");

        // Once idle, lazily captured graphs are evicted to make room; pinned ones never.
        family.set_eviction_idle(std::chrono::seconds(0));
        failures += check(family.ensure(mixed(3), body) == nullptr,
                          "evicting every idle graph does not make the floor, so still refuse");
        failures += check(family.evictions() == 1, "the idle lazy graph must be evicted");
        failures += check(family.graph_count() == 2, "pinned startup graphs must survive");
        failures += check(family.ensure(128, body) == bucket_128 &&
                              family.ensure(256, body) == bucket_256,
                          "pinned graphs must still replay");
        failures += check(family.lazy_graph_bytes() == 0, "an evicted graph must be uncharged");

        fake_free_bytes = kRoomy;
        failures += check(family.ensure(mixed(3), body) != nullptr,
                          "capture must resume once the device has room");
    }
    {
        PrefillGraphFamily family(device, 256, 4096, 0, 0);
        family.set_free_bytes_probe(&fake_free);
        fake_free_bytes = kRoomy;
        failures += check(family.ensure(128, body) != nullptr, "startup ignores a zero budget");
        family.finish_startup();
        failures += check(family.ensure(mixed(1), body) == nullptr,
                          "a zero budget must capture nothing after startup");
    }

    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    CUDA_CHECK(cudaFree(scratch));
    return failures;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP prefill graph budget: no CUDA device\n";
        return 77;
    }
    sinfer::DeviceContext device(0);
    const int failures = budget_cases(device);
    if (failures != 0) {
        std::cerr << failures << " prefill graph budget check(s) failed\n";
        return 1;
    }
    std::cout << "PASS prefill graph budget\n";
    return 0;
}
