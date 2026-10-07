#include "core/device.h"
#include "core/sleep.h"
#include "core/unified_memory.h"
#include "runtime/engine/request_memory.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <array>
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

int expect(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

template <class Exception, class Fn>
int expect_throws(Fn&& fn, const char* message) {
    try {
        fn();
    } catch (const Exception&) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    int failures = 0;
    sinfer::DeviceContext device(0);
    sinfer::runtime::RequestMemory memory(device, 1024);
    failures += expect(memory.summary().capacity_bytes == 1024,
                       "constructor did not freeze the requested capacity");

    memory.activate(128, 64);
    const void* first = memory.region().data;
    failures +=
        expect(first != nullptr && memory.region().size == 128 && memory.region().alignment == 64 &&
                   memory.summary().used_bytes == 128 && memory.summary().peak_used_bytes == 128,
               "first activation reported the wrong active/peak state");
    memory.deactivate();
    failures += expect(memory.summary().used_bytes == 0 && memory.summary().peak_used_bytes == 128,
                       "deactivation did not retain the peak");

    memory.activate(256, 256);
    failures += expect(memory.region().data == first,
                       "a later activation changed the frozen device pointer");
    failures += expect_throws<std::invalid_argument>(
        [&] { memory.activate(1025, 256); }, "activation beyond frozen capacity did not throw");
    failures += expect_throws<std::invalid_argument>(
        [&] { memory.activate(128, 3); }, "non-power-of-two activation alignment did not throw");
    failures += expect_throws<std::invalid_argument>([&] { memory.activate(128, 512); },
                                                     "over-aligned activation did not throw");
    failures += expect(memory.region().data == first && memory.region().alignment == 256 &&
                           memory.summary().used_bytes == 256,
                       "rejected activation changed the active allocation");

    memory.reset_peak();
    failures += expect(memory.summary().peak_used_bytes == 256,
                       "reset_peak did not preserve current active usage");
    memory.deactivate();
    memory.reset_peak();
    failures += expect(memory.summary().peak_used_bytes == 0,
                       "reset_peak on inactive memory did not clear the peak");

    sinfer::runtime::RequestMemory empty(device, 0);
    empty.activate(0, 1);
    failures += expect(empty.region().data == nullptr && empty.summary().capacity_bytes == 0,
                       "zero-capacity request memory exposed a device allocation");

    memory.activate_lane(3, 256, 256);
    memory.activate_lane(7, 512, 256);
    const auto left = memory.region(3), right = memory.region(7);
    CUDA_CHECK(cudaMemset(left.data, 0x37, left.size));
    CUDA_CHECK(cudaMemset(right.data, 0x71, right.size));
    memory.deactivate_lane(3);
    memory.activate_lane(5, 256, 256);
    CUDA_CHECK(cudaMemset(memory.region(5).data, 0x55, 256));
    std::array<unsigned char, 512> surviving{};
    CUDA_CHECK(cudaMemcpy(surviving.data(), right.data, right.size, cudaMemcpyDeviceToHost));
    failures += expect(memory.region(7).data == right.data &&
        std::all_of(surviving.begin(), surviving.end(), [](auto b) { return b == 0x71; }),
        "reusing one image slot overwrote another active image");
    failures += expect(!memory.can_activate_lane(9, 512, 256), "full transient pool over-admitted");
    memory.deactivate();

    sinfer::set_sleepable_allocations(true);
    {
        sinfer::runtime::RequestMemory partial_image(device, 1024);
        partial_image.activate(256, 256);
        std::array<unsigned char, 256> expected{}, actual{};
        for (std::size_t i = 0; i < expected.size(); ++i) { expected[i] = static_cast<unsigned char>(i); }
        const auto region = partial_image.region();
        CUDA_CHECK(cudaMemcpy(region.data, expected.data(), expected.size(), cudaMemcpyHostToDevice));
        if (sinfer::device_is_integrated(0)) {
            // A GPU sharing the host's DRAM keeps Offload regions mapped through sleep.
            failures += expect(sinfer::sleep_device(0) == 0 && sinfer::sleep_backup_bytes(0) == 0,
                               "sleep moved the request allocation on a GPU sharing host memory");
        } else {
            failures += expect(sinfer::sleep_device(0) > 0, "sleep did not release the request allocation");
            failures += expect(sinfer::sleep_backup_bytes(0) >= expected.size(),
                               "sleep discarded the active image instead of backing it up");
        }
        sinfer::wake_device(0);
        CUDA_CHECK(cudaMemcpy(actual.data(), region.data, actual.size(), cudaMemcpyDeviceToHost));
        failures += expect(actual == expected && partial_image.region().data == region.data,
                           "wake changed the active image contents or its pointer");
    }
    sinfer::set_sleepable_allocations(false);

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
