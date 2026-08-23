#include "core/device.h"
#include "runtime/engine/request_memory.h"

#include <cuda_runtime.h>

#include <cstddef>
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
    ninfer::DeviceContext device(0);
    ninfer::runtime::RequestMemory memory(device, 1024);
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

    ninfer::runtime::RequestMemory empty(device, 0);
    empty.activate(0, 1);
    failures += expect(empty.region().data == nullptr && empty.summary().capacity_bytes == 0,
                       "zero-capacity request memory exposed a device allocation");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
