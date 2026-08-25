#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <exception>
#include <iostream>

namespace {

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_value(void* device, std::uint32_t expected, const char* label) {
    std::uint32_t actual  = 0;
    const cudaError_t err = cudaMemcpy(&actual, device, sizeof(actual), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " copy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    if (actual == expected) { return 0; }
    std::cerr << label << " expected 0x" << std::hex << expected << ", got 0x" << actual << std::dec
              << '\n';
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

    try {
        ninfer::DeviceContext device(0);
        ninfer::DeviceArena storage(sizeof(std::uint32_t));
        CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x33, sizeof(std::uint32_t), device.stream));
        device.synchronize();

        ninfer::DecodeGraphDefinition first;
        first.capture(device.stream, [&] {
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x11, sizeof(std::uint32_t), device.stream));
        });
        ninfer::DecodeGraphDefinition second;
        second.capture(device.stream, [&] {
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x22, sizeof(std::uint32_t), device.stream));
        });

        int failures = 0;
        ninfer::DecodeGraphExecutable executable;
        executable.instantiate(first);
        executable.upload(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x33333333U, "initial graph upload");

        executable.launch(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x11111111U, "first graph launch");

        executable.update(second);
        executable.upload(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x11111111U, "updated graph upload");

        executable.launch(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x22222222U, "updated graph launch");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "decode graph test failed: " << error.what() << '\n';
        return 1;
    }
}
