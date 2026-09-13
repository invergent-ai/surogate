#include "core/device.h"
#include "core/engine_context.h"
#include "ops/linear/marlin/marlin_plane.h"

#include <array>
#include <iostream>
#include <stdexcept>

using namespace sinfer;

namespace {
struct Context {
    ops::EngineOpsContext context;
    ops::EngineOpsContext* previous = &ops::current_ops_context();
    Context() { ops::bind_ops_context(&context); }
    ~Context() { ops::bind_ops_context(previous); }
};
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) { return 77; }
    try {
        DeviceContext first(0), second(1);
        Context context;
        const std::array<DeviceContext*, 2> devices{&first, &second};
        std::array<void*, 2> pointers{};
        std::array<cudaGraph_t, 2> graphs{};
        std::array<cudaGraphExec_t, 2> executables{};
        constexpr std::size_t bytes = 1ULL << 20;
        for (int index : {0, 1}) {
            auto& device = *devices[index];
            CUDA_CHECK(cudaSetDevice(device.device));
            pointers[index] = ops::detail::marlin_fused_parent(bytes, device.stream);
            require(pointers[index] != nullptr, "fused parent allocation failed");
            cudaPointerAttributes attributes{};
            CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointers[index]));
            require(attributes.device == device.device, "fused parent belongs to a different pipeline device");
            if (index == 1) {
                // Grow this device's buffer after the other stage captured its address.
                pointers[index] = ops::detail::marlin_fused_parent(4 * bytes, device.stream);
                require(pointers[index] != nullptr, "fused parent growth failed");
            }
            CUDA_CHECK(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeThreadLocal));
            require(ops::detail::marlin_fused_parent(bytes, device.stream) == pointers[index],
                    "capture changed the device's fused parent address");
            CUDA_CHECK(cudaMemsetAsync(pointers[index], 0x51 + index, bytes, device.stream));
            CUDA_CHECK(cudaStreamEndCapture(device.stream, &graphs[index]));
            CUDA_CHECK(cudaGraphInstantiate(&executables[index], graphs[index], nullptr, nullptr, 0));
        }
        for (int iteration = 0; iteration < 3; ++iteration) {
            for (int index : {0, 1}) {
                auto& device = *devices[index];
                CUDA_CHECK(cudaSetDevice(device.device));
                require(ops::detail::marlin_fused_parent(bytes, device.stream) == pointers[index],
                        "another device changed the captured parent address");
                CUDA_CHECK(cudaGraphLaunch(executables[index], device.stream));
            }
            for (int index : {0, 1}) {
                CUDA_CHECK(cudaSetDevice(index));
                devices[index]->synchronize();
                unsigned char value = 0;
                CUDA_CHECK(cudaMemcpy(&value, pointers[index], 1, cudaMemcpyDeviceToHost));
                require(value == 0x51 + index, "pipeline devices overwrote each other's fused output");
            }
        }
        for (int index : {0, 1}) {
            CUDA_CHECK(cudaSetDevice(index));
            CUDA_CHECK(cudaGraphExecDestroy(executables[index]));
            CUDA_CHECK(cudaGraphDestroy(graphs[index]));
        }
        std::cout << "Marlin fused output buffers and captured graphs are isolated per device\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
