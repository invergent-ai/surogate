// Deterministic NVML diagnostics across concurrent callers, without a GPU dependency.
#include "core/device_footprint.h"
#include <cuda_runtime.h>
#include <nvml.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <latch>
#include <new>
#include <string_view>
#include <thread>
#include <vector>
#include <unistd.h>

namespace {
thread_local int mode = 0;
thread_local bool fail_allocation = false;
int library;
nvmlReturn_t init() { return NVML_SUCCESS; }
nvmlReturn_t device_by_pci(const char*, nvmlDevice_t* device) {
    if (mode == 1) { return NVML_ERROR_NOT_FOUND; }
    *device = reinterpret_cast<nvmlDevice_t>(static_cast<std::uintptr_t>(mode + 1));
    return NVML_SUCCESS;
}
nvmlReturn_t processes(nvmlDevice_t device, unsigned* count, nvmlProcessInfo_t* output) {
    const int selected = static_cast<int>(reinterpret_cast<std::uintptr_t>(device)) - 1;
    if (selected == 3) { return NVML_ERROR_NO_PERMISSION; }
    if (!output) { *count = selected == 2 ? 0 : 1; return *count ? NVML_ERROR_INSUFFICIENT_SIZE : NVML_SUCCESS; }
    output[0] = {};
    output[0].pid = getpid() + (selected == 5 ? 1 : 0);
    output[0].usedGpuMemory = selected == 4 ? 0 : 1234;
    return NVML_SUCCESS;
}
}

void* operator new(std::size_t bytes) {
    if (fail_allocation) { fail_allocation = false; throw std::bad_alloc(); }
    if (void* pointer = std::malloc(bytes ? bytes : 1)) { return pointer; }
    throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }

extern "C" void* __wrap_dlopen(const char*, int) { return &library; }
extern "C" void* __wrap_dlsym(void* handle, const char* symbol) {
    assert(handle == &library);
    if (std::strcmp(symbol, "nvmlInit_v2") == 0) { return reinterpret_cast<void*>(&init); }
    if (std::strcmp(symbol, "nvmlDeviceGetHandleByPciBusId_v2") == 0) { return reinterpret_cast<void*>(&device_by_pci); }
    if (std::strcmp(symbol, "nvmlDeviceGetComputeRunningProcesses_v3") == 0) { return reinterpret_cast<void*>(&processes); }
    return nullptr;
}
extern "C" cudaError_t CUDARTAPI cudaMemGetInfo(std::size_t* free, std::size_t* total) {
    *free = 10000 + mode; *total = 20000; return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaGetDevice(int* device) { *device = mode; return cudaSuccess; }
extern "C" cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char* output, int length, int) {
    if (mode == 6) { return cudaErrorInvalidDevice; }
    std::strncpy(output, "0000:01:00.0", length); return cudaSuccess;
}

int main() {
    using namespace sinfer;
    assert(sample_device_footprint().attributed);
    std::latch sampled(1), changed(1);
    std::thread first([&] {
        mode = 1;
        assert(!sample_device_footprint().attributed);
        const char* snapshot = device_footprint_attribution_note();
        sampled.count_down();
        changed.wait();
        assert(std::string_view(device_footprint_attribution_note()).find("no NVML device") != std::string_view::npos);
        assert(std::string_view(snapshot).find("no NVML device") != std::string_view::npos);
    });
    sampled.wait();
    assert(sample_device_footprint().attributed);
    changed.count_down();
    first.join();
    const char* expected[] = {"", "no NVML device", "no listed compute processes", "process list is unavailable",
                              "without a usage figure", "not in the device's list", "cudaDeviceGetPCIBusId failed"};
    std::vector<std::thread> readers;
    for (int worker = 0; worker < 8; ++worker) {
        readers.emplace_back([&, worker] {
            for (int step = 0; step < 1000; ++step) {
                mode = (worker + step) % 7;
                const auto sample = sample_device_footprint();
                assert(sample.attributed == (mode == 0));
                assert(sample.device_free_bytes == static_cast<std::size_t>(10000 + mode));
                assert(sample.process_used_bytes == (mode == 0 ? 1234U : 0U));
                const auto note = std::string_view(device_footprint_attribution_note());
                assert(mode == 0 ? note.empty() : note.find(expected[mode]) != std::string_view::npos);
            }
        });
    }
    for (auto& reader : readers) { reader.join(); }
    mode = 0;
    fail_allocation = true;
    assert(!sample_device_footprint().attributed);
    assert(!fail_allocation);
    assert(std::string_view(device_footprint_attribution_note()).find("allocation failed") != std::string_view::npos);
    assert(sample_device_footprint().attributed);
    std::cout << "NVML diagnostic isolation, stable note pointers and allocation failure recovery passed\n";
}
