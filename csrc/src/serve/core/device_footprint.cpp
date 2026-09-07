#include "core/device_footprint.h"

#include <cuda_runtime.h>
#include <nvml.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unistd.h>
#include <vector>

#include <dlfcn.h>

namespace sinfer {
namespace {

// NVML is loaded at runtime rather than linked. The CUDA toolkit ships a stub
// libnvidia-ml.so for linking, and linking it makes the build depend on which
// copy the loader finds first; the driver's own library is the one that can
// answer, and dlopen names it exactly. It is also genuinely optional: without
// it the engine still runs, it just cannot separate its own allocations from
// another process's.
struct Nvml {
    void* handle                                                                    = nullptr;
    nvmlReturn_t (*init)()                                                          = nullptr;
    nvmlReturn_t (*device_by_pci)(const char*, nvmlDevice_t*)                       = nullptr;
    nvmlReturn_t (*compute_procs)(nvmlDevice_t, unsigned int*, nvmlProcessInfo_t*)  = nullptr;
    bool ready                                                                      = false;
    std::string note = "NVML has not been probed";

    template <class Fn> bool bind(Fn& slot, const char* symbol) {
        slot = reinterpret_cast<Fn>(dlsym(handle, symbol));
        if (slot == nullptr) {
            note = std::string("NVML is loaded but ") + symbol + " is missing";
            return false;
        }
        return true;
    }

    Nvml() {
        handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (handle == nullptr) {
            note = "libnvidia-ml.so.1 is not loadable";
            return;
        }
        if (!bind(init, "nvmlInit_v2") || !bind(device_by_pci, "nvmlDeviceGetHandleByPciBusId_v2")) {
            return;
        }
        // The v3 process list is the current one; v2 shares the field layout
        // this code reads, so an older driver still attributes correctly.
        if (!bind(compute_procs, "nvmlDeviceGetComputeRunningProcesses_v3") &&
            !bind(compute_procs, "nvmlDeviceGetComputeRunningProcesses_v2")) {
            return;
        }
        if (init() != NVML_SUCCESS) {
            note = "nvmlInit failed";
            return;
        }
        ready = true;
        note.clear();
    }
};

Nvml& nvml() {
    static Nvml instance;
    return instance;
}

// The NVML handle for a CUDA device, resolved through the PCI bus id so the two
// libraries' device orderings cannot disagree (CUDA_VISIBLE_DEVICES reorders
// one and not the other).
bool device_handle(int cuda_device, nvmlDevice_t& out) noexcept {
    Nvml& lib = nvml();
    if (!lib.ready) { return false; }
    char bus_id[64] = {};
    if (cudaDeviceGetPCIBusId(bus_id, static_cast<int>(sizeof(bus_id)), cuda_device) !=
        cudaSuccess) {
        lib.note = "cudaDeviceGetPCIBusId failed";
        return false;
    }
    if (lib.device_by_pci(bus_id, &out) != NVML_SUCCESS) {
        lib.note = "no NVML device for this PCI bus id";
        return false;
    }
    return true;
}

bool process_used(int cuda_device, std::size_t& out) noexcept {
    nvmlDevice_t device{};
    if (!device_handle(cuda_device, device)) { return false; }
    Nvml& lib = nvml();
    unsigned int count = 0;
    // The first call reports the count; INSUFFICIENT_SIZE is the expected
    // answer when there is at least one process.
    const nvmlReturn_t probe = lib.compute_procs(device, &count, nullptr);
    if (probe != NVML_SUCCESS && probe != NVML_ERROR_INSUFFICIENT_SIZE) {
        lib.note = "the device's process list is unavailable (permission or namespace)";
        return false;
    }
    if (count == 0) { return false; }
    std::vector<nvmlProcessInfo_t> processes(count);
    if (lib.compute_procs(device, &count, processes.data()) != NVML_SUCCESS) {
        lib.note = "the device's process list is unavailable (permission or namespace)";
        return false;
    }
    const auto self = static_cast<unsigned int>(getpid());
    for (unsigned int i = 0; i < count; ++i) {
        if (processes[i].pid != self) { continue; }
        // A process that is on the device but whose usage the driver will not
        // report reads as NVML_VALUE_NOT_AVAILABLE, which is not a byte count.
        if (processes[i].usedGpuMemory == 0 ||
            processes[i].usedGpuMemory == static_cast<unsigned long long>(-1)) {
            lib.note = "this process is listed without a usage figure";
            return false;
        }
        out = static_cast<std::size_t>(processes[i].usedGpuMemory);
        lib.note.clear();
        return true;
    }
    // Under a PID namespace the device lists host pids, so this process is not
    // in a list that nonetheless describes it.
    lib.note = "this process is not in the device's list (PID namespace?)";
    return false;
}

} // namespace

DeviceFootprint sample_device_footprint() noexcept {
    DeviceFootprint sample;
    std::size_t total = 0;
    if (cudaMemGetInfo(&sample.device_free_bytes, &total) != cudaSuccess) {
        sample.device_free_bytes = 0;
    }
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) { return sample; }
    sample.attributed = process_used(device, sample.process_used_bytes);
    return sample;
}

DeviceFootprintDelta device_footprint_delta(const DeviceFootprint& before,
                                            const DeviceFootprint& after) noexcept {
    if (before.attributed && after.attributed) {
        return {after.process_used_bytes > before.process_used_bytes
                    ? after.process_used_bytes - before.process_used_bytes
                    : 0,
                true};
    }
    return {before.device_free_bytes > after.device_free_bytes
                ? before.device_free_bytes - after.device_free_bytes
                : 0,
            false};
}

const char* device_footprint_attribution_note() noexcept { return nvml().note.c_str(); }

} // namespace sinfer
