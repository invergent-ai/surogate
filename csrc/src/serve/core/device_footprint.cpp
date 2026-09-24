#include "core/device_footprint.h"

#include <cuda_runtime.h>
#include <nvml.h>

#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <mutex>
#include <vector>

#include <dlfcn.h>

namespace sinfer {
namespace {

// A diagnostic belongs to the caller's latest sample. Literal storage also keeps
// returned pointers valid across later samples, without allocating on failure paths.
thread_local const char* last_note = nullptr;

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
    const char* note = "NVML has not been probed";

    template <class Fn> bool bind(Fn& slot, const char* symbol, const char* missing) {
        slot = reinterpret_cast<Fn>(dlsym(handle, symbol));
        if (slot == nullptr) {
            note = missing;
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
        if (!bind(init, "nvmlInit_v2", "NVML is loaded but nvmlInit_v2 is missing") ||
            !bind(device_by_pci, "nvmlDeviceGetHandleByPciBusId_v2",
                  "NVML is loaded but nvmlDeviceGetHandleByPciBusId_v2 is missing")) {
            return;
        }
        // The v3 process list is the current one; v2 shares the field layout
        // this code reads, so an older driver still attributes correctly.
        if (!bind(compute_procs, "nvmlDeviceGetComputeRunningProcesses_v3",
                  "NVML is loaded but nvmlDeviceGetComputeRunningProcesses_v3 is missing") &&
            !bind(compute_procs, "nvmlDeviceGetComputeRunningProcesses_v2",
                  "NVML is loaded but nvmlDeviceGetComputeRunningProcesses_v2 is missing")) {
            return;
        }
        if (init() != NVML_SUCCESS) {
            note = "nvmlInit failed";
            return;
        }
        ready = true;
        note = "";
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
    if (!lib.ready) { last_note = lib.note; return false; }
    char bus_id[64] = {};
    if (cudaDeviceGetPCIBusId(bus_id, static_cast<int>(sizeof(bus_id)), cuda_device) !=
        cudaSuccess) {
        last_note = "cudaDeviceGetPCIBusId failed";
        return false;
    }
    if (lib.device_by_pci(bus_id, &out) != NVML_SUCCESS) {
        last_note = "no NVML device for this PCI bus id";
        return false;
    }
    return true;
}

bool process_used(int cuda_device, std::size_t& out) noexcept try {
    nvmlDevice_t device{};
    if (!device_handle(cuda_device, device)) { return false; }
    Nvml& lib = nvml();
    unsigned int count = 0;
    // The first call reports the count; INSUFFICIENT_SIZE is the expected
    // answer when there is at least one process.
    const nvmlReturn_t probe = lib.compute_procs(device, &count, nullptr);
    if (probe != NVML_SUCCESS && probe != NVML_ERROR_INSUFFICIENT_SIZE) {
        last_note = "the device's process list is unavailable (permission or namespace)";
        return false;
    }
    if (count == 0) { last_note = "this device has no listed compute processes"; return false; }
    // A process that starts on the card between the two calls makes the list outgrow the
    // count: ask again with room to spare rather than give up on the sample.
    std::vector<nvmlProcessInfo_t> processes;
    nvmlReturn_t listed = NVML_ERROR_INSUFFICIENT_SIZE;
    for (int attempt = 0; attempt < 4 && listed == NVML_ERROR_INSUFFICIENT_SIZE; ++attempt) {
        count += 8;
        processes.assign(count, nvmlProcessInfo_t{});
        listed = lib.compute_procs(device, &count, processes.data());
    }
    if (listed != NVML_SUCCESS) {
        last_note = "the device's process list is unavailable (permission or namespace)";
        return false;
    }
    const auto self = static_cast<unsigned int>(getpid());
    for (unsigned int i = 0; i < count; ++i) {
        if (processes[i].pid != self) { continue; }
        // A process that is on the device but whose usage the driver will not
        // report reads as NVML_VALUE_NOT_AVAILABLE, which is not a byte count.
        if (processes[i].usedGpuMemory == 0 ||
            processes[i].usedGpuMemory == static_cast<unsigned long long>(-1)) {
            last_note = "this process is listed without a usage figure";
            return false;
        }
        out = static_cast<std::size_t>(processes[i].usedGpuMemory);
        last_note = "";
        return true;
    }
    // Under a PID namespace the device lists host pids, so this process is not
    // in a list that nonetheless describes it.
    last_note = "this process is not in the device's list (PID namespace?)";
    return false;
} catch (...) {
    last_note = "NVML process list allocation failed";
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
    if (cudaGetDevice(&device) != cudaSuccess) { last_note = "cudaGetDevice failed"; return sample; }
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

const char* device_footprint_attribution_note() noexcept { return last_note != nullptr ? last_note : nvml().note; }

namespace {

struct DeviceLimit {
    std::size_t bytes         = 0;
    std::size_t baseline_free = 0; ///< fallback: free memory plus this process's usage, raised on every read
    std::size_t used          = 0; ///< the last usage figure
    bool attributed           = false; ///< NVML has attributed this process's usage on the device
    std::chrono::steady_clock::time_point sampled{};
    bool warned               = false;
};

// NVML's process list is a few driver calls; the budget is read under the elastic ledger's lock
// and on every /kv_stats and /metrics scrape, so a figure younger than this is reused.
constexpr auto kUsageRefresh = std::chrono::milliseconds(50);

// The fallback cannot see the CUDA context this process created before the limit was set.
constexpr std::size_t kFallbackContextBytes = std::size_t{512} << 20;

std::mutex& limit_mutex() {
    static std::mutex instance;
    return instance;
}

std::map<int, DeviceLimit>& limits() {
    static std::map<int, DeviceLimit> instance;
    return instance;
}

/// Free memory on `device`, selecting it for the query and restoring the caller's device.
std::size_t free_on(int device) noexcept {
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess) { previous = -1; }
    if (previous != device && cudaSetDevice(device) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    std::size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        (void)cudaGetLastError();
        free_bytes = 0;
    }
    if (previous >= 0 && previous != device) { (void)cudaSetDevice(previous); }
    return free_bytes;
}

} // namespace

void set_device_memory_limit(int device, std::size_t bytes) noexcept {
    const std::lock_guard<std::mutex> lock(limit_mutex());
    if (bytes == 0) {
        limits().erase(device);
        return;
    }
    DeviceLimit& entry = limits()[device];
    if (entry.bytes == bytes) { return; }
    entry               = DeviceLimit{};
    entry.bytes         = bytes;
    entry.baseline_free = free_on(device);
}

std::size_t device_memory_limit(int device) noexcept {
    const std::lock_guard<std::mutex> lock(limit_mutex());
    const auto it = limits().find(device);
    return it == limits().end() ? 0 : it->second.bytes;
}

std::size_t device_budget_free_bytes(int device) noexcept {
    const std::size_t free_bytes = free_on(device);
    const std::lock_guard<std::mutex> lock(limit_mutex());
    const auto it = limits().find(device);
    if (it == limits().end()) { return free_bytes; }
    DeviceLimit& entry = it->second;
    const auto now     = std::chrono::steady_clock::now();
    if (!entry.attributed || now - entry.sampled >= kUsageRefresh) {
        std::size_t used = 0;
        if (process_used(device, used)) {
            entry.used       = used;
            entry.attributed = true;
            entry.sampled    = now;
        } else if (!entry.attributed) {
            // No attribution on this device: what left the device's free memory since the limit
            // was set counts as this process's, plus the context it could not see. Memory another
            // process frees afterwards must not be credited to this one, so the baseline rises
            // with free memory and the figure never falls: a sizing budget errs small.
            entry.baseline_free = std::max(entry.baseline_free, free_bytes + entry.used);
            entry.used          = entry.baseline_free - free_bytes;
            if (!entry.warned) {
                entry.warned = true;
                std::fprintf(stderr,
                             "gpu memory limit: NVML cannot attribute this process's usage on device "
                             "%d (%s); counting every allocation there since the limit was set, "
                             "plus %zu MiB for the CUDA context\n",
                             device, device_footprint_attribution_note(),
                             kFallbackContextBytes >> 20);
            }
        }
        // Once NVML has attributed usage here, a failed sample keeps the last figure rather than
        // switching to the fallback's very different one.
    }
    const std::size_t used = entry.attributed ? entry.used : entry.used + kFallbackContextBytes;
    const std::size_t room = entry.bytes > used ? entry.bytes - used : 0;
    return std::min(free_bytes, room);
}

bool budgeted_mem_get_info(std::size_t& free_bytes, std::size_t& total_bytes) noexcept {
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) { return true; }
    const std::size_t limit = device_memory_limit(device);
    if (limit == 0) { return true; }
    free_bytes  = std::min(free_bytes, device_budget_free_bytes(device));
    total_bytes = std::min(total_bytes, limit);
    return true;
}

} // namespace sinfer
