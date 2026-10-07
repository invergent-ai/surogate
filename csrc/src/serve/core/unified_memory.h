#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace sinfer {

/// Free memory on a GPU that shares the host's DRAM.
///
/// On an integrated GPU (GB10 in the DGX Spark, Jetson) a device allocation comes out of system
/// memory, and cudaMemGetInfo's `free` is the kernel's MemFree, which leaves out the page cache.
/// cudaMalloc reclaims that cache as it allocates: on a DGX Spark, allocations ran about 30 GiB
/// past the reported free figure, up to MemAvailable less 8 GiB, with no swap. So right after a
/// model file has been read the device looks tens of GiB fuller than it is, and anything sized
/// to "free" leaves that memory idle. There the figure that means "what can still be allocated"
/// is MemAvailable, and since the host's own processes live in the same memory, a reserve stays
/// with them.
///
/// Header-only so the trainer and the serve engine read the same figure; it depends only on the
/// CUDA runtime.

/// MemAvailable from /proc/meminfo, in bytes. Zero when it cannot be read.
inline std::size_t host_available_bytes() noexcept {
    std::FILE* meminfo = std::fopen("/proc/meminfo", "r");
    if (meminfo == nullptr) { return 0; }
    char line[256];
    unsigned long long kib = 0;
    bool found             = false;
    while (std::fgets(line, sizeof(line), meminfo) != nullptr) {
        if (std::sscanf(line, "MemAvailable: %llu kB", &kib) == 1) {
            found = true;
            break;
        }
    }
    std::fclose(meminfo);
    return found ? static_cast<std::size_t>(kib) << 10 : 0;
}

/// System memory left to the host on an integrated GPU: the OS, this process's own host side
/// and anything else running on the machine. 8 GiB unless SUROGATE_UNIFIED_MEMORY_RESERVE_MIB
/// says otherwise.
inline std::size_t unified_memory_host_reserve_bytes() noexcept {
    static const std::size_t reserve = [] {
        if (const char* raw = std::getenv("SUROGATE_UNIFIED_MEMORY_RESERVE_MIB");
            raw != nullptr && *raw != '\0') {
            char* end                    = nullptr;
            const unsigned long long mib = std::strtoull(raw, &end, 10);
            if (end != raw && *end == '\0') { return static_cast<std::size_t>(mib) << 20; }
            std::fprintf(stderr,
                         "SUROGATE_UNIFIED_MEMORY_RESERVE_MIB=%s is not a whole number of MiB; "
                         "keeping 8192\n",
                         raw);
        }
        return std::size_t{8} << 30;
    }();
    return reserve;
}

/// Whether `device` allocates from system memory rather than memory of its own.
inline bool device_is_integrated(int device) noexcept {
    int integrated = 0;
    if (cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return integrated != 0;
}

/// cudaMemGetInfo for the current device, where `free` means what can still be allocated: on an
/// integrated GPU, MemAvailable less the host reserve (never more than `total`); on any other
/// GPU, or when /proc/meminfo cannot be read, the driver's figure unchanged.
inline cudaError_t device_mem_get_info(std::size_t* free_bytes, std::size_t* total_bytes) noexcept {
    const cudaError_t status = cudaMemGetInfo(free_bytes, total_bytes);
    if (status != cudaSuccess) { return status; }
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        (void)cudaGetLastError();
        return cudaSuccess;
    }
    if (!device_is_integrated(device)) { return cudaSuccess; }
    const std::size_t available = host_available_bytes();
    if (available == 0) { return cudaSuccess; }
    const std::size_t reserve = unified_memory_host_reserve_bytes();
    *free_bytes               = std::min(available > reserve ? available - reserve : 0, *total_bytes);
    return cudaSuccess;
}

} // namespace sinfer
