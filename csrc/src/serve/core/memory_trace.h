#pragma once

// SUROGATE_SERVE_MEM_TRACE=1: `mem-trace` lines on stderr that follow free memory through a
// load. Automatic KV sizing reads free memory once and plans the rest of startup against it, so
// anything startup allocates that the plan does not count shows up here as free memory falling
// faster than the plan says it should. On a GPU that shares the host's memory (GB10) the
// resident set is in the same pool, so it is printed too.

#include "core/elastic_kv_region.h"
#include "core/unified_memory.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace sinfer {

[[nodiscard]] inline bool memory_trace_enabled() noexcept {
    static const bool enabled = std::getenv("SUROGATE_SERVE_MEM_TRACE") != nullptr;
    return enabled;
}

/// This process's resident set (VmRSS), in bytes. Zero when /proc cannot be read.
[[nodiscard]] inline std::size_t process_resident_bytes() noexcept {
    std::FILE* status = std::fopen("/proc/self/status", "r");
    if (status == nullptr) { return 0; }
    char line[256];
    unsigned long long kib = 0;
    while (std::fgets(line, sizeof(line), status) != nullptr) {
        if (std::sscanf(line, "VmRSS: %llu kB", &kib) == 1) { break; }
    }
    std::fclose(status);
    return static_cast<std::size_t>(kib) << 10;
}

/// One line for `phase`: free memory as the sizing reads it (device_mem_get_info), the KV the
/// elastic regions hold mapped, and the resident set. Silent unless the trace is enabled.
inline void trace_memory_phase(const char* phase) noexcept {
    if (!memory_trace_enabled()) { return; }
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        (void)cudaGetLastError();
        return;
    }
    std::size_t free_bytes = 0, total_bytes = 0;
    if (device_mem_get_info(&free_bytes, &total_bytes) != cudaSuccess) {
        (void)cudaGetLastError();
        return;
    }
    std::fprintf(stderr, "mem-trace %s: free=%zu MiB kv-mapped=%zu MiB rss=%zu MiB\n", phase,
                 free_bytes >> 20, elastic_kv_mapped_bytes(device) >> 20,
                 process_resident_bytes() >> 20);
    std::fflush(stderr);
}

} // namespace sinfer
