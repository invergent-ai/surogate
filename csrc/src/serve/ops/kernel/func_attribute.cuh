#pragma once

// Kernel attributes (dynamic shared memory opt-in) are per device. A launcher that set them
// once under a function-local static configured only the first device it ran on; with
// pipeline stages on several devices the same kernel launches on every one of them, so the
// attribute is applied once per (device, kernel, attribute) here.

#include <cuda_runtime.h>

#include <cstddef>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <utility>

namespace sinfer::ops {

template <class Kernel>
inline cudaError_t set_func_attribute_per_device(Kernel* kernel, cudaFuncAttribute attribute, int value) {
    static std::mutex mutex;
    static std::set<std::tuple<int, const void*, int>> configured;
    int device        = 0;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) { return error; }
    const void* key = reinterpret_cast<const void*>(kernel);
    std::lock_guard<std::mutex> lock(mutex);
    if (configured.count({device, key, static_cast<int>(attribute)}) != 0) { return cudaSuccess; }
    error = cudaFuncSetAttribute(kernel, attribute, value);
    if (error == cudaSuccess) { configured.insert({device, key, static_cast<int>(attribute)}); }
    return error;
}

/// Device-wide co-resident CTAs of `kernel` at this block size and dynamic shared memory: the
/// largest grid a cooperative launch of it admits on the current device. Computed once per
/// (device, kernel), after the launcher has set the kernel's shared-memory attribute; 0 when the
/// device cannot be queried.
template <class Kernel>
inline int cooperative_capacity_per_device(Kernel* kernel, int threads, std::size_t dynamic_smem) {
    static std::mutex mutex;
    static std::map<std::pair<int, const void*>, int> capacity;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) { return 0; }
    const auto key = std::make_pair(device, reinterpret_cast<const void*>(kernel));
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto found = capacity.find(key); found != capacity.end()) { return found->second; }
    int per_sm = 0, sms = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, threads, dynamic_smem) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
        return 0;
    }
    return capacity[key] = per_sm * sms;
}

} // namespace sinfer::ops
