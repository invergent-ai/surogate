#pragma once

// Kernel attributes (dynamic shared memory opt-in) are per device. A launcher that set them
// once under a function-local static configured only the first device it ran on; with
// pipeline stages on several devices the same kernel launches on every one of them, so the
// attribute is applied once per (device, kernel, attribute) here.

#include <cuda_runtime.h>

#include <mutex>
#include <set>
#include <tuple>

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

} // namespace sinfer::ops
