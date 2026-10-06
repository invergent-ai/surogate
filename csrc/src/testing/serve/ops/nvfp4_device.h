#pragma once

// NVFP4 weights run on sm_120 only: ops/linear/nvfp4/nvfp4_format.cpp refuses every other
// device (Ada and Hopper included), and the MoE runner is built for 120a alone. A test with
// NVFP4 cases asks this first and skips them elsewhere, keeping its other formats' cases.

#include <cuda_runtime.h>

#include <cstdio>

namespace sinfer::test {

inline bool nvfp4_device(const char* what) {
    int device = 0, major = 0, minor = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
        return false;
    }
    if (major == 12 && minor == 0) { return true; }
    std::printf("SKIP %s: NVFP4 weights need an sm_120 device (this one is sm_%d%d)\n", what, major, minor);
    return false;
}

} // namespace sinfer::test
