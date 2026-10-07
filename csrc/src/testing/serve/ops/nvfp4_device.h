#pragma once

// NVFP4 weights run on the sm_12x devices the build carries FP4 cubins for (sm_120 under 120a,
// sm_121 under 121a, both under 120f): ops/linear/nvfp4/nvfp4_format.cpp refuses every other
// device (Ada and Hopper included), and the MoE runner is built for those targets alone. A test
// with NVFP4 cases asks this first and skips them elsewhere, keeping its other formats' cases.

#include "core/device.h"

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
    if (fp4_tensor_cores(major * 10 + minor)) { return true; }
    std::printf("SKIP %s: NVFP4 weights need an sm_120/sm_121 device this build has FP4 code for "
                "(this one is sm_%d%d)\n", what, major, minor);
    return false;
}

} // namespace sinfer::test
