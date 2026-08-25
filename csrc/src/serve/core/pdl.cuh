#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace ninfer::pdl {

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t dynamic_smem_bytes = 0;
    cudaStream_t stream            = nullptr;
};

// Launches a consumer kernel as a programmatic dependent of the immediately preceding producer
// kernel in the same stream. Every consumer control path that reads producer output must first call
// wait_for_dependencies().
// surogate vendor patch (csrc/src/serve/PATCHES.md): programmatic dependent
// launch (PDL) is sm_90+. On older devices (sm_89 port) launches fall back to
// ordinary stream ordering — semantically identical, without the overlap.
inline bool pdl_supported() {
    static const bool supported = [] {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) return false;
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) !=
            cudaSuccess)
            return false;
        return major >= 9;
    }();
    return supported;
}

template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline cudaError_t
launch_dependent(const LaunchConfig& launch, void (*kernel)(KernelArgs...), CallArgs&&... args) {
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config{};
    config.gridDim          = launch.grid;
    config.blockDim         = launch.block;
    config.dynamicSmemBytes = launch.dynamic_smem_bytes;
    config.stream           = launch.stream;
    if (pdl_supported()) {
        config.attrs    = &attribute;
        config.numAttrs = 1;
    }

    return cudaLaunchKernelEx(&config, kernel, std::forward<CallArgs>(args)...);
}

// Every producer CTA must call this at least once or exit. This enables dependent scheduling but
// does not make producer writes visible to the consumer.
__device__ __forceinline__ void trigger_dependents() {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Call on every consumer control path before its first access to producer-dependent data.
__device__ __forceinline__ void wait_for_dependencies() {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif
}

} // namespace ninfer::pdl
