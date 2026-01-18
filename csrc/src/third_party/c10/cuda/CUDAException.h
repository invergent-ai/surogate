// Minimal c10 CUDAException compatibility header for vendored kernels.
#pragma once

#include <cuda_runtime.h>
#include <cstdio>

#ifndef C10_CUDA_CHECK
#define C10_CUDA_CHECK(EXPR) do { \
    cudaError_t _err = (EXPR); \
    if (_err != cudaSuccess) { \
        printf("C10_CUDA_CHECK failed: %s (%s:%d)\n", cudaGetErrorString(_err), __FILE__, __LINE__); \
    } \
} while (0)
#endif

#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
#endif
