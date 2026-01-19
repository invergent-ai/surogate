// Minimal ATen atomicAdd shim for vendored kernels.
// This is NOT a full ATen implementation; it provides only the pieces needed
// by the selective_scan kernels (gpuAtomicAdd for float/double/complex).

#pragma once

#include <cuda_runtime.h>
#include <c10/util/complex.h>

__device__ __forceinline__ void gpuAtomicAdd(float* addr, float val) {
    atomicAdd(addr, val);
}

__device__ __forceinline__ void gpuAtomicAdd(double* addr, double val) {
    atomicAdd(addr, val);
}

__device__ __forceinline__ void gpuAtomicAdd(c10::complex<float>* addr, c10::complex<float> val) {
    atomicAdd(reinterpret_cast<float*>(addr) + 0, val.real_);
    atomicAdd(reinterpret_cast<float*>(addr) + 1, val.imag_);
}

__device__ __forceinline__ void gpuAtomicAdd(c10::complex<double>* addr, c10::complex<double> val) {
    atomicAdd(reinterpret_cast<double*>(addr) + 0, val.real_);
    atomicAdd(reinterpret_cast<double*>(addr) + 1, val.imag_);
}
