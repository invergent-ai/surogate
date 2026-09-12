// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#ifndef SUROGATE_SRC_KERNELS_MOE_MOE_COMMON_CUH
#define SUROGATE_SRC_KERNELS_MOE_MOE_COMMON_CUH

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "kernels/kernels.h"
#include "kernels/kernel_utils.cuh"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

template <typename T>
constexpr cudaDataType_t cublas_dtype() {
    if constexpr (std::is_same_v<T, float>)
        return CUDA_R_32F;
    else if constexpr (std::is_same_v<T, nv_bfloat16>)
        return CUDA_R_16BF;
    else if constexpr (std::is_same_v<T, half>)
        return CUDA_R_16F;
    else
        static_assert(!sizeof(T), "Unsupported type for cuBLAS");
}

/// Device-resident pointer arrays for cublasGemmGroupedBatchedEx without a
/// per-call allocation. The three arrays are staged through a ring of pinned
/// slots (one H2D copy per call) into a matching ring of device slots. Slots
/// are reused one generation (kSlots calls) later; one event recorded after
/// the generation's last copy is waited on before the next generation begins,
/// which only blocks when the host has run more than kSlots grouped GEMMs
/// ahead of the device. The device arrays are reused in stream order after the
/// GEMM that read them. The ring is per thread: one worker thread per GPU, one
/// stream each; a stream change drains the previous stream first.
template <typename T>
struct MoeGroupedPtrArrays {
    const T** A = nullptr;
    const T** B = nullptr;
    T** C = nullptr;
};

namespace moe_ptr_ring {
struct Slot {
    void* pinned = nullptr;
    void* device = nullptr;
    std::size_t capacity = 0;  // bytes per array; A, B, C laid out consecutively
};
struct Ring {
    static constexpr int kSlots = 256;
    Slot slots[kSlots];
    int next = 0;
    cudaStream_t stream = nullptr;          // stream the current generation copies on
    cudaEvent_t generation_done = nullptr;  // recorded after the generation's last copy
    ~Ring() {
        // Best effort at thread exit: the driver may already be torn down.
        if (generation_done) (void)cudaEventDestroy(generation_done);
        for (auto& s : slots) {
            if (s.pinned) (void)cudaFreeHost(s.pinned);
            if (s.device) (void)cudaFree(s.device);
        }
    }
};
inline Ring& ring() {
    static thread_local Ring r;
    return r;
}
}  // namespace moe_ptr_ring

template <typename T>
inline MoeGroupedPtrArrays<T> stage_moe_grouped_ptr_arrays(const std::vector<const T*>& A_vec,
                                                          const std::vector<const T*>& B_vec,
                                                          const std::vector<T*>& C_vec,
                                                          cudaStream_t stream) {
    using moe_ptr_ring::Ring;
    auto& ring = moe_ptr_ring::ring();
    if (ring.stream && ring.stream != stream) {
        // Another stream takes over this thread's ring: drain the old one so no
        // copy of the previous generation can still be reading a pinned slot.
        CUDA_CHECK(cudaStreamSynchronize(ring.stream));
        ring.stream = nullptr;
    }
    if (ring.next == 0 && ring.stream && ring.generation_done) {
        // Every copy of the previous generation precedes this event in stream order.
        CUDA_CHECK(cudaEventSynchronize(ring.generation_done));
    }
    ring.stream = stream;
    auto& slot = ring.slots[ring.next];
    const std::size_t count = A_vec.size();
    const std::size_t array_bytes = (count * sizeof(T*) + 255) / 256 * 256;
    if (slot.capacity < array_bytes) {
        // Growth is rare (the expert count is fixed per model); cudaFree waits
        // for the device, which also covers the GEMM that last read the slot.
        if (slot.pinned) CUDA_CHECK(cudaFreeHost(slot.pinned));
        if (slot.device) CUDA_CHECK(cudaFree(slot.device));
        const std::size_t cap = std::max<std::size_t>(array_bytes, 4096);
        CUDA_CHECK(cudaHostAlloc(&slot.pinned, 3 * cap, cudaHostAllocDefault));
        CUDA_CHECK(cudaMalloc(&slot.device, 3 * cap));
        slot.capacity = cap;
    }
    auto* host = static_cast<std::byte*>(slot.pinned);
    auto* dev = static_cast<std::byte*>(slot.device);
    std::memcpy(host, A_vec.data(), count * sizeof(T*));
    std::memcpy(host + slot.capacity, B_vec.data(), count * sizeof(T*));
    std::memcpy(host + 2 * slot.capacity, C_vec.data(), count * sizeof(T*));
    CUDA_CHECK(cudaMemcpyAsync(dev, host, 3 * slot.capacity, cudaMemcpyHostToDevice, stream));
    ring.next = (ring.next + 1) % Ring::kSlots;
    if (ring.next == 0) {
        if (!ring.generation_done) {
            CUDA_CHECK(cudaEventCreateWithFlags(&ring.generation_done, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventRecord(ring.generation_done, stream));
    }
    MoeGroupedPtrArrays<T> out;
    out.A = reinterpret_cast<const T**>(dev);
    out.B = reinterpret_cast<const T**>(dev + slot.capacity);
    out.C = reinterpret_cast<T**>(dev + 2 * slot.capacity);
    return out;
}

#endif  // SUROGATE_SRC_KERNELS_MOE_MOE_COMMON_CUH
