// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/ep/ep_buffer_pool.h"

#include <climits>
#include <cstddef>

#include <cuda_runtime.h>

#include "utilities/utils.h"

namespace ep {

EPBufferPool::~EPBufferPool() {
    clear_all();
}

void* EPBufferPool::acquire(std::size_t need) {
    // Zero-size acquires still return a real buffer: EP ranks can carry
    // 0-row tensors (a rank receiving no tokens after rebalancing — routine
    // for all-padding tail chunks under chunked-sequence training) whose
    // stored Data pointer must be non-null for downstream resolution.
    if (need == 0) need = 256;
    if (need == 0) return nullptr;

    // Best-fit: smallest buffer that satisfies the request.
    std::size_t best_size = SIZE_MAX;
    int best_idx = -1;
    for (std::size_t i = 0; i < mPool.size(); ++i) {
        if (mPool[i].bytes >= need && mPool[i].bytes < best_size) {
            best_size = mPool[i].bytes;
            best_idx = static_cast<int>(i);
        }
    }
    if (best_idx >= 0) {
        void* ptr = mPool[static_cast<std::size_t>(best_idx)].ptr;
        mPool.erase(mPool.begin() + best_idx);
        return ptr;
    }
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, need));
    return ptr;
}

void EPBufferPool::release(void* ptr, std::size_t bytes) {
    if (ptr && bytes > 0) {
        mPool.push_back({ptr, bytes});
    }
}

void EPBufferPool::retire(void* ptr, std::size_t bytes) {
    if (ptr && bytes > 0) {
        mRetired.push_back({ptr, bytes});
    }
}

void EPBufferPool::clear_retired() {
    for (auto& e : mRetired) {
        if (e.ptr) cudaFree(e.ptr);
    }
    mRetired.clear();
}

void EPBufferPool::clear_pool() {
    for (auto& e : mPool) {
        if (e.ptr) cudaFree(e.ptr);
    }
    mPool.clear();
}

void EPBufferPool::clear_all() {
    clear_retired();
    clear_pool();
}

}  // namespace ep
