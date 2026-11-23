// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Pinned host blocks recycled by size, and a byte buffer on top of them, for bytes that cross the PCIe bus every step
// (the dispatch-PP stage boundaries). cudaMemcpyAsync to or from pinned memory is one DMA at bus speed; pageable
// memory is staged through the driver's bounce buffer and blocks the calling thread.

#pragma once

#include <cstddef>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace surogate {

/// Blocks stay pinned until the process exits. Past SUROGATE_PINNED_POOL_GIB (default 32) of pinned blocks, or when
/// pinning fails, a block is plain pageable memory: a step with many long rows keeps working, at the old copy speed.
class PinnedPool {
public:
    static PinnedPool& instance() {
        static PinnedPool* pool = new PinnedPool();  // never destroyed: blocks outlive the CUDA context at exit
        return *pool;
    }

    void* acquire(std::size_t bytes) {
        if (bytes == 0) return nullptr;
        std::lock_guard<std::mutex> lock(mMutex);
        auto it = mFree.find(bytes);
        if (it != mFree.end() && !it->second.empty()) {
            void* p = it->second.back();
            it->second.pop_back();
            return p;
        }
        if (mPinnedBytes + bytes <= mBudget) {
            void* p = nullptr;
            if (cudaHostAlloc(&p, bytes, cudaHostAllocPortable) == cudaSuccess) {
                mPinnedBytes += bytes;
                return p;
            }
            (void)cudaGetLastError();
        }
        void* p = ::operator new(bytes);
        mPageable.insert(p);
        return p;
    }

    void release(void* p, std::size_t bytes) noexcept {
        if (!p) return;
        std::lock_guard<std::mutex> lock(mMutex);
        if (mPageable.erase(p)) {
            ::operator delete(p);
            return;
        }
        mFree[bytes].push_back(p);
    }

private:
    PinnedPool() {
        const char* env = std::getenv("SUROGATE_PINNED_POOL_GIB");
        const double gib = env ? std::atof(env) : 32.0;
        mBudget = static_cast<std::size_t>((gib > 0.0 ? gib : 0.0) * static_cast<double>(1ull << 30));
    }
    std::mutex mMutex;
    std::map<std::size_t, std::vector<void*>> mFree;
    std::unordered_set<void*> mPageable;
    std::size_t mPinnedBytes = 0;
    std::size_t mBudget = 0;
};

/// A pinned byte block, written once (filled by a device-to-host copy) and only read afterwards. Copies share the
/// block instead of duplicating it: a stage boundary is both injected into the next stage and kept for the train
/// step, and copying 32 MB per unit on the host was a cost on the pipeline's critical path.
class PinnedBytes {
public:
    PinnedBytes() = default;
    explicit PinnedBytes(std::size_t bytes) { resize(bytes); }

    /// A fresh block of `bytes` (contents undefined). The previous block is released once no copy shares it.
    void resize(std::size_t bytes) {
        if (bytes == 0) {
            mBlock.reset();
            return;
        }
        auto* data = static_cast<std::byte*>(PinnedPool::instance().acquire(bytes));
        mBlock = std::shared_ptr<Block>(new Block{data, bytes});
    }
    std::byte* data() noexcept { return mBlock ? mBlock->data : nullptr; }
    const std::byte* data() const noexcept { return mBlock ? mBlock->data : nullptr; }
    std::size_t size() const noexcept { return mBlock ? mBlock->size : 0; }
    bool empty() const noexcept { return size() == 0; }

private:
    struct Block {
        std::byte* data;
        std::size_t size;
        ~Block() { PinnedPool::instance().release(data, size); }
    };
    std::shared_ptr<Block> mBlock;
};

}  // namespace surogate
