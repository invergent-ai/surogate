// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// SavedTensorCache: the single owner of persistent saved-tensor buffers used across
// forward/backward (gated-delta recurrent states, rope/qk-norm caches, MoE expert
// bookkeeping, and the SaveForBwd persist fallback). Historically these lived as three
// loose maps (mMoeSavedBuffers/Sizes/ArenaBacked) plus a bump offset, mutated directly by
// ~15 call sites with their own cudaMalloc/cudaFree -- which made the arena-vs-fallback
// free logic easy to get wrong (freeing an arena-backed pointer is a CUDA error).
//
// This class centralizes the lifecycle:
//   - acquire(): the one get-or-create. Its resize path frees the old buffer ONLY when it
//     is not arena-backed, so an arena pointer is never cudaFree'd.
//   - free_all(): the one place buffers are freed. Always arena-aware. Used at teardown,
//     and is the single hook a per-stage reset (dispatch-PP recompute:false) would call.
//   - find/size_of/put: thin accessors for callers that manage their own keys (MoE ops).
//
// "Saved" = persists across the forward-exit / backward-entry boundary. Arena-backed
// entries point into PhaseArenas and are owned there (never freed here).

#ifndef SUROGATE_SRC_EXECUTOR_SAVED_TENSOR_CACHE_H
#define SUROGATE_SRC_EXECUTOR_SAVED_TENSOR_CACHE_H

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "utilities/comm.h"  // CUDA_CHECK

namespace dsl {

class SavedTensorCache {
public:
    // Result of an arena allocation attempt: {ptr, arena_backed}. ptr==nullptr means the
    // arena couldn't serve the request and the cache should fall back to cudaMalloc.
    struct ArenaAlloc {
        void* ptr = nullptr;
        bool arena_backed = false;
    };
    using ArenaAllocFn = std::function<ArenaAlloc(std::size_t bytes)>;

    /// Get-or-create the buffer for `key`, sized for at least `bytes`. Reuses an existing
    /// buffer when it is large enough. On (re)allocation: frees the old buffer first iff it
    /// is a plain (non-arena) allocation, then tries `arena_alloc` (if provided) and falls
    /// back to cudaMalloc. Throws during CUDA graph capture if a large-enough buffer is
    /// absent (capture cannot cudaMalloc). Returns the device pointer.
    void* acquire(const std::string& key,
                  std::size_t bytes,
                  bool capturing,
                  const char* op_name,
                  const ArenaAllocFn& arena_alloc = {}) {
        auto it = mBuffers.find(key);
        if (it != mBuffers.end() && it->second != nullptr && mSizes[key] >= bytes) {
            return it->second;
        }
        if (capturing) {
            throw std::runtime_error(std::string(op_name ? op_name : "compiled_op") +
                                     ": missing preallocated saved buffer for '" + key +
                                     "' during CUDA graph capture");
        }
        // Recycle the prior plain buffer into the free-list (cudaFree is a device-sync
        // point -- pooling avoids that churn when stages reset and re-acquire). Arena
        // pointers are owned by PhaseArenas; just drop the key.
        if (it != mBuffers.end() && it->second != nullptr && !is_arena_backed(key)) {
            mPool.push_back({it->second, mSizes[key]});
        }
        ArenaAlloc a = arena_alloc ? arena_alloc(bytes) : ArenaAlloc{};
        if (a.ptr == nullptr) {
            a.ptr = take_from_pool(bytes);  // reuse a pooled buffer before cudaMalloc
            if (a.ptr != nullptr) bytes = pool_take_bytes_;
        }
        if (a.ptr == nullptr) {
            CUDA_CHECK(cudaMalloc(&a.ptr, bytes));
            a.arena_backed = false;
        }
        mBuffers[key] = a.ptr;
        mSizes[key] = bytes;
        mArenaBacked[key] = a.arena_backed;
        return a.ptr;
    }

    /// Record an externally-produced buffer for `key` (e.g. a slot the caller resolved in
    /// an arena). Frees a prior plain allocation under the same key first.
    void put(const std::string& key, void* ptr, std::size_t bytes, bool arena_backed) {
        auto it = mBuffers.find(key);
        if (it != mBuffers.end() && it->second != nullptr && it->second != ptr && !is_arena_backed(key)) {
            CUDA_CHECK(cudaFree(it->second));
        }
        mBuffers[key] = ptr;
        mSizes[key] = bytes;
        mArenaBacked[key] = arena_backed;
    }

    /// Device pointer for `key`, or nullptr if absent.
    void* find(const std::string& key) const {
        auto it = mBuffers.find(key);
        return it == mBuffers.end() ? nullptr : it->second;
    }

    bool contains(const std::string& key) const {
        return mBuffers.find(key) != mBuffers.end();
    }

    std::size_t size_of(const std::string& key) const {
        auto it = mSizes.find(key);
        return it == mSizes.end() ? 0 : it->second;
    }

    bool is_arena_backed(const std::string& key) const {
        auto it = mArenaBacked.find(key);
        return it != mArenaBacked.end() && it->second;
    }

    /// Recycle all active plain buffers into the free-list and clear the keys + bump,
    /// WITHOUT cudaFree. dispatch-PP calls this per resident stage: the next stage's
    /// re-acquire pops these buffers back (uniform layers -> same sizes -> exact reuse), so
    /// memory is bounded to one stage's worth and there is no cudaFree/cudaMalloc churn.
    /// Arena-backed entries are owned by PhaseArenas (just dropped; the bump reset reuses
    /// the slab).
    void reset_to_pool() noexcept {
        for (auto& [key, buffer] : mBuffers) {
            if (buffer != nullptr && !is_arena_backed(key)) {
                mPool.push_back({buffer, mSizes[key]});
            }
        }
        mBuffers.clear();
        mSizes.clear();
        mArenaBacked.clear();
        mBumpOffset = 0;
    }

    /// Free every plain (non-arena) buffer, the free-list, and clear all keys + bump. The
    /// single place these buffers are released -- always arena-aware. Safe in a destructor.
    void free_all() noexcept {
        for (auto& [key, buffer] : mBuffers) {
            if (buffer != nullptr && !is_arena_backed(key)) {
                cudaFree(buffer);
            }
        }
        for (auto& [buffer, sz] : mPool) {
            if (buffer != nullptr) cudaFree(buffer);
        }
        mBuffers.clear();
        mSizes.clear();
        mArenaBacked.clear();
        mPool.clear();
        mBumpOffset = 0;
    }

    /// Bump offset into the (caller-owned) arena, for arena allocators that pack many
    /// saved tensors into one slab.
    std::size_t& bump_offset() {
        return mBumpOffset;
    }

    // Diagnostics (saved_buffers_* interface).
    int count() const {
        return static_cast<int>(mBuffers.size());
    }
    std::size_t total_plain_bytes() const {
        std::size_t total = 0;
        for (const auto& [key, bytes] : mSizes) {
            if (!is_arena_backed(key)) total += bytes;
        }
        return total;
    }

    // Raw-map accessors. Callers that manage their own get-or-create / resize keep using
    // these maps directly; the cache still owns them and free_all() remains the single,
    // arena-aware release point. Prefer acquire()/put()/free_all() for new code.
    std::unordered_map<std::string, void*>& buffers() {
        return mBuffers;
    }
    std::unordered_map<std::string, std::size_t>& sizes() {
        return mSizes;
    }
    const std::unordered_map<std::string, std::size_t>& sizes() const {
        return mSizes;
    }
    std::unordered_map<std::string, bool>& arena_backed() {
        return mArenaBacked;
    }

private:
    // Pop a recycled buffer of at least `bytes` (best fit) from the free-list. Returns
    // nullptr if none; on success sets pool_take_bytes_ to the popped buffer's real size.
    void* take_from_pool(std::size_t bytes) {
        std::size_t best = mPool.size();
        for (std::size_t i = 0; i < mPool.size(); ++i) {
            if (mPool[i].second >= bytes && (best == mPool.size() || mPool[i].second < mPool[best].second)) {
                best = i;
            }
        }
        if (best == mPool.size()) return nullptr;
        void* ptr = mPool[best].first;
        pool_take_bytes_ = mPool[best].second;
        mPool[best] = mPool.back();
        mPool.pop_back();
        return ptr;
    }

    std::unordered_map<std::string, void*> mBuffers;
    std::unordered_map<std::string, std::size_t> mSizes;
    std::unordered_map<std::string, bool> mArenaBacked;
    std::vector<std::pair<void*, std::size_t>> mPool;  // recycled buffers (dispatch per-stage)
    std::size_t pool_take_bytes_ = 0;
    std::size_t mBumpOffset = 0;
};

}  // namespace dsl

#endif  // SUROGATE_SRC_EXECUTOR_SAVED_TENSOR_CACHE_H
