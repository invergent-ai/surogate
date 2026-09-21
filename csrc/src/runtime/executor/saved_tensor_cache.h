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
#include <cstdio>
#include <cstdlib>
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
        if (auto hit = mHostMirror.find(key); hit != mHostMirror.end()) {
            hit->second.valid = false;  // being rewritten
        }
        ArenaAlloc a = arena_alloc ? arena_alloc(bytes) : ArenaAlloc{};
        if (a.ptr == nullptr) {
            a.ptr = take_from_pool(bytes);  // reuse a pooled buffer before cudaMalloc
            if (a.ptr != nullptr) bytes = pool_take_bytes_;
        }
        if (a.ptr == nullptr) {
            const cudaError_t err = cudaMalloc(&a.ptr, bytes);
            if (err != cudaSuccess) {
                (void)cudaGetLastError();
                std::size_t free_b = 0, total_b = 0;
                cudaMemGetInfo(&free_b, &total_b);
                throw std::runtime_error(std::string(op_name ? op_name : "compiled_op") + ": cudaMalloc(" +
                                         std::to_string(bytes) + ") for saved tensor '" + key +
                                         "' failed: " + cudaGetErrorString(err) +
                                         " (saved-tensor cache already holds " + std::to_string(count()) +
                                         " buffers / " + std::to_string(total_plain_bytes()) +
                                         " plain bytes; device free " + std::to_string(free_b) + "/" +
                                         std::to_string(total_b) + ")");
            }
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
        if (auto hit = mHostMirror.find(key); hit != mHostMirror.end()) {
            hit->second.valid = false;
        }
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
        for (auto& [key, hm] : mHostMirror) {
            if (hm.ptr) cudaFreeHost(hm.ptr);
        }
        mHostMirror.clear();
        mBuffers.clear();
        mSizes.clear();
        mArenaBacked.clear();
        mPool.clear();
        mBumpOffset = 0;
    }

    /// Offload all of layer `layer_idx`'s plain (non-arena) buffers to pinned
    /// host mirrors and recycle the device buffers into the pool. Stream-ordered
    /// on `stream`: pool reuse happens via acquire() on the same stream, so the
    /// D2H completes before any overwrite. Enables seq scaling: saved tensors
    /// dominate step VRAM (~6.8 GB at seq 256 on Laguna-S, linear in tokens).
    void offload_layer(int layer_idx, cudaStream_t stream) {
        size_t moved_bytes=0,moved_count=0;
        const std::string prefix = "blocks[" + std::to_string(layer_idx) + "].";
        for (auto& [key, buf] : mBuffers) {
            if (buf == nullptr || is_arena_backed(key) || key.rfind(prefix, 0) != 0) continue;
            auto& hm = mHostMirror[key];
            const std::size_t bytes = mSizes[key];
            if (hm.capacity < bytes) {
                if (hm.ptr) CUDA_CHECK(cudaFreeHost(hm.ptr));
                CUDA_CHECK(cudaHostAlloc(&hm.ptr, bytes, cudaHostAllocDefault));
                hm.capacity = bytes;
            }
            CUDA_CHECK(cudaMemcpyAsync(hm.ptr, buf, bytes, cudaMemcpyDeviceToHost, stream));
            moved_bytes+=bytes;++moved_count;
            hm.bytes = bytes;
            hm.valid = true;
            mPool.push_back({buf, bytes});
            buf = nullptr;
        }
        if(std::getenv("JEV_SAVED_OFFLOAD_TRACE"))std::fprintf(stderr,"[saved-trace] offload layer=%d moved=%zu bytes=%zu remaining_count=%d remaining_bytes=%zu\n",layer_idx,moved_count,moved_bytes,count(),total_plain_bytes());
    }

    /// Restore layer `layer_idx`'s offloaded buffers to the device (called at
    /// backward layer start, before any of the layer's ops read them).
    void restore_layer(int layer_idx, cudaStream_t stream) {
        size_t moved_bytes=0,moved_count=0;
        const std::string prefix = "blocks[" + std::to_string(layer_idx) + "].";
        for (auto& [key, hm] : mHostMirror) {
            if (!hm.valid || key.rfind(prefix, 0) != 0) continue;
            auto bit = mBuffers.find(key);
            if (bit != mBuffers.end() && bit->second != nullptr) {
                hm.valid = false;  // device copy exists (e.g. replay re-acquired)
                continue;
            }
            std::size_t bytes = hm.bytes;
            void* dev = take_from_pool(bytes);
            if (dev != nullptr) {
                bytes = pool_take_bytes_;
            } else {
                CUDA_CHECK(cudaMalloc(&dev, bytes));
            }
            CUDA_CHECK(cudaMemcpyAsync(dev, hm.ptr, hm.bytes, cudaMemcpyHostToDevice, stream));
            moved_bytes+=hm.bytes;++moved_count;
            mBuffers[key] = dev;
            mSizes[key] = bytes;
            mArenaBacked[key] = false;
            hm.valid = false;
        }
        if(std::getenv("JEV_SAVED_OFFLOAD_TRACE"))std::fprintf(stderr,"[saved-trace] restore layer=%d moved=%zu bytes=%zu remaining_count=%d remaining_bytes=%zu\n",layer_idx,moved_count,moved_bytes,count(),total_plain_bytes());
    }

    // Only call after the graph has consumed every declared use of this layer.
    // Pool reuse follows the same MainStream ordering as offload_layer/acquire.
    // Future-dependent layers are NOT passed here; their device/host state is retained.
    std::size_t retire_layer(int layer_idx) {
        const std::string prefix="blocks["+std::to_string(layer_idx)+"].";
        std::size_t retired=0;
        for(auto& [key,buf]:mBuffers) {
            if(key.rfind(prefix,0)!=0 || is_arena_backed(key)) continue;
            if(buf) { retired+=mSizes[key]; mPool.push_back({buf,mSizes[key]}); buf=nullptr; }
            if(auto it=mHostMirror.find(key);it!=mHostMirror.end()) it->second.valid=false;
        }
        return retired;
    }
    // Exact compiled producer key, retired only after all owning-layer uses.
    std::size_t retire_key(const std::string& key) {
        auto it=mBuffers.find(key);
        if(it==mBuffers.end() || is_arena_backed(key)) return 0;
        std::size_t bytes=0;
        if(it->second) { bytes=mSizes[key]; mPool.push_back({it->second,bytes}); it->second=nullptr; }
        if(auto host=mHostMirror.find(key);host!=mHostMirror.end()) host->second.valid=false;
        return bytes;
    }
    struct Residency { std::size_t resident=0, host_valid=0, host_allocated=0, arena=0; };
    Residency layer_residency(int layer_idx) const {
        Residency r; const std::string prefix="blocks["+std::to_string(layer_idx)+"].";
        for(const auto& [key,buf]:mBuffers) if(key.rfind(prefix,0)==0 && buf) {
            if(is_arena_backed(key)) r.arena+=size_of(key); else r.resident+=size_of(key);
        }
        for(const auto& [key,hm]:mHostMirror) if(key.rfind(prefix,0)==0) {
            r.host_allocated+=hm.capacity; if(hm.valid) r.host_valid+=hm.bytes;
        }
        return r;
    }

    void trace_inventory(int device, int layer, const char* phase) const {
        if(!std::getenv("JEV_CACHE_INVENTORY_TRACE")) return;
        std::size_t active=0,unscoped=0,pool=0,largest=0;
        for(const auto& [key,buf]:mBuffers) {
            if(buf && !is_arena_backed(key)) {
                active+=size_of(key); if(key.rfind("blocks[",0)!=0) unscoped+=size_of(key);
                std::fprintf(stderr,"[cache-entry] device=%d layer=%d phase=%s bytes=%zu key=%s\n",device,layer,phase,size_of(key),key.c_str());
            }
        }
        for(const auto& [ptr,bytes]:mPool) { pool+=bytes;largest=std::max(largest,bytes);
            std::fprintf(stderr,"[cache-pool] device=%d layer=%d phase=%s bytes=%zu\n",device,layer,phase,bytes);
        }
        std::size_t free=0,total=0;CUDA_CHECK(cudaMemGetInfo(&free,&total));
        std::fprintf(stderr,"[cache-inventory] device=%d layer=%d phase=%s active=%zu unscoped=%zu pooled=%zu pool_count=%zu largest=%zu free=%zu total=%zu\n",device,layer,phase,active,unscoped,pool,mPool.size(),largest,free,total);
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
    struct HostMirror {
        void* ptr = nullptr;
        std::size_t bytes = 0;
        std::size_t capacity = 0;
        bool valid = false;
    };
    std::unordered_map<std::string, HostMirror> mHostMirror;  // pinned offload copies
    std::vector<std::pair<void*, std::size_t>> mPool;  // recycled buffers (dispatch per-stage)
    std::size_t pool_take_bytes_ = 0;
    std::size_t mBumpOffset = 0;
};

}  // namespace dsl

#endif  // SUROGATE_SRC_EXECUTOR_SAVED_TENSOR_CACHE_H
