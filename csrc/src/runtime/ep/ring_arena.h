// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Ring-slab arena for LLEP foreign-expert weight staging.
//
// Foreign expert weights (base + received LoRA slices) are fetched fresh at
// every LLEP dispatch and released at the next one. Allocating them with
// cudaMalloc/cudaFree costs ~200 driver-lock round-trips per layer per pass;
// with 8 in-process ranks that serializes into seconds per step. This arena
// replaces those with pointer bumps over a small ring of persistent slabs:
//
//   begin(stream)   opens a scope on the least-recently-used slot and makes
//                   `stream` wait until the slot's previous contents are dead
//   alloc(bytes)    bump-allocates in the open scope (grows by whole chunks;
//                   chunk count converges after the first pass, then zero
//                   driver calls in steady state)
//   release(slot,s) closes a scope: contents are dead once all work enqueued
//                   on `s` so far completes (event-guarded reuse). Pass a
//                   null stream only when the device is known idle (teardown).
//
// Single-threaded by design: one arena per EPStrategy, driven by one worker.

#ifndef SUROGATE_SRC_RUNTIME_EP_RING_ARENA_H
#define SUROGATE_SRC_RUNTIME_EP_RING_ARENA_H

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "utilities/utils.h"

namespace ep {

class RingSlabArena {
public:
    static constexpr int kNumSlots = 4;
    static constexpr std::size_t kMinChunkBytes = 32ull << 20;
    static constexpr std::size_t kAlign = 256;

    RingSlabArena() = default;
    RingSlabArena(const RingSlabArena&) = delete;
    RingSlabArena& operator=(const RingSlabArena&) = delete;

    /// Open an allocation scope. Returns the slot id the caller must later
    /// pass to release(). `stream` is ordered after the slot's prior release.
    int begin(cudaStream_t stream) {
        int slot = -1;
        for (int i = 1; i <= kNumSlots; ++i) {
            const int cand = (mCur + i) % kNumSlots;
            if (!mSlots[cand].active) {
                slot = cand;
                break;
            }
        }
        if (slot < 0) {
            throw std::runtime_error("RingSlabArena: all slots active (scope leak)");
        }
        Slot& s = mSlots[slot];
        if (s.released_valid) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, s.released));
            s.released_valid = false;
        }
        s.chunk_idx = 0;
        s.off = 0;
        s.active = true;
        mCur = slot;
        return slot;
    }

    /// Redirect alloc() to an already-active slot (adoption of a prefetched
    /// scope). Bump state is preserved — allocation continues where the
    /// prefetch left off.
    void reopen(int slot) {
        if (slot < 0 || slot >= kNumSlots || !mSlots[slot].active) {
            throw std::runtime_error("RingSlabArena: reopen of inactive slot");
        }
        mCur = slot;
    }

    /// Bump-allocate in the currently open scope.
    void* alloc(std::size_t bytes) {
        if (mCur < 0 || !mSlots[mCur].active) {
            throw std::runtime_error("RingSlabArena: alloc with no open scope");
        }
        Slot& s = mSlots[mCur];
        bytes = (bytes + kAlign - 1) & ~(kAlign - 1);
        while (s.chunk_idx < s.chunks.size()) {
            Chunk& c = s.chunks[s.chunk_idx];
            if (s.off + bytes <= c.cap) {
                void* p = static_cast<std::byte*>(c.base) + s.off;
                s.off += bytes;
                return p;
            }
            ++s.chunk_idx;
            s.off = 0;
        }
        Chunk c;
        c.cap = bytes > kMinChunkBytes ? bytes : kMinChunkBytes;
        CUDA_CHECK(cudaMalloc(&c.base, c.cap));
        s.chunks.push_back(c);
        s.chunk_idx = s.chunks.size() - 1;
        s.off = bytes;
        return c.base;
    }

    /// Close a scope. With a stream, reuse is deferred (via event) until all
    /// work enqueued on it so far completes; with nullptr the slot is reusable
    /// immediately (caller guarantees the device is idle).
    void release(int slot, cudaStream_t stream) {
        if (slot < 0 || slot >= kNumSlots) return;
        Slot& s = mSlots[slot];
        if (!s.active) return;
        s.active = false;
        if (stream) {
            if (!s.released) {
                CUDA_CHECK(cudaEventCreateWithFlags(&s.released, cudaEventDisableTiming));
            }
            CUDA_CHECK(cudaEventRecord(s.released, stream));
            s.released_valid = true;
        } else {
            s.released_valid = false;
        }
    }

    /// Free all slabs and events. Call only when the device is safe to touch;
    /// the destructor deliberately performs no CUDA calls (teardown may run
    /// with a dead context).
    void free_all() {
        for (Slot& s : mSlots) {
            for (Chunk& c : s.chunks) {
                if (c.base) cudaFree(c.base);
            }
            s.chunks.clear();
            if (s.released) {
                cudaEventDestroy(s.released);
                s.released = nullptr;
            }
            s.released_valid = false;
            s.active = false;
            s.chunk_idx = 0;
            s.off = 0;
        }
        mCur = -1;
    }

private:
    struct Chunk {
        void* base = nullptr;
        std::size_t cap = 0;
    };
    struct Slot {
        std::vector<Chunk> chunks;
        std::size_t chunk_idx = 0;
        std::size_t off = 0;
        cudaEvent_t released = nullptr;
        bool released_valid = false;
        bool active = false;
    };
    Slot mSlots[kNumSlots];
    int mCur = -1;
};

}  // namespace ep

#endif  // SUROGATE_SRC_RUNTIME_EP_RING_ARENA_H
