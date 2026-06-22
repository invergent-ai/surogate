// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Process-global store for FROZEN base master weights shared across the per-GPU
// weight managers (dispatch-PP, offload_master + LoRA). Without it each of the N
// GPU managers pins its own copy of the (read-only, frozen) base in host memory:
// N x cudaHostAlloc of the whole model -> N x the pinning time AND N x the RAM.
//
// Instead the base is allocated ONCE (keyed by tensor name) and every GPU manager
// points its master at the same host buffer (pinned host memory is accessible from
// every GPU, so all GPUs can DMA-stream from the one copy). Population (file read +
// pinning) is done exactly once per tensor under a populate-once latch; the buffer
// is page-locked with cudaHostRegister AFTER the read (registering already-resident
// anonymous memory is ~3x faster than cudaHostAlloc's allocate+zero+pin, and skips
// the wasted zero-fill that the subsequent read overwrites).
//
// Only valid for FROZEN masters (LoRA base): they are read once and never written,
// so sharing one copy across GPUs is safe. NOT for trainable/updated masters.

#ifndef SUROGATE_SRC_RUNTIME_DSL_SHARED_MASTER_STORE_H
#define SUROGATE_SRC_RUNTIME_DSL_SHARED_MASTER_STORE_H

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>

namespace dsl {

class SharedMasterStore {
public:
    /// Reserve (or look up) the shared host buffer for `name`. The first caller
    /// allocates `bytes` of pageable host memory (lazy; instant); later callers
    /// for the same name get the same pointer. Thread-safe.
    void* reserve(const std::string& name, std::size_t bytes);

    /// Try to become the single populator of `name`. Returns true for exactly one
    /// caller (which must then read the data, call register_and_finish); all other
    /// callers get false and should wait_populated() then reuse the buffer.
    bool try_claim(const std::string& name);

    /// Mark `name` populated and page-lock its buffer for DMA streaming. Called by
    /// the populator after the file read completes (data resident). Wakes waiters.
    void register_and_finish(const std::string& name);

    /// Block until `name` has been populated by its claimer.
    void wait_populated(const std::string& name);

    /// True if `name` has a reserved shared buffer.
    bool has(const std::string& name) const;

    /// Unregister + free every shared buffer. Call at trainer teardown, after all
    /// GPU work that streams from these buffers has completed.
    void clear();

private:
    struct Entry {
        void* data = nullptr;
        std::size_t bytes = 0;
        bool claimed = false;
        bool populated = false;
    };
    mutable std::mutex mMutex;
    std::condition_variable mCond;
    std::unordered_map<std::string, Entry> mEntries;
};

/// Process-global instance (one model loaded at a time in the dispatch-PP path).
SharedMasterStore& shared_master_store();

}  // namespace dsl

#endif  // SUROGATE_SRC_RUNTIME_DSL_SHARED_MASTER_STORE_H
