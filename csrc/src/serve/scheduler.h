// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Unified serving scheduler — vLLM v1 style.
//
// Single token budget per step shared by decode (1 token/req) and chunked
// prefill (up to long_prefill_token_threshold tokens/req).  No separate
// "prefill phase" or "decode phase".

#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace serve {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct SchedulerConfig {
    int max_num_batched_tokens = 2048;    // unified token budget per step
    int max_num_seqs = 64;                // max concurrent sequences
    int long_prefill_token_threshold = 0; // 0 = auto (4% of max_seq_len)
    int max_num_partial_prefills = 1;     // max NEW requests admitted per step
    int max_seq_len = 4096;               // model max sequence length
    bool enable_chunked_prefill = true;   // allow partial prefills
};

// ---------------------------------------------------------------------------
// Lightweight per-slot view used by the scheduler.
// The server populates this from its ActiveGeneration for each slot.
// ---------------------------------------------------------------------------

struct SlotView {
    bool finished = false;
    bool is_prefill_chunk = true;
    int num_prompt_tokens = 0;
    int num_computed_tokens = 0;
};

// ---------------------------------------------------------------------------
// Scheduler output — what to run this step
// ---------------------------------------------------------------------------

struct SchedulerOutput {
    // Indices into the pending queue for newly admitted requests.
    // The server pops these from pending_ and creates slots.
    int num_new_admitted = 0;

    // Total tokens scheduled across all requests this step.
    int total_scheduled_tokens = 0;

    // Prefill chunk size to pass to the engine (= resolved threshold).
    int prefill_chunk_size = 512;

    // True if there are any active requests (running or newly admitted).
    bool has_work = false;
};

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

class Scheduler {
public:
    explicit Scheduler(SchedulerConfig cfg);

    /// Main scheduling pass.
    ///
    /// @param slots       Per-slot views of running requests (populated by server).
    /// @param num_slots   Number of running slots.
    /// @param pending_size Number of requests in the pending queue.
    /// @param free_slots  Number of free engine slots.
    /// @param pending_prompt_lens  Prompt lengths of pending requests (front to back).
    SchedulerOutput schedule(
        const SlotView* slots,
        int num_slots,
        int pending_size,
        int free_slots,
        const int* pending_prompt_lens,
        int num_pending);

    /// Resolved long prefill threshold (auto-computed if config is 0).
    int long_prefill_threshold() const { return resolved_threshold_; }

    /// The config.
    const SchedulerConfig& config() const { return cfg_; }

private:
    SchedulerConfig cfg_;
    int resolved_threshold_;
};

}  // namespace serve
