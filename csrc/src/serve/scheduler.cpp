// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "serve/scheduler.h"

#include <algorithm>

namespace serve {

Scheduler::Scheduler(SchedulerConfig cfg) : cfg_(std::move(cfg)) {
    if (cfg_.long_prefill_token_threshold > 0) {
        resolved_threshold_ = cfg_.long_prefill_token_threshold;
    } else {
        // vLLM default: 4% of max_model_len, minimum 64.
        resolved_threshold_ = std::max(64, cfg_.max_seq_len * 4 / 100);
    }
}

SchedulerOutput Scheduler::schedule(
        const SlotView* slots,
        int num_slots,
        int pending_size,
        int free_slots,
        const int* pending_prompt_lens,
        int num_pending) {

    SchedulerOutput output;
    output.prefill_chunk_size = resolved_threshold_;

    int token_budget = cfg_.max_num_batched_tokens;

    // -----------------------------------------------------------------------
    // Phase 1: Account for RUNNING requests (already have engine slots).
    //
    // Decode requests consume 1 token.
    // In-progress prefill requests consume up to long_prefill_threshold tokens.
    // -----------------------------------------------------------------------
    for (int i = 0; i < num_slots; ++i) {
        const auto& sv = slots[i];
        if (sv.finished) continue;

        if (!sv.is_prefill_chunk) {
            // Decode: 1 token
            token_budget -= 1;
        } else {
            // In-progress prefill: chunk up to threshold
            const int remaining = sv.num_prompt_tokens - sv.num_computed_tokens;
            const int chunk = std::min(remaining, resolved_threshold_);
            token_budget -= std::max(1, chunk);
        }
    }

    // -----------------------------------------------------------------------
    // Phase 2: Admit WAITING requests with remaining budget.
    //
    // Only admit up to max_num_partial_prefills new requests per step.
    // Each new request gets its first prefill chunk (up to threshold tokens).
    // -----------------------------------------------------------------------
    int num_new = 0;

    for (int i = 0; i < num_pending && token_budget > 0; ++i) {
        if (num_new >= cfg_.max_num_partial_prefills) break;
        if (num_new >= free_slots) break;

        const int prompt_len = pending_prompt_lens[i];
        const int first_chunk = std::min(prompt_len, resolved_threshold_);

        // If chunked prefill is disabled, the entire prompt must fit.
        if (!cfg_.enable_chunked_prefill && prompt_len > token_budget) {
            break;
        }

        // If we already admitted requests and this one won't fit, stop.
        if (num_new > 0 && first_chunk > token_budget) {
            break;
        }

        token_budget -= std::min(first_chunk, token_budget);
        ++num_new;
    }

    output.num_new_admitted = num_new;
    output.total_scheduled_tokens = cfg_.max_num_batched_tokens - token_budget;
    output.has_work = (num_slots > 0 && !std::all_of(slots, slots + num_slots,
        [](const SlotView& sv) { return sv.finished; })) || num_new > 0;
    return output;
}

}  // namespace serve
