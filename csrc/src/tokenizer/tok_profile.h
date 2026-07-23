// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Env-gated phase profiler for the training encode path. Splits the cost of
// encode_for_training() into chat-template rendering (minja/Jinja),
// pretokenization (regex split) and BPE merge, so we can see which one
// actually dominates SFT data preparation. Enable with SUROGATE_TOK_PROFILE=1.
//
// Thread-safe: encode_for_training_batch() runs many encode_for_training()
// calls concurrently across std::thread workers, so every counter is atomic.
// When disabled, constructing a TokTimer is a single cached-bool check and
// costs nothing measurable — the timers are placed at per-segment granularity,
// never per-token.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>

namespace tokenizer {

// Aggregate nanosecond counters. render + pretok + bpe are subsets of total;
// (total - render - pretok - bpe) is "other" (json build, added-token scan,
// substring/vector churn, template ChatMessage->json conversion).
struct TokProfile {
    std::atomic<uint64_t> render_ns{0};       // apply_chat_template (minja render + ctx build + json)
    std::atomic<uint64_t> render_walk_ns{0};  // subset of render_ns: just minja TemplateNode::render()
    std::atomic<uint64_t> pretok_ns{0};       // unicode_regex_split
    std::atomic<uint64_t> bpe_ns{0};          // encode_piece loop (BPE merge)
    std::atomic<uint64_t> total_ns{0};        // full encode_for_training call
    std::atomic<uint64_t> calls{0};           // encode_for_training invocations
    std::atomic<uint64_t> renders{0};         // apply_chat_template invocations
    std::atomic<uint64_t> tokens{0};          // output input_ids produced
};

// True iff SUROGATE_TOK_PROFILE=1 (evaluated once, then cached).
bool tok_profile_enabled();

// The process-global accumulator.
TokProfile& tok_profile();

// Human-readable summary (per-phase ns, %, tokens/s). Returns "" when no
// samples were collected. If reset is true, zeroes the counters afterward.
std::string tok_profile_report(bool reset = false);

// Scoped timer: adds its lifetime to `sink` on destruction. No-op (no clock
// read) when profiling is disabled.
struct TokTimer {
    std::atomic<uint64_t>* sink;
    std::chrono::steady_clock::time_point t0;

    explicit TokTimer(std::atomic<uint64_t>& s) : sink(tok_profile_enabled() ? &s : nullptr) {
        if (sink) t0 = std::chrono::steady_clock::now();
    }
    ~TokTimer() {
        if (sink) {
            const auto dt = std::chrono::steady_clock::now() - t0;
            sink->fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count()),
                            std::memory_order_relaxed);
        }
    }

    TokTimer(const TokTimer&) = delete;
    TokTimer& operator=(const TokTimer&) = delete;
};

}  // namespace tokenizer
