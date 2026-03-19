// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DecodeContext: manages per-step state for autoregressive generation.
//
// Holds the KV-cache, per-sequence tracking (positions, EOS),
// and GPU buffers needed for decode (cu_seqlens, seq_lens, etc.).

#ifndef SUROGATE_SRC_RUNTIME_INFER_DECODE_CONTEXT_H
#define SUROGATE_SRC_RUNTIME_INFER_DECODE_CONTEXT_H

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "runtime/infer/kv_cache.h"
#include "utilities/stack.h"
#include "utilities/tensor.h"

namespace infer {

/// Configuration for a generation batch.
struct GenerationConfig {
    int batch_size = 0;         // Total sequences (M prompts * N completions)
    int max_gen_len = 0;        // Maximum generation length (decode steps)
    int max_prompt_len = 0;     // Maximum prompt length (for total seq_len budget)
    float temperature = 1.0f;   // Sampling temperature
    int eos_token_id = 2;       // End-of-sequence token ID
};

/// Per-batch generation state tracking.
struct BatchGenerationState {
    int batch_size = 0;

    std::vector<int> seq_lens;          // Current total length per sequence (prompt + generated)
    std::vector<bool> finished;         // EOS hit or max_len reached
    std::vector<int> finish_positions;  // Where each sequence stopped generating
    int num_active = 0;                 // Count of unfinished sequences

    void init(int batch_size_, int initial_seq_len) {
        batch_size = batch_size_;
        seq_lens.assign(static_cast<std::size_t>(batch_size_), initial_seq_len);
        finished.assign(static_cast<std::size_t>(batch_size_), false);
        finish_positions.assign(static_cast<std::size_t>(batch_size_), 0);
        num_active = batch_size_;
    }

    bool all_finished() const { return num_active == 0; }

    void mark_finished(int idx, int pos) {
        if (!finished[static_cast<std::size_t>(idx)]) {
            finished[static_cast<std::size_t>(idx)] = true;
            finish_positions[static_cast<std::size_t>(idx)] = pos;
            num_active--;
        }
    }
};

/// Trajectory: stores generated tokens and logprobs on the host.
/// Populated incrementally during generation via streaming D2H.
struct Trajectory {
    std::vector<int32_t> tokens;        // prompt + completion token IDs
    std::vector<float> logprobs;        // per-token log-probabilities (generation only)
    int prompt_len = 0;                 // boundary between prompt and completion
    int completion_len = 0;             // actual generated length (before padding)
    float reward = 0.0f;               // set during scoring phase
};

/// DecodeContext: manages all GPU state for a generation batch.
///
/// Lifecycle:
///   1. Construct with config + arena
///   2. Call prefill() to process prompts and populate KV-cache
///   3. Call decode_step() repeatedly until all_finished()
///   4. Read trajectories
///   5. Destruct (arena memory freed via stack restore)
struct DecodeContext {
    // Configuration
    GenerationConfig config;
    int num_layers = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int max_total_len = 0;  // max_prompt_len + max_gen_len

    // KV-cache (arena-backed)
    KVCache kv_cache;

    // Batch state
    BatchGenerationState state;

    // GPU buffers (arena-backed)
    int* seq_lens_gpu = nullptr;        // [batch_size] current seq lengths on GPU
    int32_t* cu_seqlens_q_gpu = nullptr; // [batch_size + 1] cumulative Q lengths
    int32_t* cu_seqlens_k_gpu = nullptr; // [batch_size + 1] cumulative K lengths

    // Host trajectories (populated during generation)
    std::vector<Trajectory> trajectories;

    // Arena checkpoint for cleanup
    DeviceMemoryStack::Checkpoint arena_checkpoint;

    /// Initialize and allocate from arena.
    void init(const GenerationConfig& cfg, int num_layers_, int num_kv_heads_,
              int head_dim_, DeviceMemoryStack& arena) {
        config = cfg;
        num_layers = num_layers_;
        num_kv_heads = num_kv_heads_;
        head_dim = head_dim_;
        max_total_len = cfg.max_prompt_len + cfg.max_gen_len;

        arena_checkpoint = arena.checkpoint();

        // Allocate KV-cache
        kv_cache = KVCache(num_layers_, cfg.batch_size, max_total_len,
                           num_kv_heads_, head_dim_, KVDType::BF16);
        kv_cache.allocate(arena);

        // Allocate GPU tracking buffers
        auto seq_lens_buf = arena.allocate(
            static_cast<std::size_t>(cfg.batch_size) * sizeof(int), "decode_seq_lens");
        seq_lens_gpu = reinterpret_cast<int*>(seq_lens_buf);

        auto cu_q_buf = arena.allocate(
            static_cast<std::size_t>(cfg.batch_size + 1) * sizeof(int32_t), "decode_cu_seqlens_q");
        cu_seqlens_q_gpu = reinterpret_cast<int32_t*>(cu_q_buf);

        auto cu_k_buf = arena.allocate(
            static_cast<std::size_t>(cfg.batch_size + 1) * sizeof(int32_t), "decode_cu_seqlens_k");
        cu_seqlens_k_gpu = reinterpret_cast<int32_t*>(cu_k_buf);

        // Initialize batch state
        state.init(cfg.batch_size, 0);

        // Initialize trajectories
        trajectories.resize(static_cast<std::size_t>(cfg.batch_size));
    }

    /// Release arena memory.
    void release(DeviceMemoryStack& arena) {
        arena.restore(arena_checkpoint);
    }
};

}  // namespace infer

#endif  // SUROGATE_SRC_RUNTIME_INFER_DECODE_CONTEXT_H
