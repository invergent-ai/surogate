// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// ContinuousGenerationEngine: iteration-level continuous batching for serving.
//
// Unlike GenerationSession (fixed batch, run-to-completion), this engine
// maintains a pool of sequence slots.  New sequences can join at any decode
// step and finished sequences release their KV-cache pages immediately.
//
// Decode state (seq_lens, last_tokens, finished, block_table) lives
// persistently on the GPU.  The compact batch is only rebuilt when the
// active set changes (add/release), not every token.  step(max_tokens)
// runs a multi-token inner loop in C++ to minimize Python round-trips.

#ifndef SUROGATE_SRC_RUNTIME_INFER_CONTINUOUS_ENGINE_H
#define SUROGATE_SRC_RUNTIME_INFER_CONTINUOUS_ENGINE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "runtime/infer/decode_state.h"
#include "runtime/infer/page_pool.h"
#include "runtime/core/forward_hooks.h"

// Forward declarations — full definitions needed only in .cpp.
namespace modules { struct ModelConfig; }
namespace dsl {
class DslRunState;
class DslParamStore;
class GraphExecutor;
}
class NCCLCommunicator;
class DeviceMemoryStack;
struct RuntimeOptions;

namespace infer {

/// Per-sequence state within the continuous engine.
struct SequenceSlot {
    bool active = false;
    int seq_len = 0;              // current KV-cache length (after prefill)
    int generated_count = 0;      // number of output tokens generated
    int max_gen_len = 0;          // generation budget
    int32_t last_token = 0;       // most recent token (feed to next decode step)
    bool finished = false;        // EOS or max_len reached
    int32_t eos_token_id = 0;
    float temperature = 0.0f;
    int top_k = 0;
    float top_p = 1.0f;
    float min_p = 0.0f;
    uint64_t rng_offset = 0;
    std::vector<int> page_ids;    // allocated pages from the pool
    int compact_idx = -1;         // position in the compact batch (-1 = not in batch)
};

/// Result from step(max_tokens).
struct ContinuousStepResult {
    std::vector<int> slot_ids;              // active slot IDs
    std::vector<std::vector<int32_t>> tokens;  // per-slot tokens generated this call
    std::vector<int> finished;              // 1 if slot is finished
    std::vector<int> completion_lens;       // total generated count per slot
};

class ContinuousGenerationEngine {
public:
    ContinuousGenerationEngine() = default;
    ~ContinuousGenerationEngine();

    void init(int max_slots, int max_seq_len, int total_pages,
              bool use_cuda_graphs,
              dsl::DslRunState& run_state,
              dsl::DslParamStore& weights,
              const modules::ModelConfig& config,
              const RuntimeOptions& options,
              DeviceMemoryStack& arena);

    /// Pre-warm prefill graph cache for common prompt lengths.
    /// Eliminates compilation latency on the first real request.
    void warm_prefill_graphs(dsl::GraphExecutor& graph_executor,
                             NCCLCommunicator& comm);

    /// Add a new sequence: prefill the prompt and assign a slot.
    /// @return Slot ID (>= 0), or -1 if no free slots or pages.
    int add_sequence(const std::vector<int32_t>& prompt_ids,
                     int max_gen_len, float temperature, int32_t eos_token_id,
                     int top_k, float top_p, float min_p,
                     int prefill_chunk_size,
                     dsl::GraphExecutor& graph_executor,
                     NCCLCommunicator& comm,
                     const modules::ForwardHook* hook = nullptr);

    /// Batch-add multiple sequences: prefills them together (B>1) for speed.
    /// @return Slot IDs per prompt (-1 for any that couldn't be added).
    struct BatchAddConfig {
        int max_gen_len = 512;
        float temperature = 1.0f;
        int32_t eos_token_id = 2;
        int top_k = 0;
        float top_p = 1.0f;
        float min_p = 0.0f;
        int prefill_chunk_size = 256;
    };
    std::vector<int> add_sequences_batch(
        const std::vector<std::vector<int32_t>>& prompts,
        const BatchAddConfig& config,
        dsl::GraphExecutor& graph_executor,
        NCCLCommunicator& comm,
        const modules::ForwardHook* hook = nullptr);

    /// Decode up to max_tokens for ALL active sequences.
    /// Runs a tight C++ loop — no Python round-trips until return.
    ContinuousStepResult step(int max_tokens,
                              dsl::GraphExecutor& graph_executor,
                              NCCLCommunicator& comm,
                              const modules::ForwardHook* hook = nullptr);

    void release_slot(int slot_id);
    int num_active() const;
    int num_free_slots() const;
    int num_free_pages() const;
    void destroy();
    bool initialized() const { return initialized_; }

private:
    int next_bucket(int B) const;

    /// Rebuild compact batch arrays from active slots.
    /// Only called when batch_dirty_ is true.
    void rebuild_compact_batch();
    /// Run one decode token step on the current compact batch (GPU-only, no sync).
    void run_one_decode_step(dsl::GraphExecutor& graph_executor,
                             NCCLCommunicator& comm,
                             const modules::ForwardHook* hook);

    bool initialized_ = false;
    int max_slots_ = 0;
    int max_seq_len_ = 0;
    int max_pages_per_seq_ = 0;
    int vocab_size_ = 0;
    int32_t fallback_token_id_ = 0;
    bool use_cuda_graphs_ = false;
    bool cuda_graph_primed_ = false;

    // Slot pool
    std::vector<SequenceSlot> slots_;
    std::vector<int> free_slot_stack_;

    // Page pool
    PagePool page_pool_;
    static constexpr int kPageBlockSize = 256;

    // Compact batch tracking
    bool batch_dirty_ = true;           // need to rebuild compact batch
    int compact_B_ = 0;                 // number of active sequences in compact batch
    int compact_B_padded_ = 0;          // padded to CUDA graph bucket
    std::vector<int> active_slot_ids_;  // [compact_B_] slot IDs in compact order

    // Pre-allocated GPU buffers (sized for max_slots_)
    int32_t* last_tokens_gpu_ = nullptr;
    int* seq_lens_gpu_ = nullptr;
    int* finished_gpu_ = nullptr;
    int32_t* position_ids_gpu_ = nullptr;
    int32_t* cu_seqlens_q_gpu_ = nullptr;
    int32_t* seqused_k_gpu_ = nullptr;
    int* block_table_gpu_ = nullptr;      // [max_slots_, max_pages_per_seq_]
    float* logits_gpu_ = nullptr;         // [max_slots_, vocab_size_]
    float* probs_gpu_ = nullptr;
    int32_t* sampled_tokens_gpu_ = nullptr;
    float* temperature_gpu_ = nullptr;
    int32_t* completion_lens_gpu_ = nullptr;
    void* softmax_ws_ = nullptr;
    std::size_t softmax_ws_bytes_ = 0;

    // Prefill scratch: block table for B=1 prefill
    int* prefill_block_table_gpu_ = nullptr;

    // Host-side buffers for D2H readback
    std::vector<int32_t> sampled_tokens_host_;
    std::vector<int> finished_host_;

    // CUDA graph bucket sizes
    std::vector<int> graph_buckets_;

    // Persistent DecodeState (rebuilt only when batch changes)
    DecodeState decode_state_{};
    int decode_state_max_seqlen_k_ = 0;

    // Full-step CUDA graph (captures forward + sampling in one graph)
    cudaGraphExec_t full_step_graph_exec_ = nullptr;
    DeviceMemoryStack::Checkpoint full_step_graph_checkpoint_{};
    bool full_step_primed_ = false;

    // Sampling config (from first active slot — uniform params)
    float batch_temperature_ = 1.0f;
    int batch_top_k_ = 0;
    float batch_top_p_ = 1.0f;
    float batch_min_p_ = 0.0f;
    bool greedy_ = false;

    // References (not owned)
    dsl::DslRunState* run_state_ = nullptr;
    dsl::DslParamStore* weights_ = nullptr;
    const modules::ModelConfig* config_ = nullptr;
    const RuntimeOptions* options_ = nullptr;
    DeviceMemoryStack* arena_ = nullptr;
};

}  // namespace infer

#endif  // SUROGATE_SRC_RUNTIME_INFER_CONTINUOUS_ENGINE_H
