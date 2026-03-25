// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// ContinuousGenerationEngine: iteration-level continuous batching for serving.
//
// This engine maintains a pool of sequence slots.  New sequences can join at
// any decode step and finished sequences release their KV-cache pages
// immediately.
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
    int seq_len = 0;              // current KV-cache length (tokens in KV cache)
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

    // Chunked prefill state
    std::vector<int32_t> prompt;  // full prompt token IDs (kept until prefill done)
    int prefill_progress = 0;     // tokens prefilled so far (0..prompt.size())
    bool prefilling() const { return prefill_progress < static_cast<int>(prompt.size()); }
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
              DeviceMemoryStack& arena,
              int max_num_batched_tokens = 0);

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

    /// Flat-token step: prefill new prompts + decode active sequences in ONE
    /// forward pass.  Uses execute_flat_tokens() with FlashInfer's
    /// BatchPrefillWithPagedKVCache for unified attention.
    ///
    /// All tokens (prefill + decode) are concatenated into a flat array.
    /// Each request has variable Q length (prompt_len for prefill, 1 for decode).
    /// Returns one sampled token per request (from the last Q token's logits).
    struct FlatStepConfig {
        int max_gen_len = 512;
        float temperature = 1.0f;
        int32_t eos_token_id = 2;
        int top_k = 0;
        float top_p = 1.0f;
        float min_p = 0.0f;
        int prefill_chunk_size = 256;  // max prefill tokens per request per step
        int multi_decode_steps = 1;    // N graph replays per pure-decode step (1 = single-step)
    };
    struct FlatStepResult {
        std::vector<int> new_slot_ids;           // slot IDs for new prompts (-1 if failed)
        std::vector<int> active_slot_ids;        // all active slots (including new)
        std::vector<int32_t> sampled_tokens;     // [decode_steps * batch_size] flat, step-major
        std::vector<int> finished;               // 1 if finished (after all steps)
        std::vector<int> completion_lens;        // total generated per slot (after all steps)
        int decode_steps = 1;                    // number of tokens per slot in sampled_tokens
    };
    FlatStepResult flat_step(
        const std::vector<std::vector<int32_t>>& new_prompts,
        const FlatStepConfig& config,
        dsl::GraphExecutor& graph_executor,
        NCCLCommunicator& comm,
        const modules::ForwardHook* hook = nullptr);

    /// Non-blocking flat_step: launches GPU work and returns immediately.
    /// Returns new_slot_ids for the new prompts (available immediately).
    /// Call flat_step_collect() to sync and get sampled tokens + slot updates.
    std::vector<int> flat_step_launch(
        const std::vector<std::vector<int32_t>>& new_prompts,
        const FlatStepConfig& config,
        dsl::GraphExecutor& graph_executor,
        NCCLCommunicator& comm,
        const modules::ForwardHook* hook = nullptr);

    /// Collect results from a previous flat_step_launch.
    /// Blocks on cudaEventSynchronize and updates slot state.
    FlatStepResult flat_step_collect();

    /// True if a flat_step_launch is pending collection.
    bool has_pending_step() const { return has_pending_sampled_; }

    void release_slot(int slot_id);
    int num_active() const;
    int num_free_slots() const;
    int num_free_pages() const;
    void destroy();
    bool initialized() const { return initialized_; }

private:
    void run_sampling_dispatch(int batch_size, int V, float temperature,
                               int top_k, float top_p, float min_p,
                               cudaStream_t stream);
    int next_bucket(int B) const;
    int next_prefill_token_bucket(int total_tokens) const;

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

    // Async D2H readback for flat_step (eliminates cudaStreamSynchronize).
    // Pinned host buffer + event allow the CPU to prepare the next batch
    // while the GPU finishes sampling + D2H copy.
    int32_t* pinned_sampled_ = nullptr;      // cudaMallocHost'd pinned buffer [max_slots_]
    int32_t* pinned_multi_sampled_ = nullptr; // [kMaxMultiSteps * max_slots_]
    static constexpr int kMaxMultiSteps = 16;
    cudaEvent_t sampled_ready_event_ = nullptr;  // recorded after D2H copy
    bool has_pending_sampled_ = false;        // true if event needs sync
    int pending_decode_steps_ = 1;            // multi-step count for pending collect

    // ========================================================================
    // Pinned host staging buffers for H2D copies in flat_step().
    // With pageable memory, cudaMemcpyAsync secretly synchronizes (~631μs/call).
    // With pinned memory, it returns in ~5μs — eliminating ~6s of host overhead.
    // Allocated once at init, reused every step.
    // ========================================================================
    struct PinnedStaging {
        int32_t* token_ids = nullptr;        // [max_batched_tokens]
        int32_t* position_ids = nullptr;     // [max_batched_tokens]
        int32_t* token_to_req = nullptr;     // [max_batched_tokens]
        int32_t* kv_write_pos = nullptr;     // [max_batched_tokens]
        int32_t* q_indptr = nullptr;         // [max_slots + 1]
        int*     seq_lens_current = nullptr; // [max_slots]
        int32_t* seq_lens_k = nullptr;       // [max_slots]
        int*     finished_flags = nullptr;   // [max_slots]
        int*     block_table = nullptr;      // [max_slots * max_pages_per_seq]
        int32_t* last_token_indices = nullptr; // [max_slots]
        // rebuild_compact_batch buffers
        int32_t* compact_last_tokens = nullptr; // [max_slots]
        int*     compact_seq_lens = nullptr;    // [max_slots]
        int*     compact_finished = nullptr;    // [max_slots]
        float*   compact_temp = nullptr;        // [max_slots]
        int*     compact_block_table = nullptr; // [max_slots * max_pages_per_seq]

        bool allocated = false;

        void free_all() {
            if (!allocated) return;
            cudaFreeHost(token_ids);
            cudaFreeHost(position_ids);
            cudaFreeHost(token_to_req);
            cudaFreeHost(kv_write_pos);
            cudaFreeHost(q_indptr);
            cudaFreeHost(seq_lens_current);
            cudaFreeHost(seq_lens_k);
            cudaFreeHost(finished_flags);
            cudaFreeHost(block_table);
            cudaFreeHost(last_token_indices);
            cudaFreeHost(compact_last_tokens);
            cudaFreeHost(compact_seq_lens);
            cudaFreeHost(compact_finished);
            cudaFreeHost(compact_temp);
            cudaFreeHost(compact_block_table);
            allocated = false;
        }
    };
    PinnedStaging pinned_;
    int pending_batch_size_ = 0;
    std::vector<int> pending_active_sids_;
    std::vector<int> pending_q_lens_;         // per-slot q_lens from previous step
    std::vector<int> pending_new_slot_ids_;   // new_slot_ids for deferred flat_step_collect
    bool pending_compact_batch_needs_rebuild_ = false;
    FlatStepResult pending_result_;           // saved result from flat_step_launch
    bool deferred_sync_ = false;              // when true, flat_step skips sync + Phase 8

    // CUDA graph bucket sizes
    std::vector<int> graph_buckets_;
    std::vector<int> prefill_token_buckets_;

    // Persistent DecodeState (rebuilt only when batch changes)
    DecodeState decode_state_{};
    int decode_state_max_seqlen_k_ = 0;

    // Full-step CUDA graph (captures forward + sampling in one graph)
    cudaGraphExec_t full_step_graph_exec_ = nullptr;
    DeviceMemoryStack::Checkpoint full_step_graph_checkpoint_{};
    bool full_step_primed_ = false;
    int prev_compact_B_padded_ = 0;  // track padded batch size for graph invalidation

    // Pre-allocated flat-step GPU buffers (avoid per-step cudaMallocAsync)
    int max_batched_tokens_ = 0;           // token budget for flat_step
    int prefill_rr_cursor_ = 0;            // fairness cursor for budgeted prefill admission

    // Packed flat-step metadata: 5 arrays in one contiguous GPU + pinned buffer.
    // ONE H2D copy per step replaces 5 separate cudaMemcpyAsync calls.
    // Layout: [token_to_req:MBT] [kv_write_pos:MBT] [q_indptr:S+1] [seq_lens_k:S] [last_tok_idx:S]
    void* flat_packed_gpu_ = nullptr;          // contiguous GPU buffer
    void* flat_packed_pinned_ = nullptr;       // contiguous pinned host buffer
    std::size_t flat_packed_bytes_ = 0;        // total size of packed buffer
    // Offsets into packed buffer (in bytes)
    std::size_t flat_off_token_to_req_ = 0;
    std::size_t flat_off_kv_write_pos_ = 0;
    std::size_t flat_off_q_indptr_ = 0;
    std::size_t flat_off_seq_lens_k_ = 0;
    std::size_t flat_off_last_token_idx_ = 0;
    // Typed pointers into GPU packed buffer (set once at init)
    int32_t* flat_token_to_req_gpu_ = nullptr;
    int32_t* flat_kv_write_pos_gpu_ = nullptr;
    int32_t* flat_q_indptr_gpu_ = nullptr;
    int32_t* flat_last_token_indices_gpu_ = nullptr;
    int32_t* flat_seq_lens_k_gpu_ = nullptr;
    int32_t* flat_page_indptr_gpu_ = nullptr;   // [max_slots_ + 1]
    int32_t* flat_page_indices_gpu_ = nullptr;  // [max_slots_ * max_pages_per_seq_]
    int32_t* flat_last_page_len_gpu_ = nullptr; // [max_slots_]
    void* flat_plan_int_ws_gpu_ = nullptr;      // 16MB
    void* flat_plan_float_ws_gpu_ = nullptr;    // 64MB
    static constexpr std::size_t kPlanIntWsSize = 16 * 1024 * 1024;
    static constexpr std::size_t kPlanFloatWsSize = 64 * 1024 * 1024;

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
