// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Expert Parallelism strategy interface.
//
// Decouples EP token dispatch/combine semantics from the graph executor so
// multiple expert-parallel schemes (static, LLEP, DeepEP, capacity-capped,
// etc.) can be swapped without touching the executor or per-op kernels.
//
// The concrete strategies in this file:
//   - `StaticEP`:   classic 1:1 expert→GPU mapping (no rebalancing).
//   - `LLEP`:       Least-Loaded EP with LPT-based expert spilling and
//                   NCCL P2P weight transfers between EP peers.
//
// Both strategies share a common implementation for token A2A plumbing,
// buffer pooling, and persistent per-layer state. Subclass hooks control
// when LLEP re-routing engages.

#ifndef SUROGATE_SRC_RUNTIME_EP_EP_STRATEGY_H
#define SUROGATE_SRC_RUNTIME_EP_EP_STRATEGY_H

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "runtime/ep/ep_buffer_pool.h"
#include "runtime/ep/ep_state.h"
#include "runtime/ep/ring_arena.h"
#include "runtime/ep/weight_transfer.h"

struct RuntimeOptions;

namespace dsl {
class CompiledExecutor;
struct CompiledOp;
}  // namespace dsl

namespace ep {

/// Abstract EP strategy. Owns all per-layer EP/LLEP state, the shared GPU
/// buffer pool, the (lazy) weight-transfer stream, and the cross-layer
/// shared buffers. Strategies are constructed once per CompiledExecutor
/// and live for the entire executor lifetime.
class EPStrategy {
public:
    explicit EPStrategy(const RuntimeOptions& options);
    virtual ~EPStrategy();

    EPStrategy(const EPStrategy&) = delete;
    EPStrategy& operator=(const EPStrategy&) = delete;

    /// Human-readable strategy name (for diagnostics/logs).
    virtual const char* name() const = 0;

    /// True if the strategy can engage LLEP-style rebalancing for a layer.
    /// Static strategies return false; LLEP returns true subject to env flags.
    virtual bool supports_llep() const {
        return false;
    }

    // -----------------------------------------------------------------------
    // Op dispatch — invoked by the executor's ep_dispatch/ep_combine thunks.
    // -----------------------------------------------------------------------
    void dispatch_forward(dsl::CompiledExecutor& exec, const dsl::CompiledOp& op);
    void dispatch_backward(dsl::CompiledExecutor& exec, const dsl::CompiledOp& op);
    void combine_forward(dsl::CompiledExecutor& exec, const dsl::CompiledOp& op);
    void combine_backward(dsl::CompiledExecutor& exec, const dsl::CompiledOp& op);

    // -----------------------------------------------------------------------
    // State accessors
    // -----------------------------------------------------------------------
    std::unordered_map<int, EpLayerState>& ep_states() {
        return mEpStates;
    }
    const std::unordered_map<int, EpLayerState>& ep_states() const {
        return mEpStates;
    }
    std::unordered_map<int, LLEPLayerState>& llep_states() {
        return mLLEPStates;
    }
    const std::unordered_map<int, LLEPLayerState>& llep_states() const {
        return mLLEPStates;
    }
    std::unordered_map<int, EPLayerMeta>& layer_meta() {
        return mEPLayerMeta;
    }
    const std::unordered_map<int, EPLayerMeta>& layer_meta() const {
        return mEPLayerMeta;
    }

    EPBufferPool& buffer_pool() {
        return mBufferPool;
    }

    /// Lazily-created CUDA stream used for P2P expert weight transfer (LLEP).
    /// Returns nullptr for strategies that never transfer weights.
    cudaStream_t weight_transfer_stream();

    /// Lazily-created CUDA stream for the tiny EP control exchanges (imbalance
    /// all-reduce + token-split A2As). These depend only on host-side routing
    /// counts, so running them off MainStream keeps their completion waits from
    /// covering the whole pipeline backlog.
    cudaStream_t control_stream();

    /// Returns the replay-aware cache key used to distinguish forward vs.
    /// replay-forward EP state. Mirrors the historical CompiledExecutor
    /// helper so existing callers stay source-compatible.
    int ep_state_key(int layer_idx, bool in_replay) const;

    /// Release LLEP per-layer GPU allocations (LoRA + foreign weights) for
    /// all layers and drop the LLEP state maps. Called by LLEP at the start
    /// of every ep_dispatch to bound foreign-weight lifetime to a single
    /// active layer.
    void free_all_llep_layers();

    /// Destructor-time cleanup. Frees per-layer EP state buffers, shared
    /// cross-layer buffers, the LLEP state map, the buffer pool, and the
    /// weight-transfer stream.
    void cleanup_all();

protected:
    /// Subclass hook: given layer context, decide whether LLEP rebalancing
    /// should be enabled for this layer. Strategies that never use LLEP
    /// return false; LLEP honours env flags and architecture compatibility.
    virtual bool enable_llep_for_layer(bool separate_up_projection) const = 0;

    const RuntimeOptions& mOptions;

private:
    // ------------------------------------------------------------------
    // Forward-dispatch phase split. All definitions live in
    // ep_strategy.cpp; DispatchForwardCtx is an opaque bag of locals
    // passed between phases so each helper can stay single-purpose.
    // ------------------------------------------------------------------
    struct DispatchForwardCtx;

    void parse_forward_layout(dsl::CompiledExecutor& exec,
                              const dsl::CompiledOp& op,
                              const Tensor& permuted_input,
                              DispatchForwardCtx& ctx);
    void detect_llep_imbalance(dsl::CompiledExecutor& exec, DispatchForwardCtx& ctx);
    void plan_expert_mapping(DispatchForwardCtx& ctx, bool sticky);
    Tensor apply_llep_send_reorder(dsl::CompiledExecutor& exec,
                                   DispatchForwardCtx& ctx,
                                   const Tensor& permuted_input,
                                   EpLayerState& ep_state);
    void exchange_token_splits(dsl::CompiledExecutor& exec, DispatchForwardCtx& ctx);
    bool launch_llep_transfers(dsl::CompiledExecutor& exec,
                               DispatchForwardCtx& ctx,
                               Tensor*& native_gate_up_out,
                               Tensor*& native_down_out);
    Tensor run_token_a2a(dsl::CompiledExecutor& exec,
                         const DispatchForwardCtx& ctx,
                         const Tensor& send_src,
                         void*& recv_buf_out,
                         std::size_t& recv_bytes_out);
    void permute_recv_tokens(dsl::CompiledExecutor& exec,
                             DispatchForwardCtx& ctx,
                             const Tensor& recv_hidden,
                             Tensor& sorted_recv_out,
                             Tensor& local_scatter_out,
                             Tensor& merged_offsets_out);
    void persist_dispatch_state(dsl::CompiledExecutor& exec,
                                DispatchForwardCtx& ctx,
                                const Tensor& merged_offsets_t,
                                Tensor& merged_offsets_persisted_out);
    void finalize_llep_state(dsl::CompiledExecutor& exec,
                             DispatchForwardCtx& ctx,
                             const Tensor& native_gate_up,
                             const Tensor& native_down);
    void finalize_native_only_state(dsl::CompiledExecutor& exec, DispatchForwardCtx& ctx);

    /// Prefetch the next dispatch layer's foreign expert weights on the
    /// weight-transfer stream so the H2D overlaps this layer's compute.
    /// Replay: the next-lower layer, from the step's cached forward plan.
    /// Forward: the next-higher layer, from its sticky plan (skipped when
    /// that layer's next dispatch will recompute the plan).
    void maybe_prefetch_next_layer(dsl::CompiledExecutor& exec, const DispatchForwardCtx& ctx);

    /// Invert `ep_state.llep_send_reorder_gpu` and gather `input` rows
    /// through it into a persistent [total_send, hidden] buffer. Caller
    /// supplies the persistent buffer (field on `EpLayerState`) so the
    /// returned Tensor has a stable pointer across the backward pass.
    Tensor apply_llep_inverse_reorder(dsl::CompiledExecutor& exec,
                                      const EpLayerState& ep_state,
                                      const Tensor& input,
                                      int hidden_size,
                                      int elem_sz,
                                      void*& out_buf,
                                      std::size_t& out_bytes);

    /// Populate `llep.{gate_up,down}_weight_ptrs` + identity fields from the
    /// merged expert list. When `foreign_weights` is null, every merged
    /// expert must be native (fallback path). Otherwise non-native experts
    /// resolve through `foreign_weights` (LLEP path).
    void populate_llep_weight_pointers(LLEPLayerState& llep,
                                       const DispatchForwardCtx& ctx,
                                       const Tensor& native_gate_up,
                                       const Tensor& native_down,
                                       const ep::ForeignExpertWeights* foreign_weights);

    std::unordered_map<int, EpLayerState> mEpStates;
    std::unordered_map<int, LLEPLayerState> mLLEPStates;
    std::unordered_map<int, EPLayerMeta> mEPLayerMeta;
    EPBufferPool mBufferPool;

    /// Reusable PINNED host staging for the small count/split copies in the
    /// dispatch path. Async copies to/from PAGEABLE memory can block inside the
    /// driver's staging pool when all worker threads copy at once — with NCCL
    /// kernels at the stream heads this convoys into a cross-rank deadlock.
    /// Pinned staging never blocks the enqueue. Grow-only, keyed by call site.
    void* pinned_scratch(int slot, std::size_t bytes);
    std::unordered_map<int, std::pair<void*, std::size_t>> mPinnedScratch;

    cudaStream_t mWeightTransferStream = nullptr;
    cudaStream_t mControlStream = nullptr;

    /// Ring-slab arena backing foreign-expert weight staging (base weights +
    /// received LoRA slices). Replaces the per-expert cudaMalloc/cudaFree storm
    /// on the LLEP dispatch path.
    RingSlabArena mForeignArena;

    /// Chunked-sequence: per-(layer, chunk) snapshot of the plan and the
    /// exchanged splits from the chunk's phase-A forward. Phase-B re-forwards
    /// restore from here (routing is deterministic), skipping the imbalance
    /// all-reduce, LPT and the two count A2As per layer.
    struct ChunkPlanSnapshot {
        bool use_llep = false;
        LPTPlan plan;
        std::vector<int> expert_to_gpu;
        std::vector<int> merged_experts;
        std::vector<int> global_to_merged;
        std::vector<int> send_splits;
        std::vector<int> recv_splits;
        std::vector<int> recv_all_counts;
        int total_send = 0;
        int total_recv = 0;
    };
    std::unordered_map<long, ChunkPlanSnapshot> mChunkPlanCache;  ///< key: layer*4096 + chunk

    /// One in-flight backward prefetch (see maybe_prefetch_next_replay_layer).
    struct PrefetchedForeign {
        int layer_idx = -1;
        int arena_slot = -1;
        ForeignExpertWeights weights;
        std::vector<std::pair<int, int>> receive_set;
    };
    std::optional<PrefetchedForeign> mPrefetched;
};

/// Classic 1:1 expert→GPU mapping. No rebalancing, no P2P weight transfer.
/// Kept as a standalone strategy for tests and as an explicit opt-out for
/// future callers that want to disable LLEP entirely at construction time.
class StaticEP final : public EPStrategy {
public:
    using EPStrategy::EPStrategy;

    const char* name() const override {
        return "StaticEP";
    }
    bool supports_llep() const override {
        return false;
    }

protected:
    bool enable_llep_for_layer(bool /*separate_up_projection*/) const override {
        return false;
    }
};

/// Least-Loaded EP: detects routing imbalance at dispatch time and spills
/// tokens + weights to underutilised peers via NCCL P2P.
///
/// LLEP engages automatically per-layer when `ep_load_balance_threshold`
/// is reached; below the threshold it falls back to static expert→GPU
/// mapping transparently. Nemotron-style MoE layers that keep a separate
/// `experts_up` projection are not currently LLEP-compatible — those
/// layers stay on the static path.
class LLEP final : public EPStrategy {
public:
    using EPStrategy::EPStrategy;

    const char* name() const override {
        return "LLEP";
    }
    bool supports_llep() const override {
        return true;
    }

protected:
    bool enable_llep_for_layer(bool separate_up_projection) const override;
};

/// Factory: build the default EP strategy.
///
/// Always returns `LLEP`. It's the more capable strategy and degrades
/// gracefully into static mapping when rebalancing isn't warranted
/// (per-layer imbalance below `ep_load_balance_threshold`) or not
/// supported (Nemotron-style `experts_up` layers). Callers that want
/// pure static mapping can construct `StaticEP` directly.
std::unique_ptr<EPStrategy> create_strategy(const RuntimeOptions& options);

}  // namespace ep

#endif  // SUROGATE_SRC_RUNTIME_EP_EP_STRATEGY_H
