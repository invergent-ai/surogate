// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL gradient store - manages parameter gradients for DSL execution.

#ifndef SUROGATE_SRC_DSL_DSL_GRAD_STORE_H
#define SUROGATE_SRC_DSL_DSL_GRAD_STORE_H

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "utilities/allocator.h"
#include "utilities/tensor.h"

namespace dsl {

class DslParamStore;
class HookRegistry;
enum class HookEventKind;
struct CompiledGraph;
struct PhaseArenas;

}  // namespace dsl

class NCCLCommunicator;

namespace dsl {

/// Configuration for gradient management (mirrors ModularGradientManager::Config)
struct DslGradStoreConfig {
    int num_shards = 1;                  ///< Number of ZeRO shards (world_size)
    int shard_idx = 0;                   ///< This rank's shard index
    bool shard_gradients = false;        ///< ZeRO-2: shard gradients across ranks
    bool use_all_to_all_reduce = false;  ///< Use all-to-all instead of reduce-scatter
    int num_layers = 0;                  ///< Number of transformer layers
};

// Stores parameter gradients for DSL execution.
class DslGradStore {
public:
    DslGradStore(const DslParamStore& params,
                 const std::shared_ptr<TensorAllocator>& allocator,
                 bool offload_grads = false,
                 EAllocationType offload_alloc = EAllocationType::PINNED,
                 int num_shards = 1,
                 bool tied_embeddings = false,
                 std::optional<ETensorDType> grad_dtype_override = std::nullopt,
                 bool cpu_training = false);

    /// Configure multi-GPU gradient reduction
    void configure(const DslGradStoreConfig& config);
    void set_schema_hook_registry(const HookRegistry* registry, std::vector<std::string> schema_ids_by_layer);

    void start_micro_step(cudaStream_t stream, int micro_step, int total_steps);
    void end_micro_step(cudaStream_t stream, NCCLCommunicator& comm);

    Tensor* get_param_grad(const std::string& name, bool& accumulate);

    void zero_all(cudaStream_t stream);
    void reduce_all(NCCLCommunicator& comm, cudaStream_t stream);

    /// Start async all-reduce on all gradients (non-blocking).
    /// Call wait_for_reduce() or synchronize on AllReduceDone event before using gradients.
    void reduce_all_async(NCCLCommunicator& comm, cudaStream_t stream, cudaEvent_t done_event);

    /// Notify that a layer's backward pass is complete.
    /// For ZeRO-1: triggers reduce-scatter on the last micro-step.
    /// For ZeRO-2: triggers reduce-scatter every micro-step with deferred accumulation.
    void notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm);

    /// Wait for a layer's reduction to complete (call before optimizer uses sharded grads)
    void wait_for_block_reduce(int layer_idx, cudaStream_t stream);

    /// Check if async reduce has been started (for avoiding redundant reduce in update)
    bool is_reduce_pending() const {
        return mReducePending;
    }
    void clear_reduce_pending() {
        mReducePending = false;
    }

    /// Check if per-layer overlapped reduction is enabled
    /// Returns true only if multi-GPU AND we have layer gradients to reduce
    bool is_overlapped_enabled() const {
        return mConfig.num_shards > 1 && mHasLayerGrads;
    }

    [[nodiscard]] bool is_first_micro_step() const {
        return mMicroStep == 0;
    }
    [[nodiscard]] bool is_last_micro_step() const {
        return mIsLastMicroStep;
    }

    const std::vector<std::string>& param_names() const {
        return mParamOrder;
    }
    const std::unordered_map<std::string, Tensor>& grads() const {
        return mGrads;
    }

    /// Get sharded gradients for optimizer (returns mShardedGrads if ZeRO-2 offload, else mGrads)
    const std::unordered_map<std::string, Tensor>& sharded_grads() const {
        return mShardedGrads.empty() ? mGrads : mShardedGrads;
    }

    /// Check if gradient offloading is active (ZeRO-2 with offload_grads=true)
    [[nodiscard]] bool is_offloading() const {
        return !mShardedGrads.empty();
    }

    /// Get gradients for a specific layer (for per-layer operations)
    std::vector<Tensor*> get_layer_grads(int layer_idx);

    /// Get sharded gradients for a specific layer (for optimizer with offloading)
    std::vector<Tensor*> get_layer_sharded_grads(int layer_idx);

    // ========================================================================
    // CPU-RAM centric gradient streaming (per-layer D2H)
    // ========================================================================

    /// Enable streaming mode: allocate rotating GPU buffers + CPU pinned storage.
    /// Call after constructor and configure(), before first backward.
    void enable_streaming(const DslParamStore& params);

    /// Bind GPU buffer for this layer's gradients (call at layer_start in backward).
    void prepare_layer_grads(int layer_idx, cudaStream_t stream);

    /// Async D2H copy layer gradients to CPU (call at layer_end in backward).
    void offload_layer_grads(int layer_idx, cudaStream_t compute_stream, cudaStream_t copy_stream);
    void accumulate_layer_to_sharded(int layer_idx, cudaStream_t stream);

    /// Per-layer NCCL all-reduce (multi-GPU). Call at layer_end BEFORE offload.
    void reduce_layer_grads(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm);

    /// Copy non-block grads (embedding/lm_head/norm) to CPU. Call once after backward.
    void offload_non_block_grads(cudaStream_t stream);

    /// Wait for all outstanding D2H copies to complete.
    void wait_all_offloads(cudaStream_t stream);

    /// Get CPU gradient for optimizer.
    const Tensor& get_cpu_grad(const std::string& name) const;

    /// Get the full CPU gradient map (for norm computation).
    const std::unordered_map<std::string, Tensor>& get_cpu_grads_map() const {
        return mCpuGrads;
    }

    /// Layer gradient names (for rebinding in compiled executor).
    const std::vector<std::string>& layer_grad_names(int layer_idx) const;

    /// Is per-layer streaming active?
    [[nodiscard]] bool is_streaming_grads() const {
        return mStreamGrads;
    }

    /// GPU-side accumulated gradient norm (sum of squares). Updated during offload.
    Tensor& layer_norm_accum() {
        return mLayerNormAccum;
    }

    /// Maximum (offset + bytes) across all locally-allocated device-resident
    /// gradients whose tid lands in `graph`'s Accumulator region. Zero when
    /// any gradient path is non-local (ZeRO-2 sharded, offloaded, streaming,
    /// or CPU-training) — those keep allocator-owned storage. Used by
    /// `GraphExecutor::compile_graphs` to clamp `mPhaseArenas.accumulator_bytes`.
    [[nodiscard]] std::size_t rebindable_accumulator_bytes(const CompiledGraph& graph) const;

    /// Route every locally-allocated device-resident gradient in `mGrads`
    /// through the Accumulator arena: `cudaMemcpyAsync` bytes into
    /// `arenas.accumulator_ptr + meta.offset`, `mAllocator->free` the
    /// original, rebind `.Data` to the arena slot. Aliased entries
    /// (tied embedding → lm_head grad) share a slot via pointer dedup.
    /// Must run after `allocate_phase_arenas`; no-op when
    /// `rebindable_accumulator_bytes()` returns 0.
    void rebind_to_accumulator_arena(const CompiledGraph& graph, const PhaseArenas& arenas, cudaStream_t stream);

private:
    void build_layer_grad_map();
    void build_zero_segments();
    void scatter_reduce_layer(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm);
    void create_layer_events(int num_layers);
    void destroy_layer_events() noexcept;
    void allocate_sharded_grads();
    void accumulate_to_sharded(int layer_idx, cudaStream_t stream);
    int dispatch_schema_layer_hooks(HookEventKind event, int layer_idx, cudaStream_t stream, void* payload = nullptr);

    std::shared_ptr<TensorAllocator> mAllocator;
    std::unordered_map<std::string, Tensor> mGrads;         ///< Full gradients (always on device for NCCL)
    std::unordered_map<std::string, Tensor> mShardedGrads;  ///< ZeRO-2: sharded gradient storage (may be on host)
    std::vector<std::string> mParamOrder;
    std::optional<ETensorDType> mGradDtypeOverride;
    bool mAccumulate = false;
    bool mReducePending = false;  ///< True if async reduce has been started
    int mMicroStep = 0;
    bool mIsLastMicroStep = false;

    // Offload configuration (stored for deferred sharded allocation in configure())
    bool mOffloadGrads = false;
    bool mCpuTraining = false;
    EAllocationType mOffloadAlloc = EAllocationType::PINNED;

    // Per-layer gradient organization (for overlapped reduction)
    DslGradStoreConfig mConfig;
    std::vector<std::vector<std::string>> mLayerGradNames;  ///< Gradient names per layer
    std::vector<cudaEvent_t> mLayerReduceEvents;            ///< One event per layer
    bool mHasLayerGrads = false;  ///< True if we have any layer gradients (false for LoRA-only)
    const HookRegistry* mSchemaHookRegistry = nullptr;
    std::vector<std::string> mHookSchemaIdsByLayer;
    bool mSchemaHookDispatchEnabled = false;

    // Bulk gradient zeroing (single kernel launch instead of per-param memsets)
    Tensor mZeroPtrs;    ///< Device array of gradient data pointers (uint64_t)
    Tensor mZeroSizes;   ///< Device array of gradient byte sizes (uint64_t)
    int mZeroCount = 0;  ///< Number of segments

    // ZeRO-2 double-buffering state (for deferred accumulation)
    struct BlockState {
        int LayerIdx = -1;
        bool NeedsAccumulation = false;
        cudaEvent_t Event = nullptr;
    };
    std::array<BlockState, 2> mBlockStates;  ///< Double-buffer for overlapped layers

    // ========================================================================
    // CPU-RAM centric gradient streaming state
    // ========================================================================
    bool mStreamGrads = false;

    // CPU pinned storage for ALL layers' gradients
    std::unordered_map<std::string, Tensor> mCpuGrads;

    // Double-buffered GPU gradient pool
    static constexpr int kNumGradSlots = 2;
    struct GradBufferSlot {
        std::unordered_map<std::string, Tensor> buffers;  ///< base_name → GPU tensor
        int layer_idx = -1;
        cudaEvent_t d2h_done = nullptr;      ///< D2H copy complete (safe to reuse)
        cudaEvent_t compute_done = nullptr;  ///< backward + reduce done (safe to D2H)
        cudaEvent_t reduce_done = nullptr;   ///< NCCL reduce done (safe to D2H, multi-GPU)
    };
    std::array<GradBufferSlot, kNumGradSlots> mGradSlots;
    int mActiveGradSlot = 0;

    // Per-layer gradient norm accumulator (single float on GPU)
    Tensor mLayerNormAccum;

    // CPU staging buffer for micro-step accumulation (pinned, max-layer sized)
    std::unordered_map<std::string, Tensor> mCpuStagingBuffer;

    // Track pending D2H operations for wait_all_offloads()
    std::vector<cudaEvent_t> mPendingD2HEvents;

    void allocate_streaming_buffers(const DslParamStore& params);
    void create_streaming_events();
    void destroy_streaming_events() noexcept;
    bool is_block_grad(const std::string& name) const;

    /// Strip layer index from param name: "blocks[5].attn_q_weight" → "blocks[].attn_q_weight"
    static std::string base_grad_name(const std::string& name);
};

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_DSL_GRAD_STORE_H
