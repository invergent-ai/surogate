// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Compiled operation dispatch for DSL Graph executor.
//
// This module eliminates runtime dispatch overhead by pre-compiling operations
// into direct function pointer calls with pre-resolved tensors and attributes.

#ifndef SUROGATE_SRC_EXECUTOR_COMPILED_OPS_H
#define SUROGATE_SRC_EXECUTOR_COMPILED_OPS_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

#include "runtime/dsl/forward_plan.h"
#include "runtime/ep/ep_state.h"
#include "runtime/executor/graph_executor_internal.h"
#include "runtime/dsl/ir.h"
#include "runtime/dsl/tensor_slot.h"
#include "runtime/dsl/tensor_slot_registry.h"
#include "kernels/kernels.h"
#include "runtime/lora/lora_types.h"
#include "utilities/stack.h"
#include "utilities/tensor.h"
#include "runtime/dsl/graph_compiler.h"
#include "runtime/jit/gated_delta_rule_kernels.h"

namespace ep {
class EPStrategy;
}  // namespace ep

namespace modules {
struct ModelConfig;
class ModularLoRAConfig;
class ModularLoRAWeightsManager;
class ModularLoRAGradsManager;
struct LoRARunState;
enum class MatmulOp;
enum class ForwardHookPoint;
enum class BackwardHookPoint;
using ForwardHook = std::function<void(int, cudaStream_t, ForwardHookPoint, void*)>;
using BackwardHook = std::function<void(int, bool, cudaStream_t, BackwardHookPoint, void*)>;
}  // namespace modules

namespace recipes {
class Recipe;
}  // namespace recipes

class NCCLCommunicator;
struct RuntimeOptions;

namespace dsl {

class DslRunState;
class DslParamStore;
class DslGradStore;
class DslWeightManager;

// ============================================================================
// Compiled Executor
// ============================================================================

class CompiledExecutor {
public:
    // EPStrategy moves forward/backward dispatch + combine off the executor
    // but still needs to drive the executor's tensor resolution, per-layer
    // MoE bookkeeping, and the NCCL/LoRA handles it depends on. Grant access.
    friend class ::ep::EPStrategy;

    CompiledExecutor(DslRunState& run_state,
                     DslParamStore& weights,
                     DslGradStore& grads,
                     const modules::ModelConfig& config,
                     const RuntimeOptions& options);
    ~CompiledExecutor();

    // Execute a compiled forward graph
    void
    execute_forward(const CompiledGraph& graph, NCCLCommunicator& comm, bool full, const modules::ForwardHook* hook);

    // Replay a single layer's forward ops to regenerate activations for backward.
    // This is the torch-style gradient checkpointing: save only the layer input (residual),
    // discard intermediates during forward, replay forward during backward.
    // After this call, stack contains the replayed data. The caller must restore the stack
    // after backward ops consume the data.
    void replay_layer_forward(int layer_idx,
                              long B,
                              long T,
                              const CompiledGraph& fwd_graph,
                              const modules::ForwardHook* hook);

    // Execute a compiled backward graph
    // When skip_zeroing is true, gradient buffers are assumed to already be zeroed by the caller
    // (e.g., GraphExecutor::backward_with_hook zeros them before calling execute_backward).
    void execute_backward(const CompiledGraph& graph,
                          NCCLCommunicator& comm,
                          int grad_accum_steps,
                          int micro_step,
                          const modules::BackwardHook* hook,
                          bool skip_zeroing = false);

    // Set optional components
    void set_lora_state(const modules::ModularLoRAConfig* config,
                        modules::ModularLoRAWeightsManager* weights,
                        modules::ModularLoRAGradsManager* grads,
                        modules::LoRARunState* run_state);

    void set_weight_manager(DslWeightManager* weight_manager);
    void set_recipe(const recipes::Recipe* recipe);
    void set_hook_context(void* context);
    /// Set the GPU buffer that receives per-token log P(target|context) values.
    /// When non-null, dispatch_fused_lm_head_loss writes log-probs and returns early
    /// (no loss accumulation, no gradient state update).
    void set_logprobs_context(float* logprobs_gpu) {
        mLogprobsGpu = logprobs_gpu;
    }

    /// Set per-token inverse temperatures (1 / T) for logprob/CE computation.
    /// When non-null, logits are scaled by inv_temperature before logsoftmax,
    /// and gradients are chained through the scaling in backward.
    void set_inv_temperature_context(const float* inv_temperature_gpu) {
        mInvTemperatureGpu = inv_temperature_gpu;
    }

    /// Set the GPU buffer containing per-token custom d_loss values for GRPO backward.
    /// When non-null, dispatch_fused_lm_head_loss_backward copies these values into
    /// d_loss instead of seeding with 1.0 (standard cross-entropy training).
    /// Buffer must contain B*T float32 values (same layout as the loss tensor).
    /// Lifetime must extend through the execute_backward call.
    void set_custom_dloss_context(float* custom_dloss_gpu) {
        mCustomDLossGpu = custom_dloss_gpu;
    }

    /// Set document masking context for Flash Attention varlen dispatch.
    /// When set, dispatch_flash_attention routes to flash varlen instead of cuDNN.
    void set_doc_masking_context(const std::int32_t* cu_seqlens_gpu,
                                 const std::int32_t* cu_seqlens_cpu,
                                 int num_docs,
                                 int max_seqlen,
                                 int total_q) {
        mCuSeqlensGpu = cu_seqlens_gpu;
        mCuSeqlensCpu = cu_seqlens_cpu;
        mNumDocs = num_docs;
        mMaxDocSeqlen = max_seqlen;
        mTotalDocTokens = total_q;
    }
    void clear_doc_masking_context() {
        mCuSeqlensGpu = nullptr;
        mCuSeqlensCpu = nullptr;
        mNumDocs = 0;
        mMaxDocSeqlen = 0;
        mTotalDocTokens = 0;
    }

    void set_recompute_fn(std::function<void(int, long, long, bool)> fn);
    void set_recompute_enabled(bool enabled);
    void set_recompute_use_graphs(bool enabled) {
        mRecomputeUseGraphs = enabled;
    }
    void set_capturing(bool capturing) {
        mCapturing = capturing;
    }
    void set_debug_dump_fn(std::function<void(const std::vector<std::string>&, int)> fn) {
        mDebugDumpFn = std::move(fn);
    }
    void set_debug_dump_layer_fn(std::function<void(int)> fn) {
        mDebugDumpLayerFn = std::move(fn);
    }
    void set_debug_dump_backward_layer_fn(std::function<void(int)> fn) {
        mDebugDumpBackwardLayerFn = std::move(fn);
    }

    // Cache management
    void set_fp8_cache(std::unordered_map<std::string, FP8WeightCacheEntry>* cache);
    void set_fp8_cache_transposed(std::unordered_map<std::string, FP8WeightCacheEntry>* cache_t);
    void set_fp4_cache(std::unordered_map<std::string, FP4WeightCacheEntry>* cache,
                       std::unordered_map<std::string, FP4WeightCacheEntry>* cache_t);
    void set_saved_tensors(std::unordered_map<std::string, Tensor>* saved);
    void set_save_list(const std::vector<std::string>* save_list);

    /// Bind phase-tree arenas (M5.d). When set and arenas are allocated,
    /// persist_saved_layer_tensors writes SaveForBwd tids directly into the
    /// pre-allocated arena slot instead of cudaMalloc'ing per-name buffers.
    void set_phase_arenas(dsl::PhaseArenas* arenas) {
        mPhaseArenas = arenas;
    }
    void set_forward_plan(std::vector<LayerForwardPlan>* plan) {
        mForwardPlan = plan;
    }

    // For embedding backward (requires CPU-side inputs for deterministic bucketing)
    void set_last_inputs_cpu(const Tensor* inputs_cpu);

    // RNG seed for embedding backward
    void set_rng_seed_fn(std::function<unsigned int()> fn);

    // Set embedding output names from forward graph (for binding d_embed_N to d_embeddings)
    void set_embedding_outputs(const std::vector<std::string>& names) {
        mEmbeddingOutputs = names;
    }

    // Set slot registry for DSL-driven tensor mapping
    void set_slot_registry(const TensorSlotRegistry* registry) {
        mSlotRegistry = registry;
    }

    // Set batch/sequence dimensions before execution
    void set_dimensions(long B, long T) {
        mB = B;
        mT = T;
    }

    // Expose mapped tensors for test/debug (returns nullptr if not found).
    const Tensor* try_get_tensor(const std::string& name) const;
    // Debug-only: resolve by SSA-stripped name or fallback to simplified activations.
    const Tensor* try_get_tensor_fuzzy(const std::string& name);

    // Save specified tensors to the saved map (for backward use)
    void save_tensors(const std::vector<std::string>& save_list, bool force_persist = false);
    // Preallocate persistent buffers for saved tensors before CUDA graph capture.
    // This avoids cudaMalloc during capture when recompute requires persistent saves.
    void prepare_saved_buffers_for_capture(const std::vector<std::string>& save_list,
                                           const CompiledGraph* capture_graph = nullptr);

private:
    // Execute an MLP tile group in chunks along the sequence dimension.
    // Used when long_context mode is enabled to reduce peak MLP activation memory.
    void execute_tiled_mlp(const CompiledGraph& graph,
                           const MlpTileGroup& group,
                           long B,
                           long T,
                           const modules::ForwardHook* hook);

    // Execute tiled MLP backward: combined forward recompute + backward per chunk.
    // Recomputes MLP intermediates per-chunk to avoid full-size [B*T, 2M] / [B*T, M] tensors.
    void execute_tiled_mlp_backward(const CompiledGraph& bwd_graph,
                                    const MlpTileGroup& group,
                                    long B,
                                    long T,
                                    const modules::BackwardHook* hook);

    // Save MoE layer tensors to persistent storage at layer boundaries
    void save_moe_layer_tensors(int layer_idx);

public:
    // Direct dispatch functions. Public so op_registry trampolines can
    // take their address; there is no meaningful encapsulation to lose —
    // these functions ARE the executor's per-op entry points.
    void dispatch_embedding(const CompiledOp& op);
    void dispatch_zeros(const CompiledOp& op);
    void dispatch_ones(const CompiledOp& op);
    void dispatch_fused_residual_rmsnorm(const CompiledOp& op);
    void dispatch_rmsnorm(const CompiledOp& op);
    void dispatch_layernorm(const CompiledOp& op);
    void dispatch_view(const CompiledOp& op);
    void dispatch_transpose(const CompiledOp& op);
    void dispatch_split(const CompiledOp& op);
    void dispatch_narrow(const CompiledOp& op);
    void dispatch_concat(const CompiledOp& op);
    void dispatch_add(const CompiledOp& op);
    void dispatch_matmul(const CompiledOp& op, const modules::ForwardHook* hook);
    void dispatch_bias_add(const CompiledOp& op);
    void dispatch_swiglu(const CompiledOp& op);
    void dispatch_gpt_oss_moe_act(const CompiledOp& op);
    void dispatch_silu(const CompiledOp& op);
    void dispatch_gelu(const CompiledOp& op);
    void dispatch_relu2(const CompiledOp& op);
    void dispatch_mul(const CompiledOp& op);
    void dispatch_scale(const CompiledOp& op);
    void dispatch_mask_scatter(const CompiledOp& op);
    void dispatch_deepstack_inject(const CompiledOp& op);
    void dispatch_matmul_swiglu(const CompiledOp& op, const modules::ForwardHook* hook);
    void dispatch_qkv_qk_norm(const CompiledOp& op);
    void dispatch_qkv_qk_norm_rope(const CompiledOp& op);
    void dispatch_mrope(const CompiledOp& op);
    void dispatch_rope(const CompiledOp& op);
    void dispatch_flash_attention(const CompiledOp& op);
    void dispatch_cross_entropy_loss(const CompiledOp& op);
    void dispatch_fused_lm_head_loss(const CompiledOp& op);
    // MoE forward dispatch
    void dispatch_moe_softmax(const CompiledOp& op);
    void dispatch_moe_sigmoid(const CompiledOp& op);
    void dispatch_moe_topk(const CompiledOp& op);
    void dispatch_moe_permute(const CompiledOp& op);
    void dispatch_moe_grouped_gemm(const CompiledOp& op);
    void dispatch_moe_grouped_gemm_gate_up(const CompiledOp& op);
    void dispatch_moe_grouped_gemm_down(const CompiledOp& op);
    void dispatch_moe_unpermute(const CompiledOp& op);
    void dispatch_moe_expert_bias_add(const CompiledOp& op);
    // Expert Parallelism forward dispatch
    void dispatch_ep_dispatch(const CompiledOp& op);
    void dispatch_ep_combine(const CompiledOp& op);

    // Backward dispatch functions
    void dispatch_view_backward(const CompiledOp& op);
    void dispatch_add_backward(const CompiledOp& op);
    void dispatch_matmul_backward(const CompiledOp& op, const modules::BackwardHook* hook);
    void dispatch_bias_add_backward(const CompiledOp& op);
    void dispatch_swiglu_backward(const CompiledOp& op);
    void dispatch_gpt_oss_moe_act_backward(const CompiledOp& op);
    void dispatch_silu_backward(const CompiledOp& op);
    void dispatch_gelu_backward(const CompiledOp& op);
    void dispatch_relu2_backward(const CompiledOp& op);
    void dispatch_mul_backward(const CompiledOp& op);
    void dispatch_scale_backward(const CompiledOp& op);
    void dispatch_narrow_backward(const CompiledOp& op);
    void dispatch_mask_scatter_backward(const CompiledOp& op);
    void dispatch_deepstack_inject_backward(const CompiledOp& op);
    void dispatch_matmul_swiglu_backward(const CompiledOp& op, const modules::BackwardHook* hook);
    void dispatch_qkv_qk_norm_backward(const CompiledOp& op);
    void dispatch_rope_backward(const CompiledOp& op);
    void dispatch_qkv_qk_norm_rope_backward(const CompiledOp& op);
    void dispatch_mrope_backward(const CompiledOp& op);
    void dispatch_flash_attention_backward(const CompiledOp& op);
    void dispatch_zeros_backward(const CompiledOp& op);
    void dispatch_fused_residual_rmsnorm_backward(const CompiledOp& op);
    void dispatch_rmsnorm_backward(const CompiledOp& op);
    void dispatch_layernorm_backward(const CompiledOp& op);
    void dispatch_embedding_backward(const CompiledOp& op);
    void dispatch_cross_entropy_loss_backward(const CompiledOp& op);
    void dispatch_fused_lm_head_loss_backward(const CompiledOp& op);
    // MoE backward dispatch
    void dispatch_moe_softmax_backward(const CompiledOp& op);
    void dispatch_moe_sigmoid_backward(const CompiledOp& op);
    void dispatch_moe_topk_backward(const CompiledOp& op);
    void dispatch_moe_permute_backward(const CompiledOp& op);
    void dispatch_moe_grouped_gemm_backward(const CompiledOp& op);
    void dispatch_moe_grouped_gemm_gate_up_backward(const CompiledOp& op);
    void dispatch_moe_grouped_gemm_down_backward(const CompiledOp& op);
    void dispatch_moe_unpermute_backward(const CompiledOp& op);
    void dispatch_moe_expert_bias_add_backward(const CompiledOp& op);
    // Expert Parallelism backward dispatch
    void dispatch_ep_dispatch_backward(const CompiledOp& op);
    void dispatch_ep_combine_backward(const CompiledOp& op);

    // Mamba/SSM forward dispatch
    void dispatch_mamba_split_proj(const CompiledOp& op);
    void dispatch_mamba_conv1d(const CompiledOp& op);
    void dispatch_mamba_split_conv_out(const CompiledOp& op);
    void dispatch_mamba_ssm_scan(const CompiledOp& op);
    void dispatch_mamba_gated_rmsnorm(const CompiledOp& op);
    void dispatch_mamba_out_proj(const CompiledOp& op, const modules::ForwardHook* hook);
    // Qwen3.5 gated delta rule forward dispatch
    void dispatch_gated_delta_rule_common(const CompiledOp& op, const char* op_name);
    void dispatch_chunk_gated_delta_rule(const CompiledOp& op);
    void dispatch_chunk_gated_delta_rule_backward(const CompiledOp& op);
    void dispatch_qwen3_5_decay(const CompiledOp& op);
    void dispatch_qwen3_5_decay_backward(const CompiledOp& op);
    void dispatch_repeat_interleave_heads(const CompiledOp& op);
    void dispatch_repeat_interleave_heads_backward(const CompiledOp& op);

    // Mamba/SSM backward dispatch
    void dispatch_mamba_split_proj_backward(const CompiledOp& op);
    void dispatch_mamba_conv1d_backward(const CompiledOp& op);
    void dispatch_mamba_split_conv_out_backward(const CompiledOp& op);
    void dispatch_mamba_ssm_scan_backward(const CompiledOp& op);
    void dispatch_mamba_gated_rmsnorm_backward(const CompiledOp& op);
    void dispatch_mamba_out_proj_backward(const CompiledOp& op, const modules::BackwardHook* hook);

private:
    // Tensor resolution (pre-resolved, O(1) lookup)
    Tensor& resolve_tensor(const TensorRef& ref);
    Tensor& ensure_output_tensor(const TensorRef& ref);

    /// Runtime check: for the op about to dispatch (or just dispatched),
    /// resolve every input and output to its current (Data, bytes) pair and
    /// assert no input/output byte range overlaps. cuBLAS GEMM and similar
    /// kernels have undefined behavior when input and output aliasing —
    /// this check catches such bugs early (e.g., compile-time arena
    /// coloring that didn't account for runtime slot-alias view resolution).
    /// No-op unless SUROGATE_CHECK_OP_IO_ALIASING=1. Runs before op.fn;
    /// logs the offending (input, output) pair and, if the env is set to
    /// "abort", throws to halt execution.
    void check_op_io_aliasing(const CompiledOp& op, std::size_t op_idx, const char* phase);
    Tensor* try_resolve_saved_live(const std::string& name, const Tensor& saved);
    Tensor resolve_moe_expert_offsets(const CompiledOp& op);

    // Get host-side MoE expert offsets for a layer, using cache or syncing from device.
    const int* get_or_sync_moe_host_offsets(int layer_idx, const int* device_offsets, int num_experts);

    // EP replay-aware cache key helpers.
    // For EP, replay forward can run before backward and overwrite per-layer metadata.
    // Keying EP metadata by (layer, replay_slot) prevents replay=1 clobbering replay=0.
    [[nodiscard]] int ep_state_key(int layer_idx) const {
        if (layer_idx < 0) return layer_idx;
        if (mOptions.EPSize <= 1) return layer_idx;
        return (layer_idx << 1) | (mInReplay ? 1 : 0);
    }
    [[nodiscard]] std::string moe_saved_key(int layer_idx, const char* suffix) const {
        std::string key = "blocks[" + std::to_string(layer_idx) + "]." + suffix;
        if (mOptions.EPSize > 1) {
            key += (mInReplay ? "#r1" : "#r0");
        }
        return key;
    }

    // Layer boundary handling
    void handle_layer_start(int layer_idx);
    void handle_layer_end(int layer_idx);

    // State
    DslRunState& mRunState;
    DslParamStore& mWeights;
    DslGradStore& mGrads;
    const modules::ModelConfig& mConfig;
    const RuntimeOptions& mOptions;

    // Expert Parallelism strategy: owns all per-layer EP/LLEP state, the GPU
    // buffer pool, shared cross-layer buffers, and the weight-transfer stream.
    // Built via ep::create_strategy(); defaults to LLEP with per-layer
    // auto-fallback to static mapping when rebalancing isn't warranted.
    //
    // Declared here (immediately after mOptions) so the reference aliases
    // below can be initialized in the same list-order as declaration,
    // avoiding -Wreorder warnings.
    std::unique_ptr<ep::EPStrategy> mEpStrategy;

    // Reference aliases into the strategy's state maps. Existing MoE/EP
    // dispatch code keeps using `mEpStates`/`mLLEPStates`/`mEPLayerMeta`
    // unchanged; each name now resolves to a map owned by the strategy.
    std::unordered_map<int, ep::EpLayerState>& mEpStates;
    std::unordered_map<int, ep::LLEPLayerState>& mLLEPStates;
    std::unordered_map<int, ep::EPLayerMeta>& mEPLayerMeta;

    // JIT-compiled Triton kernels for gated delta rule (loaded once from manifests)
    GatedDeltaRuleKernels mGdrKernels;

    // Log-prob extraction context (null in training mode)
    float* mLogprobsGpu = nullptr;

    // Custom per-token d_loss for GRPO backward (null = standard d_loss=1 seeding)
    float* mCustomDLossGpu = nullptr;
    const float* mInvTemperatureGpu = nullptr;

    // Document masking context for Flash Attention varlen (null = disabled)
    const std::int32_t* mCuSeqlensGpu = nullptr;
    const std::int32_t* mCuSeqlensCpu = nullptr;
    int mNumDocs = 0;
    int mMaxDocSeqlen = 0;
    int mTotalDocTokens = 0;

    // Optional components
    const modules::ModularLoRAConfig* mLoRAConfig = nullptr;
    modules::ModularLoRAWeightsManager* mLoRAWeights = nullptr;
    modules::ModularLoRAGradsManager* mLoRAGrads = nullptr;
    modules::LoRARunState* mLoRARunState = nullptr;
    std::vector<char> mLoraBSeenClean;
    DslWeightManager* mWeightManager = nullptr;
    const recipes::Recipe* mRecipe = nullptr;
    void* mHookContext = nullptr;
    std::function<void(int, long, long, bool)> mRecomputeFn;
    bool mRecomputeEnabled = false;
    bool mRecomputeUseGraphs = true;
    int mLastRecomputeLayer = -1;
    NCCLCommunicator* mComm = nullptr;

    // Caches
    std::unordered_map<std::string, FP8WeightCacheEntry>* mFP8Cache = nullptr;
    std::unordered_map<std::string, FP8WeightCacheEntry>* mFP8CacheT = nullptr;
    std::unordered_map<std::string, FP4WeightCacheEntry>* mFP4Cache = nullptr;
    std::unordered_map<std::string, FP4WeightCacheEntry>* mFP4CacheT = nullptr;
    std::unordered_map<std::string, Tensor>* mSaved = nullptr;
    const std::vector<std::string>* mSaveList = nullptr;  // Tensors to preserve for backward
    dsl::PhaseArenas* mPhaseArenas = nullptr;             // Non-owning; set via set_phase_arenas
    // Phase 3 subsystem #4: per-step bump cursor into the bwd_cross_layer arena.
    // Reset at the start of each backward call; advanced by allocate_bwd_cross_layer.
    std::size_t mBwdCrossLayerBumpOffset = 0;

    // Phase 3 subsystem #6: cross-step monotonic bump cursor into moe_saved
    // arena. Never reset - MoE save buffers are keyed by name + persist for the
    // executor's lifetime. Size growth re-bumps (same semantics as today's
    // cudaFree+cudaMalloc cycle, which also wastes the old buffer).
    std::size_t mMoeSavedBumpOffset = 0;
    std::unordered_map<std::string, bool> mMoeSavedArenaBacked;

    /// Allocate `nbytes` for a backward cross-layer persistence. Uses
    /// mPhaseArenas.bwd_cross_layer_ptr when bound and sufficient space remains,
    /// else falls back to cudaMalloc. Returns the buffer pointer via `out` and
    /// sets `arena_backed` true if the arena was used; caller must add
    /// cudaMalloc-backed buffers to mPersistedBackwardTensors for later free.
    struct BwdXLayerAlloc {
        std::byte* ptr = nullptr;
        bool arena_backed = false;
    };
    BwdXLayerAlloc allocate_bwd_cross_layer(std::size_t nbytes);

    /// Allocate `nbytes` for an MoE save buffer. Cross-step monotonic bump in
    /// mPhaseArenas.moe_saved_ptr; cudaMalloc fallback when arena exhausted or
    /// unbound. Caller owns cudaFree of non-arena_backed pointers.
    struct MoeSavedAlloc {
        std::byte* ptr = nullptr;
        bool arena_backed = false;
    };
    MoeSavedAlloc allocate_moe_saved(std::size_t nbytes);
    std::unordered_set<std::string> mSaveSet;  // Fast lookup for save list
    std::vector<LayerForwardPlan>* mForwardPlan = nullptr;
    std::function<void(const std::vector<std::string>&, int)> mDebugDumpFn;
    std::function<void(int)> mDebugDumpLayerFn;
    std::function<void(int)> mDebugDumpBackwardLayerFn;

    // For embedding backward
    const Tensor* mLastInputsCpu = nullptr;
    std::function<unsigned int()> mRngSeedFn;
    std::vector<std::string> mEmbeddingOutputs;         // Forward graph embedding output names
    const TensorSlotRegistry* mSlotRegistry = nullptr;  // DSL slot registry for global gradient binding

    // Execution state
    long mB = 0;
    long mT = 0;
    int mMicroStep = 0;
    int mCurrentLayer = -1;
    int mPrefetchDirection = 1;  // +1 for forward, -1 for backward
    bool mCapturing = false;
    bool mInReplay = false;    ///< True during replay_layer_forward
    int mReplayLayerIdx = -1;  ///< Layer being replayed

    // Deferred stack checkpoint from replay_layer_forward.
    // Stack restore is deferred until backward ops consume the replay data.
    bool mHasDeferredReplayCheckpoint = false;
    DeviceMemoryStack::Checkpoint mDeferredReplayCheckpoint{};
    std::size_t mDeferredReplayTempMark = 0;
    std::vector<void*> mReplayCopiedBuffers;  // persistent copies of stack-resident saved tensors

    // Pre-allocated arena used by replay_layer_forward's stack-to-persistent
    // copies. The arena's base pointer is stable across CUDA graph captures
    // and replays (unlike cudaMallocAsync inside a captured graph, which
    // bakes a capture-time pointer that's re-allocated at replay, yielding
    // stale pointers in captured kernels). Bump-allocated; offset reset at
    // each replay-layer entry so only the current layer's copies occupy
    // the arena.
    std::byte* mReplayPersistArena = nullptr;
    std::size_t mReplayPersistCapacity = 0;
    std::size_t mReplayPersistOffset = 0;

    /// Bump-allocate `bytes` from the replay-persist arena. Allocates the
    /// arena lazily on first call, rounded up to at least 256 MiB. Returns
    /// nullptr if the request is larger than what the arena can provide
    /// (caller falls back to cudaMallocAsync).
    std::byte* allocate_replay_persist(std::size_t bytes);

    /// Pre-allocate the replay-persist arena if not yet allocated. Must be
    /// called outside any CUDA stream capture (cudaMalloc during capture
    /// is illegal). Called by execute_forward at step entry so subsequent
    /// backward capture sees a stable arena base pointer.
    void ensure_replay_persist_arena();

    // Temporary tensor storage (for stack-allocated tensors)
    std::vector<Tensor> mTemps;

    // Integer-indexed tensor storage (flat vector indexed by compile-time tensor IDs).
    // Indexed by TensorRef::tensor_id assigned during graph compilation.
    // Eliminates string hashing/comparison in the hot resolve_tensor path.
    std::vector<Tensor> mTensors;
    // Name-indexed tensor overrides for cases where multiple tensor names share one tensor_id/slot.
    std::unordered_map<std::string, Tensor> mNamedTensors;
    std::vector<bool> mSaveMask;  // Per-tensor-id: true if in save list (for prune)
    const CompiledGraph* mCurrentGraph = nullptr;

    // Bind a named tensor into the flat vector using the current graph's name-to-id map.
    // No-op if the name is not in the graph (e.g., optional visual embeds when disabled).
    void bind_tensor(const std::string& name, const Tensor& t) {
        if (!name.empty()) {
            mNamedTensors[name] = t;
        }
        if (mCurrentGraph) {
            int id = mCurrentGraph->find_tensor_id(name);
            if (id >= 0 && id < static_cast<int>(mTensors.size())) {
                mTensors[static_cast<std::size_t>(id)] = t;
            }
        }
    }

    // Store a tensor by its pre-resolved TensorRef into the flat vector.
    // Use this in dispatch functions instead of direct mTensors assignment.
    void store_tensor(const TensorRef& ref, const Tensor& t) {
        if (!ref.name.empty()) {
            mNamedTensors[ref.name] = t;
        }
        if (ref.tensor_id >= 0 && static_cast<std::size_t>(ref.tensor_id) < mTensors.size()) {
            mTensors[ref.tensor_id] = t;
        }
    }

    /// Phase 4 M3 prerequisite. Invalidate cached Tensor entries for a slot
    /// whose backing storage is about to be freed (e.g. a Stack-allocated
    /// simplified-activation slot at layer_end). Erases mNamedTensors[name]
    /// AND nulls mTensors[tid].Data so resolve_tensor / ensure_output_tensor
    /// on the next access re-resolve rather than returning a dangling
    /// pointer. Safe on unknown names. Required before extending the
    /// ffn_temps_on_stack pattern to slots whose backward consumers read
    /// them via resolve_tensor(op.inputs[...]) — otherwise the
    /// mNamedTensors cache survives Stack.restore with a stale pointer.
    void invalidate_cached_slot(const std::string& name) {
        if (!name.empty()) {
            mNamedTensors.erase(name);
        }
        if (mCurrentGraph) {
            const int id = mCurrentGraph->find_tensor_id(name);
            if (id >= 0 && id < static_cast<int>(mTensors.size())) {
                mTensors[static_cast<std::size_t>(id)].Data = nullptr;
            }
        }
    }

    // Gradient accumulation tracking (set of gradient tensor names that need accumulation)
    std::unordered_set<std::string> mAccumulateTensors;

    // Cross-layer backward tensors persisted from stack to cudaMalloc.
    // Freed at end of backward pass.
    std::vector<std::byte*> mPersistedBackwardTensors;

    // Persistent storage for MoE expert_offsets (needs to survive from forward to backward)
    std::vector<int> mMoEExpertOffsetsData;
    Tensor mMoEExpertOffsets;              // Views into mMoEExpertOffsetsData
    void* mMoEExpertOffsetsGPU = nullptr;  // Persistent GPU buffer (not stack-allocated)
    size_t mMoEExpertOffsetsGPUSize = 0;   // Size in bytes

    // Host-side MoE expert offsets cache.
    // Key is layer_idx for non-EP, ep_state_key(layer_idx) for EP.
    // Populated once per key (forward: in permute/ep_dispatch; backward: on first access).
    // Avoids redundant D2H synchronization in grouped GEMM ops within the same layer.
    std::unordered_map<int, std::vector<int>> mMoEHostOffsetsCache;

    // Persistent storage for MoE saved tensors (per-layer copies to prevent buffer reuse corruption)
    // Maps tensor name to persistent GPU buffer (cudaMalloc'd, NOT from stack allocator)
    std::unordered_map<std::string, void*> mMoeSavedBuffers;
    std::unordered_map<std::string, size_t> mMoeSavedSizes;

    void clear_replay_copied_refs();

    // Reusable per-forward layer tracking vectors (avoid heap allocation every forward call).
    std::vector<DeviceMemoryStack::Checkpoint> mLayerCheckpoints;
    std::vector<std::size_t> mLayerTempMarks;
    std::vector<char> mLayerActive;

    // ========================================================================
    // Split-attention CUDA graph mode
    // ========================================================================
    // When enabled, each layer's forward/backward is split into segments
    // around FlashAttention ops. Non-attention segments are captured as CUDA
    // graphs; attention runs eagerly with dynamic cu_seqlens (doc masking).
    bool mSplitAttentionGraphs = false;
    std::size_t mSegmentDispatchedUntil = 0;  ///< Ops before this index already dispatched by segments

    struct SegmentGraphExec {
        cudaGraphExec_t exec = nullptr;
        DeviceMemoryStack::Checkpoint checkpoint{};
        DeviceMemoryStack::Checkpoint post_checkpoint{};  // stack state AFTER dispatch (for replay advance)
        // Tensor entries produced during capture. On graph replay the dispatch
        // functions don't run, so mTensors isn't populated. We snapshot the
        // entries after the initial capture and restore them on replay so that
        // cross-segment tensor lookups (e.g. attention reading QKV) still work.
        // Stack addresses are stable because the checkpoint is restored before replay.
        std::vector<std::pair<int, Tensor>> tensor_snapshot;         // by tensor_id
        std::vector<std::pair<std::string, Tensor>> named_snapshot;  // by name
        std::vector<std::pair<std::string, Tensor>> saved_snapshot;  // mSaved entries written by dispatch
    };

    // Forward segment graphs: [layer_idx][segment_idx]
    std::vector<std::vector<SegmentGraphExec>> mFwdSegGraphs;
    // Backward segment graphs: [accum_mode 0/1][layer_idx][segment_idx]
    std::vector<std::vector<SegmentGraphExec>> mBwdSegGraphs[2];
    long mSegGraphB = 0, mSegGraphT = 0;

    /// Dispatch a single forward op (extracted from the switch in execute_forward).
    void dispatch_forward_op(const CompiledOp& op, const modules::ForwardHook* hook);
    /// Dispatch a single backward op (extracted from the switch in execute_backward).
    void dispatch_backward_op(const CompiledOp& op, const modules::BackwardHook* hook);

public:
    void set_split_attention_graphs(bool enabled) {
        mSplitAttentionGraphs = enabled;
    }
    void reset_segment_graphs();
    void resize_segment_graphs(const CompiledGraph& fwd_graph, const CompiledGraph& bwd_graph);
    /// Total bytes of persistent saved buffers (untracked by TensorAllocator).
    size_t saved_buffers_total_bytes() const {
        size_t total = 0;
        for (const auto& [name, sz] : mMoeSavedSizes)
            total += sz;
        return total;
    }
    /// Number of persistent saved buffers.
    int saved_buffers_count() const {
        return static_cast<int>(mMoeSavedSizes.size());
    }
    /// Per-buffer sizes for diagnostics.
    const std::unordered_map<std::string, size_t>& saved_buffers_sizes() const {
        return mMoeSavedSizes;
    }
};

// ============================================================================
// Utility functions
// ============================================================================

// Convert string operation type to enum (used during compilation)
CompiledOpType op_type_from_string(const std::string& op_type);

// Convert enum to string (for debugging)
const char* op_type_to_string(CompiledOpType type);

}  // namespace dsl

#endif  // SUROGATE_SRC_EXECUTOR_COMPILED_OPS_H
