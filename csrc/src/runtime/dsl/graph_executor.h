// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL Graph executor (DSL-driven).

#ifndef SUROGATE_SRC_DSL_GRAPH_EXECUTOR_H
#define SUROGATE_SRC_DSL_GRAPH_EXECUTOR_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "runtime/dsl/graph_executor_internal.h"
#include "runtime/dsl/ir.h"
#include "runtime/dsl/forward_plan.h"
#include "runtime/core/forward_hooks.h"
#include "runtime/core/backward_hooks.h"
#include "runtime/infer/decode_state.h"
#include "utilities/stack.h"
#include "utilities/tensor.h"

class NCCLCommunicator;
struct RuntimeOptions;

namespace modules {
struct ModelConfig;
struct ModularLoRAConfig;
class ModularLoRAWeightsManager;
class ModularLoRAGradsManager;
struct LoRARunState;
struct MatmulContext;
enum class MatmulOp;
}
namespace dsl {
class DslRunState;
class DslParamStore;
class DslGradStore;
class DslWeightManager;
class GraphCompiler;
class CompiledExecutor;
struct CompiledGraph;
struct FP8WeightCacheEntry;
struct FP4WeightCacheEntry;
}

namespace dsl {

// Options for GraphExecutor construction
struct GraphExecutorOptions {
    // If true and module has no backward graph, derive one automatically using autodiff
    bool auto_backward = false;

    // Name of the loss tensor for autodiff (used when deriving backward)
    std::string loss_name = "loss";

    // Whether to print derived backward graph for debugging
    bool debug_print_backward = false;
};

class IGraphExecutor {
public:
    virtual ~IGraphExecutor() = default;

    virtual void forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) = 0;
    virtual float validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) = 0;
    virtual void backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) = 0;

    // Hook-enabled forward/backward methods (matching ModularModel interface)
    virtual void forward_with_hook(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step,
                                   const modules::ForwardHook& hook) {
        // Default: forward without hooks
        forward(inputs, position_ids, comm, micro_step);
    }

    virtual float validate_with_hook(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm,
                                     int micro_step, const modules::ForwardHook& hook) {
        // Default: validate without hooks
        return validate(inputs, position_ids, targets, comm, micro_step);
    }

    virtual void backward_with_hook(Tensor inputs, Tensor targets, NCCLCommunicator& comm,
                                    int grad_accum_steps, int micro_step, const modules::BackwardHook& hook) {
        // Default: backward without hooks
        backward(inputs, targets, comm, grad_accum_steps, micro_step);
    }

    // Check if backward graph was auto-derived
    virtual bool has_derived_backward() const = 0;

    // Get the backward graph (either from module or derived)
    virtual const Graph* backward_graph() const = 0;

    // RNG state helpers (for checkpointing/repro)
    virtual std::vector<std::byte> rng_state() const = 0;
    virtual void set_rng_state(const std::vector<std::byte>& state) = 0;

    // Control internal CUDA graph usage (forward/backward graphs inside DSL executor).
    virtual void set_internal_graphs_enabled(bool enabled) { (void)enabled; }
    virtual bool internal_graphs_enabled() const { return false; }

    // Optional LoRA state wiring (no-op for implementations that don't support it).
    virtual void set_lora_state(const modules::ModularLoRAConfig*,
                                modules::ModularLoRAWeightsManager*,
                                modules::ModularLoRAGradsManager*,
                                modules::LoRARunState*) {}

    // Set optional hook context (opaque pointer passed to hooks)
    virtual void set_hook_context(void* context) { (void)context; }

    /// Total bytes of untracked persistent saved buffers (raw cudaMalloc).
    virtual size_t saved_buffers_total_bytes() const { return 0; }
    /// Number of persistent saved buffers.
    virtual int saved_buffers_count() const { return 0; }
    /// Per-buffer sizes for diagnostics.
    virtual const std::unordered_map<std::string, size_t>& saved_buffers_sizes() const {
        static const std::unordered_map<std::string, size_t> empty;
        return empty;
    }

    /// Document masking for packed sequences (Flash Attention varlen).
    virtual void set_doc_masking(const std::int32_t* /*cu_seqlens_cpu*/, int /*num_docs*/,
                                 int /*max_seqlen*/, int /*total_q*/) {}
    virtual void clear_doc_masking() {}
};

class GraphExecutor final : public IGraphExecutor {
public:
    GraphExecutor(const Module& module,
                  DslRunState& run_state,
                  DslParamStore& weights,
                  DslGradStore& grads,
                  const modules::ModelConfig& config,
                  const RuntimeOptions& options,
                  const GraphExecutorOptions& exec_options = {});
    ~GraphExecutor() override;

    void set_lora_state(const modules::ModularLoRAConfig* config,
                        modules::ModularLoRAWeightsManager* weights,
                        modules::ModularLoRAGradsManager* grads,
                        modules::LoRARunState* run_state);

    // Set optional weight manager for streaming/sharding
    void set_weight_manager(DslWeightManager* weight_manager);

    void forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) override;
    float validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) override;
    void backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) override;

    // Hook-enabled methods (matching ModularModel interface for LoRA integration)
    void forward_with_hook(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step,
                           const modules::ForwardHook& hook) override;
    float validate_with_hook(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm,
                             int micro_step, const modules::ForwardHook& hook) override;
    void backward_with_hook(Tensor inputs, Tensor targets, NCCLCommunicator& comm,
                            int grad_accum_steps, int micro_step, const modules::BackwardHook& hook) override;

    void set_hook_context(void* context) override { mHookContext = context; }

    bool has_derived_backward() const override { return mDerivedBackward.has_value(); }
    const Graph* backward_graph() const override { return mBackward; }

    std::vector<std::byte> rng_state() const;
    void set_rng_state(const std::vector<std::byte>& state);
    void set_internal_graphs_enabled(bool enabled) override;
    bool internal_graphs_enabled() const override;

    size_t saved_buffers_total_bytes() const override;
    int saved_buffers_count() const override;
    const std::unordered_map<std::string, size_t>& saved_buffers_sizes() const override;

    /// Execute a forward pass to extract per-token log-probabilities.
    ///
    /// input_ids_cpu: CPU int32 token IDs, shape [B*T] (row-major).
    /// targets_cpu:   CPU int32 target IDs, shape [B*T]; -100 = masked.
    /// logprobs_cpu:  CPU output buffer, shape [B*T]; receives log P(target|context).
    ///                Masked positions (target == -100) receive 0.
    /// hook:          Optional LoRA forward hook (nullptr to skip LoRA, e.g. reference model).
    void execute_logprobs_forward(long B, long T,
                                   const std::int32_t* input_ids_cpu,
                                   const std::int32_t* targets_cpu,
                                   float* logprobs_cpu,
                                   const modules::ForwardHook* hook,
                                   NCCLCommunicator& comm,
                                   const std::int32_t* position_ids_cpu = nullptr,
                                   const float* temperatures_cpu = nullptr);

    /// Execute a backward pass with custom per-token d_loss values (for GRPO).
    ///
    /// Identical to backward_with_hook() except the d_loss tensor is seeded from
    /// per_token_grads_cpu instead of being filled with 1.0. This allows Python
    /// to feed externally-computed per-token GRPO gradients back through the model.
    ///
    /// per_token_grads_cpu: CPU float32 buffer of shape [B*T].
    ///   Values represent dL_GRPO/d(log_prob_policy)[t] for each token.
    ///   Masked positions should be 0.
    /// hook: Optional backward hook (for LoRA gradient computation; may be nullptr).
    void backward_with_custom_dloss(Tensor inputs, Tensor targets,
                                     const float* per_token_grads_cpu,
                                     NCCLCommunicator& comm,
                                     int grad_accum_steps, int micro_step,
                                     const modules::BackwardHook* hook,
                                     const float* temperatures_cpu = nullptr);

    /// Same as backward_with_custom_dloss(), but per-token gradients are
    /// already on GPU (shape [B*T], FP32).
    void backward_with_custom_dloss_gpu(Tensor inputs, Tensor targets,
                                        const float* per_token_grads_gpu,
                                        NCCLCommunicator& comm,
                                        int grad_accum_steps, int micro_step,
                                        const modules::BackwardHook* hook,
                                        const float* inv_temperature_gpu = nullptr);

    /// Execute a full-sequence prefill forward, populating the KV-cache.
    ///
    /// Runs the compiled forward graph with B sequences and T=prompt_len.
    /// The DecodeState's prefill_mode flag causes dispatch_rope to bulk-store
    /// K/V into the KV-cache, while self-attention runs normally (no decode interception).
    /// The fused LM head writes logits to DecodeState::logits_out_gpu if set.
    ///
    /// After this call, the KV-cache contains entries for positions [0, T) per batch item.
    void execute_prefill(long B, long T,
                         const std::int32_t* token_ids_cpu,
                         const std::int32_t* position_ids_cpu,
                         infer::DecodeState& decode_state,
                         NCCLCommunicator& comm,
                         const modules::ForwardHook* hook = nullptr);

    /// Execute a single-token decode step with KV-cache.
    ///
    /// Runs the compiled forward graph with B sequences and T=1,
    /// using the provided DecodeState for KV-cache read/write.
    /// After this call, the logits for the next token are in the
    /// run state's activation buffers (accessible via the compiled executor).
    ///
    /// @param B           Batch size (number of sequences).
    /// @param token_ids   [B] last token IDs on GPU (int32).
    /// @param position_ids [B] position IDs on GPU (int32).
    /// @param decode_state KV-cache and decode tracking state.
    /// @param comm        NCCL communicator (for DP gradient sync — unused in decode).
    /// @param hook        Optional LoRA forward hook.
    /// @param use_cuda_graph  If true, capture the first call and replay on subsequent calls.
    ///                        Requires fixed batch size across calls. Input buffers (token_ids,
    ///                        position_ids) must be at stable GPU addresses updated in-place.
    void execute_decode_step(long B,
                             const std::int32_t* token_ids_gpu,
                             const std::int32_t* position_ids_gpu,
                             infer::DecodeState& decode_state,
                             NCCLCommunicator& comm,
                             const modules::ForwardHook* hook = nullptr,
                             bool use_cuda_graph = false);

    /// Estimate the peak stack usage during backward execution.
    ///
    /// Compiles the backward graph (if not already compiled) and walks its ops to
    /// compute the maximum stack-resident memory within any single layer's backward
    /// pass.  This accounts for both graph-level Temporary-slot outputs and known
    /// op-internal temporaries (e.g. ChunkGatedDeltaRuleBackward).
    ///
    /// @return Estimated peak stack bytes (0 if no backward graph exists).
    long estimate_backward_stack_peak(long B, long T);

    /// Set document masking context for Flash Attention varlen dispatch.
    /// cu_seqlens_cpu: (num_docs + 1,) int32 cumulative token offsets on CPU.
    /// Copies to GPU and propagates to CompiledExecutor.
    void set_doc_masking(const std::int32_t* cu_seqlens_cpu, int num_docs,
                         int max_seqlen, int total_q);
    void clear_doc_masking();
    void set_inv_temperature_context(const float* inv_temperature_gpu);

    /// Invalidate the cached decode CUDA graph.
    /// Must be called after generate() when the arena is restored, since
    /// the graph captures pointers into the arena that become stale.
    void invalidate_decode_graph();

private:
    void init(const GraphExecutorOptions& options);
    void reset_cuda_graphs();
    void reset_forward_plan();
    void record_matmul_plan(int layer_idx, modules::MatmulOp op, const MatmulForwardPlan& plan);
    void record_attn_plan(int layer_idx, const AttnForwardPlan& plan);
    const LayerForwardPlan* forward_plan(int layer_idx) const;

    // Internal hook invocation helpers
    void invoke_forward_hook(int layer_idx, modules::ForwardHookPoint point, cudaStream_t stream,
                             const modules::ForwardHook* hook) {
        if (hook && *hook) {
            (*hook)(layer_idx, stream, point, mHookContext);
        }
    }

    void invoke_backward_hook(int layer_idx, bool accumulate, modules::BackwardHookPoint point,
                              cudaStream_t stream, const modules::BackwardHook* hook) {
        if (hook && *hook) {
            (*hook)(layer_idx, accumulate, stream, point, mHookContext);
        }
    }

    const Module& mModule;
    DslRunState& mRunState;
    DslParamStore& mWeights;
    DslGradStore& mGrads;
    const modules::ModelConfig& mConfig;
    const RuntimeOptions& mOptions;

    // Optional LoRA state (owned by DslModel)
    const modules::ModularLoRAConfig* mLoRAConfig = nullptr;
    modules::ModularLoRAWeightsManager* mLoRAWeights = nullptr;
    modules::ModularLoRAGradsManager* mLoRAGrads = nullptr;
    modules::LoRARunState* mLoRARunState = nullptr;

    // Hook context (opaque pointer passed to hook callbacks)
    void* mHookContext = nullptr;

    // Forward hook stored for replay (LoRA deltas must be applied during recompute)
    modules::ForwardHook mReplayForwardHook;
    bool mHasReplayForwardHook = false;

    // Optional weight manager for streaming/sharding (owned by DslModel)
    DslWeightManager* mWeightManager = nullptr;

    const Graph* mForward;
    const Graph* mBackward;

    // Holds derived backward graph if autodiff was used
    std::optional<Graph> mDerivedBackward;
    // Holds a reordered backward graph when we need a mutable copy
    std::optional<Graph> mReorderedBackward;

    // Combined save list (forward.save + autodiff-computed saves)
    std::vector<std::string> mSaveList;
    std::vector<std::string> mEmbeddingOutputs;

    std::unordered_map<std::string, Tensor> mSaved;
    std::unordered_map<std::string, std::string> mViewSources;
    std::unordered_map<std::string, std::string> mViewSourcesReverse;
    Tensor mLastInputsCpu{};
    bool mFP8ScalingInitialized = false;
    std::vector<LayerForwardPlan> mForwardPlan;

    // FP8/FP4 weight caches (use namespace-level types for compatibility with CompiledExecutor)
    std::unordered_map<std::string, FP8WeightCacheEntry> mFP8WeightCache;
    std::unordered_map<std::string, FP8WeightCacheEntry> mFP8WeightCacheT;  ///< Backward dinp (transposed layout)
    std::unordered_map<std::string, FP4WeightCacheEntry> mFP4WeightCache;    ///< Forward pass (normal layout)
    std::unordered_map<std::string, FP4WeightCacheEntry> mFP4WeightCacheT;   ///< Backward dgrad (transposed layout)

    unsigned int next_rng_seed();

    void prime_fp8_weight_cache(const std::vector<char>& required);
    const Tensor* get_fp8_cached_weight(const std::string& name, Tensor& weight, cudaStream_t stream);
    void prime_fp8_weight_cache_transposed(const std::vector<char>& required);
    const Tensor* get_fp8_cached_weight_transposed(const std::string& name, Tensor& weight, cudaStream_t stream);

    // FP4 weight cache helpers (for NVFP4 recipe on Blackwell+)
    void prime_fp4_weight_cache(const std::vector<char>& required, bool prime_backward = true);
    const FP4WeightCacheEntry* get_fp4_cached_weight(const std::string& name, Tensor& weight, cudaStream_t stream);
    const FP4WeightCacheEntry* get_fp4_cached_weight_transposed(const std::string& name, Tensor& weight, cudaStream_t stream);

    // Weight prefetching for layer-by-layer execution
    void prefetch_layer_weights(int layer_idx, cudaStream_t stream);
    void wait_for_prefetch(int layer_idx, cudaStream_t stream);
    void build_layer_weight_map();
    void build_layer_boundaries();  // Pre-compute layer start/end operation indices

    std::minstd_rand mRng{42};

    // Document masking (Flash Attention varlen) — GPU cu_seqlens buffer
    std::int32_t* mCuSeqlensGpu = nullptr;
    int mCuSeqlensCount = 0;  // num_docs + 1
    int mDocMaskingNumDocs = 0;
    int mDocMaskingMaxSeqlen = 0;
    int mDocMaskingTotalQ = 0;

    // Layer-to-weight-names map for prefetching
    std::vector<std::vector<std::string>> mLayerWeightNames;
    int mPrefetchedLayer = -1;
    cudaEvent_t mPrefetchEvent = nullptr;
    bool mPrefetchEnabled = false;
    bool mHasLossOp = false;
    bool mWeightCachesPrimed = false;  // True after first eager FP8/FP4 cache priming

    // Pre-computed layer boundaries for predictable prefetch
    struct LayerBoundary {
        int layer_idx = -1;
        std::size_t start_op_idx = 0;  // First operation for this layer
        std::size_t end_op_idx = 0;    // One past last operation for this layer
    };
    std::vector<LayerBoundary> mLayerBoundaries;  // Sorted by start_op_idx

    // CUDA graph capture (optional)
    bool mGraphsEnabled = false; // Forward graphs
    bool mBackwardGraphsEnabled = false;
    bool mBackwardGraphCapturable = true;
    std::size_t mBackwardGraphCut = 0;
    bool mLoggedQwen35LoraFwdGraphDisable = false;
    bool mLoggedQwen35LoraBwdGraphDisable = false;
    long mGraphB = 0;
    long mGraphT = 0;
    cudaGraphExec_t mForwardGraph = nullptr;
    cudaGraphExec_t mBackwardGraph[2]{nullptr, nullptr}; // [0]=accumulate false, [1]=true
    DeviceMemoryStack::Checkpoint mForwardCheckpoint{};
    DeviceMemoryStack::Checkpoint mBackwardCheckpoint[2]{};

    // Per-layer CUDA graph execution (more granular than whole-graph capture)
    bool mPerLayerGraphsEnabled = false;

    // ========================================================================
    // Decode CUDA graph capture (Phase 4)
    // ========================================================================
    struct DecodeGraphSignature {
        long Batch = 0;
        const void* InputsData = nullptr;
        const void* PositionIDsData = nullptr;
        const void* SeqLensGpu = nullptr;
        const void* CuSeqlensQGpu = nullptr;
        const void* CuSeqlensKGpu = nullptr;
        const void* FinishedGpu = nullptr;
        const void* LogitsOutGpu = nullptr;
        const void* KData = nullptr;
        const void* VData = nullptr;
        const void* KPages = nullptr;
        const void* VPages = nullptr;
        const void* KScales = nullptr;
        const void* VScales = nullptr;
        const void* KScalesFp8 = nullptr;
        const void* VScalesFp8 = nullptr;
        const void* KScalesPagedFp8 = nullptr;
        const void* VScalesPagedFp8 = nullptr;
        const void* BlockTableGpu = nullptr;
        std::uint64_t RecurrentStateHash = 0;
        std::uint64_t RecurrentStateBytesHash = 0;
        std::uint64_t ConvStateHash = 0;
        std::uint64_t ConvStateBytesHash = 0;
        int BlockTableStride = 0;
        int PageBlockSize = 0;
        int TotalPages = 0;
        int MaxSeqLen = 0;
        int MaxSeqlenK = 0;
        int NumKvHeads = 0;
        int HeadDim = 0;
        int VocabSize = 0;
        int PrefillPosOffset = 0;
        std::size_t PerBufferBytes = 0;
        std::size_t PerPoolBytes = 0;
        bool Paged = false;
        bool Fp8 = false;
        bool PrefillMode = false;

        bool operator==(const DecodeGraphSignature& rhs) const {
            return Batch == rhs.Batch
                && InputsData == rhs.InputsData
                && PositionIDsData == rhs.PositionIDsData
                && SeqLensGpu == rhs.SeqLensGpu
                && CuSeqlensQGpu == rhs.CuSeqlensQGpu
                && CuSeqlensKGpu == rhs.CuSeqlensKGpu
                && FinishedGpu == rhs.FinishedGpu
                && LogitsOutGpu == rhs.LogitsOutGpu
                && KData == rhs.KData
                && VData == rhs.VData
                && KPages == rhs.KPages
                && VPages == rhs.VPages
                && KScales == rhs.KScales
                && VScales == rhs.VScales
                && KScalesFp8 == rhs.KScalesFp8
                && VScalesFp8 == rhs.VScalesFp8
                && KScalesPagedFp8 == rhs.KScalesPagedFp8
                && VScalesPagedFp8 == rhs.VScalesPagedFp8
                && BlockTableGpu == rhs.BlockTableGpu
                && RecurrentStateHash == rhs.RecurrentStateHash
                && RecurrentStateBytesHash == rhs.RecurrentStateBytesHash
                && ConvStateHash == rhs.ConvStateHash
                && ConvStateBytesHash == rhs.ConvStateBytesHash
                && BlockTableStride == rhs.BlockTableStride
                && PageBlockSize == rhs.PageBlockSize
                && TotalPages == rhs.TotalPages
                && MaxSeqLen == rhs.MaxSeqLen
                && MaxSeqlenK == rhs.MaxSeqlenK
                && NumKvHeads == rhs.NumKvHeads
                && HeadDim == rhs.HeadDim
                && VocabSize == rhs.VocabSize
                && PrefillPosOffset == rhs.PrefillPosOffset
                && PerBufferBytes == rhs.PerBufferBytes
                && PerPoolBytes == rhs.PerPoolBytes
                && Paged == rhs.Paged
                && Fp8 == rhs.Fp8
                && PrefillMode == rhs.PrefillMode;
        }
    };

    struct DecodeGraphCacheEntry {
        cudaGraphExec_t Exec = nullptr;
        DeviceMemoryStack::Checkpoint Checkpoint{};
        bool Primed = false;
        bool HasSignature = false;
        DecodeGraphSignature Signature{};
    };
    // Full-mode decode graph cache keyed by decode batch size (B).
    std::unordered_map<long, DecodeGraphCacheEntry> mDecodeGraphCache;

    // ========================================================================
    // Compiled execution (operations pre-compiled into direct function calls)
    // ========================================================================
    std::unique_ptr<GraphCompiler> mCompiler;
    std::unique_ptr<CompiledExecutor> mCompiledExecutor;
    std::unique_ptr<CompiledGraph> mCompiledForward;
    std::unique_ptr<CompiledGraph> mCompiledBackward;
    long mCompiledB = 0;
    long mCompiledT = 0;

    // Separate compiled forward graph for decode (execute_decode_step).
    // Decode uses (large B, T=1) which differs from training (small B, large T).
    // Keeping them separate prevents decode from corrupting training state.
    std::unique_ptr<CompiledGraph> mDecodeCompiledForward;
    long mDecodeCompiledB = 0;
    long mDecodeCompiledT = 0;

    // Prefill compiled forward cache keyed by (B,T). GRPO prefill lengths vary
    // across prompts, so caching avoids frequent recompilation.
    std::unordered_map<std::uint64_t, std::unique_ptr<CompiledGraph>> mPrefillCompiledForwardCache;

    void init_compiled_execution();
    void compile_graphs(long B, long T);
    void execute_forward(long B, long T, NCCLCommunicator& comm, bool full,
                         const modules::ForwardHook* hook);
    void execute_backward(long B, long T, NCCLCommunicator& comm, int grad_accum_steps,
                          int micro_step, const modules::BackwardHook* hook,
                          bool skip_zeroing = false);
};

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_GRAPH_EXECUTOR_H
