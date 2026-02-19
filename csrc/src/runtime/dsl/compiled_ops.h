// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Compiled operation dispatch for DSL Graph executor.
//
// This module eliminates runtime dispatch overhead by pre-compiling operations
// into direct function pointer calls with pre-resolved tensors and attributes.

#ifndef SUROGATE_SRC_DSL_COMPILED_OPS_H
#define SUROGATE_SRC_DSL_COMPILED_OPS_H

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
#include "runtime/dsl/graph_executor_internal.h"
#include "runtime/dsl/ir.h"
#include "runtime/dsl/tensor_slot.h"
#include "runtime/dsl/tensor_slot_registry.h"
#include "kernels/kernels.h"
#include "runtime/lora/lora_types.h"
#include "utilities/stack.h"
#include "utilities/tensor.h"
#include "runtime/dsl/graph_compiler.h"

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
}

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
    CompiledExecutor(DslRunState& run_state,
                     DslParamStore& weights,
                     DslGradStore& grads,
                     const modules::ModelConfig& config,
                     const RuntimeOptions& options);
    ~CompiledExecutor();

    // Execute a compiled forward graph
    void execute_forward(const CompiledGraph& graph,
                         NCCLCommunicator& comm,
                         bool full,
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

    /// Set the GPU buffer containing per-token custom d_loss values for GRPO backward.
    /// When non-null, dispatch_fused_lm_head_loss_backward copies these values into
    /// d_loss instead of seeding with 1.0 (standard cross-entropy training).
    /// Buffer must contain B*T float32 values (same layout as the loss tensor).
    /// Lifetime must extend through the execute_backward call.
    void set_custom_dloss_context(float* custom_dloss_gpu) {
        mCustomDLossGpu = custom_dloss_gpu;
    }

    void set_recompute_fn(std::function<void(int, long, long, bool)> fn);
    void set_recompute_enabled(bool enabled);
    void set_recompute_use_graphs(bool enabled) { mRecomputeUseGraphs = enabled; }
    void set_capturing(bool capturing) { mCapturing = capturing; }
    void set_debug_dump_fn(std::function<void(const std::vector<std::string>&, int)> fn) {
        mDebugDumpFn = std::move(fn);
    }
    void set_debug_dump_layer_fn(std::function<void(int)> fn) {
        mDebugDumpLayerFn = std::move(fn);
    }

    // Cache management
    void set_fp8_cache(std::unordered_map<std::string, FP8WeightCacheEntry>* cache);
    void set_fp8_cache_transposed(std::unordered_map<std::string, FP8WeightCacheEntry>* cache_t);
    void set_fp4_cache(std::unordered_map<std::string, FP4WeightCacheEntry>* cache,
                       std::unordered_map<std::string, FP4WeightCacheEntry>* cache_t);
    void set_saved_tensors(std::unordered_map<std::string, Tensor>* saved);
    void set_save_list(const std::vector<std::string>* save_list);
    void set_forward_plan(std::vector<LayerForwardPlan>* plan) { mForwardPlan = plan; }

    // For embedding backward (requires CPU-side inputs for deterministic bucketing)
    void set_last_inputs_cpu(const Tensor* inputs_cpu);

    // RNG seed for embedding backward
    void set_rng_seed_fn(std::function<unsigned int()> fn);

    // Set embedding output names from forward graph (for binding d_embed_N to d_embeddings)
    void set_embedding_outputs(const std::vector<std::string>& names) { mEmbeddingOutputs = names; }

    // Set slot registry for DSL-driven tensor mapping
    void set_slot_registry(const TensorSlotRegistry* registry) { mSlotRegistry = registry; }

    // Set batch/sequence dimensions before execution
    void set_dimensions(long B, long T) { mB = B; mT = T; }

    // Expose mapped tensors for test/debug (returns nullptr if not found).
    const Tensor* try_get_tensor(const std::string& name) const;
    // Debug-only: resolve by SSA-stripped name or fallback to simplified activations.
    const Tensor* try_get_tensor_fuzzy(const std::string& name);

    // Save specified tensors to the saved map (for backward use)
    void save_tensors(const std::vector<std::string>& save_list, bool force_persist = false);
    // Preallocate persistent buffers for saved tensors before CUDA graph capture.
    // This avoids cudaMalloc during capture when recompute requires persistent saves.
    void prepare_saved_buffers_for_capture(const std::vector<std::string>& save_list);

private:
    // Save MoE layer tensors to persistent storage at layer boundaries
    void save_moe_layer_tensors(int layer_idx);

    // Direct dispatch functions (no string comparison)
    void dispatch_embedding(const CompiledOp& op);
    void dispatch_zeros(const CompiledOp& op);
    void dispatch_fused_residual_rmsnorm(const CompiledOp& op);
    void dispatch_layernorm(const CompiledOp& op);
    void dispatch_view(const CompiledOp& op);
    void dispatch_add(const CompiledOp& op);
    void dispatch_matmul(const CompiledOp& op, const modules::ForwardHook* hook);
    void dispatch_bias_add(const CompiledOp& op);
    void dispatch_swiglu(const CompiledOp& op);
    void dispatch_gpt_oss_moe_act(const CompiledOp& op);
    void dispatch_silu(const CompiledOp& op);
    void dispatch_gelu(const CompiledOp& op);
    void dispatch_relu2(const CompiledOp& op);
    void dispatch_mul(const CompiledOp& op);
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

    // Mamba/SSM backward dispatch
    void dispatch_mamba_split_proj_backward(const CompiledOp& op);
    void dispatch_mamba_conv1d_backward(const CompiledOp& op);
    void dispatch_mamba_split_conv_out_backward(const CompiledOp& op);
    void dispatch_mamba_ssm_scan_backward(const CompiledOp& op);
    void dispatch_mamba_gated_rmsnorm_backward(const CompiledOp& op);
    void dispatch_mamba_out_proj_backward(const CompiledOp& op, const modules::BackwardHook* hook);

    // Tensor resolution (pre-resolved, O(1) lookup)
    Tensor& resolve_tensor(const TensorRef& ref);
    Tensor& ensure_output_tensor(const TensorRef& ref);
    Tensor* try_resolve_saved_live(const std::string& name, const Tensor& saved);
    Tensor resolve_moe_expert_offsets(const CompiledOp& op);

    // Get host-side MoE expert offsets for a layer, using cache or syncing from device.
    const int* get_or_sync_moe_host_offsets(int layer_idx,
                                             const int* device_offsets,
                                             int num_experts);

    // Layer boundary handling
    void handle_layer_start(int layer_idx);
    void handle_layer_end(int layer_idx);

    // State
    DslRunState& mRunState;
    DslParamStore& mWeights;
    DslGradStore& mGrads;
    const modules::ModelConfig& mConfig;
    const RuntimeOptions& mOptions;

    // Log-prob extraction context (null in training mode)
    float*   mLogprobsGpu          = nullptr;

    // Custom per-token d_loss for GRPO backward (null = standard d_loss=1 seeding)
    float*   mCustomDLossGpu       = nullptr;

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
    std::unordered_set<std::string> mSaveSet;             // Fast lookup for save list
    std::vector<LayerForwardPlan>* mForwardPlan = nullptr;
    std::function<void(const std::vector<std::string>&, int)> mDebugDumpFn;
    std::function<void(int)> mDebugDumpLayerFn;

    // For embedding backward
    const Tensor* mLastInputsCpu = nullptr;
    std::function<unsigned int()> mRngSeedFn;
    std::vector<std::string> mEmbeddingOutputs;  // Forward graph embedding output names
    const TensorSlotRegistry* mSlotRegistry = nullptr;  // DSL slot registry for global gradient binding

    // Execution state
    long mB = 0;
    long mT = 0;
    int mMicroStep = 0;
    int mCurrentLayer = -1;
    int mPrefetchDirection = 1;  // +1 for forward, -1 for backward
    bool mCapturing = false;

    // Temporary tensor storage (for stack-allocated tensors)
    std::vector<Tensor> mTemps;

    // Integer-indexed tensor storage (flat vector indexed by compile-time tensor IDs).
    // Indexed by TensorRef::tensor_id assigned during graph compilation.
    // Eliminates string hashing/comparison in the hot resolve_tensor path.
    std::vector<Tensor> mTensors;
    std::vector<bool> mSaveMask;               // Per-tensor-id: true if in save list (for prune)
    const CompiledGraph* mCurrentGraph = nullptr;

    // Bind a named tensor into the flat vector using the current graph's name-to-id map.
    // No-op if the name is not in the graph (e.g., optional visual embeds when disabled).
    void bind_tensor(const std::string& name, const Tensor& t) {
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
        if (ref.tensor_id >= 0 && static_cast<std::size_t>(ref.tensor_id) < mTensors.size()) {
            mTensors[ref.tensor_id] = t;
        }
    }

    // Gradient accumulation tracking (set of gradient tensor names that need accumulation)
    std::unordered_set<std::string> mAccumulateTensors;

    // Cross-layer backward tensors persisted from stack to cudaMalloc.
    // Freed at end of backward pass.
    std::vector<std::byte*> mPersistedBackwardTensors;

    // Persistent storage for MoE expert_offsets (needs to survive from forward to backward)
    std::vector<int> mMoEExpertOffsetsData;
    Tensor mMoEExpertOffsets;  // Views into mMoEExpertOffsetsData
    void* mMoEExpertOffsetsGPU = nullptr;  // Persistent GPU buffer (not stack-allocated)
    size_t mMoEExpertOffsetsGPUSize = 0;   // Size in bytes

    // Per-layer host-side MoE expert offsets cache.
    // Populated once per layer (forward: in dispatch_moe_permute; backward: on first access).
    // Avoids redundant D2H synchronization in grouped GEMM ops within the same layer.
    std::unordered_map<int, std::vector<int>> mMoEHostOffsetsCache;

    // Persistent storage for MoE saved tensors (per-layer copies to prevent buffer reuse corruption)
    // Maps tensor name to persistent GPU buffer (cudaMalloc'd, NOT from stack allocator)
    std::unordered_map<std::string, void*> mMoeSavedBuffers;
    std::unordered_map<std::string, size_t> mMoeSavedSizes;

    // Expert Parallelism (EP) per-layer state: send/recv splits, token reorder mapping
    struct EpLayerState {
        std::vector<int> send_splits;      // tokens sent to each EP peer
        std::vector<int> recv_splits;      // tokens received from each EP peer
        int total_send = 0;                // sum of send_splits
        int total_recv = 0;                // sum of recv_splits
        void* send_order_gpu = nullptr;    // GPU buffer: reorder indices for recv re-sort gather
        void* recv_reorder_gpu = nullptr;  // GPU buffer: reorder indices for recv → local expert order
        size_t send_order_bytes = 0;
        size_t recv_reorder_bytes = 0;
        // LLEP send reorder: when LLEP is active, tokens are reordered before A2A.
        // llep_send_reorder_gpu[new_pos] = old_pos in expert-sorted input.
        // Null when standard EP (no reorder needed).
        void* llep_send_reorder_gpu = nullptr;
        size_t llep_send_reorder_bytes = 0;
        void* local_scatter_gpu = nullptr; // local_scatter [total_recv] indices output
        size_t local_scatter_bytes = 0;
    };
    std::unordered_map<int, EpLayerState> mEpStates;  // keyed by layer_idx

    // LLEP (Least-Loaded EP) per-layer state for dynamic load balancing.
    // When active, merged weight tensors contain native + foreign expert weights,
    // and the GEMM ops use these instead of the QLoRA-resolved weights.
    struct LLEPLayerState {
        bool active = false;               // Whether LLEP rebalancing is active this step
        int num_merged_experts = 0;        // Total experts on this GPU (native + foreign)

        // Per-expert weight pointers (indexed by merged expert index 0..num_merged-1).
        // Each pointer points to one expert's weight slice in either:
        //  - native dequant buffer (for native experts), or
        //  - foreign weight receive buffer (for received foreign experts).
        // No contiguous merged buffer needed — saves ~465 MB GPU memory.
        std::vector<const void*> gate_up_weight_ptrs;  // [num_merged]: ptr to gate_up [2*D, C]
        std::vector<const void*> down_weight_ptrs;     // [num_merged]: ptr to down [C, D]
        ETensorDType weight_dtype = ETensorDType::BF16;

        std::vector<int> merged_offsets_host;  // Host expert offsets for merged set
        void* merged_offsets_gpu = nullptr;    // GPU expert offsets
        size_t merged_offsets_gpu_bytes = 0;
        // Map from merged expert index → global expert ID
        std::vector<int> merged_to_global;
        // Map from global expert ID → merged expert index (-1 if not on this GPU)
        std::vector<int> global_to_merged;

        // Merged LoRA weights [num_merged, ...] for use in GEMM dispatch.
        // Built from native expert LoRA + transferred foreign expert LoRA.
        // Null/inactive when LoRA is not enabled.
        // Per-layer owned GPU memory (NOT shared across layers).
        modules::LoRAGroupedExpertWeights<Tensor> merged_lora;
        bool has_merged_lora = false;
        // Per-layer owned GPU pointers for merged LoRA (freed on layer state clear)
        std::vector<void*> owned_lora_ptrs;

        // Foreign weight P2P receive buffers — owned by this state.
        // Must stay alive as long as weight pointers reference them.
        std::vector<void*> owned_foreign_ptrs;

        void free_lora_gpu() {
            for (void* p : owned_lora_ptrs) {
                if (p) cudaFree(p);
            }
            owned_lora_ptrs.clear();
            has_merged_lora = false;
        }
        void free_foreign_gpu() {
            for (void* p : owned_foreign_ptrs) {
                if (p) cudaFree(p);
            }
            owned_foreign_ptrs.clear();
            gate_up_weight_ptrs.clear();
            down_weight_ptrs.clear();
        }
    };
    std::unordered_map<int, LLEPLayerState> mLLEPStates;  // keyed by layer_idx

    // Lightweight per-layer EP metadata — survives LLEP state clearing.
    // Backward uses this to reconstruct native-only weight pointers when
    // the full LLEP state has been freed to save GPU memory.
    struct EPLayerMeta {
        int num_merged = 0;                // total experts on this GPU (native + foreign)
        int native_start = 0;              // first native expert's global ID
        int num_local = 0;                 // number of native experts
        std::vector<int> merged_to_global; // merged_idx → global expert ID
    };
    std::unordered_map<int, EPLayerMeta> mEPLayerMeta;  // keyed by layer_idx

    // Shared GPU buffers for EP combine / dispatch_backward output (off-stack).
    // Only one layer uses these at a time, so sharing saves ~1.2 GB vs per-layer.
    void* mSharedEpCombinedGpu = nullptr;      // ep_combine reverse A2A output [total_send, hidden]
    size_t mSharedEpCombinedBytes = 0;
    void* mSharedEpSortedRecvGpu = nullptr;    // ep_dispatch_backward output [total_send, hidden]
    size_t mSharedEpSortedRecvBytes = 0;
    void* mSharedEpLlepCombineGpu = nullptr;   // ep_combine LLEP reorder output [total_send, hidden]
    size_t mSharedEpLlepCombineBytes = 0;

    // CUDA stream for LLEP weight transfer (overlaps with token A2A on MainStream).
    // Lazily created on first LLEP activation. Uses separate NCCL weight_transfer_comm.
    cudaStream_t mWeightTransferStream = nullptr;

    // GPU buffer pool for EP intermediates — eliminates cudaStreamSynchronize + cudaFree
    // barriers in the hot path. All EP ops run on MainStream, so CUDA stream ordering
    // guarantees safe buffer reuse without explicit synchronization.
    // Pattern: acquire() finds a recycled buffer >= requested size (or cudaMalloc's a new one);
    //          release() returns the buffer to the pool for reuse by subsequent layers.
    struct EpPoolEntry { void* ptr; size_t bytes; };
    std::vector<EpPoolEntry> mEpBufPool;
    void* ep_buf_acquire(size_t need);
    void ep_buf_release(void* ptr, size_t bytes);

    // Retired shared EP buffers — old allocations kept alive until end-of-step because
    // save_moe_layer_tensors copies tensor data at layer boundaries, and the copy source
    // may reference the old buffer via stored mTensors entries. Cannot go into mEpBufPool
    // because the pool recycles buffers for temporary EP intermediates, which would overwrite
    // data still referenced by saved tensors. Freed at destructor time.
    std::vector<EpPoolEntry> mEpRetiredBufs;

    // Reusable per-forward layer tracking vectors (avoid heap allocation every forward call).
    std::vector<DeviceMemoryStack::Checkpoint> mLayerCheckpoints;
    std::vector<std::size_t> mLayerTempMarks;
    std::vector<char> mLayerActive;

public:
    /// Total bytes of persistent saved buffers (untracked by TensorAllocator).
    size_t saved_buffers_total_bytes() const {
        size_t total = 0;
        for (const auto& [name, sz] : mMoeSavedSizes) total += sz;
        return total;
    }
    /// Number of persistent saved buffers.
    int saved_buffers_count() const { return static_cast<int>(mMoeSavedSizes.size()); }
    /// Per-buffer sizes for diagnostics.
    const std::unordered_map<std::string, size_t>& saved_buffers_sizes() const { return mMoeSavedSizes; }
};

// ============================================================================
// Utility functions
// ============================================================================

// Convert string operation type to enum (used during compilation)
CompiledOpType op_type_from_string(const std::string& op_type);

// Convert enum to string (for debugging)
const char* op_type_to_string(CompiledOpType type);

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_COMPILED_OPS_H
