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
    void execute_backward(const CompiledGraph& graph,
                          NCCLCommunicator& comm,
                          int grad_accum_steps,
                          int micro_step,
                          const modules::BackwardHook* hook);

    // Set optional components
    void set_lora_state(const modules::ModularLoRAConfig* config,
                        modules::ModularLoRAWeightsManager* weights,
                        modules::ModularLoRAGradsManager* grads,
                        modules::LoRARunState* run_state);

    void set_weight_manager(DslWeightManager* weight_manager);
    void set_recipe(const recipes::Recipe* recipe);
    void set_hook_context(void* context);
    void set_recompute_fn(std::function<void(int, long, long, bool)> fn);
    void set_recompute_enabled(bool enabled);
    void set_recompute_use_graphs(bool enabled) { mRecomputeUseGraphs = enabled; }
    void set_capturing(bool capturing) { mCapturing = capturing; }

    // Cache management
    void set_fp8_cache(std::unordered_map<std::string, FP8WeightCacheEntry>* cache);
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

    // Save specified tensors to the saved map (for backward use)
    void save_tensors(const std::vector<std::string>& save_list);
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
    void dispatch_view(const CompiledOp& op);
    void dispatch_add(const CompiledOp& op);
    void dispatch_matmul(const CompiledOp& op, const modules::ForwardHook* hook);
    void dispatch_bias_add(const CompiledOp& op);
    void dispatch_swiglu(const CompiledOp& op);
    void dispatch_silu(const CompiledOp& op);
    void dispatch_mul(const CompiledOp& op);
    void dispatch_matmul_swiglu(const CompiledOp& op);
    void dispatch_qkv_qk_norm_rope(const CompiledOp& op);
    void dispatch_rope(const CompiledOp& op);
    void dispatch_flash_attention(const CompiledOp& op);
    void dispatch_cross_entropy_loss(const CompiledOp& op);
    void dispatch_fused_lm_head_loss(const CompiledOp& op);
    // MoE forward dispatch
    void dispatch_moe_softmax(const CompiledOp& op);
    void dispatch_moe_sigmoid(const CompiledOp& op);
    void dispatch_moe_topk(const CompiledOp& op);
    void dispatch_moe_permute(const CompiledOp& op);
    void dispatch_moe_grouped_gemm_gate_up(const CompiledOp& op);
    void dispatch_moe_grouped_gemm_down(const CompiledOp& op);
    void dispatch_moe_unpermute(const CompiledOp& op);

    // Backward dispatch functions
    void dispatch_view_backward(const CompiledOp& op);
    void dispatch_add_backward(const CompiledOp& op);
    void dispatch_matmul_backward(const CompiledOp& op, const modules::BackwardHook* hook);
    void dispatch_bias_add_backward(const CompiledOp& op);
    void dispatch_swiglu_backward(const CompiledOp& op);
    void dispatch_silu_backward(const CompiledOp& op);
    void dispatch_mul_backward(const CompiledOp& op);
    void dispatch_matmul_swiglu_backward(const CompiledOp& op, const modules::BackwardHook* hook);
    void dispatch_rope_backward(const CompiledOp& op);
    void dispatch_qkv_qk_norm_rope_backward(const CompiledOp& op);
    void dispatch_flash_attention_backward(const CompiledOp& op);
    void dispatch_zeros_backward(const CompiledOp& op);
    void dispatch_fused_residual_rmsnorm_backward(const CompiledOp& op);
    void dispatch_embedding_backward(const CompiledOp& op);
    void dispatch_cross_entropy_loss_backward(const CompiledOp& op);
    void dispatch_fused_lm_head_loss_backward(const CompiledOp& op);
    // MoE backward dispatch
    void dispatch_moe_softmax_backward(const CompiledOp& op);
    void dispatch_moe_sigmoid_backward(const CompiledOp& op);
    void dispatch_moe_topk_backward(const CompiledOp& op);
    void dispatch_moe_permute_backward(const CompiledOp& op);
    void dispatch_moe_grouped_gemm_gate_up_backward(const CompiledOp& op);
    void dispatch_moe_grouped_gemm_down_backward(const CompiledOp& op);
    void dispatch_moe_unpermute_backward(const CompiledOp& op);

    // Tensor resolution (pre-resolved, O(1) lookup)
    Tensor& resolve_tensor(const TensorRef& ref);
    Tensor& ensure_output_tensor(const TensorRef& ref);
    Tensor* try_resolve_saved_live(const std::string& name, const Tensor& saved);

    // Layer boundary handling
    void handle_layer_start(int layer_idx);
    void handle_layer_end(int layer_idx);

    // State
    DslRunState& mRunState;
    DslParamStore& mWeights;
    DslGradStore& mGrads;
    const modules::ModelConfig& mConfig;
    const RuntimeOptions& mOptions;

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
    std::unordered_map<std::string, FP4WeightCacheEntry>* mFP4Cache = nullptr;
    std::unordered_map<std::string, FP4WeightCacheEntry>* mFP4CacheT = nullptr;
    std::unordered_map<std::string, Tensor>* mSaved = nullptr;
    const std::vector<std::string>* mSaveList = nullptr;  // Tensors to preserve for backward
    std::unordered_set<std::string> mSaveSet;             // Fast lookup for save list
    std::vector<LayerForwardPlan>* mForwardPlan = nullptr;

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
    bool mCapturing = false;

    // Temporary tensor storage (for stack-allocated tensors)
    std::vector<Tensor> mTemps;
    std::unordered_map<std::string, Tensor> mTensorMap;

    // Gradient accumulation tracking (set of gradient tensor names that need accumulation)
    std::unordered_set<std::string> mAccumulateTensors;

    // Persistent storage for MoE expert_offsets (needs to survive from forward to backward)
    std::vector<int> mMoEExpertOffsetsData;
    Tensor mMoEExpertOffsets;  // Views into mMoEExpertOffsetsData
    void* mMoEExpertOffsetsGPU = nullptr;  // Persistent GPU buffer (not stack-allocated)
    size_t mMoEExpertOffsetsGPUSize = 0;   // Size in bytes

    // Persistent storage for MoE saved tensors (per-layer copies to prevent buffer reuse corruption)
    // Maps tensor name to persistent GPU buffer (cudaMalloc'd, NOT from stack allocator)
    std::unordered_map<std::string, void*> mMoESavedBuffers;
    std::unordered_map<std::string, size_t> mMoESavedSizes;
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
