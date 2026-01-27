// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Compiled operation dispatch for DSL Graph executor.
//
// This module eliminates runtime dispatch overhead by pre-compiling operations
// into direct function pointer calls with pre-resolved tensors and attributes.

#ifndef SUROGATE_SRC_DSL_COMPILED_OPS_H
#define SUROGATE_SRC_DSL_COMPILED_OPS_H

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

#include "dsl/forward_plan.h"
#include "dsl/graph_executor_internal.h"
#include "dsl/ir.h"
#include "kernels/kernels.h"
#include "utilities/tensor.h"

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
// Operation Type Enumeration (compile-time dispatch)
// ============================================================================

enum class CompiledOpType : std::uint8_t {
    Embedding,
    Zeros,
    FusedResidualRMSNorm,
    View,
    Add,
    Matmul,
    MatmulBias,
    BiasAdd,
    SwiGLU,
    MatmulSwiGLU,
    QKVQKNormRoPE,
    RoPE,
    FlashAttention,
    CrossEntropyLoss,
    FusedLMHeadLoss,
    // Backward operations
    ViewBackward,
    AddBackward,
    MatmulBackward,
    BiasAddBackward,
    SwiGLUBackward,
    MatmulSwiGLUBackward,
    RoPEBackward,
    QKVQKNormRoPEBackward,
    FlashAttentionBackward,
    ZerosBackward,
    FusedResidualRMSNormBackward,
    EmbeddingBackward,
    CrossEntropyLossBackward,
    FusedLMHeadLossBackward,
    // Sentinel
    Unknown
};

// ============================================================================
// Pre-resolved tensor slots
// ============================================================================

// Tensor slot types for pre-resolution
enum class TensorSlot : std::uint8_t {
    // Activation slots (layer-indexed)
    BlockLN1,
    BlockLN1RSTD,
    BlockLN2,
    BlockLN2RSTD,
    BlockQRSTD,
    BlockKRSTD,
    BlockQKV,
    BlockQKVRoPE,
    BlockLSE,
    BlockAtt,
    BlockAttOut,
    BlockResidualAtt,
    BlockMLPUp,
    BlockSwiGLU,
    BlockMLPDown,
    BlockResidualFFN,
    // Gradient slots (layer-indexed)
    BlockDLN1,
    BlockDQKV,
    BlockDAtt,
    BlockDSwiGLU,
    BlockDMLPUp,
    BlockDMLPDown,
    BlockDLN2,
    BlockDResAtt,
    BlockDResFFN,
    // Global activations
    Encoded,
    LNFinal,
    LNFinalRSTD,
    FinalResidual,
    FreqCis,
    // Inputs
    TokenIDs,
    PositionIDs,
    Targets,
    Losses,
    DLoss,
    // Named parameter (uses name lookup)
    Parameter,
    // Temporary (stack-allocated)
    Temporary,
    // Saved tensor (from forward pass)
    Saved,
    // Already in tensor map
    Mapped,
};

// Pre-resolved tensor reference
struct TensorRef {
    TensorSlot slot = TensorSlot::Mapped;
    int layer_idx = -1;          // For block-indexed slots
    std::string name;            // For Parameter/Saved/Mapped slots
    std::vector<long> shape;     // Pre-computed shape (empty = use base tensor shape)
    ETensorDType dtype = ETensorDType::BF16;
};

// ============================================================================
// Pre-resolved attributes
// ============================================================================

struct CompiledAttrs {
    // Common attributes
    float eps = 1e-6f;
    EMMTranspose transpose = EMMTranspose::NN;
    int rotary_dim = 0;
    bool compute_accuracy = false;

    // Shape info
    std::vector<long> shape;
    std::string shape_like;  // Reference tensor name for runtime shape lookup (used by view backward)

    // Matmul-specific
    std::optional<modules::MatmulOp> matmul_op;
    int layer_idx = -1;
    bool allow_quant = false;

    // Hook-specific
    std::optional<modules::ForwardHookPoint> forward_hook_point;
    std::optional<modules::BackwardHookPoint> backward_hook_point;
};

// ============================================================================
// Compiled Operation
// ============================================================================

struct CompiledOp {
    CompiledOpType type = CompiledOpType::Unknown;
    std::uint16_t original_idx = 0;     // Index in original operation list (for debugging)

    // Pre-resolved inputs/outputs
    std::vector<TensorRef> inputs;
    std::vector<TensorRef> outputs;

    // Pre-resolved attributes
    CompiledAttrs attrs;

    // Layer boundary info (for prefetch optimization)
    int layer_start = -1;               // If >= 0, this op starts a new layer
    int layer_end = -1;                 // If >= 0, this op ends a layer

    // Debug info
    std::string op_id;                  // Original operation ID
};

// ============================================================================
// Compiled Graph
// ============================================================================

struct CompiledGraph {
    std::string name;
    std::vector<CompiledOp> ops;

    // Layer boundary indices for O(1) prefetch scheduling
    std::vector<std::size_t> layer_start_indices;  // ops[layer_start_indices[L]] starts layer L
    std::vector<std::size_t> layer_end_indices;    // ops[layer_end_indices[L]-1] ends layer L

    // Pre-computed skip mask for partial execution
    std::vector<char> required_mask;

    // Statistics
    std::size_t total_ops = 0;
    std::size_t matmul_ops = 0;
    std::size_t view_ops = 0;
};

// ============================================================================
// Graph Compiler
// ============================================================================

class GraphCompiler {
public:
    GraphCompiler(const Module& module,
                  const modules::ModelConfig& config,
                  const RuntimeOptions& options,
                  DslParamStore& weights,
                  DslGradStore& grads);

    // Compile a forward or backward graph
    CompiledGraph compile(const Graph& graph, long B, long T);

    // Update batch/sequence dimensions for shape resolution
    void update_dimensions(long B, long T);

private:
    CompiledOpType classify_op(const std::string& op_type) const;

    TensorRef resolve_tensor_ref(const std::string& name, bool is_output,
                                 const Operation& op, const ShapeEnv& env);

    CompiledAttrs resolve_attrs(const Operation& op, CompiledOpType type,
                                const ShapeEnv& env);

    void annotate_layer_boundaries(CompiledGraph& graph);

    // Shape validation methods
    struct TensorShape {
        std::vector<long> dims;
        bool inferred = false;  // true if inferred, false if from IR
        std::string source_op;  // Operation that produced this tensor
    };

    bool resolve_tensor_shape(const std::string& name, std::vector<long>& shape);
    void infer_output_shapes(const Operation& op, CompiledOpType type,
                            const std::vector<std::vector<long>>& input_shapes,
                            std::vector<std::vector<long>>& output_shapes);
    void validate_operation_shapes(const Operation& op, CompiledOpType type, size_t op_index);

    const Module& mModule;
    const modules::ModelConfig& mConfig;
    const RuntimeOptions& mOptions;
    DslParamStore& mWeights;
    DslGradStore& mGrads;
    ShapeEnv mShapeEnv;
    long mB = 0;
    long mT = 0;
    std::unordered_map<std::string, std::vector<long>> mExtraShapes;
    std::unordered_map<std::string, TensorShape> mTensorShapes;
};

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

    // Set batch/sequence dimensions before execution
    void set_dimensions(long B, long T) { mB = B; mT = T; }

    // Save specified tensors to the saved map (for backward use)
    void save_tensors(const std::vector<std::string>& save_list);

private:
    // Direct dispatch functions (no string comparison)
    void dispatch_embedding(const CompiledOp& op);
    void dispatch_zeros(const CompiledOp& op);
    void dispatch_fused_residual_rmsnorm(const CompiledOp& op);
    void dispatch_view(const CompiledOp& op);
    void dispatch_add(const CompiledOp& op);
    void dispatch_matmul(const CompiledOp& op, const modules::ForwardHook* hook);
    void dispatch_bias_add(const CompiledOp& op);
    void dispatch_swiglu(const CompiledOp& op);
    void dispatch_matmul_swiglu(const CompiledOp& op);
    void dispatch_qkv_qk_norm_rope(const CompiledOp& op);
    void dispatch_rope(const CompiledOp& op);
    void dispatch_flash_attention(const CompiledOp& op);
    void dispatch_cross_entropy_loss(const CompiledOp& op);
    void dispatch_fused_lm_head_loss(const CompiledOp& op);

    // Backward dispatch functions
    void dispatch_view_backward(const CompiledOp& op);
    void dispatch_add_backward(const CompiledOp& op);
    void dispatch_matmul_backward(const CompiledOp& op, const modules::BackwardHook* hook);
    void dispatch_bias_add_backward(const CompiledOp& op);
    void dispatch_swiglu_backward(const CompiledOp& op);
    void dispatch_matmul_swiglu_backward(const CompiledOp& op, const modules::BackwardHook* hook);
    void dispatch_rope_backward(const CompiledOp& op);
    void dispatch_qkv_qk_norm_rope_backward(const CompiledOp& op);
    void dispatch_flash_attention_backward(const CompiledOp& op);
    void dispatch_zeros_backward(const CompiledOp& op);
    void dispatch_fused_residual_rmsnorm_backward(const CompiledOp& op);
    void dispatch_embedding_backward(const CompiledOp& op);
    void dispatch_cross_entropy_loss_backward(const CompiledOp& op);
    void dispatch_fused_lm_head_loss_backward(const CompiledOp& op);

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

    // Execution state
    long mB = 0;
    long mT = 0;
    int mCurrentLayer = -1;
    bool mCapturing = false;

    // Temporary tensor storage (for stack-allocated tensors)
    std::vector<Tensor> mTemps;
    std::unordered_map<std::string, Tensor> mTensorMap;

    // Gradient accumulation tracking (set of gradient tensor names that need accumulation)
    std::unordered_set<std::string> mAccumulateTensors;
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
