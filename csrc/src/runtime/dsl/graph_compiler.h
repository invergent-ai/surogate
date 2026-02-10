// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Compiled operation dispatch for DSL Graph executor.
//
// This module eliminates runtime dispatch overhead by pre-compiling operations
// into direct function pointer calls with pre-resolved tensors and attributes.

#ifndef SUROGATE_SRC_DSL_GRAPH_COMPILER_H
#define SUROGATE_SRC_DSL_GRAPH_COMPILER_H

#include <string>
#include <vector>
#include <optional>
#include <limits>

#include "runtime/dsl/tensor_slot.h"
#include "runtime/dsl/tensor_slot_registry.h"
#include "kernels/kernels.h"

namespace modules {
struct ModelConfig;
enum class MatmulOp;
enum class ForwardHookPoint;
enum class BackwardHookPoint;
}

struct RuntimeOptions;

namespace dsl {

// Helper function to strip SSA-style numeric suffixes from tensor names
std::string strip_ssa_suffix(const std::string& field);

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
    Silu,
    Relu2,
    Mul,
    MatmulSwiGLU,
    QKVQKNormRoPE,
    RoPE,
    FlashAttention,
    CrossEntropyLoss,
    FusedLMHeadLoss,
    // MoE forward operations
    MoESoftmax,
    MoESigmoid,
    MoETopK,
    MoEPermute,
    MoEGroupedGemm,
    MoEGroupedGemmGateUp,
    MoEGroupedGemmDown,
    MoEUnpermute,
    // Backward operations
    ViewBackward,
    AddBackward,
    MatmulBackward,
    BiasAddBackward,
    SwiGLUBackward,
    SiluBackward,
    Relu2Backward,
    MulBackward,
    MatmulSwiGLUBackward,
    RoPEBackward,
    QKVQKNormRoPEBackward,
    FlashAttentionBackward,
    ZerosBackward,
    FusedResidualRMSNormBackward,
    EmbeddingBackward,
    CrossEntropyLossBackward,
    FusedLMHeadLossBackward,
    // MoE backward operations
    MoESoftmaxBackward,
    MoESigmoidBackward,
    MoETopKBackward,
    MoEPermuteBackward,
    MoEGroupedGemmBackward,
    MoEGroupedGemmGateUpBackward,
    MoEGroupedGemmDownBackward,
    MoEUnpermuteBackward,
    // Mamba/SSM forward operations
    MambaSplitProj,
    MambaConv1d,
    MambaSplitConvOut,
    MambaSsmScan,
    MambaGatedRMSNorm,
    MambaOutProj,
    // Mamba/SSM backward operations
    MambaSplitProjBackward,
    MambaConv1dBackward,
    MambaSplitConvOutBackward,
    MambaSsmScanBackward,
    MambaGatedRMSNormBackward,
    MambaOutProjBackward,
    // Sentinel
    Unknown
};

// ============================================================================
// Pre-resolved tensor slots
// ============================================================================

// TensorSlot enum is defined in runtime/dsl/tensor_slot.h to break circular dependencies

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

    // MoE-specific
    int top_k = 0;
    bool normalize_weights = true;
    float scaling_factor = 1.0f;

    // Mamba/SSM-specific
    int mamba_num_heads = 0;
    int mamba_head_dim = 0;
    int ssm_state_size = 0;
    int n_groups = 0;
    int conv_kernel = 4;
    int chunk_size = 256;
    int intermediate_size = 0;
    int conv_dim = 0;
    float dt_min = 0.0f;
    float dt_max = std::numeric_limits<float>::infinity();
    bool dt_softplus = true;
    bool use_conv_bias = true;
    std::string activation;  // for mamba_conv1d (e.g., "silu")
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

    // Get the slot registry (for passing to CompiledExecutor)
    const TensorSlotRegistry& slot_registry() const { return mSlotRegistry; }

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
    TensorSlotRegistry mSlotRegistry;  ///< Maps tensor names to slots (from DSL or built-in)
    long mB = 0;
    long mT = 0;
    std::unordered_map<std::string, std::vector<long>> mExtraShapes;
    std::unordered_map<std::string, TensorShape> mTensorShapes;
    std::unordered_map<std::string, ETensorDType> mTensorDtypes;
};


}

#endif  // SUROGATE_SRC_DSL_GRAPH_COMPILER_H
