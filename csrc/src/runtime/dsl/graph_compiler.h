// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Compiled operation dispatch for DSL Graph executor.
//
// This module eliminates runtime dispatch overhead by pre-compiling operations
// into direct function pointer calls with pre-resolved tensors and attributes.

#ifndef SUROGATE_SRC_DSL_GRAPH_COMPILER_H
#define SUROGATE_SRC_DSL_GRAPH_COMPILER_H

#include <array>
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>
#include <limits>

#include "runtime/dsl/tensor_slot.h"
#include "runtime/dsl/tensor_slot_registry.h"
#include "runtime/dsl/fusions/fusion_pass.h"
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
    Ones,
    FusedResidualRMSNorm,
    LayerNorm,
    View,
    Transpose,
    Split,
    Concat,
    Add,
    Matmul,
    MatmulBias,
    BiasAdd,
    SwiGLU,
    GptOssMoeAct,
    Silu,
    Gelu,
    Relu2,
    Mul,
    MaskScatter,
    DeepstackInject,
    MatmulSwiGLU,
    QKVQKNorm,
    QKVQKNormRoPE,
    MRoPE,
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
    MoEExpertBiasAdd,
    // Expert Parallelism forward operations
    EpDispatch,
    EpCombine,
    // Backward operations
    ViewBackward,
    AddBackward,
    MatmulBackward,
    BiasAddBackward,
    SwiGLUBackward,
    GptOssMoeActBackward,
    SiluBackward,
    GeluBackward,
    Relu2Backward,
    MulBackward,
    MaskScatterBackward,
    DeepstackInjectBackward,
    MatmulSwiGLUBackward,
    QKVQKNormBackward,
    RoPEBackward,
    QKVQKNormRoPEBackward,
    MRoPEBackward,
    FlashAttentionBackward,
    ZerosBackward,
    FusedResidualRMSNormBackward,
    LayerNormBackward,
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
    MoEExpertBiasAddBackward,
    // Expert Parallelism backward operations
    EpDispatchBackward,
    EpCombineBackward,
    // Mamba/SSM forward operations
    MambaSplitProj,
    MambaConv1d,
    MambaSplitConvOut,
    MambaSsmScan,
    MambaGatedRMSNorm,
    MambaOutProj,
    // Qwen3.5 GDN fused projection (inference-only, decode)
    GdnFusedProj,
    // Qwen3.5 gated delta rule forward operations
    ChunkGatedDeltaRule,
    Qwen3_5Decay,
    RepeatInterleaveHeads,
    // Qwen3.5 gated delta rule backward operations
    ChunkGatedDeltaRuleBackward,
    Qwen3_5DecayBackward,
    RepeatInterleaveHeadsBackward,
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
    int tensor_id = -1;          // Index into CompiledExecutor::mTensors flat vector (compile-time assigned)
    std::string name;            // For Parameter/Saved/Mapped slots
    std::vector<long> shape;     // Pre-computed shape (empty = use base tensor shape)
    ETensorDType dtype = ETensorDType::BF16;
    bool is_gradient = false;    // True for gradient tensors (d_ prefix) — avoids runtime string checks
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
    std::array<int, 3> mrope_section{0, 0, 0};
    int window_size = 0;

    // Shape info
    std::vector<long> shape;
    std::string shape_like;  // Reference tensor name for runtime shape lookup (used by view backward)
    int shape_like_tensor_id = -1;  // Pre-resolved tensor_id for shape_like (avoids runtime map lookup)

    // MoE side-channel tensor IDs (pre-resolved to avoid runtime string lookups)
    int moe_offsets_tensor_id = -1;    // Pre-resolved "moe_expert_offsets"
    int moe_gather_tensor_id = -1;     // Pre-resolved "moe_gather_indices"

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
    bool topk_softmax = false;
    float topk_rounding_scale = 0.0f;
    bool topk_sort_by_index = false;
    bool gate_up_interleaved = false;

    // Expert Parallelism
    int ep_size = 1;
    int num_experts = 0;

    // GPT-OSS gated MoE activation
    float gpt_oss_alpha = 1.702f;
    float gpt_oss_limit = 7.0f;

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
    bool norm_before_gate = false;
    int repeat_factor = 1;

    // Gated delta rule specific
    float delta_rule_scale = 0.0f;  // 0.0 means "derive from K at runtime"
    bool use_qk_l2norm_in_kernel = false;

    // Tensor split/concat attributes
    int split_concat_dim = 0;
    std::vector<long> split_sizes;

    // Tensor transpose attributes
    int dim0 = 0;
    int dim1 = 1;
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
// Tensor metadata for integer-indexed pruning (replaces runtime string parsing)
// ============================================================================

struct TensorMeta {
    static constexpr uint8_t kCrossLayer   = 1 << 0;  // name starts with "layer"
    static constexpr uint8_t kMoeOffsets   = 1 << 1;  // name == "moe_expert_offsets"
    static constexpr uint8_t kDBlocks      = 1 << 2;  // name starts with "d_blocks["
    static constexpr uint8_t kBlocks       = 1 << 3;  // name starts with "blocks["
    static constexpr uint8_t kMoeGather    = 1 << 4;  // name == "moe_gather_indices"

    uint8_t flags = 0;
    int block_layer_idx = -1;  // For "blocks[N].*" or "d_blocks[N].*", the parsed N

    bool is_cross_layer() const { return flags & kCrossLayer; }
    bool is_moe_offsets() const { return flags & kMoeOffsets; }
    bool is_d_blocks() const { return flags & kDBlocks; }
    bool is_blocks() const { return flags & kBlocks; }
    bool is_moe_gather() const { return flags & kMoeGather; }
};

// ============================================================================
// Graph Segment (for split-attention CUDA graph capture)
// ============================================================================

/// A contiguous range of ops within a layer that can be captured as a single
/// CUDA graph, or must run eagerly (e.g., FlashAttention with doc masking).
struct GraphSegment {
    std::size_t start_op;    ///< Inclusive start index in CompiledGraph::ops
    std::size_t end_op;      ///< Exclusive end index in CompiledGraph::ops
    bool eager;              ///< true = run eagerly (attention ops with dynamic cu_seqlens)
};

// ============================================================================
// MLP Tile Group (for long-context tiled execution)
// ============================================================================

struct MlpTileGroup {
    std::size_t start_op_idx;  // first op in MLP sequence (view before up-proj matmul)
    std::size_t end_op_idx;    // last op in MLP sequence (view after down-proj matmul)
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

    // Pre-computed last-use information for tensor lifetime management in backward pass.
    // last_use_names[i] contains the names of tensors whose last use is at op index i.
    // last_use_index maps tensor name -> last op index that references it.
    // Both computed once during graph compilation instead of rebuilt every backward call.
    std::vector<std::vector<std::string>> last_use_names;
    std::unordered_map<std::string, std::size_t> last_use_index;

    // Integer-indexed tensor ID system for O(1) runtime tensor lookups.
    // All tensor names referenced by ops or init bindings are assigned a unique integer ID
    // during compilation. At runtime, tensors are stored in a flat vector indexed by these IDs.
    int num_tensors = 0;
    std::unordered_map<std::string, int> tensor_name_to_id;  // name -> tensor_id (for init bindings + debug)
    std::vector<TensorMeta> tensor_meta;                     // per-ID pruning metadata
    std::unordered_map<std::string, int> ssa_base_to_id;     // SSA-stripped name -> highest-suffix tensor_id

    // Look up or return -1
    int find_tensor_id(const std::string& name) const {
        // Guard against querying moved-from / not-yet-populated maps.
        if (tensor_name_to_id.empty() || tensor_name_to_id.bucket_count() == 0) {
            return -1;
        }
        auto it = tensor_name_to_id.find(name);
        return (it != tensor_name_to_id.end()) ? it->second : -1;
    }

    // MLP tile groups for long-context tiled execution.
    // When non-empty, the executor processes these op ranges in T-chunks.
    // Forward groups: view → matmul_up → view → swiglu → view → matmul_down → view
    // Backward groups: view_bwd → matmul_bwd(down) → ... → matmul_bwd(up) → view_bwd
    std::vector<MlpTileGroup> mlp_tile_groups;

    // Per-layer segments for split-attention CUDA graph mode.
    // When populated, each layer is split into alternating graph-captured and
    // eager segments around FlashAttention/FlashAttentionBackward ops.
    // layer_segments[L] = ordered segments for layer L.
    std::vector<std::vector<GraphSegment>> layer_segments;

    /// Populate layer_segments by scanning each layer for graph-breaking ops.
    /// Call after annotate_layer_boundaries().
    /// @param mode  The compile mode — InferenceDecode allows GDN ops in graphs.
    void compute_layer_segments(std::uint8_t mode = 0);

    // Global segments spanning the entire model (forward inference only).
    // Adjacent non-eager segments across layer boundaries are merged,
    // reducing graph launches from ~72 to ~37 for a 36-layer model.
    std::vector<GraphSegment> global_segments;

    /// Populate global_segments by merging adjacent graphable segments
    /// from layer_segments. Call after compute_layer_segments().
    void compute_global_segments();

    // Statistics
    std::size_t total_ops = 0;
    std::size_t matmul_ops = 0;
    std::size_t view_ops = 0;
};

enum class GraphCompileMode : std::uint8_t {
    Default = 0,
    InferencePrefill,
    InferenceDecode,
    InferenceFlatTokens,
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
    CompiledGraph compile(const Graph& graph,
                          long B,
                          long T,
                          GraphCompileMode mode = GraphCompileMode::Default);

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
    Graph apply_inference_fusion_passes(const Graph& graph, const struct FusionContext& ctx) const;

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
    bool mDebugShapes = false;  // Set via SUROGATE_DEBUG_SHAPES env var

    // Tensor ID assignment state (per-compile, reset at start of compile())
    std::unordered_map<std::string, int> mTensorIdMap;  // name -> tensor_id
    int mNextTensorId = 0;

    // Assign or retrieve a tensor ID for the given name
    int assign_tensor_id(const std::string& name);

    // Register well-known external tensor names (init bindings, MoE side-channel, etc.)
    void register_external_names(CompiledGraph& graph);

    // Build per-ID metadata for pruning (flags, block_layer_idx, ssa_base_to_id)
    void build_tensor_metadata(CompiledGraph& graph);
};


}

#endif  // SUROGATE_SRC_DSL_GRAPH_COMPILER_H
