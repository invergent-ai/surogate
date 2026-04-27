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
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "runtime/dsl/dsl_runtime_config.h"
#include <optional>
#include <limits>

#include "runtime/dsl/tensor_slot.h"
#include "runtime/dsl/tensor_slot_registry.h"
#include "runtime/dsl/tensor_role.h"
#include "runtime/dsl/fusion_rule_registry.h"
#include "runtime/executor/op_descriptor_types.h"
#include "runtime/lora/lora_types.h"
#include "kernels/kernels.h"

namespace modules {
struct ModelConfig;
enum class MatmulOp;
enum class ForwardHookPoint;
enum class BackwardHookPoint;
}  // namespace modules

struct RuntimeOptions;

namespace dsl {

// Helper function to strip SSA-style numeric suffixes from tensor names
std::string strip_ssa_suffix(const std::string& field);

/// @brief Compiled LoRA slice attached to a matmul op.
///
/// One entry per LoRA target declared on the op's weight input. The graph
/// compiler resolves ``name`` to ``id`` once; hot-path dispatch indexes
/// the block storage by ``id`` and only touches ``name`` when a target is
/// unknown (for dropout-seed hashing + error messages).
struct LoRASlice {
    modules::LoRATargetId id = modules::LoRATargetId::Unknown;
    std::string name;         ///< Raw target name; only read when id == Unknown or on error.
    std::string schema_slot;  ///< Structural BlockSchema param slot inferred from the weight name; diagnostics only.
    int offset = 0;           ///< Element offset on the output dim (0 for unfused).
    int size = 0;             ///< Output slice size in elements (0 = full output dim).
    bool grouped = false;     ///< True for MoE batched-expert LoRA (uses grouped GEMM path).
};

class DslRunState;
class DslParamStore;
class DslGradStore;
class DslWeightManager;
class CompiledExecutor;
struct CompiledOp;

// Dispatch function pointer baked into each CompiledOp at graph-compile
// time. Replaces the switch-on-op.type pattern. Declared here so
// CompiledOp can embed it; implementations live in per-op files that
// register via REGISTER_OP (see runtime/executor/op_registry.h).
using OpExecFn = void (*)(CompiledExecutor&, const CompiledOp&, const void* hook);

// ============================================================================
// Operation Type Enumeration (compile-time dispatch)
// ============================================================================

enum class CompiledOpType : std::uint8_t {
    Embedding,
    Zeros,
    Ones,
    FusedResidualRMSNorm,
    RMSNorm,
    LayerNorm,
    View,
    Transpose,
    Split,
    Narrow,
    Concat,
    Add,
    Matmul,
    MatmulBias,
    BiasAdd,
    SwiGLU,
    GeluGlu,
    GptOssMoeAct,
    Silu,
    Gelu,
    Relu2,
    Mul,
    Scale,
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
    GeluGluBackward,
    GptOssMoeActBackward,
    SiluBackward,
    GeluBackward,
    Relu2Backward,
    MulBackward,
    ScaleBackward,
    NarrowBackward,
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
    RMSNormBackward,
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
    int layer_idx = -1;       // For block-indexed slots
    int tensor_id = -1;       // Index into CompiledExecutor::mTensors flat vector (compile-time assigned)
    std::string name;         // For Parameter/Saved/Mapped slots
    std::vector<long> shape;  // Pre-computed shape (empty = use base tensor shape)
    ETensorDType dtype = ETensorDType::BF16;
    bool is_gradient = false;  // True for gradient tensors (d_ prefix) — avoids runtime string checks
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
    std::string shape_like;         // Reference tensor name for runtime shape lookup (used by view backward)
    int shape_like_tensor_id = -1;  // Pre-resolved tensor_id for shape_like (avoids runtime map lookup)

    // MoE side-channel tensor IDs (pre-resolved to avoid runtime string lookups)
    int moe_offsets_tensor_id = -1;  // Pre-resolved "moe_expert_offsets"
    int moe_gather_tensor_id = -1;   // Pre-resolved "moe_gather_indices"

    // Matmul-specific
    std::optional<modules::MatmulOp> matmul_op;
    int layer_idx = -1;
    bool allow_quant = false;

    // Activation-slot alias point. The matmul dispatch uses this to rebind
    // block-scope slot entries (``qkv`` / ``att_out`` / ``mlp_up`` /
    // ``mlp_down``) to the freshly-produced matmul output so backward
    // replay reads the live buffer. Only a handful of ``ForwardHookPoint``
    // values are consumed for this purpose; they are not invoked as
    // callbacks anymore (LoRA dispatch is slice-driven).
    std::optional<modules::ForwardHookPoint> forward_hook_point;
    std::string forward_hook_schema_slot;  ///< Phase 5 structural equivalent of forward_hook_point; diagnostics only.

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

    // Tensor narrow attributes
    int narrow_start = 0;
    int narrow_length = 0;

    // Tensor transpose attributes
    int dim0 = 0;
    int dim1 = 1;

    // Constant scale factor (scale op)
    float scale_factor = 1.0f;

    // Logit softcapping (fused_lm_head_loss)
    float softcap = 0.0f;  // 0 = disabled

    // Attention softmax scale override (flash_attention op).
    // 0.0f (default) means: use 1/sqrt(head_dim) as usual.
    // Non-zero means: use this exact value as the softmax scale (e.g.,
    // Gemma4 passes 1.0 because QK-norm provides the implicit scaling).
    float softmax_scale = 0.0f;

    // LoRA slices declared by the DSL for this op's weight input. Populated
    // from the weight TensorInfo::lora_targets during graph compilation.
    // Empty = no LoRA runs for this op. Each slice names a semantic
    // target (q/k/v/o/gate/up/down/router/expert_*/shared_*) and its offset
    // and size along the weight's output dim. The runtime looks up the
    // matching LoRA weight by ``name`` in the LoRAWeightsManager.
    std::vector<LoRASlice> lora_slices;
};

// ============================================================================
// Compiled Operation
// ============================================================================

struct CompiledOp {
    CompiledOpType type = CompiledOpType::Unknown;
    std::uint16_t original_idx = 0;  // Index in original operation list (for debugging)

    // Dispatch function, populated by the graph compiler from OpRegistry.
    // Execute paths call `op.fn(exec, op, hook)` directly — no switch, no
    // registry lookup in the hot path. Null means "no handler for this
    // op in the current graph direction" → dispatch throws.
    OpExecFn fn = nullptr;

    // Pre-resolved inputs/outputs
    std::vector<TensorRef> inputs;
    std::vector<TensorRef> outputs;

    // Pre-resolved attributes
    CompiledAttrs attrs;

    // Descriptor facets copied from OpRegistry at compile time. These are
    // metadata only for now; execution continues through `fn`.
    OpSemanticKind semantic_kind = OpSemanticKind::Unknown;
    DistributionKind distribution_kind = DistributionKind::Replicated;
    OpCapabilities default_caps{};
    EpilogueSupport epilogue_support{};
    StorageCompatibility storage_compat{};
    MoECapabilities moe_caps{};
    MatmulCapabilities matmul_caps{};
    CommunicationProfile comm_profile{};
    GroupedSemantics grouped_semantics{};
    std::uint32_t descriptor_flags = 0;

    // Layer boundary info (for prefetch optimization)
    int layer_start = -1;  // If >= 0, this op starts a new layer
    int layer_end = -1;    // If >= 0, this op ends a layer

    // Debug info
    std::string op_id;  // Original operation ID
};

// ============================================================================
// Tensor metadata for integer-indexed pruning (replaces runtime string parsing)
// ============================================================================

/// First-class tensor classification, resolved once at compile time by
/// authoritative lookup (against the parameter store / ops / slot registry).
/// Runtime code queries this instead of string-matching names — which is
/// ambiguous and has caused silent clobber bugs when a name like
/// `d_blocks[N].mlp_x_flat_from_7` was mistaken for a parameter-gradient
/// accumulator and collapsed onto the base tensor's buffer.
enum class TensorKind : uint8_t {
    Unknown = 0,
    ForwardParam,       ///< Present in DslParamStore
    ForwardActivation,  ///< Produced by a forward op (base_producer_op_idx set)
    ParamGrad,          ///< d_<param>; base_param_tid set
    ActivationGrad,     ///< d_<activation>; base_producer_tid set
    AccumTemp,          ///< _from_N / _accum_N variant; base_grad_tid set
    LossInput,          ///< d_loss (and its views)
    Scratch,            ///< Everything else (zeros, constants, unknown intermediates)
};

/// Typed memory region (design/buffer-runtime-v4.md, M2). Each tensor is
/// assigned to exactly one region by derive_regions(). Shadow-only in M2; the
/// layout pass (M3) consumes region + block_layer_idx to bake buffer offsets.
enum class RegionKind : std::uint8_t {
    Unknown = 0,           ///< Not yet classified (includes cross-graph forward activations in backward)
    FwdStack,              ///< Block-scoped forward activation (bump arena; nested FwdBlock frames)
    BwdStack,              ///< Block-scoped backward gradient / temporary (bump arena)
    SaveForBwd,            ///< Forward activation persisted across block boundary for backward
    Accumulator,           ///< Gradient accumulator (parameter grad + autodiff accum temps)
    Persistent,            ///< Training-wide parameters (weights). Slab layout is owner-controlled
                           ///< (DslParamStore / DslWeightManager / LoRA) with runtime-chosen offsets,
                           ///< NOT the compile-time `meta.offset` (which would collide with non-weight
                           ///< tids when WM/QLoRA clamps DslParamStore's slab to zero).
    PersistentActivation,  ///< Training-wide non-block activations / I/O buffers (encoded, xF,
                           ///< ln_final_rstd, output, freq_cis, d_ln_final, loss inputs). Separate
                           ///< from Persistent because compile-time offsets are authoritative here:
                           ///< arena base is a fixed region and rebind writes at `base + meta.offset`.
    ModelScopePersistent,  ///< Forward intermediate produced outside any block (e.g. Gemma4
                           ///< PLI phase outputs: scale_*, pli_proj_rn_flat, pli_narrow_layer*,
                           ///< per_layer_inputs) AND read by at least one backward op. Keeps a
                           ///< stable per-step address so step-N+1's graph-replay refresh
                           ///< save_tensors can still resolve it — the plain Stack-temp
                           ///< allocation from ensure_output_tensor would be overwritten by
                           ///< later per-layer temp_allocs. Gated by `cross_layer_global` flag
                           ///< in TensorMeta. Arena lives in PhaseArenas.model_scope_persistent_*.
    Recomputed,            ///< Forward activation replayed during backward (reuses FwdStack arena)
    GatheredWeight,        ///< ZeRO-3 all-gathered weight shards (unused in Llama prototype)
    BwdCrossLayer,  ///< Backward-produced tensor consumed by a later backward block (MoE aux-loss; unused in Llama)
};

const char* region_kind_name(RegionKind k);

struct TensorMeta {
    static constexpr uint8_t kCrossLayer = 1 << 0;  // name starts with "layer"
    static constexpr uint8_t kMoeOffsets = 1 << 1;  // name == "moe_expert_offsets"
    static constexpr uint8_t kDBlocks = 1 << 2;     // name starts with "d_blocks["
    static constexpr uint8_t kBlocks = 1 << 3;      // name starts with "blocks["
    static constexpr uint8_t kMoeGather = 1 << 4;   // name == "moe_gather_indices"

    uint8_t flags = 0;
    int block_layer_idx = -1;  // For "blocks[N].*" or "d_blocks[N].*", the parsed N

    // Classification (populated by classify_tensors() after build_tensor_metadata).
    TensorKind kind = TensorKind::Unknown;
    TensorRole role{};
    int base_param_tid = -1;     ///< ParamGrad -> tid of the parameter
    int base_producer_tid = -1;  ///< ActivationGrad -> tid of the forward activation
    int base_grad_tid = -1;      ///< AccumTemp -> tid of the non-accum parent gradient

    // Region assignment (populated by derive_regions() after classify_tensors).
    RegionKind region = RegionKind::Unknown;

    // Shadow-mode baked offset (populated by compute_layout()). Interpretation
    // depends on region:
    //   - Persistent / Accumulator: byte offset in the respective arena
    //   - FwdStack / BwdStack:      frame-local offset (runtime adds frame base)
    //   - SaveForBwd:               byte offset within SaveForBwd[block_layer_idx]
    // SIZE_MAX means "not assigned" (no bytes, or not in current graph).
    std::size_t offset = SIZE_MAX;

    /// Size in bytes of the tid's storage (populated by compute_layout). Zero
    /// for unassigned / dtype-unresolved tids.
    std::size_t bytes = 0;

    /// Backward-read-without-save-list-promotion flag. Set by
    /// finalize_save_for_bwd for tids produced in forward and consumed in
    /// backward that the save list filter excluded (recompute-mode
    /// candidates). Under arena consumption these tids CANNOT be aliased
    /// with later-forward-op outputs — replay produces them and backward
    /// reads them after the rest of the forward frame has executed, so
    /// their bytes must remain exclusive through the full forward op range.
    /// The legacy Stack bump allocator preserves them automatically (no
    /// reuse); arena coloring needs this flag to lengthen LayoutInfo's
    /// last_use to the frame end.
    bool retain_through_forward = false;

    /// Model-scope tensor (not a block-scoped activation) that's produced by
    /// an op in the forward graph and consumed by ops in more than one
    /// layer (e.g., Gemma4 `per_layer_inputs` — PLI tensor sliced per layer
    /// via compiler-synthesized narrow ops). Populated by pass 2 of
    /// `promote_cross_layer_fwd_reads`. Used by the executor's `mSaveMask`
    /// builder to keep the tensor alive across layer boundaries, and by
    /// `execute_backward`'s fwd-snapshot restore to carry the pointer into
    /// backward without blowing the BwdCrossLayer arena budget.
    bool cross_layer_global = false;

    bool is_cross_layer() const {
        return flags & kCrossLayer;
    }
    bool is_moe_offsets() const {
        return flags & kMoeOffsets;
    }
    bool is_d_blocks() const {
        return flags & kDBlocks;
    }
    bool is_blocks() const {
        return flags & kBlocks;
    }
    bool is_moe_gather() const {
        return flags & kMoeGather;
    }
    bool is_param_grad() const {
        return kind == TensorKind::ParamGrad;
    }
    bool is_accum_temp() const {
        return kind == TensorKind::AccumTemp;
    }
};

const char* tensor_kind_name(TensorKind k);

// ============================================================================
// Graph Segment (for split-attention CUDA graph capture)
// ============================================================================

/// A contiguous range of ops within a layer that can be captured as a single
/// CUDA graph, or must run eagerly (e.g., FlashAttention with doc masking).
struct GraphSegment {
    std::size_t start_op;  ///< Inclusive start index in CompiledGraph::ops
    std::size_t end_op;    ///< Exclusive end index in CompiledGraph::ops
    bool eager;            ///< true = run eagerly (attention ops with dynamic cu_seqlens)
};

// ============================================================================
// MLP Tile Group (for long-context tiled execution)
// ============================================================================

struct MlpTileGroup {
    std::size_t start_op_idx;  // first op in MLP sequence (view before up-proj matmul)
    std::size_t end_op_idx;    // last op in MLP sequence (view after down-proj matmul)
};

// ============================================================================
// Phase Tree (design/buffer-runtime-v4.md)
// ============================================================================

/// Structural phase kinds. A phase bounds a contiguous op range
/// [op_start, op_end). Children are nested scopes (e.g., FwdBlockSeq contains
/// FwdBlock[i]).
enum class PhaseKind : std::uint8_t {
    Custom,       ///< Generic scope (prologue/epilogue; root wrapper)
    FwdBlockSeq,  ///< Forward block sequence; fwd arena root
    BwdBlockSeq,  ///< Backward block sequence; bwd arena root
    FwdBlock,     ///< Single forward block; block_index set
    BwdBlock,     ///< Single backward block; block_index set
};

const char* phase_kind_name(PhaseKind k);

struct PhaseNode {
    PhaseKind kind = PhaseKind::Custom;
    std::size_t op_start = 0;         ///< Inclusive op index
    std::size_t op_end = 0;           ///< Exclusive op index
    int block_index = -1;             ///< FwdBlock/BwdBlock only; -1 otherwise
    std::vector<PhaseNode> children;  ///< Nested scopes; op ranges are sub-ranges of parent
    std::string label;                ///< Debug label (e.g., "FwdBlock[3]")
};

/// Pretty-print a phase tree for validation.
std::string dump_phase_tree(const PhaseNode& root);

// ============================================================================
// Instruction Stream
// ============================================================================

/// Flat-stream primitives emitted from the phase tree. Subsumes today's
/// implicit layer-boundary behavior (stack checkpoint/restore, save-list
/// prep, last-use pruning). Shadow-only in M4: emitted into CompiledGraph
/// but not yet consumed at runtime. The interpreter in M5 will execute
/// these instructions against baked offsets.
enum class InstKind : std::uint8_t {
    PhaseEnter,       ///< Enter a phase (stack checkpoint if FwdBlock/BwdBlock).
    PhaseExit,        ///< Exit a phase (stack restore if FwdBlock/BwdBlock).
    SegmentDispatch,  ///< Execute ops [op_start, op_end); graph-captured or eager.
    PruneByLastUse,   ///< Release tensors whose last-use falls in [op_start, op_end).
    RecomputeBlock,   ///< Replay forward block (block_index) into FwdStack before backward dispatch.
};

const char* inst_kind_name(InstKind k);

struct Instruction {
    InstKind kind = InstKind::PhaseEnter;

    // PhaseEnter / PhaseExit payload:
    PhaseKind phase_kind = PhaseKind::Custom;
    int block_index = -1;  ///< >= 0 for FwdBlock/BwdBlock

    // SegmentDispatch / PruneByLastUse payload:
    std::size_t op_start = 0;
    std::size_t op_end = 0;
    bool eager = false;  ///< SegmentDispatch only: true = eager, false = graph-captured
};

/// Pretty-print an instruction stream for validation.
std::string dump_instruction_stream(const std::vector<Instruction>& stream);

/// Static invariant check for an instruction stream. Verifies:
///   - PhaseEnter and PhaseExit are balanced and properly nested.
///   - SegmentDispatch op ranges cover [0, num_ops) exactly once (no gaps or
///     overlaps) — i.e., the flat-stream partitioning is complete.
///   - PruneByLastUse ranges are disjoint (no double-prune).
/// Returns empty string on success; otherwise a human-readable error list.
std::string validate_instruction_stream(const std::vector<Instruction>& stream, std::size_t num_ops);

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
    std::vector<std::string> tensor_id_to_name;              // O(1) reverse of tensor_name_to_id (hot-path friendly)
    std::vector<TensorMeta> tensor_meta;                     // per-ID pruning metadata
    std::unordered_map<std::string, int> ssa_base_to_id;     // SSA-stripped name -> highest-suffix tensor_id

    // Look up or return -1
    int find_tensor_id(const std::string& name) const {
        auto it = tensor_name_to_id.find(name);
        return (it != tensor_name_to_id.end()) ? it->second : -1;
    }

    /// O(1) reverse lookup: tensor_id -> canonical name. Returns empty string_view
    /// if `tid` is out of range. Preferred over scanning `tensor_name_to_id` on
    /// dispatcher hot paths (used by e.g. `base_param_from_grad_kind`).
    std::string_view name_for_tensor_id(int tid) const {
        if (tid < 0 || static_cast<std::size_t>(tid) >= tensor_id_to_name.size()) {
            return {};
        }
        return tensor_id_to_name[static_cast<std::size_t>(tid)];
    }

    /// Debuggability (P4.7): tid -> TensorMeta with region/offset/bytes
    /// pre-resolved. Pair with name_for_tensor_id() for error messages and
    /// debug dumps. Returns nullptr when tid is out of range.
    const TensorMeta* meta_for_tensor_id(int tid) const {
        if (tid < 0 || static_cast<std::size_t>(tid) >= tensor_meta.size()) {
            return nullptr;
        }
        return &tensor_meta[static_cast<std::size_t>(tid)];
    }

    const TensorRole* role_for_tensor_id(int tid) const {
        if (const TensorMeta* meta = meta_for_tensor_id(tid)) {
            return &meta->role;
        }
        return nullptr;
    }

    /// Reverse lookup table `(layer_idx, slot) -> tid`, populated by
    /// compute_layout. Row layout:
    /// `slot_tid_by_layer[layer_idx][static_cast<std::size_t>(slot)]`.
    /// Returns -1 when the (layer, slot) pair has no tid in this graph.
    std::vector<std::array<int, static_cast<std::size_t>(TensorSlot::Mapped) + 1>> slot_tid_by_layer;

    int slot_to_tid(int layer_idx, TensorSlot slot) const {
        if (layer_idx < 0 || static_cast<std::size_t>(layer_idx) >= slot_tid_by_layer.size()) {
            return -1;
        }
        const auto slot_idx = static_cast<std::size_t>(slot);
        if (slot_idx >= slot_tid_by_layer[static_cast<std::size_t>(layer_idx)].size()) {
            return -1;
        }
        return slot_tid_by_layer[static_cast<std::size_t>(layer_idx)][slot_idx];
    }

    /// Debuggability (P4.7): name -> TensorMeta via name_to_id + meta_for_tensor_id.
    const TensorMeta* meta_for_name(const std::string& name) const {
        int tid = find_tensor_id(name);
        return tid >= 0 ? meta_for_tensor_id(tid) : nullptr;
    }

    const TensorRole* role_for_name(const std::string& name) const {
        int tid = find_tensor_id(name);
        return tid >= 0 ? role_for_tensor_id(tid) : nullptr;
    }

    std::size_t count_tensors_with_quant_state(QuantState state) const {
        std::size_t count = 0;
        for (const auto& meta : tensor_meta) {
            if (meta.role.quant_state == state) {
                ++count;
            }
        }
        return count;
    }

    std::size_t count_ops_with_comm(CommunicationKind kind) const {
        std::size_t count = 0;
        for (const auto& op : ops) {
            if (op.comm_profile.kind == kind) {
                ++count;
            }
        }
        return count;
    }

    std::size_t count_grouped_ops() const {
        std::size_t count = 0;
        for (const auto& op : ops) {
            if (op.grouped_semantics.is_grouped) {
                ++count;
            }
        }
        return count;
    }

    std::size_t count_ops_with_capability(std::uint32_t flag) const {
        std::size_t count = 0;
        for (const auto& op : ops) {
            if (op.default_caps.has(flag)) {
                ++count;
            }
        }
        return count;
    }

    std::size_t count_ops_with_epilogue(std::uint32_t flag) const {
        std::size_t count = 0;
        for (const auto& op : ops) {
            if (op.epilogue_support.has(flag)) {
                ++count;
            }
        }
        return count;
    }

    std::size_t count_ops_with_moe_capability(std::uint32_t flag) const {
        std::size_t count = 0;
        for (const auto& op : ops) {
            if (op.moe_caps.has(flag)) {
                ++count;
            }
        }
        return count;
    }

    std::size_t count_ops_with_matmul_capability(std::uint32_t flag) const {
        std::size_t count = 0;
        for (const auto& op : ops) {
            if (op.matmul_caps.has(flag)) {
                ++count;
            }
        }
        return count;
    }

    std::size_t count_ops_supporting_storage(StorageTier tier) const {
        std::size_t count = 0;
        for (const auto& op : ops) {
            if (op.storage_compat.supports(tier)) {
                ++count;
            }
        }
        return count;
    }

    std::size_t count_fusion_candidate_starts() const {
        return FusionRuleRegistry::instance().count_matching_starts(ops);
    }

    /// Debuggability (P4.7): format "tid=5 name='blocks[3].ln1' region=FwdStack
    /// block=3 offset=0x1000 bytes=32768". Intended for error messages and
    /// exception rewrites. Returns "<tid=N unknown>" when tid is invalid.
    std::string describe_tensor_id(int tid) const;

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

    /// Populate layer_segments by scanning each layer for FlashAttention ops.
    /// Call after annotate_layer_boundaries().
    void compute_layer_segments();

    /// Shadow-mode phase tree (design/buffer-runtime-v4.md). Built post-hoc by
    /// GraphCompiler::build_phase_tree() after annotate_layer_boundaries().
    /// Validates that the phase reconstruction covers every op and nests
    /// cleanly. Empty optional if the graph has no layer boundaries (e.g.,
    /// non-block-stacked compiles).
    std::optional<PhaseNode> phase_tree;

    /// Per-region peak bytes populated by compute_layout().
    /// Consumed by compute_arena_sizes() to size the phase arenas.
    std::size_t persistent_bytes = 0;
    std::size_t persistent_activation_bytes = 0;
    std::size_t model_scope_persistent_bytes = 0;
    std::size_t accumulator_bytes = 0;
    std::size_t fwd_stack_peak = 0;  // max over frames
    std::size_t bwd_stack_peak = 0;  // max over frames
    std::size_t save_for_bwd_bytes = 0;
    std::vector<std::size_t> save_for_bwd_block_bytes;  // per-block sizes

    /// FNV-1a 64-bit hash of the fully-baked layout (populated by
    /// compute_layout). Mixes every tid's (region, block_layer_idx, offset,
    /// bytes) plus the per-region peaks. Intended for distributed
    /// determinism checks — all ranks running the same compile should
    /// produce byte-identical hashes. `0` if compute_layout has not run.
    std::uint64_t layout_hash = 0;

    /// Shadow-mode instruction stream emitted from the phase tree (M4).
    /// Flat linear sequence of primitives that would drive the M5 interpreter.
    /// Empty if phase_tree is empty.
    std::vector<Instruction> instruction_stream;

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

    // Compile a forward or backward graph. `is_backward=true` selects the
    // backward dispatch function for each op when both a forward and a
    // backward are registered (e.g. View, Zeros, MatmulBackward).
    CompiledGraph compile(const Graph& graph, long B, long T, bool is_backward = false);

    // Reset the per-compile-pair namespace. This clears:
    //   - tensor-id map (so fresh forward+backward get consistent tids)
    //   - mExtraShapes, mTensorShapes, mTensorDtypes (shape/dtype databases)
    // These all share the same lifetime: forward and backward are compiled
    // in sequence and intentionally share this state so:
    //   - A tensor name gets ONE tid across both graphs (no slot collisions
    //     when forward TensorRefs leak into backward).
    //   - Backward can look up forward tensor shapes (e.g., zeros ops with
    //     shape_like referencing a forward split output).
    // Call this once before recompiling the forward+backward pair.
    void reset_tid_namespace();

    // Update batch/sequence dimensions for shape resolution
    void update_dimensions(long B, long T);

    // Get the slot registry (for passing to CompiledExecutor)
    const TensorSlotRegistry& slot_registry() const {
        return mSlotRegistry;
    }

private:
    CompiledOpType classify_op(const std::string& op_type) const;

    TensorRef resolve_tensor_ref(const std::string& name, bool is_output, const Operation& op, const ShapeEnv& env);

    CompiledAttrs resolve_attrs(const Operation& op, CompiledOpType type, const ShapeEnv& env, const Graph& graph);

    void annotate_layer_boundaries(CompiledGraph& graph);

    /// Build the phase tree from layer boundary indices. Must be called
    /// after annotate_layer_boundaries(). The tree wraps a prologue (ops
    /// before the first layer), a FwdBlockSeq/BwdBlockSeq containing
    /// per-layer FwdBlock/BwdBlock nodes, and an epilogue. When
    /// `SUROGATE_DEBUG_PHASE_TREE=1` is set, the tree is dumped to stderr.
    void build_phase_tree(CompiledGraph& graph, bool is_backward);

    /// Assign each tid a RegionKind based on TensorMeta::kind + block_layer_idx
    /// (populated by classify_tensors). Shadow-only in M2: consumed only by the
    /// debug dump gated on `SUROGATE_DEBUG_REGIONS=1`. M3's layout pass will
    /// read TensorMeta::region to bake buffer offsets.
    void derive_regions(CompiledGraph& graph, bool is_backward);

    /// Promote FwdStack tids read across layer boundaries to SaveForBwd so
    /// the source layer's buffer survives later layers' FwdStack reuse.
    /// Example: Gemma4 shared-KV layers read `blocks[L_source].qkv_rope` as
    /// their `kv_source`; under shared-arena coloring the reading layer's
    /// writes can overlap the source's bytes before flash-attention consumes
    /// K/V, corrupting outputs. SaveForBwd is per-layer persistent, so this
    /// restores the producer/consumer invariant across layers.
    void promote_cross_layer_fwd_reads(CompiledGraph& graph);

    /// Flatten the phase tree to a linear instruction stream (M4). Emits
    /// PhaseEnter/PhaseExit around each phase, plus SegmentDispatch +
    /// PruneByLastUse for leaf phases (ops live at leaves). Shadow-only;
    /// M5's interpreter consumes this stream. Dump gated on
    /// `SUROGATE_DEBUG_INSTR_STREAM=1`.
    void emit_instruction_stream(CompiledGraph& graph);

    // Shape validation methods
    struct TensorShape {
        std::vector<long> dims;
        bool inferred = false;  // true if inferred, false if from IR
        std::string source_op;  // Operation that produced this tensor
    };

    bool resolve_tensor_shape(const std::string& name, std::vector<long>& shape);
    void infer_output_shapes(const Operation& op,
                             CompiledOpType type,
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
    bool mDebugShapes = false;      // Set via SUROGATE_DEBUG_SHAPES env var
    bool mHasHybridBlocks = false;  // True if model uses HybridStackedBlocks

    // Per-layer dimensions for hybrid models (populated from IR param shapes)
    std::vector<BlockTypeDims> mPerLayerDims;
    ShapeEnv make_layer_env(int layer_idx) const;

    /// Rewrite a global-dims shape into per-layer dims when the field name
    /// matches a per-layer-varying slot (qkv / att / q* / mlp_up / swiglu
    /// and their _flat / _norm / _normed variants). No-op for non-hybrid
    /// models (mPerLayerDims empty) or unrecognized fields.
    void apply_per_layer_dim_override(std::vector<long>& shape, const std::string& base_field, int layer_idx) const;

    // Tensor ID assignment state (per-compile, reset at start of compile())
    std::unordered_map<std::string, int> mTensorIdMap;  // name -> tensor_id
    int mNextTensorId = 0;

    // Assign or retrieve a tensor ID for the given name
    int assign_tensor_id(const std::string& name);

    // Register well-known external tensor names (init bindings, MoE side-channel, etc.)
    void register_external_names(CompiledGraph& graph);

    // Build per-ID metadata for pruning (flags, block_layer_idx, ssa_base_to_id)
    void build_tensor_metadata(CompiledGraph& graph);

    // Classify every tensor (kind + base_*_tid) by authoritative lookup against
    // the parameter store and the op producer map. Called after
    // build_tensor_metadata. Replaces ad-hoc string predicates like
    // base_param_from_grad / should_alias_autodiff_accum_name (Phase 0: data
    // only; callers still use the legacy predicates until Phase 1 flips them).
    void classify_tensors(CompiledGraph& graph);
};

/// Layout peaks + offset baking. For each region, computes offsets and peak bytes:
///   - Persistent, Accumulator, SaveForBwd: bump (sum of tensor bytes)
///   - FwdStack, BwdStack: per-block-frame first-fit-by-offset coloring
/// Writes TensorMeta::offset for every live tid and populates
/// graph.layout_hash for distributed determinism checks. Dump gated on
/// `SUROGATE_DEBUG_LAYOUT=1`.
void compute_layout(CompiledGraph& graph, bool is_backward, bool fwd_per_layer_sections = false);

/// Recompute the FNV-1a 64-bit layout hash from tensor_meta + peaks. Called
/// internally by compute_layout(); exposed so distributed init can
/// cross-check graph.layout_hash across ranks (NCCL/MPI allreduce on the
/// 64-bit value; any mismatch means coloring / region assignment diverged
/// and the ranks would run incompatible buffer plans).
std::uint64_t compute_layout_hash(const CompiledGraph& graph);

/// Phase-tree arena allocator. Allocates one flat device buffer per region
/// family. Sizes computed from TensorMeta::offset + bytes across both fwd
/// and bwd compiles — the union size, since fwd and bwd frames reuse the
/// same stack arenas at different times.
struct PhaseArenas {
    std::byte* persistent_ptr = nullptr;
    std::size_t persistent_bytes = 0;

    // PersistentActivation: training-wide non-block activations / I/O buffers.
    // Arena base is a fixed region; rebind writes at `base + meta.offset`.
    // Separate from the Persistent arena so WM/LoRA/QLoRA can claim the
    // Persistent arena without colliding with compile-time activation offsets.
    std::byte* persistent_activation_ptr = nullptr;
    std::size_t persistent_activation_bytes = 0;

    // ModelScopePersistent: forward intermediates produced outside any block
    // AND read by backward (e.g., Gemma4 PLI pli_proj_rn_flat that feeds
    // rmsnorm_backward's x input; pli_narrow_layer* per-layer slices of
    // per_layer_inputs). Separate from PersistentActivation because those
    // are I/O-like buffers with stable compile-time offsets, while this
    // region holds live intermediates whose Stack-temp-alloc would be
    // overwritten by later temp_allocs before backward reads them. Arena
    // base is a fixed region; rebind at `base + meta.offset` makes the
    // forward producer write into the persistent slot directly.
    std::byte* model_scope_persistent_ptr = nullptr;
    std::size_t model_scope_persistent_bytes = 0;

    std::byte* accumulator_ptr = nullptr;
    std::size_t accumulator_bytes = 0;

    // FwdStack / BwdStack: peak across all block frames (frames don't coexist).
    std::byte* fwd_stack_ptr = nullptr;
    std::size_t fwd_stack_bytes = 0;

    std::byte* bwd_stack_ptr = nullptr;
    std::size_t bwd_stack_bytes = 0;

    // SaveForBwd: concatenated per-block slots. Block i's save slot starts at
    // save_for_bwd_ptr + save_for_bwd_block_bases[i].
    std::byte* save_for_bwd_ptr = nullptr;
    std::size_t save_for_bwd_bytes = 0;
    std::vector<std::size_t> save_for_bwd_block_bases;

    // Unified stack arena. Sized to match the DeviceMemoryStack capacity
    // so DslRunState.Stack can be rebased onto this buffer via
    // set_stack_buffer(). Separate from fwd_stack_bytes / bwd_stack_bytes
    // because those are block-local coloring peaks and don't cover
    // non-block ops (LM-head d_logits, embeddings, prologue temps).
    std::byte* unified_stack_ptr = nullptr;
    std::size_t unified_stack_bytes = 0;

    // BwdCrossLayer arena. Bump-allocator destination for tids produced
    // during backward that survive past their block's layer_end
    // (d_router_logits for MoE aux-loss, cross-layer d_residuals).
    std::byte* bwd_cross_layer_ptr = nullptr;
    std::size_t bwd_cross_layer_bytes = 0;

    // MoE-saved arena. Cross-step monotonic bump allocator for the
    // name-keyed persistent buffers MoE routing saves at layer boundaries
    // (compiled_ops_save.cpp save_moe_layer_tensors +
    // prepare_saved_buffers_for_capture + persist_saved_source_now).
    std::byte* moe_saved_ptr = nullptr;
    std::size_t moe_saved_bytes = 0;

    bool allocated = false;
};

/// Compute arena sizes from baked offsets in both graphs. Must be called after
/// finalize_save_for_bwd(). Does NOT cudaMalloc — just populates sizes.
void compute_arena_sizes(PhaseArenas& arenas,
                         const CompiledGraph& fwd,
                         const CompiledGraph& bwd,
                         int num_layers,
                         std::size_t stack_bytes = 0,
                         std::size_t bwd_cross_layer_bytes = 0,
                         std::size_t moe_saved_bytes = 0);

/// Static upper bound on bytes the BwdCrossLayer arena needs across one
/// backward call. Mirrors the runtime persist logic in
/// `compiled_ops_execute.cpp::bwd_layer_end_cleanup`: at each layer-end,
/// every named tid that's stack-resident (FwdStack/BwdStack/Recomputed/
/// Unknown region) AND has last_use beyond that layer-end op gets copied
/// into the arena; this function sums those bytes once per tid (the
/// runtime guarantees each tid is persisted at most once per backward
/// call). Over-counts when slot-aliased tids share a runtime pointer
/// (runtime dedups, compile time can't), so callers should treat the
/// result as an upper bound and add a small slack margin.
std::size_t estimate_bwd_cross_layer_bytes(const CompiledGraph& bwd);

/// cudaMalloc all arenas at their computed sizes. arenas.allocated is set to
/// true on success. Safe to call only after compute_arena_sizes().
void allocate_phase_arenas(PhaseArenas& arenas);

/// cudaFree all arenas and reset pointers.
void release_phase_arenas(PhaseArenas& arenas);

/// Resolve a tid to its arena-backed device pointer (what the pointer WOULD
/// be if the arena was consumed for this region). Returns nullptr if:
///   - arenas is not allocated
///   - tid is out of range
///   - TensorMeta::region is not an arena-backed region
///   - TensorMeta::offset is SIZE_MAX (unassigned)
///   - region is FwdStack/BwdStack but the block_layer_idx is invalid
/// Caller supplies the block contexts so block-scoped regions resolve to the
/// correct frame base; pass -1 when outside a block.
std::byte* resolve_tid_in_arena(const PhaseArenas& arenas, const CompiledGraph& graph, int tid);

/// Shadow-mode validator reporting how well the arena plan covers the graph's
/// live tids. Self-contained — uses TensorMeta::{region, offset, bytes}
/// populated by compute_layout(); no runtime state required. Logs when
/// `SUROGATE_DEBUG_ARENA_COVERAGE=1`.
struct ArenaCoverage {
    std::size_t covered = 0;        // tids whose region has an allocated arena slot
    std::size_t total = 0;          // tids with region != Unknown
    std::size_t size_exceeded = 0;  // offset+bytes past the arena's capacity
};
ArenaCoverage validate_arena_coverage(const PhaseArenas& arenas, const CompiledGraph& graph);

/// Op-operand coverage audit. Counts how many TensorRef uses (across all
/// ops' inputs + outputs) index a tid that would be served by a baked
/// arena read — i.e. meta.region != Unknown AND the corresponding arena
/// is allocated.
///
/// Distinct from ArenaCoverage, which counts distinct tids. Operands are
/// the reads that actually matter at dispatch time: one tid consumed by N
/// ops contributes N here.
struct OpOperandCoverage {
    std::size_t covered_inputs = 0;   ///< op.inputs entries with an arena-served tid
    std::size_t covered_outputs = 0;  ///< op.outputs entries with an arena-served tid
    std::size_t total_inputs = 0;
    std::size_t total_outputs = 0;
    std::array<std::size_t, 10> by_region{};  ///< per-RegionKind count of covered operands
};
OpOperandCoverage validate_op_operand_coverage(const PhaseArenas& arenas, const CompiledGraph& graph);

/// Hot-path accessor for the baked (region, offset, bytes, dtype) operand
/// view. Resolves `ref.tensor_id` against `graph.tensor_meta`. `dtype`
/// comes from the ref (per-use overrides are legal), everything else from
/// the tid. Returns nullopt if the tid is out of range, the region is
/// Unknown, or the offset is unassigned.
struct BakedOperand {
    RegionKind region;
    int block_layer_idx;
    std::size_t offset;
    std::size_t bytes;
    ETensorDType dtype;
};
std::optional<BakedOperand> baked_view(const CompiledGraph& graph, const TensorRef& ref);

/// Cross-graph SaveForBwd promotion. derive_regions() runs per-direction
/// and cannot see across the pair, so block activations that are produced
/// in forward and consumed in backward land in FwdStack / BwdStack. This
/// pass walks both graphs, identifies shared tids, and promotes their
/// region to SaveForBwd in both compiles, then re-runs compute_layout()
/// on both so offsets reflect the final region assignment. Safe to call
/// once per (forward, backward) compile pair; idempotent.
///
/// When `save_names` is provided, promotion is restricted to tids whose
/// name appears in the set — matching the runtime save list
/// (GraphExecutor::mSaveList). Tids that cross the fwd→bwd boundary but
/// are NOT in the save list (handled via Stack or recompute at runtime)
/// stay in FwdStack / BwdStack. Passing an empty set promotes zero tids
/// (valid in recompute/forward-replay modes where the runtime never
/// consumes arena-backed saves). Passing
/// `std::nullopt` disables filtering entirely, preserving prior behavior
/// for callers that haven't plumbed the save list.
void finalize_save_for_bwd(CompiledGraph& fwd,
                           CompiledGraph& bwd,
                           std::optional<std::unordered_set<std::string>> save_names = std::nullopt,
                           bool fwd_per_layer_sections = false);

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_GRAPH_COMPILER_H
