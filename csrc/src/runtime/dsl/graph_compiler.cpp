// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Compiled operation dispatch for DSL Graph executor.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "runtime/dsl/graph_compiler.h"
#include "runtime/dsl/ir.h"
#include "runtime/executor/graph_executor_helpers.h"
#include "runtime/executor/graph_executor_utils.h"
#include "runtime/executor/op_registry.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/core/backward_hooks.h"
#include "runtime/core/forward_hooks.h"
#include "runtime/lora/lora_types.h"

namespace dsl {

namespace {

std::string schema_slot_from_weight_name(std::string_view weight_name) {
    const std::size_t dot = weight_name.rfind('.');
    std::string_view slot = (dot == std::string_view::npos) ? weight_name : weight_name.substr(dot + 1);
    if (!slot.empty() && slot.back() == '?') {
        slot.remove_suffix(1);
    }
    return std::string(slot);
}

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value) return false;
    const std::string_view text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
}

void set_forward_hook_schema_slot(CompiledAttrs& attrs, modules::ForwardHookPoint point, std::string_view schema_slot) {
    attrs.forward_hook_point = point;
    attrs.forward_hook_schema_slot = std::string(schema_slot);
}

TensorRoleKind role_kind_from_tensor_kind(TensorKind kind) {
    switch (kind) {
        case TensorKind::ForwardParam: return TensorRoleKind::Param;
        case TensorKind::ForwardActivation: return TensorRoleKind::Activation;
        case TensorKind::ParamGrad: return TensorRoleKind::ParamGrad;
        case TensorKind::ActivationGrad:
        case TensorKind::AccumTemp: return TensorRoleKind::ActivationGrad;
        case TensorKind::LossInput: return TensorRoleKind::LossInput;
        case TensorKind::Scratch: return TensorRoleKind::Scratch;
        case TensorKind::Unknown: return TensorRoleKind::Unknown;
    }
    return TensorRoleKind::Unknown;
}

}  // namespace

/// Strip trailing SSA-style numeric suffix (e.g., "qkv_rope_7" -> "qkv_rope")
/// The DSL IR generates unique tensor names with suffixes like _0, _7, _10, etc.
/// This function removes these suffixes for field name matching.
std::string strip_ssa_suffix(const std::string& field) {
    auto pos = field.rfind('_');
    if (pos == std::string::npos || pos == 0) {
        return field;
    }
    // Check if everything after the underscore is digits
    bool all_digits = true;
    for (std::size_t i = pos + 1; i < field.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(field[i]))) {
            all_digits = false;
            break;
        }
    }
    if (all_digits && pos + 1 < field.size()) {
        return field.substr(0, pos);
    }
    return field;
}

namespace {

inline bool is_capture_unsafe_op_type(CompiledOpType type) {
    switch (type) {
        // MoE routing / grouped GEMM rely on per-step host metadata and dynamic routing.
        // (MoESigmoid/MoESoftmax are plain element-wise kernels reused by non-MoE
        // models like Qwen3.5; they're capture-safe and intentionally excluded.)
        case CompiledOpType::MoETopK:
        case CompiledOpType::MoEPermute:
        case CompiledOpType::MoEGroupedGemm:
        case CompiledOpType::MoEGroupedGemmGateUp:
        case CompiledOpType::MoEGroupedGemmDown:
        case CompiledOpType::MoEUnpermute:
        case CompiledOpType::MoEExpertBiasAdd:
        case CompiledOpType::MoETopKBackward:
        case CompiledOpType::MoEPermuteBackward:
        case CompiledOpType::MoEGroupedGemmBackward:
        case CompiledOpType::MoEGroupedGemmGateUpBackward:
        case CompiledOpType::MoEGroupedGemmDownBackward:
        case CompiledOpType::MoEUnpermuteBackward:
        case CompiledOpType::MoEExpertBiasAddBackward:
        // EP ops perform per-step host-side split/reorder bookkeeping.
        case CompiledOpType::EpDispatch:
        case CompiledOpType::EpCombine:
        case CompiledOpType::EpDispatchBackward:
        case CompiledOpType::EpCombineBackward: return true;
        default: return false;
    }
}

// ============================================================================
// Global tensor role table (single source of truth for "special" names)
// ============================================================================
//
// Names like "xF", "encoded", "h_out", "xN", "residualN", "freq_cis" are a
// contract between the Python DSL and the C++ runtime: the Python side emits
// these names at specific points in the graph, and the compiler needs to
// recognize them to assign the right shape/dtype/slot. Historically each
// function that cared did its own `name == "xF"` string comparison, and those
// lists drifted out of sync — renaming a tensor in Python silently broke
// whichever compiler site forgot to update.
//
// This enum + `global_role_for_name()` consolidate every such check into ONE
// place. Downstream sites dispatch on the enum. To add a new global tensor,
// add a role below AND register it in `global_role_for_name()`; there is no
// other place to update.
//
// Where possible we defer to `builtin_slot_from_name()` (the slot-registry
// table in tensor_slot_registry.cpp); only names that don't have a dedicated
// TensorSlot enum value (xN, residualN, flat aliases) live in the fallback
// block below.
enum class GlobalRole {
    None,
    Encoded,           // x0 / encoded — embedding output, {B, T, C}
    StackOutput,       // xN — block stack top-of-column, {B, T, C}
    StackResidual,     // residualN / residual0 — block stack residual, {B, T, C}
    FinalNormOutput,   // xF / ln_final — final-norm output, {B, T, C}
    FinalNormOutFlat,  // xF_flat — {B*T, C}
    FinalResidual,     // final_residual / residual_final — {B, T, C}
    FinalNormRstd,     // ln_final_rstd — {B, T}
    TokenIds,          // token_ids / position_ids — {B, T} (int32)
    LossBatch,         // loss / losses / targets / labels / d_loss — {B * T}
    RopeFreqs,         // freq_cis / rope_freqs — handled specially elsewhere
};

GlobalRole global_role_for_name(std::string_view name) {
    // Prefer the slot registry's built-in mapping. Any name registered there
    // has already been classified; we only need to translate the TensorSlot
    // to a GlobalRole.
    const TensorSlot slot = builtin_slot_from_name(std::string(name));
    switch (slot) {
        case TensorSlot::Encoded: return GlobalRole::Encoded;
        case TensorSlot::LNFinal: return GlobalRole::FinalNormOutput;
        case TensorSlot::LNFinalRSTD: return GlobalRole::FinalNormRstd;
        case TensorSlot::FinalResidual: return GlobalRole::FinalResidual;
        case TensorSlot::FreqCis: return GlobalRole::RopeFreqs;
        case TensorSlot::TokenIDs:
        case TensorSlot::PositionIDs: return GlobalRole::TokenIds;
        case TensorSlot::Targets:
        case TensorSlot::Losses:
        case TensorSlot::DLoss: return GlobalRole::LossBatch;
        default: break;
    }

    // Fallback: names that have no dedicated TensorSlot enum value.
    // Keep this list tight; add new roles above rather than inflating it.
    if (name == "xN") return GlobalRole::StackOutput;
    if (name == "residualN" || name == "residual0") return GlobalRole::StackResidual;
    if (name == "xF_flat") return GlobalRole::FinalNormOutFlat;
    if (name == "ln_final_flat") return GlobalRole::FinalNormOutput;
    if (name == "labels") return GlobalRole::LossBatch;
    // Substring matches for rope frequencies (handles qualified names like
    // "blocks[0].rope_freqs" when they surface at the global level).
    if (name.find("rope_freqs") != std::string_view::npos || name.find("freq_cis") != std::string_view::npos) {
        return GlobalRole::RopeFreqs;
    }
    return GlobalRole::None;
}

bool infer_known_tensor_shape(std::string_view name,
                              const modules::ModelConfig& config,
                              long B,
                              long T,
                              std::vector<long>& shape) {
    if (starts_with(name, kSavedPrefix)) {
        name = name.substr(kSavedPrefix.size());
    }
    // Strip gradient prefix so d_blocks[N].field matches the same patterns
    // as blocks[N].field — gradient tensors have the same shape as activations.
    if (starts_with(name, "d_")) {
        name = name.substr(2);
    }

    int layer_idx = -1;
    std::string field;
    if (parse_block_param(name, layer_idx, field)) {
        // Strip autodiff accumulation suffixes (_from_NNN, _accum_NNN) so that
        // gradient-accumulation tensors like "ln2_flat_from_497" match "ln2_flat".
        // strip_ssa_suffix only handles trailing _NNN, but autodiff generates
        // "_from_<opid>" and "_accum_<counter>" which need two-part stripping.
        for (const char* pat : {"_from_", "_accum_"}) {
            auto pos = field.find(pat);
            if (pos == std::string::npos) continue;
            size_t after_pos = pos + std::strlen(pat);
            bool all_digits = after_pos < field.size();
            for (size_t i = after_pos; i < field.size(); ++i) {
                if (!std::isdigit(static_cast<unsigned char>(field[i]))) {
                    all_digits = false;
                    break;
                }
            }
            if (all_digits) {
                field = field.substr(0, pos);
                break;
            }
        }

        const long C = config.HiddenSize;
        const long D = config.IntermediateSize;
        const long MUp = config.mlp_up_rows();
        const long Hq = config.NumQueryHeads;
        const long Hkv = config.NumKeyValHeads;
        const long Hs = config.head_size();
        const long QKV = config.qkv_channels();

        if (field == "ln1" || field == "ln2" || field == "att_out" || field == "mlp_down" || field == "res_att" ||
            field == "res_ffn" || field == "res_in") {
            shape = {B, T, C};
            return true;
        }
        if (field == "ln1_flat" || field == "ln2_flat" || field == "att_out_flat" || field == "mlp_down_flat") {
            shape = {B * T, C};
            return true;
        }
        if (field == "ln1_rstd" || field == "ln2_rstd") {
            shape = {B, T};
            return true;
        }
        // NOTE: qkv, qkv_flat, qkv_rope, att, att_flat, lse, q_rstd, k_rstd
        // shapes are resolved from the DSL activation layout in hybrid models
        // (per-block-type overrides in compile loop below). These global fallbacks
        // are needed for non-hybrid models and for initial shape validation.
        if (field == "qkv" || field == "qkv_rope") {
            shape = {B, T, QKV};
            return true;
        }
        if (field == "qkv_flat" || field == "qkv_biased") {
            shape = {B * T, QKV};
            return true;
        }
        if (field == "q_rstd") {
            shape = {B, T, Hq};
            return true;
        }
        if (field == "k_rstd") {
            shape = {B, T, Hkv};
            return true;
        }
        if (field == "att") {
            shape = {B, T, Hq * Hs};
            return true;
        }
        if (field == "att_flat") {
            shape = {B * T, Hq * Hs};
            return true;
        }
        if (field == "lse") {
            shape = {B, Hq, T};
            return true;
        }
        if (field == "mlp_up") {
            shape = {B, T, MUp};
            return true;
        }
        if (field == "mlp_up_flat") {
            shape = {B * T, MUp};
            return true;
        }
        if (field == "swiglu") {
            shape = {B, T, D};
            return true;
        }
        if (field == "swiglu_flat") {
            shape = {B * T, D};
            return true;
        }
    }

    // Global tensor names are routed through the single GlobalRole table above
    // — add new globals there, not here.
    switch (global_role_for_name(name)) {
        case GlobalRole::Encoded:
        case GlobalRole::StackOutput:
        case GlobalRole::StackResidual:
        case GlobalRole::FinalNormOutput:
        case GlobalRole::FinalResidual: shape = {B, T, config.HiddenSize}; return true;
        case GlobalRole::FinalNormOutFlat: shape = {B * T, config.HiddenSize}; return true;
        case GlobalRole::FinalNormRstd:
        case GlobalRole::TokenIds: shape = {B, T}; return true;
        case GlobalRole::LossBatch: shape = {B * T}; return true;
        case GlobalRole::RopeFreqs:
        case GlobalRole::None: break;
    }

    return false;
}

}  // namespace

// ============================================================================
// Operation type conversion
// ============================================================================

CompiledOpType op_type_from_string(const std::string& op_type) {
    // Use a static lookup table for O(1) average case
    static const std::unordered_map<std::string, CompiledOpType> type_map = {
        {"embedding", CompiledOpType::Embedding},
        {"zeros", CompiledOpType::Zeros},
        {"ones", CompiledOpType::Ones},
        {"fused_residual_rmsnorm", CompiledOpType::FusedResidualRMSNorm},
        {"rmsnorm", CompiledOpType::RMSNorm},
        {"layernorm", CompiledOpType::LayerNorm},
        {"view", CompiledOpType::View},
        {"transpose", CompiledOpType::Transpose},
        {"transpose_backward", CompiledOpType::Transpose},
        {"split", CompiledOpType::Split},
        {"split_backward", CompiledOpType::Split},
        {"narrow", CompiledOpType::Narrow},
        {"concat", CompiledOpType::Concat},
        {"concat_backward", CompiledOpType::Concat},
        {"add", CompiledOpType::Add},
        {"matmul", CompiledOpType::Matmul},
        {"matmul_bias", CompiledOpType::MatmulBias},
        {"bias_add", CompiledOpType::BiasAdd},
        {"swiglu", CompiledOpType::SwiGLU},
        {"gelu_glu", CompiledOpType::GeluGlu},
        {"gpt_oss_moe_act", CompiledOpType::GptOssMoeAct},
        {"silu", CompiledOpType::Silu},
        {"sigmoid", CompiledOpType::MoESigmoid},
        {"gelu", CompiledOpType::Gelu},
        {"relu2", CompiledOpType::Relu2},
        {"mul", CompiledOpType::Mul},
        {"scale", CompiledOpType::Scale},
        {"mask_scatter", CompiledOpType::MaskScatter},
        {"deepstack_inject", CompiledOpType::DeepstackInject},
        {"matmul_swiglu", CompiledOpType::MatmulSwiGLU},
        {"qkv_qk_norm", CompiledOpType::QKVQKNorm},
        {"qkv_qk_norm_rope", CompiledOpType::QKVQKNormRoPE},
        {"mrope", CompiledOpType::MRoPE},
        {"rope", CompiledOpType::RoPE},
        {"flash_attention", CompiledOpType::FlashAttention},
        {"flash_attention_qkv", CompiledOpType::FlashAttention},
        {"cross_entropy", CompiledOpType::CrossEntropyLoss},
        {"cross_entropy_loss", CompiledOpType::CrossEntropyLoss},
        {"fused_lm_head_loss", CompiledOpType::FusedLMHeadLoss},
        {"lm_head_loss", CompiledOpType::FusedLMHeadLoss},
        // MoE forward operations
        {"moe_softmax", CompiledOpType::MoESoftmax},
        {"moe_sigmoid", CompiledOpType::MoESigmoid},
        {"moe_topk", CompiledOpType::MoETopK},
        {"moe_permute", CompiledOpType::MoEPermute},
        {"moe_grouped_gemm", CompiledOpType::MoEGroupedGemm},
        {"moe_grouped_gemm_gate_up", CompiledOpType::MoEGroupedGemmGateUp},
        {"moe_grouped_gemm_down", CompiledOpType::MoEGroupedGemmDown},
        {"moe_unpermute", CompiledOpType::MoEUnpermute},
        {"moe_expert_bias_add", CompiledOpType::MoEExpertBiasAdd},
        // Expert Parallelism forward operations
        {"ep_dispatch", CompiledOpType::EpDispatch},
        {"ep_combine", CompiledOpType::EpCombine},
        // Backward operations
        {"view_backward", CompiledOpType::ViewBackward},
        {"add_backward", CompiledOpType::AddBackward},
        {"matmul_backward", CompiledOpType::MatmulBackward},
        {"bias_add_backward", CompiledOpType::BiasAddBackward},
        {"swiglu_backward", CompiledOpType::SwiGLUBackward},
        {"gelu_glu_backward", CompiledOpType::GeluGluBackward},
        {"gpt_oss_moe_act_backward", CompiledOpType::GptOssMoeActBackward},
        {"silu_backward", CompiledOpType::SiluBackward},
        {"sigmoid_backward", CompiledOpType::MoESigmoidBackward},
        {"gelu_backward", CompiledOpType::GeluBackward},
        {"relu2_backward", CompiledOpType::Relu2Backward},
        {"mul_backward", CompiledOpType::MulBackward},
        {"scale_backward", CompiledOpType::ScaleBackward},
        {"narrow_backward", CompiledOpType::NarrowBackward},
        {"mask_scatter_backward", CompiledOpType::MaskScatterBackward},
        {"deepstack_inject_backward", CompiledOpType::DeepstackInjectBackward},
        {"matmul_swiglu_backward", CompiledOpType::MatmulSwiGLUBackward},
        {"qkv_qk_norm_backward", CompiledOpType::QKVQKNormBackward},
        {"rope_backward", CompiledOpType::RoPEBackward},
        {"qkv_qk_norm_rope_backward", CompiledOpType::QKVQKNormRoPEBackward},
        {"mrope_backward", CompiledOpType::MRoPEBackward},
        {"flash_attention_backward", CompiledOpType::FlashAttentionBackward},
        {"zeros_backward", CompiledOpType::ZerosBackward},
        {"ones_backward", CompiledOpType::ZerosBackward},
        {"fused_residual_rmsnorm_backward", CompiledOpType::FusedResidualRMSNormBackward},
        {"rmsnorm_backward", CompiledOpType::RMSNormBackward},
        {"layernorm_backward", CompiledOpType::LayerNormBackward},
        {"embedding_backward", CompiledOpType::EmbeddingBackward},
        {"cross_entropy_backward", CompiledOpType::CrossEntropyLossBackward},
        {"fused_lm_head_loss_backward", CompiledOpType::FusedLMHeadLossBackward},
        // MoE backward operations
        {"moe_softmax_backward", CompiledOpType::MoESoftmaxBackward},
        {"moe_sigmoid_backward", CompiledOpType::MoESigmoidBackward},
        {"moe_topk_backward", CompiledOpType::MoETopKBackward},
        {"moe_permute_backward", CompiledOpType::MoEPermuteBackward},
        {"moe_grouped_gemm_backward", CompiledOpType::MoEGroupedGemmBackward},
        {"moe_grouped_gemm_gate_up_backward", CompiledOpType::MoEGroupedGemmGateUpBackward},
        {"moe_grouped_gemm_down_backward", CompiledOpType::MoEGroupedGemmDownBackward},
        {"moe_unpermute_backward", CompiledOpType::MoEUnpermuteBackward},
        {"moe_expert_bias_add_backward", CompiledOpType::MoEExpertBiasAddBackward},
        // Expert Parallelism backward operations
        {"ep_dispatch_backward", CompiledOpType::EpDispatchBackward},
        {"ep_combine_backward", CompiledOpType::EpCombineBackward},
        // Mamba/SSM forward operations
        {"mamba_split_proj", CompiledOpType::MambaSplitProj},
        {"mamba_conv1d", CompiledOpType::MambaConv1d},
        {"mamba_split_conv_out", CompiledOpType::MambaSplitConvOut},
        {"mamba_ssm_scan", CompiledOpType::MambaSsmScan},
        {"mamba_gated_rmsnorm", CompiledOpType::MambaGatedRMSNorm},
        {"mamba_out_proj", CompiledOpType::MambaOutProj},
        // Qwen3.5 gated delta rule forward operations
        {"chunk_gated_delta_rule", CompiledOpType::ChunkGatedDeltaRule},
        {"qwen3_5_decay", CompiledOpType::Qwen3_5Decay},
        {"repeat_interleave_heads", CompiledOpType::RepeatInterleaveHeads},
        // Qwen3.5 gated delta rule backward operations
        {"chunk_gated_delta_rule_backward", CompiledOpType::ChunkGatedDeltaRuleBackward},
        {"qwen3_5_decay_backward", CompiledOpType::Qwen3_5DecayBackward},
        {"repeat_interleave_heads_backward", CompiledOpType::RepeatInterleaveHeadsBackward},
        // Mamba/SSM backward operations
        {"mamba_split_proj_backward", CompiledOpType::MambaSplitProjBackward},
        {"mamba_conv1d_backward", CompiledOpType::MambaConv1dBackward},
        {"mamba_split_conv_out_backward", CompiledOpType::MambaSplitConvOutBackward},
        {"mamba_ssm_scan_backward", CompiledOpType::MambaSsmScanBackward},
        {"mamba_gated_rmsnorm_backward", CompiledOpType::MambaGatedRMSNormBackward},
        {"mamba_out_proj_backward", CompiledOpType::MambaOutProjBackward},
    };

    auto it = type_map.find(op_type);
    return it != type_map.end() ? it->second : CompiledOpType::Unknown;
}

// ============================================================================
// GraphCompiler implementation
// ============================================================================

GraphCompiler::GraphCompiler(const Module& module,
                             const modules::ModelConfig& config,
                             const RuntimeOptions& options,
                             DslParamStore& weights,
                             DslGradStore& grads)
    : mModule(module),
      mConfig(config),
      mOptions(options),
      mWeights(weights),
      mGrads(grads) {
    // Initialize slot registry from DSL layout (no built-in fallback - all slots must be
    // explicitly declared in Python DSL)
    if (mModule.activation_layout.has_value()) {
        mSlotRegistry.init_from_layout(*mModule.activation_layout);
    }
    // If no layout, registry remains empty - all tensors will use Mapped slot

    // Enable shape debug output via SUROGATE_DEBUG_SHAPES=1
    if (const char* env = std::getenv("SUROGATE_DEBUG_SHAPES")) {
        mDebugShapes = (std::string(env) == "1");
    }

    // Build per-layer dimensions from IR param shapes. Detects hybrid models
    // (different block types with different head_size/QKV/MLP dims) by checking
    // if any block params have varying shapes.
    if (mModule.forward.has_value()) {
        const auto& graph = mModule.forward.value();
        const int num_layers = config.NumLayers;
        const long hq = config.NumQueryHeads;
        const long default_hkv = config.NumKeyValHeads;
        const long default_hs = config.head_size();
        const long default_dff = config.IntermediateSize;

        mPerLayerDims.resize(static_cast<std::size_t>(num_layers));
        for (int i = 0; i < num_layers; ++i) {
            auto& d = mPerLayerDims[static_cast<std::size_t>(i)];
            d.head_size = default_hs;
            d.qkv_channels = default_hs * (hq + 2 * default_hkv);
            d.attn_dim = hq * default_hs;
            d.intermediate = default_dff;
            d.mlp_up = 2 * default_dff;
        }
        for (const auto& [name, info] : graph.params) {
            int layer_idx = -1;
            std::string field;
            if (!parse_block_param(name, layer_idx, field)) continue;
            if (layer_idx < 0 || layer_idx >= num_layers || info.shape.size() < 2) continue;
            long s0 = (info.shape[0].kind == DimKind::Concrete) ? info.shape[0].value : 0;
            long s1 = (info.shape[1].kind == DimKind::Concrete) ? info.shape[1].value : 0;
            if (s0 == 0 || s1 == 0) continue;
            auto& d = mPerLayerDims[static_cast<std::size_t>(layer_idx)];
            if (field == "qkv_weight") {
                d.qkv_channels = s0;
                long total_heads = hq + 2 * default_hkv;
                if (total_heads > 0) d.head_size = s0 / total_heads;
                d.attn_dim = hq * d.head_size;
            } else if (field == "self_attn_q_weight") {
                d.qkv_channels = s0;
                if (hq > 0) d.head_size = s0 / hq;
                d.attn_dim = s0;
            } else if (field == "out_weight") {
                d.attn_dim = s1;
                if (hq > 0) d.head_size = s1 / hq;
            } else if (field == "mlp_down_weight") {
                d.intermediate = s1;
                d.mlp_up = s1;
            } else if (field == "mlp_gate_weight") {
                d.intermediate = s0;
                d.mlp_up = s0;
            }
        }
        // Detect hybrid blocks: check if per-layer dims actually differ
        for (std::size_t pi = 1; pi < mPerLayerDims.size(); ++pi) {
            if (mPerLayerDims[pi].head_size != mPerLayerDims[0].head_size ||
                mPerLayerDims[pi].qkv_channels != mPerLayerDims[0].qkv_channels ||
                mPerLayerDims[pi].attn_dim != mPerLayerDims[0].attn_dim ||
                mPerLayerDims[pi].intermediate != mPerLayerDims[0].intermediate) {
                mHasHybridBlocks = true;
                break;
            }
        }
        if (!mHasHybridBlocks) {
            // All layers have the same dims — no need for per-layer tracking
            mPerLayerDims.clear();
        }
    }
}

ShapeEnv GraphCompiler::make_layer_env(int layer_idx) const {
    ShapeEnv env = mShapeEnv;  // Start from global env
    if (layer_idx >= 0 && static_cast<std::size_t>(layer_idx) < mPerLayerDims.size()) {
        const auto& d = mPerLayerDims[static_cast<std::size_t>(layer_idx)];
        env.values["D"] = d.head_size;
        env.values["QKV"] = d.qkv_channels;
        env.values["AttnDim"] = d.attn_dim;
        env.values["M"] = d.intermediate;
        env.values["MUp"] = d.mlp_up;
    }
    return env;
}

// Rewrite a shape computed with global dims to per-layer dims. The shape
// fallbacks (resolve_tensor_shape / infer_known_tensor_shape) use
// `mConfig.head_size()` / `mConfig.qkv_channels()` / etc. directly — wrong
// for hybrid block types whose attn_dim / intermediate / mlp_up differ per
// layer. Call this after any fallback-produced shape for a block-scoped
// tensor whose field name maps to one of the per-layer-varying dims.
// Gemma4 needs this for every full-attention block when the default is
// sliding: `blocks[L].att` / `att_flat` / `qkv*` / `q*` / `mlp_up*` /
// `swiglu*` would otherwise end up sized to the sliding defaults.
void GraphCompiler::apply_per_layer_dim_override(std::vector<long>& shape,
                                                 const std::string& base_field,
                                                 int layer_idx) const {
    if (shape.empty()) return;
    if (layer_idx < 0 || static_cast<std::size_t>(layer_idx) >= mPerLayerDims.size()) return;
    const auto& pld = mPerLayerDims[static_cast<std::size_t>(layer_idx)];
    const long B = mB;
    const long T = mT;
    // QKV family — total channels across Q+K+V.
    if (base_field == "qkv" || base_field == "qkv_rope" || base_field == "qkv_norm" || base_field == "qkv_normed") {
        shape = {B, T, pld.qkv_channels};
        return;
    }
    if (base_field == "qkv_flat" || base_field == "qkv_biased" || base_field == "qkv_normed_flat" ||
        base_field == "qkv_norm_flat") {
        shape = {B * T, pld.qkv_channels};
        return;
    }
    // Q-only (shared-KV attention variants).
    if (base_field == "q_flat" || base_field == "q_rn_flat" || base_field == "q_normed_flat" ||
        base_field == "q_roped" || base_field == "q_normed") {
        // Flat form is [B*T, attn_dim]; non-flat is [B, T, attn_dim]. Use
        // shape.size() to decide — view ops may land here with either rank.
        if (shape.size() == 2) {
            shape = {B * T, pld.attn_dim};
        } else {
            shape = {B, T, pld.attn_dim};
        }
        return;
    }
    // Attention output (post-softmax, before projection).
    if (base_field == "att") {
        shape = {B, T, pld.attn_dim};
        return;
    }
    if (base_field == "att_flat") {
        shape = {B * T, pld.attn_dim};
        return;
    }
    // MLP intermediate.
    if (base_field == "mlp_up") {
        shape = {B, T, pld.mlp_up};
        return;
    }
    if (base_field == "mlp_up_flat") {
        shape = {B * T, pld.mlp_up};
        return;
    }
    if (base_field == "swiglu") {
        shape = {B, T, pld.intermediate};
        return;
    }
    if (base_field == "swiglu_flat") {
        shape = {B * T, pld.intermediate};
        return;
    }
}

void GraphCompiler::update_dimensions(long B, long T) {
    mB = B;
    mT = T;

    // Use make_shape_env + augment_shape_env to get the same symbols
    // as the non-compiled execution path. This ensures DSL IR symbol names
    // (e.g., d_model, hidden_size, num_query_heads) are available.
    mShapeEnv = make_shape_env(mModule, B, T);
    augment_shape_env(mShapeEnv, mModule.config);

    // Also ensure standard short symbols from ModelConfig are present
    // (in case DSL IR uses the canonical short names)
    mShapeEnv.values["C"] = mConfig.HiddenSize;
    mShapeEnv.values["D"] = mConfig.head_size();
    const long moe_m = (mConfig.MoeIntermediateSize > 0) ? mConfig.MoeIntermediateSize : mConfig.IntermediateSize;
    const long up_factor = mConfig.mlp_up_factor();
    mShapeEnv.values["M"] = moe_m;
    mShapeEnv.values["MUp"] = up_factor * moe_m;
    mShapeEnv.values["V"] = mConfig.VocabSize;
    mShapeEnv.values["Hq"] = mConfig.NumQueryHeads;
    mShapeEnv.values["Hkv"] = mConfig.NumKeyValHeads;
    mShapeEnv.values["QKV"] = mConfig.qkv_channels();
    mShapeEnv.values["AttnDim"] = mConfig.NumQueryHeads * mConfig.head_size();

    // MoE dimensions
    if (mConfig.NumExperts > 0) {
        mShapeEnv.values["E"] = mConfig.NumExperts;
    }
    if (mConfig.NumExpertsPerTok > 0) {
        mShapeEnv.values["K"] = mConfig.NumExpertsPerTok;
    }
    // Shared expert intermediate size (default to regular intermediate size)
    if (mConfig.moe_config.has_value() && mConfig.moe_config->shared_expert_size > 0) {
        mShapeEnv.values["SharedM"] = mConfig.moe_config->shared_expert_size;
        mShapeEnv.values["SharedMUp"] = up_factor * mConfig.moe_config->shared_expert_size;
    } else {
        mShapeEnv.values["SharedM"] = mConfig.IntermediateSize;
        mShapeEnv.values["SharedMUp"] = up_factor * mConfig.IntermediateSize;
    }
}

CompiledOpType GraphCompiler::classify_op(const std::string& op_type) const {
    return op_type_from_string(op_type);
}

TensorRef
GraphCompiler::resolve_tensor_ref(const std::string& name, bool is_output, const Operation& op, const ShapeEnv& env) {
    TensorRef ref;
    ref.name = name;
    // Pre-compute gradient flag at compile time to avoid runtime string prefix checks.
    ref.is_gradient = starts_with(name, "d_");

    // Check for saved tensor prefix
    std::string effective_name = name;
    if (starts_with(name, kSavedPrefix)) {
        const std::string stripped = std::string(name.substr(kSavedPrefix.size()));
        ref.slot = TensorSlot::Saved;
        ref.name = stripped;
        // Populate shape/dtype from DSL slot registry when available.
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(stripped, layer_idx, field)) {
            ref.layer_idx = layer_idx;
            const std::string base_field = strip_ssa_suffix(field);
            if (auto slot_entry = mSlotRegistry.lookup(base_field)) {
                if (!slot_entry->shape.empty()) {
                    ref.shape = resolve_shape(slot_entry->shape, mShapeEnv);
                }
                if (slot_entry->dtype.has_value()) {
                    ref.dtype = *slot_entry->dtype;
                }
            }
        } else if (auto slot_entry = mSlotRegistry.lookup(strip_ssa_suffix(stripped))) {
            if (!slot_entry->shape.empty()) {
                ref.shape = resolve_shape(slot_entry->shape, env);
            }
            if (slot_entry->dtype.has_value()) {
                ref.dtype = *slot_entry->dtype;
            }
        }
        // Override with infer_known_tensor_shape when available.
        {
            std::vector<long> known_shape;
            if (infer_known_tensor_shape(stripped, mConfig, mB, mT, known_shape)) {
                ref.shape = known_shape;
            }
        }
        if (ref.shape.empty()) {
            auto it = mExtraShapes.find(ref.name);
            if (it != mExtraShapes.end()) {
                ref.shape = it->second;
            }
        }
        // Hybrid per-layer override for saved refs. Both `resolve_shape`
        // (with the global mShapeEnv) and `infer_known_tensor_shape`
        // (hard-coded to global dims) produce the wrong size for block
        // tids whose per-layer head_size / intermediate differs from the
        // default. Gemma4's `saved.blocks[34].att_flat` (final layer
        // forced to full_attention) would otherwise come out Hq*256 = 2048
        // instead of the correct Hq*512 = 4096.
        int parsed_layer_idx = -1;
        std::string parsed_field;
        if (parse_block_param(stripped, parsed_layer_idx, parsed_field)) {
            apply_per_layer_dim_override(ref.shape, strip_ssa_suffix(parsed_field), parsed_layer_idx);
        }
        ref.tensor_id = assign_tensor_id(ref.name);
        return ref;
    }

    // Check for block-indexed tensors
    int layer_idx = -1;
    std::string field;
    if (parse_block_param(effective_name, layer_idx, field)) {
        ref.layer_idx = layer_idx;

        // Strip SSA-style numeric suffix (e.g., "qkv_rope_7" -> "qkv_rope")
        // The DSL IR generates unique tensor names with suffixes like _0, _7, _10, etc.
        const std::string base_field = strip_ssa_suffix(field);

        // Map field to slot using the registry (supports both built-in and DSL-defined slots)
        if (auto slot_entry = mSlotRegistry.lookup(base_field)) {
            ref.slot = slot_entry->slot;

            // Handle global slots that appear with block indices (e.g., rope_freqs)
            if (slot_entry->scope == ActivationScope::Global) {
                ref.layer_idx = -1;  // Global, not layer-indexed
                ref.tensor_id = assign_tensor_id(ref.name);
                return ref;
            }

            // Use shape from DSL, resolved with per-layer env for hybrid models
            if (!slot_entry->shape.empty()) {
                ref.shape = resolve_shape(slot_entry->shape, env);
                // For hybrid models, the slot has concrete shapes from the first
                // block type. Override with per-layer dims when they differ.
                if (layer_idx >= 0 && static_cast<std::size_t>(layer_idx) < mPerLayerDims.size() &&
                    !ref.shape.empty()) {
                    const auto& pld = mPerLayerDims[static_cast<std::size_t>(layer_idx)];
                    const long B = mB;
                    const long T = mT;
                    // Map field to per-layer dimension
                    if (base_field == "qkv" || base_field == "qkv_rope") {
                        ref.shape = {B, T, pld.qkv_channels};
                    } else if (base_field == "qkv_flat" || base_field == "qkv_biased") {
                        ref.shape = {B * T, pld.qkv_channels};
                    } else if (base_field == "att") {
                        ref.shape = {B, T, pld.attn_dim};
                    } else if (base_field == "att_flat") {
                        ref.shape = {B * T, pld.attn_dim};
                    } else if (base_field == "lse") {
                        ref.shape = {B, mConfig.NumQueryHeads, T};
                    } else if (base_field == "mlp_up") {
                        ref.shape = {B, T, pld.mlp_up};
                    } else if (base_field == "mlp_up_flat") {
                        ref.shape = {B * T, pld.mlp_up};
                    } else if (base_field == "swiglu") {
                        ref.shape = {B, T, pld.intermediate};
                    } else if (base_field == "swiglu_flat") {
                        ref.shape = {B * T, pld.intermediate};
                    }
                }
            }
            // Override with extra shapes — but NOT for per-layer-overridden tensors
            // (the per-layer dims are more authoritative than view-inferred shapes).
            if (!mPerLayerDims.empty() && layer_idx >= 0) {
                // Per-layer dims already applied — skip mExtraShapes
            } else if (auto it = mExtraShapes.find(ref.name); it != mExtraShapes.end()) {
                ref.shape = it->second;
            }
        } else if (mWeights.has(effective_name)) {
            // Block-indexed weight (e.g., blocks[0].ln1_weight)
            ref.slot = TensorSlot::Parameter;
            const auto& tmpl = mWeights.template_tensor(effective_name);
            if (tmpl.Rank > 0) {
                ref.shape.assign(tmpl.Sizes.begin(), tmpl.Sizes.begin() + tmpl.Rank);
            }
            ref.tensor_id = assign_tensor_id(ref.name);
            return ref;
        } else {
            ref.slot = TensorSlot::Mapped;
        }

        // For block-indexed Mapped tensors without a slot registry shape,
        // try inferred shapes from validate_operation_shapes and pre-computed
        // view shapes. Without this, HybridStackedBlocks intermediate tensors
        // (e.g., shared_up_out, shared_act) would have empty shapes.
        if (ref.shape.empty()) {
            std::vector<long> resolved;
            if (resolve_tensor_shape(ref.name, resolved)) {
                ref.shape = std::move(resolved);
            } else {
                auto it = mExtraShapes.find(ref.name);
                if (it != mExtraShapes.end()) {
                    ref.shape = it->second;
                }
            }
        }

        // Final per-layer override for hybrid models. `resolve_tensor_shape`
        // and its `infer_known_tensor_shape` fallback use the GLOBAL
        // head_size / intermediate / mlp_up — wrong for hybrid blocks whose
        // dims differ per layer (Gemma4's sliding vs. full_attention, last
        // layer forced to full). The earlier per-layer override block at
        // lines 648-672 only runs when `slot_entry->shape` is non-empty,
        // which skips `att` / `att_flat` / `swiglu` / `qkv_norm*` since the
        // slot registry stores slot-id only (no shape). Without this
        // fallback, the backward graph's recompute scratch for a
        // full-attention block comes out sliding-sized.
        apply_per_layer_dim_override(ref.shape, base_field, layer_idx);

        ref.tensor_id = assign_tensor_id(ref.name);
        return ref;
    }

    // Check for gradient tensors
    if (starts_with(name, "d_")) {
        const std::string base = name.substr(2);
        if (parse_block_param(base, layer_idx, field)) {
            ref.layer_idx = layer_idx;

            // For hybrid models (per-layer dims differ), resolve symbolic slot
            // shapes against the LAYER env, not the global env, so gradient
            // slot shapes match the forward activation shapes at this layer.
            const ShapeEnv layer_env = (layer_idx >= 0 && mHasHybridBlocks) ? make_layer_env(layer_idx) : mShapeEnv;

            const std::string base_field = strip_ssa_suffix(field);

            // Look up gradient slot using "d_<field>" name (e.g., "d_ln1", "d_qkv")
            const std::string grad_name = "d_" + base_field;
            if (auto slot_entry = mSlotRegistry.lookup(grad_name)) {
                ref.slot = slot_entry->slot;
                if (!slot_entry->shape.empty()) {
                    ref.shape = resolve_shape(slot_entry->shape, layer_env);
                }
            } else {
                if (auto act_entry = mSlotRegistry.lookup(base_field)) {
                    if (!act_entry->shape.empty()) {
                        ref.shape = resolve_shape(act_entry->shape, layer_env);
                    }
                }
                ref.slot = TensorSlot::Mapped;
            }

            // Override with infer_known_tensor_shape when available.
            // The slot registry may return a parent slot's shape for aliases
            // (e.g., "mlp_down_flat" is an alias of "mlp_down" with shape (B,T,C)),
            // but _flat tensors need 2D shape (B*T,C). infer_known_tensor_shape
            // correctly distinguishes _flat vs non-flat shapes.
            {
                std::vector<long> known_shape;
                if (infer_known_tensor_shape(base, mConfig, mB, mT, known_shape)) {
                    ref.shape = known_shape;
                }
            }

            // For hybrid models, apply per-layer dim overrides to match the
            // forward activation shapes. Without this, gradient slots for
            // hybrid-dim fields end up sized with global/default dims —
            // downstream view_backward will claim more elements than the
            // underlying allocation holds.
            apply_per_layer_dim_override(ref.shape, base_field, layer_idx);

            if (ref.shape.empty()) {
                auto it = mExtraShapes.find(base);
                if (it != mExtraShapes.end()) {
                    ref.shape = it->second;
                } else {
                    // Fall back to forward tensor shapes inferred during forward
                    // compile. Requires reset_tid_namespace() to have preserved
                    // mTensorShapes across the forward+backward pair. This lets
                    // gradient refs like d_blocks[N].v_normed_2d (RMSNorm
                    // output) inherit their forward activation's shape.
                    auto it2 = mTensorShapes.find(base);
                    if (it2 != mTensorShapes.end() && !it2->second.dims.empty()) {
                        ref.shape = it2->second.dims;
                    }
                }
            }
            ref.tensor_id = assign_tensor_id(ref.name);
            return ref;
        }
    }

    // Check for global tensors using registry (supports built-in and DSL-defined slots)
    if (auto slot_entry = mSlotRegistry.lookup(name)) {
        ref.slot = slot_entry->slot;
        // Apply dtype override from registry if specified
        if (slot_entry->dtype.has_value()) {
            ref.dtype = *slot_entry->dtype;
        }
    } else if (global_role_for_name(name) == GlobalRole::RopeFreqs) {
        // Rope frequencies — recognized via the centralized GlobalRole table
        // so qualified names like "blocks[0].rope_freqs" also resolve here.
        ref.slot = TensorSlot::FreqCis;
    } else if (mWeights.has(name)) {
        ref.slot = TensorSlot::Parameter;
        const auto& tmpl = mWeights.template_tensor(name);
        if (tmpl.Rank > 0) {
            ref.shape.assign(tmpl.Sizes.begin(), tmpl.Sizes.begin() + tmpl.Rank);
        }
    } else {
        ref.slot = TensorSlot::Mapped;
    }

    if (ref.shape.empty()) {
        std::vector<long> resolved;
        if (resolve_tensor_shape(ref.name, resolved)) {
            ref.shape = std::move(resolved);
        } else {
            auto it = mExtraShapes.find(ref.name);
            if (it != mExtraShapes.end()) {
                ref.shape = it->second;
            }
        }
    }
    // For global gradient names (d_<X>) whose forward counterpart <X> has a
    // known shape, inherit it. Handles non-block-prefixed gradients like
    // d_scale_4, d_pli_proj_scaled, and autodiff-accumulated names like
    // d_scale_8_from_1468 (strip "_from_N"/"_accum_N" to get base name).
    // Without this they fall through to the {B,T,C} default.
    if (ref.shape.empty() && starts_with(ref.name, "d_")) {
        std::string fwd_name = ref.name.substr(2);
        // Strip autodiff accumulation suffixes.
        for (const char* pat : {"_from_", "_accum_"}) {
            auto pos = fwd_name.find(pat);
            if (pos == std::string::npos) continue;
            std::size_t after_pos = pos + std::strlen(pat);
            bool all_digits = after_pos < fwd_name.size();
            for (std::size_t i = after_pos; i < fwd_name.size(); ++i) {
                if (!std::isdigit(static_cast<unsigned char>(fwd_name[i]))) {
                    all_digits = false;
                    break;
                }
            }
            if (all_digits) {
                fwd_name = fwd_name.substr(0, pos);
                break;
            }
        }
        std::vector<long> resolved;
        if (resolve_tensor_shape(fwd_name, resolved)) {
            ref.shape = std::move(resolved);
        } else {
            auto it = mExtraShapes.find(fwd_name);
            if (it != mExtraShapes.end()) {
                ref.shape = it->second;
            }
        }
    }
    if (auto it = mTensorDtypes.find(ref.name); it != mTensorDtypes.end()) {
        ref.dtype = it->second;
    }
    ref.tensor_id = assign_tensor_id(ref.name);
    return ref;
}

CompiledAttrs
GraphCompiler::resolve_attrs(const Operation& op, CompiledOpType type, const ShapeEnv& env, const Graph& graph) {
    CompiledAttrs attrs;

    // Populate LoRA slices from the weight input's declared targets.
    // For matmul-family ops, the weight is inputs[1]; for *Backward forms
    // that take (d_out, x, w, ...) it is inputs[2]. MoE grouped-gemm ops
    // also carry targets on inputs[1] (the stacked expert weights).
    //
    // Each slice is validated against the weight's declared output
    // dimension at IR-load time: a bounds violation here signals a DSL
    // annotation bug (declared offset/size does not fit the weight
    // shape) and must surface before any forward pass silently applies
    // a zero-contribution LoRA.
    auto inject_lora_slices = [&](std::size_t weight_idx) {
        if (op.inputs.size() <= weight_idx) {
            return;
        }
        const std::string& weight_name = op.inputs[weight_idx];
        auto it = graph.params.find(weight_name);
        if (it == graph.params.end() || it->second.lora_targets.empty()) {
            return;
        }
        const auto& weight_info = it->second;

        // Resolve the weight's output dimension when possible. Grouped
        // MoE weights have shape ``[E, out, in]``; others are ``[out,
        // in]``. A shape we cannot resolve here (e.g. a runtime-symbolic
        // dim not in ``env``) falls through to the runtime bounds check
        // in ``apply_lora_slices_forward``.
        auto resolve_out_dim = [&](bool grouped) -> std::optional<long> {
            const auto& shape = weight_info.shape;
            const std::size_t idx = grouped ? 1 : 0;
            if (shape.size() <= idx) return std::nullopt;
            try {
                return resolve_dim(shape[idx], env);
            } catch (const std::exception&) {
                return std::nullopt;
            }
        };

        attrs.lora_slices.reserve(weight_info.lora_targets.size());
        for (const auto& t : weight_info.lora_targets) {
            if (t.offset < 0 || t.size < 0) {
                throw std::runtime_error("graph_compiler: LoRA target '" + t.name + "' on weight '" + weight_name +
                                         "' has negative offset/size (offset=" + std::to_string(t.offset) +
                                         ", size=" + std::to_string(t.size) + ")");
            }
            const auto out_dim = resolve_out_dim(t.grouped);
            if (out_dim.has_value()) {
                const long off = static_cast<long>(t.offset);
                const long sz = static_cast<long>(t.size);
                // size=0 means "from offset to end", so the implied end is out_dim.
                const long end = (sz == 0) ? *out_dim : (off + sz);
                if (off < 0 || end > *out_dim || off > end) {
                    throw std::runtime_error("graph_compiler: LoRA target '" + t.name + "' on weight '" + weight_name +
                                             "' is out of bounds (offset=" + std::to_string(t.offset) + ", size=" +
                                             std::to_string(t.size) + ", weight out_dim=" + std::to_string(*out_dim) +
                                             "). Check the LoRATarget declaration in the DSL module.");
                }
            }
            LoRASlice slice;
            slice.id = modules::lora_target_from_name(t.name);
            // Keep the raw name only for unknown targets (hot path avoids it):
            // unknown ids need the name for dropout-seed hashing and for error
            // messages if the runtime shape check fails.
            if (slice.id == modules::LoRATargetId::Unknown) {
                slice.name = t.name;
            }
            slice.schema_slot = schema_slot_from_weight_name(weight_name);
            slice.offset = t.offset;
            slice.size = t.size;
            slice.grouped = t.grouped;
            attrs.lora_slices.push_back(std::move(slice));
        }
    };

    switch (type) {
        case CompiledOpType::Matmul:
        case CompiledOpType::MatmulBias:
        case CompiledOpType::MatmulSwiGLU:
        case CompiledOpType::MoEGroupedGemm:
        case CompiledOpType::MoEGroupedGemmGateUp:
        case CompiledOpType::MoEGroupedGemmDown: inject_lora_slices(/*weight_idx=*/1); break;
        case CompiledOpType::MatmulBackward:
        case CompiledOpType::MatmulSwiGLUBackward:
        case CompiledOpType::MoEGroupedGemmBackward:
        case CompiledOpType::MoEGroupedGemmGateUpBackward:
        case CompiledOpType::MoEGroupedGemmDownBackward: inject_lora_slices(/*weight_idx=*/2); break;
        default: break;
    }

    // Epsilon for normalization ops
    if (auto* eps_attr = find_attr(op.attrs, "eps")) {
        if (auto v = attr_double(*eps_attr)) {
            attrs.eps = static_cast<float>(*v);
        }
    } else {
        attrs.eps = static_cast<float>(mConfig.RmsNormEps);
    }

    // Transpose mode for matmul ops
    attrs.transpose = parse_transpose(op.attrs);

    // Rotary dimension for RoPE
    if (auto* rd_attr = find_attr(op.attrs, "rotary_dim")) {
        if (auto v = attr_int(*rd_attr)) {
            attrs.rotary_dim = static_cast<int>(*v);
        } else if (auto s = attr_string(*rd_attr)) {
            attrs.rotary_dim = static_cast<int>(resolve_dim(Dim::symbolic(*s), env));
        }
    } else {
        attrs.rotary_dim = mConfig.head_size();
    }

    // Shape attribute (direct shape or shape_like reference). When the
    // shape_like target's shape is known statically (via mExtraShapes or
    // known-tensor inference), bake it into attrs.shape so view_backward's
    // runtime dispatch uses the compile-time value and doesn't consult
    // mTensors[shape_like_tid] at runtime. Runtime consultation would
    // pick up the forward-canonical shape, but consumers expect an
    // infer-path shape on Q3.5's Hq/Hkv attention views.
    if (auto* shape_attr = find_attr(op.attrs, "shape")) {
        attrs.shape = resolve_attr_shape(*shape_attr, env);
    } else if (auto* shape_like_attr = find_attr(op.attrs, "shape_like")) {
        if (auto ref_name = attr_string(*shape_like_attr)) {
            attrs.shape_like = *ref_name;
            std::string effective = attrs.shape_like;
            const std::string saved_prefix = "saved.";
            if (starts_with(effective, saved_prefix)) {
                effective = effective.substr(saved_prefix.size());
            }
            std::vector<long> ref_shape;
            auto it = mExtraShapes.find(effective);
            if (it != mExtraShapes.end()) {
                ref_shape = it->second;
            } else if (!resolve_tensor_shape(effective, ref_shape)) {
                infer_known_tensor_shape(effective, mConfig, mB, mT, ref_shape);
            }
            // Hybrid per-layer override. `resolve_tensor_shape` /
            // `infer_known_tensor_shape` are wired to global dims; for
            // block-scoped tensors whose per-layer attn_dim / intermediate /
            // mlp_up differ from the default, ref_shape is wrong. Gemma4
            // block 34 (final full-attention layer) would otherwise see
            // view_backward's target_shape fall through to the sliding
            // value Hq*256 instead of Hq*512.
            int pl_layer_idx = -1;
            std::string pl_field;
            if (!ref_shape.empty() && parse_block_param(effective, pl_layer_idx, pl_field)) {
                apply_per_layer_dim_override(ref_shape, strip_ssa_suffix(pl_field), pl_layer_idx);
            }
            if (!ref_shape.empty()) {
                attrs.shape = std::move(ref_shape);
            }
        }
    }

    // Common axis attribute for split/concat-like operations.
    if (auto* dim_attr = find_attr(op.attrs, "dim")) {
        if (auto v = attr_int(*dim_attr)) {
            attrs.split_concat_dim = static_cast<int>(*v);
        }
    }

    if (auto* dim0_attr = find_attr(op.attrs, "dim0")) {
        if (auto v = attr_int(*dim0_attr)) {
            attrs.dim0 = static_cast<int>(*v);
        }
    }
    if (auto* dim1_attr = find_attr(op.attrs, "dim1")) {
        if (auto v = attr_int(*dim1_attr)) {
            attrs.dim1 = static_cast<int>(*v);
        }
    }

    // Split sizes for split operation.
    if (type == CompiledOpType::Split) {
        if (auto* split_attr = find_attr(op.attrs, "split_size")) {
            if (auto list = attr_list_int(*split_attr)) {
                attrs.split_sizes = *list;
            } else if (auto v = attr_int(*split_attr)) {
                attrs.split_sizes = {static_cast<long>(*v)};
            }
        } else if (auto* sections_attr = find_attr(op.attrs, "sections")) {
            if (auto list = attr_list_int(*sections_attr)) {
                attrs.split_sizes = *list;
            } else if (auto v = attr_int(*sections_attr)) {
                attrs.split_sizes = {static_cast<long>(*v)};
            }
        }
    }

    // Narrow attributes (start, length along dim).
    if (type == CompiledOpType::Narrow) {
        if (auto* start_attr = find_attr(op.attrs, "start")) {
            if (auto v = attr_int(*start_attr)) {
                attrs.narrow_start = static_cast<int>(*v);
            }
        }
        if (auto* len_attr = find_attr(op.attrs, "length")) {
            if (auto v = attr_int(*len_attr)) {
                attrs.narrow_length = static_cast<int>(*v);
            }
        }
    }

    if (auto* acc_attr = find_attr(op.attrs, "compute_accuracy")) {
        if (auto v = attr_bool(*acc_attr)) {
            attrs.compute_accuracy = *v;
        }
    }

    if (auto* factor_attr = find_attr(op.attrs, "factor")) {
        if (auto v = attr_double(*factor_attr)) {
            attrs.scale_factor = static_cast<float>(*v);
        }
    }

    if (auto* softcap_attr = find_attr(op.attrs, "softcap")) {
        if (auto v = attr_double(*softcap_attr)) {
            attrs.softcap = static_cast<float>(*v);
        }
    }

    // Flash-attention softmax scale override. 0.0f keeps the kernel default
    // (1/sqrt(head_dim)); any other value is used verbatim. Gemma4 sets
    // softmax_scale=1.0 because Q/K-norm already provides the implicit scaling.
    if (auto* sm_scale_attr = find_attr(op.attrs, "softmax_scale")) {
        if (auto v = attr_double(*sm_scale_attr)) {
            attrs.softmax_scale = static_cast<float>(*v);
        }
    }

    if (auto* window_attr = find_attr(op.attrs, "window_size")) {
        if (auto v = attr_int(*window_attr)) {
            attrs.window_size = static_cast<int>(*v);
        }
    }

    if (auto* mrope_attr = find_attr(op.attrs, "mrope_section")) {
        if (auto list = attr_list_int(*mrope_attr)) {
            if (list->size() >= 3) {
                attrs.mrope_section = {static_cast<int>((*list)[0]),
                                       static_cast<int>((*list)[1]),
                                       static_cast<int>((*list)[2])};
            }
        } else if (auto s = attr_string(*mrope_attr)) {
            if (*s == "mrope_section") {
                attrs.mrope_section = mConfig.Rope.mrope_section;
            }
        }
    }

    // Matmul-specific attributes
    if (type == CompiledOpType::Matmul || type == CompiledOpType::MatmulBias) {
        if (op.inputs.size() > 1) {
            int layer_idx = -1;
            auto matmul_op = matmul_op_from_weight(op.inputs[1], layer_idx);
            const bool is_gate_projection = is_mlp_gate_weight(op.inputs[1]);
            attrs.matmul_op = matmul_op;
            attrs.layer_idx = layer_idx;
            attrs.allow_quant = matmul_op.has_value() && allow_quant_layer(mOptions, mConfig, layer_idx);
            if (matmul_op.has_value()) {
                switch (*matmul_op) {
                    case modules::MatmulOp::QKV:
                        set_forward_hook_schema_slot(attrs, modules::ForwardHookPoint::AfterQKVProjection, "qkv");
                        break;
                    case modules::MatmulOp::AttnOut:
                        set_forward_hook_schema_slot(attrs,
                                                     modules::ForwardHookPoint::AfterAttnOutProjection,
                                                     "att_out");
                        break;
                    case modules::MatmulOp::MLPUp:
                        if (!is_gate_projection) {
                            set_forward_hook_schema_slot(attrs,
                                                         modules::ForwardHookPoint::AfterMLPUpProjection,
                                                         "mlp_up");
                        }
                        break;
                    case modules::MatmulOp::MLPDown:
                        set_forward_hook_schema_slot(attrs,
                                                     modules::ForwardHookPoint::AfterMLPDownProjection,
                                                     "mlp_down");
                        break;
                    default: break;
                }
            }
        }
    }

    // MatmulSwiGLU: fused MLP up+gate matmul (forward) still needs layer/op attrs for recipes.
    if (type == CompiledOpType::MatmulSwiGLU) {
        if (op.inputs.size() > 1) {
            int layer_idx = -1;
            auto matmul_op = matmul_op_from_weight(op.inputs[1], layer_idx);
            attrs.matmul_op = matmul_op;
            attrs.layer_idx = layer_idx;
            attrs.allow_quant = matmul_op.has_value() && allow_quant_layer(mOptions, mConfig, layer_idx);
            if (matmul_op.has_value() && *matmul_op == modules::MatmulOp::MLPUp) {
                set_forward_hook_schema_slot(attrs, modules::ForwardHookPoint::AfterMLPUpProjection, "mlp_up");
            }
        }
    }

    // MatmulBackward / MatmulSwiGLUBackward: still derive ``matmul_op``
    // from the weight field name for recipe-dispatch and quantization
    // routing. LoRA backward is slice-driven via
    // ``CompiledAttrs::lora_slices`` populated from the weight's
    // DSL-declared ``LoRATargetIR`` list above, so no backward_hook_point
    // is set.
    if (type == CompiledOpType::MatmulBackward && op.inputs.size() > 2) {
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(op.inputs[2], layer_idx, field)) {
            if (field == "qkv_weight") {
                attrs.matmul_op = modules::MatmulOp::QKV;
            } else if (field == "out_weight") {
                attrs.matmul_op = modules::MatmulOp::AttnOut;
            } else if (field == "mlp_up_weight" || field == "up_weight" || field == "mlp_gate_weight" ||
                       field == "gate_weight") {
                attrs.matmul_op = modules::MatmulOp::MLPUp;
            } else if (field == "mlp_down_weight" || field == "down_weight") {
                attrs.matmul_op = modules::MatmulOp::MLPDown;
            }
            attrs.layer_idx = layer_idx;
            attrs.allow_quant = attrs.matmul_op.has_value() && allow_quant_layer(mOptions, mConfig, layer_idx);
        }
    }

    if (type == CompiledOpType::MatmulSwiGLUBackward && op.inputs.size() > 2) {
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(op.inputs[2], layer_idx, field)) {
            if (field == "mlp_up_weight" || field == "up_weight") {
                attrs.matmul_op = modules::MatmulOp::MLPUp;
            }
            attrs.layer_idx = layer_idx;
            attrs.allow_quant = attrs.matmul_op.has_value() && allow_quant_layer(mOptions, mConfig, layer_idx);
        }
    }

    // MoE-specific attributes
    if (type == CompiledOpType::MoETopK || type == CompiledOpType::MoEPermute || type == CompiledOpType::MoEUnpermute ||
        type == CompiledOpType::MoETopKBackward || type == CompiledOpType::MoEPermuteBackward ||
        type == CompiledOpType::MoEUnpermuteBackward) {
        // top_k attribute
        if (auto* top_k_attr = find_attr(op.attrs, "top_k")) {
            if (auto v = attr_int(*top_k_attr)) {
                attrs.top_k = static_cast<int>(*v);
            }
        } else {
            // Default from model config
            attrs.top_k = static_cast<int>(mConfig.NumExpertsPerTok);
        }

        // normalize_weights attribute
        if (auto* norm_attr = find_attr(op.attrs, "normalize")) {
            if (auto v = attr_bool(*norm_attr)) {
                attrs.normalize_weights = *v;
            }
        }
        if (auto* soft_attr = find_attr(op.attrs, "softmax")) {
            if (auto v = attr_bool(*soft_attr)) {
                attrs.topk_softmax = *v;
            }
        }

        // scaling_factor attribute (e.g. routed_scaling_factor for Nemotron-H)
        if (auto* sf_attr = find_attr(op.attrs, "scaling_factor")) {
            if (auto v = attr_double(*sf_attr)) {
                attrs.scaling_factor = static_cast<float>(*v);
            }
        }
        if (auto* round_attr = find_attr(op.attrs, "topk_rounding_scale")) {
            if (auto v = attr_double(*round_attr)) {
                attrs.topk_rounding_scale = static_cast<float>(*v);
            } else if (auto v_int = attr_int(*round_attr)) {
                attrs.topk_rounding_scale = static_cast<float>(*v_int);
            }
        }
        if (auto* sort_attr = find_attr(op.attrs, "topk_sort_by_index")) {
            if (auto v = attr_bool(*sort_attr)) {
                attrs.topk_sort_by_index = *v;
            } else if (auto v_int = attr_int(*sort_attr)) {
                attrs.topk_sort_by_index = (*v_int != 0);
            }
        }
    }

    if (type == CompiledOpType::MoEGroupedGemmGateUp || type == CompiledOpType::MoEGroupedGemmGateUpBackward) {
        if (auto* interleaved_attr = find_attr(op.attrs, "gate_up_interleaved")) {
            if (auto v = attr_bool(*interleaved_attr)) {
                attrs.gate_up_interleaved = *v;
            } else if (auto v_int = attr_int(*interleaved_attr)) {
                attrs.gate_up_interleaved = (*v_int != 0);
            }
        }
    }

    if (type == CompiledOpType::GptOssMoeAct || type == CompiledOpType::GptOssMoeActBackward) {
        if (auto* alpha_attr = find_attr(op.attrs, "alpha")) {
            if (auto v = attr_double(*alpha_attr)) {
                attrs.gpt_oss_alpha = static_cast<float>(*v);
            }
        }
        if (auto* limit_attr = find_attr(op.attrs, "limit")) {
            if (auto v = attr_double(*limit_attr)) {
                attrs.gpt_oss_limit = static_cast<float>(*v);
            }
        }
    }

    // Expert Parallelism attributes
    if (type == CompiledOpType::EpDispatch || type == CompiledOpType::EpCombine ||
        type == CompiledOpType::EpDispatchBackward || type == CompiledOpType::EpCombineBackward) {
        if (auto* ep_attr = find_attr(op.attrs, "ep_size")) {
            if (auto v = attr_int(*ep_attr)) {
                attrs.ep_size = static_cast<int>(*v);
            }
        }
        if (auto* ne_attr = find_attr(op.attrs, "num_experts")) {
            if (auto v = attr_int(*ne_attr)) {
                attrs.num_experts = static_cast<int>(*v);
            }
        } else {
            attrs.num_experts = static_cast<int>(mConfig.NumExperts);
        }
        if (auto* tk_attr = find_attr(op.attrs, "top_k")) {
            if (auto v = attr_int(*tk_attr)) {
                attrs.top_k = static_cast<int>(*v);
            }
        } else {
            attrs.top_k = static_cast<int>(mConfig.NumExpertsPerTok);
        }
    }

    // Mamba/SSM-specific attributes
    if (type == CompiledOpType::MambaSplitProj || type == CompiledOpType::MambaConv1d ||
        type == CompiledOpType::MambaSplitConvOut || type == CompiledOpType::MambaSsmScan ||
        type == CompiledOpType::MambaGatedRMSNorm || type == CompiledOpType::MambaOutProj ||
        type == CompiledOpType::MambaSplitProjBackward || type == CompiledOpType::MambaConv1dBackward ||
        type == CompiledOpType::MambaSplitConvOutBackward || type == CompiledOpType::MambaSsmScanBackward ||
        type == CompiledOpType::MambaGatedRMSNormBackward || type == CompiledOpType::MambaOutProjBackward) {
        // Mamba dimensions from attributes
        if (auto* attr = find_attr(op.attrs, "num_heads")) {
            if (auto v = attr_int(*attr)) {
                attrs.mamba_num_heads = static_cast<int>(*v);
            }
        }
        if (auto* attr = find_attr(op.attrs, "head_dim")) {
            if (auto v = attr_int(*attr)) {
                attrs.mamba_head_dim = static_cast<int>(*v);
            }
        }
        if (auto* attr = find_attr(op.attrs, "ssm_state_size")) {
            if (auto v = attr_int(*attr)) {
                attrs.ssm_state_size = static_cast<int>(*v);
            }
        }
        if (auto* attr = find_attr(op.attrs, "n_groups")) {
            if (auto v = attr_int(*attr)) {
                attrs.n_groups = static_cast<int>(*v);
            }
        }
        if (auto* attr = find_attr(op.attrs, "conv_kernel")) {
            if (auto v = attr_int(*attr)) {
                attrs.conv_kernel = static_cast<int>(*v);
            }
        }
        if (auto* attr = find_attr(op.attrs, "chunk_size")) {
            if (auto v = attr_int(*attr)) {
                attrs.chunk_size = static_cast<int>(*v);
            }
        }
        if (auto* attr = find_attr(op.attrs, "intermediate_size")) {
            if (auto v = attr_int(*attr)) {
                attrs.intermediate_size = static_cast<int>(*v);
            }
        }
        if (auto* attr = find_attr(op.attrs, "conv_dim")) {
            if (auto v = attr_int(*attr)) {
                attrs.conv_dim = static_cast<int>(*v);
            }
        }
        if (auto* attr = find_attr(op.attrs, "dt_min")) {
            if (auto v = attr_double(*attr)) {
                attrs.dt_min = static_cast<float>(*v);
            }
        }
        if (auto* attr = find_attr(op.attrs, "dt_max")) {
            if (auto v = attr_double(*attr)) {
                attrs.dt_max = static_cast<float>(*v);
            }
        }
        if (auto* attr = find_attr(op.attrs, "dt_softplus")) {
            if (auto v = attr_bool(*attr)) {
                attrs.dt_softplus = *v;
            }
        }
        if (auto* attr = find_attr(op.attrs, "use_conv_bias")) {
            if (auto v = attr_bool(*attr)) {
                attrs.use_conv_bias = *v;
            }
        }
        if (auto* attr = find_attr(op.attrs, "activation")) {
            if (auto v = attr_string(*attr)) {
                attrs.activation = *v;
            }
        }
        if (auto* attr = find_attr(op.attrs, "norm_before_gate")) {
            if (auto v = attr_bool(*attr)) {
                attrs.norm_before_gate = *v;
            } else if (auto v_int = attr_int(*attr)) {
                attrs.norm_before_gate = (*v_int != 0);
            }
        }
        // n_groups for gated rmsnorm (passed directly from graph builder)
        if (auto* attr = find_attr(op.attrs, "n_groups")) {
            if (auto v = attr_int(*attr)) {
                attrs.n_groups = static_cast<int>(*v);
            }
        }
        // Legacy: group_size for gated rmsnorm (compute n_groups from intermediate_size / group_size)
        if (auto* attr = find_attr(op.attrs, "group_size")) {
            if (auto v = attr_int(*attr)) {
                if (attrs.intermediate_size > 0 && *v > 0) {
                    attrs.n_groups = attrs.intermediate_size / static_cast<int>(*v);
                }
            }
        }
    }

    if (type == CompiledOpType::RepeatInterleaveHeads || type == CompiledOpType::RepeatInterleaveHeadsBackward) {
        if (auto* attr = find_attr(op.attrs, "repeats")) {
            if (auto v = attr_int(*attr)) {
                attrs.repeat_factor = static_cast<int>(*v);
            }
        }
    }

    // Qwen3.5 gated delta rule attributes
    if (type == CompiledOpType::ChunkGatedDeltaRule || type == CompiledOpType::ChunkGatedDeltaRuleBackward) {
        if (auto* attr = find_attr(op.attrs, "chunk_size")) {
            if (auto v = attr_int(*attr)) {
                attrs.chunk_size = static_cast<int>(*v);
            }
        }
        if (auto* attr = find_attr(op.attrs, "scale")) {
            if (auto v = attr_double(*attr)) {
                attrs.delta_rule_scale = static_cast<float>(*v);
            } else if (auto v_int = attr_int(*attr)) {
                attrs.delta_rule_scale = static_cast<float>(*v_int);
            }
        }
        if (auto* attr = find_attr(op.attrs, "use_qk_l2norm_in_kernel")) {
            if (auto v = attr_bool(*attr)) {
                attrs.use_qk_l2norm_in_kernel = *v;
            } else if (auto v_int = attr_int(*attr)) {
                attrs.use_qk_l2norm_in_kernel = (*v_int != 0);
            }
        }
    }

    // Pre-resolve tensor IDs for side-channel lookups (avoids runtime string hash)
    if (!attrs.shape_like.empty()) {
        // Strip __saved_ prefix if present for shape_like references
        std::string effective = attrs.shape_like;
        if (starts_with(effective, kSavedPrefix)) {
            effective = effective.substr(kSavedPrefix.size());
        }
        auto it = mTensorIdMap.find(effective);
        if (it != mTensorIdMap.end()) {
            attrs.shape_like_tensor_id = it->second;
        }
    }
    if (mConfig.NumExperts > 0) {
        // MoE ops need pre-resolved IDs for expert_offsets and gather_indices
        if (type == CompiledOpType::MoEGroupedGemm || type == CompiledOpType::MoEGroupedGemmGateUp ||
            type == CompiledOpType::MoEGroupedGemmDown || type == CompiledOpType::MoEGroupedGemmBackward ||
            type == CompiledOpType::MoEGroupedGemmGateUpBackward ||
            type == CompiledOpType::MoEGroupedGemmDownBackward || type == CompiledOpType::MoEExpertBiasAdd ||
            type == CompiledOpType::MoEExpertBiasAddBackward || type == CompiledOpType::MoEPermute ||
            type == CompiledOpType::MoEPermuteBackward) {
            if (auto it = mTensorIdMap.find("moe_expert_offsets"); it != mTensorIdMap.end()) {
                attrs.moe_offsets_tensor_id = it->second;
            }
        }
        if (type == CompiledOpType::MoEPermuteBackward) {
            if (auto it = mTensorIdMap.find("moe_gather_indices"); it != mTensorIdMap.end()) {
                attrs.moe_gather_tensor_id = it->second;
            }
        }
    }

    if (env_flag_enabled("SUROGATE_HOOK_SCHEMA_PARITY")) {
        if (attrs.forward_hook_point.has_value() && attrs.forward_hook_schema_slot.empty()) {
            throw std::runtime_error("graph_compiler: op '" + op.name +
                                     "' has legacy forward hook point without schema slot parity");
        }
        for (const LoRASlice& slice : attrs.lora_slices) {
            if (slice.schema_slot.empty()) {
                throw std::runtime_error("graph_compiler: op '" + op.name +
                                         "' has LoRA slice without schema slot parity");
            }
        }
    }

    return attrs;
}

void GraphCompiler::annotate_layer_boundaries(CompiledGraph& graph) {
    graph.layer_start_indices.resize(mConfig.NumLayers, SIZE_MAX);
    graph.layer_end_indices.resize(mConfig.NumLayers, SIZE_MAX);

    int current_layer = -1;
    std::size_t layer_start = 0;

    auto is_grad_ref = [](const TensorRef& ref) -> bool {
        if (!ref.name.empty() && ref.name.size() > 2 && ref.name[0] == 'd' && ref.name[1] == '_') {
            return true;
        }
        switch (ref.slot) {
            case TensorSlot::BlockDLN1:
            case TensorSlot::BlockDQKV:
            case TensorSlot::BlockDAtt:
            case TensorSlot::BlockDSwiGLU:
            case TensorSlot::BlockDMLPUp:
            case TensorSlot::BlockDMLPDown:
            case TensorSlot::BlockDHOut:
            case TensorSlot::BlockDLN2:
            case TensorSlot::BlockDResAtt:
            case TensorSlot::BlockDResFFN:
            case TensorSlot::DLoss: return true;
            default: return false;
        }
    };

    auto ref_layer_idx = [&](const TensorRef& ref) -> int {
        if (ref.layer_idx >= 0) {
            return ref.layer_idx;
        }
        if (ref.name.empty()) {
            return -1;
        }
        std::string_view name = ref.name;
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            return layer_idx;
        }
        return -1;
    };

    auto ref_layer_idx_any = [&](const TensorRef& ref) -> int {
        if (ref.layer_idx >= 0) {
            return ref.layer_idx;
        }
        if (ref.name.empty()) {
            return -1;
        }
        std::string_view name = ref.name;
        if (name.rfind("d_", 0) == 0) {
            name.remove_prefix(2);
        }
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            return layer_idx;
        }
        return -1;
    };

    for (std::size_t i = 0; i < graph.ops.size(); ++i) {
        auto& op = graph.ops[i];

        // Check inputs/outputs for layer index. Use the highest layer index found,
        // since some ops (e.g., LN1 fused residual) consume previous-layer tensors
        // but are parameterized by the current layer's weights.
        int detected_layer = -1;
        for (const auto& ref : op.inputs) {
            if (is_grad_ref(ref)) {
                continue;
            }
            const int layer_idx = ref_layer_idx(ref);
            if (layer_idx >= 0) {
                detected_layer = std::max(detected_layer, layer_idx);
            }
        }
        for (const auto& ref : op.outputs) {
            if (is_grad_ref(ref)) {
                continue;
            }
            const int layer_idx = ref_layer_idx(ref);
            if (layer_idx >= 0) {
                detected_layer = std::max(detected_layer, layer_idx);
            }
        }
        if (detected_layer < 0) {
            for (const auto& ref : op.inputs) {
                const int layer_idx = ref_layer_idx_any(ref);
                if (layer_idx >= 0) {
                    detected_layer = std::max(detected_layer, layer_idx);
                }
            }
            for (const auto& ref : op.outputs) {
                const int layer_idx = ref_layer_idx_any(ref);
                if (layer_idx >= 0) {
                    detected_layer = std::max(detected_layer, layer_idx);
                }
            }
        }
        if (op.attrs.layer_idx >= 0) {
            detected_layer = std::max(detected_layer, op.attrs.layer_idx);
        }

        if (detected_layer >= 0 && detected_layer != current_layer) {
            // End previous layer
            if (current_layer >= 0 && current_layer < static_cast<int>(mConfig.NumLayers)) {
                graph.layer_end_indices[current_layer] = i;
                graph.ops[i - 1].layer_end = current_layer;
            }

            // Start new layer
            current_layer = detected_layer;
            if (current_layer < static_cast<int>(mConfig.NumLayers)) {
                graph.layer_start_indices[current_layer] = i;
                op.layer_start = current_layer;
            }
        }
    }

    // End final layer
    if (current_layer >= 0 && current_layer < static_cast<int>(mConfig.NumLayers)) {
        graph.layer_end_indices[current_layer] = graph.ops.size();
        if (!graph.ops.empty()) {
            graph.ops.back().layer_end = current_layer;
        }
    }
}

// ============================================================================
// Phase Tree — build from layer boundaries
// ============================================================================

const char* phase_kind_name(PhaseKind k) {
    switch (k) {
        case PhaseKind::Custom: return "Custom";
        case PhaseKind::FwdBlockSeq: return "FwdBlockSeq";
        case PhaseKind::BwdBlockSeq: return "BwdBlockSeq";
        case PhaseKind::FwdBlock: return "FwdBlock";
        case PhaseKind::BwdBlock: return "BwdBlock";
    }
    return "?";
}

namespace {

void dump_phase_tree_rec(std::ostringstream& os, const PhaseNode& node, int indent) {
    for (int i = 0; i < indent; ++i)
        os << "  ";
    const char* kind_str = phase_kind_name(node.kind);
    os << kind_str;
    if (!node.label.empty() && node.label != kind_str) {
        os << " " << node.label;
    }
    os << " ops=[" << node.op_start << ".." << node.op_end << ") n=" << (node.op_end - node.op_start);
    if (node.block_index >= 0) {
        os << " block=" << node.block_index;
    }
    os << "\n";
    for (const auto& child : node.children) {
        dump_phase_tree_rec(os, child, indent + 1);
    }
}

}  // namespace

std::string dump_phase_tree(const PhaseNode& root) {
    std::ostringstream os;
    dump_phase_tree_rec(os, root, 0);
    return os.str();
}

void GraphCompiler::build_phase_tree(CompiledGraph& graph, bool is_backward) {
    const std::size_t num_layers = graph.layer_start_indices.size();
    const std::size_t num_ops = graph.ops.size();

    struct BlockRange {
        int layer_idx;
        std::size_t start;
        std::size_t end;
    };
    std::vector<BlockRange> blocks;
    blocks.reserve(num_layers);

    // annotate_layer_boundaries uses SIZE_MAX as a "layer not seen" sentinel.
    // For backward, layer_start_indices[L] decreases with L (layer N-1 runs
    // first), so we sort by start-op to get execution order uniformly.
    for (std::size_t L = 0; L < num_layers; ++L) {
        auto s = graph.layer_start_indices[L];
        auto e = graph.layer_end_indices[L];
        if (s != SIZE_MAX && e != SIZE_MAX && s < e && e <= num_ops) {
            blocks.push_back({static_cast<int>(L), s, e});
        }
    }
    std::sort(blocks.begin(), blocks.end(), [](const BlockRange& a, const BlockRange& b) { return a.start < b.start; });

    if (blocks.empty()) {
        graph.phase_tree.reset();
        return;
    }

    const std::size_t first_op = blocks.front().start;
    const std::size_t last_op = blocks.back().end;

    PhaseNode root;
    root.kind = PhaseKind::Custom;
    root.op_start = 0;
    root.op_end = num_ops;
    root.label = is_backward ? "BackwardCompile" : "ForwardCompile";

    if (first_op > 0) {
        PhaseNode prologue;
        prologue.kind = PhaseKind::Custom;
        prologue.op_start = 0;
        prologue.op_end = first_op;
        prologue.label = "Prologue";
        root.children.push_back(std::move(prologue));
    }

    PhaseNode seq;
    seq.kind = is_backward ? PhaseKind::BwdBlockSeq : PhaseKind::FwdBlockSeq;
    seq.op_start = first_op;
    seq.op_end = last_op;
    seq.label = phase_kind_name(seq.kind);
    seq.children.reserve(blocks.size());

    // Grow each block's end to meet the next block's start so every op in
    // [first_op, last_op) belongs to exactly one block (absorbs inter-block
    // glue ops in hybrid architectures).
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        PhaseNode block;
        block.kind = is_backward ? PhaseKind::BwdBlock : PhaseKind::FwdBlock;
        block.block_index = blocks[i].layer_idx;
        block.op_start = blocks[i].start;
        block.op_end = (i + 1 < blocks.size()) ? blocks[i + 1].start : blocks[i].end;
        {
            std::ostringstream lbl;
            lbl << phase_kind_name(block.kind) << "[" << block.block_index << "]";
            block.label = lbl.str();
        }
        seq.children.push_back(std::move(block));
    }

    root.children.push_back(std::move(seq));

    if (last_op < num_ops) {
        PhaseNode epilogue;
        epilogue.kind = PhaseKind::Custom;
        epilogue.op_start = last_op;
        epilogue.op_end = num_ops;
        epilogue.label = "Epilogue";
        root.children.push_back(std::move(epilogue));
    }

    graph.phase_tree = std::move(root);

    if (const char* env = std::getenv("SUROGATE_DEBUG_PHASE_TREE")) {
        if (std::string(env) == "1") {
            std::cerr << "[phase-tree] " << (is_backward ? "backward" : "forward") << " compile (" << graph.name
                      << "):\n"
                      << dump_phase_tree(*graph.phase_tree);
        }
    }
}

// ============================================================================
// Region Derivation
// ============================================================================

const char* region_kind_name(RegionKind k) {
    switch (k) {
        case RegionKind::Unknown: return "Unknown";
        case RegionKind::FwdStack: return "FwdStack";
        case RegionKind::BwdStack: return "BwdStack";
        case RegionKind::SaveForBwd: return "SaveForBwd";
        case RegionKind::Accumulator: return "Accumulator";
        case RegionKind::Persistent: return "Persistent";
        case RegionKind::PersistentActivation: return "PersistentActivation";
        case RegionKind::ModelScopePersistent: return "ModelScopePersistent";
        case RegionKind::Recomputed: return "Recomputed";
        case RegionKind::GatheredWeight: return "GatheredWeight";
        case RegionKind::BwdCrossLayer: return "BwdCrossLayer";
    }
    return "?";
}

void GraphCompiler::derive_regions(CompiledGraph& graph, bool is_backward) {
    // Map TensorKind (populated by classify_tensors) to a region, consulting
    // block_layer_idx to distinguish block-scoped from global tensors. M2
    // intentionally stops at this coarse classification; SaveForBwd detection
    // (save-list integration) and Recomputed (recompute-plan integration)
    // arrive in later milestones.
    for (auto& meta : graph.tensor_meta) {
        switch (meta.kind) {
            case TensorKind::ForwardParam: meta.region = RegionKind::Persistent; break;
            case TensorKind::ForwardActivation:
                meta.region = (meta.block_layer_idx >= 0) ? RegionKind::FwdStack : RegionKind::PersistentActivation;
                break;
            case TensorKind::ActivationGrad:
                meta.region = (meta.block_layer_idx >= 0) ? RegionKind::BwdStack : RegionKind::PersistentActivation;
                break;
            case TensorKind::ParamGrad:
            case TensorKind::AccumTemp: meta.region = RegionKind::Accumulator; break;
            case TensorKind::LossInput: meta.region = RegionKind::PersistentActivation; break;
            case TensorKind::Scratch:
                // Pessimistic default for un-typed intermediates (views,
                // zeros, constants). Correct when the scratch dies within a
                // block; M3 will re-verify against last-use spans.
                meta.region = is_backward ? RegionKind::BwdStack : RegionKind::FwdStack;
                break;
            case TensorKind::Unknown:
                // Includes cross-graph references (e.g., forward activations
                // read by backward). Left as Unknown for now; save-list
                // integration in a later milestone will promote these to
                // SaveForBwd.
                break;
        }
    }

    if (const char* env = std::getenv("SUROGATE_DEBUG_REGIONS")) {
        if (std::string(env) == "1") {
            std::array<int, 10> counts{};  // Indexed by RegionKind.
            for (const auto& meta : graph.tensor_meta) {
                counts[static_cast<std::size_t>(meta.region)]++;
            }
            std::cerr << "[regions] " << (is_backward ? "backward" : "forward") << " compile (" << graph.name << "):";
            for (std::size_t i = 0; i < counts.size(); ++i) {
                if (counts[i] > 0) {
                    std::cerr << " " << region_kind_name(static_cast<RegionKind>(i)) << "=" << counts[i];
                }
            }
            std::cerr << "\n";
        }
    }
}

// ============================================================================
// Cross-layer FwdStack read promotion
// ============================================================================

void GraphCompiler::promote_cross_layer_fwd_reads(CompiledGraph& graph) {
    // Scan forward ops for cross-layer reads: an op in layer L_op that
    // references a tensor whose `block_layer_idx == L_src` (L_src != L_op)
    // in a FwdStack region. Without promotion those tids share the same
    // arena offset across layers, so layer L_op's writes can clobber
    // layer L_src's bytes before L_op's later ops read them.
    //
    // Promotes the source tid to SaveForBwd (per-layer persistent storage).
    // The reading op's own tid stays in FwdStack because its bytes are
    // produced and consumed within its own layer.
    const std::size_t num_layers = graph.layer_start_indices.size();
    if (num_layers == 0) return;

    auto op_to_layer = [&](std::size_t op_idx) -> int {
        for (std::size_t L = 0; L < num_layers; ++L) {
            const std::size_t s = graph.layer_start_indices[L];
            const std::size_t e = (L < graph.layer_end_indices.size()) ? graph.layer_end_indices[L] : SIZE_MAX;
            if (s == SIZE_MAX || e == SIZE_MAX) continue;
            if (op_idx >= s && op_idx < e) return static_cast<int>(L);
        }
        return -1;
    };

    int promoted = 0;
    std::vector<std::string> promoted_names;
    for (std::size_t i = 0; i < graph.ops.size(); ++i) {
        const int op_layer = op_to_layer(i);
        if (op_layer < 0) continue;
        for (const auto& ref : graph.ops[i].inputs) {
            if (ref.tensor_id < 0 || static_cast<std::size_t>(ref.tensor_id) >= graph.tensor_meta.size()) continue;
            auto& meta = graph.tensor_meta[static_cast<std::size_t>(ref.tensor_id)];
            if (meta.block_layer_idx < 0) continue;             // global tensor
            if (meta.block_layer_idx == op_layer) continue;     // same-layer read
            if (meta.region != RegionKind::FwdStack) continue;  // already persistent
            meta.region = RegionKind::SaveForBwd;
            ++promoted;
            if (std::getenv("SUROGATE_DEBUG_REGIONS")) {
                promoted_names.emplace_back(graph.name_for_tensor_id(ref.tensor_id));
            }
        }
    }

    // Second pass: flag model-scope tensors consumed by ops in multiple
    // layers as `cross_layer_global`. These are produced outside any block
    // (e.g., Gemma4 per_layer_inputs in the PLI phase) and consumed by
    // every layer via compiler-synthesized narrow/view ops. Without this
    // flag, layer-end pruning in prune_stack_tensors evicts them from
    // mTensors after the first layer finishes, making them unresolvable
    // during backward (and specifically during replay_layer_forward under
    // gradient-checkpointing recompute). The runtime's mSaveMask extension
    // consults this flag to preserve the tid across layer boundaries.
    const std::size_t num_tensors_n = graph.tensor_meta.size();
    std::vector<char> produced_here(num_tensors_n, 0);
    for (const auto& op : graph.ops) {
        for (const auto& ref : op.outputs) {
            if (ref.tensor_id >= 0 && static_cast<std::size_t>(ref.tensor_id) < num_tensors_n) {
                produced_here[ref.tensor_id] = 1;
            }
        }
    }
    std::unordered_map<int, std::unordered_set<int>> consumer_layers;
    for (std::size_t i = 0; i < graph.ops.size(); ++i) {
        const int op_layer = op_to_layer(i);
        if (op_layer < 0) continue;  // model-scope op, not per-layer
        for (const auto& ref : graph.ops[i].inputs) {
            if (ref.tensor_id < 0 || static_cast<std::size_t>(ref.tensor_id) >= num_tensors_n) continue;
            const auto& meta = graph.tensor_meta[static_cast<std::size_t>(ref.tensor_id)];
            if (meta.is_blocks()) continue;           // block-scoped: handled by pass 1
            if (meta.block_layer_idx >= 0) continue;  // cross-layer block read: handled by pass 1
            if (!produced_here[ref.tensor_id]) continue;
            consumer_layers[ref.tensor_id].insert(op_layer);
        }
    }
    int clg_count = 0;
    std::vector<std::string> clg_names_dbg;
    for (const auto& [tid, layers] : consumer_layers) {
        if (layers.size() < 2) continue;
        auto& meta = graph.tensor_meta[static_cast<std::size_t>(tid)];
        if (meta.cross_layer_global) continue;
        meta.cross_layer_global = true;
        ++clg_count;
        if (std::getenv("SUROGATE_DEBUG_REGIONS")) {
            clg_names_dbg.emplace_back(graph.name_for_tensor_id(tid));
        }
    }

    if (const char* env = std::getenv("SUROGATE_DEBUG_REGIONS")) {
        if (std::string(env) == "1") {
            if (promoted > 0) {
                std::cerr << "[regions] cross-layer fwd promotion (" << graph.name << "): " << promoted
                          << " tids promoted FwdStack→SaveForBwd";
                for (const auto& n : promoted_names)
                    std::cerr << " " << n;
                std::cerr << "\n";
            }
            if (clg_count > 0) {
                std::cerr << "[regions] cross-layer-global flagged (" << graph.name << "): " << clg_count
                          << " model-scope tids";
                for (const auto& n : clg_names_dbg)
                    std::cerr << " " << n;
                std::cerr << "\n";
            }
        }
    }
}

// ============================================================================
// Cross-graph SaveForBwd promotion
// ============================================================================

void finalize_save_for_bwd(CompiledGraph& fwd,
                           CompiledGraph& bwd,
                           std::optional<std::unordered_set<std::string>> save_names,
                           bool fwd_per_layer_sections) {
    // Tids are shared across the fwd+bwd pair (reset_tid_namespace contract).
    const int num_tids = std::max(fwd.num_tensors, bwd.num_tensors);
    if (num_tids <= 0) return;

    std::vector<char> produced_in_fwd(num_tids, 0);
    std::vector<char> consumed_in_bwd(num_tids, 0);

    for (const auto& op : fwd.ops) {
        for (const auto& ref : op.outputs) {
            if (ref.tensor_id >= 0 && ref.tensor_id < num_tids) {
                produced_in_fwd[ref.tensor_id] = 1;
            }
        }
    }
    for (const auto& op : bwd.ops) {
        for (const auto& ref : op.inputs) {
            if (ref.tensor_id >= 0 && ref.tensor_id < num_tids) {
                consumed_in_bwd[ref.tensor_id] = 1;
            }
        }
    }

    auto promote = [](CompiledGraph& g, int tid) {
        if (tid < 0 || tid >= g.num_tensors) return;
        auto& meta = g.tensor_meta[static_cast<std::size_t>(tid)];
        // Only block activations are SaveForBwd candidates. Guard rules out
        // globals like TokenIDs that are consumed in bwd but stay Persistent.
        if (!meta.is_blocks()) return;
        if (meta.region == RegionKind::FwdStack || meta.region == RegionKind::BwdStack) {
            meta.region = RegionKind::SaveForBwd;
        }
    };

    // Detect model-scope forward tensors consumed by backward and assign
    // RegionKind::ModelScopePersistent. This surfaces after attempt-3's
    // whack-a-mole approach proved unsafe: preserving mTensors[tid] blocks
    // pruner clears but doesn't stop stack-memory reuse by later
    // temp_allocs — dangling pointer → UAF on backward read. Routing to a
    // dedicated arena with stable compile-time offsets fixes that
    // structurally (no temp_alloc overlap). Gemma4-E2B candidates:
    // pli_proj_rn_flat (rmsnorm-bwd x input), scale_8 (per_layer_inputs —
    // already flagged cross_layer_global by pass 2 of
    // promote_cross_layer_fwd_reads). Both fwd and bwd meta get the region
    // so arena ptr resolution (resolve_tid_in_arena) works from either
    // graph's perspective.
    // Flag model-scope forward tensors consumed by backward as
    // `cross_layer_global`. These live in the PersistentActivation arena
    // by default (derive_regions) but aren't bound at runtime except via
    // the named-buffer path (rebind_non_block_to_persistent_arena covers
    // x0/xF/ln_final/etc). Under force-capture, intermediates like
    // `pli_proj_rn_flat` and `scale_8` end up as Stack temps that get
    // overwritten by later ops → stale when backward reads them. The flag
    // selects a narrow subset for populate_fwd_stack_bindings to bind
    // directly into the PersistentActivation arena, bypassing temp_alloc.
    int msp_promoted = 0;
    for (int tid = 0; tid < num_tids; ++tid) {
        if (tid >= fwd.num_tensors) continue;
        auto& fmeta = fwd.tensor_meta[static_cast<std::size_t>(tid)];
        if (fmeta.is_blocks()) continue;
        if (fmeta.block_layer_idx >= 0) continue;
        if (!produced_in_fwd[tid]) continue;
        if (!consumed_in_bwd[tid]) continue;
        // Must already be a PersistentActivation (model-scope forward
        // activation) — that's where derive_regions puts them.
        if (fmeta.region != RegionKind::PersistentActivation) continue;
        if (fmeta.cross_layer_global) continue;
        fmeta.cross_layer_global = true;
        if (tid < bwd.num_tensors) {
            bwd.tensor_meta[static_cast<std::size_t>(tid)].cross_layer_global = true;
        }
        ++msp_promoted;
    }

    // When save_names is provided, a tid that crosses the fwd→bwd
    // boundary but is NOT in the runtime save list is satisfied via
    // Stack residency, recompute, or forward replay — none need a
    // SaveForBwd slot. Leave those tids in FwdStack/BwdStack. An empty
    // set promotes nothing (valid in recompute modes). Nullopt disables
    // filtering entirely (all fwd∧bwd-crossing block activations
    // promoted, prior behavior). Name lookup uses fwd first (SaveForBwd
    // sources are fwd activations), then bwd.
    const bool filter_by_save_names = save_names.has_value();
    auto in_save_list = [&](int tid) {
        if (!filter_by_save_names) return true;
        const auto& set = *save_names;
        auto name = fwd.name_for_tensor_id(tid);
        if (!name.empty() && set.count(std::string(name))) return true;
        name = bwd.name_for_tensor_id(tid);
        if (!name.empty() && set.count(std::string(name))) return true;
        return false;
    };

    int promoted = 0;
    int skipped_by_save_list = 0;
    // Pass 1: promote fwd→bwd-crossing tids in save_list to SaveForBwd.
    for (int tid = 0; tid < num_tids; ++tid) {
        if (!(produced_in_fwd[tid] && consumed_in_bwd[tid])) continue;
        if (!in_save_list(tid)) {
            ++skipped_by_save_list;
            continue;
        }
        promote(fwd, tid);
        promote(bwd, tid);
        ++promoted;
    }

    // Pass 2: flag retain_through_forward on recompute-replay candidates.
    //
    // Under recompute (empty/subset save list), block-scope FwdStack tids
    // that are NOT promoted to SaveForBwd are regenerated by replay and
    // read by backward — either directly (static bwd graph ref) or
    // indirectly (LoRA backward hooks, per-slot reads).
    // Hook-driven consumers don't appear in the bwd graph's ref list, so
    // `consumed_in_bwd` is a lower bound: fwd→bwd crossers we find there
    // need retention; other fwd-produced block-scope tids MIGHT also be
    // hook-read but enumerating hooks at compile time is brittle. Choose
    // the broader rule — retain every non-promoted block-scope FwdStack
    // tid — so coloring behaves like the legacy Stack bump allocator
    // (no within-block aliasing of forward-produced activations).
    // Narrow the rule to recompute mode: under no-recompute, everything
    // fwd→bwd-crossing is promoted to SaveForBwd, and ephemeral fwd-only
    // tids don't need retention — retention there just bloats the arena.
    //
    // Detection: recompute mode runs finalize_save_for_bwd with an empty
    // save name set (the call site at compile_graphs passes save_name_set
    // unconditionally with fewer entries under recompute). Empty-set is
    // the clean signal; save_names.empty() after the promotion pass.
    int retained_through_fwd = 0;
    const bool recompute_mode = filter_by_save_names && save_names && save_names->empty();
    // Apply retain_through_forward to every block-scope FwdStack tid in two
    // cases: (1) recompute_mode (empty save list + replay regenerates) — the
    // original rule; (2) fwd_per_layer_sections (no-recompute) — per-layer
    // sectioning stops cross-layer clobber, but within-layer coloring still
    // reuses bytes for disjoint-lifetime tids. Under no-recompute,
    // save_tensors runs after forward and captures per-slot arena
    // pointers; if coloring reused those bytes mid-forward, the saved
    // snapshot is the LATER tid's data, not the slot's. retain_through_forward
    // extends live ranges to frame end so coloring cannot reuse.
    if (recompute_mode || fwd_per_layer_sections) {
        for (auto& meta : fwd.tensor_meta) {
            if (!meta.is_blocks()) continue;
            if (meta.region != RegionKind::FwdStack) continue;
            if (!meta.retain_through_forward) {
                meta.retain_through_forward = true;
                ++retained_through_fwd;
            }
        }
    }

    if (const char* env = std::getenv("SUROGATE_DEBUG_REGIONS")) {
        if (std::string(env) == "1") {
            std::cerr << "[regions] SaveForBwd promotion: " << promoted << " tids";
            if (filter_by_save_names) {
                std::cerr << " (skipped " << skipped_by_save_list << " not in save list, "
                          << "retain_through_forward=" << retained_through_fwd << ")";
            }
            std::cerr << "\n";
        }
    }

    // Re-run layout now that regions are finalized. The first pass in
    // compile() baked offsets under stale regions (SaveForBwd candidates sat
    // in FwdStack / BwdStack frames); re-running fixes them. Same path
    // handles retain_through_forward tids — compute_layout consults the
    // flag to extend their LayoutInfo.last_use — and
    // ModelScopePersistent tids newly assigned by the pass above.
    if (promoted > 0 || retained_through_fwd > 0 || msp_promoted > 0) {
        for (auto& m : fwd.tensor_meta)
            m.offset = SIZE_MAX;
        for (auto& m : bwd.tensor_meta)
            m.offset = SIZE_MAX;
        compute_layout(fwd, /*is_backward=*/false, fwd_per_layer_sections);
        compute_layout(bwd, /*is_backward=*/true, fwd_per_layer_sections);
    }
    if (msp_promoted > 0 && std::getenv("SUROGATE_DEBUG_REGIONS")) {
        std::cerr << "[regions] ModelScopePersistent promotion: " << msp_promoted
                  << " tids (model-scope fwd reads from bwd)\n";
    }
}

// ============================================================================
// Instruction Stream — emit from phase tree
// ============================================================================

const char* inst_kind_name(InstKind k) {
    switch (k) {
        case InstKind::PhaseEnter: return "PhaseEnter";
        case InstKind::PhaseExit: return "PhaseExit";
        case InstKind::SegmentDispatch: return "SegmentDispatch";
        case InstKind::PruneByLastUse: return "PruneByLastUse";
        case InstKind::RecomputeBlock: return "RecomputeBlock";
    }
    return "?";
}

namespace {

void emit_phase(std::vector<Instruction>& out, const PhaseNode& node) {
    out.push_back({InstKind::PhaseEnter, node.kind, node.block_index, node.op_start, node.op_end, false});

    // Emit RecomputeBlock inside BwdBlock leaves so the forward-replay
    // dispatch is explicit rather than inlined in PhaseEnter. Always
    // emitted; the interpreter no-ops when mRecomputeEnabled is false.
    if (node.kind == PhaseKind::BwdBlock && node.block_index >= 0) {
        out.push_back({InstKind::RecomputeBlock, node.kind, node.block_index, node.op_start, node.op_end, false});
    }

    if (node.children.empty()) {
        // Leaf phase: the ops in [op_start, op_end) are executed here. For M4
        // we emit a single synthetic graph-captured segment; a later milestone
        // will subdivide per CompiledGraph::layer_segments (split-attention
        // mode around FlashAttention / capture-unsafe ops).
        if (node.op_end > node.op_start) {
            out.push_back({InstKind::SegmentDispatch, node.kind, node.block_index, node.op_start, node.op_end, false});
            out.push_back({InstKind::PruneByLastUse, node.kind, node.block_index, node.op_start, node.op_end, false});
        }
    } else {
        for (const auto& child : node.children) {
            emit_phase(out, child);
        }
    }

    out.push_back({InstKind::PhaseExit, node.kind, node.block_index, node.op_start, node.op_end, false});
}

}  // namespace

std::string dump_instruction_stream(const std::vector<Instruction>& stream) {
    std::ostringstream os;
    int depth = 0;
    for (const auto& inst : stream) {
        if (inst.kind == InstKind::PhaseExit && depth > 0) --depth;
        for (int i = 0; i < depth; ++i)
            os << "  ";
        os << inst_kind_name(inst.kind);
        if (inst.kind == InstKind::PhaseEnter || inst.kind == InstKind::PhaseExit) {
            os << " " << phase_kind_name(inst.phase_kind);
            if (inst.block_index >= 0) os << "[" << inst.block_index << "]";
        } else if (inst.kind == InstKind::RecomputeBlock) {
            os << " block=" << inst.block_index;
        } else {
            os << " ops=[" << inst.op_start << ".." << inst.op_end << ")";
            if (inst.kind == InstKind::SegmentDispatch) {
                os << " mode=" << (inst.eager ? "eager" : "graph");
            }
        }
        os << "\n";
        if (inst.kind == InstKind::PhaseEnter) ++depth;
    }
    return os.str();
}

std::string validate_instruction_stream(const std::vector<Instruction>& stream, std::size_t num_ops) {
    std::ostringstream errs;

    // 1) Phase nesting: stack-tracked; each PhaseExit must match the
    // most-recent PhaseEnter by kind + block_index.
    std::vector<Instruction> nesting;
    for (std::size_t i = 0; i < stream.size(); ++i) {
        const auto& inst = stream[i];
        if (inst.kind == InstKind::PhaseEnter) {
            nesting.push_back(inst);
        } else if (inst.kind == InstKind::PhaseExit) {
            if (nesting.empty()) {
                errs << "  unmatched PhaseExit at index " << i << "\n";
                continue;
            }
            const auto& top = nesting.back();
            if (top.phase_kind != inst.phase_kind || top.block_index != inst.block_index) {
                errs << "  PhaseExit at " << i << " does not match enclosing PhaseEnter "
                     << phase_kind_name(top.phase_kind) << "[" << top.block_index << "]\n";
            }
            nesting.pop_back();
        }
    }
    if (!nesting.empty()) {
        errs << "  " << nesting.size() << " unclosed PhaseEnter(s)\n";
    }

    // 2) SegmentDispatch coverage: exactly one per op.
    std::vector<char> covered(num_ops, 0);
    for (std::size_t i = 0; i < stream.size(); ++i) {
        const auto& inst = stream[i];
        if (inst.kind != InstKind::SegmentDispatch) continue;
        if (inst.op_start > inst.op_end || inst.op_end > num_ops) {
            errs << "  SegmentDispatch at " << i << " has invalid range [" << inst.op_start << ".." << inst.op_end
                 << ")\n";
            continue;
        }
        for (std::size_t op = inst.op_start; op < inst.op_end; ++op) {
            if (covered[op]) {
                errs << "  op " << op << " covered by multiple SegmentDispatches (second at " << i << ")\n";
                break;  // report once per instruction
            }
            covered[op] = 1;
        }
    }
    std::size_t uncovered = 0;
    for (char c : covered) {
        if (!c) ++uncovered;
    }
    if (uncovered > 0) {
        errs << "  " << uncovered << " op(s) not covered by any SegmentDispatch\n";
    }

    // 3) PruneByLastUse ranges must be disjoint.
    std::vector<char> pruned(num_ops, 0);
    for (std::size_t i = 0; i < stream.size(); ++i) {
        const auto& inst = stream[i];
        if (inst.kind != InstKind::PruneByLastUse) continue;
        if (inst.op_start > inst.op_end || inst.op_end > num_ops) continue;
        for (std::size_t op = inst.op_start; op < inst.op_end; ++op) {
            if (pruned[op]) {
                errs << "  op " << op << " pruned by multiple PruneByLastUse ranges (second at " << i << ")\n";
                break;
            }
            pruned[op] = 1;
        }
    }

    return errs.str();
}

void GraphCompiler::emit_instruction_stream(CompiledGraph& graph) {
    graph.instruction_stream.clear();
    if (!graph.phase_tree.has_value()) return;
    emit_phase(graph.instruction_stream, *graph.phase_tree);

    if (const char* env = std::getenv("SUROGATE_DEBUG_INSTR_STREAM")) {
        if (std::string(env) == "1") {
            std::cerr << "[instr-stream] (" << graph.name << "):\n"
                      << dump_instruction_stream(graph.instruction_stream);
            const std::string errs = validate_instruction_stream(graph.instruction_stream, graph.ops.size());
            if (errs.empty()) {
                std::cerr << "[instr-stream] validator: OK (" << graph.instruction_stream.size()
                          << " instructions over " << graph.ops.size() << " ops)\n";
            } else {
                std::cerr << "[instr-stream] validator: ERRORS\n" << errs;
            }
        }
    }
}

// ============================================================================
// Shadow-mode Layout (design/buffer-runtime-v4.md, M3)
// ============================================================================

namespace {

struct LayoutInfo {
    std::size_t bytes = 0;
    std::size_t first_use = SIZE_MAX;
    std::size_t last_use = 0;
    bool live = false;  ///< Produced or consumed by an op in this graph
};

std::size_t bytes_for_ref(const TensorRef& ref) {
    if (ref.shape.empty()) return 0;
    std::size_t elems = 1;
    for (long d : ref.shape) {
        if (d <= 0) return 0;  // unresolved / placeholder
        elems *= static_cast<std::size_t>(d);
    }
    return elems * static_cast<std::size_t>(get_dtype_size(ref.dtype));
}

/// Max-clique-sum over a frame's tids: at each op index, sum bytes of tids
/// whose lifetime covers that index. For 1D interval graphs this IS the
/// optimal peak achievable by any coloring.
std::size_t frame_optimal_peak(const std::vector<int>& tids,
                               const std::vector<LayoutInfo>& info,
                               std::size_t frame_first,
                               std::size_t frame_last) {
    if (tids.empty() || frame_last < frame_first) return 0;
    std::size_t peak = 0;
    for (std::size_t t = frame_first; t <= frame_last; ++t) {
        std::size_t live = 0;
        for (int tid : tids) {
            const auto& ti = info[static_cast<std::size_t>(tid)];
            if (ti.first_use <= t && t <= ti.last_use) {
                live += ti.bytes;
            }
        }
        peak = std::max(peak, live);
    }
    return peak;
}

/// First-fit-by-offset greedy for 1D-interval coloring. Returns per-tid
/// frame-local byte offsets; sets peak_out to the bytes consumed. Non-optimal
/// but correct (no two live tensors share bytes) and close to the max-clique
/// peak for typical transformer-block activation patterns.
std::vector<std::size_t>
color_frame(const std::vector<int>& tids, const std::vector<LayoutInfo>& info, std::size_t& peak_out) {
    std::vector<std::size_t> offsets(tids.size(), SIZE_MAX);
    if (tids.empty()) {
        peak_out = 0;
        return offsets;
    }

    std::vector<std::size_t> order(tids.size());
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        const auto& ia = info[static_cast<std::size_t>(tids[a])];
        const auto& ib = info[static_cast<std::size_t>(tids[b])];
        if (ia.first_use != ib.first_use) return ia.first_use < ib.first_use;
        return ia.bytes > ib.bytes;  // prefer larger tensors first at same first_use
    });

    struct Slot {
        std::size_t off, size, last;
    };
    std::vector<Slot> live;
    std::size_t peak = 0;

    for (std::size_t idx : order) {
        const auto& ti = info[static_cast<std::size_t>(tids[idx])];

        // Expire slots freed strictly before ti.first_use.
        live.erase(std::remove_if(live.begin(), live.end(), [&](const Slot& s) { return s.last < ti.first_use; }),
                   live.end());

        std::sort(live.begin(), live.end(), [](const Slot& a, const Slot& b) { return a.off < b.off; });

        // Find smallest gap of size >= ti.bytes.
        std::size_t off = 0;
        for (const auto& s : live) {
            if (off + ti.bytes <= s.off) break;
            off = std::max(off, s.off + s.size);
        }

        offsets[idx] = off;
        live.push_back({off, ti.bytes, ti.last_use});
        peak = std::max(peak, off + ti.bytes);
    }

    peak_out = peak;
    return offsets;
}

/// FNV-1a 64-bit. Mixed in a stable, platform-independent way so all ranks
/// produce the same hash for the same graph — prerequisite for the
/// cross-rank layout determinism assertion in design/buffer-runtime-v4.md.
constexpr std::uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
constexpr std::uint64_t kFnvPrime = 0x100000001b3ULL;

std::uint64_t fnv1a_mix(std::uint64_t h, std::uint64_t v) {
    // Little-endian byte order so the hash is portable across ranks.
    for (int i = 0; i < 8; ++i) {
        h ^= static_cast<std::uint64_t>((v >> (i * 8)) & 0xff);
        h *= kFnvPrime;
    }
    return h;
}

std::string fmt_bytes(std::size_t b) {
    std::ostringstream os;
    if (b >= (1ULL << 30)) {
        os << (b / double(1ULL << 30)) << "GB";
    } else if (b >= (1ULL << 20)) {
        os << (b / double(1ULL << 20)) << "MB";
    } else if (b >= (1ULL << 10)) {
        os << (b / double(1ULL << 10)) << "KB";
    } else {
        os << b << "B";
    }
    return os.str();
}

}  // namespace

std::uint64_t compute_layout_hash(const CompiledGraph& graph) {
    std::uint64_t h = kFnvOffset;
    // Per-tid (region, block_layer_idx, offset, bytes) quadruple. Tids are
    // indexed positionally and assigned deterministically by the compiler,
    // so iterating in order is rank-identical.
    for (const auto& meta : graph.tensor_meta) {
        h = fnv1a_mix(h, static_cast<std::uint64_t>(meta.region));
        h = fnv1a_mix(h, static_cast<std::uint64_t>(meta.block_layer_idx + 1));
        h = fnv1a_mix(h, static_cast<std::uint64_t>(meta.offset));
        h = fnv1a_mix(h, static_cast<std::uint64_t>(meta.bytes));
    }
    // Per-region peaks: any drift here shows up even if tid sets happened to
    // match (unlikely but cheap to guard against).
    h = fnv1a_mix(h, static_cast<std::uint64_t>(graph.persistent_bytes));
    h = fnv1a_mix(h, static_cast<std::uint64_t>(graph.persistent_activation_bytes));
    h = fnv1a_mix(h, static_cast<std::uint64_t>(graph.model_scope_persistent_bytes));
    h = fnv1a_mix(h, static_cast<std::uint64_t>(graph.accumulator_bytes));
    h = fnv1a_mix(h, static_cast<std::uint64_t>(graph.fwd_stack_peak));
    h = fnv1a_mix(h, static_cast<std::uint64_t>(graph.bwd_stack_peak));
    h = fnv1a_mix(h, static_cast<std::uint64_t>(graph.save_for_bwd_bytes));
    for (std::size_t b : graph.save_for_bwd_block_bytes) {
        h = fnv1a_mix(h, static_cast<std::uint64_t>(b));
    }
    return h;
}

void compute_layout(CompiledGraph& graph, bool is_backward, bool fwd_per_layer_sections) {
    const std::size_t num_tids = static_cast<std::size_t>(graph.num_tensors);
    std::vector<LayoutInfo> info(num_tids);

    for (std::size_t i = 0; i < graph.ops.size(); ++i) {
        const auto& op = graph.ops[i];
        auto visit = [&](const TensorRef& ref) {
            if (ref.tensor_id < 0 || static_cast<std::size_t>(ref.tensor_id) >= num_tids) return;
            auto& ti = info[static_cast<std::size_t>(ref.tensor_id)];
            ti.first_use = std::min(ti.first_use, i);
            ti.last_use = std::max(ti.last_use, i);
            ti.live = true;
            // Take max over refs. A tid can be referenced via views of
            // different rank and — more importantly — via refs that
            // resolved to different dtypes. The forward output of an op
            // whose compile-time ref is left at the default BF16 dtype
            // followed by a backward-graph ref whose resolution correctly
            // sets FP32 must size the physical buffer to the larger of
            // the two, otherwise the arena slice is half the size the
            // kernel actually writes and the tail overflows into the
            // next slot.
            ti.bytes = std::max(ti.bytes, bytes_for_ref(ref));
        };
        for (const auto& ref : op.inputs)
            visit(ref);
        for (const auto& ref : op.outputs)
            visit(ref);
    }

    // Extend live ranges for tids flagged retain_through_forward by
    // finalize_save_for_bwd — these are backward-read forward activations
    // that recompute regenerates via replay. Their arena bytes must stay
    // exclusive through the full forward op range so later-forward-op
    // outputs can't clobber them. (Legacy Stack bump allocator does this
    // automatically; arena coloring needs the extension.)
    if (!graph.ops.empty()) {
        const std::size_t frame_end = graph.ops.size() - 1;
        for (std::size_t tid = 0; tid < num_tids; ++tid) {
            if (!info[tid].live) continue;
            if (!graph.tensor_meta[tid].retain_through_forward) continue;
            info[tid].last_use = std::max(info[tid].last_use, frame_end);
        }
    }

    // Mirror the sizes onto TensorMeta so later passes (coverage validator,
    // eventual interpreter) don't need LayoutInfo.
    for (std::size_t tid = 0; tid < num_tids; ++tid) {
        graph.tensor_meta[tid].bytes = info[tid].bytes;
    }

    const std::size_t num_layers = graph.layer_start_indices.size();

    // Bucket live tids by (region, block). SaveForBwd is treated per-block
    // (each block has its own persistent slot); FwdStack / BwdStack coloring
    // is per-frame (frames don't coexist, so offsets are frame-local).
    std::vector<int> persistent_tids;
    std::vector<int> persistent_activation_tids;
    std::vector<int> model_scope_persistent_tids;
    std::vector<int> accumulator_tids;
    std::vector<std::vector<int>> fwd_frame(num_layers);
    std::vector<std::vector<int>> bwd_frame(num_layers);
    std::vector<std::vector<int>> save_for_bwd(num_layers);

    for (std::size_t tid = 0; tid < num_tids; ++tid) {
        const auto& meta = graph.tensor_meta[tid];
        const auto& ti = info[tid];
        if (!ti.live || ti.bytes == 0) continue;
        auto block_bucket = [&](std::vector<std::vector<int>>& buckets) {
            if (meta.block_layer_idx >= 0 && static_cast<std::size_t>(meta.block_layer_idx) < num_layers) {
                buckets[static_cast<std::size_t>(meta.block_layer_idx)].push_back(static_cast<int>(tid));
            }
        };
        switch (meta.region) {
            case RegionKind::Persistent: persistent_tids.push_back(static_cast<int>(tid)); break;
            case RegionKind::PersistentActivation: persistent_activation_tids.push_back(static_cast<int>(tid)); break;
            case RegionKind::ModelScopePersistent: model_scope_persistent_tids.push_back(static_cast<int>(tid)); break;
            case RegionKind::Accumulator: accumulator_tids.push_back(static_cast<int>(tid)); break;
            case RegionKind::FwdStack: block_bucket(fwd_frame); break;
            case RegionKind::BwdStack: block_bucket(bwd_frame); break;
            case RegionKind::SaveForBwd: block_bucket(save_for_bwd); break;
            default: break;
        }
    }

    // Build tid → (slot, layer_idx) map so coloring can collapse aliased
    // views (e.g., blocks[L].swiglu + blocks[L].swiglu_flat share
    // TensorSlot::BlockSwiGLU). Runtime resolves all such refs through one
    // arena offset per slot, so the coloring must treat them as a single
    // allocation or arena consumption will corrupt.
    auto is_block_scope_slot = [](TensorSlot s) {
        switch (s) {
            case TensorSlot::Mapped:
            case TensorSlot::Parameter:
            case TensorSlot::Saved:
            case TensorSlot::TokenIDs:
            case TensorSlot::PositionIDs:
            case TensorSlot::Targets:
            case TensorSlot::Losses:
            case TensorSlot::DLoss:
            case TensorSlot::Encoded:
            case TensorSlot::LNFinal:
            case TensorSlot::LNFinalRSTD:
            case TensorSlot::FinalResidual:
            case TensorSlot::FreqCis: return false;
            default: return true;
        }
    };
    struct TidSlotInfo {
        TensorSlot slot = TensorSlot::Mapped;
        int layer_idx = -1;
        bool set = false;
    };
    std::vector<TidSlotInfo> tid_slot(num_tids);
    auto record_ref = [&](const TensorRef& ref, bool is_output) {
        if (ref.tensor_id < 0 || static_cast<std::size_t>(ref.tensor_id) >= num_tids) return;
        auto& rec = tid_slot[static_cast<std::size_t>(ref.tensor_id)];
        if (!rec.set || (is_output && is_block_scope_slot(ref.slot))) {
            rec.slot = ref.slot;
            rec.layer_idx = ref.layer_idx;
            rec.set = true;
        }
    };
    for (const auto& op : graph.ops) {
        for (const auto& ref : op.outputs)
            record_ref(ref, true);
        for (const auto& ref : op.inputs)
            record_ref(ref, false);
    }

    // Collapse slot-aliased tids within each FwdStack/BwdStack frame into
    // coloring groups. Each group contributes ONE entry to the coloring
    // input — its representative tid — with bytes = max across members
    // and live range = union across members. The resulting offset is
    // propagated to every member after coloring.
    struct AliasGroup {
        int representative_tid = -1;
        std::vector<int> member_tids;  // includes representative
    };
    auto collapse_frame_aliases = [&](std::vector<std::vector<int>>& frames) -> std::vector<std::vector<AliasGroup>> {
        std::vector<std::vector<AliasGroup>> per_layer_groups(frames.size());
        for (std::size_t L = 0; L < frames.size(); ++L) {
            auto& tids = frames[L];
            std::unordered_map<int, std::size_t> slot_to_group_idx;
            std::vector<AliasGroup>& groups = per_layer_groups[L];
            std::vector<int> collapsed_tids;
            for (int tid : tids) {
                const auto& rec = tid_slot[static_cast<std::size_t>(tid)];
                if (rec.set && is_block_scope_slot(rec.slot) && rec.layer_idx == static_cast<int>(L)) {
                    const int key = static_cast<int>(rec.slot);
                    auto it = slot_to_group_idx.find(key);
                    if (it == slot_to_group_idx.end()) {
                        slot_to_group_idx[key] = groups.size();
                        AliasGroup g;
                        g.representative_tid = tid;
                        g.member_tids.push_back(tid);
                        groups.push_back(std::move(g));
                        collapsed_tids.push_back(tid);
                    } else {
                        groups[it->second].member_tids.push_back(tid);
                    }
                } else {
                    // Not slot-aliased — keep as a one-tid "group" so the
                    // post-coloring propagation loop is uniform.
                    AliasGroup g;
                    g.representative_tid = tid;
                    g.member_tids.push_back(tid);
                    groups.push_back(std::move(g));
                    collapsed_tids.push_back(tid);
                }
            }
            tids = std::move(collapsed_tids);
            // Update the representative's coloring info to reflect the union
            // of member live ranges and max bytes.
            for (auto& g : groups) {
                if (g.member_tids.size() <= 1) continue;
                auto& rep_info = info[static_cast<std::size_t>(g.representative_tid)];
                for (int member : g.member_tids) {
                    if (member == g.representative_tid) continue;
                    const auto& mi = info[static_cast<std::size_t>(member)];
                    rep_info.first_use = std::min(rep_info.first_use, mi.first_use);
                    rep_info.last_use = std::max(rep_info.last_use, mi.last_use);
                    rep_info.bytes = std::max(rep_info.bytes, mi.bytes);
                }
                // Keep bytes on tensor_meta in sync with the union size —
                // downstream readers use meta.bytes for arena sizing.
                graph.tensor_meta[static_cast<std::size_t>(g.representative_tid)].bytes = rep_info.bytes;
            }
        }
        return per_layer_groups;
    };
    auto fwd_alias_groups = collapse_frame_aliases(fwd_frame);
    auto bwd_alias_groups = collapse_frame_aliases(bwd_frame);

    // Bump: sort by tid for deterministic ordering across ranks.
    auto bump_sort = [](std::vector<int>& tids) {
        std::sort(tids.begin(), tids.end());
    };
    bump_sort(persistent_tids);
    bump_sort(persistent_activation_tids);
    bump_sort(accumulator_tids);
    for (auto& v : save_for_bwd)
        bump_sort(v);

    // Align each bump-allocated offset to `kBumpAlignment`. Without this,
    // a tensor with `bytes` not divisible by the alignment leaves the
    // cursor at a fractional offset and every subsequent tensor is
    // misaligned. cuBLASLt (BF16 tensor-core matmul) and vectorized init
    // kernels (fill_normal float4/bf16x4) both require >=16-byte aligned
    // pointers. Gemma4's per-layer `layer_scalar` is 2 bytes (bf16
    // scalar), so without this, every weight after block[0].layer_scalar
    // is placed at a misaligned offset in the Persistent arena.
    constexpr std::size_t kBumpAlignment = 16;
    auto align_up = [](std::size_t n, std::size_t a) {
        return (n + a - 1) & ~(a - 1);
    };
    auto bump_assign = [&](const std::vector<int>& tids) -> std::size_t {
        std::size_t offset = 0;
        for (int tid : tids) {
            graph.tensor_meta[static_cast<std::size_t>(tid)].offset = offset;
            offset = align_up(offset + info[static_cast<std::size_t>(tid)].bytes, kBumpAlignment);
        }
        return offset;
    };

    const std::size_t persistent_bytes = bump_assign(persistent_tids);
    const std::size_t persistent_activation_bytes = bump_assign(persistent_activation_tids);
    const std::size_t model_scope_persistent_bytes = bump_assign(model_scope_persistent_tids);
    const std::size_t accumulator_bytes = bump_assign(accumulator_tids);
    std::size_t save_for_bwd_bytes = 0;
    std::vector<std::size_t> save_for_bwd_block_bytes(num_layers, 0);
    for (std::size_t L = 0; L < num_layers; ++L) {
        const std::size_t block_sz = bump_assign(save_for_bwd[L]);
        save_for_bwd_block_bytes[L] = block_sz;
        save_for_bwd_bytes += block_sz;
    }

    // Per-frame coloring with optional per-layer sectioning.
    //
    // Under `section_per_layer`, each layer gets its own
    // [L*peak, (L+1)*peak) slice of the arena, so every (L, SLOT) pair
    // resolves to a distinct address. This is the correctness
    // requirement for FwdStack under recompute: replay of layer K
    // regenerates layer K's activations to the arena, and subsequent
    // backward ops for layer K read them. If two layers shared arena bytes,
    // an interleaved backward op that reads layer K's activation after
    // another layer's replay had overwritten the shared bytes would get
    // stale data. Legacy Stack avoided this via per-layer checkpoints.
    //
    // Without sectioning, frames share arena bytes (nested-stack
    // semantics — frames don't coexist). BwdStack is a per-layer scratch
    // that the layer-sequential backward dispatch frees before moving to
    // the next layer, so sharing across layers is safe.
    //
    // Memory cost of sectioning: num_layers × peak_per_frame. Within a
    // layer's section, different slots still share bytes when their live
    // ranges are disjoint (per-frame coloring preserved).
    auto color_frames = [&](const std::vector<std::vector<int>>& frames,
                            const std::vector<std::vector<AliasGroup>>& alias_groups,
                            bool section_per_layer) -> std::pair<std::size_t, std::size_t> {
        std::size_t naive_max = 0;
        std::size_t coloring_per_frame_peak = 0;
        std::vector<std::size_t> per_layer_peak(frames.size(), 0);
        std::vector<std::vector<std::size_t>> per_layer_offsets(frames.size());
        for (std::size_t L = 0; L < frames.size(); ++L) {
            const auto& tids = frames[L];
            std::size_t naive = 0;
            for (int tid : tids)
                naive += info[static_cast<std::size_t>(tid)].bytes;
            naive_max = std::max(naive_max, naive);

            std::size_t peak = 0;
            per_layer_offsets[L] = color_frame(tids, info, peak);
            per_layer_peak[L] = peak;
            coloring_per_frame_peak = std::max(coloring_per_frame_peak, peak);
        }
        const std::size_t stride = section_per_layer ? coloring_per_frame_peak : 0;
        std::size_t total_bytes = section_per_layer ? 0 : coloring_per_frame_peak;
        for (std::size_t L = 0; L < frames.size(); ++L) {
            const auto& tids = frames[L];
            const std::size_t base = section_per_layer ? L * stride : 0;
            for (std::size_t i = 0; i < tids.size(); ++i) {
                graph.tensor_meta[static_cast<std::size_t>(tids[i])].offset = base + per_layer_offsets[L][i];
            }
            if (L < alias_groups.size()) {
                for (const auto& g : alias_groups[L]) {
                    if (g.member_tids.size() <= 1) continue;
                    const std::size_t rep_off =
                        graph.tensor_meta[static_cast<std::size_t>(g.representative_tid)].offset;
                    for (int member : g.member_tids) {
                        if (member == g.representative_tid) continue;
                        graph.tensor_meta[static_cast<std::size_t>(member)].offset = rep_off;
                        graph.tensor_meta[static_cast<std::size_t>(member)].bytes =
                            graph.tensor_meta[static_cast<std::size_t>(g.representative_tid)].bytes;
                    }
                }
            }
            if (section_per_layer) {
                total_bytes = std::max(total_bytes, base + per_layer_peak[L]);
            }
        }
        return {naive_max, total_bytes};
    };

    // FwdStack and BwdStack: per-frame coloring, single shared arena per
    // region. Matches design/buffer-runtime-v4.md §Region vocabulary:
    // FwdStack is a nested phase scope re-entered fresh per block;
    // recompute's `Recomputed[i]` lands in the same arena just-in-time
    // during BwdBlock[i] replay. Peak = max over layers, not sum.
    //
    // Correctness prerequisite: replay_layer_forward must iterate ops
    // with a half-open [start, end) range so it doesn't execute layer
    // L+1's first op (qwen3's deferred-residual fused_residual_rmsnorm
    // computes layer L+1's LN1 from layer L's residual and MLP-down
    // outputs) as part of layer L's replay. Under shared arena, that
    // extra op overwrote layer L's just-replayed LN1 before layer L's
    // backward ops read it. See the `idx < end` bound in
    // compiled_ops_execute.cpp.
    auto [fwd_naive, fwd_colored] =
        color_frames(fwd_frame, fwd_alias_groups, /*section_per_layer=*/(fwd_per_layer_sections && !is_backward));
    auto [bwd_naive, bwd_colored] = color_frames(bwd_frame, bwd_alias_groups, /*section_per_layer=*/false);

    // Expose peaks for compute_arena_sizes.
    graph.persistent_bytes = persistent_bytes;
    graph.persistent_activation_bytes = persistent_activation_bytes;
    graph.model_scope_persistent_bytes = model_scope_persistent_bytes;
    graph.accumulator_bytes = accumulator_bytes;
    graph.fwd_stack_peak = fwd_colored;
    graph.bwd_stack_peak = bwd_colored;
    graph.save_for_bwd_bytes = save_for_bwd_bytes;
    graph.save_for_bwd_block_bytes = std::move(save_for_bwd_block_bytes);

    // Compute optimal peak per frame (max-clique-sum) for comparison vs our
    // first-fit-by-offset coloring. Reports how close the greedy is to the
    // theoretical bound.
    auto frame_optimal = [&](const std::vector<std::vector<int>>& frames) -> std::size_t {
        std::size_t best = 0;
        for (std::size_t L = 0; L < frames.size(); ++L) {
            const auto s = graph.layer_start_indices[L];
            const auto e = graph.layer_end_indices[L];
            if (s == SIZE_MAX || e == SIZE_MAX || s >= e) continue;
            best = std::max(best, frame_optimal_peak(frames[L], info, s, e - 1));
        }
        return best;
    };
    const std::size_t fwd_optimal = frame_optimal(fwd_frame);
    const std::size_t bwd_optimal = frame_optimal(bwd_frame);

    // Liveness validator. Per-frame coloring packs multiple tids into the
    // same arena bytes when their live ranges are disjoint — safe by
    // construction, unsafe if liveness analysis missed a producer/consumer
    // edge. Walks every frame; for each pair of tids whose
    // [offset, offset+bytes) intervals overlap, asserts their
    // [first_use, last_use] intervals are disjoint. Any violation is a
    // compile-time bug that would corrupt memory at arena consumption time.
    std::size_t coloring_violations = 0;
    auto check_frame = [&](const std::vector<std::vector<int>>& frames, const char* where) {
        for (std::size_t L = 0; L < frames.size(); ++L) {
            const auto& tids = frames[L];
            for (std::size_t i = 0; i < tids.size(); ++i) {
                const auto& mi = graph.tensor_meta[static_cast<std::size_t>(tids[i])];
                const auto& ti = info[static_cast<std::size_t>(tids[i])];
                for (std::size_t j = i + 1; j < tids.size(); ++j) {
                    const auto& mj = graph.tensor_meta[static_cast<std::size_t>(tids[j])];
                    const auto& tj = info[static_cast<std::size_t>(tids[j])];
                    const std::size_t ai_end = mi.offset + mi.bytes;
                    const std::size_t aj_end = mj.offset + mj.bytes;
                    // Bytes ranges disjoint?
                    if (ai_end <= mj.offset || aj_end <= mi.offset) continue;
                    // Bytes overlap: live ranges must be disjoint.
                    const bool lives_disjoint = ti.last_use < tj.first_use || tj.last_use < ti.first_use;
                    if (lives_disjoint) continue;
                    ++coloring_violations;
                    std::cerr << "[coloring-violation] " << where << " L=" << L << " tid=" << tids[i]
                              << " name=" << graph.name_for_tensor_id(tids[i]) << " [off=" << mi.offset
                              << ",bytes=" << mi.bytes << ",live=" << ti.first_use << ".." << ti.last_use << "]"
                              << " vs tid=" << tids[j] << " name=" << graph.name_for_tensor_id(tids[j])
                              << " [off=" << mj.offset << ",bytes=" << mj.bytes << ",live=" << tj.first_use << ".."
                              << tj.last_use << "] — bytes overlap AND live ranges overlap\n";
                }
            }
        }
    };
    // Slot-aliasing validator. The coloring collapse above should have made
    // every (layer, block-scope slot) group share one offset — this check
    // confirms and catches regressions.
    std::size_t alias_offset_splits = 0;
    auto check_slot_aliases = [&](const std::vector<std::vector<int>>& frames, const char* where) {
        for (std::size_t L = 0; L < frames.size(); ++L) {
            // Group this layer's FwdStack/BwdStack tids by their block-scope slot.
            std::unordered_map<int, std::vector<int>> by_slot;  // key = int(TensorSlot)
            for (int tid : frames[L]) {
                const auto& rec = tid_slot[static_cast<std::size_t>(tid)];
                if (!rec.set) continue;
                if (!is_block_scope_slot(rec.slot)) continue;
                if (rec.layer_idx != static_cast<int>(L)) continue;
                by_slot[static_cast<int>(rec.slot)].push_back(tid);
            }
            for (auto& [slot_int, group] : by_slot) {
                if (group.size() < 2) continue;
                const auto& first_meta = graph.tensor_meta[static_cast<std::size_t>(group.front())];
                for (std::size_t i = 1; i < group.size(); ++i) {
                    const auto& mi = graph.tensor_meta[static_cast<std::size_t>(group[i])];
                    if (mi.offset != first_meta.offset) {
                        ++alias_offset_splits;
                        std::cerr << "[coloring-alias-split] " << where << " L=" << L << " slot=" << slot_int
                                  << " tid=" << group.front() << "(" << graph.name_for_tensor_id(group.front())
                                  << ",off=" << first_meta.offset << ") vs tid=" << group[i] << "("
                                  << graph.name_for_tensor_id(group[i]) << ",off=" << mi.offset
                                  << ") — runtime slot-aliased but compile-time offsets differ\n";
                    }
                }
            }
        }
    };
    // Cross-frame coloring validator. Under shared-arena forward mode
    // (`section_per_layer=false`, enabled when recompute is on), different
    // layer frames share the same FwdStack offset space. Same-slot sharing
    // across layers is INTENTIONAL under recompute: layer L's forward
    // tensors are regenerated by backward replay, not kept live in the
    // arena past layer L's execution. The only unsafe case is a tid whose
    // RAW consumers extend past its producing layer — a genuine cross-layer
    // forward read (Gemma4 shared-KV's `kv_source = blocks[L_src].qkv_rope`
    // is the canonical example). retain_through_forward inflates every
    // FwdStack tid's last_use to frame-end for save-snapshot correctness, so
    // the raw op-scan last_use has to be recomputed here to distinguish true
    // cross-layer readers from retention artifacts.
    std::size_t cross_frame_violations = 0;
    std::ostringstream cross_frame_diag;
    if (!is_backward) {
        // Raw live ranges derived from ops only — no retain_through_forward
        // extension. Only these expose genuine cross-layer forward reads.
        std::vector<std::size_t> raw_first(num_tids, SIZE_MAX);
        std::vector<std::size_t> raw_last(num_tids, 0);
        std::vector<char> raw_live(num_tids, 0);
        for (std::size_t i = 0; i < graph.ops.size(); ++i) {
            const auto& op = graph.ops[i];
            auto visit = [&](const TensorRef& ref) {
                if (ref.tensor_id < 0 || static_cast<std::size_t>(ref.tensor_id) >= num_tids) return;
                const std::size_t tid = static_cast<std::size_t>(ref.tensor_id);
                raw_first[tid] = std::min(raw_first[tid], i);
                raw_last[tid] = std::max(raw_last[tid], i);
                raw_live[tid] = 1;
            };
            for (const auto& ref : op.inputs)
                visit(ref);
            for (const auto& ref : op.outputs)
                visit(ref);
        }

        // Flag tids whose raw consumers span past the producing layer — true
        // cross-layer forward readers. Only these need cross-frame coloring
        // protection. Everything else participates in the legitimate
        // retain-through-forward + same-slot cross-layer reuse pattern.
        std::vector<char> is_cross_reader(num_tids, 0);
        for (std::size_t tid = 0; tid < num_tids; ++tid) {
            if (!raw_live[tid]) continue;
            const auto& meta = graph.tensor_meta[tid];
            if (meta.region != RegionKind::FwdStack) continue;
            if (meta.block_layer_idx < 0) continue;
            const std::size_t L = static_cast<std::size_t>(meta.block_layer_idx);
            if (L >= num_layers) continue;
            const std::size_t layer_end =
                (graph.layer_end_indices[L] != SIZE_MAX) ? graph.layer_end_indices[L] : raw_last[tid];
            if (raw_last[tid] >= layer_end) {
                is_cross_reader[tid] = 1;
            }
        }

        const bool section_per_layer = fwd_per_layer_sections;
        if (!section_per_layer) {
            // Walk FwdStack tids; flag only when a cross-reader shares bytes
            // with another tid whose raw live range overlaps it.
            std::vector<int> fwd_tids;
            for (const auto& frame : fwd_frame) {
                for (int tid : frame)
                    fwd_tids.push_back(tid);
            }
            for (std::size_t i = 0; i < fwd_tids.size(); ++i) {
                const int ti_id = fwd_tids[i];
                const auto& mi = graph.tensor_meta[static_cast<std::size_t>(ti_id)];
                for (std::size_t j = i + 1; j < fwd_tids.size(); ++j) {
                    const int tj_id = fwd_tids[j];
                    const auto& mj = graph.tensor_meta[static_cast<std::size_t>(tj_id)];
                    if (mi.block_layer_idx == mj.block_layer_idx) continue;  // same-frame handled elsewhere
                    if (!is_cross_reader[ti_id] && !is_cross_reader[tj_id]) continue;
                    const std::size_t ai_end = mi.offset + mi.bytes;
                    const std::size_t aj_end = mj.offset + mj.bytes;
                    if (ai_end <= mj.offset || aj_end <= mi.offset) continue;
                    const std::size_t i_first = raw_first[ti_id], i_last = raw_last[ti_id];
                    const std::size_t j_first = raw_first[tj_id], j_last = raw_last[tj_id];
                    const bool lives_disjoint = i_last < j_first || j_last < i_first;
                    if (lives_disjoint) continue;
                    if (cross_frame_violations < 10) {
                        cross_frame_diag << "[cross-frame-violation] " << graph.name << " tid=" << ti_id
                                         << " name=" << graph.name_for_tensor_id(ti_id)
                                         << " layer=" << mi.block_layer_idx << " [off=" << mi.offset
                                         << ",bytes=" << mi.bytes << ",raw_live=" << i_first << ".." << i_last << "]"
                                         << " vs tid=" << tj_id << " name=" << graph.name_for_tensor_id(tj_id)
                                         << " layer=" << mj.block_layer_idx << " [off=" << mj.offset
                                         << ",bytes=" << mj.bytes << ",raw_live=" << j_first << ".." << j_last
                                         << "] — bytes overlap AND raw live ranges overlap"
                                         << " (cross_reader: i=" << (is_cross_reader[ti_id] ? "y" : "n")
                                         << " j=" << (is_cross_reader[tj_id] ? "y" : "n") << ")\n";
                    }
                    ++cross_frame_violations;
                }
            }
        }
    }

    if (const char* env = std::getenv("SUROGATE_CHECK_FRAME_COLORING")) {
        if (std::string(env) == "1") {
            check_frame(fwd_frame, is_backward ? "bwd.fwd_frame?" : "fwd_frame");
            check_frame(bwd_frame, is_backward ? "bwd_frame" : "fwd.bwd_frame?");
            check_slot_aliases(fwd_frame, is_backward ? "bwd.fwd_frame?" : "fwd_frame");
            check_slot_aliases(bwd_frame, is_backward ? "bwd_frame" : "fwd.bwd_frame?");
            std::cerr << "[coloring-check] graph='" << graph.name << "' pairwise_violations=" << coloring_violations
                      << " alias_offset_splits=" << alias_offset_splits
                      << " cross_frame_violations=" << cross_frame_violations << "\n";
        }
    }

    if (cross_frame_violations > 0) {
        std::cerr << cross_frame_diag.str();
        throw std::runtime_error(
            "GraphCompiler::compute_layout: cross-frame arena coloring violation in graph '" + graph.name + "' (" +
            std::to_string(cross_frame_violations) +
            " overlap(s)). A forward activation produced in one layer shares arena bytes with another "
            "layer's tensor whose live range overlaps it, which corrupts cross-layer reads at runtime. "
            "Fix the IR (e.g., mark the source tensor with a persistent share policy, or add its tid to "
            "promote_cross_layer_fwd_reads) rather than silencing this check.");
    }

    // Populate CompiledGraph::slot_tid_by_layer so downstream dispatchers
    // can resolve (layer_idx, slot) → tid in O(1) instead of constructing
    // "blocks[L].<slot_name>" and hitting tensor_name_to_id. tid_slot
    // above already carries the canonical (slot, layer) for every tid;
    // fold it into the per-layer array.
    {
        constexpr std::size_t kSlotCount = static_cast<std::size_t>(TensorSlot::Mapped) + 1;
        graph.slot_tid_by_layer.assign(num_layers, {});
        for (auto& row : graph.slot_tid_by_layer) {
            row.fill(-1);
        }
        for (std::size_t tid = 0; tid < num_tids; ++tid) {
            const auto& rec = tid_slot[tid];
            if (!rec.set) continue;
            if (rec.layer_idx < 0 || static_cast<std::size_t>(rec.layer_idx) >= num_layers) continue;
            const auto slot_idx = static_cast<std::size_t>(rec.slot);
            if (slot_idx >= kSlotCount) continue;
            // First writer wins; downstream ops that reference the same
            // (layer, slot) with aliased tids resolve to the producer's
            // canonical tid because record_ref favors outputs over inputs.
            if (graph.slot_tid_by_layer[static_cast<std::size_t>(rec.layer_idx)][slot_idx] < 0) {
                graph.slot_tid_by_layer[static_cast<std::size_t>(rec.layer_idx)][slot_idx] = static_cast<int>(tid);
            }
        }
    }

    // Layout hash — determinism check point for distributed runs (Phase 2
    // step 5). Every rank running the same compile must produce the same
    // 64-bit value; callers can cross-rank-compare via NCCL/MPI allreduce.
    graph.layout_hash = compute_layout_hash(graph);

    if (const char* env = std::getenv("SUROGATE_DEBUG_LAYOUT")) {
        if (std::string(env) == "1") {
            std::cerr << "[layout] " << (is_backward ? "backward" : "forward") << " compile (" << graph.name << "):\n"
                      << "  Persistent           = " << fmt_bytes(persistent_bytes) << "\n"
                      << "  PersistentActivation = " << fmt_bytes(persistent_activation_bytes) << "\n"
                      << "  ModelScopePersistent = " << fmt_bytes(model_scope_persistent_bytes) << "\n"
                      << "  Accumulator  = " << fmt_bytes(accumulator_bytes) << "\n"
                      << "  SaveForBwd   = " << fmt_bytes(save_for_bwd_bytes) << "\n"
                      << "  FwdStack     = " << fmt_bytes(fwd_colored) << " (naive " << fmt_bytes(fwd_naive)
                      << ", optimal " << fmt_bytes(fwd_optimal) << ")\n"
                      << "  BwdStack     = " << fmt_bytes(bwd_colored) << " (naive " << fmt_bytes(bwd_naive)
                      << ", optimal " << fmt_bytes(bwd_optimal) << ")\n"
                      << "  layout_hash  = 0x" << std::hex << graph.layout_hash << std::dec << "\n";
        }
    }
}

// ============================================================================
// Phase Arenas — allocation skeleton
// ============================================================================

void compute_arena_sizes(PhaseArenas& arenas,
                         const CompiledGraph& fwd,
                         const CompiledGraph& bwd,
                         int num_layers,
                         std::size_t stack_bytes,
                         std::size_t bwd_cross_layer_bytes,
                         std::size_t moe_saved_bytes) {
    // UnifiedStack arena sizing. The design calls for Stack to be backed
    // by the phase-arena bookkeeping, but the adoption sequence
    // (cudaMalloc → memcpy-rebase → free original) briefly holds two
    // Stack-sized buffers on-device; on Qwen3 0.6B that's +1.1 GiB
    // transient, past the `<2% peak-memory` benchmark gate. Skip the
    // adoption (unified_stack_bytes = 0) and keep the allocator-owned
    // Stack as-is — functionally equivalent (Stack pointers are valid
    // either way), and steady-state memory is unchanged.
    (void)stack_bytes;
    arenas.unified_stack_bytes = 0;
    arenas.bwd_cross_layer_bytes = bwd_cross_layer_bytes;
    arenas.moe_saved_bytes = moe_saved_bytes;

    // Persistent (weights) + Accumulator (grads): both default-on. The
    // executor clamps each to what its owner can actually rebind —
    // rebindable_persistent_bytes (base + WM) and rebindable_accumulator_bytes
    // (grad store) drop the slab to zero when the config (QLoRA, streaming,
    // sharding, offload) keeps storage elsewhere.
    arenas.persistent_bytes = std::max(fwd.persistent_bytes, bwd.persistent_bytes);
    arenas.persistent_activation_bytes = std::max(fwd.persistent_activation_bytes, bwd.persistent_activation_bytes);
    arenas.model_scope_persistent_bytes = std::max(fwd.model_scope_persistent_bytes, bwd.model_scope_persistent_bytes);
    arenas.accumulator_bytes = std::max(fwd.accumulator_bytes, bwd.accumulator_bytes);

    // FwdStack, BwdStack: single-arena routing per the plan — FwdStack
    // arena sized to the max forward frame peak, BwdStack to the max
    // backward frame peak, both reused across layers. Replay regenerates
    // each layer's activations into the arena just-in-time before
    // backward ops read, matching `Recomputed[i]` in
    // design/buffer-runtime-v4.md.
    arenas.fwd_stack_bytes = fwd.fwd_stack_peak;
    arenas.bwd_stack_bytes = bwd.bwd_stack_peak;

    // SaveForBwd: per-block slots persist across the fwd-exit / bwd-entry
    // boundary. finalize_save_for_bwd populates the same tids in both
    // compiles, so per-block sizes should match; take max as a safety.
    arenas.save_for_bwd_block_bases.assign(static_cast<std::size_t>(num_layers), 0);
    std::size_t total = 0;
    for (int L = 0; L < num_layers; ++L) {
        const auto idx = static_cast<std::size_t>(L);
        const std::size_t fwd_bytes = idx < fwd.save_for_bwd_block_bytes.size() ? fwd.save_for_bwd_block_bytes[idx] : 0;
        const std::size_t bwd_bytes = idx < bwd.save_for_bwd_block_bytes.size() ? bwd.save_for_bwd_block_bytes[idx] : 0;
        arenas.save_for_bwd_block_bases[idx] = total;
        total += std::max(fwd_bytes, bwd_bytes);
    }
    arenas.save_for_bwd_bytes = total;
}

std::size_t estimate_bwd_cross_layer_bytes(const CompiledGraph& bwd) {
    if (bwd.layer_end_indices.empty() || bwd.num_tensors <= 0) {
        return 0;
    }

    auto is_stack_resident = [](RegionKind r) {
        return r == RegionKind::FwdStack || r == RegionKind::BwdStack || r == RegionKind::Recomputed ||
               r == RegionKind::Unknown;
    };

    // Collect layer-end op indices in execution order. layer_end_indices[L]
    // is past-the-end; the op that ends layer L sits at index le-1.
    std::vector<std::size_t> layer_end_ops;
    layer_end_ops.reserve(bwd.layer_end_indices.size());
    for (std::size_t le : bwd.layer_end_indices) {
        if (le == SIZE_MAX || le == 0) continue;
        layer_end_ops.push_back(le - 1);
    }
    if (layer_end_ops.empty()) return 0;
    std::sort(layer_end_ops.begin(), layer_end_ops.end());

    std::size_t total = 0;
    for (int tid = 0; tid < bwd.num_tensors; ++tid) {
        const auto& meta = bwd.tensor_meta[static_cast<std::size_t>(tid)];
        if (!is_stack_resident(meta.region)) continue;
        if (meta.bytes == 0) continue;
        const std::string_view name = bwd.name_for_tensor_id(tid);
        if (name.empty()) continue;
        auto it = bwd.last_use_index.find(std::string(name));
        if (it == bwd.last_use_index.end()) continue;
        const std::size_t last_use = it->second;
        // Persisted at the first layer-end whose op_idx < last_use. If
        // none of the layer ends fall before last_use, the tid never
        // crosses a layer boundary and won't be persisted.
        if (last_use <= layer_end_ops.front()) continue;
        total += meta.bytes;
    }
    return total;
}

namespace {

void cuda_malloc_or_die(std::byte** out, std::size_t bytes, const char* label) {
    if (bytes == 0) {
        *out = nullptr;
        return;
    }
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(out), bytes);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "allocate_phase_arenas: cudaMalloc(" << bytes << ") for " << label
            << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

}  // namespace

void allocate_phase_arenas(PhaseArenas& arenas) {
    if (arenas.allocated) return;
    cuda_malloc_or_die(&arenas.persistent_ptr, arenas.persistent_bytes, "persistent");
    cuda_malloc_or_die(&arenas.persistent_activation_ptr, arenas.persistent_activation_bytes, "persistent_activation");
    cuda_malloc_or_die(&arenas.model_scope_persistent_ptr,
                       arenas.model_scope_persistent_bytes,
                       "model_scope_persistent");
    cuda_malloc_or_die(&arenas.accumulator_ptr, arenas.accumulator_bytes, "accumulator");
    cuda_malloc_or_die(&arenas.fwd_stack_ptr, arenas.fwd_stack_bytes, "fwd_stack");
    cuda_malloc_or_die(&arenas.bwd_stack_ptr, arenas.bwd_stack_bytes, "bwd_stack");
    cuda_malloc_or_die(&arenas.save_for_bwd_ptr, arenas.save_for_bwd_bytes, "save_for_bwd");
    cuda_malloc_or_die(&arenas.unified_stack_ptr, arenas.unified_stack_bytes, "unified_stack");
    cuda_malloc_or_die(&arenas.bwd_cross_layer_ptr, arenas.bwd_cross_layer_bytes, "bwd_cross_layer");
    cuda_malloc_or_die(&arenas.moe_saved_ptr, arenas.moe_saved_bytes, "moe_saved");
    arenas.allocated = true;

    if (const char* env = std::getenv("SUROGATE_DEBUG_LAYOUT")) {
        if (std::string(env) == "1") {
            auto mb = [](std::size_t b) {
                return b / double(1ULL << 20);
            };
            std::cerr << "[arena] allocated:\n"
                      << "  Persistent    = " << mb(arenas.persistent_bytes) << " MB @ "
                      << static_cast<void*>(arenas.persistent_ptr) << "\n"
                      << "  PersistentAct = " << mb(arenas.persistent_activation_bytes) << " MB @ "
                      << static_cast<void*>(arenas.persistent_activation_ptr) << "\n"
                      << "  ModelScopePer = " << mb(arenas.model_scope_persistent_bytes) << " MB @ "
                      << static_cast<void*>(arenas.model_scope_persistent_ptr) << "\n"
                      << "  Accumulator   = " << mb(arenas.accumulator_bytes) << " MB @ "
                      << static_cast<void*>(arenas.accumulator_ptr) << "\n"
                      << "  FwdStack      = " << mb(arenas.fwd_stack_bytes) << " MB @ "
                      << static_cast<void*>(arenas.fwd_stack_ptr) << "\n"
                      << "  BwdStack      = " << mb(arenas.bwd_stack_bytes) << " MB @ "
                      << static_cast<void*>(arenas.bwd_stack_ptr) << "\n"
                      << "  SaveForBwd    = " << mb(arenas.save_for_bwd_bytes) << " MB @ "
                      << static_cast<void*>(arenas.save_for_bwd_ptr) << "\n"
                      << "  UnifiedStack  = " << mb(arenas.unified_stack_bytes) << " MB @ "
                      << static_cast<void*>(arenas.unified_stack_ptr) << "\n"
                      << "  TOTAL         = "
                      << mb(arenas.persistent_bytes + arenas.persistent_activation_bytes +
                            arenas.model_scope_persistent_bytes + arenas.accumulator_bytes + arenas.fwd_stack_bytes +
                            arenas.bwd_stack_bytes + arenas.save_for_bwd_bytes + arenas.unified_stack_bytes)
                      << " MB\n";
        }
    }
}

std::byte* resolve_tid_in_arena(const PhaseArenas& arenas, const CompiledGraph& graph, int tid) {
    if (!arenas.allocated || tid < 0 || tid >= graph.num_tensors) return nullptr;
    const auto& meta = graph.tensor_meta[static_cast<std::size_t>(tid)];
    if (meta.offset == SIZE_MAX) return nullptr;
    switch (meta.region) {
        case RegionKind::Persistent: return arenas.persistent_ptr + meta.offset;
        case RegionKind::PersistentActivation: return arenas.persistent_activation_ptr + meta.offset;
        case RegionKind::ModelScopePersistent: return arenas.model_scope_persistent_ptr + meta.offset;
        case RegionKind::Accumulator: return arenas.accumulator_ptr + meta.offset;
        case RegionKind::FwdStack: return arenas.fwd_stack_ptr + meta.offset;  // frame-local offset
        case RegionKind::BwdStack: return arenas.bwd_stack_ptr + meta.offset;
        case RegionKind::SaveForBwd:
            if (meta.block_layer_idx < 0 ||
                static_cast<std::size_t>(meta.block_layer_idx) >= arenas.save_for_bwd_block_bases.size()) {
                return nullptr;
            }
            return arenas.save_for_bwd_ptr +
                   arenas.save_for_bwd_block_bases[static_cast<std::size_t>(meta.block_layer_idx)] + meta.offset;
        default: return nullptr;
    }
}

ArenaCoverage validate_arena_coverage(const PhaseArenas& arenas, const CompiledGraph& graph) {
    ArenaCoverage cov;
    if (!arenas.allocated) return cov;

    auto region_capacity = [&](RegionKind k) -> std::size_t {
        switch (k) {
            case RegionKind::Persistent: return arenas.persistent_bytes;
            case RegionKind::PersistentActivation: return arenas.persistent_activation_bytes;
            case RegionKind::ModelScopePersistent: return arenas.model_scope_persistent_bytes;
            case RegionKind::Accumulator: return arenas.accumulator_bytes;
            case RegionKind::FwdStack: return arenas.fwd_stack_bytes;
            case RegionKind::BwdStack: return arenas.bwd_stack_bytes;
            case RegionKind::SaveForBwd: return arenas.save_for_bwd_bytes;
            default: return 0;
        }
    };

    for (int tid = 0; tid < graph.num_tensors; ++tid) {
        const auto& meta = graph.tensor_meta[static_cast<std::size_t>(tid)];
        if (meta.region == RegionKind::Unknown) continue;
        ++cov.total;
        if (meta.offset == SIZE_MAX || meta.bytes == 0) continue;
        switch (meta.region) {
            case RegionKind::Persistent:
            case RegionKind::PersistentActivation:
            case RegionKind::ModelScopePersistent:
            case RegionKind::Accumulator:
            case RegionKind::FwdStack:
            case RegionKind::BwdStack: {
                const std::size_t end = meta.offset + meta.bytes;
                if (end > region_capacity(meta.region)) {
                    ++cov.size_exceeded;
                } else {
                    ++cov.covered;
                }
                break;
            }
            case RegionKind::SaveForBwd: {
                if (meta.block_layer_idx < 0) break;
                const auto L = static_cast<std::size_t>(meta.block_layer_idx);
                if (L >= arenas.save_for_bwd_block_bases.size()) break;
                const std::size_t slot_base = arenas.save_for_bwd_block_bases[L];
                const std::size_t slot_end = (L + 1 < arenas.save_for_bwd_block_bases.size())
                                                 ? arenas.save_for_bwd_block_bases[L + 1]
                                                 : arenas.save_for_bwd_bytes;
                const std::size_t slot_size = slot_end - slot_base;
                if (meta.offset + meta.bytes > slot_size) {
                    ++cov.size_exceeded;
                } else {
                    ++cov.covered;
                }
                break;
            }
            default: break;
        }
    }

    if (const char* env = std::getenv("SUROGATE_DEBUG_ARENA_COVERAGE")) {
        if (std::string(env) == "1") {
            const double pct =
                cov.total > 0 ? (100.0 * static_cast<double>(cov.covered) / static_cast<double>(cov.total)) : 0.0;
            std::cerr << "[arena-coverage] " << graph.name << ": " << cov.covered << "/" << cov.total << " tids ("
                      << pct << "%)";
            if (cov.size_exceeded > 0) {
                std::cerr << "  size_exceeded=" << cov.size_exceeded;
            }
            std::cerr << "\n";
        }
    }
    return cov;
}

std::optional<BakedOperand> baked_view(const CompiledGraph& graph, const TensorRef& ref) {
    const int tid = ref.tensor_id;
    if (tid < 0 || tid >= graph.num_tensors) return std::nullopt;
    const auto& meta = graph.tensor_meta[static_cast<std::size_t>(tid)];
    if (meta.region == RegionKind::Unknown) return std::nullopt;
    if (meta.offset == SIZE_MAX) return std::nullopt;
    return BakedOperand{meta.region, meta.block_layer_idx, meta.offset, meta.bytes, ref.dtype};
}

OpOperandCoverage validate_op_operand_coverage(const PhaseArenas& arenas, const CompiledGraph& graph) {
    OpOperandCoverage cov{};

    auto region_allocated = [&](RegionKind k) {
        switch (k) {
            case RegionKind::Persistent: return arenas.persistent_ptr != nullptr;
            case RegionKind::PersistentActivation: return arenas.persistent_activation_ptr != nullptr;
            case RegionKind::ModelScopePersistent: return arenas.model_scope_persistent_ptr != nullptr;
            case RegionKind::Accumulator: return arenas.accumulator_ptr != nullptr;
            case RegionKind::FwdStack: return arenas.fwd_stack_ptr != nullptr;
            case RegionKind::BwdStack: return arenas.bwd_stack_ptr != nullptr;
            case RegionKind::SaveForBwd: return arenas.save_for_bwd_ptr != nullptr;
            case RegionKind::BwdCrossLayer: return arenas.bwd_cross_layer_ptr != nullptr;
            default: return false;
        }
    };

    auto score = [&](const TensorRef& ref, std::size_t& total, std::size_t& covered) {
        ++total;
        auto bv = baked_view(graph, ref);
        if (!bv) return;
        if (!region_allocated(bv->region)) return;
        ++covered;
        const auto region_idx = static_cast<std::size_t>(bv->region);
        if (region_idx < cov.by_region.size()) ++cov.by_region[region_idx];
    };

    for (const auto& op : graph.ops) {
        for (const auto& ref : op.inputs)
            score(ref, cov.total_inputs, cov.covered_inputs);
        for (const auto& ref : op.outputs)
            score(ref, cov.total_outputs, cov.covered_outputs);
    }

    if (const char* env = std::getenv("SUROGATE_DEBUG_OPERAND_COVERAGE")) {
        if (std::string(env) == "1") {
            const std::size_t total = cov.total_inputs + cov.total_outputs;
            const std::size_t covered = cov.covered_inputs + cov.covered_outputs;
            const double pct = total > 0 ? (100.0 * static_cast<double>(covered) / static_cast<double>(total)) : 0.0;
            std::cerr << "[operand-coverage] " << graph.name << ": " << covered << "/" << total << " operands (" << pct
                      << "%) in=" << cov.covered_inputs << "/" << cov.total_inputs << " out=" << cov.covered_outputs
                      << "/" << cov.total_outputs;
            for (std::size_t r = 0; r < cov.by_region.size(); ++r) {
                if (cov.by_region[r] == 0) continue;
                std::cerr << " " << region_kind_name(static_cast<RegionKind>(r)) << "=" << cov.by_region[r];
            }
            std::cerr << "\n";
        }
    }
    return cov;
}

void release_phase_arenas(PhaseArenas& arenas) {
    if (!arenas.allocated) return;
    auto free_ptr = [](std::byte*& p) {
        if (p) {
            cudaFree(p);
            p = nullptr;
        }
    };
    free_ptr(arenas.persistent_ptr);
    free_ptr(arenas.persistent_activation_ptr);
    free_ptr(arenas.model_scope_persistent_ptr);
    free_ptr(arenas.accumulator_ptr);
    free_ptr(arenas.fwd_stack_ptr);
    free_ptr(arenas.bwd_stack_ptr);
    free_ptr(arenas.save_for_bwd_ptr);
    free_ptr(arenas.unified_stack_ptr);
    free_ptr(arenas.bwd_cross_layer_ptr);
    free_ptr(arenas.moe_saved_ptr);
    arenas.persistent_bytes = 0;
    arenas.persistent_activation_bytes = 0;
    arenas.model_scope_persistent_bytes = 0;
    arenas.accumulator_bytes = 0;
    arenas.fwd_stack_bytes = 0;
    arenas.bwd_stack_bytes = 0;
    arenas.save_for_bwd_bytes = 0;
    arenas.unified_stack_bytes = 0;
    arenas.bwd_cross_layer_bytes = 0;
    arenas.moe_saved_bytes = 0;
    arenas.save_for_bwd_block_bases.clear();
    arenas.allocated = false;
}

// ============================================================================
// Debuggability surface (design/buffer-runtime-v4.md, P4.7)
// ============================================================================

std::string CompiledGraph::describe_tensor_id(int tid) const {
    if (tid < 0 || static_cast<std::size_t>(tid) >= tensor_meta.size()) {
        std::ostringstream oss;
        oss << "<tid=" << tid << " unknown>";
        return oss.str();
    }
    const auto& meta = tensor_meta[static_cast<std::size_t>(tid)];
    const auto name = name_for_tensor_id(tid);
    std::ostringstream oss;
    oss << "tid=" << tid;
    if (!name.empty()) oss << " name='" << name << "'";
    oss << " kind=" << tensor_kind_name(meta.kind);
    oss << " region=" << region_kind_name(meta.region);
    if (meta.block_layer_idx >= 0) oss << " block=" << meta.block_layer_idx;
    if (meta.offset != SIZE_MAX) {
        oss << " offset=" << meta.offset;
    }
    if (meta.bytes > 0) {
        oss << " bytes=" << meta.bytes;
    }
    return oss.str();
}

/// Env-gated full tid-table dump (SUROGATE_DEBUG_TID_TABLE=1). Emits one line
/// per tid so a user can grep for a specific tensor's region/offset/size after
/// compile. Tids without a name (external-only references) still emit their
/// meta so they can be diagnosed when they surface in error messages.
static void maybe_dump_tid_table(const CompiledGraph& graph) {
    const char* env = std::getenv("SUROGATE_DEBUG_TID_TABLE");
    if (!env || std::string(env) != "1") return;
    std::cerr << "[tid-table] " << graph.name << " (" << graph.num_tensors << " tids):\n";
    for (int tid = 0; tid < graph.num_tensors; ++tid) {
        std::cerr << "  " << graph.describe_tensor_id(tid) << "\n";
    }
}

void CompiledGraph::compute_layer_segments() {
    const int num_layers = static_cast<int>(layer_start_indices.size());
    layer_segments.resize(static_cast<std::size_t>(num_layers));

    // Build an interval map of MLP tile group op ranges within each layer.
    // These must run eagerly as a group (tiled execution uses dynamic chunk loops).
    // Key: start_op_idx → end_op_idx (inclusive), so the whole group becomes one eager segment.
    std::unordered_map<std::size_t, std::size_t> mlp_tile_starts;  // start → end+1
    for (const auto& tg : mlp_tile_groups) {
        mlp_tile_starts[tg.start_op_idx] = tg.end_op_idx + 1;
    }

    for (int L = 0; L < num_layers; ++L) {
        auto& segs = layer_segments[static_cast<std::size_t>(L)];
        segs.clear();

        const std::size_t start = layer_start_indices[static_cast<std::size_t>(L)];
        const std::size_t end = layer_end_indices[static_cast<std::size_t>(L)];
        if (start == SIZE_MAX || end == SIZE_MAX || start >= end) {
            continue;
        }

        std::size_t seg_start = start;
        for (std::size_t i = start; i < end; ++i) {
            const auto ty = ops[i].type;
            // Graph-breaking ops: must run eagerly because they are
            // capture-unsafe (dynamic cu_seqlens, JIT kernel loading,
            // MoE/EP per-step host bookkeeping, etc.)
            const bool graph_breaking = ty == CompiledOpType::FlashAttention ||
                                        ty == CompiledOpType::FlashAttentionBackward || is_capture_unsafe_op_type(ty);

            // Check if this op starts an MLP tile group
            auto tile_it = mlp_tile_starts.find(i);

            if (graph_breaking) {
                if (i > seg_start) {
                    segs.push_back({seg_start, i, /*eager=*/false});
                }
                segs.push_back({i, i + 1, /*eager=*/true});
                seg_start = i + 1;
            } else if (tile_it != mlp_tile_starts.end()) {
                // MLP tile group: emit as one eager segment
                if (i > seg_start) {
                    segs.push_back({seg_start, i, /*eager=*/false});
                }
                std::size_t tile_end = tile_it->second;
                if (tile_end > end) tile_end = end;
                segs.push_back({i, tile_end, /*eager=*/true});
                i = tile_end - 1;  // loop will ++i
                seg_start = tile_end;
            }
        }
        // Trailing graphable segment
        if (seg_start < end) {
            segs.push_back({seg_start, end, /*eager=*/false});
        }
    }
}

// ============================================================================
// Shape Validation Methods
// ============================================================================

bool GraphCompiler::resolve_tensor_shape(const std::string& name, std::vector<long>& shape) {
    auto format_shape = [](const std::vector<long>& s) -> std::string {
        std::string r = "(";
        for (size_t i = 0; i < s.size(); ++i) {
            if (i > 0) r += ", ";
            r += std::to_string(s[i]);
        }
        r += ")";
        return r;
    };

    // Check shape cache first
    auto it = mTensorShapes.find(name);
    if (it != mTensorShapes.end()) {
        shape = it->second.dims;
        if (mDebugShapes && starts_with(name, "d_")) {
            fprintf(stderr,
                    "[DEBUG_SHAPES] resolve '%s' -> cache %s (src: %s)\n",
                    name.c_str(),
                    format_shape(shape).c_str(),
                    it->second.source_op.c_str());
        }
        return true;
    }

    // Lift per-layer dim overrides into a local helper so every path that
    // writes `mTensorShapes` applies them consistently. The global mShapeEnv
    // has default head_size / qkv_channels / intermediate / mlp_up; for
    // block-scoped tensors on hybrid architectures these need per-layer
    // values.
    auto maybe_override_for_block_scope = [&](std::vector<long>& s) {
        int nli = -1;
        std::string nf;
        std::string strip_name(name);
        if (starts_with(strip_name, "d_")) strip_name = strip_name.substr(2);
        if (starts_with(strip_name, kSavedPrefix)) strip_name = strip_name.substr(kSavedPrefix.size());
        if (parse_block_param(strip_name, nli, nf)) {
            apply_per_layer_dim_override(s, strip_ssa_suffix(nf), nli);
        }
    };

    // Check IR tensor info
    auto check_tensor_info = [&](const std::unordered_map<std::string, TensorInfo>& tensors, const char* source) {
        auto it = tensors.find(name);
        if (it != tensors.end() && !it->second.shape.empty()) {
            shape = resolve_shape(it->second.shape, mShapeEnv);
            maybe_override_for_block_scope(shape);
            TensorShape ts;
            ts.dims = shape;
            ts.inferred = false;
            mTensorShapes[name] = ts;
            if (mDebugShapes && starts_with(name, "d_")) {
                fprintf(stderr,
                        "[DEBUG_SHAPES] resolve '%s' -> IR %s %s\n",
                        name.c_str(),
                        source,
                        format_shape(shape).c_str());
            }
            return true;
        }
        return false;
    };

    // Check in graph tensors
    if (check_tensor_info(mModule.forward->inputs, "fwd.inputs")) return true;
    if (check_tensor_info(mModule.forward->outputs, "fwd.outputs")) return true;
    if (check_tensor_info(mModule.forward->params, "fwd.params")) return true;
    if (check_tensor_info(mModule.forward->intermediates, "fwd.intermediates")) return true;

    // Try pattern-based inference for known tensor names
    if (infer_known_tensor_shape(name, mConfig, mB, mT, shape)) {
        maybe_override_for_block_scope(shape);
        TensorShape ts;
        ts.dims = shape;
        ts.inferred = true;
        mTensorShapes[name] = ts;
        if (mDebugShapes && starts_with(name, "d_")) {
            fprintf(stderr, "[DEBUG_SHAPES] resolve '%s' -> inferred %s\n", name.c_str(), format_shape(shape).c_str());
        }
        return true;
    }

    // Check for saved tensors (use base name)
    if (starts_with(name, kSavedPrefix)) {
        std::string base_name = std::string(name.substr(kSavedPrefix.size()));
        return resolve_tensor_shape(base_name, shape);
    }

    if (mDebugShapes && starts_with(name, "d_")) {
        fprintf(stderr, "[DEBUG_SHAPES] resolve '%s' -> FAILED (no shape found)\n", name.c_str());
    }
    return false;
}

void GraphCompiler::infer_output_shapes(const Operation& op,
                                        CompiledOpType type,
                                        const std::vector<std::vector<long>>& input_shapes,
                                        std::vector<std::vector<long>>& output_shapes) {
    output_shapes.clear();

    // Infer output shapes based on operation type
    switch (type) {
        case CompiledOpType::Matmul:
        case CompiledOpType::MatmulBias: {
            if (input_shapes.size() >= 2 && !input_shapes[0].empty() && !input_shapes[1].empty()) {
                const auto& a_shape = input_shapes[0];
                const auto& b_shape = input_shapes[1];

                // Parse transpose mode
                EMMTranspose mode = parse_transpose(op.attrs);

                // Compute output shape
                std::vector<long> out_shape;

                // Batch dims (min of both inputs)
                size_t min_rank = std::min(a_shape.size(), b_shape.size());
                for (size_t i = 0; i + 2 < min_rank; ++i) {
                    out_shape.push_back(a_shape[i]);
                }

                // M and N dimensions
                if (mode == EMMTranspose::NN || mode == EMMTranspose::NT) {
                    out_shape.push_back(a_shape[a_shape.size() - 2]);  // M
                } else {
                    out_shape.push_back(a_shape[a_shape.size() - 1]);  // M (transposed)
                }

                if (mode == EMMTranspose::NN || mode == EMMTranspose::TN) {
                    out_shape.push_back(b_shape[b_shape.size() - 1]);  // N
                } else {
                    out_shape.push_back(b_shape[b_shape.size() - 2]);  // N (transposed)
                }

                output_shapes.push_back(out_shape);
            }
            break;
        }

        case CompiledOpType::View: {
            // Output shape from attributes ("shape" for forward views,
            // "shape_like" for backward views referencing a forward tensor)
            if (auto* shape_attr = find_attr(op.attrs, "shape")) {
                auto out_shape = resolve_attr_shape(*shape_attr, mShapeEnv);
                output_shapes.push_back(out_shape);
            } else if (auto* shape_like_attr = find_attr(op.attrs, "shape_like")) {
                if (auto ref_name = attr_string(*shape_like_attr)) {
                    std::string ref = *ref_name;
                    if (starts_with(ref, kSavedPrefix)) {
                        ref = ref.substr(kSavedPrefix.size());
                    }
                    std::vector<long> ref_shape;
                    // Try mExtraShapes first (populated from forward view pre-scan)
                    auto it = mExtraShapes.find(ref);
                    if (it != mExtraShapes.end()) {
                        ref_shape = it->second;
                    } else if (!resolve_tensor_shape(ref, ref_shape)) {
                        infer_known_tensor_shape(ref, mConfig, mB, mT, ref_shape);
                    }
                    if (!ref_shape.empty()) {
                        output_shapes.push_back(ref_shape);
                    }
                }
            }
            break;
        }

        case CompiledOpType::Transpose: {
            if (input_shapes.empty() || input_shapes[0].empty()) {
                break;
            }
            auto out_shape = input_shapes[0];
            const int rank = static_cast<int>(out_shape.size());
            int dim0 = 0;
            int dim1 = 1;
            if (auto* a = find_attr(op.attrs, "dim0")) {
                if (auto v = attr_int(*a)) dim0 = static_cast<int>(*v);
            }
            if (auto* a = find_attr(op.attrs, "dim1")) {
                if (auto v = attr_int(*a)) dim1 = static_cast<int>(*v);
            }
            if (dim0 < 0) dim0 += rank;
            if (dim1 < 0) dim1 += rank;
            if (dim0 >= 0 && dim0 < rank && dim1 >= 0 && dim1 < rank && dim0 != dim1) {
                std::swap(out_shape[dim0], out_shape[dim1]);
                output_shapes.push_back(std::move(out_shape));
            }
            break;
        }

        case CompiledOpType::Concat: {
            if (input_shapes.empty() || input_shapes[0].empty()) {
                break;
            }
            const int rank = static_cast<int>(input_shapes[0].size());
            int dim = 0;
            if (auto* dim_attr = find_attr(op.attrs, "dim")) {
                if (auto v = attr_int(*dim_attr)) {
                    dim = static_cast<int>(*v);
                }
            }
            if (dim < 0) dim += rank;
            if (dim < 0 || dim >= rank) {
                break;
            }

            auto out_shape = input_shapes[0];
            long concat_dim = 0;
            bool valid = true;
            for (const auto& in_shape : input_shapes) {
                if (in_shape.size() != static_cast<std::size_t>(rank)) {
                    valid = false;
                    break;
                }
                for (int d = 0; d < rank; ++d) {
                    if (d == dim) continue;
                    if (in_shape[d] != out_shape[d]) {
                        valid = false;
                        break;
                    }
                }
                if (!valid) break;
                concat_dim += in_shape[dim];
            }
            if (valid) {
                out_shape[dim] = concat_dim;
                output_shapes.push_back(std::move(out_shape));
            }
            break;
        }

        case CompiledOpType::Split: {
            if (input_shapes.empty() || input_shapes[0].empty()) {
                break;
            }
            const auto& in_shape = input_shapes[0];
            const int rank = static_cast<int>(in_shape.size());
            int dim = 0;
            if (auto* dim_attr = find_attr(op.attrs, "dim")) {
                if (auto v = attr_int(*dim_attr)) {
                    dim = static_cast<int>(*v);
                }
            }
            if (dim < 0) dim += rank;
            if (dim < 0 || dim >= rank) {
                break;
            }

            std::vector<long> split_sizes;
            if (auto* split_attr = find_attr(op.attrs, "split_size")) {
                if (auto list = attr_list_int(*split_attr)) {
                    split_sizes = *list;
                } else if (auto v = attr_int(*split_attr)) {
                    const long chunk = static_cast<long>(*v);
                    if (chunk > 0) {
                        long rem = in_shape[dim];
                        while (rem > 0) {
                            const long take = std::min(chunk, rem);
                            split_sizes.push_back(take);
                            rem -= take;
                        }
                    }
                }
            }

            if (split_sizes.empty() && !op.outputs.empty()) {
                if (in_shape[dim] % static_cast<long>(op.outputs.size()) == 0) {
                    split_sizes.assign(op.outputs.size(), in_shape[dim] / static_cast<long>(op.outputs.size()));
                }
            }

            for (std::size_t i = 0; i < op.outputs.size(); ++i) {
                if (i >= split_sizes.size()) {
                    output_shapes.push_back({});
                    continue;
                }
                auto out_shape = in_shape;
                out_shape[dim] = split_sizes[i];
                output_shapes.push_back(std::move(out_shape));
            }
            break;
        }

        case CompiledOpType::Add: {
            // Output shape = broadcast(input shapes)
            if (!input_shapes.empty()) {
                output_shapes.push_back(input_shapes[0]);  // Simplified: assume same shape
            }
            break;
        }

        case CompiledOpType::Narrow: {
            // Output = input shape with narrow_dim replaced by length.
            if (input_shapes.empty() || input_shapes[0].empty()) break;
            const auto& in_shape = input_shapes[0];
            const int rank = static_cast<int>(in_shape.size());
            int dim = 0;
            long length = 0;
            if (auto* dim_attr = find_attr(op.attrs, "dim")) {
                if (auto v = attr_int(*dim_attr)) dim = static_cast<int>(*v);
            }
            if (auto* len_attr = find_attr(op.attrs, "length")) {
                if (auto v = attr_int(*len_attr)) length = *v;
            }
            if (dim < 0) dim += rank;
            if (dim < 0 || dim >= rank || length <= 0) break;
            auto out_shape = in_shape;
            out_shape[dim] = length;
            output_shapes.push_back(std::move(out_shape));
            break;
        }

        case CompiledOpType::SwiGLU: {
            // Output last dim = input last dim / 2
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                auto out_shape = input_shapes[0];
                out_shape.back() = out_shape.back() / 2;
                output_shapes.push_back(out_shape);
            }
            break;
        }

        case CompiledOpType::Embedding: {
            // Output = indices_shape + [embedding_dim]
            if (input_shapes.size() >= 2 && !input_shapes[1].empty()) {
                auto out_shape = input_shapes[0];         // indices shape
                out_shape.push_back(input_shapes[1][1]);  // embedding dim
                output_shapes.push_back(out_shape);
            }
            break;
        }

        case CompiledOpType::CrossEntropyLoss: {
            // Output: per-token loss [B*T]
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                const auto& logits_shape = input_shapes[0];
                if (!logits_shape.empty()) {
                    output_shapes.push_back({logits_shape[0]});
                }
            }
            break;
        }

        case CompiledOpType::CrossEntropyLossBackward: {
            // Output: d_logits shape matches logits input
            if (input_shapes.size() > 1 && !input_shapes[1].empty()) {
                output_shapes.push_back(input_shapes[1]);
            }
            break;
        }

        case CompiledOpType::FusedLMHeadLoss: {
            // Output: per-token loss [B*T]
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                output_shapes.push_back({input_shapes[0][0]});
            }
            break;
        }

        case CompiledOpType::FusedLMHeadLossBackward: {
            // Outputs: d_xF_flat [B*T, C], d_lm_head [V, C]
            if (input_shapes.size() > 1 && !input_shapes[1].empty()) {
                output_shapes.push_back(input_shapes[1]);
            }
            if (input_shapes.size() > 2 && !input_shapes[2].empty()) {
                output_shapes.push_back(input_shapes[2]);
            }
            break;
        }

        case CompiledOpType::Zeros:
        case CompiledOpType::Ones: {
            // Try to infer from 'shape' attribute
            if (auto* shape_attr = find_attr(op.attrs, "shape")) {
                auto out_shape = resolve_attr_shape(*shape_attr, mShapeEnv);
                output_shapes.push_back(out_shape);
            } else if (auto* shape_like_attr = find_attr(op.attrs, "shape_like")) {
                if (auto ref_name = attr_string(*shape_like_attr)) {
                    std::string ref = *ref_name;
                    if (starts_with(ref, kSavedPrefix)) {
                        ref = ref.substr(kSavedPrefix.size());
                    }
                    std::vector<long> ref_shape;
                    auto it = mExtraShapes.find(ref);
                    if (it != mExtraShapes.end()) {
                        ref_shape = it->second;
                    } else if (!resolve_tensor_shape(ref, ref_shape)) {
                        infer_known_tensor_shape(ref, mConfig, mB, mT, ref_shape);
                    }
                    if (!ref_shape.empty()) {
                        output_shapes.push_back(std::move(ref_shape));
                    }
                }
            }
            break;
        }

        case CompiledOpType::RoPE: {
            // RoPE output shape matches input qkv shape
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                output_shapes.push_back(input_shapes[0]);
            }
            break;
        }

        case CompiledOpType::FlashAttention: {
            // FlashAttention outputs: attn_out [B, T, Hq, D], lse [B, Hq, T]
            // Cannot infer output shape from input qkv [B, T, Hq+2*Hkv, D] without
            // knowing Hq and Hkv separately. Leave shapes uninferred.
            break;
        }

        case CompiledOpType::FusedResidualRMSNorm: {
            // Outputs: residual_out [B,T,C], y [B,T,C], rstd [B,T]
            if (input_shapes.size() >= 2 && !input_shapes[0].empty() && !input_shapes[1].empty()) {
                output_shapes.push_back(input_shapes[0]);  // residual_out same as input[0]
                output_shapes.push_back(input_shapes[1]);  // y same as input[1]
                // rstd drops the last dimension
                auto rstd_shape = input_shapes[0];
                if (!rstd_shape.empty()) {
                    rstd_shape.pop_back();
                }
                output_shapes.push_back(rstd_shape);
            }
            break;
        }

        case CompiledOpType::RMSNorm:
        case CompiledOpType::LayerNorm: {
            // Outputs: y [same as input x], rstd [input rows] (last dim dropped)
            // Used for standalone (non-fused) norm ops — e.g., Gemma4 per-head
            // Q/K/V norms with 2D [B*T*H, D] or 3D [B,T,C] input.
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                output_shapes.push_back(input_shapes[0]);  // y same as input[0]
                auto rstd_shape = input_shapes[0];
                rstd_shape.pop_back();
                output_shapes.push_back(rstd_shape);
            }
            break;
        }

        case CompiledOpType::Silu:
        case CompiledOpType::Relu2:
        case CompiledOpType::Mul:
        case CompiledOpType::SiluBackward:
        case CompiledOpType::Relu2Backward: {
            // Element-wise ops (and their backward) preserve shape
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                output_shapes.push_back(input_shapes[0]);
            }
            break;
        }

        case CompiledOpType::MulBackward: {
            // Inputs: d_out, a, b
            // Outputs: d_a, d_b
            if (input_shapes.size() >= 3) {
                if (!input_shapes[1].empty()) output_shapes.push_back(input_shapes[1]);
                if (!input_shapes[2].empty()) output_shapes.push_back(input_shapes[2]);
            } else if (!input_shapes.empty() && !input_shapes[0].empty()) {
                output_shapes.push_back(input_shapes[0]);
            }
            break;
        }

        case CompiledOpType::QKVQKNorm: {
            // Output qkv_norm has same shape as input qkv
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                output_shapes.push_back(input_shapes[0]);  // qkv_norm
                // q_rstd and k_rstd shapes - hard to infer without config
                output_shapes.push_back({});
                output_shapes.push_back({});
            }
            break;
        }

        case CompiledOpType::QKVQKNormRoPE: {
            // Output qkv_rope has same shape as input qkv
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                output_shapes.push_back(input_shapes[0]);  // qkv_rope
                // q_rstd and k_rstd shapes - hard to infer without config
                output_shapes.push_back({});
                output_shapes.push_back({});
            }
            break;
        }

        case CompiledOpType::MoESigmoid:
        case CompiledOpType::MoESoftmax: {
            // Output same shape as input
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                output_shapes.push_back(input_shapes[0]);
            }
            break;
        }

        case CompiledOpType::MoETopK: {
            // Output: routing_weights [B*T, K], routing_indices [B*T, K]
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                int top_k = 1;
                if (auto* attr = find_attr(op.attrs, "top_k")) {
                    if (auto v = attr_int(*attr)) {
                        top_k = static_cast<int>(*v);
                    }
                }
                std::vector<long> out_shape = {input_shapes[0][0], static_cast<long>(top_k)};
                output_shapes.push_back(out_shape);  // routing_weights
                output_shapes.push_back(out_shape);  // routing_indices
            }
            break;
        }

        case CompiledOpType::MoEPermute: {
            // permuted_input shape depends on scatter_indices, hard to infer statically
            break;
        }

        case CompiledOpType::MoEGroupedGemmGateUp: {
            // Output shape is [total_tokens, 2*M] but total_tokens is dynamic
            break;
        }

        case CompiledOpType::MoEGroupedGemmDown: {
            // Output shape is [total_tokens, C] but total_tokens is dynamic
            break;
        }

        case CompiledOpType::MoEUnpermute: {
            // Output shape [B*T, C] - based on routing structure
            break;
        }

        // Expert Parallelism operations (dynamic shapes)
        case CompiledOpType::EpDispatch: {
            // Output shape is variable (worst case: all tokens to this GPU)
            break;
        }
        case CompiledOpType::EpCombine: {
            // Output shape matches original permuted token count
            break;
        }

        // Mamba/SSM operations
        case CompiledOpType::MambaSplitProj: {
            // Outputs: gate [B, T, intermediate_size], conv_in [B, conv_dim, T], dt [B, T, num_heads]
            // Cannot fully infer without attributes, leave empty for runtime
            break;
        }

        case CompiledOpType::MambaConv1d: {
            // Output shape same as input (causal conv1d)
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                output_shapes.push_back(input_shapes[0]);
            }
            break;
        }

        case CompiledOpType::MambaSplitConvOut: {
            // Outputs: u [B, D, T], B [B, G, N, T], C [B, G, N, T]
            // Cannot fully infer without attributes, leave empty for runtime
            break;
        }

        case CompiledOpType::MambaSsmScan: {
            // Outputs: out [B, T, H, D], ssm_state [B, H, D, N]
            // Cannot fully infer without attributes, leave empty for runtime
            break;
        }

        case CompiledOpType::MambaGatedRMSNorm: {
            // Output same shape as input x (gated output)
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                output_shapes.push_back(input_shapes[0]);
            }
            break;
        }

        case CompiledOpType::MambaOutProj: {
            // Standard matmul output shape
            if (input_shapes.size() >= 2 && !input_shapes[0].empty() && !input_shapes[1].empty()) {
                // Same as Matmul
                const auto& a_shape = input_shapes[0];
                const auto& b_shape = input_shapes[1];
                EMMTranspose mode = parse_transpose(op.attrs);
                std::vector<long> out_shape;
                if (mode == EMMTranspose::NT || mode == EMMTranspose::NN) {
                    out_shape.push_back(a_shape[a_shape.size() - 2]);
                } else {
                    out_shape.push_back(a_shape[a_shape.size() - 1]);
                }
                if (mode == EMMTranspose::NT || mode == EMMTranspose::TT) {
                    out_shape.push_back(b_shape[b_shape.size() - 2]);
                } else {
                    out_shape.push_back(b_shape[b_shape.size() - 1]);
                }
                output_shapes.push_back(out_shape);
            }
            break;
        }

        case CompiledOpType::ChunkGatedDeltaRule: {
            // Inputs:
            //   q [B, T, H, K], k [B, T, H, K], v [B, T, H, V], g [B, T, H], beta [B, T, H]
            //   initial_state [B, H, K, V] (optional)
            // Outputs:
            //   out [B, T, H, V], final_state [B, H, K, V] (optional by caller contract)
            if (input_shapes.size() >= 3 && !input_shapes[0].empty() && !input_shapes[2].empty() &&
                input_shapes[0].size() == 4 && input_shapes[2].size() == 4) {
                const auto& q_shape = input_shapes[0];
                const auto& v_shape = input_shapes[2];
                output_shapes.push_back({q_shape[0], q_shape[1], q_shape[2], v_shape[3]});
                output_shapes.push_back({q_shape[0], q_shape[2], q_shape[3], v_shape[3]});
            }
            break;
        }

        case CompiledOpType::ChunkGatedDeltaRuleBackward: {
            // Inputs:
            //   d_out [B,T,H,V], d_final_state [B,H,K,V] (optional), q [B,T,H,K], k [B,T,H,K],
            //   v [B,T,H,V], g [B,T,H], beta [B,T,H], initial_state [B,H,K,V] (optional)
            // Outputs:
            //   d_q, d_k, d_v, d_g, d_beta, d_initial_state
            if (input_shapes.size() >= 7 && input_shapes[2].size() == 4 && input_shapes[4].size() == 4 &&
                input_shapes[5].size() == 3 && input_shapes[6].size() == 3) {
                const auto& q_shape = input_shapes[2];
                const auto& v_shape = input_shapes[4];
                const auto& g_shape = input_shapes[5];
                output_shapes.push_back(q_shape);                                           // d_q
                output_shapes.push_back(q_shape);                                           // d_k
                output_shapes.push_back(v_shape);                                           // d_v
                output_shapes.push_back(g_shape);                                           // d_g
                output_shapes.push_back(g_shape);                                           // d_beta
                output_shapes.push_back({q_shape[0], q_shape[2], q_shape[3], v_shape[3]});  // d_initial_state
            }
            break;
        }

        case CompiledOpType::Qwen3_5Decay: {
            // Output shape same as input `a` => [B,T,H]
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                output_shapes.push_back(input_shapes[0]);
            }
            break;
        }

        case CompiledOpType::Qwen3_5DecayBackward: {
            // Inputs: d_out, a, A_log, dt_bias
            // Outputs: d_a, d_A_log, d_dt_bias
            if (input_shapes.size() >= 4) {
                if (!input_shapes[1].empty()) output_shapes.push_back(input_shapes[1]);
                if (!input_shapes[2].empty()) output_shapes.push_back(input_shapes[2]);
                if (!input_shapes[3].empty()) output_shapes.push_back(input_shapes[3]);
            }
            break;
        }

        case CompiledOpType::RepeatInterleaveHeads: {
            // Input: [B,T,H,D], Output: [B,T,H*repeats,D]
            if (!input_shapes.empty() && input_shapes[0].size() == 4) {
                auto out_shape = input_shapes[0];
                int repeats = 1;
                if (auto* attr = find_attr(op.attrs, "repeats")) {
                    if (auto v = attr_int(*attr)) {
                        repeats = static_cast<int>(*v);
                    }
                }
                if (repeats <= 0) repeats = 1;
                out_shape[2] *= repeats;
                output_shapes.push_back(std::move(out_shape));
            }
            break;
        }

        case CompiledOpType::RepeatInterleaveHeadsBackward: {
            // Inputs: d_out, inp
            // Output: d_inp (same shape as inp)
            if (input_shapes.size() >= 2 && !input_shapes[1].empty()) {
                output_shapes.push_back(input_shapes[1]);
            }
            break;
        }

        default:
            // For other operations, output shape not inferred
            break;
    }
}

void GraphCompiler::validate_operation_shapes(const Operation& op, CompiledOpType type, size_t op_index) {
    using namespace shape_checker;

    // Get operation signature
    const auto* sig = OpShapeRegistry::instance().get_signature(op.name);
    if (!sig) {
        // No signature registered - skip validation (only warn in verbose mode)
        return;
    }

    // Resolve input shapes
    std::vector<std::vector<long>> input_shapes;
    input_shapes.reserve(op.inputs.size());
    std::vector<std::string> unresolved_inputs;

    for (const auto& input_name : op.inputs) {
        std::vector<long> shape;
        if (!resolve_tensor_shape(input_name, shape)) {
            unresolved_inputs.push_back(input_name);
            input_shapes.push_back({});  // Empty shape
        } else {
            input_shapes.push_back(shape);
        }
    }

    // If we couldn't resolve some input shapes, we can't validate
    if (!unresolved_inputs.empty()) {
        if (mDebugShapes && starts_with(op.name, "matmul")) {
            fprintf(stderr,
                    "[DEBUG_SHAPES] validate '%s' (id: %s) SKIPPED — unresolved inputs:",
                    op.name.c_str(),
                    op.id.c_str());
            for (const auto& u : unresolved_inputs) {
                fprintf(stderr, " '%s'", u.c_str());
            }
            fprintf(stderr, "\n");
        }
        return;
    }

    // Resolve or infer output shapes
    std::vector<std::vector<long>> output_shapes;
    output_shapes.reserve(op.outputs.size());

    for (size_t i = 0; i < op.outputs.size(); ++i) {
        const auto& output_name = op.outputs[i];
        std::vector<long> shape;

        if (resolve_tensor_shape(output_name, shape)) {
            // Shape already known (from IR or previous inference)
            output_shapes.push_back(shape);
        } else {
            // Try to infer from operation semantics
            std::vector<std::vector<long>> inferred_outputs;
            infer_output_shapes(op, type, input_shapes, inferred_outputs);

            if (i < inferred_outputs.size() && !inferred_outputs[i].empty()) {
                shape = inferred_outputs[i];
                output_shapes.push_back(shape);

                // Store inferred shape for future operations
                TensorShape ts;
                ts.dims = shape;
                ts.inferred = true;
                ts.source_op = op.id;
                mTensorShapes[output_name] = ts;
            } else {
                output_shapes.push_back({});  // Unknown shape
            }
        }
    }

    // Run validator
    if (sig->validator) {
        auto error = sig->validator(input_shapes, output_shapes, op.attrs, mShapeEnv);
        if (error) {
            // Build detailed error message
            std::ostringstream oss;
            oss << "\n╔═══════════════════════════════════════════════════════╗\n"
                << "║ Found Shape Validation Error during Graph Compilation ║\n"
                << "╚═══════════════════════════════════════════════════════╝\n\n"
                << "Operation: #" << op_index << " (id: '" << op.id << "')\n"
                << "Type:      " << op.name << "\n\n";

            // Show operation attributes if any
            bool has_attrs = false;
            std::ostringstream attrs_oss;
            if (op.attrs.find("transpose") != op.attrs.end()) {
                if (std::holds_alternative<std::string>(op.attrs.at("transpose").value)) {
                    attrs_oss << "transpose=" << std::get<std::string>(op.attrs.at("transpose").value) << " ";
                    has_attrs = true;
                }
            }
            if (op.attrs.find("eps") != op.attrs.end()) {
                if (std::holds_alternative<double>(op.attrs.at("eps").value)) {
                    attrs_oss << "eps=" << std::get<double>(op.attrs.at("eps").value) << " ";
                    has_attrs = true;
                }
            }
            if (op.attrs.find("rotary_dim") != op.attrs.end()) {
                if (std::holds_alternative<std::int64_t>(op.attrs.at("rotary_dim").value)) {
                    attrs_oss << "rotary_dim=" << std::get<std::int64_t>(op.attrs.at("rotary_dim").value) << " ";
                    has_attrs = true;
                }
            }
            if (op.attrs.find("layer_idx") != op.attrs.end()) {
                if (std::holds_alternative<std::int64_t>(op.attrs.at("layer_idx").value)) {
                    attrs_oss << "layer_idx=" << std::get<std::int64_t>(op.attrs.at("layer_idx").value) << " ";
                    has_attrs = true;
                }
            }
            if (has_attrs) {
                oss << "Attributes: " << attrs_oss.str() << "\n\n";
            }

            oss << "Inputs:\n";
            if (op.inputs.empty()) {
                oss << "  (none)\n";
            } else {
                for (size_t i = 0; i < op.inputs.size(); ++i) {
                    oss << "  [" << i << "] " << op.inputs[i] << ": ";
                    if (i < input_shapes.size() && !input_shapes[i].empty()) {
                        oss << "shape=(";
                        for (size_t j = 0; j < input_shapes[i].size(); ++j) {
                            if (j > 0) oss << ", ";
                            oss << input_shapes[i][j];
                        }
                        oss << ")";
                    } else {
                        oss << "<shape unknown>";
                    }
                    oss << "\n";
                }
            }

            oss << "\nOutputs:\n";
            for (size_t i = 0; i < op.outputs.size(); ++i) {
                oss << "  [" << i << "] " << op.outputs[i] << ": ";
                if (i < output_shapes.size() && !output_shapes[i].empty()) {
                    oss << "shape=(";
                    for (size_t j = 0; j < output_shapes[i].size(); ++j) {
                        if (j > 0) oss << ", ";
                        oss << output_shapes[i][j];
                    }
                    oss << ")";
                } else {
                    oss << "<shape unknown or not inferred>";
                }
                oss << "\n";
            }

            oss << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                << "ERROR: " << error->message << "\n";

            if (!error->hint.empty()) {
                oss << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    << "HINT:  " << error->hint << "\n";
            }

            oss << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                << "Debug Information:\n"
                << "  Graph: " << mModule.name << "\n"
                << "  Batch size (B): " << mB << "\n"
                << "  Sequence length (T): " << mT << "\n"
                << "  Hidden size: " << mConfig.HiddenSize << "\n\n";

            throw std::runtime_error(oss.str());
        }
    }
}

int GraphCompiler::assign_tensor_id(const std::string& name) {
    auto [it, inserted] = mTensorIdMap.emplace(name, mNextTensorId);
    if (inserted) mNextTensorId++;
    return it->second;
}

void GraphCompiler::register_external_names(CompiledGraph& graph) {
    // Register well-known tensor names that are bound during execute_forward/backward init
    // but may not appear in any op's TensorRef (e.g., they are injected before the dispatch loop).
    static const char* const kForwardNames[] = {
        "token_ids",
        "position_ids",
        "visual_pos_masks",
        "visual_embeds",
        "x0",
    };
    for (const char* name : kForwardNames) {
        assign_tensor_id(name);
    }

    // Backward init bindings
    static const char* const kBackwardNames[] = {
        "d_logits",
        "d_logits_flat",
        "d_xF_flat",
        "d_xF",
        "d_ln_final",
        "d_ln_final_flat",
        "d_encoded",
        "d_x0",
        "d_xN",
        "d_residualN",
    };
    for (const char* name : kBackwardNames) {
        assign_tensor_id(name);
    }

    // MoE side-channel tensors (produced by moe_permute, consumed by grouped_gemm ops)
    if (mConfig.NumExperts > 0) {
        assign_tensor_id("moe_expert_offsets");
        assign_tensor_id("moe_gather_indices");
    }

    // Deepstack visual embed tensors (dynamically named)
    if (mConfig.DeepstackVisualLayers > 0) {
        for (int i = 0; i < mConfig.DeepstackVisualLayers; ++i) {
            assign_tensor_id("deepstack_visual_embeds_" + std::to_string(i));
        }
    }

    // Parameter gradient tensors (d_<param_name>) — bound during backward init
    for (const auto& pname : mGrads.param_names()) {
        assign_tensor_id("d_" + pname);
    }
}

void GraphCompiler::build_tensor_metadata(CompiledGraph& graph) {
    graph.num_tensors = mNextTensorId;
    graph.tensor_name_to_id = mTensorIdMap;
    graph.tensor_meta.resize(static_cast<std::size_t>(mNextTensorId));
    // Build the reverse (tid -> name) vector so hot-path lookups (e.g.
    // `base_param_from_grad_kind` recovering a param name from `base_param_tid`)
    // are O(1) instead of scanning tensor_name_to_id. Names are copied once
    // here; CompiledGraph lifetime dominates dispatcher calls.
    graph.tensor_id_to_name.assign(static_cast<std::size_t>(mNextTensorId), std::string{});
    for (const auto& [name, id] : mTensorIdMap) {
        if (id >= 0 && static_cast<std::size_t>(id) < graph.tensor_id_to_name.size()) {
            graph.tensor_id_to_name[static_cast<std::size_t>(id)] = name;
        }
    }

    for (const auto& [name, id] : mTensorIdMap) {
        TensorMeta meta;

        // Check "layer" prefix (cross-layer connector tensors)
        if (name.rfind("layer", 0) == 0) {
            meta.flags |= TensorMeta::kCrossLayer;
        }

        // Check MoE special names
        if (name == "moe_expert_offsets") {
            meta.flags |= TensorMeta::kMoeOffsets;
        }
        if (name == "moe_gather_indices") {
            meta.flags |= TensorMeta::kMoeGather;
        }

        // Check "d_blocks[N]." or "d_layerN." pattern (gradient block tensors)
        if (name.rfind("d_blocks[", 0) == 0) {
            meta.flags |= TensorMeta::kDBlocks;
            auto bracket_pos = name.find('[');
            auto close_pos = name.find(']');
            if (bracket_pos != std::string::npos && close_pos != std::string::npos && close_pos > bracket_pos) {
                try {
                    meta.block_layer_idx = std::stoi(name.substr(bracket_pos + 1, close_pos - bracket_pos - 1));
                } catch (...) {
                    meta.block_layer_idx = -1;
                }
            }
        } else if (name.rfind("d_layer", 0) == 0) {
            // "d_layer{N}.xxx" — gradient of cross-layer connector
            meta.flags |= TensorMeta::kDBlocks;
            auto dot_pos = name.find('.', 7);  // skip "d_layer"
            if (dot_pos != std::string::npos && dot_pos > 7) {
                try {
                    meta.block_layer_idx = std::stoi(name.substr(7, dot_pos - 7));
                } catch (...) {
                    meta.block_layer_idx = -1;
                }
            }
        }
        // Check "blocks[N]." or "layerN." pattern (non-gradient block tensors)
        else if (name.rfind("blocks[", 0) == 0) {
            meta.flags |= TensorMeta::kBlocks;
            auto bracket_pos = name.find('[');
            auto close_pos = name.find(']');
            if (bracket_pos != std::string::npos && close_pos != std::string::npos && close_pos > bracket_pos) {
                try {
                    meta.block_layer_idx = std::stoi(name.substr(bracket_pos + 1, close_pos - bracket_pos - 1));
                } catch (...) {
                    meta.block_layer_idx = -1;
                }
            }
        } else if (name.rfind("layer", 0) == 0 && name.size() > 5 && std::isdigit(name[5])) {
            // "layer{N}.xxx" — cross-layer connector with parseable layer index
            meta.flags |= TensorMeta::kBlocks;
            auto dot_pos = name.find('.', 5);  // skip "layer"
            if (dot_pos != std::string::npos && dot_pos > 5) {
                try {
                    meta.block_layer_idx = std::stoi(name.substr(5, dot_pos - 5));
                } catch (...) {
                    meta.block_layer_idx = -1;
                }
            }
        }

        meta.role = infer_tensor_role_from_name(name, meta.block_layer_idx);
        meta.role.block_slot = static_cast<int>(resolve_block_slot(name));
        graph.tensor_meta[static_cast<std::size_t>(id)] = meta;
    }

    // Build SSA-stripped name -> highest-suffix tensor_id map
    for (const auto& [name, id] : mTensorIdMap) {
        const std::string base = strip_ssa_suffix(name);
        auto [it, inserted] = graph.ssa_base_to_id.emplace(base, id);
        if (!inserted) {
            // Keep the highest SSA suffix ID (which is the latest version)
            // Compare suffix values to determine which is "highest"
            const auto& existing_name = [&]() -> const std::string& {
                for (const auto& [n, i] : mTensorIdMap) {
                    if (i == it->second) return n;
                }
                return name;  // fallback
            }();
            // Simple heuristic: higher tensor_id = later in compilation = latest SSA version
            if (id > it->second) {
                it->second = id;
            }
        }
    }
}

// ============================================================================
// Tensor classification (Phase 0 of the TensorKind rollout)
// ============================================================================
//
// Populates TensorMeta::kind and the base_*_tid fields once at compile time
// using authoritative lookups (parameter store + per-tid op producers). This
// replaces runtime string predicates (base_param_from_grad,
// should_alias_autodiff_accum_name, etc.) that historically misclassified
// intermediate gradients as parameter gradients, causing accumulator temps
// to collide with their base tensor's buffer. Phase 0 only populates data;
// callers still use the legacy string predicates until Phase 1 flips them.

const char* tensor_kind_name(TensorKind k) {
    switch (k) {
        case TensorKind::Unknown: return "Unknown";
        case TensorKind::ForwardParam: return "ForwardParam";
        case TensorKind::ForwardActivation: return "ForwardActivation";
        case TensorKind::ParamGrad: return "ParamGrad";
        case TensorKind::ActivationGrad: return "ActivationGrad";
        case TensorKind::AccumTemp: return "AccumTemp";
        case TensorKind::LossInput: return "LossInput";
        case TensorKind::Scratch: return "Scratch";
    }
    return "Unknown";
}

namespace {

// Strip a trailing `_from_<N>` or `_accum_<N>` tag (autodiff accumulator
// variants). Returns (stripped_name, had_tag). Only strips when the tag is
// followed by digits and nothing else, so we don't accidentally chop a
// legitimate op id that happens to contain "_from_".
std::pair<std::string, bool> strip_autodiff_accum_tag(const std::string& name) {
    for (const char* tag : {"_from_", "_accum_"}) {
        const std::size_t taglen = std::strlen(tag);
        auto pos = name.find(tag);
        if (pos == std::string::npos) continue;
        const std::size_t after = pos + taglen;
        if (after >= name.size()) continue;
        bool all_digits = true;
        for (std::size_t i = after; i < name.size(); ++i) {
            if (!std::isdigit(static_cast<unsigned char>(name[i]))) {
                all_digits = false;
                break;
            }
        }
        if (all_digits) {
            return {name.substr(0, pos), true};
        }
    }
    return {name, false};
}

}  // namespace

void GraphCompiler::classify_tensors(CompiledGraph& graph) {
    // Build a tid -> producing op index map from the ops in this graph. We use
    // this to confirm that a tensor classified as ForwardActivation is indeed
    // produced by a forward op in the current compile (forward graphs only).
    // For backward graphs this map covers backward op outputs; we still run
    // the classifier over every tid, and fall back to name-based rules for
    // tids whose producer isn't visible in the current graph (cross-graph
    // references — forward activations referenced by backward).
    std::vector<int> producer_op(static_cast<std::size_t>(graph.num_tensors), -1);
    for (std::size_t op_idx = 0; op_idx < graph.ops.size(); ++op_idx) {
        for (const auto& ref : graph.ops[op_idx].outputs) {
            if (ref.tensor_id >= 0 && ref.tensor_id < graph.num_tensors) {
                producer_op[static_cast<std::size_t>(ref.tensor_id)] = static_cast<int>(op_idx);
            }
        }
    }

    auto tid_for = [&](const std::string& name) -> int {
        auto it = graph.tensor_name_to_id.find(name);
        return (it != graph.tensor_name_to_id.end()) ? it->second : -1;
    };

    // Pass 1: forward-side classification. A tid is a ForwardParam iff the
    // parameter store owns it. A tid is a ForwardActivation iff it's produced
    // by a non-gradient op in this graph AND isn't a parameter.
    for (const auto& [name, id] : graph.tensor_name_to_id) {
        if (id < 0 || static_cast<std::size_t>(id) >= graph.tensor_meta.size()) continue;
        auto& meta = graph.tensor_meta[static_cast<std::size_t>(id)];

        const bool is_grad_name = starts_with(name, "d_");

        if (!is_grad_name && mWeights.has(name)) {
            meta.kind = TensorKind::ForwardParam;
            continue;
        }

        if (!is_grad_name && producer_op[static_cast<std::size_t>(id)] >= 0) {
            // Produced by an op in this (forward) graph and not a parameter.
            meta.kind = TensorKind::ForwardActivation;
            meta.base_producer_tid = id;  // self-reference; its own op is the producer
        }
    }

    // Pass 2: gradient-side classification. For every tid whose name starts
    // with "d_", try to resolve the base it points at.
    for (const auto& [name, id] : graph.tensor_name_to_id) {
        if (id < 0 || static_cast<std::size_t>(id) >= graph.tensor_meta.size()) continue;
        auto& meta = graph.tensor_meta[static_cast<std::size_t>(id)];
        if (!starts_with(name, "d_")) continue;

        // Special-case loss inputs.
        if (name == "d_loss") {
            meta.kind = TensorKind::LossInput;
            continue;
        }

        // Check for _from_N / _accum_N accumulator suffix first. These are
        // ALWAYS distinct tensors from their base — the whole point of this
        // classification is to prevent them from collapsing onto the base.
        auto [base_grad_name, has_accum_tag] = strip_autodiff_accum_tag(name);
        if (has_accum_tag) {
            const int base_grad_tid = tid_for(base_grad_name);
            meta.kind = TensorKind::AccumTemp;
            meta.base_grad_tid = base_grad_tid;  // -1 if base isn't in this graph
            // Still record the ultimate param/activation it targets, if resolvable.
            // (We don't need it for correctness — AccumTemp is enough — but it's
            // useful telemetry.)
            const std::string stripped = base_grad_name.substr(2);  // strip "d_"
            if (mWeights.has(stripped)) {
                meta.base_param_tid = tid_for(stripped);
            }
            continue;
        }

        // Plain d_<name>: resolve what <name> refers to.
        const std::string stripped = name.substr(2);

        if (mWeights.has(stripped)) {
            meta.kind = TensorKind::ParamGrad;
            meta.base_param_tid = tid_for(stripped);
            continue;
        }

        // Forward activation grad. We prefer to verify via a tid whose kind
        // is ForwardActivation, but in a pure backward graph that tid may
        // not exist here. Fall back to: "name exists as a tid" OR
        // "producer_op unknown but name starts with common forward prefix".
        const int fwd_tid = tid_for(stripped);
        if (fwd_tid >= 0) {
            const auto& fwd_meta = graph.tensor_meta[static_cast<std::size_t>(fwd_tid)];
            if (fwd_meta.kind == TensorKind::ForwardActivation || fwd_meta.kind == TensorKind::ForwardParam) {
                // ForwardParam fallthrough handled above; if we get here the
                // forward-side classifier missed it (e.g. weight not in
                // mWeights yet). Treat conservatively.
                meta.kind =
                    (fwd_meta.kind == TensorKind::ForwardParam) ? TensorKind::ParamGrad : TensorKind::ActivationGrad;
                if (meta.kind == TensorKind::ParamGrad) {
                    meta.base_param_tid = fwd_tid;
                } else {
                    meta.base_producer_tid = fwd_tid;
                }
                continue;
            }
        }

        // No forward tid visible in this graph. Consult authoritative sources:
        //   (1) GlobalRole table — recognized global tensors (xF, encoded, xN, …).
        //   (2) Slot registry — recognized block-level slot (ln1, qkv, mlp_up, …).
        // Anything else stays Scratch. This replaces the earlier hardcoded
        // prefix list — new global tensors are added to global_role_for_name()
        // (single place), and new block slots are added to the slot registry.
        const GlobalRole role = global_role_for_name(stripped);
        bool recognized = (role != GlobalRole::None);
        if (!recognized) {
            // Strip block prefix to probe slot registry by field name.
            int lid = -1;
            std::string field;
            const std::string field_name =
                parse_block_param(stripped, lid, field) ? strip_ssa_suffix(field) : strip_ssa_suffix(stripped);
            auto slot_entry = mSlotRegistry.lookup(field_name);
            if (slot_entry.has_value() && slot_entry->slot != TensorSlot::Mapped &&
                slot_entry->slot != TensorSlot::Saved && slot_entry->slot != TensorSlot::Parameter) {
                recognized = true;
            }
        }
        meta.kind = recognized ? TensorKind::ActivationGrad : TensorKind::Scratch;
    }

    // Pass 3: everything still Unknown that's neither a gradient nor a param
    // producer gets classified as Scratch (constants, zeros, masks, etc.).
    for (std::size_t i = 0; i < graph.tensor_meta.size(); ++i) {
        if (graph.tensor_meta[i].kind == TensorKind::Unknown) {
            graph.tensor_meta[i].kind = TensorKind::Scratch;
        }
        graph.tensor_meta[i].role.kind = role_kind_from_tensor_kind(graph.tensor_meta[i].kind);
    }

    // Optional: dump classification on demand for debugging and for comparing
    // against the legacy string predicates during the rollout.
    const char* dump_env = std::getenv("SUROGATE_DEBUG_TENSOR_KIND");
    const bool dump_enabled = dump_env && std::string_view(dump_env) != "0";
    if (dump_enabled) {
        std::size_t counts[8] = {0};
        for (const auto& meta : graph.tensor_meta) {
            counts[static_cast<std::size_t>(meta.kind)]++;
        }
        std::fprintf(stderr,
                     "[classify_tensors] %s: Unknown=%zu Param=%zu Act=%zu "
                     "ParamGrad=%zu ActGrad=%zu AccumTemp=%zu Loss=%zu Scratch=%zu\n",
                     graph.name.c_str(),
                     counts[static_cast<std::size_t>(TensorKind::Unknown)],
                     counts[static_cast<std::size_t>(TensorKind::ForwardParam)],
                     counts[static_cast<std::size_t>(TensorKind::ForwardActivation)],
                     counts[static_cast<std::size_t>(TensorKind::ParamGrad)],
                     counts[static_cast<std::size_t>(TensorKind::ActivationGrad)],
                     counts[static_cast<std::size_t>(TensorKind::AccumTemp)],
                     counts[static_cast<std::size_t>(TensorKind::LossInput)],
                     counts[static_cast<std::size_t>(TensorKind::Scratch)]);
    }

    // Cross-check: the legacy `base_param_from_grad_heuristic(name)` predicate returns
    // Some(base) for ANY name starting with `d_` (after stripping
    // `_from_N`/`_accum_N`). Our classifier returns ParamGrad ONLY when the
    // base is a real parameter. For every tid where the legacy predicate says
    // "this is a param grad" but the classifier disagrees, report it — those
    // are the latent bugs (intermediate gradients misread as param grads).
    // Callers that switch to the classifier in Phase 1 will see those cases
    // routed correctly.
    const char* check_env = std::getenv("SUROGATE_CHECK_TENSOR_KIND");
    const bool check_enabled = dump_enabled || (check_env && std::string_view(check_env) != "0");
    if (check_enabled) {
        std::size_t disagreements = 0;
        for (const auto& [name, id] : graph.tensor_name_to_id) {
            if (id < 0 || static_cast<std::size_t>(id) >= graph.tensor_meta.size()) continue;
            const auto& meta = graph.tensor_meta[static_cast<std::size_t>(id)];
            auto legacy = base_param_from_grad_heuristic(name);
            if (!legacy.has_value()) {
                // Legacy says "not a gradient name" — classifier should agree
                // (no ParamGrad without a `d_` prefix).
                if (meta.kind == TensorKind::ParamGrad) {
                    std::fprintf(stderr,
                                 "[classify_tensors][DISAGREE] %s tid=%d kind=ParamGrad "
                                 "but legacy base_param_from_grad_heuristic returned nullopt\n",
                                 name.c_str(),
                                 id);
                    disagreements++;
                }
                continue;
            }
            // Legacy says "this is some d_<base>". The classifier agrees IFF
            // kind == ParamGrad AND base_param_tid resolves to <base>.
            const bool legacy_is_param_grad_guess = true;  // legacy always assumes this
            const bool classifier_says_param_grad = (meta.kind == TensorKind::ParamGrad);
            if (legacy_is_param_grad_guess && !classifier_says_param_grad) {
                // Most common case: legacy returns a base string for an
                // intermediate / activation gradient / accum temp. Classifier
                // correctly refuses to call it a ParamGrad. This is exactly
                // the bug class we're eliminating.
                disagreements++;
                if (disagreements <= 10) {
                    std::fprintf(stderr,
                                 "[classify_tensors][LEGACY_OVERREACH] %s tid=%d "
                                 "legacy_base=%s but classifier says kind=%s "
                                 "(would-be silent misroute fixed by Phase 1)\n",
                                 name.c_str(),
                                 id,
                                 legacy->c_str(),
                                 tensor_kind_name(meta.kind));
                }
            }
        }
        if (disagreements > 0) {
            std::fprintf(stderr,
                         "[classify_tensors] %s: %zu legacy/classifier disagreements "
                         "(legacy would have misclassified these as ParamGrad)\n",
                         graph.name.c_str(),
                         disagreements);
        }
    }

    // Cross-check vs DSL slot registry: for every tensor that our hardcoded
    // `global_role_for_name()` table claims is a known global (xF, x0, xN,
    // residualN, ln_final_rstd, freq_cis, …), verify the DSL already declares
    // it at Global scope in the slot registry. The DSL IS the source of truth;
    // the C++ hardcoded table is a compatibility shim that should be empty in
    // steady state. Any miss here is a DSL site that forgot to emit
    // `model._register_activation(name, ..., scope=GLOBAL)`.
    //
    // When all miss counts reach zero across every supported model, the
    // hardcoded fallback table inside global_role_for_name() can be deleted
    // and C++ can query the registry directly for everything.
    const char* coverage_env = std::getenv("SUROGATE_CHECK_GLOBAL_COVERAGE");
    const bool coverage_enabled = dump_enabled || (coverage_env && std::string_view(coverage_env) != "0");
    if (coverage_enabled && mSlotRegistry.has_dsl_layout()) {
        std::size_t hardcoded_count = 0;
        std::size_t registry_hits = 0;
        std::size_t registry_misses = 0;
        std::size_t scope_mismatches = 0;
        std::unordered_map<std::string, std::size_t> missing_summary;  // unqualified name -> count
        for (const auto& [name, id] : graph.tensor_name_to_id) {
            if (id < 0) continue;
            if (starts_with(name, "d_")) continue;
            const GlobalRole role = global_role_for_name(name);
            if (role == GlobalRole::None) continue;
            hardcoded_count++;

            // Try the qualified name first, then the unqualified form: a
            // block-qualified reference like `blocks[N].rope_freqs` points at
            // the same global slot as its unqualified counterpart. The DSL
            // registers the global ONCE; the block stack inliner produces
            // per-layer qualified references to it — they shouldn't count as
            // separate "missing" entries.
            auto entry = mSlotRegistry.lookup(name);
            std::string probe_name = name;
            if (!entry.has_value()) {
                int lid = -1;
                std::string field;
                if (parse_block_param(name, lid, field)) {
                    probe_name = strip_ssa_suffix(field);
                    entry = mSlotRegistry.lookup(probe_name);
                }
            }

            if (!entry.has_value()) {
                registry_misses++;
                missing_summary[probe_name]++;
                if (registry_misses <= 10) {
                    std::fprintf(stderr,
                                 "[classify_tensors][DSL_GAP] %s (role=%d) — registry has no "
                                 "entry for %s. DSL needs model._register_activation(%s, ..., "
                                 "scope=GLOBAL).\n",
                                 name.c_str(),
                                 static_cast<int>(role),
                                 probe_name.c_str(),
                                 probe_name.c_str());
                }
                continue;
            }
            if (entry->scope != ActivationScope::Global) {
                scope_mismatches++;
                if (scope_mismatches <= 10) {
                    std::fprintf(stderr,
                                 "[classify_tensors][DSL_SCOPE] %s: expected Global, "
                                 "registry has scope=%d\n",
                                 name.c_str(),
                                 static_cast<int>(entry->scope));
                }
                continue;
            }
            registry_hits++;
        }
        std::fprintf(stderr,
                     "[classify_tensors] %s: DSL coverage: %zu/%zu global-name references "
                     "resolved (%zu missing, %zu scope-mismatched). ",
                     graph.name.c_str(),
                     registry_hits,
                     hardcoded_count,
                     registry_misses,
                     scope_mismatches);
        if (!missing_summary.empty()) {
            std::fprintf(stderr, "Distinct missing names: [");
            bool first = true;
            for (const auto& [n, cnt] : missing_summary) {
                std::fprintf(stderr, "%s%s×%zu", first ? "" : ", ", n.c_str(), cnt);
                first = false;
            }
            std::fprintf(stderr, "]\n");
        } else {
            std::fprintf(stderr, "(no gaps)\n");
        }
    }
}

void GraphCompiler::reset_tid_namespace() {
    mTensorIdMap.clear();
    mNextTensorId = 0;
    // Also clear the shape/dtype databases here. They share the same scope as
    // the tid namespace: forward and backward compiles in one pair share them
    // so backward can look up forward tensor shapes (e.g., zeros op with
    // shape_like referencing a forward split output).
    mExtraShapes.clear();
    mTensorShapes.clear();
    mTensorDtypes.clear();
}

CompiledGraph GraphCompiler::compile(const Graph& graph, long B, long T, bool is_backward) {
    update_dimensions(B, T);

    // Note: mTensorIdMap, mNextTensorId, mExtraShapes, mTensorShapes, and
    // mTensorDtypes are NOT cleared here — they persist across the
    // forward+backward pair so backward can reference forward tensor
    // shapes/dtypes. Caller (GraphExecutor::compile_graphs) calls
    // reset_tid_namespace() once before the pair when starting fresh.

    // Initialize shape database from graph inputs and params
    for (const auto& [name, info] : graph.inputs) {
        if (!info.shape.empty()) {
            TensorShape ts;
            ts.dims = resolve_shape(info.shape, mShapeEnv);
            ts.inferred = false;
            mTensorShapes[name] = ts;
        }
        if (info.dtype) {
            mTensorDtypes[name] = *info.dtype;
        }
    }
    for (const auto& [name, info] : graph.params) {
        if (!info.shape.empty()) {
            TensorShape ts;
            ts.dims = resolve_shape(info.shape, mShapeEnv);
            ts.inferred = false;
            mTensorShapes[name] = ts;
        }
        if (info.dtype) {
            mTensorDtypes[name] = *info.dtype;
        }
    }
    for (const auto& [name, info] : graph.outputs) {
        if (!info.shape.empty()) {
            TensorShape ts;
            ts.dims = resolve_shape(info.shape, mShapeEnv);
            ts.inferred = false;
            mTensorShapes[name] = ts;
        }
        if (info.dtype) {
            mTensorDtypes[name] = *info.dtype;
        }
    }

    if (mModule.forward.has_value()) {
        const auto& fwd = *mModule.forward;
        for (const auto& op : fwd.operations) {
            const std::string& op_type =
                (op.kernel_type.empty() || op.kernel_type == "custom") ? op.name : op.kernel_type;
            if (op_type != "view" && op_type != "reshape") {
                continue;
            }
            std::vector<long> shape;
            if (auto* shape_attr = find_attr(op.attrs, "shape")) {
                shape = resolve_attr_shape(*shape_attr, mShapeEnv);
            } else if (auto* shape_like_attr = find_attr(op.attrs, "shape_like")) {
                if (auto ref_name = attr_string(*shape_like_attr)) {
                    std::string ref = *ref_name;
                    if (starts_with(ref, kSavedPrefix)) {
                        ref = ref.substr(kSavedPrefix.size());
                    }
                    auto it = mExtraShapes.find(ref);
                    if (it != mExtraShapes.end()) {
                        shape = it->second;
                    } else {
                        infer_known_tensor_shape(ref, mConfig, B, T, shape);
                    }
                }
            }
            if (!shape.empty()) {
                for (const auto& out : op.outputs) {
                    if (!out.empty()) {
                        mExtraShapes[out] = shape;
                    }
                }
            }
        }
    }

    // Also pre-scan the current graph for view/reshape ops (important for backward
    // graphs where view_backward ops use shape_like referencing forward tensors).
    // The forward pre-scan above already populated mExtraShapes with forward tensor
    // shapes, so shape_like references can resolve here.
    if (!mModule.forward.has_value() || &graph != &(*mModule.forward)) {
        for (const auto& op : graph.operations) {
            const std::string& op_type =
                (op.kernel_type.empty() || op.kernel_type == "custom") ? op.name : op.kernel_type;
            if (op_type != "view" && op_type != "reshape") {
                continue;
            }
            std::vector<long> shape;
            if (auto* shape_attr = find_attr(op.attrs, "shape")) {
                shape = resolve_attr_shape(*shape_attr, mShapeEnv);
            } else if (auto* shape_like_attr = find_attr(op.attrs, "shape_like")) {
                if (auto ref_name = attr_string(*shape_like_attr)) {
                    std::string ref = *ref_name;
                    if (starts_with(ref, kSavedPrefix)) {
                        ref = ref.substr(kSavedPrefix.size());
                    }
                    auto it = mExtraShapes.find(ref);
                    if (it != mExtraShapes.end()) {
                        shape = it->second;
                    } else {
                        infer_known_tensor_shape(ref, mConfig, B, T, shape);
                    }
                }
            }
            if (!shape.empty()) {
                for (const auto& out : op.outputs) {
                    if (!out.empty()) {
                        mExtraShapes[out] = shape;
                    }
                }
            }
        }
    }

    CompiledGraph result;
    result.name = graph.name;
    result.ops.reserve(graph.operations.size());
    result.total_ops = graph.operations.size();

    for (std::size_t idx = 0; idx < graph.operations.size(); ++idx) {
        const auto& op = graph.operations[idx];
        const std::string& op_type = (op.kernel_type.empty() || op.kernel_type == "custom") ? op.name : op.kernel_type;

        CompiledOp compiled;
        compiled.original_idx = static_cast<std::uint16_t>(idx);
        compiled.op_id = op.id;
        compiled.type = classify_op(op_type);

        if (compiled.type == CompiledOpType::Unknown) {
            throw std::runtime_error("GraphCompiler: unsupported operation type: " + op_type);
        }

        // Bake the dispatch function pointer into the op. For the
        // backward graph prefer the backward_fn; for the forward graph
        // use forward_fn. Standalone op goldens can compile explicit
        // backward-only ops without marking the whole graph as backward, so
        // fall back to backward_fn when the forward slot is intentionally empty.
        // Null means "no handler" — execute will throw when it tries to call it.
        if (const OpDescriptor* desc = OpRegistry::instance().find(compiled.type)) {
            compiled.fn = is_backward ? desc->backward_fn : (desc->forward_fn ? desc->forward_fn : desc->backward_fn);
            compiled.semantic_kind = desc->semantic_kind;
            compiled.distribution_kind = desc->distribution_kind;
            compiled.default_caps = desc->default_caps;
            compiled.epilogue_support = desc->epilogue_support;
            compiled.storage_compat = desc->storage_compat;
            compiled.moe_caps = desc->moe_caps;
            compiled.matmul_caps = desc->matmul_caps;
            compiled.comm_profile = desc->comm_profile;
            compiled.grouped_semantics = desc->grouped_semantics;
            compiled.descriptor_flags = desc->descriptor_flags;
        }

        // Validate operation shapes at compile time.
        // In hybrid models (e.g., Gemma4 with sliding + full attention), per-block
        // shapes may vary, and the global shape env only stores one set of dims.
        // Shape validation errors are therefore non-fatal warnings for hybrid models.
        try {
            validate_operation_shapes(op, compiled.type, idx);
        } catch (const std::exception& e) {
            if (mConfig.architecture == modules::ArchitectureType::Hybrid || !mConfig.layer_overrides.empty() ||
                mHasHybridBlocks) {
                // Hybrid model: shape mismatch likely due to per-block-type dimension
                // variation. Silently continue — runtime will use correct shapes.
            } else {
                std::cerr << "Shape validation failed during graph compilation.\n"
                          << "Operation: " << op.name << " (id: " << op.id << ")\n"
                          << "Error: " << e.what() << "\n";
                throw;
            }
        }

        // For hybrid models, detect the layer index from this op's tensors
        // and use a per-layer shape env with correct dimensions.
        const ShapeEnv* env_ptr = &mShapeEnv;
        ShapeEnv layer_env;
        if (!mPerLayerDims.empty()) {
            int detected_layer = -1;
            std::string field;
            for (const auto& inp : op.inputs) {
                if (parse_block_param(inp, detected_layer, field) && detected_layer >= 0) break;
            }
            if (detected_layer < 0) {
                for (const auto& out : op.outputs) {
                    if (parse_block_param(out, detected_layer, field) && detected_layer >= 0) break;
                }
            }
            if (detected_layer >= 0) {
                layer_env = make_layer_env(detected_layer);
                env_ptr = &layer_env;
            }
        }

        // Pre-resolve inputs
        compiled.inputs.reserve(op.inputs.size());
        for (std::size_t i = 0; i < op.inputs.size(); ++i) {
            auto ref = resolve_tensor_ref(op.inputs[i], false, op, *env_ptr);
            if ((compiled.type == CompiledOpType::RMSNormBackward ||
                 compiled.type == CompiledOpType::LayerNormBackward) &&
                i == 3) {
                ref.dtype = ETensorDType::FP32;
                if (compiled.inputs.size() > 1 && !compiled.inputs[1].shape.empty()) {
                    ref.shape = compiled.inputs[1].shape;
                    ref.shape.pop_back();
                }
            }
            compiled.inputs.push_back(ref);
        }

        // Pre-resolve outputs
        compiled.outputs.reserve(op.outputs.size());
        for (std::size_t i = 0; i < op.outputs.size(); ++i) {
            auto ref = resolve_tensor_ref(op.outputs[i], true, op, *env_ptr);
            bool shape_is_default_fallback = false;

            // Op-semantic dtype overrides that are independent of the slot
            // routing path. The per-slot resolution above may stamp a
            // default/activation dtype on output refs whose op produces a
            // different dtype — e.g., FlashAttention's LSE is FP32, but a
            // BlockLSE-slotted ref leaves resolve_tensor_ref with BF16.
            // The per-frame coloring reads `ref.dtype` to size the arena
            // slice; a BF16-typed LSE yields a half-sized allocation,
            // overflowing into the next slot when the FP32 kernel writes.
            // Patch the dtype here, where we know the op semantics.
            if (compiled.type == CompiledOpType::FlashAttention && i == 1) {
                ref.dtype = ETensorDType::FP32;
            }
            if ((compiled.type == CompiledOpType::RMSNorm || compiled.type == CompiledOpType::LayerNorm) && i == 1) {
                ref.dtype = ETensorDType::FP32;
            }

            // Fix dtype and shape for outputs based on operation type
            // This is needed for Mapped tensors that don't have predefined slots
            if (ref.slot == TensorSlot::Mapped) {
                const long B = mB;
                const long T = mT;
                const long C = mConfig.HiddenSize;
                const long Hq = mConfig.NumQueryHeads;
                const long Hs = mConfig.head_size();
                const long QKV = mConfig.qkv_channels();

                if (compiled.type == CompiledOpType::FusedResidualRMSNorm) {
                    // output[0] = residual_out [B, T, C] BF16
                    // output[1] = y (normalized) [B, T, C] BF16
                    // output[2] = rstd [B*T] FP32
                    if (i == 0 || i == 1) {
                        ref.dtype = !compiled.inputs.empty() ? compiled.inputs[0].dtype : ETensorDType::BF16;
                        ref.shape = {B, T, C};
                    } else if (i == 2) {
                        ref.dtype = ETensorDType::FP32;
                        ref.shape = {B * T};
                    }
                } else if (compiled.type == CompiledOpType::CrossEntropyLoss) {
                    // output[0] = loss [B*T] FP32 (per-token)
                    ref.dtype = ETensorDType::FP32;
                    ref.shape = {B * T};
                } else if (compiled.type == CompiledOpType::CrossEntropyLossBackward) {
                    // output[0] = d_logits [B*T, V] (match logits dtype)
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[1].dtype;
                        ref.shape = compiled.inputs[1].shape;
                    } else {
                        ref.dtype = ETensorDType::BF16;
                        ref.shape = {B * T, static_cast<long>(mConfig.VocabSize)};
                    }
                } else if (compiled.type == CompiledOpType::FusedLMHeadLoss) {
                    // output[0] = loss [B*T] FP32 (per-token)
                    ref.dtype = ETensorDType::FP32;
                    ref.shape = {B * T};
                } else if (compiled.type == CompiledOpType::FusedLMHeadLossBackward) {
                    // output[0] = d_xF_flat [B*T, C], output[1] = d_lm_head [V, C]
                    if (i == 0) {
                        if (compiled.inputs.size() > 1) {
                            ref.dtype = compiled.inputs[1].dtype;
                            ref.shape = compiled.inputs[1].shape;
                        } else {
                            ref.dtype = ETensorDType::BF16;
                            ref.shape = {B * T, C};
                        }
                    } else if (i == 1) {
                        if (compiled.inputs.size() > 2) {
                            ref.dtype = compiled.inputs[2].dtype;
                            ref.shape = compiled.inputs[2].shape;
                        } else {
                            ref.dtype = ETensorDType::BF16;
                            ref.shape = {static_cast<long>(mConfig.VocabSize), C};
                        }
                    }
                } else if (compiled.type == CompiledOpType::FusedResidualRMSNormBackward) {
                    // outputs: d_residual [B, T, C], d_input [B, T, C], d_weight [C]
                    const ETensorDType grad_dtype =
                        !compiled.inputs.empty() ? compiled.inputs[0].dtype : ETensorDType::BF16;
                    if (i == 0 || i == 1) {
                        ref.dtype = grad_dtype;
                        ref.shape = {B, T, C};
                    } else if (i == 2) {
                        if (compiled.inputs.size() > 3) {
                            ref.dtype = compiled.inputs[3].dtype;
                        } else {
                            ref.dtype = grad_dtype;
                        }
                        ref.shape = {C};
                    }
                } else if (compiled.type == CompiledOpType::QKVQKNorm) {
                    // output[0] = qkv_out [B, T, QKV] (match input dtype)
                    // output[1] = q_rstd [B, T, Hq] FP32
                    // output[2] = k_rstd [B, T, Hkv] FP32
                    if (i == 0) {
                        // Match input dtype (first input is qkv tensor)
                        if (!compiled.inputs.empty()) {
                            ref.dtype = compiled.inputs[0].dtype;
                        } else {
                            ref.dtype = ETensorDType::BF16;
                        }
                        if (ref.shape.empty()) {
                            ref.shape = {B, T, QKV};
                        }
                    } else if (i == 1) {
                        ref.dtype = ETensorDType::FP32;
                        if (ref.shape.empty()) {
                            ref.shape = {B, T, Hq};
                        }
                    } else if (i == 2) {
                        ref.dtype = ETensorDType::FP32;
                        if (ref.shape.empty()) {
                            ref.shape = {B, T, static_cast<long>(mConfig.NumKeyValHeads)};
                        }
                    }
                } else if (compiled.type == CompiledOpType::QKVQKNormBackward ||
                           compiled.type == CompiledOpType::QKVQKNormRoPEBackward) {
                    // outputs: d_qkv, d_q_norm_weight, d_k_norm_weight
                    // d_qkv matches qkv input; d_weight matches weight shape [D]
                    if (i == 0) {
                        if (compiled.inputs.size() > 1 && !compiled.inputs[1].shape.empty()) {
                            ref.dtype = compiled.inputs[1].dtype;
                            ref.shape = compiled.inputs[1].shape;
                        } else if (!compiled.inputs.empty()) {
                            ref.dtype = compiled.inputs[0].dtype;
                            ref.shape = compiled.inputs[0].shape;
                        }
                    } else if (i == 1) {
                        if (compiled.inputs.size() > 2 && !compiled.inputs[2].shape.empty()) {
                            ref.dtype = compiled.inputs[2].dtype;
                            ref.shape = compiled.inputs[2].shape;
                        } else {
                            ref.dtype = !compiled.inputs.empty() ? compiled.inputs[0].dtype : ETensorDType::BF16;
                            ref.shape = {static_cast<long>(mConfig.head_size())};
                        }
                    } else if (i == 2) {
                        if (compiled.inputs.size() > 3 && !compiled.inputs[3].shape.empty()) {
                            ref.dtype = compiled.inputs[3].dtype;
                            ref.shape = compiled.inputs[3].shape;
                        } else {
                            ref.dtype = !compiled.inputs.empty() ? compiled.inputs[0].dtype : ETensorDType::BF16;
                            ref.shape = {static_cast<long>(mConfig.head_size())};
                        }
                    }
                } else if (compiled.type == CompiledOpType::QKVQKNormRoPE) {
                    // output[0] = qkv_out [B, T, QKV] (match input dtype)
                    // output[1] = q_rstd [B, T, Hq] FP32
                    // output[2] = k_rstd [B, T, Hkv] FP32
                    if (i == 0) {
                        // Match input dtype (first input is qkv tensor)
                        if (!compiled.inputs.empty()) {
                            ref.dtype = compiled.inputs[0].dtype;
                        } else {
                            ref.dtype = ETensorDType::BF16;
                        }
                        if (ref.shape.empty()) {
                            ref.shape = {B, T, QKV};
                        }
                    } else if (i == 1) {
                        ref.dtype = ETensorDType::FP32;
                        if (ref.shape.empty()) {
                            ref.shape = {B, T, Hq};
                        }
                    } else if (i == 2) {
                        ref.dtype = ETensorDType::FP32;
                        if (ref.shape.empty()) {
                            ref.shape = {B, T, static_cast<long>(mConfig.NumKeyValHeads)};
                        }
                    }
                } else if (compiled.type == CompiledOpType::FlashAttention) {
                    // output[0] = out [B, T, Hq*Hs] (match qkv dtype)
                    // output[1] = lse [B, Hq, T] FP32
                    if (i == 0) {
                        if (!compiled.inputs.empty()) {
                            ref.dtype = compiled.inputs[0].dtype;
                        }
                        if (ref.shape.empty()) {
                            ref.shape = {B, T, Hq * Hs};
                        }
                    } else if (i == 1) {
                        ref.dtype = ETensorDType::FP32;
                        if (ref.shape.empty()) {
                            ref.shape = {B, Hq, T};
                        }
                    }
                } else if (compiled.type == CompiledOpType::Add || compiled.type == CompiledOpType::BiasAdd) {
                    // Match output to first input (broadcasting not supported in compiled add path).
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                        if (ref.shape.empty()) {
                            ref.shape = compiled.inputs[0].shape;
                        }
                    }
                } else if (compiled.type == CompiledOpType::AddBackward ||
                           compiled.type == CompiledOpType::BiasAddBackward) {
                    // Gradients match upstream shape/dtype.
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                        if (ref.shape.empty()) {
                            ref.shape = compiled.inputs[0].shape;
                        }
                    }
                } else if (compiled.type == CompiledOpType::Matmul || compiled.type == CompiledOpType::MatmulBias) {
                    // Infer output shape from matmul dimensions: C = A @ B
                    // NT: A [M, K], B [N, K] -> C [M, N]
                    // NN: A [M, K], B [K, N] -> C [M, N]
                    // TN: A [K, M], B [K, N] -> C [M, N]
                    // TT: A [K, M], B [N, K] -> C [M, N]
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                    }
                    if (ref.shape.empty() && compiled.inputs.size() >= 2) {
                        const auto& a_shape = compiled.inputs[0].shape;
                        const auto& b_shape = compiled.inputs[1].shape;
                        if (a_shape.size() >= 2 && b_shape.size() >= 2) {
                            // Parse transpose from op.attrs (compiled.attrs not yet resolved!)
                            EMMTranspose transpose = parse_transpose(op.attrs);
                            long M = 0, N = 0;
                            if (transpose == EMMTranspose::NT || transpose == EMMTranspose::NN) {
                                M = a_shape[0];
                            } else {
                                M = a_shape[1];
                            }
                            if (transpose == EMMTranspose::NT || transpose == EMMTranspose::TT) {
                                N = b_shape[0];
                            } else {
                                N = b_shape[1];
                            }
                            ref.shape = {M, N};
                        }
                    }
                } else if (compiled.type == CompiledOpType::MatmulSwiGLU) {
                    // outputs: out [B, T, D], up_out [M, 2D]
                    ETensorDType base_dtype = !compiled.inputs.empty() ? compiled.inputs[0].dtype : ETensorDType::BF16;
                    long Ndim = 0;
                    if (compiled.inputs.size() > 1 && compiled.inputs[1].shape.size() >= 2) {
                        Ndim = compiled.inputs[1].shape[1];
                    }
                    long Ddim = (Ndim > 0) ? (Ndim / 2) : C;
                    long Mdim = mB * mT;
                    if (!compiled.inputs.empty() && compiled.inputs[0].shape.size() >= 1) {
                        Mdim = compiled.inputs[0].shape[0];
                    }

                    if (i == 0) {
                        ref.dtype = base_dtype;
                        ref.shape = {B, T, Ddim};
                    } else if (i == 1) {
                        ref.dtype = base_dtype;
                        ref.shape = {Mdim, Ndim > 0 ? Ndim : (2 * Ddim)};
                    }
                } else if (compiled.type == CompiledOpType::RMSNorm || compiled.type == CompiledOpType::LayerNorm) {
                    // Standalone norm ops: output[0] = y (same as input x),
                    // output[1] = rstd (FP32, input rows).
                    // This covers Gemma4 per-head Q/K/V norms over 2D inputs
                    // like [B*T*H, D] that would otherwise fall through to the
                    // {B,T,C} default and break backward view/shape propagation.
                    if (!compiled.inputs.empty() && !compiled.inputs[0].shape.empty()) {
                        if (i == 0) {
                            if (ref.shape.empty()) ref.shape = compiled.inputs[0].shape;
                            ref.dtype = compiled.inputs[0].dtype;
                        } else if (i == 1) {
                            // rstd: input shape minus last dim, FP32
                            if (ref.shape.empty()) {
                                auto rstd_shape = compiled.inputs[0].shape;
                                rstd_shape.pop_back();
                                ref.shape = std::move(rstd_shape);
                            }
                            ref.dtype = ETensorDType::FP32;
                        }
                    }
                } else if (compiled.type == CompiledOpType::RMSNormBackward ||
                           compiled.type == CompiledOpType::LayerNormBackward) {
                    // Backward inputs: [0] d_out, [1] saved x, [2] weight, [3] saved rstd
                    // Outputs: [0] d_x (shape of x), [1] d_weight (shape of weight)
                    if (i == 0 && compiled.inputs.size() > 1 && !compiled.inputs[1].shape.empty()) {
                        // d_x has same shape/dtype as saved x (input[1])
                        if (ref.shape.empty()) ref.shape = compiled.inputs[1].shape;
                        ref.dtype = compiled.inputs[1].dtype;
                    } else if (i == 1 && compiled.inputs.size() > 2 && !compiled.inputs[2].shape.empty()) {
                        // d_weight has same shape/dtype as weight (input[2])
                        if (ref.shape.empty()) ref.shape = compiled.inputs[2].shape;
                        ref.dtype = compiled.inputs[2].dtype;
                    }
                } else if (compiled.type == CompiledOpType::Narrow) {
                    // output = input shape with narrow dim replaced by length.
                    // Reuses `split_concat_dim` (common "dim" attr, same as dispatch_narrow).
                    if (!compiled.inputs.empty() && !compiled.inputs[0].shape.empty() && ref.shape.empty()) {
                        const auto& in_shape = compiled.inputs[0].shape;
                        const int rank = static_cast<int>(in_shape.size());
                        int dim = compiled.attrs.split_concat_dim;
                        if (dim < 0) dim += rank;
                        const long length = compiled.attrs.narrow_length;
                        if (dim >= 0 && dim < rank && length > 0) {
                            auto out_shape = in_shape;
                            out_shape[dim] = length;
                            ref.shape = std::move(out_shape);
                        }
                    }
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                    }
                } else if (compiled.type == CompiledOpType::Zeros || compiled.type == CompiledOpType::Ones) {
                    // Preserve explicit output dtype/shape from graph.
                    // Read dtype from op attributes if specified
                    if (auto* dtype_attr = find_attr(op.attrs, "dtype")) {
                        if (auto dtype_str = attr_string(*dtype_attr)) {
                            ref.dtype = dtype_from_str(*dtype_str);
                        }
                    }
                    if (ref.shape.empty()) {
                        // Try to resolve shape_like at compile time. The
                        // split_backward autodiff rule emits zeros ops whose
                        // shape should match a forward split output; without
                        // resolving this here we'd fall back to {B,T,C},
                        // producing concats with wrong gradient shapes.
                        // Requires reset_tid_namespace() to have preserved
                        // forward's mExtraShapes/mTensorShapes for backward.
                        if (auto* shape_like_attr = find_attr(op.attrs, "shape_like")) {
                            if (auto ref_name_opt = attr_string(*shape_like_attr)) {
                                std::string ref_name = *ref_name_opt;
                                if (starts_with(ref_name, kSavedPrefix)) {
                                    ref_name = ref_name.substr(kSavedPrefix.size());
                                }
                                std::vector<long> resolved;
                                if (auto it = mExtraShapes.find(ref_name); it != mExtraShapes.end()) {
                                    resolved = it->second;
                                } else if (auto it2 = mTensorShapes.find(ref_name); it2 != mTensorShapes.end()) {
                                    resolved = it2->second.dims;
                                } else {
                                    infer_known_tensor_shape(ref_name, mConfig, B, T, resolved);
                                }
                                if (!resolved.empty()) {
                                    ref.shape = resolved;
                                }
                            }
                        }
                    }
                    if (ref.shape.empty()) {
                        ref.shape = {B, T, C};
                    }
                } else if (compiled.type == CompiledOpType::RoPE || compiled.type == CompiledOpType::RoPEBackward) {
                    // RoPE outputs match input dtype/shape.
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                        if (ref.shape.empty()) {
                            ref.shape = compiled.inputs[0].shape;
                        }
                    }
                } else if (compiled.type == CompiledOpType::SwiGLU) {
                    // Output dtype matches input; shape is input with last dim / 2.
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                        if (ref.shape.empty()) {
                            ref.shape = compiled.inputs[0].shape;
                            if (!ref.shape.empty()) {
                                ref.shape.back() = ref.shape.back() / 2;
                            }
                        }
                    }
                } else if (compiled.type == CompiledOpType::SwiGLUBackward) {
                    // Output (d_inp) matches the pre-SwiGLU input shape.
                    // inputs: d_out [N, D], inp [N, 2D] -> output: d_inp [N, 2D]
                    if (compiled.inputs.size() > 1 && !compiled.inputs[1].shape.empty()) {
                        ref.dtype = compiled.inputs[1].dtype;
                        ref.shape = compiled.inputs[1].shape;
                    } else if (!compiled.inputs.empty() && !compiled.inputs[0].shape.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                        ref.shape = compiled.inputs[0].shape;
                        if (!ref.shape.empty()) {
                            ref.shape.back() *= 2;
                        }
                    }
                } else if (compiled.type == CompiledOpType::MatmulBackward) {
                    // Match dA/dB shapes to their corresponding inputs (A/B).
                    // inputs: d_out, A_for_dB, B_for_dA -> outputs: dA, dB
                    if (i == 0 && compiled.inputs.size() > 1) {
                        ref.shape = compiled.inputs[1].shape;
                        ref.dtype = compiled.inputs[1].dtype;
                    } else if (i == 1 && compiled.inputs.size() > 2) {
                        ref.shape = compiled.inputs[2].shape;
                        ref.dtype = compiled.inputs[2].dtype;
                    } else {
                        ref.dtype = ETensorDType::BF16;
                    }
                } else if (compiled.type == CompiledOpType::MatmulSwiGLUBackward) {
                    // outputs: d_inp matches ln2 shape/dtype, d_weight matches weight shape/dtype
                    if (i == 0 && compiled.inputs.size() > 1) {
                        ref.shape = compiled.inputs[1].shape;
                        ref.dtype = compiled.inputs[1].dtype;
                    } else if (i == 1 && compiled.inputs.size() > 2) {
                        ref.shape = compiled.inputs[2].shape;
                        ref.dtype = compiled.inputs[2].dtype;
                    } else {
                        ref.dtype = ETensorDType::BF16;
                    }
                } else if (compiled.type == CompiledOpType::View || compiled.type == CompiledOpType::ViewBackward) {
                    // View preserves dtype from input; shape comes from mExtraShapes
                    // (populated by the pre-scan) or from resolve_tensor_ref.
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                    }
                    // If shape wasn't set by resolve_tensor_ref or mExtraShapes,
                    // try resolving from op attributes (shape or shape_like).
                    if (ref.shape.empty()) {
                        if (auto* shape_attr = find_attr(op.attrs, "shape")) {
                            ref.shape = resolve_attr_shape(*shape_attr, mShapeEnv);
                        } else if (auto* shape_like_attr = find_attr(op.attrs, "shape_like")) {
                            if (auto ref_name = attr_string(*shape_like_attr)) {
                                std::string sref = *ref_name;
                                if (starts_with(sref, kSavedPrefix)) {
                                    sref = sref.substr(kSavedPrefix.size());
                                }
                                // Prefer infer_known_tensor_shape for well-known names
                                // (it correctly distinguishes _flat vs non-flat shapes),
                                // then fall back to mExtraShapes / resolve_tensor_shape.
                                std::vector<long> ref_shape;
                                if (infer_known_tensor_shape(sref, mConfig, B, T, ref_shape)) {
                                    ref.shape = ref_shape;
                                } else {
                                    auto eit = mExtraShapes.find(sref);
                                    if (eit != mExtraShapes.end()) {
                                        ref.shape = eit->second;
                                    } else if (resolve_tensor_shape(sref, ref_shape)) {
                                        ref.shape = ref_shape;
                                    }
                                }
                            }
                        }
                    }
                    if (mDebugShapes && starts_with(op.outputs[i], "d_")) {
                        auto fmt = [](const std::vector<long>& s) -> std::string {
                            std::string r = "(";
                            for (size_t j = 0; j < s.size(); ++j) {
                                if (j > 0) r += ", ";
                                r += std::to_string(s[j]);
                            }
                            r += ")";
                            return r;
                        };
                        const char* source = "resolve_tensor_ref";
                        if (find_attr(op.attrs, "shape"))
                            source = "shape attr";
                        else if (find_attr(op.attrs, "shape_like"))
                            source = "shape_like attr";
                        fprintf(stderr,
                                "[DEBUG_SHAPES] View output '%s' shape=%s (via %s)\n",
                                op.outputs[i].c_str(),
                                ref.shape.empty() ? "<empty>" : fmt(ref.shape).c_str(),
                                source);
                    }
                } else if (compiled.type == CompiledOpType::MoESigmoid || compiled.type == CompiledOpType::MoESoftmax) {
                    // Output dtype/shape matches input (router logits)
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                        if (ref.shape.empty()) {
                            ref.shape = compiled.inputs[0].shape;
                        }
                    }
                } else if (compiled.type == CompiledOpType::MoETopK) {
                    // output[0] = routing_weights [B*T, K] (same dtype as input)
                    // output[1] = routing_indices [B*T, K] INT32
                    int top_k = 1;
                    if (auto* attr = find_attr(op.attrs, "top_k")) {
                        if (auto v = attr_int(*attr)) {
                            top_k = static_cast<int>(*v);
                        }
                    }
                    long BT = B * T;
                    if (!compiled.inputs.empty() && !compiled.inputs[0].shape.empty()) {
                        BT = compiled.inputs[0].shape[0];
                    }
                    if (i == 0) {
                        // routing_weights - same dtype as input probs
                        if (!compiled.inputs.empty()) {
                            ref.dtype = compiled.inputs[0].dtype;
                        }
                        ref.shape = {BT, static_cast<long>(top_k)};
                    } else if (i == 1) {
                        // routing_indices - INT32
                        ref.dtype = ETensorDType::INT32;
                        ref.shape = {BT, static_cast<long>(top_k)};
                    }
                } else if (compiled.type == CompiledOpType::MoEPermute) {
                    // output[0] = permuted_input [total_tokens, C] (same dtype as input)
                    // output[1] = scatter_indices [total_tokens] INT32
                    int top_k = 1;
                    if (auto* attr = find_attr(op.attrs, "top_k")) {
                        if (auto v = attr_int(*attr)) {
                            top_k = static_cast<int>(*v);
                        }
                    }
                    long num_tokens = B * T;
                    long hidden_size = C;
                    if (!compiled.inputs.empty() && !compiled.inputs[0].shape.empty()) {
                        num_tokens = compiled.inputs[0].shape[0];
                        if (compiled.inputs[0].shape.size() > 1) {
                            hidden_size = compiled.inputs[0].shape[1];
                        }
                    }
                    long total_tokens = num_tokens * top_k;
                    if (i == 0) {
                        // permuted_input - same dtype as input
                        if (!compiled.inputs.empty()) {
                            ref.dtype = compiled.inputs[0].dtype;
                        }
                        ref.shape = {total_tokens, hidden_size};
                    } else if (i == 1) {
                        // scatter_indices - INT32
                        ref.dtype = ETensorDType::INT32;
                        ref.shape = {total_tokens};
                    }
                } else if (compiled.type == CompiledOpType::MoEGroupedGemmGateUp) {
                    // output[0] = gate_up_out [total_tokens, 2*intermediate] (same dtype as input)
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                    }
                    // Shape is dynamic based on scatter_indices, leave empty for runtime
                } else if (compiled.type == CompiledOpType::MoEGroupedGemmDown) {
                    // output[0] = down_out [total_tokens, C] (same dtype as input)
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                    }
                    // Shape is dynamic based on scatter_indices, leave empty for runtime
                } else if (compiled.type == CompiledOpType::MoEUnpermute) {
                    // output[0] = combined_out [B*T, C] (same dtype as input)
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                    }
                    long num_tokens = B * T;
                    if (!compiled.inputs.empty() && compiled.inputs.size() > 1 && !compiled.inputs[1].shape.empty()) {
                        // routing_weights shape is [B*T, K]
                        num_tokens = compiled.inputs[1].shape[0];
                    }
                    ref.shape = {num_tokens, C};
                } else if (compiled.type == CompiledOpType::MoESigmoidBackward ||
                           compiled.type == CompiledOpType::MoESoftmaxBackward) {
                    // inputs: d_out, saved.input
                    // output: d_input (same shape/dtype as d_out, which is input[0])
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                        if (ref.shape.empty()) {
                            ref.shape = compiled.inputs[0].shape;
                        }
                    }
                } else if (compiled.type == CompiledOpType::MoETopKBackward) {
                    // inputs: d_routing_weights, saved.probs, saved.indices
                    // output: d_probs (same shape/dtype as saved.probs, which is input[1])
                    if (compiled.inputs.size() > 1 && !compiled.inputs[1].shape.empty()) {
                        ref.dtype = compiled.inputs[1].dtype;
                        ref.shape = compiled.inputs[1].shape;
                    } else if (!compiled.inputs.empty()) {
                        // Fallback: use d_routing_weights dtype, derive probs shape
                        ref.dtype = compiled.inputs[0].dtype;
                        // probs is [num_tokens, num_experts], d_routing_weights is [num_tokens, top_k]
                        // We need num_experts from config
                        long num_tokens = B * T;
                        if (!compiled.inputs[0].shape.empty()) {
                            num_tokens = compiled.inputs[0].shape[0];
                        }
                        // Default from model config, then check for explicit attr override
                        long num_experts = static_cast<long>(mConfig.NumExperts);
                        if (auto* attr = find_attr(op.attrs, "num_experts")) {
                            if (auto v = attr_int(*attr)) {
                                num_experts = *v;
                            }
                        }
                        ref.shape = {num_tokens, num_experts};
                    }
                } else if (compiled.type == CompiledOpType::MoEPermuteBackward) {
                    // inputs: d_permuted, saved.scatter_indices
                    // output: d_x (unpermuted gradient)
                    // d_x shape is [num_tokens, hidden_size] where num_tokens = scatter_indices.size() / top_k
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                    }
                    // Derive shape from scatter_indices and top_k
                    int top_k = 1;
                    if (auto* attr = find_attr(op.attrs, "top_k")) {
                        if (auto v = attr_int(*attr)) {
                            top_k = static_cast<int>(*v);
                        }
                    }
                    long total_tokens = B * T * top_k;  // permuted size
                    long hidden_size = C;
                    if (!compiled.inputs.empty() && !compiled.inputs[0].shape.empty()) {
                        total_tokens = compiled.inputs[0].shape[0];
                        if (compiled.inputs[0].shape.size() > 1) {
                            hidden_size = compiled.inputs[0].shape[1];
                        }
                    }
                    long num_tokens = total_tokens / top_k;
                    ref.shape = {num_tokens, hidden_size};
                } else if (compiled.type == CompiledOpType::MoEUnpermuteBackward) {
                    // inputs: d_out, saved.expert_out, saved.routing_weights, saved.scatter_indices
                    // outputs[0]: d_expert_out (same shape as saved.expert_out, input[1])
                    // outputs[1]: d_routing_weights (same shape as saved.routing_weights, input[2])
                    if (i == 0) {
                        // d_expert_out - same shape/dtype as saved.expert_out (input[1])
                        if (compiled.inputs.size() > 1 && !compiled.inputs[1].shape.empty()) {
                            ref.dtype = compiled.inputs[1].dtype;
                            ref.shape = compiled.inputs[1].shape;
                        } else if (!compiled.inputs.empty()) {
                            ref.dtype = compiled.inputs[0].dtype;
                            // Fallback: expert_out is [total_tokens, C]
                            int top_k = 1;
                            if (auto* attr = find_attr(op.attrs, "top_k")) {
                                if (auto v = attr_int(*attr)) {
                                    top_k = static_cast<int>(*v);
                                }
                            }
                            ref.shape = {B * T * top_k, C};
                        }
                    } else if (i == 1) {
                        // d_routing_weights - same shape/dtype as saved.routing_weights (input[2])
                        if (compiled.inputs.size() > 2 && !compiled.inputs[2].shape.empty()) {
                            ref.dtype = compiled.inputs[2].dtype;
                            ref.shape = compiled.inputs[2].shape;
                        } else if (!compiled.inputs.empty()) {
                            ref.dtype = compiled.inputs[0].dtype;
                            // Fallback: routing_weights is [num_tokens, top_k]
                            int top_k = 1;
                            if (auto* attr = find_attr(op.attrs, "top_k")) {
                                if (auto v = attr_int(*attr)) {
                                    top_k = static_cast<int>(*v);
                                }
                            }
                            ref.shape = {B * T, static_cast<long>(top_k)};
                        }
                    }
                } else if (compiled.type == CompiledOpType::MoEGroupedGemmGateUpBackward ||
                           compiled.type == CompiledOpType::MoEGroupedGemmDownBackward) {
                    // inputs: d_out, saved.inp, weights, saved.scatter_indices
                    // output: d_inp (same shape/dtype as saved.inp, input[1])
                    if (compiled.inputs.size() > 1 && !compiled.inputs[1].shape.empty()) {
                        ref.dtype = compiled.inputs[1].dtype;
                        ref.shape = compiled.inputs[1].shape;
                    } else if (compiled.inputs.size() > 3 && !compiled.inputs[3].shape.empty()) {
                        // Fallback: infer total_tokens from scatter_indices length
                        ref.dtype = compiled.inputs[0].dtype;
                        const long total_tokens = compiled.inputs[3].shape[0];
                        ref.shape = {total_tokens, C};
                    } else if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                        // Fallback: inp is permuted input [total_tokens, C]
                        int top_k = 1;
                        if (auto* attr = find_attr(op.attrs, "top_k")) {
                            if (auto v = attr_int(*attr)) {
                                top_k = static_cast<int>(*v);
                            }
                        }
                        ref.shape = {B * T * top_k, C};
                    }
                } else if (compiled.type == CompiledOpType::Silu || compiled.type == CompiledOpType::Relu2 ||
                           compiled.type == CompiledOpType::Mul || compiled.type == CompiledOpType::Scale ||
                           compiled.type == CompiledOpType::Gelu || compiled.type == CompiledOpType::SiluBackward ||
                           compiled.type == CompiledOpType::Relu2Backward ||
                           compiled.type == CompiledOpType::MulBackward ||
                           compiled.type == CompiledOpType::GeluBackward) {
                    // Element-wise ops (and their backward) preserve input shape and dtype
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                        if (ref.shape.empty()) {
                            ref.shape = compiled.inputs[0].shape;
                        }
                    }
                } else if (compiled.type == CompiledOpType::ChunkGatedDeltaRule) {
                    // output[0] = out [B, T, H, V] (match value dtype)
                    // output[1] = final_state [B, H, K, V] (kept in FP32 for cache stability)
                    if (i == 0) {
                        if (compiled.inputs.size() > 2) {
                            ref.dtype = compiled.inputs[2].dtype;
                        } else if (!compiled.inputs.empty()) {
                            ref.dtype = compiled.inputs[0].dtype;
                        }
                        if (ref.shape.empty() && compiled.inputs.size() > 2 && compiled.inputs[0].shape.size() == 4 &&
                            compiled.inputs[2].shape.size() == 4) {
                            const auto& q_shape = compiled.inputs[0].shape;
                            const auto& v_shape = compiled.inputs[2].shape;
                            ref.shape = {q_shape[0], q_shape[1], q_shape[2], v_shape[3]};
                        }
                    } else if (i == 1) {
                        ref.dtype = ETensorDType::FP32;
                        if (ref.shape.empty() && compiled.inputs.size() > 2 && compiled.inputs[0].shape.size() == 4 &&
                            compiled.inputs[2].shape.size() == 4) {
                            const auto& q_shape = compiled.inputs[0].shape;
                            const auto& v_shape = compiled.inputs[2].shape;
                            ref.shape = {q_shape[0], q_shape[2], q_shape[3], v_shape[3]};
                        }
                    }
                } else if (compiled.type == CompiledOpType::ChunkGatedDeltaRuleBackward) {
                    // output[0] d_q, output[1] d_k -> q dtype/shape
                    // output[2] d_v -> v dtype/shape
                    // output[3] d_g -> g dtype/shape
                    // output[4] d_beta -> beta dtype/shape
                    // output[5] d_initial_state -> FP32 [B,H,K,V]
                    if (i == 0 || i == 1) {
                        if (compiled.inputs.size() > 2) {
                            ref.dtype = compiled.inputs[2].dtype;
                            ref.shape = compiled.inputs[2].shape;
                        }
                    } else if (i == 2) {
                        if (compiled.inputs.size() > 4) {
                            ref.dtype = compiled.inputs[4].dtype;
                            ref.shape = compiled.inputs[4].shape;
                        }
                    } else if (i == 3) {
                        if (compiled.inputs.size() > 5) {
                            ref.dtype = compiled.inputs[5].dtype;
                            ref.shape = compiled.inputs[5].shape;
                        }
                    } else if (i == 4) {
                        if (compiled.inputs.size() > 6) {
                            ref.dtype = compiled.inputs[6].dtype;
                            ref.shape = compiled.inputs[6].shape;
                        }
                    } else if (i == 5) {
                        ref.dtype = ETensorDType::FP32;
                        if (compiled.inputs.size() > 4 && compiled.inputs[2].shape.size() == 4 &&
                            compiled.inputs[4].shape.size() == 4) {
                            const auto& q_shape = compiled.inputs[2].shape;
                            const auto& v_shape = compiled.inputs[4].shape;
                            ref.shape = {q_shape[0], q_shape[2], q_shape[3], v_shape[3]};
                        }
                    }
                } else if (compiled.type == CompiledOpType::Qwen3_5Decay) {
                    // output[0] g = -exp(A_log) * softplus(a + dt_bias), always FP32.
                    ref.dtype = ETensorDType::FP32;
                    if (ref.shape.empty() && !compiled.inputs.empty()) {
                        ref.shape = compiled.inputs[0].shape;
                    }
                } else if (compiled.type == CompiledOpType::Qwen3_5DecayBackward) {
                    // outputs: d_a (same dtype/shape as a), d_A_log (FP32), d_dt_bias (FP32)
                    if (i == 0) {
                        if (compiled.inputs.size() > 1) {
                            ref.dtype = compiled.inputs[1].dtype;
                            ref.shape = compiled.inputs[1].shape;
                        }
                    } else if (i == 1 || i == 2) {
                        ref.dtype = ETensorDType::FP32;
                        const std::size_t src_idx = (i == 1) ? 2 : 3;
                        if (compiled.inputs.size() > src_idx) {
                            ref.shape = compiled.inputs[src_idx].shape;
                        }
                    }
                } else {
                    // Default for activation tensors — this is a best-effort guess
                    // that is often wrong for Mamba/custom ops; do NOT persist.
                    // Only apply the fallback if resolve_tensor_ref didn't already
                    // produce a shape. Overwriting a resolved shape breaks
                    // gradient slots whose per-layer/per-field dims differ from
                    // the hidden-size default (e.g., d_blocks[N].self_attn_q_rn_flat
                    // in hybrid models resolves to [B*T*Hq, Hs], not [B,T,C]).
                    ref.dtype = ETensorDType::BF16;
                    if (ref.shape.empty()) {
                        ref.shape = {B, T, C};
                        shape_is_default_fallback = true;
                        static std::once_flag warned_once;
                        std::call_once(warned_once, []() {
                            fprintf(stderr,
                                    "[graph_compiler] warning: using {B,T,C} fallback for unknown "
                                    "Mapped-slot output shape; set SUROGATE_DEBUG_SHAPE_FALLBACK=1 "
                                    "to see each occurrence.\n");
                            fflush(stderr);
                        });
                        if (const char* dbg = std::getenv("SUROGATE_DEBUG_SHAPE_FALLBACK");
                            dbg && std::string(dbg) == "1") {
                            fprintf(stderr,
                                    "[graph_compiler] shape fallback {B=%ld,T=%ld,C=%ld} applied to "
                                    "op=%s type=%d output[%zu]=%s (resolve_tensor_ref gave empty shape)\n",
                                    B,
                                    T,
                                    C,
                                    op.id.c_str(),
                                    static_cast<int>(compiled.type),
                                    i,
                                    op.outputs[i].c_str());
                            fflush(stderr);
                        }
                    }
                }
            }

            // Also fix dtype for pre-allocated RSTD slots (must be FP32)
            if ((compiled.type == CompiledOpType::FusedResidualRMSNorm && i == 2) ||
                (compiled.type == CompiledOpType::QKVQKNorm && (i == 1 || i == 2)) ||
                (compiled.type == CompiledOpType::QKVQKNormRoPE && (i == 1 || i == 2))) {
                ref.dtype = ETensorDType::FP32;
            }

            // Ensure embedding output writes into the persistent encoded buffer.
            // Only the MAIN embedding (whose output is the "encoded"/"x0" slot)
            // goes to the Encoded buffer. Other embeddings (e.g., Gemma4's
            // pli_embedding with weight [vocab, n_layers * PLI_D]) have a
            // different output dim and must not be coerced to [B,T,HiddenSize]
            // or the Encoded slot; derive their dim from the weight tensor.
            if (compiled.type == CompiledOpType::Embedding && i == 0) {
                const long Bdim = mB;
                const long Tdim = mT;
                long emb_dim = mConfig.HiddenSize;
                if (compiled.inputs.size() > 1 && compiled.inputs[1].shape.size() >= 2) {
                    emb_dim = compiled.inputs[1].shape.back();
                }
                const bool is_main_embedding = (emb_dim == mConfig.HiddenSize);
                if (is_main_embedding) {
                    ref.slot = TensorSlot::Encoded;
                }
                ref.shape = {Bdim, Tdim, emb_dim};
            }

            // If an explicit gradient dtype override is configured, apply it to parameter gradients.
            if (mOptions.GradientType.has_value() && ref.is_gradient) {
                const std::string grad_name = strip_ssa_suffix(ref.name);
                if (auto base = base_param_from_grad_heuristic(grad_name)) {
                    if (mWeights.has(*base)) {
                        ref.dtype = *mOptions.GradientType;
                        if (const char* env = std::getenv("SUROGATE_DEBUG_GRAD_DTYPE")) {
                            fprintf(stderr, "[DEBUG_GRAD_DTYPE] %s -> %s\n", ref.name.c_str(), dtype_to_str(ref.dtype));
                        }
                    }
                }
            }

            if (const char* env = std::getenv("SUROGATE_DEBUG_DTYPES")) {
                const GlobalRole dbg_role = global_role_for_name(strip_ssa_suffix(ref.name));
                if (dbg_role == GlobalRole::FinalNormOutput || dbg_role == GlobalRole::FinalNormOutFlat) {
                    fprintf(stderr,
                            "[DEBUG_DTYPES] op=%s output=%s dtype=%s\n",
                            op.id.c_str(),
                            ref.name.c_str(),
                            dtype_to_str(ref.dtype));
                }
            }

            // Track output dtype and shape for downstream operations to reference.
            // This allows intermediate tensors to have their dtypes/shapes properly propagated.
            // Skip shapes from the catch-all default — they are often wrong for custom ops.
            if (!op.outputs[i].empty()) {
                mTensorDtypes[op.outputs[i]] = ref.dtype;
                if (!shape_is_default_fallback && !ref.shape.empty() &&
                    mTensorShapes.find(op.outputs[i]) == mTensorShapes.end()) {
                    TensorShape ts;
                    ts.dims = ref.shape;
                    ts.inferred = true;
                    ts.source_op = op.id;
                    mTensorShapes[op.outputs[i]] = ts;
                }
            }

            compiled.outputs.push_back(std::move(ref));
        }

        // Pre-resolve attributes (use per-layer env for hybrid models)
        compiled.attrs = resolve_attrs(op, compiled.type, *env_ptr, graph);

        // Statistics
        if (compiled.type == CompiledOpType::Matmul || compiled.type == CompiledOpType::MatmulBias ||
            compiled.type == CompiledOpType::MatmulBackward) {
            result.matmul_ops++;
        } else if (compiled.type == CompiledOpType::View || compiled.type == CompiledOpType::ViewBackward) {
            result.view_ops++;
        }

        result.ops.push_back(std::move(compiled));
    }

    // Annotate layer boundaries for prefetch
    annotate_layer_boundaries(result);

    // Build phase tree. Runs after annotate_layer_boundaries so
    // layer_{start,end}_indices are populated.
    build_phase_tree(result, is_backward);

    // Register external tensor names (init bindings, MoE side-channel, param gradients)
    // that may not appear in any op's TensorRef but are used at runtime.
    register_external_names(result);

    // Build per-tensor metadata for pruning and the SSA base-to-ID map.
    build_tensor_metadata(result);

    // Pre-compute last-use information for tensor lifetime management.
    // This avoids rebuilding the last_use map on every backward pass.
    {
        auto& last_use = result.last_use_index;
        for (std::size_t i = 0; i < result.ops.size(); ++i) {
            const auto& cop = result.ops[i];
            for (const auto& ref : cop.inputs) {
                if (!ref.name.empty()) {
                    last_use[ref.name] = i;
                }
            }
            for (const auto& ref : cop.outputs) {
                if (!ref.name.empty()) {
                    last_use[ref.name] = i;
                }
            }
        }
        result.last_use_names.resize(result.ops.size());
        for (const auto& [tname, idx] : last_use) {
            if (idx < result.last_use_names.size()) {
                result.last_use_names[idx].push_back(tname);
            }
        }
    }

    // Classify every tensor (ForwardParam / ForwardActivation / ParamGrad /
    // ActivationGrad / AccumTemp / LossInput / Scratch) using authoritative
    // lookups against the parameter store and op producer map. Runtime code
    // will read TensorMeta::kind instead of string-matching names.
    // (Phase 0: data-only; callers still use legacy string predicates until
    // Phase 1 flips them.)
    classify_tensors(result);

    // Derive region assignment. Runs after classify_tensors so
    // TensorMeta::kind is populated.
    derive_regions(result, is_backward);

    // Promote cross-layer forward reads to SaveForBwd (forward graph only).
    // Must run before compute_layout so promoted tids end up in the right
    // region buckets. Backward already has bwd_cross_layer infrastructure.
    if (!is_backward) {
        promote_cross_layer_fwd_reads(result);
    }

    // Compute per-region peak bytes and bake TensorMeta::offset. Re-run
    // by finalize_save_for_bwd() once cross-graph region promotion is
    // applied.
    compute_layout(result, is_backward, /*fwd_per_layer_sections=*/!mOptions.recompute_enabled());

    // Flatten the phase tree to a linear instruction stream.
    emit_instruction_stream(result);

    // Debuggability dump (P4.7). Runs after every compile for inspection.
    maybe_dump_tid_table(result);

    // ========================================================================
    // Detect MLP tile groups for long-context tiled execution
    // ========================================================================
    if (mOptions.LongContext) {
        const auto& ops = result.ops;
        for (std::size_t i = 0; i < ops.size(); ++i) {
            // Look for matmul ops with mlp_up_weight
            if (ops[i].type != CompiledOpType::Matmul && ops[i].type != CompiledOpType::MatmulBias) continue;

            bool is_up = false;
            for (const auto& inp : ops[i].inputs) {
                if (inp.name.size() >= 13 && inp.name.compare(inp.name.size() - 13, 13, "mlp_up_weight") == 0) {
                    is_up = true;
                    break;
                }
            }
            if (!is_up) continue;

            // Found up-proj matmul at index i.
            // The view op before it is the group start.
            if (i == 0) continue;
            std::size_t start = i - 1;
            if (ops[start].type != CompiledOpType::View) continue;

            // Walk forward to find mlp_down_weight matmul
            std::size_t down_idx = 0;
            bool found_down = false;
            for (std::size_t j = i + 1; j < ops.size() && j <= i + 5; ++j) {
                if (ops[j].type != CompiledOpType::Matmul && ops[j].type != CompiledOpType::MatmulBias) continue;
                for (const auto& inp : ops[j].inputs) {
                    if (inp.name.size() >= 15 && inp.name.compare(inp.name.size() - 15, 15, "mlp_down_weight") == 0) {
                        down_idx = j;
                        found_down = true;
                        break;
                    }
                }
                if (found_down) break;
            }
            if (!found_down) continue;

            // The view op after the down matmul is the group end
            std::size_t end = down_idx + 1;
            if (end >= ops.size() || ops[end].type != CompiledOpType::View) {
                end = down_idx;  // fallback: end at the matmul itself
            }

            result.mlp_tile_groups.push_back(MlpTileGroup{start, end});
        }
        // Also detect backward MLP tile groups (MatmulBackward ops).
        // In the backward graph, the down-proj backward comes BEFORE the up-proj backward (reversed).
        // Backward sequence: view_bwd → matmul_bwd(down) → view_bwd → swiglu_bwd → view_bwd → matmul_bwd(up) → view_bwd
        for (std::size_t i = 0; i < ops.size(); ++i) {
            if (ops[i].type != CompiledOpType::MatmulBackward) continue;

            bool is_down = false;
            for (const auto& inp : ops[i].inputs) {
                if (inp.name.size() >= 15 && inp.name.compare(inp.name.size() - 15, 15, "mlp_down_weight") == 0) {
                    is_down = true;
                    break;
                }
            }
            if (!is_down) continue;

            // Found down-proj matmul_backward at index i.
            // The view_backward before it is the group start.
            if (i == 0) continue;
            std::size_t start = i - 1;
            if (ops[start].type != CompiledOpType::ViewBackward) start = i;

            // Walk forward to find mlp_up_weight matmul_backward
            std::size_t up_idx = 0;
            bool found_up = false;
            for (std::size_t j = i + 1; j < ops.size() && j <= i + 6; ++j) {
                if (ops[j].type != CompiledOpType::MatmulBackward) continue;
                for (const auto& inp : ops[j].inputs) {
                    if (inp.name.size() >= 13 && inp.name.compare(inp.name.size() - 13, 13, "mlp_up_weight") == 0) {
                        up_idx = j;
                        found_up = true;
                        break;
                    }
                }
                if (found_up) break;
            }
            if (!found_up) continue;

            // The view_backward after the up matmul_backward is the group end
            std::size_t end = up_idx + 1;
            if (end >= ops.size() || ops[end].type != CompiledOpType::ViewBackward) {
                end = up_idx;
            }

            result.mlp_tile_groups.push_back(MlpTileGroup{start, end});
        }

        if (!result.mlp_tile_groups.empty()) {
            std::fprintf(stderr,
                         "[long_context] Detected %zu MLP tile groups for tiled execution\n",
                         result.mlp_tile_groups.size());
        }
    }

    return result;
}

}  // namespace dsl
