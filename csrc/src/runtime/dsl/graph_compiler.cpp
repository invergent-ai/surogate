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
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "runtime/dsl/graph_compiler.h"
#include "runtime/dsl/graph_executor_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/core/backward_hooks.h"
#include "runtime/core/forward_hooks.h"

namespace dsl {

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
        const long C = config.HiddenSize;
        const long D = config.IntermediateSize;
        const long MUp = config.mlp_up_rows();
        const long Hq = config.NumQueryHeads;
        const long Hkv = config.NumKeyValHeads;
        const long Hs = config.head_size();
        const long QKV = config.qkv_channels();

        if (field == "ln1" || field == "ln2" || field == "att_out" || field == "mlp_down" ||
            field == "res_att" || field == "res_ffn" || field == "res_in") {
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

    if (name == "x0" || name == "encoded" || name == "ln_final" || name == "xF" ||
        name == "final_residual" || name == "residual_final") {
        shape = {B, T, config.HiddenSize};
        return true;
    }
    if (name == "ln_final_rstd") {
        shape = {B, T};
        return true;
    }
    if (name == "token_ids" || name == "position_ids") {
        shape = {B, T};
        return true;
    }
    if (name == "targets" || name == "labels" || name == "loss" || name == "losses" || name == "d_loss") {
        shape = {B * T};
        return true;
    }

    return false;
}

}


// ============================================================================
// Operation type conversion
// ============================================================================

CompiledOpType op_type_from_string(const std::string& op_type) {
    // Use a static lookup table for O(1) average case
    static const std::unordered_map<std::string, CompiledOpType> type_map = {
        {"embedding", CompiledOpType::Embedding},
        {"zeros", CompiledOpType::Zeros},
        {"fused_residual_rmsnorm", CompiledOpType::FusedResidualRMSNorm},
        {"layernorm", CompiledOpType::LayerNorm},
        {"view", CompiledOpType::View},
        {"add", CompiledOpType::Add},
        {"matmul", CompiledOpType::Matmul},
        {"matmul_bias", CompiledOpType::MatmulBias},
        {"bias_add", CompiledOpType::BiasAdd},
        {"swiglu", CompiledOpType::SwiGLU},
        {"gpt_oss_moe_act", CompiledOpType::GptOssMoeAct},
        {"silu", CompiledOpType::Silu},
        {"gelu", CompiledOpType::Gelu},
        {"relu2", CompiledOpType::Relu2},
        {"mul", CompiledOpType::Mul},
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
        // Backward operations
        {"view_backward", CompiledOpType::ViewBackward},
        {"add_backward", CompiledOpType::AddBackward},
        {"matmul_backward", CompiledOpType::MatmulBackward},
        {"bias_add_backward", CompiledOpType::BiasAddBackward},
        {"swiglu_backward", CompiledOpType::SwiGLUBackward},
        {"gpt_oss_moe_act_backward", CompiledOpType::GptOssMoeActBackward},
        {"silu_backward", CompiledOpType::SiluBackward},
        {"gelu_backward", CompiledOpType::GeluBackward},
        {"relu2_backward", CompiledOpType::Relu2Backward},
        {"mul_backward", CompiledOpType::MulBackward},
        {"mask_scatter_backward", CompiledOpType::MaskScatterBackward},
        {"deepstack_inject_backward", CompiledOpType::DeepstackInjectBackward},
        {"matmul_swiglu_backward", CompiledOpType::MatmulSwiGLUBackward},
        {"qkv_qk_norm_backward", CompiledOpType::QKVQKNormBackward},
        {"rope_backward", CompiledOpType::RoPEBackward},
        {"qkv_qk_norm_rope_backward", CompiledOpType::QKVQKNormRoPEBackward},
        {"mrope_backward", CompiledOpType::MRoPEBackward},
        {"flash_attention_backward", CompiledOpType::FlashAttentionBackward},
        {"zeros_backward", CompiledOpType::ZerosBackward},
        {"fused_residual_rmsnorm_backward", CompiledOpType::FusedResidualRMSNormBackward},
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
        // Mamba/SSM forward operations
        {"mamba_split_proj", CompiledOpType::MambaSplitProj},
        {"mamba_conv1d", CompiledOpType::MambaConv1d},
        {"mamba_split_conv_out", CompiledOpType::MambaSplitConvOut},
        {"mamba_ssm_scan", CompiledOpType::MambaSsmScan},
        {"mamba_gated_rmsnorm", CompiledOpType::MambaGatedRMSNorm},
        {"mamba_out_proj", CompiledOpType::MambaOutProj},
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
    : mModule(module)
    , mConfig(config)
    , mOptions(options)
    , mWeights(weights)
    , mGrads(grads)
{
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
    const long moe_m = (mConfig.MoeIntermediateSize > 0)
        ? mConfig.MoeIntermediateSize
        : mConfig.IntermediateSize;
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

TensorRef GraphCompiler::resolve_tensor_ref(const std::string& name, bool is_output,
                                            const Operation& op, const ShapeEnv& env) {
    TensorRef ref;
    ref.name = name;

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
                ref.shape = resolve_shape(slot_entry->shape, mShapeEnv);
            }
            if (slot_entry->dtype.has_value()) {
                ref.dtype = *slot_entry->dtype;
            }
        }
        if (ref.shape.empty()) {
            auto it = mExtraShapes.find(ref.name);
            if (it != mExtraShapes.end()) {
                ref.shape = it->second;
            }
        }
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
                return ref;
            }

            // Use shape from DSL if available
            if (!slot_entry->shape.empty()) {
                ref.shape = resolve_shape(slot_entry->shape, mShapeEnv);
            }
            // Override with extra shapes when present (e.g., view outputs with explicit shapes).
            if (auto it = mExtraShapes.find(ref.name); it != mExtraShapes.end()) {
                ref.shape = it->second;
            }
        } else if (mWeights.has(effective_name)) {
            // Block-indexed weight (e.g., blocks[0].ln1_weight)
            ref.slot = TensorSlot::Parameter;
            const auto& tmpl = mWeights.template_tensor(effective_name);
            if (tmpl.Rank > 0) {
                ref.shape.assign(tmpl.Sizes.begin(), tmpl.Sizes.begin() + tmpl.Rank);
            }
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

        return ref;
    }

    // Check for gradient tensors
    if (starts_with(name, "d_")) {
        const std::string base = name.substr(2);
        if (parse_block_param(base, layer_idx, field)) {
            ref.layer_idx = layer_idx;

            // Look up gradient slot using "d_<field>" name (e.g., "d_ln1", "d_qkv")
            const std::string grad_name = "d_" + strip_ssa_suffix(field);
            if (auto slot_entry = mSlotRegistry.lookup(grad_name)) {
                ref.slot = slot_entry->slot;
                if (!slot_entry->shape.empty()) {
                    ref.shape = resolve_shape(slot_entry->shape, mShapeEnv);
                }
            } else {
                const std::string act_name = strip_ssa_suffix(field);
                if (auto act_entry = mSlotRegistry.lookup(act_name)) {
                    if (!act_entry->shape.empty()) {
                        ref.shape = resolve_shape(act_entry->shape, mShapeEnv);
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

            if (ref.shape.empty()) {
                const std::string base = name.substr(2);
                auto it = mExtraShapes.find(base);
                if (it != mExtraShapes.end()) {
                    ref.shape = it->second;
                }
            }
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
    } else if (name.find("rope_freqs") != std::string::npos || name.find("freq_cis") != std::string::npos) {
        // Substring match for rope frequencies (handles qualified names)
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
    if (auto it = mTensorDtypes.find(ref.name); it != mTensorDtypes.end()) {
        ref.dtype = it->second;
    }
    return ref;
}


CompiledAttrs GraphCompiler::resolve_attrs(const Operation& op, CompiledOpType type,
                                           const ShapeEnv& env) {
    CompiledAttrs attrs;

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

    // Shape attribute (direct shape or shape_like reference)
    if (auto* shape_attr = find_attr(op.attrs, "shape")) {
        attrs.shape = resolve_attr_shape(*shape_attr, env);
    } else if (auto* shape_like_attr = find_attr(op.attrs, "shape_like")) {
        // Store the reference name for runtime lookup
        if (auto ref_name = attr_string(*shape_like_attr)) {
            attrs.shape_like = *ref_name;
        }
    }

    if (auto* acc_attr = find_attr(op.attrs, "compute_accuracy")) {
        if (auto v = attr_bool(*acc_attr)) {
            attrs.compute_accuracy = *v;
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
            attrs.matmul_op = matmul_op;
            attrs.layer_idx = layer_idx;
            attrs.allow_quant = matmul_op.has_value() &&
                                allow_quant_layer(mOptions, mConfig, layer_idx);
            if (matmul_op.has_value()) {
                switch (*matmul_op) {
                    case modules::MatmulOp::QKV:
                        attrs.forward_hook_point = modules::ForwardHookPoint::AfterQKVProjection;
                        break;
                    case modules::MatmulOp::AttnOut:
                        attrs.forward_hook_point = modules::ForwardHookPoint::AfterAttnOutProjection;
                        break;
                    case modules::MatmulOp::MLPUp:
                        attrs.forward_hook_point = modules::ForwardHookPoint::AfterMLPUpProjection;
                        break;
                    case modules::MatmulOp::MLPDown:
                        attrs.forward_hook_point = modules::ForwardHookPoint::AfterMLPDownProjection;
                        break;
                    default:
                        break;
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
            attrs.allow_quant = matmul_op.has_value() &&
                                allow_quant_layer(mOptions, mConfig, layer_idx);
            if (matmul_op.has_value() && *matmul_op == modules::MatmulOp::MLPUp) {
                attrs.forward_hook_point = modules::ForwardHookPoint::AfterMLPUpProjection;
            }
        }
    }

    // MatmulBackward: weight is at inputs[2], not inputs[1]
    // Also set backward_hook_point for LoRA hook invocation
    if (type == CompiledOpType::MatmulBackward) {
        if (op.inputs.size() > 2) {
            int layer_idx = -1;
            std::string field;
            if (parse_block_param(op.inputs[2], layer_idx, field)) {
                // Set matmul_op and layer_idx
                if (field == "qkv_weight") {
                    attrs.matmul_op = modules::MatmulOp::QKV;
                    attrs.backward_hook_point = modules::BackwardHookPoint::AfterQKVBackward;
                } else if (field == "out_weight") {
                    attrs.matmul_op = modules::MatmulOp::AttnOut;
                    attrs.backward_hook_point = modules::BackwardHookPoint::AfterAttnOutBackward;
                } else if (field == "mlp_up_weight" || field == "up_weight") {
                    attrs.matmul_op = modules::MatmulOp::MLPUp;
                    attrs.backward_hook_point = modules::BackwardHookPoint::AfterMLPUpBackward;
                } else if (field == "mlp_down_weight" || field == "down_weight") {
                    attrs.matmul_op = modules::MatmulOp::MLPDown;
                    attrs.backward_hook_point = modules::BackwardHookPoint::AfterMLPDownBackward;
                }
                attrs.layer_idx = layer_idx;
                attrs.allow_quant = attrs.matmul_op.has_value() &&
                                    allow_quant_layer(mOptions, mConfig, layer_idx);
            }
        }
    }

    // MatmulSwiGLUBackward: fused MLP up+gate backward uses weight at inputs[2]
    // Set backward_hook_point so LoRA can hook into MLPUp gradients.
    if (type == CompiledOpType::MatmulSwiGLUBackward) {
        if (op.inputs.size() > 2) {
            int layer_idx = -1;
            std::string field;
            if (parse_block_param(op.inputs[2], layer_idx, field)) {
                if (field == "mlp_up_weight" || field == "up_weight") {
                    attrs.matmul_op = modules::MatmulOp::MLPUp;
                    attrs.backward_hook_point = modules::BackwardHookPoint::AfterMLPUpBackward;
                }
                attrs.layer_idx = layer_idx;
                attrs.allow_quant = attrs.matmul_op.has_value() &&
                                    allow_quant_layer(mOptions, mConfig, layer_idx);
            }
        }
    }

    // MoE-specific attributes
    if (type == CompiledOpType::MoETopK || type == CompiledOpType::MoEPermute ||
        type == CompiledOpType::MoEUnpermute || type == CompiledOpType::MoETopKBackward ||
        type == CompiledOpType::MoEPermuteBackward || type == CompiledOpType::MoEUnpermuteBackward) {
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

    if (type == CompiledOpType::MoEGroupedGemmGateUp ||
        type == CompiledOpType::MoEGroupedGemmGateUpBackward) {
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
            case TensorSlot::BlockDLN2:
            case TensorSlot::BlockDResAtt:
            case TensorSlot::BlockDResFFN:
            case TensorSlot::DLoss:
                return true;
            default:
                return false;
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
            fprintf(stderr, "[DEBUG_SHAPES] resolve '%s' -> cache %s (src: %s)\n",
                    name.c_str(), format_shape(shape).c_str(),
                    it->second.source_op.c_str());
        }
        return true;
    }

    // Check IR tensor info
    auto check_tensor_info = [&](const std::unordered_map<std::string, TensorInfo>& tensors,
                                  const char* source) {
        auto it = tensors.find(name);
        if (it != tensors.end() && !it->second.shape.empty()) {
            shape = resolve_shape(it->second.shape, mShapeEnv);
            TensorShape ts;
            ts.dims = shape;
            ts.inferred = false;
            mTensorShapes[name] = ts;
            if (mDebugShapes && starts_with(name, "d_")) {
                fprintf(stderr, "[DEBUG_SHAPES] resolve '%s' -> IR %s %s\n",
                        name.c_str(), source, format_shape(shape).c_str());
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
        TensorShape ts;
        ts.dims = shape;
        ts.inferred = true;
        mTensorShapes[name] = ts;
        if (mDebugShapes && starts_with(name, "d_")) {
            fprintf(stderr, "[DEBUG_SHAPES] resolve '%s' -> inferred %s\n",
                    name.c_str(), format_shape(shape).c_str());
        }
        return true;
    }

    // Check for saved tensors (use base name)
    if (starts_with(name, kSavedPrefix)) {
        std::string base_name = std::string(name.substr(kSavedPrefix.size()));
        return resolve_tensor_shape(base_name, shape);
    }

    if (mDebugShapes && starts_with(name, "d_")) {
        fprintf(stderr, "[DEBUG_SHAPES] resolve '%s' -> FAILED (no shape found)\n",
                name.c_str());
    }
    return false;
}

void GraphCompiler::infer_output_shapes(
    const Operation& op,
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

        case CompiledOpType::Add: {
            // Output shape = broadcast(input shapes)
            if (!input_shapes.empty()) {
                output_shapes.push_back(input_shapes[0]);  // Simplified: assume same shape
            }
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
                auto out_shape = input_shapes[0];  // indices shape
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

        case CompiledOpType::Zeros: {
            // Try to infer from 'shape' attribute
            if (auto* shape_attr = find_attr(op.attrs, "shape")) {
                auto out_shape = resolve_attr_shape(*shape_attr, mShapeEnv);
                output_shapes.push_back(out_shape);
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

        case CompiledOpType::Silu:
        case CompiledOpType::Relu2:
        case CompiledOpType::Mul:
        case CompiledOpType::SiluBackward:
        case CompiledOpType::Relu2Backward:
        case CompiledOpType::MulBackward: {
            // Element-wise ops (and their backward) preserve shape
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
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

        default:
            // For other operations, output shape not inferred
            break;
    }
}


void GraphCompiler::validate_operation_shapes(
    const Operation& op,
    CompiledOpType type,
    size_t op_index) {

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
            fprintf(stderr, "[DEBUG_SHAPES] validate '%s' (id: %s) SKIPPED — unresolved inputs:",
                    op.name.c_str(), op.id.c_str());
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
                <<   "║ Found Shape Validation Error during Graph Compilation ║\n"
                <<   "╚═══════════════════════════════════════════════════════╝\n\n"
                <<   "Operation: #" << op_index << " (id: '" << op.id << "')\n"
                <<   "Type:      " << op.name << "\n\n";

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


CompiledGraph GraphCompiler::compile(const Graph& graph, long B, long T) {
    update_dimensions(B, T);

    mExtraShapes.clear();
    mTensorShapes.clear();
    mTensorDtypes.clear();

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
        const std::string& op_type =
            (op.kernel_type.empty() || op.kernel_type == "custom") ? op.name : op.kernel_type;

        CompiledOp compiled;
        compiled.original_idx = static_cast<std::uint16_t>(idx);
        compiled.op_id = op.id;
        compiled.type = classify_op(op_type);

        if (compiled.type == CompiledOpType::Unknown) {
            throw std::runtime_error("GraphCompiler: unsupported operation type: " + op_type);
        }

        // Validate operation shapes at compile time
        try {
            validate_operation_shapes(op, compiled.type, idx);
        } catch (const std::exception& e) {
            // Re-throw with additional context if validation fails
            std::cerr << "Shape validation failed during graph compilation.\n"
                      << "Operation: " << op.name << " (id: " << op.id << ")\n"
                      << "Error: " << e.what() << "\n";
            throw;
        }

        // Pre-resolve inputs
        compiled.inputs.reserve(op.inputs.size());
        for (const auto& input : op.inputs) {
            compiled.inputs.push_back(resolve_tensor_ref(input, false, op, mShapeEnv));
        }

        // Pre-resolve outputs
        compiled.outputs.reserve(op.outputs.size());
        for (std::size_t i = 0; i < op.outputs.size(); ++i) {
            auto ref = resolve_tensor_ref(op.outputs[i], true, op, mShapeEnv);
            bool shape_is_default_fallback = false;

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
                        ref.dtype = !compiled.inputs.empty() ? compiled.inputs[0].dtype
                                                             : ETensorDType::BF16;
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
                } else if (compiled.type == CompiledOpType::Add ||
                           compiled.type == CompiledOpType::BiasAdd) {
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
                } else if (compiled.type == CompiledOpType::Matmul ||
                           compiled.type == CompiledOpType::MatmulBias) {
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
                    ETensorDType base_dtype =
                        !compiled.inputs.empty() ? compiled.inputs[0].dtype : ETensorDType::BF16;
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
                } else if (compiled.type == CompiledOpType::Zeros) {
                    // Preserve explicit output dtype/shape from graph.
                    // Read dtype from op attributes if specified
                    if (auto* dtype_attr = find_attr(op.attrs, "dtype")) {
                        if (auto dtype_str = attr_string(*dtype_attr)) {
                            ref.dtype = dtype_from_str(*dtype_str);
                        }
                    }
                    if (ref.shape.empty()) {
                        ref.shape = {B, T, C};
                    }
                } else if (compiled.type == CompiledOpType::RoPE ||
                           compiled.type == CompiledOpType::RoPEBackward) {
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
                } else if (compiled.type == CompiledOpType::View ||
                           compiled.type == CompiledOpType::ViewBackward) {
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
                        if (find_attr(op.attrs, "shape")) source = "shape attr";
                        else if (find_attr(op.attrs, "shape_like")) source = "shape_like attr";
                        fprintf(stderr, "[DEBUG_SHAPES] View output '%s' shape=%s (via %s)\n",
                                op.outputs[i].c_str(),
                                ref.shape.empty() ? "<empty>" : fmt(ref.shape).c_str(),
                                source);
                    }
                } else if (compiled.type == CompiledOpType::MoESigmoid ||
                           compiled.type == CompiledOpType::MoESoftmax) {
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
                    if (!compiled.inputs.empty() && compiled.inputs.size() > 1 &&
                        !compiled.inputs[1].shape.empty()) {
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
                } else if (compiled.type == CompiledOpType::Silu ||
                           compiled.type == CompiledOpType::Relu2 ||
                           compiled.type == CompiledOpType::Mul ||
                           compiled.type == CompiledOpType::SiluBackward ||
                           compiled.type == CompiledOpType::Relu2Backward ||
                           compiled.type == CompiledOpType::MulBackward) {
                    // Element-wise ops (and their backward) preserve input shape and dtype
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                        if (ref.shape.empty()) {
                            ref.shape = compiled.inputs[0].shape;
                        }
                    }
                } else {
                    // Default for activation tensors — this is a best-effort guess
                    // that is often wrong for Mamba/custom ops; do NOT persist.
                    ref.dtype = ETensorDType::BF16;
                    ref.shape = {B, T, C};
                    shape_is_default_fallback = true;
                }
            }

            // Also fix dtype for pre-allocated RSTD slots (must be FP32)
            if ((compiled.type == CompiledOpType::FusedResidualRMSNorm && i == 2) ||
                (compiled.type == CompiledOpType::QKVQKNorm && (i == 1 || i == 2)) ||
                (compiled.type == CompiledOpType::QKVQKNormRoPE && (i == 1 || i == 2))) {
                ref.dtype = ETensorDType::FP32;
            }

            // Ensure embedding output writes into the persistent encoded buffer.
            if (compiled.type == CompiledOpType::Embedding && i == 0) {
                const long Bdim = mB;
                const long Tdim = mT;
                const long Cdim = mConfig.HiddenSize;
                ref.slot = TensorSlot::Encoded;
                ref.shape = {Bdim, Tdim, Cdim};
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

        // Pre-resolve attributes
        compiled.attrs = resolve_attrs(op, compiled.type, mShapeEnv);

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

    return result;
}


}
