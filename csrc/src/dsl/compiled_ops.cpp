// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Compiled operation dispatch for DSL Graph executor.

#include "dsl/compiled_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "dsl/dsl_runtime.h"
#include "dsl/dsl_weight_manager.h"
#include "dsl/graph_executor_helpers.h"
#include "dsl/graph_executor_utils.h"
#include "dsl/op_shape_signatures.h"
#include "modules/backward_hooks.h"
#include "modules/forward_hooks.h"
#include "modules/fp8_scaling_config.h"
#include "modules/matmul_context.h"
#include "modules/model_config.h"
#include "recipes/recipe.h"
#include "training/runtime_options.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"

namespace dsl {
namespace {

float env_float(const char* name, float fallback) {
    if (!name || !*name) {
        return fallback;
    }
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char* end = nullptr;
    float out = std::strtof(value, &end);
    if (end == value) {
        return fallback;
    }
    return out;
}

bool infer_known_tensor_shape(std::string_view name,
                              const modules::ModelConfig& config,
                              long B,
                              long T,
                              std::vector<long>& shape) {
    if (starts_with(name, kSavedPrefix)) {
        name = name.substr(kSavedPrefix.size());
    }

    int layer_idx = -1;
    std::string field;
    if (parse_block_param(name, layer_idx, field)) {
        const long C = config.HiddenSize;
        const long D = config.IntermediateSize;
        const long Hq = config.NumQueryHeads;
        const long Hkv = config.NumKeyValHeads;
        const long Hs = config.head_size();
        const long QKV = config.qkv_channels();

        if (field == "ln1" || field == "ln2" || field == "att_out" || field == "mlp_down" ||
            field == "res_att" || field == "res_ffn") {
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
            shape = {B, T, 2 * D};
            return true;
        }
        if (field == "mlp_up_flat") {
            shape = {B * T, 2 * D};
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

    if (name == "x0" || name == "encoded" || name == "ln_final" || name == "xF" || name == "final_residual") {
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

}  // namespace

// ============================================================================
// Operation type conversion
// ============================================================================

CompiledOpType op_type_from_string(const std::string& op_type) {
    // Use a static lookup table for O(1) average case
    static const std::unordered_map<std::string, CompiledOpType> type_map = {
        {"embedding", CompiledOpType::Embedding},
        {"zeros", CompiledOpType::Zeros},
        {"fused_residual_rmsnorm", CompiledOpType::FusedResidualRMSNorm},
        {"view", CompiledOpType::View},
        {"add", CompiledOpType::Add},
        {"matmul", CompiledOpType::Matmul},
        {"matmul_bias", CompiledOpType::MatmulBias},
        {"bias_add", CompiledOpType::BiasAdd},
        {"swiglu", CompiledOpType::SwiGLU},
        {"matmul_swiglu", CompiledOpType::MatmulSwiGLU},
        {"qkv_qk_norm_rope", CompiledOpType::QKVQKNormRoPE},
        {"rope", CompiledOpType::RoPE},
        {"flash_attention", CompiledOpType::FlashAttention},
        {"flash_attention_qkv", CompiledOpType::FlashAttention},
        {"cross_entropy", CompiledOpType::CrossEntropyLoss},
        {"cross_entropy_loss", CompiledOpType::CrossEntropyLoss},
        {"fused_lm_head_loss", CompiledOpType::FusedLMHeadLoss},
        {"lm_head_loss", CompiledOpType::FusedLMHeadLoss},
        // Backward operations
        {"view_backward", CompiledOpType::ViewBackward},
        {"add_backward", CompiledOpType::AddBackward},
        {"matmul_backward", CompiledOpType::MatmulBackward},
        {"bias_add_backward", CompiledOpType::BiasAddBackward},
        {"swiglu_backward", CompiledOpType::SwiGLUBackward},
        {"matmul_swiglu_backward", CompiledOpType::MatmulSwiGLUBackward},
        {"rope_backward", CompiledOpType::RoPEBackward},
        {"qkv_qk_norm_rope_backward", CompiledOpType::QKVQKNormRoPEBackward},
        {"flash_attention_backward", CompiledOpType::FlashAttentionBackward},
        {"zeros_backward", CompiledOpType::ZerosBackward},
        {"fused_residual_rmsnorm_backward", CompiledOpType::FusedResidualRMSNormBackward},
        {"embedding_backward", CompiledOpType::EmbeddingBackward},
        {"cross_entropy_backward", CompiledOpType::CrossEntropyLossBackward},
        {"fused_lm_head_loss_backward", CompiledOpType::FusedLMHeadLossBackward},
    };

    auto it = type_map.find(op_type);
    return it != type_map.end() ? it->second : CompiledOpType::Unknown;
}

const char* op_type_to_string(CompiledOpType type) {
    switch (type) {
        case CompiledOpType::Embedding: return "embedding";
        case CompiledOpType::Zeros: return "zeros";
        case CompiledOpType::FusedResidualRMSNorm: return "fused_residual_rmsnorm";
        case CompiledOpType::View: return "view";
        case CompiledOpType::Add: return "add";
        case CompiledOpType::Matmul: return "matmul";
        case CompiledOpType::MatmulBias: return "matmul_bias";
        case CompiledOpType::BiasAdd: return "bias_add";
        case CompiledOpType::SwiGLU: return "swiglu";
        case CompiledOpType::MatmulSwiGLU: return "matmul_swiglu";
        case CompiledOpType::QKVQKNormRoPE: return "qkv_qk_norm_rope";
        case CompiledOpType::RoPE: return "rope";
        case CompiledOpType::FlashAttention: return "flash_attention";
        case CompiledOpType::CrossEntropyLoss: return "cross_entropy_loss";
        case CompiledOpType::FusedLMHeadLoss: return "fused_lm_head_loss";
        case CompiledOpType::ViewBackward: return "view_backward";
        case CompiledOpType::AddBackward: return "add_backward";
        case CompiledOpType::MatmulBackward: return "matmul_backward";
        case CompiledOpType::BiasAddBackward: return "bias_add_backward";
        case CompiledOpType::SwiGLUBackward: return "swiglu_backward";
        case CompiledOpType::MatmulSwiGLUBackward: return "matmul_swiglu_backward";
        case CompiledOpType::RoPEBackward: return "rope_backward";
        case CompiledOpType::QKVQKNormRoPEBackward: return "qkv_qk_norm_rope_backward";
        case CompiledOpType::FlashAttentionBackward: return "flash_attention_backward";
        case CompiledOpType::ZerosBackward: return "zeros_backward";
        case CompiledOpType::FusedResidualRMSNormBackward: return "fused_residual_rmsnorm_backward";
        case CompiledOpType::EmbeddingBackward: return "embedding_backward";
        case CompiledOpType::CrossEntropyLossBackward: return "cross_entropy_backward";
        case CompiledOpType::FusedLMHeadLossBackward: return "fused_lm_head_loss_backward";
        case CompiledOpType::Unknown: return "unknown";
    }
    return "unknown";
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
{}

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
    mShapeEnv.values["M"] = mConfig.IntermediateSize;
    mShapeEnv.values["MUp"] = 2 * mConfig.IntermediateSize;
    mShapeEnv.values["V"] = mConfig.VocabSize;
    mShapeEnv.values["Hq"] = mConfig.NumQueryHeads;
    mShapeEnv.values["Hkv"] = mConfig.NumKeyValHeads;
    mShapeEnv.values["QKV"] = mConfig.qkv_channels();
    mShapeEnv.values["AttnDim"] = mConfig.NumQueryHeads * mConfig.head_size();
}

CompiledOpType GraphCompiler::classify_op(const std::string& op_type) const {
    return op_type_from_string(op_type);
}

TensorRef GraphCompiler::resolve_tensor_ref(const std::string& name, bool is_output,
                                            const Operation& op, const ShapeEnv& env) {
    TensorRef ref;
    ref.name = name;

    // Check for saved tensor prefix
    if (starts_with(name, kSavedPrefix)) {
        ref.slot = TensorSlot::Saved;
        ref.name = std::string(name.substr(kSavedPrefix.size()));
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
    if (parse_block_param(name, layer_idx, field)) {
        ref.layer_idx = layer_idx;

        // Map field to slot
        if (field == "ln1" || field == "ln1_flat") {
            ref.slot = TensorSlot::BlockLN1;
        } else if (field == "ln1_rstd") {
            ref.slot = TensorSlot::BlockLN1RSTD;
        } else if (field == "ln2" || field == "ln2_flat") {
            ref.slot = TensorSlot::BlockLN2;
        } else if (field == "ln2_rstd") {
            ref.slot = TensorSlot::BlockLN2RSTD;
        } else if (field == "q_rstd") {
            ref.slot = TensorSlot::BlockQRSTD;
        } else if (field == "k_rstd") {
            ref.slot = TensorSlot::BlockKRSTD;
        } else if (field == "qkv" || field == "qkv_flat" || field == "qkv_biased") {
            ref.slot = TensorSlot::BlockQKV;
        } else if (field == "qkv_rope") {
            ref.slot = TensorSlot::BlockQKVRoPE;
        } else if (field == "lse") {
            ref.slot = TensorSlot::BlockLSE;
        } else if (field == "att" || field == "att_flat") {
            ref.slot = TensorSlot::BlockAtt;
        } else if (field == "att_out" || field == "att_out_flat") {
            ref.slot = TensorSlot::BlockAttOut;
        } else if (field == "res_att") {
            ref.slot = TensorSlot::BlockResidualAtt;
        } else if (field == "mlp_up" || field == "mlp_up_flat") {
            ref.slot = TensorSlot::BlockMLPUp;
        } else if (field == "swiglu" || field == "swiglu_flat") {
            ref.slot = TensorSlot::BlockSwiGLU;
        } else if (field == "mlp_down" || field == "mlp_down_flat") {
            ref.slot = TensorSlot::BlockMLPDown;
        } else if (field == "res_ffn") {
            ref.slot = TensorSlot::BlockResidualFFN;
        } else if (field == "rope_freqs" || field == "freq_cis") {
            // Block-indexed rope frequencies refer to global FreqCis tensor
            ref.slot = TensorSlot::FreqCis;
            ref.layer_idx = -1;  // Global, not layer-indexed
            return ref;
        } else if (mWeights.has(name)) {
            // Block-indexed weight (e.g., blocks[0].ln1_weight)
            ref.slot = TensorSlot::Parameter;
            return ref;
        } else {
            ref.slot = TensorSlot::Mapped;
        }

        // Pre-compute shape based on field
        const long B = mB;
        const long T = mT;
        const long C = mConfig.HiddenSize;
        const long D = mConfig.IntermediateSize;
        const long Hq = mConfig.NumQueryHeads;
        const long Hkv = mConfig.NumKeyValHeads;
        const long Hs = mConfig.head_size();
        const long QKV = mConfig.qkv_channels();

        if (field == "ln1" || field == "ln2" || field == "att_out" || field == "mlp_down" ||
            field == "res_att" || field == "res_ffn") {
            ref.shape = {B, T, C};
        } else if (field == "ln1_flat" || field == "ln2_flat" || field == "att_out_flat" || field == "mlp_down_flat") {
            ref.shape = {B * T, C};
        } else if (field == "ln1_rstd" || field == "ln2_rstd") {
            ref.shape = {B, T};
        } else if (field == "qkv" || field == "qkv_rope") {
            ref.shape = {B, T, QKV};
        } else if (field == "qkv_flat" || field == "qkv_biased") {
            ref.shape = {B * T, QKV};
        } else if (field == "q_rstd") {
            ref.shape = {B, T, Hq};
        } else if (field == "k_rstd") {
            ref.shape = {B, T, Hkv};
        } else if (field == "att") {
            ref.shape = {B, T, Hq * Hs};
        } else if (field == "att_flat") {
            ref.shape = {B * T, Hq * Hs};
        } else if (field == "lse") {
            ref.shape = {B, Hq, T};
        } else if (field == "mlp_up") {
            ref.shape = {B, T, 2 * D};
        } else if (field == "mlp_up_flat") {
            ref.shape = {B * T, 2 * D};
        } else if (field == "swiglu") {
            ref.shape = {B, T, D};
        } else if (field == "swiglu_flat") {
            ref.shape = {B * T, D};
        }

        return ref;
    }

    // Check for gradient tensors
    if (starts_with(name, "d_")) {
        const std::string base = name.substr(2);
        if (parse_block_param(base, layer_idx, field)) {
            ref.layer_idx = layer_idx;

            if (field == "ln1" || field == "ln1_flat") {
                ref.slot = TensorSlot::BlockDLN1;
            } else if (field == "qkv" || field == "qkv_rope" || field == "qkv_flat" || field == "qkv_biased") {
                ref.slot = TensorSlot::BlockDQKV;
            } else if (field == "att" || field == "att_flat") {
                ref.slot = TensorSlot::BlockDAtt;
            } else if (field == "att_out" || field == "att_out_flat") {
                // att_out gradients share storage with residual-att gradients
                ref.slot = TensorSlot::BlockDResAtt;
            } else if (field == "swiglu" || field == "swiglu_flat") {
                ref.slot = TensorSlot::BlockDSwiGLU;
            } else if (field == "mlp_up" || field == "mlp_up_flat") {
                ref.slot = TensorSlot::BlockDMLPUp;
            } else if (field == "mlp_down" || field == "mlp_down_flat") {
                ref.slot = TensorSlot::BlockDMLPDown;
            } else if (field == "ln2" || field == "ln2_flat") {
                ref.slot = TensorSlot::BlockDLN2;
            } else if (field == "res_att") {
                ref.slot = TensorSlot::BlockDResAtt;
            } else if (field == "res_ffn") {
                ref.slot = TensorSlot::BlockDResFFN;
            } else {
                ref.slot = TensorSlot::Mapped;
            }

            // Pre-compute shape for gradient tensors
            // Gradient tensors have same shape as corresponding activation tensors
            const long B = mB;
            const long T = mT;
            const long C = mConfig.HiddenSize;
            const long D = mConfig.IntermediateSize;
            const long Hq = mConfig.NumQueryHeads;
            const long Hs = mConfig.head_size();
            const long QKV = mConfig.qkv_channels();

            if (field == "ln1" || field == "ln2" || field == "att_out" || field == "mlp_down" ||
                field == "res_att" || field == "res_ffn") {
                ref.shape = {B, T, C};
            } else if (field == "ln1_flat" || field == "ln2_flat" || field == "att_out_flat" ||
                       field == "mlp_down_flat") {
                ref.shape = {B * T, C};
            } else if (field == "qkv" || field == "qkv_rope") {
                ref.shape = {B, T, QKV};
            } else if (field == "qkv_flat" || field == "qkv_biased") {
                ref.shape = {B * T, QKV};
            } else if (field == "att") {
                ref.shape = {B, T, Hq * Hs};
            } else if (field == "att_flat") {
                ref.shape = {B * T, Hq * Hs};
            } else if (field == "swiglu") {
                ref.shape = {B, T, D};
            } else if (field == "swiglu_flat") {
                ref.shape = {B * T, D};
            } else if (field == "mlp_up") {
                ref.shape = {B, T, 2 * D};
            } else if (field == "mlp_up_flat") {
                ref.shape = {B * T, 2 * D};
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

    // Check for global tensors
    if (name == "token_ids") {
        ref.slot = TensorSlot::TokenIDs;
    } else if (name == "position_ids") {
        ref.slot = TensorSlot::PositionIDs;
    } else if (name == "targets" || name == "labels") {
        ref.slot = TensorSlot::Targets;
        ref.dtype = ETensorDType::INT32;
    } else if (name == "loss" || name == "losses") {
        ref.slot = TensorSlot::Losses;
        ref.dtype = ETensorDType::FP32;
    } else if (name == "d_loss") {
        ref.slot = TensorSlot::DLoss;
        ref.dtype = ETensorDType::FP32;
    } else if (name == "x0" || name == "encoded") {
        ref.slot = TensorSlot::Encoded;
    } else if (name == "ln_final" || name == "xF") {
        ref.slot = TensorSlot::LNFinal;
    } else if (name == "ln_final_rstd") {
        ref.slot = TensorSlot::LNFinalRSTD;
    } else if (name == "final_residual") {
        ref.slot = TensorSlot::FinalResidual;
    } else if (name.find("rope_freqs") != std::string::npos || name.find("freq_cis") != std::string::npos) {
        ref.slot = TensorSlot::FreqCis;
    } else if (mWeights.has(name)) {
        ref.slot = TensorSlot::Parameter;
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
                } else if (field == "mlp_up_weight") {
                    attrs.matmul_op = modules::MatmulOp::MLPUp;
                    attrs.backward_hook_point = modules::BackwardHookPoint::AfterMLPUpBackward;
                } else if (field == "mlp_down_weight") {
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
                if (field == "mlp_up_weight") {
                    attrs.matmul_op = modules::MatmulOp::MLPUp;
                    attrs.backward_hook_point = modules::BackwardHookPoint::AfterMLPUpBackward;
                }
                attrs.layer_idx = layer_idx;
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
    // Check shape cache first
    auto it = mTensorShapes.find(name);
    if (it != mTensorShapes.end()) {
        shape = it->second.dims;
        return true;
    }

    // Check IR tensor info
    auto check_tensor_info = [&](const std::unordered_map<std::string, TensorInfo>& tensors) {
        auto it = tensors.find(name);
        if (it != tensors.end() && !it->second.shape.empty()) {
            shape = resolve_shape(it->second.shape, mShapeEnv);
            TensorShape ts;
            ts.dims = shape;
            ts.inferred = false;
            mTensorShapes[name] = ts;
            return true;
        }
        return false;
    };

    // Check in graph tensors
    if (check_tensor_info(mModule.forward->inputs)) return true;
    if (check_tensor_info(mModule.forward->outputs)) return true;
    if (check_tensor_info(mModule.forward->params)) return true;
    if (check_tensor_info(mModule.forward->intermediates)) return true;

    // Try pattern-based inference for known tensor names
    if (infer_known_tensor_shape(name, mConfig, mB, mT, shape)) {
        TensorShape ts;
        ts.dims = shape;
        ts.inferred = true;
        mTensorShapes[name] = ts;
        return true;
    }

    // Check for saved tensors (use base name)
    if (starts_with(name, kSavedPrefix)) {
        std::string base_name = std::string(name.substr(kSavedPrefix.size()));
        return resolve_tensor_shape(base_name, shape);
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
            // Output shape from attributes
            if (auto* shape_attr = find_attr(op.attrs, "shape")) {
                auto out_shape = resolve_attr_shape(*shape_attr, mShapeEnv);
                output_shapes.push_back(out_shape);
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
            // FlashAttention outputs: attn_out [B, T, Hq, D] (or [B, T, Hq*D]), lse [B, Hq, T]
            // For now, attn_out shape matches input qkv shape but with Hq instead of (Hq+2*Hkv)
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                output_shapes.push_back(input_shapes[0]);  // Simplified: same shape as input
                // LSE shape - hard to infer without model config, leave empty for now
                output_shapes.push_back({});
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

    // Check if validation is disabled via environment variable
    if (env_enabled("SUROGATE_NO_SHAPE_CHECK")) {
        return;
    }

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
        // Skip validation when input shapes are unknown
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
                << "  Hidden size: " << mConfig.HiddenSize << "\n\n"
                << "To disable shape checking (not recommended):\n"
                << "  export SUROGATE_NO_SHAPE_CHECK=1\n\n";

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
                      << "To disable shape checking, set SUROGATE_NO_SHAPE_CHECK=1\n";
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
                    // View preserves dtype from input; shape comes from attributes
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                    }
                    // Shape is set from attrs in resolve_attrs, not here
                } else {
                    // Default for activation tensors
                    ref.dtype = ETensorDType::BF16;
                    ref.shape = {B, T, C};
                }
            }

            // Also fix dtype for pre-allocated RSTD slots (must be FP32)
            if ((compiled.type == CompiledOpType::FusedResidualRMSNorm && i == 2) ||
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

            // Track output dtype for downstream operations to reference.
            // This allows intermediate tensors to have their dtypes properly propagated.
            if (!op.outputs[i].empty()) {
                mTensorDtypes[op.outputs[i]] = ref.dtype;
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

// ============================================================================
// CompiledExecutor implementation
// ============================================================================

CompiledExecutor::CompiledExecutor(DslRunState& run_state,
                                   DslParamStore& weights,
                                   DslGradStore& grads,
                                   const modules::ModelConfig& config,
                                   const RuntimeOptions& options)
    : mRunState(run_state)
    , mWeights(weights)
    , mGrads(grads)
    , mConfig(config)
    , mOptions(options)
{}

void CompiledExecutor::set_lora_state(const modules::ModularLoRAConfig* config,
                                      modules::ModularLoRAWeightsManager* weights,
                                      modules::ModularLoRAGradsManager* grads,
                                      modules::LoRARunState* run_state) {
    mLoRAConfig = config;
    mLoRAWeights = weights;
    mLoRAGrads = grads;
    mLoRARunState = run_state;
}

void CompiledExecutor::set_weight_manager(DslWeightManager* weight_manager) {
    mWeightManager = weight_manager;
}

void CompiledExecutor::set_recipe(const recipes::Recipe* recipe) {
    mRecipe = recipe;
}

void CompiledExecutor::set_hook_context(void* context) {
    mHookContext = context;
}

void CompiledExecutor::set_recompute_fn(std::function<void(int, long, long, bool)> fn) {
    mRecomputeFn = std::move(fn);
}

void CompiledExecutor::set_recompute_enabled(bool enabled) {
    mRecomputeEnabled = enabled;
    mLastRecomputeLayer = -1;
}

void CompiledExecutor::set_fp8_cache(std::unordered_map<std::string, FP8WeightCacheEntry>* cache) {
    mFP8Cache = cache;
}

void CompiledExecutor::set_fp4_cache(std::unordered_map<std::string, FP4WeightCacheEntry>* cache,
                                     std::unordered_map<std::string, FP4WeightCacheEntry>* cache_t) {
    mFP4Cache = cache;
    mFP4CacheT = cache_t;
}

void CompiledExecutor::set_saved_tensors(std::unordered_map<std::string, Tensor>* saved) {
    mSaved = saved;
}

void CompiledExecutor::set_save_list(const std::vector<std::string>* save_list) {
    mSaveList = save_list;
    mSaveSet.clear();
    if (save_list) {
        mSaveSet.insert(save_list->begin(), save_list->end());
    }
}

void CompiledExecutor::set_last_inputs_cpu(const Tensor* inputs_cpu) {
    mLastInputsCpu = inputs_cpu;
}

void CompiledExecutor::set_rng_seed_fn(std::function<unsigned int()> fn) {
    mRngSeedFn = std::move(fn);
}

const Tensor* CompiledExecutor::try_get_tensor(const std::string& name) const {
    auto it = mTensorMap.find(name);
    if (it == mTensorMap.end()) {
        return nullptr;
    }
    return &it->second;
}

void CompiledExecutor::save_tensors(const std::vector<std::string>& save_list) {
    if (!mSaved) {
        return;
    }

    for (const auto& name : save_list) {
        // First check the tensor map (intermediate tensors)
        auto it = mTensorMap.find(name);
        if (it != mTensorMap.end()) {
            (*mSaved)[name] = it->second;
            continue;
        }

        // Check special tensors
        if (name == "token_ids") {
            (*mSaved)[name] = mRunState.Inputs;
            continue;
        }
        if (name == "position_ids") {
            (*mSaved)[name] = mRunState.PositionIDs;
            continue;
        }

        // Try to look up as a pre-allocated activation by creating a TensorRef
        // This handles tensors like "blocks[0].ln1_rstd" that map to slots
        TensorRef ref;
        ref.name = name;
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            ref.layer_idx = layer_idx;
            // Map common saved fields
            if (field == "ln1_rstd") {
                (*mSaved)[name] = mRunState.simplified_acts(layer_idx).ln1_rstd;
            } else if (field == "ln2_rstd") {
                (*mSaved)[name] = mRunState.simplified_acts(layer_idx).ln2_rstd;
            } else if (field == "q_rstd") {
                (*mSaved)[name] = mRunState.simplified_acts(layer_idx).q_rstd;
            } else if (field == "k_rstd") {
                (*mSaved)[name] = mRunState.simplified_acts(layer_idx).k_rstd;
            } else if (field == "lse") {
                (*mSaved)[name] = mRunState.simplified_acts(layer_idx).lse;
            } else if (field == "ln1" || field == "ln1_flat") {
                (*mSaved)[name] = mRunState.simplified_acts(layer_idx).ln1;
            } else if (field == "ln2" || field == "ln2_flat") {
                (*mSaved)[name] = mRunState.simplified_acts(layer_idx).ln2;
            } else if (field == "qkv" || field == "qkv_rope") {
                (*mSaved)[name] = mRunState.simplified_acts(layer_idx).qkv;
            } else if (field == "qkv_flat") {
                // Save the flattened version for matmul backward shape resolution
                Tensor qkv = mRunState.simplified_acts(layer_idx).qkv;
                Tensor flat = view_tensor(qkv, {qkv.Sizes[0] * qkv.Sizes[1], qkv.Sizes[2]});
                (*mSaved)[name] = flat;
            } else if (field == "att" || field == "att_flat") {
                (*mSaved)[name] = mRunState.simplified_acts(layer_idx).att;
            } else if (field == "att_out" || field == "att_out_flat") {
                (*mSaved)[name] = mRunState.simplified_acts(layer_idx).att_out;
            } else if (field == "mlp_up" || field == "mlp_up_flat") {
                (*mSaved)[name] = mRunState.simplified_acts(layer_idx).mlp_up;
            } else if (field == "swiglu") {
                (*mSaved)[name] = mRunState.simplified_acts(layer_idx).swiglu;
            } else if (field == "swiglu_flat") {
                Tensor swiglu = mRunState.simplified_acts(layer_idx).swiglu;
                Tensor flat = view_tensor(swiglu, {swiglu.Sizes[0] * swiglu.Sizes[1], swiglu.Sizes[2]});
                (*mSaved)[name] = flat;
            } else if (field == "mlp_down" || field == "mlp_down_flat") {
                (*mSaved)[name] = mRunState.simplified_acts(layer_idx).mlp_down;
            } else if (field == "res_att" || field == "residual_att") {
                (*mSaved)[name] = mRunState.simplified_acts(layer_idx).residual_att;
            } else if (field == "res_ffn" || field == "residual_ffn") {
                // res_ffn is computed dynamically (residual_att + mlp_down), check mTensorMap
                auto it = mTensorMap.find(name);
                if (it != mTensorMap.end()) {
                    (*mSaved)[name] = it->second;
                } else {
                    throw std::runtime_error("CompiledExecutor: res_ffn tensor not found in map: " + name);
                }
            } else if (mWeights.has(name)) {
                (*mSaved)[name] = mWeights.get(name);
            } else {
                throw std::runtime_error("CompiledExecutor: cannot save tensor " + name);
            }
        } else if (name == "ln_final" || name == "xF") {
            (*mSaved)[name] = mRunState.non_block_activations().ln_final;
        } else if (name == "xF_flat") {
            // Save the flattened version for matmul backward
            Tensor ln_final = mRunState.non_block_activations().ln_final;
            Tensor flat = view_tensor(ln_final, {ln_final.Sizes[0] * ln_final.Sizes[1], ln_final.Sizes[2]});
            (*mSaved)[name] = flat;
        } else if (name == "ln_final_rstd") {
            (*mSaved)[name] = mRunState.non_block_activations().ln_final_rstd;
        } else if (name == "encoded" || name == "x0") {
            (*mSaved)[name] = mRunState.non_block_activations().encoded;
        } else if (name.find("rope_freqs") != std::string::npos || name.find("freq_cis") != std::string::npos) {
            (*mSaved)[name] = mRunState.non_block_activations().freq_cis;
        } else if (mWeights.has(name)) {
            (*mSaved)[name] = mWeights.get(name);
        } else {
            throw std::runtime_error("CompiledExecutor: cannot save tensor " + name);
        }
    }
}

Tensor* CompiledExecutor::try_resolve_saved_live(const std::string& name, const Tensor& saved) {
    std::vector<long> shape;
    shape.reserve(static_cast<std::size_t>(saved.Rank));
    for (int i = 0; i < saved.Rank; ++i) {
        shape.push_back(saved.Sizes[i]);
    }

    auto map_view = [&](Tensor& base) -> Tensor* {
        if (!base.Data) {
            return nullptr;
        }
        if (shape.empty() || tensor_shape_matches(base, shape)) {
            return &base;
        }
        if (shape_nelem(shape) != base.nelem()) {
            return nullptr;
        }
        auto [it, _] = mTensorMap.insert_or_assign(name, view_tensor(base, shape));
        return &it->second;
    };

    if (name == "token_ids") {
        return map_view(mRunState.Inputs);
    }
    if (name == "position_ids") {
        return map_view(mRunState.PositionIDs);
    }
    if (name == "encoded" || name == "x0") {
        return map_view(mRunState.non_block_activations().encoded);
    }
    if (name == "ln_final" || name == "xF" || name == "xF_flat") {
        return map_view(mRunState.non_block_activations().ln_final);
    }
    if (name == "ln_final_rstd") {
        return map_view(mRunState.non_block_activations().ln_final_rstd);
    }
    if (name.find("rope_freqs") != std::string::npos || name.find("freq_cis") != std::string::npos) {
        return map_view(mRunState.non_block_activations().freq_cis);
    }

    int layer_idx = -1;
    std::string field;
    if (parse_block_param(name, layer_idx, field)) {
        if (layer_idx < 0 || layer_idx >= static_cast<int>(mConfig.NumLayers)) {
            return nullptr;
        }
        auto& acts = mRunState.simplified_acts(layer_idx);
        if (field == "ln1" || field == "ln1_flat") return map_view(acts.ln1);
        if (field == "ln1_rstd") return map_view(acts.ln1_rstd);
        if (field == "ln2" || field == "ln2_flat") return map_view(acts.ln2);
        if (field == "ln2_rstd") return map_view(acts.ln2_rstd);
        if (field == "q_rstd") return map_view(acts.q_rstd);
        if (field == "k_rstd") return map_view(acts.k_rstd);
        if (field == "qkv" || field == "qkv_flat" || field == "qkv_biased") return map_view(acts.qkv);
        if (field == "qkv_rope") {
            Tensor* base = acts.qkv_rope.Data ? &acts.qkv_rope : &acts.qkv;
            return map_view(*base);
        }
        if (field == "lse") return map_view(acts.lse);
        if (field == "att" || field == "att_flat") return map_view(acts.att);
        if (field == "att_out" || field == "att_out_flat") return map_view(acts.att_out);
        if (field == "res_att" || field == "residual_att") return map_view(acts.residual_att);
        if (field == "mlp_up" || field == "mlp_up_flat") return map_view(acts.mlp_up);
        if (field == "swiglu" || field == "swiglu_flat") return map_view(acts.swiglu);
        if (field == "mlp_down" || field == "mlp_down_flat") return map_view(acts.mlp_down);
        if (field == "res_ffn" || field == "residual_ffn") {
            Tensor& res = mRunState.get_residual(layer_idx, mRunState.MainStream);
            return map_view(res);
        }
        if (field == "rope_freqs" || field == "freq_cis") {
            return map_view(mRunState.non_block_activations().freq_cis);
        }
    }

    return nullptr;
}

Tensor& CompiledExecutor::resolve_tensor(const TensorRef& ref) {
    auto& rs = mRunState;

    if (!ref.name.empty()) {
        if (auto base = base_param_from_grad(ref.name)) {
            bool accum = false;
            if (Tensor* grad = mGrads.get_param_grad(*base, accum)) {
                if (grad->Data) {
                    if (!ref.shape.empty()) {
                        auto [it, _] = mTensorMap.insert_or_assign(ref.name, view_tensor(*grad, ref.shape));
                        return it->second;
                    }
                    auto [it, _] = mTensorMap.insert_or_assign(ref.name, *grad);
                    return it->second;
                }
            }
        }
    }

    // If shape is specified and this is a pre-allocated slot, we may need to create a view
    if (!ref.shape.empty() && ref.slot != TensorSlot::Mapped && ref.slot != TensorSlot::Saved &&
        ref.slot != TensorSlot::Parameter && ref.slot != TensorSlot::Temporary) {
        // Check if we already have a view cached
        auto it = mTensorMap.find(ref.name);
        if (it != mTensorMap.end()) {
            // Verify shape matches
            bool shape_matches = (it->second.Rank == static_cast<int>(ref.shape.size()));
            if (shape_matches) {
                for (int i = 0; i < it->second.Rank && shape_matches; ++i) {
                    shape_matches = (it->second.Sizes[i] == ref.shape[i]);
                }
            }
            if (shape_matches) {
                return it->second;
            }
            // Shape doesn't match, recreate view
        }
        // Need to create a view - get the base tensor and create view
        Tensor* base = nullptr;
        switch (ref.slot) {
            case TensorSlot::TokenIDs: base = &rs.Inputs; break;
            case TensorSlot::PositionIDs: base = &rs.PositionIDs; break;
            case TensorSlot::Targets: base = &rs.Targets; break;
            case TensorSlot::Losses: base = &rs.Losses; break;
            case TensorSlot::DLoss: base = &rs.scratch().cross_entropy_dloss; break;
            case TensorSlot::BlockDLN1: base = &rs.simplified_grads(ref.layer_idx).d_ln1; break;
            case TensorSlot::BlockDQKV: base = &rs.simplified_grads(ref.layer_idx).d_qkv; break;
            case TensorSlot::BlockDAtt: base = &rs.simplified_grads(ref.layer_idx).d_att; break;
            case TensorSlot::BlockDSwiGLU: base = &rs.simplified_grads(ref.layer_idx).d_swiglu; break;
            case TensorSlot::BlockDMLPUp: base = &rs.simplified_grads(ref.layer_idx).d_mlp_up; break;
            case TensorSlot::BlockDMLPDown: base = &rs.simplified_grads(ref.layer_idx).d_mlp_down; break;
            case TensorSlot::BlockDLN2: base = &rs.simplified_grads(ref.layer_idx).d_ln2; break;
            case TensorSlot::BlockDResAtt: base = &rs.simplified_grads(ref.layer_idx).d_res_att; break;
            case TensorSlot::BlockDResFFN: base = &rs.simplified_grads(ref.layer_idx).d_res_ffn; break;
            case TensorSlot::BlockLN1: base = &rs.simplified_acts(ref.layer_idx).ln1; break;
            case TensorSlot::BlockLN2: base = &rs.simplified_acts(ref.layer_idx).ln2; break;
            case TensorSlot::BlockQKV: base = &rs.simplified_acts(ref.layer_idx).qkv; break;
            case TensorSlot::BlockAtt: base = &rs.simplified_acts(ref.layer_idx).att; break;
            case TensorSlot::BlockAttOut: base = &rs.simplified_acts(ref.layer_idx).att_out; break;
            case TensorSlot::BlockMLPUp: base = &rs.simplified_acts(ref.layer_idx).mlp_up; break;
            case TensorSlot::BlockSwiGLU: base = &rs.simplified_acts(ref.layer_idx).swiglu; break;
            case TensorSlot::BlockMLPDown: base = &rs.simplified_acts(ref.layer_idx).mlp_down; break;
            default: break;
        }
        if (base && base->Data) {
            auto [ins_it, _] = mTensorMap.insert_or_assign(ref.name, view_tensor(*base, ref.shape));
            return ins_it->second;
        }
    }

    switch (ref.slot) {
        case TensorSlot::TokenIDs:
            return rs.Inputs;
        case TensorSlot::PositionIDs:
            return rs.PositionIDs;
        case TensorSlot::Targets:
            return rs.Targets;
        case TensorSlot::Losses:
            return rs.Losses;
        case TensorSlot::DLoss:
            return rs.scratch().cross_entropy_dloss;
        case TensorSlot::Encoded:
            return rs.non_block_activations().encoded;
        case TensorSlot::LNFinal:
            return rs.non_block_activations().ln_final;
        case TensorSlot::LNFinalRSTD:
            return rs.non_block_activations().ln_final_rstd;
        case TensorSlot::FinalResidual:
            return rs.get_final_residual();
        case TensorSlot::FreqCis:
            return rs.non_block_activations().freq_cis;
        case TensorSlot::BlockLN1:
            return rs.simplified_acts(ref.layer_idx).ln1;
        case TensorSlot::BlockLN1RSTD:
            return rs.simplified_acts(ref.layer_idx).ln1_rstd;
        case TensorSlot::BlockLN2:
            return rs.simplified_acts(ref.layer_idx).ln2;
        case TensorSlot::BlockLN2RSTD:
            return rs.simplified_acts(ref.layer_idx).ln2_rstd;
        case TensorSlot::BlockQRSTD:
            return rs.simplified_acts(ref.layer_idx).q_rstd;
        case TensorSlot::BlockKRSTD:
            return rs.simplified_acts(ref.layer_idx).k_rstd;
        case TensorSlot::BlockQKV:
            return rs.simplified_acts(ref.layer_idx).qkv;
        case TensorSlot::BlockQKVRoPE: {
            auto& acts = rs.simplified_acts(ref.layer_idx);
            return acts.qkv_rope.Data ? acts.qkv_rope : acts.qkv;
        }
        case TensorSlot::BlockLSE:
            return rs.simplified_acts(ref.layer_idx).lse;
        case TensorSlot::BlockAtt:
            return rs.simplified_acts(ref.layer_idx).att;
        case TensorSlot::BlockAttOut:
            return rs.simplified_acts(ref.layer_idx).att_out;
        case TensorSlot::BlockResidualAtt:
            return rs.simplified_acts(ref.layer_idx).residual_att;
        case TensorSlot::BlockMLPUp:
            return rs.simplified_acts(ref.layer_idx).mlp_up;
        case TensorSlot::BlockSwiGLU:
            return rs.simplified_acts(ref.layer_idx).swiglu;
        case TensorSlot::BlockMLPDown:
            return rs.simplified_acts(ref.layer_idx).mlp_down;
        case TensorSlot::BlockResidualFFN:
            return rs.get_residual(ref.layer_idx, rs.MainStream);
        case TensorSlot::BlockDLN1:
            return rs.simplified_grads(ref.layer_idx).d_ln1;
        case TensorSlot::BlockDQKV:
            return rs.simplified_grads(ref.layer_idx).d_qkv;
        case TensorSlot::BlockDAtt:
            return rs.simplified_grads(ref.layer_idx).d_att;
        case TensorSlot::BlockDSwiGLU:
            return rs.simplified_grads(ref.layer_idx).d_swiglu;
        case TensorSlot::BlockDMLPUp:
            return rs.simplified_grads(ref.layer_idx).d_mlp_up;
        case TensorSlot::BlockDMLPDown:
            return rs.simplified_grads(ref.layer_idx).d_mlp_down;
        case TensorSlot::BlockDLN2:
            return rs.simplified_grads(ref.layer_idx).d_ln2;
        case TensorSlot::BlockDResAtt:
            return rs.simplified_grads(ref.layer_idx).d_res_att;
        case TensorSlot::BlockDResFFN:
            return rs.simplified_grads(ref.layer_idx).d_res_ffn;
        case TensorSlot::Parameter:
            return mWeights.get(ref.name);
        case TensorSlot::Saved:
            if (mSaved) {
                auto it = mSaved->find(ref.name);
                if (it != mSaved->end()) {
                    if (auto live_it = mTensorMap.find(ref.name); live_it != mTensorMap.end()) {
                        return live_it->second;
                    }
                    if (Tensor* live = try_resolve_saved_live(ref.name, it->second)) {
                        return *live;
                    }
                    return it->second;
                }
            }
            throw std::runtime_error("CompiledExecutor: saved tensor not found: " + ref.name);
        case TensorSlot::Mapped: {
            auto it = mTensorMap.find(ref.name);
            if (it != mTensorMap.end()) {
                return it->second;
            }
            // Debug: print similar tensor names
            if (env_enabled("SUROGATE_DSL_DEBUG_TENSOR_MAP")) {
                std::fprintf(stderr, "[DSL DEBUG] tensor not found: %s\n", ref.name.c_str());
                std::fprintf(stderr, "[DSL DEBUG] available tensors with similar prefix:\n");
                std::string prefix = ref.name.substr(0, std::min(size_t(15), ref.name.size()));
                for (const auto& [name, t] : mTensorMap) {
                    if (name.find(prefix.substr(0, 10)) != std::string::npos) {
                        std::fprintf(stderr, "  %s\n", name.c_str());
                    }
                }
            }
            throw std::runtime_error("CompiledExecutor: tensor not found: " + ref.name);
        }
        case TensorSlot::Temporary:
            throw std::runtime_error("CompiledExecutor: temporary slot requires allocation");
    }
    throw std::runtime_error("CompiledExecutor: invalid tensor slot");
}

Tensor& CompiledExecutor::ensure_output_tensor(const TensorRef& ref) {
    if (!ref.name.empty()) {
        if (auto base = base_param_from_grad(ref.name)) {
            bool accum = false;
            if (Tensor* grad = mGrads.get_param_grad(*base, accum)) {
                if (grad->Data) {
                    if (!ref.shape.empty()) {
                        auto [it, _] = mTensorMap.insert_or_assign(ref.name, view_tensor(*grad, ref.shape));
                        return it->second;
                    }
                    auto [it, _] = mTensorMap.insert_or_assign(ref.name, *grad);
                    return it->second;
                }
            }
        }
    }

    // For pre-allocated slots, just return the tensor
    if (ref.slot != TensorSlot::Mapped && ref.slot != TensorSlot::Temporary) {
        Tensor& t = resolve_tensor(ref);
        if (!t.Data) {
            mRunState.temp_acquire(t);
            mTemps.push_back(t);
        }
        if (!ref.shape.empty()) {
            // Create a view if needed
            auto [it, inserted] = mTensorMap.emplace(ref.name, view_tensor(t, ref.shape));
            if (!inserted) {
                it->second = view_tensor(t, ref.shape);
            }
            return it->second;
        }
        return t;
    }

    // For mapped/temporary tensors, allocate if needed
    auto it = mTensorMap.find(ref.name);
    if (it != mTensorMap.end()) {
        return it->second;
    }

    Tensor t = mRunState.temp_alloc(ref.dtype, ref.shape);
    mTemps.push_back(t);
    auto [ins_it, inserted] = mTensorMap.emplace(ref.name, t);
    return ins_it->second;
}

void CompiledExecutor::handle_layer_start(int layer_idx) {
    if (mWeightManager && mWeightManager->is_streaming_enabled() && !mCapturing) {
        // Wait for current layer's weights
        mWeightManager->wait_for_gather(layer_idx, mRunState.MainStream);
    }

    // Prefetch next layer
    const int next_layer = layer_idx + 1;
    if (next_layer < static_cast<int>(mConfig.NumLayers) && !mCapturing) {
        if (mWeightManager && mWeightManager->is_streaming_enabled()) {
            if (mComm) {
                mWeightManager->gather_block(next_layer, *mComm, mRunState.side_stream());
            }
        }
    }

    mCurrentLayer = layer_idx;
}

void CompiledExecutor::handle_layer_end(int layer_idx) {
    // Release previous layer's weights
    if (mWeightManager && mWeightManager->is_streaming_enabled() && !mCapturing) {
        mWeightManager->release_block(layer_idx, mRunState.MainStream);
    }

    // Offload residual if enabled
    if (mRunState.has_residual_offloading() && !mCapturing) {
        mRunState.mark_residual_ready(layer_idx, mRunState.MainStream);
        mRunState.put_residual(layer_idx, mRunState.side_stream());
    }
}

void CompiledExecutor::dispatch_embedding(const CompiledOp& op) {
    Tensor& token_ids = resolve_tensor(op.inputs[0]);
    Tensor& emb = op.inputs.size() > 1 ? resolve_tensor(op.inputs[1]) : mWeights.get("embedding");
    Tensor& out = ensure_output_tensor(op.outputs[0]);

    encoder_forward(out, token_ids, emb, std::nullopt,
                    static_cast<int>(mB), static_cast<int>(mT),
                    mConfig.HiddenSize, mConfig.VocabSize, mRunState.MainStream);
}

void CompiledExecutor::dispatch_zeros(const CompiledOp& op) {
    Tensor& out = ensure_output_tensor(op.outputs[0]);
    fill_zero(out, mRunState.MainStream);
}

void CompiledExecutor::dispatch_fused_residual_rmsnorm(const CompiledOp& op) {
    Tensor& residual_in = resolve_tensor(op.inputs[0]);
    Tensor& input = resolve_tensor(op.inputs[1]);
    Tensor& weight = resolve_tensor(op.inputs[2]);

    Tensor& residual_out = ensure_output_tensor(op.outputs[0]);
    Tensor& y = ensure_output_tensor(op.outputs[1]);
    Tensor& rstd = ensure_output_tensor(op.outputs[2]);

    // Validate dtypes before calling kernel
    if (rstd.DType != ETensorDType::FP32) {
        std::ostringstream oss;
        oss << "fused_residual_rmsnorm: rstd dtype mismatch. Expected FP32, got "
            << dtype_to_str(rstd.DType) << ". Output tensor: " << op.outputs[2].name
            << " (slot=" << static_cast<int>(op.outputs[2].slot) << ")";
        throw std::runtime_error(oss.str());
    }

    fused_residual_rmsnorm_forward(residual_out, y, rstd, residual_in, input, weight, nullptr,
                                   op.attrs.eps, static_cast<int>(mB * mT),
                                   mConfig.HiddenSize, mRunState.MainStream);

}

void CompiledExecutor::dispatch_view(const CompiledOp& op) {
    Tensor& src = resolve_tensor(op.inputs[0]);
    Tensor view = view_tensor(src, op.attrs.shape);
    mTensorMap[op.outputs[0].name] = view;
}

void CompiledExecutor::dispatch_add(const CompiledOp& op) {
    Tensor& a = resolve_tensor(op.inputs[0]);
    Tensor& b = resolve_tensor(op.inputs[1]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);
    vector_add_sr(out, a, b, 1.0f, static_cast<long>(a.nelem()), 0, mRunState.MainStream);
}

void CompiledExecutor::dispatch_matmul(const CompiledOp& op, const modules::ForwardHook* hook) {
    Tensor& a = resolve_tensor(op.inputs[0]);
    Tensor& b = resolve_tensor(op.inputs[1]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);

    std::optional<Tensor> bias;
    if (op.type == CompiledOpType::MatmulBias && op.inputs.size() > 2) {
        bias = resolve_tensor(op.inputs[2]);
    }

    int M = 0, N = 0, K = 0;
    matmul_dims(a, b, op.attrs.transpose, M, N, K);

    bool used_recipe = false;
    modules::MatmulContext ctx{};
    modules::MatmulContext* ctx_ptr = nullptr;
    if (mRecipe && op.attrs.transpose == EMMTranspose::NT && a.Sizes[0] == mB * mT) {
        if (op.attrs.allow_quant && op.attrs.matmul_op.has_value()) {
            ctx.out = &out;
            ctx.inp = &a;
            ctx.weight = &b;
            ctx.bias = bias ? &*bias : nullptr;
            ctx.B = static_cast<int>(mB);
            ctx.T = static_cast<int>(mT);
            ctx.C_in = K;
            ctx.C_out = N;
            ctx.run_state = &mRunState;
            ctx.stream = mRunState.MainStream;
            ctx.layer_idx = op.attrs.layer_idx;
            ctx.op = *op.attrs.matmul_op;
            ctx.allow_fp8 = true;
            ctx.allow_fp4 = true;

            // FP8/FP4 buffers would be set here via pre-resolved cache
            ctx.inp_quant = fp8_forward_buffer(mRunState, *op.attrs.matmul_op);
            ctx.delayed_quantizer_idx = fp8_quantizer_index(mRunState, *op.attrs.matmul_op, op.attrs.layer_idx);

            mRecipe->forward_matmul(ctx);
            used_recipe = true;
            ctx_ptr = &ctx;
        }
    }

    if (!used_recipe) {
        EMMTranspose mode_col = swap_transpose(op.attrs.transpose);
        matmul(out, b, a, bias, nullptr, nullptr,
               mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
               N, M, K, mode_col, false, mRunState.MainStream);
    }

    if (mForwardPlan && op.attrs.matmul_op.has_value() && op.attrs.layer_idx >= 0 &&
        static_cast<std::size_t>(op.attrs.layer_idx) < mForwardPlan->size() &&
        *op.attrs.matmul_op != modules::MatmulOp::LMHead) {
        MatmulForwardPlan plan{};
        plan.valid = true;
        plan.use_recipe = used_recipe;
        plan.has_bias = bias.has_value();
        if (used_recipe && ctx_ptr) {
            plan.allow_fp8 = ctx_ptr->allow_fp8;
            plan.allow_fp4 = ctx_ptr->allow_fp4;
            plan.delayed_quantizer_idx = ctx_ptr->delayed_quantizer_idx;
            plan.use_fp8_cache = (ctx_ptr->cached_weight && ctx_ptr->cached_weight->Data);
            plan.use_fp4_cache = (ctx_ptr->cached_fp4_data && ctx_ptr->cached_fp4_scales);
        }
        auto& layer_plan = (*mForwardPlan)[static_cast<std::size_t>(op.attrs.layer_idx)];
        switch (*op.attrs.matmul_op) {
            case modules::MatmulOp::QKV:
                layer_plan.qkv = plan;
                break;
            case modules::MatmulOp::AttnOut:
                layer_plan.out_proj = plan;
                break;
            case modules::MatmulOp::MLPUp:
                layer_plan.mlp_up = plan;
                break;
            case modules::MatmulOp::MLPDown:
                layer_plan.mlp_down = plan;
                break;
            default:
                break;
        }
    }

    // Hook invocation
    if (hook && *hook && op.attrs.forward_hook_point.has_value()) {
        (*hook)(op.attrs.layer_idx, mRunState.MainStream, *op.attrs.forward_hook_point, mHookContext);
    }
}

void CompiledExecutor::dispatch_bias_add(const CompiledOp& op) {
    Tensor& x = resolve_tensor(op.inputs[0]);
    Tensor& bias = resolve_tensor(op.inputs[1]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);

    const std::size_t bytes = static_cast<std::size_t>(x.nelem()) * get_dtype_size(x.DType);
    CUDA_CHECK(cudaMemcpyAsync(out.Data, x.Data, bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
    add_bias_tensor(out, bias, static_cast<int>(x.Sizes[0]), static_cast<int>(x.Sizes[1]),
                    static_cast<int>(x.Sizes[2]), mRunState.MainStream);
}

void CompiledExecutor::dispatch_swiglu(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);

    const long D = inp.Sizes[2] / 2;
    swiglu_forward(out, inp, nullptr, static_cast<int>(inp.Sizes[0]),
                   static_cast<int>(inp.Sizes[1]), static_cast<int>(D), mRunState.MainStream);
}

void CompiledExecutor::dispatch_matmul_swiglu(const CompiledOp& op) {
    Tensor& a = resolve_tensor(op.inputs[0]);
    Tensor& b = resolve_tensor(op.inputs[1]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);
    Tensor& up_out = ensure_output_tensor(op.outputs[1]);

    int M = 0, N = 0, K = 0;
    matmul_dims(a, b, op.attrs.transpose, M, N, K);
    const long D = N / 2;

    matmul(up_out, b, a, std::nullopt, nullptr, nullptr,
           mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
           N, M, K, swap_transpose(op.attrs.transpose), false, mRunState.MainStream);

    Tensor up_3d = view_tensor(up_out, {mB, mT, static_cast<long>(N)});
    Tensor out_3d = view_tensor(out, {mB, mT, D});
    swiglu_forward(out_3d, up_3d, nullptr, static_cast<int>(mB),
                   static_cast<int>(mT), static_cast<int>(D), mRunState.MainStream);
}

void CompiledExecutor::dispatch_qkv_qk_norm_rope(const CompiledOp& op) {
    Tensor& qkv = resolve_tensor(op.inputs[0]);
    Tensor& q_norm = resolve_tensor(op.inputs[1]);
    Tensor& k_norm = resolve_tensor(op.inputs[2]);
    Tensor& freqs = resolve_tensor(op.inputs[3]);
    Tensor& pos_ids = resolve_tensor(op.inputs[4]);

    Tensor& q_rstd = ensure_output_tensor(op.outputs[1]);
    Tensor& k_rstd = ensure_output_tensor(op.outputs[2]);

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    const int qkv_channels = Hs * (Hq + 2 * Hkv);

    Tensor qkv_view = (qkv.Rank == 4) ? view_tensor(qkv, {mB, mT, qkv_channels}) : qkv;
    int rotary_dim = op.attrs.rotary_dim;

    const bool rope_fusable = (rotary_dim > 0)
        && ((Hs % 2) == 0)
        && (((Hs / 2) % 32) == 0)
        && (freqs.Rank >= 2)
        && (freqs.Sizes[1] >= Hs)
        && (qkv_view.Rank == 3);

    if (mForwardPlan) {
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(op.inputs[0].name, layer_idx, field) &&
            layer_idx >= 0 && static_cast<std::size_t>(layer_idx) < mForwardPlan->size()) {
            AttnForwardPlan plan{};
            plan.valid = true;
            plan.use_qk_norm = true;
            plan.rope_fused = rope_fusable;
            plan.use_cudnn = true;
            plan.rotary_dim = rotary_dim;
            (*mForwardPlan)[static_cast<std::size_t>(layer_idx)].attn = plan;
        }
    }

    if (rope_fusable) {
        qkv_qk_norm_rope_forward(qkv_view, q_rstd, k_rstd, q_norm, k_norm,
                                 freqs, reinterpret_cast<int*>(pos_ids.Data),
                                 op.attrs.eps,
                                 static_cast<int>(mB), static_cast<int>(mT),
                                 Hq, Hkv, Hs, mRunState.MainStream);
    } else {
        const int q_rows = Hq * Hs;
        qkv_head_rmsnorm_forward(qkv_view, q_rstd, q_norm,
                                 op.attrs.eps,
                                 static_cast<int>(mB), static_cast<int>(mT),
                                 qkv_channels, Hq, Hs, 0, mRunState.MainStream);
        qkv_head_rmsnorm_forward(qkv_view, k_rstd, k_norm,
                                 op.attrs.eps,
                                 static_cast<int>(mB), static_cast<int>(mT),
                                 qkv_channels, Hkv, Hs, q_rows, mRunState.MainStream);
        rope_forward(qkv, qkv, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
                     static_cast<int>(mB), static_cast<int>(mT), Hq, Hkv, Hs, rotary_dim, mRunState.MainStream);
    }

    mTensorMap[op.outputs[0].name] = qkv;
}

void CompiledExecutor::dispatch_rope(const CompiledOp& op) {
    Tensor& qkv = resolve_tensor(op.inputs[0]);
    Tensor& freqs = resolve_tensor(op.inputs[1]);
    Tensor& pos_ids = resolve_tensor(op.inputs[2]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());

    if (mForwardPlan) {
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(op.inputs[0].name, layer_idx, field) &&
            layer_idx >= 0 && static_cast<std::size_t>(layer_idx) < mForwardPlan->size()) {
            AttnForwardPlan plan{};
            plan.valid = true;
            plan.use_qk_norm = false;
            plan.rope_fused = false;
            plan.use_cudnn = true;
            plan.rotary_dim = op.attrs.rotary_dim;
            (*mForwardPlan)[static_cast<std::size_t>(layer_idx)].attn = plan;
        }
    }

    rope_forward(out, qkv, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
                 static_cast<int>(mB), static_cast<int>(mT), Hq, Hkv, Hs,
                 op.attrs.rotary_dim, mRunState.MainStream);
}

void CompiledExecutor::dispatch_flash_attention(const CompiledOp& op) {
    Tensor& qkv = resolve_tensor(op.inputs[0]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);
    Tensor& lse = ensure_output_tensor(op.outputs[1]);

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());

    if (!mRunState.scratch().cudnn_workspace.Data) {
        mRunState.temp_acquire(mRunState.scratch().cudnn_workspace);
        mTemps.push_back(mRunState.scratch().cudnn_workspace);
    }

    attention_forward_cudnn(out, lse, qkv, mRunState.scratch().cudnn_workspace,
                            mRunState.CudnnHandle, static_cast<int>(mB), static_cast<int>(mT),
                            Hq, Hkv, Hs, mRunState.MainStream);
}

void CompiledExecutor::dispatch_cross_entropy_loss(const CompiledOp& op) {
    Tensor& logits = resolve_tensor(op.inputs[0]);
    Tensor& targets = resolve_tensor(op.inputs[1]);
    Tensor& loss = ensure_output_tensor(op.outputs[0]);

    const int BT = static_cast<int>(logits.Sizes[0]);
    const int V = static_cast<int>(logits.Sizes[1]);
    const int P = V;

    Tensor logsumexp_view{};
    Tensor* logsumexp = nullptr;
    if (mRunState.scratch().cross_entropy_logsumexp.Data) {
        logsumexp_view = mRunState.scratch().cross_entropy_logsumexp;
        logsumexp_view.Sizes[0] = BT;
        logsumexp_view.Rank = 1;
        logsumexp = &logsumexp_view;
    }

    if (V > CROSS_ENTROPY_MAX_FUSED_SIZE) {
        if (!mRunState.scratch().cross_entropy_chunk_logsumexp.Data) {
            throw std::runtime_error("cross_entropy_loss: chunk logsumexp buffer is not allocated");
        }
        const int n_chunks = (V + CROSS_ENTROPY_MAX_FUSED_SIZE - 1) / CROSS_ENTROPY_MAX_FUSED_SIZE;
        Tensor chunk_lse = mRunState.scratch().cross_entropy_chunk_logsumexp;
        chunk_lse.Sizes[0] = BT;
        chunk_lse.Sizes[1] = n_chunks;
        chunk_lse.Rank = 2;

        chunked_cross_entropy_forward(logits, loss, logsumexp, chunk_lse, targets,
                                      &mRunState.ValidTokenCount,
                                      op.attrs.compute_accuracy ? &mRunState.CorrectCount : nullptr,
                                      BT, V, P, n_chunks, mRunState.MainStream);
    } else {
        fused_cross_entropy_forward(logits, loss, logsumexp, targets,
                                    &mRunState.ValidTokenCount,
                                    op.attrs.compute_accuracy ? &mRunState.CorrectCount : nullptr,
                                    BT, V, P, mRunState.MainStream);
    }
}

void CompiledExecutor::dispatch_fused_lm_head_loss(const CompiledOp& op) {
    Tensor& xF_flat = resolve_tensor(op.inputs[0]);
    Tensor& weight = resolve_tensor(op.inputs[1]);
    Tensor& targets = resolve_tensor(op.inputs[2]);
    Tensor& loss = ensure_output_tensor(op.outputs[0]);

    const long BT = xF_flat.Sizes[0];
    const int V = static_cast<int>(weight.Sizes[0]);
    const int C = static_cast<int>(weight.Sizes[1]);
    const int P = V;

    const int lmhead_chunks = std::max(1, mOptions.LMHeadChunks);
    const long nano_batch_size = BT / static_cast<long>(lmhead_chunks);
    const int n_chunks = (V + CROSS_ENTROPY_MAX_FUSED_SIZE - 1) / CROSS_ENTROPY_MAX_FUSED_SIZE;

    const bool need_lm_head =
        mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled());
    if (need_lm_head && mComm) {
        mWeightManager->gather_lm_head(*mComm, mRunState.MainStream);
    }

    const std::size_t xf_stride = get_dtype_size(xF_flat.DType);
    const std::size_t tgt_stride = get_dtype_size(targets.DType);
    const std::size_t loss_stride = get_dtype_size(loss.DType);
    const std::size_t lse_stride = get_dtype_size(ETensorDType::FP32);
    const std::size_t chunk_lse_stride = lse_stride * static_cast<std::size_t>(n_chunks);

    for (int nano_step = 0; nano_step < lmhead_chunks; ++nano_step) {
        const long token_offset = static_cast<long>(nano_step) * nano_batch_size;

        Tensor xF_slice = xF_flat;
        xF_slice.Data = static_cast<std::byte*>(xF_slice.Data) +
                        static_cast<std::size_t>(token_offset) * xf_stride * static_cast<std::size_t>(C);
        xF_slice.Sizes[0] = nano_batch_size;
        xF_slice.Sizes[1] = C;
        xF_slice.Rank = 2;

        Tensor tgt_slice = targets;
        tgt_slice.Data = static_cast<std::byte*>(tgt_slice.Data) +
                         static_cast<std::size_t>(token_offset) * tgt_stride;
        tgt_slice.Sizes[0] = nano_batch_size;
        tgt_slice.Rank = 1;

        Tensor loss_slice = loss;
        loss_slice.Data = static_cast<std::byte*>(loss_slice.Data) +
                          static_cast<std::size_t>(token_offset) * loss_stride;
        loss_slice.Sizes[0] = nano_batch_size;
        loss_slice.Rank = 1;

        Tensor logits = mRunState.non_block_activations().output;
        logits.Sizes[0] = nano_batch_size;
        logits.Sizes[1] = V;
        logits.Rank = 2;

        matmul(logits, weight, xF_slice,
               std::nullopt, nullptr, nullptr, mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
               static_cast<int>(V), static_cast<int>(nano_batch_size), static_cast<int>(C),
               swap_transpose(EMMTranspose::NT), false, mRunState.MainStream);

        Tensor logsumexp_view{};
        Tensor* logsumexp = nullptr;
        if (mRunState.scratch().cross_entropy_logsumexp.Data) {
            logsumexp_view = mRunState.scratch().cross_entropy_logsumexp;
            logsumexp_view.Data = static_cast<std::byte*>(logsumexp_view.Data) +
                                  static_cast<std::size_t>(token_offset) * lse_stride;
            logsumexp_view.Sizes[0] = nano_batch_size;
            logsumexp_view.Rank = 1;
            logsumexp = &logsumexp_view;
        }

        if (V > CROSS_ENTROPY_MAX_FUSED_SIZE) {
            if (!mRunState.scratch().cross_entropy_chunk_logsumexp.Data) {
                throw std::runtime_error("fused_lm_head_loss: chunk logsumexp buffer is not allocated");
            }
            Tensor chunk_lse = mRunState.scratch().cross_entropy_chunk_logsumexp;
            chunk_lse.Data = static_cast<std::byte*>(chunk_lse.Data) +
                             static_cast<std::size_t>(token_offset) * chunk_lse_stride;
            chunk_lse.Sizes[0] = nano_batch_size;
            chunk_lse.Sizes[1] = n_chunks;
            chunk_lse.Rank = 2;

            chunked_cross_entropy_forward(logits, loss_slice, logsumexp, chunk_lse, tgt_slice,
                                          &mRunState.ValidTokenCount,
                                          op.attrs.compute_accuracy ? &mRunState.CorrectCount : nullptr,
                                          static_cast<int>(nano_batch_size), V, P, n_chunks, mRunState.MainStream);
        } else {
            fused_cross_entropy_forward(logits, loss_slice, logsumexp, tgt_slice,
                                        &mRunState.ValidTokenCount,
                                        op.attrs.compute_accuracy ? &mRunState.CorrectCount : nullptr,
                                        static_cast<int>(nano_batch_size), V, P, mRunState.MainStream);
        }
    }

    if (need_lm_head) {
        mWeightManager->release_lm_head(mRunState.MainStream);
    }
}

// Backward dispatch implementations
void CompiledExecutor::dispatch_view_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    std::vector<long> shape = op.attrs.shape;

    // If shape is empty, try to resolve from shape_like reference
    if (shape.empty() && !op.attrs.shape_like.empty()) {
        std::string ref_name = op.attrs.shape_like;

        // Strip "saved." prefix if present
        const std::string saved_prefix = "saved.";
        if (ref_name.rfind(saved_prefix, 0) == 0) {
            ref_name = ref_name.substr(saved_prefix.length());
        }

        // Try to find the reference tensor
        Tensor* ref = nullptr;

        // Check saved tensors first
        if (mSaved) {
            auto it = mSaved->find(ref_name);
            if (it != mSaved->end()) {
                ref = &it->second;
            }
        }

        // Check tensor map
        if (!ref) {
            auto it = mTensorMap.find(ref_name);
            if (it != mTensorMap.end()) {
                ref = &it->second;
            }
        }

        // If reference found and valid, use its shape
        if (ref && ref->Rank > 0) {
            shape.assign(ref->Sizes.begin(), ref->Sizes.begin() + ref->Rank);
        } else {
            // Fallback: infer shape based on output tensor name and input shape
            // View backward typically does one of:
            // 1. Flatten: [B,T,C] -> [B*T,C] (output name contains "_flat")
            // 2. Unflatten: [B*T,C] -> [B,T,C] (output name does not contain "_flat")
            //
            // Check output name for "_flat" suffix to determine direction
            const std::string& out_name = op.outputs[0].name;
            bool wants_flat = out_name.find("_flat") != std::string::npos;

            if (wants_flat) {
                // Flatten to rank 2: [B,T,C] -> [B*T,C] or [B*T,C] -> [B*T,C]
                if (d_out.Rank >= 3) {
                    long flat_dim = 1;
                    for (int i = 0; i < d_out.Rank - 1; ++i) {
                        flat_dim *= d_out.Sizes[i];
                    }
                    shape = {flat_dim, d_out.Sizes[d_out.Rank - 1]};
                } else if (d_out.Rank == 2) {
                    // Already flat, keep shape
                    shape = {d_out.Sizes[0], d_out.Sizes[1]};
                }
            } else {
                // Unflatten or keep shape
                if (d_out.Rank >= 3) {
                    // Already unflat, keep shape
                    shape.assign(d_out.Sizes.begin(), d_out.Sizes.begin() + d_out.Rank);
                } else if (d_out.Rank == 2 && d_out.Sizes[0] == mB * mT) {
                    // Unflatten: [B*T,C] -> [B,T,C]
                    shape = {mB, mT, d_out.Sizes[1]};
                } else if (d_out.Rank == 2) {
                    // Keep as rank 2
                    shape = {d_out.Sizes[0], d_out.Sizes[1]};
                }
            }
        }
    }

    if (shape.empty()) {
        auto shape_str = [](const Tensor& t) {
            std::string s = "[";
            for (int i = 0; i < t.Rank; ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(t.Sizes[i]);
            }
            s += "]";
            return s;
        };
        throw std::runtime_error("CompiledExecutor view_backward: cannot resolve shape for op " + op.op_id +
                                " input=" + op.inputs[0].name + " shape=" + shape_str(d_out) +
                                " output=" + op.outputs[0].name +
                                " shape_like=" + op.attrs.shape_like);
    }
    Tensor view = view_tensor(d_out, shape);
    mTensorMap[op.outputs[0].name] = view;
}

void CompiledExecutor::dispatch_add_backward(const CompiledOp& op) {
    // Addition backward: gradients pass through unchanged to both inputs
    Tensor& d_out = resolve_tensor(op.inputs[0]);

    if (env_enabled("SUROGATE_DSL_DEBUG_ADD_BWD") || env_enabled("SUROGATE_DSL_DEBUG_RECOMPUTE_GRAD")) {
        static int debug_add_count = 0;
        static Tensor debug_add_norm_buf{};
        if (!debug_add_norm_buf.Data && mRunState.Allocator && env_enabled("SUROGATE_DSL_DEBUG_RECOMPUTE_GRAD")) {
            const int num_sums = get_max_num_block_sums(mRunState.DeviceProp);
            debug_add_norm_buf = mRunState.Allocator->allocate(ETensorDType::FP32,
                                                                "debug_add_bwd_norm",
                                                                EAllocationType::ON_DEVICE,
                                                                {num_sums});
        }
        auto norm_of = [&](const Tensor& t) -> float {
            if (!debug_add_norm_buf.Data) return 0.0f;
            fill_zero(debug_add_norm_buf, mRunState.MainStream);
            global_norm_squared(debug_add_norm_buf, t, t.nelem(), mRunState.DeviceProp, mRunState.MainStream);
            deterministic_sum(debug_add_norm_buf.template get<float>(),
                              debug_add_norm_buf.template get<float>(),
                              debug_add_norm_buf.nelem(),
                              mRunState.MainStream);
            float host_sum = 0.0f;
            CUDA_CHECK(cudaMemcpyAsync(&host_sum,
                                       debug_add_norm_buf.template get<float>(),
                                       sizeof(float),
                                       cudaMemcpyDeviceToHost,
                                       mRunState.MainStream));
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            return std::sqrt(host_sum);
        };
        if (debug_add_count < 20) {
            std::fprintf(stderr,
                         "[DSL DEBUG ADD BWD] op_id=%s in=%s (norm=%g) out0=%s (slot=%d layer=%d) out1=%s (slot=%d layer=%d)\n",
                         op.op_id.c_str(),
                         op.inputs.size() > 0 ? op.inputs[0].name.c_str() : "",
                         debug_add_norm_buf.Data ? norm_of(d_out) : 0.0f,
                         op.outputs.size() > 0 ? op.outputs[0].name.c_str() : "",
                         op.outputs.size() > 0 ? static_cast<int>(op.outputs[0].slot) : -1,
                         op.outputs.size() > 0 ? op.outputs[0].layer_idx : -1,
                         op.outputs.size() > 1 ? op.outputs[1].name.c_str() : "",
                         op.outputs.size() > 1 ? static_cast<int>(op.outputs[1].slot) : -1,
                         op.outputs.size() > 1 ? op.outputs[1].layer_idx : -1);
        }
        ++debug_add_count;
    }

    // For pre-allocated gradient slots (like d_res_ffn, d_res_att), we must copy the
    // upstream gradient into the original simplified_grads buffer. Simply aliasing
    // the data pointer causes shared storage between residual and branch gradients,
    // which breaks LoRA (it does in-place dx accumulation).
    // IMPORTANT: We must get the base tensor directly from simplified_grads(), not via
    // resolve_tensor(), because resolve_tensor() may return a view from mTensorMap.
    auto assign_output = [&](const TensorRef& ref) {
        Tensor* base_grad = nullptr;
        if (ref.layer_idx >= 0) {
            auto& grads = mRunState.simplified_grads(ref.layer_idx);
            switch (ref.slot) {
                case TensorSlot::BlockDResFFN: base_grad = &grads.d_res_ffn; break;
                case TensorSlot::BlockDResAtt: base_grad = &grads.d_res_att; break;
                case TensorSlot::BlockDLN1: base_grad = &grads.d_ln1; break;
                case TensorSlot::BlockDLN2: base_grad = &grads.d_ln2; break;
                case TensorSlot::BlockDSwiGLU: base_grad = &grads.d_swiglu; break;
                case TensorSlot::BlockDAtt: base_grad = &grads.d_att; break;
                case TensorSlot::BlockDQKV: base_grad = &grads.d_qkv; break;
                case TensorSlot::BlockDMLPUp: base_grad = &grads.d_mlp_up; break;
                case TensorSlot::BlockDMLPDown: base_grad = &grads.d_mlp_down; break;
                default: break;
            }
        }

        if (base_grad) {
            if (base_grad->Data) {
                if (base_grad->DType != d_out.DType) {
                    throw std::runtime_error("dispatch_add_backward: dtype mismatch for " + ref.name);
                }
                if (base_grad->Data != d_out.Data) {
                    CUDA_CHECK(cudaMemcpyAsync(base_grad->Data, d_out.Data, d_out.bytes(),
                                               cudaMemcpyDeviceToDevice, mRunState.MainStream));
                }
                mTensorMap[ref.name] = view_tensor(*base_grad, ref.shape);
                return;
            }
            // For stack-allocated gradient temps, allocate proper storage instead of aliasing.
            // Aliasing to d_out can cause stale memory access when the stack is restored at
            // layer boundaries because the aliased memory gets recycled.
            const bool is_stack_grad = mRunState.large_bwd_temps_on_stack() &&
                (ref.slot == TensorSlot::BlockDQKV ||
                 ref.slot == TensorSlot::BlockDMLPUp ||
                 ref.slot == TensorSlot::BlockDSwiGLU);
            if (is_stack_grad) {
                // Allocate proper stack storage and copy data
                mRunState.temp_acquire(*base_grad);
                mTemps.push_back(*base_grad);
                CUDA_CHECK(cudaMemcpyAsync(base_grad->Data, d_out.Data, d_out.bytes(),
                                           cudaMemcpyDeviceToDevice, mRunState.MainStream));
                mTensorMap[ref.name] = view_tensor(*base_grad, ref.shape);
                return;
            }
            // Fall back to aliasing if the base grad has no storage yet (non-stack temps).
            base_grad->Data = d_out.Data;
            mTensorMap[ref.name] = view_tensor(*base_grad, ref.shape);
            return;
        }
        // Default: just expose d_out as-is.
        mTensorMap[ref.name] = d_out;
    };

    assign_output(op.outputs[0]);
    if (op.outputs.size() > 1) {
        assign_output(op.outputs[1]);
    }
}

void CompiledExecutor::dispatch_matmul_backward(const CompiledOp& op, const modules::BackwardHook* hook) {
    // inputs: d_out, A, B (weight)
    // outputs: dA, dB
    const std::string& weight_name = (op.inputs.size() > 2) ? op.inputs[2].name : "";
    const bool is_lm_head = (weight_name == "lm_head" || weight_name == "lm_head_weight");
    const bool skip_lm_head = is_lm_head && mOptions.LMHeadChunks > 1;

    EMMTranspose mode = op.attrs.transpose;
    const int layer_idx = op.attrs.layer_idx;
    const bool allow_quant = op.attrs.allow_quant;

    // Check if weight gradient should be skipped BEFORE allocating (frozen weights in LoRA mode)
    bool skip_weight_grad = true;
    const std::string& dB_name = op.outputs.size() > 1 ? op.outputs[1].name : "";
    if (!dB_name.empty()) {
        std::string weight_name;
        if (auto base = base_param_from_grad(dB_name)) {
            weight_name = *base;
        } else {
            weight_name = dB_name;
            if (weight_name.rfind("d_", 0) == 0) {
                weight_name = weight_name.substr(2);
            }
        }
        bool accum = false;
        Tensor* grad = mGrads.get_param_grad(weight_name, accum);
        skip_weight_grad = (grad == nullptr || !grad->Data);
    }

    if (skip_lm_head) {
        if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
            (void)ensure_output_tensor(op.outputs[0]);
        }
        if (!skip_weight_grad && op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
            (void)ensure_output_tensor(op.outputs[1]);
        }
        return;
    }

    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& a = resolve_tensor(op.inputs[1]);
    Tensor& b = resolve_tensor(op.inputs[2]);

    static bool debug_recompute = env_enabled("SUROGATE_DEBUG_RECOMPUTE");
    static bool debug_grad_norms = env_enabled("SUROGATE_DEBUG_GRAD_NORMS");
    if (debug_recompute && layer_idx >= 0) {
        std::fprintf(stderr, "[RECOMPUTE DEBUG] matmul_backward layer=%d weight=%s a_ptr=%p last_recompute=%d\n",
                     layer_idx, weight_name.c_str(), a.Data, mLastRecomputeLayer);
    }

    // Debug: compute input norms for gradient tracking
    static Tensor mm_bwd_norm_buf{};
    auto compute_norm_mm = [&](const Tensor& t) -> float {
        if (!debug_grad_norms || !t.Data) return 0.0f;
        if (!mm_bwd_norm_buf.Data && mRunState.Allocator) {
            const int num_sums = get_max_num_block_sums(mRunState.DeviceProp);
            mm_bwd_norm_buf = mRunState.Allocator->allocate(ETensorDType::FP32, "debug_mm_bwd_norm",
                                                            EAllocationType::ON_DEVICE, {num_sums});
        }
        if (!mm_bwd_norm_buf.Data) return 0.0f;
        fill_zero(mm_bwd_norm_buf, mRunState.MainStream);
        global_norm_squared(mm_bwd_norm_buf, t, t.nelem(), mRunState.DeviceProp, mRunState.MainStream);
        deterministic_sum(mm_bwd_norm_buf.template get<float>(),
                          mm_bwd_norm_buf.template get<float>(),
                          mm_bwd_norm_buf.nelem(), mRunState.MainStream);
        float host_sum = 0.0f;
        CUDA_CHECK(cudaMemcpyAsync(&host_sum, mm_bwd_norm_buf.template get<float>(),
                                   sizeof(float), cudaMemcpyDeviceToHost, mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        return std::sqrt(host_sum);
    };

    if (debug_grad_norms && layer_idx >= 0) {
        std::fprintf(stderr, "[GRAD NORM] matmul_bwd layer=%d weight=%s: d_out=%.6g a=%.6g\n",
                     layer_idx, weight_name.c_str(), compute_norm_mm(d_out), compute_norm_mm(a));
    }

    if (env_enabled("SUROGATE_DSL_DEBUG_MATMUL_BWD")) {
        static int debug_matmul_count = 0;
        static Tensor debug_norm_buf{};
        if (!debug_norm_buf.Data && mRunState.Allocator) {
            const int num_sums = get_max_num_block_sums(mRunState.DeviceProp);
            debug_norm_buf = mRunState.Allocator->allocate(ETensorDType::FP32,
                                                           "debug_matmul_bwd_norm",
                                                           EAllocationType::ON_DEVICE,
                                                           {num_sums});
        }
        const char* layer_env = std::getenv("SUROGATE_DSL_DEBUG_MATMUL_LAYER");
        const int layer_filter = layer_env ? std::atoi(layer_env) : -999999;
        const bool layer_match = !layer_env || layer_idx == layer_filter;
        if (layer_match && debug_matmul_count < 20) {
            std::string base_name;
            if (!dB_name.empty()) {
                if (auto base = base_param_from_grad(dB_name)) {
                    base_name = *base;
                }
            }
            bool grad_accum = false;
            Tensor* grad = (!base_name.empty()) ? mGrads.get_param_grad(base_name, grad_accum) : nullptr;
            std::fprintf(stderr,
                         "[DSL DEBUG MATMUL] op_id=%s layer=%d matmul_op=%d d_out=%s a=%s weight=%s dB=%s base=%s grad=%p skip=%d\n",
                         op.op_id.c_str(),
                         layer_idx,
                         static_cast<int>(op.attrs.matmul_op.value_or(modules::MatmulOp::LMHead)),
                         op.inputs.size() > 0 ? op.inputs[0].name.c_str() : "",
                         op.inputs.size() > 1 ? op.inputs[1].name.c_str() : "",
                         weight_name.c_str(),
                         dB_name.c_str(),
                         base_name.c_str(),
                         grad ? grad->Data : nullptr,
                         skip_weight_grad ? 1 : 0);
            if (debug_norm_buf.Data && debug_matmul_count < 4) {
                auto print_norm = [&](const char* name, const Tensor& t) {
                    fill_zero(debug_norm_buf, mRunState.MainStream);
                    global_norm_squared(debug_norm_buf, t, t.nelem(), mRunState.DeviceProp, mRunState.MainStream);
                    deterministic_sum(debug_norm_buf.template get<float>(),
                                      debug_norm_buf.template get<float>(),
                                      debug_norm_buf.nelem(),
                                      mRunState.MainStream);
                    float host_sum = 0.0f;
                    CUDA_CHECK(cudaMemcpyAsync(&host_sum,
                                               debug_norm_buf.template get<float>(),
                                               sizeof(float),
                                               cudaMemcpyDeviceToHost,
                                               mRunState.MainStream));
                    CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
                    const float norm = std::sqrt(host_sum);
                    std::fprintf(stderr, "  [DSL DEBUG MATMUL] %s norm=%g\n", name, norm);
                };
                print_norm("d_out", d_out);
                print_norm("a_inp", a);
            }
            ++debug_matmul_count;
        }
    }

    // Now allocate output tensors - skip dB if weights are frozen
    Tensor* dA_ptr = nullptr;
    Tensor* dB_ptr = nullptr;

    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        dA_ptr = &ensure_output_tensor(op.outputs[0]);
    }
    if (!skip_weight_grad && op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        dB_ptr = &ensure_output_tensor(op.outputs[1]);
    }

    if (!dA_ptr && !dB_ptr) {
        return;
    }

    bool do_accumulate = mAccumulateTensors.count(dB_name) > 0;
    if (!do_accumulate && !dB_name.empty()) {
        if (auto base = base_param_from_grad(dB_name)) {
            do_accumulate = mAccumulateTensors.count("d_" + *base) > 0;
        }
    }

    bool used_recipe = false;
    bool used_fp8 = false;
    bool has_dout_quant = false;

    if (mRecipe && mode == EMMTranspose::NT && a.Sizes[0] == mB * mT && allow_quant) {
        Tensor dA_tmp{};
        Tensor dB_tmp{};
        Tensor* dA_use = dA_ptr;
        Tensor* dB_use = dB_ptr;

        if (!dA_use) {
            dA_tmp = mRunState.temp_alloc(a.DType, {a.Sizes[0], a.Sizes[1]});
            mTemps.push_back(dA_tmp);
            dA_use = &dA_tmp;
        }
        if (!dB_use) {
            dB_tmp = mRunState.temp_alloc(b.DType, {b.Sizes[0], b.Sizes[1]});
            mTemps.push_back(dB_tmp);
            dB_use = &dB_tmp;
        }

        modules::MatmulContext ctx;
        ctx.dinp = dA_use;
        ctx.dweight = dB_use;
        ctx.dout = &d_out;
        ctx.inp = &a;
        ctx.weight = &b;
        ctx.B = static_cast<int>(mB);
        ctx.T = static_cast<int>(mT);
        ctx.C_in = static_cast<int>(a.Sizes[1]);
        ctx.C_out = static_cast<int>(b.Sizes[0]);
        ctx.run_state = &mRunState;
        ctx.stream = mRunState.MainStream;
        ctx.layer_idx = layer_idx;
        ctx.op = op.attrs.matmul_op.value_or(modules::MatmulOp::LMHead);
        ctx.accumulate = do_accumulate;
        ctx.skip_weight_grad = skip_weight_grad || !dB_ptr;
        ctx.allow_fp8 = allow_quant;
        ctx.allow_fp4 = allow_quant;

        if (allow_quant && op.attrs.matmul_op.has_value()) {
            ctx.dout_quant = fp8_grad_buffer(mRunState, *op.attrs.matmul_op);
            if (!ctx.dout_quant || !ctx.dout_quant->Data) {
                ctx.allow_fp8 = false;
            }
        }
        used_fp8 = ctx.allow_fp8;
        has_dout_quant = (ctx.dout_quant && ctx.dout_quant->Data);

        mRecipe->backward_matmul(ctx);
        used_recipe = true;

        if (env_enabled("SUROGATE_DSL_DEBUG_MATMUL_BWD")) {
            static int debug_recipe_count = 0;
            static Tensor debug_norm_buf_recipe{};
            if (!debug_norm_buf_recipe.Data && mRunState.Allocator) {
                const int num_sums = get_max_num_block_sums(mRunState.DeviceProp);
                debug_norm_buf_recipe = mRunState.Allocator->allocate(ETensorDType::FP32,
                                                                      "debug_matmul_bwd_recipe_norm",
                                                                      EAllocationType::ON_DEVICE,
                                                                      {num_sums});
            }
            if (debug_norm_buf_recipe.Data && debug_recipe_count < 4) {
                auto norm_of = [&](const Tensor& t) {
                    fill_zero(debug_norm_buf_recipe, mRunState.MainStream);
                    global_norm_squared(debug_norm_buf_recipe, t, t.nelem(), mRunState.DeviceProp, mRunState.MainStream);
                    deterministic_sum(debug_norm_buf_recipe.template get<float>(),
                                      debug_norm_buf_recipe.template get<float>(),
                                      debug_norm_buf_recipe.nelem(),
                                      mRunState.MainStream);
                    float host_sum = 0.0f;
                    CUDA_CHECK(cudaMemcpyAsync(&host_sum,
                                               debug_norm_buf_recipe.template get<float>(),
                                               sizeof(float),
                                               cudaMemcpyDeviceToHost,
                                               mRunState.MainStream));
                    CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
                    return std::sqrt(host_sum);
                };
                if (dA_use) {
                    std::fprintf(stderr, "  [DSL DEBUG MATMUL] recipe dA norm=%g ptr=%p\n",
                                 norm_of(*dA_use), dA_use->Data);
                }
                if (dB_use) {
                    std::fprintf(stderr,
                                 "  [DSL DEBUG MATMUL] recipe dB norm=%g ptr=%p accum=%d skip=%d\n",
                                 norm_of(*dB_use),
                                 dB_use->Data,
                                 do_accumulate ? 1 : 0,
                                 ctx.skip_weight_grad ? 1 : 0);
                }
                if (dB_ptr && dB_ptr != dB_use) {
                    std::fprintf(stderr,
                                 "  [DSL DEBUG MATMUL] recipe dB_ptr ptr=%p (view=%s)\n",
                                 dB_ptr->Data,
                                 dB_ptr->Data == dB_use->Data ? "yes" : "no");
                }
            }
            ++debug_recipe_count;
        }
    }

    if (!used_recipe) {
        // Fallback: explicit matmuls for dA and dB
        EMMTranspose mode_dA = EMMTranspose::NN;
        EMMTranspose mode_dB = EMMTranspose::NN;
        switch (mode) {
            case EMMTranspose::NN:
                mode_dA = EMMTranspose::NT;
                mode_dB = EMMTranspose::TN;
                break;
            case EMMTranspose::NT:
                mode_dA = EMMTranspose::NN;
                mode_dB = EMMTranspose::TN;
                break;
            case EMMTranspose::TN:
                mode_dA = EMMTranspose::NT;
                mode_dB = EMMTranspose::NN;
                break;
            case EMMTranspose::TT:
                mode_dA = EMMTranspose::TT;
                mode_dB = EMMTranspose::TT;
                break;
        }

        if (dA_ptr) {
            int M = 0, N = 0, K = 0;
            matmul_dims(d_out, b, mode_dA, M, N, K);
            EMMTranspose mode_col = swap_transpose(mode_dA);
            matmul(*dA_ptr, b, d_out, std::nullopt, nullptr, nullptr,
                   mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
                   N, M, K, mode_col, false, mRunState.MainStream);
        }
        if (dB_ptr && !skip_weight_grad) {
            const Tensor* lhs = nullptr;
            const Tensor* rhs = nullptr;
            EMMTranspose mode_rm = EMMTranspose::NN;
            switch (mode) {
                case EMMTranspose::NN:
                    // dB = A^T * d_out
                    lhs = &a;
                    rhs = &d_out;
                    mode_rm = EMMTranspose::TN;
                    break;
                case EMMTranspose::NT:
                    // dB = d_out^T * A
                    lhs = &d_out;
                    rhs = &a;
                    mode_rm = EMMTranspose::TN;
                    break;
                case EMMTranspose::TN:
                    // dB = A * d_out
                    lhs = &a;
                    rhs = &d_out;
                    mode_rm = EMMTranspose::NN;
                    break;
                case EMMTranspose::TT:
                    // dB = d_out^T * A^T
                    lhs = &d_out;
                    rhs = &a;
                    mode_rm = EMMTranspose::TT;
                    break;
            }

            int M = 0, N = 0, K = 0;
            matmul_dims(*lhs, *rhs, mode_rm, M, N, K);
            EMMTranspose mode_col = swap_transpose(mode_rm);
            if (env_enabled("SUROGATE_DSL_DEBUG_MATMUL_BWD")) {
                static int debug_db_count = 0;
                static Tensor debug_norm_buf_db{};
                if (!debug_norm_buf_db.Data && mRunState.Allocator) {
                    const int num_sums = get_max_num_block_sums(mRunState.DeviceProp);
                    debug_norm_buf_db = mRunState.Allocator->allocate(ETensorDType::FP32,
                                                                      "debug_matmul_bwd_db_norm",
                                                                      EAllocationType::ON_DEVICE,
                                                                      {num_sums});
                }
                if (debug_norm_buf_db.Data && debug_db_count < 4) {
                    auto norm_of = [&](const Tensor& t) {
                        fill_zero(debug_norm_buf_db, mRunState.MainStream);
                        global_norm_squared(debug_norm_buf_db, t, t.nelem(), mRunState.DeviceProp, mRunState.MainStream);
                        deterministic_sum(debug_norm_buf_db.template get<float>(),
                                          debug_norm_buf_db.template get<float>(),
                                          debug_norm_buf_db.nelem(),
                                          mRunState.MainStream);
                        float host_sum = 0.0f;
                        CUDA_CHECK(cudaMemcpyAsync(&host_sum,
                                                   debug_norm_buf_db.template get<float>(),
                                                   sizeof(float),
                                                   cudaMemcpyDeviceToHost,
                                                   mRunState.MainStream));
                        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
                        return std::sqrt(host_sum);
                    };
                    const float norm_before = norm_of(*dB_ptr);
                    std::fprintf(stderr,
                                 "  [DSL DEBUG MATMUL] dB before norm=%g M=%d N=%d K=%d mode=%d accum=%d\n",
                                 norm_before, M, N, K, static_cast<int>(mode_col), do_accumulate ? 1 : 0);
                }
                ++debug_db_count;
            }
            matmul(*dB_ptr, *rhs, *lhs, std::nullopt, nullptr, nullptr,
                   mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
                   N, M, K, mode_col, do_accumulate, mRunState.MainStream);
            if (env_enabled("SUROGATE_DSL_DEBUG_MATMUL_BWD")) {
                static int debug_db_after_count = 0;
                static Tensor debug_norm_buf_db_after{};
                if (!debug_norm_buf_db_after.Data && mRunState.Allocator) {
                    const int num_sums = get_max_num_block_sums(mRunState.DeviceProp);
                    debug_norm_buf_db_after = mRunState.Allocator->allocate(ETensorDType::FP32,
                                                                            "debug_matmul_bwd_db_norm_after",
                                                                            EAllocationType::ON_DEVICE,
                                                                            {num_sums});
                }
                if (debug_norm_buf_db_after.Data && debug_db_after_count < 4) {
                    auto norm_of = [&](const Tensor& t) {
                        fill_zero(debug_norm_buf_db_after, mRunState.MainStream);
                        global_norm_squared(debug_norm_buf_db_after, t, t.nelem(), mRunState.DeviceProp, mRunState.MainStream);
                        deterministic_sum(debug_norm_buf_db_after.template get<float>(),
                                          debug_norm_buf_db_after.template get<float>(),
                                          debug_norm_buf_db_after.nelem(),
                                          mRunState.MainStream);
                        float host_sum = 0.0f;
                        CUDA_CHECK(cudaMemcpyAsync(&host_sum,
                                                   debug_norm_buf_db_after.template get<float>(),
                                                   sizeof(float),
                                                   cudaMemcpyDeviceToHost,
                                                   mRunState.MainStream));
                        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
                        return std::sqrt(host_sum);
                    };
                    const float norm_after = norm_of(*dB_ptr);
                    std::fprintf(stderr, "  [DSL DEBUG MATMUL] dB after norm=%g\n", norm_after);
                }
                ++debug_db_after_count;
            }
        }
    }

    // Hook invocation for LoRA backward
    if (hook && *hook && op.attrs.backward_hook_point.has_value()) {
        // Temporarily map grads to current backward tensors for LoRA hooks, then restore.
        struct GradPtrs {
            std::byte* d_swiglu{nullptr};
            std::byte* d_ln2{nullptr};
            std::byte* d_att{nullptr};
            std::byte* d_ln1{nullptr};
            std::byte* d_res_ffn{nullptr};
            std::byte* d_mlp_up{nullptr};
            std::byte* d_res_att{nullptr};
            std::byte* d_qkv{nullptr};
        } prev{};

        if (op.attrs.matmul_op.has_value() && layer_idx >= 0) {
            auto& grads = mRunState.simplified_grads(layer_idx);
            prev.d_swiglu = reinterpret_cast<std::byte*>(grads.d_swiglu.Data);
            prev.d_ln2 = reinterpret_cast<std::byte*>(grads.d_ln2.Data);
            prev.d_att = reinterpret_cast<std::byte*>(grads.d_att.Data);
            prev.d_ln1 = reinterpret_cast<std::byte*>(grads.d_ln1.Data);
            prev.d_res_ffn = reinterpret_cast<std::byte*>(grads.d_res_ffn.Data);
            prev.d_mlp_up = reinterpret_cast<std::byte*>(grads.d_mlp_up.Data);
            prev.d_res_att = reinterpret_cast<std::byte*>(grads.d_res_att.Data);
            prev.d_qkv = reinterpret_cast<std::byte*>(grads.d_qkv.Data);

            if (dA_ptr) {
                switch (*op.attrs.matmul_op) {
                    case modules::MatmulOp::MLPDown:
                        grads.d_swiglu.Data = dA_ptr->Data;
                        break;
                    case modules::MatmulOp::MLPUp:
                        grads.d_ln2.Data = dA_ptr->Data;
                        break;
                    case modules::MatmulOp::AttnOut:
                        grads.d_att.Data = dA_ptr->Data;
                        break;
                    case modules::MatmulOp::QKV:
                        grads.d_ln1.Data = dA_ptr->Data;
                        break;
                    default:
                        break;
                }
            }

            switch (*op.attrs.matmul_op) {
                case modules::MatmulOp::MLPDown:
                    grads.d_res_ffn.Data = d_out.Data;
                    break;
                case modules::MatmulOp::MLPUp:
                    grads.d_mlp_up.Data = d_out.Data;
                    break;
                case modules::MatmulOp::AttnOut:
                    grads.d_res_att.Data = d_out.Data;
                    break;
                case modules::MatmulOp::QKV:
                    grads.d_qkv.Data = d_out.Data;
                    break;
                default:
                    break;
            }
        }

        // Ensure activations needed by LoRA hooks are available.
        if (layer_idx >= 0 && op.attrs.matmul_op.has_value()) {
            auto& acts = mRunState.simplified_acts(layer_idx);
            if (*op.attrs.matmul_op == modules::MatmulOp::MLPDown) {
                // LoRA backward hook needs acts.swiglu (forward activation).
                // With recompute enabled, swiglu may have been stack-allocated and freed.
                if (!acts.swiglu.Data && acts.mlp_up.Data) {
                    mRunState.temp_acquire(acts.swiglu);
                    const int Bv = static_cast<int>(mB);
                    const int Tv = static_cast<int>(mT);
                    const int D = static_cast<int>(mConfig.IntermediateSize);
                    swiglu_forward(acts.swiglu, acts.mlp_up, nullptr, Bv, Tv, D, mRunState.MainStream);
                }
            }
        }
        (*hook)(layer_idx, do_accumulate, mRunState.MainStream, *op.attrs.backward_hook_point, mHookContext);

        if (op.attrs.matmul_op.has_value() && layer_idx >= 0) {
            auto& grads = mRunState.simplified_grads(layer_idx);
            grads.d_swiglu.Data = prev.d_swiglu;
            grads.d_ln2.Data = prev.d_ln2;
            grads.d_att.Data = prev.d_att;
            grads.d_ln1.Data = prev.d_ln1;
            grads.d_res_ffn.Data = prev.d_res_ffn;
            grads.d_mlp_up.Data = prev.d_mlp_up;
            grads.d_res_att.Data = prev.d_res_att;
            grads.d_qkv.Data = prev.d_qkv;
        }
    }
}

void CompiledExecutor::dispatch_bias_add_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);

    // d_input = d_out (pass through)
    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        mTensorMap[op.outputs[0].name] = d_out;
    }

    // d_bias = sum(d_out, axis=[0,1]) for [B,T,C] or axis=0 for [N,C]
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        int Bv = 1, Tv = 1, OC = 1;
        if (d_out.Rank == 2) {
            Bv = static_cast<int>(d_out.Sizes[0]);
            Tv = 1;
            OC = static_cast<int>(d_out.Sizes[1]);
        } else {
            Bv = static_cast<int>(d_out.Sizes[0]);
            Tv = static_cast<int>(d_out.Sizes[1]);
            OC = static_cast<int>(d_out.Sizes[2]);
        }

        Tensor& d_bias = ensure_output_tensor(op.outputs[1]);
        bool accumulate = mAccumulateTensors.count(op.outputs[1].name) > 0;
        if (!accumulate && !op.outputs[1].name.empty()) {
            if (auto base = base_param_from_grad(op.outputs[1].name)) {
                accumulate = mAccumulateTensors.count("d_" + *base) > 0;
            }
        }

        // Allocate scratch buffer for bias reduction
        const int scratch_bytes = get_bias_backward_scratch_size(d_out.DType, OC, mRunState.DeviceProp);
        Tensor scratch = mRunState.temp_alloc(ETensorDType::FP32, {static_cast<long>(scratch_bytes / sizeof(float))});
        mTemps.push_back(scratch);

        if (accumulate) {
            // Accumulate into existing gradient: compute to tmp, then add
            Tensor tmp = mRunState.temp_alloc(d_out.DType, {static_cast<long>(OC)});
            mTemps.push_back(tmp);
            backward_bias(tmp, d_out, nullptr, nullptr, scratch, Bv, Tv, OC, mRunState.DeviceProp, mRunState.MainStream);
            vector_add_sr(d_bias, d_bias, tmp, 1.0f, static_cast<long>(d_bias.nelem()), 0, mRunState.MainStream);
        } else {
            backward_bias(d_bias, d_out, nullptr, nullptr, scratch, Bv, Tv, OC, mRunState.DeviceProp, mRunState.MainStream);
        }
    }
}

void CompiledExecutor::dispatch_swiglu_backward(const CompiledOp& op) {
    // inputs: d_out, input (the mlp_up output before swiglu)
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);
    Tensor& d_inp = ensure_output_tensor(op.outputs[0]);

    // For FP8 hybrid backward, record abs_max of d_mlp_up for subsequent quantization
    float* abs_max_ptr = mRunState.has_fp8_hybrid_backward()
        ? mRunState.simplified_quant_grads().d_mlp_up.abs_max()
        : nullptr;

    const long D = d_out.Sizes[2];
    swiglu_backward(d_inp, d_out, inp, abs_max_ptr,
                    static_cast<int>(d_out.Sizes[0]),
                    static_cast<int>(d_out.Sizes[1]),
                    static_cast<int>(D), mRunState.MainStream);
}

void CompiledExecutor::dispatch_matmul_swiglu_backward(const CompiledOp& op, const modules::BackwardHook* hook) {
    // Combined backward for matmul + swiglu (fused op in forward)
    // inputs: d_swiglu_out, ln2 (matmul input), mlp_up_weight, mlp_up (pre-swiglu)
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);
    Tensor& weight = resolve_tensor(op.inputs[2]);
    Tensor mlp_up = resolve_tensor(op.inputs[3]);

    const int layer_idx = op.attrs.layer_idx;

    // Recompute mlp_up if the saved tensor was stack-allocated and freed
    bool recomputed_mlp_up = false;
    if (!mlp_up.Data || (mRunState.Stack.owns(mlp_up.Data) && !mRunState.Stack.is_live(mlp_up.Data))) {
        int M = 0, N = 0, K = 0;
        Tensor inp_flat = (inp.Rank == 2) ? inp : view_tensor(inp, {mB * mT, inp.Sizes[inp.Rank - 1]});
        matmul_dims(inp_flat, weight, op.attrs.transpose, M, N, K);
        const long D2 = N;
        Tensor mlp_up_flat = mRunState.temp_alloc(inp.DType, {mB * mT, D2});
        mTemps.push_back(mlp_up_flat);

        EMMTranspose mode_col = swap_transpose(op.attrs.transpose);
        matmul(mlp_up_flat, weight, inp_flat, std::nullopt, nullptr, nullptr,
               mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
               N, M, K, mode_col, false, mRunState.MainStream);

        mlp_up = view_tensor(mlp_up_flat, {mB, mT, D2});
        if (layer_idx >= 0) {
            auto& acts = mRunState.simplified_acts(layer_idx);
            acts.mlp_up.Data = mlp_up.Data;
        }
        recomputed_mlp_up = true;
    }

    // First: swiglu backward
    Tensor* d_mlp_up_ptr = nullptr;
    if (layer_idx >= 0) {
        auto& grads = mRunState.simplified_grads(layer_idx);
        d_mlp_up_ptr = &grads.d_mlp_up;
        if (!d_mlp_up_ptr->Data) {
            mRunState.temp_acquire(*d_mlp_up_ptr);
            mTemps.push_back(*d_mlp_up_ptr);
        }
    }
    Tensor d_mlp_up = d_mlp_up_ptr ? *d_mlp_up_ptr
                                   : mRunState.temp_alloc(mlp_up.DType, {mlp_up.Sizes[0], mlp_up.Sizes[1], mlp_up.Sizes[2]});
    if (!d_mlp_up_ptr) {
        mTemps.push_back(d_mlp_up);
    }

    // For FP8 hybrid backward, record abs_max of d_mlp_up for subsequent quantization
    float* abs_max_ptr = mRunState.has_fp8_hybrid_backward()
        ? mRunState.simplified_quant_grads().d_mlp_up.abs_max()
        : nullptr;

    const long D = d_out.Sizes[2];
    swiglu_backward(d_mlp_up, d_out, mlp_up, abs_max_ptr,
                    static_cast<int>(d_out.Sizes[0]),
                    static_cast<int>(d_out.Sizes[1]),
                    static_cast<int>(D), mRunState.MainStream);

    // Then: matmul backward
    Tensor d_mlp_up_flat = view_tensor(d_mlp_up, {mB * mT, 2 * D});

    Tensor* d_inp_ptr = nullptr;
    Tensor* d_weight_ptr = nullptr;

    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        d_inp_ptr = &ensure_output_tensor(op.outputs[0]);
    }
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        d_weight_ptr = &ensure_output_tensor(op.outputs[1]);
    }

    bool do_accumulate = false;
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        do_accumulate = (mAccumulateTensors.count(op.outputs[1].name) > 0);
        if (!do_accumulate) {
            if (auto base = base_param_from_grad(op.outputs[1].name)) {
                do_accumulate = mAccumulateTensors.count("d_" + *base) > 0;
            }
        }
    }

    if (d_inp_ptr) {
        EMMTranspose mode_dA = EMMTranspose::NN;
        switch (op.attrs.transpose) {
            case EMMTranspose::NN:
                mode_dA = EMMTranspose::NT;
                break;
            case EMMTranspose::NT:
                mode_dA = EMMTranspose::NN;
                break;
            case EMMTranspose::TN:
                mode_dA = EMMTranspose::NT;
                break;
            case EMMTranspose::TT:
                mode_dA = EMMTranspose::TT;
                break;
        }

        int M = 0, N = 0, K = 0;
        matmul_dims(d_mlp_up_flat, weight, mode_dA, M, N, K);
        EMMTranspose mode_col = swap_transpose(mode_dA);
        matmul(*d_inp_ptr, weight, d_mlp_up_flat, std::nullopt, nullptr, nullptr,
               mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
               N, M, K, mode_col, false, mRunState.MainStream);
    }
    if (d_weight_ptr) {
        Tensor inp_flat = (inp.Rank == 2) ? inp : view_tensor(inp, {mB * mT, inp.Sizes[inp.Rank - 1]});

        const Tensor* lhs = nullptr;
        const Tensor* rhs = nullptr;
        EMMTranspose mode_rm = EMMTranspose::NN;
        switch (op.attrs.transpose) {
            case EMMTranspose::NN:
                lhs = &inp_flat;
                rhs = &d_mlp_up_flat;
                mode_rm = EMMTranspose::TN;
                break;
            case EMMTranspose::NT:
                lhs = &d_mlp_up_flat;
                rhs = &inp_flat;
                mode_rm = EMMTranspose::TN;
                break;
            case EMMTranspose::TN:
                lhs = &inp_flat;
                rhs = &d_mlp_up_flat;
                mode_rm = EMMTranspose::NN;
                break;
            case EMMTranspose::TT:
                lhs = &d_mlp_up_flat;
                rhs = &inp_flat;
                mode_rm = EMMTranspose::TT;
                break;
        }

        int M = 0, N = 0, K = 0;
        matmul_dims(*lhs, *rhs, mode_rm, M, N, K);
        EMMTranspose mode_col = swap_transpose(mode_rm);
        matmul(*d_weight_ptr, *rhs, *lhs, std::nullopt, nullptr, nullptr,
               mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
               N, M, K, mode_col, do_accumulate, mRunState.MainStream);
    }

    if (layer_idx >= 0 && d_inp_ptr) {
        auto& grads = mRunState.simplified_grads(layer_idx);
        grads.d_ln2.Data = d_inp_ptr->Data;
    }

    // Hook invocation for LoRA backward (MLP up/gate)
    if (hook && *hook && op.attrs.backward_hook_point.has_value()) {
        (*hook)(layer_idx, do_accumulate, mRunState.MainStream, *op.attrs.backward_hook_point, mHookContext);
    }
}

void CompiledExecutor::dispatch_rope_backward(const CompiledOp& op) {
    // inputs: d_qkv_rope, freq_cis, position_ids
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& freqs = resolve_tensor(op.inputs[1]);
    Tensor& pos_ids = resolve_tensor(op.inputs[2]);
    Tensor& d_qkv = ensure_output_tensor(op.outputs[0]);

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());

    // For FP8 hybrid backward, record abs_max of d_qkv for subsequent quantization
    float* abs_max_ptr = mRunState.has_fp8_hybrid_backward()
        ? mRunState.simplified_quant_grads().d_qkv.abs_max()
        : nullptr;

    rope_backward(d_qkv, d_out, freqs, reinterpret_cast<int*>(pos_ids.Data), abs_max_ptr,
                  static_cast<int>(mB), static_cast<int>(mT),
                  Hq, Hkv, Hs, op.attrs.rotary_dim, mRunState.MainStream);
}

void CompiledExecutor::dispatch_qkv_qk_norm_rope_backward(const CompiledOp& op) {
    // inputs (from autodiff): d_qkv_out, qkv_out (saved), q_norm_weight, k_norm_weight, q_rstd, k_rstd, freqs, pos_ids
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& qkv = resolve_tensor(op.inputs[1]);       // Saved QKV output from forward
    Tensor& q_norm = resolve_tensor(op.inputs[2]);
    Tensor& k_norm = resolve_tensor(op.inputs[3]);
    Tensor& q_rstd = resolve_tensor(op.inputs[4]);    // Saved RSTD (FP32)
    Tensor& k_rstd = resolve_tensor(op.inputs[5]);    // Saved RSTD (FP32)
    Tensor& freqs = resolve_tensor(op.inputs[6]);
    Tensor& pos_ids = resolve_tensor(op.inputs[7]);

    Tensor& d_qkv = ensure_output_tensor(op.outputs[0]);

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    const int qkv_channels = Hs * (Hq + 2 * Hkv);
    const int q_rows = Hq * Hs;

    Tensor qkv_view = (qkv.Rank == 4) ? view_tensor(qkv, {mB, mT, static_cast<long>(qkv_channels)}) : qkv;
    Tensor d_out_view = (d_out.Rank == 4) ? view_tensor(d_out, {mB, mT, static_cast<long>(qkv_channels)}) : d_out;
    Tensor d_qkv_view = (d_qkv.Rank == 4) ? view_tensor(d_qkv, {mB, mT, static_cast<long>(qkv_channels)}) : d_qkv;

    // Initialize d_qkv with upstream gradient (d_out) so V gradients pass through unchanged.
    // The fused or fallback kernels update Q/K channels in-place.
    if (d_qkv_view.Data != d_out_view.Data) {
        const std::size_t bytes = static_cast<std::size_t>(d_out_view.nelem()) * get_dtype_size(d_out_view.DType);
        CUDA_CHECK(cudaMemcpyAsync(d_qkv_view.Data, d_out_view.Data, bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
    }

    const bool disable_fused = env_enabled("SUROGATE_DISABLE_FUSED_QK_ROPE_BWD");
    if (disable_fused) {
        // Fallback: undo RoPE on gradients and activations, then run non-RoPE QK RMSNorm backward.
        const int rotary_dim = op.attrs.rotary_dim;
        rope_backward(d_qkv_view, d_qkv_view, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
                      static_cast<int>(mB), static_cast<int>(mT),
                      Hq, Hkv, Hs, rotary_dim, mRunState.MainStream);
        rope_backward(qkv_view, qkv_view, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
                      static_cast<int>(mB), static_cast<int>(mT),
                      Hq, Hkv, Hs, rotary_dim, mRunState.MainStream);
        qkv_head_rmsnorm_backward_dx(d_qkv_view, qkv_view, q_norm, q_rstd,
                                     static_cast<int>(mB), static_cast<int>(mT), qkv_channels,
                                     Hq, Hs, 0, mRunState.MainStream);
        qkv_head_rmsnorm_backward_dx(d_qkv_view, qkv_view, k_norm, k_rstd,
                                     static_cast<int>(mB), static_cast<int>(mT), qkv_channels,
                                     Hkv, Hs, q_rows, mRunState.MainStream);
    } else {
        // Combined backward for Q and K norms with RoPE
        // Q norm backward (with RoPE): channel_offset=0
        qkv_head_rmsnorm_rope_backward_dx(d_qkv_view, qkv_view, q_norm, q_rstd,
                                           freqs, reinterpret_cast<int*>(pos_ids.Data),
                                           static_cast<int>(mB), static_cast<int>(mT), qkv_channels,
                                           Hq, Hs, 0, mRunState.MainStream, nullptr);

        // K norm backward (with RoPE): channel_offset=q_rows
        qkv_head_rmsnorm_rope_backward_dx(d_qkv_view, qkv_view, k_norm, k_rstd,
                                           freqs, reinterpret_cast<int*>(pos_ids.Data),
                                           static_cast<int>(mB), static_cast<int>(mT), qkv_channels,
                                           Hkv, Hs, q_rows, mRunState.MainStream, nullptr);
    }

    // V doesn't have normalization - its gradients pass through unchanged
    // The d_out already contains the V gradients at the correct offset

    // For FP8 hybrid backward, record abs_max of the final d_qkv for subsequent quantization
    if (mRunState.has_fp8_hybrid_backward()) {
        float* abs_max_ptr = mRunState.simplified_quant_grads().d_qkv.abs_max();
        abs_max(abs_max_ptr, d_qkv_view, static_cast<long>(d_qkv_view.nelem()),
                mRunState.DeviceProp, mRunState.MainStream);
    }
}

void CompiledExecutor::dispatch_flash_attention_backward(const CompiledOp& op) {
    // inputs (from autodiff): d_out, out (attention output), lse, qkv
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& out = resolve_tensor(op.inputs[1]);
    Tensor& lse = resolve_tensor(op.inputs[2]);
    Tensor& qkv = resolve_tensor(op.inputs[3]);
    Tensor& d_qkv = ensure_output_tensor(op.outputs[0]);

    Tensor* out_ptr = &out;
    Tensor* lse_ptr = &lse;
    Tensor* qkv_ptr = &qkv;

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());

    if (!mRunState.scratch().cudnn_workspace.Data) {
        mRunState.temp_acquire(mRunState.scratch().cudnn_workspace);
        mTemps.push_back(mRunState.scratch().cudnn_workspace);
    }

    // Parse layer_idx for use in cuDNN call
    int layer_idx = -1;
    std::string field;
    parse_block_param(op.inputs[3].name, layer_idx, field);

    static bool debug_recompute = env_enabled("SUROGATE_DEBUG_RECOMPUTE");
    static bool debug_grad_norms = env_enabled("SUROGATE_DEBUG_GRAD_NORMS");
    if (debug_recompute) {
        std::fprintf(stderr, "[RECOMPUTE DEBUG] flash_attention_backward layer=%d qkv_ptr=%p out_ptr=%p lse_ptr=%p last_recompute=%d\n",
                     layer_idx, qkv_ptr->Data, out_ptr->Data, lse_ptr->Data, mLastRecomputeLayer);
    }

    // Debug: compute input norms for gradient tracking
    static Tensor attn_bwd_norm_buf{};
    auto compute_norm_attn = [&](const Tensor& t) -> float {
        if (!debug_grad_norms || !t.Data) return 0.0f;
        if (!attn_bwd_norm_buf.Data && mRunState.Allocator) {
            const int num_sums = get_max_num_block_sums(mRunState.DeviceProp);
            attn_bwd_norm_buf = mRunState.Allocator->allocate(ETensorDType::FP32, "debug_attn_bwd_norm",
                                                              EAllocationType::ON_DEVICE, {num_sums});
        }
        if (!attn_bwd_norm_buf.Data) return 0.0f;
        fill_zero(attn_bwd_norm_buf, mRunState.MainStream);
        global_norm_squared(attn_bwd_norm_buf, t, t.nelem(), mRunState.DeviceProp, mRunState.MainStream);
        deterministic_sum(attn_bwd_norm_buf.template get<float>(),
                          attn_bwd_norm_buf.template get<float>(),
                          attn_bwd_norm_buf.nelem(), mRunState.MainStream);
        float host_sum = 0.0f;
        CUDA_CHECK(cudaMemcpyAsync(&host_sum, attn_bwd_norm_buf.template get<float>(),
                                   sizeof(float), cudaMemcpyDeviceToHost, mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        return std::sqrt(host_sum);
    };

    if (debug_grad_norms) {
        std::fprintf(stderr, "[GRAD NORM] attn_bwd layer=%d BEFORE:\n", layer_idx);
        std::fprintf(stderr, "  d_out: norm=%.6g ptr=%p\n", compute_norm_attn(d_out), d_out.Data);
        std::fprintf(stderr, "  out (att): norm=%.6g ptr=%p name=%s\n", compute_norm_attn(*out_ptr), out_ptr->Data, op.inputs[1].name.c_str());
        std::fprintf(stderr, "  lse: norm=%.6g ptr=%p name=%s\n", compute_norm_attn(*lse_ptr), lse_ptr->Data, op.inputs[2].name.c_str());
        std::fprintf(stderr, "  qkv: norm=%.6g ptr=%p name=%s\n", compute_norm_attn(*qkv_ptr), qkv_ptr->Data, op.inputs[3].name.c_str());
        // Check expected values from simplified_acts
        if (layer_idx >= 0) {
            auto& acts = mRunState.simplified_acts(layer_idx);
            std::fprintf(stderr, "  [expected] acts.att: ptr=%p norm=%.6g\n", acts.att.Data, compute_norm_attn(acts.att));
            std::fprintf(stderr, "  [expected] acts.lse: ptr=%p norm=%.6g\n", acts.lse.Data, compute_norm_attn(acts.lse));
            std::fprintf(stderr, "  [expected] acts.qkv: ptr=%p norm=%.6g\n", acts.qkv.Data, compute_norm_attn(acts.qkv));
            if (out_ptr->Data != acts.att.Data) {
                std::fprintf(stderr, "  WARNING: out_ptr != acts.att!\n");
            }
            if (lse_ptr->Data != acts.lse.Data) {
                std::fprintf(stderr, "  WARNING: lse_ptr != acts.lse!\n");
            }
        }
    }

    // FIX: Zero-initialize d_qkv before cuDNN attention backward to prevent NaN from uninitialized memory.
    // The d_qkv buffer may contain stale values from previous operations, and cuDNN attention backward
    // may read parts of this buffer even though it's expected to be output-only. Without this zero-init,
    // NaN values can appear in the gradient computation and propagate through the backward pass.
    fill_zero(d_qkv, mRunState.MainStream);

    const int attn_chunks = mOptions.AttBwdChunks;
    if (attn_chunks < 1) {
        throw std::runtime_error("attn_bwd_chunks must be >= 1");
    }
    const int chunk_B = (attn_chunks == 1)
        ? static_cast<int>(mB)
        : static_cast<int>(div_exact(mB, static_cast<long>(attn_chunks)));

    // Signature: attention_backward_cudnn(dqkv, stats, out, dout, qkv, workspace, handle, B, T, Hq, Hkv, HS, stream)
    if (attn_chunks == 1) {
        attention_backward_cudnn(d_qkv, *lse_ptr, *out_ptr, d_out, *qkv_ptr,
                                 mRunState.scratch().cudnn_workspace,
                                 mRunState.CudnnHandle,
                                 static_cast<int>(mB), static_cast<int>(mT),
                                 Hq, Hkv, Hs, mRunState.MainStream);
        if (debug_grad_norms) {
            std::fprintf(stderr, "[GRAD NORM] attn_bwd layer=%d AFTER: d_qkv norm=%.6g\n",
                         layer_idx, compute_norm_attn(d_qkv));
        }
        return;
    }

    for (int chunk = 0; chunk < attn_chunks; ++chunk) {
        const long start = static_cast<long>(chunk) * static_cast<long>(chunk_B);
        const long end = start + static_cast<long>(chunk_B);
        Tensor d_out_chunk = slice(d_out, 0, start, end);
        Tensor out_chunk = slice(*out_ptr, 0, start, end);
        Tensor lse_chunk = slice(*lse_ptr, 0, start, end);
        Tensor qkv_chunk = slice(*qkv_ptr, 0, start, end);
        Tensor d_qkv_chunk = slice(d_qkv, 0, start, end);

        attention_backward_cudnn(d_qkv_chunk, lse_chunk, out_chunk, d_out_chunk, qkv_chunk,
                                 mRunState.scratch().cudnn_workspace,
                                 mRunState.CudnnHandle,
                                 chunk_B, static_cast<int>(mT),
                                 Hq, Hkv, Hs, mRunState.MainStream);
    }
}

void CompiledExecutor::dispatch_zeros_backward(const CompiledOp& op) {
    // Zeros backward is a no-op - gradient doesn't flow through zeros initialization
}

void CompiledExecutor::dispatch_fused_residual_rmsnorm_backward(const CompiledOp& op) {
    // inputs: d_y, d_residual_next (may be empty), residual_out, weight, rstd
    // outputs: d_residual, d_input, d_weight (optional)
    Tensor& d_y = resolve_tensor(op.inputs[0]);
    Tensor* residual_out_ptr = &resolve_tensor(op.inputs[2]);
    Tensor& weight = resolve_tensor(op.inputs[3]);
    Tensor& rstd = resolve_tensor(op.inputs[4]);

    int ln_layer_idx = -1;
    std::string ln_field;
    if (!op.inputs[3].name.empty()) {
        parse_block_param(op.inputs[3].name, ln_layer_idx, ln_field);
    }
    if (ln_layer_idx >= 0 && ln_field == "ln1_weight") {
        // LN1 backward expects residual_out from the forward fused residual op.
        // The residual INPUT to layer L is res_ffn[L-1] (output of previous layer).
        // For layer 0, the input is encoded (embeddings + position encoding).
        if (ln_layer_idx == 0) {
            residual_out_ptr = &mRunState.non_block_activations().encoded;
        } else {
            // Ensure residual is fetched from CPU if offloading is enabled.
            // The recompute function should have done this, but fetch again for safety
            // (e.g., FFT mode where recompute may be skipped).
            if (mRunState.has_residual_offloading()) {
                mRunState.fetch_residual(ln_layer_idx - 1, mRunState.side_stream());
            }
            residual_out_ptr = &mRunState.get_residual(ln_layer_idx - 1, mRunState.MainStream);
        }
    }
    // FIX: LN2 backward needs the recomputed residual_att from simplified_acts.
    // The backward graph may have wrong tensor names for the last layer (e.g., "StackedBlocks_N"
    // instead of "blocks[N].res_att"), causing it to resolve to stale/wrong data.
    // When recompute is enabled, always use the freshly recomputed residual_att.
    if (ln_layer_idx >= 0 && ln_field == "ln2_weight" && mRecomputeEnabled) {
        residual_out_ptr = &mRunState.simplified_acts(ln_layer_idx).residual_att;
    }
    Tensor& residual_out = *residual_out_ptr;

    // Debug: FFT recompute verification
    static bool debug_fft_recompute = env_enabled("SUROGATE_DEBUG_FFT_RECOMPUTE");
    if (debug_fft_recompute && ln_layer_idx >= 0) {
        static Tensor fft_norm_buf{};
        if (!fft_norm_buf.Data && mRunState.Allocator) {
            const int num_sums = get_max_num_block_sums(mRunState.DeviceProp);
            fft_norm_buf = mRunState.Allocator->allocate(ETensorDType::FP32,
                                                          "debug_fft_ln_bwd_norm",
                                                          EAllocationType::ON_DEVICE,
                                                          {num_sums});
        }
        auto norm_of = [&](const Tensor& t) -> float {
            if (!fft_norm_buf.Data || !t.Data) return 0.0f;
            fill_zero(fft_norm_buf, mRunState.MainStream);
            global_norm_squared(fft_norm_buf, t, t.nelem(), mRunState.DeviceProp, mRunState.MainStream);
            deterministic_sum(fft_norm_buf.template get<float>(),
                              fft_norm_buf.template get<float>(),
                              fft_norm_buf.nelem(),
                              mRunState.MainStream);
            float host_sum = 0.0f;
            CUDA_CHECK(cudaMemcpyAsync(&host_sum,
                                       fft_norm_buf.template get<float>(),
                                       sizeof(float),
                                       cudaMemcpyDeviceToHost,
                                       mRunState.MainStream));
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            return std::sqrt(host_sum);
        };

        std::fprintf(stderr, "[FFT LN BWD] layer=%d field=%s last_recompute=%d\n",
                     ln_layer_idx, ln_field.c_str(), mLastRecomputeLayer);
        std::fprintf(stderr, "  d_y: ptr=%p norm=%.6g\n", d_y.Data, norm_of(d_y));
        std::fprintf(stderr, "  residual_out: ptr=%p norm=%.6g (input[2].name=%s slot=%d)\n",
                     residual_out.Data, norm_of(residual_out),
                     op.inputs[2].name.c_str(), static_cast<int>(op.inputs[2].slot));
        std::fprintf(stderr, "  rstd: ptr=%p norm=%.6g\n", rstd.Data, norm_of(rstd));

        // Check if residual_out matches the expected recomputed value
        if (ln_field == "ln2_weight") {
            // For LN2, residual_out should be residual_att from simplified_acts
            auto& acts = mRunState.simplified_acts(ln_layer_idx);
            std::fprintf(stderr, "  acts.residual_att: ptr=%p norm=%.6g\n",
                         acts.residual_att.Data, norm_of(acts.residual_att));
            if (residual_out.Data != acts.residual_att.Data) {
                std::fprintf(stderr, "  WARNING: residual_out ptr != acts.residual_att ptr!\n");
            }
        }

        // Check for NaN/Inf
        float d_y_norm = norm_of(d_y);
        float residual_norm = norm_of(residual_out);
        float rstd_norm = norm_of(rstd);
        if (!std::isfinite(d_y_norm) || !std::isfinite(residual_norm) || !std::isfinite(rstd_norm)) {
            std::fprintf(stderr, "  ERROR: NaN/Inf detected in inputs!\n");
        }
    }

    if (env_enabled("SUROGATE_DSL_DEBUG_LN_BWD")) {
        static int debug_ln_count = 0;
        static Tensor debug_ln_norm_buf{};
        if (!debug_ln_norm_buf.Data && mRunState.Allocator) {
            const int num_sums = get_max_num_block_sums(mRunState.DeviceProp);
            debug_ln_norm_buf = mRunState.Allocator->allocate(ETensorDType::FP32,
                                                              "debug_ln_bwd_norm",
                                                              EAllocationType::ON_DEVICE,
                                                              {num_sums});
        }
        const char* ln_layer_env = std::getenv("SUROGATE_DSL_DEBUG_LN_LAYER");
        const int ln_layer_filter = ln_layer_env ? std::atoi(ln_layer_env) : -999999;
        const bool ln_layer_match = !ln_layer_env || ln_layer_idx == ln_layer_filter;
        if (ln_layer_match && debug_ln_count < 16) {
            std::fprintf(stderr,
                         "[DSL DEBUG LN BWD] layer=%d field=%s in0=%s in1=%s out0=%s slot0=%d out1=%s slot1=%d\n",
                         ln_layer_idx,
                         ln_field.c_str(),
                         op.inputs.size() > 0 ? op.inputs[0].name.c_str() : "",
                         op.inputs.size() > 1 ? op.inputs[1].name.c_str() : "",
                         op.outputs.size() > 0 ? op.outputs[0].name.c_str() : "",
                         op.outputs.size() > 0 ? static_cast<int>(op.outputs[0].slot) : -1,
                         op.outputs.size() > 1 ? op.outputs[1].name.c_str() : "",
                         op.outputs.size() > 1 ? static_cast<int>(op.outputs[1].slot) : -1);
            auto norm_of = [&](const Tensor& t) {
                fill_zero(debug_ln_norm_buf, mRunState.MainStream);
                global_norm_squared(debug_ln_norm_buf, t, t.nelem(), mRunState.DeviceProp, mRunState.MainStream);
                deterministic_sum(debug_ln_norm_buf.template get<float>(),
                                  debug_ln_norm_buf.template get<float>(),
                                  debug_ln_norm_buf.nelem(),
                                  mRunState.MainStream);
                float host_sum = 0.0f;
                CUDA_CHECK(cudaMemcpyAsync(&host_sum,
                                           debug_ln_norm_buf.template get<float>(),
                                           sizeof(float),
                                           cudaMemcpyDeviceToHost,
                                           mRunState.MainStream));
                CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
                return std::sqrt(host_sum);
            };
            if (debug_ln_norm_buf.Data) {
                // Print d_y (in0) norm - this is upstream gradient from MLPUp backward
                std::fprintf(stderr, "  [DSL DEBUG LN BWD] in0 (d_y) norm=%g\n", norm_of(d_y));
                // Print d_residual_next (in1) norm if present
                if (op.inputs.size() > 1 && !op.inputs[1].name.empty()) {
                    Tensor& in1 = resolve_tensor(op.inputs[1]);
                    std::fprintf(stderr, "  [DSL DEBUG LN BWD] in1 (d_residual_next) norm=%g\n", norm_of(in1));
                }
            }
        }
        ++debug_ln_count;
    }

    // d_residual_next is the incoming gradient from the next layer (may be zero/empty)
    Tensor d_residual_zero{};
    Tensor* d_residual_next = nullptr;
    if (!op.inputs[1].name.empty()) {
        d_residual_next = &resolve_tensor(op.inputs[1]);
    } else {
        // Allocate and zero a temporary for d_residual if none provided
        d_residual_zero = mRunState.temp_alloc(d_y.DType, {mB, mT, static_cast<long>(mConfig.HiddenSize)});
        fill_zero(d_residual_zero, mRunState.MainStream);
        mTemps.push_back(d_residual_zero);
        d_residual_next = &d_residual_zero;
    }
    Tensor* d_residual_input = d_residual_next;
    Tensor* d_residual_stream = d_residual_next;

    Tensor& d_input = ensure_output_tensor(op.outputs[1]);

    // d_weight may be nullptr if weight is frozen
    Tensor dummy_weight{};
    Tensor* d_weight_ptr = nullptr;
    bool skip_weight_grad = true;
    if (op.outputs.size() > 2 && !op.outputs[2].name.empty()) {
        d_weight_ptr = &ensure_output_tensor(op.outputs[2]);
        skip_weight_grad = false;
        if (op.outputs[2].slot == TensorSlot::Mapped || op.outputs[2].slot == TensorSlot::Temporary) {
            fill_zero(*d_weight_ptr, mRunState.MainStream);
        }
    } else {
        dummy_weight = mRunState.temp_alloc(weight.DType, {static_cast<long>(mConfig.HiddenSize)});
        mTemps.push_back(dummy_weight);
        d_weight_ptr = &dummy_weight;
    }

    const int C = mConfig.HiddenSize;

    // Determine abs_max pointer for FP8 gradient quantization.
    // LN1 backward produces d_res_ffn (gradient for previous layer's residual).
    // LN2 backward produces d_res_att (gradient for attention path).
    float* abs_max_ptr = nullptr;
    if (mRunState.has_grad_quants()) {
        const bool is_ln2 = (ln_field == "ln2_weight");
        abs_max_ptr = is_ln2
            ? mRunState.simplified_quant_grads().d_res_att.abs_max()
            : mRunState.simplified_quant_grads().d_res_ffn.abs_max();
    }

    rmsnorm_backward(d_input, *d_weight_ptr, mRunState.scratch().rmsnorm_scratch,
                     *d_residual_input, d_y, residual_out, weight, rstd,
                     abs_max_ptr,
                     static_cast<int>(mB), static_cast<int>(mT), C,
                     mRunState.DeviceProp, mRunState.MainStream, skip_weight_grad);

    // Debug: gradient norm tracking
    static bool debug_grad_norms_ln = env_enabled("SUROGATE_DEBUG_GRAD_NORMS");
    if (debug_grad_norms_ln && ln_layer_idx >= 0) {
        static Tensor grad_norm_buf_ln{};
        if (!grad_norm_buf_ln.Data && mRunState.Allocator) {
            const int num_sums = get_max_num_block_sums(mRunState.DeviceProp);
            grad_norm_buf_ln = mRunState.Allocator->allocate(ETensorDType::FP32, "debug_ln_grad_norm",
                                                              EAllocationType::ON_DEVICE, {num_sums});
        }
        auto norm_ln = [&](const Tensor& t) -> float {
            if (!grad_norm_buf_ln.Data || !t.Data) return 0.0f;
            fill_zero(grad_norm_buf_ln, mRunState.MainStream);
            global_norm_squared(grad_norm_buf_ln, t, t.nelem(), mRunState.DeviceProp, mRunState.MainStream);
            deterministic_sum(grad_norm_buf_ln.template get<float>(),
                              grad_norm_buf_ln.template get<float>(),
                              grad_norm_buf_ln.nelem(), mRunState.MainStream);
            float host_sum = 0.0f;
            CUDA_CHECK(cudaMemcpyAsync(&host_sum, grad_norm_buf_ln.template get<float>(),
                                       sizeof(float), cudaMemcpyDeviceToHost, mRunState.MainStream));
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            return std::sqrt(host_sum);
        };
        std::fprintf(stderr, "[GRAD NORM] ln_bwd layer=%d field=%s: d_y=%.6g d_input=%.6g residual_out=%.6g\n",
                     ln_layer_idx, ln_field.c_str(), norm_ln(d_y), norm_ln(d_input), norm_ln(residual_out));
    }

    // Debug: check output gradients after rmsnorm_backward
    if (debug_fft_recompute && ln_layer_idx >= 0) {
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        static Tensor fft_out_norm_buf{};
        if (!fft_out_norm_buf.Data && mRunState.Allocator) {
            const int num_sums = get_max_num_block_sums(mRunState.DeviceProp);
            fft_out_norm_buf = mRunState.Allocator->allocate(ETensorDType::FP32,
                                                              "debug_fft_ln_out_norm",
                                                              EAllocationType::ON_DEVICE,
                                                              {num_sums});
        }
        auto norm_of_out = [&](const Tensor& t) -> float {
            if (!fft_out_norm_buf.Data || !t.Data) return 0.0f;
            fill_zero(fft_out_norm_buf, mRunState.MainStream);
            global_norm_squared(fft_out_norm_buf, t, t.nelem(), mRunState.DeviceProp, mRunState.MainStream);
            deterministic_sum(fft_out_norm_buf.template get<float>(),
                              fft_out_norm_buf.template get<float>(),
                              fft_out_norm_buf.nelem(),
                              mRunState.MainStream);
            float host_sum = 0.0f;
            CUDA_CHECK(cudaMemcpyAsync(&host_sum,
                                       fft_out_norm_buf.template get<float>(),
                                       sizeof(float),
                                       cudaMemcpyDeviceToHost,
                                       mRunState.MainStream));
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            return std::sqrt(host_sum);
        };
        float d_input_norm = norm_of_out(d_input);
        std::fprintf(stderr, "[FFT LN BWD] layer=%d field=%s OUTPUT: d_input norm=%.6g\n",
                     ln_layer_idx, ln_field.c_str(), d_input_norm);
        if (!std::isfinite(d_input_norm)) {
            std::fprintf(stderr, "  ERROR: d_input has NaN/Inf after rmsnorm_backward!\n");
        }
    }

    // Copy d_input to d_residual if they're different outputs
    if (!op.outputs[0].name.empty() && op.outputs[0].name != op.outputs[1].name) {
        Tensor& d_residual = ensure_output_tensor(op.outputs[0]);
        CUDA_CHECK(cudaMemcpyAsync(d_residual.Data, d_input.Data, d_input.bytes(),
                                   cudaMemcpyDeviceToDevice, mRunState.MainStream));
    }

    // Update residual_out gradient buffer to include norm contribution.
    if (d_residual_stream && d_residual_stream->Data && d_residual_stream->Data != d_input.Data) {
        CUDA_CHECK(cudaMemcpyAsync(d_residual_stream->Data, d_input.Data, d_input.bytes(),
                                   cudaMemcpyDeviceToDevice, mRunState.MainStream));
    }

    // LN2 backward produces d_res_ffn (gradient for MLP down output). Mirror it into d_mlp_down
    // so downstream matmul backward sees a valid d_out.
    if (ln_layer_idx >= 0 && ln_field == "ln2_weight") {
        Tensor& d_residual = ensure_output_tensor(op.outputs[0]);
        Tensor& d_mlp_down = mRunState.simplified_grads(ln_layer_idx).d_mlp_down;

        if (env_enabled("SUROGATE_DSL_DEBUG_RECOMPUTE_GRAD")) {
            static Tensor debug_norm_buf_rc{};
            if (!debug_norm_buf_rc.Data && mRunState.Allocator) {
                const int num_sums = get_max_num_block_sums(mRunState.DeviceProp);
                debug_norm_buf_rc = mRunState.Allocator->allocate(ETensorDType::FP32,
                                                                   "debug_recompute_grad_norm",
                                                                   EAllocationType::ON_DEVICE,
                                                                   {num_sums});
            }
            if (debug_norm_buf_rc.Data) {
                auto norm_of = [&](const Tensor& t) {
                    fill_zero(debug_norm_buf_rc, mRunState.MainStream);
                    global_norm_squared(debug_norm_buf_rc, t, t.nelem(), mRunState.DeviceProp, mRunState.MainStream);
                    deterministic_sum(debug_norm_buf_rc.template get<float>(),
                                      debug_norm_buf_rc.template get<float>(),
                                      debug_norm_buf_rc.nelem(),
                                      mRunState.MainStream);
                    float host_sum = 0.0f;
                    CUDA_CHECK(cudaMemcpyAsync(&host_sum,
                                               debug_norm_buf_rc.template get<float>(),
                                               sizeof(float),
                                               cudaMemcpyDeviceToHost,
                                               mRunState.MainStream));
                    CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
                    return std::sqrt(host_sum);
                };
                std::fprintf(stderr,
                             "[DSL DEBUG RECOMPUTE] LN2 bwd layer=%d: d_residual=%p norm=%g, d_mlp_down=%p norm=%g (bucket=%d)\n",
                             ln_layer_idx,
                             d_residual.Data, norm_of(d_residual),
                             d_mlp_down.Data, d_mlp_down.Data ? norm_of(d_mlp_down) : 0.0f,
                             ln_layer_idx % 2);
            }
        }

        if (d_mlp_down.Data && d_mlp_down.Data != d_residual.Data) {
            CUDA_CHECK(cudaMemcpyAsync(d_mlp_down.Data, d_residual.Data, d_residual.bytes(),
                                       cudaMemcpyDeviceToDevice, mRunState.MainStream));
        }
        if (env_enabled("SUROGATE_DSL_DEBUG_LN_BWD")) {
            const char* ln_layer_env = std::getenv("SUROGATE_DSL_DEBUG_LN_LAYER");
            const int ln_layer_filter = ln_layer_env ? std::atoi(ln_layer_env) : -999999;
            if (!ln_layer_env || ln_layer_idx == ln_layer_filter) {
                static Tensor debug_ln_out_buf{};
                if (!debug_ln_out_buf.Data && mRunState.Allocator) {
                    const int num_sums = get_max_num_block_sums(mRunState.DeviceProp);
                    debug_ln_out_buf = mRunState.Allocator->allocate(ETensorDType::FP32,
                                                                     "debug_ln_bwd_out_norm",
                                                                     EAllocationType::ON_DEVICE,
                                                                     {num_sums});
                }
                if (debug_ln_out_buf.Data) {
                    auto norm_of = [&](const Tensor& t) {
                        fill_zero(debug_ln_out_buf, mRunState.MainStream);
                        global_norm_squared(debug_ln_out_buf, t, t.nelem(), mRunState.DeviceProp, mRunState.MainStream);
                        deterministic_sum(debug_ln_out_buf.template get<float>(),
                                          debug_ln_out_buf.template get<float>(),
                                          debug_ln_out_buf.nelem(),
                                          mRunState.MainStream);
                        float host_sum = 0.0f;
                        CUDA_CHECK(cudaMemcpyAsync(&host_sum,
                                                   debug_ln_out_buf.template get<float>(),
                                                   sizeof(float),
                                                   cudaMemcpyDeviceToHost,
                                                   mRunState.MainStream));
                        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
                        return std::sqrt(host_sum);
                    };
                    std::fprintf(stderr,
                                 "[DSL DEBUG LN BWD] ln2 out0 norm=%g d_mlp_down norm=%g\n",
                                 norm_of(d_residual),
                                 d_mlp_down.Data ? norm_of(d_mlp_down) : 0.0f);
                }
            }
        }
    }

    // LN1 backward writes grad for previous layer's residual stream into d_mlp_down;
    // mirror it into that layer's d_res_ffn so gradient propagation matches modular.
    if (op.outputs.size() > 1 && op.outputs[1].slot == TensorSlot::BlockDMLPDown) {
        const int prev_layer = op.outputs[1].layer_idx;
        if (prev_layer >= 0) {
            Tensor& d_res_ffn_prev = mRunState.simplified_grads(prev_layer).d_res_ffn;
            if (d_res_ffn_prev.Data && d_res_ffn_prev.Data != d_input.Data) {
                CUDA_CHECK(cudaMemcpyAsync(d_res_ffn_prev.Data, d_input.Data, d_input.bytes(),
                                           cudaMemcpyDeviceToDevice, mRunState.MainStream));
            }
            if (env_enabled("SUROGATE_DSL_DEBUG_LN_BWD")) {
                static int debug_ln_copy_count = 0;
                if (debug_ln_copy_count < 8) {
                    std::fprintf(stderr,
                                 "[DSL DEBUG LN BWD] copy to d_res_ffn_prev layer=%d ptr=%p\n",
                                 prev_layer,
                                 d_res_ffn_prev.Data);
                }
                ++debug_ln_copy_count;
            }
        }
    }
}

void CompiledExecutor::dispatch_embedding_backward(const CompiledOp& op) {
    // Skip embedding backward entirely in LoRA-only mode
    if (mRunState.is_lora_only_mode()) {
        return;
    }

    // inputs: d_encoded, token_ids
    // outputs: d_embedding (sparse update)
    Tensor& d_out = resolve_tensor(op.inputs[0]);

    if (op.outputs.empty() || op.outputs[0].name.empty()) {
        return;  // Skip if no output expected
    }

    // Get the pre-allocated gradient tensor
    auto it = mTensorMap.find(op.outputs[0].name);
    if (it == mTensorMap.end()) {
        // Gradient not allocated (embedding frozen in LoRA mode)
        return;
    }
    Tensor& d_emb = it->second;

    // encoder_backward requires CPU-side inputs for deterministic bucketing
    if (!mLastInputsCpu || !mLastInputsCpu->Data) {
        throw std::runtime_error("CompiledExecutor: embedding_backward requires CPU inputs (set_last_inputs_cpu)");
    }

    unsigned int seed = mRngSeedFn ? mRngSeedFn() : 0;

    encoder_backward(d_emb,
                     mRunState.scratch().encoder_bwd_scratch,
                     mRunState.scratch().encoder_bwd_indices,
                     mRunState.scratch().encoder_bwd_info,
                     d_out,
                     mRunState.Inputs,
                     *mLastInputsCpu,
                     static_cast<int>(mB), static_cast<int>(mT), mConfig.HiddenSize,
                     seed,
                     mRunState.MainStream,
                     mRunState.side_stream_event(),
                     mRunState.side_stream());
}

void CompiledExecutor::dispatch_cross_entropy_loss_backward(const CompiledOp& op) {
    Tensor& d_loss = resolve_tensor(op.inputs[0]);
    Tensor& logits = resolve_tensor(op.inputs[1]);
    Tensor& targets = resolve_tensor(op.inputs[2]);
    Tensor& d_logits = ensure_output_tensor(op.outputs[0]);

    const int BT = static_cast<int>(logits.Sizes[0]);
    const int V = static_cast<int>(logits.Sizes[1]);
    const int P = V;

    if (op.inputs[0].slot == TensorSlot::DLoss) {
        const long total_tokens = mB * mT;
        const int accum_steps = std::max(1, mRunState.GradAccumSteps);
        const float dloss_val = 1.0f / static_cast<float>(total_tokens * accum_steps);
        fill_constant(d_loss, dloss_val, static_cast<std::size_t>(d_loss.nelem()), mRunState.MainStream);
    }

    Tensor logsumexp_view{};
    Tensor* logsumexp = nullptr;
    if (mRunState.scratch().cross_entropy_logsumexp.Data) {
        logsumexp_view = mRunState.scratch().cross_entropy_logsumexp;
        logsumexp_view.Sizes[0] = BT;
        logsumexp_view.Rank = 1;
        logsumexp = &logsumexp_view;
    }

    if (V > CROSS_ENTROPY_MAX_FUSED_SIZE) {
        chunked_cross_entropy_backward(d_logits, logits, logsumexp, d_loss, targets,
                                       BT, V, P, mRunState.MainStream);
    } else {
        fused_cross_entropy_backward(d_logits, logits, logsumexp, d_loss, targets,
                                     BT, V, P, mRunState.MainStream);
    }
}

void CompiledExecutor::dispatch_fused_lm_head_loss_backward(const CompiledOp& op) {
    Tensor& d_loss = resolve_tensor(op.inputs[0]);
    Tensor& xF_flat = resolve_tensor(op.inputs[1]);
    Tensor& weight = resolve_tensor(op.inputs[2]);
    Tensor& targets = resolve_tensor(op.inputs[3]);

    Tensor* d_xF_ptr = nullptr;
    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        d_xF_ptr = &ensure_output_tensor(op.outputs[0]);
    }

    Tensor* d_weight_ptr = nullptr;
    bool d_weight_accumulate = false;
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        std::string weight_name = op.outputs[1].name;
        if (auto base = base_param_from_grad(weight_name)) {
            weight_name = *base;
        } else if (weight_name.rfind("d_", 0) == 0) {
            weight_name = weight_name.substr(2);
        }
        bool accum = false;
        Tensor* grad = mGrads.get_param_grad(weight_name, accum);
        d_weight_accumulate = mAccumulateTensors.count(op.outputs[1].name) > 0;
        if (!d_weight_accumulate) {
            if (auto base = base_param_from_grad(op.outputs[1].name)) {
                d_weight_accumulate = mAccumulateTensors.count("d_" + *base) > 0;
            }
        }
        if (grad && grad->Data) {
            d_weight_ptr = &ensure_output_tensor(op.outputs[1]);
        }
    }

    if (op.inputs[0].slot == TensorSlot::DLoss) {
        const long total_tokens = mB * mT;
        const int accum_steps = std::max(1, mRunState.GradAccumSteps);
        const float dloss_val = 1.0f / static_cast<float>(total_tokens * accum_steps);
        fill_constant(d_loss, dloss_val, static_cast<std::size_t>(d_loss.nelem()), mRunState.MainStream);
    }

    const long BT = xF_flat.Sizes[0];
    const int V = static_cast<int>(weight.Sizes[0]);
    const int C = static_cast<int>(weight.Sizes[1]);
    const int P = V;

    const int lmhead_chunks = std::max(1, mOptions.LMHeadChunks);
    const long nano_batch_size = BT / static_cast<long>(lmhead_chunks);
    const bool need_lm_head =
        mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled()) &&
        (mOptions.LMHeadChunks > 1);
    if (need_lm_head && mComm) {
        mWeightManager->gather_lm_head(*mComm, mRunState.MainStream);
    }

    const std::size_t xf_stride = get_dtype_size(xF_flat.DType);
    const std::size_t tgt_stride = get_dtype_size(targets.DType);
    const std::size_t lse_stride = get_dtype_size(ETensorDType::FP32);
    const std::size_t dloss_stride = get_dtype_size(d_loss.DType);

    for (int nano_step = 0; nano_step < lmhead_chunks; ++nano_step) {
        const long token_offset = static_cast<long>(nano_step) * nano_batch_size;

        Tensor xF_slice = xF_flat;
        xF_slice.Data = static_cast<std::byte*>(xF_slice.Data) +
                        static_cast<std::size_t>(token_offset) * xf_stride * static_cast<std::size_t>(C);
        xF_slice.Sizes[0] = nano_batch_size;
        xF_slice.Sizes[1] = C;
        xF_slice.Rank = 2;

        Tensor tgt_slice = targets;
        tgt_slice.Data = static_cast<std::byte*>(tgt_slice.Data) +
                         static_cast<std::size_t>(token_offset) * tgt_stride;
        tgt_slice.Sizes[0] = nano_batch_size;
        tgt_slice.Rank = 1;

        Tensor dloss_slice = d_loss;
        dloss_slice.Data = static_cast<std::byte*>(dloss_slice.Data) +
                           static_cast<std::size_t>(token_offset) * dloss_stride;
        dloss_slice.Sizes[0] = nano_batch_size;
        dloss_slice.Rank = 1;

        Tensor logits = mRunState.non_block_activations().output;
        logits.Sizes[0] = nano_batch_size;
        logits.Sizes[1] = V;
        logits.Rank = 2;

        matmul(logits, weight, xF_slice,
               std::nullopt, nullptr, nullptr, mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
               static_cast<int>(V), static_cast<int>(nano_batch_size), static_cast<int>(C),
               swap_transpose(EMMTranspose::NT), false, mRunState.MainStream);

        Tensor logsumexp_view{};
        Tensor* logsumexp = nullptr;
        if (mRunState.scratch().cross_entropy_logsumexp.Data) {
            logsumexp_view = mRunState.scratch().cross_entropy_logsumexp;
            logsumexp_view.Data = static_cast<std::byte*>(logsumexp_view.Data) +
                                  static_cast<std::size_t>(token_offset) * lse_stride;
            logsumexp_view.Sizes[0] = nano_batch_size;
            logsumexp_view.Rank = 1;
            logsumexp = &logsumexp_view;
        }

        if (V > CROSS_ENTROPY_MAX_FUSED_SIZE) {
            chunked_cross_entropy_backward(logits, logits, logsumexp, dloss_slice, tgt_slice,
                                           static_cast<int>(nano_batch_size), V, P, mRunState.MainStream);
        } else {
            fused_cross_entropy_backward(logits, logits, logsumexp, dloss_slice, tgt_slice,
                                         static_cast<int>(nano_batch_size), V, P, mRunState.MainStream);
        }

        if (d_weight_ptr) {
            const bool accumulate = d_weight_accumulate || (nano_step != 0);
            matmul(*d_weight_ptr, xF_slice, logits,
                   std::nullopt, nullptr, nullptr, mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
                   static_cast<int>(C), static_cast<int>(V), static_cast<int>(nano_batch_size),
                   swap_transpose(EMMTranspose::TN), accumulate, mRunState.MainStream);
        }

        if (d_xF_ptr) {
            Tensor d_xF_slice = *d_xF_ptr;
            const std::size_t dx_stride = get_dtype_size(d_xF_slice.DType);
            d_xF_slice.Data = static_cast<std::byte*>(d_xF_slice.Data) +
                              static_cast<std::size_t>(token_offset) * dx_stride * static_cast<std::size_t>(C);
            d_xF_slice.Sizes[0] = nano_batch_size;
            d_xF_slice.Sizes[1] = C;
            d_xF_slice.Rank = 2;

            matmul(d_xF_slice, weight, logits,
                   std::nullopt, nullptr, nullptr, mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
                   static_cast<int>(C), static_cast<int>(nano_batch_size), static_cast<int>(V),
                   swap_transpose(EMMTranspose::NN), false, mRunState.MainStream);
        }
    }

    if (need_lm_head) {
        mWeightManager->release_lm_head(mRunState.MainStream);
    }
}

void CompiledExecutor::execute_forward(const CompiledGraph& graph,
                                       NCCLCommunicator& comm,
                                       bool full,
                                       const modules::ForwardHook* hook) {
    mComm = &comm;
    mTemps.clear();
    mTensorMap.clear();
    mCurrentLayer = -1;

    // Match GraphExecutor behavior: initialize loss/counter buffers for full forward runs.
    // This avoids stale accumulation when tests call CompiledExecutor directly.
    if (full) {
        bool has_loss_op = false;
        for (const auto& op : graph.ops) {
            if (op.type == CompiledOpType::CrossEntropyLoss ||
                op.type == CompiledOpType::FusedLMHeadLoss) {
                has_loss_op = true;
                break;
            }
        }
        if (has_loss_op) {
            fill_zero(mRunState.Losses, mRunState.MainStream);
            fill_zero(mRunState.ValidTokenCount, mRunState.MainStream);
            fill_zero(mRunState.CorrectCount, mRunState.MainStream);
        }
    }
    const int num_layers = static_cast<int>(mConfig.NumLayers);
    std::vector<DeviceMemoryStack::Checkpoint> layer_checkpoints;
    std::vector<std::size_t> layer_temp_marks;
    std::vector<char> layer_active;
    if (num_layers > 0) {
        layer_checkpoints.resize(static_cast<std::size_t>(num_layers));
        layer_temp_marks.resize(static_cast<std::size_t>(num_layers));
        layer_active.assign(static_cast<std::size_t>(num_layers), 0);
    }
    auto prune_stack_tensors = [&]() {
        for (auto it = mTensorMap.begin(); it != mTensorMap.end(); ) {
            // Skip tensors that are needed for backward (in save list)
            if (mSaveSet.count(it->first) > 0) {
                ++it;
                continue;
            }
            if (it->second.Data && mRunState.Stack.owns(it->second.Data) &&
                !mRunState.Stack.is_live(it->second.Data)) {
                it = mTensorMap.erase(it);
            } else {
                ++it;
            }
        }
    };

    // Bind known inputs
    mTensorMap["token_ids"] = mRunState.Inputs;
    mTensorMap["position_ids"] = mRunState.PositionIDs;
    mTensorMap["x0"] = mRunState.non_block_activations().encoded;

    // Ensure non-block weights are gathered if streaming/offload is enabled
    if (mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled())) {
        mWeightManager->gather_embeddings(comm, mRunState.MainStream);
        mWeightManager->gather_final_norm(comm, mRunState.MainStream);
    }

    // Prefetch layer 0 before loop
    if (mConfig.NumLayers > 0 && !mCapturing) {
        if (mWeightManager && mWeightManager->is_streaming_enabled()) {
            mWeightManager->gather_block(0, comm, mRunState.side_stream());
        }
    }

    // Main dispatch loop - no string comparisons, direct function pointer dispatch
    for (std::size_t idx = 0; idx < graph.ops.size(); ++idx) {
        if (!full && !graph.required_mask.empty() && !graph.required_mask[idx]) {
            continue;
        }

        const auto& op = graph.ops[idx];

        // Handle layer boundaries
        if (op.layer_start >= 0) {
            if (op.layer_start < num_layers &&
                !layer_active[static_cast<std::size_t>(op.layer_start)]) {
                layer_checkpoints[static_cast<std::size_t>(op.layer_start)] = mRunState.Stack.checkpoint();
                layer_temp_marks[static_cast<std::size_t>(op.layer_start)] = mTemps.size();
                layer_active[static_cast<std::size_t>(op.layer_start)] = 1;
            }
            handle_layer_start(op.layer_start);
        }

        try {
            // Direct dispatch via switch (branch predictor friendly, no string compare)
            switch (op.type) {
                case CompiledOpType::Embedding:
                    dispatch_embedding(op);
                    break;
                case CompiledOpType::Zeros:
                    dispatch_zeros(op);
                    break;
                case CompiledOpType::FusedResidualRMSNorm:
                    dispatch_fused_residual_rmsnorm(op);
                    break;
                case CompiledOpType::View:
                    dispatch_view(op);
                    break;
                case CompiledOpType::Add:
                    dispatch_add(op);
                    break;
                case CompiledOpType::Matmul:
                case CompiledOpType::MatmulBias:
                    dispatch_matmul(op, hook);
                    break;
                case CompiledOpType::BiasAdd:
                    dispatch_bias_add(op);
                    break;
                case CompiledOpType::SwiGLU:
                    dispatch_swiglu(op);
                    break;
                case CompiledOpType::MatmulSwiGLU:
                    dispatch_matmul_swiglu(op);
                    break;
                case CompiledOpType::QKVQKNormRoPE:
                    dispatch_qkv_qk_norm_rope(op);
                    break;
                case CompiledOpType::RoPE:
                    dispatch_rope(op);
                    break;
                case CompiledOpType::FlashAttention:
                    dispatch_flash_attention(op);
                    break;
                case CompiledOpType::CrossEntropyLoss:
                    dispatch_cross_entropy_loss(op);
                    break;
                case CompiledOpType::FusedLMHeadLoss:
                    dispatch_fused_lm_head_loss(op);
                    break;
                default:
                    throw std::runtime_error("CompiledExecutor: unsupported forward op type");
            }
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "CompiledExecutor forward op " << idx << " (type=" << op_type_to_string(op.type)
                << ", id=" << op.op_id << "): " << e.what();
            throw std::runtime_error(oss.str());
        }

        // Handle layer end
        if (op.layer_end >= 0) {
            // Note: Forward activation stats are not printed because with recompute_block=true,
            // the activation buffers are shared across layers, so they only contain the last
            // layer's data at this point, not the per-layer values.
            if (op.layer_end < num_layers &&
                layer_active[static_cast<std::size_t>(op.layer_end)]) {
                mRunState.Stack.restore(layer_checkpoints[static_cast<std::size_t>(op.layer_end)]);
                if (mTemps.size() > layer_temp_marks[static_cast<std::size_t>(op.layer_end)]) {
                    mTemps.resize(layer_temp_marks[static_cast<std::size_t>(op.layer_end)]);
                }
                prune_stack_tensors();
                if (mRunState.ffn_temps_on_stack()) {
                    auto& acts = mRunState.simplified_acts(op.layer_end);
                    acts.mlp_up.Data = nullptr;
                    acts.swiglu.Data = nullptr;
                }
                // Note: cudnn_workspace is persistently allocated, don't clear
                layer_active[static_cast<std::size_t>(op.layer_end)] = 0;
            }
            handle_layer_end(op.layer_end);
        }
    }

    // Free temporaries
    for (auto it = mTemps.rbegin(); it != mTemps.rend(); ++it) {
        mRunState.temp_free(*it);
    }
    mTemps.clear();

    if (mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled())) {
        mWeightManager->release_embeddings(mRunState.MainStream);
        mWeightManager->release_final_norm(mRunState.MainStream);
    }
}

void CompiledExecutor::execute_backward(const CompiledGraph& graph,
                                        NCCLCommunicator& comm,
                                        int grad_accum_steps,
                                        int micro_step,
                                        const modules::BackwardHook* hook) {
    mComm = &comm;
    mRunState.reset_simplified_gradients();
    mTemps.clear();
    mTensorMap.clear();
    mAccumulateTensors.clear();
    mCurrentLayer = -1;
    mLastRecomputeLayer = -1;

    if (mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled())) {
        mWeightManager->gather_final_norm(comm, mRunState.MainStream);
        if (mOptions.LMHeadChunks <= 1) {
            mWeightManager->gather_lm_head(comm, mRunState.MainStream);
        }
    }

    // Save stack checkpoint at start of backward - we'll restore per-layer to manage memory
    auto initial_checkpoint = mRunState.Stack.checkpoint();
    int last_layer_restored = -1;
    auto clear_shared_grads = [&](int layer_idx) {
        if (!mRunState.large_bwd_temps_on_stack()) {
            return;
        }
        if (layer_idx < 0 || layer_idx >= static_cast<int>(mConfig.NumLayers)) {
            return;
        }
        auto& grads = mRunState.simplified_grads(layer_idx);
        if (grads.d_ln2.Data) {
            fill_zero(grads.d_ln2, mRunState.MainStream);
        }
        if (grads.d_att.Data) {
            fill_zero(grads.d_att, mRunState.MainStream);
        }
        if (grads.d_ln1.Data) {
            fill_zero(grads.d_ln1, mRunState.MainStream);
        }
    };
    auto prune_stack_tensors = [&]() {
        for (auto it = mTensorMap.begin(); it != mTensorMap.end(); ) {
            if (it->second.Data && mRunState.Stack.owns(it->second.Data) &&
                !mRunState.Stack.is_live(it->second.Data)) {
                it = mTensorMap.erase(it);
            } else {
                ++it;
            }
        }
    };

    // Bind initial gradient tensors (from loss computation)
    // d_logits is stored in the output buffer after loss backward (only when lmhead_chunks == 1)
    auto& output = mRunState.non_block_activations().output;
    if (!output.Data) {
        throw std::runtime_error("CompiledExecutor: output tensor has no data (B=" +
                                std::to_string(mB) + ", T=" + std::to_string(mT) + ")");
    }

    if (mOptions.LMHeadChunks <= 1) {
        Tensor logits_view = view_tensor(output, {mB, mT, static_cast<long>(mConfig.VocabSize)});
        mTensorMap["d_logits"] = logits_view;
        // Also provide flattened version for matmul backward ops
        Tensor logits_flat = view_tensor(output, {mB * mT, static_cast<long>(mConfig.VocabSize)});
        if (logits_flat.Rank != 2) {
            throw std::runtime_error("CompiledExecutor: d_logits_flat has wrong rank=" +
                                    std::to_string(logits_flat.Rank) + " expected 2");
        }
        mTensorMap["d_logits_flat"] = logits_flat;
        // Verify the map entry
        auto& check = mTensorMap["d_logits_flat"];
        if (check.Rank != 2) {
            throw std::runtime_error("CompiledExecutor: d_logits_flat in map has wrong rank=" +
                                    std::to_string(check.Rank));
        }
    }

    // Bind gradient output buffers for final layer norm backward
    Tensor d_ln_final_flat = view_tensor(mRunState.non_block_gradients().d_ln_final,
                                         {mB * mT, static_cast<long>(mConfig.HiddenSize)});
    mTensorMap["d_xF_flat"] = d_ln_final_flat;
    mTensorMap["d_xF"] = mRunState.non_block_gradients().d_ln_final;
    mTensorMap["d_ln_final"] = mRunState.non_block_gradients().d_ln_final;

    // Bind embedding output gradients (skip in LoRA-only mode - embedding backward is skipped entirely)
    if (!mRunState.is_lora_only_mode()) {
        mTensorMap["d_encoded"] = mRunState.non_block_gradients().d_embeddings;
        mTensorMap["d_x0"] = mRunState.non_block_gradients().d_embeddings;

        // Also bind autodiff-generated gradient names (d_embed_1, etc.) from forward embedding outputs
        for (const auto& emb_out : mEmbeddingOutputs) {
            std::string grad_name = "d_" + emb_out;
            mTensorMap[grad_name] = mRunState.non_block_gradients().d_embeddings;
        }

    }

    // Also bind standard inputs that backward ops may reference
    mTensorMap["token_ids"] = mRunState.Inputs;
    mTensorMap["position_ids"] = mRunState.PositionIDs;

    // Build the set of gradients that require accumulation (not the first micro-step).
    // Also bind parameter gradient tensors to mTensorMap so they're used instead of temporaries.
    // This mirrors the logic in graph_executor_backward.cpp (bind_param_grad).
    for (const auto& param_name : mGrads.param_names()) {
        if (param_name.find("rope_freqs") != std::string::npos) {
            continue;
        }
        bool accumulate = false;
        Tensor* grad_tensor = mGrads.get_param_grad(param_name, accumulate);
        if (grad_tensor && grad_tensor->Data) {
            std::string grad_name = "d_" + param_name;
            mTensorMap[grad_name] = *grad_tensor;
            if (accumulate) {
                mAccumulateTensors.insert(grad_name);
            }
        }
    }

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

    auto op_layer_idx = [&](const CompiledOp& op) -> int {
        int detected_non_grad = -1;
        for (const auto& ref : op.inputs) {
            if (!is_grad_ref(ref)) {
                const int layer_idx = ref_layer_idx(ref);
                if (layer_idx >= 0) {
                    detected_non_grad = std::max(detected_non_grad, layer_idx);
                }
            }
        }
        for (const auto& ref : op.outputs) {
            if (!is_grad_ref(ref)) {
                const int layer_idx = ref_layer_idx(ref);
                if (layer_idx >= 0) {
                    detected_non_grad = std::max(detected_non_grad, layer_idx);
                }
            }
        }
        return (detected_non_grad >= 0) ? detected_non_grad : -1;
    };

    auto op_layer_idx_any = [&](const CompiledOp& op) -> int {
        int detected_any = -1;
        for (const auto& ref : op.inputs) {
            const int layer_idx = ref_layer_idx_any(ref);
            if (layer_idx >= 0) {
                detected_any = std::max(detected_any, layer_idx);
            }
        }
        for (const auto& ref : op.outputs) {
            const int layer_idx = ref_layer_idx_any(ref);
            if (layer_idx >= 0) {
                detected_any = std::max(detected_any, layer_idx);
            }
        }
        if (op.attrs.layer_idx >= 0) {
            detected_any = std::max(detected_any, op.attrs.layer_idx);
        }
        return detected_any;
    };

    const bool skip_logits_grad = (mOptions.LMHeadChunks > 1);
    auto is_logits_grad_name = [](const std::string& name) {
        return name == "d_logits" || name == "d_logits_flat";
    };
    auto is_logits_grad_op = [&](const CompiledOp& op) {
        for (const auto& ref : op.inputs) {
            if (is_logits_grad_name(ref.name)) return true;
        }
        for (const auto& ref : op.outputs) {
            if (is_logits_grad_name(ref.name)) return true;
        }
        return false;
    };

    const int num_layers = static_cast<int>(mConfig.NumLayers);
    std::vector<std::size_t> layer_start_indices(num_layers, SIZE_MAX);
    std::vector<bool> layer_seen_any(num_layers, false);
    for (const auto& op : graph.ops) {
        if (op.layer_start >= 0 && op.layer_start < num_layers) {
            layer_start_indices[op.layer_start] = &op - graph.ops.data();
        }
    }

    // Debug logging for recompute diagnosis
    static bool debug_recompute = env_enabled("SUROGATE_DEBUG_RECOMPUTE");
    static int debug_step = 0;
    int prev_layer_idx_debug = -1;
    int layer_switches = 0;

    if (debug_recompute) {
        std::fprintf(stderr, "\n[RECOMPUTE DEBUG] === Backward pass step %d ===\n", debug_step);
        std::fprintf(stderr, "[RECOMPUTE DEBUG] recompute_enabled=%d, num_ops=%zu, num_layers=%d\n",
                     mRecomputeEnabled ? 1 : 0, graph.ops.size(), num_layers);
        ++debug_step;
    }

    for (std::size_t idx = 0; idx < graph.ops.size(); ++idx) {
        const auto& op = graph.ops[idx];
        const int op_layer_any = op_layer_idx_any(op);

        if (skip_logits_grad && is_logits_grad_op(op)) {
            continue;
        }

        if (op.layer_start >= 0) {
            handle_layer_start(op.layer_start);
            if (mRecomputeEnabled && mRecomputeFn) {
                const int layer_idx = op.layer_start;
                if (layer_idx >= 0 && layer_idx != mLastRecomputeLayer) {
                    if (debug_recompute) {
                        std::fprintf(stderr, "[RECOMPUTE DEBUG] op[%zu] layer_start=%d triggers recompute (prev=%d)\n",
                                     idx, layer_idx, mLastRecomputeLayer);
                    }
                    if (layer_idx < num_layers && !layer_seen_any[static_cast<std::size_t>(layer_idx)]) {
                        clear_shared_grads(layer_idx);
                        layer_seen_any[static_cast<std::size_t>(layer_idx)] = true;
                    }
                    mRecomputeFn(layer_idx, mB, mT, mRecomputeUseGraphs);
                    mLastRecomputeLayer = layer_idx;
                }
            }
        }

        if (mRecomputeEnabled && mRecomputeFn) {
            const int layer_idx = op_layer_idx(op);
            // Always recompute when switching layers. This is critical because:
            // - Shared buffers (ln1, ln2, qkv, mlp_up, swiglu) contain only ONE layer's data
            // - If the backward graph interleaves ops from different layers, we MUST
            //   recompute to ensure the correct layer's data is in the shared buffers
            // - The old check (missing_start || op_before_start) would skip recomputation
            //   for layer N's late ops if we had already visited layer N earlier, causing
            //   those ops to read stale data from whatever layer was recomputed last
            if (layer_idx >= 0 && layer_idx != mLastRecomputeLayer) {
                if (debug_recompute) {
                    std::fprintf(stderr, "[RECOMPUTE DEBUG] op[%zu] %s (op_layer=%d) triggers recompute (prev=%d) [INTERLEAVE]\n",
                                 idx, op_type_to_string(op.type), layer_idx, mLastRecomputeLayer);
                    ++layer_switches;
                }
                if (layer_idx < num_layers && !layer_seen_any[static_cast<std::size_t>(layer_idx)]) {
                    clear_shared_grads(layer_idx);
                    layer_seen_any[static_cast<std::size_t>(layer_idx)] = true;
                }
                mRecomputeFn(layer_idx, mB, mT, mRecomputeUseGraphs);
                mLastRecomputeLayer = layer_idx;
            }
        }

        // Debug: log each op's layer assignment
        if (debug_recompute) {
            const int layer_any = op_layer_idx_any(op);
            const int layer_non_grad = op_layer_idx(op);
            if (layer_any != prev_layer_idx_debug || op.layer_start >= 0 || op.layer_end >= 0) {
                std::fprintf(stderr, "[RECOMPUTE DEBUG] op[%zu] %s layer_any=%d layer_non_grad=%d layer_start=%d layer_end=%d id=%s\n",
                             idx, op_type_to_string(op.type), layer_any, layer_non_grad,
                             op.layer_start, op.layer_end, op.op_id.c_str());
                prev_layer_idx_debug = layer_any;
            }
        }

        // Debug: trace operation order and layers for gradient tracking
        static bool debug_grad_norms = env_enabled("SUROGATE_DEBUG_GRAD_NORMS");
        if (debug_grad_norms) {
            const int layer_idx = op_layer_idx_any(op);
            std::fprintf(stderr, "[OP TRACE] idx=%zu type=%s layer=%d id=%s\n",
                         idx, op_type_to_string(op.type), layer_idx, op.op_id.c_str());
        }

        try {
            switch (op.type) {
                // Explicit backward ops
                case CompiledOpType::ViewBackward:
                    dispatch_view_backward(op);
                    break;
                case CompiledOpType::AddBackward:
                    dispatch_add_backward(op);
                    break;
                case CompiledOpType::CrossEntropyLossBackward:
                    dispatch_cross_entropy_loss_backward(op);
                    break;
                case CompiledOpType::FusedLMHeadLossBackward:
                    dispatch_fused_lm_head_loss_backward(op);
                    break;
                case CompiledOpType::MatmulBackward:
                    dispatch_matmul_backward(op, hook);
                    // After the first matmul_backward (LM-head backward), free the output tensor
                    // to reclaim ~1.2GB of stack memory. The d_logits data has been consumed.
                    if (idx == 1) {
                        mRunState.temp_free(mRunState.non_block_activations().output);
                        mTemps.clear();
                        // Update initial_checkpoint to reflect the freed output tensor
                        // This prevents subsequent checkpoint restores from re-allocating it
                        initial_checkpoint = mRunState.Stack.checkpoint();
                    }
                    break;
                case CompiledOpType::BiasAddBackward:
                    dispatch_bias_add_backward(op);
                    break;
                case CompiledOpType::SwiGLUBackward:
                    dispatch_swiglu_backward(op);
                    break;
                case CompiledOpType::MatmulSwiGLUBackward:
                    dispatch_matmul_swiglu_backward(op, hook);
                    break;
                case CompiledOpType::RoPEBackward:
                    dispatch_rope_backward(op);
                    break;
                case CompiledOpType::QKVQKNormRoPEBackward:
                    dispatch_qkv_qk_norm_rope_backward(op);
                    break;
                case CompiledOpType::FlashAttentionBackward:
                    dispatch_flash_attention_backward(op);
                    break;
                case CompiledOpType::ZerosBackward:
                    dispatch_zeros_backward(op);
                    break;
                case CompiledOpType::FusedResidualRMSNormBackward:
                    dispatch_fused_residual_rmsnorm_backward(op);
                    break;
                case CompiledOpType::EmbeddingBackward:
                    dispatch_embedding_backward(op);
                    break;

                // Forward ops that appear in backward graph (autodiff generates these)
                // View/reshape is the same operation in forward and backward - just reshapes gradient
                case CompiledOpType::View:
                    dispatch_view_backward(op);
                    break;
                // "add" ops in the backward graph are gradient-accumulation nodes,
                // so we must execute them as forward add (sum inputs), not add-backward.
                case CompiledOpType::Add:
                    dispatch_add(op);
                    break;
                // Zeros in backward is a no-op
                case CompiledOpType::Zeros:
                    dispatch_zeros_backward(op);
                    break;

                default: {
                    std::ostringstream oss;
                    oss << "CompiledExecutor: unsupported backward op type at idx " << idx
                        << " (type=" << op_type_to_string(op.type) << ", id=" << op.op_id << ")";
                    throw std::runtime_error(oss.str());
                }
            }

            // Memory management - restore stack checkpoint periodically to free temporaries
            // This prevents memory accumulation during backward pass
            // Option 1: At layer boundaries if annotated
            if (op.layer_end >= 0 && op.layer_end != last_layer_restored) {
                // Debug: log layer completion
                static bool debug_fft_layers = env_enabled("SUROGATE_DEBUG_FFT_RECOMPUTE");
                if (debug_fft_layers) {
                    std::fprintf(stderr, "[FFT BWD] Layer %d backward complete, restoring stack checkpoint\n",
                                 op.layer_end);
                }

                // Restore stack and clear temps
                mRunState.Stack.restore(initial_checkpoint);
                mTemps.clear();
                prune_stack_tensors();
                // Note: cudnn_workspace is persistently allocated, no need to clear
                // Clear stack-allocated tensor pointers in simplified_acts/grads for this layer.
                // These pointers become stale after checkpoint restore.
                if (mRunState.ffn_temps_on_stack()) {
                    auto& acts = mRunState.simplified_acts(op.layer_end);
                    acts.mlp_up.Data = nullptr;
                    acts.swiglu.Data = nullptr;
                }
                if (mRunState.large_bwd_temps_on_stack()) {
                    auto& grads_to_clear = mRunState.simplified_grads(op.layer_end);
                    grads_to_clear.d_qkv.Data = nullptr;
                    grads_to_clear.d_mlp_up.Data = nullptr;
                    grads_to_clear.d_swiglu.Data = nullptr;
                }
                last_layer_restored = op.layer_end;
            }
            // Option 2: Every N ops as fallback (catches non-annotated layers)
            // NOTE: When recompute is disabled, we cannot aggressively prune tensors because
            // the backward graph may reference intermediate tensors (like d_blocks[N].view_K)
            // that were produced earlier but are still needed. The stack restore + prune
            // would remove these tensors from mTensorMap, causing "tensor not found" errors.
            // For now, skip periodic cleanup when recompute is disabled to preserve correctness.
            // Memory usage will be higher but the backward pass will complete successfully.
            // TODO: Implement proper tensor lifetime tracking to enable safe pruning.
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "CompiledExecutor backward op " << idx << " (type=" << op_type_to_string(op.type)
                << ", id=" << op.op_id << "): " << e.what();
            // Add inputs/outputs for debugging
            oss << "\n  inputs: [";
            for (size_t i = 0; i < op.inputs.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << op.inputs[i].name << "(slot=" << static_cast<int>(op.inputs[i].slot) << ")";
            }
            oss << "]";
            oss << "\n  outputs: [";
            for (size_t i = 0; i < op.outputs.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << op.outputs[i].name << "(slot=" << static_cast<int>(op.outputs[i].slot) << ")";
            }
            oss << "]";
            throw std::runtime_error(oss.str());
        }

    }

    // Debug summary
    if (debug_recompute) {
        std::fprintf(stderr, "[RECOMPUTE DEBUG] === Backward pass complete: %d interleaved layer switches ===\n\n",
                     layer_switches);
    }

    // Final cleanup
    mRunState.Stack.restore(initial_checkpoint);
    prune_stack_tensors();
    mTemps.clear();

    if (mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled())) {
        mWeightManager->release_final_norm(mRunState.MainStream);
        if (mOptions.LMHeadChunks <= 1) {
            mWeightManager->release_lm_head(mRunState.MainStream);
        }
    }
}

}  // namespace dsl
