// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL model wrapper (IR validation + execution).

#include "runtime/dsl/dsl_model.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "runtime/dsl/graph_executor.h"
#include "runtime/dsl/dsl_runtime.h"
#include "runtime/core/qlora_provider.h"
#include "runtime/dsl/dsl_weight_manager.h"
#include "runtime/dsl/dsl_model_internal.h"
#include "kernels/kernels.h"
#include "runtime/core/forward_hooks.h"
#include "runtime/core/backward_hooks.h"
#include "runtime/lora/lora_utils.h"
#include "runtime/lora/lora_model_utils.h"
#include "runtime/qlora/fp8_weight_provider.h"
#include "runtime/qlora/fp4_weight_provider.h"
#include "runtime/qlora/bnb_weight_provider.h"
#include "runtime/qlora/hf_mapping.h"
#include "runtime/qlora/dsl_block_weights.h"
#include "runtime/core/model_config.h"
#include "runtime/optimizers/adamw_8bit.h"
#include "runtime/optimizers/normuon.h"
#include "runtime/core/fp8_scaling_state.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"
#include "utilities/safetensors.h"

namespace dsl {
namespace {

std::string_view trim_optional(std::string_view name) {
    if (!name.empty() && name.back() == '?') {
        return name.substr(0, name.size() - 1);
    }
    return name;
}

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool graph_has_kernel(const Module& module, std::string_view kernel) {
    if (!module.forward.has_value()) {
        return false;
    }
    const auto& ops = module.forward->operations;
    for (const auto& op : ops) {
        if (op.kernel_type == kernel || op.name == kernel) {
            return true;
        }
    }
    return false;
}

modules::HfMappingSpec to_hf_mapping_spec(const DslModel::MappingSpec& spec) {
    modules::HfMappingSpec out{};
    using SrcKind = DslModel::MappingSpec::Kind;
    using DstKind = modules::HfMappingSpec::Kind;
    switch (spec.kind) {
        case SrcKind::Direct:
            out.kind = DstKind::Direct;
            break;
        case SrcKind::Fuse:
            out.kind = DstKind::Fuse;
            break;
        case SrcKind::Split:
            out.kind = DstKind::Split;
            break;
        case SrcKind::Transform:
            out.kind = DstKind::Transform;
            break;
        case SrcKind::TiedTo:
            out.kind = DstKind::TiedTo;
            break;
        case SrcKind::StackExperts:
            out.kind = DstKind::StackExperts;
            break;
        case SrcKind::Unknown:
        default:
            out.kind = DstKind::Unknown;
            break;
    }
    out.source = spec.source;
    out.sources = spec.sources;
    out.ranges = spec.ranges;
    out.fn = spec.fn;
    out.target = spec.target;
    out.dim = spec.dim;
    out.optional = spec.optional;
    out.fuse_gate_up = spec.fuse_gate_up;
    out.num_experts = spec.num_experts;
    return out;
}

bool is_qlora_param_name(std::string_view name, bool train_router) {
    const std::string_view clean = trim_optional(name);
    int layer_idx = -1;
    std::string field;
    if (internal::parse_block_param(clean, layer_idx, field)) {
        if (!field.empty() && field.back() == '?') {
            field.pop_back();
        }
        if (train_router && field.find("router") != std::string::npos) {
            return false;
        }
        // Standard block weights (Qwen3, LLaMA, etc.)
        if (field == "qkv_weight" || field == "out_weight" || field == "o_proj_weight" ||
            field == "mlp_up_weight" || field == "mlp_down_weight" ||
            field == "ln1_weight" || field == "ln2_weight" ||
            field == "q_norm_weight" || field == "k_norm_weight" ||
            field == "router_weight" ||
            field == "experts_gate_up" || field == "experts_down" ||
            field == "shared_expert_gate" || field == "shared_expert_up" ||
            field == "shared_expert_down" ||
            // Nemotron-H MoE: separate up/down without gate fusion
            field == "experts_up") {
            return true;
        }
        // Mamba/SSM weights (pattern suffixes)
        if (ends_with(field, "in_proj_weight") ||
            ends_with(field, "in_proj_bias") ||
            ends_with(field, "out_proj_weight") ||
            ends_with(field, "out_proj_bias") ||
            ends_with(field, "conv1d_weight") ||
            ends_with(field, "conv1d_bias") ||
            // Nemotron-H Mamba: uses conv_weight/bias instead of conv1d_weight/bias
            ends_with(field, "conv_weight") ||
            ends_with(field, "conv_bias") ||
            ends_with(field, "A_log") ||
            ends_with(field, "D") ||
            // Nemotron-H Mamba: D_param naming variant
            field == "D_param" ||
            ends_with(field, "dt_bias") ||
            ends_with(field, "norm_weight")) {
            return true;
        }
        return false;
    }
    return clean == "embedding" || clean == "embeddings" || clean == "embed_tokens" ||
           clean == "final_norm" || clean == "final_norm_weight" || clean == "norm" ||
           clean == "lm_head" || clean == "lm_head_weight";
}

struct DslConfigView {
    std::optional<long> d_model;
    std::optional<long> d_ff;
    std::optional<long> n_layers;
    std::optional<long> num_query_heads;
    std::optional<long> num_kv_heads;
    std::optional<long> head_size;
    std::optional<long> max_seq;
    std::optional<long> vocab_size;
    std::optional<double> eps;
    std::optional<bool> use_qkv_bias;
    std::optional<bool> use_qk_norm;
    std::optional<long> num_experts;
    std::optional<long> num_experts_per_tok;
    std::optional<long> moe_intermediate_size;
    std::optional<bool> norm_topk_prob;
    std::optional<bool> use_shared_expert;
    std::optional<long> shared_expert_intermediate;
    std::optional<std::string> mlp_activation;
    std::optional<std::string> hybrid_pattern;  ///< Hybrid arch pattern (e.g., "MEMEM*EMEMEM*...")

    // Mamba / SSM configuration (for hybrid architectures like Nemotron-H)
    std::optional<long> mamba_num_heads;
    std::optional<long> mamba_head_dim;
    std::optional<long> ssm_state_size;
    std::optional<long> n_groups;
    std::optional<long> conv_kernel;
    std::optional<long> chunk_size;
    std::optional<bool> use_conv_bias;
    std::optional<bool> use_mamba_bias;
};

std::optional<long> get_long_attr(const AttrMap& map, const char* key) {
    if (const auto* value = internal::find_key(&map, key)) {
        if (auto v = internal::as_int(*value)) {
            return *v;
        }
    }
    return std::nullopt;
}

std::optional<double> get_double_attr(const AttrMap& map, const char* key) {
    if (const auto* value = internal::find_key(&map, key)) {
        if (const auto* f64 = std::get_if<double>(&value->value)) {
            return *f64;
        }
        if (const auto* i64 = std::get_if<std::int64_t>(&value->value)) {
            return static_cast<double>(*i64);
        }
    }
    return std::nullopt;
}

std::optional<bool> get_bool_attr(const AttrMap& map, const char* key) {
    if (const auto* value = internal::find_key(&map, key)) {
        if (auto v = internal::as_bool(*value)) {
            return *v;
        }
    }
    return std::nullopt;
}

std::optional<std::string> get_string_attr(const AttrMap& map, const char* key) {
    if (const auto* value = internal::find_key(&map, key)) {
        if (auto v = internal::as_string(*value)) {
            return std::string(*v);
        }
    }
    return std::nullopt;
}

DslConfigView parse_dsl_config(const Module& module) {
    DslConfigView view;
    const auto& cfg = module.config;
    view.d_model = get_long_attr(cfg, "d_model");
    view.d_ff = get_long_attr(cfg, "d_ff");
    view.n_layers = get_long_attr(cfg, "n_layers");
    view.num_query_heads = get_long_attr(cfg, "num_query_heads");
    view.num_kv_heads = get_long_attr(cfg, "num_kv_heads");
    view.head_size = get_long_attr(cfg, "head_size");
    view.max_seq = get_long_attr(cfg, "max_seq");
    view.vocab_size = get_long_attr(cfg, "vocab_size");
    view.eps = get_double_attr(cfg, "eps");
    view.use_qkv_bias = get_bool_attr(cfg, "use_qkv_bias");
    view.use_qk_norm = get_bool_attr(cfg, "use_qk_norm");
    view.num_experts = get_long_attr(cfg, "num_experts");
    view.num_experts_per_tok = get_long_attr(cfg, "num_experts_per_tok");
    view.moe_intermediate_size = get_long_attr(cfg, "moe_intermediate_size");
    view.norm_topk_prob = get_bool_attr(cfg, "norm_topk_prob");
    view.use_shared_expert = get_bool_attr(cfg, "use_shared_expert");
    view.shared_expert_intermediate = get_long_attr(cfg, "shared_expert_intermediate");
    view.mlp_activation = get_string_attr(cfg, "mlp_activation");
    view.hybrid_pattern = get_string_attr(cfg, "hybrid_pattern");

    // Mamba / SSM configuration
    view.mamba_num_heads = get_long_attr(cfg, "mamba_num_heads");
    view.mamba_head_dim = get_long_attr(cfg, "mamba_head_dim");
    view.ssm_state_size = get_long_attr(cfg, "ssm_state_size");
    view.n_groups = get_long_attr(cfg, "n_groups");
    view.conv_kernel = get_long_attr(cfg, "conv_kernel");
    view.chunk_size = get_long_attr(cfg, "chunk_size");
    view.use_conv_bias = get_bool_attr(cfg, "use_conv_bias");
    view.use_mamba_bias = get_bool_attr(cfg, "use_mamba_bias");
    return view;
}

DslRuntimeConfig build_runtime_config(const Module& module, const PretrainedConfig& base) {
    DslRuntimeConfig runtime;
    const auto view = parse_dsl_config(module);

    runtime.use_qk_norm = view.use_qk_norm.value_or(base.UseQKNorm);
    // If the IR graph uses qkv_qk_norm_rope, force-enable qk-norm even when
    // config.json does not expose a flag (e.g., Qwen3 defaults).
    if (!runtime.use_qk_norm && graph_has_kernel(module, "qkv_qk_norm_rope")) {
        runtime.use_qk_norm = true;
    }
    runtime.num_experts = static_cast<int>(view.num_experts.value_or(0));
    runtime.num_experts_per_tok = static_cast<int>(view.num_experts_per_tok.value_or(0));
    runtime.norm_topk_prob = view.norm_topk_prob.value_or(false);
    runtime.use_shared_expert = view.use_shared_expert.value_or(false);
    runtime.shared_expert_intermediate = static_cast<int>(view.shared_expert_intermediate.value_or(0));

    if (view.moe_intermediate_size.has_value()) {
        runtime.moe_intermediate_size = static_cast<int>(view.moe_intermediate_size.value());
    } else if (runtime.num_experts > 0 && view.d_ff.has_value()) {
        runtime.moe_intermediate_size = static_cast<int>(view.d_ff.value());
    }

    // Determine mlp_up_factor based on activation type
    // Gated activations (SwiGLU, GeGLU) use factor 2, non-gated (ReLU, ReLU2, SiLU) use factor 1
    if (view.mlp_activation.has_value()) {
        const std::string& act = *view.mlp_activation;
        if (act == "relu2" || act == "ReLU2" || act == "relu" || act == "ReLU" ||
            act == "silu" || act == "SiLU" || act == "gelu" || act == "GeLU") {
            runtime.mlp_up_factor = 1;
        } else {
            // Default to gated (SwiGLU, GeGLU)
            runtime.mlp_up_factor = 2;
        }
    }

    // Hybrid architecture pattern for Nemotron-H style models
    if (view.hybrid_pattern.has_value()) {
        runtime.hybrid_pattern = *view.hybrid_pattern;
    }

    return runtime;
}

std::optional<PretrainedConfig::ArchitectureId> arch_from_string(std::string_view name) {
    std::string lower = internal::to_lower(std::string(name));
    if (lower.find("qwen3moe") != std::string::npos || lower.find("qwen3_moe") != std::string::npos) {
        return PretrainedConfig::QWEN3_MOE;
    }
    if (lower.find("qwen3") != std::string::npos) {
        return PretrainedConfig::QWEN3;
    }
    if (lower.find("qwen2") != std::string::npos || lower.find("qwen25") != std::string::npos ||
        lower.find("qwen2.5") != std::string::npos) {
        return PretrainedConfig::QWEN2;
    }
    if (lower.find("nemotron") != std::string::npos) {
        return PretrainedConfig::NEMOTRON_H;
    }
    if (lower.find("llama") != std::string::npos) {
        return PretrainedConfig::LLAMA;
    }
    return std::nullopt;
}

void apply_arch_from_hf_config(PretrainedConfig& cfg, const Module& module) {
    if (const auto* value = internal::find_key(&module.hf_config, "architecture")) {
        if (auto arch = internal::as_string(*value)) {
            if (auto mapped = arch_from_string(*arch)) {
                cfg.Architecture = *mapped;
                return;
            }
        }
    }
    if (const auto* value = internal::find_key(&module.hf_config, "model_type")) {
        if (auto arch = internal::as_string(*value)) {
            if (auto mapped = arch_from_string(*arch)) {
                cfg.Architecture = *mapped;
            }
        }
    }
}

modules::ModelConfig build_model_config(const Module& module,
                                        const PretrainedConfig& base,
                                        const DslRuntimeConfig& runtime) {
    const auto view = parse_dsl_config(module);
    modules::ModelConfig cfg;
    cfg.original_config = base.clone();

    // Copy base fields
    cfg.Architecture = base.Architecture;
    cfg.BosTokenId = base.BosTokenId;
    cfg.EosTokenId = base.EosTokenId;
    cfg.PadTokenId = base.PadTokenId;
    cfg.HiddenSize = base.HiddenSize;
    cfg.IntermediateSize = base.IntermediateSize;
    cfg.VocabSize = base.VocabSize;
    cfg.NumQueryHeads = base.NumQueryHeads;
    cfg.NumKeyValHeads = base.NumKeyValHeads;
    cfg.NumLayers = base.NumLayers;
    cfg.HeadDim = base.HeadDim;
    cfg.MaxPositionEmbeddings = base.MaxPositionEmbeddings;
    cfg.RopeTheta = base.RopeTheta;
    cfg.Rope = base.Rope;
    cfg.RmsNormEps = base.RmsNormEps;
    cfg.TiedWordEmbeddings = base.TiedWordEmbeddings;
    cfg.UseQKVBias = base.UseQKVBias;
    cfg.UseQKNorm = base.UseQKNorm;
    cfg.DType = base.DType;

    // Override with DSL-provided values when available
    if (view.d_model) cfg.HiddenSize = static_cast<int>(*view.d_model);
    if (view.d_ff) cfg.IntermediateSize = static_cast<int>(*view.d_ff);
    if (view.n_layers) cfg.NumLayers = static_cast<int>(*view.n_layers);
    if (view.num_query_heads) cfg.NumQueryHeads = static_cast<int>(*view.num_query_heads);
    if (view.num_kv_heads) cfg.NumKeyValHeads = static_cast<int>(*view.num_kv_heads);
    if (view.head_size) cfg.HeadDim = static_cast<int>(*view.head_size);
    if (view.max_seq) cfg.MaxPositionEmbeddings = static_cast<int>(*view.max_seq);
    if (view.vocab_size) cfg.VocabSize = static_cast<int>(*view.vocab_size);
    if (view.eps) cfg.RmsNormEps = static_cast<float>(*view.eps);
    if (view.use_qkv_bias) cfg.UseQKVBias = *view.use_qkv_bias;

    cfg.UseQKNorm = runtime.use_qk_norm;
    cfg.use_qk_norm = runtime.use_qk_norm;

    // Infer attention type from head counts
    if (cfg.NumKeyValHeads == 1) {
        cfg.attention_type = modules::AttentionType::MQA;
    } else if (cfg.NumKeyValHeads < cfg.NumQueryHeads) {
        cfg.attention_type = modules::AttentionType::GQA;
    } else {
        cfg.attention_type = modules::AttentionType::MHA;
    }

    // MoE configuration (DSL-driven)
    if (runtime.num_experts > 0) {
        cfg.architecture = modules::ArchitectureType::MoE;
        modules::MoEConfig moe;
        moe.num_experts = runtime.num_experts;
        moe.top_k = runtime.num_experts_per_tok > 0 ? runtime.num_experts_per_tok : 1;
        moe.moe_intermediate_size = runtime.moe_intermediate_size > 0
                                        ? runtime.moe_intermediate_size
                                        : cfg.IntermediateSize;
        moe.norm_topk_prob = runtime.norm_topk_prob;
        moe.use_shared_expert = runtime.use_shared_expert;
        moe.shared_expert_size = runtime.shared_expert_intermediate;
        cfg.moe_config = moe;

        cfg.NumExperts = moe.num_experts;
        cfg.NumExpertsPerTok = moe.top_k;
        cfg.MoeIntermediateSize = moe.moe_intermediate_size;
    } else {
        cfg.architecture = modules::ArchitectureType::Dense;
    }

    // Set activation type based on mlp_up_factor
    // factor 1 = non-gated (ReLU2), factor 2 = gated (SwiGLU)
    if (runtime.mlp_up_factor == 1) {
        cfg.activation_type = modules::ActivationType::ReLU2;
    } else {
        cfg.activation_type = modules::ActivationType::SwiGLU;
    }

    // Mamba / SSM configuration (for hybrid architectures like Nemotron-H)
    if (view.mamba_num_heads) cfg.MambaNumHeads = static_cast<int>(*view.mamba_num_heads);
    if (view.mamba_head_dim) cfg.MambaHeadDim = static_cast<int>(*view.mamba_head_dim);
    if (view.ssm_state_size) cfg.MambaSsmStateSize = static_cast<int>(*view.ssm_state_size);
    if (view.n_groups) cfg.MambaNGroups = static_cast<int>(*view.n_groups);
    if (view.conv_kernel) cfg.MambaConvKernel = static_cast<int>(*view.conv_kernel);
    if (view.chunk_size) cfg.MambaChunkSize = static_cast<int>(*view.chunk_size);
    if (view.use_conv_bias) cfg.MambaUseConvBias = *view.use_conv_bias;
    if (view.use_mamba_bias) cfg.MambaUseBias = *view.use_mamba_bias;
    // Mamba intermediate = num_heads * head_dim (distinct from MLP intermediate)
    if (view.mamba_num_heads && view.mamba_head_dim) {
        cfg.MambaIntermediateSize = static_cast<int>(*view.mamba_num_heads * *view.mamba_head_dim);
    }

    // Populate layer_overrides from hybrid_pattern for hybrid architectures
    // Pattern chars: M = Mamba, E = MoE, * = Attention, - = MLP
    if (!runtime.hybrid_pattern.empty()) {
        cfg.layer_overrides.clear();
        cfg.layer_overrides.reserve(cfg.NumLayers);

        for (int i = 0; i < cfg.NumLayers && i < static_cast<int>(runtime.hybrid_pattern.size()); ++i) {
            const char c = runtime.hybrid_pattern[i];
            modules::LayerOverride override;
            override.layer_idx = i;

            switch (c) {
                case 'M':
                    override.block_type = modules::BlockType::Mamba;
                    override.is_moe = false;
                    break;
                case 'E':
                    override.block_type = modules::BlockType::MoE;
                    override.is_moe = true;
                    break;
                case '*':
                    override.block_type = modules::BlockType::Attention;
                    override.is_moe = false;
                    break;
                case '-':
                    override.block_type = modules::BlockType::MLP;
                    override.is_moe = false;
                    break;
                default:
                    override.block_type = modules::BlockType::Dense;
                    override.is_moe = false;
                    break;
            }

            cfg.layer_overrides.push_back(override);
        }

        // Update architecture type if we have a hybrid pattern
        cfg.architecture = modules::ArchitectureType::Hybrid;
    }

    return cfg;
}

template<typename Block>
class DslQLoRAWeightProvider final : public QLoRAWeightProvider {
public:
    DslQLoRAWeightProvider(const modules::ModelConfig& cfg,
                           const RuntimeOptions& options,
                           const modules::ModularLoRAConfig& lora_config,
                           const modules::QLoRAConfig& qlora_config,
                           const std::shared_ptr<TensorAllocator>& allocator,
                           const modules::HfMapping* hf_mapping)
        : mConfig(cfg),
          mOptions(options),
          mLoRAConfig(lora_config),
          mQLoRAConfig(qlora_config),
          mAllocator(allocator),
          mHfMapping(hf_mapping) {}

    bool handles_param(std::string_view name) const override {
        return is_qlora_param_name(name, mLoRAConfig.train_router);
    }

    Tensor& resolve_param(std::string_view name, cudaStream_t stream) override {
        if (!mFP8Provider && !mFP4Provider && !mBnBProvider) {
            throw std::runtime_error("DSL QLoRA provider: weights not initialized (import_weights_qlora not called)");
        }

        const std::string_view clean = trim_optional(name);
        
        if (clean == "embedding" || clean == "embeddings" || clean == "embed_tokens") {
            return with_provider([&](auto& provider) -> Tensor& { return provider.get_embeddings(stream); });
        }
        if (clean == "final_norm" || clean == "final_norm_weight" || clean == "norm") {
             return with_provider([&](auto& provider) -> Tensor& { return provider.get_final_norm(stream); });
        }
        if (clean == "lm_head" || clean == "lm_head_weight") {
            return with_provider([&](auto& provider) -> Tensor& { return provider.get_lm_head(stream); });
        }

        int layer_idx = -1;
        std::string field;
        std::string block_type;
        if (!internal::parse_block_param_with_type(clean, layer_idx, field, block_type)) {
            throw std::runtime_error("DSL QLoRA provider: unknown param " + std::string(name));
        }
        if (!field.empty() && field.back() == '?') {
            field.pop_back();
        }

        // Map block-type-specific index to physical layer index for hybrid architectures
        // e.g., mamba_blocks[0] needs to map to the physical layer index of the first Mamba layer
        int physical_layer_idx = map_to_physical_layer(block_type, layer_idx);

        auto& block = with_provider([&](auto& provider) -> typename std::remove_reference_t<decltype(provider)>::BlockWeights& {
            return provider.get_block(physical_layer_idx, stream);
        });

        if (field == "ln1_weight" || field == "norm_weight") {
            // norm_weight is used by hybrid blocks (Mamba, Attention, MLP, MoE) for pre-block norm
            return block.ln1.weight;
        }
        if (field == "ln2_weight") {
            return block.ln2.weight;
        }
        if (field == "qkv_weight") {
            return block.attention.qkv_weight;
        }
        if (field == "out_weight" || field == "o_proj_weight") {
            return block.attention.out_weight;
        }
        if (field == "mlp_up_weight" || field == "up_weight") {
            if constexpr (requires { block.mlp_up_weight; }) {
                return block.mlp_up_weight;
            }
            throw std::runtime_error("DSL QLoRA provider: mlp_up_weight not available for " + std::string(name));
        }
        if (field == "mlp_down_weight" || field == "down_weight") {
            if constexpr (requires { block.mlp_down_weight; }) {
                return block.mlp_down_weight;
            }
            throw std::runtime_error("DSL QLoRA provider: mlp_down_weight not available for " + std::string(name));
        }
        if (field == "q_norm_weight") {
            if constexpr (requires { block.attention.q_norm_weight; }) {
                if (block.attention.q_norm_weight.has_value()) {
                    return block.attention.q_norm_weight.value();
                }
            }
            throw std::runtime_error("DSL QLoRA provider: q_norm_weight not available for " + std::string(name));
        }
        if (field == "k_norm_weight") {
            if constexpr (requires { block.attention.k_norm_weight; }) {
                if (block.attention.k_norm_weight.has_value()) {
                    return block.attention.k_norm_weight.value();
                }
            }
            throw std::runtime_error("DSL QLoRA provider: k_norm_weight not available for " + std::string(name));
        }

        if (field == "router_weight") {
            if constexpr (requires { block.router.gate; }) {
                return block.router.gate;
            }
            throw std::runtime_error("DSL QLoRA provider: router_weight not available for " + std::string(name));
        }
        if (field == "experts_gate_up" || field == "experts_up") {
            if constexpr (requires { block.experts.gate_up_proj; }) {
                return block.experts.gate_up_proj;
            }
            throw std::runtime_error("DSL QLoRA provider: experts_gate_up not available for " + std::string(name));
        }
        if (field == "experts_down") {
            if constexpr (requires { block.experts.down_proj; }) {
                return block.experts.down_proj;
            }
            throw std::runtime_error("DSL QLoRA provider: experts_down not available for " + std::string(name));
        }
        if (field == "shared_expert_gate") {
            if constexpr (requires { block.shared_expert; }) {
                if (block.shared_expert.has_value()) {
                    return block.shared_expert->gate_proj;
                }
            }
            throw std::runtime_error("DSL QLoRA provider: shared_expert_gate not available for " + std::string(name));
        }
        if (field == "shared_expert_up") {
            if constexpr (requires { block.shared_expert; }) {
                if (block.shared_expert.has_value()) {
                    return block.shared_expert->up_proj;
                }
            }
            throw std::runtime_error("DSL QLoRA provider: shared_expert_up not available for " + std::string(name));
        }
        if (field == "shared_expert_down") {
            if constexpr (requires { block.shared_expert; }) {
                if (block.shared_expert.has_value()) {
                    return block.shared_expert->down_proj;
                }
            }
            throw std::runtime_error("DSL QLoRA provider: shared_expert_down not available for " + std::string(name));
        }

        if (ends_with(field, "in_proj_weight")) {
            if constexpr (requires { block.mamba; }) {
                if (block.mamba.has_value()) {
                    return block.mamba->in_proj_weight;
                }
            }
            throw std::runtime_error("DSL QLoRA provider: in_proj_weight not available for " + std::string(name));
        }
        if (ends_with(field, "in_proj_bias")) {
            if constexpr (requires { block.mamba; }) {
                if (block.mamba.has_value() && block.mamba->in_proj_bias.has_value()) {
                    return block.mamba->in_proj_bias.value();
                }
            }
            throw std::runtime_error("DSL QLoRA provider: in_proj_bias not available for " + std::string(name));
        }
        if (ends_with(field, "out_proj_weight")) {
            if constexpr (requires { block.mamba; }) {
                if (block.mamba.has_value()) {
                    return block.mamba->out_proj_weight;
                }
            }
            throw std::runtime_error("DSL QLoRA provider: out_proj_weight not available for " + std::string(name));
        }
        if (ends_with(field, "out_proj_bias")) {
            if constexpr (requires { block.mamba; }) {
                if (block.mamba.has_value() && block.mamba->out_proj_bias.has_value()) {
                    return block.mamba->out_proj_bias.value();
                }
            }
            throw std::runtime_error("DSL QLoRA provider: out_proj_bias not available for " + std::string(name));
        }
        if (ends_with(field, "conv1d_weight") || ends_with(field, "conv_weight")) {
            if constexpr (requires { block.mamba; }) {
                if (block.mamba.has_value()) {
                    return block.mamba->conv1d_weight;
                }
            }
            throw std::runtime_error("DSL QLoRA provider: conv_weight not available for " + std::string(name));
        }
        if (ends_with(field, "conv1d_bias") || ends_with(field, "conv_bias")) {
            if constexpr (requires { block.mamba; }) {
                if (block.mamba.has_value() && block.mamba->conv1d_bias.has_value()) {
                    return block.mamba->conv1d_bias.value();
                }
            }
            throw std::runtime_error("DSL QLoRA provider: conv_bias not available for " + std::string(name));
        }
        if (ends_with(field, "A_log")) {
            if constexpr (requires { block.mamba; }) {
                if (block.mamba.has_value()) {
                    return block.mamba->A_log;
                }
            }
            throw std::runtime_error("DSL QLoRA provider: A_log not available for " + std::string(name));
        }
        if (field == "D" || field == "D_param") {
            if constexpr (requires { block.mamba; }) {
                if (block.mamba.has_value()) {
                    return block.mamba->D;
                }
            }
            throw std::runtime_error("DSL QLoRA provider: D_param not available for " + std::string(name));
        }
        if (ends_with(field, "dt_bias")) {
            if constexpr (requires { block.mamba; }) {
                if (block.mamba.has_value()) {
                    return block.mamba->dt_bias;
                }
            }
            throw std::runtime_error("DSL QLoRA provider: dt_bias not available for " + std::string(name));
        }
        if (ends_with(field, "norm_weight")) {
            if constexpr (requires { block.mamba; }) {
                if (block.mamba.has_value()) {
                    return block.mamba->norm_weight;
                }
            }
            throw std::runtime_error("DSL QLoRA provider: norm_weight not available for " + std::string(name));
        }

        throw std::runtime_error("DSL QLoRA provider: unsupported param " + std::string(name));
    }

    void import_and_quantize(const std::string& file_name, NCCLCommunicator& comm,
                             cudaStream_t stream) override {
        ensure_provider(comm);
        with_provider([&](auto& provider) { provider.import_and_quantize(file_name, comm, stream); });
    }

    void invalidate_cache() override {
        with_provider([&](auto& provider) { provider.invalidate_cache(); });
    }

    bool refresh_moe_experts(int layer_idx,
                             const modules::SelectiveExpertInfo& selection,
                             cudaStream_t stream) override {
        if (!selection.enabled || selection.num_active == 0) {
            return false;
        }
        if (mBnBProvider) {
            if (!mBnBProvider->use_selective_dequant()) {
                return false;
            }
            mBnBProvider->dequantize_selected_experts(layer_idx, selection, stream, true);
            return true;
        }
        if (mFP4Provider) {
            if (!mFP4Provider->use_selective_dequant()) {
                return false;
            }
            mFP4Provider->dequantize_selected_experts(layer_idx, selection, stream, true);
            return true;
        }
        // FP8 provider not yet wired for selective MoE refresh in DSL path.
        return false;
    }

    std::size_t quantized_weights_bytes() const override {
        return with_provider([&](const auto& provider) { return provider.quantized_weights_bytes(); });
    }

    float memory_savings_ratio() const override {
        return with_provider([&](const auto& provider) { return provider.memory_savings_ratio(); });
    }

private:
    using FP8Provider = modules::FP8WeightProvider<Block>;
    using FP4Provider = modules::FP4WeightProvider<Block>;
    using BnBProvider = modules::BnBWeightProvider<Block>;

    template<typename Fn>
    decltype(auto) with_provider(Fn&& fn) {
        if (mFP8Provider) return fn(*mFP8Provider);
        if (mFP4Provider) return fn(*mFP4Provider);
        if (mBnBProvider) return fn(*mBnBProvider);
        throw std::runtime_error("DSL QLoRA provider: no provider initialized");
    }

    template<typename Fn>
    decltype(auto) with_provider(Fn&& fn) const {
        if (mFP8Provider) return fn(*mFP8Provider);
        if (mFP4Provider) return fn(*mFP4Provider);
        if (mBnBProvider) return fn(*mBnBProvider);
        throw std::runtime_error("DSL QLoRA provider: no provider initialized");
    }

    /**
     * @brief Map block-type-specific index to physical layer index
     *
     * For hybrid architectures, DSL models use separate arrays for each block type:
     * - mamba_blocks[0], mamba_blocks[1], ... for Mamba layers
     * - attn_blocks[0], attn_blocks[1], ... for Attention layers
     * - mlp_blocks[0], mlp_blocks[1], ... for MLP layers
     * - moe_blocks[0], moe_blocks[1], ... for MoE layers
     *
     * This function maps (block_type, block_type_index) to the physical layer index
     * that the BnB weight provider uses.
     *
     * @param block_type Block type string (e.g., "mamba", "attn", "mlp", "moe", or empty)
     * @param block_type_idx Index within the block type array
     * @return Physical layer index
     */
    int map_to_physical_layer(const std::string& block_type, int block_type_idx) const {
        // If no block type specified (plain "blocks[N]"), return the index as-is
        if (block_type.empty()) {
            return block_type_idx;
        }

        // Map block type string to BlockType enum
        modules::BlockType target_type = modules::BlockType::Dense;
        if (block_type == "mamba") {
            target_type = modules::BlockType::Mamba;
        } else if (block_type == "attn" || block_type == "attention") {
            target_type = modules::BlockType::Attention;
        } else if (block_type == "mlp") {
            target_type = modules::BlockType::MLP;
        } else if (block_type == "moe") {
            target_type = modules::BlockType::MoE;
        } else {
            // Unknown block type, return index as-is
            return block_type_idx;
        }

        // Count layers of the target type until we reach block_type_idx
        int count = 0;
        for (int i = 0; i < mConfig.NumLayers; ++i) {
            const auto layer_type = mConfig.get_block_type(i);
            if (layer_type == target_type) {
                if (count == block_type_idx) {
                    return i;  // Found the N-th layer of this type
                }
                ++count;
            }
        }

        // Block type index out of range, return the index as-is (will likely cause an error later)
        return block_type_idx;
    }

    void fill_mamba_config(auto& cfg) {
        cfg.layer_is_mamba.resize(mConfig.NumLayers);
        cfg.has_mamba = false;

        // Build hybrid_pattern string from per-layer block types
        // Pattern chars: M = Mamba, E = MoE, * = Attention, - = MLP
        std::string pattern;
        pattern.reserve(mConfig.NumLayers);
        bool is_hybrid = false;

        for (int i = 0; i < mConfig.NumLayers; ++i) {
            const auto block_type = mConfig.get_block_type(i);
            const bool is_mamba = (block_type == modules::BlockType::Mamba);
            cfg.layer_is_mamba[i] = static_cast<std::uint8_t>(is_mamba ? 1 : 0);
            cfg.has_mamba = cfg.has_mamba || is_mamba;

            // Build pattern character
            char c = '-';  // Default: MLP
            switch (block_type) {
                case modules::BlockType::Mamba:
                    c = 'M';
                    is_hybrid = true;
                    break;
                case modules::BlockType::MoE:
                case modules::BlockType::SwitchMoE:
                    c = 'E';
                    is_hybrid = true;
                    break;
                case modules::BlockType::Attention:
                    c = '*';
                    is_hybrid = true;
                    break;
                case modules::BlockType::MLP:
                    c = '-';
                    break;
                default:
                    c = '-';  // Dense blocks default to MLP behavior
                    break;
            }
            pattern.push_back(c);
        }

        // Only set hybrid_pattern if we have a true hybrid architecture
        if (is_hybrid) {
            cfg.hybrid_pattern = std::move(pattern);
        }

        cfg.mamba_num_heads = mConfig.MambaNumHeads;
        cfg.mamba_head_dim = mConfig.MambaHeadDim;
        cfg.mamba_ssm_state_size = mConfig.MambaSsmStateSize;
        cfg.mamba_conv_kernel = mConfig.MambaConvKernel;
        cfg.mamba_n_groups = mConfig.MambaNGroups;
        cfg.mamba_intermediate_size = mConfig.MambaIntermediateSize;
        cfg.mamba_use_bias = mConfig.MambaUseBias;
        cfg.mamba_use_conv_bias = mConfig.MambaUseConvBias;
    }

    void ensure_provider(NCCLCommunicator& comm) {
        if (mFP8Provider || mFP4Provider || mBnBProvider) {
            return;
        }

        const bool force_full_moe_dequant =
            mQLoRAConfig.is_moe() && mLoRAConfig.enabled() &&
            (mOptions.SelectiveExpertDequant || mOptions.OffloadExperts);
        if (force_full_moe_dequant) {
            std::cerr << "[QLoRA] MoE selective expert dequant disabled\n";
            if (mOptions.OffloadExperts) {
                std::cerr << "[QLoRA] Offload experts enabled; full expert dequant will stream all experts.\n";
            }
        }

        int device_id = 0;
        CUDA_CHECK(cudaGetDevice(&device_id));
        cudaDeviceProp device_props{};
        CUDA_CHECK(cudaGetDeviceProperties(&device_props, device_id));

        if (mQLoRAConfig.is_fp8()) {
            typename FP8Provider::Config cfg{};
            cfg.num_layers = mConfig.NumLayers;
            cfg.hidden_size = mConfig.HiddenSize;
            cfg.intermediate_size = mConfig.IntermediateSize;
            cfg.num_query_heads = mConfig.NumQueryHeads;
            cfg.num_kv_heads = mConfig.NumKeyValHeads;
            cfg.head_size = mConfig.head_size();
            cfg.vocab_size = mConfig.VocabSize;
            cfg.mlp_up_factor = mConfig.mlp_up_factor();
            cfg.qlora_config = mQLoRAConfig;
            cfg.lora_config = mLoRAConfig;
            cfg.model_dtype = mConfig.DType;
            cfg.use_qk_norm = mConfig.UseQKNorm;
            cfg.tied_embeddings = mConfig.TiedWordEmbeddings;
            cfg.hf_mapping = mHfMapping;
            cfg.shard_idx = comm.rank();
            cfg.num_shards = comm.world_size();
            cfg.enable_fp8_forward = mOptions.fp8_forward_enabled();
            cfg.enable_fp8_hybrid = mOptions.fp8_hybrid_enabled();
            fill_mamba_config(cfg);
            mFP8Provider = std::make_unique<FP8Provider>(cfg, *mAllocator, device_props);
            return;
        }

        if (mQLoRAConfig.is_fp4()) {
            typename FP4Provider::Config cfg{};
            cfg.num_layers = mConfig.NumLayers;
            cfg.hidden_size = mConfig.HiddenSize;
            cfg.intermediate_size = mConfig.IntermediateSize;
            cfg.num_query_heads = mConfig.NumQueryHeads;
            cfg.num_kv_heads = mConfig.NumKeyValHeads;
            cfg.head_size = mConfig.head_size();
            cfg.vocab_size = mConfig.VocabSize;
            cfg.mlp_up_factor = mConfig.mlp_up_factor();
            cfg.qlora_config = mQLoRAConfig;
            cfg.lora_config = mLoRAConfig;
            cfg.model_dtype = mConfig.DType;
            cfg.use_qk_norm = mConfig.UseQKNorm;
            cfg.tied_embeddings = mConfig.TiedWordEmbeddings;
            cfg.hf_mapping = mHfMapping;
            cfg.shard_idx = comm.rank();
            cfg.num_shards = comm.world_size();
            cfg.selective_expert_dequant = force_full_moe_dequant ? false : mOptions.SelectiveExpertDequant;
            cfg.force_full_expert_dequant = force_full_moe_dequant;
            cfg.offload_experts = mOptions.OffloadExperts;
            fill_mamba_config(cfg);
            mFP4Provider = std::make_unique<FP4Provider>(cfg, *mAllocator, device_props);
            return;
        }

        if (mQLoRAConfig.is_bnb()) {
            typename BnBProvider::Config cfg{};
            cfg.num_layers = mConfig.NumLayers;
            cfg.hidden_size = mConfig.HiddenSize;
            cfg.intermediate_size = mConfig.IntermediateSize;
            cfg.num_query_heads = mConfig.NumQueryHeads;
            cfg.num_kv_heads = mConfig.NumKeyValHeads;
            cfg.head_size = mConfig.head_size();
            cfg.vocab_size = mConfig.VocabSize;
            cfg.mlp_up_factor = mConfig.mlp_up_factor();
            cfg.qlora_config = mQLoRAConfig;
            cfg.lora_config = mLoRAConfig;
            cfg.model_dtype = mConfig.DType;
            cfg.use_qk_norm = mConfig.UseQKNorm;
            cfg.tied_embeddings = mConfig.TiedWordEmbeddings;
            cfg.hf_mapping = mHfMapping;
            cfg.shard_idx = comm.rank();
            cfg.num_shards = comm.world_size();
            cfg.selective_expert_dequant = force_full_moe_dequant ? false : mOptions.SelectiveExpertDequant;
            cfg.force_full_expert_dequant = force_full_moe_dequant;
            cfg.offload_experts = mOptions.OffloadExperts;
            fill_mamba_config(cfg);
            mBnBProvider = std::make_unique<BnBProvider>(cfg, *mAllocator, device_props);
            return;
        }
    }

    modules::ModelConfig mConfig;
    RuntimeOptions mOptions;
    modules::ModularLoRAConfig mLoRAConfig;
    modules::QLoRAConfig mQLoRAConfig;
    std::shared_ptr<TensorAllocator> mAllocator;
    const modules::HfMapping* mHfMapping = nullptr;

    std::unique_ptr<FP8Provider> mFP8Provider;
    std::unique_ptr<FP4Provider> mFP4Provider;
    std::unique_ptr<BnBProvider> mBnBProvider;
};

}  // namespace

namespace internal {

std::unique_ptr<QLoRAWeightProvider> create_dsl_qlora_provider(
    const modules::ModelConfig& model_cfg,
    const RuntimeOptions& options,
    const modules::ModularLoRAConfig& lora_cfg,
    const modules::QLoRAConfig& qlora_cfg,
    const std::shared_ptr<TensorAllocator>& allocator,
    const modules::HfMapping* hf_mapping) {
    const bool is_moe = qlora_cfg.is_moe() || model_cfg.moe_config.has_value();
    if (is_moe) {
        return std::make_unique<DslQLoRAWeightProvider<modules::DslMoEBlock>>(
            model_cfg, options, lora_cfg, qlora_cfg, allocator, hf_mapping);
    }
    return std::make_unique<DslQLoRAWeightProvider<modules::DslDenseBlock>>(
        model_cfg, options, lora_cfg, qlora_cfg, allocator, hf_mapping);
}

}  // namespace internal

DslModel::DslModel(const PretrainedConfig& config,
                   const RuntimeOptions& options,
                   const std::string& ir_json,
                   const std::shared_ptr<TensorAllocator>& allocator,
                   const std::optional<modules::ModularLoRAConfig>& lora_config,
                   const modules::QLoRAConfig& qlora_config,
                   int shard_idx,
                   int num_shards)
    : mConfig(config.clone()),
      mAllocator(allocator ? allocator : std::make_shared<TensorAllocator>()),
      mOptions(options),
      mQLoRAConfig(qlora_config),
      mShardIdx(shard_idx),
      mNumShards(num_shards) {
    if (ir_json.empty()) {
        throw std::runtime_error("DSL model: IR JSON is empty");
    }
    nlohmann::json root = nlohmann::json::parse(ir_json);
    mIr = load_ir_from_json(root);
    if (!mIr.success) {
        std::string error_msg = "DSL model: IR compilation failed";
        if (!mIr.errors.empty()) {
            error_msg += ":\n";
            for (const auto& err : mIr.errors) {
                error_msg += "  - " + err + "\n";
            }
        }
        throw std::runtime_error(error_msg);
    }
    mModule = &pick_model_module(mIr);
    validate_ir();
    apply_arch_from_hf_config(*mConfig, *mModule);
    mRuntimeConfig = build_runtime_config(*mModule, *mConfig);
    mModelConfig = build_model_config(*mModule, *mConfig, mRuntimeConfig);

    if (!mModule->forward.has_value()) {
        throw std::runtime_error("DSL model: module missing forward graph");
    }

    std::unordered_set<std::string> external_params;
    // When QLoRA is enabled (quantized base weights), mark base model weights as "external".
    // External params are not allocated in DslParamStore - they're provided on-demand by the
    // QLoRA weight provider which holds quantized weights and dequantizes them as needed.
    // This is critical for memory efficiency: avoids allocating full-precision base weights.
    if (mQLoRAConfig.is_quantized()) {
        const bool train_router = lora_config.has_value() && lora_config->train_router;
        for (const auto& kv : mModule->forward->params) {
            const std::string& name = kv.first;
            if (is_qlora_param_name(name, train_router)) {
                external_params.insert(name);
            }
        }
    }

    const bool use_weight_manager = (options.ShardWeights || options.OffloadMaster) && !mQLoRAConfig.is_quantized();
    mParams = std::make_unique<DslParamStore>(*mModule, mModule->forward.value(),
                                              options, *mConfig, mAllocator,
                                              lora_config ? &*lora_config : nullptr,
                                              external_params.empty() ? nullptr : &external_params,
                                              use_weight_manager);
    mGrads = std::make_unique<DslGradStore>(*mParams, mAllocator,
                                            options.OffloadGrads,
                                            options.offload_alloc(),
                                            mNumShards,
                                            mConfig->TiedWordEmbeddings);

    // Create weight manager for streaming/sharding if enabled
    if (use_weight_manager) {
        mWeightManager = std::make_unique<DslWeightManager>(
            *mModule, mModule->forward.value(), options, *mConfig, mAllocator,
            lora_config ? &*lora_config : nullptr, mShardIdx, mNumShards);
        mParams->set_weight_manager(mWeightManager.get());
    }

    if (lora_config.has_value() && lora_config->enabled()) {
        mLoRAConfig = lora_config;
        mIsMoEModel = (mModelConfig.architecture == modules::ArchitectureType::MoE) || mModelConfig.moe_config.has_value();

        modules::ModularLoRAWeightsManager::Config wm{};
        wm.num_layers = mModelConfig.NumLayers;
        wm.hidden_size = mModelConfig.HiddenSize;
        wm.intermediate_size = mModelConfig.IntermediateSize;
        wm.num_query_heads = mModelConfig.NumQueryHeads;
        wm.num_kv_heads = mModelConfig.NumKeyValHeads;
        wm.head_size = mModelConfig.head_size();
        wm.lora_config = *mLoRAConfig;
        wm.work_dtype = mModelConfig.DType;
        wm.shard_idx = mShardIdx;
        wm.num_shards = mNumShards;
        wm.is_moe = mIsMoEModel;
        if (mIsMoEModel && mModelConfig.moe_config.has_value()) {
            wm.num_experts = mModelConfig.moe_config->num_experts;
            wm.moe_intermediate_size = mModelConfig.moe_config->moe_intermediate_size > 0
                                        ? mModelConfig.moe_config->moe_intermediate_size
                                        : mModelConfig.IntermediateSize;
            wm.train_router = mLoRAConfig->train_router;
        }
        mLoRAWeights = std::make_unique<modules::ModularLoRAWeightsManager>(wm, *mAllocator);

        modules::ModularLoRAGradsManager::Config gm{};
        gm.num_layers = mModelConfig.NumLayers;
        gm.hidden_size = mModelConfig.HiddenSize;
        gm.intermediate_size = mModelConfig.IntermediateSize;
        gm.num_query_heads = mModelConfig.NumQueryHeads;
        gm.num_kv_heads = mModelConfig.NumKeyValHeads;
        gm.head_size = mModelConfig.head_size();
        gm.lora_config = *mLoRAConfig;
        gm.grad_dtype = mLoRAConfig->dtype;
        gm.shard_idx = mShardIdx;
        gm.num_shards = mNumShards;
        gm.is_moe = mIsMoEModel;
        if (mIsMoEModel && mModelConfig.moe_config.has_value()) {
            gm.num_experts = mModelConfig.moe_config->num_experts;
            gm.moe_intermediate_size = mModelConfig.moe_config->moe_intermediate_size > 0
                                        ? mModelConfig.moe_config->moe_intermediate_size
                                        : mModelConfig.IntermediateSize;
            gm.train_router = mLoRAConfig->train_router;
        }
        mLoRAGrads = std::make_unique<modules::ModularLoRAGradsManager>(gm, mAllocator);
    }

    for (const auto& kv : mModule->hf_mapping) {
        mHfMapping.emplace(kv.first, internal::parse_mapping_spec(kv.second));
    }
    for (const auto& kv : mModule->hf_export) {
        mHfExport.emplace(kv.first, internal::parse_mapping_spec(kv.second));
    }
    if (!mHfMapping.empty()) {
        mQLoRAMapping = std::make_shared<modules::HfMapping>();
        mQLoRAMapping->mapping.reserve(mHfMapping.size());
        for (const auto& kv : mHfMapping) {
            mQLoRAMapping->mapping.emplace(kv.first, to_hf_mapping_spec(kv.second));
        }
    }
}

DslModel::~DslModel() = default;

modules::ModularLoRAWeightsManager& DslModel::lora_weights() {
    if (!mLoRAWeights) {
        throw std::runtime_error("DSL model: LoRA not enabled");
    }
    return *mLoRAWeights;
}

modules::ModularLoRAGradsManager& DslModel::lora_grads() {
    if (!mLoRAGrads) {
        throw std::runtime_error("DSL model: LoRA not enabled");
    }
    return *mLoRAGrads;
}

modules::LoRARunState& DslModel::lora_run_state() {
    if (!mLoRARunState) {
        throw std::runtime_error("DSL model: LoRA run state not allocated");
    }
    return *mLoRARunState;
}

const DslGradStore& DslModel::grads() const {
    if (!mGrads) {
        throw std::runtime_error("DSL model: gradients not initialized");
    }
    return *mGrads;
}

std::size_t DslModel::qlora_quantized_weights_bytes() const {
    return mQLoRAProvider ? mQLoRAProvider->quantized_weights_bytes() : 0;
}

float DslModel::qlora_memory_savings_ratio() const {
    return mQLoRAProvider ? mQLoRAProvider->memory_savings_ratio() : 1.0f;
}

void DslModel::validate_ir() {
    if (!mModule) {
        throw std::runtime_error("DSL model: no module selected");
    }
    validate_config_mapping(*mModule);
    validate_param_shapes(*mModule);
}

const Module& DslModel::pick_model_module(const IRFile& ir) const {
    const Module* candidate = nullptr;
    for (const auto& mod : ir.modules) {
        if (mod.kind != "model") {
            continue;
        }
        if (candidate) {
            throw std::runtime_error("DSL model: multiple model modules in IR");
        }
        candidate = &mod;
    }
    if (!candidate) {
        throw std::runtime_error("DSL model: no model module in IR");
    }
    return *candidate;
}

void DslModel::validate_config_mapping(const Module& module) const {
    const AttrMap* mapping = nullptr;
    auto it = module.hf_config.find("param_mapping");
    if (it != module.hf_config.end()) {
        if (const auto* map_ptr = std::get_if<AttrValue::MapPtr>(&it->second.value)) {
            if (*map_ptr) {
                mapping = map_ptr->get();
            }
        }
    }
    if (!mapping) {
        mapping = &module.hf_config;
    }

    for (const auto& kv : *mapping) {
        const auto* hf_key = std::get_if<std::string>(&kv.second.value);
        if (!hf_key) {
            continue;
        }
        auto expected = internal::get_hf_value(*mConfig, *hf_key);
        if (!expected) {
            continue;
        }
        auto it = module.config.find(kv.first);
        if (it == module.config.end()) {
            throw std::runtime_error("DSL model: missing module config param " + kv.first);
        }
        auto actual = internal::attr_to_value(it->second);
        if (!actual || !internal::values_match(*expected, *actual)) {
            throw std::runtime_error("DSL model: config mismatch for param " + kv.first);
        }
    }
}

void DslModel::validate_param_shapes(const Module& module) const {
    auto env = make_shape_env(module, /*B=*/1, /*T=*/1);
    for (const auto& kv : module.params) {
        const auto& info = kv.second;
        if (info.shape.empty()) {
            continue;
        }
        auto resolved = resolve_shape(info.shape, env);
        for (const auto dim : resolved) {
            if (dim <= 0) {
                throw std::runtime_error("DSL model: invalid shape for param " + kv.first);
            }
        }
    }
}

} // namespace dsl
