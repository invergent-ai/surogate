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
#include "runtime/qlora/generic_qlora_provider.h"
#include "runtime/qlora/dsl_qlora_pipeline.h"
#include "runtime/dsl/graph_executor_utils.h"
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

/// Convert DslModel::MappingSpec to dsl::MappingSpec for the generic pipeline.
MappingSpec to_pipeline_mapping(const DslModel::MappingSpec& src) {
    MappingSpec dst;
    using SK = DslModel::MappingSpec::Kind;
    using DK = MappingSpec::Kind;
    switch (src.kind) {
        case SK::Direct:       dst.kind = DK::Direct; break;
        case SK::Fuse:         dst.kind = DK::Fuse; break;
        case SK::Split:        dst.kind = DK::Split; break;
        case SK::Transform:    dst.kind = DK::Transform; break;
        case SK::TiedTo:       dst.kind = DK::TiedTo; break;
        case SK::StackExperts: dst.kind = DK::StackExperts; break;
        default:               dst.kind = DK::Unknown; break;
    }
    dst.source = src.source;
    dst.sources = src.sources;
    dst.ranges = src.ranges;
    dst.fn = src.fn;
    dst.target = src.target;
    dst.dim = src.dim;
    dst.optional = src.optional;
    dst.fuse_gate_up = src.fuse_gate_up;
    dst.num_experts = src.num_experts;
    return dst;
}

/// Build a MappingTable from DslModel's parsed HF mapping.
MappingTable build_mapping_table(
    const std::unordered_map<std::string, DslModel::MappingSpec>& hf_mapping) {
    MappingTable table;
    table.reserve(hf_mapping.size());
    for (const auto& kv : hf_mapping) {
        table.emplace(kv.first, to_pipeline_mapping(kv.second));
    }
    return table;
}

/// Build QuantizerConfig from QLoRAConfig and runtime options.
qlora::QuantizerConfig build_quantizer_config(
    const modules::QLoRAConfig& qlora_cfg,
    const RuntimeOptions& options) {
    qlora::QuantizerConfig qcfg;

    if (qlora_cfg.is_bnb()) {
        qcfg.format = qlora::QuantFormat::BNB_NF4;
        qcfg.block_size = qlora_cfg.block_size() > 0 ? qlora_cfg.block_size() : 64;
        qcfg.double_quant = qlora_cfg.bnb_double_quant;
        qcfg.double_quant_group_size = qlora_cfg.bnb_double_quant_group_size;
    } else if (qlora_cfg.is_fp8()) {
        qcfg.format = qlora::QuantFormat::FP8_PER_BLOCK;
        qcfg.block_size = qlora_cfg.block_size() > 0 ? qlora_cfg.block_size() : 128;
        qcfg.enable_fp8_forward = options.fp8_forward_enabled();
        qcfg.enable_fp8_hybrid = options.fp8_hybrid_enabled();
    } else if (qlora_cfg.is_fp4()) {
        qcfg.format = qlora::QuantFormat::FP4_BLOCK_2D;
        qcfg.block_size = qlora_cfg.block_size() > 0 ? qlora_cfg.block_size() : 16;
    } else {
        qcfg.format = qlora::QuantFormat::NONE;
    }

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
    qcfg.device_id = device_id;
    qcfg.sm_version = props.major * 10 + props.minor;

    return qcfg;
}

/// Build WeightLoadSpec list from the forward graph parameters.
///
/// Iterates the IR's forward graph params, resolves shapes using the module's
/// config, and creates a WeightLoadSpec for each QLoRA-managed parameter.
std::vector<qlora::WeightLoadSpec> build_weight_specs(
    const Module& module,
    const modules::ModularLoRAConfig& lora_cfg) {
    if (!module.forward.has_value()) {
        return {};
    }

    const auto& graph = module.forward.value();
    ShapeEnv env = make_shape_env(module, /*B=*/1, /*T=*/1);
    augment_shape_env(env, module.config);

    const bool train_router = lora_cfg.enabled() && lora_cfg.train_router;

    std::vector<qlora::WeightLoadSpec> specs;
    specs.reserve(graph.params.size());

    for (const auto& kv : graph.params) {
        const std::string& name = kv.first;
        const TensorInfo& info = kv.second;

        // Only include QLoRA-managed parameters
        if (!is_qlora_param_name(name, train_router)) {
            continue;
        }

        // Resolve shape from IR dimensions
        auto resolved = resolve_shape(info.shape, env);
        if (resolved.empty()) {
            continue;
        }

        qlora::WeightLoadSpec spec;
        spec.name = name;
        spec.M = static_cast<int>(resolved[0]);
        spec.K = resolved.size() >= 2 ? static_cast<int>(resolved[1]) : 0;
        spec.quantize = info.quantizable;
        spec.offload_group = info.offload_group;
        spec.sharded = false;  // QLoRA base weights are replicated (not sharded)

        specs.push_back(std::move(spec));
    }

    return specs;
}

}  // namespace

namespace internal {

std::unique_ptr<QLoRAWeightProvider> create_dsl_qlora_provider(
    const Module& module,
    const modules::ModelConfig& model_cfg,
    const PretrainedConfig& pt_config,
    const RuntimeOptions& options,
    const modules::ModularLoRAConfig& lora_cfg,
    const modules::QLoRAConfig& qlora_cfg,
    const std::shared_ptr<TensorAllocator>& allocator,
    const std::unordered_map<std::string, DslModel::MappingSpec>& hf_mapping,
    int shard_idx,
    int num_shards) {

    // Build the pipeline configuration
    qlora::DslQLoRAPipelineConfig config;
    config.mapping = build_mapping_table(hf_mapping);
    config.weight_specs = build_weight_specs(module, lora_cfg);
    config.quantizer_config = build_quantizer_config(qlora_cfg, options);
    config.shard_idx = shard_idx;
    config.num_shards = num_shards;
    config.num_experts = qlora_cfg.num_experts;
    config.moe_intermediate_size = qlora_cfg.moe_intermediate_size;

    // Configure weight manager
    config.weight_manager_config.device_id = config.quantizer_config.device_id;
    if (options.OffloadExperts && qlora_cfg.is_moe()) {
        config.weight_manager_config.enable_offloading = true;
    }

    // Use pooled dequant buffers for large models to reduce peak memory
    const int num_quantizable = static_cast<int>(std::count_if(
        config.weight_specs.begin(), config.weight_specs.end(),
        [](const qlora::WeightLoadSpec& s) { return s.quantize; }));
    if (num_quantizable > 64) {
        // For models with many quantizable weights, use a pool to limit
        // concurrent dequant buffers and reduce peak GPU memory
        config.weight_manager_config.max_dequant_cache_size = 32;
    }

    fprintf(stderr, "[QLoRA] Generic provider: %d weight specs (%d quantizable), "
                    "format=%s, shard=%d/%d\n",
            static_cast<int>(config.weight_specs.size()),
            num_quantizable,
            qlora_cfg.is_bnb() ? "BnB-NF4" :
            qlora_cfg.is_fp8() ? "FP8" :
            qlora_cfg.is_fp4() ? "FP4" : "none",
            shard_idx, num_shards);

    return std::make_unique<qlora::GenericQLoRAProvider>(
        std::move(config), pt_config, allocator);
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
