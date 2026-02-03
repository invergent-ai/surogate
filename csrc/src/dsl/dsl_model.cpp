// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL model wrapper (IR validation + execution).

#include "dsl/dsl_model.h"

#include <algorithm>
#include <atomic>
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

#include "dsl/graph_executor.h"
#include "dsl/dsl_runtime.h"
#include "dsl/qlora_provider.h"
#include "dsl/compiled_ops_helpers.h"
#include "dsl/dsl_weight_manager.h"
#include "dsl/dsl_model_internal.h"
#include "kernels/kernels.h"
#include "modules/forward_hooks.h"
#include "modules/backward_hooks.h"
#include "modules/lora/lora_utils.h"
#include "modules/lora/lora_model_utils.h"
#include "modules/qlora/fp8_weight_provider.h"
#include "modules/qlora/fp4_weight_provider.h"
#include "modules/qlora/bnb_weight_provider.h"
#include "modules/composite/transformer_block.h"
#include "models/llama/transformer_block.h"
#include "models/qwen25/transformer_block.h"
#include "models/qwen3/transformer_block.h"
#include "models/qwen3moe/qwen3_moe_block.h"
#include "modules/model_config.h"
#include "modules/optimizers/adamw_8bit.h"
#include "modules/optimizers/normuon.h"
#include "modules/fp8_scaling_state.h"
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
        if (field == "qkv_weight" || field == "out_weight" || field == "o_proj_weight" ||
            field == "mlp_up_weight" || field == "mlp_down_weight" ||
            field == "ln1_weight" || field == "ln2_weight" ||
            field == "q_norm_weight" || field == "k_norm_weight" ||
            field == "router_weight" ||
            field == "experts_gate_up" || field == "experts_down" ||
            field == "shared_expert_gate" || field == "shared_expert_up" ||
            field == "shared_expert_down") {
            return true;
        }
        if (ends_with(field, "in_proj_weight") ||
            ends_with(field, "in_proj_bias") ||
            ends_with(field, "out_proj_weight") ||
            ends_with(field, "out_proj_bias") ||
            ends_with(field, "conv1d_weight") ||
            ends_with(field, "conv1d_bias") ||
            ends_with(field, "A_log") ||
            ends_with(field, "D") ||
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

template<typename Block>
class DslQLoRAWeightProvider final : public QLoRAWeightProvider {
public:
    DslQLoRAWeightProvider(const modules::ModelConfig& cfg,
                           const RuntimeOptions& options,
                           const modules::ModularLoRAConfig& lora_config,
                           const modules::QLoRAConfig& qlora_config,
                           const std::shared_ptr<TensorAllocator>& allocator)
        : mConfig(cfg),
          mOptions(options),
          mLoRAConfig(lora_config),
          mQLoRAConfig(qlora_config),
          mAllocator(allocator) {}

    bool handles_param(std::string_view name) const override {
        return is_qlora_param_name(name, mLoRAConfig.train_router);
    }

    Tensor& resolve_param(std::string_view name, cudaStream_t stream) override {
        if (!mFP8Provider && !mFP4Provider && !mBnBProvider) {
            throw std::runtime_error("DSL QLoRA provider: weights not initialized (import_weights_qlora not called)");
        }

        const std::string_view clean = trim_optional(name);
        auto trace_nan = [&](Tensor& t, std::string_view param_name, int layer_idx) {
            const int trace = env_int("SUROGATE_QLORA_NAN_TRACE", 0);
            if (!trace || !t.Data) {
                return;
            }
            if (internal::stream_is_capturing(stream)) {
                return;
            }
            const int trace_layer = env_int("SUROGATE_QLORA_NAN_LAYER", -1);
            if (trace_layer >= 0 && trace_layer != layer_idx) {
                return;
            }
            static std::atomic<int> nan_count{0};
            const int limit = env_int("SUROGATE_QLORA_NAN_LIMIT", 8);
            if (limit > 0) {
                const int idx = nan_count.fetch_add(1);
                if (idx >= limit) {
                    return;
                }
            }
            const int full_scan = env_int("SUROGATE_QLORA_NAN_FULL", 0);
            const long rows = (t.Rank > 0) ? static_cast<long>(t.Sizes[0]) : 1;
            bool has_nan = false;
            long row = -1;
            float min_val = 0.0f;
            float max_val = 0.0f;
            if (full_scan) {
                has_nan = find_first_nan_row(t, &row, &min_val, &max_val);
            } else {
                const long sample_rows[3] = {0, rows > 0 ? rows / 2 : 0, rows > 0 ? rows - 1 : 0};
                for (long r : sample_rows) {
                    if (r < 0 || r >= rows) {
                        continue;
                    }
                    if (tensor_row_has_nan_or_inf(t, r, &min_val, &max_val)) {
                        row = r;
                        has_nan = true;
                        break;
                    }
                }
            }
            if (!has_nan) {
                return;
            }
            std::cerr << fmt::format("[QLORA_NAN_WEIGHT] name={} layer={} row={} min={} max={} dtype={}\n",
                                     param_name, layer_idx, row, min_val, max_val,
                                     static_cast<int>(t.DType));
            if (env_int("SUROGATE_QLORA_NAN_ABORT", 0)) {
                throw std::runtime_error("QLoRA weight contains NaN/Inf");
            }
        };
        auto trace_return = [&](Tensor& t, std::string_view param_name, int layer_idx) -> Tensor& {
            trace_nan(t, param_name, layer_idx);
            return t;
        };
        if (clean == "embedding" || clean == "embeddings" || clean == "embed_tokens") {
            Tensor& t = with_provider([&](auto& provider) -> Tensor& { return provider.get_embeddings(stream); });
            return trace_return(t, clean, -1);
        }
        if (clean == "final_norm" || clean == "final_norm_weight" || clean == "norm") {
            Tensor& t = with_provider([&](auto& provider) -> Tensor& { return provider.get_final_norm(stream); });
            return trace_return(t, clean, -1);
        }
        if (clean == "lm_head" || clean == "lm_head_weight") {
            Tensor& t = with_provider([&](auto& provider) -> Tensor& { return provider.get_lm_head(stream); });
            return trace_return(t, clean, -1);
        }

        int layer_idx = -1;
        std::string field;
        if (!internal::parse_block_param(clean, layer_idx, field)) {
            throw std::runtime_error("DSL QLoRA provider: unknown param " + std::string(name));
        }
        if (!field.empty() && field.back() == '?') {
            field.pop_back();
        }

        auto& block = with_provider([&](auto& provider) -> typename std::remove_reference_t<decltype(provider)>::BlockWeights& {
            return provider.get_block(layer_idx, stream);
        });

        if (field == "ln1_weight") {
            return trace_return(block.ln1.weight, clean, layer_idx);
        }
        if (field == "ln2_weight") {
            return trace_return(block.ln2.weight, clean, layer_idx);
        }
        if (field == "qkv_weight") {
            return trace_return(block.attention.qkv_weight, clean, layer_idx);
        }
        if (field == "out_weight" || field == "o_proj_weight") {
            return trace_return(block.attention.out_weight, clean, layer_idx);
        }
        if (field == "mlp_up_weight") {
            if constexpr (requires { block.mlp_up_weight; }) {
                return trace_return(block.mlp_up_weight, clean, layer_idx);
            }
            throw std::runtime_error("DSL QLoRA provider: mlp_up_weight not available for " + std::string(name));
        }
        if (field == "mlp_down_weight") {
            if constexpr (requires { block.mlp_down_weight; }) {
                return trace_return(block.mlp_down_weight, clean, layer_idx);
            }
            throw std::runtime_error("DSL QLoRA provider: mlp_down_weight not available for " + std::string(name));
        }
        if (field == "q_norm_weight") {
            if constexpr (requires { block.attention.q_norm_weight; }) {
                if (block.attention.q_norm_weight.has_value()) {
                    return trace_return(block.attention.q_norm_weight.value(), clean, layer_idx);
                }
            }
            throw std::runtime_error("DSL QLoRA provider: q_norm_weight not available for " + std::string(name));
        }
        if (field == "k_norm_weight") {
            if constexpr (requires { block.attention.k_norm_weight; }) {
                if (block.attention.k_norm_weight.has_value()) {
                    return trace_return(block.attention.k_norm_weight.value(), clean, layer_idx);
                }
            }
            throw std::runtime_error("DSL QLoRA provider: k_norm_weight not available for " + std::string(name));
        }

        if (field == "router_weight") {
            if constexpr (requires { block.router.gate; }) {
                return trace_return(block.router.gate, clean, layer_idx);
            }
            throw std::runtime_error("DSL QLoRA provider: router_weight not available for " + std::string(name));
        }
        if (field == "experts_gate_up") {
            if constexpr (requires { block.experts.gate_up_proj; }) {
                return trace_return(block.experts.gate_up_proj, clean, layer_idx);
            }
            throw std::runtime_error("DSL QLoRA provider: experts_gate_up not available for " + std::string(name));
        }
        if (field == "experts_down") {
            if constexpr (requires { block.experts.down_proj; }) {
                return trace_return(block.experts.down_proj, clean, layer_idx);
            }
            throw std::runtime_error("DSL QLoRA provider: experts_down not available for " + std::string(name));
        }
        if (field == "shared_expert_gate") {
            if constexpr (requires { block.shared_expert; }) {
                if (block.shared_expert.has_value()) {
                    return trace_return(block.shared_expert->gate_proj, clean, layer_idx);
                }
            }
            throw std::runtime_error("DSL QLoRA provider: shared_expert_gate not available for " + std::string(name));
        }
        if (field == "shared_expert_up") {
            if constexpr (requires { block.shared_expert; }) {
                if (block.shared_expert.has_value()) {
                    return trace_return(block.shared_expert->up_proj, clean, layer_idx);
                }
            }
            throw std::runtime_error("DSL QLoRA provider: shared_expert_up not available for " + std::string(name));
        }
        if (field == "shared_expert_down") {
            if constexpr (requires { block.shared_expert; }) {
                if (block.shared_expert.has_value()) {
                    return trace_return(block.shared_expert->down_proj, clean, layer_idx);
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
        if (ends_with(field, "conv1d_weight")) {
            if constexpr (requires { block.mamba; }) {
                if (block.mamba.has_value()) {
                    return block.mamba->conv1d_weight;
                }
            }
            throw std::runtime_error("DSL QLoRA provider: conv1d_weight not available for " + std::string(name));
        }
        if (ends_with(field, "conv1d_bias")) {
            if constexpr (requires { block.mamba; }) {
                if (block.mamba.has_value() && block.mamba->conv1d_bias.has_value()) {
                    return block.mamba->conv1d_bias.value();
                }
            }
            throw std::runtime_error("DSL QLoRA provider: conv1d_bias not available for " + std::string(name));
        }
        if (ends_with(field, "A_log")) {
            if constexpr (requires { block.mamba; }) {
                if (block.mamba.has_value()) {
                    return block.mamba->A_log;
                }
            }
            throw std::runtime_error("DSL QLoRA provider: A_log not available for " + std::string(name));
        }
        if (ends_with(field, "D")) {
            if constexpr (requires { block.mamba; }) {
                if (block.mamba.has_value()) {
                    return block.mamba->D;
                }
            }
            throw std::runtime_error("DSL QLoRA provider: D not available for " + std::string(name));
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

    void fill_mamba_config(auto& cfg) {
        cfg.layer_is_mamba.resize(mConfig.NumLayers);
        cfg.has_mamba = false;
        for (int i = 0; i < mConfig.NumLayers; ++i) {
            const bool is_mamba = (mConfig.get_block_type(i) == modules::BlockType::Mamba);
            cfg.layer_is_mamba[i] = static_cast<std::uint8_t>(is_mamba ? 1 : 0);
            cfg.has_mamba = cfg.has_mamba || is_mamba;
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
            std::cerr << "[DSL QLoRA] MoE selective expert dequant disabled "
                      << "(DSL path requires full expert weights for base MoE GEMM)\n";
            if (mOptions.OffloadExperts) {
                std::cerr << "[DSL QLoRA] Offload experts enabled; full expert dequant will stream all experts.\n";
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

    std::unique_ptr<FP8Provider> mFP8Provider;
    std::unique_ptr<FP4Provider> mFP4Provider;
    std::unique_ptr<BnBProvider> mBnBProvider;
};

}  // namespace

namespace internal {

std::unique_ptr<QLoRAWeightProvider> create_dsl_qlora_provider(
    const PretrainedConfig& config,
    const modules::ModelConfig& model_cfg,
    const RuntimeOptions& options,
    const modules::ModularLoRAConfig& lora_cfg,
    const modules::QLoRAConfig& qlora_cfg,
    const std::shared_ptr<TensorAllocator>& allocator) {
    switch (config.Architecture) {
        case PretrainedConfig::QWEN3_MOE:
            return std::make_unique<DslQLoRAWeightProvider<modules::Qwen3MoEBlock>>(
                model_cfg, options, lora_cfg, qlora_cfg, allocator);
        case PretrainedConfig::QWEN3:
            return std::make_unique<DslQLoRAWeightProvider<modules::Qwen3TransformerBlock>>(
                model_cfg, options, lora_cfg, qlora_cfg, allocator);
        case PretrainedConfig::QWEN2:
            return std::make_unique<DslQLoRAWeightProvider<modules::Qwen2TransformerBlock>>(
                model_cfg, options, lora_cfg, qlora_cfg, allocator);
        case PretrainedConfig::NEMOTRON_H:
            return std::make_unique<DslQLoRAWeightProvider<modules::DenseTransformerBlock<>>>(
                model_cfg, options, lora_cfg, qlora_cfg, allocator);
        case PretrainedConfig::LLAMA:
        default:
            return std::make_unique<DslQLoRAWeightProvider<modules::LlamaTransformerBlock>>(
                model_cfg, options, lora_cfg, qlora_cfg, allocator);
    }
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
      mModelConfig(modules::ModelConfig::from_pretrained_config(config)),
      mQLoRAConfig(qlora_config),
      mShardIdx(shard_idx),
      mNumShards(num_shards) {
    if (ir_json.empty()) {
        throw std::runtime_error("DSL model: IR JSON is empty");
    }
    nlohmann::json root = nlohmann::json::parse(ir_json);
    mIr = load_ir_from_json(root);
    if (!mIr.success) {
        throw std::runtime_error("DSL model: IR JSON indicates compilation failure");
    }
    mModule = &pick_model_module(mIr);
    validate_ir();

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
