// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_MODEL_FACTORY_H
#define SUROGATE_SRC_MODULES_MODEL_FACTORY_H

#include <memory>
#include <string>
#include <stdexcept>

#include "model_config.h"
#include "model/modular_model.h"
#include "composite/transformer_block.h"
#include "moe/moe_block.h"

// Model class headers for inheritance-based dispatch
#include "models/llama/llama_model.h"
#include "models/qwen25/qwen25_model.h"
#include "models/qwen3/qwen3_model.h"
#include "models/qwen3moe/qwen3_moe_model.h"
#include "models/nemotron_h/nemotron_h_model.h"
#include "models/llama/transformer_block.h"
#include "models/qwen25/transformer_block.h"
#include "models/qwen3/transformer_block.h"
#include "models/qwen3moe/qwen3_moe_block.h"

#include "dsl/dsl_model.h"

#include "training/model.h"
#include "utilities/allocator.h"
#include "lora/lora_model.h"
#include "lora/lora_config.h"
#include "qlora/qlora_config.h"

namespace modules {

/**
 * @brief Factory for creating modular transformer models
 *
 * Provides a unified interface for creating models with different architectures
 * and configurations.
 */
class ModelFactory {
public:
    /**
     * @brief Create a model from configuration
     *
     * Creates the appropriate model type based on the configuration:
     * - Dense architectures use ModularTransformerModel<DenseTransformerBlock>
     * - MoE architectures will use ModularTransformerModel<MoETransformerBlock> (future)
     * - Hybrid architectures will use mixed block types (future)
     *
     * @param config Model configuration
     * @param options Runtime options
     * @param rank Process rank for sharding
     * @param world World size
     * @param alloc Optional tensor allocator
     * @return Unique pointer to IModel interface
     */
    static std::unique_ptr<IModel> create(
        const ModelConfig& config,
        const ModelOptions& options,
        int rank,
        int world,
        const std::shared_ptr<TensorAllocator>& alloc = nullptr) {

        switch (config.architecture) {
            case ArchitectureType::Dense:
                return create_dense_model(config, options, rank, world, alloc);

            case ArchitectureType::MoE:
                return create_moe_model(config, options, rank, world, alloc);

            case ArchitectureType::Hybrid:
                return create_hybrid_model(config, options, rank, world, alloc);

            default:
                throw std::invalid_argument("Unknown architecture type");
        }
    }

    /**
     * @brief Create a model from PretrainedConfig using inheritance-based dispatch
     *
     * Dispatches to the correct model class based on the config's architecture ID:
     * - QWEN3_MOE -> Qwen3MoEModel or Qwen3HybridModel
     * - QWEN3     -> Qwen3Model
     * - QWEN2     -> Qwen2Model
     * - LLAMA     -> LlamaModel
     *
     * This follows the Transformers-like inheritance pattern where config type
     * determines model class, enabling model-specific behaviors.
     *
     * @param config PretrainedConfig (or derived) instance
     * @param options Runtime options
     * @param rank Process rank for sharding
     * @param world World size
     * @param alloc Optional tensor allocator
     * @return Unique pointer to model (as IModel)
     */
    static std::unique_ptr<IModel> create_from_pretrained_config(
        const PretrainedConfig& config,
        const RuntimeOptions& options,
        int rank,
        int world,
        const std::shared_ptr<TensorAllocator>& alloc = nullptr) {

        if (options.UseDslIr) {
            if (options.DslIrJson.empty()) {
                throw std::runtime_error("DSL IR enabled but no IR JSON provided in RuntimeOptions");
            }
            RuntimeOptions backend_options = options;
            backend_options.UseDslIr = false;
            backend_options.DslIrJson.clear();
            auto backend = create_from_pretrained_config(config, backend_options, rank, world, alloc);
            return std::make_unique<dsl::DslModel>(config, options, options.DslIrJson, alloc, std::move(backend));
        }

        ModelConfig mod_config = ModelConfig::from_pretrained_config(config);
        ModelOptions mod_options = ModelOptions::from_runtime_options(options);

        // Dispatch based on config architecture ID (most specific first)
        switch (config.Architecture) {
            case PretrainedConfig::QWEN3_MOE: {
                // Qwen3 MoE - use dynamic_cast for additional config fields
                if (const auto* moe_cfg = dynamic_cast<const Qwen3MoEConfig*>(&config)) {
                    return create_qwen3_moe_model(*moe_cfg, mod_options, rank, world, alloc);
                }
                // Fallback if dynamic_cast fails but Architecture says MoE
                return create_moe_model(mod_config, mod_options, rank, world, alloc);
            }

            case PretrainedConfig::QWEN3:
                return std::make_unique<Qwen3Model>(mod_config, mod_options, rank, world, alloc);

            case PretrainedConfig::QWEN2:
                return std::make_unique<Qwen2Model>(mod_config, mod_options, rank, world, alloc);

            case PretrainedConfig::NEMOTRON_H:
                return std::make_unique<NemotronHModel>(mod_config, mod_options, rank, world, alloc);

            case PretrainedConfig::LLAMA:
            default:
                return std::make_unique<LlamaModel>(mod_config, mod_options, rank, world, alloc);
        }
    }

    /**
     * @brief Create a LoRA-wrapped model from PretrainedConfig
     *
     * Dispatches to the correct model + LoRA wrapper based on the config's architecture ID.
     * This centralizes the type-dispatching logic that would otherwise be duplicated
     * across callers (py_train.cpp, etc.).
     *
     * @param config PretrainedConfig (or derived) instance
     * @param lora_config LoRA configuration (rank, alpha, targets, etc.)
     * @param options Runtime options
     * @param comm NCCL communicator for distributed setup
     * @param alloc Tensor allocator
     * @param qlora_config Optional QLoRA configuration for quantized base weights
     * @return Unique pointer to LoRA-wrapped model (as IModel)
     */
    static std::unique_ptr<IModel> create_lora_from_pretrained_config(
        const PretrainedConfig& config,
        const ModularLoRAConfig& lora_config,
        const RuntimeOptions& options,
        NCCLCommunicator& comm,
        const std::shared_ptr<TensorAllocator>& alloc,
        const QLoRAConfig& qlora_config = QLoRAConfig{}) {

        if (options.UseDslIr) {
            if (options.DslIrJson.empty()) {
                throw std::runtime_error("DSL IR enabled but no IR JSON provided in RuntimeOptions");
            }
            RuntimeOptions backend_options = options;
            backend_options.UseDslIr = false;
            backend_options.DslIrJson.clear();
            auto backend = create_lora_from_pretrained_config(config, lora_config, backend_options,
                                                              comm, alloc, qlora_config);
            return std::make_unique<dsl::DslModel>(config, options, options.DslIrJson, alloc, std::move(backend));
        }

        ModelConfig mod_config = ModelConfig::from_pretrained_config(config);
        ModelOptions mod_options = ModelOptions::from_runtime_options(options);

        // QLoRA: skip block weight allocation since weights are provided by QLoRA weight provider
        if (qlora_config.is_quantized()) {
            mod_options.skip_block_allocation = true;
        }

        // Dispatch based on config architecture ID (most specific first)
        switch (config.Architecture) {
            case PretrainedConfig::QWEN3_MOE: {
                // Qwen3 MoE
                using Block = Qwen3MoEBlock;
                auto base = std::make_unique<ModularTransformerModel<Block>>(
                    mod_config, mod_options, comm.rank(), comm.world_size(), alloc);
                return std::make_unique<ModularLoRAModel<Block>>(
                    std::move(base), lora_config, options, comm, alloc, qlora_config);
            }

            case PretrainedConfig::QWEN3: {
                // Qwen3 dense
                using Block = Qwen3TransformerBlock;
                auto base = std::make_unique<ModularTransformerModel<Block>>(
                    mod_config, mod_options, comm.rank(), comm.world_size(), alloc);
                return std::make_unique<ModularLoRAModel<Block>>(
                    std::move(base), lora_config, options, comm, alloc, qlora_config);
            }

            case PretrainedConfig::QWEN2: {
                // Qwen2 dense
                using Block = Qwen2TransformerBlock;
                auto base = std::make_unique<ModularTransformerModel<Block>>(
                    mod_config, mod_options, comm.rank(), comm.world_size(), alloc);
                return std::make_unique<ModularLoRAModel<Block>>(
                    std::move(base), lora_config, options, comm, alloc, qlora_config);
            }

            case PretrainedConfig::NEMOTRON_H: {
                using Block = DenseTransformerBlock<>;
                auto base = std::make_unique<ModularTransformerModel<Block>>(
                    mod_config, mod_options, comm.rank(), comm.world_size(), alloc);
                return std::make_unique<ModularLoRAModel<Block>>(
                    std::move(base), lora_config, options, comm, alloc, qlora_config);
            }

            case PretrainedConfig::LLAMA:
            default: {
                // LLaMA dense
                using Block = LlamaTransformerBlock;
                auto base = std::make_unique<ModularTransformerModel<Block>>(
                    mod_config, mod_options, comm.rank(), comm.world_size(), alloc);
                return std::make_unique<ModularLoRAModel<Block>>(
                    std::move(base), lora_config, options, comm, alloc, qlora_config);
            }
        }
    }

    /**
     * @brief Try to cast a model to a LoRA model and invoke a callback
     *
     * Helper for operations that need access to the typed ModularLoRAModel
     * (e.g., export_adapter, get_lora_gradients). Returns true if cast succeeded.
     *
     * @tparam Callback Callable taking ModularLoRAModel<Block>*
     * @param model The IModel to try casting
     * @param callback The callback to invoke if cast succeeds
     * @return true if any cast succeeded and callback was invoked
     */
    template<typename Callback>
    static bool try_lora_model(IModel* model, Callback&& callback) {
        // If this is a DslModel wrapper, unwrap to get the backend
        if (auto* dsl_model = dynamic_cast<dsl::DslModel*>(model)) {
            if (auto* backend = dsl_model->get_backend()) {
                return try_lora_model(backend, std::forward<Callback>(callback));
            }
            return false;
        }

        // Try dense block types (most specific first)
        if (auto* lora = dynamic_cast<ModularLoRAModel<Qwen3TransformerBlock>*>(model)) {
            callback(lora);
            return true;
        }
        if (auto* lora = dynamic_cast<ModularLoRAModel<Qwen2TransformerBlock>*>(model)) {
            callback(lora);
            return true;
        }
        if (auto* lora = dynamic_cast<ModularLoRAModel<LlamaTransformerBlock>*>(model)) {
            callback(lora);
            return true;
        }
        if (auto* lora = dynamic_cast<ModularLoRAModel<DenseTransformerBlock<>>*>(model)) {
            callback(lora);
            return true;
        }
        // Try MoE block types
        if (auto* lora = dynamic_cast<ModularLoRAModel<Qwen3MoEBlock>*>(model)) {
            callback(lora);
            return true;
        }
        if (auto* lora = dynamic_cast<ModularLoRAModel<StandardMoEBlock>*>(model)) {
            callback(lora);
            return true;
        }
        return false;
    }

private:
    /**
     * @brief Create a dense transformer model
     *
     * Dispatches to the correct model class based on ModelConfig's architecture
     * hint or uses inheritance-based selection when available.
     *
     * Model hierarchy for dense architectures:
     * - LlamaModel (default, uses LlamaTransformerBlock)
     * - Qwen2Model (uses Qwen2TransformerBlock with sliding window support)
     * - Qwen3Model (uses Qwen3TransformerBlock with QK normalization)
     */
    static std::unique_ptr<IModel> create_dense_model(
        const ModelConfig& config,
        const ModelOptions& options,
        int rank,
        int world,
        const std::shared_ptr<TensorAllocator>& alloc) {

        // Dispatch based on config hints
        // The config may have been converted from a PretrainedConfig, so we check
        // for model-specific features to select the correct block type.

        // Check for Qwen3-specific features (QK normalization)
        if (config.use_qk_norm) {
            return std::make_unique<Qwen3Model>(config, options, rank, world, alloc);
        }

        // Check for Qwen2-specific features (sliding window)
        if (config.use_sliding_window && config.sliding_window_size > 0) {
            return std::make_unique<Qwen2Model>(config, options, rank, world, alloc);
        }

        // Check for QKV bias (common in Qwen2, but also can be in others)
        // This is a weaker signal, so we use LlamaModel which handles bias correctly
        if (config.UseQKVBias) {
            // Qwen2 typically has bias, but if no sliding window, it might be
            // a variant - use Qwen2 block anyway for proper bias handling
            return std::make_unique<Qwen2Model>(config, options, rank, world, alloc);
        }

        // Default to LlamaModel
        return std::make_unique<LlamaModel>(config, options, rank, world, alloc);
    }

    /**
     * @brief Create a Mixture-of-Experts model
     *
     * Supports various MoE configurations:
     * - Qwen3 MoE (Qwen3MoEModel with Qwen3RouterModule)
     * - Standard top-k routing (Mixtral-style)
     * - Switch routing (top-1)
     * - With or without shared expert (Nemotron/DeepSeek style)
     *
     * Model hierarchy for MoE architectures:
     * - Qwen3MoEModel (uses Qwen3MoEBlock with Qwen3Router)
     * - StandardMoEBlock (generic MoE with TopKRouter)
     * - SwitchTransformerBlock (top-1 routing)
     */
    static std::unique_ptr<IModel> create_moe_model(
        const ModelConfig& config,
        const ModelOptions& options,
        int rank,
        int world,
        const std::shared_ptr<TensorAllocator>& alloc) {

        if (!config.moe_config.has_value()) {
            throw std::invalid_argument("MoE architecture requires moe_config");
        }

        const auto& moe_cfg = config.moe_config.value();

        // Check for Qwen3-specific features (QK norm) to use Qwen3MoEModel
        if (config.use_qk_norm) {
            return std::make_unique<Qwen3MoEModel>(config, options, rank, world, alloc);
        }

        // Select router type based on top_k
        if (moe_cfg.top_k == 1) {
            // Switch Transformer style (top-1 routing)
            using SwitchBlock = SwitchTransformerBlock;
            return std::make_unique<ModularTransformerModel<SwitchBlock>>(
                config, options, rank, world, alloc);
        } else {
            // Standard top-k routing (Mixtral style)
            using MoEBlock = StandardMoEBlock;
            return std::make_unique<ModularTransformerModel<MoEBlock>>(
                config, options, rank, world, alloc);
        }
    }

    /**
     * @brief Create a hybrid model (per-layer Attention/MLP/Mamba)
     *
     * Note: Mixed MoE/Dense per-layer routing is not supported yet.
     */
    static std::unique_ptr<IModel> create_hybrid_model(
        const ModelConfig& config,
        const ModelOptions& options,
        int rank,
        int world,
        const std::shared_ptr<TensorAllocator>& alloc) {

        // Hybrid path: use the composable DenseTransformerBlock and per-layer specs,
        // including optional MoE/SSM layers.
        using Block = DenseTransformerBlock<>;
        return std::make_unique<ModularTransformerModel<Block>>(
            config, options, rank, world, alloc);
    }
};

/**
 * @brief Configuration builder for creating models
 *
 * Provides a fluent interface for building model configurations.
 *
 * Example:
 * @code
 * auto config = ModelConfigBuilder()
 *     .architecture(ArchitectureType::Dense)
 *     .hidden_size(4096)
 *     .num_layers(32)
 *     .num_heads(32)
 *     .build();
 * @endcode
 */
class ModelConfigBuilder {
public:
    ModelConfigBuilder() = default;

    ModelConfigBuilder& architecture(ArchitectureType arch) {
        mConfig.architecture = arch;
        return *this;
    }

    ModelConfigBuilder& activation(ActivationType act) {
        mConfig.activation_type = act;
        return *this;
    }

    ModelConfigBuilder& norm(NormType norm) {
        mConfig.norm_type = norm;
        return *this;
    }

    ModelConfigBuilder& attention(AttentionType att) {
        mConfig.attention_type = att;
        return *this;
    }

    ModelConfigBuilder& hidden_size(int size) {
        mConfig.HiddenSize = size;
        return *this;
    }

    ModelConfigBuilder& intermediate_size(int size) {
        mConfig.IntermediateSize = size;
        return *this;
    }

    ModelConfigBuilder& vocab_size(int size) {
        mConfig.VocabSize = size;
        return *this;
    }

    ModelConfigBuilder& num_layers(int layers) {
        mConfig.NumLayers = layers;
        return *this;
    }

    ModelConfigBuilder& num_query_heads(int heads) {
        mConfig.NumQueryHeads = heads;
        return *this;
    }

    ModelConfigBuilder& num_kv_heads(int heads) {
        mConfig.NumKeyValHeads = heads;
        return *this;
    }

    ModelConfigBuilder& max_position_embeddings(int max_pos) {
        mConfig.MaxPositionEmbeddings = max_pos;
        return *this;
    }

    ModelConfigBuilder& rope_theta(float theta) {
        mConfig.RopeTheta = theta;
        return *this;
    }

    ModelConfigBuilder& rms_norm_eps(float eps) {
        mConfig.RmsNormEps = eps;
        return *this;
    }

    ModelConfigBuilder& tied_embeddings(bool tied) {
        mConfig.TiedWordEmbeddings = tied;
        return *this;
    }

    ModelConfigBuilder& use_qkv_bias(bool bias) {
        mConfig.UseQKVBias = bias;
        return *this;
    }

    ModelConfigBuilder& dtype(ETensorDType dt) {
        mConfig.DType = dt;
        return *this;
    }

    ModelConfigBuilder& moe(const MoEConfig& moe_cfg) {
        mConfig.moe_config = moe_cfg;
        if (mConfig.architecture == ArchitectureType::Dense) {
            mConfig.architecture = ArchitectureType::MoE;
        }
        return *this;
    }

    ModelConfigBuilder& layer_override(const LayerOverride& override) {
        mConfig.layer_overrides.push_back(override);
        if (mConfig.architecture == ArchitectureType::Dense) {
            mConfig.architecture = ArchitectureType::Hybrid;
        }
        return *this;
    }

    ModelConfigBuilder& parallel_residual(bool parallel) {
        mConfig.use_parallel_residual = parallel;
        return *this;
    }

    ModelConfigBuilder& qk_norm(bool norm) {
        mConfig.use_qk_norm = norm;
        return *this;
    }

    ModelConfigBuilder& sliding_window(int window_size) {
        mConfig.use_sliding_window = true;
        mConfig.sliding_window_size = window_size;
        return *this;
    }

    ModelConfigBuilder& rope_scaling(float factor, const std::string& type = "linear") {
        mConfig.rope_scaling_factor = factor;
        mConfig.rope_type = type;
        return *this;
    }

    ModelConfig build() const {
        return mConfig;
    }

private:
    ModelConfig mConfig;
};

/**
 * @brief Options builder for creating model options
 */
class ModelOptionsBuilder {
public:
    ModelOptionsBuilder() = default;

    ModelOptionsBuilder& recompute_swiglu(bool v) { mOptions.recompute_swiglu = v; return *this; }
    ModelOptionsBuilder& recompute_rmsnorm(bool v) { mOptions.recompute_rmsnorm = v; return *this; }
    ModelOptionsBuilder& recompute_ffn(bool v) { mOptions.recompute_ffn = v; return *this; }
    ModelOptionsBuilder& recompute_qkv(bool v) { mOptions.recompute_qkv = v; return *this; }
    ModelOptionsBuilder& recompute_attention(bool v) { mOptions.recompute_attention = v; return *this; }
    ModelOptionsBuilder& recompute_block(bool v) { mOptions.recompute_block = v; return *this; }
    ModelOptionsBuilder& offload_residuals(bool v) { mOptions.offload_residuals = v; return *this; }

    ModelOptionsBuilder& lmhead_chunks(int v) { mOptions.lmhead_chunks = v; return *this; }
    ModelOptionsBuilder& attention_bwd_chunks(int v) { mOptions.attention_bwd_chunks = v; return *this; }

    ModelOptionsBuilder& use_cuda_graphs(bool v) { mOptions.use_cuda_graphs = v; return *this; }
    ModelOptionsBuilder& trigger_timing_events(bool v) { mOptions.trigger_timing_events = v; return *this; }

    ModelOptionsBuilder& offload_master(bool v) { mOptions.offload_master = v; return *this; }
    ModelOptionsBuilder& offload_quants(bool v) { mOptions.offload_quants = v; return *this; }
    ModelOptionsBuilder& offload_optimizer(bool v) { mOptions.offload_optimizer = v; return *this; }
    ModelOptionsBuilder& offload_grads(bool v) { mOptions.offload_grads = v; return *this; }
    ModelOptionsBuilder& use_zero_copy(bool v) { mOptions.use_zero_copy = v; return *this; }
    ModelOptionsBuilder& use_write_combined(bool v) { mOptions.use_write_combined = v; return *this; }

    ModelOptionsBuilder& shard_weights(bool v) { mOptions.shard_weights = v; return *this; }
    ModelOptionsBuilder& shard_gradients(bool v) { mOptions.shard_gradients = v; return *this; }
    ModelOptionsBuilder& use_all_to_all_reduce(bool v) { mOptions.use_all_to_all_reduce = v; return *this; }
    ModelOptionsBuilder& persistent_quants(bool v) { mOptions.persistent_quants = v; return *this; }

    ModelOptionsBuilder& init_projections_to_zero(bool v) { mOptions.init_projections_to_zero = v; return *this; }
    ModelOptionsBuilder& lora_only_mode(bool v) { mOptions.lora_only_mode = v; return *this; }
    ModelOptionsBuilder& skip_base_gradients(bool v) { mOptions.skip_base_gradients = v; return *this; }

    ModelOptionsBuilder& model_dtype(ETensorDType v) { mOptions.model_dtype = v; return *this; }
    ModelOptionsBuilder& matmul_dtype(ETensorDType v) { mOptions.matmul_dtype = v; return *this; }
    ModelOptionsBuilder& gradient_dtype(ETensorDType v) { mOptions.gradient_dtype = v; return *this; }
    ModelOptionsBuilder& master_dtype(ETensorDType v) { mOptions.master_dtype = v; return *this; }

    ModelOptions build() const {
        return mOptions;
    }

private:
    ModelOptions mOptions;
};

// ============================================================================
// Predefined model configurations
// ============================================================================

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MODEL_FACTORY_H
