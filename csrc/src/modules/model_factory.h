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

#include "training/model.h"
#include "utilities/allocator.h"

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

    static std::unique_ptr<IModel> create_from_pretrained_config(
        const PretrainedConfig& config,
        const RuntimeOptions& options,
        int rank,
        int world,
        const std::shared_ptr<TensorAllocator>& alloc = nullptr) {

        ModelConfig mod_config = ModelConfig::from_pretrained_config(config);
        ModelOptions mod_options = ModelOptions::from_runtime_options(options);

        return create(mod_config, mod_options, rank, world, alloc);
    }

    // Backwards-compatible name.
    static std::unique_ptr<IModel> create_from_llama_config(
        const PretrainedConfig& config,
        const RuntimeOptions& options,
        int rank,
        int world,
        const std::shared_ptr<TensorAllocator>& alloc = nullptr) {
        return create_from_pretrained_config(config, options, rank, world, alloc);
    }

private:
    /**
     * @brief Create a dense transformer model
     */
    static std::unique_ptr<IModel> create_dense_model(
        const ModelConfig& config,
        const ModelOptions& options,
        int rank,
        int world,
        const std::shared_ptr<TensorAllocator>& alloc) {

        // Select block type based on activation and norm types
        // For now, default to DenseTransformerBlock with standard components

        using DefaultBlock = DenseTransformerBlock<>;

        return std::make_unique<ModularTransformerModel<DefaultBlock>>(
            config, options, rank, world, alloc);
    }

    /**
     * @brief Create a Mixture-of-Experts model
     *
     * Supports various MoE configurations:
     * - Standard top-k routing (Mixtral-style)
     * - Switch routing (top-1)
     * - With or without shared expert (Nemotron/DeepSeek style)
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
     * @brief Create a hybrid (mixed dense/MoE) model
     */
    static std::unique_ptr<IModel> create_hybrid_model(
        const ModelConfig& config,
        const ModelOptions& options,
        int rank,
        int world,
        const std::shared_ptr<TensorAllocator>& alloc) {

        // TODO: Implement hybrid model with per-layer block selection
        throw std::runtime_error("Hybrid models not yet implemented");

        // Future implementation would require runtime polymorphism
        // or code generation for specific layer patterns
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

namespace presets {

/**
 * @brief LLaMA 7B configuration
 */
inline ModelConfig llama_7b(ETensorDType dtype = ETensorDType::BF16) {
    return ModelConfigBuilder()
        .architecture(ArchitectureType::Dense)
        .activation(ActivationType::SwiGLU)
        .hidden_size(4096)
        .intermediate_size(11008)
        .vocab_size(32000)
        .num_layers(32)
        .num_query_heads(32)
        .num_kv_heads(32)
        .max_position_embeddings(2048)
        .rope_theta(10000.0f)
        .rms_norm_eps(1e-5f)
        .tied_embeddings(false)
        .use_qkv_bias(false)
        .dtype(dtype)
        .build();
}

/**
 * @brief LLaMA 13B configuration
 */
inline ModelConfig llama_13b(ETensorDType dtype = ETensorDType::BF16) {
    return ModelConfigBuilder()
        .architecture(ArchitectureType::Dense)
        .activation(ActivationType::SwiGLU)
        .hidden_size(5120)
        .intermediate_size(13824)
        .vocab_size(32000)
        .num_layers(40)
        .num_query_heads(40)
        .num_kv_heads(40)
        .max_position_embeddings(2048)
        .rope_theta(10000.0f)
        .rms_norm_eps(1e-5f)
        .tied_embeddings(false)
        .use_qkv_bias(false)
        .dtype(dtype)
        .build();
}

/**
 * @brief Qwen2 0.5B configuration
 */
inline ModelConfig qwen2_0_5b(ETensorDType dtype = ETensorDType::BF16) {
    return ModelConfigBuilder()
        .architecture(ArchitectureType::Dense)
        .activation(ActivationType::SwiGLU)
        .hidden_size(896)
        .intermediate_size(4864)
        .vocab_size(151936)
        .num_layers(24)
        .num_query_heads(14)
        .num_kv_heads(2)
        .max_position_embeddings(32768)
        .rope_theta(1000000.0f)
        .rms_norm_eps(1e-6f)
        .tied_embeddings(true)
        .use_qkv_bias(true)
        .dtype(dtype)
        .build();
}

/**
 * @brief Qwen2 1.5B configuration
 */
inline ModelConfig qwen2_1_5b(ETensorDType dtype = ETensorDType::BF16) {
    return ModelConfigBuilder()
        .architecture(ArchitectureType::Dense)
        .activation(ActivationType::SwiGLU)
        .hidden_size(1536)
        .intermediate_size(8960)
        .vocab_size(151936)
        .num_layers(28)
        .num_query_heads(12)
        .num_kv_heads(2)
        .max_position_embeddings(32768)
        .rope_theta(1000000.0f)
        .rms_norm_eps(1e-6f)
        .tied_embeddings(true)
        .use_qkv_bias(true)
        .dtype(dtype)
        .build();
}

/**
 * @brief Qwen2 7B configuration
 */
inline ModelConfig qwen2_7b(ETensorDType dtype = ETensorDType::BF16) {
    return ModelConfigBuilder()
        .architecture(ArchitectureType::Dense)
        .activation(ActivationType::SwiGLU)
        .hidden_size(3584)
        .intermediate_size(18944)
        .vocab_size(152064)
        .num_layers(28)
        .num_query_heads(28)
        .num_kv_heads(4)
        .max_position_embeddings(131072)
        .rope_theta(1000000.0f)
        .rms_norm_eps(1e-6f)
        .tied_embeddings(false)
        .use_qkv_bias(true)
        .dtype(dtype)
        .build();
}

/**
 * @brief Mistral 7B configuration
 */
inline ModelConfig mistral_7b(ETensorDType dtype = ETensorDType::BF16) {
    return ModelConfigBuilder()
        .architecture(ArchitectureType::Dense)
        .activation(ActivationType::SwiGLU)
        .hidden_size(4096)
        .intermediate_size(14336)
        .vocab_size(32000)
        .num_layers(32)
        .num_query_heads(32)
        .num_kv_heads(8)
        .max_position_embeddings(32768)
        .rope_theta(10000.0f)
        .rms_norm_eps(1e-5f)
        .tied_embeddings(false)
        .use_qkv_bias(false)
        .sliding_window(4096)
        .dtype(dtype)
        .build();
}

// ============================================================================
// MoE model presets
// ============================================================================

/**
 * @brief Mixtral 8x7B configuration
 *
 * 8 experts with top-2 routing per token.
 * Each expert is a standard MLP with SwiGLU activation.
 */
inline ModelConfig mixtral_8x7b(ETensorDType dtype = ETensorDType::BF16) {
    MoEConfig moe;
    moe.num_experts = 8;
    moe.top_k = 2;
    moe.use_shared_expert = false;
    moe.router_aux_loss_coef = 0.01f;
    moe.capacity_factor = 1.25f;

    return ModelConfigBuilder()
        .architecture(ArchitectureType::MoE)
        .activation(ActivationType::SwiGLU)
        .hidden_size(4096)
        .intermediate_size(14336)  // Per-expert intermediate size
        .vocab_size(32000)
        .num_layers(32)
        .num_query_heads(32)
        .num_kv_heads(8)
        .max_position_embeddings(32768)
        .rope_theta(1000000.0f)
        .rms_norm_eps(1e-5f)
        .tied_embeddings(false)
        .use_qkv_bias(false)
        .sliding_window(4096)
        .moe(moe)
        .dtype(dtype)
        .build();
}

/**
 * @brief Mixtral 8x22B configuration
 */
inline ModelConfig mixtral_8x22b(ETensorDType dtype = ETensorDType::BF16) {
    MoEConfig moe;
    moe.num_experts = 8;
    moe.top_k = 2;
    moe.use_shared_expert = false;
    moe.router_aux_loss_coef = 0.01f;
    moe.capacity_factor = 1.25f;

    return ModelConfigBuilder()
        .architecture(ArchitectureType::MoE)
        .activation(ActivationType::SwiGLU)
        .hidden_size(6144)
        .intermediate_size(16384)
        .vocab_size(32768)
        .num_layers(56)
        .num_query_heads(48)
        .num_kv_heads(8)
        .max_position_embeddings(65536)
        .rope_theta(1000000.0f)
        .rms_norm_eps(1e-5f)
        .tied_embeddings(false)
        .use_qkv_bias(false)
        .sliding_window(4096)
        .moe(moe)
        .dtype(dtype)
        .build();
}

/**
 * @brief Qwen2 MoE configuration (A14B style)
 *
 * 60 experts with top-6 routing, plus a shared expert.
 */
inline ModelConfig qwen2_moe_a14b(ETensorDType dtype = ETensorDType::BF16) {
    MoEConfig moe;
    moe.num_experts = 60;
    moe.top_k = 6;
    moe.use_shared_expert = true;
    moe.shared_expert_size = 5632;  // Shared expert size
    moe.router_aux_loss_coef = 0.001f;
    moe.capacity_factor = 1.5f;

    return ModelConfigBuilder()
        .architecture(ArchitectureType::MoE)
        .activation(ActivationType::SwiGLU)
        .hidden_size(3584)
        .intermediate_size(2560)  // Per-expert (smaller due to many experts)
        .vocab_size(151936)
        .num_layers(28)
        .num_query_heads(28)
        .num_kv_heads(4)
        .max_position_embeddings(32768)
        .rope_theta(1000000.0f)
        .rms_norm_eps(1e-6f)
        .tied_embeddings(false)
        .use_qkv_bias(true)
        .moe(moe)
        .dtype(dtype)
        .build();
}

/**
 * @brief DeepSeek MoE V2 Lite configuration
 *
 * DeepSeek-style with shared expert and fine-grained experts.
 */
inline ModelConfig deepseek_moe_16b(ETensorDType dtype = ETensorDType::BF16) {
    MoEConfig moe;
    moe.num_experts = 64;
    moe.top_k = 6;
    moe.use_shared_expert = true;
    moe.shared_expert_size = 11008;  // 2x regular expert size
    moe.router_aux_loss_coef = 0.001f;
    moe.capacity_factor = 1.25f;

    return ModelConfigBuilder()
        .architecture(ArchitectureType::MoE)
        .activation(ActivationType::SwiGLU)
        .hidden_size(4096)
        .intermediate_size(1408)  // Small per-expert size (fine-grained)
        .vocab_size(102400)
        .num_layers(28)
        .num_query_heads(32)
        .num_kv_heads(32)
        .max_position_embeddings(4096)
        .rope_theta(10000.0f)
        .rms_norm_eps(1e-6f)
        .tied_embeddings(false)
        .use_qkv_bias(false)
        .moe(moe)
        .dtype(dtype)
        .build();
}

/**
 * @brief Switch Transformer base configuration
 *
 * Single expert per token (top-1 routing) for maximum efficiency.
 */
inline ModelConfig switch_base(ETensorDType dtype = ETensorDType::BF16) {
    MoEConfig moe;
    moe.num_experts = 128;
    moe.top_k = 1;  // Switch routing
    moe.use_shared_expert = false;
    moe.router_aux_loss_coef = 0.01f;
    moe.capacity_factor = 1.25f;

    return ModelConfigBuilder()
        .architecture(ArchitectureType::MoE)
        .activation(ActivationType::GeLU)  // Original Switch uses GeLU
        .hidden_size(768)
        .intermediate_size(2048)
        .vocab_size(32128)
        .num_layers(12)
        .num_query_heads(12)
        .num_kv_heads(12)
        .max_position_embeddings(512)
        .rope_theta(10000.0f)
        .rms_norm_eps(1e-6f)
        .tied_embeddings(false)
        .use_qkv_bias(false)
        .moe(moe)
        .dtype(dtype)
        .build();
}

} // namespace presets

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MODEL_FACTORY_H
