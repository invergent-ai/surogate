// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_CONFIG_PRETRAINED_CONFIG_H
#define SUROGATE_SRC_CONFIG_PRETRAINED_CONFIG_H

#include <algorithm>
#include <memory>
#include <string_view>
#include <vector>

#include "config/rope_config.h"
#include "utilities/dtype.h"

// Forward declarations for derived config types
struct LlamaConfig;
struct Qwen2Config;
struct Qwen3Config;
struct Qwen3MoEConfig;

/**
 * @brief Base configuration class for pretrained decoder-only transformer models.
 *
 * This is the base class in a Transformers-like inheritance hierarchy:
 *   PretrainedConfig
 *   └── LlamaConfig
 *       └── Qwen2Config (adds sliding_window)
 *           └── Qwen3Config (adds qk_norm, head_dim)
 *               └── Qwen3MoEConfig (adds MoE-specific fields)
 *
 * Derived classes inherit all base fields and can add model-specific fields.
 */
struct PretrainedConfig {
    enum ArchitectureId {
        LLAMA,
        QWEN2,
        QWEN3,
        QWEN3_MOE,
    };

    // Architecture identifier
    ArchitectureId Architecture = LLAMA;

    // Token IDs
    int BosTokenId = 1;
    int EosTokenId = 2;
    int PadTokenId = 0;

    // Model dimensions
    int HiddenSize = 4096;
    int IntermediateSize = 11008;
    int VocabSize = 32000;
    int NumQueryHeads = 32;
    int NumKeyValHeads = 32;
    int NumLayers = 32;

    // Attention head dimension. If 0, defaults to HiddenSize / NumQueryHeads.
    // Some architectures (e.g., Qwen3) use an explicit head_dim that may not match HiddenSize/NumQueryHeads.
    int HeadDim = 0;

    // Position embeddings
    int MaxPositionEmbeddings = 4096;
    float RopeTheta = 10000.0f;
    RoPEConfig Rope;  // Flexible RoPE configuration (GLM4 partial, Qwen2-VL M-RoPE, etc.)

    // Normalization
    float RmsNormEps = 1e-5f;

    // Weight tying
    bool TiedWordEmbeddings = false;

    // Attention configuration
    bool UseQKVBias = false;
    bool UseQKNorm = false;

    // Data type
    ETensorDType DType = ETensorDType::BF16;

    // Virtual destructor for proper polymorphic deletion
    virtual ~PretrainedConfig() = default;

    // Clone method for polymorphic copying
    [[nodiscard]] virtual std::unique_ptr<PretrainedConfig> clone() const {
        return std::make_unique<PretrainedConfig>(*this);
    }

    // Type checking methods
    [[nodiscard]] virtual bool is_moe() const { return false; }
    [[nodiscard]] virtual bool has_sliding_window() const { return false; }
    [[nodiscard]] virtual bool has_qk_norm() const { return UseQKNorm; }

    // MoE layer query (overridden by MoE configs)
    [[nodiscard]] virtual bool is_moe_layer(int /*layer_idx*/) const { return false; }

    // Computed properties
    [[nodiscard]] int head_size() const { return HeadDim > 0 ? HeadDim : (HiddenSize / NumQueryHeads); }
    [[nodiscard]] int attn_out_channels() const { return head_size() * NumQueryHeads; }
    [[nodiscard]] int qkv_channels() const { return head_size() * (NumQueryHeads + 2 * NumKeyValHeads); }
    [[nodiscard]] std::string_view model_name() const;

    // Default constructors - public for compatibility with existing code
    PretrainedConfig() = default;
    PretrainedConfig(const PretrainedConfig&) = default;
    PretrainedConfig& operator=(const PretrainedConfig&) = default;
    PretrainedConfig(PretrainedConfig&&) = default;
    PretrainedConfig& operator=(PretrainedConfig&&) = default;
};

/**
 * @brief LlamaConfig - Configuration for LLaMA family models.
 *
 * Inherits all base PretrainedConfig fields.
 * LLaMA uses standard RoPE, RMSNorm, SwiGLU, and GQA.
 */
struct LlamaConfig : public PretrainedConfig {
    LlamaConfig() {
        Architecture = LLAMA;
    }

    [[nodiscard]] std::unique_ptr<PretrainedConfig> clone() const override {
        return std::make_unique<LlamaConfig>(*this);
    }
};

/**
 * @brief Qwen2Config - Configuration for Qwen2 family models.
 *
 * Inherits from LlamaConfig and adds sliding window attention support.
 */
struct Qwen2Config : public LlamaConfig {
    // Sliding window attention (0 = disabled)
    int SlidingWindow = 0;

    Qwen2Config() {
        Architecture = QWEN2;
    }

    [[nodiscard]] std::unique_ptr<PretrainedConfig> clone() const override {
        return std::make_unique<Qwen2Config>(*this);
    }

    [[nodiscard]] bool has_sliding_window() const override { return SlidingWindow > 0; }
};

/**
 * @brief Qwen3Config - Configuration for Qwen3 dense models.
 *
 * Inherits from Qwen2Config and adds QK normalization.
 * Qwen3 uses explicit head_dim and QK norm by default.
 */
struct Qwen3Config : public Qwen2Config {
    Qwen3Config() {
        Architecture = QWEN3;
        UseQKNorm = true;  // Qwen3 uses QK norm by default
    }

    [[nodiscard]] std::unique_ptr<PretrainedConfig> clone() const override {
        return std::make_unique<Qwen3Config>(*this);
    }

    [[nodiscard]] bool has_qk_norm() const override { return UseQKNorm; }
};

/**
 * @brief Qwen3MoEConfig - Configuration for Qwen3 Mixture of Experts models.
 *
 * Inherits from Qwen3Config and adds MoE-specific fields.
 * Supports hybrid dense/MoE layers via decoder_sparse_step and mlp_only_layers.
 */
struct Qwen3MoEConfig : public Qwen3Config {
    // MoE-specific configuration
    int NumExperts = 0;             ///< Number of routed experts (0 = not MoE)
    int NumExpertsPerTok = 0;       ///< Top-K experts selected per token
    int MoeIntermediateSize = 0;    ///< Per-expert MLP hidden dim (0 = use IntermediateSize)
    int DecoderSparseStep = 1;      ///< MoE layer frequency: MoE every N layers (1 = all MoE)
    std::vector<int> MlpOnlyLayers; ///< Explicit list of layer indices using dense MLP instead of MoE
    bool NormTopkProb = false;      ///< Normalize top-k routing weights to sum to 1
    float RouterAuxLossCoef = 0.001f; ///< Load balancing auxiliary loss coefficient

    Qwen3MoEConfig() {
        Architecture = QWEN3_MOE;
    }

    [[nodiscard]] std::unique_ptr<PretrainedConfig> clone() const override {
        return std::make_unique<Qwen3MoEConfig>(*this);
    }

    [[nodiscard]] bool is_moe() const override { return NumExperts > 0; }

    /**
     * @brief Check if a specific layer uses MoE routing.
     *
     * A layer is MoE if:
     * 1. NumExperts > 0 (model is MoE)
     * 2. Layer is NOT in mlp_only_layers
     * 3. (layer_idx + 1) % decoder_sparse_step == 0
     */
    [[nodiscard]] bool is_moe_layer(int layer_idx) const override {
        if (NumExperts == 0) return false;

        // Check if this layer is explicitly marked as dense (mlp_only_layers)
        if (std::find(MlpOnlyLayers.begin(), MlpOnlyLayers.end(), layer_idx) != MlpOnlyLayers.end()) {
            return false;
        }

        // Check decoder_sparse_step pattern: MoE if (layer_idx + 1) % step == 0
        // With step=1, all layers are MoE (default behavior)
        return (layer_idx + 1) % DecoderSparseStep == 0;
    }

    /**
     * @brief Get the intermediate size for a specific layer.
     *
     * MoE layers use MoeIntermediateSize, dense layers use IntermediateSize.
     */
    [[nodiscard]] int get_intermediate_size(int layer_idx) const {
        if (is_moe_layer(layer_idx) && MoeIntermediateSize > 0) {
            return MoeIntermediateSize;
        }
        return IntermediateSize;
    }
};

// Factory functions
std::unique_ptr<PretrainedConfig> load_pretrained_config(const char* file_name, ETensorDType dtype);
void save_pretrained_config(const PretrainedConfig& config, const char* file_name);
std::unique_ptr<PretrainedConfig> create_pretrained_config_from_name(std::string_view name, ETensorDType dtype);

// Legacy compatibility: returns a copy (use unique_ptr version for new code)
PretrainedConfig load_pretrained_config_legacy(const char* file_name, ETensorDType dtype);
PretrainedConfig create_pretrained_config_from_name_legacy(std::string_view name, ETensorDType dtype);

#endif // SUROGATE_SRC_CONFIG_PRETRAINED_CONFIG_H
