// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// PretrainedConfig - Base configuration class for pretrained models.
//
// Model-specific configs have been removed; the DSL IR + config.json drive
// all model configuration via this base struct.

#ifndef SUROGATE_SRC_CONFIG_PRETRAINED_CONFIG_H
#define SUROGATE_SRC_CONFIG_PRETRAINED_CONFIG_H

#include <memory>
#include <string>
#include <string_view>

#include "config/rope_config.h"
#include "utilities/dtype.h"

/**
 * @brief Base configuration class for pretrained decoder-only transformer models.
 *
 * This is the base class for all model configs. Model-specific subclasses have
 * been removed; DSL IR supplies any architecture-specific overrides.
 */
struct PretrainedConfig {
    enum ArchitectureId {
        LLAMA,
        QWEN2,
        QWEN3,
        QWEN3_MOE,
        NEMOTRON_H,
        GPT_OSS,
    };

    // Architecture identifier
    ArchitectureId Architecture = LLAMA;
    // Optional raw architecture/model_type strings from config.json
    std::string ArchitectureName;
    std::string ModelTypeName;

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

    // Sliding window attention (optional, per-layer when layer types provided)
    int SlidingWindow = 0;
    std::vector<int> LayerTypes;  // 0 = full attention, 1 = sliding attention

    // Normalization
    float RmsNormEps = 1e-5f;

    // Weight tying
    bool TiedWordEmbeddings = false;

    // Attention configuration
    bool UseQKVBias = false;
    bool UseQKNorm = false;

    // Vision-language inputs (Qwen3-VL style)
    bool UseVisualInputs = false;   ///< Enable visual_pos_masks/visual_embeds inputs
    int DeepstackVisualLayers = 0;  ///< Number of deepstack visual embeddings (0 = disabled)

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
    [[nodiscard]] virtual bool has_sliding_window() const { return SlidingWindow > 0; }
    [[nodiscard]] virtual bool has_qk_norm() const { return UseQKNorm; }

    [[nodiscard]] bool is_sliding_layer(int layer_idx) const {
        if (layer_idx < 0) {
            return false;
        }
        if (!LayerTypes.empty()) {
            if (layer_idx < static_cast<int>(LayerTypes.size())) {
                return LayerTypes[static_cast<std::size_t>(layer_idx)] != 0;
            }
        }
        return false;
    }

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

// Factory functions
std::unique_ptr<PretrainedConfig> load_pretrained_config(const char* file_name, ETensorDType dtype);
void save_pretrained_config(const PretrainedConfig& config, const char* file_name);
std::unique_ptr<PretrainedConfig> create_pretrained_config_from_name(std::string_view name, ETensorDType dtype);

#endif // SUROGATE_SRC_CONFIG_PRETRAINED_CONFIG_H
