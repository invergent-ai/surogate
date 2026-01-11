// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen2Model - Model class for Qwen2 family architectures
//
// Inherits from LlamaModel concept and uses Qwen2TransformerBlock.
// Extends LlamaModel with sliding window attention support.
//
// Model inheritance hierarchy:
//   LlamaModel
//   └── Qwen2Model      <-- This class
//       └── Qwen3Model
//

#ifndef SUROGATE_SRC_MODELS_QWEN2_QWEN2_MODEL_H
#define SUROGATE_SRC_MODELS_QWEN2_QWEN2_MODEL_H

#include "models/qwen25/config.h"
#include "models/qwen25/transformer_block.h"
#include "modules/model/modular_model.h"

namespace modules {

/**
 * @brief Model class for Qwen2 family architectures
 *
 * Uses Qwen2TransformerBlock which composes:
 * - Qwen2AttentionModule (GQA with RoPE, optional sliding window)
 * - SwiGLUModule (standard activation)
 * - Qwen2FusedResidualRMSNormModule (fused residual + norm)
 *
 * Extends LlamaModel with:
 * - Sliding window attention support
 * - QKV bias (typically enabled)
 *
 * Weight I/O uses the "qwen2" weight mapping, which extends "llama" mapping.
 */
class Qwen2Model : public ModularTransformerModel<Qwen2TransformerBlock> {
public:
    using Base = ModularTransformerModel<Qwen2TransformerBlock>;
    using BlockType = Qwen2TransformerBlock;

    /**
     * @brief Construct a Qwen2Model
     *
     * @param config Model configuration (should be Qwen2Config or derived)
     * @param options Runtime options
     * @param rank Process rank for sharding
     * @param world World size
     * @param alloc Optional tensor allocator
     */
    Qwen2Model(const ModelConfig& config, const ModelOptions& options,
               int rank, int world,
               const std::shared_ptr<TensorAllocator>& alloc = nullptr)
        : Base(config, options, rank, world, alloc) {}

    // ========================================================================
    // Architecture identification
    // ========================================================================

    /**
     * @brief Get the HuggingFace model type identifier
     */
    [[nodiscard]] std::string_view model_type() const override {
        return "Qwen2ForCausalLM";
    }

    /**
     * @brief Check if this is an MoE model
     */
    [[nodiscard]] static constexpr bool is_moe_model() {
        return false;
    }

    /**
     * @brief Get the config architecture ID
     */
    [[nodiscard]] static constexpr PretrainedConfig::ArchitectureId architecture_id() {
        return PretrainedConfig::QWEN2;
    }
};

/**
 * @brief Factory function to create a Qwen2Model from Qwen2Config
 *
 * @param config Qwen2Config instance
 * @param options Runtime options
 * @param rank Process rank for sharding
 * @param world World size
 * @param alloc Optional tensor allocator
 * @return Unique pointer to Qwen2Model (as IModel)
 */
inline std::unique_ptr<IModel> create_qwen2_model(
    const Qwen2Config& config,
    const ModelOptions& options,
    int rank, int world,
    const std::shared_ptr<TensorAllocator>& alloc = nullptr) {

    ModelConfig mod_config = ModelConfig::from_pretrained_config(config);
    return std::make_unique<Qwen2Model>(mod_config, options, rank, world, alloc);
}

} // namespace modules

#endif // SUROGATE_SRC_MODELS_QWEN2_QWEN2_MODEL_H
