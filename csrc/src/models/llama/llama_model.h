// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// LlamaModel - Model class for LLaMA family architectures
//
// Inherits from BasePreTrainedModel and uses LlamaTransformerBlock.
// This is the base model in the dense model inheritance hierarchy:
//
//   LlamaModel
//   └── Qwen2Model
//       └── Qwen3Model
//

#ifndef SUROGATE_SRC_MODELS_LLAMA_LLAMA_MODEL_H
#define SUROGATE_SRC_MODELS_LLAMA_LLAMA_MODEL_H

#include "models/llama/config.h"
#include "models/llama/transformer_block.h"
#include "modules/model/modular_model.h"

namespace modules {

/**
 * @brief Model class for LLaMA family architectures
 *
 * Uses LlamaTransformerBlock which composes:
 * - LlamaAttentionModule (GQA with RoPE)
 * - SwiGLUModule (standard activation)
 * - LlamaFusedResidualRMSNormModule (fused residual + norm)
 *
 * This class is a type alias for ModularTransformerModel with the LLaMA block.
 * Weight I/O uses the "llama" weight mapping.
 */
class LlamaModel : public ModularTransformerModel<LlamaTransformerBlock> {
public:
    using Base = ModularTransformerModel<LlamaTransformerBlock>;
    using BlockType = LlamaTransformerBlock;

    /**
     * @brief Construct a LlamaModel
     *
     * @param config Model configuration (should be LlamaConfig or derived)
     * @param options Runtime options
     * @param rank Process rank for sharding
     * @param world World size
     * @param alloc Optional tensor allocator
     */
    LlamaModel(const ModelConfig& config, const ModelOptions& options,
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
        return "LlamaForCausalLM";
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
        return PretrainedConfig::LLAMA;
    }
};

/**
 * @brief Factory function to create a LlamaModel from LlamaConfig
 *
 * @param config LlamaConfig instance
 * @param options Runtime options
 * @param rank Process rank for sharding
 * @param world World size
 * @param alloc Optional tensor allocator
 * @return Unique pointer to LlamaModel (as IModel)
 */
inline std::unique_ptr<IModel> create_llama_model(
    const LlamaConfig& config,
    const ModelOptions& options,
    int rank, int world,
    const std::shared_ptr<TensorAllocator>& alloc = nullptr) {

    ModelConfig mod_config = ModelConfig::from_pretrained_config(config);
    return std::make_unique<LlamaModel>(mod_config, options, rank, world, alloc);
}

} // namespace modules

#endif // SUROGATE_SRC_MODELS_LLAMA_LLAMA_MODEL_H
