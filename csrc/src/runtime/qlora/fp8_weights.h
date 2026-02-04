// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_QLORA_QLORA_WEIGHTS_H
#define SUROGATE_SRC_MODULES_QLORA_QLORA_WEIGHTS_H

#include <string>
#include <vector>

#include "block_quantized_tensor.h"
#include "hf_mapping.h"
#include "moe_weights.h"
#include "qlora_config.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"
#include "utilities/safetensors.h"
#include "utilities/tensor.h"

namespace modules {

/**
 * @brief FP8 QLoRA weights manager
 *
 * Manages quantized base model weights (FP8 with per-block scales).
 * LoRA adapters are managed separately by ModularLoRAModel.
 * Dequantization is handled by FP8WeightProvider.
 *
 * Memory layout:
 * - Base model weights: FP8 E4M3 with per-block scales (stored permanently)
 * - Embeddings/LM head: BF16 (not quantized, shared with ModularTransformerModel)
 *
 * Usage:
 * 1. Call import_and_quantize() to load and quantize base model weights
 * 2. Access quantized blocks via get_quantized_block() for FP8WeightProvider
 */
class FP8WeightsManager {
public:
    struct Config {
        int num_layers;
        int hidden_size;
        int intermediate_size;
        int num_query_heads;
        int num_kv_heads;
        int head_size;
        int vocab_size;
        int mlp_up_factor = 2;      ///< 2 for gated (SwiGLU), 1 for non-gated (ReLU2)
        QLoRAConfig qlora_config;
        bool use_qk_norm = false;  ///< Whether model uses QK-norm (Qwen3)
        bool tied_embeddings = true;  ///< Whether lm_head is tied to embeddings
        int shard_idx = 0;
        int num_shards = 1;
        const HfMapping* hf_mapping = nullptr;
    };

    FP8WeightsManager(const Config& config, TensorAllocator& allocator,
                        const cudaDeviceProp& device_props);
    ~FP8WeightsManager() = default;

    /**
     * @brief Import base model weights from safetensors and quantize to FP8
     *
     * Loads BF16 weights into a temporary buffer, quantizes with per-block
     * scales, and stores the FP8 weights + scales. The temporary buffer is
     * freed after quantization.
     *
     * @param file_name Path to safetensors file or model.safetensors.index.json
     * @param comm NCCL communicator for multi-GPU sync
     * @param stream CUDA stream for async operations
     */
    void import_and_quantize(const std::string& file_name, NCCLCommunicator& comm,
                             cudaStream_t stream);

    /**
     * @brief Get quantized block weights (for FP8WeightProvider)
     */
    [[nodiscard]] const QLoRABlockWeights& get_quantized_block(int layer_idx) const {
        return mQuantizedBlocks[layer_idx];
    }

    [[nodiscard]] QLoRABlockWeights& get_quantized_block(int layer_idx) {
        return mQuantizedBlocks[layer_idx];
    }

    /**
     * @brief Get embedding weights (not quantized)
     */
    [[nodiscard]] const QLoRAEmbeddingWeights& get_embeddings() const { return mEmbeddings; }
    [[nodiscard]] QLoRAEmbeddingWeights& get_embeddings() { return mEmbeddings; }

    /**
     * @brief Get QLoRA configuration
     */
    [[nodiscard]] const QLoRAConfig& qlora_config() const { return mConfig.qlora_config; }

    /**
     * @brief Get block size for quantization
     */
    [[nodiscard]] int block_size() const { return mConfig.qlora_config.block_size(); }

    /**
     * @brief Check if QLoRA is enabled
     */
    [[nodiscard]] bool is_quantized() const { return mConfig.qlora_config.is_quantized(); }

    /**
     * @brief Check if this is an MoE model
     */
    [[nodiscard]] bool is_moe() const { return mConfig.qlora_config.is_moe(); }

    /**
     * @brief Get number of experts (0 for dense models)
     */
    [[nodiscard]] int num_experts() const { return mConfig.qlora_config.num_experts; }

    /**
     * @brief Get MoE block weights (for MoE models)
     * @throws std::runtime_error if not an MoE model
     */
    [[nodiscard]] const MoEBlockWeights<QLoRABlockWeights, BlockQuantizedWeight>&
        get_moe_block(int layer_idx) const {
        return mMoEBlocks[layer_idx];
    }

    [[nodiscard]] MoEBlockWeights<QLoRABlockWeights, BlockQuantizedWeight>&
        get_moe_block(int layer_idx) {
        return mMoEBlocks[layer_idx];
    }

    /**
     * @brief Get total memory usage for quantized weights in bytes
     */
    [[nodiscard]] std::size_t quantized_weights_bytes() const;

    /**
     * @brief Get memory savings compared to BF16 storage
     */
    [[nodiscard]] float memory_savings_ratio() const;

private:
    Config mConfig;
    TensorAllocator* mAllocator;
    const cudaDeviceProp* mDeviceProps;
    const HfMapping* mHfMapping = nullptr;

    // Quantized base model weights for dense models (FP8 + per-block scales)
    std::vector<QLoRABlockWeights> mQuantizedBlocks;

    // Quantized base model weights for MoE models
    using FP8MoEBlockWeights = MoEBlockWeights<QLoRABlockWeights, BlockQuantizedWeight>;
    std::vector<FP8MoEBlockWeights> mMoEBlocks;

    // Embedding weights (not quantized)
    QLoRAEmbeddingWeights mEmbeddings;

    // Temporary buffer for loading BF16 weights before quantization
    // (allocated directly, not via allocator, so we can free it after use)
    Tensor mLoadBuffer;
    std::size_t mLoadBufferBytes = 0;

    // Allocation helpers
    void allocate_single_block(int layer_idx);

    // Quantization helpers
    void quantize_and_store(BlockQuantizedWeight& dest, const Tensor& src,
                            int M, int K, cudaStream_t stream);

    // Weight loading helpers
    void load_and_quantize_block(int layer_idx, SafeTensorsReader& reader,
                                 cudaStream_t stream);
    void load_embeddings(SafeTensorsReader& reader, cudaStream_t stream);

    // MoE-specific allocation and loading helpers
    void allocate_moe_block(int layer_idx);
    void load_and_quantize_moe_block(int layer_idx, SafeTensorsReader& reader,
                                      cudaStream_t stream);

    // Generic allocation helper for quantized weights
    void allocate_fp8_weight(BlockQuantizedWeight& weight, int M, int K,
                             const std::string& name_prefix);
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_QLORA_QLORA_WEIGHTS_H
