// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_QLORA_BNB_WEIGHTS_H
#define SUROGATE_SRC_MODULES_QLORA_BNB_WEIGHTS_H

#include <string>
#include <vector>

#include "bnb_block_quantized_tensor.h"
#include "hf_mapping.h"
#include "moe_weights.h"
#include "qlora_config.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"
#include "utilities/safetensors.h"
#include "utilities/tensor.h"

namespace modules {

/**
 * @brief BitsAndBytes NF4 QLoRA weights manager
 *
 * Manages quantized base model weights (NF4 with per-block absmax scaling).
 * LoRA adapters are managed separately by ModularLoRAModel.
 * Dequantization is handled by BnBWeightProvider.
 *
 * Memory layout:
 * - Base model weights: Packed 4-bit NF4 with per-block absmax (stored permanently)
 * - Double quantization: INT8 absmax values + FP32 scale/offset per group
 * - Embeddings/LM head: BF16 (not quantized)
 *
 * Usage:
 * 1. Call import_and_quantize() to load and quantize base model weights
 * 2. Access quantized blocks via get_bnb_block() for BnBWeightProvider
 */
class BnBWeightsManager {
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
        bool use_qk_norm = false;      ///< Whether model uses QK-norm (Qwen3)
        bool tied_embeddings = true;   ///< Whether lm_head is tied to embeddings
        int shard_idx = 0;
        int num_shards = 1;
        const HfMapping* hf_mapping = nullptr;

        /// When true, store MoE expert NF4 weights in pinned CPU memory instead of GPU.
        /// Experts are streamed to GPU on-demand when selected by the router.
        /// Saves ~12GB for 128-expert models, with ~20-40% throughput reduction.
        bool offload_experts = false;
    };

    BnBWeightsManager(const Config& config, TensorAllocator& allocator,
                      const cudaDeviceProp& device_props);
    ~BnBWeightsManager() = default;

    /**
     * @brief Import base model weights from safetensors and quantize to NF4
     *
     * Loads BF16 weights into a temporary buffer, quantizes with per-block
     * absmax scaling, optionally applies double quantization, and stores
     * the NF4 weights + scales. The temporary buffer is freed after quantization.
     *
     * @param file_name Path to safetensors file or model.safetensors.index.json
     * @param comm NCCL communicator for multi-GPU sync
     * @param stream CUDA stream for async operations
     */
    void import_and_quantize(const std::string& file_name, NCCLCommunicator& comm,
                             cudaStream_t stream);

    /**
     * @brief Get BnB quantized block weights (for BnBWeightProvider)
     */
    [[nodiscard]] const BnBBlockWeights& get_bnb_block(int layer_idx) const {
        return mQuantizedBlocks[layer_idx];
    }

    [[nodiscard]] BnBBlockWeights& get_bnb_block(int layer_idx) {
        return mQuantizedBlocks[layer_idx];
    }

    /**
     * @brief Get embedding weights (not quantized)
     */
    [[nodiscard]] const BnBEmbeddingWeights& get_embeddings() const { return mEmbeddings; }
    [[nodiscard]] BnBEmbeddingWeights& get_embeddings() { return mEmbeddings; }

    /**
     * @brief Get QLoRA configuration
     */
    [[nodiscard]] const QLoRAConfig& qlora_config() const { return mConfig.qlora_config; }

    /**
     * @brief Get block size for quantization
     */
    [[nodiscard]] int block_size() const { return mConfig.qlora_config.block_size(); }

    /**
     * @brief Check if double quantization is enabled
     */
    [[nodiscard]] bool double_quant() const { return mConfig.qlora_config.bnb_double_quant; }

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
     * @brief Check if expert weights are offloaded to CPU
     */
    [[nodiscard]] bool experts_offloaded() const { return mConfig.offload_experts && is_moe(); }

    /**
     * @brief Get MoE block weights (for MoE models)
     * @throws std::runtime_error if not an MoE model
     */
    [[nodiscard]] const MoEBlockWeights<BnBBlockWeights, BnBBlockQuantizedWeight>&
        get_moe_block(int layer_idx) const {
        return mMoEBlocks[layer_idx];
    }

    [[nodiscard]] MoEBlockWeights<BnBBlockWeights, BnBBlockQuantizedWeight>&
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

    // BnB scale configuration (derived from qlora_config)
    BnBBlockScaleConfig mScaleConfig;

    // Quantized base model weights for dense models (NF4 + per-block absmax)
    std::vector<BnBBlockWeights> mQuantizedBlocks;

    // Quantized base model weights for MoE models
    using BnBMoEBlockWeights = MoEBlockWeights<BnBBlockWeights, BnBBlockQuantizedWeight>;
    std::vector<BnBMoEBlockWeights> mMoEBlocks;

    // Embedding weights (not quantized)
    BnBEmbeddingWeights mEmbeddings;

    // Temporary buffer for loading BF16 weights before quantization
    Tensor mLoadBuffer;
    std::size_t mLoadBufferBytes = 0;

    // Temporary buffer for absmax values during double quantization
    Tensor mAbsmaxBuffer;

    // Staging buffers for expert weights when offloading to CPU.
    // When offload_experts is enabled, we quantize to GPU staging buffers first,
    // then copy to pinned CPU memory. This avoids GPU kernels writing directly
    // to mapped pinned memory, which can cause coherence issues.
    Tensor mExpertNF4Staging;         ///< GPU buffer for packed NF4 data
    Tensor mExpertAbsmaxStaging;      ///< GPU buffer for absmax (FP32 or UINT8)
    Tensor mExpertAbsmaxScaleStaging; ///< GPU buffer for double-quant scale
    Tensor mExpertAbsmaxOffsetStaging;///< GPU buffer for double-quant offset
    bool mExpertStagingAllocated = false;

    // Allocation helpers
    void allocate_single_block(int layer_idx);
    void allocate_bnb_weight(BnBBlockQuantizedWeight& weight, int M, int K,
                             const std::string& name_prefix);
    void allocate_bnb_weight(BnBBlockQuantizedWeight& weight, int M, int K,
                             const std::string& name_prefix, EAllocationType alloc_type);

    // Helper to get the allocation type for expert weights
    [[nodiscard]] EAllocationType expert_alloc_type() const {
        // Use pageable host memory for offloaded experts to avoid mapped host pages
        // being corrupted by stray device writes.
        return mConfig.offload_experts ? EAllocationType::ON_HOST : EAllocationType::ON_DEVICE;
    }

    // Quantization helpers
    void quantize_and_store(BnBBlockQuantizedWeight& dest, const Tensor& src,
                            int M, int K, cudaStream_t stream);
    void quantize_and_store_offloaded(BnBBlockQuantizedWeight& dest, const Tensor& src,
                                       int M, int K, cudaStream_t stream);
    void apply_double_quantization(BnBBlockQuantizedWeight& weight, cudaStream_t stream);
    void allocate_expert_staging_buffers();

    // Weight loading helpers
    void load_and_quantize_block(int layer_idx, SafeTensorsReader& reader,
                                 cudaStream_t stream);
    void load_embeddings(SafeTensorsReader& reader, cudaStream_t stream);

    // MoE-specific allocation and loading helpers
    void allocate_moe_block(int layer_idx);
    void load_and_quantize_moe_block(int layer_idx, SafeTensorsReader& reader,
                                      cudaStream_t stream);
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_QLORA_BNB_WEIGHTS_H
