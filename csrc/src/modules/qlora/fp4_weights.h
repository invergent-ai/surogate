// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// FP4 (NVFP4/E2M1) QLoRA weights manager.

#ifndef SUROGATE_SRC_MODULES_QLORA_FP4_WEIGHTS_H
#define SUROGATE_SRC_MODULES_QLORA_FP4_WEIGHTS_H

#include <string>
#include <vector>

#include "fp4_block_quantized_tensor.h"
#include "block_quantized_tensor.h"  // For QLoRAEmbeddingWeights
#include "moe_weights.h"
#include "qlora_config.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"
#include "utilities/safetensors.h"
#include "utilities/tensor.h"

namespace modules {

/**
 * @brief FP4 QLoRA weights manager
 *
 * Manages quantized base model weights using FP4 E2M1 format with two-level
 * block scaling (FP8 E4M3 block scales + FP32 global scale).
 *
 * Memory layout:
 * - Base model weights: FP4 E2M1 packed (2 values per byte)
 * - Block scales: FP8 E4M3 with F8_128x4 swizzling
 * - Global amax: FP32 per tensor
 * - Embeddings/LM head: BF16 (not quantized)
 *
 * Usage:
 * 1. Call import_and_quantize() to load and quantize base model weights
 * 2. Access quantized blocks via get_fp4_block() for weight provider
 */
class FP4WeightsManager {
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
        bool use_qk_norm = false;
        bool tied_embeddings = true;
        int shard_idx = 0;
        int num_shards = 1;

        /// When true, store MoE expert FP4 weights in pinned CPU memory instead of GPU.
        /// Experts are streamed to GPU on-demand when selected by the router.
        /// Saves ~10GB for 128-expert models, with ~20-40% throughput reduction.
        bool offload_experts = false;
    };

    FP4WeightsManager(const Config& config, TensorAllocator& allocator,
                      const cudaDeviceProp& device_props);
    ~FP4WeightsManager();

    /**
     * @brief Import base model weights from safetensors and quantize to FP4
     */
    void import_and_quantize(const std::string& file_name, NCCLCommunicator& comm,
                             cudaStream_t stream);

    /**
     * @brief Get FP4 quantized block weights
     */
    [[nodiscard]] const FP4BlockWeights& get_fp4_block(int layer_idx) const {
        return mFP4Blocks[layer_idx];
    }

    [[nodiscard]] FP4BlockWeights& get_fp4_block(int layer_idx) {
        return mFP4Blocks[layer_idx];
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
     * @brief Check if FP4 quantization is active
     */
    [[nodiscard]] bool is_fp4() const { return mConfig.qlora_config.is_fp4(); }

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
     */
    [[nodiscard]] const MoEBlockWeights<FP4BlockWeights, FP4BlockQuantizedWeight>&
        get_moe_block(int layer_idx) const {
        return mMoEBlocks[layer_idx];
    }

    [[nodiscard]] MoEBlockWeights<FP4BlockWeights, FP4BlockQuantizedWeight>&
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

    // FP4 quantized base model weights for dense models
    std::vector<FP4BlockWeights> mFP4Blocks;

    // FP4 quantized base model weights for MoE models
    using FP4MoEBlockWeights = MoEBlockWeights<FP4BlockWeights, FP4BlockQuantizedWeight>;
    std::vector<FP4MoEBlockWeights> mMoEBlocks;

    // Embedding weights (not quantized)
    QLoRAEmbeddingWeights mEmbeddings;

    // Temporary buffer for loading BF16 weights before quantization
    Tensor mLoadBuffer;
    std::size_t mLoadBufferBytes = 0;

    // Global amax storage (device memory)
    std::vector<float*> mGlobalAmaxPtrs;

    // Allocation helpers
    void allocate_fp4_blocks();
    float* allocate_global_amax();

    // Quantization helpers
    void quantize_fp4_and_store(FP4BlockQuantizedWeight& dest, const Tensor& src,
                                 int M, int K, cudaStream_t stream);

    // Weight loading helpers
    void load_and_quantize_block(int layer_idx, SafeTensorsReader& reader,
                                 cudaStream_t stream);
    void load_embeddings(SafeTensorsReader& reader, cudaStream_t stream);

    // MoE-specific allocation and loading helpers
    void allocate_moe_blocks();
    void load_and_quantize_moe_block(int layer_idx, SafeTensorsReader& reader,
                                      cudaStream_t stream);

    // Generic allocation helper for FP4 quantized weights
    void allocate_fp4_weight(FP4BlockQuantizedWeight& weight, int M, int K,
                             const std::string& name_prefix);
    void allocate_fp4_weight(FP4BlockQuantizedWeight& weight, int M, int K,
                             const std::string& name_prefix, EAllocationType alloc_type);

    // Helper to get the allocation type for expert weights
    [[nodiscard]] EAllocationType expert_alloc_type() const {
        // Use pageable host memory for offloaded experts to avoid mapped host pages
        // being corrupted by stray device writes.
        return mConfig.offload_experts ? EAllocationType::ON_HOST : EAllocationType::ON_DEVICE;
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_QLORA_FP4_WEIGHTS_H
