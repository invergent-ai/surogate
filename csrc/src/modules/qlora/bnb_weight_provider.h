// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_QLORA_BNB_WEIGHT_PROVIDER_H
#define SUROGATE_SRC_MODULES_QLORA_BNB_WEIGHT_PROVIDER_H

#include <memory>
#include <vector>

#include "qlora_config.h"
#include "bnb_weights.h"
#include "bnb_block_quantized_tensor.h"
#include "modules/composite/transformer_block.h"
#include "modules/lora/lora_config.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"

namespace modules {

/**
 * @brief Provides dequantized weights for BitsAndBytes NF4 QLoRA training
 *
 * This class wraps BnBWeightsManager and provides on-the-fly dequantization
 * of NF4 base weights to BF16 for use in the forward pass.
 *
 * Key design:
 * - Quantized weights (NF4 packed 4-bit + per-block absmax) are stored permanently
 * - Double quantization: absmax values are quantized to INT8 with scale/offset
 * - Dequantization buffers are allocated once and reused
 * - get_block() dequantizes the requested layer's weights via lookup table
 * - Compatible with the existing ModularWeightManager interface patterns
 *
 * Optimization: Forward/Backward Dequant Caching
 * - Since base weights are frozen in QLoRA, dequantized weights are identical
 *   between forward and backward passes within a single training step
 * - Uses step versioning to detect when the same layer is accessed twice
 *   within one step (forward then backward) and skips redundant dequantization
 * - Step version is incremented via new_step() at the start of each training step
 *
 * @tparam Block The transformer block type (e.g., DenseTransformerBlock<>)
 */
template<typename Block>
class BnBWeightProvider {
public:
    using BlockWeights = typename Block::Weights;
    using BlockConfig = typename Block::Config;

    struct Config {
        int num_layers;
        int hidden_size;
        int intermediate_size;
        int num_query_heads;
        int num_kv_heads;
        int head_size;
        int vocab_size;
        QLoRAConfig qlora_config;
        ModularLoRAConfig lora_config;
        ETensorDType model_dtype = ETensorDType::BF16;
        bool use_qk_norm = false;      ///< Whether model uses QK-norm (Qwen3)
        bool tied_embeddings = true;   ///< Whether lm_head is tied to embeddings
        int shard_idx = 0;
        int num_shards = 1;
    };

    BnBWeightProvider(const Config& config, TensorAllocator& allocator,
                      const cudaDeviceProp& device_props);
    ~BnBWeightProvider() = default;

    /**
     * @brief Import and quantize base model weights from file
     */
    void import_and_quantize(const std::string& file_name, NCCLCommunicator& comm,
                             cudaStream_t stream);

    /**
     * @brief Get dequantized block weights
     *
     * Dequantizes the NF4 weights for the specified layer and returns
     * a BlockWeights struct with BF16 tensors ready for matmul.
     *
     * If the same layer was already dequantized in this step (forward),
     * the cached dequantized weights are returned without re-dequantization.
     *
     * @param layer_idx Layer index
     * @param stream CUDA stream for dequantization
     * @return Reference to BlockWeights with dequantized tensors
     */
    BlockWeights& get_block(int layer_idx, cudaStream_t stream);

    /**
     * @brief Release block weights (no-op for QLoRA, kept for interface compat)
     */
    void release_block(int layer_idx, cudaStream_t stream) {
        (void)layer_idx;
        (void)stream;
        // No-op: dequant buffers are statically allocated
    }

    /**
     * @brief Signal the start of a new training step
     *
     * Call this at the start of each training step (before forward pass) to
     * increment the step version. This allows get_block() to detect when a
     * layer is accessed for the second time (backward after forward) and
     * skip redundant dequantization.
     */
    void new_step() {
        ++mStepVersion;
    }

    /**
     * @brief Legacy alias for new_step() - invalidates cache by starting new step
     */
    void invalidate_cache() {
        new_step();
    }

    /**
     * @brief Get embeddings (not quantized)
     */
    Tensor& get_embeddings(cudaStream_t stream) {
        (void)stream;
        return mBnBWeights->get_embeddings().embedding;
    }

    /**
     * @brief Get final norm weight (not quantized)
     */
    Tensor& get_final_norm(cudaStream_t stream) {
        (void)stream;
        return mBnBWeights->get_embeddings().final_norm;
    }

    /**
     * @brief Get LM head (not quantized, may be tied to embeddings)
     */
    Tensor& get_lm_head(cudaStream_t stream) {
        (void)stream;
        return mBnBWeights->get_embeddings().lm_head;
    }

    /**
     * @brief Access the underlying BnBWeightsManager
     */
    BnBWeightsManager& bnb_weights() { return *mBnBWeights; }
    const BnBWeightsManager& bnb_weights() const { return *mBnBWeights; }

    /**
     * @brief Get QLoRA config
     */
    const QLoRAConfig& qlora_config() const { return mConfig.qlora_config; }

    /**
     * @brief Get memory stats
     */
    std::size_t quantized_weights_bytes() const {
        return mBnBWeights->quantized_weights_bytes();
    }

    float memory_savings_ratio() const {
        return mBnBWeights->memory_savings_ratio();
    }

private:
    Config mConfig;
    TensorAllocator* mAllocator;
    cudaDeviceProp mDeviceProps;  // Store by value to avoid dangling pointer

    // The underlying BnB weights manager (owns quantized weights)
    std::unique_ptr<BnBWeightsManager> mBnBWeights;

    // Dequantization buffers for each projection type (BF16)
    // We allocate separate buffers for each weight type to avoid conflicts
    Tensor mDequantQKV;      // For QKV projection
    Tensor mDequantOut;      // For output projection
    Tensor mDequantGateUp;   // For gate+up projection
    Tensor mDequantDown;     // For down projection

    // Cached dequantized block weights (reused across layers)
    BlockWeights mDequantBlock;

    // =========================================================================
    // Zero-overhead forward/backward cache via step versioning
    // =========================================================================
    // Instead of allocating per-layer caches, we track which layer is currently
    // in the shared dequant buffers and what step version it was dequantized in.
    // If get_block() is called for the same layer in the same step, we skip
    // dequantization (this happens when backward accesses the same layer as forward).

    int mCurrentLayer = -1;       ///< Layer index currently in dequant buffers
    uint64_t mStepVersion = 0;    ///< Current training step version
    uint64_t mBufferVersion = 0;  ///< Step version when buffers were last filled

    void allocate_dequant_buffers();
    void setup_block_weights_structure();
    void dequantize_weight(const BnBBlockQuantizedWeight& src, Tensor& dst, cudaStream_t stream);
};

// ============================================================================
// Implementation
// ============================================================================

template<typename Block>
BnBWeightProvider<Block>::BnBWeightProvider(
    const Config& config, TensorAllocator& allocator, const cudaDeviceProp& device_props)
    : mConfig(config)
    , mAllocator(&allocator)
    , mDeviceProps(device_props)  // Copy by value
{
    // Create BnB weights manager
    BnBWeightsManager::Config bw_config{
        .num_layers = config.num_layers,
        .hidden_size = config.hidden_size,
        .intermediate_size = config.intermediate_size,
        .num_query_heads = config.num_query_heads,
        .num_kv_heads = config.num_kv_heads,
        .head_size = config.head_size,
        .vocab_size = config.vocab_size,
        .qlora_config = config.qlora_config,
        .use_qk_norm = config.use_qk_norm,
        .tied_embeddings = config.tied_embeddings,
        .shard_idx = config.shard_idx,
        .num_shards = config.num_shards
    };
    mBnBWeights = std::make_unique<BnBWeightsManager>(bw_config, allocator, device_props);

    // Allocate dequantization buffers
    allocate_dequant_buffers();

    // Set up the block weights structure with pointers to dequant buffers
    setup_block_weights_structure();
}

template<typename Block>
void BnBWeightProvider<Block>::allocate_dequant_buffers() {
    auto ctx = mAllocator->with_context("BnB_DequantBuf");

    const int hidden = mConfig.hidden_size;
    const int intermediate = mConfig.intermediate_size;
    const int num_q_heads = mConfig.num_query_heads;
    const int num_kv_heads = mConfig.num_kv_heads;
    const int head_size = mConfig.head_size;

    const int qkv_out = (num_q_heads + 2 * num_kv_heads) * head_size;

    // Allocate dequantization buffers - single layer, reused across all layers
    // NF4 always dequantizes to BF16
    mDequantQKV = mAllocator->allocate(ETensorDType::BF16, "dequant_qkv",
                                        EAllocationType::ON_DEVICE,
                                        {(long)qkv_out, (long)hidden});

    mDequantOut = mAllocator->allocate(ETensorDType::BF16, "dequant_out",
                                        EAllocationType::ON_DEVICE,
                                        {(long)hidden, (long)(num_q_heads * head_size)});

    mDequantGateUp = mAllocator->allocate(ETensorDType::BF16, "dequant_gate_up",
                                           EAllocationType::ON_DEVICE,
                                           {(long)(2 * intermediate), (long)hidden});

    mDequantDown = mAllocator->allocate(ETensorDType::BF16, "dequant_down",
                                         EAllocationType::ON_DEVICE,
                                         {(long)hidden, (long)intermediate});
}

template<typename Block>
void BnBWeightProvider<Block>::setup_block_weights_structure() {
    // Point the dequant block's weight tensors to our buffers
    // Note: We only set up the main projection weights here.
    // Layer norm weights are set per-layer since they're small and stored in BF16.

    // Set up attention weights
    mDequantBlock.attention.qkv_weight = mDequantQKV;
    mDequantBlock.attention.out_weight = mDequantOut;

    // Set up MLP weights
    mDequantBlock.mlp_up_weight = mDequantGateUp;
    mDequantBlock.mlp_down_weight = mDequantDown;
}

template<typename Block>
void BnBWeightProvider<Block>::import_and_quantize(
    const std::string& file_name, NCCLCommunicator& comm, cudaStream_t stream) {

    // Import and quantize base model weights
    mBnBWeights->import_and_quantize(file_name, comm, stream);
}

template<typename Block>
void BnBWeightProvider<Block>::dequantize_weight(const BnBBlockQuantizedWeight& src,
                                                  Tensor& dst, cudaStream_t stream) {
    if (src.double_quant) {
        // Use double-dequantization kernel: INT8 absmax → FP32 → NF4 dequant
        dequantize_bnb_nf4_double(
            dst.get<nv_bfloat16>(),
            src.data.get<unsigned char>(),
            src.absmax.get<unsigned char>(),
            src.absmax_scale.get<float>(),
            src.absmax_offset.get<float>(),
            src.M, src.K,
            src.block_size, src.double_quant_group_size,
            mDeviceProps, stream);
    } else {
        // Standard dequantization: FP32 absmax → NF4 dequant
        dequantize_bnb_nf4(
            dst.get<nv_bfloat16>(),
            src.data.get<unsigned char>(),
            src.absmax.get<float>(),
            src.M, src.K,
            src.block_size,
            mDeviceProps, stream);
    }
}

template<typename Block>
typename BnBWeightProvider<Block>::BlockWeights& BnBWeightProvider<Block>::get_block(
    int layer_idx, cudaStream_t stream) {

    const auto& qblock = mBnBWeights->get_bnb_block(layer_idx);

    // Check if we already have this layer dequantized in the current step
    // This happens when backward accesses the same layer that forward just used
    const bool cache_hit = (mCurrentLayer == layer_idx) && (mBufferVersion == mStepVersion);

    if (!cache_hit) {
        // Cache miss: need to dequantize weights

        // QKV projection
        dequantize_weight(qblock.qkv_proj, mDequantQKV, stream);

        // Output projection
        dequantize_weight(qblock.out_proj, mDequantOut, stream);

        // Gate+Up projection
        dequantize_weight(qblock.gate_up_proj, mDequantGateUp, stream);

        // Down projection
        dequantize_weight(qblock.down_proj, mDequantDown, stream);

        // Update cache metadata
        mCurrentLayer = layer_idx;
        mBufferVersion = mStepVersion;
    }
    // else: cache hit - skip dequantization, reuse existing buffer contents

    // Always update layer norm pointers (they're just references, not cached data)
    mDequantBlock.ln1.weight = qblock.ln1_weight;
    mDequantBlock.ln2.weight = qblock.ln2_weight;

    // Copy QK-norm weights if present (for models like Qwen3)
    if constexpr (requires { mDequantBlock.attention.q_norm_weight; mDequantBlock.attention.k_norm_weight; }) {
        if (qblock.q_norm_weight.has_value() && qblock.k_norm_weight.has_value()) {
            mDequantBlock.attention.q_norm_weight = qblock.q_norm_weight;
            mDequantBlock.attention.k_norm_weight = qblock.k_norm_weight;
        }
    }

    return mDequantBlock;
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_QLORA_BNB_WEIGHT_PROVIDER_H
