// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_QLORA_QLORA_WEIGHT_PROVIDER_H
#define SUROGATE_SRC_MODULES_QLORA_QLORA_WEIGHT_PROVIDER_H

#include <memory>
#include <vector>

#include "qlora_config.h"
#include "fp8_weights.h"
#include "block_quantized_tensor.h"
#include "modules/composite/transformer_block.h"
#include "modules/weights/weight_manager_types.h"
#include "modules/weights/weight_manager.h"
#include "modules/weights/weight_manager_helpers.h"
#include "modules/weights/weight_manager_allocation.h"
#include "modules/weights/weight_manager_gather.h"
#include "modules/weights/weight_manager_optimizer.h"
#include "modules/weights/weight_manager_io.h"
#include "modules/weights/weight_manager_quantization.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"

namespace modules {

/**
 * @brief Provides dequantized weights for QLoRA FP8 training
 *
 * This class wraps FP8WeightsManager and provides on-the-fly dequantization
 * of FP8 base weights to BF16 for use in the forward pass.
 *
 * Key design:
 * - Quantized weights (FP8 + per-block scales) are stored permanently
 * - Dequantization buffers are allocated once and reused
 * - get_block() dequantizes the requested layer's weights
 * - Compatible with the existing ModularWeightManager interface patterns
 *
 * Optimization: Forward/Backward Dequant Caching
 * - Since base weights are frozen in QLoRA, dequantized weights are identical
 *   between forward and backward passes within a single training step
 * - Uses step versioning to detect when the same layer is accessed twice
 *   within one step (forward then backward) and skips redundant dequantization
 * - Default mode (zero overhead): Only caches last accessed layer, saving
 *   4 dequant kernels per step (when backward starts with same layer forward ended)
 * - Step version is incremented via new_step() at the start of each training step
 *
 * @tparam Block The transformer block type (e.g., DenseTransformerBlock<>)
 */
template<typename Block>
class FP8WeightProvider {
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
        bool use_qk_norm = false;  ///< Whether model uses QK-norm (Qwen3)
        bool tied_embeddings = true;  ///< Whether lm_head is tied to embeddings
        int shard_idx = 0;
        int num_shards = 1;
        bool enable_fp8_forward = false;  ///< Use FP8 for forward pass (skip dequant)
        bool enable_fp8_hybrid = false;   ///< Use FP8 hybrid mode (skip dequant)
    };

    FP8WeightProvider(const Config& config, TensorAllocator& allocator,
                        const cudaDeviceProp& device_props);
    ~FP8WeightProvider() = default;

    /**
     * @brief Import and quantize base model weights from file
     */
    void import_and_quantize(const std::string& file_name, NCCLCommunicator& comm,
                             cudaStream_t stream);

    /**
     * @brief Get dequantized block weights
     *
     * Dequantizes the FP8 weights for the specified layer and returns
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
        return mFP8Weights->get_embeddings().embedding;
    }

    /**
     * @brief Get final norm weight (not quantized)
     */
    Tensor& get_final_norm(cudaStream_t stream);

    /**
     * @brief Get LM head (not quantized, may be tied to embeddings)
     */
    Tensor& get_lm_head(cudaStream_t stream) {
        (void)stream;
        return mFP8Weights->get_embeddings().lm_head;
    }

    /**
     * @brief Access the underlying FP8WeightsManager
     */
    FP8WeightsManager& fp8_weights() { return *mFP8Weights; }
    const FP8WeightsManager& fp8_weights() const { return *mFP8Weights; }

    /**
     * @brief Get QLoRA config
     */
    const QLoRAConfig& qlora_config() const { return mConfig.qlora_config; }

    /**
     * @brief Get memory stats
     */
    std::size_t quantized_weights_bytes() const {
        return mFP8Weights->quantized_weights_bytes();
    }

    float memory_savings_ratio() const {
        return mFP8Weights->memory_savings_ratio();
    }

    /**
     * @brief Check if FP8 forward weight caching is enabled
     */
    [[nodiscard]] bool has_fp8_forward_cache() const {
        return use_native_fp8();
    }

    /**
     * @brief Type alias for compatibility with ModularWeightManager
     */
    using FP8WeightCache = typename ModularWeightManager<Block>::FP8WeightCache;

    /**
     * @brief Get the FP8 weight cache for the current layer
     *
     * Returns a reference to an FP8WeightCache structure that points to the
     * FP8 weights that were prepared during get_block().
     * This is compatible with ModularWeightManager's fp8_weight_cache() interface.
     * Valid only after get_block() is called for the same layer.
     */
    [[nodiscard]] FP8WeightCache& get_fp8_cache() {
        return mFP8WeightCacheAlias;
    }

    [[nodiscard]] const FP8WeightCache& get_fp8_cache() const {
        return mFP8WeightCacheAlias;
    }

    /**
     * @brief Get the block weights (legacy interface)
     */
    [[nodiscard]] BlockWeights& fp8_weight_cache() {
        return mFP8Block;
    }

    [[nodiscard]] const BlockWeights& fp8_weight_cache() const {
        return mFP8Block;
    }

private:
    Config mConfig;
    TensorAllocator* mAllocator;
    cudaDeviceProp mDeviceProps;  // Store by value to avoid dangling pointer

    // The underlying QLoRA weights manager (owns quantized weights + LoRA adapters)
    std::unique_ptr<FP8WeightsManager> mFP8Weights;

    // Dequantization buffers for each projection type
    // We allocate separate buffers for each weight type to avoid conflicts
    Tensor mDequantQKV;      // For QKV projection (BF16 or FP8 depending on mode)
    Tensor mDequantOut;      // For output projection
    Tensor mDequantGateUp;   // For gate+up projection
    Tensor mDequantDown;     // For down projection

    // Per-tensor FP8 scales (when use_native_fp8() == true)
    Tensor mFP8ScaleQKV;     // Single scale for QKV
    Tensor mFP8ScaleOut;     // Single scale for output projection
    Tensor mFP8ScaleGateUp;  // Single scale for gate+up
    Tensor mFP8ScaleDown;    // Single scale for down projection

    // Final norm weight (loaded separately, not quantized)
    Tensor mFinalNormWeight;

    // Cached dequantized block weights (reused across layers)
    BlockWeights mDequantBlock;

    // FP8 block weights (no dequantization, used when FP8 matmuls are enabled)
    BlockWeights mFP8Block;

    // FP8 weight cache alias (for compatibility with ModularWeightManager interface)
    // Points to the same tensors as mFP8Block but in a different structure
    FP8WeightCache mFP8WeightCacheAlias;

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
    void setup_fp8_block_weights_structure();

    /**
     * @brief Check if we should use native FP8 weights (no dequantization)
     */
    [[nodiscard]] bool use_native_fp8() const {
        return (mConfig.enable_fp8_forward || mConfig.enable_fp8_hybrid);
    }
};

// ============================================================================
// Implementation
// ============================================================================

template<typename Block>
FP8WeightProvider<Block>::FP8WeightProvider(
    const Config& config, TensorAllocator& allocator, const cudaDeviceProp& device_props)
    : mConfig(config)
    , mAllocator(&allocator)
    , mDeviceProps(device_props)  // Copy by value
{
    // Create QLoRA weights manager
    FP8WeightsManager::Config qw_config{
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
    mFP8Weights = std::make_unique<FP8WeightsManager>(qw_config, allocator, device_props);

    // Allocate dequantization buffers
    allocate_dequant_buffers();

    // Set up the block weights structure with pointers to dequant buffers
    setup_block_weights_structure();

    // Set up FP8 block weights structure (for native FP8 path)
    setup_fp8_block_weights_structure();
}

template<typename Block>
void FP8WeightProvider<Block>::allocate_dequant_buffers() {
    auto ctx = mAllocator->with_context("FP8_DequantBuf");

    const int hidden = mConfig.hidden_size;
    const int intermediate = mConfig.intermediate_size;
    const int num_q_heads = mConfig.num_query_heads;
    const int num_kv_heads = mConfig.num_kv_heads;
    const int head_size = mConfig.head_size;

    const int qkv_out = (num_q_heads + 2 * num_kv_heads) * head_size;

    // Determine output dtype based on FP8 mode
    const ETensorDType dequant_dtype = use_native_fp8() ? ETensorDType::FP8_E4M3 : ETensorDType::BF16;

    // Allocate dequantization buffers - single layer, reused across all layers
    // When use_native_fp8() is true, these will be FP8 with per-tensor scales
    // When false, these will be BF16 (legacy path)
    mDequantQKV = mAllocator->allocate(dequant_dtype, "dequant_qkv",
                                        EAllocationType::ON_DEVICE,
                                        {(long)qkv_out, (long)hidden});

    mDequantOut = mAllocator->allocate(dequant_dtype, "dequant_out",
                                        EAllocationType::ON_DEVICE,
                                        {(long)hidden, (long)(num_q_heads * head_size)});

    mDequantGateUp = mAllocator->allocate(dequant_dtype, "dequant_gate_up",
                                           EAllocationType::ON_DEVICE,
                                           {(long)(2 * intermediate), (long)hidden});

    mDequantDown = mAllocator->allocate(dequant_dtype, "dequant_down",
                                         EAllocationType::ON_DEVICE,
                                         {(long)hidden, (long)intermediate});

    // Allocate per-tensor scale buffers (only needed for FP8 mode)
    // Note: Each buffer needs 2 floats [absmax, scale] because tensor.scale() returns Stats+1
    if (use_native_fp8()) {
        mFP8ScaleQKV = mAllocator->allocate(ETensorDType::FP32, "fp8_scale_qkv",
                                             EAllocationType::ON_DEVICE, {2});
        mFP8ScaleOut = mAllocator->allocate(ETensorDType::FP32, "fp8_scale_out",
                                             EAllocationType::ON_DEVICE, {2});
        mFP8ScaleGateUp = mAllocator->allocate(ETensorDType::FP32, "fp8_scale_gate_up",
                                                EAllocationType::ON_DEVICE, {2});
        mFP8ScaleDown = mAllocator->allocate(ETensorDType::FP32, "fp8_scale_down",
                                              EAllocationType::ON_DEVICE, {2});
    }

    // Final norm weight
    mFinalNormWeight = mAllocator->allocate(ETensorDType::BF16, "final_norm",
                                             EAllocationType::ON_DEVICE,
                                             {(long)hidden});
}

template<typename Block>
void FP8WeightProvider<Block>::setup_block_weights_structure() {
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
void FP8WeightProvider<Block>::setup_fp8_block_weights_structure() {
    // FP8 block structure: Set up pointers to FP8 buffers with attached scales
    // These will be filled with per-tensor FP8 weights (converted from per-block)
    // in get_block() using the fused kernel

    if (use_native_fp8()) {
        // Attach the per-tensor FP8 scales to the weight tensors.
        // tensor.scale() returns Stats+1, so we set Stats to the buffer base.
        // The fused kernel will write to buffer[1] (see get_block).
        mDequantQKV.Stats = reinterpret_cast<float*>(mFP8ScaleQKV.Data);
        mDequantOut.Stats = reinterpret_cast<float*>(mFP8ScaleOut.Data);
        mDequantGateUp.Stats = reinterpret_cast<float*>(mFP8ScaleGateUp.Data);
        mDequantDown.Stats = reinterpret_cast<float*>(mFP8ScaleDown.Data);
    }

    mFP8Block.attention.qkv_weight = mDequantQKV;
    mFP8Block.attention.out_weight = mDequantOut;
    mFP8Block.mlp_up_weight = mDequantGateUp;
    mFP8Block.mlp_down_weight = mDequantDown;

    // Also set up the FP8WeightCache alias for compatibility with weight manager interface
    mFP8WeightCacheAlias.qkv_weight = mDequantQKV;
    mFP8WeightCacheAlias.o_weight = mDequantOut;
    mFP8WeightCacheAlias.mlp_up_weight = mDequantGateUp;
    mFP8WeightCacheAlias.mlp_down_weight = mDequantDown;
}

template<typename Block>
void FP8WeightProvider<Block>::import_and_quantize(
    const std::string& file_name, NCCLCommunicator& comm, cudaStream_t stream) {

    // Import and quantize base model weights
    mFP8Weights->import_and_quantize(file_name, comm, stream);

    // Note: LoRA adapters are managed by ModularLoRAModel's mLoRAWeights, not here.
    // The internal FP8WeightsManager's LoRA weights are not used.

    // Load final norm weight from file (not quantized)
    SafeTensorsReader reader(file_name);
    const std::vector<std::string> final_norm_names = {
        "model.norm.weight",
        "transformer.ln_f.weight",
        "model.final_layernorm.weight"
    };
    for (const auto& name : final_norm_names) {
        bool found = false;
        for (const auto& entry : reader.entries()) {
            if (entry.name() == name) {
                entry.read_tensor(mFinalNormWeight, /*allow_cast=*/true);
                found = true;
                break;
            }
        }
        if (found) break;
    }
}

template<typename Block>
typename FP8WeightProvider<Block>::BlockWeights& FP8WeightProvider<Block>::get_block(int layer_idx, cudaStream_t stream) {
    const auto& qblock = mFP8Weights->get_quantized_block(layer_idx);

    // **OPTIMIZATION: Native FP8 Path (Fused Conversion)**
    // When FP8 matmuls are enabled, use fused_dequant_requant_per_block_to_tensor()
    // to convert per-block FP8 → per-tensor FP8 in a single operation.
    // This eliminates intermediate BF16 storage and reduces kernel launches from 3 to 2.
    const bool use_fp8_path = use_native_fp8();

    // Check if we already have this layer converted in the current step
    // This happens when backward accesses the same layer that forward just used
    const bool cache_hit = (mCurrentLayer == layer_idx) && (mBufferVersion == mStepVersion);

    if (!cache_hit) {
        // Cache miss: need to convert weights
        const int block_size = mConfig.qlora_config.block_size();

        if (use_fp8_path) {
            // **FP8 OPTIMIZED PATH**: Fused per-block → per-tensor FP8 conversion
            // Uses 2 kernel launches per projection (absmax + requant) instead of
            // 3 (dequant + per-tensor quant + scale computation)
            //
            // The fused kernel writes [absmax, scale] to the output buffer.
            // This matches Tensor's layout where Stats[0]=absmax, Stats[1]=scale.

            fused_dequant_requant_per_block_to_tensor(
                mDequantQKV.get<__nv_fp8_e4m3>(),
                mFP8ScaleQKV.get<float>(),
                qblock.qkv_proj.data.get<__nv_fp8_e4m3>(),
                qblock.qkv_proj.block_scales.get<float>(),
                qblock.qkv_proj.M, qblock.qkv_proj.K,
                block_size, mDeviceProps, stream);

            fused_dequant_requant_per_block_to_tensor(
                mDequantOut.get<__nv_fp8_e4m3>(),
                mFP8ScaleOut.get<float>(),
                qblock.out_proj.data.get<__nv_fp8_e4m3>(),
                qblock.out_proj.block_scales.get<float>(),
                qblock.out_proj.M, qblock.out_proj.K,
                block_size, mDeviceProps, stream);

            fused_dequant_requant_per_block_to_tensor(
                mDequantGateUp.get<__nv_fp8_e4m3>(),
                mFP8ScaleGateUp.get<float>(),
                qblock.gate_up_proj.data.get<__nv_fp8_e4m3>(),
                qblock.gate_up_proj.block_scales.get<float>(),
                qblock.gate_up_proj.M, qblock.gate_up_proj.K,
                block_size, mDeviceProps, stream);

            fused_dequant_requant_per_block_to_tensor(
                mDequantDown.get<__nv_fp8_e4m3>(),
                mFP8ScaleDown.get<float>(),
                qblock.down_proj.data.get<__nv_fp8_e4m3>(),
                qblock.down_proj.block_scales.get<float>(),
                qblock.down_proj.M, qblock.down_proj.K,
                block_size, mDeviceProps, stream);

        } else {
            // **BF16 LEGACY PATH**: Dequantize to BF16
            dequantize_per_block(
                mDequantQKV.get<nv_bfloat16>(),
                qblock.qkv_proj.data.get<__nv_fp8_e4m3>(),
                qblock.qkv_proj.block_scales.get<float>(),
                qblock.qkv_proj.M, qblock.qkv_proj.K,
                block_size, mDeviceProps, stream);

            dequantize_per_block(
                mDequantOut.get<nv_bfloat16>(),
                qblock.out_proj.data.get<__nv_fp8_e4m3>(),
                qblock.out_proj.block_scales.get<float>(),
                qblock.out_proj.M, qblock.out_proj.K,
                block_size, mDeviceProps, stream);

            dequantize_per_block(
                mDequantGateUp.get<nv_bfloat16>(),
                qblock.gate_up_proj.data.get<__nv_fp8_e4m3>(),
                qblock.gate_up_proj.block_scales.get<float>(),
                qblock.gate_up_proj.M, qblock.gate_up_proj.K,
                block_size, mDeviceProps, stream);

            dequantize_per_block(
                mDequantDown.get<nv_bfloat16>(),
                qblock.down_proj.data.get<__nv_fp8_e4m3>(),
                qblock.down_proj.block_scales.get<float>(),
                qblock.down_proj.M, qblock.down_proj.K,
                block_size, mDeviceProps, stream);
        }

        // Update cache metadata
        mCurrentLayer = layer_idx;
        mBufferVersion = mStepVersion;
    }
    // else: cache hit - skip conversion, reuse existing buffer contents

    // Always update layer norm pointers (they're just references, not cached data)
    if (use_fp8_path) {
        mFP8Block.ln1.weight = qblock.ln1_weight;
        mFP8Block.ln2.weight = qblock.ln2_weight;

        // Copy QK-norm weights if present (for models like Qwen3)
        if constexpr (requires { mFP8Block.attention.q_norm_weight; mFP8Block.attention.k_norm_weight; }) {
            if (qblock.q_norm_weight.has_value() && qblock.k_norm_weight.has_value()) {
                mFP8Block.attention.q_norm_weight = qblock.q_norm_weight;
                mFP8Block.attention.k_norm_weight = qblock.k_norm_weight;
            }
        }

        return mFP8Block;
    } else {
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
}

template<typename Block>
Tensor& FP8WeightProvider<Block>::get_final_norm(cudaStream_t stream) {
    (void)stream;
    return mFinalNormWeight;
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_QLORA_QLORA_WEIGHT_PROVIDER_H
