// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// FP4 weight provider: on-the-fly dequantization of FP4 weights to BF16

#ifndef SUROGATE_SRC_MODULES_QLORA_FP4_WEIGHT_PROVIDER_H
#define SUROGATE_SRC_MODULES_QLORA_FP4_WEIGHT_PROVIDER_H

#include <memory>
#include <vector>

#include "fp4_weights.h"
#include "fp4_block_quantized_tensor.h"
#include "qlora_config.h"
#include "kernels/kernels.h"
#include "modules/composite/transformer_block.h"
#include "modules/lora/lora_config.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"
#include "utilities/utils.h"

namespace modules {

/**
 * @brief Provides dequantized weights for FP4 QLoRA training
 *
 * This class wraps FP4WeightsManager and provides on-the-fly dequantization
 * of FP4 base weights to BF16 for use in the forward pass.
 *
 * Key design:
 * - Quantized weights (FP4 + two-level block scales) are stored permanently
 * - Dequantization buffers are allocated once and reused
 * - get_block() dequantizes the requested layer's weights to BF16
 * - Compatible with the existing ModularWeightManager interface patterns
 *
 * FP4 uses two-level block scaling:
 * - Level 1: FP8 E4M3 scale per 16 consecutive values
 * - Level 2: FP32 global per-tensor scale (amax)
 *
 * @tparam Block The transformer block type (e.g., DenseTransformerBlock<>)
 */
template<typename Block>
class FP4WeightProvider {
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
        bool use_qk_norm = false;
        bool tied_embeddings = true;
        int shard_idx = 0;
        int num_shards = 1;
    };

    FP4WeightProvider(const Config& config, TensorAllocator& allocator,
                      const cudaDeviceProp& device_props);
    ~FP4WeightProvider() = default;

    /**
     * @brief Import and quantize base model weights from file
     */
    void import_and_quantize(const std::string& file_name, NCCLCommunicator& comm,
                             cudaStream_t stream);

    /**
     * @brief Get dequantized block weights
     *
     * Dequantizes the FP4 weights for the specified layer and returns
     * a BlockWeights struct with BF16 tensors ready for matmul.
     *
     * Uses caching to avoid redundant dequantization within the same step
     * (forward and backward access the same layer).
     *
     * @param layer_idx Layer index
     * @param stream CUDA stream for dequantization
     * @return Reference to BlockWeights with dequantized tensors
     */
    BlockWeights& get_block(int layer_idx, cudaStream_t stream);

    /**
     * @brief Release block weights (no-op, kept for interface compat)
     */
    void release_block(int layer_idx, cudaStream_t stream) {
        (void)layer_idx;
        (void)stream;
    }

    /**
     * @brief Signal the start of a new training step
     */
    void new_step() {
        ++mStepVersion;
    }

    /**
     * @brief Legacy alias for new_step()
     */
    void invalidate_cache() {
        new_step();
    }

    /**
     * @brief Get embeddings (not quantized)
     */
    Tensor& get_embeddings(cudaStream_t stream) {
        (void)stream;
        return mFP4Weights->get_embeddings().embedding;
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
        return mFP4Weights->get_embeddings().lm_head;
    }

    /**
     * @brief Access the underlying FP4WeightsManager
     */
    FP4WeightsManager& fp4_weights() { return *mFP4Weights; }
    const FP4WeightsManager& fp4_weights() const { return *mFP4Weights; }

    /**
     * @brief Get QLoRA config
     */
    const QLoRAConfig& qlora_config() const { return mConfig.qlora_config; }

    /**
     * @brief Get memory stats
     */
    std::size_t quantized_weights_bytes() const {
        return mFP4Weights->quantized_weights_bytes();
    }

    float memory_savings_ratio() const {
        return mFP4Weights->memory_savings_ratio();
    }

private:
    Config mConfig;
    TensorAllocator* mAllocator;
    const cudaDeviceProp* mDeviceProps;

    // The underlying FP4 weights manager (owns quantized weights)
    std::unique_ptr<FP4WeightsManager> mFP4Weights;

    // Dequantization buffers (BF16, single layer, reused)
    Tensor mDequantQKV;
    Tensor mDequantOut;
    Tensor mDequantGateUp;
    Tensor mDequantDown;

    // Final norm weight (not quantized)
    Tensor mFinalNormWeight;

    // Cached dequantized block weights
    BlockWeights mDequantBlock;

    // Zero-overhead forward/backward cache via step versioning
    int mCurrentLayer = -1;
    uint64_t mStepVersion = 0;
    uint64_t mBufferVersion = 0;

    void allocate_dequant_buffers();
    void setup_block_weights_structure();
};

// ============================================================================
// Implementation
// ============================================================================

template<typename Block>
FP4WeightProvider<Block>::FP4WeightProvider(
    const Config& config, TensorAllocator& allocator, const cudaDeviceProp& device_props)
    : mConfig(config)
    , mAllocator(&allocator)
    , mDeviceProps(&device_props)
{
    // Create FP4 weights manager
    FP4WeightsManager::Config fp4_config{
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
    mFP4Weights = std::make_unique<FP4WeightsManager>(fp4_config, allocator, device_props);

    // Allocate dequantization buffers
    allocate_dequant_buffers();

    // Set up the block weights structure with pointers to dequant buffers
    setup_block_weights_structure();
}

template<typename Block>
void FP4WeightProvider<Block>::allocate_dequant_buffers() {
    auto ctx = mAllocator->with_context("FP4_DequantBuf");

    const int hidden = mConfig.hidden_size;
    const int intermediate = mConfig.intermediate_size;
    const int num_q_heads = mConfig.num_query_heads;
    const int num_kv_heads = mConfig.num_kv_heads;
    const int head_size = mConfig.head_size;

    const int qkv_out = (num_q_heads + 2 * num_kv_heads) * head_size;

    // Allocate BF16 dequantization buffers - single layer, reused across all layers
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

    // Final norm weight (not quantized)
    mFinalNormWeight = mAllocator->allocate(ETensorDType::BF16, "final_norm",
                                             EAllocationType::ON_DEVICE,
                                             {(long)hidden});
}

template<typename Block>
void FP4WeightProvider<Block>::setup_block_weights_structure() {
    // Point the dequant block's weight tensors to our buffers
    mDequantBlock.attention.qkv_weight = mDequantQKV;
    mDequantBlock.attention.out_weight = mDequantOut;
    mDequantBlock.mlp_up_weight = mDequantGateUp;
    mDequantBlock.mlp_down_weight = mDequantDown;
}

template<typename Block>
void FP4WeightProvider<Block>::import_and_quantize(
    const std::string& file_name, NCCLCommunicator& comm, cudaStream_t stream) {

    // Import and quantize base model weights to FP4
    mFP4Weights->import_and_quantize(file_name, comm, stream);

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
typename FP4WeightProvider<Block>::BlockWeights& FP4WeightProvider<Block>::get_block(
    int layer_idx, cudaStream_t stream) {

    const auto& fp4_block = mFP4Weights->get_fp4_block(layer_idx);

    // Check if we already have this layer dequantized in the current step
    const bool cache_hit = (mCurrentLayer == layer_idx) && (mBufferVersion == mStepVersion);

    if (!cache_hit) {
        // Cache miss: need to dequantize FP4 → BF16
        // Use the FP4 block dequantization kernel which handles two-level scaling:
        // FP4 data * FP8 block scale * global amax scale → BF16

        // QKV projection
        float qkv_scale = fp4_block.qkv_proj.global_decode_scale_rowwise();
        dequantize_fp4_block(
            mDequantQKV.get<nv_bfloat16>(),
            fp4_block.qkv_proj.data.get<uint8_t>(),
            fp4_block.qkv_proj.block_scales_rowwise.get<__nv_fp8_e4m3>(),
            qkv_scale,
            fp4_block.qkv_proj.M, fp4_block.qkv_proj.K,
            *mDeviceProps, stream);

        // Output projection
        float out_scale = fp4_block.out_proj.global_decode_scale_rowwise();
        dequantize_fp4_block(
            mDequantOut.get<nv_bfloat16>(),
            fp4_block.out_proj.data.get<uint8_t>(),
            fp4_block.out_proj.block_scales_rowwise.get<__nv_fp8_e4m3>(),
            out_scale,
            fp4_block.out_proj.M, fp4_block.out_proj.K,
            *mDeviceProps, stream);

        // Gate+Up projection
        float gate_up_scale = fp4_block.gate_up_proj.global_decode_scale_rowwise();
        dequantize_fp4_block(
            mDequantGateUp.get<nv_bfloat16>(),
            fp4_block.gate_up_proj.data.get<uint8_t>(),
            fp4_block.gate_up_proj.block_scales_rowwise.get<__nv_fp8_e4m3>(),
            gate_up_scale,
            fp4_block.gate_up_proj.M, fp4_block.gate_up_proj.K,
            *mDeviceProps, stream);

        // Down projection
        float down_scale = fp4_block.down_proj.global_decode_scale_rowwise();
        dequantize_fp4_block(
            mDequantDown.get<nv_bfloat16>(),
            fp4_block.down_proj.data.get<uint8_t>(),
            fp4_block.down_proj.block_scales_rowwise.get<__nv_fp8_e4m3>(),
            down_scale,
            fp4_block.down_proj.M, fp4_block.down_proj.K,
            *mDeviceProps, stream);

        // Synchronize to ensure dequantization completes before returning.
        // This fixes an intermittent hang that occurs when the dequant kernels
        // don't complete in time before subsequent matmul operations.
        // TODO: Investigate root cause - all operations are on the same stream
        // so this sync shouldn't be necessary in theory.
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Update cache metadata
        mCurrentLayer = layer_idx;
        mBufferVersion = mStepVersion;
    }

    // Update layer norm pointers (these are just references, not cached data)
    mDequantBlock.ln1.weight = fp4_block.ln1_weight;
    mDequantBlock.ln2.weight = fp4_block.ln2_weight;

    // Copy QK-norm weights if present (for models like Qwen3)
    if constexpr (requires { mDequantBlock.attention.q_norm_weight; mDequantBlock.attention.k_norm_weight; }) {
        if (fp4_block.q_norm_weight.has_value() && fp4_block.k_norm_weight.has_value()) {
            mDequantBlock.attention.q_norm_weight = fp4_block.q_norm_weight;
            mDequantBlock.attention.k_norm_weight = fp4_block.k_norm_weight;
        }
    }

    return mDequantBlock;
}

template<typename Block>
Tensor& FP4WeightProvider<Block>::get_final_norm(cudaStream_t stream) {
    (void)stream;
    return mFinalNormWeight;
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_QLORA_FP4_WEIGHT_PROVIDER_H
