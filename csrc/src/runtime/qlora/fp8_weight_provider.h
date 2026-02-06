// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_QLORA_QLORA_WEIGHT_PROVIDER_H
#define SUROGATE_SRC_MODULES_QLORA_QLORA_WEIGHT_PROVIDER_H

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>
#include <iostream>

#include <cuda_runtime_api.h>

#include "qlora_config.h"
#include "fp8_weights.h"
#include "block_quantized_tensor.h"
#include "moe_weights.h"
#include "hf_mapping.h"
#include "dsl_block_weights.h"
#include "weight_provider_types.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"
#include "runtime/moe/moe_types.h"

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
        int mlp_up_factor = 2;      ///< 2 for gated (SwiGLU), 1 for non-gated (ReLU2)
        QLoRAConfig qlora_config;
        ModularLoRAConfig lora_config;
        ETensorDType model_dtype = ETensorDType::BF16;
        bool use_qk_norm = false;  ///< Whether model uses QK-norm (Qwen3)
        bool tied_embeddings = true;  ///< Whether lm_head is tied to embeddings
        int shard_idx = 0;
        int num_shards = 1;
        bool enable_fp8_forward = false;  ///< Use FP8 for forward pass (skip dequant)
        bool enable_fp8_hybrid = false;   ///< Use FP8 hybrid mode (skip dequant)
        const HfMapping* hf_mapping = nullptr;

        // Mamba / SSM config (optional; used by Nemotron-H hybrid blocks)
        bool has_mamba = false;
        std::vector<std::uint8_t> layer_is_mamba;
        int mamba_num_heads = 0;
        int mamba_head_dim = 0;
        int mamba_ssm_state_size = 0;
        int mamba_conv_kernel = 0;
        int mamba_n_groups = 1;
        int mamba_intermediate_size = 0;
        bool mamba_use_bias = false;
        bool mamba_use_conv_bias = false;

        /// Hybrid architecture pattern (e.g., "MEMEM*EMEMEM*...") where:
        /// M = Mamba, E = MoE, * = Attention, - = MLP
        /// If empty, assumes uniform architecture (all layers same type).
        std::string hybrid_pattern;
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
     * @brief alias for new_step() - invalidates cache by starting new step
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

    // =========================================================================
    // MoE Support
    // =========================================================================

    /**
     * @brief Check if this is an MoE model
     */
    [[nodiscard]] bool is_moe() const {
        return mFP8Weights && mFP8Weights->is_moe();
    }

    /**
     * @brief Get number of experts (0 for dense models)
     */
    [[nodiscard]] int num_experts() const {
        return mFP8Weights ? mFP8Weights->num_experts() : 0;
    }

    /**
     * @brief Get router gate weights for MoE (BF16, no dequant needed)
     *
     * Router gate is small and kept in BF16.
     *
     * @param layer_idx Layer index
     * @param stream CUDA stream (unused, kept for interface consistency)
     * @return Reference to router gate tensor (num_experts, hidden_size)
     */
    Tensor& get_router_gate(int layer_idx, cudaStream_t stream);

    /**
     * @brief Dequantize only the selected experts (selective dequantization)
     *
     * This method dequantizes only the experts that were selected by the router,
     * significantly reducing memory usage for MoE models with many experts.
     *
     * The dequantized weights are placed in a compact buffer indexed by
     * selection_info.expert_to_compact mapping.
     *
     * @param layer_idx Layer index
     * @param selection_info Information about which experts were selected
     * @param stream CUDA stream for dequantization
     */
    void dequantize_selected_experts(int layer_idx, const SelectiveExpertInfo& selection_info,
                                     cudaStream_t stream);

    /**
     * @brief Check if selective expert dequantization is enabled
     */
    [[nodiscard]] bool use_selective_dequant() const {
        return mConfig.qlora_config.selective_expert_dequant && mConfig.qlora_config.is_moe();
    }

    /**
     * @brief Get the current selection info (for backward pass)
     *
     * Returns the selection info cached from the last dequantize_selected_experts call.
     * This is used in backward to match the compact indexing used in forward.
     */
    [[nodiscard]] const SelectiveExpertInfo& get_current_selection() const {
        return mCurrentSelection;
    }

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
     * @brief Get the FP8 weight cache for the current layer
     *
     * Returns a reference to an FP8WeightCache structure that points to the
     * FP8 weights that were prepared during get_block().
     * This preserves the legacy fp8_weight_cache() interface shape.
     * Valid only after get_block() is called for the same layer.
     */
    [[nodiscard]] FP8WeightCache& get_fp8_cache() {
        return mFP8WeightCacheAlias;
    }

    [[nodiscard]] const FP8WeightCache& get_fp8_cache() const {
        return mFP8WeightCacheAlias;
    }

    /**
     * @brief Get the block weights (interface)
     */
    [[nodiscard]] BlockWeights& fp8_weight_cache() {
        return mFP8Block;
    }

    [[nodiscard]] const BlockWeights& fp8_weight_cache() const {
        return mFP8Block;
    }

private:
    /**
     * @brief Get attention and expert weights for MoE blocks
     *
     * Dequantizes attention weights (QKV + output) and ALL expert weights
     * into batched tensors for efficient forward pass.
     *
     * @param layer_idx Layer index
     * @param stream CUDA stream for dequantization
     */
    void get_moe_attention_weights(int layer_idx, cudaStream_t stream);
    void load_mamba_weights(const std::string& file_name);
    const HfMappingSpec* resolve_hf_spec(const std::string& internal_name,
                                         int& layer_idx,
                                         std::string& resolved_name) const;
    bool load_tensor_from_spec(const SafeTensorsReader& reader,
                               const HfMappingSpec& spec,
                               std::string_view internal_name,
                               int layer_idx,
                               int expert_idx,
                               Tensor& target,
                               cudaStream_t stream,
                               std::string_view warn_tag) const;

    template<typename T>
    static bool copy_tensor_range_host(const Tensor& t, long offset, long count,
                                       std::vector<T>& out) {
        if (!t.Data || offset < 0 || count <= 0) {
            return false;
        }
        const std::size_t total = t.nelem();
        const std::size_t end = static_cast<std::size_t>(offset) + static_cast<std::size_t>(count);
        if (end > total) {
            return false;
        }
        if (t.DType != dtype_from_type<T>) {
            return false;
        }
        out.resize(static_cast<std::size_t>(count));
        const std::size_t byte_offset = static_cast<std::size_t>(offset) * sizeof(T);
        const std::size_t byte_count = static_cast<std::size_t>(count) * sizeof(T);
        const std::byte* src = t.Data + byte_offset;
        if (t.Device < 0) {
            std::memcpy(out.data(), src, byte_count);
        } else {
            CUDA_CHECK(cudaMemcpy(out.data(), src, byte_count, cudaMemcpyDeviceToHost));
        }
        return true;
    }

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

    template<bool HasMamba, typename Dummy = void>
    struct MambaWeightsStorage {};
    template<typename Dummy>
    struct MambaWeightsStorage<true, Dummy> {
        std::vector<std::optional<typename BlockWeights::MambaWeights>> weights;
    };
    MambaWeightsStorage<has_mamba_weights<BlockWeights>::value> mMambaWeights;

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

    // =========================================================================
    // MoE-specific members for batched expert dequantization
    // =========================================================================

    /// Batched expert dequantization buffers (all experts, for forward pass)
    /// Shape: (num_experts, mlp_up_factor * moe_intermediate, hidden_size)
    Tensor mBatchedExpertGateUp;
    /// Shape: (num_experts, hidden_size, moe_intermediate)
    Tensor mBatchedExpertDown;

    /// Shared expert dequantization buffers (optional)
    Tensor mSharedExpertGateUp;
    Tensor mSharedExpertDown;
    int mSharedExpertD = 0;
    bool mHasSharedExpert = false;

    /// Number of experts in MoE model
    int mNumMoEExperts = 0;

    /// Current selection info for selective dequantization caching
    SelectiveExpertInfo mCurrentSelection;

    /// Layer index for which the current expert selection is valid
    /// Used to avoid reusing cached experts from a different layer
    int mCurrentExpertLayer = -1;

    /// Number of experts currently in the compact buffers
    int mNumActiveExperts = 0;

    void allocate_dequant_buffers();
    void allocate_moe_expert_buffers();
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
        .mlp_up_factor = config.mlp_up_factor,
        .qlora_config = config.qlora_config,
        .use_qk_norm = config.use_qk_norm,
        .tied_embeddings = config.tied_embeddings,
        .shard_idx = config.shard_idx,
        .num_shards = config.num_shards,
        .hf_mapping = config.hf_mapping
    };
    mFP8Weights = std::make_unique<FP8WeightsManager>(qw_config, allocator, device_props);

    if constexpr (has_mamba_weights<BlockWeights>::value) {
        if (mConfig.has_mamba) {
            mMambaWeights.weights.resize(mConfig.num_layers);
        }
    }

    // Allocate dequantization buffers
    allocate_dequant_buffers();

    // Allocate MoE expert buffers if needed
    if (mFP8Weights->is_moe()) {
        allocate_moe_expert_buffers();
    }

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
    const int mlp_M = mConfig.mlp_up_factor * intermediate;
    const int num_q_heads = mConfig.num_query_heads;
    const int num_kv_heads = mConfig.num_kv_heads;
    const int head_size = mConfig.head_size;

    const int qkv_out = (num_q_heads + 2 * num_kv_heads) * head_size;

    // Determine output dtype based on FP8 mode
    const ETensorDType dequant_dtype = use_native_fp8() ? ETensorDType::FP8_E4M3 : ETensorDType::BF16;

    // Allocate dequantization buffers - single layer, reused across all layers
    // When use_native_fp8() is true, these will be FP8 with per-tensor scales
    // When false, these will be BF16
    mDequantQKV = mAllocator->allocate(dequant_dtype, "dequant_qkv",
                                        EAllocationType::ON_DEVICE,
                                        {(long)qkv_out, (long)hidden});

    mDequantOut = mAllocator->allocate(dequant_dtype, "dequant_out",
                                        EAllocationType::ON_DEVICE,
                                        {(long)hidden, (long)(num_q_heads * head_size)});

    mDequantGateUp = mAllocator->allocate(dequant_dtype, "dequant_gate_up",
                                           EAllocationType::ON_DEVICE,
                                           {(long)mlp_M, (long)hidden});

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

    // Set up MLP weights (only for dense blocks - MoE blocks have experts instead)
    if constexpr (has_mlp_weights<BlockWeights>::value) {
        mDequantBlock.mlp_up_weight = mDequantGateUp;
        mDequantBlock.mlp_down_weight = mDequantDown;
    }

    // Set up MoE expert weights (batched layout)
    if constexpr (has_moe_weights<BlockWeights>::value) {
        if (mConfig.qlora_config.is_moe()) {
            mDequantBlock.experts.use_batched = true;
            mDequantBlock.experts.gate_up_proj = mBatchedExpertGateUp;
            mDequantBlock.experts.down_proj = mBatchedExpertDown;
            if (mHasSharedExpert) {
                mDequantBlock.shared_expert.emplace();
                mDequantBlock.shared_expert->gate_proj = mSharedExpertGateUp;
                mDequantBlock.shared_expert->down_proj = mSharedExpertDown;
                if (mConfig.mlp_up_factor == 1) {
                    mDequantBlock.shared_expert->up_proj = mSharedExpertGateUp;
                } else {
                    mDequantBlock.shared_expert->up_proj = Tensor();
                }
            }
        }
    }
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

    // Set up MLP weights (only for dense blocks - MoE blocks have experts instead)
    if constexpr (has_mlp_weights<BlockWeights>::value) {
        mFP8Block.mlp_up_weight = mDequantGateUp;
        mFP8Block.mlp_down_weight = mDequantDown;
    }

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
    load_mamba_weights(file_name);

    // Note: LoRA adapters are managed by ModularLoRAModel's mLoRAWeights, not here.
    // The internal FP8WeightsManager's LoRA weights are not used.

    // Load final norm weight from file (not quantized)
    SafeTensorsReader reader(file_name);
    bool final_norm_loaded = false;
    if (mConfig.hf_mapping) {
        int map_layer = -1;
        std::string resolved_name;
        if (const auto* spec = resolve_hf_spec("final_norm", map_layer, resolved_name)) {
            final_norm_loaded = load_tensor_from_spec(reader, *spec, resolved_name,
                                                     map_layer, -1, mFinalNormWeight,
                                                     stream, "FP8");
        }
    }
    if (!final_norm_loaded) {
        const std::vector<std::string> final_norm_names = {
            "model.norm.weight",
            "model.norm_f.weight",
            "backbone.norm.weight",
            "backbone.norm_f.weight",
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
}

template<typename Block>
void FP8WeightProvider<Block>::load_mamba_weights(const std::string& file_name) {
    if constexpr (!has_mamba_weights<BlockWeights>::value) {
        (void)file_name;
        return;
    } else {
        if (!mConfig.has_mamba) {
            return;
        }

        auto ctx = mAllocator->with_context("FP8_Mamba_Weights");
        SafeTensorsReader reader(file_name);

        auto find_entry_opt = [&](std::string_view name) -> const SafeTensorEntry* {
            for (const auto& entry : reader.entries()) {
                if (entry.name() == name) {
                    return &entry;
                }
            }
            return nullptr;
        };
        cudaStream_t mapping_stream = cudaStreamDefault;

        const int hidden = mConfig.hidden_size;
        const int mamba_dim = (mConfig.mamba_intermediate_size > 0)
            ? mConfig.mamba_intermediate_size
            : mConfig.intermediate_size;
        const int groups = std::max(1, mConfig.mamba_n_groups);
        const int conv_dim = mamba_dim + 2 * groups * mConfig.mamba_ssm_state_size;
        const int proj_size = mamba_dim + conv_dim + mConfig.mamba_num_heads;

        const auto load_tensor = [&](const std::string& name,
                                     std::string_view internal_suffix,
                                     Tensor& tensor,
                                     bool required,
                                     int layer_idx) {
            bool loaded = false;
            if (mConfig.hf_mapping && !internal_suffix.empty()) {
                int map_layer = -1;
                std::string resolved_name;
                const std::string internal_name = std::string("blocks[") + std::to_string(layer_idx) +
                                                  "].mamba." + std::string(internal_suffix);
                if (const auto* spec = resolve_hf_spec(internal_name, map_layer, resolved_name)) {
                    loaded = load_tensor_from_spec(reader, *spec, resolved_name,
                                                   map_layer, -1, tensor,
                                                   mapping_stream, "FP8");
                }
            }
            if (!loaded) {
                if (const auto* entry = find_entry_opt(name)) {
                    entry->read_tensor(tensor, /*allow_cast=*/true);
                    loaded = true;
                } else if (required) {
                    std::cerr << "[FP8 WARN] missing Mamba weight: " << name << "\n";
                }
            }
        };

        for (int layer = 0; layer < mConfig.num_layers; ++layer) {
            if (!mConfig.layer_is_mamba.empty() &&
                mConfig.layer_is_mamba[static_cast<std::size_t>(layer)] == 0) {
                continue;
            }

            const std::string model_prefix = "model.layers." + std::to_string(layer);
            const std::string backbone_prefix = "backbone.layers." + std::to_string(layer);
            const bool model_has = find_entry_opt(model_prefix + ".mixer.in_proj.weight") ||
                                   find_entry_opt(model_prefix + ".mixer.conv1d.weight") ||
                                   find_entry_opt(model_prefix + ".mixer.A_log");
            const std::string prefix = model_has ? model_prefix : backbone_prefix;

            auto& mw = mMambaWeights.weights[layer].emplace();

            mw.in_proj_weight = mAllocator->allocate(mConfig.model_dtype,
                                                     ("mamba_in_proj_w_l" + std::to_string(layer)).c_str(),
                                                     EAllocationType::ON_DEVICE, {proj_size, hidden});
            if (find_entry_opt(prefix + ".mixer.in_proj.bias")) {
                mw.in_proj_bias = mAllocator->allocate(mConfig.model_dtype,
                                                       ("mamba_in_proj_b_l" + std::to_string(layer)).c_str(),
                                                       EAllocationType::ON_DEVICE, {proj_size});
            }

            mw.out_proj_weight = mAllocator->allocate(mConfig.model_dtype,
                                                      ("mamba_out_proj_w_l" + std::to_string(layer)).c_str(),
                                                      EAllocationType::ON_DEVICE, {hidden, mamba_dim});
            if (find_entry_opt(prefix + ".mixer.out_proj.bias")) {
                mw.out_proj_bias = mAllocator->allocate(mConfig.model_dtype,
                                                        ("mamba_out_proj_b_l" + std::to_string(layer)).c_str(),
                                                        EAllocationType::ON_DEVICE, {hidden});
            }

            mw.conv1d_weight = mAllocator->allocate(mConfig.model_dtype,
                                                    ("mamba_conv_w_l" + std::to_string(layer)).c_str(),
                                                    EAllocationType::ON_DEVICE, {conv_dim, 1, mConfig.mamba_conv_kernel});
            if (find_entry_opt(prefix + ".mixer.conv1d.bias")) {
                mw.conv1d_bias = mAllocator->allocate(mConfig.model_dtype,
                                                      ("mamba_conv_b_l" + std::to_string(layer)).c_str(),
                                                      EAllocationType::ON_DEVICE, {conv_dim});
            }

            mw.A_log = mAllocator->allocate(ETensorDType::FP32,
                                            ("mamba_A_log_l" + std::to_string(layer)).c_str(),
                                            EAllocationType::ON_DEVICE, {mConfig.mamba_num_heads});
            mw.D = mAllocator->allocate(ETensorDType::FP32,
                                        ("mamba_D_l" + std::to_string(layer)).c_str(),
                                        EAllocationType::ON_DEVICE, {mConfig.mamba_num_heads});
            mw.dt_bias = mAllocator->allocate(ETensorDType::FP32,
                                              ("mamba_dt_bias_l" + std::to_string(layer)).c_str(),
                                              EAllocationType::ON_DEVICE, {mConfig.mamba_num_heads});
            mw.norm_weight = mAllocator->allocate(mConfig.model_dtype,
                                                  ("mamba_norm_w_l" + std::to_string(layer)).c_str(),
                                                  EAllocationType::ON_DEVICE, {mamba_dim});

            load_tensor(prefix + ".mixer.in_proj.weight", "in_proj_weight", mw.in_proj_weight, true, layer);
            if (mw.in_proj_bias.has_value()) {
                load_tensor(prefix + ".mixer.in_proj.bias", "in_proj_bias", mw.in_proj_bias.value(), false, layer);
            }
            load_tensor(prefix + ".mixer.out_proj.weight", "out_proj_weight", mw.out_proj_weight, true, layer);
            if (mw.out_proj_bias.has_value()) {
                load_tensor(prefix + ".mixer.out_proj.bias", "out_proj_bias", mw.out_proj_bias.value(), false, layer);
            }
            load_tensor(prefix + ".mixer.conv1d.weight", "conv1d_weight", mw.conv1d_weight, true, layer);
            if (mw.conv1d_bias.has_value()) {
                load_tensor(prefix + ".mixer.conv1d.bias", "conv1d_bias", mw.conv1d_bias.value(), false, layer);
            }
            load_tensor(prefix + ".mixer.A_log", "A_log", mw.A_log, true, layer);
            load_tensor(prefix + ".mixer.D", "D", mw.D, true, layer);
            load_tensor(prefix + ".mixer.dt_bias", "dt_bias", mw.dt_bias, true, layer);
            load_tensor(prefix + ".mixer.norm.weight", "norm_weight", mw.norm_weight, true, layer);
        }
    }
}

template<typename Block>
const HfMappingSpec* FP8WeightProvider<Block>::resolve_hf_spec(const std::string& internal_name,
                                                               int& layer_idx,
                                                               std::string& resolved_name) const {
    if (!mConfig.hf_mapping) {
        return nullptr;
    }
    const HfMappingSpec* spec = mConfig.hf_mapping->find(internal_name, layer_idx);
    if (!spec) {
        return nullptr;
    }
    resolved_name = internal_name;
    if (spec->kind == HfMappingSpec::Kind::TiedTo && !spec->target.empty()) {
        int tied_layer = -1;
        const HfMappingSpec* tied = mConfig.hf_mapping->find(spec->target, tied_layer);
        if (tied) {
            resolved_name = spec->target;
            if (tied_layer >= 0) {
                layer_idx = tied_layer;
            }
            return tied;
        }
    }
    return spec;
}

template<typename Block>
bool FP8WeightProvider<Block>::load_tensor_from_spec(const SafeTensorsReader& reader,
                                                     const HfMappingSpec& spec,
                                                     std::string_view internal_name,
                                                     int layer_idx,
                                                     int expert_idx,
                                                     Tensor& target,
                                                     cudaStream_t stream,
                                                     std::string_view warn_tag) const {
    auto warn_missing = [&](const std::string& name) {
        if (!spec.optional) {
            std::cerr << "[" << warn_tag << " WARN] weight not found: " << name << "\n";
        }
    };

    switch (spec.kind) {
        case HfMappingSpec::Kind::Direct: {
            const std::string hf_name = HfMapping::format_name(
                spec.source.empty() ? std::string(internal_name) : spec.source, layer_idx, expert_idx);
            const SafeTensorEntry* entry = nullptr;
            for (const auto& e : reader.entries()) {
                if (e.name() == hf_name) {
                    entry = &e;
                    break;
                }
            }
            if (entry) {
                entry->read_tensor(target, /*allow_cast=*/true);
                return true;
            }
            warn_missing(hf_name);
            return false;
        }
        case HfMappingSpec::Kind::Fuse: {
            if (spec.dim != 0 || spec.sources.empty()) {
                return false;
            }
            const std::size_t elem_size = get_dtype_size(target.DType);
            long offset = 0;
            bool any_loaded = false;
            for (const auto& src : spec.sources) {
                const std::string hf_name = HfMapping::format_name(src, layer_idx, expert_idx);
                const SafeTensorEntry* entry = nullptr;
                for (const auto& e : reader.entries()) {
                    if (e.name() == hf_name) {
                        entry = &e;
                        break;
                    }
                }
                if (entry) {
                    if (!entry->shape().empty()) {
                        const long rows = entry->shape().at(0);
                        Tensor slice = target;
                        slice.Sizes[0] = rows;
                        slice.Data = target.Data + static_cast<std::size_t>(offset) *
                                     static_cast<std::size_t>(target.Sizes[1]) * elem_size;
                        entry->read_tensor(slice, /*allow_cast=*/true);
                        offset += rows;
                        any_loaded = true;
                    }
                } else {
                    warn_missing(hf_name);
                }
            }
            if (any_loaded && offset != target.Sizes[0]) {
                std::cerr << "[" << warn_tag << " WARN] fuse size mismatch for "
                          << internal_name << " (loaded " << offset
                          << " rows, expected " << target.Sizes[0] << ")\n";
            }
            return any_loaded;
        }
        case HfMappingSpec::Kind::Split: {
            if (spec.dim != 0 || spec.ranges.empty()) {
                return false;
            }
            const auto [start, end] = spec.ranges.front();
            if (start < 0 || end <= start) {
                return false;
            }
            const std::string hf_name = HfMapping::format_name(spec.source, layer_idx, expert_idx);
            const SafeTensorEntry* entry = nullptr;
            for (const auto& e : reader.entries()) {
                if (e.name() == hf_name) {
                    entry = &e;
                    break;
                }
            }
            if (entry) {
                long stride = 1;
                for (std::size_t i = 1; i < entry->shape().size(); ++i) {
                    stride *= entry->shape()[i];
                }
                const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(start) * stride;
                entry->read_raw(target, offset, target.nelem(), /*allow_cast=*/true);
                return true;
            }
            warn_missing(hf_name);
            return false;
        }
        case HfMappingSpec::Kind::Transform: {
            if (spec.fn != "transpose" || !mAllocator) {
                return false;
            }
            const std::string hf_name = HfMapping::format_name(spec.source, layer_idx, expert_idx);
            const SafeTensorEntry* entry = nullptr;
            for (const auto& e : reader.entries()) {
                if (e.name() == hf_name) {
                    entry = &e;
                    break;
                }
            }
            if (entry) {
                if (entry->shape().size() != 2 || target.Rank != 2) {
                    return false;
                }
                Tensor tmp = mAllocator->allocate(target.DType, ("hf_tmp_" + std::string(internal_name)).c_str(),
                                                 EAllocationType::ON_DEVICE,
                                                 {entry->shape().at(0), entry->shape().at(1)});
                entry->read_tensor(tmp, /*allow_cast=*/true);
                transpose(target, tmp, static_cast<int>(entry->shape().at(0)),
                          static_cast<int>(entry->shape().at(1)), stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                return true;
            }
            warn_missing(hf_name);
            return false;
        }
        default:
            return false;
    }
}

template<typename Block>
typename FP8WeightProvider<Block>::BlockWeights& FP8WeightProvider<Block>::get_block(int layer_idx, cudaStream_t stream) {
    // For MoE models, use the MoE-specific path that only handles attention weights
    // MoE models don't have dense MLP weights - they have per-expert weights instead
    if (is_moe()) {
        get_moe_attention_weights(layer_idx, stream);
        if constexpr (has_mamba_weights<BlockWeights>::value) {
            if (mConfig.has_mamba &&
                layer_idx >= 0 &&
                layer_idx < static_cast<int>(mMambaWeights.weights.size()) &&
                mMambaWeights.weights[static_cast<std::size_t>(layer_idx)].has_value()) {
                mDequantBlock.mamba = mMambaWeights.weights[static_cast<std::size_t>(layer_idx)];
            } else {
                mDequantBlock.mamba.reset();
            }
        }
        return mDequantBlock;
    }

    // Dense model path
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
            // **BF16 PATH**: Dequantize to BF16
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

        if constexpr (has_mamba_weights<BlockWeights>::value) {
            if (mConfig.has_mamba &&
                layer_idx >= 0 &&
                layer_idx < static_cast<int>(mMambaWeights.weights.size()) &&
                mMambaWeights.weights[static_cast<std::size_t>(layer_idx)].has_value()) {
                mFP8Block.mamba = mMambaWeights.weights[static_cast<std::size_t>(layer_idx)];
            } else {
                mFP8Block.mamba.reset();
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

        if constexpr (has_mamba_weights<BlockWeights>::value) {
            if (mConfig.has_mamba &&
                layer_idx >= 0 &&
                layer_idx < static_cast<int>(mMambaWeights.weights.size()) &&
                mMambaWeights.weights[static_cast<std::size_t>(layer_idx)].has_value()) {
                mDequantBlock.mamba = mMambaWeights.weights[static_cast<std::size_t>(layer_idx)];
            } else {
                mDequantBlock.mamba.reset();
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

// ============================================================================
// MoE Support Implementation
// ============================================================================

template<typename Block>
void FP8WeightProvider<Block>::allocate_moe_expert_buffers() {
    auto ctx = mAllocator->with_context("FP8_MoE_DequantBuf");

    const int hidden = mConfig.hidden_size;
    const int moe_inter = mConfig.qlora_config.moe_intermediate_size > 0 ?
                          mConfig.qlora_config.moe_intermediate_size : mConfig.intermediate_size;
    const int moe_M = mConfig.mlp_up_factor * moe_inter;
    const int num_experts = mConfig.qlora_config.num_experts;
    const int shared_inter = (mConfig.qlora_config.moe_shared_expert_intermediate_size > 0)
        ? mConfig.qlora_config.moe_shared_expert_intermediate_size
        : moe_inter;
    const int shared_M = mConfig.mlp_up_factor * shared_inter;

    mNumMoEExperts = num_experts;
    mHasSharedExpert = mConfig.qlora_config.num_shared_experts > 0;
    mSharedExpertD = shared_inter;

    // Allocate batched expert buffers (all experts for forward pass)
    // gate_up_proj: (num_experts, mlp_up_factor * moe_intermediate, hidden_size)
    // down_proj: (num_experts, hidden_size, moe_intermediate)
    mBatchedExpertGateUp = mAllocator->allocate(ETensorDType::BF16,
        "batched_expert_gate_up",
        EAllocationType::ON_DEVICE,
        {(long)num_experts, (long)moe_M, (long)hidden});

    mBatchedExpertDown = mAllocator->allocate(ETensorDType::BF16,
        "batched_expert_down",
        EAllocationType::ON_DEVICE,
        {(long)num_experts, (long)hidden, (long)moe_inter});

    if (mHasSharedExpert) {
        mSharedExpertGateUp = mAllocator->allocate(ETensorDType::BF16,
            "shared_expert_gate_up",
            EAllocationType::ON_DEVICE,
            {(long)shared_M, (long)hidden});

        mSharedExpertDown = mAllocator->allocate(ETensorDType::BF16,
            "shared_expert_down",
            EAllocationType::ON_DEVICE,
            {(long)hidden, (long)shared_inter});
    }
}

template<typename Block>
Tensor& FP8WeightProvider<Block>::get_router_gate(int layer_idx, cudaStream_t stream) {
    (void)stream;
    return mFP8Weights->get_moe_block(layer_idx).router_gate;
}

template<typename Block>
void FP8WeightProvider<Block>::get_moe_attention_weights(int layer_idx, cudaStream_t stream) {
    const auto& qblock = mFP8Weights->get_moe_block(layer_idx);
    const int block_size = mConfig.qlora_config.block_size();

    // Check cache for this layer's weights
    const bool cache_hit = (mCurrentLayer == layer_idx) && (mBufferVersion == mStepVersion);

    if (!cache_hit) {
        // Dequantize attention weights
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

        // Dequantize ALL expert weights into batched buffers
        // Each expert's weights are dequantized into a slice of the batched tensor
        const int hidden = mConfig.hidden_size;
        const int moe_inter = mConfig.qlora_config.moe_intermediate_size > 0 ?
                              mConfig.qlora_config.moe_intermediate_size : mConfig.intermediate_size;
        const int moe_M = mConfig.mlp_up_factor * moe_inter;

        for (int e = 0; e < mNumMoEExperts; ++e) {
            const auto& expert_weights = qblock.experts[e];

            // Create slice views into the batched buffers for this expert
            // gate_up_proj slice: offset by e * (moe_M * hidden) bytes
            Tensor gate_up_slice = Tensor::from_pointer(
                static_cast<std::byte*>(mBatchedExpertGateUp.Data) +
                    static_cast<size_t>(e) * moe_M * hidden * sizeof(nv_bfloat16),
                mBatchedExpertGateUp.Device,
                ETensorDType::BF16,
                std::array<long, 2>{moe_M, hidden}
            );

            // down_proj slice: offset by e * (hidden * moe_inter) bytes
            Tensor down_slice = Tensor::from_pointer(
                static_cast<std::byte*>(mBatchedExpertDown.Data) +
                    static_cast<size_t>(e) * hidden * moe_inter * sizeof(nv_bfloat16),
                mBatchedExpertDown.Device,
                ETensorDType::BF16,
                std::array<long, 2>{hidden, moe_inter}
            );

            // Dequantize this expert's weights into the slice
            dequantize_per_block(
                gate_up_slice.get<nv_bfloat16>(),
                expert_weights.gate_up_proj.data.get<__nv_fp8_e4m3>(),
                expert_weights.gate_up_proj.block_scales.get<float>(),
                expert_weights.gate_up_proj.M, expert_weights.gate_up_proj.K,
                block_size, mDeviceProps, stream);

            dequantize_per_block(
                down_slice.get<nv_bfloat16>(),
                expert_weights.down_proj.data.get<__nv_fp8_e4m3>(),
                expert_weights.down_proj.block_scales.get<float>(),
                expert_weights.down_proj.M, expert_weights.down_proj.K,
                block_size, mDeviceProps, stream);
        }

        // Dequantize shared expert weights (optional)
        if (mHasSharedExpert && qblock.shared_expert.has_value()) {
            const auto& shared = qblock.shared_expert.value();
            dequantize_per_block(
                mSharedExpertGateUp.get<nv_bfloat16>(),
                shared.gate_up_proj.data.get<__nv_fp8_e4m3>(),
                shared.gate_up_proj.block_scales.get<float>(),
                shared.gate_up_proj.M, shared.gate_up_proj.K,
                block_size, mDeviceProps, stream);
            dequantize_per_block(
                mSharedExpertDown.get<nv_bfloat16>(),
                shared.down_proj.data.get<__nv_fp8_e4m3>(),
                shared.down_proj.block_scales.get<float>(),
                shared.down_proj.M, shared.down_proj.K,
                block_size, mDeviceProps, stream);
        }

        mCurrentLayer = layer_idx;
        mBufferVersion = mStepVersion;
    }

    // Update layer norm pointers
    mDequantBlock.ln1.weight = qblock.ln1_weight;
    mDequantBlock.ln2.weight = qblock.ln2_weight;

    // Update router gate pointer
    if constexpr (has_moe_weights<BlockWeights>::value) {
        mDequantBlock.router.gate = qblock.router_gate;
    }

    // Copy QK-norm weights if present
    if constexpr (requires { mDequantBlock.attention.q_norm_weight; mDequantBlock.attention.k_norm_weight; }) {
        if (qblock.q_norm_weight.has_value() && qblock.k_norm_weight.has_value()) {
            mDequantBlock.attention.q_norm_weight = qblock.q_norm_weight;
            mDequantBlock.attention.k_norm_weight = qblock.k_norm_weight;
        }
    }

    if constexpr (has_mamba_weights<BlockWeights>::value) {
        mDequantBlock.mamba.reset();
    }
}

// ============================================================================
// Selective Expert Dequantization Implementation
// ============================================================================

template<typename Block>
void FP8WeightProvider<Block>::dequantize_selected_experts(
    int layer_idx, const SelectiveExpertInfo& selection_info, cudaStream_t stream)
{
    if (!selection_info.enabled || selection_info.num_active == 0) {
        return;
    }

    const auto& qblock = mFP8Weights->get_moe_block(layer_idx);
    const int hidden = mConfig.hidden_size;
    const int moe_inter = mConfig.qlora_config.moe_intermediate_size > 0 ?
                          mConfig.qlora_config.moe_intermediate_size : mConfig.intermediate_size;
    const int moe_M = mConfig.mlp_up_factor * moe_inter;

    // Check if we can reuse the current buffers
    const bool same_layer_step = (mCurrentLayer == layer_idx) && (mBufferVersion == mStepVersion);

    // IMPORTANT: Also check that we're on the same layer - don't reuse layer 0's experts for layer 1!
    bool selection_matches = same_layer_step &&
                             mCurrentExpertLayer == layer_idx &&
                             mCurrentSelection.enabled &&
                             mCurrentSelection.num_active == selection_info.num_active;
    if (selection_matches) {
        // Quick check: compare active expert lists
        for (int i = 0; i < selection_info.num_active && selection_matches; ++i) {
            if (mCurrentSelection.active_experts[i] != selection_info.active_experts[i]) {
                selection_matches = false;
            }
        }
    }

    if (selection_matches) {
        // Cache hit - buffers already contain the right experts
        return;
    }

    // Dequantize selected experts into COMPACT index positions in the buffer
    const int num_to_dequant = selection_info.num_active;
    const int block_size = mConfig.qlora_config.fp8_block_size > 0 ?
                           mConfig.qlora_config.fp8_block_size : 32;

    for (int i = 0; i < num_to_dequant; ++i) {
        const int global_expert_idx = selection_info.active_experts[i];
        const auto& expert_weights = qblock.experts[global_expert_idx];

        // Create slice views at the COMPACT index position (i, not global_expert_idx)
        Tensor gate_up_slice = Tensor::from_pointer(
            static_cast<std::byte*>(mBatchedExpertGateUp.Data) +
                static_cast<size_t>(i) * moe_M * hidden * sizeof(nv_bfloat16),
            mBatchedExpertGateUp.Device,
            ETensorDType::BF16,
            std::array<long, 2>{moe_M, hidden}
        );

        Tensor down_slice = Tensor::from_pointer(
            static_cast<std::byte*>(mBatchedExpertDown.Data) +
                static_cast<size_t>(i) * hidden * moe_inter * sizeof(nv_bfloat16),
            mBatchedExpertDown.Device,
            ETensorDType::BF16,
            std::array<long, 2>{hidden, moe_inter}
        );

        // Dequantize this expert's weights into the buffer slice
        dequantize_per_block(
            gate_up_slice.get<nv_bfloat16>(),
            expert_weights.gate_up_proj.data.get<__nv_fp8_e4m3>(),
            expert_weights.gate_up_proj.block_scales.get<float>(),
            expert_weights.gate_up_proj.M, expert_weights.gate_up_proj.K,
            block_size, mDeviceProps, stream);

        dequantize_per_block(
            down_slice.get<nv_bfloat16>(),
            expert_weights.down_proj.data.get<__nv_fp8_e4m3>(),
            expert_weights.down_proj.block_scales.get<float>(),
            expert_weights.down_proj.M, expert_weights.down_proj.K,
            block_size, mDeviceProps, stream);
    }

    // Update cache state
    mCurrentSelection = selection_info;
    mCurrentExpertLayer = layer_idx;  // Track which layer these experts are from
    mNumActiveExperts = num_to_dequant;

    // Update the expert weights tensor shapes (compact indexing)
    if constexpr (has_moe_weights<BlockWeights>::value) {
        mDequantBlock.experts.gate_up_proj = Tensor::from_pointer(
            mBatchedExpertGateUp.Data,
            mBatchedExpertGateUp.Device,
            ETensorDType::BF16,
            std::array<long, 3>{(long)num_to_dequant, (long)moe_M, (long)hidden}
        );
        mDequantBlock.experts.down_proj = Tensor::from_pointer(
            mBatchedExpertDown.Data,
            mBatchedExpertDown.Device,
            ETensorDType::BF16,
            std::array<long, 3>{(long)num_to_dequant, (long)hidden, (long)moe_inter}
        );
        mDequantBlock.experts.use_batched = true;
        mDequantBlock.experts.num_active_experts = num_to_dequant;
    }
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_QLORA_QLORA_WEIGHT_PROVIDER_H
