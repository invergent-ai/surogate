// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_QLORA_BNB_WEIGHT_PROVIDER_H
#define SUROGATE_SRC_MODULES_QLORA_BNB_WEIGHT_PROVIDER_H

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <vector>
#include <iostream>

#include <cuda_runtime_api.h>

#include <fmt/core.h>

#include "qlora_config.h"
#include "bnb_weights.h"
#include "bnb_block_quantized_tensor.h"
#include "moe_weights.h"
#include "hf_mapping.h"
#include "kernels/kernels.h"
#include "dsl_block_weights.h"
#include "runtime/lora/lora_config.h"
#include "runtime/moe/moe_types.h"
#include "weight_provider_types.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"
#include "utilities/safetensors.h"
#include "utilities/utils.h"

namespace modules {

/**
 * @brief Helper to determine hybrid layer type from pattern string
 */
enum class HybridLayerType { Mamba, MoE, Attention, MLP, Dense, Unknown };

inline HybridLayerType get_hybrid_layer_type(const std::string& pattern, int layer_idx) {
    if (pattern.empty() || layer_idx < 0 || layer_idx >= static_cast<int>(pattern.size())) {
        return HybridLayerType::Unknown;
    }
    switch (pattern[static_cast<std::size_t>(layer_idx)]) {
        case 'M': return HybridLayerType::Mamba;
        case 'E': return HybridLayerType::MoE;
        case '*': return HybridLayerType::Attention;
        case '-': return HybridLayerType::MLP;
        default: return HybridLayerType::Dense;
    }
}

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
        int mlp_up_factor = 2;      ///< 2 for gated (SwiGLU), 1 for non-gated (ReLU2)
        QLoRAConfig qlora_config;
        ModularLoRAConfig lora_config;
        ETensorDType model_dtype = ETensorDType::BF16;
        bool use_qk_norm = false;      ///< Whether model uses QK-norm (Qwen3)
        bool tied_embeddings = true;   ///< Whether lm_head is tied to embeddings
        int shard_idx = 0;
        int num_shards = 1;

        /// Enable selective expert dequantization for MoE models.
        /// When enabled, only the experts selected by the router are dequantized,
        /// reducing memory usage from O(num_experts) to O(top_k) for dequant buffers.
        bool selective_expert_dequant = true;

        /// Force full expert dequantization even when offload_experts is enabled.
        /// Useful for DSL paths that don't provide selection info for compact indexing.
        bool force_full_expert_dequant = false;

        /// Offload MoE expert NF4 weights to CPU pinned memory.
        /// When enabled, expert weights are stored in CPU memory and streamed to GPU
        /// on-demand when selected by the router. Saves ~12GB for 128-expert models.
        /// Implies selective_expert_dequant = true.
        bool offload_experts = false;

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
        const HfMapping* hf_mapping = nullptr;

        /// Hybrid architecture pattern (e.g., "MEMEM*EMEMEM*...") where:
        /// M = Mamba, E = MoE, * = Attention, - = MLP
        /// If empty, assumes uniform architecture (all layers same type).
        std::string hybrid_pattern;
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

    /**
     * @brief Check if this is an MoE model
     */
    [[nodiscard]] bool is_moe() const { return mBnBWeights->is_moe(); }

    /**
     * @brief Get number of experts (0 for dense models)
     */
    [[nodiscard]] int num_experts() const { return mBnBWeights->num_experts(); }

    // =========================================================================
    // MoE-specific methods
    // =========================================================================

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
                                     cudaStream_t stream, bool force = false);

    /**
     * @brief Check if selective expert dequantization is enabled
     */
    [[nodiscard]] bool use_selective_dequant() const {
        return mConfig.selective_expert_dequant && mConfig.qlora_config.is_moe();
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
     * @brief Get the maximum number of active experts for buffer sizing
     *
     * For selective dequant, we need to allocate enough buffer space for the
     * worst case. With large batch sizes (e.g., 32K tokens), almost all experts
     * may be selected. So we always allocate for num_experts to be safe.
     * The memory savings from selective dequant come from:
     * 1. Option A (selective_expert_dequant only): Reduced dequant buffer (~1GB savings)
     * 2. Option B (offload_experts): Expert NF4 weights on CPU (~12GB savings)
     */
    [[nodiscard]] int max_active_experts() const {
        // Always allocate buffers for all experts - with large batches,
        // nearly all experts will be selected anyway
        return mNumMoEExperts;
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

    template<bool HasMamba, typename Dummy = void>
    struct MambaWeightsStorage {};
    template<typename Dummy>
    struct MambaWeightsStorage<true, Dummy> {
        std::vector<std::optional<typename BlockWeights::MambaWeights>> weights;
    };
    MambaWeightsStorage<has_mamba_weights<BlockWeights>::value> mMambaWeights;

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

    /// Batched expert dequantization buffers
    /// When selective_expert_dequant is enabled:
    ///   Shape: (max_active_experts, mlp_up_factor * moe_intermediate, hidden_size)
    /// When disabled:
    ///   Shape: (num_experts, mlp_up_factor * moe_intermediate, hidden_size)
    Tensor mBatchedExpertGateUp;
    /// Shape: (max_active_experts or num_experts, hidden_size, moe_intermediate)
    Tensor mBatchedExpertDown;

    /// Shared expert dequantization buffers (optional)
    Tensor mSharedExpertGateUp;
    Tensor mSharedExpertDown;
    int mSharedExpertD = 0;
    bool mHasSharedExpert = false;

    /// Number of experts in MoE model
    int mNumMoEExperts = 0;

    /// Current selection info for selective dequantization caching
    /// Tracks which experts are currently in the compact buffers
    SelectiveExpertInfo mCurrentSelection;

    /// Layer index for which the current expert selection is valid
    /// Used to avoid reusing cached experts from a different layer
    int mCurrentExpertLayer = -1;

    /// Number of experts currently in the compact buffers
    int mNumActiveExperts = 0;

    // =========================================================================
    // Expert offloading support
    // =========================================================================

    /// GPU staging buffer for streaming NF4 expert data from CPU
    /// When offload_experts is enabled, this holds the NF4 packed data for one expert
    /// Size: max(expert_gate_up_bytes, expert_down_bytes)
    Tensor mExpertNF4Staging;

    /// GPU staging buffer for absmax scales (for offloaded experts)
    Tensor mExpertAbsmaxStaging;

    /// GPU staging buffer for double-quant scale (for offloaded experts)
    Tensor mExpertAbsmaxScaleStaging;

    /// GPU staging buffer for double-quant offset (for offloaded experts)
    Tensor mExpertAbsmaxOffsetStaging;

    /// Whether expert weights are offloaded to CPU
    bool mExpertsOffloaded = false;

    void allocate_dequant_buffers();
    void allocate_moe_expert_buffers();
    void allocate_offload_staging_buffers();
    void setup_block_weights_structure();
    void dequantize_weight(const BnBBlockQuantizedWeight& src, Tensor& dst, cudaStream_t stream);
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

    /// Stream NF4 expert weight from CPU to GPU staging buffer, then dequantize
    void stream_and_dequantize_expert(const BnBBlockQuantizedWeight& src, Tensor& dst,
                                      cudaStream_t stream);
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
    , mExpertsOffloaded(config.offload_experts && config.qlora_config.is_moe())
{
    // If offload_experts is enabled, force selective_expert_dequant to true
    if (mExpertsOffloaded && !mConfig.selective_expert_dequant && !mConfig.force_full_expert_dequant) {
        mConfig.selective_expert_dequant = true;
    }

    // Create BnB weights manager
    BnBWeightsManager::Config bw_config{
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
        .hf_mapping = config.hf_mapping,
        .offload_experts = config.offload_experts,
        .hybrid_pattern = config.hybrid_pattern
    };
    mBnBWeights = std::make_unique<BnBWeightsManager>(bw_config, allocator, device_props);

    if constexpr (has_mamba_weights<BlockWeights>::value) {
        if (mConfig.has_mamba) {
            mMambaWeights.weights.resize(mConfig.num_layers);
        }
    }

    // Allocate dequantization buffers
    allocate_dequant_buffers();

    // Allocate MoE expert buffers if needed
    if (config.qlora_config.is_moe()) {
        allocate_moe_expert_buffers();

        // Allocate staging buffers for CPU -> GPU streaming if offloading is enabled
        if (mExpertsOffloaded) {
            allocate_offload_staging_buffers();
        }
    }

    // Set up the block weights structure with pointers to dequant buffers
    setup_block_weights_structure();
}

template<typename Block>
void BnBWeightProvider<Block>::allocate_dequant_buffers() {
    auto ctx = mAllocator->with_context("BnB_DequantBuf");

    const int hidden = mConfig.hidden_size;
    const int intermediate = mConfig.intermediate_size;
    const int mlp_M = mConfig.mlp_up_factor * intermediate;
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
                                           {(long)mlp_M, (long)hidden});

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
void BnBWeightProvider<Block>::import_and_quantize(
    const std::string& file_name, NCCLCommunicator& comm, cudaStream_t stream) {

    // Import and quantize base model weights
    mBnBWeights->import_and_quantize(file_name, comm, stream);
    load_mamba_weights(file_name);
}

template<typename Block>
void BnBWeightProvider<Block>::load_mamba_weights(const std::string& file_name) {
    if constexpr (!has_mamba_weights<BlockWeights>::value) {
        (void)file_name;
        return;
    } else {
        if (!mConfig.has_mamba) {
            return;
        }

        auto ctx = mAllocator->with_context("BnB_Mamba_Weights");
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
                                                   mapping_stream, "BnB");
                }
            }
            if (!loaded) {
                if (const auto* entry = find_entry_opt(name)) {
                    entry->read_tensor(tensor, /*allow_cast=*/true);
                } else if (required) {
                    std::cerr << "[BnB WARN] missing Mamba weight: " << name << "\n";
                }
            }
        };

        for (int layer = 0; layer < mConfig.num_layers; ++layer) {
            if (!mConfig.layer_is_mamba.empty() &&
                mConfig.layer_is_mamba[static_cast<std::size_t>(layer)] == 0) {
                continue;
            }

            const std::string model_prefix = fmt::format("model.layers.{}", layer);
            const std::string backbone_prefix = fmt::format("backbone.layers.{}", layer);
            const bool model_has = find_entry_opt(model_prefix + ".mixer.in_proj.weight") ||
                                   find_entry_opt(model_prefix + ".mixer.conv1d.weight") ||
                                   find_entry_opt(model_prefix + ".mixer.A_log");
            const std::string prefix = model_has ? model_prefix : backbone_prefix;

            auto& mw = mMambaWeights.weights[layer].emplace();

            mw.in_proj_weight = mAllocator->allocate(mConfig.model_dtype,
                                                     fmt::format("mamba_in_proj_w_l{}", layer).c_str(),
                                                     EAllocationType::ON_DEVICE, {proj_size, hidden});
            if (find_entry_opt(prefix + ".mixer.in_proj.bias")) {
                mw.in_proj_bias = mAllocator->allocate(mConfig.model_dtype,
                                                       fmt::format("mamba_in_proj_b_l{}", layer).c_str(),
                                                       EAllocationType::ON_DEVICE, {proj_size});
            }

            mw.out_proj_weight = mAllocator->allocate(mConfig.model_dtype,
                                                      fmt::format("mamba_out_proj_w_l{}", layer).c_str(),
                                                      EAllocationType::ON_DEVICE, {hidden, mamba_dim});
            if (find_entry_opt(prefix + ".mixer.out_proj.bias")) {
                mw.out_proj_bias = mAllocator->allocate(mConfig.model_dtype,
                                                        fmt::format("mamba_out_proj_b_l{}", layer).c_str(),
                                                        EAllocationType::ON_DEVICE, {hidden});
            }

            mw.conv1d_weight = mAllocator->allocate(mConfig.model_dtype,
                                                    fmt::format("mamba_conv_w_l{}", layer).c_str(),
                                                    EAllocationType::ON_DEVICE, {conv_dim, 1, mConfig.mamba_conv_kernel});
            if (find_entry_opt(prefix + ".mixer.conv1d.bias")) {
                mw.conv1d_bias = mAllocator->allocate(mConfig.model_dtype,
                                                      fmt::format("mamba_conv_b_l{}", layer).c_str(),
                                                      EAllocationType::ON_DEVICE, {conv_dim});
            }

            mw.A_log = mAllocator->allocate(ETensorDType::FP32,
                                            fmt::format("mamba_A_log_l{}", layer).c_str(),
                                            EAllocationType::ON_DEVICE, {mConfig.mamba_num_heads});
            mw.D = mAllocator->allocate(ETensorDType::FP32,
                                        fmt::format("mamba_D_l{}", layer).c_str(),
                                        EAllocationType::ON_DEVICE, {mConfig.mamba_num_heads});
            mw.dt_bias = mAllocator->allocate(ETensorDType::FP32,
                                              fmt::format("mamba_dt_bias_l{}", layer).c_str(),
                                              EAllocationType::ON_DEVICE, {mConfig.mamba_num_heads});
            mw.norm_weight = mAllocator->allocate(mConfig.model_dtype,
                                                  fmt::format("mamba_norm_w_l{}", layer).c_str(),
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
const HfMappingSpec* BnBWeightProvider<Block>::resolve_hf_spec(const std::string& internal_name,
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
bool BnBWeightProvider<Block>::load_tensor_from_spec(const SafeTensorsReader& reader,
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
void BnBWeightProvider<Block>::dequantize_weight(const BnBBlockQuantizedWeight& src,
                                                  Tensor& dst, cudaStream_t stream) {
    // Safety check: skip if source data is not allocated
    if (src.data.Data == nullptr || src.absmax.Data == nullptr) {
        return;
    }

    // Validate dtype: NF4 packed data should be BYTE (U8)
    if (src.data.DType != ETensorDType::BYTE) {
        std::cerr << "[BnB ERROR] dequantize_weight: expected BYTE dtype for NF4 data, got "
                  << dtype_to_str(src.data.DType) << " (M=" << src.M << ", K=" << src.K << ")\n";
        return;
    }

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

    // Check for hybrid architecture - Mamba layers need special handling
    const HybridLayerType layer_type = get_hybrid_layer_type(mConfig.hybrid_pattern, layer_idx);

    // For Mamba layers in hybrid architectures, skip attention/MoE dequantization
    // Mamba layers only have layer norms + Mamba-specific weights (loaded separately)
    if (layer_type == HybridLayerType::Mamba) {
        // Mamba layers are always loaded into dense (BnB) storage, not MoE storage
        // This is true even in hybrid models with MoE layers
        const auto& qblock = mBnBWeights->get_bnb_block(layer_idx);
        mDequantBlock.ln1.weight = qblock.ln1_weight;
        mDequantBlock.ln2.weight = qblock.ln2_weight;

        // Set up Mamba weights if available
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

        // Clear attention/MLP weights to avoid using stale data
        mDequantBlock.attention.qkv_weight = Tensor();
        mDequantBlock.attention.out_weight = Tensor();
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            mDequantBlock.mlp_up_weight = Tensor();
            mDequantBlock.mlp_down_weight = Tensor();
        }

        return mDequantBlock;
    }

    // For hybrid architectures, check if this specific layer uses MoE or dense MLP
    // MoE layers ('E') use mMoEBlocks storage, while Attention ('*') and MLP ('-')
    // layers use mQuantizedBlocks storage with dense MLP weights
    const bool is_moe_layer = (layer_type == HybridLayerType::MoE ||
                               (mConfig.hybrid_pattern.empty() && is_moe()));

    // For MoE layers, use the MoE-specific path
    // MoE layers have per-expert weights instead of dense MLP
    if (is_moe_layer) {
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

    // For Attention ('*') layers in hybrid architectures: use dense path for attention only
    // For MLP ('-') layers: use dense path for MLP only (attention already skipped above)
    // This handles layers that use mQuantizedBlocks storage with dense MLP instead of MoE experts

    // Dense model path (also used for Attention/MLP layers in hybrid architectures)
    const auto& qblock = mBnBWeights->get_bnb_block(layer_idx);

    // Check if this layer has attention (hybrid MLP layers don't have attention)
    // Check if this layer has MLP (hybrid Attention layers don't have MLP in Nemotron-H)
    const bool has_attention_weights = (layer_type != HybridLayerType::MLP);
    const bool has_mlp_weights = (layer_type != HybridLayerType::Attention);  // Attention has no MLP

    // Check if we already have this layer dequantized in the current step
    // This happens when backward accesses the same layer that forward just used
    const bool cache_hit = (mCurrentLayer == layer_idx) && (mBufferVersion == mStepVersion);

    if (!cache_hit) {
        // Cache miss: need to dequantize weights

        // QKV and output projection (only for layers with attention)
        if (has_attention_weights && qblock.qkv_proj.data.Data != nullptr) {
            dequantize_weight(qblock.qkv_proj, mDequantQKV, stream);
            dequantize_weight(qblock.out_proj, mDequantOut, stream);
        }

        // Gate+Up and down projection (for layers with MLP)
        if (has_mlp_weights && qblock.gate_up_proj.data.Data != nullptr) {
            dequantize_weight(qblock.gate_up_proj, mDequantGateUp, stream);
            dequantize_weight(qblock.down_proj, mDequantDown, stream);
        }

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

// ============================================================================
// MoE-specific Implementation
// ============================================================================

template<typename Block>
void BnBWeightProvider<Block>::allocate_moe_expert_buffers() {
    auto ctx = mAllocator->with_context("BnB_MoE_DequantBuf");

    const int hidden = mConfig.hidden_size;
    const int moe_inter = mConfig.qlora_config.moe_intermediate_size > 0 ?
                          mConfig.qlora_config.moe_intermediate_size : mConfig.intermediate_size;
    const int moe_M = mConfig.mlp_up_factor * moe_inter;
    const int shared_inter = (mConfig.qlora_config.moe_shared_expert_intermediate_size > 0)
        ? mConfig.qlora_config.moe_shared_expert_intermediate_size
        : moe_inter;
    const int shared_M = mConfig.mlp_up_factor * shared_inter;
    const int num_experts = mConfig.qlora_config.num_experts;

    mNumMoEExperts = num_experts;
    mHasSharedExpert = mConfig.qlora_config.num_shared_experts > 0;
    mSharedExpertD = shared_inter;

    // Determine buffer size based on selective dequant setting
    // When selective_expert_dequant is enabled, we only need buffers for
    // the maximum number of experts that could be active at once (typically top_k * factor)
    const int buffer_experts = max_active_experts();

    if (mConfig.selective_expert_dequant) {
        fmt::print("[BnB] Selective expert dequant enabled: allocating buffers for {} experts (vs {} total)\n",
                   buffer_experts, num_experts);
    }

    // Allocate batched expert buffers
    // gate_up_proj: (buffer_experts, mlp_up_factor * moe_intermediate, hidden_size)
    // down_proj: (buffer_experts, hidden_size, moe_intermediate)
    mBatchedExpertGateUp = mAllocator->allocate(ETensorDType::BF16,
        "batched_expert_gate_up",
        EAllocationType::ON_DEVICE,
        {(long)buffer_experts, (long)moe_M, (long)hidden});

    mBatchedExpertDown = mAllocator->allocate(ETensorDType::BF16,
        "batched_expert_down",
        EAllocationType::ON_DEVICE,
        {(long)buffer_experts, (long)hidden, (long)moe_inter});

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
void BnBWeightProvider<Block>::allocate_offload_staging_buffers() {
    auto ctx = mAllocator->with_context("BnB_OffloadStaging");

    const int hidden = mConfig.hidden_size;
    const int moe_inter = mConfig.qlora_config.moe_intermediate_size > 0 ?
                          mConfig.qlora_config.moe_intermediate_size : mConfig.intermediate_size;
    const int block_size = mConfig.qlora_config.block_size();
    const bool double_quant = mConfig.qlora_config.bnb_double_quant;
    const int dq_group_size = mConfig.qlora_config.bnb_double_quant_group_size > 0 ?
                              mConfig.qlora_config.bnb_double_quant_group_size : 256;

    const int moe_M = mConfig.mlp_up_factor * moe_inter;
    const int shared_inter = (mConfig.qlora_config.moe_shared_expert_intermediate_size > 0)
        ? mConfig.qlora_config.moe_shared_expert_intermediate_size
        : moe_inter;
    const int shared_M = mConfig.mlp_up_factor * shared_inter;

    // Calculate the maximum NF4 packed size for a single expert weight
    // gate_up_proj: (moe_M, hidden) -> (moe_M * hidden) / 2 bytes
    // down_proj: (hidden, moe_inter) -> (hidden * moe_inter) / 2 bytes
    const long gate_up_elems = static_cast<long>(moe_M) * hidden;
    const long down_elems = static_cast<long>(hidden) * moe_inter;
    long max_elems = std::max(gate_up_elems, down_elems);
    if (mConfig.qlora_config.num_shared_experts > 0) {
        const long shared_gate_elems = static_cast<long>(shared_M) * hidden;
        const long shared_down_elems = static_cast<long>(hidden) * shared_inter;
        max_elems = std::max(max_elems, std::max(shared_gate_elems, shared_down_elems));
    }
    const long max_packed_bytes = (max_elems + 1) / 2;

    // Allocate staging buffer for NF4 packed data
    mExpertNF4Staging = mAllocator->allocate(ETensorDType::BYTE, "expert_nf4_staging",
                                              EAllocationType::ON_DEVICE, {max_packed_bytes});

    // Calculate max absmax blocks
    const long max_absmax_blocks = (max_elems + block_size - 1) / block_size;

    if (double_quant) {
        // Double quantization: need staging for quantized absmax + scale/offset
        mExpertAbsmaxStaging = mAllocator->allocate(ETensorDType::BYTE, "expert_absmax_staging",
                                                     EAllocationType::ON_DEVICE, {max_absmax_blocks});
        const long max_groups = (max_absmax_blocks + dq_group_size - 1) / dq_group_size;
        mExpertAbsmaxScaleStaging = mAllocator->allocate(ETensorDType::FP32, "expert_absmax_scale_staging",
                                                          EAllocationType::ON_DEVICE, {max_groups});
        mExpertAbsmaxOffsetStaging = mAllocator->allocate(ETensorDType::FP32, "expert_absmax_offset_staging",
                                                           EAllocationType::ON_DEVICE, {max_groups});
    } else {
        // No double quant: FP32 absmax
        mExpertAbsmaxStaging = mAllocator->allocate(ETensorDType::FP32, "expert_absmax_staging",
                                                     EAllocationType::ON_DEVICE, {max_absmax_blocks});
    }
}

template<typename Block>
Tensor& BnBWeightProvider<Block>::get_router_gate(int layer_idx, cudaStream_t stream) {
    (void)stream;
    return mBnBWeights->get_moe_block(layer_idx).router_gate;
}

template<typename Block>
void BnBWeightProvider<Block>::get_moe_attention_weights(int layer_idx, cudaStream_t stream) {
    const auto& qblock = mBnBWeights->get_moe_block(layer_idx);

    // Check if this layer has attention weights (hybrid architectures may have
    // layers without attention, e.g., Mamba layers)
    const HybridLayerType layer_type = get_hybrid_layer_type(mConfig.hybrid_pattern, layer_idx);
    const bool has_attention = (layer_type != HybridLayerType::Mamba &&
                                layer_type != HybridLayerType::MLP);

    // Check cache for this layer's weights
    const bool cache_hit = (mCurrentLayer == layer_idx) && (mBufferVersion == mStepVersion);

    if (!cache_hit) {
        // Dequantize attention weights only if this layer has attention
        if (has_attention && qblock.qkv_proj.data.Data != nullptr) {
            dequantize_weight(qblock.qkv_proj, mDequantQKV, stream);
            dequantize_weight(qblock.out_proj, mDequantOut, stream);
        }

        // Check if this layer has MoE experts (vs dense MLP)
        // In hybrid architectures, some layers use dense MLP ('*' Attention, '-' MLP)
        // while others use MoE experts ('E' MoE layers)
        const bool has_moe_experts = (layer_type == HybridLayerType::MoE ||
                                      layer_type == HybridLayerType::Unknown ||
                                      mConfig.hybrid_pattern.empty());

        // When selective_expert_dequant is enabled AND LoRA is enabled, skip dequantizing
        // all experts here. The LoRA forward hook will call dequantize_selected_experts()
        // later with only the router-selected experts. This saves memory.
        //
        // IMPORTANT: When LoRA is disabled (lora: false), we MUST dequantize all experts here
        // because the LoRA hook won't run and dequantize_selected_experts() won't be called.
        // Without this check, the base model would use uninitialized expert weight buffers.
        const bool use_selective_path = mConfig.selective_expert_dequant && mConfig.lora_config.enabled();

        if (has_moe_experts && !use_selective_path) {
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
                // When experts are offloaded to CPU, we need to stream them to GPU first
                if (mExpertsOffloaded) {
                    stream_and_dequantize_expert(expert_weights.gate_up_proj, gate_up_slice, stream);
                    stream_and_dequantize_expert(expert_weights.down_proj, down_slice, stream);
                } else {
                    dequantize_weight(expert_weights.gate_up_proj, gate_up_slice, stream);
                    dequantize_weight(expert_weights.down_proj, down_slice, stream);
                }
            }
        }
        // When selective mode is on, expert dequantization is deferred to dequantize_selected_experts()

        // Dequantize shared expert weights (only for MoE layers that have experts)
        if (has_moe_experts && mHasSharedExpert && qblock.shared_expert.has_value()) {
            const auto& shared = qblock.shared_expert.value();
            if (mExpertsOffloaded) {
                stream_and_dequantize_expert(shared.gate_up_proj, mSharedExpertGateUp, stream);
                stream_and_dequantize_expert(shared.down_proj, mSharedExpertDown, stream);
            } else {
                dequantize_weight(shared.gate_up_proj, mSharedExpertGateUp, stream);
                dequantize_weight(shared.down_proj, mSharedExpertDown, stream);
            }
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

template<typename Block>
void BnBWeightProvider<Block>::dequantize_selected_experts(
    int layer_idx, const SelectiveExpertInfo& selection_info, cudaStream_t stream, bool force)
{
    if (!selection_info.enabled || selection_info.num_active == 0) {
        return;
    }

    // For hybrid architectures, only MoE layers have experts to dequantize
    const HybridLayerType layer_type = get_hybrid_layer_type(mConfig.hybrid_pattern, layer_idx);
    if (!mConfig.hybrid_pattern.empty() && layer_type != HybridLayerType::MoE) {
        // This layer doesn't have MoE experts (e.g., Mamba, Attention, or MLP layer)
        return;
    }

    const auto& qblock = mBnBWeights->get_moe_block(layer_idx);
    const int hidden = mConfig.hidden_size;
    const int moe_inter = mConfig.qlora_config.moe_intermediate_size > 0 ?
                          mConfig.qlora_config.moe_intermediate_size : mConfig.intermediate_size;
    const int moe_M = mConfig.mlp_up_factor * moe_inter;

    // Check if we can reuse the current buffers
    // Cache hit if: same layer, same step, and selection is a subset of current
    const bool same_layer_step = (mCurrentLayer == layer_idx) && (mBufferVersion == mStepVersion);

    // For selective dequant, we need to check if the current selection matches
    // Simple approach: always re-dequantize if selection changed
    // (More sophisticated: check if new selection is subset of current)
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

    if (!force && selection_matches) {
        // Cache hit - buffers already contain the right experts
        return;
    }

    const int num_to_dequant = selection_info.num_active;

    for (int i = 0; i < num_to_dequant; ++i) {
        const int global_expert_idx = selection_info.active_experts[i];
        if (global_expert_idx < 0 || global_expert_idx >= mNumMoEExperts) {
            continue;
        }
        const auto& expert_weights = qblock.experts[global_expert_idx];

        // Create slice views at the GLOBAL expert index to avoid compact-index mismatch.
        Tensor gate_up_slice = Tensor::from_pointer(
            static_cast<std::byte*>(mBatchedExpertGateUp.Data) +
                static_cast<size_t>(global_expert_idx) * moe_M * hidden * sizeof(nv_bfloat16),
            mBatchedExpertGateUp.Device,
            ETensorDType::BF16,
            std::array<long, 2>{moe_M, hidden}
        );

        Tensor down_slice = Tensor::from_pointer(
            static_cast<std::byte*>(mBatchedExpertDown.Data) +
                static_cast<size_t>(global_expert_idx) * hidden * moe_inter * sizeof(nv_bfloat16),
            mBatchedExpertDown.Device,
            ETensorDType::BF16,
            std::array<long, 2>{hidden, moe_inter}
        );
       
        // Dequantize this expert's weights into the buffer slice
        // When experts are offloaded to CPU, we need to stream them to GPU first
        if (mExpertsOffloaded) {
            stream_and_dequantize_expert(expert_weights.gate_up_proj, gate_up_slice, stream);
            stream_and_dequantize_expert(expert_weights.down_proj, down_slice, stream);
        } else {
            dequantize_weight(expert_weights.gate_up_proj, gate_up_slice, stream);
            dequantize_weight(expert_weights.down_proj, down_slice, stream);
        }
    }

    // Update cache state for selective expert dequant
    // NOTE: We intentionally do NOT update mCurrentLayer/mBufferVersion here!
    // Those are used for attention weight caching in get_moe_attention_weights().
    // If we update them here, get_block() will think attention weights are cached
    // when they're not. Instead, we only track expert selection state separately.
    mCurrentSelection = selection_info;
    mCurrentExpertLayer = layer_idx;  // Track which layer these experts are from
    mNumActiveExperts = num_to_dequant;

    // Update the expert weights tensor shapes (global indexing).
    if constexpr (has_moe_weights<BlockWeights>::value) {
        mDequantBlock.experts.gate_up_proj = Tensor::from_pointer(
            mBatchedExpertGateUp.Data,
            mBatchedExpertGateUp.Device,
            ETensorDType::BF16,
            std::array<long, 3>{(long)mNumMoEExperts, (long)moe_M, (long)hidden}
        );
        mDequantBlock.experts.down_proj = Tensor::from_pointer(
            mBatchedExpertDown.Data,
            mBatchedExpertDown.Device,
            ETensorDType::BF16,
            std::array<long, 3>{(long)mNumMoEExperts, (long)hidden, (long)moe_inter}
        );
        mDequantBlock.experts.use_batched = true;
        mDequantBlock.experts.num_active_experts = num_to_dequant;
    }
}

template<typename Block>
void BnBWeightProvider<Block>::stream_and_dequantize_expert(
    const BnBBlockQuantizedWeight& src, Tensor& dst, cudaStream_t stream)
{
    // This method handles CPU-offloaded experts:
    // 1. Copy NF4 packed data from pinned CPU memory to GPU staging buffer
    // 2. Copy absmax scales from CPU to GPU staging buffer
    // 3. Dequantize from GPU staging buffer to destination
    //
    // IMPORTANT: The staging buffer is reused for each expert weight tensor.
    // Since all operations are on the same stream, they are serialized:
    //   expert[i].gate_up: memcpy -> dequant
    //   expert[i].down: memcpy -> dequant
    //   expert[i+1].gate_up: memcpy -> dequant
    //   ...
    // No explicit synchronization is needed between operations.

    // Validate input parameters
    if (src.M <= 0 || src.K <= 0 || src.block_size <= 0 || src.data.Data == nullptr || src.absmax.Data == nullptr) {
        std::cerr << "[BnB ERROR] Invalid source weight in stream_and_dequantize_expert!\n";
        std::cerr << "  src.M=" << src.M << ", src.K=" << src.K << "\n";
        std::cerr << "  src.block_size=" << src.block_size << "\n";
        std::cerr << "  src.data.Data=" << (void*)src.data.Data << "\n";
        std::cerr << "  src.absmax.Data=" << (void*)src.absmax.Data << "\n";
        return;  // Skip this expert to avoid crash
    }
    if (src.double_quant && (src.absmax_scale.Data == nullptr || src.absmax_offset.Data == nullptr || src.double_quant_group_size <= 0)) {
        std::cerr << "[BnB ERROR] Invalid double_quant parameters in stream_and_dequantize_expert!\n";
        std::cerr << "  src.double_quant=" << src.double_quant << "\n";
        std::cerr << "  src.double_quant_group_size=" << src.double_quant_group_size << "\n";
        std::cerr << "  src.absmax_scale.Data=" << (void*)src.absmax_scale.Data << "\n";
        std::cerr << "  src.absmax_offset.Data=" << (void*)src.absmax_offset.Data << "\n";
        return;  // Skip this expert to avoid crash
    }

    // Step 1: Copy NF4 packed data (HOST -> DEVICE)
    const size_t data_bytes = src.packed_bytes();
    CUDA_CHECK(cudaMemcpyAsync(mExpertNF4Staging.Data, src.data.Data,
                               data_bytes, cudaMemcpyHostToDevice, stream));

    // Step 2: Copy absmax data
    const long num_blocks = src.num_blocks();
    if (src.double_quant) {
        // Copy quantized absmax (uint8)
        cudaMemcpyAsync(mExpertAbsmaxStaging.Data, src.absmax.Data,
                        num_blocks * sizeof(unsigned char), cudaMemcpyHostToDevice, stream);

        // Copy scale and offset
        const int num_groups = src.num_groups();
        cudaMemcpyAsync(mExpertAbsmaxScaleStaging.Data, src.absmax_scale.Data,
                        num_groups * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(mExpertAbsmaxOffsetStaging.Data, src.absmax_offset.Data,
                        num_groups * sizeof(float), cudaMemcpyHostToDevice, stream);

        // Step 3: Dequantize using double-dequant kernel from staging buffers
        dequantize_bnb_nf4_double(
            dst.get<nv_bfloat16>(),
            mExpertNF4Staging.get<unsigned char>(),
            mExpertAbsmaxStaging.get<unsigned char>(),
            mExpertAbsmaxScaleStaging.get<float>(),
            mExpertAbsmaxOffsetStaging.get<float>(),
            src.M, src.K,
            src.block_size, src.double_quant_group_size,
            mDeviceProps, stream);

    } else {
        // Copy FP32 absmax
        cudaMemcpyAsync(mExpertAbsmaxStaging.Data, src.absmax.Data,
                        num_blocks * sizeof(float), cudaMemcpyHostToDevice, stream);

        // Step 3: Dequantize using standard kernel from staging buffers
        dequantize_bnb_nf4(
            dst.get<nv_bfloat16>(),
            mExpertNF4Staging.get<unsigned char>(),
            mExpertAbsmaxStaging.get<float>(),
            src.M, src.K,
            src.block_size,
            mDeviceProps, stream);
    }
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_QLORA_BNB_WEIGHT_PROVIDER_H
