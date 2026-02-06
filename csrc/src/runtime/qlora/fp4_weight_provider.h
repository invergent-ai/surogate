// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// FP4 weight provider: on-the-fly dequantization of FP4 weights to BF16

#ifndef SUROGATE_SRC_MODULES_QLORA_FP4_WEIGHT_PROVIDER_H
#define SUROGATE_SRC_MODULES_QLORA_FP4_WEIGHT_PROVIDER_H

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

#include "fp4_weights.h"
#include "fp4_block_quantized_tensor.h"
#include "moe_weights.h"
#include "qlora_config.h"
#include "hf_mapping.h"
#include "kernels/kernels.h"
#include "dsl_block_weights.h"
#include "runtime/lora/lora_config.h"
#include "runtime/moe/moe_types.h"
#include "weight_provider_types.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"
#include "utilities/safetensors.h"
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
        int mlp_up_factor = 2;      ///< 2 for gated (SwiGLU), 1 for non-gated (ReLU2)
        QLoRAConfig qlora_config;
        ModularLoRAConfig lora_config;
        ETensorDType model_dtype = ETensorDType::BF16;
        bool use_qk_norm = false;
        bool tied_embeddings = true;
        int shard_idx = 0;
        int num_shards = 1;

        /// Enable selective expert dequantization for MoE models.
        /// When enabled, only the experts selected by the router are dequantized,
        /// reducing memory usage from O(num_experts) to O(top_k) for dequant buffers.
        bool selective_expert_dequant = true;

        /// Force full expert dequantization even when offload_experts is enabled.
        /// Useful for DSL paths that don't provide selection info for compact indexing.
        bool force_full_expert_dequant = false;

        /// Offload MoE expert FP4 weights to CPU pinned memory.
        /// When enabled, expert weights are stored in CPU memory and streamed to GPU
        /// on-demand when selected by the router. Saves ~10GB for 128-expert models.
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

    // =========================================================================
    // MoE Support
    // =========================================================================

    /**
     * @brief Check if this is an MoE model
     */
    [[nodiscard]] bool is_moe() const {
        return mFP4Weights && mFP4Weights->is_moe();
    }

    /**
     * @brief Get number of experts (0 for dense models)
     */
    [[nodiscard]] int num_experts() const {
        return mFP4Weights ? mFP4Weights->num_experts() : 0;
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
     */
    [[nodiscard]] int max_active_experts() const {
        return mNumMoEExperts;
    }

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

    template<bool HasMamba, typename Dummy = void>
    struct MambaWeightsStorage {};
    template<typename Dummy>
    struct MambaWeightsStorage<true, Dummy> {
        std::vector<std::optional<typename BlockWeights::MambaWeights>> weights;
    };
    MambaWeightsStorage<has_mamba_weights<BlockWeights>::value> mMambaWeights;

    // Zero-overhead forward/backward cache via step versioning
    int mCurrentLayer = -1;
    uint64_t mStepVersion = 0;
    uint64_t mBufferVersion = 0;

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

    // =========================================================================
    // Expert offloading support
    // =========================================================================

    /// GPU staging buffer for streaming FP4 expert data from CPU
    /// Size: max(expert_gate_up_bytes, expert_down_bytes)
    Tensor mExpertFP4Staging;

    /// GPU staging buffer for FP8 block scales (for offloaded experts)
    Tensor mExpertScalesStaging;

    /// Whether expert weights are offloaded to CPU
    bool mExpertsOffloaded = false;

    void allocate_dequant_buffers();
    void allocate_moe_expert_buffers();
    void allocate_offload_staging_buffers();
    void setup_block_weights_structure();
    void dequantize_fp4_weight(const FP4BlockQuantizedWeight& src, Tensor& dst, cudaStream_t stream);

    /// Stream FP4 expert weight from CPU to GPU staging buffer, then dequantize
    void stream_and_dequantize_expert(const FP4BlockQuantizedWeight& src, Tensor& dst,
                                      cudaStream_t stream);
};

// ============================================================================
// Implementation
// ============================================================================

template<typename Block>
FP4WeightProvider<Block>::FP4WeightProvider(
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

    // Create FP4 weights manager
    FP4WeightsManager::Config fp4_config{
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
        .offload_experts = config.offload_experts
    };
    mFP4Weights = std::make_unique<FP4WeightsManager>(fp4_config, allocator, device_props);

    if constexpr (has_mamba_weights<BlockWeights>::value) {
        if (mConfig.has_mamba) {
            mMambaWeights.weights.resize(mConfig.num_layers);
        }
    }

    // Allocate dequantization buffers
    allocate_dequant_buffers();

    // Allocate MoE expert buffers if needed
    if (mFP4Weights->is_moe()) {
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
void FP4WeightProvider<Block>::allocate_dequant_buffers() {
    auto ctx = mAllocator->with_context("FP4_DequantBuf");

    const int hidden = mConfig.hidden_size;
    const int intermediate = mConfig.intermediate_size;
    const int mlp_M = mConfig.mlp_up_factor * intermediate;
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
                                           {(long)mlp_M, (long)hidden});

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
void FP4WeightProvider<Block>::import_and_quantize(
    const std::string& file_name, NCCLCommunicator& comm, cudaStream_t stream) {

    // Import and quantize base model weights to FP4
    mFP4Weights->import_and_quantize(file_name, comm, stream);
    load_mamba_weights(file_name);

    // Load final norm weight from file (not quantized)
    SafeTensorsReader reader(file_name);
    bool final_norm_loaded = false;
    if (mConfig.hf_mapping) {
        int map_layer = -1;
        std::string resolved_name;
        if (const auto* spec = resolve_hf_spec("final_norm", map_layer, resolved_name)) {
            final_norm_loaded = load_tensor_from_spec(reader, *spec, resolved_name, map_layer,
                                                      -1, mFinalNormWeight, stream, "FP4");
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
void FP4WeightProvider<Block>::load_mamba_weights(const std::string& file_name) {
    if constexpr (!has_mamba_weights<BlockWeights>::value) {
        (void)file_name;
        return;
    } else {
        if (!mConfig.has_mamba) {
            return;
        }

        auto ctx = mAllocator->with_context("FP4_Mamba_Weights");
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
                                                   mapping_stream, "FP4");
                }
            }
            if (!loaded) {
                if (const auto* entry = find_entry_opt(name)) {
                    entry->read_tensor(tensor, /*allow_cast=*/true);
                } else if (required) {
                    std::cerr << "[FP4 WARN] missing Mamba weight: " << name << "\n";
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
const HfMappingSpec* FP4WeightProvider<Block>::resolve_hf_spec(const std::string& internal_name,
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
bool FP4WeightProvider<Block>::load_tensor_from_spec(const SafeTensorsReader& reader,
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
typename FP4WeightProvider<Block>::BlockWeights& FP4WeightProvider<Block>::get_block(
    int layer_idx, cudaStream_t stream) {

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
            mDeviceProps, stream);

        // Output projection
        float out_scale = fp4_block.out_proj.global_decode_scale_rowwise();
        dequantize_fp4_block(
            mDequantOut.get<nv_bfloat16>(),
            fp4_block.out_proj.data.get<uint8_t>(),
            fp4_block.out_proj.block_scales_rowwise.get<__nv_fp8_e4m3>(),
            out_scale,
            fp4_block.out_proj.M, fp4_block.out_proj.K,
            mDeviceProps, stream);

        // Gate+Up projection
        float gate_up_scale = fp4_block.gate_up_proj.global_decode_scale_rowwise();
        dequantize_fp4_block(
            mDequantGateUp.get<nv_bfloat16>(),
            fp4_block.gate_up_proj.data.get<uint8_t>(),
            fp4_block.gate_up_proj.block_scales_rowwise.get<__nv_fp8_e4m3>(),
            gate_up_scale,
            fp4_block.gate_up_proj.M, fp4_block.gate_up_proj.K,
            mDeviceProps, stream);

        // Down projection
        float down_scale = fp4_block.down_proj.global_decode_scale_rowwise();
        dequantize_fp4_block(
            mDequantDown.get<nv_bfloat16>(),
            fp4_block.down_proj.data.get<uint8_t>(),
            fp4_block.down_proj.block_scales_rowwise.get<__nv_fp8_e4m3>(),
            down_scale,
            fp4_block.down_proj.M, fp4_block.down_proj.K,
            mDeviceProps, stream);

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

template<typename Block>
Tensor& FP4WeightProvider<Block>::get_final_norm(cudaStream_t stream) {
    (void)stream;
    return mFinalNormWeight;
}

// ============================================================================
// MoE Support Implementation
// ============================================================================

template<typename Block>
void FP4WeightProvider<Block>::allocate_moe_expert_buffers() {
    auto ctx = mAllocator->with_context("FP4_MoE_DequantBuf");

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
Tensor& FP4WeightProvider<Block>::get_router_gate(int layer_idx, cudaStream_t stream) {
    (void)stream;
    return mFP4Weights->get_moe_block(layer_idx).router_gate;
}

template<typename Block>
void FP4WeightProvider<Block>::get_moe_attention_weights(int layer_idx, cudaStream_t stream) {
    const auto& qblock = mFP4Weights->get_moe_block(layer_idx);

    // Check cache for this layer's weights
    const bool cache_hit = (mCurrentLayer == layer_idx) && (mBufferVersion == mStepVersion);

    if (!cache_hit) {
        // Dequantize attention weights
        dequantize_fp4_weight(qblock.qkv_proj, mDequantQKV, stream);
        dequantize_fp4_weight(qblock.out_proj, mDequantOut, stream);

        // When selective_expert_dequant is enabled AND LoRA is enabled, skip dequantizing
        // all experts here. The LoRA forward hook will call dequantize_selected_experts()
        // later with only the router-selected experts. This saves memory.
        //
        // IMPORTANT: When LoRA is disabled (lora: false), we MUST dequantize all experts here
        // because the LoRA hook won't run and dequantize_selected_experts() won't be called.
        const bool use_selective_path = mConfig.selective_expert_dequant && mConfig.lora_config.enabled();

        if (!use_selective_path) {
            // Dequantize ALL expert weights into batched buffers
            const int hidden = mConfig.hidden_size;
            const int moe_inter = mConfig.qlora_config.moe_intermediate_size > 0 ?
                                  mConfig.qlora_config.moe_intermediate_size : mConfig.intermediate_size;
            const int moe_M = mConfig.mlp_up_factor * moe_inter;

            for (int e = 0; e < mNumMoEExperts; ++e) {
                const auto& expert_weights = qblock.experts[e];

                // Create slice views into the batched buffers for this expert
                Tensor gate_up_slice = Tensor::from_pointer(
                    static_cast<std::byte*>(mBatchedExpertGateUp.Data) +
                        static_cast<size_t>(e) * moe_M * hidden * sizeof(nv_bfloat16),
                    mBatchedExpertGateUp.Device,
                    ETensorDType::BF16,
                    std::array<long, 2>{moe_M, hidden}
                );

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
                    dequantize_fp4_weight(expert_weights.gate_up_proj, gate_up_slice, stream);
                    dequantize_fp4_weight(expert_weights.down_proj, down_slice, stream);
                }
            }
        }
        // When selective mode is on, expert dequantization is deferred to dequantize_selected_experts()

        // Dequantize shared expert weights (optional)
        if (mHasSharedExpert && qblock.shared_expert.has_value()) {
            const auto& shared = qblock.shared_expert.value();
            if (mExpertsOffloaded) {
                stream_and_dequantize_expert(shared.gate_up_proj, mSharedExpertGateUp, stream);
                stream_and_dequantize_expert(shared.down_proj, mSharedExpertDown, stream);
            } else {
                dequantize_fp4_weight(shared.gate_up_proj, mSharedExpertGateUp, stream);
                dequantize_fp4_weight(shared.down_proj, mSharedExpertDown, stream);
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

// ============================================================================
// FP4-specific helper functions
// ============================================================================

template<typename Block>
void FP4WeightProvider<Block>::dequantize_fp4_weight(const FP4BlockQuantizedWeight& src,
                                                      Tensor& dst, cudaStream_t stream) {
    float global_scale = src.global_decode_scale_rowwise();
    dequantize_fp4_block(
        dst.get<nv_bfloat16>(),
        src.data.get<uint8_t>(),
        src.block_scales_rowwise.get<__nv_fp8_e4m3>(),
        global_scale,
        src.M, src.K,
        mDeviceProps, stream);
}

template<typename Block>
void FP4WeightProvider<Block>::allocate_offload_staging_buffers() {
    auto ctx = mAllocator->with_context("FP4_OffloadStaging");

    const int hidden = mConfig.hidden_size;
    const int moe_inter = mConfig.qlora_config.moe_intermediate_size > 0 ?
                          mConfig.qlora_config.moe_intermediate_size : mConfig.intermediate_size;
    const int moe_M = mConfig.mlp_up_factor * moe_inter;
    const int shared_inter = (mConfig.qlora_config.moe_shared_expert_intermediate_size > 0)
        ? mConfig.qlora_config.moe_shared_expert_intermediate_size
        : moe_inter;
    const int shared_M = mConfig.mlp_up_factor * shared_inter;

    // Calculate the maximum FP4 packed size for a single expert weight
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
    const long max_packed_bytes = FP4BlockScaleConfig::packed_data_bytes(
        std::max(std::max(moe_M, shared_M), hidden),
        std::max(std::max(hidden, moe_inter), shared_inter));

    // Allocate staging buffer for FP4 packed data
    mExpertFP4Staging = mAllocator->allocate(ETensorDType::BYTE, "expert_fp4_staging",
                                              EAllocationType::ON_DEVICE, {max_packed_bytes});

    // Calculate max scale tensor size
    // FP4 uses FP8 E4M3 block scales with F8_128x4 alignment
    auto [gate_up_scale_rows, gate_up_scale_cols] = FP4BlockScaleConfig::scale_dims(moe_M, hidden);
    auto [down_scale_rows, down_scale_cols] = FP4BlockScaleConfig::scale_dims(hidden, moe_inter);
    long max_scale_elems = std::max(
        static_cast<long>(gate_up_scale_rows) * gate_up_scale_cols,
        static_cast<long>(down_scale_rows) * down_scale_cols);
    if (mConfig.qlora_config.num_shared_experts > 0) {
        auto [shared_gate_rows, shared_gate_cols] = FP4BlockScaleConfig::scale_dims(shared_M, hidden);
        auto [shared_down_rows, shared_down_cols] = FP4BlockScaleConfig::scale_dims(hidden, shared_inter);
        max_scale_elems = std::max(max_scale_elems,
                                   static_cast<long>(shared_gate_rows) * shared_gate_cols);
        max_scale_elems = std::max(max_scale_elems,
                                   static_cast<long>(shared_down_rows) * shared_down_cols);
    }

    mExpertScalesStaging = mAllocator->allocate(ETensorDType::FP8_E4M3, "expert_scales_staging",
                                                 EAllocationType::ON_DEVICE, {max_scale_elems});
}

template<typename Block>
void FP4WeightProvider<Block>::stream_and_dequantize_expert(
    const FP4BlockQuantizedWeight& src, Tensor& dst, cudaStream_t stream)
{
    // This method handles CPU-offloaded FP4 experts:
    // 1. Copy FP4 packed data from pinned CPU memory to GPU staging buffer
    // 2. Copy FP8 block scales from CPU to GPU staging buffer
    // 3. Dequantize from GPU staging buffer to destination

    // Step 1: Copy FP4 packed data (HOST -> DEVICE)
    const size_t data_bytes = FP4BlockScaleConfig::packed_data_bytes(src.M, src.K);
    cudaMemcpyAsync(mExpertFP4Staging.Data, src.data.Data,
                    data_bytes, cudaMemcpyHostToDevice, stream);

    // Step 2: Copy FP8 block scales
    auto [scale_rows, scale_cols] = FP4BlockScaleConfig::scale_dims(src.M, src.K);
    const size_t scale_bytes = static_cast<size_t>(scale_rows) * scale_cols * sizeof(__nv_fp8_e4m3);
    cudaMemcpyAsync(mExpertScalesStaging.Data, src.block_scales_rowwise.Data,
                    scale_bytes, cudaMemcpyHostToDevice, stream);

    // Step 3: Dequantize from staging buffers to destination
    float global_scale = src.global_decode_scale_rowwise();
    dequantize_fp4_block(
        dst.get<nv_bfloat16>(),
        mExpertFP4Staging.get<uint8_t>(),
        mExpertScalesStaging.get<__nv_fp8_e4m3>(),
        global_scale,
        src.M, src.K,
        mDeviceProps, stream);
}

template<typename Block>
void FP4WeightProvider<Block>::dequantize_selected_experts(
    int layer_idx, const SelectiveExpertInfo& selection_info, cudaStream_t stream, bool force)
{
    if (!selection_info.enabled || selection_info.num_active == 0) {
        return;
    }

    const auto& qblock = mFP4Weights->get_moe_block(layer_idx);
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
        if (mExpertsOffloaded) {
            stream_and_dequantize_expert(expert_weights.gate_up_proj, gate_up_slice, stream);
            stream_and_dequantize_expert(expert_weights.down_proj, down_slice, stream);
        } else {
            dequantize_fp4_weight(expert_weights.gate_up_proj, gate_up_slice, stream);
            dequantize_fp4_weight(expert_weights.down_proj, down_slice, stream);
        }
    }

    // Update cache state
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

} // namespace modules

#endif // SUROGATE_SRC_MODULES_QLORA_FP4_WEIGHT_PROVIDER_H
