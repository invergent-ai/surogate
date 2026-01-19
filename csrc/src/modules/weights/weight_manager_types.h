// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_TYPES_H
#define LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_TYPES_H

#include <array>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "config/pretrained_config.h"
#include "recipes/nvfp4/nvfp4_recipe.h"
#include "utilities/allocator.h"
#include "utilities/tensor.h"

namespace modules {

// Helper type trait to detect if a block has mlp_up_weight (dense block vs MoE)
// Must be defined early since it's used in multiple template functions.
template<typename T, typename = void>
struct has_mlp_weights : std::false_type {};

template<typename T>
struct has_mlp_weights<T, std::void_t<decltype(std::declval<T>().mlp_up_weight)>> : std::true_type {};

// Helper type trait to detect if a block has Mamba weights
template<typename T, typename = void>
struct has_mamba_weights : std::false_type {};

template<typename T>
struct has_mamba_weights<T, std::void_t<decltype(std::declval<T>().mamba)>> : std::true_type {};

// Helper type trait to detect if a block has MoE-specific weights (router, experts)
template<typename T, typename = void>
struct has_moe_weights : std::false_type {};

template<typename T>
struct has_moe_weights<T, std::void_t<decltype(std::declval<T>().router)>> : std::true_type {};

// Helper type trait to detect if an attention Weights struct has QK norm weights
// (Qwen3-style attention with q_norm_weight and k_norm_weight)
template<typename T, typename = void>
struct has_qk_norm_weights : std::false_type {};

template<typename T>
struct has_qk_norm_weights<T, std::void_t<decltype(std::declval<T>().q_norm_weight)>> : std::true_type {};

/**
 * @brief Status tracking for double-buffered prefetching
 */
struct GatherStatus {
    int layer_idx = -1;                ///< Which layer is stored in this buffer
    cudaEvent_t done_event = nullptr;  ///< Event to synchronize on
    bool fetch_pending = false;        ///< Whether a gather is in progress
    bool is_ready = true;              ///< Whether buffer is available for reuse
    int version = -1;                  ///< Cache version for invalidation
};

/**
 * @brief Non-block weights (embeddings, final norm, lm head)
 */
struct NonBlockWeights {
    Tensor embeddings;         ///< (vocab_size, hidden_size)
    Tensor lm_head;            ///< (vocab_size, hidden_size) - may alias embeddings
    Tensor final_norm_weight;  ///< (hidden_size,)
};

/**
 * @brief FP8 weight cache for FP8 forward-only mode
 */
struct FP8WeightCache {
    Tensor qkv_weight;      ///< FP8 E4M3, (QKV_C, C)
    Tensor o_weight;        ///< FP8 E4M3, (C, Hq*Hs)
    Tensor mlp_up_weight;   ///< FP8 E4M3, (2*D, C)
    Tensor mlp_down_weight; ///< FP8 E4M3, (C, D)
};

/**
 * @brief Cached FP4 weights for a single layer (CUTLASS layout).
 */
struct FP4WeightCacheEntry {
    Tensor data;        ///< FP4 packed data, shape (N, K/2)
    Tensor scales;      ///< Block scales, FP8 E4M3, CUTLASS layout
    float global_amax;  ///< Global amax used during quantization (host value)
};

struct FP4WeightCache {
    FP4WeightCacheEntry qkv_weight;
    FP4WeightCacheEntry o_weight;
    FP4WeightCacheEntry mlp_up_weight;
    FP4WeightCacheEntry mlp_down_weight;
};

/**
 * @brief Configuration for ModularWeightManager.
 *
 * This is defined outside of the class for lighter includes; the manager exposes
 * it as `ModularWeightManager<Block>::Config` via a type alias.
 */
template<typename Block>
struct ModularWeightManagerConfig {
    using BlockConfig = typename Block::Config;

    int num_layers;
    BlockConfig block_config;

    // Data types
    ETensorDType master_dtype;  ///< Master weight dtype (typically BF16 or FP32)
    ETensorDType model_dtype;   ///< Non-matmul working dtype (typically BF16/FP32)
    ETensorDType matmul_dtype;  ///< Working dtype for linear/matmul weights (may be FP8)

    // Sharding
    int shard_idx;
    int num_shards;
    bool shard_weights = false;  ///< ZeRO-3/FSDP-like weight streaming (not fully implemented in modular path)

    // Offloading
    bool offload_master = false;                     ///< Offload master weights to CPU
    bool offload_quants = false;                     ///< Offload quantized weights to CPU (requires persistent_quants)
    bool use_zero_copy = false;                      ///< Use zero-copy for CPU-GPU transfers
    EAllocationType offload_alloc = EAllocationType::PINNED;  ///< Host allocation kind for offloaded weights
    bool persistent_quants = false;                  ///< Keep quantized weights around instead of re-quantizing
    bool init_projections_to_zero = false;           ///< Init attn.out + ffn.down to zero (modded-nanogpt)

    // Non-block weights config
    int vocab_size;
    int hidden_size;
    bool tied_embeddings;

    // Architecture ID for weight mapping lookup
    PretrainedConfig::ArchitectureId architecture_id = PretrainedConfig::LLAMA;

    // Hybrid layer markers (optional)
    bool has_mamba = false;
    std::vector<std::uint8_t> layer_is_mamba;  ///< 1 if layer uses Mamba block
    bool has_moe = false;
    std::vector<std::uint8_t> layer_is_moe;    ///< 1 if layer uses MoE block

    // QLoRA: skip block weight allocation (weights provided externally via set_weight_provider)
    bool skip_block_allocation = false;

    // FP8 forward-only mode: cache FP8 weights for forward pass
    bool enable_fp8_forward = false;

    // FP4 forward-only mode: cache FP4 weights for forward pass (CUTLASS layout)
    bool enable_fp4_forward = false;

    // Four Over Six (4/6) adaptive block scaling for FP4 quantization.
    bool enable_four_over_six = false;
    recipes::FourOverSixErrorMetric four_over_six_metric = recipes::FourOverSixErrorMetric::MSE;

    // MoE convenience fields (mirroring ModelConfig)
    int NumExperts = 0;
    int NumExpertsPerTok = 0;
    int MoeIntermediateSize = 0;

    [[nodiscard]] bool is_layer_moe(int layer_idx) const {
        if (layer_idx < 0) {
            return NumExperts > 0;
        }
        if (layer_idx >= static_cast<int>(layer_is_moe.size())) {
            return NumExperts > 0;
        }
        return layer_is_moe[static_cast<std::size_t>(layer_idx)] != 0;
    }
};

} // namespace modules

#endif // LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_TYPES_H
