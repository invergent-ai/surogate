// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Lightweight type traits and cache structs for QLoRA weight providers.

#ifndef SUROGATE_SRC_MODULES_QLORA_WEIGHT_PROVIDER_TYPES_H
#define SUROGATE_SRC_MODULES_QLORA_WEIGHT_PROVIDER_TYPES_H

#include <type_traits>
#include <utility>

#include "utilities/tensor.h"

namespace modules {

// Helper type trait to detect if a block has mlp_up_weight (dense block vs MoE)
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
template<typename T, typename = void>
struct has_qk_norm_weights : std::false_type {};

template<typename T>
struct has_qk_norm_weights<T, std::void_t<decltype(std::declval<T>().q_norm_weight)>> : std::true_type {};

// Minimal FP8 cache structure for QLoRA providers.
struct FP8WeightCache {
    Tensor qkv_weight;      ///< FP8 E4M3, (QKV_C, C)
    Tensor o_weight;        ///< FP8 E4M3, (C, Hq*Hs)
    Tensor mlp_up_weight;   ///< FP8 E4M3, (2*D, C)
    Tensor mlp_down_weight; ///< FP8 E4M3, (C, D)
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_QLORA_WEIGHT_PROVIDER_TYPES_H
