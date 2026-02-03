// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Helpers for MLP shape calculations across block configs.

#ifndef SUROGATE_SRC_MODULES_MLP_UTILS_H
#define SUROGATE_SRC_MODULES_MLP_UTILS_H

#include <type_traits>

namespace modules {

template<typename Config>
inline int mlp_up_factor(const Config& cfg) {
    if constexpr (requires { cfg.mlp_up_factor; }) {
        return cfg.mlp_up_factor;
    }
    return 2;  // Default to gated MLP
}

template<typename Config>
inline long mlp_up_rows(const Config& cfg) {
    if constexpr (requires { cfg.intermediate_size; }) {
        return static_cast<long>(cfg.intermediate_size) * static_cast<long>(mlp_up_factor(cfg));
    } else if constexpr (requires { cfg.IntermediateSize; }) {
        return static_cast<long>(cfg.IntermediateSize) * static_cast<long>(mlp_up_factor(cfg));
    } else {
        return 0;
    }
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MLP_UTILS_H
