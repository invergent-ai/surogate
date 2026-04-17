// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_OPTIMIZERS_OPTIMIZER_BASE_H
#define SUROGATE_SRC_MODULES_OPTIMIZERS_OPTIMIZER_BASE_H

#include <string>
#include <string_view>
#include <stdexcept>

namespace optimizers {

/**
 * @brief Supported optimizer types
 */
enum class OptimizerType {
    ADAMW,       // full-precision AdamW
    ADAMW_8BIT,  // 8-bit AdamW with softsign/sqrt quantization (FlashOptim-style)
    NORMUON      // Hybrid AdamW-8bit (1D) + NorMuon with Polar Express (2D)
};

/**
 * @brief Convert string to OptimizerType
 */
inline OptimizerType optimizer_type_from_str(std::string_view str) {
    if (str == "adamw" || str == "adam") {
        return OptimizerType::ADAMW;
    } else if (str == "adamw_8bit") {
        return OptimizerType::ADAMW_8BIT;
    } else if (str == "normuon") {
        return OptimizerType::NORMUON;
    }
    throw std::runtime_error("Unknown optimizer type: " + std::string(str));
}

/**
 * @brief Convert OptimizerType to string
 */
inline std::string_view optimizer_type_to_str(OptimizerType type) {
    switch (type) {
        case OptimizerType::ADAMW: return "adamw";
        case OptimizerType::ADAMW_8BIT: return "adamw_8bit";
        case OptimizerType::NORMUON: return "normuon";
    }
    return "unknown";
}

// Alias for consistency with other to_string functions in the codebase
inline std::string to_string(OptimizerType type) {
    return std::string(optimizer_type_to_str(type));
}

}  // namespace optimizers

#endif  // SUROGATE_SRC_MODULES_OPTIMIZERS_OPTIMIZER_BASE_H
