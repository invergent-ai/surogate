// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_FP8_SCALING_CONFIG_H
#define SUROGATE_SRC_MODULES_FP8_SCALING_CONFIG_H

#include <cmath>

#include "utilities/dtype.h"

namespace modules {

/**
 * @brief Algorithm for computing effective amax from history buffer
 *
 * TransformerEngine-style delayed scaling uses a history buffer to smooth
 * scale factor updates. The effective amax can be computed as:
 * - MAX: Maximum across entire history (more stable, recommended)
 * - MOST_RECENT: Use only the most recent amax value
 */
enum class AmaxComputeAlgo {
    MAX,         ///< Use maximum amax across entire history window
    MOST_RECENT  ///< Use only the most recent amax value
};

/**
 * @brief Configuration for FP8 delayed scaling recipe
 *
 * Delayed scaling uses the scale factor computed from the PREVIOUS iteration's
 * abs_max values, while recording current abs_max values for future scale updates.
 * This provides more stable training compared to just-in-time scaling.
 *
 * Based on TransformerEngine's DelayedScaling recipe.
 */
struct FP8ScalingConfig {
    /// Length of the amax history window (default: 1024 following TransformerEngine)
    int amax_history_len = 1024;

    /// Scaling margin: scaled_max = fp8_max * 2^(-margin)
    /// Provides headroom to avoid overflow. Default 0 means no headroom.
    float margin = 0.0f;

    /// Epsilon floor for amax values to prevent scale instability
    /// If amax < epsilon, it's clamped to epsilon before scale computation.
    /// Default 0.0 means no floor (for backward compatibility).
    float amax_epsilon = 0.0f;

    /// Algorithm for computing effective amax from history
    AmaxComputeAlgo amax_compute_algo = AmaxComputeAlgo::MAX;

    /// FP8 format for forward pass (activations)
    ETensorDType forward_dtype = ETensorDType::FP8_E4M3;

    /// FP8 format for backward pass (gradients)
    ETensorDType backward_dtype = ETensorDType::FP8_E5M2;

    /**
     * @brief Get the scaled maximum value for a given FP8 dtype
     *
     * Returns fp8_max * 2^(-margin) where fp8_max is the maximum representable
     * value for the dtype (448 for E4M3, 57344 for E5M2).
     */
    [[nodiscard]] float get_scaled_max(ETensorDType dtype) const {
        float fp8_max = 1.0f;
        if (dtype == ETensorDType::FP8_E4M3) {
            fp8_max = 448.0f;
        } else if (dtype == ETensorDType::FP8_E5M2) {
            fp8_max = 57344.0f;
        }
        // Apply margin: scaled_max = fp8_max * 2^(-margin)
        return fp8_max * std::pow(2.0f, -margin);
    }
};

/**
 * @brief Base indices for the quantizers used in delayed scaling
 *
 * Each layer gets its own set of these quantizers for optimal per-layer scaling.
 * Total quantizers = NUM_QUANTIZERS_PER_LAYER * num_layers
 */
enum class QuantizerIndex : int {
    // Forward activations (E4M3)
    FWD_LN1 = 0,     ///< Input to QKV projection (after LayerNorm 1)
    FWD_LN2 = 1,     ///< Input to MLP up projection (after LayerNorm 2)
    FWD_ATT = 2,     ///< Input to output projection (attention output)
    FWD_SWIGLU = 3,  ///< Input to MLP down projection (SwiGLU output)

    // Backward gradients (E5M2)
    BWD_D_RES_FFN = 4,   ///< Gradient after FFN residual
    BWD_D_RES_ATT = 5,   ///< Gradient after attention residual
    BWD_D_MLP_UP = 6,    ///< Gradient for MLP up output
    BWD_D_QKV = 7,       ///< Gradient for QKV output

    NUM_QUANTIZERS_PER_LAYER = 8   ///< Number of quantizers per layer
};

constexpr int NUM_QUANTIZERS_PER_LAYER = static_cast<int>(QuantizerIndex::NUM_QUANTIZERS_PER_LAYER);

/**
 * @brief Compute global quantizer index for a specific layer and quantizer type
 *
 * @param layer_idx Layer index (0-based)
 * @param base Base quantizer type (e.g., FWD_LN1, FWD_ATT, etc.)
 * @return Global quantizer index
 */
inline constexpr int get_quantizer_index(int layer_idx, QuantizerIndex base) {
    return layer_idx * NUM_QUANTIZERS_PER_LAYER + static_cast<int>(base);
}

/**
 * @brief Get total number of quantizers for a given number of layers
 *
 * @param num_layers Number of transformer layers
 * @return Total number of quantizers needed
 */
inline constexpr int get_total_quantizers(int num_layers) {
    return num_layers * NUM_QUANTIZERS_PER_LAYER;
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_FP8_SCALING_CONFIG_H
