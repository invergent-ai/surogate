// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_CONFIG_ROPE_CONFIG_H
#define SUROGATE_SRC_CONFIG_ROPE_CONFIG_H

#include <array>
#include <cmath>

/**
 * @brief Configuration for Rotary Position Embedding (RoPE)
 *
 * Supports multiple RoPE variants used by different model architectures:
 * - FULL: Standard RoPE, applies to all head dimensions (LLaMA, Qwen, Mistral)
 * - PARTIAL: Applies RoPE to a fraction of head dimensions (GLM4: partial_rotary_factor=0.5)
 * - MULTIMODAL: 3D position embeddings for vision-language models (Qwen2-VL M-RoPE)
 * - NONE: Skip RoPE entirely (useful for certain hybrid architectures)
 *
 * Example configurations:
 * @code
 * // Standard LLaMA-style RoPE
 * RoPEConfig full_rope{.mode = RoPEConfig::Mode::FULL, .theta = 10000.0f};
 *
 * // GLM4 partial RoPE (half dimensions rotated)
 * RoPEConfig partial_rope{.mode = RoPEConfig::Mode::PARTIAL, .partial_factor = 0.5f, .theta = 10000.0f};
 *
 * // Qwen2-VL multimodal RoPE
 * RoPEConfig mrope{.mode = RoPEConfig::Mode::MULTIMODAL, .mrope_section = {16, 24, 24}, .theta = 10000.0f};
 * @endcode
 */
struct RoPEConfig {
    /**
     * @brief RoPE application mode
     */
    enum class Mode {
        FULL,        ///< Standard - apply to all head dimensions
        PARTIAL,     ///< GLM4 style - apply to first (partial_factor * head_dim) dimensions
        MULTIMODAL,  ///< Qwen2-VL M-RoPE - 3D position embeddings for temporal/height/width
        NONE         ///< Skip RoPE entirely
    };

    Mode mode = Mode::FULL;

    /// Fraction of head dimensions to apply RoPE (for PARTIAL mode)
    /// GLM4 uses 0.5, meaning only first half of head dimensions are rotated
    float partial_factor = 1.0f;

    /// Section sizes for multimodal RoPE [temporal, height, width]
    /// These define how the head dimension is split for 3D position encoding
    /// Sum must equal head_dim / 2 (since RoPE works on dimension pairs)
    std::array<int, 3> mrope_section = {0, 0, 0};

    /// Base frequency for position encoding (theta in the RoPE formula)
    /// Common values: 10000.0 (original), 500000.0 (Llama 3.1), 1000000.0 (some long-context models)
    float theta = 10000.0f;

    /// Scaling factor for extended context (for models trained with scaled RoPE)
    /// 1.0 = no scaling, >1.0 = extend context length
    float scaling_factor = 1.0f;

    /**
     * @brief Compute the number of dimensions to apply RoPE to
     *
     * @param head_dim Full head dimension
     * @return Number of dimensions that will be rotated (always even)
     */
    [[nodiscard]] int rotary_dim(int head_dim) const {
        switch (mode) {
            case Mode::FULL:
                return head_dim;
            case Mode::PARTIAL:
                // Round down to even number (RoPE works on pairs)
                return (static_cast<int>(head_dim * partial_factor) / 2) * 2;
            case Mode::MULTIMODAL:
                // For M-RoPE, all dimensions are rotated but with different position IDs
                return head_dim;
            case Mode::NONE:
                return 0;
        }
        return head_dim;  // Default fallback
    }

    /**
     * @brief Check if RoPE should be applied
     */
    [[nodiscard]] bool is_enabled() const {
        return mode != Mode::NONE;
    }

    /**
     * @brief Check if this is partial RoPE (GLM4 style)
     */
    [[nodiscard]] bool is_partial() const {
        return mode == Mode::PARTIAL && partial_factor < 1.0f;
    }

    /**
     * @brief Check if this is multimodal RoPE (Qwen2-VL style)
     */
    [[nodiscard]] bool is_multimodal() const {
        return mode == Mode::MULTIMODAL;
    }

    /**
     * @brief Validate M-RoPE section configuration
     *
     * @param head_dim Full head dimension
     * @return true if mrope_section sums to head_dim/2
     */
    [[nodiscard]] bool validate_mrope_sections(int head_dim) const {
        if (mode != Mode::MULTIMODAL) return true;
        int total = mrope_section[0] + mrope_section[1] + mrope_section[2];
        return total == head_dim / 2;
    }

    /**
     * @brief Create a standard FULL mode RoPE config
     */
    static RoPEConfig full(float theta = 10000.0f) {
        return RoPEConfig{.mode = Mode::FULL, .theta = theta};
    }

    /**
     * @brief Create a PARTIAL mode RoPE config (GLM4 style)
     *
     * @param factor Fraction of dimensions to rotate (e.g., 0.5 for half)
     * @param theta Base frequency
     */
    static RoPEConfig partial(float factor, float theta = 10000.0f) {
        return RoPEConfig{.mode = Mode::PARTIAL, .partial_factor = factor, .theta = theta};
    }

    /**
     * @brief Create a NONE mode RoPE config (no rotation)
     */
    static RoPEConfig none() {
        return RoPEConfig{.mode = Mode::NONE};
    }

    /**
     * @brief Create a MULTIMODAL mode RoPE config (Qwen2-VL M-RoPE)
     *
     * @param temporal Dimensions for temporal (frame) position
     * @param height Dimensions for height position
     * @param width Dimensions for width position
     * @param theta Base frequency
     */
    static RoPEConfig multimodal(int temporal, int height, int width, float theta = 10000.0f) {
        return RoPEConfig{
            .mode = Mode::MULTIMODAL,
            .mrope_section = {temporal, height, width},
            .theta = theta
        };
    }
};

#endif // SUROGATE_SRC_CONFIG_ROPE_CONFIG_H
