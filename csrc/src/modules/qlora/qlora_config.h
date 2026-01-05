// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_QLORA_QLORA_CONFIG_H
#define SUROGATE_SRC_MODULES_QLORA_QLORA_CONFIG_H

#include <utility>

#include "recipes/nvfp4/nvfp4_recipe.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace modules {

/**
 * @brief Quantization strategy for QLoRA variants
 *
 * Defines the quantization format used for base model weights.
 * Each strategy has different trade-offs between memory savings and accuracy.
 */
enum class QLoRAQuantStrategy {
    None,           ///< No quantization (regular LoRA with BF16 base model)
    FP8,            ///< FP8 E4M3 with per-block scales
    NVFP4,          ///< FP4 E2M1 with two-level block scales for SM 100+ (Blackwell)
    BitsAndBytes    ///< Future: BitsAndBytes-style NF4/INT4 with double quantization
};

/**
 * @brief Per-block scale configuration for quantized weights
 *
 * Defines the granularity of quantization scales. Smaller blocks provide
 * better numerical accuracy but more storage overhead for scales.
 */
struct BlockScaleConfig {
    /// Block size for per-block quantization (e.g., 128 means 128x128 tiles)
    int block_size = 128;

    /**
     * @brief Compute number of scale blocks for a weight matrix
     * @param M Number of rows in the weight matrix
     * @param K Number of columns in the weight matrix
     * @return Pair of (scale_rows, scale_cols)
     */
    [[nodiscard]] std::pair<int, int> num_blocks(int M, int K) const {
        return {div_ceil(M, block_size), div_ceil(K, block_size)};
    }

    /**
     * @brief Compute total number of scales for a weight matrix
     * @param M Number of rows in the weight matrix
     * @param K Number of columns in the weight matrix
     * @return Total number of scale values needed
     */
    [[nodiscard]] long num_scales(int M, int K) const {
        auto [rows, cols] = num_blocks(M, K);
        return static_cast<long>(rows) * static_cast<long>(cols);
    }

    /**
     * @brief Compute scale tensor dimensions
     * @param M Number of rows in the weight matrix
     * @param K Number of columns in the weight matrix
     * @return Pair of (scale_rows, scale_cols)
     */
    [[nodiscard]] std::pair<int, int> scale_dims(int M, int K) const {
        return num_blocks(M, K);
    }

    /**
     * @brief Compute memory overhead for scales in bytes
     * @param M Number of rows in the weight matrix
     * @param K Number of columns in the weight matrix
     * @return Bytes needed for scale storage (FP32)
     */
    [[nodiscard]] std::size_t scale_bytes(int M, int K) const {
        return num_scales(M, K) * sizeof(float);
    }
};

/**
 * @brief QLoRA configuration
 *
 * Configures quantization of base model weights for memory-efficient LoRA training.
 * The base model is stored in a quantized format (e.g., FP8) while LoRA adapters
 * remain in full precision (BF16/FP32).
 *
 * Usage:
 * - Set `strategy` to select quantization format
 * - Configure `scale_config` for block quantization granularity
 * - LoRA adapters use `adapter_dtype` (typically BF16)
 */
struct QLoRAConfig {
    /// Whether QLoRA is enabled
    bool enabled = false;

    /// Quantization strategy for base model weights
    QLoRAQuantStrategy strategy = QLoRAQuantStrategy::None;

    /// Block scale configuration for per-block quantization
    BlockScaleConfig scale_config;

    /// Storage dtype for quantized base model weights
    ETensorDType base_dtype = ETensorDType::FP8_E4M3;

    /// Dtype for LoRA adapter weights (A/B matrices) - NOT quantized
    ETensorDType adapter_dtype = ETensorDType::BF16;

    /// Four Over Six (4/6) adaptive block scaling for NVFP4 quantization.
    /// When enabled, evaluates both max=4 and max=6 scaling per block and
    /// selects the option with lower quantization error.
    bool enable_four_over_six = false;

    /// Error metric for 4/6 selection (MSE, L1, or AbsMax)
    recipes::FourOverSixErrorMetric four_over_six_metric = recipes::FourOverSixErrorMetric::MSE;

    /**
     * @brief Check if quantization is active
     */
    [[nodiscard]] bool is_quantized() const {
        return enabled && strategy != QLoRAQuantStrategy::None;
    }

    /**
     * @brief Get block size for quantization
     */
    [[nodiscard]] int block_size() const {
        return scale_config.block_size;
    }

    /**
     * @brief Check if using FP4 quantization
     */
    [[nodiscard]] bool is_fp4() const {
        return strategy == QLoRAQuantStrategy::NVFP4;
    }

    /**
     * @brief Check if using FP8 quantization
     */
    [[nodiscard]] bool is_fp8() const {
        return strategy == QLoRAQuantStrategy::FP8;
    }

    /**
     * @brief Create FP8 QLoRA configuration with default settings
     */
    static QLoRAConfig fp8(int block_size = 128) {
        QLoRAConfig cfg;
        cfg.enabled = true;
        cfg.strategy = QLoRAQuantStrategy::FP8;
        cfg.scale_config.block_size = block_size;
        cfg.base_dtype = ETensorDType::FP8_E4M3;
        cfg.adapter_dtype = ETensorDType::BF16;
        return cfg;
    }

    /**
     * @brief Create NVFP4 QLoRA configuration with default settings
     *
     * FP4 uses two-level block scaling:
     * - Level 1: FP8 E4M3 scale per 16 consecutive values
     * - Level 2: FP32 global per-tensor scale
     *
     * Requires Blackwell GPU (SM100+) for native FP4 instructions.
     */
    static QLoRAConfig nvfp4() {
        QLoRAConfig cfg;
        cfg.enabled = true;
        cfg.strategy = QLoRAQuantStrategy::NVFP4;
        cfg.scale_config.block_size = 16;  // FP4 uses 16-element blocks
        cfg.base_dtype = ETensorDType::FP4_E2M1;
        cfg.adapter_dtype = ETensorDType::BF16;
        return cfg;
    }

    /**
     * @brief Create disabled QLoRA configuration (regular LoRA)
     */
    static QLoRAConfig none() {
        return QLoRAConfig{};
    }
};

/**
 * @brief Builder for QLoRA configuration
 */
class QLoRAConfigBuilder {
public:
    QLoRAConfigBuilder() = default;

    QLoRAConfigBuilder& enable(bool v = true) {
        mConfig.enabled = v;
        return *this;
    }

    QLoRAConfigBuilder& strategy(QLoRAQuantStrategy s) {
        mConfig.strategy = s;
        if (s != QLoRAQuantStrategy::None) {
            mConfig.enabled = true;
        }
        return *this;
    }

    QLoRAConfigBuilder& fp8() {
        mConfig.strategy = QLoRAQuantStrategy::FP8;
        mConfig.base_dtype = ETensorDType::FP8_E4M3;
        mConfig.scale_config.block_size = 128;
        mConfig.enabled = true;
        return *this;
    }

    QLoRAConfigBuilder& nvfp4() {
        mConfig.strategy = QLoRAQuantStrategy::NVFP4;
        mConfig.base_dtype = ETensorDType::FP4_E2M1;
        mConfig.scale_config.block_size = 16;  // FP4 uses 16-element blocks
        mConfig.enabled = true;
        return *this;
    }

    QLoRAConfigBuilder& block_size(int size) {
        mConfig.scale_config.block_size = size;
        return *this;
    }

    QLoRAConfigBuilder& base_dtype(ETensorDType dt) {
        mConfig.base_dtype = dt;
        return *this;
    }

    QLoRAConfigBuilder& adapter_dtype(ETensorDType dt) {
        mConfig.adapter_dtype = dt;
        return *this;
    }

    QLoRAConfigBuilder& four_over_six(bool enable,
                                       recipes::FourOverSixErrorMetric metric = recipes::FourOverSixErrorMetric::MSE) {
        mConfig.enable_four_over_six = enable;
        mConfig.four_over_six_metric = metric;
        return *this;
    }

    [[nodiscard]] QLoRAConfig build() const { return mConfig; }

private:
    QLoRAConfig mConfig;
};

/**
 * @brief Get string name for quantization strategy
 */
inline const char* to_string(QLoRAQuantStrategy strategy) {
    switch (strategy) {
        case QLoRAQuantStrategy::None: return "none";
        case QLoRAQuantStrategy::FP8: return "fp8";
        case QLoRAQuantStrategy::NVFP4: return "nvfp4";
        case QLoRAQuantStrategy::BitsAndBytes: return "bitsandbytes";
        default: return "unknown";
    }
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_QLORA_QLORA_CONFIG_H
