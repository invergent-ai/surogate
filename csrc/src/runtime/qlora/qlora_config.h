// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_QLORA_QLORA_CONFIG_H
#define SUROGATE_SRC_MODULES_QLORA_QLORA_CONFIG_H

#include <string>
#include <utility>
#include <vector>

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
    FP8,            ///< FP8 E4M3 with per-block scales (online quantization)
    NVFP4,          ///< FP4 E2M1 with two-level block scales for SM 100+ (online quantization)
    BitsAndBytes,   ///< BitsAndBytes-style NF4 with per-block absmax and double quantization

    // Pre-quantized HF model loading (no online quantization, weights already quantized)
    PrequantFP8,    ///< HF fine-grained FP8: per-block (128x128) FP8 E4M3 + FP32 inverse scales
    PrequantNVFP4,  ///< HF NVFP4 (ModelOpt): packed FP4 + FP8 block scales + FP32 global scale
    PrequantMXFP4,  ///< HF MXFP4: packed FP4 + E8M0 shared exponents per 32-element block
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

    // =========================================================================
    // BitsAndBytes-specific configuration
    // =========================================================================

    /// Enable double quantization for BnB (quantize absmax values to INT8)
    /// Reduces memory overhead by ~0.4 bits per parameter
    bool bnb_double_quant = true;

    /// Group size for double quantization (number of absmax values per group)
    int bnb_double_quant_group_size = 256;

    // =========================================================================
    // MoE (Mixture of Experts) configuration
    // =========================================================================

    /// Number of experts (0 = dense model, >0 = MoE model)
    int num_experts = 0;

    /// Number of experts selected per token (top-k routing)
    int num_experts_per_tok = 8;

    /// Per-expert MLP intermediate size (0 = use regular intermediate_size)
    int moe_intermediate_size = 0;

    /// Number of shared experts (0 = none)
    int num_shared_experts = 0;

    /// Shared expert intermediate size (0 = use moe_intermediate_size or intermediate_size)
    int moe_shared_expert_intermediate_size = 0;

    // =========================================================================
    // Pre-quantized model configuration
    // =========================================================================

    /// HF module paths that should NOT be quantized (loaded as full-precision).
    /// Populated from HF quantization_config "ignore" or "modules_to_not_convert".
    /// Examples: "lm_head", "model.layers.0.mlp.gate" (router gates in MoE).
    std::vector<std::string> modules_to_not_convert;

    /**
     * @brief Check if this is an MoE model
     */
    [[nodiscard]] bool is_moe() const { return num_experts > 0; }

    /**
     * @brief Check if shared expert is enabled
     */
    [[nodiscard]] bool use_shared_expert() const { return num_shared_experts > 0; }

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
     * @brief Check if using BitsAndBytes NF4 quantization
     */
    [[nodiscard]] bool is_bnb() const {
        return strategy == QLoRAQuantStrategy::BitsAndBytes;
    }

    /**
     * @brief Check if loading a pre-quantized HF model (no online quantization)
     */
    [[nodiscard]] bool is_prequantized() const {
        return strategy == QLoRAQuantStrategy::PrequantFP8 ||
               strategy == QLoRAQuantStrategy::PrequantNVFP4 ||
               strategy == QLoRAQuantStrategy::PrequantMXFP4;
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
     * @brief Create BitsAndBytes NF4 QLoRA configuration
     *
     * NF4 (Normal Float 4-bit) uses 16 asymmetric bins derived from a normal
     * distribution, which better represents neural network weight distributions.
     *
     * Features:
     * - Per-block absmax scaling (block_size consecutive elements share one scale)
     * - Double quantization (quantize absmax to INT8 for additional memory savings)
     * - Works on any CUDA GPU (no SM89+ or SM100+ requirement)
     *
     * @param block_size Number of consecutive elements per quantization block
     *                   Valid values: 64, 128, 256, 512 (default: 64)
     * @param double_quant Enable double quantization (default: true)
     */
    static QLoRAConfig bnb(int block_size = 64, bool double_quant = true) {
        QLoRAConfig cfg;
        cfg.enabled = true;
        cfg.strategy = QLoRAQuantStrategy::BitsAndBytes;
        cfg.scale_config.block_size = block_size;
        cfg.base_dtype = ETensorDType::BYTE;  // Packed 4-bit NF4 stored as uint8
        cfg.adapter_dtype = ETensorDType::BF16;
        cfg.bnb_double_quant = double_quant;
        return cfg;
    }

    /**
     * @brief Create config for loading HF pre-quantized fine-grained FP8 models
     *
     * For models like DeepSeek-V3/R1 with quantization_config.quant_method = "fp8".
     * Weights are stored as FP8 E4M3 with per-block (128x128) FP32 inverse scales.
     * Reuses the existing FP8_PER_BLOCK dequant path (formats are identical).
     */
    static QLoRAConfig prequant_fp8() {
        QLoRAConfig cfg;
        cfg.enabled = true;
        cfg.strategy = QLoRAQuantStrategy::PrequantFP8;
        cfg.scale_config.block_size = 128;
        cfg.base_dtype = ETensorDType::FP8_E4M3;
        cfg.adapter_dtype = ETensorDType::BF16;
        return cfg;
    }

    /**
     * @brief Create config for loading HF pre-quantized NVFP4 models
     *
     * For models quantized with NVIDIA ModelOpt (quant_method = "modelopt",
     * quant_algo = "NVFP4"). Weights are packed FP4 with two-level scaling:
     * per-16-element FP8 E4M3 block scales + per-tensor FP32 global scale.
     * Requires Blackwell GPU (SM100+).
     */
    static QLoRAConfig prequant_nvfp4() {
        QLoRAConfig cfg;
        cfg.enabled = true;
        cfg.strategy = QLoRAQuantStrategy::PrequantNVFP4;
        cfg.scale_config.block_size = 16;
        cfg.base_dtype = ETensorDType::FP4_E2M1;
        cfg.adapter_dtype = ETensorDType::BF16;
        return cfg;
    }

    /**
     * @brief Create config for loading HF pre-quantized MXFP4 models
     *
     * For models like OpenAI GPT-OSS with quantization_config.quant_method = "mxfp4".
     * Weights are packed FP4 with E8M0 shared exponents per 32-element block
     * (microscaling format).
     */
    static QLoRAConfig prequant_mxfp4() {
        QLoRAConfig cfg;
        cfg.enabled = true;
        cfg.strategy = QLoRAQuantStrategy::PrequantMXFP4;
        cfg.scale_config.block_size = 32;
        cfg.base_dtype = ETensorDType::BYTE;  // Packed FP4 stored as uint8
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
 * @brief Get string name for quantization strategy
 */
inline const char* to_string(QLoRAQuantStrategy strategy) {
    switch (strategy) {
        case QLoRAQuantStrategy::None: return "none";
        case QLoRAQuantStrategy::FP8: return "fp8";
        case QLoRAQuantStrategy::NVFP4: return "nvfp4";
        case QLoRAQuantStrategy::BitsAndBytes: return "bitsandbytes";
        case QLoRAQuantStrategy::PrequantFP8: return "prequant_fp8";
        case QLoRAQuantStrategy::PrequantNVFP4: return "prequant_nvfp4";
        case QLoRAQuantStrategy::PrequantMXFP4: return "prequant_mxfp4";
        default: return "unknown";
    }
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_QLORA_QLORA_CONFIG_H
