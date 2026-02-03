// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_QLORA_BLOCK_QUANTIZED_TENSOR_H
#define SUROGATE_SRC_MODULES_QLORA_BLOCK_QUANTIZED_TENSOR_H

#include <optional>

#include "utilities/tensor.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"
#include "qlora_config.h"

namespace modules {

/**
 * @brief A weight tensor stored in quantized format with per-block scales
 *
 * Used for QLoRA-style memory-efficient storage of base model weights.
 * Each (block_size x block_size) tile has its own scale factor for better
 * numerical accuracy compared to per-tensor quantization.
 *
 * Memory layout:
 * - data: FP8 E4M3 tensor of shape (M, K)
 * - block_scales: FP32 tensor of shape (ceil(M/block_size), ceil(K/block_size))
 */
struct BlockQuantizedWeight {
    /// Quantized weight data (FP8 E4M3)
    Tensor data;

    /// Per-block scale factors (FP32) for dequantization
    /// Each scale is the inverse of the quantization scale (abs_max / 448)
    Tensor block_scales;

    /// Original number of rows in the weight matrix
    int M = 0;

    /// Original number of columns in the weight matrix
    int K = 0;

    /// Block size used for quantization (e.g., 128)
    int block_size = 128;

    /**
     * @brief Check if the weight is properly initialized
     */
    [[nodiscard]] bool is_valid() const {
        return data.Data != nullptr && block_scales.Data != nullptr && M > 0 && K > 0;
    }

    /**
     * @brief Get total memory footprint in bytes
     */
    [[nodiscard]] std::size_t bytes() const {
        return data.bytes() + block_scales.bytes();
    }

    /**
     * @brief Get number of scale rows
     */
    [[nodiscard]] int scale_rows() const {
        return div_ceil(M, block_size);
    }

    /**
     * @brief Get number of scale columns
     */
    [[nodiscard]] int scale_cols() const {
        return div_ceil(K, block_size);
    }

    /**
     * @brief Get total number of scales
     */
    [[nodiscard]] long num_scales() const {
        return static_cast<long>(scale_rows()) * scale_cols();
    }

    /**
     * @brief Get pointer to scale for a specific block
     * @param block_row Block row index
     * @param block_col Block column index
     * @return Pointer to the scale value for this block
     */
    [[nodiscard]] float* scale_ptr(int block_row, int block_col) {
        return block_scales.get<float>() + block_row * scale_cols() + block_col;
    }

    [[nodiscard]] const float* scale_ptr(int block_row, int block_col) const {
        return block_scales.get<float>() + block_row * scale_cols() + block_col;
    }
};

/**
 * @brief Enum for weight types in a transformer block
 */
enum class QLoRAWeightType {
    QKV_PROJ,       ///< Fused Q/K/V projection
    OUT_PROJ,       ///< Attention output projection
    GATE_UP_PROJ,   ///< Fused gate+up projection (MLP)
    DOWN_PROJ,      ///< MLP down projection
    LN1_WEIGHT,     ///< First layer norm weight (not quantized)
    LN2_WEIGHT      ///< Second layer norm weight (not quantized)
};

/**
 * @brief Collection of block-quantized weights for one transformer block
 *
 * Contains all the linear projection weights for a single transformer layer,
 * stored in FP8 with per-block scales. Layer norm weights are kept in BF16
 * since they are small and quantization provides negligible benefit.
 */
struct QLoRABlockWeights {
    /// Fused Q/K/V projection (3 * num_heads * head_size, hidden_size)
    BlockQuantizedWeight qkv_proj;

    /// Attention output projection (hidden_size, num_heads * head_size)
    BlockQuantizedWeight out_proj;

    /// Fused gate+up projection (2 * intermediate_size, hidden_size)
    BlockQuantizedWeight gate_up_proj;

    /// MLP down projection (hidden_size, intermediate_size)
    BlockQuantizedWeight down_proj;

    /// First RMSNorm weight (BF16, not quantized - small tensor)
    Tensor ln1_weight;

    /// Second RMSNorm weight (BF16, not quantized - small tensor)
    Tensor ln2_weight;

    /// QK-norm weights for models like Qwen3 (optional, BF16)
    std::optional<Tensor> q_norm_weight;
    std::optional<Tensor> k_norm_weight;

    /**
     * @brief Get total memory footprint for this block
     */
    [[nodiscard]] std::size_t bytes() const {
        std::size_t total = qkv_proj.bytes() + out_proj.bytes() +
                            gate_up_proj.bytes() + down_proj.bytes() +
                            ln1_weight.bytes() + ln2_weight.bytes();
        if (q_norm_weight.has_value()) total += q_norm_weight->bytes();
        if (k_norm_weight.has_value()) total += k_norm_weight->bytes();
        return total;
    }

    /**
     * @brief Get a specific weight by type
     */
    [[nodiscard]] const BlockQuantizedWeight& get_quantized(QLoRAWeightType type) const {
        switch (type) {
            case QLoRAWeightType::QKV_PROJ: return qkv_proj;
            case QLoRAWeightType::OUT_PROJ: return out_proj;
            case QLoRAWeightType::GATE_UP_PROJ: return gate_up_proj;
            case QLoRAWeightType::DOWN_PROJ: return down_proj;
            default:
                throw std::runtime_error("get_quantized: invalid weight type for quantized weight");
        }
    }

    [[nodiscard]] BlockQuantizedWeight& get_quantized(QLoRAWeightType type) {
        switch (type) {
            case QLoRAWeightType::QKV_PROJ: return qkv_proj;
            case QLoRAWeightType::OUT_PROJ: return out_proj;
            case QLoRAWeightType::GATE_UP_PROJ: return gate_up_proj;
            case QLoRAWeightType::DOWN_PROJ: return down_proj;
            default:
                throw std::runtime_error("get_quantized: invalid weight type for quantized weight");
        }
    }

    /**
     * @brief Get layer norm weight by type
     */
    [[nodiscard]] const Tensor& get_ln(QLoRAWeightType type) const {
        switch (type) {
            case QLoRAWeightType::LN1_WEIGHT: return ln1_weight;
            case QLoRAWeightType::LN2_WEIGHT: return ln2_weight;
            default:
                throw std::runtime_error("get_ln: invalid weight type for layer norm");
        }
    }

    [[nodiscard]] Tensor& get_ln(QLoRAWeightType type) {
        switch (type) {
            case QLoRAWeightType::LN1_WEIGHT: return ln1_weight;
            case QLoRAWeightType::LN2_WEIGHT: return ln2_weight;
            default:
                throw std::runtime_error("get_ln: invalid weight type for layer norm");
        }
    }
};

/**
 * @brief Embedding weights for QLoRA (not quantized)
 *
 * Embedding and LM head weights are typically kept in full precision
 * since they are accessed sparsely and quantization provides limited benefit.
 */
struct QLoRAEmbeddingWeights {
    /// Token embedding weights (vocab_size, hidden_size)
    Tensor embedding;

    /// LM head weights (vocab_size, hidden_size) - may share with embedding
    Tensor lm_head;

    /// Whether embedding and lm_head share the same storage
    bool tied_weights = false;

    [[nodiscard]] std::size_t bytes() const {
        if (tied_weights) {
            return embedding.bytes();
        }
        return embedding.bytes() + lm_head.bytes();
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_QLORA_BLOCK_QUANTIZED_TENSOR_H
