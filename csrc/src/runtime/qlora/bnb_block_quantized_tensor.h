// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_QLORA_BNB_BLOCK_QUANTIZED_TENSOR_H
#define SUROGATE_SRC_MODULES_QLORA_BNB_BLOCK_QUANTIZED_TENSOR_H

#include <optional>

#include "utilities/tensor.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace modules {

/**
 * @brief Configuration for BitsAndBytes-style blockwise quantization
 *
 * BnB uses linear (1D) blocks rather than 2D blocks used in FP8 QLoRA.
 * The weight matrix is flattened and divided into blocks of consecutive elements.
 */
struct BnBBlockScaleConfig {
    /// Block size for per-block quantization (typically 64, but configurable)
    int block_size = 64;

    /// Whether to use double quantization (quantize absmax values to INT8)
    bool double_quant = true;

    /// Group size for double quantization (number of absmax values per group)
    int double_quant_group_size = 256;

    /**
     * @brief Compute number of absmax values needed for a weight matrix
     * @param M Number of rows in the weight matrix
     * @param K Number of columns in the weight matrix
     * @return Number of absmax values (one per block)
     */
    [[nodiscard]] long num_absmax(int M, int K) const {
        long total_elements = static_cast<long>(M) * K;
        return (total_elements + block_size - 1) / block_size;
    }

    /**
     * @brief Compute number of double-quant groups needed
     * @param num_absmax_values Number of absmax values to quantize
     * @return Number of groups for double quantization
     */
    [[nodiscard]] int num_double_quant_groups(long num_absmax_values) const {
        return static_cast<int>((num_absmax_values + double_quant_group_size - 1) / double_quant_group_size);
    }

    /**
     * @brief Compute memory for packed 4-bit data in bytes
     * @param M Number of rows
     * @param K Number of columns
     * @return Bytes needed for packed NF4 data
     */
    [[nodiscard]] std::size_t data_bytes(int M, int K) const {
        long total_elements = static_cast<long>(M) * K;
        return (total_elements + 1) / 2;  // 2 values per byte
    }

    /**
     * @brief Compute memory for absmax scales in bytes
     * @param M Number of rows
     * @param K Number of columns
     * @return Bytes needed for absmax storage (depends on double_quant)
     */
    [[nodiscard]] std::size_t absmax_bytes(int M, int K) const {
        long num = num_absmax(M, K);
        if (double_quant) {
            // INT8 quantized absmax + FP32 scale/offset per group
            int groups = num_double_quant_groups(num);
            return num * sizeof(unsigned char) +  // quantized absmax
                   groups * sizeof(float) +       // scales
                   groups * sizeof(float);        // offsets
        } else {
            // FP32 absmax values
            return num * sizeof(float);
        }
    }
};

/**
 * @brief A weight tensor stored in BitsAndBytes NF4 format with per-block absmax scaling
 *
 * NF4 (Normal Float 4) uses 16 asymmetric bins derived from a normal distribution,
 * providing better representation of neural network weight distributions.
 *
 * Memory layout:
 * - data: Packed 4-bit NF4 tensor (M*K/2 bytes, 2 values per byte)
 * - absmax: Per-block absolute maximum values for scaling
 *   - Without double_quant: FP32 tensor of shape (num_blocks,)
 *   - With double_quant: INT8 tensor + FP32 scale/offset per group
 *
 * Packing format: Each byte contains two 4-bit values
 * - High nibble (bits 4-7): First value
 * - Low nibble (bits 0-3): Second value
 */
struct BnBBlockQuantizedWeight {
    /// Quantized weight data (packed 4-bit NF4, UINT8 storage)
    Tensor data;

    /// Per-block absmax scale factors
    /// Without double_quant: FP32 tensor of shape (num_blocks,)
    /// With double_quant: UINT8 quantized absmax values
    Tensor absmax;

    /// Double quantization: per-group scale for absmax values
    /// Only used when double_quant is enabled
    Tensor absmax_scale;

    /// Double quantization: per-group offset for absmax values
    /// Only used when double_quant is enabled
    Tensor absmax_offset;

    /// Original number of rows in the weight matrix
    int M = 0;

    /// Original number of columns in the weight matrix
    int K = 0;

    /// Block size used for quantization
    int block_size = 64;

    /// Whether double quantization is used
    bool double_quant = true;

    /// Group size for double quantization
    int double_quant_group_size = 256;

    /**
     * @brief Check if the weight is properly initialized
     */
    [[nodiscard]] bool is_valid() const {
        if (data.Data == nullptr || absmax.Data == nullptr || M <= 0 || K <= 0) {
            return false;
        }
        if (double_quant) {
            return absmax_scale.Data != nullptr && absmax_offset.Data != nullptr;
        }
        return true;
    }

    /**
     * @brief Get total memory footprint in bytes
     */
    [[nodiscard]] std::size_t bytes() const {
        std::size_t total = data.bytes() + absmax.bytes();
        if (double_quant) {
            total += absmax_scale.bytes() + absmax_offset.bytes();
        }
        return total;
    }

    /**
     * @brief Get number of absmax blocks
     */
    [[nodiscard]] long num_blocks() const {
        long total = static_cast<long>(M) * K;
        return (total + block_size - 1) / block_size;
    }

    /**
     * @brief Get number of double-quant groups
     */
    [[nodiscard]] int num_groups() const {
        if (!double_quant) return 0;
        return static_cast<int>((num_blocks() + double_quant_group_size - 1) / double_quant_group_size);
    }

    /**
     * @brief Get total number of elements
     */
    [[nodiscard]] long num_elements() const {
        return static_cast<long>(M) * K;
    }

    /**
     * @brief Get number of packed bytes in data tensor
     */
    [[nodiscard]] long packed_bytes() const {
        return (num_elements() + 1) / 2;
    }
};

/**
 * @brief Collection of BnB NF4 quantized weights for one transformer block
 *
 * Contains all the linear projection weights for a single transformer layer,
 * stored in NF4 with per-block absmax scaling. Layer norm weights are kept
 * in BF16 since they are small and quantization provides negligible benefit.
 */
struct BnBBlockWeights {
    /// Fused Q/K/V projection (3 * num_heads * head_size, hidden_size)
    BnBBlockQuantizedWeight qkv_proj;

    /// Attention output projection (hidden_size, num_heads * head_size)
    BnBBlockQuantizedWeight out_proj;

    /// Fused gate+up projection (2 * intermediate_size, hidden_size)
    BnBBlockQuantizedWeight gate_up_proj;

    /// MLP down projection (hidden_size, intermediate_size)
    BnBBlockQuantizedWeight down_proj;

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
     * @brief Check if all projections are valid
     */
    [[nodiscard]] bool is_valid() const {
        return qkv_proj.is_valid() && out_proj.is_valid() &&
               gate_up_proj.is_valid() && down_proj.is_valid();
    }
};

/**
 * @brief Embedding weights for BnB QLoRA (not quantized)
 *
 * Embedding and LM head weights are kept in full precision since they are
 * accessed sparsely and quantization provides limited benefit.
 */
struct BnBEmbeddingWeights {
    /// Token embedding weights (vocab_size, hidden_size)
    Tensor embedding;

    /// LM head weights (vocab_size, hidden_size) - may share with embedding
    Tensor lm_head;

    /// Final RMSNorm weight before LM head (hidden_size,)
    Tensor final_norm;

    /// Whether embedding and lm_head share the same storage
    bool tied_weights = false;

    [[nodiscard]] std::size_t bytes() const {
        std::size_t total = embedding.bytes() + final_norm.bytes();
        if (!tied_weights) {
            total += lm_head.bytes();
        }
        return total;
    }
};

/**
 * @brief Configuration for BnB weight manager
 */
struct BnBWeightsConfig {
    int num_layers = 0;
    int hidden_size = 0;
    int num_attention_heads = 0;
    int num_kv_heads = 0;
    int head_size = 0;
    int intermediate_size = 0;
    int vocab_size = 0;
    bool has_qk_norm = false;
    bool tie_word_embeddings = true;

    /// BnB-specific configuration
    BnBBlockScaleConfig scale_config;

    /**
     * @brief Validate configuration
     */
    [[nodiscard]] bool is_valid() const {
        return num_layers > 0 && hidden_size > 0 && num_attention_heads > 0 &&
               head_size > 0 && intermediate_size > 0 && vocab_size > 0;
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_QLORA_BNB_BLOCK_QUANTIZED_TENSOR_H
