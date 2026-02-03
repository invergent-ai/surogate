// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// FP4 E2M1 block-quantized tensor structures for NVFP4 training.

#ifndef SUROGATE_SRC_MODULES_QLORA_FP4_BLOCK_QUANTIZED_TENSOR_H
#define SUROGATE_SRC_MODULES_QLORA_FP4_BLOCK_QUANTIZED_TENSOR_H

#include <optional>
#include <cstdint>

#include "utilities/tensor.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace modules {

/**
 * @brief FP4 block scale configuration for NVFP4 two-level scaling
 *
 * NVFP4 uses two-level block scaling:
 * - Level 1: FP8 E4M3 scale per 16 consecutive values
 * - Level 2: FP32 global per-tensor scale
 *
 * Block scale tensor shape: (ceil(M/128)*128, ceil(K/16/4)*4)
 * with F8_128x4 tensor reordering for cuBLAS compatibility.
 */
struct FP4BlockScaleConfig {
    /// Block size for Level 1 FP8 scales (16 for NVFP4)
    static constexpr int BLOCK_SIZE = 16;

    /// Tile size for F8_128x4 alignment (128 rows)
    static constexpr int TILE_SIZE = 128;

    /// Column alignment for F8_128x4 format (4 columns)
    static constexpr int COL_ALIGN = 4;

    /**
     * @brief Compute FP8 block scale tensor dimensions
     * @param M Number of rows in the weight matrix
     * @param K Number of columns in the weight matrix
     * @return Pair of (scale_rows, scale_cols) with F8_128x4 alignment
     */
    [[nodiscard]] static std::pair<int, int> scale_dims(int M, int K) {
        // Rows: aligned to TILE_SIZE (128)
        int scale_rows = div_ceil(M, TILE_SIZE) * TILE_SIZE;
        // Columns: K/16 scales, aligned to COL_ALIGN (4)
        int scale_cols = div_ceil(div_ceil(K, BLOCK_SIZE), COL_ALIGN) * COL_ALIGN;
        return {scale_rows, scale_cols};
    }

    /**
     * @brief Compute number of FP8 block scales (without alignment padding)
     * @param M Number of rows in the weight matrix
     * @param K Number of columns in the weight matrix
     * @return Pair of (scale_rows, scale_cols) without padding
     */
    [[nodiscard]] static std::pair<int, int> logical_scale_dims(int M, int K) {
        return {div_ceil(M, TILE_SIZE), div_ceil(K, BLOCK_SIZE)};
    }

    /**
     * @brief Compute total number of FP8 scales (with alignment)
     */
    [[nodiscard]] static long num_scales(int M, int K) {
        auto [rows, cols] = scale_dims(M, K);
        return static_cast<long>(rows) * cols;
    }

    /**
     * @brief Compute packed FP4 data size in bytes
     * @param M Number of rows
     * @param K Number of columns
     * @return Size in bytes (2 FP4 values per byte)
     */
    [[nodiscard]] static std::size_t packed_data_bytes(int M, int K) {
        return (static_cast<std::size_t>(M) * K + 1) / 2;
    }

    /**
     * @brief Compute memory for FP8 block scales in bytes
     */
    [[nodiscard]] static std::size_t scale_bytes(int M, int K) {
        return num_scales(M, K) * sizeof(__nv_fp8_e4m3);
    }
};

/**
 * @brief A weight tensor stored in FP4 E2M1 format with two-level block scaling
 *
 * Memory layout for NVFP4:
 * - data: Packed FP4 E2M1 tensor of shape (M, K/2) bytes
 * - block_scales_rowwise: FP8 E4M3 scales for row-major access, F8_128x4 swizzled
 * - block_scales_colwise: FP8 E4M3 scales for column-major access (transposed)
 * - global_scale: FP32 per-tensor scale (stored in Tensor::Stats)
 *
 * The two scale tensors (rowwise/colwise) allow efficient matmul in both TN and NT modes.
 */
struct FP4BlockQuantizedWeight {
    /// Packed FP4 data (2 values per byte)
    Tensor data;

    /// Per-block FP8 E4M3 scales for row-major access (M dimension leading)
    /// Shape: (ceil(M/128)*128, ceil(K/16/4)*4), F8_128x4 swizzled
    Tensor block_scales_rowwise;

    /// Per-block FP8 E4M3 scales for column-major access (K dimension leading)
    /// Shape: (ceil(K/128)*128, ceil(M/16/4)*4), F8_128x4 swizzled
    Tensor block_scales_colwise;

    /// Global amax for rowwise scales (stored on device for async updates)
    float* global_amax_rowwise = nullptr;

    /// Global amax for colwise scales
    float* global_amax_colwise = nullptr;

    /// Cached host copy of the true global amax (rowwise) computed at import time.
    /// Used to derive the correct global decode scale for dequantization.
    float global_amax_rowwise_host = 0.0f;

    /// Cached host global decode scale for rowwise access.
    /// For the NVFP4 two-level scaling used here:
    ///   global_encode_scale = fp8_max * fp4_max / global_amax
    ///   block_scale stores: (block_amax / fp4_max) * global_encode_scale
    /// To recover original-domain values from FP4 + block_scale, multiply by:
    ///   global_decode_scale = global_amax / (fp8_max * fp4_max)
    float global_decode_scale_rowwise_host = 1.0f;

    /// Cached host copy of the true global amax (colwise), if populated.
    float global_amax_colwise_host = 0.0f;

    /// Cached host global decode scale for colwise access.
    float global_decode_scale_colwise_host = 1.0f;

    /// Original number of rows in the weight matrix
    int M = 0;

    /// Original number of columns in the weight matrix
    int K = 0;

    /**
     * @brief Check if the weight is properly initialized
     */
    [[nodiscard]] bool is_valid() const {
        return data.Data != nullptr &&
               block_scales_rowwise.Data != nullptr &&
               M > 0 && K > 0;
    }

    /**
     * @brief Get total memory footprint in bytes
     */
    [[nodiscard]] std::size_t bytes() const {
        std::size_t total = data.bytes() + block_scales_rowwise.bytes();
        if (block_scales_colwise.Data != nullptr) {
            total += block_scales_colwise.bytes();
        }
        // Global amax values (2 floats each for rowwise/colwise)
        total += 2 * sizeof(float);
        return total;
    }

    /**
     * @brief Get FP4 data pointer
     */
    [[nodiscard]] uint8_t* fp4_data() {
        return data.get<uint8_t>();
    }

    [[nodiscard]] const uint8_t* fp4_data() const {
        return data.get<uint8_t>();
    }

    /**
     * @brief Get rowwise block scales pointer (FP8 E4M3)
     */
    [[nodiscard]] __nv_fp8_e4m3* scales_rowwise() {
        return block_scales_rowwise.get<__nv_fp8_e4m3>();
    }

    [[nodiscard]] const __nv_fp8_e4m3* scales_rowwise() const {
        return block_scales_rowwise.get<__nv_fp8_e4m3>();
    }

    /**
     * @brief Get colwise block scales pointer (FP8 E4M3)
     */
    [[nodiscard]] __nv_fp8_e4m3* scales_colwise() {
        return block_scales_colwise.Data ? block_scales_colwise.get<__nv_fp8_e4m3>() : nullptr;
    }

    [[nodiscard]] const __nv_fp8_e4m3* scales_colwise() const {
        return block_scales_colwise.Data ? block_scales_colwise.get<__nv_fp8_e4m3>() : nullptr;
    }

    /**
     * @brief Get scale dimensions for rowwise access
     */
    [[nodiscard]] std::pair<int, int> rowwise_scale_dims() const {
        return FP4BlockScaleConfig::scale_dims(M, K);
    }

    /**
     * @brief Get scale dimensions for colwise access (transposed)
     */
    [[nodiscard]] std::pair<int, int> colwise_scale_dims() const {
        return FP4BlockScaleConfig::scale_dims(K, M);
    }

    /**
     * @brief Get global decode scale for rowwise access
     * @return Scale to multiply FP4 values after block descale
     */
    [[nodiscard]] float global_decode_scale_rowwise() const {
        return global_decode_scale_rowwise_host;
    }

    [[nodiscard]] float global_decode_scale_colwise() const {
        return global_decode_scale_colwise_host;
    }
};

/**
 * @brief Collection of FP4 block-quantized weights for one transformer block
 */
struct FP4BlockWeights {
    /// Fused Q/K/V projection (qkv_out, hidden_size)
    FP4BlockQuantizedWeight qkv_proj;

    /// Attention output projection (hidden_size, num_heads * head_size)
    FP4BlockQuantizedWeight out_proj;

    /// Fused gate+up projection (2 * intermediate_size, hidden_size)
    FP4BlockQuantizedWeight gate_up_proj;

    /// MLP down projection (hidden_size, intermediate_size)
    FP4BlockQuantizedWeight down_proj;

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
};

/**
 * @brief FP4-specific block scale configuration builder
 */
struct FP4ScaleConfig {
    /**
     * @brief Compute allocations needed for FP4 quantized weight
     * @param M Number of rows
     * @param K Number of columns
     * @param include_colwise Whether to allocate colwise scales for transposed access
     * @return Tuple of (data_bytes, rowwise_scale_bytes, colwise_scale_bytes)
     */
    static std::tuple<std::size_t, std::size_t, std::size_t>
    compute_allocation_sizes(int M, int K, bool include_colwise = true) {
        std::size_t data_bytes = FP4BlockScaleConfig::packed_data_bytes(M, K);
        std::size_t rowwise_bytes = FP4BlockScaleConfig::scale_bytes(M, K);
        std::size_t colwise_bytes = include_colwise ? FP4BlockScaleConfig::scale_bytes(K, M) : 0;
        return {data_bytes, rowwise_bytes, colwise_bytes};
    }

    /**
     * @brief Get rowwise scale tensor shape
     */
    static std::pair<long, long> rowwise_scale_shape(int M, int K) {
        auto [rows, cols] = FP4BlockScaleConfig::scale_dims(M, K);
        return {rows, cols};
    }

    /**
     * @brief Get colwise scale tensor shape (for transposed access)
     */
    static std::pair<long, long> colwise_scale_shape(int M, int K) {
        auto [rows, cols] = FP4BlockScaleConfig::scale_dims(K, M);
        return {rows, cols};
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_QLORA_FP4_BLOCK_QUANTIZED_TENSOR_H
