// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// QuantizedTensor: Architecture-agnostic quantized tensor storage.
//
// This struct stores a quantized tensor regardless of which quantization
// format (NF4, FP8, FP4) is used. The IQuantizer interface produces and
// consumes these, while GenericWeightManager stores them.

#ifndef SUROGATE_SRC_RUNTIME_QLORA_QUANTIZED_TENSOR_H
#define SUROGATE_SRC_RUNTIME_QLORA_QUANTIZED_TENSOR_H

#include <cstdint>
#include <string>
#include <vector>

#include "utilities/tensor.h"

namespace qlora {

/// Quantization format identifier.
enum class QuantFormat : int {
    NONE = 0,     // Not quantized (full precision)
    BNB_NF4,      // bitsandbytes NF4 with optional double quantization
    FP8_PER_BLOCK,// Per-block FP8 E4M3 with block scales
    FP4_BLOCK_2D, // Two-level block scaling FP4 E2M1 (Blackwell)
};

/// Storage for a single quantized tensor.
///
/// This is format-agnostic: the actual layout of `data`, `scales`, and
/// `meta` depends on the QuantFormat. The IQuantizer that created the
/// QuantizedTensor knows how to interpret and dequantize it.
///
/// Memory ownership: The QuantizedTensor does NOT own its memory.
/// Buffers are allocated by GenericWeightManager (or the caller) and
/// the QuantizedTensor merely holds pointers. Lifetimes are managed
/// by the weight manager.
struct QuantizedTensor {
    /// Original matrix dimensions (rows x cols) before quantization.
    int M = 0;
    int K = 0;

    /// Quantization format used.
    QuantFormat format = QuantFormat::NONE;

    /// Block size for per-block quantization (e.g., 64 for BnB NF4).
    int block_size = 64;

    /// Primary quantized data.
    /// - BNB_NF4: packed 4-bit data (2 values per byte), BYTE dtype
    /// - FP8_PER_BLOCK: FP8 E4M3 data, FP8_E4M3 dtype
    /// - FP4_BLOCK_2D: packed FP4 data, BYTE dtype
    Tensor data;

    /// Per-block scale factors.
    /// - BNB_NF4: absmax per block (FP32 or BYTE for double-quant)
    /// - FP8_PER_BLOCK: per-block FP32 scales
    /// - FP4_BLOCK_2D: per-block FP8 E4M3 scales (row-wise)
    Tensor scales;

    /// Additional metadata (format-specific).
    /// - BNB_NF4 with double quantization: absmax_scale (FP32) and absmax_offset (FP32)
    /// - FP4_BLOCK_2D: column-wise FP8 scales
    /// - FP8_PER_BLOCK: unused
    Tensor meta;
    Tensor meta2;

    /// Whether double quantization is enabled (BnB NF4 only).
    bool double_quant = false;

    /// Double quantization group size (BnB NF4 only, typically 256).
    int double_quant_group_size = 256;

    /// Global decode scale for FP4 row-wise dequantization.
    float global_scale = 1.0f;

    /// Number of elements in the original (unquantized) tensor.
    [[nodiscard]] long nelem() const { return static_cast<long>(M) * K; }

    /// Number of quantization blocks.
    [[nodiscard]] long num_blocks() const {
        return (nelem() + block_size - 1) / block_size;
    }

    /// Size of packed data in bytes (for NF4: 2 values per byte).
    [[nodiscard]] size_t packed_bytes() const {
        if (format == QuantFormat::BNB_NF4) {
            return (static_cast<size_t>(nelem()) + 1) / 2;
        }
        // For FP8/FP4, data tensor already has correct size
        return data.is_null() ? 0 : static_cast<size_t>(data.nelem()) * get_dtype_size(data.DType);
    }

    /// Whether this tensor has been quantized.
    [[nodiscard]] bool is_quantized() const { return format != QuantFormat::NONE; }

    /// Whether this tensor's buffers are on the host (CPU).
    [[nodiscard]] bool is_on_host() const { return !data.is_null() && data.Device < 0; }
};

/// Metadata about a weight parameter from the DSL IR.
struct WeightParamInfo {
    std::string name;        // Internal parameter name (e.g., "blocks[0].qkv_weight")
    int M = 0;               // Rows
    int K = 0;               // Cols
    bool quantizable = true; // Whether this weight can be quantized
    int offload_group = -1;  // -1 = no offloading, >= 0 = group ID
    int layer_idx = -1;      // Layer index for block parameters (-1 for global)
};

}  // namespace qlora

#endif  // SUROGATE_SRC_RUNTIME_QLORA_QUANTIZED_TENSOR_H
