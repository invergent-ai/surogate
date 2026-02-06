// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// IQuantizer: Architecture-agnostic quantization interface.
//
// This interface abstracts the quantization algorithm so that
// GenericWeightManager can quantize ANY tensor without knowing
// whether it's an attention weight, MLP weight, Mamba parameter, etc.
//
// Concrete implementations:
// - BnBQuantizer: NF4 with optional double quantization
// - FP8Quantizer: Per-block FP8 E4M3
// - FP4Quantizer: Two-level block scaling FP4 E2M1

#ifndef SUROGATE_SRC_RUNTIME_QLORA_GENERIC_QUANTIZER_H
#define SUROGATE_SRC_RUNTIME_QLORA_GENERIC_QUANTIZER_H

#include <memory>
#include <string>
#include <cuda_runtime.h>

#include "utilities/tensor.h"
#include "utilities/allocator.h"
#include "runtime/qlora/quantized_tensor.h"

namespace qlora {

/// Configuration for quantizer construction.
struct QuantizerConfig {
    /// Quantization format to use.
    QuantFormat format = QuantFormat::BNB_NF4;

    /// Block size for per-block quantization.
    /// - BNB_NF4: typically 64
    /// - FP8_PER_BLOCK: typically 32 or 64
    /// - FP4_BLOCK_2D: typically 16 (row) x 16 (col)
    int block_size = 64;

    /// Enable double quantization for BnB NF4.
    /// Quantizes the absmax values themselves to INT8.
    bool double_quant = true;

    /// Group size for double quantization (BnB NF4 only).
    int double_quant_group_size = 256;

    /// Whether to enable FP8 matmul bypass (skip dequant when FP8 matmuls available).
    bool enable_fp8_forward = false;

    /// Whether to enable hybrid FP8 training.
    bool enable_fp8_hybrid = false;

    /// GPU device properties (for kernel dispatch).
    int device_id = 0;
    int sm_version = 0;  // e.g., 89 for RTX 40x0, 90 for H100
};

/// Abstract interface for tensor quantization/dequantization.
///
/// All methods are thread-safe and reentrant with respect to different
/// QuantizedTensor instances. However, concurrent quantize/dequantize
/// calls on the SAME QuantizedTensor from different streams require
/// external synchronization.
class IQuantizer {
public:
    virtual ~IQuantizer() = default;

    /// Quantize a BF16 tensor into the format-specific representation.
    ///
    /// @param input     Source BF16 tensor (must be 2D: [M, K])
    /// @param output    Pre-allocated QuantizedTensor to fill. The caller must
    ///                  have allocated output.data, output.scales, and output.meta
    ///                  via allocate_storage() first.
    /// @param stream    CUDA stream for async execution
    virtual void quantize(
        const Tensor& input,
        QuantizedTensor& output,
        cudaStream_t stream) = 0;

    /// Dequantize a quantized tensor back to BF16.
    ///
    /// @param input     Quantized tensor to dequantize
    /// @param output    Pre-allocated BF16 tensor [M, K] for the result
    /// @param stream    CUDA stream for async execution
    virtual void dequantize(
        const QuantizedTensor& input,
        Tensor& output,
        cudaStream_t stream) = 0;

    /// Allocate storage tensors for a QuantizedTensor of given dimensions.
    ///
    /// This allocates output.data, output.scales, and output.meta (if needed)
    /// using the provided allocator. The allocation type determines whether
    /// buffers are on GPU, CPU pinned memory, etc.
    ///
    /// @param M         Number of rows in the original tensor
    /// @param K         Number of columns in the original tensor
    /// @param output    QuantizedTensor to populate with allocated buffers
    /// @param allocator Tensor allocator to use
    /// @param alloc_type Where to allocate (GPU, CPU pinned, etc.)
    /// @param name      Name for allocation tracking
    virtual void allocate_storage(
        int M, int K,
        QuantizedTensor& output,
        TensorAllocator& allocator,
        EAllocationType alloc_type,
        const std::string& name) = 0;

    /// Get the quantization format this quantizer implements.
    [[nodiscard]] virtual QuantFormat format() const = 0;

    /// Get the block size used by this quantizer.
    [[nodiscard]] virtual int block_size() const = 0;

    /// Estimate the total bytes needed to store a quantized tensor of given dimensions.
    /// Useful for memory planning before allocation.
    [[nodiscard]] virtual size_t estimate_storage_bytes(int M, int K) const = 0;
};

/// Factory function to create a quantizer for the given config.
///
/// Returns nullptr if the requested format is not supported on the current hardware.
std::unique_ptr<IQuantizer> create_quantizer(const QuantizerConfig& config);

}  // namespace qlora

#endif  // SUROGATE_SRC_RUNTIME_QLORA_GENERIC_QUANTIZER_H
