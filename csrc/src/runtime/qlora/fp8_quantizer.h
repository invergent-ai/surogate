// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// FP8Quantizer: IQuantizer implementation for per-block FP8 E4M3 quantization.
//
// Wraps the existing per-block FP8 quantization kernels into the generic
// IQuantizer interface. Uses per-block FP32 scales (inverse of quantization
// scale) for on-the-fly BF16 reconstruction.
//
// FP8 E4M3 provides 8-bit representation with 4 exponent and 3 mantissa bits,
// covering a wider dynamic range than INT8 while maintaining reasonable precision.
// Per-block scaling compensates for local variation in weight magnitudes.

#ifndef SUROGATE_SRC_RUNTIME_QLORA_FP8_QUANTIZER_H
#define SUROGATE_SRC_RUNTIME_QLORA_FP8_QUANTIZER_H

#include "runtime/qlora/generic_quantizer.h"

namespace qlora {

/// Per-block FP8 E4M3 quantizer implementing the generic IQuantizer interface.
///
/// Storage layout in QuantizedTensor:
///   - data:   FP8 E4M3 values, dtype=FP8_E4M3, shape=(M, K)
///   - scales: Per-block FP32 inverse scales, dtype=FP32, shape=(num_blocks,)
///             where num_blocks = ceil(M*K / block_size)
///   - meta:   Unused (empty)
///   - meta2:  Unused (empty)
class FP8Quantizer final : public IQuantizer {
public:
    explicit FP8Quantizer(const QuantizerConfig& config);

    void quantize(
        const Tensor& input,
        QuantizedTensor& output,
        cudaStream_t stream) override;

    void dequantize(
        const QuantizedTensor& input,
        Tensor& output,
        cudaStream_t stream) override;

    void allocate_storage(
        int M, int K,
        QuantizedTensor& output,
        TensorAllocator& allocator,
        EAllocationType alloc_type,
        const std::string& name) override;

    [[nodiscard]] QuantFormat format() const override { return QuantFormat::FP8_PER_BLOCK; }
    [[nodiscard]] int block_size() const override { return mBlockSize; }
    [[nodiscard]] size_t estimate_storage_bytes(int M, int K) const override;

private:
    int mBlockSize;
    cudaDeviceProp mDeviceProps;
};

}  // namespace qlora

#endif  // SUROGATE_SRC_RUNTIME_QLORA_FP8_QUANTIZER_H
