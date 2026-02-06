// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// FP4Quantizer: IQuantizer implementation for FP4 E2M1 block quantization.
//
// Wraps the existing FP4 quantization kernels into the generic IQuantizer
// interface. Uses two-level scaling:
//   Level 1: Per-block FP8 E4M3 scales (row-wise, F8_128x4 swizzled)
//   Level 2: Per-tensor global scale (FP32)
//
// The FP4 E2M1 format provides 4-bit representation with 2 exponent and
// 1 mantissa bit. Two values are packed per byte. This format is native
// to Blackwell (SM100+) GPUs with PTX fp4 instructions.
//
// Note: FP4 quantization requires SM100+ (Blackwell) hardware.

#ifndef SUROGATE_SRC_RUNTIME_QLORA_FP4_QUANTIZER_H
#define SUROGATE_SRC_RUNTIME_QLORA_FP4_QUANTIZER_H

#include "runtime/qlora/generic_quantizer.h"

namespace qlora {

/// FP4 E2M1 block quantizer implementing the generic IQuantizer interface.
///
/// Storage layout in QuantizedTensor:
///   - data:         Packed FP4 data (2 per byte), dtype=BYTE, shape=(M*K/2,)
///   - scales:       Per-block FP8 E4M3 scales (row-wise, F8_128x4 swizzled),
///                   dtype=FP8_E4M3, shape=(ceil(M/128)*128, ceil(K/16/4)*4)
///   - meta:         Unused (global scale stored in global_scale field)
///   - meta2:        Unused
///   - global_scale: FP32 global decode scale for two-level reconstruction
///
/// Reconstruction formula:
///   value = fp4_decode(packed) * fp8_block_scale * global_decode_scale
///
/// Where:
///   global_decode_scale = global_amax / (fp8_max * fp4_max)
///   fp8_max = 448.0, fp4_max = 6.0
class FP4Quantizer final : public IQuantizer {
public:
    explicit FP4Quantizer(const QuantizerConfig& config);

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

    [[nodiscard]] QuantFormat format() const override { return QuantFormat::FP4_BLOCK_2D; }
    [[nodiscard]] int block_size() const override { return mBlockSize; }
    [[nodiscard]] size_t estimate_storage_bytes(int M, int K) const override;

private:
    int mBlockSize;
    cudaDeviceProp mDeviceProps;
};

}  // namespace qlora

#endif  // SUROGATE_SRC_RUNTIME_QLORA_FP4_QUANTIZER_H
