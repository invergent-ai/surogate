// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// MXFP4Quantizer: Dequant-only IQuantizer for HF pre-quantized MXFP4 models.
//
// Implements dequantization of the microscaling FP4 format:
//   - Packed FP4 E2M1 data (2 values per byte), dtype=BYTE
//   - E8M0 shared exponents per 32-element block, dtype=BYTE
//
// Reconstruction formula:
//   value = fp4_decode(nibble) * 2^(e8m0_exponent - 127)
//
// Quantization is NOT supported — these weights are loaded pre-quantized
// from HuggingFace safetensors and remain frozen during LoRA fine-tuning.

#ifndef SUROGATE_SRC_RUNTIME_QLORA_MXFP4_QUANTIZER_H
#define SUROGATE_SRC_RUNTIME_QLORA_MXFP4_QUANTIZER_H

#include "runtime/qlora/generic_quantizer.h"

namespace qlora {

/// MXFP4 dequant-only quantizer implementing the generic IQuantizer interface.
///
/// Storage layout in QuantizedTensor:
///   - data:         Packed FP4 E2M1 data (2 per byte), dtype=BYTE, shape=(M*K/2,)
///   - scales:       E8M0 shared exponents (1 per 32 elements), dtype=BYTE, shape=(M*K/32,)
///   - meta:         Unused
///   - meta2:        Unused
///   - global_scale: Unused (E8M0 exponents encode full scale)
class MXFP4Quantizer final : public IQuantizer {
public:
    explicit MXFP4Quantizer(const QuantizerConfig& config);

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

    [[nodiscard]] QuantFormat format() const override { return QuantFormat::HF_MXFP4; }
    [[nodiscard]] int block_size() const override { return 32; }
    [[nodiscard]] size_t estimate_storage_bytes(int M, int K) const override;

private:
    cudaDeviceProp mDeviceProps;
};

}  // namespace qlora

#endif  // SUROGATE_SRC_RUNTIME_QLORA_MXFP4_QUANTIZER_H
