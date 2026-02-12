// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// BnBQuantizer: IQuantizer implementation for BitsAndBytes NF4 quantization.
//
// Wraps the existing NF4 quantization kernels into the generic IQuantizer
// interface. Supports per-block absmax scaling and optional double quantization
// (INT8 quantization of absmax values for additional memory savings).
//
// NF4 uses 16 asymmetric bins derived from a normal distribution, which
// better represents neural network weight distributions than uniform FP4.

#ifndef SUROGATE_SRC_RUNTIME_QLORA_BNB_QUANTIZER_H
#define SUROGATE_SRC_RUNTIME_QLORA_BNB_QUANTIZER_H

#include "runtime/qlora/generic_quantizer.h"

namespace qlora {

/// BitsAndBytes NF4 quantizer implementing the generic IQuantizer interface.
///
/// Storage layout in QuantizedTensor:
///   - data:   Packed 4-bit NF4 values (2 per byte), dtype=BYTE, shape=(M*K/2,)
///   - scales: Per-block absmax values.
///             Without double quant: dtype=FP32, shape=(num_blocks,)
///             With double quant:    dtype=BYTE (INT8), shape=(num_blocks,)
///   - meta:   Double quant scale (FP32), shape=(num_groups,). Empty if !double_quant.
///   - meta2:  Double quant offset (FP32), shape=(num_groups,). Empty if !double_quant.
class BnBQuantizer final : public IQuantizer {
public:
    explicit BnBQuantizer(const QuantizerConfig& config);
    ~BnBQuantizer() override;

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

    [[nodiscard]] QuantFormat format() const override { return QuantFormat::BNB_NF4; }
    [[nodiscard]] int block_size() const override { return mBlockSize; }
    [[nodiscard]] size_t estimate_storage_bytes(int M, int K) const override;

private:
    void ensure_absmax_buffer(long num_blocks);

    int mBlockSize;
    bool mDoubleQuant;
    int mDoubleQuantGroupSize;
    cudaDeviceProp mDeviceProps;
    float* mAbsmaxBuffer = nullptr;
    long mAbsmaxCapacity = 0;
};

}  // namespace qlora

#endif  // SUROGATE_SRC_RUNTIME_QLORA_BNB_QUANTIZER_H
