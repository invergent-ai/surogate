// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// BnBQuantizer: IQuantizer implementation for BitsAndBytes NF4 quantization.

#include "runtime/qlora/bnb_quantizer.h"

#include <cuda_bf16.h>
#include <fmt/format.h>

#include "kernels/kernels.h"
#include "utilities/utils.h"

namespace qlora {

BnBQuantizer::BnBQuantizer(const QuantizerConfig& config)
    : mBlockSize(config.block_size)
    , mDoubleQuant(config.double_quant)
    , mDoubleQuantGroupSize(config.double_quant_group_size) {
    CUDA_CHECK(cudaGetDeviceProperties(&mDeviceProps, config.device_id));
}

void BnBQuantizer::quantize(
    const Tensor& input,
    QuantizedTensor& output,
    cudaStream_t stream) {
    const int M = output.M;
    const int K = output.K;
    const long num_elements = static_cast<long>(M) * K;
    const long num_blocks = (num_elements + mBlockSize - 1) / mBlockSize;

    if (mDoubleQuant) {
        // Two-step quantization:
        // 1. Quantize BF16 → NF4 with FP32 absmax into a temporary buffer on the QuantizedTensor's meta2
        //    (we reuse the scales tensor as a temporary FP32 absmax buffer, then overwrite with INT8)
        //    Actually, we need a temporary FP32 absmax buffer. Since double_quant stores INT8 in scales,
        //    we use meta2 as scratch for the FP32 absmax before double-quantizing them.

        // Allocate or use scratch. meta2 was allocated as FP32(num_groups) but we need FP32(num_blocks)
        // for the intermediate absmax. We'll use the output.meta2 tensor temporarily if it's large enough,
        // otherwise the caller should provide scratch. For simplicity and correctness, we use the
        // approach from the existing codebase: quantize with FP32 absmax into a device buffer,
        // then double-quantize the absmax in-place.

        // We need a temporary FP32 buffer of size num_blocks for the intermediate absmax.
        // The existing codebase allocates a dedicated mAbsmaxBuffer for this purpose.
        // Since our interface doesn't expose scratch buffers, we'll allocate a small temp buffer on stream.
        float* temp_absmax = nullptr;
        CUDA_CHECK(cudaMallocAsync(&temp_absmax, num_blocks * sizeof(float), stream));

        // Step 1: Quantize BF16 → NF4 with FP32 absmax
        quantize_bnb_nf4(
            output.data.get<unsigned char>(),
            temp_absmax,
            input.get<nv_bfloat16>(),
            M, K,
            mBlockSize,
            mDeviceProps,
            stream);

        // Step 2: Double-quantize absmax FP32 → INT8 with per-group scale/offset
        quantize_absmax_double(
            output.scales.get<unsigned char>(),  // INT8 quantized absmax
            output.meta.get<float>(),            // Per-group scale
            output.meta2.get<float>(),           // Per-group offset
            temp_absmax,
            static_cast<int>(num_blocks),
            mDoubleQuantGroupSize,
            mDeviceProps,
            stream);

        CUDA_CHECK(cudaFreeAsync(temp_absmax, stream));
    } else {
        // Single-step: Quantize BF16 → NF4 with FP32 absmax directly
        quantize_bnb_nf4(
            output.data.get<unsigned char>(),
            output.scales.get<float>(),
            input.get<nv_bfloat16>(),
            M, K,
            mBlockSize,
            mDeviceProps,
            stream);
    }
}

void BnBQuantizer::dequantize(
    const QuantizedTensor& input,
    Tensor& output,
    cudaStream_t stream) {
    const int M = input.M;
    const int K = input.K;

    if (input.double_quant) {
        // Fused dequantization: double-quant absmax + NF4 → BF16
        dequantize_bnb_nf4_double(
            output.get<nv_bfloat16>(),
            input.data.get<unsigned char>(),
            input.scales.get<unsigned char>(),   // INT8 quantized absmax
            input.meta.get<float>(),             // Per-group scale
            input.meta2.get<float>(),            // Per-group offset
            M, K,
            mBlockSize,
            input.double_quant_group_size,
            mDeviceProps,
            stream);
    } else {
        // Direct dequantization: FP32 absmax + NF4 → BF16
        dequantize_bnb_nf4(
            output.get<nv_bfloat16>(),
            input.data.get<unsigned char>(),
            input.scales.get<float>(),
            M, K,
            mBlockSize,
            mDeviceProps,
            stream);
    }
}

void BnBQuantizer::allocate_storage(
    int M, int K,
    QuantizedTensor& output,
    TensorAllocator& allocator,
    EAllocationType alloc_type,
    const std::string& name) {
    const long num_elements = static_cast<long>(M) * K;
    const long num_blocks = (num_elements + mBlockSize - 1) / mBlockSize;
    const long packed_bytes = (num_elements + 1) / 2;  // 2 NF4 values per byte

    output.M = M;
    output.K = K;
    output.format = QuantFormat::BNB_NF4;
    output.block_size = mBlockSize;
    output.double_quant = mDoubleQuant;
    output.double_quant_group_size = mDoubleQuantGroupSize;

    // Packed NF4 data: 4 bits per value, 2 values per byte
    output.data = allocator.allocate(
        ETensorDType::BYTE,
        fmt::format("{}.data", name).c_str(),
        alloc_type,
        {packed_bytes});

    if (mDoubleQuant) {
        const long num_groups = (num_blocks + mDoubleQuantGroupSize - 1) / mDoubleQuantGroupSize;

        // Scales: INT8-quantized absmax (one byte per block)
        output.scales = allocator.allocate(
            ETensorDType::BYTE,
            fmt::format("{}.absmax_q", name).c_str(),
            alloc_type,
            {num_blocks});

        // Meta: Per-group FP32 scale for double quantization
        output.meta = allocator.allocate(
            ETensorDType::FP32,
            fmt::format("{}.absmax_scale", name).c_str(),
            alloc_type,
            {num_groups});

        // Meta2: Per-group FP32 offset for double quantization
        output.meta2 = allocator.allocate(
            ETensorDType::FP32,
            fmt::format("{}.absmax_offset", name).c_str(),
            alloc_type,
            {num_groups});
    } else {
        // Scales: FP32 absmax (one per block)
        output.scales = allocator.allocate(
            ETensorDType::FP32,
            fmt::format("{}.absmax", name).c_str(),
            alloc_type,
            {num_blocks});
    }
}

size_t BnBQuantizer::estimate_storage_bytes(int M, int K) const {
    const long num_elements = static_cast<long>(M) * K;
    const long num_blocks = (num_elements + mBlockSize - 1) / mBlockSize;
    const long packed_bytes = (num_elements + 1) / 2;

    size_t total = static_cast<size_t>(packed_bytes);  // NF4 data

    if (mDoubleQuant) {
        const long num_groups = (num_blocks + mDoubleQuantGroupSize - 1) / mDoubleQuantGroupSize;
        total += static_cast<size_t>(num_blocks);        // INT8 absmax
        total += static_cast<size_t>(num_groups) * 4;    // FP32 scale
        total += static_cast<size_t>(num_groups) * 4;    // FP32 offset
    } else {
        total += static_cast<size_t>(num_blocks) * 4;    // FP32 absmax
    }

    return total;
}

}  // namespace qlora
