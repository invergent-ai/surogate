// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// FP8Quantizer: IQuantizer implementation for per-block FP8 E4M3 quantization.

#include "runtime/qlora/fp8_quantizer.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <fmt/format.h>

#include "kernels/kernels.h"
#include "utilities/utils.h"

namespace qlora {

FP8Quantizer::FP8Quantizer(const QuantizerConfig& config)
    : mBlockSize(config.block_size)
{
    CUDA_CHECK(cudaGetDeviceProperties(&mDeviceProps, config.device_id));
}

void FP8Quantizer::quantize(
    const Tensor& input,
    QuantizedTensor& output,
    cudaStream_t stream) {
    quantize_per_block(
        output.data.get<__nv_fp8_e4m3>(),
        output.scales.get<float>(),
        input.get<nv_bfloat16>(),
        output.M, output.K,
        mBlockSize,
        mDeviceProps,
        stream);
}

void FP8Quantizer::dequantize(
    const QuantizedTensor& input,
    Tensor& output,
    cudaStream_t stream) {
    dequantize_per_block(
        output.get<nv_bfloat16>(),
        input.data.get<__nv_fp8_e4m3>(),
        input.scales.get<float>(),
        input.M, input.K,
        mBlockSize,
        mDeviceProps,
        stream);
}

void FP8Quantizer::allocate_storage(
    int M, int K,
    QuantizedTensor& output,
    TensorAllocator& allocator,
    EAllocationType alloc_type,
    const std::string& name) {
    // The per-block kernel uses 2D tiles of (block_size x block_size).
    // Scale count = ceil(M/block_size) * ceil(K/block_size), NOT ceil(M*K/block_size).
    const long scale_rows = (static_cast<long>(M) + mBlockSize - 1) / mBlockSize;
    const long scale_cols = (static_cast<long>(K) + mBlockSize - 1) / mBlockSize;
    const long num_scales = scale_rows * scale_cols;

    output.M = M;
    output.K = K;
    output.format = QuantFormat::FP8_PER_BLOCK;
    output.block_size = mBlockSize;
    output.double_quant = false;

    // FP8 E4M3 data: 1 byte per value
    output.data = allocator.allocate(
        ETensorDType::FP8_E4M3,
        fmt::format("{}.data", name).c_str(),
        alloc_type,
        {static_cast<long>(M), static_cast<long>(K)});

    // Per-block FP32 scales (2D tile layout, flattened)
    output.scales = allocator.allocate(
        ETensorDType::FP32,
        fmt::format("{}.block_scales", name).c_str(),
        alloc_type,
        {num_scales});
}

size_t FP8Quantizer::estimate_storage_bytes(int M, int K) const {
    const long num_elements = static_cast<long>(M) * K;
    const long scale_rows = (static_cast<long>(M) + mBlockSize - 1) / mBlockSize;
    const long scale_cols = (static_cast<long>(K) + mBlockSize - 1) / mBlockSize;
    const long num_scales = scale_rows * scale_cols;

    size_t total = static_cast<size_t>(num_elements);          // FP8 data (1 byte each)
    total += static_cast<size_t>(num_scales) * sizeof(float);  // FP32 block scales (2D tiles)

    return total;
}

}  // namespace qlora
