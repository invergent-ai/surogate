// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// FP4Quantizer: IQuantizer implementation for FP4 E2M1 block quantization.

#include "runtime/qlora/fp4_quantizer.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <fmt/format.h>

#include "kernels/kernels.h"
#include "runtime/qlora/fp4_block_quantized_tensor.h"
#include "utilities/utils.h"

using modules::FP4BlockScaleConfig;

namespace qlora {

FP4Quantizer::FP4Quantizer(const QuantizerConfig& config)
    : mBlockSize(FP4BlockScaleConfig::BLOCK_SIZE)  // Always 16 for NVFP4
{
    (void)config.block_size;  // FP4 block size is fixed at 16
    CUDA_CHECK(cudaGetDeviceProperties(&mDeviceProps, config.device_id));
}

void FP4Quantizer::quantize(
    const Tensor& input,
    QuantizedTensor& output,
    cudaStream_t stream) {
    const int M = output.M;
    const int K = output.K;

    // Use auto-scale quantization: computes global_amax and derives scales automatically.
    // The global_amax is written to device memory pointed to by output.meta.Data.
    // After quantization, we read back global_amax to compute global_decode_scale.
    quantize_fp4_block_auto_scale(
        output.data.get<uint8_t>(),
        output.scales.get<__nv_fp8_e4m3>(),
        output.meta.get<float>(),  // global_amax output (single float on device)
        input.get<nv_bfloat16>(),
        M, K,
        mDeviceProps,
        stream);

    // Read back global_amax to compute global_decode_scale
    // global_decode_scale = global_amax / (fp8_max * fp4_max)
    // fp8_max = 448.0, fp4_max = 6.0
    float global_amax = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&global_amax, output.meta.Data,
                                sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    constexpr float FP8_MAX = 448.0f;
    constexpr float FP4_MAX = 6.0f;
    output.global_scale = (global_amax > 0.0f)
        ? global_amax / (FP8_MAX * FP4_MAX)
        : 1.0f;
}

void FP4Quantizer::dequantize(
    const QuantizedTensor& input,
    Tensor& output,
    cudaStream_t stream) {
    dequantize_fp4_block(
        output.get<nv_bfloat16>(),
        input.data.get<uint8_t>(),
        input.scales.get<__nv_fp8_e4m3>(),
        input.global_scale,
        input.M, input.K,
        mDeviceProps,
        stream);
}

void FP4Quantizer::allocate_storage(
    int M, int K,
    QuantizedTensor& output,
    TensorAllocator& allocator,
    EAllocationType alloc_type,
    const std::string& name) {
    const long packed_bytes = static_cast<long>(FP4BlockScaleConfig::packed_data_bytes(M, K));
    auto [scale_rows, scale_cols] = FP4BlockScaleConfig::scale_dims(M, K);

    output.M = M;
    output.K = K;
    output.format = QuantFormat::FP4_BLOCK_2D;
    output.block_size = mBlockSize;
    output.double_quant = false;
    output.global_scale = 1.0f;

    // Packed FP4 data: 2 values per byte
    output.data = allocator.allocate(
        ETensorDType::BYTE,
        fmt::format("{}.data", name).c_str(),
        alloc_type,
        {packed_bytes});

    // Per-block FP8 E4M3 scales (row-wise, F8_128x4 swizzled)
    output.scales = allocator.allocate(
        ETensorDType::FP8_E4M3,
        fmt::format("{}.block_scales", name).c_str(),
        alloc_type,
        {static_cast<long>(scale_rows), static_cast<long>(scale_cols)});

    // Global amax: single FP32 value on device (used during quantization)
    output.meta = allocator.allocate(
        ETensorDType::FP32,
        fmt::format("{}.global_amax", name).c_str(),
        EAllocationType::ON_DEVICE,  // Always on device for kernel output
        {1L});
}

size_t FP4Quantizer::estimate_storage_bytes(int M, int K) const {
    size_t data_bytes = FP4BlockScaleConfig::packed_data_bytes(M, K);
    size_t scale_bytes = FP4BlockScaleConfig::scale_bytes(M, K);
    size_t amax_bytes = sizeof(float);  // Single global amax

    return data_bytes + scale_bytes + amax_bytes;
}

}  // namespace qlora
