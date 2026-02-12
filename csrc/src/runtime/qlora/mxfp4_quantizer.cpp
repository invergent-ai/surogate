// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// MXFP4Quantizer: Dequant-only IQuantizer for HF pre-quantized MXFP4 models.

#include "runtime/qlora/mxfp4_quantizer.h"

#include <cuda_bf16.h>
#include <fmt/format.h>

#include "kernels/kernels.h"
#include "utilities/utils.h"

namespace qlora {

MXFP4Quantizer::MXFP4Quantizer(const QuantizerConfig& config) {
    CUDA_CHECK(cudaGetDeviceProperties(&mDeviceProps, config.device_id));
}

void MXFP4Quantizer::quantize(
    const Tensor& /*input*/,
    QuantizedTensor& /*output*/,
    cudaStream_t /*stream*/) {
    throw std::runtime_error(
        "MXFP4Quantizer::quantize() is not supported. "
        "MXFP4 is a read-only format for pre-quantized HF models.");
}

void MXFP4Quantizer::dequantize(
    const QuantizedTensor& input,
    Tensor& output,
    cudaStream_t stream) {
    dequantize_mxfp4(
        output.get<nv_bfloat16>(),
        input.data.get<uint8_t>(),
        input.scales.get<uint8_t>(),
        input.M, input.K,
        mDeviceProps,
        stream);
}

void MXFP4Quantizer::allocate_storage(
    int M, int K,
    QuantizedTensor& output,
    TensorAllocator& allocator,
    EAllocationType alloc_type,
    const std::string& name) {
    const long packed_bytes = static_cast<long>(M) * K / 2;
    const long num_scales = static_cast<long>(M) * K / 32;

    output.M = M;
    output.K = K;
    output.format = QuantFormat::HF_MXFP4;
    output.block_size = 32;
    output.double_quant = false;
    output.global_scale = 1.0f;

    // Packed FP4 E2M1 data: 2 values per byte
    output.data = allocator.allocate(
        ETensorDType::BYTE,
        fmt::format("{}.data", name).c_str(),
        alloc_type,
        {packed_bytes});

    // E8M0 shared exponents: one uint8 per 32-element block
    output.scales = allocator.allocate(
        ETensorDType::BYTE,
        fmt::format("{}.e8m0_scales", name).c_str(),
        alloc_type,
        {num_scales});
}

size_t MXFP4Quantizer::estimate_storage_bytes(int M, int K) const {
    const size_t data_bytes = static_cast<size_t>(M) * K / 2;    // Packed FP4 (2 per byte)
    const size_t scale_bytes = static_cast<size_t>(M) * K / 32;  // E8M0 (1 byte per 32 elements)

    return data_bytes + scale_bytes;
}

}  // namespace qlora
