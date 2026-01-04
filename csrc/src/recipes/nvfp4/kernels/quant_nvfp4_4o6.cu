// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Based on "Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling"
// arXiv:2512.02010

/**
 * @file quant_nvfp4_4o6.cu
 * @brief Host launcher functions for Four Over Six (4/6) NVFP4 quantization.
 *
 * This file contains the host-side launcher functions that dispatch to the
 * SM100+ kernels in quant_nvfp4_4o6_sm100.cu.
 *
 * The 4/6 algorithm evaluates both scaling to max=6.0 (standard) and max=4.0
 * for each block, selecting the option with lower quantization error.
 *
 * Requires: Blackwell GPU (SM100+) for native FP4 PTX instructions.
 */

#include "kernels/kernel_utils.cuh"
#include "kernels/kernels.h"
#include "recipes/nvfp4/nvfp4_recipe.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"

#include <cuda_bf16.h>
#include <stdexcept>

// Forward declarations of SM100+ kernel launchers (defined in quant_nvfp4_4o6_sm100.cu)
namespace nvfp4_4o6_sm100 {

void quantize_4o6_cutlass_mse(
    uint8_t* out_fp4, uint8_t* block_scales, const float* global_amax,
    const nv_bfloat16* in, int M, int K, int num_scale_cols, cudaStream_t stream);

void quantize_4o6_cutlass_l1(
    uint8_t* out_fp4, uint8_t* block_scales, const float* global_amax,
    const nv_bfloat16* in, int M, int K, int num_scale_cols, cudaStream_t stream);

void quantize_4o6_cutlass_absmax(
    uint8_t* out_fp4, uint8_t* block_scales, const float* global_amax,
    const nv_bfloat16* in, int M, int K, int num_scale_cols, cudaStream_t stream);

void quantize_4o6_stochastic_cutlass_mse(
    uint8_t* out_fp4, uint8_t* block_scales, const float* global_amax,
    const nv_bfloat16* in, int M, int K, int num_scale_cols,
    unsigned int seed, cudaStream_t stream);

void quantize_4o6_stochastic_cutlass_l1(
    uint8_t* out_fp4, uint8_t* block_scales, const float* global_amax,
    const nv_bfloat16* in, int M, int K, int num_scale_cols,
    unsigned int seed, cudaStream_t stream);

void quantize_4o6_stochastic_cutlass_absmax(
    uint8_t* out_fp4, uint8_t* block_scales, const float* global_amax,
    const nv_bfloat16* in, int M, int K, int num_scale_cols,
    unsigned int seed, cudaStream_t stream);

bool is_supported();

}  // namespace nvfp4_4o6_sm100

namespace {
constexpr int kBlockSize = 16;
}

// ============================================================================
// Host Launcher Functions
// ============================================================================

void quantize_nvfp4_4o6_cutlass_auto_scale(
    uint8_t* out_fp4,
    uint8_t* block_scales,
    float* global_amax,
    const nv_bfloat16* in,
    int M, int K,
    recipes::FourOverSixErrorMetric error_metric,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    if (!nvfp4_4o6_sm100::is_supported()) {
        throw std::runtime_error("Four Over Six quantization requires Blackwell GPU (SM100+)");
    }

    // Compute global amax first
    abs_max(global_amax, in, (long)M * K, dp, stream);

    const int num_scale_cols = div_ceil(K, kBlockSize);

    switch (error_metric) {
        case recipes::FourOverSixErrorMetric::MSE:
            nvfp4_4o6_sm100::quantize_4o6_cutlass_mse(
                out_fp4, block_scales, global_amax, in, M, K, num_scale_cols, stream);
            break;
        case recipes::FourOverSixErrorMetric::L1:
            nvfp4_4o6_sm100::quantize_4o6_cutlass_l1(
                out_fp4, block_scales, global_amax, in, M, K, num_scale_cols, stream);
            break;
        case recipes::FourOverSixErrorMetric::AbsMax:
            nvfp4_4o6_sm100::quantize_4o6_cutlass_absmax(
                out_fp4, block_scales, global_amax, in, M, K, num_scale_cols, stream);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void quantize_nvfp4_4o6_stochastic_cutlass_auto_scale(
    uint8_t* out_fp4,
    uint8_t* block_scales,
    float* global_amax,
    const nv_bfloat16* in,
    int M, int K,
    recipes::FourOverSixErrorMetric error_metric,
    unsigned int seed,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    if (!nvfp4_4o6_sm100::is_supported()) {
        throw std::runtime_error("Four Over Six quantization requires Blackwell GPU (SM100+)");
    }

    // Compute global amax first
    abs_max(global_amax, in, (long)M * K, dp, stream);

    const int num_scale_cols = div_ceil(K, kBlockSize);

    switch (error_metric) {
        case recipes::FourOverSixErrorMetric::MSE:
            nvfp4_4o6_sm100::quantize_4o6_stochastic_cutlass_mse(
                out_fp4, block_scales, global_amax, in, M, K, num_scale_cols, seed, stream);
            break;
        case recipes::FourOverSixErrorMetric::L1:
            nvfp4_4o6_sm100::quantize_4o6_stochastic_cutlass_l1(
                out_fp4, block_scales, global_amax, in, M, K, num_scale_cols, seed, stream);
            break;
        case recipes::FourOverSixErrorMetric::AbsMax:
            nvfp4_4o6_sm100::quantize_4o6_stochastic_cutlass_absmax(
                out_fp4, block_scales, global_amax, in, M, K, num_scale_cols, seed, stream);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Tensor-based Wrappers
// ============================================================================

void quantize_nvfp4_4o6_cutlass(
    Tensor& out_fp4,
    Tensor& block_scales,
    Tensor& global_amax,
    const Tensor& in,
    recipes::FourOverSixErrorMetric error_metric,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    if (in.DType != ETensorDType::BF16) {
        throw std::runtime_error("quantize_nvfp4_4o6_cutlass: input must be BF16");
    }
    if (out_fp4.DType != ETensorDType::BYTE && out_fp4.DType != ETensorDType::FP4_E2M1) {
        throw std::runtime_error("quantize_nvfp4_4o6_cutlass: output must be BYTE or FP4_E2M1");
    }
    if (in.Rank != 2 || out_fp4.Rank != 2) {
        throw std::runtime_error("quantize_nvfp4_4o6_cutlass: tensors must be 2D");
    }

    const int M = in.Sizes[0];
    const int K = in.Sizes[1];

    quantize_nvfp4_4o6_cutlass_auto_scale(
        out_fp4.get<uint8_t>(),
        block_scales.get<uint8_t>(),
        global_amax.get<float>(),
        in.get<nv_bfloat16>(),
        M, K, error_metric, dp, stream);
}

void quantize_nvfp4_4o6_stochastic_cutlass(
    Tensor& out_fp4,
    Tensor& block_scales,
    Tensor& global_amax,
    const Tensor& in,
    recipes::FourOverSixErrorMetric error_metric,
    unsigned int seed,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    if (in.DType != ETensorDType::BF16) {
        throw std::runtime_error("quantize_nvfp4_4o6_stochastic_cutlass: input must be BF16");
    }
    if (out_fp4.DType != ETensorDType::BYTE && out_fp4.DType != ETensorDType::FP4_E2M1) {
        throw std::runtime_error("quantize_nvfp4_4o6_stochastic_cutlass: output must be BYTE or FP4_E2M1");
    }
    if (in.Rank != 2 || out_fp4.Rank != 2) {
        throw std::runtime_error("quantize_nvfp4_4o6_stochastic_cutlass: tensors must be 2D");
    }

    const int M = in.Sizes[0];
    const int K = in.Sizes[1];

    quantize_nvfp4_4o6_stochastic_cutlass_auto_scale(
        out_fp4.get<uint8_t>(),
        block_scales.get<uint8_t>(),
        global_amax.get<float>(),
        in.get<nv_bfloat16>(),
        M, K, error_metric, seed, dp, stream);
}
