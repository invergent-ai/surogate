// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "kernels/kernels.h"
#include "utilities/utils.h"
#include "test_utils.h"

using namespace testing_utils;

namespace {

// Compute relative error between two vectors
float compute_relative_error(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1.0f;

    float max_err = 0.0f;
    float max_val = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_err = std::max(max_err, std::abs(a[i] - b[i]));
        max_val = std::max(max_val, std::max(std::abs(a[i]), std::abs(b[i])));
    }
    return (max_val > 0.0f) ? (max_err / max_val) : max_err;
}

// Compute mean absolute error
float compute_mae(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1.0f;

    float total_err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        total_err += std::abs(a[i] - b[i]);
    }
    return total_err / static_cast<float>(a.size());
}

} // anonymous namespace


TEST_CASE("per-block quantization roundtrip", "[qlora][quant]") {
    // Test parameters
    const int M = 256;
    const int K = 256;
    const int block_size = 128;

    // Generate random input data
    auto input_f32 = uniform_host(M * K, -1.0f, 1.0f, 42);
    auto input_bf16 = to_bf16(input_f32);

    // Get device properties
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    // Allocate device memory
    thrust::device_vector<nv_bfloat16> d_input(input_bf16.begin(), input_bf16.end());
    thrust::device_vector<__nv_fp8_e4m3> d_fp8(M * K);
    thrust::device_vector<float> d_scales((M / block_size) * (K / block_size));
    thrust::device_vector<nv_bfloat16> d_output(M * K);

    // Run quantization kernel
    quantize_per_block(
        thrust::raw_pointer_cast(d_fp8.data()),
        thrust::raw_pointer_cast(d_scales.data()),
        thrust::raw_pointer_cast(d_input.data()),
        M, K, block_size, props, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run dequantization kernel
    dequantize_per_block(
        thrust::raw_pointer_cast(d_output.data()),
        thrust::raw_pointer_cast(d_fp8.data()),
        thrust::raw_pointer_cast(d_scales.data()),
        M, K, block_size, props, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    std::vector<nv_bfloat16> h_output(M * K);
    thrust::copy(d_output.begin(), d_output.end(), h_output.begin());

    // Convert BF16 to float for comparison
    std::vector<float> output_f32(M * K);
    for (int i = 0; i < M * K; ++i) {
        output_f32[i] = static_cast<float>(h_output[i]);
    }

    // Compute error metrics
    float rel_err = compute_relative_error(input_f32, output_f32);
    float mae = compute_mae(input_f32, output_f32);

    INFO("Roundtrip relative error: " << rel_err);
    INFO("Roundtrip MAE: " << mae);

    // FP8 E4M3 has ~3-4% relative error for random data
    REQUIRE(rel_err < 0.15f);  // Allow 15% relative error for FP8
    REQUIRE(mae < 0.05f);      // MAE should be small for normalized input
}


TEST_CASE("per-block quantization handles large values", "[qlora][quant]") {
    // Test with values that require scaling
    const int M = 128;
    const int K = 128;
    const int block_size = 128;

    // Generate input with large values (beyond FP8 range)
    auto input_f32 = uniform_host(M * K, -1000.0f, 1000.0f, 123);
    auto input_bf16 = to_bf16(input_f32);

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    thrust::device_vector<nv_bfloat16> d_input(input_bf16.begin(), input_bf16.end());
    thrust::device_vector<__nv_fp8_e4m3> d_fp8(M * K);
    thrust::device_vector<float> d_scales(1);  // Single block
    thrust::device_vector<nv_bfloat16> d_output(M * K);

    // Quantize
    quantize_per_block(
        thrust::raw_pointer_cast(d_fp8.data()),
        thrust::raw_pointer_cast(d_scales.data()),
        thrust::raw_pointer_cast(d_input.data()),
        M, K, block_size, props, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check that scale is reasonable
    std::vector<float> h_scales(1);
    thrust::copy(d_scales.begin(), d_scales.end(), h_scales.begin());

    INFO("Block scale: " << h_scales[0]);
    REQUIRE(h_scales[0] > 0.0f);
    REQUIRE(h_scales[0] > 1.0f);  // Should be > 1 for large input values

    // Dequantize
    dequantize_per_block(
        thrust::raw_pointer_cast(d_output.data()),
        thrust::raw_pointer_cast(d_fp8.data()),
        thrust::raw_pointer_cast(d_scales.data()),
        M, K, block_size, props, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output
    std::vector<nv_bfloat16> h_output(M * K);
    thrust::copy(d_output.begin(), d_output.end(), h_output.begin());

    std::vector<float> output_f32(M * K);
    for (int i = 0; i < M * K; ++i) {
        output_f32[i] = static_cast<float>(h_output[i]);
    }

    float rel_err = compute_relative_error(input_f32, output_f32);
    INFO("Large values roundtrip relative error: " << rel_err);
    REQUIRE(rel_err < 0.15f);
}


TEST_CASE("per-block quantization with multiple blocks", "[qlora][quant]") {
    // Test with multiple blocks
    const int M = 512;
    const int K = 512;
    const int block_size = 128;
    const int scale_rows = M / block_size;
    const int scale_cols = K / block_size;

    // Generate input with different magnitude ranges per block
    std::vector<float> input_f32(M * K);
    for (int br = 0; br < scale_rows; ++br) {
        for (int bc = 0; bc < scale_cols; ++bc) {
            float block_scale = 1.0f + (br * scale_cols + bc) * 10.0f;
            for (int r = br * block_size; r < (br + 1) * block_size; ++r) {
                for (int c = bc * block_size; c < (bc + 1) * block_size; ++c) {
                    // Fill with values scaled by block index
                    input_f32[r * K + c] = block_scale * (((r + c) % 11) - 5.0f) / 5.0f;
                }
            }
        }
    }
    auto input_bf16 = to_bf16(input_f32);

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    thrust::device_vector<nv_bfloat16> d_input(input_bf16.begin(), input_bf16.end());
    thrust::device_vector<__nv_fp8_e4m3> d_fp8(M * K);
    thrust::device_vector<float> d_scales(scale_rows * scale_cols);
    thrust::device_vector<nv_bfloat16> d_output(M * K);

    // Quantize
    quantize_per_block(
        thrust::raw_pointer_cast(d_fp8.data()),
        thrust::raw_pointer_cast(d_scales.data()),
        thrust::raw_pointer_cast(d_input.data()),
        M, K, block_size, props, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify scales increase with block index
    std::vector<float> h_scales(scale_rows * scale_cols);
    thrust::copy(d_scales.begin(), d_scales.end(), h_scales.begin());

    INFO("Number of blocks: " << scale_rows << "x" << scale_cols);
    REQUIRE(h_scales[0] > 0.0f);
    REQUIRE(h_scales.back() > h_scales[0]);  // Later blocks should have larger scales

    // Dequantize
    dequantize_per_block(
        thrust::raw_pointer_cast(d_output.data()),
        thrust::raw_pointer_cast(d_fp8.data()),
        thrust::raw_pointer_cast(d_scales.data()),
        M, K, block_size, props, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output
    std::vector<nv_bfloat16> h_output(M * K);
    thrust::copy(d_output.begin(), d_output.end(), h_output.begin());

    std::vector<float> output_f32(M * K);
    for (size_t i = 0; i < h_output.size(); ++i) {
        output_f32[i] = static_cast<float>(h_output[i]);
    }

    float rel_err = compute_relative_error(input_f32, output_f32);
    INFO("Multi-block roundtrip relative error: " << rel_err);
    REQUIRE(rel_err < 0.15f);
}


TEST_CASE("per-block quantization with non-aligned dimensions", "[qlora][quant]") {
    // Test with dimensions not evenly divisible by block size
    const int M = 300;  // Not divisible by 128
    const int K = 400;  // Not divisible by 128
    const int block_size = 128;
    const int scale_rows = (M + block_size - 1) / block_size;
    const int scale_cols = (K + block_size - 1) / block_size;

    auto input_f32 = uniform_host(M * K, -2.0f, 2.0f, 456);
    auto input_bf16 = to_bf16(input_f32);

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    thrust::device_vector<nv_bfloat16> d_input(input_bf16.begin(), input_bf16.end());
    thrust::device_vector<__nv_fp8_e4m3> d_fp8(M * K);
    thrust::device_vector<float> d_scales(scale_rows * scale_cols);
    thrust::device_vector<nv_bfloat16> d_output(M * K);

    // Quantize
    quantize_per_block(
        thrust::raw_pointer_cast(d_fp8.data()),
        thrust::raw_pointer_cast(d_scales.data()),
        thrust::raw_pointer_cast(d_input.data()),
        M, K, block_size, props, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Dequantize
    dequantize_per_block(
        thrust::raw_pointer_cast(d_output.data()),
        thrust::raw_pointer_cast(d_fp8.data()),
        thrust::raw_pointer_cast(d_scales.data()),
        M, K, block_size, props, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output
    std::vector<nv_bfloat16> h_output(M * K);
    thrust::copy(d_output.begin(), d_output.end(), h_output.begin());

    std::vector<float> output_f32(M * K);
    for (int i = 0; i < M * K; ++i) {
        output_f32[i] = static_cast<float>(h_output[i]);
    }

    float rel_err = compute_relative_error(input_f32, output_f32);
    INFO("Non-aligned dimensions roundtrip relative error: " << rel_err);
    REQUIRE(rel_err < 0.15f);
}


TEST_CASE("per-block quantization different block sizes", "[qlora][quant]") {
    const int M = 256;
    const int K = 256;

    auto input_f32 = uniform_host(M * K, -1.0f, 1.0f, 789);
    auto input_bf16 = to_bf16(input_f32);

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    for (int block_size : {64, 128, 256}) {
        SECTION("block_size=" + std::to_string(block_size)) {
            const int scale_rows = (M + block_size - 1) / block_size;
            const int scale_cols = (K + block_size - 1) / block_size;

            thrust::device_vector<nv_bfloat16> d_input(input_bf16.begin(), input_bf16.end());
            thrust::device_vector<__nv_fp8_e4m3> d_fp8(M * K);
            thrust::device_vector<float> d_scales(scale_rows * scale_cols);
            thrust::device_vector<nv_bfloat16> d_output(M * K);

            // Quantize
            quantize_per_block(
                thrust::raw_pointer_cast(d_fp8.data()),
                thrust::raw_pointer_cast(d_scales.data()),
                thrust::raw_pointer_cast(d_input.data()),
                M, K, block_size, props, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Dequantize
            dequantize_per_block(
                thrust::raw_pointer_cast(d_output.data()),
                thrust::raw_pointer_cast(d_fp8.data()),
                thrust::raw_pointer_cast(d_scales.data()),
                M, K, block_size, props, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Copy output
            std::vector<nv_bfloat16> h_output(M * K);
            thrust::copy(d_output.begin(), d_output.end(), h_output.begin());

            std::vector<float> output_f32(M * K);
            for (int i = 0; i < M * K; ++i) {
                output_f32[i] = static_cast<float>(h_output[i]);
            }

            float rel_err = compute_relative_error(input_f32, output_f32);
            INFO("Block size " << block_size << " relative error: " << rel_err);
            REQUIRE(rel_err < 0.15f);
        }
    }
}


TEST_CASE("per-block quantization preserves zero", "[qlora][quant]") {
    // Test that zeros are preserved
    const int M = 128;
    const int K = 128;
    const int block_size = 128;

    std::vector<float> input_f32(M * K, 0.0f);
    auto input_bf16 = to_bf16(input_f32);

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    thrust::device_vector<nv_bfloat16> d_input(input_bf16.begin(), input_bf16.end());
    thrust::device_vector<__nv_fp8_e4m3> d_fp8(M * K);
    thrust::device_vector<float> d_scales(1);
    thrust::device_vector<nv_bfloat16> d_output(M * K);

    // Quantize
    quantize_per_block(
        thrust::raw_pointer_cast(d_fp8.data()),
        thrust::raw_pointer_cast(d_scales.data()),
        thrust::raw_pointer_cast(d_input.data()),
        M, K, block_size, props, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Dequantize
    dequantize_per_block(
        thrust::raw_pointer_cast(d_output.data()),
        thrust::raw_pointer_cast(d_fp8.data()),
        thrust::raw_pointer_cast(d_scales.data()),
        M, K, block_size, props, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output
    std::vector<nv_bfloat16> h_output(M * K);
    thrust::copy(d_output.begin(), d_output.end(), h_output.begin());

    // Check all outputs are zero
    for (int i = 0; i < M * K; ++i) {
        float val = static_cast<float>(h_output[i]);
        REQUIRE(val == 0.0f);
    }
}
