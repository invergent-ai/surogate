// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for FP4 E2M1 (NVFP4) quantization and Hadamard transform kernels.

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

// cuDNN handle helpers are implemented in src/kernels/cudnn_att.cpp and linked via surogate-common.
cudnnHandle_t create_cudnn_handle();
void destroy_cudnn_handle(cudnnHandle_t handle) noexcept;

namespace {

// FP4 E2M1 quantization levels (8 positive values, 8 negative including zeros)
// Values: ±{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
constexpr float FP4_E2M1_VALUES[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
constexpr int FP4_NUM_VALUES = 8;

// Software FP4 E2M1 quantization (for CPU reference)
float quantize_fp4_e2m1_cpu(float val) {
    float sign = (val < 0.0f) ? -1.0f : 1.0f;
    float abs_val = std::abs(val);

    // Find nearest FP4 value
    float best = 0.0f;
    float best_dist = std::abs(abs_val - 0.0f);
    for (int i = 1; i < FP4_NUM_VALUES; ++i) {
        float dist = std::abs(abs_val - FP4_E2M1_VALUES[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best = FP4_E2M1_VALUES[i];
        }
    }
    return sign * best;
}

// CPU reference for 16-point Hadamard transform
void hadamard16_cpu(float x[16]) {
    // Stage 1
    for (int i = 0; i < 16; i += 2) {
        float a = x[i];
        float b = x[i + 1];
        x[i] = a + b;
        x[i + 1] = a - b;
    }
    // Stage 2
    for (int i = 0; i < 16; i += 4) {
        float a0 = x[i], a1 = x[i + 1];
        float b0 = x[i + 2], b1 = x[i + 3];
        x[i] = a0 + b0;
        x[i + 1] = a1 + b1;
        x[i + 2] = a0 - b0;
        x[i + 3] = a1 - b1;
    }
    // Stage 3
    for (int i = 0; i < 16; i += 8) {
        float a[4], b[4];
        for (int j = 0; j < 4; ++j) {
            a[j] = x[i + j];
            b[j] = x[i + 4 + j];
        }
        for (int j = 0; j < 4; ++j) {
            x[i + j] = a[j] + b[j];
            x[i + 4 + j] = a[j] - b[j];
        }
    }
    // Stage 4
    float a[8], b[8];
    for (int i = 0; i < 8; ++i) {
        a[i] = x[i];
        b[i] = x[i + 8];
    }
    for (int i = 0; i < 8; ++i) {
        x[i] = a[i] + b[i];
        x[i + 8] = a[i] - b[i];
    }
}

// Random sign based on position and seed
float random_sign_cpu(long idx, unsigned int seed) {
    unsigned int hash = static_cast<unsigned int>(idx) ^ seed;
    hash = hash * 2654435761u;
    hash ^= (hash >> 16);
    hash *= 0x85ebca6bu;
    hash ^= (hash >> 13);
    return (hash & 1) ? 1.0f : -1.0f;
}

// CPU reference for Hadamard transform forward
void hadamard_transform_forward_cpu(
    std::vector<float>& out,
    const std::vector<float>& in,
    int M, int K, unsigned int seed)
{
    constexpr float HADAMARD_SCALE = 0.25f;
    out.resize(M * K);

    for (int row = 0; row < M; ++row) {
        for (int block_start = 0; block_start < K; block_start += 16) {
            float values[16];

            // Load and apply random sign
            for (int i = 0; i < 16 && (block_start + i) < K; ++i) {
                long col = block_start + i;
                float sign = random_sign_cpu(col, seed);
                values[i] = in[static_cast<long>(row) * K + col] * sign;
            }

            // Apply Hadamard transform
            hadamard16_cpu(values);

            // Scale and store
            for (int i = 0; i < 16 && (block_start + i) < K; ++i) {
                long idx = static_cast<long>(row) * K + block_start + i;
                out[idx] = values[i] * HADAMARD_SCALE;
            }
        }
    }
}

// Compute relative error
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


TEST_CASE("FP4 E2M1 quantization levels", "[fp4][quant]") {
    // Verify FP4 E2M1 quantization produces correct discrete levels
    std::vector<float> test_values = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.75f,
                                       2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 5.0f, 6.0f, 7.0f};

    for (float v : test_values) {
        float q = quantize_fp4_e2m1_cpu(v);
        // Should be one of the valid FP4 values
        bool valid = false;
        for (float fp4_val : FP4_E2M1_VALUES) {
            if (std::abs(q - fp4_val) < 0.001f) {
                valid = true;
                break;
            }
        }
        REQUIRE(valid);
    }

    // Test negative values
    for (float v : test_values) {
        if (v == 0.0f) continue;
        float q = quantize_fp4_e2m1_cpu(-v);
        // Quantized negative value should be <= 0 (could be -0.0 for small values)
        REQUIRE(q <= 0.0f);
    }
}


TEST_CASE("FP4 block quantization roundtrip", "[fp4][quant]") {
    // Test parameters - must be multiples of 16 for FP4
    const int M = 128;
    const int K = 128;

    // Generate random input data
    auto input_f32 = uniform_host(M * K, -1.0f, 1.0f, 42);
    auto input_bf16 = to_bf16(input_f32);

    // Convert to float for comparison
    std::vector<float> input_as_float(M * K);
    for (int i = 0; i < M * K; ++i) {
        input_as_float[i] = static_cast<float>(input_bf16[i]);
    }

    // Get device properties
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    // Allocate device memory
    thrust::device_vector<nv_bfloat16> d_input(input_bf16.begin(), input_bf16.end());

    // Compute output sizes
    const std::size_t packed_size = (static_cast<std::size_t>(M) * K + 1) / 2;
    auto [scale_rows, scale_cols] = fp4_scale_shape(M, K);

    thrust::device_vector<uint8_t> d_fp4(packed_size);
    thrust::device_vector<__nv_fp8_e4m3> d_scales(scale_rows * scale_cols);
    thrust::device_vector<float> d_global_amax(1, 0.0f);
    thrust::device_vector<nv_bfloat16> d_output(M * K);

    // Run GPU quantization using the same two-stage scaling scheme as training:
    // global_encode_scale = FP8_MAX * FP4_MAX / global_amax, global_decode_scale = 1 / global_encode_scale.
    quantize_fp4_block_auto_scale(
        thrust::raw_pointer_cast(d_fp4.data()),
        thrust::raw_pointer_cast(d_scales.data()),
        thrust::raw_pointer_cast(d_global_amax.data()),
        thrust::raw_pointer_cast(d_input.data()),
        M, K,
        props, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get global amax from device
    float h_global_amax;
    CUDA_CHECK(cudaMemcpy(&h_global_amax, thrust::raw_pointer_cast(d_global_amax.data()),
                          sizeof(float), cudaMemcpyDeviceToHost));

    // Compute global decode scale consistent with quantize_fp4_block_auto_scale.
    constexpr float FP8_E4M3_MAX = fp8_max_v<__nv_fp8_e4m3>;
    float global_decode_scale =
        (h_global_amax > 0.0f) ? (h_global_amax / (FP8_E4M3_MAX * FP4_E2M1_MAX)) : 1.0f;

    // Run GPU dequantization
    dequantize_fp4_block(
        thrust::raw_pointer_cast(d_output.data()),
        thrust::raw_pointer_cast(d_fp4.data()),
        thrust::raw_pointer_cast(d_scales.data()),
        global_decode_scale,
        M, K,
        props, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output back to host
    std::vector<nv_bfloat16> h_output(M * K);
    thrust::copy(d_output.begin(), d_output.end(), h_output.begin());

    // Convert to float for comparison
    std::vector<float> output_as_float(M * K);
    for (int i = 0; i < M * K; ++i) {
        output_as_float[i] = static_cast<float>(h_output[i]);
    }

    // Compute error metrics
    float rel_error = compute_relative_error(input_as_float, output_as_float);
    float mae = compute_mae(input_as_float, output_as_float);

    // FP4 has very limited precision (only 8 quantization levels per sign)
    // With 4-bit quantization across a [-1, 1] range, we expect significant error
    // The software emulation may have higher error than native FP4 hardware
    INFO("FP4 roundtrip relative error: " << rel_error);
    INFO("FP4 roundtrip MAE: " << mae);

    // For software emulation with block scaling, error up to 60% is acceptable
    // Native FP4 hardware would have better accuracy
    REQUIRE(rel_error < 0.65f);

    // Also verify MAE is bounded - should be less than half the input range
    REQUIRE(mae < 0.5f);
}


TEST_CASE("Hadamard transform basic properties", "[fp4][hadamard]") {
    // Test that Hadamard transform preserves energy (orthogonality)
    const int M = 64;
    const int K = 64;  // Must be multiple of 16

    auto input_f32 = uniform_host(M * K, -1.0f, 1.0f, 123);
    auto input_bf16 = to_bf16(input_f32);

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    thrust::device_vector<nv_bfloat16> d_input(input_bf16.begin(), input_bf16.end());
    thrust::device_vector<nv_bfloat16> d_output(M * K);

    unsigned int seed = 42;

    // Run GPU Hadamard forward
    hadamard_transform_forward(
        thrust::raw_pointer_cast(d_output.data()),
        thrust::raw_pointer_cast(d_input.data()),
        nullptr,  // amax_out
        M, K, seed, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output back
    std::vector<nv_bfloat16> h_output(M * K);
    thrust::copy(d_output.begin(), d_output.end(), h_output.begin());

    // Compute input and output energies
    float input_energy = 0.0f;
    float output_energy = 0.0f;
    for (int i = 0; i < M * K; ++i) {
        float in_val = static_cast<float>(input_bf16[i]);
        float out_val = static_cast<float>(h_output[i]);
        input_energy += in_val * in_val;
        output_energy += out_val * out_val;
    }

    // Energy should be approximately preserved (within BF16 precision)
    float energy_ratio = output_energy / input_energy;
    INFO("Hadamard energy ratio: " << energy_ratio);
    REQUIRE(energy_ratio > 0.9f);
    REQUIRE(energy_ratio < 1.1f);
}


TEST_CASE("Hadamard transform forward-inverse roundtrip", "[fp4][hadamard]") {
    const int M = 32;
    const int K = 32;

    auto input_f32 = uniform_host(M * K, -1.0f, 1.0f, 456);
    auto input_bf16 = to_bf16(input_f32);

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    thrust::device_vector<nv_bfloat16> d_input(input_bf16.begin(), input_bf16.end());
    thrust::device_vector<nv_bfloat16> d_transformed(M * K);
    thrust::device_vector<nv_bfloat16> d_reconstructed(M * K);

    unsigned int seed = 789;

    // Forward transform
    hadamard_transform_forward(
        thrust::raw_pointer_cast(d_transformed.data()),
        thrust::raw_pointer_cast(d_input.data()),
        nullptr, M, K, seed, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Inverse transform
    hadamard_transform_inverse(
        thrust::raw_pointer_cast(d_reconstructed.data()),
        thrust::raw_pointer_cast(d_transformed.data()),
        M, K, seed, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy reconstructed back
    std::vector<nv_bfloat16> h_reconstructed(M * K);
    thrust::copy(d_reconstructed.begin(), d_reconstructed.end(), h_reconstructed.begin());

    // Compare with original
    std::vector<float> input_as_float(M * K);
    std::vector<float> reconstructed_as_float(M * K);
    for (int i = 0; i < M * K; ++i) {
        input_as_float[i] = static_cast<float>(input_bf16[i]);
        reconstructed_as_float[i] = static_cast<float>(h_reconstructed[i]);
    }

    float rel_error = compute_relative_error(input_as_float, reconstructed_as_float);
    INFO("Hadamard roundtrip relative error: " << rel_error);
    REQUIRE(rel_error < 0.01f);  // Should be very close with BF16 precision
}


TEST_CASE("Hadamard transform CPU-GPU consistency", "[fp4][hadamard]") {
    const int M = 16;
    const int K = 32;

    auto input_f32 = uniform_host(M * K, -0.5f, 0.5f, 999);
    auto input_bf16 = to_bf16(input_f32);

    // Convert to float for CPU reference
    std::vector<float> input_as_float(M * K);
    for (int i = 0; i < M * K; ++i) {
        input_as_float[i] = static_cast<float>(input_bf16[i]);
    }

    unsigned int seed = 12345;

    // CPU reference
    std::vector<float> cpu_output;
    hadamard_transform_forward_cpu(cpu_output, input_as_float, M, K, seed);

    // GPU implementation
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    thrust::device_vector<nv_bfloat16> d_input(input_bf16.begin(), input_bf16.end());
    thrust::device_vector<nv_bfloat16> d_output(M * K);

    hadamard_transform_forward(
        thrust::raw_pointer_cast(d_output.data()),
        thrust::raw_pointer_cast(d_input.data()),
        nullptr, M, K, seed, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<nv_bfloat16> h_output(M * K);
    thrust::copy(d_output.begin(), d_output.end(), h_output.begin());

    std::vector<float> gpu_output_as_float(M * K);
    for (int i = 0; i < M * K; ++i) {
        gpu_output_as_float[i] = static_cast<float>(h_output[i]);
    }

    // Compare CPU and GPU results
    float rel_error = compute_relative_error(cpu_output, gpu_output_as_float);
    INFO("Hadamard CPU-GPU relative error: " << rel_error);
    REQUIRE(rel_error < 0.02f);  // Allow for BF16 precision differences
}


TEST_CASE("FP4 scale shape computation", "[fp4][utils]") {
    // Test that fp4_scale_shape returns correct dimensions
    {
        auto [rows, cols] = fp4_scale_shape(128, 128);
        REQUIRE(rows == 128); // rows rounded up to 128
        REQUIRE(cols == 8);   // ceil(128/16) = 8, aligned to 4 = 8
    }

    {
        auto [rows, cols] = fp4_scale_shape(256, 256);
        REQUIRE(rows == 256); // rows rounded up to 128
        REQUIRE(cols == 16);  // ceil(256/16) = 16
    }

    {
        auto [rows, cols] = fp4_scale_shape(512, 1024);
        REQUIRE(rows == 512); // rows rounded up to 128
        REQUIRE(cols == 64);  // ceil(1024/16) = 64
    }
}

TEST_CASE("FP4 cuDNN matmul matches BF16 reference (within tolerance)", "[fp4][matmul]") {
    if (!device_supports_fp4()) {
        SKIP("FP4 matmul requires Blackwell GPU (SM100+)");
    }

    // Keep dimensions small but representative; must satisfy FP4 constraints.
    constexpr int M = 128;
    constexpr int N = 512;
    constexpr int K = 256;
    constexpr int BLOCK_SIZE = 16;

    // Host inputs (float) -> BF16 rounded (to match training path).
    std::vector<float> a_f(M * K);
    std::vector<float> w_f(N * K);
    testing_utils::fill_normal(a_f, 0.0f, 0.02f, 123);
    testing_utils::fill_normal(w_f, 0.0f, 0.02f, 456);

    auto a_bf16 = testing_utils::to_bf16(a_f);
    auto w_bf16 = testing_utils::to_bf16(w_f);

    // Round-trip BF16 to float for CPU reference matmul.
    std::vector<float> a_ref = testing_utils::round_bf16(a_f);
    std::vector<float> w_ref = testing_utils::round_bf16(w_f);

    // Device props / stream.
    cudaDeviceProp dp{};
    CUDA_CHECK(cudaGetDeviceProperties(&dp, 0));

    cudaStream_t stream = nullptr;  // default stream

    // Allocate BF16 A and W.
    thrust::device_vector<nv_bfloat16> d_a(a_bf16.begin(), a_bf16.end());
    thrust::device_vector<nv_bfloat16> d_w(w_bf16.begin(), w_bf16.end());

    // Allocate FP4 packed data and FP8 block scales.
    const std::size_t a_fp4_bytes = (std::size_t)M * (std::size_t)(K / 2);
    const std::size_t w_fp4_bytes = (std::size_t)N * (std::size_t)(K / 2);
    thrust::device_vector<uint8_t> d_a_fp4(a_fp4_bytes);
    thrust::device_vector<uint8_t> d_w_fp4(w_fp4_bytes);

    const int64_t rounded_m = ((M + 127) / 128) * 128;
    const int64_t rounded_n = ((N + 127) / 128) * 128;
    const int64_t rounded_kblocks = (((K / BLOCK_SIZE) + 3) / 4) * 4;

    thrust::device_vector<__nv_fp8_e4m3> d_scale_a((std::size_t)rounded_m * (std::size_t)rounded_kblocks);
    thrust::device_vector<__nv_fp8_e4m3> d_scale_b((std::size_t)rounded_kblocks * (std::size_t)rounded_n);
    thrust::device_vector<float> d_amax_a(1);
    thrust::device_vector<float> d_amax_b(1);

    // Quantize A and W to FP4 (two-level scaling, auto scale).
    quantize_fp4_block_auto_scale(
        thrust::raw_pointer_cast(d_a_fp4.data()),
        thrust::raw_pointer_cast(d_scale_a.data()),
        thrust::raw_pointer_cast(d_amax_a.data()),
        thrust::raw_pointer_cast(d_a.data()),
        M, K, dp, stream);

    // Weights use the NVFP4 16x16 block scaling variant (TransformerEngine recipe).
    quantize_fp4_weight_2d_auto_scale(
        thrust::raw_pointer_cast(d_w_fp4.data()),
        thrust::raw_pointer_cast(d_scale_b.data()),
        thrust::raw_pointer_cast(d_amax_b.data()),
        thrust::raw_pointer_cast(d_w.data()),
        N, K, dp, stream);

    // FP4 matmul via cuDNN (FP32 output) + alpha correction.
    thrust::device_vector<float> d_out((std::size_t)M * (std::size_t)N);

    cudnnHandle_t cudnn_handle = create_cudnn_handle();

    const std::size_t ws_bytes = fp4_matmul_get_workspace_size(M, N, K, BLOCK_SIZE, cudnn_handle);
    void* ws_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ws_ptr, ws_bytes));

    fp4_matmul_f32(
        thrust::raw_pointer_cast(d_out.data()),
        thrust::raw_pointer_cast(d_a_fp4.data()),
        thrust::raw_pointer_cast(d_w_fp4.data()),
        thrust::raw_pointer_cast(d_scale_a.data()),
        thrust::raw_pointer_cast(d_scale_b.data()),
        1.0f, 1.0f,
        reinterpret_cast<std::byte*>(ws_ptr), ws_bytes,
        M, N, K, BLOCK_SIZE,
        cudnn_handle, stream);

    fp4_alpha_scale(
        thrust::raw_pointer_cast(d_out.data()),
        thrust::raw_pointer_cast(d_amax_a.data()),
        thrust::raw_pointer_cast(d_amax_b.data()),
        (long)M * N,
        dp, stream);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(ws_ptr));
    destroy_cudnn_handle(cudnn_handle);

    // Copy FP4 output to host.
    std::vector<float> out_fp4((std::size_t)M * (std::size_t)N);
    thrust::copy(d_out.begin(), d_out.end(), out_fp4.begin());

    // CPU reference: y = A @ W^T (float accumulation over BF16-rounded inputs).
    std::vector<float> out_ref((std::size_t)M * (std::size_t)N, 0.0f);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const float* a_row = a_ref.data() + (std::size_t)i * K;
            const float* w_row = w_ref.data() + (std::size_t)j * K;
            for (int kk = 0; kk < K; ++kk) {
                acc += a_row[kk] * w_row[kk];
            }
            out_ref[(std::size_t)i * N + j] = acc;
        }
    }

    // Compare relative L2 error.
    double num = 0.0;
    double den = 0.0;
    for (std::size_t idx = 0; idx < out_ref.size(); ++idx) {
        const double diff = (double)out_fp4[idx] - (double)out_ref[idx];
        num += diff * diff;
        den += (double)out_ref[idx] * (double)out_ref[idx];
    }
    const double rel_l2 = std::sqrt(num / std::max(den, 1e-30));
    INFO("FP4 matmul relative L2 error: " << rel_l2);

    // FP4 is noisy, but should be well below catastrophic error.
    REQUIRE(rel_l2 < 0.35);
}

TEST_CASE("FP4 dW matmul path (dout^T @ inp) is numerically reasonable", "[fp4][backward]") {
    if (!device_supports_fp4()) {
        SKIP("FP4 matmul requires Blackwell GPU (SM100+)");
    }

    // Representative backward dims. BT must be divisible by 16 for NVFP4 block scaling.
    constexpr int BT = 4096;
    constexpr int OC = 512;
    constexpr int C = 256;
    constexpr int BLOCK_SIZE = 16;

    std::vector<float> dout_f((std::size_t)BT * (std::size_t)OC);
    std::vector<float> inp_f((std::size_t)BT * (std::size_t)C);
    testing_utils::fill_normal(dout_f, 0.0f, 0.02f, 101);
    testing_utils::fill_normal(inp_f, 0.0f, 0.02f, 202);

    auto dout_bf16 = testing_utils::to_bf16(dout_f);
    auto inp_bf16 = testing_utils::to_bf16(inp_f);

    // CPU reference: dW = dout^T @ inp (float accumulation over BF16-rounded inputs).
    std::vector<float> dout_ref = testing_utils::round_bf16(dout_f);
    std::vector<float> inp_ref = testing_utils::round_bf16(inp_f);
    std::vector<float> dw_ref((std::size_t)OC * (std::size_t)C, 0.0f);
    for (int oc = 0; oc < OC; ++oc) {
        for (int c = 0; c < C; ++c) {
            float acc = 0.0f;
            for (int bt = 0; bt < BT; ++bt) {
                acc += dout_ref[(std::size_t)bt * OC + (std::size_t)oc] * inp_ref[(std::size_t)bt * C + (std::size_t)c];
            }
            dw_ref[(std::size_t)oc * C + (std::size_t)c] = acc;
        }
    }

    cudaDeviceProp dp{};
    CUDA_CHECK(cudaGetDeviceProperties(&dp, 0));
    cudaStream_t stream = nullptr;

    // Device BF16 inputs.
    // Transpose on CPU to avoid depending on Tensor helpers in the unit test.
    std::vector<nv_bfloat16> dout_tp((std::size_t)OC * (std::size_t)BT);
    std::vector<nv_bfloat16> inp_tp((std::size_t)C * (std::size_t)BT);
    for (int bt = 0; bt < BT; ++bt) {
        for (int oc = 0; oc < OC; ++oc) {
            dout_tp[(std::size_t)oc * (std::size_t)BT + (std::size_t)bt] =
                dout_bf16[(std::size_t)bt * (std::size_t)OC + (std::size_t)oc];
        }
        for (int c = 0; c < C; ++c) {
            inp_tp[(std::size_t)c * (std::size_t)BT + (std::size_t)bt] =
                inp_bf16[(std::size_t)bt * (std::size_t)C + (std::size_t)c];
        }
    }

    thrust::device_vector<nv_bfloat16> d_dout_tp(dout_tp.begin(), dout_tp.end());
    thrust::device_vector<nv_bfloat16> d_inp_tp(inp_tp.begin(), inp_tp.end());

    // Quantize A with stochastic rounding (activation-style scale layout).
    thrust::device_vector<uint8_t> d_a_fp4((std::size_t)OC * (std::size_t)(BT / 2));
    const auto [a_scale_rows, a_scale_cols] = fp4_scale_shape(OC, BT);
    thrust::device_vector<__nv_fp8_e4m3> d_a_scales((std::size_t)a_scale_rows * (std::size_t)a_scale_cols);
    thrust::device_vector<float> d_a_amax(1);
    quantize_fp4_block_stochastic_auto_scale(
        thrust::raw_pointer_cast(d_a_fp4.data()),
        thrust::raw_pointer_cast(d_a_scales.data()),
        thrust::raw_pointer_cast(d_a_amax.data()),
        thrust::raw_pointer_cast(d_dout_tp.data()),
        OC, BT,
        /*seed=*/777,
        dp, stream);

    // Quantize B with B-operand scale layout.
    thrust::device_vector<uint8_t> d_b_fp4((std::size_t)C * (std::size_t)(BT / 2));
    const long b_scale_rows = ((BT + 15) / 16 + 3) / 4 * 4;   // K/16 aligned to 4
    const long b_scale_cols = (C + 127) / 128 * 128;          // N aligned to 128
    thrust::device_vector<__nv_fp8_e4m3> d_b_scales((std::size_t)b_scale_rows * (std::size_t)b_scale_cols);
    thrust::device_vector<float> d_b_amax(1);
    quantize_fp4_weight_auto_scale(
        thrust::raw_pointer_cast(d_b_fp4.data()),
        thrust::raw_pointer_cast(d_b_scales.data()),
        thrust::raw_pointer_cast(d_b_amax.data()),
        thrust::raw_pointer_cast(d_inp_tp.data()),
        C, BT,
        dp, stream);

    // FP4 matmul: (OC, BT) @ (BT, C) -> (OC, C)
    thrust::device_vector<float> d_dw((std::size_t)OC * (std::size_t)C);
    cudnnHandle_t cudnn_handle = create_cudnn_handle();
    const std::size_t ws_bytes = fp4_matmul_get_workspace_size(OC, C, BT, BLOCK_SIZE, cudnn_handle);
    void* ws_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ws_ptr, ws_bytes));

    fp4_matmul_f32(
        thrust::raw_pointer_cast(d_dw.data()),
        thrust::raw_pointer_cast(d_a_fp4.data()),
        thrust::raw_pointer_cast(d_b_fp4.data()),
        thrust::raw_pointer_cast(d_a_scales.data()),
        thrust::raw_pointer_cast(d_b_scales.data()),
        1.0f, 1.0f,
        reinterpret_cast<std::byte*>(ws_ptr), ws_bytes,
        OC, C, BT, BLOCK_SIZE,
        cudnn_handle, stream);

    fp4_alpha_scale(
        thrust::raw_pointer_cast(d_dw.data()),
        thrust::raw_pointer_cast(d_a_amax.data()),
        thrust::raw_pointer_cast(d_b_amax.data()),
        (long)OC * C,
        dp, stream);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(ws_ptr));
    destroy_cudnn_handle(cudnn_handle);

    std::vector<float> dw_fp4((std::size_t)OC * (std::size_t)C);
    thrust::copy(d_dw.begin(), d_dw.end(), dw_fp4.begin());

    // Compare relative L2 error.
    double num = 0.0;
    double den = 0.0;
    for (std::size_t idx = 0; idx < dw_ref.size(); ++idx) {
        const double diff = (double)dw_fp4[idx] - (double)dw_ref[idx];
        num += diff * diff;
        den += (double)dw_ref[idx] * (double)dw_ref[idx];
    }
    const double rel_l2 = std::sqrt(num / std::max(den, 1e-30));
    INFO("FP4 dW relative L2 error: " << rel_l2);

    REQUIRE(rel_l2 < 0.60);
}
