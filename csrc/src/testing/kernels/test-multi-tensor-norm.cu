// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Tests for multi-tensor fused gradient norm kernels.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "kernels/kernels.h"
#include "utilities/utils.h"

namespace {

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// Helper: compute reference sum((val*prescale)^2) on host
double ref_norm_squared_prescaled(const std::vector<float>& vals, float prescale) {
    double sum = 0.0;
    for (float v : vals) {
        double s = (double)v * (double)prescale;
        sum += s * s;
    }
    return sum;
}

} // namespace

TEST_CASE("multi-tensor amax: single FP32 tensor", "[kernels][multi-tensor-norm][gpu]") {
    if (!cuda_available()) SKIP("CUDA not available");

    cudaDeviceProp dp{};
    CUDA_CHECK(cudaGetDeviceProperties(&dp, 0));

    std::vector<float> h_data = {1.0f, -3.0f, 2.0f, -0.5f};

    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, h_data.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), h_data.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Set up multi-tensor arrays
    const void* h_ptrs[1] = {d_data};
    size_t h_sizes[1] = {h_data.size()};
    int h_flags[1] = {0};  // FP32

    const void** d_ptrs = nullptr;
    size_t* d_sizes = nullptr;
    int* d_flags = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptrs, sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&d_sizes, sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_flags, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_ptrs, h_ptrs, sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sizes, h_sizes, sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flags, h_flags, sizeof(int), cudaMemcpyHostToDevice));

    // Amax output
    float* d_amax = nullptr;
    CUDA_CHECK(cudaMalloc(&d_amax, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_amax, 0, sizeof(float)));

    global_amax_multi_tensor(d_amax, d_ptrs, d_sizes, d_flags, 1, dp, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_amax = 0.f;
    CUDA_CHECK(cudaMemcpy(&h_amax, d_amax, sizeof(float), cudaMemcpyDeviceToHost));

    REQUIRE(h_amax == Catch::Approx(3.0f));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_ptrs));
    CUDA_CHECK(cudaFree(d_sizes));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_amax));
}

TEST_CASE("multi-tensor amax: mixed FP32 + BF16 tensors", "[kernels][multi-tensor-norm][gpu]") {
    if (!cuda_available()) SKIP("CUDA not available");

    cudaDeviceProp dp{};
    CUDA_CHECK(cudaGetDeviceProperties(&dp, 0));

    // FP32 tensor: max abs = 5.0
    std::vector<float> h_fp32 = {1.0f, -5.0f, 2.0f};
    // BF16 tensor: max abs = 7.0
    std::vector<float> h_bf16_float = {-7.0f, 3.0f, 0.5f, 1.0f};
    std::vector<nv_bfloat16> h_bf16(h_bf16_float.size());
    for (size_t i = 0; i < h_bf16_float.size(); ++i) h_bf16[i] = __float2bfloat16(h_bf16_float[i]);

    float* d_fp32 = nullptr;
    nv_bfloat16* d_bf16 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_fp32, h_fp32.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bf16, h_bf16.size() * sizeof(nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_fp32, h_fp32.data(), h_fp32.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bf16, h_bf16.data(), h_bf16.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

    const void* h_ptrs[2] = {d_fp32, d_bf16};
    size_t h_sizes[2] = {h_fp32.size(), h_bf16.size()};
    int h_flags[2] = {0, 1};  // FP32, BF16

    const void** d_ptrs = nullptr;
    size_t* d_sizes = nullptr;
    int* d_flags = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptrs, 2 * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&d_sizes, 2 * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_flags, 2 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_ptrs, h_ptrs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sizes, h_sizes, 2 * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flags, h_flags, 2 * sizeof(int), cudaMemcpyHostToDevice));

    float* d_amax = nullptr;
    CUDA_CHECK(cudaMalloc(&d_amax, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_amax, 0, sizeof(float)));

    global_amax_multi_tensor(d_amax, d_ptrs, d_sizes, d_flags, 2, dp, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_amax = 0.f;
    CUDA_CHECK(cudaMemcpy(&h_amax, d_amax, sizeof(float), cudaMemcpyDeviceToHost));

    REQUIRE(h_amax == Catch::Approx(7.0f).epsilon(0.01));

    CUDA_CHECK(cudaFree(d_fp32));
    CUDA_CHECK(cudaFree(d_bf16));
    CUDA_CHECK(cudaFree(d_ptrs));
    CUDA_CHECK(cudaFree(d_sizes));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_amax));
}

TEST_CASE("multi-tensor prescaled norm²: matches sequential per-tensor computation", "[kernels][multi-tensor-norm][gpu]") {
    if (!cuda_available()) SKIP("CUDA not available");

    cudaDeviceProp dp{};
    CUDA_CHECK(cudaGetDeviceProperties(&dp, 0));
    const int N = get_max_num_block_sums(dp);

    // Create two tensors: FP32 and BF16
    std::vector<float> h_fp32 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> h_bf16_float = {6.0f, 7.0f, 8.0f};
    std::vector<nv_bfloat16> h_bf16(h_bf16_float.size());
    for (size_t i = 0; i < h_bf16_float.size(); ++i) h_bf16[i] = __float2bfloat16(h_bf16_float[i]);

    float* d_fp32 = nullptr;
    nv_bfloat16* d_bf16 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_fp32, h_fp32.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bf16, h_bf16.size() * sizeof(nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_fp32, h_fp32.data(), h_fp32.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bf16, h_bf16.data(), h_bf16.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

    const void* h_ptrs[2] = {d_fp32, d_bf16};
    size_t h_sizes[2] = {h_fp32.size(), h_bf16.size()};
    int h_flags[2] = {0, 1};

    const void** d_ptrs = nullptr;
    size_t* d_sizes = nullptr;
    int* d_flags = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptrs, 2 * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&d_sizes, 2 * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_flags, 2 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_ptrs, h_ptrs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sizes, h_sizes, 2 * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flags, h_flags, 2 * sizeof(int), cudaMemcpyHostToDevice));

    // Prescale: 1/8 (amax=8)
    float h_prescale = 1.0f / 8.0f;
    float* d_prescale = nullptr;
    CUDA_CHECK(cudaMalloc(&d_prescale, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_prescale, &h_prescale, sizeof(float), cudaMemcpyHostToDevice));

    // Fused multi-tensor result
    float* d_out_fused = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_fused, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out_fused, 0, N * sizeof(float)));

    global_norm_squared_prescaled_multi_tensor(d_out_fused, d_ptrs, d_sizes, d_flags, 2, d_prescale, dp, nullptr);

    // Reduce to scalar
    float* d_sum_fused = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum_fused, sizeof(float)));
    deterministic_sum(d_sum_fused, d_out_fused, N, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_sum_fused = 0.f;
    CUDA_CHECK(cudaMemcpy(&h_sum_fused, d_sum_fused, sizeof(float), cudaMemcpyDeviceToHost));

    // Sequential per-tensor result
    float* d_out_seq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_seq, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out_seq, 0, N * sizeof(float)));
    global_norm_squared_prescaled(d_out_seq, d_fp32, h_fp32.size(), d_prescale, dp, nullptr);
    global_norm_squared_prescaled(d_out_seq, d_bf16, h_bf16.size(), d_prescale, dp, nullptr);
    float* d_sum_seq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum_seq, sizeof(float)));
    deterministic_sum(d_sum_seq, d_out_seq, N, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_sum_seq = 0.f;
    CUDA_CHECK(cudaMemcpy(&h_sum_seq, d_sum_seq, sizeof(float), cudaMemcpyDeviceToHost));

    // Reference: sum((val / 8)^2) for all values
    std::vector<float> all_vals = h_fp32;
    // BF16 values have rounding, use the BF16-rounded values
    for (auto bf : h_bf16) all_vals.push_back(__bfloat162float(bf));
    double ref = ref_norm_squared_prescaled(all_vals, h_prescale);

    REQUIRE(h_sum_fused == Catch::Approx(h_sum_seq).epsilon(1e-5));
    REQUIRE(h_sum_fused == Catch::Approx((float)ref).epsilon(1e-5));

    CUDA_CHECK(cudaFree(d_fp32));
    CUDA_CHECK(cudaFree(d_bf16));
    CUDA_CHECK(cudaFree(d_ptrs));
    CUDA_CHECK(cudaFree(d_sizes));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_prescale));
    CUDA_CHECK(cudaFree(d_out_fused));
    CUDA_CHECK(cudaFree(d_sum_fused));
    CUDA_CHECK(cudaFree(d_out_seq));
    CUDA_CHECK(cudaFree(d_sum_seq));
}

TEST_CASE("multi-tensor norm² (no prescale): matches sequential computation", "[kernels][multi-tensor-norm][gpu]") {
    if (!cuda_available()) SKIP("CUDA not available");

    cudaDeviceProp dp{};
    CUDA_CHECK(cudaGetDeviceProperties(&dp, 0));
    const int N = get_max_num_block_sums(dp);

    std::vector<float> h_fp32 = {1.0f, 2.0f, 3.0f};
    std::vector<float> h_bf16_float = {4.0f, 5.0f};
    std::vector<nv_bfloat16> h_bf16(h_bf16_float.size());
    for (size_t i = 0; i < h_bf16_float.size(); ++i) h_bf16[i] = __float2bfloat16(h_bf16_float[i]);

    float* d_fp32 = nullptr;
    nv_bfloat16* d_bf16 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_fp32, h_fp32.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bf16, h_bf16.size() * sizeof(nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_fp32, h_fp32.data(), h_fp32.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bf16, h_bf16.data(), h_bf16.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

    const void* h_ptrs[2] = {d_fp32, d_bf16};
    size_t h_sizes[2] = {h_fp32.size(), h_bf16.size()};
    int h_flags[2] = {0, 1};

    const void** d_ptrs = nullptr;
    size_t* d_sizes = nullptr;
    int* d_flags = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptrs, 2 * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&d_sizes, 2 * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_flags, 2 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_ptrs, h_ptrs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sizes, h_sizes, 2 * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flags, h_flags, 2 * sizeof(int), cudaMemcpyHostToDevice));

    // Fused
    float* d_out_fused = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_fused, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out_fused, 0, N * sizeof(float)));
    global_norm_squared_multi_tensor(d_out_fused, d_ptrs, d_sizes, d_flags, 2, dp, nullptr);
    float* d_sum_fused = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum_fused, sizeof(float)));
    deterministic_sum(d_sum_fused, d_out_fused, N, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_sum_fused = 0.f;
    CUDA_CHECK(cudaMemcpy(&h_sum_fused, d_sum_fused, sizeof(float), cudaMemcpyDeviceToHost));

    // Sequential
    float* d_out_seq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_seq, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out_seq, 0, N * sizeof(float)));
    global_norm_squared(d_out_seq, d_fp32, h_fp32.size(), dp, nullptr);
    global_norm_squared(d_out_seq, d_bf16, h_bf16.size(), dp, nullptr);
    float* d_sum_seq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum_seq, sizeof(float)));
    deterministic_sum(d_sum_seq, d_out_seq, N, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_sum_seq = 0.f;
    CUDA_CHECK(cudaMemcpy(&h_sum_seq, d_sum_seq, sizeof(float), cudaMemcpyDeviceToHost));

    // Reference: 1+4+9+16+25 = 55
    REQUIRE(h_sum_fused == Catch::Approx(h_sum_seq).epsilon(1e-5));
    REQUIRE(h_sum_fused == Catch::Approx(55.0f).epsilon(1e-3));

    CUDA_CHECK(cudaFree(d_fp32));
    CUDA_CHECK(cudaFree(d_bf16));
    CUDA_CHECK(cudaFree(d_ptrs));
    CUDA_CHECK(cudaFree(d_sizes));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_out_fused));
    CUDA_CHECK(cudaFree(d_sum_fused));
    CUDA_CHECK(cudaFree(d_out_seq));
    CUDA_CHECK(cudaFree(d_sum_seq));
}

TEST_CASE("multi-tensor prescaled norm²: large BF16 values use prescale to prevent overflow", "[kernels][multi-tensor-norm][gpu]") {
    if (!cuda_available()) SKIP("CUDA not available");

    cudaDeviceProp dp{};
    CUDA_CHECK(cudaGetDeviceProperties(&dp, 0));
    const int N = get_max_num_block_sums(dp);

    // Large BF16 value: ~1e18 (close to BF16 max ~3.4e38, but squaring would overflow FP32 at ~1.8e19)
    const float large_val = 1.0e18f;
    const int count = 64;
    std::vector<nv_bfloat16> h_bf16(count);
    for (int i = 0; i < count; ++i) h_bf16[i] = __float2bfloat16(large_val);

    nv_bfloat16* d_bf16 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bf16, count * sizeof(nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_bf16, h_bf16.data(), count * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

    const void* h_ptrs[1] = {d_bf16};
    size_t h_sizes[1] = {(size_t)count};
    int h_flags[1] = {1};

    const void** d_ptrs = nullptr;
    size_t* d_sizes = nullptr;
    int* d_flags = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptrs, sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&d_sizes, sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_flags, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_ptrs, h_ptrs, sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sizes, h_sizes, sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flags, h_flags, sizeof(int), cudaMemcpyHostToDevice));

    // Step 1: amax
    float* d_amax = nullptr;
    CUDA_CHECK(cudaMalloc(&d_amax, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_amax, 0, sizeof(float)));
    global_amax_multi_tensor(d_amax, d_ptrs, d_sizes, d_flags, 1, dp, nullptr);

    // Step 2: prescale
    float* d_prescale = nullptr;
    CUDA_CHECK(cudaMalloc(&d_prescale, sizeof(float)));
    compute_prescale(d_prescale, d_amax, nullptr);

    // Step 3: prescaled norm²
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));
    global_norm_squared_prescaled_multi_tensor(d_out, d_ptrs, d_sizes, d_flags, 1, d_prescale, dp, nullptr);

    float* d_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    deterministic_sum(d_sum, d_out, N, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_sum = 0.f;
    float h_amax = 0.f;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_amax, d_amax, sizeof(float), cudaMemcpyDeviceToHost));

    // Verify: prescaled sum should be count * (large_val/amax)^2 = count * 1.0 = 64
    // (all values are the same, so prescale = 1/amax, each term = 1.0)
    float bf16_val = __bfloat162float(__float2bfloat16(large_val));
    REQUIRE(h_amax == Catch::Approx(bf16_val).epsilon(0.01));
    REQUIRE(h_sum == Catch::Approx((float)count).epsilon(0.01));

    // The actual norm = amax * sqrt(sum) = amax * sqrt(64) = amax * 8
    float actual_norm = h_amax * std::sqrt(h_sum);
    float expected_norm = bf16_val * std::sqrt((float)count);
    REQUIRE(actual_norm == Catch::Approx(expected_norm).epsilon(0.01));

    // Verify the result is finite (not inf/nan from overflow)
    REQUIRE(std::isfinite(h_sum));
    REQUIRE(std::isfinite(actual_norm));

    CUDA_CHECK(cudaFree(d_bf16));
    CUDA_CHECK(cudaFree(d_ptrs));
    CUDA_CHECK(cudaFree(d_sizes));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_amax));
    CUDA_CHECK(cudaFree(d_prescale));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_sum));
}

TEST_CASE("multi-tensor: num_tensors=0 is a no-op", "[kernels][multi-tensor-norm][gpu]") {
    if (!cuda_available()) SKIP("CUDA not available");

    cudaDeviceProp dp{};
    CUDA_CHECK(cudaGetDeviceProperties(&dp, 0));
    const int N = get_max_num_block_sums(dp);

    // Amax: should remain 0
    float* d_amax = nullptr;
    CUDA_CHECK(cudaMalloc(&d_amax, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_amax, 0, sizeof(float)));
    global_amax_multi_tensor(d_amax, nullptr, nullptr, nullptr, 0, dp, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_amax = -1.f;
    CUDA_CHECK(cudaMemcpy(&h_amax, d_amax, sizeof(float), cudaMemcpyDeviceToHost));
    REQUIRE(h_amax == 0.f);

    // Norm²: should remain 0
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));

    float h_prescale = 1.0f;
    float* d_prescale = nullptr;
    CUDA_CHECK(cudaMalloc(&d_prescale, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_prescale, &h_prescale, sizeof(float), cudaMemcpyHostToDevice));

    global_norm_squared_prescaled_multi_tensor(d_out, nullptr, nullptr, nullptr, 0, d_prescale, dp, nullptr);
    global_norm_squared_multi_tensor(d_out, nullptr, nullptr, nullptr, 0, dp, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify all zeros
    std::vector<float> h_out(N, -1.f);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        REQUIRE(h_out[i] == 0.f);
    }

    CUDA_CHECK(cudaFree(d_amax));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_prescale));
}
