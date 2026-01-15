// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>

#include <cuda_bf16.h>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "kernels/kernels.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"
#include "../utilities/test_config.h"
#include "../utilities/test_utils.h"

using namespace testing_utils;

namespace {

// Count zeros in a float array
size_t count_zeros(const std::vector<float>& data, float eps = 1e-9f) {
    size_t count = 0;
    for (float v : data) {
        if (std::fabs(v) < eps) ++count;
    }
    return count;
}

// Check if two vectors are approximately equal
bool vectors_equal(const std::vector<float>& a, const std::vector<float>& b, float rtol = 1e-5f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > rtol * std::max(1.0f, std::fabs(a[i]))) {
            return false;
        }
    }
    return true;
}

// Check if two vectors are different
bool vectors_different(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return true;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return true;
    }
    return false;
}

} // namespace

TEST_CASE("LoRA dropout - zero dropout is no-op", "[lora][dropout]") {
    const int N = 1024;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create input data
    auto h_input = uniform_host(N, -1.0f, 1.0f, 42);
    auto h_original = h_input;  // Keep a copy

    // Create device tensor
    thrust::device_vector<float> d_data = to_device(h_input);
    Tensor tensor;
    tensor.Data = reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_data.data()));
    tensor.DType = ETensorDType::FP32;
    tensor.Rank = 1;
    tensor.Sizes[0] = N;

    // Apply dropout with prob=0 (should be no-op)
    lora_dropout_scale(tensor, 0.0f, 12345, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy back and verify unchanged
    auto h_result = from_device(d_data);
    REQUIRE(vectors_equal(h_result, h_original));

    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE("LoRA dropout - full dropout zeros everything", "[lora][dropout]") {
    const int N = 1024;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create input data with non-zero values
    auto h_input = uniform_host(N, 1.0f, 2.0f, 42);

    // Create device tensor
    thrust::device_vector<float> d_data = to_device(h_input);
    Tensor tensor;
    tensor.Data = reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_data.data()));
    tensor.DType = ETensorDType::FP32;
    tensor.Rank = 1;
    tensor.Sizes[0] = N;

    // Apply dropout with prob=1 (should zero everything)
    lora_dropout_scale(tensor, 1.0f, 12345, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy back and verify all zeros
    auto h_result = from_device(d_data);
    size_t num_zeros = count_zeros(h_result);
    REQUIRE(num_zeros == static_cast<size_t>(N));

    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE("LoRA dropout - deterministic with same seed", "[lora][dropout]") {
    const int N = 4096;
    const float dropout_prob = 0.3f;
    const unsigned int seed = 98765;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create input data
    auto h_input = uniform_host(N, -1.0f, 1.0f, 42);

    // First run
    thrust::device_vector<float> d_data1 = to_device(h_input);
    Tensor tensor1;
    tensor1.Data = reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_data1.data()));
    tensor1.DType = ETensorDType::FP32;
    tensor1.Rank = 1;
    tensor1.Sizes[0] = N;

    lora_dropout_scale(tensor1, dropout_prob, seed, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto h_result1 = from_device(d_data1);

    // Second run with same seed
    thrust::device_vector<float> d_data2 = to_device(h_input);
    Tensor tensor2;
    tensor2.Data = reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_data2.data()));
    tensor2.DType = ETensorDType::FP32;
    tensor2.Rank = 1;
    tensor2.Sizes[0] = N;

    lora_dropout_scale(tensor2, dropout_prob, seed, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto h_result2 = from_device(d_data2);

    // Results should be identical
    REQUIRE(vectors_equal(h_result1, h_result2));

    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE("LoRA dropout - different seeds give different results", "[lora][dropout]") {
    const int N = 4096;
    const float dropout_prob = 0.3f;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create input data
    auto h_input = uniform_host(N, -1.0f, 1.0f, 42);

    // First run with seed 1
    thrust::device_vector<float> d_data1 = to_device(h_input);
    Tensor tensor1;
    tensor1.Data = reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_data1.data()));
    tensor1.DType = ETensorDType::FP32;
    tensor1.Rank = 1;
    tensor1.Sizes[0] = N;

    lora_dropout_scale(tensor1, dropout_prob, 11111, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto h_result1 = from_device(d_data1);

    // Second run with different seed
    thrust::device_vector<float> d_data2 = to_device(h_input);
    Tensor tensor2;
    tensor2.Data = reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_data2.data()));
    tensor2.DType = ETensorDType::FP32;
    tensor2.Rank = 1;
    tensor2.Sizes[0] = N;

    lora_dropout_scale(tensor2, dropout_prob, 22222, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto h_result2 = from_device(d_data2);

    // Results should be different
    REQUIRE(vectors_different(h_result1, h_result2));

    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE("LoRA dropout - approximate dropout rate", "[lora][dropout]") {
    const int N = 100000;  // Large N for statistical accuracy
    const float dropout_prob = 0.2f;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create input data with non-zero values
    auto h_input = uniform_host(N, 1.0f, 2.0f, 42);

    // Create device tensor
    thrust::device_vector<float> d_data = to_device(h_input);
    Tensor tensor;
    tensor.Data = reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_data.data()));
    tensor.DType = ETensorDType::FP32;
    tensor.Rank = 1;
    tensor.Sizes[0] = N;

    lora_dropout_scale(tensor, dropout_prob, 54321, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto h_result = from_device(d_data);
    size_t num_zeros = count_zeros(h_result);
    float actual_rate = static_cast<float>(num_zeros) / static_cast<float>(N);

    // Allow 2% tolerance for statistical variance
    REQUIRE(actual_rate > dropout_prob - 0.02f);
    REQUIRE(actual_rate < dropout_prob + 0.02f);

    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE("LoRA dropout - inverted scaling is correct", "[lora][dropout]") {
    const int N = 10000;
    const float dropout_prob = 0.25f;
    const float expected_scale = 1.0f / (1.0f - dropout_prob);  // 1.333...
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create input with known values
    std::vector<float> h_input(N, 1.0f);  // All ones

    thrust::device_vector<float> d_data = to_device(h_input);
    Tensor tensor;
    tensor.Data = reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_data.data()));
    tensor.DType = ETensorDType::FP32;
    tensor.Rank = 1;
    tensor.Sizes[0] = N;

    lora_dropout_scale(tensor, dropout_prob, 99999, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto h_result = from_device(d_data);

    // Non-zero values should be scaled by 1/(1-p)
    int non_zero_count = 0;
    float scale_tolerance = 1e-5f;
    for (float v : h_result) {
        if (std::fabs(v) > 1e-9f) {
            ++non_zero_count;
            // Value should be exactly expected_scale (since input was 1.0)
            REQUIRE(std::fabs(v - expected_scale) < scale_tolerance);
        }
    }

    // Should have some non-zero values (statistically ~75%)
    REQUIRE(non_zero_count > N / 2);

    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE("LoRA dropout - BF16 support", "[lora][dropout]") {
    const int N = 4096;
    const float dropout_prob = 0.3f;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create input data in FP32, convert to BF16
    auto h_input_fp32 = uniform_host(N, -1.0f, 1.0f, 42);
    auto h_input_bf16 = to_bf16(h_input_fp32);

    // Create device tensor
    thrust::device_vector<nv_bfloat16> d_data = to_device(h_input_bf16);
    Tensor tensor;
    tensor.Data = reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_data.data()));
    tensor.DType = ETensorDType::BF16;
    tensor.Rank = 1;
    tensor.Sizes[0] = N;

    // Should not throw
    lora_dropout_scale(tensor, dropout_prob, 12345, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify determinism with same seed
    thrust::device_vector<nv_bfloat16> d_data2 = to_device(h_input_bf16);
    Tensor tensor2;
    tensor2.Data = reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_data2.data()));
    tensor2.DType = ETensorDType::BF16;
    tensor2.Rank = 1;
    tensor2.Sizes[0] = N;

    lora_dropout_scale(tensor2, dropout_prob, 12345, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto h_result1 = from_device(d_data);
    auto h_result2 = from_device(d_data2);

    // Results should be identical
    REQUIRE(h_result1.size() == h_result2.size());
    for (size_t i = 0; i < h_result1.size(); ++i) {
        REQUIRE(__bfloat162float(h_result1[i]) == __bfloat162float(h_result2[i]));
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE("LoRA dropout - small tensor edge case", "[lora][dropout]") {
    const int N = 7;  // Small, not aligned to vector size
    const float dropout_prob = 0.5f;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto h_input = uniform_host(N, 1.0f, 2.0f, 42);

    thrust::device_vector<float> d_data = to_device(h_input);
    Tensor tensor;
    tensor.Data = reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_data.data()));
    tensor.DType = ETensorDType::FP32;
    tensor.Rank = 1;
    tensor.Sizes[0] = N;

    // Should handle non-aligned sizes correctly
    lora_dropout_scale(tensor, dropout_prob, 11111, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto h_result = from_device(d_data);
    REQUIRE(h_result.size() == static_cast<size_t>(N));

    // Verify determinism
    thrust::device_vector<float> d_data2 = to_device(h_input);
    Tensor tensor2;
    tensor2.Data = reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_data2.data()));
    tensor2.DType = ETensorDType::FP32;
    tensor2.Rank = 1;
    tensor2.Sizes[0] = N;

    lora_dropout_scale(tensor2, dropout_prob, 11111, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto h_result2 = from_device(d_data2);
    REQUIRE(vectors_equal(h_result, h_result2));

    CUDA_CHECK(cudaStreamDestroy(stream));
}
