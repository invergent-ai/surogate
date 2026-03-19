// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Tests for sampling kernels (FlashInfer wrappers).

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "kernels/sampling.h"
#include "utilities/utils.h"

namespace {

// CPU softmax for reference
std::vector<float> cpu_softmax(const std::vector<float>& logits, int batch_size, int vocab_size) {
    std::vector<float> probs(logits.size());
    for (int b = 0; b < batch_size; ++b) {
        const float* row = logits.data() + static_cast<std::size_t>(b) * vocab_size;
        float* out = probs.data() + static_cast<std::size_t>(b) * vocab_size;

        float max_val = *std::max_element(row, row + vocab_size);
        float sum = 0.0f;
        for (int v = 0; v < vocab_size; ++v) {
            out[v] = std::exp(row[v] - max_val);
            sum += out[v];
        }
        for (int v = 0; v < vocab_size; ++v) {
            out[v] /= sum;
        }
    }
    return probs;
}

// CPU argmax for reference
std::vector<int32_t> cpu_argmax(const std::vector<float>& logits, int batch_size, int vocab_size) {
    std::vector<int32_t> tokens(static_cast<std::size_t>(batch_size));
    for (int b = 0; b < batch_size; ++b) {
        const float* row = logits.data() + static_cast<std::size_t>(b) * vocab_size;
        tokens[static_cast<std::size_t>(b)] = static_cast<int32_t>(
            std::max_element(row, row + vocab_size) - row);
    }
    return tokens;
}

}  // anonymous namespace

TEST_CASE("Softmax correctness", "[sampling][softmax]") {
    const int batch_size = 4;
    const int vocab_size = 1024;
    const std::size_t total = static_cast<std::size_t>(batch_size) * vocab_size;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::vector<float> logits_host(total);
    for (auto& v : logits_host) v = dist(rng);

    thrust::device_vector<float> logits_dev(logits_host.begin(), logits_host.end());
    thrust::device_vector<float> probs_dev(total);

    sampling_softmax(
        thrust::raw_pointer_cast(logits_dev.data()),
        thrust::raw_pointer_cast(probs_dev.data()),
        /*temperature=*/nullptr,
        batch_size, vocab_size,
        /*workspace=*/nullptr, 0,
        nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> probs_gpu(total);
    thrust::copy(probs_dev.begin(), probs_dev.end(), probs_gpu.begin());

    auto probs_cpu = cpu_softmax(logits_host, batch_size, vocab_size);

    // Check sums to 1.0
    for (int b = 0; b < batch_size; ++b) {
        float sum = 0.0f;
        for (int v = 0; v < vocab_size; ++v) {
            sum += probs_gpu[static_cast<std::size_t>(b) * vocab_size + v];
        }
        REQUIRE(std::abs(sum - 1.0f) < 0.01f);
    }

    // Check individual values
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < total; ++i) {
        max_diff = std::max(max_diff, std::abs(probs_gpu[i] - probs_cpu[i]));
    }
    INFO("Max softmax diff: " << max_diff);
    REQUIRE(max_diff < 0.001f);
}

TEST_CASE("Argmax correctness", "[sampling][argmax]") {
    const int batch_size = 8;
    const int vocab_size = 2048;
    const std::size_t total = static_cast<std::size_t>(batch_size) * vocab_size;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> logits_host(total);
    for (auto& v : logits_host) v = dist(rng);

    thrust::device_vector<float> logits_dev(logits_host.begin(), logits_host.end());
    thrust::device_vector<int32_t> output_dev(static_cast<std::size_t>(batch_size));

    sampling_argmax(
        thrust::raw_pointer_cast(logits_dev.data()),
        thrust::raw_pointer_cast(output_dev.data()),
        batch_size, vocab_size,
        nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> output_gpu(static_cast<std::size_t>(batch_size));
    thrust::copy(output_dev.begin(), output_dev.end(), output_gpu.begin());

    auto expected = cpu_argmax(logits_host, batch_size, vocab_size);

    for (int b = 0; b < batch_size; ++b) {
        REQUIRE(output_gpu[static_cast<std::size_t>(b)] == expected[static_cast<std::size_t>(b)]);
    }
}

TEST_CASE("Logprob extraction correctness", "[sampling][logprob]") {
    const int batch_size = 4;
    const int vocab_size = 512;
    const std::size_t total = static_cast<std::size_t>(batch_size) * vocab_size;

    // Create probability distribution (must sum to 1 per row)
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(0.001f, 1.0f);
    std::vector<float> probs_host(total);
    for (int b = 0; b < batch_size; ++b) {
        float sum = 0.0f;
        for (int v = 0; v < vocab_size; ++v) {
            probs_host[static_cast<std::size_t>(b) * vocab_size + v] = dist(rng);
            sum += probs_host[static_cast<std::size_t>(b) * vocab_size + v];
        }
        for (int v = 0; v < vocab_size; ++v) {
            probs_host[static_cast<std::size_t>(b) * vocab_size + v] /= sum;
        }
    }

    // Token IDs to extract logprobs for
    std::vector<int32_t> token_ids = {10, 200, 0, 511};

    thrust::device_vector<float> probs_dev(probs_host.begin(), probs_host.end());
    thrust::device_vector<int32_t> tokens_dev(token_ids.begin(), token_ids.end());
    thrust::device_vector<float> logprobs_dev(static_cast<std::size_t>(batch_size));

    sampling_extract_logprob(
        thrust::raw_pointer_cast(probs_dev.data()),
        thrust::raw_pointer_cast(tokens_dev.data()),
        thrust::raw_pointer_cast(logprobs_dev.data()),
        batch_size, vocab_size,
        nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> logprobs_gpu(static_cast<std::size_t>(batch_size));
    thrust::copy(logprobs_dev.begin(), logprobs_dev.end(), logprobs_gpu.begin());

    for (int b = 0; b < batch_size; ++b) {
        float expected = std::log(probs_host[static_cast<std::size_t>(b) * vocab_size + token_ids[static_cast<std::size_t>(b)]]);
        REQUIRE(std::abs(logprobs_gpu[static_cast<std::size_t>(b)] - expected) < 0.001f);
    }
}

TEST_CASE("Categorical sampling produces valid tokens", "[sampling][categorical]") {
    const int batch_size = 16;
    const int vocab_size = 1024;
    const std::size_t total = static_cast<std::size_t>(batch_size) * vocab_size;

    // Create uniform probability distribution
    std::vector<float> probs_host(total, 1.0f / vocab_size);

    thrust::device_vector<float> probs_dev(probs_host.begin(), probs_host.end());
    thrust::device_vector<int32_t> output_dev(static_cast<std::size_t>(batch_size));

    sampling_from_probs(
        thrust::raw_pointer_cast(probs_dev.data()),
        thrust::raw_pointer_cast(output_dev.data()),
        batch_size, vocab_size,
        /*deterministic=*/true,
        /*seed=*/42, /*offset=*/0,
        nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> output_gpu(static_cast<std::size_t>(batch_size));
    thrust::copy(output_dev.begin(), output_dev.end(), output_gpu.begin());

    // All sampled tokens should be in [0, vocab_size)
    for (int b = 0; b < batch_size; ++b) {
        REQUIRE(output_gpu[static_cast<std::size_t>(b)] >= 0);
        REQUIRE(output_gpu[static_cast<std::size_t>(b)] < vocab_size);
    }
}

TEST_CASE("Deterministic sampling is reproducible", "[sampling][deterministic]") {
    const int batch_size = 8;
    const int vocab_size = 2048;
    const std::size_t total = static_cast<std::size_t>(batch_size) * vocab_size;

    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::vector<float> logits_host(total);
    for (auto& v : logits_host) v = dist(rng);

    thrust::device_vector<float> logits_dev(logits_host.begin(), logits_host.end());
    thrust::device_vector<int32_t> output1(static_cast<std::size_t>(batch_size));
    thrust::device_vector<int32_t> output2(static_cast<std::size_t>(batch_size));

    // Sample twice with same seed
    sampling_from_logits(
        thrust::raw_pointer_cast(logits_dev.data()),
        thrust::raw_pointer_cast(output1.data()),
        batch_size, vocab_size,
        /*deterministic=*/true, /*seed=*/42, /*offset=*/0, nullptr);

    // Re-upload logits (may have been modified in-place by Gumbel noise)
    thrust::copy(logits_host.begin(), logits_host.end(), logits_dev.begin());

    sampling_from_logits(
        thrust::raw_pointer_cast(logits_dev.data()),
        thrust::raw_pointer_cast(output2.data()),
        batch_size, vocab_size,
        /*deterministic=*/true, /*seed=*/42, /*offset=*/0, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> out1(static_cast<std::size_t>(batch_size));
    std::vector<int32_t> out2(static_cast<std::size_t>(batch_size));
    thrust::copy(output1.begin(), output1.end(), out1.begin());
    thrust::copy(output2.begin(), output2.end(), out2.begin());

    for (int b = 0; b < batch_size; ++b) {
        REQUIRE(out1[static_cast<std::size_t>(b)] == out2[static_cast<std::size_t>(b)]);
    }
}
