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

TEST_CASE("Top-K sampling produces valid tokens from top K", "[sampling][topk]") {
    const int batch_size = 4;
    const int vocab_size = 1024;
    const int top_k = 5;
    const std::size_t total = static_cast<std::size_t>(batch_size) * vocab_size;

    // Create a peaked distribution: one token has prob 0.5, rest uniform
    std::vector<float> probs_host(total);
    for (int b = 0; b < batch_size; ++b) {
        float uniform = 0.5f / (vocab_size - 1);
        for (int v = 0; v < vocab_size; ++v) {
            probs_host[static_cast<std::size_t>(b) * vocab_size + v] = uniform;
        }
        // Make token (b*100) the highest prob
        probs_host[static_cast<std::size_t>(b) * vocab_size + b * 100] = 0.5f;
    }

    thrust::device_vector<float> probs_dev(probs_host.begin(), probs_host.end());
    thrust::device_vector<int32_t> output_dev(static_cast<std::size_t>(batch_size));

    sampling_top_k(
        thrust::raw_pointer_cast(probs_dev.data()),
        thrust::raw_pointer_cast(output_dev.data()),
        top_k, batch_size, vocab_size,
        true, 42, 0, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> output_gpu(static_cast<std::size_t>(batch_size));
    thrust::copy(output_dev.begin(), output_dev.end(), output_gpu.begin());

    for (int b = 0; b < batch_size; ++b) {
        REQUIRE(output_gpu[static_cast<std::size_t>(b)] >= 0);
        REQUIRE(output_gpu[static_cast<std::size_t>(b)] < vocab_size);
    }
}

TEST_CASE("Top-P sampling produces valid tokens", "[sampling][topp]") {
    const int batch_size = 4;
    const int vocab_size = 512;
    const float top_p = 0.9f;
    const std::size_t total = static_cast<std::size_t>(batch_size) * vocab_size;

    // Uniform distribution
    std::vector<float> probs_host(total, 1.0f / vocab_size);
    thrust::device_vector<float> probs_dev(probs_host.begin(), probs_host.end());
    thrust::device_vector<int32_t> output_dev(static_cast<std::size_t>(batch_size));

    sampling_top_p(
        thrust::raw_pointer_cast(probs_dev.data()),
        thrust::raw_pointer_cast(output_dev.data()),
        top_p, batch_size, vocab_size,
        true, 42, 0, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> output_gpu(static_cast<std::size_t>(batch_size));
    thrust::copy(output_dev.begin(), output_dev.end(), output_gpu.begin());

    for (int b = 0; b < batch_size; ++b) {
        REQUIRE(output_gpu[static_cast<std::size_t>(b)] >= 0);
        REQUIRE(output_gpu[static_cast<std::size_t>(b)] < vocab_size);
    }
}

TEST_CASE("Top-K + Top-P combined sampling", "[sampling][topk_topp]") {
    const int batch_size = 4;
    const int vocab_size = 256;
    const int top_k = 10;
    const float top_p = 0.8f;
    const std::size_t total = static_cast<std::size_t>(batch_size) * vocab_size;

    std::vector<float> probs_host(total, 1.0f / vocab_size);
    thrust::device_vector<float> probs_dev(probs_host.begin(), probs_host.end());
    thrust::device_vector<int32_t> output_dev(static_cast<std::size_t>(batch_size));

    sampling_top_k_top_p(
        thrust::raw_pointer_cast(probs_dev.data()),
        thrust::raw_pointer_cast(output_dev.data()),
        top_k, top_p, batch_size, vocab_size,
        true, 42, 0, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> output_gpu(static_cast<std::size_t>(batch_size));
    thrust::copy(output_dev.begin(), output_dev.end(), output_gpu.begin());

    for (int b = 0; b < batch_size; ++b) {
        REQUIRE(output_gpu[static_cast<std::size_t>(b)] >= 0);
        REQUIRE(output_gpu[static_cast<std::size_t>(b)] < vocab_size);
    }
}

TEST_CASE("Min-P sampling produces valid tokens", "[sampling][minp]") {
    const int batch_size = 4;
    const int vocab_size = 256;
    const float min_p = 0.1f;
    const std::size_t total = static_cast<std::size_t>(batch_size) * vocab_size;

    // Create distribution where most tokens have very low prob
    std::mt19937 rng(42);
    std::vector<float> probs_host(total);
    for (int b = 0; b < batch_size; ++b) {
        float sum = 0.0f;
        for (int v = 0; v < vocab_size; ++v) {
            float p = (v < 5) ? 1.0f : 0.001f;
            probs_host[static_cast<std::size_t>(b) * vocab_size + v] = p;
            sum += p;
        }
        for (int v = 0; v < vocab_size; ++v)
            probs_host[static_cast<std::size_t>(b) * vocab_size + v] /= sum;
    }

    thrust::device_vector<float> probs_dev(probs_host.begin(), probs_host.end());
    thrust::device_vector<int32_t> output_dev(static_cast<std::size_t>(batch_size));

    sampling_min_p(
        thrust::raw_pointer_cast(probs_dev.data()),
        thrust::raw_pointer_cast(output_dev.data()),
        min_p, batch_size, vocab_size,
        true, 42, 0, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> output_gpu(static_cast<std::size_t>(batch_size));
    thrust::copy(output_dev.begin(), output_dev.end(), output_gpu.begin());

    // With min_p=0.1 and peaked distribution, sampled tokens should be in the top-5
    for (int b = 0; b < batch_size; ++b) {
        REQUIRE(output_gpu[static_cast<std::size_t>(b)] >= 0);
        REQUIRE(output_gpu[static_cast<std::size_t>(b)] < vocab_size);
        // High probability tokens are indices 0-4
        REQUIRE(output_gpu[static_cast<std::size_t>(b)] < 5);
    }
}

TEST_CASE("Repetition penalty modifies logits correctly", "[sampling][rep_penalty]") {
    const int batch_size = 2;
    const int vocab_size = 16;
    const int max_seq_len = 8;
    const float penalty = 2.0f;

    // Create logits: all 1.0
    std::vector<float> logits_host(static_cast<std::size_t>(batch_size) * vocab_size, 1.0f);
    // Also add some negative logits
    logits_host[3] = -1.0f;   // batch 0, token 3
    logits_host[vocab_size + 5] = -0.5f;  // batch 1, token 5

    thrust::device_vector<float> logits_dev(logits_host.begin(), logits_host.end());

    // Token history: batch 0 used tokens [1, 3, 5], batch 1 used [2, 5]
    std::vector<int32_t> token_ids(static_cast<std::size_t>(batch_size) * max_seq_len, 0);
    token_ids[0] = 1; token_ids[1] = 3; token_ids[2] = 5;  // batch 0
    token_ids[max_seq_len] = 2; token_ids[max_seq_len + 1] = 5;  // batch 1

    thrust::device_vector<int32_t> tokens_dev(token_ids.begin(), token_ids.end());

    std::vector<int> seq_lens = {3, 2};
    thrust::device_vector<int> seq_lens_dev(seq_lens.begin(), seq_lens.end());

    sampling_repetition_penalty(
        thrust::raw_pointer_cast(logits_dev.data()),
        thrust::raw_pointer_cast(tokens_dev.data()),
        thrust::raw_pointer_cast(seq_lens_dev.data()),
        penalty, batch_size, vocab_size, max_seq_len, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> result(logits_host.size());
    thrust::copy(logits_dev.begin(), logits_dev.end(), result.begin());

    // Batch 0: tokens 1, 3, 5 should be penalized
    // Token 0: untouched → 1.0
    REQUIRE(result[0] == Catch::Approx(1.0f));
    // Token 1: positive, divided by penalty → 1.0/2.0 = 0.5
    REQUIRE(result[1] == Catch::Approx(0.5f));
    // Token 3: negative, multiplied by penalty → -1.0*2.0 = -2.0
    REQUIRE(result[3] == Catch::Approx(-2.0f));
    // Token 5: positive, divided → 0.5
    REQUIRE(result[5] == Catch::Approx(0.5f));
    // Token 7: untouched → 1.0
    REQUIRE(result[7] == Catch::Approx(1.0f));

    // Batch 1: tokens 2, 5 penalized
    // Token 2: 1.0/2.0 = 0.5
    REQUIRE(result[vocab_size + 2] == Catch::Approx(0.5f));
    // Token 5: negative (-0.5) * 2.0 = -1.0
    REQUIRE(result[vocab_size + 5] == Catch::Approx(-1.0f));
    // Token 0: untouched → 1.0
    REQUIRE(result[vocab_size + 0] == Catch::Approx(1.0f));
}

TEST_CASE("Repetition penalty=1.0 is no-op", "[sampling][rep_penalty]") {
    const int batch_size = 1;
    const int vocab_size = 8;

    std::vector<float> logits_host = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    thrust::device_vector<float> logits_dev(logits_host.begin(), logits_host.end());

    std::vector<int32_t> token_ids = {0, 1, 2, 3, 0, 0, 0, 0};
    thrust::device_vector<int32_t> tokens_dev(token_ids.begin(), token_ids.end());
    std::vector<int> seq_lens = {4};
    thrust::device_vector<int> seq_lens_dev(seq_lens.begin(), seq_lens.end());

    sampling_repetition_penalty(
        thrust::raw_pointer_cast(logits_dev.data()),
        thrust::raw_pointer_cast(tokens_dev.data()),
        thrust::raw_pointer_cast(seq_lens_dev.data()),
        1.0f, batch_size, vocab_size, 8, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> result(logits_host.size());
    thrust::copy(logits_dev.begin(), logits_dev.end(), result.begin());

    // All logits should be unchanged
    for (int i = 0; i < vocab_size; ++i) {
        REQUIRE(result[static_cast<std::size_t>(i)] == Catch::Approx(logits_host[static_cast<std::size_t>(i)]));
    }
}
