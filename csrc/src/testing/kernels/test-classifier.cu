// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//



// Fused classifier kernel tests (forward loss + in-place backward dlogits)

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>

#include <cuda_bf16.h>

#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "kernels/kernels.h"
#include "../utilities/test_config.h"
#include "../utilities/test_utils.h"
#include "utilities/utils.h"

using namespace testing_utils;

namespace {

// CPU reference: per-row softmax + cross-entropy with optional ignore_index (-100)
// Also computes in-place dlogits = (softmax - one_hot) * dloss for first V entries of each row
static void classifier_cpu(float* losses, float* dlogits,
                           const float* logits, const int* targets,
                           int BT, int V, int P, float dloss)
{
    for (int r = 0; r < BT; ++r) {
        const int ix = targets[r];
        const float* row = logits + r * P;
        float* gout = dlogits + r * P;

        if (ix == -100) {
            // masked token: zero grads (first V entries) and zero loss
            for (int i = 0; i < V; ++i) gout[i] = 0.0f;
            losses[r] = 0.0f;
            continue;
        }

        // compute logsumexp and probs for first V entries
        float maxv = -INFINITY;
        for (int i = 0; i < V; ++i) maxv = std::max(maxv, row[i]);
        float sumexp = 0.0f;
        for (int i = 0; i < V; ++i) sumexp += std::exp(row[i] - maxv);
        float invsum = 1.0f / sumexp;
        // loss
        float prob_ix = std::exp(row[ix] - maxv) * invsum;
        losses[r] = -std::log(prob_ix);
        // grads
        for (int i = 0; i < V; ++i) {
            float prob = std::exp(row[i] - maxv) * invsum;
            float indicator = (i == ix) ? 1.0f : 0.0f;
            gout[i] = (prob - indicator) * dloss;
        }
    }
}

static int count_valid_tokens(const std::vector<int>& targets) {
    int count = 0;
    for (int t : targets) {
        if (t != -100) {
            ++count;
        }
    }
    return count;
}

} // namespace

TEST_CASE("fused classifier fp32 forward/backward matches CPU", "[kernels][classifier][fp32]") {
    const auto& cfg = testing_config::get_test_config();
    const int B = cfg.B;
    const int T = cfg.T;
    const int BT = B * T;

    // Choose V as multiple of 8 to avoid edge cases with vector tails in masked rows
    const int V = 151936;
    const int P = V; // stride equal to vocab for test simplicity
    const float dloss = 0.7f; // arbitrary upstream gradient scale

    // Host buffers
    std::vector<float> h_logits((size_t)BT * P);
    fill_normal(h_logits, 0.0f, 1.0f, /*seed*/ 424242ULL);

    std::vector<int> h_targets(BT);
    {
        std::mt19937 gen(1337);
        std::uniform_int_distribution<int> dist(0, V - 1);
        for (int i = 0; i < BT; ++i) h_targets[i] = dist(gen);
        // set a few to ignore_index = -100
        if (BT >= 3) {
            h_targets[1] = -100;
            h_targets[BT/2] = -100;
        }
    }

    // CPU reference
    std::vector<float> h_losses_ref(BT, 0.0f);
    std::vector<float> h_dlogits_ref((size_t)BT * P, 0.0f);
    classifier_cpu(h_losses_ref.data(), h_dlogits_ref.data(), h_logits.data(), h_targets.data(), BT, V, P, dloss);

    // Device buffers
    thrust::device_vector<float> d_logits = to_device(h_logits);
    thrust::device_vector<float> d_losses(BT, 0.0f);
    thrust::device_vector<int> d_targets = to_device(h_targets);
    thrust::device_vector<int> d_valid_token_count(1);

    // Run kernel: write_dlogits = true => logits become gradients
    fused_classifier(thrust::raw_pointer_cast(d_logits.data()),
                     thrust::raw_pointer_cast(d_losses.data()),
                     dloss,
                     thrust::raw_pointer_cast(d_targets.data()),
                     thrust::raw_pointer_cast(d_valid_token_count.data()),
                     BT, V, P, /*write_dlogits*/ true, /*stream*/ 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back
    std::vector<float> h_losses = from_device(d_losses);
    std::vector<float> h_dlogits = from_device(d_logits);
    int h_valid_token_count = from_device(d_valid_token_count)[0];

    // Compare
    int h_valid_token_count_ref = 0;
    for (int r = 0; r < BT; ++r) {
        if (h_targets[r] != -100) h_valid_token_count_ref++;
        // loss
        REQUIRE(h_losses[r] == Catch::Approx(h_losses_ref[r]).margin(1e-6f));
        // gradients (only first V entries are defined)
        for (int i = 0; i < V; ++i) {
            REQUIRE(h_dlogits[r * P + i] == Catch::Approx(h_dlogits_ref[r * P + i]).margin(5e-5f));
        }
    }
    REQUIRE(h_valid_token_count == h_valid_token_count_ref);
}

TEST_CASE("fused classifier bf16 forward/backward matches CPU (tolerant)", "[kernels][classifier][bf16]") {
    const auto& cfg = testing_config::get_test_config();
    const int B = cfg.B;
    const int T = cfg.T;
    const int BT = B * T;

    // V multiple of 8 (x128::size for bf16 is 8)
    const int V = 16384;
    const int P = V;
    const float dloss = 0.7f;

    // Host logits in float then convert to bf16 for device
    std::vector<float> h_logits_f((size_t)BT * P);
    fill_normal(h_logits_f, 0.0f, 1.0f, /*seed*/ 777ULL);
    h_logits_f = round_bf16(h_logits_f);

    std::vector<int> h_targets(BT);
    {
        std::mt19937 gen(2025);
        std::uniform_int_distribution<int> dist(0, V - 1);
        for (int i = 0; i < BT; ++i) h_targets[i] = dist(gen);
        if (BT >= 2) h_targets[BT - 1] = -100;
    }

    // CPU reference (computed in float)
    std::vector<float> h_losses_ref(BT, 0.0f);
    std::vector<float> h_dlogits_ref((size_t)BT * P, 0.0f);

    classifier_cpu(h_losses_ref.data(), h_dlogits_ref.data(), h_logits_f.data(), h_targets.data(), BT, V, P, dloss);

    // Device
    std::vector<nv_bfloat16> h_logits_bf16 = to_bf16(h_logits_f);
    thrust::device_vector<nv_bfloat16> d_logits = to_device(h_logits_bf16);
    thrust::device_vector<float> d_losses(BT, 0.0f);
    thrust::device_vector<int> d_targets = to_device(h_targets);
    thrust::device_vector<int> d_valid_token_count(1);

    fused_classifier(thrust::raw_pointer_cast(d_logits.data()),
                     thrust::raw_pointer_cast(d_losses.data()),
                     dloss,
                     thrust::raw_pointer_cast(d_targets.data()),
                     thrust::raw_pointer_cast(d_valid_token_count.data()),
                     BT, V, P, /*write_dlogits*/ true, /*stream*/ 0);

    // Copy back
    std::vector<float> h_losses = from_device(d_losses);
    std::vector<nv_bfloat16> h_dlogits_bf16 = from_device(d_logits);
    int h_valid_token_count = from_device(d_valid_token_count)[0];

    // Convert bf16 grads to float on host to compare
    auto bf16_to_float = [](nv_bfloat16 v) {
        uint16_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        return bf16_bits_to_float(bits);
    };

    // Compare with looser tolerances due to bf16 rounding
    int h_valid_token_count_ref = 0;
    for (int r = 0; r < BT; ++r) {
        if (h_targets[r] != -100) h_valid_token_count_ref++;
        REQUIRE(h_losses[r] == Catch::Approx(h_losses_ref[r]).margin(5e-3f));
        for (int i = 0; i < V; ++i) {
            float got = bf16_to_float(h_dlogits_bf16[r * P + i]);
            float exp = h_dlogits_ref[r * P + i];
            REQUIRE(got == Catch::Approx(exp).margin(2e-3f));
        }
    }
    REQUIRE(h_valid_token_count == h_valid_token_count_ref);
}

TEST_CASE("chunked cross-entropy fp32 large vocab (V=128K)", "[kernels][classifier][chunked][fp32]") {
    const int B = 1;
    const int T = 4;
    const int BT = B * T;
    const int V = 131072;
    const int P = V;
    const int n_chunks = (V + CROSS_ENTROPY_MAX_FUSED_SIZE - 1) / CROSS_ENTROPY_MAX_FUSED_SIZE;
    const float dloss = 0.5f;

    std::vector<float> h_logits((size_t)BT * P);
    fill_normal(h_logits, 0.0f, 1.0f, /*seed*/ 12345ULL);

    std::vector<int> h_targets(BT);
    {
        std::mt19937 gen(4242);
        std::uniform_int_distribution<int> dist(0, V - 1);
        for (int i = 0; i < BT; ++i) h_targets[i] = dist(gen);
        h_targets[0] = -100;
    }

    std::vector<float> h_losses_ref(BT, 0.0f);
    std::vector<float> h_dlogits_ref((size_t)BT * P, 0.0f);
    classifier_cpu(h_losses_ref.data(), h_dlogits_ref.data(), h_logits.data(), h_targets.data(), BT, V, P, dloss);

    thrust::device_vector<float> d_logits = to_device(h_logits);
    thrust::device_vector<float> d_losses(BT, 0.0f);
    thrust::device_vector<float> d_logsumexp(BT, 0.0f);
    thrust::device_vector<float> d_chunk_lse((size_t)BT * n_chunks, 0.0f);
    thrust::device_vector<int> d_targets = to_device(h_targets);
    thrust::device_vector<int> d_valid_token_count(1, 0);
    thrust::device_vector<float> d_dloss(BT, dloss);
    thrust::device_vector<float> d_dlogits((size_t)BT * P, 0.0f);

    chunked_cross_entropy_forward(thrust::raw_pointer_cast(d_logits.data()),
                                  thrust::raw_pointer_cast(d_losses.data()),
                                  thrust::raw_pointer_cast(d_logsumexp.data()),
                                  thrust::raw_pointer_cast(d_chunk_lse.data()),
                                  thrust::raw_pointer_cast(d_targets.data()),
                                  thrust::raw_pointer_cast(d_valid_token_count.data()),
                                  /*correct_count=*/nullptr,
                                  BT, V, P, n_chunks, /*stream=*/0);

    chunked_cross_entropy_backward(thrust::raw_pointer_cast(d_dlogits.data()),
                                   thrust::raw_pointer_cast(d_logits.data()),
                                   thrust::raw_pointer_cast(d_logsumexp.data()),
                                   thrust::raw_pointer_cast(d_dloss.data()),
                                   thrust::raw_pointer_cast(d_targets.data()),
                                                                      BT, V, P, /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_losses = from_device(d_losses);
    std::vector<float> h_dlogits = from_device(d_dlogits);
    int h_valid_token_count = from_device(d_valid_token_count)[0];

    REQUIRE(h_valid_token_count == count_valid_tokens(h_targets));
    for (int r = 0; r < BT; ++r) {
        REQUIRE(h_losses[r] == Catch::Approx(h_losses_ref[r]).margin(1e-4f));
        for (int i = 0; i < V; ++i) {
            REQUIRE(h_dlogits[r * P + i] == Catch::Approx(h_dlogits_ref[r * P + i]).margin(5e-4f));
        }
    }
}

TEST_CASE("chunked cross-entropy fp32 very large vocab (V=256K)", "[kernels][classifier][chunked][fp32]") {
    const int B = 1;
    const int T = 4;
    const int BT = B * T;
    const int V = 262144;
    const int P = V;
    const int n_chunks = (V + CROSS_ENTROPY_MAX_FUSED_SIZE - 1) / CROSS_ENTROPY_MAX_FUSED_SIZE;
    const float dloss = 0.5f;

    std::vector<float> h_logits((size_t)BT * P);
    fill_normal(h_logits, 0.0f, 1.0f, /*seed*/ 54321ULL);

    std::vector<int> h_targets(BT);
    {
        std::mt19937 gen(31415);
        std::uniform_int_distribution<int> dist(0, V - 1);
        for (int i = 0; i < BT; ++i) h_targets[i] = dist(gen);
        h_targets[BT - 1] = -100;
    }

    std::vector<float> h_losses_ref(BT, 0.0f);
    std::vector<float> h_dlogits_ref((size_t)BT * P, 0.0f);
    classifier_cpu(h_losses_ref.data(), h_dlogits_ref.data(), h_logits.data(), h_targets.data(), BT, V, P, dloss);

    thrust::device_vector<float> d_logits = to_device(h_logits);
    thrust::device_vector<float> d_losses(BT, 0.0f);
    thrust::device_vector<float> d_logsumexp(BT, 0.0f);
    thrust::device_vector<float> d_chunk_lse((size_t)BT * n_chunks, 0.0f);
    thrust::device_vector<int> d_targets = to_device(h_targets);
    thrust::device_vector<int> d_valid_token_count(1, 0);
    thrust::device_vector<float> d_dloss(BT, dloss);
    thrust::device_vector<float> d_dlogits((size_t)BT * P, 0.0f);

    chunked_cross_entropy_forward(thrust::raw_pointer_cast(d_logits.data()),
                                  thrust::raw_pointer_cast(d_losses.data()),
                                  thrust::raw_pointer_cast(d_logsumexp.data()),
                                  thrust::raw_pointer_cast(d_chunk_lse.data()),
                                  thrust::raw_pointer_cast(d_targets.data()),
                                  thrust::raw_pointer_cast(d_valid_token_count.data()),
                                  /*correct_count=*/nullptr,
                                  BT, V, P, n_chunks, /*stream=*/0);

    chunked_cross_entropy_backward(thrust::raw_pointer_cast(d_dlogits.data()),
                                   thrust::raw_pointer_cast(d_logits.data()),
                                   thrust::raw_pointer_cast(d_logsumexp.data()),
                                   thrust::raw_pointer_cast(d_dloss.data()),
                                   thrust::raw_pointer_cast(d_targets.data()),
                                                                      BT, V, P, /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_losses = from_device(d_losses);
    std::vector<float> h_dlogits = from_device(d_dlogits);
    int h_valid_token_count = from_device(d_valid_token_count)[0];

    REQUIRE(h_valid_token_count == count_valid_tokens(h_targets));
    for (int r = 0; r < BT; ++r) {
        REQUIRE(h_losses[r] == Catch::Approx(h_losses_ref[r]).margin(1e-4f));
        for (int i = 0; i < V; ++i) {
            REQUIRE(h_dlogits[r * P + i] == Catch::Approx(h_dlogits_ref[r * P + i]).margin(5e-4f));
        }
    }
}

TEST_CASE("chunked cross-entropy bf16 large vocab", "[kernels][classifier][chunked][bf16]") {
    const int B = 1;
    const int T = 4;
    const int BT = B * T;
    const int V = 131072;
    const int P = V;
    const int n_chunks = (V + CROSS_ENTROPY_MAX_FUSED_SIZE - 1) / CROSS_ENTROPY_MAX_FUSED_SIZE;
    const float dloss = 0.5f;

    std::vector<float> h_logits_f((size_t)BT * P);
    fill_normal(h_logits_f, 0.0f, 1.0f, /*seed*/ 7777ULL);
    h_logits_f = round_bf16(h_logits_f);

    std::vector<int> h_targets(BT);
    {
        std::mt19937 gen(9876);
        std::uniform_int_distribution<int> dist(0, V - 1);
        for (int i = 0; i < BT; ++i) h_targets[i] = dist(gen);
        h_targets[0] = -100;
    }

    std::vector<float> h_losses_ref(BT, 0.0f);
    std::vector<float> h_dlogits_ref((size_t)BT * P, 0.0f);
    classifier_cpu(h_losses_ref.data(), h_dlogits_ref.data(), h_logits_f.data(), h_targets.data(), BT, V, P, dloss);

    std::vector<nv_bfloat16> h_logits_bf16 = to_bf16(h_logits_f);
    thrust::device_vector<nv_bfloat16> d_logits = to_device(h_logits_bf16);
    thrust::device_vector<float> d_losses(BT, 0.0f);
    thrust::device_vector<float> d_logsumexp(BT, 0.0f);
    thrust::device_vector<float> d_chunk_lse((size_t)BT * n_chunks, 0.0f);
    thrust::device_vector<int> d_targets = to_device(h_targets);
    thrust::device_vector<int> d_valid_token_count(1, 0);
    thrust::device_vector<float> d_dloss(BT, dloss);
    thrust::device_vector<nv_bfloat16> d_dlogits((size_t)BT * P);

    chunked_cross_entropy_forward(thrust::raw_pointer_cast(d_logits.data()),
                                  thrust::raw_pointer_cast(d_losses.data()),
                                  thrust::raw_pointer_cast(d_logsumexp.data()),
                                  thrust::raw_pointer_cast(d_chunk_lse.data()),
                                  thrust::raw_pointer_cast(d_targets.data()),
                                  thrust::raw_pointer_cast(d_valid_token_count.data()),
                                  /*correct_count=*/nullptr,
                                  BT, V, P, n_chunks, /*stream=*/0);

    chunked_cross_entropy_backward(thrust::raw_pointer_cast(d_dlogits.data()),
                                   thrust::raw_pointer_cast(d_logits.data()),
                                   thrust::raw_pointer_cast(d_logsumexp.data()),
                                   thrust::raw_pointer_cast(d_dloss.data()),
                                   thrust::raw_pointer_cast(d_targets.data()),
                                                                      BT, V, P, /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_losses = from_device(d_losses);
    std::vector<nv_bfloat16> h_dlogits_bf16 = from_device(d_dlogits);
    int h_valid_token_count = from_device(d_valid_token_count)[0];

    auto bf16_to_float = [](nv_bfloat16 v) {
        uint16_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        return bf16_bits_to_float(bits);
    };

    REQUIRE(h_valid_token_count == count_valid_tokens(h_targets));
    for (int r = 0; r < BT; ++r) {
        REQUIRE(h_losses[r] == Catch::Approx(h_losses_ref[r]).margin(5e-3f));
        for (int i = 0; i < V; ++i) {
            float got = bf16_to_float(h_dlogits_bf16[r * P + i]);
            float exp = h_dlogits_ref[r * P + i];
            REQUIRE(got == Catch::Approx(exp).margin(1e-2f));
        }
    }
}

TEST_CASE("chunked cross-entropy matches fused at boundary vocab", "[kernels][classifier][chunked]") {
    const int B = 1;
    const int T = 4;
    const int BT = B * T;
    const int V = CROSS_ENTROPY_MAX_FUSED_SIZE;
    const int P = V;
    const int n_chunks = (V + CROSS_ENTROPY_MAX_FUSED_SIZE - 1) / CROSS_ENTROPY_MAX_FUSED_SIZE;
    const float dloss = 0.7f;

    std::vector<float> h_logits((size_t)BT * P);
    fill_normal(h_logits, 0.0f, 1.0f, /*seed*/ 2026ULL);

    std::vector<int> h_targets(BT);
    {
        std::mt19937 gen(2026);
        std::uniform_int_distribution<int> dist(0, V - 1);
        for (int i = 0; i < BT; ++i) h_targets[i] = dist(gen);
        h_targets[0] = -100;
    }

    thrust::device_vector<int> d_targets = to_device(h_targets);
    thrust::device_vector<float> d_dloss(BT, dloss);

    thrust::device_vector<float> d_logits_fused = to_device(h_logits);
    thrust::device_vector<float> d_losses_fused(BT, 0.0f);
    thrust::device_vector<float> d_logsumexp_fused(BT, 0.0f);
    thrust::device_vector<int> d_valid_fused(1, 0);
    thrust::device_vector<float> d_dlogits_fused((size_t)BT * P, 0.0f);

    fused_cross_entropy_forward(thrust::raw_pointer_cast(d_logits_fused.data()),
                                thrust::raw_pointer_cast(d_losses_fused.data()),
                                thrust::raw_pointer_cast(d_logsumexp_fused.data()),
                                thrust::raw_pointer_cast(d_targets.data()),
                                thrust::raw_pointer_cast(d_valid_fused.data()),
                                /*correct_count=*/nullptr,
                                BT, V, P, /*stream=*/0);

    fused_cross_entropy_backward(thrust::raw_pointer_cast(d_dlogits_fused.data()),
                                 thrust::raw_pointer_cast(d_logits_fused.data()),
                                 thrust::raw_pointer_cast(d_logsumexp_fused.data()),
                                 thrust::raw_pointer_cast(d_dloss.data()),
                                 thrust::raw_pointer_cast(d_targets.data()),
                                                                  BT, V, P, /*stream=*/0);

    thrust::device_vector<float> d_logits_chunked = to_device(h_logits);
    thrust::device_vector<float> d_losses_chunked(BT, 0.0f);
    thrust::device_vector<float> d_logsumexp_chunked(BT, 0.0f);
    thrust::device_vector<float> d_chunk_lse((size_t)BT * n_chunks, 0.0f);
    thrust::device_vector<int> d_valid_chunked(1, 0);
    thrust::device_vector<float> d_dlogits_chunked((size_t)BT * P, 0.0f);

    chunked_cross_entropy_forward(thrust::raw_pointer_cast(d_logits_chunked.data()),
                                  thrust::raw_pointer_cast(d_losses_chunked.data()),
                                  thrust::raw_pointer_cast(d_logsumexp_chunked.data()),
                                  thrust::raw_pointer_cast(d_chunk_lse.data()),
                                  thrust::raw_pointer_cast(d_targets.data()),
                                  thrust::raw_pointer_cast(d_valid_chunked.data()),
                                  /*correct_count=*/nullptr,
                                  BT, V, P, n_chunks, /*stream=*/0);

    chunked_cross_entropy_backward(thrust::raw_pointer_cast(d_dlogits_chunked.data()),
                                   thrust::raw_pointer_cast(d_logits_chunked.data()),
                                   thrust::raw_pointer_cast(d_logsumexp_chunked.data()),
                                   thrust::raw_pointer_cast(d_dloss.data()),
                                   thrust::raw_pointer_cast(d_targets.data()),
                                                                      BT, V, P, /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_losses_fused = from_device(d_losses_fused);
    std::vector<float> h_losses_chunked = from_device(d_losses_chunked);
    std::vector<float> h_dlogits_fused = from_device(d_dlogits_fused);
    std::vector<float> h_dlogits_chunked = from_device(d_dlogits_chunked);
    int h_valid_fused = from_device(d_valid_fused)[0];
    int h_valid_chunked = from_device(d_valid_chunked)[0];

    REQUIRE(h_valid_fused == h_valid_chunked);
    for (int r = 0; r < BT; ++r) {
        REQUIRE(h_losses_fused[r] == Catch::Approx(h_losses_chunked[r]).margin(1e-4f));
        for (int i = 0; i < V; ++i) {
            REQUIRE(h_dlogits_fused[r * P + i] == Catch::Approx(h_dlogits_chunked[r * P + i]).margin(5e-4f));
        }
    }
}
