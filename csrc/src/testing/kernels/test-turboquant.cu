// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <cmath>
#include <random>
#include <numeric>

#include <cuda_bf16.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "kernels/turboquant.h"
#include "kernels/kernels.h"
#include "utilities/utils.h"
#include "../utilities/test_utils.h"

using namespace testing_utils;

namespace {

/// Generate N random unit-norm vectors of dimension D.
std::vector<float> random_unit_vectors(int N, int D, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(N * D);
    for (auto& v : data) v = dist(rng);
    for (int i = 0; i < N; i++) {
        float norm = 0.0f;
        for (int j = 0; j < D; j++) norm += data[i * D + j] * data[i * D + j];
        norm = std::sqrt(norm);
        for (int j = 0; j < D; j++) data[i * D + j] /= norm;
    }
    return data;
}

/// Generate N random vectors with varying norms.
std::vector<float> random_vectors(int N, int D, uint64_t seed = 123) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(N * D);
    for (auto& v : data) v = dist(rng);
    return data;
}

float dot_product(const float* a, const float* b, int D) {
    float sum = 0.0f;
    for (int i = 0; i < D; i++) sum += a[i] * b[i];
    return sum;
}

float vec_norm(const float* a, int D) {
    return std::sqrt(dot_product(a, a, D));
}

/// Softmax in-place over n elements.
void softmax(float* x, int n) {
    float max_val = *std::max_element(x, x + n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = std::exp(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/// KL divergence D_KL(p || q) = sum p[i] * log(p[i] / q[i]).
double kl_divergence(const float* p, const float* q, int n) {
    double kl = 0.0;
    for (int i = 0; i < n; i++) {
        if (p[i] > 1e-12f && q[i] > 1e-12f)
            kl += (double)p[i] * std::log((double)p[i] / (double)q[i]);
    }
    return kl;
}

/// Cosine similarity between two vectors.
double cosine_sim(const float* a, const float* b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

/// Run TurboQuant quantize + dequantize round-trip on device.
std::vector<float> turboquant_round_trip(
    const std::vector<float>& input_f, int N, int D, uint64_t seed)
{
    constexpr int WARPS = 4;
    int PACKED = turboquant::packed_size(D);

    std::vector<uint32_t> h_d1(D / 32), h_d2(D / 32);
    turboquant::generate_signs_host(h_d1.data(), h_d2.data(), D, seed);

    auto h_bf = to_bf16(input_f);
    thrust::device_vector<nv_bfloat16> d_in(h_bf.begin(), h_bf.end());
    thrust::device_vector<uint8_t> d_packed(N * PACKED);
    thrust::device_vector<nv_bfloat16> d_out(N * D);
    thrust::device_vector<uint32_t> d_d1(h_d1.begin(), h_d1.end());
    thrust::device_vector<uint32_t> d_d2(h_d2.begin(), h_d2.end());

    turboquant::quantize(
        thrust::raw_pointer_cast(d_packed.data()),
        thrust::raw_pointer_cast(d_in.data()),
        thrust::raw_pointer_cast(d_d1.data()),
        thrust::raw_pointer_cast(d_d2.data()),
        N, D, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    turboquant::dequantize(
        thrust::raw_pointer_cast(d_out.data()),
        thrust::raw_pointer_cast(d_packed.data()),
        thrust::raw_pointer_cast(d_d1.data()),
        thrust::raw_pointer_cast(d_d2.data()),
        N, D, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<nv_bfloat16> h_out_bf(N * D);
    thrust::copy(d_out.begin(), d_out.end(), h_out_bf.begin());

    std::vector<float> result(N * D);
    for (int i = 0; i < N * D; i++)
        result[i] = bf16_bits_to_float(*reinterpret_cast<uint16_t*>(&h_out_bf[i]));
    return result;
}

}  // namespace

TEST_CASE("TurboQuant packed size", "[turboquant]") {
    REQUIRE(turboquant::packed_size(64)  == 32);
    REQUIRE(turboquant::packed_size(128) == 60);
    REQUIRE(turboquant::packed_size(256) == 116);
}

TEST_CASE("TurboQuant round-trip MSE (unit vectors, D=128)", "[turboquant]") {
    constexpr int D = 128;
    constexpr int N = 1024;
    constexpr int PACKED = turboquant::packed_size(D);

    // Generate random signs
    std::vector<uint32_t> h_d1(D / 32), h_d2(D / 32);
    turboquant::generate_signs_host(h_d1.data(), h_d2.data(), D, 0xCAFE);

    // Generate random unit vectors
    auto h_input_f = random_unit_vectors(N, D);
    auto h_input_bf = to_bf16(h_input_f);

    // Device allocations
    thrust::device_vector<nv_bfloat16> d_input(h_input_bf.begin(), h_input_bf.end());
    thrust::device_vector<uint8_t> d_packed(N * PACKED);
    thrust::device_vector<nv_bfloat16> d_output(N * D);
    thrust::device_vector<uint32_t> d_d1(h_d1.begin(), h_d1.end());
    thrust::device_vector<uint32_t> d_d2(h_d2.begin(), h_d2.end());

    // Quantize
    turboquant::quantize(
        thrust::raw_pointer_cast(d_packed.data()),
        thrust::raw_pointer_cast(d_input.data()),
        thrust::raw_pointer_cast(d_d1.data()),
        thrust::raw_pointer_cast(d_d2.data()),
        N, D, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Dequantize
    turboquant::dequantize(
        thrust::raw_pointer_cast(d_output.data()),
        thrust::raw_pointer_cast(d_packed.data()),
        thrust::raw_pointer_cast(d_d1.data()),
        thrust::raw_pointer_cast(d_d2.data()),
        N, D, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back
    std::vector<nv_bfloat16> h_output_bf(N * D);
    thrust::copy(d_output.begin(), d_output.end(), h_output_bf.begin());

    // Compute MSE
    double total_mse = 0.0;
    for (int i = 0; i < N; i++) {
        double vec_mse = 0.0;
        for (int j = 0; j < D; j++) {
            float orig = h_input_f[i * D + j];
            float recon = bf16_bits_to_float(*reinterpret_cast<uint16_t*>(&h_output_bf[i * D + j]));
            double diff = orig - recon;
            vec_mse += diff * diff;
        }
        total_mse += vec_mse;
    }
    double avg_mse = total_mse / N;

    // TurboQuant 3.5-bit MSE should be well under 0.5 per unit vector.
    // Theoretical: mixed 2.5-bit MSE ≈ 0.074, QJL adds some noise but also corrects.
    INFO("Average MSE per vector: " << avg_mse);
    REQUIRE(avg_mse < 0.25);
}

TEST_CASE("TurboQuant round-trip MSE (non-unit vectors, D=128)", "[turboquant]") {
    constexpr int D = 128;
    constexpr int N = 512;
    constexpr int PACKED = turboquant::packed_size(D);

    std::vector<uint32_t> h_d1(D / 32), h_d2(D / 32);
    turboquant::generate_signs_host(h_d1.data(), h_d2.data(), D, 0xBEEF);

    auto h_input_f = random_vectors(N, D);
    auto h_input_bf = to_bf16(h_input_f);

    thrust::device_vector<nv_bfloat16> d_input(h_input_bf.begin(), h_input_bf.end());
    thrust::device_vector<uint8_t> d_packed(N * PACKED);
    thrust::device_vector<nv_bfloat16> d_output(N * D);
    thrust::device_vector<uint32_t> d_d1(h_d1.begin(), h_d1.end());
    thrust::device_vector<uint32_t> d_d2(h_d2.begin(), h_d2.end());

    turboquant::quantize(
        thrust::raw_pointer_cast(d_packed.data()),
        thrust::raw_pointer_cast(d_input.data()),
        thrust::raw_pointer_cast(d_d1.data()),
        thrust::raw_pointer_cast(d_d2.data()),
        N, D, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    turboquant::dequantize(
        thrust::raw_pointer_cast(d_output.data()),
        thrust::raw_pointer_cast(d_packed.data()),
        thrust::raw_pointer_cast(d_d1.data()),
        thrust::raw_pointer_cast(d_d2.data()),
        N, D, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<nv_bfloat16> h_output_bf(N * D);
    thrust::copy(d_output.begin(), d_output.end(), h_output_bf.begin());

    // Relative MSE per vector (MSE / ||x||^2)
    double total_rel_mse = 0.0;
    for (int i = 0; i < N; i++) {
        double vec_mse = 0.0, vec_norm_sq = 0.0;
        for (int j = 0; j < D; j++) {
            float orig = h_input_f[i * D + j];
            float recon = bf16_bits_to_float(*reinterpret_cast<uint16_t*>(&h_output_bf[i * D + j]));
            vec_norm_sq += orig * orig;
            double diff = orig - recon;
            vec_mse += diff * diff;
        }
        if (vec_norm_sq > 1e-12)
            total_rel_mse += vec_mse / vec_norm_sq;
    }
    double avg_rel_mse = total_rel_mse / N;

    INFO("Average relative MSE: " << avg_rel_mse);
    REQUIRE(avg_rel_mse < 0.25);
}

TEST_CASE("TurboQuant inner product preservation (D=128)", "[turboquant]") {
    constexpr int D = 128;
    constexpr int N = 2048;
    constexpr int PACKED = turboquant::packed_size(D);

    std::vector<uint32_t> h_d1(D / 32), h_d2(D / 32);
    turboquant::generate_signs_host(h_d1.data(), h_d2.data(), D, 0xDEAD);

    // Generate two sets of random unit vectors: x (to quantize) and y (query)
    auto h_x_f = random_unit_vectors(N, D, 100);
    auto h_y_f = random_unit_vectors(N, D, 200);
    auto h_x_bf = to_bf16(h_x_f);

    thrust::device_vector<nv_bfloat16> d_x(h_x_bf.begin(), h_x_bf.end());
    thrust::device_vector<uint8_t> d_packed(N * PACKED);
    thrust::device_vector<nv_bfloat16> d_x_recon(N * D);
    thrust::device_vector<uint32_t> d_d1(h_d1.begin(), h_d1.end());
    thrust::device_vector<uint32_t> d_d2(h_d2.begin(), h_d2.end());

    turboquant::quantize(
        thrust::raw_pointer_cast(d_packed.data()),
        thrust::raw_pointer_cast(d_x.data()),
        thrust::raw_pointer_cast(d_d1.data()),
        thrust::raw_pointer_cast(d_d2.data()),
        N, D, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    turboquant::dequantize(
        thrust::raw_pointer_cast(d_x_recon.data()),
        thrust::raw_pointer_cast(d_packed.data()),
        thrust::raw_pointer_cast(d_d1.data()),
        thrust::raw_pointer_cast(d_d2.data()),
        N, D, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<nv_bfloat16> h_x_recon_bf(N * D);
    thrust::copy(d_x_recon.begin(), d_x_recon.end(), h_x_recon_bf.begin());

    // Compute inner products: true vs reconstructed
    double sum_true_ip = 0.0, sum_recon_ip = 0.0;
    double sum_abs_error = 0.0;
    for (int i = 0; i < N; i++) {
        float true_ip = dot_product(&h_x_f[i * D], &h_y_f[i * D], D);

        // Convert reconstruction to float
        std::vector<float> recon_f(D);
        for (int j = 0; j < D; j++)
            recon_f[j] = bf16_bits_to_float(*reinterpret_cast<uint16_t*>(&h_x_recon_bf[i * D + j]));

        float recon_ip = dot_product(recon_f.data(), &h_y_f[i * D], D);

        sum_true_ip += true_ip;
        sum_recon_ip += recon_ip;
        sum_abs_error += std::abs(true_ip - recon_ip);
    }

    double mean_true  = sum_true_ip / N;
    double mean_recon = sum_recon_ip / N;
    double mean_abs_error = sum_abs_error / N;

    // Inner product should be approximately unbiased (QJL guarantees this)
    // The mean of reconstructed IPs should be close to mean of true IPs
    INFO("Mean true IP: " << mean_true << ", Mean recon IP: " << mean_recon);
    INFO("Mean absolute IP error: " << mean_abs_error);

    // Check unbiasedness: ratio should be close to 1.0
    // (with 2048 samples, Monte Carlo error is small)
    if (std::abs(mean_true) > 0.01) {
        double ratio = mean_recon / mean_true;
        REQUIRE(ratio > 0.8);
        REQUIRE(ratio < 1.2);
    }

    // Mean absolute error should be reasonable for 3.5-bit quantization
    REQUIRE(mean_abs_error < 0.3);
}

TEST_CASE("TurboQuant D=64", "[turboquant]") {
    constexpr int D = 64;
    constexpr int N = 256;
    constexpr int PACKED = turboquant::packed_size(D);

    std::vector<uint32_t> h_d1(D / 32), h_d2(D / 32);
    turboquant::generate_signs_host(h_d1.data(), h_d2.data(), D, 0xF00D);

    auto h_input_f = random_unit_vectors(N, D, 777);
    auto h_input_bf = to_bf16(h_input_f);

    thrust::device_vector<nv_bfloat16> d_input(h_input_bf.begin(), h_input_bf.end());
    thrust::device_vector<uint8_t> d_packed(N * PACKED);
    thrust::device_vector<nv_bfloat16> d_output(N * D);
    thrust::device_vector<uint32_t> d_d1(h_d1.begin(), h_d1.end());
    thrust::device_vector<uint32_t> d_d2(h_d2.begin(), h_d2.end());

    turboquant::quantize(
        thrust::raw_pointer_cast(d_packed.data()),
        thrust::raw_pointer_cast(d_input.data()),
        thrust::raw_pointer_cast(d_d1.data()),
        thrust::raw_pointer_cast(d_d2.data()),
        N, D, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    turboquant::dequantize(
        thrust::raw_pointer_cast(d_output.data()),
        thrust::raw_pointer_cast(d_packed.data()),
        thrust::raw_pointer_cast(d_d1.data()),
        thrust::raw_pointer_cast(d_d2.data()),
        N, D, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<nv_bfloat16> h_output_bf(N * D);
    thrust::copy(d_output.begin(), d_output.end(), h_output_bf.begin());

    double total_mse = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            float orig = h_input_f[i * D + j];
            float recon = bf16_bits_to_float(*reinterpret_cast<uint16_t*>(&h_output_bf[i * D + j]));
            double diff = orig - recon;
            total_mse += diff * diff;
        }
    }
    double avg_mse = total_mse / N;

    INFO("D=64 average MSE per vector: " << avg_mse);
    REQUIRE(avg_mse < 0.35);  // slightly higher for smaller D
}

TEST_CASE("TurboQuant attention accuracy vs FP16 (D=128)", "[turboquant]") {
    // Simulates multi-head attention: Q @ K^T → softmax → attention weights.
    // Compares FP16 KV cache (baseline) vs TurboQuant 3.5-bit KV cache.
    //
    // Setup: NUM_QUERIES query vectors attend over SEQ_LEN cached key vectors.
    // Measures:  cosine similarity of raw attention logits (Q@K^T)
    //            KL divergence of attention distributions (after softmax)
    //            max absolute error of attention weights

    constexpr int D = 128;          // head_dim
    constexpr int SEQ_LEN = 512;    // cached sequence length
    constexpr int NUM_QUERIES = 64; // number of query positions
    const float SCALE = 1.0f / std::sqrt((float)D);  // attention scale

    // Generate K cache vectors and Q vectors (realistic magnitudes)
    auto h_keys_f  = random_vectors(SEQ_LEN, D, 300);
    auto h_query_f = random_vectors(NUM_QUERIES, D, 400);

    // --- FP16 baseline: BF16 round-trip (what the serving engine does today) ---
    auto h_keys_bf16_vec = to_bf16(h_keys_f);
    std::vector<float> keys_fp16(SEQ_LEN * D);
    for (int i = 0; i < SEQ_LEN * D; i++)
        keys_fp16[i] = bf16_bits_to_float(
            *reinterpret_cast<uint16_t*>(&h_keys_bf16_vec[i]));

    // --- TurboQuant 3.5-bit ---
    auto keys_tq = turboquant_round_trip(h_keys_f, SEQ_LEN, D, 0xA77E);

    // --- Compute attention scores and distributions ---
    double total_logit_cosine = 0.0;
    double total_kl = 0.0;
    double max_attn_error = 0.0;
    double total_attn_abs_error = 0.0;

    for (int q = 0; q < NUM_QUERIES; q++) {
        const float* qvec = &h_query_f[q * D];

        // Compute logits: score[k] = scale * dot(query, key[k])
        std::vector<float> logits_fp16(SEQ_LEN), logits_tq(SEQ_LEN);
        for (int k = 0; k < SEQ_LEN; k++) {
            logits_fp16[k] = SCALE * dot_product(qvec, &keys_fp16[k * D], D);
            logits_tq[k]   = SCALE * dot_product(qvec, &keys_tq[k * D], D);
        }

        // Cosine similarity of raw logit vectors
        total_logit_cosine += cosine_sim(logits_fp16.data(), logits_tq.data(), SEQ_LEN);

        // Softmax → attention distributions
        std::vector<float> attn_fp16(logits_fp16), attn_tq(logits_tq);
        softmax(attn_fp16.data(), SEQ_LEN);
        softmax(attn_tq.data(), SEQ_LEN);

        // KL divergence: D_KL(fp16 || tq)
        total_kl += kl_divergence(attn_fp16.data(), attn_tq.data(), SEQ_LEN);

        // Max and mean absolute attention weight error
        for (int k = 0; k < SEQ_LEN; k++) {
            double err = std::abs((double)attn_fp16[k] - (double)attn_tq[k]);
            max_attn_error = std::max(max_attn_error, err);
            total_attn_abs_error += err;
        }
    }

    double avg_logit_cosine = total_logit_cosine / NUM_QUERIES;
    double avg_kl = total_kl / NUM_QUERIES;
    double avg_attn_mae = total_attn_abs_error / (NUM_QUERIES * SEQ_LEN);

    INFO("=== TurboQuant 3.5-bit vs FP16 (BF16) Attention Accuracy ===");
    INFO("  Logit cosine similarity : " << avg_logit_cosine << "  (1.0 = perfect)");
    INFO("  Attention KL divergence : " << avg_kl << "  (0.0 = perfect)");
    INFO("  Attention max abs error : " << max_attn_error);
    INFO("  Attention mean abs error: " << avg_attn_mae);
    INFO("  Compression             : 4.27x vs FP16, 2.13x vs FP8");

    // Logit vectors should be near-identical in direction
    // (random vectors are harder than real KV cache — real LLM heads have more structure)
    REQUIRE(avg_logit_cosine > 0.97);

    // Attention distribution KL should be small
    REQUIRE(avg_kl < 0.05);

    // Individual attention weights should not drift far
    REQUIRE(max_attn_error < 0.05);
}

TEST_CASE("TurboQuant zero vector", "[turboquant]") {
    constexpr int D = 128;
    constexpr int N = 1;
    constexpr int PACKED = turboquant::packed_size(D);

    std::vector<uint32_t> h_d1(D / 32, 0), h_d2(D / 32, 0);
    turboquant::generate_signs_host(h_d1.data(), h_d2.data(), D, 0);

    std::vector<nv_bfloat16> h_input(D);
    for (auto& v : h_input) v = make_nvbf16_from_float(0.0f);

    thrust::device_vector<nv_bfloat16> d_input(h_input.begin(), h_input.end());
    thrust::device_vector<uint8_t> d_packed(PACKED, 0);
    thrust::device_vector<nv_bfloat16> d_output(D);
    thrust::device_vector<uint32_t> d_d1(h_d1.begin(), h_d1.end());
    thrust::device_vector<uint32_t> d_d2(h_d2.begin(), h_d2.end());

    turboquant::quantize(
        thrust::raw_pointer_cast(d_packed.data()),
        thrust::raw_pointer_cast(d_input.data()),
        thrust::raw_pointer_cast(d_d1.data()),
        thrust::raw_pointer_cast(d_d2.data()),
        N, D, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    turboquant::dequantize(
        thrust::raw_pointer_cast(d_output.data()),
        thrust::raw_pointer_cast(d_packed.data()),
        thrust::raw_pointer_cast(d_d1.data()),
        thrust::raw_pointer_cast(d_d2.data()),
        N, D, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<nv_bfloat16> h_output(D);
    thrust::copy(d_output.begin(), d_output.end(), h_output.begin());

    // All outputs should be zero (or very near zero)
    for (int j = 0; j < D; j++) {
        float val = bf16_bits_to_float(*reinterpret_cast<uint16_t*>(&h_output[j]));
        REQUIRE(std::abs(val) < 1e-3f);
    }
}

TEST_CASE("TurboQuant vs FP8 benchmark (D=128)", "[turboquant][benchmark]") {
    // Simulates KV cache dequantization for a single attention head during decode.
    // The critical path is dequant (reading from cache for every attention step).
    //
    // Layout: NUM_HEADS heads × SEQ_LEN vectors of HEAD_DIM each.
    // We benchmark:
    //   1. TurboQuant dequantize (3.5-bit packed → BF16)
    //   2. FP8 per-block dequantize (FP8 + scales → BF16)
    //   3. Raw BF16 memcpy (bandwidth ceiling)

    constexpr int HEAD_DIM  = 128;
    constexpr int SEQ_LEN   = 4096;
    constexpr int NUM_HEADS = 32;
    constexpr int TOTAL_VECS = NUM_HEADS * SEQ_LEN;   // 131072 vectors
    constexpr int PACKED_TQ  = turboquant::packed_size(HEAD_DIM);  // 60 bytes
    constexpr int BLOCK_SIZE = 128;  // FP8 per-block quantization block size
    constexpr int WARMUP = 20;
    constexpr int ITERS  = 100;

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    // --- Allocations ---
    auto h_input_f = random_vectors(TOTAL_VECS, HEAD_DIM, 999);
    auto h_input_bf = to_bf16(h_input_f);

    // TurboQuant buffers
    std::vector<uint32_t> h_d1(HEAD_DIM / 32), h_d2(HEAD_DIM / 32);
    turboquant::generate_signs_host(h_d1.data(), h_d2.data(), HEAD_DIM, 0xBE4C);

    thrust::device_vector<nv_bfloat16> d_input(h_input_bf.begin(), h_input_bf.end());
    thrust::device_vector<uint8_t> d_tq_packed(TOTAL_VECS * PACKED_TQ);
    thrust::device_vector<nv_bfloat16> d_tq_output(TOTAL_VECS * HEAD_DIM);
    thrust::device_vector<uint32_t> d_d1(h_d1.begin(), h_d1.end());
    thrust::device_vector<uint32_t> d_d2(h_d2.begin(), h_d2.end());

    // FP8 buffers
    int M = TOTAL_VECS, K = HEAD_DIM;
    int scale_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int scale_cols = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    thrust::device_vector<__nv_fp8_e4m3> d_fp8_data((size_t)M * K);
    thrust::device_vector<float> d_fp8_scales((size_t)scale_rows * scale_cols);
    thrust::device_vector<nv_bfloat16> d_fp8_output((size_t)M * K);

    // BF16 memcpy buffer
    thrust::device_vector<nv_bfloat16> d_memcpy_output((size_t)M * K);

    // --- Pre-quantize both formats ---
    turboquant::quantize(
        thrust::raw_pointer_cast(d_tq_packed.data()),
        thrust::raw_pointer_cast(d_input.data()),
        thrust::raw_pointer_cast(d_d1.data()),
        thrust::raw_pointer_cast(d_d2.data()),
        TOTAL_VECS, HEAD_DIM, 0);

    quantize_per_block(
        thrust::raw_pointer_cast(d_fp8_data.data()),
        thrust::raw_pointer_cast(d_fp8_scales.data()),
        thrust::raw_pointer_cast(d_input.data()),
        M, K, BLOCK_SIZE, props, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute data sizes
    size_t tq_bytes  = (size_t)TOTAL_VECS * PACKED_TQ;
    size_t fp8_bytes = (size_t)M * K * sizeof(__nv_fp8_e4m3) +
                       (size_t)scale_rows * scale_cols * sizeof(float);
    size_t bf16_bytes = (size_t)M * K * sizeof(nv_bfloat16);
    size_t out_bytes  = (size_t)M * K * sizeof(nv_bfloat16);

    // --- CUDA events for timing ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // === Benchmark 1: TurboQuant dequantize ===
    for (int i = 0; i < WARMUP; i++)
        turboquant::dequantize(
            thrust::raw_pointer_cast(d_tq_output.data()),
            thrust::raw_pointer_cast(d_tq_packed.data()),
            thrust::raw_pointer_cast(d_d1.data()),
            thrust::raw_pointer_cast(d_d2.data()),
            TOTAL_VECS, HEAD_DIM, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < ITERS; i++)
        turboquant::dequantize(
            thrust::raw_pointer_cast(d_tq_output.data()),
            thrust::raw_pointer_cast(d_tq_packed.data()),
            thrust::raw_pointer_cast(d_d1.data()),
            thrust::raw_pointer_cast(d_d2.data()),
            TOTAL_VECS, HEAD_DIM, 0);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float tq_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&tq_ms, start, stop));
    tq_ms /= ITERS;

    // === Benchmark 2: FP8 per-block dequantize ===
    for (int i = 0; i < WARMUP; i++)
        dequantize_per_block(
            thrust::raw_pointer_cast(d_fp8_output.data()),
            thrust::raw_pointer_cast(d_fp8_data.data()),
            thrust::raw_pointer_cast(d_fp8_scales.data()),
            M, K, BLOCK_SIZE, props, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < ITERS; i++)
        dequantize_per_block(
            thrust::raw_pointer_cast(d_fp8_output.data()),
            thrust::raw_pointer_cast(d_fp8_data.data()),
            thrust::raw_pointer_cast(d_fp8_scales.data()),
            M, K, BLOCK_SIZE, props, 0);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float fp8_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&fp8_ms, start, stop));
    fp8_ms /= ITERS;

    // === Benchmark 3: Raw BF16 memcpy (bandwidth ceiling) ===
    for (int i = 0; i < WARMUP; i++)
        CUDA_CHECK(cudaMemcpyAsync(
            thrust::raw_pointer_cast(d_memcpy_output.data()),
            thrust::raw_pointer_cast(d_input.data()),
            bf16_bytes, cudaMemcpyDeviceToDevice, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < ITERS; i++)
        CUDA_CHECK(cudaMemcpyAsync(
            thrust::raw_pointer_cast(d_memcpy_output.data()),
            thrust::raw_pointer_cast(d_input.data()),
            bf16_bytes, cudaMemcpyDeviceToDevice, 0));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float memcpy_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&memcpy_ms, start, stop));
    memcpy_ms /= ITERS;

    // === Benchmark 4: TurboQuant quantize ===
    for (int i = 0; i < WARMUP; i++)
        turboquant::quantize(
            thrust::raw_pointer_cast(d_tq_packed.data()),
            thrust::raw_pointer_cast(d_input.data()),
            thrust::raw_pointer_cast(d_d1.data()),
            thrust::raw_pointer_cast(d_d2.data()),
            TOTAL_VECS, HEAD_DIM, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < ITERS; i++)
        turboquant::quantize(
            thrust::raw_pointer_cast(d_tq_packed.data()),
            thrust::raw_pointer_cast(d_input.data()),
            thrust::raw_pointer_cast(d_d1.data()),
            thrust::raw_pointer_cast(d_d2.data()),
            TOTAL_VECS, HEAD_DIM, 0);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float tq_quant_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&tq_quant_ms, start, stop));
    tq_quant_ms /= ITERS;

    // === Report ===
    double tq_read_bw   = (tq_bytes + out_bytes) / (tq_ms * 1e-3) / 1e9;
    double fp8_read_bw  = (fp8_bytes + out_bytes) / (fp8_ms * 1e-3) / 1e9;
    double memcpy_bw    = (2 * bf16_bytes) / (memcpy_ms * 1e-3) / 1e9;

    INFO("=== KV Cache Dequantize Benchmark (" << props.name << ") ===");
    INFO("  Vectors: " << TOTAL_VECS << " × " << HEAD_DIM
         << " (" << NUM_HEADS << " heads × " << SEQ_LEN << " seq)");
    INFO("");
    INFO("  BF16 memcpy (ceiling) : " << memcpy_ms << " ms  ("
         << memcpy_bw << " GB/s)");
    INFO("  FP8 dequantize        : " << fp8_ms << " ms  ("
         << fp8_read_bw << " GB/s effective)  "
         << fp8_bytes / 1048576.0 << " MB read");
    INFO("  TQ 3.5b dequantize    : " << tq_ms << " ms  ("
         << tq_read_bw << " GB/s effective)  "
         << tq_bytes / 1048576.0 << " MB read");
    INFO("  TQ 3.5b quantize      : " << tq_quant_ms << " ms");
    INFO("");
    INFO("  FP8 cache size   : " << fp8_bytes / 1048576.0 << " MB");
    INFO("  TQ cache size    : " << tq_bytes / 1048576.0 << " MB  ("
         << (double)fp8_bytes / tq_bytes << "x smaller)");
    INFO("  Dequant speedup  : TQ is "
         << (fp8_ms > tq_ms ? std::to_string(fp8_ms / tq_ms) + "x faster"
                            : std::to_string(tq_ms / fp8_ms) + "x slower")
         << " than FP8");

    // The test itself just ensures the benchmark ran without errors
    REQUIRE(tq_ms > 0);
    REQUIRE(fp8_ms > 0);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
