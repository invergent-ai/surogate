// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for fast LoRA SiLU kernels for MoE expert optimization

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <random>

#include <cuda_bf16.h>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "kernels/kernels.h"
#include "utilities/utils.h"
#include "test_config.h"
#include "test_utils.h"

using namespace testing_utils;
using Catch::Approx;

namespace {

// ============================================================================
// CPU Reference Implementations
// ============================================================================

// CPU sigmoid
static float sigmoid_cpu(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// CPU SiLU: x * sigmoid(x)
static float silu_cpu(float x) {
    return x * sigmoid_cpu(x);
}

// CPU reference: h = silu(e) * g
static void silu_mul_forward_cpu(float* h, const float* e, const float* g, int N, int D) {
    for (int i = 0; i < N * D; ++i) {
        h[i] = silu_cpu(e[i]) * g[i];
    }
}

// CPU reference: in-place backward through silu_mul
// Given dh, e, g: compute de, dg (overwrite e, g) and output h
static void silu_mul_backward_inplace_cpu(float* e, float* g, const float* dh, float* h_out, int N, int D) {
    for (int i = 0; i < N * D; ++i) {
        float e_val = e[i];
        float g_val = g[i];
        float dh_val = dh[i];

        float sig_e = sigmoid_cpu(e_val);
        float silu_e = e_val * sig_e;

        // h = silu(e) * g
        if (h_out) {
            h_out[i] = silu_e * g_val;
        }

        // d(silu(e))/de = sigmoid(e) * (1 + e * (1 - sigmoid(e)))
        float dsilu_de = sig_e * (1.0f + e_val * (1.0f - sig_e));

        // de = dh * g * d(silu(e))/de
        float de = dh_val * g_val * dsilu_de;

        // dg = dh * silu(e)
        float dg = dh_val * silu_e;

        e[i] = de;
        g[i] = dg;
    }
}

// CPU reference: split gate_up (N, 2D) into up (N, D) and gate (N, D)
// Layout: gate_up = [up | gate]
static void split_gate_up_cpu(const float* gate_up, float* up, float* gate, int N, int D) {
    for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
            up[n * D + d] = gate_up[n * 2 * D + d];         // up is first D columns
            gate[n * D + d] = gate_up[n * 2 * D + D + d];   // gate is second D columns
        }
    }
}

// CPU reference: concat dg (N, D) and de (N, D) into d_gate_up (N, 2D)
// Layout: d_gate_up = [dg | de]
static void concat_d_gate_up_cpu(const float* dg, const float* de, float* d_gate_up, int N, int D) {
    for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
            d_gate_up[n * 2 * D + d] = dg[n * D + d];       // dg is first D columns
            d_gate_up[n * 2 * D + D + d] = de[n * D + d];   // de is second D columns
        }
    }
}

// Helper to generate random data
static void fill_random(std::vector<float>& vec, float min_val = -1.0f, float max_val = 1.0f) {
    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (auto& v : vec) {
        v = dist(gen);
    }
}

// Helper to convert float to bf16 and back (to match precision)
static void round_to_bf16(std::vector<float>& vec) {
    for (auto& v : vec) {
        nv_bfloat16 bf = __float2bfloat16(v);
        v = __bfloat162float(bf);
    }
}

// Helper to compute max absolute error
static float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    float max_err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_err = std::max(max_err, std::abs(a[i] - b[i]));
    }
    return max_err;
}

// Helper to compute relative error
static float max_rel_error(const std::vector<float>& a, const std::vector<float>& b, float eps = 1e-6f) {
    float max_err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float denom = std::max(std::abs(a[i]), std::abs(b[i])) + eps;
        max_err = std::max(max_err, std::abs(a[i] - b[i]) / denom);
    }
    return max_err;
}

} // anonymous namespace

// ============================================================================
// Test Cases
// ============================================================================

TEST_CASE("silu_mul_forward correctness", "[fast_lora][silu]") {
    const int N = 1024;  // tokens
    const int D = 768;   // intermediate dim (Qwen3 MoE)

    // Allocate host data
    std::vector<float> h_e(N * D), h_g(N * D), h_h_ref(N * D), h_h_gpu(N * D);
    fill_random(h_e);
    fill_random(h_g);

    // Round to BF16 for fair comparison
    round_to_bf16(h_e);
    round_to_bf16(h_g);

    // CPU reference
    silu_mul_forward_cpu(h_h_ref.data(), h_e.data(), h_g.data(), N, D);

    // GPU execution
    thrust::device_vector<nv_bfloat16> d_e(N * D), d_g(N * D), d_h(N * D);

    // Convert and upload
    std::vector<nv_bfloat16> bf_e(N * D), bf_g(N * D);
    for (int i = 0; i < N * D; ++i) {
        bf_e[i] = __float2bfloat16(h_e[i]);
        bf_g[i] = __float2bfloat16(h_g[i]);
    }
    thrust::copy(bf_e.begin(), bf_e.end(), d_e.begin());
    thrust::copy(bf_g.begin(), bf_g.end(), d_g.begin());

    // Run kernel
    silu_mul_forward(
        thrust::raw_pointer_cast(d_h.data()),
        thrust::raw_pointer_cast(d_e.data()),
        thrust::raw_pointer_cast(d_g.data()),
        N, D, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download and convert
    std::vector<nv_bfloat16> bf_h(N * D);
    thrust::copy(d_h.begin(), d_h.end(), bf_h.begin());
    for (int i = 0; i < N * D; ++i) {
        h_h_gpu[i] = __bfloat162float(bf_h[i]);
    }

    // Compare
    float max_err = max_rel_error(h_h_ref, h_h_gpu);
    INFO("Max relative error: " << max_err);
    REQUIRE(max_err < 0.01f);  // 1% tolerance for BF16
}

TEST_CASE("silu_mul_backward_inplace correctness", "[fast_lora][silu]") {
    const int N = 1024;
    const int D = 768;

    // Allocate host data
    std::vector<float> h_e(N * D), h_g(N * D), h_dh(N * D);
    std::vector<float> h_h_ref(N * D), h_de_ref(N * D), h_dg_ref(N * D);
    std::vector<float> h_h_gpu(N * D), h_de_gpu(N * D), h_dg_gpu(N * D);

    fill_random(h_e);
    fill_random(h_g);
    fill_random(h_dh);

    round_to_bf16(h_e);
    round_to_bf16(h_g);
    round_to_bf16(h_dh);

    // CPU reference: copy e, g first since they'll be modified
    h_de_ref = h_e;
    h_dg_ref = h_g;
    silu_mul_backward_inplace_cpu(h_de_ref.data(), h_dg_ref.data(), h_dh.data(), h_h_ref.data(), N, D);

    // GPU execution
    thrust::device_vector<nv_bfloat16> d_e(N * D), d_g(N * D), d_dh(N * D), d_h(N * D);

    std::vector<nv_bfloat16> bf_e(N * D), bf_g(N * D), bf_dh(N * D);
    for (int i = 0; i < N * D; ++i) {
        bf_e[i] = __float2bfloat16(h_e[i]);
        bf_g[i] = __float2bfloat16(h_g[i]);
        bf_dh[i] = __float2bfloat16(h_dh[i]);
    }
    thrust::copy(bf_e.begin(), bf_e.end(), d_e.begin());
    thrust::copy(bf_g.begin(), bf_g.end(), d_g.begin());
    thrust::copy(bf_dh.begin(), bf_dh.end(), d_dh.begin());

    // Run kernel (in-place modifies d_e -> de, d_g -> dg)
    silu_mul_backward_inplace(
        thrust::raw_pointer_cast(d_e.data()),
        thrust::raw_pointer_cast(d_g.data()),
        thrust::raw_pointer_cast(d_dh.data()),
        thrust::raw_pointer_cast(d_h.data()),
        N, D, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download results
    std::vector<nv_bfloat16> bf_de(N * D), bf_dg(N * D), bf_h(N * D);
    thrust::copy(d_e.begin(), d_e.end(), bf_de.begin());
    thrust::copy(d_g.begin(), d_g.end(), bf_dg.begin());
    thrust::copy(d_h.begin(), d_h.end(), bf_h.begin());

    for (int i = 0; i < N * D; ++i) {
        h_de_gpu[i] = __bfloat162float(bf_de[i]);
        h_dg_gpu[i] = __bfloat162float(bf_dg[i]);
        h_h_gpu[i] = __bfloat162float(bf_h[i]);
    }

    // Compare de
    float de_err = max_rel_error(h_de_ref, h_de_gpu);
    INFO("de max relative error: " << de_err);
    REQUIRE(de_err < 0.02f);

    // Compare dg
    float dg_err = max_rel_error(h_dg_ref, h_dg_gpu);
    INFO("dg max relative error: " << dg_err);
    REQUIRE(dg_err < 0.02f);

    // Compare h
    float h_err = max_rel_error(h_h_ref, h_h_gpu);
    INFO("h max relative error: " << h_err);
    REQUIRE(h_err < 0.01f);
}

TEST_CASE("split_gate_up correctness", "[fast_lora][split]") {
    const int N = 512;
    const int D = 768;

    // Allocate host data
    std::vector<float> h_gate_up(N * 2 * D);
    std::vector<float> h_up_ref(N * D), h_gate_ref(N * D);
    std::vector<float> h_up_gpu(N * D), h_gate_gpu(N * D);

    fill_random(h_gate_up);
    round_to_bf16(h_gate_up);

    // CPU reference
    split_gate_up_cpu(h_gate_up.data(), h_up_ref.data(), h_gate_ref.data(), N, D);

    // GPU execution
    thrust::device_vector<nv_bfloat16> d_gate_up(N * 2 * D), d_up(N * D), d_gate(N * D);

    std::vector<nv_bfloat16> bf_gate_up(N * 2 * D);
    for (int i = 0; i < N * 2 * D; ++i) {
        bf_gate_up[i] = __float2bfloat16(h_gate_up[i]);
    }
    thrust::copy(bf_gate_up.begin(), bf_gate_up.end(), d_gate_up.begin());

    split_gate_up(
        thrust::raw_pointer_cast(d_gate_up.data()),
        thrust::raw_pointer_cast(d_up.data()),
        thrust::raw_pointer_cast(d_gate.data()),
        N, D, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download
    std::vector<nv_bfloat16> bf_up(N * D), bf_gate(N * D);
    thrust::copy(d_up.begin(), d_up.end(), bf_up.begin());
    thrust::copy(d_gate.begin(), d_gate.end(), bf_gate.begin());

    for (int i = 0; i < N * D; ++i) {
        h_up_gpu[i] = __bfloat162float(bf_up[i]);
        h_gate_gpu[i] = __bfloat162float(bf_gate[i]);
    }

    // Compare (should be exact since it's just data movement)
    float up_err = max_abs_error(h_up_ref, h_up_gpu);
    float gate_err = max_abs_error(h_gate_ref, h_gate_gpu);

    INFO("up max abs error: " << up_err);
    INFO("gate max abs error: " << gate_err);
    REQUIRE(up_err == 0.0f);
    REQUIRE(gate_err == 0.0f);
}

TEST_CASE("concat_d_gate_up correctness", "[fast_lora][concat]") {
    const int N = 512;
    const int D = 768;

    // Allocate host data
    std::vector<float> h_dg(N * D), h_de(N * D);
    std::vector<float> h_d_gate_up_ref(N * 2 * D), h_d_gate_up_gpu(N * 2 * D);

    fill_random(h_dg);
    fill_random(h_de);
    round_to_bf16(h_dg);
    round_to_bf16(h_de);

    // CPU reference
    concat_d_gate_up_cpu(h_dg.data(), h_de.data(), h_d_gate_up_ref.data(), N, D);

    // GPU execution
    thrust::device_vector<nv_bfloat16> d_dg(N * D), d_de(N * D), d_d_gate_up(N * 2 * D);

    std::vector<nv_bfloat16> bf_dg(N * D), bf_de(N * D);
    for (int i = 0; i < N * D; ++i) {
        bf_dg[i] = __float2bfloat16(h_dg[i]);
        bf_de[i] = __float2bfloat16(h_de[i]);
    }
    thrust::copy(bf_dg.begin(), bf_dg.end(), d_dg.begin());
    thrust::copy(bf_de.begin(), bf_de.end(), d_de.begin());

    concat_d_gate_up(
        thrust::raw_pointer_cast(d_dg.data()),
        thrust::raw_pointer_cast(d_de.data()),
        thrust::raw_pointer_cast(d_d_gate_up.data()),
        N, D, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download
    std::vector<nv_bfloat16> bf_d_gate_up(N * 2 * D);
    thrust::copy(d_d_gate_up.begin(), d_d_gate_up.end(), bf_d_gate_up.begin());

    for (int i = 0; i < N * 2 * D; ++i) {
        h_d_gate_up_gpu[i] = __bfloat162float(bf_d_gate_up[i]);
    }

    // Compare (should be exact)
    float err = max_abs_error(h_d_gate_up_ref, h_d_gate_up_gpu);
    INFO("d_gate_up max abs error: " << err);
    REQUIRE(err == 0.0f);
}

TEST_CASE("split then concat roundtrip", "[fast_lora][roundtrip]") {
    const int N = 256;
    const int D = 768;

    // Original gate_up
    std::vector<float> h_gate_up(N * 2 * D);
    fill_random(h_gate_up);
    round_to_bf16(h_gate_up);

    // GPU
    thrust::device_vector<nv_bfloat16> d_gate_up(N * 2 * D);
    thrust::device_vector<nv_bfloat16> d_up(N * D), d_gate(N * D);
    thrust::device_vector<nv_bfloat16> d_reconstructed(N * 2 * D);

    std::vector<nv_bfloat16> bf_gate_up(N * 2 * D);
    for (int i = 0; i < N * 2 * D; ++i) {
        bf_gate_up[i] = __float2bfloat16(h_gate_up[i]);
    }
    thrust::copy(bf_gate_up.begin(), bf_gate_up.end(), d_gate_up.begin());

    // Split
    split_gate_up(
        thrust::raw_pointer_cast(d_gate_up.data()),
        thrust::raw_pointer_cast(d_up.data()),
        thrust::raw_pointer_cast(d_gate.data()),
        N, D, nullptr);

    // Concat back: [up | gate] -> need to use concat_d_gate_up which does [dg | de]
    // So we swap: up goes to dg position, gate goes to de position
    concat_d_gate_up(
        thrust::raw_pointer_cast(d_up.data()),    // goes to first D columns
        thrust::raw_pointer_cast(d_gate.data()),  // goes to second D columns
        thrust::raw_pointer_cast(d_reconstructed.data()),
        N, D, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download and compare
    std::vector<nv_bfloat16> bf_reconstructed(N * 2 * D);
    thrust::copy(d_reconstructed.begin(), d_reconstructed.end(), bf_reconstructed.begin());

    std::vector<float> h_reconstructed(N * 2 * D);
    for (int i = 0; i < N * 2 * D; ++i) {
        h_reconstructed[i] = __bfloat162float(bf_reconstructed[i]);
    }

    float err = max_abs_error(h_gate_up, h_reconstructed);
    INFO("Roundtrip max abs error: " << err);
    REQUIRE(err == 0.0f);
}

TEST_CASE("silu_mul_backward_inplace gradient check", "[fast_lora][gradient]") {
    // Numerical gradient check to verify backward is correct
    const int N = 32;
    const int D = 64;
    const float eps = 1e-3f;

    std::vector<float> h_e(N * D), h_g(N * D), h_dh(N * D);
    fill_random(h_e, -0.5f, 0.5f);  // Smaller range for numerical stability
    fill_random(h_g, -0.5f, 0.5f);

    // Use ones for dh to simplify
    std::fill(h_dh.begin(), h_dh.end(), 1.0f);

    // Compute analytical gradients
    std::vector<float> h_de_anal = h_e;
    std::vector<float> h_dg_anal = h_g;
    silu_mul_backward_inplace_cpu(h_de_anal.data(), h_dg_anal.data(), h_dh.data(), nullptr, N, D);

    // Compute numerical gradients for de
    std::vector<float> h_de_num(N * D);
    for (int i = 0; i < N * D; ++i) {
        // f(e + eps)
        std::vector<float> e_plus = h_e;
        e_plus[i] += eps;
        std::vector<float> h_plus(N * D);
        silu_mul_forward_cpu(h_plus.data(), e_plus.data(), h_g.data(), N, D);
        float loss_plus = std::accumulate(h_plus.begin(), h_plus.end(), 0.0f);

        // f(e - eps)
        std::vector<float> e_minus = h_e;
        e_minus[i] -= eps;
        std::vector<float> h_minus(N * D);
        silu_mul_forward_cpu(h_minus.data(), e_minus.data(), h_g.data(), N, D);
        float loss_minus = std::accumulate(h_minus.begin(), h_minus.end(), 0.0f);

        h_de_num[i] = (loss_plus - loss_minus) / (2 * eps);
    }

    // Compare de
    float de_max_err = 0.0f;
    for (int i = 0; i < N * D; ++i) {
        float err = std::abs(h_de_anal[i] - h_de_num[i]);
        de_max_err = std::max(de_max_err, err);
    }
    INFO("de numerical vs analytical max error: " << de_max_err);
    REQUIRE(de_max_err < 0.01f);

    // Compute numerical gradients for dg
    std::vector<float> h_dg_num(N * D);
    for (int i = 0; i < N * D; ++i) {
        std::vector<float> g_plus = h_g;
        g_plus[i] += eps;
        std::vector<float> h_plus(N * D);
        silu_mul_forward_cpu(h_plus.data(), h_e.data(), g_plus.data(), N, D);
        float loss_plus = std::accumulate(h_plus.begin(), h_plus.end(), 0.0f);

        std::vector<float> g_minus = h_g;
        g_minus[i] -= eps;
        std::vector<float> h_minus(N * D);
        silu_mul_forward_cpu(h_minus.data(), h_e.data(), g_minus.data(), N, D);
        float loss_minus = std::accumulate(h_minus.begin(), h_minus.end(), 0.0f);

        h_dg_num[i] = (loss_plus - loss_minus) / (2 * eps);
    }

    // Compare dg
    float dg_max_err = 0.0f;
    for (int i = 0; i < N * D; ++i) {
        float err = std::abs(h_dg_anal[i] - h_dg_num[i]);
        dg_max_err = std::max(dg_max_err, err);
    }
    INFO("dg numerical vs analytical max error: " << dg_max_err);
    REQUIRE(dg_max_err < 0.01f);
}
