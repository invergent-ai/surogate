// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

#include <cuda_bf16.h>

#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "kernels/kernels.h"
#include "../utilities/test_config.h"
#include "../utilities/test_utils.h"

using namespace testing_utils;

namespace {

// CPU baseline for RoPE forward: rotates queries and keys, copies values
static void rope_forward_cpu(float* out, const float* in, const float* freqs,
                             int B, int T, int Nq, int Nkv, int HD) {
    const int N = Nq + 2 * Nkv; // [q, k, v]
    const int HD2 = HD / 2;
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const float* freqt = freqs + t * HD;
            for (int h = 0; h < N; ++h) {
                int qkv = (h < Nq) ? 0 : ((h < Nq + Nkv) ? 1 : 2);
                const float* base = in + (((b * T + t) * N + h) * HD);
                float* outb = out + (((b * T + t) * N + h) * HD);
                if (qkv == 2) {
                    // values: copy through
                    std::copy(base, base + HD, outb);
                } else {
                    // apply rotation
                    for (int d = 0; d < HD2; ++d) {
                        float real = base[d];
                        float imag = base[d + HD2];
                        float c = freqt[2 * d + 0];
                        float s = freqt[2 * d + 1];
                        outb[d] = real * c - imag * s;
                        outb[d + HD2] = real * s + imag * c;
                    }
                }
            }
        }
    }
}

// CPU baseline for RoPE backward: inverse rotation on q/k, copy for v
static void rope_backward_cpu(float* dinp, const float* dout, const float* freqs,
                              int B, int T, int Nq, int Nkv, int HD) {
    const int N = Nq + 2 * Nkv;
    const int HD2 = HD / 2;
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const float* freqt = freqs + t * HD;
            for (int h = 0; h < N; ++h) {
                int qkv = (h < Nq) ? 0 : ((h < Nq + Nkv) ? 1 : 2);
                const float* base = dout + (((b * T + t) * N + h) * HD);
                float* outb = dinp + (((b * T + t) * N + h) * HD);
                if (qkv == 2) {
                    std::copy(base, base + HD, outb);
                } else {
                    for (int d = 0; d < HD2; ++d) {
                        float real = base[d];
                        float imag = base[d + HD2];
                        float c = freqt[2 * d + 0];
                        float s = -freqt[2 * d + 1]; // inverse rotation
                        outb[d] = real * c - imag * s;
                        outb[d + HD2] = real * s + imag * c;
                    }
                }
            }
        }
    }
}

} // namespace

TEST_CASE("rope forward/backward fp32 matches CPU", "[kernels][rope][fp32]") {
    const auto& cfg = testing_config::get_test_config();
    const int B = cfg.B;
    const int T = cfg.T;
    const int HD = cfg.C; // reuse C as head_dim
    const int Nq = cfg.Nq;
    const int Nkv = cfg.Nkv;
    const int N = Nq + 2 * Nkv;

    // Constraints from kernel: head_dim % (2 * x64::size) == 0, x64::size=2 for float -> HD % 4 == 0
    if (HD % 4 != 0) {
        INFO("Invalid sizes for fp32: require head_dim % 4 == 0");
        FAIL("Aborting fp32 rope test due to invalid size configuration");
    }

    const size_t size_inp = (size_t)B * T * N * HD;
    const size_t size_freqs = (size_t)T * HD;

    std::vector<float> h_inp = uniform_host(size_inp, -1.0f, 1.0f, /*seed*/ 1337ULL);

    std::vector<float> h_freqs(size_freqs);
    // match kernel helper
    precompute_freqs_cis(h_freqs.data(), HD, T, 10000.0f);

    // CPU forward
    std::vector<float> h_out_cpu(size_inp);
    rope_forward_cpu(h_out_cpu.data(), h_inp.data(), h_freqs.data(), B, T, Nq, Nkv, HD);

    // Device buffers
    thrust::device_vector<float> d_inp = to_device(h_inp);
    thrust::device_vector<float> d_out(size_inp);
    thrust::device_vector<float> d_freqs = to_device(h_freqs);

    rope_forward(thrust::raw_pointer_cast(d_out.data()),
                 thrust::raw_pointer_cast(d_inp.data()),
                 thrust::raw_pointer_cast(d_freqs.data()),
                 /*position_ids=*/nullptr,
                 nullptr, B, T, Nq, Nkv, HD, 0);

    std::vector<float> h_out(size_inp);
    thrust::copy(d_out.begin(), d_out.end(), h_out.begin());
    for (size_t i = 0; i < size_inp; ++i) {
        REQUIRE(h_out[i] == Catch::Approx(h_out_cpu[i]).margin(1e-6f));
    }

    // Forward again with absmax and ensure bit-perfect identical results, and absmax matches expected
    thrust::device_vector<float> d_absmax_fwd(1);
    thrust::device_vector<float> d_out2(size_inp);
    rope_forward(thrust::raw_pointer_cast(d_out2.data()),
                 thrust::raw_pointer_cast(d_inp.data()),
                 thrust::raw_pointer_cast(d_freqs.data()),
                 /*position_ids=*/nullptr,
                 thrust::raw_pointer_cast(d_absmax_fwd.data()), B, T, Nq, Nkv, HD, 0);
    std::vector<float> h_out2 = from_device(d_out2);
    for (size_t i = 0; i < size_inp; ++i) {
        REQUIRE(h_out2[i] == h_out[i]); // bit-perfect identical
    }
    float h_absmax_fwd = from_device(d_absmax_fwd)[0];
    float expected_absmax_fwd = 0.0f;
    for (size_t i = 0; i < size_inp; ++i) expected_absmax_fwd = std::max(expected_absmax_fwd, std::fabs(h_out_cpu[i]));
    REQUIRE(h_absmax_fwd == Catch::Approx(expected_absmax_fwd).margin(1e-6f));

    // Backward: generate dout and compare dinp
    std::vector<float> h_dout = uniform_host(size_inp,  -0.5, 0.5, 424242ull);
    std::vector<float> h_dinp_cpu(size_inp);
    rope_backward_cpu(h_dinp_cpu.data(), h_dout.data(), h_freqs.data(), B, T, Nq, Nkv, HD);

    thrust::device_vector<float> d_dout = to_device(h_dout);
    thrust::device_vector<float> d_dinp(size_inp);

    rope_backward(thrust::raw_pointer_cast(d_dinp.data()),
                  thrust::raw_pointer_cast(d_dout.data()),
                  thrust::raw_pointer_cast(d_freqs.data()),
                  /*position_ids=*/nullptr,
                  nullptr,
                  B, T, Nq, Nkv, HD, 0);

    std::vector<float> h_dinp = from_device(d_dinp);
    for (size_t i = 0; i < size_inp; ++i) {
        REQUIRE(h_dinp[i] == Catch::Approx(h_dinp_cpu[i]).margin(1e-6f));
    }

    // Backward again with absmax and ensure bit-perfect identical results, and absmax matches expected
    thrust::device_vector<float> d_absmax_bwd(1);
    thrust::device_vector<float> d_dinp2(size_inp);
    rope_backward(thrust::raw_pointer_cast(d_dinp2.data()),
                  thrust::raw_pointer_cast(d_dout.data()),
                  thrust::raw_pointer_cast(d_freqs.data()),
                  /*position_ids=*/nullptr,
                  thrust::raw_pointer_cast(d_absmax_bwd.data()),
                  B, T, Nq, Nkv, HD, 0);
    std::vector<float> h_dinp2 = from_device(d_dinp2);
    for (size_t i = 0; i < size_inp; ++i) {
        REQUIRE(h_dinp2[i] == h_dinp[i]); // bit-perfect identical
    }
    float h_absmax_bwd = from_device(d_absmax_bwd)[0];
    float expected_absmax_bwd = 0.0f;
    for (size_t i = 0; i < size_inp; ++i) expected_absmax_bwd = std::max(expected_absmax_bwd, std::fabs(h_dinp_cpu[i]));
    REQUIRE(h_absmax_bwd == Catch::Approx(expected_absmax_bwd).margin(1e-6f));
}

TEST_CASE("rope forward/backward bfloat16 matches CPU (emulated)", "[kernels][rope][bf16]") {
    const auto& cfg = testing_config::get_test_config();
    const int B = cfg.B;
    const int T = cfg.T;
    const int Nq = 4;
    const int Nkv = 4;
    const int HD = cfg.C / Nq;
    const int N = Nq + 2 * Nkv;

    // For bf16, x64::size = 4 -> require HD % 8 == 0
    if (HD % 8 != 0) {
        INFO("Invalid sizes for bf16: require head_dim % 8 == 0");
        FAIL("Aborting bf16 rope test due to invalid size configuration");
    }

    const size_t size_inp = (size_t)B * T * N * HD;
    const size_t size_freqs = (size_t)T * HD;

    // Prepare float inputs and quantize to bf16 for GPU
    std::vector<float> h_inp_f = uniform_host(size_inp, -1.f, 1.f, 1337ull);
    std::vector<nv_bfloat16> h_inp_bf16 = to_bf16(h_inp_f);

    // Prepare freqs and quantize to bf16 (kernel expects bf16 freqs as well)
    std::vector<float> h_freqs_f(size_freqs);
    precompute_freqs_cis(h_freqs_f.data(), HD, T, 10000.0f);
    std::vector<nv_bfloat16> h_freqs_bf16 = to_bf16(h_freqs_f);

    // CPU baseline with bf16 emulation: quantize inputs/freqs to bf16, do math in float, quantize outputs
    std::vector<float> h_inp_q = round_bf16(h_inp_f);
    std::vector<float> h_freqs_q = round_bf16(h_freqs_f);

    std::vector<float> h_out_cpu(size_inp);
    rope_forward_cpu(h_out_cpu.data(), h_inp_q.data(), h_freqs_q.data(), B, T, Nq, Nkv, HD);
    // Quantize outputs to bf16 for comparison
    h_out_cpu = round_bf16(h_out_cpu);

    // Device buffers
    thrust::device_vector<nv_bfloat16> d_inp = to_device(h_inp_bf16);
    thrust::device_vector<nv_bfloat16> d_out(size_inp);
    thrust::device_vector<nv_bfloat16> d_dinp(size_inp);
    thrust::device_vector<nv_bfloat16> d_freqs = to_device(h_freqs_bf16);

    rope_forward(thrust::raw_pointer_cast(d_out.data()),
                 thrust::raw_pointer_cast(d_inp.data()),
                 thrust::raw_pointer_cast(d_freqs.data()),
                 /*position_ids=*/nullptr,
                 nullptr, B, T, Nq, Nkv, HD, 0);

    std::vector<nv_bfloat16> h_out_bf16 = from_device(d_out);
    for (size_t i = 0; i < size_inp; ++i) {
        uint16_t bits;
        std::memcpy(&bits, &h_out_bf16[i], sizeof(bits));
        float v = bf16_bits_to_float(bits);
        REQUIRE(v == Catch::Approx(h_out_cpu[i]).margin(3e-2f));
    }

    // Forward again with absmax, ensure bit-perfect identical bf16 outputs, check absmax
    std::vector<nv_bfloat16> h_out_bf16_ref = h_out_bf16; // keep reference for bitwise compare
    thrust::device_vector<float> d_absmax_fwd_bf16(1);
    rope_forward(thrust::raw_pointer_cast(d_out.data()),
                 thrust::raw_pointer_cast(d_inp.data()),
                 thrust::raw_pointer_cast(d_freqs.data()),
                 /*position_ids=*/nullptr,
                 thrust::raw_pointer_cast(d_absmax_fwd_bf16.data()), B, T, Nq, Nkv, HD, 0);
    h_out_bf16 = from_device(d_out);
    for (size_t i = 0; i < size_inp; ++i) {
        uint16_t a, b;
        std::memcpy(&a, &h_out_bf16[i], sizeof(a));
        std::memcpy(&b, &h_out_bf16_ref[i], sizeof(b));
        REQUIRE(a == b); // bit-perfect identical
    }
    float h_absmax_fwd_bf16 = from_device(d_absmax_fwd_bf16)[0];
    float expected_absmax_fwd_bf16 = 0.0f;
    for (size_t i = 0; i < size_inp; ++i) expected_absmax_fwd_bf16 = std::max(expected_absmax_fwd_bf16, std::fabs(h_out_cpu[i]));
    REQUIRE(h_absmax_fwd_bf16 == Catch::Approx(expected_absmax_fwd_bf16).margin(3e-2f));

    // Backward bf16
    std::vector<float> h_dout_f = uniform_host(size_inp, -0.5f, 0.5f, 424242ull);
    std::vector<nv_bfloat16> h_dout_bf16 = to_bf16(h_dout_f);
    std::vector<float> h_dout_q = round_bf16(h_dout_f);
    thrust::device_vector<nv_bfloat16> d_dout = to_device(h_dout_bf16);

    std::vector<float> h_dinp_cpu(size_inp);
    rope_backward_cpu(h_dinp_cpu.data(), h_dout_q.data(), h_freqs_q.data(), B, T, Nq, Nkv, HD);
    h_dinp_cpu = round_bf16(h_dinp_cpu);

    rope_backward(thrust::raw_pointer_cast(d_dinp.data()),
                  thrust::raw_pointer_cast(d_dout.data()),
                  thrust::raw_pointer_cast(d_freqs.data()),
                  /*position_ids=*/nullptr,
                  nullptr, B, T, Nq, Nkv, HD, 0);

    std::vector<nv_bfloat16> h_dinp_bf16 = from_device(d_dinp);
    for (size_t i = 0; i < size_inp; ++i) {
        uint16_t bits;
        std::memcpy(&bits, &h_dinp_bf16[i], sizeof(bits));
        float v = bf16_bits_to_float(bits);
        REQUIRE(v == Catch::Approx(h_dinp_cpu[i]).margin(3e-2f));
    }

    // Backward again with absmax, bitwise identity and absmax check
    std::vector<nv_bfloat16> h_dinp_bf16_ref = h_dinp_bf16;
    thrust::device_vector<float> d_absmax_bwd_bf16(1);
    rope_backward(thrust::raw_pointer_cast(d_dinp.data()),
                  thrust::raw_pointer_cast(d_dout.data()),
                  thrust::raw_pointer_cast(d_freqs.data()),
                  /*position_ids=*/nullptr,
                  thrust::raw_pointer_cast(d_absmax_bwd_bf16.data()), B, T, Nq, Nkv, HD, 0);
    h_dinp_bf16 = from_device(d_dinp);
    for (size_t i = 0; i < size_inp; ++i) {
        uint16_t a, b;
        std::memcpy(&a, &h_dinp_bf16[i], sizeof(a));
        std::memcpy(&b, &h_dinp_bf16_ref[i], sizeof(b));
        REQUIRE(a == b);
    }
    float h_absmax_bwd_bf16 = from_device(d_absmax_bwd_bf16)[0];
    float expected_absmax_bwd_bf16 = 0.0f;
    for (size_t i = 0; i < size_inp; ++i) expected_absmax_bwd_bf16 = std::max(expected_absmax_bwd_bf16, std::fabs(h_dinp_cpu[i]));
    REQUIRE(h_absmax_bwd_bf16 == Catch::Approx(expected_absmax_bwd_bf16).margin(3e-2f));
}

// New tests to ensure abs-max path participates for values and to catch sync/early-return bugs
TEST_CASE("rope absmax (fp32)", "[kernels][rope][fp32][absmax]") {
    using namespace testing_config;
    // Craft sizes to force blocks to span Q/K and V regions in the same block
    // Small problem so a single block likely processes mixed heads
    const int B = 1, T = 64, Nq = 2, Nkv = 2, HD = 32;
    const int N = Nq + 2 * Nkv;

    const size_t size_inp = (size_t)B * T * N * HD;
    const size_t size_freqs = (size_t)T * HD;

    // Random input, large outlier in V region
    std::vector<float> h_inp = uniform_host(size_inp, -0.9f, 0.9f, 2025ull);
    h_inp[129] = 10.f;
    std::vector<float> h_freqs(size_freqs);
    precompute_freqs_cis(h_freqs.data(), HD, T, 10000.0f);

    // CPU reference
    std::vector<float> h_out_cpu(size_inp);
    rope_forward_cpu(h_out_cpu.data(), h_inp.data(), h_freqs.data(), B, T, Nq, Nkv, HD);

    // Device buffers
    thrust::device_vector<float> d_inp = to_device(h_inp);
    thrust::device_vector<float> d_out(size_inp);
    thrust::device_vector<float> d_freqs = to_device(h_freqs);

    // Out-of-place with absmax
    thrust::device_vector<float> d_absmax(1);
    rope_forward(thrust::raw_pointer_cast(d_out.data()),
                 thrust::raw_pointer_cast(d_inp.data()),
                 thrust::raw_pointer_cast(d_freqs.data()),
                 /*position_ids=*/nullptr,
                 thrust::raw_pointer_cast(d_absmax.data()),
                 B, T, Nq, Nkv, HD, 0);

    std::vector<float> h_out = from_device(d_out);
    for (size_t i = 0; i < size_inp; ++i) {
        REQUIRE(h_out[i] == Catch::Approx(h_out_cpu[i]).margin(1e-6f));
    }
    float absmax = from_device(d_absmax)[0];
    float expected_abs = 10.f;
    REQUIRE(absmax == Catch::Approx(expected_abs).margin(1e-6f)); // must include V region too

    // In-place with absmax should produce identical results
    thrust::device_vector<float> d_absmax_ip(1);
    // Copy input to output buffer to perform in-place
    thrust::device_vector<float> d_io = d_inp; // start from input
    rope_forward(thrust::raw_pointer_cast(d_io.data()),
                 thrust::raw_pointer_cast(d_io.data()),
                 thrust::raw_pointer_cast(d_freqs.data()),
                 /*position_ids=*/nullptr,
                 thrust::raw_pointer_cast(d_absmax_ip.data()),
                 B, T, Nq, Nkv, HD, 0);
    std::vector<float> h_io = from_device(d_io);
    for (size_t i = 0; i < size_inp; ++i) {
        REQUIRE(h_io[i] == Catch::Approx(h_out_cpu[i]).margin(1e-6f));
    }
}

// ============================================================================
// Tests for fused RoPE kernel (TransformerEngine-style optimization)
// ============================================================================

TEST_CASE("rope_fused forward/backward fp32 matches baseline", "[kernels][rope][fused][fp32]") {
    const auto& cfg = testing_config::get_test_config();
    const int B = cfg.B;
    const int T = cfg.T;
    const int HD = cfg.C;
    const int Nq = cfg.Nq;
    const int Nkv = cfg.Nkv;
    const int N = Nq + 2 * Nkv;
    const float theta = 10000.0f;

    if (HD % 4 != 0) {
        INFO("Invalid sizes for fp32: require head_dim % 4 == 0");
        FAIL("Aborting fp32 fused rope test due to invalid size configuration");
    }

    const size_t size_inp = (size_t)B * T * N * HD;
    const size_t size_freqs = (size_t)T * HD;

    std::vector<float> h_inp = uniform_host(size_inp, -1.0f, 1.0f, 1337ULL);
    std::vector<float> h_freqs(size_freqs);
    precompute_freqs_cis(h_freqs.data(), HD, T, theta);

    // CPU reference using baseline kernel output
    thrust::device_vector<float> d_inp_base = to_device(h_inp);
    thrust::device_vector<float> d_out_base(size_inp);
    thrust::device_vector<float> d_freqs = to_device(h_freqs);

    rope_forward(thrust::raw_pointer_cast(d_out_base.data()),
                 thrust::raw_pointer_cast(d_inp_base.data()),
                 thrust::raw_pointer_cast(d_freqs.data()),
                 /*position_ids=*/nullptr, nullptr,
                 B, T, Nq, Nkv, HD, 0);

    std::vector<float> h_out_base = from_device(d_out_base);

    // Fused kernel
    thrust::device_vector<float> d_inp_fused = to_device(h_inp);
    thrust::device_vector<float> d_out_fused(size_inp);

    rope_fused_forward(thrust::raw_pointer_cast(d_out_fused.data()),
                       thrust::raw_pointer_cast(d_inp_fused.data()),
                       /*position_ids=*/nullptr, nullptr,
                       theta, B, T, Nq, Nkv, HD, 0);

    std::vector<float> h_out_fused = from_device(d_out_fused);

    // Compare fused output with baseline
    for (size_t i = 0; i < size_inp; ++i) {
        REQUIRE(h_out_fused[i] == Catch::Approx(h_out_base[i]).margin(1e-5f));
    }

    // Test backward pass
    std::vector<float> h_dout = uniform_host(size_inp, -0.5f, 0.5f, 424242ULL);

    // Baseline backward
    thrust::device_vector<float> d_dout = to_device(h_dout);
    thrust::device_vector<float> d_dinp_base(size_inp);
    rope_backward(thrust::raw_pointer_cast(d_dinp_base.data()),
                  thrust::raw_pointer_cast(d_dout.data()),
                  thrust::raw_pointer_cast(d_freqs.data()),
                  /*position_ids=*/nullptr, nullptr,
                  B, T, Nq, Nkv, HD, 0);
    std::vector<float> h_dinp_base = from_device(d_dinp_base);

    // Fused backward
    thrust::device_vector<float> d_dinp_fused(size_inp);
    rope_fused_backward(thrust::raw_pointer_cast(d_dinp_fused.data()),
                        thrust::raw_pointer_cast(d_dout.data()),
                        /*position_ids=*/nullptr, nullptr,
                        theta, B, T, Nq, Nkv, HD, 0);
    std::vector<float> h_dinp_fused = from_device(d_dinp_fused);

    // Compare fused backward with baseline
    for (size_t i = 0; i < size_inp; ++i) {
        REQUIRE(h_dinp_fused[i] == Catch::Approx(h_dinp_base[i]).margin(1e-5f));
    }
}

TEST_CASE("rope_fused forward/backward bf16 matches baseline", "[kernels][rope][fused][bf16]") {
    const auto& cfg = testing_config::get_test_config();
    const int B = cfg.B;
    const int T = cfg.T;
    const int Nq = 4;
    const int Nkv = 4;
    const int HD = cfg.C / Nq;
    const int N = Nq + 2 * Nkv;
    const float theta = 10000.0f;

    if (HD % 8 != 0) {
        INFO("Invalid sizes for bf16: require head_dim % 8 == 0");
        FAIL("Aborting bf16 fused rope test due to invalid size configuration");
    }

    const size_t size_inp = (size_t)B * T * N * HD;
    const size_t size_freqs = (size_t)T * HD;

    std::vector<float> h_inp_f = uniform_host(size_inp, -1.0f, 1.0f, 1337ULL);
    std::vector<nv_bfloat16> h_inp_bf16 = to_bf16(h_inp_f);

    std::vector<float> h_freqs_f(size_freqs);
    precompute_freqs_cis(h_freqs_f.data(), HD, T, theta);
    std::vector<nv_bfloat16> h_freqs_bf16 = to_bf16(h_freqs_f);

    // Baseline forward
    thrust::device_vector<nv_bfloat16> d_inp_base = to_device(h_inp_bf16);
    thrust::device_vector<nv_bfloat16> d_out_base(size_inp);
    thrust::device_vector<nv_bfloat16> d_freqs = to_device(h_freqs_bf16);

    rope_forward(thrust::raw_pointer_cast(d_out_base.data()),
                 thrust::raw_pointer_cast(d_inp_base.data()),
                 thrust::raw_pointer_cast(d_freqs.data()),
                 /*position_ids=*/nullptr, nullptr,
                 B, T, Nq, Nkv, HD, 0);

    std::vector<nv_bfloat16> h_out_base_bf16 = from_device(d_out_base);

    // Fused forward
    thrust::device_vector<nv_bfloat16> d_inp_fused = to_device(h_inp_bf16);
    thrust::device_vector<nv_bfloat16> d_out_fused(size_inp);

    rope_fused_forward(thrust::raw_pointer_cast(d_out_fused.data()),
                       thrust::raw_pointer_cast(d_inp_fused.data()),
                       /*position_ids=*/nullptr, nullptr,
                       theta, B, T, Nq, Nkv, HD, 0);

    std::vector<nv_bfloat16> h_out_fused_bf16 = from_device(d_out_fused);

    // Compare fused output with baseline (slightly larger margin for bf16)
    for (size_t i = 0; i < size_inp; ++i) {
        uint16_t bits_base, bits_fused;
        std::memcpy(&bits_base, &h_out_base_bf16[i], sizeof(bits_base));
        std::memcpy(&bits_fused, &h_out_fused_bf16[i], sizeof(bits_fused));
        float v_base = bf16_bits_to_float(bits_base);
        float v_fused = bf16_bits_to_float(bits_fused);
        REQUIRE(v_fused == Catch::Approx(v_base).margin(5e-2f));
    }

    // Backward pass
    std::vector<float> h_dout_f = uniform_host(size_inp, -0.5f, 0.5f, 424242ULL);
    std::vector<nv_bfloat16> h_dout_bf16 = to_bf16(h_dout_f);

    // Baseline backward
    thrust::device_vector<nv_bfloat16> d_dout = to_device(h_dout_bf16);
    thrust::device_vector<nv_bfloat16> d_dinp_base(size_inp);
    rope_backward(thrust::raw_pointer_cast(d_dinp_base.data()),
                  thrust::raw_pointer_cast(d_dout.data()),
                  thrust::raw_pointer_cast(d_freqs.data()),
                  /*position_ids=*/nullptr, nullptr,
                  B, T, Nq, Nkv, HD, 0);
    std::vector<nv_bfloat16> h_dinp_base_bf16 = from_device(d_dinp_base);

    // Fused backward
    thrust::device_vector<nv_bfloat16> d_dinp_fused(size_inp);
    rope_fused_backward(thrust::raw_pointer_cast(d_dinp_fused.data()),
                        thrust::raw_pointer_cast(d_dout.data()),
                        /*position_ids=*/nullptr, nullptr,
                        theta, B, T, Nq, Nkv, HD, 0);
    std::vector<nv_bfloat16> h_dinp_fused_bf16 = from_device(d_dinp_fused);

    // Compare
    for (size_t i = 0; i < size_inp; ++i) {
        uint16_t bits_base, bits_fused;
        std::memcpy(&bits_base, &h_dinp_base_bf16[i], sizeof(bits_base));
        std::memcpy(&bits_fused, &h_dinp_fused_bf16[i], sizeof(bits_fused));
        float v_base = bf16_bits_to_float(bits_base);
        float v_fused = bf16_bits_to_float(bits_fused);
        REQUIRE(v_fused == Catch::Approx(v_base).margin(5e-2f));
    }
}

TEST_CASE("rope_fused absmax (fp32)", "[kernels][rope][fused][fp32][absmax]") {
    const int B = 1, T = 64, Nq = 2, Nkv = 2, HD = 32;
    const int N = Nq + 2 * Nkv;
    const float theta = 10000.0f;

    const size_t size_inp = (size_t)B * T * N * HD;

    // Random input with a large outlier in V region
    std::vector<float> h_inp = uniform_host(size_inp, -0.9f, 0.9f, 2025ULL);
    h_inp[129] = 10.f;

    thrust::device_vector<float> d_inp = to_device(h_inp);
    thrust::device_vector<float> d_out(size_inp);
    thrust::device_vector<float> d_absmax(1);

    rope_fused_forward(thrust::raw_pointer_cast(d_out.data()),
                       thrust::raw_pointer_cast(d_inp.data()),
                       /*position_ids=*/nullptr,
                       thrust::raw_pointer_cast(d_absmax.data()),
                       theta, B, T, Nq, Nkv, HD, 0);

    float absmax = from_device(d_absmax)[0];
    // V values pass through unchanged, so absmax should include the 10.0 outlier
    REQUIRE(absmax == Catch::Approx(10.0f).margin(1e-5f));
}

TEST_CASE("rope_fused in-place operation (fp32)", "[kernels][rope][fused][fp32][inplace]") {
    const int B = 2, T = 32, Nq = 4, Nkv = 4, HD = 64;
    const int N = Nq + 2 * Nkv;
    const float theta = 10000.0f;

    const size_t size_inp = (size_t)B * T * N * HD;

    std::vector<float> h_inp = uniform_host(size_inp, -1.0f, 1.0f, 9999ULL);

    // Out-of-place
    thrust::device_vector<float> d_inp = to_device(h_inp);
    thrust::device_vector<float> d_out(size_inp);
    rope_fused_forward(thrust::raw_pointer_cast(d_out.data()),
                       thrust::raw_pointer_cast(d_inp.data()),
                       /*position_ids=*/nullptr, nullptr,
                       theta, B, T, Nq, Nkv, HD, 0);
    std::vector<float> h_out_oop = from_device(d_out);

    // In-place
    thrust::device_vector<float> d_inplace = to_device(h_inp);
    rope_fused_forward(thrust::raw_pointer_cast(d_inplace.data()),
                       thrust::raw_pointer_cast(d_inplace.data()),
                       /*position_ids=*/nullptr, nullptr,
                       theta, B, T, Nq, Nkv, HD, 0);
    std::vector<float> h_out_ip = from_device(d_inplace);

    // Should match
    for (size_t i = 0; i < size_inp; ++i) {
        REQUIRE(h_out_ip[i] == Catch::Approx(h_out_oop[i]).margin(1e-6f));
    }
}
