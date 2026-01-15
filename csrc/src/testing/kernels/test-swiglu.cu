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
#include <thrust/fill.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "kernels/kernels.h"
#include "utilities/utils.h"
#include "../utilities/test_config.h"
#include "../utilities/test_utils.h"

using namespace testing_utils;

namespace {

// CPU baseline SWIGLU forward: out = (x1 * x2) / (1 + exp(-x2))
static void swiglu_forward_cpu(float* out, const float* inp, int B, int T, int C) {
    const int C2 = 2 * C;
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const float* row = inp + (b * T + t) * C2;
            const float* up = row;
            const float* gate = row + C;
            float* o = out + (b * T + t) * C;
            for (int c = 0; c < C; ++c) {
                float x1 = up[c];
                float x2 = gate[c];
                float s = 1.0f / (1.0f + std::exp(-x2));
                o[c] = x1 * x2 * s;
            }
        }
    }
}

// CPU baseline SWIGLU backward: given dout(B,T,C), inp(B,T,2C) -> dinp(B,T,2C)
static void swiglu_backward_cpu(float* dinp, const float* dout, const float* inp, int B, int T, int C) {
    const int C2 = 2 * C;
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const float* row = inp + (b * T + t) * C2;
            const float* up = row;
            const float* gate = row + C;
            float* di = dinp + (b * T + t) * C2;
            float* di1 = di;
            float* di2 = di + C;
            const float* d = dout + (b * T + t) * C;
            for (int c = 0; c < C; ++c) {
                float x1 = up[c];
                float x2 = gate[c];
                float g = d[c];
                float s = 1.0f / (1.0f + std::exp(-x2));
                float dx1 = g * x2 * s;
                float dx2 = g * x1 * s * (1.0f + x2 * (1.0f - s));
                di1[c] = dx1;
                di2[c] = dx2;
            }
        }
    }
}

static float max_abs(const float* data, size_t n) {
    float m = 0.f;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(data[i]));
    return m;
}

// Inputs and grads are now generated via shared deterministic fillers

} // namespace

TEST_CASE("swiglu forward/backward fp32 matches CPU", "[kernels][swiglu][fp32]") {
    const auto& cfg = testing_config::get_test_config();
    const int B = cfg.B;
    const int T = cfg.T; // ensure B*T*C divisible by kernel block requirements
    const int C = cfg.C; // multiple of 4
    // Validate constraints for the fp32 kernels
    long long prod = 1LL * B * T * C;
    bool ok = (C % 4 == 0) && (prod % 1024 == 0);
    if (!ok) {
        INFO("Invalid sizes for fp32: require C % 4 == 0 and (B*T*C) % 1024 == 0");
        INFO("Provided: B=" << B << ", T=" << T << ", C=" << C << ", B*T*C=" << prod);
        FAIL("Aborting fp32 test due to invalid size configuration");
    }
    const int C2 = 2 * C;
    const size_t n_inp = static_cast<size_t>(B) * T * C2;
    const size_t n_out = static_cast<size_t>(B) * T * C;

    std::vector<float> h_inp = uniform_host(n_inp, -1.f, 1.f, 1337ull);

    std::vector<float> h_out_cpu(n_out);
    swiglu_forward_cpu(h_out_cpu.data(), h_inp.data(), B, T, C);
    float cpu_absmax_fwd = max_abs(h_out_cpu.data(), n_out);

    // Device buffers
    thrust::device_vector<float> d_inp = to_device(h_inp);
    thrust::device_vector<float> d_out(n_out);
    thrust::device_vector<float> d_absmax(1);

    // Forward without absmax
    swiglu_forward(thrust::raw_pointer_cast(d_out.data()),
                   thrust::raw_pointer_cast(d_inp.data()),
                   nullptr, B, T, C, /*stream*/0);
    std::vector<float> h_out = from_device(d_out);

    for (size_t i = 0; i < n_out; ++i) {
        REQUIRE(h_out[i] == Catch::Approx(h_out_cpu[i]).margin(1e-6f));
    }

    // Forward with absmax
    swiglu_forward(thrust::raw_pointer_cast(d_out.data()),
                   thrust::raw_pointer_cast(d_inp.data()),
                   thrust::raw_pointer_cast(d_absmax.data()), B, T, C, 0);
    float h_absmax = 0.f;
    thrust::copy(d_absmax.begin(), d_absmax.end(), &h_absmax);
    REQUIRE(h_absmax == Catch::Approx(cpu_absmax_fwd).margin(1e-6f));

    // Backward: prepare dout
    std::vector<float> h_dout = uniform_host(n_out, -0.5f, 0.5f, 424242ull);
    thrust::device_vector<float> d_dout = to_device(h_dout);
    thrust::device_vector<float> d_dinp(n_inp);

    // CPU backward
    std::vector<float> h_dinp_cpu(n_inp);
    swiglu_backward_cpu(h_dinp_cpu.data(), h_dout.data(), h_inp.data(), B, T, C);
    float cpu_absmax_bwd = max_abs(h_dinp_cpu.data(), n_inp);

    // GPU backward without absmax
    swiglu_backward(thrust::raw_pointer_cast(d_dinp.data()),
                    thrust::raw_pointer_cast(d_dout.data()),
                    thrust::raw_pointer_cast(d_inp.data()),
                    nullptr, B, T, C, 0);
    std::vector<float> h_dinp = from_device(d_dinp);
    for (size_t i = 0; i < n_inp; ++i) {
        REQUIRE(h_dinp[i] == Catch::Approx(h_dinp_cpu[i]).margin(1e-5f));
    }

    // GPU backward with absmax
    swiglu_backward(thrust::raw_pointer_cast(d_dinp.data()),
                    thrust::raw_pointer_cast(d_dout.data()),
                    thrust::raw_pointer_cast(d_inp.data()),
                    thrust::raw_pointer_cast(d_absmax.data()), B, T, C, 0);
    thrust::copy(d_absmax.begin(), d_absmax.end(), &h_absmax);
    REQUIRE(h_absmax == Catch::Approx(cpu_absmax_bwd).margin(1e-6f));
}

TEST_CASE("swiglu forward/backward bfloat16 matches CPU (emulated)", "[kernels][swiglu][bf16]") {
    const auto& cfg = testing_config::get_test_config();
    const int B = cfg.B;
    const int T = cfg.T; // ensure divisibility
    const int C = cfg.C; // multiple of 8 for bf16 path
    // Validate constraints for the bf16 kernels
    long long prod = 1LL * B * T * C;
    bool ok = (C % 8 == 0) && (prod % 2048 == 0);
    if (!ok) {
        INFO("Invalid sizes for bf16: require C % 8 == 0 and (B*T*C) % 2048 == 0");
        INFO("Provided: B=" << B << ", T=" << T << ", C=" << C << ", B*T*C=" << prod);
        FAIL("Aborting bf16 test due to invalid size configuration");
    }
    const int C2 = 2 * C;
    const size_t n_inp = static_cast<size_t>(B) * T * C2;
    const size_t n_out = static_cast<size_t>(B) * T * C;

    // Prepare float inputs and convert to bf16 storage for GPU
    std::vector<float> h_inp_f = uniform_host(n_inp, -1.f, 1.f, 1337ull);
    std::vector<nv_bfloat16> h_inp_bf16 = to_bf16(h_inp_f);

    // CPU forward but quantized to bf16 first (to match kernel math in bf16)
    // Convert inputs to bf16 -> back to float to emulate bf16 arithmetic for x1, x2
    std::vector<float> h_inp_q = round_bf16(h_inp_f);

    std::vector<float> h_out_cpu(n_out);
    swiglu_forward_cpu(h_out_cpu.data(), h_inp_q.data(), B, T, C);

    // Quantize output to bf16 too, since kernel stores bf16
    std::vector<float> h_out_cpu_q = round_bf16(h_out_cpu);
    float cpu_absmax_fwd = max_abs(h_out_cpu_q.data(), n_out);

    // Device allocations via Thrust
    thrust::device_vector<nv_bfloat16> d_inp = to_device(h_inp_bf16);
    thrust::device_vector<nv_bfloat16> d_out(n_out);
    thrust::device_vector<nv_bfloat16> d_dinp(n_inp);
    thrust::device_vector<float> d_absmax(1);

    // Forward with absmax (also validates without by comparing values)
    swiglu_forward(thrust::raw_pointer_cast(d_out.data()),
                   thrust::raw_pointer_cast(d_inp.data()),
                   thrust::raw_pointer_cast(d_absmax.data()), B, T, C, 0);
    std::vector<nv_bfloat16> h_out_bf16 = from_device(d_out);

    std::vector<float> h_out(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        uint16_t bits;
        std::memcpy(&bits, &h_out_bf16[i], sizeof(bits));
        h_out[i] = bf16_bits_to_float(bits);
        REQUIRE(h_out[i] == Catch::Approx(h_out_cpu_q[i]).margin(3e-2f));
    }
    float h_absmax = 0.f;
    thrust::copy(d_absmax.begin(), d_absmax.end(), &h_absmax);
    REQUIRE(h_absmax == Catch::Approx(cpu_absmax_fwd).margin(5e-3f));

    // Backward
    std::vector<float> h_dout_f = uniform_host(n_out, -0.5f, 0.5f, 424242ull);
    // Quantize dout to bf16 for GPU input and for CPU emulation
    std::vector<nv_bfloat16> h_dout_bf16 = to_bf16(h_dout_f);
    std::vector<float> h_dout_q = round_bf16(h_dout_f);
    thrust::device_vector<nv_bfloat16> d_dout = to_device(h_dout_bf16);

    // CPU backward on quantized inputs/grad
    std::vector<float> h_dinp_cpu(n_inp);
    swiglu_backward_cpu(h_dinp_cpu.data(), h_dout_q.data(), h_inp_q.data(), B, T, C);
    // Quantize gradients as kernel outputs bf16
    h_dinp_cpu = round_bf16(h_dinp_cpu);
    float cpu_absmax_bwd = max_abs(h_dinp_cpu.data(), n_inp);

    swiglu_backward(thrust::raw_pointer_cast(d_dinp.data()),
                    thrust::raw_pointer_cast(d_dout.data()),
                    thrust::raw_pointer_cast(d_inp.data()),
                    thrust::raw_pointer_cast(d_absmax.data()), B, T, C, 0);
    std::vector<nv_bfloat16> h_dinp_bf16 = from_device(d_dinp);

    for (size_t i = 0; i < n_inp; ++i) {
        uint16_t bits;
        std::memcpy(&bits, &h_dinp_bf16[i], sizeof(bits));
        float v = bf16_bits_to_float(bits);
        REQUIRE(v == Catch::Approx(h_dinp_cpu[i]).margin(3e-2f));
    }
    thrust::copy(d_absmax.begin(), d_absmax.end(), &h_absmax);
    REQUIRE(h_absmax == Catch::Approx(cpu_absmax_bwd).margin(5e-3f));
}

TEST_CASE("swiglu_forward_quant is bit-perfect to swiglu_forward + quant_with_absmax", "[kernels][swiglu][quant][fp8]") {
    // This test verifies that directly computing SwiGLU in bf16 and writing fp8 (swiglu_forward_quant)
    // is bit-identical to: bf16 swiglu_forward -> take absmax -> quantize_with_abs_max to fp8.
    const auto& cfg = testing_config::get_test_config();
    const int B = cfg.B;
    const int T = cfg.T;
    const int C = cfg.C;

    // Kernel constraints: bf16 path requires C % 8 == 0 and (B*T*C) % 1024 == 0 (we choose stronger 2048 to match other test)
    long long prod = 1LL * B * T * C;
    bool ok = (C % 8 == 0) && (prod % 2048 == 0);
    if (!ok) {
        INFO("Invalid sizes for bf16/fp8 quant test: require C % 8 == 0 and (B*T*C) % 2048 == 0");
        INFO("Provided: B=" << B << ", T=" << T << ", C=" << C << ", B*T*C=" << prod);
        FAIL("Aborting swiglu quant test due to invalid size configuration");
    }

    const int C2 = 2 * C;
    const size_t n_inp = static_cast<size_t>(B) * T * C2;
    const size_t n_out = static_cast<size_t>(B) * T * C;

    // Prepare deterministic inputs in bf16
    std::vector<float> h_inp_f = uniform_host(n_inp, -1.f, 1.f, 1337ull);
    std::vector<nv_bfloat16> h_inp_bf16 = to_bf16(h_inp_f);

    thrust::device_vector<nv_bfloat16> d_inp = to_device(h_inp_bf16);
    thrust::device_vector<nv_bfloat16> d_out_bf16(n_out);
    thrust::device_vector<float> d_absmax(1);

    // Path A: swiglu_forward (bf16) with absmax, then quantize_with_abs_max to fp8
    swiglu_forward(thrust::raw_pointer_cast(d_out_bf16.data()),
                   thrust::raw_pointer_cast(d_inp.data()),
                   thrust::raw_pointer_cast(d_absmax.data()), B, T, C, 0);

    // Device properties for quantize_with_abs_max launcher
    cudaDeviceProp dp{};
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&dp, dev));

    thrust::device_vector<__nv_fp8_e4m3> d_fp8_from_two_step(n_out);
    thrust::device_vector<float> d_scale_two_step(1);

    quantize_with_abs_max(thrust::raw_pointer_cast(d_fp8_from_two_step.data()),
                          thrust::raw_pointer_cast(d_scale_two_step.data()),
                          thrust::raw_pointer_cast(d_out_bf16.data()),
                          thrust::raw_pointer_cast(d_absmax.data()),
                          static_cast<long>(n_out), dp, 0);

    // Path B: swiglu_forward_quant (direct bf16 -> fp8) using the same absmax pointer
    thrust::device_vector<__nv_fp8_e4m3> d_fp8_direct(n_out);
    thrust::device_vector<float> d_scale_direct(1);

    swiglu_forward_quant(thrust::raw_pointer_cast(d_fp8_direct.data()),
                         thrust::raw_pointer_cast(d_scale_direct.data()),
                         thrust::raw_pointer_cast(d_inp.data()),
                         thrust::raw_pointer_cast(d_absmax.data()),
                         B, T, C, 0);

    // Compare: scales must match bit-perfectly
    std::vector<float> h_scale_two_step = from_device(d_scale_two_step);
    std::vector<float> h_scale_direct   = from_device(d_scale_direct);
    REQUIRE(h_scale_two_step.size() == 1);
    REQUIRE(h_scale_direct.size() == 1);
    REQUIRE(h_scale_two_step[0] == h_scale_direct[0]);

    // Compare: fp8 buffers must be byte-identical
    std::vector<__nv_fp8_e4m3> h_fp8_two_step = from_device(d_fp8_from_two_step);
    std::vector<__nv_fp8_e4m3> h_fp8_direct  = from_device(d_fp8_direct);
    REQUIRE(h_fp8_two_step.size() == h_fp8_direct.size());
    const size_t nbytes = h_fp8_two_step.size() * sizeof(__nv_fp8_e4m3);
    REQUIRE(std::memcmp(h_fp8_two_step.data(), h_fp8_direct.data(), nbytes) == 0);
}
