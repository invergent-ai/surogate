// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "kernels/kernels.h"
#include "utilities/tensor.h"
#include "../utilities/test_config.h"
#include "../utilities/test_utils.h"

using namespace testing_utils;

namespace {

Tensor tensor_from_device_ptr(ETensorDType dt, std::byte* ptr, const std::vector<long>& shape) {
    return Tensor::from_pointer(ptr, /*device=*/0, dt, shape);
}

} // namespace

TEST_CASE("qk_norm+rope fused forward matches baseline (fp32)", "[kernels][qk_norm][rope][fp32]") {
    const auto& cfg = testing_config::get_test_config();
    const int B = cfg.B;
    const int T = cfg.T;

    const int Hq = 8;
    const int Hkv = 4;
    const int HS = 64;
    const int qkv_channels = (Hq + 2 * Hkv) * HS;
    const float eps = 1e-5f;

    const size_t qkv_elems = static_cast<size_t>(B) * static_cast<size_t>(T) * static_cast<size_t>(qkv_channels);
    const size_t q_rstd_elems = static_cast<size_t>(B) * static_cast<size_t>(T) * static_cast<size_t>(Hq);
    const size_t k_rstd_elems = static_cast<size_t>(B) * static_cast<size_t>(T) * static_cast<size_t>(Hkv);
    const size_t freqs_elems = static_cast<size_t>(T) * static_cast<size_t>(HS);

    std::vector<float> h_qkv0 = uniform_host((long)qkv_elems, -1.0f, 1.0f, 1234ull);
    std::vector<float> h_qw = uniform_host(HS, 0.5f, 1.5f, 4321ull);
    std::vector<float> h_kw = uniform_host(HS, 0.5f, 1.5f, 9876ull);

    std::vector<float> h_freqs(freqs_elems);
    precompute_freqs_cis(h_freqs.data(), HS, T, 10000.0f);

    // Baseline buffers
    thrust::device_vector<float> d_qkv_base = to_device(h_qkv0);
    thrust::device_vector<float> d_q_rstd_base(q_rstd_elems);
    thrust::device_vector<float> d_k_rstd_base(k_rstd_elems);
    thrust::device_vector<float> d_qw = to_device(h_qw);
    thrust::device_vector<float> d_kw = to_device(h_kw);
    thrust::device_vector<float> d_freqs = to_device(h_freqs);

    Tensor qkv_base = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_qkv_base.data())), {B, T, qkv_channels});
    Tensor q_rstd_base = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_q_rstd_base.data())), {B, T, Hq});
    Tensor k_rstd_base = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_k_rstd_base.data())), {B, T, Hkv});
    Tensor q_weight = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_qw.data())), {HS});
    Tensor k_weight = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_kw.data())), {HS});
    Tensor freqs = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_freqs.data())), {T, HS});

    // Baseline: QK-norm then RoPE
    const int q_rows = Hq * HS;
    qkv_head_rmsnorm_forward(qkv_base, q_rstd_base, q_weight, eps, B, T, qkv_channels, Hq, HS, /*channel_offset=*/0, 0);
    qkv_head_rmsnorm_forward(qkv_base, k_rstd_base, k_weight, eps, B, T, qkv_channels, Hkv, HS, /*channel_offset=*/q_rows, 0);
    rope_forward(qkv_base, qkv_base, freqs, /*position_ids=*/nullptr, /*abs_max_ptr=*/nullptr, B, T, Hq, Hkv, HS, 0);

    // Fused buffers
    thrust::device_vector<float> d_qkv_fused = to_device(h_qkv0);
    thrust::device_vector<float> d_q_rstd_fused(q_rstd_elems);
    thrust::device_vector<float> d_k_rstd_fused(k_rstd_elems);

    Tensor qkv_fused = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_qkv_fused.data())), {B, T, qkv_channels});
    Tensor q_rstd_fused = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_q_rstd_fused.data())), {B, T, Hq});
    Tensor k_rstd_fused = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_k_rstd_fused.data())), {B, T, Hkv});

    qkv_qk_norm_rope_forward(qkv_fused, q_rstd_fused, k_rstd_fused, q_weight, k_weight, freqs, /*position_ids=*/nullptr,
                             eps, B, T, Hq, Hkv, HS, 0);

    std::vector<float> h_base = from_device(d_qkv_base);
    std::vector<float> h_fused = from_device(d_qkv_fused);
    REQUIRE(h_base.size() == h_fused.size());
    for (size_t i = 0; i < h_base.size(); ++i) {
        REQUIRE(h_fused[i] == Catch::Approx(h_base[i]).margin(2e-5f));
    }

    std::vector<float> h_qr0 = from_device(d_q_rstd_base);
    std::vector<float> h_qr1 = from_device(d_q_rstd_fused);
    for (size_t i = 0; i < h_qr0.size(); ++i) {
        REQUIRE(h_qr1[i] == Catch::Approx(h_qr0[i]).margin(2e-5f));
    }
    std::vector<float> h_kr0 = from_device(d_k_rstd_base);
    std::vector<float> h_kr1 = from_device(d_k_rstd_fused);
    for (size_t i = 0; i < h_kr0.size(); ++i) {
        REQUIRE(h_kr1[i] == Catch::Approx(h_kr0[i]).margin(2e-5f));
    }
}

TEST_CASE("qk_norm+rope fused backward matches baseline (fp32)", "[kernels][qk_norm][rope][backward][fp32]") {
    const auto& cfg = testing_config::get_test_config();
    const int B = cfg.B;
    const int T = cfg.T;

    const int Hq = 8;
    const int Hkv = 4;
    const int HS = 64;
    const int qkv_channels = (Hq + 2 * Hkv) * HS;
    const float eps = 1e-5f;

    const size_t qkv_elems = static_cast<size_t>(B) * static_cast<size_t>(T) * static_cast<size_t>(qkv_channels);
    const size_t q_rstd_elems = static_cast<size_t>(B) * static_cast<size_t>(T) * static_cast<size_t>(Hq);
    const size_t k_rstd_elems = static_cast<size_t>(B) * static_cast<size_t>(T) * static_cast<size_t>(Hkv);
    const size_t freqs_elems = static_cast<size_t>(T) * static_cast<size_t>(HS);

    std::vector<float> h_qkv0 = uniform_host((long)qkv_elems, -1.0f, 1.0f, 1111ull);
    std::vector<float> h_qw = uniform_host(HS, 0.5f, 1.5f, 2222ull);
    std::vector<float> h_kw = uniform_host(HS, 0.5f, 1.5f, 3333ull);

    std::vector<float> h_freqs(freqs_elems);
    precompute_freqs_cis(h_freqs.data(), HS, T, 10000.0f);

    // Forward baseline to produce qkv_rope + rstds
    thrust::device_vector<float> d_qkv_rope = to_device(h_qkv0);
    thrust::device_vector<float> d_qr(q_rstd_elems);
    thrust::device_vector<float> d_kr(k_rstd_elems);
    thrust::device_vector<float> d_qw = to_device(h_qw);
    thrust::device_vector<float> d_kw = to_device(h_kw);
    thrust::device_vector<float> d_freqs = to_device(h_freqs);

    Tensor qkv_rope = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_qkv_rope.data())), {B, T, qkv_channels});
    Tensor q_rstd = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_qr.data())), {B, T, Hq});
    Tensor k_rstd = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_kr.data())), {B, T, Hkv});
    Tensor q_weight = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_qw.data())), {HS});
    Tensor k_weight = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_kw.data())), {HS});
    Tensor freqs = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_freqs.data())), {T, HS});

    const int q_rows = Hq * HS;
    qkv_head_rmsnorm_forward(qkv_rope, q_rstd, q_weight, eps, B, T, qkv_channels, Hq, HS, /*channel_offset=*/0, 0);
    qkv_head_rmsnorm_forward(qkv_rope, k_rstd, k_weight, eps, B, T, qkv_channels, Hkv, HS, /*channel_offset=*/q_rows, 0);
    rope_forward(qkv_rope, qkv_rope, freqs, /*position_ids=*/nullptr, /*abs_max_ptr=*/nullptr, B, T, Hq, Hkv, HS, 0);

    // Random dy in post-RoPE space.
    std::vector<float> h_dy = uniform_host((long)qkv_elems, -0.5f, 0.5f, 4444ull);

    // Baseline backward:
    // dy_pre = rope_backward(dy_rope)
    // out_pre = rope_backward(out_rope)
    // dW from (dy_pre, out_pre); dx_pre from (dy_pre, out_pre, rstd)
    thrust::device_vector<float> d_dy_pre = to_device(h_dy);
    thrust::device_vector<float> d_out_pre(qkv_elems);
    thrust::device_vector<float> d_dwq(HS);
    thrust::device_vector<float> d_dwk(HS);

    Tensor dy_pre = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_dy_pre.data())), {B, T, qkv_channels});
    Tensor out_pre = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_out_pre.data())), {B, T, qkv_channels});
    Tensor dwq = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_dwq.data())), {HS});
    Tensor dwk = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_dwk.data())), {HS});

    rope_backward(dy_pre, dy_pre, freqs, /*position_ids=*/nullptr, /*abs_max_ptr=*/nullptr, B, T, Hq, Hkv, HS, 0);
    rope_backward(out_pre, qkv_rope, freqs, /*position_ids=*/nullptr, /*abs_max_ptr=*/nullptr, B, T, Hq, Hkv, HS, 0);

    qkv_head_rmsnorm_backward_dweight(dwq, dy_pre, out_pre, q_weight, B, T, qkv_channels, Hq, HS, /*channel_offset=*/0, /*accumulate=*/false, 0);
    qkv_head_rmsnorm_backward_dweight(dwk, dy_pre, out_pre, k_weight, B, T, qkv_channels, Hkv, HS, /*channel_offset=*/q_rows, /*accumulate=*/false, 0);

    qkv_head_rmsnorm_backward_dx(dy_pre, out_pre, q_weight, q_rstd, B, T, qkv_channels, Hq, HS, /*channel_offset=*/0, 0);
    qkv_head_rmsnorm_backward_dx(dy_pre, out_pre, k_weight, k_rstd, B, T, qkv_channels, Hkv, HS, /*channel_offset=*/q_rows, 0);

    // Fused backward (in-place dx) from dy_rope + out_rope.
    thrust::device_vector<float> d_dy_fused = to_device(h_dy);
    thrust::device_vector<float> d_dwq2(HS);
    thrust::device_vector<float> d_dwk2(HS);

    Tensor dy_fused = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_dy_fused.data())), {B, T, qkv_channels});
    Tensor dwq2 = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_dwq2.data())), {HS});
    Tensor dwk2 = tensor_from_device_ptr(ETensorDType::FP32, reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_dwk2.data())), {HS});

    qkv_head_rmsnorm_rope_backward_dweight(dwq2, dy_fused, qkv_rope, q_weight, freqs, /*position_ids=*/nullptr,
                                          B, T, qkv_channels, Hq, HS, /*channel_offset=*/0, /*accumulate=*/false, 0);
    qkv_head_rmsnorm_rope_backward_dweight(dwk2, dy_fused, qkv_rope, k_weight, freqs, /*position_ids=*/nullptr,
                                          B, T, qkv_channels, Hkv, HS, /*channel_offset=*/q_rows, /*accumulate=*/false, 0);

    qkv_head_rmsnorm_rope_backward_dx(dy_fused, qkv_rope, q_weight, q_rstd, freqs, /*position_ids=*/nullptr,
                                      B, T, qkv_channels, Hq, HS, /*channel_offset=*/0, 0);
    qkv_head_rmsnorm_rope_backward_dx(dy_fused, qkv_rope, k_weight, k_rstd, freqs, /*position_ids=*/nullptr,
                                      B, T, qkv_channels, Hkv, HS, /*channel_offset=*/q_rows, 0);

    // Compare dW (component-wise) and dx for Q+K channels.
    std::vector<float> h_dwq0 = from_device(d_dwq);
    std::vector<float> h_dwq1 = from_device(d_dwq2);
    for (int i = 0; i < HS; ++i) {
        REQUIRE(h_dwq1[i] == Catch::Approx(h_dwq0[i]).margin(5e-4f));
    }
    std::vector<float> h_dwk0 = from_device(d_dwk);
    std::vector<float> h_dwk1 = from_device(d_dwk2);
    for (int i = 0; i < HS; ++i) {
        REQUIRE(h_dwk1[i] == Catch::Approx(h_dwk0[i]).margin(5e-4f));
    }

    std::vector<float> h_dx0 = from_device(d_dy_pre);   // dy_pre now holds dx_pre for Q/K
    std::vector<float> h_dx1 = from_device(d_dy_fused); // dy_fused now holds dx_pre for Q/K

    const size_t tokens = static_cast<size_t>(B) * static_cast<size_t>(T);
    for (size_t tok = 0; tok < tokens; ++tok) {
        const size_t base = tok * static_cast<size_t>(qkv_channels);
        // Q and K channels should match.
        for (int c = 0; c < (Hq + Hkv) * HS; ++c) {
            REQUIRE(h_dx1[base + (size_t)c] == Catch::Approx(h_dx0[base + (size_t)c]).margin(5e-4f));
        }
        // V channels are identity through RoPE/QK-norm; both should still equal original dy.
        for (int c = (Hq + Hkv) * HS; c < qkv_channels; ++c) {
            REQUIRE(h_dx1[base + (size_t)c] == Catch::Approx(h_dx0[base + (size_t)c]).margin(5e-4f));
        }
    }
}

