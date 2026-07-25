// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Parity tests for the native DPO custom-dloss CUDA kernel against a scalar CPU
// reference implementing the sigmoid-DPO per-pair formula. The reference and the
// kernel share the shifted target-slot convention of the GRPO path:
// trainer_logprob(out_idx) = -losses[out_idx] carries logical token out_idx + 1,
// and custom_dloss[out_idx] is the gradient seed for that logical token.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/kernels.h"

namespace {

struct DpoCfg {
    float beta = 0.1f;
    int length_norm = 0;
    float loss_scale = 1.0f;
};

// Sum the response-token policy/reference log-probs of one sample [start, end).
// Response token t (loss_mask[t] != 0) is seeded at out_idx = t - 1.
void seq_sums(const std::vector<float>& losses,
              const std::vector<float>& ref,
              const std::vector<std::uint8_t>& mask,
              int start,
              int end,
              double& s_pol,
              double& s_ref,
              int& len) {
    s_pol = 0.0;
    s_ref = 0.0;
    len = 0;
    for (int t = start + 1; t < end; ++t) {
        if (mask[t] == 0) {
            continue;
        }
        const int out_idx = t - 1;
        s_pol += -static_cast<double>(losses[out_idx]);
        s_ref += static_cast<double>(ref[out_idx]);
        ++len;
    }
}

// metrics_out = {loss, correct, margin, pair_count} for a single pair, matching
// the kernel's per-pair accumulation into metrics[4].
void reference_dpo(const std::vector<float>& losses,
                   const std::vector<float>& ref,
                   const std::vector<std::uint8_t>& mask,
                   const std::vector<std::int32_t>& starts,
                   const std::vector<std::int32_t>& ends,
                   int chosen,
                   int rejected,
                   const DpoCfg& cfg,
                   std::vector<float>& dloss_out,
                   std::vector<float>& metrics_out) {
    dloss_out.assign(losses.size(), 0.0f);
    double sc_pol, sc_ref, sr_pol, sr_ref;
    int len_c, len_r;
    seq_sums(losses, ref, mask, starts[chosen], ends[chosen], sc_pol, sc_ref, len_c);
    seq_sums(losses, ref, mask, starts[rejected], ends[rejected], sr_pol, sr_ref, len_r);
    const double wc = (cfg.length_norm && len_c > 0) ? 1.0 / static_cast<double>(len_c) : 1.0;
    const double wr = (cfg.length_norm && len_r > 0) ? 1.0 / static_cast<double>(len_r) : 1.0;
    const double margin = static_cast<double>(cfg.beta) * ((sc_pol - sc_ref) * wc - (sr_pol - sr_ref) * wr);
    const double g = static_cast<double>(cfg.beta) * (1.0 / (1.0 + std::exp(margin)));  // beta * sigmoid(-margin)
    const double inv_scale = (cfg.loss_scale == 0.0f) ? 1.0 : 1.0 / static_cast<double>(cfg.loss_scale);
    for (int t = starts[chosen] + 1; t < ends[chosen]; ++t) {
        if (mask[t] != 0) {
            dloss_out[t - 1] = static_cast<float>((g * wc) * inv_scale);
        }
    }
    for (int t = starts[rejected] + 1; t < ends[rejected]; ++t) {
        if (mask[t] != 0) {
            dloss_out[t - 1] = static_cast<float>((-g * wr) * inv_scale);
        }
    }
    const double loss = -std::log(1.0 / (1.0 + std::exp(-margin)));  // -log sigmoid(margin)
    metrics_out = {static_cast<float>(loss), margin > 0.0 ? 1.0f : 0.0f, static_cast<float>(margin), 1.0f};
}

template <class T>
T* to_device(const std::vector<T>& v) {
    T* d = nullptr;
    REQUIRE(cudaMalloc(&d, v.size() * sizeof(T)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
    return d;
}

void require_close(const std::vector<float>& actual, const std::vector<float>& expected) {
    REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        INFO("index " << i);
        REQUIRE(actual[i] == Catch::Approx(expected[i]).epsilon(1e-5).margin(1e-6));
    }
}

bool gpu_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// Two samples packed back-to-back: sample 0 = chosen [0, 5), sample 1 = rejected [5, 9), BT = 9.
struct Fixture {
    std::vector<float> losses = {0.1f, 0.5f, 0.3f, 0.2f, 0.4f, 0.6f, 0.7f, 0.2f, 0.9f};
    std::vector<float> ref = {-0.2f, -0.4f, -0.6f, -0.3f, -0.5f, -0.7f, -0.1f, -0.8f, -0.3f};
    std::vector<std::uint8_t> mask = {0, 1, 1, 1, 0, 0, 1, 1, 0};
    std::vector<std::int32_t> starts = {0, 5};
    std::vector<std::int32_t> ends = {5, 9};
    std::vector<std::int32_t> pair_chosen = {0};
    std::vector<std::int32_t> pair_rejected = {1};
};

void run_case(const DpoCfg& cfg) {
    Fixture fx;
    const int BT = static_cast<int>(fx.losses.size());
    const int pair_count = static_cast<int>(fx.pair_chosen.size());

    std::vector<float> ref_dloss, ref_metrics;
    reference_dpo(fx.losses, fx.ref, fx.mask, fx.starts, fx.ends, 0, 1, cfg, ref_dloss, ref_metrics);

    float* d_losses = to_device(fx.losses);
    float* d_ref = to_device(fx.ref);
    std::uint8_t* d_mask = to_device(fx.mask);
    std::int32_t* d_starts = to_device(fx.starts);
    std::int32_t* d_ends = to_device(fx.ends);
    std::int32_t* d_pc = to_device(fx.pair_chosen);
    std::int32_t* d_pr = to_device(fx.pair_rejected);
    float* d_dloss = nullptr;
    float* d_metrics = nullptr;
    REQUIRE(cudaMalloc(&d_dloss, BT * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_metrics, 4 * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemset(d_metrics, 0, 4 * sizeof(float)) == cudaSuccess);

    compute_dpo_custom_dloss(d_dloss,
                             d_metrics,
                             d_losses,
                             d_ref,
                             d_mask,
                             d_starts,
                             d_ends,
                             d_pc,
                             d_pr,
                             pair_count,
                             BT,
                             cfg.loss_scale,
                             cfg.beta,
                             cfg.length_norm,
                             nullptr);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<float> got_dloss(BT, 0.0f);
    std::vector<float> got_metrics(4, 0.0f);
    REQUIRE(cudaMemcpy(got_dloss.data(), d_dloss, BT * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(got_metrics.data(), d_metrics, 4 * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);

    require_close(got_dloss, ref_dloss);
    require_close(got_metrics, ref_metrics);

    for (void* p : {static_cast<void*>(d_losses),
                    static_cast<void*>(d_ref),
                    static_cast<void*>(d_mask),
                    static_cast<void*>(d_starts),
                    static_cast<void*>(d_ends),
                    static_cast<void*>(d_pc),
                    static_cast<void*>(d_pr),
                    static_cast<void*>(d_dloss),
                    static_cast<void*>(d_metrics)}) {
        if (p) cudaFree(p);
    }
}

}  // namespace

TEST_CASE("dpo dloss kernel matches reference (no length norm)", "[dpo][kernels]") {
    if (!gpu_available()) SKIP("no CUDA device available");
    run_case(DpoCfg{0.1f, 0, 1.0f});
}

TEST_CASE("dpo dloss kernel matches reference (length norm)", "[dpo][kernels]") {
    if (!gpu_available()) SKIP("no CUDA device available");
    run_case(DpoCfg{0.1f, 1, 1.0f});
}

TEST_CASE("dpo dloss kernel matches reference (loss scale)", "[dpo][kernels]") {
    if (!gpu_available()) SKIP("no CUDA device available");
    run_case(DpoCfg{0.25f, 0, 4.0f});
}

TEST_CASE("dpo dloss step-0 identity gives loss log2 and +/- beta/2", "[dpo][kernels]") {
    if (!gpu_available()) SKIP("no CUDA device available");
    // pi_theta == pi_ref => margin 0 => loss log 2, g = beta/2.
    Fixture fx;
    for (std::size_t i = 0; i < fx.losses.size(); ++i) {
        fx.ref[i] = -fx.losses[i];
    }
    const DpoCfg cfg{0.1f, 0, 1.0f};
    const int BT = static_cast<int>(fx.losses.size());

    std::vector<float> ref_dloss, ref_metrics;
    reference_dpo(fx.losses, fx.ref, fx.mask, fx.starts, fx.ends, 0, 1, cfg, ref_dloss, ref_metrics);
    REQUIRE(ref_metrics[0] == Catch::Approx(std::log(2.0)).epsilon(1e-5));
    for (int t = fx.starts[0] + 1; t < fx.ends[0]; ++t) {
        if (fx.mask[t]) REQUIRE(ref_dloss[t - 1] == Catch::Approx(+0.05f).epsilon(1e-5));
    }
    for (int t = fx.starts[1] + 1; t < fx.ends[1]; ++t) {
        if (fx.mask[t]) REQUIRE(ref_dloss[t - 1] == Catch::Approx(-0.05f).epsilon(1e-5));
    }

    float* d_losses = to_device(fx.losses);
    float* d_ref = to_device(fx.ref);
    std::uint8_t* d_mask = to_device(fx.mask);
    std::int32_t* d_starts = to_device(fx.starts);
    std::int32_t* d_ends = to_device(fx.ends);
    std::int32_t* d_pc = to_device(fx.pair_chosen);
    std::int32_t* d_pr = to_device(fx.pair_rejected);
    float* d_dloss = nullptr;
    float* d_metrics = nullptr;
    REQUIRE(cudaMalloc(&d_dloss, BT * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_metrics, 4 * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemset(d_metrics, 0, 4 * sizeof(float)) == cudaSuccess);

    compute_dpo_custom_dloss(d_dloss,
                             d_metrics,
                             d_losses,
                             d_ref,
                             d_mask,
                             d_starts,
                             d_ends,
                             d_pc,
                             d_pr,
                             1,
                             BT,
                             cfg.loss_scale,
                             cfg.beta,
                             cfg.length_norm,
                             nullptr);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<float> got_dloss(BT, 0.0f);
    std::vector<float> got_metrics(4, 0.0f);
    REQUIRE(cudaMemcpy(got_dloss.data(), d_dloss, BT * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(got_metrics.data(), d_metrics, 4 * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    require_close(got_dloss, ref_dloss);
    REQUIRE(got_metrics[0] == Catch::Approx(std::log(2.0)).epsilon(1e-5));

    for (void* p : {static_cast<void*>(d_losses),
                    static_cast<void*>(d_ref),
                    static_cast<void*>(d_mask),
                    static_cast<void*>(d_starts),
                    static_cast<void*>(d_ends),
                    static_cast<void*>(d_pc),
                    static_cast<void*>(d_pr),
                    static_cast<void*>(d_dloss),
                    static_cast<void*>(d_metrics)}) {
        if (p) cudaFree(p);
    }
}
