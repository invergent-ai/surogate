// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// The native GRPO custom-dloss kernel run one chunk window at a time (chunked-sequence
// GRPO) against the same kernel run over the whole packed sequence, and both against a
// scalar CPU reference written from the loss definition. Gradients are per token, so they
// never depended on the windowing; the metrics are per-sample means, and those must not
// either. Layout as in the DPO test: trainer_logprob(out_idx) = -losses[out_idx] carries
// logical token out_idx + 1, and custom_dloss[out_idx] is the seed for that token.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/kernels.h"

namespace {

// Mirrors GrpoMetricOffset in kernels/fused_classifier.cu.
enum {
    kPolicyLoss = 0,
    kMismatchKl = 1,
    kMaskedMismatchKl = 2,
    kUnmaskedMismatchKl = 3,
    kIsMasked = 4,
    kIsMaskedLow = 5,
    kIsMaskedHigh = 6,
    kTeacherKl = 7,
    kSampleCount = 8,
    kKeepTokens = 9,
    kTotalTokens = 10,
    kMetricCount = 11,
};

struct GrpoCfg {
    float loss_scale = 23.0f;
    float ipo_mask_low = 0.2f;
    float ipo_mask_high = 0.2f;
    float adv_tau = 1.0f;
    float teacher_tau = 0.0f;
    float kl_tau = 0.1f;
    float ratio_clip = 7.389056f;
};

struct Fixture {
    std::vector<float> losses;              // [T], -trainer_logprob of logical token t + 1
    std::vector<float> inference_logprobs;  // [T], logical layout
    std::vector<float> advantages;          // [T], logical layout
    std::vector<std::uint8_t> loss_mask;    // [T], logical layout
    std::vector<std::int32_t> starts;
    std::vector<std::int32_t> ends;
};

// Deterministic values in [0, 1) without <random>, so the fixture is the same everywhere.
float unit(std::uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(state >> 8) / 16777216.0f;
}

// `prompt[i]` leading tokens of sample i carry no loss. Trainer and inference log-probs are
// drawn so that both IPO mask sides fire and most tokens are kept.
Fixture make_fixture(const std::vector<int>& lengths, const std::vector<int>& prompt) {
    Fixture fx;
    std::uint32_t state = 12345u;
    int cursor = 0;
    for (std::size_t i = 0; i < lengths.size(); ++i) {
        fx.starts.push_back(cursor);
        fx.ends.push_back(cursor + lengths[i]);
        for (int t = 0; t < lengths[i]; ++t) {
            fx.loss_mask.push_back(t >= prompt[i] ? 1 : 0);
        }
        cursor += lengths[i];
    }
    for (int t = 0; t < cursor; ++t) {
        const float trainer_lp = -0.05f - 2.0f * unit(state);
        const float shift = 1.6f * unit(state) - 0.8f;
        fx.losses.push_back(-trainer_lp);
        fx.inference_logprobs.push_back(std::min(-0.01f, trainer_lp + shift));
        fx.advantages.push_back(2.0f * unit(state) - 1.0f);
    }
    // inference_logprobs is indexed by logical token: token t + 1 pairs with losses[t].
    std::rotate(fx.inference_logprobs.rbegin(), fx.inference_logprobs.rbegin() + 1, fx.inference_logprobs.rend());
    return fx;
}

// Sums of per-sample means, as the kernel accumulates them (the caller divides by
// metrics[kSampleCount]); keep/total token counts are plain sums.
void reference_grpo(const Fixture& fx, const GrpoCfg& cfg, std::vector<float>& dloss, std::vector<float>& metrics) {
    dloss.assign(fx.losses.size(), 0.0f);
    std::vector<double> out(kMetricCount, 0.0);
    for (std::size_t s = 0; s < fx.starts.size(); ++s) {
        double policy = 0, mismatch = 0, masked_mismatch = 0, unmasked_mismatch = 0;
        double masked = 0, masked_low = 0, masked_high = 0, keep = 0, total = 0;
        for (int t = fx.starts[s] + 1; t < fx.ends[s]; ++t) {
            if (fx.loss_mask[t] == 0) {
                continue;
            }
            const double trainer_lp = -static_cast<double>(fx.losses[t - 1]);
            const double inference_lp = fx.inference_logprobs[t];
            const double log_ratio = trainer_lp - inference_lp;
            const double ratio = std::exp(log_ratio);
            const double probs_diff = std::exp(trainer_lp) - std::exp(inference_lp);
            const bool low = probs_diff < -cfg.ipo_mask_low;
            const bool high = probs_diff > cfg.ipo_mask_high;
            const double mismatch_kl = ratio - log_ratio - 1.0;
            double seed = -2.0 * cfg.kl_tau * log_ratio;
            policy += cfg.kl_tau * log_ratio * log_ratio;
            if (!(low || high)) {
                const double pg = cfg.adv_tau * fx.advantages[t] * std::min(ratio, static_cast<double>(cfg.ratio_clip));
                seed += pg;
                policy -= pg;
                keep += 1;
                unmasked_mismatch += mismatch_kl;
            } else {
                masked += 1;
                masked_mismatch += mismatch_kl;
            }
            mismatch += mismatch_kl;
            masked_low += low ? 1 : 0;
            masked_high += high ? 1 : 0;
            total += 1;
            dloss[t - 1] = static_cast<float>(seed / cfg.loss_scale);
        }
        if (total == 0) {
            continue;
        }
        out[kPolicyLoss] += policy / total;
        out[kMismatchKl] += mismatch / total;
        out[kMaskedMismatchKl] += masked_mismatch / std::max(masked, 1.0);
        out[kUnmaskedMismatchKl] += unmasked_mismatch / std::max(keep, 1.0);
        out[kIsMasked] += masked / total;
        out[kIsMaskedLow] += masked_low / total;
        out[kIsMaskedHigh] += masked_high / total;
        out[kSampleCount] += 1;
        out[kKeepTokens] += keep;
        out[kTotalTokens] += total;
    }
    metrics.assign(out.begin(), out.end());
}

template <class T>
T* to_device(const std::vector<T>& v) {
    T* d = nullptr;
    REQUIRE(cudaMalloc(&d, v.size() * sizeof(T)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
    return d;
}

bool gpu_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// Run the kernel over [0, T) in windows of `window` tokens, last window first — the order
// the chunked GRPO step walks them in. window == T is the unchunked step.
void run_kernel(const Fixture& fx,
                const GrpoCfg& cfg,
                int window,
                std::vector<float>& dloss,
                std::vector<float>& metrics) {
    const int T = static_cast<int>(fx.losses.size());
    const int sample_count = static_cast<int>(fx.starts.size());
    REQUIRE(T % window == 0);

    float* d_losses = to_device(fx.losses);
    float* d_inference = to_device(fx.inference_logprobs);
    float* d_advantages = to_device(fx.advantages);
    std::uint8_t* d_mask = to_device(fx.loss_mask);
    std::int32_t* d_starts = to_device(fx.starts);
    std::int32_t* d_ends = to_device(fx.ends);
    float* d_dloss = nullptr;
    float* d_metrics = nullptr;
    float* d_sample_metrics = nullptr;
    const std::size_t sample_metric_bytes = static_cast<std::size_t>(sample_count) * kMetricCount * sizeof(float);
    REQUIRE(cudaMalloc(&d_dloss, window * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_metrics, kMetricCount * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_sample_metrics, sample_metric_bytes) == cudaSuccess);
    // What grpo_native_upload_full does once per micro-batch.
    REQUIRE(cudaMemset(d_metrics, 0, kMetricCount * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemset(d_sample_metrics, 0, sample_metric_bytes) == cudaSuccess);

    dloss.assign(T, 0.0f);
    for (int window_start = T - window; window_start >= 0; window_start -= window) {
        // losses and custom_dloss are chunk-local; everything else is global.
        compute_grpo_custom_dloss(d_dloss,
                                  d_metrics,
                                  d_sample_metrics,
                                  d_losses + window_start,
                                  d_inference,
                                  d_advantages,
                                  d_mask,
                                  /*teacher_logprobs=*/nullptr,
                                  d_starts,
                                  d_ends,
                                  sample_count,
                                  window,
                                  cfg.loss_scale,
                                  cfg.ipo_mask_low,
                                  cfg.ipo_mask_high,
                                  cfg.adv_tau,
                                  cfg.teacher_tau,
                                  cfg.kl_tau,
                                  cfg.ratio_clip,
                                  window_start,
                                  /*stream=*/nullptr);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        REQUIRE(cudaMemcpy(dloss.data() + window_start, d_dloss, window * sizeof(float), cudaMemcpyDeviceToHost) ==
                cudaSuccess);
    }
    metrics.assign(kMetricCount, 0.0f);
    REQUIRE(cudaMemcpy(metrics.data(), d_metrics, kMetricCount * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);

    cudaFree(d_losses);
    cudaFree(d_inference);
    cudaFree(d_advantages);
    cudaFree(d_mask);
    cudaFree(d_starts);
    cudaFree(d_ends);
    cudaFree(d_dloss);
    cudaFree(d_metrics);
    cudaFree(d_sample_metrics);
}

void require_close(const std::vector<float>& actual, const std::vector<float>& expected) {
    REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        INFO("index " << i);
        REQUIRE(actual[i] == Catch::Approx(expected[i]).epsilon(1e-4).margin(1e-6));
    }
}

void check_fixture(const Fixture& fx, int window) {
    const GrpoCfg cfg;
    std::vector<float> ref_dloss, ref_metrics;
    reference_grpo(fx, cfg, ref_dloss, ref_metrics);
    // The fixture must exercise what it claims to: kept and masked tokens on both sides.
    REQUIRE(ref_metrics[kKeepTokens] > 0.0f);
    REQUIRE(ref_metrics[kIsMaskedLow] > 0.0f);
    REQUIRE(ref_metrics[kIsMaskedHigh] > 0.0f);

    std::vector<float> full_dloss, full_metrics, chunked_dloss, chunked_metrics;
    run_kernel(fx, cfg, static_cast<int>(fx.losses.size()), full_dloss, full_metrics);
    run_kernel(fx, cfg, window, chunked_dloss, chunked_metrics);

    require_close(full_dloss, ref_dloss);
    require_close(full_metrics, ref_metrics);
    require_close(chunked_dloss, ref_dloss);
    require_close(chunked_metrics, ref_metrics);
}

}  // namespace

TEST_CASE("native GRPO metrics do not depend on the chunk windows: one sample, prompt fills chunk 0",
          "[grpo][dloss][cuda]") {
    if (!gpu_available()) {
        SKIP("No CUDA device available");
    }
    // The agentic shape: a single sample per row whose prompt is longer than a chunk, so the
    // window that holds the sample's start has no loss token at all.
    const Fixture fx = make_fixture({48}, {19});
    check_fixture(fx, 8);
}

TEST_CASE("native GRPO metrics do not depend on the chunk windows: packed samples", "[grpo][dloss][cuda]") {
    if (!gpu_available()) {
        SKIP("No CUDA device available");
    }
    // Sample 0 spans four windows with an all-prompt first one, sample 1 starts mid-window
    // and crosses two boundaries, sample 2 sits inside one window, sample 3 has no loss token.
    const Fixture fx = make_fixture({27, 15, 3, 3}, {10, 2, 1, 3});
    check_fixture(fx, 8);
}
