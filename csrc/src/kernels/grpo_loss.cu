// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// GRPO custom dloss construction on GPU.

#include "kernels.h"

#include <cstdint>
#include <stdexcept>

#include "utilities/utils.h"

namespace {

__global__ void build_grpo_custom_dloss_kernel(
    const float* losses,
    const float* inference_logprobs,
    const float* advantages,
    const std::uint8_t* loss_mask,
    float* out_dloss,
    float* stats,
    int B,
    int T,
    float kl_tau,
    float adv_tau,
    float ipo_mask_low,
    float ipo_mask_high,
    float inv_loss_scale) {
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
    const int n = B * T;
    if (idx >= n) {
        return;
    }

    const int t = idx % T;
    if (t + 1 >= T) {
        out_dloss[idx] = 0.0f;
        return;
    }

    const int j = idx + 1;
    if (loss_mask[j] == 0u) {
        out_dloss[idx] = 0.0f;
        return;
    }

    const float trainer_logprob = -losses[idx];
    const float inference_logprob = inference_logprobs[j];
    const float log_ratio = trainer_logprob - inference_logprob;
    const float ratio = __expf(log_ratio);

    const float trainer_prob = __expf(trainer_logprob);
    const float inference_prob = __expf(inference_logprob);
    const float probs_diff = trainer_prob - inference_prob;
    const bool is_masked = (probs_diff > ipo_mask_high) || (probs_diff < -ipo_mask_low);
    const bool keep = !is_masked;

    const float scaled_adv = adv_tau * advantages[j];
    const float pg_term = keep ? (scaled_adv * ratio) : 0.0f;
    const float kl_term = kl_tau * log_ratio * log_ratio;

    float grad = pg_term;
    grad -= 2.0f * kl_tau * log_ratio;
    out_dloss[idx] = grad * inv_loss_scale;

    if (stats != nullptr) {
        atomicAdd(stats + 0, -pg_term + kl_term);
        atomicAdd(stats + 1, ratio - log_ratio - 1.0f);
        atomicAdd(stats + 2, is_masked ? 1.0f : 0.0f);
        atomicAdd(stats + 3, keep ? 1.0f : 0.0f);
        atomicAdd(stats + 4, 1.0f);
    }
}

}  // namespace

void build_grpo_custom_dloss_and_stats(
    const float* losses,
    const float* inference_logprobs,
    const float* advantages,
    const std::uint8_t* loss_mask,
    float* out_dloss,
    float* stats,
    int B,
    int T,
    float kl_tau,
    float adv_tau,
    float ipo_mask_low,
    float ipo_mask_high,
    float inv_loss_scale,
    cudaStream_t stream) {
    if (!losses || !inference_logprobs || !advantages || !loss_mask || !out_dloss) {
        throw std::invalid_argument("build_grpo_custom_dloss_and_stats: null pointer");
    }
    if (B <= 0 || T <= 0) {
        throw std::invalid_argument("build_grpo_custom_dloss_and_stats: invalid shape");
    }

    constexpr int kBlock = 256;
    const int n = B * T;
    const int grid = static_cast<int>(div_ceil(static_cast<std::size_t>(n), static_cast<std::size_t>(kBlock)));
    build_grpo_custom_dloss_kernel<<<grid, kBlock, 0, stream>>>(
        losses,
        inference_logprobs,
        advantages,
        loss_mask,
        out_dloss,
        stats,
        B,
        T,
        kl_tau,
        adv_tau,
        ipo_mask_low,
        ipo_mask_high,
        inv_loss_scale);
    CUDA_CHECK(cudaGetLastError());
}
