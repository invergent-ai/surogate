// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <array>
#include <cstdint>

#include "kernels/kernels.h"

TEST_CASE("native GRPO replay is CE-only", "[grpo][replay][cuda]") {
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count == 0) {
        SKIP("No CUDA device available");
    }
    REQUIRE(cudaSetDevice(0) == cudaSuccess);

    constexpr int kBT = 4;
    constexpr int kMetricCount = 19;
    const std::array<float, kBT> losses{2.0F, 1.0F, 3.0F, 0.0F};
    const std::array<float, kBT> inference_logprobs{50.0F, 50.0F, 50.0F, 50.0F};
    const std::array<float, kBT> advantages{50.0F, 50.0F, 50.0F, 50.0F};
    const std::array<float, kBT> teacher_logprobs{40.0F, 40.0F, 40.0F, 40.0F};
    const std::array<float, kBT> opd_reference_logprobs{30.0F, 30.0F, 30.0F, 30.0F};
    const std::array<float, kBT> hindsight_logprobs{60.0F, 60.0F, 60.0F, 60.0F};
    const std::array<std::uint8_t, kBT> loss_mask{0, 1, 1, 1};
    const std::array<std::uint8_t, kBT> hindsight_mask{0, 1, 1, 1};
    const std::array<std::uint8_t, kBT> replay_mask{0, 1, 1, 1};
    const std::array<float, kBT> replay_weights{1.0F, 2.0F, 0.5F, 3.0F};
    const std::array<std::int32_t, 1> sample_starts{0};
    const std::array<std::int32_t, 1> sample_ends{kBT};

    float* device_custom_dloss = nullptr;
    float* device_metrics = nullptr;
    float* device_losses = nullptr;
    float* device_inference_logprobs = nullptr;
    float* device_advantages = nullptr;
    float* device_teacher_logprobs = nullptr;
    float* device_opd_reference_logprobs = nullptr;
    float* device_hindsight_logprobs = nullptr;
    std::uint8_t* device_loss_mask = nullptr;
    std::uint8_t* device_hindsight_mask = nullptr;
    std::uint8_t* device_replay_mask = nullptr;
    float* device_replay_weights = nullptr;
    std::int32_t* device_sample_starts = nullptr;
    std::int32_t* device_sample_ends = nullptr;

#define ALLOCATE_AND_COPY(device, host)                          \
    REQUIRE(cudaMalloc(&(device), sizeof(host)) == cudaSuccess); \
    REQUIRE(cudaMemcpy((device), (host).data(), sizeof(host), cudaMemcpyHostToDevice) == cudaSuccess)

    REQUIRE(cudaMalloc(&device_custom_dloss, kBT * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&device_metrics, kMetricCount * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemset(device_metrics, 0, kMetricCount * sizeof(float)) == cudaSuccess);
    ALLOCATE_AND_COPY(device_losses, losses);
    ALLOCATE_AND_COPY(device_inference_logprobs, inference_logprobs);
    ALLOCATE_AND_COPY(device_advantages, advantages);
    ALLOCATE_AND_COPY(device_teacher_logprobs, teacher_logprobs);
    ALLOCATE_AND_COPY(device_opd_reference_logprobs, opd_reference_logprobs);
    ALLOCATE_AND_COPY(device_hindsight_logprobs, hindsight_logprobs);
    ALLOCATE_AND_COPY(device_loss_mask, loss_mask);
    ALLOCATE_AND_COPY(device_hindsight_mask, hindsight_mask);
    ALLOCATE_AND_COPY(device_replay_mask, replay_mask);
    ALLOCATE_AND_COPY(device_replay_weights, replay_weights);
    ALLOCATE_AND_COPY(device_sample_starts, sample_starts);
    ALLOCATE_AND_COPY(device_sample_ends, sample_ends);
#undef ALLOCATE_AND_COPY

    compute_grpo_custom_dloss(device_custom_dloss,
                              device_metrics,
                              device_losses,
                              device_inference_logprobs,
                              device_advantages,
                              device_loss_mask,
                              device_teacher_logprobs,
                              device_opd_reference_logprobs,
                              device_hindsight_logprobs,
                              device_hindsight_mask,
                              device_replay_mask,
                              device_replay_weights,
                              device_sample_starts,
                              device_sample_ends,
                              1,
                              kBT,
                              3.0F,
                              0.2F,
                              0.2F,
                              1.0F,
                              1.0F,
                              1.0F,
                              1.0F,
                              0.3F,
                              1.0F,
                              nullptr);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::array<float, kBT> custom_dloss{};
    std::array<float, kMetricCount> metrics{};
    REQUIRE(cudaMemcpy(custom_dloss.data(), device_custom_dloss, sizeof(custom_dloss), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(metrics.data(), device_metrics, sizeof(metrics), cudaMemcpyDeviceToHost) == cudaSuccess);

    REQUIRE(custom_dloss[0] == Catch::Approx(0.2F));
    REQUIRE(custom_dloss[1] == Catch::Approx(0.05F));
    REQUIRE(custom_dloss[2] == Catch::Approx(0.3F));
    REQUIRE(custom_dloss[3] == Catch::Approx(0.0F));
    REQUIRE(metrics[0] == Catch::Approx(1.35F));   // policy loss
    REQUIRE(metrics[1] == Catch::Approx(0.0F));    // mismatch KL
    REQUIRE(metrics[7] == Catch::Approx(0.0F));    // teacher KL
    REQUIRE(metrics[11] == Catch::Approx(0.0F));   // OPD tokens
    REQUIRE(metrics[12] == Catch::Approx(4.05F));  // weighted replay CE sum
    REQUIRE(metrics[13] == Catch::Approx(3.0F));   // replay tokens
    REQUIRE(metrics[14] == Catch::Approx(1.0F));   // sample count
    REQUIRE(metrics[15] == Catch::Approx(0.0F));   // kept IPO tokens
    REQUIRE(metrics[16] == Catch::Approx(3.0F));   // selected tokens
    REQUIRE(metrics[17] == Catch::Approx(5.5F));   // replay weight sum
    REQUIRE(metrics[18] == Catch::Approx(0.0F));   // behavior-likelihood policy samples

    REQUIRE(cudaFree(device_sample_ends) == cudaSuccess);
    REQUIRE(cudaFree(device_sample_starts) == cudaSuccess);
    REQUIRE(cudaFree(device_replay_mask) == cudaSuccess);
    REQUIRE(cudaFree(device_replay_weights) == cudaSuccess);
    REQUIRE(cudaFree(device_hindsight_mask) == cudaSuccess);
    REQUIRE(cudaFree(device_loss_mask) == cudaSuccess);
    REQUIRE(cudaFree(device_hindsight_logprobs) == cudaSuccess);
    REQUIRE(cudaFree(device_opd_reference_logprobs) == cudaSuccess);
    REQUIRE(cudaFree(device_teacher_logprobs) == cudaSuccess);
    REQUIRE(cudaFree(device_advantages) == cudaSuccess);
    REQUIRE(cudaFree(device_inference_logprobs) == cudaSuccess);
    REQUIRE(cudaFree(device_losses) == cudaSuccess);
    REQUIRE(cudaFree(device_metrics) == cudaSuccess);
    REQUIRE(cudaFree(device_custom_dloss) == cudaSuccess);
}
