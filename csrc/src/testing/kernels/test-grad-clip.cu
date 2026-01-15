// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Tests for global gradient clipping scale computation.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include "kernels/kernels.h"
#include "utilities/utils.h"

namespace {

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

} // namespace

TEST_CASE("global_norm_sqrt: grad_clip=0 disables clipping (scale=1)", "[kernels][grad-clip][gpu]") {
    if (!cuda_available()) {
        SKIP("CUDA not available");
    }

    cudaDeviceProp dp{};
    CUDA_CHECK(cudaGetDeviceProperties(&dp, 0));

    float* device_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&device_buf, 2 * sizeof(float)));

    float* host_norm = nullptr;
    CUDA_CHECK(cudaMallocHost(&host_norm, sizeof(float)));

    float n_squared = 4.0f; // norm = 2
    CUDA_CHECK(cudaMemcpy(device_buf, &n_squared, sizeof(float), cudaMemcpyHostToDevice));

    global_norm_sqrt(device_buf, host_norm, /*grad_clip=*/0.0f, /*valid_token_count=*/nullptr, /*total_tokens=*/0.0f, dp, /*stream=*/nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    float scale = 0.0f;
    CUDA_CHECK(cudaMemcpy(&scale, device_buf + 1, sizeof(float), cudaMemcpyDeviceToHost));

    REQUIRE(*host_norm == Catch::Approx(2.0f));
    REQUIRE(scale == Catch::Approx(1.0f));

    CUDA_CHECK(cudaFreeHost(host_norm));
    CUDA_CHECK(cudaFree(device_buf));
}

TEST_CASE("global_norm_sqrt: clipping scale uses grad_clip / norm", "[kernels][grad-clip][gpu]") {
    if (!cuda_available()) {
        SKIP("CUDA not available");
    }

    cudaDeviceProp dp{};
    CUDA_CHECK(cudaGetDeviceProperties(&dp, 0));

    float* device_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&device_buf, 2 * sizeof(float)));

    float* host_norm = nullptr;
    CUDA_CHECK(cudaMallocHost(&host_norm, sizeof(float)));

    {
        float n_squared = 4.0f; // norm = 2
        CUDA_CHECK(cudaMemcpy(device_buf, &n_squared, sizeof(float), cudaMemcpyHostToDevice));
        global_norm_sqrt(device_buf, host_norm, /*grad_clip=*/1.0f, /*valid_token_count=*/nullptr, /*total_tokens=*/0.0f, dp, /*stream=*/nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        float scale = 0.0f;
        CUDA_CHECK(cudaMemcpy(&scale, device_buf + 1, sizeof(float), cudaMemcpyDeviceToHost));

        REQUIRE(*host_norm == Catch::Approx(2.0f));
        REQUIRE(scale == Catch::Approx(0.5f));
    }

    {
        float n_squared = 0.25f; // norm = 0.5
        CUDA_CHECK(cudaMemcpy(device_buf, &n_squared, sizeof(float), cudaMemcpyHostToDevice));
        global_norm_sqrt(device_buf, host_norm, /*grad_clip=*/1.0f, /*valid_token_count=*/nullptr, /*total_tokens=*/0.0f, dp, /*stream=*/nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        float scale = 0.0f;
        CUDA_CHECK(cudaMemcpy(&scale, device_buf + 1, sizeof(float), cudaMemcpyDeviceToHost));

        REQUIRE(*host_norm == Catch::Approx(0.5f));
        REQUIRE(scale == Catch::Approx(1.0f));
    }

    {
        float n_squared = 4.0f; // norm = 2
        CUDA_CHECK(cudaMemcpy(device_buf, &n_squared, sizeof(float), cudaMemcpyHostToDevice));
        global_norm_sqrt(device_buf, host_norm, /*grad_clip=*/-1.0f, /*valid_token_count=*/nullptr, /*total_tokens=*/0.0f, dp, /*stream=*/nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        float scale = 0.0f;
        CUDA_CHECK(cudaMemcpy(&scale, device_buf + 1, sizeof(float), cudaMemcpyDeviceToHost));

        REQUIRE(*host_norm == Catch::Approx(2.0f));
        REQUIRE(scale == Catch::Approx(1.0f));
    }

    CUDA_CHECK(cudaFreeHost(host_norm));
    CUDA_CHECK(cudaFree(device_buf));
}

TEST_CASE("global_norm_sqrt: token scaling uses total_tokens / valid_tokens", "[kernels][grad-clip][gpu]") {
    if (!cuda_available()) {
        SKIP("CUDA not available");
    }

    cudaDeviceProp dp{};
    CUDA_CHECK(cudaGetDeviceProperties(&dp, 0));

    float* device_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&device_buf, 2 * sizeof(float)));

    int* device_valid = nullptr;
    CUDA_CHECK(cudaMalloc(&device_valid, sizeof(int)));

    float* host_norm = nullptr;
    CUDA_CHECK(cudaMallocHost(&host_norm, sizeof(float)));

    // raw norm = 0.5
    float n_squared = 0.25f;
    CUDA_CHECK(cudaMemcpy(device_buf, &n_squared, sizeof(float), cudaMemcpyHostToDevice));

    // token_scale = total_tokens / valid_tokens = 8 / 2 = 4
    int valid_tokens = 2;
    CUDA_CHECK(cudaMemcpy(device_valid, &valid_tokens, sizeof(int), cudaMemcpyHostToDevice));

    global_norm_sqrt(device_buf, host_norm, /*grad_clip=*/0.0f, /*valid_token_count=*/device_valid, /*total_tokens=*/8.0f, dp, /*stream=*/nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    float scale = 0.0f;
    CUDA_CHECK(cudaMemcpy(&scale, device_buf + 1, sizeof(float), cudaMemcpyDeviceToHost));

    REQUIRE(*host_norm == Catch::Approx(2.0f)); // scaled_norm = 0.5 * 4
    REQUIRE(scale == Catch::Approx(4.0f));      // token_scale (no clipping)

    CUDA_CHECK(cudaFreeHost(host_norm));
    CUDA_CHECK(cudaFree(device_valid));
    CUDA_CHECK(cudaFree(device_buf));
}

TEST_CASE("global_norm_sqrt: clipping with token scaling returns combined multiplier", "[kernels][grad-clip][gpu]") {
    if (!cuda_available()) {
        SKIP("CUDA not available");
    }

    cudaDeviceProp dp{};
    CUDA_CHECK(cudaGetDeviceProperties(&dp, 0));

    float* device_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&device_buf, 2 * sizeof(float)));

    int* device_valid = nullptr;
    CUDA_CHECK(cudaMalloc(&device_valid, sizeof(int)));

    float* host_norm = nullptr;
    CUDA_CHECK(cudaMallocHost(&host_norm, sizeof(float)));

    // raw norm = 0.5
    float n_squared = 0.25f;
    CUDA_CHECK(cudaMemcpy(device_buf, &n_squared, sizeof(float), cudaMemcpyHostToDevice));

    // token_scale = 8 / 2 = 4  => scaled_norm = 2.0 triggers clipping for grad_clip=1.0
    int valid_tokens = 2;
    CUDA_CHECK(cudaMemcpy(device_valid, &valid_tokens, sizeof(int), cudaMemcpyHostToDevice));

    global_norm_sqrt(device_buf, host_norm, /*grad_clip=*/1.0f, /*valid_token_count=*/device_valid, /*total_tokens=*/8.0f, dp, /*stream=*/nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    float scale = 0.0f;
    CUDA_CHECK(cudaMemcpy(&scale, device_buf + 1, sizeof(float), cudaMemcpyDeviceToHost));

    REQUIRE(*host_norm == Catch::Approx(2.0f)); // scaled_norm (for logging/comparison against grad_clip)
    REQUIRE(scale == Catch::Approx(2.0f));      // grad_clip / raw_norm = 1 / 0.5

    CUDA_CHECK(cudaFreeHost(host_norm));
    CUDA_CHECK(cudaFree(device_valid));
    CUDA_CHECK(cudaFree(device_buf));
}

