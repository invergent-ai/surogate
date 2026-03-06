// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests comparing Flash AdamW 8-bit vs BnB-style AdamW 8-bit
// and benchmarks for the flash implementation.

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "kernels/kernels.h"
#include "utilities/utils.h"
#include "../utilities/test_config.h"
#include "../utilities/test_utils.h"

using namespace testing_utils;

namespace {

// CPU reference implementation of AdamW (FP32 golden reference)
struct AdamWState {
    std::vector<float> m;
    std::vector<float> v;
    int step = 0;
};

void adamw_cpu(
    std::vector<float>& params,
    const std::vector<float>& grads,
    AdamWState& state,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float grad_scale = 1.0f
) {
    const size_t n = params.size();
    if (state.m.empty()) {
        state.m.resize(n, 0.0f);
        state.v.resize(n, 0.0f);
    }
    state.step++;
    const float bc1 = 1.0f - std::pow(beta1, state.step);
    const float bc2 = 1.0f - std::pow(beta2, state.step);

    for (size_t i = 0; i < n; ++i) {
        float g = grads[i] * grad_scale;
        state.m[i] = beta1 * state.m[i] + (1.0f - beta1) * g;
        state.v[i] = beta2 * state.v[i] + (1.0f - beta2) * g * g;
        float m_hat = state.m[i] / bc1;
        float v_hat = state.v[i] / bc2;
        params[i] = params[i] - lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * params[i]);
    }
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float d = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) d = std::max(d, std::fabs(a[i] - b[i]));
    return d;
}

float relative_error(const std::vector<float>& actual, const std::vector<float>& expected) {
    float sq_diff = 0.0f, sq_exp = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        float d = actual[i] - expected[i];
        sq_diff += d * d;
        sq_exp += expected[i] * expected[i];
    }
    return std::sqrt(sq_diff) / (std::sqrt(sq_exp) + 1e-8f);
}

std::vector<float> bf16_to_float(const std::vector<nv_bfloat16>& v) {
    std::vector<float> r(v.size());
    for (size_t i = 0; i < v.size(); ++i) r[i] = __bfloat162float(v[i]);
    return r;
}

class CudaTimer {
public:
    CudaTimer() { cudaEventCreate(&s_); cudaEventCreate(&e_); }
    ~CudaTimer() { cudaEventDestroy(s_); cudaEventDestroy(e_); }
    void start(cudaStream_t st = 0) { cudaEventRecord(s_, st); }
    float stop(cudaStream_t st = 0) {
        cudaEventRecord(e_, st); cudaEventSynchronize(e_);
        float ms; cudaEventElapsedTime(&ms, s_, e_); return ms;
    }
private:
    cudaEvent_t s_, e_;
};

} // anonymous namespace

// ============================================================================
// FLASH ADAMW 8-BIT CORRECTNESS TESTS
// ============================================================================

TEST_CASE("Flash AdamW 8-bit initialization", "[optimizers][flash_adamw8bit]") {
    const size_t n = 4096 * 16;  // 64K parameters
    const size_t num_groups = flash_adamw8bit_num_scales(n);

    thrust::device_vector<signed char> d_state1(n);
    thrust::device_vector<unsigned char> d_state2(n);
    thrust::device_vector<half> d_scales1(num_groups);
    thrust::device_vector<half> d_scales2(num_groups);

    init_flash_adamw8bit_state(
        thrust::raw_pointer_cast(d_state1.data()),
        thrust::raw_pointer_cast(d_state2.data()),
        thrust::raw_pointer_cast(d_scales1.data()),
        thrust::raw_pointer_cast(d_scales2.data()),
        n, 0
    );
    cudaDeviceSynchronize();

    std::vector<signed char> h_state1 = from_device(d_state1);
    std::vector<unsigned char> h_state2 = from_device(d_state2);

    for (size_t i = 0; i < n; ++i) {
        REQUIRE(h_state1[i] == 0);
        REQUIRE(h_state2[i] == 0);
    }
}

TEST_CASE("Flash AdamW 8-bit FP32 correctness", "[optimizers][flash_adamw8bit]") {
    const size_t n = 4096 * 16;  // 64K parameters
    const size_t num_groups = flash_adamw8bit_num_scales(n);
    const int num_steps = 10;
    const float lr = 1e-3f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float weight_decay = 0.01f;

    std::vector<float> h_params = uniform_host(n, -1.0f, 1.0f, 42);
    std::vector<float> h_params_cpu = h_params;

    thrust::device_vector<float> d_params = to_device(h_params);
    thrust::device_vector<signed char> d_state1(n);
    thrust::device_vector<unsigned char> d_state2(n);
    thrust::device_vector<half> d_scales1(num_groups);
    thrust::device_vector<half> d_scales2(num_groups);

    init_flash_adamw8bit_state(
        thrust::raw_pointer_cast(d_state1.data()),
        thrust::raw_pointer_cast(d_state2.data()),
        thrust::raw_pointer_cast(d_scales1.data()),
        thrust::raw_pointer_cast(d_scales2.data()),
        n, 0
    );

    AdamWState cpu_state;

    for (int step = 1; step <= num_steps; ++step) {
        std::vector<float> h_grads = uniform_host(n, -0.1f, 0.1f, 42 + step);
        thrust::device_vector<float> d_grads = to_device(h_grads);

        flash_adamw_update_8bit(
            thrust::raw_pointer_cast(d_params.data()),
            thrust::raw_pointer_cast(d_grads.data()),
            thrust::raw_pointer_cast(d_state1.data()),
            thrust::raw_pointer_cast(d_state2.data()),
            thrust::raw_pointer_cast(d_scales1.data()),
            thrust::raw_pointer_cast(d_scales2.data()),
            n, lr, beta1, beta2, step, eps, weight_decay,
            nullptr, nullptr, nullptr, 0
        );

        adamw_cpu(h_params_cpu, h_grads, cpu_state, lr, beta1, beta2, eps, weight_decay);
        cudaDeviceSynchronize();
    }

    std::vector<float> h_params_gpu = from_device(d_params);
    float max_diff = max_abs_diff(h_params_gpu, h_params_cpu);
    float rel_err = relative_error(h_params_gpu, h_params_cpu);

    INFO("Flash AdamW 8-bit FP32 - Max abs diff: " << max_diff);
    INFO("Flash AdamW 8-bit FP32 - Relative error: " << rel_err);
    REQUIRE(max_diff < 0.1f);
    REQUIRE(rel_err < 0.05f);
}

TEST_CASE("Flash AdamW 8-bit BF16 correctness", "[optimizers][flash_adamw8bit]") {
    const size_t n = 4096 * 16;
    const size_t num_groups = flash_adamw8bit_num_scales(n);
    const int num_steps = 10;
    const float lr = 1e-3f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float weight_decay = 0.01f;

    std::vector<float> h_params_f = uniform_host(n, -1.0f, 1.0f, 42);
    std::vector<float> h_params_cpu = h_params_f;
    std::vector<nv_bfloat16> h_params_bf16 = to_bf16(h_params_f);

    thrust::device_vector<nv_bfloat16> d_params = to_device(h_params_bf16);
    thrust::device_vector<signed char> d_state1(n);
    thrust::device_vector<unsigned char> d_state2(n);
    thrust::device_vector<half> d_scales1(num_groups);
    thrust::device_vector<half> d_scales2(num_groups);

    init_flash_adamw8bit_state(
        thrust::raw_pointer_cast(d_state1.data()),
        thrust::raw_pointer_cast(d_state2.data()),
        thrust::raw_pointer_cast(d_scales1.data()),
        thrust::raw_pointer_cast(d_scales2.data()),
        n, 0
    );

    AdamWState cpu_state;

    for (int step = 1; step <= num_steps; ++step) {
        std::vector<float> h_grads_f = uniform_host(n, -0.1f, 0.1f, 42 + step);
        std::vector<nv_bfloat16> h_grads_bf16 = to_bf16(h_grads_f);
        thrust::device_vector<nv_bfloat16> d_grads = to_device(h_grads_bf16);

        flash_adamw_update_8bit(
            thrust::raw_pointer_cast(d_params.data()),
            thrust::raw_pointer_cast(d_grads.data()),
            thrust::raw_pointer_cast(d_state1.data()),
            thrust::raw_pointer_cast(d_state2.data()),
            thrust::raw_pointer_cast(d_scales1.data()),
            thrust::raw_pointer_cast(d_scales2.data()),
            n, lr, beta1, beta2, step, eps, weight_decay,
            nullptr, nullptr, nullptr, 0
        );

        std::vector<float> h_grads_rounded = round_bf16(h_grads_f);
        adamw_cpu(h_params_cpu, h_grads_rounded, cpu_state, lr, beta1, beta2, eps, weight_decay);
        h_params_cpu = round_bf16(h_params_cpu);
        cudaDeviceSynchronize();
    }

    std::vector<nv_bfloat16> h_gpu_bf16 = from_device(d_params);
    std::vector<float> h_params_gpu = bf16_to_float(h_gpu_bf16);
    float max_diff = max_abs_diff(h_params_gpu, h_params_cpu);
    float rel_err = relative_error(h_params_gpu, h_params_cpu);

    INFO("Flash AdamW 8-bit BF16 - Max abs diff: " << max_diff);
    INFO("Flash AdamW 8-bit BF16 - Relative error: " << rel_err);
    REQUIRE(max_diff < 0.15f);
    REQUIRE(rel_err < 0.1f);
}

// ============================================================================
// HEAD-TO-HEAD COMPARISON: Flash vs BnB-style 8-bit
// ============================================================================

TEST_CASE("Flash vs BnB AdamW 8-bit accuracy comparison - FP32", "[optimizers][flash_adamw8bit][comparison]") {
    const size_t n = 4096 * 16;  // 64K parameters
    const size_t bnb_num_blocks = (n + optimizers::ADAMW8BIT_BLOCK_SIZE - 1) / optimizers::ADAMW8BIT_BLOCK_SIZE;
    const size_t flash_num_groups = flash_adamw8bit_num_scales(n);
    const int num_steps = 50;  // More steps to see divergence
    const float lr = 1e-3f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float weight_decay = 0.01f;

    // Same initial params for all three
    std::vector<float> h_params = uniform_host(n, -1.0f, 1.0f, 42);
    std::vector<float> h_params_cpu = h_params;
    std::vector<float> h_params_bnb = h_params;
    std::vector<float> h_params_flash = h_params;

    // --- BnB setup ---
    std::vector<float> h_q1(256), h_q2(256);
    create_adamw8bit_quantiles1(h_q1.data());
    create_adamw8bit_quantiles2(h_q2.data());

    thrust::device_vector<float> d_params_bnb = to_device(h_params_bnb);
    thrust::device_vector<unsigned char> d_bnb_s1(n), d_bnb_s2(n);
    thrust::device_vector<float> d_bnb_a1(bnb_num_blocks), d_bnb_a2(bnb_num_blocks);
    thrust::device_vector<float> d_q1 = to_device(h_q1), d_q2 = to_device(h_q2);

    init_adamw8bit_state(
        thrust::raw_pointer_cast(d_bnb_s1.data()),
        thrust::raw_pointer_cast(d_bnb_s2.data()),
        thrust::raw_pointer_cast(d_bnb_a1.data()),
        thrust::raw_pointer_cast(d_bnb_a2.data()),
        n, 0
    );

    // --- Flash setup ---
    thrust::device_vector<float> d_params_flash = to_device(h_params_flash);
    thrust::device_vector<signed char> d_flash_s1(n);
    thrust::device_vector<unsigned char> d_flash_s2(n);
    thrust::device_vector<half> d_flash_sc1(flash_num_groups), d_flash_sc2(flash_num_groups);

    init_flash_adamw8bit_state(
        thrust::raw_pointer_cast(d_flash_s1.data()),
        thrust::raw_pointer_cast(d_flash_s2.data()),
        thrust::raw_pointer_cast(d_flash_sc1.data()),
        thrust::raw_pointer_cast(d_flash_sc2.data()),
        n, 0
    );

    AdamWState cpu_state;

    // Track errors over steps
    std::vector<float> bnb_errors, flash_errors;

    for (int step = 1; step <= num_steps; ++step) {
        std::vector<float> h_grads = uniform_host(n, -0.1f, 0.1f, 42 + step);
        thrust::device_vector<float> d_grads = to_device(h_grads);

        // BnB 8-bit update
        adamw_update_8bit(
            thrust::raw_pointer_cast(d_params_bnb.data()),
            thrust::raw_pointer_cast(d_grads.data()),
            thrust::raw_pointer_cast(d_bnb_s1.data()),
            thrust::raw_pointer_cast(d_bnb_s2.data()),
            n, lr, beta1, beta2, step, eps, weight_decay, nullptr,
            thrust::raw_pointer_cast(d_q1.data()),
            thrust::raw_pointer_cast(d_q2.data()),
            thrust::raw_pointer_cast(d_bnb_a1.data()),
            thrust::raw_pointer_cast(d_bnb_a2.data()),
            nullptr, nullptr, 0
        );

        // Flash 8-bit update
        flash_adamw_update_8bit(
            thrust::raw_pointer_cast(d_params_flash.data()),
            thrust::raw_pointer_cast(d_grads.data()),
            thrust::raw_pointer_cast(d_flash_s1.data()),
            thrust::raw_pointer_cast(d_flash_s2.data()),
            thrust::raw_pointer_cast(d_flash_sc1.data()),
            thrust::raw_pointer_cast(d_flash_sc2.data()),
            n, lr, beta1, beta2, step, eps, weight_decay,
            nullptr, nullptr, nullptr, 0
        );

        // CPU reference
        adamw_cpu(h_params_cpu, h_grads, cpu_state, lr, beta1, beta2, eps, weight_decay);

        cudaDeviceSynchronize();

        // Measure errors at selected steps
        if (step == 1 || step == 10 || step == 25 || step == 50) {
            std::vector<float> bnb_gpu = from_device(d_params_bnb);
            std::vector<float> flash_gpu = from_device(d_params_flash);

            float bnb_rel = relative_error(bnb_gpu, h_params_cpu);
            float flash_rel = relative_error(flash_gpu, h_params_cpu);

            bnb_errors.push_back(bnb_rel);
            flash_errors.push_back(flash_rel);

            printf("Step %3d | BnB rel_err: %.6f | Flash rel_err: %.6f | "
                   "BnB max_diff: %.6f | Flash max_diff: %.6f\n",
                   step,
                   bnb_rel, flash_rel,
                   max_abs_diff(bnb_gpu, h_params_cpu),
                   max_abs_diff(flash_gpu, h_params_cpu));
        }
    }

    // Both should be reasonable vs reference
    std::vector<float> bnb_final = from_device(d_params_bnb);
    std::vector<float> flash_final = from_device(d_params_flash);

    float bnb_final_rel = relative_error(bnb_final, h_params_cpu);
    float flash_final_rel = relative_error(flash_final, h_params_cpu);

    printf("\n=== Final Comparison (50 steps, FP32 params, 64K elements) ===\n");
    printf("BnB   8-bit relative error: %.6f\n", bnb_final_rel);
    printf("Flash 8-bit relative error: %.6f\n", flash_final_rel);
    printf("Flash / BnB error ratio:    %.3fx\n", flash_final_rel / (bnb_final_rel + 1e-10f));

    // Memory comparison
    size_t bnb_state_bytes = 2 * n * sizeof(unsigned char) + 2 * bnb_num_blocks * sizeof(float);
    size_t flash_state_bytes = n * sizeof(signed char) + n * sizeof(unsigned char) + 2 * flash_num_groups * sizeof(half);
    printf("BnB   state memory: %zu bytes (%.2f bytes/param)\n",
           bnb_state_bytes, (double)bnb_state_bytes / n);
    printf("Flash state memory: %zu bytes (%.2f bytes/param)\n",
           flash_state_bytes, (double)flash_state_bytes / n);

    // Both should be within tolerance
    REQUIRE(bnb_final_rel < 0.1f);
    REQUIRE(flash_final_rel < 0.1f);
}

TEST_CASE("Flash vs BnB AdamW 8-bit accuracy comparison - BF16", "[optimizers][flash_adamw8bit][comparison]") {
    const size_t n = 4096 * 16;
    const size_t bnb_num_blocks = (n + optimizers::ADAMW8BIT_BLOCK_SIZE - 1) / optimizers::ADAMW8BIT_BLOCK_SIZE;
    const size_t flash_num_groups = flash_adamw8bit_num_scales(n);
    const int num_steps = 50;
    const float lr = 1e-3f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float weight_decay = 0.01f;

    std::vector<float> h_params_f = uniform_host(n, -1.0f, 1.0f, 42);
    std::vector<float> h_params_cpu = h_params_f;
    std::vector<nv_bfloat16> h_bf16 = to_bf16(h_params_f);

    // BnB quantiles
    std::vector<float> h_q1(256), h_q2(256);
    create_adamw8bit_quantiles1(h_q1.data());
    create_adamw8bit_quantiles2(h_q2.data());

    // BnB state
    thrust::device_vector<nv_bfloat16> d_params_bnb = to_device(h_bf16);
    thrust::device_vector<unsigned char> d_bnb_s1(n), d_bnb_s2(n);
    thrust::device_vector<float> d_bnb_a1(bnb_num_blocks), d_bnb_a2(bnb_num_blocks);
    thrust::device_vector<float> d_q1 = to_device(h_q1), d_q2 = to_device(h_q2);
    init_adamw8bit_state(
        thrust::raw_pointer_cast(d_bnb_s1.data()),
        thrust::raw_pointer_cast(d_bnb_s2.data()),
        thrust::raw_pointer_cast(d_bnb_a1.data()),
        thrust::raw_pointer_cast(d_bnb_a2.data()),
        n, 0
    );

    // Flash state
    thrust::device_vector<nv_bfloat16> d_params_flash = to_device(h_bf16);
    thrust::device_vector<signed char> d_flash_s1(n);
    thrust::device_vector<unsigned char> d_flash_s2(n);
    thrust::device_vector<half> d_flash_sc1(flash_num_groups), d_flash_sc2(flash_num_groups);
    init_flash_adamw8bit_state(
        thrust::raw_pointer_cast(d_flash_s1.data()),
        thrust::raw_pointer_cast(d_flash_s2.data()),
        thrust::raw_pointer_cast(d_flash_sc1.data()),
        thrust::raw_pointer_cast(d_flash_sc2.data()),
        n, 0
    );

    AdamWState cpu_state;

    for (int step = 1; step <= num_steps; ++step) {
        std::vector<float> h_grads_f = uniform_host(n, -0.1f, 0.1f, 42 + step);
        std::vector<nv_bfloat16> h_grads_bf16 = to_bf16(h_grads_f);
        thrust::device_vector<nv_bfloat16> d_grads = to_device(h_grads_bf16);

        adamw_update_8bit(
            thrust::raw_pointer_cast(d_params_bnb.data()),
            thrust::raw_pointer_cast(d_grads.data()),
            thrust::raw_pointer_cast(d_bnb_s1.data()),
            thrust::raw_pointer_cast(d_bnb_s2.data()),
            n, lr, beta1, beta2, step, eps, weight_decay, nullptr,
            thrust::raw_pointer_cast(d_q1.data()),
            thrust::raw_pointer_cast(d_q2.data()),
            thrust::raw_pointer_cast(d_bnb_a1.data()),
            thrust::raw_pointer_cast(d_bnb_a2.data()),
            nullptr, nullptr, 0
        );

        flash_adamw_update_8bit(
            thrust::raw_pointer_cast(d_params_flash.data()),
            thrust::raw_pointer_cast(d_grads.data()),
            thrust::raw_pointer_cast(d_flash_s1.data()),
            thrust::raw_pointer_cast(d_flash_s2.data()),
            thrust::raw_pointer_cast(d_flash_sc1.data()),
            thrust::raw_pointer_cast(d_flash_sc2.data()),
            n, lr, beta1, beta2, step, eps, weight_decay,
            nullptr, nullptr, nullptr, 0
        );

        std::vector<float> h_grads_rounded = round_bf16(h_grads_f);
        adamw_cpu(h_params_cpu, h_grads_rounded, cpu_state, lr, beta1, beta2, eps, weight_decay);
        h_params_cpu = round_bf16(h_params_cpu);
        cudaDeviceSynchronize();
    }

    std::vector<float> bnb_gpu = bf16_to_float(from_device(d_params_bnb));
    std::vector<float> flash_gpu = bf16_to_float(from_device(d_params_flash));

    float bnb_rel = relative_error(bnb_gpu, h_params_cpu);
    float flash_rel = relative_error(flash_gpu, h_params_cpu);

    printf("\n=== BF16 Comparison (50 steps, BF16 params, 64K elements) ===\n");
    printf("BnB   8-bit relative error: %.6f\n", bnb_rel);
    printf("Flash 8-bit relative error: %.6f\n", flash_rel);
    printf("Flash / BnB error ratio:    %.3fx\n", flash_rel / (bnb_rel + 1e-10f));

    REQUIRE(bnb_rel < 0.15f);
    REQUIRE(flash_rel < 0.15f);
}

// ============================================================================
// BENCHMARKS
// ============================================================================

TEST_CASE("Flash AdamW 8-bit benchmark - FP32 params", "[optimizers][flash_adamw8bit][benchmark]") {
    const size_t n = 50 * 1024 * 1024;  // 50M parameters
    const size_t num_groups = flash_adamw8bit_num_scales(n);
    const int warmup = 5, bench = 20;
    const float lr = 1e-3f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 0.01f;

    std::vector<float> h_data = uniform_host(n, -1.0f, 1.0f, 42);
    thrust::device_vector<float> d_params = to_device(h_data);
    thrust::device_vector<float> d_grads = to_device(h_data);
    thrust::device_vector<signed char> d_s1(n);
    thrust::device_vector<unsigned char> d_s2(n);
    thrust::device_vector<half> d_sc1(num_groups), d_sc2(num_groups);

    init_flash_adamw8bit_state(
        thrust::raw_pointer_cast(d_s1.data()),
        thrust::raw_pointer_cast(d_s2.data()),
        thrust::raw_pointer_cast(d_sc1.data()),
        thrust::raw_pointer_cast(d_sc2.data()),
        n, 0
    );

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    CudaTimer timer;

    for (int i = 0; i < warmup; ++i) {
        flash_adamw_update_8bit(
            thrust::raw_pointer_cast(d_params.data()),
            thrust::raw_pointer_cast(d_grads.data()),
            thrust::raw_pointer_cast(d_s1.data()),
            thrust::raw_pointer_cast(d_s2.data()),
            thrust::raw_pointer_cast(d_sc1.data()),
            thrust::raw_pointer_cast(d_sc2.data()),
            n, lr, beta1, beta2, i + 1, eps, wd,
            nullptr, nullptr, nullptr, stream
        );
    }
    cudaStreamSynchronize(stream);

    timer.start(stream);
    for (int i = 0; i < bench; ++i) {
        flash_adamw_update_8bit(
            thrust::raw_pointer_cast(d_params.data()),
            thrust::raw_pointer_cast(d_grads.data()),
            thrust::raw_pointer_cast(d_s1.data()),
            thrust::raw_pointer_cast(d_s2.data()),
            thrust::raw_pointer_cast(d_sc1.data()),
            thrust::raw_pointer_cast(d_sc2.data()),
            n, lr, beta1, beta2, warmup + i + 1, eps, wd,
            nullptr, nullptr, nullptr, stream
        );
    }
    float total_ms = timer.stop(stream);
    float avg_ms = total_ms / bench;

    // Memory: params r/w (2*4B), grads r (4B), states r/w (2*1B+2*1B), scales r/w (2*2B per group)
    double bytes = (double)n * (2*4 + 4 + 4*1) + (double)num_groups * (4*2);
    double bw = (bytes / (avg_ms / 1000.0)) / 1e9;

    size_t state_bytes = n * sizeof(signed char) + n * sizeof(unsigned char) + 2 * num_groups * sizeof(half);
    double savings = (1.0 - (double)state_bytes / (2.0 * n * sizeof(float))) * 100.0;

    printf("\n=== Flash AdamW 8-bit FP32 Benchmark ===\n");
    printf("Parameters: %.1fM\n", n / 1e6);
    printf("Avg time: %.3f ms\n", avg_ms);
    printf("Throughput: %.2f M params/ms\n", (n / 1e6) / avg_ms);
    printf("Effective bandwidth: %.1f GB/s\n", bw);
    printf("State memory: %.1f MB (%.2f bytes/param)\n",
           state_bytes / 1e6, (double)state_bytes / n);
    printf("Memory savings vs FP32 states: %.1f%%\n", savings);

    cudaStreamDestroy(stream);
    REQUIRE(avg_ms > 0);
}

TEST_CASE("Flash vs BnB AdamW 8-bit benchmark", "[optimizers][flash_adamw8bit][benchmark]") {
    const size_t n = 50 * 1024 * 1024;  // 50M parameters
    const size_t bnb_num_blocks = (n + optimizers::ADAMW8BIT_BLOCK_SIZE - 1) / optimizers::ADAMW8BIT_BLOCK_SIZE;
    const size_t flash_num_groups = flash_adamw8bit_num_scales(n);
    const int warmup = 5, bench = 20;
    const float lr = 1e-3f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 0.01f;

    std::vector<float> h_data = uniform_host(n, -1.0f, 1.0f, 42);

    // BnB setup
    std::vector<float> h_q1(256), h_q2(256);
    create_adamw8bit_quantiles1(h_q1.data());
    create_adamw8bit_quantiles2(h_q2.data());
    thrust::device_vector<float> d_q1 = to_device(h_q1), d_q2 = to_device(h_q2);

    thrust::device_vector<float> d_params_bnb = to_device(h_data);
    thrust::device_vector<float> d_grads = to_device(h_data);
    thrust::device_vector<unsigned char> d_bnb_s1(n), d_bnb_s2(n);
    thrust::device_vector<float> d_bnb_a1(bnb_num_blocks), d_bnb_a2(bnb_num_blocks);
    init_adamw8bit_state(
        thrust::raw_pointer_cast(d_bnb_s1.data()),
        thrust::raw_pointer_cast(d_bnb_s2.data()),
        thrust::raw_pointer_cast(d_bnb_a1.data()),
        thrust::raw_pointer_cast(d_bnb_a2.data()),
        n, 0
    );

    // Flash setup
    thrust::device_vector<float> d_params_flash = to_device(h_data);
    thrust::device_vector<signed char> d_flash_s1(n);
    thrust::device_vector<unsigned char> d_flash_s2(n);
    thrust::device_vector<half> d_flash_sc1(flash_num_groups), d_flash_sc2(flash_num_groups);
    init_flash_adamw8bit_state(
        thrust::raw_pointer_cast(d_flash_s1.data()),
        thrust::raw_pointer_cast(d_flash_s2.data()),
        thrust::raw_pointer_cast(d_flash_sc1.data()),
        thrust::raw_pointer_cast(d_flash_sc2.data()),
        n, 0
    );

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    CudaTimer timer;

    // --- Benchmark BnB ---
    for (int i = 0; i < warmup; ++i) {
        adamw_update_8bit(
            thrust::raw_pointer_cast(d_params_bnb.data()),
            thrust::raw_pointer_cast(d_grads.data()),
            thrust::raw_pointer_cast(d_bnb_s1.data()),
            thrust::raw_pointer_cast(d_bnb_s2.data()),
            n, lr, beta1, beta2, i + 1, eps, wd, nullptr,
            thrust::raw_pointer_cast(d_q1.data()),
            thrust::raw_pointer_cast(d_q2.data()),
            thrust::raw_pointer_cast(d_bnb_a1.data()),
            thrust::raw_pointer_cast(d_bnb_a2.data()),
            nullptr, nullptr, stream
        );
    }
    cudaStreamSynchronize(stream);

    timer.start(stream);
    for (int i = 0; i < bench; ++i) {
        adamw_update_8bit(
            thrust::raw_pointer_cast(d_params_bnb.data()),
            thrust::raw_pointer_cast(d_grads.data()),
            thrust::raw_pointer_cast(d_bnb_s1.data()),
            thrust::raw_pointer_cast(d_bnb_s2.data()),
            n, lr, beta1, beta2, warmup + i + 1, eps, wd, nullptr,
            thrust::raw_pointer_cast(d_q1.data()),
            thrust::raw_pointer_cast(d_q2.data()),
            thrust::raw_pointer_cast(d_bnb_a1.data()),
            thrust::raw_pointer_cast(d_bnb_a2.data()),
            nullptr, nullptr, stream
        );
    }
    float bnb_ms = timer.stop(stream) / bench;

    // --- Benchmark Flash ---
    for (int i = 0; i < warmup; ++i) {
        flash_adamw_update_8bit(
            thrust::raw_pointer_cast(d_params_flash.data()),
            thrust::raw_pointer_cast(d_grads.data()),
            thrust::raw_pointer_cast(d_flash_s1.data()),
            thrust::raw_pointer_cast(d_flash_s2.data()),
            thrust::raw_pointer_cast(d_flash_sc1.data()),
            thrust::raw_pointer_cast(d_flash_sc2.data()),
            n, lr, beta1, beta2, i + 1, eps, wd,
            nullptr, nullptr, nullptr, stream
        );
    }
    cudaStreamSynchronize(stream);

    timer.start(stream);
    for (int i = 0; i < bench; ++i) {
        flash_adamw_update_8bit(
            thrust::raw_pointer_cast(d_params_flash.data()),
            thrust::raw_pointer_cast(d_grads.data()),
            thrust::raw_pointer_cast(d_flash_s1.data()),
            thrust::raw_pointer_cast(d_flash_s2.data()),
            thrust::raw_pointer_cast(d_flash_sc1.data()),
            thrust::raw_pointer_cast(d_flash_sc2.data()),
            n, lr, beta1, beta2, warmup + i + 1, eps, wd,
            nullptr, nullptr, nullptr, stream
        );
    }
    float flash_ms = timer.stop(stream) / bench;

    printf("\n=== Head-to-Head Benchmark (50M FP32 params) ===\n");
    printf("BnB   avg: %.3f ms (%.2f M params/ms)\n", bnb_ms, (n / 1e6) / bnb_ms);
    printf("Flash avg: %.3f ms (%.2f M params/ms)\n", flash_ms, (n / 1e6) / flash_ms);
    printf("Flash / BnB speedup: %.2fx\n", bnb_ms / flash_ms);

    cudaStreamDestroy(stream);
    REQUIRE(bnb_ms > 0);
    REQUIRE(flash_ms > 0);
}
