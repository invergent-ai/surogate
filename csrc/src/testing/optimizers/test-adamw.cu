// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests and benchmarks for AdamW 8-bit optimizer

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <numeric>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

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

// ----------------------------------------------------------------------------
// CPU reference implementation of AdamW

struct AdamWState {
    std::vector<float> m;  // first moment
    std::vector<float> v;  // second moment
    int step = 0;
};

void adamw_cpu(
    std::vector<float>& params,
    const std::vector<float>& grads,
    AdamWState& state,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float grad_scale = 1.0f
) {
    const size_t n = params.size();
    if (state.m.empty()) {
        state.m.resize(n, 0.0f);
        state.v.resize(n, 0.0f);
    }

    state.step++;
    const float bias_correction1 = 1.0f - std::pow(beta1, state.step);
    const float bias_correction2 = 1.0f - std::pow(beta2, state.step);

    for (size_t i = 0; i < n; ++i) {
        float g = grads[i] * grad_scale;

        // Update biased first moment estimate
        state.m[i] = beta1 * state.m[i] + (1.0f - beta1) * g;

        // Update biased second raw moment estimate
        state.v[i] = beta2 * state.v[i] + (1.0f - beta2) * g * g;

        // Compute bias-corrected estimates
        float m_hat = state.m[i] / bias_correction1;
        float v_hat = state.v[i] / bias_correction2;

        // Update parameters (AdamW: weight decay applied to param, not gradient)
        params[i] = params[i] - lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * params[i]);
    }
}

// ----------------------------------------------------------------------------
// Helper functions

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

float relative_error(const std::vector<float>& actual, const std::vector<float>& expected) {
    float sum_sq_diff = 0.0f;
    float sum_sq_expected = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        float diff = actual[i] - expected[i];
        sum_sq_diff += diff * diff;
        sum_sq_expected += expected[i] * expected[i];
    }
    return std::sqrt(sum_sq_diff) / (std::sqrt(sum_sq_expected) + 1e-8f);
}

std::vector<float> bf16_to_float(const std::vector<nv_bfloat16>& v) {
    std::vector<float> result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = __bfloat162float(v[i]);
    }
    return result;
}

// ----------------------------------------------------------------------------
// Benchmark timer

class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_, stream);
    }

    float stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_, stream);
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

} // anonymous namespace

// ============================================================================
// CORRECTNESS TESTS
// ============================================================================

TEST_CASE("AdamW 8-bit initialization", "[optimizers][adamw8bit]") {
    const size_t n = 4096 * 16;  // 64K parameters
    const size_t num_blocks = (n + 2047) / 2048;

    // Allocate state tensors
    thrust::device_vector<unsigned char> d_state1(n);
    thrust::device_vector<unsigned char> d_state2(n);
    thrust::device_vector<float> d_absmax1(num_blocks);
    thrust::device_vector<float> d_absmax2(num_blocks);

    // Initialize
    init_adamw8bit_state(
        thrust::raw_pointer_cast(d_state1.data()),
        thrust::raw_pointer_cast(d_state2.data()),
        thrust::raw_pointer_cast(d_absmax1.data()),
        thrust::raw_pointer_cast(d_absmax2.data()),
        n, 0
    );
    cudaDeviceSynchronize();

    // Verify initialization
    std::vector<unsigned char> h_state1 = from_device(d_state1);
    std::vector<unsigned char> h_state2 = from_device(d_state2);
    std::vector<float> h_absmax1 = from_device(d_absmax1);
    std::vector<float> h_absmax2 = from_device(d_absmax2);

    // Check state1 is initialized to 128 (zero for signed quantization)
    for (size_t i = 0; i < n; ++i) {
        REQUIRE(h_state1[i] == 128);
    }

    // Check state2 is initialized to 0 (zero for unsigned quantization)
    for (size_t i = 0; i < n; ++i) {
        REQUIRE(h_state2[i] == 0);
    }

    // Check absmax values are small positive
    for (size_t i = 0; i < num_blocks; ++i) {
        REQUIRE(h_absmax1[i] > 0.0f);
        REQUIRE(h_absmax1[i] < 1e-5f);
        REQUIRE(h_absmax2[i] > 0.0f);
        REQUIRE(h_absmax2[i] < 1e-5f);
    }
}

TEST_CASE("AdamW 8-bit quantization maps", "[optimizers][adamw8bit]") {
    std::vector<float> h_quantiles1(256);
    std::vector<float> h_quantiles2(256);

    create_adamw8bit_quantiles1(h_quantiles1.data());
    create_adamw8bit_quantiles2(h_quantiles2.data());

    // Check that quantiles1 is sorted
    for (size_t i = 1; i < 256; ++i) {
        REQUIRE(h_quantiles1[i] >= h_quantiles1[i-1]);
    }

    // Check that quantiles2 is sorted
    for (size_t i = 1; i < 256; ++i) {
        REQUIRE(h_quantiles2[i] >= h_quantiles2[i-1]);
    }

    // quantiles1 should be signed (contain negative values)
    bool has_negative = false;
    for (float v : h_quantiles1) {
        if (v < 0) has_negative = true;
    }
    REQUIRE(has_negative);

    // quantiles2 should be unsigned (all non-negative)
    for (float v : h_quantiles2) {
        REQUIRE(v >= 0.0f);
    }

    // Both should contain 0 and values close to 1
    REQUIRE(h_quantiles1[127] == Catch::Approx(0.0f).margin(0.1f));  // Middle should be near 0
    REQUIRE(h_quantiles2[0] == Catch::Approx(0.0f).margin(0.01f));   // First should be 0 or near 0
}

TEST_CASE("AdamW 8-bit FP32 correctness", "[optimizers][adamw8bit]") {
    const size_t n = 4096 * 16;  // 64K parameters
    const size_t num_blocks = (n + 2047) / 2048;
    const int num_steps = 10;
    const float lr = 1e-3f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float weight_decay = 0.01f;

    // Initialize parameters
    std::vector<float> h_params = uniform_host(n, -1.0f, 1.0f, 42);
    std::vector<float> h_params_cpu = h_params;

    // Create quantization maps on host then copy to device
    std::vector<float> h_quantiles1(256), h_quantiles2(256);
    create_adamw8bit_quantiles1(h_quantiles1.data());
    create_adamw8bit_quantiles2(h_quantiles2.data());

    // Device buffers
    thrust::device_vector<float> d_params = to_device(h_params);
    thrust::device_vector<unsigned char> d_state1(n);
    thrust::device_vector<unsigned char> d_state2(n);
    thrust::device_vector<float> d_absmax1(num_blocks);
    thrust::device_vector<float> d_absmax2(num_blocks);
    thrust::device_vector<float> d_quantiles1 = to_device(h_quantiles1);
    thrust::device_vector<float> d_quantiles2 = to_device(h_quantiles2);

    // Initialize 8-bit state
    init_adamw8bit_state(
        thrust::raw_pointer_cast(d_state1.data()),
        thrust::raw_pointer_cast(d_state2.data()),
        thrust::raw_pointer_cast(d_absmax1.data()),
        thrust::raw_pointer_cast(d_absmax2.data()),
        n, 0
    );

    AdamWState cpu_state;

    for (int step = 1; step <= num_steps; ++step) {
        std::vector<float> h_grads = uniform_host(n, -0.1f, 0.1f, 42 + step);
        thrust::device_vector<float> d_grads = to_device(h_grads);

        // 8-bit GPU update (nullptr for gnorm_scale means use 1.0f)
        adamw_update_8bit(
            thrust::raw_pointer_cast(d_params.data()),
            thrust::raw_pointer_cast(d_grads.data()),
            thrust::raw_pointer_cast(d_state1.data()),
            thrust::raw_pointer_cast(d_state2.data()),
            n, lr, beta1, beta2, step, eps, weight_decay, nullptr,
            thrust::raw_pointer_cast(d_quantiles1.data()),
            thrust::raw_pointer_cast(d_quantiles2.data()),
            thrust::raw_pointer_cast(d_absmax1.data()),
            thrust::raw_pointer_cast(d_absmax2.data()),
            0
        );

        // CPU reference update
        adamw_cpu(h_params_cpu, h_grads, cpu_state, lr, beta1, beta2, eps, weight_decay);

        cudaDeviceSynchronize();
    }

    std::vector<float> h_params_gpu = from_device(d_params);
    float max_diff = max_abs_diff(h_params_gpu, h_params_cpu);
    float rel_err = relative_error(h_params_gpu, h_params_cpu);

    INFO("Max abs diff: " << max_diff);
    INFO("Relative error: " << rel_err);
    // 8-bit quantization introduces more error, but should still be reasonable
    REQUIRE(max_diff < 0.1f);
    REQUIRE(rel_err < 0.05f);
}

TEST_CASE("AdamW 8-bit BF16 correctness", "[optimizers][adamw8bit]") {
    const size_t n = 4096 * 16;
    const size_t num_blocks = (n + 2047) / 2048;
    const int num_steps = 10;
    const float lr = 1e-3f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float weight_decay = 0.01f;

    std::vector<float> h_params_f = uniform_host(n, -1.0f, 1.0f, 42);
    std::vector<float> h_params_cpu = h_params_f;
    std::vector<nv_bfloat16> h_params_bf16 = to_bf16(h_params_f);

    std::vector<float> h_quantiles1(256), h_quantiles2(256);
    create_adamw8bit_quantiles1(h_quantiles1.data());
    create_adamw8bit_quantiles2(h_quantiles2.data());

    thrust::device_vector<nv_bfloat16> d_params = to_device(h_params_bf16);
    thrust::device_vector<unsigned char> d_state1(n);
    thrust::device_vector<unsigned char> d_state2(n);
    thrust::device_vector<float> d_absmax1(num_blocks);
    thrust::device_vector<float> d_absmax2(num_blocks);
    thrust::device_vector<float> d_quantiles1 = to_device(h_quantiles1);
    thrust::device_vector<float> d_quantiles2 = to_device(h_quantiles2);

    init_adamw8bit_state(
        thrust::raw_pointer_cast(d_state1.data()),
        thrust::raw_pointer_cast(d_state2.data()),
        thrust::raw_pointer_cast(d_absmax1.data()),
        thrust::raw_pointer_cast(d_absmax2.data()),
        n, 0
    );

    AdamWState cpu_state;

    for (int step = 1; step <= num_steps; ++step) {
        std::vector<float> h_grads_f = uniform_host(n, -0.1f, 0.1f, 42 + step);
        std::vector<nv_bfloat16> h_grads_bf16 = to_bf16(h_grads_f);
        thrust::device_vector<nv_bfloat16> d_grads = to_device(h_grads_bf16);

        // nullptr for gnorm_scale means use 1.0f
        adamw_update_8bit(
            thrust::raw_pointer_cast(d_params.data()),
            thrust::raw_pointer_cast(d_grads.data()),
            thrust::raw_pointer_cast(d_state1.data()),
            thrust::raw_pointer_cast(d_state2.data()),
            n, lr, beta1, beta2, step, eps, weight_decay, nullptr,
            thrust::raw_pointer_cast(d_quantiles1.data()),
            thrust::raw_pointer_cast(d_quantiles2.data()),
            thrust::raw_pointer_cast(d_absmax1.data()),
            thrust::raw_pointer_cast(d_absmax2.data()),
            0
        );

        std::vector<float> h_grads_rounded = round_bf16(h_grads_f);
        adamw_cpu(h_params_cpu, h_grads_rounded, cpu_state, lr, beta1, beta2, eps, weight_decay);
        h_params_cpu = round_bf16(h_params_cpu);

        cudaDeviceSynchronize();
    }

    std::vector<nv_bfloat16> h_params_gpu_bf16 = from_device(d_params);
    std::vector<float> h_params_gpu = bf16_to_float(h_params_gpu_bf16);

    float max_diff = max_abs_diff(h_params_gpu, h_params_cpu);
    float rel_err = relative_error(h_params_gpu, h_params_cpu);

    INFO("Max abs diff: " << max_diff);
    INFO("Relative error: " << rel_err);
    REQUIRE(max_diff < 0.15f);
    REQUIRE(rel_err < 0.1f);
}

// ============================================================================
// BENCHMARKS
// ============================================================================

TEST_CASE("AdamW 8-bit benchmark - FP32 params", "[optimizers][adamw8bit][benchmark]") {
    const size_t n = 50 * 1024 * 1024;  // 50M parameters
    const size_t num_blocks = (n + 2047) / 2048;
    const int warmup_iters = 5;
    const int bench_iters = 20;
    const float lr = 1e-3f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float weight_decay = 0.01f;

    thrust::device_vector<float> d_params(n);
    thrust::device_vector<float> d_grads(n);
    thrust::device_vector<unsigned char> d_state1(n);
    thrust::device_vector<unsigned char> d_state2(n);
    thrust::device_vector<float> d_absmax1(num_blocks);
    thrust::device_vector<float> d_absmax2(num_blocks);

    std::vector<float> h_quantiles1(256), h_quantiles2(256);
    create_adamw8bit_quantiles1(h_quantiles1.data());
    create_adamw8bit_quantiles2(h_quantiles2.data());
    thrust::device_vector<float> d_quantiles1 = to_device(h_quantiles1);
    thrust::device_vector<float> d_quantiles2 = to_device(h_quantiles2);

    std::vector<float> h_data = uniform_host(n, -1.0f, 1.0f, 42);
    thrust::copy(h_data.begin(), h_data.end(), d_params.begin());
    thrust::copy(h_data.begin(), h_data.end(), d_grads.begin());

    init_adamw8bit_state(
        thrust::raw_pointer_cast(d_state1.data()),
        thrust::raw_pointer_cast(d_state2.data()),
        thrust::raw_pointer_cast(d_absmax1.data()),
        thrust::raw_pointer_cast(d_absmax2.data()),
        n, 0
    );

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    CudaTimer timer;

    for (int i = 0; i < warmup_iters; ++i) {
        adamw_update_8bit(
            thrust::raw_pointer_cast(d_params.data()),
            thrust::raw_pointer_cast(d_grads.data()),
            thrust::raw_pointer_cast(d_state1.data()),
            thrust::raw_pointer_cast(d_state2.data()),
            n, lr, beta1, beta2, i + 1, eps, weight_decay, nullptr,
            thrust::raw_pointer_cast(d_quantiles1.data()),
            thrust::raw_pointer_cast(d_quantiles2.data()),
            thrust::raw_pointer_cast(d_absmax1.data()),
            thrust::raw_pointer_cast(d_absmax2.data()),
            stream
        );
    }
    cudaStreamSynchronize(stream);

    timer.start(stream);
    for (int i = 0; i < bench_iters; ++i) {
        adamw_update_8bit(
            thrust::raw_pointer_cast(d_params.data()),
            thrust::raw_pointer_cast(d_grads.data()),
            thrust::raw_pointer_cast(d_state1.data()),
            thrust::raw_pointer_cast(d_state2.data()),
            n, lr, beta1, beta2, warmup_iters + i + 1, eps, weight_decay, nullptr,
            thrust::raw_pointer_cast(d_quantiles1.data()),
            thrust::raw_pointer_cast(d_quantiles2.data()),
            thrust::raw_pointer_cast(d_absmax1.data()),
            thrust::raw_pointer_cast(d_absmax2.data()),
            stream
        );
    }
    float total_ms = timer.stop(stream);
    float avg_ms = total_ms / bench_iters;

    // Memory for 8-bit: params (fp32), grads (fp32), states (uint8), absmax (fp32)
    double bytes_accessed = 4.0 * n * sizeof(float) +   // params r/w, grads r
                           4.0 * n * sizeof(unsigned char) +  // states r/w
                           4.0 * num_blocks * sizeof(float);  // absmax r/w
    double bandwidth_gbps = (bytes_accessed / (avg_ms / 1000.0)) / 1e9;

    // Memory savings calculation
    double fp32_state_mem = 2.0 * n * sizeof(float);  // m + v in FP32
    double int8_state_mem = 2.0 * n * sizeof(unsigned char) + 2.0 * num_blocks * sizeof(float);
    double savings = (1.0 - int8_state_mem / fp32_state_mem) * 100.0;

    printf("\n=== AdamW 8-bit FP32 params Benchmark ===\n");
    printf("Parameters: %.1fM\n", n / 1e6);
    printf("Avg time: %.3f ms\n", avg_ms);
    printf("Throughput: %.2f M params/ms\n", (n / 1e6) / avg_ms);
    printf("Effective bandwidth: %.1f GB/s\n", bandwidth_gbps);
    printf("State memory: %.1f MB (8-bit m) + %.1f MB (8-bit v) + %.2f MB (absmax) = %.1f MB\n",
           n / 1e6, n / 1e6, 2 * num_blocks * sizeof(float) / 1e6,
           int8_state_mem / 1e6);
    printf("Memory savings vs FP32 states: %.1f%%\n", savings);

    cudaStreamDestroy(stream);
    REQUIRE(avg_ms > 0);
}

TEST_CASE("AdamW 8-bit benchmark - BF16 params", "[optimizers][adamw8bit][benchmark]") {
    const size_t n = 50 * 1024 * 1024;  // 50M parameters
    const size_t num_blocks = (n + 2047) / 2048;
    const int warmup_iters = 5;
    const int bench_iters = 20;
    const float lr = 1e-3f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float weight_decay = 0.01f;

    thrust::device_vector<nv_bfloat16> d_params(n);
    thrust::device_vector<nv_bfloat16> d_grads(n);
    thrust::device_vector<unsigned char> d_state1(n);
    thrust::device_vector<unsigned char> d_state2(n);
    thrust::device_vector<float> d_absmax1(num_blocks);
    thrust::device_vector<float> d_absmax2(num_blocks);

    std::vector<float> h_quantiles1(256), h_quantiles2(256);
    create_adamw8bit_quantiles1(h_quantiles1.data());
    create_adamw8bit_quantiles2(h_quantiles2.data());
    thrust::device_vector<float> d_quantiles1 = to_device(h_quantiles1);
    thrust::device_vector<float> d_quantiles2 = to_device(h_quantiles2);

    std::vector<float> h_data = uniform_host(n, -1.0f, 1.0f, 42);
    std::vector<nv_bfloat16> h_data_bf16 = to_bf16(h_data);
    thrust::copy(h_data_bf16.begin(), h_data_bf16.end(), d_params.begin());
    thrust::copy(h_data_bf16.begin(), h_data_bf16.end(), d_grads.begin());

    init_adamw8bit_state(
        thrust::raw_pointer_cast(d_state1.data()),
        thrust::raw_pointer_cast(d_state2.data()),
        thrust::raw_pointer_cast(d_absmax1.data()),
        thrust::raw_pointer_cast(d_absmax2.data()),
        n, 0
    );

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    CudaTimer timer;

    for (int i = 0; i < warmup_iters; ++i) {
        adamw_update_8bit(
            thrust::raw_pointer_cast(d_params.data()),
            thrust::raw_pointer_cast(d_grads.data()),
            thrust::raw_pointer_cast(d_state1.data()),
            thrust::raw_pointer_cast(d_state2.data()),
            n, lr, beta1, beta2, i + 1, eps, weight_decay, nullptr,
            thrust::raw_pointer_cast(d_quantiles1.data()),
            thrust::raw_pointer_cast(d_quantiles2.data()),
            thrust::raw_pointer_cast(d_absmax1.data()),
            thrust::raw_pointer_cast(d_absmax2.data()),
            stream
        );
    }
    cudaStreamSynchronize(stream);

    timer.start(stream);
    for (int i = 0; i < bench_iters; ++i) {
        adamw_update_8bit(
            thrust::raw_pointer_cast(d_params.data()),
            thrust::raw_pointer_cast(d_grads.data()),
            thrust::raw_pointer_cast(d_state1.data()),
            thrust::raw_pointer_cast(d_state2.data()),
            n, lr, beta1, beta2, warmup_iters + i + 1, eps, weight_decay, nullptr,
            thrust::raw_pointer_cast(d_quantiles1.data()),
            thrust::raw_pointer_cast(d_quantiles2.data()),
            thrust::raw_pointer_cast(d_absmax1.data()),
            thrust::raw_pointer_cast(d_absmax2.data()),
            stream
        );
    }
    float total_ms = timer.stop(stream);
    float avg_ms = total_ms / bench_iters;

    double bytes_accessed = 4.0 * n * sizeof(nv_bfloat16) +
                           4.0 * n * sizeof(unsigned char) +
                           4.0 * num_blocks * sizeof(float);
    double bandwidth_gbps = (bytes_accessed / (avg_ms / 1000.0)) / 1e9;

    double fp32_state_mem = 2.0 * n * sizeof(float);
    double int8_state_mem = 2.0 * n * sizeof(unsigned char) + 2.0 * num_blocks * sizeof(float);
    double savings = (1.0 - int8_state_mem / fp32_state_mem) * 100.0;

    printf("\n=== AdamW 8-bit BF16 params Benchmark ===\n");
    printf("Parameters: %.1fM\n", n / 1e6);
    printf("Avg time: %.3f ms\n", avg_ms);
    printf("Throughput: %.2f M params/ms\n", (n / 1e6) / avg_ms);
    printf("Effective bandwidth: %.1f GB/s\n", bandwidth_gbps);
    printf("State memory: %.1f MB (8-bit m) + %.1f MB (8-bit v) + %.2f MB (absmax) = %.1f MB\n",
           n / 1e6, n / 1e6, 2 * num_blocks * sizeof(float) / 1e6,
           int8_state_mem / 1e6);
    printf("Memory savings vs FP32 states: %.1f%%\n", savings);

    cudaStreamDestroy(stream);
    REQUIRE(avg_ms > 0);
}
