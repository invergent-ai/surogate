// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <catch2/catch_test_macros.hpp>

#include "kernels/kernels.h"
#include "kernels/matmul_plans.h"
#include "test_utils.h"

// Forward declarations from csrc/src/kernels/matmul.cpp
cublasLtHandle_t create_cublaslt_handle();
void destroy_cublaslt_handle(cublasLtHandle_t handle) noexcept;

namespace {

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value) : name_(name) {
        const char* old = std::getenv(name);
        if (old) {
            had_old_ = true;
            old_value_ = old;
        }
        if (value) {
            setenv(name, value, 1);
        } else {
            unsetenv(name);
        }
    }

    ~ScopedEnvVar() {
        if (had_old_) {
            setenv(name_.c_str(), old_value_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
    std::string name_;
    bool had_old_ = false;
    std::string old_value_;
};

static bool cuda_available() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

static void linear_cpu(float* out,
                       const std::vector<float>& weight_oc_k,
                       const std::vector<float>& input_bt_k,
                       const std::vector<float>& bias_oc,
                       int BT, int OC, int K) {
    for (int bt = 0; bt < BT; ++bt) {
        for (int oc = 0; oc < OC; ++oc) {
            float acc = bias_oc.empty() ? 0.0f : bias_oc[oc];
            const float* w = &weight_oc_k[static_cast<size_t>(oc) * K];
            const float* x = &input_bt_k[static_cast<size_t>(bt) * K];
            for (int k = 0; k < K; ++k) {
                acc += x[k] * w[k];
            }
            out[static_cast<size_t>(bt) * OC + oc] = acc;
        }
    }
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

static float max_abs(const std::vector<float>& a) {
    float m = 0.f;
    for (float v : a) m = std::max(m, std::fabs(v));
    return m;
}

} // namespace

TEST_CASE("MatmulPlanCache matches baseline for BF16 linear + bias and supports bias pointer changes", "[matmul][plans]") {
    if (!cuda_available()) {
        SKIP("No CUDA device available");
    }

    // Keep autotune off for deterministic baseline comparisons.
    ScopedEnvVar autotune("SUROGATE_MATMUL_AUTOTUNE", "0");

    constexpr int BT = 32;
    constexpr int OC = 64;
    constexpr int K = 64;

    // Host data (float), then quantize inputs/weights to BF16 for the GPU.
    std::vector<float> h_w_f = testing_utils::uniform_host((long)OC * K, -1.f, 1.f, 1001ull);
    std::vector<float> h_x_f = testing_utils::uniform_host((long)BT * K, -1.f, 1.f, 2002ull);
    std::vector<float> h_b1 = testing_utils::uniform_host(OC, -0.25f, 0.25f, 3003ull);
    std::vector<float> h_b2 = testing_utils::uniform_host(OC, -0.25f, 0.25f, 4004ull);

    // CPU reference uses BF16-rounded inputs/weights to match GPU math.
    std::vector<float> h_w_q = testing_utils::round_bf16(h_w_f);
    std::vector<float> h_x_q = testing_utils::round_bf16(h_x_f);

    std::vector<float> h_ref1((size_t)BT * OC);
    std::vector<float> h_ref2((size_t)BT * OC);
    linear_cpu(h_ref1.data(), h_w_q, h_x_q, h_b1, BT, OC, K);
    linear_cpu(h_ref2.data(), h_w_q, h_x_q, h_b2, BT, OC, K);

    // Device buffers.
    thrust::device_vector<nv_bfloat16> d_w = testing_utils::to_device(testing_utils::to_bf16(h_w_f));
    thrust::device_vector<nv_bfloat16> d_x = testing_utils::to_device(testing_utils::to_bf16(h_x_f));
    thrust::device_vector<float> d_b1 = testing_utils::to_device(h_b1);
    thrust::device_vector<float> d_b2 = testing_utils::to_device(h_b2);
    thrust::device_vector<float> d_out_base((size_t)BT * OC);
    thrust::device_vector<float> d_out_plan((size_t)BT * OC);
    thrust::device_vector<uint8_t> d_ws(32u * 1024u * 1024u);

    cublasLtHandle_t handle = create_cublaslt_handle();
    int device_id = 0;
    REQUIRE(cudaGetDevice(&device_id) == cudaSuccess);
    MatmulPlanCache plan_cache(device_id);

    auto ws_ptr = reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_ws.data()));
    const std::size_t ws_size = d_ws.size();

    // Baseline: no plan cache.
    matmul(thrust::raw_pointer_cast(d_out_base.data()),
           thrust::raw_pointer_cast(d_w.data()),
           thrust::raw_pointer_cast(d_x.data()),
           thrust::raw_pointer_cast(d_b1.data()),
           /*scale_a=*/nullptr, /*scale_b=*/nullptr,
           handle, ws_ptr, ws_size,
           /*M=*/OC, /*N=*/BT, /*K=*/K, EMMTranspose::TN, /*accumulate=*/false, /*stream=*/0,
           /*plan_cache=*/nullptr);

    // Plan cache path.
    matmul(thrust::raw_pointer_cast(d_out_plan.data()),
           thrust::raw_pointer_cast(d_w.data()),
           thrust::raw_pointer_cast(d_x.data()),
           thrust::raw_pointer_cast(d_b1.data()),
           /*scale_a=*/nullptr, /*scale_b=*/nullptr,
           handle, ws_ptr, ws_size,
           /*M=*/OC, /*N=*/BT, /*K=*/K, EMMTranspose::TN, /*accumulate=*/false, /*stream=*/0,
           &plan_cache);

    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<float> h_base1 = testing_utils::from_device(d_out_base);
    std::vector<float> h_plan1 = testing_utils::from_device(d_out_plan);

    // Cache path should match baseline closely.
    REQUIRE(max_abs_diff(h_base1, h_plan1) < 1e-3f);

    // Both should be reasonably close to CPU reference (BF16-rounded).
    // Keep tolerance loose: GEMM accumulation order may differ from naive CPU loops.
    REQUIRE(max_abs_diff(h_plan1, h_ref1) < 5e-2f * std::max(1.0f, max_abs(h_ref1)));

    // Bias pointer update: reuse the same plan cache but pass a different bias pointer.
    matmul(thrust::raw_pointer_cast(d_out_base.data()),
           thrust::raw_pointer_cast(d_w.data()),
           thrust::raw_pointer_cast(d_x.data()),
           thrust::raw_pointer_cast(d_b2.data()),
           /*scale_a=*/nullptr, /*scale_b=*/nullptr,
           handle, ws_ptr, ws_size,
           /*M=*/OC, /*N=*/BT, /*K=*/K, EMMTranspose::TN, /*accumulate=*/false, /*stream=*/0,
           /*plan_cache=*/nullptr);

    matmul(thrust::raw_pointer_cast(d_out_plan.data()),
           thrust::raw_pointer_cast(d_w.data()),
           thrust::raw_pointer_cast(d_x.data()),
           thrust::raw_pointer_cast(d_b2.data()),
           /*scale_a=*/nullptr, /*scale_b=*/nullptr,
           handle, ws_ptr, ws_size,
           /*M=*/OC, /*N=*/BT, /*K=*/K, EMMTranspose::TN, /*accumulate=*/false, /*stream=*/0,
           &plan_cache);

    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<float> h_base2 = testing_utils::from_device(d_out_base);
    std::vector<float> h_plan2 = testing_utils::from_device(d_out_plan);

    REQUIRE(max_abs_diff(h_base2, h_plan2) < 1e-3f);
    REQUIRE(max_abs_diff(h_plan2, h_ref2) < 5e-2f * std::max(1.0f, max_abs(h_ref2)));

    destroy_cublaslt_handle(handle);
}

TEST_CASE("MatmulPlanCache works with matmul_strided_c (float) and accumulate", "[matmul][plans][strided]") {
    if (!cuda_available()) {
        SKIP("No CUDA device available");
    }

    ScopedEnvVar autotune("SUROGATE_MATMUL_AUTOTUNE", "0");

    constexpr int BT = 16;
    constexpr int K = 32;
    constexpr int total_OC = 80;
    constexpr int slice_OC = 32;
    constexpr int out_offset = 16;  // must keep pointer 16B-aligned: 16 * 4 = 64 bytes

    std::vector<float> h_w = testing_utils::uniform_host((long)slice_OC * K, -1.f, 1.f, 777ull);
    std::vector<float> h_x = testing_utils::uniform_host((long)BT * K, -1.f, 1.f, 888ull);
    std::vector<float> h_b = testing_utils::uniform_host(slice_OC, -0.1f, 0.1f, 999ull);
    std::vector<float> h_out_init = testing_utils::uniform_host((long)BT * total_OC, -0.5f, 0.5f, 111ull);

    thrust::device_vector<float> d_w = testing_utils::to_device(h_w);
    thrust::device_vector<float> d_x = testing_utils::to_device(h_x);
    thrust::device_vector<float> d_b = testing_utils::to_device(h_b);
    thrust::device_vector<float> d_out_base = testing_utils::to_device(h_out_init);
    thrust::device_vector<float> d_out_plan = testing_utils::to_device(h_out_init);
    thrust::device_vector<uint8_t> d_ws(32u * 1024u * 1024u);

    cublasLtHandle_t handle = create_cublaslt_handle();
    int device_id = 0;
    REQUIRE(cudaGetDevice(&device_id) == cudaSuccess);
    MatmulPlanCache plan_cache(device_id);

    auto ws_ptr = reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(d_ws.data()));
    const std::size_t ws_size = d_ws.size();

    float* out_base = thrust::raw_pointer_cast(d_out_base.data()) + out_offset;
    float* out_plan = thrust::raw_pointer_cast(d_out_plan.data()) + out_offset;

    // Baseline (no cache) vs plan cache should match closely.
    matmul_strided_c(out_base,
                     thrust::raw_pointer_cast(d_w.data()),
                     thrust::raw_pointer_cast(d_x.data()),
                     thrust::raw_pointer_cast(d_b.data()),
                     /*scale_a=*/nullptr, /*scale_b=*/nullptr,
                     handle, ws_ptr, ws_size,
                     /*M=*/slice_OC, /*N=*/BT, /*K=*/K, EMMTranspose::TN,
                     /*accumulate=*/true,
                     /*ldc=*/total_OC,
                     /*stream=*/0,
                     /*plan_cache=*/nullptr);

    matmul_strided_c(out_plan,
                     thrust::raw_pointer_cast(d_w.data()),
                     thrust::raw_pointer_cast(d_x.data()),
                     thrust::raw_pointer_cast(d_b.data()),
                     /*scale_a=*/nullptr, /*scale_b=*/nullptr,
                     handle, ws_ptr, ws_size,
                     /*M=*/slice_OC, /*N=*/BT, /*K=*/K, EMMTranspose::TN,
                     /*accumulate=*/true,
                     /*ldc=*/total_OC,
                     /*stream=*/0,
                     &plan_cache);

    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<float> h_base = testing_utils::from_device(d_out_base);
    std::vector<float> h_plan = testing_utils::from_device(d_out_plan);

    REQUIRE(max_abs_diff(h_base, h_plan) < 1e-3f);

    destroy_cublaslt_handle(handle);
}
