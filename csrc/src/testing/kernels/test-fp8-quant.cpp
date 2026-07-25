// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "kernels/kernels.h"

namespace {

constexpr long kElementCount = 4096;

template <class Input>
std::vector<Input> make_input() {
    std::vector<Input> input(kElementCount, Input(0.25F));
    input[173] = Input(-7.5F);
    return input;
}

template <class Output, class Input>
void require_abs_quant_launch(const cudaDeviceProp& device_properties, float expected_inverse_scale) {
    const std::vector<Input> host_input = make_input<Input>();
    const float host_abs_max = 7.5F;

    Input* device_input = nullptr;
    Output* device_output = nullptr;
    float* device_abs_max = nullptr;
    float* device_scale = nullptr;
    REQUIRE(cudaMalloc(&device_input, kElementCount * sizeof(Input)) == cudaSuccess);
    REQUIRE(cudaMalloc(&device_output, kElementCount * sizeof(Output)) == cudaSuccess);
    REQUIRE(cudaMalloc(&device_abs_max, sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&device_scale, sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemcpy(device_input,
                       host_input.data(),
                       kElementCount * sizeof(Input),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(device_abs_max, &host_abs_max, sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);

    quantize_with_abs_max(device_output,
                          device_scale,
                          device_input,
                          device_abs_max,
                          kElementCount,
                          device_properties,
                          nullptr);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    if constexpr (!std::is_same_v<Output, std::int8_t>) {
        float host_scale = 0.0F;
        REQUIRE(cudaMemcpy(&host_scale, device_scale, sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(host_scale == Catch::Approx(expected_inverse_scale));
    }

    REQUIRE(cudaFree(device_scale) == cudaSuccess);
    REQUIRE(cudaFree(device_abs_max) == cudaSuccess);
    REQUIRE(cudaFree(device_output) == cudaSuccess);
    REQUIRE(cudaFree(device_input) == cudaSuccess);
}

template <class Output, class Input>
void require_delayed_quant_launch(const cudaDeviceProp& device_properties) {
    const std::vector<Input> host_input = make_input<Input>();
    const float host_delayed_scale = 2.0F;
    const float zero = 0.0F;

    Input* device_input = nullptr;
    Output* device_output = nullptr;
    float* device_recorded_amax = nullptr;
    float* device_inverse_scale = nullptr;
    float* device_delayed_scale = nullptr;
    REQUIRE(cudaMalloc(&device_input, kElementCount * sizeof(Input)) == cudaSuccess);
    REQUIRE(cudaMalloc(&device_output, kElementCount * sizeof(Output)) == cudaSuccess);
    REQUIRE(cudaMalloc(&device_recorded_amax, sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&device_inverse_scale, sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&device_delayed_scale, sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemcpy(device_input,
                       host_input.data(),
                       kElementCount * sizeof(Input),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(device_recorded_amax, &zero, sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(
                device_delayed_scale, &host_delayed_scale, sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);

    quantize_with_delayed_scale(device_output,
                                device_recorded_amax,
                                device_inverse_scale,
                                device_input,
                                device_delayed_scale,
                                kElementCount,
                                device_properties,
                                nullptr);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    float host_recorded_amax = 0.0F;
    float host_inverse_scale = 0.0F;
    REQUIRE(cudaMemcpy(
                &host_recorded_amax, device_recorded_amax, sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(
                &host_inverse_scale, device_inverse_scale, sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(host_recorded_amax == Catch::Approx(7.5F));
    REQUIRE(host_inverse_scale == Catch::Approx(0.5F));

    REQUIRE(cudaFree(device_delayed_scale) == cudaSuccess);
    REQUIRE(cudaFree(device_inverse_scale) == cudaSuccess);
    REQUIRE(cudaFree(device_recorded_amax) == cudaSuccess);
    REQUIRE(cudaFree(device_output) == cudaSuccess);
    REQUIRE(cudaFree(device_input) == cudaSuccess);
}

}  // namespace

TEST_CASE("FP8 abs-max reduction uses a launchable occupancy shape", "[fp8][quant][cuda]") {
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count == 0) {
        SKIP("No CUDA device available");
    }

    REQUIRE(cudaSetDevice(0) == cudaSuccess);
    cudaDeviceProp device_properties{};
    REQUIRE(cudaGetDeviceProperties(&device_properties, 0) == cudaSuccess);

    // Keep N divisible by the 128-bit vector width used by both FP32 and BF16
    // specializations.  A full-size tensor is unnecessary to exercise the
    // occupancy-selected launch configuration that regressed on SM120.
    SECTION("FP32 specialization") {
        std::vector<float> host_input(kElementCount, 0.25F);
        host_input[173] = -7.5F;

        float* device_input = nullptr;
        float* device_result = nullptr;
        REQUIRE(cudaMalloc(&device_input, kElementCount * sizeof(float)) == cudaSuccess);
        REQUIRE(cudaMalloc(&device_result, sizeof(float)) == cudaSuccess);
        REQUIRE(cudaMemcpy(device_input,
                           host_input.data(),
                           kElementCount * sizeof(float),
                           cudaMemcpyHostToDevice) == cudaSuccess);

        abs_max(device_result, device_input, kElementCount, device_properties, nullptr);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

        float host_result = 0.0F;
        REQUIRE(cudaMemcpy(&host_result, device_result, sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(host_result == Catch::Approx(7.5F));

        REQUIRE(cudaFree(device_result) == cudaSuccess);
        REQUIRE(cudaFree(device_input) == cudaSuccess);
    }

    SECTION("BF16 specialization used by imported model weights") {
        std::vector<nv_bfloat16> host_input(kElementCount, nv_bfloat16(0.25F));
        host_input[173] = nv_bfloat16(-7.5F);

        nv_bfloat16* device_input = nullptr;
        float* device_result = nullptr;
        REQUIRE(cudaMalloc(&device_input, kElementCount * sizeof(nv_bfloat16)) == cudaSuccess);
        REQUIRE(cudaMalloc(&device_result, sizeof(float)) == cudaSuccess);
        REQUIRE(cudaMemcpy(device_input,
                           host_input.data(),
                           kElementCount * sizeof(nv_bfloat16),
                           cudaMemcpyHostToDevice) == cudaSuccess);

        abs_max(device_result, device_input, kElementCount, device_properties, nullptr);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

        float host_result = 0.0F;
        REQUIRE(cudaMemcpy(&host_result, device_result, sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(host_result == Catch::Approx(7.5F));

        REQUIRE(cudaFree(device_result) == cudaSuccess);
        REQUIRE(cudaFree(device_input) == cudaSuccess);
    }
}

TEST_CASE("FP8 quantization launchers use exact-kernel occupancy shapes", "[fp8][quant][cuda]") {
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count == 0) {
        SKIP("No CUDA device available");
    }

    REQUIRE(cudaSetDevice(0) == cudaSuccess);
    cudaDeviceProp device_properties{};
    REQUIRE(cudaGetDeviceProperties(&device_properties, 0) == cudaSuccess);

    SECTION("absolute-max FP32 to BF16") {
        require_abs_quant_launch<nv_bfloat16, float>(device_properties, 1.0F);
    }
    SECTION("absolute-max FP32 to INT8") {
        require_abs_quant_launch<std::int8_t, float>(device_properties, 0.0F);
    }
    SECTION("absolute-max FP32 to FP8 E4M3") {
        require_abs_quant_launch<__nv_fp8_e4m3, float>(device_properties, 7.5F / 448.0F);
    }
    SECTION("absolute-max FP32 to FP8 E5M2") {
        require_abs_quant_launch<__nv_fp8_e5m2, float>(device_properties, 7.5F / 57344.0F);
    }
    SECTION("absolute-max BF16 to INT8") {
        require_abs_quant_launch<std::int8_t, nv_bfloat16>(device_properties, 0.0F);
    }
    SECTION("absolute-max BF16 to FP8 E4M3") {
        require_abs_quant_launch<__nv_fp8_e4m3, nv_bfloat16>(device_properties, 7.5F / 448.0F);
    }
    SECTION("absolute-max BF16 to FP8 E5M2") {
        require_abs_quant_launch<__nv_fp8_e5m2, nv_bfloat16>(device_properties, 7.5F / 57344.0F);
    }
    SECTION("delayed-scale FP32 to FP8 E4M3") {
        require_delayed_quant_launch<__nv_fp8_e4m3, float>(device_properties);
    }
    SECTION("delayed-scale FP32 to FP8 E5M2") {
        require_delayed_quant_launch<__nv_fp8_e5m2, float>(device_properties);
    }
    SECTION("delayed-scale BF16 to FP8 E4M3") {
        require_delayed_quant_launch<__nv_fp8_e4m3, nv_bfloat16>(device_properties);
    }
    SECTION("delayed-scale BF16 to FP8 E5M2") {
        require_delayed_quant_launch<__nv_fp8_e5m2, nv_bfloat16>(device_properties);
    }
}
