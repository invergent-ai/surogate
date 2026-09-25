// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <set>

#include "runtime/core/fp8_run_state.h"
#include "runtime/core/run_state_types.h"

namespace {

// Every buffer's abs_max() and scale() must be its own slot inside the stats allocation, and
// scale() must be 16-byte aligned (cuBLASLt refuses FP8 scale pointers that are not).
void require_disjoint_stats(const Tensor* const* buffers, int count, const float* begin, const float* end) {
    std::set<std::uintptr_t> slots;
    for (int i = 0; i < count; ++i) {
        const float* amax = buffers[i]->Stats;  // what abs_max() returns
        const float* scale = buffers[i]->scale();
        REQUIRE(reinterpret_cast<std::uintptr_t>(scale) % 16 == 0);
        REQUIRE(amax >= begin);
        REQUIRE(scale + 1 <= end);
        REQUIRE(slots.insert(reinterpret_cast<std::uintptr_t>(amax)).second);
        REQUIRE(slots.insert(reinterpret_cast<std::uintptr_t>(scale)).second);
    }
}

}  // namespace

TEST_CASE("FP8 Stats blocks never share a scale slot", "[fp8][stats]") {
    // Stats blocks can start at any 4-byte boundary (e.g. inline after FP8 weight data);
    // try every offset within 16 bytes.
    alignas(64) static float storage[modules::fp8_stats_floats(4) + 4];
    for (int offset = 0; offset < 4; ++offset) {
        float* base = storage + offset;
        Tensor blocks[4]{};
        const Tensor* views[4];
        for (int i = 0; i < 4; ++i) {
            blocks[i].Stats = modules::fp8_stats_block(base, i);
            views[i] = &blocks[i];
        }
        require_disjoint_stats(views, 4, base, base + modules::fp8_stats_floats(4));
    }
}

TEST_CASE("FP8 forward buffers get disjoint Stats blocks", "[fp8][stats][cuda]") {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        SKIP("No CUDA device available");
    }
    REQUIRE(cudaSetDevice(0) == cudaSuccess);

    TensorAllocator allocator;
    modules::FP8ForwardQuantActivations quants;
    Tensor stats;
    modules::allocate_fp8_forward_buffers(quants, stats, allocator, 1, 16, 64, 128, 96, ETensorDType::FP8_E4M3);
    REQUIRE(stats.nelem() == modules::fp8_stats_floats(4));

    const Tensor* views[4] = {&quants.ln1, &quants.ln2, &quants.att, &quants.swiglu};
    const float* begin = stats.get<float>();
    require_disjoint_stats(views, 4, begin, begin + stats.nelem());
}

TEST_CASE("FP8 gradient-quant buffers get disjoint Stats blocks", "[fp8][stats][cuda]") {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        SKIP("No CUDA device available");
    }
    REQUIRE(cudaSetDevice(0) == cudaSuccess);

    TensorAllocator allocator;
    modules::SimplifiedQuantGradients grads;
    Tensor stats;
    modules::allocate_grad_quant_buffers(grads, stats, allocator, 1, 16, 64, 128, 96, ETensorDType::FP8_E5M2);
    REQUIRE(stats.nelem() == modules::fp8_stats_floats(4));
    REQUIRE(grads.d_res_ffn.nelem() == 16 * 64);
    REQUIRE(grads.d_res_att.nelem() == 16 * 64);
    REQUIRE(grads.d_mlp_up.nelem() == 16 * 128);
    REQUIRE(grads.d_qkv.nelem() == 16 * 96);

    const Tensor* views[4] = {&grads.d_res_ffn, &grads.d_res_att, &grads.d_mlp_up, &grads.d_qkv};
    for (const Tensor* t : views) {
        REQUIRE(t->DType == ETensorDType::FP8_E5M2);
    }
    const float* begin = stats.get<float>();
    require_disjoint_stats(views, 4, begin, begin + stats.nelem());
}
