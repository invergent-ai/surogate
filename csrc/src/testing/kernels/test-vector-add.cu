// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include "kernels/kernels.h"

TEST_CASE("Forward addition is independent of token position", "[kernels][vector-add]") {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) SKIP("CUDA device required");
    // Repeated rows expose index-dependent rounding. The final row also tests
    // the scalar tail of the vectorized kernel, as does one-token decoding.
    constexpr int rows = 37, width = 259, size = rows * width;
    for (float scale : {1.0f, 0.7f}) {
        std::vector<nv_bfloat16> left(size), right(size), actual(size), single(width);
        for (int i = 0; i < size; ++i) {
            left[i] = nv_bfloat16(std::sin((i % width) * .031f));
            right[i] = nv_bfloat16(std::cos((i % width) * .071f) * .017f);
        }
        nv_bfloat16 *a, *b, *out;
        REQUIRE(cudaMalloc(&a, size * sizeof(nv_bfloat16)) == cudaSuccess);
        REQUIRE(cudaMalloc(&b, size * sizeof(nv_bfloat16)) == cudaSuccess);
        REQUIRE(cudaMalloc(&out, size * sizeof(nv_bfloat16)) == cudaSuccess);
        REQUIRE(cudaMemcpy(a, left.data(), size * sizeof(nv_bfloat16), cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(b, right.data(), size * sizeof(nv_bfloat16), cudaMemcpyHostToDevice) == cudaSuccess);
        vector_add(out, a, b, scale, size, nullptr);
        REQUIRE(cudaMemcpy(actual.data(), out, size * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost) == cudaSuccess);
        vector_add(out, a, b, scale, width, nullptr);
        REQUIRE(cudaMemcpy(single.data(), out, width * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost) == cudaSuccess);
        for (int i = 0; i < size; ++i) {
            REQUIRE(float(actual[i]) == float(nv_bfloat16(scale * (float(left[i]) + float(right[i])))));
            REQUIRE(float(actual[i]) == float(single[i % width]));
        }
        // LoRA scaling uses the same-input specialization in place.
        vector_add(a, a, a, scale * .5f, size, nullptr);
        REQUIRE(cudaMemcpy(actual.data(), a, size * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost) == cudaSuccess);
        for (int i = 0; i < size; ++i)
            REQUIRE(float(actual[i]) == float(nv_bfloat16(scale * float(left[i]))));
        cudaFree(a);
        cudaFree(b);
        cudaFree(out);
    }
}
