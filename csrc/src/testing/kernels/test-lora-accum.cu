// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include "kernels/kernels.h"
#include "utilities/tensor.h"

namespace {
struct DeviceTensor {
    Tensor tensor;
    DeviceTensor(ETensorDType dtype, std::vector<long> shape) {
        long n = 1;
        for (long d : shape)
            n *= d;
        void* ptr = nullptr;
        REQUIRE(cudaMalloc(&ptr, n * get_dtype_size(dtype)) == cudaSuccess);
        tensor = Tensor::from_pointer(static_cast<std::byte*>(ptr), 0, dtype, shape);
    }
    ~DeviceTensor() {
        cudaFree(tensor.Data);
    }
    void put(const std::vector<nv_bfloat16>& values) {
        REQUIRE(values.size() * sizeof(nv_bfloat16) == tensor.bytes());
        REQUIRE(cudaMemcpy(tensor.Data, values.data(), tensor.bytes(), cudaMemcpyHostToDevice) == cudaSuccess);
    }
    std::vector<nv_bfloat16> get() {
        std::vector<nv_bfloat16> out(tensor.nelem());
        REQUIRE(cudaMemcpy(out.data(), tensor.Data, tensor.bytes(), cudaMemcpyDeviceToHost) == cudaSuccess);
        return out;
    }
};

// Runs the accumulate on a [rows, total] output and checks every element: the slice against a
// host sum over the rank in order with fused multiply-adds, everything else untouched. `exact`
// asks for bit equality (the vectorized kernel); otherwise one BF16 step of slack is allowed.
void check_accumulate(int rows, int total, int out, int offset, int rank, bool exact) {
    constexpr float scaling = 0.625f;
    std::vector<nv_bfloat16> output(static_cast<std::size_t>(rows) * total), B(static_cast<std::size_t>(out) * rank),
        inter(static_cast<std::size_t>(rows) * rank);
    for (std::size_t i = 0; i < output.size(); ++i)
        output[i] = nv_bfloat16(std::sin(i * 0.013f) * 2.0f);
    for (std::size_t i = 0; i < B.size(); ++i)
        B[i] = nv_bfloat16(std::cos(i * 0.37f) * 0.05f);
    for (std::size_t i = 0; i < inter.size(); ++i)
        inter[i] = nv_bfloat16(std::sin(i * 0.71f + 0.3f));

    DeviceTensor d_output(ETensorDType::BF16, {rows, total}), d_B(ETensorDType::BF16, {out, rank}),
        d_inter(ETensorDType::BF16, {rows, rank});
    d_output.put(output);
    d_B.put(B);
    d_inter.put(inter);
    REQUIRE(lora_accum_b_small_rank_bf16(d_output.tensor, d_B.tensor, d_inter.tensor, rows, total, out, offset, rank,
                                         scaling, nullptr));
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const auto actual = d_output.get();

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < total; ++col) {
            const std::size_t idx = static_cast<std::size_t>(row) * total + col;
            const float before = float(output[idx]);
            if (col < offset || col >= offset + out) {
                REQUIRE(float(actual[idx]) == before);
                continue;
            }
            float acc = 0.0f;
            for (int r = 0; r < rank; ++r)
                acc = std::fma(float(inter[static_cast<std::size_t>(row) * rank + r]),
                               float(B[static_cast<std::size_t>(col - offset) * rank + r]),
                               acc);
            const float expected = float(nv_bfloat16(std::fma(scaling, acc, before)));
            if (exact) {
                REQUIRE(float(actual[idx]) == expected);
            } else {
                REQUIRE(std::abs(float(actual[idx]) - expected) <= std::abs(expected) * 0.0079f + 1e-6f);
            }
        }
    }
}

bool have_gpu() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count;
}
}  // namespace

TEST_CASE("LoRA B accumulates into a slice of a fused projection", "[kernels][lora]") {
    if (!have_gpu()) SKIP("CUDA device required");
    // Rows that fill neither a 16-row nor a 64-row tile, a slice narrower than one 128-column
    // block and offset inside the row, the whole row, and a slice at the end of it.
    for (int rank : {8, 16, 32, 64}) {
        check_accumulate(133, 328, 200, 64, rank, true);
        check_accumulate(70, 256, 256, 0, rank, true);
        check_accumulate(17, 520, 136, 384, rank, true);
    }
}

TEST_CASE("LoRA B accumulate falls back for slices off a 16-byte boundary", "[kernels][lora]") {
    if (!have_gpu()) SKIP("CUDA device required");
    for (int rank : {8, 16, 32, 64}) {
        check_accumulate(37, 120, 37, 3, rank, false);
        check_accumulate(21, 96, 40, 4, rank, false);
    }
}

TEST_CASE("LoRA B accumulate refuses ranks it has no kernel for", "[kernels][lora]") {
    if (!have_gpu()) SKIP("CUDA device required");
    DeviceTensor output(ETensorDType::BF16, {4, 64}), B(ETensorDType::BF16, {64, 12}), inter(ETensorDType::BF16, {4, 12});
    REQUIRE_FALSE(lora_accum_b_small_rank_bf16(output.tensor, B.tensor, inter.tensor, 4, 64, 64, 0, 12, 1.0f, nullptr));
}
