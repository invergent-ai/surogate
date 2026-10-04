// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include "kernels/kernels.h"

namespace {

bool have_gpu() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// Uniform in [-1, 1); zeroes A's K columns from zero_from on, as padded tokens do in a weight gradient.
void fill(std::vector<nv_bfloat16>& a, std::vector<nv_bfloat16>& b, int m, int k, int zero_from, std::uint32_t seed) {
    std::uint32_t state = seed;
    auto next = [&state]() {
        state = state * 1664525u + 1013904223u;
        return nv_bfloat16(static_cast<float>(state >> 8) * (2.0f / 16777216.0f) - 1.0f);
    };
    for (int kk = 0; kk < k; ++kk) {
        for (int i = 0; i < m; ++i)
            a[static_cast<std::size_t>(kk) * m + i] = kk < zero_from ? next() : nv_bfloat16(0.f);
    }
    for (auto& v : b)
        v = next();
}

}  // namespace

// The autotune tunes a shape on its first eager call. Whatever it picks must give the same bits as
// the algorithm an untuned call runs, here a graph-captured one (#233), on later data too: a padded
// batch let a split-K variant look identical while the probe ran on live operands.
TEST_CASE("Autotuned matmuls match the untuned algorithm bit for bit", "[kernels][matmul][matmul-autotune]") {
    if (!have_gpu()) SKIP("CUDA device required");
    cublasLtHandle_t handle;
    REQUIRE(cublasLtCreate(&handle) == CUBLAS_STATUS_SUCCESS);
    cudaStream_t stream;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    constexpr std::size_t workspace_size = 32u << 20;
    std::byte* workspace = nullptr;
    REQUIRE(cudaMalloc(&workspace, workspace_size) == cudaSuccess);

    // NT is the weight-gradient layout: A is m x k, B is n x k, K runs over tokens. `shift` moves every
    // operand 16 bytes off its allocation, as a slice of a larger buffer is.
    struct Shape {
        int m, n, k, zero_from, shift;
    };
    for (const Shape s : {Shape{2048, 1024, 2048, 384, 0},
                          Shape{1024, 3584, 2048, 640, 0},
                          Shape{3584, 1024, 2048, 2048, 0},
                          Shape{1536, 2048, 2048, 512, 8}}) {
        INFO("m=" << s.m << " n=" << s.n << " k=" << s.k << " zero_from=" << s.zero_from << " shift=" << s.shift);
        const std::size_t a_elems = static_cast<std::size_t>(s.m) * s.k;
        const std::size_t b_elems = static_cast<std::size_t>(s.n) * s.k;
        const std::size_t c_elems = static_cast<std::size_t>(s.m) * s.n;
        std::vector<nv_bfloat16> a(a_elems), b(b_elems), captured(c_elems), tuned(c_elems);
        nv_bfloat16 *base_a, *base_b, *base_graph, *base_eager;
        REQUIRE(cudaMalloc(&base_a, (a_elems + s.shift) * sizeof(nv_bfloat16)) == cudaSuccess);
        REQUIRE(cudaMalloc(&base_b, (b_elems + s.shift) * sizeof(nv_bfloat16)) == cudaSuccess);
        REQUIRE(cudaMalloc(&base_graph, (c_elems + s.shift) * sizeof(nv_bfloat16)) == cudaSuccess);
        REQUIRE(cudaMalloc(&base_eager, (c_elems + s.shift) * sizeof(nv_bfloat16)) == cudaSuccess);
        nv_bfloat16 *da = base_a + s.shift, *db = base_b + s.shift;
        nv_bfloat16 *dc_graph = base_graph + s.shift, *dc_eager = base_eager + s.shift;
        // On the matmul stream: a plain cudaMemcpy from pageable memory can return before its DMA lands.
        auto upload = [&]() {
            REQUIRE(cudaMemcpyAsync(da, a.data(), a_elems * sizeof(nv_bfloat16), cudaMemcpyHostToDevice, stream) ==
                    cudaSuccess);
            REQUIRE(cudaMemcpyAsync(db, b.data(), b_elems * sizeof(nv_bfloat16), cudaMemcpyHostToDevice, stream) ==
                    cudaSuccess);
            REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
        };
        auto run = [&](nv_bfloat16* c) {
            matmul(c,
                   da,
                   db,
                   nullptr,
                   nullptr,
                   nullptr,
                   handle,
                   workspace,
                   workspace_size,
                   s.m,
                   s.n,
                   s.k,
                   EMMTranspose::NT,
                   false,
                   stream);
        };

        // Captured first, so the graph holds the untuned algorithm.
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        REQUIRE(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess);
        run(dc_graph);
        REQUIRE(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
        REQUIRE(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess);

        // The first eager call tunes, on padded operands.
        fill(a, b, s.m, s.k, s.zero_from, 233u);
        upload();
        run(dc_eager);

        // Then both run on dense operands.
        fill(a, b, s.m, s.k, s.k, 234u);
        upload();
        REQUIRE(cudaGraphLaunch(exec, stream) == cudaSuccess);
        run(dc_eager);
        REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
        REQUIRE(cudaMemcpy(captured.data(), dc_graph, c_elems * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost) ==
                cudaSuccess);
        REQUIRE(cudaMemcpy(tuned.data(), dc_eager, c_elems * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost) ==
                cudaSuccess);

        std::size_t differing = 0;
        for (std::size_t i = 0; i < c_elems; ++i) {
            std::uint16_t x, y;
            std::memcpy(&x, &captured[i], sizeof(x));
            std::memcpy(&y, &tuned[i], sizeof(y));
            differing += x != y;
        }
        CHECK(differing == 0);

        REQUIRE(cudaGraphExecDestroy(exec) == cudaSuccess);
        REQUIRE(cudaGraphDestroy(graph) == cudaSuccess);
        cudaFree(base_a);
        cudaFree(base_b);
        cudaFree(base_graph);
        cudaFree(base_eager);
    }
    cudaFree(workspace);
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
    REQUIRE(cublasLtDestroy(handle) == CUBLAS_STATUS_SUCCESS);
}
