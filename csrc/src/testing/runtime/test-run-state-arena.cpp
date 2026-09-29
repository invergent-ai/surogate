// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "kernels/kernels.h"
#include "runtime/dsl/dsl_run_state.h"

TEST_CASE("Persistent scratch migration preserves alignment and embedding gradients", "[arena][embedding][gpu]") {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) SKIP("CUDA device required");

    constexpr int C = 1024, V = 16;
    for (const int T : {7, 16}) {
        INFO("sequence length=" << T);
        auto allocator = std::make_shared<TensorAllocator>();
        PretrainedConfig config;
        config.HiddenSize = C;
        config.IntermediateSize = 2048;
        config.VocabSize = V;
        config.NumQueryHeads = 8;
        config.NumKeyValHeads = 2;
        config.NumLayers = 1;
        config.MaxPositionEmbeddings = 32;

        auto requirements = dsl::RuntimeRunStateRequirements::embedding();
        requirements.encoder_backward_scratch = true;
        dsl::DslRunState state(config, {}, RuntimeOptions{}, 1, T, allocator,
                              false, false, 1 << 20, nullptr, nullptr, requirements);
        auto& scratch = state.scratch();
        // The production scale buffer is eight bytes. The odd-length loss buffer
        // also exercises padding beyond the usual multiple-of-16 training shapes.
        scratch.matmul_scales = allocator->allocate(ETensorDType::FP32, "scales", {2});
        scratch.cross_entropy_dloss = allocator->allocate(ETensorDType::FP32, "dloss", {T});
        const std::vector<float> scales{0.5f, 2.0f}, dloss(T, 0.25f);
        REQUIRE(cudaMemcpy(scratch.matmul_scales.Data, scales.data(), scales.size() * sizeof(float),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(scratch.cross_entropy_dloss.Data, dloss.data(), dloss.size() * sizeof(float),
                           cudaMemcpyHostToDevice) == cudaSuccess);

        const auto bytes = state.non_graph_persistent_extras_bytes();
        auto arena = allocator->allocate(ETensorDType::BYTE, "extras", {static_cast<long>(bytes + 256)});
        REQUIRE(cudaMemset(arena.Data, 0x7b, arena.bytes()) == cudaSuccess);
        state.rebind_non_graph_persistent_to_arena(arena.Data, bytes, state.MainStream);
        std::byte* end = arena.Data;
        for (const Tensor* tensor : {&scratch.matmul_scales, &scratch.cross_entropy_dloss,
                                    &scratch.encoder_bwd_scratch}) {
            REQUIRE(reinterpret_cast<std::uintptr_t>(tensor->Data) % 256 == 0);
            REQUIRE(tensor->Data >= end);
            end = tensor->Data + tensor->bytes();
            REQUIRE(end <= arena.Data + bytes);
        }
        for (const auto& [tensor, expected] :
             {std::pair{&scratch.matmul_scales, scales}, std::pair{&scratch.cross_entropy_dloss, dloss}}) {
            std::vector<float> actual(expected.size());
            REQUIRE(cudaMemcpy(actual.data(), tensor->Data, tensor->bytes(), cudaMemcpyDeviceToHost) == cudaSuccess);
            REQUIRE(actual == expected);
        }

        auto inputs = allocator->allocate(ETensorDType::INT32, "tokens", {1, T});
        auto inputs_cpu = allocator->allocate(ETensorDType::INT32, "host_tokens", EAllocationType::PINNED, {1, T});
        auto upstream = allocator->allocate(ETensorDType::BF16, "upstream", {1, T, C});
        auto gradient = allocator->allocate(ETensorDType::BF16, "gradient", {V, C});
        std::vector<int> tokens(T);
        std::vector<nv_bfloat16> values(T * C), expected(V * C, nv_bfloat16(0.0f));
        for (int t = 0; t < T; ++t) {
            tokens[t] = (t * 3) % 5;
            for (int c = 0; c < C; ++c) {
                values[t * C + c] = nv_bfloat16((t % 3 + c % 7 - 3) * 0.0625f);
                const auto i = tokens[t] * C + c;
                expected[i] = nv_bfloat16(float(expected[i]) + float(values[t * C + c]));
            }
        }
        std::copy(tokens.begin(), tokens.end(), inputs_cpu.get<int>());
        REQUIRE(cudaMemcpy(inputs.Data, tokens.data(), inputs.bytes(), cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(upstream.Data, values.data(), upstream.bytes(), cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemset(gradient.Data, 0, gradient.bytes()) == cudaSuccess);
        for (int step = 1; step <= 2; ++step) {
            encoder_backward(gradient, scratch.encoder_bwd_scratch, scratch.encoder_bwd_indices,
                             scratch.encoder_bwd_info, upstream, inputs, inputs_cpu, 1, T, C, 42,
                             state.MainStream, state.side_stream_event(), state.side_stream());
            REQUIRE(cudaStreamSynchronize(state.MainStream) == cudaSuccess);
            std::vector<nv_bfloat16> actual(V * C);
            REQUIRE(cudaMemcpy(actual.data(), gradient.Data, gradient.bytes(), cudaMemcpyDeviceToHost) == cudaSuccess);
            for (int i = 0; i < V * C; ++i) REQUIRE(float(actual[i]) == step * float(expected[i]));
        }
        std::vector<unsigned char> guard(256);
        REQUIRE(cudaMemcpy(guard.data(), arena.Data + bytes, guard.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(std::all_of(guard.begin(), guard.end(), [](auto value) { return value == 0x7b; }));
    }
}
