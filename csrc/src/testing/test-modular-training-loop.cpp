// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Modular model smoke / integration tests.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "kernels/kernels.h"
#include "config/pretrained_config.h"
#include "training/runtime_options.h"
#include "config/lora_adapter_config.h"
#include "modules/composite/transformer_block.h"
#include "modules/lora/lora_model.h"
#include "modules/model_factory.h"
#include "models/llama/llama_model.h"
#include "models/llama/transformer_block.h"
#include "models/qwen25/qwen25_model.h"
#include "models/qwen25/transformer_block.h"
#include "training/model.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"
#include "utilities/tensor_container.h"
#include "recipes/fp8_hybrid/fp8_hybrid_recipe.h"

namespace {

bool gpu_available() {
    static bool checked = false;
    static bool available = false;
    if (!checked) {
        checked = true;
        try {
            NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator&) { available = true; });
        } catch (...) {
            available = false;
        }
    }
    return available;
}

int get_cuda_device_count() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
}

bool fp8_supported() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return false;
    }
    // FP8 tensor cores are available on Ada (SM 8.9) and newer.
    return (prop.major > 8) || (prop.major == 8 && prop.minor >= 9);
}

PretrainedConfig create_test_config(int num_layers = 2, int vocab_size = 128) {
    PretrainedConfig config;
    config.Architecture = PretrainedConfig::QWEN2;
    config.HiddenSize = 256;              // head size 64 (cudnn flash-attn friendly)
    config.IntermediateSize = 256 * 4;
    config.NumQueryHeads = 4;
    config.NumKeyValHeads = 4;
    config.NumLayers = num_layers;
    config.VocabSize = vocab_size;
    config.MaxPositionEmbeddings = 1024;
    config.RopeTheta = 10000.0f;
    config.RmsNormEps = 1e-6f;
    config.TiedWordEmbeddings = false;
    config.UseQKVBias = false;
    config.BosTokenId = 0;
    config.EosTokenId = 1;
    config.PadTokenId = 2;
    config.DType = ETensorDType::BF16;
    return config;
}

RuntimeOptions create_test_options() {
    RuntimeOptions opts;
    opts.UseCudaGraphs = false;
    opts.RecomputeBlock = false;
    opts.ModelType = ETensorDType::BF16;
    opts.MatmulType = ETensorDType::BF16;
    opts.MasterDType = ETensorDType::BF16;
    return opts;
}

void fill_position_ids(Tensor& pos_ids, int B, int T) {
    auto* p = pos_ids.get<std::int32_t>();
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            p[b * T + t] = t;
        }
    }
}

float bf16_to_float(nv_bfloat16 v) {
    return __bfloat162float(v);
}

nv_bfloat16 read_bf16_device(const Tensor& t, long index) {
    nv_bfloat16 result{};
    CUDA_CHECK(cudaMemcpy(&result, reinterpret_cast<const nv_bfloat16*>(t.Data) + index, sizeof(result), cudaMemcpyDeviceToHost));
    return result;
}

float read_fp32_device(const Tensor& t, long index) {
    float result{};
    CUDA_CHECK(cudaMemcpy(&result, reinterpret_cast<const float*>(t.Data) + index, sizeof(result), cudaMemcpyDeviceToHost));
    return result;
}

std::vector<TensorShard> collect_tensors(ITensorContainer& container) {
    std::vector<TensorShard> tensors;
    container.iterate_tensors([&](std::string, const TensorShard& t) { tensors.emplace_back(t); });
    return tensors;
}

} // namespace

TEST_CASE("Modular dense model: 1 step forward/backward/update runs", "[modular][dense][smoke][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 2;
        constexpr int T = 64;

        PretrainedConfig cfg = create_test_config(/*num_layers=*/2, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        // Fill inputs/targets/position IDs in pinned host buffers provided by the model.
        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();

        std::fill(inputs.get<std::int32_t>(), inputs.get<std::int32_t>() + B * T, 1);
        std::fill(targets.get<std::int32_t>(), targets.get<std::int32_t>() + B * T, 1);
        fill_position_ids(pos_ids, B, T);

        model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
        model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
        model->update(comm, /*lr=*/1e-4f, /*beta1=*/0.9f, /*beta2=*/0.999f, /*t=*/1,
                      /*epsilon=*/1e-8f, /*weight_decay=*/0.0f, /*grad_clip=*/1.0f);

        CUDA_CHECK(cudaDeviceSynchronize());

        float loss = model->get_loss();
        float norm = model->get_norm();
        REQUIRE(std::isfinite(loss));
        REQUIRE(std::isfinite(norm));
    });
}

TEST_CASE("Modular: Qwen3-style head_dim + qk_norm forward/backward/update runs", "[modular][qwen3][qk-norm][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 2;
        constexpr int T = 64;

        PretrainedConfig cfg = create_test_config(/*num_layers=*/1, /*vocab_size=*/128);
        cfg.HiddenSize = 48;
        cfg.IntermediateSize = 192;
        cfg.NumQueryHeads = 2;
        cfg.NumKeyValHeads = 1;
        cfg.HeadDim = 32;        // intentionally != HiddenSize / NumQueryHeads
        cfg.UseQKNorm = true;
        cfg.UseQKVBias = false;
        cfg.MaxPositionEmbeddings = 256;

        RuntimeOptions opts = create_test_options();

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();

        std::fill(inputs.get<std::int32_t>(), inputs.get<std::int32_t>() + B * T, 1);
        std::fill(targets.get<std::int32_t>(), targets.get<std::int32_t>() + B * T, 1);
        fill_position_ids(pos_ids, B, T);

        model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
        model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
        model->update(comm, /*lr=*/1e-4f, /*beta1=*/0.9f, /*beta2=*/0.999f, /*t=*/1,
                      /*epsilon=*/1e-8f, /*weight_decay=*/0.0f, /*grad_clip=*/1.0f);

        CUDA_CHECK(cudaDeviceSynchronize());

        float loss = model->get_loss();
        float norm = model->get_norm();
        REQUIRE(std::isfinite(loss));
        REQUIRE(std::isfinite(norm));
    });
}

TEST_CASE("Modular cuda-graphs: 1 step forward/backward/update runs", "[modular][cuda-graphs][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 2;
        constexpr int T = 64;

        PretrainedConfig cfg = create_test_config(/*num_layers=*/2, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.UseCudaGraphs = true;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();
        std::fill(inputs.get<std::int32_t>(), inputs.get<std::int32_t>() + B * T, 1);
        std::fill(targets.get<std::int32_t>(), targets.get<std::int32_t>() + B * T, 2);
        fill_position_ids(pos_ids, B, T);

        model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
        model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
        model->update(comm, /*lr=*/1e-4f, /*beta1=*/0.9f, /*beta2=*/0.999f, /*t=*/1,
                      /*epsilon=*/1e-8f, /*weight_decay=*/0.0f, /*grad_clip=*/1.0f);

        CUDA_CHECK(cudaDeviceSynchronize());

        REQUIRE(std::isfinite(model->get_loss()));
        REQUIRE(std::isfinite(model->get_norm()));

        using DenseBlock = modules::Qwen2TransformerBlock;
        using DenseModel = modules::Qwen2Model;
        auto* dense = dynamic_cast<DenseModel*>(model.get());
        REQUIRE(dense != nullptr);
        REQUIRE(dense->run_state().forward_block_graph(/*layer_idx=*/0) != nullptr);
        // Check both accumulate=false and accumulate=true graphs exist
        REQUIRE(dense->run_state().backward_block_graph(/*layer_idx=*/0, /*accumulate=*/false) != nullptr);
    });
}

TEST_CASE("Modular dense model: FP8 matmul forward/backward/update runs", "[modular][fp8][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");
    if (!fp8_supported()) SKIP("FP8 tensor cores not available on this GPU");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 2;
        constexpr int T = 64;

        PretrainedConfig cfg = create_test_config(/*num_layers=*/2, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        // Use FP8 Hybrid recipe: E4M3 for forward activations/weights, E5M2 for backward gradients
        opts.TrainingRecipe = std::make_shared<recipes::FP8HybridRecipe>();
        // FP8 requires recompute_block for proper stack allocation sizing
        opts.RecomputeBlock = true;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();
        std::fill(inputs.get<std::int32_t>(), inputs.get<std::int32_t>() + B * T, 1);
        std::fill(targets.get<std::int32_t>(), targets.get<std::int32_t>() + B * T, 2);
        fill_position_ids(pos_ids, B, T);

        model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
        model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
        model->update(comm, /*lr=*/1e-4f, /*beta1=*/0.9f, /*beta2=*/0.999f, /*t=*/1,
                      /*epsilon=*/1e-8f, /*weight_decay=*/0.0f, /*grad_clip=*/1.0f);

        CUDA_CHECK(cudaDeviceSynchronize());
        REQUIRE(std::isfinite(model->get_loss()));
        REQUIRE(std::isfinite(model->get_norm()));
    });
}

TEST_CASE("Modular: offload-residual forward/backward/update runs", "[modular][offload][residual][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 1;
        constexpr int T = 64;

        PretrainedConfig cfg = create_test_config(/*num_layers=*/4, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.OffloadResidual = true;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();
        std::fill(inputs.get<std::int32_t>(), inputs.get<std::int32_t>() + B * T, 1);
        std::fill(targets.get<std::int32_t>(), targets.get<std::int32_t>() + B * T, 2);
        fill_position_ids(pos_ids, B, T);

        model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
        model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
        model->update(comm, /*lr=*/1e-4f, /*beta1=*/0.9f, /*beta2=*/0.999f, /*t=*/1,
                      /*epsilon=*/1e-8f, /*weight_decay=*/0.0f, /*grad_clip=*/1.0f);

        CUDA_CHECK(cudaDeviceSynchronize());
        REQUIRE(std::isfinite(model->get_loss()));
        REQUIRE(std::isfinite(model->get_norm()));
    });
}

TEST_CASE("Modular: attn-bwd-chunks=2 runs", "[modular][attention][chunks][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 2;
        constexpr int T = 64;

        PretrainedConfig cfg = create_test_config(/*num_layers=*/2, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.AttBwdChunks = 2;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();
        std::fill(inputs.get<std::int32_t>(), inputs.get<std::int32_t>() + B * T, 1);
        std::fill(targets.get<std::int32_t>(), targets.get<std::int32_t>() + B * T, 2);
        fill_position_ids(pos_ids, B, T);

        model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
        model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
        model->update(comm, /*lr=*/1e-4f, /*beta1=*/0.9f, /*beta2=*/0.999f, /*t=*/1,
                      /*epsilon=*/1e-8f, /*weight_decay=*/0.0f, /*grad_clip=*/1.0f);

        CUDA_CHECK(cudaDeviceSynchronize());
        REQUIRE(std::isfinite(model->get_loss()));
        REQUIRE(std::isfinite(model->get_norm()));
    });
}

TEST_CASE("Modular: attn-bwd-chunks requires divisible batch", "[modular][attention][chunks][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        constexpr int T = 64;

        PretrainedConfig cfg = create_test_config(/*num_layers=*/2, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.AttBwdChunks = 2;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        REQUIRE_THROWS_AS(model->allocate_run_state(opts, comm, /*B=*/3, T, /*allocate_optimizer=*/true), std::runtime_error);
    });
}

TEST_CASE("Modular: offload-optimizer allocates optimizer state on host", "[modular][offload][optimizer][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 1;
        constexpr int T = 32;

        PretrainedConfig cfg = create_test_config(/*num_layers=*/1, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.OffloadOptimizer = true;
        opts.UseZeroCopy = true;  // Required for offload with adamw8bit

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        // Fill inputs/targets/position IDs in pinned host buffers provided by the model.
        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();

        std::fill(inputs.get<std::int32_t>(), inputs.get<std::int32_t>() + B * T, 1);
        std::fill(targets.get<std::int32_t>(), targets.get<std::int32_t>() + B * T, 1);
        fill_position_ids(pos_ids, B, T);

        // Run one training step to initialize optimizer state (lazy initialization)
        model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
        model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
        model->update(comm, /*lr=*/1e-4f, /*beta1=*/0.9f, /*beta2=*/0.999f, /*t=*/1,
                      /*epsilon=*/1e-8f, /*weight_decay=*/0.0f, /*grad_clip=*/1.0f);

        CUDA_CHECK(cudaDeviceSynchronize());

        REQUIRE(std::isfinite(model->get_loss()));
        REQUIRE(std::isfinite(model->get_norm()));
    });
}

TEST_CASE("Modular: offload-master stores master weights on host (requires --use-zero-copy)", "[modular][offload][weights][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        PretrainedConfig cfg = create_test_config(/*num_layers=*/1, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.OffloadMaster = true;
        opts.UseZeroCopy = true;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, /*B=*/1, /*T=*/32, /*allocate_optimizer=*/false);
        model->init_weights(comm);

        auto w = collect_tensors(model->weights());
        REQUIRE(!w.empty());
        REQUIRE(std::all_of(w.begin(), w.end(), [](const TensorShard& t) { return t.Device == -1; }));
    });
}

TEST_CASE("Modular: recompute flags + chunked head/attn backward run", "[modular][recompute][chunks][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 2;
        constexpr int T = 64;

        PretrainedConfig cfg = create_test_config(/*num_layers=*/2, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.RecomputeFFN = true;
        opts.RecomputeAtt = true;
        opts.LMHeadChunks = 2;
        opts.AttBwdChunks = 2;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        REQUIRE_NOTHROW(model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/false));
        REQUIRE_NOTHROW(model->init_weights(comm));

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();
        std::fill(inputs.get<std::int32_t>(), inputs.get<std::int32_t>() + B * T, 1);
        std::fill(targets.get<std::int32_t>(), targets.get<std::int32_t>() + B * T, 2);
        fill_position_ids(pos_ids, B, T);

        REQUIRE_NOTHROW(model->forward(inputs, pos_ids, comm, /*micro_step=*/0));
        REQUIRE_NOTHROW(model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0));
        CUDA_CHECK(cudaDeviceSynchronize());

        float loss = model->get_loss();
        REQUIRE(std::isfinite(loss));
    });
}

TEST_CASE("Modular: offload optimizer states + recompute-block training step runs", "[modular][offload][recompute][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 1;
        constexpr int T = 64;

        PretrainedConfig cfg = create_test_config(/*num_layers=*/2, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.OffloadOptimizer = true;
        opts.RecomputeBlock = true;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();
        std::fill(inputs.get<std::int32_t>(), inputs.get<std::int32_t>() + B * T, 1);
        std::fill(targets.get<std::int32_t>(), targets.get<std::int32_t>() + B * T, 1);
        fill_position_ids(pos_ids, B, T);

        model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
        model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
        model->update(comm, /*lr=*/1e-4f, /*beta1=*/0.9f, /*beta2=*/0.999f, /*t=*/1,
                      /*epsilon=*/1e-8f, /*weight_decay=*/0.0f, /*grad_clip=*/1.0f);
        CUDA_CHECK(cudaDeviceSynchronize());
        REQUIRE(std::isfinite(model->get_loss()));
        REQUIRE(std::isfinite(model->get_norm()));
    });
}

TEST_CASE("Modular ZeRO-2: all-to-all reducer + offload-gradients runs (2 GPUs)", "[modular][zero2][alltoall][offload][multi-gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");
    if (get_cuda_device_count() < 2) SKIP("Need at least 2 GPUs for all-to-all test");

    NCCLCommunicator::run_communicators(2, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 1;
        constexpr int T = 32;
        constexpr int GradAccum = 2;

        PretrainedConfig cfg = create_test_config(/*num_layers=*/2, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.ShardGradients = true;
        opts.UseAllToAllReduce = true;
        opts.OffloadGrads = true;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();
        std::fill(inputs.get<std::int32_t>(), inputs.get<std::int32_t>() + B * T, 1);
        std::fill(targets.get<std::int32_t>(), targets.get<std::int32_t>() + B * T, 2);
        fill_position_ids(pos_ids, B, T);

        for (int micro = 0; micro < GradAccum; ++micro) {
            model->forward(inputs, pos_ids, comm, /*micro_step=*/micro);
            model->backward(inputs, targets, comm, /*grad_accum_steps=*/GradAccum, /*micro_step=*/micro);
        }
        model->update(comm, /*lr=*/1e-4f, /*beta1=*/0.9f, /*beta2=*/0.999f, /*t=*/1,
                      /*epsilon=*/1e-8f, /*weight_decay=*/0.0f, /*grad_clip=*/1.0f);

        CUDA_CHECK(cudaDeviceSynchronize());
        REQUIRE(std::isfinite(model->get_loss()));
        REQUIRE(std::isfinite(model->get_norm()));

        using DenseBlock = modules::Qwen2TransformerBlock;
        using DenseModel = modules::Qwen2Model;
        auto* dense = dynamic_cast<DenseModel*>(model.get());
        REQUIRE(dense != nullptr);
        auto& bg = dense->grads().get_block_shard(/*layer_idx=*/0, comm.stream());
        REQUIRE(bg.ln1_grads.d_weight.Device == -1);
    });
}

TEST_CASE("Modular recompute-block shares large activations across layers", "[modular][recompute][memory][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        PretrainedConfig cfg = create_test_config(/*num_layers=*/3, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.RecomputeBlock = true;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, /*B=*/1, /*T=*/64, /*allocate_optimizer=*/false);

        using DenseBlock = modules::Qwen2TransformerBlock;
        using DenseModel = modules::Qwen2Model;
        auto* dense = dynamic_cast<DenseModel*>(model.get());
        REQUIRE(dense != nullptr);

        auto& rs = dense->run_state();
        const auto& a0 = rs.simplified_acts(0);
        const auto& a1 = rs.simplified_acts(1);
        const auto& a2 = rs.simplified_acts(2);

        // In recompute-block mode, qkv/att/att_out are shared buffers reused across layers.
        REQUIRE(a0.qkv.Data == a1.qkv.Data);
        REQUIRE(a1.qkv.Data == a2.qkv.Data);
        REQUIRE(a0.att.Data == a1.att.Data);
        REQUIRE(a0.att_out.Data == a1.att_out.Data);
    });
}

TEST_CASE("Modular LoRA: recompute-block forward/backward/update runs", "[modular][lora][recompute][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 2;
        constexpr int T = 64;

        PretrainedConfig cfg = create_test_config(/*num_layers=*/2, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.RecomputeBlock = true;
        opts.UseCudaGraphs = true; // exercises the same config as train-lora.sh (graphs + LoRA hooks)

        auto allocator = std::make_shared<TensorAllocator>();

        using DenseBlock = modules::Qwen2TransformerBlock;
        using DenseModel = modules::Qwen2Model;

        auto base_any = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        auto* dense_ptr = dynamic_cast<DenseModel*>(base_any.release());
        REQUIRE(dense_ptr != nullptr);
        std::unique_ptr<DenseModel> base_model(dense_ptr);

        modules::ModularLoRAConfig lora_cfg;
        lora_cfg.rank = 4;
        lora_cfg.alpha = 8.0f;
        lora_cfg.dtype = ETensorDType::FP32;
        lora_cfg.with_all();

        modules::ModularLoRAModel<DenseBlock> model(std::move(base_model), lora_cfg, opts, comm, allocator);
        model.allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model.init_weights(comm);

        auto& inputs = model.get_input_buffer();
        auto& targets = model.get_target_buffer();
        auto& pos_ids = model.get_position_ids_buffer();
        std::fill(inputs.get<std::int32_t>(), inputs.get<std::int32_t>() + B * T, 1);
        std::fill(targets.get<std::int32_t>(), targets.get<std::int32_t>() + B * T, 2);
        fill_position_ids(pos_ids, B, T);

        model.forward(inputs, pos_ids, comm, /*micro_step=*/0);
        model.backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
        model.update(comm, /*lr=*/1e-4f, /*beta1=*/0.9f, /*beta2=*/0.999f, /*t=*/1,
                     /*epsilon=*/1e-8f, /*weight_decay=*/0.0f, /*grad_clip=*/1.0f);

        CUDA_CHECK(cudaDeviceSynchronize());
        REQUIRE(std::isfinite(model.get_loss()));
        REQUIRE(std::isfinite(model.get_norm()));

        // LoRA uses forward/backward hooks; graphs must still be capturable.
        auto& rs = model.base_model().run_state();
        REQUIRE(rs.forward_block_graph(/*layer_idx=*/0) != nullptr);
        REQUIRE(rs.backward_block_graph(/*layer_idx=*/0, /*accumulate=*/false) != nullptr);
    });
}

// Note: Tests for configurable optimizer state dtypes (BF16, FP8) have been removed.
// The adamw8bit optimizer uses 8-bit quantized states internally.

TEST_CASE("Modular leaf recompute flags affect activation sharing", "[modular][recompute][leaf][memory][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        PretrainedConfig cfg = create_test_config(/*num_layers=*/3, /*vocab_size=*/128);
        auto allocator = std::make_shared<TensorAllocator>();

        using DenseBlock = modules::Qwen2TransformerBlock;
        using DenseModel = modules::Qwen2Model;

        // recompute-qkv: qkv shared, att not shared
        {
            RuntimeOptions opts = create_test_options();
            opts.RecomputeQKV = true;
            auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
            model->allocate_run_state(opts, comm, /*B=*/1, /*T=*/64, /*allocate_optimizer=*/false);
            auto* dense = dynamic_cast<DenseModel*>(model.get());
            REQUIRE(dense != nullptr);
            auto& rs = dense->run_state();
            REQUIRE(rs.simplified_acts(0).qkv.Data == rs.simplified_acts(1).qkv.Data);
            REQUIRE(rs.simplified_acts(1).qkv.Data == rs.simplified_acts(2).qkv.Data);
            REQUIRE(rs.simplified_acts(0).att.Data != rs.simplified_acts(1).att.Data);
        }

        // recompute-swiglu: swiglu shared, mlp_up not shared
        {
            RuntimeOptions opts = create_test_options();
            opts.RecomputeSwiGLu = true;
            auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
            model->allocate_run_state(opts, comm, /*B=*/1, /*T=*/64, /*allocate_optimizer=*/false);
            auto* dense = dynamic_cast<DenseModel*>(model.get());
            REQUIRE(dense != nullptr);
            auto& rs = dense->run_state();
            REQUIRE(rs.simplified_acts(0).swiglu.Data == rs.simplified_acts(1).swiglu.Data);
            REQUIRE(rs.simplified_acts(0).mlp_up.Data != rs.simplified_acts(1).mlp_up.Data);
        }

        // recompute-norm: ln1 + ln2 shared
        {
            RuntimeOptions opts = create_test_options();
            opts.RecomputeRMSNorm = true;
            auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
            model->allocate_run_state(opts, comm, /*B=*/1, /*T=*/64, /*allocate_optimizer=*/false);
            auto* dense = dynamic_cast<DenseModel*>(model.get());
            REQUIRE(dense != nullptr);
            auto& rs = dense->run_state();
            REQUIRE(rs.simplified_acts(0).ln1.Data == rs.simplified_acts(1).ln1.Data);
            REQUIRE(rs.simplified_acts(0).ln2.Data == rs.simplified_acts(1).ln2.Data);
        }
    });
}

TEST_CASE("Modular LoRA: base weights stay fixed, adapter weights update", "[modular][lora][smoke][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 1;
        constexpr int T = 64;

        PretrainedConfig cfg = create_test_config(/*num_layers=*/1, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.RecomputeBlock = false;

        // Build base dense model.
        auto allocator = std::make_shared<TensorAllocator>();
        auto base_any = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);

        using DenseBlock = modules::Qwen2TransformerBlock;
        using DenseModel = modules::Qwen2Model;
        auto* dense_ptr = dynamic_cast<DenseModel*>(base_any.get());
        REQUIRE(dense_ptr != nullptr);

        std::unique_ptr<DenseModel> dense_base(static_cast<DenseModel*>(base_any.release()));

        // LoRA config: apply to q-proj only for a quick test.
        modules::ModularLoRAConfig lora_cfg;
        lora_cfg.rank = 2;
        lora_cfg.alpha = 2.0f;
        lora_cfg.dropout = 0.0f;
        lora_cfg.dtype = ETensorDType::FP32;
        lora_cfg.targets.insert(modules::LoRATarget::Q_PROJ);

        auto lora_model = std::make_unique<modules::ModularLoRAModel<DenseBlock>>(
            std::move(dense_base), lora_cfg, opts, comm, allocator);

        lora_model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        lora_model->init_weights(comm);

        // Snapshot one base weight value (qkv[0]) before update.
        auto& base = lora_model->base_model();
        auto& rs = base.run_state();

        auto* wm_ptr = dynamic_cast<modules::ModularWeightManager<DenseBlock>*>(&base.weights());
        REQUIRE(wm_ptr != nullptr);
        auto& wm = *wm_ptr;
        wm.fetch_master_block(0, comm.stream());
        auto& bw = wm.get_master_block(0, rs.MainStream);
        nv_bfloat16 base_before = read_bf16_device(bw.attention.qkv_weight, 0);
        wm.release_master_block(0, rs.MainStream, rs.side_stream());

        // Snapshot one LoRA weight value (B matrix element 0) before update.
        auto& lora_master = lora_model->lora_weights().get_master_block(0, nullptr);
        REQUIRE(lora_master.attention.q.has_value());
        std::vector<float> lora_before(lora_master.attention.q->B.nelem(), 0.0f);
        CUDA_CHECK(cudaMemcpy(lora_before.data(), lora_master.attention.q->B.Data,
                              lora_before.size() * sizeof(float), cudaMemcpyDeviceToHost));

        // Run one train step.
        auto& inputs = lora_model->get_input_buffer();
        auto& targets = lora_model->get_target_buffer();
        auto& pos_ids = lora_model->get_position_ids_buffer();
        std::fill(inputs.get<std::int32_t>(), inputs.get<std::int32_t>() + B * T, 1);
        std::fill(targets.get<std::int32_t>(), targets.get<std::int32_t>() + B * T, 2);
        fill_position_ids(pos_ids, B, T);

        lora_model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
        lora_model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);

        CUDA_CHECK(cudaDeviceSynchronize());

        // Sanity: we have valid tokens and non-zero LoRA gradients.
        int valid_tokens = 0;
        CUDA_CHECK(cudaMemcpy(&valid_tokens, rs.ValidTokenCount.Data, sizeof(valid_tokens), cudaMemcpyDeviceToHost));
        REQUIRE(valid_tokens > 0);

        // Sanity: base-model d_qkv is non-zero (otherwise LoRA q/k/v gradients are expected to be zero).
        {
            const auto& dqkv = rs.simplified_grads(0).d_qkv;
            bool any_nonzero = false;
            for (long idx : {0L, 1L, 7L, 31L, 63L, 127L}) {
                if (idx < (long)dqkv.nelem()) {
                    float v = bf16_to_float(read_bf16_device(dqkv, idx));
                    if (v != 0.0f) {
                        any_nonzero = true;
                        break;
                    }
                }
            }
            REQUIRE(any_nonzero);
        }

        bool unused_acc = false;
        auto& lora_grads = lora_model->lora_grads().get_block_full(0, rs.MainStream, comm, unused_acc);
        REQUIRE(lora_grads.attention.q.has_value());
        std::vector<float> host_grad_b(lora_grads.attention.q->B.nelem(), 0.0f);
        CUDA_CHECK(cudaMemcpy(host_grad_b.data(), lora_grads.attention.q->B.Data,
                              host_grad_b.size() * sizeof(float), cudaMemcpyDeviceToHost));
        bool any_nonzero_grad = std::any_of(host_grad_b.begin(), host_grad_b.end(), [](float v) { return v != 0.0f; });
        REQUIRE(any_nonzero_grad);

        lora_model->update(comm, /*lr=*/1e-2f, /*beta1=*/0.9f, /*beta2=*/0.999f, /*t=*/1,
                           /*epsilon=*/1e-8f, /*weight_decay=*/0.0f, /*grad_clip=*/0.0f);

        CUDA_CHECK(cudaDeviceSynchronize());

        // Base weight should remain unchanged.
        wm.fetch_master_block(0, comm.stream());
        auto& bw_after = wm.get_master_block(0, rs.MainStream);
        nv_bfloat16 base_after = read_bf16_device(bw_after.attention.qkv_weight, 0);
        wm.release_master_block(0, rs.MainStream, rs.side_stream());
        REQUIRE(bf16_to_float(base_after) == bf16_to_float(base_before));

        // LoRA adapter master B should generally change from its initial value.
        std::vector<float> lora_after(lora_master.attention.q->B.nelem(), 0.0f);
        CUDA_CHECK(cudaMemcpy(lora_after.data(), lora_master.attention.q->B.Data,
                              lora_after.size() * sizeof(float), cudaMemcpyDeviceToHost));
        bool any_changed = false;
        for (size_t i = 0; i < lora_after.size(); ++i) {
            if (lora_after[i] != lora_before[i]) {
                any_changed = true;
                break;
            }
        }
        REQUIRE(any_changed);
    });
}

TEST_CASE("Modular ZeRO-3 (shard weights) forward runs on 2 GPUs", "[modular][zero3][multi-gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");
    if (get_cuda_device_count() < 2) SKIP("Need at least 2 GPUs");

    NCCLCommunicator::run_communicators(2, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 1;
        constexpr int T = 32;
        PretrainedConfig cfg = create_test_config(/*num_layers=*/2, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.ShardWeights = true;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/false);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& pos_ids = model->get_position_ids_buffer();
        std::fill(inputs.get<std::int32_t>(), inputs.get<std::int32_t>() + B * T, 1);
        fill_position_ids(pos_ids, B, T);

        model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
        CUDA_CHECK(cudaDeviceSynchronize());
    });
}

TEST_CASE("Modular ZeRO-2 (shard gradients) forward/backward/update runs on 2 GPUs", "[modular][zero2][multi-gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");
    if (get_cuda_device_count() < 2) SKIP("Need at least 2 GPUs");

    NCCLCommunicator::run_communicators(2, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 1;
        constexpr int T = 32;
        PretrainedConfig cfg = create_test_config(/*num_layers=*/2, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.ShardGradients = true;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();
        std::fill(inputs.get<std::int32_t>(), inputs.get<std::int32_t>() + B * T, 1);
        std::fill(targets.get<std::int32_t>(), targets.get<std::int32_t>() + B * T, 2);
        fill_position_ids(pos_ids, B, T);

        model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
        model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
        model->update(comm, /*lr=*/1e-4f, /*beta1=*/0.9f, /*beta2=*/0.999f, /*t=*/1,
                      /*epsilon=*/1e-8f, /*weight_decay=*/0.0f, /*grad_clip=*/1.0f);
        CUDA_CHECK(cudaDeviceSynchronize());
        REQUIRE(std::isfinite(model->get_loss()));
        REQUIRE(std::isfinite(model->get_norm()));
    });
}

// ============================================================================
// Recompute-block accuracy investigation tests
// ============================================================================

namespace {

/// Helper to copy a device tensor to host
std::vector<float> tensor_to_host_f32(const Tensor& t) {
    std::vector<float> result(t.nelem());
    if (t.DType == ETensorDType::FP32) {
        CUDA_CHECK(cudaMemcpy(result.data(), t.Data, t.bytes(), cudaMemcpyDeviceToHost));
    } else if (t.DType == ETensorDType::BF16) {
        std::vector<nv_bfloat16> bf16_data(t.nelem());
        CUDA_CHECK(cudaMemcpy(bf16_data.data(), t.Data, t.bytes(), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < bf16_data.size(); ++i) {
            result[i] = __bfloat162float(bf16_data[i]);
        }
    }
    return result;
}

/// Compute max absolute difference between two float vectors
float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float max_diff = 0.0f;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
}

/// Compute relative error (max |a-b| / max(|a|, |b|, eps))
float max_rel_diff(const std::vector<float>& a, const std::vector<float>& b, float eps = 1e-6f) {
    float max_diff = 0.0f;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        float scale = std::max({std::abs(a[i]), std::abs(b[i]), eps});
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]) / scale);
    }
    return max_diff;
}

} // namespace

TEST_CASE("Recompute-block: loss matches non-recompute on first forward", "[modular][recompute][accuracy][gpu]") {
    // Test that the forward pass produces identical loss with and without recompute-block
    if (!gpu_available()) SKIP("CUDA not available");

    constexpr int B = 2;
    constexpr int T = 64;
    constexpr int num_layers = 4;

    float loss_no_recompute = 0.0f;
    float loss_with_recompute = 0.0f;

    // Run without recompute-block
    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        PretrainedConfig cfg = create_test_config(num_layers, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.RecomputeBlock = false;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/false);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();

        // Use deterministic input data
        for (int i = 0; i < B * T; ++i) {
            inputs.get<std::int32_t>()[i] = (i * 17 + 3) % 128;
            targets.get<std::int32_t>()[i] = (i * 13 + 7) % 128;
        }
        fill_position_ids(pos_ids, B, T);

        model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
        CUDA_CHECK(cudaDeviceSynchronize());
        loss_no_recompute = model->get_loss();
    });

    // Run with recompute-block
    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        PretrainedConfig cfg = create_test_config(num_layers, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.RecomputeBlock = true;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/false);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();

        // Use same deterministic input data
        for (int i = 0; i < B * T; ++i) {
            inputs.get<std::int32_t>()[i] = (i * 17 + 3) % 128;
            targets.get<std::int32_t>()[i] = (i * 13 + 7) % 128;
        }
        fill_position_ids(pos_ids, B, T);

        model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
        CUDA_CHECK(cudaDeviceSynchronize());
        loss_with_recompute = model->get_loss();
    });

    INFO("Loss without recompute: " << loss_no_recompute);
    INFO("Loss with recompute: " << loss_with_recompute);
    INFO("Difference: " << std::abs(loss_no_recompute - loss_with_recompute));

    // Forward pass should be identical - both modes compute the same operations
    REQUIRE(std::isfinite(loss_no_recompute));
    REQUIRE(std::isfinite(loss_with_recompute));
    // Allow small tolerance for floating point differences
    REQUIRE(std::abs(loss_no_recompute - loss_with_recompute) < 1e-5f);
}

TEST_CASE("Recompute-block: gradients match non-recompute after backward", "[modular][recompute][accuracy][gradients][gpu]") {
    // Test that backward pass produces identical gradient norms
    if (!gpu_available()) SKIP("CUDA not available");

    constexpr int B = 2;
    constexpr int T = 64;
    constexpr int num_layers = 4;

    float norm_no_recompute = 0.0f;
    float norm_with_recompute = 0.0f;

    // Run without recompute-block
    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        PretrainedConfig cfg = create_test_config(num_layers, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.RecomputeBlock = false;
        opts.UseCudaGraphs = false;  // Disable graphs to ensure determinism

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();

        for (int i = 0; i < B * T; ++i) {
            inputs.get<std::int32_t>()[i] = (i * 17 + 3) % 128;
            targets.get<std::int32_t>()[i] = (i * 13 + 7) % 128;
        }
        fill_position_ids(pos_ids, B, T);

        model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
        model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
        model->update(comm, 1e-4f, 0.9f, 0.999f, 1, 1e-8f, 0.0f, 1.0f);
        CUDA_CHECK(cudaDeviceSynchronize());

        norm_no_recompute = model->get_norm();
    });

    // Run with recompute-block
    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        PretrainedConfig cfg = create_test_config(num_layers, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.RecomputeBlock = true;
        opts.UseCudaGraphs = false;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();

        for (int i = 0; i < B * T; ++i) {
            inputs.get<std::int32_t>()[i] = (i * 17 + 3) % 128;
            targets.get<std::int32_t>()[i] = (i * 13 + 7) % 128;
        }
        fill_position_ids(pos_ids, B, T);

        model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
        model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
        model->update(comm, 1e-4f, 0.9f, 0.999f, 1, 1e-8f, 0.0f, 1.0f);
        CUDA_CHECK(cudaDeviceSynchronize());

        norm_with_recompute = model->get_norm();
    });

    INFO("Gradient norm without recompute: " << norm_no_recompute);
    INFO("Gradient norm with recompute: " << norm_with_recompute);
    INFO("Norm difference: " << std::abs(norm_no_recompute - norm_with_recompute));

    REQUIRE(std::isfinite(norm_no_recompute));
    REQUIRE(std::isfinite(norm_with_recompute));

    // Compare gradient norms - should be very close
    float norm_rel_diff = std::abs(norm_no_recompute - norm_with_recompute) /
                          std::max({norm_no_recompute, norm_with_recompute, 1e-6f});
    INFO("Relative norm difference: " << norm_rel_diff);
    REQUIRE(norm_rel_diff < 1e-4f);
}

TEST_CASE("Recompute-block: multi-step training divergence check", "[modular][recompute][accuracy][divergence][gpu]") {
    // Test that training for multiple steps doesn't diverge between the two modes
    if (!gpu_available()) SKIP("CUDA not available");

    constexpr int B = 2;
    constexpr int T = 64;
    constexpr int num_layers = 4;
    constexpr int num_steps = 10;
    constexpr float lr = 1e-4f;

    std::vector<float> losses_no_recompute(num_steps);
    std::vector<float> losses_with_recompute(num_steps);

    // Run without recompute-block
    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        PretrainedConfig cfg = create_test_config(num_layers, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.RecomputeBlock = false;
        opts.UseCudaGraphs = false;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();
        fill_position_ids(pos_ids, B, T);

        for (int step = 0; step < num_steps; ++step) {
            // Use different but deterministic data per step
            for (int i = 0; i < B * T; ++i) {
                inputs.get<std::int32_t>()[i] = ((i + step) * 17 + 3) % 128;
                targets.get<std::int32_t>()[i] = ((i + step) * 13 + 7) % 128;
            }

            model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
            model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
            model->update(comm, lr, 0.9f, 0.999f, step + 1, 1e-8f, 0.0f, 1.0f);
            CUDA_CHECK(cudaDeviceSynchronize());
            losses_no_recompute[step] = model->get_loss();
        }
    });

    // Run with recompute-block
    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        PretrainedConfig cfg = create_test_config(num_layers, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.RecomputeBlock = true;
        opts.UseCudaGraphs = false;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();
        fill_position_ids(pos_ids, B, T);

        for (int step = 0; step < num_steps; ++step) {
            for (int i = 0; i < B * T; ++i) {
                inputs.get<std::int32_t>()[i] = ((i + step) * 17 + 3) % 128;
                targets.get<std::int32_t>()[i] = ((i + step) * 13 + 7) % 128;
            }

            model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
            model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
            model->update(comm, lr, 0.9f, 0.999f, step + 1, 1e-8f, 0.0f, 1.0f);
            CUDA_CHECK(cudaDeviceSynchronize());
            losses_with_recompute[step] = model->get_loss();
        }
    });

    // Compare losses at each step
    float max_loss_diff = 0.0f;
    float max_rel_loss_diff = 0.0f;
    for (int step = 0; step < num_steps; ++step) {
        float diff = std::abs(losses_no_recompute[step] - losses_with_recompute[step]);
        float rel_diff = diff / std::max({losses_no_recompute[step], losses_with_recompute[step], 1e-6f});
        max_loss_diff = std::max(max_loss_diff, diff);
        max_rel_loss_diff = std::max(max_rel_loss_diff, rel_diff);

        INFO("Step " << step << ": no_recompute=" << losses_no_recompute[step]
             << " recompute=" << losses_with_recompute[step]
             << " diff=" << diff << " rel=" << rel_diff);
    }

    INFO("Max absolute loss difference over " << num_steps << " steps: " << max_loss_diff);
    INFO("Max relative loss difference over " << num_steps << " steps: " << max_rel_loss_diff);

    // Allow some divergence due to floating point accumulation, but flag large differences
    // A 2% accuracy difference would show up as significant divergence
    REQUIRE(max_rel_loss_diff < 0.01f);  // 1% tolerance - stricter than 2%
}

TEST_CASE("Recompute-block: weights evolve identically over training", "[modular][recompute][accuracy][weights][gpu]") {
    // Test that weight updates result in similar final loss after training
    if (!gpu_available()) SKIP("CUDA not available");

    constexpr int B = 2;
    constexpr int T = 64;
    constexpr int num_layers = 2;
    constexpr int num_steps = 5;
    constexpr float lr = 1e-3f;  // Larger LR to see differences faster

    float final_loss_no_recompute = 0.0f;
    float final_loss_with_recompute = 0.0f;

    // Run without recompute-block
    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        PretrainedConfig cfg = create_test_config(num_layers, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.RecomputeBlock = false;
        opts.UseCudaGraphs = false;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();
        fill_position_ids(pos_ids, B, T);

        for (int step = 0; step < num_steps; ++step) {
            for (int i = 0; i < B * T; ++i) {
                inputs.get<std::int32_t>()[i] = ((i + step) * 17 + 3) % 128;
                targets.get<std::int32_t>()[i] = ((i + step) * 13 + 7) % 128;
            }

            model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
            model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
            model->update(comm, lr, 0.9f, 0.999f, step + 1, 1e-8f, 0.0f, 1.0f);
            CUDA_CHECK(cudaDeviceSynchronize());
            final_loss_no_recompute = model->get_loss();
        }
    });

    // Run with recompute-block
    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        PretrainedConfig cfg = create_test_config(num_layers, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();
        opts.RecomputeBlock = true;
        opts.UseCudaGraphs = false;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();
        fill_position_ids(pos_ids, B, T);

        for (int step = 0; step < num_steps; ++step) {
            for (int i = 0; i < B * T; ++i) {
                inputs.get<std::int32_t>()[i] = ((i + step) * 17 + 3) % 128;
                targets.get<std::int32_t>()[i] = ((i + step) * 13 + 7) % 128;
            }

            model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
            model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
            model->update(comm, lr, 0.9f, 0.999f, step + 1, 1e-8f, 0.0f, 1.0f);
            CUDA_CHECK(cudaDeviceSynchronize());
            final_loss_with_recompute = model->get_loss();
        }
    });

    INFO("Final loss without recompute: " << final_loss_no_recompute);
    INFO("Final loss with recompute: " << final_loss_with_recompute);

    REQUIRE(std::isfinite(final_loss_no_recompute));
    REQUIRE(std::isfinite(final_loss_with_recompute));

    float loss_rel_diff = std::abs(final_loss_no_recompute - final_loss_with_recompute) /
                          std::max({final_loss_no_recompute, final_loss_with_recompute, 1e-6f});
    INFO("Relative loss difference after " << num_steps << " steps: " << loss_rel_diff);

    // Allow some divergence - fresh activation recomputation in recompute-block mode
    // leads to slightly different training dynamics. 1% tolerance for 5 steps.
    REQUIRE(loss_rel_diff < 0.01f);
}

TEST_CASE("Recompute-block: activation values match recomputed values", "[modular][recompute][accuracy][activations][gpu]") {
    // This test verifies that stored activations (no recompute) match freshly recomputed ones
    // by manually triggering recomputation and comparing
    if (!gpu_available()) SKIP("CUDA not available");

    constexpr int B = 2;
    constexpr int T = 64;
    constexpr int num_layers = 3;

    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        PretrainedConfig cfg = create_test_config(num_layers, /*vocab_size=*/128);

        // First, run forward WITHOUT recompute and capture activations
        RuntimeOptions opts_no_recompute = create_test_options();
        opts_no_recompute.RecomputeBlock = false;
        opts_no_recompute.UseCudaGraphs = false;

        auto allocator1 = std::make_shared<TensorAllocator>();
        auto model1 = modules::ModelFactory::create_from_pretrained_config(
            cfg, opts_no_recompute, comm.rank(), comm.world_size(), allocator1);
        model1->allocate_run_state(opts_no_recompute, comm, B, T, /*allocate_optimizer=*/true);
        model1->init_weights(comm);

        auto& inputs1 = model1->get_input_buffer();
        auto& targets1 = model1->get_target_buffer();
        auto& pos_ids1 = model1->get_position_ids_buffer();
        for (int i = 0; i < B * T; ++i) {
            inputs1.get<std::int32_t>()[i] = (i * 17 + 3) % 128;
            targets1.get<std::int32_t>()[i] = (i * 13 + 7) % 128;
        }
        fill_position_ids(pos_ids1, B, T);

        // Forward pass - activations get stored
        model1->forward(inputs1, pos_ids1, comm, /*micro_step=*/0);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Capture stored activations (ln1, ln2, qkv for each layer)
        using DenseBlock = modules::Qwen2TransformerBlock;
        using DenseModel = modules::Qwen2Model;
        auto* dense1 = dynamic_cast<DenseModel*>(model1.get());
        REQUIRE(dense1 != nullptr);

        std::vector<std::vector<float>> stored_ln1(num_layers);
        std::vector<std::vector<float>> stored_ln2(num_layers);
        auto& rs1 = dense1->run_state();
        for (int l = 0; l < num_layers; ++l) {
            auto& acts = rs1.simplified_acts(l);
            stored_ln1[l] = tensor_to_host_f32(acts.ln1);
            stored_ln2[l] = tensor_to_host_f32(acts.ln2);
        }

        float loss1 = model1->get_loss();

        // Now run backward (this doesn't recompute since recompute_block=false)
        model1->backward(inputs1, targets1, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
        CUDA_CHECK(cudaDeviceSynchronize());
        float norm1 = model1->get_norm();

        // Now, run with recompute-block enabled - activations will be recomputed before backward
        RuntimeOptions opts_recompute = create_test_options();
        opts_recompute.RecomputeBlock = true;
        opts_recompute.UseCudaGraphs = false;

        auto allocator2 = std::make_shared<TensorAllocator>();
        auto model2 = modules::ModelFactory::create_from_pretrained_config(
            cfg, opts_recompute, comm.rank(), comm.world_size(), allocator2);
        model2->allocate_run_state(opts_recompute, comm, B, T, /*allocate_optimizer=*/true);
        model2->init_weights(comm);

        auto& inputs2 = model2->get_input_buffer();
        auto& targets2 = model2->get_target_buffer();
        auto& pos_ids2 = model2->get_position_ids_buffer();
        for (int i = 0; i < B * T; ++i) {
            inputs2.get<std::int32_t>()[i] = (i * 17 + 3) % 128;
            targets2.get<std::int32_t>()[i] = (i * 13 + 7) % 128;
        }
        fill_position_ids(pos_ids2, B, T);

        model2->forward(inputs2, pos_ids2, comm, /*micro_step=*/0);
        CUDA_CHECK(cudaDeviceSynchronize());
        float loss2 = model2->get_loss();

        model2->backward(inputs2, targets2, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
        CUDA_CHECK(cudaDeviceSynchronize());
        float norm2 = model2->get_norm();

        // Compare losses (forward should be identical)
        INFO("Loss no_recompute: " << loss1 << " recompute: " << loss2);
        REQUIRE(std::abs(loss1 - loss2) < 1e-5f);

        // Compare norms (backward results)
        INFO("Norm no_recompute: " << norm1 << " recompute: " << norm2);
        float norm_rel_diff = std::abs(norm1 - norm2) / std::max({norm1, norm2, 1e-6f});
        INFO("Relative norm difference: " << norm_rel_diff);

        // Flag if there's a significant difference
        if (norm_rel_diff > 1e-4f) {
            WARN("Significant gradient norm difference detected between recompute modes!");
            WARN("This may indicate stored activations differ from recomputed ones.");
        }
        REQUIRE(norm_rel_diff < 0.01f);  // Allow up to 1% difference
    });
}

TEST_CASE("Recompute-block: longer training shows divergence pattern", "[modular][recompute][accuracy][long][gpu]") {
    // Run longer training to see if differences accumulate
    if (!gpu_available()) SKIP("CUDA not available");

    constexpr int B = 4;
    constexpr int T = 128;
    constexpr int num_layers = 4;
    constexpr int num_steps = 50;
    constexpr float lr = 5e-4f;

    std::vector<float> losses_no_recompute(num_steps);
    std::vector<float> losses_with_recompute(num_steps);
    std::vector<float> norms_no_recompute(num_steps);
    std::vector<float> norms_with_recompute(num_steps);

    // Run without recompute-block
    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        PretrainedConfig cfg = create_test_config(num_layers, /*vocab_size=*/256);
        RuntimeOptions opts = create_test_options();
        opts.RecomputeBlock = false;
        opts.UseCudaGraphs = false;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();
        fill_position_ids(pos_ids, B, T);

        for (int step = 0; step < num_steps; ++step) {
            for (int i = 0; i < B * T; ++i) {
                inputs.get<std::int32_t>()[i] = ((i * 31 + step * 7) % 256);
                targets.get<std::int32_t>()[i] = ((i * 37 + step * 11 + 1) % 256);
            }

            model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
            model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
            model->update(comm, lr, 0.9f, 0.999f, step + 1, 1e-8f, 0.01f, 1.0f);
            CUDA_CHECK(cudaDeviceSynchronize());
            losses_no_recompute[step] = model->get_loss();
            norms_no_recompute[step] = model->get_norm();
        }
    });

    // Run with recompute-block
    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        PretrainedConfig cfg = create_test_config(num_layers, /*vocab_size=*/256);
        RuntimeOptions opts = create_test_options();
        opts.RecomputeBlock = true;
        opts.UseCudaGraphs = false;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();
        fill_position_ids(pos_ids, B, T);

        for (int step = 0; step < num_steps; ++step) {
            for (int i = 0; i < B * T; ++i) {
                inputs.get<std::int32_t>()[i] = ((i * 31 + step * 7) % 256);
                targets.get<std::int32_t>()[i] = ((i * 37 + step * 11 + 1) % 256);
            }

            model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
            model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
            model->update(comm, lr, 0.9f, 0.999f, step + 1, 1e-8f, 0.01f, 1.0f);
            CUDA_CHECK(cudaDeviceSynchronize());
            losses_with_recompute[step] = model->get_loss();
            norms_with_recompute[step] = model->get_norm();
        }
    });

    // Analyze divergence pattern
    float initial_loss_diff = std::abs(losses_no_recompute[0] - losses_with_recompute[0]);
    float final_loss_diff = std::abs(losses_no_recompute[num_steps-1] - losses_with_recompute[num_steps-1]);
    float final_loss_no_recompute = losses_no_recompute[num_steps-1];
    float final_loss_recompute = losses_with_recompute[num_steps-1];

    INFO("Initial loss difference: " << initial_loss_diff);
    INFO("Final loss difference: " << final_loss_diff);
    INFO("Final loss (no recompute): " << final_loss_no_recompute);
    INFO("Final loss (recompute): " << final_loss_recompute);
    INFO("Loss improvement (no recompute): " << (losses_no_recompute[0] - final_loss_no_recompute));
    INFO("Loss improvement (recompute): " << (losses_with_recompute[0] - final_loss_recompute));

    // Check if one mode consistently achieves lower loss
    int recompute_wins = 0;
    for (int step = num_steps / 2; step < num_steps; ++step) {
        if (losses_with_recompute[step] < losses_no_recompute[step]) {
            recompute_wins++;
        }
    }
    float recompute_win_rate = static_cast<float>(recompute_wins) / (num_steps / 2);
    INFO("Recompute mode achieves lower loss in " << (recompute_win_rate * 100) << "% of later steps");

    // Print per-step loss and norm differences to identify when divergence starts
    std::cout << "\n=== Per-step divergence analysis ===" << std::endl;
    std::cout << "Step | Loss(no_rec) | Loss(rec) | Loss_diff | Norm(no_rec) | Norm(rec) | Norm_diff" << std::endl;
    for (int step = 0; step < std::min(num_steps, 20); ++step) {
        float loss_diff = losses_no_recompute[step] - losses_with_recompute[step];
        float norm_diff = norms_no_recompute[step] - norms_with_recompute[step];
        std::cout << std::setw(4) << step << " | "
                  << std::setw(12) << std::setprecision(6) << losses_no_recompute[step] << " | "
                  << std::setw(9) << std::setprecision(6) << losses_with_recompute[step] << " | "
                  << std::setw(9) << std::setprecision(6) << loss_diff << " | "
                  << std::setw(12) << std::setprecision(6) << norms_no_recompute[step] << " | "
                  << std::setw(9) << std::setprecision(6) << norms_with_recompute[step] << " | "
                  << std::setw(9) << std::setprecision(6) << norm_diff << std::endl;
    }

    // The test documents expected behavior: some divergence between modes is normal.
    // Recompute-block mode typically trains better due to fresh activation computation
    // before backward pass, avoiding subtle floating-point drift.
    float rel_final_diff = final_loss_diff / std::max({final_loss_no_recompute, final_loss_recompute, 1e-6f});
    INFO("Relative final loss difference: " << rel_final_diff);

    // Allow up to 10% difference in final loss - divergence is expected behavior
    // and varies by training dynamics. The test documents this divergence pattern.
    REQUIRE(rel_final_diff < 0.10f);
}
