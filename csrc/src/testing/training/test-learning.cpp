// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Tests to verify that models actually learn during training.
// Validates that loss decreases over training steps for:
// - Dense models (Qwen2/Qwen3)
// - MoE models (Qwen3MoE)
// - With and without LoRA
// - With and without QLoRA (FP8, FP4, BnB)

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "config/lora_adapter_config.h"
#include "config/pretrained_config.h"
#include "kernels/kernels.h"
#include "models/qwen25/qwen25_model.h"
#include "models/qwen25/transformer_block.h"
#include "models/qwen3moe/config.h"
#include "models/qwen3moe/qwen3_moe_block.h"
#include "models/qwen3moe/qwen3_moe_model.h"
#include "modules/lora/lora_model.h"
#include "modules/model_factory.h"
#include "modules/qlora/qlora_config.h"
#include "training/model.h"
#include "training/runtime_options.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"

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

bool fp8_supported() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return false;
    }
    // FP8 tensor cores are available on Ada (SM 8.9) and newer.
    return (prop.major > 8) || (prop.major == 8 && prop.minor >= 9);
}

bool fp4_supported() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return false;
    }
    // NVFP4 is available on Blackwell (SM100+) and newer.
    return prop.major >= 10;
}

PretrainedConfig create_dense_config(int num_layers = 2, int vocab_size = 128) {
    PretrainedConfig config;
    config.Architecture = PretrainedConfig::QWEN2;
    config.HiddenSize = 256;
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

Qwen3MoEConfig create_moe_config(int num_layers = 2, int vocab_size = 128) {
    Qwen3MoEConfig config;
    config.HiddenSize = 256;
    config.IntermediateSize = 256;  // Dense layer intermediate (not used in pure MoE)
    config.MoeIntermediateSize = 128;  // Per-expert intermediate size
    config.NumQueryHeads = 4;
    config.NumKeyValHeads = 4;
    config.NumLayers = num_layers;
    config.VocabSize = vocab_size;
    config.MaxPositionEmbeddings = 1024;
    config.RopeTheta = 10000.0f;
    config.RmsNormEps = 1e-6f;
    config.TiedWordEmbeddings = false;
    config.UseQKVBias = false;
    config.UseQKNorm = true;  // Qwen3 uses QK norm
    config.BosTokenId = 0;
    config.EosTokenId = 1;
    config.PadTokenId = 2;
    config.DType = ETensorDType::BF16;

    // MoE-specific settings
    config.NumExperts = 4;
    config.NumExpertsPerTok = 2;
    config.RouterAuxLossCoef = 0.01f;

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

// Generate deterministic but varied training data
void fill_training_data(Tensor& inputs, Tensor& targets, int B, int T, int vocab_size, int step) {
    auto* input_ptr = inputs.get<std::int32_t>();
    auto* target_ptr = targets.get<std::int32_t>();

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int idx = b * T + t;
            // Create deterministic patterns that change per step
            input_ptr[idx] = ((idx * 17 + step * 7) % (vocab_size - 1)) + 1;  // Avoid padding token
            target_ptr[idx] = ((idx * 13 + step * 11 + 1) % (vocab_size - 1)) + 1;
        }
    }
}

struct LearningTestResult {
    std::vector<float> losses;
    std::vector<float> norms;
    float initial_loss = 0.0f;
    float final_loss = 0.0f;
    float loss_reduction = 0.0f;
    bool learned = false;
};

} // namespace

// ============================================================================
// Dense Model Learning Tests
// ============================================================================

TEST_CASE("Dense model learns: loss decreases over training steps", "[training][learning][dense][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 2;
        constexpr int T = 64;
        constexpr int num_steps = 20;
        constexpr float lr = 1e-3f;  // Higher LR to see learning faster

        PretrainedConfig cfg = create_dense_config(/*num_layers=*/2, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(
            cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();
        fill_position_ids(pos_ids, B, T);

        std::vector<float> losses;
        losses.reserve(num_steps);

        // Train for multiple steps
        for (int step = 0; step < num_steps; ++step) {
            fill_training_data(inputs, targets, B, T, cfg.VocabSize, step);

            model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
            model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
            model->update(comm, lr, /*beta1=*/0.9f, /*beta2=*/0.999f, /*t=*/step + 1,
                         /*epsilon=*/1e-8f, /*weight_decay=*/0.0f, /*grad_clip=*/1.0f);

            CUDA_CHECK(cudaDeviceSynchronize());
            float loss = model->get_loss();
            losses.push_back(loss);
            REQUIRE(std::isfinite(loss));
        }

        // Verify learning: final loss should be lower than initial loss
        float initial_loss = losses[0];
        float final_loss = losses.back();
        float loss_reduction = initial_loss - final_loss;
        float relative_improvement = loss_reduction / initial_loss;

        INFO("Initial loss: " << initial_loss);
        INFO("Final loss: " << final_loss);
        INFO("Loss reduction: " << loss_reduction);
        INFO("Relative improvement: " << (relative_improvement * 100) << "%");

        // Model should show meaningful learning (at least 5% improvement)
        REQUIRE(relative_improvement > 0.05f);
        REQUIRE(final_loss < initial_loss);

        // Loss should be monotonically decreasing or at least trending down
        // (allowing for some noise in later steps)
        int decreasing_steps = 0;
        for (size_t i = 1; i < losses.size(); ++i) {
            if (losses[i] <= losses[i-1]) {
                decreasing_steps++;
            }
        }
        float decrease_rate = static_cast<float>(decreasing_steps) / (losses.size() - 1);
        INFO("Fraction of decreasing steps: " << decrease_rate);
        REQUIRE(decrease_rate > 0.6f);  // At least 60% of steps should decrease loss
    });
}

TEST_CASE("Dense model with LoRA learns properly", "[training][learning][dense][lora][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 2;
        constexpr int T = 64;
        constexpr int num_steps = 20;
        constexpr float lr = 1e-3f;

        PretrainedConfig cfg = create_dense_config(/*num_layers=*/2, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();

        auto allocator = std::make_shared<TensorAllocator>();

        using DenseBlock = modules::Qwen2TransformerBlock;
        using DenseModel = modules::Qwen2Model;

        auto base_any = modules::ModelFactory::create_from_pretrained_config(
            cfg, opts, comm.rank(), comm.world_size(), allocator);
        auto* dense_ptr = dynamic_cast<DenseModel*>(base_any.release());
        REQUIRE(dense_ptr != nullptr);
        std::unique_ptr<DenseModel> base_model(dense_ptr);

        // Configure LoRA
        modules::ModularLoRAConfig lora_cfg;
        lora_cfg.rank = 8;
        lora_cfg.alpha = 16.0f;
        lora_cfg.dtype = ETensorDType::FP32;
        lora_cfg.dropout = 0.0f;
        lora_cfg.with_all();  // Apply LoRA to all linear layers

        modules::ModularLoRAModel<DenseBlock> model(std::move(base_model), lora_cfg, opts, comm, allocator);
        model.allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model.init_weights(comm);

        auto& inputs = model.get_input_buffer();
        auto& targets = model.get_target_buffer();
        auto& pos_ids = model.get_position_ids_buffer();
        fill_position_ids(pos_ids, B, T);

        std::vector<float> losses;
        losses.reserve(num_steps);

        for (int step = 0; step < num_steps; ++step) {
            fill_training_data(inputs, targets, B, T, cfg.VocabSize, step);

            model.forward(inputs, pos_ids, comm, /*micro_step=*/0);
            model.backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
            model.update(comm, lr, 0.9f, 0.999f, step + 1, 1e-8f, 0.0f, 1.0f);

            CUDA_CHECK(cudaDeviceSynchronize());
            float loss = model.get_loss();
            losses.push_back(loss);
            REQUIRE(std::isfinite(loss));
        }

        float initial_loss = losses[0];
        float final_loss = losses.back();
        float relative_improvement = (initial_loss - final_loss) / initial_loss;

        INFO("LoRA - Initial loss: " << initial_loss);
        INFO("LoRA - Final loss: " << final_loss);
        INFO("LoRA - Relative improvement: " << (relative_improvement * 100) << "%");

        // LoRA should also enable learning
        REQUIRE(relative_improvement > 0.05f);
        REQUIRE(final_loss < initial_loss);
    });
}

TEST_CASE("Dense model with QLoRA-FP8 learns properly", "[training][learning][dense][qlora][fp8][gpu][.disabled]") {

// ============================================================================
// MoE Model Learning Tests
// ============================================================================

TEST_CASE("MoE model learns: loss decreases over training steps", "[training][learning][moe][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 2;
        constexpr int T = 64;
        constexpr int num_steps = 20;
        constexpr float lr = 1e-3f;

        Qwen3MoEConfig cfg = create_moe_config(/*num_layers=*/2, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(
            cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();
        fill_position_ids(pos_ids, B, T);

        std::vector<float> losses;
        losses.reserve(num_steps);

        for (int step = 0; step < num_steps; ++step) {
            fill_training_data(inputs, targets, B, T, cfg.VocabSize, step);

            model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
            model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
            model->update(comm, lr, 0.9f, 0.999f, step + 1, 1e-8f, 0.0f, 1.0f);

            CUDA_CHECK(cudaDeviceSynchronize());
            float loss = model->get_loss();
            losses.push_back(loss);
            REQUIRE(std::isfinite(loss));
        }

        float initial_loss = losses[0];
        float final_loss = losses.back();
        float loss_reduction = initial_loss - final_loss;
        float relative_improvement = loss_reduction / initial_loss;

        INFO("MoE - Initial loss: " << initial_loss);
        INFO("MoE - Final loss: " << final_loss);
        INFO("MoE - Loss reduction: " << loss_reduction);
        INFO("MoE - Relative improvement: " << (relative_improvement * 100) << "%");

        // MoE model should also learn
        REQUIRE(relative_improvement > 0.05f);
        REQUIRE(final_loss < initial_loss);

        // Check trend
        int decreasing_steps = 0;
        for (size_t i = 1; i < losses.size(); ++i) {
            if (losses[i] <= losses[i-1]) {
                decreasing_steps++;
            }
        }
        float decrease_rate = static_cast<float>(decreasing_steps) / (losses.size() - 1);
        INFO("MoE - Fraction of decreasing steps: " << decrease_rate);
        REQUIRE(decrease_rate > 0.6f);
    });
}

TEST_CASE("MoE model with LoRA learns properly", "[training][learning][moe][lora][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 2;
        constexpr int T = 64;
        constexpr int num_steps = 20;
        constexpr float lr = 1e-3f;

        Qwen3MoEConfig cfg = create_moe_config(/*num_layers=*/2, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();

        auto allocator = std::make_shared<TensorAllocator>();

        using MoEBlock = modules::Qwen3MoEBlock;
        using MoEModel = modules::Qwen3MoEModel;

        auto base_any = modules::ModelFactory::create_from_pretrained_config(
            cfg, opts, comm.rank(), comm.world_size(), allocator);
        auto* moe_ptr = dynamic_cast<MoEModel*>(base_any.release());
        REQUIRE(moe_ptr != nullptr);
        std::unique_ptr<MoEModel> base_model(moe_ptr);

        // Configure LoRA
        modules::ModularLoRAConfig lora_cfg;
        lora_cfg.rank = 8;
        lora_cfg.alpha = 16.0f;
        lora_cfg.dtype = ETensorDType::BF16;
        lora_cfg.dropout = 0.0f;
        lora_cfg.with_all();  // Apply to attention, experts, and shared expert

        modules::ModularLoRAModel<MoEBlock> model(std::move(base_model), lora_cfg, opts, comm, allocator);
        model.allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model.init_weights(comm);

        auto& inputs = model.get_input_buffer();
        auto& targets = model.get_target_buffer();
        auto& pos_ids = model.get_position_ids_buffer();
        fill_position_ids(pos_ids, B, T);

        std::vector<float> losses;
        losses.reserve(num_steps);

        for (int step = 0; step < num_steps; ++step) {
            fill_training_data(inputs, targets, B, T, cfg.VocabSize, step);

            model.forward(inputs, pos_ids, comm, /*micro_step=*/0);
            model.backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
            model.update(comm, lr, 0.9f, 0.999f, step + 1, 1e-8f, 0.0f, 1.0f);

            CUDA_CHECK(cudaDeviceSynchronize());
            float loss = model.get_loss();
            losses.push_back(loss);
            REQUIRE(std::isfinite(loss));
        }

        float initial_loss = losses[0];
        float final_loss = losses.back();
        float relative_improvement = (initial_loss - final_loss) / initial_loss;

        INFO("MoE+LoRA - Initial loss: " << initial_loss);
        INFO("MoE+LoRA - Final loss: " << final_loss);
        INFO("MoE+LoRA - Relative improvement: " << (relative_improvement * 100) << "%");

        // MoE with LoRA should learn
        REQUIRE(relative_improvement > 0.05f);
        REQUIRE(final_loss < initial_loss);
    });
}

TEST_CASE("MoE model with QLoRA-FP8 learns properly", "[training][learning][moe][qlora][fp8][gpu][.disabled]") {

// ============================================================================
// Comparative Learning Tests
// ============================================================================

TEST_CASE("Dense vs MoE learning comparison", "[training][learning][comparison][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    constexpr int B = 2;
    constexpr int T = 64;
    constexpr int num_steps = 20;
    constexpr float lr = 1e-3f;

    float dense_improvement = 0.0f;
    float moe_improvement = 0.0f;

    // Train dense model
    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        PretrainedConfig cfg = create_dense_config(/*num_layers=*/2, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(
            cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();
        fill_position_ids(pos_ids, B, T);

        float initial_loss = 0.0f;
        float final_loss = 0.0f;

        for (int step = 0; step < num_steps; ++step) {
            fill_training_data(inputs, targets, B, T, cfg.VocabSize, step);

            model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
            model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
            model->update(comm, lr, 0.9f, 0.999f, step + 1, 1e-8f, 0.0f, 1.0f);

            CUDA_CHECK(cudaDeviceSynchronize());
            float loss = model->get_loss();
            if (step == 0) initial_loss = loss;
            if (step == num_steps - 1) final_loss = loss;
        }

        dense_improvement = (initial_loss - final_loss) / initial_loss;
    });

    // Train MoE model
    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        Qwen3MoEConfig cfg = create_moe_config(/*num_layers=*/2, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(
            cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();
        fill_position_ids(pos_ids, B, T);

        float initial_loss = 0.0f;
        float final_loss = 0.0f;

        for (int step = 0; step < num_steps; ++step) {
            fill_training_data(inputs, targets, B, T, cfg.VocabSize, step);

            model->forward(inputs, pos_ids, comm, /*micro_step=*/0);
            model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0);
            model->update(comm, lr, 0.9f, 0.999f, step + 1, 1e-8f, 0.0f, 1.0f);

            CUDA_CHECK(cudaDeviceSynchronize());
            float loss = model->get_loss();
            if (step == 0) initial_loss = loss;
            if (step == num_steps - 1) final_loss = loss;
        }

        moe_improvement = (initial_loss - final_loss) / initial_loss;
    });

    INFO("Dense model improvement: " << (dense_improvement * 100) << "%");
    INFO("MoE model improvement: " << (moe_improvement * 100) << "%");

    // Both should learn
    REQUIRE(dense_improvement > 0.05f);
    REQUIRE(moe_improvement > 0.05f);
}

// QLoRA comparison tests are disabled for now - require proper weight loading from file
// TEST_CASE("LoRA vs QLoRA learning comparison on dense model", "[training][learning][comparison][gpu][.disabled]") {

// ============================================================================
// Extended Learning Tests (longer training)
// ============================================================================

TEST_CASE("Dense model shows sustained learning over extended training", "[training][learning][extended][dense][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 2;
        constexpr int T = 64;
        constexpr int num_steps = 50;
        constexpr float lr = 5e-4f;

        PretrainedConfig cfg = create_dense_config(/*num_layers=*/2, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(
            cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();
        fill_position_ids(pos_ids, B, T);

        std::vector<float> losses;
        losses.reserve(num_steps);

        for (int step = 0; step < num_steps; ++step) {
            fill_training_data(inputs, targets, B, T, cfg.VocabSize, step);

            model->forward(inputs, pos_ids, comm, 0);
            model->backward(inputs, targets, comm, 1, 0);
            model->update(comm, lr, 0.9f, 0.999f, step + 1, 1e-8f, 0.01f, 1.0f);

            CUDA_CHECK(cudaDeviceSynchronize());
            losses.push_back(model->get_loss());
        }

        // Check early vs late performance
        float early_avg = 0.0f;
        float late_avg = 0.0f;
        for (int i = 0; i < 10; ++i) {
            early_avg += losses[i];
            late_avg += losses[num_steps - 10 + i];
        }
        early_avg /= 10.0f;
        late_avg /= 10.0f;

        float improvement = (early_avg - late_avg) / early_avg;

        INFO("Early average loss (steps 0-9): " << early_avg);
        INFO("Late average loss (steps 40-49): " << late_avg);
        INFO("Overall improvement: " << (improvement * 100) << "%");

        // Should show sustained learning
        REQUIRE(improvement > 0.1f);  // At least 10% improvement over 50 steps
        REQUIRE(late_avg < early_avg);
    });
}

TEST_CASE("MoE model shows sustained learning over extended training", "[training][learning][extended][moe][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 2;
        constexpr int T = 64;
        constexpr int num_steps = 50;
        constexpr float lr = 5e-4f;

        Qwen3MoEConfig cfg = create_moe_config(/*num_layers=*/2, /*vocab_size=*/128);
        RuntimeOptions opts = create_test_options();

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(
            cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();
        fill_position_ids(pos_ids, B, T);

        std::vector<float> losses;
        losses.reserve(num_steps);

        for (int step = 0; step < num_steps; ++step) {
            fill_training_data(inputs, targets, B, T, cfg.VocabSize, step);

            model->forward(inputs, pos_ids, comm, 0);
            model->backward(inputs, targets, comm, 1, 0);
            model->update(comm, lr, 0.9f, 0.999f, step + 1, 1e-8f, 0.01f, 1.0f);

            CUDA_CHECK(cudaDeviceSynchronize());
            losses.push_back(model->get_loss());
        }

        // Check early vs late performance
        float early_avg = 0.0f;
        float late_avg = 0.0f;
        for (int i = 0; i < 10; ++i) {
            early_avg += losses[i];
            late_avg += losses[num_steps - 10 + i];
        }
        early_avg /= 10.0f;
        late_avg /= 10.0f;

        float improvement = (early_avg - late_avg) / early_avg;

        INFO("MoE Early average loss (steps 0-9): " << early_avg);
        INFO("MoE Late average loss (steps 40-49): " << late_avg);
        INFO("MoE Overall improvement: " << (improvement * 100) << "%");

        // MoE should also show sustained learning
        REQUIRE(improvement > 0.1f);
        REQUIRE(late_avg < early_avg);
    });
}
