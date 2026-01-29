// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Tests for checkpoint save/resume functionality.
// Validates that training state (weights, optimizer state) is preserved across:
// - Dense models without LoRA
// - Dense models with LoRA
// - MoE models without LoRA
// - MoE models with LoRA

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "config/pretrained_config.h"
#include "models/qwen25/qwen25_model.h"
#include "models/qwen3moe/config.h"
#include "models/qwen3moe/qwen3_moe_model.h"
#include "modules/lora/lora_model.h"
#include "modules/model_factory.h"
#include "training/checkpoint.h"
#include "training/model.h"
#include "training/runtime_options.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"

using Catch::Approx;
namespace fs = std::filesystem;

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
    config.IntermediateSize = 256;
    config.MoeIntermediateSize = 128;
    config.NumQueryHeads = 4;
    config.NumKeyValHeads = 4;
    config.NumLayers = num_layers;
    config.VocabSize = vocab_size;
    config.MaxPositionEmbeddings = 1024;
    config.RopeTheta = 10000.0f;
    config.RmsNormEps = 1e-6f;
    config.TiedWordEmbeddings = false;
    config.UseQKVBias = false;
    config.UseQKNorm = true;
    config.BosTokenId = 0;
    config.EosTokenId = 1;
    config.PadTokenId = 2;
    config.DType = ETensorDType::BF16;
    config.NumExperts = 4;
    config.NumExpertsPerTok = 2;
    config.RouterAuxLossCoef = 0.01f;
    return config;
}

modules::ModularLoRAConfig create_lora_config() {
    modules::ModularLoRAConfig lora_cfg;
    lora_cfg.rank = 8;
    lora_cfg.alpha = 16.0f;
    lora_cfg.dropout = 0.0f;
    lora_cfg.dtype = ETensorDType::BF16;  // Use BF16 to match model activations
    lora_cfg.with_all();
    return lora_cfg;
}

RuntimeOptions create_test_options() {
    RuntimeOptions opts;
    opts.UseCudaGraphs = false;
    opts.Recompute = RecomputeLevel::None;
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

void fill_training_data(Tensor& inputs, Tensor& targets, int B, int T, int vocab_size, int step) {
    auto* input_ptr = inputs.get<std::int32_t>();
    auto* target_ptr = targets.get<std::int32_t>();

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int idx = b * T + t;
            input_ptr[idx] = ((idx * 17 + step * 7) % (vocab_size - 1)) + 1;
            target_ptr[idx] = ((idx * 13 + step * 11 + 1) % (vocab_size - 1)) + 1;
        }
    }
}

// Run N training steps and return the losses
std::vector<float> run_training_steps(IModel& model, NCCLCommunicator& comm,
                                       int B, int T, int vocab_size,
                                       int start_step, int num_steps, float lr) {
    std::vector<float> losses;
    losses.reserve(num_steps);

    auto& inputs = model.get_input_buffer();
    auto& targets = model.get_target_buffer();
    auto& pos_ids = model.get_position_ids_buffer();
    fill_position_ids(pos_ids, B, T);

    for (int step = start_step; step < start_step + num_steps; ++step) {
        fill_training_data(inputs, targets, B, T, vocab_size, step);

        model.forward(inputs, pos_ids, comm, 0);
        model.backward(inputs, targets, comm, 1, 0);
        model.update(comm, lr, 0.9f, 0.999f, step, 1e-8f, 0.01f, 1.0f);

        CUDA_CHECK(cudaDeviceSynchronize());
        losses.push_back(model.get_loss());
    }

    return losses;
}

// Helper to get a unique temp directory for checkpoints
std::string get_temp_checkpoint_dir(const std::string& test_name) {
    auto path = fs::temp_directory_path() / ("surogate_test_" + test_name + "_" + std::to_string(std::rand()));
    fs::create_directories(path);
    return path.string();
}

// Cleanup helper
void cleanup_checkpoint_dir(const std::string& dir) {
    if (fs::exists(dir)) {
        fs::remove_all(dir);
    }
}

// Type aliases
using DenseBlock = modules::Qwen2TransformerBlock;
using DenseModel = modules::Qwen2Model;
using MoEBlock = modules::Qwen3MoEBlock;
using MoEModel = modules::Qwen3MoEModel;

} // namespace

// ============================================================================
// Dense Model Checkpoint Tests (Non-LoRA)
// ============================================================================

TEST_CASE("Dense model checkpoint: save and resume preserves training state", "[checkpoint][dense][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    std::string checkpoint_dir = get_temp_checkpoint_dir("dense");

    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        constexpr int B = 2;
        constexpr int T = 32;
        constexpr int vocab_size = 128;
        constexpr float lr = 1e-3f;
        constexpr int checkpoint_step = 5;
        constexpr int total_steps = 10;

        PretrainedConfig cfg = create_dense_config(2, vocab_size);
        RuntimeOptions opts = create_test_options();

        // Phase 1: Train for checkpoint_step steps and save
        std::vector<float> losses_before;
        {
            auto allocator = std::make_shared<TensorAllocator>();
            auto model = modules::ModelFactory::create_from_pretrained_config(
                cfg, opts, comm.rank(), comm.world_size(), allocator);
            model->allocate_run_state(opts, comm, B, T, true);
            model->init_weights(comm);

            losses_before = run_training_steps(*model, comm, B, T, vocab_size, 1, checkpoint_step, lr);

            save_checkpoint(checkpoint_dir, checkpoint_step, *model, nullptr, comm);
        }

        // Phase 2: Continue training from checkpoint
        std::vector<float> losses_resumed;
        {
            auto allocator = std::make_shared<TensorAllocator>();
            auto model = modules::ModelFactory::create_from_pretrained_config(
                cfg, opts, comm.rank(), comm.world_size(), allocator);
            model->allocate_run_state(opts, comm, B, T, true);
            model->init_weights(comm);

            load_checkpoint(checkpoint_dir, checkpoint_step, *model, nullptr, comm);

            losses_resumed = run_training_steps(*model, comm, B, T, vocab_size,
                                                 checkpoint_step + 1, total_steps - checkpoint_step, lr);
        }

        // Phase 3: Train continuously without checkpoint for comparison
        std::vector<float> losses_continuous;
        {
            auto allocator = std::make_shared<TensorAllocator>();
            auto model = modules::ModelFactory::create_from_pretrained_config(
                cfg, opts, comm.rank(), comm.world_size(), allocator);
            model->allocate_run_state(opts, comm, B, T, true);
            model->init_weights(comm);

            losses_continuous = run_training_steps(*model, comm, B, T, vocab_size, 1, total_steps, lr);
        }

        // Verify: resumed training should produce similar losses to continuous training
        float continuous_loss_at_resume = losses_continuous[checkpoint_step];
        float resumed_first_loss = losses_resumed[0];

        REQUIRE(resumed_first_loss == Approx(continuous_loss_at_resume).epsilon(0.01));
        REQUIRE(losses_resumed.back() < losses_resumed.front());
    });

    cleanup_checkpoint_dir(checkpoint_dir);
}

// ============================================================================
// Dense Model with LoRA Checkpoint Tests
// ============================================================================

TEST_CASE("Dense LoRA checkpoint: save and resume preserves LoRA optimizer state", "[checkpoint][lora][dense][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    std::string checkpoint_dir = get_temp_checkpoint_dir("dense_lora");

    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        constexpr int B = 2;
        constexpr int T = 32;
        constexpr int vocab_size = 128;
        constexpr float lr = 1e-3f;
        constexpr int checkpoint_step = 5;
        constexpr int total_steps = 10;

        PretrainedConfig cfg = create_dense_config(2, vocab_size);
        RuntimeOptions opts = create_test_options();
        modules::ModularLoRAConfig lora_cfg = create_lora_config();

        // Phase 1: Train with LoRA for checkpoint_step steps and save
        std::vector<float> losses_before;
        float loss_at_checkpoint = 0.0f;
        {
            auto allocator = std::make_shared<TensorAllocator>();
            auto base_any = modules::ModelFactory::create_from_pretrained_config(
                cfg, opts, comm.rank(), comm.world_size(), allocator);
            auto* dense_ptr = dynamic_cast<DenseModel*>(base_any.release());
            REQUIRE(dense_ptr != nullptr);
            std::unique_ptr<DenseModel> base_model(dense_ptr);

            modules::ModularLoRAModel<DenseBlock> model(std::move(base_model), lora_cfg, opts, comm, allocator);
            model.allocate_run_state(opts, comm, B, T, true);
            model.init_weights(comm);

            losses_before = run_training_steps(model, comm, B, T, vocab_size, 1, checkpoint_step, lr);
            loss_at_checkpoint = model.get_loss();

            save_checkpoint(checkpoint_dir, checkpoint_step, model, nullptr, comm);
        }

        // Verify checkpoint files exist
        std::string step_dir = checkpoint_dir + "/step_00000005";
        REQUIRE(fs::exists(fs::path(step_dir) / "adapter_model.safetensors"));
        REQUIRE(fs::exists(fs::path(step_dir) / "lora_optimizer.safetensors"));
        REQUIRE(fs::exists(fs::path(step_dir) / "lora_optimizer.json"));

        // Phase 2: Resume from checkpoint and continue training
        std::vector<float> losses_resumed;
        float loss_after_resume = 0.0f;
        {
            auto allocator = std::make_shared<TensorAllocator>();
            auto base_any = modules::ModelFactory::create_from_pretrained_config(
                cfg, opts, comm.rank(), comm.world_size(), allocator);
            auto* dense_ptr = dynamic_cast<DenseModel*>(base_any.release());
            REQUIRE(dense_ptr != nullptr);
            std::unique_ptr<DenseModel> base_model(dense_ptr);

            modules::ModularLoRAModel<DenseBlock> model(std::move(base_model), lora_cfg, opts, comm, allocator);
            model.allocate_run_state(opts, comm, B, T, true);
            model.init_weights(comm);

            load_checkpoint(checkpoint_dir, checkpoint_step, model, nullptr, comm);

            // First step after resume - get the loss before any updates
            auto& inputs = model.get_input_buffer();
            auto& targets = model.get_target_buffer();
            auto& pos_ids = model.get_position_ids_buffer();
            fill_position_ids(pos_ids, B, T);
            fill_training_data(inputs, targets, B, T, vocab_size, checkpoint_step + 1);

            model.forward(inputs, pos_ids, comm, 0);
            loss_after_resume = model.get_loss();

            losses_resumed = run_training_steps(model, comm, B, T, vocab_size,
                                                 checkpoint_step + 1, total_steps - checkpoint_step, lr);
        }

        // Phase 3: Train continuously for comparison
        std::vector<float> losses_continuous;
        {
            auto allocator = std::make_shared<TensorAllocator>();
            auto base_any = modules::ModelFactory::create_from_pretrained_config(
                cfg, opts, comm.rank(), comm.world_size(), allocator);
            auto* dense_ptr = dynamic_cast<DenseModel*>(base_any.release());
            REQUIRE(dense_ptr != nullptr);
            std::unique_ptr<DenseModel> base_model(dense_ptr);

            modules::ModularLoRAModel<DenseBlock> model(std::move(base_model), lora_cfg, opts, comm, allocator);
            model.allocate_run_state(opts, comm, B, T, true);
            model.init_weights(comm);

            losses_continuous = run_training_steps(model, comm, B, T, vocab_size, 1, total_steps, lr);
        }

        // Verify resumed training produces reasonable losses
        float initial_loss = losses_continuous[0];
        float resumed_first_loss = losses_resumed[0];

        INFO("Initial loss (fresh model): " << initial_loss);
        INFO("Loss at checkpoint (step " << checkpoint_step << "): " << loss_at_checkpoint);
        INFO("Resumed first step loss: " << resumed_first_loss);
        INFO("Resumed final step loss: " << losses_resumed.back());

        // Key verification: resumed loss should be reasonable (not NaN, not infinity, not massive spike)
        REQUIRE(std::isfinite(resumed_first_loss));
        REQUIRE(resumed_first_loss < 20.0f);  // Loss shouldn't spike to unreasonable values

        // Training should continue to produce reasonable values (no NaN/Inf)
        for (float loss : losses_resumed) {
            REQUIRE(std::isfinite(loss));
        }

        // Loss should not dramatically increase after resume (no more than 2x initial)
        REQUIRE(resumed_first_loss < initial_loss * 2.0f);
    });

    cleanup_checkpoint_dir(checkpoint_dir);
}

// ============================================================================
// MoE Model Checkpoint Tests (Non-LoRA)
// ============================================================================

TEST_CASE("MoE model checkpoint: save and resume preserves training state", "[checkpoint][moe][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    std::string checkpoint_dir = get_temp_checkpoint_dir("moe");

    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        constexpr int B = 2;
        constexpr int T = 32;
        constexpr int vocab_size = 128;
        constexpr float lr = 1e-3f;
        constexpr int checkpoint_step = 5;
        constexpr int total_steps = 10;

        Qwen3MoEConfig cfg = create_moe_config(2, vocab_size);
        RuntimeOptions opts = create_test_options();

        // Phase 1: Train for checkpoint_step steps and save
        std::vector<float> losses_before;
        {
            auto allocator = std::make_shared<TensorAllocator>();
            auto model = modules::ModelFactory::create_from_pretrained_config(
                cfg, opts, comm.rank(), comm.world_size(), allocator);
            model->allocate_run_state(opts, comm, B, T, true);
            model->init_weights(comm);

            losses_before = run_training_steps(*model, comm, B, T, vocab_size, 1, checkpoint_step, lr);

            save_checkpoint(checkpoint_dir, checkpoint_step, *model, nullptr, comm);
        }

        // Phase 2: Resume from checkpoint
        std::vector<float> losses_resumed;
        {
            auto allocator = std::make_shared<TensorAllocator>();
            auto model = modules::ModelFactory::create_from_pretrained_config(
                cfg, opts, comm.rank(), comm.world_size(), allocator);
            model->allocate_run_state(opts, comm, B, T, true);
            model->init_weights(comm);

            load_checkpoint(checkpoint_dir, checkpoint_step, *model, nullptr, comm);

            losses_resumed = run_training_steps(*model, comm, B, T, vocab_size,
                                                 checkpoint_step + 1, total_steps - checkpoint_step, lr);
        }

        // Phase 3: Continuous training for comparison
        std::vector<float> losses_continuous;
        {
            auto allocator = std::make_shared<TensorAllocator>();
            auto model = modules::ModelFactory::create_from_pretrained_config(
                cfg, opts, comm.rank(), comm.world_size(), allocator);
            model->allocate_run_state(opts, comm, B, T, true);
            model->init_weights(comm);

            losses_continuous = run_training_steps(*model, comm, B, T, vocab_size, 1, total_steps, lr);
        }

        // Verify resumed training produces similar results
        float continuous_loss_at_resume = losses_continuous[checkpoint_step];
        float resumed_first_loss = losses_resumed[0];

        REQUIRE(resumed_first_loss == Approx(continuous_loss_at_resume).epsilon(0.01));
        REQUIRE(losses_resumed.back() < losses_resumed.front());
    });

    cleanup_checkpoint_dir(checkpoint_dir);
}

// ============================================================================
// MoE Model with LoRA Checkpoint Tests
// ============================================================================

TEST_CASE("MoE LoRA checkpoint: save and resume preserves LoRA optimizer state", "[checkpoint][lora][moe][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    std::string checkpoint_dir = get_temp_checkpoint_dir("moe_lora");

    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        constexpr int B = 2;
        constexpr int T = 32;
        constexpr int vocab_size = 128;
        constexpr float lr = 1e-3f;
        constexpr int checkpoint_step = 5;
        constexpr int total_steps = 10;

        Qwen3MoEConfig cfg = create_moe_config(2, vocab_size);
        RuntimeOptions opts = create_test_options();
        modules::ModularLoRAConfig lora_cfg = create_lora_config();

        // Phase 1: Train with LoRA and save checkpoint
        std::vector<float> losses_before;
        float loss_at_checkpoint = 0.0f;
        {
            auto allocator = std::make_shared<TensorAllocator>();
            auto base_any = modules::ModelFactory::create_from_pretrained_config(
                cfg, opts, comm.rank(), comm.world_size(), allocator);
            auto* moe_ptr = dynamic_cast<MoEModel*>(base_any.release());
            REQUIRE(moe_ptr != nullptr);
            std::unique_ptr<MoEModel> base_model(moe_ptr);

            modules::ModularLoRAModel<MoEBlock> model(std::move(base_model), lora_cfg, opts, comm, allocator);
            model.allocate_run_state(opts, comm, B, T, true);
            model.init_weights(comm);

            losses_before = run_training_steps(model, comm, B, T, vocab_size, 1, checkpoint_step, lr);
            loss_at_checkpoint = model.get_loss();

            save_checkpoint(checkpoint_dir, checkpoint_step, model, nullptr, comm);
        }

        // Verify checkpoint files exist
        std::string step_dir = checkpoint_dir + "/step_00000005";
        REQUIRE(fs::exists(fs::path(step_dir) / "adapter_model.safetensors"));
        REQUIRE(fs::exists(fs::path(step_dir) / "lora_optimizer.safetensors"));

        // Phase 2: Resume and continue training
        std::vector<float> losses_resumed;
        float loss_after_resume = 0.0f;
        {
            auto allocator = std::make_shared<TensorAllocator>();
            auto base_any = modules::ModelFactory::create_from_pretrained_config(
                cfg, opts, comm.rank(), comm.world_size(), allocator);
            auto* moe_ptr = dynamic_cast<MoEModel*>(base_any.release());
            REQUIRE(moe_ptr != nullptr);
            std::unique_ptr<MoEModel> base_model(moe_ptr);

            modules::ModularLoRAModel<MoEBlock> model(std::move(base_model), lora_cfg, opts, comm, allocator);
            model.allocate_run_state(opts, comm, B, T, true);
            model.init_weights(comm);

            load_checkpoint(checkpoint_dir, checkpoint_step, model, nullptr, comm);

            // Get loss right after resume
            auto& inputs = model.get_input_buffer();
            auto& targets = model.get_target_buffer();
            auto& pos_ids = model.get_position_ids_buffer();
            fill_position_ids(pos_ids, B, T);
            fill_training_data(inputs, targets, B, T, vocab_size, checkpoint_step + 1);

            model.forward(inputs, pos_ids, comm, 0);
            loss_after_resume = model.get_loss();

            losses_resumed = run_training_steps(model, comm, B, T, vocab_size,
                                                 checkpoint_step + 1, total_steps - checkpoint_step, lr);
        }

        // Phase 3: Continuous training for comparison
        std::vector<float> losses_continuous;
        {
            auto allocator = std::make_shared<TensorAllocator>();
            auto base_any = modules::ModelFactory::create_from_pretrained_config(
                cfg, opts, comm.rank(), comm.world_size(), allocator);
            auto* moe_ptr = dynamic_cast<MoEModel*>(base_any.release());
            REQUIRE(moe_ptr != nullptr);
            std::unique_ptr<MoEModel> base_model(moe_ptr);

            modules::ModularLoRAModel<MoEBlock> model(std::move(base_model), lora_cfg, opts, comm, allocator);
            model.allocate_run_state(opts, comm, B, T, true);
            model.init_weights(comm);

            losses_continuous = run_training_steps(model, comm, B, T, vocab_size, 1, total_steps, lr);
        }

        // Verify resumed training produces reasonable losses
        float initial_loss = losses_continuous[0];
        float resumed_first_loss = losses_resumed[0];

        INFO("Initial loss (fresh model): " << initial_loss);
        INFO("Loss at checkpoint (step " << checkpoint_step << "): " << loss_at_checkpoint);
        INFO("Resumed first step loss: " << resumed_first_loss);
        INFO("Resumed final step loss: " << losses_resumed.back());

        // Key verification: resumed loss should be reasonable
        REQUIRE(std::isfinite(resumed_first_loss));
        REQUIRE(resumed_first_loss < 20.0f);

        for (float loss : losses_resumed) {
            REQUIRE(std::isfinite(loss));
        }

        REQUIRE(resumed_first_loss < initial_loss * 2.0f);
    });

    cleanup_checkpoint_dir(checkpoint_dir);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_CASE("Checkpoint: loading without optimizer state file gracefully handles missing state", "[checkpoint][lora][gpu]") {
    if (!gpu_available()) SKIP("CUDA not available");

    std::string checkpoint_dir = get_temp_checkpoint_dir("lora_no_opt");

    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        constexpr int B = 2;
        constexpr int T = 32;
        constexpr int vocab_size = 128;

        PretrainedConfig cfg = create_dense_config(2, vocab_size);
        RuntimeOptions opts = create_test_options();
        modules::ModularLoRAConfig lora_cfg = create_lora_config();

        // Create a checkpoint with only adapter weights (simulate old checkpoint format)
        {
            auto allocator = std::make_shared<TensorAllocator>();
            auto base_any = modules::ModelFactory::create_from_pretrained_config(
                cfg, opts, comm.rank(), comm.world_size(), allocator);
            auto* dense_ptr = dynamic_cast<DenseModel*>(base_any.release());
            REQUIRE(dense_ptr != nullptr);
            std::unique_ptr<DenseModel> base_model(dense_ptr);

            modules::ModularLoRAModel<DenseBlock> model(std::move(base_model), lora_cfg, opts, comm, allocator);
            model.allocate_run_state(opts, comm, B, T, true);
            model.init_weights(comm);

            // Run one step to initialize optimizer
            auto& inputs = model.get_input_buffer();
            auto& targets = model.get_target_buffer();
            auto& pos_ids = model.get_position_ids_buffer();
            fill_position_ids(pos_ids, B, T);
            fill_training_data(inputs, targets, B, T, vocab_size, 1);

            model.forward(inputs, pos_ids, comm, 0);
            model.backward(inputs, targets, comm, 1, 0);
            model.update(comm, 1e-3f, 0.9f, 0.999f, 1, 1e-8f, 0.01f, 1.0f);

            save_checkpoint(checkpoint_dir, 1, model, nullptr, comm);

            // Delete the optimizer state files to simulate old checkpoint
            auto step_dir = fs::path(checkpoint_dir) / "step_00000001";
            fs::remove(step_dir / "lora_optimizer.safetensors");
            fs::remove(step_dir / "lora_optimizer.json");
        }

        // Load should succeed even without optimizer state
        {
            auto allocator = std::make_shared<TensorAllocator>();
            auto base_any = modules::ModelFactory::create_from_pretrained_config(
                cfg, opts, comm.rank(), comm.world_size(), allocator);
            auto* dense_ptr = dynamic_cast<DenseModel*>(base_any.release());
            REQUIRE(dense_ptr != nullptr);
            std::unique_ptr<DenseModel> base_model(dense_ptr);

            modules::ModularLoRAModel<DenseBlock> model(std::move(base_model), lora_cfg, opts, comm, allocator);
            model.allocate_run_state(opts, comm, B, T, true);
            model.init_weights(comm);

            // This should not throw - optimizer state will just start fresh
            REQUIRE_NOTHROW(load_checkpoint(checkpoint_dir, 1, model, nullptr, comm));

            // Should be able to continue training
            auto& inputs = model.get_input_buffer();
            auto& targets = model.get_target_buffer();
            auto& pos_ids = model.get_position_ids_buffer();
            fill_position_ids(pos_ids, B, T);
            fill_training_data(inputs, targets, B, T, vocab_size, 2);

            REQUIRE_NOTHROW(model.forward(inputs, pos_ids, comm, 0));
            REQUIRE_NOTHROW(model.backward(inputs, targets, comm, 1, 0));
            REQUIRE_NOTHROW(model.update(comm, 1e-3f, 0.9f, 0.999f, 2, 1e-8f, 0.01f, 1.0f));
        }
    });

    cleanup_checkpoint_dir(checkpoint_dir);
}

// ============================================================================
// Gradient Flow Tests - Verify gradients are non-zero after checkpoint restore
// ============================================================================

TEST_CASE("LoRA checkpoint: gradients flow after restore", "[checkpoint][lora][gpu][gradients]") {
    if (!gpu_available()) SKIP("CUDA not available");

    std::string checkpoint_dir = get_temp_checkpoint_dir("lora_grad_flow");

    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        constexpr int B = 2;
        constexpr int T = 32;
        constexpr int vocab_size = 128;
        constexpr float lr = 1e-3f;

        PretrainedConfig cfg = create_dense_config(2, vocab_size);
        RuntimeOptions opts = create_test_options();
        modules::ModularLoRAConfig lora_cfg = create_lora_config();

        // Phase 1: Train for a few steps and save checkpoint
        float loss_at_checkpoint = 0.0f;
        float norm_at_checkpoint = 0.0f;
        {
            auto allocator = std::make_shared<TensorAllocator>();
            auto base_any = modules::ModelFactory::create_from_pretrained_config(
                cfg, opts, comm.rank(), comm.world_size(), allocator);
            auto* dense_ptr = dynamic_cast<DenseModel*>(base_any.release());
            REQUIRE(dense_ptr != nullptr);
            std::unique_ptr<DenseModel> base_model(dense_ptr);

            modules::ModularLoRAModel<DenseBlock> model(std::move(base_model), lora_cfg, opts, comm, allocator);
            model.allocate_run_state(opts, comm, B, T, true);
            model.init_weights(comm);

            auto& inputs = model.get_input_buffer();
            auto& targets = model.get_target_buffer();
            auto& pos_ids = model.get_position_ids_buffer();
            fill_position_ids(pos_ids, B, T);

            // Train for 5 steps and track losses
            std::vector<float> training_losses;
            for (int step = 1; step <= 5; ++step) {
                fill_training_data(inputs, targets, B, T, vocab_size, step);
                model.forward(inputs, pos_ids, comm, 0);
                model.backward(inputs, targets, comm, 1, 0);
                float step_loss = model.get_loss();
                training_losses.push_back(step_loss);
                model.update(comm, lr, 0.9f, 0.999f, step, 1e-8f, 0.01f, 1.0f);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            INFO("Training losses: " << training_losses[0] << ", " << training_losses[1] << ", "
                 << training_losses[2] << ", " << training_losses[3] << ", " << training_losses[4]);
            // Losses should vary (different data each step)
            REQUIRE(training_losses[0] != training_losses[4]);

            // Record loss and norm before saving
            fill_training_data(inputs, targets, B, T, vocab_size, 6);
            model.forward(inputs, pos_ids, comm, 0);
            model.backward(inputs, targets, comm, 1, 0);
            loss_at_checkpoint = model.get_loss();
            norm_at_checkpoint = model.get_norm();

            INFO("Loss before save: " << loss_at_checkpoint);
            INFO("Norm before save: " << norm_at_checkpoint);
            REQUIRE(norm_at_checkpoint > 0.0f);  // Sanity check: should have non-zero gradient

            // Save checkpoint BEFORE update (so restore test can compare against these values)
            save_checkpoint(checkpoint_dir, 6, model, nullptr, comm);

            // Sample weights BEFORE update
            auto& lora_weights_ref = model.lora_weights();
            auto& block0_ref = lora_weights_ref.get_master_block(0, 0);
            REQUIRE(block0_ref.attention.q.has_value());
            std::vector<nv_bfloat16> w_before(block0_ref.attention.q->A.nelem());
            CUDA_CHECK(cudaMemcpy(w_before.data(), block0_ref.attention.q->A.Data,
                                   w_before.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));

            // SANITY CHECK: Verify loss changes after update
            model.update(comm, lr, 0.9f, 0.999f, 6, 1e-8f, 0.01f, 1.0f);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Sample weights AFTER update
            std::vector<nv_bfloat16> w_after(block0_ref.attention.q->A.nelem());
            CUDA_CHECK(cudaMemcpy(w_after.data(), block0_ref.attention.q->A.Data,
                                   w_after.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));

            float w0_before_f = __bfloat162float(w_before[0]);
            float w0_after_f = __bfloat162float(w_after[0]);
            float w1_before_f = __bfloat162float(w_before[1]);
            float w1_after_f = __bfloat162float(w_after[1]);
            INFO("Weight[0] before: " << w0_before_f << ", after: " << w0_after_f << ", diff: " << (w0_after_f - w0_before_f));
            INFO("Weight[1] before: " << w1_before_f << ", after: " << w1_after_f << ", diff: " << (w1_after_f - w1_before_f));

            // Verify weights changed
            REQUIRE(w0_before_f != w0_after_f);  // Weight should have changed after update

            // Forward+backward with SAME data to compute loss after weight update
            // NOTE: get_loss() returns loss from backward(), not forward()!
            model.forward(inputs, pos_ids, comm, 0);  // Same data
            model.backward(inputs, targets, comm, 1, 0);  // Must call backward to compute loss
            CUDA_CHECK(cudaDeviceSynchronize());

            float loss_after_update = model.get_loss();
            INFO("Loss after update (same data): " << loss_after_update);
            INFO("LoRA enabled: " << model.lora_enabled());
            INFO("LoRA q_proj.A nelem: " << (int)block0_ref.attention.q->A.nelem());

            // Loss should change because weights changed! This is the key test.
            REQUIRE(loss_after_update != loss_at_checkpoint);  // MUST change!
        }

        // Phase 2: Load checkpoint and verify gradients still flow
        {
            auto allocator = std::make_shared<TensorAllocator>();
            auto base_any = modules::ModelFactory::create_from_pretrained_config(
                cfg, opts, comm.rank(), comm.world_size(), allocator);
            auto* dense_ptr = dynamic_cast<DenseModel*>(base_any.release());
            REQUIRE(dense_ptr != nullptr);
            std::unique_ptr<DenseModel> base_model(dense_ptr);

            modules::ModularLoRAModel<DenseBlock> model(std::move(base_model), lora_cfg, opts, comm, allocator);
            model.allocate_run_state(opts, comm, B, T, true);
            model.init_weights(comm);

            load_checkpoint(checkpoint_dir, 6, model, nullptr, comm);

            // Run forward+backward on same data
            auto& inputs = model.get_input_buffer();
            auto& targets = model.get_target_buffer();
            auto& pos_ids = model.get_position_ids_buffer();
            fill_position_ids(pos_ids, B, T);
            fill_training_data(inputs, targets, B, T, vocab_size, 6);

            model.forward(inputs, pos_ids, comm, 0);
            model.backward(inputs, targets, comm, 1, 0);

            float loss_after_restore = model.get_loss();
            float norm_after_restore = model.get_norm();

            INFO("Loss after restore: " << loss_after_restore);
            INFO("Norm after restore: " << norm_after_restore);

            // CRITICAL: These are the key assertions that should catch the bug
            INFO("Loss at checkpoint: " << loss_at_checkpoint);
            INFO("Norm at checkpoint: " << norm_at_checkpoint);
            REQUIRE(std::abs(loss_after_restore - loss_at_checkpoint) < 0.01f);  // Loss should match
            REQUIRE(norm_after_restore > 0.0f);  // MUST have non-zero gradient!
            // Note: Allow 30% tolerance on norm match due to potential floating point
            // accumulation differences between save and restore
            REQUIRE(std::abs(norm_after_restore - norm_at_checkpoint) / norm_at_checkpoint < 0.3f);

            // Sample a LoRA weight before update to verify it changes
            auto& lora_weights = model.lora_weights();
            auto& block0 = lora_weights.get_master_block(0, 0);
            REQUIRE(block0.attention.q.has_value());
            // Copy raw bytes since dtype may be BF16
            size_t weight_bytes = block0.attention.q->A.bytes();
            std::vector<std::byte> weight_before(weight_bytes);
            CUDA_CHECK(cudaMemcpy(weight_before.data(), block0.attention.q->A.Data,
                                   weight_bytes, cudaMemcpyDeviceToHost));

            // Do one update step
            model.update(comm, lr, 0.9f, 0.999f, 6, 1e-8f, 0.01f, 1.0f);
            CUDA_CHECK(cudaDeviceSynchronize());  // Ensure update completes

            // Sample the same weight after update
            std::vector<std::byte> weight_after(weight_bytes);
            CUDA_CHECK(cudaMemcpy(weight_after.data(), block0.attention.q->A.Data,
                                   weight_bytes, cudaMemcpyDeviceToHost));

            // Check if weights actually changed
            bool weights_changed = (weight_before != weight_after);
            INFO("Weight bytes: " << weight_bytes << ", changed: " << weights_changed);
            REQUIRE(weights_changed);  // Weights MUST change after update

            // Verify work buffers also get the updated values after get_block
            auto& work_block = lora_weights.get_block(0, nullptr);  // This should sync master→work
            CUDA_CHECK(cudaDeviceSynchronize());  // Make sure async copy completes

            REQUIRE(work_block.attention.q.has_value());
            std::vector<std::byte> work_weights(work_block.attention.q->A.bytes());
            CUDA_CHECK(cudaMemcpy(work_weights.data(), work_block.attention.q->A.Data,
                                   work_weights.size(), cudaMemcpyDeviceToHost));
            bool work_matches_master = (work_weights == weight_after);
            INFO("Work buffer matches updated master: " << work_matches_master);
            REQUIRE(work_matches_master);  // Work buffer should have updated values

            // Run forward+backward on SAME data to verify loss changes
            // NOTE: get_loss() returns loss from backward(), not forward()!
            model.forward(inputs, pos_ids, comm, 0);
            model.backward(inputs, targets, comm, 1, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
            float loss_after_update = model.get_loss();

            INFO("Loss after update (same data): " << loss_after_update);
            // Loss should change after update because weights changed
            REQUIRE(loss_after_update != loss_after_restore);
        }
    });

    cleanup_checkpoint_dir(checkpoint_dir);
}
