// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Heterogeneous (Hybrid Dense+MoE) model support

#include <vector>
#include <cmath>
#include <memory>

#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "modules/model_config.h"
#include "modules/model_factory.h"
#include "modules/model/heterogeneous_model.h"
#include "utilities/allocator.h"
#include "utilities/utils.h"
#include "test_config.h"
#include "test_utils.h"

using namespace modules;
using namespace testing_utils;
using Catch::Approx;

/**
 * @brief Qwen3 MoE 15B-A2B configuration
 *
 * Qwen3 MoE architecture with:
 * - 128 experts with top-8 routing (2B active parameters per token)
 * - norm_topk_prob: normalize routing weights after top-k selection
 * - QK normalization in attention (requires use_qk_norm=true)
 * - All layers are MoE (decoder_sparse_step=1, no mlp_only_layers)
 */
inline ModelConfig qwen3_moe_15b_a2b(ETensorDType dtype = ETensorDType::BF16) {
    MoEConfig moe;
    moe.num_experts = 128;
    moe.top_k = 8;
    moe.use_shared_expert = false;
    moe.router_aux_loss_coef = 0.001f;
    moe.capacity_factor = 1.25f;
    moe.decoder_sparse_step = 1;      // All layers are MoE
    moe.mlp_only_layers = {};         // No dense-only layers
    moe.norm_topk_prob = false;       // Qwen3 default (don't normalize after top-k)
    moe.moe_intermediate_size = 768;  // Per-expert intermediate size

    return ModelConfigBuilder()
        .architecture(ArchitectureType::MoE)
        .activation(ActivationType::SwiGLU)
        .hidden_size(2048)
        .intermediate_size(6144)       // Dense MLP intermediate (not used in MoE layers)
        .vocab_size(151936)
        .num_layers(24)
        .num_query_heads(32)
        .num_kv_heads(4)
        .max_position_embeddings(32768)
        .rope_theta(10000.0f)
        .rms_norm_eps(1e-6f)
        .tied_embeddings(false)
        .use_qkv_bias(false)
        .qk_norm(true)                 // Qwen3 uses QK normalization
        .moe(moe)
        .dtype(dtype)
        .build();
}

/**
 * @brief Qwen3 MoE 30B-A3B configuration
 *
 * Larger Qwen3 MoE model with 128 experts.
 */
inline ModelConfig qwen3_moe_30b_a3b(ETensorDType dtype = ETensorDType::BF16) {
    MoEConfig moe;
    moe.num_experts = 128;
    moe.top_k = 8;
    moe.use_shared_expert = false;
    moe.router_aux_loss_coef = 0.001f;
    moe.capacity_factor = 1.25f;
    moe.decoder_sparse_step = 1;
    moe.mlp_only_layers = {};
    moe.norm_topk_prob = false;
    moe.moe_intermediate_size = 1024;

    return ModelConfigBuilder()
        .architecture(ArchitectureType::MoE)
        .activation(ActivationType::SwiGLU)
        .hidden_size(2560)
        .intermediate_size(8192)
        .vocab_size(151936)
        .num_layers(32)
        .num_query_heads(32)
        .num_kv_heads(4)
        .max_position_embeddings(32768)
        .rope_theta(10000.0f)
        .rms_norm_eps(1e-6f)
        .tied_embeddings(false)
        .use_qkv_bias(false)
        .qk_norm(true)
        .moe(moe)
        .dtype(dtype)
        .build();
}

/**
 * @brief Test configuration for hybrid model validation
 *
 * Small model for testing heterogeneous layer support.
 */
inline ModelConfig hybrid_test(ETensorDType dtype = ETensorDType::BF16) {
    MoEConfig moe;
    moe.num_experts = 4;
    moe.top_k = 2;
    moe.use_shared_expert = false;
    moe.router_aux_loss_coef = 0.01f;
    moe.capacity_factor = 1.25f;

    ModelConfig config = ModelConfigBuilder()
        .architecture(ArchitectureType::Hybrid)
        .activation(ActivationType::SwiGLU)
        .hidden_size(256)
        .intermediate_size(512)
        .vocab_size(1024)
        .num_layers(4)
        .num_query_heads(4)
        .num_kv_heads(2)
        .max_position_embeddings(512)
        .rope_theta(10000.0f)
        .rms_norm_eps(1e-5f)
        .tied_embeddings(true)
        .use_qkv_bias(false)
        .moe(moe)
        .dtype(dtype)
        .build();

    // Layer 0: Dense
    // Layer 1: MoE
    // Layer 2: Dense
    // Layer 3: MoE
    config.layer_overrides.push_back(LayerOverride::moe(1, 4, 2));
    config.layer_overrides.push_back(LayerOverride::moe(3, 4, 2));

    return config;
}

// ============================================================================
// LayerOverride Tests
// ============================================================================

TEST_CASE("LayerOverride convenience constructors", "[heterogeneous][config]") {
    SECTION("Dense layer override") {
        auto override = LayerOverride::dense(5);
        REQUIRE(override.layer_idx == 5);
        REQUIRE(override.block_type == BlockType::Dense);
        REQUIRE_FALSE(override.is_moe);
    }

    SECTION("MoE layer override") {
        auto override = LayerOverride::moe(3, 8, 2);
        REQUIRE(override.layer_idx == 3);
        REQUIRE(override.block_type == BlockType::MoE);
        REQUIRE(override.is_moe);
        REQUIRE(override.num_experts.value() == 8);
        REQUIRE(override.top_k.value() == 2);
    }

    SECTION("Switch MoE layer override") {
        auto override = LayerOverride::switch_moe(7, 128);
        REQUIRE(override.layer_idx == 7);
        REQUIRE(override.block_type == BlockType::SwitchMoE);
        REQUIRE(override.is_moe);
        REQUIRE(override.num_experts.value() == 128);
        REQUIRE(override.top_k.value() == 1);  // Switch = top-1
    }

    SECTION("Conv layer override") {
        auto override = LayerOverride::conv(2);
        REQUIRE(override.layer_idx == 2);
        REQUIRE(override.block_type == BlockType::Conv);
        REQUIRE_FALSE(override.is_moe);
    }
}

// ============================================================================
// ModelConfig Block Type Tests
// ============================================================================

TEST_CASE("ModelConfig block type queries", "[heterogeneous][config]") {
    SECTION("Dense architecture always returns Dense block type") {
        ModelConfig config;
        config.architecture = ArchitectureType::Dense;
        config.NumLayers = 12;

        for (int i = 0; i < config.NumLayers; ++i) {
            REQUIRE(config.get_block_type(i) == BlockType::Dense);
            REQUIRE_FALSE(config.is_layer_moe(i));
        }
    }

    SECTION("MoE architecture always returns MoE block type") {
        ModelConfig config;
        config.architecture = ArchitectureType::MoE;
        config.NumLayers = 12;

        for (int i = 0; i < config.NumLayers; ++i) {
            REQUIRE(config.get_block_type(i) == BlockType::MoE);
            REQUIRE(config.is_layer_moe(i));
        }
    }

    SECTION("Hybrid architecture respects layer overrides") {
        ModelConfig config;
        config.architecture = ArchitectureType::Hybrid;
        config.NumLayers = 8;

        // Set up hybrid pattern: Dense, MoE, Dense, MoE, ...
        config.layer_overrides.push_back(LayerOverride::moe(1, 8, 2));
        config.layer_overrides.push_back(LayerOverride::moe(3, 8, 2));
        config.layer_overrides.push_back(LayerOverride::moe(5, 8, 2));
        config.layer_overrides.push_back(LayerOverride::moe(7, 8, 2));

        // Check odd layers are MoE, even are Dense
        for (int i = 0; i < config.NumLayers; ++i) {
            if (i % 2 == 1) {
                REQUIRE(config.get_block_type(i) == BlockType::MoE);
                REQUIRE(config.is_layer_moe(i));
            } else {
                REQUIRE(config.get_block_type(i) == BlockType::Dense);
                REQUIRE_FALSE(config.is_layer_moe(i));
            }
        }
    }

    SECTION("get_top_k returns layer-specific value") {
        MoEConfig moe_cfg;
        moe_cfg.num_experts = 8;
        moe_cfg.top_k = 2;  // Default

        ModelConfig config;
        config.architecture = ArchitectureType::Hybrid;
        config.NumLayers = 4;
        config.moe_config = moe_cfg;

        // Layer 1 uses default top_k
        config.layer_overrides.push_back(LayerOverride::moe(1, 8, 2));
        // Layer 3 uses custom top_k
        auto override = LayerOverride::moe(3, 16, 4);
        config.layer_overrides.push_back(override);

        REQUIRE(config.get_top_k(0) == 2);  // Default (non-MoE layer)
        REQUIRE(config.get_top_k(1) == 2);  // Override specifies 2
        REQUIRE(config.get_top_k(3) == 4);  // Override specifies 4
    }
}

// ============================================================================
// Hybrid Model Preset Tests
// ============================================================================

TEST_CASE("Hybrid model presets", "[heterogeneous][presets]") {
    SECTION("hybrid_test preset configuration") {
        auto config = hybrid_test();

        REQUIRE(config.architecture == ArchitectureType::Hybrid);
        REQUIRE(config.NumLayers == 4);
        REQUIRE(config.HiddenSize == 256);
        REQUIRE(config.IntermediateSize == 512);

        // Check layer pattern
        REQUIRE(config.get_block_type(0) == BlockType::Dense);
        REQUIRE(config.get_block_type(1) == BlockType::MoE);
        REQUIRE(config.get_block_type(2) == BlockType::Dense);
        REQUIRE(config.get_block_type(3) == BlockType::MoE);

        REQUIRE(config.moe_config.has_value());
        REQUIRE(config.moe_config->num_experts == 4);
        REQUIRE(config.moe_config->top_k == 2);
    }
}

// ============================================================================
// Qwen3 MoE-style Configuration Tests
// ============================================================================

TEST_CASE("Qwen3 MoE layer pattern with decoder_sparse_step", "[heterogeneous][qwen3][config]") {
    SECTION("decoder_sparse_step=1 makes all layers MoE") {
        MoEConfig moe_cfg;
        moe_cfg.num_experts = 128;
        moe_cfg.top_k = 8;
        moe_cfg.decoder_sparse_step = 1;  // All layers MoE
        moe_cfg.mlp_only_layers = {};

        ModelConfig config;
        config.architecture = ArchitectureType::MoE;
        config.NumLayers = 24;
        config.moe_config = moe_cfg;

        // All layers should be MoE
        for (int i = 0; i < config.NumLayers; ++i) {
            REQUIRE(config.is_layer_moe(i));
        }
    }

    SECTION("decoder_sparse_step=2 makes alternating layers MoE") {
        MoEConfig moe_cfg;
        moe_cfg.num_experts = 8;
        moe_cfg.top_k = 2;
        moe_cfg.decoder_sparse_step = 2;  // Every other layer MoE
        moe_cfg.mlp_only_layers = {};

        ModelConfig config;
        config.architecture = ArchitectureType::MoE;
        config.NumLayers = 8;
        config.moe_config = moe_cfg;

        // Pattern: (layer_idx + 1) % 2 == 0 means layers 1, 3, 5, 7 are MoE
        REQUIRE_FALSE(config.is_layer_moe(0));  // (0+1) % 2 = 1 != 0
        REQUIRE(config.is_layer_moe(1));         // (1+1) % 2 = 0
        REQUIRE_FALSE(config.is_layer_moe(2));   // (2+1) % 2 = 1 != 0
        REQUIRE(config.is_layer_moe(3));         // (3+1) % 2 = 0
        REQUIRE_FALSE(config.is_layer_moe(4));
        REQUIRE(config.is_layer_moe(5));
        REQUIRE_FALSE(config.is_layer_moe(6));
        REQUIRE(config.is_layer_moe(7));
    }

    SECTION("mlp_only_layers overrides decoder_sparse_step") {
        MoEConfig moe_cfg;
        moe_cfg.num_experts = 128;
        moe_cfg.top_k = 8;
        moe_cfg.decoder_sparse_step = 1;  // All layers would be MoE
        moe_cfg.mlp_only_layers = {0, 5, 10};  // But these are forced to dense

        ModelConfig config;
        config.architecture = ArchitectureType::MoE;
        config.NumLayers = 12;
        config.moe_config = moe_cfg;

        // Layers in mlp_only_layers should be dense
        REQUIRE_FALSE(config.is_layer_moe(0));
        REQUIRE(config.is_layer_moe(1));
        REQUIRE(config.is_layer_moe(2));
        REQUIRE(config.is_layer_moe(3));
        REQUIRE(config.is_layer_moe(4));
        REQUIRE_FALSE(config.is_layer_moe(5));
        REQUIRE(config.is_layer_moe(6));
        REQUIRE(config.is_layer_moe(7));
        REQUIRE(config.is_layer_moe(8));
        REQUIRE(config.is_layer_moe(9));
        REQUIRE_FALSE(config.is_layer_moe(10));
        REQUIRE(config.is_layer_moe(11));
    }

    SECTION("layer_overrides take highest priority") {
        MoEConfig moe_cfg;
        moe_cfg.num_experts = 128;
        moe_cfg.top_k = 8;
        moe_cfg.decoder_sparse_step = 1;
        moe_cfg.mlp_only_layers = {0, 1};  // These would be dense

        ModelConfig config;
        config.architecture = ArchitectureType::MoE;
        config.NumLayers = 4;
        config.moe_config = moe_cfg;

        // But layer override forces layer 0 to be MoE
        config.layer_overrides.push_back(LayerOverride::moe(0, 128, 8));

        // Layer 0 is MoE (override wins over mlp_only_layers)
        REQUIRE(config.is_layer_moe(0));
        // Layer 1 is Dense (mlp_only_layers)
        REQUIRE_FALSE(config.is_layer_moe(1));
    }
}

TEST_CASE("Qwen3 MoE preset configurations", "[heterogeneous][qwen3][presets]") {
    SECTION("qwen3_moe_15b_a2b preset") {
        auto config = qwen3_moe_15b_a2b();

        REQUIRE(config.architecture == ArchitectureType::MoE);
        REQUIRE(config.NumLayers == 24);
        REQUIRE(config.HiddenSize == 2048);
        REQUIRE(config.NumQueryHeads == 32);
        REQUIRE(config.NumKeyValHeads == 4);
        REQUIRE(config.use_qk_norm == true);  // Qwen3 uses QK norm

        REQUIRE(config.moe_config.has_value());
        auto& moe = config.moe_config.value();
        REQUIRE(moe.num_experts == 128);
        REQUIRE(moe.top_k == 8);
        REQUIRE(moe.decoder_sparse_step == 1);
        REQUIRE(moe.mlp_only_layers.empty());
        REQUIRE(moe.moe_intermediate_size == 768);
        REQUIRE(moe.norm_topk_prob == false);

        // All layers should be MoE
        for (int i = 0; i < config.NumLayers; ++i) {
            REQUIRE(config.is_layer_moe(i));
        }
    }

    SECTION("qwen3_moe_30b_a3b preset") {
        auto config = qwen3_moe_30b_a3b();

        REQUIRE(config.architecture == ArchitectureType::MoE);
        REQUIRE(config.NumLayers == 32);
        REQUIRE(config.HiddenSize == 2560);
        REQUIRE(config.use_qk_norm == true);

        REQUIRE(config.moe_config.has_value());
        auto& moe = config.moe_config.value();
        REQUIRE(moe.num_experts == 128);
        REQUIRE(moe.top_k == 8);
        REQUIRE(moe.moe_intermediate_size == 1024);
    }
}

TEST_CASE("MoEConfig norm_topk_prob configuration", "[heterogeneous][qwen3][router]") {
    SECTION("Default norm_topk_prob is false") {
        MoEConfig moe;
        REQUIRE(moe.norm_topk_prob == false);
    }

    SECTION("norm_topk_prob can be enabled") {
        MoEConfig moe;
        moe.norm_topk_prob = true;
        REQUIRE(moe.norm_topk_prob == true);
    }

    SECTION("moe_intermediate_size default is 0") {
        MoEConfig moe;
        REQUIRE(moe.moe_intermediate_size == 0);
    }
}

// ============================================================================
// HeterogeneousBlock Tests
// ============================================================================

TEST_CASE("HeterogeneousBlock type identification", "[heterogeneous][block]") {
    SECTION("Dense block identification") {
        DefaultDenseBlock::Config cfg;
        cfg.hidden_size = 256;
        cfg.num_query_heads = 4;
        cfg.num_kv_heads = 2;
        cfg.head_size = 64;
        cfg.intermediate_size = 512;

        DefaultDenseBlock dense_block(cfg);
        DefaultHeterogeneousBlock hetero_block(dense_block);

        REQUIRE(hetero_block.is_dense());
        REQUIRE_FALSE(hetero_block.is_moe());
        REQUIRE(hetero_block.index() == 0);  // Dense is first in variant
    }

    SECTION("MoE block identification") {
        DefaultMoEBlock::Config cfg;
        cfg.hidden_size = 256;
        cfg.num_query_heads = 4;
        cfg.num_kv_heads = 2;
        cfg.head_size = 64;
        cfg.intermediate_size = 512;
        cfg.num_experts = 4;
        cfg.top_k = 2;

        DefaultMoEBlock moe_block(cfg);
        DefaultHeterogeneousBlock hetero_block(moe_block);

        REQUIRE_FALSE(hetero_block.is_dense());
        REQUIRE(hetero_block.is_moe());
        REQUIRE(hetero_block.index() == 1);  // MoE is second in variant
    }
}

// ============================================================================
// Model Factory Tests
// ============================================================================

TEST_CASE("ModelFactory hybrid model creation", "[heterogeneous][factory]") {
    SECTION("Factory rejects empty layer_overrides for Hybrid") {
        ModelConfig config;
        config.architecture = ArchitectureType::Hybrid;
        config.NumLayers = 4;
        // No layer_overrides set

        ModelOptions options;

        REQUIRE_THROWS_AS(
            ModelFactory::create(config, options, 0, 1, nullptr),
            std::invalid_argument
        );
    }

    SECTION("Factory creates hybrid model with valid config") {
        auto config = hybrid_test();
        ModelOptions options;

        // This should not throw
        auto model = ModelFactory::create(config, options, 0, 1, nullptr);
        REQUIRE(model != nullptr);
        REQUIRE(model->model_type() == "heterogeneous");
    }
}

// ============================================================================
// Integration Tests (require CUDA device)
// ============================================================================

TEST_CASE("Heterogeneous model block pattern", "[heterogeneous][cuda]") {
    auto config = hybrid_test();
    ModelOptions options;
    options.model_dtype = ETensorDType::BF16;
    options.matmul_dtype = ETensorDType::BF16;

    auto model_ptr = ModelFactory::create(config, options, 0, 1, nullptr);
    REQUIRE(model_ptr != nullptr);

    // Cast to heterogeneous model to access layer info
    auto* hetero_model = dynamic_cast<DefaultHeterogeneousModel*>(model_ptr.get());
    REQUIRE(hetero_model != nullptr);

    // Verify layer pattern
    REQUIRE(hetero_model->is_layer_dense(0));
    REQUIRE(hetero_model->is_layer_moe(1));
    REQUIRE(hetero_model->is_layer_dense(2));
    REQUIRE(hetero_model->is_layer_moe(3));

    REQUIRE(hetero_model->num_layers() == 4);
}
