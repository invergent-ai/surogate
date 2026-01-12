// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_ALLOCATION_H
#define LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_ALLOCATION_H

#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "kernels/kernels.h"
#include "modules/weights/weight_manager.h"

namespace modules {

template<typename Block>
ModularWeightManager<Block>::ModularWeightManager(const Config& config, TensorAllocator& allocator)
    : mConfig(config), mAllocator(&allocator) {

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&mDeviceProp, dev));

    // ZeRO-3/FSDP style: don't keep full per-layer weights, only a sharded master + 2 gathered buffers.
    mStreamWeights = (config.shard_weights && config.num_shards > 1);

    const bool sharded_master = (config.num_shards > 1);
    const bool separate_master_storage =
        mStreamWeights ||
        config.offload_master ||
        (config.master_dtype != config.model_dtype) ||
        (config.master_dtype != config.matmul_dtype);

    // ------------------------------------------------------------------------
    // Allocate WORK weights (full, used for forward/backward)
    // ------------------------------------------------------------------------
    // QLoRA mode: skip all weight allocation - weights are provided externally
    if (config.skip_block_allocation) {
        // All weights (including embeddings, final_norm, lm_head) will be provided via
        // set_weight_provider(), set_embeddings_provider(), etc.
        // No allocation needed here.
    } else if (mStreamWeights) {
        // Non-block work weights are persistent full tensors (used every step).
        allocate_non_block_weights(mWorkNonBlock, config.model_dtype, /*on_host=*/false, /*sharded=*/false);

        // Full block work weights are gathered into double-buffered prefetch slots.
        for (int i = 0; i < 2; ++i) {
            allocate_block_weights(mPrefetchBuffer[i], config.matmul_dtype, config.model_dtype, /*on_host=*/false, /*sharded=*/false);
            CUDA_CHECK(cudaEventCreate(&mPrefetchStatus[i].done_event));
            CUDA_CHECK(cudaEventRecord(mPrefetchStatus[i].done_event, 0));
        }
    } else {
        // Keep full per-layer work weights on device (like legacy ZeRO-1/2 path).
        mWorkBlocks.resize(config.num_layers);
        for (int i = 0; i < config.num_layers; ++i) {
            allocate_block_weights(mWorkBlocks[i], config.matmul_dtype, config.model_dtype, /*on_host=*/false, /*sharded=*/false);
        }
        allocate_non_block_weights(mWorkNonBlock, config.model_dtype, /*on_host=*/false, /*sharded=*/false);

        // One status/event per layer so we can prefetch (all-gather) the next layer safely.
        mLayerStatus.resize(config.num_layers);
        for (int i = 0; i < config.num_layers; ++i) {
            CUDA_CHECK(cudaEventCreate(&mLayerStatus[i].done_event));
            CUDA_CHECK(cudaEventRecord(mLayerStatus[i].done_event, 0));
            mLayerStatus[i].layer_idx = i;
        }
    }

    // ------------------------------------------------------------------------
    // Allocate MASTER weights (sharded, used for optimizer update/checkpointing)
    // ------------------------------------------------------------------------
    // QLoRA mode: skip master block allocation too
    if (config.skip_block_allocation) {
        // Master non-block weights only (for embeddings/final_norm checkpointing if needed)
        // Note: For QLoRA, base model weights are frozen and stored in QLoRAWeightsManager,
        // so we don't need master storage for them.
    } else if (separate_master_storage) {
        mMasterBlocks.resize(config.num_layers);
        for (int i = 0; i < config.num_layers; ++i) {
            allocate_block_weights(mMasterBlocks[i], config.master_dtype, config.master_dtype, config.offload_master, sharded_master);
        }
        allocate_non_block_weights(mMasterNonBlock, config.master_dtype, config.offload_master, sharded_master);
    } else {
        // Master weights are sharded views into the work weights.
        mMasterBlocks.resize(config.num_layers);
        for (int i = 0; i < config.num_layers; ++i) {
            auto shard = [&](const Tensor& t) -> Tensor {
                if (!sharded_master) return t;
                return static_cast<Tensor>(shard_view(t, config.shard_idx, config.num_shards));
            };

            auto& src = mWorkBlocks[i];
            auto& dst = mMasterBlocks[i];

            dst.ln1.weight = shard(src.ln1.weight);
            dst.attention.qkv_weight = shard(src.attention.qkv_weight);
            if (src.attention.qkv_bias.has_value()) {
                dst.attention.qkv_bias = shard(src.attention.qkv_bias.value());
            }
            dst.attention.out_weight = shard(src.attention.out_weight);
            dst.ln2.weight = shard(src.ln2.weight);
            if constexpr (has_mlp_weights<BlockWeights>::value) {
                dst.mlp_up_weight = shard(src.mlp_up_weight);
                dst.mlp_down_weight = shard(src.mlp_down_weight);
            }
        }

        auto shard_nb = [&](const Tensor& t) -> Tensor {
            if (!sharded_master) return t;
            return static_cast<Tensor>(shard_view(t, config.shard_idx, config.num_shards));
        };
        mMasterNonBlock.embeddings = shard_nb(mWorkNonBlock.embeddings);
        mMasterNonBlock.final_norm_weight = shard_nb(mWorkNonBlock.final_norm_weight);
        if (config.tied_embeddings) {
            mMasterNonBlock.lm_head = mMasterNonBlock.embeddings;
        } else {
            mMasterNonBlock.lm_head = shard_nb(mWorkNonBlock.lm_head);
        }
    }

    // ------------------------------------------------------------------------
    // Master shard views (TensorShard wrappers with global shapes)
    // ------------------------------------------------------------------------
    // QLoRA mode: skip shard views - non-block weights are provided externally
    if (!config.skip_block_allocation) {
        if (sharded_master) {
            mMasterEmbeddingsShardView = TensorShard(mMasterNonBlock.embeddings, config.shard_idx, config.num_shards,
                                                    std::vector<long>{config.vocab_size, config.hidden_size});
            mMasterFinalNormShardView = TensorShard(mMasterNonBlock.final_norm_weight, config.shard_idx, config.num_shards,
                                                   std::vector<long>{config.hidden_size});
            if (config.tied_embeddings) {
                mMasterLMHeadShardView = mMasterEmbeddingsShardView;
            } else {
                mMasterLMHeadShardView = TensorShard(mMasterNonBlock.lm_head, config.shard_idx, config.num_shards,
                                                     std::vector<long>{config.vocab_size, config.hidden_size});
            }
        } else {
            mMasterEmbeddingsShardView = TensorShard(mMasterNonBlock.embeddings);
            mMasterLMHeadShardView = TensorShard(mMasterNonBlock.lm_head);
            mMasterFinalNormShardView = TensorShard(mMasterNonBlock.final_norm_weight);
        }
    }

    // ------------------------------------------------------------------------
    // Allocate abs-max/scale storage and wire Stats pointers (master + work)
    // ------------------------------------------------------------------------
    mAbsMaxes = allocator.allocate(ETensorDType::FP32, "abs_maxes_modular", EAllocationType::ON_DEVICE,
                                   {6 + config.num_layers * 14});
    float* abs_maxes = mAbsMaxes.template get<float>();

    auto wire_nonblock = [&](NonBlockWeights& nb) {
        nb.embeddings.Stats = abs_maxes + 0;
        nb.final_norm_weight.Stats = abs_maxes + 2;
        nb.lm_head.Stats = abs_maxes + 4;
    };

    auto wire_block = [&](BlockWeights& b, int layer_idx) {
        float* a = abs_maxes + 6 + layer_idx * 14;
        b.attention.qkv_weight.Stats = a + 0;
        b.attention.out_weight.Stats = a + 2;
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            b.mlp_up_weight.Stats = a + 4;
            b.mlp_down_weight.Stats = a + 6;
        }
        if (b.attention.qkv_bias.has_value()) {
            b.attention.qkv_bias->Stats = a + 8;
        }
        b.ln1.weight.Stats = a + 10;
        b.ln2.weight.Stats = a + 12;
    };

    if (!config.skip_block_allocation) {
        wire_nonblock(mMasterNonBlock);
        for (int i = 0; i < config.num_layers; ++i) {
            wire_block(mMasterBlocks[i], i);
        }
    }

    // For single-GPU runs, set Stats directly on the full work weights too (no gather needed for propagation).
    if (config.num_shards == 1 && !config.skip_block_allocation) {
        wire_nonblock(mWorkNonBlock);
        if (!mStreamWeights) {
            for (int i = 0; i < config.num_layers; ++i) {
                wire_block(mWorkBlocks[i], i);
            }
        }
    }

    // ------------------------------------------------------------------------
    // Status/events for non-block gather
    // ------------------------------------------------------------------------
    CUDA_CHECK(cudaEventCreate(&mEmbeddingsStatus.done_event));
    CUDA_CHECK(cudaEventCreate(&mFinalNormStatus.done_event));
    CUDA_CHECK(cudaEventCreate(&mLMHeadStatus.done_event));
    CUDA_CHECK(cudaEventRecord(mEmbeddingsStatus.done_event, 0));
    CUDA_CHECK(cudaEventRecord(mFinalNormStatus.done_event, 0));
    CUDA_CHECK(cudaEventRecord(mLMHeadStatus.done_event, 0));

    // ------------------------------------------------------------------------
    // Optimizer staging buffers for offloaded master weights (device copies)
    // ------------------------------------------------------------------------
    // QLoRA mode: skip staging buffer allocation - base weights are frozen and stored externally
    if (config.offload_master && !config.use_zero_copy && !config.skip_block_allocation) {
        for (int i = 0; i < 2; ++i) {
            allocate_block_weights(mMasterBuffer[i], config.master_dtype, config.master_dtype, /*on_host=*/false, /*sharded=*/sharded_master);
            CUDA_CHECK(cudaEventCreate(&mMasterStatus[i].done_event));
            CUDA_CHECK(cudaEventRecord(mMasterStatus[i].done_event, 0));
        }
        allocate_non_block_weights(mMasterNonBlockDevice, config.master_dtype, /*on_host=*/false, /*sharded=*/sharded_master);
    }

    // ------------------------------------------------------------------------
    // Persistent quantized weights storage (when persistent_quants=true)
    // ------------------------------------------------------------------------
    if (config.persistent_quants && !config.skip_block_allocation) {
        const bool quants_on_host = config.offload_quants;
        mQuantBlocks.resize(config.num_layers);
        mQuantBlockVersion.resize(config.num_layers, -1);
        for (int i = 0; i < config.num_layers; ++i) {
            allocate_block_weights(mQuantBlocks[i], config.matmul_dtype, config.model_dtype, quants_on_host, sharded_master);
        }

        if (config.offload_quants && !config.use_zero_copy) {
            for (int i = 0; i < 2; ++i) {
                allocate_block_weights(mQuantBuffer[i], config.matmul_dtype, config.model_dtype, /*on_host=*/false, /*sharded=*/false);
                CUDA_CHECK(cudaEventCreate(&mQuantStatus[i].done_event));
                CUDA_CHECK(cudaEventRecord(mQuantStatus[i].done_event, 0));
            }
        }
    }

    // ------------------------------------------------------------------------
    // FP8 forward weight cache (when enable_fp8_forward=true)
    // ------------------------------------------------------------------------
    if (config.enable_fp8_forward) {
        const auto& bc = config.block_config;
        const long C = bc.hidden_size;
        const long Hq = bc.num_query_heads;
        const long Hkv = bc.num_kv_heads;
        const long Hs = bc.head_size;
        const long D = bc.intermediate_size;
        const long QKV_C = (Hq + 2 * Hkv) * Hs;

        // Stats buffer: 4 weights * 2 floats (abs_max, scale) = 8 floats
        mFP8WeightStats = mAllocator->allocate(ETensorDType::FP32, "fp8_weight_stats", EAllocationType::ON_DEVICE, {8});

        // Allocate FP8 weight buffers
        mFP8WeightCache.qkv_weight = mAllocator->allocate(ETensorDType::FP8_E4M3, "fp8_qkv_weight", EAllocationType::ON_DEVICE, {QKV_C, C});
        mFP8WeightCache.o_weight = mAllocator->allocate(ETensorDType::FP8_E4M3, "fp8_o_weight", EAllocationType::ON_DEVICE, {C, Hq * Hs});
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            mFP8WeightCache.mlp_up_weight = mAllocator->allocate(ETensorDType::FP8_E4M3, "fp8_mlp_up_weight", EAllocationType::ON_DEVICE, {2 * D, C});
            mFP8WeightCache.mlp_down_weight = mAllocator->allocate(ETensorDType::FP8_E4M3, "fp8_mlp_down_weight", EAllocationType::ON_DEVICE, {C, D});
        }

        // Assign Stats pointers: each weight gets 2 floats (abs_max at [0], scale at [1])
        float* stats = mFP8WeightStats.get<float>();
        mFP8WeightCache.qkv_weight.Stats = stats;
        mFP8WeightCache.o_weight.Stats = stats + 2;
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            mFP8WeightCache.mlp_up_weight.Stats = stats + 4;
            mFP8WeightCache.mlp_down_weight.Stats = stats + 6;
        }
    }

    // ------------------------------------------------------------------------
    // FP4 forward weight cache (when enable_fp4_forward=true)
    // ------------------------------------------------------------------------
    if (config.enable_fp4_forward) {
        const auto& bc = config.block_config;
        const long C = bc.hidden_size;
        const long Hq = bc.num_query_heads;
        const long Hkv = bc.num_kv_heads;
        const long Hs = bc.head_size;
        const long D = bc.intermediate_size;
        const long QKV_C = (Hq + 2 * Hkv) * Hs;

        mFP4WeightAmax = mAllocator->allocate(ETensorDType::FP32, "fp4_weight_amax", EAllocationType::ON_DEVICE, {4});
        mFP4WeightAmaxT = mAllocator->allocate(ETensorDType::FP32, "fp4_weight_amax_t", EAllocationType::ON_DEVICE, {4});

        auto cutlass_scale_size = [](long rows, long cols) -> long {
            constexpr int kBlockSize = 16;
            constexpr int kTileDim = 128;
            long num_scale_cols = (cols + kBlockSize - 1) / kBlockSize;
            long aligned_rows = ((rows + kTileDim - 1) / kTileDim) * kTileDim;
            long aligned_cols = ((num_scale_cols + 3) / 4) * 4;
            return aligned_rows * aligned_cols;
        };

        // QKV weight: (QKV_C, C)
        long qkv_scale_size = cutlass_scale_size(QKV_C, C);
        mFP4WeightCache.qkv_weight.data = mAllocator->allocate(ETensorDType::BYTE, "fp4_qkv_data", EAllocationType::ON_DEVICE, {QKV_C, C / 2});
        mFP4WeightCache.qkv_weight.scales = mAllocator->allocate(ETensorDType::BYTE, "fp4_qkv_scales", EAllocationType::ON_DEVICE, {qkv_scale_size});

        // O weight: (C, Hq*Hs)
        long o_scale_size = cutlass_scale_size(C, Hq * Hs);
        mFP4WeightCache.o_weight.data = mAllocator->allocate(ETensorDType::BYTE, "fp4_o_data", EAllocationType::ON_DEVICE, {C, (Hq * Hs) / 2});
        mFP4WeightCache.o_weight.scales = mAllocator->allocate(ETensorDType::BYTE, "fp4_o_scales", EAllocationType::ON_DEVICE, {o_scale_size});

        if constexpr (has_mlp_weights<BlockWeights>::value) {
            // MLP up weight: (2*D, C)
            long mlp_up_scale_size = cutlass_scale_size(2 * D, C);
            mFP4WeightCache.mlp_up_weight.data = mAllocator->allocate(ETensorDType::BYTE, "fp4_mlp_up_data", EAllocationType::ON_DEVICE, {2 * D, C / 2});
            mFP4WeightCache.mlp_up_weight.scales = mAllocator->allocate(ETensorDType::BYTE, "fp4_mlp_up_scales", EAllocationType::ON_DEVICE, {mlp_up_scale_size});

            // MLP down weight: (C, D)
            long mlp_down_scale_size = cutlass_scale_size(C, D);
            mFP4WeightCache.mlp_down_weight.data = mAllocator->allocate(ETensorDType::BYTE, "fp4_mlp_down_data", EAllocationType::ON_DEVICE, {C, D / 2});
            mFP4WeightCache.mlp_down_weight.scales = mAllocator->allocate(ETensorDType::BYTE, "fp4_mlp_down_scales", EAllocationType::ON_DEVICE, {mlp_down_scale_size});
        }
    }
}

template<typename Block>
ModularWeightManager<Block>::~ModularWeightManager() {
    for (int i = 0; i < 2; ++i) {
        if (mPrefetchStatus[i].done_event) {
            cudaEventDestroy(mPrefetchStatus[i].done_event);
        }
        if (mMasterStatus[i].done_event) {
            cudaEventDestroy(mMasterStatus[i].done_event);
        }
        if (mQuantStatus[i].done_event) {
            cudaEventDestroy(mQuantStatus[i].done_event);
        }
    }
    if (mEmbeddingsStatus.done_event) cudaEventDestroy(mEmbeddingsStatus.done_event);
    if (mFinalNormStatus.done_event) cudaEventDestroy(mFinalNormStatus.done_event);
    if (mLMHeadStatus.done_event) cudaEventDestroy(mLMHeadStatus.done_event);
}

template<typename Block>
void ModularWeightManager<Block>::maybe_enable_fp4_persistent_cache(bool weights_static) {
    if (!mConfig.enable_fp4_forward) return;

    // Already enabled: keep it enabled; if any caller marks weights as static, treat as static.
    if (mFP4PersistentCacheEnabled) {
        mFP4PersistentCacheStatic = mFP4PersistentCacheStatic || weights_static;
        return;
    }

    // Enable only on Blackwell GPUs by default
    const int sm_version = mDeviceProp.major * 10 + mDeviceProp.minor;
    if (sm_version < 100) {
        return;
    }

    // Don't enable for ZeRO-3/FSDP weight streaming: persistent caches would defeat the purpose.
    if (mStreamWeights) {
        return;
    }

    const auto& bc = mConfig.block_config;
    const long C = bc.hidden_size;
    const long Hq = bc.num_query_heads;
    const long Hkv = bc.num_kv_heads;
    const long Hs = bc.head_size;
    const long D = bc.intermediate_size;
    const long QKV_C = (Hq + 2 * Hkv) * Hs;

    // Helper to compute CUTLASS scale size (matches compute_nvfp4_cutlass_scale_size()).
    auto cutlass_scale_size = [](long rows, long cols) -> long {
        constexpr int kBlockSize = 16;
        constexpr int kTileDim = 128;
        long num_scale_cols = (cols + kBlockSize - 1) / kBlockSize;
        long aligned_rows = ((rows + kTileDim - 1) / kTileDim) * kTileDim;
        long aligned_cols = ((num_scale_cols + 3) / 4) * 4;
        return aligned_rows * aligned_cols;
    };

    const long L = mConfig.num_layers;

    // Amax storage: 4 floats per layer [qkv, o, mlp_up, mlp_down] - forward cache
    mFP4WeightAmaxAll = mAllocator->allocate(
        ETensorDType::FP32, "fp4_weight_amax_all", EAllocationType::ON_DEVICE, {L * 4});
    // Separate amax storage for transposed cache (needed when forward uses 4/6 but transposed uses standard)
    mFP4WeightAmaxAllT = mAllocator->allocate(
        ETensorDType::FP32, "fp4_weight_amax_all_t", EAllocationType::ON_DEVICE, {L * 4});

    // Forward (W) caches
    mFP4WeightDataAll[0] = mAllocator->allocate(
        ETensorDType::BYTE, "fp4_qkv_data_all", EAllocationType::ON_DEVICE, {L * QKV_C, C / 2});
    mFP4WeightScalesAll[0] = mAllocator->allocate(
        ETensorDType::BYTE, "fp4_qkv_scales_all", EAllocationType::ON_DEVICE, {L * cutlass_scale_size(QKV_C, C)});

    mFP4WeightDataAll[1] = mAllocator->allocate(
        ETensorDType::BYTE, "fp4_o_data_all", EAllocationType::ON_DEVICE, {L * C, (Hq * Hs) / 2});
    mFP4WeightScalesAll[1] = mAllocator->allocate(
        ETensorDType::BYTE, "fp4_o_scales_all", EAllocationType::ON_DEVICE, {L * cutlass_scale_size(C, Hq * Hs)});

    if constexpr (has_mlp_weights<BlockWeights>::value) {
        mFP4WeightDataAll[2] = mAllocator->allocate(
            ETensorDType::BYTE, "fp4_mlp_up_data_all", EAllocationType::ON_DEVICE, {L * (2 * D), C / 2});
        mFP4WeightScalesAll[2] = mAllocator->allocate(
            ETensorDType::BYTE, "fp4_mlp_up_scales_all", EAllocationType::ON_DEVICE, {L * cutlass_scale_size(2 * D, C)});

        mFP4WeightDataAll[3] = mAllocator->allocate(
            ETensorDType::BYTE, "fp4_mlp_down_data_all", EAllocationType::ON_DEVICE, {L * C, D / 2});
        mFP4WeightScalesAll[3] = mAllocator->allocate(
            ETensorDType::BYTE, "fp4_mlp_down_scales_all", EAllocationType::ON_DEVICE, {L * cutlass_scale_size(C, D)});
    }

    // Transposed (W^T) caches for dgrad
    mFP4WeightDataAllT[0] = mAllocator->allocate(
        ETensorDType::BYTE, "fp4_qkv_data_t_all", EAllocationType::ON_DEVICE, {L * C, QKV_C / 2});
    mFP4WeightScalesAllT[0] = mAllocator->allocate(
        ETensorDType::BYTE, "fp4_qkv_scales_t_all", EAllocationType::ON_DEVICE, {L * cutlass_scale_size(C, QKV_C)});

    mFP4WeightDataAllT[1] = mAllocator->allocate(
        ETensorDType::BYTE, "fp4_o_data_t_all", EAllocationType::ON_DEVICE, {L * (Hq * Hs), C / 2});
    mFP4WeightScalesAllT[1] = mAllocator->allocate(
        ETensorDType::BYTE, "fp4_o_scales_t_all", EAllocationType::ON_DEVICE, {L * cutlass_scale_size(Hq * Hs, C)});

    if constexpr (has_mlp_weights<BlockWeights>::value) {
        mFP4WeightDataAllT[2] = mAllocator->allocate(
            ETensorDType::BYTE, "fp4_mlp_up_data_t_all", EAllocationType::ON_DEVICE, {L * C, (2 * D) / 2});
        mFP4WeightScalesAllT[2] = mAllocator->allocate(
            ETensorDType::BYTE, "fp4_mlp_up_scales_t_all", EAllocationType::ON_DEVICE, {L * cutlass_scale_size(C, 2 * D)});

        mFP4WeightDataAllT[3] = mAllocator->allocate(
            ETensorDType::BYTE, "fp4_mlp_down_data_t_all", EAllocationType::ON_DEVICE, {L * D, C / 2});
        mFP4WeightScalesAllT[3] = mAllocator->allocate(
            ETensorDType::BYTE, "fp4_mlp_down_scales_t_all", EAllocationType::ON_DEVICE, {L * cutlass_scale_size(D, C)});
    }

    mFP4PersistentCacheVersion.assign((std::size_t)L, -1);
    mFP4PersistentCacheEnabled = true;
    mFP4PersistentCacheStatic = weights_static;

    if (mConfig.shard_idx == 0) {
        std::fprintf(
            stderr,
            "FP4 persistent weight cache enabled (SM%d): caching FP4 weights for %ld layers (forward + dgrad, %s).\n",
            sm_version,
            L,
            weights_static ? "static" : "versioned");
    }
}

// Block weight allocation - handles both dense and MoE blocks
template<typename Block>
void ModularWeightManager<Block>::allocate_block_weights(
    BlockWeights& block,
    ETensorDType matmul_dtype,
    ETensorDType other_dtype,
    bool on_host,
    bool sharded) {

    auto kind = on_host ? mConfig.offload_alloc : EAllocationType::ON_DEVICE;
    const auto& cfg = mConfig.block_config;
    long C = cfg.hidden_size;
    long HS = cfg.head_size;
    long HQ = cfg.num_query_heads;
    long HKV = cfg.num_kv_heads;
    long qkv_channels = HS * (HQ + 2 * HKV);

    auto alloc = [&](ETensorDType dtype, const char* name, const std::vector<long>& shape) -> Tensor {
        if (!sharded) {
            return mAllocator->allocate(dtype, name, kind, shape);
        }
        TensorShard shard_t = mAllocator->allocate_shard(dtype, mConfig.shard_idx, mConfig.num_shards, name, shape, kind);
        return static_cast<Tensor>(shard_t);
    };

    // norm weights + biases stay in model dtype; matmul weights may be FP8.
    block.ln1.weight = alloc(other_dtype, "ln1_w", {C});

    // Attention weights
    block.attention.qkv_weight = alloc(matmul_dtype, "attn_qkv_w", {qkv_channels, C});
    if (cfg.use_qkv_bias) {
        block.attention.qkv_bias = alloc(other_dtype, "attn_qkv_b", {qkv_channels});
    }
    block.attention.out_weight = alloc(matmul_dtype, "attn_out_w", {C, HS * HQ});
    // QK normalization weights (Qwen3-style) - only allocate if the Weights struct has these fields
    using AttentionWeightsType = std::decay_t<decltype(block.attention)>;
    if constexpr (has_qk_norm_weights<AttentionWeightsType>::value) {
        const bool use_qk_norm = [&]() -> bool {
            if constexpr (requires { cfg.use_qk_norm; }) return cfg.use_qk_norm;
            return false;
        }();
        if (use_qk_norm) {
            block.attention.q_norm_weight = alloc(other_dtype, "attn_q_norm_w", {HS});
            block.attention.k_norm_weight = alloc(other_dtype, "attn_k_norm_w", {HS});
        } else {
            block.attention.q_norm_weight.reset();
            block.attention.k_norm_weight.reset();
        }
    }

    // LN2 weights
    block.ln2.weight = alloc(other_dtype, "ln2_w", {C});

    // MLP weights (only for dense blocks)
    if constexpr (has_mlp_weights<BlockWeights>::value) {
        long D = cfg.intermediate_size;
        block.mlp_up_weight = alloc(matmul_dtype, "mlp_up_w", {2 * D, C});
        block.mlp_down_weight = alloc(matmul_dtype, "mlp_down_w", {C, D});
    }

    // MoE-specific weight allocation (router, experts) - only for MoE blocks
    if constexpr (has_moe_weights<BlockWeights>::value) {
        int num_experts = cfg.num_experts;
        long D = cfg.intermediate_size;

        // Router gate: (num_experts, hidden_size)
        block.router.gate = alloc(matmul_dtype, "router_gate_w", {num_experts, C});

        // Use batched layout for grouped GEMM efficiency
        block.experts.use_batched = true;

        // Batched expert weights
        block.experts.gate_up_proj = alloc(matmul_dtype, "experts_gate_up_w", {num_experts, 2 * D, C});
        block.experts.down_proj = alloc(matmul_dtype, "experts_down_w", {num_experts, C, D});

        // Shared expert (if configured)
        if (cfg.use_shared_expert) {
            int shared_D = cfg.shared_expert_intermediate > 0 ?
                           cfg.shared_expert_intermediate : static_cast<int>(D);
            block.shared_expert.emplace();
            block.shared_expert->gate_proj = alloc(matmul_dtype, "shared_expert_gate_w", {shared_D, C});
            block.shared_expert->up_proj = alloc(matmul_dtype, "shared_expert_up_w", {shared_D, C});
            block.shared_expert->down_proj = alloc(matmul_dtype, "shared_expert_down_w", {C, shared_D});
        }
    }
}

template<typename Block>
void ModularWeightManager<Block>::allocate_non_block_weights(
    NonBlockWeights& weights,
    ETensorDType dtype,
    bool on_host,
    bool sharded) {

    auto kind = on_host ? mConfig.offload_alloc : EAllocationType::ON_DEVICE;
    long V = mConfig.vocab_size;
    long C = mConfig.hidden_size;

    auto alloc = [&](const char* name, const std::vector<long>& shape) -> Tensor {
        if (!sharded) {
            return mAllocator->allocate(dtype, name, kind, shape);
        }
        TensorShard shard_t = mAllocator->allocate_shard(dtype, mConfig.shard_idx, mConfig.num_shards, name, shape, kind);
        return static_cast<Tensor>(shard_t);
    };

    weights.embeddings = alloc("embeddings", {V, C});
    weights.final_norm_weight = alloc("final_norm", {C});

    if (mConfig.tied_embeddings) {
        weights.lm_head = weights.embeddings;
    } else {
        weights.lm_head = alloc("lm_head", {V, C});
    }
}

} // namespace modules

#endif // LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_ALLOCATION_H
