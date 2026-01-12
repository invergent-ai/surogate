// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_GATHER_H
#define LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_GATHER_H

#include <stdexcept>
#include <string>

#include "kernels/kernels.h"
#include "modules/weights/weight_manager_helpers.h"
#include "utilities/comm.h"

namespace modules {

template<typename Block>
void ModularWeightManager<Block>::gather_block(int layer_idx, NCCLCommunicator& comm, cudaStream_t fetch_stream) {
    // If external weight provider is set (QLoRA), skip gather - weights are provided on-demand
    if (mExternalWeightProvider) {
        (void)layer_idx;
        (void)comm;
        (void)fetch_stream;
        return;
    }

    auto convert_into = [&](const Tensor& src, Tensor dst) {
        if (!src.Data || !dst.Data || src.nelem() == 0) return;

        // Fast path: identical storage.
        if (src.Data == dst.Data && src.DType == dst.DType) return;

        // Same dtype: memcpy (H2D/D2D/D2H as needed).
        if (src.DType == dst.DType) {
            CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, dst.bytes(), cudaMemcpyDefault, fetch_stream));
            return;
        }

        // Dtype conversion / quantization requires device-accessible input.
        if (src.Device == -1 && !mConfig.use_zero_copy) {
            throw std::runtime_error("ModularWeightManager::gather_block: dtype conversion from offloaded master weights requires --use-zero-copy");
        }

        if (dst.DType == ETensorDType::FP8_E4M3 || dst.DType == ETensorDType::FP8_E5M2 || dst.DType == ETensorDType::INT8) {
            if (!src.Stats) {
                throw std::runtime_error("ModularWeightManager::gather_block: FP8/INT8 gather requires Stats (abs_max/scale) on the source tensor");
            }
            quantize_with_abs_max(dst, /*scale_ptr=*/src.Stats + 1, src, /*abs_max=*/src.Stats, (long)src.nelem(), mDeviceProp, fetch_stream);
            return;
        }

        if (src.DType == ETensorDType::BF16 && dst.DType == ETensorDType::FP32) {
            convert_dtype(dst.get<float>(), reinterpret_cast<const nv_bfloat16*>(src.Data), src.nelem(), fetch_stream);
            return;
        }
        if (src.DType == ETensorDType::FP32 && dst.DType == ETensorDType::BF16) {
            convert_dtype(reinterpret_cast<nv_bfloat16*>(dst.Data), src.get<float>(), src.nelem(), fetch_stream);
            return;
        }
        if (src.DType == ETensorDType::FP16 && dst.DType == ETensorDType::BF16) {
            convert_dtype(reinterpret_cast<nv_bfloat16*>(dst.Data), reinterpret_cast<const half*>(src.Data), src.nelem(), fetch_stream);
            return;
        }

        throw std::runtime_error("ModularWeightManager::gather_block: unsupported dtype conversion");
    };

    if (!mStreamWeights) {
        auto& status = mLayerStatus.at(layer_idx);
        if (status.version == mVersion) return;

        CUDA_CHECK(cudaStreamWaitEvent(fetch_stream, status.done_event));
        status.fetch_pending = true;
        status.is_ready = false;
        status.version = mVersion;

        auto& src = mMasterBlocks.at(layer_idx);
        auto& dst = mWorkBlocks.at(layer_idx);

        // Propagate Stats pointers (abs_max/scale) for this layer.
        dst.ln1.weight.Stats = src.ln1.weight.Stats;
        dst.attention.qkv_weight.Stats = src.attention.qkv_weight.Stats;
        dst.attention.out_weight.Stats = src.attention.out_weight.Stats;
        dst.ln2.weight.Stats = src.ln2.weight.Stats;
        if (src.attention.qkv_bias.has_value() && dst.attention.qkv_bias.has_value()) {
            dst.attention.qkv_bias->Stats = src.attention.qkv_bias->Stats;
        }
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            dst.mlp_up_weight.Stats = src.mlp_up_weight.Stats;
            dst.mlp_down_weight.Stats = src.mlp_down_weight.Stats;
        }

        // Convert/copy local shard into the correct slice of the destination, then all-gather.
        auto shard_dst = [&](Tensor& full) -> Tensor {
            if (mConfig.num_shards == 1) return full;
            return static_cast<Tensor>(shard_view(full, mConfig.shard_idx, mConfig.num_shards));
        };

        convert_into(src.ln1.weight, shard_dst(dst.ln1.weight));
        convert_into(src.attention.qkv_weight, shard_dst(dst.attention.qkv_weight));
        if (src.attention.qkv_bias.has_value() && dst.attention.qkv_bias.has_value()) {
            convert_into(src.attention.qkv_bias.value(), shard_dst(dst.attention.qkv_bias.value()));
        }
        convert_into(src.attention.out_weight, shard_dst(dst.attention.out_weight));
        if constexpr (has_qk_norm_weights<std::decay_t<decltype(src.attention)>>::value) {
            if (src.attention.q_norm_weight.has_value() && dst.attention.q_norm_weight.has_value()) {
                convert_into(src.attention.q_norm_weight.value(), shard_dst(dst.attention.q_norm_weight.value()));
            }
            if (src.attention.k_norm_weight.has_value() && dst.attention.k_norm_weight.has_value()) {
                convert_into(src.attention.k_norm_weight.value(), shard_dst(dst.attention.k_norm_weight.value()));
            }
        }
        convert_into(src.ln2.weight, shard_dst(dst.ln2.weight));
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            convert_into(src.mlp_up_weight, shard_dst(dst.mlp_up_weight));
            convert_into(src.mlp_down_weight, shard_dst(dst.mlp_down_weight));
        }

        if (mConfig.num_shards == 1) {
            CUDA_CHECK(cudaEventRecord(status.done_event, fetch_stream));
            return;
        }

        comm.begin_transaction(fetch_stream);
        auto gather_full = [&](Tensor& full) {
            if (!full.Data || full.nelem() == 0) return;
            TensorShard local = shard_view(full, mConfig.shard_idx, mConfig.num_shards);
            local.Stats = full.Stats;
            comm.schedule_all_gather(local, full);
        };

        gather_full(dst.ln1.weight);
        gather_full(dst.attention.qkv_weight);
        if (dst.attention.qkv_bias.has_value()) gather_full(dst.attention.qkv_bias.value());
        gather_full(dst.attention.out_weight);
        if constexpr (has_qk_norm_weights<std::decay_t<decltype(dst.attention)>>::value) {
            if (dst.attention.q_norm_weight.has_value()) gather_full(dst.attention.q_norm_weight.value());
            if (dst.attention.k_norm_weight.has_value()) gather_full(dst.attention.k_norm_weight.value());
        }
        gather_full(dst.ln2.weight);
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            gather_full(dst.mlp_up_weight);
            gather_full(dst.mlp_down_weight);
        }

        comm.execute_transaction(status.done_event);
        return;
    }

    // Streamed ZeRO-3 mode: gather into a free prefetch buffer (double-buffered).
    int buf_idx = find_free_buffer(mPrefetchStatus);
    if (buf_idx < 0) {
        buf_idx = 0;
        wait_for_buffer(mPrefetchStatus[buf_idx], fetch_stream);
    }
    auto& status = mPrefetchStatus[buf_idx];
    if (status.layer_idx == layer_idx && status.version == mVersion) return;

    CUDA_CHECK(cudaStreamWaitEvent(fetch_stream, status.done_event));
    status.layer_idx = layer_idx;
    status.fetch_pending = true;
    status.is_ready = false;
    status.version = mVersion;

    auto& master_src = mMasterBlocks.at(layer_idx);
    auto& dst = mPrefetchBuffer[buf_idx];

    // Prefetch buffers are reused across layers; update Stats pointers for this layer.
    dst.ln1.weight.Stats = master_src.ln1.weight.Stats;
    dst.attention.qkv_weight.Stats = master_src.attention.qkv_weight.Stats;
    dst.attention.out_weight.Stats = master_src.attention.out_weight.Stats;
    dst.ln2.weight.Stats = master_src.ln2.weight.Stats;
    if (master_src.attention.qkv_bias.has_value() && dst.attention.qkv_bias.has_value()) {
        dst.attention.qkv_bias->Stats = master_src.attention.qkv_bias->Stats;
    }
    if constexpr (has_mlp_weights<BlockWeights>::value) {
        dst.mlp_up_weight.Stats = master_src.mlp_up_weight.Stats;
        dst.mlp_down_weight.Stats = master_src.mlp_down_weight.Stats;
    }

    auto shard_dst = [&](Tensor& full) -> Tensor {
        return static_cast<Tensor>(shard_view(full, mConfig.shard_idx, mConfig.num_shards));
    };

    // Determine the source for conversion: either persistent quant storage or master weights.
    const bool use_persistent_quants = mConfig.persistent_quants && !mQuantBlocks.empty();

    // Helper to quantize from master to quant storage (used when quants are stale)
    auto update_quant_block = [&]() {
        auto& quant_dst = mQuantBlocks.at(layer_idx);
        // Quantize from master weights into quant storage
        convert_into(master_src.ln1.weight, quant_dst.ln1.weight);
        convert_into(master_src.attention.qkv_weight, quant_dst.attention.qkv_weight);
        if (master_src.attention.qkv_bias.has_value() && quant_dst.attention.qkv_bias.has_value()) {
            convert_into(master_src.attention.qkv_bias.value(), quant_dst.attention.qkv_bias.value());
        }
        convert_into(master_src.attention.out_weight, quant_dst.attention.out_weight);
        if constexpr (has_qk_norm_weights<std::decay_t<decltype(master_src.attention)>>::value) {
            if (master_src.attention.q_norm_weight.has_value() && quant_dst.attention.q_norm_weight.has_value()) {
                convert_into(master_src.attention.q_norm_weight.value(), quant_dst.attention.q_norm_weight.value());
            }
            if (master_src.attention.k_norm_weight.has_value() && quant_dst.attention.k_norm_weight.has_value()) {
                convert_into(master_src.attention.k_norm_weight.value(), quant_dst.attention.k_norm_weight.value());
            }
        }
        convert_into(master_src.ln2.weight, quant_dst.ln2.weight);
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            convert_into(master_src.mlp_up_weight, quant_dst.mlp_up_weight);
            convert_into(master_src.mlp_down_weight, quant_dst.mlp_down_weight);
        }
        mQuantBlockVersion[layer_idx] = mVersion;
    };

    if (use_persistent_quants && mConfig.offload_quants && !mConfig.use_zero_copy) {
        // Offloaded quants without zero-copy: need to check if quants are stale and re-quantize.
        const bool quants_stale = (mQuantBlockVersion[layer_idx] != mVersion);

        int qbuf_idx = find_free_buffer(mQuantStatus);
        if (qbuf_idx < 0) {
            qbuf_idx = layer_idx % 2;
            wait_for_buffer(mQuantStatus[qbuf_idx], fetch_stream);
        }
        CUDA_CHECK(cudaStreamWaitEvent(fetch_stream, mQuantStatus[qbuf_idx].done_event));

        auto& quant_host = mQuantBlocks.at(layer_idx);
        auto& quant_device = mQuantBuffer[qbuf_idx];

        if (quants_stale) {
            // Quantize from master to device staging buffer
            convert_into(master_src.ln1.weight, quant_device.ln1.weight);
            convert_into(master_src.attention.qkv_weight, quant_device.attention.qkv_weight);
            if (master_src.attention.qkv_bias.has_value() && quant_device.attention.qkv_bias.has_value()) {
                convert_into(master_src.attention.qkv_bias.value(), quant_device.attention.qkv_bias.value());
            }
            convert_into(master_src.attention.out_weight, quant_device.attention.out_weight);
            convert_into(master_src.ln2.weight, quant_device.ln2.weight);
            if constexpr (has_mlp_weights<BlockWeights>::value) {
                convert_into(master_src.mlp_up_weight, quant_device.mlp_up_weight);
                convert_into(master_src.mlp_down_weight, quant_device.mlp_down_weight);
            }

            // Copy from device staging to host storage (persist for future steps)
            auto copy_d2h = [fetch_stream](const Tensor& src, Tensor& dst_t) {
                if (!src.Data || !dst_t.Data || src.nelem() == 0) return;
                CUDA_CHECK(cudaMemcpyAsync(dst_t.Data, src.Data, src.bytes(), cudaMemcpyDeviceToHost, fetch_stream));
            };
            copy_d2h(quant_device.ln1.weight, quant_host.ln1.weight);
            copy_d2h(quant_device.attention.qkv_weight, quant_host.attention.qkv_weight);
            if (quant_device.attention.qkv_bias.has_value() && quant_host.attention.qkv_bias.has_value()) {
                copy_d2h(quant_device.attention.qkv_bias.value(), quant_host.attention.qkv_bias.value());
            }
            copy_d2h(quant_device.attention.out_weight, quant_host.attention.out_weight);
            copy_d2h(quant_device.ln2.weight, quant_host.ln2.weight);
            if constexpr (has_mlp_weights<BlockWeights>::value) {
                copy_d2h(quant_device.mlp_up_weight, quant_host.mlp_up_weight);
                copy_d2h(quant_device.mlp_down_weight, quant_host.mlp_down_weight);
            }
            mQuantBlockVersion[layer_idx] = mVersion;
        } else {
            // Quants are fresh: copy from host to device staging buffer
            auto copy_h2d = [fetch_stream](const Tensor& src, Tensor& dst_t) {
                if (!src.Data || !dst_t.Data || src.nelem() == 0) return;
                CUDA_CHECK(cudaMemcpyAsync(dst_t.Data, src.Data, src.bytes(), cudaMemcpyHostToDevice, fetch_stream));
            };
            copy_h2d(quant_host.ln1.weight, quant_device.ln1.weight);
            copy_h2d(quant_host.attention.qkv_weight, quant_device.attention.qkv_weight);
            if (quant_host.attention.qkv_bias.has_value() && quant_device.attention.qkv_bias.has_value()) {
                copy_h2d(quant_host.attention.qkv_bias.value(), quant_device.attention.qkv_bias.value());
            }
            copy_h2d(quant_host.attention.out_weight, quant_device.attention.out_weight);
            copy_h2d(quant_host.ln2.weight, quant_device.ln2.weight);
            if constexpr (has_mlp_weights<BlockWeights>::value) {
                copy_h2d(quant_host.mlp_up_weight, quant_device.mlp_up_weight);
                copy_h2d(quant_host.mlp_down_weight, quant_device.mlp_down_weight);
            }
        }

        // Copy from device staging to prefetch buffer (device-to-device, same dtype - no conversion needed)
        convert_into(quant_device.ln1.weight, shard_dst(dst.ln1.weight));
        convert_into(quant_device.attention.qkv_weight, shard_dst(dst.attention.qkv_weight));
        if (quant_device.attention.qkv_bias.has_value() && dst.attention.qkv_bias.has_value()) {
            convert_into(quant_device.attention.qkv_bias.value(), shard_dst(dst.attention.qkv_bias.value()));
        }
        convert_into(quant_device.attention.out_weight, shard_dst(dst.attention.out_weight));
        if constexpr (has_qk_norm_weights<std::decay_t<decltype(quant_device.attention)>>::value) {
            if (quant_device.attention.q_norm_weight.has_value() && dst.attention.q_norm_weight.has_value()) {
                convert_into(quant_device.attention.q_norm_weight.value(), shard_dst(dst.attention.q_norm_weight.value()));
            }
            if (quant_device.attention.k_norm_weight.has_value() && dst.attention.k_norm_weight.has_value()) {
                convert_into(quant_device.attention.k_norm_weight.value(), shard_dst(dst.attention.k_norm_weight.value()));
            }
        }
        convert_into(quant_device.ln2.weight, shard_dst(dst.ln2.weight));
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            convert_into(quant_device.mlp_up_weight, shard_dst(dst.mlp_up_weight));
            convert_into(quant_device.mlp_down_weight, shard_dst(dst.mlp_down_weight));
        }

        CUDA_CHECK(cudaEventRecord(mQuantStatus[qbuf_idx].done_event, fetch_stream));
    } else if (use_persistent_quants) {
        // Persistent quants on device (or zero-copy): check if stale and re-quantize if needed.
        if (mQuantBlockVersion[layer_idx] != mVersion) {
            update_quant_block();
        }

        // Copy from quant storage to prefetch buffer
        auto& quant_src = mQuantBlocks.at(layer_idx);
        convert_into(quant_src.ln1.weight, shard_dst(dst.ln1.weight));
        convert_into(quant_src.attention.qkv_weight, shard_dst(dst.attention.qkv_weight));
        if (quant_src.attention.qkv_bias.has_value() && dst.attention.qkv_bias.has_value()) {
            convert_into(quant_src.attention.qkv_bias.value(), shard_dst(dst.attention.qkv_bias.value()));
        }
        convert_into(quant_src.attention.out_weight, shard_dst(dst.attention.out_weight));
        if constexpr (has_qk_norm_weights<std::decay_t<decltype(quant_src.attention)>>::value) {
            if (quant_src.attention.q_norm_weight.has_value() && dst.attention.q_norm_weight.has_value()) {
                convert_into(quant_src.attention.q_norm_weight.value(), shard_dst(dst.attention.q_norm_weight.value()));
            }
            if (quant_src.attention.k_norm_weight.has_value() && dst.attention.k_norm_weight.has_value()) {
                convert_into(quant_src.attention.k_norm_weight.value(), shard_dst(dst.attention.k_norm_weight.value()));
            }
        }
        convert_into(quant_src.ln2.weight, shard_dst(dst.ln2.weight));
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            convert_into(quant_src.mlp_up_weight, shard_dst(dst.mlp_up_weight));
            convert_into(quant_src.mlp_down_weight, shard_dst(dst.mlp_down_weight));
        }
    } else {
        // No persistent quants: convert from master weights (with potential quantization).
        convert_into(master_src.ln1.weight, shard_dst(dst.ln1.weight));
        convert_into(master_src.attention.qkv_weight, shard_dst(dst.attention.qkv_weight));
        if (master_src.attention.qkv_bias.has_value() && dst.attention.qkv_bias.has_value()) {
            convert_into(master_src.attention.qkv_bias.value(), shard_dst(dst.attention.qkv_bias.value()));
        }
        convert_into(master_src.attention.out_weight, shard_dst(dst.attention.out_weight));
        if constexpr (has_qk_norm_weights<std::decay_t<decltype(master_src.attention)>>::value) {
            if (master_src.attention.q_norm_weight.has_value() && dst.attention.q_norm_weight.has_value()) {
                convert_into(master_src.attention.q_norm_weight.value(), shard_dst(dst.attention.q_norm_weight.value()));
            }
            if (master_src.attention.k_norm_weight.has_value() && dst.attention.k_norm_weight.has_value()) {
                convert_into(master_src.attention.k_norm_weight.value(), shard_dst(dst.attention.k_norm_weight.value()));
            }
        }
        convert_into(master_src.ln2.weight, shard_dst(dst.ln2.weight));
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            convert_into(master_src.mlp_up_weight, shard_dst(dst.mlp_up_weight));
            convert_into(master_src.mlp_down_weight, shard_dst(dst.mlp_down_weight));
        }
    }

    comm.begin_transaction(fetch_stream);
    auto gather_full = [&](Tensor& full) {
        if (!full.Data || full.nelem() == 0) return;
        TensorShard local = shard_view(full, mConfig.shard_idx, mConfig.num_shards);
        local.Stats = full.Stats;
        comm.schedule_all_gather(local, full);
    };
    gather_full(dst.ln1.weight);
    gather_full(dst.attention.qkv_weight);
    if (dst.attention.qkv_bias.has_value()) gather_full(dst.attention.qkv_bias.value());
    gather_full(dst.attention.out_weight);
    if constexpr (has_qk_norm_weights<std::decay_t<decltype(dst.attention)>>::value) {
        if (dst.attention.q_norm_weight.has_value()) gather_full(dst.attention.q_norm_weight.value());
        if (dst.attention.k_norm_weight.has_value()) gather_full(dst.attention.k_norm_weight.value());
    }
    gather_full(dst.ln2.weight);
    if constexpr (has_mlp_weights<BlockWeights>::value) {
        gather_full(dst.mlp_up_weight);
        gather_full(dst.mlp_down_weight);
    }
    comm.execute_transaction(status.done_event);
}

template<typename Block>
typename Block::Weights& ModularWeightManager<Block>::get_block(int layer_idx, cudaStream_t stream) {
    BlockWeights* result = nullptr;

    // If external weight provider is set (QLoRA), use it instead
    if (mExternalWeightProvider) {
        result = &mExternalWeightProvider(layer_idx, stream);
    } else if (!mStreamWeights) {
        auto& status = mLayerStatus.at(layer_idx);
        wait_for_buffer(status, stream);
        status.is_ready = false;
        result = &mWorkBlocks.at(layer_idx);
    } else {
        for (int i = 0; i < 2; ++i) {
            if (mPrefetchStatus[i].layer_idx == layer_idx) {
                wait_for_buffer(mPrefetchStatus[i], stream);
                mPrefetchStatus[i].is_ready = false;
                result = &mPrefetchBuffer[i];
                break;
            }
        }
    }

    if (!result) {
        throw std::runtime_error("Block weights not prefetched: layer " + std::to_string(layer_idx));
    }

    // Quantize weights to FP8 cache if FP8 forward mode is enabled.
    if (mConfig.enable_fp8_forward && mFP8CacheLayerIdx != layer_idx) {
        quantize_weights_to_fp8_cache(*result, stream);
        mFP8CacheLayerIdx = layer_idx;
    }

    // Quantize weights to FP4 cache if FP4 forward mode is enabled.
    if (mConfig.enable_fp4_forward) {
        if (mFP4PersistentCacheEnabled) {
            const auto& cfg = mConfig.block_config;
            const long C = cfg.hidden_size;
            const long Hq = cfg.num_query_heads;
            const long Hkv = cfg.num_kv_heads;
            const long Hs = cfg.head_size;
            const long D = cfg.intermediate_size;
            const long QKV_C = (Hq + 2 * Hkv) * Hs;

            auto cutlass_scale_size = [](long rows, long cols) -> long {
                constexpr int kBlockSize = 16;
                constexpr int kTileDim = 128;
                long num_scale_cols = (cols + kBlockSize - 1) / kBlockSize;
                long aligned_rows = ((rows + kTileDim - 1) / kTileDim) * kTileDim;
                long aligned_cols = ((num_scale_cols + 3) / 4) * 4;
                return aligned_rows * aligned_cols;
            };

            const long qkv_scale_size = cutlass_scale_size(QKV_C, C);
            const long o_scale_size = cutlass_scale_size(C, Hq * Hs);
            const long qkv_scale_size_t = cutlass_scale_size(C, QKV_C);
            const long o_scale_size_t = cutlass_scale_size(Hq * Hs, C);

            long mlp_up_scale_size = 0;
            long mlp_down_scale_size = 0;
            long mlp_up_scale_size_t = 0;
            long mlp_down_scale_size_t = 0;
            if constexpr (has_mlp_weights<BlockWeights>::value) {
                mlp_up_scale_size = cutlass_scale_size(2 * D, C);
                mlp_down_scale_size = cutlass_scale_size(C, D);
                mlp_up_scale_size_t = cutlass_scale_size(C, 2 * D);
                mlp_down_scale_size_t = cutlass_scale_size(D, C);
            }

            auto set_views = [&](int l) {
                // Amax view: (4,) for this layer - forward cache
                mFP4WeightAmax = slice(mFP4WeightAmaxAll, 0, l * 4, l * 4 + 4);
                // Amax view: (4,) for this layer - transposed cache
                mFP4WeightAmaxT = slice(mFP4WeightAmaxAllT, 0, l * 4, l * 4 + 4);

                // Forward weights (W)
                mFP4WeightCache.qkv_weight.data = slice(mFP4WeightDataAll[0], 0, l * QKV_C, (l + 1) * QKV_C);
                mFP4WeightCache.qkv_weight.scales = slice(mFP4WeightScalesAll[0], 0, l * qkv_scale_size, (l + 1) * qkv_scale_size);

                mFP4WeightCache.o_weight.data = slice(mFP4WeightDataAll[1], 0, l * C, (l + 1) * C);
                mFP4WeightCache.o_weight.scales = slice(mFP4WeightScalesAll[1], 0, l * o_scale_size, (l + 1) * o_scale_size);

                if constexpr (has_mlp_weights<BlockWeights>::value) {
                    mFP4WeightCache.mlp_up_weight.data = slice(mFP4WeightDataAll[2], 0, l * (2 * D), (l + 1) * (2 * D));
                    mFP4WeightCache.mlp_up_weight.scales = slice(mFP4WeightScalesAll[2], 0, l * mlp_up_scale_size, (l + 1) * mlp_up_scale_size);

                    mFP4WeightCache.mlp_down_weight.data = slice(mFP4WeightDataAll[3], 0, l * C, (l + 1) * C);
                    mFP4WeightCache.mlp_down_weight.scales = slice(mFP4WeightScalesAll[3], 0, l * mlp_down_scale_size, (l + 1) * mlp_down_scale_size);
                }

                // Transposed weights (W^T) for dgrad
                mFP4WeightCacheT.qkv_weight.data = slice(mFP4WeightDataAllT[0], 0, l * C, (l + 1) * C);
                mFP4WeightCacheT.qkv_weight.scales = slice(mFP4WeightScalesAllT[0], 0, l * qkv_scale_size_t, (l + 1) * qkv_scale_size_t);

                mFP4WeightCacheT.o_weight.data = slice(mFP4WeightDataAllT[1], 0, l * (Hq * Hs), (l + 1) * (Hq * Hs));
                mFP4WeightCacheT.o_weight.scales = slice(mFP4WeightScalesAllT[1], 0, l * o_scale_size_t, (l + 1) * o_scale_size_t);

                if constexpr (has_mlp_weights<BlockWeights>::value) {
                    mFP4WeightCacheT.mlp_up_weight.data = slice(mFP4WeightDataAllT[2], 0, l * C, (l + 1) * C);
                    mFP4WeightCacheT.mlp_up_weight.scales = slice(mFP4WeightScalesAllT[2], 0, l * mlp_up_scale_size_t, (l + 1) * mlp_up_scale_size_t);

                    mFP4WeightCacheT.mlp_down_weight.data = slice(mFP4WeightDataAllT[3], 0, l * D, (l + 1) * D);
                    mFP4WeightCacheT.mlp_down_weight.scales = slice(mFP4WeightScalesAllT[3], 0, l * mlp_down_scale_size_t, (l + 1) * mlp_down_scale_size_t);
                }

                mFP4CacheLayerIdx = l;
            };

            if (mFP4CacheLayerIdx != layer_idx) {
                set_views(layer_idx);
            }

            const int wanted_version = mFP4PersistentCacheStatic ? 0 : mVersion;
            if (mFP4PersistentCacheVersion.at((std::size_t)layer_idx) != wanted_version) {
                quantize_weights_to_fp4_cache(*result, stream);
                quantize_weights_to_fp4_cache_transposed(*result, stream);
                mFP4PersistentCacheVersion.at((std::size_t)layer_idx) = wanted_version;
            }
        } else if (mFP4CacheLayerIdx != layer_idx) {
            quantize_weights_to_fp4_cache(*result, stream);
            mFP4CacheLayerIdx = layer_idx;
        }
    }

    return *result;
}

template<typename Block>
void ModularWeightManager<Block>::release_block(int layer_idx, cudaStream_t stream) {
    // If external weight provider is set (QLoRA), skip release - provider manages its own buffers
    if (mExternalWeightProvider) {
        (void)layer_idx;
        (void)stream;
        return;
    }

    if (!mStreamWeights) {
        auto& st = mLayerStatus.at(layer_idx);
        CUDA_CHECK(cudaEventRecord(st.done_event, stream));
        st.is_ready = true;
        return;
    }

    for (int i = 0; i < 2; ++i) {
        if (mPrefetchStatus[i].layer_idx == layer_idx) {
            CUDA_CHECK(cudaEventRecord(mPrefetchStatus[i].done_event, stream));
            mPrefetchStatus[i].is_ready = true;
            return;
        }
    }
}

template<typename Block>
void ModularWeightManager<Block>::gather_embeddings(NCCLCommunicator& comm, cudaStream_t stream) {
    // If external provider is set, skip gather - provider manages its own weights
    if (mExternalEmbeddingsProvider) {
        (void)comm;
        (void)stream;
        return;
    }
    if (mEmbeddingsStatus.version == mVersion) return;

    CUDA_CHECK(cudaStreamWaitEvent(stream, mEmbeddingsStatus.done_event));
    mEmbeddingsStatus.fetch_pending = true;
    mEmbeddingsStatus.is_ready = false;
    mEmbeddingsStatus.version = mVersion;

    // Propagate Stats pointer to the full work tensor.
    mWorkNonBlock.embeddings.Stats = mMasterNonBlock.embeddings.Stats;

    auto convert_into = [&](const Tensor& src, Tensor& dst) {
        if (!src.Data || !dst.Data || src.nelem() == 0) return;
        dst.Stats = src.Stats;
        if (src.Data == dst.Data && src.DType == dst.DType) return;
        if (src.DType == dst.DType) {
            CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, dst.bytes(), cudaMemcpyDefault, stream));
            return;
        }
        if (src.Device == -1 && !mConfig.use_zero_copy) {
            throw std::runtime_error("ModularWeightManager::gather_embeddings: dtype conversion from offloaded master weights requires --use-zero-copy");
        }
        if (dst.DType == ETensorDType::FP8_E4M3 || dst.DType == ETensorDType::FP8_E5M2 || dst.DType == ETensorDType::INT8) {
            if (!src.Stats) throw std::runtime_error("ModularWeightManager::gather_embeddings: FP8/INT8 gather requires Stats on the source tensor");
            quantize_with_abs_max(dst, /*scale_ptr=*/src.Stats + 1, src, /*abs_max=*/src.Stats, (long)src.nelem(), mDeviceProp, stream);
            return;
        }
        if (src.DType == ETensorDType::BF16 && dst.DType == ETensorDType::FP32) {
            convert_dtype(dst.get<float>(), reinterpret_cast<const nv_bfloat16*>(src.Data), src.nelem(), stream);
            return;
        }
        if (src.DType == ETensorDType::FP32 && dst.DType == ETensorDType::BF16) {
            convert_dtype(reinterpret_cast<nv_bfloat16*>(dst.Data), src.get<float>(), src.nelem(), stream);
            return;
        }
        throw std::runtime_error("ModularWeightManager::gather_embeddings: unsupported dtype conversion");
    };

    Tensor local_dst = (mConfig.num_shards == 1)
        ? mWorkNonBlock.embeddings
        : static_cast<Tensor>(shard_view(mWorkNonBlock.embeddings, mConfig.shard_idx, mConfig.num_shards));

    convert_into(mMasterNonBlock.embeddings, local_dst);

    if (mConfig.num_shards == 1) {
        CUDA_CHECK(cudaEventRecord(mEmbeddingsStatus.done_event, stream));
        return;
    }

    comm.begin_transaction(stream);
    TensorShard local = shard_view(mWorkNonBlock.embeddings, mConfig.shard_idx, mConfig.num_shards);
    local.Stats = mWorkNonBlock.embeddings.Stats;
    comm.schedule_all_gather(local, mWorkNonBlock.embeddings);
    comm.execute_transaction(mEmbeddingsStatus.done_event);
}

template<typename Block>
Tensor& ModularWeightManager<Block>::get_embeddings(cudaStream_t stream) {
    if (mExternalEmbeddingsProvider) {
        return mExternalEmbeddingsProvider(stream);
    }
    wait_for_buffer(mEmbeddingsStatus, stream);
    return mWorkNonBlock.embeddings;
}

template<typename Block>
void ModularWeightManager<Block>::release_embeddings(cudaStream_t stream) {
    if (mExternalEmbeddingsProvider) {
        (void)stream;
        return;
    }
    CUDA_CHECK(cudaEventRecord(mEmbeddingsStatus.done_event, stream));
    mEmbeddingsStatus.is_ready = true;
}

template<typename Block>
void ModularWeightManager<Block>::gather_final_norm(NCCLCommunicator& comm, cudaStream_t stream) {
    if (mExternalFinalNormProvider) {
        (void)comm;
        (void)stream;
        return;
    }
    if (mFinalNormStatus.version == mVersion) return;

    CUDA_CHECK(cudaStreamWaitEvent(stream, mFinalNormStatus.done_event));
    mFinalNormStatus.fetch_pending = true;
    mFinalNormStatus.is_ready = false;
    mFinalNormStatus.version = mVersion;

    mWorkNonBlock.final_norm_weight.Stats = mMasterNonBlock.final_norm_weight.Stats;

    auto convert_into = [&](const Tensor& src, Tensor& dst) {
        if (!src.Data || !dst.Data || src.nelem() == 0) return;
        dst.Stats = src.Stats;
        if (src.Data == dst.Data && src.DType == dst.DType) return;
        if (src.DType == dst.DType) {
            CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, dst.bytes(), cudaMemcpyDefault, stream));
            return;
        }
        if (src.Device == -1 && !mConfig.use_zero_copy) {
            throw std::runtime_error("ModularWeightManager::gather_final_norm: dtype conversion from offloaded master weights requires --use-zero-copy");
        }
        if (dst.DType == ETensorDType::FP8_E4M3 || dst.DType == ETensorDType::FP8_E5M2 || dst.DType == ETensorDType::INT8) {
            if (!src.Stats) throw std::runtime_error("ModularWeightManager::gather_final_norm: FP8/INT8 gather requires Stats on the source tensor");
            quantize_with_abs_max(dst, /*scale_ptr=*/src.Stats + 1, src, /*abs_max=*/src.Stats, (long)src.nelem(), mDeviceProp, stream);
            return;
        }
        if (src.DType == ETensorDType::BF16 && dst.DType == ETensorDType::FP32) {
            convert_dtype(dst.get<float>(), reinterpret_cast<const nv_bfloat16*>(src.Data), src.nelem(), stream);
            return;
        }
        if (src.DType == ETensorDType::FP32 && dst.DType == ETensorDType::BF16) {
            convert_dtype(reinterpret_cast<nv_bfloat16*>(dst.Data), src.get<float>(), src.nelem(), stream);
            return;
        }
        throw std::runtime_error("ModularWeightManager::gather_final_norm: unsupported dtype conversion");
    };

    Tensor local_dst = (mConfig.num_shards == 1)
        ? mWorkNonBlock.final_norm_weight
        : static_cast<Tensor>(shard_view(mWorkNonBlock.final_norm_weight, mConfig.shard_idx, mConfig.num_shards));
    convert_into(mMasterNonBlock.final_norm_weight, local_dst);

    if (mConfig.num_shards == 1) {
        CUDA_CHECK(cudaEventRecord(mFinalNormStatus.done_event, stream));
        return;
    }

    comm.begin_transaction(stream);
    TensorShard local = shard_view(mWorkNonBlock.final_norm_weight, mConfig.shard_idx, mConfig.num_shards);
    local.Stats = mWorkNonBlock.final_norm_weight.Stats;
    comm.schedule_all_gather(local, mWorkNonBlock.final_norm_weight);
    comm.execute_transaction(mFinalNormStatus.done_event);
}

template<typename Block>
Tensor& ModularWeightManager<Block>::get_final_norm(cudaStream_t stream) {
    if (mExternalFinalNormProvider) {
        return mExternalFinalNormProvider(stream);
    }
    wait_for_buffer(mFinalNormStatus, stream);
    return mWorkNonBlock.final_norm_weight;
}

template<typename Block>
void ModularWeightManager<Block>::release_final_norm(cudaStream_t stream) {
    if (mExternalFinalNormProvider) {
        (void)stream;
        return;
    }
    CUDA_CHECK(cudaEventRecord(mFinalNormStatus.done_event, stream));
    mFinalNormStatus.is_ready = true;
}

template<typename Block>
void ModularWeightManager<Block>::gather_lm_head(NCCLCommunicator& comm, cudaStream_t stream) {
    if (mExternalLMHeadProvider) {
        (void)comm;
        (void)stream;
        return;
    }
    if (mConfig.tied_embeddings) return;
    if (mLMHeadStatus.version == mVersion) return;

    CUDA_CHECK(cudaStreamWaitEvent(stream, mLMHeadStatus.done_event));
    mLMHeadStatus.fetch_pending = true;
    mLMHeadStatus.is_ready = false;
    mLMHeadStatus.version = mVersion;

    mWorkNonBlock.lm_head.Stats = mMasterNonBlock.lm_head.Stats;

    auto convert_into = [&](const Tensor& src, Tensor& dst) {
        if (!src.Data || !dst.Data || src.nelem() == 0) return;
        dst.Stats = src.Stats;
        if (src.Data == dst.Data && src.DType == dst.DType) return;
        if (src.DType == dst.DType) {
            CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, dst.bytes(), cudaMemcpyDefault, stream));
            return;
        }
        if (src.Device == -1 && !mConfig.use_zero_copy) {
            throw std::runtime_error("ModularWeightManager::gather_lm_head: dtype conversion from offloaded master weights requires --use-zero-copy");
        }
        if (dst.DType == ETensorDType::FP8_E4M3 || dst.DType == ETensorDType::FP8_E5M2 || dst.DType == ETensorDType::INT8) {
            if (!src.Stats) throw std::runtime_error("ModularWeightManager::gather_lm_head: FP8/INT8 gather requires Stats on the source tensor");
            quantize_with_abs_max(dst, /*scale_ptr=*/src.Stats + 1, src, /*abs_max=*/src.Stats, (long)src.nelem(), mDeviceProp, stream);
            return;
        }
        if (src.DType == ETensorDType::BF16 && dst.DType == ETensorDType::FP32) {
            convert_dtype(dst.get<float>(), reinterpret_cast<const nv_bfloat16*>(src.Data), src.nelem(), stream);
            return;
        }
        if (src.DType == ETensorDType::FP32 && dst.DType == ETensorDType::BF16) {
            convert_dtype(reinterpret_cast<nv_bfloat16*>(dst.Data), src.get<float>(), src.nelem(), stream);
            return;
        }
        throw std::runtime_error("ModularWeightManager::gather_lm_head: unsupported dtype conversion");
    };

    Tensor local_dst = (mConfig.num_shards == 1)
        ? mWorkNonBlock.lm_head
        : static_cast<Tensor>(shard_view(mWorkNonBlock.lm_head, mConfig.shard_idx, mConfig.num_shards));
    convert_into(mMasterNonBlock.lm_head, local_dst);

    if (mConfig.num_shards == 1) {
        CUDA_CHECK(cudaEventRecord(mLMHeadStatus.done_event, stream));
        return;
    }

    comm.begin_transaction(stream);
    TensorShard local = shard_view(mWorkNonBlock.lm_head, mConfig.shard_idx, mConfig.num_shards);
    local.Stats = mWorkNonBlock.lm_head.Stats;
    comm.schedule_all_gather(local, mWorkNonBlock.lm_head);
    comm.execute_transaction(mLMHeadStatus.done_event);
}

template<typename Block>
Tensor& ModularWeightManager<Block>::get_lm_head(cudaStream_t stream) {
    if (mExternalLMHeadProvider) {
        return mExternalLMHeadProvider(stream);
    }
    if (mConfig.tied_embeddings) {
        if (mExternalEmbeddingsProvider) {
            return mExternalEmbeddingsProvider(stream);
        }
        wait_for_buffer(mEmbeddingsStatus, stream);
        return mWorkNonBlock.embeddings;
    }
    wait_for_buffer(mLMHeadStatus, stream);
    return mWorkNonBlock.lm_head;
}

template<typename Block>
void ModularWeightManager<Block>::release_lm_head(cudaStream_t stream) {
    if (mExternalLMHeadProvider || (mConfig.tied_embeddings && mExternalEmbeddingsProvider)) {
        (void)stream;
        return;
    }
    CUDA_CHECK(cudaEventRecord(mLMHeadStatus.done_event, stream));
    mLMHeadStatus.is_ready = true;
}

} // namespace modules

#endif // LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_GATHER_H
