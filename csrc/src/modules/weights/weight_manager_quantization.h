// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_QUANTIZATION_H
#define LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_QUANTIZATION_H

#include "kernels/kernels.h"
#include "modules/weights/weight_manager_helpers.h"

namespace modules {

template<typename Block>
void ModularWeightManager<Block>::quantize_weights_to_fp8_cache(const BlockWeights& src, cudaStream_t stream) {
    // Skip if FP8 forward caching is not enabled
    if (!mConfig.enable_fp8_forward || !mFP8WeightCache.qkv_weight.Data) {
        return;
    }

    // Helper lambda to quantize a single weight tensor to FP8
    auto quantize_weight = [&](const Tensor& bf16_weight, Tensor& fp8_weight) {
        if (!bf16_weight.Data || !fp8_weight.Data) return;

        const long N = bf16_weight.nelem();
        if (N == 0) return;
        if (fp8_weight.DType != ETensorDType::FP8_E4M3) {
            // Cache is currently designed for E4M3 forward weights.
            return;
        }

        // QLoRA may provide weights already in FP8.
        if (bf16_weight.DType == ETensorDType::FP8_E4M3) {
            CUDA_CHECK(cudaMemcpyAsync(fp8_weight.Data, bf16_weight.Data, fp8_weight.bytes(), cudaMemcpyDefault, stream));
            if (bf16_weight.Stats && fp8_weight.Stats) {
                CUDA_CHECK(cudaMemcpyAsync(fp8_weight.Stats, bf16_weight.Stats, 2 * sizeof(float), cudaMemcpyDefault, stream));
            }
            return;
        }
        if (bf16_weight.DType != ETensorDType::BF16 && bf16_weight.DType != ETensorDType::FP32) {
            // Unsupported source dtype for on-the-fly quantization.
            return;
        }

        // Compute abs_max for this weight
        abs_max(fp8_weight.abs_max(), bf16_weight, N, mDeviceProp, stream);

        // Quantize to FP8 using the computed abs_max
        quantize_with_abs_max(fp8_weight, fp8_weight.scale(), bf16_weight, fp8_weight.abs_max(),
                              N, mDeviceProp, stream);
    };

    // Quantize all weight tensors to FP8 cache
    quantize_weight(src.attention.qkv_weight, mFP8WeightCache.qkv_weight);
    quantize_weight(src.attention.out_weight, mFP8WeightCache.o_weight);

    if constexpr (has_mlp_weights<BlockWeights>::value) {
        quantize_weight(src.mlp_up_weight, mFP8WeightCache.mlp_up_weight);
        quantize_weight(src.mlp_down_weight, mFP8WeightCache.mlp_down_weight);
    }
}

template<typename Block>
void ModularWeightManager<Block>::quantize_weights_to_fp4_cache(const BlockWeights& src, cudaStream_t stream) {
    // Skip if FP4 forward caching is not enabled
    if (!mConfig.enable_fp4_forward || !mFP4WeightCache.qkv_weight.data.Data) {
        return;
    }

    const auto& cfg = mConfig.block_config;
    const int C = static_cast<int>(cfg.hidden_size);
    const int Hq = static_cast<int>(cfg.num_query_heads);
    const int Hkv = static_cast<int>(cfg.num_kv_heads);
    const int Hs = static_cast<int>(cfg.head_size);
    const int D = static_cast<int>(cfg.intermediate_size);
    const int QKV_C = (Hq + 2 * Hkv) * Hs;

    auto quantize_fp4_weight = [&](const Tensor& bf16_weight, FP4WeightCacheEntry& fp4_cache,
                                   int N, int K) {
        if (!bf16_weight.Data || !fp4_cache.data.Data) return;

        // Only support BF16 source weights for now
        if (bf16_weight.DType != ETensorDType::BF16) {
            return;
        }

        // Use global amax buffer: 4 floats for qkv, o, mlp_up, mlp_down
        float* amax_ptr = mFP4WeightAmax.template get<float>();
        int amax_offset = 0;
        if (&fp4_cache == &mFP4WeightCache.qkv_weight) amax_offset = 0;
        else if (&fp4_cache == &mFP4WeightCache.o_weight) amax_offset = 1;
        else if constexpr (has_mlp_weights<BlockWeights>::value) {
            if (&fp4_cache == &mFP4WeightCache.mlp_up_weight) amax_offset = 2;
            else if (&fp4_cache == &mFP4WeightCache.mlp_down_weight) amax_offset = 3;
        }

        if (mConfig.enable_four_over_six) {
            quantize_nvfp4_4o6_cutlass_auto_scale(
                fp4_cache.data.template get<uint8_t>(),
                fp4_cache.scales.template get<uint8_t>(),
                amax_ptr + amax_offset,
                bf16_weight.template get<nv_bfloat16>(),
                N, K,
                mConfig.four_over_six_metric,
                mDeviceProp, stream);
        } else {
            quantize_nvfp4_weight_cutlass_auto_scale(
                fp4_cache.data.template get<uint8_t>(),
                fp4_cache.scales.template get<uint8_t>(),
                amax_ptr + amax_offset,
                bf16_weight.template get<nv_bfloat16>(),
                N, K,
                mDeviceProp, stream);
        }
    };

    // QKV weight: (QKV_C, C)
    quantize_fp4_weight(src.attention.qkv_weight, mFP4WeightCache.qkv_weight, QKV_C, C);

    // O weight: (C, Hq*Hs)
    quantize_fp4_weight(src.attention.out_weight, mFP4WeightCache.o_weight, C, Hq * Hs);

    if constexpr (has_mlp_weights<BlockWeights>::value) {
        quantize_fp4_weight(src.mlp_up_weight, mFP4WeightCache.mlp_up_weight, 2 * D, C);
        quantize_fp4_weight(src.mlp_down_weight, mFP4WeightCache.mlp_down_weight, C, D);
    }
}

template<typename Block>
void ModularWeightManager<Block>::quantize_weights_to_fp4_cache_transposed(const BlockWeights& src, cudaStream_t stream) {
    // Skip if FP4 caching is not enabled or transposed cache buffers are not available
    if (!mConfig.enable_fp4_forward || !mFP4WeightCacheT.qkv_weight.data.Data) {
        return;
    }

    const auto& cfg = mConfig.block_config;
    const int C = static_cast<int>(cfg.hidden_size);
    const int Hq = static_cast<int>(cfg.num_query_heads);
    const int Hkv = static_cast<int>(cfg.num_kv_heads);
    const int Hs = static_cast<int>(cfg.head_size);
    const int D = static_cast<int>(cfg.intermediate_size);
    const int QKV_C = (Hq + 2 * Hkv) * Hs;

    auto quantize_fp4_weight_t = [&](const Tensor& bf16_weight, FP4WeightCacheEntry& fp4_cache_t,
                                     int N, int K) {
        if (!bf16_weight.Data || !fp4_cache_t.data.Data) return;

        // Only support BF16 source weights for now
        if (bf16_weight.DType != ETensorDType::BF16) {
            return;
        }

        // Use separate amax buffer for transposed cache (4 floats: qkv, o, mlp_up, mlp_down)
        float* amax_ptr = mFP4WeightAmaxT.template get<float>();
        int amax_offset = 0;
        if (&fp4_cache_t == &mFP4WeightCacheT.qkv_weight) amax_offset = 0;
        else if (&fp4_cache_t == &mFP4WeightCacheT.o_weight) amax_offset = 1;
        else if constexpr (has_mlp_weights<BlockWeights>::value) {
            if (&fp4_cache_t == &mFP4WeightCacheT.mlp_up_weight) amax_offset = 2;
            else if (&fp4_cache_t == &mFP4WeightCacheT.mlp_down_weight) amax_offset = 3;
        }

        quantize_nvfp4_weight_cutlass_transpose_auto_scale(
            fp4_cache_t.data.template get<uint8_t>(),
            fp4_cache_t.scales.template get<uint8_t>(),
            amax_ptr + amax_offset,
            bf16_weight.template get<nv_bfloat16>(),
            N, K,
            mDeviceProp, stream);
    };

    // QKV weight: (QKV_C, C) -> transposed cache stores (C, QKV_C)
    quantize_fp4_weight_t(src.attention.qkv_weight, mFP4WeightCacheT.qkv_weight, QKV_C, C);

    // O weight: (C, Hq*Hs) -> transposed cache stores (Hq*Hs, C)
    quantize_fp4_weight_t(src.attention.out_weight, mFP4WeightCacheT.o_weight, C, Hq * Hs);

    if constexpr (has_mlp_weights<BlockWeights>::value) {
        quantize_fp4_weight_t(src.mlp_up_weight, mFP4WeightCacheT.mlp_up_weight, 2 * D, C);
        quantize_fp4_weight_t(src.mlp_down_weight, mFP4WeightCacheT.mlp_down_weight, C, D);
    }
}

} // namespace modules

#endif // LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_QUANTIZATION_H
