// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_HELPERS_H
#define LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_HELPERS_H

#include <utility>

#include "modules/weights/weight_manager.h"
#include "kernels/kernels.h"

namespace modules {

template<typename Block>
inline const typename ModularWeightManager<Block>::Config& ModularWeightManager<Block>::config() const {
    return mConfig;
}

template<typename Block>
inline int ModularWeightManager<Block>::num_layers() const {
    return mConfig.num_layers;
}

template<typename Block>
inline bool ModularWeightManager<Block>::has_fp8_forward_cache() const {
    return mConfig.enable_fp8_forward || (mExternalFP8CacheProvider != nullptr);
}

template<typename Block>
inline typename ModularWeightManager<Block>::FP8WeightCache& ModularWeightManager<Block>::fp8_weight_cache() {
    if (mExternalFP8CacheProvider) {
        return mExternalFP8CacheProvider();
    }
    return mFP8WeightCache;
}

template<typename Block>
inline const typename ModularWeightManager<Block>::FP8WeightCache& ModularWeightManager<Block>::fp8_weight_cache() const {
    if (mExternalFP8CacheProvider) {
        return mExternalFP8CacheProvider();
    }
    return mFP8WeightCache;
}

template<typename Block>
inline void ModularWeightManager<Block>::set_fp8_cache_provider(FP8CacheProvider provider) {
    mExternalFP8CacheProvider = std::move(provider);
}

template<typename Block>
inline void ModularWeightManager<Block>::clear_fp8_cache_provider() {
    mExternalFP8CacheProvider = nullptr;
}

template<typename Block>
inline bool ModularWeightManager<Block>::has_fp4_forward_cache() const {
    return mConfig.enable_fp4_forward;
}

template<typename Block>
inline typename ModularWeightManager<Block>::FP4WeightCache& ModularWeightManager<Block>::fp4_weight_cache() {
    return mFP4WeightCache;
}

template<typename Block>
inline const typename ModularWeightManager<Block>::FP4WeightCache& ModularWeightManager<Block>::fp4_weight_cache() const {
    return mFP4WeightCache;
}

template<typename Block>
inline bool ModularWeightManager<Block>::has_fp4_dgrad_cache() const {
    return mFP4PersistentCacheEnabled;
}

template<typename Block>
inline typename ModularWeightManager<Block>::FP4WeightCache& ModularWeightManager<Block>::fp4_weight_cache_transposed() {
    return mFP4WeightCacheT;
}

template<typename Block>
inline const typename ModularWeightManager<Block>::FP4WeightCache& ModularWeightManager<Block>::fp4_weight_cache_transposed() const {
    return mFP4WeightCacheT;
}

template<typename Block>
inline void ModularWeightManager<Block>::set_four_over_six(bool enable, recipes::FourOverSixErrorMetric metric) {
    mConfig.enable_four_over_six = enable;
    mConfig.four_over_six_metric = metric;
}

template<typename Block>
inline Tensor& ModularWeightManager<Block>::fp4_weight_amax() {
    return mFP4WeightAmax;
}

template<typename Block>
inline const Tensor& ModularWeightManager<Block>::fp4_weight_amax() const {
    return mFP4WeightAmax;
}

template<typename Block>
inline Tensor& ModularWeightManager<Block>::fp4_weight_amax_transposed() {
    return mFP4WeightAmaxT;
}

template<typename Block>
inline const Tensor& ModularWeightManager<Block>::fp4_weight_amax_transposed() const {
    return mFP4WeightAmaxT;
}

template<typename Block>
inline void ModularWeightManager<Block>::set_weight_provider(WeightProvider provider) {
    mExternalWeightProvider = std::move(provider);
}

template<typename Block>
inline void ModularWeightManager<Block>::set_embeddings_provider(NonBlockProvider provider) {
    mExternalEmbeddingsProvider = std::move(provider);
}

template<typename Block>
inline void ModularWeightManager<Block>::set_final_norm_provider(NonBlockProvider provider) {
    mExternalFinalNormProvider = std::move(provider);
}

template<typename Block>
inline void ModularWeightManager<Block>::set_lm_head_provider(NonBlockProvider provider) {
    mExternalLMHeadProvider = std::move(provider);
}

template<typename Block>
inline void ModularWeightManager<Block>::clear_weight_provider() {
    mExternalWeightProvider = nullptr;
    mExternalEmbeddingsProvider = nullptr;
    mExternalFinalNormProvider = nullptr;
    mExternalLMHeadProvider = nullptr;
}

template<typename Block>
inline bool ModularWeightManager<Block>::has_weight_provider() const {
    return static_cast<bool>(mExternalWeightProvider);
}

template<typename Block>
inline void ModularWeightManager<Block>::invalidate() {
    ++mVersion;
}

template<typename Block>
inline int ModularWeightManager<Block>::find_free_buffer(const std::array<GatherStatus, 2>& status) const {
    for (int i = 0; i < 2; ++i) {
        if (status[i].is_ready) return i;
    }
    return -1;
}

template<typename Block>
inline void ModularWeightManager<Block>::wait_for_buffer(GatherStatus& status, cudaStream_t stream) const {
    if (status.fetch_pending) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, status.done_event));
        status.fetch_pending = false;
    }
}

} // namespace modules

#endif // LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_HELPERS_H
