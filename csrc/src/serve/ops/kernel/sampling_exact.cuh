#pragma once
#include "ops/kernel/sampling_device.cuh"

namespace sinfer::ops {
// Full-vocabulary untruncated speculative probabilities and residual draws.
__device__ inline bool sampling_wide(const SamplingConfig& cfg) {
    return cfg.temperature > 0 && (cfg.top_k <= 0 || cfg.top_k > kSamplerFastCandidates);
}
struct ExactSamplingScratch {
    double sum[kSamplerBlock];
    float max[kSamplerBlock];
    unsigned long long keys[kSamplerBlock / 32];
};
__device__ inline double exact_sum(double value, ExactSamplingScratch& scratch) {
    scratch.sum[threadIdx.x] = value;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step /= 2) {
        if (threadIdx.x < step) { scratch.sum[threadIdx.x] += scratch.sum[threadIdx.x + step]; }
        __syncthreads();
    }
    const double result = scratch.sum[0];
    __syncthreads();
    return result;
}
__device__ inline float exact_max(float value, ExactSamplingScratch& scratch) {
    scratch.max[threadIdx.x] = value;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step /= 2) {
        if (threadIdx.x < step) { scratch.max[threadIdx.x] = fmaxf(scratch.max[threadIdx.x], scratch.max[threadIdx.x + step]); }
        __syncthreads();
    }
    const float result = scratch.max[0];
    __syncthreads();
    return result;
}
struct ExactSample { int token; float candidate_probability; };
__device__ inline ExactSample sampling_untruncated_exact(
    const __nv_bfloat16* logits, int domain, const SamplingConfig& cfg, int position, int purpose,
    ExactSamplingScratch& scratch, int candidate = -1, int excluded = -1,
    const int32_t* overlay = nullptr, int overlay_len = 0) {
    float max = -CUDART_INF_F;
    for (int v = threadIdx.x; v < domain; v += blockDim.x) {
        max = fmaxf(max, sampling_adjusted_logit(__bfloat162float(logits[v]), v, cfg, overlay, overlay_len));
    }
    max = exact_max(max, scratch);
    const float floor = cfg.min_p > 0 ? max + cfg.temperature * logf(cfg.min_p) : -CUDART_INF_F;
    double total = 0;
    unsigned long long best = 0;
    for (int v = threadIdx.x; v < domain; v += blockDim.x) {
        const float x = sampling_adjusted_logit(__bfloat162float(logits[v]), v, cfg, overlay, overlay_len);
        if (!isfinite(x) || x < floor) { continue; }
        total += exp((double(x) - max) / cfg.temperature);
        if (v != excluded) {
            const auto key = sampling_sort_key((x - max) / cfg.temperature + sampling_gumbel(cfg.seed, position, purpose, v), v);
            if (key > best) { best = key; }
        }
    }
    total = exact_sum(total, scratch);
    best = sampling_block_max_key(best, scratch.keys);
    float probability = 0;
    if (candidate >= 0 && candidate < domain) {
        const float x = sampling_adjusted_logit(__bfloat162float(logits[candidate]), candidate, cfg, overlay, overlay_len);
        if (isfinite(x) && x >= floor && total > 0) {
            probability = static_cast<float>(exp((double(x) - max) / cfg.temperature) / total);
        }
    }
    // Only thread 0 consumes token; every thread sees the same probability.
    return {sampling_key_index(best), probability};
}

} // namespace sinfer::ops
