#pragma once
#include "ops/kernel/sampling_device.cuh"

namespace sinfer::ops {
// Wide filters use radix selection over the complete vocabulary. No candidate
// truncation or vocabulary-sized sort buffer is needed. Each bisection fixes
// a bit of the ordered (logit, token-id) cutoff; ties use the usual lower ID.
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
__device__ inline ExactSample sampling_exact(
    const __nv_bfloat16* logits, int domain, const SamplingConfig& cfg, int position, int purpose,
    ExactSamplingScratch& scratch, int candidate = -1, int excluded = -1,
    const int32_t* overlay = nullptr, int overlay_len = 0) {
    float max = -CUDART_INF_F;
    for (int v = threadIdx.x; v < domain; v += blockDim.x) {
        max = fmaxf(max, sampling_adjusted_logit(__bfloat162float(logits[v]), v, cfg, overlay, overlay_len));
    }
    max = exact_max(max, scratch);
    unsigned long long cutoff = 0;
    if (cfg.top_k > 0 && cfg.top_k < domain) {
        double count = 0;
        for (int v = threadIdx.x; v < domain; v += blockDim.x) {
            count += isfinite(sampling_adjusted_logit(__bfloat162float(logits[v]), v, cfg, overlay, overlay_len));
        }
        const int k = min(cfg.top_k, static_cast<int>(exact_sum(count, scratch)));
        unsigned long long lo = 0, hi = ~0ULL;
        while (lo < hi) {
            const auto mid = lo + (hi - lo) / 2 + 1;
            double above = 0;
            for (int v = threadIdx.x; v < domain; v += blockDim.x) {
                const float x = sampling_adjusted_logit(__bfloat162float(logits[v]), v, cfg, overlay, overlay_len);
                above += isfinite(x) && sampling_sort_key(x, v) >= mid;
            }
            if (exact_sum(above, scratch) >= k) { lo = mid; } else { hi = mid - 1; }
        }
        cutoff = lo;
    }
    const float floor = cfg.min_p > 0 ? max + cfg.temperature * logf(cfg.min_p) : -CUDART_INF_F;
    if (cfg.top_p < 1.0F) {
        double total = 0;
        for (int v = threadIdx.x; v < domain; v += blockDim.x) {
            const float x = sampling_adjusted_logit(__bfloat162float(logits[v]), v, cfg, overlay, overlay_len);
            if (isfinite(x) && x >= floor && sampling_sort_key(x, v) >= cutoff) { total += exp((double(x) - max) / cfg.temperature); }
        }
        total = exact_sum(total, scratch);
        // top_p=0 retains the best token, just as the bounded path does.
        const double target = fmax(double(cfg.top_p) * total, 1e-300);
        unsigned long long lo = cutoff, hi = ~0ULL;
        while (lo < hi) {
            const auto mid = lo + (hi - lo) / 2 + 1;
            double mass = 0;
            for (int v = threadIdx.x; v < domain; v += blockDim.x) {
                const float x = sampling_adjusted_logit(__bfloat162float(logits[v]), v, cfg, overlay, overlay_len);
                if (isfinite(x) && x >= floor && sampling_sort_key(x, v) >= mid) { mass += exp((double(x) - max) / cfg.temperature); }
            }
            if (exact_sum(mass, scratch) >= target) { lo = mid; } else { hi = mid - 1; }
        }
        cutoff = lo;
    }
    double total = 0;
    unsigned long long best = 0;
    for (int v = threadIdx.x; v < domain; v += blockDim.x) {
        const float x = sampling_adjusted_logit(__bfloat162float(logits[v]), v, cfg, overlay, overlay_len);
        if (!isfinite(x) || x < floor || sampling_sort_key(x, v) < cutoff) { continue; }
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
        if (isfinite(x) && x >= floor && sampling_sort_key(x, candidate) >= cutoff && total > 0) {
            probability = static_cast<float>(exp((double(x) - max) / cfg.temperature) / total);
        }
    }
    // Only thread 0 consumes token; every thread sees the same probability.
    return {sampling_key_index(best), probability};
}

static __global__ void sampling_wide_kernel(const __nv_bfloat16* logits, int32_t* out,
    const SamplingConfig* configs, const int32_t* positions, int purpose, int domain, int physical_rows) {
    const int row = blockIdx.x;
    const SamplingConfig cfg = configs[row];
    if (!sampling_wide(cfg) || sampling_untruncated(cfg)) { return; }
    __shared__ ExactSamplingScratch scratch;
    const auto result = sampling_exact(logits + int64_t(row) * physical_rows, domain, cfg,
                                       positions[row], purpose, scratch);
    if (threadIdx.x == 0) {
        out[row] = result.token;
        if (cfg.token_counts && result.token < domain) { atomicAdd(cfg.token_counts + result.token, 1); }
    }
}
} // namespace sinfer::ops
