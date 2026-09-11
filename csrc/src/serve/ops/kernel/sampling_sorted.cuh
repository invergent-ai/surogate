#pragma once
#include "ops/kernel/sampling_exact.cuh"
#include "core/device.h"
#include <cub/block/block_radix_sort.cuh>

namespace sinfer::ops {
using SamplingTileSort = cub::BlockRadixSort<unsigned long long, kSamplerBlock, kSamplerSortItems>;

struct SortedSamplingScratch {
    SamplingTileSort::TempStorage sort;
    unsigned long long local_keys[kSamplerSortTile];
    unsigned long long warp_keys[kSamplerBlock / 32];
    double mass[kSamplerBlock];
    double normalizer;
    int support;
};

__device__ inline bool sampling_filtered_wide(const SamplingConfig& cfg) {
    return sampling_wide(cfg) && !sampling_untruncated(cfg);
}

// Sort fixed-size tiles in parallel, then merge them without device-side allocation.
// The unique (adjusted logit, token ID) key preserves the sampler's exact tie order.
static __global__ void sampling_sort_tiles_kernel(const __nv_bfloat16* logits,
    unsigned long long* keys, const SamplingConfig* configs, const int32_t* drafts,
    const int32_t* extents, int domain, int physical_rows, int width, size_t row_stride) {
    const int row = blockIdx.z, col = blockIdx.y;
    const SamplingConfig cfg = configs[row];
    if (!sampling_filtered_wide(cfg) || (extents && col > max(0, extents[row]))) { return; }
    const auto* overlay = drafts ? drafts + row * (width - 1) : nullptr;
    const auto* column = logits + (int64_t(row) * width + col) * physical_rows;
    auto* out = reinterpret_cast<unsigned long long*>(reinterpret_cast<char*>(keys) + row * row_stride) + int64_t(col) * domain;
    const int base = blockIdx.x * kSamplerSortTile;
    unsigned long long items[kSamplerSortItems];
    for (int i = 0; i < kSamplerSortItems; ++i) {
        const int v = base + threadIdx.x + i * kSamplerBlock;
        float x = v < domain ? sampling_adjusted_logit(__bfloat162float(column[v]), v, cfg, overlay, col) : -CUDART_INF_F;
        items[i] = isfinite(x) ? sampling_sort_key(x, v) : 0;
    }
    __shared__ SamplingTileSort::TempStorage temp;
    SamplingTileSort(temp).SortDescendingBlockedToStriped(items);
    for (int i = 0; i < kSamplerSortItems; ++i) {
        const int v = base + threadIdx.x + i * kSamplerBlock;
        if (v < domain) { out[v] = items[i]; }
    }
}

static __global__ void sampling_merge_keys_kernel(const unsigned long long* source,
    unsigned long long* destination, const SamplingConfig* configs, const int32_t* extents,
    int domain, int width, int run, size_t row_stride) {
    const int row = blockIdx.z, col = blockIdx.y;
    if (!sampling_filtered_wide(configs[row]) || (extents && col > max(0, extents[row]))) { return; }
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= domain) { return; }
    const auto* in = reinterpret_cast<const unsigned long long*>(reinterpret_cast<const char*>(source) + row * row_stride) + int64_t(col) * domain;
    auto* out = reinterpret_cast<unsigned long long*>(reinterpret_cast<char*>(destination) + row * row_stride) + int64_t(col) * domain;
    const int base = int(int64_t(index) / (2LL * run) * (2LL * run));
    const int left = min(run, domain - base), right = min(run, domain - base - left);
    const int diagonal = index - base;
    int low = max(0, diagonal - right), high = min(diagonal, left);
    // Merge-path partition for this output element; equal padding keys take the left run.
    while (low < high) {
        const int mid = low + (high - low + 1) / 2;
        const int other = diagonal - mid;
        if (other == right || in[base + mid - 1] >= in[base + left + other]) { low = mid; }
        else { high = mid - 1; }
    }
    const int a = low, b = diagonal - a;
    const auto ka = a < left ? in[base + a] : 0ULL;
    const auto kb = b < right ? in[base + left + b] : 0ULL;
    out[index] = max(ka, kb);
}

// row_stride is the caller-owned layout stride, including padding for speculative rows.
inline const unsigned long long* sampling_sort_launch(const Tensor& logits, int domain,
    const SamplingConfig* configs, const int32_t* drafts, const int32_t* extents,
    int width, int batch, SamplingWorkspace scratch, size_t row_stride, cudaStream_t stream) {
    if (domain <= kSamplerSortTile) { return nullptr; }
    auto* input = scratch.sort_keys_a;
    auto* output = scratch.sort_keys_b;
    sampling_sort_tiles_kernel<<<dim3(div_up(domain, kSamplerSortTile), width, batch), kSamplerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), input, configs, drafts, extents,
        domain, logits.ne[0], width, row_stride);
    CUDA_CHECK(cudaGetLastError());
    for (int64_t run = kSamplerSortTile; run < domain; run *= 2) {
        sampling_merge_keys_kernel<<<dim3(div_up(domain, kSamplerBlock), width, batch), kSamplerBlock, 0, stream>>>(
            input, output, configs, extents, domain, width, static_cast<int>(run), row_stride);
        CUDA_CHECK(cudaGetLastError());
        auto* previous = input; input = output; output = previous;
    }
    return input;
}

__device__ inline const unsigned long long* sampling_sort_local(const __nv_bfloat16* logits,
    int domain, const SamplingConfig& cfg, SortedSamplingScratch& scratch,
    const int32_t* overlay = nullptr, int col = 0) {
    unsigned long long items[kSamplerSortItems];
    for (int i = 0; i < kSamplerSortItems; ++i) {
        const int v = threadIdx.x + i * kSamplerBlock;
        const float x = v < domain ? sampling_adjusted_logit(__bfloat162float(logits[v]), v, cfg, overlay, col) : -CUDART_INF_F;
        items[i] = isfinite(x) ? sampling_sort_key(x, v) : 0;
    }
    SamplingTileSort(scratch.sort).SortDescendingBlockedToStriped(items);
    for (int i = 0; i < kSamplerSortItems; ++i) { scratch.local_keys[threadIdx.x + i * kSamplerBlock] = items[i]; }
    __syncthreads();
    return scratch.local_keys;
}

__device__ inline ExactSample sampling_sorted(const unsigned long long* sorted, int domain,
    const SamplingConfig& cfg, int position, int purpose, SortedSamplingScratch& scratch,
    unsigned long long candidate_key = 0, int excluded = -1) {
    if (!sorted[0]) { return {INT_MAX, 0}; }
    const float maximum = sampling_key_float(sorted[0]);
    if (threadIdx.x == 0) {
        int low = 0, high = cfg.top_k > 0 ? min(domain, cfg.top_k) : domain;
        const float floor = cfg.min_p > 0 ? maximum + cfg.temperature * logf(cfg.min_p) : -CUDART_INF_F;
        while (low < high) {
            const int mid = low + (high - low) / 2;
            if (sorted[mid] && sampling_key_float(sorted[mid]) >= floor) { low = mid + 1; }
            else { high = mid; }
        }
        scratch.support = low;
        scratch.normalizer = 1;
    }
    __syncthreads();
    if (cfg.top_p < 1 || candidate_key) {
        const int segment = div_up(scratch.support, kSamplerBlock);
        const int begin = threadIdx.x * segment, end = min(begin + segment, scratch.support);
        double mass = 0;
        for (int i = begin; i < end; ++i) {
            mass += exp((double(sampling_key_float(sorted[i])) - maximum) / cfg.temperature);
        }
        scratch.mass[threadIdx.x] = mass;
        __syncthreads();
        // Inclusive scan of contiguous segment masses locates the nucleus boundary.
        for (int step = 1; step < kSamplerBlock; step *= 2) {
            const double add = threadIdx.x >= step ? scratch.mass[threadIdx.x - step] : 0;
            __syncthreads();
            scratch.mass[threadIdx.x] += add;
            __syncthreads();
        }
        const double total = scratch.mass[kSamplerBlock - 1];
        const double prefix = threadIdx.x ? scratch.mass[threadIdx.x - 1] : 0;
        if (cfg.top_p < 1) {
            const double target = fmax(double(cfg.top_p) * total, 1e-300);
            if (prefix < target && scratch.mass[threadIdx.x] >= target) {
                double cumulative = prefix;
                for (int i = begin; i < end; ++i) {
                    cumulative += exp((double(sampling_key_float(sorted[i])) - maximum) / cfg.temperature);
                    if (cumulative >= target || i + 1 == end) {
                        scratch.support = i + 1;
                        scratch.normalizer = cumulative;
                        break;
                    }
                }
            }
        } else if (threadIdx.x == 0) { scratch.normalizer = total; }
        __syncthreads();
    }
    unsigned long long best = 0;
    for (int i = threadIdx.x; i < scratch.support; i += kSamplerBlock) {
        const auto key = sorted[i];
        const int token = sampling_key_index(key);
        if (token == excluded) { continue; }
        const auto draw = sampling_sort_key((sampling_key_float(key) - maximum) / cfg.temperature +
            sampling_gumbel(cfg.seed, position, purpose, token), token);
        if (draw > best) { best = draw; }
    }
    best = sampling_block_max_key(best, scratch.warp_keys);
    float probability = 0;
    if (candidate_key && scratch.support && candidate_key >= sorted[scratch.support - 1]) {
        probability = static_cast<float>(exp((double(sampling_key_float(candidate_key)) - maximum) / cfg.temperature) / scratch.normalizer);
    }
    return {sampling_key_index(best), probability};
}

static __global__ void sampling_wide_kernel(const __nv_bfloat16* logits, const unsigned long long* sorted,
    int32_t* out, const SamplingConfig* configs, const int32_t* positions, int purpose, int domain, int physical_rows) {
    const int row = blockIdx.x;
    const SamplingConfig cfg = configs[row];
    if (!sampling_filtered_wide(cfg)) { return; }
    __shared__ SortedSamplingScratch scratch;
    const auto* keys = sorted ? sorted + int64_t(row) * domain :
        sampling_sort_local(logits + int64_t(row) * physical_rows, domain, cfg, scratch);
    const auto result = sampling_sorted(keys, domain, cfg, positions[row], purpose, scratch);
    if (threadIdx.x == 0) {
        out[row] = result.token;
        if (cfg.token_counts && result.token < domain) { atomicAdd(cfg.token_counts + result.token, 1); }
    }
}
} // namespace sinfer::ops
