#pragma once
#include "ops/kernel/sampling_exact.cuh"
#include "core/device.h"
#include "ops/kernel/func_attribute.cuh"
#include <cub/block/block_radix_sort.cuh>

#include <algorithm>

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

// Which rows of a call a route takes, staged once per CTA. Several launches of a sampling
// call have work only in rows of one route -- the sort and merge passes in sorted-route
// rows, the candidate tiles in top-k rows -- and are launched whatever the rows ask, so a
// captured graph keeps one node set. With a CTA per work unit, a round whose rows all
// take another route still scheduled every unit's CTA, and each read its row's config
// before it could return: at 64 columns of a 151,936-token vocabulary, 9,536 sort CTAs,
// 8 merge passes of 38,016 and 19,008 candidate tiles, 251 us of an all-greedy round on
// an H100. These launches now run as many CTAs as the device holds at once. Each CTA
// counts the admitted (row, column) pairs of up to kSamplerStagedRows rows into a prefix
// in shared memory and strides over the admitted units only, so a launch with nothing
// to do costs one wave and one config read a row, and one with work runs the same units
// at the same occupancy. (Stepping over the whole [batch][width][units] space instead,
// skipping a row a jump at a time, kept an all-greedy merge pass at 31 us: about 36 steps
// a CTA at 64 rows.)
inline constexpr int kSamplerStagedRows = kSamplerBlock;

struct SamplingRowAdmission {
    int first[kSamplerStagedRows + 1]; // admitted pairs before each staged row; [count] = all
    int warp_total[kSamplerBlock / 32];
};

// Stages rows [base, base + count): row r admits min(width, extent + 1) columns when
// take(r) holds (every column without `extents`). Returns the admitted pairs.
template <class Take>
__device__ inline int sampling_stage_rows(SamplingRowAdmission& rows, int base, int count, int width,
                                          const int32_t* extents, Take&& take) {
    static_assert(kSamplerStagedRows == kSamplerBlock, "one thread stages each row");
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    int cols = 0;
    if (static_cast<int>(threadIdx.x) < count && take(base + static_cast<int>(threadIdx.x))) {
        cols = extents ? min(width, max(0, extents[base + threadIdx.x]) + 1) : width;
    }
    int scan = cols; // inclusive within the warp
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, scan, d);
        if (lane >= d) { scan += other; }
    }
    if (lane == 31) { rows.warp_total[warp] = scan; }
    __syncthreads();
    if (warp == 0) {
        int total = lane < kSamplerBlock / 32 ? rows.warp_total[lane] : 0;
#pragma unroll
        for (int d = 1; d < kSamplerBlock / 32; d <<= 1) {
            const int other = __shfl_up_sync(0xffffffffu, total, d);
            if (lane >= d) { total += other; }
        }
        if (lane < kSamplerBlock / 32) { rows.warp_total[lane] = total; }
    }
    __syncthreads();
    const int before = (warp > 0 ? rows.warp_total[warp - 1] : 0) + scan - cols;
    rows.first[threadIdx.x] = before;
    if (threadIdx.x == kSamplerBlock - 1) { rows.first[kSamplerStagedRows] = before + cols; }
    __syncthreads();
    return rows.first[count];
}

// Calls body(row, col, unit) for every admitted unit of a [batch][width][units] space, CTAs
// striding over the admitted ones from blockIdx.x, kSamplerStagedRows rows at a time. The
// walk is uniform across a CTA, so the body may synchronise.
template <class Take, class Body>
__device__ inline void sampling_for_each_unit(SamplingRowAdmission& rows, int batch, int width, int units,
                                              const int32_t* extents, Take&& take, Body&& body) {
    for (int base = 0; base < batch; base += kSamplerStagedRows) {
        const int count = min(batch - base, kSamplerStagedRows);
        const int total = sampling_stage_rows(rows, base, count, width, extents, take) * units;
        for (int item = static_cast<int>(blockIdx.x); item < total; item += static_cast<int>(gridDim.x)) {
            const int pair = item / units;
            int low = 0, high = count - 1; // the staged row holding `pair`
            while (low < high) {
                const int mid = (low + high + 1) / 2;
                if (rows.first[mid] <= pair) { low = mid; } else { high = mid - 1; }
            }
            body(base + low, pair - rows.first[low], item - pair * units);
        }
        __syncthreads(); // rows is restaged for the next chunk
    }
}

// Sort fixed-size tiles in parallel, then merge them without device-side allocation.
// The unique (adjusted logit, token ID) key preserves the sampler's exact tie order.
static __global__ void sampling_sort_tiles_kernel(const __nv_bfloat16* logits,
    unsigned long long* keys, const SamplingConfig* configs, const int32_t* drafts,
    const int32_t* extents, int domain, int physical_rows, int width, int batch, size_t row_stride) {
    __shared__ SamplingTileSort::TempStorage temp;
    __shared__ SamplingRowAdmission rows;
    const auto sorted_route = [&](int row) { return sampling_filtered_wide(configs[row]); };
    sampling_for_each_unit(rows, batch, width, div_up(domain, kSamplerSortTile), extents, sorted_route,
                           [&](int row, int col, int tile) {
        const SamplingConfig cfg = configs[row];
        const auto* overlay = drafts ? drafts + row * (width - 1) : nullptr;
        const auto* column = logits + (int64_t(row) * width + col) * physical_rows;
        auto* out = reinterpret_cast<unsigned long long*>(reinterpret_cast<char*>(keys) + row * row_stride) + int64_t(col) * domain;
        const int base = tile * kSamplerSortTile;
        unsigned long long items[kSamplerSortItems];
        for (int i = 0; i < kSamplerSortItems; ++i) {
            const int v = base + threadIdx.x + i * kSamplerBlock;
            float x = v < domain ? sampling_adjusted_logit(__bfloat162float(column[v]), v, cfg, overlay, col) : -CUDART_INF_F;
            items[i] = isfinite(x) ? sampling_sort_key(x, v) : 0;
        }
        SamplingTileSort(temp).SortDescendingBlockedToStriped(items);
        for (int i = 0; i < kSamplerSortItems; ++i) {
            const int v = base + threadIdx.x + i * kSamplerBlock;
            if (v < domain) { out[v] = items[i]; }
        }
        __syncthreads(); // temp is reused by the next tile
    });
}

static __global__ void sampling_merge_keys_kernel(const unsigned long long* source,
    unsigned long long* destination, const SamplingConfig* configs, const int32_t* extents,
    int domain, int width, int batch, int run, size_t row_stride) {
    __shared__ SamplingRowAdmission rows;
    const auto sorted_route = [&](int row) { return sampling_filtered_wide(configs[row]); };
    sampling_for_each_unit(rows, batch, width, div_up(domain, kSamplerBlock), extents, sorted_route,
                           [&](int row, int col, int block) {
        const int index = block * kSamplerBlock + threadIdx.x;
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
    });
}

// The grid for `items` work units of `kernel`: every unit, or as many CTAs as the
// device keeps resident at once when that is fewer.
template <class Kernel>
inline unsigned int sampling_resident_grid(Kernel* kernel, int64_t items) {
    const int resident = cooperative_capacity_per_device(kernel, kSamplerBlock, 0);
    const int64_t grid = resident > 0 ? std::min<int64_t>(items, resident) : items;
    return static_cast<unsigned int>(std::max<int64_t>(grid, 1));
}

// row_stride is the caller-owned layout stride, including padding for speculative rows.
inline const unsigned long long* sampling_sort_launch(const Tensor& logits, int domain,
    const SamplingConfig* configs, const int32_t* drafts, const int32_t* extents,
    int width, int batch, SamplingWorkspace scratch, size_t row_stride, cudaStream_t stream) {
    if (domain <= kSamplerSortTile) { return nullptr; }
    auto* input = scratch.sort_keys_a;
    auto* output = scratch.sort_keys_b;
    const int64_t tiles = int64_t(div_up(domain, kSamplerSortTile)) * width * batch;
    sampling_sort_tiles_kernel<<<sampling_resident_grid(sampling_sort_tiles_kernel, tiles), kSamplerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), input, configs, drafts, extents,
        domain, logits.ne[0], width, batch, row_stride);
    CUDA_CHECK(cudaGetLastError());
    const int64_t blocks = int64_t(div_up(domain, kSamplerBlock)) * width * batch;
    const unsigned int merge_grid = sampling_resident_grid(sampling_merge_keys_kernel, blocks);
    for (int64_t run = kSamplerSortTile; run < domain; run *= 2) {
        sampling_merge_keys_kernel<<<merge_grid, kSamplerBlock, 0, stream>>>(
            input, output, configs, extents, domain, width, batch, static_cast<int>(run), row_stride);
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
