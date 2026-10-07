#pragma once

// Implements: include/sinfer/ops/sampling.h
// Match: contiguous BF16 logits, physical stride >= token domain, and at most
// sixteen columns on the multi-block route.
// Algorithm assumptions: 256-thread/2-item partial tiles feed bounded top-20
// group merges through caller-owned workspace; unsupported finite geometries
// use the semantically identical single-block fallback.

#include "ops/kernel/sampling_sorted.cuh"

namespace sinfer::ops {

/// One token per column, drawn from the whole vocabulary.
///
/// The candidate routes below keep the best `cap` tokens and draw from those. That
/// is what top_k asks for, but a row that asked for *no* limit was getting the same
/// twenty -- so the tail of the distribution could not be sampled at all, and a
/// trainer comparing its own full-vocabulary log-probability against ours was
/// comparing against a different policy than the one that generated.
///
/// This route draws exactly, in one pass, by the Gumbel-max identity: adding an
/// independent Gumbel(0,1) to each scaled logit and taking the argmax is a draw
/// from the softmax over every token. No sort, no normalisation, no workspace --
/// and one block per column, which is affordable because it is one more read of a
/// logits row the head has already produced.
///
/// Only rows `sampling_untruncated` claims are handled; the others return at once
/// and are served by the candidate routes, which skip these rows in turn. Exactly
/// one route writes each column, so neither the token counts nor the output is
/// touched twice.
__global__ void sampling_update_greedy_targets_kernel(
    const __nv_bfloat16* logits, std::int32_t* targets, const SamplingConfig* configs,
    int domain, int physical_rows, int width) {
    const int col = blockIdx.x;
    const int row = blockIdx.y;
    const SamplingConfig cfg = configs[row];
    if (cfg.temperature > 0.0F || (cfg.logit_bias == nullptr && cfg.token_bitmask == nullptr && cfg.suppressed_count == 0)) { return; }
    __shared__ unsigned long long warp_keys[kSamplerBlock / 32];
    unsigned long long best = 0;
    const auto base = (static_cast<std::int64_t>(row) * width + col) * physical_rows;
    for (int v = threadIdx.x; v < domain; v += blockDim.x) {
        if (sampling_suppressed(cfg, v, col)) { continue; }
        const float value = __bfloat162float(logits[base + v]) +
                            (cfg.logit_bias != nullptr ? cfg.logit_bias[v] : 0.0F);
        const auto key = sampling_sort_key(value, v);
        if (key > best) { best = key; }
    }
    best = sampling_block_max_key(best, warp_keys);
    if (threadIdx.x == 0) { targets[row * width + col] = sampling_key_index(best); }
}

__launch_bounds__(kSamplerBlock) __global__
    void sampling_full_vocab_kernel(const __nv_bfloat16* logits, std::int32_t* out,
                                    const SamplingConfig* configs,
                                    const std::int32_t* logical_positions, std::int32_t purpose,
                                    std::int32_t token_domain, std::int32_t physical_rows) {
    const int col            = static_cast<int>(blockIdx.x);
    const SamplingConfig cfg = configs[col];
    if (!sampling_untruncated(cfg)) { return; }

    const std::int64_t base = static_cast<std::int64_t>(col) * physical_rows;
    const int tid           = static_cast<int>(threadIdx.x);
    const float inv_temp    = 1.0f / cfg.temperature;
    const int position      = logical_positions[col];

    __shared__ float red_val[kSamplerBlock];
    __shared__ int red_idx[kSamplerBlock];

    // min_p is a floor on a token's probability relative to the most likely one,
    // which on the scaled logits is a shift of the row maximum -- so it needs the
    // maximum, and only then. Without it the argmax below is shift-invariant and
    // this pass would buy nothing.
    float floor_value = -CUDART_INF_F;
    if (cfg.min_p > 0.0f) {
        float local = -CUDART_INF_F;
        for (int v = tid; v < token_domain; v += kSamplerBlock) {
            local = fmaxf(local,
                          sampling_adjusted_logit(__bfloat162float(logits[base + v]), v, cfg) *
                              inv_temp);
        }
        red_val[tid] = local;
        __syncthreads();
        for (int s = kSamplerBlock / 2; s > 0; s >>= 1) {
            if (tid < s) { red_val[tid] = fmaxf(red_val[tid], red_val[tid + s]); }
            __syncthreads();
        }
        floor_value = red_val[0] + __logf(cfg.min_p);
        __syncthreads();
    }

    float best = -CUDART_INF_F;
    int best_v = INT_MAX;
    for (int v = tid; v < token_domain; v += kSamplerBlock) {
        const float x =
            sampling_adjusted_logit(__bfloat162float(logits[base + v]), v, cfg) * inv_temp;
        if (x < floor_value) { continue; }
        const float z = x + sampling_gumbel(cfg.seed, position, purpose, static_cast<unsigned>(v));
        if (sampling_better(z, v, best, best_v)) {
            best   = z;
            best_v = v;
        }
    }
    red_val[tid] = best;
    red_idx[tid] = best_v;
    __syncthreads();
    for (int s = kSamplerBlock / 2; s > 0; s >>= 1) {
        if (tid < s &&
            sampling_better(red_val[tid + s], red_idx[tid + s], red_val[tid], red_idx[tid])) {
            red_val[tid] = red_val[tid + s];
            red_idx[tid] = red_idx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        // Every token can fall below the min_p floor only if the floor is above the
        // maximum, which it cannot be; the guard is for a row of non-finite logits.
        const int picked = red_idx[0] == INT_MAX ? 0 : red_idx[0];
        out[col]         = picked;
        if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
    }
}

__launch_bounds__(kSamplerBlock) __global__
    void sample_row_kernel(const __nv_bfloat16* logits, std::int32_t* out,
                           const SamplingConfig* configs, const std::int32_t* logical_positions,
                           std::int32_t purpose, std::int32_t token_domain,
                           std::int32_t physical_rows) {
    const int row            = static_cast<int>(blockIdx.x);
    const std::int64_t base  = static_cast<std::int64_t>(row) * physical_rows;
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = configs[row];

    __shared__ float red_val[kSamplerBlock];
    __shared__ int red_idx[kSamplerBlock];

    // Greedy: exact argmax over raw logits. Bit-identical to argmax().
    if (!(cfg.temperature > 0.0f)) {
        float bv = -CUDART_INF_F;
        int bi   = INT_MAX;
        for (int v = tid; v < token_domain; v += blockDim.x) {
            if (sampling_suppressed(cfg, v)) { continue; }
            const float x = __bfloat162float(logits[base + v]) +
                            (cfg.logit_bias != nullptr ? cfg.logit_bias[v] : 0.0F);
            if (sampling_better(x, v, bv, bi)) {
                bv = x;
                bi = v;
            }
        }
        red_val[tid] = bv;
        red_idx[tid] = bi;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s &&
                sampling_better(red_val[tid + s], red_idx[tid + s], red_val[tid], red_idx[tid])) {
                red_val[tid] = red_val[tid + s];
                red_idx[tid] = red_idx[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) { out[row] = red_idx[0]; }
        return;
    }

    // Untruncated rows belong to sampling_full_vocab_kernel. Returning before the
    // draw is what keeps the token counts single-counted.
    if (sampling_wide(cfg)) { return; }

    const int partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const int group_count    = sampler_group_count(partial_blocks);
    // No-op when the scratch/group path owns this shape (see sample_batch_launch).
    if (sampler_multiblock_ok(token_domain, static_cast<int>(gridDim.x), partial_blocks,
                              group_count)) {
        return;
    }

    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ float merge_val[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int merge_idx[kSamplerBlock * kSamplerFastCandidates];

    if (token_domain <= kSamplerTileItems) {
        sampling_build_truncated_small(logits, base, token_domain, cfg, red_val, red_idx, cand_val,
                                       cand_idx, prob, &n_support);
    } else {
        sampling_build_truncated_block_fast(logits, base, token_domain, cfg, merge_val, merge_idx,
                                            cand_val, cand_idx, prob, &n_support);
    }

    if (tid != 0) { return; }
    const int support = n_support;
    const float u     = sampling_uniform(cfg.seed, logical_positions[row], purpose, 0u);
    float acc         = 0.0f;
    int picked        = cand_idx[support - 1];
    for (int j = 0; j < support; ++j) {
        acc += prob[j]; // prob is normalized: goal == u
        if (u < acc) {
            picked = cand_idx[j];
            break;
        }
    }
    out[row] = picked;
    if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
}

/// Greedy columns: the bias-adjusted argmax over the allowed tokens, ties to the
/// lower id, with token_counts left alone -- what the candidate route's greedy branch
/// produced, from one key per token, so the result is identical however the row is
/// split. Each CTA takes one kSamplerGreedyChunkItems chunk of one row and the last
/// to finish reduces the chunk maxima.
///
/// The candidate route read a greedy row in 512-item tiles, one CTA each: at 64
/// columns of a 151,936-token vocabulary that was 19,008 CTAs and 81 us for 19 MB
/// on an H100, bound by scheduling and two outstanding 2-byte loads per thread.
/// Rows without bias or a barred token are read 16 bytes at a time.
///
/// Launched after sampling_partial_topk_kernel, whose first CTA zeroes the row's
/// group_done counter; the last CTA here zeroes it again for the next call.
__launch_bounds__(kSamplerBlock) __global__
    void sampling_greedy_kernel(const __nv_bfloat16* logits, std::int32_t* out,
                                const SamplingConfig* cfg_ptr, std::int32_t token_domain,
                                std::int32_t physical_rows, std::int32_t chunks,
                                SamplingWorkspace workspace) {
    const int col = static_cast<int>(blockIdx.y);
    const int chunk = static_cast<int>(blockIdx.x);
    __shared__ unsigned long long warp_keys[kSamplerBlock / 32];
    __shared__ int is_last;

    const __nv_bfloat16* row = logits + static_cast<std::int64_t>(col) * physical_rows;
    const int begin          = chunk * kSamplerGreedyChunkItems;
    const int end            = min(token_domain, begin + kSamplerGreedyChunkItems);
    // The chunk is requested before the row's config arrives, so the two latencies
    // overlap; a stochastic row's CTA reads its chunk for nothing, from L2, where the
    // head just left it. The config's few fields are read through a reference.
    constexpr int kVector = 8;
    constexpr int kUnroll = kSamplerGreedyChunkItems / (kSamplerBlock * kVector);
    const bool aligned    = (reinterpret_cast<std::uintptr_t>(row + begin) & 15u) == 0;
    const int vector_end  = aligned ? begin + (end - begin) / kVector * kVector : begin;
    uint4 packed[kUnroll];
#pragma unroll
    for (int step = 0; step < kUnroll; ++step) {
        const int v   = begin + (step * kSamplerBlock + static_cast<int>(threadIdx.x)) * kVector;
        packed[step]  = v < vector_end ? *reinterpret_cast<const uint4*>(row + v) : make_uint4(0, 0, 0, 0);
    }
    const SamplingConfig& cfg = cfg_ptr[col];
    if (cfg.temperature > 0.0f) { return; }
    const float* bias = cfg.logit_bias;
    const bool plain  = bias == nullptr && cfg.token_bitmask == nullptr && cfg.suppressed_count == 0;
    unsigned long long best = 0ull;
    int scalar_begin        = begin;
    if (plain) {
#pragma unroll
        for (int step = 0; step < kUnroll; ++step) {
            const int v = begin + (step * kSamplerBlock + static_cast<int>(threadIdx.x)) * kVector;
            if (v < vector_end) {
                const auto* values = reinterpret_cast<const __nv_bfloat16*>(&packed[step]);
#pragma unroll
                for (int i = 0; i < kVector; ++i) {
                    // + 0: the sum the biased path forms, so a denormal flushes the same way.
                    const unsigned long long key = sampling_sort_key(__bfloat162float(values[i]) + 0.0F, v + i);
                    if (key > best) { best = key; }
                }
            }
        }
        scalar_begin = vector_end;
    }
    for (int v = scalar_begin + static_cast<int>(threadIdx.x); v < end; v += kSamplerBlock) {
        // The raw logit plus bias: a greedy row takes no temperature or penalty, and
        // a barred token is skipped outright.
        if (!plain && sampling_suppressed(cfg, v)) { continue; }
        const float x = __bfloat162float(row[v]) + (bias != nullptr ? bias[v] : 0.0F);
        const unsigned long long key = sampling_sort_key(x, v);
        if (key > best) { best = key; }
    }
    best = sampling_block_max_key(best, warp_keys);
    if (threadIdx.x == 0) {
        workspace.partial_keys[sampling_partial_offset(workspace, col, chunk, 0)] = best;
        __threadfence();
        is_last = atomicAdd(&workspace.group_done[col], 1) + 1 == chunks ? 1 : 0;
    }
    __syncthreads();
    if (!is_last) { return; }
    best = 0ull;
    for (int p = static_cast<int>(threadIdx.x); p < chunks; p += kSamplerBlock) {
        const unsigned long long key = __ldcg(&workspace.partial_keys[sampling_partial_offset(workspace, col, p, 0)]);
        if (key > best) { best = key; }
    }
    best = sampling_block_max_key(best, warp_keys);
    if (threadIdx.x == 0) {
        out[col]                  = sampling_key_index(best);
        workspace.group_done[col] = 0;
    }
}

// Candidate tiles of the rows sampled from a bounded candidate set (a stochastic row that
// is neither wide nor greedy): the best `cap` keys of each 512-token tile. A resident
// grid walks the [batch][partial] tiles (see sampling_for_each_unit); the first CTA zeroes
// every row's group_done counter, which the greedy and finalize launches after it count on.
__launch_bounds__(kSamplerBlock) __global__
    void sampling_partial_topk_kernel(const __nv_bfloat16* logits, const SamplingConfig* cfg_ptr,
                                      std::int32_t token_domain, std::int32_t physical_rows,
                                      std::int32_t batch, SamplingWorkspace workspace) {
    __shared__ typename SamplingPartialSort::TempStorage sort_storage;
    __shared__ SamplingRowAdmission rows;
    if (blockIdx.x == 0) {
        for (int col = threadIdx.x; col < batch; col += blockDim.x) { workspace.group_done[col] = 0; }
    }
    const auto candidate_route = [&](int col) {
        const SamplingConfig& cfg = cfg_ptr[col];
        return cfg.temperature > 0.0f && !sampling_wide(cfg);
    };
    sampling_for_each_unit(rows, batch, 1, div_up(token_domain, kSamplerPartialTileItems), nullptr,
                           candidate_route, [&](int col, int, int partial) {
        const SamplingConfig cfg = cfg_ptr[col];
        unsigned long long keys[kSamplerItemsPerThread];
        const int cap           = sampling_candidate_cap(cfg, token_domain);
        const std::int64_t base = static_cast<std::int64_t>(col) * physical_rows;
        const int tile_start    = partial * kSamplerPartialTileItems;
#pragma unroll
        for (int item = 0; item < kSamplerItemsPerThread; ++item) {
            const int v = tile_start + item * blockDim.x + threadIdx.x;
            if (v < token_domain) {
                const float x = sampling_adjusted_logit(__bfloat162float(logits[base + v]), v, cfg);
                keys[item]    = sampling_sort_key(x, v);
            } else {
                keys[item] = 0ull;
            }
        }
        SamplingPartialSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
        for (int item = 0; item < kSamplerItemsPerThread; ++item) {
            const int rank = threadIdx.x * kSamplerItemsPerThread + item;
            if (rank < cap) {
                const int off               = sampling_partial_offset(workspace, col, partial, rank);
                workspace.partial_keys[off] = keys[item];
            }
        }
        __syncthreads(); // sort_storage is reused by the next tile
    });
}

__launch_bounds__(kSamplerGroupBlock) __global__ void sampling_group_finalize_sample_kernel(
    std::int32_t* out, const SamplingConfig* cfg_ptr, const std::int32_t* logical_positions,
    std::int32_t purpose, std::int32_t token_domain, std::int32_t partial_blocks,
    std::int32_t group_count, SamplingWorkspace workspace) {
    const int group          = static_cast<int>(blockIdx.x);
    const int col            = static_cast<int>(blockIdx.y);
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = cfg_ptr[col];
    __shared__ typename SamplingGroupSort::TempStorage sort_storage;
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ int is_last;
    unsigned long long keys[kSamplerGroupItemsPerThread];

    if (sampling_wide(cfg)) { return; } // sampling_full_vocab_kernel owns this column
    if (!(cfg.temperature > 0.0f)) { return; } // sampling_greedy_kernel owns this column
    const int cap = sampling_candidate_cap(cfg, token_domain);
    // The preceding partial launch initializes group_done[col], so caller-owned
    // workspace does not rely on prior contents or a separate memset launch.

    const int group_begin = group * kSamplerPartialsPerGroup;
    int group_partials    = partial_blocks - group_begin;
    if (group_partials < 0) { group_partials = 0; }
    if (group_partials > kSamplerPartialsPerGroup) { group_partials = kSamplerPartialsPerGroup; }

    const int group_n = group_partials * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < group_n) {
            const int partial = group_begin + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = workspace.partial_keys[off];
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingGroupSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            const int out_off =
                sampling_partial_offset(workspace, col, partial_blocks + group, rank);
            workspace.partial_keys[out_off] = keys[item];
        }
    }
    __syncthreads();

    if (tid == 0) {
        __threadfence();
        const int done = atomicAdd(&workspace.group_done[col], 1) + 1;
        is_last        = (done == group_count) ? 1 : 0;
    }
    __syncthreads();
    if (!is_last) { return; }

    const int final_n = group_count * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < final_n) {
            const int partial = partial_blocks + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = workspace.partial_keys[off];
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingGroupSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            cand_val[rank] = sampling_key_float(keys[item]);
            cand_idx[rank] = sampling_key_index(keys[item]);
        }
    }
    __syncthreads();

    sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);
    if (tid == 0) {
        const int support = n_support;
        const float u     = sampling_uniform(cfg.seed, logical_positions[col], purpose, 0u);
        float acc         = 0.0f;
        int picked        = cand_idx[support - 1];
        for (int j = 0; j < support; ++j) {
            acc += prob[j];
            if (u < acc) {
                picked = cand_idx[j];
                break;
            }
        }
        out[col] = picked;
        if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
        workspace.group_done[col] = 0;
    }
}

} // namespace sinfer::ops
