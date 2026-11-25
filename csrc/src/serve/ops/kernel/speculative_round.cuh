#pragma once

// Implements: include/sinfer/ops/speculative_round.h
// Match: contiguous request-major state and BF16 or FP32 verification logits.
// Algorithm assumptions: small vocabularies use one cooperative block; the
// registered full-vocabulary stochastic route uses the sampling partial/group
// pipeline and caller-owned workspace, while greedy commit remains one thread.

#include "ops/kernel/sampling_sorted.cuh"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops {

// Reply-phase sampler (SamplingConfig::phase_switch_armed): the drafts a row may verify. 1 (in its thought): those
// before its first drafted close token (phase_switch_token); 2 (its output has not yet said whether it thinks): none.
// 0 (every other row): all of them. Idempotent: the cut of a cut extent is itself.
__device__ inline int speculative_phase_cut(const std::int32_t* row_drafts, int extent, const SamplingConfig* configs,
                                            int row) {
    if (configs == nullptr) { return extent; }
    const SamplingConfig& cfg = configs[row];
    if (cfg.phase_switch_armed == 0) { return extent; }
    if (cfg.phase_switch_armed == 2) { return 0; }
    for (int i = 0; i < extent; ++i) {
        if (row_drafts[i] == cfg.phase_switch_token) { return i; }
    }
    return extent;
}

__global__ void speculative_prepare_verify_inputs_kernel(const std::int32_t* anchors,
                                                         const std::int32_t* drafts,
                                                         const std::int32_t* base_positions,
                                                         std::int32_t* current_extents,
                                                         std::int32_t* verify_ids,
                                                         std::int32_t* positions, std::int32_t k,
                                                         const SamplingConfig* configs) {
    // One block per row (the launcher's grid is (1, B)): the write-back below follows every thread's read.
    const int row = static_cast<int>(blockIdx.y);
    const int T   = k + 1;
    int extent    = current_extents[row];
    extent        = extent < 0 ? 0 : (extent > k ? k : extent);
    // Reply-phase cut: the accept reads the cut extent, but the verify keeps the round's shape -- its positions run
    // sequentially over the uncut extent, as the host staged the valid columns and RoPE positions -- and only the ids
    // past the cut are the anchor (a token sequence like any other, at its own positions and cache rows). So the
    // forward of every column up to the cut is exactly the option-off forward, and no cache row is written twice.
    int cut = extent;
    if (configs != nullptr && configs[row].phase_switch_armed != 0) {
        cut = speculative_phase_cut(drafts + row * k, extent, configs, row);
    }
    if (cut != extent) {
        __syncthreads();
        // No device printf here: a kernel that can print makes the runtime set up its printf buffer, which lands in
        // the CUDA Graph preparation window (+52 MiB measured) and can exceed the planned graph allowance.
        if (threadIdx.x == 0) { current_extents[row] = cut; }
    }
    for (int j = static_cast<int>(threadIdx.x); j < T; j += blockDim.x) {
        const int off = row * T + j;
        verify_ids[off] =
            j == 0 ? anchors[row] : (j <= cut ? drafts[row * k + j - 1] : anchors[row]);
        if (positions != nullptr) {
            positions[off] = base_positions[row] + (j <= extent ? j : extent);
        }
    }
}

template <typename T>
__device__ inline T* speculative_workspace_offset(T* ptr, std::size_t byte_offset) {
    return ptr == nullptr
               ? nullptr
               : reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(ptr) + byte_offset);
}

__device__ inline SamplingWorkspace
speculative_workspace_row(SamplingWorkspace workspace, std::size_t row_stride, std::int32_t row) {
    const std::size_t offset = static_cast<std::size_t>(row) * row_stride;
    workspace.partial_keys   = speculative_workspace_offset(workspace.partial_keys, offset);
    workspace.dist_idx       = speculative_workspace_offset(workspace.dist_idx, offset);
    workspace.dist_prob      = speculative_workspace_offset(workspace.dist_prob, offset);
    workspace.dist_support   = speculative_workspace_offset(workspace.dist_support, offset);
    workspace.group_done     = speculative_workspace_offset(workspace.group_done, offset);
    workspace.speculative_finalize_count =
        speculative_workspace_offset(workspace.speculative_finalize_count, offset);
    return workspace;
}

// Commits the round's accepted tokens plus one correction/bonus token, then
// advances the target length. The greedy branch
// (config temperature <= 0) is bit-identical to the original argmax accept: keep
// the longest draft prefix whose target argmax matches, then take the target
// argmax at the divergence column. The sampling branch (temperature > 0) runs
// distribution-correct speculative rejection sampling over the verify logits with
// a one-hot (greedy) draft: accept drafts[i] with probability p_i(drafts[i]) under
// the truncated target distribution, resample from the masked residual on the
// first rejection, and draw a bonus from the last column when every draft accepts.
// The draft-proposal path stays greedy, so q is one-hot and the accept test
// collapses to `u < p_i(drafts[i])`. Launch with a single block of kSamplerBlock
// threads; only thread 0 performs the sequential accept/commit while the whole
// block cooperates on the per-column truncated-distribution build.
template <typename LogitT>
__global__ void speculative_accept_wide_kernel(
    const LogitT* logits, const int32_t* drafts, const int32_t* extents,
    int32_t* lengths, int32_t* anchors, int32_t* tokens, int32_t* counts, int32_t* accepted,
    const SamplingConfig* configs, int domain, int physical_rows, int k,
    const unsigned long long* sorted, size_t sorted_row_stride, int skip_untruncated = 0) {
    const int row = blockIdx.x, width = k + 1;
    const SamplingConfig cfg = configs[row];
    if (!sampling_wide(cfg)) { return; }
    // Untruncated rows go to the split route below (speculative_exact_*_kernel) when the launcher runs it.
    if (skip_untruncated && sampling_untruncated(cfg)) { return; }
    const int extent = min(k, max(0, extents[row]));
    const int length = lengths[row];
    const auto* row_drafts = drafts + row * k;
    __shared__ union { ExactSamplingScratch untruncated; SortedSamplingScratch filtered; } scratch;
    __shared__ int stop;
    __shared__ int produced;
    for (int col = 0; col <= extent; ++col) {
        const int draft = col < extent ? row_drafts[col] : -1;
        const int purpose = col < extent ? kSamplePurposeSpeculativeCorrection : kSamplePurposeSpeculativeBonus;
        const auto* column = logits + (int64_t(row) * width + col) * physical_rows;
        ExactSample result;
        if (sampling_untruncated(cfg)) {
            result = sampling_untruncated_exact(column, domain, cfg, length + col + 1,
                purpose, scratch.untruncated, draft, draft, row_drafts, col);
        } else {
            const auto* keys = sorted ? reinterpret_cast<const unsigned long long*>(
                reinterpret_cast<const char*>(sorted) + row * sorted_row_stride) + int64_t(col) * domain :
                sampling_sort_local(column, domain, cfg, scratch.filtered, row_drafts, col);
            const float x = draft >= 0 && draft < domain ? sampling_adjusted_logit(
                logit_value(column[draft]), draft, cfg, row_drafts, col) : -CUDART_INF_F;
            result = sampling_sorted(keys, domain, cfg, length + col + 1, purpose, scratch.filtered,
                isfinite(x) ? sampling_sort_key(x, draft) : 0, draft);
        }
        if (threadIdx.x == 0) {
            const bool keep = col < extent && sampling_uniform(cfg.seed, length + col + 1,
                kSamplePurposeSpeculativeAccept, 0) < result.candidate_probability;
            tokens[row * width + col] = keep ? draft : result.token;
            stop = !keep;
            produced = col + 1;
        }
        __syncthreads();
        if (stop) { break; }
    }
    if (threadIdx.x == 0) {
        counts[row] = produced;
        accepted[row] = produced - 1;
        anchors[row] = tokens[row * width + produced - 1];
        lengths[row] = length + produced;
        for (int i = produced; i < width; ++i) { tokens[row * width + i] = 0; }
        if (cfg.token_counts) {
            for (int i = 0; i < produced; ++i) {
                const int token = tokens[row * width + i];
                if (token >= 0 && token < domain) { atomicAdd(cfg.token_counts + token, 1); }
            }
        }
    }
}

// ---- Split route for untruncated stochastic rows ---------------------------------------------------------------
// speculative_accept_wide_kernel draws an untruncated row's verify columns with ONE CTA, column after column
// (sampling_untruncated_exact: two passes over the vocabulary, the second with a double-precision exp per token).
// Measured on Inkling (201k vocabulary, M6 benchmark rows): ~1.1 ms per verified column, ~8 ms of a k = 7 round at
// temperature 1, while MTP-off decoding pays nothing for temperature 1 (its draw is split over chunks since S1).
// These three kernels compute the same draws with sampling_untruncated_exact's per-token expressions unchanged, cut
// into kSamplerFullVocabChunkItems-token chunks, every verified column in parallel:
//   1. speculative_exact_max_kernel: per (row, column, chunk) the maximum adjusted logit;
//   2. speculative_exact_partial_kernel: the column maximum from the chunk maxima (exact), then per chunk the double
//      sum of exp((x - max) / T) over the tokens at or above the min_p floor and the best Gumbel key (the draft
//      excluded on a draft column; correction / bonus purposes and positions as before);
//   3. speculative_exact_finalize_kernel: per row, each column's normalizer = its chunk sums in a fixed order, the
//      draft's probability, the sequential accept / correction / bonus walk and the commit of the wide kernel.
// The maximum and the Gumbel winner do not depend on the partition (exact max; strict total order on the keys), so
// the corrections and bonuses drawn are the old route's; the normalizer is the same double sum in another order
// (relative difference ~1e-16), so the accept test u < p(draft) can differ from the old route only when u lies within
// that rounding of p. Deterministic: fixed reduction orders, no atomics, RNG keyed by (seed, position, purpose).
// Partials: [3][width][chunks] u64 per row (max bits, sum bits, key), carved after the batch's sampling layouts for a
// target that opts in (speculative_split_partial_row_bytes; split_routes) -- no other target launches these kernels.
__device__ inline unsigned long long* speculative_exact_slot(unsigned long long* partials, std::size_t row_stride,
                                                             int row, int slot, int width, int col, int chunks) {
    auto* base = reinterpret_cast<unsigned long long*>(reinterpret_cast<unsigned char*>(partials) +
                                                       static_cast<std::size_t>(row) * row_stride);
    return base + (static_cast<std::int64_t>(slot) * width + col) * chunks;
}

// The column maximum from its chunk maxima (every thread of the block returns it).
__device__ inline float speculative_exact_column_max(const unsigned long long* maxima, int chunks,
                                                     ExactSamplingScratch& scratch) {
    float m = -CUDART_INF_F;
    for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
        m = fmaxf(m, __uint_as_float(static_cast<unsigned int>(maxima[c])));
    }
    return exact_max(m, scratch);
}

template <typename LogitT>
__launch_bounds__(kSamplerBlock) __global__ void speculative_exact_max_kernel(
    const LogitT* logits, const std::int32_t* drafts, const std::int32_t* extents,
    const SamplingConfig* configs, int domain, int physical_rows, int k, int batch,
    unsigned long long* partials, std::size_t row_stride, int chunks) {
    __shared__ SamplingRowAdmission rows;
    __shared__ ExactSamplingScratch scratch;
    const int width = k + 1;
    // the exact route's rows, and the coupled rows that need a min_p floor
    const auto take = [&](int row) {
        const SamplingConfig& c = configs[row];
        return sampling_untruncated(c) && (c.speculative_coupled == 0 || c.min_p > 0.0f);
    };
    sampling_for_each_unit(rows, batch, width, chunks, extents, take, [&](int row, int col, int chunk) {
        const SamplingConfig cfg    = configs[row];
        const std::int32_t* overlay = drafts + row * k;
        const LogitT* column = logits + (static_cast<std::int64_t>(row) * width + col) * physical_rows;
        const int begin      = chunk * kSamplerFullVocabChunkItems;
        const int end        = min(domain, begin + kSamplerFullVocabChunkItems);
        float m              = -CUDART_INF_F;
        for (int v = begin + static_cast<int>(threadIdx.x); v < end; v += blockDim.x) {
            m = fmaxf(m, sampling_adjusted_logit(logit_value(column[v]), v, cfg, overlay, col));
        }
        m = exact_max(m, scratch);
        if (threadIdx.x == 0) {
            speculative_exact_slot(partials, row_stride, row, 0, width, col, chunks)[chunk] = __float_as_uint(m);
        }
    });
}

template <typename LogitT>
__launch_bounds__(kSamplerBlock) __global__ void speculative_exact_partial_kernel(
    const LogitT* logits, const std::int32_t* drafts, const std::int32_t* extents,
    const std::int32_t* lengths, const SamplingConfig* configs, int domain, int physical_rows, int k,
    int batch, unsigned long long* partials, std::size_t row_stride, int chunks) {
    __shared__ SamplingRowAdmission rows;
    __shared__ ExactSamplingScratch scratch;
    const int width = k + 1;
    const auto take = [&](int row) {
        return sampling_untruncated(configs[row]) && configs[row].speculative_coupled == 0;
    };
    sampling_for_each_unit(rows, batch, width, chunks, extents, take, [&](int row, int col, int chunk) {
        const SamplingConfig cfg    = configs[row];
        const int extent            = min(k, max(0, extents[row]));
        const std::int32_t* overlay = drafts + row * k;
        const LogitT* column = logits + (static_cast<std::int64_t>(row) * width + col) * physical_rows;
        const float top =
            speculative_exact_column_max(speculative_exact_slot(partials, row_stride, row, 0, width, col, chunks),
                                         chunks, scratch);
        const float floor  = cfg.min_p > 0 ? top + cfg.temperature * logf(cfg.min_p) : -CUDART_INF_F;
        const int excluded = col < extent ? overlay[col] : -1;
        const int purpose  = col < extent ? kSamplePurposeSpeculativeCorrection : kSamplePurposeSpeculativeBonus;
        const int position = lengths[row] + col + 1;
        const int begin    = chunk * kSamplerFullVocabChunkItems;
        const int end      = min(domain, begin + kSamplerFullVocabChunkItems);
        double total            = 0;
        unsigned long long best = 0;
        for (int v = begin + static_cast<int>(threadIdx.x); v < end; v += blockDim.x) {
            const float x = sampling_adjusted_logit(logit_value(column[v]), v, cfg, overlay, col);
            if (!isfinite(x) || x < floor) { continue; }
            total += exp((double(x) - top) / cfg.temperature);
            if (v != excluded) {
                const auto key = sampling_sort_key(
                    (x - top) / cfg.temperature + sampling_gumbel(cfg.seed, position, purpose, v), v);
                if (key > best) { best = key; }
            }
        }
        total = exact_sum(total, scratch);
        best  = sampling_block_max_key(best, scratch.keys);
        if (threadIdx.x == 0) {
            speculative_exact_slot(partials, row_stride, row, 1, width, col, chunks)[chunk] =
                static_cast<unsigned long long>(__double_as_longlong(total));
            speculative_exact_slot(partials, row_stride, row, 2, width, col, chunks)[chunk] = best;
        }
        __syncthreads(); // scratch.keys is reused by the next unit
    });
}

template <typename LogitT>
__launch_bounds__(kSamplerBlock) __global__ void speculative_exact_finalize_kernel(
    const LogitT* logits, const std::int32_t* drafts, const std::int32_t* extents, std::int32_t* lengths,
    std::int32_t* anchors, std::int32_t* tokens, std::int32_t* counts, std::int32_t* accepted,
    const SamplingConfig* configs, int domain, int physical_rows, int k, unsigned long long* partials,
    std::size_t row_stride, int chunks) {
    const int row = static_cast<int>(blockIdx.x), width = k + 1;
    const SamplingConfig cfg = configs[row];
    if (!sampling_untruncated(cfg) || cfg.speculative_coupled != 0) { return; }
    const int extent                = min(k, max(0, extents[row]));
    const int length                = lengths[row];
    const std::int32_t* row_drafts  = drafts + row * k;
    __shared__ ExactSamplingScratch scratch;
    __shared__ int stop;
    __shared__ int produced;
    for (int col = 0; col <= extent; ++col) {
        const unsigned long long* sums = speculative_exact_slot(partials, row_stride, row, 1, width, col, chunks);
        const unsigned long long* keys = speculative_exact_slot(partials, row_stride, row, 2, width, col, chunks);
        const float top =
            speculative_exact_column_max(speculative_exact_slot(partials, row_stride, row, 0, width, col, chunks),
                                         chunks, scratch);
        double s               = 0;
        unsigned long long b   = 0;
        for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
            s += __longlong_as_double(static_cast<long long>(sums[c]));
            if (keys[c] > b) { b = keys[c]; }
        }
        const double total            = exact_sum(s, scratch);
        const unsigned long long best = sampling_block_max_key(b, scratch.keys);
        if (threadIdx.x == 0) {
            const int draft   = col < extent ? row_drafts[col] : -1;
            float probability = 0;
            if (draft >= 0 && draft < domain) {
                const LogitT* column = logits + (static_cast<std::int64_t>(row) * width + col) * physical_rows;
                const float floor    = cfg.min_p > 0 ? top + cfg.temperature * logf(cfg.min_p) : -CUDART_INF_F;
                const float x = sampling_adjusted_logit(logit_value(column[draft]), draft, cfg, row_drafts, col);
                if (isfinite(x) && x >= floor && total > 0) {
                    probability = static_cast<float>(exp((double(x) - top) / cfg.temperature) / total);
                }
            }
            const bool keep = col < extent && sampling_uniform(cfg.seed, length + col + 1,
                                                               kSamplePurposeSpeculativeAccept, 0) < probability;
            tokens[row * width + col] = keep ? draft : sampling_key_index(best);
            stop     = !keep;
            produced = col + 1;
        }
        __syncthreads();
        if (stop) { break; }
    }
    if (threadIdx.x == 0) {
        counts[row]   = produced;
        accepted[row] = produced - 1;
        anchors[row]  = tokens[row * width + produced - 1];
        lengths[row]  = length + produced;
        for (int i = produced; i < width; ++i) { tokens[row * width + i] = 0; }
        if (cfg.token_counts) {
            for (int i = 0; i < produced; ++i) {
                const int token = tokens[row * width + i];
                if (token >= 0 && token < domain) { atomicAdd(cfg.token_counts + token, 1); }
            }
        }
    }
}

// ---- Coupled route : untruncated rows with cfg.speculative_coupled -------------------
// Each verified column's token is drawn EXACTLY as ordinary decoding draws the token that follows the column's fed
// token: the Gumbel-max over sampling_adjusted_logit(x) * (1 / T), above the min_p floor, keyed by (seed, the fed
// token's cache position = lengths[row] + col, kSamplePurposeDecode) -- sampling_full_vocab_kernel's per-token
// expression verbatim, in chunks as S1 (sampling_full_vocab_partial_kernel), the winner under the same strict order
// (sampling_better). A draft is accepted iff it equals its column's token (the greedy rule). The committed tokens are
// thus the ones MTP-off decoding samples with the same seed: exact by construction, token-identical with MTP on and
// off at temperature > 0, and independent of the drafts, k, the batch and timing. P(accept d) = p(d), the one-hot
// rule's rate. Columns use the round's overlay (drafts 0 .. col-1) exactly as the other routes. Partials: slot 0 the
// chunk max of the adjusted logits (speculative_exact_max_kernel; min_p rows), slot 1 the chunk's best z (float bits),
// slot 2 its token.
__device__ inline bool speculative_coupled_row(const SamplingConfig& cfg) {
    return sampling_untruncated(cfg) && cfg.speculative_coupled != 0;
}

template <typename LogitT>
__launch_bounds__(kSamplerBlock) __global__ void speculative_coupled_partial_kernel(
    const LogitT* logits, const std::int32_t* drafts, const std::int32_t* extents,
    const std::int32_t* lengths, const SamplingConfig* configs, int domain, int physical_rows, int k,
    int batch, unsigned long long* partials, std::size_t row_stride, int chunks, float no_floor) {
    __shared__ SamplingRowAdmission rows;
    __shared__ ExactSamplingScratch scratch;
    __shared__ float red_val[kSamplerBlock];
    __shared__ int red_idx[kSamplerBlock];
    const int width = k + 1;
    const int tid   = static_cast<int>(threadIdx.x);
    const auto take = [&](int row) { return speculative_coupled_row(configs[row]); };
    sampling_for_each_unit(rows, batch, width, chunks, extents, take, [&](int row, int col, int chunk) {
        const SamplingConfig cfg    = configs[row];
        const std::int32_t* overlay = drafts + row * k;
        const LogitT* column = logits + (static_cast<std::int64_t>(row) * width + col) * physical_rows;
        const float inv_temp = 1.0f / cfg.temperature;
        const SampleKey key  = sample_key(cfg, lengths[row] + col, kSamplePurposeDecode);
        const int position   = key.position;
        float floor_value    = no_floor;
        if (cfg.min_p > 0.0f) {
            // max over v of RN(x * inv_temp) == RN((max over v of x) * inv_temp): rounding is monotone (inv_temp > 0).
            // The explicit RN product is the rounded row maximum sampling_full_vocab_kernel adds __logf(min_p) to; a
            // plain product here could contract into one fma with that sum (this file builds with fmad) and move the
            // floor by 1 ulp (reviews R5 minor 2, R7 minor 1).
            const float top = speculative_exact_column_max(
                speculative_exact_slot(partials, row_stride, row, 0, width, col, chunks), chunks, scratch);
            const float scaled_top = __fmul_rn(top, inv_temp);
            floor_value            = scaled_top + __logf(cfg.min_p);
        }
        const int begin = chunk * kSamplerFullVocabChunkItems;
        const int end   = min(domain, begin + kSamplerFullVocabChunkItems);
        float best = -CUDART_INF_F;
        int best_v = INT_MAX;
        for (int v = begin + tid; v < end; v += kSamplerBlock) {
            const float x = sampling_adjusted_logit(logit_value(column[v]), v, cfg, overlay, col) * inv_temp;
            if (x < floor_value) { continue; }
            const float z = x + sampling_gumbel(cfg.seed, position, key.purpose, static_cast<unsigned>(v));
            if (sampling_better(z, v, best, best_v)) {
                best   = z;
                best_v = v;
            }
        }
        red_val[tid] = best;
        red_idx[tid] = best_v;
        __syncthreads();
        for (int s = kSamplerBlock / 2; s > 0; s >>= 1) {
            if (tid < s && sampling_better(red_val[tid + s], red_idx[tid + s], red_val[tid], red_idx[tid])) {
                red_val[tid] = red_val[tid + s];
                red_idx[tid] = red_idx[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            speculative_exact_slot(partials, row_stride, row, 1, width, col, chunks)[chunk] =
                __float_as_uint(red_val[0]);
            speculative_exact_slot(partials, row_stride, row, 2, width, col, chunks)[chunk] =
                static_cast<unsigned int>(red_idx[0]);
        }
        __syncthreads(); // red_val / red_idx are reused by the next unit
    });
}

template <typename LogitT>
__launch_bounds__(kSamplerBlock) __global__ void speculative_coupled_finalize_kernel(
    const std::int32_t* drafts, const std::int32_t* extents, std::int32_t* lengths, std::int32_t* anchors,
    std::int32_t* tokens, std::int32_t* counts, std::int32_t* accepted, const SamplingConfig* configs, int domain,
    int k, unsigned long long* partials, std::size_t row_stride, int chunks) {
    const int row = static_cast<int>(blockIdx.x), width = k + 1;
    const int tid = static_cast<int>(threadIdx.x);
    const SamplingConfig cfg = configs[row];
    if (!speculative_coupled_row(cfg)) { return; }
    const int extent               = min(k, max(0, extents[row]));
    const int length               = lengths[row];
    const std::int32_t* row_drafts = drafts + row * k;
    __shared__ float red_val[kSamplerBlock];
    __shared__ int red_idx[kSamplerBlock];
    __shared__ int stop;
    __shared__ int produced;
    for (int col = 0; col <= extent; ++col) {
        const unsigned long long* zs  = speculative_exact_slot(partials, row_stride, row, 1, width, col, chunks);
        const unsigned long long* ids = speculative_exact_slot(partials, row_stride, row, 2, width, col, chunks);
        float best = -CUDART_INF_F;
        int best_v = INT_MAX;
        for (int c = tid; c < chunks; c += kSamplerBlock) {
            const float z = __uint_as_float(static_cast<unsigned int>(zs[c]));
            const int v   = static_cast<int>(static_cast<unsigned int>(ids[c]));
            if (sampling_better(z, v, best, best_v)) {
                best   = z;
                best_v = v;
            }
        }
        red_val[tid] = best;
        red_idx[tid] = best_v;
        __syncthreads();
        for (int s = kSamplerBlock / 2; s > 0; s >>= 1) {
            if (tid < s && sampling_better(red_val[tid + s], red_idx[tid + s], red_val[tid], red_idx[tid])) {
                red_val[tid] = red_val[tid + s];
                red_idx[tid] = red_idx[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            // as sampling_full_vocab_kernel: no candidate at all (a row of non-finite logits) picks token 0
            const int t     = red_idx[0] == INT_MAX ? 0 : red_idx[0];
            const bool keep = col < extent && row_drafts[col] == t;
            tokens[row * width + col] = t;
            stop     = !keep;
            produced = col + 1;
        }
        __syncthreads();
        if (stop) { break; }
    }
    if (tid == 0) {
        counts[row]   = produced;
        accepted[row] = produced - 1;
        anchors[row]  = tokens[row * width + produced - 1];
        lengths[row]  = length + produced;
        for (int i = produced; i < width; ++i) { tokens[row * width + i] = 0; }
        if (cfg.token_counts) {
            for (int i = 0; i < produced; ++i) {
                const int token = tokens[row * width + i];
                if (token >= 0 && token < domain) { atomicAdd(cfg.token_counts + token, 1); }
            }
        }
    }
}

template <typename LogitT>
__launch_bounds__(kSamplerBlock) __global__ void speculative_accept_greedy_drafts_kernel(
    const std::int32_t* target_tokens, const LogitT* logits, const std::int32_t* drafts,
    const std::int32_t* current_extents, std::int32_t* lengths, std::int32_t* anchors,
    std::int32_t* licensed_tokens, std::int32_t* licensed_counts, std::int32_t* accepted,
    const SamplingConfig* configs, std::int32_t token_domain, std::int32_t physical_rows,
    std::int32_t k) {
    const int tid                   = threadIdx.x;
    const int row                   = static_cast<int>(blockIdx.x);
    const int cols                  = k + 1;
    int extent                      = current_extents[row];
    extent                          = extent < 0 ? 0 : (extent > k ? k : extent);
    const SamplingConfig cfg        = configs[row];
    if (sampling_wide(cfg)) { return; }
    const std::int32_t* row_targets = target_tokens + row * cols;
    const std::int32_t* row_drafts  = drafts + row * k;
    std::int32_t* row_tokens        = licensed_tokens + row * cols;
    const LogitT* row_logits =
        logits + static_cast<std::int64_t>(row) * cols * physical_rows;

    if (!(cfg.temperature > 0.0f)) {
        if (tid == 0) {
            int a = 0;
            while (a < extent && row_targets[a] == row_drafts[a]) { ++a; }
            const int t_star = row_targets[a];

            for (int i = 0; i <= k; ++i) { row_tokens[i] = 0; }
            for (int i = 0; i < a; ++i) { row_tokens[i] = row_drafts[i]; }
            row_tokens[a] = t_star;

            const int produced   = a + 1;
            licensed_counts[row] = produced;
            accepted[row]        = a;
            anchors[row]         = t_star;
            lengths[row] += produced;
        }
        return;
    }

    __shared__ float red_val[kSamplerBlock];
    __shared__ int red_idx[kSamplerBlock];
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ float merge_val[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int merge_idx[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int n_support;
    __shared__ int a_sh;
    __shared__ int done_sh;
    __shared__ int tstar_sh;
    __shared__ int L_sh;

    const int partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const int group_count    = sampler_group_count(partial_blocks);
    // No-op when the scratch/group path owns this shape.
    if (sampler_multiblock_ok(token_domain, cols, partial_blocks, group_count)) { return; }

    if (tid == 0) {
        a_sh     = 0;
        done_sh  = 0;
        tstar_sh = 0;
        L_sh     = lengths[row];
    }
    __syncthreads();

    for (int i = 0; i <= extent; ++i) {
        // Column i is only reached when drafts[0..i-1] were all accepted, so the
        // round-local penalty overlay for this column is exactly those i drafts.
        const std::int64_t base = static_cast<std::int64_t>(i) * physical_rows;
        if (token_domain <= kSamplerTileItems) {
            sampling_build_truncated_small(row_logits, base, token_domain, cfg, red_val, red_idx,
                                           cand_val, cand_idx, prob, &n_support, row_drafts, i);
        } else {
            sampling_build_truncated_block_fast(row_logits, base, token_domain, cfg, merge_val,
                                                merge_idx, cand_val, cand_idx, prob, &n_support,
                                                row_drafts, i);
        }
        if (tid == 0 && done_sh == 0) {
            const int L = L_sh;
            if (i < extent) {
                const int d = row_drafts[i];
                float pd    = 0.0f;
                for (int j = 0; j < n_support; ++j) {
                    if (cand_idx[j] == d) {
                        pd = prob[j];
                        break;
                    }
                }
                const float u =
                    sampling_uniform(cfg.seed, L + i + 1, kSamplePurposeSpeculativeAccept, 0u);
                if (u < pd) {
                    a_sh = i + 1; // accept drafts[i], keep verifying
                } else {
                    const float ur = sampling_uniform(cfg.seed, L + i + 1,
                                                      kSamplePurposeSpeculativeCorrection, 0u);
                    tstar_sh       = sampling_pick_from_support(cand_idx, prob, n_support, d, ur);
                    done_sh        = 1;
                }
            } else {
                // Every draft accepted: bonus token from the last verify column.
                const float u =
                    sampling_uniform(cfg.seed, L + extent + 1, kSamplePurposeSpeculativeBonus, 0u);
                tstar_sh = sampling_pick_from_support(cand_idx, prob, n_support, -1, u);
                done_sh  = 1;
            }
        }
        __syncthreads();
        if (done_sh) { break; }
    }

    if (tid == 0) {
        const int a     = a_sh;
        const int tstar = tstar_sh;
        const int L     = L_sh;

        for (int i = 0; i <= k; ++i) { row_tokens[i] = 0; }
        for (int i = 0; i < a; ++i) { row_tokens[i] = row_drafts[i]; }
        row_tokens[a] = tstar;

        const int produced   = a + 1;
        licensed_counts[row] = produced;
        accepted[row]        = a;
        anchors[row]         = tstar;
        lengths[row]         = L + produced;
        if (cfg.token_counts != nullptr) {
            for (int i = 0; i < produced; ++i) { atomicAdd(&cfg.token_counts[row_tokens[i]], 1); }
        }
    }
}

// Candidate tiles of every verification column a bounded-candidate row licenses, walked
// by a resident grid over [batch][cols][partial] (see sampling_for_each_unit).
template <typename LogitT>
__launch_bounds__(kSamplerBlock) __global__ void speculative_sampling_partial_topk_kernel(
    const LogitT* logits, const std::int32_t* drafts, const std::int32_t* current_extents,
    const SamplingConfig* configs, std::int32_t token_domain, std::int32_t physical_rows,
    std::int32_t cols, std::int32_t k, std::int32_t batch, SamplingWorkspace workspace,
    std::size_t workspace_row_stride) {
    __shared__ typename SamplingPartialSort::TempStorage sort_storage;
    __shared__ SamplingRowAdmission rows;
    const auto candidate_route = [&](int row) {
        const SamplingConfig& cfg = configs[row];
        return cfg.temperature > 0.0f && !sampling_wide(cfg) && token_domain > kSamplerTileItems;
    };
    // Columns past a row's extent (clamped to [0, k]) are not verified; col <= k always.
    sampling_for_each_unit(rows, batch, cols, div_up(token_domain, kSamplerPartialTileItems),
                           current_extents, candidate_route, [&](int row, int col, int partial) {
        const SamplingConfig cfg             = configs[row];
        const SamplingWorkspace row_workspace = speculative_workspace_row(workspace, workspace_row_stride, row);
        if (partial == 0 && threadIdx.x == 0) {
            row_workspace.group_done[col] = 0;
            if (col == 0) { *row_workspace.speculative_finalize_count = 0; }
        }

        unsigned long long keys[kSamplerItemsPerThread];

        const int cap                  = sampling_candidate_cap(cfg, token_domain);
        const std::int64_t base        = (static_cast<std::int64_t>(row) * cols + col) * physical_rows;
        const std::int32_t* row_drafts = drafts + row * k;
        const int tile_start           = partial * kSamplerPartialTileItems;
        // Column col's penalty overlay is the first `col` drafts (see accept loop);
        // applying it before top-k selection lets it change the candidate set, not
        // just the post-truncation probabilities.
#pragma unroll
        for (int item = 0; item < kSamplerItemsPerThread; ++item) {
            const int v = tile_start + item * blockDim.x + threadIdx.x;
            if (v < token_domain) {
                const float x = sampling_adjusted_logit(logit_value(logits[base + v]), v, cfg,
                                                        row_drafts, col);
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
                const int off                   = sampling_partial_offset(row_workspace, col, partial, rank);
                row_workspace.partial_keys[off] = keys[item];
            }
        }
        __syncthreads(); // sort_storage is reused by the next tile
    });
}

__launch_bounds__(kSamplerGroupBlock) __global__ void speculative_sampling_group_finalize_kernel(
    const std::int32_t* target_tokens, const std::int32_t* drafts,
    const std::int32_t* current_extents, std::int32_t* lengths, std::int32_t* anchors,
    std::int32_t* licensed_tokens, std::int32_t* licensed_counts, std::int32_t* accepted,
    const SamplingConfig* configs, std::int32_t token_domain, std::int32_t cols,
    std::int32_t partial_blocks, std::int32_t group_count, SamplingWorkspace workspace,
    std::size_t workspace_row_stride) {
    const int row   = static_cast<int>(blockIdx.z);
    const int group = static_cast<int>(blockIdx.x);
    const int col   = static_cast<int>(blockIdx.y);
    const int tid   = threadIdx.x;
    const int k     = cols - 1;
    int extent      = current_extents[row];
    extent          = extent < 0 ? 0 : (extent > k ? k : extent);
    if (col > extent) { return; }
    const SamplingConfig cfg        = configs[row];
    if (sampling_wide(cfg)) { return; }
    const std::int32_t* row_targets = target_tokens + row * cols;
    const std::int32_t* row_drafts  = drafts + row * k;
    std::int32_t* row_tokens        = licensed_tokens + row * cols;
    if (token_domain <= kSamplerTileItems) { return; }

    if (!(cfg.temperature > 0.0f)) {
        if (tid == 0 && col == 0 && group == 0) {
            int a = 0;
            while (a < extent && row_targets[a] == row_drafts[a]) { ++a; }
            const int t_star = row_targets[a];
            for (int i = 0; i <= k; ++i) { row_tokens[i] = 0; }
            for (int i = 0; i < a; ++i) { row_tokens[i] = row_drafts[i]; }
            row_tokens[a]        = t_star;
            const int produced   = a + 1;
            licensed_counts[row] = produced;
            accepted[row]        = a;
            anchors[row]         = t_star;
            lengths[row] += produced;
        }
        return;
    }

    workspace = speculative_workspace_row(workspace, workspace_row_stride, row);

    __shared__ typename SamplingGroupSort::TempStorage sort_storage;
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ int is_last_group;
    unsigned long long keys[kSamplerGroupItemsPerThread];

    const int cap = sampling_candidate_cap(cfg, token_domain);
    // The preceding partial launch initializes all caller-owned counters. CUDA
    // stream ordering makes those writes visible before this launch begins.

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
        is_last_group  = (done == group_count) ? 1 : 0;
    }
    __syncthreads();
    if (!is_last_group) { return; }

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
        workspace.dist_support[col] = n_support;
        for (int j = 0; j < n_support; ++j) {
            const int off            = sampling_dist_offset(col, j);
            workspace.dist_idx[off]  = cand_idx[j];
            workspace.dist_prob[off] = prob[j];
        }
        workspace.group_done[col] = 0;
        __threadfence();
        const int done_cols = atomicAdd(workspace.speculative_finalize_count, 1) + 1;
        if (done_cols == extent + 1) {
            const int L = lengths[row];
            int a       = 0;
            int tstar   = 0;
            for (int i = 0; i <= extent; ++i) {
                const int n            = workspace.dist_support[i];
                const int* dist_idx    = workspace.dist_idx + sampling_dist_offset(i, 0);
                const float* dist_prob = workspace.dist_prob + sampling_dist_offset(i, 0);
                if (i < extent) {
                    const int d = row_drafts[i];
                    float pd    = 0.0f;
                    for (int j = 0; j < n; ++j) {
                        if (dist_idx[j] == d) {
                            pd = dist_prob[j];
                            break;
                        }
                    }
                    const float u =
                        sampling_uniform(cfg.seed, L + i + 1, kSamplePurposeSpeculativeAccept, 0u);
                    if (u < pd) {
                        a = i + 1;
                        continue;
                    }
                    const float ur = sampling_uniform(cfg.seed, L + i + 1,
                                                      kSamplePurposeSpeculativeCorrection, 0u);
                    tstar          = sampling_pick_from_support(dist_idx, dist_prob, n, d, ur);
                    break;
                }
                const float u =
                    sampling_uniform(cfg.seed, L + extent + 1, kSamplePurposeSpeculativeBonus, 0u);
                tstar = sampling_pick_from_support(dist_idx, dist_prob, n, -1, u);
            }
            for (int i = 0; i <= k; ++i) { row_tokens[i] = 0; }
            for (int i = 0; i < a; ++i) { row_tokens[i] = row_drafts[i]; }
            row_tokens[a]        = tstar;
            const int produced   = a + 1;
            licensed_counts[row] = produced;
            accepted[row]        = a;
            anchors[row]         = tstar;
            lengths[row]         = L + produced;
            if (cfg.token_counts != nullptr) {
                for (int i = 0; i < produced; ++i) {
                    atomicAdd(&cfg.token_counts[row_tokens[i]], 1);
                }
            }
            *workspace.speculative_finalize_count = 0;
        }
    }
}

__global__ void speculative_select_accepted_hidden_kernel(const __nv_bfloat16* hidden,
                                                          const std::int32_t* selectors,
                                                          __nv_bfloat16* out, std::int32_t rows,
                                                          std::int32_t cols) {
    const int batch = static_cast<int>(blockIdx.y);
    const int row   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) { return; }
    const int col = selectors[batch];
    if (col < 0 || col >= cols) { return; }
    out[static_cast<std::int64_t>(batch) * rows + row] =
        hidden[(static_cast<std::int64_t>(batch) * cols + col) * rows + row];
}

__global__ void proposal_remap_token_ids_kernel(std::int32_t* proposal_tokens,
                                                std::int32_t proposal_count,
                                                const std::int32_t* id_map, std::int32_t n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= proposal_count) { return; }
    const int idx = proposal_tokens[i];
    if (idx >= 0 && idx < n) { proposal_tokens[i] = id_map[idx]; }
}

} // namespace sinfer::ops
