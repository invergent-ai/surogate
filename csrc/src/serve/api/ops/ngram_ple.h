#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr std::int32_t kNgramPleMaxHeads = 16;
inline constexpr std::int32_t kNgramPleMaxNgram = 3;

/**
 * Hashed n-gram row selection. Head h of an n-gram of order n (2 or 3) selects table row
 *
 *   mixed = ctx[0]*m[0] ^ ctx[1]*m[1] (^ ctx[2]*m[2])      (uint64, wrapping)
 *   row   = mixed % head_vocab_sizes[h] + head_offsets[h]
 *
 * where ctx[0] is the token itself and ctx[k] its k-th predecessor. A predecessor that is
 * missing, precedes the sequence start, or is the EOS token cuts the context: it and every
 * older predecessor read as EOS. The token's own EOS does not cut its context.
 */
struct NgramPleHash {
    std::uint64_t multipliers[kNgramPleMaxNgram] = {};
    std::int32_t head_offsets[kNgramPleMaxHeads]     = {};
    std::int32_t head_vocab_sizes[kNgramPleMaxHeads] = {};
    std::int32_t ngram     = 0; ///< 2 or 3
    std::int32_t heads     = 0; ///< (ngram - 1) * heads_per_ngram
    std::int32_t eos_token = 0;
};

/**
 * The n-gram embedding table: IQ4_NL rows (blocks of 32 values in 18 bytes: FP16 scale and 16
 * nibble bytes; value = scale * kvalues[nibble]) at any device-accessible address, typically
 * pinned host memory mapped into the device address space.
 */
struct NgramPleTable {
    const void* rows       = nullptr;
    std::int64_t row_count = 0;
    std::int32_t row_bytes = 0; ///< head_dim / 32 * 18
    std::int32_t head_dim  = 0; ///< values per row, a multiple of 32
};

/**
 *   key          BF16_CTRL [streams*hidden, heads*head_dim]
 *   value        BF16_CTRL [hidden, heads*head_dim]
 *   norm_key     FP32 [streams*hidden] folded gamma
 *   norm_query   FP32 [streams*hidden]
 *   norm_conv    FP32 [streams*hidden]
 *   convolution  BF16 [streams*hidden, kernel] (tap k of channel c at [c, k]); tap k reads
 *                (kernel-1-k)*dilation positions back
 */
struct NgramPleWeights {
    Weight key;
    Weight value;
    Tensor norm_key;
    Tensor norm_query;
    Tensor norm_conv;
    Tensor convolution;
};

/**
 * Per-column metadata for one call: `ids` I32 [T] token ids; `segment_begin` I32 [T] the
 * column index where the column's sequence segment starts in this call; `slots` I32 [T] the
 * column's persistent state slot; `segment_last` I32 [T] non-zero on the last column of its
 * segment (that column writes the slot's new history and convolution state).
 */
struct NgramPleColumns {
    Tensor ids;
    Tensor segment_begin;
    Tensor slots;
    Tensor segment_last;
};

/**
 * Persistent per-slot state, slot slowest so one slot is contiguous: `history` I32
 * [ngram-1, slots] (oldest predecessor first, EOS where none) and `conv_state` BF16
 * [(kernel-1)*dilation, streams*hidden, slots] (the last normalised columns seen, oldest
 * first).
 */
struct NgramPleState {
    Tensor history;
    Tensor conv_state;
};

/**
 * Returns the transient capacity `ngram_ple_forward` needs for every T in [min_tokens,
 * max_tokens].
 */
[[nodiscard]] std::size_t ngram_ple_workspace_capacity_bytes(std::int32_t streams,
                                                             std::int32_t hidden,
                                                             std::int32_t embed_dim,
                                                             std::int32_t heads,
                                                             std::int32_t min_tokens,
                                                             std::int32_t max_tokens);

/**
 * residual[:,t] += ple(residual[:,t], tokens) for every column, with
 *
 *   emb   = concat_h table[row(h,t)]                       [heads*head_dim]
 *   key   = group_norm(key_w · emb, norm_key)              per stream
 *   query = group_norm(residual, norm_query)               per stream
 *   gate  = signed_sqrt(<key_s, query_s> / sqrt(hidden))   per stream, |.| clamped at 1e-6
 *   gv    = sigmoid(gate_s) * (value_w · emb)              per stream
 *   out   = gv + silu(dilated_depthwise_conv(group_norm(gv, norm_conv)))
 *
 * `residual` is contiguous BF16 [streams*hidden, T]. The projections run through cuBLASLt.
 * The per-slot state is read for the first columns of each segment and rewritten by the
 * segment's last column; every slot appears in at most one segment per call.
 */
/// flags[clamp(base + count - 1)] = 1 where `count_scalar` is a device I32 scalar (a bucket's
/// valid column count); flags is I32 [T].
void ngram_ple_mark_segment_last(Tensor& flags, const Tensor& count_scalar, std::int32_t base,
                                 cudaStream_t stream);

void ngram_ple_forward(Tensor& residual, const NgramPleColumns& columns, const NgramPleHash& hash,
                       const NgramPleTable& table, const NgramPleWeights& weights,
                       NgramPleState& state, std::int32_t streams, std::int32_t conv_kernel,
                       std::int32_t conv_dilation, float eps, WorkspaceArena& workspace,
                       cudaStream_t stream);

} // namespace ninfer::ops
