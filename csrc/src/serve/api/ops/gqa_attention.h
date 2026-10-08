#pragma once

#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sinfer::ops {

inline constexpr std::uint32_t kGqaAttentionMaximumVisibleKeys = 1048576;

/// QSA sparse selection (design/INFERENCE.md, phase 4): one bit per block of `block` cache cells
/// for every query column, `stride` words apart. A key whose block bit is clear scores -inf; a
/// null `words` is the dense path and every kernel compiles to exactly what it was.
struct GqaBlockMask {
    const std::uint32_t* words = nullptr;
    std::int32_t stride        = 0;
    std::int32_t block         = 0;
    // One complete image block, in absolute token positions. Text remains causal.
    // A query inside this block can read every image key; its left window is unchanged.
    std::int32_t image_begin = 0;
    std::int32_t image_end = 0;
    __host__ __device__ std::int32_t last_key(std::int32_t query) const {
        return query >= image_begin && query < image_end ? image_end - 1 : query;
    }
    __host__ __device__ std::int32_t tile_last_key(std::int32_t first, std::int32_t last) const {
        return first < image_end && last >= image_begin && last < image_end ? image_end - 1 : last;
    }
};

struct GqaExecutionEnvelope {
    std::uint32_t min_visible_keys = 0;
    std::uint32_t max_visible_keys = 0;

    /// Causal sliding window, zero for a layer that sees its whole context. A
    /// query at absolute position i admits keys j with `i - j < sliding_window`
    /// -- exactly `sliding_window` keys including its own, which is
    /// FlashAttention's `window_size = (sliding_window - 1, 0)` and what vLLM
    /// passes for Gemma 3. A route that cannot honour it must refuse rather than
    /// return unwindowed attention, which is a wrong answer that looks right.
    std::int32_t sliding_window = 0;
};

/**
 * What the attention layers of one round share. Every layer of a round attends from the same
 * positions, valid columns and table rows, and Hopper's decode and verify route (FlashAttention-3)
 * derives its per-row arrays and its scheduler's metadata from them. Passed a round, the round's
 * first layer computes those into `storage` and its later layers over the same inputs read them,
 * as vLLM builds FA3's scheduler metadata once per step, instead of every layer running two small
 * kernels before its attention. Every other route ignores it.
 *
 * `storage` is device I32 [storage_ints(rows)] for rounds of up to `rows` sequences, at an
 * address that outlives the rounds, so a captured round replays with it. The caller calls
 * begin() before a round's first layer and passes the round to each of its layers; a layer's
 * positions, valid columns and table rows must hold the same values for every layer of the round
 * that passes the same tensors. Rounds on one storage must not overlap on the device: they run on
 * one stream. The rest is host bookkeeping (gqa_attention.cpp). SUROGATE_SERVE_GQA_ROUND_METADATA=0
 * makes every layer compute its own.
 */
class GqaRoundMetadata {
public:
    /// The input sets one round may hold, and the layers each may serve (one tile counter each).
    static constexpr std::int32_t kEntries  = 4;
    static constexpr std::int32_t kLaunches = 256;

    struct Entry {
        const void* positions     = nullptr;
        const void* valid_columns = nullptr;
        const void* kv_table_rows = nullptr;
        std::int32_t width = 0, batch = 0, head_dim = 0, q_heads = 0, kv_heads = 0;
        std::int32_t sliding_window = 0, splits = 0;
        /// Layers launched on it this round; the first computed it.
        std::int32_t launches = 0;
    };

    /// Per input set: the rows' segment arrays (4 rows + 1 ints, padded to four), FA3's scheduler
    /// metadata (four vectors of rows padded to four) and kLaunches tile counters.
    [[nodiscard]] static constexpr std::int32_t entry_ints(std::int32_t rows) noexcept {
        return (4 * rows + 4) / 4 * 4 + (rows + 3) / 4 * 4 * 4 + kLaunches;
    }
    [[nodiscard]] static constexpr std::int32_t storage_ints(std::int32_t rows) noexcept {
        return kEntries * entry_ints(rows);
    }

    GqaRoundMetadata() = default;
    GqaRoundMetadata(Tensor storage, std::int32_t rows) noexcept : storage_(storage), rows_(rows) {}

    void begin() noexcept { used_ = 0; }

    [[nodiscard]] const Tensor& storage() const noexcept { return storage_; }
    [[nodiscard]] std::int32_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::int32_t used() const noexcept { return used_; }
    [[nodiscard]] Entry& entry(std::int32_t index) noexcept { return entries_[index]; }
    /// Starts entry `used()`, which the caller has checked is below kEntries.
    Entry& add(const Entry& entry) noexcept { return entries_[used_++] = entry; }

private:
    Tensor storage_;
    std::int32_t rows_ = 0;
    std::int32_t used_ = 0;
    Entry entries_[kEntries]{};
};

/**
 * Shared numerical contract for A1/A2/A3.
 *
 * Public q/k/v inputs and BF16 cache values are interpreted after their BF16 storage boundary.
 * INT8-G64 cache rows use one FP16 scale for each contiguous 64-element group. For BF16 source
 * values x, their exact observable encoding is:
 *
 *   a          = max_i abs(FP32(x[i]))
 *   scale_bits = FP16_RNE(a / 127)
 *   s          = FP32(scale_bits)
 *   inv        = s == 0 ? 0 : FP32(1 / s)
 *   code[i]    = s == 0 ? 0 : I8(clamp(RNE_even(FP32(x[i]) * inv), -127, 127))
 *   decode[i]  = FP32(code[i]) * s
 *
 * A1 and A2 produce identical code and scale bits. The common ideal attention oracle uses BF16 Q
 * and logical cache values (BF16 values for a BF16 cache, FP32 decode above for INT8-G64), then
 * evaluates score dot products, stable softmax, and value reduction in FP64. The BF16 Op output is
 * promoted to FP64 for comparison with that result.
 *
 * The registered INT8 implementation defines Q8-G64, paired with INT8-G64 K, as its native query
 * compute profile. Its profile-defined query quantization and any narrower staging do not replace
 * BF16 Q in the ideal oracle. BF16-cache and INT8-cache compute profiles therefore have separate
 * named numerical criteria owned by the GQA conformance test. Those envelopes apply to the
 * registered geometries, tested token extents, conformance matrix, and target-representative
 * activation range; they are not a universal error bound for arbitrary adversarial BF16 tensors.
 * A1 and A3 are each qualified directly against the ideal oracle. A1-versus-A3 parity is only an
 * additional consistency check.
 */

/**
 * Returns the transient arena capacity required for every W in the inclusive interval at one
 * exact logical batch size. Head geometry, cache dtype, and execution envelope are the fixed
 * implementation profile. Invalid profiles or intervals throw; a legal B=1 prompt route may
 * return zero.
 */
[[nodiscard]] std::size_t
gqa_attention_workspace_capacity_bytes(std::int32_t head_dim, std::int32_t q_heads,
                                       std::int32_t kv_heads,
                                       DType cache_dtype, GqaExecutionEnvelope envelope,
                                       std::int32_t batch_size, std::int32_t min_width,
                                       std::int32_t max_width);

/**
 * A1: append K/V for B independent sequences and compute causal grouped-query attention. Let
 * Vb=W when valid_columns is empty and Vb=valid_columns[b] otherwise. For row b, query head h,
 * kvh=floor(h/group), 0<=j<Vb, p=positions[j,b], and that row's populated cache history [0,p]:
 *
 *   score[x]      = scale * dot(q[:,h,j,b], K_cache[b][:,x,kvh]), 0 <= x <= p
 *   probability   = softmax_x(score)
 *   ideal[:,h,j,b] = sum_x probability[x] * V_cache[b][:,x,kvh].
 *
 * The registered geometries are `[256,24|4,W,B]` group 6 and `[256,16|2,W,B]` group 8.
 * q/k/v/out are contiguous BF16 in request-major order, positions is contiguous I32 [W,B], and
 * kv_table_rows is contiguous I32 [B]. valid_columns is either contiguous I32 [B], or an empty
 * Tensor meaning every row has exactly W valid columns. This dense/masked choice is part of the
 * call topology; it is not inferred by copying device metadata to the host. B=1 accepts every
 * positive W in the current prefill/decode domain; B=2..8 accepts W=1..16. Cache storage is BF16
 * or INT8-G64 under the shared numerical contract above. PagedKVBatchLayerView supplies shared
 * planes and the complete block-table matrix; kv_table_rows[b] selects one row for sequence b.
 *
 * In masked form, every row's valid columns are the prefix [0,valid_columns[b]); positions in that
 * prefix are sequential and address populated causal histories. Each nonempty row repeats its
 * final valid position through the invalid tail; an empty row uses zero positions. Other
 * invalid-tail inputs contain safe dummy values. A1 does not modify cache for invalid columns and
 * writes exact BF16 zero to their output. The caller guarantees that the maximum final valid
 * position plus one over nonempty rows lies in the declared execution envelope. The envelope is a
 * host launch-resource promise over that batch maximum; it does not alter any row's causal mask.
 *
 * q/k/v/positions/valid_columns/kv_table_rows/out, every cache plane/table, and live workspace
 * suballocations are pairwise non-overlapping. The Op overwrites every addressed cache row but
 * owns no persistent frontier, allocation, request identity, or commit authority.
 */
void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   const Tensor& valid_columns, const Tensor& kv_table_rows, float scale,
                   PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream,
                   GqaBlockMask selection = {}, GqaRoundMetadata* round = nullptr);

/**
 * A2: perform only the cache-write part of A1. k/v are contiguous BF16 `[256,4|2,T]`, positions is
 * contiguous sequential I32 [T], and every addressed code and INT8 scale is overwritten. It reads
 * no unrelated cache row, receives no execution envelope, and owns no persistent frontier.
 */
void gqa_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                   PagedKVLayerView cache, cudaStream_t stream);

/**
 * A3: compute causal attention from an already populated cache without accepting new K/V or
 * mutating any cache plane. q/out are contiguous BF16 `[256,24|16,T]`, positions is contiguous
 * sequential I32 [T], and the mathematical formula and execution-envelope contract are identical
 * to A1. Caller workspace is reported by gqa_attention_workspace_capacity_bytes().
 */
void gqa_attention_cached(const Tensor& q, const Tensor& positions, float scale,
                          const PagedKVLayerView& cache, GqaExecutionEnvelope envelope,
                          WorkspaceArena& workspace, Tensor& out, cudaStream_t stream,
                          GqaBlockMask selection = {});

/**
 * A3 across a batch of sequences: A1's shapes and execution envelope, over a cache this call
 * does not write.
 *
 * This is what a layer that shares an earlier layer's keys and values runs. Gemma 4's E-series
 * ends in a run of them -- twenty of E2B's thirty-five -- and they hold a query projection and
 * nothing else, so there is no key or value to append and the planes they read belong to
 * another layer. Every route A1 can take is available here; the only difference is the append.
 */
void gqa_attention_cached(const Tensor& q, const Tensor& positions, const Tensor& valid_columns,
                          const Tensor& kv_table_rows, float scale, PagedKVBatchLayerView cache,
                          GqaExecutionEnvelope envelope, WorkspaceArena& workspace, Tensor& out,
                          cudaStream_t stream, GqaBlockMask selection = {},
                          GqaRoundMetadata* round = nullptr);

/// One sequence's part of gqa_attention_packed_prompts: its query columns
/// [column, column + width) of q, k, v, positions and out, the block-table row its cache is
/// on, and the envelope its own call would declare.
struct GqaPackedSegment {
    std::int32_t column    = 0;
    std::int32_t width     = 0;
    std::int32_t table_row = 0;
    GqaExecutionEnvelope envelope{};
};

/**
 * Whether gqa_attention_packed_prompts serves a segment of `width` queries under `envelope`
 * exactly as gqa_attention (or gqa_attention_cached) does alone: the prompt-tile route over a
 * BF16 or FP8 cache, without a QSA selection or an image block (callers check those).
 */
bool gqa_attention_packs_prompt(std::int32_t head_dim, std::int32_t q_heads,
                                std::int32_t kv_heads, DType cache_dtype, std::int32_t width,
                                GqaExecutionEnvelope envelope);

/**
 * The prompt attention of several sequences at once (#14's packed prefill rounds). Each segment
 * is cut into the query tiles its own call would use, and tiles of one width from every segment
 * share a launch -- as many as the workspace's free room holds, up to 64 MiB of partials --
 * instead of each segment's few-CTA launches running one after another. Each tile reads only its
 * own sequence's keys over the same absolute key partitions, and the reducer sums its own
 * partitions in order, so every output bit is what the segment's own call writes. On Hopper the
 * segments FlashAttention-3 takes alone (wide ones over a BF16 cache) share one varlen FA3
 * launch instead, which computes each segment's tiles exactly as its one-segment launch does.
 *
 * q and out are `[D, Hq, N]`, positions `[N]`, and k/v `[D, Hkv, N]` over the same N columns.
 * Every segment must pass gqa_attention_packs_prompt, with its positions sequential and its last
 * position below its envelope's max_visible_keys, as its own call requires; all share one sliding
 * window. With k and v, their columns are appended to the cache first (A1); with empty k and v the
 * cache already holds them (A3). `max_lanes_per_launch` (0: no limit) caps the tiles in one launch.
 * Not for a stream that is being captured: the tile metadata is copied from the host.
 */
void gqa_attention_packed_prompts(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& positions,
                                  const std::vector<GqaPackedSegment>& segments, float scale,
                                  PagedKVBatchLayerView cache, WorkspaceArena& workspace,
                                  Tensor& out, cudaStream_t stream,
                                  std::int32_t max_lanes_per_launch = 0);

/**
 * Prompt attention over an e4m3 cache takes its query-key product in FP8 (PATCHES.md #119). Where
 * the tensor-core prompt kernel serves a prompt (every GPU but Hopper, at head dim 128 or 256),
 * each query row is rounded to e4m3 against its own absolute maximum and multiplies the cache's K
 * codes on the e4m3 tensor cores at twice the bf16 rate; the probabilities still meet V in bf16.
 * Hopper's FlashAttention-3 FP8 kernel already rounds its queries (and probabilities) to e4m3.
 * On by default; SUROGATE_SERVE_PROMPT_ATTENTION_FP8_QK=0 starts the process with the bf16
 * product. Process-wide; a relaxed atomic, read at each launch.
 */
void set_prompt_attention_fp8_query(bool enabled) noexcept;
[[nodiscard]] bool prompt_attention_fp8_query() noexcept;

} // namespace sinfer::ops
