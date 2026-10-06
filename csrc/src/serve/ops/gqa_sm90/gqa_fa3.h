#pragma once
// Attention on Hopper through FlashAttention-3's sm90 forward kernel (wgmma, warp specialised,
// intra-warpgroup softmax overlap), the kernel vLLM runs on H100/H200 for prompts and decode alike.
// Vendored at src/third_party/flash_attn3 (see the NOTICE there).
//
// It reads queries straight from the projection output and keys/values straight from the paged
// cache, which the caller has already appended this round's keys to: the cache's page is
// 64 slots of one KV head (`((page * kv_heads + h) * 64 + slot) * head_dim + d`), which FA3's
// non-TMA paged path takes as plain strides. One launch covers any number of segments ("varlen"):
// segment s is columns `q_offsets[s] .. + q_lengths[s]` of the query tensor, its keys are the
// first `kv_lengths[s]` positions of block-table row `kv_rows[s]`, and its queries are the last
// `q_lengths[s]` of those positions (causal, bottom-right aligned, as in a prefill). All four
// arrays live in device memory, so a launch is graph-capture safe.
//
// Prompts run one segment per sequence and no split. Decode and verify rows over a BF16 cache run
// one segment per row, and with `max_splits` above one a segment's keys may be spread over
// several CTAs: FA3's prepare kernel picks each segment's split count on the device from the
// lengths it reads, so a captured graph splits a long history and leaves short ones whole, and
// the combine merges the partials (row_splits picks the bound as FA3's own API does).
//
// Built for one shape family: head dim 64, 128 or 256, any query group, causal or a causal sliding
// window, no softcap, over either cache dtype (gqa_fa3_launch.h; one instantiation each). Over a BF16 cache it runs in BF16. Over an e4m3 cache it runs FA3's FP8 kernel the
// way vLLM does for `--kv-cache-dtype fp8`: the queries are cast to e4m3 at scale 1 (saturating,
// like the cache's own keys), both products run on FP8 wgmma, and the output is BF16. The cache
// carries no scales, so every descale is 1. Everything here is sm_90a-specific; builds without
// 90a link a stub where `available()` is false and every other entry point throws.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail::gqa_fa3 {

struct PagedLaunch {
    std::int32_t head_dim = 0;          ///< 64, 128 or 256 (supports)
    const void* q = nullptr;            ///< BF16 [total_q][q_heads][head_dim]
    void* out     = nullptr;            ///< BF16, the same shape as q
    /// [physical_pages][kv_heads][64][head_dim], BF16, or e4m3 with `fp8_cache`.
    const void* k_pages = nullptr;
    const void* v_pages = nullptr;
    /// The pages hold e4m3 codes: the queries are quantized into the workspace and the FP8
    /// kernel runs (workspace_bytes with `fp8_cache` sizes the codes).
    bool fp8_cache = false;
    const std::int32_t* block_tables = nullptr;  ///< I32 [table_rows][logical_pages]
    std::int32_t logical_pages  = 0;
    std::int32_t table_rows     = 0;
    std::int32_t physical_pages = 0;
    std::int32_t q_heads  = 0;
    std::int32_t kv_heads = 0;
    std::int32_t segments = 0;
    std::int32_t total_q  = 0;          ///< columns of q and out
    std::int32_t max_q    = 0;          ///< an upper bound on every q_lengths[s]
    /// Device I32 [segments + 1]: first column of each segment (the last entry is unread).
    const std::int32_t* q_offsets  = nullptr;
    const std::int32_t* q_lengths  = nullptr;  ///< device I32 [segments]
    const std::int32_t* kv_lengths = nullptr;  ///< device I32 [segments]
    const std::int32_t* kv_rows    = nullptr;  ///< device I32 [segments]
    float scale = 0.0f;
    /// A causal sliding window in keys, 0 for none: a query at absolute position i sees keys j
    /// with `i - j < sliding_window` (FA3's local mask with `window_size_left = sliding_window - 1`).
    std::int32_t sliding_window = 0;
    /// The most CTAs one segment's keys may be spread over (row_splits), 1 for none. The scratch
    /// must have been sized for it (workspace_bytes with `splits`).
    std::int32_t max_splits = 1;
};

/// True when this build carries the sm_90a kernel, the current device is an sm_90 part and
/// SUROGATE_SERVE_GQA_FA3 is not "0".
[[nodiscard]] bool available() noexcept;

/// Whether a geometry has a kernel here.
[[nodiscard]] bool supports(std::int32_t head_dim, std::int32_t q_heads,
                            std::int32_t kv_heads) noexcept;

/// The narrowest segment, in query columns, the prompt route sends here
/// (SUROGATE_SERVE_GQA_FA3_MIN_COLUMNS, default 32). Narrower segments stay on the split-KV
/// tile kernels, which spread a long history over every SM where FA3 would run one CTA per
/// 128 packed query rows.
[[nodiscard]] std::int32_t min_columns() noexcept;

/// Whether decode and verify rows take FA3 as well (SUROGATE_SERVE_GQA_FA3_ROWS, on unless "0").
[[nodiscard]] bool rows_enabled() noexcept;

/// The most CTAs a segment's keys are split over, and the bytes every split's FP32 partials may
/// take together (row_splits caps the split count by both).
inline constexpr std::int32_t kMaxSplits = 64;
inline constexpr std::size_t kPartialBudget = std::size_t{32} << 20;

/// One split's FP32 partial outputs and log-sum-exps over `total_q` columns.
[[nodiscard]] inline std::size_t split_bytes(std::int32_t head_dim, std::int32_t q_heads,
                                             std::int32_t total_q) noexcept {
    return static_cast<std::size_t>(q_heads) * total_q * (head_dim + 1) * sizeof(float);
}

/// Scratch `run` needs: the per-row log-sum-exp FA3 always writes, its scheduler's metadata
/// (four per-segment vectors and a tile counter), over an e4m3 cache the queries' e4m3 codes and,
/// when the launch may split (`partials`), room for the partials of as many splits as row_splits
/// can give it: kMaxSplits of them or kPartialBudget, whichever is less, plus the alignment of
/// the two planes. That reservation grows with `total_q` whatever the split count a round gets,
/// so a plan sized for a batch holds every smaller one. Each piece is 256-byte aligned. Plain
/// arithmetic, so workspace planning can call it in any build.
[[nodiscard]] inline std::size_t workspace_bytes(std::int32_t head_dim, std::int32_t q_heads,
                                                 std::int32_t total_q, std::int32_t segments,
                                                 bool fp8_cache = false,
                                                 bool partials = false) noexcept {
    const auto align = [](std::size_t bytes) { return (bytes + 255) / 256 * 256; };
    const std::size_t rounded = (static_cast<std::size_t>(segments) + 3) / 4 * 4;
    const std::size_t rows    = static_cast<std::size_t>(q_heads) * total_q;
    const std::size_t reserve =
        std::min(static_cast<std::size_t>(kMaxSplits) * split_bytes(head_dim, q_heads, total_q),
                 kPartialBudget) + 512;
    return align(rows * sizeof(float)) + align((rounded * 4 + 1) * sizeof(std::int32_t)) +
           (fp8_cache ? align(rows * static_cast<std::size_t>(head_dim)) : 0) +
           (partials ? align(reserve) : 0);
}

/// How a launch of decode or verify rows splits: `splits` is its bound (PagedLaunch::max_splits),
/// FA3's own heuristic (flash_api.cpp's get_num_splits for a varlen launch, which counts one
/// segment because the prepare kernel splits each one by what the round actually holds) capped
/// at kMaxSplits and by kPartialBudget; `partials` is whether the heuristic splits at all, which
/// the scratch reserves for (workspace_bytes) even where the cap leaves one split. Over an e4m3
/// cache neither: the FP8 kernel never splits. It asks the current device for its SM count, so
/// planning, like the launch, runs with the serving device current.
struct RowSplits {
    std::int32_t splits = 1;
    bool partials       = false;
};
[[nodiscard]] RowSplits row_splits(std::int32_t head_dim, std::int32_t q_heads,
                                   std::int32_t kv_heads, std::int32_t total_q,
                                   std::int32_t max_q, std::int32_t max_keys, bool fp8_cache,
                                   std::int32_t sliding_window) noexcept;

/// Device I32 scratch the segment arrays take, in elements: offsets (segments + 1), lengths,
/// kv lengths and rows (segments each).
[[nodiscard]] inline std::int32_t metadata_ints(std::int32_t segments) noexcept {
    return 4 * segments + 1;
}

/// A one-segment launch's arrays, filled on the stream from the round's device positions: the
/// segment is all `tokens` columns and sees `positions[0] + tokens` keys. `metadata` holds
/// metadata_ints(1) ints laid out [offsets(2) | lengths(1) | kv_lengths(1) | rows(1)], and the
/// row is copied from `kv_row` (device; null means row 0), so a captured graph replays with a
/// new one.
void prompt_metadata(const std::int32_t* positions, std::int32_t tokens,
                     const std::int32_t* kv_row, std::int32_t* metadata, cudaStream_t stream);

/// A packed round's kv lengths, on the stream: `kv_lengths[s] = positions[q_offsets[s]] +
/// q_lengths[s]` (the segment's queries are the last of its keys).
void segment_kv_lengths(const std::int32_t* positions, const std::int32_t* q_offsets,
                        const std::int32_t* q_lengths, std::int32_t segments,
                        std::int32_t* kv_lengths, cudaStream_t stream);

/// A batch of rows' arrays, filled on the stream: row r is segment r, its columns
/// `r * width .. + valid`, where `valid` is `valid_columns[r]` (device; null means every column),
/// and it sees `positions[r * width] + valid` keys of block-table row `rows[r]` (device; null
/// means row r). A row with no valid column is an empty segment. `metadata` holds
/// metadata_ints(batch) ints.
void rows_metadata(const std::int32_t* positions, std::int32_t width, std::int32_t batch,
                   const std::int32_t* valid_columns, const std::int32_t* rows,
                   std::int32_t* metadata, cudaStream_t stream);

void run(const PagedLaunch& args, void* workspace, std::size_t workspace_capacity,
         cudaStream_t stream);

} // namespace sinfer::ops::detail::gqa_fa3
