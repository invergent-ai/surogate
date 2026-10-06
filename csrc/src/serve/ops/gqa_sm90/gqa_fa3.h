#pragma once
// Prompt attention on Hopper through FlashAttention-3's sm90 forward kernel (wgmma, warp
// specialised, intra-warpgroup softmax overlap), the kernel vLLM runs for prefill on H100/H200.
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
// Built for one shape family: head dim 64, 128 or 256, any query group, causal or a causal sliding
// window, no softcap, over either cache dtype (gqa_fa3_launch.h; one instantiation each). Over a BF16 cache it runs in BF16. Over an e4m3 cache it runs FA3's FP8 kernel the
// way vLLM does for `--kv-cache-dtype fp8`: the queries are cast to e4m3 at scale 1 (saturating,
// like the cache's own keys), both products run on FP8 wgmma, and the output is BF16. The cache
// carries no scales, so every descale is 1. Everything here is sm_90a-specific; builds without
// 90a link a stub where `available()` is false and every other entry point throws.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail::gqa_fa3 {

struct PagedPrefill {
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

/// Scratch `run` needs: the per-row log-sum-exp FA3 always writes, its scheduler's metadata
/// (four per-segment vectors and a tile counter) and, over an e4m3 cache, the queries' e4m3
/// codes, each 256-byte aligned. Plain arithmetic, so workspace planning can call it in any build.
[[nodiscard]] inline std::size_t workspace_bytes(std::int32_t head_dim, std::int32_t q_heads,
                                                 std::int32_t total_q, std::int32_t segments,
                                                 bool fp8_cache = false) noexcept {
    const auto align = [](std::size_t bytes) { return (bytes + 255) / 256 * 256; };
    const std::size_t rounded = (static_cast<std::size_t>(segments) + 3) / 4 * 4;
    const std::size_t rows    = static_cast<std::size_t>(q_heads) * total_q;
    return align(rows * sizeof(float)) + align((rounded * 4 + 1) * sizeof(std::int32_t)) +
           (fp8_cache ? align(rows * static_cast<std::size_t>(head_dim)) : 0);
}

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

void run(const PagedPrefill& args, void* workspace, std::size_t workspace_capacity,
         cudaStream_t stream);

} // namespace sinfer::ops::detail::gqa_fa3
