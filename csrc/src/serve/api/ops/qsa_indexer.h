#pragma once

// QSA sparse indexer (design/INFERENCE.md, phase 4). On the full-attention layers of
// Qwen3.8-Flash-Next a small "lightning indexer" scores blocks of `block` cached cells and the
// attention of each query is restricted to the highest-scoring blocks, so a context far longer
// than the attention budget costs a budget-sized attention. llama.cpp's `build_qsa_top_k`
// (study/llama.cpp-master/src/models/qwen4exp.cpp:469) is the oracle.
//
// The engine keeps the indexer keys in the text KV pool, one BF16 plane of `head_dim` per
// full-attention layer, sharing pages and block tables with K/V. Two states live in that plane:
//
//   * cell p of an incomplete block holds the token's RAW indexer key (no norm, no rotation);
//   * once a block is complete, the cell of its FIRST position holds the block's key — the mean
//     of its `block` raw keys, RMS-normalised and roped at the block's first position. The other
//     cells of a complete block are dead.
//
// A block is complete as soon as its last position is cached, so `qsa_indexer_append` folds it
// in the same launch that writes the raw keys.

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

struct QsaIndexerGeometry {
    std::int32_t head_dim   = 0; // indexer key/query width
    std::int32_t heads      = 0; // query heads
    std::int32_t block      = 0; // cells per block (the compress ratio)
    std::int32_t top_k      = 0; // cell budget
    std::int32_t rotary_dim = 0;
    float rope_theta        = 0.0F;
    float rms_eps           = 1.0e-6F;
};

/// Words of a per-row block bitmask covering `keys` cells: one bit per block, LSB first.
[[nodiscard]] std::int32_t qsa_block_mask_words(std::int32_t keys, std::int32_t block);

/// True when every cell of a `keys`-long history is visible to a query at `position`, i.e. the
/// complete blocks fit in the budget and the selection would be the identity. The caller skips
/// the indexer entirely in that case; it is the reason contexts below the budget are exact.
[[nodiscard]] bool qsa_selection_is_dense(std::int32_t keys, const QsaIndexerGeometry& geometry);

/// Writes the raw indexer keys of `T` new columns into their cells and folds every block those
/// columns complete.
///   keys      BF16 [head_dim, T]   raw indexer keys, in column order
///   positions I32  [T]             absolute cache position of each column, ascending
///   key_norm  BF16 [head_dim]      RMSNorm gain of the block key
///   cache     the layer's view; `indexer_pages` and `block_table` are read and written
///   table_rows I32 [S]  block-table row per sequence; column c belongs to sequence
///                       `c / columns_per_row` (one sequence for the whole call when
///                       `columns_per_row` is the column count, one column each when it is 1)
void qsa_indexer_append(const Tensor& keys, const Tensor& positions, const Tensor& table_rows,
                        std::int32_t columns_per_row, const Tensor& key_norm,
                        const QsaIndexerGeometry& geometry, PagedKVBatchLayerView cache,
                        cudaStream_t stream);

/// Selects the visible blocks of every query row and writes its bitmask.
///   q          BF16 [head_dim, heads, rows]  normalised and roped indexer queries
///   positions  I32  [rows]                   absolute position of each query
///   table_rows I32  [rows]                   block-table row of each query's sequence
///   mask       I32  [words, rows]            output bitmask words, `qsa_block_mask_words()` long
/// Bit b of row r is set when block b is visible to that query. The blocks of the incomplete
/// tail are always visible; the complete blocks past the budget are cut on a block boundary,
/// which selects at most the reference's cells and never more (the reference tops its whole
/// blocks up with `block - 1` cells of the next one).
void qsa_indexer_select(const Tensor& q, const Tensor& positions, const Tensor& table_rows,
                        std::int32_t columns_per_row, const QsaIndexerGeometry& geometry,
                        PagedKVBatchLayerView cache, std::int32_t keys, WorkspaceArena& workspace,
                        Tensor& mask, cudaStream_t stream);

/// Transient bytes `qsa_indexer_select` needs for `rows` queries over a `keys`-long history.
[[nodiscard]] std::size_t qsa_indexer_select_workspace_capacity_bytes(
    std::int32_t rows, std::int32_t keys, const QsaIndexerGeometry& geometry);

} // namespace ninfer::ops
