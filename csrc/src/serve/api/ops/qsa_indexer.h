#pragma once

// QSA sparse indexer (design/INFERENCE.md, phase 4). On the full-attention layers of
// Qwen3.8-Flash-Next a small "lightning indexer" scores blocks of `block` cached cells and the
// attention of each query is restricted to the highest-scoring blocks, so a context far longer
// than the attention budget costs a budget-sized attention. llama.cpp's `build_qsa_top_k`
// (study/llama.cpp-master/src/models/qwen4exp.cpp:469) is the oracle.
//
// Each physical page stores every raw token key followed by one pooled key per complete
// block. Pooling never overwrites raw keys: speculative rollback may rewrite a suffix that
// starts inside a previously completed block, which must then be folded again.
//
// A block is complete as soon as its last position is cached, so `qsa_indexer_append` folds it
// in the same launch that writes the raw keys.

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace sinfer::ops {

// Per page: 64 raw keys, 16 pooled keys, and 16 aligned int32[4] block positions (162 of the
// width). The latter retain the first member's three mRoPE axes across chunk boundaries. The
// rest is padding, and it is what keeps an elastic pool's granule small: the region maps runs
// of pages in which every plane spans whole 2 MiB quanta, and a 162-wide page (20,736 bytes,
// 2^8 x 81) needed 8,192 pages per run. Rounded up to a whole run, the capacity it committed
// overshot the planned KV by up to 8 GiB, and Flash-Next refused a 131,072-token context and
// MTP at 32 sequences on a DGX Spark. At 192 (24,576 bytes, 2^13 x 3) a run is 256 pages, for
// 4.4% more KV bytes per token.
inline constexpr std::int32_t kQsaIndexerStorageHeadDim = 192;

struct QsaIndexerGeometry {
    std::int32_t head_dim   = 0; // indexer key/query width
    std::int32_t heads      = 0; // query heads
    std::int32_t block      = 0; // cells per block (the compress ratio)
    std::int32_t top_k      = 0; // cell budget
    std::int32_t rotary_dim = 0;
    float rope_theta        = 0.0F;
    float rms_eps           = 1.0e-6F;
    std::int32_t mrope_height = 0;
    std::int32_t mrope_width = 0;
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
///   rope_positions optional I32 [T] or axis-strided [T,3], defaults to cache positions;
///                  the first member's coordinates are retained with each pooled block.
void qsa_indexer_append(const Tensor& keys, const Tensor& positions, const Tensor& table_rows,
                        std::int32_t columns_per_row, const Tensor& key_norm,
                        const QsaIndexerGeometry& geometry, PagedKVBatchLayerView cache,
                        cudaStream_t stream, const Tensor& rope_positions = {});

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

/// Query columns per list of `qsa_tile_union`: the prompt attention kernel's tile.
inline constexpr std::int32_t kQsaTileRows = 64;

/// Ints in one list of `qsa_tile_union` over a `keys`-long history: room for every block.
[[nodiscard]] std::int32_t qsa_tile_union_stride(std::int32_t keys, std::int32_t block);

/// Bytes of `qsa_tile_union`'s lists and counts for `rows` mask rows, as the arena lays them out.
[[nodiscard]] std::size_t qsa_tile_union_bytes(std::int32_t rows, std::int32_t keys,
                                               std::int32_t block);

/// For each tile of `kQsaTileRows` consecutive rows of `qsa_indexer_select`'s mask, the blocks
/// any of its rows selected, in ascending order. Attention over one sequence's prompt chunk
/// then reads only those blocks' keys.
///   mask   I32 [words, rows]
///   blocks I32 [qsa_tile_union_stride(), tiles]   tiles = ceil(rows / kQsaTileRows)
///   counts I32 [tiles]                            the length of each list
void qsa_tile_union(const Tensor& mask, Tensor& blocks, Tensor& counts, cudaStream_t stream);

/// Transient bytes `qsa_indexer_select` needs for `rows` queries over a `keys`-long history.
[[nodiscard]] std::size_t qsa_indexer_select_workspace_capacity_bytes(
    std::int32_t rows, std::int32_t keys, const QsaIndexerGeometry& geometry);

} // namespace sinfer::ops
