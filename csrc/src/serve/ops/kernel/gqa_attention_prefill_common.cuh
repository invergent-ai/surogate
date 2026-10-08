#pragma once

// Shared Qwen3.6 GQA dimensions and leaf PTX helpers used by the independently tuned
// BF16 and INT8 prompt kernels. This file deliberately owns no staging policy,
// shared-memory arena, warp schedule, or kernel body.

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/gqa_attention_geometry.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace sinfer::ops {

inline constexpr int kGqaPrefillBr      = 64;
// Use the decode kernel's 32-key tiles for ordinary heads. A 512-wide
// absorbed head uses 16 keys to stay within the shared-memory limit.
template <int HeadDim>
inline constexpr int kGqaPrefillBcFor = HeadDim > 256 ? 16 : 32;
inline constexpr int kGqaPrefillBc       = kGqaPrefillBcFor<256>;
inline constexpr int kGqaPrefillThreads  = 128;

// Dynamic shared-memory arena of the BF16 prompt kernel: one Q tile plus the K
// and V tiles it streams over, all bf16. The head dimension is a template
// parameter rather than a file constant, so a launcher asks for the arena of the
// geometry it is about to launch: 64 KiB at head dim 256, 32 KiB at 128.
template <int HeadDim>
inline constexpr int kGqaPrefillSmemBytes =
    (kGqaPrefillBr + 2 * kGqaPrefillBcFor<HeadDim>) * HeadDim *
    static_cast<int>(sizeof(__nv_bfloat16));

// Both arenas stay within the device opt-in shared-memory limit.
static_assert(kGqaPrefillSmemBytes<256> == 65536);
// The 512-wide arena may not exceed it either: 101,376 bytes is what the card opts in to.
static_assert(kGqaPrefillSmemBytes<512> == 98304);

// An e4m3 cache can't be staged with cp.async straight into the bf16 tiles: its codes have to
// be widened on the way. Where the arena has room, the kernel lands each key block's raw codes,
// K and V, with cp.async into a second arena behind the tiles (two slots each, a block ahead)
// and widens them in shared memory between its MMAs, so the cache reads overlap the tensor-core
// work as the bf16 cache's do. A 512-wide head has no room and widens straight from global
// memory.
template <int HeadDim>
inline constexpr int kGqaPrefillFp8RawBytes = 4 * kGqaPrefillBcFor<HeadDim> * HeadDim;
template <int HeadDim>
inline constexpr bool kGqaPrefillFp8Raw =
    kGqaPrefillSmemBytes<HeadDim> + kGqaPrefillFp8RawBytes<HeadDim> <= 101376;
template <int HeadDim>
inline constexpr int kGqaPrefillFp8SmemBytes =
    kGqaPrefillSmemBytes<HeadDim> + (kGqaPrefillFp8Raw<HeadDim> ? kGqaPrefillFp8RawBytes<HeadDim> : 0);
static_assert(kGqaPrefillFp8SmemBytes<256> == 98304);
static_assert(!kGqaPrefillFp8Raw<512>);

// FP8 query-key product (ops::prompt_attention_fp8_query, on by default): Q is quantized to
// e4m3 against each row's absolute maximum and multiplies the cache's K codes as they land, on
// the e4m3 tensor cores at twice the bf16 rate; only V is widened. The arena holds the e4m3 Q
// tile and its row scales, one raw block each of K and V codes, and the widened V tile: under
// half of an SM's 100 KiB at head dim 256, so two CTAs share an SM. The 16-byte swizzle needs
// eight chunks a row.
template <int HeadDim>
inline constexpr bool kGqaPrefillFp8QkRegistered = HeadDim == 128 || HeadDim == 256;
template <int HeadDim>
inline constexpr int kGqaPrefillFp8QkSmemBytes =
    kGqaPrefillBr * HeadDim + kGqaPrefillBr * static_cast<int>(sizeof(float)) +
    2 * kGqaPrefillBcFor<HeadDim> * HeadDim +
    kGqaPrefillBcFor<HeadDim> * HeadDim * static_cast<int>(sizeof(__nv_bfloat16));
static_assert(kGqaPrefillFp8QkSmemBytes<256> == 49408);

struct GqaPrefillDirectMetadata {
    const std::int32_t* table;
    /// Causal sliding window, zero for unbounded. A query at absolute position i
    /// admits keys j with `i - j < sliding_window`.
    std::int32_t window = 0;

    __device__ __forceinline__ std::int32_t valid_tokens(std::int32_t width) const { return width; }

    __device__ __forceinline__ const std::int32_t* block_table() const { return table; }
};

template <bool Masked>
struct GqaPrefillBatchMetadata {
    const std::int32_t* tables;
    const std::int32_t* valid_columns;
    const std::int32_t* table_rows;
    std::int32_t table_stride;
    /// Causal sliding window, zero for unbounded. See GqaPrefillDirectMetadata.
    std::int32_t window = 0;

    __device__ __forceinline__ std::int32_t valid_tokens(std::int32_t width) const {
        if constexpr (Masked) {
            const std::int32_t valid = valid_columns[0];
            return valid <= 0 ? 0 : (valid < width ? valid : width);
        }
        return width;
    }

    // No row tensor selects row 0, as the split-KV tiles' metadata reads it.
    __device__ __forceinline__ const std::int32_t* block_table() const {
        const std::int32_t row = table_rows == nullptr ? 0 : table_rows[0];
        return tables + static_cast<std::int64_t>(row) * table_stride;
    }
};

// `q_heads` is the query head count of the tensors: the geometry's own, or, for the BF16 kernel,
// a query group the registry does not carry over the geometry's KV heads.
template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_prefill_q_index(int q_head, int d, int token,
                                                            int q_heads = Geometry::QHeads) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(Geometry::HeadDim) *
                                              (static_cast<std::int64_t>(q_head) +
                                               static_cast<std::int64_t>(q_heads) * token);
}

template <typename Geometry>
__device__ __forceinline__ void gqa_prefill_zero_output_rows(__nv_bfloat16* out, int q_head,
                                                             int row_begin, int row_end, int tid,
                                                             int threads,
                                                             int q_heads = Geometry::QHeads) {
    if (row_begin >= row_end) { return; }
    constexpr int D    = Geometry::HeadDim;
    const int elements = (row_end - row_begin) * D;
    for (int element = tid; element < elements; element += threads) {
        const int row = row_begin + element / D;
        const int d   = element - (row - row_begin) * D;
        out[gqa_prefill_q_index<Geometry>(q_head, d, row, q_heads)] = __float2bfloat16(0.0f);
    }
}

// XOR-swizzled b16 element address. INT8 operands use the same layout by packing
// two consecutive signed bytes into each b16 lane before ldmatrix.
__device__ __forceinline__ int gqa_prefill_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ unsigned gqa_prefill_swz_addr(unsigned lane_base, unsigned ck,
                                                         unsigned as, unsigned r) {
    return lane_base + ((ck | as) ^ r);
}

} // namespace sinfer::ops
