// LoRA on the Marlin MoE route (see marlin_moe_lora.h).

#include "ops/sparse_moe/marlin/marlin_moe_lora.h"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"

#include <cuda_bf16.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

namespace sinfer::ops::detail {
namespace {

using namespace nvcuda;

constexpr int kMaxRows   = 64;  // Marlin's largest MoE block
constexpr int kKTile     = 64;  // shrink: K columns staged per step
constexpr int kNTile     = 64;  // expand: output columns per CUDA block
constexpr int kMaxRankTiles = 4; // rank <= 64 in 16-wide WMMA tiles
constexpr int kThreads   = 128; // four warps

// The row a padded block slot stands for, or -1 for padding; `rows_total` rows are real.
__device__ __forceinline__ int block_row(const std::int32_t* sorted_ids, int block, int b, int i, int rows_total) {
    if (sorted_ids == nullptr) {
        const int r = b * block + i;
        return r < rows_total ? r : -1;
    }
    const int r = sorted_ids[b * block + i];
    return r < rows_total ? r : -1;
}

__device__ __forceinline__ int row_slot(const std::int32_t* slots, int stride, int token) {
    return slots == nullptr ? -1 : slots[static_cast<std::int64_t>(token) * stride];
}

// low[row][part][0:rank] = A[expert, part, slot] . in(row): one CUDA block per (Marlin block, part, slot). A row
// whose token selected a different slot is a zero row of this pass and keeps the value its own pass writes.
__global__ __launch_bounds__(kThreads) void moe_lora_shrink_kernel(
    const __nv_bfloat16* __restrict__ input, int input_per_token, const std::int32_t* __restrict__ sorted_ids,
    const std::int32_t* __restrict__ expert_ids, int fixed_expert, const std::int32_t* __restrict__ past_padded,
    int block, int rows_total, int top_k, const LoraBank* __restrict__ banks, int part_first, int parts,
    const std::int32_t* __restrict__ slots, int slot_stride, int slot_count, int rank, int k,
    __nv_bfloat16* __restrict__ low) {
    const int b    = static_cast<int>(blockIdx.x);
    const int part = part_first + static_cast<int>(blockIdx.y) % parts;
    const int slot = static_cast<int>(blockIdx.y) / parts;
    if (slot >= slot_count) { return; }
    if (past_padded != nullptr && b * block >= *past_padded) { return; }
    if (past_padded == nullptr && b * block >= rows_total) { return; }
    const int expert = expert_ids != nullptr ? expert_ids[b] : fixed_expert;
    if (expert < 0) { return; }
    const LoraBank bank = banks[2 + 3 * expert + part];
    if (bank.a == nullptr || bank.k != k) { return; }

    __shared__ int rows_s[kMaxRows];
    __shared__ int any_s;
    if (threadIdx.x == 0) { any_s = 0; }
    __syncthreads();
    for (int i = threadIdx.x; i < block; i += kThreads) {
        int r = block_row(sorted_ids, block, b, i, rows_total);
        if (r >= 0 && row_slot(slots, slot_stride, r / top_k) != slot) { r = -1; }
        rows_s[i] = r;
        if (r >= 0) { atomicOr(&any_s, 1); }
    }
    __syncthreads();
    if (!any_s) { return; }

    __shared__ __align__(32) __nv_bfloat16 x_s[kMaxRows * kKTile];
    __shared__ __align__(32) __nv_bfloat16 a_s[kMaxRankTiles * 16 * kKTile];
    __shared__ __align__(32) float c_s[kMaxRows * kMaxRankTiles * 16];
    const int m_tiles = (block + 15) / 16;
    const int n_tiles = (rank + 15) / 16;
    const int warp    = static_cast<int>(threadIdx.x) / 32;
    // Each warp owns some (m, n) output tiles.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    int owned = 0;
    int tile_m[4], tile_n[4];
    for (int t = warp; t < m_tiles * n_tiles && owned < 4; t += kThreads / 32) {
        tile_m[owned] = t / n_tiles;
        tile_n[owned] = t % n_tiles;
        wmma::fill_fragment(acc[owned], 0.0F);
        ++owned;
    }
    const auto* a_base = static_cast<const __nv_bfloat16*>(bank.a) + static_cast<std::int64_t>(slot) * bank.a_stride;
    for (int k0 = 0; k0 < k; k0 += kKTile) {
        for (int idx = threadIdx.x; idx < m_tiles * 16 * kKTile; idx += kThreads) {
            const int i = idx / kKTile, kk = idx % kKTile;
            const int r = i < block ? rows_s[i] : -1;
            __nv_bfloat16 v = __float2bfloat16(0.0F);
            if (r >= 0 && k0 + kk < k) {
                const std::int64_t src = input_per_token ? static_cast<std::int64_t>(r / top_k) : r;
                v = input[src * k + k0 + kk];
            }
            x_s[i * kKTile + kk] = v;
        }
        for (int idx = threadIdx.x; idx < n_tiles * 16 * kKTile; idx += kThreads) {
            const int rr = idx / kKTile, kk = idx % kKTile;
            a_s[rr * kKTile + kk] = (rr < rank && k0 + kk < k) ? a_base[static_cast<std::int64_t>(rr) * k + k0 + kk]
                                                              : __float2bfloat16(0.0F);
        }
        __syncthreads();
        for (int o = 0; o < owned; ++o) {
            for (int kk = 0; kk < kKTile; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
                wmma::load_matrix_sync(fa, x_s + tile_m[o] * 16 * kKTile + kk, kKTile);
                wmma::load_matrix_sync(fb, a_s + tile_n[o] * 16 * kKTile + kk, kKTile);
                wmma::mma_sync(acc[o], fa, fb, acc[o]);
            }
        }
        __syncthreads();
    }
    const int c_ld = n_tiles * 16;
    for (int o = 0; o < owned; ++o) {
        wmma::store_matrix_sync(c_s + tile_m[o] * 16 * c_ld + tile_n[o] * 16, acc[o], c_ld, wmma::mem_row_major);
    }
    __syncthreads();
    const int rank_stride = n_tiles * 16;
    for (int idx = threadIdx.x; idx < block * rank; idx += kThreads) {
        const int i = idx / rank, rr = idx % rank;
        const int r = rows_s[i];
        if (r < 0) { continue; }
        low[(static_cast<std::int64_t>(r) * 3 + part) * rank + rr] = __float2bfloat16(c_s[i * rank_stride + rr]);
    }
}

// out[row][column_offset(part) + c] += weight(row) * B[expert, part, slot][c] . low[row][part]: one CUDA block per
// (Marlin block, 64-column tile, part, slot).
__global__ __launch_bounds__(kThreads) void moe_lora_expand_kernel(
    __nv_bfloat16* __restrict__ out, int out_ld, const __nv_bfloat16* __restrict__ low,
    const std::int32_t* __restrict__ sorted_ids, const std::int32_t* __restrict__ expert_ids, int fixed_expert,
    const std::int32_t* __restrict__ past_padded, int block, int rows_total, int top_k,
    const LoraBank* __restrict__ banks, int part_first, int parts, int offset_first, int offset_second,
    const std::int32_t* __restrict__ slots, int slot_stride, int slot_count, int rank, int n,
    const float* __restrict__ row_weights) {
    const int b     = static_cast<int>(blockIdx.x);
    const int ntile = static_cast<int>(blockIdx.y);
    const int part  = part_first + static_cast<int>(blockIdx.z) % parts;
    const int slot  = static_cast<int>(blockIdx.z) / parts;
    if (slot >= slot_count || ntile * kNTile >= n) { return; }
    if (past_padded != nullptr && b * block >= *past_padded) { return; }
    if (past_padded == nullptr && b * block >= rows_total) { return; }
    const int expert = expert_ids != nullptr ? expert_ids[b] : fixed_expert;
    if (expert < 0) { return; }
    const LoraBank bank = banks[2 + 3 * expert + part];
    if (bank.b == nullptr || bank.n != n) { return; }

    __shared__ int rows_s[kMaxRows];
    __shared__ int any_s;
    if (threadIdx.x == 0) { any_s = 0; }
    __syncthreads();
    for (int i = threadIdx.x; i < block; i += kThreads) {
        int r = block_row(sorted_ids, block, b, i, rows_total);
        if (r >= 0 && row_slot(slots, slot_stride, r / top_k) != slot) { r = -1; }
        rows_s[i] = r;
        if (r >= 0) { atomicOr(&any_s, 1); }
    }
    __syncthreads();
    if (!any_s) { return; }

    const int m_tiles  = (block + 15) / 16;
    const int k_tiles  = (rank + 15) / 16;
    const int k_padded = k_tiles * 16;
    __shared__ __align__(32) __nv_bfloat16 l_s[kMaxRows * kMaxRankTiles * 16];
    __shared__ __align__(32) __nv_bfloat16 b_s[kNTile * kMaxRankTiles * 16];
    __shared__ __align__(32) float c_s[kMaxRows * kNTile];
    for (int idx = threadIdx.x; idx < m_tiles * 16 * k_padded; idx += kThreads) {
        const int i = idx / k_padded, rr = idx % k_padded;
        const int r = i < block ? rows_s[i] : -1;
        l_s[idx] = (r >= 0 && rr < rank) ? low[(static_cast<std::int64_t>(r) * 3 + part) * rank + rr]
                                         : __float2bfloat16(0.0F);
    }
    const auto* b_base = static_cast<const __nv_bfloat16*>(bank.b) + static_cast<std::int64_t>(slot) * bank.b_stride;
    for (int idx = threadIdx.x; idx < kNTile * k_padded; idx += kThreads) {
        const int c = idx / k_padded, rr = idx % k_padded;
        const int column = ntile * kNTile + c;
        b_s[idx] = (column < n && rr < rank) ? b_base[static_cast<std::int64_t>(column) * bank.rank + rr]
                                             : __float2bfloat16(0.0F);
    }
    __syncthreads();
    const int warp = static_cast<int>(threadIdx.x) / 32;
    for (int t = warp; t < m_tiles * (kNTile / 16); t += kThreads / 32) {
        const int tm = t / (kNTile / 16), tn = t % (kNTile / 16);
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0F);
        for (int kt = 0; kt < k_tiles; ++kt) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
            wmma::load_matrix_sync(fa, l_s + tm * 16 * k_padded + kt * 16, k_padded);
            wmma::load_matrix_sync(fb, b_s + tn * 16 * k_padded + kt * 16, k_padded);
            wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(c_s + tm * 16 * kNTile + tn * 16, acc, kNTile, wmma::mem_row_major);
    }
    __syncthreads();
    const int offset = part == part_first ? offset_first : offset_second;
    for (int idx = threadIdx.x; idx < block * kNTile; idx += kThreads) {
        const int i = idx / kNTile, c = idx % kNTile;
        const int r = rows_s[i];
        const int column = ntile * kNTile + c;
        if (r < 0 || column >= n) { continue; }
        const float w = row_weights != nullptr ? row_weights[r] : 1.0F;
        __nv_bfloat16* dst = out + static_cast<std::int64_t>(r) * out_ld + offset + column;
        *dst = __float2bfloat16(__bfloat162float(*dst) + w * c_s[i * kNTile + c]);
    }
}

// Narrow rounds (a decode or verify round: an expert holds a row or two) gain nothing from grouping, and the
// grouped kernels' fixed per-block cost is most of a narrow round's time: these walk the rows directly.
constexpr int kNarrowRows        = 1024;
// SUROGATE_SERVE_MARLIN_LORA_NARROW_ROWS overrides the crossover (0: grouped at every width).
int narrow_rows() {
    static const int value = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_MARLIN_LORA_NARROW_ROWS");
        return raw != nullptr && *raw != '\0' ? std::atoi(raw) : kNarrowRows;
    }();
    return value;
}
constexpr int kRowsShrinkThreads = 512; // sixteen warps, one per rank row of a 16-wide pass
constexpr int kRowsExpandThreads = 256; // one output column a thread

// low[row][part][rank] for each real row: one block per (row, part); warp w computes rank rows w, w + 16, ...
__global__ __launch_bounds__(kRowsShrinkThreads) void moe_lora_shrink_rows_kernel(
    const __nv_bfloat16* __restrict__ input, int input_per_token, const std::int32_t* __restrict__ ids,
    int fixed_expert, int rows_total, int top_k, const LoraBank* __restrict__ banks, int part_first, int parts,
    const std::int32_t* __restrict__ slots, int slot_stride, int rank, int k, __nv_bfloat16* __restrict__ low) {
    const int r    = static_cast<int>(blockIdx.x);
    const int part = part_first + static_cast<int>(blockIdx.y);
    if (r >= rows_total || static_cast<int>(blockIdx.y) >= parts) { return; }
    const int slot = row_slot(slots, slot_stride, r / top_k);
    if (slot < 0) { return; }
    const int expert = ids != nullptr ? ids[r] : fixed_expert;
    if (expert < 0) { return; }
    const LoraBank bank = banks[2 + 3 * expert + part];
    if (bank.a == nullptr || bank.k != k) { return; }
    const std::int64_t src = input_per_token ? static_cast<std::int64_t>(r / top_k) : r;
    const __nv_bfloat16* x = input + src * k;
    const auto* a_base = static_cast<const __nv_bfloat16*>(bank.a) + static_cast<std::int64_t>(slot) * bank.a_stride;
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    for (int rr = warp; rr < rank; rr += kRowsShrinkThreads / 32) {
        const __nv_bfloat16* a = a_base + static_cast<std::int64_t>(rr) * k;
        float sum = 0.0F;
        for (int kk = lane * 8; kk < k; kk += 32 * 8) {
            const uint4 av = *reinterpret_cast<const uint4*>(a + kk);
            const uint4 xv = *reinterpret_cast<const uint4*>(x + kk);
            const auto* a2 = reinterpret_cast<const __nv_bfloat162*>(&av);
            const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 af = __bfloat1622float2(a2[j]);
                const float2 xf = __bfloat1622float2(x2[j]);
                sum = fmaf(af.x, xf.x, sum);
                sum = fmaf(af.y, xf.y, sum);
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) { sum += __shfl_xor_sync(0xffffffffU, sum, offset); }
        if (lane == 0) { low[(static_cast<std::int64_t>(r) * 3 + part) * rank + rr] = __float2bfloat16(sum); }
    }
}

// out[row][offset + c] += weight * B[c] . low[row][part]: one block per (256 columns, row, part).
__global__ __launch_bounds__(kRowsExpandThreads) void moe_lora_expand_rows_kernel(
    __nv_bfloat16* __restrict__ out, int out_ld, const __nv_bfloat16* __restrict__ low,
    const std::int32_t* __restrict__ ids, int fixed_expert, int rows_total, int top_k,
    const LoraBank* __restrict__ banks, int part_first, int parts, int offset_first, int offset_second,
    const std::int32_t* __restrict__ slots, int slot_stride, int rank, int n, const float* __restrict__ row_weights) {
    const int r    = static_cast<int>(blockIdx.y);
    const int part = part_first + static_cast<int>(blockIdx.z);
    if (r >= rows_total || static_cast<int>(blockIdx.z) >= parts) { return; }
    const int slot = row_slot(slots, slot_stride, r / top_k);
    if (slot < 0) { return; }
    const int expert = ids != nullptr ? ids[r] : fixed_expert;
    if (expert < 0) { return; }
    const LoraBank bank = banks[2 + 3 * expert + part];
    if (bank.b == nullptr || bank.n != n) { return; }
    __shared__ float low_s[kMaxRankTiles * 16];
    for (int rr = threadIdx.x; rr < rank; rr += kRowsExpandThreads) {
        low_s[rr] = __bfloat162float(low[(static_cast<std::int64_t>(r) * 3 + part) * rank + rr]);
    }
    __syncthreads();
    const int column = static_cast<int>(blockIdx.x) * kRowsExpandThreads + static_cast<int>(threadIdx.x);
    if (column >= n) { return; }
    const auto* b = static_cast<const __nv_bfloat16*>(bank.b) + static_cast<std::int64_t>(slot) * bank.b_stride +
                    static_cast<std::int64_t>(column) * bank.rank;
    float sum = 0.0F;
    for (int rr = 0; rr < rank; ++rr) { sum = fmaf(__bfloat162float(b[rr]), low_s[rr], sum); }
    const float w = row_weights != nullptr ? row_weights[r] : 1.0F;
    __nv_bfloat16* dst = out + static_cast<std::int64_t>(r) * out_ld + (part == part_first ? offset_first : offset_second) + column;
    *dst = __float2bfloat16(__bfloat162float(*dst) + w * sum);
}

// Wide rounds, the fast path: one CUDA block per 64-slot tile of ONE expert's sorted rows (the tile table,
// laid out per call from Marlin's padded offsets) or of 64 dense rows (the shared expert), so an expert's
// A and B are read once per tile, not once per 8-48-row Marlin block; operands staged with cp.async,
// mma.sync m16n8k16 in the grouped kernels' k order (the same products and sums, so the same bits), the
// delta added by 16-byte read-modify-writes. Ranks up to 32 (a multiple of 8), parts x rank <= 64.
constexpr int kTileRows     = 64;
constexpr int kFastThreads  = 128; // four warps, sixteen rows each
constexpr int kShrinkK      = 64;  // K columns per cp.async stage
constexpr int kExpandCols   = 128; // output columns per CUDA block

// SUROGATE_SERVE_MARLIN_LORA_FAST=0 keeps wide rounds on the grouped WMMA kernels above.
bool fast_enabled() {
    static const bool value = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_MARLIN_LORA_FAST");
        return raw == nullptr || *raw == '\0' || std::atoi(raw) != 0;
    }();
    return value;
}

// tiles[t] = {expert, first padded slot}: each expert's padded span [offsets[e], offsets[e + 1]) cut
// into 64-slot tiles, in expert order. One block, a thread an expert.
__global__ void lora_tile_map_kernel(const std::int32_t* __restrict__ offsets, int experts,
                                     int2* __restrict__ tiles, std::int32_t* __restrict__ tile_count) {
    __shared__ int scan[1024];
    const int e     = static_cast<int>(threadIdx.x);
    const int start = e < experts ? offsets[e] : 0;
    const int count = e < experts ? (offsets[e + 1] - start + kTileRows - 1) / kTileRows : 0;
    scan[e] = count;
    __syncthreads();
    // Hillis-Steele inclusive scan, as offsets_kernel.
    for (int stride = 1; stride < static_cast<int>(blockDim.x); stride <<= 1) {
        const int add = e >= stride ? scan[e - stride] : 0;
        __syncthreads();
        scan[e] += add;
        __syncthreads();
    }
    const int first = scan[e] - count;
    for (int j = 0; j < count; ++j) { tiles[first + j] = make_int2(e, start + j * kTileRows); }
    if (e == static_cast<int>(blockDim.x) - 1) { *tile_count = scan[e]; }
}

// This block's tile: its expert and rows_s[i] = the row of slot i, or -1 (padding, past the tile, or a
// row whose token selected another slot). False (for the whole block) when the tile holds no row.
__device__ __forceinline__ bool fast_tile_rows(const std::int32_t* sorted_ids, const int2* tiles,
                                               const std::int32_t* tile_count, const std::int32_t* offsets,
                                               int fixed_expert, int rows_total, int top_k,
                                               const std::int32_t* slots, int slot_stride, int slot,
                                               int* rows_s, int& expert) {
    const int t = static_cast<int>(blockIdx.x);
    int first = 0;
    int last  = 0;
    if (sorted_ids != nullptr) {
        if (t >= *tile_count) { return false; }
        const int2 tile = tiles[t];
        expert          = tile.x;
        first           = tile.y;
        last            = min(first + kTileRows, offsets[expert + 1]);
    } else {
        expert = fixed_expert;
        first  = t * kTileRows;
        last   = min(first + kTileRows, rows_total);
    }
    int r = -1;
    if (threadIdx.x < kTileRows) {
        const int s = first + static_cast<int>(threadIdx.x);
        if (s < last) {
            r = sorted_ids != nullptr ? sorted_ids[s] : s;
            if (r >= rows_total || row_slot(slots, slot_stride, r / top_k) != slot) { r = -1; }
        }
        rows_s[threadIdx.x] = r;
    }
    return __syncthreads_or(r >= 0) != 0;
}

// low[row][part][rank] = A[expert, part, slot] . in(row) for the tile's rows, every part at once (the
// input row is staged once): a 64 x N x k GEMM, N = parts x rank16, K in 64-column cp.async stages.
template <int N>
__global__ __launch_bounds__(kFastThreads) void moe_lora_shrink_fast_kernel(
    const __nv_bfloat16* __restrict__ input, int input_per_token, const std::int32_t* __restrict__ sorted_ids,
    const int2* __restrict__ tiles, const std::int32_t* __restrict__ tile_count,
    const std::int32_t* __restrict__ offsets, int fixed_expert, int rows_total, int top_k,
    const LoraBank* __restrict__ banks, int part_first, int parts, const std::int32_t* __restrict__ slots,
    int slot_stride, int rank, int k, __nv_bfloat16* __restrict__ low) {
    constexpr int kStages = N <= 32 ? 3 : 2;
    __shared__ int rows_s[kTileRows];
    __shared__ std::int64_t src_s[kTileRows];
    __shared__ __align__(128) __nv_bfloat16 xs[kStages][kTileRows * kShrinkK];
    __shared__ __align__(128) __nv_bfloat16 as[kStages][N * kShrinkK];
    const int slot = static_cast<int>(blockIdx.y);
    int expert     = -1;
    if (!fast_tile_rows(sorted_ids, tiles, tile_count, offsets, fixed_expert, rows_total, top_k, slots,
                        slot_stride, slot, rows_s, expert)) {
        return;
    }
    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int rank16 = N / parts;
    const __nv_bfloat16* a_part[2] = {nullptr, nullptr};
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        if (p < parts) {
            const LoraBank bank = banks[2 + 3 * expert + part_first + p];
            if (bank.a != nullptr && bank.k == k) {
                a_part[p] = static_cast<const __nv_bfloat16*>(bank.a) + static_cast<std::int64_t>(slot) * bank.a_stride;
            }
        }
    }
    if (tid < kTileRows) {
        const int r = rows_s[tid];
        src_s[tid]  = r < 0 ? -1 : static_cast<std::int64_t>(input_per_token ? r / top_k : r) * k;
    }
    __syncthreads();
    const auto stage_inputs = [&](int stage, int kt) {
        const int k0 = kt * kShrinkK;
        for (int c = tid; c < kTileRows * (kShrinkK / 8); c += kFastThreads) {
            const int row          = c / (kShrinkK / 8);
            const int k8           = c - row * (kShrinkK / 8);
            const std::int64_t src = src_s[row];
            cp_async_zfill<16, Cache::cg>(&xs[stage][row * kShrinkK + gemm_swz64(row, k8 * 8)],
                                          input + (src >= 0 ? src : 0) + k0 + k8 * 8, src >= 0 ? 16 : 0);
        }
        for (int c = tid; c < N * (kShrinkK / 8); c += kFastThreads) {
            const int nrow = c / (kShrinkK / 8);
            const int k8   = c - nrow * (kShrinkK / 8);
            const int p    = nrow / rank16;
            const int rr   = nrow - p * rank16;
            const __nv_bfloat16* a = p == 0 ? a_part[0] : a_part[1];
            const bool real        = a != nullptr && rr < rank;
            cp_async_zfill<16, Cache::cg>(&as[stage][nrow * kShrinkK + gemm_swz64(nrow, k8 * 8)],
                                          real ? a + static_cast<std::int64_t>(rr) * k + k0 + k8 * 8 : input,
                                          real ? 16 : 0);
        }
    };
    const int k_tiles = k / kShrinkK;
#pragma unroll
    for (int stage = 0; stage < kStages - 1; ++stage) {
        if (stage < k_tiles) { stage_inputs(stage, stage); }
        cp_commit();
    }
    float acc[N / 8][4] = {};
    const int a_row = warp * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
    const int a_col = (lane >> 4) << 3;
    const int b_row = lane & 7;
    const int b_col = ((lane >> 3) & 1) << 3;
#pragma unroll 1
    for (int kt = 0; kt < k_tiles; ++kt) {
        cp_wait<kStages - 2>();
        __syncthreads();
        const int next = kt + kStages - 1;
        if (next < k_tiles) { stage_inputs(next % kStages, next); }
        cp_commit();
        const int stage = kt % kStages;
#pragma unroll
        for (int ki = 0; ki < kShrinkK / 16; ++ki) {
            unsigned af[4];
            const int ac = ki * 16 + a_col;
            ldmatrix_x4(af[0], af[1], af[2], af[3],
                        smem_addr(&xs[stage][a_row * kShrinkK + gemm_swz64(a_row, ac)]));
#pragma unroll
            for (int j = 0; j < N / 8; ++j) {
                unsigned bf[2];
                const int br = j * 8 + b_row;
                const int bc = ki * 16 + b_col;
                ldmatrix_x2(bf[0], bf[1], smem_addr(&as[stage][br * kShrinkK + gemm_swz64(br, bc)]));
                mma_bf16(acc[j][0], acc[j][1], acc[j][2], acc[j][3], af[0], af[1], af[2], af[3], bf[0], bf[1]);
            }
        }
    }
    cp_wait<0>();
    const int gid = lane >> 2;
    const int lid = lane & 3;
#pragma unroll
    for (int j = 0; j < N / 8; ++j) {
        const int col = j * 8 + 2 * lid;
        const int p   = col / rank16;
        const int rr  = col - p * rank16;
        if (rr >= rank) { continue; }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = rows_s[warp * 16 + gid + 8 * h];
            if (r < 0) { continue; }
            *reinterpret_cast<__nv_bfloat162*>(&low[(static_cast<std::int64_t>(r) * 3 + part_first + p) * rank + rr]) =
                __floats2bfloat162_rn(acc[j][2 * h], acc[j][2 * h + 1]);
        }
    }
}

// out[row][offset(part) + c] += weight(row) * B[expert, part, slot][c] . low[row][part] for the tile's rows
// and 128 output columns: a 64 x 128 x rank16 MMA, the delta staged through shared memory for 16-byte
// read-modify-writes of the product.
template <int KR>
__global__ __launch_bounds__(kFastThreads) void moe_lora_expand_fast_kernel(
    __nv_bfloat16* __restrict__ out, int out_ld, const __nv_bfloat16* __restrict__ low,
    const std::int32_t* __restrict__ sorted_ids, const int2* __restrict__ tiles,
    const std::int32_t* __restrict__ tile_count, const std::int32_t* __restrict__ offsets, int fixed_expert,
    int rows_total, int top_k, const LoraBank* __restrict__ banks, int part_first, int parts, int offset_first,
    int offset_second, const std::int32_t* __restrict__ slots, int slot_stride, int rank, int n,
    const float* __restrict__ row_weights) {
    constexpr int kLd           = KR + 8;           // padded operand rows: conflict-free ldmatrix
    constexpr int kCLd          = kExpandCols + 4;  // padded FP32 delta rows
    constexpr int kOperandBytes = (kTileRows + kExpandCols) * kLd * 2;
    constexpr int kDeltaBytes   = kTileRows * kCLd * 4;
    __shared__ int rows_s[kTileRows];
    __shared__ float w_s[kTileRows];
    __shared__ __align__(128) unsigned char smem[kOperandBytes > kDeltaBytes ? kOperandBytes : kDeltaBytes];
    auto* l_s = reinterpret_cast<__nv_bfloat16*>(smem);
    auto* b_s = l_s + kTileRows * kLd;
    auto* c_s = reinterpret_cast<float*>(smem); // after the MMA
    const int col0 = static_cast<int>(blockIdx.y) * kExpandCols;
    const int part = part_first + static_cast<int>(blockIdx.z) % parts;
    const int slot = static_cast<int>(blockIdx.z) / parts;
    int expert     = -1;
    if (!fast_tile_rows(sorted_ids, tiles, tile_count, offsets, fixed_expert, rows_total, top_k, slots,
                        slot_stride, slot, rows_s, expert)) {
        return;
    }
    const LoraBank bank = banks[2 + 3 * expert + part];
    if (bank.b == nullptr || bank.n != n) { return; }
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid < kTileRows) {
        const int r = rows_s[tid];
        w_s[tid]    = r >= 0 && row_weights != nullptr ? row_weights[r] : 1.0F;
    }
    for (int c = tid; c < kTileRows * (KR / 8); c += kFastThreads) {
        const int i    = c / (KR / 8);
        const int r8   = c - i * (KR / 8);
        const int r    = rows_s[i];
        const bool real = r >= 0 && r8 * 8 < rank;
        cp_async_zfill<16, Cache::cg>(&l_s[i * kLd + r8 * 8],
                                      real ? low + (static_cast<std::int64_t>(r) * 3 + part) * rank + r8 * 8 : low,
                                      real ? 16 : 0);
    }
    const auto* b_base = static_cast<const __nv_bfloat16*>(bank.b) + static_cast<std::int64_t>(slot) * bank.b_stride;
    for (int c = tid; c < kExpandCols * (KR / 8); c += kFastThreads) {
        const int col   = c / (KR / 8);
        const int r8    = c - col * (KR / 8);
        const bool real = r8 * 8 < rank;
        cp_async_zfill<16, Cache::cg>(&b_s[col * kLd + r8 * 8],
                                      real ? b_base + static_cast<std::int64_t>(col0 + col) * bank.rank + r8 * 8
                                           : b_base,
                                      real ? 16 : 0);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    float acc[kExpandCols / 8][4] = {};
    const int a_row = warp * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
    const int a_col = (lane >> 4) << 3;
    const int b_row = lane & 7;
    const int b_col = ((lane >> 3) & 1) << 3;
#pragma unroll
    for (int kk = 0; kk < KR / 16; ++kk) {
        unsigned af[4];
        ldmatrix_x4(af[0], af[1], af[2], af[3], smem_addr(&l_s[a_row * kLd + kk * 16 + a_col]));
#pragma unroll
        for (int j = 0; j < kExpandCols / 8; ++j) {
            unsigned bf[2];
            ldmatrix_x2(bf[0], bf[1], smem_addr(&b_s[(j * 8 + b_row) * kLd + kk * 16 + b_col]));
            mma_bf16(acc[j][0], acc[j][1], acc[j][2], acc[j][3], af[0], af[1], af[2], af[3], bf[0], bf[1]);
        }
    }
    __syncthreads(); // the operands are dead: the delta takes their shared memory
    const int gid = lane >> 2;
    const int lid = lane & 3;
#pragma unroll
    for (int j = 0; j < kExpandCols / 8; ++j) {
        const int col  = j * 8 + 2 * lid;
        const int row0 = warp * 16 + gid;
        *reinterpret_cast<float2*>(&c_s[row0 * kCLd + col])       = make_float2(acc[j][0], acc[j][1]);
        *reinterpret_cast<float2*>(&c_s[(row0 + 8) * kCLd + col]) = make_float2(acc[j][2], acc[j][3]);
    }
    __syncthreads();
    const int offset = part == part_first ? offset_first : offset_second;
    for (int c = tid; c < kTileRows * (kExpandCols / 8); c += kFastThreads) {
        const int i  = c / (kExpandCols / 8);
        const int c8 = c - i * (kExpandCols / 8);
        const int r  = rows_s[i];
        if (r < 0) { continue; }
        auto* dst = reinterpret_cast<uint4*>(out + static_cast<std::int64_t>(r) * out_ld + offset + col0 + c8 * 8);
        uint4 v    = *dst;
        auto* v2   = reinterpret_cast<__nv_bfloat162*>(&v);
        const float w      = w_s[i];
        const float* delta = &c_s[i * kCLd + c8 * 8];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 o = __bfloat1622float2(v2[e]);
            v2[e] = __floats2bfloat162_rn(o.x + w * delta[2 * e], o.y + w * delta[2 * e + 1]);
        }
        *dst = v;
    }
}

} // namespace

std::int32_t marlin_moe_lora_tile_capacity(std::int32_t assignments, std::int32_t experts) {
    // Each expert's span is padded to its Marlin block (<= 64 rows): sum of ceil(span / 64) <=
    // (assignments + 63 * experts) / 64 + experts.
    return (assignments + (kTileRows - 1) * experts) / kTileRows + experts + 1;
}

void marlin_moe_lora_delta(const MarlinMoeLoraCall& call, cudaStream_t stream) {
    if (call.rank <= 0 || call.rank > kMaxRankTiles * 16 || call.block <= 0 || call.block > kMaxRows ||
        call.parts < 1 || call.parts > 2 || call.k <= 0 || call.n <= 0 || call.slot_count <= 0) {
        throw std::invalid_argument("marlin moe lora: call outside the kernels' domain");
    }
    const int part_first = call.down ? 2 : 0;
    const int blocks     = call.block_capacity;
    if (call.rows <= narrow_rows() && call.k % 256 == 0) {
        const dim3 shrink(static_cast<unsigned>(call.rows), static_cast<unsigned>(call.parts));
        moe_lora_shrink_rows_kernel<<<shrink, kRowsShrinkThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(call.input), call.input_per_token ? 1 : 0, call.ids,
            call.fixed_expert, call.rows, call.top_k, call.banks, part_first, call.parts, call.slots,
            call.slot_stride, call.rank, call.k, static_cast<__nv_bfloat16*>(call.low));
        CUDA_CHECK(cudaGetLastError());
        const dim3 expand(static_cast<unsigned>((call.n + kRowsExpandThreads - 1) / kRowsExpandThreads),
                          static_cast<unsigned>(call.rows), static_cast<unsigned>(call.parts));
        moe_lora_expand_rows_kernel<<<expand, kRowsExpandThreads, 0, stream>>>(
            static_cast<__nv_bfloat16*>(call.out), call.out_ld, static_cast<const __nv_bfloat16*>(call.low),
            call.ids, call.fixed_expert, call.rows, call.top_k, call.banks, part_first, call.parts,
            call.column_offset[0], call.column_offset[1], call.slots, call.slot_stride, call.rank, call.n,
            call.row_weights);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const int rank16 = (call.rank + 15) / 16 * 16;
    const bool tiled = call.sorted_ids == nullptr ||
                       (call.offsets != nullptr && call.tiles != nullptr && call.tile_count != nullptr &&
                        call.experts > 0 && call.experts <= 1024 && call.tile_capacity > 0);
    if (fast_enabled() && tiled && call.rank % 8 == 0 && rank16 <= 32 && call.parts * rank16 <= 64 &&
        call.k % kShrinkK == 0 && call.n % kExpandCols == 0) {
        unsigned tiles = 0;
        if (call.sorted_ids != nullptr) {
            lora_tile_map_kernel<<<1, static_cast<unsigned>((call.experts + 31) / 32 * 32), 0, stream>>>(
                call.offsets, call.experts, call.tiles, call.tile_count);
            CUDA_CHECK(cudaGetLastError());
            tiles = static_cast<unsigned>(call.tile_capacity);
        } else {
            tiles = static_cast<unsigned>((call.rows + kTileRows - 1) / kTileRows);
        }
        const auto* input = static_cast<const __nv_bfloat16*>(call.input);
        auto* low         = static_cast<__nv_bfloat16*>(call.low);
        const dim3 shrink(tiles, static_cast<unsigned>(call.slot_count));
        const auto launch_shrink = [&](auto width) {
            moe_lora_shrink_fast_kernel<decltype(width)::value><<<shrink, kFastThreads, 0, stream>>>(
                input, call.input_per_token ? 1 : 0, call.sorted_ids, call.tiles, call.tile_count, call.offsets,
                call.fixed_expert, call.rows, call.top_k, call.banks, part_first, call.parts, call.slots,
                call.slot_stride, call.rank, call.k, low);
        };
        if (call.parts * rank16 == 16) {
            launch_shrink(std::integral_constant<int, 16>{});
        } else if (call.parts * rank16 == 32) {
            launch_shrink(std::integral_constant<int, 32>{});
        } else {
            launch_shrink(std::integral_constant<int, 64>{});
        }
        CUDA_CHECK(cudaGetLastError());
        const dim3 expand(tiles, static_cast<unsigned>(call.n / kExpandCols),
                          static_cast<unsigned>(call.parts * call.slot_count));
        const auto launch_expand = [&](auto width) {
            moe_lora_expand_fast_kernel<decltype(width)::value><<<expand, kFastThreads, 0, stream>>>(
                static_cast<__nv_bfloat16*>(call.out), call.out_ld, low, call.sorted_ids, call.tiles,
                call.tile_count, call.offsets, call.fixed_expert, call.rows, call.top_k, call.banks, part_first,
                call.parts, call.column_offset[0], call.column_offset[1], call.slots, call.slot_stride, call.rank,
                call.n, call.row_weights);
        };
        if (rank16 == 16) {
            launch_expand(std::integral_constant<int, 16>{});
        } else {
            launch_expand(std::integral_constant<int, 32>{});
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    {
        const dim3 grid(static_cast<unsigned>(blocks), static_cast<unsigned>(call.parts * call.slot_count));
        moe_lora_shrink_kernel<<<grid, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(call.input), call.input_per_token ? 1 : 0, call.sorted_ids,
            call.expert_ids, call.fixed_expert, call.past_padded, call.block, call.rows, call.top_k, call.banks,
            part_first, call.parts, call.slots, call.slot_stride, call.slot_count, call.rank, call.k,
            static_cast<__nv_bfloat16*>(call.low));
        CUDA_CHECK(cudaGetLastError());
    }
    {
        const dim3 grid(static_cast<unsigned>(blocks), static_cast<unsigned>((call.n + kNTile - 1) / kNTile),
                        static_cast<unsigned>(call.parts * call.slot_count));
        moe_lora_expand_kernel<<<grid, kThreads, 0, stream>>>(
            static_cast<__nv_bfloat16*>(call.out), call.out_ld, static_cast<const __nv_bfloat16*>(call.low),
            call.sorted_ids, call.expert_ids, call.fixed_expert, call.past_padded, call.block, call.rows, call.top_k,
            call.banks, part_first, call.parts, call.column_offset[0], call.column_offset[1], call.slots,
            call.slot_stride, call.slot_count, call.rank, call.n, call.row_weights);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace sinfer::ops::detail
