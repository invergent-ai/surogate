// LoRA on the Marlin MoE route (see marlin_moe_lora.h).

#include "ops/sparse_moe/marlin/marlin_moe_lora.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

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

} // namespace

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
