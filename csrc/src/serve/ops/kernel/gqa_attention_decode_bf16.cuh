#pragma once

#include "api/ops/gqa_attention.h"

// sinfer::ops - split-KV GQA small-T attention, BF16 KV-cache partial kernel.
// Standalone from the int8 kernel (gqa_attention_decode_i8.cuh): shared scaffolding
// lives in gqa_attention_decode.cuh, but the body/append/load are not shared so the
// bf16 path can be tuned independently. Processes one KV head, one query-head
// subgroup, and one token tile; a reducer combines the split-local partials.

#include <cuda_bf16.h>
#include <math_constants.h>

#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"

#include <cstdint>


namespace sinfer::ops {

// CacheT selects the KV cache storage: __nv_bfloat16 for the full-precision
// cache, std::uint8_t for the e4m3 one. Only the append and the tile stage
// differ -- the codes are widened to bf16 on the way into shared memory, so the
// MMA path, the swizzle and the shared-memory budget are identical either way.
// That keeps this a bandwidth change rather than a second attention kernel,
// which is the distinction that matters: decode here runs at about half its
// KV-read roofline, so halving the bytes is the win, and the int8 cache needs
// its own kernel only because it feeds IMMA instead.
// QSA sparse selection (design/INFERENCE.md, phase 4): `block_mask` holds one bit per block of
// `SparseBlock` cells for every query column, and a key whose block bit is clear scores -inf.
// `Sparse == false` compiles to exactly the dense kernel.
template <bool Sparse, int SparseBlock>
__device__ __forceinline__ bool gqa_block_visible(const std::uint32_t* row_words, int key) {
    if constexpr (!Sparse) {
        return true;
    } else {
        const int block = key / SparseBlock;
        return ((row_words[block >> 5] >> (block & 31)) & 1U) != 0U;
    }
}

/// The kernel's shared tiles, sized from the same constants the body derives.
///
/// They are *dynamic* shared memory, not static: a 512-wide head wants 68 KB and Ada caps a
/// block's static allocation at 48 KB, while its dynamic cap with `cudaFuncSetAttribute` is
/// 99 KB. The same kernel therefore compiles for sm_89 and sm_120 alike, and the launcher
/// raises the limit once per device before the first launch.
template <typename Geometry, int WarpsPerCta>
struct GqaSmallTTcSmem {
    static constexpr int kBc       = 32;
    static constexpr int kQkvRows  = 2 * kBc;
    static constexpr int kPageIds  = 64;
    static constexpr int kQkvBytes = kQkvRows * Geometry::HeadDim * static_cast<int>(sizeof(__nv_bfloat16));
    static constexpr int kPBytes   = WarpsPerCta * 16 * kBc * static_cast<int>(sizeof(__nv_bfloat16));
    static constexpr int kPageBytes = kPageIds * static_cast<int>(sizeof(std::int32_t));
    static constexpr int kBytes    = kQkvBytes + kPBytes + kPageBytes;
};

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput, typename CacheT = __nv_bfloat16, bool Sparse = false,
          int SparseBlock = 4>
__launch_bounds__(128, 2) __global__ void gqa_attention_small_t_tc_partial_bf16_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, CacheT* cache_k,
    CacheT* cache_v, const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity,
    std::int32_t sliding_window, float scale,
    float* partial_acc, float* partial_m, float* partial_l,
    GqaBlockMask block_mask = GqaBlockMask{}) {
    static_assert(TokenTile >= 1 && TokenTile <= 32);
    static_assert(WarpsPerCta >= 1 && WarpsPerCta <= 4);

    constexpr bool kFp8Cache = GqaKvIsFp8<CacheT>::value;
    static_assert(kFp8Cache || sizeof(CacheT) == sizeof(__nv_bfloat16),
                  "KV cache storage must be bf16 or e4m3 codes");

    // One 8-value K/V pair from the cache into the swizzled shared tile. The
    // bf16 cache moves the bytes asynchronously and untouched; the e4m3 cache
    // has to widen them, so it loads and converts inline. That costs the
    // async stage, but the surrounding loop commits and waits on every tile
    // rather than pipelining across them, so nothing was being overlapped.
    const auto stage_cache_pair = [](__nv_bfloat16* k_dst, __nv_bfloat16* v_dst,
                                     const CacheT* k_src, const CacheT* v_src) {
        if constexpr (kFp8Cache) {
            const int2 k_raw = load_vec<int2>(k_src);
            const int2 v_raw = load_vec<int2>(v_src);
            store_vec(k_dst, gqa_kv_dequant_fp8x8_raw(k_raw));
            store_vec(v_dst, gqa_kv_dequant_fp8x8_raw(v_raw));
        } else {
            sinfer::ops::cp_async<16>(k_dst, k_src);
            sinfer::ops::cp_async<16>(v_dst, v_src);
        }
    };

    constexpr int Wc      = WarpsPerCta;
    constexpr int Br      = Wc * 16;
    constexpr int Bc      = 32;
    constexpr int D       = Geometry::HeadDim;
    constexpr int Threads = Wc * 32;
    constexpr int QKNt    = Bc / 8;
    constexpr int QKKs    = D / 16;
    constexpr int PVNt    = D / 8;
    constexpr int PVKs    = Bc / 16;
    // The GQA Op's 262144-key maximum envelope spans at most 49 pages in one 27B split.
    constexpr int PageIds       = 64;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    constexpr int QkvRows       = 2 * Bc;

    static_assert(QkvRows >= Br);

    using Smem = GqaSmallTTcSmem<Geometry, WarpsPerCta>;
    static_assert(Smem::kQkvBytes == QkvRows * D * static_cast<int>(sizeof(__nv_bfloat16)));
    static_assert(Smem::kPBytes == Wc * 16 * Bc * static_cast<int>(sizeof(__nv_bfloat16)));
    static_assert(Smem::kPageIds == PageIds);
    extern __shared__ __align__(16) unsigned char gqa_small_t_smem[];
    auto* qkv_s = reinterpret_cast<__nv_bfloat16*>(gqa_small_t_smem);
    auto* p_s   = reinterpret_cast<__nv_bfloat16*>(gqa_small_t_smem + Smem::kQkvBytes);
    auto* physical_pages_s =
        reinterpret_cast<std::int32_t*>(gqa_small_t_smem + Smem::kQkvBytes + Smem::kPBytes);
    __nv_bfloat16* k_s = qkv_s;
    __nv_bfloat16* v_s = qkv_s + Bc * D;

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    int valid_tokens      = tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }
    const int row_count = tokens * Geometry::GroupSize;

    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(Geometry::HeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(Geometry::HeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(Geometry::HeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * Geometry::HeadDim * Geometry::QHeads * tokens *
                       split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = 0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] =
                    0.0f;
            }
        }
    };

    // A lane step is `tokens` of GroupSize query rows each, and every one has to land in this
    // CTA's Br-row tile: the launcher's warp count is what makes that so. When it does not, the
    // CTA used to return without writing anything, and a group of sixty-four over two warps
    // found that the silent way -- zeros for output and a cache row never appended. An
    // impossible state is reported as one instead of read as an answer.
    if (tokens > TokenTile || row_count > Br) { __trap(); }
    if (kv_head < 0 || kv_head >= Geometry::KVHeads || tokens < 1 || split_count <= 0) {
        return;
    }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    // Partition the keys this tile can actually see, not every key ever written.
    // Below `key_lo` the window excludes every token in the tile, so those keys
    // were staged and scored only to be masked away.
    // Absolute partitions keep each query's arithmetic independent of its batch or tile.
    const int first_partition = gqa_key_partition(gqa_small_t_key_lo(first_pos, sliding_window));
    const int key_hi = last_pos + 1;
    const int active_split_count =
        min(split_count, gqa_key_partition(last_pos) + 1 - first_partition);
    if (split >= active_split_count) { return; }

    const int split_start = gqa_key_partition_begin(first_partition + split);
    const int split_end = min(gqa_key_partition_begin(first_partition + split + 1), key_hi);
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < page_count; page += Threads) {
        physical_pages_s[page] = block_table[first_page + page];
    }

    if constexpr (CacheInput::writes_cache) {
        // The owning split writes each new row. Current attention reads those rows directly from
        // input below, so no split depends on another split's cache write.
        for (int chunk = tid; chunk < valid_tokens * (D / 8); chunk += Threads) {
            const int token = chunk / (D / 8);
            const int d     = (chunk - token * (D / 8)) * 8;
            const int p_tok = pos[token];
            if (p_tok >= split_start && p_tok < split_end && p_tok >= 0 &&
                p_tok < logical_capacity) {
                const std::int64_t new_off = gqa_kv_new_index<Geometry>(kv_head, d, token);
                // Every lane looks its own page up. The lane0-plus-broadcast form
                // this replaced was valid only while a warp was guaranteed to
                // share one `p_tok`, which holds exactly when D / 8 == 32, i.e.
                // head dim 256: at 128 a warp spans two tokens, so the broadcast
                // both hangs (a full-mask shuffle under a predicate the warp no
                // longer agrees on) and writes the upper half of the warp to the
                // wrong page. The lookup is one L1-resident int load next to the
                // int4 K/V traffic either side of it.
                const int physical_page = paged_kv_physical_page(block_table, p_tok);
                const std::int64_t cache_off =
                    gqa_cache_index<Geometry>(physical_page, kv_head, d, p_tok & kPagedKVPageMask);
                if constexpr (kFp8Cache) {
                    gqa_kv_store_fp8x8(&cache_k[cache_off], &input.k[new_off]);
                    gqa_kv_store_fp8x8(&cache_v[cache_off], &input.v[new_off]);
                } else {
                    store_vec(&cache_k[cache_off], load_vec<int4>(&input.k[new_off]));
                    store_vec(&cache_v[cache_off], load_vec<int4>(&input.v[new_off]));
                }
            }
        }
        __syncthreads();
    }

    for (int idx = tid; idx < Br * D; idx += Threads) {
        const int row = idx / D;
        const int d   = idx - row * D;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (row < row_count && gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            value = q[gqa_q_index<Geometry>(q_head, d, token)];
        }
        qkv_s[row * D + gqa_small_t_tc_swz(row, d)] = value;
    }
    __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    const int warp_row0 = warp * 16;
    __nv_bfloat16* p_sw = &p_s[warp * 16 * Bc];

    unsigned af_q[QKKs][4];
#pragma unroll
    for (int k = 0; k < QKKs; ++k) {
        const int arow = warp_row0 + a_rowoff;
        const int acol = k * 16 + a_coloff;
        ldmatrix_x4(af_q[k][0], af_q[k][1], af_q[k][2], af_q[k][3],
                    smem_addr(&qkv_s[arow * D + gqa_small_t_tc_swz(arow, acol)]));
    }
    __syncthreads();
    int physical_page = physical_pages_s[0];
    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;
        if (kb != 0 && (k0 & kPagedKVPageMask) == 0) {
            physical_page = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];
        }
        // Stage the bf16 K/V key tile with one cp.async wave (16B/thread, high MLP).
        // Current-step tokens come from k_new/v_new; tail slots are zeroed.
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
            const int key_l      = chunk / (D / 8);
            const int d          = (chunk - key_l * (D / 8)) * 8;
            const int key        = k0 + key_l;
            __nv_bfloat16* k_dst = &k_s[key_l * D + gqa_small_t_tc_swz(key_l, d)];
            __nv_bfloat16* v_dst = &v_s[key_l * D + gqa_small_t_tc_swz(key_l, d)];
            if (key >= split_start && key < split_end) {
                if constexpr (CacheInput::writes_cache) {
                    const int new_token = key - first_pos;
                    const bool from_new =
                        new_token >= 0 && new_token < valid_tokens && key >= first_pos;
                    if (from_new) {
                        const std::int64_t off = gqa_kv_new_index<Geometry>(kv_head, d, new_token);
                        if constexpr (kFp8Cache) {
                            store_vec(k_dst, gqa_kv_round_fp8x8(&input.k[off]));
                            store_vec(v_dst, gqa_kv_round_fp8x8(&input.v[off]));
                        } else {
                            sinfer::ops::cp_async<16>(k_dst, &input.k[off]);
                            sinfer::ops::cp_async<16>(v_dst, &input.v[off]);
                        }
                    } else {
                        const std::int64_t off = gqa_cache_index<Geometry>(
                            physical_page, kv_head, d, key & kPagedKVPageMask);
                        stage_cache_pair(k_dst, v_dst, &cache_k[off], &cache_v[off]);
                    }
                } else {
                    const std::int64_t off = gqa_cache_index<Geometry>(physical_page, kv_head, d,
                                                                       key & kPagedKVPageMask);
                    stage_cache_pair(k_dst, v_dst, &cache_k[off], &cache_v[off]);
                }
            } else {
                store_vec(k_dst, make_int4(0, 0, 0, 0));
                store_vec(v_dst, make_int4(0, 0, 0, 0));
            }
        }
        sinfer::ops::cp_commit();
        sinfer::ops::cp_wait<0>();
        __syncthreads();

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
                unsigned bf[2];
                const int brow = nt * 8 + b_rin;
                const int bcol = k * 16 + b_koff;
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(&k_s[brow * D + gqa_small_t_tc_swz(brow, bcol)]));
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af_q[k][0],
                         af_q[k][1], af_q[k][2], af_q[k][3], bf[0], bf[1]);
            }
        }

        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head0, token0);
        gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head1, token1);
        const int qabs0 = (row0 < row_count) ? pos[token0] : -1;
        const int qabs1 = (row1 < row_count) ? pos[token1] : -1;
        // A query column's own mask row; the heads of a column share one selection. The mask is
        // laid out like q — [words, full_width, batch] — so a batched round indexes its own
        // sequence's rows and a chunked launch adds its column offset.
        const std::int64_t mask_base =
            static_cast<std::int64_t>(MultiBatch ? batch : 0) * full_width + column_base;
        const std::uint32_t* mask0 =
            Sparse && row0 < row_count
                ? block_mask.words + (mask_base + token0) * block_mask.stride
                : nullptr;
        const std::uint32_t* mask1 =
            Sparse && row1 < row_count
                ? block_mask.words + (mask_base + token1) * block_mask.stride
                : nullptr;

        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0 = nt * 8 + 2 * lid;
            const int col1 = col0 + 1;
            const int key0 = k0 + col0;
            const int key1 = col1 + k0;
            score[nt][0] = (row0 < row_count && key0 >= split_start && key0 < split_end &&
                            key0 <= qabs0 && gqa_within_window(qabs0, key0, sliding_window) && gqa_block_visible<Sparse, SparseBlock>(mask0, key0))
                               ? score[nt][0] * scale
                               : -CUDART_INF_F;
            score[nt][1] = (row0 < row_count && key1 >= split_start && key1 < split_end &&
                            key1 <= qabs0 && gqa_within_window(qabs0, key1, sliding_window) && gqa_block_visible<Sparse, SparseBlock>(mask0, key1))
                               ? score[nt][1] * scale
                               : -CUDART_INF_F;
            score[nt][2] = (row1 < row_count && key0 >= split_start && key0 < split_end &&
                            key0 <= qabs1 && gqa_within_window(qabs1, key0, sliding_window) && gqa_block_visible<Sparse, SparseBlock>(mask1, key0))
                               ? score[nt][2] * scale
                               : -CUDART_INF_F;
            score[nt][3] = (row1 < row_count && key1 >= split_start && key1 < split_end &&
                            key1 <= qabs1 && gqa_within_window(qabs1, key1, sliding_window) && gqa_block_visible<Sparse, SparseBlock>(mask1, key1))
                               ? score[nt][3] * scale
                               : -CUDART_INF_F;
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0    = fmaxf(m0, bm0);
        const float nm1    = fmaxf(m1, bm1);
        const float alpha0 = (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
        const float alpha1 = (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx((m1 - nm1) * Log2E);

        const float beta0 = (bm0 == -CUDART_INF_F) ? 0.0f : exp2_approx((bm0 - nm0) * Log2E);
        const float beta1 = (bm1 == -CUDART_INF_F) ? 0.0f : exp2_approx((bm1 - nm1) * Log2E);

        float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0  = nt * 8 + 2 * lid;
            const int col1  = col0 + 1;
            const float p00 = (bm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][0] - bm0) * Log2E)
                                  : 0.0f;
            const float p01 = (bm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][1] - bm0) * Log2E)
                                  : 0.0f;
            const float p10 = (bm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][2] - bm1) * Log2E)
                                  : 0.0f;
            const float p11 = (bm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][3] - bm1) * Log2E)
                                  : 0.0f;
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col0)]           = __float2bfloat16(p00);
            p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col1)]           = __float2bfloat16(p01);
            p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col0)] = __float2bfloat16(p10);
            p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col1)] = __float2bfloat16(p11);
        }
        bl0 = warp_sum<4>(bl0, FullMask);
        bl1 = warp_sum<4>(bl1, FullMask);

        l0 = __fmaf_rn(l0, alpha0, bl0 * beta0);
        l1 = __fmaf_rn(l1, alpha1, bl1 * beta1);
        m0 = nm0;
        m1 = nm1;
        __syncwarp();

#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            float tile_acc[4] = {};
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned pf[4];
                const int pcol = k * 16 + a_coloff;
                ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                            smem_addr(&p_sw[a_rowoff * Bc + gqa_small_t_tc_swz32(a_rowoff, pcol)]));
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_s[vrow * D + gqa_small_t_tc_swz(vrow, vcol)]));
                mma_bf16(tile_acc[0], tile_acc[1], tile_acc[2], tile_acc[3], pf[0], pf[1], pf[2], pf[3],
                         vf[0], vf[1]);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[n][i] = __fmaf_rn(acc[n][i], i < 2 ? alpha0 : alpha1,
                                      tile_acc[i] * (i < 2 ? beta0 : beta1));
            }
        }
        __syncthreads();
    }

    if (lid == 0) {
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < row_count) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m0;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l0;
        }
        if (row1 < row_count) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m1;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l1;
        }
    }

    // Preserve split numerators until normalization. Rounding the unnormalized
    // sum to BF16 here adds a second rounding absent from prompt attention.
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d = n * 8 + 2 * lid;
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < row_count) {
            int q_head = 0, token = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                const auto dst = gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens);
                store_vec(&partial_acc[dst], make_float2(acc[n][0], acc[n][1]));
            }
        }
        if (row1 < row_count) {
            int q_head = 0, token = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                const auto dst = gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens);
                store_vec(&partial_acc[dst], make_float2(acc[n][2], acc[n][3]));
            }
        }
    }
}

} // namespace sinfer::ops
