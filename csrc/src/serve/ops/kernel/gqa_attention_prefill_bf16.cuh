#pragma once

#include "api/ops/gqa_attention.h"

// gqa_block_visible is shared with the decode kernel.
#include "ops/kernel/gqa_attention_decode_bf16.cuh"

// BF16-only GQA prompt kernel. INT8 has an independent kernel body and resource
// policy in gqa_attention_prefill_i8.cuh.
//
//   * Br = 64 query rows and Bc = 32 key columns per CTA tile.
//   * 4 warps / 128 threads; each warp owns 16 query rows of the tile.
//   * Q, K, V staged in (Br + 2*Bc) * head_dim bf16 of dynamic shared memory
//     (64 KiB at head dim 256, 32 KiB at 128), single-buffered, with
//     the cp.async of the next K/V tile overlapped against the current
//     QK / PV tensor-core work (exactly FA's single-buffer overlap pattern).
//   * m16n8k16 bf16 MMA for both S = Q Kᵀ and O += P V, online softmax in exp2.
//
// The op first writes the new chunk K/V into absolute positions in the paged cache,
// then computes causal GQA attention for
// every chunk token over all cached history using bottom-right causal alignment
// (query row i attends to keys [0, base_pos + i]).

#include <math_constants.h>

#include "ops/kernel/gqa_attention_prefill_common.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"

namespace sinfer::ops {

template <typename Geometry, typename Metadata, typename CacheT = __nv_bfloat16>
__global__ void gqa_attention_prefill_fill_bf16_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, Metadata metadata,
    CacheT* __restrict__ cache_k, CacheT* __restrict__ cache_v, std::int32_t width) {
    constexpr bool kFp8Cache = GqaKvIsFp8<CacheT>::value;
    constexpr int D        = Geometry::HeadDim;
    constexpr int VecElems = 8; // 8 bf16 == 16 B, matching the cache row alignment.
    const int tokens       = metadata.valid_tokens(width);
    const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t n =
        static_cast<std::int64_t>(tokens) * Geometry::KVHeads * (D / VecElems);
    if (idx >= n) { return; }

    const int vec                   = static_cast<int>(idx % (D / VecElems));
    const int tmp                   = static_cast<int>(idx / (D / VecElems));
    const int kv_head               = tmp % Geometry::KVHeads;
    const int token                 = tmp / Geometry::KVHeads;
    const int d                     = vec * VecElems;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    // Every lane looks its own page up. The lane0-plus-broadcast form this
    // replaced was valid only while a warp was guaranteed to share one
    // `position`, which holds exactly when D / VecElems == 32, i.e. head dim 256.
    // At 128 a warp spans two `tmp` values, which always differ in kv_head and
    // straddle a token boundary once every KVHeads warps, so the broadcast wrote
    // those lanes' K/V to the wrong cache page -- silently, with no hang here
    // only because n stays a multiple of 32 for this geometry.
    const int physical_page = paged_kv_physical_page(block_table, position);
    const std::int64_t src_off =
        static_cast<std::int64_t>(d) +
        static_cast<std::int64_t>(D) * (kv_head + Geometry::KVHeads * token);

    const std::int64_t cache_off = paged_kv_element_offset<D, Geometry::KVHeads>(
        physical_page, kv_head, position & kPagedKVPageMask, d);
    if constexpr (kFp8Cache) {
        gqa_kv_store_fp8x8(&cache_k[cache_off], &k[src_off]);
        gqa_kv_store_fp8x8(&cache_v[cache_off], &v[src_off]);
    } else {
        store_vec(&cache_k[cache_off], load_vec<int4>(&k[src_off]));
        store_vec(&cache_v[cache_off], load_vec<int4>(&v[src_off]));
    }
}

// Stage one [Bc, D] K or V tile from the per-kv-head contiguous cache into the
// swizzled smem buffer. Keys beyond max_query_abs (which the causal mask always
// drops) are zeroed so the padded/uninitialized cache tail never feeds NaNs into
// the tensor cores. Mirrors FA's predicated K/V cp.async + Clear_OOB path.
template <typename Geometry, typename CacheT = __nv_bfloat16>
__device__ __forceinline__ void gqa_prefill_stage_kv(__nv_bfloat16* dst, const CacheT* cache,
                                                     int kv_head, int k0, int max_query_abs,
                                                     int physical_page, int tid) {
    constexpr int D         = Geometry::HeadDim;
    constexpr int Bc        = kGqaPrefillBcFor<Geometry::HeadDim>;
    constexpr int Threads   = kGqaPrefillThreads;
    constexpr int VecPerRow = D / 8; // 8 bf16 per 16B cp.async
    const bool full_tile    = (k0 + Bc - 1) <= max_query_abs;
    // Block base pointer computed once (int64); per-element offsets stay 32-bit.
    constexpr bool kFp8Cache = GqaKvIsFp8<CacheT>::value;
    const CacheT* cache_block =
        cache + paged_kv_element_offset<D, Geometry::KVHeads>(
                    physical_page, kv_head, k0 & kPagedKVPageMask, 0);
    // e4m3 codes are widened on the way into the swizzled tile; bf16 moves untouched.
    const auto stage_one = [](__nv_bfloat16* p, const CacheT* src) {
        if constexpr (kFp8Cache) {
            store_vec(p, gqa_kv_dequant_fp8x8_from(src));
        } else {
            cp_async<16, Cache::cg>(p, src);
        }
    };
    if (full_tile) {
#pragma unroll
        for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
            const int key_l  = chunk / VecPerRow;
            const int d      = (chunk % VecPerRow) * 8;
            __nv_bfloat16* p = &dst[key_l * D + gqa_prefill_swz(key_l, d)];
            stage_one(p, &cache_block[key_l * D + d]);
        }
    } else {
#pragma unroll
        for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
            const int key_l  = chunk / VecPerRow;
            const int d      = (chunk % VecPerRow) * 8;
            __nv_bfloat16* p = &dst[key_l * D + gqa_prefill_swz(key_l, d)];
            if ((k0 + key_l) <= max_query_abs) {
                stage_one(p, &cache_block[key_l * D + d]);
            } else {
                store_vec(p, make_int4(0, 0, 0, 0));
            }
        }
    }
}

// FlashAttention-2 forward, one CTA per (query 64-row block, query head). Grid is
// (ceil(tokens/64), q_heads). seqlen_q = tokens, seqlen_k = base_pos + tokens, with
// bottom-right causal alignment (query row i sees keys [0, base_pos + i]).
// `Sparse` adds the QSA block selection (design/INFERENCE.md, phase 4): one bit per block of
// `SparseBlock` cells for every query row of the chunk. `Sparse == false` is the dense kernel.
template <typename Geometry, typename Metadata, typename CacheT = __nv_bfloat16,
          bool Sparse = false, int SparseBlock = 4>
__launch_bounds__(kGqaPrefillThreads, 1) __global__
    void gqa_attention_prefill_bf16_kernel(const __nv_bfloat16* __restrict__ q,
                                           const CacheT* __restrict__ cache_k,
                                           const CacheT* __restrict__ cache_v,
                                           Metadata metadata,
                                           const std::int32_t* __restrict__ positions, float scale,
                                           __nv_bfloat16* __restrict__ out, std::int32_t width,
                                           GqaBlockMask block_mask = GqaBlockMask{}) {
    constexpr int D             = Geometry::HeadDim;
    constexpr int Br            = kGqaPrefillBr;      // 64 query rows
    constexpr int Bc            = kGqaPrefillBcFor<Geometry::HeadDim>;      // 64 key cols
    constexpr int Threads       = kGqaPrefillThreads; // 128
    constexpr int QKNt          = Bc / 8;             // 8  QK score n-tiles
    constexpr int QKKs          = D / 16;             // QK contraction steps over head_dim
    constexpr int PVNt          = D / 8;              // PV output n-tiles
    constexpr int PVKs          = Bc / 16;            // 4  PV contraction steps over keys
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(Threads == 128);
    // ldmatrix addresses below are byte offsets into the swizzled tiles, so every
    // row/tile step is a multiple of the row's byte width. Deriving them from D
    // rather than writing 512/4096/8192 is what makes the kernel head-dim generic;
    // the swizzle itself (8 chunks of 8 bf16) needs 8 <= D/8, i.e. D >= 64.
    static_assert(D >= 64 && D % 64 == 0, "head dimension must be a multiple of 64");
    constexpr unsigned RowBytes    = static_cast<unsigned>(D) * 2u;  // 512 at D=256
    constexpr unsigned EightRows   = RowBytes * 8u;                  // 4096 at D=256
    constexpr unsigned SixteenRows = RowBytes * 16u;                 // 8192 at D=256

    extern __shared__ __align__(16) __nv_bfloat16 gqa_smem[];
    __nv_bfloat16* q_s = gqa_smem;     // [Br, D] swizzled
    __nv_bfloat16* k_s = q_s + Br * D; // [Bc, D] swizzled
    __nv_bfloat16* v_s = k_s + Bc * D; // [Bc, D] swizzled

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);

    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        gqa_prefill_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid, Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat     = lane >> 3;
    const int a_rin     = lane & 7;
    const int a_rowoff  = a_rin + ((a_mat & 1) << 3);
    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_row0 = warp * 16; // this warp owns rows [warp_row0, warp_row0+16)

    // Per-lane precomputed swizzled ldmatrix base addresses (see gqa_prefill_swz_addr).
    const unsigned q_sbase = smem_addr(q_s);
    const unsigned k_sbase = smem_addr(k_s);
    const unsigned v_sbase = smem_addr(v_s);
    // Q A-fragment: row = warp_row0 + a_rowoff, col = k*16 + a_coloff.
    const unsigned q_lane_base = q_sbase + static_cast<unsigned>(warp_row0 + a_rowoff) * RowBytes;
    const unsigned q_as        = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned q_r         = static_cast<unsigned>(a_rin << 4);
    // K B-fragment via ldmatrix.x4 (2 n-tiles/instr): lanes 16-31 fetch the +8-key
    // half (one EightRows step), lanes with bit3 set fetch the +8 d-contract half.
    const unsigned k_lane_base =
        k_sbase + static_cast<unsigned>(b_rin) * RowBytes + static_cast<unsigned>(lane >> 4) * EightRows;
    const unsigned k_as = static_cast<unsigned>((b_koff >> 3) << 4);
    const unsigned k_r  = static_cast<unsigned>(b_rin << 4);
    // V B-fragment via ldmatrix.x4.trans (2 n-tiles/instr): row = k*16 + (bit3)*8 + b_rin,
    // col = n*8 + (lane>>4)*8.
    const unsigned v_lane_base = v_sbase + static_cast<unsigned>((lane >> 3) & 1) * EightRows +
                                 static_cast<unsigned>(b_rin) * RowBytes;
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r  = static_cast<unsigned>(b_rin << 4);

    // Stage Q into smem once via cp.async (overlaps with the K(0) prologue load
    // below); it stays resident for the whole key loop. Global Q rows are D bf16
    // contiguous, with a token stride of D*QHeads.
    {
        constexpr int VecPerRow      = D / 8;
        constexpr int QRowStride     = D * Geometry::QHeads; // global stride between tokens
        const __nv_bfloat16* q_block = q + gqa_prefill_q_index<Geometry>(q_head, 0, q0);
        if (q0 + Br <= tokens) {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk / VecPerRow;
                const int d      = (chunk % VecPerRow) * 8;
                __nv_bfloat16* p = &q_s[row * D + gqa_prefill_swz(row, d)];
                cp_async<16, Cache::cg>(p, &q_block[row * QRowStride + d]);
            }
        } else {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk / VecPerRow;
                const int d      = (chunk % VecPerRow) * 8;
                __nv_bfloat16* p = &q_s[row * D + gqa_prefill_swz(row, d)];
                if (q0 + row < tokens) {
                    cp_async<16, Cache::cg>(p, &q_block[row * QRowStride + d]);
                } else {
                    store_vec(p, make_int4(0, 0, 0, 0));
                }
            }
        }
    }

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = block_mask.tile_last_key(base_pos + q0, base_pos + q0 + tile_rows - 1);
    const int n_block_max   = (max_query_abs / Bc) + 1;
    // A window makes the oldest keys invisible to every query in this tile, so
    // the loop need not start at zero. The tile's earliest query sits at
    // `base_pos + q0`, and it admits keys from `base_pos + q0 - window + 1`;
    // anything below that is masked for all Br rows. Starting at that key's
    // block turns the scan from O(context) into O(window) per tile -- the mask
    // was correct without this, just paid for on every key ever written.
    // A sparse chunk carries no window (the block mask is its selection), so it
    // keeps starting at zero.
    const int first_visible_key =
        metadata.window > 0 ? (base_pos + q0) - metadata.window + 1 : 0;
    const int n_block_min = first_visible_key > 0 ? (first_visible_key / Bc) : 0;


    // A key block and a KV page were the same thing while a block was 64 keys; at 16 they are
    // not, and the page is the one the block's first key lives in. The INT8 prompt kernel
    // always indexed it this way.
    int physical_page    = block_table[(n_block_min * Bc) >> kPagedKVPageShift];

    // Prologue: commit Q, then kick off the first key block the loop will read.
    // The loop's wait<0> below drains both.
    sinfer::ops::cp_commit();
    gqa_prefill_stage_kv<Geometry, CacheT>(k_s, cache_k, kv_head, n_block_min * Bc, max_query_abs,
                                           physical_page, tid);
    sinfer::ops::cp_commit();

    for (int kb = n_block_min; kb < n_block_max; ++kb) {
        const int k0                 = kb * Bc;
        const int next_physical_page =
            (kb + 1 < n_block_max) ? block_table[((kb + 1) * Bc) >> kPagedKVPageShift] : physical_page;

        sinfer::ops::cp_wait<0>(); // K(kb) landed (also publishes q_s / prev PV done)
        __syncthreads();

        // Overlap V(kb) load against the QK MMA below.
        gqa_prefill_stage_kv<Geometry, CacheT>(v_s, cache_v, kv_head, k0, max_query_abs, physical_page,
                                       tid);
        sinfer::ops::cp_commit();

        // S = Q Kᵀ for this warp's 16 rows over all Bc keys, in registers.
        // Software-pipelined like cute's gemm: issue the ldmatrix for contraction
        // step k+1 while the m16n8k16 MMAs for step k run, so the LSU (ldmatrix)
        // and tensor pipes overlap instead of stalling on each other.
        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
        // Swizzled ldmatrix addresses via precomputed per-lane bases + immediates.
        unsigned af[2][4];
        unsigned bf[2][QKNt][2];
        {
            ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                        gqa_prefill_swz_addr(q_lane_base, 0u, q_as, q_r));
#pragma unroll
            for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                ldmatrix_x4(bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                            gqa_prefill_swz_addr(
                                k_lane_base + static_cast<unsigned>(nt2) * EightRows, 0u, k_as,
                                k_r));
            }
        }
#pragma unroll
        for (int k = 0; k < QKKs; ++k) {
            const int cur = k & 1;
            const int nxt = cur ^ 1;
            if (k + 1 < QKKs) {
                const unsigned ck = static_cast<unsigned>((k + 1) << 5);
                ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                            gqa_prefill_swz_addr(q_lane_base, ck, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                    ldmatrix_x4(
                        bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0], bf[nxt][nt2 + 1][1],
                        gqa_prefill_swz_addr(k_lane_base + static_cast<unsigned>(nt2) * EightRows,
                                             ck, k_as, k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                         af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
            }
        }

#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { score[nt][i] *= scale; }
        }

        const int row0             = warp_row0 + gid;
        const int row1             = warp_row0 + gid + 8;
        const int qrow0            = q0 + row0;
        const int qrow1            = q0 + row1;
        const int qabs0            = (qrow0 < tokens) ? base_pos + qrow0 : -1;
        const int qabs1            = (qrow1 < tokens) ? base_pos + qrow1 : -1;
        // A sparse chunk never has a "full" tile: every key must pass its query's selection.
        //
        // "Every key in this tile is causally visible" is not "every key in this tile
        // needs no mask". Under a sliding window a fully-causal tile is precisely an
        // OLD-key tile -- the ones furthest below the diagonal are the first to fall
        // out of the window -- so the tile is only mask-free when its oldest key is
        // still within the window of its newest query.
        const bool full_score_tile = !Sparse && (q0 + Br <= tokens) &&
                                     ((k0 + Bc - 1) <= (base_pos + q0)) &&
                                     gqa_within_window(max_query_abs, k0, metadata.window) && block_mask.image_end == 0;
        const std::uint32_t* mask0 =
            Sparse && qrow0 < tokens
                ? block_mask.words + static_cast<std::int64_t>(qrow0) * block_mask.stride
                : nullptr;
        const std::uint32_t* mask1 =
            Sparse && qrow1 < tokens
                ? block_mask.words + static_cast<std::int64_t>(qrow1) * block_mask.stride
                : nullptr;

        // Block row-max on scaled scores, matching decode.
        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                score[nt][0] = (qrow0 < tokens && key0 <= block_mask.last_key(qabs0) && gqa_within_window(qabs0, key0, metadata.window) &&
                                gqa_block_visible<Sparse, SparseBlock>(mask0, key0))
                                   ? score[nt][0]
                                   : -CUDART_INF_F;
                score[nt][1] = (qrow0 < tokens && key1 <= block_mask.last_key(qabs0) && gqa_within_window(qabs0, key1, metadata.window) &&
                                gqa_block_visible<Sparse, SparseBlock>(mask0, key1))
                                   ? score[nt][1]
                                   : -CUDART_INF_F;
                score[nt][2] = (qrow1 < tokens && key0 <= block_mask.last_key(qabs1) && gqa_within_window(qabs1, key0, metadata.window) &&
                                gqa_block_visible<Sparse, SparseBlock>(mask1, key0))
                                   ? score[nt][2]
                                   : -CUDART_INF_F;
                score[nt][3] = (qrow1 < tokens && key1 <= block_mask.last_key(qabs1) && gqa_within_window(qabs1, key1, metadata.window) &&
                                gqa_block_visible<Sparse, SparseBlock>(mask1, key1))
                                   ? score[nt][3]
                                   : -CUDART_INF_F;
                bm0            = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1            = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0        = fmaxf(m0, bm0);
        const float nm1        = fmaxf(m1, bm1);
        // A window can mask a whole tile, leaving both maxima at -inf; the rescale
        // would then be exp2(-inf + inf) = NaN and would poison the row's accumulator
        // for every later tile. The running sums are zero there, so alpha is too.
        // (The decode kernels have carried this guard since they were written.)
        const float alpha0 =
            (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx(((m0 - nm0) * Log2E));
        const float alpha1 =
            (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx(((m1 - nm1) * Log2E));

        // P = exp2(S - m), repacked into the PV A-fragment layout, plus local block row-sum.
        // Reduce each tile before accumulating the denominator, matching decode.
        const float beta0 = (bm0 == -CUDART_INF_F) ? 0.0f : exp2_approx((bm0 - nm0) * Log2E);
        const float beta1 = (bm1 == -CUDART_INF_F) ? 0.0f : exp2_approx((bm1 - nm1) * Log2E);
        float bl0 = 0.0f, bl1 = 0.0f;
        unsigned p_frag[PVKs][4];
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float p00 = exp2_approx(((score[nt][0] - bm0) * Log2E));
                const float p01 = exp2_approx(((score[nt][1] - bm0) * Log2E));
                const float p10 = exp2_approx(((score[nt][2] - bm1) * Log2E));
                const float p11 = exp2_approx(((score[nt][3] - bm1) * Log2E));
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    p_frag[pk][0] = pack_bf16x2(p00, p01);
                    p_frag[pk][1] = pack_bf16x2(p10, p11);
                } else {
                    p_frag[pk][2] = pack_bf16x2(p00, p01);
                    p_frag[pk][3] = pack_bf16x2(p10, p11);
                }
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float p00 = (score[nt][0] > -CUDART_INF_F)
                                      ? exp2_approx(((score[nt][0] - bm0) * Log2E))
                                      : 0.0f;
                const float p01 = (score[nt][1] > -CUDART_INF_F)
                                      ? exp2_approx(((score[nt][1] - bm0) * Log2E))
                                      : 0.0f;
                const float p10 = (score[nt][2] > -CUDART_INF_F)
                                      ? exp2_approx(((score[nt][2] - bm1) * Log2E))
                                      : 0.0f;
                const float p11 = (score[nt][3] > -CUDART_INF_F)
                                      ? exp2_approx(((score[nt][3] - bm1) * Log2E))
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    p_frag[pk][0] = pack_bf16x2(p00, p01);
                    p_frag[pk][1] = pack_bf16x2(p10, p11);
                } else {
                    p_frag[pk][2] = pack_bf16x2(p00, p01);
                    p_frag[pk][3] = pack_bf16x2(p10, p11);
                }
            }
        }

        bl0 = warp_sum<4>(bl0, FullMask);
        bl1 = warp_sum<4>(bl1, FullMask);
        l0 = __fmaf_rn(l0, alpha0, bl0 * beta0);
        l1 = __fmaf_rn(l1, alpha1, bl1 * beta1);
        m0 = nm0;
        m1 = nm1;

        sinfer::ops::cp_wait<0>(); // V(kb) landed; QK done reading k_s
        __syncthreads();

        // Prefetch K(kb+1) into the (now-free) K buffer, overlapping the PV MMA.
        if (kb + 1 < n_block_max) {
            physical_page = next_physical_page;
            gqa_prefill_stage_kv<Geometry, CacheT>(k_s, cache_k, kv_head, (kb + 1) * Bc, max_query_abs,
                                           physical_page, tid);
            sinfer::ops::cp_commit();
        }

        // Form each tile output before rescaling it. P is rounded against
        // this tile's maximum, independent of earlier tiles or split boundaries.
#pragma unroll
        for (int n2 = 0; n2 < PVNt; n2 += 2) {
            float tile_acc[2][4] = {};
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned vf[4];
                const unsigned col = static_cast<unsigned>(n2 << 4);
                ldmatrix_x4_t(vf[0], vf[1], vf[2], vf[3],
                    gqa_prefill_swz_addr(v_lane_base + static_cast<unsigned>(k) * SixteenRows,
                                         col, v_as, v_r));
                mma_bf16(tile_acc[0][0], tile_acc[0][1], tile_acc[0][2], tile_acc[0][3],
                         p_frag[k][0], p_frag[k][1], p_frag[k][2], p_frag[k][3], vf[0], vf[1]);
                mma_bf16(tile_acc[1][0], tile_acc[1][1], tile_acc[1][2], tile_acc[1][3],
                         p_frag[k][0], p_frag[k][1], p_frag[k][2], p_frag[k][3], vf[2], vf[3]);
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    acc[n2 + j][i] = __fmaf_rn(acc[n2 + j][i], i < 2 ? alpha0 : alpha1,
                                               tile_acc[j][i] * (i < 2 ? beta0 : beta1));
                }
            }
        }
    }

    // Use the same final division as decode before rounding the output to BF16.
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0    = n * 8 + 2 * lid;
        const int qrow0 = q0 + warp_row0 + gid;
        const int qrow1 = q0 + warp_row0 + gid + 8;
        if (qrow0 < tokens) {
            *reinterpret_cast<unsigned*>(&out[gqa_prefill_q_index<Geometry>(q_head, d0, qrow0)]) =
                pack_bf16x2(l0 > 0.0f ? acc[n][0] / l0 : 0.0f, l0 > 0.0f ? acc[n][1] / l0 : 0.0f);
        }
        if (qrow1 < tokens) {
            *reinterpret_cast<unsigned*>(&out[gqa_prefill_q_index<Geometry>(q_head, d0, qrow1)]) =
                pack_bf16x2(l1 > 0.0f ? acc[n][2] / l1 : 0.0f, l1 > 0.0f ? acc[n][3] / l1 : 0.0f);
        }
    }
    gqa_prefill_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid, Threads);
}

} // namespace sinfer::ops
