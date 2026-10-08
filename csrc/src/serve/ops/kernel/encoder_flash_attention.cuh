#pragma once

// sinfer::ops — encoder attention over a whole batch of sequences in one launch.
//
// FlashAttention-2 forward, adapted from the prompt kernel (gqa_attention_prefill_bf16.cuh):
// Br = 64 query rows by Bc = 32 keys a tile, four warps that own 16 rows each, m16n8k16 BF16 MMA
// for S = Q Kᵀ and O += P V, online softmax in exp2, the next K tile's cp.async overlapped with
// the PV product. What differs is what an encoder needs: K and V come straight from the
// projections rather than a paged cache, the sequences of a batch lie end to end and none may see
// another, and the mask is bidirectional with a symmetric window -- or causal, for the encoders
// that pool their last token.
//
// Grid (ceil(longest / Br), batch, q_heads): a CTA past its sequence's end returns at once.
//
// Included only by its launcher (encoder_attention.cu). See docs/op-development.md §6.

#include "ops/kernel/gqa_attention_prefill_common.cuh"

#include <math_constants.h>

#include <cstdint>

namespace sinfer::ops {

inline constexpr int kEncoderFlashBr      = 64;
inline constexpr int kEncoderFlashBc      = 32;
inline constexpr int kEncoderFlashThreads = 128;

/// One Q tile and the K and V tiles it streams over, all BF16: 64 KiB at head dim 256.
template <int HeadDim>
inline constexpr int kEncoderFlashSmemBytes =
    (kEncoderFlashBr + 2 * kEncoderFlashBc) * HeadDim * static_cast<int>(sizeof(__nv_bfloat16));

/// Whether query `query` of a `length`-token sequence admits key `key`: the key exists, a
/// positive window holds it within `abs(query - key) < window`, and a causal mask drops it when
/// it comes after the query. Positions are the sequence's own.
__device__ __forceinline__ bool encoder_flash_admits(int query, int key, int length, int window,
                                                     bool causal) {
    return key < length && (!causal || key <= query) &&
           (window <= 0 || (query > key ? query - key : key - query) < window);
}

/// Stages rows [first, first + Rows) of a sequence's head into a swizzled [Rows, D] tile; row r
/// of the sequence is `D` contiguous BF16 at `src + r * stride`. Rows at or past `limit` are
/// zero, so nothing from the next sequence, or past the buffer, reaches the tensor cores.
template <int D, int Rows>
__device__ __forceinline__ void encoder_flash_stage(__nv_bfloat16* dst, const __nv_bfloat16* src,
                                                    std::int64_t stride, int first, int limit,
                                                    int tid) {
    constexpr int VecPerRow = D / 8; // 8 BF16 per 16-byte cp.async
#pragma unroll
    for (int chunk = tid; chunk < Rows * VecPerRow; chunk += kEncoderFlashThreads) {
        const int row    = chunk / VecPerRow;
        const int d      = (chunk % VecPerRow) * 8;
        __nv_bfloat16* p = &dst[row * D + gqa_prefill_swz(row, d)];
        if (first + row < limit) {
            cp_async<16, Cache::cg>(p, &src[static_cast<std::int64_t>(first + row) * stride + d]);
        } else {
            store_vec(p, make_int4(0, 0, 0, 0));
        }
    }
}

/// `segments` is I32 [2, batch] on the device: each sequence's first column, then its length.
/// q and out are [q_heads * D, columns], k and v [kv_heads * D, columns], all contiguous.
template <int D>
__launch_bounds__(kEncoderFlashThreads, 1) __global__
    void encoder_flash_attention_kernel(const __nv_bfloat16* __restrict__ q,
                                        const __nv_bfloat16* __restrict__ k,
                                        const __nv_bfloat16* __restrict__ v,
                                        const std::int32_t* __restrict__ segments,
                                        std::int32_t batch, std::int32_t kv_heads,
                                        std::int32_t window, bool causal, float scale,
                                        __nv_bfloat16* __restrict__ out) {
    constexpr int Br            = kEncoderFlashBr;
    constexpr int Bc            = kEncoderFlashBc;
    constexpr int QKNt          = Bc / 8;  // score n-tiles
    constexpr int QKKs          = D / 16;  // QK contraction steps over the head
    constexpr int PVNt          = D / 8;   // output n-tiles
    constexpr int PVKs          = Bc / 16; // PV contraction steps over the keys
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    static_assert(kEncoderFlashThreads == 128);
    static_assert(D >= 64 && D % 64 == 0, "head dimension must be a multiple of 64");
    constexpr unsigned RowBytes    = static_cast<unsigned>(D) * 2u;
    constexpr unsigned EightRows   = RowBytes * 8u;
    constexpr unsigned SixteenRows = RowBytes * 16u;

    const int sequence = static_cast<int>(blockIdx.y);
    const int length   = segments[batch + sequence];
    const int q0       = static_cast<int>(blockIdx.x) * Br;
    if (q0 >= length) { return; }
    const int column  = segments[sequence];
    const int q_head  = static_cast<int>(blockIdx.z);
    const int q_heads = static_cast<int>(gridDim.z);
    const int kv_head = q_head / (q_heads / kv_heads);

    const std::int64_t q_rows  = static_cast<std::int64_t>(q_heads) * D;
    const std::int64_t kv_rows = static_cast<std::int64_t>(kv_heads) * D;
    const __nv_bfloat16* q_seq = q + column * q_rows + q_head * D;
    const __nv_bfloat16* k_seq = k + column * kv_rows + kv_head * D;
    const __nv_bfloat16* v_seq = v + column * kv_rows + kv_head * D;
    __nv_bfloat16* out_seq     = out + column * q_rows + q_head * D;

    extern __shared__ __align__(16) __nv_bfloat16 encoder_flash_smem[];
    __nv_bfloat16* q_s = encoder_flash_smem; // [Br, D] swizzled
    __nv_bfloat16* k_s = q_s + Br * D;       // [Bc, D] swizzled
    __nv_bfloat16* v_s = k_s + Bc * D;       // [Bc, D] swizzled

    const int tid       = static_cast<int>(threadIdx.x);
    const int warp      = tid >> 5;
    const int lane      = tid & 31;
    const int gid       = lane >> 2;
    const int lid       = lane & 3;
    const int a_mat     = lane >> 3;
    const int a_rin     = lane & 7;
    const int a_rowoff  = a_rin + ((a_mat & 1) << 3);
    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_row0 = warp * 16;

    // Swizzled ldmatrix bases, exactly as the prompt kernel derives them.
    const unsigned q_lane_base =
        smem_addr(q_s) + static_cast<unsigned>(warp_row0 + a_rowoff) * RowBytes;
    const unsigned q_as = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned q_r  = static_cast<unsigned>(a_rin << 4);
    const unsigned k_lane_base = smem_addr(k_s) + static_cast<unsigned>(b_rin) * RowBytes +
                                 static_cast<unsigned>(lane >> 4) * EightRows;
    const unsigned k_as = static_cast<unsigned>((b_koff >> 3) << 4);
    const unsigned k_r  = static_cast<unsigned>(b_rin << 4);
    const unsigned v_lane_base = smem_addr(v_s) +
                                 static_cast<unsigned>((lane >> 3) & 1) * EightRows +
                                 static_cast<unsigned>(b_rin) * RowBytes;
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r  = static_cast<unsigned>(b_rin << 4);

    // The keys any row of this tile admits, in the sequence's own positions.
    const int last_query = min(q0 + Br, length) - 1;
    const int key_begin  = window > 0 ? max(0, q0 - window + 1) : 0;
    const int key_end    = causal ? last_query + 1
                         : window > 0 ? min(length, last_query + window) : length;
    const int n_block_min = key_begin / Bc;
    const int n_block_max = (key_end + Bc - 1) / Bc;

    encoder_flash_stage<D, Br>(q_s, q_seq, q_rows, q0, length, tid);
    sinfer::ops::cp_commit();
    encoder_flash_stage<D, Bc>(k_s, k_seq, kv_rows, n_block_min * Bc, length, tid);
    sinfer::ops::cp_commit();

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;
    const int qrow0 = q0 + warp_row0 + gid;
    const int qrow1 = qrow0 + 8;

    for (int kb = n_block_min; kb < n_block_max; ++kb) {
        const int k0 = kb * Bc;

        sinfer::ops::cp_wait<0>(); // K(kb) landed (and Q, on the first pass)
        __syncthreads();

        // V(kb) loads under the QK product.
        encoder_flash_stage<D, Bc>(v_s, v_seq, kv_rows, k0, length, tid);
        sinfer::ops::cp_commit();

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
        unsigned af[2][4];
        unsigned bf[2][QKNt][2];
        ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                    gqa_prefill_swz_addr(q_lane_base, 0u, q_as, q_r));
#pragma unroll
        for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
            ldmatrix_x4(bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                        gqa_prefill_swz_addr(k_lane_base + static_cast<unsigned>(nt2) * EightRows,
                                             0u, k_as, k_r));
        }
#pragma unroll
        for (int ks = 0; ks < QKKs; ++ks) {
            const int cur = ks & 1;
            const int nxt = cur ^ 1;
            if (ks + 1 < QKKs) {
                const unsigned ck = static_cast<unsigned>((ks + 1) << 5);
                ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                            gqa_prefill_swz_addr(q_lane_base, ck, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                    ldmatrix_x4(bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0],
                                bf[nxt][nt2 + 1][1],
                                gqa_prefill_swz_addr(
                                    k_lane_base + static_cast<unsigned>(nt2) * EightRows, ck, k_as,
                                    k_r));
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

        // A tile needs no mask when all of its rows and keys exist and every pair is admitted.
        const int widest = max(q0 + Br - 1 - k0, k0 + Bc - 1 - q0);
        const bool full_tile = q0 + Br <= length && k0 + Bc <= length &&
                               (!causal || k0 + Bc - 1 <= q0) && (window <= 0 || widest < window);
        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
        if (full_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        } else {
            const bool row0 = qrow0 < length;
            const bool row1 = qrow1 < length;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                if (!(row0 && encoder_flash_admits(qrow0, key0, length, window, causal))) {
                    score[nt][0] = -CUDART_INF_F;
                }
                if (!(row0 && encoder_flash_admits(qrow0, key1, length, window, causal))) {
                    score[nt][1] = -CUDART_INF_F;
                }
                if (!(row1 && encoder_flash_admits(qrow1, key0, length, window, causal))) {
                    score[nt][2] = -CUDART_INF_F;
                }
                if (!(row1 && encoder_flash_admits(qrow1, key1, length, window, causal))) {
                    score[nt][3] = -CUDART_INF_F;
                }
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        // A window or the sequence's end can mask a row's whole tile; its rescale must then be
        // zero rather than exp2(-inf + inf).
        const float nm0    = fmaxf(m0, bm0);
        const float nm1    = fmaxf(m1, bm1);
        const float alpha0 = m0 == -CUDART_INF_F ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
        const float alpha1 = m1 == -CUDART_INF_F ? 0.0f : exp2_approx((m1 - nm1) * Log2E);
        const float beta0  = bm0 == -CUDART_INF_F ? 0.0f : exp2_approx((bm0 - nm0) * Log2E);
        const float beta1  = bm1 == -CUDART_INF_F ? 0.0f : exp2_approx((bm1 - nm1) * Log2E);

        // P = exp2(S - block max), packed into the PV A-fragment layout. A masked score is -inf
        // and comes out exactly 0; a row masked across the whole tile subtracts 0 instead of its
        // -inf maximum, so it yields zeros rather than NaN.
        const float pm0 = bm0 == -CUDART_INF_F ? 0.0f : bm0;
        const float pm1 = bm1 == -CUDART_INF_F ? 0.0f : bm1;
        float bl0 = 0.0f, bl1 = 0.0f;
        unsigned p_frag[PVKs][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const float p00 = exp2_approx((score[nt][0] - pm0) * Log2E);
            const float p01 = exp2_approx((score[nt][1] - pm0) * Log2E);
            const float p10 = exp2_approx((score[nt][2] - pm1) * Log2E);
            const float p11 = exp2_approx((score[nt][3] - pm1) * Log2E);
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
        bl0 = warp_sum<4>(bl0, FullMask);
        bl1 = warp_sum<4>(bl1, FullMask);
        l0  = __fmaf_rn(l0, alpha0, bl0 * beta0);
        l1  = __fmaf_rn(l1, alpha1, bl1 * beta1);
        m0  = nm0;
        m1  = nm1;

        sinfer::ops::cp_wait<0>(); // V(kb) landed; the QK product is done with k_s
        __syncthreads();

        // K(kb + 1) loads under the PV product.
        if (kb + 1 < n_block_max) {
            encoder_flash_stage<D, Bc>(k_s, k_seq, kv_rows, k0 + Bc, length, tid);
            sinfer::ops::cp_commit();
        }

#pragma unroll
        for (int n2 = 0; n2 < PVNt; n2 += 2) {
            float tile_acc[2][4] = {};
#pragma unroll
            for (int ks = 0; ks < PVKs; ++ks) {
                unsigned vf[4];
                const unsigned col = static_cast<unsigned>(n2 << 4);
                ldmatrix_x4_t(vf[0], vf[1], vf[2], vf[3],
                              gqa_prefill_swz_addr(
                                  v_lane_base + static_cast<unsigned>(ks) * SixteenRows, col, v_as,
                                  v_r));
                mma_bf16(tile_acc[0][0], tile_acc[0][1], tile_acc[0][2], tile_acc[0][3],
                         p_frag[ks][0], p_frag[ks][1], p_frag[ks][2], p_frag[ks][3], vf[0], vf[1]);
                mma_bf16(tile_acc[1][0], tile_acc[1][1], tile_acc[1][2], tile_acc[1][3],
                         p_frag[ks][0], p_frag[ks][1], p_frag[ks][2], p_frag[ks][3], vf[2], vf[3]);
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

    // Every row admits at least its own key, so a row inside the sequence has l > 0.
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0 = n * 8 + 2 * lid;
        if (qrow0 < length) {
            *reinterpret_cast<unsigned*>(&out_seq[qrow0 * q_rows + d0]) =
                pack_bf16x2(acc[n][0] / l0, acc[n][1] / l0);
        }
        if (qrow1 < length) {
            *reinterpret_cast<unsigned*>(&out_seq[qrow1 * q_rows + d0]) =
                pack_bf16x2(acc[n][2] / l1, acc[n][3] / l1);
        }
    }
}

} // namespace sinfer::ops
