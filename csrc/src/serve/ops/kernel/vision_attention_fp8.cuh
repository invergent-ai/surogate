#pragma once

// FP8 vision attention (--vision-fp8 on sm_12x): S = Q K^T and O += P V on the e4m3 tensor cores
// (m16n8k32, twice the BF16 MMA rate), the rest -- tiles, online softmax, segments -- as the BF16
// kernel in vision_attention.cuh.
//
// Operands, prepared once per call by the kernels at the end of this file:
//   * Q and K: e4m3 codes with one float scale per (token, head), amax / 448. A score is
//     sq_i * sk_j * (q8_i . k8_j), so the scales leave the MMA untouched.
//   * V: e4m3 codes with one float scale per (head, channel), amax over the item's tokens / 448,
//     applied to the output column: O[:, d] = sv_d * sum_j P_ij v8_jd.
//   * P: rounded to e4m3 from the score registers after scaling by 448, so a block's largest
//     probability lands on e4m3's largest finite value; l is summed from the unscaled ones and the
//     448 divided out at the end. The text prompt kernel's FP8 P V does the same
//     (gqa_attention_prefill_bf16.cuh, kGqaPrefillPv8Scale).
// Each operand row is the head's codes zero-padded to a multiple of 16 bytes (80 at D = 72): the
// QK product reads them in 32-byte k steps (96 at D = 72, the tail chunk zero-filled when staged),
// the PV product 16 columns per ldmatrix (80).
//
// P is the m16n8k32 A operand straight from the score registers, which wants keys 4l..4l+3 of each
// 16 in thread l of a quad while a score fragment gives it columns 2l, 2l+1 of each 8-wide n-tile;
// K's rows are therefore staged in a fixed key order per 32 keys (SageAttention2's trick, the sum
// over keys being order-free) -- the text kernel's gqa_prefill_pv8_row / _column_key -- and V's
// stay in key order for ldmatrix .trans .b8 to transpose.

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/vision_attention.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <cstdint>

// ldmatrix .m16n16 .trans .b8 is an sm_12x family instruction; the launcher gates the route on
// the device's compute capability (as gqa_attention_prefill_bf16.cuh does).
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1200 && \
    __CUDA_ARCH_FAMILY_SPECIFIC__ < 1300
#define SINFER_VISION_FP8_LDMATRIX_B8 1
#endif

namespace sinfer::ops {

inline constexpr int kVisionFp8Br      = 64;
inline constexpr int kVisionFp8Bc      = 64;
inline constexpr int kVisionFp8Threads = 128;
inline constexpr int kVisionFp8SmemRow = 128; // bytes per staged row: 8 XOR-swizzled 16-byte chunks
inline constexpr float kVisionFp8PScale = 448.0F;

/// Bytes per prepared operand row in global memory: the head's codes, zero-padded to 16.
template <int D> inline constexpr int kVisionFp8RowBytes = (D + 15) / 16 * 16;
/// 16-byte chunks of a row the QK product's 32-byte k steps read (the last may be padding).
template <int D> inline constexpr int kVisionFp8QkChunks = (D + 31) / 32 * 2;
/// Output columns the PV product computes (whole 16-column ldmatrix loads).
template <int D> inline constexpr int kVisionFp8VColumns = (D + 15) / 16 * 16;

__device__ __forceinline__ int vision_fp8_k_row(int key) {
    return (key & 16) | ((key & 2) << 2) | ((key >> 1) & 6) | (key & 1);
}
__device__ __forceinline__ int vision_fp8_column_key(int column) {
    return (column & 16) | ((column & 6) << 1) | ((column >> 2) & 2) | (column & 1);
}

__device__ __forceinline__ void vision_fp8_ldmatrix_x2_trans_b8(unsigned& b0, unsigned& b1,
                                                                unsigned& b2, unsigned& b3,
                                                                unsigned addr) {
#if !defined(__CUDA_ARCH__) || defined(SINFER_VISION_FP8_LDMATRIX_B8)
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
                 : "r"(addr));
#else
    (void)addr;
    b0 = b1 = b2 = b3 = 0U;
    __trap();
#endif
}

/// Stage `rows` operand rows from `src` (one head's plane) starting at token `row0` into `dst`,
/// `Chunks` 16-byte chunks each, chunk c of staged row r at r * 128 + ((c ^ (r & 7)) << 4).
/// Rows past `end`, and chunks past the global row, land as zeros. `PermuteK` stages key r at
/// row (r & 32) | vision_fp8_k_row(r & 31).
template <int D, int Chunks, bool PermuteK>
__device__ __forceinline__ void vision_fp8_stage(std::uint8_t* dst, const std::uint8_t* src,
                                                 int row0, int end, int rows, int tid) {
    constexpr int RowBytes = kVisionFp8RowBytes<D>;
    constexpr int GlobalChunks = RowBytes / 16;
    for (int item = tid; item < rows * Chunks; item += kVisionFp8Threads) {
        const int r     = item / Chunks;
        const int chunk = item - r * Chunks;
        const int token = row0 + r;
        const bool live = token < end && chunk < GlobalChunks;
        const int srow  = PermuteK ? ((r & 32) | vision_fp8_k_row(r & 31)) : r;
        std::uint8_t* to = dst + srow * kVisionFp8SmemRow + ((chunk ^ (srow & 7)) << 4);
        const std::uint8_t* from =
            src + static_cast<std::int64_t>(live ? token : row0) * RowBytes + (live ? chunk * 16 : 0);
        cp_async_zfill<16, Cache::cg>(to, from, live ? 16 : 0);
    }
}

template <int D>
__launch_bounds__(kVisionFp8Threads, 4) __global__ void vision_attention_fp8_kernel(
    const std::uint8_t* __restrict__ q8, const std::uint8_t* __restrict__ k8,
    const std::uint8_t* __restrict__ v8, const float* __restrict__ sq, const float* __restrict__ sk,
    const float* __restrict__ vamax, const VisionAttentionTile* __restrict__ tiles,
    std::int32_t patches, __nv_bfloat16* __restrict__ out, float scale) {
    constexpr int Br        = kVisionFp8Br;
    constexpr int Bc        = kVisionFp8Bc;
    constexpr int SRow      = kVisionFp8SmemRow;
    constexpr int RowBytes  = kVisionFp8RowBytes<D>;
    constexpr int QkChunks  = kVisionFp8QkChunks<D>;
    constexpr int VChunks   = kVisionFp8VColumns<D> / 16;
    constexpr int Ks8       = QkChunks / 2;
    constexpr int QKNt      = Bc / 8;
    constexpr int PVNt      = kVisionFp8VColumns<D> / 8;
    constexpr int VColumns  = kVisionFp8VColumns<D>;
    constexpr unsigned FullMask = 0xffffffffu;
    static_assert(kVisionAttentionInvSqrtD<D> > 0.0f, "vision_attention_fp8: head dim 64 or 72");
    static_assert(QkChunks <= 8 && VChunks <= 8, "a staged row is eight 16-byte chunks");
    const float ScaleLog2E = (scale > 0 ? scale : kVisionAttentionInvSqrtD<D>) * 1.4426950408889634074f;

    VisionAttentionTile tile;
    if (tiles != nullptr) {
        tile = tiles[blockIdx.x];
    } else {
        tile = {static_cast<std::int32_t>(blockIdx.x) * Br, 0, patches, 0};
    }
    if (tile.q0 < 0) { return; }
    const int head = static_cast<int>(blockIdx.y);
    const std::int64_t plane = static_cast<std::int64_t>(patches) * RowBytes;
    q8 += head * plane;
    k8 += head * plane;
    v8 += head * plane;
    sq += static_cast<std::int64_t>(head) * patches;
    sk += static_cast<std::int64_t>(head) * patches;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    __shared__ __align__(128) std::uint8_t q_s[Br * SRow];
    __shared__ __align__(128) std::uint8_t k_s[Bc * SRow];
    __shared__ __align__(128) std::uint8_t v_s[Bc * SRow];
    __shared__ float sk_s[Bc];

    const int warp_row0 = warp * 16;
    // A (Q) ldmatrix.x4: lanes 0-7 rows 0-7 and 8-15 rows 8-15 of a k step's low 16 bytes, 16-31
    // the same of its high 16 -- byte for byte the BF16 kernel's addressing.
    const int a_mat         = lane >> 3;
    const int q_row         = warp_row0 + (lane & 7) + ((a_mat & 1) << 3);
    const unsigned q_base   = smem_addr(q_s) + static_cast<unsigned>(q_row * SRow);
    const int q_hi          = a_mat >> 1;
    const int q_xor         = q_row & 7;
    // B (K) ldmatrix.x4 per n-tile pair: lanes 0-7 keys 0-7 low, 8-15 keys 0-7 high, 16-23 keys
    // 8-15 low, 24-31 keys 8-15 high.
    const int k_rowl        = (lane & 7) + ((lane >> 4) << 3);
    const int k_hi          = (lane >> 3) & 1;
    const unsigned k_sbase  = smem_addr(k_s);
    const unsigned v_sbase  = smem_addr(v_s);

    vision_fp8_stage<D, QkChunks, false>(q_s, q8, tile.q0, tile.end, Br, tid);
    cp_commit();
    vision_fp8_stage<D, QkChunks, true>(k_s, k8, tile.begin, tile.end, Bc, tid);
    if (tid < Bc) { sk_s[tid] = tile.begin + tid < tile.end ? sk[tile.begin + tid] : 0.0F; }
    cp_commit();

    const int row0     = warp_row0 + gid;
    const int query0   = tile.q0 + row0;
    const int query1   = query0 + 8;
    const float sq0    = query0 < tile.end ? sq[query0] : 0.0F;
    const float sq1    = query1 < tile.end ? sq[query1] : 0.0F;

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f; }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    const int key_blocks = (tile.end - tile.begin + Bc - 1) / Bc;
    for (int kb = 0; kb < key_blocks; ++kb) {
        const int key0 = tile.begin + kb * Bc;
        cp_wait<0>();
        __syncthreads();
        vision_fp8_stage<D, VChunks, false>(v_s, v8, key0, tile.end, Bc, tid);
        cp_commit();

        // S = Q K^T over Ks8 32-byte steps.
        float s[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) { s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f; }
#pragma unroll
        for (int ks = 0; ks < Ks8; ++ks) {
            unsigned a[4];
            ldmatrix_x4(a[0], a[1], a[2], a[3],
                        q_base + (static_cast<unsigned>((2 * ks + q_hi) ^ q_xor) << 4));
#pragma unroll
            for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                const int row = nt2 * 8 + k_rowl;
                unsigned b[4];
                ldmatrix_x4(b[0], b[1], b[2], b[3],
                            k_sbase + static_cast<unsigned>(row * SRow) +
                                (static_cast<unsigned>((2 * ks + k_hi) ^ (row & 7)) << 4));
                mma_fp8_e4m3(s[nt2][0], s[nt2][1], s[nt2][2], s[nt2][3], a[0], a[1], a[2], a[3],
                             b[0], b[1]);
                mma_fp8_e4m3(s[nt2 + 1][0], s[nt2 + 1][1], s[nt2 + 1][2], s[nt2 + 1][3], a[0], a[1],
                             a[2], a[3], b[2], b[3]);
            }
        }

        // Dequantize, mask, and the block's row maxima. Column c holds key
        // (c & 32) | vision_fp8_column_key(c & 31); a column pair is a key pair.
        float block_max0 = -CUDART_INF_F;
        float block_max1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int column = nt * 8 + 2 * lid;
            const int local  = (column & 32) | vision_fp8_column_key(column & 31);
            const float ska  = sk_s[local];
            const float skb  = sk_s[local + 1];
            const bool live_a = key0 + local < tile.end;
            const bool live_b = key0 + local + 1 < tile.end;
            s[nt][0] = query0 < tile.end && live_a ? s[nt][0] * sq0 * ska : -CUDART_INF_F;
            s[nt][1] = query0 < tile.end && live_b ? s[nt][1] * sq0 * skb : -CUDART_INF_F;
            s[nt][2] = query1 < tile.end && live_a ? s[nt][2] * sq1 * ska : -CUDART_INF_F;
            s[nt][3] = query1 < tile.end && live_b ? s[nt][3] * sq1 * skb : -CUDART_INF_F;
            block_max0 = fmaxf(block_max0, fmaxf(s[nt][0], s[nt][1]));
            block_max1 = fmaxf(block_max1, fmaxf(s[nt][2], s[nt][3]));
        }
        block_max0 = warp_max<4>(block_max0, FullMask);
        block_max1 = warp_max<4>(block_max1, FullMask);
        const float next_m0 = fmaxf(m0, block_max0);
        const float next_m1 = fmaxf(m1, block_max1);
        const float m0_l2   = next_m0 * ScaleLog2E;
        const float m1_l2   = next_m1 * ScaleLog2E;
        const float alpha0  = exp2_approx(__fmaf_rn(m0, ScaleLog2E, -m0_l2));
        const float alpha1  = exp2_approx(__fmaf_rn(m1, ScaleLog2E, -m1_l2));

        // P, and P * 448 packed in e4m3 as the k32 A operand of each 32-key half: p8[g][0]/[1] rows
        // gid/gid+8 over n-tiles 4g, 4g+1, p8[g][2]/[3] over 4g+2, 4g+3; a pair's first key low.
        float block_sum0 = 0.0f;
        float block_sum1 = 0.0f;
        unsigned p8[2][4] = {{0U, 0U, 0U, 0U}, {0U, 0U, 0U, 0U}};
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const float p00 = s[nt][0] > -CUDART_INF_F ? exp2_approx(__fmaf_rn(s[nt][0], ScaleLog2E, -m0_l2)) : 0.0f;
            const float p01 = s[nt][1] > -CUDART_INF_F ? exp2_approx(__fmaf_rn(s[nt][1], ScaleLog2E, -m0_l2)) : 0.0f;
            const float p10 = s[nt][2] > -CUDART_INF_F ? exp2_approx(__fmaf_rn(s[nt][2], ScaleLog2E, -m1_l2)) : 0.0f;
            const float p11 = s[nt][3] > -CUDART_INF_F ? exp2_approx(__fmaf_rn(s[nt][3], ScaleLog2E, -m1_l2)) : 0.0f;
            block_sum0 += p00 + p01;
            block_sum1 += p10 + p11;
            constexpr float S = kVisionFp8PScale;
            const unsigned c0 = __nv_cvt_float2_to_fp8x2(make_float2(p00 * S, p01 * S), __NV_SATFINITE, __NV_E4M3);
            const unsigned c1 = __nv_cvt_float2_to_fp8x2(make_float2(p10 * S, p11 * S), __NV_SATFINITE, __NV_E4M3);
            const int g     = nt >> 2;
            const int reg   = ((nt & 3) >> 1) * 2;
            const int shift = (nt & 1) * 16;
            p8[g][reg] |= (c0 & 0xffffU) << shift;
            p8[g][reg + 1] |= (c1 & 0xffffU) << shift;
        }
        l0 = __fmaf_rn(l0, alpha0, block_sum0);
        l1 = __fmaf_rn(l1, alpha1, block_sum1);
        m0 = next_m0;
        m1 = next_m1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        cp_wait<0>();
        __syncthreads();
        if (kb + 1 < key_blocks) {
            vision_fp8_stage<D, QkChunks, true>(k_s, k8, key0 + Bc, tile.end, Bc, tid);
            if (tid < Bc) { sk_s[tid] = key0 + Bc + tid < tile.end ? sk[key0 + Bc + tid] : 0.0F; }
            cp_commit();
        }

        // O += P V per 32-key half: lane l's ldmatrix row is the half's key l.
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const unsigned v_row = v_sbase + static_cast<unsigned>((32 * g + lane) * SRow);
            const unsigned v_xor = static_cast<unsigned>(lane & 7);
#pragma unroll
            for (int n2 = 0; n2 < PVNt; n2 += 2) {
                unsigned b[4];
                vision_fp8_ldmatrix_x2_trans_b8(b[0], b[1], b[2], b[3],
                                                v_row + ((static_cast<unsigned>(n2 >> 1) ^ v_xor) << 4));
                mma_fp8_e4m3(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p8[g][0], p8[g][1],
                             p8[g][2], p8[g][3], b[0], b[2]);
                mma_fp8_e4m3(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3],
                             p8[g][0], p8[g][1], p8[g][2], p8[g][3], b[1], b[3]);
            }
        }
    }

    l0 = warp_sum<4>(l0, FullMask);
    l1 = warp_sum<4>(l1, FullMask);
    // P met V scaled by 448; l was summed from the unscaled probabilities.
    const float inv_l0 = l0 > 0.0f ? __frcp_rn(l0 * kVisionFp8PScale) : 0.0f;
    const float inv_l1 = l1 > 0.0f ? __frcp_rn(l1 * kVisionFp8PScale) : 0.0f;
    const float* column_amax = vamax + static_cast<std::int64_t>(head) * VColumns;
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0 = n * 8 + 2 * lid;
        if (d0 >= D) { continue; }
        const float sv0 = column_amax[d0] * (1.0f / 448.0f);
        const float sv1 = column_amax[d0 + 1] * (1.0f / 448.0f);
        if (query0 < tile.end) {
            const std::int64_t offset = (static_cast<std::int64_t>(query0) * gridDim.y + head) * D + d0;
            store_vec(&out[offset], pack_bf16x2(acc[n][0] * inv_l0 * sv0, acc[n][1] * inv_l0 * sv1));
        }
        if (query1 < tile.end) {
            const std::int64_t offset = (static_cast<std::int64_t>(query1) * gridDim.y + head) * D + d0;
            store_vec(&out[offset], pack_bf16x2(acc[n][2] * inv_l1 * sv0, acc[n][3] * inv_l1 * sv1));
        }
    }
}

// ---- operand preparation ------------------------------------------------------------------
//
// Inputs are BF16 rows, d contiguous, heads `stride_h` and tokens `stride_t` elements apart (the
// tower's packed QKV). Rows are written head-major ([head][token]) so the attention kernel reads
// one head's plane contiguously.

template <int D>
__device__ __forceinline__ void vision_fp8_load_row(const __nv_bfloat16* src, float (&v)[D]) {
#pragma unroll
    for (int c = 0; c < D / 8; ++c) {
        const uint4 packed = reinterpret_cast<const uint4*>(src)[c];
        const auto* pairs  = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float2 f       = __bfloat1622float2(pairs[p]);
            v[c * 8 + 2 * p]     = f.x;
            v[c * 8 + 2 * p + 1] = f.y;
        }
    }
}

/// Q and K: one thread per (token, head, operand), tokens slowest so a warp reads two tokens' rows
/// contiguously; each row becomes RowBytes e4m3 codes (zero tail) and the scale amax / 448.
template <int D>
__global__ void vision_fp8_prepare_qk_kernel(const __nv_bfloat16* __restrict__ q,
                                             const __nv_bfloat16* __restrict__ k,
                                             std::int64_t stride_h, std::int64_t stride_t,
                                             std::int32_t heads, std::int32_t patches,
                                             std::uint8_t* __restrict__ q8, std::uint8_t* __restrict__ k8,
                                             float* __restrict__ sq, float* __restrict__ sk) {
    constexpr int RowBytes = kVisionFp8RowBytes<D>;
    const std::int64_t item = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t rows = static_cast<std::int64_t>(heads) * patches;
    if (item >= 2 * rows) { return; }
    const bool is_k          = item >= rows;
    const std::int64_t index = is_k ? item - rows : item;
    const int token          = static_cast<int>(index / heads);
    const int head           = static_cast<int>(index - static_cast<std::int64_t>(token) * heads);
    float x[D];
    vision_fp8_load_row<D>((is_k ? k : q) + head * stride_h + static_cast<std::int64_t>(token) * stride_t, x);
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < D; ++i) { amax = fmaxf(amax, fabsf(x[i])); }
    const float scale   = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
    const float inverse = 1.0f / scale;
    uint4 codes[RowBytes / 16];
    auto* pairs = reinterpret_cast<__nv_fp8x2_storage_t*>(codes);
#pragma unroll
    for (int p = 0; p < RowBytes / 2; ++p) {
        pairs[p] = 2 * p < D ? __nv_cvt_float2_to_fp8x2(make_float2(x[2 * p] * inverse, x[2 * p + 1] * inverse),
                                                       __NV_SATFINITE, __NV_E4M3)
                             : static_cast<__nv_fp8x2_storage_t>(0);
    }
    const std::int64_t row = static_cast<std::int64_t>(head) * patches + token;
    auto* dst = reinterpret_cast<uint4*>((is_k ? k8 : q8) + row * RowBytes);
#pragma unroll
    for (int c = 0; c < RowBytes / 16; ++c) { dst[c] = codes[c]; }
    (is_k ? sk : sq)[row] = scale;
}

/// V's per-(head, channel) amax: a block per `tokens_per_block` tokens, thread t the t-th run of
/// eight channels of a token's V row (all heads), so each token's row is read contiguously; one
/// atomic per channel per block. `amax` holds non-negative floats as their bits, from zero.
template <int D>
__global__ void vision_fp8_v_amax_kernel(const __nv_bfloat16* __restrict__ v, std::int64_t stride_h,
                                         std::int64_t stride_t, std::int32_t heads,
                                         std::int32_t patches, std::int32_t tokens_per_block,
                                         float* __restrict__ amax) {
    const int t = static_cast<int>(threadIdx.x);
    if (t >= heads * (D / 8)) { return; }
    const int head  = t / (D / 8);
    const int chunk = t - head * (D / 8);
    const int begin = static_cast<int>(blockIdx.x) * tokens_per_block;
    const int end   = min(patches, begin + tokens_per_block);
    const __nv_bfloat16* src = v + head * stride_h + chunk * 8;
    float m[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
    for (int token = begin; token < end; ++token) {
        const uint4 packed = *reinterpret_cast<const uint4*>(src + static_cast<std::int64_t>(token) * stride_t);
        const auto* pairs  = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float2 f = __bfloat1622float2(pairs[p]);
            m[2 * p]       = fmaxf(m[2 * p], fabsf(f.x));
            m[2 * p + 1]   = fmaxf(m[2 * p + 1], fabsf(f.y));
        }
    }
    int* out = reinterpret_cast<int*>(amax) + head * kVisionFp8VColumns<D> + chunk * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) { atomicMax(out + i, __float_as_int(m[i])); }
}

/// V's codes: one thread per (token, head), tokens slowest, each channel divided by its scale
/// amax / 448.
template <int D>
__global__ void vision_fp8_prepare_v_kernel(const __nv_bfloat16* __restrict__ v, std::int64_t stride_h,
                                            std::int64_t stride_t, std::int32_t heads,
                                            std::int32_t patches, const float* __restrict__ amax,
                                            std::uint8_t* __restrict__ v8) {
    constexpr int RowBytes = kVisionFp8RowBytes<D>;
    const std::int64_t index = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (index >= static_cast<std::int64_t>(heads) * patches) { return; }
    const int token = static_cast<int>(index / heads);
    const int head  = static_cast<int>(index - static_cast<std::int64_t>(token) * heads);
    float x[D];
    vision_fp8_load_row<D>(v + head * stride_h + static_cast<std::int64_t>(token) * stride_t, x);
    const float* column = amax + head * kVisionFp8VColumns<D>;
    uint4 codes[RowBytes / 16];
    auto* pairs = reinterpret_cast<__nv_fp8x2_storage_t*>(codes);
#pragma unroll
    for (int p = 0; p < RowBytes / 2; ++p) {
        const int d = 2 * p;
        if (d >= D) {
            pairs[p] = static_cast<__nv_fp8x2_storage_t>(0);
            continue;
        }
        const float a0 = column[d], a1 = column[d + 1];
        pairs[p] = __nv_cvt_float2_to_fp8x2(make_float2(x[d] * (a0 > 0.0f ? 448.0f / a0 : 0.0f),
                                                        x[d + 1] * (a1 > 0.0f ? 448.0f / a1 : 0.0f)),
                                            __NV_SATFINITE, __NV_E4M3);
    }
    auto* dst = reinterpret_cast<uint4*>(v8 + (static_cast<std::int64_t>(head) * patches + token) * RowBytes);
#pragma unroll
    for (int c = 0; c < RowBytes / 16; ++c) { dst[c] = codes[c]; }
}

} // namespace sinfer::ops
