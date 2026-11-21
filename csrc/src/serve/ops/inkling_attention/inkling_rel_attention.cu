// Inkling-2-Mini relative-bias attention (ENGINE-INKLING lane c): decode / verify and prefill, the split
// merge, and the gate-and-sink epilogue. Semantics: api/ops/inkling_attention.h; numerics and plan:
// lanes/c/DESIGN-attention.md sections 1 (A1-A7), 4 and 5. Reference: vLLM's ops/attention_sm90.py
// `_rel_attention_kernel` (the sm_120 path for prefill AND decode), ops/attention.py
// `_rel_decode_combine_kernel` and `_gate_sink_kernel`.
//
// M2 form (correctness first): ONE kernel body for both routes. A CTA owns MTiles x 16 query rows of one
// KV head and walks one split of their visible keys in 32-key blocks (a block never straddles a 64-token
// page); every warp computes its own 16 rows over the whole block. Decode / verify rows are the 3 heads x W
// columns of one sequence and KV head (row = 3 j + h, the reference's `rows // G = token`); prefill rows
// are 64 consecutive columns of one query head. e4m3 codes land in shared memory with cp.async (two
// slots, a block ahead), are widened exactly to bf16 tiles, and meet bf16 Q and P on the m16n8k16 tensor
// cores. The relative bias of a block within 1,024 positions of a warp's rows is computed on the same
// MMA from r_out and the layer's rel table (one k16 step from zero, as the reference's `tl.dot(r, proj)`),
// rounded to bf16 into a per-warp strip, and gathered per score. Perf stages (fused merge, key-split
// warps for decode, PDL prefetch) follow after the op tests pass (DESIGN section 8).

#include "ops/inkling_attention/inkling_attention_common.h"
#include "ops/inkling_attention/inkling_attention_device.cuh"

#include "core/device.h"
#include "core/pdl.cuh"

#include <math_constants.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace inkling_detail {
namespace {

// Keys per block: 64 for prefill (aligned at key 0 like vLLM's sm_120 prefill kernel, BLOCK_N 64, so each row's
// running maximum -- and so the bf16 rounding of P -- follows the reference block by block), 32 for decode.
inline constexpr int kBnPrefill    = 64;
inline constexpr int kBnDecode     = 32;
inline constexpr int kRowBytes     = Geo::kHeadDim * 2; // a bf16 tile row
inline constexpr int kPrefillRows  = 64;

enum Mode : int { kGated = 0, kOutLse = 1, kPartial = 2 };

struct AttnParams {
    const __nv_bfloat16* q;     // [128, 24, N]
    const __nv_bfloat16* r;     // [16, 24, N]
    const __nv_bfloat16* gate;  // [3072, N] view
    std::int64_t gate_stride;
    const __nv_bfloat16* sink;  // [24, N] view
    std::int64_t sink_stride;
    const __nv_bfloat16* table; // [16, kRelTableRows]
    const std::uint8_t* k_pages;
    const std::uint8_t* v_pages;
    const std::int32_t* block_tables;
    std::int32_t table_stride;
    const std::int32_t* positions;
    const std::int32_t* valid_columns;
    const std::int32_t* table_rows;
    std::int32_t width, batch, splits;
    float qk_scale_log2;
    __nv_bfloat16* out;  // Gated: [3072, N]
    float* out_f32;      // OutLse: [128, 24, N]
    float* lse;          // OutLse: [24, N]
    float* part_o;       // Partial: [128, 24, N, S]
    float* part_lse2;    // Partial: [24, N, S]
    // Numerics variant bits (SUROGATE_INKLING_ATTN_VARIANT; the vLLM bit-identity experiment, DESIGN 1 A3-A5):
    // 1 bias as s + bf16(b) * log2e (else fma), 2|4 P V update 0 = scale acc then chain the MMAs into it,
    // 2 = chain from zero then fma(acc, alpha, pv), 4 = chain from zero then acc * alpha + pv, 8 o = acc / l
    // with div.rn (else div.full), 16 lse with accurate log2f (else lg2.approx), 32 l = l * alpha + sum unfused.
    int variant;
};

template <int MTiles, int Bn>
struct AttnSmem {
    static constexpr int kStripCols   = Bn + 16; // distances a warp's strip covers (row spread <= 15)
    static constexpr int kStripStride = Bn + 24;
    std::uint8_t raw_k[2][Bn * Geo::kHeadDim];
    std::uint8_t raw_v[2][Bn * Geo::kHeadDim];
    __nv_bfloat16 k[Bn * Geo::kHeadDim];
    __nv_bfloat16 v[Bn * Geo::kHeadDim];
    __nv_bfloat16 strip[MTiles][16 * kStripStride];
};

__device__ __forceinline__ unsigned word_or_zero(const __nv_bfloat16* p, bool ok) {
    return ok ? load_vec<unsigned>(p) : 0U;
}

template <bool Local, bool Decode, int MTiles, int Mode, int Bn>
__global__ __launch_bounds__(MTiles * 32) void rel_attention_kernel(const AttnParams p) {
    constexpr int Threads      = MTiles * 32;
    constexpr int NT           = Bn / 8;  // score n-tiles
    constexpr int KK           = Bn / 16; // P V k-steps
    constexpr int kBn          = Bn;
    using Smem                 = AttnSmem<MTiles, Bn>;
    constexpr int kStripCols   = Smem::kStripCols;
    constexpr int kStripStride = Smem::kStripStride;
    extern __shared__ __align__(16) unsigned char attn_smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(attn_smem_raw);
    pdl::wait_for_dependencies();

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    int b = 0, kvh = 0, split = 0, j_base = 0, head_fixed = 0;
    if constexpr (Decode) {
        split = static_cast<int>(blockIdx.x);
        kvh   = static_cast<int>(blockIdx.y);
        b     = static_cast<int>(blockIdx.z);
    } else {
        j_base     = static_cast<int>(blockIdx.x) * kPrefillRows;
        head_fixed = static_cast<int>(blockIdx.y);
        kvh        = head_fixed / Geo::kGroup;
        b          = static_cast<int>(blockIdx.z) / p.splits;
        split      = static_cast<int>(blockIdx.z) % p.splits;
    }
    const int W     = p.width;
    int valid       = W;
    if (p.valid_columns != nullptr) { valid = max(0, min(p.valid_columns[b], W)); }
    const std::int32_t* positions = p.positions + static_cast<std::int64_t>(W) * b;
    const std::int32_t* table_row =
        p.block_tables + static_cast<std::int64_t>(p.table_rows[b]) * p.table_stride;
    const std::int64_t n_columns = static_cast<std::int64_t>(W) * p.batch;

    // Column and query head of a CTA row; a row "exists" while its column is inside the tensors.
    const auto row_column = [&](int rho) { return Decode ? rho / Geo::kGroup : j_base + rho; };
    const auto row_head   = [&](int rho) {
        return Decode ? kvh * Geo::kGroup + rho % Geo::kGroup : head_fixed;
    };
    const auto row_exists = [&](int rho) {
        return Decode ? rho < Geo::kGroup * W : j_base + rho < W;
    };

    // The CTA's valid columns are a prefix-clipped range of consecutive positions.
    const int j_lo  = Decode ? 0 : j_base;
    const int j_hi  = Decode ? valid - 1 : min(j_base + kPrefillRows - 1, valid - 1);
    const bool any  = j_lo <= j_hi;
    const int pmin  = any ? positions[j_lo] : 0;
    const int pmax  = any ? positions[j_hi] : -1;
    const int hi    = pmax + 1; // keys [0, hi) exist in the plane
    int kb_begin = 0, kb_end = 0;
    if (any) {
        const int lo     = Local ? max(0, pmin - (Geo::kWindow - 1)) : 0;
        const int kb_lo  = lo / kBn;
        const int nblk   = (hi + kBn - 1) / kBn - kb_lo;
        kb_begin         = kb_lo + static_cast<int>(static_cast<std::int64_t>(split) * nblk / p.splits);
        kb_end           = kb_lo + static_cast<int>(static_cast<std::int64_t>(split + 1) * nblk / p.splits);
    }

    // This warp's rows.
    const int w_rho0   = warp * 16;
    const int w_j0     = row_column(w_rho0);
    const int w_j1     = min(row_column(w_rho0 + 15), valid - 1);
    const bool w_any   = w_j0 <= w_j1;
    const int pmin_w   = w_any ? positions[w_j0] : 0;
    const int pmax_w   = w_any ? positions[w_j1] : -1;
    const bool w_full  = Decode ? (w_rho0 + 15) < Geo::kGroup * valid : (j_base + w_rho0 + 15) < valid;

    // This thread's two rows (gid, gid + 8).
    int rj[2], rh[2], rpos[2];
    bool rvalid[2], rexists[2];
    std::int64_t rn[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int rho = w_rho0 + gid + 8 * i;
        rj[i]         = row_column(rho);
        rh[i]         = row_head(rho);
        rexists[i]    = row_exists(rho);
        rvalid[i]     = rexists[i] && rj[i] < valid;
        rpos[i]       = rvalid[i] ? positions[rj[i]] : -1;
        rn[i]         = static_cast<std::int64_t>(rj[i]) + static_cast<std::int64_t>(W) * b;
    }

    // Q (all eight k16 steps) and R fragments, held in registers for the whole key walk.
    unsigned af[8][4];
    unsigned ar[4];
    {
        const __nv_bfloat16* q0 = p.q + (rn[0] * Geo::kQHeads + rh[0]) * Geo::kHeadDim + 2 * lid;
        const __nv_bfloat16* q1 = p.q + (rn[1] * Geo::kQHeads + rh[1]) * Geo::kHeadDim + 2 * lid;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            af[ks][0] = word_or_zero(q0 + 16 * ks, rvalid[0]);
            af[ks][1] = word_or_zero(q1 + 16 * ks, rvalid[1]);
            af[ks][2] = word_or_zero(q0 + 16 * ks + 8, rvalid[0]);
            af[ks][3] = word_or_zero(q1 + 16 * ks + 8, rvalid[1]);
        }
        const __nv_bfloat16* r0 = p.r + (rn[0] * Geo::kQHeads + rh[0]) * Geo::kRelDim + 2 * lid;
        const __nv_bfloat16* r1 = p.r + (rn[1] * Geo::kQHeads + rh[1]) * Geo::kRelDim + 2 * lid;
        ar[0] = word_or_zero(r0, rvalid[0]);
        ar[1] = word_or_zero(r1, rvalid[1]);
        ar[2] = word_or_zero(r0 + 8, rvalid[0]);
        ar[3] = word_or_zero(r1 + 8, rvalid[1]);
    }

    float acc[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0F; }
    float m[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float l[2] = {0.0F, 0.0F};

    // Land block kb's K and V codes in raw slot `slot`; keys at or past `hi` are zero-filled (an
    // unwritten tail can hold NaN codes, and P V is an MMA).
    const auto stage = [&](int kb, int slot) {
        const int k0           = kb * kBn;
        const std::int32_t pg  = table_row[k0 >> kPageShift];
        const std::int64_t off = kv_page_offset(pg, kvh, k0 & (kPageTokens - 1));
        const int keys_ok      = hi - k0;
        for (int c = tid; c < 2 * kBn * 8; c += Threads) {
            const int which = c / (kBn * 8);
            const int cc    = c - which * (kBn * 8);
            const int key   = cc >> 3;
            const int part  = cc & 7;
            std::uint8_t* dst = (which ? sm.raw_v[slot] : sm.raw_k[slot]) + key * Geo::kHeadDim + part * 16;
            const std::uint8_t* src = (which ? p.v_pages : p.k_pages) + off + key * Geo::kHeadDim + part * 16;
            cp_async_zfill<16>(dst, src, key < keys_ok ? 16 : 0);
        }
    };
    // Widen raw slot `slot` into the swizzled bf16 tiles (16-byte chunk c of key row k at c ^ (k & 7)).
    const auto widen = [&](int slot) {
        for (int c = tid; c < 2 * kBn * 16; c += Threads) {
            const int which = c / (kBn * 16);
            const int cc    = c - which * (kBn * 16);
            const int key   = cc >> 4;
            const int ch    = cc & 15;
            const std::uint8_t* src = (which ? sm.raw_v[slot] : sm.raw_k[slot]) + key * Geo::kHeadDim + ch * 8;
            __nv_bfloat16* dst = (which ? sm.v : sm.k) + key * Geo::kHeadDim + ((ch ^ (key & 7)) << 3);
            store_vec(dst, e4m3x8_to_bf16x8(load_vec<int2>(src)));
        }
    };

    const unsigned k_base = smem_addr(sm.k);
    const unsigned v_base = smem_addr(sm.v);
    __nv_bfloat16* strip  = sm.strip[warp];

    if (kb_begin < kb_end) { stage(kb_begin, 0); }
    cp_commit();
    if (kb_begin + 1 < kb_end) { stage(kb_begin + 1, 1); }
    cp_commit();

    for (int kb = kb_begin; kb < kb_end; ++kb) {
        const int slot = (kb - kb_begin) & 1;
        cp_wait<1>();
        __syncthreads(); // block kb visible; every warp is done with the previous tiles
        widen(slot);
        __syncthreads(); // tiles ready; the raw slot is free
        if (kb + 2 < kb_end) { stage(kb + 2, slot); }
        cp_commit();

        const int k0 = kb * kBn;
        // Warp-uniform skip: no row of this warp sees any key of the block.
        if (!w_any || k0 > pmax_w || (Local && k0 + kBn - 1 < pmin_w - (Geo::kWindow - 1))) { continue; }

        // S = Q K^T: eight chained k16 steps over the head dimension in natural order, from zero.
        float sc[NT][4];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) { sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.0F; }
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
#pragma unroll
            for (int nt2 = 0; nt2 < NT; nt2 += 2) {
                const int key   = nt2 * 8 + (lane & 7) + ((lane >> 4) << 3);
                const int chunk = 2 * ks + ((lane >> 3) & 1);
                unsigned b0, b1, b2, b3;
                ldmatrix_x4(b0, b1, b2, b3,
                            k_base + static_cast<unsigned>(key * kRowBytes + ((chunk ^ (key & 7)) << 4)));
                mma_bf16(sc[nt2][0], sc[nt2][1], sc[nt2][2], sc[nt2][3], af[ks][0], af[ks][1], af[ks][2],
                         af[ks][3], b0, b1);
                mma_bf16(sc[nt2 + 1][0], sc[nt2 + 1][1], sc[nt2 + 1][2], sc[nt2 + 1][3], af[ks][0],
                         af[ks][1], af[ks][2], af[ks][3], b2, b3);
            }
        }
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) { sc[nt][e] = __fmul_rn(sc[nt][e], p.qk_scale_log2); }
        }

        // Relative bias, for blocks within the extent of some row of this warp.
        const int d0 = pmin_w - (k0 + kBn - 1); // distance of strip column 0
        if (d0 < Geo::kRelExtent) {
            __syncwarp();
#pragma unroll
            for (int ct = 0; ct < kStripCols / 8; ++ct) {
                const __nv_bfloat16* t =
                    p.table + static_cast<std::int64_t>(d0 + 8 * ct + gid + Geo::kRelTablePad) * Geo::kRelDim + 2 * lid;
                float c0 = 0.0F, c1 = 0.0F, c2 = 0.0F, c3 = 0.0F;
                mma_bf16(c0, c1, c2, c3, ar[0], ar[1], ar[2], ar[3], load_vec<unsigned>(t),
                         load_vec<unsigned>(t + 8));
                store_vec(strip + gid * kStripStride + 8 * ct + 2 * lid, pack_bf16x2(c0, c1));
                store_vec(strip + (gid + 8) * kStripStride + 8 * ct + 2 * lid, pack_bf16x2(c2, c3));
            }
            __syncwarp();
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int i = e >> 1;
                    if (!rvalid[i]) { continue; }
                    const int kc = nt * 8 + 2 * lid + (e & 1);
                    const int c  = (rpos[i] - pmin_w) + (kBn - 1) - kc;
                    const float bias = __bfloat162float(strip[(gid + 8 * i) * kStripStride + c]);
                    sc[nt][e] = (p.variant & 1) ? __fadd_rn(sc[nt][e], __fmul_rn(bias, kLog2e))
                                                : fmaf(bias, kLog2e, sc[nt][e]);
                }
            }
        }

        // Causal / window / validity mask, only where some element can be invisible.
        const bool need_mask = !w_full || (k0 + kBn - 1 > pmin_w) ||
                               (Local && k0 <= pmax_w - Geo::kWindow);
        if (need_mask) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int i   = e >> 1;
                    const int key = k0 + nt * 8 + 2 * lid + (e & 1);
                    const bool vis =
                        rvalid[i] && key <= rpos[i] && (!Local || rpos[i] - key < Geo::kWindow);
                    if (!vis) { sc[nt][e] = -CUDART_INF_F; }
                }
            }
        }

        // Online softmax in the log2 domain (A4).
        float bm[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            bm[0] = fmaxf(bm[0], fmaxf(sc[nt][0], sc[nt][1]));
            bm[1] = fmaxf(bm[1], fmaxf(sc[nt][2], sc[nt][3]));
        }
        float alpha[2], msafe[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            bm[i]          = warp_max<4>(bm[i]);
            const float mn = fmaxf(m[i], bm[i]);
            msafe[i]       = mn == -CUDART_INF_F ? 0.0F : mn;
            alpha[i]       = ptx_ex2(__fsub_rn(m[i], msafe[i]));
            m[i]           = mn;
        }
        float bl[2] = {0.0F, 0.0F};
        unsigned pk[KK][4];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            float pv[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                pv[e] = ptx_ex2(__fsub_rn(sc[nt][e], msafe[e >> 1]));
                bl[e >> 1] = __fadd_rn(bl[e >> 1], pv[e]);
            }
            const int kk = nt >> 1;
            if ((nt & 1) == 0) {
                pk[kk][0] = pack_bf16x2(pv[0], pv[1]);
                pk[kk][1] = pack_bf16x2(pv[2], pv[3]);
            } else {
                pk[kk][2] = pack_bf16x2(pv[0], pv[1]);
                pk[kk][3] = pack_bf16x2(pv[2], pv[3]);
            }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float sum = warp_sum<4>(bl[i]);
            l[i] = (p.variant & 32) ? __fadd_rn(__fmul_rn(l[i], alpha[i]), sum) : fmaf(l[i], alpha[i], sum);
        }
        const int pv_mode = (p.variant >> 1) & 3;
        if (pv_mode == 0 && __any_sync(kFullWarpMask, alpha[0] != 1.0F || alpha[1] != 1.0F)) {
#pragma unroll
            for (int n = 0; n < 16; ++n) {
                acc[n][0] = __fmul_rn(acc[n][0], alpha[0]);
                acc[n][1] = __fmul_rn(acc[n][1], alpha[0]);
                acc[n][2] = __fmul_rn(acc[n][2], alpha[1]);
                acc[n][3] = __fmul_rn(acc[n][3], alpha[1]);
            }
        }

        // O += P V (keys in order, KK k16 steps). Mode 0: the MMA chain runs into the rescaled accumulator;
        // modes 1 / 2: it runs from zero and joins the rescaled accumulator after (experiment, see AttnParams).
#pragma unroll
        for (int n2 = 0; n2 < 16; n2 += 2) {
            float t0[4] = {0.0F, 0.0F, 0.0F, 0.0F}, t1[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
            for (int kk = 0; kk < KK; ++kk) {
                const int key   = kk * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
                const int chunk = n2 + (lane >> 4);
                unsigned v0, v1, v2, v3;
                ldmatrix_x4_t(v0, v1, v2, v3,
                              v_base + static_cast<unsigned>(key * kRowBytes + ((chunk ^ (key & 7)) << 4)));
                if (pv_mode == 0) {
                    mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], pk[kk][0], pk[kk][1], pk[kk][2],
                             pk[kk][3], v0, v1);
                    mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3], pk[kk][0],
                             pk[kk][1], pk[kk][2], pk[kk][3], v2, v3);
                } else {
                    mma_bf16(t0[0], t0[1], t0[2], t0[3], pk[kk][0], pk[kk][1], pk[kk][2], pk[kk][3], v0, v1);
                    mma_bf16(t1[0], t1[1], t1[2], t1[3], pk[kk][0], pk[kk][1], pk[kk][2], pk[kk][3], v2, v3);
                }
            }
            if (pv_mode != 0) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float a = alpha[e >> 1];
                    acc[n2][e]     = pv_mode == 1 ? fmaf(acc[n2][e], a, t0[e]) : __fadd_rn(__fmul_rn(acc[n2][e], a), t0[e]);
                    acc[n2 + 1][e] = pv_mode == 1 ? fmaf(acc[n2 + 1][e], a, t1[e])
                                                  : __fadd_rn(__fmul_rn(acc[n2 + 1][e], a), t1[e]);
                }
            }
        }
    }
    cp_wait<0>();

    // Epilogue (A5-A7). Row i of this thread holds acc[n][2 i], acc[n][2 i + 1] at dims 8 n + 2 lid, + 1.
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        if (!rexists[i]) { continue; }
        const std::int64_t n  = rn[i];
        const int h           = rh[i];
        const bool has        = rvalid[i] && l[i] > 0.0F;
        const float lse2      = has ? __fadd_rn(m[i], (p.variant & 16) ? log2f(l[i]) : ptx_lg2(l[i])) : -CUDART_INF_F;
        const auto divide     = [&](float a) { return (p.variant & 8) ? __fdiv_rn(a, l[i]) : ptx_div_full(a, l[i]); };
        if constexpr (Mode == kGated) {
            __nv_bfloat16* dst = p.out + n * Geo::kAttnWidth + h * Geo::kHeadDim + 2 * lid;
            if (!rvalid[i]) {
#pragma unroll
                for (int t = 0; t < 16; ++t) { store_vec(dst + 8 * t, 0U); }
                continue;
            }
            const float lse      = __fmul_rn(lse2, kLn2);
            const float sinkv    = __bfloat162float(p.sink[n * p.sink_stride + h]);
            const float sig_sink = ref_sigmoid(__fsub_rn(lse, sinkv));
            const __nv_bfloat16* g = p.gate + n * p.gate_stride + h * Geo::kHeadDim + 2 * lid;
#pragma unroll
            for (int t = 0; t < 16; ++t) {
                const float o0 = has ? divide(acc[t][2 * i]) : 0.0F;
                const float o1 = has ? divide(acc[t][2 * i + 1]) : 0.0F;
                const float2 gv = bf16x2_bits_to_float2(load_vec<unsigned>(g + 8 * t));
                const float y0  = __fmul_rn(__fmul_rn(o0, ref_sigmoid(gv.x)), sig_sink);
                const float y1  = __fmul_rn(__fmul_rn(o1, ref_sigmoid(gv.y)), sig_sink);
                store_vec(dst + 8 * t, pack_bf16x2(y0, y1));
            }
        } else if constexpr (Mode == kOutLse) {
            float* dst = p.out_f32 + (n * Geo::kQHeads + h) * Geo::kHeadDim + 2 * lid;
#pragma unroll
            for (int t = 0; t < 16; ++t) {
                const float o0 = has ? divide(acc[t][2 * i]) : 0.0F;
                const float o1 = has ? divide(acc[t][2 * i + 1]) : 0.0F;
                store_vec(dst + 8 * t, make_float2(o0, o1));
            }
            // Columns past the valid count come out zero, as the reference zeroes its padding rows.
            if (lid == 0) { p.lse[n * Geo::kQHeads + h] = rvalid[i] ? __fmul_rn(lse2, kLn2) : 0.0F; }
        } else {
            const std::int64_t part = (static_cast<std::int64_t>(split) * n_columns + n) * Geo::kQHeads + h;
            float* dst              = p.part_o + part * Geo::kHeadDim + 2 * lid;
#pragma unroll
            for (int t = 0; t < 16; ++t) {
                const float o0 = has ? divide(acc[t][2 * i]) : 0.0F;
                const float o1 = has ? divide(acc[t][2 * i + 1]) : 0.0F;
                store_vec(dst + 8 * t, make_float2(o0, o1));
            }
            if (lid == 0) { p.part_lse2[part] = lse2; }
        }
    }
}

// ---- decode / verify v2 (rows of W <= 10: 1 or 2 row tiles) -----------------------------------------------------
// A row tile's 16 rows are served by FOUR warps, warp kq owning keys 8 kq .. 8 kq + 7 of every 32-key block with its
// own running max, sum and accumulator (P V on m16n8k8, P straight from the score registers); the four are merged at
// the end as split-KV partials are (FlashDecoding). The next block's e4m3 codes are prefetched into registers
// while the current block is computed, then widened into the swizzled bf16 tiles: no raw ring in shared memory. The
// merge buffer aliases the tiles. Same numerics class as the M2 kernel (P rounded against each warp's own running
// maximum, like any split).
template <int MTiles>
struct DecodeSmem {
    static constexpr int kWarps       = 4 * MTiles;
    static constexpr int kStripStride = 24;  // 16 distances + pad
    static constexpr int kTileBytes   = 2 * 32 * Geo::kHeadDim * 2;         // K + V bf16 tiles
    static constexpr int kMergeBytes  = kWarps * 16 * 64 * 4;              // half a head dim of fp32 acc a warp
    static constexpr int kUnionBytes  = kTileBytes > kMergeBytes ? kTileBytes : kMergeBytes;
    alignas(16) unsigned char u[kUnionBytes];
    __nv_bfloat16 strip[kWarps][16 * kStripStride];
    float mrow[kWarps][16];
    float lrow[kWarps][16];
};

template <bool Local, int MTiles, int Mode>
__global__ __launch_bounds__(MTiles * 128) void rel_decode_kernel(const AttnParams p) {
    constexpr int Threads = MTiles * 128;
    constexpr int NR      = 512 / Threads; // 8-byte code chunks of K (and of V) a thread prefetches per block
    using Smem            = DecodeSmem<MTiles>;
    extern __shared__ __align__(16) unsigned char dec_smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(dec_smem_raw);
    __nv_bfloat16* k_tile = reinterpret_cast<__nv_bfloat16*>(sm.u);
    __nv_bfloat16* v_tile = k_tile + 32 * Geo::kHeadDim;
    float* mbuf           = reinterpret_cast<float*>(sm.u);
    pdl::wait_for_dependencies();

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int mt   = warp >> 2;
    const int kq   = warp & 3;
    const int split = static_cast<int>(blockIdx.x);
    const int kvh   = static_cast<int>(blockIdx.y);
    const int b     = static_cast<int>(blockIdx.z);
    const int W     = p.width;
    int valid       = W;
    if (p.valid_columns != nullptr) { valid = max(0, min(p.valid_columns[b], W)); }
    const std::int32_t* positions = p.positions + static_cast<std::int64_t>(W) * b;
    const std::int32_t* table_row = p.block_tables + static_cast<std::int64_t>(p.table_rows[b]) * p.table_stride;
    const std::int64_t n_columns  = static_cast<std::int64_t>(W) * p.batch;

    // CTA key range (all rows), split into contiguous 32-key blocks.
    const bool any = valid > 0;
    const int pmin = any ? positions[0] : 0;
    const int pmax = any ? positions[valid - 1] : -1;
    const int hi   = pmax + 1;
    int kb_begin = 0, kb_end = 0;
    if (any) {
        const int lo    = Local ? max(0, pmin - (Geo::kWindow - 1)) : 0;
        const int kb_lo = lo / 32;
        const int nblk  = (hi + 31) / 32 - kb_lo;
        kb_begin        = kb_lo + static_cast<int>(static_cast<std::int64_t>(split) * nblk / p.splits);
        kb_end          = kb_lo + static_cast<int>(static_cast<std::int64_t>(split + 1) * nblk / p.splits);
    }

    // This warp's row tile (rows 16 mt .. 16 mt + 15; row = 3 j + h).
    const int w_rho0  = mt * 16;
    const int w_j0    = w_rho0 / Geo::kGroup;
    const int w_j1    = min((w_rho0 + 15) / Geo::kGroup, valid - 1);
    const bool w_any  = w_j0 <= w_j1;
    const int pmin_w  = w_any ? positions[w_j0] : 0;
    const int pmax_w  = w_any ? positions[w_j1] : -1;
    const bool w_full = (w_rho0 + 15) < Geo::kGroup * valid;

    int rj[2], rh[2], rpos[2];
    bool rvalid[2], rexists[2];
    std::int64_t rn[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int rho = w_rho0 + gid + 8 * i;
        rj[i]         = rho / Geo::kGroup;
        rh[i]         = kvh * Geo::kGroup + rho % Geo::kGroup;
        rexists[i]    = rho < Geo::kGroup * W;
        rvalid[i]     = rexists[i] && rj[i] < valid;
        rpos[i]       = rvalid[i] ? positions[rj[i]] : -1;
        rn[i]         = static_cast<std::int64_t>(rj[i]) + static_cast<std::int64_t>(W) * b;
    }
    unsigned af[8][4];
    unsigned ar[4];
    {
        const __nv_bfloat16* q0 = p.q + (rn[0] * Geo::kQHeads + rh[0]) * Geo::kHeadDim + 2 * lid;
        const __nv_bfloat16* q1 = p.q + (rn[1] * Geo::kQHeads + rh[1]) * Geo::kHeadDim + 2 * lid;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            af[ks][0] = word_or_zero(q0 + 16 * ks, rvalid[0]);
            af[ks][1] = word_or_zero(q1 + 16 * ks, rvalid[1]);
            af[ks][2] = word_or_zero(q0 + 16 * ks + 8, rvalid[0]);
            af[ks][3] = word_or_zero(q1 + 16 * ks + 8, rvalid[1]);
        }
        const __nv_bfloat16* r0 = p.r + (rn[0] * Geo::kQHeads + rh[0]) * Geo::kRelDim + 2 * lid;
        const __nv_bfloat16* r1 = p.r + (rn[1] * Geo::kQHeads + rh[1]) * Geo::kRelDim + 2 * lid;
        ar[0] = word_or_zero(r0, rvalid[0]);
        ar[1] = word_or_zero(r1, rvalid[1]);
        ar[2] = word_or_zero(r0 + 8, rvalid[0]);
        ar[3] = word_or_zero(r1 + 8, rvalid[1]);
    }

    float acc[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0F; }
    float m[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float l[2] = {0.0F, 0.0F};

    int2 rk[NR], rv[NR];
    const auto load_raw = [&](int kb) {
        const int k0          = kb * 32;
        const std::int64_t off = kv_page_offset(table_row[k0 >> kPageShift], kvh, k0 & (kPageTokens - 1));
        const int keys_ok     = hi - k0;
#pragma unroll
        for (int i = 0; i < NR; ++i) {
            const int c   = tid + i * Threads;
            const int key = c >> 4;
            const std::int64_t o = off + key * Geo::kHeadDim + (c & 15) * 8;
            const bool ok = key < keys_ok;
            rk[i] = ok ? load_ldg<int2>(p.k_pages + o) : make_int2(0, 0);
            rv[i] = ok ? load_ldg<int2>(p.v_pages + o) : make_int2(0, 0);
        }
    };
    const unsigned k_base = smem_addr(k_tile);
    const unsigned v_base = smem_addr(v_tile);
    __nv_bfloat16* strip  = sm.strip[warp];
    constexpr int kSS     = Smem::kStripStride;

    if (kb_begin < kb_end) { load_raw(kb_begin); }
    for (int kb = kb_begin; kb < kb_end; ++kb) {
        __syncthreads(); // every warp is done with the previous block's tiles
#pragma unroll
        for (int i = 0; i < NR; ++i) {
            const int c = tid + i * Threads, key = c >> 4, part = c & 15;
            const int e = key * Geo::kHeadDim + ((part ^ (key & 7)) << 3);
            store_vec(k_tile + e, e4m3x8_to_bf16x8(rk[i]));
            store_vec(v_tile + e, e4m3x8_to_bf16x8(rv[i]));
        }
        __syncthreads();
        if (kb + 1 < kb_end) { load_raw(kb + 1); } // in flight during this block's math

        const int kw0 = kb * 32 + 8 * kq; // this warp's first key
        if (!w_any || kw0 > pmax_w || (Local && kw0 + 7 < pmin_w - (Geo::kWindow - 1))) { continue; }

        // S for 8 keys: eight chained k16 steps (two per ldmatrix.x4).
        float sc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
        for (int ks2 = 0; ks2 < 4; ++ks2) {
            const int key   = 8 * kq + (lane & 7);
            const int chunk = 4 * ks2 + (lane >> 3);
            unsigned b0, b1, b2, b3;
            ldmatrix_x4(b0, b1, b2, b3, k_base + static_cast<unsigned>(key * kRowBytes + ((chunk ^ (key & 7)) << 4)));
            mma_bf16(sc[0], sc[1], sc[2], sc[3], af[2 * ks2][0], af[2 * ks2][1], af[2 * ks2][2], af[2 * ks2][3], b0, b1);
            mma_bf16(sc[0], sc[1], sc[2], sc[3], af[2 * ks2 + 1][0], af[2 * ks2 + 1][1], af[2 * ks2 + 1][2],
                     af[2 * ks2 + 1][3], b2, b3);
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) { sc[e] = __fmul_rn(sc[e], p.qk_scale_log2); }

        const int d0 = pmin_w - (kw0 + 7); // distance of strip column 0
        if (d0 < Geo::kRelExtent) {
            __syncwarp();
#pragma unroll
            for (int ct = 0; ct < 2; ++ct) {
                const __nv_bfloat16* t =
                    p.table + static_cast<std::int64_t>(d0 + 8 * ct + gid + Geo::kRelTablePad) * Geo::kRelDim + 2 * lid;
                float c0 = 0.0F, c1 = 0.0F, c2 = 0.0F, c3 = 0.0F;
                mma_bf16(c0, c1, c2, c3, ar[0], ar[1], ar[2], ar[3], load_vec<unsigned>(t), load_vec<unsigned>(t + 8));
                store_vec(strip + gid * kSS + 8 * ct + 2 * lid, pack_bf16x2(c0, c1));
                store_vec(strip + (gid + 8) * kSS + 8 * ct + 2 * lid, pack_bf16x2(c2, c3));
            }
            __syncwarp();
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = e >> 1;
                if (!rvalid[i]) { continue; }
                const int c      = (rpos[i] - pmin_w) + 7 - (2 * lid + (e & 1));
                const float bias = __bfloat162float(strip[(gid + 8 * i) * kSS + c]);
                sc[e] = (p.variant & 1) ? __fadd_rn(sc[e], __fmul_rn(bias, kLog2e)) : fmaf(bias, kLog2e, sc[e]);
            }
        }
        if (!w_full || (kw0 + 7 > pmin_w) || (Local && kw0 <= pmax_w - Geo::kWindow)) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i   = e >> 1;
                const int key = kw0 + 2 * lid + (e & 1);
                if (!(rvalid[i] && key <= rpos[i] && (!Local || rpos[i] - key < Geo::kWindow))) { sc[e] = -CUDART_INF_F; }
            }
        }
        float alpha[2], msafe[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float bm = warp_max<4>(fmaxf(sc[2 * i], sc[2 * i + 1]));
            const float mn = fmaxf(m[i], bm);
            msafe[i]       = mn == -CUDART_INF_F ? 0.0F : mn;
            alpha[i]       = ptx_ex2(__fsub_rn(m[i], msafe[i]));
            m[i]           = mn;
        }
        float pv[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) { pv[e] = ptx_ex2(__fsub_rn(sc[e], msafe[e >> 1])); }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float sum = warp_sum<4>(__fadd_rn(pv[2 * i], pv[2 * i + 1]));
            l[i] = (p.variant & 32) ? __fadd_rn(__fmul_rn(l[i], alpha[i]), sum) : fmaf(l[i], alpha[i], sum);
        }
        if (__any_sync(kFullWarpMask, alpha[0] != 1.0F || alpha[1] != 1.0F)) {
#pragma unroll
            for (int n = 0; n < 16; ++n) {
                acc[n][0] = __fmul_rn(acc[n][0], alpha[0]);
                acc[n][1] = __fmul_rn(acc[n][1], alpha[0]);
                acc[n][2] = __fmul_rn(acc[n][2], alpha[1]);
                acc[n][3] = __fmul_rn(acc[n][3], alpha[1]);
            }
        }
        const unsigned pa0 = pack_bf16x2(pv[0], pv[1]), pa1 = pack_bf16x2(pv[2], pv[3]);
#pragma unroll
        for (int n4 = 0; n4 < 4; ++n4) {
            const int key   = 8 * kq + (lane & 7);
            const int chunk = 4 * n4 + (lane >> 3);
            unsigned v0, v1, v2, v3;
            ldmatrix_x4_t(v0, v1, v2, v3, v_base + static_cast<unsigned>(key * kRowBytes + ((chunk ^ (key & 7)) << 4)));
            mma_bf16_k8(acc[4 * n4][0], acc[4 * n4][1], acc[4 * n4][2], acc[4 * n4][3], pa0, pa1, v0);
            mma_bf16_k8(acc[4 * n4 + 1][0], acc[4 * n4 + 1][1], acc[4 * n4 + 1][2], acc[4 * n4 + 1][3], pa0, pa1, v1);
            mma_bf16_k8(acc[4 * n4 + 2][0], acc[4 * n4 + 2][1], acc[4 * n4 + 2][2], acc[4 * n4 + 2][3], pa0, pa1, v2);
            mma_bf16_k8(acc[4 * n4 + 3][0], acc[4 * n4 + 3][1], acc[4 * n4 + 3][2], acc[4 * n4 + 3][3], pa0, pa1, v3);
        }
    }

    // Merge the four key-quarter warps of each row tile, half a head dim at a time; warp kq then finishes dims
    // 64 h + 16 kq .. + 15 of its tile's 16 rows (lane: row lane / 2, dims 8 (lane & 1) .. + 7).
    if (lid == 0) {
        sm.mrow[warp][gid]     = m[0];
        sm.mrow[warp][gid + 8] = m[1];
        sm.lrow[warp][gid]     = l[0];
        sm.lrow[warp][gid + 8] = l[1];
    }
    const int r      = lane >> 1;             // merge row within the tile
    const int rho    = w_rho0 + r;
    const int mj     = rho / Geo::kGroup;
    const int mh     = kvh * Geo::kGroup + rho % Geo::kGroup;
    const bool mex   = rho < Geo::kGroup * W;
    const bool mval  = mex && mj < valid;
    const std::int64_t mn_col = static_cast<std::int64_t>(mj) + static_cast<std::int64_t>(W) * b;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        __syncthreads(); // tiles (h = 0) / the previous half's buffer (h = 1) are free
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int n = 8 * h + t, d = 8 * t + 2 * lid;
            float* row0 = mbuf + (static_cast<std::int64_t>(warp) * 16 + gid) * 64 + d;
            float* row1 = mbuf + (static_cast<std::int64_t>(warp) * 16 + gid + 8) * 64 + d;
            store_vec(row0, make_float2(acc[n][0], acc[n][1]));
            store_vec(row1, make_float2(acc[n][2], acc[n][3]));
        }
        __syncthreads();
        // The four partials of this row (warps 4 mt .. 4 mt + 3).
        float mstar = -CUDART_INF_F;
#pragma unroll
        for (int w = 0; w < 4; ++w) { mstar = fmaxf(mstar, sm.mrow[4 * mt + w][r]); }
        const float ms = mstar == -CUDART_INF_F ? 0.0F : mstar;
        float f[4], L = 0.0F;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            f[w] = ptx_ex2(__fsub_rn(sm.mrow[4 * mt + w][r], ms)); // -inf -> 0
            L    = fmaf(f[w], sm.lrow[4 * mt + w][r], L);
        }
        const bool has   = mval && L > 0.0F;
        const float lse2 = has ? __fadd_rn(ms, (p.variant & 16) ? log2f(L) : ptx_lg2(L)) : -CUDART_INF_F;
        const int dloc   = 16 * kq + 8 * (lane & 1);            // within the half
        const int dim    = 64 * h + dloc;
        float o[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            float a = 0.0F;
#pragma unroll
            for (int w = 0; w < 4; ++w) { a = fmaf(f[w], mbuf[(static_cast<std::int64_t>(4 * mt + w) * 16 + r) * 64 + dloc + c], a); }
            o[c] = has ? ((p.variant & 8) ? __fdiv_rn(a, L) : ptx_div_full(a, L)) : 0.0F;
        }
        if (!mex) { continue; }
        if constexpr (Mode == kGated) {
            __nv_bfloat16* dst = p.out + mn_col * Geo::kAttnWidth + mh * Geo::kHeadDim + dim;
            if (!mval) {
                store_vec(dst, make_uint4(0, 0, 0, 0));
                continue;
            }
            const float sig_sink =
                ref_sigmoid(__fsub_rn(__fmul_rn(lse2, kLn2), __bfloat162float(p.sink[mn_col * p.sink_stride + mh])));
            const __nv_bfloat16* g = p.gate + mn_col * p.gate_stride + mh * Geo::kHeadDim + dim;
            unsigned packed[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float2 gv = bf16x2_bits_to_float2(load_vec<unsigned>(g + 2 * c));
                packed[c]       = pack_bf16x2(__fmul_rn(__fmul_rn(o[2 * c], ref_sigmoid(gv.x)), sig_sink),
                                              __fmul_rn(__fmul_rn(o[2 * c + 1], ref_sigmoid(gv.y)), sig_sink));
            }
            store_vec(dst, make_uint4(packed[0], packed[1], packed[2], packed[3]));
        } else if constexpr (Mode == kOutLse) {
            float* dst = p.out_f32 + (mn_col * Geo::kQHeads + mh) * Geo::kHeadDim + dim;
            store_vec(dst, make_float4(o[0], o[1], o[2], o[3]));
            store_vec(dst + 4, make_float4(o[4], o[5], o[6], o[7]));
            if (h == 0 && kq == 0 && (lane & 1) == 0) { p.lse[mn_col * Geo::kQHeads + mh] = mval ? __fmul_rn(lse2, kLn2) : 0.0F; }
        } else {
            const std::int64_t part = (static_cast<std::int64_t>(split) * n_columns + mn_col) * Geo::kQHeads + mh;
            float* dst              = p.part_o + part * Geo::kHeadDim + dim;
            store_vec(dst, make_float4(o[0], o[1], o[2], o[3]));
            store_vec(dst + 4, make_float4(o[4], o[5], o[6], o[7]));
            if (h == 0 && kq == 0 && (lane & 1) == 0) { p.part_lse2[part] = lse2; }
        }
    }
}

// The split merge of `_rel_decode_combine_kernel` (A6), then the gate and sink (A7) or out / lse.
template <int Mode>
__global__ __launch_bounds__(128) void combine_kernel(const AttnParams p) {
    const std::int64_t n         = blockIdx.x;
    const int h                  = static_cast<int>(blockIdx.y);
    const int d                  = static_cast<int>(threadIdx.x);
    const int W                  = p.width;
    const int b                  = static_cast<int>(n / W);
    const int j                  = static_cast<int>(n % W);
    const std::int64_t n_columns = static_cast<std::int64_t>(W) * p.batch;
    int valid                    = W;
    if (p.valid_columns != nullptr) { valid = max(0, min(p.valid_columns[b], W)); }
    const bool ok = j < valid;

    float lmax = -CUDART_INF_F;
    for (int s = 0; s < p.splits; ++s) {
        lmax = fmaxf(lmax, p.part_lse2[(s * n_columns + n) * Geo::kQHeads + h]);
    }
    const float lm = lmax == -CUDART_INF_F ? 0.0F : lmax;
    float wsum     = 0.0F;
    for (int s = 0; s < p.splits; ++s) {
        wsum = __fadd_rn(wsum, ptx_ex2_ftz(__fsub_rn(p.part_lse2[(s * n_columns + n) * Geo::kQHeads + h], lm)));
    }
    const float denom = wsum == 0.0F ? 1.0F : wsum;
    float o           = 0.0F;
    for (int s = 0; s < p.splits; ++s) {
        const std::int64_t part = (s * n_columns + n) * Geo::kQHeads + h;
        const float w           = ptx_ex2_ftz(__fsub_rn(p.part_lse2[part], lm));
        if (w > 0.0F) { o = __fadd_rn(o, __fmul_rn(p.part_o[part * Geo::kHeadDim + d], ptx_div_full(w, denom))); }
    }
    const float lse = __fmul_rn(__fadd_rn(ptx_lg2_ftz(wsum), lm), kLn2);
    if constexpr (Mode == kGated) {
        float y = 0.0F;
        if (ok) {
            const float g    = __bfloat162float(p.gate[n * p.gate_stride + h * Geo::kHeadDim + d]);
            const float sink = __bfloat162float(p.sink[n * p.sink_stride + h]);
            y = __fmul_rn(__fmul_rn(o, ref_sigmoid(g)), ref_sigmoid(__fsub_rn(lse, sink)));
        }
        p.out[n * Geo::kAttnWidth + h * Geo::kHeadDim + d] = __float2bfloat16_rn(y);
    } else {
        p.out_f32[(n * Geo::kQHeads + h) * Geo::kHeadDim + d] = ok ? o : 0.0F;
        if (d == 0) { p.lse[n * Geo::kQHeads + h] = ok ? lse : 0.0F; }
    }
}

// The reference's `_gate_sink_kernel` (fusion off): bf16((x * sigmoid(gate)) * sigmoid(lse - sink)).
__global__ void gate_sink_kernel(const float* out, const float* lse, const __nv_bfloat16* gate,
                                 std::int64_t gate_stride, const __nv_bfloat16* sink,
                                 std::int64_t sink_stride, __nv_bfloat16* result, std::int64_t columns) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= columns * Geo::kAttnWidth) { return; }
    const std::int64_t n = i / Geo::kAttnWidth;
    const int col        = static_cast<int>(i % Geo::kAttnWidth);
    const int h          = col / Geo::kHeadDim;
    const float x        = out[i];
    const float g        = __bfloat162float(gate[n * gate_stride + col]);
    const float s        = __bfloat162float(sink[n * sink_stride + h]);
    const float y        = __fmul_rn(__fmul_rn(x, ref_sigmoid(g)), ref_sigmoid(__fsub_rn(lse[n * Geo::kQHeads + h], s)));
    result[i]            = __float2bfloat16_rn(y);
}

// ---- planning ---------------------------------------------------------------------------------------

struct Plan {
    bool decode    = true;
    int mtiles     = 1;
    int splits     = 1;
    dim3 grid;
};

int device_sms() {
    static const int sms = [] {
        int device = 0, count = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
            return 170;
        }
        return count > 0 ? count : 170;
    }();
    return sms;
}

// SUROGATE_INKLING_ATTN_SPLITS=N forces N splits (tests and measurement only; read per call so a test
// can compare split counts in one process).
int forced_splits() {
    const char* raw = std::getenv("SUROGATE_INKLING_ATTN_SPLITS");
    return raw == nullptr ? 0 : std::max(0, std::atoi(raw));
}

Plan make_plan(InklingLayerKind kind, std::int32_t width, std::int32_t batch, std::int32_t max_keys) {
    Plan plan;
    const int sms = device_sms();
    if (width <= Geo::kMaxSmallColumns) {
        // Decode / verify: the split count depends on (B, W, kind) only, never on the context length, so
        // a captured graph replays at any length (the reference's plan_decode_attention rule).
        plan.decode = true;
        plan.mtiles = (Geo::kGroup * width + 15) / 16;
        const int work = Geo::kKvHeads * batch;
        const bool v2  = plan.mtiles <= 2 && !decode_v1_forced();
        plan.splits    = kind == InklingLayerKind::Global ? std::clamp((2 * sms + work - 1) / work, 1, 64)
                         : v2 ? std::clamp((2 * sms + work - 1) / work, 1, 17)
                              : std::clamp((sms + work - 1) / work, 1, 9);
    } else {
        plan.decode      = false;
        plan.mtiles      = 4;
        const int tiles  = (width + kPrefillRows - 1) / kPrefillRows;
        const long ctas  = static_cast<long>(tiles) * Geo::kQHeads * batch;
        if (ctas >= 2L * sms) {
            plan.splits = 1;
        } else {
            const int by_keys = std::max(1, (max_keys + 1023) / 1024);
            plan.splits = std::clamp(static_cast<int>((2L * sms + ctas - 1) / ctas), 1, std::min(32, by_keys));
        }
    }
    if (forced_splits() > 0) { plan.splits = std::min(forced_splits(), 64); }
    plan.grid = plan.decode ? dim3(static_cast<unsigned>(plan.splits), Geo::kKvHeads, static_cast<unsigned>(batch))
                            : dim3(static_cast<unsigned>((width + kPrefillRows - 1) / kPrefillRows), Geo::kQHeads,
                                   static_cast<unsigned>(batch * plan.splits));
    return plan;
}

std::size_t partial_bytes(const Plan& plan, std::int64_t columns) {
    if (plan.splits <= 1) { return 0; }
    const std::int64_t rows = columns * Geo::kQHeads * plan.splits;
    return static_cast<std::size_t>(rows * Geo::kHeadDim * 4) + 256 + static_cast<std::size_t>(rows * 4) + 256;
}

template <bool Local, bool Decode, int MTiles, int Mode>
void launch_kernel(const Plan& plan, const AttnParams& p, cudaStream_t stream) {
    constexpr int Bn    = Decode ? kBnDecode : kBnPrefill;
    constexpr int bytes = static_cast<int>(sizeof(AttnSmem<MTiles, Bn>));
    auto* kernel        = rel_attention_kernel<Local, Decode, MTiles, Mode, Bn>;
    static const bool configured = [&] {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        return true;
    }();
    (void)configured;
    const pdl::LaunchConfig launch{plan.grid, dim3(MTiles * 32), static_cast<std::size_t>(bytes), stream};
    CUDA_CHECK(pdl::launch_dependent(launch, kernel, p));
}

bool decode_v1_forced() {
    const char* raw = std::getenv("SUROGATE_INKLING_DECODE_V1"); // A/B against the M2 decode kernel
    return raw != nullptr && raw[0] == '1';
}

template <bool Local, int MTiles, int Mode>
void launch_decode_v2(const Plan& plan, const AttnParams& p, cudaStream_t stream) {
    constexpr int bytes = static_cast<int>(sizeof(DecodeSmem<MTiles>));
    auto* kernel        = rel_decode_kernel<Local, MTiles, Mode>;
    static const bool configured = [&] {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        return true;
    }();
    (void)configured;
    const pdl::LaunchConfig launch{plan.grid, dim3(MTiles * 128), static_cast<std::size_t>(bytes), stream};
    CUDA_CHECK(pdl::launch_dependent(launch, kernel, p));
}

template <bool Local, int Mode>
void dispatch_mode(const Plan& plan, const AttnParams& p, cudaStream_t stream) {
    if (!plan.decode) { return launch_kernel<Local, false, 4, Mode>(plan, p, stream); }
    if (plan.mtiles <= 2 && !decode_v1_forced()) {
        return plan.mtiles == 1 ? launch_decode_v2<Local, 1, Mode>(plan, p, stream)
                                : launch_decode_v2<Local, 2, Mode>(plan, p, stream);
    }
    switch (plan.mtiles) {
    case 1: return launch_kernel<Local, true, 1, Mode>(plan, p, stream);
    case 2: return launch_kernel<Local, true, 2, Mode>(plan, p, stream);
    case 3: return launch_kernel<Local, true, 3, Mode>(plan, p, stream);
    default: throw std::logic_error("inkling rel attention: unplanned row tiles");
    }
}

void dispatch(InklingLayerKind kind, int mode, const Plan& plan, const AttnParams& p, cudaStream_t stream) {
    const bool local = kind == InklingLayerKind::Local;
    switch (mode) {
    case kGated: return local ? dispatch_mode<true, kGated>(plan, p, stream) : dispatch_mode<false, kGated>(plan, p, stream);
    case kOutLse: return local ? dispatch_mode<true, kOutLse>(plan, p, stream) : dispatch_mode<false, kOutLse>(plan, p, stream);
    default: return local ? dispatch_mode<true, kPartial>(plan, p, stream) : dispatch_mode<false, kPartial>(plan, p, stream);
    }
}

AttnParams make_params(const InklingAttentionInputs& in, const InklingAttentionRows& rows,
                       const PagedKVBatchLayerView& cache, const RowsShape& rs, bool need_gate,
                       const char* op) {
    AttnParams p{};
    const std::int64_t n = rs.columns;
    require_contiguous(in.q, DType::BF16, n * Geo::kQHeads * Geo::kHeadDim, op, "q");
    require_contiguous(in.r, DType::BF16, n * Geo::kQHeads * Geo::kRelDim, op, "r");
    require_contiguous(in.rel_table, DType::BF16, Geo::kRelDim * Geo::kRelTableRows, op, "rel_table");
    p.q     = static_cast<const __nv_bfloat16*>(in.q.data);
    p.r     = static_cast<const __nv_bfloat16*>(in.r.data);
    p.table = static_cast<const __nv_bfloat16*>(in.rel_table.data);
    if (need_gate) {
        p.gate        = static_cast<const __nv_bfloat16*>(in.gate.data);
        p.gate_stride = column_stride_elements(in.gate, DType::BF16, {Geo::kAttnWidth}, n, 1, op, "gate", 4);
        p.sink        = static_cast<const __nv_bfloat16*>(in.sink.data);
        p.sink_stride = column_stride_elements(in.sink, DType::BF16, {Geo::kQHeads}, n, 1, op, "sink", 2);
    }
    check_kv_cache(cache, op);
    p.k_pages       = static_cast<const std::uint8_t*>(cache.k_pages.data);
    p.v_pages       = static_cast<const std::uint8_t*>(cache.v_pages.data);
    p.block_tables  = static_cast<const std::int32_t*>(cache.block_tables.data);
    p.table_stride  = cache.block_tables.ne[0];
    p.positions     = static_cast<const std::int32_t*>(rows.positions.data);
    p.valid_columns = rs.masked ? static_cast<const std::int32_t*>(rows.valid_columns.data) : nullptr;
    p.table_rows    = static_cast<const std::int32_t*>(rows.kv_table_rows.data);
    p.width         = rs.width;
    p.batch         = rs.batch;
    // Triton computes `softmax_scale * LOG2E` in fp32 inside the kernel; one IEEE product here.
    const float scale = static_cast<float>(0.08838834764831845);
    p.qk_scale_log2   = scale * kLog2e;
    // Default 17 = bias added unfused (s + bf16(b) * log2e) and lse with the accurate log2f: decided against lane
    // a's capture of vLLM's own sm_120 prefill kernel (M3 run 2, visits 0 and 5: fp32 out bit-equal 47.5 -> 52.1 %,
    // lse 82.8 -> 89.1 % vs variant 0). The env override remains for the bit-identity experiment.
    const char* variant = std::getenv("SUROGATE_INKLING_ATTN_VARIANT");
    p.variant         = variant == nullptr ? 17 : std::atoi(variant);
    return p;
}

void run(const InklingAttentionInputs& in, InklingLayerKind kind, const InklingAttentionRows& rows,
         const PagedKVBatchLayerView& cache, WorkspaceArena& workspace, int final_mode, Tensor* out,
         Tensor* out_f32, Tensor* lse, cudaStream_t stream, const char* op) {
    const RowsShape rs = check_rows(rows, op);
    AttnParams p       = make_params(in, rows, cache, rs, final_mode == kGated, op);
    const std::int64_t n = rs.columns;
    if (final_mode == kGated) {
        require_contiguous(*out, DType::BF16, n * Geo::kAttnWidth, op, "out");
        p.out = static_cast<__nv_bfloat16*>(out->data);
    } else {
        require_contiguous(*out_f32, DType::FP32, n * Geo::kQHeads * Geo::kHeadDim, op, "out");
        require_contiguous(*lse, DType::FP32, n * Geo::kQHeads, op, "lse");
        p.out_f32 = static_cast<float*>(out_f32->data);
        p.lse     = static_cast<float*>(lse->data);
    }
    const Plan plan = make_plan(kind, rs.width, rs.batch, static_cast<std::int32_t>(rows.envelope.max_visible_keys));
    p.splits        = plan.splits;
    if (plan.splits == 1) {
        dispatch(kind, final_mode, plan, p, stream);
        return;
    }
    auto scope       = workspace.scope();
    const std::int64_t part_rows = n * Geo::kQHeads * plan.splits;
    p.part_o    = static_cast<float*>(workspace.alloc_bytes(static_cast<std::size_t>(part_rows * Geo::kHeadDim * 4)).data);
    p.part_lse2 = static_cast<float*>(workspace.alloc_bytes(static_cast<std::size_t>(part_rows * 4)).data);
    dispatch(kind, kPartial, plan, p, stream);
    const dim3 grid(static_cast<unsigned>(n), Geo::kQHeads);
    if (final_mode == kGated) {
        combine_kernel<kGated><<<grid, 128, 0, stream>>>(p);
    } else {
        combine_kernel<kOutLse><<<grid, 128, 0, stream>>>(p);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace
} // namespace inkling_detail

std::size_t inkling_rel_attention_workspace_bytes(InklingLayerKind kind, std::int32_t width,
                                                  std::int32_t batch, GqaExecutionEnvelope envelope) {
    using namespace inkling_detail;
    if (width <= 0 || batch <= 0 || envelope.max_visible_keys > static_cast<std::uint32_t>(1 << 20)) {
        throw std::invalid_argument("inkling_rel_attention_workspace_bytes: outside the supported domain");
    }
    const Plan plan = make_plan(kind, width, batch, static_cast<std::int32_t>(envelope.max_visible_keys));
    return partial_bytes(plan, static_cast<std::int64_t>(width) * batch);
}

std::size_t inkling_attention_workspace_capacity_bytes(InklingLayerKind kind, std::int32_t batch,
                                                       std::int32_t min_width, std::int32_t max_width,
                                                       GqaExecutionEnvelope envelope) {
    using namespace inkling_detail;
    if (batch <= 0 || min_width <= 0 || max_width < min_width) {
        throw std::invalid_argument("inkling_attention_workspace_capacity_bytes: outside the supported domain");
    }
    std::size_t most = 0;
    for (std::int32_t w = min_width; w <= max_width; ++w) {
        const std::int64_t n   = static_cast<std::int64_t>(w) * batch;
        const std::size_t qr   = static_cast<std::size_t>(n * Geo::kQHeads * (Geo::kHeadDim + Geo::kRelDim) * 2) + 512;
        most                   = std::max(most, qr + inkling_rel_attention_workspace_bytes(kind, w, batch, envelope));
    }
    return most;
}

void inkling_attention(const InklingProjections& projections, const InklingAttentionWeights& weights,
                       InklingLayerKind kind, const InklingAttentionRows& rows,
                       const InklingConvStateArgs& conv, const PagedKVBatchLayerView& cache,
                       WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    using namespace inkling_detail;
    const RowsShape rs   = check_rows(rows, "inkling_attention");
    const std::int64_t n = rs.columns;
    auto scope           = workspace.scope();
    InklingPrologueOutputs qr;
    qr.q = Tensor(workspace.alloc_bytes(static_cast<std::size_t>(n * Geo::kQHeads * Geo::kHeadDim * 2)).data,
                  DType::BF16, {Geo::kHeadDim, Geo::kQHeads, static_cast<std::int32_t>(n)});
    qr.r = Tensor(workspace.alloc_bytes(static_cast<std::size_t>(n * Geo::kQHeads * Geo::kRelDim * 2)).data,
                  DType::BF16, {Geo::kRelDim, Geo::kQHeads, static_cast<std::int32_t>(n)});
    inkling_attention_prologue(projections, weights, kind, rows, conv, cache, qr, stream);
    const InklingAttentionInputs in{qr.q, qr.r, projections.gate, projections.sink, weights.rel_table};
    inkling_rel_attention(in, kind, rows, cache, workspace, out, stream);
}

void inkling_rel_attention(const InklingAttentionInputs& inputs, InklingLayerKind kind,
                           const InklingAttentionRows& rows, const PagedKVBatchLayerView& cache,
                           WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    inkling_detail::run(inputs, kind, rows, cache, workspace, inkling_detail::kGated, &out, nullptr, nullptr,
                        stream, "inkling_rel_attention");
}

void inkling_rel_attention_out_lse(const InklingAttentionInputs& inputs, InklingLayerKind kind,
                                   const InklingAttentionRows& rows,
                                   const PagedKVBatchLayerView& cache, WorkspaceArena& workspace,
                                   Tensor& out, Tensor& lse, cudaStream_t stream) {
    inkling_detail::run(inputs, kind, rows, cache, workspace, inkling_detail::kOutLse, nullptr, &out, &lse,
                        stream, "inkling_rel_attention_out_lse");
}

void inkling_gate_sink(const Tensor& out, const Tensor& lse, const Tensor& gate, const Tensor& sink,
                       Tensor& result, cudaStream_t stream) {
    using namespace inkling_detail;
    const char* op = "inkling_gate_sink";
    if (out.dtype != DType::FP32 || !out.is_contiguous() || out.numel() % Geo::kAttnWidth != 0) {
        fail(op, "out must be contiguous FP32 [128, 24, N]");
    }
    const std::int64_t n = out.numel() / Geo::kAttnWidth;
    require_contiguous(lse, DType::FP32, n * Geo::kQHeads, op, "lse");
    require_contiguous(result, DType::BF16, n * Geo::kAttnWidth, op, "result");
    const std::int64_t gs = column_stride_elements(gate, DType::BF16, {Geo::kAttnWidth}, n, 1, op, "gate", 2);
    const std::int64_t ss = column_stride_elements(sink, DType::BF16, {Geo::kQHeads}, n, 1, op, "sink", 2);
    if (n == 0) { return; }
    const std::int64_t total = n * Geo::kAttnWidth;
    gate_sink_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
        static_cast<const float*>(out.data), static_cast<const float*>(lse.data),
        static_cast<const __nv_bfloat16*>(gate.data), gs, static_cast<const __nv_bfloat16*>(sink.data), ss,
        static_cast<__nv_bfloat16*>(result.data), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops
