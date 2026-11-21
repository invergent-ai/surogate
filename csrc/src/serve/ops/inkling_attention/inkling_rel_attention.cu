// Inkling-2-Mini relative-bias attention (ENGINE-INKLING lane c): ONE batch- and path-invariant per-row arithmetic
// for decode, MTP verify, prefill, chunked prefill and the trainer's forward; the group fold; the gate-and-sink
// epilogue. Semantics: api/ops/inkling_attention.h; numerics: lanes/c/DESIGN-attention.md section 1 (A1-A7).
// Reference: vLLM's ops/attention_sm90.py `_rel_attention_kernel`, ops/attention.py `_gate_sink_kernel`.
//
// INVARIANCE (agent-EN decision 10:2x, FEASIBILITY-ANSWER.md; lanes/c/NOTES.md "INVARIANT ATTENTION"). The bits of a
// row (query token x head) depend only on its own q / r / gate / sink, its position and the keys in the plane --
// never on the batch, the width W, the envelope, the chunk cut, the split count or the other rows of a tile:
// - Keys go in fixed GROUPS whose boundaries depend only on the absolute key position (GroupSpec, chosen at run time
//   by SUROGATE_INKLING_ATTN_GROUPS: p1 = 512 global / 128 local, p2 = 2,048 global growing to 8,192 from key 65,536
//   / 256 local). Inside a group: 64-key blocks aligned at key 0, in key order, one running softmax state per row
//   (log2 domain; P rounded to bf16 against the running maximum; P V as k16 MMA steps in key order). A group's
//   state (m, l, acc) is folded into the row's total in group order by ONE sequential formula (fold_*), in
//   registers / shared memory (in-kernel route) or by fold_kernel over per-group partials (split route): the same
//   device functions, so both routes give the same bits. No atomics.
// - Every per-element step is the same instruction sequence on every route: S = eight chained m16n8k16 steps over
//   the head dimension from zero; the bias = one k16 MMA from zero, rounded to bf16; masks by select (keys behind
//   the local window, past the row's position or outside the CTA's key range score -inf and get P = 0; their codes
//   are zero-filled, never read). A block's row sum l is ((Q0 + Q1) + Q2) + Q3, Qq = the quad butterfly of the
//   thread's four P values of key quarter q in order: a warp that owns all 64 keys (prefill, KS = 1) and KS warps
//   that split the block's keys for S and its head dims for P V (decode / verify, KS = 2 or 4, exchanging the
//   block maximum, the quarter sums and the P fragments through shared memory) compute the same numbers.
// - The gate / sink epilogue applies to the folded (o, lse) on every route (gated_*).
// - Attention reads every key back from the e4m3 plane, the current chunk's own keys included.

#include "ops/inkling_attention/inkling_attention_common.h"
#include "ops/inkling_attention/inkling_attention_device.cuh"

#include "core/device.h"
#include "core/pdl.cuh"

#include <math_constants.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace inkling_detail {
namespace {

inline constexpr int kBn          = 64;                 // keys per block, aligned at key 0
inline constexpr int kRowBytes    = Geo::kHeadDim * 2;  // a bf16 tile row
inline constexpr int kPrefillRows = 64;                 // query columns of one prefill CTA (4 row tiles)
inline constexpr int kMaxWorkers  = 64;
inline constexpr int kSmemLimit   = 101376;             // sm_120 opt-in shared memory per block

enum Mode : int { kGated = 0, kOutLse = 1, kPartial = 2 };

/// Key groups, a function of the absolute key position only: keys below `threshold` in groups of 1 << shift1,
/// keys from `threshold` in groups of 1 << shift2 (threshold a multiple of both sizes).
struct GroupSpec {
    int shift1    = 9;
    int shift2    = 9;
    int threshold = 1 << 30;
};

__host__ __device__ __forceinline__ int group_of(const GroupSpec& g, int key) {
    return key < g.threshold ? key >> g.shift1 : (g.threshold >> g.shift1) + ((key - g.threshold) >> g.shift2);
}
__host__ __device__ __forceinline__ int group_begin(const GroupSpec& g, int group) {
    const int below = g.threshold >> g.shift1;
    return group < below ? group << g.shift1 : g.threshold + ((group - below) << g.shift2);
}

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
    std::int32_t width, batch;
    std::int32_t workers;       // CTAs per (row set): worker w takes the groups g_first + w, + workers, ...
    std::int32_t max_slots;     // partial slots per row (split route)
    GroupSpec groups;
    float qk_scale_log2;
    __nv_bfloat16* out;  // Gated: [3072, N]
    float* out_f32;      // OutLse: [128, 24, N]
    float* lse;          // OutLse: [24, N]
    float2* part_ml;     // Partial: [slot][N][24] (m in the log2 domain, l)
    float* part_acc;     // Partial: [slot][N][24][128] (unnormalised)
    // Numerics variant bits (SUROGATE_INKLING_ATTN_VARIANT; the vLLM bit-identity experiment, DESIGN 1 A3-A5; one value
    // per call, so every route of a call uses the same): 1 bias as s + bf16(b) * log2e (else fma), 8 o = A / L with
    // div.rn (else div.full), 16 lse with accurate log2f (else lg2.approx), 32 l = l * alpha + sum unfused.
    int variant;
};

__device__ __forceinline__ unsigned word_or_zero(const __nv_bfloat16* p, bool ok) {
    return ok ? load_vec<unsigned>(p) : 0U;
}

__device__ __forceinline__ void named_barrier(int id, int threads) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

// ---- the fold and the epilogue: ONE definition for the in-kernel route and fold_kernel ----------------------------

/// Coefficients that fold a group with running maximum m_g (> -inf) into a total with maximum M (-inf when empty).
__device__ __forceinline__ void fold_coeffs(float M, float m_g, float& M_new, float& a, float& c) {
    M_new = fmaxf(M, m_g);
    a     = ptx_ex2(__fsub_rn(M, M_new)); // empty total: ex2(-inf) = 0
    c     = ptx_ex2(__fsub_rn(m_g, M_new));
}
__device__ __forceinline__ float fold_value(float total, float a, float part, float c) {
    return __fmaf_rn(total, a, __fmul_rn(part, c));
}
__device__ __forceinline__ float final_lse2(float M, float L, int variant) {
    return __fadd_rn(M, (variant & 16) ? log2f(L) : ptx_lg2(L));
}
__device__ __forceinline__ float final_o(float A, float L, int variant) {
    return (variant & 8) ? __fdiv_rn(A, L) : ptx_div_full(A, L);
}
__device__ __forceinline__ float gated_sink_factor(float lse2, const __nv_bfloat16* sink) {
    return ref_sigmoid(__fsub_rn(__fmul_rn(lse2, kLn2), __bfloat162float(*sink)));
}
__device__ __forceinline__ float gated_value(float o, float gate, float sig_sink) {
    return __fmul_rn(__fmul_rn(o, ref_sigmoid(gate)), sig_sink);
}

// ---- the attention kernel -----------------------------------------------------------------------------------------

template <int MTiles, int KS, int Raw, bool Fold>
struct AttnSmem {
    static constexpr int kWarps       = MTiles * KS;
    static constexpr int kKeysW       = kBn / KS;         // keys of a warp within a block
    static constexpr int kStripCols   = kKeysW + 16;      // distances a warp's strip covers (row spread <= 15)
    static constexpr int kStripStride = kStripCols + 8;
    static constexpr int kDtw         = 16 / KS;          // output dim n-tiles of a warp
    static constexpr int kX           = KS > 1 ? MTiles : 1; // (zero-length arrays are not C++)
    static constexpr int kF           = Fold ? kWarps : 1;
    static constexpr int kRawBytes    = Raw * 2 * kBn * Geo::kHeadDim; // Raw = 0: register-staged (decode)
    __nv_bfloat16 k[kBn * Geo::kHeadDim];
    __nv_bfloat16 v[kBn * Geo::kHeadDim];
    __nv_bfloat16 strip[kWarps][16 * kStripStride];
    float xmax[kX][KS][16];
    float xq[kX][4][16];
    uint4 xpk[kX][4][32];
    float fold_a[kF][kDtw * 4][32]; // thread-private fold totals: [warp][dim tile x element][lane]
    alignas(16) std::uint8_t raw[kRawBytes > 0 ? kRawBytes : 16]; // [slot][K | V][key][128]
};

template <int MTiles, int KS, bool Fold>
constexpr int pick_raw() {
    return sizeof(AttnSmem<MTiles, KS, 3, Fold>) <= kSmemLimit ? 3 : sizeof(AttnSmem<MTiles, KS, 2, Fold>) <= kSmemLimit ? 2 : 1;
}

// A CTA owns MTiles row tiles of 16 rows and walks the blocks of its groups; warp (mt, kq) serves row tile mt and key
// quarter / head-dim share kq of KS. Decode / verify rows are the 3 heads x W columns of one sequence and KV head (row
// = 3 j + h); prefill rows are 64 consecutive columns of one query head. Mode kPartial writes each group's state per
// row (split route, fold_kernel folds); kGated / kOutLse fold in the kernel (workers == 1) and finish.
template <bool Local, bool Decode, int MTiles, int KS, int Mode, int Raw>
__global__ __launch_bounds__(MTiles * KS * 32, Decode ? 2 : 1) void rel_attention_kernel(const AttnParams p) {
    constexpr bool Fold    = Mode != kPartial;
    constexpr int Threads  = MTiles * KS * 32;
    constexpr int NTW      = 8 / KS;  // score n-tiles of a warp
    constexpr int QW       = 4 / KS;  // key quarters of a warp
    constexpr int DTW      = 16 / KS; // P V dim n-tiles of a warp
    using Smem             = AttnSmem<MTiles, KS, Raw, Fold>;
    constexpr int KeysW    = Smem::kKeysW;
    constexpr int kSS      = Smem::kStripStride;
    extern __shared__ __align__(16) unsigned char attn_smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(attn_smem_raw);
    pdl::wait_for_dependencies();

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int mt   = warp / KS;
    const int kq   = warp % KS;

    int b = 0, kvh = 0, worker = 0, j_base = 0, head_fixed = 0;
    if constexpr (Decode) {
        worker = static_cast<int>(blockIdx.x);
        kvh    = static_cast<int>(blockIdx.y);
        b      = static_cast<int>(blockIdx.z);
    } else {
        j_base     = static_cast<int>(blockIdx.x) * kPrefillRows;
        head_fixed = static_cast<int>(blockIdx.y);
        kvh        = head_fixed / Geo::kGroup;
        b          = static_cast<int>(blockIdx.z) / p.workers;
        worker     = static_cast<int>(blockIdx.z) % p.workers;
    }
    const int W = p.width;
    int valid   = W;
    if (p.valid_columns != nullptr) { valid = max(0, min(p.valid_columns[b], W)); }
    const std::int32_t* positions = p.positions + static_cast<std::int64_t>(W) * b;
    const std::int32_t* table_row = p.block_tables + static_cast<std::int64_t>(p.table_rows[b]) * p.table_stride;
    const std::int64_t n_columns  = static_cast<std::int64_t>(W) * p.batch;
    const GroupSpec gs            = p.groups;

    const auto row_column = [&](int rho) { return Decode ? rho / Geo::kGroup : j_base + rho; };
    const auto row_head   = [&](int rho) { return Decode ? kvh * Geo::kGroup + rho % Geo::kGroup : head_fixed; };
    const auto row_exists = [&](int rho) { return Decode ? rho < Geo::kGroup * W : j_base + rho < W; };

    // The CTA's valid columns are consecutive positions; its keys [lo, hi), its groups [g_first, g_last].
    const int j_lo = Decode ? 0 : j_base;
    const int j_hi = Decode ? valid - 1 : min(j_base + kPrefillRows - 1, valid - 1);
    const bool any = j_lo <= j_hi;
    const int pmin = any ? positions[j_lo] : 0;
    const int pmax = any ? positions[j_hi] : -1;
    const int hi   = pmax + 1;
    const int lo   = Local ? max(0, pmin - (Geo::kWindow - 1)) : 0;
    const int g_first = any ? group_of(gs, lo) : 0;
    const int g_last  = any ? group_of(gs, hi - 1) : -1;

    // This warp's row tile.
    const int w_rho0  = mt * 16;
    const int w_j0    = row_column(w_rho0);
    const int w_j1    = min(row_column(w_rho0 + 15), valid - 1);
    const bool w_any  = w_j0 <= w_j1;
    const int pmin_w  = w_any ? positions[w_j0] : 0;
    const int pmax_w  = w_any ? positions[w_j1] : -1;
    const bool w_full = Decode ? (w_rho0 + 15) < Geo::kGroup * valid : (j_base + w_rho0 + 15) < valid;

    // This thread's two rows (gid, gid + 8) and their group ranges.
    int rj[2], rh[2], rpos[2], rg0[2], rg1[2];
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
        rg0[i]        = rvalid[i] ? group_of(gs, Local ? max(0, rpos[i] - (Geo::kWindow - 1)) : 0) : 0;
        rg1[i]        = rvalid[i] ? group_of(gs, rpos[i]) : -1;
    }

    // Q (eight k16 steps) and R fragments, in registers for the whole walk.
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

    // Running state of the current group (this warp's head-dim share) and the folded total (Fold).
    float acc[DTW][4];
#pragma unroll
    for (int n = 0; n < DTW; ++n) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0F; }
    float m[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float l[2] = {0.0F, 0.0F};
    float Mt[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float Lt[2] = {0.0F, 0.0F};
    if constexpr (Fold) {
#pragma unroll
        for (int i = 0; i < DTW * 4; ++i) { sm.fold_a[warp][i][lane] = 0.0F; }
    }

    // This worker's groups [g_start, g_stop): the CTA's blocks split into `workers` contiguous, equal block ranges,
    // each group owned by the worker whose range holds the group's first block (balanced by keys to one group; free:
    // every group is computed the same way wherever it runs).
    int g_start = g_first, g_stop = g_last + 1;
    if (p.workers > 1 && any) {
        const int kb_lo   = lo >> 6;
        const int kb_last = (hi - 1) >> 6;
        const int nb      = kb_last + 1 - kb_lo;
        const auto first_group_from = [&](int blk) {
            if (blk <= kb_lo) { return g_first; }
            if (blk > kb_last) { return g_last + 1; }
            int g = group_of(gs, blk * kBn);
            if (group_begin(gs, g) < blk * kBn) { ++g; }
            return g;
        };
        g_start = first_group_from(kb_lo + static_cast<int>(static_cast<std::int64_t>(worker) * nb / p.workers));
        g_stop  = first_group_from(kb_lo + static_cast<int>(static_cast<std::int64_t>(worker + 1) * nb / p.workers));
    }

    // Block cursor over the worker's groups, each group's blocks in key order.
    struct Cursor {
        int g, kb, kb_end;
    };
    const auto start = [&](Cursor& c, int g) {
        c.g = g;
        if (g < g_stop) {
            c.kb     = max(group_begin(gs, g), lo) >> 6;
            c.kb_end = ((min(group_begin(gs, g + 1), hi) - 1) >> 6) + 1;
        }
    };
    const auto advance = [&](Cursor& c) {
        if (++c.kb == c.kb_end) { start(c, c.g + 1); }
    };

    // Land block kb's K and V codes in raw slot `slot`; keys outside [lo, hi) are zero-filled and never read.
    const auto stage = [&](int kb, int slot) {
        const int k0           = kb * kBn;
        const std::int32_t pg  = table_row[k0 >> kPageShift];
        const std::int64_t off = kv_page_offset(pg, kvh, k0 & (kPageTokens - 1));
        for (int c = tid; c < 2 * kBn * 8; c += Threads) {
            const int which = c / (kBn * 8);
            const int cc    = c - which * (kBn * 8);
            const int key   = cc >> 3;
            const int part  = cc & 7;
            const int kabs  = k0 + key;
            std::uint8_t* dst = sm.raw + (slot * 2 + which) * (kBn * Geo::kHeadDim) + key * Geo::kHeadDim + part * 16;
            const std::uint8_t* src = (which ? p.v_pages : p.k_pages) + off + key * Geo::kHeadDim + part * 16;
            cp_async_zfill<16>(dst, src, (kabs >= lo && kabs < hi) ? 16 : 0);
        }
    };
    // Widen raw slot `slot` into the swizzled bf16 tiles (16-byte chunk c of key row k at c ^ (k & 7)).
    const auto widen = [&](int slot) {
        for (int c = tid; c < 2 * kBn * 16; c += Threads) {
            const int which = c / (kBn * 16);
            const int cc    = c - which * (kBn * 16);
            const int key   = cc >> 4;
            const int ch    = cc & 15;
            const std::uint8_t* src = sm.raw + (slot * 2 + which) * (kBn * Geo::kHeadDim) + key * Geo::kHeadDim + ch * 8;
            __nv_bfloat16* dst      = (which ? sm.v : sm.k) + key * Geo::kHeadDim + ((ch ^ (key & 7)) << 3);
            store_vec(dst, e4m3x8_to_bf16x8(load_vec<int2>(src)));
        }
    };

    const unsigned k_base = smem_addr(sm.k);
    const unsigned v_base = smem_addr(sm.v);
    __nv_bfloat16* strip  = sm.strip[warp];

    // One 64-key block for this warp's row tile (tile-uniform skip: no row of the tile sees a key of the block).
    const auto compute_block = [&](int kb) {
        const int k0 = kb * kBn;
        if (!w_any || k0 > pmax_w || (Local && k0 + kBn - 1 < pmin_w - (Geo::kWindow - 1))) { return; }
        const int kbase = kq * KeysW; // this warp's keys within the block

        // S = Q K^T for this warp's keys: eight chained k16 steps over the head dimension in natural order, from zero.
        float sc[NTW][4];
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt) { sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.0F; }
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
#pragma unroll
            for (int nt2 = 0; nt2 < NTW; nt2 += 2) {
                const int key   = kbase + nt2 * 8 + (lane & 7) + ((lane >> 4) << 3);
                const int chunk = 2 * ks + ((lane >> 3) & 1);
                unsigned b0, b1, b2, b3;
                ldmatrix_x4(b0, b1, b2, b3, k_base + static_cast<unsigned>(key * kRowBytes + ((chunk ^ (key & 7)) << 4)));
                mma_bf16(sc[nt2][0], sc[nt2][1], sc[nt2][2], sc[nt2][3], af[ks][0], af[ks][1], af[ks][2], af[ks][3], b0, b1);
                mma_bf16(sc[nt2 + 1][0], sc[nt2 + 1][1], sc[nt2 + 1][2], sc[nt2 + 1][3], af[ks][0], af[ks][1], af[ks][2],
                         af[ks][3], b2, b3);
            }
        }
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) { sc[nt][e] = __fmul_rn(sc[nt][e], p.qk_scale_log2); }
        }

        // Relative bias for this warp's keys within the extent: strip column c = distance d0 + c, per row a k16 MMA
        // from zero rounded to bf16 (distances >= 1,024 read the zero padding: adding it is exact).
        const int d0 = pmin_w - (k0 + kbase + KeysW - 1);
        if (d0 < Geo::kRelExtent) {
            __syncwarp();
#pragma unroll
            for (int ct = 0; ct < Smem::kStripCols / 8; ++ct) {
                const __nv_bfloat16* t =
                    p.table + static_cast<std::int64_t>(d0 + 8 * ct + gid + Geo::kRelTablePad) * Geo::kRelDim + 2 * lid;
                float c0 = 0.0F, c1 = 0.0F, c2 = 0.0F, c3 = 0.0F;
                mma_bf16(c0, c1, c2, c3, ar[0], ar[1], ar[2], ar[3], load_vec<unsigned>(t), load_vec<unsigned>(t + 8));
                store_vec(strip + gid * kSS + 8 * ct + 2 * lid, pack_bf16x2(c0, c1));
                store_vec(strip + (gid + 8) * kSS + 8 * ct + 2 * lid, pack_bf16x2(c2, c3));
            }
            __syncwarp();
#pragma unroll
            for (int nt = 0; nt < NTW; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int i = e >> 1;
                    if (!rvalid[i]) { continue; }
                    const int kc     = nt * 8 + 2 * lid + (e & 1);
                    const int c      = (rpos[i] - pmin_w) + (KeysW - 1) - kc;
                    const float bias = __bfloat162float(strip[(gid + 8 * i) * kSS + c]);
                    sc[nt][e] = (p.variant & 1) ? __fadd_rn(sc[nt][e], __fmul_rn(bias, kLog2e)) : fmaf(bias, kLog2e, sc[nt][e]);
                }
            }
        }

        // Causal / window / validity mask by select, only where some element can be invisible.
        const bool need_mask = !w_full || (k0 + kBn - 1 > pmin_w) || (Local && k0 <= pmax_w - Geo::kWindow);
        if (need_mask) {
#pragma unroll
            for (int nt = 0; nt < NTW; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int i   = e >> 1;
                    const int key = k0 + kbase + nt * 8 + 2 * lid + (e & 1);
                    const bool vis = rvalid[i] && key <= rpos[i] && (!Local || rpos[i] - key < Geo::kWindow);
                    if (!vis) { sc[nt][e] = -CUDART_INF_F; }
                }
            }
        }

        // Block maximum per row (exact, order-free), across the KS warps of the tile.
        float bm[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt) {
            bm[0] = fmaxf(bm[0], fmaxf(sc[nt][0], sc[nt][1]));
            bm[1] = fmaxf(bm[1], fmaxf(sc[nt][2], sc[nt][3]));
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) { bm[i] = warp_max<4>(bm[i]); }
        if constexpr (KS > 1) {
            if (lid == 0) {
                sm.xmax[mt][kq][gid]     = bm[0];
                sm.xmax[mt][kq][gid + 8] = bm[1];
            }
            named_barrier(1 + mt, KS * 32);
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                float x = sm.xmax[mt][0][gid + 8 * i];
#pragma unroll
                for (int w = 1; w < KS; ++w) { x = fmaxf(x, sm.xmax[mt][w][gid + 8 * i]); }
                bm[i] = x;
            }
        }
        float alpha[2], msafe[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float mn = fmaxf(m[i], bm[i]);
            msafe[i]       = mn == -CUDART_INF_F ? 0.0F : mn;
            alpha[i]       = ptx_ex2(__fsub_rn(m[i], msafe[i]));
            m[i]           = mn;
        }

        // P (bf16 fragments) and the canonical row sum: quarter sums Qq (thread order, then the quad butterfly). The
        // KS warps of a tile exchange their quarters through shared memory (every warp then reads all four back).
        unsigned pk[4][4];
        float qs[4][2];
        unsigned po[QW][4];
        float qo[QW][2];
#pragma unroll
        for (int qi = 0; qi < QW; ++qi) {
            float pv[2][4];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
                for (int e = 0; e < 4; ++e) { pv[h][e] = ptx_ex2(__fsub_rn(sc[2 * qi + h][e], msafe[e >> 1])); }
            }
            po[qi][0] = pack_bf16x2(pv[0][0], pv[0][1]);
            po[qi][1] = pack_bf16x2(pv[0][2], pv[0][3]);
            po[qi][2] = pack_bf16x2(pv[1][0], pv[1][1]);
            po[qi][3] = pack_bf16x2(pv[1][2], pv[1][3]);
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const float t = __fadd_rn(__fadd_rn(__fadd_rn(pv[0][2 * i], pv[0][2 * i + 1]), pv[1][2 * i]), pv[1][2 * i + 1]);
                qo[qi][i]     = warp_sum<4>(t);
            }
        }
        if constexpr (KS == 1) {
#pragma unroll
            for (int qg = 0; qg < 4; ++qg) {
#pragma unroll
                for (int x = 0; x < 4; ++x) { pk[qg][x] = po[qg][x]; }
                qs[qg][0] = qo[qg][0];
                qs[qg][1] = qo[qg][1];
            }
        } else {
#pragma unroll
            for (int qi = 0; qi < QW; ++qi) {
                const int qg = kq * QW + qi;
                if (lid == 0) {
                    sm.xq[mt][qg][gid]     = qo[qi][0];
                    sm.xq[mt][qg][gid + 8] = qo[qi][1];
                }
                sm.xpk[mt][qg][lane] = make_uint4(po[qi][0], po[qi][1], po[qi][2], po[qi][3]);
            }
            named_barrier(1 + mt, KS * 32);
#pragma unroll
            for (int qg = 0; qg < 4; ++qg) {
                qs[qg][0]     = sm.xq[mt][qg][gid];
                qs[qg][1]     = sm.xq[mt][qg][gid + 8];
                const uint4 x = sm.xpk[mt][qg][lane];
                pk[qg][0]     = x.x;
                pk[qg][1]     = x.y;
                pk[qg][2]     = x.z;
                pk[qg][3]     = x.w;
            }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float bl = __fadd_rn(__fadd_rn(__fadd_rn(qs[0][i], qs[1][i]), qs[2][i]), qs[3][i]);
            l[i] = (p.variant & 32) ? __fadd_rn(__fmul_rn(l[i], alpha[i]), bl) : fmaf(l[i], alpha[i], bl);
        }
        if (__any_sync(kFullWarpMask, alpha[0] != 1.0F || alpha[1] != 1.0F)) {
#pragma unroll
            for (int n = 0; n < DTW; ++n) {
                acc[n][0] = __fmul_rn(acc[n][0], alpha[0]);
                acc[n][1] = __fmul_rn(acc[n][1], alpha[0]);
                acc[n][2] = __fmul_rn(acc[n][2], alpha[1]);
                acc[n][3] = __fmul_rn(acc[n][3], alpha[1]);
            }
        }
        // O += P V for this warp's head dims: four k16 steps over the block's keys in order.
#pragma unroll
        for (int n2 = 0; n2 < DTW; n2 += 2) {
            const int dn = kq * DTW + n2;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const int key   = kk * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
                const int chunk = dn + (lane >> 4);
                unsigned v0, v1, v2, v3;
                ldmatrix_x4_t(v0, v1, v2, v3, v_base + static_cast<unsigned>(key * kRowBytes + ((chunk ^ (key & 7)) << 4)));
                mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], pk[kk][0], pk[kk][1], pk[kk][2], pk[kk][3], v0, v1);
                mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3], pk[kk][0], pk[kk][1], pk[kk][2],
                         pk[kk][3], v2, v3);
            }
        }
    };

    // Group g ended: fold it into the total (Fold) or write it as the rows' partial (split route); reset the state.
    const auto finish_group = [&](int g) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            if (!rvalid[i] || g < rg0[i] || g > rg1[i]) { continue; }
            if constexpr (Fold) {
                if (l[i] > 0.0F) {
                    float Mn, a, c;
                    fold_coeffs(Mt[i], m[i], Mn, a, c);
                    Mt[i] = Mn;
                    Lt[i] = fold_value(Lt[i], a, l[i], c);
#pragma unroll
                    for (int n = 0; n < DTW; ++n) {
                        float& x0 = sm.fold_a[warp][n * 4 + 2 * i][lane];
                        float& x1 = sm.fold_a[warp][n * 4 + 2 * i + 1][lane];
                        x0        = fold_value(x0, a, acc[n][2 * i], c);
                        x1        = fold_value(x1, a, acc[n][2 * i + 1], c);
                    }
                }
            } else {
                const int slot = g - rg0[i];
                if (slot >= p.max_slots) { continue; } // contract violation: fold_kernel poisons the row
                const std::int64_t part = (static_cast<std::int64_t>(slot) * n_columns + rn[i]) * Geo::kQHeads + rh[i];
                if (kq == 0 && lid == 0) { p.part_ml[part] = make_float2(m[i], l[i]); }
                float* dst = p.part_acc + part * Geo::kHeadDim + 8 * kq * DTW + 2 * lid;
#pragma unroll
                for (int n = 0; n < DTW; ++n) { store_vec(dst + 8 * n, make_float2(acc[n][2 * i], acc[n][2 * i + 1])); }
            }
        }
#pragma unroll
        for (int n = 0; n < DTW; ++n) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0F; }
        m[0] = m[1] = -CUDART_INF_F;
        l[0] = l[1] = 0.0F;
    };

    Cursor ld, cu;
    start(ld, g_start);
    start(cu, g_start);
    if constexpr (Raw == 0) {
        // Register staging (decode / verify): the next block's codes are loaded into registers while the current one
        // is computed, then widened into the tiles; no raw ring, so two CTAs fit an SM.
        constexpr int NR = (kBn * 16 + Threads - 1) / Threads; // 8-byte chunks of K (and of V) per thread
        int2 rk[NR], rv[NR];
        const auto load_raw = [&](int kb) {
            const int k0           = kb * kBn;
            const std::int64_t off = kv_page_offset(table_row[k0 >> kPageShift], kvh, 0);
#pragma unroll
            for (int i = 0; i < NR; ++i) {
                const int c    = tid + i * Threads;
                const int key  = c >> 4;
                const int kabs = k0 + key;
                const bool ok  = c < kBn * 16 && kabs >= lo && kabs < hi;
                const std::int64_t o = off + key * Geo::kHeadDim + (c & 15) * 8;
                rk[i] = ok ? load_ldg<int2>(p.k_pages + o) : make_int2(0, 0);
                rv[i] = ok ? load_ldg<int2>(p.v_pages + o) : make_int2(0, 0);
            }
        };
        if (ld.g < g_stop) {
            load_raw(ld.kb);
            advance(ld);
        }
        while (cu.g < g_stop) {
            __syncthreads(); // every warp is done with the previous block's tiles
#pragma unroll
            for (int i = 0; i < NR; ++i) {
                const int c = tid + i * Threads;
                if (c < kBn * 16) {
                    const int key = c >> 4;
                    const int e   = key * Geo::kHeadDim + (((c & 15) ^ (key & 7)) << 3);
                    store_vec(sm.k + e, e4m3x8_to_bf16x8(rk[i]));
                    store_vec(sm.v + e, e4m3x8_to_bf16x8(rv[i]));
                }
            }
            __syncthreads();
            if (ld.g < g_stop) {
                load_raw(ld.kb); // in flight during this block's math
                advance(ld);
            }
            compute_block(cu.kb);
            const int g = cu.g;
            advance(cu);
            if (cu.g != g) { finish_group(g); }
        }
    } else {
#pragma unroll
        for (int s = 0; s < Raw; ++s) {
            if (ld.g < g_stop) {
                stage(ld.kb, s);
                advance(ld);
            }
            cp_commit();
        }
        int it = 0;
        while (cu.g < g_stop) {
            const int slot = it % Raw;
            cp_wait<Raw - 1>();
            __syncthreads(); // block visible; every warp is done with the previous tiles
            widen(slot);
            __syncthreads(); // tiles ready; the raw slot is free
            if (ld.g < g_stop) {
                stage(ld.kb, slot);
                advance(ld);
            }
            cp_commit();
            compute_block(cu.kb);
            const int g = cu.g;
            advance(cu);
            if (cu.g != g) { finish_group(g); }
            ++it;
        }
        cp_wait<0>();
    }

    if constexpr (Fold) {
        // Epilogue (A5-A7) on the folded total. Row i of this thread holds dims 8 (kq DTW + n) + 2 lid, + 1.
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            if (!rexists[i]) { continue; }
            const std::int64_t n = rn[i];
            const int h          = rh[i];
            const bool has       = rvalid[i] && Lt[i] > 0.0F;
            const float lse2     = has ? final_lse2(Mt[i], Lt[i], p.variant) : -CUDART_INF_F;
            const int d0         = 8 * kq * DTW + 2 * lid;
            if constexpr (Mode == kGated) {
                __nv_bfloat16* dst = p.out + n * Geo::kAttnWidth + h * Geo::kHeadDim + d0;
                if (!rvalid[i]) {
#pragma unroll
                    for (int t = 0; t < DTW; ++t) { store_vec(dst + 8 * t, 0U); }
                    continue;
                }
                const float sig_sink   = gated_sink_factor(lse2, p.sink + n * p.sink_stride + h);
                const __nv_bfloat16* g = p.gate + n * p.gate_stride + h * Geo::kHeadDim + d0;
#pragma unroll
                for (int t = 0; t < DTW; ++t) {
                    const float o0  = has ? final_o(sm.fold_a[warp][t * 4 + 2 * i][lane], Lt[i], p.variant) : 0.0F;
                    const float o1  = has ? final_o(sm.fold_a[warp][t * 4 + 2 * i + 1][lane], Lt[i], p.variant) : 0.0F;
                    const float2 gv = bf16x2_bits_to_float2(load_vec<unsigned>(g + 8 * t));
                    store_vec(dst + 8 * t, pack_bf16x2(gated_value(o0, gv.x, sig_sink), gated_value(o1, gv.y, sig_sink)));
                }
            } else {
                float* dst = p.out_f32 + (n * Geo::kQHeads + h) * Geo::kHeadDim + d0;
#pragma unroll
                for (int t = 0; t < DTW; ++t) {
                    const float o0 = has ? final_o(sm.fold_a[warp][t * 4 + 2 * i][lane], Lt[i], p.variant) : 0.0F;
                    const float o1 = has ? final_o(sm.fold_a[warp][t * 4 + 2 * i + 1][lane], Lt[i], p.variant) : 0.0F;
                    store_vec(dst + 8 * t, make_float2(o0, o1));
                }
                // Columns past the valid count come out zero, as the reference zeroes its padding rows.
                if (kq == 0 && lid == 0) { p.lse[n * Geo::kQHeads + h] = rvalid[i] ? __fmul_rn(lse2, kLn2) : 0.0F; }
            }
        }
    }
}

// The split route's fold: per (column, head), the row's group partials folded in group order by the same formula as
// the in-kernel route, then the same epilogue. A thread per head dim. The fold's scalars are computed in parallel
// (exactly: the running maximum is a max scan, and every coefficient is fold_coeffs of the same two operands as in
// the sequential walk); only the fma chains of L and of A[d] walk the slots in order, with their loads in flight.
template <bool Local, int Mode>
__global__ __launch_bounds__(128) void fold_kernel(const AttnParams p) {
    extern __shared__ float fold_smem[]; // [4][max_slots]: m, l, a, c
    __shared__ float seg_max[128];
    pdl::wait_for_dependencies();
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
    bool poison   = false;
    int count     = 0;
    if (ok) {
        const GroupSpec gs = p.groups;
        const int pos      = p.positions[static_cast<std::int64_t>(W) * b + j];
        const int g0       = group_of(gs, Local ? max(0, pos - (Geo::kWindow - 1)) : 0);
        const int ns       = group_of(gs, pos) - g0 + 1;
        poison             = ns > p.max_slots;
        count              = poison ? 0 : ns;
    }
    float* sm_m = fold_smem;
    float* sm_l = sm_m + p.max_slots;
    float* sm_a = sm_l + p.max_slots;
    float* sm_c = sm_a + p.max_slots;
    const std::int64_t stride = n_columns * Geo::kQHeads;
    const float2* ml          = p.part_ml + n * Geo::kQHeads + h;
    for (int s = d; s < count; s += 128) {
        const float2 st = ml[s * stride];
        sm_m[s]         = st.x;
        sm_l[s]         = st.y;
    }
    __syncthreads();
    const int per = (count + 127) / 128;
    const int s0  = min(d * per, count);
    const int s1  = min(s0 + per, count);
    float mx      = -CUDART_INF_F;
    for (int s = s0; s < s1; ++s) {
        if (sm_l[s] > 0.0F) { mx = fmaxf(mx, sm_m[s]); }
    }
    seg_max[d] = mx;
    __syncthreads();
    float Mp = -CUDART_INF_F, M = -CUDART_INF_F;
    for (int t = 0; t < 128; ++t) {
        if (t == d) { Mp = M; }
        M = fmaxf(M, seg_max[t]);
    }
    for (int s = s0; s < s1; ++s) {
        if (sm_l[s] > 0.0F) {
            float Mn, a, c;
            fold_coeffs(Mp, sm_m[s], Mn, a, c);
            sm_a[s] = a;
            sm_c[s] = c;
            Mp      = Mn;
        }
    }
    __syncthreads();
    float L = 0.0F, A = 0.0F;
    const float* pa = p.part_acc + (n * Geo::kQHeads + h) * Geo::kHeadDim + d;
    for (int s = 0; s < count; s += 8) {
        float v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) { v[i] = s + i < count ? pa[(s + i) * stride * Geo::kHeadDim] : 0.0F; }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (s + i < count && sm_l[s + i] > 0.0F) {
                L = fold_value(L, sm_a[s + i], sm_l[s + i], sm_c[s + i]);
                A = fold_value(A, sm_a[s + i], v[i], sm_c[s + i]);
            }
        }
    }
    const bool has   = ok && L > 0.0F;
    const float lse2 = has ? final_lse2(M, L, p.variant) : -CUDART_INF_F;
    float o          = has ? final_o(A, L, p.variant) : 0.0F;
    if (poison) { o = CUDART_NAN_F; }
    if constexpr (Mode == kGated) {
        float y = 0.0F;
        if (ok) {
            const float g = __bfloat162float(p.gate[n * p.gate_stride + h * Geo::kHeadDim + d]);
            y             = gated_value(o, g, gated_sink_factor(lse2, p.sink + n * p.sink_stride + h));
        }
        p.out[n * Geo::kAttnWidth + h * Geo::kHeadDim + d] = __float2bfloat16_rn(y);
    } else {
        p.out_f32[(n * Geo::kQHeads + h) * Geo::kHeadDim + d] = ok ? o : 0.0F;
        if (d == 0) { p.lse[n * Geo::kQHeads + h] = ok ? __fmul_rn(lse2, kLn2) : 0.0F; }
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

// ---- planning (free: every plan gives the same bits) -----------------------------------------------------------------

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

int env_int(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    return raw == nullptr || raw[0] == '\0' ? fallback : std::atoi(raw);
}

/// SUROGATE_INKLING_ATTN_GROUPS: "p1" (default: 512 global / 128 local), "p2" (2,048 global, 8,192 from key 65,536 /
/// 256 local). Read per call, so one process can compare both (measurement and tests; frozen by agent-EN later).
GroupSpec group_spec(InklingLayerKind kind) {
    const char* raw = std::getenv("SUROGATE_INKLING_ATTN_GROUPS");
    const bool p2   = raw != nullptr && std::strcmp(raw, "p2") == 0;
    if (raw != nullptr && raw[0] != '\0' && !p2 && std::strcmp(raw, "p1") != 0) {
        throw std::invalid_argument("SUROGATE_INKLING_ATTN_GROUPS must be p1 or p2");
    }
    GroupSpec g;
    if (kind == InklingLayerKind::Global) {
        g.shift1    = p2 ? 11 : 9;
        g.shift2    = p2 ? 13 : 9;
        g.threshold = p2 ? 65536 : (1 << 30);
    } else {
        g.shift1 = g.shift2 = p2 ? 8 : 7;
        g.threshold         = 1 << 30;
    }
    return g;
}

struct Plan {
    bool decode    = true;
    int mtiles     = 1;
    int ks         = 4;
    int workers    = 1;
    int max_slots  = 0;
    dim3 grid;
};

/// Slots a row needs at most: global = groups up to the envelope's key count; local = groups a window touches.
int max_slots_for(InklingLayerKind kind, const GroupSpec& gs, std::int32_t max_keys) {
    if (kind == InklingLayerKind::Local) { return ((Geo::kWindow - 1) >> gs.shift1) + 2; }
    return group_of(gs, std::max(max_keys, 1) - 1) + 1;
}

std::size_t partial_bytes(const Plan& plan, std::int64_t columns) {
    if (plan.workers <= 1) { return 0; }
    const std::int64_t rows = columns * Geo::kQHeads * plan.max_slots;
    return static_cast<std::size_t>(rows * Geo::kHeadDim * 4) + 256 + static_cast<std::size_t>(rows * 8) + 256;
}

// Decode / verify: workers from (B, W, kind) and the envelope's slot count only (a captured graph replays at any
// length up to its envelope). Prefill: the split route only for few query tiles and a small partial buffer.
// SUROGATE_INKLING_ATTN_SPLITS=N forces N workers; SUROGATE_INKLING_ATTN_CTAS sets the decode CTA target;
// SUROGATE_INKLING_ATTN_PARTIAL_MB caps the partial buffer (tests and measurement).
Plan make_plan(InklingLayerKind kind, const GroupSpec& gs, std::int32_t width, std::int32_t batch, std::int32_t max_keys) {
    Plan plan;
    const int sms        = device_sms();
    plan.max_slots       = max_slots_for(kind, gs, max_keys);
    const int slot_cap   = kind == InklingLayerKind::Global ? kMaxWorkers : plan.max_slots;
    if (width <= Geo::kMaxSmallColumns) {
        plan.decode      = true;
        plan.mtiles      = (Geo::kGroup * width + 15) / 16;
        plan.ks          = plan.mtiles == 1 ? 4 : 2;
        const int work   = Geo::kKvHeads * batch;
        const int target = env_int("SUROGATE_INKLING_ATTN_CTAS", 2 * sms); // two CTAs per SM: one wave
        plan.workers     = std::clamp(target / work, 1, slot_cap);
    } else {
        plan.decode     = false;
        plan.mtiles     = 4;
        plan.ks         = 1;
        const int tiles = (width + kPrefillRows - 1) / kPrefillRows;
        const long ctas = static_cast<long>(tiles) * Geo::kQHeads * batch;
        plan.workers    = ctas >= 2L * sms ? 1 : std::clamp(static_cast<int>((2L * sms + ctas - 1) / ctas), 1,
                                                            std::min(32, plan.max_slots));
    }
    const int forced = env_int("SUROGATE_INKLING_ATTN_SPLITS", 0);
    if (forced > 0) { plan.workers = std::min(forced, kMaxWorkers); }
    const std::size_t cap = static_cast<std::size_t>(env_int("SUROGATE_INKLING_ATTN_PARTIAL_MB", 1024)) << 20;
    if (forced <= 0 && partial_bytes(plan, static_cast<std::int64_t>(width) * batch) > cap) { plan.workers = 1; }
    plan.grid = plan.decode ? dim3(static_cast<unsigned>(plan.workers), Geo::kKvHeads, static_cast<unsigned>(batch))
                            : dim3(static_cast<unsigned>((width + kPrefillRows - 1) / kPrefillRows), Geo::kQHeads,
                                   static_cast<unsigned>(batch * plan.workers));
    return plan;
}

template <bool Local, bool Decode, int MTiles, int KS, int Mode>
void launch_kernel(const Plan& plan, const AttnParams& p, cudaStream_t stream) {
    constexpr int Raw   = Decode ? 0 : pick_raw<MTiles, KS, Mode != kPartial>();
    constexpr int bytes = static_cast<int>(sizeof(AttnSmem<MTiles, KS, Raw, Mode != kPartial>));
    static_assert(bytes <= kSmemLimit, "inkling attention: shared memory over the sm_120 limit");
    auto* kernel = rel_attention_kernel<Local, Decode, MTiles, KS, Mode, Raw>;
    static const bool configured = [&] {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        return true;
    }();
    (void)configured;
    const pdl::LaunchConfig launch{plan.grid, dim3(MTiles * KS * 32), static_cast<std::size_t>(bytes), stream};
    CUDA_CHECK(pdl::launch_dependent(launch, kernel, p));
}

template <bool Local, int Mode>
void dispatch_mode(const Plan& plan, const AttnParams& p, cudaStream_t stream) {
    if (!plan.decode) { return launch_kernel<Local, false, 4, 1, Mode>(plan, p, stream); }
    switch (plan.mtiles) {
    case 1: return launch_kernel<Local, true, 1, 4, Mode>(plan, p, stream);
    case 2: return launch_kernel<Local, true, 2, 2, Mode>(plan, p, stream);
    case 3: return launch_kernel<Local, true, 3, 2, Mode>(plan, p, stream);
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

template <int Mode>
void launch_fold(InklingLayerKind kind, std::int64_t n, const AttnParams& p, cudaStream_t stream) {
    const std::size_t smem = static_cast<std::size_t>(p.max_slots) * 16;
    if (smem > 48 * 1024) { throw std::invalid_argument("inkling rel attention: too many partial slots for the fold"); }
    const pdl::LaunchConfig launch{dim3(static_cast<unsigned>(n), Geo::kQHeads), dim3(Geo::kHeadDim), smem, stream};
    if (kind == InklingLayerKind::Local) {
        CUDA_CHECK(pdl::launch_dependent(launch, fold_kernel<true, Mode>, p));
    } else {
        CUDA_CHECK(pdl::launch_dependent(launch, fold_kernel<false, Mode>, p));
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
    // a's capture of vLLM's own sm_120 prefill kernel (M3 run 2). The env override remains for the experiment.
    p.variant = env_int("SUROGATE_INKLING_ATTN_VARIANT", 17) & (1 | 8 | 16 | 32);
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
    p.groups        = group_spec(kind);
    const Plan plan = make_plan(kind, p.groups, rs.width, rs.batch, static_cast<std::int32_t>(rows.envelope.max_visible_keys));
    p.workers       = plan.workers;
    p.max_slots     = plan.max_slots;
    if (plan.workers == 1) {
        dispatch(kind, final_mode, plan, p, stream);
        return;
    }
    auto scope                   = workspace.scope();
    const std::int64_t part_rows = n * Geo::kQHeads * plan.max_slots;
    p.part_acc = static_cast<float*>(workspace.alloc_bytes(static_cast<std::size_t>(part_rows * Geo::kHeadDim * 4)).data);
    p.part_ml  = static_cast<float2*>(workspace.alloc_bytes(static_cast<std::size_t>(part_rows * 8)).data);
    dispatch(kind, kPartial, plan, p, stream);
    if (final_mode == kGated) {
        launch_fold<kGated>(kind, n, p, stream);
    } else {
        launch_fold<kOutLse>(kind, n, p, stream);
    }
}

} // namespace
} // namespace inkling_detail

std::size_t inkling_rel_attention_workspace_bytes(InklingLayerKind kind, std::int32_t width,
                                                  std::int32_t batch, GqaExecutionEnvelope envelope) {
    using namespace inkling_detail;
    if (width <= 0 || batch <= 0 || envelope.max_visible_keys > static_cast<std::uint32_t>(1 << 20)) {
        throw std::invalid_argument("inkling_rel_attention_workspace_bytes: outside the supported domain");
    }
    const Plan plan = make_plan(kind, group_spec(kind), width, batch, static_cast<std::int32_t>(envelope.max_visible_keys));
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
