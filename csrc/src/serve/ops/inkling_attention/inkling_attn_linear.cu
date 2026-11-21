// Inkling-2-Mini FP8 attention projections, W8A16 (ENGINE-INKLING lane c; DESIGN-attention.md section 6,
// agent-EN 08:2x: per-channel and 1x128 scale cells first).
//
// Weights: the block route's FP8 Weight as lane b binds it -- e4m3 codes [n][k] row-major and an fp32 scale grid
// [n / rows_per][k / k_per] row-major, cell = (scale_ne[0] = k_per, scale_ne[1] = rows_per): per-channel (k, 1),
// 1x128 (128, 1), 128x128 (128, 128). Activations stay BF16 (never quantised); every code is widened exactly; the
// dot products accumulate in fp32 and each 128-column cell's partial is multiplied by its scale in fp32 before it
// joins the row total (one multiply after the whole row for a per-channel cell). Output rounded once to bf16.
//
// Two kernels, chosen by the column count N of the call: a narrow MMA kernel for N <= 16 (decode, verify: bytes-bound,
// every code read once and widened once) and a tile GEMM beyond; both issue the same per-element MMA chain, so the
// output bits do not depend on N (agent-EN 10:2x). One launch covers up to four projections that share K and N (a
// 3-input visit's q-side, K and V row ranges, each from its own input).

#include "ops/inkling_attention/inkling_attention_common.h"
#include "ops/inkling_attention/inkling_attention_device.cuh"

#include "core/device.h"
#include "core/pdl.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace inkling_detail {
namespace {

inline constexpr int kMaxSegments = 4;

struct LinearSegment {
    const std::uint8_t* codes;   // the weight's code plane (row 0)
    const float* scales;         // the weight's scale grid
    const __nv_bfloat16* x;      // [K, N] contiguous
    __nv_bfloat16* out;          // [rows, N], column stride out_stride
    std::int64_t out_stride;     // elements
    std::int32_t row_begin;      // first weight row
    std::int32_t rows;
    std::int32_t k_per, rows_per, kcells;
    std::int32_t block_begin;    // first CTA (row block) of this segment
};

struct LinearParams {
    LinearSegment seg[kMaxSegments];
    std::int32_t segments;
    std::int32_t columns; // N
};

__device__ __forceinline__ int segment_of(const LinearParams& p, int block) {
    int s = 0;
#pragma unroll
    for (int i = 1; i < kMaxSegments; ++i) {
        if (i < p.segments && block >= p.seg[i].block_begin) { s = i; }
    }
    return s;
}

// ---- the ONE per-element arithmetic (both kernels; agent-EN decision 10:2x: width-invariant attn_linear) -----------
// Output (token t, weight row r) is a chain of bf16 m16n8k16 MMA steps with fp32 accumulation, x as operand A (m =
// tokens) and the exactly widened codes as operand B (n = weight rows). K is walked in 32-column chunks in order; a
// chunk is two steps: step A puts K offsets 8 l + {0, 1} at k slots 2 l + {0, 1} and 8 l + {2, 3} at 2 l + 8 + {0, 1}
// (l = 0..3), step B likewise 8 l + {4, 5} and 8 l + {6, 7}. So lane (g, l) of a fragment holds the 8 consecutive K
// values [8 l, 8 l + 8) of its token / row: one 16-byte load of x and one 8-byte load of codes per chunk. A 128-column
// cell's steps chain into `part` from zero; at the cell's end acc = fmaf(scale, part, acc) (1x128 / 128x128) or
// acc = acc + part (per-channel; the row's scale multiplies acc once at the end), cells in order. Cells are
// independent chains, so a kernel may interleave them (it does not change any element's operations). Output rounded
// once to bf16. The
// narrow kernel (N <= 16, one m16 tile) and the GEMM (N > 16) issue exactly this chain per element, so a column's
// bits do not depend on N (test: attn_linear width invariance).

/// One chunk's two MMA steps into c: a token fragment pair (u: token gid, v: token gid + 8) and a code fragment w.
/// Pinned = the volatile MMA (program order kept: a hand-written software pipeline); otherwise the compiler schedules.
/// Same instruction, same bits.
template <bool Pinned = false>
__device__ __forceinline__ void chunk_steps(float (&c)[4], const uint4& u, const uint4& v, const int4& w) {
    if constexpr (Pinned) {
        mma_bf16(c[0], c[1], c[2], c[3], u.x, v.x, u.y, v.y, static_cast<unsigned>(w.x), static_cast<unsigned>(w.y));
        mma_bf16(c[0], c[1], c[2], c[3], u.z, v.z, u.w, v.w, static_cast<unsigned>(w.z), static_cast<unsigned>(w.w));
    } else {
        mma_bf16_nv(c[0], c[1], c[2], c[3], u.x, v.x, u.y, v.y, static_cast<unsigned>(w.x), static_cast<unsigned>(w.y));
        mma_bf16_nv(c[0], c[1], c[2], c[3], u.z, v.z, u.w, v.w, static_cast<unsigned>(w.z), static_cast<unsigned>(w.w));
    }
}

/// Cell end: acc = fmaf(scale, part, acc) per element (rows 2 lid, 2 lid + 1 of the n8 tile = elements 0 / 2, 1 / 3),
/// or acc = acc + part for a per-channel row (PerRow).
template <bool PerRow>
__device__ __forceinline__ void cell_end(float (&acc)[4], float (&part)[4], float s0, float s1) {
    if constexpr (PerRow) {
#pragma unroll
        for (int e = 0; e < 4; ++e) { acc[e] = __fadd_rn(acc[e], part[e]); }
    } else {
        acc[0] = fmaf(s0, part[0], acc[0]);
        acc[1] = fmaf(s1, part[1], acc[1]);
        acc[2] = fmaf(s0, part[2], acc[2]);
        acc[3] = fmaf(s1, part[3], acc[3]);
    }
    part[0] = part[1] = part[2] = part[3] = 0.0F;
}

__device__ __forceinline__ float cell_scale(const LinearSegment& s, int local, int cell) {
    const int row = s.row_begin + min(local, s.rows - 1);
    return s.scales[static_cast<std::int64_t>(row / s.rows_per) * s.kcells + cell];
}

// ---- narrow: N <= 16 columns (decode, verify) ------------------------------------------------------------------------
// A warp owns 8 weight rows (one n8 tile) and all N tokens (one m16 tile). Its rows are one contiguous slab of codes:
// lane 0 asks L2 for the whole slab at entry (before the PDL wait: weights never depend on the previous kernel), so
// every byte is in flight at once. Lane (gid, lid) then streams row gid's codes, 8 a chunk, through a register ring
// kNarrowDepth chunks deep, and tokens gid / gid + 8 of x (L1 / L2 resident) through a ring kNarrowXDepth deep; the
// loads and MMAs are pinned in this order (volatile), so the rings keep their distance. Two cells run interleaved.
inline constexpr int kNarrowWarps  = 2;
inline constexpr int kNarrowRows   = kNarrowWarps * 8;
inline constexpr int kNarrowDepth  = 16;
inline constexpr int kNarrowXDepth = 4;

template <int K, bool PerRow>
__global__ __launch_bounds__(kNarrowWarps * 32) void narrow_kernel(const LinearParams p) {
    constexpr int Chunks   = K / 32;
    constexpr int Cells    = K / 128;
    static_assert(Cells % 2 == 0, "cells run in pairs");
    const int block        = static_cast<int>(blockIdx.x);
    const LinearSegment& s = p.seg[segment_of(p, block)];
    const int warp         = static_cast<int>(threadIdx.x) >> 5;
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const int gid          = lane >> 2;
    const int lid          = lane & 3;
    const int local0       = (block - s.block_begin) * kNarrowRows + warp * 8;
    // A row past the end re-reads the last row and writes nothing.
    const std::uint8_t* row = s.codes + static_cast<std::int64_t>(s.row_begin + min(local0 + gid, s.rows - 1)) * K + 8 * lid;

    if (lane == 0) {
        const int first = min(local0, s.rows - 1);
        const int rows  = min(8, s.rows - first);
        prefetch_l2_bulk(s.codes + static_cast<std::int64_t>(s.row_begin + first) * K, static_cast<unsigned>(rows * K));
    }
    // Consumption order: cell pair (2 i, 2 i + 1), chunk j of the first then chunk j of the second, j = 0..3.
    const auto order = [](int t) { return (t >> 3) * 8 + ((t & 1) << 2) + ((t >> 1) & 3); };
    uint2 ring[kNarrowDepth];
#pragma unroll
    for (int i = 0; i < kNarrowDepth; ++i) { ring[i] = ldg_pinned_v2(row + 32 * order(i)); }
    pdl::wait_for_dependencies();

    const int n_tok         = p.columns;
    const bool t0           = gid < n_tok;
    const bool t1           = gid + 8 < n_tok;
    const __nv_bfloat16* x0 = s.x + static_cast<std::int64_t>(t0 ? gid : 0) * K + 8 * lid;
    const __nv_bfloat16* x1 = s.x + static_cast<std::int64_t>(t1 ? gid + 8 : 0) * K + 8 * lid;
    const auto mask         = [](uint4 v, bool ok) { return ok ? v : make_uint4(0, 0, 0, 0); };
    uint4 xr0[kNarrowXDepth], xr1[kNarrowXDepth];
#pragma unroll
    for (int i = 0; i < kNarrowXDepth; ++i) {
        xr0[i] = ldg_pinned_v4(x0 + 32 * order(i));
        xr1[i] = ldg_pinned_v4(x1 + 32 * order(i));
    }
    float acc[4]   = {0.0F, 0.0F, 0.0F, 0.0F};
    float part0[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    float part1[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
    for (int t = 0; t < Chunks; ++t) {
        const int ch   = order(t);
        const uint2 cw = ring[t % kNarrowDepth];
        if (t + kNarrowDepth < Chunks) { ring[t % kNarrowDepth] = ldg_pinned_v2(row + 32 * order(t + kNarrowDepth)); }
        const uint4 u = mask(xr0[t % kNarrowXDepth], t0);
        const uint4 v = mask(xr1[t % kNarrowXDepth], t1);
        if (t + kNarrowXDepth < Chunks) {
            xr0[t % kNarrowXDepth] = ldg_pinned_v4(x0 + 32 * order(t + kNarrowXDepth));
            xr1[t % kNarrowXDepth] = ldg_pinned_v4(x1 + 32 * order(t + kNarrowXDepth));
        }
        const int4 w = e4m3x8_to_bf16x8(make_int2(static_cast<int>(cw.x), static_cast<int>(cw.y)));
        if ((t & 1) == 0) {
            chunk_steps<true>(part0, u, v, w);
        } else {
            chunk_steps<true>(part1, u, v, w);
        }
        if ((t & 7) == 7) { // both cells of the pair complete: join them in cell order
            const int c0 = ch >> 2; // = the second cell
            float s00 = 1.0F, s01 = 1.0F, s10 = 1.0F, s11 = 1.0F;
            if constexpr (!PerRow) {
                s00 = cell_scale(s, local0 + 2 * lid, c0 - 1);
                s01 = cell_scale(s, local0 + 2 * lid + 1, c0 - 1);
                s10 = cell_scale(s, local0 + 2 * lid, c0);
                s11 = cell_scale(s, local0 + 2 * lid + 1, c0);
            }
            cell_end<PerRow>(acc, part0, s00, s01);
            cell_end<PerRow>(acc, part1, s10, s11);
        }
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int local = local0 + 2 * lid + (e & 1);
        const int tok   = gid + 8 * (e >> 1);
        if (local >= s.rows || tok >= n_tok) { continue; }
        const float y = PerRow ? __fmul_rn(acc[e], s.scales[(s.row_begin + local) / s.rows_per]) : acc[e];
        s.out[static_cast<std::int64_t>(tok) * s.out_stride + local] = __float2bfloat16_rn(y);
    }
}

// ---- GEMM: N > 16 columns ------------------------------------------------------------------------------------------
// CTA tile 64 tokens x 64 weight rows, K in steps of 64 (two chunks; two steps per 128-column cell). 4 warps in a 2 x 2
// grid, each 32 tokens (2 m16) x 32 rows (4 n8). Codes and activations land with cp.async (two stages); the codes are
// widened into a bf16 tile. Both tiles keep K contiguous per token / row with 16-byte chunk c of line q stored at
// c ^ ((q & 1) << 2): the fragment loads (lane (gid, lid): chunk 4 c + lid of lines gid) are conflict-free.
inline constexpr int kTm = 64, kTn = 64, kTk = 64;

struct GemmSmem {
    std::uint8_t codes[2][kTn * kTk];      // 2 x 4 KiB, [row][k]
    __nv_bfloat16 x[2][kTm * kTk];         // 2 x 8 KiB, [token][k], swizzled
    __nv_bfloat16 w[kTn * kTk];            // 8 KiB, [row][k], swizzled
};

__device__ __forceinline__ int swz64(int line, int chunk) { return line * kTk + ((chunk ^ ((line & 1) << 2)) << 3); }

template <int K, bool PerRow>
__global__ __launch_bounds__(128) void gemm_kernel(const LinearParams p) {
    __shared__ __align__(16) GemmSmem sm;
    const int block        = static_cast<int>(blockIdx.x);
    const LinearSegment& s = p.seg[segment_of(p, block)];
    const int tid          = static_cast<int>(threadIdx.x);
    const int warp         = tid >> 5;
    const int lane         = tid & 31;
    const int gid          = lane >> 2;
    const int lid          = lane & 3;
    const int wm           = warp >> 1; // warp token block (32 tokens)
    const int wn           = warp & 1;  // warp row block (32 rows)
    const int local0       = (block - s.block_begin) * kTn;
    const int col0         = static_cast<int>(blockIdx.y) * kTm;
    constexpr int Steps    = K / kTk;

    // codes: 64 rows x 64 B = 256 chunks of 16 B; activations: 64 tokens x 128 B = 512 chunks.
    const auto stage_codes = [&](int step, int slot) {
        const int k0 = step * kTk;
        for (int c = tid; c < 256; c += 128) {
            const int row = c >> 2, part = c & 3;
            const int local = local0 + row;
            const bool ok   = local < s.rows;
            const std::uint8_t* src =
                s.codes + static_cast<std::int64_t>(s.row_begin + (ok ? local : 0)) * K + k0 + part * 16;
            cp_async_zfill<16>(&sm.codes[slot][row * kTk + part * 16], src, ok ? 16 : 0);
        }
    };
    const auto stage_x = [&](int step, int slot) {
        const int k0 = step * kTk;
        for (int c = tid; c < 512; c += 128) {
            const int tok = c >> 3, chunk = c & 7;
            const bool ok = col0 + tok < p.columns;
            const __nv_bfloat16* src = s.x + static_cast<std::int64_t>(ok ? col0 + tok : 0) * K + k0 + chunk * 8;
            cp_async_zfill<16>(&sm.x[slot][swz64(tok, chunk)], src, ok ? 16 : 0);
        }
    };

    float acc[2][4][4], part[2][4][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) { acc[mt][nt][e] = part[mt][nt][e] = 0.0F; }
        }
    }
    // Weight rows of this thread's accumulator elements: tile nt, element parity e -> local0 + 32 wn + 8 nt + 2 lid + e.
    const auto out_row = [&](int nt, int e) { return local0 + 32 * wn + 8 * nt + 2 * lid + e; };

    // The first two steps' weight codes go out before the PDL wait (weights never depend on the previous
    // kernel); the activations after it. Group 0 = codes 0 + codes 1 + x 0, group 1 = x 1.
    stage_codes(0, 0);
    if (Steps > 1) { stage_codes(1, 1); }
    pdl::wait_for_dependencies();
    stage_x(0, 0);
    cp_commit();
    if (Steps > 1) { stage_x(1, 1); }
    cp_commit();
    for (int st = 0; st < Steps; ++st) {
        const int slot = st & 1;
        cp_wait<1>();
        __syncthreads();
        // Widen this step's codes: 64 rows x 64 codes, 8 a chunk = 512 chunks.
        for (int c = tid; c < 512; c += 128) {
            const int row = c >> 3, chunk = c & 7;
            store_vec(&sm.w[swz64(row, chunk)], e4m3x8_to_bf16x8(load_vec<int2>(&sm.codes[slot][row * kTk + chunk * 8])));
        }
        __syncthreads();
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            uint4 u[2], v[2];
            int4 w[4];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int tok = 32 * wm + 16 * mt + gid;
                u[mt]         = load_vec<uint4>(&sm.x[slot][swz64(tok, 4 * c + lid)]);
                v[mt]         = load_vec<uint4>(&sm.x[slot][swz64(tok + 8, 4 * c + lid)]);
            }
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) { w[nt] = load_vec<int4>(&sm.w[swz64(32 * wn + 8 * nt + gid, 4 * c + lid)]); }
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) { chunk_steps(part[mt][nt], u[mt], v[mt], w[nt]); }
            }
        }
        __syncthreads(); // the w tile and this stage's slots are free
        if (st + 2 < Steps) {
            stage_codes(st + 2, slot);
            stage_x(st + 2, slot);
        }
        cp_commit();
        // A cell ends every 128 columns (two steps): join its partial to the total.
        if ((st & 1) == 1) {
            const int cell = st >> 1;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const float s0 = PerRow ? 1.0F : cell_scale(s, out_row(nt, 0), cell);
                const float s1 = PerRow ? 1.0F : cell_scale(s, out_row(nt, 1), cell);
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) { cell_end<PerRow>(acc[mt][nt], part[mt][nt], s0, s1); }
            }
        }
    }
    cp_wait<0>();

#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int local = out_row(nt, e);
            if (local >= s.rows) { continue; }
            const float sc = PerRow ? s.scales[(s.row_begin + local) / s.rows_per] : 1.0F;
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int tok = col0 + 32 * wm + 16 * mt + gid + 8 * h;
                    if (tok >= p.columns) { continue; }
                    const float v = acc[mt][nt][2 * h + e];
                    s.out[static_cast<std::int64_t>(tok) * s.out_stride + local] =
                        __float2bfloat16_rn(PerRow ? __fmul_rn(v, sc) : v);
                }
            }
        }
    }
}

struct Prepared {
    LinearParams params{};
    int k      = 0;
    int blocks = 0;
};

Prepared prepare(std::span<const InklingProjection> projections, bool gemm, const char* op) {
    if (projections.empty() || projections.size() > static_cast<std::size_t>(kMaxSegments)) {
        fail(op, "one to four projections per call");
    }
    Prepared pr;
    pr.k = projections[0].weight != nullptr ? projections[0].weight->k : 0;
    if (pr.k != 2304 && pr.k != 3072) { fail(op, "K must be 2304 (qkvr) or 3072 (wo_ud)"); }
    std::int64_t columns = -1;
    int blocks           = 0;
    for (std::size_t i = 0; i < projections.size(); ++i) {
        const InklingProjection& pj = projections[i];
        if (pj.x == nullptr || pj.weight == nullptr || pj.out == nullptr) { fail(op, "null projection member"); }
        const Weight& w = *pj.weight;
        const bool fp8  = (w.qtype == QType::FP8_E4M3FN_ROW_F32S || w.qtype == QType::FP8_E4M3FN_BLK128_F32S) &&
                         w.layout == QuantLayout::Fp8Block128 && w.scale_dtype == DType::FP32;
        const int k_per = w.scale_ne[0], rows_per = w.scale_ne[1];
        if (!fp8 || w.k != pr.k || w.qdata == nullptr || w.scales == nullptr ||
            !((k_per == w.k && rows_per == 1) || (k_per == 128 && (rows_per == 1 || rows_per == 128)))) {
            fail(op, "weights must be block-route FP8 (per-channel, 1x128 or 128x128 fp32 scales), all with K = " +
                         std::to_string(pr.k) + " (BF16 weights: ops::linear_rows)");
        }
        const Tensor& x = *pj.x;
        if (x.dtype != DType::BF16 || !x.is_contiguous() || x.ne[0] != pr.k ||
            reinterpret_cast<std::uintptr_t>(x.data) % 16 != 0) {
            fail(op, "x must be contiguous, 16-byte aligned BF16 [K, N]");
        }
        const std::int64_t n = x.numel() / pr.k;
        if (columns >= 0 && n != columns) { fail(op, "every projection of a call has the same N"); }
        columns         = n;
        const Tensor& o = *pj.out;
        const int rows  = o.ne[0];
        if (o.dtype != DType::BF16 || o.nb[0] != 2 || o.ne[1] != n || o.ne[2] != 1 || o.ne[3] != 1 || rows <= 0 ||
            o.nb[1] < static_cast<std::int64_t>(rows) * 2 || o.nb[1] % 2 != 0 || pj.row_begin < 0 ||
            pj.row_begin + rows > w.n) {
            fail(op, "out must be BF16 [rows, N] with dim 0 contiguous, inside the weight's rows");
        }
        LinearSegment& s = pr.params.seg[i];
        s.codes          = static_cast<const std::uint8_t*>(w.qdata);
        s.scales         = static_cast<const float*>(w.scales);
        s.x              = static_cast<const __nv_bfloat16*>(x.data);
        s.out            = static_cast<__nv_bfloat16*>(o.data);
        s.out_stride     = o.nb[1] / 2;
        s.row_begin      = pj.row_begin;
        s.rows           = rows;
        s.k_per          = k_per;
        s.rows_per       = rows_per;
        s.kcells         = w.k / k_per;
        s.block_begin    = blocks;
        blocks += gemm ? (rows + kTn - 1) / kTn : (rows + kNarrowRows - 1) / kNarrowRows;
    }
    pr.params.segments = static_cast<std::int32_t>(projections.size());
    pr.params.columns  = static_cast<std::int32_t>(columns);
    pr.blocks          = blocks;
    return pr;
}

} // namespace
} // namespace inkling_detail

std::size_t inkling_attn_linear_workspace_bytes(std::span<const InklingProjection>, std::int32_t) {
    return 0; // both kernels stage in shared memory only
}

void inkling_attn_linear(std::span<const InklingProjection> projections, WorkspaceArena*, cudaStream_t stream) {
    using namespace inkling_detail;
    const char* op = "inkling_attn_linear";
    if (projections.empty() || projections[0].x == nullptr) { fail(op, "no projections"); }
    const std::int64_t n = projections[0].x->numel() / std::max<std::int64_t>(projections[0].x->ne[0], 1);
    const bool gemm      = n > 16;
    const Prepared pr    = prepare(projections, gemm, op);
    if (pr.params.columns <= 0) { return; }
    // One cell kind per launch: the accumulator chain is fixed at compile time.
    const bool per_row = pr.params.seg[0].k_per == pr.k;
    for (int i = 1; i < pr.params.segments; ++i) {
        if ((pr.params.seg[i].k_per == pr.k) != per_row) { fail(op, "projections of one call share their scale cell kind"); }
    }
    if (!gemm) {
        const pdl::LaunchConfig launch{dim3(static_cast<unsigned>(pr.blocks)), dim3(kNarrowWarps * 32), 0, stream};
        if (pr.k == 2304) {
            CUDA_CHECK(per_row ? pdl::launch_dependent(launch, narrow_kernel<2304, true>, pr.params)
                               : pdl::launch_dependent(launch, narrow_kernel<2304, false>, pr.params));
        } else {
            CUDA_CHECK(per_row ? pdl::launch_dependent(launch, narrow_kernel<3072, true>, pr.params)
                               : pdl::launch_dependent(launch, narrow_kernel<3072, false>, pr.params));
        }
        return;
    }
    const pdl::LaunchConfig launch{
        dim3(static_cast<unsigned>(pr.blocks), static_cast<unsigned>((pr.params.columns + kTm - 1) / kTm)),
        dim3(128), 0, stream};
    if (pr.k == 2304) {
        CUDA_CHECK(per_row ? pdl::launch_dependent(launch, gemm_kernel<2304, true>, pr.params)
                           : pdl::launch_dependent(launch, gemm_kernel<2304, false>, pr.params));
    } else {
        CUDA_CHECK(per_row ? pdl::launch_dependent(launch, gemm_kernel<3072, true>, pr.params)
                           : pdl::launch_dependent(launch, gemm_kernel<3072, false>, pr.params));
    }
}

} // namespace sinfer::ops
