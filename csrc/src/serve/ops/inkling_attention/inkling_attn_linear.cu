// Inkling-2-Mini FP8 attention projections, W8A16 (ENGINE-INKLING lane c; DESIGN-attention.md section 6,
// agent-EN 08:2x: per-channel and 1x128 scale cells first).
//
// Weights: the block route's FP8 Weight as lane b binds it -- e4m3 codes [n][k] row-major and an fp32 scale grid
// [n / rows_per][k / k_per] row-major, cell = (scale_ne[0] = k_per, scale_ne[1] = rows_per): per-channel (k, 1),
// 1x128 (128, 1), 128x128 (128, 128). Activations stay BF16 (never quantised); every code is widened exactly; the
// dot products accumulate in fp32 and each 128-column cell's partial is multiplied by its scale in fp32 before it
// joins the row total (one multiply after the whole row for a per-channel cell). Output rounded once to bf16.
//
// Two kernels, chosen by the column count N of the call: a GEMV for N <= 8 (decode, verify: bytes-bound, every
// code read once and converted once) and a tensor-core GEMM beyond (codes widened to bf16 in shared memory, bf16
// m16n8k16 MMA). One launch covers up to four projections that share K and N (a 3-input visit's q-side, K and V
// row ranges, each from its own input). M2 form, untuned.

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

__device__ __forceinline__ void codes8_to_float(uint2 raw, float (&w)[8]) {
    const std::uint32_t words[2] = {raw.x, raw.y};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const std::uint16_t pair = static_cast<std::uint16_t>(words[i >> 1] >> ((i & 1) * 16));
        const __half2_raw h      = __nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3);
        const __half2 hv         = *reinterpret_cast<const __half2*>(&h);
        w[2 * i]                 = __low2float(hv);
        w[2 * i + 1]             = __high2float(hv);
    }
}

// ---- GEMV: N <= 8 columns --------------------------------------------------------------------------------
// A CTA of 4 warps owns 8 rows of one segment, 2 a warp. Lane l reads codes [256 s + 8 l, + 8) of each row for
// every step s (all steps issued before any arithmetic: the whole row pair is in flight at once); 8 codes never
// straddle a 128-column cell.
inline constexpr int kGemvWarps       = 4;
inline constexpr int kGemvRowsPerWarp = 2;
inline constexpr int kGemvRowsPerCta  = kGemvWarps * kGemvRowsPerWarp;

template <int Steps, int Tokens>
__global__ __launch_bounds__(kGemvWarps * 32) void gemv_kernel(const LinearParams p) {
    constexpr int K = Steps * 256;
    pdl::wait_for_dependencies();
    const int block        = static_cast<int>(blockIdx.x);
    const LinearSegment& s = p.seg[segment_of(p, block)];
    const int warp         = static_cast<int>(threadIdx.x) >> 5;
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const int local0       = (block - s.block_begin) * kGemvRowsPerCta + warp * kGemvRowsPerWarp;

    uint2 raw[kGemvRowsPerWarp][Steps];
#pragma unroll
    for (int r = 0; r < kGemvRowsPerWarp; ++r) {
        const int local = min(local0 + r, s.rows - 1); // a row past the end re-reads the last row, writes nothing
        const std::uint8_t* row = s.codes + static_cast<std::int64_t>(s.row_begin + local) * K + lane * 8;
#pragma unroll
        for (int st = 0; st < Steps; ++st) { raw[r][st] = load_ldg<uint2>(row + st * 256); }
    }

    const bool per_row = s.k_per >= K;
    float acc[kGemvRowsPerWarp][Tokens];
#pragma unroll
    for (int r = 0; r < kGemvRowsPerWarp; ++r) {
#pragma unroll
        for (int t = 0; t < Tokens; ++t) { acc[r][t] = 0.0F; }
    }
#pragma unroll
    for (int st = 0; st < Steps; ++st) {
        const int k0 = st * 256 + lane * 8;
        float xv[Tokens][8];
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            const uint4 xr = load_ldg<uint4>(s.x + static_cast<std::int64_t>(t) * K + k0);
            const std::uint32_t xw[4] = {xr.x, xr.y, xr.z, xr.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f = bf16x2_bits_to_float2(xw[i]);
                xv[t][2 * i]     = f.x;
                xv[t][2 * i + 1] = f.y;
            }
        }
#pragma unroll
        for (int r = 0; r < kGemvRowsPerWarp; ++r) {
            float w[8];
            codes8_to_float(raw[r][st], w);
            float cell_scale = 1.0F;
            if (!per_row) {
                const int row = s.row_begin + min(local0 + r, s.rows - 1);
                cell_scale    = s.scales[static_cast<std::int64_t>(row / s.rows_per) * s.kcells + k0 / s.k_per];
            }
#pragma unroll
            for (int t = 0; t < Tokens; ++t) {
                float d = 0.0F;
#pragma unroll
                for (int i = 0; i < 8; ++i) { d = fmaf(w[i], xv[t][i], d); }
                acc[r][t] = per_row ? __fadd_rn(acc[r][t], d) : fmaf(cell_scale, d, acc[r][t]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < kGemvRowsPerWarp; ++r) {
        const int local = local0 + r;
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            const float total = warp_sum(acc[r][t]);
            if (lane == 0 && local < s.rows) {
                const int row = s.row_begin + local;
                const float y = per_row ? __fmul_rn(total, s.scales[row / s.rows_per]) : total;
                s.out[static_cast<std::int64_t>(t) * s.out_stride + local] = __float2bfloat16_rn(y);
            }
        }
    }
}

// ---- GEMM: N > 8 columns ---------------------------------------------------------------------------------
// CTA tile 64 weight rows x 64 columns, K in steps of 64 (two per 128-column cell). 4 warps in a 2 x 2 grid,
// each 32 rows x 32 columns (2 m16 x 4 n8 tiles). Codes and activations land with cp.async (two stages); the
// codes are widened into a swizzled bf16 tile; A = widened codes (row-major), B = activations ([n][k] = col-major
// k x n). A cell's fp32 partial is scaled into the accumulator when the cell ends.
inline constexpr int kTm = 64, kTn = 64, kTk = 64;

struct GemmSmem {
    std::uint8_t codes[2][kTm * kTk];      // 2 x 4 KiB
    __nv_bfloat16 x[2][kTn * kTk];         // 2 x 8 KiB, swizzled
    __nv_bfloat16 a[kTm * kTk];            // 8 KiB, swizzled
};

__device__ __forceinline__ int swz64(int row, int chunk) { return row * kTk + ((chunk ^ (row & 7)) << 3); }

template <int K, bool PerRow>
__global__ __launch_bounds__(128) void gemm_kernel(const LinearParams p) {
    __shared__ __align__(16) GemmSmem sm;
    pdl::wait_for_dependencies();
    const int block        = static_cast<int>(blockIdx.x);
    const LinearSegment& s = p.seg[segment_of(p, block)];
    const int tid          = static_cast<int>(threadIdx.x);
    const int warp         = tid >> 5;
    const int lane         = tid & 31;
    const int gid          = lane >> 2;
    const int lid          = lane & 3;
    const int wm           = warp >> 1; // warp row (32 rows)
    const int wn           = warp & 1;  // warp column (32 columns)
    const int local0       = (block - s.block_begin) * kTm;
    const int col0         = static_cast<int>(blockIdx.y) * kTn;
    constexpr int Steps    = K / kTk;

    const auto stage = [&](int step, int slot) {
        const int k0 = step * kTk;
        // codes: 64 rows x 64 B = 256 chunks of 16 B; activations: 64 columns x 128 B = 512 chunks.
        for (int c = tid; c < 256 + 512; c += 128) {
            if (c < 256) {
                const int row = c >> 2, part = c & 3;
                const int local = local0 + row;
                const bool ok   = local < s.rows;
                const std::uint8_t* src =
                    s.codes + static_cast<std::int64_t>(s.row_begin + (ok ? local : 0)) * K + k0 + part * 16;
                cp_async_zfill<16>(&sm.codes[slot][row * kTk + part * 16], src, ok ? 16 : 0);
            } else {
                const int cc = c - 256;
                const int col = cc >> 3, chunk = cc & 7;
                const bool ok = col0 + col < p.columns;
                const __nv_bfloat16* src =
                    s.x + static_cast<std::int64_t>(ok ? col0 + col : 0) * K + k0 + chunk * 8;
                cp_async_zfill<16>(&sm.x[slot][swz64(col, chunk)], src, ok ? 16 : 0);
            }
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
    // The four weight rows this thread's fragments touch (m tile mt, half h): local0 + 32 wm + 16 mt + gid + 8 h.
    const auto frag_row = [&](int mt, int h) { return local0 + 32 * wm + 16 * mt + gid + 8 * h; };

    stage(0, 0);
    cp_commit();
    if (Steps > 1) { stage(1, 1); }
    cp_commit();
    const unsigned a_base = smem_addr(sm.a);
    for (int st = 0; st < Steps; ++st) {
        const int slot = st & 1;
        cp_wait<1>();
        __syncthreads();
        // Widen this step's codes: 64 x 64 codes, 8 a chunk = 512 chunks.
        for (int c = tid; c < 512; c += 128) {
            const int row = c >> 3, chunk = c & 7;
            store_vec(&sm.a[swz64(row, chunk)], e4m3x8_to_bf16x8(load_vec<int2>(&sm.codes[slot][row * kTk + chunk * 8])));
        }
        __syncthreads();
        const unsigned x_base = smem_addr(sm.x[slot]);
#pragma unroll
        for (int ks = 0; ks < kTk / 16; ++ks) {
            unsigned af[2][4];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int row   = 32 * wm + 16 * mt + (lane & 7) + (((lane >> 3) & 1) << 3);
                const int chunk = 2 * ks + (lane >> 4);
                ldmatrix_x4(af[mt][0], af[mt][1], af[mt][2], af[mt][3],
                            a_base + static_cast<unsigned>(swz64(row, chunk) * 2));
            }
#pragma unroll
            for (int nt2 = 0; nt2 < 4; nt2 += 2) {
                const int col   = 32 * wn + nt2 * 8 + (lane & 7) + ((lane >> 4) << 3);
                const int chunk = 2 * ks + ((lane >> 3) & 1);
                unsigned b0, b1, b2, b3;
                ldmatrix_x4(b0, b1, b2, b3, x_base + static_cast<unsigned>(swz64(col, chunk) * 2));
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    float(&c0)[4] = PerRow ? acc[mt][nt2] : part[mt][nt2];
                    float(&c1)[4] = PerRow ? acc[mt][nt2 + 1] : part[mt][nt2 + 1];
                    mma_bf16(c0[0], c0[1], c0[2], c0[3], af[mt][0], af[mt][1], af[mt][2], af[mt][3], b0, b1);
                    mma_bf16(c1[0], c1[1], c1[2], c1[3], af[mt][0], af[mt][1], af[mt][2], af[mt][3], b2, b3);
                }
            }
        }
        __syncthreads(); // the a tile and this stage's slots are free
        if (st + 2 < Steps) { stage(st + 2, slot); }
        cp_commit();
        // A cell ends every k_per columns (1x128: every two steps): scale its partial into the total.
        if (!PerRow && ((st + 1) * kTk) % s.k_per == 0) {
            const int kb = (st * kTk) / s.k_per;
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int local = min(frag_row(mt, h), s.rows - 1);
                    const float sc =
                        s.scales[static_cast<std::int64_t>((s.row_begin + local) / s.rows_per) * s.kcells + kb];
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt) {
                        acc[mt][nt][2 * h]     = fmaf(sc, part[mt][nt][2 * h], acc[mt][nt][2 * h]);
                        acc[mt][nt][2 * h + 1] = fmaf(sc, part[mt][nt][2 * h + 1], acc[mt][nt][2 * h + 1]);
                        part[mt][nt][2 * h] = part[mt][nt][2 * h + 1] = 0.0F;
                    }
                }
            }
        }
    }
    cp_wait<0>();

#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int local = frag_row(mt, h);
            if (local >= s.rows) { continue; }
            const float sc = PerRow ? s.scales[(s.row_begin + local) / s.rows_per] : 1.0F;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int col = col0 + 32 * wn + 8 * nt + 2 * lid + e;
                    if (col >= p.columns) { continue; }
                    const float v = acc[mt][nt][2 * h + e];
                    s.out[static_cast<std::int64_t>(col) * s.out_stride + local] =
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
        blocks += gemm ? (rows + kTm - 1) / kTm : (rows + kGemvRowsPerCta - 1) / kGemvRowsPerCta;
    }
    pr.params.segments = static_cast<std::int32_t>(projections.size());
    pr.params.columns  = static_cast<std::int32_t>(columns);
    pr.blocks          = blocks;
    return pr;
}

template <int Steps>
void launch_gemv(const Prepared& pr, cudaStream_t stream) {
    const pdl::LaunchConfig launch{dim3(static_cast<unsigned>(pr.blocks)), dim3(kGemvWarps * 32), 0, stream};
    switch (pr.params.columns) {
    case 1: CUDA_CHECK(pdl::launch_dependent(launch, gemv_kernel<Steps, 1>, pr.params)); break;
    case 2: CUDA_CHECK(pdl::launch_dependent(launch, gemv_kernel<Steps, 2>, pr.params)); break;
    case 3: CUDA_CHECK(pdl::launch_dependent(launch, gemv_kernel<Steps, 3>, pr.params)); break;
    case 4: CUDA_CHECK(pdl::launch_dependent(launch, gemv_kernel<Steps, 4>, pr.params)); break;
    case 5: CUDA_CHECK(pdl::launch_dependent(launch, gemv_kernel<Steps, 5>, pr.params)); break;
    case 6: CUDA_CHECK(pdl::launch_dependent(launch, gemv_kernel<Steps, 6>, pr.params)); break;
    case 7: CUDA_CHECK(pdl::launch_dependent(launch, gemv_kernel<Steps, 7>, pr.params)); break;
    case 8: CUDA_CHECK(pdl::launch_dependent(launch, gemv_kernel<Steps, 8>, pr.params)); break;
    default: throw std::logic_error("inkling_attn_linear: GEMV past 8 columns");
    }
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
    const bool gemm      = n > 8;
    const Prepared pr    = prepare(projections, gemm, op);
    if (pr.params.columns <= 0) { return; }
    if (!gemm) {
        if (pr.k == 2304) { launch_gemv<9>(pr, stream); } else { launch_gemv<12>(pr, stream); }
        return;
    }
    const pdl::LaunchConfig launch{
        dim3(static_cast<unsigned>(pr.blocks), static_cast<unsigned>((pr.params.columns + kTn - 1) / kTn)),
        dim3(128), 0, stream};
    // One cell kind per launch: the accumulator layout is fixed at compile time.
    const bool per_row = pr.params.seg[0].k_per == pr.k;
    for (int i = 1; i < pr.params.segments; ++i) {
        if ((pr.params.seg[i].k_per == pr.k) != per_row) { fail(op, "projections of one call share their scale cell kind"); }
    }
    if (pr.k == 2304) {
        CUDA_CHECK(per_row ? pdl::launch_dependent(launch, gemm_kernel<2304, true>, pr.params)
                           : pdl::launch_dependent(launch, gemm_kernel<2304, false>, pr.params));
    } else {
        CUDA_CHECK(per_row ? pdl::launch_dependent(launch, gemm_kernel<3072, true>, pr.params)
                           : pdl::launch_dependent(launch, gemm_kernel<3072, false>, pr.params));
    }
}

} // namespace sinfer::ops
