#include "ops/linear/fp8_block/fp8_block.h"
#include "ops/linear/fp8_block/fp8_block_sm90_gemm.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/ggml/ggml_dispatch.h" // scratch_for: the engine-slot scratch, graph-safe

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sinfer::ops::detail::fp8_block {
namespace {

constexpr int kBlock        = 128; // the scale block, along rows and along k
constexpr int kGemvMaxTokens = 4;  // at most this many columns run the GEMV on exact activations
constexpr int kTileRows     = 64;  // rows per CTA of the tensor-core tile
constexpr int kTileCols     = 64;  // tokens per CTA
constexpr int kTileWarps    = 8;   // one n8 fragment per warp
constexpr int kTileStride   = kBlock + 16; // a staged row: 128 codes plus a pad that keeps ldmatrix conflict-free
constexpr int kTileStages   = 2;
constexpr int kBlocksPerSm  = 2;
constexpr std::size_t kAlign = 256;
// Hopper's CUTLASS kernel writes one packed output, so consecutive row ranges of a parent run as
// one GEMM into a staging plane and a split -- on rounds up to this many tokens, where separate
// launches leave most SMs idle and the plane is small.
constexpr int kStagedMaxTokens = 128;

/// Where a launch's rows land: up to four consecutive row ranges of one weight, each in its own
/// [rows, T] output -- a parent's q/k/v or q/k/gate/v projections as one launch. One output is
/// one segment. Every boundary is a whole 64-row tile.
struct Segments {
    static constexpr int kMax = 4;
    int count = 0;
    int end[kMax] = {};              // cumulative row ends, from the launch's first row
    __nv_bfloat16* out[kMax] = {};
    __host__ __device__ int rows() const { return end[count - 1]; }
};

struct Segment {
    __nv_bfloat16* out;
    int begin; // the segment's first row, from the launch's first row
    int rows;
};

__device__ __forceinline__ Segment segment_of(const Segments& s, int row) {
    int i = 0;
    while (i + 1 < s.count && row >= s.end[i]) { ++i; }
    const int begin = i == 0 ? 0 : s.end[i - 1];
    return {s.out[i], begin, s.end[i] - begin};
}

// ---- activations: E4M3 per token per 128, the recipe's convention ----
// The scale of (token, block) sits at token * kblocks + block, or block * scale_stride(tokens) +
// token when `kb_major` -- the layout Hopper's CUTLASS kernel reads its activation scales in (see
// fp8_block_sm90_gemm.h), each block's row padded to a multiple of four tokens so that it starts
// 16-byte aligned for the kernel's TMA. The tile kernel reads either.
__device__ __forceinline__ std::size_t act_scale_index(int token, int block, int tokens, int kblocks,
                                                       bool kb_major) {
    return kb_major ? static_cast<std::size_t>(block) * sm90_scale_stride(tokens) + token
                    : static_cast<std::size_t>(token) * kblocks + block;
}

// A warp per (token, 128-block): four values a lane in one 8-byte load, the block's absolute max
// by shuffles, four codes in one 4-byte store. The arithmetic is per element and unchanged from a
// thread per value, so the codes and scales are too; the old shape (a 128-thread CTA per block,
// scalar loads) ran a 2,048 x 5,120 activation at a quarter of the memory bandwidth.
constexpr int kQuantizeWarps = 8;

// A warp's (token, 128-block): its lane's four values in, the block's absolute max by shuffles,
// four codes in one 4-byte store.
__device__ __forceinline__ void quantize_warp_block(float2 lo, float2 hi, int lane, std::uint8_t* codes,
                                                    float* scale) {
    float amax = fmaxf(fmaxf(fabsf(lo.x), fabsf(lo.y)), fmaxf(fabsf(hi.x), fabsf(hi.y)));
    for (int o = 16; o > 0; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o)); }
    const float s       = amax > 0.0f ? amax / 448.0f : 1.0f;
    const float inverse = 1.0f / s;
    const auto c01 = __nv_cvt_float2_to_fp8x2(make_float2(lo.x * inverse, lo.y * inverse), __NV_SATFINITE, __NV_E4M3);
    const auto c23 = __nv_cvt_float2_to_fp8x2(make_float2(hi.x * inverse, hi.y * inverse), __NV_SATFINITE, __NV_E4M3);
    *reinterpret_cast<std::uint32_t*>(codes) = static_cast<std::uint32_t>(c01) | static_cast<std::uint32_t>(c23) << 16;
    if (lane == 0) { *scale = s; }
}

__global__ __launch_bounds__(kQuantizeWarps * 32) void quantize_blocks_kernel(
        const __nv_bfloat16* __restrict__ x, int k, int tokens, std::uint8_t* __restrict__ codes,
        float* __restrict__ scales, bool kb_major) {
    const int kblocks = k / kBlock;
    const long long item = static_cast<long long>(blockIdx.x) * kQuantizeWarps + (threadIdx.x >> 5);
    if (item >= static_cast<long long>(tokens) * kblocks) { return; }
    const int lane  = static_cast<int>(threadIdx.x & 31);
    const int token = static_cast<int>(item / kblocks);
    const int block = static_cast<int>(item % kblocks);
    const std::size_t at = static_cast<std::size_t>(token) * k + block * kBlock + lane * 4;
    const uint2 packed = *reinterpret_cast<const uint2*>(x + at);
    const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&packed.x));
    const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&packed.y));
    quantize_warp_block(lo, hi, lane, codes + at, scales + act_scale_index(token, block, tokens, kblocks, kb_major));
}

void launch_quantize_blocks(const __nv_bfloat16* x, int k, int tokens, std::uint8_t* codes,
                            float* scales, bool kb_major, cudaStream_t stream) {
    const long long items = static_cast<long long>(tokens) * (k / kBlock);
    const auto grid = static_cast<unsigned>((items + kQuantizeWarps - 1) / kQuantizeWarps);
    quantize_blocks_kernel<<<grid, kQuantizeWarps * 32, 0, stream>>>(x, k, tokens, codes, scales, kb_major);
    CUDA_CHECK(cudaGetLastError());
}

// silu(gate) * up from a [2k, T] plane -- each token's k gate rows, then its k up rows -- formed and
// quantised in one pass, for a down projection that never needs the BF16 activation. Each value is
// rounded to BF16 before it is quantised, as silu_mul's output is, so the codes and scales are the
// ones silu_mul then launch_quantize_blocks produce.
__global__ __launch_bounds__(kQuantizeWarps * 32) void swiglu_quantize_blocks_kernel(
        const __nv_bfloat16* __restrict__ packed, int k, int tokens, float limit,
        std::uint8_t* __restrict__ codes, float* __restrict__ scales, bool kb_major) {
    const int kblocks = k / kBlock;
    const long long item = static_cast<long long>(blockIdx.x) * kQuantizeWarps + (threadIdx.x >> 5);
    if (item >= static_cast<long long>(tokens) * kblocks) { return; }
    const int lane  = static_cast<int>(threadIdx.x & 31);
    const int token = static_cast<int>(item / kblocks);
    const int block = static_cast<int>(item % kblocks);
    const int row   = block * kBlock + lane * 4;
    const __nv_bfloat16* column = packed + static_cast<std::size_t>(token) * 2 * k;
    const uint2 g = *reinterpret_cast<const uint2*>(column + row);
    const uint2 u = *reinterpret_cast<const uint2*>(column + k + row);
    const auto act = [limit](std::uint32_t gate2, std::uint32_t up2) {
        const float2 gf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gate2));
        const float2 uf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&up2));
        return __bfloat1622float2(__floats2bfloat162_rn(swiglu_clamped(gf.x, uf.x, limit),
                                                        swiglu_clamped(gf.y, uf.y, limit)));
    };
    quantize_warp_block(act(g.x, u.x), act(g.y, u.y), lane,
                        codes + static_cast<std::size_t>(token) * k + row,
                        scales + act_scale_index(token, block, tokens, kblocks, kb_major));
}

// ---- decode: a warp per row on exact BF16 activations, the block scale per lane ----
__device__ __forceinline__ float2 e4m3x2_to_float2(std::uint16_t bits) {
    __nv_fp8x2_e4m3 v;
    v.__x = bits;
    return static_cast<float2>(v);
}

/// The scale grid's cell, [k per scale, rows per scale]: 128 x 128 for the block export, k x 1
/// for a per-channel one. A scale index is (row / rows_per) * (k / k_per) + kofs / k_per.
struct ScaleCell {
    int k_per;
    int rows_per;
};

// A weight word read once per call: through the non-coherent path, not kept in L1. Not volatile,
// so the compiler may issue several before their first use.
__device__ __forceinline__ uint4 load_streaming(const std::uint8_t* p) {
    uint4 v;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(p));
    return v;
}

// The decode GEMV: a CTA of kGemvWarps warps takes kGemvRows rows and splits their K, warp w
// reading the 512-code chunks w, w + kGemvWarps, ... (kGemvUnroll chunks of each row in flight);
// a lane's 16 codes of a chunk sit inside one 128-block, so each chunk is scaled once. The CTA
// then sums its warps' partials in warp order. Against a warp per whole row (and the scale
// cell's runtime divisions), one token over the q/k/v, o, gate/up and down shapes of 8B and 27B
// models takes 24 % less time on an H100 (gate/up 24576 x 4096: 47.1 -> 34.2 us, 2.9 TB/s).
// Past one token the kernel is bound by instructions, not bytes: each code is converted once a
// row and each activation once a chunk (shared by the CTA's rows), not once per (row, token)
// pair, with the same sums in the same order: on an H100 four tokens take 31-47 % less time
// over the same shapes (down 4096 x 12288: 42.6 -> 28.8 us), two tokens 15-26 % less.
constexpr int kGemvRows   = 2;
constexpr int kGemvWarps  = 2;
constexpr int kGemvUnroll = 2;
constexpr int kGemvChunk  = 512; // codes a warp reads per step: 32 lanes x 16
static_assert(kTileRows % kGemvRows == 0, "a launch's rows are whole 64-row tiles");

// A lane's 16 codes as floats, in code order.
__device__ __forceinline__ void codes16(const uint4 packed, float (&w)[16]) {
    const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 lo = e4m3x2_to_float2(static_cast<std::uint16_t>(words[i] & 0xffffu));
        const float2 hi = e4m3x2_to_float2(static_cast<std::uint16_t>(words[i] >> 16));
        w[4 * i]     = lo.x;
        w[4 * i + 1] = lo.y;
        w[4 * i + 2] = hi.x;
        w[4 * i + 3] = hi.y;
    }
}

// One token's 16 activations as floats. VecX: they are 16-byte aligned and come in two vector
// loads.
template <bool VecX>
__device__ __forceinline__ void acts16(const __nv_bfloat16* xp, float (&v)[16]) {
    std::uint32_t xw[8];
    if constexpr (VecX) {
        const uint4 lo = *reinterpret_cast<const uint4*>(xp);
        const uint4 hi = *reinterpret_cast<const uint4*>(xp + 8);
        xw[0] = lo.x; xw[1] = lo.y; xw[2] = lo.z; xw[3] = lo.w;
        xw[4] = hi.x; xw[5] = hi.y; xw[6] = hi.z; xw[7] = hi.w;
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) { xw[i] = *reinterpret_cast<const std::uint32_t*>(xp + 2 * i); }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xw[i]));
        v[2 * i]     = f.x;
        v[2 * i + 1] = f.y;
    }
}

/// PerRow: the per-channel cell (one scale a row); otherwise the 128 x 128 block cell, whose
/// indices are shifts -- a division by the cell's runtime size per load cost the short rows a
/// quarter of their time.
template <int Tokens, bool VecX, bool PerRow>
__global__ __launch_bounds__(kGemvWarps * 32) void gemv_kernel(const std::uint8_t* __restrict__ codes,
                                                               const float* __restrict__ scales,
                                                               const __nv_bfloat16* __restrict__ x,
                                                               const Segments segments, int k, int tokens,
                                                               bool accumulate) {
    __shared__ float partials[kGemvWarps][kGemvRows][Tokens];
    const int warp = static_cast<int>(threadIdx.x >> 5);
    const int lane = static_cast<int>(threadIdx.x & 31);
    const int row0 = static_cast<int>(blockIdx.x) * kGemvRows;
    const auto scale_of = [&](int r, int base) {
        return PerRow ? scales[row0 + r]
                      : scales[static_cast<std::size_t>((row0 + r) >> 7) * (k >> 7) + (base >> 7)];
    };
    float acc[kGemvRows][Tokens];
#pragma unroll
    for (int r = 0; r < kGemvRows; ++r) {
#pragma unroll
        for (int t = 0; t < Tokens; ++t) { acc[r][t] = 0.0f; }
    }
    const auto chunk = [&](const int base, const uint4 (&packed)[kGemvRows]) {
        float xv[Tokens][16];
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            if (t < tokens) { acts16<VecX>(x + static_cast<std::size_t>(t) * k + base, xv[t]); }
        }
#pragma unroll
        for (int r = 0; r < kGemvRows; ++r) {
            const float s = scale_of(r, base);
            float w[16];
            codes16(packed[r], w);
#pragma unroll
            for (int t = 0; t < Tokens; ++t) {
                if (t < tokens) {
                    float part = 0.0f; // the 16 products in code order, then scaled once
#pragma unroll
                    for (int i = 0; i < 16; ++i) { part = fmaf(w[i], xv[t][i], part); }
                    acc[r][t] = fmaf(s, part, acc[r][t]);
                }
            }
        }
    };
    const auto load = [&](const int base, uint4 (&packed)[kGemvRows]) {
#pragma unroll
        for (int r = 0; r < kGemvRows; ++r) {
            packed[r] = load_streaming(codes + static_cast<std::size_t>(row0 + r) * k + base);
        }
    };
    constexpr int kStride = kGemvWarps * kGemvChunk;
    int base = (warp * 32 + lane) * 16;
    for (; base + (kGemvUnroll - 1) * kStride < k; base += kGemvUnroll * kStride) {
        uint4 packed[kGemvUnroll][kGemvRows];
#pragma unroll
        for (int u = 0; u < kGemvUnroll; ++u) { load(base + u * kStride, packed[u]); }
#pragma unroll
        for (int u = 0; u < kGemvUnroll; ++u) { chunk(base + u * kStride, packed[u]); }
    }
    for (; base < k; base += kStride) {
        uint4 packed[kGemvRows];
        load(base, packed);
        chunk(base, packed);
    }
#pragma unroll
    for (int r = 0; r < kGemvRows; ++r) {
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            float v = acc[r][t];
            for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, o); }
            if (lane == 0) { partials[warp][r][t] = v; }
        }
    }
    __syncthreads();
    const int item = static_cast<int>(threadIdx.x);
    if (item < kGemvRows * Tokens) {
        const int r = item / Tokens, t = item % Tokens;
        if (t < tokens) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < kGemvWarps; ++w) { v += partials[w][r][t]; }
            const int row       = row0 + r;
            const Segment seg   = segment_of(segments, row);
            __nv_bfloat16* slot = seg.out + static_cast<std::size_t>(t) * seg.rows + (row - seg.begin);
            if (accumulate) { v += __bfloat162float(*slot); }
            *slot = __float2bfloat16_rn(v);
        }
    }
}

template <bool VecX, bool PerRow>
void launch_gemv(const std::uint8_t* codes, const float* scales, const __nv_bfloat16* x,
                 const Segments& segments, int k, int tokens, bool accumulate, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(segments.rows() / kGemvRows));
    const auto launch = [&](auto kernel) {
        kernel<<<grid, kGemvWarps * 32, 0, stream>>>(codes, scales, x, segments, k, tokens, accumulate);
    };
    switch (tokens) {
    case 1: launch(gemv_kernel<1, VecX, PerRow>); break;
    case 2: launch(gemv_kernel<2, VecX, PerRow>); break;
    case 3: launch(gemv_kernel<3, VecX, PerRow>); break;
    default: launch(gemv_kernel<4, VecX, PerRow>); break;
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---- narrow rounds: tensor cores on exact BF16 activations ----
// Two to sixteen tokens (Hopper; see mma_enabled): each code becomes a BF16 in registers
// (exactly: every E4M3 value is one) and m16n8k16 MMAs take the products against the BF16
// activations as they are, so a round's error is the GEMV's, not the quantised tile's (on an
// H100, sampled against a double reference: 1.5e-3, the BF16 output's own rounding, against the
// tile's 2.4e-2), and nothing quantises the activations first. A CTA takes 16 rows and splits K
// over W warps by 128-blocks (warp w: blocks w, w + W, ..., two in flight; W per launch in
// launch_mma_split). Lane (g = lane / 4, t = lane % 4) holds, of each block, codes [16t, 16t + 16)
// and [64 + 16t, 80 + 16t) of rows g and g + 8 and the same 32 activations of token g of each
// 8-token group: the MMA's k order permuted alike on both sides, so each load is 16 contiguous
// bytes and each row's 128 codes two fully used 64-byte runs. A block's eight MMAs sum unscaled,
// then fold in at the block's scale; the CTA sums its warps' partials in warp order. At four
// tokens over the q/k/v, o, gate/up and down shapes of an 8B model on an H100 (us): 11.5 / 8.6 /
// 37.8 / 21.1, against the GEMV's 14.4 / 11.1 / 48.7 / 28.6 and the narrow CUTLASS tile's 18.2 /
// 17.7 / 43.0 / 42.7 with its quantise (its grid has no K split, so a 4096-row weight gets 32
// CTAs); flat from one to eight tokens, 5-10 % slower at sixteen.
constexpr int kMmaMaxTokens = 16;
constexpr int kMmaRows      = 16;
constexpr int kMmaUnroll    = 2;
static_assert(kTileRows % kMmaRows == 0 && kBlock % kMmaRows == 0, "a CTA's rows share one scale row");

__device__ __forceinline__ unsigned e4m3x2_to_bf16x2(std::uint32_t two) {
    const float2 f = e4m3x2_to_float2(static_cast<std::uint16_t>(two & 0xffffu));
    const __nv_bfloat162 v = __floats2bfloat162_rn(f.x, f.y); // exact
    return *reinterpret_cast<const unsigned*>(&v);
}

__device__ __forceinline__ uint4 load_activations(const __nv_bfloat16* p) {
    uint4 v;
    asm("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ std::uint32_t word_of(const uint4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

/// NT: 8-token groups; W: the warps K is split over. VecX: x is 16-byte aligned (eight
/// activations a load, else two a load); PerRow: one scale a row, else the 128 x 128 cell.
template <int NT, int W, bool VecX, bool PerRow>
__global__ __launch_bounds__(W * 32) void mma_kernel(const std::uint8_t* __restrict__ codes,
                                                     const float* __restrict__ scales,
                                                     const __nv_bfloat16* __restrict__ x,
                                                     const Segments segments, int k, int tokens,
                                                     bool accumulate) {
    __shared__ float partials[W][kMmaRows][NT * 8 + 1];
    const int warp = static_cast<int>(threadIdx.x >> 5);
    const int lane = static_cast<int>(threadIdx.x & 31);
    const int g = lane >> 2, t = lane & 3;
    const int row0    = static_cast<int>(blockIdx.x) * kMmaRows;
    const int kblocks = k >> 7;
    const float* block_scales = scales + static_cast<std::size_t>(row0 >> 7) * kblocks;
    const std::uint8_t* w_lane = codes + static_cast<std::size_t>(row0 + g) * k + 16 * t;
    const std::size_t eight_rows = static_cast<std::size_t>(8) * k;
    const __nv_bfloat16* x_lane[NT];
    bool live[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const int token = 8 * nt + g;
        live[nt]   = token < tokens;
        x_lane[nt] = x + static_cast<std::size_t>(live[nt] ? token : 0) * k + 16 * t;
    }
    const auto activations = [&](const __nv_bfloat16* p) {
        if constexpr (VecX) {
            return load_activations(p);
        } else {
            const auto* words = reinterpret_cast<const std::uint32_t*>(p);
            return make_uint4(words[0], words[1], words[2], words[3]);
        }
    };
    float acc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) { acc[nt][e] = 0.0f; }
    }
    for (int kb0 = warp; kb0 < kblocks; kb0 += W * kMmaUnroll) {
        uint4 a[kMmaUnroll][2][2]; // [block][row g, g + 8][codes 16t.., 64 + 16t..]
        uint4 b[kMmaUnroll][NT][4]; // [block][group][activations 16t.. (two), 64 + 16t.. (two)]
        float s[kMmaUnroll];
#pragma unroll
        for (int u = 0; u < kMmaUnroll; ++u) {
            const int kb = kb0 + u * W;
            if (kb < kblocks) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
#pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        a[u][h][q] = load_streaming(w_lane + h * eight_rows + kb * kBlock + 64 * q);
                    }
                }
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        b[u][nt][i] = live[nt] ? activations(x_lane[nt] + kb * kBlock + 64 * (i >> 1) + 8 * (i & 1))
                                               : make_uint4(0, 0, 0, 0);
                    }
                }
                s[u] = PerRow ? 1.0f : block_scales[kb]; // a per-row scale goes on at the end
            }
        }
#pragma unroll
        for (int u = 0; u < kMmaUnroll; ++u) {
            if (kb0 + u * W < kblocks) {
                float blk[NT][4];
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) { blk[nt][e] = 0.0f; }
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    // codes 4j..4j+3 of the lane's 32: rows g and g + 8, k (2t, 2t+1) and (2t+8, 2t+9)
                    const std::uint32_t w0 = word_of(a[u][0][j >> 2], j & 3);
                    const std::uint32_t w8 = word_of(a[u][1][j >> 2], j & 3);
                    const unsigned a0 = e4m3x2_to_bf16x2(w0), a2 = e4m3x2_to_bf16x2(w0 >> 16);
                    const unsigned a1 = e4m3x2_to_bf16x2(w8), a3 = e4m3x2_to_bf16x2(w8 >> 16);
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        mma_bf16(blk[nt][0], blk[nt][1], blk[nt][2], blk[nt][3], a0, a1, a2, a3,
                                 word_of(b[u][nt][j >> 1], (2 * j) & 3), word_of(b[u][nt][j >> 1], (2 * j + 1) & 3));
                    }
                }
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) { acc[nt][e] = fmaf(s[u], blk[nt][e], acc[nt][e]); }
                }
            }
        }
    }
    // c0, c1: row g, tokens 2t and 2t + 1 of the group; c2, c3: row g + 8
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        partials[warp][g][8 * nt + 2 * t]         = acc[nt][0];
        partials[warp][g][8 * nt + 2 * t + 1]     = acc[nt][1];
        partials[warp][g + 8][8 * nt + 2 * t]     = acc[nt][2];
        partials[warp][g + 8][8 * nt + 2 * t + 1] = acc[nt][3];
    }
    __syncthreads();
    for (int item = static_cast<int>(threadIdx.x); item < kMmaRows * NT * 8; item += W * 32) {
        const int token = item / kMmaRows, r = item % kMmaRows;
        if (token < tokens) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < W; ++w) { v += partials[w][r][token]; }
            if constexpr (PerRow) { v *= scales[row0 + r]; } // the blocks summed unscaled
            const int row       = row0 + r;
            const Segment seg   = segment_of(segments, row);
            __nv_bfloat16* slot = seg.out + static_cast<std::size_t>(token) * seg.rows + (row - seg.begin);
            if (accumulate) { v += __bfloat162float(*slot); }
            *slot = __float2bfloat16_rn(v);
        }
    }
}

// ---- prefill: a 64-row x 64-token tile, the block scale applied once per 128 of K ----
struct TileSmem {
    alignas(16) std::uint8_t W[kTileStages][kTileRows * kTileStride];
    alignas(16) std::uint8_t X[kTileStages][kTileCols * kTileStride];
    float Sw[kTileStages][kTileRows];
    float Sx[kTileStages][kTileCols];
};

template <bool Accumulate>
__global__ __launch_bounds__(kTileWarps * 32, kBlocksPerSm) void tile_kernel(
    const std::uint8_t* __restrict__ w_codes, const float* __restrict__ w_scales, ScaleCell cell,
    const std::uint8_t* __restrict__ x_codes, const float* __restrict__ x_scales,
    const Segments segments, int k, int tokens, bool kb_major) {
    __shared__ TileSmem sm;
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int kblocks    = k / kBlock;
    const int kcells     = k / cell.k_per;
    const int row_blocks = segments.rows() / kTileRows;
    const int col_blocks = (tokens + kTileCols - 1) / kTileCols;
    const int total_work = row_blocks * col_blocks;
    for (int work = static_cast<int>(blockIdx.x); work < total_work; work += static_cast<int>(gridDim.x)) {
        const int row_block = work / col_blocks;
        const int col_block = work - row_block * col_blocks;
        const int row0      = row_block * kTileRows;
        const int col0      = col_block * kTileCols;
        const int cols      = tokens - col0 < kTileCols ? tokens - col0 : kTileCols;

        auto stage = [&](int slot, int kb) {
            const int k0 = kb * kBlock;
            // 64 rows x 8 chunks of 16 codes, twice: the weight tile and the activation tile
            for (int item = tid; item < kTileRows * 8; item += kTileWarps * 32) {
                const int r = item >> 3, c = item & 7;
                cp_async<16, Cache::cg>(&sm.W[slot][r * kTileStride + 16 * c],
                                        w_codes + static_cast<std::size_t>(row0 + r) * k + k0 + 16 * c);
            }
            for (int item = tid; item < kTileCols * 8; item += kTileWarps * 32) {
                const int t = item >> 3, c = item & 7;
                const bool valid = t < cols;
                cp_async_zfill<16, Cache::cg>(&sm.X[slot][t * kTileStride + 16 * c],
                                              x_codes + static_cast<std::size_t>(valid ? col0 + t : 0) * k + k0 + 16 * c,
                                              valid ? 16 : 0);
            }
            if (tid < kTileRows) {
                sm.Sw[slot][tid] = w_scales[static_cast<std::size_t>((row0 + tid) / cell.rows_per) * kcells + (kb * kBlock) / cell.k_per];
            } else if (tid < kTileRows + kTileCols) {
                const int t = tid - kTileRows;
                sm.Sx[slot][t] = t < cols ? x_scales[act_scale_index(col0 + t, kb, tokens, kblocks, kb_major)] : 0.0f;
            }
        };

        float acc[4][4];
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
            for (int e = 0; e < 4; ++e) { acc[mi][e] = 0.0f; }
        }
#pragma unroll
        for (int s = 0; s < kTileStages; ++s) {
            if (s < kblocks) { stage(s, s); }
            cp_commit();
        }
#pragma unroll 1
        for (int kb = 0; kb < kblocks; ++kb) {
            const int slot = kb & 1;
            cp_wait<kTileStages - 1>();
            __syncthreads();
            if (warp * 8 < cols) {
                float blk[4][4];
#pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) { blk[mi][e] = 0.0f; }
                }
#pragma unroll
                for (int g = 0; g < kBlock / 32; ++g) {
                    unsigned b0, b1;
                    {
                        const int token = warp * 8 + (lane & 7);
                        const int cofs  = 32 * g + ((lane >> 3) & 1) * 16;
                        ldmatrix_x2(b0, b1, smem_addr(&sm.X[slot][token * kTileStride + cofs]));
                    }
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        unsigned a[4];
                        const int local = mi * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                        const int cofs  = 32 * g + (lane >> 4) * 16;
                        ldmatrix_x4(a[0], a[1], a[2], a[3], smem_addr(&sm.W[slot][local * kTileStride + cofs]));
                        mma_fp8_e4m3(blk[mi][0], blk[mi][1], blk[mi][2], blk[mi][3], a[0], a[1], a[2], a[3], b0, b1);
                    }
                }
                const int c0 = warp * 8 + 2 * lid;
                const float sx0 = sm.Sx[slot][c0], sx1 = sm.Sx[slot][c0 + 1];
#pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    const int r0 = mi * 16 + gid, r1 = r0 + 8;
                    const float sw0 = sm.Sw[slot][r0], sw1 = sm.Sw[slot][r1];
                    acc[mi][0] = fmaf(blk[mi][0], sw0 * sx0, acc[mi][0]);
                    acc[mi][1] = fmaf(blk[mi][1], sw0 * sx1, acc[mi][1]);
                    acc[mi][2] = fmaf(blk[mi][2], sw1 * sx0, acc[mi][2]);
                    acc[mi][3] = fmaf(blk[mi][3], sw1 * sx1, acc[mi][3]);
                }
            }
            __syncthreads();
            const int next = kb + kTileStages;
            if (next < kblocks) { stage(slot, next); }
            cp_commit();
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * 8 < cols) {
            const Segment seg = segment_of(segments, row0); // a tile never straddles two
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int r0 = row0 - seg.begin + mi * 16 + gid, r1 = r0 + 8;
                const int lc0 = warp * 8 + 2 * lid, lc1 = lc0 + 1;
                auto store = [&](int col, int row, float value) {
                    __nv_bfloat16& slot = seg.out[static_cast<std::size_t>(col) * seg.rows + row];
                    if constexpr (Accumulate) { value += __bfloat162float(slot); }
                    slot = __float2bfloat16_rn(value);
                };
                if (lc0 < cols) { store(col0 + lc0, r0, acc[mi][0]); store(col0 + lc0, r1, acc[mi][2]); }
                if (lc1 < cols) { store(col0 + lc1, r0, acc[mi][1]); store(col0 + lc1, r1, acc[mi][3]); }
            }
        }
        __syncthreads();
    }
}

// The staged plane [rows, T] of a chained launch into its segments, 16 bytes a thread: every
// boundary is a whole 64-row tile, so no vector straddles two outputs.
__global__ void split_segments_kernel(const __nv_bfloat16* __restrict__ staged, const Segments segments,
                                      int tokens) {
    const int vectors = segments.rows() / 8;
    const long long item = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (item >= static_cast<long long>(vectors) * tokens) { return; }
    const int token = static_cast<int>(item / vectors);
    const int row   = static_cast<int>(item % vectors) * 8;
    const Segment seg = segment_of(segments, row);
    *reinterpret_cast<uint4*>(seg.out + static_cast<std::size_t>(token) * seg.rows + (row - seg.begin)) =
        *reinterpret_cast<const uint4*>(staged + static_cast<std::size_t>(token) * segments.rows() + row);
}

int sm_count() {
    static const int sms = [] {
        int device = 0, count = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
        return count;
    }();
    return sms;
}

int persistent_blocks() { return kBlocksPerSm * sm_count(); }

template <int NT, int W, bool VecX, bool PerRow>
int mma_blocks_per_sm() {
    static const int blocks = [] {
        int count = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&count, mma_kernel<NT, W, VecX, PerRow>, W * 32, 0));
        return std::max(count, 1);
    }();
    return blocks;
}

// K over eight warps when one wave holds every CTA, over four when even four make two waves or
// more; otherwise the split whose last wave leaves fewer SMs idle, four on a near tie. On an
// H100 that picks eight for 4096-row weights (o, down: 8.6 against 9.7 us at 4096 x 4096) and
// four for 5120 x 17408 (36.3 against 41.4: eight make 1.2 waves).
template <int NT, bool VecX, bool PerRow>
void launch_mma_split(const std::uint8_t* codes, const float* scales, const __nv_bfloat16* x,
                const Segments& segments, int k, int tokens, bool accumulate, cudaStream_t stream) {
    const int ctas = segments.rows() / kMmaRows;
    const double waves8 = double(ctas) / (sm_count() * mma_blocks_per_sm<NT, 8, VecX, PerRow>());
    const double waves4 = double(ctas) / (sm_count() * mma_blocks_per_sm<NT, 4, VecX, PerRow>());
    const auto filled = [](double waves) { return waves / std::ceil(waves); };
    const bool eight = waves8 <= 1.0 || (waves4 < 2.0 && filled(waves8) > filled(waves4) + 0.05);
    if (eight) {
        mma_kernel<NT, 8, VecX, PerRow><<<ctas, 8 * 32, 0, stream>>>(codes, scales, x, segments, k, tokens, accumulate);
    } else {
        mma_kernel<NT, 4, VecX, PerRow><<<ctas, 4 * 32, 0, stream>>>(codes, scales, x, segments, k, tokens, accumulate);
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_mma(const std::uint8_t* codes, const float* scales, const __nv_bfloat16* x,
                const Segments& segments, int k, int tokens, bool accumulate, bool per_row, cudaStream_t stream) {
    const bool vec_x = (reinterpret_cast<std::uintptr_t>(x) & 15u) == 0;
    const auto launch = tokens <= 8
        ? (vec_x ? (per_row ? launch_mma_split<1, true, true> : launch_mma_split<1, true, false>)
                 : (per_row ? launch_mma_split<1, false, true> : launch_mma_split<1, false, false>))
        : (vec_x ? (per_row ? launch_mma_split<2, true, true> : launch_mma_split<2, true, false>)
                 : (per_row ? launch_mma_split<2, false, true> : launch_mma_split<2, false, false>));
    launch(codes, scales, x, segments, k, tokens, accumulate, stream);
}

std::size_t codes_bytes(std::int32_t k, std::int32_t tokens) noexcept {
    return (static_cast<std::size_t>(k) * tokens + kAlign - 1) / kAlign * kAlign;
}

void require_x_out(const Tensor& x, std::int32_t k, const Tensor& out, std::int32_t n, const char* op) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16 || !x.is_contiguous() || !out.is_contiguous() ||
        x.ne[0] != k || out.ne[0] != n || x.ne[1] != out.ne[1] || x.ne[1] <= 0) {
        throw std::invalid_argument(std::string(op) + ": x [k, T] and out [n, T] must be contiguous BF16");
    }
}

int device_cc() noexcept {
    static const int cc = [] {
        int device = 0, major = 0, minor = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
            return 0;
        }
        return major * 10 + minor;
    }();
    return cc;
}

// The tensor-core narrow kernel takes the rounds past the GEMV's always-widths up to
// kMmaMaxTokens: on Hopper by default (measured there; elsewhere the GEMV and the engine's tile
// keep them).
// SUROGATE_SERVE_FP8_BLOCK_MMA=0 turns it off, =1 on, on any GPU.
bool mma_enabled() noexcept {
    static const int forced = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_FP8_BLOCK_MMA");
        if (raw == nullptr) { return -1; }
        return std::string_view(raw) == "0" ? 0 : std::string_view(raw) == "1" ? 1 : -1;
    }();
    return forced >= 0 ? forced == 1 : device_cc() == 90;
}

// Rounds up to this many tokens run the GEMV on every weight: one where the narrow kernel takes
// the next widths, three on Hopper without it, four elsewhere; SUROGATE_SERVE_FP8_BLOCK_GEMV_TOKENS
// =N (0 to 4) sets it. Alone on an H100 the GEMV reads one and two tokens faster (34.0 against
// 37.5 us at 24576 x 4096), but in a decode round its two-token kernel ran 41 % slower than alone
// while the narrow kernel ran within 5 % of its own: Qwen3-8B-FP8 decodes 212 tok/s a stream with
// two users against 184 on the GEMV, Qwen3.6-27B-FP8 75 against 74. One token stays: the 27B
// decodes 80 tok/s on the GEMV against 78 (the 8B 210 against 217).
int gemv_always_tokens() noexcept {
    static const int forced = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_FP8_BLOCK_GEMV_TOKENS");
        if (raw == nullptr || raw[0] < '0' || raw[0] > '0' + kGemvMaxTokens || raw[1] != '\0') { return -1; }
        return raw[0] - '0';
    }();
    if (forced >= 0) { return forced; }
    if (mma_enabled()) { return 1; }
    return sm90_gemm_available() ? 3 : kGemvMaxTokens;
}

bool mma_serves(std::int32_t tokens) noexcept {
    return mma_enabled() && tokens > gemv_always_tokens() && tokens <= kMmaMaxTokens;
}

bool block_cell(const Weight& w) noexcept { return w.scale_ne[0] == kBlock && w.scale_ne[1] == kBlock; }

// Past its always-widths the GEMV gives way to the narrow kernel where that runs. Without it
// (SUROGATE_SERVE_FP8_BLOCK_MMA=0): the GEMV's work grows with its columns, and Hopper's narrow
// CUTLASS tile reads the weight once for up to 16 tokens, on activations quantised per token per
// 128 as every wider round's are. On an H100 the GEMV is the faster to three tokens for every
// shape and at four below 96 row blocks of
// 128 (an 8B model's q/k/v 14.8 us against 18.3, the down projections of 8B and 27B models 29-44
// against 44-60), while wider weights move to the tile at four (24576 x 4096: 48.9 us on the
// GEMV, 44.2 quantised on the tile; 12288 x 5120 about even). A per-channel weight has no CUTLASS
// route and keeps the GEMV to four tokens.
bool gemv_serves(std::int32_t rows, std::int32_t tokens, bool block) noexcept {
    if (tokens > kGemvMaxTokens) { return false; }
    if (tokens <= gemv_always_tokens()) { return true; }
    if (mma_enabled()) { return false; }
    return !block || rows / kBlock < 96;
}

// Whether a round reads x's BF16 activations as they are -- the GEMV or the narrow kernel --
// rather than quantised planes.
bool exact_serves(std::int32_t rows, std::int32_t tokens, bool block) noexcept {
    return gemv_serves(rows, tokens, block) || mma_serves(tokens);
}

/// One row range of one weight against every column of x, [k, tokens] BF16, written or
/// accumulated into the segments' outputs. `prepared` holds x's activation planes already (x is
/// then not read past the GEMV widths). A chained launch (more than one segment) on Hopper's
/// CUTLASS kernel needs `staging`, [rows, T] BF16, for the kernel's packed output.
void run(const __nv_bfloat16* xin, std::int32_t tokens, const Weight& w, std::int32_t row_begin,
         const Segments& segments, bool accumulate, WorkspaceArena* workspace, cudaStream_t stream,
         const char* op, std::byte* prepared = nullptr, __nv_bfloat16* staging = nullptr,
         std::int32_t route_rows = 0) {
    require_fp8_block_weight(w, op);
    const std::int32_t rows = segments.rows();
    if (row_begin < 0 || rows <= 0 || row_begin + rows > w.n || (row_begin % w.scale_ne[1]) != 0 ||
        (row_begin % kTileRows) != 0 || (rows % kTileRows) != 0) {
        throw std::invalid_argument(std::string(op) + ": row range must start on a scale row and be whole 64-row tiles");
    }
    const std::int32_t k = w.k, kblocks = k / kBlock;
    const ScaleCell cell{w.scale_ne[0], w.scale_ne[1]};
    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata) + static_cast<std::size_t>(row_begin) * k;
    const auto* scales = static_cast<const float*>(w.scales) +
                         static_cast<std::size_t>(row_begin / cell.rows_per) * (k / cell.k_per);
    // A parent's ranges run apart take the route the whole parent takes, so their bits match it.
    if (gemv_serves(route_rows > 0 ? route_rows : rows, tokens, cell.k_per == kBlock && cell.rows_per == kBlock)) {
        const bool vec_x   = (reinterpret_cast<std::uintptr_t>(xin) & 15u) == 0;
        const bool per_row = cell.rows_per == 1;
        const auto launch  = vec_x ? (per_row ? launch_gemv<true, true> : launch_gemv<true, false>)
                                   : (per_row ? launch_gemv<false, true> : launch_gemv<false, false>);
        launch(codes, scales, xin, segments, k, tokens, accumulate, stream);
        return;
    }
    if (mma_serves(tokens)) {
        launch_mma(codes, scales, xin, segments, k, tokens, accumulate, cell.rows_per == 1, stream);
        return;
    }
    const std::size_t bytes = workspace_bytes(k, tokens);
    auto scope              = workspace != nullptr ? std::optional(workspace->scope()) : std::nullopt;
    std::byte* scratch      = prepared != nullptr ? prepared : workspace != nullptr
                                  ? static_cast<std::byte*>(workspace->alloc_bytes(bytes, kAlign).data)
                                  : static_cast<std::byte*>(ggml::scratch_for(bytes, stream));
    auto* x_codes  = reinterpret_cast<std::uint8_t*>(scratch);
    auto* x_scales = reinterpret_cast<float*>(scratch + codes_bytes(k, tokens));
    // On Hopper the activation scales are laid out for its CUTLASS kernel, which `prepared`
    // planes follow too (linear_projections asks the same question).
    const bool kb_major = sm90_gemm_available();
    if (prepared == nullptr) {
        launch_quantize_blocks(xin, k, tokens, x_codes, x_scales, kb_major, stream);
    }
    // Hopper: vLLM's wgmma kernel for the 128 x 128 block grid. A per-channel weight, or a launch
    // the kernel declines, takes the engine's own tile below on the same activation planes.
    if (kb_major && cell.k_per == kBlock && cell.rows_per == kBlock) {
        if (segments.count == 1 &&
            sm90_gemm(x_codes, x_scales, codes, scales, segments.out[0], accumulate, tokens, rows, k, stream)) {
            return;
        }
        if (segments.count > 1 && staging != nullptr && !accumulate &&
            sm90_gemm(x_codes, x_scales, codes, scales, staging, false, tokens, rows, k, stream)) {
            const long long items = static_cast<long long>(rows / 8) * tokens;
            split_segments_kernel<<<static_cast<unsigned>((items + 255) / 256), 256, 0, stream>>>(staging, segments, tokens);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    const int work = (rows / kTileRows) * ((tokens + kTileCols - 1) / kTileCols);
    const int grid = std::min(work, persistent_blocks());
    if (accumulate) {
        tile_kernel<true><<<grid, kTileWarps * 32, 0, stream>>>(codes, scales, cell, x_codes, x_scales, segments, k, tokens, kb_major);
    } else {
        tile_kernel<false><<<grid, kTileWarps * 32, 0, stream>>>(codes, scales, cell, x_codes, x_scales, segments, k, tokens, kb_major);
    }
    CUDA_CHECK(cudaGetLastError());
}

/// One output: the single-segment launch every plain linear takes.
void run(const Tensor& x, const Weight& w, std::int32_t row_begin, std::int32_t rows, Tensor& out,
         bool accumulate, WorkspaceArena* workspace, cudaStream_t stream, const char* op,
         std::byte* prepared = nullptr, std::int32_t route_rows = 0) {
    require_fp8_block_weight(w, op);
    require_x_out(x, w.k, out, rows, op);
    Segments one;
    one.count  = 1;
    one.end[0] = rows;
    one.out[0] = static_cast<__nv_bfloat16*>(out.data);
    run(static_cast<const __nv_bfloat16*>(x.data), x.ne[1], w, row_begin, one, accumulate, workspace,
        stream, op, prepared, nullptr, route_rows);
}

/// Projections that are consecutive row ranges of one parent, in order, as the segments of one
/// launch; false when they are not (different weights, a gap, more than four).
bool chain(std::span<const LinearProjection> projections, Segments& segments, std::int32_t& row_begin) {
    if (projections.size() < 2 || projections.size() > static_cast<std::size_t>(Segments::kMax)) { return false; }
    const Weight& w = projections.front().weight;
    row_begin = std::max(0, projections.front().row_begin);
    std::int32_t next = row_begin;
    segments.count = 0;
    for (const auto& p : projections) {
        const Weight& pw = p.weight;
        if (pw.qdata != w.qdata || pw.scales != w.scales || pw.qtype != w.qtype || pw.n != w.n ||
            pw.k != w.k || pw.scale_ne[0] != w.scale_ne[0] || pw.scale_ne[1] != w.scale_ne[1] ||
            std::max(0, p.row_begin) != next ||
            (reinterpret_cast<std::uintptr_t>(p.out.data) & 15u) != 0) {
            return false;
        }
        next += p.out.ne[0];
        segments.end[segments.count]   = next - row_begin;
        segments.out[segments.count++] = static_cast<__nv_bfloat16*>(p.out.data);
    }
    return true;
}

// A chained launch's staging plane follows the activation planes, whose bytes workspace_bytes
// already counts with an alignment's slack: it starts at most that far in.
std::size_t staging_offset(std::int32_t k, std::int32_t tokens) {
    const std::size_t planes = codes_bytes(k, tokens) +
                               static_cast<std::size_t>(sm90_scale_stride(tokens)) * (k / kBlock) * sizeof(float);
    return (planes + kAlign - 1) / kAlign * kAlign;
}

} // namespace

bool is_fp8_block_qtype(QType qtype) noexcept {
    return qtype == QType::FP8_E4M3FN_BLK128_F32S || qtype == QType::FP8_E4M3FN_ROW_F32S;
}

void require_fp8_block_weight(const Weight& w, const char* op) {
    const bool per_row = w.qtype == QType::FP8_E4M3FN_ROW_F32S;
    if (!is_fp8_block_qtype(w.qtype) || w.layout != QuantLayout::Fp8Block128 || w.qdata == nullptr ||
        w.scales == nullptr || w.ndim != 2 || w.n <= 0 || w.k <= 0 || (w.n % kTileRows) != 0 ||
        (w.k % kBlock) != 0 || w.scale_dtype != DType::FP32 || w.padded_shape[0] != w.n ||
        w.padded_shape[1] != w.k || (!per_row && (w.n % kBlock) != 0) ||
        w.scale_ne[0] != (per_row ? w.k : kBlock) || w.scale_ne[1] != (per_row ? 1 : kBlock)) {
        throw std::invalid_argument(std::string(op) + ": weight must be block- or row-scaled FP8, [n, k] with k whole 128-blocks and an FP32 scale grid");
    }
}

std::size_t workspace_bytes(std::int32_t k, std::int32_t tokens) noexcept {
    if (k <= 0 || tokens <= gemv_always_tokens() || mma_serves(tokens)) { return 0; }
    return codes_bytes(k, tokens) +
           static_cast<std::size_t>(sm90_scale_stride(tokens)) * (k / kBlock) * sizeof(float) + kAlign;
}

std::size_t linear_workspace_capacity_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                            std::int32_t max_tokens) {
    if (output_rows <= 0 || input_rows <= 0 || (input_rows % kBlock) != 0 || max_tokens <= 0) {
        throw std::invalid_argument("fp8 block linear workspace: k must be a whole number of 128-blocks");
    }
    return workspace_bytes(input_rows, max_tokens);
}

Weight weight_rows(const Weight& w, std::int32_t row_begin, std::int32_t rows) {
    require_fp8_block_weight(w, "fp8 block weight_rows");
    const std::int32_t rows_per = w.scale_ne[1], kcells = w.k / w.scale_ne[0];
    if (row_begin < 0 || rows <= 0 || row_begin + rows > w.n || (row_begin % rows_per) != 0 ||
        (row_begin % kTileRows) != 0 || (rows % kTileRows) != 0) {
        throw std::invalid_argument("fp8 block weight_rows: the range must be whole scale rows and whole 64-row tiles");
    }
    Weight out          = w;
    out.qdata           = static_cast<const std::byte*>(w.qdata) + static_cast<std::size_t>(row_begin) * w.k;
    out.payload         = out.qdata;
    out.scales          = static_cast<const float*>(w.scales) + static_cast<std::size_t>(row_begin / rows_per) * kcells;
    out.payload_bytes   = static_cast<std::uint64_t>(rows) * w.k; // the codes; the scales follow elsewhere
    out.n               = rows;
    out.shape[0]        = rows;
    out.padded_shape[0] = rows;
    return out;
}

void linear(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena* workspace, cudaStream_t stream) {
    run(x, w, 0, w.n, out, false, workspace, stream, "fp8 block linear");
}

void linear_add(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena* workspace, cudaStream_t stream) {
    run(x, w, 0, w.n, residual, true, workspace, stream, "fp8 block linear_add");
}

bool quantizes_activations(const Weight& w, std::int32_t tokens) noexcept {
    return !exact_serves(w.n, tokens, block_cell(w));
}

void swiglu_linear_add(const Tensor& packed, const Weight& w, Tensor& residual, float limit,
                       WorkspaceArena* workspace, cudaStream_t stream) {
    const char* op = "fp8 block swiglu linear_add";
    require_fp8_block_weight(w, op);
    const std::int32_t k = w.k, tokens = packed.ne[1];
    if (packed.dtype != DType::BF16 || !packed.is_contiguous() || packed.ne[0] != 2 * k ||
        (reinterpret_cast<std::uintptr_t>(packed.data) & 7u) != 0) {
        throw std::invalid_argument(std::string(op) + ": packed must be [2k, T] contiguous BF16, 8-byte aligned");
    }
    if (residual.dtype != DType::BF16 || !residual.is_contiguous() || residual.ne[0] != w.n ||
        residual.ne[1] != tokens || !quantizes_activations(w, tokens)) {
        throw std::invalid_argument(std::string(op) + ": residual must be [n, T] contiguous BF16, T past the GEMV widths");
    }
    auto scope              = workspace != nullptr ? std::optional(workspace->scope()) : std::nullopt;
    const std::size_t bytes = workspace_bytes(k, tokens);
    auto* planes = workspace != nullptr ? static_cast<std::byte*>(workspace->alloc_bytes(bytes, kAlign).data)
                                        : static_cast<std::byte*>(ggml::scratch_for(bytes, stream));
    const long long items = static_cast<long long>(tokens) * (k / kBlock);
    swiglu_quantize_blocks_kernel<<<static_cast<unsigned>((items + kQuantizeWarps - 1) / kQuantizeWarps),
                                    kQuantizeWarps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(packed.data), k, tokens, limit,
        reinterpret_cast<std::uint8_t*>(planes), reinterpret_cast<float*>(planes + codes_bytes(k, tokens)),
        sm90_gemm_available());
    CUDA_CHECK(cudaGetLastError());
    Segments one;
    one.count  = 1;
    one.end[0] = w.n;
    one.out[0] = static_cast<__nv_bfloat16*>(residual.data);
    run(nullptr, tokens, w, 0, one, true, workspace, stream, op, planes);
}

void project_rows(const Tensor& x, const Weight& w, std::int32_t row_begin, Tensor& out,
                  WorkspaceArena* workspace, cudaStream_t stream) {
    run(x, w, row_begin, out.ne[0], out, false, workspace, stream, "fp8 block project_rows");
}

void linear_projections(const Tensor& x, std::span<const LinearProjection> projections,
                        WorkspaceArena* workspace, cudaStream_t stream) {
    for (const auto& p : projections) {
        const Weight view = weight_rows(p.weight, std::max(0, p.row_begin), p.out.ne[0]);
        require_x_out(x, view.k, p.out, view.n, "fp8 block linear projections");
    }
    if (projections.empty()) { return; }
    auto scope = workspace != nullptr ? std::optional(workspace->scope()) : std::nullopt;
    const int k = x.ne[0], tokens = x.ne[1];
    // Consecutive row ranges of one parent -- q/k/v, q/k/gate/v, qkv/z -- run as one launch: the
    // GEMV and the engine's tile write each range to its own output; Hopper's CUTLASS kernel,
    // which writes one packed output, stages the rows and splits them, on narrow rounds only.
    Segments segments;
    std::int32_t chain_begin = 0;
    const bool chained   = chain(projections, segments, chain_begin);
    const Weight& parent = projections.front().weight;
    // Quantised once for every launch that reads the planes; a GEMV among them reads x.
    bool gemm = chained && !exact_serves(segments.rows(), tokens, block_cell(parent));
    for (const auto& p : projections) { gemm |= !chained && !exact_serves(p.out.ne[0], tokens, block_cell(p.weight)); }
    const bool cutlass   = gemm && sm90_gemm_available() && parent.scale_ne[0] == kBlock &&
                         parent.scale_ne[1] == kBlock;
    const std::size_t plane_bytes = gemm ? workspace_bytes(k, tokens) : 0;
    std::size_t staging_bytes =
        chained && cutlass && tokens <= kStagedMaxTokens
            ? static_cast<std::size_t>(segments.rows()) * tokens * sizeof(__nv_bfloat16)
            : 0;
    if (staging_bytes != 0 && workspace != nullptr) {
        // A caller's arena sized for the separate launches (linear_workspace_capacity_bytes)
        // keeps them; one sized by projections_workspace_capacity_bytes has the plane's room.
        const auto base = reinterpret_cast<std::uintptr_t>(workspace->base());
        const std::size_t at =
            (base + workspace->used() + kAlign - 1) / kAlign * kAlign - base;
        if (at + plane_bytes + staging_bytes > workspace->capacity()) { staging_bytes = 0; }
    }
    std::byte* prepared    = nullptr;
    __nv_bfloat16* staging = nullptr;
    if (plane_bytes != 0) {
        std::size_t bytes = plane_bytes + staging_bytes;
        if (workspace == nullptr && chained && cutlass && tokens > kStagedMaxTokens) {
            // The engine-slot scratch grows only outside a capture, warmed at the widest round:
            // past the staged widths, keep room for every narrower round's staged launch too.
            bytes = std::max(bytes, workspace_bytes(k, kStagedMaxTokens) +
                                        static_cast<std::size_t>(segments.rows()) * kStagedMaxTokens *
                                            sizeof(__nv_bfloat16));
        }
        prepared = workspace != nullptr ? static_cast<std::byte*>(workspace->alloc_bytes(bytes, kAlign).data)
                                        : static_cast<std::byte*>(ggml::scratch_for(bytes, stream));
        if (staging_bytes != 0) { staging = reinterpret_cast<__nv_bfloat16*>(prepared + staging_offset(k, tokens)); }
        launch_quantize_blocks(static_cast<const __nv_bfloat16*>(x.data), k, tokens,
                               reinterpret_cast<std::uint8_t*>(prepared),
                               reinterpret_cast<float*>(prepared + codes_bytes(k, tokens)),
                               sm90_gemm_available(), stream);
    }
    if (chained && (!cutlass || staging != nullptr)) {
        run(static_cast<const __nv_bfloat16*>(x.data), tokens, parent, chain_begin, segments, false,
            workspace, stream, "fp8 block linear projections", prepared, staging);
        return;
    }
    for (const auto& p : projections) {
        run(x, p.weight, std::max(0, p.row_begin), p.out.ne[0], p.out, false, workspace, stream,
            "fp8 block linear projections", prepared, chained ? segments.rows() : 0);
    }
}

std::size_t projections_workspace_capacity_bytes(std::int32_t parent_rows, std::int32_t input_rows,
                                                 std::int32_t max_tokens) {
    const std::size_t planes = linear_workspace_capacity_bytes(parent_rows, input_rows, max_tokens);
    if (max_tokens <= gemv_always_tokens() || mma_serves(max_tokens) || parent_rows <= 0) { return planes; }
    return planes + static_cast<std::size_t>(parent_rows) *
                                  std::min(max_tokens, kStagedMaxTokens) * sizeof(__nv_bfloat16);
}

} // namespace sinfer::ops::detail::fp8_block
