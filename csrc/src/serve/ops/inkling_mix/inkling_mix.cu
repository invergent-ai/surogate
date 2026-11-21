// Inkling-2-Mini hidden-stream ops (lane e1): api/ops/inkling_mix.h. One translation unit: kernels,
// launchers and the host-side validation. Compiled with -fmad=false (CMakeLists.txt): every product /
// sum the reference does not fuse stays unfused, and the fma the reference does use is an explicit fmaf.
// Rounding points: rl/engine-inkling-2026-11-21/lanes/e1/DESIGN-target.md section 3.

#include "api/ops/inkling_mix.h"

#include "core/device.h" // CUDA_CHECK
#include "core/pdl.cuh"
#include "ops/common/mma.cuh"
#include "ops/kernel/func_attribute.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace sinfer::ops {
namespace {

using bf16 = __nv_bfloat16;
using G    = InklingMixGeometry;

constexpr int kD       = G::kHidden;     // 2304
constexpr int kThreads = 256;
constexpr int kPer     = kD / kThreads;  // 9
static_assert(kD % kThreads == 0, "hidden width must split over the block");

/// Non-FTZ rsqrt.approx.f32, as `ops/norm.py` `rsqrt` emits it.
__device__ __forceinline__ float rsqrt_approx(float x) {
    float y;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

struct PlaneDev {
    const void* data   = nullptr;
    std::int64_t stride = 0;
    int fp32           = 0;
};

__device__ __forceinline__ float load_plane(const PlaneDev& p, std::int64_t t, int d) {
    if (p.fp32) { return static_cast<const float*>(p.data)[t * p.stride + d]; }
    return __bfloat162float(static_cast<const bf16*>(p.data)[t * p.stride + d]);
}

/// Sum of `n` values per thread over the block; every thread gets the totals. `scratch` holds
/// [kThreads / 32][n] floats. Lane order is fixed, so the result is deterministic.
template <int N>
__device__ __forceinline__ void block_sum(float (&v)[N], float* scratch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int i = 0; i < N; ++i) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) { v[i] += __shfl_xor_sync(0xffffffffU, v[i], off); }
    }
    __syncthreads(); // scratch may still be read from an earlier reduction
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < N; ++i) { scratch[warp * N + i] = v[i]; }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < N; ++i) {
        float s = 0.0F;
#pragma unroll
        for (int w = 0; w < kThreads / 32; ++w) { s += scratch[w * N + i]; }
        v[i] = s;
    }
}

// ---------------------------------------------------------------------------------------- attn_res

struct AttnResArgs {
    PlaneDev src[G::kMaxStack + 1]; // stack..., last
    PlaneDev addend;
    int has_addend          = 0;
    const float* queries    = nullptr; // [D, Q]
    int num_queries         = 0;
    const bf16* norm_weight = nullptr; // [D, Q] or null
    float norm_eps          = 0.0F;
    void* out               = nullptr; // [D, T, Q]
    std::int64_t columns    = 0;
};

/// The N sources of a column (stack..., last [+ addend]) in registers.
template <int N>
__device__ __forceinline__ void load_sources(const PlaneDev* src, const PlaneDev& addend, int has_addend,
                                             std::int64_t t, float (&x)[N][kPer]) {
#pragma unroll
    for (int n = 0; n < N; ++n) {
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int d = threadIdx.x + i * kThreads;
            float v     = load_plane(src[n], t, d);
            if (n == N - 1 && has_addend) { v = v + load_plane(addend, t, d); }
            x[n][i] = v;
        }
    }
}

/// Per-source rstd: fma sum of squares per thread (the reference's tl.fma accumulation), block sum, rsqrt.approx.
template <int N>
__device__ __forceinline__ void source_rstd(const float (&x)[N][kPer], float (&rstd)[N], float* scratch) {
    float sq[N];
#pragma unroll
    for (int n = 0; n < N; ++n) {
        float s = 0.0F;
#pragma unroll
        for (int i = 0; i < kPer; ++i) { s = fmaf(x[n][i], x[n][i], s); }
        sq[n] = s;
    }
    block_sum<N>(sq, scratch);
#pragma unroll
    for (int n = 0; n < N; ++n) { rstd[n] = rsqrt_approx(fmaf(sq[n], 1.0F / 2304.0F, 1e-6F)); }
}

/// One query's fp32 mix: softmax over the sources' normalised logits, normalised values, fma chain, * 1 / sum.
template <int N>
__device__ __forceinline__ void mix_query(const float (&x)[N][kPer], const float (&rstd)[N], const float* query,
                                          float (&mix)[kPer], float* scratch) {
    float dot[N];
#pragma unroll
    for (int n = 0; n < N; ++n) {
        float s = 0.0F;
#pragma unroll
        for (int i = 0; i < kPer; ++i) { s = fmaf(query[threadIdx.x + i * kThreads], x[n][i], s); }
        dot[n] = s;
    }
    block_sum<N>(dot, scratch);
    float logit[N];
    float mx = -INFINITY;
#pragma unroll
    for (int n = 0; n < N; ++n) {
        logit[n] = dot[n] * rstd[n];
        mx       = fmaxf(mx, logit[n]);
    }
    float p[N];
    float total = 0.0F;
#pragma unroll
    for (int n = 0; n < N; ++n) {
        p[n]  = expf(logit[n] - mx);
        total = total + p[n];
    }
    const float inv_sum = __fdiv_rn(1.0F, total);
#pragma unroll
    for (int n = 0; n < N; ++n) { p[n] = p[n] * rstd[n]; } // normalize_values
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        float acc = 0.0F;
#pragma unroll
        for (int n = 0; n < N; ++n) { acc = fmaf(p[n], x[n][i], acc); }
        mix[i] = acc * inv_sum;
    }
}

template <int N>
__global__ void __launch_bounds__(kThreads) attn_res_kernel(const __grid_constant__ AttnResArgs a) {
    pdl::wait_for_dependencies(); // PDL: every read / write after the producer grid completes
    pdl::trigger_dependents(); // PDL: the next kernel may launch now (it waits for this grid before any read)
    __shared__ float scratch[(kThreads / 32) * (N > 2 ? N : 2)];
    const std::int64_t t = blockIdx.x;
    float x[N][kPer];
    load_sources<N>(a.src, a.addend, a.has_addend, t, x);
    float rstd[N];
    source_rstd<N>(x, rstd, scratch);
    for (int q = 0; q < a.num_queries; ++q) {
        float mix[kPer];
        mix_query<N>(x, rstd, a.queries + static_cast<std::int64_t>(q) * kD, mix, scratch);
        const std::int64_t base = (static_cast<std::int64_t>(q) * a.columns + t) * kD;
        if (a.norm_weight != nullptr) {
            float ss[1] = {0.0F};
#pragma unroll
            for (int i = 0; i < kPer; ++i) { ss[0] = ss[0] + mix[i] * mix[i]; }
            block_sum<1>(ss, scratch);
            const float out_rstd = rsqrt_approx(__fdiv_rn(ss[0], 2304.0F) + a.norm_eps);
            const bf16* w        = a.norm_weight + static_cast<std::int64_t>(q) * kD;
            bf16* out            = static_cast<bf16*>(a.out) + base;
#pragma unroll
            for (int i = 0; i < kPer; ++i) {
                const int d = threadIdx.x + i * kThreads;
                out[d]      = __float2bfloat16_rn((mix[i] * out_rstd) * __bfloat162float(w[d]));
            }
        } else {
            float* out = static_cast<float*>(a.out) + base;
#pragma unroll
            for (int i = 0; i < kPer; ++i) { out[threadIdx.x + i * kThreads] = mix[i]; }
        }
    }
}

// ------------------------------------------------------------------- blend inputs (fused attn_res x 2 + pair norm)

struct BlendInputsArgs {
    PlaneDev src[G::kMaxStack + 1]; // dest sources: stack..., last
    PlaneDev addend;
    int has_addend             = 0;
    const float* dest_queries  = nullptr; // [D, DQ]
    int dest_q                 = 1;       // 1 (broadcast over the streams) or `streams`
    PlaneDev bank[2];                     // the message sources: the previous pass's B6, P12
    const float* msg_queries   = nullptr; // [D, S]
    int streams                = 1;
    float* dest                = nullptr; // FP32 [D, T, DQ]
    float* msg                 = nullptr; // FP32 [D, T, S]
    bf16* pair                 = nullptr; // BF16 [2D, T, S]
    std::int64_t columns       = 0;
};

/// inkling_attn_res (dest, no norm) + inkling_attn_res (message, no norm) + inkling_pair_rms_norm of one column, with
/// the same device code and order as the three separate kernels: bit-identical to them.
template <int N>
__global__ void __launch_bounds__(kThreads) blend_inputs_kernel(const __grid_constant__ BlendInputsArgs a) {
    pdl::wait_for_dependencies(); // PDL: every read / write after the producer grid completes
    pdl::trigger_dependents(); // PDL: the next kernel may launch now (it waits for this grid before any read)
    __shared__ float scratch[(kThreads / 32) * (N > 2 ? N : 2)];
    const std::int64_t t = blockIdx.x;
    float x[N][kPer];
    load_sources<N>(a.src, a.addend, a.has_addend, t, x);
    float rstd[N];
    source_rstd<N>(x, rstd, scratch);
    float y[2][kPer];
    load_sources<2>(a.bank, a.addend, 0, t, y);
    float yr[2];
    source_rstd<2>(y, yr, scratch);
    float dmix[kPer];
    for (int s = 0; s < a.streams; ++s) {
        if (s == 0 || a.dest_q > 1) {
            const int q = a.dest_q > 1 ? s : 0;
            mix_query<N>(x, rstd, a.dest_queries + static_cast<std::int64_t>(q) * kD, dmix, scratch);
            float* out = a.dest + (static_cast<std::int64_t>(q) * a.columns + t) * kD;
#pragma unroll
            for (int i = 0; i < kPer; ++i) { out[threadIdx.x + i * kThreads] = dmix[i]; }
        }
        float mmix[kPer];
        mix_query<2>(y, yr, a.msg_queries + static_cast<std::int64_t>(s) * kD, mmix, scratch);
        float* mo = a.msg + (static_cast<std::int64_t>(s) * a.columns + t) * kD;
#pragma unroll
        for (int i = 0; i < kPer; ++i) { mo[threadIdx.x + i * kThreads] = mmix[i]; }
        // pair_rms_norm_kernel's arithmetic
        float ss[2] = {0.0F, 0.0F};
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            ss[0] = ss[0] + dmix[i] * dmix[i];
            ss[1] = ss[1] + mmix[i] * mmix[i];
        }
        block_sum<2>(ss, scratch);
        const float rd = rsqrt_approx(__fdiv_rn(ss[0], 2304.0F) + 1e-6F);
        const float rm = rsqrt_approx(__fdiv_rn(ss[1], 2304.0F) + 1e-6F);
        bf16* o        = a.pair + (static_cast<std::int64_t>(s) * a.columns + t) * (2 * kD);
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int d = threadIdx.x + i * kThreads;
            o[d]        = __float2bfloat16_rn(dmix[i] * rd);
            o[kD + d]   = __float2bfloat16_rn(mmix[i] * rm);
        }
    }
}

// ------------------------------------------------------------------------------- embed and rms_norm

__global__ void __launch_bounds__(kThreads)
    embed_kernel(const std::int32_t* ids, const bf16* table, const bf16* w, float eps, bf16* out,
                 std::int64_t out_stride) {
    __shared__ float scratch[kThreads / 32];
    const std::int64_t t   = blockIdx.x;
    const bf16* row        = table + static_cast<std::int64_t>(ids[t]) * kD;
    float x[kPer];
    float ss[1] = {0.0F};
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        x[i]  = __bfloat162float(row[threadIdx.x + i * kThreads]);
        ss[0] = ss[0] + x[i] * x[i];
    }
    block_sum<1>(ss, scratch);
    const float rstd = rsqrt_approx(__fdiv_rn(ss[0], 2304.0F) + eps);
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int d             = threadIdx.x + i * kThreads;
        out[t * out_stride + d] = __float2bfloat16_rn((x[i] * rstd) * __bfloat162float(w[d]));
    }
}

__global__ void __launch_bounds__(kThreads)
    rms_norm_kernel(PlaneDev x_plane, const bf16* w, int num_weights, float eps, bf16* out,
                    std::int64_t columns) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents(); // PDL: the next kernel may launch now (it waits for this grid before any read)
    __shared__ float scratch[kThreads / 32];
    const std::int64_t t = blockIdx.x;
    float x[kPer];
    float ss[1] = {0.0F};
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        x[i]  = load_plane(x_plane, t, threadIdx.x + i * kThreads);
        ss[0] = ss[0] + x[i] * x[i];
    }
    block_sum<1>(ss, scratch);
    const float rstd = rsqrt_approx(__fdiv_rn(ss[0], 2304.0F) + eps);
    for (int q = 0; q < num_weights; ++q) {
        bf16* o = out + (static_cast<std::int64_t>(q) * columns + t) * kD;
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int d = threadIdx.x + i * kThreads;
            o[d] = __float2bfloat16_rn((x[i] * rstd) * __bfloat162float(w[q * kD + d]));
        }
    }
}

// -------------------------------------------------------------------------------- loop controller

__global__ void __launch_bounds__(kThreads)
    pair_rms_norm_kernel(const float* dest, std::int64_t dest_stream_stride, const float* msg,
                         bf16* out, std::int64_t columns) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents(); // PDL: the next kernel may launch now (it waits for this grid before any read)
    __shared__ float scratch[(kThreads / 32) * 2];
    const std::int64_t t = blockIdx.x;
    const std::int64_t s = blockIdx.y;
    const float* dp      = dest + s * dest_stream_stride + t * kD;
    const float* mp      = msg + (s * columns + t) * kD;
    float dv[kPer], mv[kPer];
    float ss[2] = {0.0F, 0.0F};
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int d = threadIdx.x + i * kThreads;
        dv[i]       = dp[d];
        mv[i]       = mp[d];
        ss[0]       = ss[0] + dv[i] * dv[i];
        ss[1]       = ss[1] + mv[i] * mv[i];
    }
    block_sum<2>(ss, scratch);
    const float rd = rsqrt_approx(__fdiv_rn(ss[0], 2304.0F) + 1e-6F);
    const float rm = rsqrt_approx(__fdiv_rn(ss[1], 2304.0F) + 1e-6F);
    bf16* o        = out + (s * columns + t) * (2 * kD);
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int d = threadIdx.x + i * kThreads;
        o[d]        = __float2bfloat16_rn(dv[i] * rd);
        o[kD + d]   = __float2bfloat16_rn(mv[i] * rm);
    }
}

__global__ void __launch_bounds__(kThreads)
    blend_rms_norm_kernel(const float* dest, std::int64_t dest_stream_stride, const float* msg,
                          const float* gain_forget, const bf16* w, int w_stream_stride, float eps,
                          bf16* out, std::int64_t columns) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents(); // PDL: the next kernel may launch now (it waits for this grid before any read)
    __shared__ float scratch[kThreads / 32];
    const std::int64_t t   = blockIdx.x;
    const std::int64_t s   = blockIdx.y;
    const std::int64_t row = s * columns + t;
    const float* dp        = dest + s * dest_stream_stride + t * kD;
    const float* mp        = msg + row * kD;
    const float* gf        = gain_forget + row * (2 * kD);
    float xv[kPer];
    float ss[1] = {0.0F};
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int d        = threadIdx.x + i * kThreads;
        const float gain   = tanhf(gf[d]);
        const float forget = tanhf(gf[kD + d]);
        xv[i]              = (1.0F - forget) * dp[d] + gain * mp[d];
        ss[0]              = ss[0] + xv[i] * xv[i];
    }
    block_sum<1>(ss, scratch);
    const float rstd = rsqrt_approx(__fdiv_rn(ss[0], 2304.0F) + eps);
    const bf16* ws   = w + s * w_stream_stride;
    bf16* o          = out + row * kD;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int d = threadIdx.x + i * kThreads;
        o[d]        = __float2bfloat16_rn((xv[i] * rstd) * __bfloat162float(ws[d]));
    }
}

// ---------------------------------------------------- controller tail (one cluster: split reduce + gain/forget + blend)

namespace cg = cooperative_groups;

constexpr int kTailCtas     = 8;                 // cluster size
constexpr int kTailChannels = kD / kTailCtas;    // 288 channels per CTA: 18 m16 tiles of gain rows, 18 of forget
constexpr int kTailRows     = 2 * kTailChannels; // 576 rows of the gain/forget weights per CTA
constexpr int kTailTokens   = 8;                 // one n8 tile of columns per cluster
constexpr int kRank         = 64;                // the controllers' rank (k of gain/forget)
constexpr int kTailSplits   = 18;                // `down` 64 x 4608 in chunks of 256 (inkling_gemm_split)
constexpr int kTailSmem     = kTailRows * kRank * 2 + kTailTokens * kRank * 2 + kTailTokens * kTailChannels * 4;
static_assert(kTailRows * kTailTokens * 4 <= kTailRows * kRank * 2, "G aliases the A tile");

__device__ __forceinline__ int tail_swizzle(int row, int col) {
    return (col & ~63) + ((((col & 63) >> 3) ^ (row & 7)) << 3) + (col & 7);
}

struct TailArgs {
    const float* partials       = nullptr; // [splits][S][T][64], inkling_gemm_inv_partials' layout
    const bf16* gf_weight       = nullptr; // stream 0's [4608][64]; stream s at + s * 4608 * 64
    const float* dest           = nullptr;
    std::int64_t dest_stream    = 0;       // 0 (broadcast) or T * 2304
    const float* msg            = nullptr; // [2304, T, S]
    const bf16* norm_weight     = nullptr;
    int w_stream                = 0;       // 0 (broadcast) or 2304
    float eps                   = 0.0F;
    bf16* out                   = nullptr; // [2304, T, S]
    std::int64_t columns        = 0;
    int streams                 = 1;
    int push_min                = 1;       // lane mx tail: the push scheme from this many columns of a cluster on
};

/// inkling_gemm_inv's split reduce (h = bf16(p0 + p1 + ... + p17)) + inkling_gemm_inv FP32 of the gain/forget weights
/// (the same m16n8k16 fragments at the same row / column positions, k = 64 in 4 steps from zero) +
/// blend_rms_norm_kernel (the same expressions; the sum of squares recomputed by every CTA in that kernel's thread
/// layout from the cluster's x through distributed shared memory): bit-identical to the three launches.
__global__ void __cluster_dims__(kTailCtas, 1, 1) __launch_bounds__(kThreads)
    controller_tail_kernel(const __grid_constant__ TailArgs a) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents(); // PDL: the next kernel may launch now (it waits for this grid before any read)
    extern __shared__ __align__(16) unsigned char tail_smem[];
    auto* As = reinterpret_cast<bf16*>(tail_smem);                      // [576][64] swizzled (then G: fp32 [576][8])
    auto* Bs = As + kTailRows * kRank;                                 // [8][64] swizzled
    auto* X  = reinterpret_cast<float*>(Bs + kTailTokens * kRank);     // [8][288]
    auto* G  = reinterpret_cast<float*>(tail_smem);
    __shared__ float scratch[kThreads / 32];
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = static_cast<int>(cluster.block_rank());
    const int s    = static_cast<int>(blockIdx.y);
    const int t0   = (static_cast<int>(blockIdx.x) / kTailCtas) * kTailTokens;
    const int nt   = min(kTailTokens, static_cast<int>(a.columns) - t0);
    const int c0   = rank * kTailChannels;
    const int tid  = static_cast<int>(threadIdx.x);
    // 1. A: gain rows c0 .. c0 + 287 and forget rows 2304 + c0 .. of stream s
    const bf16* W = a.gf_weight + static_cast<std::int64_t>(s) * (2 * kD) * kRank;
    for (int item = tid; item < kTailRows * (kRank / 8); item += kThreads) {
        const int row = item / (kRank / 8), kk = (item - row * (kRank / 8)) * 8;
        const int src = row < kTailChannels ? c0 + row : kD + c0 + (row - kTailChannels);
        cp_async_zfill<16, Cache::cg>(&As[row * kRank + tail_swizzle(row, kk)],
                                      &W[static_cast<std::int64_t>(src) * kRank + kk], 16);
    }
    cp_commit();
    // 2. B: h = bf16(the partials added in chunk order), zero past the last column (the GEMM's zero fill)
    const std::int64_t plane = static_cast<std::int64_t>(a.streams) * a.columns * kRank;
    for (int item = tid; item < kTailTokens * kRank; item += kThreads) {
        const int j = item / kRank, r = item - j * kRank;
        float h     = 0.0F;
        if (j < nt) {
            const std::int64_t i = (static_cast<std::int64_t>(s) * a.columns + t0 + j) * kRank + r;
            h                    = a.partials[i];
            for (int c = 1; c < kTailSplits; ++c) { h += a.partials[c * plane + i]; }
        }
        Bs[j * kRank + tail_swizzle(j, r)] = __float2bfloat16_rn(h);
    }
    cp_wait<0>();
    __syncthreads();
    // 3. the gain/forget GEMM of the 36 m16 tiles: warp w takes w, w + 8, ...
    const int warp = tid >> 5, lane = tid & 31, gid = lane >> 2, lid = lane & 3;
    const int a_matrix = lane >> 3, a_inner = lane & 7;
    const int a_row_off = a_inner + ((a_matrix & 1) << 3), a_col_off = (a_matrix >> 1) << 3;
    const int b_inner = lane & 7, b_k_off = ((lane >> 3) & 1) << 3;
    constexpr int kTiles = kTailRows / 16, kWarps = kThreads / 32, kPerWarp = (kTiles + kWarps - 1) / kWarps;
    float acc[kPerWarp][4];
#pragma unroll
    for (int u = 0; u < kPerWarp; ++u) {
        acc[u][0] = acc[u][1] = acc[u][2] = acc[u][3] = 0.0F;
        const int mt = warp + u * kWarps;
        if (mt >= kTiles) { continue; }
#pragma unroll
        for (int k_step = 0; k_step < kRank / 16; ++k_step) {
            unsigned af[4], bfr[2];
            const int row = mt * 16 + a_row_off, col = k_step * 16 + a_col_off;
            ldmatrix_x4(af[0], af[1], af[2], af[3], smem_addr(&As[row * kRank + tail_swizzle(row, col)]));
            const int bcol = k_step * 16 + b_k_off;
            ldmatrix_x2(bfr[0], bfr[1], smem_addr(&Bs[b_inner * kRank + tail_swizzle(b_inner, bcol)]));
            mma_bf16(acc[u][0], acc[u][1], acc[u][2], acc[u][3], af[0], af[1], af[2], af[3], bfr[0], bfr[1]);
        }
    }
    __syncthreads(); // G overwrites A
#pragma unroll
    for (int u = 0; u < kPerWarp; ++u) {
        const int mt = warp + u * kWarps;
        if (mt >= kTiles) { continue; }
        const int r0 = mt * 16 + gid;
        G[r0 * kTailTokens + 2 * lid]           = acc[u][0];
        G[r0 * kTailTokens + 2 * lid + 1]       = acc[u][1];
        G[(r0 + 8) * kTailTokens + 2 * lid]     = acc[u][2];
        G[(r0 + 8) * kTailTokens + 2 * lid + 1] = acc[u][3];
    }
    __syncthreads();
    // 4. x of this CTA's channels (blend_rms_norm_kernel's expression)
    for (int item = tid; item < nt * kTailChannels; item += kThreads) {
        const int j = item / kTailChannels, c = item - j * kTailChannels;
        const int d = c0 + c;
        const std::int64_t t = t0 + j;
        const float gain     = tanhf(G[c * kTailTokens + j]);
        const float forget   = tanhf(G[(kTailChannels + c) * kTailTokens + j]);
        const float dv       = a.dest[s * a.dest_stream + t * kD + d];
        const float mv       = a.msg[(s * a.columns + t) * kD + d];
        X[j * kTailChannels + c] = (1.0F - forget) * dv + gain * mv;
    }
    cluster.sync();
    // 5. per column: the norm's sum of squares in blend_rms_norm_kernel's layout (thread tid sums d = tid + 256 i),
    //    reading x where it lives; then this CTA's outputs
    for (int j = 0; j < nt; ++j) {
        float ss[1] = {0.0F};
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int d     = tid + i * kThreads;
            const float* rx = cluster.map_shared_rank(X, d / kTailChannels);
            const float x   = rx[j * kTailChannels + d % kTailChannels];
            ss[0]           = ss[0] + x * x;
        }
        block_sum<1>(ss, scratch);
        const float rstd = rsqrt_approx(__fdiv_rn(ss[0], 2304.0F) + a.eps);
        const bf16* ws   = a.norm_weight + static_cast<std::int64_t>(s) * a.w_stream;
        bf16* o          = a.out + (static_cast<std::int64_t>(s) * a.columns + t0 + j) * kD;
        for (int c = tid; c < kTailChannels; c += kThreads) {
            const int d = c0 + c;
            o[d]        = __float2bfloat16_rn((X[j * kTailChannels + c] * rstd) * __bfloat162float(ws[d]));
        }
    }
    cluster.sync(); // the other CTAs may still read this CTA's x
}

// ------------------------------------------------------------------------------------- short conv

constexpr int kConvChannelTile = 128;
constexpr int kConvColumnTile  = 32;

struct ConvArgs {
    const bf16* x            = nullptr;
    std::int64_t x_stride    = 0;
    int channels             = 0;
    const bf16* weight       = nullptr; // [4, channels]: tap k of channel c at c * 4 + k
    int width                = 0;
    int batch                = 0;
    const std::int32_t* positions = nullptr; // [W, B]
    const std::int32_t* valid     = nullptr; // [B] or null
    const std::int32_t* slots     = nullptr; // [B]
    bf16* state              = nullptr;      // already offset by the stream's channel offset
    std::int64_t slot_stride = 0;
    std::int64_t tap_stride  = 0;
    const bf16* residual     = nullptr;
    std::int64_t residual_stride = 0;
    const bf16* addend       = nullptr;
    std::int64_t addend_stride = 0;
    bf16* out                = nullptr;
    std::int64_t out_stride  = 0;
    /// 1 when one CTA covers each row's columns (W <= kConvColumnTile) and nothing is captured: the conv kernel
    /// writes the new window itself (read, then write; no other thread owns the row's channel) -- the decode /
    /// verify form, one launch instead of two.
    int fuse_window          = 0;
};

/// The value at sequence index i of row b: S = [state tap 0, 1, 2, x column 0, 1, ...].
__device__ __forceinline__ float conv_seq(const ConvArgs& a, int b, std::int64_t slot, int i, int c) {
    if (i < 3) { return __bfloat162float(a.state[slot * a.slot_stride + i * a.tap_stride + c]); }
    const std::int64_t n = static_cast<std::int64_t>(i - 3) + static_cast<std::int64_t>(a.width) * b;
    return __bfloat162float(a.x[n * a.x_stride + c]);
}

__global__ void __launch_bounds__(kConvChannelTile)
    short_conv_kernel(const __grid_constant__ ConvArgs a) {
    pdl::wait_for_dependencies(); // PDL: every read / write after the producer grid completes
    pdl::trigger_dependents(); // PDL: the next kernel may launch now (it waits for this grid before any read)
    const int c = blockIdx.x * kConvChannelTile + threadIdx.x;
    const int b = blockIdx.z;
    if (c >= a.channels) { return; }
    const std::int64_t slot = a.slots[b];
    const int valid = a.valid != nullptr ? a.valid[b] : a.width;
    float w[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        w[k] = __bfloat162float(a.weight[static_cast<std::int64_t>(c) * 4 + k]);
    }
    const float scale = 0.44721359549995793F; // fp32(5 ** -0.5)
    const int j0      = blockIdx.y * kConvColumnTile;
    const int j1      = min(j0 + kConvColumnTile, a.width);
    for (int j = j0; j < j1; ++j) {
        if (j >= valid) { break; }
        const std::int64_t n = static_cast<std::int64_t>(j) + static_cast<std::int64_t>(a.width) * b;
        const int pos        = a.positions[n];
        float acc            = 0.0F;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int lag = 3 - k;
            const float v = (pos - lag >= 0) ? conv_seq(a, b, slot, j + k, c) : 0.0F;
            acc           = fmaf(v, w[k], acc);
        }
        const float cur = __bfloat162float(a.x[n * a.x_stride + c]);
        bf16 y          = __float2bfloat16_rn((acc + cur) * scale);
        if (a.addend != nullptr) {
            float base = __bfloat162float(a.addend[n * a.addend_stride + c]);
            if (a.residual != nullptr) {
                base = __bfloat162float(a.residual[n * a.residual_stride + c]) + base;
            }
            y = __float2bfloat16_rn(base + __bfloat162float(y));
        }
        a.out[n * a.out_stride + c] = y;
    }
    if (a.fuse_window && valid > 0) { // conv_window_kernel's update, by the only thread that owns (c, b)
        float upd[3];
#pragma unroll
        for (int i = 0; i < 3; ++i) { upd[i] = conv_seq(a, b, slot, valid + i, c); }
#pragma unroll
        for (int i = 0; i < 3; ++i) { a.state[slot * a.slot_stride + i * a.tap_stride + c] = __float2bfloat16_rn(upd[i]); }
    }
}

/// New windows after the conv: per row, the 3 inputs before index `valid` of S go to the row's
/// slot (in place); with a capture, the 3 before `capture_column` go to `capture_slot` first. Each
/// thread reads every value it needs before it writes.
__global__ void __launch_bounds__(kConvChannelTile)
    conv_window_kernel(const __grid_constant__ ConvArgs a, int capture_column, int capture_slot,
                       int update_slots) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents(); // PDL: the next kernel may launch now (it waits for this grid before any read)
    const int c = blockIdx.x * kConvChannelTile + threadIdx.x;
    const int b = blockIdx.y;
    if (c >= a.channels) { return; }
    const std::int64_t slot = a.slots[b];
    const int valid = a.valid != nullptr ? a.valid[b] : a.width;
    float upd[3], cap[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        upd[i] = conv_seq(a, b, slot, valid + i, c);
        cap[i] = capture_slot >= 0 ? conv_seq(a, b, slot, capture_column + i, c) : 0.0F;
    }
    if (capture_slot >= 0) {
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            a.state[static_cast<std::int64_t>(capture_slot) * a.slot_stride + i * a.tap_stride + c] =
                __float2bfloat16_rn(cap[i]);
        }
    }
    if (update_slots && valid > 0) {
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            a.state[slot * a.slot_stride + i * a.tap_stride + c] = __float2bfloat16_rn(upd[i]);
        }
    }
}

__global__ void conv_copy_slot_kernel(bf16* state, std::int64_t slot_stride, std::int64_t tap_stride,
                                      int channels, int src, int dst) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) { return; }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        state[dst * slot_stride + i * tap_stride + c] = state[src * slot_stride + i * tap_stride + c];
    }
}

// ----------------------------------------------------------------------- verify records and the fold

__global__ void conv_backup_kernel(const bf16* pool, std::int64_t visit_stride,
                                   std::int64_t slot_stride, const std::int32_t* slots,
                                   bf16* old_records, int visits) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y;
    const int b = blockIdx.z;
    if (c >= G::kConvChannels) { return; }
    const bf16* src = pool + v * visit_stride + static_cast<std::int64_t>(slots[b]) * slot_stride;
    bf16* dst = old_records + (static_cast<std::int64_t>(b) * visits + v) * 3 * G::kConvChannels;
#pragma unroll
    for (int i = 0; i < 3; ++i) { dst[i * G::kConvChannels + c] = src[i * G::kConvChannels + c]; }
}

__global__ void conv_record_kernel(const bf16* x, std::int64_t x_stride, int channels, int width,
                                   int visit, int channel_offset, bf16* inputs, int visits,
                                   int width_max) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y;
    const int b = blockIdx.z;
    if (c >= channels) { return; }
    const std::int64_t n = static_cast<std::int64_t>(j) + static_cast<std::int64_t>(width) * b;
    bf16* dst = inputs + ((static_cast<std::int64_t>(b) * visits + visit) * width_max + j) *
                             G::kConvChannels;
    dst[channel_offset + c] = x[n * x_stride + c];
}

constexpr int kMaxFoldRows = 128;
struct FoldArgs {
    InklingConvFoldRow rows[kMaxFoldRows];
    int count = 0;
};

__global__ void conv_fold_kernel(const __grid_constant__ FoldArgs f, bf16* pool,
                                 std::int64_t visit_stride, std::int64_t slot_stride,
                                 const bf16* old_records, const bf16* inputs, int visits,
                                 int width_max) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y;
    const int r = blockIdx.z;
    if (c >= G::kConvChannels || r >= f.count) { return; }
    const InklingConvFoldRow row = f.rows[r];
    const bf16* old = old_records + (static_cast<std::int64_t>(row.row) * visits + v) * 3 * G::kConvChannels;
    const bf16* in  = inputs + (static_cast<std::int64_t>(row.row) * visits + v) * width_max * G::kConvChannels;
    bf16 window[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int idx = row.committed + i; // index into [old 0..2 | inputs 0..]
        window[i] = idx < 3 ? old[idx * G::kConvChannels + c] : in[(idx - 3) * G::kConvChannels + c];
    }
    bf16* dst = pool + v * visit_stride + static_cast<std::int64_t>(row.slot) * slot_stride;
#pragma unroll
    for (int i = 0; i < 3; ++i) { dst[i * G::kConvChannels + c] = window[i]; }
}

__global__ void softcap_f32_kernel(float* logits, int rows, int tokens, std::int64_t ldc, float cap) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents(); // PDL: the next kernel may launch now (it waits for this grid before any read)
    const std::int64_t i     = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = static_cast<std::int64_t>(rows) * tokens;
    if (i >= total) { return; }
    const std::int64_t t = i / rows;
    const std::int64_t r = i - t * rows;
    float& x             = logits[t * ldc + r];
    x                    = tanhf(__fdiv_rn(x, cap)) * cap;
}

// =================================================================== lane mx: latency-cut kernels (decode / verify)
//
// The same arithmetic as the reference kernels above, in the same per-element order, with the same reduction trees
// (each value: the xor-shuffle butterfly within a warp, then 0 + warp 0 + ... + warp 7): bit-identical to them
// (invariance test I8 compares the two bit for bit). What changes is WHEN and WHERE things happen:
//  - weights (AttnRes queries, norm weights, conv taps, the controllers' gain / forget tile) are read BEFORE the PDL
//    wait: nothing inside a step writes them, so the reads overlap the producer's tail (activations are read only
//    after the wait, through plain loads -- never ld.global.nc, which can return stale lines under PDL); the AttnRes
//    stack / bank columns get an L2 bulk prefetch (a hint, no register) there too;
//  - every activation load of a row is issued before its first use (the reference kernels' per-source dtype branches
//    made the compiler wait for each source before it issued the next one: ~7 dependent memory round trips);
//  - attn_res runs one CTA per (column, query) and blend_inputs one per (column, dest query | message stream): the
//    reference's single CTA per column was issue-bound at T = 1 (four schedulers for every query's work);
//  - a CTA's independent reductions share one pass (sums of squares + dots), and a pass uses alternating scratch
//    halves, so it needs one barrier instead of two;
//  - controller tail: each CTA pushes its x into the CTA that reduces the column (DSMEM stores) instead of every CTA
//    pulling every column's x (72 DSMEM loads per thread at T = 8 saturated the LSU queue).

// The kernels keep the reference kernels' names inside namespace mx, so the nsys tooling (perf/m4/nsys_kernels.py,
// perf/h2/op_table.py: base name, batch from short_conv_kernel's grid.z) reads both paths alike.
namespace mx {

constexpr int kMxWarps = kThreads / 32;

__device__ __forceinline__ float bits_to_f32(unsigned short b) {
    return __uint_as_float(static_cast<unsigned>(b) << 16U); // == __bfloat162float
}

__device__ __forceinline__ unsigned short ld_bits(const void* base, std::int64_t i) {
    return static_cast<const unsigned short*>(base)[i];
}

/// An L2 prefetch of one column of a bf16 plane (4,608 bytes; cp.async.bulk.prefetch, sm_90+). A hint, not a load: it
/// writes no register and L2 is the coherence point, so it is safe before the PDL wait even if a producer still writes
/// the line (the post-wait load reads whatever L2 holds then). Used for AttnRes stack / bank sources, which completed
/// visits wrote long ago and the step's weight stream has usually evicted from L2.
__device__ __forceinline__ void prefetch_column_l2(const PlaneDev& p, std::int64_t t) {
    const auto* row = static_cast<const unsigned short*>(p.data) + t * p.stride;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(row), "r"(static_cast<unsigned>(kD * 2)));
}

/// block_sum's arithmetic for M values, with ONE barrier: `half` (16-byte aligned, kMxWarps * M floats) must differ
/// from the half the previous pass of this block used (a thread that writes here has passed the previous pass's
/// barrier, so every read of the other half is done; the pass before that used this half and all its reads happened
/// before the previous pass's barrier). Branch-free over the M values (a runtime bound would serialise the shuffle
/// chains); callers zero the values they do not use. Warp w's sum of value i sits at half[i * kMxWarps + w], read
/// back as two float4 and added in warp order from 0 -- the same sum as block_sum's.
template <int M>
__device__ __forceinline__ void block_sum_pass(float (&v)[M], float* half) {
    static_assert(kMxWarps == 8, "two float4 per value");
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int i = 0; i < M; ++i) { v[i] += __shfl_xor_sync(0xffffffffU, v[i], off); }
    }
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < M; ++i) { half[i * kMxWarps + warp] = v[i]; }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < M; ++i) {
        const float4 lo = *reinterpret_cast<const float4*>(&half[i * kMxWarps]);
        const float4 hi = *reinterpret_cast<const float4*>(&half[i * kMxWarps + 4]);
        float s         = 0.0F;
        s += lo.x;
        s += lo.y;
        s += lo.z;
        s += lo.w;
        s += hi.x;
        s += hi.y;
        s += hi.z;
        s += hi.w;
        v[i] = s;
    }
}

/// mix_query after its block sum: `dot` = the reduced dots of the N sources (r[off .. off + N)).
template <int N, int M>
__device__ __forceinline__ void softmax_mix(const float (&x)[N][kPer], const float (&rstd)[N], const float (&r)[M],
                                            int off, float (&mix)[kPer]) {
    float logit[N];
    float mx = -INFINITY;
#pragma unroll
    for (int n = 0; n < N; ++n) {
        logit[n] = r[off + n] * rstd[n];
        mx       = fmaxf(mx, logit[n]);
    }
    float p[N];
    float total = 0.0F;
#pragma unroll
    for (int n = 0; n < N; ++n) {
        p[n]  = expf(logit[n] - mx);
        total = total + p[n];
    }
    const float inv_sum = __fdiv_rn(1.0F, total);
#pragma unroll
    for (int n = 0; n < N; ++n) { p[n] = p[n] * rstd[n]; } // normalize_values
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        float acc = 0.0F;
#pragma unroll
        for (int n = 0; n < N; ++n) { acc = fmaf(p[n], x[n][i], acc); }
        mix[i] = acc * inv_sum;
    }
}

/// The N sources of a column, bf16 planes only (stack..., last [+ addend]): all loads first, then the conversions.
template <int N, bool kAdd>
__device__ __forceinline__ void load_sources_bf16(const PlaneDev* src, const PlaneDev& addend, std::int64_t t,
                                                  float (&x)[N][kPer]) {
    unsigned short raw[N][kPer];
    unsigned short rad[kPer];
#pragma unroll
    for (int n = 0; n < N; ++n) {
#pragma unroll
        for (int i = 0; i < kPer; ++i) { raw[n][i] = ld_bits(src[n].data, t * src[n].stride + threadIdx.x + i * kThreads); }
    }
    if constexpr (kAdd) {
#pragma unroll
        for (int i = 0; i < kPer; ++i) { rad[i] = ld_bits(addend.data, t * addend.stride + threadIdx.x + i * kThreads); }
    }
#pragma unroll
    for (int n = 0; n < N; ++n) {
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            float v = bits_to_f32(raw[n][i]);
            if constexpr (kAdd) {
                if (n == N - 1) { v = v + bits_to_f32(rad[i]); }
            }
            x[n][i] = v;
        }
    }
}

/// One query of one column (mix_query's arithmetic): one reduction pass over the sources' sums of squares and the
/// query's dots (scratch[0]), the softmax mix, and this thread's part of the output norm's sum of squares (returned;
/// the caller reduces it with scratch[1]).
template <int N>
__device__ __forceinline__ float single_query_mix(const float (&x)[N][kPer], const float (&qv)[kPer], float (&mix)[kPer],
                                                  float (*scratch)[kMxWarps * 2 * N]) {
    float r[2 * N];
#pragma unroll
    for (int n = 0; n < N; ++n) {
        float s = 0.0F;
#pragma unroll
        for (int i = 0; i < kPer; ++i) { s = fmaf(x[n][i], x[n][i], s); }
        r[n] = s;
    }
#pragma unroll
    for (int n = 0; n < N; ++n) {
        float s = 0.0F;
#pragma unroll
        for (int i = 0; i < kPer; ++i) { s = fmaf(qv[i], x[n][i], s); }
        r[N + n] = s;
    }
    block_sum_pass<2 * N>(r, scratch[0]);
    float rstd[N];
#pragma unroll
    for (int n = 0; n < N; ++n) { rstd[n] = rsqrt_approx(fmaf(r[n], 1.0F / 2304.0F, 1e-6F)); }
    softmax_mix<N, 2 * N>(x, rstd, r, N, mix);
    float ss[1] = {0.0F};
#pragma unroll
    for (int i = 0; i < kPer; ++i) { ss[0] = ss[0] + mix[i] * mix[i]; }
    return ss[0]; // the caller reduces it (or not)
}

/// attn_res_kernel<N> for bf16 sources, one CTA per (column, query): the queries of a column share nothing but their
/// sources, so each runs as its own CTA (the sources are read once per query; the sums of squares are recomputed
/// with the same expression). Per CTA: one pass for the sums of squares + the dots, one for the output norm.
template <int N, bool kAdd>
__global__ void __launch_bounds__(kThreads) attn_res_kernel(const __grid_constant__ AttnResArgs a) {
    const int tid   = static_cast<int>(threadIdx.x);
    const int q     = static_cast<int>(blockIdx.y);
    const bool norm = a.norm_weight != nullptr;
    float qv[kPer];
    unsigned short wb[kPer];
#pragma unroll
    for (int i = 0; i < kPer; ++i) { // weights: before the wait
        qv[i] = a.queries[q * kD + tid + i * kThreads];
        wb[i] = norm ? ld_bits(a.norm_weight, q * kD + tid + i * kThreads) : static_cast<unsigned short>(0);
    }
    if (tid < N - 1) { prefetch_column_l2(a.src[tid], blockIdx.x); } // stack sources (not `last`)
    pdl::wait_for_dependencies(); // PDL: every activation read / write after the producer grid completes
    pdl::trigger_dependents();
    __shared__ __align__(16) float scratch[2][kMxWarps * 2 * N];
    const std::int64_t t = blockIdx.x;
    float x[N][kPer];
    load_sources_bf16<N, kAdd>(a.src, a.addend, t, x);
    float mix[kPer];
    float ss[1] = {single_query_mix<N>(x, qv, mix, scratch)};
    if (norm) {
        block_sum_pass<1>(ss, scratch[1]);
        const float out_rstd = rsqrt_approx(__fdiv_rn(ss[0], 2304.0F) + a.norm_eps);
        bf16* out = static_cast<bf16*>(a.out) + (static_cast<std::int64_t>(q) * a.columns + t) * kD;
#pragma unroll
        for (int i = 0; i < kPer; ++i) { out[tid + i * kThreads] = __float2bfloat16_rn((mix[i] * out_rstd) * bits_to_f32(wb[i])); }
    } else {
        float* out = static_cast<float*>(a.out) + (static_cast<std::int64_t>(q) * a.columns + t) * kD;
#pragma unroll
        for (int i = 0; i < kPer; ++i) { out[tid + i * kThreads] = mix[i]; }
    }
}

/// blend_inputs_kernel<N> for bf16 planes, one CTA per (column, role): roles [0, DQ) = the dest mix of query q (and the
/// dest half of the pair norm of every stream that uses it), roles [DQ, DQ + S) = the message mix of stream s (and the
/// message half of its pair norm). The pair norm's two halves depend only on their own mix (rd on dest, rm on msg),
/// so the roles share nothing: no exchange between CTAs. Each role = one query's mix, as attn_res_kernel.
template <int N, bool kAdd>
__global__ void __launch_bounds__(kThreads) blend_inputs_kernel(const __grid_constant__ BlendInputsArgs a) {
    const int tid       = static_cast<int>(threadIdx.x);
    const int role      = static_cast<int>(blockIdx.y);
    const bool dest     = role < a.dest_q;
    const int q         = dest ? role : role - a.dest_q; // dest query, or message stream
    const float* query  = dest ? a.dest_queries + static_cast<std::int64_t>(q) * kD
                               : a.msg_queries + static_cast<std::int64_t>(q) * kD;
    float qv[kPer];
#pragma unroll
    for (int i = 0; i < kPer; ++i) { qv[i] = query[tid + i * kThreads]; } // weights: before the wait
    if (dest) {
        if (tid < N - 1) { prefetch_column_l2(a.src[tid], blockIdx.x); } // stack sources (not `last`)
    } else if (tid < 2) {
        prefetch_column_l2(a.bank[tid], blockIdx.x); // the previous pass's bank
    }
    pdl::wait_for_dependencies(); // PDL: every activation read / write after the producer grid completes
    pdl::trigger_dependents();
    const std::int64_t t = blockIdx.x;
    float mix[kPer];
    float ss[1];
    if (dest) {
        __shared__ __align__(16) float scratch[2][kMxWarps * 2 * N];
        float x[N][kPer];
        load_sources_bf16<N, kAdd>(a.src, a.addend, t, x);
        ss[0] = single_query_mix<N>(x, qv, mix, scratch);
        float* out = a.dest + (static_cast<std::int64_t>(q) * a.columns + t) * kD;
#pragma unroll
        for (int i = 0; i < kPer; ++i) { out[tid + i * kThreads] = mix[i]; }
        block_sum_pass<1>(ss, scratch[1]);
        const float rd = rsqrt_approx(__fdiv_rn(ss[0], 2304.0F) + 1e-6F);
        for (int s = 0; s < a.streams; ++s) { // dest broadcast (DQ = 1): every stream; DQ = S: stream q
            if (a.dest_q > 1 && s != q) { continue; }
            bf16* o = a.pair + (static_cast<std::int64_t>(s) * a.columns + t) * (2 * kD);
#pragma unroll
            for (int i = 0; i < kPer; ++i) { o[tid + i * kThreads] = __float2bfloat16_rn(mix[i] * rd); }
        }
    } else {
        __shared__ __align__(16) float mscratch[2][kMxWarps * 4];
        float y[2][kPer];
        load_sources_bf16<2, false>(a.bank, a.addend, t, y);
        ss[0] = single_query_mix<2>(y, qv, mix, mscratch);
        float* mo = a.msg + (static_cast<std::int64_t>(q) * a.columns + t) * kD;
#pragma unroll
        for (int i = 0; i < kPer; ++i) { mo[tid + i * kThreads] = mix[i]; }
        block_sum_pass<1>(ss, mscratch[1]);
        const float rm = rsqrt_approx(__fdiv_rn(ss[0], 2304.0F) + 1e-6F);
        bf16* o        = a.pair + (static_cast<std::int64_t>(q) * a.columns + t) * (2 * kD);
#pragma unroll
        for (int i = 0; i < kPer; ++i) { o[kD + tid + i * kThreads] = __float2bfloat16_rn(mix[i] * rm); }
    }
}

/// controller_tail_kernel with: the gain / forget tile and the norm weights requested before the PDL wait; dest /
/// msg loaded right after the wait (they do not depend on the GEMM); the norm's sums of squares by the push scheme
/// below (the reference walks the columns one by one: a DSMEM round trip, two barriers and a weight load each).
__global__ void __cluster_dims__(kTailCtas, 1, 1) __launch_bounds__(kThreads)
    controller_tail_kernel(const __grid_constant__ TailArgs a) {
    extern __shared__ __align__(16) unsigned char tail_smem[];
    auto* As = reinterpret_cast<bf16*>(tail_smem);                      // [576][64] swizzled (then G: fp32 [576][8])
    auto* Bs = As + kTailRows * kRank;                                 // [8][64] swizzled
    auto* X  = reinterpret_cast<float*>(Bs + kTailTokens * kRank);     // [8][288]
    auto* G  = reinterpret_cast<float*>(tail_smem);                    // fp32 [576][kGStride]
    auto* XF = X + kTailTokens * kTailChannels; // push scheme: [2304] the column this CTA reduces
    auto* RS = XF + kD;                         // push scheme: [8] every column's rstd
    constexpr int kGStride = kTailTokens + 1; // odd: step 4's reads of one column over consecutive rows hit 32 banks
    static_assert(kTailRows * kGStride * 4 <= kTailRows * kRank * 2, "G aliases the A tile");
    __shared__ __align__(16) float scratch[kMxWarps * kTailTokens];
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = static_cast<int>(cluster.block_rank());
    const int s    = static_cast<int>(blockIdx.y);
    const int t0   = (static_cast<int>(blockIdx.x) / kTailCtas) * kTailTokens;
    const int nt   = min(kTailTokens, static_cast<int>(a.columns) - t0);
    const int c0   = rank * kTailChannels;
    const int tid  = static_cast<int>(threadIdx.x);
    // Two ways to give every thread of a reducing CTA the 9 x values of its sum-of-squares chain (d = tid + 256 i, x
    // of channel d lives in CTA d / 288); same chains, same tree, same bits:
    //  pull (only with SUROGATE_SERVE_INKLING_TAIL_PUSH_MIN > 1, below that many columns): after the cluster barrier
    //  every CTA reads the column's x from the other CTAs (DSMEM loads);
    //  push (default, every column count): each CTA stores its x of column j straight into CTA j's column buffer,
    //  CTA j reduces it from its own shared memory and stores rstd into every CTA. At 8 columns the pull form makes
    //  each thread issue 72 DSMEM loads, which saturate the LSU queue (ncu 18:3x: stall_lg 42 %, the T = 8 tail
    //  ~11-15 us); push also wins at 1 column (bench 18:4x).
    const bool push = nt >= a.push_min;
    if (push) { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); } // this CTA has started
    // 1. A (weights, before the wait): gain rows c0 .. c0 + 287 and forget rows 2304 + c0 .. of stream s
    const bf16* W = a.gf_weight + static_cast<std::int64_t>(s) * (2 * kD) * kRank;
    for (int item = tid; item < kTailRows * (kRank / 8); item += kThreads) {
        const int row = item / (kRank / 8), kk = (item - row * (kRank / 8)) * 8;
        const int src = row < kTailChannels ? c0 + row : kD + c0 + (row - kTailChannels);
        cp_async_zfill<16, Cache::cg>(&As[row * kRank + tail_swizzle(row, kk)],
                                      &W[static_cast<std::int64_t>(src) * kRank + kk], 16);
    }
    cp_commit();
    constexpr int kOut = (kTailChannels + kThreads - 1) / kThreads; // 2 output channels per thread at most
    unsigned short wn[kOut];
#pragma unroll
    for (int k = 0; k < kOut; ++k) {
        const int c = tid + k * kThreads;
        wn[k] = c < kTailChannels ? ld_bits(a.norm_weight, static_cast<std::int64_t>(s) * a.w_stream + c0 + c)
                                  : static_cast<unsigned short>(0);
    }
    pdl::wait_for_dependencies(); // PDL: every activation read / write after the producer grid completes
    pdl::trigger_dependents();
    // 4a. dest / msg of this CTA's channels (blend_inputs' outputs), issued first
    constexpr int kItems = (kTailTokens * kTailChannels + kThreads - 1) / kThreads; // 9
    float dv[kItems], mv[kItems];
#pragma unroll
    for (int k = 0; k < kItems; ++k) {
        const int item = tid + k * kThreads;
        dv[k] = mv[k] = 0.0F;
        if (item < nt * kTailChannels) {
            const int j = item / kTailChannels, c = item - j * kTailChannels;
            const std::int64_t t = t0 + j;
            dv[k] = a.dest[s * a.dest_stream + t * kD + c0 + c];
            mv[k] = a.msg[(s * a.columns + t) * kD + c0 + c];
        }
    }
    // 2. B: h = bf16(the partials added in chunk order), zero past the last column (the GEMM's zero fill)
    const std::int64_t plane = static_cast<std::int64_t>(a.streams) * a.columns * kRank;
    static_assert(kTailTokens * kRank == 2 * kThreads, "two B items per thread");
    float p[2][kTailSplits];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int item = tid + u * kThreads;
        const int j = item / kRank, r = item - j * kRank;
        const std::int64_t i = (static_cast<std::int64_t>(s) * a.columns + t0 + j) * kRank + r;
#pragma unroll
        for (int c = 0; c < kTailSplits; ++c) { p[u][c] = j < nt ? a.partials[c * plane + i] : 0.0F; }
    }
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int item = tid + u * kThreads;
        const int j = item / kRank, r = item - j * kRank;
        float h     = 0.0F;
        if (j < nt) {
            h = p[u][0];
#pragma unroll
            for (int c = 1; c < kTailSplits; ++c) { h += p[u][c]; }
        }
        Bs[j * kRank + tail_swizzle(j, r)] = __float2bfloat16_rn(h);
    }
    cp_wait<0>();
    __syncthreads();
    // 3. the gain/forget GEMM of the 36 m16 tiles: warp w takes w, w + 8, ... (controller_tail_kernel's fragments)
    const int warp = tid >> 5, lane = tid & 31, gid = lane >> 2, lid = lane & 3;
    const int a_matrix = lane >> 3, a_inner = lane & 7;
    const int a_row_off = a_inner + ((a_matrix & 1) << 3), a_col_off = (a_matrix >> 1) << 3;
    const int b_inner = lane & 7, b_k_off = ((lane >> 3) & 1) << 3;
    constexpr int kTiles = kTailRows / 16, kPerWarp = (kTiles + kMxWarps - 1) / kMxWarps;
    float acc[kPerWarp][4];
#pragma unroll
    for (int u = 0; u < kPerWarp; ++u) {
        acc[u][0] = acc[u][1] = acc[u][2] = acc[u][3] = 0.0F;
        const int mt = warp + u * kMxWarps;
        if (mt >= kTiles) { continue; }
#pragma unroll
        for (int k_step = 0; k_step < kRank / 16; ++k_step) {
            unsigned af[4], bfr[2];
            const int row = mt * 16 + a_row_off, col = k_step * 16 + a_col_off;
            ldmatrix_x4(af[0], af[1], af[2], af[3], smem_addr(&As[row * kRank + tail_swizzle(row, col)]));
            const int bcol = k_step * 16 + b_k_off;
            ldmatrix_x2(bfr[0], bfr[1], smem_addr(&Bs[b_inner * kRank + tail_swizzle(b_inner, bcol)]));
            mma_bf16(acc[u][0], acc[u][1], acc[u][2], acc[u][3], af[0], af[1], af[2], af[3], bfr[0], bfr[1]);
        }
    }
    __syncthreads(); // G overwrites A
#pragma unroll
    for (int u = 0; u < kPerWarp; ++u) {
        const int mt = warp + u * kMxWarps;
        if (mt >= kTiles) { continue; }
        const int r0 = mt * 16 + gid;
        G[r0 * kGStride + 2 * lid]           = acc[u][0];
        G[r0 * kGStride + 2 * lid + 1]       = acc[u][1];
        G[(r0 + 8) * kGStride + 2 * lid]     = acc[u][2];
        G[(r0 + 8) * kGStride + 2 * lid + 1] = acc[u][3];
    }
    __syncthreads();
    // 4b. x of this CTA's channels (blend_rms_norm_kernel's expression); push: also into CTA j's column buffer
    if (push) { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); } // every CTA of the cluster has started
#pragma unroll
    for (int k = 0; k < kItems; ++k) {
        const int item = tid + k * kThreads;
        if (item < nt * kTailChannels) {
            const int j = item / kTailChannels, c = item - j * kTailChannels;
            const float gain   = tanhf(G[c * kGStride + j]);
            const float forget = tanhf(G[(kTailChannels + c) * kGStride + j]);
            const float x      = (1.0F - forget) * dv[k] + gain * mv[k];
            X[j * kTailChannels + c] = x;
            if (push) { cluster.map_shared_rank(XF, j)[c0 + c] = x; }
        }
    }
    cluster.sync();
    if (push) {
        if (rank < nt) { // column j = rank: blend_rms_norm_kernel's chain and block sum, from this CTA's buffer
            float ss[1] = {0.0F};
#pragma unroll
            for (int i = 0; i < kPer; ++i) {
                const float x = XF[tid + i * kThreads];
                ss[0]         = ss[0] + x * x;
            }
            block_sum_pass<1>(ss, scratch);
            if (tid < kTailCtas) {
                cluster.map_shared_rank(RS, tid)[rank] = rsqrt_approx(__fdiv_rn(ss[0], 2304.0F) + a.eps);
            }
        }
        cluster.sync(); // RS complete; no remote access after this barrier
#pragma unroll
        for (int j = 0; j < kTailTokens; ++j) {
            if (j >= nt) { break; }
            const float rstd = RS[j];
            bf16* o          = a.out + (static_cast<std::int64_t>(s) * a.columns + t0 + j) * kD;
#pragma unroll
            for (int k = 0; k < kOut; ++k) {
                const int c = tid + k * kThreads;
                if (c < kTailChannels) {
                    o[c0 + c] = __float2bfloat16_rn((X[j * kTailChannels + c] * rstd) * bits_to_f32(wn[k]));
                }
            }
        }
        return;
    }
    // 5. every column's sum of squares in blend_rms_norm_kernel's layout (thread tid sums d = tid + 256 i), one pass
    const float* rx[kPer];
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int d = tid + i * kThreads;
        rx[i]       = cluster.map_shared_rank(X, d / kTailChannels) + d % kTailChannels;
    }
    float xs[kTailTokens][kPer];
#pragma unroll
    for (int j = 0; j < kTailTokens; ++j) {
        if (j < nt) {
#pragma unroll
            for (int i = 0; i < kPer; ++i) { xs[j][i] = rx[i][j * kTailChannels]; }
        }
    }
    float ss[kTailTokens];
#pragma unroll
    for (int j = 0; j < kTailTokens; ++j) {
        float acc_ss = 0.0F;
        if (j < nt) {
#pragma unroll
            for (int i = 0; i < kPer; ++i) { acc_ss = acc_ss + xs[j][i] * xs[j][i]; }
        }
        ss[j] = acc_ss;
    }
    // this CTA's reads of the other CTAs' x are complete (their values are in ss): arrive now, wait before exit
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    block_sum_pass<kTailTokens>(ss, scratch);
#pragma unroll
    for (int j = 0; j < kTailTokens; ++j) {
        if (j >= nt) { break; }
        const float rstd = rsqrt_approx(__fdiv_rn(ss[j], 2304.0F) + a.eps);
        bf16* o          = a.out + (static_cast<std::int64_t>(s) * a.columns + t0 + j) * kD;
#pragma unroll
        for (int k = 0; k < kOut; ++k) {
            const int c = tid + k * kThreads;
            if (c < kTailChannels) {
                o[c0 + c] = __float2bfloat16_rn((X[j * kTailChannels + c] * rstd) * bits_to_f32(wn[k]));
            }
        }
    }
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); // the other CTAs may still read this CTA's x
}

/// short_conv_kernel's decode / verify form (W <= kConvFastWidth, one CTA column tile, window update fused, no
/// capture): the taps before the PDL wait, then every load of the row (positions, x, residual, addend; the slot's
/// window once the slot index is known) before the first use.
constexpr int kConvFastWidth = 8;

__global__ void __launch_bounds__(kConvChannelTile) short_conv_kernel(const __grid_constant__ ConvArgs a) {
    const int c     = blockIdx.x * kConvChannelTile + threadIdx.x;
    const int b     = blockIdx.z;
    const bool live = c < a.channels;
    float w[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    if (live) { // weights: before the wait
#pragma unroll
        for (int k = 0; k < 4; ++k) { w[k] = bits_to_f32(ld_bits(a.weight, static_cast<std::int64_t>(c) * 4 + k)); }
    }
    pdl::wait_for_dependencies(); // PDL: every activation read / write after the producer grid completes
    pdl::trigger_dependents();
    if (!live) { return; }
    const std::int64_t slot = a.slots[b];
    const int valid         = a.valid != nullptr ? a.valid[b] : a.width;
    int pos[kConvFastWidth];
    unsigned short xr[kConvFastWidth], ar[kConvFastWidth], rr[kConvFastWidth];
#pragma unroll
    for (int j = 0; j < kConvFastWidth; ++j) {
        pos[j] = 0;
        xr[j] = ar[j] = rr[j] = 0;
        if (j < a.width) {
            const std::int64_t n = static_cast<std::int64_t>(j) + static_cast<std::int64_t>(a.width) * b;
            pos[j]               = a.positions[n];
            xr[j]                = ld_bits(a.x, n * a.x_stride + c);
            if (a.addend != nullptr) { ar[j] = ld_bits(a.addend, n * a.addend_stride + c); }
            if (a.residual != nullptr) { rr[j] = ld_bits(a.residual, n * a.residual_stride + c); }
        }
    }
    // S = [state tap 0, 1, 2, x column 0, 1, ...] (conv_seq); the slot is read only for a row with valid columns
    // (the reference reads or writes it exactly then)
    float seq[3 + kConvFastWidth];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        seq[i] = valid > 0 ? bits_to_f32(ld_bits(a.state, slot * a.slot_stride + i * a.tap_stride + c)) : 0.0F;
    }
#pragma unroll
    for (int j = 0; j < kConvFastWidth; ++j) { seq[3 + j] = bits_to_f32(xr[j]); }
    const float scale = 0.44721359549995793F; // fp32(5 ** -0.5)
#pragma unroll
    for (int j = 0; j < kConvFastWidth; ++j) {
        if (j < a.width && j < valid) {
            const std::int64_t n = static_cast<std::int64_t>(j) + static_cast<std::int64_t>(a.width) * b;
            float acc            = 0.0F;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int lag = 3 - k;
                const float v = (pos[j] - lag >= 0) ? seq[j + k] : 0.0F;
                acc           = fmaf(v, w[k], acc);
            }
            const float cur = seq[3 + j];
            bf16 y          = __float2bfloat16_rn((acc + cur) * scale);
            if (a.addend != nullptr) {
                float base = bits_to_f32(ar[j]);
                if (a.residual != nullptr) { base = bits_to_f32(rr[j]) + base; }
                y = __float2bfloat16_rn(base + __bfloat162float(y));
            }
            a.out[n * a.out_stride + c] = y;
        }
    }
    if (valid > 0) { // conv_window_kernel's update: S[valid + i] (a state tap from registers, or the x column again,
                     // as conv_seq reads it; no dynamic register index, which the compiler turns into local memory)
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const int idx = valid + i;
            float u;
            if (idx < 3) {
                u = idx == 0 ? seq[0] : (idx == 1 ? seq[1] : seq[2]);
            } else {
                const std::int64_t n = static_cast<std::int64_t>(idx - 3) + static_cast<std::int64_t>(a.width) * b;
                u                    = bits_to_f32(ld_bits(a.x, n * a.x_stride + c));
            }
            a.state[slot * a.slot_stride + i * a.tap_stride + c] = __float2bfloat16_rn(u);
        }
    }
}

} // namespace mx

// Reference-kernel switch (tests and A/B): SUROGATE_SERVE_INKLING_MIX_REFERENCE=1 or inkling_mix_use_reference_kernels.
std::atomic<int> g_mix_reference{-1};

bool use_reference_kernels() {
    int v = g_mix_reference.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* raw = std::getenv("SUROGATE_SERVE_INKLING_MIX_REFERENCE");
        v               = raw != nullptr && raw[0] != '\0' && std::atoi(raw) != 0 ? 1 : 0;
        g_mix_reference.store(v, std::memory_order_relaxed);
    }
    return v != 0;
}

/// The latency-cut kernels serve decode / verify widths; prefill keeps the reference kernels (same bits either way).
constexpr std::int32_t kFastColumns = 64;

template <int N>
void attn_res_fast_n(bool add, const dim3& grid, cudaStream_t stream, const AttnResArgs& a) {
    const pdl::LaunchConfig cfg{grid, dim3(kThreads), 0, stream};
    CUDA_CHECK(add ? pdl::launch_dependent(cfg, mx::attn_res_kernel<N, true>, a)
                   : pdl::launch_dependent(cfg, mx::attn_res_kernel<N, false>, a));
}

template <int N>
void blend_inputs_fast_n(bool add, const dim3& grid, cudaStream_t stream, const BlendInputsArgs& a) {
    const pdl::LaunchConfig cfg{grid, dim3(kThreads), 0, stream};
    CUDA_CHECK(add ? pdl::launch_dependent(cfg, mx::blend_inputs_kernel<N, true>, a)
                   : pdl::launch_dependent(cfg, mx::blend_inputs_kernel<N, false>, a));
}

// ------------------------------------------------------------------------------------- validation

[[noreturn]] void fail(const char* op, const std::string& what) {
    throw std::invalid_argument(std::string(op) + ": " + what);
}

PlaneDev plane_dev(const InklingPlane& p, const char* op, const char* name, bool allow_fp32) {
    if (p.data == nullptr) { fail(op, std::string(name) + " is null"); }
    if (p.dtype != DType::BF16 && !(allow_fp32 && p.dtype == DType::FP32)) {
        fail(op, std::string(name) + " has an unsupported dtype");
    }
    if (p.column_stride < kD) { fail(op, std::string(name) + " column stride is below 2304"); }
    return PlaneDev{p.data, p.column_stride, p.dtype == DType::FP32 ? 1 : 0};
}

void require(bool ok, const char* op, const char* what) {
    if (!ok) { fail(op, what); }
}

bool contiguous_shape(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t n1, std::int32_t n2) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous() && t.ne[0] == n0 &&
           t.ne[1] == n1 && t.ne[2] == n2 && t.ne[3] == 1;
}

unsigned grid_count(std::int64_t n, std::int64_t per) {
    return static_cast<unsigned>((n + per - 1) / per);
}

ConvArgs conv_args(const InklingPlane& x, std::int32_t channels, const Tensor* weight,
                   const InklingConvRows& rows, const InklingConvStateView& state,
                   std::int32_t channel_offset, const char* op) {
    require(x.data != nullptr && x.dtype == DType::BF16 && x.column_stride >= channels, op,
            "x must be a BF16 plane at least `channels` wide");
    require(channels > 0 && channel_offset >= 0 && channel_offset + channels <= G::kConvChannels, op,
            "channel range is outside the conv state");
    require(rows.width > 0 && rows.batch > 0, op, "rows need width and batch");
    require(contiguous_shape(rows.positions, DType::I32, rows.width, rows.batch, 1) ||
                contiguous_shape(rows.positions, DType::I32, rows.width * rows.batch, 1, 1),
            op, "positions must be I32 [W, B]");
    require(rows.valid_columns.data == nullptr ||
                contiguous_shape(rows.valid_columns, DType::I32, rows.batch, 1, 1),
            op, "valid_columns must be I32 [B] or empty");
    require(contiguous_shape(rows.state_slots, DType::I32, rows.batch, 1, 1), op,
            "state_slots must be I32 [B]");
    require(state.base != nullptr && state.slot_stride >= 3 * state.tap_stride &&
                state.tap_stride >= G::kConvChannels,
            op, "conv state view is invalid");
    ConvArgs a;
    a.x        = static_cast<const bf16*>(x.data);
    a.x_stride = x.column_stride;
    a.channels = channels;
    if (weight != nullptr) {
        require(contiguous_shape(*weight, DType::BF16, 4, channels, 1), op,
                "weight must be BF16 [4, channels]");
        a.weight = static_cast<const bf16*>(weight->data);
    }
    a.width       = rows.width;
    a.batch       = rows.batch;
    a.positions   = static_cast<const std::int32_t*>(rows.positions.data);
    a.valid       = static_cast<const std::int32_t*>(rows.valid_columns.data);
    a.slots       = static_cast<const std::int32_t*>(rows.state_slots.data);
    a.state       = static_cast<bf16*>(state.base) + channel_offset;
    a.slot_stride = state.slot_stride;
    a.tap_stride  = state.tap_stride;
    return a;
}

} // namespace

void inkling_attn_res(const InklingAttnResSources& sources, std::int32_t columns,
                      const Tensor& queries, const Tensor* norm_weight, float norm_eps, Tensor& out,
                      cudaStream_t stream) {
    constexpr const char* op = "inkling_attn_res";
    require(columns > 0, op, "columns must be positive");
    require(sources.stack_count >= 0 && sources.stack_count <= G::kMaxStack, op,
            "stack_count must be in [0, 5]");
    require(queries.dtype == DType::FP32 && queries.is_contiguous() && queries.ne[0] == kD &&
                queries.ne[1] >= 1 && queries.ne[1] <= 4 && queries.ne[2] == 1 && queries.data,
            op, "queries must be FP32 [2304, Q], Q in 1..4");
    const int q = queries.ne[1];
    AttnResArgs a;
    for (int n = 0; n < sources.stack_count; ++n) {
        a.src[n] = plane_dev(sources.stack[n], op, "stack source", false);
    }
    a.src[sources.stack_count] = plane_dev(sources.last, op, "last", true);
    if (sources.addend.data != nullptr) {
        a.addend     = plane_dev(sources.addend, op, "addend", false);
        a.has_addend = 1;
    }
    a.queries     = static_cast<const float*>(queries.data);
    a.num_queries = q;
    if (norm_weight != nullptr) {
        require(contiguous_shape(*norm_weight, DType::BF16, kD, q, 1), op,
                "norm_weight must be BF16 [2304, Q]");
        require(contiguous_shape(out, DType::BF16, kD, columns, q), op,
                "out must be BF16 [2304, columns, Q] with a norm");
        a.norm_weight = static_cast<const bf16*>(norm_weight->data);
        a.norm_eps    = norm_eps;
    } else {
        require(contiguous_shape(out, DType::FP32, kD, columns, q), op,
                "out must be FP32 [2304, columns, Q] without a norm");
    }
    a.out     = out.data;
    a.columns = columns;
    const dim3 grid(static_cast<unsigned>(columns));
    if (!use_reference_kernels() && columns <= kFastColumns && sources.last.dtype == DType::BF16) {
        const bool add = a.has_addend != 0;
        const dim3 g2(static_cast<unsigned>(columns), static_cast<unsigned>(q)); // one CTA per (column, query)
        switch (sources.stack_count + 1) {
        case 1: attn_res_fast_n<1>(add, g2, stream, a); break;
        case 2: attn_res_fast_n<2>(add, g2, stream, a); break;
        case 3: attn_res_fast_n<3>(add, g2, stream, a); break;
        case 4: attn_res_fast_n<4>(add, g2, stream, a); break;
        case 5: attn_res_fast_n<5>(add, g2, stream, a); break;
        case 6: attn_res_fast_n<6>(add, g2, stream, a); break;
        default: fail(op, "unsupported source count");
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    switch (sources.stack_count + 1) {
    case 1: CUDA_CHECK(pdl::launch_dependent({grid, dim3(kThreads), 0, stream}, attn_res_kernel<1>, a)); break;
    case 2: CUDA_CHECK(pdl::launch_dependent({grid, dim3(kThreads), 0, stream}, attn_res_kernel<2>, a)); break;
    case 3: CUDA_CHECK(pdl::launch_dependent({grid, dim3(kThreads), 0, stream}, attn_res_kernel<3>, a)); break;
    case 4: CUDA_CHECK(pdl::launch_dependent({grid, dim3(kThreads), 0, stream}, attn_res_kernel<4>, a)); break;
    case 5: CUDA_CHECK(pdl::launch_dependent({grid, dim3(kThreads), 0, stream}, attn_res_kernel<5>, a)); break;
    case 6: CUDA_CHECK(pdl::launch_dependent({grid, dim3(kThreads), 0, stream}, attn_res_kernel<6>, a)); break;
    default: fail(op, "unsupported source count");
    }
    CUDA_CHECK(cudaGetLastError());
}

void inkling_embed(const Tensor& ids, const void* table, std::int32_t table_rows,
                   const Tensor& norm_weight, float eps, const InklingPlane& out,
                   cudaStream_t stream) {
    constexpr const char* op = "inkling_embed";
    require(ids.dtype == DType::I32 && ids.is_contiguous() && ids.data && ids.numel() > 0, op,
            "ids must be contiguous I32");
    require(table != nullptr && table_rows > 0, op, "table is missing");
    require(contiguous_shape(norm_weight, DType::BF16, kD, 1, 1), op, "norm_weight must be BF16 [2304]");
    const PlaneDev o = plane_dev(out, op, "out", false);
    embed_kernel<<<static_cast<unsigned>(ids.numel()), kThreads, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const bf16*>(table),
        static_cast<const bf16*>(norm_weight.data), eps, static_cast<bf16*>(out.data), o.stride);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_rms_norm(const InklingPlane& x, std::int32_t columns, const Tensor& weight, float eps,
                      Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "inkling_rms_norm";
    require(columns > 0, op, "columns must be positive");
    const PlaneDev xp = plane_dev(x, op, "x", false);
    require(weight.dtype == DType::BF16 && weight.is_contiguous() && weight.ne[0] == kD &&
                weight.ne[1] >= 1 && weight.ne[2] == 1 && weight.data,
            op, "weight must be BF16 [2304, Q]");
    const int q = weight.ne[1];
    require(contiguous_shape(out, DType::BF16, kD, columns, q), op, "out must be BF16 [2304, columns, Q]");
    rms_norm_kernel<<<static_cast<unsigned>(columns), kThreads, 0, stream>>>(
        xp, static_cast<const bf16*>(weight.data), q, eps, static_cast<bf16*>(out.data), columns);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_pair_rms_norm(const Tensor& dest, const Tensor& message, Tensor& out,
                           cudaStream_t stream) {
    constexpr const char* op = "inkling_pair_rms_norm";
    const int t = message.ne[1];
    const int s = message.ne[2];
    require(contiguous_shape(message, DType::FP32, kD, t, s) && t > 0 && s > 0, op,
            "message must be FP32 [2304, T, S]");
    require(contiguous_shape(dest, DType::FP32, kD, t, s) || contiguous_shape(dest, DType::FP32, kD, t, 1),
            op, "dest must be FP32 [2304, T, S] or [2304, T, 1]");
    require(contiguous_shape(out, DType::BF16, 2 * kD, t, s), op, "out must be BF16 [4608, T, S]");
    const std::int64_t dest_stream = dest.ne[2] == 1 ? 0 : static_cast<std::int64_t>(t) * kD;
    pair_rms_norm_kernel<<<dim3(static_cast<unsigned>(t), static_cast<unsigned>(s)), kThreads, 0, stream>>>(
        static_cast<const float*>(dest.data), dest_stream, static_cast<const float*>(message.data),
        static_cast<bf16*>(out.data), t);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_blend_inputs(const InklingAttnResSources& dest_sources, const Tensor& dest_queries,
                          const InklingAttnResSources& bank, const Tensor& message_queries, std::int32_t columns,
                          Tensor& dest, Tensor& message, Tensor& pair, cudaStream_t stream) {
    constexpr const char* op = "inkling_blend_inputs";
    require(columns > 0, op, "columns must be positive");
    require(dest_sources.stack_count >= 0 && dest_sources.stack_count <= G::kMaxStack, op,
            "stack_count must be in [0, 5]");
    require(bank.stack_count == 1 && bank.addend.data == nullptr, op, "the bank is one stack source + last");
    const int streams = message_queries.ne[1];
    const int dq      = dest_queries.ne[1];
    require(message_queries.dtype == DType::FP32 && message_queries.ne[0] == kD && streams >= 1 && streams <= 3 &&
                message_queries.data != nullptr,
            op, "message_queries must be FP32 [2304, S], S in 1..3");
    require(dest_queries.dtype == DType::FP32 && dest_queries.ne[0] == kD && (dq == 1 || dq == streams) &&
                dest_queries.data != nullptr,
            op, "dest_queries must be FP32 [2304, 1 or S]");
    require(contiguous_shape(dest, DType::FP32, kD, columns, dq), op, "dest must be FP32 [2304, T, DQ]");
    require(contiguous_shape(message, DType::FP32, kD, columns, streams), op, "message must be FP32 [2304, T, S]");
    require(contiguous_shape(pair, DType::BF16, 2 * kD, columns, streams), op, "pair must be BF16 [4608, T, S]");
    BlendInputsArgs a;
    for (int n = 0; n < dest_sources.stack_count; ++n) {
        a.src[n] = plane_dev(dest_sources.stack[n], op, "stack source", false);
    }
    a.src[dest_sources.stack_count] = plane_dev(dest_sources.last, op, "last", true);
    if (dest_sources.addend.data != nullptr) {
        a.addend     = plane_dev(dest_sources.addend, op, "addend", false);
        a.has_addend = 1;
    }
    a.bank[0]      = plane_dev(bank.stack[0], op, "bank source", false);
    a.bank[1]      = plane_dev(bank.last, op, "bank last", true);
    a.dest_queries = static_cast<const float*>(dest_queries.data);
    a.dest_q       = dq;
    a.msg_queries  = static_cast<const float*>(message_queries.data);
    a.streams      = streams;
    a.dest         = static_cast<float*>(dest.data);
    a.msg          = static_cast<float*>(message.data);
    a.pair         = static_cast<bf16*>(pair.data);
    a.columns      = columns;
    const dim3 grid(static_cast<unsigned>(columns));
    if (!use_reference_kernels() && columns <= kFastColumns && dest_sources.last.dtype == DType::BF16 &&
        bank.stack[0].dtype == DType::BF16 && bank.last.dtype == DType::BF16) {
        const bool add = a.has_addend != 0;
        const dim3 g2(static_cast<unsigned>(columns), static_cast<unsigned>(dq + streams)); // (column, role)
        switch (dest_sources.stack_count + 1) {
        case 1: blend_inputs_fast_n<1>(add, g2, stream, a); break;
        case 2: blend_inputs_fast_n<2>(add, g2, stream, a); break;
        case 3: blend_inputs_fast_n<3>(add, g2, stream, a); break;
        case 4: blend_inputs_fast_n<4>(add, g2, stream, a); break;
        case 5: blend_inputs_fast_n<5>(add, g2, stream, a); break;
        case 6: blend_inputs_fast_n<6>(add, g2, stream, a); break;
        default: fail(op, "unsupported source count");
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    switch (dest_sources.stack_count + 1) {
    case 1: CUDA_CHECK(pdl::launch_dependent({grid, dim3(kThreads), 0, stream}, blend_inputs_kernel<1>, a)); break;
    case 2: CUDA_CHECK(pdl::launch_dependent({grid, dim3(kThreads), 0, stream}, blend_inputs_kernel<2>, a)); break;
    case 3: CUDA_CHECK(pdl::launch_dependent({grid, dim3(kThreads), 0, stream}, blend_inputs_kernel<3>, a)); break;
    case 4: CUDA_CHECK(pdl::launch_dependent({grid, dim3(kThreads), 0, stream}, blend_inputs_kernel<4>, a)); break;
    case 5: CUDA_CHECK(pdl::launch_dependent({grid, dim3(kThreads), 0, stream}, blend_inputs_kernel<5>, a)); break;
    case 6: CUDA_CHECK(pdl::launch_dependent({grid, dim3(kThreads), 0, stream}, blend_inputs_kernel<6>, a)); break;
    default: fail(op, "unsupported source count");
    }
    CUDA_CHECK(cudaGetLastError());
}

bool inkling_controller_tail_supported(std::int32_t columns) noexcept {
    static const bool available = [] {
        if (std::getenv("SUROGATE_SERVE_INKLING_NO_TAIL") != nullptr) { return false; }
        int device = 0, cluster = 0;
        if (cudaGetDevice(&device) != cudaSuccess) { return false; }
        if (cudaDeviceGetAttribute(&cluster, cudaDevAttrClusterLaunch, device) != cudaSuccess) { return false; }
        return cluster != 0;
    }();
    return available && columns > 0 && columns <= 32;
}

void inkling_controller_tail(const float* down_partials, const void* gain_forget_weight, const Tensor& dest,
                             const Tensor& message, const Tensor& norm_weight, float eps, Tensor& out,
                             cudaStream_t stream) {
    constexpr const char* op = "inkling_controller_tail";
    const int t = message.ne[1];
    const int s = message.ne[2];
    require(down_partials != nullptr && gain_forget_weight != nullptr, op, "partials and weights are required");
    require(contiguous_shape(message, DType::FP32, kD, t, s) && t > 0 && s > 0 && s <= 65535, op,
            "message must be FP32 [2304, T, S]");
    require(contiguous_shape(dest, DType::FP32, kD, t, s) || contiguous_shape(dest, DType::FP32, kD, t, 1),
            op, "dest must be FP32 [2304, T, S] or [2304, T, 1]");
    require(contiguous_shape(norm_weight, DType::BF16, kD, s, 1) ||
                contiguous_shape(norm_weight, DType::BF16, kD, 1, 1),
            op, "norm_weight must be BF16 [2304, S] or [2304, 1]");
    require(contiguous_shape(out, DType::BF16, kD, t, s), op, "out must be BF16 [2304, T, S]");
    require((reinterpret_cast<std::uintptr_t>(gain_forget_weight) & 15u) == 0, op, "weights must be 16-byte aligned");
    TailArgs a;
    a.partials    = down_partials;
    a.gf_weight   = static_cast<const bf16*>(gain_forget_weight);
    a.dest        = static_cast<const float*>(dest.data);
    a.dest_stream = dest.ne[2] == 1 ? 0 : static_cast<std::int64_t>(t) * kD;
    a.msg         = static_cast<const float*>(message.data);
    a.norm_weight = static_cast<const bf16*>(norm_weight.data);
    a.w_stream    = norm_weight.ne[1] == 1 ? 0 : kD;
    a.eps         = eps;
    a.out         = static_cast<bf16*>(out.data);
    a.columns     = t;
    a.streams     = s;
    const dim3 grid(static_cast<unsigned>(kTailCtas * ((t + kTailTokens - 1) / kTailTokens)), static_cast<unsigned>(s));
    if (!use_reference_kernels()) {
        // SUROGATE_SERVE_INKLING_TAIL_PUSH_MIN (A/B knob): the push scheme from this many columns of a cluster on.
        // Default 1 = always (bench 18:4x, L2-cold T = 1: push 2.15 us vs pull 2.74 us per call; same bits).
        static const int push_min = [] {
            const char* e = std::getenv("SUROGATE_SERVE_INKLING_TAIL_PUSH_MIN");
            return e != nullptr ? std::max(1, std::atoi(e)) : 1;
        }();
        a.push_min = push_min;
        constexpr int kSmem = kTailSmem + kD * 4 + kTailTokens * 4; // + the push scheme's column buffer and rstd
        CUDA_CHECK(set_func_attribute_per_device(mx::controller_tail_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 kSmem));
        CUDA_CHECK(pdl::launch_dependent({grid, dim3(kThreads), static_cast<std::size_t>(kSmem), stream},
                                         mx::controller_tail_kernel, a));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    CUDA_CHECK(set_func_attribute_per_device(controller_tail_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             kTailSmem));
    CUDA_CHECK(pdl::launch_dependent({grid, dim3(kThreads), static_cast<std::size_t>(kTailSmem), stream},
                                     controller_tail_kernel, a));
    CUDA_CHECK(cudaGetLastError());
}

void inkling_blend_rms_norm(const Tensor& dest, const Tensor& message, const Tensor& gain_forget,
                            const Tensor& norm_weight, float eps, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "inkling_blend_rms_norm";
    const int t = message.ne[1];
    const int s = message.ne[2];
    require(contiguous_shape(message, DType::FP32, kD, t, s) && t > 0 && s > 0, op,
            "message must be FP32 [2304, T, S]");
    require(contiguous_shape(dest, DType::FP32, kD, t, s) || contiguous_shape(dest, DType::FP32, kD, t, 1),
            op, "dest must be FP32 [2304, T, S] or [2304, T, 1]");
    require(contiguous_shape(gain_forget, DType::FP32, 2 * kD, t, s), op,
            "gain_forget must be FP32 [4608, T, S]");
    require(contiguous_shape(norm_weight, DType::BF16, kD, s, 1) ||
                contiguous_shape(norm_weight, DType::BF16, kD, 1, 1),
            op, "norm_weight must be BF16 [2304, S] or [2304, 1]");
    require(contiguous_shape(out, DType::BF16, kD, t, s), op, "out must be BF16 [2304, T, S]");
    const std::int64_t dest_stream = dest.ne[2] == 1 ? 0 : static_cast<std::int64_t>(t) * kD;
    const int w_stream             = norm_weight.ne[1] == 1 ? 0 : kD;
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(static_cast<unsigned>(t), static_cast<unsigned>(s)), dim3(kThreads), 0, stream}, blend_rms_norm_kernel,
        static_cast<const float*>(dest.data), dest_stream, static_cast<const float*>(message.data),
        static_cast<const float*>(gain_forget.data), static_cast<const bf16*>(norm_weight.data), w_stream, eps,
        static_cast<bf16*>(out.data), static_cast<std::int64_t>(t)));
    CUDA_CHECK(cudaGetLastError());
}

void inkling_short_conv(const InklingPlane& x, std::int32_t channels, const Tensor& weight,
                        const InklingConvRows& rows, const InklingConvStateView& state,
                        std::int32_t channel_offset, const InklingPlane* residual,
                        const InklingPlane* addend, const InklingPlane& out, cudaStream_t stream,
                        std::int32_t capture_column, std::int32_t capture_slot) {
    constexpr const char* op = "inkling_short_conv";
    ConvArgs a = conv_args(x, channels, &weight, rows, state, channel_offset, op);
    require(out.data != nullptr && out.dtype == DType::BF16 && out.column_stride >= channels, op,
            "out must be a BF16 plane");
    a.out        = static_cast<bf16*>(out.data);
    a.out_stride = out.column_stride;
    if (addend != nullptr) {
        require(addend->data != nullptr && addend->dtype == DType::BF16, op, "addend must be BF16");
        a.addend        = static_cast<const bf16*>(addend->data);
        a.addend_stride = addend->column_stride;
    }
    if (residual != nullptr) {
        require(addend != nullptr, op, "a residual needs an addend (the reference's form)");
        require(residual->data != nullptr && residual->dtype == DType::BF16, op,
                "residual must be BF16");
        a.residual        = static_cast<const bf16*>(residual->data);
        a.residual_stride = residual->column_stride;
    }
    require(capture_slot < 0 || (rows.batch == 1 && capture_column >= 0 && capture_column <= rows.width),
            op, "a capture needs one row and a column in [0, W]");
    const dim3 grid(grid_count(channels, kConvChannelTile), grid_count(rows.width, kConvColumnTile),
                    static_cast<unsigned>(rows.batch));
    a.fuse_window = capture_slot < 0 && rows.width <= kConvColumnTile ? 1 : 0;
    if (a.fuse_window && rows.width <= mx::kConvFastWidth && !use_reference_kernels()) {
        CUDA_CHECK(pdl::launch_dependent({grid, dim3(kConvChannelTile), 0, stream}, mx::short_conv_kernel, a));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    CUDA_CHECK(pdl::launch_dependent({grid, dim3(kConvChannelTile), 0, stream}, short_conv_kernel, a));
    CUDA_CHECK(cudaGetLastError());
    if (!a.fuse_window) {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(grid_count(channels, kConvChannelTile), static_cast<unsigned>(rows.batch)), dim3(kConvChannelTile), 0,
             stream},
            conv_window_kernel, a, capture_column, capture_slot, 1));
        CUDA_CHECK(cudaGetLastError());
    }
}

void inkling_conv_capture_window(const InklingPlane& x, std::int32_t channels,
                                 const InklingConvRows& rows, const InklingConvStateView& state,
                                 std::int32_t channel_offset, std::int32_t column,
                                 std::int32_t dst_slot, cudaStream_t stream) {
    constexpr const char* op = "inkling_conv_capture_window";
    const ConvArgs a = conv_args(x, channels, nullptr, rows, state, channel_offset, op);
    require(rows.batch == 1 && column >= 0 && column <= rows.width && dst_slot >= 0, op,
            "one row, a column in [0, W] and a slot");
    conv_window_kernel<<<dim3(grid_count(channels, kConvChannelTile), 1), kConvChannelTile, 0, stream>>>(
        a, column, dst_slot, 0);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_conv_update_window(const InklingPlane& x, std::int32_t channels, const InklingConvRows& rows,
                                const InklingConvStateView& state, std::int32_t channel_offset,
                                cudaStream_t stream) {
    constexpr const char* op = "inkling_conv_update_window";
    const ConvArgs a = conv_args(x, channels, nullptr, rows, state, channel_offset, op);
    conv_window_kernel<<<dim3(grid_count(channels, kConvChannelTile), static_cast<unsigned>(rows.batch)),
                         kConvChannelTile, 0, stream>>>(a, -1, -1, 1);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_conv_copy_slot(const InklingConvStateView& state, std::int32_t channel_offset,
                            std::int32_t channels, std::int32_t src, std::int32_t dst,
                            cudaStream_t stream) {
    constexpr const char* op = "inkling_conv_copy_slot";
    require(state.base != nullptr && src >= 0 && dst >= 0 && channels > 0 && channel_offset >= 0 &&
                channel_offset + channels <= G::kConvChannels,
            op, "invalid state view, slots or channel range");
    if (src == dst) { return; }
    conv_copy_slot_kernel<<<grid_count(channels, 256), 256, 0, stream>>>(
        static_cast<bf16*>(state.base) + channel_offset, state.slot_stride, state.tap_stride, channels,
        src, dst);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_conv_backup(const InklingConvPool& pool, const Tensor& slots, std::int32_t batch,
                         const InklingConvRecords& records, cudaStream_t stream) {
    constexpr const char* op = "inkling_conv_backup";
    require(pool.base && pool.visits > 0 && records.old && records.visits == pool.visits, op,
            "pool and records disagree");
    require(batch > 0 && batch <= records.batch_max && slots.dtype == DType::I32 &&
                slots.numel() >= batch && slots.data,
            op, "slots must be I32 [B], B <= the records' batch");
    conv_backup_kernel<<<dim3(grid_count(G::kConvChannels, 256), static_cast<unsigned>(pool.visits),
                              static_cast<unsigned>(batch)),
                         256, 0, stream>>>(static_cast<const bf16*>(pool.base), pool.visit_stride,
                                           pool.slot_stride, static_cast<const std::int32_t*>(slots.data),
                                           static_cast<bf16*>(records.old), pool.visits);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_conv_record(const InklingPlane& x, std::int32_t channels, std::int32_t width,
                         std::int32_t batch, std::int32_t visit, std::int32_t channel_offset,
                         const InklingConvRecords& records, cudaStream_t stream) {
    constexpr const char* op = "inkling_conv_record";
    require(x.data && x.dtype == DType::BF16 && x.column_stride >= channels, op, "x must be a BF16 plane");
    require(records.inputs && visit >= 0 && visit < records.visits && width > 0 &&
                width <= records.width_max && batch > 0 && batch <= records.batch_max,
            op, "round outside the records");
    require(channels > 0 && channel_offset >= 0 && channel_offset + channels <= G::kConvChannels, op,
            "channel range is outside the conv state");
    conv_record_kernel<<<dim3(grid_count(channels, 256), static_cast<unsigned>(width),
                              static_cast<unsigned>(batch)),
                         256, 0, stream>>>(static_cast<const bf16*>(x.data), x.column_stride, channels,
                                           width, visit, channel_offset,
                                           static_cast<bf16*>(records.inputs), records.visits,
                                           records.width_max);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_conv_fold(const InklingConvPool& pool, std::span<const InklingConvFoldRow> rows,
                       const InklingConvRecords& records, cudaStream_t stream) {
    constexpr const char* op = "inkling_conv_fold";
    require(pool.base && records.old && records.inputs && records.visits == pool.visits, op,
            "pool and records disagree");
    if (rows.empty()) { return; }
    require(rows.size() <= static_cast<std::size_t>(kMaxFoldRows), op, "too many rows");
    FoldArgs f;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        require(r.row >= 0 && r.row < records.batch_max && r.slot >= 0 && r.committed >= 0 &&
                    r.committed <= records.width_max,
                op, "row outside the records");
        f.rows[i] = r;
    }
    f.count = static_cast<int>(rows.size());
    conv_fold_kernel<<<dim3(grid_count(G::kConvChannels, 256), static_cast<unsigned>(pool.visits),
                            static_cast<unsigned>(rows.size())),
                       256, 0, stream>>>(f, static_cast<bf16*>(pool.base), pool.visit_stride,
                                         pool.slot_stride, static_cast<const bf16*>(records.old),
                                         static_cast<const bf16*>(records.inputs), records.visits,
                                         records.width_max);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_mix_use_reference_kernels(bool on) noexcept {
    g_mix_reference.store(on ? 1 : 0, std::memory_order_relaxed);
}

void inkling_softcap_f32(float* logits, std::int32_t rows, std::int32_t tokens, std::int32_t ldc,
                         float cap, cudaStream_t stream) {
    constexpr const char* op = "inkling_softcap_f32";
    require(logits != nullptr && rows > 0 && tokens > 0 && ldc >= rows && cap > 0.0F, op,
            "invalid logits or cap");
    const std::int64_t total = static_cast<std::int64_t>(rows) * tokens;
    softcap_f32_kernel<<<grid_count(total, 256), 256, 0, stream>>>(logits, rows, tokens, ldc, cap);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops
