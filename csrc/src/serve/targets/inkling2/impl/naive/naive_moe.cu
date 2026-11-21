// NAIVE Inkling-2 routed experts (lane e1; replaced by lane d's `inkling_moe`): `inkling_route`,
// `inkling_quant_nvfp4_4o6`, `inkling_quant_mxfp8`, `inkling_softcap_swiglu` and `inkling_moe(act_quant=True)` of
// inkling_ref.py. One warp per (token-slot, output row) with an fp32 dot over dequantised operands. Tokens are
// processed in tiles of kTile to bound the scratch. Correctness only.

#include "targets/inkling2/impl/naive/naive.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace sinfer::targets::inkling2::naive {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kD      = 2304;
constexpr int kI      = 1152;
constexpr int kE      = 128;
constexpr int kTopK   = 8;
constexpr int kTile   = 256; // tokens per pass

__device__ __forceinline__ float f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float e4m3(std::uint8_t bits) {
    __nv_fp8_e4m3 v;
    v.__x = bits;
    return static_cast<float>(v);
}
__device__ __forceinline__ float r8(float x) { return static_cast<float>(__nv_fp8_e4m3(x)); }

__device__ __constant__ float kE2m1[16] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
                                           -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};

/// e2m1 RNE value of x (thresholds of `ops/nvfp4.py` `_e2m1`, saturating).
__device__ __forceinline__ float e2m1_value(float x) {
    const float a = fabsf(x);
    float v;
    if (a <= 0.25F) { v = 0.0F; }
    else if (a < 0.75F) { v = 0.5F; }
    else if (a <= 1.25F) { v = 1.0F; }
    else if (a < 1.75F) { v = 1.5F; }
    else if (a <= 2.5F) { v = 2.0F; }
    else if (a < 3.5F) { v = 3.0F; }
    else if (a <= 5.0F) { v = 4.0F; }
    else { v = 6.0F; }
    return signbit(x) ? -v : v;
}

/// e8m0 exponent -> 2^(e - 127) (e = 0 -> 2^-127).
__device__ __forceinline__ float e8m0(std::uint8_t e) {
    return e == 0 ? __int_as_float(0x00400000) : __int_as_float(static_cast<int>(e) << 23);
}

/// Stored NVFP4 block-scale byte of (row, group) in `blockscale-k16-m128x4-v1`.
__device__ __forceinline__ std::int64_t nvfp4_scale_index(std::int64_t row, int group, int groups) {
    const int k_tiles  = groups / 4;
    const std::int64_t n128 = row / 128;
    const int i4  = static_cast<int>((row % 128) / 32);
    const int i32 = static_cast<int>(row % 32);
    const int kt  = group / 4;
    const int j4  = group % 4;
    return ((((n128 * k_tiles + kt) * 32 + i32) * 4 + i4) * 4 + j4);
}

/// Dequantised weight (without the NVFP4 per-expert global) of bank row `row`, column `c`.
__device__ __forceinline__ float weight_value(const ExpertBankView& w, int nvfp4, std::int64_t row, int c) {
    if (nvfp4) {
        const std::uint8_t byte = w.codes[row * (w.columns / 2) + c / 2];
        const int code          = (c & 1) ? (byte >> 4) : (byte & 15);
        return kE2m1[code] * e4m3(w.scales[nvfp4_scale_index(row, c / 16, w.columns / 16)]);
    }
    return e4m3(w.codes[row * w.columns + c]) * e8m0(w.scales[row * (w.columns / 32) + c / 32]);
}

// ---------------------------------------------------------------------------------------------- route

__global__ void __launch_bounds__(kE) route_kernel(const bf16* x, const bf16* router, const float* bias,
                                                    std::int32_t* ids, float* gammas, std::int32_t* capture) {
    __shared__ float logit_s[kE];
    __shared__ float key_s[kE];
    const int t = blockIdx.x;
    const int e = threadIdx.x;
    const bf16* xr = x + static_cast<std::int64_t>(t) * kD;
    const bf16* wr = router + static_cast<std::int64_t>(e) * kD;
    float logit = 0.0F;
    for (int g0 = 0; g0 < kD; g0 += 1024) {
        float part = 0.0F;
        const int g1 = min(g0 + 1024, kD);
        for (int k = g0; k < g1; ++k) { part = fmaf(f(xr[k]), f(wr[k]), part); }
        logit = g0 == 0 ? part : logit + part;
    }
    const float sig = 1.0F / (1.0F + expf(-logit));
    logit_s[e]      = logit;
    key_s[e]        = sig + bias[e];
    __syncthreads();
    if (e != 0) { return; }
    int chosen[kTopK];
    unsigned long long used0 = 0ULL, used1 = 0ULL;
    for (int s = 0; s < kTopK; ++s) {
        int best = -1;
        for (int c = 0; c < kE; ++c) {
            const bool used = c < 64 ? ((used0 >> c) & 1ULL) : ((used1 >> (c - 64)) & 1ULL);
            if (used) { continue; }
            if (best < 0 || key_s[c] > key_s[best]) { best = c; }
        }
        chosen[s] = best;
        if (best < 64) { used0 |= 1ULL << best; } else { used1 |= 1ULL << (best - 64); }
    }
    float lp[kTopK];
    float mx = -INFINITY;
    for (int s = 0; s < kTopK; ++s) {
        const float l = logit_s[chosen[s]];
        lp[s]         = fminf(l, 0.0F) - log1pf(expf(-fabsf(l)));
        mx            = fmaxf(mx, lp[s]);
    }
    float sh[kTopK];
    for (int s = 0; s < kTopK; ++s) { sh[s] = expf(lp[s] - mx); }
    const float total = ((sh[0] + sh[2]) + (sh[1] + sh[3])) + ((sh[4] + sh[6]) + (sh[5] + sh[7]));
    const float lse   = mx + logf(total);
    for (int s = 0; s < kTopK; ++s) {
        ids[t * kTopK + s]    = chosen[s];
        gammas[t * kTopK + s] = expf(lp[s] - lse) * 8.0F;
        if (capture != nullptr) { capture[static_cast<std::int64_t>(t) * kTopK + s] = chosen[s]; }
    }
}

// ------------------------------------------------------------------------------- activation quantisers

/// One row of K values -> dequantised (without the outer scale) values + amax. NVFP4 four-over-six or MXFP8.
/// One CTA per row; 256 threads; threads own 16-blocks (NVFP4) or 32-blocks (MXFP8).
__global__ void __launch_bounds__(256) quant_kernel(const bf16* x, int k, int nvfp4, float* values, float* amax_out) {
    __shared__ float red[8];
    const std::int64_t row = blockIdx.x;
    const bf16* xr        = x + row * k;
    float* vr             = values + row * k;
    float local = 0.0F;
    for (int c = threadIdx.x; c < k; c += 256) { local = fmaxf(local, fabsf(f(xr[c]))); }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) { local = fmaxf(local, __shfl_xor_sync(0xffffffffU, local, off)); }
    if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = local; }
    __syncthreads();
    float amax = red[0];
    for (int w = 1; w < 8; ++w) { amax = fmaxf(amax, red[w]); }
    if (threadIdx.x == 0) { amax_out[row] = amax; }
    if (nvfp4) {
        const float enc   = fminf(1536.0F / amax, FLT_MAX);
        const float dec   = 1.0F / enc;
        const float outer = amax * (1.0F / 1536.0F);
        const float e6    = enc * (1.0F / 6.0F);
        const float e4    = enc * 0.25F;
        for (int blk = threadIdx.x; blk < k / 16; blk += 256) {
            float xb[16];
            float bmax = 0.0F;
            for (int i = 0; i < 16; ++i) {
                xb[i] = f(xr[blk * 16 + i]);
                bmax  = fmaxf(bmax, fabsf(xb[i]));
            }
            const float s6   = r8(fminf(bmax * e6, 448.0F));
            const float s4   = r8(fminf(bmax * e4, 448.0F));
            const float inv6 = fminf(1.0F / (s6 * dec), FLT_MAX);
            const float inv4 = fminf(1.0F / (s4 * dec), FLT_MAX);
            float v6[16], v4[16], e6s[16], e4s[16];
            for (int i = 0; i < 16; ++i) {
                v6[i]  = e2m1_value(xb[i] * inv6);
                v4[i]  = e2m1_value(xb[i] * inv4);
                // |rec - x| with rec = (v * s) * outer fused into one rounding, as the reference kernel
                // compiles it (lane d: 3 of 36,864 blocks flip their choice unfused).
                e6s[i] = fabsf(fmaf(v6[i] * s6, outer, -xb[i]));
                e4s[i] = fabsf(fmaf(v4[i] * s4, outer, -xb[i]));
            }
            for (int width = 16; width > 1; width /= 2) { // pairwise tree of adjacent pairs
                for (int i = 0; i < width / 2; ++i) {
                    e6s[i] = e6s[2 * i] + e6s[2 * i + 1];
                    e4s[i] = e4s[2 * i] + e4s[2 * i + 1];
                }
            }
            const bool four = e4s[0] < e6s[0];
            for (int i = 0; i < 16; ++i) { vr[blk * 16 + i] = four ? v4[i] * s4 : v6[i] * s6; }
        }
    } else {
        for (int blk = threadIdx.x; blk < k / 32; blk += 256) {
            float bmax = 0.0F;
            for (int i = 0; i < 32; ++i) { bmax = fmaxf(bmax, fabsf(f(xr[blk * 32 + i]))); }
            const float raw = bmax * (1.0F / 448.0F);
            int exponent    = ((__float_as_int(raw) + 0x007FFFFF) & 0x7F800000) >> 23;
            if (raw <= __int_as_float(0x00400000)) { exponent = 0; }
            exponent = min(exponent, 254);
            const float scale   = fmaxf(__int_as_float(exponent << 23), __int_as_float(0x00400000));
            const float inverse = exponent == 0 ? __int_as_float(0x7F000000) : 1.0F / scale; // 2^127
            for (int i = 0; i < 32; ++i) {
                const float v = fminf(fmaxf(f(xr[blk * 32 + i]) * inverse, -448.0F), 448.0F);
                vr[blk * 32 + i] = r8(v) * e8m0(static_cast<std::uint8_t>(exponent));
            }
        }
    }
}

// ------------------------------------------------------------------------------------------- GEMMs

struct GemmArgs {
    const float* a;          // [rows_a][k] dequantised activations, row = token-slot (or token for w13)
    const float* a_amax;     // [rows_a]
    int a_per_token;         // 1: activation row = token (w13); 0: = token-slot (w2)
    const std::int32_t* ids; // [T][8]
    ExpertBankView w;
    int nvfp4;
    int k;
    int rows;                // output rows per expert
    bf16* out;               // [T*8][rows]
};

/// One warp per (token-slot, output row): out = bf16(acc * row scale) (NVFP4) or bf16(acc) (MXFP8).
__global__ void __launch_bounds__(256) expert_gemm_kernel(const __grid_constant__ GemmArgs g) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const std::int64_t ts = blockIdx.x;             // token * 8 + slot
    const int r           = blockIdx.y * 8 + warp;  // output row
    if (r >= g.rows) { return; }
    const std::int64_t t   = ts / kTopK;
    const int e            = g.ids[ts];
    const std::int64_t arow = g.a_per_token ? t : ts;
    const float* av        = g.a + arow * g.k;
    const std::int64_t wrow = static_cast<std::int64_t>(e) * g.rows + r;
    float acc = 0.0F;
    for (int c = lane; c < g.k; c += 32) { acc = fmaf(av[c], weight_value(g.w, g.nvfp4, wrow, c), acc); }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) { acc += __shfl_xor_sync(0xffffffffU, acc, off); }
    if (lane != 0) { return; }
    float y = acc;
    if (g.nvfp4) {
        const float rcp   = 1.0F / (1536.0F * g.w.outer_scale_denominator);
        const float scale = (g.a_amax[arow] * g.w.amax[e]) * rcp;
        y                 = acc * scale;
    }
    g.out[ts * g.rows + r] = __float2bfloat16_rn(y);
}

__global__ void swiglu_kernel(const bf16* gate_up, const float* gammas, bf16* y, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const std::int64_t ts = i / kI;
    const int jj          = static_cast<int>(i % kI);
    const float g = f(gate_up[ts * (2 * kI) + 2 * jj]);
    const float u = f(gate_up[ts * (2 * kI) + 2 * jj + 1]);
    const float sig = 1.0F / (1.0F + expf(-g));
    const float gc  = 4.0F * tanhf(g * 0.25F);
    const float uc  = 25.0F * tanhf(u / 25.0F);
    y[i] = __float2bfloat16_rn(((gc * sig) * uc) * gammas[ts]);
}

__global__ void combine_kernel(const bf16* down, bf16* out, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const std::int64_t t = i / kD;
    const int r          = static_cast<int>(i % kD);
    float acc            = 0.0F;
    for (int s = 0; s < kTopK; ++s) { acc = acc + f(down[(t * kTopK + s) * kD + r]); }
    out[i] = __float2bfloat16_rn(acc);
}

std::size_t round_up(std::size_t bytes) { return (bytes + 255U) & ~std::size_t{255}; }

} // namespace

std::size_t moe_workspace_bytes(std::int32_t tokens) {
    const std::size_t t  = static_cast<std::size_t>(std::min(tokens, kTile));
    const std::size_t ts = t * kTopK;
    return round_up(ts * 4) + round_up(ts * 4) + round_up(t * kD * 4) + round_up(t * 4) + round_up(ts * 2 * kI * 2) +
           round_up(ts * kI * 2) + round_up(ts * kI * 4) + round_up(ts * 4) + round_up(ts * kD * 2) + 4096;
}

void moe(const MoeCall& call, WorkspaceArena& workspace, cudaStream_t stream) {
    if (call.x == nullptr || call.layer.router == nullptr || call.out == nullptr || call.tokens <= 0) {
        throw std::invalid_argument("naive inkling moe: incomplete call");
    }
    const MoeLayer& L = call.layer;
    const int nvfp4   = L.w13.nvfp4 ? 1 : 0;
    auto scope = workspace.scope();
    const int tile = std::min(call.tokens, kTile);
    auto alloc = [&](DType dtype, std::int32_t n0, std::int32_t n1) { return workspace.alloc(dtype, {n0, n1}); };
    Tensor ids    = alloc(DType::I32, kTopK, tile);
    Tensor gammas = alloc(DType::FP32, kTopK, tile);
    Tensor xq     = alloc(DType::FP32, kD, tile);
    Tensor xamax  = alloc(DType::FP32, tile, 1);
    Tensor gu     = alloc(DType::BF16, 2 * kI, tile * kTopK);
    Tensor y      = alloc(DType::BF16, kI, tile * kTopK);
    Tensor yq     = alloc(DType::FP32, kI, tile * kTopK);
    Tensor yamax  = alloc(DType::FP32, tile * kTopK, 1);
    Tensor down   = alloc(DType::BF16, kD, tile * kTopK);
    for (int t0 = 0; t0 < call.tokens; t0 += tile) {
        const int n = std::min(tile, call.tokens - t0);
        const bf16* x = static_cast<const bf16*>(call.x) + static_cast<std::int64_t>(t0) * kD;
        route_kernel<<<n, kE, 0, stream>>>(x, static_cast<const bf16*>(L.router), L.router_bias,
                                           static_cast<std::int32_t*>(ids.data), static_cast<float*>(gammas.data),
                                           call.capture ? call.capture + static_cast<std::int64_t>(t0) * kTopK : nullptr);
        CUDA_CHECK(cudaGetLastError());
        quant_kernel<<<n, 256, 0, stream>>>(x, kD, nvfp4, static_cast<float*>(xq.data), static_cast<float*>(xamax.data));
        CUDA_CHECK(cudaGetLastError());
        GemmArgs g13{static_cast<const float*>(xq.data), static_cast<const float*>(xamax.data), 1,
                     static_cast<const std::int32_t*>(ids.data), L.w13, nvfp4, kD, 2 * kI,
                     static_cast<bf16*>(gu.data)};
        expert_gemm_kernel<<<dim3(static_cast<unsigned>(n * kTopK), (2 * kI) / 8), 256, 0, stream>>>(g13);
        CUDA_CHECK(cudaGetLastError());
        const std::int64_t ycount = static_cast<std::int64_t>(n) * kTopK * kI;
        swiglu_kernel<<<static_cast<unsigned>((ycount + 255) / 256), 256, 0, stream>>>(
            static_cast<const bf16*>(gu.data), static_cast<const float*>(gammas.data), static_cast<bf16*>(y.data), ycount);
        CUDA_CHECK(cudaGetLastError());
        quant_kernel<<<n * kTopK, 256, 0, stream>>>(static_cast<const bf16*>(y.data), kI, nvfp4,
                                                    static_cast<float*>(yq.data), static_cast<float*>(yamax.data));
        CUDA_CHECK(cudaGetLastError());
        GemmArgs g2{static_cast<const float*>(yq.data), static_cast<const float*>(yamax.data), 0,
                    static_cast<const std::int32_t*>(ids.data), L.w2, nvfp4, kI, kD, static_cast<bf16*>(down.data)};
        expert_gemm_kernel<<<dim3(static_cast<unsigned>(n * kTopK), kD / 8), 256, 0, stream>>>(g2);
        CUDA_CHECK(cudaGetLastError());
        const std::int64_t ocount = static_cast<std::int64_t>(n) * kD;
        combine_kernel<<<static_cast<unsigned>((ocount + 255) / 256), 256, 0, stream>>>(
            static_cast<const bf16*>(down.data), static_cast<bf16*>(call.out) + static_cast<std::int64_t>(t0) * kD, ocount);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace sinfer::targets::inkling2::naive
