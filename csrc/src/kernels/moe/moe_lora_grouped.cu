// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

// Fused grouped LoRA for routed experts (replaces the per-expert cuBLAS
// micro-GEMMs of the expert LoRA path).
//
// Tokens arrive permuted into expert-contiguous order: expert e owns rows
// [expert_offsets[e], expert_offsets[e + 1]). LoRA A is [E, rank, in] and
// B is [E, out, rank], both contiguous bf16. Each CTA owns one (expert, tile
// of up to 256 tokens) and only ever touches that expert's rows and weights,
// so nothing needs atomics. The rank-r intermediates (h = x·A^T, g = d_out·B)
// are accumulated in fp32 registers, parked in shared memory as fp32, and the
// LoRA scaling is applied once, in fp32, when the product is added to the bf16
// destination. The cuBLAS path instead rounds the intermediate to bf16 twice
// (after the first GEMM and after scaling) and, for gate_up, rounds the
// product to bf16 before adding it to the output.
//
//   forward  (1 launch):  out[t, :] (+)= s · (x[t, :]·A_e^T) · B_e^T
//   backward (2 launches):
//     tile kernel:   h = x·A^T,  g = d_out·B                     (fp32, smem)
//                    dx[t, :] (+)= s · g[t, :]·A_e
//                    dA_tile = g^T·x,  dB_tile = d_out^T·h      (fp32 partials)
//     reduce kernel: dA_e (+)= s · Σ_tiles dA_tile, dB_e (+)= s · Σ_tiles dB_tile
//                    summed in tile order: bit-reproducible run to run.
//
// Thread layout (256 threads per CTA):
//   projection passes (h, g): 32 token slots × 8 lanes; lane l reads 16 bytes
//   (8 bf16) at column l*8 + 64*j of its token's row while the weight chunk
//   sits in shared memory; the 8 lanes are reduced with xor shuffles.
//   expansion passes (out, dx, dA, dB): each thread owns a bf16x2 column pair
//   and walks the tile's tokens, so a warp touches 128 contiguous bytes per row.
//
// Tiles are numbered expert-major (expert e owns ceil(T_e / tile) consecutive
// ids); a CTA finds its expert with a block scan over the offsets, so the grid
// needs no per-launch host→device upload.

#include "kernels/kernels.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "utilities/utils.h"

namespace {

constexpr int kThreads = 256;
constexpr int kLanes = 8;                      // lanes per token slot (projection passes)
constexpr int kSlots = kThreads / kLanes;      // token slots per projection sweep
constexpr int kChunk = 256;                    // weight columns staged in smem per step
constexpr int kMaxTile = 256;                  // tokens per CTA (multiple of kSlots)
constexpr int kColumnsPerPass = 2 * kThreads;  // columns per expansion step
constexpr int kScanInts = 16;                  // ints of smem used by the tile scan
constexpr int kScanBytes = kScanInts * static_cast<int>(sizeof(int));

static_assert(kChunk % (kLanes * 8) == 0, "a chunk must be a whole number of lane sweeps");
static_assert(kMaxTile % kSlots == 0, "tile must be a multiple of the slot count");

struct TileLoc {
    int expert;
    int row0;  // first permuted row of the tile
    int rows;  // tokens in the tile (1..tile_tokens)
};

__device__ __forceinline__ void unpack8(const uint4 v, float* f) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __bfloat1622float2(p[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

template <int R>
__device__ __forceinline__ void load_row(const float* __restrict__ p, float* f) {
    const float4* p4 = reinterpret_cast<const float4*>(p);
#pragma unroll
    for (int q = 0; q < R / 4; ++q) {
        const float4 v = p4[q];
        f[4 * q] = v.x;
        f[4 * q + 1] = v.y;
        f[4 * q + 2] = v.z;
        f[4 * q + 3] = v.w;
    }
}

__device__ __forceinline__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ int expert_tiles(const int* __restrict__ offsets, int e, int tile_tokens) {
    const int n = offsets[e + 1] - offsets[e];
    return n > 0 ? ceil_div(n, tile_tokens) : 0;
}

// Inclusive scan of one int per thread across the block; s_warp holds 8 ints
// and may be reused by the caller afterwards.
__device__ __forceinline__ int block_inclusive_scan(int v, int* s_warp) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int n = __shfl_up_sync(0xffffffffu, v, d);
        if (lane >= d) v += n;
    }
    if (lane == 31) s_warp[warp] = v;
    __syncthreads();
    int prefix = 0;
    for (int w = 0; w < warp; ++w) {
        prefix += s_warp[w];
    }
    __syncthreads();
    return v + prefix;
}

// Resolves flat tile id `tile` to (expert, row0, rows). Returns false when
// `tile` lies past the last tile (grids may be sized from an upper bound).
// Block-uniform; s_scan holds kScanInts ints.
__device__ bool
locate_tile(const int* __restrict__ offsets, int num_experts, int tile_tokens, int tile, int* s_scan, TileLoc& loc) {
    int carry = 0;
    for (int base = 0; base < num_experts; base += kThreads) {
        const int e = base + static_cast<int>(threadIdx.x);
        const int cnt = (e < num_experts) ? expert_tiles(offsets, e, tile_tokens) : 0;
        const int incl = block_inclusive_scan(cnt, s_scan);
        const int local = tile - carry;
        if (cnt > 0 && local >= incl - cnt && local < incl) {
            const int start = offsets[e];
            const int skip = (local - (incl - cnt)) * tile_tokens;
            s_scan[8] = e;
            s_scan[9] = start + skip;
            s_scan[10] = min(tile_tokens, offsets[e + 1] - start - skip);
        }
        if (threadIdx.x == kThreads - 1) s_scan[11] = incl;
        __syncthreads();
        const int total = s_scan[11];
        const bool found = local < total;
        if (found) {
            loc.expert = s_scan[8];
            loc.row0 = s_scan[9];
            loc.rows = s_scan[10];
        }
        __syncthreads();
        if (found) return true;
        carry += total;
    }
    return false;
}

// First tile id and tile count of `expert` under the numbering of locate_tile.
__device__ void expert_tile_range(const int* __restrict__ offsets,
                                  int num_experts,
                                  int tile_tokens,
                                  int expert,
                                  int* s_scan,
                                  int& first,
                                  int& count) {
    int carry = 0;
    for (int base = 0; base < num_experts; base += kThreads) {
        const int e = base + static_cast<int>(threadIdx.x);
        const int cnt = (e < num_experts) ? expert_tiles(offsets, e, tile_tokens) : 0;
        const int incl = block_inclusive_scan(cnt, s_scan);
        if (e == expert) {
            s_scan[8] = carry + incl - cnt;
            s_scan[9] = cnt;
        }
        if (threadIdx.x == kThreads - 1) s_scan[11] = incl;
        __syncthreads();
        carry += s_scan[11];
        __syncthreads();
    }
    first = s_scan[8];
    count = s_scan[9];
}

// dst[t][r] += Σ_k src[row0 + t][k] · W[r][k] for the tile's rows, with W
// staged through `chunk` (R × kChunk bf16) kChunk columns at a time. For
// kTransposed the weight is stored [k][r] (LoRA B) and is transposed while
// staging. src rows are k_dim wide (k_dim % 8 == 0). Block-uniform loops.
template <int R, bool kTransposed>
__device__ void project_tile(float* __restrict__ dst,
                             const nv_bfloat16* __restrict__ src,
                             const nv_bfloat16* __restrict__ w,
                             int k_dim,
                             int row0,
                             int rows,
                             nv_bfloat16* __restrict__ chunk) {
    const int slot = static_cast<int>(threadIdx.x) / kLanes;
    const int lane = static_cast<int>(threadIdx.x) % kLanes;
    for (int k0 = 0; k0 < k_dim; k0 += kChunk) {
        const int kn = min(kChunk, k_dim - k0);
        if constexpr (kTransposed) {
            for (int c = threadIdx.x; c < kn; c += kThreads) {
                const uint4* row = reinterpret_cast<const uint4*>(w + static_cast<size_t>(k0 + c) * R);
#pragma unroll
                for (int q = 0; q < R / 8; ++q) {
                    const uint4 v = row[q];
                    const nv_bfloat16* p = reinterpret_cast<const nv_bfloat16*>(&v);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        chunk[(q * 8 + i) * kChunk + c] = p[i];
                    }
                }
            }
        } else {
            const int vecs = kn / 8;
            for (int v = threadIdx.x; v < R * vecs; v += kThreads) {
                const int r = v / vecs;
                const int c = (v - r * vecs) * 8;
                *reinterpret_cast<uint4*>(chunk + r * kChunk + c) =
                    *reinterpret_cast<const uint4*>(w + static_cast<size_t>(r) * k_dim + k0 + c);
            }
        }
        __syncthreads();
        for (int sub = 0; sub < rows; sub += kSlots) {
            const int t = sub + slot;
            float acc[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                acc[r] = 0.f;
            }
            if (t < rows) {
                const nv_bfloat16* srow = src + static_cast<size_t>(row0 + t) * k_dim + k0;
                for (int j = lane * 8; j < kn; j += kLanes * 8) {
                    float xf[8];
                    unpack8(*reinterpret_cast<const uint4*>(srow + j), xf);
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        float wf[8];
                        unpack8(*reinterpret_cast<const uint4*>(chunk + r * kChunk + j), wf);
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            acc[r] = fmaf(xf[i], wf[i], acc[r]);
                        }
                    }
                }
            }
            // Butterfly over the 8 lanes: every lane ends with the same bits.
#pragma unroll
            for (int r = 0; r < R; ++r) {
                acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], 1);
                acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], 2);
                acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], 4);
            }
            if (t < rows) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    if ((r % kLanes) == lane) dst[t * R + r] += acc[r];
                }
            }
        }
        __syncthreads();
    }
}

// Shared memory: [scan ints][weight chunk][h][g (backward only)].
__host__ __device__ constexpr size_t chunk_bytes(int rank) {
    return static_cast<size_t>(rank) * kChunk * sizeof(nv_bfloat16);
}
__host__ __device__ constexpr size_t tile_bytes(int rank, int tile_tokens) {
    return static_cast<size_t>(rank) * tile_tokens * sizeof(float);
}
__host__ __device__ constexpr size_t smem_bytes(int rank, int tile_tokens, bool with_g) {
    return kScanBytes + chunk_bytes(rank) + tile_bytes(rank, tile_tokens) * (with_g ? 2 : 1);
}

template <int R>
__global__ void __launch_bounds__(kThreads) moe_lora_grouped_forward_kernel(nv_bfloat16* __restrict__ out,
                                                                            const nv_bfloat16* __restrict__ x,
                                                                            const nv_bfloat16* __restrict__ A,
                                                                            const nv_bfloat16* __restrict__ B,
                                                                            const int* __restrict__ expert_offsets,
                                                                            int num_experts,
                                                                            int in_features,
                                                                            int out_features,
                                                                            int tile_tokens,
                                                                            float scaling,
                                                                            int accumulate) {
    extern __shared__ __align__(16) unsigned char smem[];
    int* s_scan = reinterpret_cast<int*>(smem);
    nv_bfloat16* chunk = reinterpret_cast<nv_bfloat16*>(smem + kScanBytes);
    float* h = reinterpret_cast<float*>(smem + kScanBytes + chunk_bytes(R));

    TileLoc loc;
    if (!locate_tile(expert_offsets, num_experts, tile_tokens, static_cast<int>(blockIdx.x), s_scan, loc)) return;
    const nv_bfloat16* A_e = A + static_cast<size_t>(loc.expert) * R * in_features;
    const nv_bfloat16* B_e = B + static_cast<size_t>(loc.expert) * out_features * R;

    for (int i = threadIdx.x; i < loc.rows * R; i += kThreads) {
        h[i] = 0.f;
    }
    project_tile<R, false>(h, x, A_e, in_features, loc.row0, loc.rows, chunk);

    // out[t, o:o+2] (+)= s · h[t, :] · B_e[o:o+2, :]^T
    for (int c0 = 0; c0 < out_features; c0 += kColumnsPerPass) {
        const int o = c0 + 2 * static_cast<int>(threadIdx.x);
        if (o >= out_features) break;
        float b0[R], b1[R];
        const uint4* bp = reinterpret_cast<const uint4*>(B_e + static_cast<size_t>(o) * R);
#pragma unroll
        for (int q = 0; q < R / 8; ++q) {
            unpack8(bp[q], b0 + 8 * q);
            unpack8(bp[R / 8 + q], b1 + 8 * q);
        }
        nv_bfloat16* col = out + static_cast<size_t>(loc.row0) * out_features + o;
#pragma unroll 4
        for (int t = 0; t < loc.rows; ++t) {
            float hr[R];
            load_row<R>(h + t * R, hr);
            float v0 = 0.f, v1 = 0.f;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                v0 = fmaf(hr[r], b0[r], v0);
                v1 = fmaf(hr[r], b1[r], v1);
            }
            __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(col + static_cast<size_t>(t) * out_features);
            float2 cur = accumulate ? __bfloat1622float2(*p) : make_float2(0.f, 0.f);
            cur.x = fmaf(scaling, v0, cur.x);
            cur.y = fmaf(scaling, v1, cur.y);
            *p = __float22bfloat162_rn(cur);
        }
    }
}

// Per-tile partial layout: [R × in] (dA) followed by [out × R] (dB), fp32.
template <int R, bool kWgrad>
__global__ void __launch_bounds__(kThreads) moe_lora_grouped_backward_kernel(nv_bfloat16* __restrict__ dx,
                                                                             float* __restrict__ partials,
                                                                             const nv_bfloat16* __restrict__ d_out,
                                                                             const nv_bfloat16* __restrict__ x,
                                                                             const nv_bfloat16* __restrict__ A,
                                                                             const nv_bfloat16* __restrict__ B,
                                                                             const int* __restrict__ expert_offsets,
                                                                             int num_experts,
                                                                             int in_features,
                                                                             int out_features,
                                                                             int tile_tokens,
                                                                             float scaling,
                                                                             int dx_accumulate) {
    extern __shared__ __align__(16) unsigned char smem[];
    int* s_scan = reinterpret_cast<int*>(smem);
    nv_bfloat16* chunk = reinterpret_cast<nv_bfloat16*>(smem + kScanBytes);
    float* h = reinterpret_cast<float*>(smem + kScanBytes + chunk_bytes(R));
    float* g = h + tile_tokens * R;

    TileLoc loc;
    if (!locate_tile(expert_offsets, num_experts, tile_tokens, static_cast<int>(blockIdx.x), s_scan, loc)) return;
    const nv_bfloat16* A_e = A + static_cast<size_t>(loc.expert) * R * in_features;
    const nv_bfloat16* B_e = B + static_cast<size_t>(loc.expert) * out_features * R;

    for (int i = threadIdx.x; i < loc.rows * R; i += kThreads) {
        h[i] = 0.f;
        g[i] = 0.f;
    }
    if constexpr (kWgrad) {
        project_tile<R, false>(h, x, A_e, in_features, loc.row0, loc.rows, chunk);
    }
    project_tile<R, true>(g, d_out, B_e, out_features, loc.row0, loc.rows, chunk);

    const size_t slab = static_cast<size_t>(R) * (in_features + out_features);
    float* pa = partials + static_cast<size_t>(blockIdx.x) * slab;
    float* pb = pa + static_cast<size_t>(R) * in_features;

    // dx[t, i:i+2] (+)= s · g[t, :] · A_e[:, i:i+2];  dA_tile[:, i:i+2] = g^T · x[:, i:i+2]
    for (int c0 = 0; c0 < in_features; c0 += kColumnsPerPass) {
        const int i = c0 + 2 * static_cast<int>(threadIdx.x);
        if (i >= in_features) break;
        float a0[R], a1[R], da0[R], da1[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float2 av = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(A_e + static_cast<size_t>(r) * in_features + i));
            a0[r] = av.x;
            a1[r] = av.y;
            da0[r] = 0.f;
            da1[r] = 0.f;
        }
        const nv_bfloat16* xcol = x + static_cast<size_t>(loc.row0) * in_features + i;
        nv_bfloat16* dxcol = dx + static_cast<size_t>(loc.row0) * in_features + i;
#pragma unroll 4
        for (int t = 0; t < loc.rows; ++t) {
            float gr[R];
            load_row<R>(g + t * R, gr);
            float v0 = 0.f, v1 = 0.f;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                v0 = fmaf(gr[r], a0[r], v0);
                v1 = fmaf(gr[r], a1[r], v1);
            }
            __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(dxcol + static_cast<size_t>(t) * in_features);
            float2 cur = dx_accumulate ? __bfloat1622float2(*p) : make_float2(0.f, 0.f);
            cur.x = fmaf(scaling, v0, cur.x);
            cur.y = fmaf(scaling, v1, cur.y);
            *p = __float22bfloat162_rn(cur);
            if constexpr (kWgrad) {
                const float2 xv = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(xcol + static_cast<size_t>(t) * in_features));
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    da0[r] = fmaf(gr[r], xv.x, da0[r]);
                    da1[r] = fmaf(gr[r], xv.y, da1[r]);
                }
            }
        }
        if constexpr (kWgrad) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                *reinterpret_cast<float2*>(pa + static_cast<size_t>(r) * in_features + i) = make_float2(da0[r], da1[r]);
            }
        }
    }

    // dB_tile[o:o+2, :] = d_out[:, o:o+2]^T · h
    if constexpr (kWgrad) {
        for (int c0 = 0; c0 < out_features; c0 += kColumnsPerPass) {
            const int o = c0 + 2 * static_cast<int>(threadIdx.x);
            if (o >= out_features) break;
            float db0[R], db1[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                db0[r] = 0.f;
                db1[r] = 0.f;
            }
            const nv_bfloat16* dcol = d_out + static_cast<size_t>(loc.row0) * out_features + o;
#pragma unroll 4
            for (int t = 0; t < loc.rows; ++t) {
                float hr[R];
                load_row<R>(h + t * R, hr);
                const float2 dv = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(dcol + static_cast<size_t>(t) * out_features));
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    db0[r] = fmaf(dv.x, hr[r], db0[r]);
                    db1[r] = fmaf(dv.y, hr[r], db1[r]);
                }
            }
            float4* dst = reinterpret_cast<float4*>(pb + static_cast<size_t>(o) * R);
#pragma unroll
            for (int q = 0; q < R / 4; ++q) {
                dst[q] = make_float4(db0[4 * q], db0[4 * q + 1], db0[4 * q + 2], db0[4 * q + 3]);
                dst[R / 4 + q] = make_float4(db1[4 * q], db1[4 * q + 1], db1[4 * q + 2], db1[4 * q + 3]);
            }
        }
    }
}

// dW_e (+)= s · Σ_{tiles of e, in order} partial_tile. Experts without tokens
// are left untouched, matching the cuBLAS grouped path which skips them.
__global__ void __launch_bounds__(kThreads) moe_lora_grouped_reduce_kernel(nv_bfloat16* __restrict__ dA,
                                                                           nv_bfloat16* __restrict__ dB,
                                                                           const float* __restrict__ partials,
                                                                           const int* __restrict__ expert_offsets,
                                                                           int num_experts,
                                                                           int a_elems,
                                                                           int b_elems,
                                                                           int tile_tokens,
                                                                           float scaling,
                                                                           int accumulate) {
    __shared__ int s_scan[kScanInts];
    const int e = static_cast<int>(blockIdx.y);
    int first = 0, count = 0;
    expert_tile_range(expert_offsets, num_experts, tile_tokens, e, s_scan, first, count);
    if (count == 0) return;
    const int slab = a_elems + b_elems;
    const int j = static_cast<int>(blockIdx.x * kThreads + threadIdx.x) * 4;
    if (j >= slab) return;

    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    const float* p = partials + static_cast<size_t>(first) * slab + j;
    for (int t = 0; t < count; ++t, p += slab) {
        const float4 v = *reinterpret_cast<const float4*>(p);
        sum.x += v.x;
        sum.y += v.y;
        sum.z += v.z;
        sum.w += v.w;
    }
    nv_bfloat16* dst = (j < a_elems) ? dA + static_cast<size_t>(e) * a_elems + j
                                     : dB + static_cast<size_t>(e) * b_elems + (j - a_elems);
    __nv_bfloat162* d2 = reinterpret_cast<__nv_bfloat162*>(dst);
    float2 c0 = accumulate ? __bfloat1622float2(d2[0]) : make_float2(0.f, 0.f);
    float2 c1 = accumulate ? __bfloat1622float2(d2[1]) : make_float2(0.f, 0.f);
    c0.x = fmaf(scaling, sum.x, c0.x);
    c0.y = fmaf(scaling, sum.y, c0.y);
    c1.x = fmaf(scaling, sum.z, c1.x);
    c1.y = fmaf(scaling, sum.w, c1.y);
    d2[0] = __float22bfloat162_rn(c0);
    d2[1] = __float22bfloat162_rn(c1);
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

int sm_count_for_current_device() {
    static std::atomic<int> cache[64];
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    const bool cacheable = dev >= 0 && dev < 64;
    if (cacheable && cache[dev].load(std::memory_order_relaxed) > 0) return cache[dev].load(std::memory_order_relaxed);
    int n = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
    n = std::max(n, 1);
    if (cacheable) cache[dev].store(n, std::memory_order_relaxed);
    return n;
}

struct Tiling {
    int tile_tokens = 0;
    long total_tiles = 0;  // exact with host offsets, else an upper bound
};

long count_tiles(const int* host_offsets, int num_experts, int tile_tokens) {
    long n = 0;
    for (int e = 0; e < num_experts; ++e) {
        const int c = host_offsets[e + 1] - host_offsets[e];
        if (c > 0) n += (c + tile_tokens - 1) / tile_tokens;
    }
    return n;
}

// Largest tile (fewest partials, best weight reuse) that still yields about
// two CTAs per SM. Without the host copy of the offsets the tile is fixed and
// the grid is an upper bound whose surplus CTAs exit in locate_tile.
Tiling choose_tiling(const int* host_offsets, int num_experts, int total_tokens, int tile_override) {
    Tiling t;
    if (tile_override > 0) {
        t.tile_tokens = std::min(kMaxTile, std::max(kSlots, tile_override / kSlots * kSlots));
    } else if (!host_offsets) {
        t.tile_tokens = 64;
    } else {
        const long target = 2L * sm_count_for_current_device();
        t.tile_tokens = kSlots;
        for (int tile = kMaxTile; tile > kSlots; tile /= 2) {
            if (count_tiles(host_offsets, num_experts, tile) >= target) {
                t.tile_tokens = tile;
                break;
            }
        }
    }
    t.total_tiles = host_offsets ? count_tiles(host_offsets, num_experts, t.tile_tokens)
                                 : static_cast<long>(total_tokens) / t.tile_tokens + num_experts;
    return t;
}

bool aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

bool rank_supported(int rank) {
    return rank == 8 || rank == 16 || rank == 32;
}

bool shape_supported(int num_experts, int in_features, int out_features, int rank) {
    return num_experts > 0 && in_features > 0 && (in_features % 8) == 0 && out_features > 0 &&
           (out_features % 8) == 0 && rank_supported(rank);
}

template <typename Kernel>
void enable_large_smem(Kernel kernel, size_t bytes) {
    if (bytes > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
    }
}

template <int R>
void launch_forward(nv_bfloat16* out,
                    const nv_bfloat16* x,
                    const nv_bfloat16* A,
                    const nv_bfloat16* B,
                    const int* expert_offsets,
                    int num_experts,
                    int in_features,
                    int out_features,
                    const Tiling& tiling,
                    float scaling,
                    bool accumulate,
                    cudaStream_t stream) {
    const size_t smem = smem_bytes(R, tiling.tile_tokens, false);
    enable_large_smem(moe_lora_grouped_forward_kernel<R>, smem);
    moe_lora_grouped_forward_kernel<R>
        <<<static_cast<unsigned>(tiling.total_tiles), kThreads, smem, stream>>>(out,
                                                                                x,
                                                                                A,
                                                                                B,
                                                                                expert_offsets,
                                                                                num_experts,
                                                                                in_features,
                                                                                out_features,
                                                                                tiling.tile_tokens,
                                                                                scaling,
                                                                                accumulate ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());
}

template <int R, bool kWgrad>
void launch_backward(nv_bfloat16* dx,
                     nv_bfloat16* dA,
                     nv_bfloat16* dB,
                     const nv_bfloat16* d_out,
                     const nv_bfloat16* x,
                     const nv_bfloat16* A,
                     const nv_bfloat16* B,
                     const int* expert_offsets,
                     int num_experts,
                     int in_features,
                     int out_features,
                     const Tiling& tiling,
                     float scaling,
                     bool dx_accumulate,
                     bool grad_accumulate,
                     float* workspace,
                     cudaStream_t stream) {
    const size_t smem = smem_bytes(R, tiling.tile_tokens, true);
    enable_large_smem(moe_lora_grouped_backward_kernel<R, kWgrad>, smem);
    moe_lora_grouped_backward_kernel<R, kWgrad>
        <<<static_cast<unsigned>(tiling.total_tiles), kThreads, smem, stream>>>(dx,
                                                                                workspace,
                                                                                d_out,
                                                                                x,
                                                                                A,
                                                                                B,
                                                                                expert_offsets,
                                                                                num_experts,
                                                                                in_features,
                                                                                out_features,
                                                                                tiling.tile_tokens,
                                                                                scaling,
                                                                                dx_accumulate ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());
    if constexpr (kWgrad) {
        const int a_elems = R * in_features;
        const int b_elems = out_features * R;
        const int slab = a_elems + b_elems;
        const dim3 grid((slab / 4 + kThreads - 1) / kThreads, static_cast<unsigned>(num_experts));
        moe_lora_grouped_reduce_kernel<<<grid, kThreads, 0, stream>>>(dA,
                                                                      dB,
                                                                      workspace,
                                                                      expert_offsets,
                                                                      num_experts,
                                                                      a_elems,
                                                                      b_elems,
                                                                      tiling.tile_tokens,
                                                                      scaling,
                                                                      grad_accumulate ? 1 : 0);
        CUDA_CHECK(cudaGetLastError());
    }
}

}  // namespace

bool moe_lora_grouped_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("SUROGATE_FUSED_EXPERT_LORA");
        return !(env && std::strcmp(env, "0") == 0);
    }();
    return enabled;
}

bool moe_lora_grouped_forward_bf16(nv_bfloat16* out,
                                   const nv_bfloat16* x,
                                   const nv_bfloat16* A,
                                   const nv_bfloat16* B,
                                   const int* expert_offsets,
                                   const int* host_offsets,
                                   int num_experts,
                                   int total_tokens,
                                   int in_features,
                                   int out_features,
                                   int rank,
                                   float scaling,
                                   bool accumulate,
                                   cudaStream_t stream,
                                   int tile_tokens) {
    if (!shape_supported(num_experts, in_features, out_features, rank)) return false;
    if (!out || !x || !A || !B || !expert_offsets) return false;
    if (!aligned16(out) || !aligned16(x) || !aligned16(A) || !aligned16(B)) return false;
    if (total_tokens <= 0) return true;
    const Tiling tiling = choose_tiling(host_offsets, num_experts, total_tokens, tile_tokens);
    if (tiling.total_tiles <= 0) return true;
    switch (rank) {
        case 8:
            launch_forward<8>(out,
                              x,
                              A,
                              B,
                              expert_offsets,
                              num_experts,
                              in_features,
                              out_features,
                              tiling,
                              scaling,
                              accumulate,
                              stream);
            return true;
        case 16:
            launch_forward<16>(out,
                               x,
                               A,
                               B,
                               expert_offsets,
                               num_experts,
                               in_features,
                               out_features,
                               tiling,
                               scaling,
                               accumulate,
                               stream);
            return true;
        case 32:
            launch_forward<32>(out,
                               x,
                               A,
                               B,
                               expert_offsets,
                               num_experts,
                               in_features,
                               out_features,
                               tiling,
                               scaling,
                               accumulate,
                               stream);
            return true;
        default: return false;
    }
}

std::size_t moe_lora_grouped_backward_workspace_floats(const int* host_offsets,
                                                       int num_experts,
                                                       int total_tokens,
                                                       int in_features,
                                                       int out_features,
                                                       int rank,
                                                       int tile_tokens) {
    if (!shape_supported(num_experts, in_features, out_features, rank) || total_tokens <= 0) return 0;
    const Tiling tiling = choose_tiling(host_offsets, num_experts, total_tokens, tile_tokens);
    return static_cast<std::size_t>(tiling.total_tiles) * static_cast<std::size_t>(rank) *
           (static_cast<std::size_t>(in_features) + static_cast<std::size_t>(out_features));
}

bool moe_lora_grouped_backward_bf16(nv_bfloat16* dx,
                                    nv_bfloat16* dA,
                                    nv_bfloat16* dB,
                                    const nv_bfloat16* d_out,
                                    const nv_bfloat16* x,
                                    const nv_bfloat16* A,
                                    const nv_bfloat16* B,
                                    const int* expert_offsets,
                                    const int* host_offsets,
                                    int num_experts,
                                    int total_tokens,
                                    int in_features,
                                    int out_features,
                                    int rank,
                                    float scaling,
                                    bool dx_accumulate,
                                    bool grad_accumulate,
                                    float* workspace,
                                    std::size_t workspace_floats,
                                    cudaStream_t stream,
                                    int tile_tokens) {
    if (!shape_supported(num_experts, in_features, out_features, rank)) return false;
    if (!dx || !d_out || !x || !A || !B || !expert_offsets) return false;
    if (!aligned16(dx) || !aligned16(d_out) || !aligned16(x) || !aligned16(A) || !aligned16(B)) return false;
    const bool want_wgrad = (dA != nullptr) || (dB != nullptr);
    if (want_wgrad && (!dA || !dB || !aligned16(dA) || !aligned16(dB))) return false;
    if (total_tokens <= 0) return true;
    const Tiling tiling = choose_tiling(host_offsets, num_experts, total_tokens, tile_tokens);
    if (tiling.total_tiles <= 0) return true;
    if (want_wgrad) {
        const std::size_t need = static_cast<std::size_t>(tiling.total_tiles) * static_cast<std::size_t>(rank) *
                                 (static_cast<std::size_t>(in_features) + static_cast<std::size_t>(out_features));
        if (!workspace || !aligned16(workspace) || workspace_floats < need) return false;
    }
#define SUROGATE_MOE_LORA_LAUNCH_BWD(R, W) \
    launch_backward<R, W>(dx,              \
                          dA,              \
                          dB,              \
                          d_out,           \
                          x,               \
                          A,               \
                          B,               \
                          expert_offsets,  \
                          num_experts,     \
                          in_features,     \
                          out_features,    \
                          tiling,          \
                          scaling,         \
                          dx_accumulate,   \
                          grad_accumulate, \
                          workspace,       \
                          stream)
    switch (rank) {
        case 8:
            if (want_wgrad) {
                SUROGATE_MOE_LORA_LAUNCH_BWD(8, true);
            } else {
                SUROGATE_MOE_LORA_LAUNCH_BWD(8, false);
            }
            return true;
        case 16:
            if (want_wgrad) {
                SUROGATE_MOE_LORA_LAUNCH_BWD(16, true);
            } else {
                SUROGATE_MOE_LORA_LAUNCH_BWD(16, false);
            }
            return true;
        case 32:
            if (want_wgrad) {
                SUROGATE_MOE_LORA_LAUNCH_BWD(32, true);
            } else {
                SUROGATE_MOE_LORA_LAUNCH_BWD(32, false);
            }
            return true;
        default: return false;
    }
#undef SUROGATE_MOE_LORA_LAUNCH_BWD
}
