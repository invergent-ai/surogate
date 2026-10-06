#include "ops/linear/fp8_block/fp8_block.h"
#include "ops/linear/fp8_block/fp8_block_sm90_gemm.h"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/ggml/ggml_dispatch.h" // scratch_for: the engine-slot scratch, graph-safe

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>

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
    float amax = fmaxf(fmaxf(fabsf(lo.x), fabsf(lo.y)), fmaxf(fabsf(hi.x), fabsf(hi.y)));
    for (int o = 16; o > 0; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o)); }
    const float scale   = amax > 0.0f ? amax / 448.0f : 1.0f;
    const float inverse = 1.0f / scale;
    const auto c01 = __nv_cvt_float2_to_fp8x2(make_float2(lo.x * inverse, lo.y * inverse), __NV_SATFINITE, __NV_E4M3);
    const auto c23 = __nv_cvt_float2_to_fp8x2(make_float2(hi.x * inverse, hi.y * inverse), __NV_SATFINITE, __NV_E4M3);
    *reinterpret_cast<std::uint32_t*>(codes + at) =
        static_cast<std::uint32_t>(c01) | static_cast<std::uint32_t>(c23) << 16;
    if (lane == 0) { scales[act_scale_index(token, block, tokens, kblocks, kb_major)] = scale; }
}

void launch_quantize_blocks(const __nv_bfloat16* x, int k, int tokens, std::uint8_t* codes,
                            float* scales, bool kb_major, cudaStream_t stream) {
    const long long items = static_cast<long long>(tokens) * (k / kBlock);
    const auto grid = static_cast<unsigned>((items + kQuantizeWarps - 1) / kQuantizeWarps);
    quantize_blocks_kernel<<<grid, kQuantizeWarps * 32, 0, stream>>>(x, k, tokens, codes, scales, kb_major);
    CUDA_CHECK(cudaGetLastError());
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

constexpr int kGemvUnroll = 4; // a lane's 16-code words in flight at once

/// VecX: the activations are 16-byte aligned and a lane reads its 16 in two vector loads.
template <int Tokens, bool VecX>
__global__ __launch_bounds__(128) void gemv_kernel(const std::uint8_t* __restrict__ codes,
                                                   const float* __restrict__ scales, ScaleCell cell,
                                                   const __nv_bfloat16* __restrict__ x,
                                                   const Segments segments, int k, int tokens,
                                                   bool accumulate) {
    const int warp = static_cast<int>(threadIdx.x >> 5);
    const int lane = static_cast<int>(threadIdx.x & 31);
    const int row  = static_cast<int>(blockIdx.x) * 4 + warp;
    if (row >= segments.rows()) { return; }
    const int kcells = k / cell.k_per;
    const std::uint8_t* wrow = codes + static_cast<std::size_t>(row) * k;
    const float* srow        = scales + static_cast<std::size_t>(row / cell.rows_per) * kcells;
    float acc[Tokens];
#pragma unroll
    for (int t = 0; t < Tokens; ++t) { acc[t] = 0.0f; }
    // One step: a lane's 16 codes at `base` against each token's 16 activations, summed in code
    // order and scaled once (a lane's 16 sit inside one 128-block).
    const auto step = [&](const uint4 packed, const float s, const int base) {
        const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
        float part[Tokens];
#pragma unroll
        for (int t = 0; t < Tokens; ++t) { part[t] = 0.0f; }
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            if (t < tokens) {
                const __nv_bfloat16* xp = x + static_cast<std::size_t>(t) * k + base;
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
                for (int w = 0; w < 4; ++w) {
                    const float2 w01 = e4m3x2_to_float2(static_cast<std::uint16_t>(words[w] & 0xffffu));
                    const float2 w23 = e4m3x2_to_float2(static_cast<std::uint16_t>(words[w] >> 16));
                    const float2 x01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xw[2 * w]));
                    const float2 x23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xw[2 * w + 1]));
                    part[t] = fmaf(w01.x, x01.x, part[t]);
                    part[t] = fmaf(w01.y, x01.y, part[t]);
                    part[t] = fmaf(w23.x, x23.x, part[t]);
                    part[t] = fmaf(w23.y, x23.y, part[t]);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < Tokens; ++t) { acc[t] = fmaf(s, part[t], acc[t]); }
    };
    // 32 lanes x 16 codes = 512 values a step. A warp per row keeps too few loads in flight to
    // cover the memory latency one word at a time, so a lane loads kGemvUnroll words, then sums.
    int base = lane * 16;
    for (; base + (kGemvUnroll - 1) * 512 < k; base += kGemvUnroll * 512) {
        uint4 packed[kGemvUnroll];
        float s[kGemvUnroll];
#pragma unroll
        for (int u = 0; u < kGemvUnroll; ++u) {
            packed[u] = load_streaming(wrow + base + u * 512);
            s[u]      = srow[(base + u * 512) / cell.k_per];
        }
#pragma unroll
        for (int u = 0; u < kGemvUnroll; ++u) { step(packed[u], s[u], base + u * 512); }
    }
    for (; base < k; base += 512) { step(load_streaming(wrow + base), srow[base / cell.k_per], base); }
    const Segment seg = segment_of(segments, row);
#pragma unroll
    for (int t = 0; t < Tokens; ++t) {
        float v = acc[t];
        for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, o); }
        if (lane == 0 && t < tokens) {
            __nv_bfloat16* slot = seg.out + static_cast<std::size_t>(t) * seg.rows + (row - seg.begin);
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

int persistent_blocks() {
    static const int blocks = [] {
        int device = 0, sms = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        return kBlocksPerSm * sms;
    }();
    return blocks;
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

/// One row range of one weight against every column of x, written or accumulated into the
/// segments' outputs. A chained launch (more than one segment) on Hopper's CUTLASS kernel needs
/// `staging`, [rows, T] BF16, for the kernel's packed output.
void run(const Tensor& x, const Weight& w, std::int32_t row_begin, const Segments& segments,
         bool accumulate, WorkspaceArena* workspace, cudaStream_t stream, const char* op,
         std::byte* prepared = nullptr, __nv_bfloat16* staging = nullptr) {
    require_fp8_block_weight(w, op);
    const std::int32_t rows = segments.rows();
    if (row_begin < 0 || rows <= 0 || row_begin + rows > w.n || (row_begin % w.scale_ne[1]) != 0 ||
        (row_begin % kTileRows) != 0 || (rows % kTileRows) != 0) {
        throw std::invalid_argument(std::string(op) + ": row range must start on a scale row and be whole 64-row tiles");
    }
    const std::int32_t k = w.k, tokens = x.ne[1], kblocks = k / kBlock;
    const ScaleCell cell{w.scale_ne[0], w.scale_ne[1]};
    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata) + static_cast<std::size_t>(row_begin) * k;
    const auto* scales = static_cast<const float*>(w.scales) +
                         static_cast<std::size_t>(row_begin / cell.rows_per) * (k / cell.k_per);
    if (tokens <= kGemvMaxTokens) {
        const dim3 grid(static_cast<unsigned>((rows + 3) / 4));
        const auto* xin = static_cast<const __nv_bfloat16*>(x.data);
        const auto launch = [&](auto kernel) {
            kernel<<<grid, 128, 0, stream>>>(codes, scales, cell, xin, segments, k, tokens, accumulate);
        };
        if ((reinterpret_cast<std::uintptr_t>(xin) & 15u) == 0) {
            switch (tokens) {
            case 1: launch(gemv_kernel<1, true>); break;
            case 2: launch(gemv_kernel<2, true>); break;
            case 3: launch(gemv_kernel<3, true>); break;
            default: launch(gemv_kernel<4, true>); break;
            }
        } else {
            switch (tokens) {
            case 1: launch(gemv_kernel<1, false>); break;
            case 2: launch(gemv_kernel<2, false>); break;
            case 3: launch(gemv_kernel<3, false>); break;
            default: launch(gemv_kernel<4, false>); break;
            }
        }
        CUDA_CHECK(cudaGetLastError());
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
        launch_quantize_blocks(static_cast<const __nv_bfloat16*>(x.data), k, tokens, x_codes, x_scales,
                               kb_major, stream);
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
         std::byte* prepared = nullptr) {
    require_fp8_block_weight(w, op);
    require_x_out(x, w.k, out, rows, op);
    Segments one;
    one.count  = 1;
    one.end[0] = rows;
    one.out[0] = static_cast<__nv_bfloat16*>(out.data);
    run(x, w, row_begin, one, accumulate, workspace, stream, op, prepared);
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
    if (k <= 0 || tokens <= kGemvMaxTokens) { return 0; }
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
    const bool gemm      = tokens > kGemvMaxTokens;
    const Weight& parent = projections.front().weight;
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
        run(x, parent, chain_begin, segments, false, workspace, stream, "fp8 block linear projections",
            prepared, staging);
        return;
    }
    for (const auto& p : projections) {
        run(x, p.weight, std::max(0, p.row_begin), p.out.ne[0], p.out, false, workspace, stream,
            "fp8 block linear projections", prepared);
    }
}

std::size_t projections_workspace_capacity_bytes(std::int32_t parent_rows, std::int32_t input_rows,
                                                 std::int32_t max_tokens) {
    const std::size_t planes = linear_workspace_capacity_bytes(parent_rows, input_rows, max_tokens);
    if (max_tokens <= kGemvMaxTokens || parent_rows <= 0) { return planes; }
    return planes + static_cast<std::size_t>(parent_rows) *
                                  std::min(max_tokens, kStagedMaxTokens) * sizeof(__nv_bfloat16);
}

} // namespace sinfer::ops::detail::fp8_block
