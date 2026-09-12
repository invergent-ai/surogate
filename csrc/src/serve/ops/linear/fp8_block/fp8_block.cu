#include "ops/linear/fp8_block/fp8_block.h"

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

// ---- activations: E4M3 per token per 128, the recipe's convention ----
__global__ void quantize_blocks_kernel(const __nv_bfloat16* __restrict__ x, int k, int tokens,
                                       std::uint8_t* __restrict__ codes,
                                       float* __restrict__ scales) {
    const int token  = static_cast<int>(blockIdx.y);
    const int block  = static_cast<int>(blockIdx.x);
    const int i      = block * kBlock + static_cast<int>(threadIdx.x);
    const float v    = __bfloat162float(x[static_cast<std::size_t>(token) * k + i]);
    float amax       = fabsf(v);
    __shared__ float partial[kBlock / 32];
    for (int o = 16; o > 0; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o)); }
    if ((threadIdx.x & 31) == 0) { partial[threadIdx.x >> 5] = amax; }
    __syncthreads();
    amax = fmaxf(fmaxf(partial[0], partial[1]), fmaxf(partial[2], partial[3]));
    const float scale   = amax > 0.0f ? amax / 448.0f : 1.0f;
    const float inverse = 1.0f / scale;
    codes[static_cast<std::size_t>(token) * k + i] =
        __nv_cvt_float_to_fp8(v * inverse, __NV_SATFINITE, __NV_E4M3);
    if (threadIdx.x == 0) { scales[static_cast<std::size_t>(token) * (k / kBlock) + block] = scale; }
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

template <int Tokens>
__global__ __launch_bounds__(128) void gemv_kernel(const std::uint8_t* __restrict__ codes,
                                                   const float* __restrict__ scales, ScaleCell cell,
                                                   const __nv_bfloat16* __restrict__ x, int rows,
                                                   int k, int tokens,
                                                   __nv_bfloat16* __restrict__ out,
                                                   bool accumulate) {
    const int warp = static_cast<int>(threadIdx.x >> 5);
    const int lane = static_cast<int>(threadIdx.x & 31);
    const int row  = static_cast<int>(blockIdx.x) * 4 + warp;
    if (row >= rows) { return; }
    const int kcells = k / cell.k_per;
    const std::uint8_t* wrow = codes + static_cast<std::size_t>(row) * k;
    const float* srow        = scales + static_cast<std::size_t>(row / cell.rows_per) * kcells;
    float acc[Tokens];
#pragma unroll
    for (int t = 0; t < Tokens; ++t) { acc[t] = 0.0f; }
    // 32 lanes x 16 codes = 512 values a step; a lane's 16 sit inside one 128-block
    for (int base = lane * 16; base < k; base += 512) {
        const uint4 packed = *reinterpret_cast<const uint4*>(wrow + base);
        const float s      = srow[base / cell.k_per];
        const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
        float part[Tokens];
#pragma unroll
        for (int t = 0; t < Tokens; ++t) { part[t] = 0.0f; }
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float2 w01 = e4m3x2_to_float2(static_cast<std::uint16_t>(words[w] & 0xffffu));
            const float2 w23 = e4m3x2_to_float2(static_cast<std::uint16_t>(words[w] >> 16));
#pragma unroll
            for (int t = 0; t < Tokens; ++t) {
                if (t < tokens) {
                    const __nv_bfloat16* xp = x + static_cast<std::size_t>(t) * k + base + 4 * w;
                    const float2 x01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(xp));
                    const float2 x23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(xp + 2));
                    part[t] = fmaf(w01.x, x01.x, part[t]);
                    part[t] = fmaf(w01.y, x01.y, part[t]);
                    part[t] = fmaf(w23.x, x23.x, part[t]);
                    part[t] = fmaf(w23.y, x23.y, part[t]);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < Tokens; ++t) { acc[t] = fmaf(s, part[t], acc[t]); }
    }
#pragma unroll
    for (int t = 0; t < Tokens; ++t) {
        float v = acc[t];
        for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, o); }
        if (lane == 0 && t < tokens) {
            __nv_bfloat16* slot = out + static_cast<std::size_t>(t) * rows + row;
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
    const std::uint8_t* __restrict__ x_codes, const float* __restrict__ x_scales, int rows, int k,
    int tokens, __nv_bfloat16* __restrict__ out) {
    __shared__ TileSmem sm;
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int kblocks    = k / kBlock;
    const int kcells     = k / cell.k_per;
    const int row_blocks = rows / kTileRows;
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
                sm.Sx[slot][t] = t < cols ? x_scales[static_cast<std::size_t>(col0 + t) * kblocks + kb] : 0.0f;
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
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int r0 = row0 + mi * 16 + gid, r1 = r0 + 8;
                const int lc0 = warp * 8 + 2 * lid, lc1 = lc0 + 1;
                auto store = [&](int col, int row, float value) {
                    __nv_bfloat16& slot = out[static_cast<std::size_t>(col) * rows + row];
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

/// One row range of one weight against every column of x, written or accumulated into out.
void run(const Tensor& x, const Weight& w, std::int32_t row_begin, std::int32_t rows, Tensor& out,
         bool accumulate, WorkspaceArena* workspace, cudaStream_t stream, const char* op,
         std::byte* prepared = nullptr) {
    require_fp8_block_weight(w, op);
    if (row_begin < 0 || rows <= 0 || row_begin + rows > w.n || (row_begin % w.scale_ne[1]) != 0 ||
        (row_begin % kTileRows) != 0 || (rows % kTileRows) != 0) {
        throw std::invalid_argument(std::string(op) + ": row range must start on a scale row and be whole 64-row tiles");
    }
    require_x_out(x, w.k, out, rows, op);
    const std::int32_t k = w.k, tokens = x.ne[1], kblocks = k / kBlock;
    const ScaleCell cell{w.scale_ne[0], w.scale_ne[1]};
    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata) + static_cast<std::size_t>(row_begin) * k;
    const auto* scales = static_cast<const float*>(w.scales) +
                         static_cast<std::size_t>(row_begin / cell.rows_per) * (k / cell.k_per);
    auto* o = static_cast<__nv_bfloat16*>(out.data);
    if (tokens <= kGemvMaxTokens) {
        const dim3 grid(static_cast<unsigned>((rows + 3) / 4));
        const auto* xin = static_cast<const __nv_bfloat16*>(x.data);
        switch (tokens) {
        case 1: gemv_kernel<1><<<grid, 128, 0, stream>>>(codes, scales, cell, xin, rows, k, tokens, o, accumulate); break;
        case 2: gemv_kernel<2><<<grid, 128, 0, stream>>>(codes, scales, cell, xin, rows, k, tokens, o, accumulate); break;
        case 3: gemv_kernel<3><<<grid, 128, 0, stream>>>(codes, scales, cell, xin, rows, k, tokens, o, accumulate); break;
        default: gemv_kernel<4><<<grid, 128, 0, stream>>>(codes, scales, cell, xin, rows, k, tokens, o, accumulate); break;
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
    if (prepared == nullptr) {
        quantize_blocks_kernel<<<dim3(static_cast<unsigned>(kblocks), static_cast<unsigned>(tokens)), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), k, tokens, x_codes, x_scales);
        CUDA_CHECK(cudaGetLastError());
    }
    const int work = (rows / kTileRows) * ((tokens + kTileCols - 1) / kTileCols);
    const int grid = std::min(work, persistent_blocks());
    if (accumulate) {
        tile_kernel<true><<<grid, kTileWarps * 32, 0, stream>>>(codes, scales, cell, x_codes, x_scales, rows, k, tokens, o);
    } else {
        tile_kernel<false><<<grid, kTileWarps * 32, 0, stream>>>(codes, scales, cell, x_codes, x_scales, rows, k, tokens, o);
    }
    CUDA_CHECK(cudaGetLastError());
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
    return codes_bytes(k, tokens) + static_cast<std::size_t>(tokens) * (k / kBlock) * sizeof(float) + kAlign;
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
    auto scope = workspace != nullptr ? std::optional(workspace->scope()) : std::nullopt;
    std::byte* prepared = nullptr;
    const int k = x.ne[0], tokens = x.ne[1];
    if (tokens > kGemvMaxTokens && !projections.empty()) {
        const auto bytes = workspace_bytes(k, tokens);
        prepared = workspace != nullptr ? static_cast<std::byte*>(workspace->alloc_bytes(bytes, kAlign).data)
                                         : static_cast<std::byte*>(ggml::scratch_for(bytes, stream));
        quantize_blocks_kernel<<<dim3(k / kBlock, tokens), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), k, tokens,
            reinterpret_cast<std::uint8_t*>(prepared),
            reinterpret_cast<float*>(prepared + codes_bytes(k, tokens)));
        CUDA_CHECK(cudaGetLastError());
    }
    for (const auto& p : projections) {
        run(x, p.weight, std::max(0, p.row_begin), p.out.ne[0], p.out, false, workspace, stream,
            "fp8 block linear projections", prepared);
    }
}

} // namespace sinfer::ops::detail::fp8_block
