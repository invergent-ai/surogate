// The weight-only NVFP4 mixture on Marlin (see sparse_moe_nvfp4.h).

#include "ops/sparse_moe/marlin/sparse_moe_nvfp4.h"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/marlin/marlin_gemm.h"
#include "ops/linear/marlin/marlin_nvfp4.h"
#include "ops/sparse_moe/marlin/marlin_moe_gemm.h"
#include "ops/sparse_moe/marlin/marlin_moe_lora.h"
#include "ops/sparse_moe/sparse_moe_route.cuh"
#include "api/ops/linear.h"
#include "api/ops/linear_swiglu.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sinfer::ops::detail {
namespace {

constexpr int kExperts = 256;
constexpr int kTopK    = 8;
// Router widths at or below this run one warp per router row over every token; wider rounds
// take the tiled kernel. Both accumulate in FP32 over BF16 operands.
constexpr int kNarrowRouterTokens = 32;
constexpr std::size_t kAlign      = 256;

std::size_t aligned(std::size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

// vLLM's fused_marlin_moe block size: the smallest block a typical expert's rows fill to 90 %.
int moe_block_size(std::int32_t tokens, std::int32_t experts) {
    for (const int block : {8, 16, 32, 48}) {
        if (static_cast<double>(tokens) * kTopK / experts / block < 0.9) { return block; }
    }
    return 64;
}

std::int32_t padded_capacity(std::int32_t assignments, std::int32_t experts) {
    return assignments + experts * 63;
}

std::int32_t block_capacity(std::int32_t assignments, std::int32_t experts) {
    return assignments / 8 + experts + 1;
}

struct SliceWorkspace {
    float* logits             = nullptr; // [tokens][router_rows]
    std::int32_t* ids         = nullptr; // [tokens][top_k]
    float* alpha              = nullptr; // [tokens][top_k]
    float* shared_scale       = nullptr; // [tokens]
    std::int32_t* offsets     = nullptr; // [experts + 1]
    std::int32_t* sorted_ids  = nullptr; // padded rows
    std::int32_t* expert_ids  = nullptr; // per block
    std::int32_t* past_padded = nullptr; // [1]
    __nv_bfloat16* product    = nullptr; // [tokens * top_k][max(2 * intermediate, hidden)]
    __nv_bfloat16* activation = nullptr; // [tokens * top_k][intermediate]
    __nv_bfloat16* shared_up  = nullptr; // [tokens][2 * shared]
    __nv_bfloat16* shared_act = nullptr; // [tokens][shared]
    __nv_bfloat16* shared_out = nullptr; // [tokens][hidden]
    __nv_bfloat16* lora_low   = nullptr; // [tokens * top_k][3][max rank], adapter rounds
    __nv_bfloat16* lora_shared_low = nullptr; // [tokens][3][max rank]
};

// One layout for sizing and for carving, so the two cannot drift.
template <class Allocate>
SliceWorkspace layout(const SparseMoeGeometry& g, std::int32_t tokens, Allocate&& allocate) {
    const std::int32_t assignments = tokens * kTopK;
    SliceWorkspace w;
    w.logits       = static_cast<float*>(allocate(sizeof(float) * tokens * g.router_rows()));
    w.ids          = static_cast<std::int32_t*>(allocate(sizeof(std::int32_t) * assignments));
    w.alpha        = static_cast<float*>(allocate(sizeof(float) * assignments));
    w.shared_scale = static_cast<float*>(allocate(sizeof(float) * tokens));
    w.offsets      = static_cast<std::int32_t*>(allocate(sizeof(std::int32_t) * (g.experts + 1)));
    w.sorted_ids   = static_cast<std::int32_t*>(
        allocate(sizeof(std::int32_t) * padded_capacity(assignments, g.experts)));
    w.expert_ids   = static_cast<std::int32_t*>(
        allocate(sizeof(std::int32_t) * block_capacity(assignments, g.experts)));
    w.past_padded  = static_cast<std::int32_t*>(allocate(sizeof(std::int32_t)));
    w.product      = static_cast<__nv_bfloat16*>(allocate(
        sizeof(__nv_bfloat16) * assignments * std::max(2 * g.intermediate, g.hidden)));
    w.activation   = static_cast<__nv_bfloat16*>(
        allocate(sizeof(__nv_bfloat16) * assignments * g.intermediate));
    if (g.has_shared()) {
        w.shared_up  = static_cast<__nv_bfloat16*>(
            allocate(sizeof(__nv_bfloat16) * tokens * 2 * g.shared_intermediate));
        w.shared_act = static_cast<__nv_bfloat16*>(
            allocate(sizeof(__nv_bfloat16) * tokens * g.shared_intermediate));
        w.shared_out = static_cast<__nv_bfloat16*>(
            allocate(sizeof(__nv_bfloat16) * tokens * g.hidden));
        w.lora_shared_low = static_cast<__nv_bfloat16*>(
            allocate(sizeof(__nv_bfloat16) * tokens * 3 * kSparseMoeNvfp4MaxLoraRank));
    }
    w.lora_low = static_cast<__nv_bfloat16*>(
        allocate(sizeof(__nv_bfloat16) * assignments * 3 * kSparseMoeNvfp4MaxLoraRank));
    return w;
}

// -- router --------------------------------------------------------------------------------

// One warp per router row, every token of a narrow round: the row is read once.
__global__ void router_narrow_kernel(const __nv_bfloat16* __restrict__ router, int rows,
                                     const __nv_bfloat16* __restrict__ x, int tokens, int hidden,
                                     float* __restrict__ logits) {
    const int row  = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / 32);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (row >= rows) { return; }
    float accumulator[kNarrowRouterTokens];
#pragma unroll
    for (int t = 0; t < kNarrowRouterTokens; ++t) { accumulator[t] = 0.0F; }
    const __nv_bfloat16* weight = router + static_cast<std::size_t>(row) * hidden;
    for (int k = lane * 8; k < hidden; k += 32 * 8) {
        const uint4 packed = *reinterpret_cast<const uint4*>(weight + k);
        const auto* w2     = reinterpret_cast<const __nv_bfloat162*>(&packed);
        float wf[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 v = __bfloat1622float2(w2[i]);
            wf[2 * i]      = v.x;
            wf[2 * i + 1]  = v.y;
        }
#pragma unroll
        for (int t = 0; t < kNarrowRouterTokens; ++t) {
            if (t < tokens) {
                const uint4 xp = *reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(t) * hidden + k);
                const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(&xp);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 v = __bfloat1622float2(x2[i]);
                    accumulator[t] = fmaf(wf[2 * i], v.x, accumulator[t]);
                    accumulator[t] = fmaf(wf[2 * i + 1], v.y, accumulator[t]);
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < kNarrowRouterTokens; ++t) {
        if (t < tokens) {
            const float sum = warp_reduce_sum(accumulator[t]);
            if (lane == 0) { logits[static_cast<std::size_t>(t) * rows + row] = sum; }
        }
    }
}

// A 32-row x 32-token tile per block, K in chunks of 32 through shared memory.
constexpr int kTile = 32;

// The router of a wide round on tensor cores (the prefill family's router MMA, over this route's
// runtime router rows and hidden width): FP32 logits [tokens][rows] from BF16 operands. A 64-token,
// 16-row tile per block, K staged 64 at a time through two cp.async stages.
constexpr int kRouterBM      = 16;
constexpr int kRouterBN      = 64;
constexpr int kRouterBK      = 64;
constexpr int kRouterStages  = 2;
constexpr int kRouterWarps   = 8;
constexpr int kRouterThreads = 32 * kRouterWarps;

__global__ __launch_bounds__(kRouterThreads, 2) void router_mma_kernel(
    const __nv_bfloat16* __restrict__ weight, int rows, const __nv_bfloat16* __restrict__ x, int tokens,
    int hidden, float* __restrict__ logits) {
    __shared__ __align__(16) __nv_bfloat16 Ws[kRouterStages][kRouterBM * kRouterBK];
    __shared__ __align__(16) __nv_bfloat16 Xs[kRouterStages][kRouterBN * kRouterBK];
    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int row0   = static_cast<int>(blockIdx.x) * kRouterBM;
    const int token0 = static_cast<int>(blockIdx.y) * kRouterBN;
    float acc[4] = {};
    const auto stage_inputs = [&](int stage, int kt) {
        const int k0 = kt * kRouterBK;
        for (int item = tid; item < kRouterBM * (kRouterBK / 8); item += kRouterThreads) {
            const int row = item / (kRouterBK / 8);
            const int k8  = item - row * (kRouterBK / 8);
            const int rr  = row0 + row;
            auto* dst     = &Ws[stage][row * kRouterBK + gemm_swz64(row, k8 * 8)];
            const auto* src = weight + static_cast<std::int64_t>(rr < rows ? rr : 0) * hidden + k0 + k8 * 8;
            cp_async_zfill<16, Cache::cg>(dst, src, rr < rows ? 16 : 0);
        }
        for (int item = tid; item < kRouterBN * (kRouterBK / 8); item += kRouterThreads) {
            const int col = item / (kRouterBK / 8);
            const int k8  = item - col * (kRouterBK / 8);
            const int tt  = token0 + col;
            auto* dst     = &Xs[stage][col * kRouterBK + gemm_swz64(col, k8 * 8)];
            const auto* src = x + static_cast<std::int64_t>(tt < tokens ? tt : 0) * hidden + k0 + k8 * 8;
            cp_async_zfill<16, Cache::cg>(dst, src, tt < tokens ? 16 : 0);
        }
    };
    const int k_tiles = hidden / kRouterBK;
#pragma unroll
    for (int stage = 0; stage < kRouterStages; ++stage) {
        if (stage < k_tiles) { stage_inputs(stage, stage); }
        cp_commit();
    }
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;
#pragma unroll 1
    for (int kt = 0; kt < k_tiles; ++kt) {
        const int stage = kt & 1;
        cp_wait<kRouterStages - 1>();
        __syncthreads();
#pragma unroll
        for (int ki = 0; ki < kRouterBK / 16; ++ki) {
            unsigned af[4];
            unsigned bf[2];
            const int acol = ki * 16 + a_coloff;
            const int brow = warp * 8 + b_rin;
            const int bcol = ki * 16 + b_koff;
            ldmatrix_x4(af[0], af[1], af[2], af[3],
                        smem_addr(&Ws[stage][a_rowoff * kRouterBK + gemm_swz64(a_rowoff, acol)]));
            ldmatrix_x2(bf[0], bf[1], smem_addr(&Xs[stage][brow * kRouterBK + gemm_swz64(brow, bcol)]));
            mma_bf16(acc[0], acc[1], acc[2], acc[3], af[0], af[1], af[2], af[3], bf[0], bf[1]);
        }
        __syncthreads();
        const int next = kt + kRouterStages;
        if (next < k_tiles) { stage_inputs(stage, next); }
        cp_commit();
    }
    cp_wait<0>();
    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int r0  = row0 + gid;
    const int r1  = r0 + 8;
    const int c0  = token0 + warp * 8 + 2 * lid;
    const int c1  = c0 + 1;
    if (r0 < rows && c0 < tokens) { logits[static_cast<std::int64_t>(c0) * rows + r0] = acc[0]; }
    if (r0 < rows && c1 < tokens) { logits[static_cast<std::int64_t>(c1) * rows + r0] = acc[1]; }
    if (r1 < rows && c0 < tokens) { logits[static_cast<std::int64_t>(c0) * rows + r1] = acc[2]; }
    if (r1 < rows && c1 < tokens) { logits[static_cast<std::int64_t>(c1) * rows + r1] = acc[3]; }
}

__global__ void router_tiled_kernel(const __nv_bfloat16* __restrict__ router, int rows,
                                    const __nv_bfloat16* __restrict__ x, int tokens, int hidden,
                                    float* __restrict__ logits) {
    __shared__ float weight_tile[kTile][kTile + 1];
    __shared__ float input_tile[kTile][kTile + 1];
    const int tx       = static_cast<int>(threadIdx.x) % kTile; // token within the tile
    const int ty       = static_cast<int>(threadIdx.x) / kTile; // 0..7
    const int row_base = static_cast<int>(blockIdx.x) * kTile;
    const int token0   = static_cast<int>(blockIdx.y) * kTile;
    float accumulator[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    for (int k0 = 0; k0 < hidden; k0 += kTile) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = ty + 8 * i;
            const int row = row_base + r;
            weight_tile[r][tx] = row < rows
                ? __bfloat162float(router[static_cast<std::size_t>(row) * hidden + k0 + tx]) : 0.0F;
            const int token = token0 + r;
            input_tile[r][tx] = token < tokens
                ? __bfloat162float(x[static_cast<std::size_t>(token) * hidden + k0 + tx]) : 0.0F;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kTile; ++kk) {
            const float input = input_tile[tx][kk];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                accumulator[i] = fmaf(weight_tile[ty + 8 * i][kk], input, accumulator[i]);
            }
        }
        __syncthreads();
    }
    const int token = token0 + tx;
    if (token >= tokens) { return; }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = row_base + ty + 8 * i;
        if (row < rows) { logits[static_cast<std::size_t>(token) * rows + row] = accumulator[i]; }
    }
}

// One warp per token: the softmax top-k of the routed logits and the shared expert's sigmoid
// gate, exactly as every other route of this op selects (ties to the lower id).
template <bool HasShared>
__global__ void select_kernel(const float* __restrict__ logits, int rows, int tokens,
                              std::int32_t* __restrict__ ids, float* __restrict__ alpha,
                              float* __restrict__ shared_scale) {
    __shared__ float selected[8][kTopK];
    const int warp  = static_cast<int>(threadIdx.x) / 32;
    const int token = static_cast<int>(blockIdx.x) * 8 + warp;
    if (token >= tokens) { return; }
    float unused_shared = 0.0F;
    sparse_moe_select_top_k_warp<kExperts, kTopK, HasShared>(
        logits + static_cast<std::size_t>(token) * rows, ids + token * kTopK,
        alpha + token * kTopK, HasShared ? shared_scale + token : &unused_shared, selected[warp]);
}

// -- moe_align_block_size --------------------------------------------------------------------

// One block per expert: how many of the round's assignments chose it.
__global__ void count_kernel(const std::int32_t* __restrict__ ids, int assignments,
                             std::int32_t* __restrict__ counts) {
    const int expert = static_cast<int>(blockIdx.x);
    int local        = 0;
    for (int i = static_cast<int>(threadIdx.x); i < assignments; i += blockDim.x) {
        local += ids[i] == expert ? 1 : 0;
    }
    local = warp_reduce_sum(local);
    __shared__ int partial[32];
    if ((threadIdx.x & 31) == 0) { partial[threadIdx.x / 32] = local; }
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < static_cast<int>(blockDim.x) / 32; ++w) { total += partial[w]; }
        counts[expert] = total;
    }
}

// One block: each expert's rows padded to whole blocks, their starts, the expert of every block
// and the padded total. `offsets` holds the counts on entry and the padded starts on exit.
__global__ void offsets_kernel(std::int32_t* __restrict__ offsets, int experts, int block_size,
                               std::int32_t* __restrict__ expert_ids,
                               std::int32_t* __restrict__ past_padded) {
    __shared__ int padded[1024];
    const int e = static_cast<int>(threadIdx.x);
    const int count = e < experts ? offsets[e] : 0;
    padded[e] = (count + block_size - 1) / block_size * block_size;
    __syncthreads();
    // Hillis-Steele inclusive scan; experts <= blockDim.x <= 1024.
    for (int stride = 1; stride < static_cast<int>(blockDim.x); stride <<= 1) {
        const int add = e >= stride ? padded[e - stride] : 0;
        __syncthreads();
        padded[e] += add;
        __syncthreads();
    }
    const int end   = padded[e];
    const int start = end - (count + block_size - 1) / block_size * block_size;
    __syncthreads();
    if (e < experts) {
        offsets[e] = start;
        for (int block = start / block_size; block < end / block_size; ++block) { expert_ids[block] = e; }
    }
    if (e == experts - 1) {
        offsets[experts] = end;
        *past_padded     = end;
    }
}

// One block per expert: its assignments in token order (deterministic), then the padding,
// which carries `assignments` -- the row id Marlin reads as out of range.
__global__ void scatter_kernel(const std::int32_t* __restrict__ ids, int assignments,
                               const std::int32_t* __restrict__ offsets, int block_size,
                               std::int32_t* __restrict__ sorted_ids) {
    const int expert = static_cast<int>(blockIdx.x);
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int warp   = static_cast<int>(threadIdx.x) / 32;
    const int warps  = static_cast<int>(blockDim.x) / 32;
    __shared__ int warp_counts[32];
    __shared__ int cursor;
    if (threadIdx.x == 0) { cursor = 0; }
    __syncthreads();
    const int start = offsets[expert];
    for (int base = 0; base < assignments; base += blockDim.x) {
        const int i       = base + static_cast<int>(threadIdx.x);
        const bool match  = i < assignments && ids[i] == expert;
        const unsigned ballot = __ballot_sync(0xffffffffu, match);
        if (lane == 0) { warp_counts[warp] = __popc(ballot); }
        __syncthreads();
        int before = cursor;
        for (int w = 0; w < warp; ++w) { before += warp_counts[w]; }
        if (match) { sorted_ids[start + before + __popc(ballot & ((1u << lane) - 1u))] = i; }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 0; w < warps; ++w) { cursor += warp_counts[w]; }
        }
        __syncthreads();
    }
    const int end = start + (cursor + block_size - 1) / block_size * block_size;
    for (int slot = start + cursor + static_cast<int>(threadIdx.x); slot < end; slot += blockDim.x) {
        sorted_ids[slot] = assignments;
    }
}

// -- combine --------------------------------------------------------------------------------

// destination[t] += sum_j routed[t * top_k + j] (router weight already applied) +
// shared_scale[t] * shared[t], in FP32, eight columns per thread.
__global__ void combine_kernel(const __nv_bfloat16* __restrict__ routed,
                               const __nv_bfloat16* __restrict__ shared,
                               const float* __restrict__ shared_scale, int hidden, int tokens,
                               __nv_bfloat16* __restrict__ destination) {
    const int chunk = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    if (chunk * 8 >= hidden || token >= tokens) { return; }
    const std::size_t column = static_cast<std::size_t>(chunk) * 8;
    float sum[8];
    {
        const uint4 packed = *reinterpret_cast<const uint4*>(destination + static_cast<std::size_t>(token) * hidden + column);
        const auto* v      = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(v[i]);
            sum[2 * i]     = f.x;
            sum[2 * i + 1] = f.y;
        }
    }
    float paths[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (int j = 0; j < kTopK; ++j) {
        const uint4 packed = *reinterpret_cast<const uint4*>(
            routed + (static_cast<std::size_t>(token) * kTopK + j) * hidden + column);
        const auto* v = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(v[i]);
            paths[2 * i] += f.x;
            paths[2 * i + 1] += f.y;
        }
    }
    if (shared != nullptr) {
        const float gate   = shared_scale[token];
        const uint4 packed = *reinterpret_cast<const uint4*>(shared + static_cast<std::size_t>(token) * hidden + column);
        const auto* v      = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(v[i]);
            paths[2 * i] = fmaf(gate, f.x, paths[2 * i]);
            paths[2 * i + 1] = fmaf(gate, f.y, paths[2 * i + 1]);
        }
    }
    uint4 out;
    auto* o = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        o[i] = __floats2bfloat162_rn(sum[2 * i] + paths[2 * i], sum[2 * i + 1] + paths[2 * i + 1]);
    }
    *reinterpret_cast<uint4*>(destination + static_cast<std::size_t>(token) * hidden + column) = out;
}

void run_slice(const SparseMoeGeometry& g, const Tensor& x, const Tensor& router_x,
               const SparseMoeWeights& weights, Tensor& destination, const SliceWorkspace& w,
               const Nvfp4MarlinScratch& scratch, cudaStream_t stream,
               const SparseMoeRoundHook* hook, std::int32_t first_token, WorkspaceArena& workspace,
               const SparseMoeNvfp4Adapters* adapters) {
    const std::int32_t tokens      = x.ne[1];
    const std::int32_t assignments = tokens * kTopK;
    const int rows                 = g.router_rows();
    const auto* router = static_cast<const __nv_bfloat16*>(weights.router_shared_gate.qdata);
    const auto* input  = static_cast<const __nv_bfloat16*>(router_x.data);
    if (tokens <= kNarrowRouterTokens) {
        const int warps_per_block = 8;
        router_narrow_kernel<<<(rows + warps_per_block - 1) / warps_per_block, 32 * warps_per_block, 0,
                               stream>>>(router, rows, input, tokens, g.hidden, w.logits);
    } else {
        const dim3 grid((rows + kTile - 1) / kTile, (tokens + kTile - 1) / kTile);
        if (g.hidden % kRouterBK == 0) {
            const dim3 mma_grid((rows + kRouterBM - 1) / kRouterBM, (tokens + kRouterBN - 1) / kRouterBN);
            router_mma_kernel<<<mma_grid, kRouterThreads, 0, stream>>>(router, rows, input, tokens, g.hidden,
                                                                       w.logits);
        } else {
            router_tiled_kernel<<<grid, 256, 0, stream>>>(router, rows, input, tokens, g.hidden, w.logits);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    if (g.has_shared()) {
        select_kernel<true><<<(tokens + 7) / 8, 256, 0, stream>>>(w.logits, rows, tokens, w.ids,
                                                                  w.alpha, w.shared_scale);
    } else {
        select_kernel<false><<<(tokens + 7) / 8, 256, 0, stream>>>(w.logits, rows, tokens, w.ids,
                                                                   w.alpha, w.shared_scale);
    }
    CUDA_CHECK(cudaGetLastError());
    if (hook != nullptr && hook->capture != nullptr) {
        const Tensor ids(w.ids, DType::I32, {kTopK, tokens});
        hook->capture(hook->capture_context, ids, hook->capture_base + first_token, tokens, stream);
    }

    const int block = moe_block_size(tokens, g.experts);
    count_kernel<<<g.experts, 256, 0, stream>>>(w.ids, assignments, w.offsets);
    offsets_kernel<<<1, 1024, 0, stream>>>(w.offsets, g.experts, block, w.expert_ids, w.past_padded);
    scatter_kernel<<<g.experts, 256, 0, stream>>>(w.ids, assignments, w.offsets, block, w.sorted_ids);
    CUDA_CHECK(cudaGetLastError());

    const Weight& gate_up = weights.routed_gate_up;
    const Weight& down    = weights.routed_down;
    marlin_moe_gemm_nvfp4_bf16(x.data, gate_up.qdata, gate_up.scales, gate_up.marlin_global_scale,
                               w.product, scratch.c_tmp, w.sorted_ids, w.expert_ids, w.past_padded,
                               w.alpha, tokens, kTopK, /*mul_topk_weights=*/false,
                               2 * g.intermediate, g.hidden, g.experts, block, scratch.locks,
                               scratch.sm_count, stream);
    CUDA_CHECK(cudaGetLastError());
    // An adapter round: each routed path's delta, on the same expert blocks.
    MarlinMoeLoraCall lora;
    if (adapters != nullptr) {
        lora.ids            = w.ids;
        lora.sorted_ids     = w.sorted_ids;
        lora.expert_ids     = w.expert_ids;
        lora.past_padded    = w.past_padded;
        lora.block          = block;
        lora.block_capacity = block_capacity(assignments, g.experts);
        lora.rows           = assignments;
        lora.top_k          = kTopK;
        lora.banks          = adapters->banks;
        lora.slots          = adapters->slots;
        lora.slot_stride    = adapters->slot_stride;
        lora.slot_count     = adapters->slot_count;
        lora.rank           = adapters->rank;
        lora.low            = w.lora_low;
        MarlinMoeLoraCall gate_up = lora;
        gate_up.input            = x.data;
        gate_up.input_per_token  = true;
        gate_up.out              = w.product;
        gate_up.out_ld           = 2 * g.intermediate;
        gate_up.k                = g.hidden;
        gate_up.n                = g.intermediate;
        gate_up.parts            = 2;
        gate_up.column_offset[0] = weights.routed_up_first ? g.intermediate : 0; // gate
        gate_up.column_offset[1] = weights.routed_up_first ? 0 : g.intermediate; // up
        marlin_moe_lora_delta(gate_up, stream);
    }
    marlin_moe_silu_mul(w.product, w.activation, assignments, g.intermediate, stream,
                        weights.routed_up_first);
    marlin_moe_gemm_nvfp4_bf16(w.activation, down.qdata, down.scales, down.marlin_global_scale,
                               w.product, scratch.c_tmp, w.sorted_ids, w.expert_ids, w.past_padded,
                               w.alpha, assignments, 1, /*mul_topk_weights=*/true, g.hidden,
                               g.intermediate, g.experts, block, scratch.locks, scratch.sm_count,
                               stream);
    CUDA_CHECK(cudaGetLastError());
    if (adapters != nullptr) {
        MarlinMoeLoraCall down = lora;
        down.input           = w.activation;
        down.input_per_token = false;
        down.top_k           = kTopK; // a row's token still selects its slot
        down.out             = w.product;
        down.out_ld          = g.hidden;
        down.k               = g.intermediate;
        down.n               = g.hidden;
        down.down            = true;
        down.parts           = 1;
        down.row_weights     = w.alpha;
        marlin_moe_lora_delta(down, stream);
    }

    if (g.has_shared() && weights.shared_gate_up.qtype != QType::NVFP4_A16) {
        // A checkpoint whose experts were rewritten at load (--moe-route marlin) keeps its shared
        // expert as stored: its own kernels, BF16 activations, into the same combine.
        auto scope = workspace.scope();
        Tensor activation(w.shared_act, DType::BF16, {g.shared_intermediate, tokens});
        Tensor shared_out(w.shared_out, DType::BF16, {g.hidden, tokens});
        if (adapters == nullptr) {
            linear_swiglu(x, weights.shared_gate_up, activation, LinearPolicy::A16Only, workspace, stream);
            linear(activation, weights.shared_down, shared_out, LinearPolicy::A16Only, workspace, stream);
        } else {
            // The adapter's shared-expert deltas go in before the SiLU and into its output: the
            // stored [gate; up] product is materialised for them.
            Tensor product(w.shared_up, DType::BF16, {2 * g.shared_intermediate, tokens});
            linear(x, weights.shared_gate_up, product, LinearPolicy::A16Only, workspace, stream);
            MarlinMoeLoraCall shared;
            shared.fixed_expert     = g.experts;
            shared.block            = 64;
            shared.block_capacity   = (tokens + 63) / 64;
            shared.rows             = tokens;
            shared.top_k            = 1;
            shared.banks            = adapters->banks;
            shared.slots            = adapters->slots;
            shared.slot_stride      = adapters->slot_stride;
            shared.slot_count       = adapters->slot_count;
            shared.rank             = adapters->rank;
            shared.low              = w.lora_shared_low;
            shared.input            = x.data;
            shared.out              = w.shared_up;
            shared.out_ld           = 2 * g.shared_intermediate;
            shared.k                = g.hidden;
            shared.n                = g.shared_intermediate;
            shared.parts            = 2;
            shared.column_offset[0] = 0;
            shared.column_offset[1] = g.shared_intermediate;
            marlin_moe_lora_delta(shared, stream);
            marlin_moe_silu_mul(w.shared_up, w.shared_act, tokens, g.shared_intermediate, stream);
            linear(activation, weights.shared_down, shared_out, LinearPolicy::A16Only, workspace, stream);
            MarlinMoeLoraCall shared_down = shared;
            shared_down.input            = w.shared_act;
            shared_down.out              = w.shared_out;
            shared_down.out_ld           = g.hidden;
            shared_down.k                = g.shared_intermediate;
            shared_down.n                = g.hidden;
            shared_down.down             = true;
            shared_down.parts            = 1;
            marlin_moe_lora_delta(shared_down, stream);
        }
    } else if (g.has_shared()) {
        const Weight& shared_gate_up = weights.shared_gate_up;
        const Weight& shared_down    = weights.shared_down;
        marlin_gemm_nvfp4_bf16(x.data, shared_gate_up.qdata, shared_gate_up.scales,
                               shared_gate_up.marlin_global_scale, w.shared_up, scratch.c_tmp,
                               scratch.locks, tokens, shared_gate_up.n, shared_gate_up.k,
                               scratch.sm_count, stream);
        CUDA_CHECK(cudaGetLastError());
        marlin_moe_silu_mul(w.shared_up, w.shared_act, tokens, g.shared_intermediate, stream);
        marlin_gemm_nvfp4_bf16(w.shared_act, shared_down.qdata, shared_down.scales,
                               shared_down.marlin_global_scale, w.shared_out, scratch.c_tmp,
                               scratch.locks, tokens, shared_down.n, shared_down.k,
                               scratch.sm_count, stream);
        CUDA_CHECK(cudaGetLastError());
    }
    const dim3 grid((g.hidden / 8 + 127) / 128, tokens);
    combine_kernel<<<grid, 128, 0, stream>>>(w.product, w.shared_out, w.shared_scale, g.hidden,
                                             tokens, static_cast<__nv_bfloat16*>(destination.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool sparse_moe_nvfp4_geometry(const SparseMoeGeometry& geometry) noexcept {
    return geometry.experts == kExperts && geometry.experts_per_token == kTopK &&
           geometry.gating == SparseMoeGating::SoftmaxTopK && geometry.routed_scale == 1.0F &&
           geometry.swiglu_limit == 0.0F && geometry.activation == GatedActivation::Silu &&
           !geometry.per_expert_scaled && (!geometry.has_shared() || geometry.shared_gated) &&
           geometry.hidden % 128 == 0 && geometry.intermediate % 128 == 0 &&
           geometry.shared_intermediate % 128 == 0;
}

std::size_t sparse_moe_nvfp4_workspace_bytes(const SparseMoeGeometry& geometry,
                                             std::int32_t max_tokens) {
    if (max_tokens <= 0) { throw std::invalid_argument("sparse_moe nvfp4: invalid token count"); }
    const std::int32_t slice = std::min(max_tokens, kSparseMoeNvfp4Slice);
    std::size_t bytes        = 0;
    (void)layout(geometry, slice, [&](std::size_t size) -> void* {
        bytes += aligned(size);
        return nullptr;
    });
    return bytes + kAlign;
}

void sparse_moe_nvfp4(const SparseMoeGeometry& geometry, const Tensor& x, const Tensor& router_x,
                      const SparseMoeWeights& weights, Tensor& destination,
                      WorkspaceArena& workspace, cudaStream_t stream,
                      const SparseMoeRoundHook* hook, const SparseMoeNvfp4Adapters* adapters) {
    const Nvfp4MarlinScratch scratch = nvfp4_marlin_scratch();
    const std::int32_t tokens        = x.ne[1];
    for (std::int32_t offset = 0; offset < tokens; offset += kSparseMoeNvfp4Slice) {
        const std::int32_t count = std::min(kSparseMoeNvfp4Slice, tokens - offset);
        auto scope               = workspace.scope();
        const SliceWorkspace views = layout(geometry, count, [&](std::size_t size) -> void* {
            return workspace.alloc_bytes(aligned(size), kAlign).data;
        });
        const Tensor x_slice      = x.slice(1, offset, count);
        const Tensor router_slice = router_x.slice(1, offset, count);
        Tensor destination_slice  = destination.slice(1, offset, count);
        // A slice's tokens are the round's [offset, offset + count): their slots follow them.
        SparseMoeNvfp4Adapters slice_adapters;
        if (adapters != nullptr) {
            slice_adapters = *adapters;
            if (slice_adapters.slot_stride != 0) {
                slice_adapters.slots += static_cast<std::ptrdiff_t>(offset) * slice_adapters.slot_stride;
            }
        }
        run_slice(geometry, x_slice, router_slice, weights, destination_slice, views, scratch,
                  stream, hook, offset, workspace, adapters != nullptr ? &slice_adapters : nullptr);
    }
}

} // namespace sinfer::ops::detail
