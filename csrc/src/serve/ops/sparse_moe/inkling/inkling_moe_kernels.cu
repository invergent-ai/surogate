// Inkling MoE: router, activation quantisers, standalone SiTU, combine and the LoRA shrink.
// Numerics: inkling_numerics.cuh; design: rl/engine-inkling-2026-11-21/lanes/d/DESIGN-moe.md.

#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/kernel/func_attribute.cuh"
#include "ops/sparse_moe/inkling/inkling_gemm_common.cuh"
#include "ops/sparse_moe/inkling/inkling_moe_internal.h"
#include "ops/sparse_moe/inkling/inkling_numerics.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace sinfer::ops::detail::inkling {

static_assert(std::bit_cast<std::uint32_t>(kInv1536) == 0x3A2AAAABU);
static_assert(std::bit_cast<std::uint32_t>(kOneSixth) == 0x3E2AAAABU);
static_assert(std::bit_cast<std::uint32_t>(kInv448) == 0x3B124925U);
static_assert(std::bit_cast<std::uint32_t>(kInv25) == 0x3D23D70AU);
static_assert(std::bit_cast<std::uint32_t>(kTwoPowMinus127) == 0x00400000U);
static_assert(std::bit_cast<std::uint32_t>(kTwoPow127) == 0x7F000000U);

namespace {

void check_launch(const char* what) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("inkling_moe: ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

__device__ __forceinline__ int live_count(LiveTokens live) {
    return live.device != nullptr ? min(*live.device, live.tokens) : live.tokens;
}

/// PDL audit (agent-EN 17:48, lane c's FP8 finding): per-step activations (x, y, logits, ids, gammas, down rows)
/// are written by a kernel that may still be in flight when this grid is launched (PDL) and live at arena addresses
/// reused every visit, so they are NEVER read through the non-coherent read-only path (ld.global.nc / __ldg / a
/// `const __restrict__` pointer the compiler turns into LDG.CONSTANT): explicit ld.global.cg (L2) loads instead.
/// Same bytes: no bit changes. The bf16 ROW loads (x / y in the quantisers) are explicit weak ld.global.ca (load_row16
/// in inkling_gemm_common.cuh, shared with the fused-y down: an L1 hit on the second pass; a bare dereference of these
/// `const __restrict__` inputs would be lowered to ld.global.nc, review R4); single-touch reads (logits,
/// ids, lora rows, combine / softcap operands) are ld.global.cg. Test-only controls (`inkling_test_read_only_loads`):
/// 1 = the row loads back on ld.global.nc (negative control), 2 = ld.global.cg.

/// 16 bf16 at `source` (32-byte aligned) as fp32 (load_row16).
template <int RowLoad>
__device__ __forceinline__ void load16(const __nv_bfloat16* source, float* x) {
    const uint4* words = reinterpret_cast<const uint4*>(source);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint4 v                 = load_row16<RowLoad>(words + half);
        const std::uint32_t parts[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            x[half * 8 + 2 * i]     = __uint_as_float(parts[i] << 16);
            x[half * 8 + 2 * i + 1] = __uint_as_float(parts[i] & 0xFFFF0000U);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Router logits: grid (16 expert tiles of 8, ceil(T / 16)), 3 warps = the 3 K groups.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void router_logits_tile(const __nv_bfloat16* __restrict__ x,
                                                   const __nv_bfloat16* __restrict__ router,
                                                   float* __restrict__ logits, std::int32_t tokens,
                                                   std::int32_t* __restrict__ zero, std::int32_t zero_words,
                                                   int e_tile, int m_tile) {
    const int thread = static_cast<int>(threadIdx.x);
    if (zero != nullptr && e_tile == 0 && m_tile == 0) {
        for (int i = thread; i < zero_words; i += 96) { zero[i] = 0; }
    }
    const int warp     = thread >> 5;
    const int lane     = thread & 31;
    const int group_id = lane >> 2;
    const int quad     = lane & 3;
    const int e0       = e_tile * 8;
    const int m0       = m_tile * 16;
    const int k_begin  = warp * kRouterGroup;
    const int k_end    = min(k_begin + kRouterGroup, kHidden);
    const int row0     = m0 + group_id;
    const int row1     = row0 + 8;
    const bool valid0  = row0 < tokens;
    const bool valid1  = row1 < tokens;
    // x through L2 (PDL audit: written by the predecessor, reused address).
    const auto* x0     = reinterpret_cast<const unsigned int*>(x + static_cast<std::int64_t>(valid0 ? row0 : 0) * kHidden);
    const auto* x1     = reinterpret_cast<const unsigned int*>(x + static_cast<std::int64_t>(valid1 ? row1 : 0) * kHidden);
    const auto* w      = reinterpret_cast<const std::uint32_t*>(router + static_cast<std::int64_t>(e0 + group_id) * kHidden);

    // One chain of m16n8k16 MMAs from zero over the group, k ascending: `_k_group_tile`'s
    // `partial = tl.dot(x, w, partial)` over 128-wide tiles is this chain.
    float c0 = 0.0F, c1 = 0.0F, c2 = 0.0F, c3 = 0.0F;
    for (int k = k_begin; k < k_end; k += 16) {
        const int lo = (k + 2 * quad) >> 1;
        const int hi = (k + 8 + 2 * quad) >> 1;
        const std::uint32_t a0 = valid0 ? __ldcg(x0 + lo) : 0U;
        const std::uint32_t a1 = valid1 ? __ldcg(x1 + lo) : 0U;
        const std::uint32_t a2 = valid0 ? __ldcg(x0 + hi) : 0U;
        const std::uint32_t a3 = valid1 ? __ldcg(x1 + hi) : 0U;
        mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, w[lo], w[hi]);
    }
    __shared__ float partial[3][16][8];
    partial[warp][group_id][2 * quad]         = c0;
    partial[warp][group_id][2 * quad + 1]     = c1;
    partial[warp][group_id + 8][2 * quad]     = c2;
    partial[warp][group_id + 8][2 * quad + 1] = c3;
    __syncthreads();
    for (int i = thread; i < 128; i += 96) {
        const int row = i >> 3;
        const int col = i & 7;
        const int token = m0 + row;
        if (token < tokens) {
            // `_group_sum`: total = p0; total += p1; total += p2.
            logits[static_cast<std::int64_t>(token) * kExperts + e0 + col] =
                __fadd_rn(__fadd_rn(partial[0][row][col], partial[1][row][col]), partial[2][row][col]);
        }
    }
}

__global__ __launch_bounds__(96) void router_logits_kernel(const __nv_bfloat16* __restrict__ x,
                                                           const __nv_bfloat16* __restrict__ router,
                                                           float* __restrict__ logits,
                                                           std::int32_t tokens,
                                                           std::int32_t* __restrict__ zero,
                                                           std::int32_t zero_words) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    router_logits_tile(x, router, logits, tokens, zero, zero_words, static_cast<int>(blockIdx.x),
                       static_cast<int>(blockIdx.y));
}

// ---------------------------------------------------------------------------------------------
// Router logits, wide rounds: a CTA = 32 tokens x `Experts` experts, 8 warps (2 x 16-token rows
// x 4 expert groups), x and W streamed through shared memory in 64-wide K chunks (cp.async,
// 4 stages, 128-byte rows swizzled by row). Each output is the SAME arithmetic as the kernel
// above: per 1024-wide K group one chain of bf16 m16n8k16 MMAs from zero, k ascending, the
// canonical fragments (ldmatrix gives the registers the direct loads give), then
// `(p0 + p1) + p2`.
// ---------------------------------------------------------------------------------------------
inline constexpr int kRouterTileTokens = 32;
inline constexpr int kRouterKChunk     = 64;
inline constexpr int kRouterStages     = 4;

template <int Experts>
struct RouterTiledShared {
    alignas(128) __nv_bfloat16 x[kRouterStages][kRouterTileTokens * kRouterKChunk];
    alignas(128) __nv_bfloat16 w[kRouterStages][Experts * kRouterKChunk];
};

/// Element offset of (row, k) in a 64-wide bf16 chunk row: 16-byte segment k / 8 at (k / 8) ^ (row & 7).
__device__ __forceinline__ int router_swizzled(int row, int k) {
    return row * kRouterKChunk + ((((k >> 3) ^ (row & 7))) << 3) + (k & 7);
}

template <int Experts>
__global__ __launch_bounds__(256) void router_logits_tiled_kernel(const __nv_bfloat16* __restrict__ x,
                                                                  const __nv_bfloat16* __restrict__ router,
                                                                  float* __restrict__ logits,
                                                                  std::int32_t tokens,
                                                                  std::int32_t* __restrict__ zero,
                                                                  std::int32_t zero_words) {
    constexpr int kWarpExperts = Experts / 4;
    constexpr int kMmaN        = kWarpExperts / 8;
    constexpr int kChunks      = kHidden / kRouterKChunk;       // 36
    constexpr int kGroupChunks = kRouterGroup / kRouterKChunk;  // 16
    static_assert(kMmaN >= 1 && kWarpExperts % 8 == 0);
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto& shared = *reinterpret_cast<RouterTiledShared<Experts>*>(shared_bytes);

    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    const int thread = static_cast<int>(threadIdx.x);
    if (zero != nullptr && blockIdx.x == 0 && blockIdx.y == 0) {
        for (int i = thread; i < zero_words; i += 256) { zero[i] = 0; }
    }
    const int m0 = static_cast<int>(blockIdx.x) * kRouterTileTokens;
    const int e0 = static_cast<int>(blockIdx.y) * Experts;
    const auto load_stage = [&](int stage, int chunk) {
        {
            const int row   = thread >> 3;
            const int seg   = thread & 7;
            const int token = m0 + row;
            const bool valid = token < tokens;
            cp_async_zfill<16, Cache::cg>(shared.x[stage] + router_swizzled(row, seg * 8),
                                          x + static_cast<std::int64_t>(valid ? token : 0) * kHidden +
                                              chunk * kRouterKChunk + seg * 8,
                                          valid ? 16 : 0);
        }
#pragma unroll
        for (int task = thread; task < Experts * 8; task += 256) {
            const int row = task >> 3;
            const int seg = task & 7;
            cp_async<16, Cache::cg>(shared.w[stage] + router_swizzled(row, seg * 8),
                                    router + static_cast<std::int64_t>(e0 + row) * kHidden +
                                        chunk * kRouterKChunk + seg * 8);
        }
    };
#pragma unroll
    for (int stage = 0; stage < kRouterStages - 1; ++stage) {
        load_stage(stage, stage);
        cp_commit();
    }

    const int lane   = thread & 31;
    const int warp   = thread >> 5;
    const int warp_m = warp & 1;
    const int warp_n = warp >> 1;
    float total[kMmaN][4] = {};
    float acc[kMmaN][4]   = {};
    for (int chunk = 0; chunk < kChunks; ++chunk) {
        cp_wait<kRouterStages - 2>();
        __syncthreads();
        if (chunk + kRouterStages - 1 < kChunks) {
            load_stage((chunk + kRouterStages - 1) % kRouterStages, chunk + kRouterStages - 1);
        }
        cp_commit();
        const int stage = chunk % kRouterStages;
#pragma unroll
        for (int kk = 0; kk < kRouterKChunk / 16; ++kk) {
            unsigned a0, a1, a2, a3;
            {
                const int row = warp_m * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int k   = kk * 16 + (lane >> 4) * 8;
                ldmatrix_x4(a0, a1, a2, a3, smem_addr(shared.x[stage] + router_swizzled(row, k)));
            }
#pragma unroll
            for (int j = 0; j < kMmaN; ++j) {
                const int row = warp_n * kWarpExperts + j * 8 + (lane & 7);
                const int k   = kk * 16 + ((lane >> 3) & 1) * 8;
                unsigned b0, b1;
                ldmatrix_x2(b0, b1, smem_addr(shared.w[stage] + router_swizzled(row, k)));
                mma_bf16(acc[j][0], acc[j][1], acc[j][2], acc[j][3], a0, a1, a2, a3, b0, b1);
            }
        }
        if ((chunk + 1) % kGroupChunks == 0 || chunk + 1 == kChunks) {
            // `_group_sum`: total = p0; total += p1; total += p2.
            const bool first = chunk < kGroupChunks;
#pragma unroll
            for (int j = 0; j < kMmaN; ++j) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    total[j][i] = first ? acc[j][i] : __fadd_rn(total[j][i], acc[j][i]);
                    acc[j][i]   = 0.0F;
                }
            }
        }
    }
    const int row0 = m0 + warp_m * 16 + (lane >> 2);
#pragma unroll
    for (int j = 0; j < kMmaN; ++j) {
        const int column = e0 + warp_n * kWarpExperts + j * 8 + 2 * (lane & 3);
        if (row0 < tokens) {
            *reinterpret_cast<float2*>(logits + static_cast<std::int64_t>(row0) * kExperts + column) =
                make_float2(total[j][0], total[j][1]);
        }
        if (row0 + 8 < tokens) {
            *reinterpret_cast<float2*>(logits + static_cast<std::int64_t>(row0 + 8) * kExperts + column) =
                make_float2(total[j][2], total[j][3]);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Selection: a warp per token, 4 tokens per block; the last block lays out the expert rows.
// ---------------------------------------------------------------------------------------------
/// One warp selects token `token` (< live.tokens): zeroes its fused-combine counters (and its output row
/// when it is padding), picks the top 8, writes ids / gammas / capture and, with `out.counts`, takes a
/// rank inside each expert. The logits are read through L2 (ld.cg) on every path: written by the logits kernel or by
/// other CTAs of the same grid.
__device__ __forceinline__ void select_token(const float* __restrict__ logits, const float* __restrict__ bias,
                                             LiveTokens live, const RouteOutputs& out, int token, int lane) {
    const int tokens = live.tokens;
    const int alive  = live_count(live);
    if (token < tokens && out.arrivals != nullptr) {
        for (int i = lane; i < kArrivalTiles; i += 32) {
            out.arrivals[static_cast<std::int64_t>(token) * kArrivalTiles + i] = 0;
        }
    }
    if (token < tokens && token >= alive && out.combined != nullptr) {
        uint4* row = reinterpret_cast<uint4*>(out.combined + static_cast<std::int64_t>(token) * kHidden);
        for (int i = lane; i < kHidden / 8; i += 32) { row[i] = make_uint4(0U, 0U, 0U, 0U); }
    }
    if (token < tokens) {
        const float* row = logits + static_cast<std::int64_t>(token) * kExperts;
        // Logits through L2 on both paths (PDL audit: written by the logits kernel / other CTAs of this grid).
        const auto logit = [&](int expert) { return __ldcg(row + expert); };
        if (token < alive) {
            std::uint64_t keys[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int expert = lane + 32 * i;
                keys[i]          = router_key(logit(expert), bias[expert], expert);
            }
            int my_expert = -1;
#pragma unroll
            for (int slot = 0; slot < kTopK; ++slot) {
                std::uint64_t local = keys[0];
#pragma unroll
                for (int i = 1; i < 4; ++i) { local = keys[i] > local ? keys[i] : local; }
                const std::uint64_t best = warp_max_key48(local);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (keys[i] == best) { keys[i] = 0; }
                }
                if (lane == slot) { my_expert = router_key_expert(best); }
            }
            const float my_log_prob = lane < kTopK ? router_log_sigmoid(logit(my_expert)) : 0.0F;
            float log_probs[kTopK];
#pragma unroll
            for (int slot = 0; slot < kTopK; ++slot) {
                log_probs[slot] = __shfl_sync(0xFFFFFFFFU, my_log_prob, slot);
            }
            float weights[kTopK];
            router_weights(log_probs, weights);
            float gamma = 0.0F;
#pragma unroll
            for (int slot = 0; slot < kTopK; ++slot) {
                if (lane == slot) { gamma = weights[slot]; }
            }
            if (lane < kTopK) {
                const int index    = token * kTopK + lane;
                out.ids[index]     = my_expert;
                if (out.local_ids != nullptr) { out.local_ids[index] = my_expert; }
                out.gammas[index]  = gamma;
                if (out.counts != nullptr) {
                    out.ranks[index] = atomicAdd(out.counts + my_expert, 1);
                }
                if (out.capture != nullptr && out.capture_first + token < out.capture_capacity) {
                    out.capture[static_cast<std::int64_t>(out.capture_first + token) * out.capture_stride + lane] =
                        my_expert;
                }
            }
        } else if (lane < kTopK) {
            const int index   = token * kTopK + lane;
            out.ids[index]    = -1;
            if (out.local_ids != nullptr) { out.local_ids[index] = -1; }
            out.gammas[index] = 0.0F;
            if (out.counts != nullptr) { out.ranks[index] = -1; }
        }
    }
}

/// The CTA that holds the last ticket lays out the rows: offsets by expert (an integer prefix of the
/// counts), then, with `out.last_cta_places`, each routed slot at offsets[expert] + rank. Threads
/// [0, 128) scan (one expert each); all `Threads` place. Called by every thread of the CTA.
/// `Local`: counts / ids / ranks are this CTA's shared-memory copies (`out.counts`, `out.local_ids`,
/// `out.ranks`), read after a barrier; otherwise they come from global memory through L2.
template <int Threads, bool Local = false>
__device__ __forceinline__ void place_rows(const RouteOutputs& out, int tokens, int thread) {
    static_assert(Threads >= kExperts && Threads % 32 == 0);
    __shared__ int offsets[kExperts + 1];
    __shared__ int warp_totals[kExperts / 32];
    const int warp = thread >> 5, lane = thread & 31;
    int count = 0, inclusive = 0;
    if (thread < kExperts) {
        count     = Local ? out.counts[thread] : __ldcg(out.counts + thread);
        inclusive = count;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int other = __shfl_up_sync(0xFFFFFFFFU, inclusive, offset);
            if (lane >= offset) { inclusive += other; }
        }
        if (lane == 31) { warp_totals[warp] = inclusive; }
    }
    __syncthreads();
    if (thread < kExperts) {
        int before = 0;
        for (int w = 0; w < warp; ++w) { before += warp_totals[w]; }
        const int exclusive = before + inclusive - count;
        offsets[thread]     = exclusive;
        out.offsets[thread] = exclusive;
        if (thread == kExperts - 1) {
            offsets[kExperts]     = exclusive + count;
            out.offsets[kExperts] = exclusive + count;
        }
    }
    if (!out.last_cta_places) { return; }
    __syncthreads();
    for (int index = thread; index < tokens * kTopK; index += Threads) {
        const int expert = Local ? out.local_ids[index] : __ldcg(out.ids + index);
        if (expert >= 0) { out.sorted[offsets[expert] + (Local ? out.ranks[index] : __ldcg(out.ranks + index))] = index; }
    }
}

__global__ __launch_bounds__(128) void router_select_kernel(const float* __restrict__ logits,
                                                            const float* __restrict__ bias,
                                                            LiveTokens live, RouteOutputs out) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    const int thread = static_cast<int>(threadIdx.x);
    select_token(logits, bias, live, out, static_cast<int>(blockIdx.x) * 4 + (thread >> 5), thread & 31);
    if (out.counts == nullptr) { return; }
    // The last block to finish lays out the rows.
    __shared__ int is_last;
    __threadfence();
    __syncthreads();
    if (thread == 0) { is_last = atomicAdd(out.ticket, 1) == static_cast<int>(gridDim.x) - 1; }
    __syncthreads();
    if (!is_last) { return; }
    __threadfence();
    place_rows<128>(out, live.tokens, thread);
}

// ---------------------------------------------------------------------------------------------
// Placement of wide rounds: a thread per routed slot (the select kernel wrote ids, ranks and
// the offsets). Positions are offsets[e] + rank; their order inside an expert never changes a
// row's arithmetic.
// ---------------------------------------------------------------------------------------------
__global__ __launch_bounds__(256) void router_place_kernel(RouteOutputs out, std::int32_t slots) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    const int index = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    if (index >= slots) { return; }
    const int expert = out.ids[index];
    if (expert >= 0) { out.sorted[out.offsets[expert] + out.ranks[index]] = index; }
}

// ---------------------------------------------------------------------------------------------
// Row quantisers: a warp per row, 8 rows per block.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ bool row_is_live(int row, const std::int32_t* row_ids, LiveTokens live) {
    if (row_ids != nullptr) { return __ldcg(row_ids + row) >= 0; }
    return row < live_count(live);
}

/// One row's NVFP4 four-over-six by one warp.
template <int RowLoad>
__device__ __forceinline__ void quant_nvfp4_row(const __nv_bfloat16* __restrict__ input, std::int32_t rows,
                                                std::int32_t k, const std::int32_t* __restrict__ row_ids,
                                                LiveTokens live, std::uint8_t* __restrict__ codes,
                                                std::uint8_t* __restrict__ scales, float* __restrict__ amax_out,
                                                int row) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (row >= rows || !row_is_live(row, row_ids, live)) { return; }
    const __nv_bfloat16* source = input + static_cast<std::int64_t>(row) * k;
    const int blocks            = k / 16;
    float amax                  = 0.0F;
    for (int block = lane; block < blocks; block += 32) {
        float x[16];
        load16<RowLoad>(source + block * 16, x);
#pragma unroll
        for (int j = 0; j < 16; ++j) { amax = fmaxf(amax, fabsf(x[j])); }
    }
    amax                         = warp_max(amax);
    const Nvfp4RowScales row_scales = nvfp4_row_scales(amax);
    for (int block = lane; block < blocks; block += 32) {
        float x[16];
        load16<RowLoad>(source + block * 16, x);
        const Nvfp4Block quantized = nvfp4_quantize_block_4o6(x, row_scales, nvfp4_block_in_head(block, k));
        *reinterpret_cast<uint2*>(codes + static_cast<std::int64_t>(row) * (k / 2) + block * 8) =
            make_uint2(static_cast<std::uint32_t>(quantized.codes),
                       static_cast<std::uint32_t>(quantized.codes >> 32));
        scales[static_cast<std::int64_t>(row) * blocks + block] = quantized.scale;
    }
    if (lane == 0) { amax_out[row] = amax; }
}

template <int RowLoad>
__global__ __launch_bounds__(256) void quant_nvfp4_kernel(const __nv_bfloat16* __restrict__ input,
                                                          std::int32_t rows, std::int32_t k,
                                                          const std::int32_t* __restrict__ row_ids,
                                                          LiveTokens live,
                                                          std::uint8_t* __restrict__ codes,
                                                          std::uint8_t* __restrict__ scales,
                                                          float* __restrict__ amax_out) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    quant_nvfp4_row<RowLoad>(input, rows, k, row_ids, live, codes, scales, amax_out,
                    static_cast<int>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5));
}

/// One row's MXFP8 quantisation by one warp.
template <int RowLoad>
__device__ __forceinline__ void quant_mxfp8_row(const __nv_bfloat16* __restrict__ input, std::int32_t rows,
                                                std::int32_t k, const std::int32_t* __restrict__ row_ids,
                                                LiveTokens live, std::uint8_t* __restrict__ codes,
                                                std::uint8_t* __restrict__ exponents,
                                                __nv_bfloat16* __restrict__ dequantised, int row) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (row >= rows || !row_is_live(row, row_ids, live)) { return; }
    const __nv_bfloat16* source = input + static_cast<std::int64_t>(row) * k;
    const int blocks            = k / 32;
    for (int block = lane; block < blocks; block += 32) {
        float x[32];
        load16<RowLoad>(source + block * 32, x);
        load16<RowLoad>(source + block * 32 + 16, x + 16);
        float amax = 0.0F;
#pragma unroll
        for (int j = 0; j < 32; ++j) { amax = fmaxf(amax, fabsf(x[j])); }
        float inverse                 = 0.0F;
        const std::uint32_t exponent  = mxfp8_block_exponent(amax, inverse);
        const float scale             = e8m0_to_f32(exponent);
        std::uint32_t packed[8];
        std::uint32_t deq[16];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            std::uint32_t word = 0;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                word |= static_cast<std::uint32_t>(mxfp8_encode(x[4 * q + b], inverse)) << (8 * b);
            }
            packed[q] = word;
        }
#pragma unroll
        for (int p = 0; p < 16; ++p) {
            const std::uint16_t pair = static_cast<std::uint16_t>(packed[p >> 1] >> (16 * (p & 1)));
            const __nv_bfloat162 v   = mxfp8_dequant_pair(pair, scale);
            deq[p]                   = *reinterpret_cast<const std::uint32_t*>(&v);
        }
        const std::int64_t base = static_cast<std::int64_t>(row) * k + block * 32;
        if (codes != nullptr) {
            auto* destination = reinterpret_cast<uint4*>(codes + base);
            destination[0]    = make_uint4(packed[0], packed[1], packed[2], packed[3]);
            destination[1]    = make_uint4(packed[4], packed[5], packed[6], packed[7]);
        }
        if (exponents != nullptr) {
            exponents[static_cast<std::int64_t>(row) * blocks + block] = static_cast<std::uint8_t>(exponent);
        }
        if (dequantised != nullptr) {
            auto* destination = reinterpret_cast<uint4*>(dequantised + base);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                destination[q] = make_uint4(deq[4 * q], deq[4 * q + 1], deq[4 * q + 2], deq[4 * q + 3]);
            }
        }
    }
}

template <int RowLoad>
__global__ __launch_bounds__(256) void quant_mxfp8_kernel(const __nv_bfloat16* __restrict__ input,
                                                          std::int32_t rows, std::int32_t k,
                                                          const std::int32_t* __restrict__ row_ids,
                                                          LiveTokens live,
                                                          std::uint8_t* __restrict__ codes,
                                                          std::uint8_t* __restrict__ exponents,
                                                          __nv_bfloat16* __restrict__ dequantised) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    quant_mxfp8_row<RowLoad>(input, rows, k, row_ids, live, codes, exponents, dequantised,
                    static_cast<int>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5));
}

// ---------------------------------------------------------------------------------------------
// Router logits, narrow rounds, staged: a block = 8 experts x 8 tokens, 3 warps (the K groups).
// The block's 8 router rows and 8 token rows are copied into shared memory with cp.async in one
// round trip, then each warp runs its chain from shared memory: the old tile kernel paid one
// dependent global round trip per k step (64 per group: ~25 us at T = 1 with cold weights).
// Same MMA, same fragments (a0 / a2 = token row g, a1 / a3 = 0: rows 8-15 are padding, whose
// outputs are discarded and never touch rows 0-7), same chains, same `(p0 + p1) + p2`.
// ---------------------------------------------------------------------------------------------
inline constexpr int kStagedRowBytes   = kHidden * 2 + 16; ///< 4624: 8 rows x 4 quads hit 32 banks
inline constexpr int kStagedTileTokens = 8;
inline constexpr int kStagedSharedBytes = 2 * 8 * kStagedRowBytes;

/// The tile's 8 router rows into `staged` (cp.async, one commit group). The router weights are model
/// weights no kernel writes, so this may run BEFORE the PDL dependency wait (the router rows then stream
/// while the predecessor still runs; same bytes, same bits).
template <int Threads>
__device__ __forceinline__ void stage_router_rows(const __nv_bfloat16* __restrict__ router, int e_tile,
                                                  unsigned char* staged) {
    const int thread = static_cast<int>(threadIdx.x);
    const int e0     = e_tile * 8;
    constexpr int kChunks = kHidden / 8; // 16-byte chunks per row
    for (int i = thread; i < 8 * kChunks; i += Threads) {
        const int row   = i / kChunks;
        const int chunk = i - row * kChunks;
        cp_async<16, Cache::cg>(staged + row * kStagedRowBytes + chunk * 16,
                                router + static_cast<std::int64_t>(e0 + row) * kHidden + chunk * 8);
    }
    cp_commit();
}

/// `Threads` >= 96: warps 0-2 run the K groups' MMA chains; all threads stage. `router_staged`: the
/// caller already issued stage_router_rows (before its dependency wait).
template <int Threads>
__device__ __forceinline__ void router_logits_staged_tile(const __nv_bfloat16* __restrict__ x,
                                                          const __nv_bfloat16* __restrict__ router,
                                                          float* __restrict__ logits, std::int32_t tokens,
                                                          std::int32_t* __restrict__ zero, std::int32_t zero_words,
                                                          int e_tile, int m_tile, unsigned char* staged,
                                                          bool router_staged = false) {
    const int thread = static_cast<int>(threadIdx.x);
    if (zero != nullptr && e_tile == 0 && m_tile == 0) {
        for (int i = thread; i < zero_words; i += Threads) { zero[i] = 0; }
    }
    const int e0 = e_tile * 8;
    const int m0 = m_tile * kStagedTileTokens;
    unsigned char* w_rows = staged;
    unsigned char* x_rows = staged + 8 * kStagedRowBytes;
    constexpr int kChunks = kHidden / 8; // 16-byte chunks per row
    if (!router_staged) { stage_router_rows<Threads>(router, e_tile, w_rows); }
    for (int i = thread; i < 8 * kChunks; i += Threads) {
        const int row    = i / kChunks;
        const int chunk  = i - row * kChunks;
        const int token  = m0 + row;
        const bool valid = token < tokens;
        cp_async_zfill<16, Cache::cg>(x_rows + row * kStagedRowBytes + chunk * 16,
                                      x + static_cast<std::int64_t>(valid ? token : 0) * kHidden + chunk * 8,
                                      valid ? 16 : 0);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    const int warp     = thread >> 5;
    const int lane     = thread & 31;
    const int group_id = lane >> 2;
    const int quad     = lane & 3;
    const int k_begin  = warp * kRouterGroup;
    const int k_end    = min(k_begin + kRouterGroup, kHidden);
    const auto* x0 = reinterpret_cast<const std::uint32_t*>(x_rows + group_id * kStagedRowBytes);
    const auto* w  = reinterpret_cast<const std::uint32_t*>(w_rows + group_id * kStagedRowBytes);
    __shared__ float partial[3][kStagedTileTokens][8];
    if (Threads == 96 || warp < 3) {
        float c0 = 0.0F, c1 = 0.0F, c2 = 0.0F, c3 = 0.0F;
#pragma unroll 8
        for (int k = k_begin; k < k_end; k += 16) {
            const int lo = (k + 2 * quad) >> 1;
            const int hi = (k + 8 + 2 * quad) >> 1;
            mma_bf16(c0, c1, c2, c3, x0[lo], 0U, x0[hi], 0U, w[lo], w[hi]);
        }
        partial[warp][group_id][2 * quad]     = c0;
        partial[warp][group_id][2 * quad + 1] = c1;
    }
    __syncthreads();
    if (thread < kStagedTileTokens * 8) {
        const int row   = thread >> 3;
        const int col   = thread & 7;
        const int token = m0 + row;
        if (token < tokens) {
            // `_group_sum`: total = p0; total += p1; total += p2.
            logits[static_cast<std::int64_t>(token) * kExperts + e0 + col] =
                __fadd_rn(__fadd_rn(partial[0][row][col], partial[1][row][col]), partial[2][row][col]);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Narrow rounds (decode / verify): router logits tiles and the x quantisation in ONE grid
// (DESIGN-moe 2.1; one launch fewer). Blocks [0, 16 * ceil(T / 16)) are router tiles, the rest
// quantise x rows, a warp per row. Each part runs the same code as its standalone kernel.
// ---------------------------------------------------------------------------------------------
struct XQuantOutputs {
    std::uint8_t* codes      = nullptr; ///< NVFP4
    std::uint8_t* scales     = nullptr; ///< NVFP4
    float* amax              = nullptr; ///< NVFP4
    __nv_bfloat16* dequantised = nullptr; ///< MXFP8
};

template <bool Nvfp4, int RowLoad>
__global__ __launch_bounds__(96) void route_quant_narrow_kernel(const __nv_bfloat16* __restrict__ x,
                                                                const __nv_bfloat16* __restrict__ router,
                                                                float* __restrict__ logits, std::int32_t tokens,
                                                                std::int32_t* __restrict__ zero,
                                                                std::int32_t zero_words, LiveTokens live,
                                                                XQuantOutputs q) {
    extern __shared__ __align__(16) unsigned char staged[];
    const int block        = static_cast<int>(blockIdx.x);
    const int router_tiles = (kExperts / 8) * ((tokens + kStagedTileTokens - 1) / kStagedTileTokens);
    // Router rows before the dependency wait (weights only; every read of x and every write after it).
    if (block < router_tiles) { stage_router_rows<96>(router, block % (kExperts / 8), staged); }
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    if (block < router_tiles) {
        router_logits_staged_tile<96>(x, router, logits, tokens, zero, zero_words, block % (kExperts / 8),
                                  block / (kExperts / 8), staged, true);
        return;
    }
    const int row = (block - router_tiles) * 3 + (static_cast<int>(threadIdx.x) >> 5);
    if constexpr (Nvfp4) {
        quant_nvfp4_row<RowLoad>(x, tokens, kHidden, nullptr, live, q.codes, q.scales, q.amax, row);
    } else {
        quant_mxfp8_row<RowLoad>(x, tokens, kHidden, nullptr, live, nullptr, nullptr, q.dequantised, row);
    }
}

// ---------------------------------------------------------------------------------------------
// Narrow rounds with the persistent zeroed state (InklingMoeOptions::state): router tiles, x
// quantisation AND the selection in one grid. The 16 router CTAs of token tile m arrive on
// tile_arrivals[m]; the last to arrive selects the tile's 8 tokens (a warp each, select_token) and
// takes a ticket; the CTA with the last ticket lays out the rows (place_rows) and returns every
// counter of the call to zero (self-cleaning: the next call starts from zeros without a memset).
// The counters decide only WHICH CTA selects / places, never the order of any floating-point sum:
// logits, selection and weights are the standalone kernels' code on the same inputs.
// ---------------------------------------------------------------------------------------------
inline constexpr int kFusedRouteThreads = 256;


template <bool Nvfp4, int RowLoad>
__global__ __launch_bounds__(kFusedRouteThreads) void route_select_narrow_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ router, const float* __restrict__ bias,
    float* __restrict__ logits, LiveTokens live, XQuantOutputs q, RouteOutputs out, std::int32_t* tile_arrivals) {
    extern __shared__ __align__(16) unsigned char staged[];
    const int tokens       = live.tokens;
    const int block        = static_cast<int>(blockIdx.x);
    const int thread       = static_cast<int>(threadIdx.x);
    const int token_tiles  = (tokens + kStagedTileTokens - 1) / kStagedTileTokens;
    const int router_tiles = (kExperts / 8) * token_tiles;
    // Router rows before the dependency wait (weights only; every read of x / state and every write after it).
    if (block < router_tiles) { stage_router_rows<kFusedRouteThreads>(router, block % (kExperts / 8), staged); }
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    if (block >= router_tiles) {
        const int row = (block - router_tiles) * (kFusedRouteThreads / 32) + (thread >> 5);
        if constexpr (Nvfp4) {
            quant_nvfp4_row<RowLoad>(x, tokens, kHidden, nullptr, live, q.codes, q.scales, q.amax, row);
        } else {
            quant_mxfp8_row<RowLoad>(x, tokens, kHidden, nullptr, live, nullptr, nullptr, q.dequantised, row);
        }
        return;
    }
    const int m_tile = block / (kExperts / 8);
    router_logits_staged_tile<kFusedRouteThreads>(x, router, logits, tokens, nullptr, 0, block % (kExperts / 8),
                                                  m_tile, staged, true);
    __shared__ int role;
    __threadfence();
    __syncthreads();
    if (thread == 0) { role = atomicAdd(tile_arrivals + m_tile, 1) == kExperts / 8 - 1; }
    __syncthreads();
    if (role == 0) { return; }
    __threadfence();
    if (thread == 0) { tile_arrivals[m_tile] = 0; } // no other CTA touches it again in this call
    static_assert(kFusedRouteThreads / 32 == kStagedTileTokens);
    if (token_tiles == 1) {
        // One token tile (T <= 8): this selector is the only one, so it also places, from its own shared
        // memory: ranks from shared-memory counters, no ticket, no read-back of counts / ids / ranks through
        // L2 (4 dependent L2 round trips less). Ranks only choose a row's position inside its expert, never
        // its arithmetic; the state words stay zero (never touched).
        __shared__ int local_counts[kExperts];
        __shared__ int local_ids[kStagedTileTokens * kTopK];
        __shared__ int local_ranks[kStagedTileTokens * kTopK];
        if (thread < kExperts) { local_counts[thread] = 0; }
        __syncthreads();
        RouteOutputs local = out;
        local.counts    = local_counts;
        local.ranks     = local_ranks;
        local.local_ids = local_ids;
        select_token(logits, bias, live, local, thread >> 5, thread & 31);
        __syncthreads();
        place_rows<kFusedRouteThreads, true>(local, tokens, thread);
        return;
    }
    select_token(logits, bias, live, out, m_tile * kStagedTileTokens + (thread >> 5), thread & 31);
    __threadfence();
    __syncthreads();
    if (thread == 0) { role = atomicAdd(out.ticket, 1) == token_tiles - 1; }
    __syncthreads();
    if (role == 0) { return; }
    __threadfence();
    place_rows<kFusedRouteThreads>(out, tokens, thread);
    __syncthreads();
    if (thread < kExperts) { out.counts[thread] = 0; }
    if (thread == 0) { *out.ticket = 0; }
}

// ---------------------------------------------------------------------------------------------
// Standalone SiTU (sub-op test entry): a thread per output column.
// ---------------------------------------------------------------------------------------------
__global__ __launch_bounds__(256) void softcap_swiglu_kernel(const __nv_bfloat16* __restrict__ gate_up,
                                                             const float* __restrict__ gamma,
                                                             std::int32_t rows,
                                                             __nv_bfloat16* __restrict__ y) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::int64_t>(rows) * kIntermediate) { return; }
    const std::int64_t row = index / kIntermediate;
    const int column       = static_cast<int>(index - row * kIntermediate);
    const std::uint32_t pair = __ldcg(reinterpret_cast<const unsigned int*>(gate_up + row * kGateUpRows) + column);
    y[index] = situ(__uint_as_float(pair << 16), __uint_as_float(pair & 0xFFFF0000U), __ldcg(gamma + row));
}

// ---------------------------------------------------------------------------------------------
// Combine: out[t][c] = bf16(((0 + d0) + d1) ... + d7), slots in selection order.
// ---------------------------------------------------------------------------------------------
__global__ __launch_bounds__(128) void combine_kernel(const __nv_bfloat16* __restrict__ down,
                                                      const std::int32_t* __restrict__ ids,
                                                      LiveTokens live,
                                                      __nv_bfloat16* __restrict__ out) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    const int token  = static_cast<int>(blockIdx.x);
    const int column = (static_cast<int>(blockIdx.y) * 128 + static_cast<int>(threadIdx.x)) * 2;
    float acc0       = 0.0F;
    float acc1       = 0.0F;
    if (token < live_count(live)) {
#pragma unroll
        for (int slot = 0; slot < kTopK; ++slot) {
            const int index = token * kTopK + slot;
            if (__ldcg(ids + index) >= 0) {
                const std::uint32_t pair = __ldcg(reinterpret_cast<const unsigned int*>(
                    down + static_cast<std::int64_t>(index) * kHidden + column));
                acc0 = __fadd_rn(acc0, __uint_as_float(pair << 16));
                acc1 = __fadd_rn(acc1, __uint_as_float(pair & 0xFFFF0000U));
            }
        }
    }
    *reinterpret_cast<__nv_bfloat162*>(out + static_cast<std::int64_t>(token) * kHidden + column) =
        __floats2bfloat162_rn(acc0, acc1);
}

// ---------------------------------------------------------------------------------------------
// LoRA shrink: a warp per (slot, bank); 16 ranks per pass; fixed per-lane k order and a fixed
// butterfly, so a row's low vector never depends on the round.
// ---------------------------------------------------------------------------------------------
__global__ __launch_bounds__(256) void lora_shrink_kernel(const __nv_bfloat16* __restrict__ source,
                                                          std::int32_t source_k, bool gate_up,
                                                          const std::int32_t* __restrict__ ids,
                                                          LiveTokens live, LoraRound lora) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int banks = gate_up ? 2 : 1;
    const int task  = static_cast<int>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5);
    const int index = task / banks;
    const int bank_index = task - index * banks;
    if (index >= live.tokens * kTopK) { return; }
    const int token  = index / kTopK;
    const int expert = __ldcg(ids + index);
    if (expert < 0 || token >= live_count(live)) { return; }
    const int slot = lora_slot_of(lora, token);
    if (slot < 0) { return; }
    const LoraBank& bank = lora_bank(lora, expert, gate_up ? bank_index : 2);
    const __nv_bfloat16* x =
        source + static_cast<std::int64_t>(gate_up ? token : index) * source_k;
    const auto* a = static_cast<const __nv_bfloat16*>(bank.a) + static_cast<std::int64_t>(slot) * bank.a_stride;
    float* low    = lora.low + static_cast<std::int64_t>(index) * lora.low_stride + bank_index * bank.rank;
    for (int r0 = 0; r0 < bank.rank; r0 += 16) {
        float acc[16];
#pragma unroll
        for (int r = 0; r < 16; ++r) { acc[r] = 0.0F; }
        for (int k = lane * 8; k < source_k; k += 256) {
            const uint4 xv = __ldcg(reinterpret_cast<const uint4*>(x + k)); // x / y rows through L2 (PDL audit)
            const std::uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                if (r0 + r < bank.rank) {
                    const uint4 av = *reinterpret_cast<const uint4*>(
                        a + static_cast<std::int64_t>(r0 + r) * source_k + k);
                    const std::uint32_t aw[4] = {av.x, av.y, av.z, av.w};
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        acc[r] = __fmaf_rn(__uint_as_float(aw[i] << 16), __uint_as_float(xw[i] << 16), acc[r]);
                        acc[r] = __fmaf_rn(__uint_as_float(aw[i] & 0xFFFF0000U),
                                           __uint_as_float(xw[i] & 0xFFFF0000U), acc[r]);
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            const float total = warp_sum(acc[r]);
            if (lane == 0 && r0 + r < bank.rank) { low[r0 + r] = total; }
        }
    }
}

// ---- Unit-weight decode for the trainer's dX (api/ops/inkling_moe.h `inkling_decode_expert_units`) ----------
/// e2m1 magnitude code m (3 bits) -> fp32 value: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (exact).
__device__ __forceinline__ float e2m1_value(std::uint32_t code) {
    const std::uint32_t m = code & 7U, e = m >> 1, mant = m & 1U;
    const std::uint32_t magnitude = e == 0U ? (mant != 0U ? 0x3F000000U : 0U) : (((e + 126U) << 23) | (mant << 22));
    return __uint_as_float(magnitude | ((code & 8U) << 28));
}

/// One thread per 16-column block of one stacked weight row: its 8 code bytes (low nibble = even
/// column) and its e4m3 scale in `blockscale-k16-m128x4-v1` (128-row x 4-group tiles of 512 bytes).
/// unit = bf16(e2m1 * scale): the product has <= 6 significant bits, so both roundings are exact.
__global__ __launch_bounds__(256) void decode_nvfp4_units_kernel(const std::uint8_t* __restrict__ codes,
                                                                 const std::uint8_t* __restrict__ scales,
                                                                 std::int32_t k, std::int64_t first_row,
                                                                 std::int64_t rows, __nv_bfloat16* __restrict__ unit) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    const std::int64_t groups = k / 16;
    const std::int64_t task   = static_cast<std::int64_t>(blockIdx.x) * 256 + threadIdx.x;
    if (task >= rows * groups) { return; }
    const std::int64_t local = task / groups;
    const std::int64_t group = task - local * groups;
    const std::int64_t row   = first_row + local;
    const std::int64_t at    = ((row / 128) * (groups / 4) + group / 4) * 512 + (row % 32) * 16 +
                            ((row % 128) / 32) * 4 + group % 4;
    const float scale        = e4m3_to_f32(scales[at]);
    const uint2 packed       = *reinterpret_cast<const uint2*>(codes + row * (k / 2) + group * 8);
    const std::uint32_t words[2] = {packed.x, packed.y};
    std::uint32_t out[8];
#pragma unroll
    for (int w = 0; w < 2; ++w) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const std::uint32_t byte = (words[w] >> (8 * b)) & 0xFFU;
            const __nv_bfloat162 pair = __floats2bfloat162_rn(__fmul_rn(e2m1_value(byte & 15U), scale),
                                                              __fmul_rn(e2m1_value(byte >> 4), scale));
            out[4 * w + b] = *reinterpret_cast<const std::uint32_t*>(&pair);
        }
    }
    auto* destination = reinterpret_cast<uint4*>(unit + local * k + group * 16);
    destination[0]    = make_uint4(out[0], out[1], out[2], out[3]);
    destination[1]    = make_uint4(out[4], out[5], out[6], out[7]);
}

/// One thread per 16 columns of one stacked MXFP8 weight row: bf16(e4m3 * 2^(e - 127)) exactly as the
/// GEMM's in-register dequantisation (`mxfp8_dequant_pair`).
__global__ __launch_bounds__(256) void decode_mxfp8_units_kernel(const std::uint8_t* __restrict__ codes,
                                                                 const std::uint8_t* __restrict__ exponents,
                                                                 std::int32_t k, std::int64_t first_row,
                                                                 std::int64_t rows, __nv_bfloat16* __restrict__ unit) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    const std::int64_t chunks = k / 16;
    const std::int64_t task   = static_cast<std::int64_t>(blockIdx.x) * 256 + threadIdx.x;
    if (task >= rows * chunks) { return; }
    const std::int64_t local = task / chunks;
    const std::int64_t chunk = task - local * chunks;
    const std::int64_t row   = first_row + local;
    const float scale        = e8m0_to_f32(exponents[row * (k / 32) + chunk / 2]);
    const uint4 packed       = *reinterpret_cast<const uint4*>(codes + row * k + chunk * 16);
    const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    std::uint32_t out[8];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const __nv_bfloat162 pair = mxfp8_dequant_pair(static_cast<std::uint16_t>(words[w] >> (16 * h)), scale);
            out[2 * w + h] = *reinterpret_cast<const std::uint32_t*>(&pair);
        }
    }
    auto* destination = reinterpret_cast<uint4*>(unit + local * k + chunk * 16);
    destination[0]    = make_uint4(out[0], out[1], out[2], out[3]);
    destination[1]    = make_uint4(out[4], out[5], out[6], out[7]);
}

/// factor[i] = amax[first + i] / den (NVFP4, div_rn: `inkling_ref.nvfp4_global`) or 1 (MXFP8).
__global__ __launch_bounds__(128) void expert_factor_kernel(const float* __restrict__ amax,
                                                            const float* __restrict__ den, std::int32_t first,
                                                            std::int32_t count, float* __restrict__ factor) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    const int i = static_cast<int>(blockIdx.x) * 128 + static_cast<int>(threadIdx.x);
    if (i >= count) { return; }
    factor[i] = amax != nullptr ? __fdiv_rn(amax[first + i], *den) : 1.0F;
}

template <class... KernelArgs, class... CallArgs>
void launch(const char* what, dim3 grid, dim3 block, cudaStream_t stream,
            void (*kernel)(KernelArgs...), CallArgs&&... args) {
    if (grid.x == 0 || grid.y == 0) { return; }
    const cudaError_t status = pdl::launch_dependent({grid, block, 0, stream}, kernel,
                                                     std::forward<CallArgs>(args)...);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("inkling_moe: ") + what + ": " + cudaGetErrorString(status));
    }
    check_launch(what);
}

} // namespace

template <int Experts>
void launch_router_tiled(const __nv_bfloat16* x, const __nv_bfloat16* router, float* logits,
                         std::int32_t tokens, std::int32_t* zero, std::int32_t zero_words,
                         cudaStream_t stream) {
    constexpr std::size_t kShared = sizeof(RouterTiledShared<Experts>);
    auto* kernel = router_logits_tiled_kernel<Experts>;
    cudaError_t status = ::sinfer::ops::set_func_attribute_per_device(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kShared));
    if (status == cudaSuccess) {
        const dim3 grid(static_cast<unsigned>((tokens + kRouterTileTokens - 1) / kRouterTileTokens),
                        kExperts / Experts);
        status = pdl::launch_dependent({grid, dim3(256), kShared, stream}, kernel, x, router, logits, tokens,
                                       zero, zero_words);
    }
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("inkling_moe: router logits (tiled): ") + cudaGetErrorString(status));
    }
}

/// Rounds above this width take the tiled router kernel (same bits; a speed choice).
inline constexpr std::int32_t kRouterTiledMinTokens = 65;

void launch_router_logits(const __nv_bfloat16* x, const __nv_bfloat16* router, float* logits,
                          std::int32_t tokens, std::int32_t* zero, std::int32_t zero_words,
                          cudaStream_t stream) {
    if (tokens >= 2048) {
        launch_router_tiled<128>(x, router, logits, tokens, zero, zero_words, stream);
        return;
    }
    if (tokens >= kRouterTiledMinTokens) {
        launch_router_tiled<32>(x, router, logits, tokens, zero, zero_words, stream);
        return;
    }
    launch("router logits", dim3(kExperts / 8, static_cast<unsigned>((tokens + 15) / 16)), dim3(96),
           stream, router_logits_kernel, x, router, logits, tokens, zero, zero_words);
}

void launch_router_select(const float* logits, const float* bias, LiveTokens live,
                          const RouteOutputs& out, cudaStream_t stream) {
    launch("router select", dim3(static_cast<unsigned>((live.tokens + 3) / 4)), dim3(128), stream,
           router_select_kernel, logits, bias, live, out);
}

void launch_route_quant_narrow(const __nv_bfloat16* x, const __nv_bfloat16* router, float* logits,
                               std::int32_t tokens, std::int32_t* zero, std::int32_t zero_words, LiveTokens live,
                               bool nvfp4, std::uint8_t* x_codes, std::uint8_t* x_scales, float* x_amax,
                               __nv_bfloat16* x_dequantised, cudaStream_t stream) {
    XQuantOutputs q;
    q.codes       = x_codes;
    q.scales      = x_scales;
    q.amax        = x_amax;
    q.dequantised = x_dequantised;
    const int blocks = (kExperts / 8) * ((tokens + kStagedTileTokens - 1) / kStagedTileTokens) + (tokens + 2) / 3;
    const int mode = row_load_mode_host();
    auto* kernel   = nvfp4 ? (mode == 1 ? route_quant_narrow_kernel<true, 1>
                                        : mode == 2 ? route_quant_narrow_kernel<true, 2> : route_quant_narrow_kernel<true, 0>)
                           : (mode == 1 ? route_quant_narrow_kernel<false, 1>
                                        : mode == 2 ? route_quant_narrow_kernel<false, 2> : route_quant_narrow_kernel<false, 0>);
    cudaError_t status = ::sinfer::ops::set_func_attribute_per_device(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kStagedSharedBytes);
    if (status == cudaSuccess) {
        status = pdl::launch_dependent({dim3(static_cast<unsigned>(blocks)), dim3(96),
                                        static_cast<std::size_t>(kStagedSharedBytes), stream},
                                       kernel, x, router, logits, tokens, zero, zero_words, live, q);
    }
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("inkling_moe: route + x quant (narrow): ") + cudaGetErrorString(status));
    }
}

void launch_route_select_narrow(const __nv_bfloat16* x, const __nv_bfloat16* router, const float* bias,
                                float* logits, LiveTokens live, bool nvfp4, std::uint8_t* x_codes,
                                std::uint8_t* x_scales, float* x_amax, __nv_bfloat16* x_dequantised,
                                const RouteOutputs& out, std::int32_t* tile_arrivals, cudaStream_t stream) {
    if (live.tokens > kNarrowStateMaxTokens || out.counts == nullptr || out.ticket == nullptr ||
        !out.last_cta_places) {
        throw std::invalid_argument("inkling_moe: the fused narrow route needs T <= 64, counters and last-CTA placement");
    }
    XQuantOutputs q;
    q.codes       = x_codes;
    q.scales      = x_scales;
    q.amax        = x_amax;
    q.dequantised = x_dequantised;
    const int token_tiles = (live.tokens + kStagedTileTokens - 1) / kStagedTileTokens;
    const int blocks      = (kExperts / 8) * token_tiles + (live.tokens + kFusedRouteThreads / 32 - 1) / (kFusedRouteThreads / 32);
    const int mode = row_load_mode_host();
    auto* kernel   = nvfp4 ? (mode == 1 ? route_select_narrow_kernel<true, 1>
                                        : mode == 2 ? route_select_narrow_kernel<true, 2> : route_select_narrow_kernel<true, 0>)
                           : (mode == 1 ? route_select_narrow_kernel<false, 1>
                                        : mode == 2 ? route_select_narrow_kernel<false, 2> : route_select_narrow_kernel<false, 0>);
    cudaError_t status = ::sinfer::ops::set_func_attribute_per_device(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kStagedSharedBytes);
    if (status == cudaSuccess) {
        status = pdl::launch_dependent({dim3(static_cast<unsigned>(blocks)), dim3(kFusedRouteThreads),
                                        static_cast<std::size_t>(kStagedSharedBytes), stream},
                                       kernel, x, router, bias, logits, live, q, out, tile_arrivals);
    }
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("inkling_moe: route + select + x quant (narrow): ") +
                                 cudaGetErrorString(status));
    }
}

void launch_router_place(const RouteOutputs& out, std::int32_t tokens, cudaStream_t stream) {
    const std::int32_t slots = tokens * kTopK;
    launch("router place", dim3(static_cast<unsigned>((slots + 255) / 256)), dim3(256), stream,
           router_place_kernel, out, slots);
}

void launch_quant_nvfp4(const __nv_bfloat16* input, std::int32_t rows, std::int32_t k,
                        const std::int32_t* row_ids, LiveTokens live, std::uint8_t* codes,
                        std::uint8_t* scales, float* amax, cudaStream_t stream) {
    if (k % 16 != 0) { throw std::invalid_argument("inkling_quant_nvfp4_4o6: k must be a multiple of 16"); }
    const int mode = row_load_mode_host();
    launch("quant nvfp4", dim3(static_cast<unsigned>((rows + 7) / 8)), dim3(256), stream,
           mode == 1 ? quant_nvfp4_kernel<1> : mode == 2 ? quant_nvfp4_kernel<2> : quant_nvfp4_kernel<0>, input, rows, k,
           row_ids, live, codes, scales, amax);
}

void launch_quant_mxfp8(const __nv_bfloat16* input, std::int32_t rows, std::int32_t k,
                        const std::int32_t* row_ids, LiveTokens live, std::uint8_t* codes,
                        std::uint8_t* exponents, __nv_bfloat16* dequantised, cudaStream_t stream) {
    if (k % 32 != 0) { throw std::invalid_argument("inkling_quant_mxfp8: k must be a multiple of 32"); }
    const int mode = row_load_mode_host();
    launch("quant mxfp8", dim3(static_cast<unsigned>((rows + 7) / 8)), dim3(256), stream,
           mode == 1 ? quant_mxfp8_kernel<1> : mode == 2 ? quant_mxfp8_kernel<2> : quant_mxfp8_kernel<0>, input, rows, k,
           row_ids, live, codes, exponents, dequantised);
}

void launch_softcap_swiglu(const __nv_bfloat16* gate_up, const float* gamma, std::int32_t rows,
                           __nv_bfloat16* y, cudaStream_t stream) {
    const std::int64_t total = static_cast<std::int64_t>(rows) * kIntermediate;
    launch("softcap swiglu", dim3(static_cast<unsigned>((total + 255) / 256)), dim3(256), stream,
           softcap_swiglu_kernel, gate_up, gamma, rows, y);
}

void launch_combine(const __nv_bfloat16* down, const std::int32_t* ids, LiveTokens live,
                    __nv_bfloat16* out, cudaStream_t stream) {
    launch("combine", dim3(static_cast<unsigned>(live.tokens), kHidden / 256), dim3(128), stream,
           combine_kernel, down, ids, live, out);
}

void launch_lora_shrink(const __nv_bfloat16* source, std::int32_t source_k, bool gate_up,
                        const std::int32_t* ids, LiveTokens live, const LoraRound& lora,
                        cudaStream_t stream) {
    const std::int64_t tasks = static_cast<std::int64_t>(live.tokens) * kTopK * (gate_up ? 2 : 1);
    launch("lora shrink", dim3(static_cast<unsigned>((tasks + 7) / 8)), dim3(256), stream,
           lora_shrink_kernel, source, source_k, gate_up, ids, live, lora);
}

void launch_decode_expert_units(const InklingExpertBank& bank, bool nvfp4, std::int32_t k, std::int32_t first,
                                std::int32_t count, __nv_bfloat16* unit, float* factor, cudaStream_t stream) {
    const std::int64_t first_row = static_cast<std::int64_t>(first) * kGateUpRows;
    const std::int64_t rows      = static_cast<std::int64_t>(count) * kGateUpRows;
    const std::int64_t tasks     = rows * (k / 16);
    const dim3 grid(static_cast<unsigned>((tasks + 255) / 256));
    if (nvfp4) {
        launch("decode nvfp4 units", grid, dim3(256), stream, decode_nvfp4_units_kernel, bank.codes, bank.scales, k,
               first_row, rows, unit);
    } else {
        launch("decode mxfp8 units", grid, dim3(256), stream, decode_mxfp8_units_kernel, bank.codes, bank.scales, k,
               first_row, rows, unit);
    }
    if (factor != nullptr) {
        launch("expert factor", dim3(static_cast<unsigned>((count + 127) / 128)), dim3(128), stream,
               expert_factor_kernel, nvfp4 ? bank.amax : nullptr, nvfp4 ? bank.outer_den : nullptr, first, count,
               factor);
    }
}

namespace {

/// Benchmark predecessor (inkling_bench_predecessor): triggers its dependents at entry, then spins on the
/// global timer (the wait loop of sparse_moe_fault_warp_skew, ops/sparse_moe/prefill/sparse_moe_prefill_body.inc).
__global__ __launch_bounds__(32) void bench_predecessor_kernel(std::int32_t nanoseconds) {
    pdl::trigger_dependents();
    if (nanoseconds <= 0) { return; }
    unsigned long long start = 0;
    unsigned long long now   = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start)::"memory");
    do {
        __nanosleep(200);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)::"memory");
    } while (now - start < static_cast<unsigned long long>(nanoseconds));
}

} // namespace

/// Test-only "planter" (pdl-chain negative control): every CTA reads the whole buffer through the non-coherent path,
/// so each SM's L1 / texture cache holds the buffer's CURRENT (soon stale) lines when the writer overwrites it.
__global__ __launch_bounds__(256) void test_plant_reads_kernel(const uint4* buffer, std::int64_t words, int* sink) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    std::uint32_t acc = 0;
    for (std::int64_t i = threadIdx.x; i < words; i += 256) {
        const uint4 v = __ldg(buffer + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9E3779B9U) { sink[0] = 1; } // keeps the loads; never true in practice (a race-free write anyway)
}

void launch_test_plant_reads(const void* buffer, std::int64_t bytes, int* sink, cudaStream_t stream) {
    int device = 0, sms = 1;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    launch("test plant reads", dim3(static_cast<unsigned>(2 * sms)), dim3(256), stream, test_plant_reads_kernel,
           static_cast<const uint4*>(buffer), bytes / 16, sink);
}

/// Test-only PDL writer (pdl-chain regression): waits for its predecessor (the previous MoE call may still read
/// `dst`), lets its dependents launch at once (as the engine's kernels do), then copies `words` 16-byte words.
/// `hostile` (negative control): 2 CTAs per SM; after the trigger -- the dependent's CTAs are being launched -- every
/// CTA first reads ALL of the old `dst` through ld.global.nc, planting old lines in the non-coherent cache of every
/// SM the dependent may run on, and only then writes its slice.
__global__ __launch_bounds__(256) void test_copy_rows_kernel(const uint4* source, uint4* destination, std::int64_t words,
                                                             int hostile, int* sink) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    if (hostile != 0) {
        std::uint32_t acc = 0;
        for (std::int64_t i = threadIdx.x; i < words; i += 256) {
            const uint4 v = __ldg(destination + i);
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
        if (acc == 0x9E3779B9U) { sink[0] = 1; }
        __syncthreads();
    }
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * 256 + threadIdx.x; i < words;
         i += static_cast<std::int64_t>(gridDim.x) * 256) {
        destination[i] = __ldcg(source + i);
    }
}

void launch_test_copy_rows(const void* source, void* destination, std::int64_t bytes, int hostile, int* sink,
                           cudaStream_t stream) {
    const std::int64_t words = bytes / 16;
    int blocks               = static_cast<int>(std::min<std::int64_t>((words + 255) / 256, 1024));
    if (hostile != 0) {
        int device = 0, sms = 1;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
        blocks = 2 * sms;
    }
    launch("test copy rows", dim3(static_cast<unsigned>(blocks)), dim3(256), stream, test_copy_rows_kernel,
           static_cast<const uint4*>(source), static_cast<uint4*>(destination), words, hostile, sink);
}

void set_read_only_control(int mode) { row_load_mode_host() = mode; }

void launch_bench_predecessor(std::int32_t nanoseconds, cudaStream_t stream) {
    launch("bench predecessor", dim3(1), dim3(32), stream, bench_predecessor_kernel, nanoseconds);
}

} // namespace sinfer::ops::detail::inkling
