// Inkling-2-Mini attention LoRA delta (S10): per-token, fixed-order arithmetic (api/ops/inkling_mix.h
// inkling_lora_delta). Compiled with -fmad=false: every fused multiply-add is an explicit fmaf.

#include "api/ops/inkling_mix.h"
#include "api/ops/lora.h"

#include "core/device.h" // CUDA_CHECK
#include "core/pdl.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace sinfer::ops {
namespace {

constexpr int kMaxRank = 64;
// Shrink tiles (ranks per block = warps, tokens per warp): narrow rounds take the small tile (more blocks), wider rounds the
// large one (A rows and x columns staged once for more work). The arithmetic of every (token, rank) value is the same in both.
constexpr int kNarrowWarps = 4, kNarrowTokens = 4;
constexpr int kWideWarps = 8, kWideTokens = 8;
constexpr int kWideMinTokens = 16; // rounds of at least this many tokens take the wide tile
constexpr int kStageK = 1152; // shrink: k elements per shared-memory stage (<= 36 KiB a block)
constexpr int kExpandThreads = 128; // a thread per output row (rows of all slices, flattened)
constexpr int kExpandTokens  = 4;   // tokens per expand thread: interleaved chains sharing the B row held in registers

struct LoraDeltaArgs {
    const LoraBank* table          = nullptr;
    InklingLoraSlice slices[kInklingLoraMaxSlices]{};
    std::int32_t row_end[kInklingLoraMaxSlices]{}; // slice s owns the flattened rows [row_end[s - 1], row_end[s])
    std::int32_t slice_count       = 0;
    std::int32_t rank              = 0;
    std::int32_t tokens            = 0;
    const std::int32_t* token_slot = nullptr;
    const std::int32_t* uniform    = nullptr;
    float* low                     = nullptr; // [tokens][slices][rank]
};

__device__ __forceinline__ int slot_of(const LoraDeltaArgs& a, int t) {
    if (a.token_slot != nullptr) { return a.token_slot[t]; }
    return a.uniform != nullptr ? *a.uniform : -1;
}

__device__ __forceinline__ float bf16_lo(std::uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(std::uint32_t w) { return __uint_as_float(w & 0xFFFF0000U); }

// The N per-token chains of one lane of a shrink warp: acc[i] = fma chain over k = lane, lane + 32, ... (ascending) of
// row[k] * x_i[k], from 0 -- the same chain per (token, rank, lane) whatever N is; tokens with live[i] false are skipped.
template <int N>
__device__ __forceinline__ void shrink_chains(const __nv_bfloat16* row, const __nv_bfloat16* const (&x)[N],
                                              const bool (&live)[N], int k_total, int lane, float (&acc)[N]) {
#pragma unroll 4
    for (int k = lane; k < k_total; k += 32) {
        const float av = __bfloat162float(__ldca(row + k)); // adapter weights: a global, cached load
#pragma unroll
        for (int i = 0; i < N; ++i) {
            if (live[i]) { acc[i] = fmaf(av, __bfloat162float(x[i][k]), acc[i]); }
        }
    }
}

__device__ __forceinline__ std::uint32_t smem_u32(const void* p) {
    return static_cast<std::uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15U) == 0U; }

// grid (rank blocks, slices, token blocks): low[t][s][r] = sum_k A[r, k] * x[k, t]; lane l chains k = l, l + 32, ... from 0,
// then a fixed xor tree over the 32 lane sums. A warp owns one rank r and up to kShrinkTokens tokens. When the block's live
// tokens share one slot (the usual round) and the rows are 16-byte aligned, the block first copies its A rows and x columns
// into shared memory with 16-byte cp.async (every load in flight at once) and the chains read them from there; otherwise
// (mixed slots, odd shapes) each warp reads global memory directly. Each (token, rank) value is the same fp32 chain and tree
// in every case and as a warp computing that token alone, so the result is independent of the batch and of this tiling.
template <int kShrinkWarps, int kShrinkTokens>
__global__ void __launch_bounds__(kShrinkWarps * 32) lora_shrink_kernel(const __grid_constant__ LoraDeltaArgs a) {
    extern __shared__ uint4 stage[]; // [kShrinkWarps][kStageK] A rows, then [kStageK][kShrinkTokens] x (bf16)
    pdl::wait_for_dependencies();
    pdl::trigger_dependents(); // PDL: the next kernel may launch now (it waits for this grid before any read)
    const int r0 = static_cast<int>(blockIdx.x) * kShrinkWarps;
    const int r  = r0 + static_cast<int>(threadIdx.x >> 5);
    const int s  = static_cast<int>(blockIdx.y);
    const int t0 = static_cast<int>(blockIdx.z) * kShrinkTokens;
    const InklingLoraSlice& sl = a.slices[s];
    const int lane             = static_cast<int>(threadIdx.x & 31);
    int slot[kShrinkTokens];
    bool live[kShrinkTokens];
    int common = -1;
    bool mixed = false;
#pragma unroll
    for (int i = 0; i < kShrinkTokens; ++i) {
        const int t = t0 + i;
        slot[i]     = t < a.tokens ? slot_of(a, t) : -1;
        live[i]     = slot[i] >= 0;
        if (live[i]) {
            mixed  = mixed || (common >= 0 && slot[i] != common);
            common = common < 0 ? slot[i] : common;
        }
    }
    if (common < 0) { return; } // no adapted token in this block (block-uniform)
    const LoraBank& bank = a.table[sl.bank];
    const int K          = bank.k;
    const auto* A        = static_cast<const __nv_bfloat16*>(bank.a);
    const auto* X        = static_cast<const __nv_bfloat16*>(sl.x);
    float acc[kShrinkTokens];
#pragma unroll
    for (int i = 0; i < kShrinkTokens; ++i) { acc[i] = 0.0F; }
    const __nv_bfloat16* rows = A + static_cast<std::int64_t>(mixed ? 0 : common) * bank.a_stride + static_cast<std::int64_t>(r0) * K;
    const bool staged = !mixed && (K & 7) == 0 && (sl.x_stride & 7) == 0 && aligned16(rows) &&
                        aligned16(X); // block-uniform
    if (staged) {
        static_assert(kShrinkTokens == 4 || kShrinkTokens == 8, "x columns are staged as 8- or 16-byte k rows");
        static_assert(kStageK % 32 == 0, "a stage keeps every lane's k order (and whole 16-byte chunks)");
        using XRow  = std::conditional_t<kShrinkTokens == 8, uint4, uint2>; // the tile's x at one k, token-minor
        auto* sa    = reinterpret_cast<__nv_bfloat16*>(stage);                // [kShrinkWarps][kStageK]
        auto* sx    = reinterpret_cast<XRow*>(sa + kShrinkWarps * kStageK);   // [kStageK] x the tile's tokens
        const int nr = min(kShrinkWarps, a.rank - r0);
        const __nv_bfloat16* row = sa + (r - r0) * kStageK;
        // K in stages of kStageK (a multiple of 32: lane l still owns k = l, l + 32, ... in ascending order)
        for (int k0 = 0; k0 < K; k0 += kStageK) {
            const int len    = min(kStageK, K - k0);
            const int chunks = len >> 3; // 16-byte chunks per row
            // A rows: 16-byte cp.async, all in flight at once
            for (int i = static_cast<int>(threadIdx.x); i < nr * chunks; i += kShrinkWarps * 32) {
                const int rr = i / chunks;
                const int c  = (i - rr * chunks) << 3;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :
                             : "r"(smem_u32(sa + rr * kStageK + c)), "l"(rows + static_cast<std::int64_t>(rr) * K + k0 + c)
                             : "memory");
            }
            asm volatile("cp.async.commit_group;\n" : : : "memory");
            // x columns, transposed to [k][token] so one shared load gives a k's value for every token of the tile:
            // 8 consecutive k of each token (one 16-byte load each, after the wait, via L2), regrouped by k in registers
            for (int c = static_cast<int>(threadIdx.x); c < chunks; c += kShrinkWarps * 32) {
                std::uint32_t v[kShrinkTokens][4];
#pragma unroll
                for (int i = 0; i < kShrinkTokens; ++i) {
                    const uint4 w = live[i] ? __ldcg(reinterpret_cast<const uint4*>(
                                                  X + static_cast<std::int64_t>(t0 + i) * sl.x_stride + k0 + 8 * c))
                                            : make_uint4(0U, 0U, 0U, 0U);
                    v[i][0] = w.x; v[i][1] = w.y; v[i][2] = w.z; v[i][3] = w.w;
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const unsigned sel = (e & 1) != 0 ? 0x7632U : 0x5410U; // the high or the low halves of two words
                    std::uint32_t q[kShrinkTokens / 2];
#pragma unroll
                    for (int h = 0; h < kShrinkTokens / 2; ++h) {
                        q[h] = __byte_perm(v[2 * h][e >> 1], v[2 * h + 1][e >> 1], sel);
                    }
                    if constexpr (kShrinkTokens == 8) {
                        sx[8 * c + e] = make_uint4(q[0], q[1], q[2], q[3]);
                    } else {
                        sx[8 * c + e] = make_uint2(q[0], q[1]);
                    }
                }
            }
            asm volatile("cp.async.wait_group 0;\n" : : : "memory");
            __syncthreads();
            if (r < a.rank) {
                // every tile token's chain runs (a token without an adapter reads zeros and is never written)
#pragma unroll 4
                for (int k = lane; k < len; k += 32) {
                    const float av = __bfloat162float(row[k]);
                    const XRow xv  = sx[k];
                    std::uint32_t xw[kShrinkTokens / 2];
                    if constexpr (kShrinkTokens == 8) {
                        xw[0] = xv.x; xw[1] = xv.y; xw[2] = xv.z; xw[3] = xv.w;
                    } else {
                        xw[0] = xv.x; xw[1] = xv.y;
                    }
#pragma unroll
                    for (int h = 0; h < kShrinkTokens / 2; ++h) {
                        acc[2 * h]     = fmaf(av, bf16_lo(xw[h]), acc[2 * h]);
                        acc[2 * h + 1] = fmaf(av, bf16_hi(xw[h]), acc[2 * h + 1]);
                    }
                }
            }
            __syncthreads(); // the stage is overwritten next
        }
        if (r >= a.rank) { return; }
    } else {
        if (r >= a.rank) { return; } // warp-uniform
        if (!mixed) {
            const __nv_bfloat16* x[kShrinkTokens];
#pragma unroll
            for (int i = 0; i < kShrinkTokens; ++i) {
                x[i] = X + static_cast<std::int64_t>(live[i] ? t0 + i : t0) * sl.x_stride;
            }
            shrink_chains<kShrinkTokens>(rows + static_cast<std::int64_t>(r - r0) * K, x, live, K, lane, acc);
        } else {
#pragma unroll
            for (int i = 0; i < kShrinkTokens; ++i) {
                float one[1] = {0.0F};
                if (live[i]) {
                    const __nv_bfloat16* x[1] = {X + static_cast<std::int64_t>(t0 + i) * sl.x_stride};
                    const bool on[1]          = {true};
                    const __nv_bfloat16* row =
                        A + static_cast<std::int64_t>(slot[i]) * bank.a_stride + static_cast<std::int64_t>(r) * K;
                    shrink_chains<1>(row, x, on, K, lane, one);
                }
                acc[i] = one[0];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < kShrinkTokens; ++i) {
        float v = acc[i];
        for (int m = 16; m >= 1; m >>= 1) { v += __shfl_xor_sync(0xffffffffU, v, m); }
        if (lane == 0 && live[i]) {
            a.low[(static_cast<std::int64_t>(t0 + i) * a.slice_count + s) * a.rank + r] = v;
        }
    }
}

// B's row j of `slot` into registers: B[j, 2q] in the low half of b[q], B[j, 2q + 1] in the high half (q < rank / 2 rounded
// up; the rest is left as is).
__device__ __forceinline__ void load_b_row(const LoraDeltaArgs& a, const LoraBank& bank, int slot, int j,
                                           std::uint32_t (&b)[kMaxRank / 2]) {
    const int rank = a.rank;
    const auto* B  = static_cast<const __nv_bfloat16*>(bank.b) + static_cast<std::int64_t>(slot) * bank.b_stride +
                    static_cast<std::int64_t>(j) * rank;
    if ((rank & 7) == 0 && aligned16(B)) {
#pragma unroll
        for (int c = 0; c < kMaxRank / 8; ++c) {
            if (c * 8 < rank) {
                const uint4 v = __ldca(reinterpret_cast<const uint4*>(B + c * 8)); // adapter weights
                b[4 * c]      = v.x;
                b[4 * c + 1]  = v.y;
                b[4 * c + 2]  = v.z;
                b[4 * c + 3]  = v.w;
            }
        }
    } else {
        const auto* B16 = reinterpret_cast<const unsigned short*>(B);
#pragma unroll
        for (int q = 0; q < kMaxRank / 2; ++q) {
            const std::uint32_t lo = 2 * q < rank ? __ldca(B16 + 2 * q) : 0U;
            const std::uint32_t hi = 2 * q + 1 < rank ? __ldca(B16 + 2 * q + 1) : 0U;
            b[q]                   = lo | (hi << 16);
        }
    }
}

__device__ __forceinline__ float b_at(const std::uint32_t (&b)[kMaxRank / 2], int r) {
    return (r & 1) != 0 ? bf16_hi(b[r >> 1]) : bf16_lo(b[r >> 1]);
}

// N delta chains over the same B row, interleaved (one independent chain per token): delta[i] = fma chain over r = 0..rank-1
// of B[j, r] * low_i[r], from 0. Tokens with live[i] false are skipped.
template <int N>
__device__ __forceinline__ void expand_chains(const std::uint32_t (&b)[kMaxRank / 2], int rank, const float* const (&low)[N],
                                              const bool (&live)[N], bool vec, float (&delta)[N]) {
    if (vec) {
#pragma unroll
        for (int c = 0; c < kMaxRank / 4; ++c) {
            if (c * 4 < rank) {
#pragma unroll
                for (int i = 0; i < N; ++i) {
                    if (live[i]) {
                        const float4 l = *reinterpret_cast<const float4*>(low[i] + c * 4);
                        delta[i]       = fmaf(bf16_lo(b[2 * c]), l.x, delta[i]);
                        delta[i]       = fmaf(bf16_hi(b[2 * c]), l.y, delta[i]);
                        delta[i]       = fmaf(bf16_lo(b[2 * c + 1]), l.z, delta[i]);
                        delta[i]       = fmaf(bf16_hi(b[2 * c + 1]), l.w, delta[i]);
                    }
                }
            }
        }
    } else {
#pragma unroll
        for (int r = 0; r < kMaxRank; ++r) {
            if (r < rank) {
#pragma unroll
                for (int i = 0; i < N; ++i) {
                    if (live[i]) { delta[i] = fmaf(b_at(b, r), low[i][r], delta[i]); }
                }
            }
        }
    }
}

// grid (row blocks over the slices' flattened rows, token blocks): delta[j] = sum_r B[j, r] * low[r] (fp32 chain from 0, r
// order); the output row becomes bf16_rn(out + delta) where delta != 0. A thread owns one output row j of one slice and up
// to kExpandTokens tokens, with B's row j in registers: when those tokens share one slot their chains run interleaved,
// otherwise token by token (reloading B when the slot changes). Every chain is the same whatever the grouping.
__global__ void __launch_bounds__(kExpandThreads) lora_expand_kernel(const __grid_constant__ LoraDeltaArgs a) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents(); // PDL: the next kernel may launch now (it waits for this grid before any read)
    const int g = static_cast<int>(blockIdx.x) * kExpandThreads + static_cast<int>(threadIdx.x);
    int s       = 0;
    while (s < a.slice_count && g >= a.row_end[s]) { ++s; }
    if (s == a.slice_count) { return; }
    const int j                = g - (s == 0 ? 0 : a.row_end[s - 1]);
    const InklingLoraSlice& sl = a.slices[s];
    const LoraBank& bank       = a.table[sl.bank];
    if (j >= sl.rows || j >= bank.n) { return; }
    const int rank = a.rank;
    const int t0   = static_cast<int>(blockIdx.y) * kExpandTokens;
    int slot[kExpandTokens];
    bool live[kExpandTokens];
    int common = -1;
    bool mixed = false;
#pragma unroll
    for (int i = 0; i < kExpandTokens; ++i) {
        const int t = t0 + i;
        slot[i]     = t < a.tokens ? slot_of(a, t) : -1;
        live[i]     = slot[i] >= 0;
        if (live[i]) {
            mixed  = mixed || (common >= 0 && slot[i] != common);
            common = common < 0 ? slot[i] : common;
        }
    }
    if (common < 0) { return; }
    // low rows are 16-byte aligned for every token when the workspace base is and rank % 4 == 0
    const bool vec = (rank & 3) == 0 && aligned16(a.low);
    std::uint32_t b[kMaxRank / 2];
    float delta[kExpandTokens];
#pragma unroll
    for (int i = 0; i < kExpandTokens; ++i) { delta[i] = 0.0F; }
    if (!mixed) {
        load_b_row(a, bank, common, j, b);
        const float* low[kExpandTokens];
#pragma unroll
        for (int i = 0; i < kExpandTokens; ++i) {
            low[i] = a.low + (static_cast<std::int64_t>(live[i] ? t0 + i : t0) * a.slice_count + s) * rank;
        }
        expand_chains<kExpandTokens>(b, rank, low, live, vec, delta);
    } else {
        int held = -1;
#pragma unroll
        for (int i = 0; i < kExpandTokens; ++i) {
            if (!live[i]) { continue; }
            if (slot[i] != held) {
                load_b_row(a, bank, slot[i], j, b);
                held = slot[i];
            }
            const float* low[1] = {a.low + (static_cast<std::int64_t>(t0 + i) * a.slice_count + s) * rank};
            const bool on[1]    = {true};
            float one[1]        = {0.0F};
            expand_chains<1>(b, rank, low, on, vec, one);
            delta[i] = one[0];
        }
    }
#pragma unroll
    for (int i = 0; i < kExpandTokens; ++i) {
        if (live[i] && delta[i] != 0.0F) {
            auto* out = static_cast<__nv_bfloat16*>(sl.out) + static_cast<std::int64_t>(t0 + i) * sl.out_stride + j;
            *out      = __float2bfloat16_rn(__bfloat162float(*out) + delta[i]);
        }
    }
}

void require(bool ok, const char* what) {
    if (!ok) { throw std::invalid_argument(std::string("inkling_lora_delta: ") + what); }
}

} // namespace

std::size_t inkling_lora_delta_workspace_bytes(std::int32_t slices, std::int32_t max_rank,
                                               std::int32_t tokens) noexcept {
    const std::size_t bytes = static_cast<std::size_t>(slices) * static_cast<std::size_t>(max_rank) *
                              static_cast<std::size_t>(tokens) * sizeof(float);
    return (bytes + 255U) & ~std::size_t{255};
}

void inkling_lora_delta(const void* table, const InklingLoraSlice* slices, std::int32_t slice_count,
                        std::int32_t max_rank, std::int32_t tokens, const std::int32_t* token_slot,
                        const std::int32_t* uniform_slot, void* workspace, std::size_t workspace_bytes,
                        cudaStream_t stream) {
    require(table != nullptr, "no bank table");
    require(slice_count >= 1 && slice_count <= kInklingLoraMaxSlices, "1..6 slices");
    require(max_rank >= 1 && max_rank <= kMaxRank, "max_rank in [1, 64]");
    require(tokens >= 0, "negative token count");
    require(token_slot != nullptr || uniform_slot != nullptr, "token slots or a uniform slot");
    if (tokens == 0) { return; }
    require(workspace != nullptr &&
                workspace_bytes >= inkling_lora_delta_workspace_bytes(slice_count, max_rank, tokens),
            "workspace too small");
    LoraDeltaArgs a;
    a.table       = static_cast<const LoraBank*>(table);
    a.slice_count = slice_count;
    a.rank        = max_rank;
    a.tokens      = tokens;
    a.token_slot  = token_slot;
    a.uniform     = uniform_slot;
    a.low         = static_cast<float*>(workspace);
    int total     = 0;
    for (int s = 0; s < slice_count; ++s) {
        const InklingLoraSlice& sl = slices[s];
        require(sl.x != nullptr && sl.out != nullptr && sl.bank >= 0, "slice without input / output / bank");
        require(sl.x_stride > 0 && sl.out_stride >= sl.rows && sl.rows > 0, "slice strides / rows");
        a.slices[s]  = sl;
        total       += static_cast<int>(sl.rows);
        a.row_end[s] = total; // the expand's row blocks cover every slice's rows once
    }
    const auto token_blocks = [&](int per) { return static_cast<unsigned>((tokens + per - 1) / per); };
    require(token_blocks(kNarrowTokens) <= 65535U, "too many tokens for one call");
    const auto shrink = [&](auto kernel, int warps, int per_warp) {
        const std::size_t bytes = static_cast<std::size_t>(warps + per_warp) * kStageK * sizeof(__nv_bfloat16);
        CUDA_CHECK(pdl::launch_dependent({dim3(static_cast<unsigned>((max_rank + warps - 1) / warps),
                                               static_cast<unsigned>(slice_count), token_blocks(per_warp)),
                                          dim3(static_cast<unsigned>(warps * 32)), bytes, stream},
                                         kernel, a));
    };
    if (tokens >= kWideMinTokens) {
        shrink(lora_shrink_kernel<kWideWarps, kWideTokens>, kWideWarps, kWideTokens);
    } else {
        shrink(lora_shrink_kernel<kNarrowWarps, kNarrowTokens>, kNarrowWarps, kNarrowTokens);
    }
    CUDA_CHECK(pdl::launch_dependent({dim3(static_cast<unsigned>((total + kExpandThreads - 1) / kExpandThreads),
                                           token_blocks(kExpandTokens)),
                                      dim3(kExpandThreads), 0, stream},
                                     lora_expand_kernel, a));
}

} // namespace sinfer::ops
