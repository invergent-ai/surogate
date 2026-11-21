// Inkling MoE: router, activation quantisers, standalone SiTU, combine and the LoRA shrink.
// Numerics: inkling_numerics.cuh; design: rl/engine-inkling-2026-11-21/lanes/d/DESIGN-moe.md.

#include "core/pdl.cuh"
#include "ops/common/mma.cuh"
#include "ops/sparse_moe/inkling/inkling_gemm_common.cuh"
#include "ops/sparse_moe/inkling/inkling_moe_internal.h"
#include "ops/sparse_moe/inkling/inkling_numerics.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

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

/// 16 bf16 at `source` (32-byte aligned) as fp32.
__device__ __forceinline__ void load16(const __nv_bfloat16* source, float* x) {
    const uint4* words = reinterpret_cast<const uint4*>(source);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint4 v                 = words[half];
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
__global__ __launch_bounds__(96) void router_logits_kernel(const __nv_bfloat16* __restrict__ x,
                                                           const __nv_bfloat16* __restrict__ router,
                                                           float* __restrict__ logits,
                                                           std::int32_t tokens,
                                                           std::int32_t* __restrict__ zero,
                                                           std::int32_t zero_words) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    const int thread = static_cast<int>(threadIdx.x);
    if (zero != nullptr && blockIdx.x == 0 && blockIdx.y == 0) {
        for (int i = thread; i < zero_words; i += 96) { zero[i] = 0; }
    }
    const int warp     = thread >> 5;
    const int lane     = thread & 31;
    const int group_id = lane >> 2;
    const int quad     = lane & 3;
    const int e0       = static_cast<int>(blockIdx.x) * 8;
    const int m0       = static_cast<int>(blockIdx.y) * 16;
    const int k_begin  = warp * kRouterGroup;
    const int k_end    = min(k_begin + kRouterGroup, kHidden);
    const int row0     = m0 + group_id;
    const int row1     = row0 + 8;
    const bool valid0  = row0 < tokens;
    const bool valid1  = row1 < tokens;
    const auto* x0     = reinterpret_cast<const std::uint32_t*>(x + static_cast<std::int64_t>(valid0 ? row0 : 0) * kHidden);
    const auto* x1     = reinterpret_cast<const std::uint32_t*>(x + static_cast<std::int64_t>(valid1 ? row1 : 0) * kHidden);
    const auto* w      = reinterpret_cast<const std::uint32_t*>(router + static_cast<std::int64_t>(e0 + group_id) * kHidden);

    // One chain of m16n8k16 MMAs from zero over the group, k ascending: `_k_group_tile`'s
    // `partial = tl.dot(x, w, partial)` over 128-wide tiles is this chain.
    float c0 = 0.0F, c1 = 0.0F, c2 = 0.0F, c3 = 0.0F;
    for (int k = k_begin; k < k_end; k += 16) {
        const int lo = (k + 2 * quad) >> 1;
        const int hi = (k + 8 + 2 * quad) >> 1;
        const std::uint32_t a0 = valid0 ? x0[lo] : 0U;
        const std::uint32_t a1 = valid1 ? x1[lo] : 0U;
        const std::uint32_t a2 = valid0 ? x0[hi] : 0U;
        const std::uint32_t a3 = valid1 ? x1[hi] : 0U;
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

// ---------------------------------------------------------------------------------------------
// Selection: a warp per token, 4 tokens per block; the last block lays out the expert rows.
// ---------------------------------------------------------------------------------------------
__global__ __launch_bounds__(128) void router_select_kernel(const float* __restrict__ logits,
                                                            const float* __restrict__ bias,
                                                            LiveTokens live, RouteOutputs out) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    const int thread = static_cast<int>(threadIdx.x);
    const int warp   = thread >> 5;
    const int lane   = thread & 31;
    const int token  = static_cast<int>(blockIdx.x) * 4 + warp;
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
        if (token < alive) {
            std::uint64_t keys[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int expert = lane + 32 * i;
                keys[i]          = router_key(row[expert], bias[expert], expert);
            }
            int my_expert = -1;
#pragma unroll
            for (int slot = 0; slot < kTopK; ++slot) {
                std::uint64_t local = keys[0];
#pragma unroll
                for (int i = 1; i < 4; ++i) { local = keys[i] > local ? keys[i] : local; }
                const std::uint64_t best = warp_max_u64(local);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (keys[i] == best) { keys[i] = 0; }
                }
                if (lane == slot) { my_expert = router_key_expert(best); }
            }
            const float my_log_prob = lane < kTopK ? router_log_sigmoid(row[my_expert]) : 0.0F;
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
            out.gammas[index] = 0.0F;
            if (out.counts != nullptr) { out.ranks[index] = -1; }
        }
    }
    if (out.counts == nullptr) { return; }

    // The last block to finish lays out the rows: offsets by expert, then each routed slot at
    // offsets[expert] + rank.
    __shared__ int is_last;
    __shared__ int offsets[kExperts + 1];
    __shared__ int warp_totals[4];
    __threadfence();
    __syncthreads();
    if (thread == 0) { is_last = atomicAdd(out.ticket, 1) == static_cast<int>(gridDim.x) - 1; }
    __syncthreads();
    if (!is_last) { return; }
    __threadfence();
    const int count = __ldcg(out.counts + thread);
    int inclusive   = count;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(0xFFFFFFFFU, inclusive, offset);
        if (lane >= offset) { inclusive += other; }
    }
    if (lane == 31) { warp_totals[warp] = inclusive; }
    __syncthreads();
    int before = 0;
    for (int w = 0; w < warp; ++w) { before += warp_totals[w]; }
    const int exclusive = before + inclusive - count;
    offsets[thread]     = exclusive;
    out.offsets[thread] = exclusive;
    if (thread == kExperts - 1) {
        offsets[kExperts]     = exclusive + count;
        out.offsets[kExperts] = exclusive + count;
    }
    if (!out.last_cta_places) { return; }
    __syncthreads();
    for (int index = thread; index < tokens * kTopK; index += 128) {
        const int expert = __ldcg(out.ids + index);
        if (expert >= 0) { out.sorted[offsets[expert] + __ldcg(out.ranks + index)] = index; }
    }
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
    if (row_ids != nullptr) { return row_ids[row] >= 0; }
    return row < live_count(live);
}

__global__ __launch_bounds__(256) void quant_nvfp4_kernel(const __nv_bfloat16* __restrict__ input,
                                                          std::int32_t rows, std::int32_t k,
                                                          const std::int32_t* __restrict__ row_ids,
                                                          LiveTokens live,
                                                          std::uint8_t* __restrict__ codes,
                                                          std::uint8_t* __restrict__ scales,
                                                          float* __restrict__ amax_out) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row  = static_cast<int>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5);
    if (row >= rows || !row_is_live(row, row_ids, live)) { return; }
    const __nv_bfloat16* source = input + static_cast<std::int64_t>(row) * k;
    const int blocks            = k / 16;
    float amax                  = 0.0F;
    for (int block = lane; block < blocks; block += 32) {
        float x[16];
        load16(source + block * 16, x);
#pragma unroll
        for (int j = 0; j < 16; ++j) { amax = fmaxf(amax, fabsf(x[j])); }
    }
    amax                         = warp_max(amax);
    const Nvfp4RowScales row_scales = nvfp4_row_scales(amax);
    for (int block = lane; block < blocks; block += 32) {
        float x[16];
        load16(source + block * 16, x);
        const Nvfp4Block quantized = nvfp4_quantize_block_4o6(x, row_scales, nvfp4_block_in_head(block, k));
        *reinterpret_cast<uint2*>(codes + static_cast<std::int64_t>(row) * (k / 2) + block * 8) =
            make_uint2(static_cast<std::uint32_t>(quantized.codes),
                       static_cast<std::uint32_t>(quantized.codes >> 32));
        scales[static_cast<std::int64_t>(row) * blocks + block] = quantized.scale;
    }
    if (lane == 0) { amax_out[row] = amax; }
}

__global__ __launch_bounds__(256) void quant_mxfp8_kernel(const __nv_bfloat16* __restrict__ input,
                                                          std::int32_t rows, std::int32_t k,
                                                          const std::int32_t* __restrict__ row_ids,
                                                          LiveTokens live,
                                                          std::uint8_t* __restrict__ codes,
                                                          std::uint8_t* __restrict__ exponents,
                                                          __nv_bfloat16* __restrict__ dequantised) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row  = static_cast<int>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5);
    if (row >= rows || !row_is_live(row, row_ids, live)) { return; }
    const __nv_bfloat16* source = input + static_cast<std::int64_t>(row) * k;
    const int blocks            = k / 32;
    for (int block = lane; block < blocks; block += 32) {
        float x[32];
        load16(source + block * 32, x);
        load16(source + block * 32 + 16, x + 16);
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
    const std::uint32_t pair =
        reinterpret_cast<const std::uint32_t*>(gate_up + row * kGateUpRows)[column];
    y[index] = situ(__uint_as_float(pair << 16), __uint_as_float(pair & 0xFFFF0000U), gamma[row]);
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
            if (ids[index] >= 0) {
                const std::uint32_t pair = *reinterpret_cast<const std::uint32_t*>(
                    down + static_cast<std::int64_t>(index) * kHidden + column);
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
    const int expert = ids[index];
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
            const uint4 xv = *reinterpret_cast<const uint4*>(x + k);
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

void launch_router_logits(const __nv_bfloat16* x, const __nv_bfloat16* router, float* logits,
                          std::int32_t tokens, std::int32_t* zero, std::int32_t zero_words,
                          cudaStream_t stream) {
    launch("router logits", dim3(kExperts / 8, static_cast<unsigned>((tokens + 15) / 16)), dim3(96),
           stream, router_logits_kernel, x, router, logits, tokens, zero, zero_words);
}

void launch_router_select(const float* logits, const float* bias, LiveTokens live,
                          const RouteOutputs& out, cudaStream_t stream) {
    launch("router select", dim3(static_cast<unsigned>((live.tokens + 3) / 4)), dim3(128), stream,
           router_select_kernel, logits, bias, live, out);
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
    launch("quant nvfp4", dim3(static_cast<unsigned>((rows + 7) / 8)), dim3(256), stream,
           quant_nvfp4_kernel, input, rows, k, row_ids, live, codes, scales, amax);
}

void launch_quant_mxfp8(const __nv_bfloat16* input, std::int32_t rows, std::int32_t k,
                        const std::int32_t* row_ids, LiveTokens live, std::uint8_t* codes,
                        std::uint8_t* exponents, __nv_bfloat16* dequantised, cudaStream_t stream) {
    if (k % 32 != 0) { throw std::invalid_argument("inkling_quant_mxfp8: k must be a multiple of 32"); }
    launch("quant mxfp8", dim3(static_cast<unsigned>((rows + 7) / 8)), dim3(256), stream,
           quant_mxfp8_kernel, input, rows, k, row_ids, live, codes, exponents, dequantised);
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

} // namespace sinfer::ops::detail::inkling
