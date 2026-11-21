// Inkling MoE MXFP8 (W8A8) grouped expert GEMMs, as the reference's sm_120 `_mxfp8_gemm`
// (ref/.../ops/moe_sm120.py): both operands dequantised to bf16 (exact: e4m3 x 2^k), bf16
// m16n8k16 MMAs with fp32 accumulators from zero, K ascending, WEIGHTS = operand A (16 rows),
// TOKENS = operand B (8 columns). Each thread converts its own canonical A-fragment elements in
// registers. Activations arrive already dequantised (bf16 rows written by the quantiser, the
// same bits `_dequant_e4m3` gives them in-kernel). Output bf16(acc) (the reference's expert
// scale is 1.0). gate_up rows are interleaved (2j = gate j, 2j + 1 = up j): MMA rows r < 8 map
// to the gate rows and r >= 8 to the up rows of the same 8 columns, so one thread holds each
// (gate, up) pair for the SiTU epilogue.

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
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sinfer::ops::detail::inkling {
namespace {

template <int BlockM, int BlockN, int WarpsM, int WarpsN, int Stages>
struct Mxfp8Schedule {
    static constexpr int kBlockM     = BlockM; ///< weight rows
    static constexpr int kBlockN     = BlockN; ///< token rows
    static constexpr int kBlockK     = 128;    ///< 4 E8M0 groups per weight row per stage
    static constexpr int kWarpsM     = WarpsM;
    static constexpr int kWarpsN     = WarpsN;
    static constexpr int kStages     = Stages;
    static constexpr int kThreads    = WarpsM * WarpsN * 32;
    static constexpr int kWarpM      = BlockM / WarpsM;
    static constexpr int kWarpN      = BlockN / WarpsN;
    static constexpr int kMmaM       = kWarpM / 16;
    static constexpr int kMmaN       = kWarpN / 8;
    static constexpr int kCodeStride = kBlockK + 16;    ///< bytes; 16-byte aligned rows, bank shift
    static constexpr int kActStride  = kBlockK + 8;     ///< bf16 elements
    static_assert(kWarpM % 16 == 0 && kWarpN % 8 == 0);
};

inline constexpr int kMxfp8KWidth = 2;

using NarrowSchedule = Mxfp8Schedule<64, 8, 4, 1, 3>;
using WideSchedule   = Mxfp8Schedule<128, 64, 4, 2, 2>; ///< 32 x 32 warp tiles: each dequantised A fragment feeds 4 MMAs

template <class Schedule>
struct Mxfp8Shared {
    alignas(16) std::uint8_t codes[Schedule::kStages][Schedule::kBlockM * Schedule::kCodeStride];
    alignas(16) std::uint8_t scales[Schedule::kStages][Schedule::kBlockM * 4];
    alignas(16) __nv_bfloat16 act[Schedule::kStages][Schedule::kBlockN * Schedule::kActStride];
};

template <class Schedule, int K>
__device__ __forceinline__ void stage_tiles(const std::uint8_t* weight_codes,
                                            const std::uint8_t* weight_scales,
                                            const __nv_bfloat16* activations, const int* source_rows,
                                            Mxfp8Shared<Schedule>& shared, int stage, int k_tile, int prefetch) {
    constexpr int kChunksPerRow = Schedule::kBlockK / 16;
    for (int task = static_cast<int>(threadIdx.x); task < Schedule::kBlockM * kChunksPerRow;
         task += Schedule::kThreads) {
        const int row   = task / kChunksPerRow;
        const int chunk = task - row * kChunksPerRow;
        cp_async_cg16_hint(shared.codes[stage] + row * Schedule::kCodeStride + chunk * 16,
                           weight_codes + static_cast<std::int64_t>(row) * K + k_tile * Schedule::kBlockK + chunk * 16,
                           prefetch);
    }
    for (int row = static_cast<int>(threadIdx.x); row < Schedule::kBlockM; row += Schedule::kThreads) {
        cp_async<4>(shared.scales[stage] + row * 4,
                    weight_scales + static_cast<std::int64_t>(row) * (K / 32) + k_tile * 4);
    }
    constexpr int kActChunks = Schedule::kBlockK / 8;
    for (int task = static_cast<int>(threadIdx.x); task < Schedule::kBlockN * kActChunks;
         task += Schedule::kThreads) {
        const int token  = task / kActChunks;
        const int chunk  = task - token * kActChunks;
        const int source = source_rows[token];
        const bool valid = source >= 0;
        cp_async_zfill<16, Cache::cg>(
            shared.act[stage] + token * Schedule::kActStride + chunk * 8,
            activations + static_cast<std::int64_t>(valid ? source : 0) * K + k_tile * Schedule::kBlockK + chunk * 8,
            valid ? 16 : 0);
    }
}

__device__ __forceinline__ unsigned dequant_fragment(const std::uint8_t* codes, float scale) {
    const __nv_bfloat162 pair = mxfp8_dequant_pair(*reinterpret_cast<const std::uint16_t*>(codes), scale);
    return *reinterpret_cast<const unsigned*>(&pair);
}

/// The K index behind fragment position pair (2q, 2q + 1) [k_lo] and (2q + 8, 2q + 9) [k_hi] of MMA
/// step `step` of a stage. KWidth 2 = the canonical m16n8k16 layout. KWidth > 2 = Triton's dot-operand
/// layout for operands upcast from narrow types: each thread holds KWidth consecutive K values, split over
/// KWidth / 4 MMAs (the same permutation on A and B, so the dot product is unchanged; the order of the
/// sum inside each MMA is not).
template <int KWidth>
__device__ __forceinline__ int mxfp8_k_lo(int step, int quad) {
    if constexpr (KWidth == 2) {
        return step * 16 + 2 * quad;
    } else {
        constexpr int kGroup = KWidth * 4;
        constexpr int kSteps = kGroup / 16;
        return (step / kSteps) * kGroup + KWidth * quad + 4 * (step % kSteps);
    }
}
template <int KWidth>
__device__ __forceinline__ int mxfp8_k_hi(int step, int quad) {
    return KWidth == 2 ? mxfp8_k_lo<KWidth>(step, quad) + 8 : mxfp8_k_lo<KWidth>(step, quad) + 2;
}

template <class Schedule, int K, bool GateUp, int KWidth>
__global__ __launch_bounds__(Schedule::kThreads) void mxfp8_expert_gemm_kernel(GemmParams p) {
    extern __shared__ __align__(16) unsigned char mxfp8_shared_bytes[];
    auto& shared = *reinterpret_cast<Mxfp8Shared<Schedule>*>(mxfp8_shared_bytes);
    __shared__ int tile_info[3];
    __shared__ int slots[Schedule::kBlockN];
    __shared__ int source_rows[Schedule::kBlockN];
    __shared__ int done[Schedule::kBlockN];
    __shared__ int done_count;

    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    if (threadIdx.x == 0) { done_count = 0; }
    find_expert_tile<Schedule::kBlockN>(p.offsets, static_cast<int>(blockIdx.x), tile_info);
    const int expert = tile_info[0];
    if (expert < 0) { return; }
    const int row_begin = tile_info[1];
    const int row_end   = tile_info[2];
    for (int row = static_cast<int>(threadIdx.x); row < Schedule::kBlockN; row += Schedule::kThreads) {
        const int index  = row_begin + row;
        const int slot   = index < row_end ? p.sorted[index] : -1;
        slots[row]       = slot;
        source_rows[row] = slot >= 0 ? (slot >> p.a_row_shift) : -1;
    }
    __syncthreads();

    const int n_begin = static_cast<int>(blockIdx.y) * Schedule::kBlockM;
    const std::int64_t weight_row = static_cast<std::int64_t>(expert) * kGateUpRows + n_begin;
    const std::uint8_t* weight_codes  = p.weights.codes + weight_row * K;
    const std::uint8_t* weight_scales = p.weights.scales + weight_row * (K / 32);
    constexpr int kKTiles     = K / Schedule::kBlockK;
    constexpr int kWaitGroups = kKTiles < Schedule::kStages ? kKTiles - 1 : Schedule::kStages - 1;
    static_assert(K % Schedule::kBlockK == 0);

#pragma unroll
    for (int stage = 0; stage < Schedule::kStages; ++stage) {
        if (stage < kKTiles) {
            stage_tiles<Schedule, K>(weight_codes, weight_scales, p.a_bf16, source_rows, shared, stage, stage, p.l2_prefetch);
            cp_commit();
        }
    }

    float accumulators[Schedule::kMmaM][Schedule::kMmaN][4] = {};
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int warp_m   = warp / Schedule::kWarpsN;
    const int warp_n   = warp - warp_m * Schedule::kWarpsN;
    const int group_id = lane >> 2;
    const int quad     = lane & 3;

    for (int k_tile = 0; k_tile < kKTiles; ++k_tile) {
        const int stage = k_tile % Schedule::kStages;
        cp_wait<kWaitGroups>();
        __syncthreads();
        const std::uint8_t* codes  = shared.codes[stage];
        const std::uint8_t* scales = shared.scales[stage];
        const __nv_bfloat16* act   = shared.act[stage];
#pragma unroll
        for (int kk = 0; kk < Schedule::kBlockK / 16; ++kk) {
            const int k_lo = mxfp8_k_lo<KWidth>(kk, quad);
            const int k_hi = mxfp8_k_hi<KWidth>(kk, quad);
            unsigned a_fragments[Schedule::kMmaM][4];
            unsigned b_fragments[Schedule::kMmaN][2];
#pragma unroll
            for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
                const int base  = warp_m * Schedule::kWarpM + mma_m * 16;
                const int row_a = GateUp ? base + 2 * group_id : base + group_id;
                const int row_b = GateUp ? base + 2 * group_id + 1 : base + group_id + 8;
                const float scale_a_lo = e8m0_to_f32(scales[row_a * 4 + (k_lo >> 5)]);
                const float scale_b_lo = e8m0_to_f32(scales[row_b * 4 + (k_lo >> 5)]);
                const float scale_a_hi = e8m0_to_f32(scales[row_a * 4 + (k_hi >> 5)]);
                const float scale_b_hi = e8m0_to_f32(scales[row_b * 4 + (k_hi >> 5)]);
                a_fragments[mma_m][0] = dequant_fragment(codes + row_a * Schedule::kCodeStride + k_lo, scale_a_lo);
                a_fragments[mma_m][1] = dequant_fragment(codes + row_b * Schedule::kCodeStride + k_lo, scale_b_lo);
                a_fragments[mma_m][2] = dequant_fragment(codes + row_a * Schedule::kCodeStride + k_hi, scale_a_hi);
                a_fragments[mma_m][3] = dequant_fragment(codes + row_b * Schedule::kCodeStride + k_hi, scale_b_hi);
            }
#pragma unroll
            for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
                const int token = warp_n * Schedule::kWarpN + mma_n * 8 + group_id;
                const __nv_bfloat16* row = act + token * Schedule::kActStride;
                b_fragments[mma_n][0] = *reinterpret_cast<const unsigned*>(row + k_lo);
                b_fragments[mma_n][1] = *reinterpret_cast<const unsigned*>(row + k_hi);
            }
#pragma unroll
            for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
#pragma unroll
                for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
                    mma_bf16(accumulators[mma_m][mma_n][0], accumulators[mma_m][mma_n][1],
                             accumulators[mma_m][mma_n][2], accumulators[mma_m][mma_n][3],
                             a_fragments[mma_m][0], a_fragments[mma_m][1], a_fragments[mma_m][2],
                             a_fragments[mma_m][3], b_fragments[mma_n][0], b_fragments[mma_n][1]);
                }
            }
        }
        __syncthreads();
        const int next_k_tile = k_tile + Schedule::kStages;
        if (next_k_tile < kKTiles) {
            stage_tiles<Schedule, K>(weight_codes, weight_scales, p.a_bf16, source_rows, shared, stage,
                                     next_k_tile, p.l2_prefetch);
        }
        cp_commit();
    }

    // Epilogue. Accumulator c0 = (row g, token 2q), c1 = (row g, token 2q + 1), c2 = (row g + 8,
    // token 2q), c3 = (row g + 8, token 2q + 1).
#pragma unroll
    for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
        const int base = warp_m * Schedule::kWarpM + mma_m * 16;
#pragma unroll
        for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int token_local = warp_n * Schedule::kWarpN + mma_n * 8 + 2 * quad + half;
                const int slot        = slots[token_local];
                if (slot < 0) { continue; }
                const int adapter = lora_slot_of(p.lora, slot >> 3);
                const float* low  = adapter >= 0 ? p.lora.low + static_cast<std::int64_t>(slot) * p.lora.low_stride : nullptr;
                float first  = accumulators[mma_m][mma_n][half];
                float second = accumulators[mma_m][mma_n][2 + half];
                if constexpr (GateUp) {
                    const int column = (n_begin + base) / 2 + group_id;
                    if (adapter >= 0) {
                        const LoraBank& gate = lora_bank(p.lora, expert, 0);
                        const LoraBank& up   = lora_bank(p.lora, expert, 1);
                        first  = __fadd_rn(first, lora_expand(gate, adapter, column, low));
                        second = __fadd_rn(second, lora_expand(up, adapter, column, low + gate.rank));
                    }
                    if (p.gate_up_tap != nullptr) {
                        *reinterpret_cast<__nv_bfloat162*>(p.gate_up_tap + static_cast<std::int64_t>(slot) * kGateUpRows +
                                                           2 * column) = __floats2bfloat162_rn(first, second);
                    }
                    p.out[static_cast<std::int64_t>(slot) * kIntermediate + column] =
                        situ(round_bf16(first), round_bf16(second), p.gammas[slot]);
                } else {
                    const int n_first  = n_begin + base + group_id;
                    const int n_second = n_first + 8;
                    if (adapter >= 0) {
                        const LoraBank& down = lora_bank(p.lora, expert, 2);
                        first  = __fadd_rn(first, lora_expand(down, adapter, n_first, low));
                        second = __fadd_rn(second, lora_expand(down, adapter, n_second, low));
                    }
                    __nv_bfloat16* row = p.out + static_cast<std::int64_t>(slot) * kHidden;
                    row[n_first]       = __float2bfloat16_rn(first);
                    row[n_second]      = __float2bfloat16_rn(second);
                }
            }
        }
    }
    if constexpr (!GateUp) {
        if (p.arrivals != nullptr) {
            combine_arrive<Schedule::kBlockM, Schedule::kThreads>(
                p, Schedule::kBlockN, n_begin, [&](int r) { return slots[r]; }, done, &done_count,
                static_cast<int>(threadIdx.x), [] { __syncthreads(); });
        }
    }
}

template <class Schedule, int K, bool GateUp, int KWidth>
cudaError_t launch_kwidth(const dim3& grid, const GemmParams& params, cudaStream_t stream) {
    constexpr std::size_t kShared = sizeof(Mxfp8Shared<Schedule>);
    auto* kernel = mxfp8_expert_gemm_kernel<Schedule, K, GateUp, KWidth>;
    const cudaError_t status = ::sinfer::ops::set_func_attribute_per_device(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kShared));
    if (status != cudaSuccess) { return status; }
    return pdl::launch_dependent({grid, dim3(Schedule::kThreads), kShared, stream}, kernel, params);
}

template <class Schedule, int K, bool GateUp>
void launch_schedule(const GemmParams& params, std::int32_t slots, cudaStream_t stream) {
    const int max_tiles = std::min(kExperts, slots) + slots / Schedule::kBlockN;
    if (max_tiles == 0) { return; }
    const dim3 grid(static_cast<unsigned>(max_tiles), kGateUpRows / Schedule::kBlockM);
    // One fragment layout at every width (canonical, KWidth 2). M3 run 3 measured the alternatives against
    // the vLLM capture (layer 20): KWidth 4 = the same 27 / 4.7 M gate_up elements off (same K set per MMA),
    // 8 -> 25, 16 -> 44: none reproduces Triton's K grouping; the residual is recorded, not chased.
    const cudaError_t status = launch_kwidth<Schedule, K, GateUp, kMxfp8KWidth>(grid, params, stream);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("inkling_moe: mxfp8 expert gemm: ") + cudaGetErrorString(status));
    }
}

void require(const GemmParams& params) {
    if (params.weights.codes == nullptr || params.weights.scales == nullptr || params.a_bf16 == nullptr) {
        throw std::invalid_argument("inkling_moe: an MXFP8 bank needs codes and scales");
    }
}

} // namespace

void launch_mxfp8_gate_up(const GemmParams& params, std::int32_t slots, GemmSchedule schedule,
                          cudaStream_t stream) {
    require(params);
    const GemmParams p = with_l2_prefetch(params);
    if (schedule == GemmSchedule::Narrow) {
        launch_schedule<NarrowSchedule, kHidden, true>(p, slots, stream);
    } else {
        launch_schedule<WideSchedule, kHidden, true>(p, slots, stream);
    }
}

void launch_mxfp8_down(const GemmParams& params, std::int32_t slots, GemmSchedule schedule,
                       cudaStream_t stream) {
    require(params);
    const GemmParams p = with_l2_prefetch(params);
    if (schedule == GemmSchedule::Narrow) {
        launch_schedule<NarrowSchedule, kIntermediate, false>(p, slots, stream);
    } else {
        launch_schedule<WideSchedule, kIntermediate, false>(p, slots, stream);
    }
}

std::int32_t mxfp8_tile_rows(GemmSchedule schedule) {
    return schedule == GemmSchedule::Narrow ? NarrowSchedule::kBlockN : WideSchedule::kBlockN;
}

} // namespace sinfer::ops::detail::inkling
