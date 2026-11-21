// Inkling MoE NVFP4 (W4A4) grouped expert GEMMs: gate_up with the softcapped-SwiGLU epilogue, and
// down. The mainloop is `ops/linear/nvfp4/nvfp4_w4a4_mma.cuh`'s (canonical m16n8k64 block-scaled
// fragments, one fp32 accumulator per output from zero, K ascending): the instruction and operand
// roles of the reference's `_expert_gemm` (`tl.dot_scaled(a, a_scale, "e2m1", w, w_scale, "e2m1")`,
// tokens = A, weights = B). What differs from the dense kernel: token rows are gathered from an
// expert's routed slots, the weight rows start at the tile's expert, and the epilogues apply the
// per-row output scale (x_amax * w_amax[e]) * rcp_rn(1536 * den) of `ops/moe.py` `_row_scales`.

#include "core/pdl.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
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

template <int K>
struct Nvfp4Geometry {
    static constexpr int kInputRows        = K;
    static constexpr int kOutputRows       = kGateUpRows; // both projections have 2304 rows
    static constexpr int kCodeBytesPerRow  = K / 2;
    static constexpr int kScaleTilesPerRow = K / 64;
};

// BlockK 128: 1152 is not a multiple of 256. Static shared memory stays under 48 KB.
using NarrowSchedule = Nvfp4W4a4MmaSchedule<16, 64, 128, 1, 4, 4, 4>;
using WideSchedule   = Nvfp4W4a4MmaSchedule<64, 128, 128, 2, 4, 3, 2>;

template <class Schedule, int K>
__device__ __forceinline__ void stage_gathered_activation(const GemmParams& p, const int* source_rows,
                                                          Nvfp4W4a4SharedStorage<Schedule>& shared,
                                                          int stage, int k_tile) {
    using Geometry           = Nvfp4Geometry<K>;
    constexpr int kCodeTasks = Schedule::kBlockM * Schedule::kSegmentsPerRow;
    for (int task = static_cast<int>(threadIdx.x); task < kCodeTasks; task += Schedule::kThreads) {
        const int row           = task / Schedule::kSegmentsPerRow;
        const int segment       = task - row * Schedule::kSegmentsPerRow;
        const int source        = source_rows[row];
        const bool valid        = source >= 0;
        const int physical_byte = nvfp4_w4a4_swizzled_byte<Schedule>(row, segment * 16);
        auto* destination = shared.a_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
        const auto* input = p.a_codes +
                            static_cast<std::int64_t>(valid ? source : 0) * Geometry::kCodeBytesPerRow +
                            k_tile * Schedule::kCodeRowBytes + segment * 16;
        cp_async_zfill<16, Cache::cg>(destination, input, valid ? 16 : 0);
    }
    static_assert(Schedule::kK64PerStage <= 4);
    constexpr int kScaleBytes = Schedule::kK64PerStage * 4;
    for (int row = static_cast<int>(threadIdx.x); row < Schedule::kBlockM; row += Schedule::kThreads) {
        const int source  = source_rows[row];
        const bool valid  = source >= 0;
        auto* destination = &shared.a_scale4[stage][row * Schedule::kK64PerStage];
        const auto* input = p.a_scales + (static_cast<std::int64_t>(valid ? source : 0) *
                                              Geometry::kScaleTilesPerRow +
                                          k_tile * Schedule::kK64PerStage) *
                                             4;
        cp_async_zfill<kScaleBytes>(destination, input, valid ? kScaleBytes : 0);
    }
}

template <class Schedule, int K, bool GateUp>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void nvfp4_expert_gemm_kernel(
    GemmParams p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1200
    // Block-scaled FP4 MMA is sm_120+; Inkling's W4A4 experts are never dispatched below it.
    __trap();
#else
    using Geometry = Nvfp4Geometry<K>;
    __shared__ Nvfp4W4a4SharedStorage<Schedule> shared;
    __shared__ int tile_info[3];
    __shared__ int slots[Schedule::kBlockM];
    __shared__ int source_rows[Schedule::kBlockM];

    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    find_expert_tile<Schedule::kBlockM>(p.offsets, static_cast<int>(blockIdx.x), tile_info);
    const int expert = tile_info[0];
    if (expert < 0) { return; }
    const int row_begin = tile_info[1];
    const int row_end   = tile_info[2];
    for (int row = static_cast<int>(threadIdx.x); row < Schedule::kBlockM; row += Schedule::kThreads) {
        const int index   = row_begin + row;
        const int slot    = index < row_end ? p.sorted[index] : -1;
        slots[row]        = slot;
        source_rows[row]  = slot >= 0 ? (slot >> p.a_row_shift) : -1;
    }
    __syncthreads();

    const int n_begin       = static_cast<int>(blockIdx.y) * Schedule::kBlockN;
    const int weight_begin  = expert * kGateUpRows + n_begin;
    constexpr int kKTiles   = K / Schedule::kBlockK;
    constexpr int kWaitGroups = kKTiles < Schedule::kStages ? kKTiles - 1 : Schedule::kStages - 1;
    static_assert(K % Schedule::kBlockK == 0);

#pragma unroll
    for (int stage = 0; stage < Schedule::kStages; ++stage) {
        if (stage < kKTiles) {
            stage_gathered_activation<Schedule, K>(p, source_rows, shared, stage, stage);
            stage_nvfp4_w4a4_weight<Geometry, Schedule>(p.weights.codes, p.weights.scales, shared,
                                                        stage, stage, weight_begin,
                                                        Nvfp4W4a4IdentityRows{});
            cp_commit();
        }
    }

    float accumulators[Schedule::kMmaM][Schedule::kMmaN][4] = {};
    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int warp          = static_cast<int>(threadIdx.x) >> 5;
    const int warp_m        = warp / Schedule::kWarpsN;
    const int warp_n        = warp - warp_m * Schedule::kWarpsN;
    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row       = lane >> 2;

    for (int k_tile = 0; k_tile < kKTiles; ++k_tile) {
        const int stage = k_tile % Schedule::kStages;
        cp_wait<kWaitGroups>();
        __syncthreads();
#pragma unroll
        for (int local_k64 = 0; local_k64 < Schedule::kK64PerStage; ++local_k64) {
            unsigned a_fragments[Schedule::kMmaM][4];
            unsigned b_fragments[Schedule::kMmaN][2];
            unsigned a_scales[Schedule::kMmaM];
            unsigned b_scales[Schedule::kMmaN];
#pragma unroll
            for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
                const int row           = warp_m * Schedule::kWarpM + mma_m * 16 + a_row_offset;
                const int logical_byte  = local_k64 * 32 + a_column_byte;
                const int physical_byte = nvfp4_w4a4_swizzled_byte<Schedule>(row, logical_byte);
                const auto* address = shared.a_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
                ldmatrix_x4(a_fragments[mma_m][0], a_fragments[mma_m][1], a_fragments[mma_m][2],
                            a_fragments[mma_m][3], smem_addr(address));
                const int scale_row = warp_m * Schedule::kWarpM + mma_m * 16 + sfa_row;
                a_scales[mma_m] = shared.a_scale4[stage][scale_row * Schedule::kK64PerStage + local_k64];
            }
#pragma unroll
            for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
                const int row           = warp_n * Schedule::kWarpN + mma_n * 8 + b_row_offset;
                const int logical_byte  = local_k64 * 32 + b_column_byte;
                const int physical_byte = nvfp4_w4a4_swizzled_byte<Schedule>(row, logical_byte);
                const auto* address = shared.b_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
                ldmatrix_x2(b_fragments[mma_n][0], b_fragments[mma_n][1], smem_addr(address));
                const int scale_row = warp_n * Schedule::kWarpN + mma_n * 8 + sfb_row;
                constexpr int kQuartilesPerTile  = Schedule::kBlockN >= 128 ? 4 : Schedule::kBlockN / 32;
                constexpr int kScaleBytesPerTask = kQuartilesPerTile * 4;
                const int row_tile               = scale_row / 128;
                const int row_in_tile            = scale_row & 127;
                const int row_mod32              = row_in_tile & 31;
                const int row_quartile           = row_in_tile >> 5;
                const std::uint8_t* scale_address =
                    shared.b_scales[stage] +
                    ((row_tile * Schedule::kK64PerStage + local_k64) * 32 + row_mod32) * kScaleBytesPerTask +
                    row_quartile * 4;
                b_scales[mma_n] = load_vec<unsigned>(scale_address);
            }
#pragma unroll
            for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
#pragma unroll
                for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
                    mma_nvfp4_e4m3(accumulators[mma_m][mma_n][0], accumulators[mma_m][mma_n][1],
                                   accumulators[mma_m][mma_n][2], accumulators[mma_m][mma_n][3],
                                   a_fragments[mma_m][0], a_fragments[mma_m][1], a_fragments[mma_m][2],
                                   a_fragments[mma_m][3], b_fragments[mma_n][0], b_fragments[mma_n][1],
                                   a_scales[mma_m], b_scales[mma_n]);
                }
            }
        }
        __syncthreads();
        const int next_k_tile = k_tile + Schedule::kStages;
        if (next_k_tile < kKTiles) {
            stage_gathered_activation<Schedule, K>(p, source_rows, shared, stage, next_k_tile);
            stage_nvfp4_w4a4_weight<Geometry, Schedule>(p.weights.codes, p.weights.scales, shared,
                                                        stage, next_k_tile, weight_begin,
                                                        Nvfp4W4a4IdentityRows{});
        }
        cp_commit();
    }

    // Epilogue: v = acc * ((amax_row * amax_w[e]) * rcp_rn(1536 * den)) (+ adapter delta), bf16.
    const float rcp_denominator = __frcp_rn(__fmul_rn(1536.0F, *p.weights.outer_den));
    const float weight_amax     = p.weights.amax[expert];
    const int accumulator_row   = lane >> 2;
    const int accumulator_col   = 2 * (lane & 3);
#pragma unroll
    for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int local_row = warp_m * Schedule::kWarpM + mma_m * 16 + accumulator_row + 8 * half;
            const int slot      = slots[local_row];
            if (slot < 0) { continue; }
            const float row_scale =
                __fmul_rn(__fmul_rn(p.a_amax[source_rows[local_row]], weight_amax), rcp_denominator);
            const int adapter = lora_slot_of(p.lora, slot >> 3);
            const float* low  = adapter >= 0 ? p.lora.low + static_cast<std::int64_t>(slot) * p.lora.low_stride : nullptr;
#pragma unroll
            for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
                const int n  = n_begin + warp_n * Schedule::kWarpN + mma_n * 8 + accumulator_col;
                float value0 = __fmul_rn(accumulators[mma_m][mma_n][2 * half], row_scale);
                float value1 = __fmul_rn(accumulators[mma_m][mma_n][2 * half + 1], row_scale);
                if constexpr (GateUp) {
                    const int column = n >> 1;
                    if (adapter >= 0) {
                        const LoraBank& gate = lora_bank(p.lora, expert, 0);
                        const LoraBank& up   = lora_bank(p.lora, expert, 1);
                        value0 = __fadd_rn(value0, lora_expand(gate, adapter, column, low));
                        value1 = __fadd_rn(value1, lora_expand(up, adapter, column, low + gate.rank));
                    }
                    p.out[static_cast<std::int64_t>(slot) * kIntermediate + column] =
                        situ(round_bf16(value0), round_bf16(value1), p.gammas[slot]);
                } else {
                    if (adapter >= 0) {
                        const LoraBank& down = lora_bank(p.lora, expert, 2);
                        value0 = __fadd_rn(value0, lora_expand(down, adapter, n, low));
                        value1 = __fadd_rn(value1, lora_expand(down, adapter, n + 1, low));
                    }
                    *reinterpret_cast<__nv_bfloat162*>(p.out + static_cast<std::int64_t>(slot) * kHidden + n) =
                        __floats2bfloat162_rn(value0, value1);
                }
            }
        }
    }
#endif
}

template <class Schedule, int K, bool GateUp>
void launch_schedule(const GemmParams& params, std::int32_t slots, cudaStream_t stream) {
    const int max_tiles = std::min(kExperts, slots) + slots / Schedule::kBlockM;
    if (max_tiles == 0) { return; }
    const dim3 grid(static_cast<unsigned>(max_tiles), kGateUpRows / Schedule::kBlockN);
    const cudaError_t status = pdl::launch_dependent({grid, dim3(Schedule::kThreads), 0, stream},
                                                     nvfp4_expert_gemm_kernel<Schedule, K, GateUp>, params);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("inkling_moe: nvfp4 expert gemm: ") + cudaGetErrorString(status));
    }
}

void require(const GemmParams& params) {
    if (params.weights.codes == nullptr || params.weights.scales == nullptr ||
        params.weights.amax == nullptr || params.weights.outer_den == nullptr) {
        throw std::invalid_argument("inkling_moe: an NVFP4 bank needs codes, scales, amax and outer_den");
    }
}

} // namespace

void launch_nvfp4_gate_up(const GemmParams& params, std::int32_t slots, GemmSchedule schedule,
                          cudaStream_t stream) {
    require(params);
    if (schedule == GemmSchedule::Narrow) {
        launch_schedule<NarrowSchedule, kHidden, true>(params, slots, stream);
    } else {
        launch_schedule<WideSchedule, kHidden, true>(params, slots, stream);
    }
}

void launch_nvfp4_down(const GemmParams& params, std::int32_t slots, GemmSchedule schedule,
                       cudaStream_t stream) {
    require(params);
    if (schedule == GemmSchedule::Narrow) {
        launch_schedule<NarrowSchedule, kIntermediate, false>(params, slots, stream);
    } else {
        launch_schedule<WideSchedule, kIntermediate, false>(params, slots, stream);
    }
}

std::int32_t nvfp4_tile_rows(GemmSchedule schedule) {
    return schedule == GemmSchedule::Narrow ? NarrowSchedule::kBlockM : WideSchedule::kBlockM;
}

} // namespace sinfer::ops::detail::inkling
