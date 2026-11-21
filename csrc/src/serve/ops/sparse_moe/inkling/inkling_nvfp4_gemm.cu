// Inkling MoE NVFP4 (W4A4) grouped expert GEMMs: gate_up with the softcapped-SwiGLU epilogue, and
// down. The mainloop is `ops/linear/nvfp4/nvfp4_w4a4_mma.cuh`'s (canonical m16n8k64 block-scaled
// fragments, one fp32 accumulator per output from zero, K ascending): the instruction and operand
// roles of the reference's `_expert_gemm` (`tl.dot_scaled(a, a_scale, "e2m1", w, w_scale, "e2m1")`,
// tokens = A, weights = B). What differs from the dense kernel: token rows are gathered from an
// expert's routed slots, the weight rows start at the tile's expert, and the epilogues apply the
// per-row output scale (x_amax * w_amax[e]) * rcp_rn(1536 * den) of `ops/moe.py` `_row_scales`.

//
// Two kernels share that arithmetic (bit-identical rows; only the tiling differs):
// - `nvfp4_expert_gemm_kernel`: one CTA per (expert tile, N tile), all warps load and compute
//   (narrow decode / verify widths);
// - `nvfp4_expert_gemm_persistent`: prefill. One CTA per SM loops over the round's (expert tile,
//   N tile) items, expert-major with N inner. A producer warpgroup streams the gathered token rows
//   and the expert's weight rows with cp.async into a ring of stages (completion signalled on an
//   mbarrier per stage, `cp.async.mbarrier.arrive.noinc`); 8 consumer warps (64 x 64 warp tiles)
//   run the same m16n8k64 chain per output and release each stage on a second mbarrier. Warps
//   skip MMAs of 16-row groups past the tile's last routed row (those rows are never stored).

#include "core/pdl.cuh"
#include "ops/kernel/func_attribute.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma.cuh"
#include "ops/sparse_moe/inkling/inkling_gemm_common.cuh"
#include "ops/sparse_moe/inkling/inkling_moe_internal.h"
#include "ops/sparse_moe/inkling/inkling_numerics.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace sinfer::ops::detail::inkling {
namespace {

/// One routed row's outputs of one MMA row group (`half` = accumulator rows +0 / +8): scale,
/// adapter delta, then SiTU -> y (gate_up) or the bf16 down row. `n_first` = the output row of
/// mma_n 0 for this lane (n_begin + warp offset + accumulator column).
template <bool GateUp, int MmaN>
__device__ __forceinline__ void store_routed_row(const GemmParams& p, int expert, int slot, float row_amax,
                                                 float weight_amax, float rcp_denominator, int n_first,
                                                 const float (&acc)[MmaN][4], int half) {
    const float row_scale = __fmul_rn(__fmul_rn(row_amax, weight_amax), rcp_denominator);
    const int adapter     = lora_slot_of(p.lora, slot >> 3);
    const float* low = adapter >= 0 ? p.lora.low + static_cast<std::int64_t>(slot) * p.lora.low_stride : nullptr;
#pragma unroll
    for (int mma_n = 0; mma_n < MmaN; ++mma_n) {
        const int n  = n_first + mma_n * 8;
        float value0 = __fmul_rn(acc[mma_n][2 * half], row_scale);
        float value1 = __fmul_rn(acc[mma_n][2 * half + 1], row_scale);
        if constexpr (GateUp) {
            const int column = n >> 1;
            if (adapter >= 0) {
                const LoraBank& gate = lora_bank(p.lora, expert, 0);
                const LoraBank& up   = lora_bank(p.lora, expert, 1);
                value0 = __fadd_rn(value0, lora_expand(gate, adapter, column, low));
                value1 = __fadd_rn(value1, lora_expand(up, adapter, column, low + gate.rank));
            }
            if (p.gate_up_tap != nullptr) {
                *reinterpret_cast<__nv_bfloat162*>(p.gate_up_tap + static_cast<std::int64_t>(slot) * kGateUpRows + n) =
                    __floats2bfloat162_rn(value0, value1);
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

/// Narrow down with the y quantisation fused (DESIGN-moe 2.1): the tile's routed y rows quantised
/// (four-over-six, the quant kernel's code) into a resident shared A operand; row stride 592 bytes
/// (576 + 16: the 8 rows of an ldmatrix phase hit distinct bank groups).
inline constexpr int kFusedYRowBytes = kIntermediate / 2 + 16;
inline constexpr int kFusedYScaleWords = kIntermediate / 64;

/// 16 bf16 at `source` (32-byte aligned) as fp32 (the quant kernel's load16).
__device__ __forceinline__ void load16_bf16(const __nv_bfloat16* source, float* x) {
    const uint4* words = reinterpret_cast<const uint4*>(source);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint4 v                = words[half];
        const std::uint32_t parts[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            x[half * 8 + 2 * i]     = __uint_as_float(parts[i] << 16);
            x[half * 8 + 2 * i + 1] = __uint_as_float(parts[i] & 0xFFFF0000U);
        }
    }
}

template <class Schedule, int K, bool GateUp, bool FusedYQuant = false>
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
    __shared__ int done[Schedule::kBlockM];
    __shared__ int done_count;
    // Fused y quantisation only (1-element arrays otherwise).
    __shared__ __align__(16) std::uint8_t y_codes[FusedYQuant ? Schedule::kBlockM * kFusedYRowBytes : 16];
    __shared__ std::uint32_t y_scales[FusedYQuant ? Schedule::kBlockM * kFusedYScaleWords : 1];
    __shared__ float y_amax[FusedYQuant ? Schedule::kBlockM : 1];

    if constexpr (!FusedYQuant) { pdl::wait_for_dependencies(); }
    pdl::trigger_dependents();
    if (threadIdx.x == 0) { done_count = 0; }
    // Fused: the routing (offsets / sorted, written by the select kernel two or more launches back,
    // complete before the gate_up kernel's wait returned) is read through L2 BEFORE this kernel's
    // dependency wait, so the expert's w2 stages stream while the predecessor finishes.
    find_expert_tile<Schedule::kBlockM, FusedYQuant>(p.offsets, static_cast<int>(blockIdx.x), tile_info);
    const int expert = tile_info[0];
    if (expert < 0) {
        if constexpr (FusedYQuant) { pdl::wait_for_dependencies(); }
        return;
    }
    const int row_begin = tile_info[1];
    const int row_end   = tile_info[2];
    for (int row = static_cast<int>(threadIdx.x); row < Schedule::kBlockM; row += Schedule::kThreads) {
        const int index   = row_begin + row;
        const int slot    = index < row_end ? (FusedYQuant ? __ldcg(p.sorted + index) : p.sorted[index]) : -1;
        slots[row]        = slot;
        source_rows[row]  = slot >= 0 ? (slot >> p.a_row_shift) : -1;
    }
    __syncthreads();

    const int n_begin       = static_cast<int>(blockIdx.y) * Schedule::kBlockN;
    const int weight_begin  = expert * kGateUpRows + n_begin;
    constexpr int kKTiles   = K / Schedule::kBlockK;
    constexpr int kWaitGroups = kKTiles < Schedule::kStages ? kKTiles - 1 : Schedule::kStages - 1;
    static_assert(K % Schedule::kBlockK == 0);
    static_assert(!FusedYQuant || (!GateUp && K == kIntermediate));

    if constexpr (FusedYQuant) {
#pragma unroll
        for (int stage = 0; stage < Schedule::kStages; ++stage) {
            if (stage < kKTiles) {
                stage_nvfp4_w4a4_weight<Geometry, Schedule>(p.weights.codes, p.weights.scales, shared,
                                                            stage, stage, weight_begin,
                                                            Nvfp4W4a4IdentityRows{});
                cp_commit();
            }
        }
        pdl::wait_for_dependencies();
        // y rows -> four-over-six (quant_nvfp4_row's arithmetic), a warp per row.
        const int quant_lane = static_cast<int>(threadIdx.x) & 31;
        for (int row = static_cast<int>(threadIdx.x) >> 5; row < Schedule::kBlockM; row += Schedule::kThreads / 32) {
            const int source = source_rows[row];
            std::uint8_t* codes_row = y_codes + row * kFusedYRowBytes;
            std::uint32_t* scales_row = y_scales + row * kFusedYScaleWords;
            if (source < 0) {
                // Unrouted tile rows: zero codes (their MMA rows are never stored).
                for (int i = quant_lane; i < kIntermediate / 32; i += 32) {
                    reinterpret_cast<uint4*>(codes_row)[i] = make_uint4(0U, 0U, 0U, 0U);
                }
                if (quant_lane < kFusedYScaleWords) { scales_row[quant_lane] = 0U; }
                continue;
            }
            const __nv_bfloat16* y_row = p.a_bf16 + static_cast<std::int64_t>(source) * kIntermediate;
            constexpr int kBlocks = kIntermediate / 16;
            float amax = 0.0F;
            for (int block = quant_lane; block < kBlocks; block += 32) {
                float x[16];
                load16_bf16(y_row + block * 16, x);
#pragma unroll
                for (int j = 0; j < 16; ++j) { amax = fmaxf(amax, fabsf(x[j])); }
            }
            amax = warp_max(amax);
            const Nvfp4RowScales row_scales = nvfp4_row_scales(amax);
            for (int block = quant_lane; block < kBlocks; block += 32) {
                float x[16];
                load16_bf16(y_row + block * 16, x);
                const Nvfp4Block quantized =
                    nvfp4_quantize_block_4o6(x, row_scales, nvfp4_block_in_head(block, kIntermediate));
                *reinterpret_cast<uint2*>(codes_row + block * 8) =
                    make_uint2(static_cast<std::uint32_t>(quantized.codes),
                               static_cast<std::uint32_t>(quantized.codes >> 32));
                reinterpret_cast<std::uint8_t*>(scales_row)[block] = quantized.scale;
            }
            if (quant_lane == 0) { y_amax[row] = amax; }
        }
    } else {
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
                const int scale_row     = warp_m * Schedule::kWarpM + mma_m * 16 + sfa_row;
                if constexpr (FusedYQuant) {
                    const auto* address = y_codes + row * kFusedYRowBytes + k_tile * Schedule::kCodeRowBytes + logical_byte;
                    ldmatrix_x4(a_fragments[mma_m][0], a_fragments[mma_m][1], a_fragments[mma_m][2],
                                a_fragments[mma_m][3], smem_addr(address));
                    a_scales[mma_m] = y_scales[scale_row * kFusedYScaleWords + k_tile * Schedule::kK64PerStage + local_k64];
                } else {
                    const int physical_byte = nvfp4_w4a4_swizzled_byte<Schedule>(row, logical_byte);
                    const auto* address = shared.a_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
                    ldmatrix_x4(a_fragments[mma_m][0], a_fragments[mma_m][1], a_fragments[mma_m][2],
                                a_fragments[mma_m][3], smem_addr(address));
                    a_scales[mma_m] = shared.a_scale4[stage][scale_row * Schedule::kK64PerStage + local_k64];
                }
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
            if constexpr (!FusedYQuant) {
                stage_gathered_activation<Schedule, K>(p, source_rows, shared, stage, next_k_tile);
            }
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
            float row_amax;
            if constexpr (FusedYQuant) {
                row_amax = y_amax[local_row];
            } else {
                row_amax = p.a_amax[source_rows[local_row]];
            }
            store_routed_row<GateUp, Schedule::kMmaN>(p, expert, slot, row_amax, weight_amax,
                                                      rcp_denominator,
                                                      n_begin + warp_n * Schedule::kWarpN + accumulator_col,
                                                      accumulators[mma_m], half);
        }
    }
    if constexpr (!GateUp) {
        if (p.arrivals != nullptr) {
            combine_arrive<Schedule::kBlockN, Schedule::kThreads>(
                p, Schedule::kBlockM, n_begin, [&](int r) { return slots[r]; }, done, &done_count,
                static_cast<int>(threadIdx.x), [] { __syncthreads(); });
        }
    }
#endif
}

// ---------------------------------------------------------------------------------------------
// Persistent prefill kernel.
// ---------------------------------------------------------------------------------------------
template <int BlockM, int BlockN, int WarpsM, int WarpsN, int Stages>
struct PersistentSchedule {
    static constexpr int kBlockM         = BlockM; ///< token rows (MMA operand A)
    static constexpr int kBlockN         = BlockN; ///< weight rows (operand B)
    static constexpr int kBlockK         = 128;
    static constexpr int kStages         = Stages;
    static constexpr int kWarpsM         = WarpsM;
    static constexpr int kWarpsN         = WarpsN;
    static constexpr int kConsumerWarps  = WarpsM * WarpsN;
    static constexpr int kConsumerThreads = kConsumerWarps * 32;
    static constexpr int kProducerThreads = 128; ///< one producer warpgroup (setmaxnreg)
    static constexpr int kThreads        = kConsumerThreads + kProducerThreads;
    static constexpr int kWarpM          = BlockM / WarpsM;
    static constexpr int kWarpN          = BlockN / WarpsN;
    static constexpr int kMmaM           = kWarpM / 16;
    static constexpr int kMmaN           = kWarpN / 8;
    static constexpr int kK64PerStage    = kBlockK / 64;
    static constexpr int kCodeRowBytes   = kBlockK / 2; ///< 64
    static constexpr int kSegmentsPerRow = kCodeRowBytes / 16;
    static constexpr int kNTiles         = kGateUpRows / BlockN;
    static constexpr int kScaleRowTiles  = BlockN / 128;
    static_assert(kGateUpRows % BlockN == 0 && BlockN % 128 == 0);
    static_assert(kWarpM % 16 == 0 && kWarpN % 8 == 0);
    static_assert(kSegmentsPerRow == 4);
};

template <class Schedule>
struct PersistentStorage {
    alignas(128) std::uint8_t a_codes[Schedule::kStages][Schedule::kBlockM * Schedule::kCodeRowBytes];
    alignas(128) std::uint8_t b_codes[Schedule::kStages][Schedule::kBlockN * Schedule::kCodeRowBytes];
    alignas(16) std::uint32_t a_scale4[Schedule::kStages][Schedule::kBlockM * Schedule::kK64PerStage];
    alignas(16) std::uint8_t b_scales[Schedule::kStages][Schedule::kBlockN * Schedule::kK64PerStage * 4];
    alignas(8) std::uint64_t full[Schedule::kStages];
    alignas(8) std::uint64_t empty[Schedule::kStages];
    int tile_prefix[kExperts + 1]; ///< exclusive prefix of BlockM tiles per expert; [128] = total
    int source_rows[Schedule::kBlockM]; ///< producer-private: token row per tile row (-1 = none)
    int done[Schedule::kBlockM];        ///< consumers: tokens completed by this item (fused combine)
    int done_count;
};

/// 64-byte code rows: 16-byte segment s of row r lives at segment s ^ ((r >> 1) & 3), so the 8
/// rows of one ldmatrix phase hit 8 distinct bank groups (the TMA kernel's 64B swizzle).
__device__ __forceinline__ int persistent_swizzled_byte(int row, int logical_byte) {
    return (((logical_byte >> 4) ^ ((row >> 1) & 3)) << 4) | (logical_byte & 15);
}

__device__ __forceinline__ void cp_async_arrive_noinc(std::uint64_t* barrier) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" : : "r"(smem_addr(barrier)) : "memory");
}

struct PersistentItem {
    int expert;
    int row_begin;
    int row_end;
    int n_begin;
};

/// Item `item` (= tile * kNTiles + n tile) from the smem tile prefix: the expert is the last one
/// whose first tile is <= the tile (experts without rows have no tiles).
template <class Schedule>
__device__ __forceinline__ PersistentItem persistent_item(const int* tile_prefix, const std::int32_t* offsets,
                                                          int item) {
    const int tile = item / Schedule::kNTiles;
    const int n    = item - tile * Schedule::kNTiles;
    int lo = 0, hi = kExperts - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (tile_prefix[mid] <= tile) { lo = mid; } else { hi = mid - 1; }
    }
    PersistentItem result;
    result.expert    = lo;
    result.row_begin = offsets[lo] + (tile - tile_prefix[lo]) * Schedule::kBlockM;
    result.row_end   = min(result.row_begin + Schedule::kBlockM, offsets[lo + 1]);
    result.n_begin   = n * Schedule::kBlockN;
    return result;
}

template <class Schedule, int K>
__device__ __forceinline__ void persistent_fill_stage(const GemmParams& p, PersistentStorage<Schedule>& shared,
                                                      int stage, int k_tile, int weight_begin, int lane) {
    constexpr int kStride = Schedule::kProducerThreads;
    constexpr int kCodeBytesPerRow  = K / 2;
    constexpr int kScaleBytesPerRow = K / 16;
    constexpr int kScaleTilesPerRow = K / 64;
    // Token rows (gathered): 4 lanes per row, one 16-byte segment each.
    constexpr int kATasks = Schedule::kBlockM * Schedule::kSegmentsPerRow;
#pragma unroll 2
    for (int task = lane; task < kATasks; task += kStride) {
        const int row     = task >> 2;
        const int segment = task & 3;
        const int source  = shared.source_rows[row];
        const bool valid  = source >= 0;
        auto* destination = shared.a_codes[stage] + row * Schedule::kCodeRowBytes +
                            persistent_swizzled_byte(row, segment * 16);
        const auto* input = p.a_codes + static_cast<std::int64_t>(valid ? source : 0) * kCodeBytesPerRow +
                            k_tile * Schedule::kCodeRowBytes + segment * 16;
        cp_async_zfill<16, Cache::cg>(destination, input, valid ? 16 : 0);
    }
    constexpr int kAScaleBytes = Schedule::kK64PerStage * 4;
#pragma unroll
    for (int row = lane; row < Schedule::kBlockM; row += kStride) {
        const int source  = shared.source_rows[row];
        const bool valid  = source >= 0;
        const auto* input = p.a_scales + static_cast<std::int64_t>(valid ? source : 0) * kScaleBytesPerRow +
                            k_tile * kAScaleBytes;
        cp_async_zfill<kAScaleBytes>(&shared.a_scale4[stage][row * Schedule::kK64PerStage], input,
                                     valid ? kAScaleBytes : 0);
    }
    // Weight rows (contiguous inside the expert).
    constexpr int kBTasks = Schedule::kBlockN * Schedule::kSegmentsPerRow;
#pragma unroll 2
    for (int task = lane; task < kBTasks; task += kStride) {
        const int row     = task >> 2;
        const int segment = task & 3;
        auto* destination = shared.b_codes[stage] + row * Schedule::kCodeRowBytes +
                            persistent_swizzled_byte(row, segment * 16);
        const auto* input = p.weights.codes + static_cast<std::int64_t>(weight_begin + row) * kCodeBytesPerRow +
                            k_tile * Schedule::kCodeRowBytes + segment * 16;
        cp_async<16, Cache::cg>(destination, input);
    }
    // Weight scales: the bank's 128-row tile layout, one 16-byte group (4 row quartiles) per task.
    constexpr int kBScaleTasks = Schedule::kScaleRowTiles * Schedule::kK64PerStage * 32;
#pragma unroll
    for (int task = lane; task < kBScaleTasks; task += kStride) {
        const int row_tile  = task / (Schedule::kK64PerStage * 32);
        const int remainder = task - row_tile * Schedule::kK64PerStage * 32;
        const int local_k64 = remainder >> 5;
        const int row_mod32 = remainder & 31;
        const int global_k64      = k_tile * Schedule::kK64PerStage + local_k64;
        const int persistent_tile = (weight_begin >> 7) + row_tile;
        auto* destination = shared.b_scales[stage] +
                            ((row_tile * Schedule::kK64PerStage + local_k64) * 32 + row_mod32) * 16;
        const auto* input = p.weights.scales +
                            static_cast<std::int64_t>(persistent_tile * kScaleTilesPerRow + global_k64) * 512 +
                            row_mod32 * 16;
        cp_async<16>(destination, input);
    }
}

template <class Schedule, int K, bool GateUp>
__global__ __launch_bounds__(Schedule::kThreads, 1) void nvfp4_expert_gemm_persistent(GemmParams p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1200
    __trap();
#else
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto& shared = *reinterpret_cast<PersistentStorage<Schedule>*>(shared_bytes);
    constexpr int kKTiles = K / Schedule::kBlockK;
    static_assert(K % Schedule::kBlockK == 0);

    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    const int thread = static_cast<int>(threadIdx.x);
    if (thread < 32) {
        // Tiles per expert -> exclusive prefix (4 experts per lane, one warp scan).
        int bounds[5];
#pragma unroll
        for (int i = 0; i < 5; ++i) { bounds[i] = p.offsets[4 * thread + i]; }
        int tiles[4];
        int local = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            tiles[i] = (bounds[i + 1] - bounds[i] + Schedule::kBlockM - 1) / Schedule::kBlockM;
            local += tiles[i];
        }
        int inclusive = local;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int other = __shfl_up_sync(0xFFFFFFFFU, inclusive, offset);
            if (thread >= offset) { inclusive += other; }
        }
        int before = inclusive - local;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            shared.tile_prefix[4 * thread + i] = before;
            before += tiles[i];
        }
        if (thread == 31) { shared.tile_prefix[kExperts] = inclusive; }
    } else if (thread == 32) {
        shared.done_count = 0;
#pragma unroll
        for (int stage = 0; stage < Schedule::kStages; ++stage) {
            nvfp4_mbarrier_init(&shared.full[stage], Schedule::kProducerThreads);
            nvfp4_mbarrier_init(&shared.empty[stage], Schedule::kConsumerWarps);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
    }
    __syncthreads();
    const int total_items = shared.tile_prefix[kExperts] * Schedule::kNTiles;
    const int lane        = thread & 31;

    if (thread >= Schedule::kConsumerThreads) {
        // Producer warpgroup: gives registers to the consumers (the TMA kernel's split).
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" : : : "memory");
        const int producer = thread - Schedule::kConsumerThreads;
        int fill = 0;
        for (int item = static_cast<int>(blockIdx.x); item < total_items; item += static_cast<int>(gridDim.x)) {
            const PersistentItem tile = persistent_item<Schedule>(shared.tile_prefix, p.offsets, item);
            asm volatile("bar.sync 2, %0;" : : "n"(Schedule::kProducerThreads) : "memory");
            for (int row = producer; row < Schedule::kBlockM; row += Schedule::kProducerThreads) {
                const int index = tile.row_begin + row;
                shared.source_rows[row] = index < tile.row_end ? (p.sorted[index] >> p.a_row_shift) : -1;
            }
            asm volatile("bar.sync 2, %0;" : : "n"(Schedule::kProducerThreads) : "memory");
            const int weight_begin = tile.expert * kGateUpRows + tile.n_begin;
            for (int k_tile = 0; k_tile < kKTiles; ++k_tile, ++fill) {
                const int stage = fill % Schedule::kStages;
                nvfp4_mbarrier_wait(&shared.empty[stage], 1U ^ ((fill / Schedule::kStages) & 1U));
                persistent_fill_stage<Schedule, K>(p, shared, stage, k_tile, weight_begin, producer);
                cp_async_arrive_noinc(&shared.full[stage]);
            }
        }
        asm volatile("cp.async.wait_all;" : : : "memory");
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" : : : "memory");

    // Consumer warps.
    const int warp          = thread >> 5;
    const int warp_m        = warp / Schedule::kWarpsN;
    const int warp_n        = warp - warp_m * Schedule::kWarpsN;
    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row       = lane >> 2;
    const float rcp_denominator = __frcp_rn(__fmul_rn(1536.0F, *p.weights.outer_den));
    int fill = 0;
    for (int item = static_cast<int>(blockIdx.x); item < total_items; item += static_cast<int>(gridDim.x)) {
        const PersistentItem tile = persistent_item<Schedule>(shared.tile_prefix, p.offsets, item);
        // 16-row groups of this warp that hold routed rows (warp-uniform).
        const int groups = (tile.row_end - tile.row_begin + 15) >> 4;
        const int active = min(max(groups - warp_m * Schedule::kMmaM, 0), Schedule::kMmaM);
        float accumulators[Schedule::kMmaM][Schedule::kMmaN][4] = {};
        for (int k_tile = 0; k_tile < kKTiles; ++k_tile, ++fill) {
            const int stage = fill % Schedule::kStages;
            nvfp4_mbarrier_wait(&shared.full[stage], (fill / Schedule::kStages) & 1U);
            if (active > 0) {
#pragma unroll
                for (int local_k64 = 0; local_k64 < Schedule::kK64PerStage; ++local_k64) {
                    unsigned a_fragments[Schedule::kMmaM][4];
                    unsigned b_fragments[Schedule::kMmaN][2];
                    unsigned a_scales[Schedule::kMmaM];
                    unsigned b_scales[Schedule::kMmaN];
#pragma unroll
                    for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
                        if (mma_m < active) {
                            const int row = warp_m * Schedule::kWarpM + mma_m * 16 + a_row_offset;
                            const auto* address = shared.a_codes[stage] + row * Schedule::kCodeRowBytes +
                                                  persistent_swizzled_byte(row, local_k64 * 32 + a_column_byte);
                            ldmatrix_x4(a_fragments[mma_m][0], a_fragments[mma_m][1], a_fragments[mma_m][2],
                                        a_fragments[mma_m][3], smem_addr(address));
                            const int scale_row = warp_m * Schedule::kWarpM + mma_m * 16 + sfa_row;
                            a_scales[mma_m] = shared.a_scale4[stage][scale_row * Schedule::kK64PerStage + local_k64];
                        }
                    }
#pragma unroll
                    for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
                        const int row = warp_n * Schedule::kWarpN + mma_n * 8 + b_row_offset;
                        const auto* address = shared.b_codes[stage] + row * Schedule::kCodeRowBytes +
                                              persistent_swizzled_byte(row, local_k64 * 32 + b_column_byte);
                        ldmatrix_x2(b_fragments[mma_n][0], b_fragments[mma_n][1], smem_addr(address));
                        const int scale_row    = warp_n * Schedule::kWarpN + mma_n * 8 + sfb_row;
                        const int row_tile     = scale_row >> 7;
                        const int row_in_tile  = scale_row & 127;
                        b_scales[mma_n] = load_vec<unsigned>(
                            shared.b_scales[stage] +
                            ((row_tile * Schedule::kK64PerStage + local_k64) * 32 + (row_in_tile & 31)) * 16 +
                            (row_in_tile >> 5) * 4);
                    }
#pragma unroll
                    for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
                        if (mma_m < active) {
#pragma unroll
                            for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
                                mma_nvfp4_e4m3(accumulators[mma_m][mma_n][0], accumulators[mma_m][mma_n][1],
                                               accumulators[mma_m][mma_n][2], accumulators[mma_m][mma_n][3],
                                               a_fragments[mma_m][0], a_fragments[mma_m][1],
                                               a_fragments[mma_m][2], a_fragments[mma_m][3],
                                               b_fragments[mma_n][0], b_fragments[mma_n][1], a_scales[mma_m],
                                               b_scales[mma_n]);
                            }
                        }
                    }
                }
            }
            __syncwarp();
            if (lane == 0) { nvfp4_mbarrier_arrive(&shared.empty[stage]); }
        }

        // Epilogue straight from the accumulators (the producer is already filling the next item).
        const float weight_amax   = p.weights.amax[tile.expert];
        const int accumulator_row = lane >> 2;
        const int accumulator_col = 2 * (lane & 3);
#pragma unroll
        for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
            if (mma_m >= active) { continue; }
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int index = tile.row_begin + warp_m * Schedule::kWarpM + mma_m * 16 + accumulator_row + 8 * half;
                if (index >= tile.row_end) { continue; }
                const int slot = p.sorted[index];
                store_routed_row<GateUp, Schedule::kMmaN>(p, tile.expert, slot, p.a_amax[slot >> p.a_row_shift], weight_amax,
                                                          rcp_denominator,
                                                          tile.n_begin + warp_n * Schedule::kWarpN + accumulator_col,
                                                          accumulators[mma_m], half);
            }
        }
        if constexpr (!GateUp) {
            if (p.arrivals != nullptr) {
                combine_arrive<Schedule::kBlockN, Schedule::kConsumerThreads>(
                    p, Schedule::kBlockM, tile.n_begin,
                    [&](int r) { return tile.row_begin + r < tile.row_end ? p.sorted[tile.row_begin + r] : -1; },
                    shared.done, &shared.done_count, thread,
                    [] { asm volatile("bar.sync 1, %0;" : : "n"(Schedule::kConsumerThreads) : "memory"); });
            }
        }
    }
#endif
}

template <class Schedule, int K, bool GateUp, bool FusedYQuant = false>
void launch_schedule(const GemmParams& params, std::int32_t slots, cudaStream_t stream) {
    const int max_tiles = std::min(kExperts, slots) + slots / Schedule::kBlockM;
    if (max_tiles == 0) { return; }
    const dim3 grid(static_cast<unsigned>(max_tiles), kGateUpRows / Schedule::kBlockN);
    const cudaError_t status = pdl::launch_dependent({grid, dim3(Schedule::kThreads), 0, stream},
                                                     nvfp4_expert_gemm_kernel<Schedule, K, GateUp, FusedYQuant>, params);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("inkling_moe: nvfp4 expert gemm: ") + cudaGetErrorString(status));
    }
}

using PersistentWide = PersistentSchedule<128, 256, 2, 4, 3>; ///< warp tiles 64 x 64
using PersistentAlt  = PersistentSchedule<128, 128, 2, 4, 4>; ///< warp tiles 64 x 32 (timing knob)
using PersistentTall = PersistentSchedule<256, 128, 4, 2, 3>; ///< warp tiles 64 x 64 (timing knob)

int multiprocessors() {
    static int cached[64] = {};
    int device            = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 64) { return 1; }
    if (cached[device] == 0) {
        int count = 0;
        if (cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || count <= 0) {
            count = 1;
        }
        cached[device] = count;
    }
    return cached[device];
}

template <class Schedule, int K, bool GateUp>
void launch_persistent(const GemmParams& params, std::int32_t slots, cudaStream_t stream) {
    const int max_tiles = std::min(kExperts, slots) + slots / Schedule::kBlockM;
    if (max_tiles == 0) { return; }
    const int items = max_tiles * Schedule::kNTiles;
    const dim3 grid(static_cast<unsigned>(std::min(items, multiprocessors())));
    constexpr std::size_t kShared = sizeof(PersistentStorage<Schedule>);
    auto* kernel = nvfp4_expert_gemm_persistent<Schedule, K, GateUp>;
    cudaError_t status = ::sinfer::ops::set_func_attribute_per_device(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kShared));
    if (status == cudaSuccess) {
        status = pdl::launch_dependent({grid, dim3(Schedule::kThreads), kShared, stream}, kernel, params);
    }
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("inkling_moe: nvfp4 persistent expert gemm: ") + cudaGetErrorString(status));
    }
}

/// Prefill kernel choice. TIMING KNOB (removed once measured): SUROGATE_INKLING_NVFP4_PREFILL=old
/// (the v1 per-tile kernel) | new (persistent 128 x 256 for both) | alt (persistent, 128 x 128 tiles) |
/// tall (persistent, 256 x 128); default: gate_up persistent 128 x 256, down per-tile.
/// Every choice computes the same bits.
int prefill_variant() {
    static const int variant = [] {
        const char* value = std::getenv("SUROGATE_INKLING_NVFP4_PREFILL");
        if (value == nullptr) { return 0; }
        if (std::strcmp(value, "new") == 0) { return 4; }
        if (std::strcmp(value, "old") == 0) { return 1; }
        if (std::strcmp(value, "alt") == 0) { return 2; }
        if (std::strcmp(value, "tall") == 0) { return 3; }
        return 0;
    }();
    return variant;
}

template <int K, bool GateUp>
void launch_prefill(const GemmParams& params, std::int32_t slots, cudaStream_t stream) {
    switch (prefill_variant()) {
    case 1: launch_schedule<WideSchedule, K, GateUp>(params, slots, stream); return;
    case 2: launch_persistent<PersistentAlt, K, GateUp>(params, slots, stream); return;
    case 3: launch_persistent<PersistentTall, K, GateUp>(params, slots, stream); return;
    case 4: launch_persistent<PersistentWide, K, GateUp>(params, slots, stream); return;
    default:
        // Measured 12:03 (T = 4,096, cold): gate_up persistent 514 us vs per-tile 549; down per-tile 332 vs
        // persistent 367 (the fused combine overlaps better with two CTAs per SM).
        if constexpr (GateUp) {
            launch_persistent<PersistentWide, K, GateUp>(params, slots, stream);
        } else {
            launch_schedule<WideSchedule, K, GateUp>(params, slots, stream);
        }
        return;
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
        launch_prefill<kHidden, true>(params, slots, stream);
    }
}

void launch_nvfp4_down(const GemmParams& params, std::int32_t slots, GemmSchedule schedule,
                       cudaStream_t stream) {
    require(params);
    if (schedule == GemmSchedule::Narrow) {
        // The y quantisation runs in this kernel's prologue (a_bf16 = y rows); w2 streams before the wait.
        if (params.a_bf16 == nullptr) { throw std::invalid_argument("inkling_moe: narrow NVFP4 down needs the y rows"); }
        launch_schedule<NarrowSchedule, kIntermediate, false, true>(params, slots, stream);
    } else {
        launch_prefill<kIntermediate, false>(params, slots, stream);
    }
}

std::int32_t nvfp4_tile_rows(GemmSchedule schedule) {
    return schedule == GemmSchedule::Narrow ? NarrowSchedule::kBlockM : PersistentWide::kBlockM;
}

} // namespace sinfer::ops::detail::inkling
