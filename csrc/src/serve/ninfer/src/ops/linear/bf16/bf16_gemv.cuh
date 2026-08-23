#pragma once

#include "ops/linear/bf16/bf16_config.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int Values>
struct alignas(Values* static_cast<int>(sizeof(__nv_bfloat16))) Bf16GemvPack {
    static_assert(Values == 4 || Values == 8 || Values == 16);
    std::uint32_t words[Values / 2];
};

static_assert(sizeof(Bf16GemvPack<4>) == 8);
static_assert(sizeof(Bf16GemvPack<8>) == 16);
static_assert(sizeof(Bf16GemvPack<16>) == 32);

template <int Values>
__device__ __forceinline__ Bf16GemvPack<Values> load_bf16_pack(const __nv_bfloat16* pointer) {
    if constexpr (Values <= 8) {
        return load_vec<Bf16GemvPack<Values>>(pointer);
    } else {
        Bf16GemvPack<Values> result;
        const Bf16GemvPack<8> low  = load_vec<Bf16GemvPack<8>>(pointer);
        const Bf16GemvPack<8> high = load_vec<Bf16GemvPack<8>>(pointer + 8);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            result.words[pair]     = low.words[pair];
            result.words[pair + 4] = high.words[pair];
        }
        return result;
    }
}

template <Bf16WeightCache Cache, int Values>
__device__ __forceinline__ Bf16GemvPack<Values>
load_bf16_weight_pack(const __nv_bfloat16* pointer) {
    if constexpr (Cache == Bf16WeightCache::Default) {
        return load_bf16_pack<Values>(pointer);
    } else if constexpr (Values == 4) {
        uint2 bits;
        asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];\n"
                     : "=r"(bits.x), "=r"(bits.y)
                     : "l"(pointer));
        return load_vec<Bf16GemvPack<Values>>(&bits);
    } else if constexpr (Values == 8) {
        uint4 bits;
        asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
                     : "l"(pointer));
        return load_vec<Bf16GemvPack<Values>>(&bits);
    } else {
        Bf16GemvPack<Values> result;
        const Bf16GemvPack<8> low  = load_bf16_weight_pack<Cache, 8>(pointer);
        const Bf16GemvPack<8> high = load_bf16_weight_pack<Cache, 8>(pointer + 8);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            result.words[pair]     = low.words[pair];
            result.words[pair + 4] = high.words[pair];
        }
        return result;
    }
}

template <int Values, int AccumulatorChains>
__device__ __forceinline__ void accumulate_bf16_packs(const Bf16GemvPack<Values>& weight,
                                                      const Bf16GemvPack<Values>& activation,
                                                      float (&accumulators)[AccumulatorChains]) {
#pragma unroll
    for (int pair = 0; pair < Values / 2; ++pair) {
        const float2 w           = bf16x2_bits_to_float2(weight.words[pair]);
        const float2 x           = bf16x2_bits_to_float2(activation.words[pair]);
        constexpr int kChainMask = AccumulatorChains - 1;
        accumulators[(2 * pair) & kChainMask] =
            fmaf(w.x, x.x, accumulators[(2 * pair) & kChainMask]);
        accumulators[(2 * pair + 1) & kChainMask] =
            fmaf(w.y, x.y, accumulators[(2 * pair + 1) & kChainMask]);
    }
}

struct Bf16ContiguousOutput {
    __nv_bfloat16* data;

    __device__ __forceinline__ void store(std::int32_t parent_row, __nv_bfloat16 value) const {
        data[parent_row] = value;
    }
};

struct Bf16StoreEpilogue {
    template <class Output>
    __device__ __forceinline__ void operator()(const Output& output, std::int32_t parent_row,
                                               float accumulator) const {
        output.store(parent_row, __float2bfloat16_rn(accumulator));
    }
};

template <class Geometry, class Schedule>
struct Bf16GemvSharedStorage {
    static constexpr int kActivationElements =
        Schedule::kActivationAccess == Bf16ActivationAccess::Shared ? Geometry::kInputRows : 8;
    static constexpr int kReductionWarps = Schedule::kWarpsPerRow > 1 ? Schedule::kWarpsPerRow : 1;

    alignas(16) __nv_bfloat16 activation[kActivationElements];
    float partials[Schedule::kRowGroupsPerCta][Schedule::kRowsPerWarp][kReductionWarps];
};

template <class Geometry, class Schedule>
__device__ __forceinline__ const __nv_bfloat16*
prepare_bf16_activation(const __nv_bfloat16* x, Bf16GemvSharedStorage<Geometry, Schedule>& shared) {
    if constexpr (Schedule::kActivationAccess == Bf16ActivationAccess::Direct) {
        return x;
    } else {
        static_assert((Geometry::kInputRows % 8) == 0);
        constexpr int kPacks = Geometry::kInputRows / 8;
        auto* destination    = reinterpret_cast<Bf16GemvPack<8>*>(shared.activation);
        const auto* source   = reinterpret_cast<const Bf16GemvPack<8>*>(x);
        for (int pack = static_cast<int>(threadIdx.x); pack < kPacks; pack += Schedule::kThreads) {
            destination[pack] = source[pack];
        }
        __syncthreads();
        return shared.activation;
    }
}

template <class Geometry, class Schedule>
__device__ __forceinline__ std::int32_t bf16_phase_offset(int phase, int warp_in_row, int lane) {
    constexpr int kValuesPerWarp = kWarpSize * Schedule::kValuesPerLane;
    return phase * Schedule::kWarpsPerRow * kValuesPerWarp + warp_in_row * kValuesPerWarp +
           lane * Schedule::kValuesPerLane;
}

template <class Geometry, class Schedule>
__device__ __forceinline__ Bf16GemvPack<Schedule::kValuesPerLane>
load_bf16_activation_phase(const __nv_bfloat16* activation, int phase, int warp_in_row, int lane) {
    const int offset = bf16_phase_offset<Geometry, Schedule>(phase, warp_in_row, lane);
    return load_bf16_pack<Schedule::kValuesPerLane>(activation + offset);
}

template <class Geometry, class Schedule>
__device__ __forceinline__ Bf16GemvPack<Schedule::kValuesPerLane>
load_bf16_weight_phase(const __nv_bfloat16* weight, int row, int phase, int warp_in_row, int lane) {
    const int offset = bf16_phase_offset<Geometry, Schedule>(phase, warp_in_row, lane);
    return load_bf16_weight_pack<Schedule::kWeightCache, Schedule::kValuesPerLane>(
        weight + static_cast<std::int64_t>(row) * Geometry::kInputRows + offset);
}

template <class Schedule, int Phases>
__device__ __forceinline__ int bf16_phase_index(int iteration, int row0) {
    if constexpr (Schedule::kPhaseOrder == Bf16PhaseOrder::Sequential) {
        return iteration;
    } else {
        const int row_group = row0 / Schedule::kRowsPerWarp;
        int phase           = iteration + (row_group * Schedule::kPhaseStride) % Phases;
        if (phase >= Phases) { phase -= Phases; }
        return phase;
    }
}

template <class Geometry, class Schedule>
__device__ __forceinline__ void compute_bf16_gemv_rows(
    const __nv_bfloat16* activation, const __nv_bfloat16* weight, int row0, int warp_in_row,
    int lane, float (&accumulators)[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains]) {
    constexpr int kValuesPerPhase = Schedule::kWarpsPerRow * kWarpSize * Schedule::kValuesPerLane;
    static_assert((Geometry::kInputRows % kValuesPerPhase) == 0);
    constexpr int kPhases = Geometry::kInputRows / kValuesPerPhase;
    using Pack            = Bf16GemvPack<Schedule::kValuesPerLane>;

    if constexpr (Schedule::kPrefetchDepth == 1) {
#pragma unroll Schedule::kPhaseUnroll
        for (int iteration = 0; iteration < kPhases; ++iteration) {
            const int phase     = bf16_phase_index<Schedule, kPhases>(iteration, row0);
            const Pack x_values = load_bf16_activation_phase<Geometry, Schedule>(activation, phase,
                                                                                 warp_in_row, lane);
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                const Pack w_values = load_bf16_weight_phase<Geometry, Schedule>(
                    weight, row0 + local_row, phase, warp_in_row, lane);
                accumulate_bf16_packs(w_values, x_values, accumulators[local_row]);
            }
        }
    } else {
        const int first_phase = bf16_phase_index<Schedule, kPhases>(0, row0);
        Pack current_x = load_bf16_activation_phase<Geometry, Schedule>(activation, first_phase,
                                                                        warp_in_row, lane);
        Pack current_w[Schedule::kRowsPerWarp];
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            current_w[local_row] = load_bf16_weight_phase<Geometry, Schedule>(
                weight, row0 + local_row, first_phase, warp_in_row, lane);
        }

#pragma unroll Schedule::kPhaseUnroll
        for (int iteration = 0; iteration < kPhases; ++iteration) {
            Pack next_x;
            Pack next_w[Schedule::kRowsPerWarp];
            if (iteration + 1 < kPhases) {
                const int next_phase = bf16_phase_index<Schedule, kPhases>(iteration + 1, row0);
                next_x = load_bf16_activation_phase<Geometry, Schedule>(activation, next_phase,
                                                                        warp_in_row, lane);
#pragma unroll
                for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                    next_w[local_row] = load_bf16_weight_phase<Geometry, Schedule>(
                        weight, row0 + local_row, next_phase, warp_in_row, lane);
                }
            }
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                accumulate_bf16_packs(current_w[local_row], current_x, accumulators[local_row]);
            }
            if (iteration + 1 < kPhases) {
                current_x = next_x;
#pragma unroll
                for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                    current_w[local_row] = next_w[local_row];
                }
            }
        }
    }
}

template <class Geometry, class Schedule, class Output, class Epilogue = Bf16StoreEpilogue>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void bf16_gemv_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight, Output output,
    Epilogue epilogue = {}) {
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);

    __shared__ Bf16GemvSharedStorage<Geometry, Schedule> shared;
    const __nv_bfloat16* activation = prepare_bf16_activation<Geometry, Schedule>(x, shared);

    const int lane        = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp        = static_cast<int>(threadIdx.x) / kWarpSize;
    const int row_group   = warp / Schedule::kWarpsPerRow;
    const int warp_in_row = warp % Schedule::kWarpsPerRow;
    const int cta_row0    = static_cast<int>(blockIdx.x) * Schedule::kRowsPerCta;
    const int row0        = cta_row0 + row_group * Schedule::kRowsPerWarp;
    float accumulators[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains] = {};

    compute_bf16_gemv_rows<Geometry, Schedule>(activation, weight, row0, warp_in_row, lane,
                                               accumulators);

    float warp_totals[Schedule::kRowsPerWarp];
#pragma unroll
    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
        float total = 0.0F;
#pragma unroll
        for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
            total += accumulators[local_row][chain];
        }
        warp_totals[local_row] = warp_reduce_sum(total);
    }

    if constexpr (Schedule::kWarpsPerRow == 1) {
        if (lane == 0) {
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                epilogue(output, row0 + local_row, warp_totals[local_row]);
            }
        }
    } else {
        if (lane == 0) {
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                shared.partials[row_group][local_row][warp_in_row] = warp_totals[local_row];
            }
        }
        __syncthreads();
        if (warp_in_row == 0) {
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                const float partial = lane < Schedule::kWarpsPerRow
                                          ? shared.partials[row_group][local_row][lane]
                                          : 0.0F;
                const float total   = warp_reduce_sum(partial);
                if (lane == 0) { epilogue(output, row0 + local_row, total); }
            }
        }
    }
}

} // namespace ninfer::ops::detail
