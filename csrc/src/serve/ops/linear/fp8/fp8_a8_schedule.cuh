#pragma once

#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_config.h"

namespace sinfer::ops::detail {

template <class Geometry>
struct Fp8LinearA8ProductionSchedule;

using Fp8LinearA8BatchSchedule = Fp8MmaSchedule<32, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                                Fp8MmaFragmentPipeline::PingPong,
                                                Fp8MmaRaster::TokenFast>;

// Batch (decode) schedule, per geometry.
//
// The launch grid is (output_rows / kBlockRows) * token_tiles, so at decode
// widths the row tiling alone decides occupancy. With 128-row tiles an N=5120
// weight produces 40 CTAs on a 170-SM device — a quarter of the machine — and
// measured 324 GB/s against the 894 GB/s the N=16384 shapes reach with the same
// schedule. Those two shapes (o_proj and down_proj on the 27B) are ~8% of its
// device time, so the tiling, not the kernel, is what is costing them.
//
// Narrower rows cost nothing at decode: the weight is read once either way, and
// more CTAs is strictly better while the device is this far from full.
template <class Geometry>
struct Fp8LinearA8BatchScheduleFor {
    using Type = Fp8LinearA8BatchSchedule;
};

template <>
struct Fp8LinearA8BatchScheduleFor<Fp8Residual6144Geometry> {
    using Type = Fp8MmaSchedule<32, 64, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8BatchScheduleFor<Fp8Residual17408Geometry> {
    using Type = Fp8MmaSchedule<32, 64, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8AttnInputGeometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8GdnInputGeometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8MlpGateUpGeometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8Residual6144Geometry> {
    // 64-row tiles, not 128. The launch grid is
    // (output_rows / kBlockRows) * token_tiles, so at N=5120 a 128-row tile
    // yields 40 CTAs on a 170-SM device — a quarter of the machine — and these
    // shapes measured 324-334 GB/s where the N=16384 shapes reach 894 GB/s on
    // the same schedule. The batch schedule is not the fix: the 27B serves 48
    // lanes, so decode t exceeds the 32-token batch band and lands here.
    using Type = Fp8MmaSchedule<64, 64, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8Residual17408Geometry> {
    // 64-row tiles, not 128. The launch grid is
    // (output_rows / kBlockRows) * token_tiles, so at N=5120 a 128-row tile
    // yields 40 CTAs on a 170-SM device — a quarter of the machine — and these
    // shapes measured 324-334 GB/s where the N=16384 shapes reach 894 GB/s on
    // the same schedule. The batch schedule is not the fix: the 27B serves 48
    // lanes, so decode t exceeds the 32-token batch band and lands here.
    using Type = Fp8MmaSchedule<64, 64, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

} // namespace sinfer::ops::detail
