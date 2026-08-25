#pragma once

#include "ops/linear/bf16/bf16_gemm_mma.cuh"

namespace ninfer::ops::detail {

// Measured large-T production schedule for the BF16 computation core. Geometry remains a template
// argument so an exact problem can replace any tile, pipeline, cache, raster, or fragment choice
// without changing either Linear or semantic-Op dispatch.
template <class Geometry>
using Bf16MmaProductionSchedule =
    Bf16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                    Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;

} // namespace ninfer::ops::detail
