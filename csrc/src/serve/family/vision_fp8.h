#pragma once
// MXFP8 copies of a vision tower's block projections (--vision-fp8, api/ops/vision_fp8.h).
//
// Built once per loaded tower when the program starts, beside the BF16 weights it reads from
// (which stay: every route that is not the MXFP8 one keeps working). The program holds the
// copies; a VisionContext finds them by the tower they were built from.

#include "api/family/vision.h"
#include "api/family/vision_geometry.h"
#include "ops/linear/mxfp8/mxfp8.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace sinfer::family {

struct VisionFp8Block {
    ops::detail::Mxfp8Matrix qkv, projection, fc1, fc2;
};

struct VisionFp8Weights {
    std::vector<VisionFp8Block> blocks;
    void* storage     = nullptr; ///< one allocation holding every block's codes and scales
    std::size_t bytes = 0;
    int device        = 0;
    VisionFp8Weights() = default;
    VisionFp8Weights(const VisionFp8Weights&)            = delete;
    VisionFp8Weights& operator=(const VisionFp8Weights&) = delete;
    ~VisionFp8Weights();
};

/// Quantizes `vision`'s block projections to MXFP8 on the current device and registers the
/// copies for find_vision_fp8. Returns null -- after saying why on stderr -- when the device
/// cannot run MXFP8, a projection is not device-resident contiguous BF16 (--offload-vision),
/// cuBLASLt offers no algorithm for the tower's shapes, or the copies do not fit.
[[nodiscard]] std::shared_ptr<const VisionFp8Weights>
prepare_vision_fp8(const VisionWeights& vision, const VisionGeometry& geometry, cudaStream_t stream);

/// The copies prepare_vision_fp8 built from `vision`, while their owner holds them; else null.
[[nodiscard]] std::shared_ptr<const VisionFp8Weights> find_vision_fp8(const VisionWeights* vision);

} // namespace sinfer::family
