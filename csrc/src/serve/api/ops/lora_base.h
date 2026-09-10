#pragma once
#include "core/tensor.h"
#include <vector>

namespace sinfer::ops {
// A serving bank may be assembled from rows of several represented weights.
struct LoraBaseView {
    Weight weight;
    int row_offset          = 0;
    int output_offset       = 0;
    int rows                = 0;
    bool transpose          = false;
    const float* multiplier = nullptr;
};

// Read-only, bounded-memory load-time reduction. Does not modify the base weight.
std::vector<float> lora_weight_norms(const std::vector<LoraBaseView>& base,
                                     const std::vector<std::uint16_t>& a,
                                     const std::vector<std::uint16_t>& b, int rank, int in, int out,
                                     float scale);
} // namespace sinfer::ops
