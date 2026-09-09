#pragma once

#include "family/impl/load/host_bank.h"

namespace sinfer::family {
// Decode a stored expert row, preserving segment formats and column permutations.
void decode_host_row(const HostObjectPlan& source, std::int64_t row, float* output);
void quantize_host_row(const float* values, std::int32_t columns,
                       std::int8_t* codes, std::uint16_t* scales);
} // namespace sinfer::family
