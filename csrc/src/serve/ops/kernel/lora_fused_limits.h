#pragma once

// The fused delta kernel's compile-time limits, visible to C++ callers.
namespace sinfer::ops {
inline constexpr int kLoraFusedPairLimit = 3;
inline constexpr int kLoraFusedRankLimit = 64;
} // namespace sinfer::ops
