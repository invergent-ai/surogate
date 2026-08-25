#pragma once

#include "ops/linear_attention/gated_delta_net/common.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net::chunked {

inline constexpr int BT    = kChunkSize;
inline constexpr int BC    = 16;
inline constexpr int MMA_M = 16;
inline constexpr int MMA_N = 8;
inline constexpr int MMA_K = 8;

static_assert(BT % BC == 0, "BT must be a multiple of BC");
static_assert(BT % MMA_M == 0, "BT must be a multiple of MMA_M");

} // namespace ninfer::ops::detail::gated_delta_net::chunked
