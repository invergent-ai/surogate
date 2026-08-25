#pragma once

// surogate vendor patch (PATCHES.md #22): W4 decode kernel over the derived
// NVFP4 plane (fp4 profile, stage 2).
//
// Twin of w8_k2048_decode.cuh with half the weight traffic: one warp per
// output row, 256 values per phase, 4 e2m1 code bytes per lane per phase
// (one u32 load vs the W8 kernel's uint2), ue4m3 per-16 block scales
// shuffled from the low half-warp, and the f32 per-row scale applied once
// at the end. Decode stays A16 (BF16 activations) — at T=1 the weight
// bytes are the bottleneck, not compute.
//
// Graph discipline: the engine's pre-capture warmup decode derives the
// plane eagerly (program_impl runs one eager decode + synchronize before
// capturing), so capture bakes the plane pointers; w4fp4_plane_for
// refuses to derive or wait on events while a stream is capturing.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/w8/w8_k2048_decode.cuh"  // W8DecodeStoreEpilogue, shared Output types

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

__device__ __forceinline__ float w4fp4_e2m1_decode(unsigned nib) {
    // Direct fp32 bit assembly (e2m1 is a float format): m >= 2 -> exponent
    // 126 + (m >> 1) with mantissa bit (m & 1); m in {0, 1} -> m * 0.5f.
    const unsigned m    = nib & 7u;
    const unsigned bits = (m >= 2u ? ((126u + (m >> 1)) << 23) | ((m & 1u) << 22)
                                   : m * 0x3F000000u) |
                          ((nib & 8u) << 28);
    return __uint_as_float(bits);
}

__device__ __forceinline__ float w4fp4_ue4m3_decode(unsigned byte) {
    const int e = static_cast<int>(byte >> 3) & 0xF;
    const int m = static_cast<int>(byte) & 7;
    if (e == 0) { return ldexpf(static_cast<float>(m) / 8.0f, -6); }
    return ldexpf(1.0f + static_cast<float>(m) / 8.0f, e - 7);
}

template <std::int32_t Rows, std::int32_t RowsPerCta, class Output,
          class Epilogue = W8DecodeStoreEpilogue, std::int32_t K = 2048>
__global__ __launch_bounds__(RowsPerCta * 32, 2) void w4fp4_decode_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ sf, const float* __restrict__ row_scales, Output output,
    Epilogue epilogue = {}) {
    static_assert(Rows > 0 && RowsPerCta > 0 && (Rows % RowsPerCta) == 0);
    static_assert(RowsPerCta * 32 <= 1024);
    constexpr int kK                 = K;
    constexpr int kValuesPerLane     = 8;
    constexpr int kValuesPerPhase    = 32 * kValuesPerLane;  // 256
    constexpr int kSfPerPhase        = kValuesPerPhase / 16; // 16 ue4m3 bytes
    constexpr int kPhases            = kK / kValuesPerPhase;
    constexpr unsigned kFullWarpMask = 0xffffffffu;
    static_assert(kK % kValuesPerPhase == 0);

    const int lane               = static_cast<int>(threadIdx.x) & 31;
    const int warp               = static_cast<int>(threadIdx.x) >> 5;
    const int cta_row0           = static_cast<int>(blockIdx.x) * RowsPerCta;
    const int row                = cta_row0 + warp;
    const std::uint8_t* code_row = codes + static_cast<std::int64_t>(row) * (kK / 2);
    const std::uint8_t* sf_row   = sf + static_cast<std::int64_t>(row) * (kK / 16);

    float accumulator = 0.0f;
#pragma unroll
    for (int phase = 0; phase < kPhases; ++phase) {
        // Lane l's 8 values live entirely inside 16-group (l >> 1).
        unsigned sf_bits = 0;
        if (lane < kSfPerPhase) { sf_bits = sf_row[phase * kSfPerPhase + lane]; }
        sf_bits           = __shfl_sync(kFullWarpMask, sf_bits, lane >> 1);
        const float scale = w4fp4_ue4m3_decode(sf_bits);

        const int phase_k    = phase * kValuesPerPhase + lane * kValuesPerLane;
        const std::uint32_t packed =
            *reinterpret_cast<const std::uint32_t*>(code_row + phase_k / 2);
        const uint4 values = load_vec<uint4>(x + phase_k);
        const float2 x0    = bf16x2_bits_to_float2(values.x);
        const float2 x1    = bf16x2_bits_to_float2(values.y);
        const float2 x2    = bf16x2_bits_to_float2(values.z);
        const float2 x3    = bf16x2_bits_to_float2(values.w);

        float partial = 0.0f;
        partial       = fmaf(w4fp4_e2m1_decode(packed & 0xFu), x0.x, partial);
        partial       = fmaf(w4fp4_e2m1_decode((packed >> 4) & 0xFu), x0.y, partial);
        partial       = fmaf(w4fp4_e2m1_decode((packed >> 8) & 0xFu), x1.x, partial);
        partial       = fmaf(w4fp4_e2m1_decode((packed >> 12) & 0xFu), x1.y, partial);
        partial       = fmaf(w4fp4_e2m1_decode((packed >> 16) & 0xFu), x2.x, partial);
        partial       = fmaf(w4fp4_e2m1_decode((packed >> 20) & 0xFu), x2.y, partial);
        partial       = fmaf(w4fp4_e2m1_decode((packed >> 24) & 0xFu), x3.x, partial);
        partial       = fmaf(w4fp4_e2m1_decode((packed >> 28) & 0xFu), x3.y, partial);
        accumulator   = fmaf(partial, scale, accumulator);
    }

    accumulator *= row_scales[row];
    accumulator = warp_reduce_sum(accumulator);
    if (lane == 0) { epilogue(output, cta_row0, row, accumulator); }
}

// surogate vendor patch (PATCHES.md #28): batched twin for T=2..16. Weight
// codes and block scales are loaded and decoded ONCE per phase and fanned
// across up to MaxTokens resident accumulators, so the round keeps the W4
// plane's halved weight traffic under batching (activations are tiny and
// L2-resident across the row warps). `tokens` is the runtime batch
// (2..MaxTokens); pick the smallest MaxTokens bucket covering it to bound
// register pressure.
template <std::int32_t Rows, std::int32_t RowsPerCta, class Output, std::int32_t MaxTokens,
          std::int32_t K = 2048>
__global__ __launch_bounds__(RowsPerCta * 32, 2) void w4fp4_decode_batch_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ sf, const float* __restrict__ row_scales, Output output,
    std::int32_t tokens) {
    static_assert(Rows > 0 && RowsPerCta > 0 && (Rows % RowsPerCta) == 0);
    static_assert(MaxTokens >= 2 && MaxTokens <= 16); // 32-bucket rejected by measurement (PATCHES.md #29): register spill loses to the tile
    constexpr int kK                 = K;
    constexpr int kValuesPerLane     = 8;
    constexpr int kValuesPerPhase    = 32 * kValuesPerLane;
    constexpr int kSfPerPhase        = kValuesPerPhase / 16;
    constexpr int kPhases            = kK / kValuesPerPhase;
    constexpr unsigned kFullWarpMask = 0xffffffffu;
    static_assert(kK % kValuesPerPhase == 0);

    const int lane               = static_cast<int>(threadIdx.x) & 31;
    const int warp               = static_cast<int>(threadIdx.x) >> 5;
    const int cta_row0           = static_cast<int>(blockIdx.x) * RowsPerCta;
    const int row                = cta_row0 + warp;
    const std::uint8_t* code_row = codes + static_cast<std::int64_t>(row) * (kK / 2);
    const std::uint8_t* sf_row   = sf + static_cast<std::int64_t>(row) * (kK / 16);

    float accumulator[MaxTokens];
#pragma unroll
    for (int t = 0; t < MaxTokens; ++t) { accumulator[t] = 0.0f; }

#pragma unroll
    for (int phase = 0; phase < kPhases; ++phase) {
        unsigned sf_bits = 0;
        if (lane < kSfPerPhase) { sf_bits = sf_row[phase * kSfPerPhase + lane]; }
        sf_bits           = __shfl_sync(kFullWarpMask, sf_bits, lane >> 1);
        const float scale = w4fp4_ue4m3_decode(sf_bits);

        const int phase_k = phase * kValuesPerPhase + lane * kValuesPerLane;
        const std::uint32_t packed =
            *reinterpret_cast<const std::uint32_t*>(code_row + phase_k / 2);
        float weightv[kValuesPerLane];
#pragma unroll
        for (int i = 0; i < kValuesPerLane; ++i) {
            weightv[i] = w4fp4_e2m1_decode((packed >> (4 * i)) & 0xFu);
        }

        for (int t = 0; t < tokens; ++t) {
            const uint4 values = load_vec<uint4>(x + static_cast<std::int64_t>(t) * kK + phase_k);
            const float2 x0    = bf16x2_bits_to_float2(values.x);
            const float2 x1    = bf16x2_bits_to_float2(values.y);
            const float2 x2    = bf16x2_bits_to_float2(values.z);
            const float2 x3    = bf16x2_bits_to_float2(values.w);
            float partial      = 0.0f;
            partial            = fmaf(weightv[0], x0.x, partial);
            partial            = fmaf(weightv[1], x0.y, partial);
            partial            = fmaf(weightv[2], x1.x, partial);
            partial            = fmaf(weightv[3], x1.y, partial);
            partial            = fmaf(weightv[4], x2.x, partial);
            partial            = fmaf(weightv[5], x2.y, partial);
            partial            = fmaf(weightv[6], x3.x, partial);
            partial            = fmaf(weightv[7], x3.y, partial);
            accumulator[t]     = fmaf(partial, scale, accumulator[t]);
        }
    }

    const float row_scale = row_scales[row];
    for (int t = 0; t < tokens; ++t) {
        float value = warp_reduce_sum(accumulator[t] * row_scale);
        if (lane == 0) {
            *output.tile(cta_row0).at(row, t) = __float2bfloat16_rn(value);
        }
    }
}

} // namespace ninfer::ops::detail
