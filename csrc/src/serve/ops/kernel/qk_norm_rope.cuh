#pragma once

// Implements: api/ops/qk_norm_rope.h
// The per-head query/key RMSNorm and the RoPE after it in one kernel, as vLLM fuses them: a warp
// per head and a block per (token, eight heads), the rope kernel's grid. A warp holds its head the
// way the rmsnorm warp kernels do (BF16x2 pairs lane + 32k), reduces it the way the rmsnorm kernel
// the separate op would dispatch does, rounds the normalised head to BF16 as that op writes it,
// and rotates it with the coefficients the rope kernel the separate op would dispatch computes.
// A rotated pair's partner channel is `rotary_dim / 2` away, i.e. `rotary_dim / 4` BF16x2 slots:
// a multiple of 32 puts it in the same lane (another register), and a power of two below 32 in
// lane `lane ^ (rotary_dim / 4)` of the first register. The output is bit for bit the two ops'.

#include "ops/common/warp.cuh"
#include "ops/kernel/rmsnorm.cuh"
#include "ops/kernel/rope.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace sinfer::ops {

/// Which rmsnorm kernel's reduction the norm reproduces: rmsnorm_d128_bf16x2_kernel's (head dim
/// 128, plain gain) or rmsnorm_warp_bf16x2_kernel's (head dims 64 to 256).
enum class QkNormForm { D128, Warp };

/// Which rope kernel's coefficients the rotation reproduces: rope_generic_kernel's, or
/// rope_fixed_kernel's Text1D / TextMrope tables.
enum class QkRopeAngles { Generic, Text1D, TextMrope };

inline constexpr int kQkNormRopeWarps = 8;

template <RmsEpilogue Epilogue, QkNormForm Form, QkRopeAngles Angles>
static __global__ __launch_bounds__(kQkNormRopeWarps * 32) void qk_norm_rope_kernel(
        const __nv_bfloat162* q, const __nv_bfloat162* k, const __nv_bfloat162* q_weight,
        const __nv_bfloat162* k_weight, __nv_bfloat162* q_out, __nv_bfloat162* k_out,
        const std::int32_t* positions, std::int32_t axes, std::int32_t head_dim,
        std::int32_t rotary_dim, std::int32_t active_pairs, float theta, std::int32_t height_pairs,
        std::int32_t width_pairs, std::int32_t q_heads, std::int32_t k_heads, std::int32_t tokens,
        float eps) {
    constexpr int kMaxSlots = 4; // BF16x2 slots a lane holds: head dims up to 256
    const int token = static_cast<int>(blockIdx.x);
    if (token >= tokens) { return; }
    const int lane          = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int combined_head = static_cast<int>(blockIdx.y) * kQkNormRopeWarps + (static_cast<int>(threadIdx.x) >> 5);
    const bool active       = combined_head < q_heads + k_heads;
    const int slots         = head_dim / 2;
    const int half          = rotary_dim / 2;

    // The head and its gain go out first, so the loads are in flight while the block derives the
    // angles (in double on the generic path).
    __nv_bfloat162 values[kMaxSlots];
    float2 gains[kMaxSlots];
    __nv_bfloat162* row = nullptr;
    if (active) {
        const bool is_q            = combined_head < q_heads;
        const int head             = is_q ? combined_head : combined_head - q_heads;
        const std::int64_t offset  = (static_cast<std::int64_t>(token) * (is_q ? q_heads : k_heads) + head) * slots;
        const __nv_bfloat162* in   = (is_q ? q : k) + offset;
        const __nv_bfloat162* gain = is_q ? q_weight : k_weight;
        row                        = (is_q ? q_out : k_out) + offset;
#pragma unroll
        for (int s = 0; s < kMaxSlots; ++s) {
            const int slot = lane + s * kWarpSize;
            if (slot < slots) {
                values[s] = in[slot];
                gains[s]  = rmsnorm_gain2<Epilogue>(gain, slot);
            }
        }
    }

    __shared__ float cos_cache[kRopeMaxHalf];
    __shared__ float sin_cache[kRopeMaxHalf];
    for (int pair = static_cast<int>(threadIdx.x); pair < half; pair += static_cast<int>(blockDim.x)) {
        if constexpr (Angles == QkRopeAngles::Generic) {
            generic_pair_sincos(positions, axes, tokens, token, pair, head_dim, rotary_dim,
                                active_pairs, theta, height_pairs, width_pairs, 1.0F,
                                &sin_cache[pair], &cos_cache[pair]);
        } else {
            fixed_sincos<Angles == QkRopeAngles::Text1D ? RopeKernelMode::Text1D : RopeKernelMode::TextMrope>(
                positions, tokens, token, pair, &sin_cache[pair], &cos_cache[pair]);
        }
    }

    // The norm, written as the kernel the separate op dispatches writes it, so the sum rounds alike.
    float2 normed[kMaxSlots];
    if (active) {
        float inv = 0.0f;
        if constexpr (Form == QkNormForm::D128) {
            const float2 x0 = __bfloat1622float2(values[0]);
            const float2 x1 = __bfloat1622float2(values[1]);
            float sum       = x0.x * x0.x + x0.y * x0.y + x1.x * x1.x + x1.y * x1.y;
            sum             = warp_reduce_sum(sum);
            inv             = lane == 0 ? rsqrtf(sum * (1.0f / 128.0f) + eps) : 0.0f;
        } else {
            float sum = 0.0f;
#pragma unroll
            for (int s = 0; s < kMaxSlots; ++s) {
                if (lane + s * kWarpSize < slots) {
                    const float2 xf = __bfloat1622float2(values[s]);
                    sum += xf.x * xf.x + xf.y * xf.y;
                }
            }
            sum = warp_reduce_sum(sum);
            inv = lane == 0 ? rsqrtf(sum / static_cast<float>(head_dim) + eps) : 0.0f;
        }
        inv = __shfl_sync(kFullWarpMask, inv, 0);
#pragma unroll
        for (int s = 0; s < kMaxSlots; ++s) {
            if (lane + s * kWarpSize < slots) {
                const float2 xf = __bfloat1622float2(values[s]);
                // Rounded to BF16, as the norm's plane holds it when the rope reads it.
                normed[s] = __bfloat1622float2(
                    __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(xf.x, inv, gains[s].x, 0.0f),
                                          rmsnorm_epilogue<Epilogue>(xf.y, inv, gains[s].y, 0.0f)));
            }
        }
    }
    __syncthreads();
    if (!active) { return; }

    // Slot u (< 2 * quarter) is rotated: the first half's slots pair with u + quarter.
    const int quarter = rotary_dim / 4;
    if (quarter % kWarpSize == 0) {
        const int apart = quarter / kWarpSize; // registers between a slot and its partner
#pragma unroll
        for (int s = 0; s < kMaxSlots; ++s) {
            const int slot = lane + s * kWarpSize;
            if (slot >= slots) { continue; }
            if (slot >= 2 * quarter) {
                row[slot] = __floats2bfloat162_rn(normed[s].x, normed[s].y);
                continue;
            }
            const bool lower    = slot < quarter;
            const int pair      = 2 * (lower ? slot : slot - quarter);
            float2 partner      = normed[0];
#pragma unroll
            for (int t = 0; t < kMaxSlots; ++t) {
                if (t == (lower ? s + apart : s - apart)) { partner = normed[t]; }
            }
            const float2 first  = lower ? normed[s] : partner;
            const float2 second = lower ? partner : normed[s];
            const float c0 = cos_cache[pair], c1 = cos_cache[pair + 1];
            const float s0 = sin_cache[pair], s1 = sin_cache[pair + 1];
            row[slot] = lower ? __floats2bfloat162_rn(rope_rotated_first(first.x, second.x, c0, s0),
                                                      rope_rotated_first(first.y, second.y, c1, s1))
                              : __floats2bfloat162_rn(rope_rotated_second(first.x, second.x, c0, s0),
                                                      rope_rotated_second(first.y, second.y, c1, s1));
        }
    } else {
        // quarter is a power of two below 32: every rotated slot is in the first register.
        float2 partner;
        partner.x = __shfl_xor_sync(kFullWarpMask, normed[0].x, quarter);
        partner.y = __shfl_xor_sync(kFullWarpMask, normed[0].y, quarter);
#pragma unroll
        for (int s = 0; s < kMaxSlots; ++s) {
            const int slot = lane + s * kWarpSize;
            if (slot >= slots) { continue; }
            if (slot >= 2 * quarter) {
                row[slot] = __floats2bfloat162_rn(normed[s].x, normed[s].y);
                continue;
            }
            const bool lower    = slot < quarter;
            const int pair      = 2 * (lower ? slot : slot - quarter);
            const float2 first  = lower ? normed[0] : partner;
            const float2 second = lower ? partner : normed[0];
            const float c0 = cos_cache[pair], c1 = cos_cache[pair + 1];
            const float s0 = sin_cache[pair], s1 = sin_cache[pair + 1];
            row[slot] = lower ? __floats2bfloat162_rn(rope_rotated_first(first.x, second.x, c0, s0),
                                                      rope_rotated_first(first.y, second.y, c1, s1))
                              : __floats2bfloat162_rn(rope_rotated_second(first.x, second.x, c0, s0),
                                                      rope_rotated_second(first.y, second.y, c1, s1));
        }
    }
}

} // namespace sinfer::ops
