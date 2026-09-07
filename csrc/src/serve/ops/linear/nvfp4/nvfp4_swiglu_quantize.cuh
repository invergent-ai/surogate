#pragma once

// The SwiGLU activation quantised straight into the down projection's W4A4 operand.
//
// On the wide route the gate/up GEMM writes a BF16 [2I, T] plane, `silu_mul` reads it and
// writes a BF16 [I, T] activation, and the down projection's quantiser reads that back to
// write FP4 codes and block scales: on a 27B prompt round that is 160 MB written and read
// for nothing but the intermediate. This kernel reads the projected plane once and writes
// the codes and scales the GEMM consumes, rounding each SwiGLU value to BF16 first so the
// codes are exactly the ones the unfused pair produces.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace sinfer::ops::detail {

/// `projected` is the gate/up GEMM's token-major [2 * K, T] BF16 plane (gate rows first),
/// K = `Geometry::kInputRows` the down projection's K; `codes`/`scales` are the W4A4 workspace
/// in the layout `nvfp4_w4a4_quantize_kernel` writes.
template <class Geometry, int Threads = 256, bool TiledScales = false>
__global__ __launch_bounds__(Threads, 512 / Threads) void nvfp4_w4a4_swiglu_quantize_kernel(
    const __nv_bfloat16* __restrict__ projected, std::uint8_t* __restrict__ codes,
    std::uint8_t* __restrict__ scales, std::int32_t tokens, float input_scale_divisor,
    float limit) {
    static_assert(Threads == 128 || Threads == 256 || Threads == 512);
    constexpr int kGroupsPerRow            = Geometry::kInputRows / 16;
    constexpr std::int64_t kProjectedRows  = 2LL * Geometry::kInputRows;
    const int task =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int tasks = tokens * kGroupsPerRow;
    if (task >= tasks) { return; }

    const int token = task / kGroupsPerRow;
    const int group = task - token * kGroupsPerRow;
    const __nv_bfloat16* gate =
        projected + static_cast<std::int64_t>(token) * kProjectedRows + group * 16;
    const __nv_bfloat16* up = gate + Geometry::kInputRows;

    const uint4 gate0 = load_vec<uint4>(gate);
    const uint4 gate1 = load_vec<uint4>(gate + 8);
    const uint4 up0   = load_vec<uint4>(up);
    const uint4 up1   = load_vec<uint4>(up + 8);
    const std::uint32_t gate_bits[8] = {gate0.x, gate0.y, gate0.z, gate0.w,
                                        gate1.x, gate1.y, gate1.z, gate1.w};
    const std::uint32_t up_bits[8]   = {up0.x, up0.y, up0.z, up0.w, up1.x, up1.y, up1.z, up1.w};

    float2 values[8];
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        __nv_bfloat162 g, u;
        g = *reinterpret_cast<const __nv_bfloat162*>(&gate_bits[pair]);
        u = *reinterpret_cast<const __nv_bfloat162*>(&up_bits[pair]);
        // What `silu_mul` writes for this pair, rounded to BF16 as it rounds it, so the
        // unfused pair and this kernel quantise the same values.
        const float r0 = swiglu_clamped(__low2float(g), __low2float(u), limit);
        const float r1 = swiglu_clamped(__high2float(g), __high2float(u), limit);
        values[pair]   = __bfloat1622float2(__floats2bfloat162_rn(r0, r1));
    }
    const Nvfp4QuantizedK16 quantized = quantize_nvfp4_k16_values(values, input_scale_divisor);
    auto* code_destination =
        codes + static_cast<std::int64_t>(token) * Geometry::kCodeBytesPerRow + group * 8;
    store_vec(code_destination, make_uint2(quantized.codes_lo, quantized.codes_hi));
    if constexpr (TiledScales) {
        scales[nvfp4_tiled_scale_offset(token, group, Geometry::kInputRows / 64)] =
            quantized.scale;
    } else {
        scales[static_cast<std::int64_t>(token) * kGroupsPerRow + group] = quantized.scale;
    }
}

} // namespace sinfer::ops::detail
