#include "ops/linear_swiglu/w8/w8_linear_swiglu_kernels.h"

#include "core/device.h"

#include <stdexcept>
#include <string>
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/w8a8/w4fp4_decode.cuh"
#include "ops/linear/w8a8/w4fp4_plane.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace sinfer::ops::detail {
namespace {

// surogate vendor patch (PATCHES.md #13): geometry is templated so the
// qwen3.5-0.8b mlp (3584 x 2, k=1024) shares this kernel with the 35B shape.
template <int RowsPerCta, int Intermediate = 6144, int K = 2048>
__global__ __launch_bounds__(RowsPerCta * 32, 2) void w8_linear_swiglu_decode_pair_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out) {
    constexpr int kIntermediate   = Intermediate;
    constexpr int kK              = K;
    constexpr int kGroupsPerRow   = kK / 32;
    constexpr int kValuesPerLane  = 8;
    constexpr int kValuesPerPhase = 32 * kValuesPerLane;
    constexpr int kGroupsPerPhase = kValuesPerPhase / 32;
    constexpr int kPhases         = kK / kValuesPerPhase;
    constexpr unsigned kMask      = 0xffffffffu;

    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int warp          = static_cast<int>(threadIdx.x) >> 5;
    const int row           = static_cast<int>(blockIdx.x) * RowsPerCta + warp;
    const int up_row        = row + kIntermediate;
    const auto* gate_row    = codes + static_cast<std::int64_t>(row) * kK;
    const auto* up_codes    = codes + static_cast<std::int64_t>(up_row) * kK;
    const auto* gate_scales = scales + static_cast<std::int64_t>(row) * kGroupsPerRow * 2;
    const auto* up_scales   = scales + static_cast<std::int64_t>(up_row) * kGroupsPerRow * 2;

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;
#pragma unroll
    for (int phase = 0; phase < kPhases; ++phase) {
        unsigned gate_scale_bits = 0;
        unsigned up_scale_bits   = 0;
        if (lane < kGroupsPerPhase) {
            gate_scale_bits = *reinterpret_cast<const std::uint16_t*>(
                gate_scales + static_cast<std::int64_t>(phase * kGroupsPerPhase + lane) * 2);
            up_scale_bits = *reinterpret_cast<const std::uint16_t*>(
                up_scales + static_cast<std::int64_t>(phase * kGroupsPerPhase + lane) * 2);
        }
        gate_scale_bits        = __shfl_sync(kMask, gate_scale_bits, lane >> 2);
        up_scale_bits          = __shfl_sync(kMask, up_scale_bits, lane >> 2);
        const float gate_scale = __half2float(__ushort_as_half(gate_scale_bits));
        const float up_scale   = __half2float(__ushort_as_half(up_scale_bits));

        const int phase_k       = phase * kValuesPerPhase + lane * kValuesPerLane;
        const uint2 gate_packed = load_vec<uint2>(gate_row + phase_k);
        const uint2 up_packed   = load_vec<uint2>(up_codes + phase_k);
        const uint4 values      = load_vec<uint4>(x + phase_k);
        const float2 xv[4]      = {
            bf16x2_bits_to_float2(values.x),
            bf16x2_bits_to_float2(values.y),
            bf16x2_bits_to_float2(values.z),
            bf16x2_bits_to_float2(values.w),
        };
#pragma unroll
        for (int word_index = 0; word_index < 2; ++word_index) {
            const std::uint32_t gate_word = (&gate_packed.x)[word_index];
            const std::uint32_t up_word   = (&up_packed.x)[word_index];
#pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                const int shift              = byte * 8;
                const float2 activation_pair = xv[word_index * 2 + (byte >> 1)];
                const float activation = (byte & 1) == 0 ? activation_pair.x : activation_pair.y;
                const float gate_value =
                    static_cast<float>(static_cast<std::int8_t>(gate_word >> shift)) * gate_scale;
                const float up_value =
                    static_cast<float>(static_cast<std::int8_t>(up_word >> shift)) * up_scale;
                gate_acc = fmaf(gate_value, activation, gate_acc);
                up_acc   = fmaf(up_value, activation, up_acc);
            }
        }
    }

    gate_acc = warp_reduce_sum(gate_acc);
    up_acc   = warp_reduce_sum(up_acc);
    if (lane == 0) { out[row] = __float2bfloat16_rn(silu(gate_acc) * up_acc); }
}


// surogate vendor patch (PATCHES.md #22): fp4 profile twin — gate/up rows
// read from the derived NVFP4 plane (half the weight traffic of the pair).
template <int RowsPerCta, int Intermediate, int K>
__global__ __launch_bounds__(RowsPerCta * 32, 2) void w4fp4_swiglu_decode_pair_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ sf, const float* __restrict__ row_scales,
    __nv_bfloat16* __restrict__ out) {
    constexpr int kIntermediate   = Intermediate;
    constexpr int kK              = K;
    constexpr int kValuesPerLane  = 8;
    constexpr int kValuesPerPhase = 32 * kValuesPerLane;
    constexpr int kSfPerPhase     = kValuesPerPhase / 16;
    constexpr int kPhases         = kK / kValuesPerPhase;
    constexpr unsigned kMask      = 0xffffffffu;

    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int row        = static_cast<int>(blockIdx.x) * RowsPerCta + warp;
    const int up_row     = row + kIntermediate;
    const auto* gate_row = codes + static_cast<std::int64_t>(row) * (kK / 2);
    const auto* up_codes = codes + static_cast<std::int64_t>(up_row) * (kK / 2);
    const auto* gate_sf  = sf + static_cast<std::int64_t>(row) * (kK / 16);
    const auto* up_sf    = sf + static_cast<std::int64_t>(up_row) * (kK / 16);

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;
#pragma unroll
    for (int phase = 0; phase < kPhases; ++phase) {
        unsigned gate_sf_bits = 0;
        unsigned up_sf_bits   = 0;
        if (lane < kSfPerPhase) {
            gate_sf_bits = gate_sf[phase * kSfPerPhase + lane];
            up_sf_bits   = up_sf[phase * kSfPerPhase + lane];
        }
        gate_sf_bits           = __shfl_sync(kMask, gate_sf_bits, lane >> 1);
        up_sf_bits             = __shfl_sync(kMask, up_sf_bits, lane >> 1);
        const float gate_scale = w4fp4_ue4m3_decode(gate_sf_bits);
        const float up_scale   = w4fp4_ue4m3_decode(up_sf_bits);

        const int phase_k = phase * kValuesPerPhase + lane * kValuesPerLane;
        const std::uint32_t gate_packed =
            *reinterpret_cast<const std::uint32_t*>(gate_row + phase_k / 2);
        const std::uint32_t up_packed =
            *reinterpret_cast<const std::uint32_t*>(up_codes + phase_k / 2);
        const uint4 values = load_vec<uint4>(x + phase_k);
        const float2 xv[4] = {
            bf16x2_bits_to_float2(values.x),
            bf16x2_bits_to_float2(values.y),
            bf16x2_bits_to_float2(values.z),
            bf16x2_bits_to_float2(values.w),
        };
        float gate_partial = 0.0f;
        float up_partial   = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float2 pair      = xv[j >> 1];
            const float activation = (j & 1) == 0 ? pair.x : pair.y;
            gate_partial =
                fmaf(w4fp4_e2m1_decode((gate_packed >> (4 * j)) & 0xFu), activation, gate_partial);
            up_partial =
                fmaf(w4fp4_e2m1_decode((up_packed >> (4 * j)) & 0xFu), activation, up_partial);
        }
        gate_acc = fmaf(gate_partial, gate_scale, gate_acc);
        up_acc   = fmaf(up_partial, up_scale, up_acc);
    }

    gate_acc = warp_reduce_sum(gate_acc) * row_scales[row];
    up_acc   = warp_reduce_sum(up_acc) * row_scales[up_row];
    if (lane == 0) { out[row] = __float2bfloat16_rn(silu(gate_acc) * up_acc); }
}

template <int RowsPerCta, int Intermediate = 6144, int K = 2048>
void launch_decode(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    static_assert(Intermediate % RowsPerCta == 0);
    // surogate vendor patch (PATCHES.md #22): fp4 profile decode.
    if (w8_prefill_quant_mode() == PrefillQuantMode::Fp4) {
        const W4Fp4Plane plane = w4fp4_plane_for(w, stream);
        if (plane.codes != nullptr) {
            w4fp4_swiglu_decode_pair_kernel<RowsPerCta, Intermediate, K>
                <<<Intermediate / RowsPerCta, RowsPerCta * 32, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(x.data), plane.codes, plane.sf,
                    plane.row_scales, static_cast<__nv_bfloat16*>(out.data));
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    w8_linear_swiglu_decode_pair_kernel<RowsPerCta, Intermediate, K>
        <<<Intermediate / RowsPerCta, RowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

// The kernel bakes both extents, so the dispatch has to name the whole geometry.
// Keying it on the hidden extent alone was wrong the moment two models shared a
// hidden size and differed in intermediate -- qwen3.5-0.8b and qwen3-0.6b both
// have k=1024 with intermediate 3584 and 3072 -- and it failed silently: the
// 0.6b ran 3584 rows of another model's weight and produced NaNs. An
// unregistered pair now says so.
template <int RowsPerCta>
void dispatch_decode(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t intermediate = w.n / 2;
    if (w.k == 2048 && intermediate == 6144) {
        launch_decode<RowsPerCta, 6144, 2048>(x, w, out, stream);
        return;
    }
    if (w.k == 1024 && intermediate == 3584) {
        launch_decode<RowsPerCta, 3584, 1024>(x, w, out, stream);
        return;
    }
    // qwen3-0.6b mlp (2x3072, k=1024).
    if (w.k == 1024 && intermediate == 3072) {
        launch_decode<RowsPerCta, 3072, 1024>(x, w, out, stream);
        return;
    }
    if (w.k == 2560 && intermediate == 9216) {
        launch_decode<RowsPerCta, 9216, 2560>(x, w, out, stream);
        return;
    }
    // tinyllama-1.1b: 11264 gate_up rows over hidden 2048, so 5632 out.
    if (w.k == 2048 && intermediate == 5632) {
        launch_decode<RowsPerCta, 5632, 2048>(x, w, out, stream);
        return;
    }
    throw std::invalid_argument("W8 LinearSwiGLU decode: no instantiation for gate_up_rows " +
                                std::to_string(w.n) + " over k " + std::to_string(w.k));
}

void w8_linear_swiglu_decode_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    dispatch_decode<8>(x, w, out, stream);
}

void w8_linear_swiglu_decode_pair_r4_launch(const Tensor& x, const Weight& w, Tensor& out,
                                            cudaStream_t stream) {
    dispatch_decode<4>(x, w, out, stream);
}

void w8_linear_swiglu_decode_pair_r16_launch(const Tensor& x, const Weight& w, Tensor& out,
                                             cudaStream_t stream) {
    // surogate vendor patches (PATCHES.md #13/#18): per-geometry decode.
    dispatch_decode<16>(x, w, out, stream);
}

} // namespace sinfer::ops::detail
