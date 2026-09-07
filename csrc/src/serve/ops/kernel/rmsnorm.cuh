#pragma once

// sinfer::ops - RMSNorm kernels over contiguous BF16 rows.

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace sinfer::ops {

enum class RmsEpilogue {
    Offset,
    Plain,
    Gated,
    GatedSigmoid,
    /// `out += normalised`, with and without the unit offset on the gain. The
    /// second operand is the destination itself, which is what lets a norm land
    /// straight on a residual instead of through a plane and a second kernel.
    OffsetAdd,
    PlainAdd,
    /// No gain at all: `out = x * inv`. Gemma 4 normalises its attention *value* with
    /// `RMSNorm(..., with_scale=False)` -- a norm that has no weight to read, not one whose
    /// weight happens to be ones. There is no tensor behind it, which is why this is an
    /// epilogue rather than a caller passing a ones vector it would have to keep somewhere.
    ///
    /// Not interchangeable with `l2norm`, which is the other weightless normalisation here:
    /// that divides by `sqrt(sum)` where this divides by `sqrt(mean + eps)`, and its eps sits
    /// inside a different expression.
    Unweighted,
};

/// Epilogues that read the second per-element operand `z`: the gated ones
/// multiply by an activation of it, the accumulating ones add it.
/// Epilogues that read a per-channel gain. All but the weightless one -- and the
/// distinction has to be a compile-time trait rather than a null check, because a kernel
/// that indexed `weight[pair]` for `Unweighted` would read whatever pointer the caller
/// passed for a tensor that does not exist.
template <RmsEpilogue Epilogue>
inline constexpr bool kRmsEpilogueReadsWeight = Epilogue != RmsEpilogue::Unweighted;

/// The gain for one BF16 pair, or one, for an epilogue that has no weight.
template <RmsEpilogue Epilogue>
__device__ __forceinline__ float2 rmsnorm_gain2(const __nv_bfloat162* weight, int pair) {
    if constexpr (kRmsEpilogueReadsWeight<Epilogue>) {
        return __bfloat1622float2(weight[pair]);
    } else {
        return float2{1.0F, 1.0F};
    }
}

/// The same, one channel at a time, for the generic path.
template <RmsEpilogue Epilogue>
__device__ __forceinline__ float rmsnorm_gain(const __nv_bfloat16* weight, std::int64_t index) {
    if constexpr (kRmsEpilogueReadsWeight<Epilogue>) {
        return __bfloat162float(weight[index]);
    } else {
        return 1.0F;
    }
}

template <RmsEpilogue Epilogue>
inline constexpr bool kRmsEpilogueReadsOperand =
    Epilogue == RmsEpilogue::Gated || Epilogue == RmsEpilogue::GatedSigmoid ||
    Epilogue == RmsEpilogue::OffsetAdd || Epilogue == RmsEpilogue::PlainAdd;

template <RmsEpilogue Epilogue>
__device__ __forceinline__ float rmsnorm_epilogue(float x, float inv, float weight, float z) {
    if constexpr (Epilogue == RmsEpilogue::Offset || Epilogue == RmsEpilogue::OffsetAdd) {
        weight += 1.0f;
    }
    float value = x * inv * weight;
    if constexpr (Epilogue == RmsEpilogue::Gated) { value *= silu(z); }
    if constexpr (Epilogue == RmsEpilogue::GatedSigmoid) { value *= sigmoid(z); }
    if constexpr (Epilogue == RmsEpilogue::OffsetAdd || Epilogue == RmsEpilogue::PlainAdd) {
        // Round the normalised term to BF16 before adding, because that is what
        // the two kernels this replaces did: the norm wrote a BF16 plane and the
        // residual add read it back. Keeping the rounding keeps the fused form
        // bit-identical to them, and it is also what the reference does.
        value = __bfloat162float(__float2bfloat16(value)) + z;
    }
    return value;
}

// Fast geometry for D in {64, 128, 192, 256}. One warp owns one row, keeps the input in
// registers, and uses only warp shuffles for the reduction. Block is a scheduling choice rather
// than part of the row geometry.
template <RmsEpilogue Epilogue, int Block>
__launch_bounds__(Block) __global__
    void rmsnorm_warp_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                    const __nv_bfloat162* z, __nv_bfloat162* out, std::int32_t d,
                                    std::int64_t rows, float eps) {
    static_assert(Block % kWarpSize == 0);
    constexpr int kWarpsPerBlock   = Block / kWarpSize;
    constexpr int kMaxPairsPerLane = 4;
    const int lane                 = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp                 = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t row         = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) { return; }

    const int pairs             = d / 2;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    __nv_bfloat162 values[kMaxPairsPerLane];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < kMaxPairsPerLane; ++k) {
        const int pair = lane + k * kWarpSize;
        if (pair < pairs) {
            values[k]       = x[row_base + pair];
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    sum       = warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum / static_cast<float>(d) + eps) : 0.0f;
    inv       = __shfl_sync(kFullWarpMask, inv, 0);

#pragma unroll
    for (int k = 0; k < kMaxPairsPerLane; ++k) {
        const int pair = lane + k * kWarpSize;
        if (pair < pairs) {
            const float2 xf = __bfloat1622float2(values[k]);
            const float2 wf = rmsnorm_gain2<Epilogue>(weight, pair);
            float2 zf{0.0f, 0.0f};
            if constexpr (kRmsEpilogueReadsOperand<Epilogue>) {
                zf = __bfloat1622float2(z[row_base + pair]);
            }
            out[row_base + pair] =
                __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, zf.x),
                                      rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, zf.y));
        }
    }
}

// Implements: include/sinfer/ops/rmsnorm.h
// Match: aligned contiguous BF16, plain epilogue, D=128, sm_120a.
// Algorithm assumptions: exactly two BF16x2 values per lane; one warp owns one logical row.
template <RmsEpilogue Epilogue, int Block>
__launch_bounds__(Block) __global__
    void rmsnorm_d128_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                    const __nv_bfloat162* z, __nv_bfloat162* out, std::int64_t rows,
                                    float eps) {
    static_assert(Block % kWarpSize == 0);
    constexpr int kWarpsPerBlock = Block / kWarpSize;
    constexpr int kPairsPerRow   = 64;
    const int lane               = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp               = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t row       = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) { return; }

    const std::int64_t row_base = row * kPairsPerRow;
    const int pair0             = lane;
    const int pair1             = lane + kWarpSize;
    const __nv_bfloat162 value0 = x[row_base + pair0];
    const __nv_bfloat162 value1 = x[row_base + pair1];
    const float2 x0             = __bfloat1622float2(value0);
    const float2 x1             = __bfloat1622float2(value1);
    float sum                   = x0.x * x0.x + x0.y * x0.y + x1.x * x1.x + x1.y * x1.y;
    sum                         = warp_reduce_sum(sum);
    float inv                   = lane == 0 ? rsqrtf(sum * (1.0f / 128.0f) + eps) : 0.0f;
    inv                         = __shfl_sync(kFullWarpMask, inv, 0);

    const float2 w0 = rmsnorm_gain2<Epilogue>(weight, pair0);
    const float2 w1 = rmsnorm_gain2<Epilogue>(weight, pair1);
    float2 z0{0.0f, 0.0f};
    float2 z1{0.0f, 0.0f};
    if constexpr (kRmsEpilogueReadsOperand<Epilogue>) {
        z0 = __bfloat1622float2(z[row_base + pair0]);
        z1 = __bfloat1622float2(z[row_base + pair1]);
    }
    out[row_base + pair0] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x0.x, inv, w0.x, z0.x),
                              rmsnorm_epilogue<Epilogue>(x0.y, inv, w0.y, z0.y));
    out[row_base + pair1] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x1.x, inv, w1.x, z1.x),
                              rmsnorm_epilogue<Epilogue>(x1.y, inv, w1.y, z1.y));
}

// Fast geometry for wide rows. One CTA owns one row and keeps up to MaxPairsPerThread BF16x2
// values per lane. The launcher admits only widths evenly divisible by the CTA vector span.
template <RmsEpilogue Epilogue, int Block, int MaxPairsPerThread>
__launch_bounds__(Block) __global__
    void rmsnorm_cta_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                   const __nv_bfloat162* z, __nv_bfloat162* out, std::int32_t d,
                                   std::int64_t rows, float eps) {
    static_assert(Block % kWarpSize == 0);
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const int pairs             = d / 2;
    const int pairs_per_thread  = pairs / Block;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    __nv_bfloat162 values[MaxPairsPerThread];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < MaxPairsPerThread; ++k) {
        if (k < pairs_per_thread) {
            const int pair  = static_cast<int>(threadIdx.x) + k * Block;
            values[k]       = x[row_base + pair];
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    __shared__ float warp_sums[Block / kWarpSize];
    __shared__ float inv_shared;
    const float block_sum = block_reduce_sum<Block>(sum, warp_sums);
    if (threadIdx.x == 0) { inv_shared = rsqrtf(block_sum / static_cast<float>(d) + eps); }
    __syncthreads();
    const float inv = inv_shared;

#pragma unroll
    for (int k = 0; k < MaxPairsPerThread; ++k) {
        if (k < pairs_per_thread) {
            const int pair  = static_cast<int>(threadIdx.x) + k * Block;
            const float2 xf = __bfloat1622float2(values[k]);
            const float2 wf = rmsnorm_gain2<Epilogue>(weight, pair);
            float2 zf{0.0f, 0.0f};
            if constexpr (kRmsEpilogueReadsOperand<Epilogue>) {
                zf = __bfloat1622float2(z[row_base + pair]);
            }
            out[row_base + pair] =
                __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, zf.x),
                                      rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, zf.y));
        }
    }
}

// Implements: include/sinfer/ops/rmsnorm.h
// Match: aligned contiguous BF16, plain epilogue, D=2048, sm_120a.
// Algorithm assumptions: exactly two BF16x2 values per thread; one 512-thread CTA owns one row.
template <RmsEpilogue Epilogue>
__launch_bounds__(512) __global__
    void rmsnorm_d2048_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                     const __nv_bfloat162* z, __nv_bfloat162* out,
                                     std::int64_t rows, float eps) {
    constexpr int kBlock       = 512;
    constexpr int kPairsPerRow = 1024;
    const std::int64_t row     = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const std::int64_t row_base = row * kPairsPerRow;
    const int pair0             = static_cast<int>(threadIdx.x);
    const int pair1             = pair0 + kBlock;
    const __nv_bfloat162 value0 = x[row_base + pair0];
    const __nv_bfloat162 value1 = x[row_base + pair1];
    const float2 x0             = __bfloat1622float2(value0);
    const float2 x1             = __bfloat1622float2(value1);
    const float local_sum       = x0.x * x0.x + x0.y * x0.y + x1.x * x1.x + x1.y * x1.y;

    __shared__ float warp_sums[kBlock / kWarpSize];
    __shared__ float inv_shared;
    const float sum = block_reduce_sum<kBlock>(local_sum, warp_sums);
    if (threadIdx.x == 0) { inv_shared = rsqrtf(sum * (1.0f / 2048.0f) + eps); }
    __syncthreads();
    const float inv = inv_shared;

    const float2 w0 = rmsnorm_gain2<Epilogue>(weight, pair0);
    const float2 w1 = rmsnorm_gain2<Epilogue>(weight, pair1);
    float2 z0{0.0f, 0.0f};
    float2 z1{0.0f, 0.0f};
    if constexpr (kRmsEpilogueReadsOperand<Epilogue>) {
        z0 = __bfloat1622float2(z[row_base + pair0]);
        z1 = __bfloat1622float2(z[row_base + pair1]);
    }
    out[row_base + pair0] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x0.x, inv, w0.x, z0.x),
                              rmsnorm_epilogue<Epilogue>(x0.y, inv, w0.y, z0.y));
    out[row_base + pair1] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x1.x, inv, w1.x, z1.x),
                              rmsnorm_epilogue<Epilogue>(x1.y, inv, w1.y, z1.y));
}

// Functional fallback outside the aligned fast domains. It intentionally favors a simple complete
// implementation over another family of shape-specific paths.
template <RmsEpilogue Epilogue>
__launch_bounds__(256) __global__
    void rmsnorm_generic_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                const __nv_bfloat16* z, __nv_bfloat16* out, std::int32_t d,
                                std::int64_t rows, float eps) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const std::int64_t base = row * static_cast<std::int64_t>(d);
    float sum               = 0.0f;
    for (std::int64_t i = threadIdx.x; i < static_cast<std::int64_t>(d); i += blockDim.x) {
        const float xv = __bfloat162float(x[base + i]);
        sum += xv * xv;
    }

    __shared__ float scratch[256];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) { scratch[threadIdx.x] += scratch[threadIdx.x + stride]; }
        __syncthreads();
    }

    const float inv = rsqrtf(scratch[0] / static_cast<float>(d) + eps);
    for (std::int64_t i = threadIdx.x; i < static_cast<std::int64_t>(d); i += blockDim.x) {
        const std::int64_t index = base + i;
        const float xv           = __bfloat162float(x[index]);
        const float wv           = rmsnorm_gain<Epilogue>(weight, i);
        float zv                 = 0.0f;
        if constexpr (kRmsEpilogueReadsOperand<Epilogue>) { zv = __bfloat162float(z[index]); }
        out[index] = __float2bfloat16_rn(rmsnorm_epilogue<Epilogue>(xv, inv, wv, zv));
    }
}

} // namespace sinfer::ops
