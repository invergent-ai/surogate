// The shape-generic route over row-scaled FP8 weights.
//
// The registered geometries (fp8_config.h) each carry hand-tuned decode, small-T and A8
// schedules, and every one of them is a Qwen3.8-27B shape. A weight of any other shape in this
// format -- another size of a family whose converter emits it, a fused parent projected by one
// half, a vocabulary of another width -- had no launch at all and was refused where it ran.
// This is the fallback that serves it: one runtime-shaped kernel over BF16 activations, and at
// prefill widths the A8 route's cuBLASLt GEMM, which is shape-generic already. Neither is tuned
// for any particular shape; they are what makes an unregistered shape servable rather than a
// named failure, exactly as `is_nvfp4_generic_problem` does for NVFP4 (PATCHES.md #84).
//
// The mainloop stages the activation columns of four tokens in shared memory and gives each warp
// one output row, so the weight -- the larger operand by far -- is read once per four tokens and
// the activations are read once per row block.

#include "ops/linear/fp8/fp8_launch.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_gemv.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace sinfer::ops::detail {
namespace {

constexpr int kGenericThreads        = 256;
constexpr int kGenericWarps          = kGenericThreads / 32; ///< output rows per block
constexpr int kGenericTokensPerBlock = 4;
constexpr int kGenericChunk          = 1024; ///< activation values staged per pass, per token
constexpr int kGenericValuesPerLane  = 8;    ///< codes a lane reads at a time (one uint2)

template <bool Accumulate>
__global__ __launch_bounds__(kGenericThreads) void fp8_generic_kernel(
    const std::uint8_t* __restrict__ codes, const __nv_bfloat16* __restrict__ row_scales,
    const __nv_bfloat16* __restrict__ activation, __nv_bfloat16* __restrict__ out,
    std::int32_t rows, std::int32_t k, std::int32_t tokens, std::int32_t out_ld) {
    __shared__ float staged[kGenericTokensPerBlock][kGenericChunk];

    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int row        = static_cast<int>(blockIdx.x) * kGenericWarps + warp;
    const int token_base = static_cast<int>(blockIdx.y) * kGenericTokensPerBlock;
    const int token_count =
        min(kGenericTokensPerBlock, tokens - token_base); // >= 1: the grid covers no more
    // A row past the end still walks the loop (its warp takes part in the block's barriers) but
    // reads row zero's codes and writes nothing.
    const std::uint8_t* row_codes =
        codes + static_cast<std::int64_t>(row < rows ? row : 0) * k;

    float totals[kGenericTokensPerBlock];
#pragma unroll
    for (int t = 0; t < kGenericTokensPerBlock; ++t) { totals[t] = 0.0F; }

    for (int base = 0; base < k; base += kGenericChunk) {
        const int count = min(kGenericChunk, k - base);
        for (int t = 0; t < token_count; ++t) {
            const __nv_bfloat16* column =
                activation + static_cast<std::int64_t>(token_base + t) * k + base;
            for (int i = static_cast<int>(threadIdx.x); i < count; i += kGenericThreads) {
                staged[t][i] = __bfloat162float(column[i]);
            }
        }
        __syncthreads();
        // `count` is a whole number of 32 values (K is), so a lane whose offset is inside the
        // chunk has its whole eight-value pack inside it.
        for (int offset = lane * kGenericValuesPerLane; offset < count;
             offset += 32 * kGenericValuesPerLane) {
            const uint2 pack           = load_vec<uint2>(row_codes + base + offset);
            const auto* packed_pairs   = reinterpret_cast<const std::uint16_t*>(&pack);
            float weights[kGenericValuesPerLane];
#pragma unroll
            for (int pair = 0; pair < kGenericValuesPerLane / 2; ++pair) {
                const float2 decoded  = decode_fp8_e4m3x2(packed_pairs[pair]);
                weights[2 * pair]     = decoded.x;
                weights[2 * pair + 1] = decoded.y;
            }
            for (int t = 0; t < token_count; ++t) {
#pragma unroll
                for (int value = 0; value < kGenericValuesPerLane; ++value) {
                    totals[t] = fmaf(weights[value], staged[t][offset + value], totals[t]);
                }
            }
        }
        __syncthreads();
    }

    const float row_scale = row < rows ? __bfloat162float(row_scales[row]) : 0.0F;
    for (int t = 0; t < token_count; ++t) {
        const float total = warp_reduce_sum(totals[t]);
        if (lane == 0 && row < rows) {
            __nv_bfloat16& destination =
                out[static_cast<std::int64_t>(token_base + t) * out_ld + row];
            const float value = total * row_scale;
            destination       = __float2bfloat16_rn(
                Accumulate ? __bfloat162float(destination) + value : value);
        }
    }
}

} // namespace

void launch_fp8_generic(const Tensor& x, const Weight& weight, Tensor& out, bool accumulate,
                        cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16 || x.ne[0] != weight.k ||
        out.ne[0] != weight.n || out.ne[1] != tokens || tokens <= 0) {
        throw std::invalid_argument("fp8 generic linear: shapes do not match the weight");
    }
    if ((weight.k % 32) != 0) {
        throw std::invalid_argument("fp8 generic linear: K must be a whole number of 32 values");
    }
    const dim3 grid(static_cast<unsigned>((weight.n + kGenericWarps - 1) / kGenericWarps),
                    static_cast<unsigned>((tokens + kGenericTokensPerBlock - 1) /
                                          kGenericTokensPerBlock));
    const auto* codes      = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* row_scales = static_cast<const __nv_bfloat16*>(weight.scales);
    const auto* input      = static_cast<const __nv_bfloat16*>(x.data);
    auto* destination      = static_cast<__nv_bfloat16*>(out.data);
    if (accumulate) {
        fp8_generic_kernel<true><<<grid, kGenericThreads, 0, stream>>>(
            codes, row_scales, input, destination, weight.n, weight.k, tokens, weight.n);
    } else {
        fp8_generic_kernel<false><<<grid, kGenericThreads, 0, stream>>>(
            codes, row_scales, input, destination, weight.n, weight.k, tokens, weight.n);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail
