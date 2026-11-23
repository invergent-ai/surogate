#include "ops/linear/mxfp8/mxfp8.h"

#include "api/ops/vision_fp8.h"
#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <atomic>
#include <stdexcept>

namespace sinfer::ops {
namespace {
std::atomic<bool> g_vision_fp8{false};
} // namespace

void set_vision_fp8(bool enabled) noexcept { g_vision_fp8.store(enabled, std::memory_order_relaxed); }
bool vision_fp8() noexcept { return g_vision_fp8.load(std::memory_order_relaxed); }

} // namespace sinfer::ops

namespace sinfer::ops::detail {
namespace {

constexpr int kBlock = 32; // values per UE8M0 scale
// UE8M0 2^0: what the tile padding holds, so cuBLASLt never reads a NaN (0xFF) scale.
constexpr unsigned char kUnitScale = 127;

std::int32_t scale_groups(std::int32_t k) noexcept { return mxfp8_padded_k(k) / kBlock; }
std::int32_t scale_tiles_per_row(std::int32_t k) noexcept { return (scale_groups(k) + 3) / 4; }

// ceil(log2(ratio)) for a finite positive float, from its bits: the exponent, plus one unless
// the mantissa is zero (an exact power of two). Subnormal ratios floor at -127.
__device__ __forceinline__ int ceil_log2(float ratio) {
    const unsigned bits = __float_as_uint(ratio);
    const int biased    = static_cast<int>((bits >> 23) & 0xffU);
    if (biased == 0) { return -127; }
    return biased - 127 + ((bits & 0x7fffffU) != 0U ? 1 : 0);
}

// tanh.approx.f32 (sm_75+): about 2^-11 relative error, far inside the e4m3 rounding (2^-4)
// that follows it, at a fraction of tanhf's cost on the tower's widest pass.
__device__ __forceinline__ float fast_tanh(float x) {
    float y;
    asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float gelu_tanh_fast(float x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654F;
    return 0.5F * x * (1.0F + fast_tanh(kSqrt2OverPi * (x + 0.044715F * x * x * x)));
}

// Four lanes per 32-value block, eight values (one 16-byte load) each: a warp reads 512
// contiguous bytes of a row, and the block's amax is two shuffles. K is a multiple of 8, so a
// chunk is all source or all padding; the padded row length is a multiple of 32, so the four
// lanes of a block are always one row's.
template <bool Gelu>
__global__ void mxfp8_quantize_kernel(const __nv_bfloat16* __restrict__ source,
                                      std::int64_t row_stride, std::int32_t rows, std::int32_t k,
                                      std::uint8_t* __restrict__ codes,
                                      std::uint8_t* __restrict__ scales) {
    const std::int32_t padded = (k + kBlock - 1) / kBlock * kBlock;
    const std::int32_t chunks = padded / 8; // per row
    const std::int64_t chunk  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const bool live           = chunk < static_cast<std::int64_t>(rows) * chunks;
    const auto row            = live ? static_cast<std::int32_t>(chunk / chunks) : 0;
    const auto column         = live ? static_cast<std::int32_t>(chunk - static_cast<std::int64_t>(row) * chunks) * 8 : 0;

    float v[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    if (live && column < k) {
        const uint4 packed = *reinterpret_cast<const uint4*>(source + row * row_stride + column);
        const auto* pairs  = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 f   = __bfloat1622float2(pairs[pair]);
            v[2 * pair]      = f.x;
            v[2 * pair + 1]  = f.y;
        }
    }
    float amax = 0.0F;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if constexpr (Gelu) { v[i] = gelu_tanh_fast(v[i]); }
        amax = fmaxf(amax, fabsf(v[i]));
    }
    // The block's four lanes are adjacent and 4-aligned in the warp (the grid is a multiple of
    // the warp, and chunks per row a multiple of four); dead lanes contribute zero.
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffU, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffU, amax, 2));
    if (!live) { return; }
    // 2^e with e = ceil(log2(amax / 448)): the block's largest value lands at or under e4m3's
    // largest finite value, so nothing saturates. A zero block keeps 2^-127.
    const int exponent  = amax > 0.0F ? max(-127, min(127, ceil_log2(amax * (1.0F / 448.0F)))) : -127;
    const float inverse = exp2f(static_cast<float>(-exponent));
    uint2 out;
    auto* out_pairs = reinterpret_cast<__nv_fp8x2_storage_t*>(&out);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        out_pairs[pair] = __nv_cvt_float2_to_fp8x2(
            make_float2(v[2 * pair] * inverse, v[2 * pair + 1] * inverse), __NV_SATFINITE, __NV_E4M3);
    }
    *reinterpret_cast<uint2*>(codes + static_cast<std::int64_t>(row) * padded + column) = out;
    if ((column & (kBlock - 1)) == 0) {
        scales[nvfp4_tiled_scale_offset(row, column / kBlock, (padded / kBlock + 3) / 4)] =
            static_cast<std::uint8_t>(exponent + 127);
    }
}

} // namespace

std::size_t mxfp8_codes_bytes(std::int32_t rows, std::int32_t k) noexcept {
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(mxfp8_padded_k(k));
}

std::size_t mxfp8_scales_bytes(std::int32_t rows, std::int32_t k) noexcept {
    const std::size_t row_tiles = (static_cast<std::size_t>(rows) + 127) / 128;
    return row_tiles * static_cast<std::size_t>(scale_tiles_per_row(k)) * 512;
}

bool mxfp8_device_supported() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int major = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    return major >= 10;
}

void mxfp8_quantize(const void* source, std::int64_t row_stride, Mxfp8Transform transform,
                    Mxfp8Matrix& out, cudaStream_t stream) {
    if (source == nullptr || out.codes == nullptr || out.scales == nullptr || out.rows <= 0 ||
        out.k <= 0 || (out.k % 8) != 0 || row_stride < out.k || (row_stride % 8) != 0 ||
        (reinterpret_cast<std::uintptr_t>(source) & 15U) != 0 ||
        (reinterpret_cast<std::uintptr_t>(out.codes) & 15U) != 0) {
        throw std::invalid_argument("mxfp8_quantize: rows of k BF16 values (k a multiple of 8), "
                                    "16-byte aligned, row stride a multiple of 8 and at least k");
    }
    CUDA_CHECK(cudaMemsetAsync(out.scales, kUnitScale, mxfp8_scales_bytes(out.rows, out.k), stream));
    const std::int64_t items = static_cast<std::int64_t>(out.rows) * (mxfp8_padded_k(out.k) / 8);
    const unsigned grid      = static_cast<unsigned>((items + 255) / 256);
    const auto* in           = static_cast<const __nv_bfloat16*>(source);
    auto* codes              = static_cast<std::uint8_t*>(out.codes);
    auto* scales             = static_cast<std::uint8_t*>(out.scales);
    if (transform == Mxfp8Transform::GeluTanh) {
        mxfp8_quantize_kernel<true><<<grid, 256, 0, stream>>>(in, row_stride, out.rows, out.k, codes, scales);
    } else {
        mxfp8_quantize_kernel<false><<<grid, 256, 0, stream>>>(in, row_stride, out.rows, out.k, codes, scales);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail
