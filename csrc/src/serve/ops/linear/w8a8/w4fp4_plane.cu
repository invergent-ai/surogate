// surogate vendor patch (PATCHES.md #21): NVFP4 plane derivation, activation
// quantization, and the prefill quant mode switch (see w4fp4_plane.h).

#include "ops/linear/w8a8/w4fp4_plane.h"

#include "core/device.h"
#include "ops/linear/w8a8/w8fp8_plane.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <mutex>
#include <unordered_map>

namespace sinfer::ops::detail {
namespace {

constexpr int kThreads = 256;
constexpr float kFp4Range = 448.0f * 6.0f;  // ue4m3 max * e2m1 max

__device__ __forceinline__ int e2m1_encode(float value) {
    // Nearest of {0, .5, 1, 1.5, 2, 3, 4, 6} with sign bit 8.
    const int sign      = value < 0.0f ? 8 : 0;
    const float mag     = fabsf(value);
    int code;
    if (mag < 0.25f) { code = 0; }
    else if (mag < 0.75f) { code = 1; }
    else if (mag < 1.25f) { code = 2; }
    else if (mag < 1.75f) { code = 3; }
    else if (mag < 2.5f) { code = 4; }
    else if (mag < 3.5f) { code = 5; }
    else if (mag < 5.0f) { code = 6; }
    else { code = 7; }
    return sign | code;
}

__device__ __forceinline__ float ue4m3_decode(int byte) {
    const int e = (byte >> 3) & 0xF;
    const int m = byte & 7;
    if (e == 0) { return ldexpf(static_cast<float>(m) / 8.0f, -6); }
    return ldexpf(1.0f + static_cast<float>(m) / 8.0f, e - 7);
}

__device__ __forceinline__ int ue4m3_encode_rn(float value) {
    // Nearest ue4m3 (used for the folded cutlass scales; the row-major plane
    // keeps encode_up semantics for the exact-clamp guarantee).
    if (!(value > 0.0f)) { return 0; }
    if (value >= 448.0f) { return 0x7F; }
    int e;
    const float frac = frexpf(value, &e);
    int exp_field = (e - 1) + 7;
    int m         = static_cast<int>(nearbyintf((2.0f * frac - 1.0f) * 8.0f));
    if (m == 8) { m = 0; ++exp_field; }
    if (exp_field <= 0) {
        m = static_cast<int>(nearbyintf(ldexpf(value, 6) * 8.0f));
        if (m > 7) { return 1 << 3; }
        return m < 0 ? 0 : m;
    }
    if (exp_field > 15) { return 0x7F; }
    return (exp_field << 3) | m;
}

// Sm1xx SF atom layout offset for (row m, 16-group kg); kg_total must be a
// multiple of 4 (all engine K are multiples of 64).
__device__ __forceinline__ std::int64_t sf_atom_offset(int m, int kg, int kg_total) {
    return (static_cast<std::int64_t>(m / 128) * (kg_total / 4) + kg / 4) * 512 +
           (m % 32) * 16 + ((m / 32) & 3) * 4 + (kg & 3);
}

__device__ __forceinline__ int ue4m3_encode_up(float value) {
    // Smallest ue4m3 >= value (round up so encoded elements never exceed
    // the e2m1 range); clamps to [0, 448].
    if (!(value > 0.0f)) { return 0; }
    if (value >= 448.0f) { return 0x7F; }
    int e;
    const float frac = frexpf(value, &e);  // value = frac * 2^e, frac in [0.5, 1)
    // value = (2 * frac) * 2^(e-1); mantissa m = ceil((2*frac - 1) * 8)
    int exp_field = (e - 1) + 7;
    int m         = static_cast<int>(ceilf((2.0f * frac - 1.0f) * 8.0f));
    if (m == 8) { m = 0; ++exp_field; }
    if (exp_field <= 0) {
        // subnormal: value = m/8 * 2^-6
        m = static_cast<int>(ceilf(ldexpf(value, 6) * 8.0f));
        if (m > 7) { return 1 << 3; }  // e=1, m=0 = 2^-6
        return m;
    }
    if (exp_field > 15) { return 0x7F; }
    return (exp_field << 3) | m;
}

// One CTA per weight row: rowmax, then per-16 block scales + e2m1 encode.
__global__ void w4fp4_derive_kernel(const std::int8_t* __restrict__ codes,
                                    const __half* __restrict__ scales,
                                    std::uint8_t* __restrict__ fp4_codes,
                                    std::uint8_t* __restrict__ sf_plane,
                                    std::uint8_t* __restrict__ sf_atom,
                                    float* __restrict__ row_scales, int k) {
    __shared__ float red[kThreads];
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int kg  = k / 32;
    const auto* row_codes  = codes + static_cast<std::int64_t>(row) * k;
    const auto* row_groups = scales + static_cast<std::int64_t>(row) * kg;
    const auto dequant     = [&](int i) {
        return static_cast<float>(row_codes[i]) * __half2float(row_groups[i >> 5]);
    };

    float local = 0.0f;
    for (int i = tid; i < k; i += kThreads) { local = fmaxf(local, fabsf(dequant(i))); }
    red[tid] = local;
    __syncthreads();
#pragma unroll
    for (int step = kThreads / 2; step > 0; step >>= 1) {
        if (tid < step) { red[tid] = fmaxf(red[tid], red[tid + step]); }
        __syncthreads();
    }
    const float rowmax    = red[0] > 0.0f ? red[0] : 1.0f;
    const float row_scale = rowmax / kFp4Range;
    if (tid == 0) { row_scales[row] = row_scale; }

    auto* out_codes = fp4_codes + static_cast<std::int64_t>(row) * (k / 2);
    auto* out_sf    = sf_plane + static_cast<std::int64_t>(row) * (k / 16);
    for (int g = tid; g < k / 16; g += kThreads) {
        float gmax = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) { gmax = fmaxf(gmax, fabsf(dequant(g * 16 + i))); }
        const int sf_byte  = ue4m3_encode_up(gmax / (6.0f * row_scale));
        const float sf_dec = ue4m3_decode(sf_byte);
        const float inv    = sf_dec > 0.0f ? 1.0f / (row_scale * sf_dec) : 0.0f;
        out_sf[g] = static_cast<std::uint8_t>(sf_byte);
        sf_atom[sf_atom_offset(row, g, k / 16)] =
            static_cast<std::uint8_t>(ue4m3_encode_rn(sf_dec * row_scale));
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int lo = e2m1_encode(dequant(g * 16 + 2 * p) * inv);
            const int hi = e2m1_encode(dequant(g * 16 + 2 * p + 1) * inv);
            out_codes[g * 8 + p] = static_cast<std::uint8_t>(lo | (hi << 4));
        }
    }
}

// One CTA per token. Vectorized rewrite (PATCHES.md #23): each thread owns
// one 16-group, loads it ONCE as two uint4 (the old kernel read every value
// from global twice, scalar), reduces the group max from registers, and
// encodes register-resident values.
__global__ void w4fp4_act_quant_kernel(const __nv_bfloat16* __restrict__ x,
                                       std::uint8_t* __restrict__ codes,
                                       std::uint8_t* __restrict__ sf_plane,
                                       float* __restrict__ scales, int hidden) {
    __shared__ float red[kThreads];
    const int token = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);
    const __nv_bfloat16* row = x + static_cast<std::int64_t>(token) * hidden;
    const int groups         = hidden / 16;

    // Pass 1: token absmax from vectorized group loads.
    float local = 0.0f;
    for (int g = tid; g < groups; g += kThreads) {
        const uint4 v0 = *reinterpret_cast<const uint4*>(row + g * 16);
        const uint4 v1 = *reinterpret_cast<const uint4*>(row + g * 16 + 8);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float2 a = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&(&v0.x)[w]));
            const float2 b = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&(&v1.x)[w]));
            local = fmaxf(local, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)),
                                       fmaxf(fabsf(b.x), fabsf(b.y))));
        }
    }
    red[tid] = local;
    __syncthreads();
#pragma unroll
    for (int step = kThreads / 2; step > 0; step >>= 1) {
        if (tid < step) { red[tid] = fmaxf(red[tid], red[tid + step]); }
        __syncthreads();
    }
    const float tokmax = red[0] > 0.0f ? red[0] : 1.0f;
    const float scale  = tokmax / kFp4Range;
    if (tid == 0) { scales[token] = scale; }

    auto* out_codes = codes + static_cast<std::int64_t>(token) * (hidden / 2);
    auto* out_sf    = sf_plane + static_cast<std::int64_t>(token) * (hidden / 16);
    for (int g = tid; g < groups; g += kThreads) {
        const uint4 v0 = *reinterpret_cast<const uint4*>(row + g * 16);
        const uint4 v1 = *reinterpret_cast<const uint4*>(row + g * 16 + 8);
        float vals[16];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float2 a = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&(&v0.x)[w]));
            const float2 b = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&(&v1.x)[w]));
            vals[2 * w]     = a.x;
            vals[2 * w + 1] = a.y;
            vals[8 + 2 * w]     = b.x;
            vals[8 + 2 * w + 1] = b.y;
        }
        float gmax = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) { gmax = fmaxf(gmax, fabsf(vals[i])); }
        const int sf_byte  = ue4m3_encode_up(gmax / (6.0f * scale));
        const float sf_dec = ue4m3_decode(sf_byte);
        const float inv    = sf_dec > 0.0f ? 1.0f / (scale * sf_dec) : 0.0f;
        out_sf[g] = static_cast<std::uint8_t>(sf_byte);
        std::uint8_t packed[8];
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int lo = e2m1_encode(vals[2 * p] * inv);
            const int hi = e2m1_encode(vals[2 * p + 1] * inv);
            packed[p]    = static_cast<std::uint8_t>(lo | (hi << 4));
        }
        *reinterpret_cast<std::uint64_t*>(out_codes + g * 8) =
            *reinterpret_cast<const std::uint64_t*>(packed);
    }
}

// cutlass-path activation quantization: folded per-token scale, atom SF.
__global__ void w4fp4_act_quant_atom_kernel(const __nv_bfloat16* __restrict__ x,
                                            std::uint8_t* __restrict__ codes,
                                            std::uint8_t* __restrict__ sf_atom,
                                            int hidden) {
    __shared__ float red[kThreads];
    const int token = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);
    const __nv_bfloat16* row = x + static_cast<std::int64_t>(token) * hidden;
    const int groups         = hidden / 16;

    float local = 0.0f;
    for (int g = tid; g < groups; g += kThreads) {
        const uint4 v0 = *reinterpret_cast<const uint4*>(row + g * 16);
        const uint4 v1 = *reinterpret_cast<const uint4*>(row + g * 16 + 8);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float2 a = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&(&v0.x)[w]));
            const float2 b = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&(&v1.x)[w]));
            local = fmaxf(local, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)),
                                       fmaxf(fabsf(b.x), fabsf(b.y))));
        }
    }
    red[tid] = local;
    __syncthreads();
#pragma unroll
    for (int step = kThreads / 2; step > 0; step >>= 1) {
        if (tid < step) { red[tid] = fmaxf(red[tid], red[tid + step]); }
        __syncthreads();
    }
    const float tokmax = red[0] > 0.0f ? red[0] : 1.0f;
    const float scale  = tokmax / kFp4Range;

    auto* out_codes = codes + static_cast<std::int64_t>(token) * (hidden / 2);
    for (int g = tid; g < groups; g += kThreads) {
        const uint4 v0 = *reinterpret_cast<const uint4*>(row + g * 16);
        const uint4 v1 = *reinterpret_cast<const uint4*>(row + g * 16 + 8);
        float vals[16];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float2 a = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&(&v0.x)[w]));
            const float2 b = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&(&v1.x)[w]));
            vals[2 * w]         = a.x;
            vals[2 * w + 1]     = a.y;
            vals[8 + 2 * w]     = b.x;
            vals[8 + 2 * w + 1] = b.y;
        }
        float gmax = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) { gmax = fmaxf(gmax, fabsf(vals[i])); }
        const int sf_byte  = ue4m3_encode_up(gmax / (6.0f * scale));
        const float sf_dec = ue4m3_decode(sf_byte);
        const float inv    = sf_dec > 0.0f ? 1.0f / (scale * sf_dec) : 0.0f;
        sf_atom[sf_atom_offset(token, g, groups)] =
            static_cast<std::uint8_t>(ue4m3_encode_rn(sf_dec * scale));
        std::uint8_t packed[8];
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int lo = e2m1_encode(vals[2 * p] * inv);
            const int hi = e2m1_encode(vals[2 * p + 1] * inv);
            packed[p]    = static_cast<std::uint8_t>(lo | (hi << 4));
        }
        *reinterpret_cast<std::uint64_t*>(out_codes + g * 8) =
            *reinterpret_cast<const std::uint64_t*>(packed);
    }
}

struct PlaneEntry {
    std::uint8_t* codes   = nullptr;
    std::uint8_t* sf      = nullptr;
    std::uint8_t* sf_atom = nullptr;
    float* row_scales     = nullptr;
    cudaEvent_t ready     = nullptr;
};

std::mutex g_mutex;
std::size_t g_allocated_bytes = 0;
std::unordered_map<const void*, PlaneEntry> g_planes;
PrefillQuantMode g_mode = PrefillQuantMode::Fp8;

constexpr std::size_t align16(std::size_t bytes) noexcept {
    return (bytes + 15u) & ~std::size_t{15u};
}

} // namespace

void w8_prefill_quant_set_mode(PrefillQuantMode mode) noexcept { g_mode = mode; }

PrefillQuantMode w8_prefill_quant_mode() noexcept { return g_mode; }

std::size_t w4fp4_plane_bytes() noexcept {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_allocated_bytes;
}

W4Fp4Plane w4fp4_plane_for(const Weight& weight, cudaStream_t stream) {
    // The fp4 profile needs the sm_120a block-scale mma (and the W4 decode
    // path is gated with it): require CC 12.x.
    if (w8_device_compute_capability() < 120) { return {nullptr, nullptr, nullptr, nullptr}; }
    if (!w8fp8_plane_enabled() || g_mode != PrefillQuantMode::Fp4 ||
        weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group != 32 || weight.qdata == nullptr ||
        weight.scales == nullptr || (weight.k % 256) != 0) {
        return {nullptr, nullptr, nullptr, nullptr};
    }

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    (void)cudaStreamIsCapturing(stream, &capture_status);
    const bool capturing = capture_status != cudaStreamCaptureStatusNone;

    std::lock_guard<std::mutex> lock(g_mutex);
    auto found = g_planes.find(weight.qdata);
    if (found != g_planes.end()) {
        if (!capturing && found->second.ready != nullptr) {
            cudaStreamWaitEvent(stream, found->second.ready, 0);
        }
        return {found->second.codes, found->second.sf, found->second.row_scales,
                found->second.sf_atom};
    }

    if (capturing) { return {nullptr, nullptr, nullptr, nullptr}; }

    const std::size_t code_bytes    = static_cast<std::size_t>(weight.n) * (weight.k / 2);
    const std::size_t sf_bytes      = static_cast<std::size_t>(weight.n) * (weight.k / 16);
    const std::size_t sf_atom_bytes = w4fp4_sf_atom_bytes(weight.n, weight.k);
    const std::size_t scale_bytes   = static_cast<std::size_t>(weight.n) * sizeof(float);
    const std::size_t total         = code_bytes + sf_bytes + sf_atom_bytes + scale_bytes;

    std::size_t free_bytes = 0, total_bytes = 0;
    const auto fail = [&](PlaneEntry entry) {
        cudaFree(entry.codes);
        cudaFree(entry.sf);
        cudaFree(entry.sf_atom);
        cudaFree(entry.row_scales);
        if (entry.ready != nullptr) { cudaEventDestroy(entry.ready); }
        g_planes.emplace(weight.qdata, PlaneEntry{});
        return W4Fp4Plane{nullptr, nullptr, nullptr};
    };
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess || free_bytes < 2 * total) {
        return fail(PlaneEntry{});
    }

    std::size_t free_pre_alloc = 0;
    (void)cudaMemGetInfo(&free_pre_alloc, &total_bytes);

    PlaneEntry entry;
    if (cudaMalloc(&entry.codes, code_bytes) != cudaSuccess) { return fail(entry); }
    if (cudaMalloc(&entry.sf, sf_bytes) != cudaSuccess) { return fail(entry); }
    if (cudaMalloc(&entry.sf_atom, sf_atom_bytes) != cudaSuccess) { return fail(entry); }
    if (cudaMemsetAsync(entry.sf_atom, 0, sf_atom_bytes, stream) != cudaSuccess) {
        return fail(entry);
    }
    if (cudaMalloc(&entry.row_scales, scale_bytes) != cudaSuccess) { return fail(entry); }
    if (cudaEventCreateWithFlags(&entry.ready, cudaEventDisableTiming) != cudaSuccess) {
        return fail(entry);
    }

    w4fp4_derive_kernel<<<weight.n, kThreads, 0, stream>>>(
        static_cast<const std::int8_t*>(weight.qdata),
        static_cast<const __half*>(weight.scales), entry.codes, entry.sf, entry.sf_atom,
        entry.row_scales, weight.k);
    if (cudaGetLastError() != cudaSuccess ||
        cudaEventRecord(entry.ready, stream) != cudaSuccess) {
        return fail(entry);
    }

    std::size_t free_post_alloc = 0;
    (void)cudaMemGetInfo(&free_post_alloc, &total_bytes);
    g_allocated_bytes +=
        free_pre_alloc > free_post_alloc ? free_pre_alloc - free_post_alloc : 0;
    g_planes.emplace(weight.qdata, entry);
    return {entry.codes, entry.sf, entry.row_scales, entry.sf_atom};
}

W4Fp4QuantizedActivations w4fp4_act_quant(const Tensor& x, void* workspace,
                                          cudaStream_t stream) {
    const std::int32_t hidden = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    auto* codes = static_cast<std::uint8_t*>(workspace);
    auto* sf    = codes + align16(static_cast<std::size_t>(hidden / 2) * tokens);
    auto* scales = reinterpret_cast<float*>(
        sf + align16(static_cast<std::size_t>(hidden / 16) * tokens));
    w4fp4_act_quant_kernel<<<tokens, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), codes, sf, scales, hidden);
    CUDA_CHECK(cudaGetLastError());
    return {codes, sf, scales};
}

W4Fp4AtomActivations w4fp4_act_quant_atom(const Tensor& x, void* workspace,
                                          cudaStream_t stream) {
    const std::int32_t hidden = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    auto* codes = static_cast<std::uint8_t*>(workspace);
    auto* sf    = codes + align16(static_cast<std::size_t>(hidden / 2) * tokens);
    const std::size_t sf_bytes = w4fp4_sf_atom_bytes(tokens, hidden);
    CUDA_CHECK(cudaMemsetAsync(sf, 0, sf_bytes, stream));
    w4fp4_act_quant_atom_kernel<<<tokens, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), codes, sf, hidden);
    CUDA_CHECK(cudaGetLastError());
    return {codes, sf};
}

const float* w4fp4_alpha_one() {
    static const float* device_one = [] {
        float* p          = nullptr;
        const float value = 1.0f;
        if (cudaMalloc(&p, sizeof(float)) != cudaSuccess) { return static_cast<float*>(nullptr); }
        cudaMemcpy(p, &value, sizeof(float), cudaMemcpyHostToDevice);
        return p;
    }();
    return device_one;
}

std::size_t w4fp4_cutlass_workspace_bytes(std::int32_t parent_rows, std::int32_t input_rows,
                                          std::int32_t max_tokens,
                                          bool with_stage_buffer) noexcept {
    const std::size_t codes =
        align16(static_cast<std::size_t>(input_rows / 2) * max_tokens);
    const std::size_t sf = align16(w4fp4_sf_atom_bytes(max_tokens, input_rows));
    const std::size_t stage =
        with_stage_buffer
            ? align16(static_cast<std::size_t>(parent_rows) * max_tokens * sizeof(__nv_bfloat16))
            : 0;
    return codes + sf + stage;
}

} // namespace sinfer::ops::detail
