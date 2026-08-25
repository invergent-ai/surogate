// surogate vendor patch (PATCHES.md #20): FP8-e4m3 prefill plane derivation
// and registry (see w8fp8_plane.h for the scheme).

#include "ops/linear/w8a8/w8fp8_plane.h"

#include "ops/linear/w8a8/w4fp4_plane.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace ninfer::ops::detail {
namespace {

constexpr int kThreads = 256;

// One CTA per weight row: rowmax over dequantized values, then e4m3 encode
// of w / rowmax. Group scales are fp16 at [row, k/32].
__global__ void w8fp8_derive_kernel(const std::int8_t* __restrict__ codes,
                                    const __half* __restrict__ scales,
                                    std::uint8_t* __restrict__ fp8_codes,
                                    float* __restrict__ row_scales, int k) {
    __shared__ float red[kThreads];
    const int row  = static_cast<int>(blockIdx.x);
    const int tid  = static_cast<int>(threadIdx.x);
    const int kg   = k / 32;
    const auto* row_codes  = codes + static_cast<std::int64_t>(row) * k;
    const auto* row_groups = scales + static_cast<std::int64_t>(row) * kg;

    float local = 0.0f;
    for (int i = tid; i < k; i += kThreads) {
        const float value =
            static_cast<float>(row_codes[i]) * __half2float(row_groups[i >> 5]);
        local = fmaxf(local, fabsf(value));
    }
    red[tid] = local;
    __syncthreads();
#pragma unroll
    for (int step = kThreads / 2; step > 0; step >>= 1) {
        if (tid < step) { red[tid] = fmaxf(red[tid], red[tid + step]); }
        __syncthreads();
    }
    const float rowmax = red[0];
    const float inv    = rowmax > 0.0f ? 1.0f / rowmax : 0.0f;
    if (tid == 0) { row_scales[row] = rowmax > 0.0f ? rowmax : 1.0f; }

    auto* out = fp8_codes + static_cast<std::int64_t>(row) * k;
    for (int i = tid; i < k; i += kThreads) {
        const float value =
            static_cast<float>(row_codes[i]) * __half2float(row_groups[i >> 5]);
        out[i] = __nv_cvt_float_to_fp8(value * inv, __NV_SATFINITE, __NV_E4M3);
    }
}

__global__ void w8fp8_act_quant_kernel(const __nv_bfloat16* __restrict__ x,
                                       std::uint8_t* __restrict__ codes,
                                       float* __restrict__ scales, int hidden) {
    __shared__ float red[kThreads];
    const int token = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);
    const __nv_bfloat16* row = x + static_cast<std::int64_t>(token) * hidden;

    float local = 0.0f;
    for (int i = tid; i < hidden; i += kThreads) {
        local = fmaxf(local, fabsf(__bfloat162float(row[i])));
    }
    red[tid] = local;
    __syncthreads();
#pragma unroll
    for (int step = kThreads / 2; step > 0; step >>= 1) {
        if (tid < step) { red[tid] = fmaxf(red[tid], red[tid + step]); }
        __syncthreads();
    }
    const float absmax = red[0];
    const float scale  = absmax > 0.0f ? absmax * (1.0f / 448.0f) : 1.0f;
    const float inv    = absmax > 0.0f ? 448.0f / absmax : 0.0f;
    if (tid == 0) { scales[token] = scale; }
    std::uint8_t* out = codes + static_cast<std::int64_t>(token) * hidden;
    for (int i = tid; i < hidden; i += kThreads) {
        out[i] = __nv_cvt_float_to_fp8(__bfloat162float(row[i]) * inv, __NV_SATFINITE,
                                       __NV_E4M3);
    }
}

struct PlaneEntry {
    std::uint8_t* codes   = nullptr;
    float* row_scales     = nullptr;
    cudaEvent_t ready     = nullptr;  // recorded on the deriving stream
};

std::mutex g_mutex;
std::size_t g_allocated_bytes = 0;
std::unordered_map<const void*, PlaneEntry> g_planes;
bool g_enabled = false;

bool env_vetoed() {
    static const bool vetoed = [] {
        const char* env = std::getenv("SUROGATE_SERVE_FP8_PREFILL");
        return env != nullptr && env[0] == '0';
    }();
    return vetoed;
}

} // namespace

void w8fp8_plane_set_enabled(bool enabled) noexcept { g_enabled = enabled; }

bool w8fp8_plane_enabled() noexcept { return g_enabled && !env_vetoed(); }

std::size_t w8fp8_plane_bytes() noexcept {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_allocated_bytes;
}

std::size_t w8_derived_plane_bytes() noexcept {
    return w8fp8_plane_bytes() + w4fp4_plane_bytes();
}

W8Fp8Plane w8fp8_plane_for(const Weight& weight, cudaStream_t stream) {
    if (!w8fp8_plane_enabled() || weight.qtype != QType::W8G32_F16S ||
        weight.layout != QuantLayout::RowSplit || weight.scale_dtype != DType::FP16 ||
        weight.group != 32 || weight.qdata == nullptr || weight.scales == nullptr ||
        (weight.k % 32) != 0) {
        return {nullptr, nullptr};
    }

    // Graph capture discipline: a capturing stream may LOOK UP a finished
    // plane (the pre-capture warmup decode derived and synchronized it) but
    // must not derive (cudaMalloc) or wait on external events.
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    (void)cudaStreamIsCapturing(stream, &capture_status);
    const bool capturing = capture_status != cudaStreamCaptureStatusNone;

    std::lock_guard<std::mutex> lock(g_mutex);
    auto found = g_planes.find(weight.qdata);
    if (found != g_planes.end()) {
        // Order this stream after the deriving stream (no-op once complete).
        if (!capturing && found->second.ready != nullptr) {
            cudaStreamWaitEvent(stream, found->second.ready, 0);
        }
        return {found->second.codes, found->second.row_scales};
    }

    if (capturing) { return {nullptr, nullptr}; }

    const std::size_t code_bytes = static_cast<std::size_t>(weight.n) * weight.k;
    const std::size_t scale_bytes = static_cast<std::size_t>(weight.n) * sizeof(float);

    // VRAM guard: keep a 2x margin so a raced allocation cannot starve the
    // serving engine's own reservations.
    std::size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
        free_bytes < 2 * (code_bytes + scale_bytes)) {
        g_planes.emplace(weight.qdata, PlaneEntry{});  // do not retry every call
        return {nullptr, nullptr};
    }

    std::size_t free_pre_alloc = 0;
    (void)cudaMemGetInfo(&free_pre_alloc, &total_bytes);

    PlaneEntry entry;
    if (cudaMalloc(&entry.codes, code_bytes) != cudaSuccess) {
        g_planes.emplace(weight.qdata, PlaneEntry{});
        return {nullptr, nullptr};
    }
    if (cudaMalloc(&entry.row_scales, scale_bytes) != cudaSuccess) {
        cudaFree(entry.codes);
        g_planes.emplace(weight.qdata, PlaneEntry{});
        return {nullptr, nullptr};
    }

    if (cudaEventCreateWithFlags(&entry.ready, cudaEventDisableTiming) != cudaSuccess) {
        cudaFree(entry.codes);
        cudaFree(entry.row_scales);
        g_planes.emplace(weight.qdata, PlaneEntry{});
        return {nullptr, nullptr};
    }

    // Stream-ordered publication: the caller's GEMM follows on the same
    // stream; other streams wait on the event at lookup.
    w8fp8_derive_kernel<<<weight.n, kThreads, 0, stream>>>(
        static_cast<const std::int8_t*>(weight.qdata),
        static_cast<const __half*>(weight.scales), entry.codes, entry.row_scales, weight.k);
    if (cudaGetLastError() != cudaSuccess ||
        cudaEventRecord(entry.ready, stream) != cudaSuccess) {
        cudaFree(entry.codes);
        cudaFree(entry.row_scales);
        cudaEventDestroy(entry.ready);
        g_planes.emplace(weight.qdata, PlaneEntry{});
        return {nullptr, nullptr};
    }

    std::size_t free_post_alloc = 0;
    (void)cudaMemGetInfo(&free_post_alloc, &total_bytes);
    g_allocated_bytes +=
        free_pre_alloc > free_post_alloc ? free_pre_alloc - free_post_alloc : 0;
    g_planes.emplace(weight.qdata, entry);
    return {entry.codes, entry.row_scales};
}

W8Fp8QuantizedActivations w8fp8_act_quant(const Tensor& x, void* workspace,
                                          cudaStream_t stream) {
    const std::int32_t hidden = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    auto* codes  = static_cast<std::uint8_t*>(workspace);
    const std::size_t code_bytes =
        (static_cast<std::size_t>(hidden) * tokens + 15u) & ~std::size_t{15u};
    auto* scales = reinterpret_cast<float*>(codes + code_bytes);
    w8fp8_act_quant_kernel<<<tokens, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), codes, scales, hidden);
    CUDA_CHECK(cudaGetLastError());
    return {codes, scales};
}

} // namespace ninfer::ops::detail
