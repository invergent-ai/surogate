// Inkling-2-Mini hidden-stream ops (lane e1): api/ops/inkling_mix.h. One translation unit: kernels,
// launchers and the host-side validation. Compiled with -fmad=false (CMakeLists.txt): every product /
// sum the reference does not fuse stays unfused, and the fma the reference does use is an explicit fmaf.
// Rounding points: rl/engine-inkling-2026-11-21/lanes/e1/DESIGN-target.md section 3.

#include "api/ops/inkling_mix.h"

#include "core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sinfer::ops {
namespace {

using bf16 = __nv_bfloat16;
using G    = InklingMixGeometry;

constexpr int kD       = G::kHidden;     // 2304
constexpr int kThreads = 256;
constexpr int kPer     = kD / kThreads;  // 9
static_assert(kD % kThreads == 0, "hidden width must split over the block");

/// Non-FTZ rsqrt.approx.f32, as `ops/norm.py` `rsqrt` emits it.
__device__ __forceinline__ float rsqrt_approx(float x) {
    float y;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

struct PlaneDev {
    const void* data   = nullptr;
    std::int64_t stride = 0;
    int fp32           = 0;
};

__device__ __forceinline__ float load_plane(const PlaneDev& p, std::int64_t t, int d) {
    if (p.fp32) { return static_cast<const float*>(p.data)[t * p.stride + d]; }
    return __bfloat162float(static_cast<const bf16*>(p.data)[t * p.stride + d]);
}

/// Sum of `n` values per thread over the block; every thread gets the totals. `scratch` holds
/// [kThreads / 32][n] floats. Lane order is fixed, so the result is deterministic.
template <int N>
__device__ __forceinline__ void block_sum(float (&v)[N], float* scratch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int i = 0; i < N; ++i) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) { v[i] += __shfl_xor_sync(0xffffffffU, v[i], off); }
    }
    __syncthreads(); // scratch may still be read from an earlier reduction
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < N; ++i) { scratch[warp * N + i] = v[i]; }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < N; ++i) {
        float s = 0.0F;
#pragma unroll
        for (int w = 0; w < kThreads / 32; ++w) { s += scratch[w * N + i]; }
        v[i] = s;
    }
}

// ---------------------------------------------------------------------------------------- attn_res

struct AttnResArgs {
    PlaneDev src[G::kMaxStack + 1]; // stack..., last
    PlaneDev addend;
    int has_addend          = 0;
    const float* queries    = nullptr; // [D, Q]
    int num_queries         = 0;
    const bf16* norm_weight = nullptr; // [D, Q] or null
    float norm_eps          = 0.0F;
    void* out               = nullptr; // [D, T, Q]
    std::int64_t columns    = 0;
};

template <int N>
__global__ void __launch_bounds__(kThreads) attn_res_kernel(const __grid_constant__ AttnResArgs a) {
    __shared__ float scratch[(kThreads / 32) * (N > 2 ? N : 2)];
    const std::int64_t t = blockIdx.x;
    float x[N][kPer];
#pragma unroll
    for (int n = 0; n < N; ++n) {
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int d = threadIdx.x + i * kThreads;
            float v     = load_plane(a.src[n], t, d);
            if (n == N - 1 && a.has_addend) { v = v + load_plane(a.addend, t, d); }
            x[n][i] = v;
        }
    }
    // Per-source sum of squares (fma chain per thread, as the reference's tl.fma accumulation).
    float sq[N];
#pragma unroll
    for (int n = 0; n < N; ++n) {
        float s = 0.0F;
#pragma unroll
        for (int i = 0; i < kPer; ++i) { s = fmaf(x[n][i], x[n][i], s); }
        sq[n] = s;
    }
    block_sum<N>(sq, scratch);
    float rstd[N];
#pragma unroll
    for (int n = 0; n < N; ++n) { rstd[n] = rsqrt_approx(fmaf(sq[n], 1.0F / 2304.0F, 1e-6F)); }

    for (int q = 0; q < a.num_queries; ++q) {
        const float* query = a.queries + static_cast<std::int64_t>(q) * kD;
        float dot[N];
#pragma unroll
        for (int n = 0; n < N; ++n) {
            float s = 0.0F;
#pragma unroll
            for (int i = 0; i < kPer; ++i) { s = fmaf(query[threadIdx.x + i * kThreads], x[n][i], s); }
            dot[n] = s;
        }
        block_sum<N>(dot, scratch);
        float logit[N];
        float mx = -INFINITY;
#pragma unroll
        for (int n = 0; n < N; ++n) {
            logit[n] = dot[n] * rstd[n];
            mx       = fmaxf(mx, logit[n]);
        }
        float p[N];
        float total = 0.0F;
#pragma unroll
        for (int n = 0; n < N; ++n) {
            p[n]  = expf(logit[n] - mx);
            total = total + p[n];
        }
        const float inv_sum = __fdiv_rn(1.0F, total);
#pragma unroll
        for (int n = 0; n < N; ++n) { p[n] = p[n] * rstd[n]; } // normalize_values
        float mix[kPer];
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            float acc = 0.0F;
#pragma unroll
            for (int n = 0; n < N; ++n) { acc = fmaf(p[n], x[n][i], acc); }
            mix[i] = acc * inv_sum;
        }
        const std::int64_t base = (static_cast<std::int64_t>(q) * a.columns + t) * kD;
        if (a.norm_weight != nullptr) {
            float ss[1] = {0.0F};
#pragma unroll
            for (int i = 0; i < kPer; ++i) { ss[0] = ss[0] + mix[i] * mix[i]; }
            block_sum<1>(ss, scratch);
            const float out_rstd = rsqrt_approx(__fdiv_rn(ss[0], 2304.0F) + a.norm_eps);
            const bf16* w        = a.norm_weight + static_cast<std::int64_t>(q) * kD;
            bf16* out            = static_cast<bf16*>(a.out) + base;
#pragma unroll
            for (int i = 0; i < kPer; ++i) {
                const int d = threadIdx.x + i * kThreads;
                out[d]      = __float2bfloat16_rn((mix[i] * out_rstd) * __bfloat162float(w[d]));
            }
        } else {
            float* out = static_cast<float*>(a.out) + base;
#pragma unroll
            for (int i = 0; i < kPer; ++i) { out[threadIdx.x + i * kThreads] = mix[i]; }
        }
    }
}

// ------------------------------------------------------------------------------- embed and rms_norm

__global__ void __launch_bounds__(kThreads)
    embed_kernel(const std::int32_t* ids, const bf16* table, const bf16* w, float eps, bf16* out,
                 std::int64_t out_stride) {
    __shared__ float scratch[kThreads / 32];
    const std::int64_t t   = blockIdx.x;
    const bf16* row        = table + static_cast<std::int64_t>(ids[t]) * kD;
    float x[kPer];
    float ss[1] = {0.0F};
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        x[i]  = __bfloat162float(row[threadIdx.x + i * kThreads]);
        ss[0] = ss[0] + x[i] * x[i];
    }
    block_sum<1>(ss, scratch);
    const float rstd = rsqrt_approx(__fdiv_rn(ss[0], 2304.0F) + eps);
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int d             = threadIdx.x + i * kThreads;
        out[t * out_stride + d] = __float2bfloat16_rn((x[i] * rstd) * __bfloat162float(w[d]));
    }
}

__global__ void __launch_bounds__(kThreads)
    rms_norm_kernel(PlaneDev x_plane, const bf16* w, int num_weights, float eps, bf16* out,
                    std::int64_t columns) {
    __shared__ float scratch[kThreads / 32];
    const std::int64_t t = blockIdx.x;
    float x[kPer];
    float ss[1] = {0.0F};
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        x[i]  = load_plane(x_plane, t, threadIdx.x + i * kThreads);
        ss[0] = ss[0] + x[i] * x[i];
    }
    block_sum<1>(ss, scratch);
    const float rstd = rsqrt_approx(__fdiv_rn(ss[0], 2304.0F) + eps);
    for (int q = 0; q < num_weights; ++q) {
        bf16* o = out + (static_cast<std::int64_t>(q) * columns + t) * kD;
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int d = threadIdx.x + i * kThreads;
            o[d] = __float2bfloat16_rn((x[i] * rstd) * __bfloat162float(w[q * kD + d]));
        }
    }
}

// -------------------------------------------------------------------------------- loop controller

__global__ void __launch_bounds__(kThreads)
    pair_rms_norm_kernel(const float* dest, std::int64_t dest_stream_stride, const float* msg,
                         bf16* out, std::int64_t columns) {
    __shared__ float scratch[(kThreads / 32) * 2];
    const std::int64_t t = blockIdx.x;
    const std::int64_t s = blockIdx.y;
    const float* dp      = dest + s * dest_stream_stride + t * kD;
    const float* mp      = msg + (s * columns + t) * kD;
    float dv[kPer], mv[kPer];
    float ss[2] = {0.0F, 0.0F};
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int d = threadIdx.x + i * kThreads;
        dv[i]       = dp[d];
        mv[i]       = mp[d];
        ss[0]       = ss[0] + dv[i] * dv[i];
        ss[1]       = ss[1] + mv[i] * mv[i];
    }
    block_sum<2>(ss, scratch);
    const float rd = rsqrt_approx(__fdiv_rn(ss[0], 2304.0F) + 1e-6F);
    const float rm = rsqrt_approx(__fdiv_rn(ss[1], 2304.0F) + 1e-6F);
    bf16* o        = out + (s * columns + t) * (2 * kD);
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int d = threadIdx.x + i * kThreads;
        o[d]        = __float2bfloat16_rn(dv[i] * rd);
        o[kD + d]   = __float2bfloat16_rn(mv[i] * rm);
    }
}

__global__ void __launch_bounds__(kThreads)
    blend_rms_norm_kernel(const float* dest, std::int64_t dest_stream_stride, const float* msg,
                          const float* gain_forget, const bf16* w, int w_stream_stride, float eps,
                          bf16* out, std::int64_t columns) {
    __shared__ float scratch[kThreads / 32];
    const std::int64_t t   = blockIdx.x;
    const std::int64_t s   = blockIdx.y;
    const std::int64_t row = s * columns + t;
    const float* dp        = dest + s * dest_stream_stride + t * kD;
    const float* mp        = msg + row * kD;
    const float* gf        = gain_forget + row * (2 * kD);
    float xv[kPer];
    float ss[1] = {0.0F};
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int d        = threadIdx.x + i * kThreads;
        const float gain   = tanhf(gf[d]);
        const float forget = tanhf(gf[kD + d]);
        xv[i]              = (1.0F - forget) * dp[d] + gain * mp[d];
        ss[0]              = ss[0] + xv[i] * xv[i];
    }
    block_sum<1>(ss, scratch);
    const float rstd = rsqrt_approx(__fdiv_rn(ss[0], 2304.0F) + eps);
    const bf16* ws   = w + s * w_stream_stride;
    bf16* o          = out + row * kD;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int d = threadIdx.x + i * kThreads;
        o[d]        = __float2bfloat16_rn((xv[i] * rstd) * __bfloat162float(ws[d]));
    }
}

// ------------------------------------------------------------------------------------- short conv

constexpr int kConvChannelTile = 128;
constexpr int kConvColumnTile  = 32;

struct ConvArgs {
    const bf16* x            = nullptr;
    std::int64_t x_stride    = 0;
    int channels             = 0;
    const bf16* weight       = nullptr; // [4, channels]: tap k of channel c at c * 4 + k
    int width                = 0;
    int batch                = 0;
    const std::int32_t* positions = nullptr; // [W, B]
    const std::int32_t* valid     = nullptr; // [B] or null
    const std::int32_t* slots     = nullptr; // [B]
    bf16* state              = nullptr;      // already offset by the stream's channel offset
    std::int64_t slot_stride = 0;
    std::int64_t tap_stride  = 0;
    const bf16* residual     = nullptr;
    std::int64_t residual_stride = 0;
    const bf16* addend       = nullptr;
    std::int64_t addend_stride = 0;
    bf16* out                = nullptr;
    std::int64_t out_stride  = 0;
};

/// The value at sequence index i of row b: S = [state tap 0, 1, 2, x column 0, 1, ...].
__device__ __forceinline__ float conv_seq(const ConvArgs& a, int b, std::int64_t slot, int i, int c) {
    if (i < 3) { return __bfloat162float(a.state[slot * a.slot_stride + i * a.tap_stride + c]); }
    const std::int64_t n = static_cast<std::int64_t>(i - 3) + static_cast<std::int64_t>(a.width) * b;
    return __bfloat162float(a.x[n * a.x_stride + c]);
}

__global__ void __launch_bounds__(kConvChannelTile)
    short_conv_kernel(const __grid_constant__ ConvArgs a) {
    const int c = blockIdx.x * kConvChannelTile + threadIdx.x;
    const int b = blockIdx.z;
    if (c >= a.channels) { return; }
    const std::int64_t slot = a.slots[b];
    const int valid = a.valid != nullptr ? a.valid[b] : a.width;
    float w[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        w[k] = __bfloat162float(a.weight[static_cast<std::int64_t>(c) * 4 + k]);
    }
    const float scale = 0.44721359549995793F; // fp32(5 ** -0.5)
    const int j0      = blockIdx.y * kConvColumnTile;
    const int j1      = min(j0 + kConvColumnTile, a.width);
    for (int j = j0; j < j1; ++j) {
        if (j >= valid) { break; }
        const std::int64_t n = static_cast<std::int64_t>(j) + static_cast<std::int64_t>(a.width) * b;
        const int pos        = a.positions[n];
        float acc            = 0.0F;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int lag = 3 - k;
            const float v = (pos - lag >= 0) ? conv_seq(a, b, slot, j + k, c) : 0.0F;
            acc           = fmaf(v, w[k], acc);
        }
        const float cur = __bfloat162float(a.x[n * a.x_stride + c]);
        bf16 y          = __float2bfloat16_rn((acc + cur) * scale);
        if (a.addend != nullptr) {
            float base = __bfloat162float(a.addend[n * a.addend_stride + c]);
            if (a.residual != nullptr) {
                base = __bfloat162float(a.residual[n * a.residual_stride + c]) + base;
            }
            y = __float2bfloat16_rn(base + __bfloat162float(y));
        }
        a.out[n * a.out_stride + c] = y;
    }
}

/// New windows after the conv: per row, the 3 inputs before index `valid` of S go to the row's
/// slot (in place); with a capture, the 3 before `capture_column` go to `capture_slot` first. Each
/// thread reads every value it needs before it writes.
__global__ void __launch_bounds__(kConvChannelTile)
    conv_window_kernel(const __grid_constant__ ConvArgs a, int capture_column, int capture_slot,
                       int update_slots) {
    const int c = blockIdx.x * kConvChannelTile + threadIdx.x;
    const int b = blockIdx.y;
    if (c >= a.channels) { return; }
    const std::int64_t slot = a.slots[b];
    const int valid = a.valid != nullptr ? a.valid[b] : a.width;
    float upd[3], cap[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        upd[i] = conv_seq(a, b, slot, valid + i, c);
        cap[i] = capture_slot >= 0 ? conv_seq(a, b, slot, capture_column + i, c) : 0.0F;
    }
    if (capture_slot >= 0) {
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            a.state[static_cast<std::int64_t>(capture_slot) * a.slot_stride + i * a.tap_stride + c] =
                __float2bfloat16_rn(cap[i]);
        }
    }
    if (update_slots && valid > 0) {
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            a.state[slot * a.slot_stride + i * a.tap_stride + c] = __float2bfloat16_rn(upd[i]);
        }
    }
}

__global__ void conv_copy_slot_kernel(bf16* state, std::int64_t slot_stride, std::int64_t tap_stride,
                                      int channels, int src, int dst) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) { return; }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        state[dst * slot_stride + i * tap_stride + c] = state[src * slot_stride + i * tap_stride + c];
    }
}

// ----------------------------------------------------------------------- verify records and the fold

__global__ void conv_backup_kernel(const bf16* pool, std::int64_t visit_stride,
                                   std::int64_t slot_stride, const std::int32_t* slots,
                                   bf16* old_records, int visits) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y;
    const int b = blockIdx.z;
    if (c >= G::kConvChannels) { return; }
    const bf16* src = pool + v * visit_stride + static_cast<std::int64_t>(slots[b]) * slot_stride;
    bf16* dst = old_records + (static_cast<std::int64_t>(b) * visits + v) * 3 * G::kConvChannels;
#pragma unroll
    for (int i = 0; i < 3; ++i) { dst[i * G::kConvChannels + c] = src[i * G::kConvChannels + c]; }
}

__global__ void conv_record_kernel(const bf16* x, std::int64_t x_stride, int channels, int width,
                                   int visit, int channel_offset, bf16* inputs, int visits,
                                   int width_max) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y;
    const int b = blockIdx.z;
    if (c >= channels) { return; }
    const std::int64_t n = static_cast<std::int64_t>(j) + static_cast<std::int64_t>(width) * b;
    bf16* dst = inputs + ((static_cast<std::int64_t>(b) * visits + visit) * width_max + j) *
                             G::kConvChannels;
    dst[channel_offset + c] = x[n * x_stride + c];
}

constexpr int kMaxFoldRows = 128;
struct FoldArgs {
    InklingConvFoldRow rows[kMaxFoldRows];
    int count = 0;
};

__global__ void conv_fold_kernel(const __grid_constant__ FoldArgs f, bf16* pool,
                                 std::int64_t visit_stride, std::int64_t slot_stride,
                                 const bf16* old_records, const bf16* inputs, int visits,
                                 int width_max) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y;
    const int r = blockIdx.z;
    if (c >= G::kConvChannels || r >= f.count) { return; }
    const InklingConvFoldRow row = f.rows[r];
    const bf16* old = old_records + (static_cast<std::int64_t>(row.row) * visits + v) * 3 * G::kConvChannels;
    const bf16* in  = inputs + (static_cast<std::int64_t>(row.row) * visits + v) * width_max * G::kConvChannels;
    bf16 window[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int idx = row.committed + i; // index into [old 0..2 | inputs 0..]
        window[i] = idx < 3 ? old[idx * G::kConvChannels + c] : in[(idx - 3) * G::kConvChannels + c];
    }
    bf16* dst = pool + v * visit_stride + static_cast<std::int64_t>(row.slot) * slot_stride;
#pragma unroll
    for (int i = 0; i < 3; ++i) { dst[i * G::kConvChannels + c] = window[i]; }
}

__global__ void softcap_f32_kernel(float* logits, int rows, int tokens, std::int64_t ldc, float cap) {
    const std::int64_t i     = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = static_cast<std::int64_t>(rows) * tokens;
    if (i >= total) { return; }
    const std::int64_t t = i / rows;
    const std::int64_t r = i - t * rows;
    float& x             = logits[t * ldc + r];
    x                    = tanhf(__fdiv_rn(x, cap)) * cap;
}

// ------------------------------------------------------------------------------------- validation

[[noreturn]] void fail(const char* op, const std::string& what) {
    throw std::invalid_argument(std::string(op) + ": " + what);
}

PlaneDev plane_dev(const InklingPlane& p, const char* op, const char* name, bool allow_fp32) {
    if (p.data == nullptr) { fail(op, std::string(name) + " is null"); }
    if (p.dtype != DType::BF16 && !(allow_fp32 && p.dtype == DType::FP32)) {
        fail(op, std::string(name) + " has an unsupported dtype");
    }
    if (p.column_stride < kD) { fail(op, std::string(name) + " column stride is below 2304"); }
    return PlaneDev{p.data, p.column_stride, p.dtype == DType::FP32 ? 1 : 0};
}

void require(bool ok, const char* op, const char* what) {
    if (!ok) { fail(op, what); }
}

bool contiguous_shape(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t n1, std::int32_t n2) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous() && t.ne[0] == n0 &&
           t.ne[1] == n1 && t.ne[2] == n2 && t.ne[3] == 1;
}

unsigned grid_count(std::int64_t n, std::int64_t per) {
    return static_cast<unsigned>((n + per - 1) / per);
}

ConvArgs conv_args(const InklingPlane& x, std::int32_t channels, const Tensor* weight,
                   const InklingConvRows& rows, const InklingConvStateView& state,
                   std::int32_t channel_offset, const char* op) {
    require(x.data != nullptr && x.dtype == DType::BF16 && x.column_stride >= channels, op,
            "x must be a BF16 plane at least `channels` wide");
    require(channels > 0 && channel_offset >= 0 && channel_offset + channels <= G::kConvChannels, op,
            "channel range is outside the conv state");
    require(rows.width > 0 && rows.batch > 0, op, "rows need width and batch");
    require(contiguous_shape(rows.positions, DType::I32, rows.width, rows.batch, 1) ||
                contiguous_shape(rows.positions, DType::I32, rows.width * rows.batch, 1, 1),
            op, "positions must be I32 [W, B]");
    require(rows.valid_columns.data == nullptr ||
                contiguous_shape(rows.valid_columns, DType::I32, rows.batch, 1, 1),
            op, "valid_columns must be I32 [B] or empty");
    require(contiguous_shape(rows.state_slots, DType::I32, rows.batch, 1, 1), op,
            "state_slots must be I32 [B]");
    require(state.base != nullptr && state.slot_stride >= 3 * state.tap_stride &&
                state.tap_stride >= G::kConvChannels,
            op, "conv state view is invalid");
    ConvArgs a;
    a.x        = static_cast<const bf16*>(x.data);
    a.x_stride = x.column_stride;
    a.channels = channels;
    if (weight != nullptr) {
        require(contiguous_shape(*weight, DType::BF16, 4, channels, 1), op,
                "weight must be BF16 [4, channels]");
        a.weight = static_cast<const bf16*>(weight->data);
    }
    a.width       = rows.width;
    a.batch       = rows.batch;
    a.positions   = static_cast<const std::int32_t*>(rows.positions.data);
    a.valid       = static_cast<const std::int32_t*>(rows.valid_columns.data);
    a.slots       = static_cast<const std::int32_t*>(rows.state_slots.data);
    a.state       = static_cast<bf16*>(state.base) + channel_offset;
    a.slot_stride = state.slot_stride;
    a.tap_stride  = state.tap_stride;
    return a;
}

} // namespace

void inkling_attn_res(const InklingAttnResSources& sources, std::int32_t columns,
                      const Tensor& queries, const Tensor* norm_weight, float norm_eps, Tensor& out,
                      cudaStream_t stream) {
    constexpr const char* op = "inkling_attn_res";
    require(columns > 0, op, "columns must be positive");
    require(sources.stack_count >= 0 && sources.stack_count <= G::kMaxStack, op,
            "stack_count must be in [0, 5]");
    require(queries.dtype == DType::FP32 && queries.is_contiguous() && queries.ne[0] == kD &&
                queries.ne[1] >= 1 && queries.ne[1] <= 4 && queries.ne[2] == 1 && queries.data,
            op, "queries must be FP32 [2304, Q], Q in 1..4");
    const int q = queries.ne[1];
    AttnResArgs a;
    for (int n = 0; n < sources.stack_count; ++n) {
        a.src[n] = plane_dev(sources.stack[n], op, "stack source", false);
    }
    a.src[sources.stack_count] = plane_dev(sources.last, op, "last", true);
    if (sources.addend.data != nullptr) {
        a.addend     = plane_dev(sources.addend, op, "addend", false);
        a.has_addend = 1;
    }
    a.queries     = static_cast<const float*>(queries.data);
    a.num_queries = q;
    if (norm_weight != nullptr) {
        require(contiguous_shape(*norm_weight, DType::BF16, kD, q, 1), op,
                "norm_weight must be BF16 [2304, Q]");
        require(contiguous_shape(out, DType::BF16, kD, columns, q), op,
                "out must be BF16 [2304, columns, Q] with a norm");
        a.norm_weight = static_cast<const bf16*>(norm_weight->data);
        a.norm_eps    = norm_eps;
    } else {
        require(contiguous_shape(out, DType::FP32, kD, columns, q), op,
                "out must be FP32 [2304, columns, Q] without a norm");
    }
    a.out     = out.data;
    a.columns = columns;
    const dim3 grid(static_cast<unsigned>(columns));
    switch (sources.stack_count + 1) {
    case 1: attn_res_kernel<1><<<grid, kThreads, 0, stream>>>(a); break;
    case 2: attn_res_kernel<2><<<grid, kThreads, 0, stream>>>(a); break;
    case 3: attn_res_kernel<3><<<grid, kThreads, 0, stream>>>(a); break;
    case 4: attn_res_kernel<4><<<grid, kThreads, 0, stream>>>(a); break;
    case 5: attn_res_kernel<5><<<grid, kThreads, 0, stream>>>(a); break;
    case 6: attn_res_kernel<6><<<grid, kThreads, 0, stream>>>(a); break;
    default: fail(op, "unsupported source count");
    }
    CUDA_CHECK(cudaGetLastError());
}

void inkling_embed(const Tensor& ids, const void* table, std::int32_t table_rows,
                   const Tensor& norm_weight, float eps, const InklingPlane& out,
                   cudaStream_t stream) {
    constexpr const char* op = "inkling_embed";
    require(ids.dtype == DType::I32 && ids.is_contiguous() && ids.data && ids.numel() > 0, op,
            "ids must be contiguous I32");
    require(table != nullptr && table_rows > 0, op, "table is missing");
    require(contiguous_shape(norm_weight, DType::BF16, kD, 1, 1), op, "norm_weight must be BF16 [2304]");
    const PlaneDev o = plane_dev(out, op, "out", false);
    embed_kernel<<<static_cast<unsigned>(ids.numel()), kThreads, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const bf16*>(table),
        static_cast<const bf16*>(norm_weight.data), eps, static_cast<bf16*>(out.data), o.stride);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_rms_norm(const InklingPlane& x, std::int32_t columns, const Tensor& weight, float eps,
                      Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "inkling_rms_norm";
    require(columns > 0, op, "columns must be positive");
    const PlaneDev xp = plane_dev(x, op, "x", false);
    require(weight.dtype == DType::BF16 && weight.is_contiguous() && weight.ne[0] == kD &&
                weight.ne[1] >= 1 && weight.ne[2] == 1 && weight.data,
            op, "weight must be BF16 [2304, Q]");
    const int q = weight.ne[1];
    require(contiguous_shape(out, DType::BF16, kD, columns, q), op, "out must be BF16 [2304, columns, Q]");
    rms_norm_kernel<<<static_cast<unsigned>(columns), kThreads, 0, stream>>>(
        xp, static_cast<const bf16*>(weight.data), q, eps, static_cast<bf16*>(out.data), columns);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_pair_rms_norm(const Tensor& dest, const Tensor& message, Tensor& out,
                           cudaStream_t stream) {
    constexpr const char* op = "inkling_pair_rms_norm";
    const int t = message.ne[1];
    const int s = message.ne[2];
    require(contiguous_shape(message, DType::FP32, kD, t, s) && t > 0 && s > 0, op,
            "message must be FP32 [2304, T, S]");
    require(contiguous_shape(dest, DType::FP32, kD, t, s) || contiguous_shape(dest, DType::FP32, kD, t, 1),
            op, "dest must be FP32 [2304, T, S] or [2304, T, 1]");
    require(contiguous_shape(out, DType::BF16, 2 * kD, t, s), op, "out must be BF16 [4608, T, S]");
    const std::int64_t dest_stream = dest.ne[2] == 1 ? 0 : static_cast<std::int64_t>(t) * kD;
    pair_rms_norm_kernel<<<dim3(static_cast<unsigned>(t), static_cast<unsigned>(s)), kThreads, 0, stream>>>(
        static_cast<const float*>(dest.data), dest_stream, static_cast<const float*>(message.data),
        static_cast<bf16*>(out.data), t);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_blend_rms_norm(const Tensor& dest, const Tensor& message, const Tensor& gain_forget,
                            const Tensor& norm_weight, float eps, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "inkling_blend_rms_norm";
    const int t = message.ne[1];
    const int s = message.ne[2];
    require(contiguous_shape(message, DType::FP32, kD, t, s) && t > 0 && s > 0, op,
            "message must be FP32 [2304, T, S]");
    require(contiguous_shape(dest, DType::FP32, kD, t, s) || contiguous_shape(dest, DType::FP32, kD, t, 1),
            op, "dest must be FP32 [2304, T, S] or [2304, T, 1]");
    require(contiguous_shape(gain_forget, DType::FP32, 2 * kD, t, s), op,
            "gain_forget must be FP32 [4608, T, S]");
    require(contiguous_shape(norm_weight, DType::BF16, kD, s, 1) ||
                contiguous_shape(norm_weight, DType::BF16, kD, 1, 1),
            op, "norm_weight must be BF16 [2304, S] or [2304, 1]");
    require(contiguous_shape(out, DType::BF16, kD, t, s), op, "out must be BF16 [2304, T, S]");
    const std::int64_t dest_stream = dest.ne[2] == 1 ? 0 : static_cast<std::int64_t>(t) * kD;
    const int w_stream             = norm_weight.ne[1] == 1 ? 0 : kD;
    blend_rms_norm_kernel<<<dim3(static_cast<unsigned>(t), static_cast<unsigned>(s)), kThreads, 0, stream>>>(
        static_cast<const float*>(dest.data), dest_stream, static_cast<const float*>(message.data),
        static_cast<const float*>(gain_forget.data), static_cast<const bf16*>(norm_weight.data),
        w_stream, eps, static_cast<bf16*>(out.data), t);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_short_conv(const InklingPlane& x, std::int32_t channels, const Tensor& weight,
                        const InklingConvRows& rows, const InklingConvStateView& state,
                        std::int32_t channel_offset, const InklingPlane* residual,
                        const InklingPlane* addend, const InklingPlane& out, cudaStream_t stream,
                        std::int32_t capture_column, std::int32_t capture_slot) {
    constexpr const char* op = "inkling_short_conv";
    ConvArgs a = conv_args(x, channels, &weight, rows, state, channel_offset, op);
    require(out.data != nullptr && out.dtype == DType::BF16 && out.column_stride >= channels, op,
            "out must be a BF16 plane");
    a.out        = static_cast<bf16*>(out.data);
    a.out_stride = out.column_stride;
    if (addend != nullptr) {
        require(addend->data != nullptr && addend->dtype == DType::BF16, op, "addend must be BF16");
        a.addend        = static_cast<const bf16*>(addend->data);
        a.addend_stride = addend->column_stride;
    }
    if (residual != nullptr) {
        require(addend != nullptr, op, "a residual needs an addend (the reference's form)");
        require(residual->data != nullptr && residual->dtype == DType::BF16, op,
                "residual must be BF16");
        a.residual        = static_cast<const bf16*>(residual->data);
        a.residual_stride = residual->column_stride;
    }
    require(capture_slot < 0 || (rows.batch == 1 && capture_column >= 0 && capture_column <= rows.width),
            op, "a capture needs one row and a column in [0, W]");
    const dim3 grid(grid_count(channels, kConvChannelTile), grid_count(rows.width, kConvColumnTile),
                    static_cast<unsigned>(rows.batch));
    short_conv_kernel<<<grid, kConvChannelTile, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
    conv_window_kernel<<<dim3(grid_count(channels, kConvChannelTile), static_cast<unsigned>(rows.batch)),
                         kConvChannelTile, 0, stream>>>(a, capture_column, capture_slot, 1);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_conv_capture_window(const InklingPlane& x, std::int32_t channels,
                                 const InklingConvRows& rows, const InklingConvStateView& state,
                                 std::int32_t channel_offset, std::int32_t column,
                                 std::int32_t dst_slot, cudaStream_t stream) {
    constexpr const char* op = "inkling_conv_capture_window";
    const ConvArgs a = conv_args(x, channels, nullptr, rows, state, channel_offset, op);
    require(rows.batch == 1 && column >= 0 && column <= rows.width && dst_slot >= 0, op,
            "one row, a column in [0, W] and a slot");
    conv_window_kernel<<<dim3(grid_count(channels, kConvChannelTile), 1), kConvChannelTile, 0, stream>>>(
        a, column, dst_slot, 0);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_conv_copy_slot(const InklingConvStateView& state, std::int32_t channel_offset,
                            std::int32_t channels, std::int32_t src, std::int32_t dst,
                            cudaStream_t stream) {
    constexpr const char* op = "inkling_conv_copy_slot";
    require(state.base != nullptr && src >= 0 && dst >= 0 && channels > 0 && channel_offset >= 0 &&
                channel_offset + channels <= G::kConvChannels,
            op, "invalid state view, slots or channel range");
    if (src == dst) { return; }
    conv_copy_slot_kernel<<<grid_count(channels, 256), 256, 0, stream>>>(
        static_cast<bf16*>(state.base) + channel_offset, state.slot_stride, state.tap_stride, channels,
        src, dst);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_conv_backup(const InklingConvPool& pool, const Tensor& slots, std::int32_t batch,
                         const InklingConvRecords& records, cudaStream_t stream) {
    constexpr const char* op = "inkling_conv_backup";
    require(pool.base && pool.visits > 0 && records.old && records.visits == pool.visits, op,
            "pool and records disagree");
    require(batch > 0 && batch <= records.batch_max && slots.dtype == DType::I32 &&
                slots.numel() >= batch && slots.data,
            op, "slots must be I32 [B], B <= the records' batch");
    conv_backup_kernel<<<dim3(grid_count(G::kConvChannels, 256), static_cast<unsigned>(pool.visits),
                              static_cast<unsigned>(batch)),
                         256, 0, stream>>>(static_cast<const bf16*>(pool.base), pool.visit_stride,
                                           pool.slot_stride, static_cast<const std::int32_t*>(slots.data),
                                           static_cast<bf16*>(records.old), pool.visits);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_conv_record(const InklingPlane& x, std::int32_t channels, std::int32_t width,
                         std::int32_t batch, std::int32_t visit, std::int32_t channel_offset,
                         const InklingConvRecords& records, cudaStream_t stream) {
    constexpr const char* op = "inkling_conv_record";
    require(x.data && x.dtype == DType::BF16 && x.column_stride >= channels, op, "x must be a BF16 plane");
    require(records.inputs && visit >= 0 && visit < records.visits && width > 0 &&
                width <= records.width_max && batch > 0 && batch <= records.batch_max,
            op, "round outside the records");
    require(channels > 0 && channel_offset >= 0 && channel_offset + channels <= G::kConvChannels, op,
            "channel range is outside the conv state");
    conv_record_kernel<<<dim3(grid_count(channels, 256), static_cast<unsigned>(width),
                              static_cast<unsigned>(batch)),
                         256, 0, stream>>>(static_cast<const bf16*>(x.data), x.column_stride, channels,
                                           width, visit, channel_offset,
                                           static_cast<bf16*>(records.inputs), records.visits,
                                           records.width_max);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_conv_fold(const InklingConvPool& pool, std::span<const InklingConvFoldRow> rows,
                       const InklingConvRecords& records, cudaStream_t stream) {
    constexpr const char* op = "inkling_conv_fold";
    require(pool.base && records.old && records.inputs && records.visits == pool.visits, op,
            "pool and records disagree");
    if (rows.empty()) { return; }
    require(rows.size() <= static_cast<std::size_t>(kMaxFoldRows), op, "too many rows");
    FoldArgs f;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        require(r.row >= 0 && r.row < records.batch_max && r.slot >= 0 && r.committed >= 0 &&
                    r.committed <= records.width_max,
                op, "row outside the records");
        f.rows[i] = r;
    }
    f.count = static_cast<int>(rows.size());
    conv_fold_kernel<<<dim3(grid_count(G::kConvChannels, 256), static_cast<unsigned>(pool.visits),
                            static_cast<unsigned>(rows.size())),
                       256, 0, stream>>>(f, static_cast<bf16*>(pool.base), pool.visit_stride,
                                         pool.slot_stride, static_cast<const bf16*>(records.old),
                                         static_cast<const bf16*>(records.inputs), records.visits,
                                         records.width_max);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_softcap_f32(float* logits, std::int32_t rows, std::int32_t tokens, std::int32_t ldc,
                         float cap, cudaStream_t stream) {
    constexpr const char* op = "inkling_softcap_f32";
    require(logits != nullptr && rows > 0 && tokens > 0 && ldc >= rows && cap > 0.0F, op,
            "invalid logits or cap");
    const std::int64_t total = static_cast<std::int64_t>(rows) * tokens;
    softcap_f32_kernel<<<grid_count(total, 256), 256, 0, stream>>>(logits, rows, tokens, ldc, cap);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops
