// NAIVE Inkling-2 attention (lane e1; replaced by lane c's kernels): `inkling_attention_prologue`,
// `inkling_rel_attention` and `inkling_gate_sink` of inkling_ref.py, one CTA per column (prologue) and per
// (column, head) (attention), fp32 throughout. Correctness only.

#include "targets/inkling2/impl/naive/naive.h"

#include "core/device.h"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <stdexcept>

namespace sinfer::targets::inkling2::naive {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kHeadDim = 128;
constexpr int kQHeads  = 24;
constexpr int kKvHeads = 8;
constexpr int kRel     = 16;
constexpr int kRelExt  = 1024;
constexpr int kWindow  = 1024;
constexpr int kRowsR    = 3072;
constexpr int kRowsGate = 3456;
constexpr int kRowsSink = 6528;
constexpr int kRowsK    = 6552;
constexpr int kRowsV    = 7576;
constexpr float kQkClip = 22.627416997969522F;
constexpr float kRelClip = 8.0F;

__device__ __forceinline__ float f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

__device__ __forceinline__ float rsqrt_approx(float x) {
    float y;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

/// Sum over the 128 threads (4 warps); every thread gets the total.
__device__ __forceinline__ float sum128(float v, float* scratch) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) { v += __shfl_xor_sync(0xffffffffU, v, off); }
    __syncthreads();
    if ((threadIdx.x & 31) == 0) { scratch[threadIdx.x >> 5] = v; }
    __syncthreads();
    return (scratch[0] + scratch[1]) + (scratch[2] + scratch[3]);
}

__device__ __forceinline__ float max128(float v, float* scratch) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xffffffffU, v, off)); }
    __syncthreads();
    if ((threadIdx.x & 31) == 0) { scratch[threadIdx.x >> 5] = v; }
    __syncthreads();
    return fmaxf(fmaxf(scratch[0], scratch[1]), fmaxf(scratch[2], scratch[3]));
}

struct Args {
    const bf16* qkvr;
    std::int64_t qkvr_stride;
    int width, batch;
    const std::int32_t* positions;
    const std::int32_t* valid;
    const std::int32_t* table_rows;
    const std::int32_t* block_tables;
    std::int64_t table_stride;
    void* k_pages;
    void* v_pages;
    int fp8;
    int global;
    const bf16* q_scale;
    const bf16* r_scale;
    const bf16* rel_proj;
    const bf16* k_conv;  // local: (c, k) at c * 4 + k (oldest tap first)
    const bf16* v_conv;
    const bf16* k_static; // global: (c, lag) at lag * 1024 + c
    const bf16* v_static;
    const bf16* k_dyn;    // global: [32, N], row = kv_head * 4 + lag
    const bf16* v_dyn;
    std::int64_t dyn_stride;
    const bf16* state;
    std::int64_t slot_stride, tap_stride;
    const std::int32_t* slots;
    bf16* q_out;
    bf16* r_out;
    bf16* out;
};

__device__ __forceinline__ bool column_valid(const Args& a, int n, int& b, int& j) {
    b = n / a.width;
    j = n - b * a.width;
    const int valid = a.valid != nullptr ? a.valid[b] : a.width;
    return j < valid;
}

__device__ __forceinline__ float tau_of(int pos) {
    // 1 + 0.1 * ln(max((pos + 1) * (1/8192), 1)), product and sum rounded separately (-fmad=false)
    const float t = static_cast<float>(pos + 1) * (1.0F / 8192.0F);
    return 1.0F + 0.1F * logf(fmaxf(t, 1.0F));
}

/// The conv input of stream channel `c` (K: row base kRowsK, V: kRowsV) at sequence index i of row b:
/// S = [state tap 0, 1, 2, column 0, 1, ...].
__device__ __forceinline__ float conv_input(const Args& a, int row_base, int state_channel, int b, int i, int c) {
    if (i < 3) {
        const std::int64_t slot = a.slots[b];
        return f(a.state[slot * a.slot_stride + i * a.tap_stride + state_channel + c]);
    }
    const std::int64_t n = static_cast<std::int64_t>(i - 3) + static_cast<std::int64_t>(a.width) * b;
    return f(a.qkvr[n * a.qkvr_stride + row_base + c]);
}

/// `inkling_short_conv` (static, local) or `inkling_dynamic_conv` (global) of one K / V channel -> bf16 value.
__device__ float kv_conv(const Args& a, bool is_k, int b, int j, int n, int pos, int c) {
    const int row_base      = is_k ? kRowsK : kRowsV;
    const int state_channel = is_k ? 0 : 1024;
    const float cur         = conv_input(a, row_base, state_channel, b, j + 3, c);
    if (!a.global) {
        const bf16* w = is_k ? a.k_conv : a.v_conv;
        float acc     = 0.0F;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int lag = 3 - k;
            const float v = pos - lag >= 0 ? conv_input(a, row_base, state_channel, b, j + 3 - lag, c) : 0.0F;
            acc           = fmaf(v, f(w[c * 4 + k]), acc);
        }
        return rbf((acc + cur) * 0.44721359549995793F);
    }
    const bf16* sw  = is_k ? a.k_static : a.v_static;
    const bf16* dyn = is_k ? a.k_dyn : a.v_dyn;
    const int head  = c / kHeadDim;
    float acc       = 0.0F;
#pragma unroll
    for (int lag = 0; lag < 4; ++lag) {
        const float v = pos - lag >= 0 ? conv_input(a, row_base, state_channel, b, j + 3 - lag, c) : 0.0F;
        const float w = f(sw[lag * 1024 + c]) + f(dyn[static_cast<std::int64_t>(n) * a.dyn_stride + head * 4 + lag]);
        acc           = lag == 0 ? v * w : fmaf(v, w, acc);
    }
    return rbf((acc + cur) * 0.33333333333333331F);
}

__global__ void __launch_bounds__(128) prologue_kernel(const __grid_constant__ Args a) {
    __shared__ float scratch[4];
    const int n = blockIdx.x;
    const int d = threadIdx.x;
    int b, j;
    if (!column_valid(a, n, b, j)) { return; }
    const int pos     = a.positions[n];
    const float tau   = a.global ? tau_of(pos) : 1.0F;
    const bf16* col   = a.qkvr + static_cast<std::int64_t>(n) * a.qkvr_stride;
    // q: scale, L2 clip, bf16; global: * tau, bf16
    for (int h = 0; h < kQHeads; ++h) {
        const float v     = f(col[h * kHeadDim + d]) * f(a.q_scale[h * kHeadDim + d]);
        const float norm  = sqrtf(sum128(v * v, scratch));
        const float scale = fminf(kQkClip / (norm + 1e-6F), 1.0F);
        float y           = rbf(v * scale);
        if (a.global) { y = rbf(y * tau); }
        a.q_out[static_cast<std::int64_t>(n) * 3072 + h * kHeadDim + d] = __float2bfloat16_rn(y);
    }
    // r: 24 heads x 16, one head per 16 lanes
    for (int h0 = 0; h0 < kQHeads; h0 += 8) {
        const int h  = h0 + d / kRel;
        const int jj = d % kRel;
        const float v = f(col[kRowsR + h * kRel + jj]) * f(a.r_scale[h * kRel + jj]);
        float s       = v * v;
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) { s += __shfl_xor_sync(0xffffffffU, s, off); }
        const float scale = fminf(kRelClip / (sqrtf(s) + 1e-6F), 1.0F);
        float y           = rbf(v * scale);
        if (a.global) { y = rbf(y * tau); }
        a.r_out[static_cast<std::int64_t>(n) * 384 + h * kRel + jj] = __float2bfloat16_rn(y);
    }
    // k / v: conv -> bf16 -> clip / unweighted RMS norm -> cache (e4m3 straight from fp32, or bf16)
    const std::int32_t* table = a.block_tables + static_cast<std::int64_t>(a.table_rows[b]) * a.table_stride;
    const std::int64_t page   = table[pos >> ops::kPagedKVPageShift];
    for (int hk = 0; hk < kKvHeads; ++hk) {
        const int c     = hk * kHeadDim + d;
        const float kv  = kv_conv(a, true, b, j, n, pos, c);
        const float vv  = kv_conv(a, false, b, j, n, pos, c);
        const float kn  = sqrtf(sum128(kv * kv, scratch));
        const float kc  = kv * fminf(kQkClip / (kn + 1e-6F), 1.0F);
        const float vs  = sum128(vv * vv, scratch);
        const float vc  = vv * rsqrt_approx(__fdiv_rn(vs, 128.0F) + 1e-6F);
        const std::int64_t off =
            static_cast<std::int64_t>(kHeadDim) * sinfer::kPagedKVPageSize * (hk + static_cast<std::int64_t>(kKvHeads) * page) +
            static_cast<std::int64_t>(kHeadDim) * (pos & ops::kPagedKVPageMask) + d;
        if (a.fp8) {
            static_cast<__nv_fp8_e4m3*>(a.k_pages)[off] = __nv_fp8_e4m3(kc);
            static_cast<__nv_fp8_e4m3*>(a.v_pages)[off] = __nv_fp8_e4m3(vc);
        } else {
            static_cast<bf16*>(a.k_pages)[off] = __float2bfloat16_rn(kc);
            static_cast<bf16*>(a.v_pages)[off] = __float2bfloat16_rn(vc);
        }
    }
}

__device__ __forceinline__ float load_kv(const void* pages, int fp8, std::int64_t off) {
    if (fp8) { return static_cast<float>(static_cast<const __nv_fp8_e4m3*>(pages)[off]); }
    return f(static_cast<const bf16*>(pages)[off]);
}

__global__ void __launch_bounds__(128) attention_kernel(const __grid_constant__ Args a) {
    __shared__ float q_s[kHeadDim];
    __shared__ float r_s[kRel];
    __shared__ float s_s[128];
    __shared__ int page_s[128];
    __shared__ float scratch[4];
    const int n  = blockIdx.x;
    const int h  = blockIdx.y;
    const int d  = threadIdx.x;
    int b, j;
    bf16* out = a.out + static_cast<std::int64_t>(n) * 3072 + h * kHeadDim;
    if (!column_valid(a, n, b, j)) {
        out[d] = __float2bfloat16_rn(0.0F);
        return;
    }
    const int hk  = h / 3;
    const int pos = a.positions[n];
    q_s[d] = f(a.q_out[static_cast<std::int64_t>(n) * 3072 + h * kHeadDim + d]);
    if (d < kRel) { r_s[d] = f(a.r_out[static_cast<std::int64_t>(n) * 384 + h * kRel + d]); }
    __syncthreads();
    const std::int32_t* table = a.block_tables + static_cast<std::int64_t>(a.table_rows[b]) * a.table_stride;
    const int lo = a.global ? 0 : max(0, pos - (kWindow - 1));
    const int hi = pos;
    float m = -INFINITY, l = 0.0F, acc = 0.0F;
    for (int k0 = lo; k0 <= hi; k0 += 128) {
        const int key = k0 + d;
        float s       = -INFINITY;
        int page      = 0;
        if (key <= hi) {
            page = table[key >> ops::kPagedKVPageShift];
            const std::int64_t base =
                static_cast<std::int64_t>(kHeadDim) * sinfer::kPagedKVPageSize * (hk + static_cast<std::int64_t>(kKvHeads) * page) +
                static_cast<std::int64_t>(kHeadDim) * (key & ops::kPagedKVPageMask);
            float dot = 0.0F;
            for (int e = 0; e < kHeadDim; ++e) { dot = fmaf(q_s[e], load_kv(a.k_pages, a.fp8, base + e), dot); }
            s = dot * 0.08838834764831845F;
            const int dist = pos - key;
            if (dist < kRelExt) {
                float bias = 0.0F;
#pragma unroll
                for (int e = 0; e < kRel; ++e) { bias = fmaf(r_s[e], f(a.rel_proj[e * kRelExt + dist]), bias); }
                s = s + rbf(bias);
            }
        }
        s_s[d]    = s;
        page_s[d] = page;
        const float chunk_max = max128(s, scratch);
        const float m_new     = fmaxf(m, chunk_max);
        const float rescale   = m == -INFINITY ? 0.0F : expf(m - m_new);
        l   = l * rescale;
        acc = acc * rescale;
        const int count = min(128, hi - k0 + 1);
        for (int i = 0; i < count; ++i) {
            const float p = expf(s_s[i] - m_new);
            l += p;
            const int key_i = k0 + i;
            const std::int64_t off =
                static_cast<std::int64_t>(kHeadDim) * sinfer::kPagedKVPageSize * (hk + static_cast<std::int64_t>(kKvHeads) * page_s[i]) +
                static_cast<std::int64_t>(kHeadDim) * (key_i & ops::kPagedKVPageMask) + d;
            acc = fmaf(p, load_kv(a.v_pages, a.fp8, off), acc);
        }
        m = m_new;
        __syncthreads();
    }
    const float o    = acc / l;
    const float lse  = m + logf(l);
    const bf16* col  = a.qkvr + static_cast<std::int64_t>(n) * a.qkvr_stride;
    const float gate = f(col[kRowsGate + h * kHeadDim + d]);
    const float sink = f(col[kRowsSink + h]);
    const float sg   = 1.0F / (1.0F + expf(-gate));
    const float ss   = 1.0F / (1.0F + expf(-(lse - sink)));
    out[d]           = __float2bfloat16_rn((o * sg) * ss);
}

__global__ void fill_i32_kernel(std::int32_t* data, std::int32_t count, std::int32_t value) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { data[i] = value; }
}

} // namespace

void attention(const AttentionCall& c, cudaStream_t stream) {
    if (c.qkvr == nullptr || c.layer.q_scale == nullptr || c.width <= 0 || c.batch <= 0 || c.positions == nullptr ||
        c.table_rows == nullptr || c.block_tables == nullptr || c.k_pages == nullptr || c.v_pages == nullptr ||
        c.state == nullptr || c.slots == nullptr || c.q_out == nullptr || c.r_out == nullptr || c.out == nullptr) {
        throw std::invalid_argument("naive inkling attention: incomplete call");
    }
    const AttentionLayer& L = c.layer;
    Args a{};
    a.qkvr         = static_cast<const bf16*>(c.qkvr);
    a.qkvr_stride  = c.qkvr_stride;
    a.width        = c.width;
    a.batch        = c.batch;
    a.positions    = c.positions;
    a.valid        = c.valid;
    a.table_rows   = c.table_rows;
    a.block_tables = c.block_tables;
    a.table_stride = c.table_stride;
    a.k_pages      = c.k_pages;
    a.v_pages      = c.v_pages;
    a.fp8          = c.fp8 ? 1 : 0;
    a.global       = L.global ? 1 : 0;
    a.q_scale      = static_cast<const bf16*>(L.q_scale);
    a.r_scale      = static_cast<const bf16*>(L.r_scale);
    a.rel_proj     = static_cast<const bf16*>(L.rel_proj);
    if (L.global) {
        if (c.k_dyn == nullptr || c.v_dyn == nullptr || L.k_static == nullptr || L.v_static == nullptr) {
            throw std::invalid_argument("naive inkling attention: a global layer needs static taps and dynamic taps");
        }
        a.k_static   = static_cast<const bf16*>(L.k_static);
        a.v_static   = static_cast<const bf16*>(L.v_static);
        a.k_dyn      = static_cast<const bf16*>(c.k_dyn);
        a.v_dyn      = static_cast<const bf16*>(c.v_dyn);
        a.dyn_stride = c.dyn_stride;
    } else {
        a.k_conv = static_cast<const bf16*>(L.k_conv);
        a.v_conv = static_cast<const bf16*>(L.v_conv);
    }
    a.state       = static_cast<const bf16*>(c.state);
    a.slot_stride = c.slot_stride;
    a.tap_stride  = c.tap_stride;
    a.slots       = c.slots;
    a.q_out       = static_cast<bf16*>(c.q_out);
    a.r_out       = static_cast<bf16*>(c.r_out);
    a.out         = static_cast<bf16*>(c.out);
    const unsigned columns = static_cast<unsigned>(c.width) * static_cast<unsigned>(c.batch);
    prologue_kernel<<<columns, 128, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
    attention_kernel<<<dim3(columns, kQHeads), 128, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
}

void fill_i32(std::int32_t* data, std::int32_t count, std::int32_t value, cudaStream_t stream) {
    if (count <= 0) { return; }
    fill_i32_kernel<<<(count + 255) / 256, 256, 0, stream>>>(data, count, value);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::targets::inkling2::naive
