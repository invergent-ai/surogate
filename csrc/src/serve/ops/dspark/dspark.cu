// DSpark drafter ops: api/ops/dspark.h. One translation unit: kernels, launchers and the
// host-side validation. Compiled with -fmad=false (CMakeLists.txt): every product / sum the reference
// (ref/dspark_ref.py) does not fuse stays unfused; dot products use explicit fmaf (their order differs from the
// reference's GEMMs anyway, and the golden-vector tolerances say so).

#include "api/ops/dspark.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace {

using bf16 = __nv_bfloat16;
using G    = DSparkGeometry;

constexpr int kD       = G::kHidden; // 2304
constexpr int kThreads = 256;
constexpr int kPer     = kD / kThreads; // 9
static_assert(kD % kThreads == 0);
constexpr int kMaxLanes = 16; // dspark_markov_vocab keeps every lane's latent in shared memory

__device__ __forceinline__ float f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ bf16 tob(float v) { return __float2bfloat16_rn(v); }

/// Deterministic block sum over blockDim.x threads (a multiple of 32, <= 1024): a fixed shuffle tree per warp,
/// then the warps' sums in warp order. Every thread gets the total. `scratch` >= blockDim.x / 32 floats.
__device__ __forceinline__ float block_sum(float v, float* scratch) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) { v = __fadd_rn(v, __shfl_xor_sync(0xffffffffU, v, off)); }
    __syncthreads();
    if ((threadIdx.x & 31) == 0) { scratch[threadIdx.x >> 5] = v; }
    __syncthreads();
    float total = 0.0F;
    const int warps = static_cast<int>(blockDim.x >> 5);
    for (int w = 0; w < warps; ++w) { total = __fadd_rn(total, scratch[w]); }
    return total;
}

/// rstd = 1 / sqrt(sum / n + eps), all IEEE (torch CPU's rsqrt = div(1, sqrt)).
__device__ __forceinline__ float rstd_of(float sum, float n, float eps) {
    return __fdiv_rn(1.0F, __fsqrt_rn(__fadd_rn(__fdiv_rn(sum, n), eps)));
}

// ------------------------------------------------------------------------------------------------ feature mix
__global__ void __launch_bounds__(kThreads)
feature_mix_kernel(const bf16* features, std::int64_t stride, const float* alpha, const bf16* norms, int rows,
                   float eps, bf16* out, int columns) {
    __shared__ float scratch[kThreads / 32];
    const int t     = blockIdx.x;
    const bf16* col = features + static_cast<std::int64_t>(t) * stride;
    float x[G::kFeatures][kPer];
#pragma unroll
    for (int fi = 0; fi < G::kFeatures; ++fi) {
#pragma unroll
        for (int i = 0; i < kPer; ++i) { x[fi][i] = f(col[fi * kD + threadIdx.x + i * kThreads]); }
    }
    for (int r = 0; r < rows; ++r) {
        float m[kPer];
        float ss = 0.0F;
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            float acc = __fmul_rn(x[0][i], alpha[r * 4 + 0]);
#pragma unroll
            for (int fi = 1; fi < G::kFeatures; ++fi) { acc = __fadd_rn(acc, __fmul_rn(x[fi][i], alpha[r * 4 + fi])); }
            m[i] = acc;
            ss   = __fadd_rn(ss, __fmul_rn(acc, acc));
        }
        const float rstd = rstd_of(block_sum(ss, scratch), static_cast<float>(kD), eps);
        bf16* o          = out + (static_cast<std::int64_t>(r) * columns + t) * kD;
        const bf16* w    = norms + static_cast<std::int64_t>(r) * kD;
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int e = threadIdx.x + i * kThreads;
            o[e]        = tob(__fmul_rn(__fmul_rn(m[i], rstd), f(w[e])));
        }
    }
}

// ------------------------------------------------------------------------------------- per-head norm + RoPE
__global__ void __launch_bounds__(128)
head_norm_rope_kernel(const bf16* x, std::int64_t x_stride, const bf16* weight, const std::int32_t* positions,
                      const float* inv_freq, float eps, bf16* out, std::int64_t out_stride) {
    __shared__ float scratch[4];
    __shared__ float y_s[G::kHeadDim];
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const float v    = f(x[static_cast<std::int64_t>(t) * x_stride + h * G::kHeadDim + d]);
    const float rstd = rstd_of(block_sum(__fmul_rn(v, v), scratch), static_cast<float>(G::kHeadDim), eps);
    const float y    = rbf(__fmul_rn(__fmul_rn(v, rstd), f(weight[d])));
    float o          = y;
    if (positions != nullptr) {
        y_s[d] = y;
        __syncthreads();
        constexpr int half = G::kHeadDim / 2;
        const int i        = d & (half - 1);
        // positions is a per-round buffer written by an earlier node of the round's graph: read it with ld.cg (the
        // compiler emitted ld.global.nc here on type-based alias grounds; PDL audit). inv_freq is a load-time
        // constant. Same bits.
        const float angle  = __fmul_rn(static_cast<float>(__ldcg(positions + t)), inv_freq[i]);
        const float c      = cosf(angle);
        const float s      = sinf(angle);
        const float y1     = y_s[i];
        const float y2     = y_s[i + half];
        o = d < half ? __fsub_rn(__fmul_rn(y1, c), __fmul_rn(y2, s)) : __fadd_rn(__fmul_rn(y2, c), __fmul_rn(y1, s));
    }
    out[static_cast<std::int64_t>(t) * out_stride + h * G::kHeadDim + d] = tob(o);
}

// ------------------------------------------------------------------------------------------ add + RMSNorm
__global__ void __launch_bounds__(kThreads)
add_rms_norm_kernel(const bf16* x, const bf16* resid, const bf16* weight, float eps, bf16* resid_out, bf16* h) {
    __shared__ float scratch[kThreads / 32];
    const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * kD;
    float s[kPer];
    float ss = 0.0F;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int e = threadIdx.x + i * kThreads;
        const float r = f(resid[base + e]);
        s[i]          = x != nullptr ? __fadd_rn(f(x[base + e]), r) : r;
        ss            = __fadd_rn(ss, __fmul_rn(s[i], s[i]));
    }
    const float rstd = rstd_of(block_sum(ss, scratch), static_cast<float>(kD), eps);
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int e = threadIdx.x + i * kThreads;
        if (resid_out != nullptr) { resid_out[base + e] = tob(s[i]); }
        h[base + e] = tob(__fmul_rn(__fmul_rn(s[i], rstd), f(weight[e])));
    }
}

// ------------------------------------------------------------------------------------------------ SiLU * up
__global__ void silu_mul_kernel(const bf16* gate_up, int columns, bf16* y) {
    const std::int64_t idx   = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = static_cast<std::int64_t>(G::kIntermediate) * columns;
    if (idx >= total) { return; }
    const std::int64_t t = idx / G::kIntermediate;
    const int i          = static_cast<int>(idx - t * G::kIntermediate);
    const bf16* col      = gate_up + t * (2 * G::kIntermediate);
    const float g        = f(col[i]);
    const float u        = f(col[G::kIntermediate + i]);
    y[idx]               = tob(__fmul_rn(__fdiv_rn(g, __fadd_rn(1.0F, expf(-g))), u));
}

// ---------------------------------------------------------------------------------------------- embed block
__global__ void __launch_bounds__(kThreads)
embed_block_kernel(const std::int32_t* anchors, const bf16* embed, const bf16* embed_norm, const bf16* mask_embed,
                   float eps, bf16* out) {
    __shared__ float scratch[kThreads / 32];
    const int b       = blockIdx.x;
    const bf16* row   = embed + static_cast<std::int64_t>(anchors[b]) * kD;
    bf16* lane        = out + static_cast<std::int64_t>(b) * G::kBlock * kD;
    float e[kPer];
    float ss = 0.0F;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        e[i] = f(row[threadIdx.x + i * kThreads]);
        ss   = __fadd_rn(ss, __fmul_rn(e[i], e[i]));
    }
    const float rstd = rstd_of(block_sum(ss, scratch), static_cast<float>(kD), eps);
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int c = threadIdx.x + i * kThreads;
        lane[c]     = tob(__fmul_rn(__fmul_rn(e[i], rstd), f(embed_norm[c])));
        const bf16 m = mask_embed[c];
#pragma unroll
        for (int j = 1; j < G::kBlock; ++j) { lane[j * kD + c] = m; }
    }
}

__global__ void block_positions_kernel(const std::int32_t* anchor_positions, int batch, std::int32_t* positions) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch * G::kBlock) { positions[i] = anchor_positions[i / G::kBlock] + (i % G::kBlock); }
}

__global__ void block_query_positions_kernel(const std::int32_t* anchor_positions, int batch, std::int32_t* positions) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch * G::kBlock) { positions[i] = anchor_positions[i / G::kBlock] + (G::kBlock - 1); }
}

// ------------------------------------------------------------------------------------------- paged KV write
struct PlaneDev {
    void* k_pages;
    void* v_pages;
    const std::int32_t* block_tables;
    std::int64_t table_stride;
    int fp8;
    std::int32_t limit;
};

PlaneDev plane_dev(const DSparkPagedPlane& p, const char* op) {
    if (p.k_pages == nullptr || p.v_pages == nullptr || p.block_tables == nullptr || p.table_stride <= 0) {
        throw std::invalid_argument(std::string(op) + ": the draft KV plane is incomplete");
    }
    return PlaneDev{p.k_pages, p.v_pages, p.block_tables, p.table_stride, p.fp8 ? 1 : 0, p.limit};
}

__device__ __forceinline__ std::int64_t plane_offset(std::int32_t page, int head, std::int32_t pos, int d) {
    return paged_kv_element_offset<G::kHeadDim, G::kKvHeads>(page, head, pos & kPagedKVPageMask, d);
}

__device__ __forceinline__ float load_kv(const void* pages, int fp8, std::int64_t off) {
    if (fp8) { return static_cast<float>(static_cast<const __nv_fp8_e4m3*>(pages)[off]); }
    return f(static_cast<const bf16*>(pages)[off]);
}

__device__ __forceinline__ void store_kv(void* pages, int fp8, std::int64_t off, float v) {
    if (fp8) {
        static_cast<__nv_fp8_e4m3*>(pages)[off] = __nv_fp8_e4m3(v); // satfinite, round to nearest even
    } else {
        static_cast<bf16*>(pages)[off] = tob(v);
    }
}

__global__ void __launch_bounds__(kThreads)
kv_write_kernel(const bf16* k, const bf16* v, std::int64_t stride, const std::int32_t* positions, int width,
                const std::int32_t* counts, const std::int32_t* table_rows, PlaneDev plane) {
    const int n = blockIdx.x;
    const int b = n / width;
    const int j = n - b * width;
    if (counts != nullptr && j >= counts[b]) { return; }
    const std::int32_t pos    = positions[n];
    if (pos >= plane.limit) { return; }
    const std::int32_t* table = plane.block_tables + static_cast<std::int64_t>(table_rows[b]) * plane.table_stride;
    const std::int32_t page   = paged_kv_physical_page(table, pos);
    const bf16* kc            = k + static_cast<std::int64_t>(n) * stride;
    const bf16* vc            = v + static_cast<std::int64_t>(n) * stride;
    for (int e = threadIdx.x; e < G::kKvRows; e += kThreads) {
        const std::int64_t off = plane_offset(page, e / G::kHeadDim, pos, e % G::kHeadDim);
        store_kv(plane.k_pages, plane.fp8, off, f(kc[e]));
        store_kv(plane.v_pages, plane.fp8, off, f(vc[e]));
    }
}

// --------------------------------------------------------------------------------------- block attention
constexpr int kGroup   = G::kQHeads / G::kKvHeads; // 3
constexpr int kRows    = kGroup * G::kBlock;       // 24 query rows per (kv head, lane)
constexpr int kKeyTile = 128;

__global__ void __launch_bounds__(128)
block_attention_kernel(const bf16* q, std::int64_t q_stride, const std::int32_t* anchors,
                       const std::int32_t* table_rows, int window, float scale, PlaneDev plane, bf16* out) {
    __shared__ float q_s[kRows][G::kHeadDim];
    __shared__ float p_s[kRows][kKeyTile];
    __shared__ float rescale_s[kRows];
    __shared__ float l_s[kRows];
    const int hk = blockIdx.x;
    const int b  = blockIdx.y;
    const int d  = threadIdx.x;
    // row r = j * 3 + g: block column j, query head hk * 3 + g
    for (int r = 0; r < kRows; ++r) {
        const int j = r / kGroup, g = r - (r / kGroup) * kGroup;
        const std::int64_t n = static_cast<std::int64_t>(b) * G::kBlock + j;
        q_s[r][d]            = f(q[n * q_stride + (hk * kGroup + g) * G::kHeadDim + d]);
    }
    const std::int32_t anchor = anchors[b];
    const std::int32_t hi     = min(anchor + G::kBlock, plane.limit); // exclusive
    const std::int32_t lo     = window > 0 ? max(0, hi - window) : 0;
    const std::int32_t* table = plane.block_tables + static_cast<std::int64_t>(table_rows[b]) * plane.table_stride;
    float m = -INFINITY; // running max of row d (d < kRows)
    float l = 0.0F;
    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) { acc[r] = 0.0F; }
    __syncthreads();
    for (std::int32_t k0 = lo; k0 < hi; k0 += kKeyTile) {
        const int count = min(kKeyTile, hi - k0);
        // scores of key k0 + d for every row
        if (d < count) {
            const std::int32_t key = k0 + d;
            const std::int64_t off = plane_offset(paged_kv_physical_page(table, key), hk, key, 0);
            float dot[kRows];
#pragma unroll
            for (int r = 0; r < kRows; ++r) { dot[r] = 0.0F; }
            for (int e = 0; e < G::kHeadDim; ++e) {
                const float kv = load_kv(plane.k_pages, plane.fp8, off + e);
#pragma unroll
                for (int r = 0; r < kRows; ++r) { dot[r] = fmaf(q_s[r][e], kv, dot[r]); }
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r) { p_s[r][d] = __fmul_rn(dot[r], scale); }
        }
        __syncthreads();
        // online softmax per row (thread r owns row r): p = exp(s - m_new), l over the unrounded p
        if (d < kRows) {
            float cmax = -INFINITY;
            for (int i = 0; i < count; ++i) { cmax = fmaxf(cmax, p_s[d][i]); }
            const float m_new   = fmaxf(m, cmax);
            const float rescale = m == -INFINITY ? 0.0F : expf(__fsub_rn(m, m_new));
            l                   = __fmul_rn(l, rescale);
            for (int i = 0; i < count; ++i) {
                const float p = expf(__fsub_rn(p_s[d][i], m_new));
                l             = __fadd_rn(l, p);
                p_s[d][i]     = rbf(p); // P rounded to bf16 for P.V (reference)
            }
            m            = m_new;
            rescale_s[d] = rescale;
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < kRows; ++r) { acc[r] = __fmul_rn(acc[r], rescale_s[r]); }
        for (int i = 0; i < count; ++i) {
            const std::int32_t key = k0 + i;
            const float v = load_kv(plane.v_pages, plane.fp8,
                                    plane_offset(paged_kv_physical_page(table, key), hk, key, d));
#pragma unroll
            for (int r = 0; r < kRows; ++r) { acc[r] = fmaf(p_s[r][i], v, acc[r]); }
        }
        __syncthreads();
    }
    if (d < kRows) { l_s[d] = l; }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        const int j = r / kGroup, g = r - (r / kGroup) * kGroup;
        const std::int64_t n = static_cast<std::int64_t>(b) * G::kBlock + j;
        out[n * G::kQRows + (hk * kGroup + g) * G::kHeadDim + d] = tob(__fdiv_rn(acc[r], l_s[r]));
    }
}

// --------------------------------------------------------------------------------------------- f_rnn store
__global__ void __launch_bounds__(kThreads)
store_last_column_kernel(const bf16* src, int width, const std::int32_t* counts, const std::int32_t* lanes,
                         bf16* dst) {
    const int b = blockIdx.x;
    const int c = counts != nullptr ? counts[b] : width;
    if (c < 1) { return; }
    const bf16* s = src + (static_cast<std::int64_t>(b) * width + (c - 1)) * kD;
    bf16* o       = dst + static_cast<std::int64_t>(lanes != nullptr ? lanes[b] : b) * kD;
    for (int e = threadIdx.x; e < kD; e += kThreads) { o[e] = s[e]; }
}

// ------------------------------------------------------------------------------------------- Markov head
__global__ void __launch_bounds__(kThreads)
markov_pack_kernel(const bf16* hidden, const bf16* f_rnn, const std::int32_t* lanes, int k, bf16* x) {
    const int n = blockIdx.x;
    const int b = n / k;
    const int j = n - b * k;
    const bf16* h  = hidden + (static_cast<std::int64_t>(b) * G::kBlock + 1 + j) * kD;
    const bf16* fr = f_rnn + static_cast<std::int64_t>(lanes != nullptr ? lanes[b] : b) * kD;
    bf16* o        = x + static_cast<std::int64_t>(n) * (2 * kD);
    for (int e = threadIdx.x; e < kD; e += kThreads) {
        o[e]      = h[e];
        o[kD + e] = fr[e];
    }
}

constexpr int kGateIndices = 8; // gate indices i per CTA (one warp each: rows i, 256 + i, 512 + i)

__global__ void __launch_bounds__(32 * kGateIndices)
markov_gate_kernel(const float* ctx, int j, int k, const bf16* bias, const bf16* w_sp, const bf16* w1,
                   const std::int32_t* prev_base, int prev_stride, const bf16* state_in, bf16* state_out,
                   bf16* latent) {
    __shared__ float in_s[G::kRnnStateCols];
    const int b    = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const std::int64_t prev = prev_base[static_cast<std::int64_t>(b) * prev_stride];
    for (int c = threadIdx.x; c < G::kRnnStateCols; c += blockDim.x) {
        in_s[c] = c < G::kRank ? (state_in != nullptr ? f(state_in[static_cast<std::int64_t>(b) * G::kRank + c]) : 0.0F)
                               : f(w1[prev * G::kRank + (c - G::kRank)]);
    }
    __syncthreads();
    const int i = blockIdx.x * kGateIndices + warp;
    float raw[3];
#pragma unroll
    for (int t = 0; t < 3; ++t) {
        const int n = t * G::kRank + i;
        // lane covers inputs [16 lane, 16 lane + 16): two 16-byte loads of the row
        const uint4* row = reinterpret_cast<const uint4*>(w_sp + static_cast<std::int64_t>(n) * G::kRnnStateCols) + 2 * lane;
        const uint4 w0 = row[0], w1v = row[1];
        const bf16* a = reinterpret_cast<const bf16*>(&w0);
        const bf16* c = reinterpret_cast<const bf16*>(&w1v);
        float dot = 0.0F;
#pragma unroll
        for (int e = 0; e < 8; ++e) { dot = fmaf(f(a[e]), in_s[16 * lane + e], dot); }
#pragma unroll
        for (int e = 0; e < 8; ++e) { dot = fmaf(f(c[e]), in_s[16 * lane + 8 + e], dot); }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) { dot = __fadd_rn(dot, __shfl_xor_sync(0xffffffffU, dot, off)); }
        const float c1 = __fadd_rn(ctx[(static_cast<std::int64_t>(b) * k + j) * G::kRnnRows + n], f(bias[n]));
        raw[t]         = __fadd_rn(c1, dot);
    }
    if (lane == 0) {
        float g          = rbf(raw[0]);
        const float cand = rbf(tanhf(rbf(raw[1])));
        const float o    = rbf(raw[2]);
        g                = rbf(__fdiv_rn(1.0F, __fadd_rn(1.0F, expf(-g))));
        const float keep   = rbf(__fmul_rn(g, in_s[i]));
        const float update = rbf(__fmul_rn(rbf(__fsub_rn(1.0F, g)), cand));
        state_out[static_cast<std::int64_t>(b) * G::kRank + i] = tob(__fadd_rn(keep, update));
        latent[static_cast<std::int64_t>(b) * G::kRank + i]    = tob(tanhf(o));
    }
}

__global__ void __launch_bounds__(kThreads)
markov_vocab_kernel(const bf16* base, std::int64_t base_rows, int row, int batch, const bf16* fold_w,
                    const std::uint8_t* fold_codes, const float* fold_scales, const bf16* fold_b, const bf16* latent,
                    int domain, const std::int32_t* ids, float* part_val, std::int32_t* part_id, bf16* bias_out,
                    bf16* logits_out, const SamplingConfig* sampling, const std::int32_t* anchor_positions,
                    int draft_index) {
    __shared__ float lat_s[kMaxLanes][G::kRank];
    __shared__ float wv_s[kThreads / 32][kMaxLanes];
    __shared__ int wi_s[kThreads / 32][kMaxLanes];
    // coupled sampled drafts: per lane, whether to sample, 1 / T and the Gumbel key (dspark.h)
    __shared__ int cp_s[kMaxLanes];
    __shared__ float it_s[kMaxLanes];
    __shared__ unsigned long long seed_s[kMaxLanes];
    __shared__ int kpos_s[kMaxLanes], kpur_s[kMaxLanes];
    for (int idx = threadIdx.x; idx < batch * G::kRank; idx += kThreads) {
        lat_s[idx / G::kRank][idx % G::kRank] = f(latent[idx]);
    }
    if (threadIdx.x < batch) {
        const int b = threadIdx.x;
        int cp      = 0;
        if (sampling != nullptr) {
            const SamplingConfig cfg = sampling[b];
            cp = cfg.speculative_coupled != 0 && sampling_untruncated(cfg) ? 1 : 0;
            if (cp) {
                const SampleKey key = sample_key(cfg, anchor_positions[b] + draft_index, kSamplePurposeDecode);
                it_s[b]   = 1.0F / cfg.temperature;
                seed_s[b] = cfg.seed;
                kpos_s[b] = key.position;
                kpur_s[b] = key.purpose;
            }
        }
        cp_s[b] = cp;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int v0   = blockIdx.x * G::kVocabTile + warp * 32;
    float best_v[kMaxLanes];
    int best_i[kMaxLanes];
#pragma unroll
    for (int b = 0; b < kMaxLanes; ++b) {
        best_v[b] = -INFINITY;
        best_i[b] = INT_MAX;
    }
    for (int rr = 0; rr < 32; ++rr) {
        const int v = v0 + rr;
        if (v >= domain) { break; }
        float wf[8];
        float row_scale = 1.0F;
        if (fold_codes != nullptr) { // FP8 fold: 8 e4m3 codes per lane, the row scale after the sum
            const uint2 packed = reinterpret_cast<const uint2*>(fold_codes + static_cast<std::int64_t>(v) * G::kRank)[lane];
            const __nv_fp8_e4m3* q8 = reinterpret_cast<const __nv_fp8_e4m3*>(&packed);
#pragma unroll
            for (int e = 0; e < 8; ++e) { wf[e] = static_cast<float>(q8[e]); }
            row_scale = fold_scales[v];
        } else {
            const uint4 packed = reinterpret_cast<const uint4*>(fold_w + static_cast<std::int64_t>(v) * G::kRank)[lane];
            const bf16* w8     = reinterpret_cast<const bf16*>(&packed);
#pragma unroll
            for (int e = 0; e < 8; ++e) { wf[e] = f(w8[e]); }
        }
        const float fb     = f(fold_b[v]);
        // lane b of the warp draws lane b's Gumbel for this token once; the b loop takes it with a shuffle
        float g_mine = 0.0F;
        if (lane < batch && cp_s[lane]) {
            g_mine = sampling_gumbel(seed_s[lane], kpos_s[lane], kpur_s[lane],
                                     static_cast<unsigned>(ids != nullptr ? ids[v] : v));
        }
        for (int b = 0; b < batch; ++b) {
            float p = 0.0F;
#pragma unroll
            for (int e = 0; e < 8; ++e) { p = fmaf(wf[e], lat_s[b][lane * 8 + e], p); }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) { p = __fadd_rn(p, __shfl_xor_sync(0xffffffffU, p, off)); }
            if (fold_codes != nullptr) { p = __fmul_rn(p, row_scale); }
            const float bias  = rbf(__fadd_rn(p, fb));
            const float s     = rbf(__fadd_rn(f(base[(static_cast<std::int64_t>(row) + G::kBlock * b) * base_rows + v]),
                                              bias));
            const float logit = rbf(__fmul_rn(G::kSoftcap, tanhf(__fdiv_rn(s, G::kSoftcap))));
            const float g     = __shfl_sync(0xffffffffU, g_mine, b);
            const float z     = cp_s[b] ? __fadd_rn(__fmul_rn(logit, it_s[b]), g) : logit;
            if (z > best_v[b]) { // rows ascend: the first maximum is the lowest id
                best_v[b] = z;
                best_i[b] = v;
            }
            if (lane == 0) {
                if (bias_out != nullptr) { bias_out[static_cast<std::int64_t>(b) * domain + v] = tob(bias); }
                if (logits_out != nullptr) { logits_out[static_cast<std::int64_t>(b) * domain + v] = tob(logit); }
            }
        }
    }
    if (lane == 0) {
        for (int b = 0; b < batch; ++b) {
            wv_s[warp][b] = best_v[b];
            wi_s[warp][b] = best_i[b];
        }
    }
    __syncthreads();
    if (threadIdx.x < batch) {
        const int b = threadIdx.x;
        float bv    = -INFINITY;
        int bi      = INT_MAX;
        for (int w = 0; w < kThreads / 32; ++w) { // warps cover ascending row ranges
            if (wv_s[w][b] > bv) {
                bv = wv_s[w][b];
                bi = wi_s[w][b];
            }
        }
        part_val[static_cast<std::int64_t>(blockIdx.x) * batch + b] = bv;
        part_id[static_cast<std::int64_t>(blockIdx.x) * batch + b] =
            bi == INT_MAX ? INT_MAX : (ids != nullptr ? ids[bi] : bi);
    }
}

__global__ void __launch_bounds__(kThreads)
markov_pick_kernel(const float* part_val, const std::int32_t* part_id, int tiles, int batch, int j, int stride,
                   std::int32_t* drafts) {
    __shared__ float v_s[kThreads];
    __shared__ int i_s[kThreads];
    const int b = blockIdx.x;
    float bv    = -INFINITY;
    int bi      = INT_MAX;
    for (int t = threadIdx.x; t < tiles; t += kThreads) { // a thread's tiles ascend
        const float v = part_val[static_cast<std::int64_t>(t) * batch + b];
        const int id  = part_id[static_cast<std::int64_t>(t) * batch + b];
        if (v > bv || (v == bv && id < bi)) {
            bv = v;
            bi = id;
        }
    }
    v_s[threadIdx.x] = bv;
    i_s[threadIdx.x] = bi;
    __syncthreads();
    // fixed tree over the 256 thread winners: larger value, lowest id on ties (order-independent rule)
    for (int half = kThreads / 2; half > 0; half >>= 1) {
        if (threadIdx.x < half) {
            const float ov = v_s[threadIdx.x + half];
            const int oi   = i_s[threadIdx.x + half];
            if (ov > v_s[threadIdx.x] || (ov == v_s[threadIdx.x] && oi < i_s[threadIdx.x])) {
                v_s[threadIdx.x] = ov;
                i_s[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        bi = i_s[0];
        drafts[j + static_cast<std::int64_t>(stride) * b] = bi == INT_MAX ? 0 : bi;
    }
}

// --------------------------------------------------------------------------------------------------- fold
constexpr int kFoldRows  = 32;
constexpr int kFoldChunk = 32;

__global__ void __launch_bounds__(G::kRank)
fold_kernel(const bf16* unembed, int rows, const std::int32_t* ids, const bf16* hidden_out, const bf16* bias,
            bf16* fold_w, bf16* fold_b) {
    __shared__ float u_s[kFoldRows][kFoldChunk];
    __shared__ float h_s[kFoldChunk][G::kRank];
    const int r  = threadIdx.x;
    const int v0 = blockIdx.x * kFoldRows;
    double acc[kFoldRows];
#pragma unroll
    for (int v = 0; v < kFoldRows; ++v) { acc[v] = 0.0; }
    double accb = 0.0; // fold_b of row v0 + r (r < kFoldRows)
    for (int c0 = 0; c0 < kD; c0 += kFoldChunk) {
        __syncthreads();
        for (int idx = r; idx < kFoldRows * kFoldChunk; idx += G::kRank) {
            const int v = idx / kFoldChunk, cc = idx - (idx / kFoldChunk) * kFoldChunk;
            const int row = v0 + v;
            float u = 0.0F;
            if (row < rows) {
                const std::int64_t src = ids != nullptr ? ids[row] : row;
                u = f(unembed[src * kD + c0 + cc]);
            }
            u_s[v][cc] = u;
        }
        for (int cc = 0; cc < kFoldChunk; ++cc) { h_s[cc][r] = f(hidden_out[static_cast<std::int64_t>(c0 + cc) * G::kRank + r]); }
        __syncthreads();
        for (int cc = 0; cc < kFoldChunk; ++cc) {
            const double h = static_cast<double>(h_s[cc][r]);
#pragma unroll
            for (int v = 0; v < kFoldRows; ++v) { acc[v] = fma(static_cast<double>(u_s[v][cc]), h, acc[v]); }
        }
        if (r < kFoldRows) {
            for (int cc = 0; cc < kFoldChunk; ++cc) {
                accb = fma(static_cast<double>(u_s[r][cc]), static_cast<double>(f(bias[c0 + cc])), accb);
            }
        }
    }
#pragma unroll
    for (int v = 0; v < kFoldRows; ++v) {
        const int row = v0 + v;
        if (row < rows) { fold_w[static_cast<std::int64_t>(row) * G::kRank + r] = __double2bfloat16(acc[v]); }
    }
    if (r < kFoldRows && v0 + r < rows) { fold_b[v0 + r] = __double2bfloat16(accb); }
}

// ------------------------------------------------------------------------------------------ FP8 weights (W8A16)
__global__ void __launch_bounds__(kThreads)
quantize_rows_kernel(const bf16* w, int k, std::uint8_t* codes, float* scales) {
    __shared__ float red[kThreads / 32];
    const std::int64_t r = blockIdx.x;
    const bf16* row      = w + r * k;
    float amax           = 0.0F;
    for (int c = threadIdx.x; c < k; c += kThreads) { amax = fmaxf(amax, fabsf(f(row[c]))); }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffU, amax, off)); }
    if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = amax; }
    __syncthreads();
    float m = 0.0F;
    for (int i = 0; i < kThreads / 32; ++i) { m = fmaxf(m, red[i]); }
    const float scale = fmaxf(__fdiv_rn(m, 448.0F), 1e-12F);
    if (threadIdx.x == 0) { scales[r] = scale; }
    for (int c = threadIdx.x; c < k; c += kThreads) {
        const __nv_fp8_e4m3 q(__fdiv_rn(f(row[c]), scale)); // satfinite, round to nearest even
        codes[r * k + c] = *reinterpret_cast<const std::uint8_t*>(&q);
    }
}

constexpr int kW8Rows = 8;  // rows per CTA (one warp each)
constexpr int kW8Cols = 8;  // columns per pass

__global__ void __launch_bounds__(32 * kW8Rows)
gemm_w8_kernel(const std::uint8_t* codes, const float* scales, int n, int k, const bf16* x, int tokens, bf16* out,
               int ldc) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int r    = blockIdx.x * kW8Rows + warp;
    if (r >= n) { return; }
    const std::uint8_t* row = codes + static_cast<std::int64_t>(r) * k;
    const int chunks        = k / 16;
    for (int t0 = 0; t0 < tokens; t0 += kW8Cols) {
        const int nt = min(kW8Cols, tokens - t0);
        float acc[kW8Cols];
#pragma unroll
        for (int t = 0; t < kW8Cols; ++t) { acc[t] = 0.0F; }
        for (int c = lane; c < chunks; c += 32) {
            const uint4 packed = reinterpret_cast<const uint4*>(row)[c];
            const __nv_fp8_e4m3* q8 = reinterpret_cast<const __nv_fp8_e4m3*>(&packed);
            float wv[16];
#pragma unroll
            for (int e = 0; e < 16; ++e) { wv[e] = static_cast<float>(q8[e]); }
#pragma unroll
            for (int t = 0; t < kW8Cols; ++t) {
                if (t < nt) {
                    const uint4* xp = reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(t0 + t) * k + c * 16);
                    const uint4 x0 = xp[0], x1 = xp[1];
                    const bf16* a = reinterpret_cast<const bf16*>(&x0);
                    const bf16* b = reinterpret_cast<const bf16*>(&x1);
                    float s = acc[t];
#pragma unroll
                    for (int e = 0; e < 8; ++e) { s = fmaf(wv[e], f(a[e]), s); }
#pragma unroll
                    for (int e = 0; e < 8; ++e) { s = fmaf(wv[8 + e], f(b[e]), s); }
                    acc[t] = s;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < kW8Cols; ++t) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) { acc[t] = __fadd_rn(acc[t], __shfl_xor_sync(0xffffffffU, acc[t], off)); }
        }
        if (lane < nt) {
            float v = acc[0];
#pragma unroll
            for (int t = 1; t < kW8Cols; ++t) {
                if (lane == t) { v = acc[t]; }
            }
            out[static_cast<std::int64_t>(t0 + lane) * ldc + r] = tob(__fmul_rn(v, scales[r]));
        }
    }
}

void require(bool ok, const char* op, const char* what) {
    if (!ok) { throw std::invalid_argument(std::string(op) + ": " + what); }
}

} // namespace

void dspark_rope_inv_freq(float theta, float* host_out64) {
    for (int i = 0; i < G::kHeadDim / 2; ++i) {
        const float exponent = static_cast<float>(2 * i) / static_cast<float>(G::kHeadDim);
        host_out64[i]        = 1.0F / std::pow(theta, exponent);
    }
}

void dspark_feature_mix(const void* features, std::int64_t feature_stride, std::int32_t columns,
                        const float* alpha, const void* norms, std::int32_t rows, float eps, void* out,
                        cudaStream_t stream) {
    constexpr const char* op = "dspark_feature_mix";
    require(features != nullptr && alpha != nullptr && norms != nullptr && out != nullptr, op, "null pointer");
    require(columns > 0 && rows >= 1 && rows <= G::kMixRows && feature_stride >= G::kFeatureRows, op,
            "bad shape");
    feature_mix_kernel<<<static_cast<unsigned>(columns), kThreads, 0, stream>>>(
        static_cast<const bf16*>(features), feature_stride, alpha, static_cast<const bf16*>(norms), rows, eps,
        static_cast<bf16*>(out), columns);
    CUDA_CHECK(cudaGetLastError());
}

void dspark_head_norm_rope(const void* x, std::int64_t x_stride, std::int32_t heads, std::int32_t columns,
                           const void* weight, const std::int32_t* positions, const float* inv_freq, float eps,
                           void* out, std::int64_t out_stride, cudaStream_t stream) {
    constexpr const char* op = "dspark_head_norm_rope";
    require(x != nullptr && weight != nullptr && out != nullptr, op, "null pointer");
    require(positions == nullptr || inv_freq != nullptr, op, "RoPE needs inv_freq");
    require(heads > 0 && columns > 0 && x_stride >= heads * G::kHeadDim && out_stride >= heads * G::kHeadDim, op,
            "bad shape");
    head_norm_rope_kernel<<<dim3(static_cast<unsigned>(columns), static_cast<unsigned>(heads)), 128, 0, stream>>>(
        static_cast<const bf16*>(x), x_stride, static_cast<const bf16*>(weight), positions, inv_freq, eps,
        static_cast<bf16*>(out), out_stride);
    CUDA_CHECK(cudaGetLastError());
}

void dspark_add_rms_norm(const void* x, const void* resid, const void* weight, float eps, std::int32_t columns,
                         void* resid_out, void* h, cudaStream_t stream) {
    constexpr const char* op = "dspark_add_rms_norm";
    require(resid != nullptr && weight != nullptr && h != nullptr && columns > 0, op, "bad arguments");
    add_rms_norm_kernel<<<static_cast<unsigned>(columns), kThreads, 0, stream>>>(
        static_cast<const bf16*>(x), static_cast<const bf16*>(resid), static_cast<const bf16*>(weight), eps,
        static_cast<bf16*>(resid_out), static_cast<bf16*>(h));
    CUDA_CHECK(cudaGetLastError());
}

void dspark_silu_mul(const void* gate_up, std::int32_t columns, void* y, cudaStream_t stream) {
    require(gate_up != nullptr && y != nullptr && columns > 0, "dspark_silu_mul", "bad arguments");
    const std::int64_t total = static_cast<std::int64_t>(G::kIntermediate) * columns;
    silu_mul_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
        static_cast<const bf16*>(gate_up), columns, static_cast<bf16*>(y));
    CUDA_CHECK(cudaGetLastError());
}

void dspark_embed_block(const std::int32_t* anchors, std::int32_t batch, const void* embed,
                        std::int32_t embed_rows, const void* embed_norm, const void* mask_embed, float eps,
                        void* out, cudaStream_t stream) {
    require(anchors != nullptr && embed != nullptr && embed_rows > 0 && embed_norm != nullptr &&
                mask_embed != nullptr && out != nullptr && batch > 0,
            "dspark_embed_block", "bad arguments");
    embed_block_kernel<<<static_cast<unsigned>(batch), kThreads, 0, stream>>>(
        anchors, static_cast<const bf16*>(embed), static_cast<const bf16*>(embed_norm),
        static_cast<const bf16*>(mask_embed), eps, static_cast<bf16*>(out));
    CUDA_CHECK(cudaGetLastError());
}

void dspark_block_positions(const std::int32_t* anchor_positions, std::int32_t batch, std::int32_t* positions,
                            cudaStream_t stream) {
    require(anchor_positions != nullptr && positions != nullptr && batch > 0, "dspark_block_positions",
            "bad arguments");
    block_positions_kernel<<<static_cast<unsigned>((batch * G::kBlock + 127) / 128), 128, 0, stream>>>(
        anchor_positions, batch, positions);
    CUDA_CHECK(cudaGetLastError());
}

void dspark_block_query_positions(const std::int32_t* anchor_positions, std::int32_t batch, std::int32_t* positions,
                                  cudaStream_t stream) {
    require(anchor_positions != nullptr && positions != nullptr && batch > 0, "dspark_block_query_positions",
            "bad arguments");
    block_query_positions_kernel<<<static_cast<unsigned>((batch * G::kBlock + 127) / 128), 128, 0, stream>>>(
        anchor_positions, batch, positions);
    CUDA_CHECK(cudaGetLastError());
}

void dspark_kv_write(const void* k, const void* v, std::int64_t stride, const std::int32_t* positions,
                     std::int32_t width, std::int32_t batch, const std::int32_t* counts,
                     const std::int32_t* table_rows, const DSparkPagedPlane& plane, cudaStream_t stream) {
    constexpr const char* op = "dspark_kv_write";
    require(k != nullptr && v != nullptr && positions != nullptr && table_rows != nullptr, op, "null pointer");
    require(width > 0 && batch > 0 && stride >= G::kKvRows, op, "bad shape");
    kv_write_kernel<<<static_cast<unsigned>(width * batch), kThreads, 0, stream>>>(
        static_cast<const bf16*>(k), static_cast<const bf16*>(v), stride, positions, width, counts, table_rows,
        plane_dev(plane, op));
    CUDA_CHECK(cudaGetLastError());
}

void dspark_block_attention(const void* q, std::int64_t q_stride, const std::int32_t* anchor_positions,
                            const std::int32_t* table_rows, std::int32_t batch, std::int32_t window, float scale,
                            const DSparkPagedPlane& plane, void* out, cudaStream_t stream) {
    constexpr const char* op = "dspark_block_attention";
    require(q != nullptr && anchor_positions != nullptr && table_rows != nullptr && out != nullptr, op,
            "null pointer");
    require(batch > 0 && q_stride >= G::kQRows && window >= 0, op, "bad shape");
    block_attention_kernel<<<dim3(G::kKvHeads, static_cast<unsigned>(batch)), 128, 0, stream>>>(
        static_cast<const bf16*>(q), q_stride, anchor_positions, table_rows, window, scale, plane_dev(plane, op),
        static_cast<bf16*>(out));
    CUDA_CHECK(cudaGetLastError());
}

void dspark_store_last_column(const void* src, std::int32_t width, std::int32_t batch,
                              const std::int32_t* counts, const std::int32_t* lanes, void* dst,
                              cudaStream_t stream) {
    require(src != nullptr && dst != nullptr && width > 0 && batch > 0, "dspark_store_last_column",
            "bad arguments");
    store_last_column_kernel<<<static_cast<unsigned>(batch), kThreads, 0, stream>>>(
        static_cast<const bf16*>(src), width, counts, lanes, static_cast<bf16*>(dst));
    CUDA_CHECK(cudaGetLastError());
}

void dspark_markov_pack(const void* hidden, const void* f_rnn, const std::int32_t* lanes, std::int32_t k,
                        std::int32_t batch, void* x, cudaStream_t stream) {
    require(hidden != nullptr && f_rnn != nullptr && x != nullptr && k >= 1 && k <= G::kMaxDrafts && batch > 0,
            "dspark_markov_pack", "bad arguments");
    markov_pack_kernel<<<static_cast<unsigned>(k * batch), kThreads, 0, stream>>>(
        static_cast<const bf16*>(hidden), static_cast<const bf16*>(f_rnn), lanes, k, static_cast<bf16*>(x));
    CUDA_CHECK(cudaGetLastError());
}

void dspark_markov_gate(const float* ctx, std::int32_t j, std::int32_t k, std::int32_t batch, const void* bias,
                        const void* w_sp, const void* w1, const std::int32_t* prev_base, std::int32_t prev_stride,
                        const void* state_in, void* state_out, void* latent, cudaStream_t stream) {
    require(ctx != nullptr && bias != nullptr && w_sp != nullptr && w1 != nullptr && prev_base != nullptr &&
                state_out != nullptr && latent != nullptr && state_in != state_out && batch > 0 && k >= 1 && j >= 0 &&
                j < k,
            "dspark_markov_gate", "bad arguments (state_in must not alias state_out)");
    markov_gate_kernel<<<dim3(G::kRank / kGateIndices, static_cast<unsigned>(batch)), 32 * kGateIndices, 0, stream>>>(
        ctx, j, k, static_cast<const bf16*>(bias), static_cast<const bf16*>(w_sp), static_cast<const bf16*>(w1),
        prev_base, prev_stride, static_cast<const bf16*>(state_in), static_cast<bf16*>(state_out),
        static_cast<bf16*>(latent));
    CUDA_CHECK(cudaGetLastError());
}

void dspark_markov_vocab(const void* base, std::int32_t base_rows, std::int32_t row, std::int32_t batch,
                         const void* fold_w, const void* fold_b, const void* latent, std::int32_t domain,
                         const std::int32_t* ids, float* part_val, std::int32_t* part_id, void* bias_out,
                         void* logits_out, cudaStream_t stream, const void* fold_codes, const float* fold_scales,
                         const SamplingConfig* sampling, const std::int32_t* anchor_positions,
                         std::int32_t draft_index) {
    constexpr const char* op = "dspark_markov_vocab";
    require((fold_codes == nullptr) == (fold_scales == nullptr), op, "FP8 fold needs codes and scales");
    require(base != nullptr && (fold_w != nullptr || fold_codes != nullptr) && fold_b != nullptr && latent != nullptr && part_val != nullptr &&
                part_id != nullptr,
            op, "null pointer");
    require(batch > 0 && batch <= kMaxLanes && domain > 0 && base_rows >= domain && row >= 1 && row < G::kBlock, op,
            "bad shape (batch <= 16)");
    require(sampling == nullptr || anchor_positions != nullptr, op, "sampled drafts need the anchor positions");
    markov_vocab_kernel<<<static_cast<unsigned>(dspark_vocab_tiles(domain)), kThreads, 0, stream>>>(
        static_cast<const bf16*>(base), base_rows, row, batch, static_cast<const bf16*>(fold_w),
        static_cast<const std::uint8_t*>(fold_codes), fold_scales, static_cast<const bf16*>(fold_b),
        static_cast<const bf16*>(latent), domain, ids, part_val, part_id,
        static_cast<bf16*>(bias_out), static_cast<bf16*>(logits_out), sampling, anchor_positions, draft_index);
    CUDA_CHECK(cudaGetLastError());
}

void dspark_markov_pick(const float* part_val, const std::int32_t* part_id, std::int32_t tiles, std::int32_t batch,
                        std::int32_t j, std::int32_t stride, std::int32_t* drafts, cudaStream_t stream) {
    require(part_val != nullptr && part_id != nullptr && drafts != nullptr && tiles > 0 && batch > 0 && j >= 0 &&
                j < stride,
            "dspark_markov_pick", "bad arguments");
    markov_pick_kernel<<<static_cast<unsigned>(batch), kThreads, 0, stream>>>(part_val, part_id, tiles, batch, j,
                                                                              stride, drafts);
    CUDA_CHECK(cudaGetLastError());
}

void dspark_fold(const void* unembed, std::int32_t rows, const std::int32_t* ids, const void* hidden_out,
                 const void* hidden_out_bias, void* fold_w, void* fold_b, cudaStream_t stream) {
    require(unembed != nullptr && hidden_out != nullptr && hidden_out_bias != nullptr && fold_w != nullptr &&
                fold_b != nullptr && rows > 0,
            "dspark_fold", "bad arguments");
    fold_kernel<<<static_cast<unsigned>((rows + kFoldRows - 1) / kFoldRows), G::kRank, 0, stream>>>(
        static_cast<const bf16*>(unembed), rows, ids, static_cast<const bf16*>(hidden_out),
        static_cast<const bf16*>(hidden_out_bias), static_cast<bf16*>(fold_w), static_cast<bf16*>(fold_b));
    CUDA_CHECK(cudaGetLastError());
}

void dspark_quantize_rows_e4m3(const void* w, std::int32_t n, std::int32_t k, void* codes, float* scales,
                               cudaStream_t stream) {
    require(w != nullptr && codes != nullptr && scales != nullptr && n > 0 && k > 0, "dspark_quantize_rows_e4m3",
            "bad arguments");
    quantize_rows_kernel<<<static_cast<unsigned>(n), kThreads, 0, stream>>>(
        static_cast<const bf16*>(w), k, static_cast<std::uint8_t*>(codes), scales);
    CUDA_CHECK(cudaGetLastError());
}

void dspark_gemm_w8(const void* codes, const float* scales, std::int32_t n, std::int32_t k, const void* x,
                    std::int32_t tokens, void* out, std::int32_t ldc, cudaStream_t stream) {
    constexpr const char* op = "dspark_gemm_w8";
    require(codes != nullptr && scales != nullptr && x != nullptr && out != nullptr, op, "null pointer");
    require(n > 0 && k > 0 && k % 16 == 0 && tokens > 0 && ldc >= n, op, "bad shape (k % 16, ldc >= n)");
    require(reinterpret_cast<std::uintptr_t>(codes) % 16 == 0 && reinterpret_cast<std::uintptr_t>(x) % 16 == 0, op,
            "codes and x must be 16-byte aligned");
    gemm_w8_kernel<<<static_cast<unsigned>((n + kW8Rows - 1) / kW8Rows), 32 * kW8Rows, 0, stream>>>(
        static_cast<const std::uint8_t*>(codes), scales, n, k, static_cast<const bf16*>(x), tokens,
        static_cast<bf16*>(out), ldc);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops
