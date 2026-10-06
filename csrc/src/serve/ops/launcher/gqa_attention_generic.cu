// sinfer::ops - gqa_attention for head layouts the registry has no tuned kernel for.
//
// A shape is served when its (head dim, KV heads) pair is registered, whatever its query group,
// but only fully registered shapes reach the tuned decode and prompt kernels. Every other shape
// (Qwen3-8B's 32 query heads over 8, Llama 3 8B's, ...) used to take one warp per query head
// and column walking the whole history one key at a time, and each batch row as its own launch:
// on an H100 a Qwen3-8B decode step spent ~3.5 ms per layer there, about 8 tokens/s.
//
// This route splits each query's history across CTAs (flash-decoding) when the columns alone
// do not fill the device, has four warps per CTA working on different keys, gives each key a
// group of lanes so several keys are in flight per warp, and serves every batch row in one
// launch. The append is one launch as well. Arithmetic is FP32 with an online softmax in the
// base-2 domain; the partials merge in a fixed order, and under --batch-invariant a query is
// never split, so its result depends on its own history alone.
#include "ops/launcher/gqa_attention.h"

#include "api/ops/batch_invariant.h"
#include "api/ops/gqa_attention.h"
#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace sinfer::ops::detail {
namespace {

constexpr int kGenericWarps   = 4;
constexpr int kGenericThreads = kGenericWarps * 32;
constexpr int kGenericMaxSplits = 64;

struct GenericAttentionParams {
    const __nv_bfloat16* q;
    const void* k_pages;
    const void* v_pages;
    const __half* k_scales;
    const __half* v_scales;
    const std::int32_t* positions;     // [width, batch]: each column's absolute position
    const std::int32_t* tables;        // block tables
    const std::int32_t* table_rows;    // [batch] row of each sequence, or null: row 0
    std::int64_t table_stride;
    const std::int32_t* valid_columns; // [batch] or null: every column valid
    int q_heads;
    int kv_heads;
    int width;
    int window;
    float scale_log2;                  // softmax scale times log2(e)
    int splits;
    __nv_bfloat16* out;                // [dim, q_heads, width, batch]
    float* partial_acc;                // [columns, q_heads, splits, dim] when splits > 1
    float* partial_ml;                 // [columns, q_heads, splits, 2]
};

__device__ __forceinline__ const std::int32_t* generic_block_table(const GenericAttentionParams& p,
                                                                   int row) {
    const std::int64_t table_row = p.table_rows != nullptr ? p.table_rows[row] : 0;
    return p.tables + table_row * p.table_stride;
}

__device__ __forceinline__ bool generic_column_valid(const std::int32_t* valid_columns, int row,
                                                     int column) {
    return valid_columns == nullptr || column < valid_columns[row];
}

// Element offset of (key, kv_head, d) in the paged cache, the runtime-extent form of
// paged_kv_element_offset: pages of kPagedKVPageSize keys, heads inside a page, keys inside
// a head, the head dimension innermost.
__device__ __forceinline__ std::int64_t generic_cell(const std::int32_t* table, int kv_heads,
                                                     int kv_head, int key) {
    const std::int64_t page = paged_kv_physical_page(table, key);
    return (page * kv_heads + kv_head) * kPagedKVPageSize + (key & kPagedKVPageMask);
}

// E cache values starting at element `d` of the row at `cell`, widened to FP32.
template <typename CacheT, int E, int Dim>
__device__ __forceinline__ void generic_load(const CacheT* data, const __half* scales,
                                             std::int64_t cell, int d, float (&out)[E]) {
    const CacheT* row = data + cell * Dim + d;
    if constexpr (std::is_same_v<CacheT, __nv_bfloat16>) {
#pragma unroll
        for (int chunk = 0; chunk < E / 8; ++chunk) {
            const int4 raw = *reinterpret_cast<const int4*>(row + chunk * 8);
            const auto* values = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 pair = __bfloat1622float2(values[i]);
                out[chunk * 8 + 2 * i]     = pair.x;
                out[chunk * 8 + 2 * i + 1] = pair.y;
            }
        }
    } else if constexpr (std::is_same_v<CacheT, std::uint8_t>) {
#pragma unroll
        for (int chunk = 0; chunk < E / 8; ++chunk) {
            const int4 widened =
                gqa_kv_dequant_fp8x8_raw(*reinterpret_cast<const int2*>(row + chunk * 8));
            const auto* values = reinterpret_cast<const __nv_bfloat162*>(&widened);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 pair = __bfloat1622float2(values[i]);
                out[chunk * 8 + 2 * i]     = pair.x;
                out[chunk * 8 + 2 * i + 1] = pair.y;
            }
        }
    } else {
        // int8 codes with one FP16 scale per kGqaKvQuantGroup (64) values; E divides 64, so a
        // lane's slice never straddles two groups.
        const float group_scale =
            __half2float(scales[cell * (Dim / kGqaKvQuantGroup) + d / kGqaKvQuantGroup]);
#pragma unroll
        for (int chunk = 0; chunk < E / 8; ++chunk) {
            const int2 raw = *reinterpret_cast<const int2*>(row + chunk * 8);
            const auto* codes = reinterpret_cast<const std::int8_t*>(&raw);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                out[chunk * 8 + i] = static_cast<float>(codes[i]) * group_scale;
            }
        }
    }
}

// Merge (m_b, l_b, acc_b) into (m, l, acc). Running maxima are in the base-2 domain; an empty
// side carries m = -inf and l = 0 and contributes nothing.
template <int E>
__device__ __forceinline__ void generic_merge(float& m, float& l, float (&acc)[E], float m_b,
                                              float l_b, const float (&acc_b)[E]) {
    const float top = fmaxf(m, m_b);
    const float a   = m == -CUDART_INF_F ? 0.0f : exp2f(m - top);
    const float b   = m_b == -CUDART_INF_F ? 0.0f : exp2f(m_b - top);
    l = l * a + l_b * b;
#pragma unroll
    for (int i = 0; i < E; ++i) { acc[i] = acc[i] * a + acc_b[i] * b; }
    m = top;
}

// One CTA per (column, query head, split). Each key is read by a group of L = Dim / E lanes,
// E values per lane, so a warp has 32 / L keys in flight and the four warps take interleaved
// groups of keys; the groups and the warps merge in a fixed order at the end.
template <typename CacheT, int Dim>
__global__ __launch_bounds__(kGenericThreads) void gqa_generic_split_kernel(
    GenericAttentionParams p) {
    constexpr int E            = Dim >= 512 ? 16 : 8;
    constexpr int L            = Dim / E;
    constexpr int KeysPerWarp  = 32 / L;
    constexpr int KeysPerRound = KeysPerWarp * kGenericWarps;
    static_assert(L <= 32 && 32 % L == 0, "a key's lanes must tile a warp");

    const int column = blockIdx.x; // row * width + position within the row
    const int q_head = blockIdx.y;
    const int split  = blockIdx.z;
    const int row    = column / p.width;
    const int tid    = threadIdx.x;
    const int warp   = tid / 32;
    const int lane   = tid % 32;
    const int slot   = lane / L;    // which of the warp's keys this lane works on
    const int d      = (lane % L) * E;

    const std::int64_t head_row = static_cast<std::int64_t>(column) * p.q_heads + q_head;
    const bool valid = generic_column_valid(p.valid_columns, row, column - row * p.width);
    const int last   = p.positions[column];
    const int first  = p.window > 0 ? max(0, last + 1 - p.window) : 0;
    const int keys   = valid ? last - first + 1 : 0;
    const int span   = (keys + p.splits - 1) / p.splits;
    const int begin  = first + split * span;
    const int end    = min(last + 1, begin + span);

    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc[E];
#pragma unroll
    for (int i = 0; i < E; ++i) { acc[i] = 0.0f; }

    if (keys > 0 && begin < end) {
        float qv[E];
        {
            const __nv_bfloat16* qrow = p.q + head_row * Dim + d;
#pragma unroll
            for (int chunk = 0; chunk < E / 8; ++chunk) {
                const int4 raw = *reinterpret_cast<const int4*>(qrow + chunk * 8);
                const auto* values = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 pair = __bfloat1622float2(values[i]);
                    qv[chunk * 8 + 2 * i]     = pair.x * p.scale_log2;
                    qv[chunk * 8 + 2 * i + 1] = pair.y * p.scale_log2;
                }
            }
        }
        const int kv_head = q_head / (p.q_heads / p.kv_heads);
        const std::int32_t* table = generic_block_table(p, row);
        const auto* keys_data   = static_cast<const CacheT*>(p.k_pages);
        const auto* values_data = static_cast<const CacheT*>(p.v_pages);
        for (int base = begin; base < end; base += KeysPerRound) {
            const int key      = base + warp * KeysPerWarp + slot;
            const bool present = key < end;
            float kv[E];
            float score = 0.0f;
            std::int64_t cell = 0;
            if (present) {
                cell = generic_cell(table, p.kv_heads, kv_head, key);
                generic_load<CacheT, E, Dim>(keys_data, p.k_scales, cell, d, kv);
#pragma unroll
                for (int i = 0; i < E; ++i) { score = fmaf(qv[i], kv[i], score); }
            }
#pragma unroll
            for (int offset = L / 2; offset > 0; offset /= 2) {
                score += __shfl_xor_sync(0xffffffffU, score, offset);
            }
            if (present) {
                generic_load<CacheT, E, Dim>(values_data, p.v_scales, cell, d, kv);
                const float top        = fmaxf(m, score);
                const float correction = exp2f(m - top);
                const float weight     = exp2f(score - top);
                l = l * correction + weight;
#pragma unroll
                for (int i = 0; i < E; ++i) { acc[i] = fmaf(acc[i], correction, weight * kv[i]); }
                m = top;
            }
        }
    }

    // The warp's key groups, lowest slot first.
#pragma unroll
    for (int offset = L; offset < 32; offset *= 2) {
        const float m_b = __shfl_xor_sync(0xffffffffU, m, offset);
        const float l_b = __shfl_xor_sync(0xffffffffU, l, offset);
        float acc_b[E];
#pragma unroll
        for (int i = 0; i < E; ++i) { acc_b[i] = __shfl_xor_sync(0xffffffffU, acc[i], offset); }
        if ((lane & offset) == 0) {
            generic_merge(m, l, acc, m_b, l_b, acc_b);
        } else {
            float m_a = m_b;
            float l_a = l_b;
            generic_merge(m_a, l_a, acc_b, m, l, acc);
            m = m_a;
            l = l_a;
#pragma unroll
            for (int i = 0; i < E; ++i) { acc[i] = acc_b[i]; }
        }
    }

    // Then the four warps, in order.
    __shared__ float shared_acc[kGenericWarps][Dim];
    __shared__ float shared_ml[kGenericWarps][2];
    if (lane < L) {
#pragma unroll
        for (int i = 0; i < E; ++i) { shared_acc[warp][d + i] = acc[i]; }
        if (lane == 0) {
            shared_ml[warp][0] = m;
            shared_ml[warp][1] = l;
        }
    }
    __syncthreads();
    if (warp != 0 || lane >= L) { return; }
    for (int other = 1; other < kGenericWarps; ++other) {
        float acc_b[E];
#pragma unroll
        for (int i = 0; i < E; ++i) { acc_b[i] = shared_acc[other][d + i]; }
        generic_merge(m, l, acc, shared_ml[other][0], shared_ml[other][1], acc_b);
    }

    if (p.splits == 1) {
        const float inverse = l > 0.0f ? 1.0f / l : 0.0f;
        __nv_bfloat16* out = p.out + head_row * Dim + d;
#pragma unroll
        for (int i = 0; i < E; ++i) { out[i] = __float2bfloat16(acc[i] * inverse); }
        return;
    }
    const std::int64_t partial = head_row * p.splits + split;
    float* partial_acc = p.partial_acc + partial * Dim + d;
#pragma unroll
    for (int i = 0; i < E; ++i) { partial_acc[i] = acc[i]; }
    if (lane == 0) {
        p.partial_ml[partial * 2]     = m;
        p.partial_ml[partial * 2 + 1] = l;
    }
}

// One CTA per (column, query head): the splits merged in split order.
template <int Dim>
__global__ __launch_bounds__(128) void gqa_generic_reduce_kernel(GenericAttentionParams p) {
    const int column = blockIdx.x;
    const int q_head = blockIdx.y;
    const std::int64_t head_row = static_cast<std::int64_t>(column) * p.q_heads + q_head;
    const float* ml = p.partial_ml + head_row * p.splits * 2;
    float top = -CUDART_INF_F;
    for (int split = 0; split < p.splits; ++split) {
        if (ml[split * 2 + 1] > 0.0f) { top = fmaxf(top, ml[split * 2]); }
    }
    float total = 0.0f;
    for (int split = 0; split < p.splits; ++split) {
        const float l = ml[split * 2 + 1];
        if (l > 0.0f) { total += l * exp2f(ml[split * 2] - top); }
    }
    const float inverse = total > 0.0f ? 1.0f / total : 0.0f;
    const float* acc    = p.partial_acc + head_row * p.splits * Dim;
    for (int d = threadIdx.x; d < Dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int split = 0; split < p.splits; ++split) {
            const float l = ml[split * 2 + 1];
            if (l > 0.0f) { sum += acc[static_cast<std::int64_t>(split) * Dim + d] * exp2f(ml[split * 2] - top); }
        }
        p.out[head_row * Dim + d] = __float2bfloat16(sum * inverse);
    }
}

// One thread per 8 values of one (column, KV head) row: K and V into the paged cache, every
// batch row in the same launch.
template <typename CacheT>
__global__ void gqa_generic_append_kernel(const __nv_bfloat16* __restrict__ k,
                                          const __nv_bfloat16* __restrict__ v,
                                          GenericAttentionParams p, int dim, int columns,
                                          CacheT* __restrict__ cache_k,
                                          CacheT* __restrict__ cache_v) {
    const int vectors      = dim / 8;
    const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<std::int64_t>(columns) * p.kv_heads * vectors) { return; }
    const int d       = static_cast<int>(idx % vectors) * 8;
    const int kv_head = static_cast<int>((idx / vectors) % p.kv_heads);
    const int column  = static_cast<int>(idx / vectors / p.kv_heads);
    const int row     = column / p.width;
    if (!generic_column_valid(p.valid_columns, row, column - row * p.width)) { return; }
    const std::int64_t source = (static_cast<std::int64_t>(column) * p.kv_heads + kv_head) * dim + d;
    const std::int64_t cell =
        generic_cell(generic_block_table(p, row), p.kv_heads, kv_head, p.positions[column]);
    const std::int64_t target = cell * dim + d;
    if constexpr (std::is_same_v<CacheT, std::uint8_t>) {
        gqa_kv_store_fp8x8(&cache_k[target], &k[source]);
        gqa_kv_store_fp8x8(&cache_v[target], &v[source]);
    } else {
        *reinterpret_cast<int4*>(&cache_k[target]) = *reinterpret_cast<const int4*>(&k[source]);
        *reinterpret_cast<int4*>(&cache_v[target]) = *reinterpret_cast<const int4*>(&v[source]);
    }
}

int device_multiprocessors() {
    static std::atomic<int> cached[64] = {};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 64) { return 108; }
    int known = cached[device].load(std::memory_order_relaxed);
    if (known == 0) {
        if (cudaDeviceGetAttribute(&known, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
            known <= 0) {
            known = 108;
        }
        cached[device].store(known, std::memory_order_relaxed);
    }
    return known;
}

template <typename Visitor>
void dispatch_generic_dim(int dim, Visitor&& visitor) {
    switch (dim) {
    case 64: return visitor.template operator()<64>();
    case 128: return visitor.template operator()<128>();
    case 256: return visitor.template operator()<256>();
    case 512: return visitor.template operator()<512>();
    default:
        throw std::invalid_argument("gqa_attention: no generic kernel for head dim " +
                                    std::to_string(dim));
    }
}

GenericAttentionParams generic_params(const Tensor& q, const Tensor& positions,
                                      const Tensor& valid_columns, const Tensor& table_rows,
                                      const Tensor& tables, std::int64_t table_stride,
                                      const Tensor& k_pages, const Tensor& v_pages,
                                      const Tensor& k_scales, const Tensor& v_scales,
                                      int kv_heads, int window, float scale, Tensor& out) {
    return GenericAttentionParams{
        .q             = static_cast<const __nv_bfloat16*>(q.data),
        .k_pages       = k_pages.data,
        .v_pages       = v_pages.data,
        .k_scales      = static_cast<const __half*>(k_scales.data),
        .v_scales      = static_cast<const __half*>(v_scales.data),
        .positions     = static_cast<const std::int32_t*>(positions.data),
        .tables        = static_cast<const std::int32_t*>(tables.data),
        .table_rows    = static_cast<const std::int32_t*>(table_rows.data),
        .table_stride  = table_stride,
        .valid_columns = static_cast<const std::int32_t*>(valid_columns.data),
        .q_heads       = q.ne[1],
        .kv_heads      = kv_heads,
        .width         = q.ne[2],
        .window        = window,
        .scale_log2    = scale * 1.4426950408889634f,
        .splits        = 1,
        .out           = static_cast<__nv_bfloat16*>(out.data),
        .partial_acc   = nullptr,
        .partial_ml    = nullptr,
    };
}

void launch_generic(GenericAttentionParams p, int dim, int columns, int splits, DType cache_dtype,
                    DeviceSpan workspace, cudaStream_t stream) {
    if (splits < 1 || splits > kGenericMaxSplits) {
        throw std::invalid_argument("gqa_attention: generic split count out of range");
    }
    p.splits = splits;
    if (p.splits > 1) {
        const std::size_t need = gqa_generic_attention_workspace_bytes(dim, p.q_heads, columns, p.splits);
        if (workspace.data == nullptr || workspace.bytes < need) {
            throw std::invalid_argument("gqa_attention: generic split workspace is short");
        }
        const std::size_t acc_bytes = round_up<std::size_t>(
            static_cast<std::size_t>(columns) * p.q_heads * p.splits * dim * sizeof(float), 256);
        p.partial_acc = static_cast<float*>(workspace.data);
        p.partial_ml  = reinterpret_cast<float*>(static_cast<std::byte*>(workspace.data) + acc_bytes);
    }
    const dim3 grid(static_cast<unsigned>(columns), static_cast<unsigned>(p.q_heads),
                    static_cast<unsigned>(p.splits));
    dispatch_generic_dim(dim, [&]<int Dim>() {
        if (cache_dtype == DType::BF16) {
            gqa_generic_split_kernel<__nv_bfloat16, Dim><<<grid, kGenericThreads, 0, stream>>>(p);
        } else if (cache_dtype == DType::FP8_E4M3FN) {
            gqa_generic_split_kernel<std::uint8_t, Dim><<<grid, kGenericThreads, 0, stream>>>(p);
        } else if (cache_dtype == DType::I8) {
            if constexpr (Dim >= kGqaKvQuantGroup) {
                gqa_generic_split_kernel<std::int8_t, Dim><<<grid, kGenericThreads, 0, stream>>>(p);
            } else {
                throw std::invalid_argument("gqa_attention: int8 KV needs head dim >= 64");
            }
        } else {
            throw std::invalid_argument("gqa_attention: unsupported cache dtype for the generic kernel");
        }
        CUDA_CHECK(cudaGetLastError());
        if (p.splits > 1) {
            gqa_generic_reduce_kernel<Dim>
                <<<dim3(static_cast<unsigned>(columns), static_cast<unsigned>(p.q_heads)), 128, 0,
                   stream>>>(p);
            CUDA_CHECK(cudaGetLastError());
        }
    });
}

} // namespace

int gqa_generic_attention_splits(std::int32_t q_heads, std::int32_t columns,
                                 std::uint32_t max_visible_keys) {
    if (batch_invariant() || q_heads <= 0 || columns <= 0) { return 1; }
    // Two CTAs per SM is enough to cover the cache reads' latency; past that a split only adds
    // partial traffic and a reduce.
    const std::int64_t target = 2LL * device_multiprocessors();
    const std::int64_t ctas   = static_cast<std::int64_t>(q_heads) * columns;
    if (ctas >= target) { return 1; }
    std::int64_t splits = (target + ctas - 1) / ctas;
    splits = std::min<std::int64_t>(splits, kGenericMaxSplits);
    splits = std::min<std::int64_t>(splits, std::max<std::int64_t>(1, (max_visible_keys + 63) / 64));
    return static_cast<int>(std::max<std::int64_t>(splits, 1));
}

std::size_t gqa_generic_attention_workspace_bytes(std::int32_t head_dim, std::int32_t q_heads,
                                                  std::int32_t columns, std::int32_t splits) {
    if (splits <= 1) { return 0; }
    const std::size_t partials = static_cast<std::size_t>(columns) * q_heads * splits;
    return round_up<std::size_t>(partials * head_dim * sizeof(float), 256) +
           round_up<std::size_t>(partials * 2 * sizeof(float), 256);
}

bool gqa_generic_attention_serves(std::int32_t head_dim, DType cache_dtype) {
    return (head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512) &&
           (cache_dtype == DType::BF16 || cache_dtype == DType::FP8_E4M3FN ||
            (cache_dtype == DType::I8 && head_dim >= kGqaKvQuantGroup));
}

void gqa_generic_attention_launch(const Tensor& q, const Tensor& positions,
                                  const Tensor& valid_columns, const Tensor& table_rows,
                                  float scale, const PagedKVBatchLayerView& cache,
                                  std::int32_t sliding_window, std::int32_t splits,
                                  DeviceSpan workspace, Tensor& out, cudaStream_t stream) {
    const int columns = q.ne[2] * q.ne[3];
    GenericAttentionParams p = generic_params(
        q, positions, valid_columns, table_rows, cache.block_tables, cache.block_tables.ne[0],
        cache.k_pages, cache.v_pages, cache.k_scale_pages, cache.v_scale_pages,
        cache.num_kv_heads, sliding_window, scale, out);
    launch_generic(p, q.ne[0], columns, splits, cache.dtype, workspace, stream);
}

void gqa_generic_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                  const PagedKVLayerView& cache, std::int32_t sliding_window,
                                  std::int32_t splits, DeviceSpan workspace, Tensor& out,
                                  cudaStream_t stream) {
    GenericAttentionParams p = generic_params(
        q, positions, Tensor{}, Tensor{}, cache.block_table, 0, cache.k_pages, cache.v_pages,
        cache.k_scale_pages, cache.v_scale_pages, cache.num_kv_heads, sliding_window, scale, out);
    launch_generic(p, q.ne[0], q.ne[2], splits, cache.dtype, workspace, stream);
}

void gqa_generic_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                  const Tensor& valid_columns, const Tensor& table_rows,
                                  const PagedKVBatchLayerView& cache, cudaStream_t stream) {
    if (cache.dtype == DType::I8) {
        // The int8 codec's group scales live with the tuned fill kernels.
        gqa_kv_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
        return;
    }
    const int dim     = k.ne[0];
    const int columns = k.ne[2] * k.ne[3];
    GenericAttentionParams p{};
    p.positions     = static_cast<const std::int32_t*>(positions.data);
    p.tables        = static_cast<const std::int32_t*>(cache.block_tables.data);
    p.table_rows    = static_cast<const std::int32_t*>(table_rows.data);
    p.table_stride  = cache.block_tables.ne[0];
    p.valid_columns = static_cast<const std::int32_t*>(valid_columns.data);
    p.kv_heads      = cache.num_kv_heads;
    p.width         = k.ne[2];
    const std::int64_t threads = static_cast<std::int64_t>(columns) * p.kv_heads * (dim / 8);
    const auto blocks = static_cast<unsigned>(div_up(threads, static_cast<std::int64_t>(256)));
    const auto* kd = static_cast<const __nv_bfloat16*>(k.data);
    const auto* vd = static_cast<const __nv_bfloat16*>(v.data);
    if (cache.dtype == DType::FP8_E4M3FN) {
        gqa_generic_append_kernel<std::uint8_t><<<blocks, 256, 0, stream>>>(
            kd, vd, p, dim, columns, static_cast<std::uint8_t*>(cache.k_pages.data),
            static_cast<std::uint8_t*>(cache.v_pages.data));
    } else {
        gqa_generic_append_kernel<__nv_bfloat16><<<blocks, 256, 0, stream>>>(
            kd, vd, p, dim, columns, static_cast<__nv_bfloat16*>(cache.k_pages.data),
            static_cast<__nv_bfloat16*>(cache.v_pages.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail
