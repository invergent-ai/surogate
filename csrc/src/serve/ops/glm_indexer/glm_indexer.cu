#include "api/ops/glm_indexer.h"
#include "api/ops/qsa_indexer.h"
#include "ops/common/warp.cuh"
#include "core/device.h"

#include <cuda_bf16.h>
#include <math_constants.h>
#include <climits>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace {
constexpr int kThreads = 128;
constexpr int kWarps = 4;
constexpr std::size_t kScratchLimit = 64ULL << 20;
void require(bool ok, const char* what) {
    if (!ok) { throw std::invalid_argument(std::string("glm_indexer: ") + what); }
}
void geometry_check(const GlmIndexerGeometry& g) {
    require(g.head_dim > 0 && g.head_dim <= 256 && g.head_dim % 32 == 0 &&
            g.heads > 0 && g.heads <= 64 && g.block > 0 && g.block <= 64 &&
            g.top_k > 0 && g.top_k % g.block == 0 &&
            g.top_k <= INT32_MAX - g.block && std::isfinite(g.norm_epsilon) && g.norm_epsilon > 0,
            "invalid geometry");
}
void tensor_check(const Tensor& t, DType dtype, std::int64_t count, const char* label) {
    require(t.data && t.dtype == dtype && t.is_contiguous() && t.numel() == count, label);
}
void cache_check(const Tensor& positions, const Tensor& tables, const Tensor& valid, int width,
                 int tokens, const GlmIndexerGeometry& g, PagedKVBatchLayerView cache) {
    geometry_check(g);
    tensor_check(positions, DType::I32, tokens, "positions must be I32 [T]");
    require(width > 0 && tokens > 0 && tokens % width == 0, "columns must partition into sequences");
    tensor_check(tables, DType::I32, tokens / width, "table rows must cover every sequence");
    if (valid.data) { tensor_check(valid, DType::I32, tokens / width, "invalid valid-column shape"); }
    require(cache.block_tables.data && cache.block_tables.dtype == DType::I32 &&
            cache.block_tables.is_contiguous(), "invalid block table");
    require(cache.indexer_pages.data && cache.indexer_pages.dtype == DType::FP32 &&
            cache.indexer_pages.ne[0] == 3 * g.head_dim && cache.indexer_pages.is_contiguous(),
            "cache must carry three FP32 indexer components per cell");
}
__device__ long long offset(const int* table, int pos, int dim) {
    return (static_cast<long long>(table[pos / kPagedKVPageSize]) * kPagedKVPageSize +
            pos % kPagedKVPageSize) * (3 * dim);
}
__device__ bool valid_column(int token, int width, const int* valid) {
    return !valid || token % width < valid[token / width];
}
__global__ void project_kernel(const float* w, const __nv_bfloat16* x, float* out, int n, int k, int tokens) {
    const int row = blockIdx.x * kWarps + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (row >= n) { return; }
    for (int token = blockIdx.y; token < tokens; token += gridDim.y) {
        float value = 0;
        for (int c = lane; c < k; c += 32) {
            value = fmaf(w[static_cast<long long>(row) * k + c],
                         __bfloat162float(x[static_cast<long long>(token) * k + c]), value);
        }
        value = warp_reduce_sum(value);
        if (lane == 0) { out[static_cast<long long>(token) * n + row] = value; }
    }
}
__global__ void append_kernel(const float* keys, const float* gates, const float* gain,
    const float* bias, const int* pos, const int* rows, const int* valid, int width,
    const int* tables, int table_stride, float* cache, int tokens, int dim, float eps) {
    const int token = blockIdx.x * kWarps + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (token >= tokens || !valid_column(token, width, valid)) { return; }
    const int* table = tables + static_cast<long long>(rows[token / width]) * table_stride;
    const long long at = static_cast<long long>(token) * dim;
    float sum = 0;
    for (int d = lane; d < dim; d += 32) { sum += keys[at + d]; }
    const float mean = __shfl_sync(0xffffffff, warp_reduce_sum(sum), 0) / dim;
    float square = 0;
    for (int d = lane; d < dim; d += 32) {
        const float v = keys[at + d] - mean;
        square = fmaf(v, v, square);
    }
    const float inv = rsqrtf(__shfl_sync(0xffffffff, warp_reduce_sum(square), 0) / dim + eps);
    float* dst = cache + offset(table, pos[token], dim);
    for (int d = lane; d < dim; d += 32) {
        dst[d] = (keys[at + d] - mean) * inv * gain[d] + bias[d];
        dst[dim + d] = gates[at + d];
    }
}
__global__ void pool_kernel(const float* ape, const int* pos, const int* rows, const int* valid,
    int width, const int* tables, int stride, float* cache, int tokens, int dim, int pool) {
    const int token = blockIdx.x;
    if (token >= tokens || !valid_column(token, width, valid) || (pos[token] + 1) % pool) { return; }
    const int* table = tables + static_cast<long long>(rows[token / width]) * stride;
    const int first = pos[token] + 1 - pool;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float maximum = -CUDART_INF_F;
        for (int r = 0; r < pool; ++r) {
            maximum = fmaxf(maximum, cache[offset(table, first + r, dim) + dim + d] + ape[r * dim + d]);
        }
        float denominator = 0, numerator = 0;
        for (int r = 0; r < pool; ++r) {
            const float* src = cache + offset(table, first + r, dim);
            const float weight = expf(src[dim + d] + ape[r * dim + d] - maximum);
            denominator += weight;
            numerator = fmaf(weight, src[d], numerator);
        }
        cache[offset(table, first, dim) + 2 * dim + d] = numerator / denominator;
    }
}
__device__ unsigned ordered(float value) {
    // Signed head weights make pool scores negative too. Map IEEE floats to an
    // unsigned monotonic order; canonicalize signed zero for deterministic ties.
    const unsigned bits = __float_as_uint(value == 0 ? 0.F : value);
    return bits & 0x80000000U ? ~bits : bits ^ 0x80000000U;
}
__global__ void select_kernel(const float* q, const float* weights, const int* positions,
    const int* rows, const int* valid, int width, int first_row, const int* tables, int stride,
    const float* cache, unsigned* masks, float* scores, int score_stride, int words,
    int dim, int heads, int pool, int budget) {
    const int row = first_row + blockIdx.x;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    unsigned* mask = masks + static_cast<long long>(row) * words;
    for (int i = tid; i < words; i += kThreads) { mask[i] = 0; }
    if (!valid_column(row, width, valid)) { return; }
    __syncthreads();
    const int visible = positions[row] + 1;
    const int complete = visible / pool;
    const int* table = tables + static_cast<long long>(rows[row / width]) * stride;
    if (visible % pool && tid == 0) { atomicOr(mask + (complete >> 5), 1U << (complete & 31)); }
    if (complete <= budget) {
        for (int b = tid; b < complete; b += kThreads) { atomicOr(mask + (b >> 5), 1U << (b & 31)); }
        return;
    }
    float* local_scores = scores + static_cast<long long>(blockIdx.x) * score_stride;
    const float scale = rsqrtf(static_cast<float>(dim * heads));
    for (int b = warp; b < complete; b += kWarps) {
        const float* key = cache + offset(table, b * pool, dim) + 2 * dim;
        float total = 0;
        for (int h = 0; h < heads; ++h) {
            const float* query = q + (static_cast<long long>(row) * heads + h) * dim;
            float dot = 0;
            for (int d = lane; d < dim; d += 32) { dot = fmaf(query[d], key[d], dot); }
            dot = warp_reduce_sum(dot);
            if (lane == 0) { total = fmaf(fmaxf(dot, 0.F), weights[row * heads + h] * scale, total); }
        }
        if (lane == 0) { local_scores[b] = total; }
    }
    __syncthreads();
    __shared__ unsigned threshold;
    __shared__ int counts[kWarps];
    if (tid == 0) { threshold = 0; }
    __syncthreads();
    for (int bit = 31; bit >= 0; --bit) {
        const unsigned candidate = threshold | (1U << bit);
        int count = 0;
        for (int b = tid; b < complete; b += kThreads) { count += ordered(local_scores[b]) >= candidate; }
        const float sum = warp_reduce_sum(static_cast<float>(count));
        if (lane == 0) { counts[warp] = static_cast<int>(sum); }
        __syncthreads();
        if (tid == 0 && counts[0] + counts[1] + counts[2] + counts[3] >= budget) { threshold = candidate; }
        __syncthreads();
    }
    int count = 0;
    for (int b = tid; b < complete; b += kThreads) {
        if (ordered(local_scores[b]) > threshold) {
            atomicOr(mask + (b >> 5), 1U << (b & 31));
            ++count;
        }
    }
    const float sum = warp_reduce_sum(static_cast<float>(count));
    if (lane == 0) { counts[warp] = static_cast<int>(sum); }
    __syncthreads();
    if (tid == 0) {
        int remaining = budget - counts[0] - counts[1] - counts[2] - counts[3];
        for (int b = 0; b < complete && remaining; ++b) {
            if (ordered(local_scores[b]) == threshold) {
                mask[b >> 5] |= 1U << (b & 31);
                --remaining;
            }
        }
    }
}
int row_tile(int rows, int keys, int pool) {
    const std::size_t blocks = (static_cast<std::size_t>(keys) + pool - 1) / pool;
    return static_cast<int>(std::max<std::size_t>(1, std::min<std::size_t>(rows, kScratchLimit / (blocks * 4))));
}
} // namespace

void glm_indexer_project(const Tensor& w, const Tensor& x, Tensor& out, cudaStream_t stream) {
    const int k = w.ne[0], n = w.ne[1], tokens = x.ne[1];
    require(k > 0 && n > 0 && tokens > 0, "invalid projection geometry");
    tensor_check(w, DType::FP32, static_cast<std::int64_t>(k) * n, "weights must be FP32 [K,N]");
    tensor_check(x, DType::BF16, static_cast<std::int64_t>(k) * tokens, "input must be BF16 [K,T]");
    tensor_check(out, DType::FP32, static_cast<std::int64_t>(n) * tokens, "output must be FP32 [N,T]");
    project_kernel<<<dim3((n + kWarps - 1) / kWarps, std::min(tokens, 65535)), kThreads, 0, stream>>>(
        static_cast<const float*>(w.data), static_cast<const __nv_bfloat16*>(x.data),
        static_cast<float*>(out.data), n, k, tokens);
    CUDA_CHECK(cudaGetLastError());
}
void glm_indexer_append(const Tensor& keys, const Tensor& gates, const Tensor& norm,
    const Tensor& bias, const Tensor& ape, const Tensor& positions, const Tensor& tables,
    const Tensor& valid, int width, const GlmIndexerGeometry& g, PagedKVBatchLayerView cache,
    cudaStream_t stream) {
    const int tokens = keys.ne[1];
    cache_check(positions, tables, valid, width, tokens, g, cache);
    tensor_check(keys, DType::FP32, static_cast<std::int64_t>(g.head_dim) * tokens, "invalid keys");
    tensor_check(gates, DType::FP32, keys.numel(), "invalid gates");
    tensor_check(norm, DType::FP32, g.head_dim, "invalid norm");
    tensor_check(bias, DType::FP32, g.head_dim, "invalid bias");
    tensor_check(ape, DType::FP32, static_cast<std::int64_t>(g.head_dim) * g.block, "invalid pooling bias");
    append_kernel<<<(tokens + kWarps - 1) / kWarps, kThreads, 0, stream>>>(
        static_cast<const float*>(keys.data), static_cast<const float*>(gates.data),
        static_cast<const float*>(norm.data), static_cast<const float*>(bias.data),
        static_cast<const int*>(positions.data), static_cast<const int*>(tables.data),
        static_cast<const int*>(valid.data), width, static_cast<const int*>(cache.block_tables.data),
        cache.block_tables.ne[0], static_cast<float*>(cache.indexer_pages.data), tokens, g.head_dim, g.norm_epsilon);
    pool_kernel<<<tokens, kThreads, 0, stream>>>(static_cast<const float*>(ape.data),
        static_cast<const int*>(positions.data), static_cast<const int*>(tables.data),
        static_cast<const int*>(valid.data), width, static_cast<const int*>(cache.block_tables.data),
        cache.block_tables.ne[0], static_cast<float*>(cache.indexer_pages.data), tokens, g.head_dim, g.block);
    CUDA_CHECK(cudaGetLastError());
}
std::size_t glm_indexer_select_workspace_capacity_bytes(int rows, int keys, const GlmIndexerGeometry& g) {
    geometry_check(g);
    require(rows > 0 && keys > 0 && keys <= INT32_MAX - g.block, "invalid selection extents");
    return static_cast<std::size_t>(row_tile(rows, keys, g.block)) * ((keys + g.block - 1) / g.block) * 4 + 256;
}
void glm_indexer_select(const Tensor& q, const Tensor& head_weights, const Tensor& positions,
    const Tensor& tables, const Tensor& valid, int width, const GlmIndexerGeometry& g,
    PagedKVBatchLayerView cache, int keys, WorkspaceArena& workspace, Tensor& mask, cudaStream_t stream) {
    const int tokens = q.ne[2];
    cache_check(positions, tables, valid, width, tokens, g, cache);
    require(keys > 0 && keys <= INT32_MAX - g.block, "invalid history extent");
    tensor_check(q, DType::FP32, static_cast<std::int64_t>(g.head_dim) * g.heads * tokens, "invalid query shape");
    tensor_check(head_weights, DType::FP32, static_cast<std::int64_t>(g.heads) * tokens, "invalid head weights");
    const int words = qsa_block_mask_words(keys, g.block);
    tensor_check(mask, DType::I32, static_cast<std::int64_t>(words) * tokens, "invalid mask");
    const int tile = row_tile(tokens, keys, g.block), blocks = (keys + g.block - 1) / g.block;
    auto scope = workspace.scope();
    auto scores = workspace.alloc(DType::FP32, {blocks, tile});
    for (int first = 0; first < tokens; first += tile) {
        select_kernel<<<std::min(tile, tokens - first), kThreads, 0, stream>>>(
            static_cast<const float*>(q.data), static_cast<const float*>(head_weights.data),
            static_cast<const int*>(positions.data), static_cast<const int*>(tables.data),
            static_cast<const int*>(valid.data), width, first,
            static_cast<const int*>(cache.block_tables.data), cache.block_tables.ne[0],
            static_cast<const float*>(cache.indexer_pages.data), static_cast<unsigned*>(mask.data),
            static_cast<float*>(scores.data), blocks, words, g.head_dim, g.heads, g.block, g.top_k / g.block);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace sinfer::ops
