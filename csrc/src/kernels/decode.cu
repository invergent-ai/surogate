#include "kernels/decode.h"
#include "utilities/utils.h"
#include <cuda_bf16.h>
#include <cmath>
#include <stdexcept>

namespace {
constexpr int Page = 128;

// Each block reads one physical page. Four warps calculate four QK dots at a
// time, then update an FP32 softmax accumulator. No full-prefix KV gather.
template <int MaxD, bool Batched = false, bool Dense = false>
__global__ void page_attention(const nv_bfloat16* qkv,
                               const nv_bfloat16* const* pages,
                               float* partial,
                               int position,
                               int T,
                               int Hq,
                               int Hkv,
                               int D,
                               int window,
                               float scale,
                               int np,
                               const DecodeCacheBinding* bindings = nullptr,
                               const int* cu_seqlens = nullptr, int num_docs = 0,
                               int query_start = 0, int query_count = 0) {
    const int active_t = query_count ? query_count : T;
    const int batch = Batched || Dense ? blockIdx.z / active_t : 0;
    const int local_query = blockIdx.z % active_t;
    int head = blockIdx.y, query = local_query + query_start;
    const int global_query = batch * T + query;
    int document_start = batch * T;
    if constexpr (Dense) {
        if (cu_seqlens && num_docs > 0) {
            int lo = 0, hi = num_docs;
            while (lo + 1 < hi) {
                const int mid = (lo + hi) / 2;
                if (cu_seqlens[mid] <= global_query) lo = mid;
                else hi = mid;
            }
            document_start = cu_seqlens[lo];
        }
        position = batch * T - document_start;
    }
    if constexpr (Batched) {
        position = bindings[batch].length;
        pages = reinterpret_cast<const nv_bfloat16* const*>(bindings[batch].data);
    }
    const int first_page = window > 0 ? max(0, position + query - window + 1) / Page : 0;
    const int page = blockIdx.x + (Batched || Dense ? first_page : 0);
    const int scratch_query = batch * active_t + local_query;
    auto* target = partial + ((scratch_query * Hq + head) * np + blockIdx.x) * (D + 2);
    if (page * Page > position + query) {
        for (int d = threadIdx.x; d < D + 2; d += blockDim.x) target[d] = d == D ? -INFINITY : 0;
        return;
    }
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int kvhead = head / (Hq / Hkv), absolute = position + query;
    const nv_bfloat16* q = qkv + (global_query * (Hq + 2 * Hkv) + head) * D;
    const nv_bfloat16* kv = Dense ? nullptr : pages[page];
    __shared__ float scores[4], weights[4], alpha, maximum, total;
    float accum[(MaxD + 127) / 128] = {};
    if (tid == 0) {
        maximum = -INFINITY;
        total = 0;
    }
    __syncthreads();
    for (int start = 0; start < Page; start += 4) {
        int token = page * Page + start + warp;
        bool valid = token <= absolute && (!window || token > absolute - window);
        float dot = 0;
        if (valid) {
            for (int d = lane; d < D; d += 32) {
                const auto key = Dense ? qkv[((document_start + token) * (Hq + 2 * Hkv) + Hq + kvhead) * D + d]
                                       : kv[((start + warp) * 2 * Hkv + kvhead) * D + d];
                dot += __bfloat162float(q[d]) * __bfloat162float(key);
            }
        }
        for (int delta = 16; delta; delta /= 2)
            dot += __shfl_down_sync(0xffffffff, dot, delta);
        if (lane == 0) scores[warp] = valid ? dot * scale : -INFINITY;
        __syncthreads();
        if (tid == 0) {
            float next = maximum;
            for (int i = 0; i < 4; ++i)
                next = fmaxf(next, scores[i]);
            alpha = isfinite(next) && isfinite(maximum) ? expf(maximum - next) : 0;
            total *= alpha;
            for (int i = 0; i < 4; ++i) {
                weights[i] = isfinite(scores[i]) ? expf(scores[i] - next) : 0;
                total += weights[i];
            }
            maximum = next;
        }
        __syncthreads();
        for (int j = 0; j < (MaxD + 127) / 128; ++j) {
            int d = tid + j * 128;
            accum[j] *= alpha;
            if (d < D)
                for (int i = 0; i < 4; ++i)
                    if (weights[i] != 0) {
                        const auto value = Dense ? qkv[((document_start + page * Page + start + i) * (Hq + 2 * Hkv) + Hq + Hkv + kvhead) * D + d]
                                                 : kv[((start + i) * 2 * Hkv + Hkv + kvhead) * D + d];
                        accum[j] += weights[i] * __bfloat162float(value);
                    }
        }
        __syncthreads();
    }
    for (int j = 0; j < (MaxD + 127) / 128; ++j)
        if (tid + j * 128 < D) target[tid + j * 128] = accum[j];
    if (tid == 0) {
        target[D] = maximum;
        target[D + 1] = total;
    }
}

__global__ void merge_pages(const float* partial, nv_bfloat16* out, float* lse, int T, int H, int D, int np,
                             int query_start = 0, int full_t = 0) {
    int row = blockIdx.x, d = threadIdx.x;
    if (!full_t) full_t = T;
    const int batch = row / (H * T), query = (row / H) % T + query_start, head = row % H;
    float maximum = -INFINITY, total = 0;
    for (int p = 0; p < np; ++p)
        maximum = fmaxf(maximum, partial[(row * np + p) * (D + 2) + D]);
    for (int p = 0; p < np; ++p) {
        const auto* src = partial + (row * np + p) * (D + 2);
        if (src[D + 1] > 0) total += expf(src[D] - maximum) * src[D + 1];
    }
    for (; d < D; d += blockDim.x) {
        float value = 0;
        for (int p = 0; p < np; ++p) {
            const auto* src = partial + (row * np + p) * (D + 2);
            if (src[D + 1] > 0) value += expf(src[D] - maximum) * src[d];
        }
        out[((batch * full_t + query) * H + head) * D + d] = __float2bfloat16(value / total);
    }
    if (threadIdx.x == 0) lse[(batch * H + head) * full_t + query] = maximum + logf(total);
}

__global__ void append_batch_kv(const nv_bfloat16* qkv, const DecodeCacheBinding* bindings,
                                int T, int Hq, int Hkv, int D) {
    const int row = blockIdx.x / T, token = blockIdx.x % T;
    const int absolute = bindings[row].length + token;
    auto* const* pages = reinterpret_cast<nv_bfloat16* const*>(bindings[row].data);
    auto* destination = pages[absolute / Page] + (absolute % Page) * 2 * Hkv * D;
    const auto* source = qkv + ((row * T + token) * (Hq + 2 * Hkv) + Hq) * D;
    for (int col = threadIdx.x; col < 2 * Hkv * D; col += blockDim.x) destination[col] = source[col];
}

__global__ void
gather_pages(const nv_bfloat16* const* pages, const int* indices, nv_bfloat16* out, int* slots, int count, int width) {
    int slot = blockIdx.x, index = indices[slot];
    const nv_bfloat16* src = index >= 0 ? pages[index / Page] : nullptr;
    for (int d = threadIdx.x; d < width; d += blockDim.x)
        out[slot * width + d] = src ? src[(index % Page) * width + d] : __float2bfloat16(0.f);
    if (threadIdx.x == 0 && slots) slots[slot] = index >= 0 ? slot : -1;
}

__device__ int document_index(const int* cu_seqlens, int num_docs, int token) {
    int lo = 0, hi = num_docs;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) / 2;
        if (cu_seqlens[mid] <= token) lo = mid;
        else hi = mid;
    }
    return lo;
}

template <bool Training = false>
__global__ void delta_rule(const nv_bfloat16* q,
                           const nv_bfloat16* k,
                           const nv_bfloat16* v,
                           const float* g,
                           const nv_bfloat16* beta,
                           const DecodeCacheBinding* bindings,
                           nv_bfloat16* out,
                           float* final_state,
                           int T,
                           int H,
                           int K,
                           int V,
                           float scale,
                           float* checkpoints = nullptr,
                           const int* cu_seqlens = nullptr,
                           int num_docs = 0) {
    int head = blockIdx.x, col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= V) return;
    const int row = blockIdx.z;
    const bool initial = Training || bindings[row].length == 0;
    auto* state = Training ? final_state + row * H * K * V : static_cast<float*>(bindings[row].data);
    q += row * T * H * K;
    k += row * T * H * K;
    v += row * T * H * V;
    g += row * T * H;
    beta += row * T * H;
    out += row * T * H * V;
    float* s = state + head * K * V + col;
    if (initial)
        for (int d = 0; d < K; ++d)
            s[d * V] = 0;
    int doc = cu_seqlens ? document_index(cu_seqlens, num_docs, row * T) : 0;
    const int NC = (T + DELTA_RULE_CHECKPOINT - 1) / DELTA_RULE_CHECKPOINT;
    for (int t = 0; t < T; ++t) {
        if constexpr (Training) {
            if (cu_seqlens && row * T + t == cu_seqlens[doc + 1]) {
                ++doc;
                for (int d = 0; d < K; ++d) s[d * V] = 0;
            }
            if (checkpoints && t % DELTA_RULE_CHECKPOINT == 0)
                for (int d = 0; d < K; ++d)
                    checkpoints[(((long(row) * NC + t / DELTA_RULE_CHECKPOINT) * H + head) * K + d) * V + col] = s[d * V];
        }
        const auto* kt = k + (t * H + head) * K;
        const auto* qt = q + (t * H + head) * K;
        float decay = expf(g[t * H + head]), prediction = 0;
        for (int d = 0; d < K; ++d)
            prediction += (s[d * V] * decay) * __bfloat162float(kt[d]);
        float residual =
            (__bfloat162float(v[(t * H + head) * V + col]) - prediction) * __bfloat162float(beta[t * H + head]);
        float value = 0;
        for (int d = 0; d < K; ++d) {
            float next = s[d * V] * decay + __bfloat162float(kt[d]) * residual;
            s[d * V] = next;
            value += next * __bfloat162float(qt[d]);
        }
        out[(t * H + head) * V + col] = __float2bfloat16(value * scale);
    }
    if (final_state)
        for (int d = 0; d < K; ++d)
            final_state[((row * H + head) * K + d) * V + col] = s[d * V];
}

// Recompute bounded intervals of the FP32 recurrence for its exact adjoint.
// Each block owns one head, so reductions cannot cross packed documents.
template <int MaxK>
__global__ void delta_rule_backward(const nv_bfloat16* q, const nv_bfloat16* k, const nv_bfloat16* v,
                                    const float* g, const nv_bfloat16* beta, const nv_bfloat16* dout,
                                    nv_bfloat16* dq, nv_bfloat16* dk, nv_bfloat16* dv, float* dg,
                                    nv_bfloat16* dbeta, const float* checkpoints,
                                    const int* cu_seqlens, int num_docs, int T, int H, int K, int V, float scale) {
    const int row = blockIdx.y, head = blockIdx.x, col = threadIdx.x;
    const bool active = col < V;
    const int NC = (T + DELTA_RULE_CHECKPOINT - 1) / DELTA_RULE_CHECKPOINT;
    extern __shared__ float reductions[];
    const int warps = blockDim.x / 32, warp = col / 32, lane = col % 32;
    float* qgrad = reductions;
    float* kgrad = qgrad + K * warps;
    float* ggrad = kgrad + K * warps;
    float* bgrad = ggrad + warps;
    auto warp_sum = [](float value) {
        for (int offset = 16; offset; offset /= 2) value += __shfl_down_sync(0xffffffff, value, offset);
        return value;
    };
    float state[MaxK], adj[MaxK] = {};
    for (int t = T - 1; t >= 0; --t) {
        const int token = row * T + t;
        const int doc_start = cu_seqlens ? cu_seqlens[document_index(cu_seqlens, num_docs, token)] - row * T : 0;
        const int start = t / DELTA_RULE_CHECKPOINT * DELTA_RULE_CHECKPOINT;
        if (active) {
            for (int d = 0; d < K; ++d)
                state[d] = start < doc_start ? 0.f : checkpoints[(((long(row) * NC + t / DELTA_RULE_CHECKPOINT) * H + head) * K + d) * V + col];
            for (int r = max(start, doc_start); r < t; ++r) {
                const long i = (long(row) * T + r) * H + head;
                const float decay = expf(g[i]);
                float prediction = 0.f;
                for (int d = 0; d < K; ++d) prediction += (state[d] * decay) * __bfloat162float(k[i * K + d]);
                const float residual = (__bfloat162float(v[i * V + col]) - prediction) * __bfloat162float(beta[i]);
                for (int d = 0; d < K; ++d) state[d] = state[d] * decay + __bfloat162float(k[i * K + d]) * residual;
            }
        }
        const long i = (long(row) * T + t) * H + head;
        const float decay = expf(g[i]), be = __bfloat162float(beta[i]);
        const float dy = active ? __bfloat162float(dout[i * V + col]) * scale : 0.f;
        float error = 0.f, residual = 0.f, du = 0.f;
        if (active) {
            float prediction = 0.f;
            for (int d = 0; d < K; ++d) prediction += (state[d] * decay) * __bfloat162float(k[i * K + d]);
            error = __bfloat162float(v[i * V + col]) - prediction;
            residual = error * be;
            for (int d = 0; d < K; ++d) {
                adj[d] += __bfloat162float(q[i * K + d]) * dy;
                du += __bfloat162float(k[i * K + d]) * adj[d];
            }
            dv[i * V + col] = __float2bfloat16(be * du);
        }
        float ddecay = 0.f;
        for (int d = 0; d < K; ++d) {
            const float key = __bfloat162float(k[i * K + d]), decayed = active ? state[d] * decay : 0.f;
            const float dq_part = warp_sum(dy * (decayed + key * residual));
            const float dk_part = warp_sum(active ? adj[d] * residual - be * du * decayed : 0.f);
            if (!lane) { qgrad[d * warps + warp] = dq_part; kgrad[d * warps + warp] = dk_part; }
            if (active) {
                const float ds = adj[d] - key * be * du;
                ddecay += ds * decayed;
                adj[d] = t == doc_start ? 0.f : ds * decay;
            }
        }
        const float dg_part = warp_sum(ddecay), db_part = warp_sum(error * du);
        if (!lane) { ggrad[warp] = dg_part; bgrad[warp] = db_part; }
        __syncthreads();
        // Sum warp partials in a fixed order, avoiding scheduling-dependent
        // atomic accumulation at BF16 rounding boundaries.
        for (int d = col; d < K; d += blockDim.x) {
            float qsum = 0.f, ksum = 0.f;
            for (int w = 0; w < warps; ++w) { qsum += qgrad[d * warps + w]; ksum += kgrad[d * warps + w]; }
            dq[i * K + d] = __float2bfloat16(qsum);
            dk[i * K + d] = __float2bfloat16(ksum);
        }
        if (!col) {
            float gsum = 0.f, bsum = 0.f;
            for (int w = 0; w < warps; ++w) { gsum += ggrad[w]; bsum += bgrad[w]; }
            dg[i] = gsum; dbeta[i] = __float2bfloat16(bsum);
        }
        __syncthreads();
    }
}

template <bool Scatter>
__global__ void copy_state(const DecodeCacheBinding* bindings, int* batch, long words) {
    const int row = blockIdx.y;
    auto* state = static_cast<int*>(bindings[row].data);
    for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += gridDim.x * blockDim.x) {
        if constexpr (Scatter) state[i] = batch[row * words + i];
        else batch[row * words + i] = bindings[row].length ? state[i] : 0;
    }
}

template <typename T>
__global__ void conv_input(const T* input, const DecodeCacheBinding* bindings, T* extended,
                           int length, int channels, int tail) {
    const int row = blockIdx.y;
    const auto* state = static_cast<const T*>(bindings[row].data);
    for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < channels * (length + tail); i += gridDim.x * blockDim.x) {
        const int c = i / (length + tail), t = i % (length + tail);
        extended[row * channels * (length + tail) + i] = t >= tail
            ? input[(row * channels + c) * length + t - tail]
            : bindings[row].length ? state[c * tail + t] : T{};
    }
}

template <typename T>
__global__ void conv_output(const T* computed, const T* extended, const DecodeCacheBinding* bindings,
                            T* output, int length, int channels, int tail) {
    const int row = blockIdx.y;
    auto* state = static_cast<T*>(bindings[row].data);
    for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < channels * (length + tail); i += gridDim.x * blockDim.x) {
        const int c = i / (length + tail), t = i % (length + tail);
        const long base = (row * channels + c) * (length + tail);
        if (t < length) output[(row * channels + c) * length + t] = computed[base + tail + t];
        else state[c * tail + t - length] = extended[base + t];
    }
}

template <typename T>
__global__ void gather_rows(const T* input, const int* positions, T* output, int length, int width) {
    const int row = blockIdx.x;
    for (int col = threadIdx.x; col < width; col += blockDim.x)
        output[row * width + col] = input[(static_cast<long>(row) * length + positions[row]) * width + col];
}

template <typename D>
__global__ void append_pages(const D* input, const DecodeCacheBinding* bindings, int T, int width) {
    const int row = blockIdx.x / T, token = blockIdx.x % T;
    const int absolute = bindings[row].length + token;
    auto* const* pages = static_cast<D* const*>(bindings[row].data);
    auto* dst = pages[absolute / Page] + (absolute % Page) * width;
    for (int col = threadIdx.x; col < width; col += blockDim.x) dst[col] = input[blockIdx.x * width + col];
}

__global__ void gather_pages_batch(const DecodeCacheBinding* bindings, const int* indices,
                                   nv_bfloat16* out, int* slots, int T, int query_start, int count, int width) {
    const int slot = blockIdx.x, query = blockIdx.y + query_start, row = query / T;
    const int index = indices[query * count + slot];
    auto* const* pages = static_cast<nv_bfloat16* const*>(bindings[row].data);
    const auto* src = index >= 0 ? pages[index / Page] : nullptr;
    const long output_slot = blockIdx.y * count + slot;
    for (int d = threadIdx.x; d < width; d += blockDim.x)
        out[output_slot * width + d] = src ? src[(index % Page) * width + d] : __float2bfloat16(0.f);
    if (!threadIdx.x) slots[output_slot] = index >= 0 ? slot : -1;
}
}  // namespace

void rollout_consistent_attention(const Tensor& qkv, const Tensor& out, const Tensor& lse,
                                  const Tensor& scratch, const int* cu_seqlens, int num_docs,
                                  int B, int T, int Hq, int Hkv, int D, int window,
                                  float scale, int query_tile, cudaStream_t stream) {
    if (qkv.DType != ETensorDType::BF16 || D <= 0 || D > 1024 || Hkv <= 0 || Hq % Hkv || query_tile <= 0)
        throw std::invalid_argument("Invalid rollout-consistent attention geometry");
    int np = (T + Page - 1) / Page;
    if (window > 0) np = std::min(np, (window + 2 * Page - 2) / Page);
    for (int start = 0; start < T; start += query_tile) {
        const int count = std::min(query_tile, T - start);
        auto launch = [&]<int N>() {
            page_attention<N, false, true><<<dim3(np, Hq, B * count), 128, 0, stream>>>(
                qkv.get<nv_bfloat16>(), nullptr, reinterpret_cast<float*>(scratch.Data),
                0, T, Hq, Hkv, D, window, scale ? scale : 1.f / sqrtf(D), np,
                nullptr, cu_seqlens, num_docs, start, count);
        };
        if (D <= 128) launch.template operator()<128>();
        else if (D <= 256) launch.template operator()<256>();
        else if (D <= 512) launch.template operator()<512>();
        else launch.template operator()<1024>();
        merge_pages<<<B * count * Hq, 128, 0, stream>>>(reinterpret_cast<float*>(scratch.Data),
            reinterpret_cast<nv_bfloat16*>(out.Data), reinterpret_cast<float*>(lse.Data), count, Hq, D, np, start, T);
    }
    CUDA_CHECK(cudaGetLastError());
}

void decode_paged_attention(const Tensor& qkv,
                            const Tensor& out,
                            const Tensor& lse,
                            const Tensor& pages,
                            const Tensor& scratch,
                            int position,
                            int T,
                            int Hq,
                            int Hkv,
                            int D,
                            int window,
                            float scale,
                            cudaStream_t stream) {
    if (qkv.DType != ETensorDType::BF16 || D <= 0 || D > 1024 || Hq % Hkv)
        throw std::runtime_error("Paged decode attention requires BF16 GQA with head size <= 1024");
    int np = (position + T + Page - 1) / Page;
    dim3 grid(np, Hq, T);
    auto launch = [&]<int N>() {
        page_attention<N><<<grid, 128, 0, stream>>>(qkv.get<nv_bfloat16>(),
                                                    reinterpret_cast<const nv_bfloat16* const*>(pages.Data),
                                                    reinterpret_cast<float*>(scratch.Data),
                                                    position,
                                                    T,
                                                    Hq,
                                                    Hkv,
                                                    D,
                                                    window,
                                                    scale ? scale : 1.f / sqrtf(D),
                                                    np);
    };
    if (D <= 128)
        launch.template operator()<128>();
    else if (D <= 256)
        launch.template operator()<256>();
    else if (D <= 512)
        launch.template operator()<512>();
    else
        launch.template operator()<1024>();
    merge_pages<<<T * Hq, 128, 0, stream>>>(reinterpret_cast<float*>(scratch.Data),
                                            reinterpret_cast<nv_bfloat16*>(out.Data),
                                            reinterpret_cast<float*>(lse.Data),
                                            T,
                                            Hq,
                                            D,
                                            np);
    CUDA_CHECK(cudaGetLastError());
}

void decode_append_kv_batch(const Tensor& qkv, const Tensor& bindings, int B, int T,
                            int Hq, int Hkv, int D, cudaStream_t stream) {
    append_batch_kv<<<B * T, 256, 0, stream>>>(qkv.get<nv_bfloat16>(),
        reinterpret_cast<const DecodeCacheBinding*>(bindings.Data), T, Hq, Hkv, D);
    CUDA_CHECK(cudaGetLastError());
}

void decode_paged_attention_batch(const Tensor& qkv, const Tensor& out, const Tensor& lse,
                                  const Tensor& bindings, const Tensor& scratch, int B, int T,
                                  int Hq, int Hkv, int D, int pages, int window, float scale,
                                  cudaStream_t stream) {
    if (qkv.DType != ETensorDType::BF16 || D <= 0 || D > 1024 || Hkv <= 0 || Hq % Hkv)
        throw std::runtime_error("Paged batch attention requires BF16 GQA with head size <= 1024");
    auto launch = [&]<int N>() {
        page_attention<N, true><<<dim3(pages, Hq, B * T), 128, 0, stream>>>(qkv.get<nv_bfloat16>(), nullptr,
            reinterpret_cast<float*>(scratch.Data), 0, T, Hq, Hkv, D, window,
            scale ? scale : 1.f / sqrtf(D), pages, reinterpret_cast<const DecodeCacheBinding*>(bindings.Data));
    };
    if (D <= 128) launch.template operator()<128>();
    else if (D <= 256) launch.template operator()<256>();
    else if (D <= 512) launch.template operator()<512>();
    else launch.template operator()<1024>();
    merge_pages<<<B * T * Hq, 128, 0, stream>>>(reinterpret_cast<float*>(scratch.Data),
        reinterpret_cast<nv_bfloat16*>(out.Data), reinterpret_cast<float*>(lse.Data), T, Hq, D, pages);
    CUDA_CHECK(cudaGetLastError());
}

void decode_gather_pages(const Tensor& pages,
                         const Tensor& indices,
                         const Tensor& out,
                         const Tensor& slots,
                         int count,
                         int width,
                         cudaStream_t stream) {
    if (!count) return;
    gather_pages<<<count, 256, 0, stream>>>(reinterpret_cast<const nv_bfloat16* const*>(pages.Data),
                                            indices.get<int>(),
                                            reinterpret_cast<nv_bfloat16*>(out.Data),
                                            slots.Data ? reinterpret_cast<int*>(slots.Data) : nullptr,
                                            count,
                                            width);
    CUDA_CHECK(cudaGetLastError());
}

void decode_delta_rule(const Tensor& q,
                       const Tensor& k,
                       const Tensor& v,
                       const Tensor& g,
                       const Tensor& beta,
                       const Tensor& bindings,
                       const Tensor& out,
                       const Tensor& final_state,
                       float scale,
                       cudaStream_t stream) {
    if (g.DType != ETensorDType::FP32 || q.DType != ETensorDType::BF16 || beta.DType != ETensorDType::BF16)
        throw std::runtime_error("Delta decode requires BF16 activations and FP32 decay");
    int T = q.Sizes[1], H = q.Sizes[2], K = q.Sizes[3], V = v.Sizes[3];
    delta_rule<<<dim3(H, (V + 63) / 64, q.Sizes[0]), 64, 0, stream>>>(q.get<nv_bfloat16>(),
                                                          k.get<nv_bfloat16>(),
                                                          v.get<nv_bfloat16>(),
                                                          g.get<float>(),
                                                          beta.get<nv_bfloat16>(),
                                                          reinterpret_cast<const DecodeCacheBinding*>(bindings.Data),
                                                          reinterpret_cast<nv_bfloat16*>(out.Data),
                                                          reinterpret_cast<float*>(final_state.Data),
                                                          T,
                                                          H,
                                                          K,
                                                          V,
                                                          scale);
    CUDA_CHECK(cudaGetLastError());
}

void rollout_delta_rule(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                         const Tensor& beta, const Tensor& out, const Tensor& state,
                         const Tensor& checkpoints, const int* cu_seqlens, int num_docs,
                         float scale, cudaStream_t stream) {
    const int T = q.Sizes[1], H = q.Sizes[2], K = q.Sizes[3], V = v.Sizes[3];
    delta_rule<true><<<dim3(H, (V + 63) / 64, q.Sizes[0]), 64, 0, stream>>>(
        q.get<nv_bfloat16>(), k.get<nv_bfloat16>(), v.get<nv_bfloat16>(), g.get<float>(),
        beta.get<nv_bfloat16>(), nullptr, reinterpret_cast<nv_bfloat16*>(out.Data),
        reinterpret_cast<float*>(state.Data), T, H, K, V, scale,
        reinterpret_cast<float*>(checkpoints.Data), cu_seqlens, num_docs);
    CUDA_CHECK(cudaGetLastError());
}

void rollout_delta_rule_backward(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                                  const Tensor& beta, const Tensor& dout, const Tensor& dq,
                                  const Tensor& dk, const Tensor& dv, const Tensor& dg, const Tensor& dbeta,
                                  const Tensor& checkpoints, const int* cu_seqlens, int num_docs,
                                  float scale, cudaStream_t stream) {
    const int T = q.Sizes[1], H = q.Sizes[2], K = q.Sizes[3], V = v.Sizes[3];
    if (K > 256 || V > 1024) throw std::runtime_error("Rollout delta rule head dimensions exceed 256/1024");
    auto launch = [&]<int MaxK>() {
        const int threads = ((V + 31) / 32) * 32;
        const int shared = (2 * K + 2) * (threads / 32) * sizeof(float);
        if (shared > 48 * 1024)
            CUDA_CHECK(cudaFuncSetAttribute(delta_rule_backward<MaxK>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
        delta_rule_backward<MaxK><<<dim3(H, q.Sizes[0]), threads, shared, stream>>>(
            q.get<nv_bfloat16>(), k.get<nv_bfloat16>(), v.get<nv_bfloat16>(), g.get<float>(),
            beta.get<nv_bfloat16>(), dout.get<nv_bfloat16>(), reinterpret_cast<nv_bfloat16*>(dq.Data),
            reinterpret_cast<nv_bfloat16*>(dk.Data), reinterpret_cast<nv_bfloat16*>(dv.Data),
            reinterpret_cast<float*>(dg.Data), reinterpret_cast<nv_bfloat16*>(dbeta.Data),
            checkpoints.get<float>(), cu_seqlens, num_docs, T, H, K, V, scale);
    };
    if (K <= 32) launch.template operator()<32>();
    else if (K <= 64) launch.template operator()<64>();
    else if (K <= 128) launch.template operator()<128>();
    else launch.template operator()<256>();
    CUDA_CHECK(cudaGetLastError());
}

void decode_copy_state(const Tensor& bindings, const Tensor& state, int B, bool scatter, cudaStream_t stream) {
    if (state.bytes() % (B * sizeof(int))) throw std::invalid_argument("Unaligned recurrent decode state");
    const long words = state.bytes() / (B * sizeof(int));
    auto* meta = reinterpret_cast<const DecodeCacheBinding*>(bindings.Data);
    dim3 grid(std::min<long>(256, (words + 255) / 256), B);
    if (scatter) copy_state<true><<<grid, 256, 0, stream>>>(meta, reinterpret_cast<int*>(state.Data), words);
    else copy_state<false><<<grid, 256, 0, stream>>>(meta, reinterpret_cast<int*>(state.Data), words);
    CUDA_CHECK(cudaGetLastError());
}

void decode_conv_input(const Tensor& x, const Tensor& bindings, const Tensor& extended, int tail, cudaStream_t stream) {
    const int B = x.Sizes[0], C = x.Sizes[1], T = x.Sizes[2];
    auto* meta = reinterpret_cast<const DecodeCacheBinding*>(bindings.Data);
    auto launch = [&]<typename D>() {
        conv_input<<<dim3((C * (T + tail) + 255) / 256, B), 256, 0, stream>>>(
            reinterpret_cast<const D*>(x.Data), meta, reinterpret_cast<D*>(extended.Data), T, C, tail);
    };
    if (x.DType == ETensorDType::BF16) launch.template operator()<uint16_t>();
    else if (x.DType == ETensorDType::FP32) launch.template operator()<uint32_t>();
    else throw std::invalid_argument("Decode convolution requires BF16 or FP32");
    CUDA_CHECK(cudaGetLastError());
}

void decode_conv_output(const Tensor& computed, const Tensor& extended, const Tensor& bindings,
                        const Tensor& output, int tail, cudaStream_t stream) {
    const int B = output.Sizes[0], C = output.Sizes[1], T = output.Sizes[2];
    auto* meta = reinterpret_cast<const DecodeCacheBinding*>(bindings.Data);
    auto launch = [&]<typename D>() {
        conv_output<<<dim3((C * (T + tail) + 255) / 256, B), 256, 0, stream>>>(
            reinterpret_cast<const D*>(computed.Data), reinterpret_cast<const D*>(extended.Data), meta,
            reinterpret_cast<D*>(output.Data), T, C, tail);
    };
    if (output.DType == ETensorDType::BF16) launch.template operator()<uint16_t>();
    else if (output.DType == ETensorDType::FP32) launch.template operator()<uint32_t>();
    else throw std::invalid_argument("Decode convolution requires BF16 or FP32");
    CUDA_CHECK(cudaGetLastError());
}

void decode_gather_rows(const Tensor& input, const Tensor& positions, const Tensor& output,
                        int B, int T, int C, cudaStream_t stream) {
    auto launch = [&]<typename D>() {
        gather_rows<<<B, 256, 0, stream>>>(reinterpret_cast<const D*>(input.Data), positions.get<int>(),
                                          reinterpret_cast<D*>(output.Data), T, C);
    };
    if (input.DType == ETensorDType::BF16) launch.template operator()<uint16_t>();
    else if (input.DType == ETensorDType::FP32) launch.template operator()<uint32_t>();
    else throw std::invalid_argument("Decode projection requires BF16 or FP32");
    CUDA_CHECK(cudaGetLastError());
}

void decode_append_pages_batch(const Tensor& input, const Tensor& bindings, int B, int T, int width, cudaStream_t stream) {
    auto* meta = reinterpret_cast<const DecodeCacheBinding*>(bindings.Data);
    if (get_dtype_size(input.DType) == 2)
        append_pages<<<B * T, 256, 0, stream>>>(reinterpret_cast<const uint16_t*>(input.Data), meta, T, width);
    else if (get_dtype_size(input.DType) == 4)
        append_pages<<<B * T, 256, 0, stream>>>(reinterpret_cast<const uint32_t*>(input.Data), meta, T, width);
    else throw std::invalid_argument("Unsupported paged decode element size");
    CUDA_CHECK(cudaGetLastError());
}

void decode_gather_pages_batch(const Tensor& bindings, const Tensor& indices, const Tensor& out, const Tensor& slots,
                               int T, int query_start, int queries, int count, int width, cudaStream_t stream) {
    if (!count) return;
    gather_pages_batch<<<dim3(count, queries), 256, 0, stream>>>(
        reinterpret_cast<const DecodeCacheBinding*>(bindings.Data), indices.get<int>(),
        reinterpret_cast<nv_bfloat16*>(out.Data), reinterpret_cast<int*>(slots.Data), T, query_start, count, width);
    CUDA_CHECK(cudaGetLastError());
}
