#include "kernels/decode.h"
#include "utilities/utils.h"
#include <cuda_bf16.h>
#include <cmath>
#include <stdexcept>

namespace {
constexpr int Page = 128;

// Each block reads one physical page. Four warps calculate four QK dots at a
// time, then update an FP32 softmax accumulator. No full-prefix KV gather.
template <int MaxD, bool Batched = false>
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
                               const DecodeCacheBinding* bindings = nullptr) {
    const int batch = Batched ? blockIdx.z / T : 0;
    int head = blockIdx.y, query = blockIdx.z % T;
    if constexpr (Batched) {
        position = bindings[batch].length;
        pages = reinterpret_cast<const nv_bfloat16* const*>(bindings[batch].data);
    }
    const int first_page = window > 0 ? max(0, position + query - window + 1) / Page : 0;
    const int page = blockIdx.x + (Batched ? first_page : 0);
    const int global_query = batch * T + query;
    auto* target = partial + ((global_query * Hq + head) * np + blockIdx.x) * (D + 2);
    if (page * Page > position + query) {
        for (int d = threadIdx.x; d < D + 2; d += blockDim.x) target[d] = d == D ? -INFINITY : 0;
        return;
    }
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int kvhead = head / (Hq / Hkv), absolute = position + query;
    const nv_bfloat16* q = qkv + (global_query * (Hq + 2 * Hkv) + head) * D;
    const nv_bfloat16* kv = pages[page];
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
            for (int d = lane; d < D; d += 32)
                dot += __bfloat162float(q[d]) * __bfloat162float(kv[((start + warp) * 2 * Hkv + kvhead) * D + d]);
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
                    if (weights[i] != 0)
                        accum[j] += weights[i] * __bfloat162float(kv[((start + i) * 2 * Hkv + Hkv + kvhead) * D + d]);
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

__global__ void merge_pages(const float* partial, nv_bfloat16* out, float* lse, int T, int H, int D, int np) {
    int row = blockIdx.x, d = threadIdx.x;
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
        out[row * D + d] = __float2bfloat16(value / total);
    }
    if (threadIdx.x == 0) lse[((row / (H * T)) * H + row % H) * T + (row / H) % T] = maximum + logf(total);
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

__global__ void delta_rule(const nv_bfloat16* q,
                           const nv_bfloat16* k,
                           const nv_bfloat16* v,
                           const float* g,
                           const nv_bfloat16* beta,
                           float* state,
                           nv_bfloat16* out,
                           int T,
                           int H,
                           int K,
                           int V,
                           bool initial,
                           float scale) {
    int head = blockIdx.x, col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= V) return;
    float* s = state + head * K * V + col;
    if (initial)
        for (int d = 0; d < K; ++d)
            s[d * V] = 0;
    for (int t = 0; t < T; ++t) {
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
}
}  // namespace

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
                       const Tensor& state,
                       const Tensor& out,
                       bool initial,
                       float scale,
                       cudaStream_t stream) {
    if (g.DType != ETensorDType::FP32 || q.DType != ETensorDType::BF16 || beta.DType != ETensorDType::BF16)
        throw std::runtime_error("Delta decode requires BF16 activations and FP32 decay");
    int T = q.Sizes[1], H = q.Sizes[2], K = q.Sizes[3], V = v.Sizes[3];
    delta_rule<<<dim3(H, (V + 63) / 64), 64, 0, stream>>>(q.get<nv_bfloat16>(),
                                                          k.get<nv_bfloat16>(),
                                                          v.get<nv_bfloat16>(),
                                                          g.get<float>(),
                                                          beta.get<nv_bfloat16>(),
                                                          reinterpret_cast<float*>(state.Data),
                                                          reinterpret_cast<nv_bfloat16*>(out.Data),
                                                          T,
                                                          H,
                                                          K,
                                                          V,
                                                          initial,
                                                          scale);
    CUDA_CHECK(cudaGetLastError());
}
