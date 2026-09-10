#include "kernels/decode.h"
#include "utilities/utils.h"
#include <cuda_bf16.h>
#include <cmath>
#include <stdexcept>

namespace {
constexpr int Page = 128;

// Each block reads one physical page. Four warps calculate four QK dots at a
// time, then update an FP32 softmax accumulator. No full-prefix KV gather.
template <int MaxD>
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
                               int np) {
    int page = blockIdx.x, head = blockIdx.y, query = blockIdx.z;
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int kvhead = head / (Hq / Hkv), absolute = position + query;
    const nv_bfloat16* q = qkv + (query * (Hq + 2 * Hkv) + head) * D;
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
    auto* target = partial + ((query * Hq + head) * np + page) * (D + 2);
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
    if (threadIdx.x == 0) lse[(row % H) * T + row / H] = maximum + logf(total);
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
