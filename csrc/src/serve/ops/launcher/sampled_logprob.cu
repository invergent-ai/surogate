#include "ops/launcher/sampled_logprob.h"

#include "core/device.h"
#include "api/ops/sampled_logprob.h"
#include <algorithm>
#include "ops/common/warp.cuh"

#include <cmath>
#include <cuda_bf16.h>

namespace sinfer::ops::detail {
namespace {

constexpr int kThreads = 512;

/// One block per column: a max pass and a sum pass over the vocabulary, each
/// reduced within warps and then across them through shared memory. The scale is
/// applied to every logit before both passes, so the maximum subtracted is the
/// maximum of the scaled row and the sum stays in range.
__global__ void __launch_bounds__(kThreads)
sampled_logprob_kernel(const __nv_bfloat16* __restrict__ logits, const int* __restrict__ tokens,
                       float* __restrict__ out, const SamplingConfig* __restrict__ configs,
                       const int vocab, const int domain) {
    constexpr int kWarps = kThreads / 32;
    __shared__ float partial[kWarps];
    const int column         = static_cast<int>(blockIdx.x);
    const __nv_bfloat16* row = logits + static_cast<std::size_t>(column) * vocab;
    const int tid            = static_cast<int>(threadIdx.x);
    const int warp           = tid >> 5;
    const int lane           = tid & 31;

    // A greedy row has no temperature of its own; its distribution is the raw one.
    const float temperature = configs[column].temperature;
    const float inverse     = temperature > 0.0f ? 1.0f / temperature : 1.0f;

    float local_max = -INFINITY;
    for (int v = tid; v < domain; v += kThreads) {
        local_max = fmaxf(local_max, __bfloat162float(row[v]) * inverse);
    }
    const float warp_best = warp_max(local_max);
    if (lane == 0) { partial[warp] = warp_best; }
    __syncthreads();
    float block_max = -INFINITY;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) { block_max = fmaxf(block_max, partial[w]); }
    __syncthreads();

    float local_sum = 0.0f;
    for (int v = tid; v < domain; v += kThreads) {
        local_sum += expf(__bfloat162float(row[v]) * inverse - block_max);
    }
    local_sum = warp_sum(local_sum);
    if (lane == 0) { partial[warp] = local_sum; }
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) { total += partial[w]; }
        const int token = tokens[column];
        const float chosen =
            token >= 0 && token < domain ? __bfloat162float(row[token]) * inverse : NAN;
        out[column] = chosen - (block_max + logf(total));
    }
}

} // namespace

void sampled_logprob_launch(const Tensor& logits, const Tensor& tokens, Tensor& out,
                            std::int32_t token_domain, const SamplingConfig* configs,
                            cudaStream_t stream) {
    const int vocab   = logits.ne[0];
    const int columns = logits.ne[1];
    if (columns <= 0) { return; }
    sampled_logprob_kernel<<<static_cast<unsigned>(columns), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), static_cast<const int*>(tokens.data),
        static_cast<float*>(out.data), configs, vocab, static_cast<int>(token_domain));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail

namespace sinfer::ops {
namespace {
__global__ void score_kernel(const __nv_bfloat16* logits, TokenLogprob* out,
                             int token, int domain, int count) {
    __shared__ float values[512];
    __shared__ int indices[512];
    __shared__ float normalizer;
    const int tid = threadIdx.x;
    float best = -INFINITY;
    for (int i = tid; i < domain; i += 512) best = fmaxf(best, __bfloat162float(logits[i]));
    values[tid] = best;
    __syncthreads();
    for (int step = 256; step; step >>= 1) {
        if (tid < step) values[tid] = fmaxf(values[tid], values[tid + step]);
        __syncthreads();
    }
    float maximum = values[0];
    float sum = 0;
    int rank = 0;
    const float selected = __bfloat162float(logits[token]);
    for (int i = tid; i < domain; i += 512) {
        const float value = __bfloat162float(logits[i]);
        sum += expf(value - maximum);
        rank += value >= selected;
    }
    __syncthreads();
    values[tid] = sum;
    indices[tid] = rank;
    __syncthreads();
    for (int step = 256; step; step >>= 1) {
        if (tid < step) { values[tid] += values[tid + step]; indices[tid] += indices[tid + step]; }
        __syncthreads();
    }
    if (tid == 0) {
        normalizer = maximum + logf(values[0]);
        out[0] = {token, selected - normalizer, indices[0]};
    }
    __syncthreads();
    for (int k = 0; k < count; ++k) {
        float value = -INFINITY;
        int index = domain;
        for (int i = tid; i < domain; i += 512) {
            bool used = false;
            for (int j = 1; j <= k; ++j) used |= out[j].token_id == i;
            float v = __bfloat162float(logits[i]);
            if (!used && (v > value || (v == value && i < index))) { value = v; index = i; }
        }
        values[tid] = value; indices[tid] = index;
        __syncthreads();
        for (int step = 256; step; step >>= 1) {
            if (tid < step && (values[tid + step] > values[tid] ||
                (values[tid + step] == values[tid] && indices[tid + step] < indices[tid]))) {
                values[tid] = values[tid + step]; indices[tid] = indices[tid + step];
            }
            __syncthreads();
        }
        if (tid == 0) out[k + 1] = {indices[0], values[0] - normalizer, k + 1};
        __syncthreads();
    }
}
}
TokenScore score_logprobs(const Tensor& logits, TokenId token, int domain, int top_k,
                         cudaStream_t stream) {
    if (logits.dtype != DType::BF16 || !logits.data || domain <= 0 || domain > logits.ne[0] ||
        token < 0 || token >= domain || top_k < 0 || top_k > 20) {
        throw std::invalid_argument("invalid log-probability scoring arguments");
    }
    top_k = std::min(top_k, domain);
    TokenLogprob* device_out = nullptr;
    CUDA_CHECK(cudaMallocAsync(&device_out, 21 * sizeof(TokenLogprob), stream));
    std::array<TokenLogprob, 21> host;
    try {
        score_kernel<<<1, 512, 0, stream>>>(static_cast<const __nv_bfloat16*>(logits.data),
                                          device_out, token, domain, top_k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(host.data(), device_out, (top_k + 1) * sizeof(TokenLogprob),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (...) { cudaFreeAsync(device_out, stream); throw; }
    CUDA_CHECK(cudaFreeAsync(device_out, stream));
    return {host[0], {host.begin() + 1, host.begin() + top_k + 1}};
}
} // namespace sinfer::ops
