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
__global__ void score_kernel(const __nv_bfloat16* all_logits, const int* tokens,
                             RawTokenScores* output, int vocab, int domain,
                             const SamplingConfig* configs, int uniform_count,
                             int width, const int* counts) {
    const int column = blockIdx.x;
    const int lane = column / width;
    const int count = configs ? configs[lane].top_logprobs : uniform_count;
    if (count < 0 || (counts && column % width >= counts[lane])) return;
    const auto* logits = all_logits + std::size_t(column) * vocab;
    const int token = tokens[column];
    if (token < 0 || token >= domain) return;
    RawTokenScores& out = output[column];
    auto* top = reinterpret_cast<TokenLogprob*>(&out.top);
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
        out.selected = {token, selected - normalizer, indices[0]};
        out.count = min(min(count, domain), 20);
    }
    __syncthreads();
    for (int k = 0; k < out.count; ++k) {
        float value = -INFINITY;
        int index = domain;
        for (int i = tid; i < domain; i += 512) {
            bool used = false;
            for (int j = 0; j < k; ++j) used |= top[j].token_id == i;
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
        if (tid == 0) top[k] = {indices[0], values[0] - normalizer, k + 1};
        __syncthreads();
    }
}
}
void score_logprobs_device(const Tensor& logits, const Tensor& tokens, int domain,
                          const SamplingConfig* configs, RawTokenScores* output,
                          cudaStream_t stream, int width, const int* counts) {
    const int columns = logits.ne[1] * logits.ne[2];
    if (!output || !configs || width <= 0 || columns <= 0 || columns % width ||
        logits.dtype != DType::BF16 || !logits.data || domain <= 0 || domain > logits.ne[0] ||
        tokens.dtype != DType::I32 || !tokens.data || tokens.ne[0] * tokens.ne[1] < columns) {
        throw std::invalid_argument("invalid batched log-probability scoring arguments");
    }
    score_kernel<<<columns, 512, 0, stream>>>(static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<const int*>(tokens.data), output, logits.ne[0], domain, configs, -1, width, counts);
    CUDA_CHECK(cudaGetLastError());
}

std::vector<TokenScore> score_logprobs_batch(const Tensor& logits, std::span<const TokenId> tokens,
                                            int domain, int top_k, cudaStream_t stream) {
    const auto columns = tokens.size();
    if (logits.dtype != DType::BF16 || !logits.data || domain <= 0 || domain > logits.ne[0] ||
        columns == 0 || columns > std::size_t(logits.ne[1]) || top_k < 0 || top_k > 20 ||
        std::any_of(tokens.begin(), tokens.end(), [domain](auto token) { return token < 0 || token >= domain; })) {
        throw std::invalid_argument("invalid log-probability scoring arguments");
    }
    void* storage = nullptr;
    CUDA_CHECK(cudaMallocAsync(&storage, columns * (sizeof(RawTokenScores) + sizeof(TokenId)), stream));
    auto* device_out = static_cast<RawTokenScores*>(storage);
    auto* device_tokens = reinterpret_cast<TokenId*>(device_out + columns);
    std::vector<RawTokenScores> host(columns);
    try {
        CUDA_CHECK(cudaMemcpyAsync(device_tokens, tokens.data(), tokens.size_bytes(), cudaMemcpyHostToDevice, stream));
        score_kernel<<<columns, 512, 0, stream>>>(static_cast<const __nv_bfloat16*>(logits.data),
            device_tokens, device_out, logits.ne[0], domain, nullptr, top_k, 1, nullptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(host.data(), device_out, columns * sizeof(RawTokenScores),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (...) { cudaFreeAsync(storage, stream); throw; }
    CUDA_CHECK(cudaFreeAsync(storage, stream));
    std::vector<TokenScore> result;
    result.reserve(columns);
    for (const auto& row : host) result.push_back({row.selected, {row.top.begin(), row.top.begin() + row.count}});
    return result;
}

TokenScore score_logprobs(const Tensor& logits, TokenId token, int domain, int top_k,
                         cudaStream_t stream) {
    return score_logprobs_batch(logits, std::span<const TokenId>(&token, 1), domain, top_k, stream).front();
}
} // namespace sinfer::ops
