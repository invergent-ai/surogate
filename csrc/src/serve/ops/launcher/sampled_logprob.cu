#include "ops/launcher/sampled_logprob.h"

#include "core/device.h"
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
