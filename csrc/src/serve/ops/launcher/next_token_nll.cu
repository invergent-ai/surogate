#include "ops/launcher/next_token_nll.h"

#include "core/device.h"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cmath>

namespace sinfer::ops::detail {
namespace {

constexpr int kThreads = 512;

/// One block per column: a max pass and a sum pass over the vocabulary, each reduced within
/// warps and then across them through shared memory.
__global__ void __launch_bounds__(kThreads)
next_token_nll_kernel(const __nv_bfloat16* __restrict__ logits, const int* __restrict__ targets,
                      float* __restrict__ out, int* __restrict__ argmax, const int vocab,
                      const int domain) {
    constexpr int kWarps = kThreads / 32;
    __shared__ float partial[kWarps];
    __shared__ int partial_index[kWarps];
    const int column          = static_cast<int>(blockIdx.x);
    const __nv_bfloat16* row  = logits + static_cast<std::size_t>(column) * vocab;
    const int tid             = static_cast<int>(threadIdx.x);
    const int warp            = tid >> 5;
    const int lane            = tid & 31;

    float local_max = -INFINITY;
    int local_index = 0;
    for (int v = tid; v < domain; v += kThreads) {
        const float value = __bfloat162float(row[v]);
        if (value > local_max) { local_max = value; local_index = v; }
    }
    const float warp_best = warp_max(local_max);
    // The lowest index among the lanes holding the warp's maximum, so ties resolve the same
    // way every run.
    const unsigned holders = __ballot_sync(0xffffffffu, local_max == warp_best);
    const int leader        = __ffs(static_cast<int>(holders)) - 1;
    const int warp_index    = __shfl_sync(0xffffffffu, local_index, leader);
    if (lane == 0) { partial[warp] = warp_best; partial_index[warp] = warp_index; }
    __syncthreads();
    float block_max = -INFINITY;
    int block_index = 0;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
        if (partial[w] > block_max) { block_max = partial[w]; block_index = partial_index[w]; }
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int v = tid; v < domain; v += kThreads) {
        local_sum += expf(__bfloat162float(row[v]) - block_max);
    }
    local_sum = warp_sum(local_sum);
    if (lane == 0) { partial[warp] = local_sum; }
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) { total += partial[w]; }
        const int target         = targets[column];
        const float target_logit = target >= 0 && target < domain
                                       ? __bfloat162float(row[target])
                                       : NAN;
        out[column] = (block_max + logf(total)) - target_logit;
        if (argmax != nullptr) { argmax[column] = block_index; }
    }
}

} // namespace

void next_token_nll_launch(const Tensor& logits, const Tensor& targets, Tensor& out,
                           Tensor* argmax, std::int32_t token_domain, cudaStream_t stream) {
    const int vocab   = logits.ne[0];
    const int columns = logits.ne[1];
    next_token_nll_kernel<<<static_cast<unsigned>(columns), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), static_cast<const int*>(targets.data),
        static_cast<float*>(out.data), argmax != nullptr ? static_cast<int*>(argmax->data) : nullptr,
        vocab, static_cast<int>(token_domain));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail
