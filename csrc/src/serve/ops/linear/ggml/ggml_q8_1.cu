// Ported from llama.cpp (ggml/src/ggml-cuda), MIT License, Copyright (c) 2023-2026 The ggml
// authors. The block layouts and the vec-dot arithmetic are kept verbatim so that a GGUF's
// K-quant tensors are served as the bytes the file holds; only the surrounding names change.
#include "ops/linear/ggml/ggml_q8_1.h"

#include "ops/common/warp.cuh"

#include <stdexcept>

namespace sinfer::ops::detail::ggml {
namespace {

// One warp per 32-value block: k % 32 == 0 and the CTA is whole warps, so a warp's lanes are
// exactly one block's values and the two reductions are warp-wide.
__global__ void quantize_q8_1_kernel(const __nv_bfloat16* __restrict__ x, block_q8_1* __restrict__ y,
                                     const int k) {
    const int i0 = static_cast<int>(blockDim.x) * blockIdx.x + threadIdx.x;
    const int t  = blockIdx.y;
    if (i0 >= k) { return; }
    const float xi = __bfloat162float(x[static_cast<std::size_t>(t) * k + i0]);
    float amax = fabsf(xi);
    float sum  = xi;
    amax = warp_max(amax);
    sum  = warp_sum(sum);
    const float d  = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : static_cast<int8_t>(roundf(xi / d));
    const int ib   = i0 / QK8_1;
    const int iqs  = i0 % QK8_1;
    block_q8_1& block = y[static_cast<std::size_t>(t) * (k / QK8_1) + ib];
    block.qs[iqs] = q;
    if (iqs == 0) { block.ds = __floats2half2_rn(d, sum); }
}

} // namespace

void quantize_q8_1_launch(const __nv_bfloat16* x, std::int32_t k, std::int32_t tokens,
                          block_q8_1* out, cudaStream_t stream) {
    if (x == nullptr || out == nullptr || k <= 0 || (k % QK8_1) != 0 || tokens <= 0) {
        throw std::invalid_argument("quantize_q8_1: x [k, tokens] with k a multiple of 32");
    }
    constexpr int kThreads = 256;
    const dim3 grid((k + kThreads - 1) / kThreads, tokens);
    quantize_q8_1_kernel<<<grid, kThreads, 0, stream>>>(x, out, k);
}

} // namespace sinfer::ops::detail::ggml
