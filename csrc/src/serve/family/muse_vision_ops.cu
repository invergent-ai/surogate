#include "core/tensor.h"
#include "core/device.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cuda_bf16.h>

namespace sinfer::family {
namespace {
__global__ void pixel_shuffle(const __nv_bfloat16* input, __nv_bfloat16* output, int h, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int token = i / (h * 4), feature = i % (h * 4);
        output[i] = input[(token * 4 + feature % 4) * h + feature / 4];
    }
}
} // namespace

void muse_pixel_shuffle(const Tensor& input, Tensor& output, cudaStream_t stream) {
    if (input.dtype != DType::BF16 || output.dtype != DType::BF16 || input.ne[1] % 4 ||
        output.ne[0] != input.ne[0] * 4 || output.ne[1] != input.ne[1] / 4 ||
        !input.is_contiguous() || !output.is_contiguous()) {
        throw std::invalid_argument("invalid Muse-Glimmer pixel shuffle dimensions");
    }
    const int n = input.ne[0] * input.ne[1];
    pixel_shuffle<<<(n + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data), static_cast<__nv_bfloat16*>(output.data),
        input.ne[0], n);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace sinfer::family

namespace sinfer::family {
namespace {
__global__ void muse_rope(const int* positions, __nv_bfloat16* q, __nv_bfloat16* k, int dim,
                          int heads, int tokens, float theta) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * heads * (dim / 2)) { return; }
    const int pair = i % (dim / 2), th = i / (dim / 2), token = th / heads;
    const int axis = pair / (dim / 4), freq = pair % (dim / 4);
    const float angle = positions[axis * tokens + token] * powf(theta, -2.F * freq / (dim / 2));
    // The checkpoint computes these tables in FP32, then stores them in BF16.
    const float c = __bfloat162float(__float2bfloat16_rn(cosf(angle)));
    const float s = __bfloat162float(__float2bfloat16_rn(sinf(angle)));
    const int at = th * dim + pair, other = at + dim / 2;
    for (int which = 0; which < 2; ++which) {
        auto* x       = which ? k : q;
        const float a = __bfloat162float(x[at]), b = __bfloat162float(x[other]);
        x[at]    = __float2bfloat16_rn(__fadd_rn(__fmul_rn(a, c), -__fmul_rn(b, s)));
        x[other] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s)));
    }
}
} // namespace

void muse_vision_rope(const Tensor& positions, float theta, Tensor& q, Tensor& k,
                      cudaStream_t stream) {
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16 || !std::equal(q.ne, q.ne + 4, k.ne) ||
        q.ne[0] % 4 || q.ne[0] <= 0 || q.ne[1] <= 0 || q.ne[2] <= 0 || q.ne[3] != 1 ||
        !q.is_contiguous() || !k.is_contiguous() || positions.dtype != DType::I32 ||
        positions.ne[0] != q.ne[2] || positions.ne[1] != 2 || positions.ne[2] != 1 ||
        positions.ne[3] != 1 || !positions.is_contiguous() || !(theta > 0) ||
        !std::isfinite(theta)) {
        throw std::invalid_argument("invalid Muse-Glimmer vision RoPE geometry");
    }
    const int n = q.ne[0] / 2 * q.ne[1] * q.ne[2];
    muse_rope<<<(n + 255) / 256, 256, 0, stream>>>(
        static_cast<const int*>(positions.data), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data), q.ne[0], q.ne[1], q.ne[2], theta);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace sinfer::family
