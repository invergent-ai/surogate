#include "family/inkling_vision_ops.h"
#include "core/device.h"
#include "core/tensor.h"

#include <algorithm>
#include <cuda_bf16.h>
#include <stdexcept>

namespace sinfer::family {
namespace {
__global__ void inkling_rope(const int* positions, __nv_bfloat16* q, __nv_bfloat16* k, int dim, int heads,
                             int tokens) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * heads * (dim / 2)) { return; }
    const int pair = i % (dim / 2), th = i / (dim / 2), token = th / heads;
    const int quarter = dim / 4;
    const int axis = pair < quarter ? 0 : 1, freq = pair % quarter;
    // vLLM's _rope_inv_freq: 10000 ** (-(4 j) / head_dim) in float64, rounded to float32.
    const float inv_freq = static_cast<float>(pow(10000.0, -static_cast<double>(4 * freq) / dim));
    const float angle    = __fmul_rn(static_cast<float>(positions[axis * tokens + token]), inv_freq);
    const float c = cosf(angle), s = sinf(angle);
    const int at = th * dim + 2 * pair;
    for (int which = 0; which < 2; ++which) {
        __nv_bfloat16* x = which ? k : q;
        const float even = __bfloat162float(x[at]), odd = __bfloat162float(x[at + 1]);
        x[at]     = __float2bfloat16_rn(__fmaf_rn(even, c, -__fmul_rn(odd, s)));
        x[at + 1] = __float2bfloat16_rn(__fmaf_rn(odd, c, __fmul_rn(even, s)));
    }
}

__global__ void scaled_residual(const __nv_bfloat16* update, const __nv_bfloat16* gamma, __nv_bfloat16* x, int dim,
                                long long n) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    const float scaled = __bfloat162float(
        __float2bfloat16_rn(__fmul_rn(__bfloat162float(update[i]), __bfloat162float(gamma[i % dim]))));
    x[i] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(x[i]), scaled));
}

__global__ void cls_embed(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) { out[i] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(a[i]), __bfloat162float(b[i]))); }
}

__global__ void scatter_columns(const __nv_bfloat16* src, const int* indices, __nv_bfloat16* dst, int dim, int count,
                                long long column_elements) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(dim) * count) { return; }
    const long long column = i / dim, d = i % dim;
    dst[static_cast<long long>(indices[column]) * column_elements + d] = src[i];
}

bool bf16_contiguous(const Tensor& t) { return t.dtype == DType::BF16 && t.is_contiguous() && t.data != nullptr; }
long long elements(const Tensor& t) { return static_cast<long long>(t.bytes() / sizeof(__nv_bfloat16)); }
} // namespace

void inkling_vision_rope(const Tensor& positions, Tensor& q, Tensor& k, cudaStream_t stream) {
    if (!bf16_contiguous(q) || !bf16_contiguous(k) || !std::equal(q.ne, q.ne + 4, k.ne) || q.ne[0] % 4 ||
        q.ne[0] <= 0 || q.ne[1] <= 0 || q.ne[2] <= 0 || q.ne[3] != 1 || positions.dtype != DType::I32 ||
        positions.ne[0] != q.ne[2] || positions.ne[1] != 2 || positions.ne[2] != 1 || positions.ne[3] != 1 ||
        !positions.is_contiguous()) {
        throw std::invalid_argument("invalid Inkling-2 vision RoPE geometry");
    }
    const int n = q.ne[0] / 2 * q.ne[1] * q.ne[2];
    inkling_rope<<<(n + 255) / 256, 256, 0, stream>>>(
        static_cast<const int*>(positions.data), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data), q.ne[0], q.ne[1], q.ne[2]);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_vision_scaled_residual(const Tensor& update, const Tensor& gamma, Tensor& x, cudaStream_t stream) {
    if (!bf16_contiguous(update) || !bf16_contiguous(gamma) || !bf16_contiguous(x) ||
        !std::equal(update.ne, update.ne + 4, x.ne) || gamma.ne[0] != x.ne[0] || elements(gamma) != gamma.ne[0]) {
        throw std::invalid_argument("invalid Inkling-2 vision LayerScale residual");
    }
    const long long n = elements(x);
    if (n == 0) { return; }
    scaled_residual<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(update.data), static_cast<const __nv_bfloat16*>(gamma.data),
        static_cast<__nv_bfloat16*>(x.data), x.ne[0], n);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_vision_cls(const Tensor& class_embedding, const Tensor& position, Tensor& out, cudaStream_t stream) {
    const int dim = class_embedding.ne[0];
    if (!bf16_contiguous(class_embedding) || !bf16_contiguous(position) || !bf16_contiguous(out) ||
        elements(class_embedding) != dim || elements(position) != dim || elements(out) != dim) {
        throw std::invalid_argument("invalid Inkling-2 vision CLS embedding");
    }
    cls_embed<<<(dim + 255) / 256, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(class_embedding.data),
                                                     static_cast<const __nv_bfloat16*>(position.data),
                                                     static_cast<__nv_bfloat16*>(out.data), dim);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_scatter_columns(const Tensor& src, const Tensor& indices, void* dst, std::int64_t column_elements,
                             cudaStream_t stream) {
    if (!bf16_contiguous(src) || src.ne[2] != 1 || src.ne[3] != 1 || indices.dtype != DType::I32 ||
        !indices.is_contiguous() || indices.ne[0] != src.ne[1] || dst == nullptr || column_elements < src.ne[0]) {
        throw std::invalid_argument("invalid Inkling-2 visual column scatter");
    }
    const long long n = static_cast<long long>(src.ne[0]) * src.ne[1];
    if (n == 0) { return; }
    scatter_columns<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(src.data), static_cast<const int*>(indices.data),
        static_cast<__nv_bfloat16*>(dst), src.ne[0], src.ne[1], column_elements);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace sinfer::family
