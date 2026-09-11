#include "family/gemma_vision_ops.h"
#include "core/device.h"
#include <cuda_bf16.h>
#include <cmath>
#include <stdexcept>

namespace sinfer::family::gemma_vision {
namespace {
using BF = __nv_bfloat16;

__global__ void position_kernel(const BF* table, const int* pos, BF* x, int h, int patches,
                                int axis_size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h * patches) { return; }
    const int p = i / h, d = i % h;
    const int y = pos[p], px = pos[patches + p];
    const BF value = __float2bfloat16(__bfloat162float(table[px * h + d]) +
                                      __bfloat162float(table[(axis_size + y) * h + d]));
    x[i]           = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(value));
}

__global__ void rope_kernel(const int* pos, BF* q, BF* k, int d, int heads, int patches,
                            float theta) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d / 2 * heads * patches) { return; }
    const int pair = i % (d / 2), row = i / (d / 2), p = row / heads;
    const int quarter = d / 4, axis = pair / quarter, local = pair % quarter;
    const int index   = row * d + axis * (d / 2) + local;
    const float angle = pos[(1 - axis) * patches + p] * powf(theta, -float(2 * local) / (d / 2));
    const BF c = __float2bfloat16(cosf(angle)), s = __float2bfloat16(sinf(angle));
    for (BF* data : {q, k}) {
        const BF a = data[index], b = data[index + quarter];
        const BF ac           = __float2bfloat16(__bfloat162float(a) * __bfloat162float(c));
        const BF bs           = __float2bfloat16(__bfloat162float(b) * __bfloat162float(s));
        const BF bc           = __float2bfloat16(__bfloat162float(b) * __bfloat162float(c));
        const BF as           = __float2bfloat16(__bfloat162float(a) * __bfloat162float(s));
        data[index]           = __float2bfloat16(__bfloat162float(ac) - __bfloat162float(bs));
        data[index + quarter] = __float2bfloat16(__bfloat162float(bc) + __bfloat162float(as));
    }
}

__global__ void pool_kernel(const BF* x, int h, int tokens, int merge, float scale,
                            const float* bias, const float* gain, BF* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h * tokens) { return; }
    const int t = i / h, d = i % h;
    float value = 0;
    for (int j = 0; j < merge; ++j) { value += __bfloat162float(x[(t * merge + j) * h + d]); }
    value = __bfloat162float(__float2bfloat16(value / merge)) * scale;
    if (bias) { value = (value - bias[d]) * gain[d]; }
    out[i] = __float2bfloat16(value);
}

__global__ void clamp_kernel(const BF* x, const float* bounds, BF* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2bfloat16(fminf(fmaxf(__bfloat162float(x[i]), bounds[0]), bounds[1]));
    }
}

void matrix(const Tensor& x) {
    if (x.dtype != DType::BF16 || !x.data || !x.is_contiguous() || x.ne[0] <= 0 || x.ne[1] <= 0) {
        throw std::invalid_argument("Gemma vision requires contiguous BF16 matrices");
    }
}
} // namespace

void position_add(const Tensor& table, const Tensor& positions, Tensor& x, cudaStream_t stream) {
    matrix(x);
    matrix(table);
    if (table.ne[0] != x.ne[0] || table.ne[1] % 2 || positions.dtype != DType::I32 ||
        positions.ne[0] != x.ne[1] || positions.ne[1] != 2 || !positions.is_contiguous()) {
        throw std::invalid_argument("Gemma vision position shape mismatch");
    }
    position_kernel<<<(x.ne[0] * x.ne[1] + 255) / 256, 256, 0, stream>>>(
        (const BF*)table.data, (const int*)positions.data, (BF*)x.data, x.ne[0], x.ne[1],
        table.ne[1] / 2);
    CUDA_CHECK(cudaGetLastError());
}

void spatial_rope(const Tensor& positions, float theta, Tensor& q, Tensor& k, cudaStream_t stream) {
    matrix(q);
    matrix(k);
    if (q.ne[0] % 4 || q.bytes() != k.bytes() || positions.ne[0] != q.ne[2] ||
        positions.ne[1] != 2 || positions.dtype != DType::I32 || !std::isfinite(theta) ||
        theta <= 0) {
        throw std::invalid_argument("Gemma vision rotary shape mismatch");
    }
    const int n = q.ne[0] / 2 * q.ne[1] * q.ne[2];
    rope_kernel<<<(n + 255) / 256, 256, 0, stream>>>((const int*)positions.data, (BF*)q.data,
                                                     (BF*)k.data, q.ne[0], q.ne[1], q.ne[2], theta);
    CUDA_CHECK(cudaGetLastError());
}

void pool(const Tensor& x, int merge_unit, float scale, const Tensor* bias, const Tensor* gain,
          Tensor& out, cudaStream_t stream) {
    matrix(x);
    matrix(out);
    if (merge_unit <= 0 || x.ne[0] != out.ne[0] || x.ne[1] != out.ne[1] * merge_unit ||
        bool(bias) != bool(gain) ||
        (bias && (bias->dtype != DType::FP32 || gain->dtype != DType::FP32 ||
                  bias->ne[0] != x.ne[0] || gain->ne[0] != x.ne[0]))) {
        throw std::invalid_argument("Gemma vision pooling shape mismatch");
    }
    const int n = out.ne[0] * out.ne[1];
    pool_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        (const BF*)x.data, x.ne[0], out.ne[1], merge_unit, scale,
        bias ? (const float*)bias->data : nullptr, gain ? (const float*)gain->data : nullptr,
        (BF*)out.data);
    CUDA_CHECK(cudaGetLastError());
}

void clamp(const Tensor& x, const Tensor& bounds, int offset, Tensor& out, cudaStream_t stream) {
    matrix(x);
    matrix(out);
    if (x.bytes() != out.bytes() || bounds.dtype != DType::FP32 || bounds.ne[0] != 4 ||
        (offset != 0 && offset != 2)) {
        throw std::invalid_argument("Gemma vision clamp shape mismatch");
    }
    const int n = x.bytes() / 2;
    clamp_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        (const BF*)x.data, (const float*)bounds.data + offset, (BF*)out.data, n);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace sinfer::family::gemma_vision
