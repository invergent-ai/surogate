// The LoopedSharedExpert of an Inkling-2 decision adapter (System 1.5 `modeling.py`), elementwise parts, with the
// reference's BF16 roundings (each torch op on BF16 tensors rounds its result once; checked against torch 2.11 CUDA):
//   x   = bf16(bf16(rms_norm(h) * bf16(1 + scale)) + shift),  rms_norm = bf16(h * rsqrt(mean(h^2) + eps) * w) in fp32
//   act = bf16(bf16(silu(g)) * u)
//   h'  = bf16(h + bf16(bf16(1 + gate) * out))
// The projections between them are the engine's invariant BF16 GEMM (fp32 accumulation, one rounding).

#include "targets/inkling2/impl/decision/kernels.h"

#include "core/device.h"

#include <cuda_bf16.h>

namespace sinfer::targets::inkling2::detail::decision {

namespace {

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

/// One block per column: the RMS of the column of `h` (stride `h_column` elements), then the modulated input.
__global__ void modulate_kernel(const __nv_bfloat16* __restrict__ h, std::int64_t h_column,
                                const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ scale,
                                const __nv_bfloat16* __restrict__ shift, __nv_bfloat16* __restrict__ out, int d,
                                float eps) {
    const int t = blockIdx.x;
    const __nv_bfloat16* col = h + static_cast<std::int64_t>(t) * h_column;
    float ss = 0.0F;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float v = __bfloat162float(col[i]);
        ss += v * v;
    }
    __shared__ float part[32];
    for (int o = 16; o > 0; o >>= 1) { ss += __shfl_xor_sync(0xffffffffU, ss, o); }
    if ((threadIdx.x & 31) == 0) { part[threadIdx.x >> 5] = ss; }
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? part[threadIdx.x] : 0.0F;
        for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffffU, v, o); }
        if (threadIdx.x == 0) { part[0] = v; }
    }
    __syncthreads();
    const float inv = rsqrtf(part[0] / static_cast<float>(d) + eps);
    __nv_bfloat16* o = out + static_cast<std::int64_t>(t) * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float y  = bf(__bfloat162float(col[i]) * inv * __bfloat162float(w[i]));
        const float s1 = bf(1.0F + __bfloat162float(scale[i]));
        o[i]           = __float2bfloat16_rn(bf(y * s1) + __bfloat162float(shift[i]));
    }
}

/// gate_up [2F, T] (rows [0, F) gate, [F, 2F) up) -> act [F, T].
__global__ void swiglu_kernel(const __nv_bfloat16* __restrict__ gate_up, __nv_bfloat16* __restrict__ act, int f,
                              std::int64_t n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    const std::int64_t t = i / f, j = i % f;
    const float g = __bfloat162float(gate_up[t * 2 * f + j]);
    const float u = __bfloat162float(gate_up[t * 2 * f + f + j]);
    act[i]        = __float2bfloat16_rn(bf(g / (1.0F + __expf(-g))) * u);
}

/// h[:, t] (stride h_column) += bf16(bf16(1 + gate) * out[:, t]).
__global__ void residual_kernel(__nv_bfloat16* __restrict__ h, std::int64_t h_column,
                                const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ out, int d,
                                std::int64_t n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    const std::int64_t t = i / d, j = i % d;
    __nv_bfloat16* p = h + t * h_column + j;
    const float g1   = bf(1.0F + __bfloat162float(gate[j]));
    *p               = __float2bfloat16_rn(__bfloat162float(*p) + bf(g1 * __bfloat162float(out[i])));
}

} // namespace

void shared_expert_modulate(const void* h, std::int64_t h_column, const void* norm_weight, const void* scale,
                            const void* shift, void* out, int d, int tokens, float eps, cudaStream_t s) {
    modulate_kernel<<<tokens, 256, 0, s>>>(static_cast<const __nv_bfloat16*>(h), h_column,
                                           static_cast<const __nv_bfloat16*>(norm_weight),
                                           static_cast<const __nv_bfloat16*>(scale),
                                           static_cast<const __nv_bfloat16*>(shift), static_cast<__nv_bfloat16*>(out),
                                           d, eps);
    CUDA_CHECK(cudaGetLastError());
}

void shared_expert_swiglu(const void* gate_up, void* act, int f, int tokens, cudaStream_t s) {
    const std::int64_t n = static_cast<std::int64_t>(f) * tokens;
    swiglu_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(static_cast<const __nv_bfloat16*>(gate_up),
                                                                        static_cast<__nv_bfloat16*>(act), f, n);
    CUDA_CHECK(cudaGetLastError());
}

void shared_expert_residual(void* h, std::int64_t h_column, const void* gate, const void* out, int d, int tokens,
                            cudaStream_t s) {
    const std::int64_t n = static_cast<std::int64_t>(d) * tokens;
    residual_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(
        static_cast<__nv_bfloat16*>(h), h_column, static_cast<const __nv_bfloat16*>(gate),
        static_cast<const __nv_bfloat16*>(out), d, n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::targets::inkling2::detail::decision
