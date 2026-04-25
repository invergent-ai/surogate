// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Fused GELU-GLU kernel: h = gelu_tanh(gate) * up.
//
// Operates on two separate rank-N tensors `gate` and `up` (same shape) and
// produces a single output `h`. Matches HuggingFace `gelu_pytorch_tanh` for
// the gate activation, so this is a drop-in replacement for:
//     gate_act = gelu(gate)
//     h        = gate_act * up
// in the non-`fuse_gate_up` MLP path. Saves two kernel launches per MLP per
// direction and one HBM round-trip for the intermediate `gate_act` buffer
// that we no longer need to materialize.
//
// Mirrors Unsloth's geglu_approx_{forward,backward}_kernel from
// study/unsloth/unsloth/kernels/geglu.py (tanh-approx variant — exact-erf is
// not needed for Gemma4 which uses gelu_pytorch_tanh).

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "kernels.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

namespace {

// GELU tanh approximation (matches HuggingFace gelu_pytorch_tanh). Duplicated
// from activation.cu's anonymous namespace rather than exposing that file's
// helpers through a header — these are 10 lines and go out of sync rarely.
__device__ __forceinline__ float gelu_tanh_fp32(float x) {
    constexpr float k0 = 0.7978845608f;  // sqrt(2/pi)
    constexpr float k1 = 0.044715f;
    float x3 = x * x * x;
    float tanh_out = tanhf(k0 * (x + k1 * x3));
    return 0.5f * x * (1.0f + tanh_out);
}

// Derivative of gelu_tanh, reusing the inner tanh value.
__device__ __forceinline__ float gelu_tanh_grad_fp32(float x) {
    constexpr float k0 = 0.7978845608f;
    constexpr float k1 = 0.044715f;
    float x2 = x * x;
    float x3 = x2 * x;
    float u = k0 * (x + k1 * x3);
    float t = tanhf(u);
    float dt = (1.0f - t * t) * k0 * (1.0f + 3.0f * k1 * x2);
    return 0.5f * (1.0f + t) + 0.5f * x * dt;
}

// ---- Forward ---------------------------------------------------------------
//
//  out[i] = gelu_tanh(gate[i]) * up[i]
//
// Vectorized with x128 loads/stores (8-way for bf16/fp16, 4-way for fp32).
// Grid: 1D; thread strided over elements.
template <typename floatX>
__global__ void gelu_glu_forward_kernel(floatX* __restrict__ out,
                                        const floatX* __restrict__ gate,
                                        const floatX* __restrict__ up,
                                        long n) {
    using x128 = GenericVector<floatX, 16 / sizeof(floatX)>;
    long idx = (long)(blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx + x128::size <= n) {
        x128 g = x128::load_cs(gate + idx);
        x128 u = x128::load_cs(up + idx);
        x128 o;
        for (int k = 0; k < x128::size; ++k) {
            float gf = (float)g[k];
            float uf = (float)u[k];
            o[k] = (floatX)(gelu_tanh_fp32(gf) * uf);
        }
        o.store(out + idx);
    } else if (idx < n) {
        // Tail: elementwise (MoE/EP dynamic batches can be unaligned).
        for (long j = idx; j < n; ++j) {
            float gf = (float)gate[j];
            float uf = (float)up[j];
            out[j] = (floatX)(gelu_tanh_fp32(gf) * uf);
        }
    }
}

// ---- Backward --------------------------------------------------------------
//
//  Given dout = dL/dh where h = gelu(gate) * up:
//     d_gate = dout * up * gelu_grad(gate)
//     d_up   = dout * gelu(gate)
//
// Single pass reads gate/up/dout and writes both grads. When the caller reuses
// the gate buffer for its own gradient (the common MLP backward layout), pass
// the same pointer for `gate` and `d_gate` — the kernel loads `gate` before
// writing so in-place is safe.
template <typename floatX>
__global__ void gelu_glu_backward_kernel(floatX* __restrict__ d_gate,
                                         floatX* __restrict__ d_up,
                                         const floatX* __restrict__ dout,
                                         const floatX* __restrict__ gate,
                                         const floatX* __restrict__ up,
                                         long n) {
    using x128 = GenericVector<floatX, 16 / sizeof(floatX)>;
    long idx = (long)(blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx + x128::size <= n) {
        x128 dO = x128::load_cs(dout + idx);
        x128 G = x128::load_cs(gate + idx);
        x128 U = x128::load_cs(up + idx);
        x128 dG, dU;
        for (int k = 0; k < x128::size; ++k) {
            float g = (float)G[k];
            float u = (float)U[k];
            float d = (float)dO[k];
            float fg = gelu_tanh_fp32(g);
            float dfg = gelu_tanh_grad_fp32(g);
            dG[k] = (floatX)(d * u * dfg);
            dU[k] = (floatX)(d * fg);
        }
        dG.store(d_gate + idx);
        dU.store(d_up + idx);
    } else if (idx < n) {
        for (long j = idx; j < n; ++j) {
            float g = (float)gate[j];
            float u = (float)up[j];
            float d = (float)dout[j];
            float fg = gelu_tanh_fp32(g);
            float dfg = gelu_tanh_grad_fp32(g);
            d_gate[j] = (floatX)(d * u * dfg);
            d_up[j] = (floatX)(d * fg);
        }
    }
}

// ---- Launchers -------------------------------------------------------------

template <typename floatX>
void gelu_glu_forward_imp(floatX* out, const floatX* gate, const floatX* up, long n, cudaStream_t stream) {
    if (n <= 0) return;
    using x128 = GenericVector<floatX, 16 / sizeof(floatX)>;
    const int block_size = 256;
    const long grid_size = (n + (long)block_size * x128::size - 1) / ((long)block_size * x128::size);
    gelu_glu_forward_kernel<floatX><<<(int)grid_size, block_size, 0, stream>>>(out, gate, up, n);
    CUDA_CHECK(cudaGetLastError());
}

template <typename floatX>
void gelu_glu_backward_imp(floatX* d_gate,
                           floatX* d_up,
                           const floatX* dout,
                           const floatX* gate,
                           const floatX* up,
                           long n,
                           cudaStream_t stream) {
    if (n <= 0) return;
    using x128 = GenericVector<floatX, 16 / sizeof(floatX)>;
    const int block_size = 256;
    const long grid_size = (n + (long)block_size * x128::size - 1) / ((long)block_size * x128::size);
    gelu_glu_backward_kernel<floatX><<<(int)grid_size, block_size, 0, stream>>>(d_gate, d_up, dout, gate, up, n);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

// ---- Public API ------------------------------------------------------------

void gelu_glu_forward(nv_bfloat16* out, const nv_bfloat16* gate, const nv_bfloat16* up, long n, cudaStream_t stream) {
    gelu_glu_forward_imp(out, gate, up, n, stream);
}

void gelu_glu_forward(float* out, const float* gate, const float* up, long n, cudaStream_t stream) {
    gelu_glu_forward_imp(out, gate, up, n, stream);
}

void gelu_glu_backward(nv_bfloat16* d_gate,
                       nv_bfloat16* d_up,
                       const nv_bfloat16* dout,
                       const nv_bfloat16* gate,
                       const nv_bfloat16* up,
                       long n,
                       cudaStream_t stream) {
    gelu_glu_backward_imp(d_gate, d_up, dout, gate, up, n, stream);
}

void gelu_glu_backward(float* d_gate,
                       float* d_up,
                       const float* dout,
                       const float* gate,
                       const float* up,
                       long n,
                       cudaStream_t stream) {
    gelu_glu_backward_imp(d_gate, d_up, dout, gate, up, n, stream);
}
