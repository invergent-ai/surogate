// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file fast_lora_silu.cu
 * @brief CUDA kernels for fast LoRA fusion in MoE experts.
 *
 * Provides optimized kernels for the SiLU-based activation used in MoE experts
 * with LoRA, enabling in-place backward computation to reduce memory footprint.
 *
 * Key insight: By storing only e (gate output) and g (up output) during forward,
 * we can reconstruct h = silu(e) * g on-the-fly and compute gradients in-place,
 * saving one activation tensor (N x D) per expert.
 *
 * Forward:  h = silu(e) * g = e * sigmoid(e) * g
 * Backward: Given dh,
 *   de = dh * g * sigmoid(e) * (1 + e * (1 - sigmoid(e)))
 *   dg = dh * silu(e) = dh * e * sigmoid(e)
 */

#include <cassert>

#include <cuda_bf16.h>

#include "kernels.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"
#include "utilities/vec.cuh"
#include "kernel_utils.cuh"

// ============================================================================
// SiLU forward from separate e, g tensors
// ============================================================================

/**
 * @brief Compute h = silu(e) * g from separate gate and up outputs.
 *
 * Unlike the standard swiglu_forward which takes concatenated [up, gate],
 * this kernel takes separate e (gate output) and g (up output) tensors.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] h Output tensor (N, D).
 * @param[in] e Gate projection output (N, D).
 * @param[in] g Up projection output (N, D).
 * @param N Number of tokens.
 * @param D Intermediate dimension.
 */
template<typename floatX>
__global__ void silu_mul_forward_kernel(floatX* __restrict__ h,
                                        const floatX* __restrict__ e,
                                        const floatX* __restrict__ g,
                                        int N, int D) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;

    const int total = N * D;
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx >= total) return;

    x128 e_vec = x128::load_cs(e + idx);
    x128 g_vec = x128::load_cs(g + idx);
    x128 h_vec;

    for (int k = 0; k < x128::size; ++k) {
        float e_val = (float)e_vec[k];
        float g_val = (float)g_vec[k];
        // h = silu(e) * g = e * sigmoid(e) * g
        float sig_e = 1.0f / (1.0f + expf(-e_val));
        h_vec[k] = (floatX)(e_val * sig_e * g_val);
    }

    h_vec.store(h + idx);
}

// ============================================================================
// In-place SiLU backward for fast LoRA
// ============================================================================

/**
 * @brief In-place backward through SiLU multiplication.
 *
 * This kernel is the core of the fast LoRA optimization. Given the stored
 * e (gate output) and g (up output), it:
 * 1. Computes h = silu(e) * g on-the-fly (for down LoRA gradient computation)
 * 2. Computes de and dg from dh
 * 3. Overwrites e with de and g with dg IN-PLACE
 *
 * This eliminates the need to store h during forward, saving N*D elements.
 *
 * Math:
 *   h = silu(e) * g = e * sigmoid(e) * g
 *   de = dh * g * d(silu(e))/de = dh * g * sigmoid(e) * (1 + e * (1 - sigmoid(e)))
 *   dg = dh * silu(e) = dh * e * sigmoid(e)
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[in,out] e Gate output (N, D) -> becomes de after call.
 * @param[in,out] g Up output (N, D) -> becomes dg after call.
 * @param[in] dh Upstream gradient (N, D).
 * @param[out] h_out Optional: reconstructed h for down LoRA backward (can be nullptr).
 * @param N Number of tokens.
 * @param D Intermediate dimension.
 */
template<typename floatX>
__global__ void silu_mul_backward_inplace_kernel(floatX* __restrict__ e,
                                                  floatX* __restrict__ g,
                                                  const floatX* __restrict__ dh,
                                                  floatX* __restrict__ h_out,
                                                  int N, int D) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;

    const int total = N * D;
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx >= total) return;

    x128 e_vec = x128::load(e + idx);
    x128 g_vec = x128::load(g + idx);
    x128 dh_vec = x128::load_cs(dh + idx);

    x128 de_vec, dg_vec, h_vec;

    for (int k = 0; k < x128::size; ++k) {
        float e_val = (float)e_vec[k];
        float g_val = (float)g_vec[k];
        float dh_val = (float)dh_vec[k];

        // Compute sigmoid(e)
        float sig_e = 1.0f / (1.0f + expf(-e_val));
        // silu(e) = e * sigmoid(e)
        float silu_e = e_val * sig_e;

        // Reconstruct h = silu(e) * g
        float h_val = silu_e * g_val;

        // d(silu(e))/de = sigmoid(e) * (1 + e * (1 - sigmoid(e)))
        //               = sigmoid(e) + e * sigmoid(e) * (1 - sigmoid(e))
        //               = sigmoid(e) * (1 + e - e * sigmoid(e))
        float dsilu_de = sig_e * (1.0f + e_val * (1.0f - sig_e));

        // de = dh * g * d(silu(e))/de
        float de_val = dh_val * g_val * dsilu_de;

        // dg = dh * silu(e)
        float dg_val = dh_val * silu_e;

        de_vec[k] = (floatX)de_val;
        dg_vec[k] = (floatX)dg_val;
        h_vec[k] = (floatX)h_val;
    }

    // Overwrite e with de, g with dg (in-place)
    de_vec.store(e + idx);
    dg_vec.store(g + idx);

    // Optionally output h for down LoRA gradient computation
    if (h_out) {
        h_vec.store(h_out + idx);
    }
}

/**
 * @brief Fused backward that also accumulates dx contributions.
 *
 * Extended version that computes de, dg AND accumulates their contributions
 * to dx in a single pass, reducing memory traffic further.
 *
 * This is useful when we want to compute:
 *   dx += de @ W_gate^T + dg @ W_up^T
 * but we do it element-wise here for the LoRA part (since LoRA is low-rank).
 *
 * For now, we keep this simpler and do the matmuls separately.
 */

// ============================================================================
// Split and concatenate utilities for gate_up tensor
// ============================================================================

/**
 * @brief Split a (N, 2D) tensor into two (N, D) tensors.
 *
 * Splits gate_up = [up | gate] into separate up and gate tensors.
 * Layout assumption: up is columns [0, D), gate is columns [D, 2D).
 *
 * @tparam floatX Data type.
 * @param[in] gate_up Input tensor (N, 2D).
 * @param[out] up Up projection output (N, D) - first half.
 * @param[out] gate Gate projection output (N, D) - second half.
 * @param N Number of tokens.
 * @param D Intermediate dimension.
 */
template<typename floatX>
__global__ void split_gate_up_kernel(const floatX* __restrict__ gate_up,
                                     floatX* __restrict__ up,
                                     floatX* __restrict__ gate,
                                     int N, int D) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;

    // Each thread handles one vector in the output
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx >= N * D) return;

    const int row = idx / D;
    const int col = idx % D;

    // Input layout: [up (D cols) | gate (D cols)] per row
    const int up_offset = row * 2 * D + col;
    const int gate_offset = row * 2 * D + D + col;

    x128 up_vec = x128::load_cs(gate_up + up_offset);
    x128 gate_vec = x128::load_cs(gate_up + gate_offset);

    up_vec.store(up + idx);
    gate_vec.store(gate + idx);
}

/**
 * @brief Concatenate two (N, D) tensors into a (N, 2D) tensor.
 *
 * Concatenates de and dg into d_gate_up = [dg | de] for backward through gate_up.
 * Layout: dg (up gradient) is columns [0, D), de (gate gradient) is columns [D, 2D).
 *
 * @tparam floatX Data type.
 * @param[in] dg Gradient w.r.t. up output (N, D).
 * @param[in] de Gradient w.r.t. gate output (N, D).
 * @param[out] d_gate_up Output tensor (N, 2D).
 * @param N Number of tokens.
 * @param D Intermediate dimension.
 */
template<typename floatX>
__global__ void concat_d_gate_up_kernel(const floatX* __restrict__ dg,
                                        const floatX* __restrict__ de,
                                        floatX* __restrict__ d_gate_up,
                                        int N, int D) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;

    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx >= N * D) return;

    const int row = idx / D;
    const int col = idx % D;

    // Output layout: [dg (D cols) | de (D cols)] per row
    const int dg_offset = row * 2 * D + col;
    const int de_offset = row * 2 * D + D + col;

    x128 dg_vec = x128::load_cs(dg + idx);
    x128 de_vec = x128::load_cs(de + idx);

    dg_vec.store(d_gate_up + dg_offset);
    de_vec.store(d_gate_up + de_offset);
}

/**
 * @brief Split interleaved gate_up = [gate0, up0, gate1, up1, ...] into up and gate.
 */
template<typename floatX>
__global__ void split_gate_up_interleaved_kernel(const floatX* __restrict__ gate_up,
                                                 floatX* __restrict__ up,
                                                 floatX* __restrict__ gate,
                                                 int N, int D) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * D;
    if (idx >= total) return;
    const int base = idx * 2;
    gate[idx] = gate_up[base];
    up[idx] = gate_up[base + 1];
}

/**
 * @brief Add interleaved gate/up contributions into gate_up in-place.
 */
template<typename floatX>
__global__ void add_gate_up_interleaved_kernel(floatX* __restrict__ gate_up,
                                               const floatX* __restrict__ up,
                                               const floatX* __restrict__ gate,
                                               int N, int D) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * D;
    if (idx >= total) return;
    const int base = idx * 2;
    gate_up[base] += gate[idx];
    gate_up[base + 1] += up[idx];
}

template<typename floatX>
__global__ void add_gate_up_interleaved_gate_kernel(floatX* __restrict__ gate_up,
                                                    const floatX* __restrict__ gate,
                                                    int N, int D) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * D;
    if (idx >= total) return;
    gate_up[idx * 2] += gate[idx];
}

template<typename floatX>
__global__ void add_gate_up_interleaved_up_kernel(floatX* __restrict__ gate_up,
                                                  const floatX* __restrict__ up,
                                                  int N, int D) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * D;
    if (idx >= total) return;
    gate_up[idx * 2 + 1] += up[idx];
}

// ============================================================================
// Kernel launchers
// ============================================================================

template<typename floatX>
void silu_mul_forward_impl(floatX* h, const floatX* e, const floatX* g,
                           int N, int D, cudaStream_t stream) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;

    const int total = N * D;
    const int block_size = 256;
    assert(total % x128::size == 0);
    const int grid_size = div_ceil(total, static_cast<int>(block_size * x128::size));

    silu_mul_forward_kernel<<<grid_size, block_size, 0, stream>>>(h, e, g, N, D);
    CUDA_CHECK(cudaGetLastError());
}

void silu_mul_forward(nv_bfloat16* h, const nv_bfloat16* e, const nv_bfloat16* g,
                      int N, int D, cudaStream_t stream) {
    silu_mul_forward_impl(h, e, g, N, D, stream);
}

void silu_mul_forward(float* h, const float* e, const float* g,
                      int N, int D, cudaStream_t stream) {
    silu_mul_forward_impl(h, e, g, N, D, stream);
}

template<typename floatX>
void silu_mul_backward_inplace_impl(floatX* e, floatX* g, const floatX* dh, floatX* h_out,
                                     int N, int D, cudaStream_t stream) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;

    const int total = N * D;
    const int block_size = 256;
    assert(total % x128::size == 0);
    const int grid_size = div_ceil(total, static_cast<int>(block_size * x128::size));

    silu_mul_backward_inplace_kernel<<<grid_size, block_size, 0, stream>>>(e, g, dh, h_out, N, D);
    CUDA_CHECK(cudaGetLastError());
}

void silu_mul_backward_inplace(nv_bfloat16* e, nv_bfloat16* g, const nv_bfloat16* dh,
                                nv_bfloat16* h_out, int N, int D, cudaStream_t stream) {
    silu_mul_backward_inplace_impl(e, g, dh, h_out, N, D, stream);
}

void silu_mul_backward_inplace(float* e, float* g, const float* dh,
                                float* h_out, int N, int D, cudaStream_t stream) {
    silu_mul_backward_inplace_impl(e, g, dh, h_out, N, D, stream);
}

template<typename floatX>
void split_gate_up_impl(const floatX* gate_up, floatX* up, floatX* gate,
                        int N, int D, cudaStream_t stream) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;

    const int total = N * D;
    const int block_size = 256;
    assert(D % x128::size == 0);
    const int grid_size = div_ceil(total, static_cast<int>(block_size * x128::size));

    split_gate_up_kernel<<<grid_size, block_size, 0, stream>>>(gate_up, up, gate, N, D);
    CUDA_CHECK(cudaGetLastError());
}

void split_gate_up(const nv_bfloat16* gate_up, nv_bfloat16* up, nv_bfloat16* gate,
                   int N, int D, cudaStream_t stream) {
    split_gate_up_impl(gate_up, up, gate, N, D, stream);
}

void split_gate_up(const float* gate_up, float* up, float* gate,
                   int N, int D, cudaStream_t stream) {
    split_gate_up_impl(gate_up, up, gate, N, D, stream);
}

template<typename floatX>
void concat_d_gate_up_impl(const floatX* dg, const floatX* de, floatX* d_gate_up,
                           int N, int D, cudaStream_t stream) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;

    const int total = N * D;
    const int block_size = 256;
    assert(D % x128::size == 0);
    const int grid_size = div_ceil(total, static_cast<int>(block_size * x128::size));

    concat_d_gate_up_kernel<<<grid_size, block_size, 0, stream>>>(dg, de, d_gate_up, N, D);
    CUDA_CHECK(cudaGetLastError());
}

void concat_d_gate_up(const nv_bfloat16* dg, const nv_bfloat16* de, nv_bfloat16* d_gate_up,
                      int N, int D, cudaStream_t stream) {
    concat_d_gate_up_impl(dg, de, d_gate_up, N, D, stream);
}

void concat_d_gate_up(const float* dg, const float* de, float* d_gate_up,
                      int N, int D, cudaStream_t stream) {
    concat_d_gate_up_impl(dg, de, d_gate_up, N, D, stream);
}

// Tensor overloads
void silu_mul_forward(Tensor& h, const Tensor& e, const Tensor& g,
                      int N, int D, cudaStream_t stream) {
    assert(e.DType == g.DType && e.DType == h.DType);
    if (e.DType == ETensorDType::BF16) {
        silu_mul_forward(h.get<nv_bfloat16>(), e.get<nv_bfloat16>(), g.get<nv_bfloat16>(), N, D, stream);
    } else if (e.DType == ETensorDType::FP32) {
        silu_mul_forward(h.get<float>(), e.get<float>(), g.get<float>(), N, D, stream);
    } else {
        throw std::runtime_error("silu_mul_forward: unsupported dtype");
    }
}

void silu_mul_backward_inplace(Tensor& e, Tensor& g, const Tensor& dh,
                                Tensor* h_out, int N, int D, cudaStream_t stream) {
    assert(e.DType == g.DType && e.DType == dh.DType);
    if (e.DType == ETensorDType::BF16) {
        nv_bfloat16* h_ptr = h_out ? h_out->get<nv_bfloat16>() : nullptr;
        silu_mul_backward_inplace(e.get<nv_bfloat16>(), g.get<nv_bfloat16>(),
                                   dh.get<nv_bfloat16>(), h_ptr, N, D, stream);
    } else if (e.DType == ETensorDType::FP32) {
        float* h_ptr = h_out ? h_out->get<float>() : nullptr;
        silu_mul_backward_inplace(e.get<float>(), g.get<float>(),
                                   dh.get<float>(), h_ptr, N, D, stream);
    } else {
        throw std::runtime_error("silu_mul_backward_inplace: unsupported dtype");
    }
}

void split_gate_up(const Tensor& gate_up, Tensor& up, Tensor& gate,
                   int N, int D, cudaStream_t stream) {
    assert(gate_up.DType == up.DType && gate_up.DType == gate.DType);
    if (gate_up.DType == ETensorDType::BF16) {
        split_gate_up(gate_up.get<nv_bfloat16>(), up.get<nv_bfloat16>(),
                      gate.get<nv_bfloat16>(), N, D, stream);
    } else if (gate_up.DType == ETensorDType::FP32) {
        split_gate_up(gate_up.get<float>(), up.get<float>(), gate.get<float>(), N, D, stream);
    } else {
        throw std::runtime_error("split_gate_up: unsupported dtype");
    }
}

template<typename floatX>
static void split_gate_up_interleaved_impl(const floatX* gate_up, floatX* up, floatX* gate,
                                           int N, int D, cudaStream_t stream) {
    const int total = N * D;
    if (total <= 0) return;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    split_gate_up_interleaved_kernel<<<grid_size, block_size, 0, stream>>>(
        gate_up, up, gate, N, D);
    CUDA_CHECK(cudaGetLastError());
}

void split_gate_up_interleaved(const nv_bfloat16* gate_up, nv_bfloat16* up, nv_bfloat16* gate,
                               int N, int D, cudaStream_t stream) {
    split_gate_up_interleaved_impl(gate_up, up, gate, N, D, stream);
}

void split_gate_up_interleaved(const float* gate_up, float* up, float* gate,
                               int N, int D, cudaStream_t stream) {
    split_gate_up_interleaved_impl(gate_up, up, gate, N, D, stream);
}

void split_gate_up_interleaved(const Tensor& gate_up, Tensor& up, Tensor& gate,
                               int N, int D, cudaStream_t stream) {
    if (gate_up.DType == ETensorDType::BF16) {
        split_gate_up_interleaved(gate_up.get<nv_bfloat16>(), up.get<nv_bfloat16>(),
                                  gate.get<nv_bfloat16>(), N, D, stream);
    } else if (gate_up.DType == ETensorDType::FP32) {
        split_gate_up_interleaved(gate_up.get<float>(), up.get<float>(), gate.get<float>(), N, D, stream);
    } else {
        throw std::runtime_error("split_gate_up_interleaved: unsupported dtype");
    }
}

template<typename floatX>
static void add_gate_up_interleaved_impl(floatX* gate_up, const floatX* up, const floatX* gate,
                                         int N, int D, cudaStream_t stream) {
    const int total = N * D;
    if (total <= 0) return;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    add_gate_up_interleaved_kernel<<<grid_size, block_size, 0, stream>>>(
        gate_up, up, gate, N, D);
    CUDA_CHECK(cudaGetLastError());
}

void add_gate_up_interleaved(nv_bfloat16* gate_up, const nv_bfloat16* up, const nv_bfloat16* gate,
                             int N, int D, cudaStream_t stream) {
    add_gate_up_interleaved_impl(gate_up, up, gate, N, D, stream);
}

void add_gate_up_interleaved(float* gate_up, const float* up, const float* gate,
                             int N, int D, cudaStream_t stream) {
    add_gate_up_interleaved_impl(gate_up, up, gate, N, D, stream);
}

void add_gate_up_interleaved(Tensor& gate_up, const Tensor& up, const Tensor& gate,
                             int N, int D, cudaStream_t stream) {
    if (gate_up.DType == ETensorDType::BF16) {
        add_gate_up_interleaved(gate_up.get<nv_bfloat16>(), up.get<nv_bfloat16>(),
                                gate.get<nv_bfloat16>(), N, D, stream);
    } else if (gate_up.DType == ETensorDType::FP32) {
        add_gate_up_interleaved(gate_up.get<float>(), up.get<float>(), gate.get<float>(), N, D, stream);
    } else {
        throw std::runtime_error("add_gate_up_interleaved: unsupported dtype");
    }
}

template<typename floatX>
static void add_gate_up_interleaved_gate_impl(floatX* gate_up, const floatX* gate,
                                              int N, int D, cudaStream_t stream) {
    const int total = N * D;
    if (total <= 0) return;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    add_gate_up_interleaved_gate_kernel<<<grid_size, block_size, 0, stream>>>(
        gate_up, gate, N, D);
    CUDA_CHECK(cudaGetLastError());
}

void add_gate_up_interleaved_gate(nv_bfloat16* gate_up, const nv_bfloat16* gate,
                                  int N, int D, cudaStream_t stream) {
    add_gate_up_interleaved_gate_impl(gate_up, gate, N, D, stream);
}

void add_gate_up_interleaved_gate(float* gate_up, const float* gate,
                                  int N, int D, cudaStream_t stream) {
    add_gate_up_interleaved_gate_impl(gate_up, gate, N, D, stream);
}

void add_gate_up_interleaved_gate(Tensor& gate_up, const Tensor& gate,
                                  int N, int D, cudaStream_t stream) {
    if (gate_up.DType == ETensorDType::BF16) {
        add_gate_up_interleaved_gate(gate_up.get<nv_bfloat16>(), gate.get<nv_bfloat16>(), N, D, stream);
    } else if (gate_up.DType == ETensorDType::FP32) {
        add_gate_up_interleaved_gate(gate_up.get<float>(), gate.get<float>(), N, D, stream);
    } else {
        throw std::runtime_error("add_gate_up_interleaved_gate: unsupported dtype");
    }
}

template<typename floatX>
static void add_gate_up_interleaved_up_impl(floatX* gate_up, const floatX* up,
                                            int N, int D, cudaStream_t stream) {
    const int total = N * D;
    if (total <= 0) return;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    add_gate_up_interleaved_up_kernel<<<grid_size, block_size, 0, stream>>>(
        gate_up, up, N, D);
    CUDA_CHECK(cudaGetLastError());
}

void add_gate_up_interleaved_up(nv_bfloat16* gate_up, const nv_bfloat16* up,
                                int N, int D, cudaStream_t stream) {
    add_gate_up_interleaved_up_impl(gate_up, up, N, D, stream);
}

void add_gate_up_interleaved_up(float* gate_up, const float* up,
                                int N, int D, cudaStream_t stream) {
    add_gate_up_interleaved_up_impl(gate_up, up, N, D, stream);
}

void add_gate_up_interleaved_up(Tensor& gate_up, const Tensor& up,
                                int N, int D, cudaStream_t stream) {
    if (gate_up.DType == ETensorDType::BF16) {
        add_gate_up_interleaved_up(gate_up.get<nv_bfloat16>(), up.get<nv_bfloat16>(), N, D, stream);
    } else if (gate_up.DType == ETensorDType::FP32) {
        add_gate_up_interleaved_up(gate_up.get<float>(), up.get<float>(), N, D, stream);
    } else {
        throw std::runtime_error("add_gate_up_interleaved_up: unsupported dtype");
    }
}

void concat_d_gate_up(const Tensor& dg, const Tensor& de, Tensor& d_gate_up,
                      int N, int D, cudaStream_t stream) {
    assert(dg.DType == de.DType && dg.DType == d_gate_up.DType);
    if (dg.DType == ETensorDType::BF16) {
        concat_d_gate_up(dg.get<nv_bfloat16>(), de.get<nv_bfloat16>(),
                         d_gate_up.get<nv_bfloat16>(), N, D, stream);
    } else if (dg.DType == ETensorDType::FP32) {
        concat_d_gate_up(dg.get<float>(), de.get<float>(), d_gate_up.get<float>(), N, D, stream);
    } else {
        throw std::runtime_error("concat_d_gate_up: unsupported dtype");
    }
}
