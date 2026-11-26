// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Inkling-2 trainer: the routed experts' backward for one group of experts, rows expert-sorted.
//
// The function differentiated (the trainer's reference, per row = one (token, expert) assignment with weight gamma):
//   gu  = bf16((x @ U13^T) * f13 + s * (x @ A13^T) @ B13^T)          [2I]  interleaved: 2j gate, 2j+1 up
//   y   = bf16(((4 tanh(g / 4) * sigmoid(g)) * (25 tanh(u / 25))) * gamma) [I]
//   dn  = bf16((y @ U2^T) * f2 + s * (y @ A2^T) @ B2^T)              [D]
//   moe_out[token] = bf16(sum over the token's rows of dn)
// U13 / U2 are the expert's exactly decoded weights (bf16 units), f13 / f2 its global scale (1 for MXFP8), A / B the
// LoRA work values (fp32), s = alpha / rank. The backward goes straight through every bf16 rounding. The cotangent
// that enters a unit GEMM is first scaled by f and rounded to bf16 (tensor-core GEMM with fp32 accumulation and
// output), as the reference backward does; the LoRA paths stay fp32.
//
// Nothing here allocates device memory: the caller passes a workspace (moe_group_bwd_workspace_bytes).

#pragma once

#include <cstddef>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace inkling {

struct MoeGroupBwdArgs {
    // the sequence
    const nv_bfloat16* x = nullptr;  // [T, D] MLP input (bf16 values)
    const float* g = nullptr;        // [T, D] cotangent of the MoE output
    int T = 0, D = 0, I = 0;         // D = 2304, I = 1152
    // the group's rows, sorted by expert
    const int* tok = nullptr;    // [R] device: token of each row
    const float* gam = nullptr;  // [R] device: routing weight of each row
    const int* seg = nullptr;    // [ng + 1] HOST: row offsets of the group's experts in tok / gam
    int ng = 0;
    // the experts' weights
    const nv_bfloat16* u13 = nullptr;  // [ng, 2I, D]
    const nv_bfloat16* u2 = nullptr;   // [ng, D, I]
    const float* f13 = nullptr;        // [ng] HOST
    const float* f2 = nullptr;         // [ng] HOST
    // LoRA (all null = no adapter)
    const float* A13 = nullptr;  // [ng, r, D]
    const float* B13 = nullptr;  // [ng, 2I, r]
    const float* A2 = nullptr;   // [ng, r, I]
    const float* B2 = nullptr;   // [ng, D, r]
    float scale = 0.0f;
    int rank = 0;
    // outputs
    float* dx = nullptr;    // [T, D] accumulated (+=)
    float* dgam = nullptr;  // [R] written
    float* dA13 = nullptr;  // accumulated; each may be null = not needed
    float* dB13 = nullptr;
    float* dA2 = nullptr;
    float* dB2 = nullptr;
    // rows of one expert handled at a time (bounds the workspace); 0 = 8192
    int max_rows = 0;
    void* workspace = nullptr;
    std::size_t workspace_bytes = 0;
};

/// Workspace for a call with at most `rows` rows of one expert handled at a time.
std::size_t moe_group_bwd_workspace_bytes(int rows, int D, int I, int rank);

/// Throws std::runtime_error on a CUDA / cuBLAS error or a short workspace.
void moe_group_bwd(const MoeGroupBwdArgs& a, cudaStream_t stream);

}  // namespace inkling
