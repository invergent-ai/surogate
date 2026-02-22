// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Scatter separate dQ/dK/dV buffers into interleaved dQKV format.
// Handles GQA by reducing Hq expanded heads to Hkv KV heads.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

// Scatter dQ (contiguous) into the Q portion of interleaved dQKV.
// dQ: (total_q, Hq, HS) contiguous.
// dqkv: (total_q, H, HS) interleaved, where H = Hq + 2*Hkv.
// Each thread handles one element.
__global__ void scatter_dq_kernel(
        nv_bfloat16* __restrict__ dqkv,
        const nv_bfloat16* __restrict__ dq,
        int total_q, int Hq, int Hkv, int HS) {
    const int H = Hq + 2 * Hkv;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = total_q * Hq * HS;
    if (idx >= total) return;

    const int d = idx % HS;
    const int h = (idx / HS) % Hq;
    const int t = idx / (Hq * HS);

    dqkv[t * H * HS + h * HS + d] = dq[idx];
}

// Reduce dk_expanded (Hq heads) to Hkv KV heads and scatter into the K portion of interleaved dQKV.
// dk_expanded: (total_q, Hq, HS) contiguous — each KV head has h_ratio query-head slots.
// dqkv K section starts at offset Hq*HS from dqkv base.
// For MHA (Hq == Hkv, h_ratio == 1): this is a simple copy.
__global__ void reduce_scatter_dk_kernel(
        nv_bfloat16* __restrict__ dqkv,
        const nv_bfloat16* __restrict__ dk_expanded,
        int total_q, int Hq, int Hkv, int HS) {
    const int H = Hq + 2 * Hkv;
    const int h_ratio = Hq / Hkv;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = total_q * Hkv * HS;
    if (idx >= total) return;

    const int d = idx % HS;
    const int hkv = (idx / HS) % Hkv;
    const int t = idx / (Hkv * HS);

    // Sum h_ratio heads from dk_expanded
    float sum = 0.0f;
    for (int r = 0; r < h_ratio; ++r) {
        int src_idx = t * Hq * HS + (hkv * h_ratio + r) * HS + d;
        sum += __bfloat162float(dk_expanded[src_idx]);
    }

    // Write to K section of interleaved buffer
    dqkv[t * H * HS + (Hq + hkv) * HS + d] = __float2bfloat16(sum);
}

// Same as reduce_scatter_dk_kernel but for V section.
__global__ void reduce_scatter_dv_kernel(
        nv_bfloat16* __restrict__ dqkv,
        const nv_bfloat16* __restrict__ dv_expanded,
        int total_q, int Hq, int Hkv, int HS) {
    const int H = Hq + 2 * Hkv;
    const int h_ratio = Hq / Hkv;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = total_q * Hkv * HS;
    if (idx >= total) return;

    const int d = idx % HS;
    const int hkv = (idx / HS) % Hkv;
    const int t = idx / (Hkv * HS);

    // Sum h_ratio heads from dv_expanded
    float sum = 0.0f;
    for (int r = 0; r < h_ratio; ++r) {
        int src_idx = t * Hq * HS + (hkv * h_ratio + r) * HS + d;
        sum += __bfloat162float(dv_expanded[src_idx]);
    }

    // Write to V section of interleaved buffer
    dqkv[t * H * HS + (Hq + Hkv + hkv) * HS + d] = __float2bfloat16(sum);
}

}  // anonymous namespace

void scatter_reduce_dqkv(
        nv_bfloat16* dqkv,
        const nv_bfloat16* dq,
        const nv_bfloat16* dk_expanded,
        const nv_bfloat16* dv_expanded,
        int total_q, int Hq, int Hkv, int HS,
        cudaStream_t stream) {
    constexpr int kThreads = 256;

    // Scatter dQ
    {
        const int n = total_q * Hq * HS;
        const int blocks = (n + kThreads - 1) / kThreads;
        scatter_dq_kernel<<<blocks, kThreads, 0, stream>>>(
            dqkv, dq, total_q, Hq, Hkv, HS);
    }

    // Reduce+scatter dK
    {
        const int n = total_q * Hkv * HS;
        const int blocks = (n + kThreads - 1) / kThreads;
        reduce_scatter_dk_kernel<<<blocks, kThreads, 0, stream>>>(
            dqkv, dk_expanded, total_q, Hq, Hkv, HS);
    }

    // Reduce+scatter dV
    {
        const int n = total_q * Hkv * HS;
        const int blocks = (n + kThreads - 1) / kThreads;
        reduce_scatter_dv_kernel<<<blocks, kThreads, 0, stream>>>(
            dqkv, dv_expanded, total_q, Hq, Hkv, HS);
    }
}

// Reduce dk_expanded/dv_expanded (Hq heads each) to Hkv KV heads and scatter
// into the K and V sections of interleaved dqkv. dQ is assumed to already be
// in the correct position in dqkv (written directly by flash backward).
void reduce_scatter_dkv(
        nv_bfloat16* dqkv,
        const nv_bfloat16* dk_expanded,
        const nv_bfloat16* dv_expanded,
        int total_q, int Hq, int Hkv, int HS,
        cudaStream_t stream) {
    constexpr int kThreads = 256;

    // Reduce+scatter dK
    {
        const int n = total_q * Hkv * HS;
        const int blocks = (n + kThreads - 1) / kThreads;
        reduce_scatter_dk_kernel<<<blocks, kThreads, 0, stream>>>(
            dqkv, dk_expanded, total_q, Hq, Hkv, HS);
    }

    // Reduce+scatter dV
    {
        const int n = total_q * Hkv * HS;
        const int blocks = (n + kThreads - 1) / kThreads;
        reduce_scatter_dv_kernel<<<blocks, kThreads, 0, stream>>>(
            dqkv, dv_expanded, total_q, Hq, Hkv, HS);
    }
}
