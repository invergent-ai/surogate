// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Fused GPU kernel for EP dispatch: prepares the A2A send buffer
// directly in destination-GPU order, eliminating CPU-side sort.
//
// Replaces the pattern:
//   1. CPU: compute send_order[]
//   2. H2D copy send_order
//   3. moe_permute_tokens using send_order
// with a single GPU kernel that reads expert_offsets + expert_to_gpu
// mapping and writes tokens directly to the correct send buffer position.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

// Kernel: compute per-token send position and scatter into send buffer.
//
// Each thread handles one token. It reads the token's expert assignment
// from its position in the expert-sorted input, looks up which GPU the
// expert is assigned to, and writes the token to the correct position
// in the destination-ordered send buffer.
//
// Also optionally outputs the send_order mapping (for backward):
//   send_order[write_pos] = token_idx
template <typename T>
__global__ void ep_scatter_send_buffer_kernel(
    T* __restrict__ send_buf,             // [total_tokens, hidden_size] output
    const T* __restrict__ input,          // [total_tokens, hidden_size] expert-sorted
    const int* __restrict__ expert_offsets,// [num_experts + 1]
    const int* __restrict__ expert_to_gpu, // [num_experts] → GPU id
    const int* __restrict__ peer_write_offsets, // [num_experts] per-expert write offset in send buf
    int* __restrict__ send_order,         // [total_tokens] output: send_order[write_pos] = token_idx (may be null)
    int num_experts,
    int hidden_size,
    int total_tokens) {

    const int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= total_tokens) return;

    // Find which expert this token belongs to (binary search on expert_offsets)
    int lo = 0, hi = num_experts;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (expert_offsets[mid + 1] <= token_idx) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const int expert_id = lo;

    // Compute position within this expert's token range
    const int expert_start = expert_offsets[expert_id];
    const int pos_in_expert = token_idx - expert_start;

    // Get the write position for this token in the send buffer
    const int write_pos = peer_write_offsets[expert_id] + pos_in_expert;

    // Record the mapping for backward pass
    if (send_order) {
        send_order[write_pos] = token_idx;
    }

    // Copy hidden_size elements
    const T* src = input + static_cast<int64_t>(token_idx) * hidden_size;
    T* dst = send_buf + static_cast<int64_t>(write_pos) * hidden_size;

    for (int c = 0; c < hidden_size; ++c) {
        dst[c] = src[c];
    }
}

// Kernel to compute per-expert write offsets in the send buffer.
// Groups experts by destination GPU, producing cumulative offsets.
// peer_write_offsets[e] = starting position for expert e's tokens in send buffer.
__global__ void ep_compute_write_offsets_kernel(
    int* __restrict__ peer_write_offsets,  // [num_experts]
    const int* __restrict__ expert_offsets, // [num_experts + 1]
    const int* __restrict__ expert_to_gpu, // [num_experts]
    int num_experts,
    int ep_size) {

    // Single-thread kernel (num_experts is small, typically 64-256)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Pass 1: compute per-peer total token counts
    int peer_offsets[64];  // Max 64 GPUs
    for (int p = 0; p < ep_size && p < 64; ++p) {
        peer_offsets[p] = 0;
    }
    for (int e = 0; e < num_experts; ++e) {
        int gpu = expert_to_gpu[e];
        peer_offsets[gpu] += expert_offsets[e + 1] - expert_offsets[e];
    }

    // Pass 2: prefix sum to get per-peer start offsets
    int accum = 0;
    int peer_starts[64];
    for (int p = 0; p < ep_size && p < 64; ++p) {
        peer_starts[p] = accum;
        accum += peer_offsets[p];
    }

    // Pass 3: compute per-expert write offset within each peer's region
    // Reset peer_offsets to use as running counters
    for (int p = 0; p < ep_size && p < 64; ++p) {
        peer_offsets[p] = peer_starts[p];
    }
    for (int e = 0; e < num_experts; ++e) {
        int gpu = expert_to_gpu[e];
        peer_write_offsets[e] = peer_offsets[gpu];
        peer_offsets[gpu] += expert_offsets[e + 1] - expert_offsets[e];
    }
}

}  // namespace

// C++ linkage to match kernels.h declarations

void ep_fused_prepare_send_buffer_bf16(
    nv_bfloat16* send_buf,
    const nv_bfloat16* input,
    const int* expert_offsets,
    const int* expert_to_gpu,
    int* peer_write_offsets,  // temporary [num_experts] buffer
    int* send_order,          // [total_tokens] output (may be null)
    int num_experts,
    int ep_size,
    int hidden_size,
    int total_tokens,
    cudaStream_t stream) {

    if (total_tokens <= 0) return;

    // Step 1: Compute per-expert write offsets on GPU
    ep_compute_write_offsets_kernel<<<1, 1, 0, stream>>>(
        peer_write_offsets, expert_offsets, expert_to_gpu,
        num_experts, ep_size);

    // Step 2: Scatter tokens into send buffer + record send_order
    const int threads_x = 256;
    const int blocks = (total_tokens + threads_x - 1) / threads_x;
    ep_scatter_send_buffer_kernel<nv_bfloat16><<<blocks, threads_x, 0, stream>>>(
        send_buf, input, expert_offsets, expert_to_gpu, peer_write_offsets,
        send_order, num_experts, hidden_size, total_tokens);
}

void ep_fused_prepare_send_buffer_fp32(
    float* send_buf,
    const float* input,
    const int* expert_offsets,
    const int* expert_to_gpu,
    int* peer_write_offsets,
    int* send_order,
    int num_experts,
    int ep_size,
    int hidden_size,
    int total_tokens,
    cudaStream_t stream) {

    if (total_tokens <= 0) return;

    ep_compute_write_offsets_kernel<<<1, 1, 0, stream>>>(
        peer_write_offsets, expert_offsets, expert_to_gpu,
        num_experts, ep_size);

    const int threads_x = 256;
    const int blocks = (total_tokens + threads_x - 1) / threads_x;
    ep_scatter_send_buffer_kernel<float><<<blocks, threads_x, 0, stream>>>(
        send_buf, input, expert_offsets, expert_to_gpu, peer_write_offsets,
        send_order, num_experts, hidden_size, total_tokens);
}

