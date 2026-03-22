// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// CUDA kernels for KV-cache management during autoregressive generation.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "kernels/attention_decode.h"
#include "utilities/utils.h"

namespace {

/// Append one token's K and V from interleaved QKV to separate K/V caches.
/// Grid: (batch_size, Hkv)  Block: (Hs) — one thread per element.
///
/// QKV layout: [batch_size, 1, (Hq + 2*Hkv), Hs]
///   K starts at head offset Hq, V starts at head offset Hq + Hkv.
///
/// K-cache layout: [batch_size, max_seq_len, Hkv, Hs]
/// V-cache layout: [batch_size, max_seq_len, Hkv, Hs]
__global__ void kv_cache_append_bf16_kernel(
        nv_bfloat16* __restrict__ k_cache,
        nv_bfloat16* __restrict__ v_cache,
        const nv_bfloat16* __restrict__ qkv_rope,
        const int* __restrict__ seq_lens_gpu,
        int max_seq_len, int Hq, int Hkv, int Hs) {

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int dim_idx = threadIdx.x;

    if (dim_idx >= Hs) return;

    const int seq_pos = seq_lens_gpu[batch_idx];  // Position to write at
    const int H_total = Hq + 2 * Hkv;

    // Source: QKV[batch_idx, 0, head, dim]
    // QKV stride: batch_stride = 1 * H_total * Hs (T=1)
    const int qkv_offset = batch_idx * H_total * Hs;
    const int k_src_offset = qkv_offset + (Hq + head_idx) * Hs + dim_idx;
    const int v_src_offset = qkv_offset + (Hq + Hkv + head_idx) * Hs + dim_idx;

    // Destination: cache[batch_idx, seq_pos, head_idx, dim_idx]
    // Cache stride: [batch_size, max_seq_len, Hkv, Hs]
    const long cache_offset = static_cast<long>(batch_idx) * max_seq_len * Hkv * Hs
                            + static_cast<long>(seq_pos) * Hkv * Hs
                            + head_idx * Hs
                            + dim_idx;

    k_cache[cache_offset] = qkv_rope[k_src_offset];
    v_cache[cache_offset] = qkv_rope[v_src_offset];
}

/// Fill cu_seqlens_q and seqused_k for decode attention.
/// cu_seqlens_q: [0, 1, 2, ..., batch_size]  (cumulative, batch_size+1 elements)
/// seqused_k:    [s0+1, s1+1, ...]            (per-sequence, batch_size elements)
/// Grid: 1  Block: max(batch_size + 1, batch_size)
__global__ void fill_decode_cu_seqlens_kernel(
        int32_t* __restrict__ cu_seqlens_q,
        int32_t* __restrict__ seqused_k,
        const int* __restrict__ seq_lens_gpu,
        int batch_size) {

    const int idx = threadIdx.x;

    // Q: cumulative seqlens [0, 1, 2, ..., batch_size]
    if (idx <= batch_size) {
        cu_seqlens_q[idx] = idx;
    }

    // K: per-sequence lengths (after KV append: seq_lens[i] + 1)
    if (idx < batch_size) {
        seqused_k[idx] = seq_lens_gpu[idx] + 1;
    }
}

__global__ void fill_iota_i32_kernel(
        int32_t* __restrict__ out,
        int n) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out[idx] = idx;
    }
}

/// Bulk-store all T positions of K/V from interleaved QKV into contiguous KV-cache.
/// Grid: (B*T, Hkv)  Block: (Hs)
__global__ void kv_cache_store_bf16_kernel(
        nv_bfloat16* __restrict__ k_cache,
        nv_bfloat16* __restrict__ v_cache,
        const nv_bfloat16* __restrict__ qkv_rope,
        int T, int max_seq_len, int Hq, int Hkv, int Hs, int start_pos) {

    const int bt_idx = blockIdx.x;  // = batch_idx * T + t
    const int head_idx = blockIdx.y;
    const int dim_idx = threadIdx.x;
    if (dim_idx >= Hs) return;

    const int batch_idx = bt_idx / T;
    const int t = bt_idx % T;
    const int t_abs = start_pos + t;
    if (t_abs >= max_seq_len) return;
    const int H_total = Hq + 2 * Hkv;

    // Source: QKV[batch_idx, t, head, dim]
    const long src_base = static_cast<long>(bt_idx) * H_total * Hs;
    const nv_bfloat16 k_val = qkv_rope[src_base + (Hq + head_idx) * Hs + dim_idx];
    const nv_bfloat16 v_val = qkv_rope[src_base + (Hq + Hkv + head_idx) * Hs + dim_idx];

    // Dest: cache[batch_idx, t, head_idx, dim_idx]
    const long dst = static_cast<long>(batch_idx) * max_seq_len * Hkv * Hs
                   + static_cast<long>(t_abs) * Hkv * Hs
                   + head_idx * Hs + dim_idx;
    k_cache[dst] = k_val;
    v_cache[dst] = v_val;
}

/// Bulk-store all T positions of K/V into paged KV-cache.
/// Grid: (B*T, Hkv)  Block: (Hs)
__global__ void kv_cache_store_paged_bf16_kernel(
        nv_bfloat16* __restrict__ k_pages,
        nv_bfloat16* __restrict__ v_pages,
        const nv_bfloat16* __restrict__ qkv_rope,
        const int* __restrict__ block_table,
        int block_table_stride,
        int page_block_size,
        int T, int Hq, int Hkv, int Hs, int start_pos) {

    const int bt_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int dim_idx = threadIdx.x;
    if (dim_idx >= Hs) return;

    const int batch_idx = bt_idx / T;
    const int t = bt_idx % T;
    const int t_abs = start_pos + t;
    const int H_total = Hq + 2 * Hkv;
    const int page_elems = page_block_size * Hkv * Hs;

    const long src_base = static_cast<long>(bt_idx) * H_total * Hs;
    const nv_bfloat16 k_val = qkv_rope[src_base + (Hq + head_idx) * Hs + dim_idx];
    const nv_bfloat16 v_val = qkv_rope[src_base + (Hq + Hkv + head_idx) * Hs + dim_idx];

    const int vp = t_abs / page_block_size;
    const int po = t_abs % page_block_size;
    const int pp = block_table[batch_idx * block_table_stride + vp];
    const long dst = static_cast<long>(pp) * page_elems + po * Hkv * Hs + head_idx * Hs + dim_idx;
    k_pages[dst] = k_val;
    v_pages[dst] = v_val;
}

/// Append one token's K/V to paged KV-cache.
/// Grid: (batch_size, Hkv)  Block: (Hs)
/// Looks up the correct page via block_table, then writes at the offset within that page.
__global__ void kv_cache_append_paged_bf16_kernel(
        nv_bfloat16* __restrict__ k_pages,
        nv_bfloat16* __restrict__ v_pages,
        const nv_bfloat16* __restrict__ qkv_rope,
        const int* __restrict__ seq_lens_gpu,
        const int* __restrict__ block_table,
        int block_table_stride,
        int page_block_size,
        int Hq, int Hkv, int Hs) {

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int dim_idx = threadIdx.x;

    if (dim_idx >= Hs) return;

    const int seq_pos = seq_lens_gpu[batch_idx];
    const int H_total = Hq + 2 * Hkv;
    const int page_elems = page_block_size * Hkv * Hs;

    // Source: QKV[batch_idx, 0, head, dim]
    const int qkv_offset = batch_idx * H_total * Hs;
    const int k_src_offset = qkv_offset + (Hq + head_idx) * Hs + dim_idx;
    const int v_src_offset = qkv_offset + (Hq + Hkv + head_idx) * Hs + dim_idx;

    // Destination: look up physical page via block table
    const int virtual_page = seq_pos / page_block_size;
    const int page_offset = seq_pos % page_block_size;
    const int physical_page = block_table[batch_idx * block_table_stride + virtual_page];

    const long dest_offset = static_cast<long>(physical_page) * page_elems
                           + page_offset * Hkv * Hs
                           + head_idx * Hs
                           + dim_idx;

    k_pages[dest_offset] = qkv_rope[k_src_offset];
    v_pages[dest_offset] = qkv_rope[v_src_offset];
}

/// Mask finished tokens: set to 0 (pad) for finished sequences.
/// Grid: 1  Block: batch_size
__global__ void mask_finished_tokens_kernel(
        int32_t* __restrict__ token_ids,
        const int* __restrict__ finished_gpu,
        int batch_size) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batch_size) return;
    if (finished_gpu[idx] != 0) {
        token_ids[idx] = 0;
    }
}

/// Update per-sequence decode state after one sampling step.
/// Active sequence:
///   seq_len += 1
///   completion_len += 1
///   if sampled token is EOS -> mark finished
__global__ void update_generation_state_kernel(
        const int32_t* __restrict__ sampled_tokens,
        int* __restrict__ finished_gpu,
        int* __restrict__ seq_lens_gpu,
        int32_t* __restrict__ completion_lens_gpu,
        int32_t eos_token_id,
        int batch_size) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batch_size) return;
    if (finished_gpu[idx] != 0) {
        return;
    }

    seq_lens_gpu[idx] += 1;
    completion_lens_gpu[idx] += 1;
    if (sampled_tokens[idx] == eos_token_id) {
        finished_gpu[idx] = 1;
    }
}

__global__ void count_active_sequences_kernel(
        const int* __restrict__ finished_gpu,
        int* __restrict__ active_count_gpu,
        int batch_size) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batch_size) return;
    if (finished_gpu[idx] == 0) {
        atomicAdd(active_count_gpu, 1);
    }
}

}  // anonymous namespace

void mask_finished_tokens(
        int32_t* token_ids,
        const int* finished_gpu,
        int batch_size,
        cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    mask_finished_tokens_kernel<<<blocks, threads, 0, stream>>>(
        token_ids, finished_gpu, batch_size);
}

void update_generation_state(
        const int32_t* sampled_tokens,
        int* finished_gpu,
        int* seq_lens_gpu,
        int32_t* completion_lens_gpu,
        int32_t eos_token_id,
        int batch_size,
        cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    update_generation_state_kernel<<<blocks, threads, 0, stream>>>(
        sampled_tokens, finished_gpu, seq_lens_gpu, completion_lens_gpu, eos_token_id, batch_size);
}

void count_active_sequences(
        const int* finished_gpu,
        int* active_count_gpu,
        int batch_size,
        cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(active_count_gpu, 0, sizeof(int), stream));
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    count_active_sequences_kernel<<<blocks, threads, 0, stream>>>(
        finished_gpu, active_count_gpu, batch_size);
}

void kv_cache_store_bf16(
        nv_bfloat16* k_cache, nv_bfloat16* v_cache,
        const nv_bfloat16* qkv_rope,
        int B, int T, int max_seq_len,
        int Hq, int Hkv, int Hs,
        int start_pos,
        cudaStream_t stream) {

    dim3 grid(B * T, Hkv);
    dim3 block(Hs);
    kv_cache_store_bf16_kernel<<<grid, block, 0, stream>>>(
        k_cache, v_cache, qkv_rope, T, max_seq_len, Hq, Hkv, Hs, start_pos);
}

void kv_cache_store_paged_bf16(
        nv_bfloat16* k_pages, nv_bfloat16* v_pages,
        const nv_bfloat16* qkv_rope,
        const int* block_table, int block_table_stride,
        int page_block_size,
        int B, int T,
        int Hq, int Hkv, int Hs,
        int start_pos,
        cudaStream_t stream) {

    dim3 grid(B * T, Hkv);
    dim3 block(Hs);
    kv_cache_store_paged_bf16_kernel<<<grid, block, 0, stream>>>(
        k_pages, v_pages, qkv_rope,
        block_table, block_table_stride, page_block_size,
        T, Hq, Hkv, Hs, start_pos);
}

void kv_cache_append_paged_bf16(
        nv_bfloat16* k_pages, nv_bfloat16* v_pages,
        const nv_bfloat16* qkv_rope,
        const int* seq_lens_gpu,
        const int* block_table, int block_table_stride,
        int page_block_size,
        int batch_size, int Hq, int Hkv, int Hs,
        cudaStream_t stream) {

    dim3 grid(batch_size, Hkv);
    dim3 block(Hs);
    kv_cache_append_paged_bf16_kernel<<<grid, block, 0, stream>>>(
        k_pages, v_pages, qkv_rope, seq_lens_gpu,
        block_table, block_table_stride, page_block_size,
        Hq, Hkv, Hs);
}

void kv_cache_append_bf16(
        nv_bfloat16* k_cache, nv_bfloat16* v_cache,
        const nv_bfloat16* qkv_rope,
        const int* seq_lens_gpu,
        int batch_size, int max_seq_len,
        int Hq, int Hkv, int Hs,
        cudaStream_t stream) {

    dim3 grid(batch_size, Hkv);
    dim3 block(Hs);  // head_dim threads — typically 128
    kv_cache_append_bf16_kernel<<<grid, block, 0, stream>>>(
        k_cache, v_cache, qkv_rope, seq_lens_gpu,
        max_seq_len, Hq, Hkv, Hs);
}

void kv_cache_broadcast_prefix(
        nv_bfloat16* k_cache, nv_bfloat16* v_cache,
        int src_slot, const int* dst_slots, int num_copies,
        int prefix_len, int max_seq_len,
        int Hkv, int Hs,
        cudaStream_t stream) {

    // Each batch slot occupies [max_seq_len, Hkv, Hs] elements.
    // We copy [prefix_len, Hkv, Hs] from src to each dst.
    const std::size_t slot_stride = static_cast<std::size_t>(max_seq_len) * Hkv * Hs;
    const std::size_t copy_elems = static_cast<std::size_t>(prefix_len) * Hkv * Hs;
    const std::size_t copy_bytes = copy_elems * sizeof(nv_bfloat16);

    const nv_bfloat16* k_src = k_cache + static_cast<std::size_t>(src_slot) * slot_stride;
    const nv_bfloat16* v_src = v_cache + static_cast<std::size_t>(src_slot) * slot_stride;

    for (int i = 0; i < num_copies; ++i) {
        const int dst = dst_slots[i];
        nv_bfloat16* k_dst = k_cache + static_cast<std::size_t>(dst) * slot_stride;
        nv_bfloat16* v_dst = v_cache + static_cast<std::size_t>(dst) * slot_stride;

        cudaMemcpyAsync(k_dst, k_src, copy_bytes, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(v_dst, v_src, copy_bytes, cudaMemcpyDeviceToDevice, stream);
    }
}

void fill_decode_cu_seqlens(
        int32_t* cu_seqlens_q,
        int32_t* cu_seqlens_k,
        const int* seq_lens_gpu,
        int batch_size,
        cudaStream_t stream) {

    // batch_size + 1 threads — small kernel, single block
    fill_decode_cu_seqlens_kernel<<<1, batch_size + 1, 0, stream>>>(
        cu_seqlens_q, cu_seqlens_k, seq_lens_gpu, batch_size);
}

void fill_iota_i32(
        int32_t* out,
        int n,
        cudaStream_t stream) {
    if (!out || n <= 0) {
        return;
    }
    constexpr int kThreads = 256;
    const int blocks = (n + kThreads - 1) / kThreads;
    fill_iota_i32_kernel<<<blocks, kThreads, 0, stream>>>(out, n);
}
