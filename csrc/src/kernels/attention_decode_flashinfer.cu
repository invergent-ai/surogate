// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Paged decode attention via FlashInfer (BF16 KV-cache).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>

#include "kernels/attention_decode.h"
#include "utilities/utils.h"

namespace {

__global__ void build_flashinfer_decode_metadata_kernel(
        const int32_t* __restrict__ seqused_k,
        int batch_size,
        int page_block_size,
        int32_t* __restrict__ page_counts,
        int32_t* __restrict__ indptr,
        int32_t* __restrict__ last_page_len,
        int32_t* __restrict__ request_indices,
        int32_t* __restrict__ kv_tile_indices,
        int32_t* __restrict__ kv_chunk_size) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int32_t acc = 0;
    indptr[0] = 0;
    for (int i = 0; i < batch_size; ++i) {
        int32_t k_len = seqused_k ? seqused_k[i] : 0;
        if (k_len < 1) {
            k_len = 1;
        }
        int32_t num_pages = (k_len + page_block_size - 1) / page_block_size;
        if (num_pages < 1) {
            num_pages = 1;
        }
        page_counts[i] = num_pages;
        acc += num_pages;
        indptr[i + 1] = acc;

        const int32_t rem = k_len % page_block_size;
        last_page_len[i] = (rem == 0) ? page_block_size : rem;

        request_indices[i] = i;
        kv_tile_indices[i] = 0;
    }
    kv_chunk_size[0] = page_block_size;
}

__global__ void gather_flashinfer_page_indices_kernel(
        const int* __restrict__ block_table,
        int block_table_stride,
        const int32_t* __restrict__ page_counts,
        const int32_t* __restrict__ indptr,
        int32_t* __restrict__ out_indices,
        int batch_size) {
    const int seq_idx = blockIdx.x;
    if (seq_idx >= batch_size) return;

    const int page_idx = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
    const int count = page_counts[seq_idx];
    if (page_idx >= count) return;

    const int src = block_table[seq_idx * block_table_stride + page_idx];
    out_indices[indptr[seq_idx] + page_idx] = static_cast<int32_t>(src);
}

template <int HeadDim>
void dispatch_flashinfer_paged_decode(
        nv_bfloat16* out,
        float* lse,
        const nv_bfloat16* q,
        const nv_bfloat16* k_pages,
        const nv_bfloat16* v_pages,
        const int32_t* page_indptr,
        const int32_t* page_indices,
        const int32_t* last_page_len,
        int32_t* request_indices,
        int32_t* kv_tile_indices,
        int32_t* kv_chunk_size,
        int batch_size, int Hq, int Hkv, int page_block_size,
        bool enable_pdl,
        cudaStream_t stream) {
    using Params = flashinfer::BatchDecodeParams<nv_bfloat16, nv_bfloat16, nv_bfloat16, int32_t>;
    using AttentionVariant = flashinfer::DefaultAttention<
        /*use_custom_mask=*/false,
        /*use_sliding_window=*/false,
        /*use_logits_soft_cap=*/false,
        /*use_alibi=*/false>;

    flashinfer::paged_kv_t<nv_bfloat16, int32_t> paged_kv(
        static_cast<uint32_t>(Hkv),
        static_cast<uint32_t>(page_block_size),
        static_cast<uint32_t>(HeadDim),
        static_cast<uint32_t>(batch_size),
        flashinfer::QKVLayout::kNHD,
        const_cast<nv_bfloat16*>(k_pages),
        const_cast<nv_bfloat16*>(v_pages),
        const_cast<int32_t*>(page_indices),
        const_cast<int32_t*>(page_indptr),
        const_cast<int32_t*>(last_page_len));

    Params params;
    params.q = const_cast<nv_bfloat16*>(q);
    params.q_rope_offset = nullptr;
    params.paged_kv = paged_kv;
    params.o = out;
    params.lse = lse;
    params.maybe_alibi_slopes = nullptr;
    params.padded_batch_size = static_cast<uint32_t>(batch_size);
    params.num_qo_heads = static_cast<uint32_t>(Hq);
    params.q_stride_n = static_cast<int32_t>(Hq * HeadDim);
    params.q_stride_h = static_cast<int32_t>(HeadDim);
    params.window_left = -1;
    params.logits_soft_cap = 0.0f;
    params.sm_scale = 1.0f / std::sqrt(static_cast<float>(HeadDim));
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;

    params.request_indices = request_indices;
    params.kv_tile_indices = kv_tile_indices;
    params.o_indptr = const_cast<int32_t*>(page_indptr);
    params.kv_chunk_size_ptr = kv_chunk_size;
    params.block_valid_mask = nullptr;
    params.partition_kv = false;

    auto err = flashinfer::BatchDecodeWithPagedKVCacheDispatched<
        static_cast<uint32_t>(HeadDim),
        flashinfer::PosEncodingMode::kNone,
        AttentionVariant>(params, /*tmp_v=*/nullptr, /*tmp_s=*/nullptr, enable_pdl, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("attention_decode_flashinfer_paged failed: ") + cudaGetErrorString(err));
    }
}

}  // namespace

void attention_decode_flashinfer_paged(
        nv_bfloat16* out, float* lse,
        const nv_bfloat16* q,
        const nv_bfloat16* k_pages, const nv_bfloat16* v_pages,
        const int32_t* seqused_k,
        const int* block_table, int block_table_stride,
        int page_block_size,
        int batch_size, int Hq, int Hkv, int Hs,
        int32_t* scratch_page_counts,
        int32_t* scratch_indptr,
        int32_t* scratch_last_page_len,
        int32_t* scratch_indices,
        int32_t* scratch_request_indices,
        int32_t* scratch_kv_tile_indices,
        int32_t* scratch_kv_chunk_size,
        bool enable_pdl,
        cudaStream_t stream) {
    if (batch_size <= 0 || Hs <= 0) {
        return;
    }
    if (!out || !q || !k_pages || !v_pages || !seqused_k || !block_table ||
        !scratch_page_counts || !scratch_indptr || !scratch_last_page_len ||
        !scratch_indices || !scratch_request_indices || !scratch_kv_tile_indices ||
        !scratch_kv_chunk_size) {
        throw std::runtime_error("attention_decode_flashinfer_paged: null pointer input");
    }

    build_flashinfer_decode_metadata_kernel<<<1, 1, 0, stream>>>(
        seqused_k,
        batch_size,
        page_block_size,
        scratch_page_counts,
        scratch_indptr,
        scratch_last_page_len,
        scratch_request_indices,
        scratch_kv_tile_indices,
        scratch_kv_chunk_size);
    CUDA_CHECK(cudaGetLastError());

    constexpr int kThreads = 128;
    const int grid_y = (block_table_stride + kThreads - 1) / kThreads;
    dim3 grid(static_cast<unsigned int>(batch_size), static_cast<unsigned int>(grid_y));
    gather_flashinfer_page_indices_kernel<<<grid, kThreads, 0, stream>>>(
        block_table,
        block_table_stride,
        scratch_page_counts,
        scratch_indptr,
        scratch_indices,
        batch_size);
    CUDA_CHECK(cudaGetLastError());

    if (Hs == 64) {
        dispatch_flashinfer_paged_decode<64>(
            out, lse, q, k_pages, v_pages,
            scratch_indptr, scratch_indices, scratch_last_page_len,
            scratch_request_indices, scratch_kv_tile_indices, scratch_kv_chunk_size,
            batch_size, Hq, Hkv, page_block_size, enable_pdl, stream);
    } else if (Hs == 96) {
        dispatch_flashinfer_paged_decode<96>(
            out, lse, q, k_pages, v_pages,
            scratch_indptr, scratch_indices, scratch_last_page_len,
            scratch_request_indices, scratch_kv_tile_indices, scratch_kv_chunk_size,
            batch_size, Hq, Hkv, page_block_size, enable_pdl, stream);
    } else if (Hs == 128) {
        dispatch_flashinfer_paged_decode<128>(
            out, lse, q, k_pages, v_pages,
            scratch_indptr, scratch_indices, scratch_last_page_len,
            scratch_request_indices, scratch_kv_tile_indices, scratch_kv_chunk_size,
            batch_size, Hq, Hkv, page_block_size, enable_pdl, stream);
    } else if (Hs == 256) {
        dispatch_flashinfer_paged_decode<256>(
            out, lse, q, k_pages, v_pages,
            scratch_indptr, scratch_indices, scratch_last_page_len,
            scratch_request_indices, scratch_kv_tile_indices, scratch_kv_chunk_size,
            batch_size, Hq, Hkv, page_block_size, enable_pdl, stream);
    } else {
        throw std::runtime_error(
            "attention_decode_flashinfer_paged: unsupported head_size (supported: 64/96/128/256)");
    }
}
