// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Flat-token attention with paged KV-cache via FlashInfer.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/utils.cuh>

#include "kernels/attention_flat_paged.h"
#include "runtime/infer/decode_state.h"
#include "utilities/utils.h"

static_assert(sizeof(flashinfer::PrefillPlanInfo) <= infer::DecodeState::kPlanInfoSize,
    "DecodeState::kPlanInfoSize too small for PrefillPlanInfo");

namespace {

template <typename KVType, int HeadDim>
void dispatch_flat_paged_attention(
        nv_bfloat16* out,
        float* lse,
        const nv_bfloat16* q,
        const KVType* k_pages,
        const KVType* v_pages,
        const int32_t* q_indptr,
        const int32_t* page_indptr,
        const int32_t* page_indices,
        const int32_t* last_page_len,
        int32_t* request_indices,
        int32_t* qo_tile_indices,
        int32_t* kv_tile_indices,
        int32_t* o_indptr,
        int32_t* kv_chunk_size_ptr,
        bool* block_valid_mask,
        nv_bfloat16* tmp_v,
        float* tmp_s,
        int32_t* merge_indptr,
        int batch_size, int padded_batch_size,
        int total_num_rows,
        bool split_kv,
        int Hq, int Hkv, int page_block_size,
        int q_stride_n,
        uint32_t cta_tile_q,
        cudaStream_t stream) {
    if (Hkv <= 0 || Hq <= 0 || (Hq % Hkv) != 0) {
        throw std::runtime_error("attention_flat_paged: invalid head config for GQA");
    }
    using Params = flashinfer::BatchPrefillPagedParams<nv_bfloat16, KVType, nv_bfloat16, int32_t>;
    using AttentionVariant = flashinfer::DefaultAttention<
        /*use_custom_mask=*/false,
        /*use_sliding_window=*/false,
        /*use_logits_soft_cap=*/false,
        /*use_alibi=*/false>;

    flashinfer::paged_kv_t<KVType, int32_t> paged_kv(
        static_cast<uint32_t>(Hkv),
        static_cast<uint32_t>(page_block_size),
        static_cast<uint32_t>(HeadDim),
        static_cast<uint32_t>(batch_size),
        flashinfer::QKVLayout::kNHD,
        const_cast<KVType*>(k_pages),
        const_cast<KVType*>(v_pages),
        const_cast<int32_t*>(page_indices),
        const_cast<int32_t*>(page_indptr),
        const_cast<int32_t*>(last_page_len));

    Params params;
    params.q = const_cast<nv_bfloat16*>(q);
    params.paged_kv = paged_kv;
    params.maybe_custom_mask = nullptr;
    params.q_indptr = const_cast<int32_t*>(q_indptr);
    params.maybe_mask_indptr = nullptr;
    params.maybe_q_rope_offset = nullptr;
    params.o = out;
    params.lse = lse;
    params.maybe_alibi_slopes = nullptr;
    // Required by FlashInfer batched prefill kernels when using default ctor.
    params.group_size = flashinfer::uint_fastdiv(static_cast<uint32_t>(Hq / Hkv));
    params.num_qo_heads = static_cast<uint32_t>(Hq);
    params.q_stride_n = static_cast<int32_t>(q_stride_n > 0 ? q_stride_n : Hq * HeadDim);
    params.q_stride_h = static_cast<int32_t>(HeadDim);
    params.window_left = -1;
    params.logits_soft_cap = 0.0f;
    params.sm_scale = 1.0f / std::sqrt(static_cast<float>(HeadDim));
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;

    params.request_indices = request_indices;
    params.qo_tile_indices = qo_tile_indices;
    params.kv_tile_indices = kv_tile_indices;
    params.merge_indptr = merge_indptr;
    params.o_indptr = o_indptr;
    params.block_valid_mask = block_valid_mask;
    params.kv_chunk_size_ptr = kv_chunk_size_ptr;
    params.max_total_num_rows = static_cast<uint32_t>(total_num_rows);
    params.total_num_rows = nullptr;
    params.padded_batch_size = static_cast<uint32_t>(padded_batch_size);
    params.partition_kv = split_kv;
    params.maybe_prefix_len_ptr = nullptr;
    params.maybe_token_pos_in_items_ptr = nullptr;
    params.token_pos_in_items_len = 0;
    params.maybe_max_item_len_ptr = nullptr;

    auto launch = [&](auto cta_tag) -> cudaError_t {
        constexpr uint32_t CTQ = decltype(cta_tag)::value;
        return flashinfer::BatchPrefillWithPagedKVCacheDispatched<
            CTQ,
            static_cast<uint32_t>(HeadDim),
            static_cast<uint32_t>(HeadDim),
            flashinfer::PosEncodingMode::kNone,
            /*USE_FP16_QK_REDUCTION=*/false,
            flashinfer::MaskMode::kCausal,
            AttentionVariant>(params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
    };

    cudaError_t err;
    if (cta_tile_q == 128) {
        err = launch(std::integral_constant<uint32_t, 128>{});
    } else if (cta_tile_q == 64) {
        err = launch(std::integral_constant<uint32_t, 64>{});
    } else {
        err = launch(std::integral_constant<uint32_t, 16>{});
    }
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("attention_flat_paged dispatch failed: ") + cudaGetErrorString(err));
    }
}

}  // namespace

// ============================================================================
// Flat KV-cache store: per-token write with request mapping
// ============================================================================

__global__ void kv_cache_store_flat_paged_bf16_kernel(
        nv_bfloat16* __restrict__ k_pages,
        nv_bfloat16* __restrict__ v_pages,
        const nv_bfloat16* __restrict__ qkv_rope,
        const int32_t* __restrict__ token_to_req,
        const int32_t* __restrict__ kv_write_pos,
        const int* __restrict__ block_table,
        int block_table_stride,
        int page_block_size,
        int Hq, int Hkv, int Hs) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int dim_idx = threadIdx.x;
    if (dim_idx >= Hs) return;

    const int req_idx = token_to_req[token_idx];
    const int write_pos = kv_write_pos[token_idx];
    if (req_idx < 0 || write_pos < 0) {
        return;
    }
    const int H_total = Hq + 2 * Hkv;
    const int page_elems = page_block_size * Hkv * Hs;

    const long src_base = static_cast<long>(token_idx) * H_total * Hs;
    const nv_bfloat16 k_val = qkv_rope[src_base + (Hq + head_idx) * Hs + dim_idx];
    const nv_bfloat16 v_val = qkv_rope[src_base + (Hq + Hkv + head_idx) * Hs + dim_idx];

    const int vp = write_pos / page_block_size;
    if (vp < 0 || vp >= block_table_stride) {
        return;
    }
    const int po = write_pos % page_block_size;
    const int pp = block_table[req_idx * block_table_stride + vp];
    const long dst = static_cast<long>(pp) * page_elems + po * Hkv * Hs + head_idx * Hs + dim_idx;
    k_pages[dst] = k_val;
    v_pages[dst] = v_val;
}

void kv_cache_store_flat_paged_bf16(
        nv_bfloat16* k_pages, nv_bfloat16* v_pages,
        const nv_bfloat16* qkv_rope,
        const int32_t* token_to_req,
        const int32_t* kv_write_pos,
        const int* block_table, int block_table_stride,
        int page_block_size,
        int total_tokens, int Hq, int Hkv, int Hs,
        cudaStream_t stream) {
    if (total_tokens <= 0) return;
    dim3 grid(static_cast<unsigned int>(total_tokens), static_cast<unsigned int>(Hkv));
    dim3 block(static_cast<unsigned int>(Hs));
    kv_cache_store_flat_paged_bf16_kernel<<<grid, block, 0, stream>>>(
        k_pages, v_pages, qkv_rope, token_to_req, kv_write_pos,
        block_table, block_table_stride, page_block_size, Hq, Hkv, Hs);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void kv_cache_store_flat_paged_fp8_unit_scale_kernel(
        __nv_fp8_e4m3* __restrict__ k_pages_fp8,
        __nv_fp8_e4m3* __restrict__ v_pages_fp8,
        float* __restrict__ k_scales,
        float* __restrict__ v_scales,
        const nv_bfloat16* __restrict__ qkv_rope,
        const int32_t* __restrict__ token_to_req,
        const int32_t* __restrict__ kv_write_pos,
        const int* __restrict__ block_table,
        int block_table_stride,
        int page_block_size,
        int Hq, int Hkv, int Hs) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int dim_idx = threadIdx.x;
    if (dim_idx >= Hs) return;

    const int req_idx = token_to_req[token_idx];
    const int write_pos = kv_write_pos[token_idx];
    if (req_idx < 0 || write_pos < 0) return;

    const int vp = write_pos / page_block_size;
    if (vp < 0 || vp >= block_table_stride) return;
    const int po = write_pos % page_block_size;
    const int pp = block_table[req_idx * block_table_stride + vp];

    const int H_total = Hq + 2 * Hkv;
    const long src_base = static_cast<long>(token_idx) * H_total * Hs;
    const nv_bfloat16 k_bf = qkv_rope[src_base + (Hq + head_idx) * Hs + dim_idx];
    const nv_bfloat16 v_bf = qkv_rope[src_base + (Hq + Hkv + head_idx) * Hs + dim_idx];

    const int page_elems = page_block_size * Hkv * Hs;
    const long dst = static_cast<long>(pp) * page_elems + po * Hkv * Hs + head_idx * Hs + dim_idx;
    __nv_fp8_e4m3 qk;
    __nv_fp8_e4m3 qv;
    qk.__x = __nv_cvt_float_to_fp8(
        __bfloat162float(k_bf),
        __nv_saturation_t::__NV_SATFINITE,
        __nv_fp8_interpretation_t::__NV_E4M3);
    qv.__x = __nv_cvt_float_to_fp8(
        __bfloat162float(v_bf),
        __nv_saturation_t::__NV_SATFINITE,
        __nv_fp8_interpretation_t::__NV_E4M3);
    k_pages_fp8[dst] = qk;
    v_pages_fp8[dst] = qv;

    if (dim_idx == 0 && k_scales && v_scales) {
        const long scale_idx = static_cast<long>(pp) * page_block_size * Hkv
            + static_cast<long>(po) * Hkv + head_idx;
        k_scales[scale_idx] = 1.0f;
        v_scales[scale_idx] = 1.0f;
    }
}

void kv_cache_store_flat_paged_fp8(
        __nv_fp8_e4m3* k_pages_fp8, __nv_fp8_e4m3* v_pages_fp8,
        float* k_scales, float* v_scales,
        const nv_bfloat16* qkv_rope,
        const int32_t* token_to_req,
        const int32_t* kv_write_pos,
        const int* block_table, int block_table_stride,
        int page_block_size,
        int total_tokens, int Hq, int Hkv, int Hs,
        cudaStream_t stream) {
    if (total_tokens <= 0) return;
    dim3 grid(static_cast<unsigned int>(total_tokens), static_cast<unsigned int>(Hkv));
    dim3 block(static_cast<unsigned int>(Hs));
    kv_cache_store_flat_paged_fp8_unit_scale_kernel<<<grid, block, 0, stream>>>(
        k_pages_fp8, v_pages_fp8, k_scales, v_scales, qkv_rope,
        token_to_req, kv_write_pos, block_table, block_table_stride,
        page_block_size, Hq, Hkv, Hs);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// PrefillPlan: run once per flat_step, results cached across layers
// ============================================================================

void flat_attention_plan(
        const int32_t* h_q_indptr,     // [batch_size + 1] on HOST
        const int32_t* h_seq_lens_k,   // [batch_size] on HOST
        const int* h_block_table,      // [batch_size * block_table_stride] on HOST
        int block_table_stride,
        int page_block_size,
        int batch_size,
        int total_q_tokens,
        int Hq, int Hkv, int Hs,
        // Outputs (GPU buffers, caller allocates):
        int32_t* d_page_indptr,        // [batch_size + 1]
        int32_t* d_page_indices,       // [max_total_pages]
        int32_t* d_last_page_len,      // [batch_size]
        void* d_int_ws,                // int workspace
        void* d_float_ws,              // float workspace
        std::size_t int_ws_size,
        std::size_t float_ws_size,
        // Output plan info:
        void* plan_info_opaque,
        cudaStream_t stream) {
    if (!h_q_indptr || batch_size < 0 || total_q_tokens < 0) {
        throw std::runtime_error("flat_attention_plan: invalid inputs");
    }
    if (h_q_indptr[batch_size] != total_q_tokens) {
        throw std::runtime_error(
            "flat_attention_plan: q_indptr total does not match total_q_tokens");
    }
    auto& plan_info = *reinterpret_cast<flashinfer::PrefillPlanInfo*>(plan_info_opaque);
    // Build page metadata on CPU — reuse static buffers to avoid per-step heap allocation.
    static thread_local std::vector<int32_t> h_page_indptr;
    static thread_local std::vector<int32_t> h_last_page_len;
    h_page_indptr.assign(batch_size + 1, 0);
    h_last_page_len.resize(batch_size);
    int total_pages = 0;
    for (int i = 0; i < batch_size; ++i) {
        int k_len = std::max(1, static_cast<int>(h_seq_lens_k[i]));
        int num_pages = (k_len + page_block_size - 1) / page_block_size;
        total_pages += num_pages;
        h_page_indptr[i + 1] = total_pages;
        int rem = k_len % page_block_size;
        h_last_page_len[i] = (rem == 0) ? page_block_size : rem;
    }

    static thread_local std::vector<int32_t> h_page_indices;
    h_page_indices.resize(total_pages);
    for (int i = 0; i < batch_size; ++i) {
        int start = h_page_indptr[i];
        int count = h_page_indptr[i + 1] - start;
        for (int p = 0; p < count; ++p) {
            h_page_indices[start + p] = static_cast<int32_t>(
                h_block_table[i * block_table_stride + p]);
        }
    }

    // Upload page metadata.
    CUDA_CHECK(cudaMemcpyAsync(d_page_indptr, h_page_indptr.data(),
        (batch_size + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_page_indices, h_page_indices.data(),
        total_pages * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_last_page_len, h_last_page_len.data(),
        batch_size * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    // Build KV indptr (cumulative KV lengths).
    static thread_local std::vector<int32_t> h_kv_indptr;
    h_kv_indptr.assign(batch_size + 1, 0);
    for (int i = 0; i < batch_size; ++i) {
        h_kv_indptr[i + 1] = h_kv_indptr[i] + h_seq_lens_k[i];
    }

    // Page-locked host buffer for PrefillPlan — reuse across calls to avoid
    // allocating+zeroing 16MB of heap memory every step (~8ms overhead).
    static thread_local std::vector<char> page_locked_buf;
    if (page_locked_buf.size() < int_ws_size) {
        page_locked_buf.resize(int_ws_size, 0);
    }

    auto plan_err = flashinfer::PrefillPlan<int32_t>(
        d_float_ws, float_ws_size,
        d_int_ws, page_locked_buf.data(), int_ws_size,
        plan_info,
        const_cast<int32_t*>(h_q_indptr),
        h_kv_indptr.data(),
        static_cast<uint32_t>(total_q_tokens),
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(Hq),
        static_cast<uint32_t>(Hkv),
        static_cast<uint32_t>(Hs),
        static_cast<uint32_t>(Hs),
        static_cast<uint32_t>(page_block_size),
        /*enable_cuda_graph=*/false,
        /*sizeof_dtype_o=*/sizeof(nv_bfloat16),
        /*window_left=*/-1,
        /*fixed_split_size=*/0,
        // Keep flat-token path on the non-split prefill kernel for correctness.
        // This avoids merge-workspace layout mismatches across FlashInfer versions.
        /*disable_split_kv=*/true,
        /*num_colocated_ctas=*/0,
        stream);
    if (plan_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("flat_attention_plan failed: ") + cudaGetErrorString(plan_err));
    }
}

// ============================================================================
// Flat-token attention: uses pre-computed plan (no per-layer sync)
// ============================================================================

template <typename KVType>
void attention_flat_paged_flashinfer_impl(
        nv_bfloat16* out, float* lse,
        const nv_bfloat16* q,
        const KVType* k_pages, const KVType* v_pages,
        const int32_t* q_indptr_gpu,
        const int32_t* seq_lens_k,
        const int* block_table, int block_table_stride,
        int page_block_size,
        int batch_size, int total_q_tokens,
        int padded_batch_size_hint,
        int Hq, int Hkv, int Hs,
        // Pre-computed plan buffers (from flat_attention_plan):
        int32_t* page_indptr_gpu,
        int32_t* page_indices_gpu,
        int32_t* last_page_len_gpu,
        // Plan workspace (contains request_indices, tile_indices, etc.):
        void* int_ws_gpu,
        void* float_ws_gpu,
        const void* plan_info_opaque,
        // Unused scratch (kept for API compat):
        int32_t* /*scratch_request_indices*/,
        int32_t* /*scratch_qo_tile_indices*/,
        int32_t* /*scratch_kv_chunk_size*/,
        cudaStream_t stream,
        int q_stride_n) {
    (void)seq_lens_k;
    (void)block_table;
    (void)block_table_stride;
    (void)padded_batch_size_hint;

    if (batch_size <= 0 || total_q_tokens <= 0 || Hs <= 0) return;
    const auto& plan_info = *reinterpret_cast<const flashinfer::PrefillPlanInfo*>(plan_info_opaque);
    if (plan_info.split_kv) {
        throw std::runtime_error(
            "attention_flat_paged: split_kv plan is unsupported in this path");
    }

    // Extract plan pointers from int workspace.
    auto get_ptr = [&](std::size_t offset) -> int32_t* {
        return reinterpret_cast<int32_t*>(static_cast<char*>(int_ws_gpu) + offset);
    };

    int32_t* d_request_indices = get_ptr(plan_info.request_indices_offset);
    int32_t* d_qo_tile_indices = get_ptr(plan_info.qo_tile_indices_offset);
    int32_t* d_kv_tile_indices = get_ptr(plan_info.kv_tile_indices_offset);
    int32_t* d_o_indptr = get_ptr(plan_info.o_indptr_offset);
    int32_t* d_kv_chunk_size = get_ptr(plan_info.kv_chunk_size_ptr_offset);
    int32_t* d_merge_indptr = plan_info.split_kv
        ? get_ptr(plan_info.merge_indptr_offset) : nullptr;
    bool* d_block_valid_mask = plan_info.split_kv
        ? reinterpret_cast<bool*>(static_cast<char*>(int_ws_gpu) + plan_info.block_valid_mask_offset)
        : nullptr;
    nv_bfloat16* d_tmp_v = plan_info.split_kv
        ? reinterpret_cast<nv_bfloat16*>(static_cast<char*>(float_ws_gpu) + plan_info.v_offset)
        : nullptr;
    float* d_tmp_s = plan_info.split_kv
        ? reinterpret_cast<float*>(static_cast<char*>(float_ws_gpu) + plan_info.s_offset)
        : nullptr;

    const int pbs = static_cast<int>(plan_info.padded_batch_size);
    const uint32_t cta_tile_q = plan_info.cta_tile_q;

    if (Hs == 64) {
        dispatch_flat_paged_attention<KVType, 64>(
            out, lse, q, k_pages, v_pages,
            q_indptr_gpu, page_indptr_gpu, page_indices_gpu, last_page_len_gpu,
            d_request_indices, d_qo_tile_indices, d_kv_tile_indices,
            d_o_indptr, d_kv_chunk_size, d_block_valid_mask,
            d_tmp_v, d_tmp_s, d_merge_indptr,
            batch_size, pbs, total_q_tokens, plan_info.split_kv,
            Hq, Hkv, page_block_size, q_stride_n, cta_tile_q, stream);
    } else if (Hs == 96) {
        dispatch_flat_paged_attention<KVType, 96>(
            out, lse, q, k_pages, v_pages,
            q_indptr_gpu, page_indptr_gpu, page_indices_gpu, last_page_len_gpu,
            d_request_indices, d_qo_tile_indices, d_kv_tile_indices,
            d_o_indptr, d_kv_chunk_size, d_block_valid_mask,
            d_tmp_v, d_tmp_s, d_merge_indptr,
            batch_size, pbs, total_q_tokens, plan_info.split_kv,
            Hq, Hkv, page_block_size, q_stride_n, cta_tile_q, stream);
    } else if (Hs == 128) {
        dispatch_flat_paged_attention<KVType, 128>(
            out, lse, q, k_pages, v_pages,
            q_indptr_gpu, page_indptr_gpu, page_indices_gpu, last_page_len_gpu,
            d_request_indices, d_qo_tile_indices, d_kv_tile_indices,
            d_o_indptr, d_kv_chunk_size, d_block_valid_mask,
            d_tmp_v, d_tmp_s, d_merge_indptr,
            batch_size, pbs, total_q_tokens, plan_info.split_kv,
            Hq, Hkv, page_block_size, q_stride_n, cta_tile_q, stream);
    } else if (Hs == 256) {
        dispatch_flat_paged_attention<KVType, 256>(
            out, lse, q, k_pages, v_pages,
            q_indptr_gpu, page_indptr_gpu, page_indices_gpu, last_page_len_gpu,
            d_request_indices, d_qo_tile_indices, d_kv_tile_indices,
            d_o_indptr, d_kv_chunk_size, d_block_valid_mask,
            d_tmp_v, d_tmp_s, d_merge_indptr,
            batch_size, pbs, total_q_tokens, plan_info.split_kv,
            Hq, Hkv, page_block_size, q_stride_n, cta_tile_q, stream);
    } else {
        throw std::runtime_error("attention_flat_paged: unsupported head_size");
    }
}

void attention_flat_paged_flashinfer(
        nv_bfloat16* out, float* lse,
        const nv_bfloat16* q,
        const nv_bfloat16* k_pages, const nv_bfloat16* v_pages,
        const int32_t* q_indptr_gpu,
        const int32_t* seq_lens_k,
        const int* block_table, int block_table_stride,
        int page_block_size,
        int batch_size, int total_q_tokens,
        int padded_batch_size_hint,
        int Hq, int Hkv, int Hs,
        int32_t* page_indptr_gpu,
        int32_t* page_indices_gpu,
        int32_t* last_page_len_gpu,
        void* int_ws_gpu,
        void* float_ws_gpu,
        const void* plan_info_opaque,
        int32_t* scratch_request_indices,
        int32_t* scratch_qo_tile_indices,
        int32_t* scratch_kv_chunk_size,
        cudaStream_t stream,
        int q_stride_n) {
    attention_flat_paged_flashinfer_impl<nv_bfloat16>(
        out, lse, q, k_pages, v_pages,
        q_indptr_gpu, seq_lens_k, block_table, block_table_stride,
        page_block_size, batch_size, total_q_tokens, padded_batch_size_hint,
        Hq, Hkv, Hs,
        page_indptr_gpu, page_indices_gpu, last_page_len_gpu,
        int_ws_gpu, float_ws_gpu, plan_info_opaque,
        scratch_request_indices, scratch_qo_tile_indices, scratch_kv_chunk_size,
        stream, q_stride_n);
}

void attention_flat_paged_flashinfer_fp8(
        nv_bfloat16* out, float* lse,
        const nv_bfloat16* q,
        const __nv_fp8_e4m3* k_pages, const __nv_fp8_e4m3* v_pages,
        const int32_t* q_indptr_gpu,
        const int32_t* seq_lens_k,
        const int* block_table, int block_table_stride,
        int page_block_size,
        int batch_size, int total_q_tokens,
        int padded_batch_size_hint,
        int Hq, int Hkv, int Hs,
        int32_t* page_indptr_gpu,
        int32_t* page_indices_gpu,
        int32_t* last_page_len_gpu,
        void* int_ws_gpu,
        void* float_ws_gpu,
        const void* plan_info_opaque,
        int32_t* scratch_request_indices,
        int32_t* scratch_qo_tile_indices,
        int32_t* scratch_kv_chunk_size,
        cudaStream_t stream,
        int q_stride_n) {
    attention_flat_paged_flashinfer_impl<__nv_fp8_e4m3>(
        out, lse, q, k_pages, v_pages,
        q_indptr_gpu, seq_lens_k, block_table, block_table_stride,
        page_block_size, batch_size, total_q_tokens, padded_batch_size_hint,
        Hq, Hkv, Hs,
        page_indptr_gpu, page_indices_gpu, last_page_len_gpu,
        int_ws_gpu, float_ws_gpu, plan_info_opaque,
        scratch_request_indices, scratch_qo_tile_indices, scratch_kv_chunk_size,
        stream, q_stride_n);
}

// ============================================================================
// gather_rows_bf16: extract specific rows from a 2D tensor
// ============================================================================

__global__ void gather_rows_kernel(
    nv_bfloat16* __restrict__ dst,
    const nv_bfloat16* __restrict__ src,
    const int32_t* __restrict__ indices,
    int cols) {
    const int row = blockIdx.x;
    const int src_row = indices[row];
    const nv_bfloat16* src_ptr = src + static_cast<int64_t>(src_row) * cols;
    nv_bfloat16* dst_ptr = dst + static_cast<int64_t>(row) * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        dst_ptr[c] = src_ptr[c];
    }
}

void gather_rows_bf16(
    void* dst, const void* src,
    const int32_t* indices,
    int batch, int cols,
    cudaStream_t stream) {
    if (batch <= 0) return;
    const int threads = std::min(1024, cols);
    gather_rows_kernel<<<batch, threads, 0, stream>>>(
        reinterpret_cast<nv_bfloat16*>(dst),
        reinterpret_cast<const nv_bfloat16*>(src),
        indices, cols);
}
