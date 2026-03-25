// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file attention_decode_tq.cu
 * @brief Host launchers for TurboQuant 3.5-bit fused decode attention + KV append.
 *
 * Provides:
 *   - attention_decode_paged_tq()    — fused Q@K^T → softmax → O from TQ cache
 *   - kv_cache_append_paged_tq()    — quantize BF16 K/V → TQ packed into page pool
 *   - kv_cache_store_paged_tq()     — bulk prefill version
 */

#include "attention_decode_tq.cuh"
#include "turboquant.h"
#include "utilities/utils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <random>

// ===========================================================================
// Sign generation utility
// ===========================================================================

void tq_generate_kv_signs_host(
    uint32_t* signs_out,
    int num_kv_heads, int head_dim,
    uint64_t seed)
{
    const int words_per_sign = head_dim / 32;
    const int words_per_head = 4 * words_per_sign;  // d1_k, d2_k, d1_v, d2_v
    std::mt19937_64 rng(seed);
    for (int h = 0; h < num_kv_heads; h++) {
        for (int s = 0; s < 4; s++) {
            for (int w = 0; w < words_per_sign; w++) {
                uint32_t word = 0;
                for (int b = 0; b < 32; b++)
                    if (rng() & 1) word |= (1u << b);
                signs_out[h * words_per_head + s * words_per_sign + w] = word;
            }
        }
    }
}

namespace tq_decode {

// ---------------------------------------------------------------------------
// KV cache append: quantize single token K/V into TQ paged cache
// ---------------------------------------------------------------------------

/// Extract K/V from interleaved QKV, TurboQuant-compress, write to page pool.
/// Grid: (batch_size, Hkv)  Block: (32) — one warp per (batch, kv_head)
template <int EPT>
__global__ void kv_cache_append_paged_tq_kernel(
    uint8_t* __restrict__ k_pages,            // TQ packed K pool (this layer)
    uint8_t* __restrict__ v_pages,            // TQ packed V pool (this layer)
    const nv_bfloat16* __restrict__ qkv_rope, // [B, 1, Hq+2*Hkv, Hs]
    const uint32_t* __restrict__ kv_signs,    // [Hkv, 4, D/32]
    const int* __restrict__ seq_lens_gpu,     // [B] current lengths (write position)
    const int* __restrict__ block_table,      // [B, block_table_stride]
    int block_table_stride,
    int page_block_size,
    int Hq, int Hkv, int Hs)
{
    constexpr int D = EPT * 32;
    constexpr int TQ_PACKED = 7 * D / 16 + 4;
    constexpr int SIGNS_PER_HEAD = 4 * (D / 32);

    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int lane = threadIdx.x;

    // Shared memory for TQ packing (index staging)
    extern __shared__ uint8_t smem[];

    const int seq_pos = seq_lens_gpu[batch_idx];
    const int virt_page = seq_pos / page_block_size;
    const int page_off  = seq_pos % page_block_size;
    const int phys_page = block_table[batch_idx * block_table_stride + virt_page];

    const long page_kv_stride = (long)page_block_size * Hkv * TQ_PACKED;

    // Load per-head signs
    const uint32_t* head_signs = kv_signs + head_idx * SIGNS_PER_HEAD;
    constexpr int SW = D / 32;

    // QKV layout: [B, 1, Hq+2*Hkv, Hs]
    const int H_total = Hq + 2 * Hkv;
    const nv_bfloat16* row = qkv_rope + (long)batch_idx * H_total * Hs;

    // ---- Quantize K ----
    {
        const nv_bfloat16* k_src = row + (Hq + head_idx) * Hs;
        float vals[EPT];
        #pragma unroll
        for (int i = 0; i < EPT; i++)
            vals[i] = __bfloat162float(k_src[lane * EPT + i]);

        // Compute norm
        float nsq = 0.0f;
        #pragma unroll
        for (int i = 0; i < EPT; i++) nsq += vals[i] * vals[i];
        nsq = warp_sum(nsq);
        float norm_x = sqrtf(nsq);
        float inv_norm = (norm_x > 1e-12f) ? (1.0f / norm_x) : 0.0f;
        #pragma unroll
        for (int i = 0; i < EPT; i++) vals[i] *= inv_norm;

        // Rotation: D1_k signs + WHT
        uint32_t d1w = head_signs[0 * SW + lane / (32 / EPT)];
        int d1b = (lane % (32 / EPT)) * EPT;
        apply_signs<EPT>(vals, d1w, d1b);
        warp_wht<EPT>(vals, lane);

        // Quantize (mixed 3/2 bit)
        uint8_t indices[EPT];
        float centroids[EPT];
        bool first_half = (lane < 16);
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            float v = vals[i];
            if (first_half) {
                // 3-bit
                uint8_t idx;
                if (v < 0.0f) { idx = (v < -1.050f) ? ((v < -1.748f) ? 0 : 1) : ((v < -0.501f) ? 2 : 3); }
                else          { idx = (v < 1.050f) ? ((v < 0.501f) ? 4 : 5) : ((v < 1.748f) ? 6 : 7); }
                indices[i] = idx;
                centroids[i] = kCB3[idx];
            } else {
                uint8_t idx;
                if (v < 0.0f) idx = (v < -0.982f) ? 0 : 1;
                else          idx = (v < 0.982f) ? 2 : 3;
                indices[i] = idx;
                centroids[i] = kCB2[idx];
            }
        }

        // Residual + QJL
        float residual[EPT];
        #pragma unroll
        for (int i = 0; i < EPT; i++) residual[i] = vals[i] - centroids[i];

        float rsq = 0.0f;
        #pragma unroll
        for (int i = 0; i < EPT; i++) rsq += residual[i] * residual[i];
        rsq = warp_sum(rsq);
        float gamma = sqrtf(rsq);

        uint32_t d2w = head_signs[1 * SW + lane / (32 / EPT)];
        int d2b = (lane % (32 / EPT)) * EPT;
        apply_signs<EPT>(residual, d2w, d2b);
        warp_wht<EPT>(residual, lane);

        // Pack output
        uint8_t* out_ptr = k_pages
            + (long)phys_page * page_kv_stride
            + (long)page_off * Hkv * TQ_PACKED
            + (long)head_idx * TQ_PACKED;

        // QJL ballots
        uint32_t ballots[EPT];
        #pragma unroll
        for (int i = 0; i < EPT; i++)
            ballots[i] = __ballot_sync(0xFFFFFFFF, residual[i] >= 0.0f);

        constexpr int OFF_QJL = 3 * D / 16 + D / 8;
        if (lane < EPT)
            reinterpret_cast<uint32_t*>(out_ptr + OFF_QJL)[lane] = ballots[lane];

        // Norms
        constexpr int OFF_NORMS = OFF_QJL + D / 8;
        if (lane == 0) {
            reinterpret_cast<__half*>(out_ptr + OFF_NORMS)[0] = __float2half(gamma);
            reinterpret_cast<__half*>(out_ptr + OFF_NORMS)[1] = __float2half(norm_x);
        }

        // Stage indices to smem, pack
        #pragma unroll
        for (int i = 0; i < EPT; i++)
            smem[lane * EPT + i] = indices[i];
        __syncwarp();

        for (int g = lane; g < D / 16; g += 32) {
            uint8_t a=smem[g*8], b=smem[g*8+1], c=smem[g*8+2], d=smem[g*8+3];
            uint8_t e=smem[g*8+4], f=smem[g*8+5], gg=smem[g*8+6], h=smem[g*8+7];
            out_ptr[g*3+0] = a | (b << 3) | ((c & 3) << 6);
            out_ptr[g*3+1] = (c >> 2) | (d << 1) | (e << 4) | ((f & 1) << 7);
            out_ptr[g*3+2] = (f >> 1) | (gg << 2) | (h << 5);
        }

        constexpr int OFF_2B = 3 * D / 16;
        for (int g = lane; g < D / 8; g += 32) {
            int base = D/2 + g*4;
            out_ptr[OFF_2B + g] = smem[base] | (smem[base+1] << 2)
                                | (smem[base+2] << 4) | (smem[base+3] << 6);
        }
    }

    __syncwarp();

    // ---- Quantize V (same logic, different signs) ----
    {
        const nv_bfloat16* v_src = row + (Hq + Hkv + head_idx) * Hs;
        float vals[EPT];
        #pragma unroll
        for (int i = 0; i < EPT; i++)
            vals[i] = __bfloat162float(v_src[lane * EPT + i]);

        float nsq = 0.0f;
        #pragma unroll
        for (int i = 0; i < EPT; i++) nsq += vals[i] * vals[i];
        nsq = warp_sum(nsq);
        float norm_x = sqrtf(nsq);
        float inv_norm = (norm_x > 1e-12f) ? (1.0f / norm_x) : 0.0f;
        #pragma unroll
        for (int i = 0; i < EPT; i++) vals[i] *= inv_norm;

        uint32_t d1w = head_signs[2 * SW + lane / (32 / EPT)];
        int d1b = (lane % (32 / EPT)) * EPT;
        apply_signs<EPT>(vals, d1w, d1b);
        warp_wht<EPT>(vals, lane);

        uint8_t indices[EPT];
        float centroids[EPT];
        bool first_half = (lane < 16);
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            float v = vals[i];
            if (first_half) {
                uint8_t idx;
                if (v < 0.0f) { idx = (v < -1.050f) ? ((v < -1.748f) ? 0 : 1) : ((v < -0.501f) ? 2 : 3); }
                else          { idx = (v < 1.050f) ? ((v < 0.501f) ? 4 : 5) : ((v < 1.748f) ? 6 : 7); }
                indices[i] = idx;
                centroids[i] = kCB3[idx];
            } else {
                uint8_t idx;
                if (v < 0.0f) idx = (v < -0.982f) ? 0 : 1;
                else          idx = (v < 0.982f) ? 2 : 3;
                indices[i] = idx;
                centroids[i] = kCB2[idx];
            }
        }

        float residual[EPT];
        #pragma unroll
        for (int i = 0; i < EPT; i++) residual[i] = vals[i] - centroids[i];

        float rsq = 0.0f;
        #pragma unroll
        for (int i = 0; i < EPT; i++) rsq += residual[i] * residual[i];
        rsq = warp_sum(rsq);
        float gamma = sqrtf(rsq);

        uint32_t d2w = head_signs[3 * SW + lane / (32 / EPT)];
        int d2b = (lane % (32 / EPT)) * EPT;
        apply_signs<EPT>(residual, d2w, d2b);
        warp_wht<EPT>(residual, lane);

        uint8_t* out_ptr = v_pages
            + (long)phys_page * page_kv_stride
            + (long)page_off * Hkv * TQ_PACKED
            + (long)head_idx * TQ_PACKED;

        uint32_t ballots[EPT];
        #pragma unroll
        for (int i = 0; i < EPT; i++)
            ballots[i] = __ballot_sync(0xFFFFFFFF, residual[i] >= 0.0f);

        constexpr int OFF_QJL = 3 * D / 16 + D / 8;
        if (lane < EPT)
            reinterpret_cast<uint32_t*>(out_ptr + OFF_QJL)[lane] = ballots[lane];

        constexpr int OFF_NORMS = OFF_QJL + D / 8;
        if (lane == 0) {
            reinterpret_cast<__half*>(out_ptr + OFF_NORMS)[0] = __float2half(gamma);
            reinterpret_cast<__half*>(out_ptr + OFF_NORMS)[1] = __float2half(norm_x);
        }

        #pragma unroll
        for (int i = 0; i < EPT; i++)
            smem[lane * EPT + i] = indices[i];
        __syncwarp();

        for (int g = lane; g < D / 16; g += 32) {
            uint8_t a=smem[g*8], b=smem[g*8+1], c=smem[g*8+2], d=smem[g*8+3];
            uint8_t e=smem[g*8+4], f=smem[g*8+5], gg=smem[g*8+6], h=smem[g*8+7];
            out_ptr[g*3+0] = a | (b << 3) | ((c & 3) << 6);
            out_ptr[g*3+1] = (c >> 2) | (d << 1) | (e << 4) | ((f & 1) << 7);
            out_ptr[g*3+2] = (f >> 1) | (gg << 2) | (h << 5);
        }

        constexpr int OFF_2B = 3 * D / 16;
        for (int g = lane; g < D / 8; g += 32) {
            int base = D/2 + g*4;
            out_ptr[OFF_2B + g] = smem[base] | (smem[base+1] << 2)
                                | (smem[base+2] << 4) | (smem[base+3] << 6);
        }
    }
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

/// Compute number of splits to saturate the GPU.
/// Target: ≥4 blocks per SM, min 32 tokens per split.
static int compute_num_splits(int batch_size, int Hkv, int max_seq_len, int num_sms) {
    const int base_blocks = batch_size * Hkv;
    const int target_blocks = num_sms * 12;  // target ≥12 blocks/SM for full occupancy
    int splits = (target_blocks + base_blocks - 1) / base_blocks;
    splits = std::max(splits, 1);
    // Ensure each split has ≥32 tokens (otherwise overhead dominates)
    const int max_splits = std::max(max_seq_len / 32, 1);
    splits = std::min(splits, max_splits);
    // Round to nice numbers for even division
    if (splits > 64) splits = 64;
    else if (splits > 32) splits = 32;
    else if (splits > 16) splits = 16;
    return splits;
}

template <int HEAD_DIM, int GQA_GROUP>
static void launch_splitk(
    nv_bfloat16* out, const nv_bfloat16* q,
    const uint8_t* k_pages, const uint8_t* v_pages,
    const uint32_t* kv_signs,
    const int32_t* seq_lens,
    const int* block_table, int block_table_stride,
    int page_block_size,
    int batch_size, int Hq, int Hkv,
    float sm_scale, cudaStream_t stream, int q_stride_n,
    int num_splits,
    float* O_partial, float* ml_partial)
{
    // Pass 1: split-K attention
    dim3 grid1(batch_size, Hkv, num_splits);
    dim3 block1(32, GQA_GROUP);
    int smem1 = GQA_GROUP * HEAD_DIM;

    attention_decode_paged_tq_splitk_kernel<HEAD_DIM, GQA_GROUP>
        <<<grid1, block1, smem1, stream>>>(
            O_partial, ml_partial, q, k_pages, v_pages, kv_signs, seq_lens,
            block_table, block_table_stride, page_block_size,
            Hkv, Hq, sm_scale, q_stride_n, num_splits);

    // Pass 2: merge splits
    dim3 grid2(batch_size, Hq);
    dim3 block2(32);
    attention_reduce_splitk_kernel<HEAD_DIM>
        <<<grid2, block2, 0, stream>>>(
            out, O_partial, ml_partial, Hq, num_splits, q_stride_n);
}

template <int HEAD_DIM>
static void launch_attention_decode_paged_tq(
    nv_bfloat16* out, const nv_bfloat16* q,
    const uint8_t* k_pages, const uint8_t* v_pages,
    const uint32_t* kv_signs,
    const int32_t* seq_lens,
    const int* block_table, int block_table_stride,
    int page_block_size,
    int batch_size, int Hq, int Hkv, int Hs,
    float sm_scale, cudaStream_t stream, int q_stride_n,
    int num_splits,
    float* O_partial, float* ml_partial)
{
    const int gqa_group = Hq / Hkv;

    switch (gqa_group) {
    case 1:
        launch_splitk<HEAD_DIM, 1>(out, q, k_pages, v_pages, kv_signs, seq_lens,
            block_table, block_table_stride, page_block_size,
            batch_size, Hq, Hkv, sm_scale, stream, q_stride_n,
            num_splits, O_partial, ml_partial);
        break;
    case 4:
        launch_splitk<HEAD_DIM, 4>(out, q, k_pages, v_pages, kv_signs, seq_lens,
            block_table, block_table_stride, page_block_size,
            batch_size, Hq, Hkv, sm_scale, stream, q_stride_n,
            num_splits, O_partial, ml_partial);
        break;
    case 8:
        launch_splitk<HEAD_DIM, 8>(out, q, k_pages, v_pages, kv_signs, seq_lens,
            block_table, block_table_stride, page_block_size,
            batch_size, Hq, Hkv, sm_scale, stream, q_stride_n,
            num_splits, O_partial, ml_partial);
        break;
    default:
        launch_splitk<HEAD_DIM, 1>(out, q, k_pages, v_pages, kv_signs, seq_lens,
            block_table, block_table_stride, page_block_size,
            batch_size, Hq, Hkv, sm_scale, stream, q_stride_n,
            num_splits, O_partial, ml_partial);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace tq_decode

// ===========================================================================
// Public API (declared in attention_decode_tq.h)
// ===========================================================================

void attention_decode_paged_tq(
    nv_bfloat16* out,
    const nv_bfloat16* q,
    const uint8_t* k_pages,
    const uint8_t* v_pages,
    const uint32_t* kv_signs,
    const int32_t* seq_lens,
    const int* block_table, int block_table_stride,
    int page_block_size,
    int batch_size, int Hq, int Hkv, int Hs,
    cudaStream_t stream, int q_stride_n,
    float* scratch_O, float* scratch_ml, int num_sms)
{
    float sm_scale = 1.0f / sqrtf((float)Hs);

    // Determine split count
    // Use a heuristic max seq_len from page count (callers pass actual via seq_lens on GPU)
    int max_possible_seq = block_table_stride * page_block_size;
    if (num_sms <= 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        num_sms = prop.multiProcessorCount;
    }
    int num_splits = tq_decode::compute_num_splits(batch_size, Hkv, max_possible_seq, num_sms);

    // Allocate scratch if not provided
    float* O_partial = scratch_O;
    float* ml_partial = scratch_ml;
    bool own_scratch = (O_partial == nullptr);
    if (own_scratch) {
        size_t O_bytes  = (size_t)batch_size * Hq * num_splits * Hs * sizeof(float);
        size_t ml_bytes = (size_t)batch_size * Hq * num_splits * 2 * sizeof(float);
        CUDA_CHECK(cudaMallocAsync(&O_partial, O_bytes, stream));
        CUDA_CHECK(cudaMallocAsync(&ml_partial, ml_bytes, stream));
    }

    switch (Hs) {
    case 64:
        tq_decode::launch_attention_decode_paged_tq<64>(
            out, q, k_pages, v_pages, kv_signs, seq_lens,
            block_table, block_table_stride, page_block_size,
            batch_size, Hq, Hkv, Hs, sm_scale, stream, q_stride_n,
            num_splits, O_partial, ml_partial);
        break;
    case 128:
        tq_decode::launch_attention_decode_paged_tq<128>(
            out, q, k_pages, v_pages, kv_signs, seq_lens,
            block_table, block_table_stride, page_block_size,
            batch_size, Hq, Hkv, Hs, sm_scale, stream, q_stride_n,
            num_splits, O_partial, ml_partial);
        break;
    case 256:
        tq_decode::launch_attention_decode_paged_tq<256>(
            out, q, k_pages, v_pages, kv_signs, seq_lens,
            block_table, block_table_stride, page_block_size,
            batch_size, Hq, Hkv, Hs, sm_scale, stream, q_stride_n,
            num_splits, O_partial, ml_partial);
        break;
    }

    if (own_scratch) {
        CUDA_CHECK(cudaFreeAsync(O_partial, stream));
        CUDA_CHECK(cudaFreeAsync(ml_partial, stream));
    }
}

void kv_cache_append_paged_tq(
    uint8_t* k_pages, uint8_t* v_pages,
    const nv_bfloat16* qkv_rope,
    const uint32_t* kv_signs,
    const int* seq_lens_gpu,
    const int* block_table, int block_table_stride,
    int page_block_size,
    int batch_size, int Hq, int Hkv, int Hs,
    cudaStream_t stream)
{
    dim3 grid(batch_size, Hkv);
    dim3 block(32);

    switch (Hs) {
    case 64: {
        int smem = 64;
        tq_decode::kv_cache_append_paged_tq_kernel<2><<<grid, block, smem, stream>>>(
            k_pages, v_pages, qkv_rope, kv_signs, seq_lens_gpu,
            block_table, block_table_stride, page_block_size,
            Hq, Hkv, Hs);
        break;
    }
    case 128: {
        int smem = 128;
        tq_decode::kv_cache_append_paged_tq_kernel<4><<<grid, block, smem, stream>>>(
            k_pages, v_pages, qkv_rope, kv_signs, seq_lens_gpu,
            block_table, block_table_stride, page_block_size,
            Hq, Hkv, Hs);
        break;
    }
    case 256: {
        int smem = 256;
        tq_decode::kv_cache_append_paged_tq_kernel<8><<<grid, block, smem, stream>>>(
            k_pages, v_pages, qkv_rope, kv_signs, seq_lens_gpu,
            block_table, block_table_stride, page_block_size,
            Hq, Hkv, Hs);
        break;
    }
    }
    CUDA_CHECK(cudaGetLastError());
}
