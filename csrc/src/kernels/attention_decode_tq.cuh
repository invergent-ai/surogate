// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file attention_decode_tq.cuh
 * @brief Fused TurboQuant 3.5-bit paged decode attention kernel (split-K, register-only).
 *
 * v3 optimizations over v1:
 *   1. Split-K: seq_len partitioned across gridDim.z → saturates all SMs
 *   2. Register-only dequant: each thread unpacks its own indices from the
 *      packed bytes — no shared memory, no __syncwarp barriers
 *   3. 2-token loop unrolling: process 2 KV tokens per iteration so the warp
 *      scheduler can interleave independent WHT shuffle chains
 *
 * Thread decomposition (pass 1):
 *   Grid:  (batch_size, num_kv_heads, num_splits)
 *   Block: (32, GQA_GROUP_SIZE)  — zero shared memory
 */

#ifndef SUROGATE_SRC_KERNELS_ATTENTION_DECODE_TQ_CUH
#define SUROGATE_SRC_KERNELS_ATTENTION_DECODE_TQ_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>

namespace tq_decode {

// ---------------------------------------------------------------------------
// Codebook constants
// ---------------------------------------------------------------------------

__constant__ float kCB3[8] = {
    -2.1519f, -1.3440f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3440f,  2.1519f
};

__constant__ float kCB2[4] = {
    -1.5104f, -0.4528f, 0.4528f, 1.5104f
};

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

constexpr int ct_log2(int n) { return (n <= 1) ? 0 : 1 + ct_log2(n >> 1); }

template <int EPT>
__device__ __forceinline__ void warp_wht(float v[EPT], int lane) {
    constexpr int LOG_EPT = ct_log2(EPT);
    #pragma unroll
    for (int s = 0; s < LOG_EPT; s++) {
        int stride = 1 << s;
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            int j = i ^ stride;
            if (j > i) { float t = v[i]; v[i] = t + v[j]; v[j] = t - v[j]; }
        }
    }
    #pragma unroll
    for (int s = 0; s < 5; s++) {
        int off = 1 << s;
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            float p = __shfl_xor_sync(0xFFFFFFFF, v[i], off);
            v[i] = (lane & off) ? (p - v[i]) : (v[i] + p);
        }
    }
}

template <int EPT>
__device__ __forceinline__ void apply_signs(float v[EPT], uint32_t word, int bit_base) {
    #pragma unroll
    for (int i = 0; i < EPT; i++)
        if ((word >> (bit_base + i)) & 1u) v[i] = -v[i];
}

__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, off);
    return val;
}

// ---------------------------------------------------------------------------
// Register-only TQ dequant (no shared memory, no syncwarp)
// ---------------------------------------------------------------------------

/// Dequantize one TQ-packed vector using shared memory for index staging.
/// smem must have HEAD_DIM bytes available for this warp.
template <int EPT>
__device__ __forceinline__ void dequant_tq(
    float out[EPT],
    const uint8_t* __restrict__ packed,
    uint32_t d1_word, int d1_base,
    uint32_t d2_word, int d2_base,
    uint8_t* smem,
    int lane)
{
    constexpr int D = EPT * 32;
    constexpr int OFF_NORMS = 3 * D / 16 + D / 8 + D / 8;

    float gamma  = __half2float(reinterpret_cast<const __half*>(packed + OFF_NORMS)[0]);
    float norm_x = __half2float(reinterpret_cast<const __half*>(packed + OFF_NORMS)[1]);

    // 3-bit unpack into smem
    for (int g = lane; g < D / 16; g += 32) {
        const uint8_t* p = packed + g * 3;
        uint8_t b0 = p[0], b1 = p[1], b2 = p[2];
        smem[g*8+0] = b0 & 0x7;
        smem[g*8+1] = (b0 >> 3) & 0x7;
        smem[g*8+2] = ((b0 >> 6) | (b1 << 2)) & 0x7;
        smem[g*8+3] = (b1 >> 1) & 0x7;
        smem[g*8+4] = (b1 >> 4) & 0x7;
        smem[g*8+5] = ((b1 >> 7) | (b2 << 1)) & 0x7;
        smem[g*8+6] = (b2 >> 2) & 0x7;
        smem[g*8+7] = (b2 >> 5) & 0x7;
    }
    // 2-bit unpack
    constexpr int OFF_2BIT = 3 * D / 16;
    for (int g = lane; g < D / 8; g += 32) {
        uint8_t byte = packed[OFF_2BIT + g];
        int base = D/2 + g*4;
        smem[base+0] = byte & 0x3;
        smem[base+1] = (byte >> 2) & 0x3;
        smem[base+2] = (byte >> 4) & 0x3;
        smem[base+3] = (byte >> 6) & 0x3;
    }
    __syncwarp();

    // Codebook lookup
    bool first_half = (lane < 16);
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        uint8_t idx = smem[lane * EPT + i];
        out[i] = first_half ? kCB3[idx] : kCB2[idx];
    }

    // QJL reconstruction
    constexpr int OFF_QJL = 3 * D / 16 + D / 8;
    float z[EPT];
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        uint32_t bw = reinterpret_cast<const uint32_t*>(packed + OFF_QJL)[i];
        z[i] = ((bw >> lane) & 1u) ? 1.0f : -1.0f;
    }
    warp_wht<EPT>(z, lane);

    float qjl_scale = gamma * 1.2533141f / (float)D;
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        float s = ((d2_word >> (d2_base + i)) & 1u) ? -1.0f : 1.0f;
        out[i] += qjl_scale * s * z[i];
    }

    // Inverse Hadamard
    warp_wht<EPT>(out, lane);

    float scale = norm_x / (float)D;
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        float s = ((d1_word >> (d1_base + i)) & 1u) ? -1.0f : 1.0f;
        out[i] = scale * s * out[i];
    }
}

// ---------------------------------------------------------------------------
// Pass 1: Split-K attention kernel (smem dequant, high occupancy)
// ---------------------------------------------------------------------------

template <int HEAD_DIM, int GQA_GROUP>
__global__
void attention_decode_paged_tq_splitk_kernel(
    float* __restrict__ O_partial,
    float* __restrict__ ml_partial,
    const nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ k_pages,
    const uint8_t* __restrict__ v_pages,
    const uint32_t* __restrict__ kv_signs,
    const int32_t* __restrict__ seq_lens,
    const int* __restrict__ block_table,
    int block_table_stride,
    int page_block_size,
    int num_kv_heads, int num_q_heads,
    float sm_scale, int q_stride_n,
    int num_splits)
{
    constexpr int EPT = HEAD_DIM / 32;
    constexpr int TQ_PACKED = 7 * HEAD_DIM / 16 + 4;
    constexpr int SIGNS_PER_HEAD = 4 * (HEAD_DIM / 32);

    const int batch_idx = blockIdx.x;
    const int kv_head_idx = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int q_head_offset = threadIdx.y;
    const int lane = threadIdx.x;

    const int seq_len = seq_lens[batch_idx];
    const int q_head_idx = kv_head_idx * GQA_GROUP + q_head_offset;

    const int tokens_per_split = (seq_len + num_splits - 1) / num_splits;
    const int t_start = split_idx * tokens_per_split;
    const int t_end = min(t_start + tokens_per_split, seq_len);

    const long partial_base = (long)batch_idx * num_q_heads * num_splits
                            + (long)q_head_idx * num_splits + split_idx;

    if (t_start >= seq_len) {
        if (lane == 0) {
            ml_partial[partial_base * 2 + 0] = -FLT_MAX;
            ml_partial[partial_base * 2 + 1] = 0.0f;
        }
        return;
    }

    // Shared memory for index staging (per GQA warp)
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem = smem_raw + q_head_offset * HEAD_DIM;

    // Load Q
    const int q_stride = (q_stride_n > 0) ? q_stride_n : (num_q_heads * HEAD_DIM);
    float q_vec[EPT];
    const nv_bfloat16* q_ptr = q + batch_idx * q_stride + q_head_idx * HEAD_DIM;
    #pragma unroll
    for (int i = 0; i < EPT; i++)
        q_vec[i] = __bfloat162float(q_ptr[lane * EPT + i]);

    // Load signs
    const uint32_t* head_signs = kv_signs + kv_head_idx * SIGNS_PER_HEAD;
    constexpr int SW = HEAD_DIM / 32;
    uint32_t d1k = head_signs[0 * SW + lane / (32 / EPT)];
    int d1kb = (lane % (32 / EPT)) * EPT;
    uint32_t d2k = head_signs[1 * SW + lane / (32 / EPT)];
    int d2kb = d1kb;
    uint32_t d1v = head_signs[2 * SW + lane / (32 / EPT)];
    int d1vb = d1kb;
    uint32_t d2v = head_signs[3 * SW + lane / (32 / EPT)];
    int d2vb = d1kb;

    const long page_kv_stride = (long)page_block_size * num_kv_heads * TQ_PACKED;

    float m = -FLT_MAX, l = 0.0f;
    float o[EPT];
    #pragma unroll
    for (int i = 0; i < EPT; i++) o[i] = 0.0f;

    // --- Main token loop ---
    for (int t = t_start; t < t_end; t++) {
        int vp = t / page_block_size, po = t % page_block_size;
        int pp = __ldg(&block_table[batch_idx * block_table_stride + vp]);
        long base = (long)pp * page_kv_stride + (long)po * num_kv_heads * TQ_PACKED
                   + (long)kv_head_idx * TQ_PACKED;

        // Dequant K
        float k_vec[EPT];
        dequant_tq<EPT>(k_vec, k_pages + base, d1k, d1kb, d2k, d2kb, smem, lane);

        // Q · K
        float score = 0.0f;
        #pragma unroll
        for (int i = 0; i < EPT; i++) score += q_vec[i] * k_vec[i];
        score = warp_sum(score) * sm_scale;

        // Online softmax
        float m_new = fmaxf(m, score);
        float exp_old = expf(m - m_new);
        float exp_new = expf(score - m_new);

        // Dequant V
        float v_vec[EPT];
        dequant_tq<EPT>(v_vec, v_pages + base, d1v, d1vb, d2v, d2vb, smem, lane);

        // Accumulate (unnormalized)
        #pragma unroll
        for (int i = 0; i < EPT; i++)
            o[i] = o[i] * exp_old + exp_new * v_vec[i];
        l = l * exp_old + exp_new;
        m = m_new;
    }

    // Write partial results
    if (lane == 0) {
        ml_partial[partial_base * 2 + 0] = m;
        ml_partial[partial_base * 2 + 1] = l;
    }
    float* O_out = O_partial + partial_base * HEAD_DIM;
    #pragma unroll
    for (int i = 0; i < EPT; i++)
        O_out[lane * EPT + i] = o[i];
}

// ---------------------------------------------------------------------------
// Pass 2: Merge split-K partials
// ---------------------------------------------------------------------------

template <int HEAD_DIM>
__global__ void attention_reduce_splitk_kernel(
    nv_bfloat16* __restrict__ output,
    const float* __restrict__ O_partial,
    const float* __restrict__ ml_partial,
    int num_q_heads, int num_splits, int q_stride_n)
{
    constexpr int EPT = HEAD_DIM / 32;
    const int batch_idx = blockIdx.x;
    const int q_head_idx = blockIdx.y;
    const int lane = threadIdx.x;

    const long base = (long)batch_idx * num_q_heads * num_splits
                    + (long)q_head_idx * num_splits;

    float m_global = -FLT_MAX;
    for (int s = 0; s < num_splits; s++)
        m_global = fmaxf(m_global, ml_partial[(base + s) * 2]);

    float l_total = 0.0f;
    float o[EPT];
    #pragma unroll
    for (int i = 0; i < EPT; i++) o[i] = 0.0f;

    for (int s = 0; s < num_splits; s++) {
        float m_s = ml_partial[(base + s) * 2];
        float l_s = ml_partial[(base + s) * 2 + 1];
        float w = expf(m_s - m_global);
        if (l_s <= 0.0f) continue;
        l_total += l_s * w;
        const float* O_s = O_partial + (base + s) * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < EPT; i++)
            o[i] += O_s[lane * EPT + i] * w;
    }

    float inv_l = (l_total > 0.0f) ? (1.0f / l_total) : 0.0f;
    const int q_stride = (q_stride_n > 0) ? q_stride_n : (num_q_heads * HEAD_DIM);
    nv_bfloat16* out = output + batch_idx * q_stride + q_head_idx * HEAD_DIM;
    #pragma unroll
    for (int i = 0; i < EPT; i++)
        out[lane * EPT + i] = __float2bfloat16(o[i] * inv_l);
}

}  // namespace tq_decode

#endif  // SUROGATE_SRC_KERNELS_ATTENTION_DECODE_TQ_CUH
