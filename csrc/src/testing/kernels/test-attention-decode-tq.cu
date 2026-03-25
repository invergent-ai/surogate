// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

/**
 * Tests for fused TurboQuant 3.5-bit paged decode attention.
 * Validates:
 *   1. KV append + fused attention round-trip correctness vs BF16 reference
 *   2. Performance benchmark vs FP8 per-block dequant + attention
 */

#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "kernels/attention_decode_tq.h"
#include "utilities/utils.h"
#include "../utilities/test_utils.h"

using namespace testing_utils;

namespace {

std::vector<float> randn(int n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

/// CPU reference: BF16 decode attention (single query attending over seq_len keys).
/// q: [Hq, D], k: [seq_len, Hkv, D], v: [seq_len, Hkv, D], out: [Hq, D]
/// GQA: Q head h attends to KV head h / gqa_group.
void cpu_decode_attention(
    float* out, const float* q, const float* k, const float* v,
    int seq_len, int Hq, int Hkv, int D)
{
    int gqa_group = Hq / Hkv;
    float sm_scale = 1.0f / std::sqrt((float)D);

    for (int qh = 0; qh < Hq; qh++) {
        int kvh = qh / gqa_group;
        const float* q_head = q + qh * D;

        // Compute scores
        std::vector<float> scores(seq_len);
        float max_score = -1e30f;
        for (int t = 0; t < seq_len; t++) {
            float s = 0.0f;
            for (int d = 0; d < D; d++)
                s += q_head[d] * k[t * Hkv * D + kvh * D + d];
            scores[t] = s * sm_scale;
            max_score = std::max(max_score, scores[t]);
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            scores[t] = std::exp(scores[t] - max_score);
            sum_exp += scores[t];
        }
        for (int t = 0; t < seq_len; t++)
            scores[t] /= sum_exp;

        // Weighted V sum
        float* o = out + qh * D;
        for (int d = 0; d < D; d++) o[d] = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            for (int d = 0; d < D; d++)
                o[d] += scores[t] * v[t * Hkv * D + kvh * D + d];
        }
    }
}

}  // namespace

TEST_CASE("Fused TQ decode attention correctness (D=128)", "[tq-attention]") {
    constexpr int D = 128;
    constexpr int Hkv = 4;
    constexpr int Hq = 16;       // GQA group = 4
    constexpr int SEQ_LEN = 64;  // Short for correctness test
    constexpr int BATCH = 2;
    constexpr int PAGE_BLOCK = 256;
    constexpr int TQ_PACKED = tq_packed_size(D);

    // Pages needed: ceil(SEQ_LEN / PAGE_BLOCK) = 1 per sequence
    constexpr int PAGES_PER_SEQ = 1;
    constexpr int TOTAL_PAGES = BATCH * PAGES_PER_SEQ;

    // Generate random Q and K/V data
    auto h_q_f = randn(BATCH * Hq * D, 42);
    auto h_k_f = randn(BATCH * SEQ_LEN * Hkv * D, 100);
    auto h_v_f = randn(BATCH * SEQ_LEN * Hkv * D, 200);

    // CPU reference attention
    std::vector<float> h_ref_out(BATCH * Hq * D);
    for (int b = 0; b < BATCH; b++) {
        cpu_decode_attention(
            h_ref_out.data() + b * Hq * D,
            h_q_f.data() + b * Hq * D,
            h_k_f.data() + b * SEQ_LEN * Hkv * D,
            h_v_f.data() + b * SEQ_LEN * Hkv * D,
            SEQ_LEN, Hq, Hkv, D);
    }

    // Generate random signs
    int signs_total = Hkv * 4 * (D / 32);
    std::vector<uint32_t> h_signs(signs_total);
    tq_generate_kv_signs_host(h_signs.data(), Hkv, D, 0xABCD);

    // Allocate page pools
    long page_kv_stride = (long)PAGE_BLOCK * Hkv * TQ_PACKED;
    long pool_bytes = (long)TOTAL_PAGES * page_kv_stride;

    thrust::device_vector<uint8_t> d_k_pages(pool_bytes, 0);
    thrust::device_vector<uint8_t> d_v_pages(pool_bytes, 0);
    thrust::device_vector<uint32_t> d_signs(h_signs.begin(), h_signs.end());

    // Block table: identity mapping (page i = physical page i)
    std::vector<int> h_block_table(BATCH * PAGES_PER_SEQ);
    for (int b = 0; b < BATCH; b++)
        h_block_table[b * PAGES_PER_SEQ] = b;  // page 0 of batch b = physical page b
    thrust::device_vector<int> d_block_table(h_block_table.begin(), h_block_table.end());

    // Populate KV cache: append token by token using kv_cache_append_paged_tq
    // Build QKV buffer: [1, 1, Hq+2*Hkv, D] per token
    int H_total = Hq + 2 * Hkv;
    std::vector<float> h_qkv(H_total * D);

    for (int b = 0; b < BATCH; b++) {
        // seq_lens tracks current position
        std::vector<int> h_seq_lens = {0};  // one batch at a time
        thrust::device_vector<int> d_seq_lens(1);

        for (int t = 0; t < SEQ_LEN; t++) {
            // Build QKV: Q is garbage (not used for append), K and V from test data
            std::fill(h_qkv.begin(), h_qkv.end(), 0.0f);
            for (int h = 0; h < Hkv; h++) {
                for (int d = 0; d < D; d++) {
                    h_qkv[(Hq + h) * D + d] = h_k_f[b * SEQ_LEN * Hkv * D + t * Hkv * D + h * D + d];
                    h_qkv[(Hq + Hkv + h) * D + d] = h_v_f[b * SEQ_LEN * Hkv * D + t * Hkv * D + h * D + d];
                }
            }

            auto h_qkv_bf = to_bf16(h_qkv);
            thrust::device_vector<nv_bfloat16> d_qkv(h_qkv_bf.begin(), h_qkv_bf.end());

            h_seq_lens[0] = t;
            thrust::copy(h_seq_lens.begin(), h_seq_lens.end(), d_seq_lens.begin());

            // Adjust block table for single-batch append
            std::vector<int> h_bt_single = {(int)(b * PAGES_PER_SEQ)};
            // Actually use the batch's physical page
            h_bt_single[0] = b;
            thrust::device_vector<int> d_bt_single(h_bt_single.begin(), h_bt_single.end());

            kv_cache_append_paged_tq(
                thrust::raw_pointer_cast(d_k_pages.data()),
                thrust::raw_pointer_cast(d_v_pages.data()),
                thrust::raw_pointer_cast(d_qkv.data()),
                thrust::raw_pointer_cast(d_signs.data()),
                thrust::raw_pointer_cast(d_seq_lens.data()),
                thrust::raw_pointer_cast(d_bt_single.data()),
                1,  // block_table_stride
                PAGE_BLOCK,
                1,   // batch_size = 1 (appending one batch at a time)
                Hq, Hkv, D,
                0);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // Now run fused decode attention
    auto h_q_bf = to_bf16(h_q_f);
    thrust::device_vector<nv_bfloat16> d_q(h_q_bf.begin(), h_q_bf.end());
    thrust::device_vector<nv_bfloat16> d_out(BATCH * Hq * D);

    std::vector<int32_t> h_seq_lens_all(BATCH, SEQ_LEN);
    thrust::device_vector<int32_t> d_seq_lens_all(h_seq_lens_all.begin(), h_seq_lens_all.end());

    attention_decode_paged_tq(
        thrust::raw_pointer_cast(d_out.data()),
        thrust::raw_pointer_cast(d_q.data()),
        thrust::raw_pointer_cast(d_k_pages.data()),
        thrust::raw_pointer_cast(d_v_pages.data()),
        thrust::raw_pointer_cast(d_signs.data()),
        thrust::raw_pointer_cast(d_seq_lens_all.data()),
        thrust::raw_pointer_cast(d_block_table.data()),
        PAGES_PER_SEQ,
        PAGE_BLOCK,
        BATCH, Hq, Hkv, D,
        0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back and compare
    std::vector<nv_bfloat16> h_out_bf(BATCH * Hq * D);
    thrust::copy(d_out.begin(), d_out.end(), h_out_bf.begin());

    // Compute cosine similarity per head
    double total_cosine = 0.0;
    int num_heads = BATCH * Hq;
    for (int i = 0; i < num_heads; i++) {
        double dot = 0, na = 0, nb = 0;
        for (int d = 0; d < D; d++) {
            float ref = h_ref_out[i * D + d];
            float got = bf16_bits_to_float(*reinterpret_cast<uint16_t*>(&h_out_bf[i * D + d]));
            dot += (double)ref * got;
            na += (double)ref * ref;
            nb += (double)got * got;
        }
        double cosine = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
        total_cosine += cosine;
    }
    double avg_cosine = total_cosine / num_heads;

    INFO("Fused TQ attention vs CPU BF16 reference:");
    INFO("  Average cosine similarity: " << avg_cosine);

    // TQ 3.5-bit will have some distortion, but attention output should be close
    REQUIRE(avg_cosine > 0.95);
}

TEST_CASE("Fused TQ decode attention benchmark (D=128)", "[tq-attention][benchmark]") {
    constexpr int D = 128;
    constexpr int Hkv = 8;
    constexpr int Hq = 32;       // GQA group = 4
    constexpr int SEQ_LEN = 4096;
    constexpr int BATCH = 8;
    constexpr int PAGE_BLOCK = 256;
    constexpr int TQ_PACKED = tq_packed_size(D);
    constexpr int WARMUP = 20;
    constexpr int ITERS = 100;

    constexpr int PAGES_PER_SEQ = (SEQ_LEN + PAGE_BLOCK - 1) / PAGE_BLOCK;
    constexpr int TOTAL_PAGES = BATCH * PAGES_PER_SEQ;

    // Allocate (don't need real data — just measuring kernel time)
    long page_kv_stride = (long)PAGE_BLOCK * Hkv * TQ_PACKED;
    long pool_bytes = (long)TOTAL_PAGES * page_kv_stride;

    thrust::device_vector<uint8_t> d_k_pages(pool_bytes, 0);
    thrust::device_vector<uint8_t> d_v_pages(pool_bytes, 0);

    int signs_total = Hkv * 4 * (D / 32);
    std::vector<uint32_t> h_signs(signs_total);
    tq_generate_kv_signs_host(h_signs.data(), Hkv, D, 0x1234);
    thrust::device_vector<uint32_t> d_signs(h_signs.begin(), h_signs.end());

    std::vector<int> h_bt(BATCH * PAGES_PER_SEQ);
    for (int b = 0; b < BATCH; b++)
        for (int p = 0; p < PAGES_PER_SEQ; p++)
            h_bt[b * PAGES_PER_SEQ + p] = b * PAGES_PER_SEQ + p;
    thrust::device_vector<int> d_bt(h_bt.begin(), h_bt.end());

    std::vector<int32_t> h_sl(BATCH, SEQ_LEN);
    thrust::device_vector<int32_t> d_sl(h_sl.begin(), h_sl.end());

    thrust::device_vector<nv_bfloat16> d_q(BATCH * Hq * D);
    thrust::device_vector<nv_bfloat16> d_out(BATCH * Hq * D);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < WARMUP; i++)
        attention_decode_paged_tq(
            thrust::raw_pointer_cast(d_out.data()),
            thrust::raw_pointer_cast(d_q.data()),
            thrust::raw_pointer_cast(d_k_pages.data()),
            thrust::raw_pointer_cast(d_v_pages.data()),
            thrust::raw_pointer_cast(d_signs.data()),
            thrust::raw_pointer_cast(d_sl.data()),
            thrust::raw_pointer_cast(d_bt.data()),
            PAGES_PER_SEQ, PAGE_BLOCK,
            BATCH, Hq, Hkv, D, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < ITERS; i++)
        attention_decode_paged_tq(
            thrust::raw_pointer_cast(d_out.data()),
            thrust::raw_pointer_cast(d_q.data()),
            thrust::raw_pointer_cast(d_k_pages.data()),
            thrust::raw_pointer_cast(d_v_pages.data()),
            thrust::raw_pointer_cast(d_signs.data()),
            thrust::raw_pointer_cast(d_sl.data()),
            thrust::raw_pointer_cast(d_bt.data()),
            PAGES_PER_SEQ, PAGE_BLOCK,
            BATCH, Hq, Hkv, D, 0);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= ITERS;

    // Compute effective bandwidth
    size_t tq_read = (size_t)BATCH * SEQ_LEN * Hkv * TQ_PACKED * 2;  // K+V
    size_t q_read  = (size_t)BATCH * Hq * D * 2;
    size_t total_bytes = tq_read + q_read;
    double bw = total_bytes / (ms * 1e-3) / 1e9;

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    INFO("=== Fused TQ Decode Attention Benchmark (" << props.name << ") ===");
    INFO("  Batch=" << BATCH << " SeqLen=" << SEQ_LEN
         << " Hq=" << Hq << " Hkv=" << Hkv << " D=" << D);
    INFO("  Time: " << ms << " ms");
    INFO("  KV read: " << tq_read / 1048576.0 << " MB (TQ 3.5-bit)");
    INFO("  Effective BW: " << bw << " GB/s");
    INFO("  Equivalent FP8 read would be: "
         << (size_t)BATCH * SEQ_LEN * Hkv * D * 2 / 1048576.0 << " MB");

    REQUIRE(ms > 0);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
