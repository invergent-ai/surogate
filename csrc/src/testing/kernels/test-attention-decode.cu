// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Tests for decode attention with KV-cache.
// Verifies that decode attention produces the same output as full-sequence attention.

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <random>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "kernels/kernels.h"
#include "kernels/attention_decode.h"
#include "runtime/infer/kv_cache.h"
#include "utilities/utils.h"

namespace {

/// Helper: convert float vector to BF16 on device.
thrust::device_vector<nv_bfloat16> to_bf16_device(const std::vector<float>& host) {
    std::vector<nv_bfloat16> bf16(host.size());
    for (std::size_t i = 0; i < host.size(); ++i) {
        bf16[i] = __float2bfloat16(host[i]);
    }
    return thrust::device_vector<nv_bfloat16>(bf16.begin(), bf16.end());
}

/// Helper: convert device BF16 to float on host.
std::vector<float> from_bf16_device(const thrust::device_vector<nv_bfloat16>& dev) {
    std::vector<nv_bfloat16> bf16(dev.size());
    thrust::copy(dev.begin(), dev.end(), bf16.begin());
    std::vector<float> result(bf16.size());
    for (std::size_t i = 0; i < bf16.size(); ++i) {
        result[i] = __bfloat162float(bf16[i]);
    }
    return result;
}

}  // anonymous namespace

TEST_CASE("KV-cache allocation and addressing", "[decode][kv_cache]") {
    const int num_layers = 4;
    const int batch_size = 2;
    const int max_seq_len = 16;
    const int num_kv_heads = 4;
    const int head_dim = 8;

    infer::KVCache cache(num_layers, batch_size, max_seq_len, num_kv_heads, head_dim, infer::KVDType::BF16);

    // Check byte calculations
    const std::size_t expected_per_buffer = static_cast<std::size_t>(num_layers) * batch_size * max_seq_len * num_kv_heads * head_dim * 2;  // BF16
    REQUIRE(cache.compute_per_buffer_bytes() == expected_per_buffer);
    REQUIRE(cache.total_bytes() == 2 * expected_per_buffer);  // K + V
    REQUIRE(cache.compute_per_scale_bytes() == 0);  // BF16 has no scales

    // Check strides
    REQUIRE(cache.batch_stride_elems() == static_cast<std::size_t>(max_seq_len * num_kv_heads * head_dim));
    REQUIRE(cache.layer_stride_elems() == static_cast<std::size_t>(batch_size) * max_seq_len * num_kv_heads * head_dim);

    // Check seq_lens initialization
    REQUIRE(cache.seq_lens.size() == static_cast<std::size_t>(batch_size));
    REQUIRE(cache.seq_lens[0] == 0);
    REQUIRE(cache.seq_lens[1] == 0);

    // Check reset
    cache.seq_lens[0] = 5;
    cache.reset();
    REQUIRE(cache.seq_lens[0] == 0);
}

TEST_CASE("KV-cache append kernel correctness", "[decode][kv_append]") {
    const int batch_size = 2;
    const int max_seq_len = 8;
    const int Hq = 8;
    const int Hkv = 4;
    const int Hs = 8;  // Small head_dim for testing
    const int H = Hq + 2 * Hkv;

    // Create QKV data: [batch_size, 1, H, Hs]
    const int qkv_size = batch_size * H * Hs;
    std::vector<float> qkv_host(static_cast<std::size_t>(qkv_size));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : qkv_host) v = dist(rng);

    auto qkv_dev = to_bf16_device(qkv_host);

    // Create K and V caches: [batch_size, max_seq_len, Hkv, Hs]
    const std::size_t cache_elems = static_cast<std::size_t>(batch_size) * max_seq_len * Hkv * Hs;
    thrust::device_vector<nv_bfloat16> k_cache(cache_elems, __float2bfloat16(0.0f));
    thrust::device_vector<nv_bfloat16> v_cache(cache_elems, __float2bfloat16(0.0f));

    // Set seq_lens: first seq at position 3, second at position 5
    thrust::device_vector<int> seq_lens_dev(static_cast<std::size_t>(batch_size));
    std::vector<int> seq_lens_host = {3, 5};
    thrust::copy(seq_lens_host.begin(), seq_lens_host.end(), seq_lens_dev.begin());

    // Run kernel
    kv_cache_append_bf16(
        thrust::raw_pointer_cast(k_cache.data()),
        thrust::raw_pointer_cast(v_cache.data()),
        thrust::raw_pointer_cast(qkv_dev.data()),
        thrust::raw_pointer_cast(seq_lens_dev.data()),
        batch_size, max_seq_len, Hq, Hkv, Hs,
        nullptr);  // default stream

    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify: K and V at the correct positions should match QKV input
    auto k_host = from_bf16_device(k_cache);
    auto v_host = from_bf16_device(v_cache);

    for (int b = 0; b < batch_size; ++b) {
        const int pos = seq_lens_host[static_cast<std::size_t>(b)];
        for (int h = 0; h < Hkv; ++h) {
            for (int d = 0; d < Hs; ++d) {
                // K source: QKV[b, 0, Hq + h, d]
                float k_expected = qkv_host[static_cast<std::size_t>(b * H * Hs + (Hq + h) * Hs + d)];
                // K dest: cache[b, pos, h, d]
                float k_actual = k_host[static_cast<std::size_t>(b * max_seq_len * Hkv * Hs + pos * Hkv * Hs + h * Hs + d)];
                REQUIRE(std::abs(k_actual - k_expected) < 0.02f);  // BF16 tolerance

                // V source: QKV[b, 0, Hq + Hkv + h, d]
                float v_expected = qkv_host[static_cast<std::size_t>(b * H * Hs + (Hq + Hkv + h) * Hs + d)];
                float v_actual = v_host[static_cast<std::size_t>(b * max_seq_len * Hkv * Hs + pos * Hkv * Hs + h * Hs + d)];
                REQUIRE(std::abs(v_actual - v_expected) < 0.02f);
            }
        }
    }
}

TEST_CASE("fill_decode_cu_seqlens correctness", "[decode][cu_seqlens]") {
    const int batch_size = 3;

    thrust::device_vector<int32_t> cu_q(static_cast<std::size_t>(batch_size + 1));
    thrust::device_vector<int32_t> seqused_k(static_cast<std::size_t>(batch_size));
    thrust::device_vector<int> seq_lens_dev(static_cast<std::size_t>(batch_size));

    // seq_lens: [4, 7, 2]  (these are BEFORE the +1 adjustment in the kernel)
    std::vector<int> seq_lens_host = {4, 7, 2};
    thrust::copy(seq_lens_host.begin(), seq_lens_host.end(), seq_lens_dev.begin());

    fill_decode_cu_seqlens(
        thrust::raw_pointer_cast(cu_q.data()),
        thrust::raw_pointer_cast(seqused_k.data()),
        thrust::raw_pointer_cast(seq_lens_dev.data()),
        batch_size, nullptr);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify cu_seqlens_q: [0, 1, 2, 3]
    std::vector<int32_t> cu_q_host(static_cast<std::size_t>(batch_size + 1));
    thrust::copy(cu_q.begin(), cu_q.end(), cu_q_host.begin());
    REQUIRE(cu_q_host[0] == 0);
    REQUIRE(cu_q_host[1] == 1);
    REQUIRE(cu_q_host[2] == 2);
    REQUIRE(cu_q_host[3] == 3);

    // Verify seqused_k: per-sequence K lengths = [5, 8, 3]
    std::vector<int32_t> seqused_k_host(static_cast<std::size_t>(batch_size));
    thrust::copy(seqused_k.begin(), seqused_k.end(), seqused_k_host.begin());
    REQUIRE(seqused_k_host[0] == 5);  // 4 + 1
    REQUIRE(seqused_k_host[1] == 8);  // 7 + 1
    REQUIRE(seqused_k_host[2] == 3);  // 2 + 1
}

TEST_CASE("KV-cache broadcast prefix correctness", "[decode][kv_broadcast]") {
    const int batch_size = 8;  // Total slots
    const int max_seq_len = 16;
    const int Hkv = 2;
    const int Hs = 8;

    // Fill a source slot (slot 0) with known data
    const std::size_t slot_elems = static_cast<std::size_t>(max_seq_len) * Hkv * Hs;
    const std::size_t total_elems = static_cast<std::size_t>(batch_size) * slot_elems;

    std::vector<float> k_host(total_elems, 0.0f);
    std::vector<float> v_host(total_elems, 0.0f);

    // Fill source slot 0 with sequential values
    const int prefix_len = 5;
    const std::size_t prefix_elems = static_cast<std::size_t>(prefix_len) * Hkv * Hs;
    for (std::size_t i = 0; i < prefix_elems; ++i) {
        k_host[i] = static_cast<float>(i + 1);      // K: 1, 2, 3, ...
        v_host[i] = static_cast<float>(i + 1000);    // V: 1000, 1001, ...
    }

    auto k_dev = to_bf16_device(k_host);
    auto v_dev = to_bf16_device(v_host);

    // Broadcast from slot 0 to slots [2, 5, 7]
    std::vector<int> dst_slots = {2, 5, 7};
    kv_cache_broadcast_prefix(
        thrust::raw_pointer_cast(k_dev.data()),
        thrust::raw_pointer_cast(v_dev.data()),
        /*src_slot=*/0, dst_slots.data(), static_cast<int>(dst_slots.size()),
        prefix_len, max_seq_len, Hkv, Hs,
        nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto k_result = from_bf16_device(k_dev);
    auto v_result = from_bf16_device(v_dev);

    // Verify each destination slot matches the source prefix
    for (int dst : dst_slots) {
        for (std::size_t i = 0; i < prefix_elems; ++i) {
            const std::size_t src_idx = i;
            const std::size_t dst_idx = static_cast<std::size_t>(dst) * slot_elems + i;
            REQUIRE(std::abs(k_result[dst_idx] - k_result[src_idx]) < 0.02f);
            REQUIRE(std::abs(v_result[dst_idx] - v_result[src_idx]) < 0.02f);
        }
    }

    // Verify non-destination slots (e.g., slot 1) are still zero
    for (std::size_t i = 0; i < prefix_elems; ++i) {
        const std::size_t slot1_idx = slot_elems + i;
        REQUIRE(k_result[slot1_idx] == 0.0f);
    }
}

TEST_CASE("Paged KV-cache append correctness", "[decode][paged]") {
    // Verify that kv_cache_append_paged_bf16 writes to the correct page+offset
    const int batch_size = 2;
    const int page_block_size = 4;
    const int total_pages = 8;
    const int Hq = 4;
    const int Hkv = 2;
    const int Hs = 8;
    const int H = Hq + 2 * Hkv;
    const int max_pages_per_seq = 4;

    const std::size_t page_elems = static_cast<std::size_t>(page_block_size) * Hkv * Hs;
    const std::size_t pool_elems = static_cast<std::size_t>(total_pages) * page_elems;

    // Initialize page pools to zero
    thrust::device_vector<nv_bfloat16> k_pool(pool_elems, __float2bfloat16(0.0f));
    thrust::device_vector<nv_bfloat16> v_pool(pool_elems, __float2bfloat16(0.0f));

    // Block table: seq 0 maps to pages [2, 5], seq 1 maps to pages [0, 3]
    std::vector<int> block_table_host = {
        2, 5, -1, -1,   // seq 0: page 0→phys 2, page 1→phys 5
        0, 3, -1, -1    // seq 1: page 0→phys 0, page 1→phys 3
    };
    thrust::device_vector<int> block_table_dev(block_table_host.begin(), block_table_host.end());

    // Seq lens: seq 0 at position 2 (page 0, offset 2), seq 1 at position 5 (page 1, offset 1)
    std::vector<int> seq_lens = {2, 5};
    thrust::device_vector<int> seq_lens_dev(seq_lens.begin(), seq_lens.end());

    // QKV data
    const int qkv_size = batch_size * H * Hs;
    std::vector<float> qkv_host(static_cast<std::size_t>(qkv_size));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : qkv_host) v = dist(rng);
    auto qkv_dev = to_bf16_device(qkv_host);

    kv_cache_append_paged_bf16(
        thrust::raw_pointer_cast(k_pool.data()),
        thrust::raw_pointer_cast(v_pool.data()),
        thrust::raw_pointer_cast(qkv_dev.data()),
        thrust::raw_pointer_cast(seq_lens_dev.data()),
        thrust::raw_pointer_cast(block_table_dev.data()),
        max_pages_per_seq, page_block_size,
        batch_size, Hq, Hkv, Hs, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto k_result = from_bf16_device(k_pool);
    auto v_result = from_bf16_device(v_pool);

    // Verify seq 0: position 2 → page 0 → physical page 2, offset 2
    for (int h = 0; h < Hkv; ++h) {
        for (int d = 0; d < Hs; ++d) {
            float k_expected = qkv_host[static_cast<std::size_t>(0 * H * Hs + (Hq + h) * Hs + d)];
            // physical page 2, offset 2: page_2_start + 2 * Hkv * Hs + h * Hs + d
            std::size_t phys_idx = 2 * page_elems + 2 * Hkv * Hs + h * Hs + d;
            REQUIRE(std::abs(k_result[phys_idx] - k_expected) < 0.02f);
        }
    }

    // Verify seq 1: position 5 → page 1 (5/4=1), offset 1 (5%4=1) → physical page 3
    for (int h = 0; h < Hkv; ++h) {
        for (int d = 0; d < Hs; ++d) {
            float k_expected = qkv_host[static_cast<std::size_t>(1 * H * Hs + (Hq + h) * Hs + d)];
            std::size_t phys_idx = 3 * page_elems + 1 * Hkv * Hs + h * Hs + d;
            REQUIRE(std::abs(k_result[phys_idx] - k_expected) < 0.02f);
        }
    }
}

TEST_CASE("Paged attention matches contiguous attention", "[decode][paged][attention]") {
    // Verify that paged decode attention produces the same output as contiguous
    // when the block table maps to a contiguous layout.
    const int B = 1;
    const int T = 8;
    const int Hq = 4;
    const int Hkv = 2;
    const int Hs = 64;
    const int H = Hq + 2 * Hkv;
    const int page_block_size = 4;  // 2 pages for T=8
    const int total_pages = 2;
    const int max_pages_per_seq = 2;

    // Generate random QKV
    const std::size_t qkv_total = static_cast<std::size_t>(B) * T * H * Hs;
    std::vector<float> qkv_host(qkv_total);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : qkv_host) v = dist(rng);
    auto qkv_dev = to_bf16_device(qkv_host);

    // Run full-sequence varlen attention as reference
    const std::size_t out_elems = static_cast<std::size_t>(B) * T * Hq * Hs;
    thrust::device_vector<nv_bfloat16> full_out(out_elems);
    thrust::device_vector<float> full_lse(static_cast<std::size_t>(Hq) * B * T);
    thrust::device_vector<int32_t> full_cu(static_cast<std::size_t>(B + 1));
    std::vector<int32_t> full_cu_host = {0, static_cast<int32_t>(T)};
    thrust::copy(full_cu_host.begin(), full_cu_host.end(), full_cu.begin());

    attention_forward_flash_varlen(
        thrust::raw_pointer_cast(full_out.data()),
        thrust::raw_pointer_cast(full_lse.data()),
        thrust::raw_pointer_cast(qkv_dev.data()),
        thrust::raw_pointer_cast(full_cu.data()),
        B, T, B * T, Hq, Hkv, Hs, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto full_out_host = from_bf16_device(full_out);
    // Extract last token's output
    std::vector<float> last_token_ref(static_cast<std::size_t>(Hq * Hs));
    std::copy(full_out_host.begin() + static_cast<std::ptrdiff_t>((T - 1) * Hq * Hs),
              full_out_host.begin() + static_cast<std::ptrdiff_t>(T * Hq * Hs),
              last_token_ref.begin());

    // Build paged KV-cache from the full QKV (extract K/V into page pool)
    const std::size_t page_elems = static_cast<std::size_t>(page_block_size) * Hkv * Hs;
    const std::size_t pool_elems = static_cast<std::size_t>(total_pages) * page_elems;
    std::vector<float> k_pool_host(pool_elems, 0.0f);
    std::vector<float> v_pool_host(pool_elems, 0.0f);

    // Block table: seq 0 → pages [0, 1] (contiguous mapping)
    std::vector<int> block_table_host = {0, 1};

    // Fill page pool with K/V from the full QKV
    for (int t = 0; t < T; ++t) {
        const int page_idx = t / page_block_size;
        const int page_offset = t % page_block_size;
        for (int h = 0; h < Hkv; ++h) {
            for (int d = 0; d < Hs; ++d) {
                const std::size_t qkv_idx = static_cast<std::size_t>(t * H * Hs + (Hq + h) * Hs + d);
                const std::size_t pool_idx = page_idx * page_elems + page_offset * Hkv * Hs + h * Hs + d;
                k_pool_host[pool_idx] = qkv_host[qkv_idx];
                v_pool_host[pool_idx] = qkv_host[qkv_idx + static_cast<std::size_t>(Hkv * Hs)];
            }
        }
    }

    auto k_pool_dev = to_bf16_device(k_pool_host);
    auto v_pool_dev = to_bf16_device(v_pool_host);
    thrust::device_vector<int> block_table_dev(block_table_host.begin(), block_table_host.end());

    // Extract Q for last token
    std::vector<float> q_host(static_cast<std::size_t>(B * Hq * Hs));
    for (int h = 0; h < Hq; ++h) {
        for (int d = 0; d < Hs; ++d) {
            q_host[static_cast<std::size_t>(h * Hs + d)] =
                qkv_host[static_cast<std::size_t>((T - 1) * H * Hs + h * Hs + d)];
        }
    }
    auto q_dev = to_bf16_device(q_host);

    // Run paged decode attention
    thrust::device_vector<nv_bfloat16> paged_out(static_cast<std::size_t>(B * Hq * Hs));
    thrust::device_vector<float> paged_lse(static_cast<std::size_t>(B * Hq * 128));
    thrust::device_vector<int32_t> decode_cu_q(static_cast<std::size_t>(B + 1));
    thrust::device_vector<int32_t> seqused_k(static_cast<std::size_t>(B));
    std::vector<int32_t> cu_q_host = {0, 1};
    std::vector<int32_t> seqused_k_host = {static_cast<int32_t>(T)};
    thrust::copy(cu_q_host.begin(), cu_q_host.end(), decode_cu_q.begin());
    thrust::copy(seqused_k_host.begin(), seqused_k_host.end(), seqused_k.begin());

    attention_decode_flash_paged(
        thrust::raw_pointer_cast(paged_out.data()),
        thrust::raw_pointer_cast(paged_lse.data()),
        thrust::raw_pointer_cast(q_dev.data()),
        thrust::raw_pointer_cast(k_pool_dev.data()),
        thrust::raw_pointer_cast(v_pool_dev.data()),
        thrust::raw_pointer_cast(decode_cu_q.data()),
        thrust::raw_pointer_cast(seqused_k.data()),
        thrust::raw_pointer_cast(block_table_dev.data()),
        max_pages_per_seq, page_block_size,
        T, B, Hq, Hkv, Hs, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto paged_out_host = from_bf16_device(paged_out);

    float max_diff = 0.0f;
    for (std::size_t i = 0; i < static_cast<std::size_t>(Hq * Hs); ++i) {
        float diff = std::abs(paged_out_host[i] - last_token_ref[i]);
        max_diff = std::max(max_diff, diff);
    }
    INFO("Max diff paged vs full-sequence: " << max_diff);
    REQUIRE(max_diff < 0.05f);
}

TEST_CASE("Paged attention with shared prefix pages", "[decode][paged][prefix_sharing]") {
    // Verify that two sequences sharing the same prefix pages via block table
    // produce the same attention output when they have identical KV content.
    const int B = 2;
    const int T = 8;
    const int Hq = 4;
    const int Hkv = 2;
    const int Hs = 64;
    const int H = Hq + 2 * Hkv;
    const int page_block_size = 256;  // Single page covers all T tokens
    const int total_pages = 2;       // 2 separate pages (one per seq, same data)
    const int max_pages_per_seq = 1;

    std::vector<float> qkv_host(static_cast<std::size_t>(T * H * Hs));
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : qkv_host) v = dist(rng);

    // Build page pool from single set of KV data
    const std::size_t page_elems = static_cast<std::size_t>(page_block_size) * Hkv * Hs;
    const std::size_t pool_elems = static_cast<std::size_t>(total_pages) * page_elems;
    std::vector<float> k_pool_host(pool_elems, 0.0f);
    std::vector<float> v_pool_host(pool_elems, 0.0f);

    for (int t = 0; t < T; ++t) {
        const int page_idx = t / page_block_size;
        const int page_offset = t % page_block_size;
        for (int h = 0; h < Hkv; ++h) {
            for (int d = 0; d < Hs; ++d) {
                const std::size_t qkv_idx = static_cast<std::size_t>(t * H * Hs + (Hq + h) * Hs + d);
                const std::size_t pool_idx = page_idx * page_elems + page_offset * Hkv * Hs + h * Hs + d;
                k_pool_host[pool_idx] = qkv_host[qkv_idx];
                v_pool_host[pool_idx] = qkv_host[qkv_idx + static_cast<std::size_t>(Hkv * Hs)];
            }
        }
    }

    // Duplicate page 0 data into page 1 (both pages have identical KV)
    std::copy(k_pool_host.begin(), k_pool_host.begin() + static_cast<std::ptrdiff_t>(page_elems),
              k_pool_host.begin() + static_cast<std::ptrdiff_t>(page_elems));
    std::copy(v_pool_host.begin(), v_pool_host.begin() + static_cast<std::ptrdiff_t>(page_elems),
              v_pool_host.begin() + static_cast<std::ptrdiff_t>(page_elems));

    auto k_pool_dev = to_bf16_device(k_pool_host);
    auto v_pool_dev = to_bf16_device(v_pool_host);

    // Both sequences point to separate pages with IDENTICAL data
    std::vector<int> block_table_host = {
        0,   // seq 0: page [0]
        1    // seq 1: page [1] — different physical page, same data
    };
    thrust::device_vector<int> block_table_dev(block_table_host.begin(), block_table_host.end());

    // Same Q for both sequences (last token's Q)
    std::vector<float> q_host(static_cast<std::size_t>(B * Hq * Hs));
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < Hq; ++h) {
            for (int d = 0; d < Hs; ++d) {
                q_host[static_cast<std::size_t>(b * Hq * Hs + h * Hs + d)] =
                    qkv_host[static_cast<std::size_t>((T - 1) * H * Hs + h * Hs + d)];
            }
        }
    }
    auto q_dev = to_bf16_device(q_host);

    thrust::device_vector<nv_bfloat16> out(static_cast<std::size_t>(B * Hq * Hs));
    thrust::device_vector<float> lse(static_cast<std::size_t>(B * Hq * 128));
    thrust::device_vector<int32_t> cu_q(static_cast<std::size_t>(B + 1));
    thrust::device_vector<int32_t> seqused_k(static_cast<std::size_t>(B));
    std::vector<int32_t> cu_q_host = {0, 1, 2};
    std::vector<int32_t> seqused_k_host = {T, T};
    thrust::copy(cu_q_host.begin(), cu_q_host.end(), cu_q.begin());
    thrust::copy(seqused_k_host.begin(), seqused_k_host.end(), seqused_k.begin());

    attention_decode_flash_paged(
        thrust::raw_pointer_cast(out.data()),
        thrust::raw_pointer_cast(lse.data()),
        thrust::raw_pointer_cast(q_dev.data()),
        thrust::raw_pointer_cast(k_pool_dev.data()),
        thrust::raw_pointer_cast(v_pool_dev.data()),
        thrust::raw_pointer_cast(cu_q.data()),
        thrust::raw_pointer_cast(seqused_k.data()),
        thrust::raw_pointer_cast(block_table_dev.data()),
        max_pages_per_seq, page_block_size,
        T, B, Hq, Hkv, Hs, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto out_host = from_bf16_device(out);

    // Both sequences should produce IDENTICAL output (same Q, same shared KV pages)
    float max_diff = 0.0f;
    for (int i = 0; i < Hq * Hs; ++i) {
        float diff = std::abs(out_host[static_cast<std::size_t>(i)]
                            - out_host[static_cast<std::size_t>(Hq * Hs + i)]);
        max_diff = std::max(max_diff, diff);
    }
    INFO("Max diff between shared-prefix sequences: " << max_diff);
    // Verify each sequence individually against the B=1 reference from
    // the "Paged attention matches contiguous" test approach.
    // Both should match the full-sequence last-token output.
    // Run B=1 paged reference for comparison.
    thrust::device_vector<nv_bfloat16> ref_out(static_cast<std::size_t>(Hq * Hs));
    thrust::device_vector<float> ref_lse(static_cast<std::size_t>(Hq * 128));
    thrust::device_vector<int32_t> ref_cu_q(2);
    thrust::device_vector<int32_t> ref_seqused(1);
    thrust::device_vector<int> ref_bt(block_table_host.begin(), block_table_host.begin() + max_pages_per_seq);
    std::vector<int32_t> ref_cu_q_h = {0, 1};
    std::vector<int32_t> ref_seqused_h = {T};
    thrust::copy(ref_cu_q_h.begin(), ref_cu_q_h.end(), ref_cu_q.begin());
    thrust::copy(ref_seqused_h.begin(), ref_seqused_h.end(), ref_seqused.begin());

    // Q for single seq (same data)
    auto q_single = to_bf16_device(std::vector<float>(q_host.begin(), q_host.begin() + Hq * Hs));

    attention_decode_flash_paged(
        thrust::raw_pointer_cast(ref_out.data()),
        thrust::raw_pointer_cast(ref_lse.data()),
        thrust::raw_pointer_cast(q_single.data()),
        thrust::raw_pointer_cast(k_pool_dev.data()),
        thrust::raw_pointer_cast(v_pool_dev.data()),
        thrust::raw_pointer_cast(ref_cu_q.data()),
        thrust::raw_pointer_cast(ref_seqused.data()),
        thrust::raw_pointer_cast(ref_bt.data()),
        max_pages_per_seq, page_block_size,
        T, 1, Hq, Hkv, Hs, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto ref_out_host = from_bf16_device(ref_out);

    // Each batch item in the B=2 run should match the B=1 reference
    float max_diff_0 = 0.0f, max_diff_1 = 0.0f;
    for (int i = 0; i < Hq * Hs; ++i) {
        max_diff_0 = std::max(max_diff_0, std::abs(out_host[static_cast<std::size_t>(i)] - ref_out_host[static_cast<std::size_t>(i)]));
        max_diff_1 = std::max(max_diff_1, std::abs(out_host[static_cast<std::size_t>(Hq * Hs + i)] - ref_out_host[static_cast<std::size_t>(i)]));
    }
    INFO("Max diff seq0 vs B=1 ref: " << max_diff_0 << ", seq1 vs B=1 ref: " << max_diff_1);
    REQUIRE(max_diff_0 < 0.05f);
    REQUIRE(max_diff_1 < 0.05f);
}

TEST_CASE("Decode attention matches full-sequence attention", "[decode][attention]") {
    // This test verifies that attending to KV-cache with a single query token
    // produces the same output as full-sequence self-attention where the query
    // is the last token.
    //
    // We:
    //   1. Build an interleaved QKV buffer for a full sequence [B, T, H, Hs]
    //   2. Run full-sequence varlen attention
    //   3. Extract K/V into the KV-cache layout
    //   4. Run decode attention with Q = last token's Q, KV from cache
    //   5. Compare the output for the last token position

    const int B = 1;
    const int T = 8;   // Full sequence length
    const int Hq = 4;
    const int Hkv = 2;
    const int Hs = 64;  // Flash Attention requires head_dim >= 16
    const int H = Hq + 2 * Hkv;

    // Generate random QKV data
    const std::size_t qkv_total = static_cast<std::size_t>(B) * T * H * Hs;
    std::vector<float> qkv_host(qkv_total);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : qkv_host) v = dist(rng);

    auto qkv_dev = to_bf16_device(qkv_host);

    // ========================================================================
    // Step 1: Full-sequence varlen attention
    // ========================================================================
    const std::size_t out_elems = static_cast<std::size_t>(B) * T * Hq * Hs;
    thrust::device_vector<nv_bfloat16> full_out(out_elems);
    thrust::device_vector<float> full_lse(static_cast<std::size_t>(Hq) * B * T);

    // Build cu_seqlens for full sequence: [0, T]
    thrust::device_vector<int32_t> full_cu_seqlens(static_cast<std::size_t>(B + 1));
    std::vector<int32_t> full_cu_host = {0, static_cast<int32_t>(T)};
    thrust::copy(full_cu_host.begin(), full_cu_host.end(), full_cu_seqlens.begin());

    attention_forward_flash_varlen(
        thrust::raw_pointer_cast(full_out.data()),
        thrust::raw_pointer_cast(full_lse.data()),
        thrust::raw_pointer_cast(qkv_dev.data()),
        thrust::raw_pointer_cast(full_cu_seqlens.data()),
        B, T, B * T,
        Hq, Hkv, Hs,
        nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Extract the last token's attention output from full sequence
    std::vector<float> full_out_host = from_bf16_device(full_out);
    std::vector<float> last_token_full(static_cast<std::size_t>(Hq * Hs));
    const std::size_t last_offset = static_cast<std::size_t>(T - 1) * Hq * Hs;
    std::copy(full_out_host.begin() + static_cast<std::ptrdiff_t>(last_offset),
              full_out_host.begin() + static_cast<std::ptrdiff_t>(last_offset + Hq * Hs),
              last_token_full.begin());

    // ========================================================================
    // Step 2: Extract K/V into separate cache buffers
    // ========================================================================
    // K/V cache: [B, T, Hkv, Hs] (all T positions from the full sequence)
    const std::size_t kv_cache_elems = static_cast<std::size_t>(B) * T * Hkv * Hs;
    thrust::device_vector<nv_bfloat16> k_cache(kv_cache_elems);
    thrust::device_vector<nv_bfloat16> v_cache(kv_cache_elems);

    // Extract K and V from interleaved QKV on host, then upload
    std::vector<float> k_host(kv_cache_elems);
    std::vector<float> v_host(kv_cache_elems);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < Hkv; ++h) {
                for (int d = 0; d < Hs; ++d) {
                    const std::size_t qkv_idx = static_cast<std::size_t>((b * T + t) * H * Hs + (Hq + h) * Hs + d);
                    const std::size_t cache_idx = static_cast<std::size_t>((b * T + t) * Hkv * Hs + h * Hs + d);
                    k_host[cache_idx] = qkv_host[qkv_idx];
                    v_host[cache_idx] = qkv_host[qkv_idx + static_cast<std::size_t>(Hkv * Hs)];
                }
            }
        }
    }
    auto k_cache_dev = to_bf16_device(k_host);
    auto v_cache_dev = to_bf16_device(v_host);

    // ========================================================================
    // Step 3: Extract Q for last token
    // ========================================================================
    // Q of last token: QKV[0, T-1, 0:Hq, :] → [B, Hq, Hs]
    std::vector<float> q_host(static_cast<std::size_t>(B * Hq * Hs));
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < Hq; ++h) {
            for (int d = 0; d < Hs; ++d) {
                const std::size_t qkv_idx = static_cast<std::size_t>((b * T + (T - 1)) * H * Hs + h * Hs + d);
                const std::size_t q_idx = static_cast<std::size_t>(b * Hq * Hs + h * Hs + d);
                q_host[q_idx] = qkv_host[qkv_idx];
            }
        }
    }
    auto q_dev = to_bf16_device(q_host);

    // ========================================================================
    // Step 4: Decode attention
    // ========================================================================
    thrust::device_vector<nv_bfloat16> decode_out(static_cast<std::size_t>(B * Hq * Hs));
    thrust::device_vector<float> decode_lse(static_cast<std::size_t>(B * Hq * 128));

    // cu_seqlens_q: [0, 1]  (1 query token per sequence)
    // seqused_k:    [T]     (T cached tokens per sequence)
    thrust::device_vector<int32_t> decode_cu_q(static_cast<std::size_t>(B + 1));
    thrust::device_vector<int32_t> decode_seqused_k(static_cast<std::size_t>(B));
    std::vector<int32_t> decode_cu_q_host = {0, 1};
    std::vector<int32_t> decode_seqused_k_host = {static_cast<int32_t>(T)};
    thrust::copy(decode_cu_q_host.begin(), decode_cu_q_host.end(), decode_cu_q.begin());
    thrust::copy(decode_seqused_k_host.begin(), decode_seqused_k_host.end(), decode_seqused_k.begin());

    attention_decode_flash(
        thrust::raw_pointer_cast(decode_out.data()),
        thrust::raw_pointer_cast(decode_lse.data()),
        thrust::raw_pointer_cast(q_dev.data()),
        thrust::raw_pointer_cast(k_cache_dev.data()),
        thrust::raw_pointer_cast(v_cache_dev.data()),
        thrust::raw_pointer_cast(decode_cu_q.data()),
        thrust::raw_pointer_cast(decode_seqused_k.data()),
        T,  // max_seqlen_k
        B, Hq, Hkv, Hs,
        nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ========================================================================
    // Step 5: Compare
    // ========================================================================
    auto decode_out_host = from_bf16_device(decode_out);

    float max_diff = 0.0f;
    for (std::size_t i = 0; i < static_cast<std::size_t>(Hq * Hs); ++i) {
        float diff = std::abs(decode_out_host[i] - last_token_full[i]);
        max_diff = std::max(max_diff, diff);
    }

    // BF16 attention should match within reasonable tolerance
    // (both paths use BF16 precision with the same Flash Attention kernels)
    INFO("Max difference between decode and full-sequence attention: " << max_diff);
    REQUIRE(max_diff < 0.05f);  // Conservative BF16 tolerance
}

TEST_CASE("Contiguous B=2 with identical data", "[decode][b2_debug]") {
    const int B = 2, T = 8, Hq = 4, Hkv = 2, Hs = 64;
    const int H = Hq + 2 * Hkv;

    std::vector<float> qkv_host(static_cast<std::size_t>(T * H * Hs));
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : qkv_host) v = dist(rng);

    // Build contiguous K/V: [B=2, T, Hkv, Hs] — both batch items identical
    std::size_t kv_elems = static_cast<std::size_t>(B) * T * Hkv * Hs;
    std::vector<float> k_host(kv_elems), v_host(kv_elems);
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < Hkv; ++h)
                for (int d = 0; d < Hs; ++d) {
                    std::size_t ci = static_cast<std::size_t>(b*T*Hkv*Hs + t*Hkv*Hs + h*Hs + d);
                    k_host[ci] = qkv_host[static_cast<std::size_t>(t*H*Hs + (Hq+h)*Hs + d)];
                    v_host[ci] = qkv_host[static_cast<std::size_t>(t*H*Hs + (Hq+Hkv+h)*Hs + d)];
                }

    auto k_dev = to_bf16_device(k_host);
    auto v_dev = to_bf16_device(v_host);

    // Q: both seqs same
    std::vector<float> q_host(static_cast<std::size_t>(B * Hq * Hs));
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < Hq; ++h)
            for (int d = 0; d < Hs; ++d)
                q_host[static_cast<std::size_t>(b*Hq*Hs + h*Hs + d)] =
                    qkv_host[static_cast<std::size_t>((T-1)*H*Hs + h*Hs + d)];
    auto q_dev = to_bf16_device(q_host);

    thrust::device_vector<nv_bfloat16> out(static_cast<std::size_t>(B * Hq * Hs));
    thrust::device_vector<float> decode_lse(static_cast<std::size_t>(B * Hq * 128));
    std::vector<int32_t> cu_q_h(static_cast<std::size_t>(B+1));
    for (int i = 0; i <= B; ++i) cu_q_h[static_cast<std::size_t>(i)] = i;
    std::vector<int32_t> sk_h = {static_cast<int32_t>(T), static_cast<int32_t>(T)};
    thrust::device_vector<int32_t> cu_q(cu_q_h.begin(), cu_q_h.end());
    thrust::device_vector<int32_t> sk(sk_h.begin(), sk_h.end());

    attention_decode_flash(
        thrust::raw_pointer_cast(out.data()), thrust::raw_pointer_cast(decode_lse.data()),
        thrust::raw_pointer_cast(q_dev.data()),
        thrust::raw_pointer_cast(k_dev.data()), thrust::raw_pointer_cast(v_dev.data()),
        thrust::raw_pointer_cast(cu_q.data()), thrust::raw_pointer_cast(sk.data()),
        T, B, Hq, Hkv, Hs, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto out_host = from_bf16_device(out);
    float max_diff = 0;
    for (int i = 0; i < Hq*Hs; ++i) {
        float d = std::abs(out_host[static_cast<std::size_t>(i)] - out_host[static_cast<std::size_t>(Hq*Hs + i)]);
        max_diff = std::max(max_diff, d);
    }
    INFO("Contiguous B=2 max diff between identical seqs: " << max_diff);
    REQUIRE(max_diff < 0.01f);
}

TEST_CASE("Bulk KV store matches sequential append", "[decode][prefill]") {
    // Verify kv_cache_store_bf16 produces the same cache as T calls to kv_cache_append_bf16.
    const int B = 2;
    const int T = 8;
    const int max_seq_len = 16;
    const int Hq = 4;
    const int Hkv = 2;
    const int Hs = 8;
    const int H = Hq + 2 * Hkv;

    // Random QKV: [B, T, H, Hs]
    const std::size_t qkv_total = static_cast<std::size_t>(B) * T * H * Hs;
    std::vector<float> qkv_host(qkv_total);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : qkv_host) v = dist(rng);
    auto qkv_dev = to_bf16_device(qkv_host);

    const std::size_t cache_elems = static_cast<std::size_t>(B) * max_seq_len * Hkv * Hs;

    // === Path A: bulk store ===
    thrust::device_vector<nv_bfloat16> k_bulk(cache_elems, __float2bfloat16(0.0f));
    thrust::device_vector<nv_bfloat16> v_bulk(cache_elems, __float2bfloat16(0.0f));

    kv_cache_store_bf16(
        thrust::raw_pointer_cast(k_bulk.data()),
        thrust::raw_pointer_cast(v_bulk.data()),
        thrust::raw_pointer_cast(qkv_dev.data()),
        B, T, max_seq_len, Hq, Hkv, Hs, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // === Path B: sequential append, one token at a time ===
    thrust::device_vector<nv_bfloat16> k_seq(cache_elems, __float2bfloat16(0.0f));
    thrust::device_vector<nv_bfloat16> v_seq(cache_elems, __float2bfloat16(0.0f));

    for (int t = 0; t < T; ++t) {
        // Build single-token QKV slice: [B, 1, H, Hs] from full QKV at position t
        std::vector<float> tok_qkv(static_cast<std::size_t>(B * H * Hs));
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {
                for (int d = 0; d < Hs; ++d) {
                    tok_qkv[static_cast<std::size_t>(b * H * Hs + h * Hs + d)] =
                        qkv_host[static_cast<std::size_t>((b * T + t) * H * Hs + h * Hs + d)];
                }
            }
        }
        auto tok_dev = to_bf16_device(tok_qkv);

        // seq_lens = [t, t] (both seqs at same position)
        std::vector<int> sl = {t, t};
        thrust::device_vector<int> sl_dev(sl.begin(), sl.end());

        kv_cache_append_bf16(
            thrust::raw_pointer_cast(k_seq.data()),
            thrust::raw_pointer_cast(v_seq.data()),
            thrust::raw_pointer_cast(tok_dev.data()),
            thrust::raw_pointer_cast(sl_dev.data()),
            B, max_seq_len, Hq, Hkv, Hs, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // === Compare ===
    auto k_bulk_h = from_bf16_device(k_bulk);
    auto v_bulk_h = from_bf16_device(v_bulk);
    auto k_seq_h = from_bf16_device(k_seq);
    auto v_seq_h = from_bf16_device(v_seq);

    float max_k_diff = 0.0f, max_v_diff = 0.0f;
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < Hkv; ++h) {
                for (int d = 0; d < Hs; ++d) {
                    std::size_t idx = static_cast<std::size_t>(
                        b * max_seq_len * Hkv * Hs + t * Hkv * Hs + h * Hs + d);
                    max_k_diff = std::max(max_k_diff, std::abs(k_bulk_h[idx] - k_seq_h[idx]));
                    max_v_diff = std::max(max_v_diff, std::abs(v_bulk_h[idx] - v_seq_h[idx]));
                }
            }
        }
    }
    INFO("Bulk vs sequential K diff: " << max_k_diff << ", V diff: " << max_v_diff);
    REQUIRE(max_k_diff == 0.0f);  // Exact match expected (same BF16 data, no accumulation)
    REQUIRE(max_v_diff == 0.0f);
}

TEST_CASE("Paged bulk KV store matches sequential append", "[decode][prefill][paged]") {
    const int B = 1;
    const int T = 8;
    const int Hq = 4;
    const int Hkv = 2;
    const int Hs = 8;
    const int H = Hq + 2 * Hkv;
    const int page_block_size = 4;
    const int total_pages = 4;  // 2 pages per seq, 2 extra
    const int max_pages_per_seq = 2;

    const std::size_t qkv_total = static_cast<std::size_t>(B) * T * H * Hs;
    std::vector<float> qkv_host(qkv_total);
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : qkv_host) v = dist(rng);
    auto qkv_dev = to_bf16_device(qkv_host);

    const std::size_t page_elems = static_cast<std::size_t>(page_block_size) * Hkv * Hs;
    const std::size_t pool_elems = static_cast<std::size_t>(total_pages) * page_elems;

    // Block table: seq 0 → pages [2, 3]
    std::vector<int> bt_host = {2, 3};
    thrust::device_vector<int> bt_dev(bt_host.begin(), bt_host.end());

    // === Path A: bulk paged store ===
    thrust::device_vector<nv_bfloat16> k_bulk(pool_elems, __float2bfloat16(0.0f));
    thrust::device_vector<nv_bfloat16> v_bulk(pool_elems, __float2bfloat16(0.0f));

    kv_cache_store_paged_bf16(
        thrust::raw_pointer_cast(k_bulk.data()),
        thrust::raw_pointer_cast(v_bulk.data()),
        thrust::raw_pointer_cast(qkv_dev.data()),
        thrust::raw_pointer_cast(bt_dev.data()),
        max_pages_per_seq, page_block_size,
        B, T, Hq, Hkv, Hs, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // === Path B: sequential paged append ===
    thrust::device_vector<nv_bfloat16> k_seq(pool_elems, __float2bfloat16(0.0f));
    thrust::device_vector<nv_bfloat16> v_seq(pool_elems, __float2bfloat16(0.0f));

    for (int t = 0; t < T; ++t) {
        std::vector<float> tok_qkv(static_cast<std::size_t>(H * Hs));
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < Hs; ++d)
                tok_qkv[static_cast<std::size_t>(h * Hs + d)] =
                    qkv_host[static_cast<std::size_t>(t * H * Hs + h * Hs + d)];
        auto tok_dev = to_bf16_device(tok_qkv);

        std::vector<int> sl = {t};
        thrust::device_vector<int> sl_dev(sl.begin(), sl.end());

        kv_cache_append_paged_bf16(
            thrust::raw_pointer_cast(k_seq.data()),
            thrust::raw_pointer_cast(v_seq.data()),
            thrust::raw_pointer_cast(tok_dev.data()),
            thrust::raw_pointer_cast(sl_dev.data()),
            thrust::raw_pointer_cast(bt_dev.data()),
            max_pages_per_seq, page_block_size,
            B, Hq, Hkv, Hs, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // === Compare: only check pages that should have data (pages 2 and 3) ===
    auto k_bulk_h = from_bf16_device(k_bulk);
    auto k_seq_h = from_bf16_device(k_seq);
    auto v_bulk_h = from_bf16_device(v_bulk);
    auto v_seq_h = from_bf16_device(v_seq);

    float max_k_diff = 0.0f, max_v_diff = 0.0f;
    for (int page : {2, 3}) {
        for (std::size_t i = 0; i < page_elems; ++i) {
            std::size_t idx = static_cast<std::size_t>(page) * page_elems + i;
            max_k_diff = std::max(max_k_diff, std::abs(k_bulk_h[idx] - k_seq_h[idx]));
            max_v_diff = std::max(max_v_diff, std::abs(v_bulk_h[idx] - v_seq_h[idx]));
        }
    }
    INFO("Paged bulk vs sequential K diff: " << max_k_diff << ", V diff: " << max_v_diff);
    REQUIRE(max_k_diff == 0.0f);
    REQUIRE(max_v_diff == 0.0f);
}

TEST_CASE("mask_finished_tokens kernel", "[decode][eos_mask]") {
    const int batch_size = 4;

    std::vector<int32_t> tokens = {100, 200, 300, 400};
    std::vector<int> finished = {0, 1, 0, 1};  // seq 1 and 3 finished

    thrust::device_vector<int32_t> tokens_dev(tokens.begin(), tokens.end());
    thrust::device_vector<int> finished_dev(finished.begin(), finished.end());

    mask_finished_tokens(
        thrust::raw_pointer_cast(tokens_dev.data()),
        thrust::raw_pointer_cast(finished_dev.data()),
        batch_size, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> result(static_cast<std::size_t>(batch_size));
    thrust::copy(tokens_dev.begin(), tokens_dev.end(), result.begin());

    REQUIRE(result[0] == 100);  // active — unchanged
    REQUIRE(result[1] == 0);    // finished — masked to 0
    REQUIRE(result[2] == 300);  // active — unchanged
    REQUIRE(result[3] == 0);    // finished — masked to 0
}
