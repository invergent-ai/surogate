// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Tests for FP8 E4M3 KV-cache: quantize, dequant, and attention correctness.

#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "kernels/attention_decode.h"
#include "kernels/kernels.h"
#include "utilities/utils.h"

namespace {

thrust::device_vector<nv_bfloat16> to_bf16_device(const std::vector<float>& host) {
    std::vector<nv_bfloat16> bf16(host.size());
    for (std::size_t i = 0; i < host.size(); ++i) bf16[i] = __float2bfloat16(host[i]);
    return thrust::device_vector<nv_bfloat16>(bf16.begin(), bf16.end());
}

std::vector<float> from_bf16_device(const thrust::device_vector<nv_bfloat16>& dev) {
    std::vector<nv_bfloat16> bf16(dev.size());
    thrust::copy(dev.begin(), dev.end(), bf16.begin());
    std::vector<float> result(bf16.size());
    for (std::size_t i = 0; i < bf16.size(); ++i) result[i] = __bfloat162float(bf16[i]);
    return result;
}

}  // anonymous namespace

TEST_CASE("FP8 KV-cache append + dequant round-trip", "[decode][fp8]") {
    // Verify that quantize→dequant preserves values within FP8 precision.
    const int batch_size = 2;
    const int max_seq_len = 8;
    const int Hq = 4;
    const int Hkv = 2;
    const int Hs = 8;
    const int H = Hq + 2 * Hkv;

    // Random BF16 QKV data
    const int qkv_size = batch_size * H * Hs;
    std::vector<float> qkv_host(static_cast<std::size_t>(qkv_size));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : qkv_host) v = dist(rng);
    auto qkv_dev = to_bf16_device(qkv_host);

    // FP8 KV cache
    const std::size_t cache_elems = static_cast<std::size_t>(batch_size) * max_seq_len * Hkv * Hs;
    thrust::device_vector<__nv_fp8_e4m3> k_fp8(cache_elems);
    thrust::device_vector<__nv_fp8_e4m3> v_fp8(cache_elems);

    // Scales
    const std::size_t scale_elems = static_cast<std::size_t>(batch_size) * max_seq_len * Hkv;
    thrust::device_vector<float> k_scales(scale_elems, 0.0f);
    thrust::device_vector<float> v_scales(scale_elems, 0.0f);

    // Write at position 3 for seq 0, position 5 for seq 1
    std::vector<int> seq_lens_host = {3, 5};
    thrust::device_vector<int> seq_lens_dev(seq_lens_host.begin(), seq_lens_host.end());

    kv_cache_append_fp8(
        thrust::raw_pointer_cast(k_fp8.data()),
        thrust::raw_pointer_cast(v_fp8.data()),
        thrust::raw_pointer_cast(k_scales.data()),
        thrust::raw_pointer_cast(v_scales.data()),
        thrust::raw_pointer_cast(qkv_dev.data()),
        thrust::raw_pointer_cast(seq_lens_dev.data()),
        batch_size, max_seq_len, Hq, Hkv, Hs, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Now dequant back to BF16
    // First set seq_lens to include the appended token (pos+1)
    std::vector<int> seq_lens_after = {4, 6};
    thrust::device_vector<int> seq_lens_after_dev(seq_lens_after.begin(), seq_lens_after.end());

    thrust::device_vector<nv_bfloat16> k_bf16(cache_elems);
    thrust::device_vector<nv_bfloat16> v_bf16(cache_elems);

    kv_cache_dequant_fp8_to_bf16(
        thrust::raw_pointer_cast(k_bf16.data()),
        thrust::raw_pointer_cast(v_bf16.data()),
        thrust::raw_pointer_cast(k_fp8.data()),
        thrust::raw_pointer_cast(v_fp8.data()),
        thrust::raw_pointer_cast(k_scales.data()),
        thrust::raw_pointer_cast(v_scales.data()),
        thrust::raw_pointer_cast(seq_lens_after_dev.data()),
        batch_size, max_seq_len, Hkv, Hs, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto k_result = from_bf16_device(k_bf16);

    // Check that the values at the written positions are close to the original
    for (int b = 0; b < batch_size; ++b) {
        const int pos = seq_lens_host[static_cast<std::size_t>(b)];
        for (int h = 0; h < Hkv; ++h) {
            for (int d = 0; d < Hs; ++d) {
                float original = qkv_host[static_cast<std::size_t>(b * H * Hs + (Hq + h) * Hs + d)];
                float recovered = k_result[static_cast<std::size_t>(
                    b * max_seq_len * Hkv * Hs + pos * Hkv * Hs + h * Hs + d)];
                // FP8 E4M3 has ~0.5% relative error for typical values
                float rel_err = std::abs(original) > 0.01f
                    ? std::abs(recovered - original) / std::abs(original)
                    : std::abs(recovered - original);
                REQUIRE(rel_err < 0.15f);  // FP8 E4M3 has 3-4 mantissa bits
            }
        }
    }
}

TEST_CASE("FP8 decode attention vs BF16 decode attention", "[decode][fp8][attention]") {
    // Verify that FP8 KV-cache decode attention produces results close to BF16.
    const int B = 1;
    const int T = 8;
    const int Hq = 4;
    const int Hkv = 2;
    const int Hs = 64;
    const int H = Hq + 2 * Hkv;

    // Generate random QKV
    std::vector<float> qkv_host(static_cast<std::size_t>(T * H * Hs));
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : qkv_host) v = dist(rng);

    // ========== BF16 reference path ==========
    // Extract K/V into BF16 cache
    const std::size_t kv_elems = static_cast<std::size_t>(B) * T * Hkv * Hs;
    std::vector<float> k_bf16_host(kv_elems), v_bf16_host(kv_elems);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < Hkv; ++h)
            for (int d = 0; d < Hs; ++d) {
                std::size_t ci = static_cast<std::size_t>(t * Hkv * Hs + h * Hs + d);
                k_bf16_host[ci] = qkv_host[static_cast<std::size_t>(t * H * Hs + (Hq + h) * Hs + d)];
                v_bf16_host[ci] = qkv_host[static_cast<std::size_t>(t * H * Hs + (Hq + Hkv + h) * Hs + d)];
            }
    auto k_bf16_dev = to_bf16_device(k_bf16_host);
    auto v_bf16_dev = to_bf16_device(v_bf16_host);

    // Q: last token
    std::vector<float> q_host(static_cast<std::size_t>(Hq * Hs));
    for (int h = 0; h < Hq; ++h)
        for (int d = 0; d < Hs; ++d)
            q_host[static_cast<std::size_t>(h * Hs + d)] =
                qkv_host[static_cast<std::size_t>((T - 1) * H * Hs + h * Hs + d)];
    auto q_dev = to_bf16_device(q_host);

    thrust::device_vector<nv_bfloat16> bf16_out(static_cast<std::size_t>(Hq * Hs));
    thrust::device_vector<float> bf16_lse(static_cast<std::size_t>(Hq * 128));
    std::vector<int32_t> seqused = {T};
    thrust::device_vector<int32_t> seqused_dev(seqused.begin(), seqused.end());

    attention_decode_flash(
        thrust::raw_pointer_cast(bf16_out.data()),
        thrust::raw_pointer_cast(bf16_lse.data()),
        thrust::raw_pointer_cast(q_dev.data()),
        thrust::raw_pointer_cast(k_bf16_dev.data()),
        thrust::raw_pointer_cast(v_bf16_dev.data()),
        nullptr, thrust::raw_pointer_cast(seqused_dev.data()),
        T, T, B, Hq, Hkv, Hs, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ========== FP8 path ==========
    // Quantize K/V to FP8
    thrust::device_vector<__nv_fp8_e4m3> k_fp8(kv_elems);
    thrust::device_vector<__nv_fp8_e4m3> v_fp8(kv_elems);
    const std::size_t scale_elems = static_cast<std::size_t>(B) * T * Hkv;
    thrust::device_vector<float> k_sc(scale_elems), v_sc(scale_elems);

    // Append each token one at a time to build the FP8 cache
    for (int t = 0; t < T; ++t) {
        // Build a single-token QKV for this position
        std::vector<float> tok_qkv(static_cast<std::size_t>(H * Hs));
        std::copy(qkv_host.begin() + t * H * Hs,
                  qkv_host.begin() + (t + 1) * H * Hs,
                  tok_qkv.begin());
        auto tok_dev = to_bf16_device(tok_qkv);

        std::vector<int> sl = {t};
        thrust::device_vector<int> sl_dev(sl.begin(), sl.end());

        kv_cache_append_fp8(
            thrust::raw_pointer_cast(k_fp8.data()),
            thrust::raw_pointer_cast(v_fp8.data()),
            thrust::raw_pointer_cast(k_sc.data()),
            thrust::raw_pointer_cast(v_sc.data()),
            thrust::raw_pointer_cast(tok_dev.data()),
            thrust::raw_pointer_cast(sl_dev.data()),
            1, T, Hq, Hkv, Hs, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Dequant FP8 → BF16
    thrust::device_vector<nv_bfloat16> k_dq(kv_elems), v_dq(kv_elems);
    std::vector<int> full_sl = {T};
    thrust::device_vector<int> full_sl_dev(full_sl.begin(), full_sl.end());

    kv_cache_dequant_fp8_to_bf16(
        thrust::raw_pointer_cast(k_dq.data()),
        thrust::raw_pointer_cast(v_dq.data()),
        thrust::raw_pointer_cast(k_fp8.data()),
        thrust::raw_pointer_cast(v_fp8.data()),
        thrust::raw_pointer_cast(k_sc.data()),
        thrust::raw_pointer_cast(v_sc.data()),
        thrust::raw_pointer_cast(full_sl_dev.data()),
        1, T, Hkv, Hs, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run BF16 attention on dequantized FP8 cache
    thrust::device_vector<nv_bfloat16> fp8_out(static_cast<std::size_t>(Hq * Hs));
    thrust::device_vector<float> fp8_lse(static_cast<std::size_t>(Hq * 128));

    attention_decode_flash(
        thrust::raw_pointer_cast(fp8_out.data()),
        thrust::raw_pointer_cast(fp8_lse.data()),
        thrust::raw_pointer_cast(q_dev.data()),
        thrust::raw_pointer_cast(k_dq.data()),
        thrust::raw_pointer_cast(v_dq.data()),
        nullptr, thrust::raw_pointer_cast(seqused_dev.data()),
        T, T, B, Hq, Hkv, Hs, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compare
    auto bf16_out_h = from_bf16_device(bf16_out);
    auto fp8_out_h = from_bf16_device(fp8_out);

    float max_diff = 0.0f;
    for (std::size_t i = 0; i < static_cast<std::size_t>(Hq * Hs); ++i) {
        max_diff = std::max(max_diff, std::abs(bf16_out_h[i] - fp8_out_h[i]));
    }
    INFO("Max diff FP8 vs BF16 attention: " << max_diff);
    // FP8 quantization introduces ~1-5% relative error in KV values, which
    // propagates through attention. Tolerance is higher than BF16-only tests.
    REQUIRE(max_diff < 0.15f);
}

TEST_CASE("FP8 paged bulk store honors start_pos", "[decode][fp8][paged][prefill]") {
    const int B = 1;
    const int T = 5;
    const int start_pos = 3;
    const int max_seq_len = 16;
    const int Hq = 4;
    const int Hkv = 2;
    const int Hs = 8;
    const int H = Hq + 2 * Hkv;
    const int page_block_size = 4;
    const int max_pages_per_seq = 4;
    const int total_pages = 6;

    // Random BF16 QKV input [B, T, H, Hs]
    const std::size_t qkv_elems = static_cast<std::size_t>(B) * T * H * Hs;
    std::vector<float> qkv_host(qkv_elems);
    std::mt19937 rng(314);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : qkv_host) v = dist(rng);
    auto qkv_dev = to_bf16_device(qkv_host);

    // Paged FP8 KV storage + scales
    const std::size_t page_elems = static_cast<std::size_t>(page_block_size) * Hkv * Hs;
    const std::size_t pool_elems = static_cast<std::size_t>(total_pages) * page_elems;
    const std::size_t page_scale_elems = static_cast<std::size_t>(page_block_size) * Hkv;
    const std::size_t scale_pool_elems = static_cast<std::size_t>(total_pages) * page_scale_elems;

    thrust::device_vector<__nv_fp8_e4m3> k_fp8(pool_elems);
    thrust::device_vector<__nv_fp8_e4m3> v_fp8(pool_elems);
    thrust::device_vector<float> k_sc(scale_pool_elems, 0.0f);
    thrust::device_vector<float> v_sc(scale_pool_elems, 0.0f);

    // Virtual pages -> physical pages for seq 0.
    std::vector<int> bt_host = {1, 2, 4, 5};
    thrust::device_vector<int> bt_dev(bt_host.begin(), bt_host.end());

    kv_cache_store_paged_fp8(
        thrust::raw_pointer_cast(k_fp8.data()),
        thrust::raw_pointer_cast(v_fp8.data()),
        thrust::raw_pointer_cast(k_sc.data()),
        thrust::raw_pointer_cast(v_sc.data()),
        thrust::raw_pointer_cast(qkv_dev.data()),
        thrust::raw_pointer_cast(bt_dev.data()), max_pages_per_seq,
        page_block_size, B, T, Hq, Hkv, Hs, start_pos, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Dequantize to contiguous BF16 [B, max_seq_len, Hkv, Hs].
    const std::size_t contig_elems = static_cast<std::size_t>(B) * max_seq_len * Hkv * Hs;
    thrust::device_vector<nv_bfloat16> k_dq(contig_elems, __float2bfloat16(0.0f));
    thrust::device_vector<nv_bfloat16> v_dq(contig_elems, __float2bfloat16(0.0f));
    std::vector<int> sl_host = {start_pos + T};
    thrust::device_vector<int> sl_dev(sl_host.begin(), sl_host.end());

    kv_cache_dequant_paged_fp8_to_bf16(
        thrust::raw_pointer_cast(k_dq.data()),
        thrust::raw_pointer_cast(v_dq.data()),
        thrust::raw_pointer_cast(k_fp8.data()),
        thrust::raw_pointer_cast(v_fp8.data()),
        thrust::raw_pointer_cast(k_sc.data()),
        thrust::raw_pointer_cast(v_sc.data()),
        thrust::raw_pointer_cast(sl_dev.data()),
        thrust::raw_pointer_cast(bt_dev.data()), max_pages_per_seq,
        page_block_size, B, max_seq_len, Hkv, Hs, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto k_dq_h = from_bf16_device(k_dq);

    float max_rel_err = 0.0f;
    for (int t = 0; t < T; ++t) {
        const int abs_t = start_pos + t;
        for (int h = 0; h < Hkv; ++h) {
            for (int d = 0; d < Hs; ++d) {
                const float ref = qkv_host[static_cast<std::size_t>(t * H * Hs + (Hq + h) * Hs + d)];
                const float got = k_dq_h[static_cast<std::size_t>(abs_t * Hkv * Hs + h * Hs + d)];
                const float rel_err = std::abs(ref) > 0.01f
                    ? std::abs(got - ref) / std::abs(ref)
                    : std::abs(got - ref);
                max_rel_err = std::max(max_rel_err, rel_err);
            }
        }
    }
    INFO("FP8 paged bulk start_pos max rel error: " << max_rel_err);
    REQUIRE(max_rel_err < 0.2f);
}
