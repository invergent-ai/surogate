// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Parity tests for chunked-sequence (KV-prefix) attention against the dense
// varlen path: a T-token sequence processed as N chunks with a growing KV
// cache and reverse-order backward with dK/dV accumulation must reproduce the
// full-sequence forward output, LSE and dQKV to bf16 tolerance. Covers full
// causal and sliding-window (window crossing chunk boundaries), GQA shapes.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "runtime/attention/attention_kernels.h"

namespace {

struct DeviceBuf {
    void* ptr = nullptr;
    explicit DeviceBuf(std::size_t bytes) {
        REQUIRE(cudaMalloc(&ptr, bytes) == cudaSuccess);
        REQUIRE(cudaMemset(ptr, 0, bytes) == cudaSuccess);
    }
    ~DeviceBuf() {
        cudaFree(ptr);
    }
    template <typename T>
    T* as() {
        return static_cast<T*>(ptr);
    }
};

std::vector<nv_bfloat16> random_bf16(std::size_t n, unsigned seed) {
    std::srand(seed);
    std::vector<nv_bfloat16> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = __float2bfloat16((static_cast<float>(std::rand()) / RAND_MAX) * 2.0f - 1.0f);
    }
    return v;
}

float max_abs_diff_bf16(const std::vector<nv_bfloat16>& a, const std::vector<nv_bfloat16>& b) {
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(__bfloat162float(a[i]) - __bfloat162float(b[i])));
    }
    return m;
}

float max_abs_diff_f32(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

void run_parity_case(int T_total, int chunk, int Hq, int Hkv, int HS, int window) {
    const int H = Hq + 2 * Hkv;
    const int num_chunks = T_total / chunk;
    const std::size_t qkv_elems = static_cast<std::size_t>(T_total) * H * HS;
    const std::size_t out_elems = static_cast<std::size_t>(T_total) * Hq * HS;
    const std::size_t lse_elems = static_cast<std::size_t>(Hq) * T_total;
    const std::size_t kv_elems = static_cast<std::size_t>(T_total) * Hkv * HS;
    const int HS_rounded = HS <= 128 ? ((HS + 31) / 32) * 32 : ((HS + 63) / 64) * 64;

    const auto h_qkv = random_bf16(qkv_elems, 1234);
    const auto h_dout = random_bf16(out_elems, 4321);

    cudaStream_t stream = nullptr;

    DeviceBuf qkv(qkv_elems * 2);
    DeviceBuf dout(out_elems * 2);
    REQUIRE(cudaMemcpy(qkv.ptr, h_qkv.data(), qkv_elems * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dout.ptr, h_dout.data(), out_elems * 2, cudaMemcpyHostToDevice) == cudaSuccess);

    // ---- Reference: dense varlen over the full sequence -----------------
    DeviceBuf ref_out(out_elems * 2);
    DeviceBuf ref_lse(lse_elems * 4);
    DeviceBuf ref_dqkv(qkv_elems * 2);
    {
        const int32_t cu[2] = {0, T_total};
        DeviceBuf cu_gpu(sizeof(cu));
        REQUIRE(cudaMemcpy(cu_gpu.ptr, cu, sizeof(cu), cudaMemcpyHostToDevice) == cudaSuccess);

        attention_forward_flash_varlen(ref_out.as<nv_bfloat16>(),
                                       ref_lse.as<float>(),
                                       qkv.as<nv_bfloat16>(),
                                       cu_gpu.as<int32_t>(),
                                       /*B_ragged=*/1,
                                       /*max_seqlen=*/T_total,
                                       /*total_q=*/T_total,
                                       Hq,
                                       Hkv,
                                       HS,
                                       stream,
                                       0.0f,
                                       window);

        const std::size_t dq_accum_elems = static_cast<std::size_t>(T_total + 128) * Hq * HS_rounded;
        DeviceBuf dq_accum(dq_accum_elems * 4);
        DeviceBuf dsoftmax(static_cast<std::size_t>(Hq) * (T_total + 128) * 4);
        DeviceBuf dk_exp(static_cast<std::size_t>(T_total) * Hq * HS * 2);
        DeviceBuf dv_exp(static_cast<std::size_t>(T_total) * Hq * HS * 2);

        attention_backward_flash_varlen(ref_dqkv.as<nv_bfloat16>(),
                                        ref_lse.as<float>(),
                                        ref_out.as<nv_bfloat16>(),
                                        dout.as<nv_bfloat16>(),
                                        qkv.as<nv_bfloat16>(),
                                        cu_gpu.as<int32_t>(),
                                        dq_accum.as<float>(),
                                        dsoftmax.as<float>(),
                                        dk_exp.as<nv_bfloat16>(),
                                        dv_exp.as<nv_bfloat16>(),
                                        /*B_ragged=*/1,
                                        T_total,
                                        T_total,
                                        Hq,
                                        Hkv,
                                        HS,
                                        /*deterministic=*/false,
                                        stream,
                                        0.0f,
                                        window);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    }

    // ---- Chunked: forward chunks 0..N-1, backward N-1..0 ----------------
    DeviceBuf chk_out(out_elems * 2);
    DeviceBuf chk_lse_all(static_cast<std::size_t>(num_chunks) * Hq * chunk * 4);
    DeviceBuf chk_dqkv(qkv_elems * 2);
    DeviceBuf k_cache(kv_elems * 2);
    DeviceBuf v_cache(kv_elems * 2);
    DeviceBuf dk_accum(kv_elems * 4);
    DeviceBuf dv_accum(kv_elems * 4);

    for (int c = 0; c < num_chunks; ++c) {
        const int pos = c * chunk;
        const int kv_len = pos + chunk;
        const nv_bfloat16* qkv_c = qkv.as<nv_bfloat16>() + static_cast<std::size_t>(pos) * H * HS;

        append_kv_to_cache(k_cache.as<nv_bfloat16>(),
                           v_cache.as<nv_bfloat16>(),
                           qkv_c,
                           pos,
                           chunk,
                           Hq,
                           Hkv,
                           HS,
                           stream);

        const int32_t cu_q[2] = {0, chunk};
        const int32_t cu_k[2] = {0, kv_len};
        DeviceBuf cu_q_gpu(sizeof(cu_q));
        DeviceBuf cu_k_gpu(sizeof(cu_k));
        REQUIRE(cudaMemcpy(cu_q_gpu.ptr, cu_q, sizeof(cu_q), cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(cu_k_gpu.ptr, cu_k, sizeof(cu_k), cudaMemcpyHostToDevice) == cudaSuccess);

        attention_forward_flash_kvprefix(chk_out.as<nv_bfloat16>() + static_cast<std::size_t>(pos) * Hq * HS,
                                         chk_lse_all.as<float>() + static_cast<std::size_t>(c) * Hq * chunk,
                                         qkv_c,
                                         k_cache.as<nv_bfloat16>(),
                                         v_cache.as<nv_bfloat16>(),
                                         cu_q_gpu.as<int32_t>(),
                                         cu_k_gpu.as<int32_t>(),
                                         chunk,
                                         kv_len,
                                         Hq,
                                         Hkv,
                                         HS,
                                         stream,
                                         0.0f,
                                         window);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    }

    for (int c = num_chunks - 1; c >= 0; --c) {
        const int pos = c * chunk;
        const int kv_len = pos + chunk;
        const nv_bfloat16* qkv_c = qkv.as<nv_bfloat16>() + static_cast<std::size_t>(pos) * H * HS;

        const int32_t cu_q[2] = {0, chunk};
        const int32_t cu_k[2] = {0, kv_len};
        DeviceBuf cu_q_gpu(sizeof(cu_q));
        DeviceBuf cu_k_gpu(sizeof(cu_k));
        REQUIRE(cudaMemcpy(cu_q_gpu.ptr, cu_q, sizeof(cu_q), cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(cu_k_gpu.ptr, cu_k, sizeof(cu_k), cudaMemcpyHostToDevice) == cudaSuccess);

        const std::size_t dq_accum_elems = static_cast<std::size_t>(chunk + 128) * Hq * HS_rounded;
        DeviceBuf dq_accum(dq_accum_elems * 4);
        DeviceBuf dsoftmax(static_cast<std::size_t>(Hq) * (chunk + 128) * 4);
        DeviceBuf dk_exp(static_cast<std::size_t>(kv_len) * Hq * HS * 2);
        DeviceBuf dv_exp(static_cast<std::size_t>(kv_len) * Hq * HS * 2);

        attention_backward_flash_kvprefix(chk_dqkv.as<nv_bfloat16>() + static_cast<std::size_t>(pos) * H * HS,
                                          dk_accum.as<float>(),
                                          dv_accum.as<float>(),
                                          chk_lse_all.as<float>() + static_cast<std::size_t>(c) * Hq * chunk,
                                          chk_out.as<nv_bfloat16>() + static_cast<std::size_t>(pos) * Hq * HS,
                                          dout.as<nv_bfloat16>() + static_cast<std::size_t>(pos) * Hq * HS,
                                          qkv_c,
                                          k_cache.as<nv_bfloat16>(),
                                          v_cache.as<nv_bfloat16>(),
                                          cu_q_gpu.as<int32_t>(),
                                          cu_k_gpu.as<int32_t>(),
                                          dq_accum.as<float>(),
                                          dsoftmax.as<float>(),
                                          dk_exp.as<nv_bfloat16>(),
                                          dv_exp.as<nv_bfloat16>(),
                                          pos,
                                          chunk,
                                          kv_len,
                                          Hq,
                                          Hkv,
                                          HS,
                                          /*deterministic=*/false,
                                          stream,
                                          0.0f,
                                          window);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    }

    // ---- Compare --------------------------------------------------------
    std::vector<nv_bfloat16> h_ref_out(out_elems), h_chk_out(out_elems);
    std::vector<nv_bfloat16> h_ref_dqkv(qkv_elems), h_chk_dqkv(qkv_elems);
    REQUIRE(cudaMemcpy(h_ref_out.data(), ref_out.ptr, out_elems * 2, cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(h_chk_out.data(), chk_out.ptr, out_elems * 2, cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(h_ref_dqkv.data(), ref_dqkv.ptr, qkv_elems * 2, cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(h_chk_dqkv.data(), chk_dqkv.ptr, qkv_elems * 2, cudaMemcpyDeviceToHost) == cudaSuccess);

    // LSE: reference is unpadded (Hq, T_total); chunked is per-chunk (Hq, chunk).
    std::vector<float> h_ref_lse(lse_elems), h_chk_lse(lse_elems);
    REQUIRE(cudaMemcpy(h_ref_lse.data(), ref_lse.ptr, lse_elems * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    {
        std::vector<float> tmp(static_cast<std::size_t>(num_chunks) * Hq * chunk);
        REQUIRE(cudaMemcpy(tmp.data(), chk_lse_all.ptr, tmp.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
        for (int c = 0; c < num_chunks; ++c) {
            for (int h = 0; h < Hq; ++h) {
                for (int t = 0; t < chunk; ++t) {
                    h_chk_lse[static_cast<std::size_t>(h) * T_total + c * chunk + t] =
                        tmp[static_cast<std::size_t>(c) * Hq * chunk + static_cast<std::size_t>(h) * chunk + t];
                }
            }
        }
    }

    INFO("case T=" << T_total << " chunk=" << chunk << " Hq=" << Hq << " Hkv=" << Hkv << " HS=" << HS
                   << " window=" << window);
    REQUIRE(max_abs_diff_bf16(h_ref_out, h_chk_out) < 2e-2f);
    REQUIRE(max_abs_diff_f32(h_ref_lse, h_chk_lse) < 2e-3f);
    REQUIRE(max_abs_diff_bf16(h_ref_dqkv, h_chk_dqkv) < 5e-2f);
}

}  // namespace

TEST_CASE("chunked kv-prefix attention matches dense (full causal, GQA)", "[attention][chunked]") {
    run_parity_case(/*T_total=*/512, /*chunk=*/256, /*Hq=*/8, /*Hkv=*/2, /*HS=*/128, /*window=*/0);
}

TEST_CASE("chunked kv-prefix attention matches dense (4 chunks)", "[attention][chunked]") {
    run_parity_case(/*T_total=*/512, /*chunk=*/128, /*Hq=*/8, /*Hkv=*/2, /*HS=*/128, /*window=*/0);
}

TEST_CASE("chunked kv-prefix attention matches dense (sliding window across chunks)", "[attention][chunked]") {
    run_parity_case(/*T_total=*/512, /*chunk=*/256, /*Hq=*/8, /*Hkv=*/2, /*HS=*/128, /*window=*/384);
}

TEST_CASE("chunked kv-prefix attention matches dense (MHA)", "[attention][chunked]") {
    run_parity_case(/*T_total=*/512, /*chunk=*/256, /*Hq=*/4, /*Hkv=*/4, /*HS=*/64, /*window=*/0);
}
