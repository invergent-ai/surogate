// Correctness + timing for the vendored Marlin W8G32 path against the serve
// weight layout: random codes/scales -> repack -> gemm, checked against a
// straightforward dequant reference, timed on the qwen3.5-4b decode shapes.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ops/linear/marlin/marlin_gemm.h"
#include "ops/linear/marlin/marlin_repack.h"

#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x)                                                                       \
    do {                                                                               \
        cudaError_t err = (x);                                                         \
        if (err != cudaSuccess) {                                                      \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,                     \
                         cudaGetErrorString(err));                                     \
            std::exit(1);                                                              \
        }                                                                              \
    } while (0)

using sinfer::ops::detail::marlin_b_out_words;
using sinfer::ops::detail::marlin_c_tmp_floats;
using sinfer::ops::detail::marlin_gemm_bf16;
using sinfer::ops::detail::marlin_repack_fp8_row;
using sinfer::ops::detail::marlin_repack_w8g32;
using sinfer::ops::detail::marlin_workspace_locks_count;

int main() {
    int sms = 0;
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    struct Shape { int n, k; const char* name; };
    const Shape shapes[] = {{2560, 9216, "down"},    {18432, 2560, "gate_up"},
                            {12288, 2560, "gdn_in"}, {2560, 4096, "out"},
                            {10240, 2560, "attn_in"}, {248320, 2560, "vocab"},
                            {7168, 1024, "q08_gate_up"}, {1024, 3584, "q08_down"},
                            {8192, 1024, "q08_gdn_in"}, {1024, 2048, "q08_out"},
                            {6144, 1024, "q08_attn_in"}, {248320, 1024, "q08_vocab"}};
    const int ts[] = {16, 24, 32, 48};

    for (const Shape& sh : shapes) {
        const int n = sh.n, k = sh.k;
        std::vector<std::uint8_t> h_codes(static_cast<std::size_t>(n) * k);
        std::vector<__half> h_scales(static_cast<std::size_t>(n) * (k / 32));
        srand(7);
        for (auto& c : h_codes) { c = static_cast<std::uint8_t>(rand() & 0xff); }
        for (auto& s : h_scales) {
            s = __float2half(0.001f + 0.002f * (rand() % 1000) / 1000.0f);
        }
        void *d_codes, *d_scales, *d_gptq, *d_b, *d_s;
        CHECK(cudaMalloc(&d_codes, h_codes.size()));
        CHECK(cudaMalloc(&d_scales, h_scales.size() * 2));
        CHECK(cudaMalloc(&d_gptq, static_cast<std::size_t>(k / 4) * n * 4));
        CHECK(cudaMalloc(&d_b, marlin_b_out_words(n, k) * 4));
        CHECK(cudaMalloc(&d_s, static_cast<std::size_t>(k / 32) * n * 2));
        CHECK(cudaMemcpy(d_codes, h_codes.data(), h_codes.size(), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_scales, h_scales.data(), h_scales.size() * 2,
                         cudaMemcpyHostToDevice));
        marlin_repack_w8g32(d_codes, d_scales, n, k, d_gptq, d_b, d_s, nullptr);
        CHECK(cudaDeviceSynchronize());

        const int t_max = 48;
        std::vector<__nv_bfloat16> h_a(static_cast<std::size_t>(t_max) * k);
        for (auto& v : h_a) { v = __float2bfloat16((rand() % 2000 - 1000) / 1000.0f); }
        void *d_a, *d_c, *d_ctmp;
        int* d_locks;
        CHECK(cudaMalloc(&d_a, h_a.size() * 2));
        CHECK(cudaMalloc(&d_c, static_cast<std::size_t>(t_max) * n * 2));
        CHECK(cudaMalloc(&d_ctmp, marlin_c_tmp_floats(sms, t_max) * 4));
        CHECK(cudaMalloc(&d_locks, marlin_workspace_locks_count(sms) * 4));
        CHECK(cudaMemset(d_locks, 0, marlin_workspace_locks_count(sms) * 4));
        CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * 2, cudaMemcpyHostToDevice));

        // correctness: one-hot activations pinpoint permutation errors —
        // out[row, col] must equal code[col, khot] * scale * x exactly (up
        // to bf16 rounding).
        {
            std::vector<__nv_bfloat16> h_hot(static_cast<std::size_t>(24) * k,
                                             __float2bfloat16(0.0f));
            for (int row = 0; row < 24; ++row) {
                const int khot = (row * 331 + 17) % k;
                h_hot[static_cast<std::size_t>(row) * k + khot] = __float2bfloat16(1.0f);
            }
            CHECK(cudaMemcpy(d_a, h_hot.data(), h_hot.size() * 2, cudaMemcpyHostToDevice));
        }
        marlin_gemm_bf16(d_a, d_b, d_s, d_c, d_ctmp, d_locks, 24, n, k, 32, false, sms,
                         nullptr);
        CHECK(cudaDeviceSynchronize());
        std::vector<__nv_bfloat16> h_c(static_cast<std::size_t>(24) * n);
        CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * 2, cudaMemcpyDeviceToHost));
        double max_rel = 0.0;
        for (int probe = 0; probe < 240; ++probe) {
            const int row  = probe % 24;
            const int khot = (row * 331 + 17) % k;
            const int col  = (probe * 104729 + 5) % n;
            const std::int8_t code =
                static_cast<std::int8_t>(h_codes[static_cast<std::size_t>(col) * k + khot]);
            const float scale =
                __half2float(h_scales[static_cast<std::size_t>(col) * (k / 32) + khot / 32]);
            const double ref = static_cast<double>(code) * scale;
            const double got = __bfloat162float(h_c[static_cast<std::size_t>(row) * n + col]);
            const double rel = std::fabs(got - ref) / (std::fabs(ref) + 1e-6);
            if (rel > max_rel) { max_rel = rel; }
        }
        CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * 2, cudaMemcpyHostToDevice));
        std::printf("%-8s N=%6d K=%5d  max_rel=%.4f %s\n", sh.name, n, k, max_rel,
                    max_rel < 0.05 ? "OK" : "FAIL");

        for (int t : ts) {
            for (int i = 0; i < 20; ++i) {
                marlin_gemm_bf16(d_a, d_b, d_s, d_c, d_ctmp, d_locks, t, n, k, 32, false, sms,
                                 nullptr);
            }
            CHECK(cudaDeviceSynchronize());
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            cudaEventRecord(e0);
            const int iters = 200;
            for (int i = 0; i < iters; ++i) {
                marlin_gemm_bf16(d_a, d_b, d_s, d_c, d_ctmp, d_locks, t, n, k, 32, false, sms,
                                 nullptr);
            }
            cudaEventRecord(e1);
            CHECK(cudaEventSynchronize(e1));
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            const double us  = ms * 1000.0 / iters;
            const double gbs = static_cast<double>(n) * k / 1e9 / (us / 1e6);
            std::printf("  T=%2d  %7.1f us  %6.0f GB/s\n", t, us, gbs);
        }
        cudaFree(d_codes); cudaFree(d_scales); cudaFree(d_gptq); cudaFree(d_b);
        cudaFree(d_s); cudaFree(d_a); cudaFree(d_c); cudaFree(d_ctmp); cudaFree(d_locks);
    }

    // FP8 residency path (PATCHES.md #38): the 27B's decode shapes, checked
    // the same way — one-hot activations make each output equal exactly one
    // code times its channel scale.
    std::printf("\n== FP8 (e4m3, per-channel scales) ==\n");
    const Shape fp8_shapes[] = {{34816, 5120, "gate_up"}, {5120, 17408, "down"},
                                {16384, 5120, "gdn_in"},  {5120, 6144, "out"},
                                {14336, 5120, "attn_in"}};
    for (const Shape& sh : fp8_shapes) {
        const int n = sh.n, k = sh.k;
        std::vector<std::uint8_t> h_codes(static_cast<std::size_t>(n) * k);
        std::vector<__nv_bfloat16> h_scales(static_cast<std::size_t>(n));
        srand(11);
        for (auto& c : h_codes) {
            const float v = ((rand() % 2000) - 1000) / 500.0f;
            const __nv_fp8_e4m3 code(v);
            c = static_cast<std::uint8_t>(code.__x);
        }
        for (auto& v : h_scales) { v = __float2bfloat16(0.01f + 0.02f * (rand() % 100) / 100.0f); }
        void *d_codes, *d_scales, *d_gptq, *d_b, *d_s;
        CHECK(cudaMalloc(&d_codes, h_codes.size()));
        CHECK(cudaMalloc(&d_scales, h_scales.size() * 2));
        CHECK(cudaMalloc(&d_gptq, static_cast<std::size_t>(k / 4) * n * 4));
        CHECK(cudaMalloc(&d_b, marlin_b_out_words(n, k) * 4));
        CHECK(cudaMalloc(&d_s, static_cast<std::size_t>(n) * 2));
        CHECK(cudaMemcpy(d_codes, h_codes.data(), h_codes.size(), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_scales, h_scales.data(), h_scales.size() * 2, cudaMemcpyHostToDevice));
        marlin_repack_fp8_row(d_codes, d_scales, n, k, d_gptq, d_b, d_s, nullptr);
        CHECK(cudaDeviceSynchronize());

        const int t = 32;
        std::vector<__nv_bfloat16> h_hot(static_cast<std::size_t>(t) * k, __float2bfloat16(0.0f));
        for (int row = 0; row < t; ++row) {
            h_hot[static_cast<std::size_t>(row) * k + (row * 331 + 17) % k] =
                __float2bfloat16(1.0f);
        }
        void *d_a, *d_c, *d_ctmp;
        int* d_locks;
        CHECK(cudaMalloc(&d_a, h_hot.size() * 2));
        CHECK(cudaMalloc(&d_c, static_cast<std::size_t>(t) * n * 2));
        CHECK(cudaMalloc(&d_ctmp, marlin_c_tmp_floats(sms, t) * 4));
        CHECK(cudaMalloc(&d_locks, marlin_workspace_locks_count(sms) * 4));
        CHECK(cudaMemset(d_locks, 0, marlin_workspace_locks_count(sms) * 4));
        CHECK(cudaMemcpy(d_a, h_hot.data(), h_hot.size() * 2, cudaMemcpyHostToDevice));
        marlin_gemm_bf16(d_a, d_b, d_s, d_c, d_ctmp, d_locks, t, n, k, /*group_size=*/-1,
                         /*b_is_fp8=*/true, sms, nullptr);
        CHECK(cudaDeviceSynchronize());
        std::vector<__nv_bfloat16> h_c(static_cast<std::size_t>(t) * n);
        CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * 2, cudaMemcpyDeviceToHost));
        double max_rel = 0.0;
        for (int probe = 0; probe < 320; ++probe) {
            const int row  = probe % t;
            const int khot = (row * 331 + 17) % k;
            const int col  = (probe * 104729 + 5) % n;
            __nv_fp8_e4m3 code{};
            code.__x = h_codes[static_cast<std::size_t>(col) * k + khot];
            const double ref = static_cast<double>(static_cast<float>(code)) *
                               __bfloat162float(h_scales[col]);
            const double got = __bfloat162float(h_c[static_cast<std::size_t>(row) * n + col]);
            const double rel = std::fabs(got - ref) / (std::fabs(ref) + 1e-6);
            if (rel > max_rel) { max_rel = rel; }
        }
        std::printf("%-8s N=%6d K=%5d  max_rel=%.4f %s\n", sh.name, n, k, max_rel,
                    max_rel < 0.05 ? "OK" : "FAIL");
        for (int tt : {16, 24, 32}) {
            for (int i = 0; i < 20; ++i) {
                marlin_gemm_bf16(d_a, d_b, d_s, d_c, d_ctmp, d_locks, tt, n, k, -1, true, sms,
                                 nullptr);
            }
            CHECK(cudaDeviceSynchronize());
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            cudaEventRecord(e0);
            for (int i = 0; i < 200; ++i) {
                marlin_gemm_bf16(d_a, d_b, d_s, d_c, d_ctmp, d_locks, tt, n, k, -1, true, sms,
                                 nullptr);
            }
            cudaEventRecord(e1);
            CHECK(cudaEventSynchronize(e1));
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            const double us = ms * 1000.0 / 200;
            std::printf("  T=%2d  %7.1f us  %6.0f GB/s\n", tt, us,
                        static_cast<double>(n) * k / 1e9 / (us / 1e6));
        }
        cudaFree(d_codes); cudaFree(d_scales); cudaFree(d_gptq); cudaFree(d_b);
        cudaFree(d_s); cudaFree(d_a); cudaFree(d_c); cudaFree(d_ctmp); cudaFree(d_locks);
    }
    return 0;
}
