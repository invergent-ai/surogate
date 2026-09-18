// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
// Compare optimized matrix products bit-for-bit with GGML's original dot kernels.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <bit>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

extern "C" {
void ggml_vec_dot_f32(int, float*, size_t, const float*, size_t, const float*, size_t, int);
void ggml_vec_dot_f16(int, float*, size_t, ggml_fp16_t*, size_t, ggml_fp16_t*, size_t, int);
}

static void check(ggml_backend_t backend, ggml_type type, int k, int m, int n, int batches, int threads, bool padded) {
    auto* ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    if (!ctx) throw std::runtime_error("Cannot allocate test context");
    const int stride = k + (padded ? 7 : 0);
    auto* a_storage = ggml_new_tensor_2d(ctx, type, stride, m);
    auto* b_storage = ggml_new_tensor_3d(ctx, type, stride, n, batches);
    const size_t row_bytes = stride * ggml_type_size(type);
    auto* a = padded ? ggml_view_2d(ctx, a_storage, k, m, row_bytes, 0) : a_storage;
    auto* b = padded ? ggml_view_3d(ctx, b_storage, k, n, batches, row_bytes, row_bytes * n, 0) : b_storage;
    auto* result = ggml_mul_mat(ctx, a, b);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) throw std::runtime_error("Cannot allocate matrix test buffers");
    std::mt19937 random(1009 + k * 7 + m * 13 + n * 19 + batches);
    auto values = [&](size_t size) {
        std::vector<float> data(size);
        for (auto& x : data) {
            // Include cancellation and mixed magnitudes without NaNs/infinities.
            x = float(int(random() % 20001) - 10000) / (1 << (random() % 14));
        }
        return data;
    };
    auto av = values(stride * m), bv = values(stride * n * batches);
    std::vector<ggml_fp16_t> ah(av.size()), bh(bv.size());
    if (type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(a_storage, av.data(), 0, av.size() * sizeof(float));
        ggml_backend_tensor_set(b_storage, bv.data(), 0, bv.size() * sizeof(float));
    } else {
        ggml_fp32_to_fp16_row(av.data(), ah.data(), av.size());
        ggml_fp32_to_fp16_row(bv.data(), bh.data(), bv.size());
        ggml_backend_tensor_set(a_storage, ah.data(), 0, ah.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(b_storage, bh.data(), 0, bh.size() * sizeof(ggml_fp16_t));
    }
    ggml_backend_cpu_set_n_threads(backend, threads);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("Matrix graph failed");
    std::vector<float> actual(m * n * batches);
    ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));
    for (int c = 0; c < n * batches; ++c) {
        for (int r = 0; r < m; ++r) {
            float expected;
            if (type == GGML_TYPE_F32)
                ggml_vec_dot_f32(k, &expected, 0, av.data() + r * stride, 0, bv.data() + c * stride, 0, 1);
            else
                ggml_vec_dot_f16(k, &expected, 0, ah.data() + r * stride, 0, bh.data() + c * stride, 0, 1);
            if (std::bit_cast<uint32_t>(actual[c * m + r]) != std::bit_cast<uint32_t>(expected)) {
                std::cerr << "Mismatch: type=" << type << " K=" << k << " M=" << m << " N=" << n
                          << " batches=" << batches << " threads=" << threads << " padded=" << padded << " row=" << r
                          << " column=" << c << '\n';
                throw std::runtime_error("CPU matrix kernel changed reference arithmetic");
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main() {
#if defined(__x86_64__)
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("avx512bw") ||
        !__builtin_cpu_supports("avx512dq") || !__builtin_cpu_supports("avx512vl") || !__builtin_cpu_supports("f16c") ||
        !__builtin_cpu_supports("fma"))
        return 77;
#else
    return 77;
#endif
    try {
        auto backend = ggml_backend_cpu_init();
        int cases = 0;
        for (int threads : {1, 2, 4, 8}) {
            for (bool padded : {false, true}) {
                for (int k : {63, 64, 65, 81, 128, 189, 297, 378, 594, 756, 1188, 2376}) {
                    for (int m : {1, 2, 3, 8, 17}) {
                        for (int n : {1, 2, 3, 4, 7}) {
                            check(backend, GGML_TYPE_F16, k, m, n, 1, threads, padded);
                            ++cases;
                        }
                    }
                }
                for (int k : {63, 64, 65, 128, 768, 2304, 3072}) {
                    for (int m : {1, 2, 3, 16, 31}) {
                        for (int n : {1, 2, 3}) {
                            check(backend, GGML_TYPE_F32, k, m, n, 1, threads, padded);
                            ++cases;
                        }
                        check(backend, GGML_TYPE_F32, k, m, 1, 2, threads, padded);
                        ++cases;
                    }
                }
            }
        }
        ggml_backend_free(backend);
        std::cout << cases << " matrix cases matched the original dot kernels exactly\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
