// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Deterministic value dump for surogate gated delta rule (forward + backward).
// Use together with tests/test_gated_delta_rule_reference.py (FLA values).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "../utilities/test_utils.h"
#include "kernels/kernels.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace testing_utils;

namespace {

std::vector<float> make_signal(std::size_t n, float a, float b, float c, float d) {
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float x = static_cast<float>(i);
        out[i] = 0.7f * std::sin(x * a + b) + 0.3f * std::cos(x * c + d);
    }
    return out;
}

std::vector<float> bf16_to_float(const std::vector<nv_bfloat16>& in) {
    std::vector<float> out(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
        std::uint16_t bits;
        std::memcpy(&bits, &in[i], sizeof(bits));
        out[i] = bf16_bits_to_float(bits);
    }
    return out;
}

json stats(const std::vector<float>& x) {
    double sum = 0.0;
    double l1 = 0.0;
    double l2 = 0.0;
    double linf = 0.0;
    for (float v : x) {
        const double dv = static_cast<double>(v);
        sum += dv;
        l1 += std::fabs(dv);
        l2 += dv * dv;
        linf = std::max(linf, std::fabs(dv));
    }

    json first8 = json::array();
    for (std::size_t i = 0; i < std::min<std::size_t>(8, x.size()); ++i) {
        first8.push_back(static_cast<double>(x[i]));
    }

    json j;
    j["numel"] = x.size();
    j["sum"] = sum;
    j["mean"] = x.empty() ? 0.0 : (sum / static_cast<double>(x.size()));
    j["l1"] = l1;
    j["l2"] = std::sqrt(l2);
    j["linf"] = linf;
    j["first8"] = std::move(first8);
    json values = json::array();
    for (float v : x) {
        values.push_back(static_cast<double>(v));
    }
    j["values"] = std::move(values);
    return j;
}

fs::path find_repo_root() {
    fs::path cwd = fs::current_path();
    for (int i = 0; i < 10; ++i) {
        if (fs::exists(cwd / "csrc") && fs::exists(cwd / "tests")) {
            return cwd;
        }
        if (!cwd.has_parent_path()) {
            break;
        }
        cwd = cwd.parent_path();
    }
    throw std::runtime_error("Could not locate repository root from current path");
}

template <typename T>
Tensor tensor_view(thrust::device_vector<T>& vec, ETensorDType dtype, const std::vector<long>& shape) {
    return Tensor::from_pointer(
        reinterpret_cast<std::byte*>(thrust::raw_pointer_cast(vec.data())),
        /*device=*/0,
        dtype,
        shape);
}

template <typename Fn>
float benchmark_cuda_ms(int warmup_iters, int bench_iters, cudaStream_t stream, Fn&& fn) {
    for (int i = 0; i < warmup_iters; ++i) {
        fn();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < bench_iters; ++i) {
        fn();
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / static_cast<float>(bench_iters);
}

struct BenchConfig {
    int B;
    int T;
    int H;
    int K;
    int V;
};

struct DumpConfig {
    const char* case_name;
    int B;
    int T;
    int H;
    int K;
    int V;
};

json benchmark_case(const BenchConfig& cfg, int warmup_iters, int bench_iters) {
    const int B = cfg.B;
    const int T = cfg.T;
    const int H = cfg.H;
    const int K = cfg.K;
    const int V = cfg.V;
    const int chunk_size = 64;
    const bool use_qk_l2norm_in_kernel = true;
    const float scale = 1.0f / std::sqrt(static_cast<float>(K));
    const cudaStream_t stream = 0;

    const std::size_t n_qkvk = static_cast<std::size_t>(B) * T * H * K;
    const std::size_t n_qvv = static_cast<std::size_t>(B) * T * H * V;
    const std::size_t n_gh = static_cast<std::size_t>(B) * T * H;
    const std::size_t n_state = static_cast<std::size_t>(B) * H * K * V;

    std::vector<float> q_f = make_signal(n_qkvk, 0.173f, 0.31f, 0.097f, -0.22f);
    std::vector<float> k_f = make_signal(n_qkvk, 0.137f, -0.41f, 0.083f, 0.57f);
    std::vector<float> v_f = make_signal(n_qvv, 0.191f, 0.73f, 0.121f, -0.35f);

    std::vector<float> g_raw = make_signal(n_gh, 0.157f, -0.27f, 0.109f, 0.44f);
    std::vector<float> g_f(n_gh);
    for (std::size_t i = 0; i < n_gh; ++i) {
        g_f[i] = -0.7f + 0.2f * std::sin(g_raw[i]);
    }

    std::vector<float> beta_raw = make_signal(n_gh, 0.113f, 0.19f, 0.071f, -0.63f);
    std::vector<float> beta_f(n_gh);
    for (std::size_t i = 0; i < n_gh; ++i) {
        beta_f[i] = 1.0f / (1.0f + std::exp(-beta_raw[i]));
    }

    std::vector<float> initial_state_f = make_signal(n_state, 0.167f, -0.52f, 0.061f, 0.28f);
    std::vector<float> d_out_f = make_signal(n_qvv, 0.149f, 0.66f, 0.101f, -0.14f);
    std::vector<float> d_final_state_f = make_signal(n_state, 0.089f, -0.33f, 0.055f, 0.47f);

    thrust::device_vector<nv_bfloat16> d_q = to_device(to_bf16(q_f));
    thrust::device_vector<nv_bfloat16> d_k = to_device(to_bf16(k_f));
    thrust::device_vector<nv_bfloat16> d_v = to_device(to_bf16(v_f));
    thrust::device_vector<nv_bfloat16> d_g = to_device(to_bf16(g_f));
    thrust::device_vector<nv_bfloat16> d_beta = to_device(to_bf16(beta_f));
    thrust::device_vector<float> d_initial_state = to_device(initial_state_f);

    thrust::device_vector<nv_bfloat16> d_out(n_qvv);
    thrust::device_vector<float> d_final_state(n_state);
    thrust::device_vector<float> d_forward_scratch(n_state);

    thrust::device_vector<nv_bfloat16> d_dq(n_qkvk);
    thrust::device_vector<nv_bfloat16> d_dk(n_qkvk);
    thrust::device_vector<nv_bfloat16> d_dv(n_qvv);
    thrust::device_vector<nv_bfloat16> d_dg(n_gh);
    thrust::device_vector<nv_bfloat16> d_dbeta(n_gh);
    thrust::device_vector<float> d_dinitial(n_state);
    thrust::device_vector<nv_bfloat16> d_dout = to_device(to_bf16(d_out_f));
    thrust::device_vector<float> d_dfinal = to_device(d_final_state_f);

    const int num_chunks = (T + chunk_size - 1) / chunk_size;
    const std::size_t n_checkpoints =
        static_cast<std::size_t>(B) * H * static_cast<std::size_t>(num_chunks + 1) * K * V;
    const int Lp = 64;
    constexpr std::size_t kWsAlignFloats = 128 / sizeof(float);
    const std::size_t c_storage_floats = static_cast<std::size_t>(K) * K;
    auto align_up = [](std::size_t x, std::size_t align) {
        return ((x + align - 1) / align) * align;
    };
    std::size_t chunk_ws_stride = 0;
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(Lp) * Lp; // M
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(Lp) * Lp; // A
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(Lp) * K;  // W
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(Lp) * V;  // VNEW
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(Lp) * V;  // DU
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(Lp) * K;  // DW
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(Lp) * K;  // DQ
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(Lp) * K;  // DK
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(Lp);      // DG
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(Lp);      // DB
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(K) * V;   // DHT1
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + c_storage_floats;                  // C fp32
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + 1;                                  // EG
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats);                                       // total stride
    const std::size_t dh_storage_per_chunk = static_cast<std::size_t>(K) * V;
    const std::size_t workspace_size =
        static_cast<std::size_t>(num_chunks) * chunk_ws_stride
        + static_cast<std::size_t>(num_chunks) * dh_storage_per_chunk;
    thrust::device_vector<float> d_checkpoints(n_checkpoints);
    thrust::device_vector<float> d_state_scratch(static_cast<std::size_t>(B) * H * workspace_size);

    Tensor q_t = tensor_view(d_q, ETensorDType::BF16, {B, T, H, K});
    Tensor k_t = tensor_view(d_k, ETensorDType::BF16, {B, T, H, K});
    Tensor v_t = tensor_view(d_v, ETensorDType::BF16, {B, T, H, V});
    Tensor g_t = tensor_view(d_g, ETensorDType::BF16, {B, T, H});
    Tensor beta_t = tensor_view(d_beta, ETensorDType::BF16, {B, T, H});
    Tensor init_t = tensor_view(d_initial_state, ETensorDType::FP32, {B, H, K, V});
    Tensor out_t = tensor_view(d_out, ETensorDType::BF16, {B, T, H, V});
    Tensor final_state_t = tensor_view(d_final_state, ETensorDType::FP32, {B, H, K, V});
    Tensor forward_scratch_t = tensor_view(d_forward_scratch, ETensorDType::FP32, {B, H, K, V});

    Tensor dq_t = tensor_view(d_dq, ETensorDType::BF16, {B, T, H, K});
    Tensor dk_t = tensor_view(d_dk, ETensorDType::BF16, {B, T, H, K});
    Tensor dv_t = tensor_view(d_dv, ETensorDType::BF16, {B, T, H, V});
    Tensor dg_t = tensor_view(d_dg, ETensorDType::BF16, {B, T, H});
    Tensor dbeta_t = tensor_view(d_dbeta, ETensorDType::BF16, {B, T, H});
    Tensor dinit_t = tensor_view(d_dinitial, ETensorDType::FP32, {B, H, K, V});
    Tensor dout_t = tensor_view(d_dout, ETensorDType::BF16, {B, T, H, V});
    Tensor dfinal_t = tensor_view(d_dfinal, ETensorDType::FP32, {B, H, K, V});
    Tensor checkpoints_t =
        tensor_view(d_checkpoints, ETensorDType::FP32, {B, H, num_chunks + 1, K, V});
    Tensor state_scratch_t =
        tensor_view(d_state_scratch, ETensorDType::FP32, {B, H, static_cast<long>(workspace_size)});

    auto run_fwd = [&]() {
        gated_delta_rule_chunk_forward_v2(
            out_t,
            final_state_t,
            forward_scratch_t,
            q_t,
            k_t,
            v_t,
            g_t,
            beta_t,
            &init_t,
            scale,
            chunk_size,
            use_qk_l2norm_in_kernel,
            &checkpoints_t,   // save checkpoints during forward
            stream);
    };
    auto run_bwd = [&]() {
        gated_delta_rule_chunk_backward_v2(
            dq_t,
            dk_t,
            dv_t,
            dg_t,
            dbeta_t,
            dinit_t,
            dout_t,
            &dfinal_t,
            q_t,
            k_t,
            v_t,
            g_t,
            beta_t,
            &init_t,
            scale,
            chunk_size,
            use_qk_l2norm_in_kernel,
            checkpoints_t,
            state_scratch_t,
            /*skip_checkpoint=*/true,
            stream);
    };
    auto run_total = [&]() {
        run_fwd();
        run_bwd();
    };

    const float fwd_ms = benchmark_cuda_ms(warmup_iters, bench_iters, stream, run_fwd);
    const float bwd_ms = benchmark_cuda_ms(warmup_iters, bench_iters, stream, run_bwd);
    const float total_ms = benchmark_cuda_ms(warmup_iters, bench_iters, stream, run_total);

    return json{
        {"B", B},
        {"T", T},
        {"H", H},
        {"K", K},
        {"V", V},
        {"chunk_size", chunk_size},
        {"dtype", "bfloat16"},
        {"use_qk_l2norm_in_kernel", use_qk_l2norm_in_kernel},
        {"forward_ms", fwd_ms},
        {"backward_ms", bwd_ms},
        {"total_ms", total_ms},
    };
}

json dump_case(const DumpConfig& cfg) {
    const int B = cfg.B;
    const int T = cfg.T;
    const int H = cfg.H;
    const int K = cfg.K;
    const int V = cfg.V;
    const int chunk_size = 64;
    const bool use_qk_l2norm_in_kernel = true;
    const float scale = 1.0f / std::sqrt(static_cast<float>(K));

    const std::size_t n_qkvk = static_cast<std::size_t>(B) * T * H * K;
    const std::size_t n_qvv = static_cast<std::size_t>(B) * T * H * V;
    const std::size_t n_gh = static_cast<std::size_t>(B) * T * H;
    const std::size_t n_state = static_cast<std::size_t>(B) * H * K * V;

    std::vector<float> q_f = make_signal(n_qkvk, 0.173f, 0.31f, 0.097f, -0.22f);
    std::vector<float> k_f = make_signal(n_qkvk, 0.137f, -0.41f, 0.083f, 0.57f);
    std::vector<float> v_f = make_signal(n_qvv, 0.191f, 0.73f, 0.121f, -0.35f);

    std::vector<float> g_raw = make_signal(n_gh, 0.157f, -0.27f, 0.109f, 0.44f);
    std::vector<float> g_f(n_gh);
    for (std::size_t i = 0; i < n_gh; ++i) {
        g_f[i] = -0.7f + 0.2f * std::sin(g_raw[i]);
    }

    std::vector<float> beta_raw = make_signal(n_gh, 0.113f, 0.19f, 0.071f, -0.63f);
    std::vector<float> beta_f(n_gh);
    for (std::size_t i = 0; i < n_gh; ++i) {
        beta_f[i] = 1.0f / (1.0f + std::exp(-beta_raw[i]));
    }

    std::vector<float> initial_state_f = make_signal(n_state, 0.167f, -0.52f, 0.061f, 0.28f);
    std::vector<float> d_out_f = make_signal(n_qvv, 0.149f, 0.66f, 0.101f, -0.14f);
    std::vector<float> d_final_state_f = make_signal(n_state, 0.089f, -0.33f, 0.055f, 0.47f);

    thrust::device_vector<nv_bfloat16> d_q = to_device(to_bf16(q_f));
    thrust::device_vector<nv_bfloat16> d_k = to_device(to_bf16(k_f));
    thrust::device_vector<nv_bfloat16> d_v = to_device(to_bf16(v_f));
    thrust::device_vector<nv_bfloat16> d_g = to_device(to_bf16(g_f));
    thrust::device_vector<nv_bfloat16> d_beta = to_device(to_bf16(beta_f));
    thrust::device_vector<float> d_initial_state = to_device(initial_state_f);

    thrust::device_vector<nv_bfloat16> d_out(n_qvv);
    thrust::device_vector<float> d_final_state(n_state);

    Tensor q_t = tensor_view(d_q, ETensorDType::BF16, {B, T, H, K});
    Tensor k_t = tensor_view(d_k, ETensorDType::BF16, {B, T, H, K});
    Tensor v_t = tensor_view(d_v, ETensorDType::BF16, {B, T, H, V});
    Tensor g_t = tensor_view(d_g, ETensorDType::BF16, {B, T, H});
    Tensor beta_t = tensor_view(d_beta, ETensorDType::BF16, {B, T, H});
    Tensor init_t = tensor_view(d_initial_state, ETensorDType::FP32, {B, H, K, V});
    Tensor out_t = tensor_view(d_out, ETensorDType::BF16, {B, T, H, V});
    Tensor final_state_t = tensor_view(d_final_state, ETensorDType::FP32, {B, H, K, V});

    thrust::device_vector<float> d_forward_scratch(n_state);
    Tensor forward_scratch_t = tensor_view(d_forward_scratch, ETensorDType::FP32, {B, H, K, V});

    const int num_chunks = (T + chunk_size - 1) / chunk_size;
    const std::size_t n_checkpoints =
        static_cast<std::size_t>(B) * H * static_cast<std::size_t>(num_chunks + 1) * K * V;
    thrust::device_vector<float> d_checkpoints(n_checkpoints);
    Tensor checkpoints_fwd_t =
        tensor_view(d_checkpoints, ETensorDType::FP32, {B, H, num_chunks + 1, K, V});

    gated_delta_rule_chunk_forward_v2(
        out_t,
        final_state_t,
        forward_scratch_t,
        q_t,
        k_t,
        v_t,
        g_t,
        beta_t,
        &init_t,
        scale,
        chunk_size,
        use_qk_l2norm_in_kernel,
        &checkpoints_fwd_t,
        /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::device_vector<nv_bfloat16> d_dq(n_qkvk);
    thrust::device_vector<nv_bfloat16> d_dk(n_qkvk);
    thrust::device_vector<nv_bfloat16> d_dv(n_qvv);
    thrust::device_vector<nv_bfloat16> d_dg(n_gh);
    thrust::device_vector<nv_bfloat16> d_dbeta(n_gh);
    thrust::device_vector<float> d_dinitial(n_state);

    thrust::device_vector<nv_bfloat16> d_dout = to_device(to_bf16(d_out_f));
    thrust::device_vector<float> d_dfinal = to_device(d_final_state_f);
    const int Lp = 64;
    constexpr std::size_t kWsAlignFloats = 128 / sizeof(float);
    const std::size_t c_storage_floats = static_cast<std::size_t>(K) * K;
    auto align_up = [](std::size_t x, std::size_t align) {
        return ((x + align - 1) / align) * align;
    };
    std::size_t chunk_ws_stride = 0;
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(Lp) * Lp; // M
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(Lp) * Lp; // A
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(Lp) * K;  // W
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(Lp) * V;  // VNEW
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(Lp) * V;  // DU
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(Lp) * K;  // DW
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(Lp) * K;  // DQ
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(Lp) * K;  // DK
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(Lp);      // DG
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(Lp);      // DB
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<std::size_t>(K) * V;   // DHT1
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + c_storage_floats;                  // C fp32
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + 1;                                  // EG
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats);                                       // total stride
    const std::size_t dh_storage_per_chunk = static_cast<std::size_t>(K) * V;
    const std::size_t workspace_size =
        static_cast<std::size_t>(num_chunks) * chunk_ws_stride
        + static_cast<std::size_t>(num_chunks) * dh_storage_per_chunk;
    thrust::device_vector<float> d_state_scratch(static_cast<std::size_t>(B) * H * workspace_size);

    Tensor dq_t = tensor_view(d_dq, ETensorDType::BF16, {B, T, H, K});
    Tensor dk_t = tensor_view(d_dk, ETensorDType::BF16, {B, T, H, K});
    Tensor dv_t = tensor_view(d_dv, ETensorDType::BF16, {B, T, H, V});
    Tensor dg_t = tensor_view(d_dg, ETensorDType::BF16, {B, T, H});
    Tensor dbeta_t = tensor_view(d_dbeta, ETensorDType::BF16, {B, T, H});
    Tensor dinit_t = tensor_view(d_dinitial, ETensorDType::FP32, {B, H, K, V});
    Tensor dout_t = tensor_view(d_dout, ETensorDType::BF16, {B, T, H, V});
    Tensor dfinal_t = tensor_view(d_dfinal, ETensorDType::FP32, {B, H, K, V});
    Tensor checkpoints_t =
        tensor_view(d_checkpoints, ETensorDType::FP32, {B, H, num_chunks + 1, K, V});
    Tensor state_scratch_t =
        tensor_view(d_state_scratch, ETensorDType::FP32, {B, H, static_cast<long>(workspace_size)});

    gated_delta_rule_chunk_backward_v2(
        dq_t,
        dk_t,
        dv_t,
        dg_t,
        dbeta_t,
        dinit_t,
        dout_t,
        &dfinal_t,
        q_t,
        k_t,
        v_t,
        g_t,
        beta_t,
        &init_t,
        scale,
        chunk_size,
        use_qk_l2norm_in_kernel,
        checkpoints_t,
        state_scratch_t,
        /*skip_checkpoint=*/true,
        /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> out_h = bf16_to_float(from_device(d_out));
    const std::vector<float> final_state_h = from_device(d_final_state);
    const std::vector<float> dq_h = bf16_to_float(from_device(d_dq));
    const std::vector<float> dk_h = bf16_to_float(from_device(d_dk));
    const std::vector<float> dv_h = bf16_to_float(from_device(d_dv));
    const std::vector<float> dg_h = bf16_to_float(from_device(d_dg));
    const std::vector<float> dbeta_h = bf16_to_float(from_device(d_dbeta));
    const std::vector<float> dinit_h = from_device(d_dinitial);

    json report;
    report["meta"] = {
        {"case_name", cfg.case_name},
        {"B", B},
        {"T", T},
        {"H", H},
        {"K", K},
        {"V", V},
        {"dtype", "bfloat16"},
        {"chunk_size", chunk_size},
        {"scale", scale},
        {"use_qk_l2norm_in_kernel", use_qk_l2norm_in_kernel},
    };
    report["out"] = stats(out_h);
    report["final_state"] = stats(final_state_h);
    report["d_query"] = stats(dq_h);
    report["d_key"] = stats(dk_h);
    report["d_value"] = stats(dv_h);
    report["d_g"] = stats(dg_h);
    report["d_beta"] = stats(dbeta_h);
    report["d_initial_state"] = stats(dinit_h);
    return report;
}

}  // namespace

TEST_CASE("gated delta rule deterministic surogate value dump", "[kernels][gated_delta_rule][dump]") {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        SUCCEED("CUDA not available; skipping gated delta rule value dump");
        return;
    }
    CUDA_CHECK(cudaSetDevice(0));

    const std::vector<DumpConfig> configs{
        {"single_chunk_small", 1, 8, 2, 4, 3},
        {"multi_chunk_tail", 1, 130, 2, 64, 64},
    };

    json report;
    report["meta"] = {
        {"impl", "surogate_cuda"},
        {"num_cases", configs.size()},
    };
    report["cases"] = json::array();
    for (const auto& cfg : configs) {
        report["cases"].push_back(dump_case(cfg));
    }

    const fs::path repo_root = find_repo_root();
    const fs::path out_dir = repo_root / "tests" / "artifacts";
    fs::create_directories(out_dir);
    const fs::path out_path = out_dir / "gated_delta_rule_surogate_values.json";
    std::ofstream ofs(out_path);
    REQUIRE(ofs.is_open());
    ofs << report.dump(2) << "\n";
    ofs.close();

    INFO("Wrote surogate gated delta values to: " << out_path.string());
    REQUIRE(fs::exists(out_path));
}

TEST_CASE("gated delta rule surogate perf benchmark", "[kernels][gated_delta_rule][benchmark][!benchmark]") {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        SUCCEED("CUDA not available; skipping gated delta rule benchmark");
        return;
    }
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    constexpr int warmup_iters = 1;
    constexpr int bench_iters = 3;

    const std::vector<BenchConfig> configs{
        {1, 256, 8, 64, 64},
        {1, 512, 8, 64, 64},
    };

    json report;
    report["meta"] = {
        {"device", std::string(props.name)},
        {"warmup_iters", warmup_iters},
        {"bench_iters", bench_iters},
    };
    report["cases"] = json::array();

    for (const auto& cfg : configs) {
        std::cout << "[surogate gdr benchmark] running B=" << cfg.B
                  << " T=" << cfg.T << " H=" << cfg.H
                  << " K=" << cfg.K << " V=" << cfg.V << "...\n";
        report["cases"].push_back(benchmark_case(cfg, warmup_iters, bench_iters));
    }

    const fs::path repo_root = find_repo_root();
    const fs::path out_dir = repo_root / "tests" / "artifacts";
    fs::create_directories(out_dir);
    const fs::path out_path = out_dir / "gated_delta_rule_surogate_perf.json";

    std::ofstream ofs(out_path);
    REQUIRE(ofs.is_open());
    ofs << report.dump(2) << "\n";
    ofs.close();

    std::cout << "\n[surogate gdr benchmark] wrote " << out_path << "\n";
    for (const auto& c : report["cases"]) {
        std::cout << "  B=" << c["B"] << " T=" << c["T"] << " H=" << c["H"]
                  << " K=" << c["K"] << " V=" << c["V"]
                  << " | fwd=" << c["forward_ms"] << " ms"
                  << " bwd=" << c["backward_ms"] << " ms"
                  << " total=" << c["total_ms"] << " ms\n";
    }
    REQUIRE(fs::exists(out_path));
}
