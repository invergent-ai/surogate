// cuBLASLt block-scaled FP4 against the in-house W4A4 GEMM on identical quantized inputs:
// the stored weight codes and 128x4-tiled scales are consumed in place; only the activation
// scales are re-tiled from the quantizer's token-major layout.
#include "core/engine_context.h"
#include "core/tensor.h"
#include "ops/linear/nvfp4/nvfp4_cublaslt.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/quantized_weight.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

using namespace sinfer;

namespace {

bool cuda_ok(cudaError_t status, const char* what) {
    if (status == cudaSuccess) { return true; }
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    return false;
}

bool lt_ok(cublasStatus_t status, const char* what) {
    if (status == CUBLAS_STATUS_SUCCESS) { return true; }
    std::fprintf(stderr, "%s: cublasLt status %d\n", what, static_cast<int>(status));
    return false;
}

std::int64_t tiled_scale_offset(int row, int group, int tiles_per_row) {
    const int m_tile    = row / 128;
    const int row_inner = row - m_tile * 128;
    return (static_cast<std::int64_t>(m_tile) * tiles_per_row + group / 4) * 512 +
           (row_inner & 31) * 16 + (row_inner >> 5) * 4 + (group & 3);
}

float bf16_to_float(std::uint16_t bits) {
    std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t word;
    std::memcpy(&word, &value, sizeof(word));
    word += 0x7FFFu + ((word >> 16) & 1u);
    return static_cast<std::uint16_t>(word >> 16);
}

int run_case(int n, int k, int tokens) {
    std::printf("NVFP4 cuBLASLt parity [%d,%d] T=%d\n", n, k, tokens);
    test::quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    const test::quantized_weight::PackedWeight host_weight =
        test::quantized_weight::make_patterned_weight(QType::NVFP4, n, k, 911U, options);
    void* device_payload = nullptr;
    if (!cuda_ok(cudaMalloc(&device_payload, host_weight.payload.size()), "malloc weight")) return 1;
    cudaMemcpy(device_payload, host_weight.payload.data(), host_weight.payload.size(), cudaMemcpyHostToDevice);
    const Weight weight = host_weight.device_weight(device_payload);

    std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * tokens);
    std::uint32_t state = 0x9E3779B9u;
    for (auto& value : activation) {
        state = state * 1664525u + 1013904223u;
        activation[&value - activation.data()] = float_to_bf16((static_cast<float>(state >> 8) / 16777216.0F) * 2.0F - 1.0F);
    }
    void* device_x = nullptr;
    cudaMalloc(&device_x, activation.size() * sizeof(std::uint16_t));
    cudaMemcpy(device_x, activation.data(), activation.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    Tensor x(device_x, DType::BF16, {k, tokens});

    std::uint8_t* codes  = nullptr;
    std::uint8_t* scales = nullptr;
    const std::size_t code_bytes  = static_cast<std::size_t>(tokens) * (k / 2);
    const std::size_t scale_bytes = static_cast<std::size_t>(tokens) * (k / 16);
    cudaMalloc(reinterpret_cast<void**>(&codes), code_bytes);
    cudaMalloc(reinterpret_cast<void**>(&scales), scale_bytes);
    const ops::detail::Nvfp4W4a4Workspace workspace{codes, scales};

    const std::size_t out_elements = static_cast<std::size_t>(n) * tokens;
    void* device_ref = nullptr;
    void* device_lt  = nullptr;
    cudaMalloc(&device_ref, out_elements * 2);
    cudaMalloc(&device_lt, out_elements * 2);
    Tensor out_ref(device_ref, DType::BF16, {n, tokens});
    ops::detail::launch_nvfp4_w4a4(x, weight, out_ref, workspace, nullptr);
    if (!cuda_ok(cudaDeviceSynchronize(), "in-house W4A4")) return 1;

    std::vector<std::uint8_t> host_scales(scale_bytes);
    cudaMemcpy(host_scales.data(), scales, scale_bytes, cudaMemcpyDeviceToHost);
    const int rows_padded    = (tokens + 127) / 128 * 128;
    const int groups_per_row = k / 16;
    const int tiles_per_row  = k / 64;
    std::vector<std::uint8_t> tiled(static_cast<std::size_t>(rows_padded) * groups_per_row, 0);
    for (int t = 0; t < tokens; ++t) {
        for (int g = 0; g < groups_per_row; ++g) {
            tiled[tiled_scale_offset(t, g, tiles_per_row)] = host_scales[static_cast<std::size_t>(t) * groups_per_row + g];
        }
    }
    std::uint8_t* device_tiled = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&device_tiled), tiled.size());
    cudaMemcpy(device_tiled, tiled.data(), tiled.size(), cudaMemcpyHostToDevice);

    cublasLtHandle_t handle = nullptr;
    if (!lt_ok(cublasLtCreate(&handle), "create")) return 1;
    cublasLtMatmulDesc_t op = nullptr;
    if (!lt_ok(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc")) return 1;
    const cublasOperation_t trans_a = CUBLAS_OP_T;
    const cublasOperation_t trans_b = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));
    const cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode));
    const void* a_scale = weight.scales;
    const void* b_scale = device_tiled;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale));
    cublasLtMatrixLayout_t layout_a = nullptr, layout_b = nullptr, layout_d = nullptr;
    cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_4F_E2M1, k, n, k);
    cublasLtMatrixLayoutCreate(&layout_b, CUDA_R_4F_E2M1, k, tokens, k);
    cublasLtMatrixLayoutCreate(&layout_d, CUDA_R_16BF, n, tokens, n);
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulPreferenceCreate(&preference);
    const std::size_t workspace_bytes = std::size_t{64} << 20;
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes));
    cublasLtMatmulHeuristicResult_t heuristics[4];
    int found = 0;
    if (!lt_ok(cublasLtMatmulAlgoGetHeuristic(handle, op, layout_a, layout_b, layout_d, layout_d, preference, 4, heuristics, &found), "heuristic") || found == 0) {
        std::fprintf(stderr, "no cuBLASLt algorithm\n");
        return 1;
    }
    void* lt_workspace = nullptr;
    cudaMalloc(&lt_workspace, workspace_bytes);
    const float alpha = 1.0F / (options.input_scale_divisor * options.weight_scale_divisor);
    const float beta  = 0.0F;
    if (!lt_ok(cublasLtMatmul(handle, op, &alpha, weight.qdata, layout_a, codes, layout_b, &beta, device_lt, layout_d, device_lt, layout_d, &heuristics[0].algo, lt_workspace, workspace_bytes, nullptr), "matmul")) return 1;
    if (!cuda_ok(cudaDeviceSynchronize(), "cuBLASLt")) return 1;

    std::vector<std::uint16_t> host_ref(out_elements), host_lt(out_elements);
    cudaMemcpy(host_ref.data(), device_ref, out_elements * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lt.data(), device_lt, out_elements * 2, cudaMemcpyDeviceToHost);
    double max_abs = 0.0, max_ref = 0.0;
    std::size_t loose = 0;
    for (std::size_t i = 0; i < out_elements; ++i) {
        const double r = bf16_to_float(host_ref[i]);
        const double l = bf16_to_float(host_lt[i]);
        const double d = std::fabs(r - l);
        if (d > max_abs) max_abs = d;
        if (std::fabs(r) > max_ref) max_ref = std::fabs(r);
        if (d > 0.02 * (std::fabs(r) + 1e-3)) ++loose;
    }
    const double rel = max_ref > 0.0 ? max_abs / max_ref : 0.0;

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    float ms_ref = 0.0F, ms_lt = 0.0F;
    for (int i = 0; i < 3; ++i) ops::detail::launch_nvfp4_w4a4(x, weight, out_ref, workspace, nullptr);
    cudaEventRecord(begin);
    for (int i = 0; i < 20; ++i) ops::detail::launch_nvfp4_w4a4(x, weight, out_ref, workspace, nullptr);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms_ref, begin, end);
    for (int i = 0; i < 3; ++i) cublasLtMatmul(handle, op, &alpha, weight.qdata, layout_a, codes, layout_b, &beta, device_lt, layout_d, device_lt, layout_d, &heuristics[0].algo, lt_workspace, workspace_bytes, nullptr);
    cudaEventRecord(begin);
    for (int i = 0; i < 20; ++i) cublasLtMatmul(handle, op, &alpha, weight.qdata, layout_a, codes, layout_b, &beta, device_lt, layout_d, device_lt, layout_d, &heuristics[0].algo, lt_workspace, workspace_bytes, nullptr);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms_lt, begin, end);
    const double flops = 2.0 * n * k * tokens;
    std::printf("  max|ref|=%.3f max_abs_diff=%.4f rel=%.2e loose=%zu/%zu | in-house(quant+gemm) %.3f ms (%.0f TFLOP/s) cuBLASLt %.3f ms (%.0f TFLOP/s)\n",
                max_ref, max_abs, rel, loose, out_elements, ms_ref / 20, flops / (ms_ref / 20) / 1e9, ms_lt / 20, flops / (ms_lt / 20) / 1e9);
    return rel > 1e-2 ? 1 : 0;
}

// Two engines in one process each drive the plane from their own thread, stream and
// EngineOpsContext, on the same inputs, with their launches queued deep enough that the
// device runs them side by side. Before the plane was homed per engine both engines shared
// one cuBLASLt workspace; a stream-K algorithm keeps its cross-CTA barrier flags there, and
// the second engine's kernel overwrote flags the first was spinning on -- the multi-model
// wedge. A wedge here never returns, so a watchdog fails the test instead of hanging CI.
int run_two_engine_case(int n, int k, int tokens, int iterations) {
    std::printf("NVFP4 cuBLASLt two engines [%d,%d] T=%d x%d\n", n, k, tokens, iterations);
    test::quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    const test::quantized_weight::PackedWeight host_weight =
        test::quantized_weight::make_patterned_weight(QType::NVFP4, n, k, 4243U, options);
    void* device_payload = nullptr;
    if (!cuda_ok(cudaMalloc(&device_payload, host_weight.payload.size()), "malloc weight")) return 1;
    cudaMemcpy(device_payload, host_weight.payload.data(), host_weight.payload.size(), cudaMemcpyHostToDevice);
    const Weight weight = host_weight.device_weight(device_payload);

    std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * tokens);
    std::uint32_t state = 0x2545F491u;
    for (auto& value : activation) {
        state = state * 1664525u + 1013904223u;
        value = float_to_bf16((static_cast<float>(state >> 8) / 16777216.0F) * 2.0F - 1.0F);
    }
    void* device_x = nullptr;
    cudaMalloc(&device_x, activation.size() * sizeof(std::uint16_t));
    cudaMemcpy(device_x, activation.data(), activation.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    Tensor x(device_x, DType::BF16, {k, tokens});

    // The in-house route quantizes the activation into the workspace on the way; only its
    // token-major scales need re-tiling into the layout the cuBLASLt route consumes.
    std::uint8_t* codes  = nullptr;
    std::uint8_t* scales = nullptr;
    const std::size_t code_bytes  = static_cast<std::size_t>(tokens) * (k / 2);
    const std::size_t scale_bytes = static_cast<std::size_t>(tokens) * (k / 16);
    cudaMalloc(reinterpret_cast<void**>(&codes), code_bytes);
    cudaMalloc(reinterpret_cast<void**>(&scales), scale_bytes);
    const ops::detail::Nvfp4W4a4Workspace workspace{codes, scales};
    const std::size_t out_elements = static_cast<std::size_t>(n) * tokens;
    void* device_scratch           = nullptr;
    cudaMalloc(&device_scratch, out_elements * 2);
    Tensor scratch(device_scratch, DType::BF16, {n, tokens});
    ops::detail::launch_nvfp4_w4a4(x, weight, scratch, workspace, nullptr);
    if (!cuda_ok(cudaDeviceSynchronize(), "in-house W4A4")) return 1;
    std::vector<std::uint8_t> host_scales(scale_bytes);
    cudaMemcpy(host_scales.data(), scales, scale_bytes, cudaMemcpyDeviceToHost);
    const int rows_padded    = (tokens + 127) / 128 * 128;
    const int groups_per_row = k / 16;
    const int tiles_per_row  = k / 64;
    std::vector<std::uint8_t> tiled(static_cast<std::size_t>(rows_padded) * groups_per_row, 0);
    for (int t = 0; t < tokens; ++t) {
        for (int g = 0; g < groups_per_row; ++g) {
            tiled[tiled_scale_offset(t, g, tiles_per_row)] = host_scales[static_cast<std::size_t>(t) * groups_per_row + g];
        }
    }
    std::uint8_t* device_tiled = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&device_tiled), tiled.size());
    cudaMemcpy(device_tiled, tiled.data(), tiled.size(), cudaMemcpyHostToDevice);

    // Reference from the plane itself, alone, on the process-default context.
    void* device_ref = nullptr;
    cudaMalloc(&device_ref, out_elements * 2);
    ops::detail::nvfp4_cublaslt_gemm(weight, 0, n, codes, device_tiled,
                                     static_cast<__nv_bfloat16*>(device_ref), n, tokens, 0.0F, nullptr);
    if (!cuda_ok(cudaDeviceSynchronize(), "reference cuBLASLt")) return 1;
    std::vector<std::uint16_t> host_ref(out_elements);
    cudaMemcpy(host_ref.data(), device_ref, out_elements * 2, cudaMemcpyDeviceToHost);

    constexpr int kEngines = 2;
    ops::EngineOpsContext contexts[kEngines];
    void* outputs[kEngines] = {nullptr, nullptr};
    for (void*& out : outputs) { cudaMalloc(&out, out_elements * 2); }
    std::mutex mutex;
    std::condition_variable done_cv;
    int done_count = 0;
    std::atomic<int> thread_failures{0};
    std::vector<std::thread> engines;
    for (int e = 0; e < kEngines; ++e) {
        engines.emplace_back([&, e] {
            ops::bind_ops_context(&contexts[e]);
            ops::detail::nvfp4_cublaslt_prewarm(); // the engine allocates before it races
            cudaStream_t stream = nullptr;
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            for (int i = 0; i < iterations; ++i) {
                ops::detail::nvfp4_cublaslt_gemm(weight, 0, n, codes, device_tiled,
                                                 static_cast<__nv_bfloat16*>(outputs[e]), n, tokens,
                                                 0.0F, stream);
            }
            if (cudaStreamSynchronize(stream) != cudaSuccess) { thread_failures.fetch_add(1); }
            cudaStreamDestroy(stream);
            ops::bind_ops_context(nullptr);
            const std::lock_guard<std::mutex> lock(mutex);
            ++done_count;
            done_cv.notify_all();
        });
    }
    {
        std::unique_lock<std::mutex> lock(mutex);
        if (!done_cv.wait_for(lock, std::chrono::seconds(180), [&] { return done_count == kEngines; })) {
            std::fprintf(stderr, "FAIL: two-engine cuBLASLt run did not finish in 180 s -- the device is wedged\n");
            std::fflush(stderr);
            std::_Exit(2); // the threads are parked in cudaStreamSynchronize and cannot be joined
        }
    }
    for (std::thread& engine : engines) { engine.join(); }
    if (thread_failures.load() != 0) {
        std::fprintf(stderr, "cuBLASLt stream failed under two engines\n");
        return 1;
    }

    int failures = 0;
    std::vector<std::uint16_t> host_out(out_elements);
    for (int e = 0; e < kEngines; ++e) {
        cudaMemcpy(host_out.data(), outputs[e], out_elements * 2, cudaMemcpyDeviceToHost);
        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < out_elements; ++i) {
            if (host_out[i] != host_ref[i]) { ++mismatches; }
        }
        std::printf("  engine %d: %zu/%zu elements differ from the solo run\n", e, mismatches, out_elements);
        if (mismatches != 0) { ++failures; }
    }
    return failures;
}

} // namespace

int main() {
    // The reference must stay on the in-house kernels whatever the route default is; the
    // route reads its switch once, so pin it before the first GEMM.
    setenv("SUROGATE_SERVE_NVFP4_CUBLASLT", "0", 1);
    int failures = 0;
    failures += run_case(16384, 5120, 1088);
    failures += run_case(5120, 6144, 300);
    failures += run_case(34816, 5120, 1024);
    // Few output tiles for many SMs: the shape cuBLASLt answers with a stream-K algorithm.
    failures += run_two_engine_case(1024, 5120, 1024, 200);
    std::printf(failures ? "FAIL NVFP4 cuBLASLt parity\n" : "OK NVFP4 cuBLASLt parity\n");
    return failures;
}
