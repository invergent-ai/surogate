// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Based on llm.c https://github.com/karpathy/llm.c

#include <cublasLt.h>
#include <fmt/core.h>

#include "kernels.h"
#include "kernels/matmul_plans.h"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
inline void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(fmt::format("cuBLAS ERROR ({}) at {}:{}", (int)status, file, line));
    }
}
#define CUBLAS_CHECK(status) { cublasCheck((status), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// Matmul plan cache (Proposal A)

namespace {

struct MatmulAutotuneOptions {
    bool enabled = false;
    int topk = 16;
    int iters = 15;
    int warmup = 3;
    bool verbose = false;
};

static bool parse_bool_env(std::string_view value, bool default_value) {
    if (value.empty()) {
        return default_value;
    }
    auto ieq = [](char a, char b) { return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b)); };
    auto equals_ci = [&](std::string_view target) {
        if (value.size() != target.size()) return false;
        for (size_t i = 0; i < value.size(); ++i) {
            if (!ieq(value[i], target[i])) return false;
        }
        return true;
    };
    if (equals_ci("1") || equals_ci("true") || equals_ci("yes") || equals_ci("on")) return true;
    if (equals_ci("0") || equals_ci("false") || equals_ci("no") || equals_ci("off")) return false;
    return default_value;
}

static int parse_int_env(std::string_view value, int default_value) {
    if (value.empty()) {
        return default_value;
    }
    const std::string tmp(value);
    char* end = nullptr;
    long parsed = std::strtol(tmp.c_str(), &end, 10);
    if (end == nullptr || *end != '\0') {
        return default_value;
    }
    if (parsed < 0) {
        return default_value;
    }
    if (parsed > std::numeric_limits<int>::max()) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

static MatmulAutotuneOptions read_matmul_autotune_env() {
    MatmulAutotuneOptions opts;
    if (const char* v = std::getenv("SUROGATE_MATMUL_AUTOTUNE")) {
        opts.enabled = parse_bool_env(v, false);
    }
    if (const char* v = std::getenv("SUROGATE_MATMUL_AUTOTUNE_TOPK")) {
        opts.topk = parse_int_env(v, opts.topk);
    }
    if (const char* v = std::getenv("SUROGATE_MATMUL_AUTOTUNE_ITERS")) {
        opts.iters = parse_int_env(v, opts.iters);
    }
    if (const char* v = std::getenv("SUROGATE_MATMUL_AUTOTUNE_WARMUP")) {
        opts.warmup = parse_int_env(v, opts.warmup);
    }
    if (const char* v = std::getenv("SUROGATE_MATMUL_AUTOTUNE_VERBOSE")) {
        opts.verbose = parse_bool_env(v, false);
    }

    opts.topk = std::clamp(opts.topk, 1, 64);
    opts.iters = std::clamp(opts.iters, 1, 100);
    opts.warmup = std::clamp(opts.warmup, 0, 50);
    return opts;
}

static const char* dtype_name(cudaDataType dt) {
    switch (dt) {
        case CUDA_R_32F: return "f32";
        case CUDA_R_16BF: return "bf16";
        case CUDA_R_8F_E4M3: return "fp8e4m3";
        case CUDA_R_8F_E5M2: return "fp8e5m2";
        case CUDA_R_8I: return "i8";
        default: return "unknown";
    }
}

static int32_t algo_config_id(const cublasLtMatmulAlgo_t& algo) {
    int32_t id = -1;
    size_t written = 0;
    if (cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof(id), &written) != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }
    return id;
}

static void set_scale_pointer_attr(cublasLtMatmulDesc_t desc, cublasLtMatmulDescAttributes_t attr, const float* scale_ptr) {
    // cublasLt expects the pointer value to be passed by address.
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, attr, &scale_ptr, sizeof(scale_ptr)));
}

static void set_bias_pointer_attr(cublasLtMatmulDesc_t desc, const void* bias_ptr) {
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
}

struct MatmulPlanKey {
    int m = 0;
    int n = 0;
    int k = 0;
    int ldc = 0;
    int dtype_d = 0;
    int dtype_a = 0;
    int dtype_b = 0;
    int dtype_bias = 0;
    bool transA = false;
    bool transB = false;
    bool has_bias = false;
    bool has_scale_a = false;
    bool has_scale_b = false;
    std::size_t workspace_size = 0;

    bool operator==(const MatmulPlanKey& other) const = default;
};

struct MatmulPlanKeyHash {
    std::size_t operator()(const MatmulPlanKey& key) const noexcept {
        auto h = std::size_t{0};
        auto combine = [&](std::size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        combine(std::hash<int>{}(key.m));
        combine(std::hash<int>{}(key.n));
        combine(std::hash<int>{}(key.k));
        combine(std::hash<int>{}(key.ldc));
        combine(std::hash<int>{}(key.dtype_d));
        combine(std::hash<int>{}(key.dtype_a));
        combine(std::hash<int>{}(key.dtype_b));
        combine(std::hash<int>{}(key.dtype_bias));
        combine(std::hash<bool>{}(key.transA));
        combine(std::hash<bool>{}(key.transB));
        combine(std::hash<bool>{}(key.has_bias));
        combine(std::hash<bool>{}(key.has_scale_a));
        combine(std::hash<bool>{}(key.has_scale_b));
        combine(std::hash<std::size_t>{}(key.workspace_size));
        return h;
    }
};

struct MatmulPlan {
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr;
    cublasLtMatrixLayout_t b_layout = nullptr;
    cublasLtMatrixLayout_t c_layout = nullptr;
    cublasLtMatrixLayout_t d_layout = nullptr;
    cublasLtMatmulAlgo_t algo{};
    std::size_t algo_workspace_size = 0;
    float tuned_ms = -1.0f;
    bool has_algo = false;
    bool tuned = false;

    MatmulPlan() = default;
    MatmulPlan(const MatmulPlan&) = delete;
    MatmulPlan& operator=(const MatmulPlan&) = delete;

    MatmulPlan(MatmulPlan&& other) noexcept {
        *this = std::move(other);
    }
    MatmulPlan& operator=(MatmulPlan&& other) noexcept {
        if (this == &other) return *this;
        this->~MatmulPlan();
        op_desc = other.op_desc;
        a_layout = other.a_layout;
        b_layout = other.b_layout;
        c_layout = other.c_layout;
        d_layout = other.d_layout;
        algo = other.algo;
        algo_workspace_size = other.algo_workspace_size;
        tuned_ms = other.tuned_ms;
        has_algo = other.has_algo;
        tuned = other.tuned;
        other.op_desc = nullptr;
        other.a_layout = nullptr;
        other.b_layout = nullptr;
        other.c_layout = nullptr;
        other.d_layout = nullptr;
        other.has_algo = false;
        other.tuned = false;
        other.tuned_ms = -1.0f;
        other.algo_workspace_size = 0;
        return *this;
    }

    ~MatmulPlan() noexcept {
        if (op_desc) {
            (void)cublasLtMatmulDescDestroy(op_desc);
            op_desc = nullptr;
        }
        if (a_layout) {
            (void)cublasLtMatrixLayoutDestroy(a_layout);
            a_layout = nullptr;
        }
        if (b_layout) {
            (void)cublasLtMatrixLayoutDestroy(b_layout);
            b_layout = nullptr;
        }
        if (c_layout) {
            (void)cublasLtMatrixLayoutDestroy(c_layout);
            c_layout = nullptr;
        }
        if (d_layout) {
            (void)cublasLtMatrixLayoutDestroy(d_layout);
            d_layout = nullptr;
        }
    }
};

static void build_plan_descriptors(MatmulPlan& plan, const MatmulPlanKey& key) {
    const bool transA = key.transA;
    const bool transB = key.transB;
    const int m = key.m;
    const int n = key.n;
    const int k = key.k;

    const auto dtype_a = static_cast<cudaDataType>(key.dtype_a);
    const auto dtype_b = static_cast<cudaDataType>(key.dtype_b);
    const auto dtype_d = static_cast<cudaDataType>(key.dtype_d);

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&plan.op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSA, (transA) ? &opTranspose : &opNoTranspose, sizeof(opTranspose)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSB, (transB) ? &opTranspose : &opNoTranspose, sizeof(opNoTranspose)));

    cublasLtEpilogue_t epilogue = key.has_bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (key.has_bias) {
        const auto bias_dtype = static_cast<cudaDataType>(key.dtype_bias);
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_dtype, sizeof(bias_dtype)));
        // Initialize to nullptr; updated per call.
        set_bias_pointer_attr(plan.op_desc, nullptr);
    }

    // Initialize scale pointers (updated per call). Note: for non-FP8 paths these remain nullptr.
    if (key.has_scale_a) {
        set_scale_pointer_attr(plan.op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, nullptr);
    }
    if (key.has_scale_b) {
        set_scale_pointer_attr(plan.op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, nullptr);
    }

    // Matrix layouts
    if (transA) {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.a_layout, dtype_a, k, m, k));
    } else {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.a_layout, dtype_a, m, k, m));
    }
    if (transB) {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.b_layout, dtype_b, n, k, n));
    } else {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.b_layout, dtype_b, k, n, k));
    }
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.c_layout, dtype_d, m, n, key.ldc));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.d_layout, dtype_d, m, n, key.ldc));
}

static cublasLtMatmulHeuristicResult_t choose_top1_heuristic(cublasLtHandle_t handle,
                                                            cublasLtMatmulDesc_t op_desc,
                                                            cublasLtMatrixLayout_t a_layout,
                                                            cublasLtMatrixLayout_t b_layout,
                                                            cublasLtMatrixLayout_t c_layout,
                                                            cublasLtMatrixLayout_t d_layout,
                                                            std::size_t workspace_size) {
    cublasLtMatmulPreference_t preference = nullptr;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &workspace_size, sizeof(workspace_size)));

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returnedResults = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, op_desc, a_layout, b_layout, c_layout, d_layout,
                                               preference, 1, &heuristic, &returnedResults));
    (void)cublasLtMatmulPreferenceDestroy(preference);
    if (returnedResults == 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("No cuBLASLt heuristic algorithm found");
    }
    return heuristic;
}

static float benchmark_algo(cublasLtHandle_t handle,
                            cublasLtMatmulDesc_t op_desc,
                            cublasLtMatrixLayout_t a_layout,
                            cublasLtMatrixLayout_t b_layout,
                            cublasLtMatrixLayout_t c_layout,
                            cublasLtMatrixLayout_t d_layout,
                            const cublasLtMatmulAlgo_t& algo,
                            void* d, const void* a, const void* b,
                            void* workspace, std::size_t workspace_size,
                            int warmup, int iters,
                            cudaStream_t stream) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float alpha = 1.f;
    float beta = 0.f;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        auto st = cublasLtMatmul(handle, op_desc,
                                 &alpha, a, a_layout, b, b_layout, &beta, d, c_layout, d, d_layout,
                                 &algo, workspace, workspace_size, stream);
        if (st != CUBLAS_STATUS_SUCCESS) {
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            return std::numeric_limits<float>::infinity();
        }
    }

    std::vector<float> times;
    times.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        auto st = cublasLtMatmul(handle, op_desc,
                                 &alpha, a, a_layout, b, b_layout, &beta, d, c_layout, d, d_layout,
                                 &algo, workspace, workspace_size, stream);
        if (st != CUBLAS_STATUS_SUCCESS) {
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            return std::numeric_limits<float>::infinity();
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    if (times.empty()) {
        return std::numeric_limits<float>::infinity();
    }
    auto mid = times.begin() + (times.size() / 2);
    std::nth_element(times.begin(), mid, times.end());
    return *mid;
}

static bool stream_is_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) {
        return false;
    }
    return status != cudaStreamCaptureStatusNone;
}

} // namespace

struct MatmulPlanCache::Impl {
    int device_id = -1;
    MatmulAutotuneOptions opts;
    std::unordered_map<MatmulPlanKey, std::unique_ptr<MatmulPlan>, MatmulPlanKeyHash> plans;

    explicit Impl(int did) : device_id(did), opts(read_matmul_autotune_env()) {}

    MatmulPlan& get_or_create_plan(const MatmulPlanKey& key) {
        auto it = plans.find(key);
        if (it != plans.end()) {
            return *it->second;
        }
        auto plan = std::make_unique<MatmulPlan>();
        build_plan_descriptors(*plan, key);
        auto [insert_it, inserted] = plans.emplace(key, std::move(plan));
        return *insert_it->second;
    }

    void ensure_algorithm(MatmulPlan& plan,
                          const MatmulPlanKey& key,
                          cublasLtHandle_t handle,
                          const void* bias_ptr,
                          const float* scale_a,
                          const float* scale_b) {
        // Always set runtime pointers before heuristic selection (FP8 requires scale pointers).
        if (key.has_bias) {
            set_bias_pointer_attr(plan.op_desc, bias_ptr);
        }
        if (key.has_scale_a) {
            set_scale_pointer_attr(plan.op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, scale_a);
        }
        if (key.has_scale_b) {
            set_scale_pointer_attr(plan.op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, scale_b);
        }

        if (plan.has_algo) {
            return;
        }

        auto heuristic = choose_top1_heuristic(handle, plan.op_desc, plan.a_layout, plan.b_layout, plan.c_layout, plan.d_layout, key.workspace_size);
        plan.algo = heuristic.algo;
        plan.algo_workspace_size = heuristic.workspaceSize;
        plan.has_algo = true;
        plan.tuned = false;
        plan.tuned_ms = -1.0f;

        if (opts.verbose) {
            const auto dtypeA = static_cast<cudaDataType>(key.dtype_a);
            const auto dtypeB = static_cast<cudaDataType>(key.dtype_b);
            const auto dtypeD = static_cast<cudaDataType>(key.dtype_d);
            fmt::print("[matmul-plan][dev{}] init m={} n={} k={} ldc={} A={} B={} D={} bias={} algo_id={} ws={}B\n",
                       device_id, key.m, key.n, key.k, key.ldc,
                       dtype_name(dtypeA), dtype_name(dtypeB), dtype_name(dtypeD),
                       key.has_bias ? "yes" : "no",
                       algo_config_id(plan.algo),
                       plan.algo_workspace_size);
        }
    }

    void maybe_autotune(MatmulPlan& plan,
                        const MatmulPlanKey& key,
                        cublasLtHandle_t handle,
                        void* d, const void* a, const void* b, const void* bias_ptr,
                        const float* scale_a, const float* scale_b,
                        std::byte* workspace, cudaStream_t stream,
                        bool accumulate) {
        if (!opts.enabled) return;
        if (plan.tuned) return;
        if (accumulate) return;  // Avoid perturbing accumulation buffers during tuning.
        if (stream_is_capturing(stream)) return;

        // Ensure runtime pointers set for this call before querying heuristics.
        if (key.has_bias) {
            set_bias_pointer_attr(plan.op_desc, bias_ptr);
        }
        if (key.has_scale_a) {
            set_scale_pointer_attr(plan.op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, scale_a);
        }
        if (key.has_scale_b) {
            set_scale_pointer_attr(plan.op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, scale_b);
        }

        const int topk = opts.topk;
        std::vector<cublasLtMatmulHeuristicResult_t> heuristics(static_cast<size_t>(topk));
        int returned = 0;

        cublasLtMatmulPreference_t preference = nullptr;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                         &key.workspace_size, sizeof(key.workspace_size)));

        CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, plan.op_desc, plan.a_layout, plan.b_layout, plan.c_layout, plan.d_layout,
                                                   preference, topk, heuristics.data(), &returned));
        (void)cublasLtMatmulPreferenceDestroy(preference);

        float best_ms = std::numeric_limits<float>::infinity();
        cublasLtMatmulAlgo_t best_algo{};
        std::size_t best_ws = 0;
        bool found = false;

        for (int i = 0; i < returned; ++i) {
            const auto& h = heuristics[static_cast<size_t>(i)];
            if (h.state != CUBLAS_STATUS_SUCCESS) continue;
            if (h.workspaceSize > key.workspace_size) continue;

            float ms = benchmark_algo(handle, plan.op_desc, plan.a_layout, plan.b_layout, plan.c_layout, plan.d_layout,
                                      h.algo, d, a, b,
                                      workspace, key.workspace_size,
                                      opts.warmup, opts.iters,
                                      stream);
            if (ms < best_ms) {
                best_ms = ms;
                best_algo = h.algo;
                best_ws = h.workspaceSize;
                found = true;
            }
        }

        if (!found || !std::isfinite(best_ms)) {
            return;
        }

        plan.algo = best_algo;
        plan.algo_workspace_size = best_ws;
        plan.has_algo = true;
        plan.tuned = true;
        plan.tuned_ms = best_ms;

        if (opts.verbose) {
            const auto dtypeA = static_cast<cudaDataType>(key.dtype_a);
            const auto dtypeB = static_cast<cudaDataType>(key.dtype_b);
            const auto dtypeD = static_cast<cudaDataType>(key.dtype_d);
            fmt::print("[matmul-plan][dev{}] tuned m={} n={} k={} ldc={} A={} B={} D={} bias={} algo_id={} med_ms={:.4f} ws={}B\n",
                       device_id, key.m, key.n, key.k, key.ldc,
                       dtype_name(dtypeA), dtype_name(dtypeB), dtype_name(dtypeD),
                       key.has_bias ? "yes" : "no",
                       algo_config_id(plan.algo),
                       plan.tuned_ms,
                       plan.algo_workspace_size);
        }
    }

    void run_matmul(MatmulPlan& plan,
                    const MatmulPlanKey& key,
                    cublasLtHandle_t handle,
                    void* d, const void* a, const void* b, const void* bias_ptr,
                    const float* scale_a, const float* scale_b,
                    std::byte* workspace, cudaStream_t stream,
                    bool accumulate) {
        // Always set runtime pointers before the actual call.
        if (key.has_bias) {
            set_bias_pointer_attr(plan.op_desc, bias_ptr);
        }
        if (key.has_scale_a) {
            set_scale_pointer_attr(plan.op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, scale_a);
        }
        if (key.has_scale_b) {
            set_scale_pointer_attr(plan.op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, scale_b);
        }

        float alpha = 1.f;
        float beta = accumulate ? 1.f : 0.f;

        auto st = cublasLtMatmul(handle, plan.op_desc,
                                 &alpha, a, plan.a_layout, b, plan.b_layout, &beta, d, plan.c_layout, d, plan.d_layout,
                                 &plan.algo, workspace, key.workspace_size, stream);
        if (st == CUBLAS_STATUS_SUCCESS) {
            CUDA_CHECK(cudaGetLastError());
            return;
        }

        // Fallback: refresh algo from top-1 heuristic and retry once.
        plan.has_algo = false;
        plan.tuned = false;
        plan.tuned_ms = -1.0f;

        ensure_algorithm(plan, key, handle, bias_ptr, scale_a, scale_b);

        st = cublasLtMatmul(handle, plan.op_desc,
                            &alpha, a, plan.a_layout, b, plan.b_layout, &beta, d, plan.c_layout, d, plan.d_layout,
                            &plan.algo, workspace, key.workspace_size, stream);
        CUBLAS_CHECK(st);
        CUDA_CHECK(cudaGetLastError());
    }
};

MatmulPlanCache::MatmulPlanCache(int device_id) : impl_(std::make_unique<Impl>(device_id)) {}
MatmulPlanCache::~MatmulPlanCache() = default;
MatmulPlanCache::MatmulPlanCache(MatmulPlanCache&&) noexcept = default;
MatmulPlanCache& MatmulPlanCache::operator=(MatmulPlanCache&&) noexcept = default;

void MatmulPlanCache::matmul(cublasLtHandle_t handle,
                             void* d, const void* a, const void* b, const void* bias,
                             std::byte* workspace, std::size_t workspace_size,
                             int m, int n, int k, int ldc,
                             int dtype_d, int dtype_a, int dtype_b, int dtype_bias,
                             bool transA, bool transB,
                             const float* scale_a, const float* scale_b,
                             bool accumulate,
                             cudaStream_t stream) {
    if (!impl_) {
        throw std::runtime_error("MatmulPlanCache: uninitialized");
    }
    MatmulPlanKey key;
    key.m = m;
    key.n = n;
    key.k = k;
    key.ldc = ldc;
    key.dtype_d = dtype_d;
    key.dtype_a = dtype_a;
    key.dtype_b = dtype_b;
    key.transA = transA;
    key.transB = transB;
    key.has_bias = (bias != nullptr);
    key.dtype_bias = key.has_bias ? dtype_bias : 0;
    key.has_scale_a = (scale_a != nullptr);
    key.has_scale_b = (scale_b != nullptr);
    key.workspace_size = workspace_size;

    MatmulPlan& plan = impl_->get_or_create_plan(key);
    impl_->ensure_algorithm(plan, key, handle, bias, scale_a, scale_b);
    impl_->maybe_autotune(plan, key, handle, d, a, b, bias, scale_a, scale_b, workspace, stream, accumulate);
    impl_->run_matmul(plan, key, handle, d, a, b, bias, scale_a, scale_b, workspace, stream, accumulate);
}

// ----------------------------------------------------------------------------
// Setup

cublasLtHandle_t create_cublaslt_handle() {
    cublasLtHandle_t handle;
    CUBLAS_CHECK(cublasLtCreate(&handle));
    return handle;
}

void destroy_cublaslt_handle(cublasLtHandle_t handle) noexcept {
    if (!handle) {
        return;
    }
    (void)cublasLtDestroy(handle);
}

// ----------------------------------------------------------------------------
// kernel launchers

/**
 * @brief Performs matrix multiplication using cuBLASLt: D = alpha * op(A) * op(B) + beta * C + bias.
 * 
 * Wrapper around cublasLtMatmul that is meant to support everything we need in llm.c
 * https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
 * 
 * This function wraps the cuBLASLt API to perform high-performance matrix multiplication,
 * supporting various data types (including FP8 via scaling factors), transposition modes,
 * and optional bias addition. It handles the creation of descriptors, layout definitions,
 * and heuristic search for the best algorithm.
 *
 * @tparam FloatC Type of the output matrix D and input accumulator C.
 * @tparam FloatA Type of the input matrix A.
 * @tparam FloatB Type of the input matrix B.
 * @tparam FloatBias Type of the bias vector.
 *
 * @param d Pointer to the output matrix D in device memory. Acts as both source (C) and destination (D).
 * @param a Pointer to the input matrix A in device memory.
 * @param b Pointer to the input matrix B in device memory.
 * @param bias Pointer to the bias vector in device memory. Can be nullptr if no bias is required.
 * @param workspace Pointer to device memory used as workspace for the operation.
 * @param workspace_size Size of the workspace buffer in bytes.
 * @param m Number of rows in the resulting matrix.
 * @param n Number of columns in the resulting matrix.
 * @param k Inner dimension of the matrix multiplication.
 * @param stream CUDA stream to execute the operation on.
 * @param handle Valid cuBLASLt handle.
 * @param scale_a Pointer to the scaling factor for matrix A (host or device). Used primarily for FP8 inputs.
 * @param scale_b Pointer to the scaling factor for matrix B (host or device). Used primarily for FP8 inputs.
 * @param mode Transposition mode for matrices A and B (e.g., NN, NT, TN, TT).
 * @param accumulate If true, accumulates the result into D (beta = 1.0). If false, overwrites D (beta = 0.0).
 * @param ldc_override Optional override for the leading dimension of C/D. If <= 0, defaults to m.
 *
 * @throws std::runtime_error If input pointers (a, b, d, bias) are not 16-byte aligned.
 * @throws std::runtime_error If scaling pointers are provided for non-byte-sized types (i.e., types other than FP8).
 * @throws std::runtime_error If no suitable cuBLASLt algorithm heuristic is found for the given configuration.
 */
template<class FloatC, class FloatA, class FloatB, class FloatBias>
void matmul_cublaslt(FloatC* d, const FloatA* a, const FloatB* b, const FloatBias* bias,
                     std::byte* workspace, std::size_t workspace_size,
                     int m, int n, int k, cudaStream_t stream, cublasLtHandle_t handle,
                     const float* scale_a, const float* scale_b, EMMTranspose mode, bool accumulate, int ldc_override = -1,
                     MatmulPlanCache* plan_cache = nullptr)
{
    bool has_bias = (bias != nullptr);

    // check alignment (some modes work unaligned, but it is always best to be aligned for performance)
    if(((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 || ((uintptr_t)d % 16) != 0 || ((uintptr_t)bias % 16) != 0) {
        throw std::runtime_error("All cuBLASLt pointers must be aligned!");
    }

    if (scale_a && sizeof(FloatA) != 1) {
        throw std::runtime_error("Scaling A is only supported for FP8");
    }
    if (scale_b && sizeof(FloatB) != 1) {
        throw std::runtime_error("Scaling B is only supported for FP8");
    }

    bool transA = mode == EMMTranspose::TN || mode == EMMTranspose::TT;
    bool transB = mode == EMMTranspose::NT || mode == EMMTranspose::TT;

    int ldc = (ldc_override > 0) ? ldc_override : m;
    if (plan_cache) {
        plan_cache->matmul(handle,
                           static_cast<void*>(d),
                           static_cast<const void*>(a),
                           static_cast<const void*>(b),
                           static_cast<const void*>(bias),
                           workspace, workspace_size,
                           m, n, k, ldc,
                           static_cast<int>(to_cuda_lib_type_enum<FloatC>),
                           static_cast<int>(to_cuda_lib_type_enum<FloatA>),
                           static_cast<int>(to_cuda_lib_type_enum<FloatB>),
                           static_cast<int>(to_cuda_lib_type_enum<FloatBias>),
                           transA, transB,
                           scale_a, scale_b,
                           accumulate,
                           stream);
        return;
    }

    // create the operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    int returnedResults = 0;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristic;

    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, (transA) ? &opTranspose : &opNoTranspose, sizeof(opTranspose)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, (transB) ? &opTranspose : &opNoTranspose, sizeof(opNoTranspose)));

    // define matrix layouts
    cublasLtMatrixLayout_t ALayout;
    cublasLtMatrixLayout_t BLayout;
    cublasLtMatrixLayout_t DLayout;
    cublasLtMatrixLayout_t CLayout;
    if (transA) {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&ALayout, to_cuda_lib_type_enum<FloatA>, k, m, k));
    } else {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&ALayout, to_cuda_lib_type_enum<FloatA>, m, k, m));
    }
    if (transB) {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&BLayout, to_cuda_lib_type_enum<FloatB>, n, k, n));
    } else {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&BLayout, to_cuda_lib_type_enum<FloatB>, k, n, k));
    }
    // cuBLASLt requires C in FP8 mode to be BF16 or FP32... (sigh)
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&CLayout, to_cuda_lib_type_enum<FloatC>, m, n, ldc));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&DLayout, to_cuda_lib_type_enum<FloatC>, m, n, ldc));

    // create a preference handle with specified max workspace
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &workspace_size, sizeof(workspace_size)));

    // setup epilogue and associated pointers for bias & gelu
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    if(has_bias){
        epilogue = CUBLASLT_EPILOGUE_BIAS;
    }
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (has_bias) {
        // cuBLASLt requires bias in FP8 mode to be BF16... (sigh)
        cublasDataType_t bias_data_type = to_cuda_lib_type_enum<FloatBias>; // force BF16 bias for FP8 mode
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }

    if(scale_a) {
        // scale pointer validity was checked above
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a, sizeof(&scale_a)));
    }
    if(scale_b) {
        // scale pointer validity was checked above
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b, sizeof(&scale_b)));
    }

    // find a suitable algorithm (cached internally so shouldn't take much CPU time in practice)
    cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, ALayout, BLayout, CLayout, DLayout,
                                   preference, 1, &heuristic, &returnedResults);
    if (returnedResults == 0) {
        throw std::runtime_error(fmt::format("No cuBLASLt algorithm: m: {}, n: {}, k: {}, bias: {}", n, m, k, has_bias));
    }

    // set whether to accumulate (i.e. D += C) or not - note this isn't considered in algorithm selection (?!)
    float one = 1.f;
    float zero = 0.f;
    float* alpha = &one;
    float* beta = accumulate ? &one : &zero;

    // call the matmul
    CUBLAS_CHECK(cublasLtMatmul(handle, operationDesc,
                               alpha, a, ALayout, b, BLayout, beta, d, CLayout, d, DLayout,
                               &heuristic.algo, workspace, workspace_size, stream));
    CUDA_CHECK(cudaGetLastError());

    // cleanups
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(operationDesc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(ALayout));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(BLayout));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(CLayout));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(DLayout));
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Performs matrix multiplication using cuBLASLt.
 *
 * This function acts as a wrapper around `matmul_cublaslt` to execute a matrix multiplication
 * operation of the form C = alpha * (A x B) + beta * C + bias, potentially with scaling factors.
 *
 * @param c Pointer to the output matrix C (device memory).
 * @param a Pointer to the input matrix A (device memory).
 * @param b Pointer to the input matrix B (device memory).
 * @param bias Pointer to the bias vector (device memory). Can be nullptr if no bias is applied.
 * @param scale_a Pointer to the scaling factor for matrix A (device memory). Used for quantized operations.
 * @param scale_b Pointer to the scaling factor for matrix B (device memory). Used for quantized operations.
 * @param handle The cuBLASLt handle used to manage the library context.
 * @param workspace Pointer to the workspace memory required by cuBLASLt (device memory).
 * @param workspace_size Size of the workspace memory in bytes.
 * @param M The number of rows in matrix A and C.
 * @param N The number of columns in matrix B and C.
 * @param K The number of columns in matrix A and rows in matrix B.
 * @param mode Enum specifying the transposition mode for the matrices (e.g., Transpose, NoTranspose).
 * @param accumulate If true, accumulates the result into the existing values of C (beta = 1). If false, overwrites C (beta = 0).
 * @param stream The CUDA stream on which the operation will be executed.
 */
void matmul(float* c, const float* a, const float* b, const float* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream, MatmulPlanCache* plan_cache) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate, -1, plan_cache);
}

/**
 * @brief Performs a strided matrix multiplication using cuBLASLt.
 *
 * This function acts as a wrapper around `matmul_cublaslt` to execute a matrix multiplication
 * operation of the form C = alpha * (A * B) + beta * C + bias, specifically handling
 * strided memory layouts and potential scaling factors for quantization.
 *
 * @param[out] c Pointer to the output matrix C (in device memory).
 * @param[in] a Pointer to the input matrix A (in device memory).
 * @param[in] b Pointer to the input matrix B (in device memory).
 * @param[in] bias Pointer to the bias vector (in device memory). Can be nullptr if no bias is applied.
 * @param[in] scale_a Pointer to the scaling factor for matrix A (host or device memory). Used for dequantization/scaling.
 * @param[in] scale_b Pointer to the scaling factor for matrix B (host or device memory). Used for dequantization/scaling.
 * @param[in] handle The cuBLASLt handle used to manage the operation context.
 * @param[in] workspace Pointer to the workspace memory allocated on the device.
 * @param[in] workspace_size Size of the workspace memory in bytes.
 * @param[in] M Number of rows in matrix A and C.
 * @param[in] N Number of columns in matrix B and C.
 * @param[in] K Number of columns in A and rows in B.
 * @param[in] mode Transpose mode for the operation (e.g., indicating if A or B are transposed).
 * @param[in] accumulate If true, accumulates the result into C (beta != 0). If false, overwrites C (beta = 0).
 * @param[in] ldc Leading dimension of matrix C.
 * @param[in] stream The CUDA stream on which the operation will be executed.
 */
void matmul_strided_c(nv_bfloat16* c, const nv_bfloat16* a, const nv_bfloat16* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
                      cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
                      int M, int N, int K, EMMTranspose mode, bool accumulate, int ldc, cudaStream_t stream, MatmulPlanCache* plan_cache) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate, ldc, plan_cache);
}

void matmul_strided_c(float* c, const float* a, const float* b, const float* bias, const float* scale_a, const float* scale_b,
                      cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
                      int M, int N, int K, EMMTranspose mode, bool accumulate, int ldc, cudaStream_t stream, MatmulPlanCache* plan_cache) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate, ldc, plan_cache);
}

/**
 * @brief Performs matrix multiplication using cuBLASLt: C = alpha * (A x B) + beta * C + bias.
 *
 * This function acts as a wrapper around `matmul_cublaslt` to execute a matrix multiplication
 * operation on the GPU. It supports mixed-precision inputs (nv_bfloat16) and float output,
 * along with optional bias addition and scaling.
 *
 * @param[out] c Pointer to the output matrix C (float).
 * @param[in] a Pointer to the input matrix A (nv_bfloat16).
 * @param[in] b Pointer to the input matrix B (nv_bfloat16).
 * @param[in] bias Pointer to the bias vector (float). Can be nullptr if no bias is applied.
 * @param[in] scale_a Pointer to the scaling factor for matrix A (float).
 * @param[in] scale_b Pointer to the scaling factor for matrix B (float).
 * @param[in] handle The cuBLASLt handle used to manage the library context.
 * @param[in] workspace Pointer to the device memory workspace required by cuBLASLt.
 * @param[in] workspace_size Size of the workspace in bytes.
 * @param[in] M Number of rows in matrix A and C.
 * @param[in] N Number of columns in matrix B and C.
 * @param[in] K Number of columns in A and rows in B (inner dimension).
 * @param[in] mode Transposition mode for the operation (e.g., transpose A, transpose B).
 * @param[in] accumulate If true, accumulates the result into C (beta != 0). If false, overwrites C (beta = 0).
 * @param[in] stream The CUDA stream on which the operation will be executed.
 */
void matmul(float* c, const nv_bfloat16* a, const nv_bfloat16* b, const float* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream, MatmulPlanCache* plan_cache) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate, -1, plan_cache);
}

/**
 * @brief Performs matrix multiplication using FP8 inputs and float output with cuBLASLt.
 *
 * This function acts as a wrapper around `matmul_cublaslt` to execute the operation:
 * C = alpha * (op(A) * op(B)) + beta * C + bias
 * where A and B are 8-bit floating point matrices (e4m3), and C is a 32-bit floating point matrix.
 * It handles scaling factors for the quantized inputs and optional bias addition.
 *
 * @param c Pointer to the output matrix C (float).
 * @param a Pointer to the input matrix A (__nv_fp8_e4m3).
 * @param b Pointer to the input matrix B (__nv_fp8_e4m3).
 * @param bias Pointer to the bias vector (float). Can be nullptr if no bias is applied.
 * @param scale_a Pointer to the scaling factor for matrix A (float).
 * @param scale_b Pointer to the scaling factor for matrix B (float).
 * @param handle The cuBLASLt handle used to manage the library context.
 * @param workspace Pointer to the device memory workspace required by cuBLASLt.
 * @param workspace_size Size of the workspace in bytes.
 * @param M Number of rows in matrix A and C.
 * @param N Number of columns in matrix B and C.
 * @param K Number of columns in A and rows in B (inner dimension).
 * @param mode Enum specifying the transposition mode for matrices A and B (e.g., Transpose, NoTranspose).
 * @param accumulate If true, accumulates the result into the existing values of C (beta = 1). If false, overwrites C (beta = 0).
 * @param stream The CUDA stream on which the operation will be executed.
 */
void matmul(float* c, const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, const float* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream, MatmulPlanCache* plan_cache) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate, -1, plan_cache);
}

/**
 * @brief Performs matrix multiplication using cuBLASLt with FP8 inputs and FP32 output.
 *
 * This function acts as a wrapper around `matmul_cublaslt` to execute a matrix multiplication
 * operation of the form C = alpha * (op(A) * op(B)) + beta * C + bias, where A and B are
 * 8-bit floating point matrices, and C is a 32-bit floating point matrix.
 *
 * @param[out] c Pointer to the output matrix C (FP32).
 * @param[in] a Pointer to the input matrix A (__nv_fp8_e4m3).
 * @param[in] b Pointer to the input matrix B (__nv_fp8_e4m3).
 * @param[in] bias Pointer to the bias vector (nv_bfloat16). Can be nullptr if no bias is applied.
 * @param[in] scale_a Pointer to the scaling factor for matrix A (float).
 * @param[in] scale_b Pointer to the scaling factor for matrix B (float).
 * @param[in] handle The cuBLASLt handle used to manage the library context.
 * @param[in] workspace Pointer to the device memory workspace required by cuBLASLt.
 * @param[in] workspace_size Size of the workspace in bytes.
 * @param[in] M Number of rows in matrix A and C.
 * @param[in] N Number of columns in matrix B and C.
 * @param[in] K Number of columns in A and rows in B (inner dimension).
 * @param[in] mode Enum specifying the transpose operation for matrices (e.g., Transpose, NoTranspose).
 * @param[in] accumulate If true, accumulates the result into the existing values of C (beta = 1.0).
 *                       If false, overwrites C (beta = 0.0).
 * @param[in] stream The CUDA stream on which the operation will be executed.
 */
void matmul(float* c, const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream, MatmulPlanCache* plan_cache) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate, -1, plan_cache);
}

/**
 * @brief Performs matrix multiplication using cuBLASLt with support for bfloat16 precision, bias addition, and scaling.
 *
 * This function acts as a wrapper around `matmul_cublaslt` to execute the operation:
 * C = alpha * (op(A) * op(B)) + beta * C + bias
 * where alpha and beta are derived from scale factors and accumulation settings.
 *
 * @param[out] c Pointer to the output matrix C in GPU memory.
 * @param[in] a Pointer to the input matrix A in GPU memory (nv_bfloat16).
 * @param[in] b Pointer to the input matrix B in GPU memory (nv_bfloat16).
 * @param[in] bias Pointer to the bias vector in GPU memory (nv_bfloat16). Can be nullptr if no bias is applied.
 * @param[in] scale_a Pointer to the scaling factor for matrix A (float).
 * @param[in] scale_b Pointer to the scaling factor for matrix B (float).
 * @param[in] handle The cuBLASLt handle used to manage the library context.
 * @param[in] workspace Pointer to the workspace memory allocated on the GPU.
 * @param[in] workspace_size Size of the workspace memory in bytes.
 * @param[in] M The number of rows in matrix A and C.
 * @param[in] N The number of columns in matrix B and C.
 * @param[in] K The number of columns in matrix A and rows in matrix B.
 * @param[in] mode Enum specifying the transpose operation for matrices A and B (e.g., Transpose, NoTranspose).
 * @param[in] accumulate If true, accumulates the result into the existing values of C (beta != 0). If false, overwrites C.
 * @param[in] stream The CUDA stream on which the operation will be executed.
 */
void matmul(nv_bfloat16* c, const nv_bfloat16* a, const nv_bfloat16* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream, MatmulPlanCache* plan_cache) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate, -1, plan_cache);
}

/**
 * @brief Performs matrix multiplication using cuBLASLt with FP8 inputs and BF16 output.
 *
 * This function acts as a wrapper around `matmul_cublaslt` to execute the operation:
 * C = alpha * (A * B) + beta * C + bias
 * where A and B are 8-bit floating point matrices, and C and bias are bfloat16.
 *
 * @param[out] c Pointer to the output matrix C (nv_bfloat16).
 * @param[in] a Pointer to the input matrix A (__nv_fp8_e4m3).
 * @param[in] b Pointer to the input matrix B (__nv_fp8_e4m3).
 * @param[in] bias Pointer to the bias vector (nv_bfloat16). Can be nullptr if no bias is applied.
 * @param[in] scale_a Pointer to the scaling factor for matrix A (float).
 * @param[in] scale_b Pointer to the scaling factor for matrix B (float).
 * @param[in] handle The cuBLASLt handle used to manage the library context.
 * @param[in] workspace Pointer to the device memory workspace required by cuBLASLt.
 * @param[in] workspace_size Size of the workspace in bytes.
 * @param[in] M Number of rows in matrix A and C.
 * @param[in] N Number of columns in matrix B and C.
 * @param[in] K Number of columns in A and rows in B.
 * @param[in] mode Transpose mode for the operation (EMMTranspose).
 * @param[in] accumulate If true, accumulates the result into C (beta != 0). If false, overwrites C (beta = 0).
 * @param[in] stream The CUDA stream on which the operation will be executed.
 */
void matmul(nv_bfloat16* c, const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream, MatmulPlanCache* plan_cache) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate, -1, plan_cache);
}

/**
 * @brief Performs matrix multiplication using cuBLASLt with mixed precision support.
 *
 * This function computes the matrix product C = A * B (or variations based on transpose mode),
 * optionally adding a bias vector and scaling factors. It acts as a wrapper around the
 * `matmul_cublaslt` implementation.
 *
 * @param[out] c Pointer to the output matrix C in nv_bfloat16 format.
 * @param[in] a Pointer to the input matrix A in __nv_fp8_e4m3 format.
 * @param[in] b Pointer to the input matrix B in __nv_fp8_e5m2 format.
 * @param[in] bias Pointer to the bias vector in nv_bfloat16 format (can be nullptr).
 * @param[in] scale_a Pointer to the scaling factor for matrix A (float).
 * @param[in] scale_b Pointer to the scaling factor for matrix B (float).
 * @param[in] handle The cuBLASLt handle to use for the operation.
 * @param[in] workspace Pointer to the device memory workspace required by cuBLASLt.
 * @param[in] workspace_size Size of the workspace in bytes.
 * @param[in] M Number of rows in matrix A and C.
 * @param[in] N Number of columns in matrix B and C.
 * @param[in] K Number of columns in A and rows in B (inner dimension).
 * @param[in] mode Transpose mode for the operation (e.g., Transpose, NoTranspose).
 * @param[in] accumulate If true, accumulates the result into the existing values of C (C += A * B).
 *                       If false, overwrites C (C = A * B).
 * @param[in] stream The CUDA stream to execute the kernel on.
 */
void matmul(nv_bfloat16* c, const __nv_fp8_e4m3* a, const __nv_fp8_e5m2* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream, MatmulPlanCache* plan_cache) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate, -1, plan_cache);
}
