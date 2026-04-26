// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Based on llm.c https://github.com/karpathy/llm.c

#include <cublasLt.h>
#include <cublas_v2.h>
#include <fmt/core.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "kernels.h"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

namespace {
std::mutex g_fallback_cublas_mutex;
std::unordered_map<int, cublasHandle_t> g_fallback_cublas_handles;

struct MatmulTraceKey {
    int m;
    int n;
    int k;
    int mode;
    int lda;
    int ldb;
    int ldc;
    int a_type;
    int b_type;
    int c_type;
    int has_bias;
    int alpha_milli;
    int beta_milli;
    int algo_id;
    int tile_id;
    int stages_id;
    int splitk;

    bool operator==(const MatmulTraceKey& other) const {
        return m == other.m && n == other.n && k == other.k && mode == other.mode && lda == other.lda &&
               ldb == other.ldb && ldc == other.ldc && a_type == other.a_type && b_type == other.b_type &&
               c_type == other.c_type && has_bias == other.has_bias && alpha_milli == other.alpha_milli &&
               beta_milli == other.beta_milli && algo_id == other.algo_id && tile_id == other.tile_id &&
               stages_id == other.stages_id && splitk == other.splitk;
    }
};

struct MatmulTraceKeyHash {
    std::size_t operator()(const MatmulTraceKey& key) const {
        std::size_t h = 1469598103934665603ull;
        auto mix = [&](int v) {
            h ^= static_cast<std::size_t>(v) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        };
        mix(key.m);
        mix(key.n);
        mix(key.k);
        mix(key.mode);
        mix(key.lda);
        mix(key.ldb);
        mix(key.ldc);
        mix(key.a_type);
        mix(key.b_type);
        mix(key.c_type);
        mix(key.has_bias);
        mix(key.alpha_milli);
        mix(key.beta_milli);
        mix(key.algo_id);
        mix(key.tile_id);
        mix(key.stages_id);
        mix(key.splitk);
        return h;
    }
};

std::mutex g_matmul_trace_mutex;
std::unordered_map<MatmulTraceKey, std::size_t, MatmulTraceKeyHash> g_matmul_trace_counts;
std::once_flag g_matmul_trace_atexit_once;

struct MatmulAutotuneKey {
    int device;
    int m;
    int n;
    int k;
    int mode;
    int lda;
    int ldb;
    int ldc;
    int a_type;
    int b_type;
    int c_type;
    int bias_type;
    int has_bias;
    int has_scale_a;
    int has_scale_b;
    int alpha_milli;
    int beta_milli;
    int compute_type;
    int workspace_mb;

    bool operator==(const MatmulAutotuneKey& other) const {
        return device == other.device && m == other.m && n == other.n && k == other.k && mode == other.mode &&
               lda == other.lda && ldb == other.ldb && ldc == other.ldc && a_type == other.a_type &&
               b_type == other.b_type && c_type == other.c_type && bias_type == other.bias_type &&
               has_bias == other.has_bias && has_scale_a == other.has_scale_a && has_scale_b == other.has_scale_b &&
               alpha_milli == other.alpha_milli && beta_milli == other.beta_milli &&
               compute_type == other.compute_type && workspace_mb == other.workspace_mb;
    }
};

struct MatmulAutotuneKeyHash {
    std::size_t operator()(const MatmulAutotuneKey& key) const {
        std::size_t h = 1469598103934665603ull;
        auto mix = [&](int v) {
            h ^= static_cast<std::size_t>(v) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        };
        mix(key.device);
        mix(key.m);
        mix(key.n);
        mix(key.k);
        mix(key.mode);
        mix(key.lda);
        mix(key.ldb);
        mix(key.ldc);
        mix(key.a_type);
        mix(key.b_type);
        mix(key.c_type);
        mix(key.bias_type);
        mix(key.has_bias);
        mix(key.has_scale_a);
        mix(key.has_scale_b);
        mix(key.alpha_milli);
        mix(key.beta_milli);
        mix(key.compute_type);
        mix(key.workspace_mb);
        return h;
    }
};

struct MatmulAutotuneEntry {
    cublasLtMatmulAlgo_t algo;
    float elapsed_ms = 0.0f;
    int candidate_index = -1;
};

std::mutex g_matmul_autotune_mutex;
std::unordered_map<MatmulAutotuneKey, MatmulAutotuneEntry, MatmulAutotuneKeyHash> g_matmul_autotune_cache;

static bool matmul_trace_enabled() {
    static const bool enabled = (std::getenv("SUROGATE_DEBUG_MATMUL_SHAPES") != nullptr);
    return enabled;
}

static int round_milli(float value) {
    return static_cast<int>(value * 1000.0f + (value >= 0.0f ? 0.5f : -0.5f));
}

static bool matmul_autotune_enabled() {
    static const bool enabled = (std::getenv("SUROGATE_DISABLE_MATMUL_AUTOTUNE") == nullptr);
    return enabled;
}

static bool matmul_autotune_debug_enabled() {
    static const bool enabled = (std::getenv("SUROGATE_DEBUG_MATMUL_AUTOTUNE") != nullptr);
    return enabled;
}

static int matmul_autotune_repetitions() {
    static const int reps = []() {
        if (const char* env = std::getenv("SUROGATE_MATMUL_AUTOTUNE_REPS")) {
            char* end = nullptr;
            const long val = std::strtol(env, &end, 10);
            if (end != env && val > 0) return static_cast<int>(std::min<long>(val, 16));
        }
        return 3;
    }();
    return reps;
}

static int matmul_autotune_max_candidates() {
    static const int candidates = []() {
        if (const char* env = std::getenv("SUROGATE_MATMUL_AUTOTUNE_MAX_CANDIDATES")) {
            char* end = nullptr;
            const long val = std::strtol(env, &end, 10);
            if (end != env && val > 0) return static_cast<int>(std::min<long>(val, 32));
        }
        return 8;
    }();
    return candidates;
}

static std::size_t matmul_autotune_max_temp_bytes() {
    static const std::size_t bytes = []() {
        if (const char* env = std::getenv("SUROGATE_MATMUL_AUTOTUNE_MAX_TEMP_MB")) {
            char* end = nullptr;
            const long val = std::strtol(env, &end, 10);
            if (end != env && val > 0) return static_cast<std::size_t>(val) * 1024ull * 1024ull;
        }
        return static_cast<std::size_t>(512) * 1024ull * 1024ull;
    }();
    return bytes;
}

static bool stream_is_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    return (cudaStreamIsCapturing(stream, &cap) == cudaSuccess) && (cap != cudaStreamCaptureStatusNone);
}

static int get_algo_attr_int(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attr) {
    int value = -1;
    std::size_t written = 0;
    cublasStatus_t status = cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, &value, sizeof(value), &written);
    if (status != CUBLAS_STATUS_SUCCESS || written == 0) {
        return -1;
    }
    return value;
}

static void dump_matmul_trace() {
    std::vector<std::pair<MatmulTraceKey, std::size_t>> entries;
    {
        std::lock_guard<std::mutex> lock(g_matmul_trace_mutex);
        entries.reserve(g_matmul_trace_counts.size());
        for (const auto& item : g_matmul_trace_counts) {
            entries.push_back(item);
        }
    }

    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

    std::fprintf(stderr, "[MATMUL-SHAPES] unique=%zu\n", entries.size());
    const std::size_t limit = std::min<std::size_t>(entries.size(), 128);
    for (std::size_t i = 0; i < limit; ++i) {
        const auto& key = entries[i].first;
        std::fprintf(stderr,
                     "[MATMUL-SHAPES] count=%zu m=%d n=%d k=%d mode=%d lda=%d ldb=%d ldc=%d "
                     "types=%d/%d/%d bias=%d alpha_milli=%d beta_milli=%d algo=%d tile=%d stages=%d splitk=%d\n",
                     entries[i].second,
                     key.m,
                     key.n,
                     key.k,
                     key.mode,
                     key.lda,
                     key.ldb,
                     key.ldc,
                     key.a_type,
                     key.b_type,
                     key.c_type,
                     key.has_bias,
                     key.alpha_milli,
                     key.beta_milli,
                     key.algo_id,
                     key.tile_id,
                     key.stages_id,
                     key.splitk);
    }
}

static void record_matmul_trace(int m,
                                int n,
                                int k,
                                EMMTranspose mode,
                                int lda,
                                int ldb,
                                int ldc,
                                int a_type,
                                int b_type,
                                int c_type,
                                bool has_bias,
                                float alpha,
                                float beta,
                                const cublasLtMatmulAlgo_t& algo) {
    if (!matmul_trace_enabled()) {
        return;
    }
    std::call_once(g_matmul_trace_atexit_once, []() { std::atexit(dump_matmul_trace); });

    MatmulTraceKey key{
        m,
        n,
        k,
        static_cast<int>(mode),
        lda,
        ldb,
        ldc,
        a_type,
        b_type,
        c_type,
        has_bias ? 1 : 0,
        round_milli(alpha),
        round_milli(beta),
        get_algo_attr_int(algo, CUBLASLT_ALGO_CONFIG_ID),
        get_algo_attr_int(algo, CUBLASLT_ALGO_CONFIG_TILE_ID),
        get_algo_attr_int(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID),
        get_algo_attr_int(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
    };

    std::lock_guard<std::mutex> lock(g_matmul_trace_mutex);
    ++g_matmul_trace_counts[key];
}

static bool matmul_heuristic_debug_enabled() {
    static const bool enabled = (std::getenv("SUROGATE_DEBUG_MATMUL_HEURISTICS") != nullptr);
    return enabled;
}

static void maybe_dump_matmul_heuristics(int m,
                                         int n,
                                         int k,
                                         EMMTranspose mode,
                                         int lda,
                                         int ldb,
                                         int ldc,
                                         int a_type,
                                         int b_type,
                                         int c_type,
                                         bool has_bias,
                                         const cublasLtMatmulHeuristicResult_t* heuristics,
                                         int returned_results) {
    if (!matmul_heuristic_debug_enabled()) {
        return;
    }
    static std::mutex log_mutex;
    static int log_count = 0;
    std::lock_guard<std::mutex> lock(log_mutex);
    if (log_count >= 64) {
        return;
    }
    ++log_count;
    std::fprintf(stderr,
                 "[MATMUL-HEUR] shape m=%d n=%d k=%d mode=%d lda=%d ldb=%d ldc=%d types=%d/%d/%d bias=%d "
                 "returned=%d\n",
                 m,
                 n,
                 k,
                 static_cast<int>(mode),
                 lda,
                 ldb,
                 ldc,
                 a_type,
                 b_type,
                 c_type,
                 has_bias ? 1 : 0,
                 returned_results);
    for (int i = 0; i < returned_results; ++i) {
        const auto& result = heuristics[i];
        std::fprintf(stderr,
                     "[MATMUL-HEUR]   idx=%d state=%d workspace=%zu waves=%.3f algo=%d tile=%d stages=%d splitk=%d\n",
                     i,
                     static_cast<int>(result.state),
                     static_cast<std::size_t>(result.workspaceSize),
                     static_cast<double>(result.wavesCount),
                     get_algo_attr_int(result.algo, CUBLASLT_ALGO_CONFIG_ID),
                     get_algo_attr_int(result.algo, CUBLASLT_ALGO_CONFIG_TILE_ID),
                     get_algo_attr_int(result.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID),
                     get_algo_attr_int(result.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM));
    }
}

template <class FloatC, class FloatA, class FloatB>
static bool try_get_cached_matmul_algo(const MatmulAutotuneKey& key, cublasLtMatmulAlgo_t& algo_out) {
    if (!matmul_autotune_enabled()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_matmul_autotune_mutex);
    auto it = g_matmul_autotune_cache.find(key);
    if (it == g_matmul_autotune_cache.end()) {
        return false;
    }
    algo_out = it->second.algo;
    return true;
}

template <class FloatC, class FloatA, class FloatB>
static bool tune_matmul_algo(const MatmulAutotuneKey& key,
                             cublasLtHandle_t handle,
                             cublasLtMatmulDesc_t operationDesc,
                             cublasLtMatrixLayout_t ALayout,
                             cublasLtMatrixLayout_t BLayout,
                             cublasLtMatrixLayout_t CLayout,
                             cublasLtMatrixLayout_t DLayout,
                             const float* alpha,
                             const FloatA* a,
                             const FloatB* b,
                             const float* beta,
                             std::byte* workspace,
                             std::size_t workspace_size,
                             int ldc,
                             int n,
                             cudaStream_t stream,
                             const cublasLtMatmulHeuristicResult_t* heuristics,
                             int returned_results,
                             cublasLtMatmulAlgo_t& algo_out) {
    if (!matmul_autotune_enabled()) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(g_matmul_autotune_mutex);
        auto it = g_matmul_autotune_cache.find(key);
        if (it != g_matmul_autotune_cache.end()) {
            algo_out = it->second.algo;
            return true;
        }
    }

    if (stream_is_capturing(stream) || returned_results <= 1) {
        return false;
    }

    const std::size_t temp_elems = static_cast<std::size_t>(ldc) * static_cast<std::size_t>(n);
    const std::size_t temp_bytes = temp_elems * sizeof(FloatC);
    if (temp_bytes == 0 || temp_bytes > matmul_autotune_max_temp_bytes()) {
        return false;
    }

    FloatC* temp_d = nullptr;
    cudaError_t alloc_status = cudaMalloc(&temp_d, temp_bytes);
    if (alloc_status != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int max_candidates = std::min(returned_results, matmul_autotune_max_candidates());
    const int reps = matmul_autotune_repetitions();
    float best_ms = INFINITY;
    int best_idx = -1;
    cublasLtMatmulAlgo_t best_algo{};

    for (int i = 0; i < max_candidates; ++i) {
        const auto& candidate = heuristics[i];
        if (candidate.state != CUBLAS_STATUS_SUCCESS || candidate.workspaceSize > workspace_size) {
            continue;
        }

        cublasStatus_t status = cublasLtMatmul(handle,
                                               operationDesc,
                                               alpha,
                                               a,
                                               ALayout,
                                               b,
                                               BLayout,
                                               beta,
                                               temp_d,
                                               CLayout,
                                               temp_d,
                                               DLayout,
                                               &candidate.algo,
                                               workspace,
                                               workspace_size,
                                               stream);
        if (status != CUBLAS_STATUS_SUCCESS) {
            continue;
        }

        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int r = 0; r < reps; ++r) {
            status = cublasLtMatmul(handle,
                                    operationDesc,
                                    alpha,
                                    a,
                                    ALayout,
                                    b,
                                    BLayout,
                                    beta,
                                    temp_d,
                                    CLayout,
                                    temp_d,
                                    DLayout,
                                    &candidate.algo,
                                    workspace,
                                    workspace_size,
                                    stream);
            if (status != CUBLAS_STATUS_SUCCESS) {
                break;
            }
        }
        if (status != CUBLAS_STATUS_SUCCESS) {
            continue;
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        elapsed_ms /= static_cast<float>(reps);
        if (elapsed_ms < best_ms) {
            best_ms = elapsed_ms;
            best_idx = i;
            best_algo = candidate.algo;
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(temp_d));

    if (best_idx < 0) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(g_matmul_autotune_mutex);
        auto [it, inserted] = g_matmul_autotune_cache.emplace(key, MatmulAutotuneEntry{best_algo, best_ms, best_idx});
        if (!inserted) {
            algo_out = it->second.algo;
            return true;
        }
    }

    if (matmul_autotune_debug_enabled()) {
        std::fprintf(stderr,
                     "[MATMUL-AUTOTUNE] m=%d n=%d k=%d mode=%d types=%d/%d/%d beta_milli=%d best_idx=%d "
                     "elapsed_ms=%.4f algo=%d tile=%d stages=%d splitk=%d candidates=%d\n",
                     key.m,
                     key.n,
                     key.k,
                     key.mode,
                     key.a_type,
                     key.b_type,
                     key.c_type,
                     key.beta_milli,
                     best_idx,
                     best_ms,
                     get_algo_attr_int(best_algo, CUBLASLT_ALGO_CONFIG_ID),
                     get_algo_attr_int(best_algo, CUBLASLT_ALGO_CONFIG_TILE_ID),
                     get_algo_attr_int(best_algo, CUBLASLT_ALGO_CONFIG_STAGES_ID),
                     get_algo_attr_int(best_algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
                     max_candidates);
    }

    algo_out = best_algo;
    return true;
}
}  // namespace

cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;

static cublasHandle_t get_fallback_cublas_handle() {
    int device = -1;
    CUDA_CHECK(cudaGetDevice(&device));

    std::lock_guard<std::mutex> lock(g_fallback_cublas_mutex);
    auto it = g_fallback_cublas_handles.find(device);
    if (it != g_fallback_cublas_handles.end()) {
        return it->second;
    }

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    // Allow tensor cores for BF16/FP16.
    (void)cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    g_fallback_cublas_handles.emplace(device, handle);
    return handle;
}

void init_cublas_fallback_handle() {
    (void)get_fallback_cublas_handle();
}

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
inline void cublasCheck(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(fmt::format("cuBLAS ERROR ({}) at {}:{}", (int)status, file, line));
    }
}
#ifdef CUBLAS_CHECK
#undef CUBLAS_CHECK
#endif
#define CUBLAS_CHECK(status)                       \
    {                                              \
        cublasCheck((status), __FILE__, __LINE__); \
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
template <class FloatC, class FloatA, class FloatB, class FloatBias>
void matmul_cublaslt(FloatC* d,
                     const FloatA* a,
                     const FloatB* b,
                     const FloatBias* bias,
                     std::byte* workspace,
                     std::size_t workspace_size,
                     int m,
                     int n,
                     int k,
                     cudaStream_t stream,
                     cublasLtHandle_t handle,
                     const float* scale_a,
                     const float* scale_b,
                     EMMTranspose mode,
                     float alpha_val,
                     float beta_val,
                     int ldc_override = -1,
                     int lda_override = -1,
                     int ldb_override = -1) {
    static const bool debug_fallback = (std::getenv("SUROGATE_DEBUG_MATMUL_FALLBACK") != nullptr);
    static int fallback_log_count = 0;

    bool has_bias = (bias != nullptr);

    // check alignment (some modes work unaligned, but it is always best to be aligned for performance)
    if (((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 || ((uintptr_t)d % 16) != 0 ||
        ((uintptr_t)bias % 16) != 0) {
        throw std::runtime_error("All cuBLASLt pointers must be aligned!");
    }

    // Prefer BF16-native compute mode for BF16 IO to improve robustness on small-K shapes.
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    if constexpr (std::is_same_v<FloatA, nv_bfloat16> && std::is_same_v<FloatB, nv_bfloat16>) {
        compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
    }

    // create the operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, compute_type, CUDA_R_32F));

    int returnedResults = 0;
    cublasLtMatmulPreference_t preference;
    bool transA = mode == EMMTranspose::TN || mode == EMMTranspose::TT;
    bool transB = mode == EMMTranspose::NT || mode == EMMTranspose::TT;

    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc,
                                                CUBLASLT_MATMUL_DESC_TRANSA,
                                                (transA) ? &opTranspose : &opNoTranspose,
                                                sizeof(opTranspose)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc,
                                                CUBLASLT_MATMUL_DESC_TRANSB,
                                                (transB) ? &opTranspose : &opNoTranspose,
                                                sizeof(opNoTranspose)));

    // define matrix layouts
    cublasLtMatrixLayout_t ALayout;
    cublasLtMatrixLayout_t BLayout;
    cublasLtMatrixLayout_t DLayout;
    cublasLtMatrixLayout_t CLayout;
    int lda = (lda_override > 0) ? lda_override : (transA ? k : m);
    if (transA) {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&ALayout, to_cuda_lib_type_enum<FloatA>, k, m, lda));
    } else {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&ALayout, to_cuda_lib_type_enum<FloatA>, m, k, lda));
    }
    int ldb = (ldb_override > 0) ? ldb_override : (transB ? n : k);
    if (transB) {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&BLayout, to_cuda_lib_type_enum<FloatB>, n, k, ldb));
    } else {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&BLayout, to_cuda_lib_type_enum<FloatB>, k, n, ldb));
    }
    int ldc = (ldc_override > 0) ? ldc_override : m;
    // cuBLASLt requires C in FP8 mode to be BF16 or FP32... (sigh)
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&CLayout, to_cuda_lib_type_enum<FloatC>, m, n, ldc));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&DLayout, to_cuda_lib_type_enum<FloatC>, m, n, ldc));

    // create a preference handle with specified max workspace
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
                                                      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &workspace_size,
                                                      sizeof(workspace_size)));

    // setup epilogue and associated pointers for bias & gelu
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    if (has_bias) {
        epilogue = CUBLASLT_EPILOGUE_BIAS;
    }
    CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (has_bias) {
        // cuBLASLt requires bias in FP8 mode to be BF16... (sigh)
        cublasDataType_t bias_data_type = to_cuda_lib_type_enum<FloatBias>;  // force BF16 bias for FP8 mode
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc,
                                                    CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                    &bias_data_type,
                                                    sizeof(bias_data_type)));
        CUBLAS_CHECK(
            cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }

    if (scale_a) {
        if (sizeof(FloatA) != 1) {
            throw std::runtime_error("Scaling A is only supported for FP8");
        }
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc,
                                                    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                    &scale_a,
                                                    sizeof(&scale_a)));
    }
    if (scale_b) {
        if (sizeof(FloatB) != 1) {
            throw std::runtime_error("Scaling B is only supported for FP8");
        }
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc,
                                                    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                    &scale_b,
                                                    sizeof(&scale_b)));
    }

    // find a suitable algorithm (cached internally so shouldn't take much CPU time in practice)
    // Request multiple algorithms to allow fallback if primary fails
    constexpr int kMaxAlgos = 32;
    cublasLtMatmulHeuristicResult_t heuristics[kMaxAlgos];
    cublasLtMatmulAlgoGetHeuristic(handle,
                                   operationDesc,
                                   ALayout,
                                   BLayout,
                                   CLayout,
                                   DLayout,
                                   preference,
                                   kMaxAlgos,
                                   heuristics,
                                   &returnedResults);
    maybe_dump_matmul_heuristics(m,
                                 n,
                                 k,
                                 mode,
                                 lda,
                                 ldb,
                                 ldc,
                                 static_cast<int>(to_cuda_lib_type_enum<FloatA>),
                                 static_cast<int>(to_cuda_lib_type_enum<FloatB>),
                                 static_cast<int>(to_cuda_lib_type_enum<FloatC>),
                                 has_bias,
                                 heuristics,
                                 returnedResults);
    if (returnedResults == 0) {
        throw std::runtime_error(
            fmt::format("No cuBLASLt algorithm: m: {}, n: {}, k: {}, bias: {}", n, m, k, has_bias));
    }

    // Use explicit alpha/beta values for output scaling and accumulation
    float* alpha = const_cast<float*>(&alpha_val);
    float* beta = const_cast<float*>(&beta_val);

    // Try algorithms in order until one succeeds.
    // Some heuristics can be reported with non-success state; skip those.
    cublasStatus_t matmul_status = CUBLAS_STATUS_NOT_SUPPORTED;
    cublasStatus_t gemm_status = CUBLAS_STATUS_NOT_SUPPORTED;
    bool fallback_tried = false;
    int algos_tried = 0;

    int device_id_for_key = -1;
    CUDA_CHECK(cudaGetDevice(&device_id_for_key));
    const MatmulAutotuneKey autotune_key{
        device_id_for_key,
        m,
        n,
        k,
        static_cast<int>(mode),
        lda,
        ldb,
        ldc,
        static_cast<int>(to_cuda_lib_type_enum<FloatA>),
        static_cast<int>(to_cuda_lib_type_enum<FloatB>),
        static_cast<int>(to_cuda_lib_type_enum<FloatC>),
        static_cast<int>(to_cuda_lib_type_enum<FloatBias>),
        has_bias ? 1 : 0,
        scale_a ? 1 : 0,
        scale_b ? 1 : 0,
        round_milli(alpha_val),
        round_milli(beta_val),
        static_cast<int>(compute_type),
        static_cast<int>(workspace_size / (1024ull * 1024ull)),
    };

    cublasLtMatmulAlgo_t tuned_algo{};
    if (tune_matmul_algo<FloatC, FloatA, FloatB>(autotune_key,
                                                 handle,
                                                 operationDesc,
                                                 ALayout,
                                                 BLayout,
                                                 CLayout,
                                                 DLayout,
                                                 alpha,
                                                 a,
                                                 b,
                                                 beta,
                                                 workspace,
                                                 workspace_size,
                                                 ldc,
                                                 n,
                                                 stream,
                                                 heuristics,
                                                 returnedResults,
                                                 tuned_algo)) {
        matmul_status = cublasLtMatmul(handle,
                                       operationDesc,
                                       alpha,
                                       a,
                                       ALayout,
                                       b,
                                       BLayout,
                                       beta,
                                       d,
                                       CLayout,
                                       d,
                                       DLayout,
                                       &tuned_algo,
                                       workspace,
                                       workspace_size,
                                       stream);
        ++algos_tried;
        if (matmul_status == CUBLAS_STATUS_SUCCESS) {
            record_matmul_trace(m,
                                n,
                                k,
                                mode,
                                lda,
                                ldb,
                                ldc,
                                static_cast<int>(to_cuda_lib_type_enum<FloatA>),
                                static_cast<int>(to_cuda_lib_type_enum<FloatB>),
                                static_cast<int>(to_cuda_lib_type_enum<FloatC>),
                                has_bias,
                                alpha_val,
                                beta_val,
                                tuned_algo);
        } else {
            std::lock_guard<std::mutex> lock(g_matmul_autotune_mutex);
            g_matmul_autotune_cache.erase(autotune_key);
        }
    }

    for (int i = 0; i < returnedResults; ++i) {
        if (matmul_status == CUBLAS_STATUS_SUCCESS) {
            break;
        }
        if (heuristics[i].state != CUBLAS_STATUS_SUCCESS) {
            continue;
        }
        matmul_status = cublasLtMatmul(handle,
                                       operationDesc,
                                       alpha,
                                       a,
                                       ALayout,
                                       b,
                                       BLayout,
                                       beta,
                                       d,
                                       CLayout,
                                       d,
                                       DLayout,
                                       &heuristics[i].algo,
                                       workspace,
                                       workspace_size,
                                       stream);
        ++algos_tried;
        if (matmul_status == CUBLAS_STATUS_SUCCESS) {
            record_matmul_trace(m,
                                n,
                                k,
                                mode,
                                lda,
                                ldb,
                                ldc,
                                static_cast<int>(to_cuda_lib_type_enum<FloatA>),
                                static_cast<int>(to_cuda_lib_type_enum<FloatB>),
                                static_cast<int>(to_cuda_lib_type_enum<FloatC>),
                                has_bias,
                                alpha_val,
                                beta_val,
                                heuristics[i].algo);
            break;
        }
    }
    if (matmul_status != CUBLAS_STATUS_SUCCESS) {
        auto maybe_log_fallback = [&](const char* phase) {
            if (!debug_fallback || fallback_log_count >= 256) {
                return;
            }
            ++fallback_log_count;
            cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
            const bool capturing =
                (cudaStreamIsCapturing(stream, &cap) == cudaSuccess) && (cap != cudaStreamCaptureStatusNone);
            std::fprintf(stderr,
                         "[MATMUL-FALLBACK] phase=%s capture=%d m=%d n=%d k=%d mode=%d "
                         "A_type=%d B_type=%d C_type=%d status=%d\n",
                         phase,
                         (int)capturing,
                         m,
                         n,
                         k,
                         (int)mode,
                         (int)to_cuda_lib_type_enum<FloatA>,
                         (int)to_cuda_lib_type_enum<FloatB>,
                         (int)to_cuda_lib_type_enum<FloatC>,
                         (int)matmul_status);
        };

        maybe_log_fallback("cublasLt_failed");

        // Fallback to cuBLAS GEMM for non-FP8 types when cuBLASLt fails.
        // This is slower but more robust for certain BF16 shapes.
        if constexpr (sizeof(FloatA) != 1 && sizeof(FloatB) != 1) {
            const bool has_bias = (bias != nullptr);
            const int ldc_fallback = (ldc_override > 0) ? ldc_override : m;

            // Strided output with bias isn't supported by the simple bias kernel.
            // Also, non-unit alpha is not supported in fallback path yet.
            if ((!has_bias || ldc_fallback == m) && alpha_val == 1.0f) {
                fallback_tried = true;
                maybe_log_fallback("trying_cublas_gemmex");
                cublasHandle_t fallback_handle = get_fallback_cublas_handle();
                CUBLAS_CHECK(cublasSetStream(fallback_handle, stream));

                const cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
                const cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
                const int lda_fb = (lda_override > 0) ? lda_override : (transA ? k : m);
                const int ldb_fb = (ldb_override > 0) ? ldb_override : (transB ? n : k);

                // Use the explicit alpha/beta values
                float alpha_copy = alpha_val;
                float beta_copy = beta_val;
                float* alpha_ptr = &alpha_copy;
                float* beta_ptr = &beta_copy;

                auto try_gemm = [&](cublasComputeType_t compute_type, cublasGemmAlgo_t algo) {
                    return cublasGemmEx(fallback_handle,
                                        opA,
                                        opB,
                                        m,
                                        n,
                                        k,
                                        alpha_ptr,
                                        a,
                                        to_cuda_lib_type_enum<FloatA>,
                                        lda_fb,
                                        b,
                                        to_cuda_lib_type_enum<FloatB>,
                                        ldb_fb,
                                        beta_ptr,
                                        d,
                                        to_cuda_lib_type_enum<FloatC>,
                                        ldc_fallback,
                                        compute_type,
                                        algo);
                };

                // Some BF16 shapes are accepted only with FAST_16BF compute while others
                // accept plain FP32 compute. Try both for robustness.
                constexpr cublasComputeType_t bf16_compute_candidates[] = {CUBLAS_COMPUTE_32F_FAST_16BF,
                                                                           CUBLAS_COMPUTE_32F};
                constexpr cublasComputeType_t default_compute_candidates[] = {CUBLAS_COMPUTE_32F};

                const cublasComputeType_t* compute_candidates = default_compute_candidates;
                int compute_candidate_count = 1;
                if constexpr (std::is_same_v<FloatA, nv_bfloat16> && std::is_same_v<FloatB, nv_bfloat16>) {
                    compute_candidates = bf16_compute_candidates;
                    compute_candidate_count =
                        static_cast<int>(sizeof(bf16_compute_candidates) / sizeof(bf16_compute_candidates[0]));
                }

                for (int ci = 0; ci < compute_candidate_count && gemm_status != CUBLAS_STATUS_SUCCESS; ++ci) {
                    gemm_status = try_gemm(compute_candidates[ci], CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    if (gemm_status == CUBLAS_STATUS_SUCCESS) {
                        break;
                    }
                    // Retry without tensor ops for maximum compatibility.
                    gemm_status = try_gemm(compute_candidates[ci], CUBLAS_GEMM_DEFAULT);
                }

                if (gemm_status == CUBLAS_STATUS_SUCCESS) {
                    maybe_log_fallback("cublas_gemmex_success");
                    if (has_bias) {
                        if constexpr (std::is_same_v<FloatC, float> && std::is_same_v<FloatBias, float>) {
                            // Treat output as row-major (n x m) for bias add; only product matters.
                            add_bias(d, bias, /*B=*/1, /*T=*/n, /*OC=*/m, stream);
                        } else if constexpr (std::is_same_v<FloatC, nv_bfloat16> &&
                                             std::is_same_v<FloatBias, nv_bfloat16>) {
                            add_bias(d, bias, /*B=*/1, /*T=*/n, /*OC=*/m, stream);
                        } else {
                            throw std::runtime_error("cuBLASLt failed and fallback bias add has incompatible dtype");
                        }
                    }
                    // Success via fallback.
                    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
                    CUBLAS_CHECK(cublasLtMatmulDescDestroy(operationDesc));
                    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(ALayout));
                    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(BLayout));
                    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(CLayout));
                    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(DLayout));
                    CUDA_CHECK(cudaGetLastError());
                    return;
                }
            }
        }
        int device_id = -1;
        cudaGetDevice(&device_id);
        if (fallback_tried) {
            throw std::runtime_error(
                fmt::format("cuBLAS ERROR ({}) at {}:{} - device: {}, m: {}, n: {}, k: {}, mode: {}, transA: {}, "
                            "transB: {}, lda: {}, ldb: {}, ldc: {}, ws_size: {}, A_type: {}, B_type: {}, C_type: {}, "
                            "algos_tried: {}/{}, gemm_status: {}",
                            (int)matmul_status,
                            __FILE__,
                            __LINE__,
                            device_id,
                            m,
                            n,
                            k,
                            (int)mode,
                            (int)transA,
                            (int)transB,
                            lda,
                            ldb,
                            ldc,
                            workspace_size,
                            (int)to_cuda_lib_type_enum<FloatA>,
                            (int)to_cuda_lib_type_enum<FloatB>,
                            (int)to_cuda_lib_type_enum<FloatC>,
                            algos_tried,
                            returnedResults,
                            (int)gemm_status));
        }
        throw std::runtime_error(fmt::format(
            "cuBLAS ERROR ({}) at {}:{} - device: {}, m: {}, n: {}, k: {}, mode: {}, transA: {}, transB: {}, lda: {}, "
            "ldb: {}, ldc: {}, ws_size: {}, A_type: {}, B_type: {}, C_type: {}, algos_tried: {}/{}",
            (int)matmul_status,
            __FILE__,
            __LINE__,
            device_id,
            m,
            n,
            k,
            (int)mode,
            (int)transA,
            (int)transB,
            lda,
            ldb,
            ldc,
            workspace_size,
            (int)to_cuda_lib_type_enum<FloatA>,
            (int)to_cuda_lib_type_enum<FloatB>,
            (int)to_cuda_lib_type_enum<FloatC>,
            algos_tried,
            returnedResults));
    }
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
void matmul(float* c,
            const float* a,
            const float* b,
            const float* bias,
            const float* scale_a,
            const float* scale_b,
            cublasLtHandle_t handle,
            std::byte* workspace,
            std::size_t workspace_size,
            int M,
            int N,
            int K,
            EMMTranspose mode,
            bool accumulate,
            cudaStream_t stream) {
    float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;
    matmul_cublaslt(c,
                    a,
                    b,
                    bias,
                    workspace,
                    workspace_size,
                    M,
                    N,
                    K,
                    stream,
                    handle,
                    scale_a,
                    scale_b,
                    mode,
                    alpha,
                    beta);
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
void matmul_strided_c(nv_bfloat16* c,
                      const nv_bfloat16* a,
                      const nv_bfloat16* b,
                      const nv_bfloat16* bias,
                      const float* scale_a,
                      const float* scale_b,
                      cublasLtHandle_t handle,
                      std::byte* workspace,
                      std::size_t workspace_size,
                      int M,
                      int N,
                      int K,
                      EMMTranspose mode,
                      bool accumulate,
                      int ldc,
                      cudaStream_t stream) {
    float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;
    matmul_strided_c(c,
                     a,
                     b,
                     bias,
                     scale_a,
                     scale_b,
                     handle,
                     workspace,
                     workspace_size,
                     M,
                     N,
                     K,
                     mode,
                     alpha,
                     beta,
                     ldc,
                     stream);
}

void matmul_strided_c(nv_bfloat16* c,
                      const nv_bfloat16* a,
                      const nv_bfloat16* b,
                      const nv_bfloat16* bias,
                      const float* scale_a,
                      const float* scale_b,
                      cublasLtHandle_t handle,
                      std::byte* workspace,
                      std::size_t workspace_size,
                      int M,
                      int N,
                      int K,
                      EMMTranspose mode,
                      float alpha,
                      float beta,
                      int ldc,
                      cudaStream_t stream) {
    matmul_cublaslt(c,
                    a,
                    b,
                    bias,
                    workspace,
                    workspace_size,
                    M,
                    N,
                    K,
                    stream,
                    handle,
                    scale_a,
                    scale_b,
                    mode,
                    alpha,
                    beta,
                    ldc);
}

void matmul_strided_c(float* c,
                      const float* a,
                      const float* b,
                      const float* bias,
                      const float* scale_a,
                      const float* scale_b,
                      cublasLtHandle_t handle,
                      std::byte* workspace,
                      std::size_t workspace_size,
                      int M,
                      int N,
                      int K,
                      EMMTranspose mode,
                      bool accumulate,
                      int ldc,
                      cudaStream_t stream) {
    float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;
    matmul_cublaslt(c,
                    a,
                    b,
                    bias,
                    workspace,
                    workspace_size,
                    M,
                    N,
                    K,
                    stream,
                    handle,
                    scale_a,
                    scale_b,
                    mode,
                    alpha,
                    beta,
                    ldc);
}

/**
 * @brief Performs matrix multiplication with explicit leading dimensions for A, B, and C.
 *
 * Allows reading from strided input matrices and writing to strided output.
 * Use lda/ldb/ldc = -1 for default leading dimensions.
 */
void matmul_strided(nv_bfloat16* c,
                    const nv_bfloat16* a,
                    const nv_bfloat16* b,
                    const nv_bfloat16* bias,
                    const float* scale_a,
                    const float* scale_b,
                    cublasLtHandle_t handle,
                    std::byte* workspace,
                    std::size_t workspace_size,
                    int M,
                    int N,
                    int K,
                    EMMTranspose mode,
                    bool accumulate,
                    int lda,
                    int ldb,
                    int ldc,
                    cudaStream_t stream) {
    float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;
    matmul_cublaslt(c,
                    a,
                    b,
                    bias,
                    workspace,
                    workspace_size,
                    M,
                    N,
                    K,
                    stream,
                    handle,
                    scale_a,
                    scale_b,
                    mode,
                    alpha,
                    beta,
                    ldc,
                    lda,
                    ldb);
}

void matmul_strided(float* c,
                    const float* a,
                    const float* b,
                    const float* bias,
                    const float* scale_a,
                    const float* scale_b,
                    cublasLtHandle_t handle,
                    std::byte* workspace,
                    std::size_t workspace_size,
                    int M,
                    int N,
                    int K,
                    EMMTranspose mode,
                    bool accumulate,
                    int lda,
                    int ldb,
                    int ldc,
                    cudaStream_t stream) {
    float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;
    matmul_cublaslt(c,
                    a,
                    b,
                    bias,
                    workspace,
                    workspace_size,
                    M,
                    N,
                    K,
                    stream,
                    handle,
                    scale_a,
                    scale_b,
                    mode,
                    alpha,
                    beta,
                    ldc,
                    lda,
                    ldb);
}

void matmul_strided(float* c,
                    const nv_bfloat16* a,
                    const nv_bfloat16* b,
                    const float* bias,
                    const float* scale_a,
                    const float* scale_b,
                    cublasLtHandle_t handle,
                    std::byte* workspace,
                    std::size_t workspace_size,
                    int M,
                    int N,
                    int K,
                    EMMTranspose mode,
                    bool accumulate,
                    int lda,
                    int ldb,
                    int ldc,
                    cudaStream_t stream) {
    float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;
    matmul_cublaslt(c,
                    a,
                    b,
                    bias,
                    workspace,
                    workspace_size,
                    M,
                    N,
                    K,
                    stream,
                    handle,
                    scale_a,
                    scale_b,
                    mode,
                    alpha,
                    beta,
                    ldc,
                    lda,
                    ldb);
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
void matmul(float* c,
            const nv_bfloat16* a,
            const nv_bfloat16* b,
            const float* bias,
            const float* scale_a,
            const float* scale_b,
            cublasLtHandle_t handle,
            std::byte* workspace,
            std::size_t workspace_size,
            int M,
            int N,
            int K,
            EMMTranspose mode,
            bool accumulate,
            cudaStream_t stream) {
    float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;
    matmul_cublaslt(c,
                    a,
                    b,
                    bias,
                    workspace,
                    workspace_size,
                    M,
                    N,
                    K,
                    stream,
                    handle,
                    scale_a,
                    scale_b,
                    mode,
                    alpha,
                    beta);
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
void matmul(float* c,
            const __nv_fp8_e4m3* a,
            const __nv_fp8_e4m3* b,
            const float* bias,
            const float* scale_a,
            const float* scale_b,
            cublasLtHandle_t handle,
            std::byte* workspace,
            std::size_t workspace_size,
            int M,
            int N,
            int K,
            EMMTranspose mode,
            bool accumulate,
            cudaStream_t stream) {
    float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;
    matmul_cublaslt(c,
                    a,
                    b,
                    bias,
                    workspace,
                    workspace_size,
                    M,
                    N,
                    K,
                    stream,
                    handle,
                    scale_a,
                    scale_b,
                    mode,
                    alpha,
                    beta);
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
void matmul(float* c,
            const __nv_fp8_e4m3* a,
            const __nv_fp8_e4m3* b,
            const nv_bfloat16* bias,
            const float* scale_a,
            const float* scale_b,
            cublasLtHandle_t handle,
            std::byte* workspace,
            std::size_t workspace_size,
            int M,
            int N,
            int K,
            EMMTranspose mode,
            bool accumulate,
            cudaStream_t stream) {
    float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;
    matmul_cublaslt(c,
                    a,
                    b,
                    bias,
                    workspace,
                    workspace_size,
                    M,
                    N,
                    K,
                    stream,
                    handle,
                    scale_a,
                    scale_b,
                    mode,
                    alpha,
                    beta);
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
void matmul(nv_bfloat16* c,
            const nv_bfloat16* a,
            const nv_bfloat16* b,
            const nv_bfloat16* bias,
            const float* scale_a,
            const float* scale_b,
            cublasLtHandle_t handle,
            std::byte* workspace,
            std::size_t workspace_size,
            int M,
            int N,
            int K,
            EMMTranspose mode,
            bool accumulate,
            cudaStream_t stream) {
    float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;
    matmul_cublaslt(c,
                    a,
                    b,
                    bias,
                    workspace,
                    workspace_size,
                    M,
                    N,
                    K,
                    stream,
                    handle,
                    scale_a,
                    scale_b,
                    mode,
                    alpha,
                    beta);
}

/**
 * @brief Performs BF16 matrix multiplication with explicit alpha/beta: C = alpha * (A @ B) + beta * C
 *
 * This overload allows fusing scaling into the matmul epilogue for better performance.
 * Useful for LoRA backward pass where scaling factors need to be applied.
 */
void matmul(nv_bfloat16* c,
            const nv_bfloat16* a,
            const nv_bfloat16* b,
            const nv_bfloat16* bias,
            const float* scale_a,
            const float* scale_b,
            cublasLtHandle_t handle,
            std::byte* workspace,
            std::size_t workspace_size,
            int M,
            int N,
            int K,
            EMMTranspose mode,
            float alpha,
            float beta,
            cudaStream_t stream) {
    matmul_cublaslt(c,
                    a,
                    b,
                    bias,
                    workspace,
                    workspace_size,
                    M,
                    N,
                    K,
                    stream,
                    handle,
                    scale_a,
                    scale_b,
                    mode,
                    alpha,
                    beta);
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
void matmul(nv_bfloat16* c,
            const __nv_fp8_e4m3* a,
            const __nv_fp8_e4m3* b,
            const nv_bfloat16* bias,
            const float* scale_a,
            const float* scale_b,
            cublasLtHandle_t handle,
            std::byte* workspace,
            std::size_t workspace_size,
            int M,
            int N,
            int K,
            EMMTranspose mode,
            bool accumulate,
            cudaStream_t stream) {
    float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;
    matmul_cublaslt(c,
                    a,
                    b,
                    bias,
                    workspace,
                    workspace_size,
                    M,
                    N,
                    K,
                    stream,
                    handle,
                    scale_a,
                    scale_b,
                    mode,
                    alpha,
                    beta);
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
void matmul(nv_bfloat16* c,
            const __nv_fp8_e4m3* a,
            const __nv_fp8_e5m2* b,
            const nv_bfloat16* bias,
            const float* scale_a,
            const float* scale_b,
            cublasLtHandle_t handle,
            std::byte* workspace,
            std::size_t workspace_size,
            int M,
            int N,
            int K,
            EMMTranspose mode,
            bool accumulate,
            cudaStream_t stream) {
    float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;
    matmul_cublaslt(c,
                    a,
                    b,
                    bias,
                    workspace,
                    workspace_size,
                    M,
                    N,
                    K,
                    stream,
                    handle,
                    scale_a,
                    scale_b,
                    mode,
                    alpha,
                    beta);
}

void matmul(nv_bfloat16* c,
            const __nv_fp8_e5m2* a,
            const __nv_fp8_e4m3* b,
            const nv_bfloat16* bias,
            const float* scale_a,
            const float* scale_b,
            cublasLtHandle_t handle,
            std::byte* workspace,
            std::size_t workspace_size,
            int M,
            int N,
            int K,
            EMMTranspose mode,
            bool accumulate,
            cudaStream_t stream) {
    float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;
    matmul_cublaslt(c,
                    a,
                    b,
                    bias,
                    workspace,
                    workspace_size,
                    M,
                    N,
                    K,
                    stream,
                    handle,
                    scale_a,
                    scale_b,
                    mode,
                    alpha,
                    beta);
}
