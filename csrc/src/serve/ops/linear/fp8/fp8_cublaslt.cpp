#include "ops/linear/fp8/fp8_cublaslt.h"

#include "core/device.h"

#include <cublasLt.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace sinfer::ops::detail {
namespace {

constexpr std::size_t kWorkspaceBytes = std::size_t{32} << 20;

void check(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("fp8 cuBLASLt ") + what + ": status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

std::int32_t env_int(const char* name, std::int32_t fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') { return fallback; }
    const long parsed = std::strtol(raw, nullptr, 10);
    return parsed > 0 ? static_cast<std::int32_t>(parsed) : fallback;
}

struct PlanKey {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t tokens;
    bool operator==(const PlanKey& o) const {
        return rows == o.rows && k == o.k && tokens == o.tokens;
    }
};
struct PlanKeyHash {
    std::size_t operator()(const PlanKey& key) const {
        std::size_t h = static_cast<std::size_t>(key.rows);
        h = h * 1315423911u + static_cast<std::size_t>(key.k);
        return h * 1315423911u + static_cast<std::size_t>(key.tokens);
    }
};

struct Plan {
    cublasLtMatmulDesc_t op  = nullptr;
    cublasLtMatrixLayout_t a = nullptr;
    cublasLtMatrixLayout_t b = nullptr;
    cublasLtMatrixLayout_t c = nullptr;
    cublasLtMatmulAlgo_t algo{};
};

struct DeviceState {
    cublasLtHandle_t handle = nullptr;
    void* workspace         = nullptr;
    float* unit_scale       = nullptr; // scalar 1.0 for both operands
    std::unordered_map<PlanKey, Plan, PlanKeyHash> plans;
    std::mutex mutex;
};

DeviceState& state_for_current_device() {
    static std::mutex registry_mutex;
    static std::unordered_map<int, std::unique_ptr<DeviceState>> registry;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    const std::lock_guard<std::mutex> lock(registry_mutex);
    auto& slot = registry[device];
    if (!slot) {
        auto state = std::make_unique<DeviceState>();
        check(cublasLtCreate(&state->handle), "create");
        CUDA_CHECK(cudaMalloc(&state->workspace, kWorkspaceBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state->unit_scale), sizeof(float)));
        const float one = 1.0F;
        CUDA_CHECK(cudaMemcpy(state->unit_scale, &one, sizeof(one), cudaMemcpyHostToDevice));
        slot = std::move(state);
    }
    return *slot;
}

const Plan& plan_for(DeviceState& state, const PlanKey& key) {
    const auto found = state.plans.find(key);
    if (found != state.plans.end()) { return found->second; }
    Plan plan;
    check(cublasLtMatmulDescCreate(&plan.op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc");
    const cublasOperation_t trans_a = CUBLAS_OP_T;
    const cublasOperation_t trans_b = CUBLAS_OP_N;
    check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a,
                                         sizeof(trans_a)),
          "transa");
    check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b,
                                         sizeof(trans_b)),
          "transb");
    const void* a_scale = state.unit_scale;
    const void* b_scale = state.unit_scale;
    check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale,
                                         sizeof(a_scale)),
          "a scale pointer");
    check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale,
                                         sizeof(b_scale)),
          "b scale pointer");
    // A is the weight [rows x k] with k contiguous, read transposed; B the activation codes
    // [tokens x k] with k contiguous; the fp32 staging is token-major.
    check(cublasLtMatrixLayoutCreate(&plan.a, CUDA_R_8F_E4M3, key.k, key.rows, key.k), "a");
    check(cublasLtMatrixLayoutCreate(&plan.b, CUDA_R_8F_E4M3, key.k, key.tokens, key.k), "b");
    check(cublasLtMatrixLayoutCreate(&plan.c, CUDA_R_32F, key.rows, key.tokens, key.rows), "c");
    cublasLtMatmulPreference_t preference = nullptr;
    check(cublasLtMatmulPreferenceCreate(&preference), "preference");
    const std::size_t workspace_bytes = kWorkspaceBytes;
    check(cublasLtMatmulPreferenceSetAttribute(preference,
                                               CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &workspace_bytes, sizeof(workspace_bytes)),
          "workspace");
    cublasLtMatmulHeuristicResult_t result{};
    int found_count = 0;
    check(cublasLtMatmulAlgoGetHeuristic(state.handle, plan.op, plan.a, plan.b, plan.c, plan.c,
                                         preference, 1, &result, &found_count),
          "heuristic");
    cublasLtMatmulPreferenceDestroy(preference);
    if (found_count == 0) {
        throw std::runtime_error("fp8 cuBLASLt: no algorithm for rows=" +
                                 std::to_string(key.rows) + " k=" + std::to_string(key.k) +
                                 " tokens=" + std::to_string(key.tokens));
    }
    plan.algo = result.algo;
    return state.plans.emplace(key, plan).first->second;
}

} // namespace

void fp8_cublaslt_prewarm() { (void)state_for_current_device(); }

bool fp8_cublaslt_route(std::int32_t tokens) {
    static const bool enabled = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_FP8_CUBLASLT");
        return raw == nullptr || *raw == '\0' || std::strcmp(raw, "0") != 0;
    }();
    static const std::int32_t min_tokens =
        env_int("SUROGATE_SERVE_FP8_CUBLASLT_MIN_TOKENS", kFp8CublasLtDefaultMinTokens);
    return enabled && tokens >= min_tokens;
}

void fp8_cublaslt_gemm(const Weight& weight, std::int32_t row_begin, std::int32_t rows,
                       const std::uint8_t* activation_codes, float* staging, std::int32_t tokens,
                       cudaStream_t stream) {
    if (row_begin < 0 || rows <= 0 || (rows % 16) != 0 || row_begin + rows > weight.n ||
        weight.k <= 0 || (weight.k % 32) != 0 || tokens <= 0) {
        throw std::invalid_argument("fp8 cuBLASLt: invalid problem");
    }
    DeviceState& state = state_for_current_device();
    const std::lock_guard<std::mutex> lock(state.mutex);
    const Plan& plan = plan_for(state, PlanKey{rows, weight.k, tokens});
    const auto* weight_codes = static_cast<const std::uint8_t*>(weight.qdata) +
                               static_cast<std::size_t>(row_begin) * static_cast<std::size_t>(weight.k);
    const float alpha = 1.0F;
    const float beta  = 0.0F;
    check(cublasLtMatmul(state.handle, plan.op, &alpha, weight_codes, plan.a, activation_codes,
                         plan.b, &beta, staging, plan.c, staging, plan.c, &plan.algo,
                         state.workspace, kWorkspaceBytes, stream),
          "matmul");
}

} // namespace sinfer::ops::detail
