// Inkling-2-Mini (lane e1): BF16 x BF16 -> FP32 GEMM through cuBLASLt, for the two products the
// reference keeps in fp32 -- the loop controllers' gain / forget projection (`torch.bmm(...,
// out_dtype=float32)`) and the LM head (head_dtype float32). Same plan-cache pattern as
// ops/linear/bf16/bf16_cublaslt.cpp, with an FP32 output layout.

#include "api/ops/inkling_mix.h"

#include "core/device.h"
#include "core/engine_context.h"

#include <cublasLt.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace sinfer::ops {
namespace {

constexpr std::size_t kWorkspaceBytes = 32u << 20;

void check(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("inkling fp32-out cuBLASLt ") + what +
                                 " failed: status " + std::to_string(static_cast<int>(status)));
    }
}

struct PlanKey {
    std::int32_t n, k, tokens, ldc;
    bool operator==(const PlanKey& o) const noexcept {
        return n == o.n && k == o.k && tokens == o.tokens && ldc == o.ldc;
    }
};

struct PlanKeyHash {
    std::size_t operator()(const PlanKey& key) const noexcept {
        return (static_cast<std::size_t>(key.n) * 0x9E3779B1u) ^
               (static_cast<std::size_t>(key.k) * 0x85EBCA77u) ^
               (static_cast<std::size_t>(key.tokens) * 0xC2B2AE3Du) ^
               (static_cast<std::size_t>(key.ldc) * 0x27D4EB2Fu);
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
    std::mutex mutex;
    std::unordered_map<PlanKey, Plan, PlanKeyHash> plans;
};

struct InklingF32GemmState {
    std::mutex mutex;
    std::unordered_map<int, std::unique_ptr<DeviceState>> by_device;
    ~InklingF32GemmState() {
        for (auto& [device, state] : by_device) {
            (void)device;
            if (state->workspace != nullptr) { (void)cudaFree(state->workspace); }
        }
    }
};

DeviceState& state_for_current_device() {
    InklingF32GemmState& plane = engine_slot<InklingF32GemmState>();
    int device                 = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    const std::lock_guard<std::mutex> lock(plane.mutex);
    auto& slot = plane.by_device[device];
    if (slot == nullptr) {
        auto state = std::make_unique<DeviceState>();
        check(cublasLtCreate(&state->handle), "create");
        if (const cudaError_t status = cudaMalloc(&state->workspace, kWorkspaceBytes);
            status != cudaSuccess) {
            (void)cublasLtDestroy(state->handle);
            throw std::runtime_error(
                "inkling fp32-out cuBLASLt: no workspace on device " + std::to_string(device) +
                " (a graph capture before inkling_gemm_f32_prepare?): " + cudaGetErrorString(status));
        }
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
    check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)),
          "transa");
    check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)),
          "transb");
    // A = the weight [n x k] (k contiguous) read transposed; B = the activation [tokens x k] (k
    // contiguous); C = [tokens x n] with n contiguous and ldc per column: the engine's [n, tokens].
    check(cublasLtMatrixLayoutCreate(&plan.a, CUDA_R_16BF, key.k, key.n, key.k), "a");
    check(cublasLtMatrixLayoutCreate(&plan.b, CUDA_R_16BF, key.k, key.tokens, key.k), "b");
    check(cublasLtMatrixLayoutCreate(&plan.c, CUDA_R_32F, key.n, key.tokens, key.ldc), "c");
    cublasLtMatmulPreference_t preference = nullptr;
    check(cublasLtMatmulPreferenceCreate(&preference), "preference");
    const std::size_t workspace_bytes = kWorkspaceBytes;
    check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &workspace_bytes, sizeof(workspace_bytes)),
          "workspace");
    cublasLtMatmulHeuristicResult_t result{};
    int found_count = 0;
    check(cublasLtMatmulAlgoGetHeuristic(state.handle, plan.op, plan.a, plan.b, plan.c, plan.c,
                                         preference, 1, &result, &found_count),
          "heuristic");
    cublasLtMatmulPreferenceDestroy(preference);
    if (found_count == 0) {
        throw std::runtime_error("inkling fp32-out cuBLASLt: no algorithm for n=" + std::to_string(key.n) +
                                 " k=" + std::to_string(key.k) + " tokens=" + std::to_string(key.tokens));
    }
    plan.algo = result.algo;
    return state.plans.emplace(key, plan).first->second;
}

bool aligned(const void* pointer) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & 15u) == 0;
}

} // namespace

void inkling_gemm_f32_prepare(std::int32_t n, std::int32_t k, std::int32_t tokens, std::int32_t ldc) {
    if (n <= 0 || k <= 0 || tokens <= 0 || ldc < n) {
        throw std::invalid_argument("inkling_gemm_f32_prepare: invalid problem");
    }
    DeviceState& state = state_for_current_device();
    const std::lock_guard<std::mutex> lock(state.mutex);
    (void)plan_for(state, PlanKey{n, k, tokens, ldc});
}

void inkling_gemm_bf16_f32(const void* weight, std::int32_t n, std::int32_t k, const void* x,
                           std::int32_t tokens, float* out, std::int32_t ldc, cudaStream_t stream) {
    if (n <= 0 || k <= 0 || tokens <= 0 || ldc < n || (n % 8) != 0 || (k % 8) != 0 ||
        (ldc % 8) != 0 || !aligned(weight) || !aligned(x) || !aligned(out)) {
        throw std::invalid_argument("inkling_gemm_bf16_f32: n, k and ldc must be positive multiples "
                                    "of 8 with ldc >= n and every operand 16-byte aligned");
    }
    DeviceState& state = state_for_current_device();
    const std::lock_guard<std::mutex> lock(state.mutex);
    const Plan& plan  = plan_for(state, PlanKey{n, k, tokens, ldc});
    const float alpha = 1.0F;
    const float beta  = 0.0F;
    check(cublasLtMatmul(state.handle, plan.op, &alpha, weight, plan.a, x, plan.b, &beta, out, plan.c,
                         out, plan.c, &plan.algo, state.workspace, kWorkspaceBytes, stream),
          "matmul");
}

} // namespace sinfer::ops
