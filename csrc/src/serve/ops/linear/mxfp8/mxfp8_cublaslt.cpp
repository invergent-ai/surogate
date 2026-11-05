#include "ops/linear/mxfp8/mxfp8.h"

#include "core/device.h"
#include "core/engine_context.h"

#include <cublasLt.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace sinfer::ops::detail {
namespace {

constexpr std::size_t kWorkspaceBytes = 32u << 20;

void check(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("mxfp8 cuBLASLt ") + what + " failed: status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

struct PlanKey {
    std::int32_t rows;
    std::int32_t k_padded;
    std::int32_t tokens;
    bool bias;
    bool accumulate;
    bool operator==(const PlanKey& other) const noexcept {
        return rows == other.rows && k_padded == other.k_padded && tokens == other.tokens &&
               bias == other.bias && accumulate == other.accumulate;
    }
};

struct PlanKeyHash {
    std::size_t operator()(const PlanKey& key) const noexcept {
        return (static_cast<std::size_t>(key.rows) * 0x9E3779B1u) ^
               (static_cast<std::size_t>(key.k_padded) * 0x85EBCA77u) ^
               (static_cast<std::size_t>(key.tokens) * 0xC2B2AE3Du) ^
               (static_cast<std::size_t>(key.bias) * 0x27D4EB2Fu) ^
               (static_cast<std::size_t>(key.accumulate) * 0x165667B1u);
    }
};

struct Plan {
    cublasLtMatmulDesc_t op  = nullptr;
    cublasLtMatrixLayout_t a = nullptr;
    cublasLtMatrixLayout_t c = nullptr;
    cublasLtMatrixLayout_t b = nullptr;
    cublasLtMatmulAlgo_t algo{};
    bool valid = false; // cuBLASLt offered an algorithm (a miss is cached so it is asked once)
};

struct DeviceState {
    cublasLtHandle_t handle = nullptr;
    void* workspace         = nullptr;
    std::mutex mutex;
    std::unordered_map<PlanKey, Plan, PlanKeyHash> plans;
};

// Per engine and per device, as the BF16 route's state is (bf16_cublaslt.cpp): a stream-K
// algorithm keeps cross-CTA flags in its workspace, so two engines must not share one.
struct PlaneState {
    std::mutex mutex;
    std::unordered_map<int, std::unique_ptr<DeviceState>> by_device;
    ~PlaneState() {
        for (auto& [device, state] : by_device) {
            (void)device;
            if (state->workspace != nullptr) { (void)cudaFree(state->workspace); }
        }
    }
};

DeviceState& state_for_current_device() {
    PlaneState& plane = engine_slot<PlaneState>();
    int device        = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    const std::lock_guard<std::mutex> lock(plane.mutex);
    auto& slot = plane.by_device[device];
    if (slot == nullptr) {
        auto state = std::make_unique<DeviceState>();
        check(cublasLtCreate(&state->handle), "create");
        if (const cudaError_t status = cudaMalloc(&state->workspace, kWorkspaceBytes);
            status != cudaSuccess) {
            (void)cublasLtDestroy(state->handle);
            throw std::runtime_error("mxfp8 cuBLASLt: could not allocate the workspace on device " +
                                     std::to_string(device) + ": " + cudaGetErrorString(status));
        }
        slot = std::move(state);
    }
    return *slot;
}

const Plan& plan_for(DeviceState& state, const PlanKey& key) {
    if (const auto found = state.plans.find(key); found != state.plans.end()) {
        return found->second;
    }
    Plan plan;
    check(cublasLtMatmulDescCreate(&plan.op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc");
    // A is the weight [rows x k] with k contiguous, read transposed; B is the activation
    // [tokens x k] with k contiguous; D is [tokens x rows] with rows contiguous, i.e. the
    // engine's [rows, tokens] tensor. FP8 operands must be "TN", which this is.
    const cublasOperation_t trans_a = CUBLAS_OP_T;
    const cublasOperation_t trans_b = CUBLAS_OP_N;
    check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a,
                                         sizeof(trans_a)),
          "transa");
    check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b,
                                         sizeof(trans_b)),
          "transb");
    const cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode,
                                         sizeof(mode)),
          "a scale mode");
    check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode,
                                         sizeof(mode)),
          "b scale mode");
    // The heuristic refuses a block-scaled problem whose scale pointers are null
    // (CUBLAS_STATUS_INVALID_VALUE); each call sets the real ones, so any aligned device
    // address serves here.
    const void* placeholder = state.workspace;
    check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                         &placeholder, sizeof(placeholder)),
          "a scale placeholder");
    check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                         &placeholder, sizeof(placeholder)),
          "b scale placeholder");
    if (key.bias) {
        const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                             sizeof(epilogue)),
              "epilogue");
        const cudaDataType_t bias_type = CUDA_R_16BF;
        check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                             &bias_type, sizeof(bias_type)),
              "bias type");
    }
    check(cublasLtMatrixLayoutCreate(&plan.a, CUDA_R_8F_E4M3, key.k_padded, key.rows, key.k_padded),
          "a");
    check(cublasLtMatrixLayoutCreate(&plan.b, CUDA_R_8F_E4M3, key.k_padded, key.tokens,
                                     key.k_padded),
          "b");
    check(cublasLtMatrixLayoutCreate(&plan.c, CUDA_R_16BF, key.rows, key.tokens, key.rows), "c");
    cublasLtMatmulPreference_t preference = nullptr;
    check(cublasLtMatmulPreferenceCreate(&preference), "preference");
    const std::size_t workspace_bytes = kWorkspaceBytes;
    check(cublasLtMatmulPreferenceSetAttribute(preference,
                                               CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &workspace_bytes, sizeof(workspace_bytes)),
          "workspace");
    cublasLtMatmulHeuristicResult_t result{};
    int found_count = 0;
    // A problem the library cannot run reports either no result or NOT_SUPPORTED; both are a
    // miss for the caller, which then keeps its BF16 route.
    const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        state.handle, plan.op, plan.a, plan.b, plan.c, plan.c, preference, 1, &result, &found_count);
    cublasLtMatmulPreferenceDestroy(preference);
    if (status != CUBLAS_STATUS_SUCCESS && status != CUBLAS_STATUS_NOT_SUPPORTED) {
        check(status, "heuristic");
    }
    plan.valid = status == CUBLAS_STATUS_SUCCESS && found_count > 0;
    if (plan.valid) { plan.algo = result.algo; }
    return state.plans.emplace(key, plan).first->second;
}

bool aligned16(const void* pointer) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & 15U) == 0;
}

} // namespace

void mxfp8_cublaslt_prewarm() { (void)state_for_current_device(); }

bool mxfp8_gemm_available(std::int32_t rows, std::int32_t k, std::int32_t tokens, bool bias,
                          bool accumulate) {
    if (rows <= 0 || k <= 0 || tokens <= 0 || (rows % 8) != 0) {
        return false;
    }
    DeviceState& state = state_for_current_device();
    const std::lock_guard<std::mutex> lock(state.mutex);
    return plan_for(state, PlanKey{rows, mxfp8_padded_k(k), tokens, bias, accumulate}).valid;
}

bool mxfp8_gemm(const Mxfp8Matrix& weight, const Mxfp8Matrix& x, const void* bias, bool accumulate,
                void* out, cudaStream_t stream) {
    if (weight.k != x.k || weight.rows <= 0 || x.rows <= 0 || !aligned16(weight.codes) ||
        !aligned16(x.codes) || !aligned16(weight.scales) || !aligned16(x.scales) ||
        !aligned16(out) || (bias != nullptr && !aligned16(bias))) {
        throw std::invalid_argument("mxfp8_gemm: operands must agree on k and be 16-byte aligned");
    }
    if ((weight.rows % 8) != 0) { return false; }
    DeviceState& state = state_for_current_device();
    const std::lock_guard<std::mutex> lock(state.mutex);
    const Plan& plan = plan_for(
        state, PlanKey{weight.rows, mxfp8_padded_k(weight.k), x.rows, bias != nullptr, accumulate});
    if (!plan.valid) { return false; }
    // The scale (and bias) pointers are per-call operands stored on the shared descriptor; the
    // lock keeps them from changing between these sets and the enqueue that reads them.
    check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                         &weight.scales, sizeof(weight.scales)),
          "a scales");
    check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &x.scales,
                                         sizeof(x.scales)),
          "b scales");
    if (bias != nullptr) {
        check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias,
                                             sizeof(bias)),
              "bias pointer");
    }
    const float alpha = 1.0F;
    const float beta  = accumulate ? 1.0F : 0.0F;
    check(cublasLtMatmul(state.handle, plan.op, &alpha, weight.codes, plan.a, x.codes, plan.b,
                         &beta, out, plan.c, out, plan.c, &plan.algo, state.workspace,
                         kWorkspaceBytes, stream),
          "matmul");
    return true;
}

} // namespace sinfer::ops::detail
