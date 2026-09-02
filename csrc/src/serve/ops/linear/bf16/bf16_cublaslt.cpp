#include "ops/linear/bf16/bf16_cublaslt.h"

#include "core/device.h"
#include "core/engine_context.h"

#include <cublasLt.h>
#include <cuda_bf16.h>

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
        throw std::runtime_error(std::string("bf16 cuBLASLt ") + what + " failed: status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

struct PlanKey {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t tokens;
    bool operator==(const PlanKey& other) const noexcept {
        return rows == other.rows && k == other.k && tokens == other.tokens;
    }
};

struct PlanKeyHash {
    std::size_t operator()(const PlanKey& key) const noexcept {
        return (static_cast<std::size_t>(key.rows) * 0x9E3779B1u) ^
               (static_cast<std::size_t>(key.k) * 0x85EBCA77u) ^
               (static_cast<std::size_t>(key.tokens) * 0xC2B2AE3Du);
    }
};

struct Plan {
    cublasLtMatmulDesc_t op    = nullptr;
    cublasLtMatrixLayout_t a   = nullptr;
    cublasLtMatrixLayout_t b   = nullptr;
    cublasLtMatrixLayout_t c   = nullptr;
    cublasLtMatmulAlgo_t algo{};
};

struct DeviceState {
    cublasLtHandle_t handle = nullptr;
    void* workspace         = nullptr;
    std::mutex mutex;
    std::unordered_map<PlanKey, Plan, PlanKeyHash> plans;
};

// Homed per engine (core/engine_context.h), not per process: two models in one process must
// not share a workspace. The mutex below only serializes the enqueue; the matmuls then run
// concurrently on the engines' own streams, and a stream-K algorithm keeps its cross-CTA
// barrier flags in the workspace, so one buffer for both wedged the device under multi-model
// load -- one engine's kernel spinning on a flag the other's had overwritten. Still keyed by
// device inside: pipeline stages on several devices each run this route.
struct PlaneState {
    std::mutex mutex;
    std::unordered_map<int, std::unique_ptr<DeviceState>> by_device;
    ~PlaneState() {
        // The device buffers go back. The cuBLASLt objects are left to the library: destroying
        // a handle during static teardown of the process-default context races the driver's
        // own shutdown, and an engine's handful of descriptors is not worth that.
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
        CUDA_CHECK(cudaMalloc(&state->workspace, kWorkspaceBytes));
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
    // A is the weight [rows x k] with k contiguous, read transposed; B is the activation
    // [tokens x k] with k contiguous; C is [tokens x rows] with rows contiguous, i.e. the
    // engine's [rows, tokens] tensor.
    check(cublasLtMatrixLayoutCreate(&plan.a, CUDA_R_16BF, key.k, key.rows, key.k), "a");
    check(cublasLtMatrixLayoutCreate(&plan.b, CUDA_R_16BF, key.k, key.tokens, key.k), "b");
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
    check(cublasLtMatmulAlgoGetHeuristic(state.handle, plan.op, plan.a, plan.b, plan.c, plan.c,
                                         preference, 1, &result, &found_count),
          "heuristic");
    cublasLtMatmulPreferenceDestroy(preference);
    if (found_count == 0) {
        throw std::runtime_error("bf16 cuBLASLt: no algorithm for rows=" +
                                 std::to_string(key.rows) + " k=" + std::to_string(key.k) +
                                 " tokens=" + std::to_string(key.tokens));
    }
    plan.algo = result.algo;
    return state.plans.emplace(key, plan).first->second;
}

void require_operands(const Weight& weight, const Tensor& x, const Tensor& out) {
    if (weight.qtype != QType::BF16_CTRL || weight.layout != QuantLayout::Contiguous ||
        weight.qdata == nullptr || weight.ndim != 2 || weight.n <= 0 || weight.k <= 0 ||
        (weight.k % 8) != 0 || (weight.n % 8) != 0 ||
        (reinterpret_cast<std::uintptr_t>(weight.qdata) & 15u) != 0) {
        throw std::invalid_argument("bf16 cuBLASLt: weight must be aligned contiguous BF16 [n,k] "
                                    "with n and k multiples of 8");
    }
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16 || !x.is_contiguous() ||
        !out.is_contiguous() || x.ne[0] != weight.k || out.ne[0] != weight.n ||
        x.ne[1] != out.ne[1] || x.ne[1] <= 0 || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[2] != 1 ||
        out.ne[3] != 1 || x.data == nullptr || out.data == nullptr ||
        (reinterpret_cast<std::uintptr_t>(x.data) & 15u) != 0 ||
        (reinterpret_cast<std::uintptr_t>(out.data) & 15u) != 0) {
        throw std::invalid_argument("bf16 cuBLASLt: x must be BF16 [k,T] and out BF16 [n,T], "
                                    "contiguous and 16-byte aligned");
    }
}

} // namespace

void bf16_cublaslt_prewarm() { (void)state_for_current_device(); }

void bf16_cublaslt_prepare(std::int32_t rows, std::int32_t k, std::int32_t tokens) {
    if (rows <= 0 || k <= 0 || tokens <= 0) {
        throw std::invalid_argument("bf16 cuBLASLt: invalid problem to prepare");
    }
    DeviceState& state = state_for_current_device();
    const std::lock_guard<std::mutex> lock(state.mutex);
    (void)plan_for(state, PlanKey{rows, k, tokens});
}

namespace {

void gemm_with_beta(const Weight& weight, const Tensor& x, Tensor& out, float beta,
                    cudaStream_t stream) {
    require_operands(weight, x, out);
    DeviceState& state = state_for_current_device();
    const std::lock_guard<std::mutex> lock(state.mutex);
    const Plan& plan  = plan_for(state, PlanKey{weight.n, weight.k, x.ne[1]});
    const float alpha = 1.0F;
    check(cublasLtMatmul(state.handle, plan.op, &alpha, weight.qdata, plan.a, x.data, plan.b,
                         &beta, out.data, plan.c, out.data, plan.c, &plan.algo, state.workspace,
                         kWorkspaceBytes, stream),
          "matmul");
}

} // namespace

void bf16_cublaslt_gemm(const Weight& weight, const Tensor& x, Tensor& out, cudaStream_t stream) {
    gemm_with_beta(weight, x, out, 0.0F, stream);
}

void bf16_cublaslt_gemm_accumulate(const Weight& weight, const Tensor& x, Tensor& out,
                                   cudaStream_t stream) {
    gemm_with_beta(weight, x, out, 1.0F, stream);
}

} // namespace sinfer::ops::detail
