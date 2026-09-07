// sinfer::ops — encoder_attention launcher.
//
// Two strided-batched cuBLASLt products around one masked-softmax kernel. With
// three query heads over a sequence of at most a couple of thousand tokens the
// score matrix is tens of megabytes, so materialising it costs less than a
// hand-written flash kernel would cost to get right -- and the second product
// then reads it as an ordinary matrix.
//
// Orientation, which is the whole of the difficulty. Scores are held
// column-major as s[key, query]: the key axis is contiguous, so the softmax
// reduces over stride-1 data, and the same buffer feeds the second product with
// no transpose. Q, K and V arrive as separate contiguous tensors -- the form
// per-head QK norm and rope leave them in -- so the batch strides do the work:
// Q advances one head per batch element while the shared K and V advance none.
//
// The only translation unit that includes this op's kernel header.
// See docs/op-development.md §2.
#include "ops/launcher/encoder_attention.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/encoder_softmax.cuh"

#include <cublasLt.h>

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
        throw std::runtime_error(std::string("encoder_attention cuBLASLt: ") + what + " failed (" +
                                 std::to_string(static_cast<int>(status)) + ")");
    }
}

struct DeviceState {
    cublasLtHandle_t handle = nullptr;
    void* workspace         = nullptr;
    std::mutex mutex;
};

DeviceState& state_for_current_device() {
    static std::mutex registry_mutex;
    static std::unordered_map<int, std::unique_ptr<DeviceState>> registry;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    const std::lock_guard<std::mutex> lock(registry_mutex);
    auto& slot = registry[device];
    if (slot == nullptr) {
        auto state = std::make_unique<DeviceState>();
        check(cublasLtCreate(&state->handle), "create");
        CUDA_CHECK(cudaMalloc(&state->workspace, kWorkspaceBytes));
        slot = std::move(state);
    }
    return *slot;
}

/// One layout, optionally batched by a fixed element stride.
struct Layout {
    cublasLtMatrixLayout_t handle = nullptr;

    Layout(cudaDataType type, std::int64_t rows, std::int64_t cols, std::int64_t ld, int batch,
           std::int64_t stride) {
        check(cublasLtMatrixLayoutCreate(&handle, type, rows, cols, ld), "layout");
        check(cublasLtMatrixLayoutSetAttribute(handle, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch,
                                               sizeof(batch)),
              "batch count");
        check(cublasLtMatrixLayoutSetAttribute(
                  handle, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)),
              "batch stride");
    }
    ~Layout() { cublasLtMatrixLayoutDestroy(handle); }
    Layout(const Layout&)            = delete;
    Layout& operator=(const Layout&) = delete;
};

struct Desc {
    cublasLtMatmulDesc_t handle = nullptr;

    Desc(cublasOperation_t trans_a, cublasOperation_t trans_b) {
        check(cublasLtMatmulDescCreate(&handle, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc");
        check(cublasLtMatmulDescSetAttribute(handle, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a,
                                             sizeof(trans_a)),
              "transa");
        check(cublasLtMatmulDescSetAttribute(handle, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b,
                                             sizeof(trans_b)),
              "transb");
    }
    ~Desc() { cublasLtMatmulDescDestroy(handle); }
    Desc(const Desc&)            = delete;
    Desc& operator=(const Desc&) = delete;
};

void matmul(DeviceState& state, const Desc& desc, const void* a, const Layout& la, const void* b,
            const Layout& lb, void* c, const Layout& lc, cudaStream_t stream) {
    cublasLtMatmulPreference_t preference = nullptr;
    check(cublasLtMatmulPreferenceCreate(&preference), "preference");
    const std::size_t workspace_bytes = kWorkspaceBytes;
    check(cublasLtMatmulPreferenceSetAttribute(preference,
                                               CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &workspace_bytes, sizeof(workspace_bytes)),
          "workspace");
    cublasLtMatmulHeuristicResult_t result{};
    int found = 0;
    check(cublasLtMatmulAlgoGetHeuristic(state.handle, desc.handle, la.handle, lb.handle,
                                         lc.handle, lc.handle, preference, 1, &result, &found),
          "heuristic");
    cublasLtMatmulPreferenceDestroy(preference);
    if (found == 0) { throw std::runtime_error("encoder_attention: no cuBLASLt algorithm"); }

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    check(cublasLtMatmul(state.handle, desc.handle, &alpha, a, la.handle, b, lb.handle, &beta, c,
                         lc.handle, c, lc.handle, &result.algo, state.workspace, kWorkspaceBytes,
                         stream),
          "matmul");
}

} // namespace

void encoder_attention_prewarm() { (void)state_for_current_device(); }

void encoder_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                              std::int32_t window, float scale, Tensor& out, void* workspace,
                              cudaStream_t stream) {
    const auto tokens         = static_cast<std::int32_t>(q.ne[1]);
    const auto head_dim       = static_cast<std::int32_t>(k.ne[0]);
    const auto q_rows         = static_cast<std::int64_t>(q.ne[0]);
    const auto q_heads        = static_cast<std::int32_t>(q_rows / head_dim);
    const auto out_rows       = q_rows;
    const std::int64_t matrix = static_cast<std::int64_t>(tokens) * tokens;

    const auto* q_data = static_cast<const __nv_bfloat16*>(q.data);
    const auto* k_data = static_cast<const __nv_bfloat16*>(k.data);
    const auto* v_data = static_cast<const __nv_bfloat16*>(v.data);

    // Scores first (FP32), probabilities after them (BF16), in one scratch block.
    auto* scores = static_cast<float*>(workspace);
    auto* probs  = reinterpret_cast<__nv_bfloat16*>(scores + static_cast<std::size_t>(q_heads) *
                                                                 matrix);

    DeviceState& state = state_for_current_device();
    const std::lock_guard<std::mutex> lock(state.mutex);

    {
        // s[key, query] = K^T Q. A is the shared key head, so its batch stride is
        // zero; Q advances one head per batch element.
        const Desc desc(CUBLAS_OP_T, CUBLAS_OP_N);
        const Layout la(CUDA_R_16BF, head_dim, tokens, head_dim, q_heads, 0);
        const Layout lb(CUDA_R_16BF, head_dim, tokens, q_rows, q_heads, head_dim);
        const Layout lc(CUDA_R_32F, tokens, tokens, tokens, q_heads, matrix);
        matmul(state, desc, k_data, la, q_data, lb, scores, lc, stream);
    }

    const dim3 grid(static_cast<unsigned int>(tokens), static_cast<unsigned int>(q_heads));
    encoder_softmax_kernel<<<grid, kEncoderSoftmaxBlock, 0, stream>>>(scores, probs, tokens,
                                                                     window, scale);
    CUDA_CHECK(cudaGetLastError());

    {
        // out[dim, query] = V P. V is shared across heads; P advances one matrix.
        const Desc desc(CUBLAS_OP_N, CUBLAS_OP_N);
        const Layout la(CUDA_R_16BF, head_dim, tokens, head_dim, q_heads, 0);
        const Layout lb(CUDA_R_16BF, tokens, tokens, tokens, q_heads, matrix);
        const Layout lc(CUDA_R_16BF, head_dim, tokens, out_rows, q_heads, head_dim);
        matmul(state, desc, v_data, la, probs, lb, out.data, lc, stream);
    }
}

} // namespace sinfer::ops::detail
