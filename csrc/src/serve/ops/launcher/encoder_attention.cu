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
#include "api/ops/encoder_attention.h"
#include <algorithm>

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
                              cudaStream_t stream, std::int32_t kv_heads, bool causal) {
    const auto tokens = static_cast<std::int32_t>(q.ne[1]);
    const auto head_dim = static_cast<std::int32_t>(k.ne[0] / kv_heads);
    const auto q_rows = static_cast<std::int64_t>(q.ne[0]);
    const auto kv_rows = static_cast<std::int64_t>(k.ne[0]);
    const auto q_heads = static_cast<std::int32_t>(q_rows / head_dim);
    const int group = q_heads / kv_heads;
    // Batch one query head from every KV group, then advance within each group.
    // Multi-query attention can batch all Q heads with a zero K/V batch stride.
    const int batches = kv_heads == 1 ? q_heads : kv_heads;
    const int passes = kv_heads == 1 ? 1 : group;
    const int q_stride = kv_heads == 1 ? head_dim : group * head_dim;
    const int kv_stride = kv_heads == 1 ? 0 : head_dim;
    const auto* q_data = static_cast<const __nv_bfloat16*>(q.data);
    const auto* k_data = static_cast<const __nv_bfloat16*>(k.data);
    const auto* v_data = static_cast<const __nv_bfloat16*>(v.data);
    auto* out_data = static_cast<__nv_bfloat16*>(out.data);
    auto* scores = static_cast<float*>(workspace);
    DeviceState& state = state_for_current_device();
    const std::lock_guard<std::mutex> lock(state.mutex);
    for (int first = 0; first < tokens; first += kEncoderAttentionQueryTile) {
        const int width = std::min(kEncoderAttentionQueryTile, tokens - first);
        const int key_first = window > 0 ? std::max(0, first - window + 1) : 0;
        const int key_end = causal ? first + width
            : window > 0 ? std::min(tokens, first + width - 1 + window) : tokens;
        const int keys = key_end - key_first;
        const std::int64_t matrix = static_cast<std::int64_t>(keys) * width;
        auto* probs = reinterpret_cast<__nv_bfloat16*>(scores + batches * matrix);
        const Desc score_desc(CUBLAS_OP_T, CUBLAS_OP_N);
        const Desc value_desc(CUBLAS_OP_N, CUBLAS_OP_N);
        const Layout key_layout(CUDA_R_16BF, head_dim, keys, kv_rows, batches, kv_stride);
        const Layout query_layout(CUDA_R_16BF, head_dim, width, q_rows, batches, q_stride);
        const Layout score_layout(CUDA_R_32F, keys, width, keys, batches, matrix);
        const Layout prob_layout(CUDA_R_16BF, keys, width, keys, batches, matrix);
        for (int pass = 0; pass < passes; ++pass) {
            const auto offset = static_cast<std::int64_t>(first) * q_rows + pass * head_dim;
            matmul(state, score_desc, k_data + key_first * kv_rows, key_layout,
                   q_data + offset, query_layout, scores, score_layout, stream);
            encoder_softmax_kernel<<<dim3(width, batches), kEncoderSoftmaxBlock, 0, stream>>>(
                scores, probs, keys, width, first, key_first, window, scale, causal);
            CUDA_CHECK(cudaGetLastError());
            matmul(state, value_desc, v_data + key_first * kv_rows, key_layout,
                   probs, prob_layout, out_data + offset, query_layout, stream);
        }
    }
}

} // namespace sinfer::ops::detail
