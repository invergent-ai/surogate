#include "ops/linear/ggml/ggml_dispatch.h"

#include "core/device.h"
#include "core/engine_context.h"
#include "ops/linear/ggml/ggml_linear.h"
#include "ops/linear/ggml/ggml_q8_1.h"

#include <cuda_bf16.h>

#include <memory>
#include <optional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace sinfer::ops::detail::ggml {

bool is_ggml_qtype(QType qtype) noexcept {
    switch (qtype) {
    case QType::Q2_K:
    case QType::Q3_K:
    case QType::Q4_K:
    case QType::Q5_K:
    case QType::Q6_K:
    case QType::Q8_0:
    case QType::Q4_1:
    case QType::Q5_1:
        return true;
    default:
        return false;
    }
}

GgmlType ggml_type_for(QType qtype) {
    switch (qtype) {
    case QType::Q2_K: return GgmlType::Q2_K;
    case QType::Q3_K: return GgmlType::Q3_K;
    case QType::Q4_K: return GgmlType::Q4_K;
    case QType::Q5_K: return GgmlType::Q5_K;
    case QType::Q6_K: return GgmlType::Q6_K;
    case QType::Q8_0: return GgmlType::Q8_0;
    case QType::Q4_1: return GgmlType::Q4_1;
    case QType::Q5_1: return GgmlType::Q5_1;
    default: break;
    }
    throw std::invalid_argument("ggml: weight qtype is not a GGML block format");
}

void require_ggml_weight(const Weight& w, const char* op) {
    const std::int32_t values = is_ggml_qtype(w.qtype) ? block_values(ggml_type_for(w.qtype)) : 0;
    if (!is_ggml_qtype(w.qtype) || w.layout != QuantLayout::GgmlBlocks || w.qdata == nullptr ||
        w.ndim != 2 || w.n <= 0 || w.k <= 0 || (w.k % values) != 0 ||
        w.group != static_cast<std::int32_t>(values) || w.padded_shape[0] != w.n ||
        w.padded_shape[1] != w.k) {
        throw std::invalid_argument(std::string(op) +
                                    ": weight must be a GGML block format in GgmlBlocks layout, "
                                    "[n, k] with k a whole number of blocks");
    }
    const std::size_t expected = static_cast<std::size_t>(w.n) * (w.k / values) *
                                 static_cast<std::size_t>(block_bytes(ggml_type_for(w.qtype)));
    if (w.payload_bytes < expected) {
        throw std::invalid_argument(std::string(op) + ": GGML weight payload is too small");
    }
}

namespace {

struct DeviceScratch {
    void* data        = nullptr;
    std::size_t bytes = 0;
};

struct ScratchState {
    std::mutex mutex;
    std::unordered_map<int, DeviceScratch> by_device;
    std::size_t retired_bytes = 0; // outgrown buffers kept alive for the graphs holding them
    ~ScratchState() {
        for (auto& [device, scratch] : by_device) {
            (void)device;
            if (scratch.data != nullptr) { (void)cudaFree(scratch.data); }
        }
    }
};

bool stream_is_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return status != cudaStreamCaptureStatusNone;
}

std::size_t round_up(std::size_t value, std::size_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

} // namespace

void* scratch_for(std::size_t bytes, cudaStream_t stream) {
    ScratchState& state = engine_slot<ScratchState>();
    int device          = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    const std::lock_guard<std::mutex> lock(state.mutex);
    DeviceScratch& scratch = state.by_device[device];
    if (scratch.bytes >= bytes) { return scratch.data; }
    if (stream_is_capturing(stream)) {
        throw std::runtime_error("ggml linear: activation scratch of " + std::to_string(bytes) +
                                 " bytes was first needed inside a graph capture; the warm-up "
                                 "did not reach this width");
    }
    // A captured graph keeps the pointer it was recorded with, so an outgrown buffer is never
    // freed: it stays valid for every graph that captured it (each one sized for its own
    // width), and the new, larger buffer serves the calls from here on. The retired buffers
    // are a few MiB at most and are counted in scratch_bytes().
    const std::size_t grown = round_up(bytes, std::size_t{1} << 20);
    void* data              = nullptr;
    CUDA_CHECK(cudaMalloc(&data, grown));
    state.retired_bytes += scratch.bytes;
    scratch.data  = data;
    scratch.bytes = grown;
    return data;
}

std::size_t scratch_bytes() noexcept {
    ScratchState& state = engine_slot<ScratchState>();
    const std::lock_guard<std::mutex> lock(state.mutex);
    std::size_t total = state.retired_bytes;
    for (const auto& [device, scratch] : state.by_device) {
        (void)device;
        total += scratch.bytes;
    }
    return total;
}

namespace {

struct Scratch {
    void* data        = nullptr;
    std::size_t bytes = 0;
};

Scratch scratch(WorkspaceArena* workspace, std::size_t bytes, cudaStream_t stream) {
    if (workspace != nullptr) {
        const DeviceSpan span = workspace->alloc_bytes(bytes, 256);
        return {span.data, span.bytes};
    }
    return {scratch_for(bytes, stream), bytes};
}

void require_x_out(const Tensor& x, std::int32_t k, const Tensor& out, std::int32_t n, const char* op) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16 || !x.is_contiguous() ||
        !out.is_contiguous() || x.ne[0] != k || out.ne[0] != n || x.ne[1] != out.ne[1] ||
        x.ne[1] <= 0 || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[2] != 1 || out.ne[3] != 1 ||
        x.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": x must be BF16 [k, T] and out BF16 [n, T], contiguous");
    }
}

} // namespace

void ggml_linear(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena* workspace,
                 cudaStream_t stream) {
    require_ggml_weight(w, "ggml linear");
    require_x_out(x, w.k, out, w.n, "ggml linear");
    const std::int32_t tokens = x.ne[1];
    const std::size_t bytes   = linear_workspace_bytes(w.n, w.k, tokens);
    auto scope                = workspace != nullptr ? std::optional(workspace->scope()) : std::nullopt;
    const Scratch s           = scratch(workspace, bytes, stream);
    linear_launch(ggml_type_for(w.qtype), w.qdata, w.n, w.k, static_cast<const __nv_bfloat16*>(x.data),
                  tokens, static_cast<__nv_bfloat16*>(out.data), s.data, s.bytes, stream);
}

void ggml_linear_add(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena* workspace,
                     cudaStream_t stream) {
    require_ggml_weight(w, "ggml linear_add");
    require_x_out(x, w.k, residual, w.n, "ggml linear_add");
    const std::int32_t tokens = x.ne[1];
    const std::size_t bytes   = linear_workspace_bytes(w.n, w.k, tokens);
    auto scope                = workspace != nullptr ? std::optional(workspace->scope()) : std::nullopt;
    const Scratch s           = scratch(workspace, bytes, stream);
    linear_add_launch(ggml_type_for(w.qtype), w.qdata, w.n, w.k, static_cast<const __nv_bfloat16*>(x.data),
                      tokens, static_cast<__nv_bfloat16*>(residual.data), s.data, s.bytes, stream);
}

void ggml_project_rows(const Tensor& x, const Weight& w, std::int32_t row_begin, Tensor& out,
                       WorkspaceArena* workspace, cudaStream_t stream) {
    require_ggml_weight(w, "ggml project_rows");
    const std::int32_t rows = out.ne[0];
    if (row_begin < 0 || rows <= 0 || row_begin + rows > w.n) {
        throw std::invalid_argument("ggml project_rows: row range outside the parent");
    }
    require_x_out(x, w.k, out, rows, "ggml project_rows");
    const GgmlType type       = ggml_type_for(w.qtype);
    const std::size_t row_bytes =
        static_cast<std::size_t>(w.k / block_values(type)) * block_bytes(type);
    const auto* blocks = static_cast<const std::byte*>(w.qdata) + static_cast<std::size_t>(row_begin) * row_bytes;
    const std::int32_t tokens = x.ne[1];
    // the row range is the matrix this call actually multiplies, so it sizes the tile too
    const std::size_t bytes   = linear_workspace_bytes(rows, w.k, tokens);
    auto scope                = workspace != nullptr ? std::optional(workspace->scope()) : std::nullopt;
    const Scratch s           = scratch(workspace, bytes, stream);
    linear_launch(type, blocks, rows, w.k, static_cast<const __nv_bfloat16*>(x.data), tokens,
                  static_cast<__nv_bfloat16*>(out.data), s.data, s.bytes, stream);
}

std::size_t ggml_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                std::int32_t input_rows,
                                                std::int32_t max_tokens) {
    if (output_rows <= 0 || input_rows <= 0 || (input_rows % QK8_0) != 0 || max_tokens <= 0) {
        throw std::invalid_argument("ggml linear workspace: k must be a whole number of blocks");
    }
    return linear_workspace_bytes(output_rows, input_rows, max_tokens) + 256;
}

} // namespace sinfer::ops::detail::ggml
