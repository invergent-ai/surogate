#include "ops/linear/ggml/ggml_dispatch.h"

#include "core/device.h"
#include "core/engine_context.h"
#include "ops/linear/ggml/ggml_linear.h"
#include "ops/linear/ggml/ggml_q8_1.h"
#include "ops/linear/ggml/ggml_swiglu.h"

#include <cuda_bf16.h>

#include <memory>
#include <optional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace sinfer::ops::detail::ggml {

// Both maps are written over the one list, so a format added there is admitted here by
// construction; the two enums spell every name the same way.
bool is_ggml_qtype(QType qtype) noexcept {
    switch (qtype) {
#define SINFER_GGML_IS_CASE(NAME) case QType::NAME:
        SINFER_GGML_FOR_EACH_TYPE(SINFER_GGML_IS_CASE)
#undef SINFER_GGML_IS_CASE
        return true;
    default:
        return false;
    }
}

GgmlType ggml_type_for(QType qtype) {
    switch (qtype) {
#define SINFER_GGML_MAP_CASE(NAME) case QType::NAME: return GgmlType::NAME;
        SINFER_GGML_FOR_EACH_TYPE(SINFER_GGML_MAP_CASE)
#undef SINFER_GGML_MAP_CASE
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
    std::size_t expected = 0;
    if (w.segment_count == 0) {
        expected = static_cast<std::size_t>(w.n) * (w.k / values) *
                   static_cast<std::size_t>(block_bytes(ggml_type_for(w.qtype)));
    } else {
        std::int32_t row = 0;
        for (std::int32_t i = 0; i < w.segment_count; ++i) {
            const WeightSegment& s = w.segments[i];
            if (!is_ggml_qtype(s.qtype) || s.qdata == nullptr || s.row_begin != row || s.rows <= 0 ||
                (w.k % block_values(ggml_type_for(s.qtype))) != 0) {
                throw std::invalid_argument(std::string(op) + ": GGML weight segments must be "
                                            "consecutive typed row runs at the weight's k");
            }
            expected += static_cast<std::size_t>(s.rows) * (w.k / block_values(ggml_type_for(s.qtype))) *
                        static_cast<std::size_t>(block_bytes(ggml_type_for(s.qtype)));
            row += s.rows;
        }
        if (row != w.n || w.segments[0].qtype != w.qtype) {
            throw std::invalid_argument(std::string(op) + ": GGML weight segments must cover the rows");
        }
    }
    if (w.payload_bytes < expected) {
        throw std::invalid_argument(std::string(op) + ": GGML weight payload is too small");
    }
}

namespace {

/// A parent whose rows come in more than one format is only ever projected by rows; the
/// whole-parent routes have one type to hand the kernel.
void require_homogeneous(const Weight& w, const char* op) {
    if (w.segment_count != 0) {
        throw std::invalid_argument(std::string(op) + ": a segmented parent is projected by rows");
    }
}

} // namespace

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

/// The activation a launch reads, and the scratch for its int8 planes. A weight read from a
/// GGUF whose columns the file keeps in another order (the GDN output projection of a
/// reordered geometry, whose columns follow the V heads) carries a group map instead of a
/// rearranged copy, and the map is applied to the activation: its 32-row groups are moved to
/// where the stored columns expect them, into a buffer that outlives the launch -- the
/// engine-slot scratch, grown during warm-up, or the arena when it also holds the planes.
struct Input {
    const __nv_bfloat16* x = nullptr;
    Scratch planes;
};

Input input_for(const Tensor& x, const Weight& w, WorkspaceArena* workspace, std::size_t plane_bytes,
                cudaStream_t stream) {
    Input in{static_cast<const __nv_bfloat16*>(x.data), {}};
    if (w.input_group_map == nullptr) {
        in.planes = scratch(workspace, plane_bytes, stream);
        return in;
    }
    const std::size_t x_bytes = (static_cast<std::size_t>(w.k) * x.ne[1] * sizeof(__nv_bfloat16) + 255) & ~std::size_t{255};
    __nv_bfloat16* permuted   = nullptr;
    if (workspace != nullptr) {
        permuted  = static_cast<__nv_bfloat16*>(scratch_for(x_bytes, stream));
        in.planes = scratch(workspace, plane_bytes, stream);
    } else {
        auto* base = static_cast<std::byte*>(scratch_for(x_bytes + plane_bytes, stream));
        permuted   = reinterpret_cast<__nv_bfloat16*>(base);
        in.planes  = {base + x_bytes, plane_bytes};
    }
    permute_column_groups_launch(static_cast<const __nv_bfloat16*>(x.data), w.k, x.ne[1],
                                 w.input_group_map, permuted, stream);
    in.x = permuted;
    return in;
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
    require_homogeneous(w, "ggml linear");
    require_x_out(x, w.k, out, w.n, "ggml linear");
    const std::int32_t tokens = x.ne[1];
    const std::size_t bytes   = linear_workspace_bytes(w.n, w.k, tokens);
    auto scope                = workspace != nullptr ? std::optional(workspace->scope()) : std::nullopt;
    const Input in            = input_for(x, w, workspace, bytes, stream);
    linear_launch(ggml_type_for(w.qtype), w.qdata, w.n, w.k, in.x, tokens,
                  static_cast<__nv_bfloat16*>(out.data), in.planes.data, in.planes.bytes, stream);
}

void ggml_linear_add(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena* workspace,
                     cudaStream_t stream) {
    require_ggml_weight(w, "ggml linear_add");
    require_homogeneous(w, "ggml linear_add");
    require_x_out(x, w.k, residual, w.n, "ggml linear_add");
    const std::int32_t tokens = x.ne[1];
    const std::size_t bytes   = linear_workspace_bytes(w.n, w.k, tokens);
    auto scope                = workspace != nullptr ? std::optional(workspace->scope()) : std::nullopt;
    const Input in            = input_for(x, w, workspace, bytes, stream);
    linear_add_launch(ggml_type_for(w.qtype), w.qdata, w.n, w.k, in.x, tokens,
                      static_cast<__nv_bfloat16*>(residual.data), in.planes.data, in.planes.bytes, stream);
}

Weight ggml_weight_rows(const Weight& w, std::int32_t row_begin, std::int32_t rows) {
    require_ggml_weight(w, "ggml weight_rows");
    if (row_begin < 0 || rows <= 0 || row_begin + rows > w.n) {
        throw std::invalid_argument("ggml weight_rows: row range outside the parent");
    }
    // The range's format and bytes: the parent's, or those of the segment holding it. A
    // component of a fused parent is one source tensor and so one format, which is why a
    // range never straddles two segments; a caller asking for more is asking the wrong op.
    Weight out             = w;
    std::int32_t local_row = row_begin;
    const void* base       = w.qdata;
    if (w.segment_count != 0) {
        const WeightSegment* segment = nullptr;
        for (std::int32_t i = 0; i < w.segment_count; ++i) {
            const WeightSegment& s = w.segments[i];
            if (row_begin >= s.row_begin && row_begin < s.row_begin + s.rows) { segment = &s; break; }
        }
        if (segment == nullptr || row_begin + rows > segment->row_begin + segment->rows) {
            throw std::invalid_argument("ggml weight_rows: the row range straddles typed segments");
        }
        out.qtype     = segment->qtype;
        base          = segment->qdata;
        local_row     = row_begin - segment->row_begin;
        out.segments  = nullptr;
        out.segment_count = 0;
    }
    const GgmlType type = ggml_type_for(out.qtype);
    const std::size_t row_bytes =
        static_cast<std::size_t>(w.k / block_values(type)) * block_bytes(type);
    out.qdata           = static_cast<const std::byte*>(base) + static_cast<std::size_t>(local_row) * row_bytes;
    out.payload         = out.qdata;
    out.payload_bytes   = static_cast<std::uint64_t>(rows) * row_bytes;
    out.group           = block_values(type);
    out.group_size      = static_cast<std::uint32_t>(out.group);
    out.n               = rows;
    out.shape[0]        = rows;
    out.padded_shape[0] = rows;
    return out;
}

void ggml_project_rows(const Tensor& x, const Weight& w, std::int32_t row_begin, Tensor& out,
                       WorkspaceArena* workspace, cudaStream_t stream) {
    const std::int32_t rows = out.ne[0];
    if (row_begin < 0 || rows <= 0 || row_begin + rows > w.n) {
        throw std::invalid_argument("ggml project_rows: row range outside the parent");
    }
    require_x_out(x, w.k, out, rows, "ggml project_rows");
    const Weight view         = ggml_weight_rows(w, row_begin, rows);
    const GgmlType type       = ggml_type_for(view.qtype);
    const std::int32_t tokens = x.ne[1];
    // the row range is the matrix this call actually multiplies, so it sizes the tile too
    const std::size_t bytes   = linear_workspace_bytes(rows, w.k, tokens);
    auto scope                = workspace != nullptr ? std::optional(workspace->scope()) : std::nullopt;
    const Input in            = input_for(x, w, workspace, bytes, stream);
    linear_launch(type, view.qdata, rows, w.k, in.x, tokens, static_cast<__nv_bfloat16*>(out.data),
                  in.planes.data, in.planes.bytes, stream);
}

bool ggml_swiglu_decode(const Tensor& x, const Weight& w, Tensor& out,
                       WorkspaceArena& workspace, cudaStream_t stream) {
    if (!is_ggml_qtype(w.qtype) || x.ne[1] <= 0 || x.ne[1] > 8) { return false; }
    require_ggml_weight(w, "ggml swiglu decode");
    if (w.n % 2 != 0) { throw std::invalid_argument("ggml swiglu decode: odd gate/up row count"); }
    const int rows = w.n / 2;
    const Weight gate = ggml_weight_rows(w, 0, rows);
    const Weight up = ggml_weight_rows(w, rows, rows);
    const auto gate_type = ggml_type_for(gate.qtype), up_type = ggml_type_for(up.qtype);
    if (!swiglu_decode_admits(gate_type, up_type, rows, w.k, x.ne[1])) { return false; }
    require_x_out(x, w.k, out, rows, "ggml swiglu decode");
    const std::size_t bytes = linear_workspace_bytes(rows, w.k, x.ne[1]);
    auto scope = workspace.scope();
    const Input in = input_for(x, w, &workspace, bytes, stream);
    swiglu_decode_launch(gate_type, gate.qdata, up_type, up.qdata, rows, w.k, in.x, x.ne[1],
                         static_cast<__nv_bfloat16*>(out.data), in.planes.data, in.planes.bytes, stream);
    return true;
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
