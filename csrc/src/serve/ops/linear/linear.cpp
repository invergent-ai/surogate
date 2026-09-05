#include "ops/linear/marlin/marlin_plane.h"
#include "api/ops/linear.h"

#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/ggml/ggml_dispatch.h"
#include "ops/linear/fp8_block/fp8_block.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q5/q5_dispatch.h"
#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/w8/w8_dispatch.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace {

std::int64_t checked_numel(const Tensor& tensor, const char* label) {
    std::int64_t total = 1;
    for (const std::int32_t extent : tensor.ne) {
        if (extent <= 0) {
            throw std::invalid_argument(std::string("linear: ") + label +
                                        " dimensions must be positive");
        }
        if (total > std::numeric_limits<std::int64_t>::max() / extent) {
            throw std::overflow_error("linear: tensor size overflows int64");
        }
        total *= extent;
    }
    return total;
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_linear_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear: invalid compute policy");
}

void validate_linear_semantics(const Tensor& x, const Weight& w, const Tensor& out,
                               LinearPolicy policy) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear: x/out must be BF16");
    }
    (void)checked_numel(x, "x");
    (void)checked_numel(out, "out");
    if (x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("linear: x must have shape [K,T]");
    }
    if (out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("linear: out must have shape [N,T]");
    }
    if (w.n <= 0 || w.k <= 0) {
        // An unbound weight reaches here as zeros, and the rule alone does not say whose. The
        // operand shapes do: they are the caller's geometry, which names the projection.
        throw std::invalid_argument(
            "linear: weight n/k must be positive (n " + std::to_string(w.n) + ", k " +
            std::to_string(w.k) + ") for x [" + std::to_string(x.ne[0]) + "," +
            std::to_string(x.ne[1]) + "] into out [" + std::to_string(out.ne[0]) + "," +
            std::to_string(out.ne[1]) + "]");
    }
    if (x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1]) {
        // Name the shapes: a mismatch here is a caller's geometry against a weight's, and the
        // numbers say which of the two is wrong far faster than the rule does.
        throw std::invalid_argument(
            "linear: expected [K,T] x [N,K] -> [N,T], got x [" + std::to_string(x.ne[0]) + "," +
            std::to_string(x.ne[1]) + "] weight [" + std::to_string(w.n) + "," +
            std::to_string(w.k) + "] out [" + std::to_string(out.ne[0]) + "," +
            std::to_string(out.ne[1]) + "]");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear: x/out must be non-null and 16-byte aligned");
    }
    validate_linear_policy(policy);
}

void dispatch_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                     WorkspaceArena* workspace, cudaStream_t stream) {
    switch (w.qtype) {
    case QType::Q4G64_F16S:
        detail::q4_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q5G64_F16S:
        detail::q5_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q6G64_F16S:
        detail::q6_dispatch(x, w, out, policy, stream);
        return;
    case QType::W8G32_F16S:
        // Marlin band (PATCHES.md #33): the vocab head and any plain W8 GEMM
        // in the batch band run the vendored kernel straight into `out`
        // (its row-major [T,N] result is this column-major [N,T] buffer).
        if (x.ne[1] >= detail::marlin_min_band_tokens() && x.ne[1] <= detail::marlin_fixed_m() &&
            detail::marlin_w8_run(x, w, out, stream)) {
            return;
        }
        detail::w8_dispatch(x, w, out, policy, stream);
        return;
    case QType::BF16_CTRL:
        detail::bf16_dispatch(x, w, out, policy, stream);
        return;
    case QType::NVFP4:
        detail::nvfp4_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP8_E4M3FN_ROW_BF16S:
        detail::fp8_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP8_E4M3FN_BLK128_F32S:
    case QType::FP8_E4M3FN_ROW_F32S:
        // Block- and row-scaled FP8 quantise their activation per token per 128 whatever the
        // policy: that is the recipe's own compute, not a profile the caller opts into.
        detail::fp8_block::linear(x, w, out, workspace, stream);
        return;
#define SINFER_GGML_QTYPE_CASE(NAME) case QType::NAME:
    SINFER_GGML_FOR_EACH_TYPE(SINFER_GGML_QTYPE_CASE)
#undef SINFER_GGML_QTYPE_CASE
        // The K-quant route quantises its activation to int8 per 32 whatever the policy:
        // that is the format's native compute, not an A8 profile the caller opts into.
        detail::ggml::ggml_linear(x, w, out, workspace, stream);
        return;
    case QType::FP32_CTRL:
    case QType::I32_CTRL:
        break;
    }
    throw std::invalid_argument("linear: unsupported weight qtype");
}

} // namespace

std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                            std::int32_t input_rows, LinearPolicy policy,
                                            std::int32_t min_tokens, std::int32_t max_tokens) {
    validate_linear_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear workspace: invalid token interval");
    }

    switch (qtype) {
    case QType::Q4G64_F16S:
        (void)detail::select_q4_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q4_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q5G64_F16S:
        (void)detail::select_q5_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q5_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q6G64_F16S:
        (void)detail::select_q6_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q6_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::W8G32_F16S:
        (void)detail::select_w8_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_w8_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::BF16_CTRL:
        (void)detail::select_bf16_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_bf16_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::NVFP4:
        if (!detail::is_nvfp4_linear_problem(output_rows, input_rows) ||
            (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
            throw std::invalid_argument("linear workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                             min_tokens, max_tokens);
    case QType::FP8_E4M3FN_ROW_BF16S:
        return detail::fp8_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                           min_tokens, max_tokens);
    case QType::FP8_E4M3FN_BLK128_F32S:
    case QType::FP8_E4M3FN_ROW_F32S:
        return detail::fp8_block::linear_workspace_capacity_bytes(output_rows, input_rows, max_tokens);
#define SINFER_GGML_QTYPE_CASE(NAME) case QType::NAME:
    SINFER_GGML_FOR_EACH_TYPE(SINFER_GGML_QTYPE_CASE)
#undef SINFER_GGML_QTYPE_CASE
        return detail::ggml::ggml_linear_workspace_capacity_bytes(output_rows, input_rows,
                                                                   max_tokens);
    case QType::FP32_CTRL:
    case QType::I32_CTRL:
        break;
    }
    throw std::invalid_argument("linear workspace: unsupported weight qtype");
}

void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, policy);
    dispatch_linear(x, w, out, policy, &workspace, stream);
}

void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, LinearPolicy::A16Only);
    dispatch_linear(x, w, out, LinearPolicy::A16Only, nullptr, stream);
}

namespace {

/// The row range as a weight of its own, for a format whose rows are independently addressable.
/// Row-split groupwise formats keep their code, high-bit and scale planes separately, each a
/// whole number of bytes per row, so a row offset is the same offset in each plane; contiguous
/// BF16 is one plane. GGML blocks are handled by their own route, which knows the block stride.
Weight weight_row_view(const Weight& w, std::int32_t row_begin, std::int32_t rows) {
    Weight out = w;
    out.n = rows;
    out.shape[0] = rows;
    out.padded_shape[0] = rows;
    if (w.layout == QuantLayout::Contiguous) {
        const std::uint64_t row_bytes = static_cast<std::uint64_t>(w.padded_shape[1]) * sizeof(std::uint16_t);
        out.payload = static_cast<const std::byte*>(w.payload) + static_cast<std::uint64_t>(row_begin) * row_bytes;
        out.qdata   = out.payload;
        out.payload_bytes = static_cast<std::uint64_t>(rows) * row_bytes;
        return out;
    }
    const std::uint64_t groups = static_cast<std::uint64_t>(w.padded_shape[1] / w.group);
    // Bytes per row in each plane. The low plane carries four bits per value for the groupwise
    // trio and eight for W8, which at their group sizes is the same 32 bytes per group for all
    // four -- derived rather than written down, so a format with another pairing still lands right.
    const std::uint64_t low_bits = w.qtype == QType::W8G32_F16S ? 8 : 4;
    const std::uint64_t low_row  = groups * (static_cast<std::uint64_t>(w.group) * low_bits / 8);
    const std::uint64_t high_row = groups * (w.qtype == QType::Q5G64_F16S   ? std::uint64_t{8}
                                             : w.qtype == QType::Q6G64_F16S ? std::uint64_t{16}
                                                                            : std::uint64_t{0});
    const std::uint64_t scale_row = groups * 2;
    out.qdata  = static_cast<const std::byte*>(w.qdata) + static_cast<std::uint64_t>(row_begin) * low_row;
    out.qhigh  = high_row == 0 ? nullptr
                               : static_cast<const std::byte*>(w.qhigh) +
                                     static_cast<std::uint64_t>(row_begin) * high_row;
    out.scales = static_cast<const std::byte*>(w.scales) + static_cast<std::uint64_t>(row_begin) * scale_row;
    return out;
}

} // namespace

Weight weight_rows(const Weight& w, std::int32_t row_begin, std::int32_t rows) {
    if (row_begin < 0 || rows <= 0 || row_begin + rows > w.n) {
        throw std::invalid_argument("weight_rows: row range outside the parent");
    }
    if (detail::ggml::is_ggml_qtype(w.qtype)) { return detail::ggml::ggml_weight_rows(w, row_begin, rows); }
    if (detail::fp8_block::is_fp8_block_qtype(w.qtype)) { return detail::fp8_block::weight_rows(w, row_begin, rows); }
    if (w.layout != QuantLayout::RowSplit && w.layout != QuantLayout::Contiguous) {
        throw std::invalid_argument("weight_rows: this weight's rows are not independently addressable");
    }
    return weight_row_view(w, row_begin, rows);
}

void linear_rows(const Tensor& x, const Weight& w, std::int32_t row_begin, Tensor& out,
                 WorkspaceArena* workspace, cudaStream_t stream) {
    const std::int32_t rows = out.ne[0];
    if (row_begin < 0 || rows <= 0 || row_begin + rows > w.n) {
        throw std::invalid_argument("linear_rows: row range outside the parent");
    }
    if (detail::ggml::is_ggml_qtype(w.qtype)) {
        detail::ggml::ggml_project_rows(x, w, row_begin, out, workspace, stream);
        return;
    }
    if (detail::fp8_block::is_fp8_block_qtype(w.qtype)) {
        detail::fp8_block::project_rows(x, w, row_begin, out, workspace, stream);
        return;
    }
    if (w.layout != QuantLayout::RowSplit && w.layout != QuantLayout::Contiguous) {
        throw std::invalid_argument("linear_rows: this weight's rows are not independently "
                                    "addressable (layout " +
                                    std::to_string(static_cast<int>(w.layout)) + ")");
    }
    const Weight view = weight_row_view(w, row_begin, rows);
    if (workspace != nullptr) {
        linear(x, view, out, LinearPolicy::A16Only, *workspace, stream);
    } else {
        linear(x, view, out, stream);
    }
}

} // namespace sinfer::ops
