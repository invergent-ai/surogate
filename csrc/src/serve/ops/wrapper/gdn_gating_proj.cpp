#include "api/ops/gdn_gating_proj.h"
#include "api/ops/rmsnorm.h"

#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_bf16_weight(const Weight& w, std::int32_t rows, std::int32_t input_rows,
                         const char* name) {
    const std::uint64_t payload_bytes = static_cast<std::uint64_t>(rows) *
                                        static_cast<std::uint64_t>(input_rows) *
                                        sizeof(std::uint16_t);
    if (w.qtype != QType::BF16_CTRL || w.layout != QuantLayout::Contiguous ||
        w.payload_bytes < payload_bytes || w.ndim != 2 || w.n != rows || w.k != input_rows ||
        w.shape[0] != rows || w.shape[1] != input_rows || w.padded_shape[0] != rows ||
        w.padded_shape[1] != input_rows || w.qhigh != nullptr || w.scales != nullptr ||
        w.high_plane_bytes != 0 || w.group != 0 || w.group_size != 0 || !aligned_to(w.qdata, 16)) {
        throw std::invalid_argument(std::string("gdn_gating_proj: invalid ") + name);
    }
}

Weight bf16_row_view(const Weight& parent, std::int32_t row_begin, std::int32_t rows) {
    const std::size_t row_bytes = static_cast<std::size_t>(parent.k) * sizeof(std::uint16_t);
    const auto* data            = static_cast<const std::uint8_t*>(parent.qdata) +
                       static_cast<std::size_t>(row_begin) * row_bytes;
    Weight view          = parent;
    view.payload         = data;
    view.payload_bytes   = static_cast<std::uint64_t>(rows) * row_bytes;
    view.qdata           = data;
    view.shape[0]        = rows;
    view.padded_shape[0] = rows;
    view.n               = rows;
    return view;
}

struct GdnControlParentGeometry {
    std::int32_t input_rows;
    std::int32_t heads;
};

GdnControlParentGeometry require_bf16_parent(const Weight& parent) {
    if (parent.n <= 0 || parent.n % 2 != 0 || parent.k <= 0) {
        throw std::invalid_argument("gdn_gating_proj: ab_weight must have shape [2*H,K]");
    }
    require_bf16_weight(parent, parent.n, parent.k, "ab_weight");
    return {.input_rows = parent.k, .heads = parent.n / 2};
}

void require_vector_tensor(const Tensor& t, DType dtype, std::int32_t n0, const char* op,
                           const char* name) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || !aligned_to(t.data, dtype == DType::FP32 ? 4 : 16)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
    }
}

void require_sequence_tensor(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t tokens,
                             const char* op, const char* name) {
    if (tokens <= 0 || t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != tokens || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || !aligned_to(t.data, dtype == DType::FP32 ? 4 : 16)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
    }
}

} // namespace

std::size_t gdn_gating_proj_workspace_capacity_bytes(std::int32_t heads, std::int32_t input_rows,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    if (heads <= 0 || input_rows <= 0 || min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("gdn_gating_proj: invalid dimensions or token interval");
    }
    if (!detail::bf16_gdn_gating_admits({heads, input_rows, min_tokens})) { return 0; }
    return detail::bf16_gdn_gating_capacity_workspace_bytes(heads, input_rows, min_tokens,
                                                            max_tokens);
}

std::size_t gdn_norm_gating_proj_workspace_capacity_bytes(std::int32_t heads,
                                                          std::int32_t input_rows,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens) {
    if (heads <= 0 || input_rows <= 0 || min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("gdn_norm_gating_proj: invalid dimensions or token interval");
    }
    if (!detail::bf16_gdn_gating_admits({heads, input_rows, min_tokens})) { return 0; }
    return detail::bf16_gdn_norm_gating_capacity_workspace_bytes(heads, input_rows, min_tokens,
                                                                 max_tokens);
}

void gdn_gating_proj(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                     const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g,
                     Tensor& beta, cudaStream_t stream) {
    constexpr const char* op  = "gdn_gating_proj";
    const std::int32_t tokens = x.ne[1];
    const std::int32_t heads = a_weight.n;
    const std::int32_t rows  = a_weight.k;
    if (heads <= 0 || rows <= 0 || tokens <= 0) {
        throw std::invalid_argument(std::string(op) + ": dimensions must be positive");
    }
    require_sequence_tensor(x, DType::BF16, rows, tokens, op, "x");
    require_vector_tensor(A_log, DType::FP32, heads, op, "A_log");
    require_vector_tensor(dt_bias, DType::FP32, heads, op, "dt_bias");
    require_sequence_tensor(g, DType::FP32, heads, tokens, op, "g");
    require_sequence_tensor(beta, DType::FP32, heads, tokens, op, "beta");
    require_bf16_weight(a_weight, heads, rows, "a_weight");
    require_bf16_weight(b_weight, heads, rows, "b_weight");

    if (detail::bf16_gdn_gating_admits({g.ne[0], x.ne[0], tokens})) {
        detail::bf16_gdn_gating_dispatch(x, a_weight, b_weight, A_log, dt_bias, ws, g, beta, stream);
    } else {
        detail::bf16_gdn_gating_proj_generic_launch(x, a_weight, b_weight, A_log, dt_bias,
                                                   g, beta, stream);
    }
}

void gdn_gating_proj(const Tensor& x, const Weight& ab_weight, const Tensor& A_log,
                     const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g, Tensor& beta,
                     cudaStream_t stream) {
    constexpr const char* op                = "gdn_gating_proj";
    const std::int32_t tokens               = x.ne[1];
    const GdnControlParentGeometry geometry = require_bf16_parent(ab_weight);
    require_sequence_tensor(x, DType::BF16, geometry.input_rows, tokens, op, "x");
    require_vector_tensor(A_log, DType::FP32, geometry.heads, op, "A_log");
    require_vector_tensor(dt_bias, DType::FP32, geometry.heads, op, "dt_bias");
    require_sequence_tensor(g, DType::FP32, geometry.heads, tokens, op, "g");
    require_sequence_tensor(beta, DType::FP32, geometry.heads, tokens, op, "beta");

    const Weight a_weight = bf16_row_view(ab_weight, 0, geometry.heads);
    const Weight b_weight = bf16_row_view(ab_weight, geometry.heads, geometry.heads);
    if (detail::bf16_gdn_gating_admits({g.ne[0], x.ne[0], tokens})) {
        detail::bf16_gdn_gating_dispatch(x, a_weight, b_weight, A_log, dt_bias, ws, g, beta, stream);
    } else {
        detail::bf16_gdn_gating_proj_generic_launch(x, a_weight, b_weight, A_log, dt_bias,
                                                   g, beta, stream);
    }
}

void gdn_norm_gating_proj(const Tensor& x, const Tensor& norm_weight, float eps,
                          const Weight& a_weight, const Weight& b_weight, const Tensor& A_log,
                          const Tensor& dt_bias, WorkspaceArena& ws, Tensor& h, Tensor& g,
                          Tensor& beta, cudaStream_t stream) {
    constexpr const char* op  = "gdn_norm_gating_proj";
    const std::int32_t tokens = x.ne[1];
    const std::int32_t heads = a_weight.n;
    const std::int32_t rows  = a_weight.k;
    if (heads <= 0 || rows <= 0 || tokens <= 0) {
        throw std::invalid_argument(std::string(op) + ": dimensions must be positive");
    }
    if (!(eps > 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument("gdn_norm_gating_proj: eps must be positive and finite");
    }
    require_sequence_tensor(x, DType::BF16, rows, tokens, op, "x");
    require_vector_tensor(norm_weight, DType::BF16, rows, op, "norm_weight");
    require_sequence_tensor(h, DType::BF16, rows, tokens, op, "h");
    require_vector_tensor(A_log, DType::FP32, heads, op, "A_log");
    require_vector_tensor(dt_bias, DType::FP32, heads, op, "dt_bias");
    require_sequence_tensor(g, DType::FP32, heads, tokens, op, "g");
    require_sequence_tensor(beta, DType::FP32, heads, tokens, op, "beta");
    require_bf16_weight(a_weight, heads, rows, "a_weight");
    require_bf16_weight(b_weight, heads, rows, "b_weight");

    if (detail::bf16_gdn_gating_admits({g.ne[0], x.ne[0], tokens})) {
        detail::bf16_gdn_norm_gating_dispatch(x, norm_weight, eps, h, a_weight, b_weight, A_log,
                                              dt_bias, ws, g, beta, stream);
    } else {
        rmsnorm(x, norm_weight, eps, true, h, stream);
        detail::bf16_gdn_gating_proj_generic_launch(h, a_weight, b_weight, A_log, dt_bias,
                                                   g, beta, stream);
    }
}

void gdn_norm_gating_proj(const Tensor& x, const Tensor& norm_weight, float eps,
                          const Weight& ab_weight, const Tensor& A_log, const Tensor& dt_bias,
                          WorkspaceArena& ws, Tensor& h, Tensor& g, Tensor& beta,
                          cudaStream_t stream) {
    constexpr const char* op  = "gdn_norm_gating_proj";
    const std::int32_t tokens = x.ne[1];
    if (!(eps > 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument("gdn_norm_gating_proj: eps must be positive and finite");
    }
    const GdnControlParentGeometry geometry = require_bf16_parent(ab_weight);
    require_sequence_tensor(x, DType::BF16, geometry.input_rows, tokens, op, "x");
    require_vector_tensor(norm_weight, DType::BF16, geometry.input_rows, op, "norm_weight");
    require_sequence_tensor(h, DType::BF16, geometry.input_rows, tokens, op, "h");
    require_vector_tensor(A_log, DType::FP32, geometry.heads, op, "A_log");
    require_vector_tensor(dt_bias, DType::FP32, geometry.heads, op, "dt_bias");
    require_sequence_tensor(g, DType::FP32, geometry.heads, tokens, op, "g");
    require_sequence_tensor(beta, DType::FP32, geometry.heads, tokens, op, "beta");

    const Weight a_weight = bf16_row_view(ab_weight, 0, geometry.heads);
    const Weight b_weight = bf16_row_view(ab_weight, geometry.heads, geometry.heads);
    if (detail::bf16_gdn_gating_admits({g.ne[0], x.ne[0], tokens})) {
        detail::bf16_gdn_norm_gating_dispatch(x, norm_weight, eps, h, a_weight, b_weight, A_log,
                                              dt_bias, ws, g, beta, stream);
    } else {
        rmsnorm(x, norm_weight, eps, true, h, stream);
        detail::bf16_gdn_gating_proj_generic_launch(h, a_weight, b_weight, A_log, dt_bias,
                                                   g, beta, stream);
    }
}

} // namespace sinfer::ops
