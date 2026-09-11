#include "core/limits.h"
#include "core/device.h"
#include "ops/linear/marlin/marlin_plane.h"
#include "api/ops/gdn_input_proj.h"

#include "api/ops/linear.h"
#include "ops/linear/w8a8/w4fp4_plane.h"

#include "core/layout.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ops/gdn_input_proj/ggml/ggml_gdn_input.h"
#include "ops/linear/ggml/ggml_dispatch.h"
#include "ops/linear/fp8_block/fp8_block.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"
#include "ops/gdn_input_proj/w8/w8_gdn_input_kernels.h"
#include "ops/gdn_input_proj/w8/w8_gdn_input_plan.h"
#include "ops/linear/w8a8/w8a8_dispatch.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sinfer::ops {

namespace {
// A parent split by row range straight into its outputs: the K-quants and block-scaled FP8
// both project any range, so the fused ops need no registered shape for either.
bool row_projectable(QType qtype) {
    return qtype == QType::BF16_CTRL || detail::ggml::is_ggml_qtype(qtype) ||
           detail::fp8_block::is_fp8_block_qtype(qtype);
}
void project_rows_any(const Tensor& x, const Weight& w, std::int32_t row_begin, Tensor& out,
                      WorkspaceArena* workspace, cudaStream_t stream) {
    if (w.qtype == QType::BF16_CTRL) {
        linear_rows(x, w, row_begin, out, workspace, stream);
    } else if (detail::fp8_block::is_fp8_block_qtype(w.qtype)) {
        detail::fp8_block::project_rows(x, w, row_begin, out, workspace, stream);
    } else {
        detail::ggml::ggml_project_rows(x, w, row_begin, out, workspace, stream);
    }
}
std::size_t row_projectable_workspace_capacity_bytes(QType qtype, std::int32_t rows, std::int32_t k,
                                                     std::int32_t max_tokens) {
    if (qtype == QType::BF16_CTRL) { return 0; }
    return detail::fp8_block::is_fp8_block_qtype(qtype)
               ? detail::fp8_block::linear_workspace_capacity_bytes(rows, k, max_tokens)
               : detail::ggml::ggml_linear_workspace_capacity_bytes(rows, k, max_tokens);
}
} // namespace
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t cols, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != cols ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("gdn_input_proj: invalid ") + label);
    }
}

void require_conv_tensor(const Tensor& tensor, std::int32_t rows, std::int32_t width,
                         std::int32_t batch, const char* op, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != width ||
        tensor.ne[2] != batch || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + label);
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}


void require_single_parent_nonoverlap(const Tensor& x, const Tensor& qkv, const Tensor& z) {
    if (overlaps(x, qkv) || overlaps(x, z) || overlaps(qkv, z)) {
        throw std::invalid_argument("gdn_input_proj: x, qkv, and z must not overlap");
    }
}

struct ConvGeometry {
    std::int32_t width;
    std::int32_t batch;
    std::int32_t aggregate_columns;
};

ConvGeometry require_snapshot_input(const Tensor& x, std::int32_t hidden) {
    constexpr std::int32_t kMaximumBatch = kMaximumBatchColumns;
    constexpr std::int32_t kMaximumWidth = 16;
    const std::int32_t width             = x.ne[1];
    const std::int32_t batch             = x.ne[2];
    if (width <= 0 || batch <= 0 || batch > kMaximumBatch || (batch > 1 && width > kMaximumWidth)) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot: unsupported B/W domain");
    }
    require_conv_tensor(x, hidden, width, batch, "gdn_input_proj_conv_snapshot", "x");
    return {width, batch, width * batch};
}

ConvGeometry require_record_input(const Tensor& x, std::int32_t hidden) {
    constexpr std::int32_t kMaximumBatch = kMaximumBatchColumns;
    constexpr std::int32_t kMinimumWidth = 2;
    constexpr std::int32_t kMaximumWidth = 16;
    const std::int32_t width             = x.ne[1];
    const std::int32_t batch             = x.ne[2];
    if (width < kMinimumWidth || width > kMaximumWidth || batch <= 0 || batch > kMaximumBatch) {
        throw std::invalid_argument("gdn_input_proj_conv_record: unsupported B/T domain");
    }
    require_conv_tensor(x, hidden, width, batch, "gdn_input_proj_conv_record", "x");
    return {width, batch, width * batch};
}

void require_snapshot_operands(const Tensor& conv_weight, const Tensor& conv_states,
                               const Tensor& valid_columns, const Tensor& initial_state_slots,
                               const Tensor& snapshot_base_slots, std::int32_t channels,
                               ConvGeometry geometry) {
    require_matrix(conv_weight, channels, 4, "conv weight");
    if (conv_states.dtype != DType::BF16 || conv_states.ne[0] != channels ||
        conv_states.ne[1] != 3 || conv_states.ne[2] < geometry.aggregate_columns ||
        conv_states.ne[3] != 1 || !conv_states.is_contiguous() ||
        !aligned_to(conv_states.data, 16)) {
        throw std::invalid_argument(
            "gdn_input_proj_conv_snapshot: invalid convolution snapshot state");
    }
    const auto valid_selector = [batch = geometry.batch](const Tensor& selector) {
        return selector.dtype == DType::I32 && selector.ne[0] == batch && selector.ne[1] == 1 &&
               selector.ne[2] == 1 && selector.ne[3] == 1 && selector.is_contiguous() &&
               selector.data != nullptr;
    };
    if (!valid_selector(initial_state_slots) || !valid_selector(snapshot_base_slots)) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot: invalid state selector");
    }
    if (valid_columns.data != nullptr) {
        if (!valid_selector(valid_columns)) {
            throw std::invalid_argument("gdn_input_proj_conv_snapshot: invalid valid columns");
        }
    }
}

Tensor flatten_columns(const Tensor& tensor, std::int32_t rows, ConvGeometry geometry) {
    return Tensor(tensor.data, tensor.dtype, {rows, geometry.aggregate_columns});
}

void require_record_operands(const Tensor& conv_weight, const Tensor& conv_states,
                             const Tensor& valid_columns, const Tensor& initial_state_slots,
                             std::int32_t channels, ConvGeometry geometry) {
    require_matrix(conv_weight, channels, 4, "conv weight");
    if (conv_states.dtype != DType::BF16 || conv_states.ne[0] != channels ||
        conv_states.ne[1] != 3 || conv_states.ne[2] <= 0 || conv_states.ne[3] != 1 ||
        !conv_states.is_contiguous() || !aligned_to(conv_states.data, 16)) {
        throw std::invalid_argument("gdn_input_proj_conv_record: invalid convolution state");
    }
    const auto valid_selector = [batch = geometry.batch](const Tensor& selector) {
        return selector.dtype == DType::I32 && selector.ne[0] == batch && selector.ne[1] == 1 &&
               selector.ne[2] == 1 && selector.ne[3] == 1 && selector.is_contiguous() &&
               selector.data != nullptr;
    };
    if (!valid_selector(initial_state_slots)) {
        throw std::invalid_argument("gdn_input_proj_conv_record: invalid initial state selector");
    }
    if (valid_columns.data != nullptr && !valid_selector(valid_columns)) {
        throw std::invalid_argument("gdn_input_proj_conv_record: invalid valid columns");
    }
}

bool overlaps_range(const Tensor& tensor, const void* base, std::size_t bytes) {
    if (tensor.data == nullptr || base == nullptr || bytes == 0) { return false; }
    const auto tensor_begin = reinterpret_cast<std::uintptr_t>(tensor.data);
    const auto range_begin  = reinterpret_cast<std::uintptr_t>(base);
    return tensor_begin < range_begin + bytes && range_begin < tensor_begin + tensor.bytes();
}

void require_record_nonoverlap(const Tensor& x, const Tensor& conv_weight,
                               const Tensor& conv_states, const Tensor& valid_columns,
                               const Tensor& initial_state_slots, const Tensor& conv_record,
                               const Tensor& query, const Tensor& key, const Tensor& value,
                               const Tensor& z, const WorkspaceArena& workspace) {
    const std::array<const Tensor*, 10> tensors{
        &x,           &conv_weight, &conv_states, &valid_columns, &initial_state_slots,
        &conv_record, &query,       &key,         &value,         &z};
    for (std::size_t lhs = 0; lhs < tensors.size(); ++lhs) {
        if (tensors[lhs]->data == nullptr) { continue; }
        for (std::size_t rhs = lhs + 1; rhs < tensors.size(); ++rhs) {
            if (tensors[rhs]->data != nullptr && overlaps(*tensors[lhs], *tensors[rhs])) {
                throw std::invalid_argument(
                    "gdn_input_proj_conv_record: tensor operands must not overlap");
            }
        }
        if (overlaps_range(*tensors[lhs], workspace.base(), workspace.capacity())) {
            throw std::invalid_argument(
                "gdn_input_proj_conv_record: tensor operand overlaps live workspace");
        }
    }
}

void require_snapshot_nonoverlap(const Tensor& x, const Tensor& conv_weight,
                                 const Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_state_slots,
                                 const Tensor& snapshot_base_slots, const Tensor& query,
                                 const Tensor& key, const Tensor& value, const Tensor& z,
                                 const WorkspaceArena& workspace) {
    const std::array<const Tensor*, 10> tensors{&x,
                                                &conv_weight,
                                                &conv_states,
                                                &valid_columns,
                                                &initial_state_slots,
                                                &snapshot_base_slots,
                                                &query,
                                                &key,
                                                &value,
                                                &z};
    for (std::size_t lhs = 0; lhs < tensors.size(); ++lhs) {
        if (tensors[lhs]->data == nullptr) { continue; }
        for (std::size_t rhs = lhs + 1; rhs < tensors.size(); ++rhs) {
            const bool shared_state_selectors =
                tensors[lhs] == &initial_state_slots && tensors[rhs] == &snapshot_base_slots;
            if (!shared_state_selectors && tensors[rhs]->data != nullptr &&
                overlaps(*tensors[lhs], *tensors[rhs])) {
                throw std::invalid_argument(
                    "gdn_input_proj_conv_snapshot: tensor operands must not overlap");
            }
        }
        if (overlaps_range(*tensors[lhs], workspace.base(), workspace.capacity())) {
            throw std::invalid_argument(
                "gdn_input_proj_conv_snapshot: tensor operand overlaps live workspace");
        }
    }
}

template <std::size_t Count>
void require_parent_nonoverlap(const Weight& weight,
                               const std::array<const Tensor*, Count>& tensors,
                               const WorkspaceArena& workspace, const char* operation) {
    for (const Tensor* tensor : tensors) {
        if (tensor->data != nullptr &&
            overlaps_range(*tensor, weight.payload,
                           static_cast<std::size_t>(weight.payload_bytes))) {
            throw std::invalid_argument(std::string(operation) +
                                        ": tensor operand overlaps parent weight");
        }
    }
    if (weight.payload != nullptr && workspace.base() != nullptr && workspace.capacity() != 0) {
        const auto weight_begin    = reinterpret_cast<std::uintptr_t>(weight.payload);
        const auto workspace_begin = reinterpret_cast<std::uintptr_t>(workspace.base());
        if (weight_begin < workspace_begin + workspace.capacity() &&
            workspace_begin < weight_begin + weight.payload_bytes) {
            throw std::invalid_argument(std::string(operation) +
                                        ": parent weight overlaps live workspace");
        }
    }
}

void require_snapshot_capacity_domain(std::int32_t batch_size, std::int32_t min_width,
                                      std::int32_t max_width) {
    constexpr std::int32_t kMaximumBatch = kMaximumBatchColumns;
    constexpr std::int32_t kMaximumWidth = 16;
    if (batch_size <= 0 || batch_size > kMaximumBatch || min_width <= 0 || max_width < min_width ||
        (batch_size > 1 && max_width > kMaximumWidth)) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot workspace: invalid B/W domain");
    }
}

void require_record_capacity_domain(std::int32_t batch_size, std::int32_t min_width,
                                    std::int32_t max_width) {
    constexpr std::int32_t kMaximumBatch = kMaximumBatchColumns;
    constexpr std::int32_t kMinimumWidth = 2;
    constexpr std::int32_t kMaximumWidth = 16;
    if (batch_size <= 0 || batch_size > kMaximumBatch || min_width < kMinimumWidth ||
        max_width < min_width || max_width > kMaximumWidth) {
        throw std::invalid_argument("gdn_input_proj_conv_record workspace: invalid B/T domain");
    }
}

void require_rowsplit(const Weight& weight, QType qtype, std::int32_t rows, const char* label) {
    const bool q4_planes =
        qtype != QType::Q4G64_F16S || (weight.qhigh == nullptr && weight.high_plane_bytes == 0);
    const bool q5_planes =
        qtype != QType::Q5G64_F16S || (weight.qhigh != nullptr && weight.high_plane_bytes != 0);
    if (weight.qtype != qtype || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 64 || weight.group != 64 ||
        weight.ndim != 2 || weight.n != rows || weight.k != 5120 || weight.shape[0] != rows ||
        weight.shape[1] != 5120 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != 5120 || !q4_planes || !q5_planes ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 4) ||
        (qtype == QType::Q5G64_F16S && !aligned_to(weight.qhigh, 16))) {
        throw std::invalid_argument(std::string("gdn_input_proj: invalid ") + label);
    }
}

void require_w8_rowsplit(const Weight& weight, std::int32_t rows, const char* label) {
    // surogate vendor patch (PATCHES.md #13/#18): K 2048 (35B), 1024
    // (qwen3.5-0.8b), or 2560 (qwen3.5-4b).
    const std::int32_t k = weight.k;
    if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 32 || weight.group != 32 ||
        weight.ndim != 2 || weight.n != rows || (k != 2048 && k != 1024 && k != 2560) ||
        weight.shape[0] != rows || weight.shape[1] != k || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != k || weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument(std::string("gdn_input_proj: invalid ") + label);
    }
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("gdn_input_proj: invalid compute policy");
}

void dispatch_single_parent(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                            LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    validate_policy(policy);
    if (weight.qtype == QType::BF16_CTRL && policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("BF16 input projection admits only A16");
    }
    const std::int32_t cols = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("gdn_input_proj: T must be positive"); }

    if (row_projectable(weight.qtype)) {
        // A K-quant or block-FP8 parent splits by row range straight into the two outputs:
        // qkv rows first, z rows after, the split read off the output views.
        if (qkv.ne[0] + z.ne[0] != weight.n) {
            throw std::invalid_argument("gdn_input_proj: row-projected parent rows must equal qkv + z rows");
        }
        project_rows_any(x, weight, 0, qkv, workspace, stream);
        project_rows_any(x, weight, qkv.ne[0], z, workspace, stream);
        return;
    }

    if (weight.qtype == QType::NVFP4) {
        // The registered geometry is the 27B's; other shapes take the split from the output
        // views and run on the cuBLASLt route (#84).
        const bool registered = weight.n == 16384 && weight.k == 5120;
        const std::int32_t kHidden  = registered ? 5120 : weight.k;
        const std::int32_t kQkvRows = registered ? 10240 : qkv.ne[0];
        const std::int32_t kZRows   = registered ? 6144 : z.ne[0];
        const std::int32_t kRows    = kQkvRows + kZRows;
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
            throw std::invalid_argument("NVFP4 gdn_input_proj admits only A16 or A4");
        }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(qkv, kQkvRows, cols, "qkv");
        require_matrix(z, kZRows, cols, "z");
        require_single_parent_nonoverlap(x, qkv, z);
        detail::validate_nvfp4_weight(weight, "nvfp4 gdn_input_proj");
        if (weight.n != kRows || weight.k != kHidden ||
            (!registered && !detail::is_nvfp4_generic_problem(weight.n, weight.k))) {
            throw std::invalid_argument("nvfp4 gdn_input_proj: unsupported weight shape");
        }
        detail::nvfp4_gdn_input_dispatch(x, weight, qkv, z, policy, workspace, stream);
        return;
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        constexpr std::int32_t kHidden  = 5120;
        constexpr std::int32_t kQkvRows = 10240;
        constexpr std::int32_t kZRows   = 6144;
        constexpr std::int32_t kRows    = kQkvRows + kZRows;
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
            throw std::invalid_argument("FP8 gdn_input_proj admits only A16 or A8");
        }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(qkv, kQkvRows, cols, "qkv");
        require_matrix(z, kZRows, cols, "z");
        require_single_parent_nonoverlap(x, qkv, z);
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("fp8 gdn_input_proj: unsupported weight shape");
        }
        detail::validate_fp8_weight(weight, "fp8 gdn_input_proj");
        detail::fp8_gdn_input_dispatch(x, weight, qkv, z, policy, workspace, stream);
        return;
    }

    // surogate vendor patch (PATCHES.md #13/#16): the small fused structure
    // (0.8b k=1024 and 2b k=2048) is keyed on parent rows, not hidden.
    const bool small_fused      = weight.n == 8192;
    const std::int32_t kHidden  = weight.k;
    const std::int32_t kQkvRows = small_fused ? 6144 : 8192;
    const std::int32_t kZRows   = small_fused ? 2048 : 4096;
    const std::int32_t kRows    = kQkvRows + kZRows;
    if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("W8 gdn_input_proj admits A16 or A8");
    }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(qkv, kQkvRows, cols, "qkv");
    require_matrix(z, kZRows, cols, "z");
    require_single_parent_nonoverlap(x, qkv, z);
    require_w8_rowsplit(weight, kRows, "query/key/value/z weight");
    // surogate vendor patch (PATCHES.md #17): large-T prefill under AllowA8
    // runs the W8A8-int IMMA path (split2 epilogue writes qkv/z directly).
    if (policy == LinearPolicy::AllowA8 && cols >= detail::kW8A8MinTokens &&
        workspace != nullptr) {
        detail::w8a8_gemm_split2(x, weight, qkv, z, *workspace, stream);
        return;
    }
    // Preserve the same K reduction across narrow and wide A16 batches.
    detail::w8_gdn_input_dispatch(x, weight, qkv, z, stream);
}

detail::Q4Q5GdnInputConvPlan resolve_q4_q5_conv_plan(std::int32_t tokens, std::int32_t batch_size) {
    return detail::q4_q5_gdn_input_conv_resolve_plan({5120, 4096, 12288, 10240, 6144, 5120, tokens},
                                                     batch_size);
}

detail::W8GdnInputConvPlan resolve_w8_conv_plan(std::int32_t tokens, std::int32_t batch_size) {
    return detail::w8_gdn_input_conv_resolve_plan({2048, 8192, 4096, 12288, 2048, tokens},
                                                  batch_size);
}

struct ProjectedWorkspace {
    Tensor projected;
};

template <class Allocator>
ProjectedWorkspace allocate_projected_workspace(Allocator& allocator, std::int32_t channels,
                                                std::int32_t tokens) {
    ProjectedWorkspace out;
    out.projected = allocator.alloc(DType::BF16, {channels, tokens});
    return out;
}

std::size_t composed_snapshot_capacity(std::int32_t channels, std::int32_t aggregate_columns,
                                       std::size_t projection_workspace_bytes) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_projected_workspace(layout, channels, aggregate_columns);
    if (projection_workspace_bytes != 0) { (void)layout.alloc_bytes(projection_workspace_bytes); }
    return layout.peak_bytes(1);
}

template <class Project>
void compose_batched_snapshot(const Tensor& x, const Tensor& conv_weight, Tensor& conv_states,
                              const Tensor& valid_columns, const Tensor& initial_state_slots,
                              const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                              Tensor& value, Tensor& z, std::int32_t query_rows,
                              std::int32_t key_rows, std::int32_t value_rows, ConvGeometry geometry,
                              WorkspaceArena& workspace, cudaStream_t stream, Project&& project) {
    const std::int32_t channels = query_rows + key_rows + value_rows;
    auto scope                  = workspace.scope();
    ProjectedWorkspace scratch =
        allocate_projected_workspace(workspace, channels, geometry.aggregate_columns);

    Tensor x_flat = flatten_columns(x, x.ne[0], geometry);
    Tensor z_flat = flatten_columns(z, z.ne[0], geometry);
    project(x_flat, scratch.projected, z_flat);

    Tensor projected(scratch.projected.data, DType::BF16,
                     {channels, geometry.width, geometry.batch});
    detail::gdn_projected_conv_snapshot_launch(projected, conv_weight, conv_states, valid_columns,
                                               initial_state_slots, snapshot_base_slots, query, key,
                                               value, stream);
}

template <class Project>
void compose_record(const Tensor& x, const Tensor& conv_weight, const Tensor& conv_states,
                    const Tensor& valid_columns, const Tensor& initial_state_slots,
                    Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                    ConvGeometry geometry, WorkspaceArena& workspace, cudaStream_t stream,
                    Project&& project) {
    auto scope         = workspace.scope();
    Tensor x_flat      = flatten_columns(x, x.ne[0], geometry);
    Tensor record_flat = flatten_columns(conv_record, conv_record.ne[0], geometry);
    Tensor z_flat      = flatten_columns(z, z.ne[0], geometry);
    project(x_flat, record_flat, z_flat);
    detail::gdn_projected_conv_record_launch(conv_record, conv_weight, conv_states, valid_columns,
                                             initial_state_slots, query, key, value, stream);
}

void dispatch_single_parent_snapshot(const Tensor& x, const Weight& weight,
                                     const Tensor& conv_weight, Tensor& conv_states,
                                     const Tensor& valid_columns, const Tensor& initial_state_slots,
                                     const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                     Tensor& value, Tensor& z, LinearPolicy policy,
                                     WorkspaceArena& workspace, cudaStream_t stream) {
    validate_policy(policy);
    if (weight.qtype == QType::BF16_CTRL && policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("BF16 input projection admits only A16");
    }

    if (row_projectable(weight.qtype)) {
        // K-quant or block-FP8 parent: the row split comes from the output views, the projection is the
        // plain single-parent form, and the conv is the shared projected-conv tail.
        const std::int32_t kQueryRows = query.ne[0];
        const std::int32_t kKeyRows   = key.ne[0];
        const std::int32_t kValueRows = value.ne[0];
        const std::int32_t kZRows     = z.ne[0];
        const std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
        const ConvGeometry geometry   = require_snapshot_input(x, weight.k);
        if (weight.n != kChannels + kZRows) {
            throw std::invalid_argument(
                "gdn_input_proj_conv_snapshot: K-quant parent rows must equal q+k+v+z");
        }
        require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                                  snapshot_base_slots, kChannels, geometry);
        require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "query");
        require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "key");
        require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "value");
        require_conv_tensor(z, kZRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "z");
        if (geometry.batch > 1) {
            compose_batched_snapshot(
                x, conv_weight, conv_states, valid_columns, initial_state_slots,
                snapshot_base_slots, query, key, value, z, kQueryRows, kKeyRows, kValueRows,
                geometry, workspace, stream,
                [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                    gdn_input_proj(x_flat, weight, projected, z_flat, policy, workspace, stream);
                });
            return;
        }
        auto scope = workspace.scope();
        ProjectedWorkspace scratch =
            allocate_projected_workspace(workspace, kChannels, geometry.width);
        gdn_input_proj(x, weight, scratch.projected, z, policy, workspace, stream);
        detail::gdn_projected_conv_snapshot_launch(scratch.projected, conv_weight, conv_states,
                                                   valid_columns, initial_state_slots,
                                                   snapshot_base_slots, query, key, value, stream);
        return;
    }

    if (weight.qtype == QType::NVFP4) {
        // Only the 27B geometry has fused NVFP4 conv schedules; other shapes derive their row
        // split the way the W8 path does and project-then-conv through cuBLASLt (#84).
        const bool registered              = weight.n == 16384 && weight.k == 5120;
        const std::int32_t kHidden         = registered ? 5120 : weight.k;
        const std::int32_t kQueryRows      = registered ? 2048 : query.ne[0];
        const std::int32_t kKeyRows        = registered ? 2048 : key.ne[0];
        const std::int32_t kValueRows      = registered ? 6144 : value.ne[0];
        const std::int32_t kZRows          = registered ? 6144 : z.ne[0];
        const std::int32_t kChannels       = kQueryRows + kKeyRows + kValueRows;
        const std::int32_t kParentRows     = kChannels + kZRows;
        const ConvGeometry geometry        = require_snapshot_input(x, kHidden);
        detail::validate_nvfp4_weight(weight, "nvfp4 gdn_input_proj_conv_snapshot");
        if (weight.n != kParentRows || weight.k != kHidden ||
            (!registered && !detail::is_nvfp4_generic_problem(weight.n, weight.k))) {
            throw std::invalid_argument(
                "nvfp4 gdn_input_proj_conv_snapshot: unsupported weight shape");
        }
        require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                                  snapshot_base_slots, kChannels, geometry);
        require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "query");
        require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "key");
        require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "value");
        require_conv_tensor(z, kZRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "z");
        if (!registered) {
            if (geometry.batch > 1) {
                compose_batched_snapshot(
                    x, conv_weight, conv_states, valid_columns, initial_state_slots,
                    snapshot_base_slots, query, key, value, z, kQueryRows, kKeyRows, kValueRows,
                    geometry, workspace, stream,
                    [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                        gdn_input_proj(x_flat, weight, projected, z_flat, policy, workspace,
                                       stream);
                    });
                return;
            }
            auto scope = workspace.scope();
            ProjectedWorkspace scratch =
                allocate_projected_workspace(workspace, kChannels, geometry.width);
            gdn_input_proj(x, weight, scratch.projected, z, policy, workspace, stream);
            detail::gdn_projected_conv_snapshot_launch(scratch.projected, conv_weight, conv_states,
                                                       valid_columns, initial_state_slots,
                                                       snapshot_base_slots, query, key, value,
                                                       stream);
            return;
        }
        const detail::Nvfp4GdnConvPlan plan =
            detail::nvfp4_gdn_conv_resolve_plan(policy, geometry.width, geometry.batch);
        if (geometry.batch > 1) {
            if (plan.schedule != detail::Nvfp4GdnConvScheduleId::Materialized) {
                throw std::logic_error("batched NVFP4 GDN conv selected a fused schedule");
            }
            compose_batched_snapshot(x, conv_weight, conv_states, valid_columns,
                                     initial_state_slots, snapshot_base_slots, query, key, value, z,
                                     kQueryRows, kKeyRows, kValueRows, geometry, workspace, stream,
                                     [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                                         gdn_input_proj(x_flat, weight, projected, z_flat, policy,
                                                        workspace, stream);
                                     });
            return;
        }
        detail::nvfp4_gdn_snapshot_dispatch(x, weight, conv_weight, conv_states, valid_columns,
                                            initial_state_slots, snapshot_base_slots, query, key,
                                            value, z, policy, workspace, stream);
        return;
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        constexpr std::int32_t kHidden     = 5120;
        constexpr std::int32_t kQueryRows  = 2048;
        constexpr std::int32_t kKeyRows    = 2048;
        constexpr std::int32_t kValueRows  = 6144;
        constexpr std::int32_t kZRows      = 6144;
        constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
        constexpr std::int32_t kParentRows = kChannels + kZRows;
        const ConvGeometry geometry        = require_snapshot_input(x, kHidden);
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
            throw std::invalid_argument("FP8 gdn_input_proj_conv_snapshot admits only A16 or A8");
        }
        detail::validate_fp8_weight(weight, "fp8 gdn_input_proj_conv_snapshot");
        if (weight.n != kParentRows || weight.k != kHidden) {
            throw std::invalid_argument(
                "fp8 gdn_input_proj_conv_snapshot: unsupported weight shape");
        }
        require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                                  snapshot_base_slots, kChannels, geometry);
        require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "query");
        require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "key");
        require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "value");
        require_conv_tensor(z, kZRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "z");
        require_snapshot_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                                    snapshot_base_slots, query, key, value, z, workspace);
        const std::array<const Tensor*, 10> tensors{&x,
                                                    &conv_weight,
                                                    &conv_states,
                                                    &valid_columns,
                                                    &initial_state_slots,
                                                    &snapshot_base_slots,
                                                    &query,
                                                    &key,
                                                    &value,
                                                    &z};
        require_parent_nonoverlap(weight, tensors, workspace, "fp8 gdn_input_proj_conv_snapshot");
        detail::fp8_gdn_snapshot_dispatch(x, weight, conv_weight, conv_states, valid_columns,
                                          initial_state_slots, snapshot_base_slots, query, key,
                                          value, z, policy, workspace, stream);
        return;
    }

    // surogate vendor patch (PATCHES.md #13/#16): the small fused structure
    // (0.8b k=1024, 2b k=2048; both 16 symmetric V heads) is keyed on parent
    // rows — the 35B parent has 12288.
    const bool small_fused        = weight.n == 8192;
    const std::int32_t kHidden    = weight.k;
    const std::int32_t kQueryRows = 2048;
    const std::int32_t kKeyRows   = 2048;
    const std::int32_t kValueRows = small_fused ? 2048 : 4096;
    const std::int32_t kZRows     = small_fused ? 2048 : 4096;
    const std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
    const ConvGeometry geometry       = require_snapshot_input(x, kHidden);
    if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("W8 gdn_input_proj_conv_snapshot admits A16 or A8");
    }
    require_w8_rowsplit(weight, kChannels + kZRows, "query/key/value/z weight");
    require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                              snapshot_base_slots, kChannels, geometry);
    require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "query");
    require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "key");
    require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "value");
    require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_snapshot",
                        "z");
    if (geometry.batch > 1) {
        compose_batched_snapshot(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                                 snapshot_base_slots, query, key, value, z, kQueryRows, kKeyRows,
                                 kValueRows, geometry, workspace, stream,
                                 [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                                     gdn_input_proj(x_flat, weight, projected, z_flat, stream);
                                 });
        return;
    }

    const detail::W8GdnInputConvPlan plan = resolve_w8_conv_plan(geometry.width, geometry.batch);
    if (plan.schedule == detail::W8GdnInputConvScheduleId::DecodeFused) {
        detail::w8_gdn_input_decode_conv_snapshot_launch(
            x, weight, conv_weight, conv_states, valid_columns, initial_state_slots,
            snapshot_base_slots, query, key, value, z, stream);
        return;
    }
    if (plan.schedule == detail::W8GdnInputConvScheduleId::SplitKMmaFused) {
        // The launcher covers every admitted W8 geometry, including 0.8B/2B/4B.
        // Keep execution on the fused route whose workspace requirement is zero.
        detail::w8_gdn_input_splitk_conv_snapshot_launch(
            x, weight, conv_weight, conv_states, valid_columns, initial_state_slots,
            snapshot_base_slots, query, key, value, z, stream);
        return;
    }

    auto scope                 = workspace.scope();
    ProjectedWorkspace scratch = allocate_projected_workspace(workspace, kChannels, geometry.width);
    gdn_input_proj(x, weight, scratch.projected, z, stream);
    detail::gdn_projected_conv_snapshot_launch(scratch.projected, conv_weight, conv_states,
                                               valid_columns, initial_state_slots,
                                               snapshot_base_slots, query, key, value, stream);
}

void dispatch_single_parent_record(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                   const Tensor& conv_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots, Tensor& conv_record,
                                   Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                   LinearPolicy policy, WorkspaceArena& workspace,
                                   cudaStream_t stream) {
    validate_policy(policy);
    if (weight.qtype == QType::BF16_CTRL && policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("BF16 input projection admits only A16");
    }

    if (row_projectable(weight.qtype)) {
        const std::int32_t kQueryRows = query.ne[0];
        const std::int32_t kKeyRows   = key.ne[0];
        const std::int32_t kValueRows = value.ne[0];
        const std::int32_t kZRows     = z.ne[0];
        const std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
        const ConvGeometry geometry   = require_record_input(x, weight.k);
        if (weight.n != kChannels + kZRows) {
            throw std::invalid_argument(
                "gdn_input_proj_conv_record: K-quant parent rows must equal q+k+v+z");
        }
        require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                                kChannels, geometry);
        require_conv_tensor(conv_record, kChannels, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "conv record");
        require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "query");
        require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "key");
        require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "value");
        require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                            "z");
        require_record_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                                  conv_record, query, key, value, z, workspace);
        if (geometry.batch > 1) {
            compose_record(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                           conv_record, query, key, value, z, geometry, workspace, stream,
                           [&](const Tensor& x_flat, Tensor& record_flat, Tensor& z_flat) {
                               gdn_input_proj(x_flat, weight, record_flat, z_flat, policy,
                                              workspace, stream);
                           });
            return;
        }
        auto scope = workspace.scope();
        gdn_input_proj(x, weight, conv_record, z, policy, workspace, stream);
        detail::gdn_projected_conv_record_launch(conv_record, conv_weight, conv_states,
                                                 valid_columns, initial_state_slots, query, key,
                                                 value, stream);
        return;
    }
    if (weight.qtype == QType::NVFP4) {
        // See the snapshot path: outside the 27B geometry the split comes from the weight and
        // the projection runs on cuBLASLt (#84).
        const bool registered              = weight.n == 16384 && weight.k == 5120;
        const std::int32_t kHidden         = registered ? 5120 : weight.k;
        const std::int32_t kQueryRows      = registered ? 2048 : query.ne[0];
        const std::int32_t kKeyRows        = registered ? 2048 : key.ne[0];
        const std::int32_t kValueRows      = registered ? 6144 : value.ne[0];
        const std::int32_t kZRows          = registered ? 6144 : z.ne[0];
        const std::int32_t kChannels       = kQueryRows + kKeyRows + kValueRows;
        const std::int32_t kParentRows     = kChannels + kZRows;
        const ConvGeometry geometry        = require_record_input(x, kHidden);
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
            throw std::invalid_argument("NVFP4 gdn_input_proj_conv_record admits only A16 or A4");
        }
        detail::validate_nvfp4_weight(weight, "nvfp4 gdn_input_proj_conv_record");
        if (weight.n != kParentRows || weight.k != kHidden ||
            (!registered && !detail::is_nvfp4_generic_problem(weight.n, weight.k))) {
            throw std::invalid_argument(
                "nvfp4 gdn_input_proj_conv_record: unsupported weight shape");
        }
        require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                                kChannels, geometry);
        require_conv_tensor(conv_record, kChannels, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "conv record");
        require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "query");
        require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "key");
        require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "value");
        require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                            "z");
        require_record_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                                  conv_record, query, key, value, z, workspace);

        const detail::Nvfp4GdnConvPlan plan =
            registered ? detail::nvfp4_gdn_conv_resolve_plan(policy, geometry.width, geometry.batch)
                       : detail::Nvfp4GdnConvPlan{detail::Nvfp4GdnConvScheduleId::Materialized};
        if (plan.schedule == detail::Nvfp4GdnConvScheduleId::Materialized && geometry.batch > 1) {
            compose_record(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                           conv_record, query, key, value, z, geometry, workspace, stream,
                           [&](const Tensor& x_flat, Tensor& record_flat, Tensor& z_flat) {
                               gdn_input_proj(x_flat, weight, record_flat, z_flat, policy,
                                              workspace, stream);
                           });
            return;
        }
        if (plan.schedule == detail::Nvfp4GdnConvScheduleId::SmallTFusedA16) {
            detail::nvfp4_gdn_record_small_t_launch(x, weight, conv_weight, conv_states,
                                                    valid_columns, initial_state_slots, conv_record,
                                                    query, key, value, z, stream);
            return;
        }

        auto scope = workspace.scope();
        gdn_input_proj(x, weight, conv_record, z, policy, workspace, stream);
        if (registered) {
            detail::nvfp4_gdn_record_post_launch(conv_record, conv_weight, conv_states, valid_columns,
                                                initial_state_slots, query, key, value, stream);
        } else {
            detail::gdn_projected_conv_record_launch(conv_record, conv_weight, conv_states, valid_columns,
                                                     initial_state_slots, query, key, value, stream);
        }
        return;
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        constexpr std::int32_t kHidden     = 5120;
        constexpr std::int32_t kQueryRows  = 2048;
        constexpr std::int32_t kKeyRows    = 2048;
        constexpr std::int32_t kValueRows  = 6144;
        constexpr std::int32_t kZRows      = 6144;
        constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
        constexpr std::int32_t kParentRows = kChannels + kZRows;
        const ConvGeometry geometry        = require_record_input(x, kHidden);
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
            throw std::invalid_argument("FP8 gdn_input_proj_conv_record admits only A16 or A8");
        }
        detail::validate_fp8_weight(weight, "fp8 gdn_input_proj_conv_record");
        if (weight.n != kParentRows || weight.k != kHidden) {
            throw std::invalid_argument("fp8 gdn_input_proj_conv_record: unsupported weight shape");
        }
        require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                                kChannels, geometry);
        require_conv_tensor(conv_record, kChannels, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "conv record");
        require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "query");
        require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "key");
        require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "value");
        require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                            "z");
        require_record_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                                  conv_record, query, key, value, z, workspace);
        const std::array<const Tensor*, 10> tensors{
            &x,           &conv_weight, &conv_states, &valid_columns, &initial_state_slots,
            &conv_record, &query,       &key,         &value,         &z};
        require_parent_nonoverlap(weight, tensors, workspace, "fp8 gdn_input_proj_conv_record");
        detail::fp8_gdn_record_dispatch(x, weight, conv_weight, conv_states, valid_columns,
                                        initial_state_slots, conv_record, query, key, value, z,
                                        policy, workspace, stream);
        return;
    }

    // surogate vendor patch (PATCHES.md #13/#16/#18): the record path keyed the 35B
    // geometry as constants while its snapshot sibling and the NVFP4 record path had
    // already been generalized. The small fused structure (0.8b k=1024, 2b k=2048; both
    // 16 symmetric V heads) is keyed on parent rows -- the 35B and 4b parents carry
    // 12288 -- and the launcher selects its own per-geometry table from the weight, so
    // deriving the split here is all that stood between those targets and MTP.
    const bool small_fused        = weight.n == 8192;
    const std::int32_t kHidden    = weight.k;
    const std::int32_t kQueryRows = 2048;
    const std::int32_t kKeyRows   = 2048;
    const std::int32_t kValueRows = small_fused ? 2048 : 4096;
    const std::int32_t kZRows     = small_fused ? 2048 : 4096;
    const std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
    const ConvGeometry geometry   = require_record_input(x, kHidden);
    if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("W8 gdn_input_proj_conv_record admits A16 or A8");
    }
    require_w8_rowsplit(weight, kChannels + kZRows, "query/key/value/z weight");
    require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots, kChannels,
                            geometry);
    require_conv_tensor(conv_record, kChannels, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "conv record");
    require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "query");
    require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                        "key");
    require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "value");
    require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                        "z");
    require_record_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                              conv_record, query, key, value, z, workspace);

    if (geometry.batch > 1) {
        compose_record(x, conv_weight, conv_states, valid_columns, initial_state_slots, conv_record,
                       query, key, value, z, geometry, workspace, stream,
                       [&](const Tensor& x_flat, Tensor& record_flat, Tensor& z_flat) {
                           gdn_input_proj(x_flat, weight, record_flat, z_flat, stream);
                       });
        return;
    }
    const detail::W8GdnInputConvPlan plan = resolve_w8_conv_plan(geometry.width, geometry.batch);
    if (plan.schedule != detail::W8GdnInputConvScheduleId::SplitKMmaFused) {
        throw std::logic_error("W8 ReplaySSM record domain selected a non-record schedule");
    }
    detail::w8_gdn_input_splitk_conv_record_launch(x, weight, conv_weight, conv_states,
                                                   valid_columns, initial_state_slots, conv_record,
                                                   query, key, value, z, stream);
}

} // namespace

void gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    Tensor& qkv, Tensor& z, cudaStream_t stream) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQkRows     = 4096;
    constexpr std::int32_t kValueRows  = 6144;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kQkvRows    = kQkRows + kValueRows;
    constexpr std::int32_t kParentRows = kValueRows + kZRows;
    const std::int32_t cols            = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("gdn_input_proj: T must be positive"); }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(qkv, kQkvRows, cols, "qkv");
    require_matrix(z, kZRows, cols, "z");
    require_rowsplit(qk_weight, QType::Q4G64_F16S, kQkRows, "qk weight");
    require_rowsplit(value_z_weight, QType::Q5G64_F16S, kParentRows, "value/z weight");

    detail::q4_q5_gdn_input_dispatch(x, qk_weight, value_z_weight, qkv, z, stream);
}

std::size_t gdn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                                    std::int32_t input_rows, LinearPolicy policy,
                                                    std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    validate_policy(policy);
    if (parent_qtype == QType::BF16_CTRL && policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("BF16 input projection admits only A16");
    }
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("gdn_input_proj workspace: invalid token interval");
    }
    if (row_projectable(parent_qtype)) {
        return row_projectable_workspace_capacity_bytes(parent_qtype, parent_rows, input_rows, max_tokens);
    }
    if (parent_qtype == QType::NVFP4) {
        const bool registered = parent_rows == detail::Nvfp4GdnInputGeometry::kOutputRows &&
                                input_rows == detail::Nvfp4GdnInputGeometry::kInputRows;
        // Shapes outside the registered geometry run on the cuBLASLt route (#84).
        if ((!registered && !detail::is_nvfp4_generic_problem(parent_rows, input_rows)) ||
            (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
            throw std::invalid_argument("gdn_input_proj workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_gdn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens,
                                                                parent_rows, input_rows);
    }
    if (parent_qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        if (parent_rows != detail::Fp8GdnInputGeometry::kOutputRows ||
            input_rows != detail::Fp8GdnInputGeometry::kInputRows ||
            (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8)) {
            throw std::invalid_argument("gdn_input_proj workspace: unsupported FP8 profile");
        }
        return detail::fp8_gdn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    // surogate vendor patches (PATCHES.md #13/#16/#17): W8 fused parents
    // (35B 12288/2048; 0.8b 8192/1024; 2b 8192/2048) at A16 or AllowA8;
    // AllowA8 large-T sizes the quantized-activation workspace.
    // A parent shape the fused W8 GDN kernels are not registered for takes no workspace from
    // them: the profile is sized for what it binds (the 27B's parents are groupwise Q4/Q5 or
    // native K-quants), and a W8 parent of such a shape is refused where it would run.
    if (parent_qtype == QType::W8G32_F16S &&
        (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) &&
        !((parent_rows == 12288 && (input_rows == 2048 || input_rows == 2560)) ||
          (parent_rows == 8192 && (input_rows == 1024 || input_rows == 2048)))) {
        return 0;
    }
    if (parent_qtype == QType::W8G32_F16S &&
        (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) &&
        ((parent_rows == 12288 && (input_rows == 2048 || input_rows == 2560)) ||
         (parent_rows == 8192 && (input_rows == 1024 || input_rows == 2048)))) {
        const std::int32_t qkv_rows = parent_rows == 12288 ? 8192 : 6144;
        const std::int32_t z_rows   = parent_rows == 12288 ? 4096 : 2048;
        (void)detail::w8_gdn_input_resolve_plan(
            {input_rows, qkv_rows, z_rows, parent_rows, input_rows, min_tokens});
        (void)detail::w8_gdn_input_resolve_plan(
            {input_rows, qkv_rows, z_rows, parent_rows, input_rows, max_tokens});
        if (policy == LinearPolicy::AllowA8 && max_tokens >= detail::kW8A8MinTokens) {
            // surogate patch (PATCHES.md #25): the cutlass fp4 path needs its
            // atom-SF quant buffers plus the [tokens, parent] stage.
            if (detail::w8_prefill_quant_mode() != detail::PrefillQuantMode::Fp4) {
                return detail::w8a8_act_quant_bytes(input_rows, max_tokens);
            }
            const std::size_t a8 = detail::w8a8_act_quant_bytes(input_rows, max_tokens);
            const std::size_t fp4 = detail::w4fp4_cutlass_workspace_bytes(
                parent_rows, input_rows, max_tokens, true);
            return a8 > fp4 ? a8 : fp4;
        }
        return 0;
    }
    throw std::invalid_argument("gdn_input_proj workspace: unsupported parent profile");
}

void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_value_z_weight, qkv, z, policy, &workspace, stream);
}

namespace {

// The split pair's two GEMMs, into whatever planes the caller wants the halves in (#87).
void project_split(const Tensor& x, const Weight& qkv_weight, const Weight& z_weight, Tensor& qkv,
                   Tensor& z, LinearPolicy first_policy, LinearPolicy second_policy, WorkspaceArena& workspace,
                   cudaStream_t stream) {
    validate_policy(first_policy);
    validate_policy(second_policy);
    if (qkv_weight.k != z_weight.k || qkv_weight.k != x.ne[0]) {
        throw std::invalid_argument("gdn_input_proj split: halves disagree on K");
    }
    if (qkv.ne[0] != qkv_weight.n || z.ne[0] != z_weight.n) {
        throw std::invalid_argument("gdn_input_proj split: destination rows do not match a half");
    }
    linear(x, qkv_weight, qkv, first_policy, workspace, stream);
    linear(x, z_weight, z, second_policy, workspace, stream);
}

/// The convolution plane and z from a query|key + value|z pair, for any row-addressable format.
/// A row range of a column-major plane is not contiguous, so the plane's two pieces are projected
/// into scratch and placed with strided copies -- the same split output layout used to
/// cut one product into [qkv | z]. Only z, being its parent's tail and a plane of its own, is
/// projected straight into place. The registered Q4/Q5 pair never reaches here: it has fused
/// kernels, and the public entry delegates to them.
void project_pair(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                  Tensor& qkv, Tensor& z, LinearPolicy first_policy, LinearPolicy second_policy, WorkspaceArena& workspace,
                  cudaStream_t stream) {
    validate_policy(first_policy);
    validate_policy(second_policy);
    constexpr std::size_t kBf16 = sizeof(std::uint16_t);
    const std::int32_t cols       = x.ne[1];
    const std::int32_t qk_rows    = qk_weight.n;
    const std::int32_t value_rows = qkv.ne[0] - qk_rows;
    const std::int32_t z_rows     = z.ne[0];
    if (qk_weight.k != value_z_weight.k || qk_weight.k != x.ne[0]) {
        throw std::invalid_argument("gdn_input_proj pair: halves disagree on K");
    }
    if (value_rows <= 0 || value_z_weight.n != value_rows + z_rows) {
        throw std::invalid_argument(
            "gdn_input_proj pair: destination rows do not match the halves");
    }
    const std::size_t plane_pitch = static_cast<std::size_t>(qkv.ne[0]) * kBf16;
    {
        auto scope  = workspace.scope();
        Tensor head = workspace.alloc(DType::BF16, {qk_rows, cols});
        linear(x, qk_weight, head, first_policy, workspace, stream);
        const std::size_t bytes = static_cast<std::size_t>(qk_rows) * kBf16;
        CUDA_CHECK(cudaMemcpy2DAsync(qkv.data, plane_pitch, head.data, bytes, bytes,
                                     static_cast<std::size_t>(cols), cudaMemcpyDeviceToDevice,
                                     stream));
    }
    {
        auto scope = workspace.scope();
        Tensor tail = workspace.alloc(DType::BF16, {value_z_weight.n, cols});
        linear(x, value_z_weight, tail, second_policy, workspace, stream);
        const std::size_t tail_pitch = static_cast<std::size_t>(value_z_weight.n) * kBf16;
        const std::size_t value_bytes = static_cast<std::size_t>(value_rows) * kBf16;
        CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::byte*>(qkv.data) +
                                         static_cast<std::size_t>(qk_rows) * kBf16,
                                     plane_pitch, tail.data, tail_pitch, value_bytes,
                                     static_cast<std::size_t>(cols), cudaMemcpyDeviceToDevice, stream));
        const std::size_t z_bytes = static_cast<std::size_t>(z_rows) * kBf16;
        CUDA_CHECK(cudaMemcpy2DAsync(z.data, z_bytes,
                                     static_cast<std::byte*>(tail.data) + value_bytes,
                                     tail_pitch, z_bytes, static_cast<std::size_t>(cols),
                                     cudaMemcpyDeviceToDevice, stream));
    }
}

std::size_t pair_projection_capacity(QType qk_qtype, QType value_z_qtype, std::int32_t qk_rows,
                                     std::int32_t value_rows, std::int32_t z_rows,
                                     std::int32_t input_rows, LinearPolicy first_policy, LinearPolicy second_policy,
                                     std::int32_t min_tokens, std::int32_t max_tokens) {
    WorkspaceLayoutBuilder layout;
    {
        auto scope = layout.scope();
        (void)layout.alloc(DType::BF16, {qk_rows, max_tokens});
        (void)layout.alloc_bytes(linear_workspace_capacity_bytes(
            qk_qtype, qk_rows, input_rows, first_policy, min_tokens, max_tokens));
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc(DType::BF16, {value_rows + z_rows, max_tokens});
        (void)layout.alloc_bytes(linear_workspace_capacity_bytes(
            value_z_qtype, value_rows + z_rows, input_rows, second_policy, min_tokens, max_tokens));
    }
    return layout.peak_bytes(1);
}

/// Whether the pair is the registered Q4 q/k + Q5 value/z form, which has fused kernels.
bool pair_is_registered(const Weight& qk_weight, const Weight& value_z_weight) {
    return qk_weight.qtype == QType::Q4G64_F16S && value_z_weight.qtype == QType::Q5G64_F16S &&
           qk_weight.layout == QuantLayout::RowSplit &&
           value_z_weight.layout == QuantLayout::RowSplit &&
           qk_weight.n == 4096 && qk_weight.k == 5120 &&
           value_z_weight.n == 12288 && value_z_weight.k == 5120;
}

std::size_t split_projection_capacity(QType qtype, QType z_qtype, std::int32_t qkv_rows, std::int32_t z_rows,
                                      std::int32_t input_rows, LinearPolicy first_policy, LinearPolicy second_policy,
                                      std::int32_t min_tokens, std::int32_t max_tokens) {
    // The two GEMMs are sequential and each scopes its own scratch, so the peak is the larger.
    return std::max(linear_workspace_capacity_bytes(qtype, qkv_rows, input_rows, first_policy, min_tokens,
                                                    max_tokens),
                    linear_workspace_capacity_bytes(z_qtype, z_rows, input_rows, second_policy, min_tokens,
                                                    max_tokens));
}


} // namespace

void gdn_input_proj_split(const Tensor& x, const Weight& query_key_value_weight,
                          const Weight& z_weight, Tensor& qkv, Tensor& z, LinearPolicy first_policy, LinearPolicy second_policy,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    require_single_parent_nonoverlap(x, qkv, z);
    project_split(x, query_key_value_weight, z_weight, qkv, z, first_policy, second_policy, workspace, stream);
}

std::size_t gdn_input_proj_split_workspace_capacity_bytes(QType qtype, QType z_qtype, std::int32_t qkv_rows,
                                                          std::int32_t z_rows,
                                                          std::int32_t input_rows,
                                                          LinearPolicy first_policy, LinearPolicy second_policy,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens) {
    validate_policy(first_policy);
    validate_policy(second_policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("gdn_input_proj split workspace: invalid token interval");
    }
    return split_projection_capacity(qtype, z_qtype, qkv_rows, z_rows, input_rows, first_policy, second_policy, min_tokens,
                                     max_tokens);
}

void gdn_input_proj_conv_snapshot_split(
    const Tensor& x, const Weight& query_key_value_weight, const Weight& z_weight,
    const Tensor& conv_weight, Tensor& conv_states, const Tensor& valid_columns,
    const Tensor& initial_state_slots, const Tensor& snapshot_base_slots, Tensor& query,
    Tensor& key, Tensor& value, Tensor& z, LinearPolicy first_policy, LinearPolicy second_policy, WorkspaceArena& workspace,
    cudaStream_t stream) {
    validate_policy(first_policy);
    validate_policy(second_policy);
    const std::int32_t kQueryRows = query.ne[0];
    const std::int32_t kKeyRows   = key.ne[0];
    const std::int32_t kValueRows = value.ne[0];
    if (kQueryRows + kKeyRows + kValueRows != query_key_value_weight.n ||
        kValueRows != z_weight.n) {
        throw std::invalid_argument("gdn_input_proj split: output rows disagree with stored matrices");
    }
    const std::int32_t kChannels  = query_key_value_weight.n;
    const ConvGeometry geometry   = require_snapshot_input(x, query_key_value_weight.k);
    require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                              snapshot_base_slots, kChannels, geometry);
    require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "query");
    require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "key");
    require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "value");
    require_conv_tensor(z, z_weight.n, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "z");
    // One token, both halves K-quant: project and convolve in one pass per half, so the
    // projected plane and the convolution launch that consumed it both disappear. That launch
    // count is what made the split lose to the fused W8 parent at decode.
    if (detail::ggml_gdn_input_decode_admits(query_key_value_weight, z_weight, geometry.batch,
                                             geometry.width)) {
        auto fused_scope         = workspace.scope();
        const DeviceSpan fused_y = workspace.alloc_bytes(
            detail::ggml_gdn_input_decode_workspace_bytes(query_key_value_weight.k), 256);
        detail::ggml_gdn_input_conv_snapshot_decode_launch(
            x, query_key_value_weight, z_weight, conv_weight, conv_states, valid_columns,
            initial_state_slots, snapshot_base_slots, query, key, value, z, fused_y.data,
            fused_y.bytes, stream);
        return;
    }
    if (geometry.batch > 1) {
        compose_batched_snapshot(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                                 snapshot_base_slots, query, key, value, z, kQueryRows, kKeyRows,
                                 kValueRows, geometry, workspace, stream,
                                 [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                                     project_split(x_flat, query_key_value_weight, z_weight,
                                                   projected, z_flat, first_policy, second_policy, workspace, stream);
                                 });
        return;
    }
    auto scope                 = workspace.scope();
    ProjectedWorkspace scratch = allocate_projected_workspace(workspace, kChannels, geometry.width);
    Tensor z_flat              = flatten_columns(z, z.ne[0], geometry);
    project_split(x, query_key_value_weight, z_weight, scratch.projected, z_flat, first_policy, second_policy, workspace,
                  stream);
    detail::gdn_projected_conv_snapshot_launch(scratch.projected, conv_weight, conv_states,
                                               valid_columns, initial_state_slots,
                                               snapshot_base_slots, query, key, value, stream);
}

std::size_t gdn_input_proj_conv_snapshot_split_workspace_capacity_bytes(
    QType qtype, QType z_qtype, std::int32_t qkv_rows, std::int32_t z_rows, std::int32_t input_rows,
    LinearPolicy first_policy, LinearPolicy second_policy, std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    validate_policy(first_policy);
    validate_policy(second_policy);
    require_snapshot_capacity_domain(batch_size, min_width, max_width);
    const std::int32_t widest = batch_size * max_width;
    std::size_t plane_bytes   = 0;
    if (batch_size > 1) {
        plane_bytes = composed_snapshot_capacity(qkv_rows, widest, 0);
    } else {
        WorkspaceLayoutBuilder layout;
        (void)allocate_projected_workspace(layout, qkv_rows, max_width);
        plane_bytes = layout.peak_bytes(1);
    }
    return plane_bytes + split_projection_capacity(qtype, z_qtype, qkv_rows, z_rows, input_rows, first_policy, second_policy,
                                                   batch_size * min_width, widest);
}

void gdn_input_proj_pair(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                         Tensor& qkv, Tensor& z, LinearPolicy first_policy, LinearPolicy second_policy, WorkspaceArena& workspace,
                         cudaStream_t stream) {
    validate_policy(first_policy);
    validate_policy(second_policy);
    if (x.ne[1] <= 0) { throw std::invalid_argument("gdn_input_proj pair: T must be positive"); }
    if (pair_is_registered(qk_weight, value_z_weight)) {
        gdn_input_proj(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    }
    project_pair(x, qk_weight, value_z_weight, qkv, z, first_policy, second_policy, workspace, stream);
}

std::size_t gdn_input_proj_pair_workspace_capacity_bytes(
    QType qk_qtype, QType value_z_qtype, std::int32_t qk_rows, std::int32_t value_rows,
    std::int32_t z_rows, std::int32_t input_rows, LinearPolicy first_policy, LinearPolicy second_policy, std::int32_t min_tokens,
    std::int32_t max_tokens) {
    validate_policy(first_policy);
    validate_policy(second_policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("gdn_input_proj pair workspace: invalid token interval");
    }
    return pair_projection_capacity(qk_qtype, value_z_qtype, qk_rows, value_rows, z_rows,
                                    input_rows, first_policy, second_policy, min_tokens, max_tokens);
}

void gdn_input_proj_conv_snapshot_pair(
    const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
    const Tensor& conv_weight, Tensor& conv_states, const Tensor& valid_columns,
    const Tensor& initial_state_slots, const Tensor& snapshot_base_slots, Tensor& query,
    Tensor& key, Tensor& value, Tensor& z, LinearPolicy first_policy, LinearPolicy second_policy, WorkspaceArena& workspace,
    cudaStream_t stream) {
    validate_policy(first_policy);
    validate_policy(second_policy);
    if (pair_is_registered(qk_weight, value_z_weight)) {
        gdn_input_proj_conv_snapshot(x, qk_weight, value_z_weight, conv_weight, conv_states,
                                     valid_columns, initial_state_slots, snapshot_base_slots,
                                     query, key, value, z, workspace, stream);
        return;
    }
    // Row counts come from the operands: this form serves whatever geometry the artifact
    // declares, the way the K-quant single-parent branch does.
    const std::int32_t kQueryRows = query.ne[0];
    const std::int32_t kKeyRows   = key.ne[0];
    const std::int32_t kValueRows = value.ne[0];
    const std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
    const ConvGeometry geometry   = require_snapshot_input(x, qk_weight.k);
    require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                              snapshot_base_slots, kChannels, geometry);
    require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "query");
    require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "key");
    require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "value");
    require_conv_tensor(z, z.ne[0], geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "z");
    if (geometry.batch > 1) {
        compose_batched_snapshot(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                                 snapshot_base_slots, query, key, value, z, kQueryRows, kKeyRows,
                                 kValueRows, geometry, workspace, stream,
                                 [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                                     project_pair(x_flat, qk_weight, value_z_weight, projected,
                                                  z_flat, first_policy, second_policy, workspace, stream);
                                 });
        return;
    }
    auto scope                 = workspace.scope();
    ProjectedWorkspace scratch = allocate_projected_workspace(workspace, kChannels, geometry.width);
    project_pair(x, qk_weight, value_z_weight, scratch.projected, z, first_policy, second_policy, workspace, stream);
    detail::gdn_projected_conv_snapshot_launch(scratch.projected, conv_weight, conv_states,
                                               valid_columns, initial_state_slots,
                                               snapshot_base_slots, query, key, value, stream);
}

std::size_t gdn_input_proj_conv_snapshot_pair_workspace_capacity_bytes(
    QType qk_qtype, QType value_z_qtype, std::int32_t qk_rows, std::int32_t value_rows,
    std::int32_t z_rows, std::int32_t input_rows, LinearPolicy first_policy, LinearPolicy second_policy, std::int32_t batch_size,
    std::int32_t min_width, std::int32_t max_width) {
    validate_policy(first_policy);
    validate_policy(second_policy);
    require_snapshot_capacity_domain(batch_size, min_width, max_width);
    const std::int32_t widest   = batch_size * max_width;
    const std::int32_t channels = qk_rows + value_rows;
    std::size_t plane_bytes     = 0;
    if (batch_size > 1) {
        plane_bytes = composed_snapshot_capacity(channels, widest, 0);
    } else {
        WorkspaceLayoutBuilder layout;
        (void)allocate_projected_workspace(layout, channels, max_width);
        plane_bytes = layout.peak_bytes(1);
    }
    return plane_bytes + pair_projection_capacity(qk_qtype, value_z_qtype, qk_rows, value_rows,
                                                  z_rows, input_rows, first_policy, second_policy,
                                                  batch_size * min_width, widest);
}

void gdn_input_proj_conv_record_pair(
    const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
    const Tensor& conv_weight, const Tensor& conv_states, const Tensor& valid_columns,
    const Tensor& initial_state_slots, Tensor& conv_record, Tensor& query, Tensor& key,
    Tensor& value, Tensor& z, LinearPolicy first_policy, LinearPolicy second_policy, WorkspaceArena& workspace,
    cudaStream_t stream) {
    validate_policy(first_policy);
    validate_policy(second_policy);
    if (pair_is_registered(qk_weight, value_z_weight)) {
        gdn_input_proj_conv_record(x, qk_weight, value_z_weight, conv_weight, conv_states,
                                   valid_columns, initial_state_slots, conv_record, query, key,
                                   value, z, workspace, stream);
        return;
    }
    const std::int32_t kQueryRows = query.ne[0];
    const std::int32_t kKeyRows   = key.ne[0];
    const std::int32_t kValueRows = value.ne[0];
    const std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
    const ConvGeometry geometry   = require_record_input(x, qk_weight.k);
    require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots, kChannels,
                            geometry);
    require_conv_tensor(conv_record, kChannels, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "conv record");
    require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "query");
    require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "key");
    require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "value");
    require_conv_tensor(z, z.ne[0], geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                        "z");
    auto scope         = workspace.scope();
    Tensor x_flat      = flatten_columns(x, x.ne[0], geometry);
    Tensor record_flat = flatten_columns(conv_record, conv_record.ne[0], geometry);
    Tensor z_flat      = flatten_columns(z, z.ne[0], geometry);
    project_pair(x_flat, qk_weight, value_z_weight, record_flat, z_flat, first_policy, second_policy, workspace, stream);
    detail::gdn_projected_conv_record_launch(conv_record, conv_weight, conv_states, valid_columns,
                                             initial_state_slots, query, key, value, stream);
}

std::size_t gdn_input_proj_conv_record_pair_workspace_capacity_bytes(
    QType qk_qtype, QType value_z_qtype, std::int32_t qk_rows, std::int32_t value_rows,
    std::int32_t z_rows, std::int32_t input_rows, LinearPolicy first_policy, LinearPolicy second_policy, std::int32_t batch_size,
    std::int32_t min_width, std::int32_t max_width) {
    validate_policy(first_policy);
    validate_policy(second_policy);
    require_snapshot_capacity_domain(batch_size, min_width, max_width);
    const std::int32_t widest = batch_size * max_width;
    return pair_projection_capacity(qk_qtype, value_z_qtype, qk_rows, value_rows, z_rows,
                                    input_rows, first_policy, second_policy, batch_size * min_width, widest);
}

void gdn_input_proj_conv_record_split(
    const Tensor& x, const Weight& query_key_value_weight, const Weight& z_weight,
    const Tensor& conv_weight, const Tensor& conv_states, const Tensor& valid_columns,
    const Tensor& initial_state_slots, Tensor& conv_record, Tensor& query, Tensor& key,
    Tensor& value, Tensor& z, LinearPolicy first_policy, LinearPolicy second_policy, WorkspaceArena& workspace,
    cudaStream_t stream) {
    validate_policy(first_policy);
    validate_policy(second_policy);
    const std::int32_t kQueryRows = query.ne[0];
    const std::int32_t kKeyRows   = key.ne[0];
    const std::int32_t kValueRows = value.ne[0];
    if (kQueryRows + kKeyRows + kValueRows != query_key_value_weight.n ||
        kValueRows != z_weight.n) {
        throw std::invalid_argument("gdn_input_proj split: output rows disagree with stored matrices");
    }
    const std::int32_t kChannels  = query_key_value_weight.n;
    const ConvGeometry geometry   = require_record_input(x, query_key_value_weight.k);
    require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots, kChannels,
                            geometry);
    require_conv_tensor(conv_record, kChannels, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "conv record");
    require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "query");
    require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "key");
    require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "value");
    require_conv_tensor(z, z_weight.n, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "z");
    // One token, both halves K-quant: project and convolve in one pass per half, so the
    // projected plane and the convolution launch that consumed it both disappear. That launch
    // count is what made the split lose to the fused W8 parent at decode.
    if (detail::ggml_gdn_input_decode_admits(query_key_value_weight, z_weight, geometry.batch,
                                             geometry.width)) {
        auto fused_scope         = workspace.scope();
        const DeviceSpan fused_y = workspace.alloc_bytes(
            detail::ggml_gdn_input_decode_workspace_bytes(query_key_value_weight.k), 256);
        detail::ggml_gdn_input_conv_record_decode_launch(
            x, query_key_value_weight, z_weight, conv_weight, conv_states, valid_columns,
            initial_state_slots, conv_record, query, key, value, z, fused_y.data, fused_y.bytes,
            stream);
        return;
    }
    auto scope         = workspace.scope();
    Tensor x_flat      = flatten_columns(x, x.ne[0], geometry);
    Tensor record_flat = flatten_columns(conv_record, conv_record.ne[0], geometry);
    Tensor z_flat      = flatten_columns(z, z.ne[0], geometry);
    project_split(x_flat, query_key_value_weight, z_weight, record_flat, z_flat, first_policy, second_policy, workspace,
                  stream);
    detail::gdn_projected_conv_record_launch(conv_record, conv_weight, conv_states, valid_columns,
                                             initial_state_slots, query, key, value, stream);
}

std::size_t gdn_input_proj_conv_record_split_workspace_capacity_bytes(
    QType qtype, QType z_qtype, std::int32_t qkv_rows, std::int32_t z_rows, std::int32_t input_rows,
    LinearPolicy first_policy, LinearPolicy second_policy, std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    validate_policy(first_policy);
    validate_policy(second_policy);
    require_snapshot_capacity_domain(batch_size, min_width, max_width);
    // The record plane is the caller's; only the two projections need scratch.
    return split_projection_capacity(qtype, z_qtype, qkv_rows, z_rows, input_rows, first_policy, second_policy,
                                     batch_size * min_width, batch_size * max_width);
}

void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    cudaStream_t stream) {
    dispatch_single_parent(x, query_key_value_z_weight, qkv, z, LinearPolicy::A16Only, nullptr,
                           stream);
}

std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    std::int32_t query_rows, std::int32_t key_rows, std::int32_t value_rows,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    const bool q4_q5 = query_rows == 2048 && key_rows == 2048 && value_rows == 6144;
    // surogate vendor patch (PATCHES.md #13): value_rows 2048 = qwen3.5-0.8b (16 V heads).
    const bool w8 = query_rows == 2048 && key_rows == 2048 &&
                    (value_rows == 4096 || value_rows == 2048);
    if (!q4_q5 && !w8) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot workspace: unregistered shape");
    }
    require_snapshot_capacity_domain(batch_size, min_width, max_width);
    const std::int32_t channels = query_rows + key_rows + value_rows;
    if (batch_size > 1) {
        if (q4_q5) {
            (void)resolve_q4_q5_conv_plan(min_width, batch_size);
            (void)resolve_q4_q5_conv_plan(max_width, batch_size);
        } else {
            (void)resolve_w8_conv_plan(min_width, batch_size);
            (void)resolve_w8_conv_plan(max_width, batch_size);
        }
        return composed_snapshot_capacity(channels, batch_size * max_width, 0);
    }

    std::int32_t largest_materialized_width = 0;
    if (q4_q5) {
        (void)resolve_q4_q5_conv_plan(min_width, 1);
        (void)resolve_q4_q5_conv_plan(max_width, 1);
        if (max_width >= 7) {
            largest_materialized_width = max_width;
        } else if (min_width <= 4 && max_width >= 4) {
            largest_materialized_width = 4;
        }
    } else {
        (void)resolve_w8_conv_plan(min_width, 1);
        (void)resolve_w8_conv_plan(max_width, 1);
        if (max_width >= 17) { largest_materialized_width = max_width; }
    }
    if (largest_materialized_width == 0) { return 0; }
    WorkspaceLayoutBuilder layout;
    (void)allocate_projected_workspace(layout, channels, largest_materialized_width);
    return layout.peak_bytes(1);
}

std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    QType parent_qtype, std::int32_t parent_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    validate_policy(policy);
    if (parent_qtype == QType::BF16_CTRL && policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("BF16 input projection admits only A16");
    }
    require_snapshot_capacity_domain(batch_size, min_width, max_width);
    if (row_projectable(parent_qtype)) {
        // The split is read off the output views at run time; here only the parent is known,
        // so the projected plane is sized for every parent row (z included) -- sufficient,
        // and a few MiB over at most -- plus the int8 activation scratch.
        const std::int32_t widest = batch_size * max_width;
        return composed_snapshot_capacity(
            parent_rows, widest,
            row_projectable_workspace_capacity_bytes(parent_qtype, parent_rows, input_rows, widest));
    }
    if (parent_qtype == QType::FP8_E4M3FN_ROW_BF16S &&
        parent_rows == detail::Fp8GdnInputGeometry::kOutputRows &&
        input_rows == detail::Fp8GdnInputGeometry::kInputRows &&
        (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8)) {
        return detail::fp8_gdn_snapshot_workspace_capacity_bytes(policy, batch_size, min_width,
                                                                 max_width);
    }
    // surogate vendor patch (PATCHES.md #13): W8 fused parents (35B 12288/2048,
    // qwen3.5-0.8b 8192/1024) size through the split-dimension path.
    if (parent_qtype == QType::W8G32_F16S &&
        (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) &&
        ((parent_rows == 12288 && (input_rows == 2048 || input_rows == 2560)) ||
         (parent_rows == 8192 && (input_rows == 1024 || input_rows == 2048)))) {
        const std::int32_t value_rows = parent_rows == 12288 ? 4096 : 2048;
        return gdn_input_proj_conv_snapshot_workspace_capacity_bytes(2048, 2048, value_rows,
                                                                     batch_size, min_width,
                                                                     max_width);
    }
    if (parent_qtype == QType::NVFP4 &&
        (parent_rows != detail::Nvfp4GdnInputGeometry::kOutputRows ||
         input_rows != detail::Nvfp4GdnInputGeometry::kInputRows) &&
        detail::is_nvfp4_generic_problem(parent_rows, input_rows)) {
        // Project-then-conv: the BF16 plane plus the W4A4 scratch the projection needs. The
        // plane is unconditional here — unlike W8, NVFP4 has no fused conv kernel to fall back
        // on at narrow widths, so the split-dimension helper's zero would be wrong (#84).
        // The metadata query does not include the output split. The full parent bounds
        // the convolution plane for any valid q/k/value/z widths.
        const std::int32_t channels = parent_rows;
        const std::int32_t widest     = batch_size * max_width;
        std::size_t plane_bytes       = 0;
        if (batch_size > 1) {
            plane_bytes = composed_snapshot_capacity(channels, widest, 0);
        } else {
            WorkspaceLayoutBuilder layout;
            (void)allocate_projected_workspace(layout, channels, max_width);
            plane_bytes = layout.peak_bytes(1);
        }
        return plane_bytes + detail::nvfp4_gdn_input_workspace_capacity_bytes(
                                 policy, batch_size * min_width, widest, parent_rows, input_rows);
    }
    if (parent_qtype != QType::NVFP4 || parent_rows != detail::Nvfp4GdnInputGeometry::kOutputRows ||
        input_rows != detail::Nvfp4GdnInputGeometry::kInputRows) {
        throw std::invalid_argument(
            "gdn_input_proj_conv_snapshot workspace: unsupported single-parent profile");
    }
    if (batch_size == 1) {
        return detail::nvfp4_gdn_snapshot_workspace_capacity_bytes(policy, min_width, max_width);
    }

    (void)detail::nvfp4_gdn_conv_resolve_plan(policy, min_width, batch_size);
    (void)detail::nvfp4_gdn_conv_resolve_plan(policy, max_width, batch_size);

    constexpr std::int32_t kChannels       = 10240;
    const std::int32_t aggregate_columns   = batch_size * max_width;
    const std::size_t projection_workspace = gdn_input_proj_workspace_capacity_bytes(
        parent_qtype, parent_rows, input_rows, policy, batch_size * min_width, aggregate_columns);
    return composed_snapshot_capacity(kChannels, aggregate_columns, projection_workspace);
}

std::size_t gdn_input_proj_conv_record_workspace_capacity_bytes(
    std::int32_t query_rows, std::int32_t key_rows, std::int32_t value_rows,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    const bool q4_q5 = query_rows == 2048 && key_rows == 2048 && value_rows == 6144;
    // surogate vendor patch (PATCHES.md #13): value_rows 2048 = qwen3.5-0.8b (16 V heads).
    const bool w8 = query_rows == 2048 && key_rows == 2048 &&
                    (value_rows == 4096 || value_rows == 2048);
    if (!q4_q5 && !w8) {
        throw std::invalid_argument("gdn_input_proj_conv_record workspace: unregistered shape");
    }
    require_record_capacity_domain(batch_size, min_width, max_width);
    if (q4_q5) {
        (void)resolve_q4_q5_conv_plan(min_width, batch_size);
        (void)resolve_q4_q5_conv_plan(max_width, batch_size);
    } else {
        (void)resolve_w8_conv_plan(min_width, batch_size);
        (void)resolve_w8_conv_plan(max_width, batch_size);
    }
    return 0;
}

std::size_t gdn_input_proj_conv_record_workspace_capacity_bytes(
    QType parent_qtype, std::int32_t parent_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    validate_policy(policy);
    if (parent_qtype == QType::BF16_CTRL && policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("BF16 input projection admits only A16");
    }
    require_record_capacity_domain(batch_size, min_width, max_width);
    if (row_projectable(parent_qtype)) {
        return row_projectable_workspace_capacity_bytes(parent_qtype, parent_rows, input_rows,
                                                          batch_size * max_width);
    }
    if (parent_qtype == QType::FP8_E4M3FN_ROW_BF16S &&
        parent_rows == detail::Fp8GdnInputGeometry::kOutputRows &&
        input_rows == detail::Fp8GdnInputGeometry::kInputRows &&
        (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8)) {
        return detail::fp8_gdn_record_workspace_capacity_bytes(policy, batch_size, min_width,
                                                               max_width);
    }
    // surogate vendor patch (PATCHES.md #13/#15): W8 fused parents (35B
    // 12288/2048, qwen3.5-0.8b 8192/1024) size through the split-dimension
    // record path, mirroring the snapshot overload above.
    if (parent_qtype == QType::W8G32_F16S &&
        (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) &&
        ((parent_rows == 12288 && (input_rows == 2048 || input_rows == 2560)) ||
         (parent_rows == 8192 && (input_rows == 1024 || input_rows == 2048)))) {
        const std::int32_t value_rows = parent_rows == 12288 ? 4096 : 2048;
        return gdn_input_proj_conv_record_workspace_capacity_bytes(2048, 2048, value_rows,
                                                                   batch_size, min_width,
                                                                   max_width);
    }
    if (parent_qtype == QType::NVFP4 &&
        (parent_rows != detail::Nvfp4GdnInputGeometry::kOutputRows ||
         input_rows != detail::Nvfp4GdnInputGeometry::kInputRows) &&
        detail::is_nvfp4_generic_problem(parent_rows, input_rows) &&
        (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA4)) {
        // The record plane is the caller's; only the projection scratch is ours (#84).
        return detail::nvfp4_gdn_input_workspace_capacity_bytes(policy, batch_size * min_width,
                                                                batch_size * max_width, parent_rows,
                                                                input_rows);
    }
    if (parent_qtype != QType::NVFP4 || parent_rows != detail::Nvfp4GdnInputGeometry::kOutputRows ||
        input_rows != detail::Nvfp4GdnInputGeometry::kInputRows ||
        (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
        throw std::invalid_argument(
            "gdn_input_proj_conv_record workspace: unsupported single-parent profile");
    }
    const detail::Nvfp4GdnConvPlan minimum_plan =
        detail::nvfp4_gdn_conv_resolve_plan(policy, min_width, batch_size);
    const detail::Nvfp4GdnConvPlan maximum_plan =
        detail::nvfp4_gdn_conv_resolve_plan(policy, max_width, batch_size);
    if (batch_size == 1) {
        if (minimum_plan.schedule == detail::Nvfp4GdnConvScheduleId::DecodeFusedA16) {
            throw std::logic_error("ReplaySSM record planner admitted NVFP4 decode");
        }
        if (maximum_plan.schedule == detail::Nvfp4GdnConvScheduleId::SmallTFusedA16) { return 0; }
        return detail::nvfp4_gdn_input_workspace_capacity_bytes(
            LinearPolicy::AllowA4, std::max(min_width, 4), max_width, parent_rows, input_rows);
    }
    return detail::nvfp4_gdn_input_workspace_capacity_bytes(
        policy, batch_size * min_width, batch_size * max_width, parent_rows, input_rows);
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& qk_weight,
                                  const Weight& value_z_weight, const Tensor& conv_weight,
                                  Tensor& conv_states, const Tensor& valid_columns,
                                  const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQueryRows  = 2048;
    constexpr std::int32_t kKeyRows    = 2048;
    constexpr std::int32_t kValueRows  = 6144;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
    constexpr std::int32_t kParentRows = kValueRows + kZRows;
    const ConvGeometry geometry        = require_snapshot_input(x, kHidden);
    require_rowsplit(qk_weight, QType::Q4G64_F16S, kQueryRows + kKeyRows, "qk weight");
    require_rowsplit(value_z_weight, QType::Q5G64_F16S, kParentRows, "value/z weight");
    require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                              snapshot_base_slots, kChannels, geometry);
    require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "query");
    require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "key");
    require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "value");
    require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_snapshot",
                        "z");

    if (geometry.batch > 1) {
        compose_batched_snapshot(
            x, conv_weight, conv_states, valid_columns, initial_state_slots, snapshot_base_slots,
            query, key, value, z, kQueryRows, kKeyRows, kValueRows, geometry, ws, stream,
            [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                gdn_input_proj(x_flat, qk_weight, value_z_weight, projected, z_flat, stream);
            });
        return;
    }

    const detail::Q4Q5GdnInputConvPlan plan =
        resolve_q4_q5_conv_plan(geometry.width, geometry.batch);
    if (plan.schedule == detail::Q4Q5GdnInputConvScheduleId::ProjectionEpilogueFused) {
        detail::q4_q5_gdn_input_conv_snapshot_launch(
            x, qk_weight, value_z_weight, conv_weight, conv_states, valid_columns,
            initial_state_slots, snapshot_base_slots, query, key, value, z, stream);
        return;
    }

    auto scope                 = ws.scope();
    ProjectedWorkspace scratch = allocate_projected_workspace(ws, kChannels, geometry.width);
    gdn_input_proj(x, qk_weight, value_z_weight, scratch.projected, z, stream);
    detail::gdn_projected_conv_snapshot_launch(scratch.projected, conv_weight, conv_states,
                                               valid_columns, initial_state_slots,
                                               snapshot_base_slots, query, key, value, stream);
}

void gdn_input_proj_conv_record(const Tensor& x, const Weight& qk_weight,
                                const Weight& value_z_weight, const Tensor& conv_weight,
                                const Tensor& conv_states, const Tensor& valid_columns,
                                const Tensor& initial_state_slots, Tensor& conv_record,
                                Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQueryRows  = 2048;
    constexpr std::int32_t kKeyRows    = 2048;
    constexpr std::int32_t kValueRows  = 6144;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
    constexpr std::int32_t kParentRows = kValueRows + kZRows;
    const ConvGeometry geometry        = require_record_input(x, kHidden);
    require_rowsplit(qk_weight, QType::Q4G64_F16S, kQueryRows + kKeyRows, "qk weight");
    require_rowsplit(value_z_weight, QType::Q5G64_F16S, kParentRows, "value/z weight");
    require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots, kChannels,
                            geometry);
    require_conv_tensor(conv_record, kChannels, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "conv record");
    require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "query");
    require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                        "key");
    require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "value");
    require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                        "z");
    require_record_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                              conv_record, query, key, value, z, workspace);

    const detail::Q4Q5GdnInputConvPlan plan =
        resolve_q4_q5_conv_plan(geometry.width, geometry.batch);
    if (plan.schedule == detail::Q4Q5GdnInputConvScheduleId::ProjectionEpilogueFused) {
        detail::q4_q5_gdn_input_conv_record_launch(x, qk_weight, value_z_weight, conv_weight,
                                                   conv_states, valid_columns, initial_state_slots,
                                                   conv_record, query, key, value, z, stream);
        return;
    }
    compose_record(x, conv_weight, conv_states, valid_columns, initial_state_slots, conv_record,
                   query, key, value, z, geometry, workspace, stream,
                   [&](const Tensor& x_flat, Tensor& record_flat, Tensor& z_flat) {
                       gdn_input_proj(x_flat, qk_weight, value_z_weight, record_flat, z_flat,
                                      stream);
                   });
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, LinearPolicy policy, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    dispatch_single_parent_snapshot(x, query_key_value_z_weight, conv_weight, conv_states,
                                    valid_columns, initial_state_slots, snapshot_base_slots, query,
                                    key, value, z, policy, ws, stream);
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    dispatch_single_parent_snapshot(x, query_key_value_z_weight, conv_weight, conv_states,
                                    valid_columns, initial_state_slots, snapshot_base_slots, query,
                                    key, value, z, LinearPolicy::A16Only, ws, stream);
}

void gdn_input_proj_conv_record(const Tensor& x, const Weight& query_key_value_z_weight,
                                const Tensor& conv_weight, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_state_slots,
                                Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                Tensor& z, LinearPolicy policy, WorkspaceArena& workspace,
                                cudaStream_t stream) {
    dispatch_single_parent_record(x, query_key_value_z_weight, conv_weight, conv_states,
                                  valid_columns, initial_state_slots, conv_record, query, key,
                                  value, z, policy, workspace, stream);
}

void gdn_input_proj_conv_record(const Tensor& x, const Weight& query_key_value_z_weight,
                                const Tensor& conv_weight, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_state_slots,
                                Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    dispatch_single_parent_record(x, query_key_value_z_weight, conv_weight, conv_states,
                                  valid_columns, initial_state_slots, conv_record, query, key,
                                  value, z, LinearPolicy::A16Only, workspace, stream);
}

} // namespace sinfer::ops
