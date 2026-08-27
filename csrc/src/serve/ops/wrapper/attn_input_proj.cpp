#include "api/ops/attn_input_proj.h"
#include "ops/linear/w8a8/w4fp4_plane.h"

#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"
#include "core/device.h"
#include "ops/linear/marlin/marlin_plane.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"
#include "ops/attn_input_proj/w8/w8_attn_input_plan.h"
#include "ops/linear/w8a8/w8a8_dispatch.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {


bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t cols, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != cols ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
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
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_w8_rowsplit(const Weight& weight, std::int32_t rows, const char* label) {
    if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 32 || weight.group != 32 ||
        // surogate vendor patch (PATCHES.md #18): K 2560 = qwen3.5-4b.
        weight.ndim != 2 || weight.n != rows ||
        (weight.k != 2048 && weight.k != 1024 && weight.k != 2560) ||
        weight.shape[0] != rows || weight.shape[1] != weight.k || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != weight.k || weight.qhigh != nullptr ||
        weight.high_plane_bytes != 0 || !aligned_to(weight.qdata, 16) ||
        !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_bf16_contiguous(const Weight& weight, std::int32_t rows, std::int32_t hidden,
                             const char* label) {
    const std::uint64_t payload_bytes = static_cast<std::uint64_t>(rows) *
                                        static_cast<std::uint64_t>(hidden) * sizeof(std::uint16_t);
    if (weight.qtype != QType::BF16_CTRL || weight.layout != QuantLayout::Contiguous ||
        weight.payload_bytes < payload_bytes || weight.high_plane_bytes != 0 || weight.ndim != 2 ||
        weight.n != rows || weight.k != hidden || weight.shape[0] != rows ||
        weight.shape[1] != hidden || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != hidden || weight.qhigh != nullptr || weight.scales != nullptr ||
        weight.group_size != 0 || weight.group != 0 || !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("attn_input_proj: invalid compute policy");
}

void dispatch_single_parent(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                            cudaStream_t stream) {
    validate_policy(policy);
    if (weight.qtype == QType::BF16_CTRL) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("BF16 attn_input_proj admits only A16");
        }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        require_bf16_contiguous(weight, kRows, kHidden, "query/key/gate/value weight");
        detail::bf16_attn_input_dispatch(x, weight, q, gate, k, v, stream);
        return;
    }

    if (weight.qtype == QType::NVFP4) {
        // Keyed on parent rows the way the W8 branch below is: 14336 is the 27B, 10240 the
        // 4B (q,k,gate,v), 5120 the small fused parent (#84).
        const bool registered      = weight.n == 14336;
        const bool small_fused     = weight.n == 5120;
        const bool q4b             = weight.n == 10240;
        const std::int32_t kHidden = registered ? 5120 : weight.k;
        const std::int32_t kQRows  = registered ? 6144 : (small_fused ? 2048 : 4096);
        const std::int32_t kKvRows = (registered || q4b) ? 1024 : 512;
        const std::int32_t kRows   = weight.n;
        const std::int32_t cols    = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
            throw std::invalid_argument("NVFP4 attn_input_proj admits only A16 or A4");
        }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        detail::validate_nvfp4_weight(weight, "nvfp4 attn_input_proj");
        if (weight.n != kRows || weight.k != kHidden ||
            2 * kQRows + 2 * kKvRows != kRows ||
            (!registered && !detail::is_nvfp4_generic_problem(weight.n, weight.k))) {
            throw std::invalid_argument("nvfp4 attn_input_proj: unsupported weight shape");
        }
        detail::nvfp4_attn_input_dispatch(x, weight, q, gate, k, v, policy, workspace, stream);
        return;
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
            throw std::invalid_argument("FP8 attn_input_proj admits only A16 or A8");
        }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("fp8 attn_input_proj: unsupported weight shape");
        }
        detail::validate_fp8_weight(weight, "fp8 attn_input_proj");
        detail::fp8_attn_input_dispatch(x, weight, q, gate, k, v, policy, workspace, stream);
        return;
    }

    // surogate vendor patches (PATCHES.md #13/#16): the small fused parent
    // (5120 rows) covers the 0.8b (hidden 1024) and 2b (hidden 2048); the 35B
    // parent has 9216 rows. Keyed on parent rows, not hidden.
    const bool small_fused     = weight.n == 5120;
    const bool q4b             = weight.n == 10240;
    const std::int32_t kHidden = weight.k;
    const std::int32_t kQRows  = small_fused ? 2048 : 4096;
    const std::int32_t kKvRows = q4b ? 1024 : 512;
    const std::int32_t kRows   = weight.n;
    const std::int32_t cols        = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
    if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
        // surogate vendor patch (PATCHES.md #17): AllowA8 accepted; this
        // family executes A16 until its IMMA path lands.
        throw std::invalid_argument("W8 attn_input_proj admits A16 or A8");
    }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(gate, kQRows, cols, "gate");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_w8_rowsplit(weight, kRows, "query/key/gate/value weight");
    // surogate vendor patch (PATCHES.md #17): large-T prefill under AllowA8
    // runs the W8A8-int IMMA path (split4 epilogue, fused row order q|k|gate|v).
    if (policy == LinearPolicy::AllowA8 && cols >= detail::kW8A8MinTokens &&
        workspace != nullptr) {
        detail::w8a8_gemm_split4(x, weight, q, k, gate, v, *workspace, stream);
        return;
    }
    detail::w8_attn_input_dispatch(x, weight, q, gate, k, v, stream);
}

} // namespace

std::size_t attn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                                     std::int32_t input_rows, LinearPolicy policy,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("attn_input_proj workspace: invalid token interval");
    }

    switch (parent_qtype) {
    case QType::BF16_CTRL:
        if (parent_rows != 14336 || input_rows != 5120 || policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported BF16 profile");
        }
        return 0;
    case QType::NVFP4: {
        const bool registered = parent_rows == detail::Nvfp4AttnInputGeometry::kOutputRows &&
                                input_rows == detail::Nvfp4AttnInputGeometry::kInputRows;
        // Shapes outside the registered geometry run on the cuBLASLt route (#84).
        if ((!registered && !detail::is_nvfp4_generic_problem(parent_rows, input_rows)) ||
            (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_attn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens,
                                                                 parent_rows, input_rows);
    }
    case QType::FP8_E4M3FN_ROW_BF16S:
        if (parent_rows != detail::Fp8AttnInputGeometry::kOutputRows ||
            input_rows != detail::Fp8AttnInputGeometry::kInputRows) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported FP8 profile");
        }
        return detail::fp8_attn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    case QType::W8G32_F16S:
        if (!((parent_rows == 9216 && input_rows == 2048) ||
              (parent_rows == 5120 &&
               (input_rows == 1024 || input_rows == 2048)) ||
              (parent_rows == 10240 && input_rows == 2560)) ||
            (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8)) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported W8 profile");
        }
        {
            // surogate vendor patch (PATCHES.md #18): qwen3.5-4b qkgv is
            // 10240 rows (q4096 | k1024 | gate4096 | v1024) at hidden 2560.
            const std::int32_t q_rows  = parent_rows == 5120 ? 2048 : 4096;
            const std::int32_t kv_rows = parent_rows == 10240 ? 1024 : 512;
            (void)detail::w8_attn_input_resolve_plan(
                {input_rows, q_rows, kv_rows, parent_rows, input_rows, min_tokens});
            (void)detail::w8_attn_input_resolve_plan(
                {input_rows, q_rows, kv_rows, parent_rows, input_rows, max_tokens});
        }
        // surogate vendor patch (PATCHES.md #17): AllowA8 large-T runs the
        // W8A8-int IMMA path, which needs quantized-activation workspace.
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
    case QType::Q4G64_F16S:
    case QType::Q5G64_F16S:
    case QType::Q6G64_F16S:
    case QType::FP32_CTRL:
    case QType::I32_CTRL:
        break;
    }
    throw std::invalid_argument("attn_input_proj workspace: unsupported parent qtype");
}

void attn_input_proj(const Tensor& x, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     cudaStream_t stream) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    const std::int32_t cols        = x.ne[1];
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(gate, kQRows, cols, "gate");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_rowsplit(query_key_weight, QType::Q4G64_F16S, kQRows + kKvRows, "query/key weight");
    require_rowsplit(gate_value_weight, QType::Q5G64_F16S, kQRows + kKvRows, "gate/value weight");

    detail::q4_q5_attn_input_dispatch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                      stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                     WorkspaceArena& workspace, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_gate_value_weight, q, gate, k, v, policy, &workspace,
                           stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_gate_value_weight, q, gate, k, v, LinearPolicy::A16Only,
                           nullptr, stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_value_weight, Tensor& q, Tensor& k,
                     Tensor& v, cudaStream_t stream) {
    constexpr std::int32_t kHidden = 2048;
    constexpr std::int32_t kQRows  = 4096;
    constexpr std::int32_t kKvRows = 1024;
    constexpr std::int32_t kRows   = 6144;
    const std::int32_t cols        = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_w8_rowsplit(query_key_value_weight, kRows, "query/key/value weight");

    detail::w8_attn_input_dispatch(x, query_key_value_weight, q, k, v, stream);
}

} // namespace ninfer::ops
