#include "api/ops/linear_swiglu.h"

#include "api/ops/silu_mul.h"
#include "ops/linear/marlin/marlin_plane.h"

#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/w8/w8_linear_swiglu_plan.h"
#include "ops/linear_swiglu/w8a8/w8a8_linear_swiglu.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear_swiglu: invalid compute policy");
}

} // namespace

std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                   std::int32_t input_rows, LinearPolicy policy,
                                                   std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens || (gate_up_rows % 2) != 0) {
        throw std::invalid_argument("linear_swiglu workspace: invalid profile or token interval");
    }
    if (qtype == QType::W8G32_F16S) {
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
            throw std::invalid_argument("linear_swiglu workspace: W8 admits A16 or A8");
        }
        (void)detail::w8_linear_swiglu_resolve_plan(
            {gate_up_rows, gate_up_rows / 2, input_rows, input_rows, min_tokens});
        (void)detail::w8_linear_swiglu_resolve_plan(
            {gate_up_rows, gate_up_rows / 2, input_rows, input_rows, max_tokens});
        // surogate vendor patch (PATCHES.md #17): under AllowA8 the large-T
        // band runs the W8A8-int IMMA path (per-token int8 activations),
        // which needs quantized-activation and gemm-buffer workspace.
        if (policy == LinearPolicy::AllowA8 && max_tokens >= detail::kW8A8MinTokens) {
            return detail::w8a8_linear_swiglu_workspace_bytes(gate_up_rows, input_rows,
                                                              max_tokens);
        }
        return 0;
    }
    if (qtype == QType::Q4G64_F16S) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("linear_swiglu workspace: Q4 admits only A16");
        }
        return detail::q4_linear_swiglu_capacity_workspace_bytes(
            gate_up_rows, gate_up_rows / 2, input_rows, input_rows, min_tokens, max_tokens);
    }
    if (qtype == QType::NVFP4 && gate_up_rows == 34816 && input_rows == 5120) {
        return detail::nvfp4_linear_swiglu_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    if (qtype == QType::FP8_E4M3FN_ROW_BF16S && gate_up_rows == 34816 && input_rows == 5120) {
        return detail::fp8_linear_swiglu_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    throw std::invalid_argument("linear_swiglu workspace: unsupported weight format");
}

std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                   std::int32_t input_rows, std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    return linear_swiglu_workspace_capacity_bytes(qtype, gate_up_rows, input_rows,
                                                  LinearPolicy::A16Only, min_tokens, max_tokens);
}

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                   WorkspaceArena& ws, cudaStream_t stream) {
    validate_policy(policy);
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear_swiglu: x/out must be BF16");
    }
    const std::int32_t t   = x.ne[1];
    const bool large_shape = x.ne[0] == 5120 && out.ne[0] == 17408 && gate_up_weight.n == 34816 &&
                             gate_up_weight.k == 5120 && gate_up_weight.padded_shape[0] == 34816 &&
                             gate_up_weight.padded_shape[1] == 5120;
    const bool w8_shape = x.ne[0] == 2048 && out.ne[0] == 6144 && gate_up_weight.n == 12288 &&
                          gate_up_weight.k == 2048 && gate_up_weight.padded_shape[0] == 12288 &&
                          gate_up_weight.padded_shape[1] == 2048;
    // surogate vendor patch (PATCHES.md #13): qwen3.5-0.8b mlp (1024 -> 2x3584).
    const bool q08_shape = x.ne[0] == 1024 && out.ne[0] == 3584 && gate_up_weight.n == 7168 &&
                           gate_up_weight.k == 1024 && gate_up_weight.padded_shape[0] == 7168 &&
                           gate_up_weight.padded_shape[1] == 1024;
    // surogate vendor patch (PATCHES.md #18): qwen3.5-4b mlp (2560 -> 2x9216).
    const bool q4b_shape = x.ne[0] == 2560 && out.ne[0] == 9216 && gate_up_weight.n == 18432 &&
                           gate_up_weight.k == 2560 && gate_up_weight.padded_shape[0] == 18432 &&
                           gate_up_weight.padded_shape[1] == 2560;
    if (t <= 0 || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[1] != t || out.ne[2] != 1 ||
        out.ne[3] != 1 || (!large_shape && !w8_shape && !q08_shape && !q4b_shape)) {
        throw std::invalid_argument("linear_swiglu: invalid tensor shape");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear_swiglu: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear_swiglu: x/out must be non-null and 16-byte aligned");
    }

    const bool common_row_split =
        gate_up_weight.layout == QuantLayout::RowSplit &&
        gate_up_weight.scale_dtype == DType::FP16 && gate_up_weight.ndim == 2 &&
        gate_up_weight.shape[0] == gate_up_weight.n &&
        gate_up_weight.shape[1] == gate_up_weight.k && gate_up_weight.qdata != nullptr &&
        gate_up_weight.scales != nullptr;
    const bool q4_weight = large_shape && gate_up_weight.qtype == QType::Q4G64_F16S &&
                           gate_up_weight.group_size == 64 && gate_up_weight.group == 64 &&
                           common_row_split;
    const bool w8_weight = (w8_shape || q08_shape || q4b_shape) &&
                           gate_up_weight.qtype == QType::W8G32_F16S &&
                           gate_up_weight.group_size == 32 && gate_up_weight.group == 32 &&
                           gate_up_weight.qhigh == nullptr &&
                           gate_up_weight.high_plane_bytes == 0 && common_row_split;
    const bool nvfp4_weight = large_shape && gate_up_weight.qtype == QType::NVFP4;
    const bool fp8_weight   = large_shape && gate_up_weight.qtype == QType::FP8_E4M3FN_ROW_BF16S;
    if (!q4_weight && !w8_weight && !nvfp4_weight && !fp8_weight) {
        throw std::invalid_argument("linear_swiglu: unsupported weight");
    }

    if (fp8_weight) {
        (void)detail::validate_fp8_weight(gate_up_weight, "fp8 linear_swiglu");
        // Marlin band (PATCHES.md #38): the 27B's largest decode GEMM.
        if (t >= detail::marlin_min_band_tokens() && t <= detail::kMarlinMaxBandTokens) {
            const detail::MarlinScratch scratch = detail::marlin_scratch();
            if (detail::marlin_fp8_plane_for(gate_up_weight, stream).b_packed != nullptr &&
                scratch.gemm_out != nullptr) {
                Tensor fused(scratch.gemm_out, DType::BF16, {gate_up_weight.n, t});
                if (detail::marlin_fp8_run(x, gate_up_weight, fused, stream)) {
                    const std::int32_t half = gate_up_weight.n / 2;
                    Tensor gate = fused.slice(0, 0, half);
                    Tensor up   = fused.slice(0, half, half);
                    silu_mul(gate, up, out, stream);
                    return;
                }
            }
        }
        detail::fp8_linear_swiglu_dispatch(x, gate_up_weight, out, policy, ws, stream);
        return;
    }

    if (nvfp4_weight) {
        (void)detail::validate_nvfp4_weight(gate_up_weight, "nvfp4 linear_swiglu");
        detail::nvfp4_linear_swiglu_dispatch(x, gate_up_weight, out, policy, ws, stream);
        return;
    }

    if (policy != LinearPolicy::A16Only && !(w8_weight && policy == LinearPolicy::AllowA8)) {
        throw std::invalid_argument("linear_swiglu: Q4 admits only A16; W8 admits A16 or A8");
    }
    if (!aligned_to(gate_up_weight.qdata, 16) ||
        !aligned_to(gate_up_weight.scales, w8_weight ? 16 : 4)) {
        throw std::invalid_argument("linear_swiglu: required code/scale alignment is missing");
    }

    if (w8_weight) {
        // Marlin band (PATCHES.md #33): gate_up is the largest decode GEMM;
        // the vendored kernel runs it 2.2x faster and silu_mul then folds
        // the [2*out, T] result down.
        if (t >= detail::marlin_min_band_tokens() && t <= detail::kMarlinMaxBandTokens) {
            const detail::MarlinScratch scratch =
                detail::marlin_scratch_for(gate_up_weight, stream);
            if (scratch.gemm_out != nullptr) {
                Tensor fused(scratch.gemm_out, DType::BF16, {gate_up_weight.n, t});
                if (detail::marlin_w8_run(x, gate_up_weight, fused, stream)) {
                    const std::int32_t half = gate_up_weight.n / 2;
                    Tensor gate = fused.slice(0, 0, half);
                    Tensor up   = fused.slice(0, half, half);
                    silu_mul(gate, up, out, stream);
                    return;
                }
            }
        }
        // surogate vendor patch (PATCHES.md #17): AllowA8 large-T prefill runs
        // the W8A8-int IMMA path; decode and small T stay on the A16 kernels.
        if (policy == LinearPolicy::AllowA8 && t >= detail::kW8A8MinTokens) {
            detail::w8a8_linear_swiglu_dispatch(x, gate_up_weight, out, ws, stream);
        } else {
            detail::w8_linear_swiglu_dispatch(x, gate_up_weight, out, stream);
        }
    } else {
        detail::q4_linear_swiglu_dispatch(x, gate_up_weight, out, ws, stream);
    }
}

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, WorkspaceArena& ws,
                   cudaStream_t stream) {
    linear_swiglu(x, gate_up_weight, out, LinearPolicy::A16Only, ws, stream);
}

} // namespace ninfer::ops
