#include "api/ops/linear_swiglu_down_add.h"

#include "api/ops/linear.h"
#include "core/layout.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_cublaslt.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace sinfer::ops {

namespace {

bool fused_swiglu_vetoed() {
    static const bool vetoed = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_NVFP4_FUSED_SWIGLU");
        return raw != nullptr && std::strcmp(raw, "0") == 0;
    }();
    return vetoed;
}

} // namespace

bool linear_swiglu_down_add_admits(const Weight& gate_up, const Weight& down, LinearPolicy policy,
                                   std::int32_t tokens) {
    if (fused_swiglu_vetoed() || policy != LinearPolicy::AllowA4 || tokens <= 0) { return false; }
    if (gate_up.qtype != QType::NVFP4 || down.qtype != QType::NVFP4) { return false; }
    if (gate_up.n != 2 * down.k || gate_up.k != down.n) { return false; }
    if (!detail::is_nvfp4_linear_problem(gate_up.n, gate_up.k) ||
        !detail::is_nvfp4_linear_problem(down.n, down.k)) {
        return false;
    }
    // The gate/up GEMM must take its wide route too, so the projected plane is the only
    // intermediate; below the wide width the fused small-T SwiGLU kernels are the better route.
    return detail::nvfp4_cublaslt_route(tokens) && detail::nvfp4_linear_add_w4a4_wide(down, tokens);
}

std::size_t linear_swiglu_down_add_workspace_capacity_bytes(std::int32_t intermediate,
                                                            std::int32_t hidden,
                                                            LinearPolicy policy,
                                                            std::int32_t min_tokens,
                                                            std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear_swiglu_down_add workspace: invalid token interval");
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {2 * intermediate, max_tokens});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(linear_workspace_capacity_bytes(QType::NVFP4, 2 * intermediate,
                                                                hidden, policy, min_tokens,
                                                                max_tokens));
    }
    {
        auto scope = layout.scope();
        (void)detail::allocate_nvfp4_w4a4_workspace(layout, max_tokens, intermediate);
    }
    return layout.peak_bytes(1);
}

void linear_swiglu_down_add(const Tensor& x, const Weight& gate_up, const Weight& down,
                            Tensor& residual, LinearPolicy policy, float limit, WorkspaceArena& ws,
                            cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (!linear_swiglu_down_add_admits(gate_up, down, policy, tokens)) {
        throw std::invalid_argument("linear_swiglu_down_add: the fused route does not serve this call");
    }
    auto scope       = ws.scope();
    Tensor projected = ws.alloc(DType::BF16, {gate_up.n, tokens});
    {
        auto gemm_scope = ws.scope();
        linear(x, gate_up, projected, policy, ws, stream);
    }
    const detail::Nvfp4W4a4Workspace operand =
        detail::allocate_nvfp4_w4a4_workspace(ws, tokens, down.k);
    detail::launch_nvfp4_w4a4_swiglu_quantize(projected, down, operand, limit, stream,
                                              detail::Nvfp4ScaleLayout::Tiled);
    detail::nvfp4_linear_add_w4a4_wide_gemm(down, operand, residual, tokens, stream);
}

} // namespace sinfer::ops
