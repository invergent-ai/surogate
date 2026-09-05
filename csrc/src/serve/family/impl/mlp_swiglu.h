#pragma once

// The SwiGLU MLP of a fused-parent target, and the route it takes when an
// adapter adapts gate or up.
//
// These targets store gate and up as one weight and hand it to `linear_swiglu`,
// which projects and activates in a single kernel. That is the fast route and it
// stays the default -- but it never materialises the two halves, so there is no
// tensor for a `gate_proj` or `up_proj` delta to be added to. Every adapter PEFT
// writes names both of them, so refusing them refused the adapter.
//
// When either half is bound this takes the parent apart instead: `linear_rows`
// projects each half of the same weight, the deltas land on the halves, and the
// activation runs elementwise. It costs one extra launch and two [I,T] buffers,
// and it is the only route on which a gate or up adapter means anything.
//
// Which route runs is decided by whether the bank exists, and banks are frozen
// in the engine's construction window -- before the first graph capture. So a
// captured graph records one route and every replay takes the same one.

#include "api/ops/linear.h"
#include "api/ops/linear_swiglu.h"
#include "api/ops/silu_mul.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "family/impl/lora_hook.h"

#include <cstdint>
#include <exception>

#include <cuda_runtime.h>

namespace sinfer::family {

/// Whether this fused parent's two halves can be projected on their own.
///
/// `weight_rows` admits the formats whose rows are independently addressable and
/// refuses the rest rather than mis-reading them, so this asks it. A target calls
/// this once, at bind time, and registers a refusal naming the format when the
/// answer is no -- which is better than binding the module and throwing on every
/// forward pass.
[[nodiscard]] inline bool swiglu_halves_addressable(const Weight& gate_up) noexcept {
    const std::int32_t rows = gate_up.n / 2;
    if (rows <= 0 || gate_up.n != 2 * rows) { return false; }
    try {
        (void)ops::weight_rows(gate_up, 0, rows);
        (void)ops::weight_rows(gate_up, rows, rows);
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

/// `activation[i,t] = SiLU(gate[i,t]) * up[i,t]`, with adapters on either half
/// when they are bound. `activation` is [I,T] and comes from the caller's arena.
inline void swiglu_mlp(const Tensor& hidden, const Weight& gate_up, Tensor& activation,
                       ops::LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) {
    if (!lora_bound(gate_up, kGatePort) && !lora_bound(gate_up, kUpPort)) {
        ops::linear_swiglu(hidden, gate_up, activation, policy, workspace, stream);
        return;
    }
    auto scope                 = workspace.scope();
    const std::int32_t rows    = gate_up.n / 2;
    const std::int32_t columns = hidden.ne[1];
    // Gate rows precede up rows in the parent, which is `linear_swiglu`'s own
    // contract; the halves are read here in that order.
    Tensor gate = workspace.alloc(DType::BF16, {rows, columns});
    Tensor up   = workspace.alloc(DType::BF16, {rows, columns});
    ops::linear_rows(hidden, gate_up, 0, gate, &workspace, stream);
    ops::linear_rows(hidden, gate_up, rows, up, &workspace, stream);
    apply_lora_gate_up(gate_up, hidden, gate, up, stream);
    ops::silu_mul(gate, up, activation, stream);
}

/// The transient storage `swiglu_mlp` may take, over both of its routes.
///
/// Which route a deployment runs is not known when the workspace is planned --
/// it depends on whether adapters were enabled -- so the plan must hold either.
/// The builder's peak is a maximum over scopes, so laying both out here is what
/// takes the larger of the two rather than their sum.
inline void swiglu_mlp_layout(WorkspaceLayoutBuilder& layout, std::int32_t intermediate,
                              std::int32_t hidden, QType gate_up_qtype, ops::LinearPolicy policy,
                              std::int32_t first, std::int32_t last) {
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
            gate_up_qtype, 2 * intermediate, hidden, policy, first, last));
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc(DType::BF16, {intermediate, last});
        (void)layout.alloc(DType::BF16, {intermediate, last});
        (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(gate_up_qtype, intermediate,
                                                                     hidden, policy, first, last));
    }
}

} // namespace sinfer::family
