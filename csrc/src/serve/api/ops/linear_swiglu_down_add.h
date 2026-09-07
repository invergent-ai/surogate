#pragma once
// sinfer::ops - the SwiGLU MLP on the wide NVFP4 route without its activation:
// residual += down @ silu(gate) * up, with silu(gate) * up quantised straight into the down
// projection's W4A4 operand. The unfused pair (linear_swiglu, then linear_add) writes a BF16
// [I, T] activation that the down projection's quantiser reads back; on a 27B prompt round that
// is 160 MB through HBM for an intermediate nobody keeps. Same numbers: each SwiGLU value is
// rounded to BF16 before it is quantised, exactly as the pair rounds it.

#include "core/arena.h"
#include "core/tensor.h"
#include "api/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops {

/// Whether the fused route serves this pair at `tokens`: both weights NVFP4, the fused parent
/// twice the down projection's K and the down projection's rows the parent's K, an A4 policy,
/// and a width the wide GEMM takes. `SUROGATE_SERVE_NVFP4_FUSED_SWIGLU=0` vetoes it.
[[nodiscard]] bool linear_swiglu_down_add_admits(const Weight& gate_up, const Weight& down,
                                                 LinearPolicy policy, std::int32_t tokens);

/// Transient capacity of the fused route for every T in [min_tokens, max_tokens]: the projected
/// [2I, T] BF16 plane beside the larger of the gate/up GEMM's own scratch and the down
/// projection's quantised operand.
[[nodiscard]] std::size_t linear_swiglu_down_add_workspace_capacity_bytes(
    std::int32_t intermediate, std::int32_t hidden, LinearPolicy policy, std::int32_t min_tokens,
    std::int32_t max_tokens);

/// residual [hidden, T] += down @ (silu(gate) * up) with gate/up = gate_up @ x; `limit` > 0
/// clamps the SwiGLU as `silu_mul` does, 0 leaves it unclamped. The caller checked `admits`.
void linear_swiglu_down_add(const Tensor& x, const Weight& gate_up, const Weight& down,
                            Tensor& residual, LinearPolicy policy, float limit, WorkspaceArena& ws,
                            cudaStream_t stream);

} // namespace sinfer::ops
