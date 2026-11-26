#pragma once

// The LoopedSharedExpert's elementwise kernels (decision/kernels.cu). All BF16; `h` is a residual slot plane (column
// stride `h_column` elements), the other tensors contiguous [rows, tokens].

#include <cuda_runtime.h>

#include <cstdint>

namespace sinfer::targets::inkling2::detail::decision {

/// out [d, tokens] = bf16(bf16(rms_norm(h) * bf16(1 + scale)) + shift); rms_norm with `norm_weight`, eps `eps`.
void shared_expert_modulate(const void* h, std::int64_t h_column, const void* norm_weight, const void* scale,
                            const void* shift, void* out, int d, int tokens, float eps, cudaStream_t s);
/// act [f, tokens] = bf16(bf16(silu(gate)) * up), gate_up [2f, tokens] (gate rows first).
void shared_expert_swiglu(const void* gate_up, void* act, int f, int tokens, cudaStream_t s);
/// h += bf16(bf16(1 + gate) * out), out [d, tokens].
void shared_expert_residual(void* h, std::int64_t h_column, const void* gate, const void* out, int d, int tokens,
                            cudaStream_t s);

} // namespace sinfer::targets::inkling2::detail::decision
