#pragma once

// surogate vendor patch (PATCHES.md #17): W8A8-int IMMA linear_swiglu
// (large-T prefill path; see the .cu for the pipeline description).

#include "core/arena.h"
#include "ops/linear/w8a8/w8a8_act_quant.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail {

[[nodiscard]] std::size_t w8a8_linear_swiglu_workspace_bytes(std::int32_t gate_up_rows,
                                                             std::int32_t input_rows,
                                                             std::int32_t max_tokens) noexcept;

void w8a8_linear_swiglu_dispatch(const Tensor& x, const Weight& gate_up_weight, Tensor& out,
                                 WorkspaceArena& workspace, cudaStream_t stream);

} // namespace sinfer::ops::detail
