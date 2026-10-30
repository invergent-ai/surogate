#pragma once

// LoRA on the Marlin MoE route: the routed paths' adapter deltas, on tensor cores, over the same
// expert-sorted blocks the Marlin GEMMs run (WMMA BF16 16x16x16, FP32 accumulate).
//
// A round's gate/up delta goes into Marlin's raw [rows, 2 * intermediate] product before the
// SiLU, the down delta into its [rows, hidden] product (weighted like it by the row's routing
// weight) before the combine -- so an adapted path computes exactly what the adapter route's
// decode kernels compute, at tensor-core width. Rows whose token selected no adapter (slot -1)
// or another slot add nothing in this slot's pass; one pass per configured slot.

#include "api/ops/lora.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace sinfer::ops::detail {

struct MarlinMoeLoraCall {
    /// BF16: [tokens][k] (`input_per_token`, gate/up) or [rows][k] (down).
    const void* input    = nullptr;
    bool input_per_token = true;
    /// BF16 [rows][out_ld]: the product the delta is added to.
    void* out          = nullptr;
    std::int32_t out_ld = 0;
    /// BF16 [rows][3][rank]: the low-rank intermediates (workspace).
    void* low = nullptr;
    /// Marlin's routing (sorted assignment ids per padded block slot, expert per block, padded row
    /// count), or null for a dense mixture of `rows` rows of `fixed_expert` (the shared expert).
    const std::int32_t* sorted_ids  = nullptr;
    const std::int32_t* expert_ids  = nullptr;
    const std::int32_t* past_padded = nullptr;
    std::int32_t fixed_expert       = -1;
    /// The rows' experts in row order ([tokens][top_k] routing ids), for the narrow-round kernels.
    const std::int32_t* ids         = nullptr;
    std::int32_t block              = 0;
    std::int32_t block_capacity     = 0; ///< launch upper bound on blocks
    std::int32_t rows               = 0; ///< real rows (assignments, or tokens)
    std::int32_t top_k              = 1; ///< rows per token
    /// The mixture's bank table (LoraStore::bank_table): [2 + 3 * expert + {gate, up, down}].
    const LoraBank* banks = nullptr;
    /// Per-token adapter slot (stride 1) or the round's uniform cell (stride 0).
    const std::int32_t* slots = nullptr;
    std::int32_t slot_stride  = 0;
    std::int32_t slot_count   = 1;
    std::int32_t rank         = 0; ///< the store's max rank
    std::int32_t k            = 0;
    std::int32_t n            = 0; ///< output columns per part
    /// down: the down part only, weighted by `row_weights`; otherwise gate then up, each at its
    /// column offset in the fused product.
    bool down                       = false;
    std::int32_t parts              = 2;
    std::int32_t column_offset[2]   = {0, 0};
    const float* row_weights        = nullptr;
    /// Wide rounds of a routed mixture: Marlin's padded expert offsets ([experts + 1]) and the
    /// 64-slot tile table the call lays out from them (workspace, `tile_capacity` entries).
    const std::int32_t* offsets  = nullptr;
    std::int32_t experts         = 0;
    int2* tiles                  = nullptr;
    std::int32_t* tile_count     = nullptr;
    std::int32_t tile_capacity   = 0;
};

/// Tile-table entries a routed call of `assignments` rows over `experts` experts can need.
std::int32_t marlin_moe_lora_tile_capacity(std::int32_t assignments, std::int32_t experts);

void marlin_moe_lora_delta(const MarlinMoeLoraCall& call, cudaStream_t stream);

} // namespace sinfer::ops::detail
