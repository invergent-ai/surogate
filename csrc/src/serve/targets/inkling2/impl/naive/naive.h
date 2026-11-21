#pragma once

// NAIVE "correct-first" kernels of the Inkling-2 target (lane e1), standing in for lane c's attention and lane
// d's MoE until those land (DESIGN-target.md 5). Straight transcriptions of `inkling_ref.py`
// (`inkling_attention_prologue`, `inkling_rel_attention`, `inkling_gate_sink`, `inkling_route`,
// `inkling_quant_nvfp4_4o6`, `inkling_quant_mxfp8`, `inkling_softcap_swiglu`, `inkling_moe`) with fp32
// accumulation; slow, never tuned, never used for a speed number. Replaced call by call as c / d deliver.

#include "core/arena.h"
#include "core/tensor.h"
#include "targets/inkling2/impl/load/bindings.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::targets::inkling2::naive {

/// One attention call of a visit: B rows of W columns (column n = j + W * b; a prompt segment is B = 1).
struct AttentionCall {
    // inputs: the visit's fused projection [8600, N] (q | r | gate | sink | k | v rows), column stride in elements
    const void* qkvr         = nullptr;
    std::int64_t qkvr_stride = 0;
    std::int32_t width = 0, batch = 0;
    const std::int32_t* positions = nullptr;  ///< [W, B]
    const std::int32_t* valid     = nullptr;  ///< [B] or null
    // the visit's KV plane
    const std::int32_t* table_rows   = nullptr; ///< [B]
    const std::int32_t* block_tables = nullptr; ///< [logical_pages, rows]
    std::int64_t table_stride        = 0;       ///< logical pages per row
    void* k_pages = nullptr;
    void* v_pages = nullptr;
    bool fp8      = true; ///< unit-scale e4m3 planes (else bf16)
    // the layer
    const load::InklingLayerWeights* layer = nullptr;
    const void* k_dyn = nullptr; ///< global: BF16 [32, N] = k_conv_proj(x_k)
    const void* v_dyn = nullptr;
    std::int64_t dyn_stride = 32;
    // the visit's conv state, K at channels [0, 1024), V at [1024, 2048); slot per row
    void* state              = nullptr;
    std::int64_t slot_stride = 0;
    std::int64_t tap_stride  = 0;
    const std::int32_t* slots = nullptr; ///< [B]
    // outputs
    void* q_out = nullptr; ///< BF16 [3072, N] scratch (clipped, tau-scaled q)
    void* r_out = nullptr; ///< BF16 [384, N] scratch
    void* out   = nullptr; ///< BF16 [3072, N] contiguous: gated, sunk attention (the wo_ud input)
};

/// Prologue (K / V convs from the slot's state, clips, norms, KV write) then attention + gate / sink. The
/// conv windows are NOT advanced here: the caller updates them after (it may need the old ones first).
void attention(const AttentionCall& call, cudaStream_t stream);

/// MoE of one visit: x BF16 [2304, T] contiguous -> out BF16 [2304, T]. `capture` (null = none) receives
/// token t's 8 ids at capture + t * 8.
struct MoeCall {
    const void* x                       = nullptr;
    std::int32_t tokens                 = 0;
    const load::InklingLayerWeights* layer = nullptr;
    std::int32_t* capture               = nullptr;
    void* out                           = nullptr;
};

[[nodiscard]] std::size_t moe_workspace_bytes(std::int32_t tokens);
void moe(const MoeCall& call, WorkspaceArena& workspace, cudaStream_t stream);

/// Fill `count` I32 with `value` (capture-safe; the value is a launch parameter).
void fill_i32(std::int32_t* data, std::int32_t count, std::int32_t value, cudaStream_t stream);

} // namespace sinfer::targets::inkling2::naive
