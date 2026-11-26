#pragma once

// The visit-level entry points a decision adapter's rounds use (implemented in forward.cpp next to the chat path's
// `Variant::run_visits`, so every op call is the chat path's own `run_visit_on`). A decision round is one question's
// whole prompt (B = 1, positions 0..T-1) on private state: ONE KV plane and ONE conv-state slot, reused by every
// visit. That is exact for a whole-prompt round -- a visit's attention reads only the keys its own prologue wrote, and
// the convs read no history before position 0 -- so a decision round needs neither the chat KV pool nor a plane per
// visit, whatever its pass count.

#include "targets/inkling2/impl/load/bindings.h"
#include "targets/inkling2/impl/model.h"
#include "targets/inkling2/impl/visits.h"

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>

namespace sinfer::targets::inkling2::detail::decision {

/// How a decision round computes (SUROGATE_SERVE_INKLING_DECISION_NUMERICS, default `fast`): `reference` reproduces the System 1.5
/// reference's numerics -- the naive attention over BF16 keys, and the naive experts W4A16 / W8A16 with FP32 routers --
/// at the naive kernels' speed; `fast` runs the engine's own ops (FP8 keys, W4A4 / W8A8 experts, BF16 routers).
struct Numerics {
    bool reference = true;
    const float* router_f32[TextConfig::physical_layers]{}; ///< FP32 [128, 2304] per layer (reference)
};

struct RoundIO {
    const RuntimeModelView* model = nullptr; ///< the served base model's view
    const InklingModel* im        = nullptr;
    const DerivedQueries* queries = nullptr; ///< the adapter's (pre-scaled AttnRes / message queries + the base's rel tables)
    Tensor x;                      ///< residual BF16 [7 x 2304, T] (slot 0 filled by `embed`)
    Tensor positions;              ///< device I32 [T, 1] = 0..T-1
    Tensor meta;                   ///< device I32 [8]: [0] = KV table row 0, [4] = conv slot 0 (16-byte aligned)
    PagedKVBatchLayerView kv;      ///< one FP8 e4m3 (unit-scale) plane over T keys, identity block table
    Tensor conv;                   ///< conv state BF16 [6656, 3, 1]
    void* target_state = nullptr;  ///< Variant::visit_target_state_bytes(), zeroed once
    WorkspaceArena* work = nullptr;
    cudaStream_t stream  = nullptr;
    const Numerics* numerics = nullptr; ///< null: `fast`
};

/// Slot 0 of `io.x`: the normed embedding of `ids` (device I32 [T]), as the chat path's embed hook.
void embed(const RoundIO& io, const Tensor& ids);

/// One visit of a decision schedule: its entry `vi`, its layer's weights `L` (the adapter's replacements bound over the
/// base's), `label` naming it in dumps.
void run_visit(const RoundIO& io, const VisitInfo& vi, const load::InklingLayerWeights& L, int label);

/// The final output mix after a schedule: AttnRes over `final_stack` + the coda partial with `output_query` (pre-scaled
/// FP32 [D, 1]) and the final norm -> hidden BF16 [2304, T].
void final_mix(const RoundIO& io, const std::array<std::int8_t, 4>& final_stack, const Tensor& output_query,
               Tensor& hidden);

} // namespace sinfer::targets::inkling2::detail::decision
