#pragma once

// The projection-side half of adapter application, shared by the family's
// targets: after a projection's base GEMM ran, add the low-rank delta for
// whatever adapter each token of the round selected.
//
// The hook reads the round's selection from the thread-local published by the
// decode and prefill schedules rather than from a parameter, because it sits
// inside variant methods whose signatures the family shares -- threading a
// round context through all of them would change every target's interface to
// serve one feature. It launches only when the projection has a bank, which
// the store preallocates for every registered module before the first capture,
// so eager calls and captured graphs run the same kernel set.

#include "api/ops/lora.h"
#include "api/ops/lora_store.h"
#include "core/tensor.h"

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace sinfer::targets::qwen3_6 {

inline void apply_lora(const Weight& base, std::int32_t port, const Tensor& hidden, Tensor& out,
                       cudaStream_t stream) {
    const bool debug = std::getenv("SUROGATE_SERVE_LORA_DEBUG") != nullptr;
    if (!ops::lora_active()) {
        if (debug) { std::fprintf(stderr, "lora-hook: inactive\n"); }
        return;
    }
    const ops::LoraBank* bank = ops::lora_store_for_current_device().find(base.qdata, port);
    if (bank == nullptr) {
        if (debug) { std::fprintf(stderr, "lora-hook: no bank for this weight\n"); }
        return;
    }
    const ops::LoraRound& round = ops::lora_current_round();
    if (!round.valid()) {
        if (debug) { std::fprintf(stderr, "lora-hook: no round published\n"); }
        return;
    }
    if (debug) { std::fprintf(stderr, "lora-hook: applying\n"); }
    static const Tensor kNoIds{};
    ops::lora_delta_batched(hidden, *bank, round.slots != nullptr ? *round.slots : kNoIds,
                            round.uniform ? ops::lora_store_for_current_device().uniform_cell()
                                          : nullptr,
                            out, const_cast<Tensor&>(round.scratch), stream);
}

} // namespace sinfer::targets::qwen3_6
