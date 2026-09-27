#pragma once

#include "api/types.h"

namespace sinfer::runtime {

// Decisions expose candidate logits after one target token. Their initial and final
// scoring passes cannot benefit from a drafter; the intervening thought is an ordinary
// multi-token generation and keeps speculation. GPU-prefix readouts already use this
// target-only route with every backend.
template <class Options>
bool dflash_candidate_readout(SpeculativeBackend backend, const Options& options) noexcept {
    return backend == SpeculativeBackend::DFlash && !options.gpu_prefix && !options.save_gpu_prefix &&
           options.requested_output_tokens == 1 && !options.next_token_candidates.empty();
}

template <class Options>
bool target_only_readout(SpeculativeBackend backend, const Options& options) noexcept {
    return options.gpu_prefix || options.save_gpu_prefix || dflash_candidate_readout(backend, options);
}

// Target-only prompts cannot share a speculative verification round. When both kinds
// of work are active, give their prefill-only pack a turn after a decode round, then
// return to decoding. The existing opt-in still controls packing drafter prefills.
inline bool can_pack_prefill_only(SpeculativeBackend backend, bool candidate_readout,
                                  bool pack_dflash, bool have_decoders,
                                  bool previous_unit_was_decode) noexcept {
    const bool bypass = backend == SpeculativeBackend::DFlash && candidate_readout;
    const bool enabled = backend == SpeculativeBackend::None || bypass ||
                         (backend == SpeculativeBackend::DFlash && pack_dflash);
    return enabled && (!have_decoders || (bypass && previous_unit_was_decode));
}

} // namespace sinfer::runtime
