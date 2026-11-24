#pragma once

#include "api/ops/sampling.h"
#include "api/types.h"

#include <cstddef>
#include <span>

namespace sinfer::runtime {

/// The reply-phase sampler of one request (SamplingOverrides::reply_*): its thought is drawn with the request's own
/// config, every token after the model's reasoning close token with the reply values.
///
/// Host side, per round: advance() walks the tokens the request has generated since the last call and notes the
/// first close token; apply() turns the request's own config into the one for the round's draws. Before the close
/// that is the request's config unchanged except for the two phase_switch fields (which only the speculative verify
/// reads: a round accepts no draft at or after a drafted close, so every token it commits is a thought token or the
/// close itself, drawn exactly as without the option); after it, the same config with the reply values. The random
/// stream is keyed by position and purpose (ops/sampling.h), so it is the same with the option on or off.
struct ReplyPhase {
    /// The request waits for its thought to close (reply values set, the model has a single close token).
    bool armed = false;
    /// The close token has been generated: every later draw uses `values`.
    bool replying = false;
    TokenId switch_token = -1;
    /// Ledger index advance() has read up to; starts at the end of the request's own prompt.
    std::size_t frontier = 0;
    ReplySamplingParameters values;

    /// Notes the first close token among ledger[frontier, size). Returns true when this call found it.
    bool advance(std::span<const TokenId> ledger) noexcept {
        if (!armed || replying) { return false; }
        while (frontier < ledger.size()) {
            if (ledger[frontier++] == switch_token) {
                replying = true;
                return true;
            }
        }
        return false;
    }

    /// The config the request's next draws use; `config` is the request's own.
    [[nodiscard]] ops::SamplingConfig apply(ops::SamplingConfig config) const noexcept {
        if (!armed) { return config; }
        if (replying) {
            config.temperature        = values.temperature;
            config.top_k              = values.top_k;
            config.top_p              = values.top_p;
            config.min_p              = values.min_p;
            config.phase_switch_armed = 0;
        } else {
            config.phase_switch_token = switch_token;
            config.phase_switch_armed = 1;
        }
        return config;
    }
};

} // namespace sinfer::runtime
