#pragma once

// Inkling-2 decision adapters (System 1.5) served beside the base model: the runners an inkling2 engine built at load
// for `--decision-adapter name=path` (targets/inkling2/impl/decision/runner.h), by name, for the decisions endpoint.
// A process holds one served inkling2 base; its adapters are registered when its frontend is made.
//
// `run` executes one question on the engine's device with the runner's own state. The caller must hold the engine's
// execution paused (`Engine::with_execution_paused`): a decision round uses the model's weights and the stream state
// the engine's own rounds use, and it must not interleave with them.

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace sinfer::inkling2_decision {

struct Question {
    std::vector<std::int32_t> ids;
    std::int32_t decide_idx = 0;
    std::vector<std::int32_t> option_idx;
};

struct Result {
    int loops_used = 0;
    std::vector<float> logits;          ///< the joint head's option logits at the stopping pass
    double temperature = 1.0;           ///< the adapter's calibrated exit temperature for that pass
    std::vector<double> probabilities;  ///< softmax(logits / temperature)
    std::optional<float> halt_score;    ///< the last halting score computed
    std::optional<float> escalate_prob;
    float state_drift = 0.0F;
    int l1_pred = -1;                   ///< pass 1's CoreGate choice
    std::vector<float> aux;             ///< the aux probes' probabilities (empty without the head)
    double seconds = 0.0;
    std::string capture;                ///< the capture file's name when SUROGATE_SERVE_INKLING_DECISION_CAPTURE is set
};

struct Info {
    int max_loops = 5;
    std::int32_t max_tokens = 0;        ///< the longest prompt a runner takes
    std::int32_t max_state_tokens = 8192;
    std::vector<std::string> special_tokens; ///< state, question, option open, option close, decide
    std::map<std::string, double> modes;     ///< calibrated halting thresholds by mode ("full" never halts)
};

/// Is `model` a registered decision adapter?
[[nodiscard]] bool available(std::string_view model);
[[nodiscard]] std::vector<std::string> names();
[[nodiscard]] Info info(std::string_view model);
/// One question under a halting threshold (> 1: never halts) and at most `max_loops` passes.
[[nodiscard]] Result run(std::string_view model, const Question& question, double threshold, int max_loops);

} // namespace sinfer::inkling2_decision
