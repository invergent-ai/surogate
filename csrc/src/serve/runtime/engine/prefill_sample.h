#pragma once

#include "runtime/contract/types.h"

#include <stdexcept>

namespace sinfer::runtime {

// A pipeline flight retains its first sampled token and score until consumption.
struct PipelinePrefillSample {
    TokenId token = 0;
    float logprob = 0;

    void capture(PrefillStepResult& step) {
        if (step.round.tokens.size() != 1 || step.round.logprobs.size() > 1) {
            throw std::logic_error("pipeline stages expect one sampled token from a finished prefill");
        }
        token = step.round.tokens.front();
        step.round.tokens = {&token, 1};
        if (!step.round.logprobs.empty()) {
            logprob = step.round.logprobs.front();
            step.round.logprobs = {&logprob, 1};
        }
    }
};

} // namespace sinfer::runtime
