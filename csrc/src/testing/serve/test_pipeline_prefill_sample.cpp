#include "runtime/engine/prefill_sample.h"

#include <cassert>
#include <iostream>

using namespace sinfer::runtime;

int main() {
    sinfer::TokenId shared_token = 11;
    float shared_score = -0.31F;
    PipelinePrefillSample first, second;
    PrefillStepResult a{.round = {{&shared_token, 1}, {&shared_score, 1}}, .complete = true};
    first.capture(a);
    shared_token = 22;
    shared_score = -2.44F;
    PrefillStepResult b{.round = {{&shared_token, 1}, {&shared_score, 1}}, .complete = true};
    second.capture(b);
    shared_token = 33;
    shared_score = -99.0F;
    assert(a.round.tokens.front() == 11 && a.round.logprobs.front() == -0.31F);
    assert(b.round.tokens.front() == 22 && b.round.logprobs.front() == -2.44F);
    PrefillStepResult unscored{.round = {{&shared_token, 1}, {}}, .complete = true};
    first.capture(unscored);
    assert(unscored.round.tokens.front() == 33 && unscored.round.logprobs.empty());
    assert(b.round.logprobs.front() == -2.44F);
    std::cout << "Pipeline prefill tokens and optional scores survive shared egress reuse\n";
}
