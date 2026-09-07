#include "runtime/contract/sampling.h"

#include <api/targets/qwen3_5/package.h>
#include <api/targets/qwen3_5_moe/package.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

bool same_preset(const sinfer::SamplingPreset& actual, const sinfer::SamplingPreset& expected) {
    return actual.temperature == expected.temperature && actual.top_k == expected.top_k &&
           actual.top_p == expected.top_p && actual.min_p == expected.min_p &&
           actual.presence_penalty == expected.presence_penalty &&
           actual.frequency_penalty == expected.frequency_penalty;
}

bool throws_invalid(const auto& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

bool throws_runtime(const auto& operation) {
    try {
        operation();
    } catch (const std::runtime_error&) { return true; }
    return false;
}

} // namespace

int main() {
    using Dense27 = sinfer::targets::qwen3_5::Package;
    using Moe35   = sinfer::targets::qwen3_5_moe::Package;

    int failures = 0;

    const sinfer::ModelSamplingDefaults qwen3_6 = Dense27::sampling_defaults(Dense27::model_id);
    const sinfer::ModelSamplingDefaults qwen3_8 =
        Dense27::sampling_defaults(Dense27::qwen3_8_model_id);
    const sinfer::ModelSamplingDefaults qwen3_6_35 = Moe35::sampling_defaults(Moe35::model_id);

    const sinfer::SamplingPreset dense_thinking{
        .temperature = 1.0F, .top_k = 20, .top_p = 0.95F, .min_p = 0.0F};
    const sinfer::SamplingPreset dense_non_thinking{
        .temperature      = 0.7F,
        .top_k            = 20,
        .top_p            = 0.8F,
        .min_p            = 0.0F,
        .presence_penalty = 1.5F,
    };
    const sinfer::SamplingPreset moe_thinking{
        .temperature      = 1.0F,
        .top_k            = 20,
        .top_p            = 0.95F,
        .min_p            = 0.0F,
        .presence_penalty = 1.5F,
    };

    failures += check(same_preset(qwen3_6.thinking, dense_thinking),
                      "Qwen3.6-27B thinking defaults mismatch");
    failures += check(same_preset(qwen3_6.non_thinking, dense_non_thinking),
                      "Qwen3.6-27B non-thinking defaults mismatch");
    failures += check(same_preset(qwen3_8.thinking, dense_thinking) &&
                          same_preset(qwen3_8.non_thinking, dense_non_thinking),
                      "Qwen3.8-27B defaults mismatch");
    failures += check(same_preset(qwen3_6_35.thinking, moe_thinking) &&
                          same_preset(qwen3_6_35.non_thinking, dense_non_thinking),
                      "Qwen3.6-35B-A3B defaults mismatch");
    failures += check(throws_runtime([] { (void)Dense27::sampling_defaults("unknown"); }),
                      "unknown model received dense-27B sampling defaults");

    const sinfer::ResolvedSamplingParameters thinking = sinfer::runtime::resolve_sampling(
        qwen3_8, sinfer::SamplingMode::Thinking, sinfer::SamplingOverrides{});
    const sinfer::ResolvedSamplingParameters non_thinking = sinfer::runtime::resolve_sampling(
        qwen3_8, sinfer::SamplingMode::NonThinking, sinfer::SamplingOverrides{});
    failures += check(thinking.temperature == 1.0F && thinking.top_p == 0.95F &&
                          thinking.presence_penalty == 0.0F && thinking.seed == 0,
                      "omitted overrides did not select Qwen3.8 thinking defaults");
    failures += check(non_thinking.temperature == 0.7F && non_thinking.top_p == 0.8F &&
                          non_thinking.presence_penalty == 1.5F,
                      "omitted overrides did not select Qwen3.8 non-thinking defaults");

    sinfer::SamplingOverrides overrides;
    overrides.temperature       = 0.0F;
    overrides.top_k             = 0;
    overrides.top_p             = 0.0F;
    overrides.min_p             = 0.0F;
    overrides.presence_penalty  = 0.0F;
    overrides.frequency_penalty = -1.0F;
    overrides.seed              = 123;
    const sinfer::ResolvedSamplingParameters overridden =
        sinfer::runtime::resolve_sampling(qwen3_8, sinfer::SamplingMode::NonThinking, overrides);
    failures += check(overridden.temperature == 0.0F && overridden.top_k == 0 &&
                          overridden.top_p == 0.0F && overridden.presence_penalty == 0.0F &&
                          overridden.frequency_penalty == -1.0F && overridden.seed == 123,
                      "explicit zero sampling overrides were lost");

    overrides.temperature = std::numeric_limits<float>::quiet_NaN();
    failures += check(throws_invalid([&] {
                          (void)sinfer::runtime::resolve_sampling(
                              qwen3_8, sinfer::SamplingMode::Thinking, overrides);
                      }),
                      "non-finite sampling override was accepted");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
