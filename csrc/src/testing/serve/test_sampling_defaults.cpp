#include "runtime/contract/sampling.h"
#include "serve/translate.h"

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
           actual.frequency_penalty == expected.frequency_penalty &&
           actual.repetition_penalty == expected.repetition_penalty;
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

    for (const std::string_view config : {"", "{}", R"({"eos_token_id":[2,3],"top_p":null})"}) {
        const auto defaults = sinfer::runtime::load_sampling_defaults(qwen3_8, config);
        failures += check(same_preset(defaults.thinking, qwen3_8.thinking) &&
                              same_preset(defaults.non_thinking, qwen3_8.non_thinking),
                          "absent sampling settings changed family defaults");
    }
    const auto configured = sinfer::runtime::load_sampling_defaults(qwen3_8, R"({
        "eos_token_id": [2,3], "temperature": 0.42, "top_k": 7, "top_p": null,
        "min_p": 0.1, "repetition_penalty": 1.12, "frequency_penalty": -0.4
    })");
    for (const auto mode : {sinfer::SamplingMode::Thinking, sinfer::SamplingMode::NonThinking}) {
        const auto resolved = sinfer::runtime::resolve_sampling(configured, mode, {});
        failures += check(resolved.temperature == 0.42F && resolved.top_k == 7 &&
                              resolved.min_p == 0.1F && resolved.repetition_penalty == 1.12F &&
                              resolved.frequency_penalty == -0.4F &&
                              resolved.top_p == qwen3_8.for_mode(mode).top_p &&
                              resolved.presence_penalty == qwen3_8.for_mode(mode).presence_penalty,
                          "generation config did not override only its specified fields in both modes");
    }
    const auto zero = sinfer::runtime::load_sampling_defaults(configured, R"({
        "temperature":0, "top_k":0, "top_p":0, "min_p":0,
        "presence_penalty":0, "frequency_penalty":0, "repetition_penalty":1
    })");
    failures += check(same_preset(zero.thinking, sinfer::SamplingPreset{.top_p = 0}) &&
                          same_preset(zero.non_thinking, zero.thinking),
                      "explicit zero generation config settings were lost");
    failures += check(sinfer::runtime::load_sampling_defaults(configured, R"({"top_k":-1})")
                              .thinking.top_k == 0,
                      "generation config top_k=-1 did not disable filtering");

    // Exercise the real wire/server translator and Engine resolver together. Each level
    // must only replace fields it actually supplies, including explicit zero overrides.
    sinfer::serve::ServeOptions server;
    server.sampling_overrides.temperature = 0.5F;
    server.sampling_overrides.top_k = 11;
    sinfer::serve::GenerationRequest request;
    auto resolve = [&] {
        const auto options = sinfer::serve::to_request_options(request, server);
        return sinfer::runtime::resolve_sampling(
            configured, sinfer::SamplingMode::NonThinking, options.execution.sampling);
    };
    auto resolved = resolve();
    failures += check(resolved.temperature == 0.5F && resolved.top_k == 11 &&
                          resolved.min_p == 0.1F && resolved.top_p == 0.8F,
                      "CLI > generation config > family precedence failed");
    request.sampling.temperature = 0.25;
    request.sampling.top_k = 0;
    request.sampling.repetition_penalty = 1.0;
    resolved = resolve();
    failures += check(resolved.temperature == 0.25F && resolved.top_k == 0 &&
                          resolved.repetition_penalty == 1.0F && resolved.min_p == 0.1F &&
                          resolved.top_p == 0.8F,
                      "request > CLI > generation config > family precedence failed");
    server.greedy = true;
    failures += check(resolve().temperature == 0.0F, "--greedy lost its forced temperature");

    for (const std::string_view config : {
             "null", "[]", "{", R"({"temperature":true})", R"({"temperature":"0.7"})",
             R"({"temperature":-1})", R"({"temperature":3})", R"({"temperature":1e999})",
             R"({"top_k":1.5})", R"({"top_k":true})", R"({"top_k":-2})",
             R"({"top_k":2147483648})", R"({"top_k":18446744073709551615})",
             R"({"top_p":1.1})", R"({"min_p":-0.1})", R"({"presence_penalty":3})",
             R"({"frequency_penalty":-3})", R"({"repetition_penalty":0})",
             R"({"repetition_penalty":-1})", R"({"repetition_penalty":1e-100})"}) {
        bool rejected = false;
        try {
            (void)sinfer::runtime::load_sampling_defaults(qwen3_8, config);
        } catch (const std::invalid_argument& error) {
            rejected = std::string_view(error.what()).starts_with("generation_config.json:");
        }
        failures += check(rejected, "invalid generation config accepted or missing config error context");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
