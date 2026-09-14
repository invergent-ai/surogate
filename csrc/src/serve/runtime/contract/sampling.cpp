#include "runtime/contract/sampling.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace sinfer::runtime {
namespace {

void validate(const ResolvedSamplingParameters& sampling) {
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty) ||
        !std::isfinite(sampling.repetition_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (sampling.temperature < 0.0F || sampling.temperature > 2.0F) {
        throw std::invalid_argument("temperature must be in [0,2]");
    }
    if (sampling.top_k < 0) { throw std::invalid_argument("top_k must be nonnegative"); }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in [0,1]");
    }
    if (sampling.min_p < 0.0F || sampling.min_p > 1.0F) {
        throw std::invalid_argument("min_p must be in [0,1]");
    }
    if (sampling.presence_penalty < -2.0F || sampling.presence_penalty > 2.0F) {
        throw std::invalid_argument("presence_penalty must be in [-2,2]");
    }
    if (sampling.frequency_penalty < -2.0F || sampling.frequency_penalty > 2.0F) {
        throw std::invalid_argument("frequency_penalty must be in [-2,2]");
    }
    if (sampling.repetition_penalty <= 0.0F) {
        throw std::invalid_argument("repetition_penalty must be positive");
    }
}

} // namespace

ModelSamplingDefaults load_sampling_defaults(ModelSamplingDefaults defaults,
                                             std::string_view generation_config_json) {
    if (generation_config_json.empty()) { return defaults; }
    try {
        const auto config = nlohmann::json::parse(generation_config_json);
        if (!config.is_object()) { throw std::invalid_argument("must be a JSON object"); }
        const auto read_float = [&](const char* field, float SamplingPreset::*member,
                                    double low, double high, bool positive = false) {
            const auto it = config.find(field);
            if (it == config.end() || it->is_null()) { return; }
            if (!it->is_number()) {
                throw std::invalid_argument(std::string(field) + " must be a number");
            }
            const double value = it->get<double>();
            if (!std::isfinite(value) || value < low || value > high ||
                (positive && static_cast<float>(value) <= 0.0F)) {
                throw std::invalid_argument(std::string(field) + " is outside the supported range");
            }
            defaults.thinking.*member = defaults.non_thinking.*member = static_cast<float>(value);
        };
        read_float("temperature", &SamplingPreset::temperature, 0.0, 2.0);
        read_float("top_p", &SamplingPreset::top_p, 0.0, 1.0);
        read_float("min_p", &SamplingPreset::min_p, 0.0, 1.0);
        read_float("presence_penalty", &SamplingPreset::presence_penalty, -2.0, 2.0);
        read_float("frequency_penalty", &SamplingPreset::frequency_penalty, -2.0, 2.0);
        read_float("repetition_penalty", &SamplingPreset::repetition_penalty, 0.0,
                   std::numeric_limits<float>::max(), true);
        if (const auto it = config.find("top_k"); it != config.end() && !it->is_null()) {
            if (!it->is_number_integer() || it->get<double>() < -1.0 ||
                it->get<double>() > std::numeric_limits<std::int32_t>::max()) {
                throw std::invalid_argument("top_k must be -1 (no limit) or a nonnegative int32");
            }
            const auto value = it->get<std::int32_t>();
            defaults.thinking.top_k = defaults.non_thinking.top_k = value == -1 ? 0 : value;
        }
    } catch (const std::exception& error) {
        throw std::invalid_argument(std::string("generation_config.json: ") + error.what());
    }
    return defaults;
}

ResolvedSamplingParameters resolve_sampling(const ModelSamplingDefaults& defaults,
                                            SamplingMode mode, const SamplingOverrides& overrides) {
    const SamplingPreset& preset = defaults.for_mode(mode);
    ResolvedSamplingParameters resolved{
        .temperature       = overrides.temperature.value_or(preset.temperature),
        .top_k             = overrides.top_k.value_or(preset.top_k),
        .top_p             = overrides.top_p.value_or(preset.top_p),
        .min_p             = overrides.min_p.value_or(preset.min_p),
        .presence_penalty  = overrides.presence_penalty.value_or(preset.presence_penalty),
        .frequency_penalty = overrides.frequency_penalty.value_or(preset.frequency_penalty),
        .repetition_penalty = overrides.repetition_penalty.value_or(preset.repetition_penalty),
        .seed              = overrides.seed.value_or(0),
    };
    resolved.logit_bias = overrides.logit_bias;
    for (const auto& [token, bias] : resolved.logit_bias) {
        if (token < 0 || !std::isfinite(bias) || bias < -100.0F || bias > 100.0F) {
            throw std::invalid_argument("logit_bias requires nonnegative token ids and finite values in [-100,100]");
        }
    }
    validate(resolved);
    return resolved;
}

} // namespace sinfer::runtime
