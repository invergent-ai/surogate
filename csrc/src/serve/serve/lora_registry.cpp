// LoRA adapter discovery and validation (serve/lora_registry.h).

#include "serve/lora_registry.h"

#include <nlohmann/json.hpp>

#include <cctype>
#include <cstring>
#include <fstream>
#include <cmath>
#include <limits>
#include <regex>
#include <stdexcept>

namespace sinfer::serve {
namespace {

using Json = nlohmann::ordered_json;

[[noreturn]] void bad(const std::string& name, const std::string& reason) {
    throw std::invalid_argument("--lora-modules '" + name + "': " + reason);
}

[[noreturn]] void unsupported(const std::string& name, const std::string& reason) {
    bad(name, "cannot load adapter: " + reason +
        ". Merge the adapter into its base checkpoint with `surogate merge`, "
        "then convert and serve the merged checkpoint.");
}

Json read_json(const std::string& name, const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { bad(name, "cannot read " + path.string()); }
    Json parsed = Json::parse(file, nullptr, false);
    if (parsed.is_discarded()) { bad(name, path.string() + " is not valid JSON"); }
    return parsed;
}

/// The safetensors header: an 8-byte little-endian length, then that many bytes
/// of JSON describing every tensor's dtype, shape and byte range.
Json read_safetensors_header(const std::string& name, const std::filesystem::path& path,
                             std::uint64_t& payload_begin) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { bad(name, "cannot read " + path.string()); }
    std::uint64_t header_bytes = 0;
    file.read(reinterpret_cast<char*>(&header_bytes), sizeof(header_bytes));
    if (!file || header_bytes == 0 || header_bytes > (1ULL << 28)) {
        bad(name, path.string() + " has no usable safetensors header");
    }
    std::string header(static_cast<std::size_t>(header_bytes), '\0');
    file.read(header.data(), static_cast<std::streamsize>(header_bytes));
    if (!file) { bad(name, path.string() + " is truncated inside its header"); }
    Json parsed = Json::parse(header, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) {
        bad(name, path.string() + " has a malformed safetensors header");
    }
    payload_begin = sizeof(header_bytes) + header_bytes;
    return parsed;
}

/// PEFT names a pair `<module>.lora_A.weight` / `.lora_B.weight`, under a
/// `base_model.model.` prefix that varies with how the adapter was exported.
/// The base module name is what remains once both are stripped.
enum class AdapterTensorKind { A, B, Magnitude, Bias, LoraBias, Base };
bool split_module(const std::string& module, std::int32_t& layer, std::string& kind);
bool split_lora_key(std::string_view key, std::string& module, AdapterTensorKind& kind) {
    const std::pair<std::string_view, AdapterTensorKind> suffixes[] = {
        {".lora_A.weight", AdapterTensorKind::A}, {".lora_B.weight", AdapterTensorKind::B},
        {".lora_embedding_A", AdapterTensorKind::A}, {".lora_embedding_B", AdapterTensorKind::B},
        {".lora_magnitude_vector.weight", AdapterTensorKind::Magnitude},
        {".lora_magnitude_vector", AdapterTensorKind::Magnitude},
        {".lora_B.bias", AdapterTensorKind::LoraBias}, {".bias", AdapterTensorKind::Bias},
        {".weight", AdapterTensorKind::Base}};
    bool matched = false;
    for (auto [suffix, type] : suffixes) {
        if (key.ends_with(suffix)) { key.remove_suffix(suffix.size()); kind = type; matched = true; break; }
    }
    if (!matched && (key.ends_with(".dt_bias") || key.ends_with(".expert_bias") || key.ends_with(".e_score_correction_bias"))) { kind = AdapterTensorKind::Bias; matched = true; }
    if (!matched) { return false; }
    for (const std::string_view prefix : {"base_model.model.", "base_model."}) {
        if (key.starts_with(prefix)) { key.remove_prefix(prefix.size()); break; }
    }
    if (const auto at = key.find(".modules_to_save."); at != std::string_view::npos) { key = key.substr(0, at); }
    if (key.ends_with(".base_layer")) { key.remove_suffix(std::string_view(".base_layer").size()); }
    module = std::string(key);
    return true;
}

struct RawTensor {
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

RawTensor read_entry(const std::string& name, const std::string& key, const Json& entry) {
    if (!entry.is_object() || !entry.contains("dtype") || !entry.contains("shape") ||
        !entry.contains("data_offsets")) {
        bad(name, "tensor '" + key + "' has an incomplete safetensors entry");
    }
    RawTensor raw;
    raw.dtype = entry.at("dtype").get<std::string>();
    for (const Json& extent : entry.at("shape")) { raw.shape.push_back(extent.get<std::int64_t>()); }
    const Json& offsets = entry.at("data_offsets");
    if (!offsets.is_array() || offsets.size() != 2) {
        bad(name, "tensor '" + key + "' has malformed data_offsets");
    }
    raw.begin = offsets.at(0).get<std::uint64_t>();
    raw.end   = offsets.at(1).get<std::uint64_t>();
    return raw;
}

} // namespace

void LoraRegistry::load(const std::vector<std::pair<std::string, std::string>>& modules,
                        std::uint32_t max_rank) {
    for (const auto& [name, path_text] : modules) {
        const std::filesystem::path directory(path_text);
        if (!std::filesystem::is_directory(directory)) {
            bad(name, path_text + " is not a directory");
        }
        const std::filesystem::path config_path  = directory / "adapter_config.json";
        const std::filesystem::path weights_path = directory / "adapter_model.safetensors";
        if (!std::filesystem::is_regular_file(config_path)) {
            bad(name, "no adapter_config.json in " + path_text);
        }
        if (!std::filesystem::is_regular_file(weights_path)) {
            bad(name, "no adapter_model.safetensors in " + path_text);
        }

        const Json config = read_json(name, config_path);
        if (config.value("peft_type", std::string("LORA")) != "LORA") {
            bad(name, "peft_type is " + config.value("peft_type", std::string("?")) +
                          ", only LORA adapters are served");
        }
        if (config.contains("modules_to_save") && !config.at("modules_to_save").is_null()) {
            for (const auto& module : config.at("modules_to_save")) {
                int layer = 0; std::string target;
                if (!module.is_string()) { bad(name, "modules_to_save must contain module names"); }
                const auto module_name = module.get<std::string>();
                if (!split_module(module_name, layer, target) || layer != -1 || target.ends_with(".bias")) {
                    unsupported(name, "full-weight module '" + module_name +
                        "' in modules_to_save is unsupported; full replacements are supported only for embeddings and output heads");
                }
            }
        }
        LoraAdapter adapter;
        adapter.name         = name;
        adapter.directory    = directory;
        adapter.weights_file = weights_path;
        adapter.rank         = config.value("r", 0);
        adapter.alpha        = config.value("lora_alpha", 0.0);
        if (adapter.rank <= 0) { bad(name, "adapter_config.json has no positive rank r"); }
        // PEFT scales the delta by alpha/r. `use_rslora` scales by alpha/sqrt(r)
        // instead, and silently applying the wrong one would change every adapted
        // projection by a constant factor -- fluent output, quietly wrong.
        if (config.value("use_rslora", false)) {
            adapter.scale = adapter.alpha / std::sqrt(static_cast<double>(adapter.rank));
        } else {
            adapter.scale = adapter.alpha / static_cast<double>(adapter.rank);
        }
        if (config.contains("target_modules") && config.at("target_modules").is_array()) {
            for (const Json& module : config.at("target_modules")) {
                if (module.is_string()) { adapter.target_modules.push_back(module.get<std::string>()); }
            }
        }

        std::uint64_t payload_begin = 0;
        const Json header = read_safetensors_header(name, weights_path, payload_begin);
        std::map<std::string, LoraTensorPair> pairs;
        for (const auto& [key, entry] : header.items()) {
            if (key == "__metadata__") { continue; }
            std::string module;
            AdapterTensorKind kind;
            if (!split_lora_key(key, module, kind)) {
                unsupported(name, "unsupported adapter tensor '" + key + "'");
            }
            const RawTensor raw = read_entry(name, key, entry);
            if (raw.dtype != "BF16" && raw.dtype != "F16" && raw.dtype != "F32") {
                unsupported(name, "tensor '" + key + "' has unsupported dtype " + raw.dtype +
                    "; adapter weights must be BF16, F16 or F32");
            }
            std::uint64_t elements = 1;
            if (raw.shape.empty()) { bad(name, "adapter tensor is scalar"); }
            for (auto extent : raw.shape) {
                if (extent <= 0 || extent > std::numeric_limits<std::int32_t>::max() ||
                    elements > std::numeric_limits<std::uint64_t>::max() / 4 / extent) {
                    bad(name, "adapter tensor shape is invalid");
                }
                elements *= extent;
            }
            if (raw.end < raw.begin || raw.end - raw.begin != elements * (raw.dtype == "F32" ? 4U : 2U) ||
                raw.end > std::filesystem::file_size(weights_path) - payload_begin) {
                bad(name, "tensor '" + key + "' has inconsistent shape or byte range");
            }
            LoraTensorPair& pair = pairs[module];
            pair.module          = module;
            if (kind == AdapterTensorKind::Base) {
                int layer = 0; std::string target;
                if (!split_module(module, layer, target) || layer != -1 || raw.shape.size() != 2) {
                    unsupported(name, "saved full-weight tensor '" + key +
                        "' is unsupported; only embedding and output-head matrices can be replaced");
                }
                if (pair.base_weight.bytes) { bad(name, "duplicate saved weight for '" + module + "'"); }
                pair.base_weight = {payload_begin + raw.begin, raw.end - raw.begin, elements, raw.dtype};
                const bool embedding = target != "lm_head";
                pair.base_in = static_cast<int>(raw.shape[embedding ? 0 : 1]);
                pair.base_out = static_cast<int>(raw.shape[embedding ? 1 : 0]);
                continue;
            }
            if (kind != AdapterTensorKind::A && kind != AdapterTensorKind::B) {
                if ((kind == AdapterTensorKind::LoraBias && !config.value("lora_bias", false)) ||
                    (kind == AdapterTensorKind::Magnitude && !config.value("use_dora", false))) {
                    bad(name, "tensor '" + key + "' is not declared by adapter_config.json");
                }
                auto& ref = kind == AdapterTensorKind::Magnitude ? pair.magnitude :
                            kind == AdapterTensorKind::Bias ? pair.bias : pair.lora_bias;
                if (ref.bytes) { bad(name, "duplicate adapter tensor for '" + module + "'"); }
                ref = {payload_begin + raw.begin, raw.end - raw.begin, elements, raw.dtype};
                continue;
            }
            if (raw.shape.size() < 2 || elements / raw.shape[0] > std::numeric_limits<std::int32_t>::max()) {
                bad(name, "adapter A/B tensor is not a matrix or flattened convolution");
            }
            if (kind == AdapterTensorKind::B && raw.shape.size() > 2) {
                for (std::size_t i = 2; i < raw.shape.size(); ++i) {
                    if (raw.shape[i] != 1) { unsupported(name, "LoRA B tensor '" + key + "' has an unsupported spatial kernel; only unit spatial kernels are supported"); }
                }
            }
            if (kind == AdapterTensorKind::A) {
                // A is [r, in]
                pair.rank      = static_cast<std::int32_t>(raw.shape[0]);
                pair.in_dim    = static_cast<std::int32_t>(elements / raw.shape[0]);
                pair.a_offset  = payload_begin + raw.begin;
                pair.a_bytes   = raw.end - raw.begin;
                pair.a_is_bf16 = raw.dtype == "BF16";
            } else {
                // B is [out, r]
                pair.out_dim   = static_cast<std::int32_t>(raw.shape[0]);
                pair.b_offset  = payload_begin + raw.begin;
                pair.b_bytes   = raw.end - raw.begin;
                pair.b_is_bf16 = raw.dtype == "BF16";
                pair.b_rank = static_cast<std::int32_t>(raw.shape[1]);
                if (pair.rank == 0) { pair.rank = static_cast<std::int32_t>(raw.shape[1]); }
            }
        }
        if (pairs.empty()) {
            bad(name, "adapter_model.safetensors holds no lora_A/lora_B pairs");
        }
        for (auto& [module, pair] : pairs) {
            if (pair.bias.bytes && config.value("bias", std::string("none")) == "none" &&
                !pair.base_weight.bytes) {
                bad(name, "saved bias for '" + module + "' is not declared in the adapter config");
            }
            if (pair.a_bytes == 0 && pair.b_bytes == 0 && pair.base_weight.bytes) {
                if (pair.magnitude.bytes || pair.lora_bias.bytes) { bad(name, "saved module has adapter extras without A/B"); }
                pair.rank = pair.b_rank = 1; pair.in_dim = pair.base_in; pair.out_dim = pair.base_out;
                if (pair.bias.bytes && pair.bias.elements != static_cast<std::uint64_t>(pair.out_dim)) { bad(name, "saved module bias shape is inconsistent"); }
                adapter.pairs.push_back(pair); continue;
            }
            if (pair.a_bytes == 0 && pair.b_bytes == 0 && pair.bias.bytes &&
                !pair.magnitude.bytes && !pair.lora_bias.bytes) {
                pair.bias_only = true;
                pair.rank = pair.b_rank = pair.in_dim = 1;
                pair.out_dim = static_cast<std::int32_t>(pair.bias.elements);
                pair.module += ".bias";
                adapter.pairs.push_back(pair);
                continue;
            }
            if (pair.a_bytes == 0 || pair.b_bytes == 0) {
                bad(name, "module '" + module + "' has only one half of its A/B pair");
            }
            if (pair.base_weight.bytes && (pair.base_in != pair.in_dim || pair.base_out != pair.out_dim)) {
                bad(name, "saved embedding/head weight disagrees with its adapter");
            }
            // PEFT matches overrides against the complete module name, in config order.
            const auto pattern_value = [&](const char* field, double fallback) {
                if (!config.contains(field) || config.at(field).is_null()) { return fallback; }
                if (!config.at(field).is_object()) { bad(name, std::string(field) + " must be an object"); }
                for (const auto& [pattern, value] : config.at(field).items()) {
                    try {
                        if (std::regex_match(module, std::regex("(.*\\.)?(" + pattern + ")$"))) {
                            return value.get<double>();
                        }
                    } catch (const std::regex_error&) { bad(name, "invalid module pattern '" + pattern + "'"); }
                }
                return fallback;
            };
            const double rank = pattern_value("rank_pattern", adapter.rank);
            const double alpha = pattern_value("alpha_pattern", adapter.alpha);
            if (!std::isfinite(rank) || rank <= 0 || rank != std::floor(rank) || rank > max_rank ||
                pair.rank != rank || pair.b_rank != pair.rank || !std::isfinite(alpha)) {
                bad(name, "module '" + module + "' has inconsistent rank/alpha or exceeds --max-lora-rank");
            }
            pair.scale = alpha / (config.value("use_rslora", false) ? std::sqrt(rank) : rank);
            if (config.value("use_dora", false) && !pair.magnitude.bytes) {
                bad(name, "DoRA module '" + module + "' has no magnitude vector");
            }
            for (const auto* ref : {&pair.magnitude, &pair.bias, &pair.lora_bias}) {
                if (ref->bytes && ref->elements != static_cast<std::uint64_t>(pair.out_dim)) {
                    bad(name, "magnitude or bias shape disagrees with '" + module + "'");
                }
            }
            adapter.pairs.push_back(pair);
        }
        adapters_.emplace(name, std::move(adapter));
    }
}

const LoraAdapter* LoraRegistry::find(const std::string& name) const {
    const auto found = adapters_.find(name);
    return found == adapters_.end() ? nullptr : &found->second;
}



namespace {

std::uint16_t f32_to_bf16(float value) {
    std::uint32_t word = 0;
    std::memcpy(&word, &value, sizeof(word));
    word += 0x7FFFU + ((word >> 16U) & 1U);
    return static_cast<std::uint16_t>(word >> 16U);
}

float f16_to_f32(std::uint16_t bits) {
    const std::uint32_t sign     = (bits & 0x8000U) << 16U;
    std::uint32_t exponent       = (bits >> 10U) & 0x1FU;
    std::uint32_t mantissa       = bits & 0x3FFU;
    if (exponent == 0) {
        if (mantissa == 0) {
            float zero = 0.0F;
            std::uint32_t word = sign;
            std::memcpy(&zero, &word, sizeof(zero));
            return zero;
        }
        while ((mantissa & 0x400U) == 0) { mantissa <<= 1U; --exponent; }
        exponent += 113;
        mantissa &= 0x3FFU;
    } else if (exponent == 31) {
        exponent = 255;
    } else {
        exponent += 112;
    }
    const std::uint32_t word = sign | (exponent << 23U) | (mantissa << 13U);
    float value              = 0.0F;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

/// Reads `bytes` at `offset` and returns them as BF16, whatever they were stored as.
std::vector<std::uint16_t> read_as_bf16(std::ifstream& file, std::uint64_t offset,
                                        std::uint64_t bytes, bool is_bf16, bool is_f16,
                                        std::size_t expected) {
    file.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::uint16_t> out(expected);
    if (is_bf16) {
        file.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
        return out;
    }
    if (is_f16) {
        std::vector<std::uint16_t> raw(expected);
        file.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(bytes));
        for (std::size_t i = 0; i < expected; ++i) { out[i] = f32_to_bf16(f16_to_f32(raw[i])); }
        return out;
    }
    std::vector<float> raw(expected);
    file.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(bytes));
    for (std::size_t i = 0; i < expected; ++i) { out[i] = f32_to_bf16(raw[i]); }
    return out;
}

/// Preserve the complete path inside a layer, including expert and shared-expert names.
bool split_module(const std::string& module, std::int32_t& layer, std::string& kind) {
    for (const std::string_view prefix : {"model.vision_tower.vision_model.", "vision_tower.vision_model.",
        "model.vision_model.", "vision_model.", "model.visual.", "visual.", "model.vision_tower.", "vision_tower."}) {
        if (module.starts_with(prefix)) { layer = -2; kind = "vision." + module.substr(prefix.size()); return true; }
    }
    for (const std::string_view prefix : {"model.embed_vision.", "embed_vision."}) {
        if (module.starts_with(prefix)) { layer = -2; kind = "vision.embed_vision." + module.substr(prefix.size()); return true; }
    }
    for (const std::string_view prefix : {"model.multi_modal_projector.", "multi_modal_projector."}) {
        if (module.starts_with(prefix)) { layer = -2; kind = "vision.projector." + module.substr(prefix.size()); return true; }
    }
    for (const std::string_view prefix : {"model.language_model.model.", "model.language_model.",
        "language_model.model.", "language_model.", "model.text_model.model.", "model.text_model.",
        "text_model.", "model.model.", "model.", ""}) {
        if (!module.starts_with(prefix)) { continue; }
        const auto name = module.substr(prefix.size());
        if (name == "embed_tokens" || name == "embedding" || name == "tok_embeddings" || name == "word_embeddings" ||
            name == "lm_head" || name == "lm_head.bias" || name == "embed_tokens.bias") {
            layer = -1; kind = name; return true;
        }
    }
    constexpr std::string_view kLayers = ".layers.";
    const std::size_t at = module.find(kLayers);
    const bool direct = module.starts_with("layers.");
    if (!direct && at == std::string::npos) { return false; }
    const auto prefix = direct ? std::string{} : module.substr(0, at);
    bool text = false;
    for (const std::string_view allowed : {"", "model", "model.model", "transformer", "language_model",
        "language_model.model", "model.language_model", "model.language_model.model", "text_model",
        "model.text_model", "model.text_model.model"}) {
        text |= prefix == allowed;
    }
    if (!text) { return false; }
    const std::size_t digits = direct ? std::string_view("layers.").size() : at + kLayers.size();
    std::size_t end          = digits;
    while (end < module.size() && (std::isdigit(static_cast<unsigned char>(module[end])) != 0)) {
        ++end;
    }
    if (end == digits || end >= module.size() || module[end] != '.') { return false; }
    layer = std::stoi(module.substr(digits, end - digits));
    if (end + 1 >= module.size()) { return false; }
    kind = module.substr(end + 1);
    return true;
}

} // namespace

std::vector<EngineOptions::LoraModulePayload> LoraRegistry::read_payloads(
    const LoraAdapter& adapter, std::vector<std::string>& skipped) {
    std::ifstream file(adapter.weights_file, std::ios::binary);
    if (!file) { bad(adapter.name, "cannot reopen " + adapter.weights_file.string()); }

    std::vector<EngineOptions::LoraModulePayload> payloads;
    for (const LoraTensorPair& pair : adapter.pairs) {
        EngineOptions::LoraModulePayload payload;
        if (!split_module(pair.module, payload.layer, payload.module)) {
            skipped.push_back(pair.module);
            continue;
        }
        payload.rank    = pair.rank;
        payload.in_dim  = pair.in_dim;
        payload.out_dim = pair.out_dim;
        payload.scale   = static_cast<float>(pair.scale);
        const auto read_vector = [&](const AdapterTensorRef& ref) {
            std::vector<float> values(ref.elements);
            if (!ref.bytes) { return values; }
            if (ref.dtype == "F32") {
                file.seekg(ref.offset);
                file.read(reinterpret_cast<char*>(values.data()), ref.bytes);
            } else {
                std::vector<std::uint16_t> bits(ref.elements);
                file.seekg(ref.offset);
                file.read(reinterpret_cast<char*>(bits.data()), ref.bytes);
                for (std::size_t i = 0; i < values.size(); ++i) {
                    if (ref.dtype == "F16") { values[i] = f16_to_f32(bits[i]); }
                    else {
                        const std::uint32_t word = static_cast<std::uint32_t>(bits[i]) << 16U;
                        std::memcpy(&values[i], &word, sizeof(float));
                    }
                }
            }
            return values;
        };
        if (pair.base_weight.bytes) {
            const auto& ref = pair.base_weight;
            payload.base_weight = read_as_bf16(file, ref.offset, ref.bytes, ref.dtype == "BF16", ref.dtype == "F16", ref.elements);
        }
        payload.magnitude = read_vector(pair.magnitude);
        payload.bias = read_vector(pair.bias);
        payload.lora_bias = read_vector(pair.lora_bias);
        if (pair.base_weight.bytes && !pair.a_bytes) {
            payload.a.assign(pair.in_dim, 0); payload.b.assign(pair.out_dim, 0);
            if (!file) { bad(adapter.name, "truncated while reading " + pair.module); }
            payloads.push_back(std::move(payload)); continue;
        }
        if (pair.bias_only) {
            payload.a.assign(1, 0); payload.b.assign(pair.out_dim, 0);
            if (!file) { bad(adapter.name, "truncated while reading " + pair.module); }
            payloads.push_back(std::move(payload)); continue;
        }
        payload.a = read_as_bf16(file, pair.a_offset, pair.a_bytes, pair.a_is_bf16,
                                 !pair.a_is_bf16 && pair.a_bytes ==
                                     static_cast<std::uint64_t>(pair.rank) * pair.in_dim * 2,
                                 static_cast<std::size_t>(pair.rank) * pair.in_dim);
        payload.b = read_as_bf16(file, pair.b_offset, pair.b_bytes, pair.b_is_bf16,
                                 !pair.b_is_bf16 && pair.b_bytes ==
                                     static_cast<std::uint64_t>(pair.out_dim) * pair.rank * 2,
                                 static_cast<std::size_t>(pair.out_dim) * pair.rank);
        if (!file) { bad(adapter.name, "truncated while reading " + pair.module); }
        payloads.push_back(std::move(payload));
    }
    return payloads;
}

} // namespace sinfer::serve
