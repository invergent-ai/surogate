// LoRA adapter discovery and validation (serve/lora_registry.h).

#include "serve/lora_registry.h"

#include <nlohmann/json.hpp>

#include <cstring>
#include <fstream>
#include <cmath>
#include <stdexcept>

namespace sinfer::serve {
namespace {

using Json = nlohmann::json;

[[noreturn]] void bad(const std::string& name, const std::string& reason) {
    throw std::invalid_argument("--lora-modules '" + name + "': " + reason);
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
bool split_lora_key(std::string_view key, std::string& module, bool& is_a) {
    constexpr std::string_view kA = ".lora_A.weight";
    constexpr std::string_view kB = ".lora_B.weight";
    if (key.size() > kA.size() && key.substr(key.size() - kA.size()) == kA) {
        is_a = true;
        key  = key.substr(0, key.size() - kA.size());
    } else if (key.size() > kB.size() && key.substr(key.size() - kB.size()) == kB) {
        is_a = false;
        key  = key.substr(0, key.size() - kB.size());
    } else {
        return false;
    }
    for (const std::string_view prefix : {std::string_view("base_model.model."),
                                          std::string_view("base_model.")}) {
        if (key.size() > prefix.size() && key.substr(0, prefix.size()) == prefix) {
            key = key.substr(prefix.size());
            break;
        }
    }
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
        LoraAdapter adapter;
        adapter.name         = name;
        adapter.directory    = directory;
        adapter.weights_file = weights_path;
        adapter.rank         = config.value("r", 0);
        adapter.alpha        = config.value("lora_alpha", 0.0);
        if (adapter.rank <= 0) { bad(name, "adapter_config.json has no positive rank r"); }
        if (static_cast<std::uint32_t>(adapter.rank) > max_rank) {
            bad(name, "rank " + std::to_string(adapter.rank) + " exceeds --max-lora-rank " +
                          std::to_string(max_rank));
        }
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
            bool is_a = false;
            if (!split_lora_key(key, module, is_a)) { continue; }
            const RawTensor raw = read_entry(name, key, entry);
            if (raw.shape.size() != 2) {
                bad(name, "tensor '" + key + "' is not a matrix");
            }
            if (raw.dtype != "BF16" && raw.dtype != "F16" && raw.dtype != "F32") {
                bad(name, "tensor '" + key + "' has unsupported dtype " + raw.dtype);
            }
            LoraTensorPair& pair = pairs[module];
            pair.module          = module;
            if (is_a) {
                // A is [r, in]
                pair.rank      = static_cast<std::int32_t>(raw.shape[0]);
                pair.in_dim    = static_cast<std::int32_t>(raw.shape[1]);
                pair.a_offset  = payload_begin + raw.begin;
                pair.a_bytes   = raw.end - raw.begin;
                pair.a_is_bf16 = raw.dtype == "BF16";
            } else {
                // B is [out, r]
                pair.out_dim   = static_cast<std::int32_t>(raw.shape[0]);
                pair.b_offset  = payload_begin + raw.begin;
                pair.b_bytes   = raw.end - raw.begin;
                pair.b_is_bf16 = raw.dtype == "BF16";
                if (pair.rank == 0) { pair.rank = static_cast<std::int32_t>(raw.shape[1]); }
            }
        }
        if (pairs.empty()) {
            bad(name, "adapter_model.safetensors holds no lora_A/lora_B pairs");
        }
        for (auto& [module, pair] : pairs) {
            if (pair.a_bytes == 0 || pair.b_bytes == 0) {
                bad(name, "module '" + module + "' has only one half of its A/B pair");
            }
            if (pair.rank != adapter.rank) {
                bad(name, "module '" + module + "' has rank " + std::to_string(pair.rank) +
                              " but adapter_config.json says " + std::to_string(adapter.rank));
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

} // namespace sinfer::serve
