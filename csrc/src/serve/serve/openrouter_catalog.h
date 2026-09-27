#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace sinfer::serve {

// Metadata is supplied by the operator: prices, regions and privacy declarations cannot
// be inferred from model weights. Check the envelope here; validate the complete document
// against OpenRouter's published schema before installing it. Load once at startup so a
// file replacement cannot change billing/capability declarations halfway through a run.
inline nlohmann::json load_openrouter_catalog(const std::string& path) {
    constexpr std::size_t max_bytes = 1U << 20;
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::invalid_argument("cannot read --openrouter-models-file"); }
    std::string text(max_bytes + 1, '\0');
    file.read(text.data(), static_cast<std::streamsize>(text.size()));
    const auto size = static_cast<std::size_t>(file.gcount());
    if (file.bad() || size == 0 || size > max_bytes) {
        throw std::invalid_argument("--openrouter-models-file must contain 1 byte to 1 MiB of JSON");
    }
    text.resize(size);
    const auto catalog = nlohmann::json::parse(text, nullptr, false);
    const auto invalid = [] {
        // Do not echo file contents: a mistaken path could have pointed at a secret.
        throw std::invalid_argument("invalid OpenRouter catalog: expected data with V2 model documents");
    };
    if (!catalog.is_object() || !catalog.contains("data") || !catalog.at("data").is_array() ||
        catalog.at("data").empty()) { invalid(); }
    for (const auto& model : catalog.at("data")) {
        if (!model.is_object()) { invalid(); }
        for (const char* field : {"id", "name", "schema_version"}) {
            if (!model.contains(field) || !model.at(field).is_string() ||
                model.at(field).get_ref<const std::string&>().empty()) { invalid(); }
        }
        const auto& version = model.at("schema_version").get_ref<const std::string&>();
        if (!version.starts_with("2.") || version.size() < 3 ||
            !std::all_of(version.begin() + 2, version.end(), [](char c) { return c >= '0' && c <= '9'; })) {
            invalid();
        }
        for (const char* field : {"input_modalities", "output_modalities"}) {
            if (!model.contains(field) || !model.at(field).is_array() || model.at(field).empty()) {
                invalid();
            }
            for (const auto& modality : model.at(field)) {
                if (!modality.is_object() || !modality.contains("type") ||
                    !modality.at("type").is_string()) { invalid(); }
                if (std::string_view(field) == "output_modalities" &&
                    (!modality.contains("supported_parameters") ||
                     !modality.at("supported_parameters").is_object())) { invalid(); }
            }
        }
        if (model.contains("object") || model.contains("owned_by")) { invalid(); }
        if (model.contains("is_ready") && !model.at("is_ready").is_boolean()) { invalid(); }
    }
    return catalog;
}

inline void validate_openrouter_catalog_models(const nlohmann::json& catalog,
                                                const std::vector<std::string>& served_ids) {
    for (const auto& model : catalog.at("data")) {
        const auto& id = model.at("id").get_ref<const std::string&>();
        if (std::find(served_ids.begin(), served_ids.end(), id) == served_ids.end()) {
            throw std::invalid_argument("OpenRouter catalog contains a model that is not served");
        }
    }
}

} // namespace sinfer::serve
