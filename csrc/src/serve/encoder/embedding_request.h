#pragma once

#include "encoder/embedding_input.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <bit>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace sinfer::encoder {

struct EmbeddingRequestError : std::invalid_argument {
    std::string param;
    std::string code;
    int status;
    EmbeddingRequestError(std::string message, std::string field,
                          std::string error_code = "invalid_value", int http_status = 400)
        : std::invalid_argument(std::move(message)), param(std::move(field)),
          code(std::move(error_code)), status(http_status) {}
};

struct EmbeddingRequest {
    std::vector<std::vector<std::int32_t>> sequences;
    std::int32_t dimensions = 0;
    bool base64 = false;
};

inline EmbeddingRequest parse_embedding_request(
    const nlohmann::json& body, const std::string& model, std::int32_t vocab,
    std::int32_t max_tokens, std::int32_t dimensions,
    const std::function<std::vector<std::int32_t>(const std::string&)>& encode) {
    if (!body.is_object()) { throw EmbeddingRequestError("request must be an object", "body"); }
    if (body.contains("model") && !body["model"].is_null()) {
        if (!body["model"].is_string()) {
            throw EmbeddingRequestError("model must be a string", "model");
        }
        if (body["model"].get<std::string>() != model) {
            throw EmbeddingRequestError("model not found: " + body["model"].get<std::string>(),
                                         "model", "model_not_found", 404);
        }
    }
    EmbeddingRequest request;
    request.dimensions = dimensions;
    if (body.contains("dimensions") && !body["dimensions"].is_null()) {
        const auto& count = body["dimensions"];
        if (!count.is_number_integer() || count < 1 || count > dimensions) {
            throw EmbeddingRequestError("dimensions must be an integer in [1," +
                                         std::to_string(dimensions) + "]", "dimensions");
        }
        request.dimensions = count.get<std::int32_t>();
    }
    if (body.contains("encoding_format") && !body["encoding_format"].is_null()) {
        const auto& format = body["encoding_format"];
        if (!format.is_string() || (format != "float" && format != "base64")) {
            throw EmbeddingRequestError("encoding_format must be 'float' or 'base64'",
                                         "encoding_format");
        }
        request.base64 = format == "base64";
    }
    if (!body.contains("input")) { throw EmbeddingRequestError("input is required", "input"); }
    const auto add = [&](const nlohmann::json& value, bool text) {
        std::vector<std::int32_t> tokens;
        if (text) {
            if (!value.is_string() || value.get_ref<const std::string&>().empty()) {
                throw EmbeddingRequestError("input texts must be non-empty strings", "input");
            }
            tokens = encode(value.get_ref<const std::string&>());
        } else {
            if (!value.is_array() || value.empty()) {
                throw EmbeddingRequestError("input token sequences must be non-empty arrays", "input");
            }
            if (value.size() > static_cast<std::size_t>(max_tokens)) {
                throw EmbeddingRequestError("input sequence exceeds the model token limit", "input");
            }
            tokens.reserve(value.size());
            for (const auto& id : value) {
                // Check in JSON's integer domain before narrowing: large unsigned IDs,
                // booleans, and floating-point values must never become GPU row indices.
                if (!id.is_number_integer() || id < 0 || id >= vocab) {
                    throw EmbeddingRequestError("input token IDs must be integers in [0," +
                                                 std::to_string(vocab - 1) + "]", "input");
                }
                tokens.push_back(id.get<std::int32_t>());
            }
        }
        try { validate_embedding_input(tokens, vocab, max_tokens); }
        catch (const std::invalid_argument& error) { throw EmbeddingRequestError(error.what(), "input"); }
        request.sequences.push_back(std::move(tokens));
    };
    const auto& input = body["input"];
    if (input.is_string()) { add(input, true); }
    else if (input.is_array() && !input.empty()) {
        if (input.front().is_number()) { add(input, false); }
        else {
            const bool text = input.front().is_string();
            for (const auto& sequence : input) { add(sequence, text); }
        }
    } else { throw EmbeddingRequestError("input must be text or a non-empty array", "input"); }
    return request;
}

inline nlohmann::json encode_embedding(std::vector<float> values, std::int32_t dimensions,
                                       bool base64) {
    if (dimensions <= 0 || static_cast<std::size_t>(dimensions) > values.size()) {
        throw std::logic_error("encoder returned fewer dimensions than requested");
    }
    if (static_cast<std::size_t>(dimensions) < values.size()) {
        values.resize(dimensions);
        // Matryoshka embeddings retain the leading coordinates and are normalized
        // again after shortening. Preserve the existing full-size output unchanged.
        double norm = 0.0;
        for (const float value : values) { norm += static_cast<double>(value) * value; }
        if (norm > 0.0) {
            norm = std::sqrt(norm);
            for (float& value : values) { value = static_cast<float>(value / norm); }
        }
    }
    if (!base64) { return values; }
    std::string bytes;
    bytes.reserve(values.size() * sizeof(float));
    for (const float value : values) {
        const auto bits = std::bit_cast<std::uint32_t>(value);
        for (int shift = 0; shift < 32; shift += 8) {
            bytes.push_back(static_cast<char>((bits >> shift) & 0xff));
        }
    }
    return httplib::detail::base64_encode(bytes);
}

} // namespace sinfer::encoder
