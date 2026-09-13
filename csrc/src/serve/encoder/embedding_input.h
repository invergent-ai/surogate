#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace sinfer::encoder {

inline void validate_embedding_input(std::span<const std::int32_t> tokens,
                                     std::int32_t vocab, std::int32_t max_tokens) {
    if (tokens.empty()) { throw std::invalid_argument("input sequences must not be empty"); }
    if (tokens.size() > static_cast<std::size_t>(max_tokens)) {
        throw std::invalid_argument("input sequence exceeds the model limit of " +
                                    std::to_string(max_tokens) + " tokens");
    }
    for (const auto id : tokens) {
        if (id < 0 || id >= vocab) {
            throw std::invalid_argument("input token IDs must be in [0," +
                                        std::to_string(vocab - 1) + "]");
        }
    }
}

} // namespace sinfer::encoder
