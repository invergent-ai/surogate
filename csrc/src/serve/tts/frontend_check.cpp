// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
// Line-oriented test driver; never used by the serving process.
#include "frontend.h"
#include <nlohmann/json.hpp>
#include <iostream>

int main() {
    for (std::string line; std::getline(std::cin, line);) {
        try {
            auto text = nlohmann::json::parse(line).get<std::string>();
            std::cout << nlohmann::json{{"chunks", sinfer::tts::tokenize(text)},
                                        {"normalized", sinfer::tts::normalize(text)}}
                             .dump()
                      << '\n';
        } catch (const std::exception& e) {
            std::cout << nlohmann::json{{"error", e.what()}}.dump() << '\n';
        }
    }
}
