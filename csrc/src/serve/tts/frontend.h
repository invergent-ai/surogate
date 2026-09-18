// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace sinfer::tts {
// Romanian frontend 2.2.2, numeric-list boundaries, Magpie v2607 ByT5.
std::string normalize(const std::string& text);
std::vector<std::vector<int32_t>> tokenize(const std::string& text);
std::string casefold(const std::string& text);
} // namespace sinfer::tts
