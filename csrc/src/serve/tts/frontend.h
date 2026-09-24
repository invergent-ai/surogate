// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sinfer::tts {
// Romanian frontend 2.2.2, numeric-list boundaries, Magpie v2607 ByT5.
std::string normalize(const std::string& text);
std::vector<std::vector<int32_t>> tokenize(const std::string& text);
/// The characters of a speech input, as billed and as the 4096-character limit counts them:
/// Unicode code points of the UTF-8 text as sent (ICU countChar32), so a precomposed Romanian
/// letter with a diacritic is one character, not the two bytes it takes. A decomposed one (base
/// letter plus combining mark) is two, and whitespace and bracketed spans that are not spoken
/// count too. Each ill-formed UTF-8 subsequence counts as one U+FFFD, as ICU and Python read it
/// (the HTTP layer refuses such input before it gets here).
std::size_t input_characters(const std::string& text);
std::string casefold(const std::string& text);
} // namespace sinfer::tts
