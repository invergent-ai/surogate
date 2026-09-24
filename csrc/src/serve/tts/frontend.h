// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sinfer::tts {
// Romanian frontend 2.2.2, numeric-list boundaries, Magpie v2607 ByT5.
std::string normalize(const std::string& text);
/// The input's text chunks (sentences once it is 45 words or longer), at most `max_characters`
/// characters of input (see input_characters). Longer inputs are split into more sentences, not
/// longer ones: there may be a chunk for every 16 characters allowed (at least 64). The text as
/// spoken (numbers, dates and abbreviations written out) may be four times the limit, and at most
/// max_spoken_characters: about 24 minutes of speech, within the 96 MiB a request may produce.
constexpr std::size_t default_max_characters = 4096;
constexpr std::size_t max_spoken_characters  = 16384;
std::vector<std::vector<int32_t>> tokenize(const std::string& text,
                                           std::size_t max_characters = default_max_characters);
/// The characters of a speech input, as billed and as the input limit counts them:
/// Unicode code points of the UTF-8 text as sent (ICU countChar32), so a precomposed Romanian
/// letter with a diacritic is one character, not the two bytes it takes. A decomposed one (base
/// letter plus combining mark) is two, and whitespace and bracketed spans that are not spoken
/// count too. Each ill-formed UTF-8 subsequence counts as one U+FFFD, as ICU and Python read it
/// (the HTTP layer refuses such input before it gets here).
std::size_t input_characters(const std::string& text);
std::string casefold(const std::string& text);
} // namespace sinfer::tts
