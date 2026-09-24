// TTS bills characters, and reports the count it billed in X-Usage-Characters (SUROGATE-CHANGES
// #13). The count is Unicode code points, the same count as the 4096-character limit; a byte
// count overcharges Romanian, whose letters with diacritics take two UTF-8 bytes each (an
// 86-character sentence measured 94 bytes). Model-free: the frontend only.
#include "frontend.h"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

using sinfer::tts::input_characters;
using sinfer::tts::tokenize;

namespace {

bool refused(const std::string& input) {
    try {
        (void)tokenize(input);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

std::string repeat(const std::string& piece, std::size_t times) {
    std::string out;
    for (std::size_t i = 0; i < times; ++i)
        out += piece;
    return out;
}

}  // namespace

int main() {
    assert(input_characters("") == 0);
    assert(input_characters("Buna ziua") == 9);
    // Every Romanian diacritic is one character: ă â î ș ț and their capitals.
    const std::string diacritics = "ăâîșțĂÂÎȘȚ";
    assert(diacritics.size() == 20 && input_characters(diacritics) == 10);
    // As in the spike, where an 86-character sentence counted 94 bytes: this one has 88
    // characters, 9 of them with diacritics, in 97 bytes.
    const std::string sentence =
        "Bună ziua, comanda dumneavoastră a fost expediată și ajunge mâine până în ora prânzului.";
    assert(input_characters(sentence) == 88 && sentence.size() == 97);
    // Code points, as Python's len() counts them: a character outside the BMP is one (two UTF-16
    // units, four bytes); a combining mark is its own code point.
    assert(input_characters("😀") == 1);
    assert(input_characters("s\xCC\xA6") == 2);  // s + U+0326 COMBINING COMMA BELOW
    assert(input_characters("日本語") == 3);
    // A malformed byte counts once, as ICU reads it (U+FFFD).
    assert(input_characters("a\xFF"
                            "b") == 3);

    // The billed count is the count the 4096-character limit enforces: 4096 characters of
    // Romanian (4617 bytes, 56 sentences, within the frontend's chunk limits) are accepted, one
    // more is refused.
    const std::string sentence74 = "Ăsta este un test mai lung cu diacritice ăâîșț și încă ceva de spus aici. ";
    assert(input_characters(sentence74) == 74);
    const std::string at_limit = repeat(sentence74, 55) + repeat("ă", 26);
    assert(input_characters(at_limit) == 4096 && at_limit.size() == 4617);
    assert(!refused(at_limit));
    assert(input_characters(at_limit + "ă") == 4097);
    assert(refused(at_limit + "ă"));

    std::cout << "TTS input character checks passed\n";
    return 0;
}
