from __future__ import annotations

import math
import struct

from tools.artifact.numeric import (
    decode_e2m1_word,
    decode_e4m3fn_word,
    valid_nvfp4_scale_word,
    valid_positive_fp32_word,
)


def _e4m3fn_oracle(word: int) -> float:
    sign = -1.0 if word >> 7 else 1.0
    exponent = (word >> 3) & 0xF
    fraction = word & 0x7
    if exponent == 0:
        return (
            math.copysign(0.0, sign)
            if fraction == 0
            else sign * fraction / 512.0
        )
    if exponent == 15 and fraction == 7:
        return math.copysign(math.nan, sign)
    return sign * (8 + fraction) * (2.0 ** (exponent - 10))


def test_all_e2m1_words_decode_exactly_and_preserve_signed_zero():
    magnitudes = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    for word in range(16):
        actual = decode_e2m1_word(word)
        expected = math.copysign(
            magnitudes[word & 7], -1.0 if word & 8 else 1.0
        )
        assert actual == expected
        assert math.copysign(1.0, actual) == math.copysign(1.0, expected)


def test_all_e4m3fn_words_match_independent_scalar_oracle():
    for word in range(256):
        actual = decode_e4m3fn_word(word)
        expected = _e4m3fn_oracle(word)
        if math.isnan(expected):
            assert math.isnan(actual)
            assert math.copysign(1.0, actual) == math.copysign(1.0, expected)
        else:
            assert actual == expected
            if expected == 0.0:
                assert math.copysign(1.0, actual) == math.copysign(
                    1.0, expected
                )


def test_nvfp4_stored_scale_and_divisor_word_validity():
    assert all(valid_nvfp4_scale_word(word) for word in range(0x7F))
    assert not valid_nvfp4_scale_word(0x7F)
    assert not any(valid_nvfp4_scale_word(word) for word in range(0x80, 0x100))

    positive = struct.unpack("<I", struct.pack("<f", 2.5))[0]
    assert valid_positive_fp32_word(positive)
    for word in (0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000):
        assert not valid_positive_fp32_word(word)
