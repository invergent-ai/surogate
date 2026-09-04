"""Closed registry of persistent SInfer tensor numeric formats."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from types import MappingProxyType
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class DirectFormat:
    """One fixed-width word per logical tensor element."""

    name: str
    word_bytes: int


@dataclass(frozen=True, slots=True)
class QuantFormat:
    """Signed grouped codes with one binary16 multiplier per group."""

    name: str
    bits: int
    group_size: int
    qmin: int
    qmax: int


@dataclass(frozen=True, slots=True)
class Nvfp4Format:
    """E2M1 weights with one E4M3FN scale word per K-axis group."""

    name: str
    group_size: int


@dataclass(frozen=True, slots=True)
class Fp8RowFormat:
    """E4M3FN weights with one BF16 multiplier per logical row."""

    name: str


@dataclass(frozen=True, slots=True)
class Fp8BlockFormat:
    """E4M3FN weights with one FP32 multiplier per 128x128 block: Hugging Face's fine-grained
    FP8 (`weight_scale_inv` over `weight_block_size = [128, 128]`)."""

    name: str
    block: int = 128


NumericFormat: TypeAlias = DirectFormat | QuantFormat | Nvfp4Format | Fp8RowFormat | Fp8BlockFormat


BF16 = DirectFormat("BF16", 2)
FP32 = DirectFormat("FP32", 4)
I32 = DirectFormat("I32", 4)

Q4G64_F16S = QuantFormat("Q4G64_F16S", 4, 64, -8, 7)
Q5G64_F16S = QuantFormat("Q5G64_F16S", 5, 64, -16, 15)
Q6G64_F16S = QuantFormat("Q6G64_F16S", 6, 64, -32, 31)
W8G32_F16S = QuantFormat("W8G32_F16S", 8, 32, -127, 127)
NVFP4 = Nvfp4Format("NVFP4", 16)
FP8_E4M3FN_ROW_BF16S = Fp8RowFormat("FP8_E4M3FN_ROW_BF16S")
FP8_E4M3FN_BLK128_F32S = Fp8BlockFormat("FP8_E4M3FN_BLK128_F32S")

@dataclass(frozen=True, slots=True)
class GgmlBlockFormat:
    """A GGML block format kept as the GGUF stores it.

    Every K-quant is a 256-value superblock. The rest are plain 32-value blocks: Q8_0 with one
    scale, Q4_1/Q5_1 with a scale and an additive minimum, and IQ4_NL whose four-bit codes index
    a sixteen-entry table of int8 levels rather than standing for themselves. A quantiser
    reaches for a 32-value block when the reduction axis is not a multiple of 256 and no
    superblock fits.
    """

    name: str
    bits: float
    block_bytes: int
    values_per_block: int = 256

Q2_K = GgmlBlockFormat("Q2_K", 2.625, 84)
Q3_K = GgmlBlockFormat("Q3_K", 3.4375, 110)
Q4_K = GgmlBlockFormat("Q4_K", 4.5, 144)
Q5_K = GgmlBlockFormat("Q5_K", 5.5, 176)
Q6_K = GgmlBlockFormat("Q6_K", 6.5625, 210)
Q8_0 = GgmlBlockFormat("Q8_0", 8.5, 34, 32)
Q4_1 = GgmlBlockFormat("Q4_1", 5.0, 20, 32)
Q5_1 = GgmlBlockFormat("Q5_1", 6.0, 24, 32)
IQ4_NL = GgmlBlockFormat("IQ4_NL", 4.5, 18, 32)
Q4_0 = GgmlBlockFormat("Q4_0", 4.5, 18, 32)
Q5_0 = GgmlBlockFormat("Q5_0", 5.5, 22, 32)
# The importance-matrix quants (codebook rows plus signs, superblocks of 256), the ternary
# pair, the two microscaling floats (MXFP4 32 under an E8M0 exponent, NVFP4 64 under four
# UE4M3 sub-scales) and the plain 1- and 2-bit blocks (128 and 64 values under one scale).
IQ2_XXS = GgmlBlockFormat("IQ2_XXS", 2.0625, 66)
IQ2_XS = GgmlBlockFormat("IQ2_XS", 2.3125, 74)
IQ2_S = GgmlBlockFormat("IQ2_S", 2.5625, 82)
IQ3_XXS = GgmlBlockFormat("IQ3_XXS", 3.0625, 98)
IQ3_S = GgmlBlockFormat("IQ3_S", 3.4375, 110)
IQ1_S = GgmlBlockFormat("IQ1_S", 1.5625, 50)
IQ1_M = GgmlBlockFormat("IQ1_M", 1.75, 56)
IQ4_XS = GgmlBlockFormat("IQ4_XS", 4.25, 136)
TQ1_0 = GgmlBlockFormat("TQ1_0", 1.6875, 54)
TQ2_0 = GgmlBlockFormat("TQ2_0", 2.0625, 66)
MXFP4 = GgmlBlockFormat("MXFP4", 4.25, 17, 32)
NVFP4_GGML = GgmlBlockFormat("NVFP4_GGML", 4.5, 36, 64)
Q1_0 = GgmlBlockFormat("Q1_0", 1.125, 18, 128)
Q2_0 = GgmlBlockFormat("Q2_0", 2.25, 18, 64)


DIRECT_FORMATS = MappingProxyType(
    {item.name: item for item in (BF16, FP32, I32)}
)
QUANT_FORMATS = MappingProxyType(
    {
        item.name: item
        for item in (Q4G64_F16S, Q5G64_F16S, Q6G64_F16S, W8G32_F16S)
    }
)
NVFP4_FORMATS = MappingProxyType({NVFP4.name: NVFP4})
FP8_ROW_FORMATS = MappingProxyType(
    {FP8_E4M3FN_ROW_BF16S.name: FP8_E4M3FN_ROW_BF16S}
)
GGML_BLOCK_FORMATS = MappingProxyType(
    {item.name: item for item in (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, Q4_1, Q5_1, IQ4_NL, Q4_0, Q5_0,
                            IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M, IQ4_XS, TQ1_0, TQ2_0, MXFP4, NVFP4_GGML, Q1_0, Q2_0)}
)
FP8_BLOCK_FORMATS = {FP8_E4M3FN_BLK128_F32S.name: FP8_E4M3FN_BLK128_F32S}

NUMERIC_FORMATS = MappingProxyType(
    {**DIRECT_FORMATS, **QUANT_FORMATS, **NVFP4_FORMATS, **FP8_ROW_FORMATS, **FP8_BLOCK_FORMATS,
     **GGML_BLOCK_FORMATS}
)


_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def decode_e2m1_word(word: int) -> float:
    """Decode one exact four-bit E2M1 word, including signed zero."""

    if type(word) is not int or not 0 <= word <= 0xF:
        raise ValueError("E2M1 word must be an integer in [0, 15]")
    magnitude = _E2M1_MAGNITUDES[word & 0x7]
    return math.copysign(magnitude, -1.0 if word & 0x8 else 1.0)


def decode_e4m3fn_word(word: int) -> float:
    """Decode one exact eight-bit E4M3FN word."""

    if type(word) is not int or not 0 <= word <= 0xFF:
        raise ValueError("E4M3FN word must be an integer in [0, 255]")
    sign = -1.0 if word & 0x80 else 1.0
    exponent = (word >> 3) & 0xF
    fraction = word & 0x7
    if exponent == 0:
        if fraction == 0:
            return math.copysign(0.0, sign)
        return sign * fraction * (2.0**-9)
    if exponent == 0xF and fraction == 0x7:
        return math.copysign(math.nan, sign)
    return sign * (1.0 + fraction / 8.0) * (2.0 ** (exponent - 7))


def valid_nvfp4_scale_word(word: int) -> bool:
    """Return whether *word* is an admitted nonnegative finite E4M3FN scale."""

    return (
        type(word) is int
        and 0 <= word <= 0xFF
        and word & 0x80 == 0
        and word != 0x7F
    )


def valid_fp8_weight_word(word: int) -> bool:
    """Return whether *word* is a finite E4M3FN weight code."""

    return type(word) is int and 0 <= word <= 0xFF and (word & 0x7F) != 0x7F


def valid_fp8_row_scale_word(word: int) -> bool:
    """Return whether *word* is a nonnegative finite BF16 multiplier."""

    if type(word) is not int or not 0 <= word <= 0xFFFF or word & 0x8000:
        return False
    value = struct.unpack("<f", struct.pack("<I", word << 16))[0]
    return math.isfinite(value)


def valid_positive_fp32_word(word: int) -> bool:
    """Return whether an IEEE binary32 word represents a finite positive value."""

    if type(word) is not int or not 0 <= word <= 0xFFFFFFFF:
        return False
    value = struct.unpack("<f", struct.pack("<I", word))[0]
    return math.isfinite(value) and value > 0.0


def get_format(name: str) -> NumericFormat:
    """Return the registered format named *name*."""

    try:
        return NUMERIC_FORMATS[name]
    except KeyError:
        raise ValueError(f"unknown numeric format: {name!r}") from None


__all__ = [
    "GGML_BLOCK_FORMATS",
    "GgmlBlockFormat",
    "Q2_K",
    "Q3_K",
    "Q4_K",
    "Q5_K",
    "Q6_K",
    "Q8_0",
    "Q4_1",
    "Q5_1",
    "IQ4_NL",
    "Q4_0",
    "Q5_0",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ3_XXS",
    "IQ3_S",
    "IQ1_S",
    "IQ1_M",
    "IQ4_XS",
    "TQ1_0",
    "TQ2_0",
    "MXFP4",
    "NVFP4_GGML",
    "Q1_0",
    "Q2_0",
    "BF16",
    "DIRECT_FORMATS",
    "DirectFormat",
    "FP8_E4M3FN_ROW_BF16S",
    "FP8_ROW_FORMATS",
    "FP32",
    "Fp8RowFormat",
    "Fp8BlockFormat",
    "FP8_E4M3FN_BLK128_F32S",
    "I32",
    "NUMERIC_FORMATS",
    "NVFP4",
    "NVFP4_FORMATS",
    "Nvfp4Format",
    "NumericFormat",
    "Q4G64_F16S",
    "Q5G64_F16S",
    "Q6G64_F16S",
    "QUANT_FORMATS",
    "QuantFormat",
    "W8G32_F16S",
    "decode_e2m1_word",
    "decode_e4m3fn_word",
    "get_format",
    "valid_fp8_row_scale_word",
    "valid_fp8_weight_word",
    "valid_nvfp4_scale_word",
    "valid_positive_fp32_word",
]
