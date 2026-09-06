#pragma once

// Host decode of GGML blocks into the W8G32 form the CPU expert kernels read. The device
// gather does the same thing on its way into the slot pool; this is the host's copy of that
// one step, so an expert the cache missed and the host computed answers the same as one it hit.

#include "core/tensor.h"

#include <cstdint>

namespace sinfer::ops {

/// Decodes `k` consecutive weights of one GGML-block row into 32 int8 codes and one FP16
/// scale per group (amax/127, exactly as the gather requantises). `codes` holds k bytes and
/// `scales` k/32 halves. `k` must be a multiple of the block's value count.
/// Returns false when `type` is not a block format this build decodes, so a caller that
/// reaches here with something else is refused rather than reading one layout as another.
[[nodiscard]] bool ggml_decode_row_w8(QType type, const void* blocks, std::int64_t k,
                                      std::int8_t* codes, std::uint16_t* scales) noexcept;

/// Repacks one row of `k` weights stored as 4-bit affine GGML blocks (Q4_K, Q4_0, Q4_1) into
/// Q4G32AM planes -- 16 code bytes, one FP16 scale and one FP16 minimum per 32 -- without
/// passing through W8. A Q4_K sub-block *is* a Q4G32AM group (scale d*sc, minimum -dmin*m, the
/// same sixteen levels; GGML's low-half/high-half nibble order repacked to the pairwise one),
/// and Q4_0 / Q4_1 the same with a fixed or a stored minimum, so the conversion is exact to
/// the FP16 rounding of the two endpoints. Through W8 it is not: the int8 grid is symmetric
/// about zero, an affine grid's levels land off it, and the refit to sixteen levels rounds
/// them again -- +0.45 % perplexity on GLM-5.3-Flash, measured. Returns false for any other
/// type; the caller then takes the W8 route and its refit.
[[nodiscard]] bool ggml_row_to_q4g32am(QType type, const void* blocks, std::int64_t k,
                                       std::uint8_t* codes, std::uint16_t* scales,
                                       std::uint16_t* mins) noexcept;

/// Decodes one row into floats, group by group, with the codec the gather uses -- the oracle
/// for the two conversions above.
[[nodiscard]] bool ggml_decode_row_float(QType type, const void* blocks, std::int64_t k,
                                         float* out) noexcept;

/// Bytes one row of `k` weights occupies in `type`'s blocks -- the stride between rows of a
/// GGML-block matrix. 0 when `type` is not a block format, or `k` is not a whole number of
/// blocks. Here rather than in the caller so a host TU needs no CUDA headers to walk the rows.
[[nodiscard]] std::int64_t ggml_row_bytes(QType type, std::int64_t k) noexcept;

} // namespace sinfer::ops
