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

/// Bytes one row of `k` weights occupies in `type`'s blocks -- the stride between rows of a
/// GGML-block matrix. 0 when `type` is not a block format, or `k` is not a whole number of
/// blocks. Here rather than in the caller so a host TU needs no CUDA headers to walk the rows.
[[nodiscard]] std::int64_t ggml_row_bytes(QType type, std::int64_t k) noexcept;

} // namespace sinfer::ops
