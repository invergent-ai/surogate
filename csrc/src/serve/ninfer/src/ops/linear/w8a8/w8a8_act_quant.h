#pragma once

// surogate vendor patch (PATCHES.md #17): per-token symmetric int8 activation
// quantization for the W8A8-int IMMA prefill path.
//
// BF16 activations x[hidden, T] (each token's hidden values contiguous)
// quantize to int8 codes[T, hidden] plus one FP32 scale per token
// (scale = absmax / 127; codes = round(x / scale), clamped to [-127, 127]).
// Measured cost is 1.5-3% of the combined quant+GEMM time at T >= 1024
// (bench/ops/w8a8_imma_probe_bench).

#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// A8 engages at and above this token count: below it the activation-quant
// pre-pass and the IMMA tile shape do not pay for themselves (measured in
// bench/ops/w8a8_imma_probe_bench), and decode must stay A16 regardless.
inline constexpr std::int32_t kW8A8MinTokens = 512;

// Workspace bytes for the quantized activation planes of one call:
// T x hidden int8 codes (16-byte aligned) followed by T fp32 scales.
[[nodiscard]] std::size_t w8a8_act_quant_bytes(std::int32_t hidden, std::int32_t tokens) noexcept;

struct W8A8QuantizedActivations {
    const std::int8_t* codes;  // [tokens, hidden]
    const float* scales;       // [tokens]
};

// Quantize x (BF16, [hidden, tokens], contiguous) into `workspace` and return
// the plane pointers. `workspace` must hold w8a8_act_quant_bytes(...) bytes
// and be 16-byte aligned.
W8A8QuantizedActivations w8a8_act_quant(const Tensor& x, void* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
