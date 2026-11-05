#pragma once
// MXFP8 vision projections (`surogate serve --vision-fp8`).
//
// Off by default. When on, and the device runs block-scaled FP8 (compute capability 10.0 and
// up), the four projections of every vision-tower block -- qkv, the attention output, fc1 and
// fc2 -- run on cuBLASLt's MXFP8 GEMM: e4m3 operands with one power-of-two scale per 32 values
// along K, FP32 accumulation, BF16 output (ops/linear/mxfp8/mxfp8.h). The weights are quantized
// once at startup into a copy beside the BF16 ones (about half their size); the activations at
// each projection. The patch embedding, the mergers, attention and the text model are
// untouched. Outputs change by FP8 rounding.
//
// The switch is process-wide and is set once, before the first model loads. Reading it is a
// relaxed atomic load.
namespace sinfer::ops {
void set_vision_fp8(bool enabled) noexcept;
[[nodiscard]] bool vision_fp8() noexcept;
} // namespace sinfer::ops
