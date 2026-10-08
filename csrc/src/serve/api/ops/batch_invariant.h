#pragma once

// Batch-invariant numerics (`surogate serve --batch-invariant`, PATCHES.md #99).
//
// Off by default. When on, no kernel choice that affects a token's arithmetic follows the width
// of the round the token shares -- how many prompt and decode columns one forward carries -- so a
// request's logits, log-probabilities and greedy tokens do not depend on what else is
// co-scheduled:
//
//   - BF16 projections (the cuBLASLt route, and the registered 27B kernels that change with T)
//     run `bf16_invariant_gemm`, whose per-element reduction order is fixed by k alone
//     (ops/linear/bf16/bf16_invariant_gemm.h);
//   - the fused GDN norm + gating projection takes its generic fixed-order kernel instead of the
//     registered routes, which split k by the token count (ops/wrapper/gdn_gating_proj.cpp);
//   - prompts are cut only at fixed multiples from position zero, never where a shared round's
//     window runs out (ProgramImplCore::invariant_prefill_piece);
//   - on Hopper, prompt attention stays on the split-KV tile kernels instead of FlashAttention-3,
//     so a query's attention is the same whether it is decoded, verified or prefilled
//     (ops/wrapper/gqa_attention.cpp).
//
// The server also turns off CUDA graphs and prefix reuse and refuses speculative decoding in
// this mode (serve/serve_options.cpp). The embedding server takes the same switch and keeps the
// encoder's projections on the W8 kernels instead of their BF16 cuBLASLt copies
// (encoder/text_embedding.cpp). Other weight formats keep their own routes (the GGML and
// K-quant ones are width-invariant by construction; FP8, NVFP4 and W8 are not verified), and
// images and multi-GPU pipelines are not covered.
//
// The switch is process-wide and is set once, before the first model loads. Reading it is a
// relaxed atomic load.

namespace sinfer::ops {

void set_batch_invariant(bool enabled) noexcept;
[[nodiscard]] bool batch_invariant() noexcept;

} // namespace sinfer::ops
