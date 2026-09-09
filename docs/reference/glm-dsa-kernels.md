# DeepSeek sparse-attention kernel review

Reviewed on 2026-09-09 at
[`Dogacel/DeepSeek-Sparse-Attention-Kernels@9734cf8`](https://github.com/Dogacel/DeepSeek-Sparse-Attention-Kernels/tree/9734cf83fe82a61466a24e0e95e7f470dff7eb47).

The repository contains useful decode optimizations, especially split sparse
attention, radix top-k, and bypassing score computation when every key is selected.
Its kernels are not used or vendored in this implementation.

## Compatibility

| Area | Reviewed implementation | GLM training and native-colocate |
|---|---|---|
| Indexer | FP8 Q/K, 64 heads, 128 channels | BF16 projections, learned channel-wise pooling; released GLM has 32 indexer heads |
| Selection | 2048 individual keys and 64-token physical pages | Select complete pools, expand their tokens, optionally include the current incomplete pool |
| Attention | Shared absorbed MLA latent KV, 512 latent and 64 positional channels, 16 query heads | NoPE; current cache contains expanded per-head K/V |
| Gradients | Decode forward only | Sparse attention backward is required for full-weight and LoRA training |
| Target | B200 contest workloads | Validated here on RTX 5090 |

Launching the upstream sparse kernel unchanged on an RTX 5090 fails with a
shared-memory resource error: **169,984 bytes required,
101,376 bytes available**. The trial used three queries, 16 heads, 512 latent
channels, 64 positional channels and 2048 selected keys. It produced no timing
or numerical result. The repository's published speedup is for its B200 contest
traces and reference implementation; it does not measure GLM training or RTX decode.

## Reuse constraints

There is no repository-wide license at the reviewed revision and no license
notice in the Triton source files. The bundled FlashInfer `topk.cuh` has an
Apache-2.0 notice. Obtain licensing for the other files before copying them, or
implement the algorithms independently using appropriately licensed components.

Code-review concerns that need resolution before adapting these kernels:

- Sparse attention synchronizes CTAs through a global atomic counter and spin
  barrier. Its target advances on the host. A captured graph would replay a fixed
  target while the device counter continues advancing, potentially consuming
  stale partial results. This concern was not tested because the kernel did not
  launch on the RTX 5090. Scheduling assumptions also need review for larger grids.
- Scratch tensors are cached globally with incomplete device/shape keys. Concurrent
  streams and changing geometry need explicit ownership and synchronization.
- The top-k binding fixes refinement scratch at 6K keys based on the contest's
  maximum padded context of 5824. Long GLM contexts require a different bound.
- Sparse attention assumes valid selected indices form a prefix. GLM selection
  can contain masked holes, which must be preserved or compacted explicitly.

For Surogate, keep the registered GLM-specific kernels in
[`surogate/kernels/triton/glm_dsa.py`](../../surogate/kernels/triton/glm_dsa.py).
Future optimizations can replace full sorting with radix selection, skip scoring
when all visible pools fit, and split attention across selected keys. Use separate
partial/reduction launches or another graph-safe synchronization scheme, and
validate forward, backward, packing, cache resets and graph replay independently.
