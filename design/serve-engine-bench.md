# Serve engine vs vLLM — Qwen3.5-0.8B reference benchmark

Date: 2026-08-24 · GPU: idle RTX 5090 (GPU 2) · batch 1, greedy, 128 decode
tokens, prompt lengths matched by token count. Engine numbers from
`surogate-engine-cli` internal timing (load excluded); vLLM numbers from the
offline `LLM` API after per-shape warmup (`language_model_only=True`,
flashinfer 0.6.16.post3, default compile/graphs).

**Not equal-width**: the engine serves the GGUF-repacked **W8** artifact
(1 byte/weight); vLLM serves the official HF checkpoint in **bf16**
(2 bytes/weight). Batch-1 decode is weight-bandwidth-bound, so the engine
carries a ~2x memory-traffic advantage into the decode columns. A
fair-width follow-up is vLLM `--quantization fp8`.

FP8 column: `surogate/Qwen3.5-0.8B-FP8` (fp8 weights, dynamic per-token
activation quantization) on the same vLLM build — the equal-width
(1 byte/weight) reference.

| prompt | engine W8 prefill | vLLM bf16 prefill | vLLM FP8 prefill | engine W8 decode | vLLM bf16 decode | vLLM FP8 decode |
|-------:|------------------:|------------------:|-----------------:|-----------------:|-----------------:|----------------:|
|     52 |         **4,368** |             2,250 |            2,425 |          **468** |              363 |             285 |
|    112 |         **8,329** |             5,762 |            4,952 |          **467** |              358 |             286 |
|    232 |        **13,548** |            10,426 |           10,890 |          **470** |              359 |             289 |
|    472 |            21,649 |        **22,309** |           20,164 |          **466** |              361 |             288 |
|    962 |            35,420 |        **36,654** |           32,819 |          **461** |              359 |             281 |
|   1912 |            41,128 |            52,381 |       **60,667** |          **450** |              357 |             288 |

## Reading

- **Decode: engine +28-30% vs bf16, +60-65% vs equal-width FP8** (460-470
  vs 357-363 vs 281-289). vLLM's FP8 decode is SLOWER than its own bf16 at
  this scale: dynamic per-token activation quantization adds per-layer
  overhead that tiny GEMMs cannot amortize — a direct validation of the
  engine's W8 + A16 (weights-only) design for small models on consumer
  cards. Given the engine's 2x weight-traffic advantage over bf16 only
  buys +30%, the engine's 0.8B decode is NOT yet bandwidth-limited — it is
  overhead/launch-bound, consistent with the untuned correctness-first q08
  route tables (PATCHES #13). Route tuning has real headroom here.
- **Prefill: engine wins short, vLLM wins long.** Engine leads ~2x at 52
  tokens and stays ahead through ~232; crossover near ~470 tokens; at 1912
  vLLM is +27%. vLLM's prefill latency is nearly flat (19-26ms) out to ~1k
  tokens — big-tile GEMM efficiency — while the engine's untuned q08
  prefill routes (mostly MmaR64C128) fall behind at scale.
- vLLM stack note: the stale `surogate.quant.vllm` entry point in the
  editable install broke vLLM startup (module deleted long ago; dist-info
  still pointed at it). Fixed in-place to match pyproject
  (`surogate.grpo.inference.patches:transformers_v5_compat`); a
  `pip install -e . --no-deps` refresh makes it permanent.

## Post-tuning update (same session)

`bench/ops/q08_route_sweep_bench` (13 tile schedules x 5 GEMM shapes x 4
token counts, direct kernel instantiation) found the loss: with only 1024
output rows, linear_add's r64c128 tile underfills the GPU. Measured
winners applied (linear_add {129-1024: r32c128, 1025+: r48c128}; swiglu
449-512: r128c80 for the q08 shape):

Second pass: default prefill chunk 1024 -> 2048 (measured +16% on the
0.8B at ~1.9k-token prompts — K=1024 GEMM tiles amortize the main loop
better at larger T — and neutral on the 27B: 1832 vs 1834 tok/s).

| prompt | engine before | engine after | vLLM bf16 | vLLM FP8 |
|-------:|--------------:|-------------:|----------:|---------:|
|     52 |         4,368 |        4,043 |     2,250 |    2,425 |
|    232 |        13,548 |   **14,897** |    10,426 |   10,890 |
|    472 |        21,649 |   **25,782** |    22,309 |   20,164 |
|    962 |        35,420 |   **37,327** |    36,654 |   32,819 |
|   1912 |        41,128 |   **52,005** |    52,381 | **60,667** |

Scoreboard after both passes: the engine beats or matches vLLM bf16 at
EVERY point (prefill and decode), and beats vLLM FP8 everywhere except
1912-class prefill (52.0k vs 60.7k, -14%). The remaining 1912-class gap is structural: the W8
kernels dequantize to BF16 MMA (~90 TF/s achieved, candidates within 10%
of each other) while vLLM FP8 runs FP8 tensor cores at ~2x the mma rate —
closing it means a W8->FP8-MMA kernel path, a future project.

## Follow-ups

1. Long-prefill (>=1200 tok) W8->FP8-MMA kernel path (structural ~40%);
   decode overhead hunt (launch counts, graph coverage).
2. Fair-width rerun: DONE (FP8 column above).
3. MTP speculative decode on 0.8B (engine-only advantage; blocked on the
   speculative-replay op family walk, PATCHES #15) — 27B shows 3.45
   tok/round at 81.6% acceptance, which would multiply the decode column.
