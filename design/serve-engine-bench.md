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

| prompt tokens | engine prefill tok/s | vLLM prefill tok/s | engine decode tok/s | vLLM decode tok/s |
|--------------:|---------------------:|-------------------:|--------------------:|------------------:|
|            52 |            **4,368** |              2,250 |             **468** |               363 |
|           112 |            **8,329** |              5,762 |             **467** |               358 |
|           232 |           **13,548** |             10,426 |             **470** |               359 |
|           472 |               21,649 |         **22,309** |             **466** |               361 |
|           962 |               35,420 |         **36,654** |             **461** |               359 |
|          1912 |               41,128 |         **52,381** |             **450** |               357 |

## Reading

- **Decode: engine +28-30%** across all context lengths (460-470 vs
  357-363). Given the 2x weight-traffic advantage only buys +30%, the
  engine's 0.8B decode is NOT yet bandwidth-limited — it is
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

## Follow-ups

1. q08 prefill route tuning targeting the >=472-token region (measured
   sweep next time a GPU is idle); decode overhead hunt (launch counts,
   graph coverage).
2. Fair-width rerun: vLLM `--quantization fp8` vs engine W8.
3. MTP speculative decode on 0.8B (engine-only advantage; blocked on the
   speculative-replay op family walk, PATCHES #15) — 27B shows 3.45
   tok/round at 81.6% acceptance, which would multiply the decode column.
