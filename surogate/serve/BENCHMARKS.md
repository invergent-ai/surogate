# Serving benchmarks — surogate serve vs vLLM vs llama.cpp

Date: 2026-08-25 · GPU: one idle RTX 5090 (32 GB, driver 590.44.01)
· Engines: **surogate serve** (this repo @ 9b75e59; prefill CUDA graphs
on, deferred rewrite checkpoint on, fp4 prefill profile on W8 artifacts),
**vLLM 0.27.1** (flashinfer 0.6.16.post3), **llama.cpp** in two builds:
**CUDA** (0.3.0-dev @ f1357e4, source build, sm_120 — llama.cpp's
strongest backend on this card, the primary rows) and the brew
**Vulkan** build (NV_coopmat2; kept where noted for reference).

## Method

One shared HTTP load generator for all three engines (streaming
`/v1/chat/completions`, per-request salted prompts so no two requests
share a prefix, identical token accounting). Three workloads per
(model, engine); all numbers exclude model load:

| workload | shape | reported |
|---|---|---|
| 1 user, prefill-heavy | ~1.9k-token prompt, 128 out, 8 sequential | TTFT p50 (≈ prefill+queue), per-stream decode |
| 1 user, decode-heavy | ~60-token prompt, 512 out, 6 sequential | per-stream decode tok/s |
| 100 users | 100 closed-loop clients, ~512-token prompts, 128 out, 90 s | aggregate output tok/s, TTFT p50, completions |

Server configs — surogate: `--max-concurrency 8 --max-pending-requests
256 --kv-capacity auto`; llama-server single-user cells: `-ngl 99
--parallel 8 -c 32768 --jinja`; llama-server 100-user cells use its
multi-user shape: `--parallel 32 -c 65536 --threads-http 32` (16 slots at
35B for VRAM); vLLM: `--max-model-len 4096` (defaults otherwise, 27B/35B
add `--gpu-memory-utilization 0.92`).

**Bit-width pairing.** Rows serve the closest available format of the
same weight class: llama.cpp serves GGUF **Q4_K_M** (~4.5 bpw); vLLM
serves **NVFP4** (4-bit, modelopt); surogate serves the artifact
repacked **from the same Q4_K_M GGUF** (W8 resident codes carrying the
GGUF's 4-bit information, fp4 compute profile) at 0.8B/4B, and the
native **NVFP4 (4-bit resident)** artifact at 27B (built from the
unsloth NVFP4 export + base checkpoint by the vendored converter).

## Qwen3.5-0.8B

| engine | weights | TTFT @1.9k | decode tok/s (1 user) | 100-user agg tok/s | 100-user TTFT p50 | 100-user reqs ok/err |
|---|---|---:|---:|---:|---:|---:|
| **surogate serve** | from GGUF Q4_K_M | **48 ms** | **503** | 5,107 † | 1.7 s | 3,720/0 |
| llama-server (CUDA) | GGUF Q4_K_M | 168 ms | 391 | 772 | 10.9 s | 612/419 |
| vLLM | NVFP4 (4-bit) | 55 ms | 364 | **5,958** | **0.41 s** | 4,200/0 |

## Qwen3.5-4B

| engine | weights | TTFT @1.9k | decode tok/s (1 user) | 100-user agg tok/s | 100-user TTFT p50 | 100-user reqs ok/err |
|---|---|---:|---:|---:|---:|---:|
| **surogate serve** | from GGUF Q4_K_M | **57 ms** | **214** | **2,118** † | 4.0 s | 1,578/0 |
| llama-server (CUDA) | GGUF Q4_K_M | 445 ms | 190 | 331 | 27.4 s | 306/174 |
| vLLM | NVFP4 (4-bit) | 71 ms | 166 | **3,390** | **0.24 s** | 2,400/0 |

## Qwen3.8-27B

| engine | weights | TTFT @1.9k | decode tok/s (1 user) | 100-user agg tok/s | 100-user TTFT p50 | 100-user reqs ok/err |
|---|---|---:|---:|---:|---:|---:|
| surogate serve | NVFP4 (4-bit resident) | 352 ms | 45 | 385 | 30.1 s | 222/0 |
| llama-server (CUDA) | GGUF Q4_K_M | 1,829 ms | **49** | 82 | 102.4 s | 135/28 |
| vLLM | NVFP4 pack (4-bit) | **254 ms** | 45 | **688** | 12.5 s | 580/0 |

vLLM footnote: auto-detection cannot load public Qwen3.8 NVFP4 exports
(`'MergedColumnParallelLinear' object has no attribute 'data'`) and the
official FP8 checkpoint OOMs on 32 GB; serving requires the explicit
config: `--quantization compressed-tensors --language-model-only
--kv-cache-dtype fp8 --max-num-seqs 32` against a pack-quantized export
(measured: sakamakismile/Qwen3.8-27B-MTP-NVFP4). llama.cpp b8500 could
not load the GGUF either (missing `blk.64.ssm_conv1d.weight`); the 0.3.0
source build serves it, and its Q4_K GEMV edges single-stream decode
(49 vs 45) while TTFT and multi-user strongly favor the engine and vLLM.

## Qwen3.6-35B-A3B (MoE)

| engine | weights | TTFT @1.9k | decode tok/s (1 user) | 100-user agg tok/s | 100-user TTFT p50 | 100-user reqs ok/err |
|---|---|---:|---:|---:|---:|---:|
| surogate serve | — | *no 4-bit MoE artifact yet* | — | — | — | — |
| llama-server (CUDA) | GGUF Q4_K_M | 608 ms | 159 | 213 | 51.0 s | 238/111 |
| vLLM | NVFP4 (4-bit) | **208 ms** | **163** | **1,565** | **2.2 s** | 1,172/0 |

The Vulkan brew build crashed mid-prefill on this MoE; the CUDA source
build serves it. The surogate W8 artifact (~35 GB) exceeds the card; an
NVFP4 MoE conversion path (the 27B family already has one) is the
missing piece.

## Reading

- **Single user / small-stream serving: surogate wins every measured
  cell** — including against llama.cpp's CUDA build on the same GGUF
  (0.8B: TTFT 48 vs 168 ms, decode 473 vs 391, 100-user 1,255 vs 641).
  Per-stream decode runs +29–30% over vLLM NVFP4 at matched widths.
  Against the llama.cpp user profile (same GGUF in, one box, few users)
  the engine is strictly better at every size that fits.
- **100-user throughput: vLLM still wins where it serves, but the gap
  collapsed** († = engine cells at `--max-concurrency 32`, same build): 1.17× at
  0.8B (5,958 vs 5,107) and 1.60× at 4B (3,390 vs 2,118); the 27B is
  1.79× (688 vs 385).
  with the engine now 2.6–3.4×
  ahead of llama-server's tuned multi-user config. The fix: batch
  T=2..16 layer GEMMs ran prefill-class MMA tiles at ~6% utilization
  because the small targets' exact-T split-K tables were never
  instantiated; instantiating them took a batch-8 round from 4.84× to
  ~1.85× a solo round. The remaining gap is scheduling (the engine still
  queues whole requests behind 8 lanes; vLLM admits continuously at
  ~100 deep) — the campaign's next phases. The
  engine's aggregate is its 8 lanes times a per-lane decode that scales
  weakly (4B: 44/stream batched vs 214 solo ≈ 1.65× total; 0.8B: 2.65×);
  vLLM runs ~100-deep continuous batching at lower per-stream speed
  (38–74) but 11× the streams, and its admission keeps TTFT at 0.2–0.4 s
  where the engine's closed-loop queue reaches ~29 s. Closing this is
  the next engine campaign: raise the 8-lane ceiling, make batched
  decode scale (weight-read amortization across lanes), and admit
  continuously instead of queueing whole requests.
- **llama-server's multi-user shape helps but does not change the
  order.** At its tuned config (32 slots, continuous batching) it gains
  +18–20% aggregate over 8 slots at 0.8B/4B (772/331 tok/s) yet still
  trails the engine's 8 lanes (1,255/354) with 30–40% of requests
  errored (connection drops), and collapses on the dense 27B (82 tok/s,
  102 s TTFT). The Vulkan brew build was strictly slower everywhere and
  crashed on the 35B MoE; only CUDA rows are shown.
- The engine's serving stack adds no measurable overhead to the
  kernels: single-stream decode over HTTP (473 @0.8B, 214 @4B) matches
  the offline CLI board (`design/serve-engine-bench.md`, same GPU),
  which also carries the per-length prefill sweeps vs vLLM
  bf16/FP8/NVFP4.

## What this campaign fixed in the engine (found by benchmarking)

- 0.8B/2B could not serve at `--max-concurrency > 1` at all: the batch
  decode path's fused GDN projected-conv had no registered geometry for
  their shared shape (6144/2048/2048/2048). Registered.
- The GGUF→artifact converters' fixture roots were stale since the
  tools/ restructure (first GGUF ingest since then failed). Fixed.
- The safetensors reader folds multimodal checkpoint naming
  (`model.language_model.*` → `model.*`) but refused the unfolded names
  the 27B NVFP4 recipe requests. Resolution now accepts both.

## Open items this board surfaced

- **Multi-user campaign** (the one place vLLM wins): raise the 8-lane
  ceiling, make batched decode scale, admit continuously. Target: close
  the 4.7–9.6× 100-user gap at 0.8B/4B.
- **NVFP4 MoE conversion** for the 35B-A3B class (vLLM's 1,565 tok/s
  aggregate there is the bar; the engine has no 4-bit MoE artifact).
- **Single-source NVFP4 ingest**: the converter currently needs the
  bf16 base + the quantized export; the export alone carries every
  tensor. Dropping the base requirement removes a 52 GB download from
  the workflow.
- 27B single-user: llama.cpp's Q4_K GEMV edges the W4 decode 49 vs 45
  tok/s and vLLM's TTFT beats the engine 254 vs 352 ms — the only model
  where the engine does not lead single-user. Both belong to the same
  look during the multi-user campaign.
- Every engine needed expert configuration to serve the 27B at all
  (vLLM: explicit quantization/loader flags; llama.cpp: a source build;
  engine: the NVFP4 conversion). The engine's `surogate serve <input>`
  one-command story is the differentiator to protect.
