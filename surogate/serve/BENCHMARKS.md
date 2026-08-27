# Serving benchmarks — surogate serve vs vLLM vs llama.cpp

Date: 2026-08-26 · GPU: one idle RTX 5090 (32 GB, driver 590.44.01)
· Engines: **surogate serve** (this repo @ 3d7e643c, prefill + mixed-round
CUDA graphs on, vendored Marlin wide band via
`SUROGATE_SERVE_MARLIN_WIDE=1`, 64 lanes, fp4 prefill profile on W8
artifacts), **vLLM 0.27.1** (flashinfer 0.6.16.post3), **llama.cpp** in
two builds: **CUDA** (0.3.0-dev @ f1357e4, source build, sm_120 —
llama.cpp's strongest backend on this card, the primary rows) and the
brew **Vulkan** build (NV_coopmat2; kept where noted for reference).

Engine configuration for the 100-user rows: 64 lanes at 0.8B/4B and 48 at
27B (each model's measured optimum — more lanes cost throughput on all
three), mixed-round CUDA graphs on, GDN recurrent state stored bf16, and
**bf16 KV throughout** — the engine rows are all at full KV precision.
int8 KV was measured (27B 593 -> 604, 4B 3,032 -> 3,113) and is NOT used
here: it changes output, visibly so at temperature 0, and a throughput
number bought with quality is not comparable to one that is not. vLLM's
27B row does use `--kv-cache-dtype fp8` in its own required config, along
with `--max-num-seqs 32`, so its 688 tok/s comes from 32 concurrent
streams at reduced KV precision rather than 100 at full.

**All 100-user figures are 90-second steady-state runs.** Shorter windows
measure the ramp — lanes still filling, contexts still short — and read
up to 25% high (the same 4B config gives 2,633 over 40 s and 2,110 over
90 s). vLLM's figures were always 90 s, so only 90 s rows are comparable.

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
| **surogate serve** | from GGUF Q4_K_M | **48 ms** | **503** | **6,692** † | 0.77 s | 4,860/0 |
| llama-server (CUDA) | GGUF Q4_K_M | 168 ms | 391 | 772 | 10.9 s | 612/419 |
| vLLM | NVFP4 (4-bit) | 55 ms | 364 | 5,958 | **0.41 s** | 4,200/0 |

## Qwen3.5-4B

| engine | weights | TTFT @1.9k | decode tok/s (1 user) | 100-user agg tok/s | 100-user TTFT p50 | 100-user reqs ok/err |
|---|---|---:|---:|---:|---:|---:|
| **surogate serve** | from GGUF Q4_K_M | **57 ms** | **214** | 3,032 † | 1.7 s | 2,212/0 |
| llama-server (CUDA) | GGUF Q4_K_M | 445 ms | 190 | 331 | 27.4 s | 306/174 |
| vLLM | NVFP4 (4-bit) | 71 ms | 166 | **3,390** | **0.24 s** | 2,400/0 |

## Qwen3.8-27B

| engine | weights | TTFT @1.9k | decode tok/s (1 user) | 100-user agg tok/s | 100-user TTFT p50 | 100-user reqs ok/err |
|---|---|---:|---:|---:|---:|---:|
| surogate serve | NVFP4 (4-bit resident) | 352 ms | 45 | 593 | 10.8 s | 432/0 |
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
  cell**, including against llama.cpp's CUDA build on the same GGUF.
  Per-stream decode runs +29–30% over vLLM NVFP4 at matched widths.
- **100 users, 0.8B: ahead of vLLM** — 6,692 vs 5,958 tok/s (+12.3%),
  zero errors over 90 seconds. TTFT 0.77 s against vLLM's 0.41 s, so vLLM
  still admits faster while the engine sustains more throughput.
- **100 users, 4B: −11%** (3,032 vs 3,390). The gap is diffuse: GEMMs are
  45% of device across two families, decode attention 7.9% at about half
  its KV-read roofline, and 13% is admission idle that measurement showed
  is not worth reclaiming (delaying a free lane's refill by one round
  costs 18%, more than the idle is worth). The fp4 path's helper kernels
  — activation quantisation, output split, swiglu — are 9% of device and
  are the clearest remaining target.
- **100 users, 27B: −14%** (593 vs 688). Not a concurrency problem: vLLM's
  figure is 32 streams at ~21.5 tok/s each and our per-stream rate is
  comparable, so the difference is duty cycle — roughly 40% of the device
  goes to prefill. Lanes above 48 lose (52 → 561, 56 → 525) because
  per-lane GDN state squeezes the KV cache until sequences thrash.
- **llama-server's multi-user shape helps but does not change the
  order**: it trails the engine by 5–8x at 0.8B/4B with 30–40% of
  requests errored, and collapses on the dense 27B.

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
