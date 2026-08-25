# Serving benchmarks — surogate serve vs vLLM vs llama.cpp

Date: 2026-08-25 · GPU: one idle RTX 5090 (32 GB, driver 590.44.01)
· Engines: **surogate serve** (this repo @ 9b75e59; prefill CUDA graphs
on, deferred rewrite checkpoint on, fp4 prefill profile on W8 artifacts),
**vLLM 0.27.1** (flashinfer 0.6.16.post3), **llama.cpp** (brew build,
**Vulkan** backend, NV_coopmat2 — no CUDA llama.cpp build exists on this
host; small models ran b8500, 27B required the 0.3.0 upgrade).

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
256 --kv-capacity auto`; llama.cpp: `-ngl 99 --parallel 8 -c 32768
--jinja`; vLLM: `--max-model-len 4096` (defaults otherwise, 27B/35B add
`--gpu-memory-utilization 0.92`).

**Bit-width pairing.** Rows serve the closest available format of the
same weight class: llama.cpp serves GGUF **Q4_K_M** (~4.5 bpw); vLLM
serves **NVFP4** (4-bit, modelopt); surogate serves the artifact
repacked **from the same Q4_K_M GGUF** (W8 resident codes carrying the
GGUF's 4-bit information, fp4 compute profile), or native NVFP4 (4-bit
resident) where the target supports it (27B family — conversion pending
below).

## Qwen3.5-0.8B

| engine | weights | TTFT @1.9k | decode tok/s (1 user) | 100-user agg tok/s | 100-user TTFT p50 | 100-user reqs ok/err |
|---|---|---:|---:|---:|---:|---:|
| **surogate serve** | from GGUF Q4_K_M | **48 ms** | **473** | 1,255 | 9.3 s | 984/0 |
| llama.cpp (Vulkan) | GGUF Q4_K_M | 199 ms | 266 | 449 | 25.8 s | 405/242 |
| vLLM | NVFP4 (4-bit) | 55 ms | 364 | **5,958** | **0.41 s** | 4,200/0 |

## Qwen3.5-4B

| engine | weights | TTFT @1.9k | decode tok/s (1 user) | 100-user agg tok/s | 100-user TTFT p50 | 100-user reqs ok/err |
|---|---|---:|---:|---:|---:|---:|
| **surogate serve** | from GGUF Q4_K_M | **57 ms** | **214** | 354 | 28.9 s | 336/0 |
| llama.cpp (Vulkan) | GGUF Q4_K_M | 407 ms | 138 | 199 | 59.0 s | 235/106 |
| vLLM | NVFP4 (4-bit) | 71 ms | 166 | **3,390** | **0.24 s** | 2,400/0 |

## Qwen3.8-27B

| engine | weights | TTFT @1.9k | decode tok/s (1 user) | 100-user agg tok/s | 100-user TTFT p50 | 100-user reqs ok/err |
|---|---|---:|---:|---:|---:|---:|
| surogate serve | NVFP4 (4-bit resident) | *pending* | *pending* | *pending* | *pending* | — |
| llama.cpp (Vulkan) | GGUF Q4_K_M | 2,330 ms † | 42 † | *pending* | *pending* | — |
| vLLM | NVFP4 (4-bit) | *pending* | *pending* | *pending* | *pending* | — |

† partial: single-user workloads only, small sample (run interrupted for
GPU handoff). llama.cpp b8500 could not load this GGUF at all (missing
`blk.64.ssm_conv1d.weight` in its qwen3.8 mapping); the 0.3.0 upgrade
loads and serves it.

## Qwen3.6-35B-A3B (MoE)

| engine | weights | TTFT @1.9k | decode tok/s (1 user) | 100-user agg tok/s | 100-user TTFT p50 | 100-user reqs ok/err |
|---|---|---:|---:|---:|---:|---:|
| surogate serve | — | *no 4-bit MoE artifact yet* | — | — | — | — |
| llama.cpp (Vulkan) | GGUF Q4_K_M | *crashed* ‡ | — | — | — | — |
| vLLM | NVFP4 (4-bit) | 208 ms | 163 | 1,565 | 2.2 s | 1,172/0 |

‡ llama.cpp (b8500, Vulkan) loaded the 22 GB Q4_K_M but the server died
silently mid-prefill (~1.5k tokens in); retry with 0.3.0 pending.
The surogate W8 artifact (~35 GB) exceeds the card; an NVFP4 MoE
conversion path (the 27B family already has one) is the missing piece.

## Reading

- **Single user / small-stream serving: surogate wins every measured
  cell.** TTFT 48–57 ms on 1.9k-token prompts (llama.cpp: 199–407 ms;
  vLLM: 55–71 ms) and per-stream decode +29–30% over vLLM NVFP4 and
  +55–78% over llama.cpp on the same-width weights. Against the
  llama.cpp user profile (same GGUF in, one box, few users) the engine
  is strictly better at every size that fits.
- **100-user throughput: vLLM wins by 4.7× (0.8B) to 9.6× (4B).** The
  engine's aggregate is its 8 lanes times a per-lane decode that scales
  weakly (4B: 44/stream batched vs 214 solo ≈ 1.65× total; 0.8B: 2.65×);
  vLLM runs ~100-deep continuous batching at lower per-stream speed
  (38–74) but 11× the streams, and its admission keeps TTFT at 0.2–0.4 s
  where the engine's closed-loop queue reaches ~29 s. Closing this is
  the next engine campaign: raise the 8-lane ceiling, make batched
  decode scale (weight-read amortization across lanes), and admit
  continuously instead of queueing whole requests.
- **llama.cpp multi-user is not production-shaped** on this backend:
  40–50% of requests errored under 100-user load at 0.8B/4B (connection
  drops), and the 35B MoE crashed outright.
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

## Pending

27B: NVFP4 artifact conversion + engine legs; vLLM legs; llama.cpp
100-user leg. 35B: llama.cpp retry on 0.3.0. All staged; blocked only on
GPU availability.
