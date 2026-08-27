# Serving benchmarks — surogate serve vs vLLM vs llama.cpp

Board of record. Two measurement campaigns are recorded here and they are
**not interchangeable**: the 100-user rows are from 2026-08-27, each engine
measured on the same card back to back; the single-user cells and the
llama.cpp rows are from 2026-08-26 on GPU2. The card is recorded with every
number because the same binary measures **6,498 tok/s on GPU2 and 9,972 on
GPU3** on the 0.8B (both x8 PCIe, same NUMA node) — a larger spread than any
engine-to-engine gap on this board.

Host: 8× RTX 5090 (32 GB, driver 590.44.01), two NUMA nodes (GPUs 0–3 / 4–7).
Engines: **surogate serve** (this repo @ 6102534), **vLLM 0.27.1**
(flashinfer 0.6.16.post3), **llama.cpp** CUDA source build (0.3.0-dev @
f1357e4, sm_120; the brew Vulkan build is kept only where noted).

Engine configuration: `--max-num-seqs 64` at 0.8B/4B and `48` at 27B (each
model's measured optimum), `--max-pending-requests 256`, prefill and
mixed-round CUDA graphs on, GDN recurrent state stored bf16, **fp8 (e4m3) KV
cache — the default since PATCHES.md #69**; `--kv-cache-dtype bf16` restores
the full-precision cache. int8 KV is not used anywhere on this board: it
changes output, visibly at temperature 0, and a throughput number bought
with quality is not comparable to one that is not.

**All 100-user figures are 90-second steady-state runs.** Shorter windows
measure the ramp and read up to 25% high (the same 4B config gave 2,633 over
40 s and 2,110 over 90 s). Prefill is reported alongside decode because the
512-in/128-out shape puts four prompt tokens through the engine for every
token it generates; a board that shows decode only shows a fifth of the work.

## Method

One shared HTTP load generator for every engine (streaming
`/v1/chat/completions`, per-request salted prompts so no two requests share a
prefix, identical token accounting; it reports `output_tok_per_s`,
`prompt_tok_per_s` and `total_tok_per_s`). All numbers exclude model load.

| workload | shape | reported |
|---|---|---|
| 100 users | 100 closed-loop clients, ~512-token prompts, 128 out, 90 s | decode tok/s, prefill tok/s, TTFT p50, completions |
| 1 user, prefill-heavy | ~1.9k-token prompt, 128 out, 8 sequential | TTFT p50, per-stream decode |
| 1 user, decode-heavy | ~60-token prompt, 512 out, 6 sequential | per-stream decode tok/s |
| decode-heavy soak | 100 users, 128-token prompts, 512 out, 300–600 s | corruption count (correctness coverage, see below) |

Server configs — vLLM: `--max-model-len 4096` (defaults otherwise; the 27B
needs `--quantization compressed-tensors --language-model-only
--kv-cache-dtype fp8 --max-num-seqs 32 --gpu-memory-utilization 0.92` against
the pack-quantized `sakamakismile/Qwen3.8-27B-MTP-NVFP4`: auto-detection
cannot load the public Qwen3.8 NVFP4 exports and the official FP8 checkpoint
OOMs on 32 GB); llama-server single-user: `-ngl 99 --parallel 8 -c 32768
--jinja`; llama-server 100-user: `--parallel 32 -c 65536 --threads-http 32`.

**Bit-width pairing.** llama.cpp serves GGUF **Q4_K_M** (~4.5 bpw); vLLM serves
**NVFP4** (4-bit); surogate serves the artifact repacked **from the same
Q4_K_M GGUF** (W8 resident codes carrying the GGUF's 4-bit information, fp4
compute profile) at 0.8B/4B, and the native **NVFP4 (4-bit resident)**
artifact at 27B.

## 100 users — same card, same day, back to back (2026-08-27)

vLLM served first, then the engine, on the card shown, quiet host, 100
users, 512/128, 90 s. Engine at fp8 KV; vLLM at its default cache (fp8 at
27B by its required config).

| model | card | engine | decode tok/s | prefill tok/s | TTFT p50 | reqs ok/err |
|---|---|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | GPU4 | **surogate serve** | **7,666** | **33,666** † | 0.60 s | 5,480/0 |
| | GPU4 | vLLM | 5,677 | 22,714 | 0.82 s | 4,001/0 |
| Qwen3.5-4B | GPU5 | **surogate serve** | **4,359** | **19,178** † | 1.05 s | 3,136/0 |
| | GPU5 | vLLM | 3,926 | 15,702 | **0.30 s** | 2,800/0 |
| Qwen3.8-27B | GPU6 | surogate serve | 827 | 3,677 † | 7.96 s | 650/0 |
| | GPU6 | **vLLM** | **1,041** | **4,162** | 8.3 s | 814/0 |

† engine prefill tok/s and TTFT p50 are taken from the server's own interval
and per-request logs of the same runs (the pairing script summarised only
decode); vLLM's are the load generator's client-side figures.

Engine: **+35%** at 0.8B, **+11%** at 4B, **−21%** at 27B on decode; ahead on
prefill at 0.8B and 4B. vLLM's own numbers moved against the 08-26 board too (4B
3,390 → 3,926, 27B 688 → 1,041), so neither side of that board was measured
under today's conditions; the 08-26 100-user rows are kept below only as
history.

## Single user (2026-08-26, GPU2)

Not re-measured on 08-27. Same shapes, same loadgen; engine at bf16 KV.

| model | engine | weights | TTFT @1.9k | decode tok/s (1 user) |
|---|---|---|---:|---:|
| Qwen3.5-0.8B | **surogate serve** | from GGUF Q4_K_M | **48 ms** | **503** |
| | llama-server (CUDA) | GGUF Q4_K_M | 168 ms | 391 |
| | vLLM | NVFP4 | 55 ms | 364 |
| Qwen3.5-4B | **surogate serve** | from GGUF Q4_K_M | **57 ms** | **214** |
| | llama-server (CUDA) | GGUF Q4_K_M | 445 ms | 190 |
| | vLLM | NVFP4 | 71 ms | 166 |
| Qwen3.8-27B | surogate serve | NVFP4 resident | 352 ms | 45 |
| | llama-server (CUDA) | GGUF Q4_K_M | 1,829 ms | **49** |
| | vLLM | NVFP4 pack | **254 ms** | 45 |

The engine leads every single-user cell except the 27B, where llama.cpp's
Q4_K GEMV edges decode (49 vs 45) and vLLM's TTFT leads (254 vs 352 ms).

## Qwen3.6-35B-A3B (MoE) — 2026-08-26, GPU2

| engine | weights | TTFT @1.9k | decode tok/s (1 user) | 100-user tok/s | 100-user TTFT p50 | reqs ok/err |
|---|---|---:|---:|---:|---:|---:|
| surogate serve | — | *no 4-bit MoE artifact yet* | — | — | — | — |
| llama-server (CUDA) | GGUF Q4_K_M | 608 ms | 159 | 213 | 51.0 s | 238/111 |
| vLLM | NVFP4 | **208 ms** | **163** | **1,565** | **2.2 s** | 1,172/0 |

The surogate W8 artifact (~35 GB) exceeds the card; an NVFP4 MoE conversion
path is the missing piece.

## Prefill, by workload shape (4B, 2026-08-27, GPU2, fp8 KV)

| shape | prompt/out | prefill tok/s | decode tok/s | total tok/s | TTFT p50 |
|---|---|---:|---:|---:|---:|
| balanced | 512 / 128 | 11,674 | 2,918 | **14,593** | 1.6 s |
| prefill-heavy | 2048 / 16 | **26,891** | 210 | **27,101** | 7.2 s |
| decode-heavy | 128 / 512 | 413 | 1,296 | 1,709 | 3.8 s |

Prefill throughput more than doubles from 512- to 2048-token prompts, so the
balanced row is nowhere near the prefill ceiling; any comparison reporting
only aggregate output tok/s is measuring the workload's prompt:generation
ratio as much as the engine. On the 0.8B the engine sustains ~43,000 prefill
tok/s alongside 9,950 decode on GPU3.

## FP8 KV cache (default since PATCHES.md #69)

| model | card | cache | 100-user decode tok/s | KV size |
|---|---|---|---:|---|
| 0.8B | GPU2 | bf16 | 6,837 | 3.71 GiB |
| 0.8B | GPU2 | fp8 | 6,529 | 2.96 GiB |
| 27B @48 | GPU2 | bf16 | 594 | 46,016 tokens |
| 27B @48 | GPU2 | fp8 | 582 | **92,096 tokens** (exactly 2×) |

A memory feature, not a throughput one: a few percent of decode buys double
the cache. The 27B is not KV-capacity-bound at 48 lanes (its bottleneck is
prefill duty cycle), so the capacity does not show up as throughput there.
Only full-attention layers hold KV planes, so a quantized cache cannot reach
a linear-attention layer; `--kv-cache-dtype-skip-layers L,...` holds named
full-attention layers at bf16.

## Correctness coverage: the decode-heavy soak

The balanced 512/128 rows cannot catch a class of bug that only appears when
lanes start below frontier 512 and cross it under sustained load. Two such
bugs were found and fixed on 08-27 (PATCHES.md #71: the mixed-graph key did
not carry the frontier band; #72: the direct GDN kernel read bf16 state as
fp32 on the eager prefill path). After both, on the fixed binary:

| shape | model | requests | corrupted streams |
|---|---|---:|---:|
| decode-heavy 128/512, 300 s | 27B | 1,060 | 0 (was 25–38 per 300 s) |
| decode-heavy 128/512, 300 s | 4B | 4,097 | 0 |
| balanced, `--enforce-eager`, 300 s | 0.8B | 14,415 | 1 (was 22 in 174) |
| balanced, 600 s × 2 cards | 0.8B | 81,319 | 3 |

The 0.8B residual — about 4 per 100,000 requests, both modes, always a mixed
round — is open and predates this work (PATCHES.md, "Open"). With the fatal
restored it kills a 0.8B worker roughly once per 25,000 requests at 100 users.

## Reading

- **Per card, the engine is ahead of vLLM at 0.8B (+35%) and 4B (+11%) and
  behind at 27B (−21%).** The 27B difference is duty cycle: roughly 40% of
  the device goes to prefill, and vLLM's row is 32 streams at fp8 KV.
- **The card is a bigger variable than the engine gap.** Two cards on the same
  NUMA node differ by 53% on the 0.8B for the same binary; the engine is
  host-round-trip-bound at the small sizes, so host contention from jobs on
  the other seven GPUs depressed every 08-26 number by a further ~30%.
- **Single user, the engine leads every cell but the 27B**, where llama.cpp's
  Q4_K GEMV edges decode and vLLM's TTFT leads.
- **llama-server's multi-user shape does not change the order**: 30–40% of
  requests errored and 5–8× behind at 0.8B/4B (08-26, GPU2).

## History: the 2026-08-26 board (GPU2, bf16 KV, busy host)

| model | surogate serve | vLLM | llama-server (CUDA) |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 6,692 | 5,958 | 772 (612/419 ok/err) |
| Qwen3.5-4B | 3,032 | 3,390 | 331 (306/174) |
| Qwen3.8-27B | 593 | 688 | 82 (135/28) |

Superseded by the same-card pairs above; kept because the fixes listed in
PATCHES.md #44–#67 were measured against it.

## What benchmarking fixed in the engine

- 0.8B/2B could not serve more than one lane: the batch decode path's fused
  GDN projected-conv had no registered geometry for their shape. Registered.
- The GGUF→artifact converters' fixture roots were stale since the tools/
  restructure. Fixed.
- The safetensors reader refused the unfolded `model.language_model.*` names
  the 27B NVFP4 recipe requests. Both accepted now.
- `--kv-capacity auto` over-committed the card by the size of the derived
  weight planes and died in graph capture (PATCHES.md #68).
- Two corruption root causes under decode-heavy load (#71, #72), found only
  because a decode-heavy soak was added to the coverage.

## Open items

- **27B, 100 users: −21%** against vLLM on the same card. Prefill duty cycle
  is the lever; lanes above 48 lose because per-lane GDN state squeezes the
  cache until sequences thrash.
- **0.8B rare corruption**, ~4 per 100,000 requests (PATCHES.md, "Open").
- **NVFP4 MoE conversion** for the 35B-A3B class (vLLM's 1,565 tok/s is the bar).
- **Single-source NVFP4 ingest**: the converter still needs the bf16 base
  alongside the quantized export.
- **Record the card with every number**, and re-measure single-user cells
  on the same card as the 100-user rows.
