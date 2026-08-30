# Serving benchmarks — surogate serve vs vLLM vs llama.cpp

Board of record, one table per model with the same columns. The narrative behind every row (superseded rows,
rejected levers, kernel profiles, the reasoning behind each lever, and the
per-model sections this table replaced on 2026-08-30) is in
`BENCHMARKS_HISTORY.md`; the dated engineering log is `design/INFERENCE.md`.

Host: 8× RTX 5090 (32 GB, driver 590.44.01), two NUMA nodes (GPUs 0–3 / 4–7),
2× EPYC 9124 (32 cores, AVX512-VNNI), 503 GB RAM. Engines: **surogate serve**
(this repo), **vLLM 0.27.1** (flashinfer 0.6.16.post3), **llama.cpp** CUDA
build (0.3.0-dev @ f1357e4; Flash-Next rows use upstream master with
`qwen4exp`, plus `ik_llama.cpp` 7cff686d for the CPU-MoE bar).

## Method

Closed-loop HTTP clients against each engine's OpenAI endpoint (streaming
`/v1/chat/completions`, salted prompts so nothing shares a prefix, the engine's
own token accounting). Numbers exclude model load; rows at 16+ users are
90-second steady state (shorter windows read up to 25 % high), one-user rows
60 s. Every engine was measured on the same card class; a number is only
comparable to its pair on the same card in the same batch (the same binary
repeats to ~1 % on a card, but cards and days differ by up to 50 %).

Columns:

- **GPUs** — cards the engine used (1, or an 8-card layer pipeline).
- **users** — concurrent closed-loop clients.
- **prefill tok/s** — prompt tokens ÷ the run's wall time (a throughput share,
  decode phases included — not a prompt-processing rate). At one user the
  prompt-processing rate is prompt tokens ÷ TTFT, marked †.
- **decode tok/s** — generated tokens ÷ wall time, summed over users (a
  one-user row is that stream's speed).
- **throughput tok/s** — prefill + decode: all tokens the engine moved per second.
- **TTFT p50** — median time to the first streamed content token.
- **≈** — the pass recorded decode but not prefill; the figure is derived from
  the workload shape (512/128 → prefill ≈ 4× decode) and so is the throughput.

**Clock caps, 2026-08-29 09:34 → 2026-08-30 07:07.** For that window a boot service locked
per-card maximum SM clocks (GPUs 0-5 at 1,700 / 2,500 / 1,500 / 2,500 / 1,800 / 2,400 MHz;
6 and 7 untouched), invisible to `clocks.max.sm`. Every rate measured inside it is
clock-limited and is **not** comparable to a row from outside it — a controlled pair read
1,519 vs 1,986 tok/s for the same engine on GPU 0 and GPU 5. The profile has been removed and
the cards verify equal within 3 %; rows dated 08-26/27/28 predate it, rows dated 2026-08-30
07:07 or later are uncapped, and any row still carrying a capped number says so in its
comment. Power stays limited to 400 W per card, which is what the clock settles against
(~2.0 GHz on the 27B).

Shapes: **512/128** unless the comment says otherwise (prefill-heavy is
2048/16, decode-heavy 128/512, single-user TTFT at a ~1.9k prompt). KV cache
fp8 (e4m3) on surogate; int8 KV is never used on this board (it changes
output). Weight pairing: llama.cpp serves GGUF Q4_K_M (~4.5 bpw), vLLM NVFP4,
surogate the native NVFP4 artifact (4B, 27B) or the artifact repacked from the
same GGUF (0.8B; 35B mixed Q4/Q5/Q6; Flash-Next W8 → Q4G32AM host bank).
Every artifact decodes bit-exact against its source before a number is
recorded, and every surogate row is probed for correctness **at** its
concurrency.

**Reading prefill and decode on the non-balanced shapes.** Both columns are
token counts divided by the same run wall time, so the shape fixes their
ratio: on prefill-heavy 2048/16 every completed request contributes 2,048
prompt tokens and 16 generated tokens, and decode tok/s is always prefill
tok/s ÷ 128 (0.8B: 81,376 / 636; 27B: 7,339 / 57 — the same 128 for vLLM).
That decode figure is completions per second × 16, not a decode speed: the
engine spends the run processing prompts and the 16-token tail is the small
share left over. On that shape the number to read is prefill (prompt
processing at 100 users) and TTFT (the queue for it); per-stream decode speed
is what the 512/128 and the decode-heavy 128/512 rows measure, where the
pinned ratio runs the other way (prefill = decode ÷ 4). Never compare a
column across shapes.

## Board of record (2026-08-30)

One table per model, the same columns throughout. **The first row of each model's table is the
current reference**: measured on 2026-08-30 after the clock caps were removed, one model at a
time on an idle host, with its vLLM pair in the same batch on a second card wherever the
checkpoint is still on this machine. Rows below it are kept for provenance — the pre-cap pass
they reproduce, the capped runs (labelled), and the other shapes — so the table shows both
what the engine does and how the number was arrived at.

### Qwen3.5-0.8B

| engine | GPUs | users | prefill tok/s | decode tok/s | throughput tok/s | TTFT p50 | comments |
|---|---:|---:|---:|---:|---:|---:|---|
| **surogate** | 1 | 100 | **46,225** | **11,166** | **57,391** | **20 ms** | GGUF Q4_K_M repack, 128 lanes, `--max-model-len 2048`, 8 client shards; 2026-08-30 07:08, uncapped GPU 0 |
| vLLM | 1 | 100 | 29,122 | 7,009 | 36,131 | 0.69 s | NVFP4 (`surogate/Qwen3.5-0.8B-NVFP4`), `--max-model-len 2048`; same batch, uncapped GPU 1. surogate **+59 % decode, 34× TTFT** |
| surogate | 1 | 100 | 32,226 | 7,785 | 40,011 | 30 ms | the same run under the clock caps (GPU 0 pinned to 1,700 MHz) — kept as the scale of the cap, not a board row |
| surogate | 1 | 100 | 27,222 | 6,577 | 33,799 | 40 ms | capped, and with the context left at auto (262,144 → a 4.25 M-token KV cache): −16 % more for a shape that never uses it |
| **surogate** | 1 | 100 | **81,376** | 636 | 82,012 | 2.39 s | prefill-heavy 2048/16 (2026-08-27, GPU4) |
| vLLM | 1 | 100 | 45,106 | 352 | 45,458 | 3.88 s | prefill-heavy, same card |
| **surogate** | 1 | 1 | 39,600 † | **503** | — | **48 ms** | 2026-08-26, GPU2, bf16 KV |
| llama.cpp | 1 | 1 | 11,300 † | 391 | — | 168 ms |  |
| vLLM | 1 | 1 | 34,500 † | 364 | — | 55 ms |  |

### Qwen3.5-4B

| engine | GPUs | users | prefill tok/s | decode tok/s | throughput tok/s | TTFT p50 | comments |
|---|---:|---:|---:|---:|---:|---:|---|
| **surogate** | 1 | 100 | **22,121** | **5,345** | **27,466** | **40 ms** | NVFP4 3.56 GiB, 128 lanes, chunk 2,048, `--max-model-len 2048`, 8 client shards; 2026-08-30 07:17, uncapped GPU 6 |
| surogate | 1 | 100 | 19,767 | 4,942 | 24,709 | 45 ms | the 2026-08-28 pass (pre-cap), which this reproduces +8 % |
| vLLM | 1 | 100 | 16,984 | 4,246 | 21,230 | 239 ms | NVFP4, 2026-08-28 pass; not re-measured — the NVFP4 4B checkpoint is no longer on this host, so the pair stays the 08-28 one (surogate +16 % on both, 5.3× TTFT) |
| **surogate** | 1 | 100 | **40,677** | 318 | 40,995 | 4.77 s | prefill-heavy 2048/16 (GPU5) |
| vLLM | 1 | 100 | 34,964 | 273 | 35,237 | 5.00 s | prefill-heavy, same card |
| **surogate** | 1 | 1 | 33,300 † | **214** | — | **57 ms** | 2026-08-26, GPU2 |
| llama.cpp | 1 | 1 | 4,300 † | 190 | — | 445 ms |  |
| vLLM | 1 | 1 | 26,800 † | 166 | — | 71 ms |  |

### Qwen3.8-27B

| engine | GPUs | users | prefill tok/s | decode tok/s | throughput tok/s | TTFT p50 | comments |
|---|---:|---:|---:|---:|---:|---:|---|
| **surogate** | 1 | 100 | **5,815** | **1,302** | **7,117** | **170 ms** | all-NVFP4, 128 lanes, chunk 4,096, `--max-model-len 2048`; 2026-08-30 07:11, uncapped GPU 2. Reproduces the 08-28 pass below within noise, so nothing regressed — the 1,068 measured under the caps was the cap |
| surogate | 1 | 100 | 5,319 | 1,330 | 6,649 | 170 ms | the 2026-08-28 pass (pre-cap): mean of two passes, cards rotated |
| vLLM | 1 | 100 | 4,156 | 1,039 | 5,195 | 8.26 s | NVFP4, 2026-08-28 pass; a same-batch uncapped pair is in flight |
| surogate | 8 | 100 | ≈ 5,300 | 1,332 | ≈ 6,600 | — | 8-stage pipeline, C3 + asynchronous prompt flights: capacity, not throughput per card |
| surogate | 8 | 1 | ≈ 78 | 19.6 | ≈ 98 | 0.43 s | 8 stages, closed pipeline |
| **surogate** | 1 | 100 | 471 | **1,884** | 2,355 | **14.2 s** | decode-heavy 128/512, 64 lanes (GPU5) |
| vLLM | 1 | 100 | 360 | 1,438 | 1,798 | 21.9 s | decode-heavy, same card |
| surogate | 1 | 100 | 7,339 | 57 | 7,396 | 25.8 s | prefill-heavy 2048/16, chunk 4,096 (GPU6); the open 27B gap |
| **vLLM** | 1 | 100 | **11,818** | 92 | 11,910 | **14.9 s** | prefill-heavy, same card |
| surogate | 1 | 1 | 5,400 † | 45 | — | 352 ms | 2026-08-26, GPU2 |
| llama.cpp | 1 | 1 | 1,040 † | **49** | — | 1,829 ms |  |
| vLLM | 1 | 1 | 7,500 † | 45 | — | **254 ms** |  |

### Qwen3.6-35B-A3B

| engine | GPUs | users | prefill tok/s | decode tok/s | throughput tok/s | TTFT p50 | comments |
|---|---:|---:|---:|---:|---:|---:|---|
| surogate | 1 | 100 | **8,209** | **1,984** | **10,193** | **0.32 s** | Q4/Q5/Q6 MoE from GGUF, 128 lanes, chunk 4,096, prefill batch 4, `--max-model-len 2048`; 2026-08-30 07:14, uncapped GPU 4. Reproduces the 08-28 pass (1,942) |
| vLLM | 1 | 100 | 8,957 | **2,166** | 11,123 | 3.13 s | `RedHatAI/Qwen3.6-35B-A3B-NVFP4`, `--max-num-seqs 128`; measured under the caps on GPU 3 (2,500 MHz) — an uncapped same-batch pair is in flight. surogate at 92 % of decode with 10× the TTFT |
| surogate | 1 | 100 | 7,933 | 1,919 | 9,852 | 0.29 s | the same configuration under the caps (GPU 5 at 2,400 MHz) |
| surogate | 1 | 100 | 5,780 | 1,394 | 7,174 | 0.35 s | capped, and with `--no-thinking`: −27 % decode on top. On this MoE the generated text changes the expert spread per round, so the thinking mode is part of the configuration; the dense 27B shows no such gap |
| surogate | 8 | 100 | ≈ 9,470 | **2,368** | ≈ 11,840 | — | 8-stage pipeline, C3 + asynchronous prompt flights |
| surogate | 8 | 1 | ≈ 200 | 50.1 | ≈ 250 | 0.35 s | 8 stages, closed pipeline |

### Qwen3.8-Flash-Next (111 GB MoE; on one card the experts live on the host)

| engine | GPUs | users | prefill tok/s | decode tok/s | throughput tok/s | TTFT p50 | comments |
|---|---:|---:|---:|---:|---:|---:|---|
| **surogate** | 1 | 1 | 132 | **32.0** | 164 | **1.06 s** | experts on the host: Q4G32AM bank (pinned, 91 GB), 3,000-slot expert cache, CPU split auto (76 % of decode misses / 46 % of prefill on 32 host threads); 2026-08-30, fixed host split |
| **surogate** | 1 | 16 | **311** | **75.3** | **386** | **2.27 s** | same defaults (81 % / 51 % measured shares); TTFT p90 12.4 s |
| surogate | 1 | 64 | 313 | 75.7 | 389 | 37.1 s | `--expert-slots 2000` so 64 lanes fit, `--pending-timeout-ms 600000` (the 30 s default expires a third of the queue at this concurrency); the round is host-bound, 64 users only queue — TTFT p90 68.7 s |
| llama.cpp | 1 | 1 | 29 | 7.1 | 36 | 2.0 s | experts on CPU (`-ot exps=CPU`, 32 threads), 2026-08-28 |
| llama.cpp | 1 | 16 | 65 | 16.3 | 81 | 29 s |  |
| ik_llama.cpp | 1 | 1 | 87 | 21.8 | 109 | 1.8 s | AVX-512 iqk CPU-MoE kernels |
| ik_llama.cpp | 1 | 16 | 96 | 23.9 | 120 | 30 s |  |
| **surogate** | 8 | 1 | ≈ 204 | **51.0** | ≈ 255 | ≈ 0.24 s | 8 stages, 3,072 slots per card (every expert resident, nothing crosses PCIe after warm-up), C3 + asynchronous prompt flights; re-validated on the fixed binary 2026-08-29 |
| **surogate** | 8 | 16 | ≈ 1,530 | **381.6** | ≈ 1,910 | **615 ms** | same; 5× the one-card 75 |
| **surogate** | 8 | 32 | ≈ 2,070 | **518.5** | ≈ 2,590 | 971 ms | same, stages materialise only their own layers (64 lanes fit beside the pool) |
| **surogate** | 8 | 64 | ≈ 2,330 | **583.6** | ≈ 2,920 | 1.14 s | same |
| llama.cpp | 8 | 1 | 157 | 39.3 | 196 | 0.95 s | `--split-mode layer`, all resident |
| llama.cpp | 8 | 16 | 156 | 39.1 | 195 | 86 s | 16 of 48 requests timed out |
| llama.cpp | 8 | 64 | 99 | 24.7 | 124 | 311 s |  |

## Reading the table

- **Weight format beats every scheduling lever on this hardware**: NVFP4 gave
  the 4B +54 % and the 27B +37 % over our own W8/mixed artifacts; the 35B is
  the one model still on a non-NVFP4 routed artifact and the one behind vLLM.
- **Lanes are 128** where the model fits them; with 100 users and 64 lanes a
  third of the load queued for a lane and that queue was the TTFT.
- **Flash-Next on one card** is host-bound: the Q4 bank halves the host bytes
  per expert (single user 22 → 32 tok/s), the CPU split at its measured share
  and the scan-resistant slot ring carry 16 users to 75 tok/s, and 64 users
  add queueing, not throughput. **Eight cards** make every expert resident
  and the pipeline delivers 5–8× the one-card figures at sub-second TTFT.
- **The 27B pipeline** is capacity, not throughput per card: with ~12 lanes per
  group a stage's per-round fixed cost does not shrink with its layer count.
- **Nothing regressed between 08-28 and 08-30.** Every apparent drop measured
  on 08-29/30 was the clock profile: uncapped, the 0.8B beats its pre-cap row
  (11,166 vs 10,095), and the 27B, 35B and 4B reproduce theirs (1,302 vs 1,330;
  1,984 vs 1,942; 5,345 vs 4,942). The detour that found it — binaries, flags,
  client shards, graph updates, all ruled out — is in INFERENCE.md.
- **Two configuration levers worth as much as a kernel change**: capping the
  context to the workload (`--max-model-len 2048`) is worth 16 % on the 0.8B,
  because auto sizes a 4.25 M-token KV cache the shape never touches; and on the
  35B MoE, `--no-thinking` costs 27 % of decode, since the generated text
  changes the expert spread per round. Both belong in every row's comment.
- Every surogate row above is from a binary that passes the correctness
  batteries at its concurrency (coherence and the strict-structure counting
  probe, `surogate/serve/tools/probe/`). The Flash-Next one-card rows are the
  first measured after the host-split fix of 2026-08-30 (`bfb87ec6`); the
  earlier 22.4 / 32.2 / 37.0 rows were measured through it.
- **How to reproduce a row.** Serving rows: `probe/board.py PORT MODEL USERS
  SECONDS 512 128 WARMUP SHARDS` against the launch line in the row's comment
  (shard the clients above ~5k tok/s; one Python process caps a fast engine).
  Engine-level questions — flags, binaries, cards, kernels — belong in
  `csrc/build-serve/serve_bench/ninfer_bench` instead, which does load, warm-up
  and five repetitions of pp512+tg128 in about 3 s.

## Open items

- **35B-A3B**: NVFP4 expert artifact — a kernel project, not a conversion: the
  sparse-MoE kernels and the expert slot cache read W8G32 routed experts only,
  so an NVFP4 routed arm (kernel + converter recipe) comes first; then a
  row-parallel decode-width routed kernel.
- **27B**: prefill at half vLLM's rate on the prefill-heavy shape (layer-loop
  fusion, a wider GDN chunked scan).
- Record the card with every number; re-measure single-user cells on the
  same card as the 100-user rows.
