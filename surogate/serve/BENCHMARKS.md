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

One table per model, the same columns throughout; **every cell is measured** — the derived
figures the board used to carry are gone. **The first row of each model's table is the
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
| vLLM | 1 | 100 | 5,003 | 1,120 | 6,123 | 8.06 s | `sakamakismile/Qwen3.8-27B-MTP-NVFP4`, `--max-num-seqs 128`; 2026-08-30 07:28, uncapped GPU 3, idle host. surogate **+16 % decode, 47× TTFT** |
| vLLM | 1 | 100 | 4,156 | 1,039 | 5,195 | 8.26 s | the 2026-08-28 pass (pre-cap), which the row above reproduces +8 % |
| surogate | 8 | 100 | 4,403 | 986 | 5,389 | 1.33 s | 8-stage pipeline, C3 + asynchronous prompt flights, `--max-model-len 2048`; 2026-08-30 08:18, uncapped. **Below the one card above** — the pipeline buys capacity, not throughput per card |
| **surogate** | 8 | 1 | **306** | **68.6** | **374** | **0.16 s** | same, one user: each card holds 6 layers, so a token costs ~0.9 ms across the eight stages against ~5 ms on one card |
| surogate | 8 | 100 / 1 | — | 1,332 / 19.6 | — | — / 0.43 s | the 2026-08-29 pass (clock-capped, in-phase clients) |
| **surogate** | 1 | 100 | 471 | **1,884** | 2,355 | **14.2 s** | decode-heavy 128/512, 64 lanes (GPU5) |
| vLLM | 1 | 100 | 360 | 1,438 | 1,798 | 21.9 s | decode-heavy, same card |
| **surogate** | 1 | 100 | **11,208** | 85 | **11,293** | 15.9 s | prefill-heavy 2048/16, chunk 4,096, `--max-model-len 4096`, 128 lanes; 2026-08-30 09:24, uncapped GPU 3. **95 % of vLLM's prefill at a comparable TTFT** — the gap the board carried was the 08-27 configuration, not the engine |
| **vLLM** | 1 | 100 | **11,818** | 92 | 11,910 | **14.9 s** | prefill-heavy, 2026-08-27 pass |
| surogate | 1 | 100 | 7,339 | 57 | 7,396 | 25.8 s | the 2026-08-27 pass that defined "the 27B prefill gap": 62 % of vLLM. Superseded by the row above |
| surogate | 1 | 1 | 5,400 † | 45 | — | 352 ms | 2026-08-26, GPU2 |
| llama.cpp | 1 | 1 | 1,040 † | **49** | — | 1,829 ms |  |
| vLLM | 1 | 1 | 7,500 † | 45 | — | **254 ms** |  |

### Qwen3.6-35B-A3B

| engine | GPUs | users | prefill tok/s | decode tok/s | throughput tok/s | TTFT p50 | comments |
|---|---:|---:|---:|---:|---:|---:|---|
| surogate | 1 | 100 | **8,209** | **1,984** | **10,193** | **0.32 s** | Q4/Q5/Q6 MoE from GGUF, 128 lanes, chunk 4,096, prefill batch 4, `--max-model-len 2048`; 2026-08-30 07:14, uncapped GPU 4. Reproduces the 08-28 pass (1,942) |
| vLLM | 1 | 100 | 8,946 | **2,162** | 11,108 | 3.17 s | `RedHatAI/Qwen3.6-35B-A3B-NVFP4`, `--max-num-seqs 128`; 2026-08-30 07:14, uncapped GPU 5, same batch as the row above. surogate at **92 % of decode with 10× the TTFT** |
| surogate | 1 | 100 | 7,933 | 1,919 | 9,852 | 0.29 s | the same configuration under the caps (GPU 5 at 2,400 MHz) |
| surogate | 1 | 100 | 5,780 | 1,394 | 7,174 | 0.35 s | capped, and with `--no-thinking`: −27 % decode on top. On this MoE the generated text changes the expert spread per round, so the thinking mode is part of the configuration; the dense 27B shows no such gap |
| **surogate** | 8 | 100 | **8,785** | **2,123** | **10,908** | **0.58 s** | 8-stage pipeline, C3 + asynchronous prompt flights, `--max-model-len 2048`; 2026-08-30 08:25, uncapped. Above the one card, and the only configuration that beats vLLM's single-card decode |
| **surogate** | 8 | 1 | **965** | **233.4** | **1,198** | **0.09 s** | same, one user: 3 B active split eight ways puts a token at ~0.9 ms |
| surogate | 8 | 100 / 1 | — | 2,368 / 50.1 | — | — / 0.35 s | the 2026-08-29 pass (clock-capped, in-phase clients) |

### Qwen3.8-Flash-Next (111 GB MoE; on one card the experts live on the host)

| engine | GPUs | users | prefill tok/s | decode tok/s | throughput tok/s | TTFT p50 | comments |
|---|---:|---:|---:|---:|---:|---:|---|
| **surogate** | 1 | 1 | 128 | **28.7** | 156 | **1.36 s** | experts on the host: Q4G32AM bank (pinned, 91 GB), 3,000-slot expert cache, CPU split auto; 2026-08-30 07:55, uncapped GPU 6 (**x16 link**, gather 46 GB/s) |
| **surogate** | 1 | 16 | **298** | **66.9** | **365** | **2.62 s** | same, 81 % / 51 % measured shares; TTFT p90 14.7 s |
| surogate | 1 | 64 | 329 | 73.8 | 403 | 40.5 s | `--expert-slots 2000` so 64 lanes fit, `--pending-timeout-ms 600000` (the 30 s default expires a third of the queue here); host-bound, so 64 users only queue — TTFT p90 70.2 s |
| surogate | 1 | 1 / 16 / 64 | 132 / 311 / 313 | 32.0 / 75.3 / 75.7 | 164 / 386 / 389 | 1.06 s / 2.27 s / 37.1 s | the same rows measured earlier the same day on GPU 1 (node 0, x16, **clock-capped at 2,500 MHz**) with the 23:59 binary. The 12 % it reads above the current row is **not** the card: the same binary now gives 66.9 / 66.5 / 65.1 on GPUs 1, 4 and 0 (2.7 % spread, no NUMA-node effect) and 66.5-67.0 across four repeats on GPU 6. The clock cap is ruled out too: re-locking that card to 2,500 MHz gives 66.3 against 69.1 released, so capping *costs* ~4 % as physics expects. Card, node, cap and run-to-run noise are all eliminated; the row is **not reproducible on the current binary** (65.1-69.1 across four cards and six repeats) and is kept only as the historical measurement it was |
| surogate | 1 | 1 / 16 | 92 / 262 | 20.8 / 58.8 | 113 / 321 | 1.71 s / 3.35 s | **on GPU 7, whose PCIe link trains at x8**: gather 23 GB/s instead of 46, and a third of the throughput. Kept as the cost of the link fault (see Open items) |
| llama.cpp | 1 | 1 | 29 | 7.1 | 36 | 2.0 s | experts on CPU (`-ot exps=CPU`, 32 threads), 2026-08-28 |
| llama.cpp | 1 | 16 | 65 | 16.3 | 81 | 29 s |  |
| ik_llama.cpp | 1 | 1 | 87 | 21.8 | 109 | 1.8 s | AVX-512 iqk CPU-MoE kernels |
| ik_llama.cpp | 1 | 16 | 96 | 23.9 | 120 | 30 s |  |
| **surogate** | 8 | 1 | **209** | **46.8** | **256** | **0.85 s** | 8 stages, 3,072 slots per card (every expert resident, nothing crosses PCIe after warm-up), C3 + asynchronous prompt flights, `--max-model-len 2048`; 2026-08-30 08:06, uncapped |
| **surogate** | 8 | 16 | **1,231** | **276.1** | **1,507** | **2.37 s** | same; 4.1× the one-card 66.9 |
| **surogate** | 8 | 32 | **1,527** | **342.3** | **1,869** | **2.42 s** | same, stages materialise only their own layers (64 lanes fit beside the pool) |
| **surogate** | 8 | 64 | **1,750** | **392.3** | **2,142** | **2.80 s** | same; 5.3× the one card, and TTFT holds under 3 s where one card is at 40 s |
| surogate | 8 | 1 / 16 / 32 / 64 | — | 51.0 / 381.6 / 488.2 / 604.1 | — | 0.24 s / 615 ms / 971 ms / 1.14 s | the 2026-08-29 pass, measured under the clock caps *and* with a load generator that started all clients together; closed-loop clients in phase make a pipeline alternate between all-prefill and all-decode rounds, which is why its decode reads high against the staggered probe used above |
| llama.cpp | 8 | 1 | 157 | 39.3 | 196 | 0.95 s | `--split-mode layer`, all resident |
| llama.cpp | 8 | 16 | 156 | 39.1 | 195 | 86 s | 16 of 48 requests timed out |
| llama.cpp | 8 | 64 | 99 | 24.7 | 124 | 311 s |  |

## Reading the table

- **Weight format beats every scheduling lever on this hardware**: NVFP4 gave
  the 4B +54 % and the 27B +37 % over our own W8/mixed artifacts; the 35B is
  the one model still on a non-NVFP4 routed artifact and the one behind vLLM.
- **Lanes are 128** where the model fits them; with 100 users and 64 lanes a
  third of the load queued for a lane and that queue was the TTFT.
- **Flash-Next on one card is host-bound at one user and GPU-bound above it.**
  Measured at 16 users, the expert cache misses on only **2.1 %** of lookups and
  just **1.7 % of paths reach the CPU**, so the split and the PCIe gather cannot
  be what limits that row — the routed kernel over the resident pool is. At one
  user misses are frequent and the host path does set the pace, which is where
  the Q4 bank's halving of host bytes paid. The slot ring and the split carry
  16 users to 66.9 tok/s (median 66.7 over four runs, spread 0.6 %),
  and 64 users add queueing rather than throughput (73.8 tok/s at a 40 s TTFT).
  The card's PCIe link width moves that number as much as any engine change —
  x8 costs a third — and in-engine NUMA placement is worth +3 %. **Eight cards**
  make every expert
  resident and the pipeline delivers 4.1× at 16 users and 5.3× at 64, with
  TTFT under 3 s throughout.
- **A pipeline is capacity, not throughput per card** — and only for a dense
  model. The 27B on eight stages serves 986 tok/s against 1,302 on one card,
  because with ~12 lanes per group a stage's per-round fixed cost does not
  shrink with its layer count. The 35B, whose active weights are a fraction of
  its size, gains instead: 2,123 against 1,984. Both slash single-user latency
  (27B 0.16 s TTFT, 35B 0.09 s) because each card holds only six layers.
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
- ~~**27B**: prefill at half vLLM's rate~~ **closed 2026-08-30.** Re-measured on
  the current binary it serves **11,208 prompt tok/s against vLLM's 11,818**
  (95 %) at a comparable TTFT, and `ninfer_bench` puts the kernels at 12,707 with
  no scheduler — so there was never a kernel deficit, and round timing shows the
  executor 97 % occupied in mixed rounds with admission, boundary and append all
  at 0 ms. The 7,339 the board carried was a 2026-08-27 configuration. The
  layer-loop fusion and wider GDN scan queued against it are dropped, not
  deferred. See `design/27B_PREFILL.md`.
- ~~**Flash-Next, one card**: overlap the miss-gather with the hit-compute~~
  **closed 2026-08-30 without building it.** The Q4 bank is 3.07 MB per expert
  and all 48 layers route, so at a 63 % hit rate the misses are 3.7 experts per
  layer — but the CPU split already sends 76–81 % of them to the host, where the
  work is forked onto a side stream and overlapped. Only ~0.74 experts per layer
  actually cross PCIe: 0.11 GB per token, **2.4 ms of a ~35 ms token (7 %)**, and
  a perfect overlap would gain less than the ±10 % spread of the host-bandwidth
  probe. It would be worth 34 % only with the split off. Revisit if the split is
  ever reduced.
- **Hardware**: GPUs 2, 3, 5 and 7 train their PCIe links at x8 although both
  ends advertise x16 — a physical path issue, not bifurcation. It costs the
  host-offloaded model half its gather bandwidth (23 vs 46 GB/s) and about a
  third of its throughput; VRAM-resident models are unaffected. Until it is
  fixed, single-card Flash-Next rows belong on GPU 0, 1, 4 or 6.
- Record the card with every number; re-measure single-user cells on the
  same card as the 100-user rows.
