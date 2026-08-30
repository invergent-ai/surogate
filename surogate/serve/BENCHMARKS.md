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
comparable to its pair on the same card in the same batch. Measured on
2026-08-30 with the clock caps gone: one binary repeats within **0.6 %** on a
card and the eight cards agree within **2.7 %**, so a difference above ~3 % is
real. (The "cards differ by up to 50 %" this section used to warn about was the
clock profile of 2026-08-29, not the hardware.)

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
tok/s ÷ 128 (0.8B: 81,376 / 636; 27B: 11,208 / 85 — the same 128 for vLLM).
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
| **surogate** | 1 | 1 | **95,000 †** | **673** | — | **20 ms** | 2026-08-30 10:16, uncapped GPU 0, fp8 KV, ~1,900-token prompt |
| vLLM | 1 | 1 | 38,000 † | 498 | — | 50 ms | same batch, uncapped GPU 1. surogate **+35 % decode, 2.5× TTFT** |
| surogate | 1 | 1 | 39,600 † | 503 | — | 48 ms | the 2026-08-26 pass (GPU 2, bf16 KV) this replaces |
| llama.cpp | 1 | 1 | **19,000 †** | **411** | — | **100 ms** | 2026-08-30 12:01, uncapped GPU 5, `llama-server -ngl 999 -c 4096 -np 1` on the same Q4_K_M GGUF. Its own prompt-eval timing is 34,500 tok/s; the † above is the board's prompt÷TTFT and carries the queueing. **Use `study/llama.cpp-master/build/bin` — the `llama-server` on `PATH` is Homebrew's Vulkan build** (no CUDA, ignores `CUDA_VISIBLE_DEVICES`) and reads 252 tg / 6,880 pp512 on `llama-bench`, roughly 40 % of the CUDA build |
| llama.cpp | 1 | 1 | 11,300 † | 391 | — | 168 ms | the 2026-08-26 pass this replaces |

### Qwen3.5-4B

| engine | GPUs | users | prefill tok/s | decode tok/s | throughput tok/s | TTFT p50 | comments |
|---|---:|---:|---:|---:|---:|---:|---|
| **surogate** | 1 | 100 | **22,121** | **5,345** | **27,466** | **40 ms** | NVFP4 3.56 GiB, 128 lanes, chunk 2,048, `--max-model-len 2048`, 8 client shards; 2026-08-30 07:17, uncapped GPU 6; re-confirmed on the one-token routing fix binary (`fc2906fc`): 21,752 / 5,256 / 40 ms on GPU 4, 3,738 requests, 0 errors — the fix cannot fire above one token, and does not |
| surogate | 1 | 100 | 19,767 | 4,942 | 24,709 | 45 ms | the 2026-08-28 pass (pre-cap), which this reproduces +8 % |
| vLLM | 1 | 100 | 18,543 | 4,481 | 23,023 | 0.23 s | `surogate/Qwen3.5-4B-NVFP4` (ModelOpt), `--max-num-seqs 128`, `--max-model-len 2048`; 2026-08-30 13:24, uncapped GPU 5, 3,200 requests, 0 errors. surogate **+19 % decode, +19 % prefill, 5.8x TTFT** |
| vLLM | 1 | 100 | 16,984 | 4,246 | 21,230 | 239 ms | the 2026-08-28 pass this replaces; the row above reproduces it +5.5 % on decode (surogate +16 % on both, 5.3× TTFT) |
| **surogate** | 1 | 100 | **40,677** | 318 | 40,995 | 4.77 s | prefill-heavy 2048/16 (GPU5) |
| vLLM | 1 | 100 | 34,964 | 273 | 35,237 | 5.00 s | prefill-heavy, same card |
| **surogate** | 1 | 1 | **63,300 †** | **313** | — | **30 ms** | 2026-08-30 14:00, uncapped GPU 6, two passes (314.3, 312.8), fp8 KV, ~1,900-token prompt, on the decode-GEMV routing fix: the 4B's five linear shapes were unregistered NVFP4 geometries and ran a 128-row cuBLASLt tile on one row at every width; at one token they now take the decode GEMV (kernel-only tg128 213.5 → 350.8). **+26 % over vLLM's 249**, at half its TTFT |
| surogate | 1 | 1 | 63,300 † | 204 | — | 30 ms | the 2026-08-30 10:19 pass this replaces — same card class, same conditions, before the routing fix |
| surogate | 1 | 1 | 33,300 † | 214 | — | 57 ms | the 2026-08-26 pass (GPU 2) this replaces — decode within 5 %, prompt processing 1.9× |
| llama.cpp | 1 | 1 | **7,000 †** | **182** | — | **270 ms** | 2026-08-30 12:05, uncapped GPU 5, CUDA build, `unsloth/Qwen3.5-4B-GGUF` Q4_K_M (fetched for this row; our side is NVFP4). Its own prompt-eval timing is 13,000 tok/s |
| llama.cpp | 1 | 1 | 4,300 † | 190 | — | 445 ms | the 2026-08-26 pass this replaces — TTFT 1.6× better, decode flat |
| vLLM | 1 | 1 | 31,700 † | 249 | — | 60 ms | same checkpoint (`surogate/Qwen3.5-4B-NVFP4`, ModelOpt); 2026-08-30 13:27, uncapped GPU 5. Led decode by 22 % for four hours — that was our routing gap, not their kernels |
| vLLM | 1 | 1 | 26,800 † | 166 | — | 71 ms | the 2026-08-26 pass this replaces |

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
| **surogate** | 1 | 1 | **11,200 †** | **70.8** | — | **170 ms** | 2026-08-30 10:21, uncapped GPU 0, fp8 KV, ~1,900-token prompt. Beats every 08-26 figure below on both axes |
| **vLLM** | 1 | 1 | **13,600 †** | **71.7** | — | **140 ms** | `sakamakismile/Qwen3.8-27B-MTP-NVFP4`; 2026-08-30 10:37, uncapped GPU 1. The one shape where vLLM leads us at one user — decode within 1 %, TTFT 20 % better |
| surogate | 1 | 1 | 5,400 † | 45 | — | 352 ms | the 2026-08-26 pass (GPU 2) this replaces |
| vLLM | 1 | 1 | 7,500 † | 45 | — | 254 ms | the 2026-08-26 pass this replaces |
| llama.cpp | 1 | 1 | **1,610 †** | **44.8** | — | **1.18 s** | 2026-08-30 12:07, uncapped GPU 5, CUDA build, `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_M (fetched for this row; our side is all-NVFP4). Its own prompt-eval timing is 2,734 tok/s |
| llama.cpp | 1 | 1 | 1,040 † | 49 | — | 1,829 ms | the 2026-08-26 pass this replaces — TTFT 1.6× better, decode within 9 % |

### Qwen3.6-35B-A3B

| engine | GPUs | users | prefill tok/s | decode tok/s | throughput tok/s | TTFT p50 | comments |
|---|---:|---:|---:|---:|---:|---:|---|
| **surogate** | 1 | 100 | **10,733** | **2,594** | **13,327** | **0.09 s** | **routed NVFP4 through TensorRT-LLM's cutlass fused MoE** (`csrc/src/third_party/trtllm_moe`, the kernel behind vLLM's row below), our router/shared expert/combine kept, our kernels below 47 tokens; `--max-model-len 2048 --max-num-seqs 128 --max-num-batched-tokens 4096`; 2026-08-30 16:11, GPU 1. Coherence 100/100 at 100 users and chunkcount 191 ok / 0 wrong, both `--no-thinking` |
| surogate | 1 | 100 | 7,806 | 1,887 | 9,693 | 0.12 s | the shipped groupwise-int artifact, measured back-to-back with the row above as its pair — same card, session and flags. The routed-NVFP4 pair is **+37 % decode and +38 % prefill** |
| **surogate** | 1 | 16 | **4,818** | **1,165** | **5,982** | **0.07 s** | routed NVFP4, same session. At 16 users a decode round is ~16 columns, below the crossover, so the gain here is the wider prefill rounds |
| surogate | 1 | 16 | 4,397 | 1,063 | 5,460 | 0.10 s | groupwise-int at 16 users, the pair for the row above (+10 % decode) |
| **surogate** | 1 | 1 | **1,308** | **316.5** | **1,625** | **0.03 s** | routed NVFP4, one user: every decode round is a single token and stays on our own kernels, so this is the weight format alone |
| surogate | 1 | 1 | 1,275 | 308.4 | 1,583 | 0.04 s | groupwise-int at one user, the pair for the row above (+2.6 % decode) |
| surogate | 1 | 100 | 8,209 | 1,984 | 10,193 | 0.32 s | Q4/Q5/Q6 MoE from GGUF, 128 lanes, chunk 4,096, prefill batch 4, `--max-model-len 2048`; 2026-08-30 07:14, uncapped GPU 4. Reproduces the 08-28 pass (1,942) |
| vLLM | 1 | 100 | 8,946 | **2,162** | 11,108 | 3.17 s | `RedHatAI/Qwen3.6-35B-A3B-NVFP4`, `--max-num-seqs 128`; 2026-08-30 07:14, uncapped GPU 5, same batch as the row above. surogate at **92 % of decode with 10× the TTFT** |
| surogate | 1 | 100 | 7,933 | 1,919 | 9,852 | 0.29 s | the same configuration under the caps (GPU 5 at 2,400 MHz) |
| surogate | 1 | 100 | 5,780 | 1,394 | 7,174 | 0.35 s | capped, and with `--no-thinking`: −27 % decode on top. On this MoE the generated text changes the expert spread per round, so the thinking mode is part of the configuration; the dense 27B shows no such gap |
| **surogate** | 8 | 100 | **8,785** | **2,123** | **10,908** | **0.58 s** | 8-stage pipeline, C3 + asynchronous prompt flights, `--max-model-len 2048`; 2026-08-30 08:25, uncapped. Above the one card, and the only configuration that beats vLLM's single-card decode |
| **surogate** | 8 | 1 | **965** | **233.4** | **1,198** | **0.09 s** | same, one user: 3 B active split eight ways puts a token at ~0.9 ms |
| surogate | 8 | 100 / 1 | — | 2,368 / 50.1 | — | — / 0.35 s | the 2026-08-29 pass (clock-capped, in-phase clients) |
| surogate | 1 | 100 | 1,463 | 354 | 1,817 | 0.73 s | **superseded 2026-08-30 by the rows at the top of this table.** Routed NVFP4 on *our own* routed kernels, not adopted — the experts read verbatim from `RedHatAI/Qwen3.6-35B-A3B-NVFP4` (the checkpoint vLLM serves above). Same card, session and flags as the pair below; 2026-08-30 11:47, GPU 3 |
| surogate | 1 | 100 | 7,581 | 1,832 | 9,413 | 0.12 s | the shipped groupwise-int artifact, measured back-to-back with the row above as its pair (within 8 % of the 07:14 board row on a different card) |
| surogate | 1 | 16 | 1,345 | 325 | 1,670 | 0.64 s | routed NVFP4 at a concurrency where decode rounds are narrow enough to avoid the MMA path on either artifact — it still loses, because a 512-token prompt chunk is a wide round |
| surogate | 1 | 16 | 4,254 | 1,028 | 5,282 | 0.10 s | groupwise-int at 16 users, the pair for the row above |

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
| ik_llama.cpp | 1 | 1 | 1,068 | 40.4 | — | — | **reported, not measured here** (2026-08-30): same commit 7cff686d on an **RTX 3090 24 GB + Ryzen 9 9950X**, AD-4.27bpw Q4_K_M, 3-run average at temperature 0, single slot, 10,006-token prompt without cache reuse, 128 generated; KV Q8_0/Q8_0 (their setting — no board row of ours quantises the KV), 22.1 GB VRAM |
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
  the 4B +54 % and the 27B +37 % over our own W8/mixed artifacts, and now the
  35B +37 % — but only once its routed experts had a kernel worth the format.
  Our own routed-NVFP4 kernels reached 354 tok/s where the groupwise-int ones
  reached 1,832; TensorRT-LLM's fused MoE on the *same* weights reaches 2,594.
  The format was never the problem on that model, and neither was the schedule.
- **A borrowed kernel is not a whole answer either.** The runner permutes,
  groups and reduces for a batch, so at one token it costs more than it saves:
  serving every width through it drops single-user decode from 347 to 182 tok/s
  on the engine bench while lifting prefill 42 %. The crossover is a measured
  constant (`kSparseMoeTrtllmMinTokens`, 47, overridable), and with it the 35B
  wins at 100, 16 and 1 users at once.
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
  The reported ik_llama.cpp figure on a Ryzen 9 9950X desktop — 40.4 tok/s at one
  user, 1.85× the 21.8 the same commit measures on this host's EPYC 9124 — sizes
  the host-CPU lever for that one-user row: a 16-core Zen 5 at desktop clocks
  against a 16-core Zen 4 server part, same expert kernels, same weights. The
  one-card single-user number is a CPU benchmark before it is an engine one.
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
- **Three configuration levers worth as much as a kernel change.** Capping the
  context to the workload (`--max-model-len 2048`) is worth 16 % on the 0.8B,
  because auto sizes a 4.25 M-token KV cache the shape never touches. On the 35B
  MoE, `--no-thinking` costs 27 % of decode, since the generated text changes the
  expert spread per round. And on the host-offloaded model, in-engine NUMA
  placement is worth 3 % (`SUROGATE_SERVE_NUMA`, default `auto`: the shared expert
  bank interleaved, the per-device staging bound to that device's node — which
  also makes the launcher's `numactl --interleave=all` redundant, measured at
  ±0.5 %). All three belong in every row's comment.
- **And one that dwarfs them, on the 27B**: the prefill-heavy row moved from
  7,339 to 11,208 prompt tok/s between the 08-27 and 08-30 configurations without
  a kernel change. Configuration is not a footnote on this board; it is most of
  the variance between rows.
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
  and five repetitions of pp512+tg128 in about 3 s (it needs
  `--corpus csrc/src/testing/serve/bench/fixtures/bench_corpus.ids`).
- **llama.cpp rows** use `study/llama.cpp-master/build/bin/llama-server -ngl 999
  -c 4096 -np 1` — **not** the `llama-server` on `PATH`, which is Homebrew's
  Vulkan build (no CUDA, ignores `CUDA_VISIBLE_DEVICES`, about 40 % of the CUDA
  build's rate). The GGUFs are in the HF cache: `unsloth/Qwen3.5-4B-GGUF`
  Q4_K_M and `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_M, fetched 2026-08-30 for these
  rows; the 0.8B one is `models/Qwen3.5-0.8B-Q4_K_M.gguf`.
- **How to rebuild the 35B's routed-NVFP4 artifact**, if the format is ever
  wanted for a model that arrives on W8:
  `python -m surogate.serve.tools.convert.qwen3_6_35b_a3b.convert --model <BF16 dir>
  --routed-nvfp4 <compressed-tensors NVFP4 dir> --out <path>.ninfer` — 77 s, and
  the converter refuses anything that is not `compressed-tensors` /
  `nvfp4-pack-quantized` (a ModelOpt export inverts the global-scale convention).

## Open items

- **Hardware: four PCIe links train at x8.** GPUs 2, 3, 5 and 7, although both
  the card and its root port advertise x16 — so a physical path issue (MCIO
  cable, seating or a retimer channel), not bifurcation. It halves the gather
  bandwidth for the host-offloaded model (23 against 46 GB/s) and costs about a
  third of its throughput; VRAM-resident models are unaffected. Until it is
  fixed, single-card Flash-Next rows belong on GPU 0, 1, 4 or 6.
- **Nothing. Every row on this board is from 2026-08-30**, on the same binary and
  uncapped cards, except the hardware item below.

### Closed on 2026-08-30

One item was fixed and five died to measurement; the reasoning is in
`design/INFERENCE.md`, kept because each was about to become days of work.

- **4B single-user decode, 204 → 313 tok/s (+53 %)** — fixed, and it reverses
  the one single-user shape vLLM led. The engine's NVFP4 linear geometries are
  compile-time templates keyed to the 27B (all K = 5,120); every other model's
  shapes are "generic" and take cuBLASLt at every width, **including one
  token** — a 128x128x256 block-scaled MMA tile on a single row. At batch 1 the
  4B's linears were 65 % of a 4.7 ms token running at 37 % of memory bandwidth
  (2.0 GB of weights, 1.12 ms floor, 3.05 ms spent) while the 27B, whose shapes
  are registered, decodes at 76 % of peak with the decode GEMV. Fix:
  `Nvfp4GemvOnlyProblem`, a hidden-2560 geometry family that exists for exactly
  one route — the decode GEMV at tokens == 1 — in all four fused ops
  (`attn_input_proj`, `gdn_input_proj`, `linear_add`, `linear_swiglu`); every
  other width stays on cuBLASLt, so the 100-user row cannot move. Kernel-only
  213.5 → 350.8; served 204 → 313 at the same 30 ms TTFT; 27B unchanged (78.5);
  coherence verified through the new q/k/gate/v and qkv/z epilogues. The
  earlier board reading — "their per-token path is better when there is nothing
  to batch" — was wrong for the 4B; the 27B pair (70.8 vs 71.7) stays parity.

- **35B-A3B: NVFP4 routed experts** — built end to end and **not adopted**. The
  arm is complete (decode codec, small-T, wide rounds as small-T slices, the
  format's per-expert second level, a converter, a `routed-nvfp4` profile, a
  21.9 GB artifact that serves coherent text) and correct at every width against
  the fp64 oracle. It is also 6.4x slower at prompt processing and flat at
  decode, because of one line of arithmetic the plan never did: **NVFP4 is 4.5
  bits per weight and Q4G64_F16S is 4.25** — an E4M3 scale every 16 values
  against an FP16 every 64. The 4B's +54 % and the 27B's +37 % came from **W8
  (8.5 bpw) → NVFP4**, halving the bytes; the 35B's routed gate/up was already
  Q4G64, so it grew 5.9 %, and the whole artifact moved 19.59 → 19.19 GiB, 2.0 %.
  A 2 % byte cut cannot move a bandwidth-bound decode: `ninfer_bench` tg128 is
  343.1 against 347.5. The rest is the missing prefill MMA arm — a 4,096-token
  chunk runs as 89 small-T slices — which costs pp2048 2,008 against 15,219.
  `design/NVFP4_ROUTED_EXPERTS.md`.
- **35B-A3B: the row-parallel routed kernel** — the lever the entry above
  deferred to, also built and also flat. At decode width an expert holds ~3
  columns and the narrow plan's four warps leave three idle, so a
  row-block-per-warp kernel should have found ~19 %; it finds nothing at 64,
  100, 128, 256 or 512 columns, and neither do blocks/SM 3→12 nor a higher
  residency hint. Idle warps were a symptom: gate/up is 63 % of a 532 µs round
  and streams 209 MB in 345 µs — **606 GB/s, 34 % of peak, on 5 % of the card's
  math rate**. It is short of memory throughput, not warps. The kernel was
  reverted and the measurement kept in `design/serve-engine-backlog.md` B1,
  which also names what to do next: get GPU counter permissions (`ncu` is
  refused on this host) and measure sector efficiency before writing anything,
  because the suspect is the weight read *shape* — a k-group's consecutive rows
  sit 1,024 bytes apart, so a block issues 128 scattered 32-byte reads per
  k-step.

- **27B prefill gap** — did not exist on the current binary: 11,208 prompt tok/s
  against vLLM's 11,818 (95 %), with the kernels at 12,707 unserved and the
  executor 97 % occupied in mixed rounds. The layer-loop fusion and wider GDN
  chunked scan queued against it are dropped. `design/27B_PREFILL.md`.
- **Flash-Next gather/compute overlap** — 0.13 ms of a token once the measured
  2.1 % miss share and the CPU split are accounted for, against the third of a
  round it was queued on.
- **The 08-29/30 "regressions"** — the per-card clock profile, not the engine.
  Every model reproduces or beats its pre-cap row.
