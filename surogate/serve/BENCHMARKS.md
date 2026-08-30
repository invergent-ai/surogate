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
`/v1/chat/completions`, salted prompts so nothing shares a prefix, the engine's own
token accounting), model load excluded. Clients are **staggered**, not started
together: in phase they make a pipeline alternate between all-prefill and
all-decode rounds, which reads high — an 8-stage Flash-Next pass measured that way
reported 381 tok/s at 16 users against 276 staggered.

Every engine was measured on the same card class, and **a number is only comparable
to its pair on the same card in the same batch**. One binary repeats within **0.6 %**
on a card and the eight cards agree within **2.7 %**, so a difference above ~3 % is
real and anything under it is not.

Measurement windows are 60-90 s of steady state after a warm-up, one-user rows 45-60 s.
The window matters: a shorter one reads high, by up to 25 % against a 90 s window on
an early pass. Today's rows were taken at 62-76 s, so they are inside that band but
not all at the same point in it — see Open items.

Shapes: **512/128** unless the comment says otherwise (prefill-heavy 2048/16,
decode-heavy 128/512, single-user TTFT at a ~1.9k prompt). KV cache fp8 (e4m3) on
surogate; int8 KV is never used on this board, because it changes the output. Weight
pairing: llama.cpp serves GGUF Q4_K_M (~4.5 bpw), vLLM NVFP4, surogate the native
NVFP4 artifact (4B, 27B, and since 2026-08-30 the 35B's routed experts, read verbatim
from the same `RedHatAI` checkpoint vLLM serves) or an artifact repacked from the same
GGUF (0.8B, the 35B's groupwise-int control rows, Flash-Next's W8 → Q4G32AM host bank).
Every artifact decodes bit-exact against its source before a number is recorded, and
every surogate row is probed for correctness **at** its concurrency.

Columns:

- **GPUs** — cards the engine used (1, or an 8-card layer pipeline).
- **users** — concurrent closed-loop clients.
- **prefill tok/s** — prompt tokens ÷ the run's wall time: a throughput share with the
  decode phases included, *not* a prompt-processing rate. At one user the
  prompt-processing rate is prompt tokens ÷ TTFT, marked †. A † figure is
  client-observed and carries queueing, so it reads below an engine's own prompt-eval
  timer — comparable between the engines here, not against a published number.
- **decode tok/s** — generated tokens ÷ wall time, summed over users (a one-user row
  is that stream's speed).
- **throughput tok/s** — prefill + decode: all tokens the engine moved per second.
- **TTFT p50** — median time to the first streamed token, reasoning or answer. Every
  model here reasons before it answers, so timing the first *content* chunk would
  report the end of the reasoning block — seconds where the truth is milliseconds.

**Never compare a column across shapes.** Both token columns divide by the same wall
time, so the shape pins their ratio: on prefill-heavy 2048/16 each request contributes
2,048 prompt and 16 generated tokens, making decode tok/s always prefill ÷ 128. That
decode figure is completions per second × 16, not a decode speed. On that shape read
prefill and TTFT; per-stream decode speed is what 512/128 and decode-heavy 128/512
measure, where the pinned ratio runs the other way.

## Board of record (2026-08-30)

One table per model, the same columns throughout; **every cell is measured** — the derived
figures the board used to carry are gone. **The first row of each model's table is the
current reference**: measured on 2026-08-30 after the clock caps were removed, one model at a
time on an idle host, with its vLLM pair in the same batch on a second card wherever the
checkpoint is still on this machine. Rows below it measure a *different* premise — another
shape, another card count, or a configuration lever — never an earlier pass of the same one:
superseded passes were removed on 2026-08-30 rather than kept for provenance, because a table
carrying three readings of one premise is read as three results. `BENCHMARKS_HISTORY.md` keeps
them.

### Qwen3.5-0.8B

| engine | GPUs | users | prefill tok/s | decode tok/s | throughput tok/s | TTFT p50 | comments |
|---|---:|---:|---:|---:|---:|---:|---|
| **surogate** | 1 | 100 | **46,225** | **11,166** | **57,391** | **20 ms** | GGUF Q4_K_M repack, 128 lanes, `--max-model-len 2048`, 8 client shards; 2026-08-30 07:08, uncapped GPU 0 |
| vLLM | 1 | 100 | 29,122 | 7,009 | 36,131 | 0.69 s | NVFP4 (`surogate/Qwen3.5-0.8B-NVFP4`), `--max-model-len 2048`; same batch, uncapped GPU 1. surogate **+59 % decode, 34× TTFT** |
| **surogate** | 1 | 100 | **81,376** | 636 | 82,012 | 2.39 s | prefill-heavy 2048/16 (2026-08-27, GPU4) |
| vLLM | 1 | 100 | 45,106 | 352 | 45,458 | 3.88 s | prefill-heavy, same card |
| **surogate** | 1 | 1 | **95,000 †** | **673** | — | **20 ms** | 2026-08-30 10:16, uncapped GPU 0, fp8 KV, ~1,900-token prompt |
| vLLM | 1 | 1 | 38,000 † | 498 | — | 50 ms | same batch, uncapped GPU 1. surogate **+35 % decode, 2.5× TTFT** |
| llama.cpp | 1 | 1 | **19,000 †** | **411** | — | **100 ms** | 2026-08-30 12:01, uncapped GPU 5, `llama-server -ngl 999 -c 4096 -np 1` on the same Q4_K_M GGUF. Its own prompt-eval timing is 34,500 tok/s; the † above is the board's prompt÷TTFT and carries the queueing. **Use `study/llama.cpp-master/build/bin` — the `llama-server` on `PATH` is Homebrew's Vulkan build** (no CUDA, ignores `CUDA_VISIBLE_DEVICES`) and reads 252 tg / 6,880 pp512 on `llama-bench`, roughly 40 % of the CUDA build |

### Qwen3.5-4B

| engine | GPUs | users | prefill tok/s | decode tok/s | throughput tok/s | TTFT p50 | comments |
|---|---:|---:|---:|---:|---:|---:|---|
| **surogate** | 1 | 100 | **22,121** | **5,345** | **27,466** | **40 ms** | NVFP4 3.56 GiB, 128 lanes, chunk 2,048, `--max-model-len 2048`, 8 client shards; 2026-08-30 07:17, uncapped GPU 6; re-confirmed on the one-token routing fix binary (`fc2906fc`): 21,752 / 5,256 / 40 ms on GPU 4, 3,738 requests, 0 errors — the fix cannot fire above one token, and does not |
| vLLM | 1 | 100 | 18,543 | 4,481 | 23,023 | 0.23 s | `surogate/Qwen3.5-4B-NVFP4` (ModelOpt), `--max-num-seqs 128`, `--max-model-len 2048`; 2026-08-30 13:24, uncapped GPU 5, 3,200 requests, 0 errors. surogate **+19 % decode, +19 % prefill, 5.8x TTFT** |
| **surogate** | 1 | 100 | **40,677** | 318 | 40,995 | 4.77 s | prefill-heavy 2048/16 (GPU5) |
| vLLM | 1 | 100 | 34,964 | 273 | 35,237 | 5.00 s | prefill-heavy, same card |
| **surogate** | 1 | 1 | **63,300 †** | **313** | — | **30 ms** | 2026-08-30 14:00, uncapped GPU 6, two passes (314.3, 312.8), fp8 KV, ~1,900-token prompt, on the decode-GEMV routing fix: the 4B's five linear shapes were unregistered NVFP4 geometries and ran a 128-row cuBLASLt tile on one row at every width; at one token they now take the decode GEMV (kernel-only tg128 213.5 → 350.8). **+26 % over vLLM's 249**, at half its TTFT |
| llama.cpp | 1 | 1 | **7,000 †** | **182** | — | **270 ms** | 2026-08-30 12:05, uncapped GPU 5, CUDA build, `unsloth/Qwen3.5-4B-GGUF` Q4_K_M (fetched for this row; our side is NVFP4). Its own prompt-eval timing is 13,000 tok/s |
| vLLM | 1 | 1 | 31,700 † | 249 | — | 60 ms | same checkpoint (`surogate/Qwen3.5-4B-NVFP4`, ModelOpt); 2026-08-30 13:27, uncapped GPU 5. Led decode by 22 % for four hours — that was our routing gap, not their kernels |

### Qwen3.8-27B

| engine | GPUs | users | prefill tok/s | decode tok/s | throughput tok/s | TTFT p50 | comments |
|---|---:|---:|---:|---:|---:|---:|---|
| **surogate** | 1 | 100 | **5,815** | **1,302** | **7,117** | **170 ms** | all-NVFP4, 128 lanes, chunk 4,096, `--max-model-len 2048`; 2026-08-30 07:11, uncapped GPU 2. Reproduces the pre-cap pass of 08-28 (1,330) within noise, so nothing regressed — the 1,068 measured under the caps was the cap |
| vLLM | 1 | 100 | 5,003 | 1,120 | 6,123 | 8.06 s | `sakamakismile/Qwen3.8-27B-MTP-NVFP4`, `--max-num-seqs 128`; 2026-08-30 07:28, uncapped GPU 3, idle host. surogate **+16 % decode, 47× TTFT** |
| surogate | 8 | 100 | 4,403 | 986 | 5,389 | 1.33 s | 8-stage pipeline, C3 + asynchronous prompt flights, `--max-model-len 2048`; 2026-08-30 08:18, uncapped. **Below the one card above** — the pipeline buys capacity, not throughput per card |
| **surogate** | 8 | 1 | **306** | **68.6** | **374** | **0.16 s** | same, one user: each card holds 6 layers, so a token costs ~0.9 ms across the eight stages against ~5 ms on one card |
| **surogate** | 1 | 100 | 471 | **1,884** | 2,355 | **14.2 s** | decode-heavy 128/512, 64 lanes (GPU5) |
| vLLM | 1 | 100 | 360 | 1,438 | 1,798 | 21.9 s | decode-heavy, same card |
| surogate | 1 | 100 | 11,557 | 87.8 | 11,645 | 15.38 s | prefill-heavy 2048/16, chunk 4,096, `--max-model-len 2304`, 128 lanes; 2026-08-30 18:57, GPU 1, same session as the vLLM row below. **89 % of vLLM's prefill** — and the shape is insensitive to the two obvious knobs: KV/admission moves it 1.5 % (context 4,096 → 2,304, 11,389 → 11,557) and the prompt chunk not at all (2,048 fails on KV entitlement; 8,192 reads 11,553 and 16,384 reads 11,538) |
| **vLLM** | 1 | 100 | **12,942** | 98.3 | **13,040** | **13.81 s** | prefill-heavy, same card and session; `--max-model-len 4096 --max-num-seqs 128`, and it ran with **less** KV than we did (66,901 tokens against our 104,384), so admission is not what separates them. Replaces a 2026-08-27 pass that read 11,818 |
| **surogate** | 1 | 1 | **11,200 †** | **70.8** | — | **170 ms** | 2026-08-30 10:21, uncapped GPU 0, fp8 KV, ~1,900-token prompt. Beats the 08-26 pass it replaces (45 tok/s at 352 ms) on both axes |
| **vLLM** | 1 | 1 | **13,600 †** | **71.7** | — | **140 ms** | `sakamakismile/Qwen3.8-27B-MTP-NVFP4`; 2026-08-30 10:37, uncapped GPU 1. The one shape where vLLM leads us at one user — decode within 1 %, TTFT 20 % better |
| llama.cpp | 1 | 1 | **1,610 †** | **44.8** | — | **1.18 s** | 2026-08-30 12:07, uncapped GPU 5, CUDA build, `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_M (fetched for this row; our side is all-NVFP4). Its own prompt-eval timing is 2,734 tok/s |

### Qwen3.6-35B-A3B

| engine | GPUs | users | prefill tok/s | decode tok/s | throughput tok/s | TTFT p50 | comments |
|---|---:|---:|---:|---:|---:|---:|---|
| **surogate** | 1 | 100 | **10,789** | **2,607** | **13,397** | **0.09 s** | **routed NVFP4 through TensorRT-LLM's cutlass fused MoE** (`csrc/src/third_party/trtllm_moe`, the kernel behind vLLM's row below), our router/shared expert/combine kept, single-token rounds kept on our own kernels; `--max-model-len 2048 --max-num-seqs 128 --max-num-batched-tokens 4096`; 2026-08-30 18:45, GPU 1, one session with its pair below. Correctness with thinking on, the mode this row is measured in: coherence 48/48 at 100 users, chunkcount 94 ok / 1 wrong / 0 garbage, longprompt 31 ok / 0 wrong |
| surogate | 1 | 100 | 7,708 | 1,863 | 9,571 | 0.12 s | the shipped groupwise-int artifact, second arm of the same session — same card and flags, servers started back to back. The routed-NVFP4 pair is **+39.9 % decode and +40.0 % prefill** |
| **surogate** | 1 | 16 | **5,829** | **1,409** | **7,238** | **0.07 s** | routed NVFP4, same session |
| surogate | 1 | 16 | 4,300 | 1,040 | 5,339 | 0.10 s | groupwise-int at 16 users, the pair for the row above (**+35.5 % decode**) |
| **surogate** | 1 | 1 | **1,307** | **316.2** | **1,623** | **0.03 s** | routed NVFP4, one user: every decode round is a single token and stays on our own kernels, so this is the weight format alone |
| surogate | 1 | 1 | 1,272 | 307.8 | 1,580 | 0.04 s | groupwise-int at one user, the pair for the row above (+2.7 % decode) |
| vLLM | 1 | 100 | 8,946 | 2,162 | 11,108 | 3.17 s | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` — the same checkpoint our routed experts are read from — `--max-num-seqs 128`; 2026-08-30 07:14, uncapped GPU 5. Against the reference at the top of this table: **−17 % decode, −17 % prefill, 35× the TTFT** |
| surogate | 8 | 100 | 8,785 | 2,123 | 10,908 | 0.58 s | 8-stage pipeline, C3 + asynchronous prompt flights, `--max-model-len 2048`; 2026-08-30 08:25, uncapped. **On the groupwise-int artifact**, where it beat the one card (1,984). It now sits 19 % *below* the one-card routed-NVFP4 row above; whether eight stages still gain on the new artifact is unmeasured — see Open items |
| **surogate** | 8 | 1 | **965** | **233.4** | **1,198** | **0.09 s** | same, one user: 3 B active split eight ways puts a token at ~0.9 ms |

### Qwen3.8-Flash-Next (111 GB MoE; on one card the experts live on the host)

| engine | GPUs | users | prefill tok/s | decode tok/s | throughput tok/s | TTFT p50 | comments |
|---|---:|---:|---:|---:|---:|---:|---|
| **surogate** | 1 | 1 | 128 | **28.7** | 156 | **1.36 s** | experts on the host: Q4G32AM bank (pinned, 91 GB), 3,000-slot expert cache, CPU split auto; 2026-08-30 07:55, uncapped GPU 6 (**x16 link**, gather 46 GB/s) |
| **surogate** | 1 | 16 | **298** | **66.9** | **365** | **2.62 s** | same, 81 % / 51 % measured shares; TTFT p90 14.7 s |
| surogate | 1 | 64 | 329 | 73.8 | 403 | 40.5 s | `--expert-slots 2000` so 64 lanes fit, `--pending-timeout-ms 600000` (the 30 s default expires a third of the queue here); host-bound, so 64 users only queue — TTFT p90 70.2 s |
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
| llama.cpp | 8 | 1 | 157 | 39.3 | 196 | 0.95 s | `--split-mode layer`, all resident |
| llama.cpp | 8 | 16 | 156 | 39.1 | 195 | 86 s | 16 of 48 requests timed out |
| llama.cpp | 8 | 64 | 99 | 24.7 | 124 | 311 s |  |

## Reading the table

- **Weight format beats every scheduling lever on this hardware**: NVFP4 gave
  the 4B +54 % and the 27B +37 % over our own W8/mixed artifacts, and now the
  35B +39.9 % — but only once its routed experts had a kernel worth the format.
  Our own routed-NVFP4 kernels reached 354 tok/s where the groupwise-int ones
  reached 1,832; TensorRT-LLM's fused MoE on the *same* weights reaches 2,607.
  The format was never the problem on that model, and neither was the schedule.
- **A borrowed kernel is not a whole answer either.** The runner permutes,
  groups and reduces for a batch, so at one token it costs more than it saves:
  serving every width through it drops single-user decode from 347 to 182 tok/s
  on the engine bench while lifting prefill 42 %. The crossover is a measured
  constant (`kSparseMoeTrtllmMinTokens`, overridable): swept at 2, 4, 8, 16 and
  47, everything from 2 to 16 measured the same and only a **single-token** round
  is worth keeping on our own kernels, so 2 is the default. Setting it to 47 — the
  intuitive "batch kernels want batches" guess — gives up 21 % at 16 users. With
  the measured value the 35B wins at 100, 16 and 1 users at once.
- **Lanes are 128** where the model fits them; with 100 users and 64 lanes a
  third of the load queued for a lane and that queue was the TTFT.
- **The 27B's prompt processing is the one place a competitor is ahead, and it is
  not the kernels.** vLLM serves 12,942 prompt tok/s on the prefill-heavy shape
  against our 11,557 — 89 %, measured in one session on one card. Our kernels do
  **12,707** at pp2048 in `ninfer_bench` with no server at all, which is vLLM's
  served number to within 2 %, so the loss sits between our kernels and our
  serving. Neither obvious knob touches it: admission is worth 1.5 % and the prompt
  chunk nothing across 2,048-16,384, and vLLM wins while running *less* KV than we
  do. The one-user 27B row is the same finding seen from the other end — a †
  prefill figure is TTFT restated, so 11,200 † against 13,600 † and 170 ms against
  140 ms are one fact, not two.
- **Flash-Next on one card is host-bound at one user and GPU-bound above it.**
  Measured at 16 users, the expert cache misses on only **2.1 %** of lookups and
  just **1.7 % of paths reach the CPU**, so the split and the PCIe gather cannot
  be what limits that row — the routed kernel over the resident pool is. At one
  user misses are frequent and the host path does set the pace, which is where
  the Q4 bank's halving of host bytes paid. The slot ring and the split carry
  16 users to 66.9 tok/s (median 66.7 over four runs, spread 0.6 %),
  and 64 users add queueing rather than throughput (73.8 tok/s at a 40 s TTFT).
  An earlier binary read 75.3 there and **that number has never reproduced**:
  card, NUMA node, clock cap and run-to-run noise were each eliminated (65.1-69.1
  across four cards and six repeats), so it is not on this board.
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
- **A pipeline is capacity, not throughput per card.** The 27B on eight stages
  serves 986 tok/s against 1,302 on one card, because with ~12 lanes per group a
  stage's per-round fixed cost does not shrink with its layer count. The 35B was
  the exception — 2,123 against 1,984 — but both of those are groupwise-int
  numbers, and one card on the routed-NVFP4 artifact now does 2,607. The eight-card
  pipeline has **not** been re-measured on that artifact, so the exception is
  unproven rather than disproven; comparing 2,123 to 2,607 across artifacts would
  be the mistake this board exists to prevent. Both configurations do slash
  single-user latency (27B 0.16 s TTFT, 35B 0.09 s), because each card holds only
  six layers.
- **Three configuration levers worth as much as a kernel change.** Capping the
  context to the workload (`--max-model-len 2048`) is worth **16 % on the 0.8B** —
  7,785 against 6,577 tok/s at 100 users, measured against itself in one
  clock-capped session, so read the ratio and not the rates — because auto sizes a
  4.25 M-token KV cache the shape never touches. On the 35B
  MoE, `--no-thinking` costs **27 % of decode** — 1,394 against 1,919 tok/s at 100
  users, measured against itself in one clock-capped session, so read the ratio and
  not the rates — since the generated text changes the expert spread per round. And on the host-offloaded model, in-engine NUMA
  placement is worth 3 % (`SUROGATE_SERVE_NUMA`, default `auto`: the shared expert
  bank interleaved, the per-device staging bound to that device's node — which
  also makes the launcher's `numactl --interleave=all` redundant, measured at
  ±0.5 %). All three belong in every row's comment.
- Every surogate row above is from a binary that passes the correctness batteries
  at its concurrency (`surogate/serve/tools/probe/`), run with thinking on — the
  mode the rows are measured in. The probes score the model's **answer** and never
  its reasoning, and report a request whose budget expired mid-reasoning as
  `truncated` rather than wrong; `probe/chat.py` carries the rule and the reason.

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
- **The 35B's eight-card pipeline has not been measured on the routed-NVFP4
  artifact.** Its 2,123 tok/s is a groupwise-int number, taken when one card did
  1,984; one card now does 2,607. Eight stages may still add capacity, or the
  borrowed kernel may have removed the reason to spread the model at all — the
  measurement is one server launch and one probe, and until it is run the "MoE
  gains from a pipeline" claim stands only for the old artifact.
- **Two rows were taken at 62 s and the rest at 76-90 s.** The window is inside the
  band the Method names, but an early pass measured a shorter window reading up to
  25 % high, and that sensitivity has never been re-checked on the current binary.
  One 90 s repeat of the 35B reference against its 62 s reading settles it.
- Otherwise **every row on this board is from 2026-08-30**, on the same binary and on
  cards with no clock cap. (A boot service capped per-card SM clocks between
  2026-08-29 09:34 and 2026-08-30 07:07 — invisible to `clocks.max.sm`, worth 1,519
  against 1,986 tok/s on a controlled pair. It is gone, no row here was measured
  inside it, and the forensics are in `BENCHMARKS_HISTORY.md`. Power stays limited to
  400 W per card, which is what the clock settles against — ~2.0 GHz on the 27B.)
