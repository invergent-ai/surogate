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
to its pair on the same card in the same batch**. Most rows were taken one model at a
time on an idle host; the four prefill-heavy 2048/16 rows on the 0.8B and 4B were taken
with three model pairs running at once, one per card, so each pair is internally fair —
both arms started and probed together under the same host load — but their absolute
rates may sit below what an idle host would give. Their comments say so. One binary repeats within **0.6 %**
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
| **surogate** | 1 | 100 | **46,225** | **11,166** | **57,391** | **20 ms** | GGUF Q4_K_M **repack** — an artifact path retired on 2026-09-02, see the native rows below; 128 lanes, `--max-model-len 2048`, 8 client shards; 2026-08-30 07:08, uncapped GPU 0 |
| vLLM | 1 | 100 | 29,122 | 7,009 | 36,131 | 0.69 s | NVFP4 (`surogate/Qwen3.5-0.8B-NVFP4`), `--max-model-len 2048`; same batch, uncapped GPU 1. surogate **+59 % decode, 34× TTFT** |
| **surogate** | 1 | 100 | **117,563** | 910.5 | **118,473** | **1.48 s** | prefill-heavy 2048/16, `--max-model-len 2304`, 128 lanes, chunk 4,096, 4 client shards; 2026-08-30 19:17, GPU 0. Replaces a 2026-08-27 pass that read 81,376 — **+44 % on the current binary** |
| vLLM | 1 | 100 | 46,990 | 363.6 | 47,354 | 3.70 s | prefill-heavy, GPU 1, launched and probed concurrently with the row above. surogate **+150 % prefill** |
| **surogate** | 1 | 1 | **95,000 †** | **673** | — | **20 ms** | 2026-08-30 10:16, uncapped GPU 0, fp8 KV, ~1,900-token prompt |
| vLLM | 1 | 1 | 38,000 † | 498 | — | 50 ms | same batch, uncapped GPU 1. surogate **+35 % decode, 2.5× TTFT** |
| llama.cpp | 1 | 1 | **19,000 †** | **411** | — | **100 ms** | 2026-08-30 12:01, uncapped GPU 5, `llama-server -ngl 999 -c 4096 -np 1` on the same Q4_K_M GGUF. Its own prompt-eval timing is 34,500 tok/s; the † above is the board's prompt÷TTFT and carries the queueing. **Use `study/llama.cpp-master/build/bin` — the `llama-server` on `PATH` is Homebrew's Vulkan build** (no CUDA, ignores `CUDA_VISIBLE_DEVICES`) and reads 252 tg / 6,880 pp512 on `llama-bench`, roughly 40 % of the CUDA build |


**GGUF served natively (2026-09-02).** The three engines on one idle 5090, the same
closed-loop client, 512/128, salted prompts, staggered workers, a 60 s window after 15 s of
warm-up, each engine at its own natural weight format: surogate and llama.cpp both read
`models/Qwen3.5-0.8B-Q4_K_M.gguf`, vLLM reads `surogate/Qwen3.5-0.8B-NVFP4`. surogate's
artifact is now the file's own K-quants — Q4_K, Q5_K and Q6_K blocks byte-for-byte, no
dequantise-and-requantise — at 713 MB.

| engine | GPUs | users | prefill tok/s | decode tok/s | throughput tok/s | latency p50 | comments |
|---|---:|---:|---:|---:|---:|---:|---|
| **surogate** | 1 | 1 | **4,372** | **802** | **5,174** | **0.16 s** | native K-quant GGUF, `--kv-capacity auto` |
| vLLM 0.27.1 | 1 | 1 | 2,053 | 346 | 2,399 | 0.25 s | NVFP4. surogate **2.3× decode, 2.1× prefill** |
| llama.cpp | 1 | 1 | 2,477 | 454 | 2,932 | 0.28 s | same GGUF, `-ngl 999 -fa 1 -np 16`. surogate **1.8× decode and prefill** |
| **surogate** | 1 | 8 | **15,071** | **2,765** | **17,836** | **0.37 s** | |
| vLLM 0.27.1 | 1 | 8 | 10,496 | 1,768 | 12,264 | 0.56 s | surogate **1.6× decode, 1.4× prefill** |
| llama.cpp | 1 | 8 | 3,721 | 683 | 4,404 | 1.53 s | surogate **4.1× decode and prefill, 4.1× latency** |

**A routed MoE served natively (2026-09-03).** `models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
(20.6 GiB, 34.66 B parameters, 256 experts of which 8 route) on one idle 5090, single stream,
a ~700-token prompt and 128 generated tokens, warm, greedy, three prompt variants (a repeated
prompt is served from the prefix cache and measures nothing). surogate and llama.cpp read the same
file; surogate's artifact carries the routed experts as the file's own Q4_K, Q5_K and Q6_K
superblocks, 34.6 B of its 34.7 B parameters copied byte for byte.

**Q6_K down, fixed (2026-09-04).** The op benchmark (`sinfer_sparse_moe_bench --codec
q4_k-q6_k`, 256 unique experts, warm, graph replay, one 5090) had Q6_K down doubling the MoE
body: 1,014 / 1,362 / 1,182 us at 128 / 512 / 1,024 tokens against q4_k-q4_k's 522 / 657 / 764.
Cause: a 210-byte block is only two-byte aligned, so the codec staged its 96-byte tile as
forty-eight scalar loads where every other codec issues six 16-byte `cp_async`. It now covers
each span with aligned 16-byte copies from the rounded-down address and folds the row's offset
into the readers: **643 / 842 / 963 us**, outputs bit-identical, 37 % of peak bandwidth against
Q4_K's 40 %.

| engine | routed expert format | prefill tok/s | decode tok/s | comments |
|---|---|---:|---:|---|
| **surogate** | native K-quant, int8 tensor-core prefill route (2026-09-03) | **13,195** | **315** | the routed experts' prefill on llama.cpp's arithmetic (int8 activations per 32 with block sums, the file's codes as the other MMA operand); decode untouched. Wikitext-2 perplexity, 145 windows of 2048: **6.2370 ± 0.040**, llama.cpp 6.2311 ± 0.040, the BF16-activation path 6.2378 |
| surogate | native K-quant, BF16 activations | 10,299 | 316 | the path before the int8 route, same session and flags as the row above; reads the GGUF's blocks unchanged |
| llama.cpp | native K-quant | 8,408 | 278 | `llama-bench -ngl 99 -p 512 -n 128 -r 3`: pp512 8,407.89 ± 1,576.40, tg128 277.96 ± 3.65. surogate **1.18x decode, 1.12x prefill** |
| surogate | dequantised to Q4G64/Q5G64 | 13,730 | 346 | the older path, kept as the ceiling this one is closing on; it re-quantises, so it is not bit-exact |

The native path's remaining prefill gap to our own row-split kernel is the K-quant decode
itself: a 64-wide tile spans two sub-block scales and carries an affine min, and Q6_K's
210-byte block forces scalar staging where the others use `cp_async`.

Isolated kernel rates on the same file, `llama-bench -p 512,2048 -n 128 -fa 1` against the
engine's own accounting: prefill **97,062** vs 39,511 (pp512) and **107,604** vs 42,863
(pp2048); decode **864** vs 809 (tg128). The concurrency gap is wider than the single-stream
gap because llama.cpp's server does not batch these as well as its kernels run.

### Qwen3.5-4B

| engine | GPUs | users | prefill tok/s | decode tok/s | throughput tok/s | TTFT p50 | comments |
|---|---:|---:|---:|---:|---:|---:|---|
| **surogate** | 1 | 100 | **22,121** | **5,345** | **27,466** | **40 ms** | NVFP4 3.56 GiB, 128 lanes, chunk 2,048, `--max-model-len 2048`, 8 client shards; 2026-08-30 07:17, uncapped GPU 6; re-confirmed on the one-token routing fix binary (`fc2906fc`): 21,752 / 5,256 / 40 ms on GPU 4, 3,738 requests, 0 errors — the fix cannot fire above one token, and does not |
| vLLM | 1 | 100 | 18,543 | 4,481 | 23,023 | 0.23 s | `surogate/Qwen3.5-4B-NVFP4` (ModelOpt), `--max-num-seqs 128`, `--max-model-len 2048`; 2026-08-30 13:24, uncapped GPU 5, 3,200 requests, 0 errors. surogate **+19 % decode, +19 % prefill, 5.8x TTFT** |
| **surogate** | 1 | 100 | **48,259** | 373.8 | **48,633** | **3.61 s** | prefill-heavy 2048/16, `--max-model-len 2304`, 128 lanes, chunk 4,096, 3 client shards; 2026-08-30 19:17, GPU 2. Replaces an undated pass that read 40,677 — **+19 % on the current binary** |
| vLLM | 1 | 100 | 36,308 | 281.2 | 36,589 | 4.80 s | prefill-heavy, GPU 3, launched and probed concurrently with the row above. surogate **+33 % prefill** |
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
| **surogate** | 1 | 100 | 768 | **2,097** | **2,865** | **13.7 s** | decode-heavy 128/512, 64 lanes, `--max-model-len 1024 --max-pending-requests 512`; 2026-08-30 19:32, GPU 4, launched and probed concurrently with the row below. Replaces an undated pass that read 1,884. The pending queue is not optional here — 100 users against 64 lanes rejects with 429 on the default |
| vLLM | 1 | 100 | 608 | 1,662 | 2,270 | 21.8 s | decode-heavy, GPU 5, same session. surogate **+26 % decode at 0.6× the TTFT** |
| surogate | 1 | 100 | 11,557 | 87.8 | 11,645 | 15.38 s | prefill-heavy 2048/16, chunk 4,096, `--max-model-len 2304`, 128 lanes; 2026-08-30 18:57, GPU 1, same session as the vLLM row below. **89 % of vLLM's prefill** — and the shape is insensitive to the two obvious knobs: KV/admission moves it 1.5 % (context 4,096 → 2,304, 11,389 → 11,557) and the prompt chunk not at all (2,048 fails on KV entitlement; 8,192 reads 11,553 and 16,384 reads 11,538) |
| **vLLM** | 1 | 100 | **12,942** | 98.3 | **13,040** | **13.81 s** | prefill-heavy, same card and session; `--max-model-len 4096 --max-num-seqs 128`, and it ran with **less** KV than we did (66,901 tokens against our 104,384), so admission is not what separates them. Replaces a 2026-08-27 pass that read 11,818 |
| **surogate** | 1 | 1 | **11,200 †** | **70.8** | — | **170 ms** | 2026-08-30 10:21, uncapped GPU 0, fp8 KV, ~1,900-token prompt. Beats the 08-26 pass it replaces (45 tok/s at 352 ms) on both axes |
| **vLLM** | 1 | 1 | **13,600 †** | **71.7** | — | **140 ms** | `sakamakismile/Qwen3.8-27B-MTP-NVFP4`; 2026-08-30 10:37, uncapped GPU 1. The one shape where vLLM leads us at one user — decode within 1 %, TTFT 20 % better |
| **surogate** | 1 | 1 | 371 | **83.3** | — | **0.24 s** | 2026-09-04 22:03, GPU 0, `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_M served natively **with MTP** (`--spec mtp --draft-tokens 1`; drafts 2 / 3: 76.9 / 69.9), every object but two Q8_0 projections read from the file: the UD mixture's mixed-type fused parents as typed segments, the V-head-permuted GDN out_proj with the permutation on the activation, and the IQ4_XS GEMV on a byte-permute table lookup (it had been a per-lane constant read: 232 us/call, 12x Q4_K's). The 46.5 row of 22:00 that morning had 33 % of the bytes requantised to Q4G64/Q5G64 and the slow IQ4_XS kernel on the rest; PPL 5.1699 -> 5.0477 against llama.cpp's 5.0166. |
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

All surogate one-card rows: 2026-09-04, the native GGUF read in place (no artifact repack — the
Q4G32AM host bank is decoded from the file's blocks at load), **engine defaults**: no
`--expert-slots`, no `--host-expert-bank`, the pool and the host worker pool size themselves;
`--cpu-moe-share auto --kv-capacity auto --max-model-len 4096`, chunk 8,192, fp8 KV, GPU 0 —
an **x16** card (GPUs 2, 3, 5 and 7 of this host are x8 and read half the gather rate). The
eight-card rows are still the 2026-08-30 `b4ee3216` pass on the repacked artifact. The 08-30
one-card rows this replaces read 32.0 / 91.1 / 110.1 decode at 1 / 16 / 64 users; the day's
recovery from 18.8 is in `design/INFERENCE.md`. llama.cpp rows: `-cmoe
-b 4096 -ub 4096`, the flags its own community benchmarks use — the earlier `-ot exps=CPU` rows
without batch flags understated it 3.9× and are gone.

| engine | GPUs | users | prefill tok/s | decode tok/s | throughput tok/s | TTFT p50 | comments |
|---|---:|---:|---:|---:|---:|---:|---|
| **surogate** | 1 | 1 | **150** | **33.6** | **183** | **0.84 s** | defaults: pool 3,004 slots (14.6 GiB, sized to leave the runtime its floor), unpinned host workers sized to the cores other jobs leave free, host round started and joined through stream memory operations (no host-function dispatch: +6 % decode over the event path), the CPU/PCIe split measured with both sides running against each other (135 vs 46 GB/s → 75 % of misses on the host), min-tokens 1 so the split fires on one-token rounds |
| llama.cpp | 1 | 1 | 93 | 20.8 | 114 | 2.95 s | the same GGUF, served by both engines as the file's own blocks now. surogate **+62 % decode, 3.5× the prompt rate** |
| **surogate** | 1 | 1 | — | **40.8** | — | **7.06 s** | 28k prompt into 131k context (23.1 GiB VRAM): **3,966 tok/s prompt processing**; decode is the post-28k stream rate |
| surogate (re-measured 2026-09-04) | 1 | 1 | — | 39.1 | — | 7.07 s | the same shape on today's tree with `--max-num-batched-tokens 8192`: 3,989 tok/s prompt processing, TTFT 7.07 s, 39.1 tok/s after the 28k prompt -- the row holds. On the defaults (prefill chunk 2,048, the compromise the 8-card pipeline wanted) the same request reads 11.6 s / 2,434 tok/s / 39.4: a one-card long-prompt serve should pass the flag |
| surogate `--spec mtp --draft-tokens 1` (2026-09-04) | 1 | 1 | 157 | **35.2** | — | 0.85 s | the NextN head at the shortest draft, 71.6 % accepted; drafts 2 / 3 read 34.6 / 32.6 (55 % / 45 %) against 32.6 without -- every extra column widens the expert gather faster than the acceptance pays |
| llama.cpp | 1 | 1 | — | 27.3 | — | 23.02 s | 28k prompt, 80k context: 1,216 tok/s prompt processing. surogate **3.3× ingestion, +49 % decode** |
| cafe-llama.cpp `-hmoe` | 1 | 1 | — | 18.8 / 4.3 | — | 3.55 s / 25.46 s | 512 and 28k prompts. Experts pinned in host memory, computed on the GPU over PCIe — **our architecture in their engine** (935-1,100 tok/s at 28k); kept as the like-for-like reference |
| **surogate** | 1 | 16 | **381** | **85.7** | **467** | **1.74 s** | defaults, `--max-num-seqs 16`: KV auto 65,536 tokens. Run-to-run spread at 16 users is ~±8 % (83.9 and 86.8 on the same day) |
| llama.cpp | 1 | 16 | 253 | 56.8 | 310 | 16.17 s | `-np 16`. surogate **+51 % decode at 9× lower TTFT** |
| **surogate** | 1 | 64 | **518** | **116.4** | **635** | **4.73 s** | defaults, `--max-num-seqs 64 --max-pending-requests 512`: the pool sized itself to 1,985 slots so 64 lanes' KV fits (74,240 tokens) — the 2,000 the old row set by hand, derived. Above the 08-30 row (110.1) for the first time |
| llama.cpp | 1 | 64 | 46 | 10.4 | 56 | 655 s | `-np 64`: CPU expert compute serialises across 64 decodes and the queue is the run — every request ~13 min. surogate **11.2×** |
| ik_llama.cpp | 1 | 1 | 87 | 21.8 | 109 | 1.8 s | AVX-512 iqk CPU-MoE kernels, `-ot exps=CPU` |
| ik_llama.cpp | 1 | 16 | 96 | 23.9 | 120 | 30 s |  |
| ik_llama.cpp | 1 | 1 | 1,068 | 40.4 | — | — | **reported, not measured here** (2026-08-30): same commit 7cff686d on an **RTX 3090 24 GB + Ryzen 9 9950X**, AD-4.27bpw Q4_K_M, 3-run average at temperature 0, single slot, 10,006-token prompt without cache reuse, 128 generated; KV Q8_0/Q8_0 (their setting — no board row of ours quantises the KV), 22.1 GB VRAM |
| **surogate** | 8 | 1 | **193** | **43.3** | **236** | **1.07 s** | 8 stages, 3,072 slots per card (every expert resident), chunk 8,192 |
| **surogate** | 8 | 16 | **1,214** | **272.3** | **1,486** | **2.27 s** | chunk 1,024 — **a pipeline wants small chunks**: chunk 8,192 reads 208 here (one giant chunk starves 8 stages of in-flight work) |
| **surogate** | 8 | 64 | **1,807** | **405.2** | **2,212** | **2.79 s** | chunk 1,024; chunk 8,192 reads 296, chunk 2,048 sits between (364) |
| llama.cpp | 8 | 1 | 157 | 39.3 | 196 | 0.95 s | `--split-mode layer`, all resident |
| llama.cpp | 8 | 16 | 156 | 39.1 | 195 | 86 s | 16 of 48 requests timed out |
| llama.cpp | 8 | 64 | 99 | 24.7 | 124 | 311 s |  |

## Prefill on the 27B GGUF, and where it goes (2026-09-04)

Same file, same probe, one card, one user, a 2,048-token prompt:

| | TTFT | the engine's own prompt-eval |
|---|---:|---:|
| **surogate** (22:07, everything native) | **752 ms** | **2,804 tok/s** |
| surogate, 12:00 the same day (33 % of bytes requantised, slow IQ4_XS GEMV) | 812 ms | 2,600 tok/s |
| llama.cpp `llama-server` | 1,265 ms | 2,565 tok/s |
| llama.cpp `llama-bench pp2048` | — | **3,190 tok/s** |

Through the server we are ahead on both. `llama-bench` is the compute bar, though: it hands the
model one 2,048-wide batch where `llama-server` splits into 512-token micro-batches, and that
number is 14 % above ours (23 % before the native pass). End to end we run 112 TFLOP of prompt
at **149 TFLOP/s effective** against its 174. Re-profiled after the native pass (22:09): the
whole prefill is the BF16 wide route -- cutlass BF16 GEMMs **75 %**, `dequantize_rows` staging
**14.5 %**, GDN core and attention ~3 % -- so the dequantise-once-then-GEMM route is the entire
compute story and the two levers left are its staging pass and cuBLASLt's own rate. The table
below is the morning's, before the native pass, kept for the route rates it measured.

An `nsys` capture with `--cuda-graph-trace=node` (the prefill is a captured graph; without that
flag the profile shows almost nothing) says where a prefill's GPU time goes:

| | share | per prefill |
|---|---:|---:|
| cutlass BF16 GEMMs (the dequantise-then-GEMM route the native K-quants take) | 42 % | ~340 ms |
| our groupwise kernels on the re-encoded Q4G64/Q5G64 halves | 38 % | ~300 ms |
| `dequantize_rows` staging for the BF16 route | 7.5 % | ~60 ms |
| GDN core, attention, norms | ~4 % | ~30 ms |

So prefill is GEMM-bound and the GPU is busy; there is no scheduling gap to reclaim. The routes
themselves are the ceiling: BF16 cuBLASLt measures **222 TFLOP/s** at the dominant MLP shape
(n=34,816, k=5,120, T=2,048), our Q4G64 kernel **187**, W8 **159**, and the fused Q4 SwiGLU
**183** -- all far below the card's 838 TFLOP/s of int8/fp8 tensor throughput, which is what
llama.cpp's MMQ computes on. Two consequences worth writing down:

- **Re-routing the groupwise weights through the BF16 path is not worth it.** It buys 222 over
  187, but the dequantise pass costs ~0.4 ms on a ~3.9 ms weight, so the crossover sits near
  T = 1,500 and the whole-model win is ~5 %.
- **The int8 dense prefill GEMM was built and measured a loss.** The routed experts' int8 tile
  (`sparse_moe_prefill_ggml_i8_*`, 498 us against 758 there) run over the dense K-quant
  linears: 2,048 tokens **881 ms / 2,390 tok/s** against 812 / 2,600 on the BF16 route, 512
  tokens 269 ms / 2,120 against 257 / 2,232. The profile says why: the int8 kernels took ~262
  ms per prefill where cuBLASLt plus staging took ~225 on the same weights, because the tile's
  instruction stream is the per-32 affine scale-apply (a convert and two FMAs per accumulator
  per MMA), not the MMA -- the same structure as llama.cpp's MMQ, which is why `llama-bench`
  is at 172 TFLOP/s effective and not 800. The experts win with that tile only because their
  alternative, a per-expert dequantisation at small M, is worse. Removed; the design lives in
  `TODOv2.md`, "Measured and rejected". What it left behind is a numerics fix: the tile's
  activation planes carried llama.cpp's raw Σx in the (scale, sum) pair while the GEMV route
  sums the codes, and on a real K-quant tensor -- whose weights are small differences of the
  scale and min terms -- the raw sum lands 2.5x further from the exact product (1.3e-2 against
  5.3e-3 relative). The planes now carry d·Σq; the routed-expert prefill inherits it.

## Accuracy gates (2026-09-04)

**The KV cache default is `auto` (2026-09-04).** The `qwen3` target's 1-3 % offset against
llama.cpp was the FP8 KV cache: e4m3's three mantissa bits are ~2 % of noise on every K and V,
and a pure-attention stack pays it in every layer (attention output 3e-2 from exact on the
engine's own q/k/v; 1.5e-3 with a BF16 cache), while a 3:1 GDN stack pays 0-0.4 % and keeps the
halved cache. `auto` is BF16 where every layer is attention and e4m3 otherwise;
`--kv-cache-dtype bf16|fp8` pins either.

Perplexity on wikitext-2 test, llama-perplexity's own 2048-token windows (the first 40),
ours eager with the raw prompt (`surogate/serve/tools/eval/perplexity.py`) against
`llama-perplexity` on the same file and windows. The bar is "no worse than llama.cpp".

| file | stored types by weight | surogate | llama.cpp |
|---|---|---:|---:|
| Qwen3.6-35B-A3B-UD-Q4_K_M (145 windows, 2026-09-03) | Q4_K/Q5_K/Q6_K | 6.2370 +/- 0.040 | 6.2311 +/- 0.040 |
| Qwen3.6-35B-A3B-UD-Q4_K_M (145 windows, 2026-09-04, int8 planes carry d·Σq; the windows differ from the row above, so compare gaps: 0.054 % against 0.095 %) | Q4_K/Q5_K/Q6_K | 5.9118 +/- 0.037 | 5.9086 +/- 0.037 |
| Qwen3-0.6B-Q4_K_M (40 windows, 2026-09-04, **BF16 KV cache** -- the new `auto` default for a pure-attention stack; fp32 reference over the same weights 17.4127) | Q4_K/Q6_K | 17.4315 +/- 0.281 | 17.5103 +/- 0.281 |
| Qwen3-0.6B-IQ4_XS (40 windows, 2026-09-04, BF16 KV cache; the FP8 row above read 18.330) | IQ4_XS/Q6_K | 17.8580 +/- 0.286 | 17.8659 +/- 0.286 |
| Qwen3.8-27B-UD-Q4_K_M (8 windows, 2026-09-04, BF16 KV cache; FP8 read 5.1699) | Q4_K/Q5_K/Q6_K/IQ4_XS/IQ3_S | 5.1495 +/- 0.133 | 5.0166 +/- 0.127 |
| Qwen3.8-27B-UD-Q4_K_M (8 windows, 2026-09-04 22:07, **everything native**: typed segments for the mixed-type fused parents, the GDN out_proj permutation on the activation; FP8 KV cache -- BF16 reads 5.0523, inside the bar) | Q4_K/Q5_K/Q6_K/IQ4_XS/IQ3_S | **5.0477 +/- 0.129** | 5.0166 +/- 0.127 |
| Qwen3.5-0.8B **NVFP4** (`surogate/Qwen3.5-0.8B-NVFP4`, ModelOpt; 40 windows, 2026-09-04) against llama.cpp on the IQ4_XS GGUF. The same NVFP4 weights dequantised exactly inside the BF16 transformers model score 17.33 with exact activations: the checkpoint, not the engine | NVFP4 W4A4 | 17.5679 +/- 0.267 | 15.1511 +/- 0.226 (IQ4_XS) |
| Qwen3.5-2B **NVFP4** (`surogate/Qwen3.5-2B-NVFP4`; 40 windows) against llama.cpp on Q4_K_M | NVFP4 W4A4 | 11.6437 +/- 0.162 | 10.2912 +/- 0.140 (Q4_K_M) |
| Qwen3.5-4B **NVFP4** (`surogate/Qwen3.5-4B-NVFP4`; 40 windows) against llama.cpp on Q4_K_M | NVFP4 W4A4 | 8.9714 +/- 0.121 | 8.2445 +/- 0.108 (Q4_K_M) |
| Qwen3.5-0.8B **block-scaled FP8** (`surogate/Qwen3.5-0.8B-FP8`, HF fine-grained [128,128]; 40 windows, 2026-09-04) -- the BF16 model itself scores **14.60** on these windows (torch), so the recipe costs 1 % | FP8 E4M3 blk128, A8 per token per 128 | **14.7418 +/- 0.220** | 15.1511 +/- 0.226 (IQ4_XS) |
| Qwen3.5-2B **block-scaled FP8** (`surogate/Qwen3.5-2B-FP8`; 40 windows) against llama.cpp on Q4_K_M; 518 tok/s decode | FP8 E4M3 blk128 | **10.1182 +/- 0.137** | 10.2912 +/- 0.140 (Q4_K_M) |
| Qwen3.5-4B **block-scaled FP8** (`surogate/Qwen3.5-4B-FP8`; 40 windows) against llama.cpp on Q4_K_M; 261 tok/s decode | FP8 E4M3 blk128 | **8.1477 +/- 0.107** | 8.2445 +/- 0.108 (Q4_K_M) |
| Qwen3.5-0.8B **per-channel FP8** (`mahadev9/Qwen3.5-0.8B-fp8`, compressed-tensors, one scale per row, dynamic activations; 40 windows) -- the same weights dequantised exactly in torch score 15.006, so the per-token-per-128 activation quantisation costs 0.7 %; 831 tok/s decode | FP8 E4M3 per row, A8 per token per 128 | **15.1148 +/- 0.227** | 15.1511 +/- 0.226 (IQ4_XS) |
| Qwen3.5-0.8B-IQ4_XS | IQ4_XS 50 %, Q6_K 43 % | **15.094 +/- 0.225** | 15.151 +/- 0.226 |
| Qwen3.5-0.8B-UD-Q2_K_XL | Q2_K/Q3_K, IQ3_S/IQ3_XXS/IQ2_S/IQ4_XS | 20.209 +/- 0.305 | 20.016 +/- 0.302 |
| Qwen3-0.6B-UD-IQ2_M | IQ2_S 34 %, IQ3_S 16 %, IQ3_XXS | **40.128 +/- 0.702** | 42.045 +/- 0.743 |
| Qwen3-0.6B-UD-IQ3_XXS | IQ3_XXS 39 %, IQ3_S, IQ2_S | **29.509 +/- 0.505** | 30.250 +/- 0.522 |
| Qwen3-0.6B-IQ4_XS | IQ4_XS 64 %, Q6_K 35 % | 18.330 +/- 0.294 | 17.866 +/- 0.286 |
| Qwen3-0.6B-Q4_K_M (control) | Q4_K 54 %, Q6_K 44 % | 17.675 +/- 0.28 | 17.510 +/- 0.28 |

The IQ2/IQ3 rows come out ahead of the reference because our vec-dots evaluate the block
scale exactly where llama.cpp's integer form truncates. The two Qwen3-0.6B rows behind the
reference are the `qwen3` target, not the formats: its Q4_K_M control shows the same offset on
old types only, and the same IQ4_XS codec matches on the 0.8B (TODOv2 item 5).

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
  **12,707** at pp2048 in `sinfer_bench` with no server at all, which is vLLM's
  served number to within 2 %, so the loss sits between our kernels and our
  serving. Neither obvious knob touches it: admission is worth 1.5 % and the prompt
  chunk nothing across 2,048-16,384, and vLLM wins while running *less* KV than we
  do. The one-user 27B row is the same finding seen from the other end — a †
  prefill figure is TTFT restated, so 11,200 † against 13,600 † and 170 ms against
  140 ms are one fact, not two.
- **Flash-Next: the CPU belongs in decode, not in prefill.** The decode split (70 %
  of misses on the host, measured shares) carries the one-user rows; the *prefill*
  split was a pure loss at every chunk width — the host GEMM runs at 16 % of VNNI
  peak and every layer's combine waits on its host tail — and turning it off is
  most of tonight's +54-64 % on one card (28.4→32.0, 66.9→91.1, 73.8→110.1) and
  the whole of the ingestion flip (484 → 3,966 tok/s at 28k, against llama.cpp's
  1,216 with its best flags). Pure-GPU prefill scales almost linearly with the
  chunk: 1,632 / 2,428 / 3,275 / 3,944 at 1k/2k/4k/8k. The expert cache itself is
  at its ceiling (2.1 % misses at 16 users); the reported ik_llama.cpp figure on a
  Ryzen 9 9950X desktop — 40.4 tok/s, 1.85× this host's EPYC on the same commit —
  still sizes the host-CPU lever for CPU-offload engines, and the x8-link cards
  (2/3/5/7) still cost a gather-bound row a third.

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
  `csrc/build-serve/serve_bench/sinfer_bench` instead, which does load, warm-up
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
  `python -m surogate.serve.convert.qwen3_5_moe.convert --model <BF16 dir>
  --routed-nvfp4 <compressed-tensors NVFP4 dir> --out <path>.sinfer` — 77 s, and
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
  1,984; one card now does 2,607. (The Flash-Next 8-card rows were re-measured
  tonight on the current binary; the 35B's were not.)

- **Two rows were taken at 62 s and the rest at 76-97 s.** The window is inside the
  band the Method names, but an early pass measured a shorter window reading up to
  25 % high, and that sensitivity has never been re-checked on the current binary.
  One 90 s repeat of the 35B reference against its 62 s reading settles it.
- **Agent-shaped traffic is not measured at all.** Every row here salts its prompts so
  nothing shares a prefix, which is the worst case for prompt processing and the
  opposite of a harness re-sending a growing conversation behind a fixed system prompt
  and tool definitions. `probe/agentloop.py` drives that shape, but its first run could
  not separate prefix reuse on from off — the conversation grew only 4.5k → 5.5k tokens,
  so the difference sat inside TTFT overhead. It needs a prompt that grows several-fold
  before it says anything.
- Otherwise **every row on this board is from 2026-08-30**, on the same binary and on
  cards with no clock cap. (A boot service capped per-card SM clocks between
  2026-08-29 09:34 and 2026-08-30 07:07 — invisible to `clocks.max.sm`, worth 1,519
  against 1,986 tok/s on a controlled pair. It is gone, no row here was measured
  inside it, and the forensics are in `BENCHMARKS_HISTORY.md`. Power stays limited to
  400 W per card, which is what the clock settles against — ~2.0 GHz on the 27B.)
