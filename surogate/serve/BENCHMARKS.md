# Serving benchmarks

## Local models — 2026-09-11

Fresh **surogate** measurements of the models currently present in `models/`. The older
vLLM and llama.cpp comparisons below retain their original dates; those engines were not
re-benchmarked in this run. Source identities, launch commands, request counts, and full
measurements are in the [JSON report](tools/bench/results/2026-09-11-local-models.json).

**Hardware:** RTX 5090 32 GB cards, driver 590.44.01, **400 W power limit**, two EPYC 9124
CPUs (32 physical cores total), and 503 GiB RAM. The earlier uncapped measurements are not
direct performance comparisons. GPUs 1 and 2 reported 100% utilization and were excluded.

**Workload:** exact 512-token prose inputs sent as token IDs and 128 generated tokens over
streaming Chat Completions. Tokenization happens before timing. Sampling is greedy, EOS
is ignored, and prefix reuse is disabled. Measurements cover throughput and latency with
fixed token counts. Small models use 60-second windows after an initial warmup and
15 seconds of settling. One-GPU MoE runs
use 180-second windows and 30 seconds of settling. Both sixteen-client GLM rows use
180-second windows, 60 seconds of settling, and client starts spread over 30 seconds.
Other pipeline runs use 90-second windows and 20 seconds of settling. Clients send another
request after their previous response finishes. Loading and
conversion are excluded.

**Reading the tables:** generated tokens/s is aggregate output throughput, including time
spent processing prompts. Prompt throughput is four times that number for this 512/128
workload. TTFT measures the first nonempty reasoning or answer output. Latency is the full
request duration. Throughput counts requests completed inside each fixed window; outstanding
requests drain afterward. Request counts are shown because slow configurations have fewer
samples. All successful rows completed without request errors.

### Generation

| Model | Placement / decoding | GPUs | Clients | Generated tokens/s | TTFT p50 | Latency p50 | Requests |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B Q5_K_M | Resident | 1 | 1 | 162.1 | 0.076 s | 0.798 s | 76 |
| Qwen3.5-9B Q5_K_M | Resident | 1 | 8 | 349.9 | 0.163 s | 2.950 s | 164 |
| Qwen3.5-9B Q5_K_M | Resident | 1 | 32 | 409.6 | 0.183 s | 9.558 s | 192 |
| Qwen3.5-9B Q5_K_M | Resident, MTP 1 | 1 | 1 | 221.9 | 0.080 s | 0.574 s | 104 |
| TinyLlama-1.1B Q5_K_M | Resident | 1 | 1 | 708.3 | 0.042 s | 0.181 s | 332 |
| TinyLlama-1.1B Q5_K_M | Resident | 1 | 8 | 2,474.7 | 0.041 s | 0.414 s | 1,160 |
| TinyLlama-1.1B Q5_K_M | Resident | 1 | 32 | 3,357.9 | 0.041 s | 1.220 s | 1,574 |
| Qwen3.8-Flash-Next UD-Q4_K_XL | Expert offload to RAM | 1 | 1 | 29.2 | 1.043 s | 4.421 s | 41 |
| Qwen3.8-Flash-Next UD-Q4_K_XL | Expert offload to RAM | 1 | 16 | 65.4 | 2.132 s | 31.240 s | 92 |
| Qwen3.8-Flash-Next UD-Q4_K_XL | Pipeline | 6 | 1 | 49.8 | 0.660 s | 2.571 s | 35 |
| Qwen3.8-Flash-Next UD-Q4_K_XL | Pipeline | 6 | 16 | 204.8 | 2.374 s | 9.644 s | 144 |
| GLM-5.3-Flash UD-Q4_K_XL | Expert offload to RAM | 1 | 1 | 11.4 | 3.968 s | 11.466 s | 16 |
| GLM-5.3-Flash UD-Q4_K_XL | Expert offload to RAM | 1 | 16 | 19.9 | 7.947 s | 109.932 s | 28 |
| GLM-5.3-Flash UD-Q4_K_XL | Pipeline + expert offload | 6 | 1 | 17.1 | 2.903 s | 7.446 s | 12 |
| GLM-5.3-Flash UD-Q4_K_XL | Pipeline + expert offload | 6 | 16 | 47.6 | 8.163 s | 42.396 s | 67 |

Qwen3.5-9B and TinyLlama baseline rows use `--max-num-seqs 32`, `--max-model-len 2048`,
`--max-num-batched-tokens 2048`, and `--kv-capacity auto`. Qwen3.5-9B ran on GPU 7;
TinyLlama ran on GPU 5. The MTP row uses one active sequence and
`--spec mtp --draft-tokens 1`, also on GPU 7. Cache precision is `auto`: FP8 for Qwen
and GLM, and BF16 for TinyLlama. The small resident models were measured on separate GPUs in parallel.

One-GPU offloaded MoEs ran separately on GPU 0 with
`--host-moe-layers all --cpu-moe-share auto --max-num-seqs 16` and a 2,048-token prompt
chunk. Their context limit is 2,048 tokens.
Flash-Next uses `--kv-capacity auto`; GLM uses `--kv-capacity 11264`. The expert bank
precision is `auto`. See the report for the resolved GPU cache and host memory allocations.
Six-GPU pipelines use physical devices `0,3,4,5,6,7`, capacity for sixteen active requests,
and the same context and prompt-chunk limits. Flash-Next is GPU-resident;
GLM also uses `--host-moe-layers auto --cpu-moe-share auto` to fit. Its six-GPU run
offloads 20 layers and allocates about **92.0 GiB** of host expert memory. The one-GPU
host expert allocations are **103.2 GiB** for Flash-Next and **191.7 GiB** for GLM.

### EmbeddingGemma-300M Q8_0

GPU 6; 512 synthetic token IDs per input, 768 output dimensions. The GGUF was prepared
with the existing cached EmbeddingGemma tokenizer. Each returned vector was checked for
the expected dimension and finite values.

| Clients | Inputs/request | Vectors/s | Input tokens/s | Request latency p50 | Requests |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 20.8 | 10,641 | 48.9 ms | 1,247 |
| 1 | 8 | 225.6 | 115,507 | 35.4 ms | 1,692 |
| 4 | 1 | 83.5 | 42,752 | 48.8 ms | 5,010 |
| 4 | 8 | 245.9 | 125,884 | 99.2 ms | 1,844 |

### Other files in `models/`

| Input | Result |
|---|---|
| `harrier-oss-v1-0.6B-Q8_0.gguf` | No embedding measurement: `--embed` rejects its Qwen3 architecture. |
| `embeddinggemma-300m/` | No safetensors measurement: encoder preparation currently accepts GGUF. This local directory also lacks its tokenizer and configuration files. The GGUF variant is measured above. |
| `dummy-glm-5.3-flash/` | Test fixture; the native generation CLI rejects this configuration. |
| `Qwen3.8-Flash-Next-frontend/`, `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | Supporting resources for Flash-Next. |
| Numbered GGUF shards | Measured together as one checkpoint per model. |

### Reproduce a measurement

From the repository root, start the server and run the client in another terminal:

```bash
CUDA_VISIBLE_DEVICES=7 surogate serve models/Qwen_Qwen3.5-9B-Q5_K_M.gguf \
  --served-model-name bench --no-thinking --no-prefix-reuse \
  --max-model-len 2048 --kv-capacity auto --kv-cache-dtype auto \
  --max-num-seqs 32 --max-num-batched-tokens 2048 \
  --max-pending-requests 512 --pending-timeout-ms 600000

python -m surogate.serve.tools.bench.serve_http_bench \
  --url http://127.0.0.1:8080 --model bench --concurrency 8 \
  --prompt-tokens 512 --output-tokens 128 --warmup 15 --seconds 60
```

See the [HTTP benchmark guide](tools/bench/README.md#http-measurements-for-local-models)
for embedding batches and report definitions.

---

## Earlier comparisons — 2026-08-30 through 2026-09-08

The following tables and notes preserve earlier runs, including checkpoints no longer in
`models/`. Their hardware settings, formats, workloads, and comparison engines differ from
the fresh local-model measurements above. References to “current” below apply to those
historical dates.

Board of record, one table per model with the same columns. The narrative behind every row (superseded rows,
rejected levers, kernel profiles, the reasoning behind each lever, and the
per-model sections this table replaced on 2026-08-30) is in
`BENCHMARKS_HISTORY.md`; the dated engineering log is `design/INFERENCE.md`.

Host: 8× RTX 5090 (32 GB, driver 590.44.01), two NUMA nodes (GPUs 0–3 / 4–7),
2× EPYC 9124 (32 cores, AVX512-VNNI), 503 GB RAM. Engines: **surogate serve**
(this repo), **vLLM 0.27.1** (flashinfer 0.6.16.post3), **llama.cpp** CUDA
build (0.3.0-dev @ f1357e4; Flash-Next rows use upstream master with
`qwen4exp`, GLM-5.3 rows a build of ggml-org PR 27754 because no released llama.cpp knows that
architecture, plus `ik_llama.cpp` 7cff686d for the CPU-MoE bar).

### Method

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
surogate; int8 KV is never used on this board, because it changes the output. A model larger
than its cards may hold some weights in pinned host memory (`--host-moe-layers`,
`--gpu-layers`); where a row does, its comment says how much, because those bytes cross PCIe
on every token that reads them and the row is not comparable with a resident one. Weight
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

### Board of record (2026-08-30)

One table per model, the same columns throughout; **every cell is measured** — the derived
figures the board used to carry are gone. **The first row of each model's table is the
current reference**: measured on 2026-08-30 after the clock caps were removed, one model at a
time on an idle host, with its vLLM pair in the same batch on a second card wherever the
checkpoint is still on this machine. Rows below it measure a *different* premise — another
shape, another card count, or a configuration lever — never an earlier pass of the same one:
superseded passes were removed on 2026-08-30 rather than kept for provenance, because a table
carrying three readings of one premise is read as three results. `BENCHMARKS_HISTORY.md` keeps
them.

#### Qwen3.5-0.8B

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

#### Qwen3.5-4B

| engine | GPUs | users | prefill tok/s | decode tok/s | throughput tok/s | TTFT p50 | comments |
|---|---:|---:|---:|---:|---:|---:|---|
| **surogate** | 1 | 100 | **22,121** | **5,345** | **27,466** | **40 ms** | NVFP4 3.56 GiB, 128 lanes, chunk 2,048, `--max-model-len 2048`, 8 client shards; 2026-08-30 07:17, uncapped GPU 6; re-confirmed on the one-token routing fix binary (`fc2906fc`): 21,752 / 5,256 / 40 ms on GPU 4, 3,738 requests, 0 errors — the fix cannot fire above one token, and does not |
| vLLM | 1 | 100 | 18,543 | 4,481 | 23,023 | 0.23 s | `surogate/Qwen3.5-4B-NVFP4` (ModelOpt), `--max-num-seqs 128`, `--max-model-len 2048`; 2026-08-30 13:24, uncapped GPU 5, 3,200 requests, 0 errors. surogate **+19 % decode, +19 % prefill, 5.8x TTFT** |
| **surogate** | 1 | 100 | **48,259** | 373.8 | **48,633** | **3.61 s** | prefill-heavy 2048/16, `--max-model-len 2304`, 128 lanes, chunk 4,096, 3 client shards; 2026-08-30 19:17, GPU 2. Replaces an undated pass that read 40,677 — **+19 % on the current binary** |
| vLLM | 1 | 100 | 36,308 | 281.2 | 36,589 | 4.80 s | prefill-heavy, GPU 3, launched and probed concurrently with the row above. surogate **+33 % prefill** |
| **surogate** | 1 | 1 | **63,300 †** | **313** | — | **30 ms** | 2026-08-30 14:00, uncapped GPU 6, two passes (314.3, 312.8), fp8 KV, ~1,900-token prompt, on the decode-GEMV routing fix: the 4B's five linear shapes were unregistered NVFP4 geometries and ran a 128-row cuBLASLt tile on one row at every width; at one token they now take the decode GEMV (kernel-only tg128 213.5 → 350.8). **+26 % over vLLM's 249**, at half its TTFT |
| llama.cpp | 1 | 1 | **7,000 †** | **182** | — | **270 ms** | 2026-08-30 12:05, uncapped GPU 5, CUDA build, `unsloth/Qwen3.5-4B-GGUF` Q4_K_M (fetched for this row; our side is NVFP4). Its own prompt-eval timing is 13,000 tok/s |
| vLLM | 1 | 1 | 31,700 † | 249 | — | 60 ms | same checkpoint (`surogate/Qwen3.5-4B-NVFP4`, ModelOpt); 2026-08-30 13:27, uncapped GPU 5. Led decode by 22 % for four hours — that was our routing gap, not their kernels |

#### Qwen3.8-27B

| engine | GPUs | users | prefill tok/s | decode tok/s | throughput tok/s | TTFT p50 | comments |
|---|---:|---:|---:|---:|---:|---:|---|
| **surogate** | 1 | 100 | **5,896** | **1,321** | **7,217** | **0.17 s** | `qwen3_8_27b_nvfp4_all.sinfer` (the `nvfp4-all` profile of vLLM's own checkpoint, 17.4 GB), 128 lanes, chunk 4,096, `--max-model-len 2304 --kv-capacity auto` (104,384 KV tokens) `--max-pending-requests 512`; 2026-09-07 16:09, GPU 7, three cells on the host. Replaces the 2026-08-30 pass (5,815 / 1,302 / 170 ms) and a 15:59 pass with 55k KV tokens (5,474 / 1,227 / 1.91 s -- the graph allowance had eaten the KV; fixed in 611a2f16). **+19 % decode at 0.02× the TTFT** |
| vLLM | 1 | 100 | 4,950 | 1,109 | 6,059 | 8.09 s | `sakamakismile/Qwen3.8-27B-MTP-NVFP4`, `--max-num-seqs 128 --max-model-len 4096`; 2026-09-07 15:28, GPU 3, one other engine loading on the host. Replaces the 2026-08-30 pass (5,003 / 1,120 / 8.06 s): unchanged |
| surogate | 1 | 100 | 4,006 | 898 | 4,904 | 6.02 s | mixed 512/128 on the `nvfp4-mlp-only` artifact at 64 lanes (as below); 2026-09-07 15:53, GPU 4. 81 % of vLLM's decode at 0.74× its TTFT: 64 lanes against its 128, and FP8 attention bytes against its NVFP4 |
| surogate | 8 | 100 | 4,403 | 986 | 5,389 | 1.33 s | 8-stage pipeline, C3 + asynchronous prompt flights, `--max-model-len 2048`; 2026-08-30 08:18, uncapped. **Below the one card above** — the pipeline buys capacity, not throughput per card |
| **surogate** | 8 | 1 | **168.5** | **40.7** | **209.2** | **0.57 s** | `--kv-capacity 704`, everything resident, the latent attention served absorbed (one 512-wide key/value head per token). Ahead of llama.cpp on all three columns; 119.6 / 28.9 / 0.66 on the first pass, 164.4 / 39.7 / 0.65 with the expanded attention |
| **surogate** | 1 | 100 | 856 | **2,345** | **3,201** | **0.11 s** | decode-heavy 128/512 on the all-NVFP4 artifact, 128 lanes, `--max-model-len 2304 --max-pending-requests 512` (104k KV tokens); 2026-09-07 16:09, GPU 2. Replaces the 2026-08-30 pass (2,097 / 13.7 s at 64 lanes) and the 15:57 pass with 55k KV tokens (2,236 / 1.12 s). **+39 % decode at 0.005× the TTFT** |
| vLLM | 1 | 100 | 616 | 1,688 | 2,304 | 21.5 s | decode-heavy 128/512, `--max-num-seqs 128 --max-model-len 1024`; 2026-09-07 15:20, GPU 3. Replaces the 2026-08-30 pass (608 / 1,662 / 21.8 s): unchanged |
| **surogate** | 1 | 100 | 691 | **1,892** | **2,583** | **3.24 s** | decode-heavy 128/512 on `qwen3_8_27b_nvfp4.sinfer` (`nvfp4-mlp-only`: FP8 attention/GDN/vocabulary + NVFP4 MLPs, 21.5 GB) at **64 lanes** -- 128 want 14.2 GB of runtime on this artifact (the GDN state per lane, not the chunk) and the card has 10.8 GiB left; `--max-model-len 2304 --kv-capacity auto --max-pending-requests 512`; 2026-09-07 15:51, GPU 7, same session as the vLLM row above. **+12 % decode at 0.15× the TTFT** |
| surogate | 1 | 100 | 12,006 | 91.2 | 12,097 | 14.80 s | prefill-heavy 2048/16 on the all-NVFP4 artifact, chunk 4,096, 128 lanes, 104k KV tokens, on defaults: the SwiGLU fused with the down projection's FP4 quantiser (ce9f7d76) and the wide projections on the CUTLASS 256x128x128 block-scaled kernel compiled whole-program (ef38f1ad); 2026-09-07 18:24, GPU 6, **simultaneous with the vLLM row below**. Replaces the 17:11 pair (11,699 vs 12,688, 92 %) and the 16:12 pair before either change (11,278 vs 12,507, 90 %); 08-30 read 11,557 / 15.38 s alone. **97 % of vLLM.** The 10 % was one kernel: at equal clocks every kernel class was at parity except the wide-N GEMMs, where vLLM's CUTLASS 256x128x128 cooperative kernel did a gate/up launch in 532 us to cuBLASLt's 905 -- and the identical CUTLASS kernel in our tree ran at a fifth of cuBLASLt's speed because `-rdc=true` (relocatable device code, from CUDA_SEPARABLE_COMPILATION on the ops library) makes ptxas spill its main loop (~900 bytes a thread, 150 local-memory instructions between the first and last MMA, 4.9 M local loads a launch); built whole-program the loop is clean and the tile wins. Ruled out with a measurement each: template parameters (token-identical to flashinfer's), CUTLASS 4.4.2/4.5.0/4.6.1, nvcc 12.9/13.0/13.1, C++17, fast-math, the GDC define, sm_120f, fp16 output, epilogue tile, scheduler, operand roles, the prefill graph, KV room, prompt chunk, cuBLASLt's other heuristics. Every route's greedy output is byte-identical to cuBLASLt's |
| **vLLM** | 1 | 100 | **12,402** | 94.2 | **12,496** | **14.42 s** | prefill-heavy 2048/16, `--max-num-seqs 128 --max-model-len 4096`; 2026-09-07 18:24, GPU 3, simultaneous with the row above. Replaces the 17:11 pair (12,688 / 14.11 s), the 2026-08-30 pass (12,942 / 13.81 s) and a 15:26 pass alone on GPU 3 (12,451 / 14.43 s): the pair's two numbers move together with the host, which is why only a pair counts |
| surogate | 1 | 100 | 8,369 | 63.6 | 8,432 | 21.25 s | prefill-heavy 2048/16 on the `nvfp4-mlp-only` artifact at 64 lanes (as above), `SUROGATE_SERVE_ROUND_TIMING=1` (cheap segment timers); 2026-09-07 15:51, GPU 6. **67 % of vLLM.** The executor is 99 % busy in prompt rounds and a 4,096-token round reads 262 ms (15.6k tok/s inside the rounds), so it is the rounds, not the scheduling: the FP8 attention/GDN projections on the row-scaled FP8 route, and the NVFP4 MLPs |
| **surogate** | 1 | 1 | — | **107.3** | — | **0.16 s** | `qwen3_8_27b_nvfp4_all.sinfer` (the `nvfp4-all` profile of vLLM's own checkpoint, `sakamakismile/Qwen3.8-27B-MTP-NVFP4`, 17.4 GB) **with MTP** (`--spec mtp --draft-tokens 1`, `--max-num-seqs 1`); 2026-09-07 15:56, GPU 5, same session as the vLLM row below. **+50 % decode at the same TTFT** (10 ms behind) |
| surogate | 1 | 1 | — | 70.1 | — | 0.18 s | the same all-NVFP4 artifact without the head, `--max-num-seqs 1`; 2026-09-07 15:56, GPU 2 (69.0 / 0.18 s on the 128-lane configuration, GPU 4). Replaces the 2026-08-30 pass (70.8 / 170 ms): decode within 2 % of vLLM, TTFT 30 ms behind |
| **surogate** | 1 | 1 | — | **95.6** | — | 0.23 s | 2026-09-07 15:46, GPU 6, `qwen3_8_27b_nvfp4.sinfer` (the `nvfp4-mlp-only` profile: MLPs 0-55 NVFP4 from `unsloth/Qwen3.8-27B-NVFP4`, attention/GDN/MLPs 56-63/vocabulary FP8 from the same export, 21.5 GB) **with MTP** (`--spec mtp --draft-tokens 1`, `--max-num-seqs 1`). `SUROGATE_SERVE_PREFILL_QUANT=fp4` reads the same (94.6 / 0.23 s): that route serves W8 planes, and this artifact has none in the text stack. Decode **+34 % over vLLM**; TTFT 80 ms behind it |
| surogate | 1 | 1 | — | 64.1 | — | 0.21 s | the same `nvfp4-mlp-only` artifact **without the head**, `--max-num-seqs 1`; 2026-09-07 15:50, GPU 5. The head is worth +49 % on it; vLLM's all-NVFP4 weights read 12 % faster than these FP8 attention bytes at one user, which is what the all-NVFP4 rows below settle |
| **vLLM** | 1 | 1 | — | **71.6** | — | **0.15 s** | `sakamakismile/Qwen3.8-27B-MTP-NVFP4`, `--max-num-seqs 128 --max-model-len 4096`; 2026-09-07 15:25, GPU 3, same session as the row above. Replaces the 2026-08-30 pass (71.7 / 140 ms): unchanged. The one shape where vLLM leads us at one user is now TTFT only |
| **surogate** | 1 | 1 | 371 | **83.3** | — | **0.24 s** | 2026-09-04 22:03, GPU 0, `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_M served natively **with MTP** (`--spec mtp --draft-tokens 1`; drafts 2 / 3: 76.9 / 69.9), every object but two Q8_0 projections read from the file: the UD mixture's mixed-type fused parents as typed segments, the V-head-permuted GDN out_proj with the permutation on the activation, and the IQ4_XS GEMV on a byte-permute table lookup (it had been a per-lane constant read: 232 us/call, 12x Q4_K's). The 46.5 row of 22:00 that morning had 33 % of the bytes requantised to Q4G64/Q5G64 and the slow IQ4_XS kernel on the rest; PPL 5.1699 -> 5.0477 against llama.cpp's 5.0166. |
| llama.cpp | 1 | 1 | **1,610 †** | **44.8** | — | **1.18 s** | 2026-08-30 12:07, uncapped GPU 5, CUDA build, `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_M (fetched for this row; our side is all-NVFP4). Its own prompt-eval timing is 2,734 tok/s |

#### Qwen3.6-35B-A3B

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

#### Qwen3.8-Flash-Next (111 GB MoE; on one card the experts live on the host)

All surogate one-card rows: 2026-09-06, the native GGUF read in place (no artifact repack — the
Q4G32AM host bank is decoded from the file's blocks at load), **engine defaults**: no
`--expert-slots`, no `--host-expert-bank`, the pool and the host worker pool size themselves;
`--cpu-moe-share auto --kv-capacity auto --max-model-len 4096`, chunk 8,192, fp8 KV, GPU 0 —
an **x16** card (GPUs 2, 3, 5 and 7 of this host are x8 and read half the gather rate). The
expert cache these rows run on is family machinery since today (`family::ExpertCache`, shared
with GLM-5.3-Flash below); the rows moved up because the registry stopped reserving 1.5× the
card's W8 bytes for derived planes this target never makes, and the pool and the KV cache got
the room. The eight-card rows are still the 2026-08-30 `b4ee3216` pass on the repacked
artifact. The 09-04 one-card rows read 33.6 / 85.7 / 116.4 decode at 1 / 16 / 64 users, the
08-30 rows before them 32.0 / 91.1 / 110.1; the recovery from 18.8 is in `design/INFERENCE.md`.
llama.cpp rows: `-cmoe -b 4096 -ub 4096`, the flags its own community benchmarks use — the
earlier `-ot exps=CPU` rows without batch flags understated it 3.9× and are gone.

**This target could not be served at all between 3fca7ba5 and 43c74128 (2026-09-08).** Its warm-up
prefills 4,096 tokens, past the indexer's 2,048-key budget, so the QSA selection engages; `auto`
gives this hybrid an e4m3 KV cache; and the prefill dispatch refused the pair. No board shape
reaches the selection -- 512-token prompts stay inside the budget -- which is why no row here ever
showed it.

| engine | GPUs | users | prefill tok/s | decode tok/s | throughput tok/s | TTFT p50 | comments |
|---|---:|---:|---:|---:|---:|---:|---|
| **surogate** | 1 | 1 | **156** | **35.2** | **192** | **0.88 s** | defaults (2026-09-07, the per-object bank): pool 3,172 slots (15.4 GiB, sized to leave the runtime its floor -- the floor no longer charges this A16-only target for FP8/Marlin planes it never derives), unpinned host workers sized to the cores other jobs leave free, host round started and joined through stream memory operations (no host-function dispatch: +6 % decode over the event path), the CPU/PCIe split measured with both sides running against each other (189 vs 46 GB/s), min-tokens 1 so the split fires on one-token rounds. The bank holds each expert half at the narrowest width that loses nothing: the Q4_K gate/up at four bits, the Q5_1 down at six, the Q8_0 down at eight -- 105.7 GiB pinned. **+69 % decode and +68 % prefill over llama.cpp, at matching perplexity** (3.3219 against its 3.3223 -- the table below). With the Q5_1 halves held as W8 instead this read 137 / 30.8 / 0.98 s |
| surogate (2026-09-08, loaded host) | 1 | 1 | 139.7 | 31.4 | 171.1 | 0.88 s | the same flags on today's binary -- the single-token MoE decode kernels (7402c756) and the QSA/e4m3 prefill fix (43c74128) -- on GPU 4, an idle **x16** card, but a **host under another job**: load average 8-23, ~11 cores busy. This target computes half its experts on the CPU, so that is what the -11 % is: TTFT is unchanged at 0.88 s and the warm-up window read 31.7. **Not comparable with the idle-host row above**; recorded because it is the only measurement of the current kernels on this target
| surogate `--host-expert-bank q4` | 1 | 1 | 163 | 36.7 | 200 | 0.83 s | the same run with every expert half requantised to four bits: **+4 % decode for +0.69 % perplexity** (3.3451). This was the default until 2026-09-07, when the cost was first measured |
| llama.cpp | 1 | 1 | 93 | 20.8 | 114 | 2.95 s | the same GGUF, served by both engines as the file's own blocks now. surogate **+62 % decode, 3.5× the prompt rate** |
| **surogate** | 1 | 1 | — | **40.8** | — | **7.06 s** | 28k prompt into 131k context (23.1 GiB VRAM): **3,966 tok/s prompt processing**; decode is the post-28k stream rate |
| surogate (re-measured 2026-09-04) | 1 | 1 | — | 39.1 | — | 7.07 s | the same shape on today's tree with `--max-num-batched-tokens 8192`: 3,989 tok/s prompt processing, TTFT 7.07 s, 39.1 tok/s after the 28k prompt -- the row holds. On the defaults (prefill chunk 2,048, the compromise the 8-card pipeline wanted) the same request reads 11.6 s / 2,434 tok/s / 39.4: a one-card long-prompt serve should pass the flag |
| surogate `--spec mtp --draft-tokens 1` (2026-09-04) | 1 | 1 | 157 | **35.2** | — | 0.85 s | the NextN head at the shortest draft, 71.6 % accepted; drafts 2 / 3 read 34.6 / 32.6 (55 % / 45 %) against 32.6 without -- every extra column widens the expert gather faster than the acceptance pays |
| llama.cpp | 1 | 1 | — | 27.3 | — | 23.02 s | 28k prompt, 80k context: 1,216 tok/s prompt processing. surogate **3.3× ingestion, +49 % decode** |
| cafe-llama.cpp `-hmoe` | 1 | 1 | — | 18.8 / 4.3 | — | 3.55 s / 25.46 s | 512 and 28k prompts. Experts pinned in host memory, computed on the GPU over PCIe — **our architecture in their engine** (935-1,100 tok/s at 28k); kept as the like-for-like reference |
| **surogate** | 1 | 16 | **431** | **96.7** | **528** | **1.83 s** | defaults, `--max-num-seqs 16`: KV auto 65,536 tokens (2026-09-07, the per-object bank). `--host-expert-bank q4` reads 454 / 101.9 / 1.71 s here, +5 % decode for the 0.69 % perplexity above; with the Q5_1 halves held as W8 this read 373 / 83.7 / 2.03 s; the 09-04 row read 381 / 85.7 / 1.74 with a 3,004-slot pool. Run-to-run spread at 16 users is ~±8 % |
| surogate (2026-09-08, loaded host) | 1 | 16 | 343.3 | 77.1 | 420.4 | 1.84 s | same binary and the same busy host, `--max-num-seqs 16` on defaults: the pool sized itself to 3,172 slots and KV auto to 65,536 tokens, matching the row above. TTFT p50 1.84 s against its 1.83; the 20 % on throughput is the host, and the 20 s warm-up window read 93.9 against the row's 96.7. **Not comparable with the idle-host row above.** An 8,192-token chunk does not fit 16 lanes on a 32 GB card -- the runtime floor takes the whole card and the engine refuses at startup, so this row is the default chunk
| llama.cpp | 1 | 16 | 253 | 56.8 | 310 | 16.17 s | `-np 16`. surogate **+51 % decode at 9× lower TTFT** |
| **surogate** | 1 | 64 | **567** | **127.3** | **694** | **3.20 s** | defaults, `--max-num-seqs 64 --max-pending-requests 512` (2026-09-07, the per-object bank): the pool sized itself to 3,172 slots and the KV cache to 160,192 tokens, where the 09-04 row (518 / 116.4 / 4.73 s) got 1,985 slots and 74,240 tokens -- the registry had reserved 1.5x the card's W8 bytes for derived planes this target never makes, and both the pool and the cache were paying for it. `--host-expert-bank q4` reads 609 / 136.7 / 1.90 s here, +7 % decode for the 0.69 % perplexity above; with the Q5_1 halves held as W8 this read 533 / 119.8 / 5.15 s |
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

**Accuracy, and what the host expert bank costs (2026-09-07).** Wikitext-2 test, the first 40
windows of 2,048 tokens, the same windows for both engines; llama.cpp is the `study/llama.cpp-glm`
build on eight cards, ours is one card with every expert in the host bank and the CPU split on.
These are the first accuracy numbers this target's bank has ever had.

| what the pinned bank holds | PPL | pinned bytes | 1-user decode |
|---|---:|---:|---:|
| the file's own blocks, kept (`SUROGATE_SERVE_HOST_BANK_NATIVE=1`) | 3.3215 ± 0.0336 | 101.0 GiB | — |
| W8 planes throughout (`--host-expert-bank w8`) | 3.3215 ± 0.0336 | 148.8 GiB | — |
| **per object: Q4G32AM for the Q4_K halves, Q5G32AM for the Q5_1 halves, W8 for the Q8_0 halves** (the default) | **3.3219 ± 0.0337** | **105.7 GiB** | **35.2** |
| per object with the Q5_1 halves held as W8 (the default for the twelve hours before) | 3.3231 ± 0.0337 | 116.7 GiB | 30.8 |
| Q4G32AM throughout (`--host-expert-bank q4`) | 3.3451 ± 0.0340 | 98.6 GiB | 36.7 |
| llama.cpp (`study/llama.cpp-glm`, eight cards) | 3.3223 ± 0.0336 | — | 20.8 |

Reading it: decoding the file's blocks into the bank loses nothing at all -- W8 planes and the
blocks themselves score identically, which is the bank's decode agreeing with the gather's to
the last digit. Requantising *every* half to four bits costs **0.69 %**, because this
checkpoint stores its down experts as Q5_1 in 43 layers and Q8_0 in 5, and four bits is not
where their levels sit. Keeping each half at the narrowest width that loses nothing recovers
it: with the Q5_1 halves as six-bit planes by an exact repack the bank reads 3.3219, a
hundredth of the error bar from llama.cpp, for 7 % more pinned bytes than the four-bit bank
(the Q8_0 halves are already exactly W8 and cost nothing). The four-bit bank remains one flag
away for a run that wants its rate and can spend the quality.

#### GLM-5.3-Flash (200 GB MoE; 181.65 GiB of weights against 256 GiB of cards)

`models/GLM-5.3-Flash-UD-Q4_K_XL-*.gguf` read in place, 2026-09-06, idle host, 512/128, one
engine at a time; the eight-card rows were re-measured after the draft head landed, on the
binary that carries it, and llama.cpp's 16 and 64-user rows with them. Ours: the
eight-stage pipeline with the latent attention served **absorbed** -- the query folded through
the key half of the expansion into the 512-wide latent, one key/value head per token in the
cache -- `--max-model-len 704 --max-num-batched-tokens 512 --kv-cache-dtype bf16`.
llama.cpp: the same file on the same eight cards, `--split-mode layer -ngl 99 -fa 1 -np N`,
built from ggml-org PR 27754 (`study/llama.cpp-glm`) — no released llama.cpp knows this
architecture.

**These rows predate 7402c756 and have not been re-measured.** That commit takes this geometry's
single-token MoE decode kernels from 107 to 83 us (gate/up) and 63.3 to 56.4 us (down) at Q4_K/Q6_K,
measured in isolation, so the decode column here is a floor rather than the current rate. The row is
an eight-card row and the two cards it would need were unavailable on 2026-09-08; 181.65 GiB of
weights does not fit six.

| engine | GPUs | users | prefill tok/s | decode tok/s | throughput tok/s | TTFT p50 | comments |
|---|---:|---:|---:|---:|---:|---:|---|
| **surogate** | 8 | 1 | **168.4** | **40.7** | **209.1** | **0.56 s** | `--kv-capacity 704`, everything resident. Ahead of llama.cpp on all three columns; it was 119.6 / 28.9 / 0.66 before the two hyper-connection kernels were fixed |
| **surogate** `--spec mtp --draft-tokens 3` | 8 | 1 | **207.5** | **50.1** | **257.6** | **0.52 s** | the checkpoint's NextN draft head on the eight-stage pipeline (2026-09-07): every stage runs the verify forward for its own layers, the last one decides, the others adopt its decision; 2.98 tokens a round accepted at 66 %. `--kv-capacity 704`, everything resident. One lane is never above `--spec-max-lanes` (default 1), so this row is the shipped default's too -- a repeat after the narrow round landed read 49.7 |
| llama.cpp | 8 | 1 | 156.6 | 37.8 | 194.4 | 1.18 s | `-np 1` |
| llama.cpp | 8 | 16 | 173.4 | 41.9 | 215.3 | 55.13 s | `-np 16`, and barely above its one-user row: the sixteen streams are served nearly one after another, so this is queueing rather than batching |
| **surogate** | 8 | 16 | **1,212.7** | **292.9** | **1,505.6** | **1.59 s** | `--kv-capacity 11264`, **everything resident**: the absorbed attention caches one 512-wide head per token, so sixteen lanes fit where the expanded form needed 10.4 GiB on a 3.5 GiB stage. 176 requests, 0 errors. **7.0× llama.cpp's decode and prefill at 1/35 of its TTFT** |
| surogate `--spec mtp --draft-tokens 3` | 8 | 16 | 679.0 | 164.0 | 843.0 | 4.29 s | the head left on at sixteen users, 80 of 80 served (2026-09-07 evening): above one lane the rounds run *narrow* -- one column per lane through the trunk, the head aligned on it and proposing nothing -- so no lane ever verifies a batch that cannot pay for it. What the head still costs here is the prompt path: under the head a prompt runs as a lone flight through the eight stages instead of riding a mixed round with seven others, and the rate is 56 % of the row above where the wide verify's was 53 %. The draft head is a latency lever, for one user; the row above is the one for sixteen |
| llama.cpp | 8 | 64 | 44.4 | 10.7 | 55.1 | 729.58 s | `-np 64` |
| **surogate** | 8 | 64 | **99.0** | **23.9** | **122.9** | **31.94 s** | `--kv-capacity auto --host-moe-layers auto`, which moved 1–2 mixture layers per stage to host memory to hold 45,056 KV tokens; those layers cross PCIe on every token and are the ceiling here. **2.2× llama.cpp's decode and prefill at 1/23 of its TTFT**; 108 of 150 requests still timed out against our 30 s admission window, where llama.cpp queues indefinitely and reports none |
| **surogate** | **1** | 1 | **53.2** | **12.9** | **66.1** | **3.27 s** | `--host-moe-layers all --cpu-moe-share auto --max-num-batched-tokens 1024` (2026-09-07, the expert cache lifted into the family, on the per-object bank): the routed experts are repacked into a 191.7 GiB pinned bank in 55 s at load -- Q4G32AM planes for the Q4_K gate/up halves, Q5G32AM for the Q5_K down halves (exact repacks both), W8 for the three Q6_K down halves, each read by its own format -- and 83 % of every round's misses are computed on the 32 host cores (host 202 GB/s against 46 over PCIe, measured overlapped) with the rest gathered into a 456-slot pool. The server times each request at 18.8 tok/s of decode after a 3.27 s prompt; the prompt is one gather of every expert of every layer over the link. **+26 % decode, +25 % prefill over llama.cpp at 58 % of its TTFT**, at parity on the perplexity gate below. With the Q5_K halves held as W8 instead (220.5 GiB) it read 51.3 / 12.4 / 3.62 s; the all-W8 bank 40.9 / 9.9 / 4.64 s; chunk 512 on that bank 36.7 / 8.9 / 6.12 s; the row before the cache 7.5 / 1.8 / 29.54 s |
| llama.cpp | **1** | 1 | 42.4 | 10.2 | 52.6 | 5.64 s | `-cmoe -t 32`: the same 172 GB off the card, the expert matmuls on 32 EPYC cores, the bytes from ordinary RAM. On the morning's binary, which streamed the experts over PCIe, it was ahead 5.7× with a fifth of our TTFT; by evening it is behind on all three columns |
| **surogate** | **1** | 16 | **84.8** | **20.5** | **105.3** | **25.78 s** | the same cache on the per-object bank, `--max-num-seqs 16 --kv-capacity 11264 --max-num-batched-tokens 2048` (2026-09-07): a 456-slot pool sized automatically, 83 % of every round's misses on the host, **16 of 16 served, 0 expired**. **+48 % decode and +48 % prefill over llama.cpp at 28 % of its TTFT.** A 16-lane round routes to most of a layer's experts, so the split carries ~8 GB of bank bytes a round; the prompt rounds, which gather every expert over the link, are what queue, and a 2,048-token chunk lets several prompts share one gather. With the Q5_K halves held as W8 this read 80.3 / 19.4 / 28.50 s; on the all-W8 bank 66.4 / 16.0 / 31.82 s with 1 of 17 expired; at chunk 512 on that bank 57.4 / 13.9 / 38.23 s with 4 of 20 expired, a tie. The row before the cache read 7.4 / 1.8 / 9.2 / 91.35 s with 22 of 28 expired |
| surogate `--spec mtp --draft-tokens 3` | **1** | 16 | 72.3 | 17.5 | 89.8 | 28.92 s | the head at sixteen users on one card (2026-09-07 11:12, card 4; its no-head pair the same hour, same settings, read 84.4 / 20.4 / 23.08 s): a prompt rides the mixed rounds under the head now, the head aligned over its columns and on every decode column, so the head costs 14 % of the decode here where a lone-prefill prompt path would have cost the row. `--max-num-seqs 16 --kv-capacity 11264 --max-num-batched-tokens 1024 --host-moe-layers all --cpu-moe-share auto`, 16 of 16 served |
| llama.cpp | **1** | 16 | 57.4 | 13.9 | 71.3 | 90.73 s | `-cmoe -t 32 --kv-unified`, 16 of 16 served |

**The single-card rows, and what changed between morning and afternoon.** Both engines keep
the same 172 GB of experts off the card. In the morning we pinned them and let the GPU read
them over PCIe, and the arithmetic set the row: eight routed experts over 42 mixture layers,
three matrices of 4096×2048 each at Q4_K, is **4.76 GB of weights per token**, which at the
~15 GB/s an x16 link sustains for a gather is 317 ms a token — 3.2 tok/s predicted, 3.1
measured, and llama.cpp, computing the same experts on 32 EPYC cores from ordinary RAM, was
ahead by six. We already had that path: Flash-Next's rows above run on a device slot pool
plus a CPU expert split that computes most of every round's misses on the host. It was
Flash-Next's alone. By the afternoon it was family machinery (`family::ExpertCache`) and GLM
was bound to it, with its experts decoded into W8 planes at load so the host reads them at
~265 GB/s instead of decoding Q4_K blocks a row at a time at 3. That is the 1.8 → 8.9 tok/s
on this row and 1.8 → 13.9 at sixteen users, within 15 % of llama.cpp at one user and level
at sixteen, at parity on the perplexity gate below. The rest of the one-user gap is measured,
not guessed: the decode round waits 35 of its 68 ms on the host round, and the host round is
bytes — 26.7 MB an expert as W8, where the file's gate and up experts are Q4_K and only its
down experts are Q5_K/Q6_K. A mixed bank (Q4G32AM planes for gate and up, W8 for down) is 27 %
fewer host bytes at no accuracy cost, and the next row.

**Reading the eight-card rows.** llama.cpp's 16-user decode is close to its one-user decode, so
it serves sixteen streams nearly one after another and its rate barely grows with users; ours
grows with lanes,
and the only question was whether the lanes fit. They did not, at first: the expanded latent
attention cached 64 heads of 256 for every token -- 902 KB a token, measured at three lane
counts and exactly linear -- and sixteen lanes needed 10.4 GiB on a stage that had 3.5. Served
absorbed the cache is one 512-wide head per token, 64× less, and sixteen lanes sit resident at
a 2.41 GiB runtime reservation. Sixty-four lanes still want 9.5 GiB against 3, so that row buys
them by moving one or two mixture layers per stage to host memory, and those layers cross PCIe
on every token: it leads llama.cpp on every column and is a fifth of the sixteen-lane rate.
What scales with the lane count now is the graph allowance (48 MiB a lane) and the delta
recurrence's state slots (2 MiB per layer per lane), not the cache.

Two things the absorbed form needed from the kernels, both found by the attention conformance
test once its geometry list carried the shape: the prompt kernel indexed the block table by
key-block index, which was the page index only while a block was 64 keys (a 512-wide head takes
a 16-key tile to stay inside the card's opt-in shared memory); and the small-T dispatch launched
width one with a 32-row tile where a group of 64 needs 64, returning silently -- zeros and a
cache row never appended. Neither had a registered shape to show on.

**Accuracy.** Wikitext-2 test, the first 40 windows of 2,048 tokens, the same windows for both:

| engine | PPL |
|---|---:|
| surogate, 8 stages, BF16 KV, eager, absorbed attention | 2.8574 ± 0.0264 |
| surogate, the same with the attention expanded | 2.8548 ± 0.0263 |
| surogate, **one card**, every expert in the W8 host bank, half of each prompt round's expert jobs computed by the host kernels (`--host-moe-layers all --cpu-moe-share auto --cpu-moe-prefill-share 0.5`, 2026-09-06) | 2.8600 ± 0.0264 |
| surogate, one card, the same bank with every prompt-round miss gathered into the device pool instead (`--host-moe-layers all --cpu-moe-share auto`) | 2.8567 ± 0.0263 |
| surogate, **one card, the per-object bank** (the default: Q4G32AM planes for the Q4_K gate/up halves, Q5G32AM for the Q5_K down halves, both by exact repack; W8 for the three Q6_K down halves), default route | 2.8586 ± 0.0263 |
| surogate, one card, the same with the Q5_K halves held as W8 instead (the 2026-09-06 default) | 2.8614 ± 0.0264 |
| surogate, the same mixed bank when its Q4 planes were reached through a W8 row and an affine refit (the first cut; replaced) | 2.8695 ± 0.0265 |
| llama.cpp (PR 27754) | 2.8541 ± 0.0263 |

Parity, 40,880 scored positions; the absorbed and expanded forms differ by the rounding of
the folded sqrt(2) and a 16-key tile order, well inside the error bar. The one-card rows are
the expert cache's accuracy gate: the routed experts repacked into planes at load, read by the
host's AVX-512 kernels with the SwiGLU clamp for most of every decode round (and half of every
prompt round in the host-scored row) and by the device pool for the rest. The W8 bank lands
where the all-GPU eight-stage number does, and the per-object bank the board rows run on
lands 0.07 % above it -- with its Q5_K halves as six-bit planes by an exact repack, where
holding them as W8 (a requantisation to amax/127) had read 0.16 % above. The first cut of the mixed bank read 0.45 % above, a systematic loss: it reached
Q4G32AM through a W8 row and an affine refit, and an int8 grid is not where a Q4_K sub-block's
sixteen levels sit. The repack that replaced it is exact to FP16 rounding of the endpoints
(0.035 % of the range on random blocks, tested). The gate script is `scratchpad/ppl_gate_glm.sh`.

#### The NextN draft head, against llama.cpp's own (2026-09-06)

Both engines serve the checkpoint's NextN block as a draft head: ours under `--spec mtp`,
llama.cpp's under `--spec-type draft-mtp` (ggml-org PR 27917, built at `study/llama.cpp-mtp`;
its architecture string is `glm5-next` where the file says `glm5next`, one line to patch).
Neither loads a second model -- both draft against the trunk's own weights. One 30-token
prompt, 160 greedy tokens, two requests, the second reported; wall is the client's clock and
the engine column is the server's own decode timing.

| placement | engine | draft | decode tok/s (wall) | engine decode | tokens / round | accepted |
|---|---|---|---:|---:|---:|---:|
| one card, experts off the GPU | llama.cpp `-cmoe -t 32` | -- | 18.2 | 19.4 | 1.00 | -- |
| | llama.cpp `-cmoe` | `draft-mtp`, 3 | **20.7** | **22.4** | 2.87 | 63.2 % |
| | surogate `--host-moe-layers all --cpu-moe-share auto`, the mixed bank (the expert cache, 2026-09-06 evening) | -- | 15.1 | 18.6 | 1.00 | -- |
| | surogate, the same | `mtp`, 3 | 18.9 | 20.4 | 2.98 | 66.0 % |
| | surogate, the W8 bank (afternoon) | -- | 12.2 | 15.1 | 1.00 | -- |
| | surogate, the same | `mtp`, 3 | 15.7 | 17.1 | **3.06** | **69.0 %** |
| | surogate `--host-moe-layers all`, before the cache | -- | 2.6 | 3.1 | 1.00 | -- |
| | surogate, the same | `mtp`, 3 | 1.3 | 1.5 | 2.94 | 64.8 % |
| eight cards, one user | llama.cpp `--split-mode layer` | -- | 50.4 | 57.4 | 1.00 | -- |
| | llama.cpp | `draft-mtp`, 3 | **63.3** | **74.7** | 2.77 | 59.4 % |
| | surogate, 8 stages | -- | 46.1 | 49.1 | 1.00 | -- |
| | surogate, 8 stages | `mtp`, 3 | **65.2** | **70.9** | 2.98 | 66.0 % |

**The head drafts as well as theirs.** 2.94-3.06 tokens a round at 65-69 % acceptance
against llama.cpp's 2.87 at 63.2 %, same block, same window, same greedy decode: the two
implementations of the same head agree on what it proposes. What differs is what a round
costs, and there speculation went opposite ways until the experts were computed on the host:
through the expert cache the head gains 10 % on our side too (18.6 → 20.4 engine decode). On
this client the one-card gap to llama.cpp's engine decode is 96 % without the head and 91 %
with it, and the wall-clock gap is wider for a reason that is not the decode: our 30-token
prompt takes 2.0 s under graphs (0.7 s eager, 0.7 s under `--spec mtp`), a round of 2..64
columns taking the decode share of the split onto the host's compute-bound grouped path. The
512-token board rows above, which are the rows of record, are unaffected and ours on every
column; the width-aware share that fixes this client's prompt is named in `design/INFERENCE.md`.

**A verify multiplies an offloaded mixture's traffic.** One token routes to 8 of 288 experts a
layer; a four-column verify routes to as many as 32 distinct ones. Where the experts are
fetched per round, that is up to 4× the bytes for the 2.94 tokens it returns, and our
single-card row loses half its rate. llama.cpp reads the same multiplied set from host RAM at
an order of magnitude more bandwidth than the PCIe link, so the round it saves outweighs the
bytes it adds and it gains 15 %. On the eight-card split, where nothing is offloaded and a
round is latency-bound, it gains 30 %.

**That last row was the one to want, and it landed on 2026-09-07.** The eight-card pipeline at
one user ran 46.1 tok/s with the GPUs 14 % busy, and it refused `--spec` on the argument that a
verify is decided on the stage holding the head while the other seven would fold their recurrent
state on a decision they never see. The argument was wrong: the fold already takes host
integers on every stage, and the verify forward was already stage-aware. What was missing was
plumbing -- each stage runs the verify for its own layers, the last stage accepts and proposes,
and its decision (the licensed tokens and the next drafts, per lane) is handed to the other
stages as bytes before every stage folds on the same integers -- plus the four things the
plumbing turned up on the way (`design/INFERENCE.md`, 2026-09-07). With the head the pipeline
reads **65.2 tok/s wall, 70.9 engine** against llama.cpp's 63.3 / 74.7 on this client, a 41 %
gain over its own 46.1, at the same 2.98 tokens a round the single card drafts. On the board's
own client the one-user row goes 40.7 → 50.1 decode.

**Where the head stops paying, and what the engine does about it (2026-09-07 evening).** The
board client, eight cards, with and without the head, the head verifying at every width:

| users | no head | verify at every width | verify at one lane, narrow above (the default now) |
|---:|---:|---:|---:|
| 1 | 40.7 | **50.1** | **49.7** |
| 2 | 67.3 | 68.5 | 66.7 |
| 4 | 114.0 | 91.7 | 100.7 |
| 8 | 215.3 | 141.0 | 136.8 |
| 16 | 292.9 | 156.5 | 164.0 |

A verify puts four columns per lane through every mixture layer and each column routes to its
own experts, so on a placement bound by expert bytes the head pays at one lane, breaks even at
two on these salted prompts, and loses from four up. `--spec-max-lanes` (default 1) is the
width above which an MTP round runs narrow: one column per lane through the trunk, sampled as
an ordinary round samples, the head aligned on it so its cache stays current, no proposals, the
recurrent state updated in place with nothing to fold. It is never the wide verify's loss, and
what remained between it and the no-head row at eight and sixteen users was the prompt path --
a draft-head prompt ran as a lone flight through the stages, where the no-head engine batches
eight prompts into one mixed round. Since 2026-09-07 11:00 a prompt rides the decode rounds
under the head too (the decode lanes narrow, the head aligned over every prompt segment, the
prompt's last token left to the final chunk); the eight-card rows above at two users and up
were measured before that and wait for the eight cards to be free again to be re-measured.

Read the two one-user numbers together rather than against each other. On this client -- a
30-token prompt, so almost no prefill -- llama.cpp's steady-state decode is ahead of ours
before it drafts at all (57.4 against 49.1), while on the cold first request our first token
arrives in 0.21 s against its 0.54 s of prompt evaluation: a pipeline pays a handoff every
round and a layer split does not. On the board's own client above, whose prompts are 512
tokens, the prefill we are much faster at carries the throughput metric and the one-user row is
ours. Neither client is wrong; they weight prefill and decode differently, and the draft head
moves only the second.

One asymmetry in that client, in llama.cpp's favour and worth naming: on the repeat of an
identical request its slot cache found 26 of the 30 prompt tokens, while we re-prefilled all
thirty. That measurement predates automatic conversation snapshots, which now retain
compatible earlier context within a bounded cache. The cold-request figures above are unaffected.

The board's own client says the same thing about the draft head on a 512-token prompt, where
prefill dominates and both engines' gains shrink toward it:

| one card, board client 512/128, 1 user | prefill tok/s | decode tok/s | tokens / round | accepted |
|---|---:|---:|---:|---:|
| llama.cpp `-cmoe -t 32` | 42.4 | 10.2 | 1.00 | -- |
| llama.cpp, `draft-mtp` 3 | 43.4 | 10.5 | 2.93 | 64.3 % |
| surogate `--host-moe-layers all --cpu-moe-share auto`, mixed bank, chunk 1,024 (the expert cache, 2026-09-06 evening) | **51.3** | **12.4** | 1.00 | -- |
| surogate, the same with `mtp` 3 (the pool sized automatically to one layer's experts) | **50.3** | **12.2** | 2.82 | 60.7 % |
| surogate, the W8 bank at chunk 512 (afternoon) | 36.1 | 8.7 | 1.00 | -- |
| surogate, the same with `mtp` 3 | 41.1 | 9.9 | 2.95 | 65.1 % |
| surogate `--host-moe-layers all`, before the cache | 7.5 | 1.8 | 1.00 | -- |
| surogate, the same with `mtp` 3 | 4.7 | 1.1 | **3.02** | **67.5 %** |

Through the expert cache both of our rows are ahead of llama.cpp's draft-head row (43.4 /
10.5) on this client, with or without the head: on the mixed bank the head no longer pays for
itself at 512/128 (12.2 against 12.4), because a four-column verify routes to up to four times
the experts and the round is bytes-bound again once the base round is fast; on the slower W8
bank it still gained 14 %. The head is worth keeping for short-prompt, long-generation
requests, where the single-request table above shows it ahead.

### Prefill on the 27B GGUF, and where it goes (2026-09-04)

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

### Accuracy gates (2026-09-04)

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

### Reading the table

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
- **The 27B's prompt processing was the one place a competitor was ahead, and it was
  one kernel.** Measured as a pair -- both engines at once, one card each, both cards at
  400 W through the pass -- vLLM served 12,507 prompt tok/s on the prefill-heavy shape
  against our 11,278 in the morning: 90 %, the same 89 % as 08-30. Every other 27B shape
  was ours on vLLM's own weights (mixed +19 % decode at 0.02x the TTFT, decode-heavy +39 %,
  one user +50 % with the head at the same TTFT). Profiling both engines at once at equal
  clocks put every kernel class at parity except the wide-N GEMMs, where vLLM's CUTLASS
  256x128x128 cooperative kernel did a gate/up launch in 532 us to cuBLASLt's 905. The
  identical CUTLASS kernel in our tree ran at a fifth of cuBLASLt's speed because it was
  compiled with `-rdc=true`; whole-program, its loop is clean and it wins. With the SwiGLU
  fused into the down projection's FP4 quantiser as well, the pair reads 12,006 against
  12,402 -- 97 %, every route byte-identical to cuBLASLt. What the day also settled: cards
  differ by +/-8 % in sustained clock at the cap and the same engine read 10,777-11,920
  across cards and host loads, so only a simultaneous pair is a number; a bench in bursts
  boosts to 2.6+ GHz and says nothing about a kernel in situ; and a kernel's SASS must be
  read for local-memory instructions inside the MMA span before its speed is compared to
  anyone's. The one-user row is no longer that finding from the other end: 0.16-0.18 s
  against 0.15 s.
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

### Open items

- **The 27B's prompt round, 3 % behind vLLM at equal clocks (2026-09-07 evening).** The
  10 % was the wide-N GEMM kernel spilling under `-rdc=true` (fixed: whole-program
  `sinfer_nvfp4_cutlass`, the 256-row tile the wide route's default) plus the unfused SwiGLU
  (fixed: `ops::linear_swiglu_down_add`). Left, each small: the GDN chunked kernels' ~0.6
  us/token, the narrow key/value slices still on cuBLASLt, and a row-scaled FP8 parent's halves,
  which are not independently addressable (`weight_row_view` admits RowSplit and Contiguous,
  and the FP8 validation insists the scale plane sit at a fixed offset from the payload), so
  an adapter on the 3.8's FP8 MLP layers is refused at bind time. The FP8 route's 27B-only
  shape table is closed: it serves any (rows, k) whose K is a whole number of 32 values
  (573e53d3), the 27B's own numbers unchanged. And a build rule worth keeping: a CUTLASS kernel under
  CUDA_SEPARABLE_COMPILATION is not the kernel flashinfer or vLLM measured -- check its SASS
  for LDL/STL between the first and last MMA before comparing.
- **Measuring at the cap.** Cards differ by +/-8 % in sustained clock at 400 W (GPUs
  2/4/5/7: 2.13/2.13/2.36/2.22 GHz under one identical GEMM) and a prompt round sits
  lower than a lone GEMM; a bench in bursts boosts to 2.6+ GHz. Pairs are two engines
  at once, one card each, with the clock sampled through the pass.

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
