# Serving benchmarks — surogate serve vs vLLM vs llama.cpp

Board of record. The full narrative (superseded rows, rejected levers, kernel
profiles, the reasoning behind each lever) is in `BENCHMARKS_HISTORY.md`.

Host: 8× RTX 5090 (32 GB, driver 590.44.01), two NUMA nodes (GPUs 0–3 / 4–7),
2× EPYC 9124 (32 cores, AVX512-VNNI). Engines: **surogate serve** (this repo),
**vLLM 0.27.1** (flashinfer 0.6.16.post3), **llama.cpp** CUDA build
(0.3.0-dev @ f1357e4; Flash-Next rows use upstream master with `qwen4exp`,
plus `ik_llama.cpp` 7cff686d for the CPU-MoE bar).

## Method

One HTTP load generator for every engine (streaming `/v1/chat/completions`,
salted prompts so nothing shares a prefix, identical token accounting). All
numbers exclude model load. 100-user rows are **90-second steady state**
(shorter windows read up to 25 % high).

| workload | shape |
|---|---|
| 100 users (board shape) | 100 closed-loop clients, ~512-token prompts, 128 out, 90 s |
| prefill-heavy | ~2k-token prompts, 16 out |
| decode-heavy | 128-token prompts, 512 out; also the 300–600 s correctness soak |
| 1 user | ~1.9k prompt / 128 out for TTFT; ~60 prompt / 512 out for decode |

Rules that keep the numbers honest:

- **A number is only comparable to its pair on the same card in the same
  batch.** The same binary measures 6,498 tok/s on GPU2 and 9,972 on GPU3
  (0.8B); the engine itself repeats to ~1 % on a given card.
- **"prefill tok/s" is prompt tokens ÷ the run's wall time**, decode phases
  included. It is a throughput share, not a prompt-processing rate. At one
  user the prompt-processing rate is prompt tokens ÷ TTFT.
- KV cache is fp8 (e4m3) by default; int8 KV is never used on this board (it
  changes output). Engine defaults: 128 lanes (64 where the model does not
  fit more), `--max-num-batched-tokens 4096` on the 27B/35B, CUDA graphs on.
- Weight pairing: llama.cpp serves GGUF Q4_K_M (~4.5 bpw), vLLM NVFP4, surogate
  the native NVFP4 artifact (4B, 27B) or the artifact repacked from the same
  GGUF (0.8B, 35B mixed Q4/Q5/Q6). All artifacts decode bit-exact against
  their source before any number is recorded.

## Board of record (2026-08-28)

Every engine on its own card, all running at once, 100 users, 512/128, 90 s,
fp8 KV. The 27B and 35B rows are means of two passes with the cards rotated.

| model | weights | surogate decode tok/s | vLLM decode tok/s | ratio | surogate TTFT p50 | vLLM TTFT p50 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | from GGUF Q4_K_M | **10,095** | 6,694 | **+51 %** | **45 ms** | 658 ms |
| Qwen3.5-4B | NVFP4 3.56 GiB | **4,942** | 4,246 | **+16 %** | **45 ms** | 239 ms |
| Qwen3.8-27B | NVFP4 all, 128 lanes | **1,330** | 1,039 | **+28 %** | **170 ms** | 8.26 s |
| Qwen3.6-35B-A3B | Q4/Q5/Q6 MoE, prefill batch 4 | 1,942 | **2,290** | −15 % | **253 ms** | 1.26 s |

Three of four ahead on throughput, all four on TTFT by 5–48×.

Single user (2026-08-26, GPU2, bf16 KV; TTFT at a 1.9k prompt):

| model | surogate TTFT / decode | llama-server TTFT / decode | vLLM TTFT / decode |
|---|---:|---:|---:|
| Qwen3.5-0.8B | **48 ms / 503** | 168 ms / 391 | 55 ms / 364 |
| Qwen3.5-4B | **57 ms / 214** | 445 ms / 190 | 71 ms / 166 |
| Qwen3.8-27B | 352 ms / 45 | 1,829 ms / **49** | **254 ms** / 45 |

## What moved each model, and what is next

- **Weight format is worth more than every scheduling lever on this
  hardware.** The 4B gained +54 % going from our W8G32 artifact to NVFP4
  (3,218 → 4,942); the 27B +37 % from the mixed NVFP4/FP8 artifact to
  all-NVFP4 (976 → 1,330). In both cases lanes, chunk width, MTP and prefill
  batching had been measured first and moved nothing (27B round model:
  28.8 ms fixed + 114 µs per column, so `throughput = 1/(6c + F/B)` is
  indifferent to batching). Only the 35B is still on a non-NVFP4 artifact.
- **Lanes: 128** (PATCHES.md #79/#80). With 100 users and 64 lanes a third of
  the load queued for a lane; that queue was the TTFT. The 27B and 35B fit
  128 lanes only after rewrite checkpoints went off by default (#73) and, on
  the 27B, the all-NVFP4 weights.
- **35B-A3B (the one behind, 85 % of vLLM).** Prefill batching pays here
  because routed experts stream once per round (per-token MoE cost falls
  from 1.26 to 0.64 µs between 640 and 2,688 columns): 1,762 → 1,942. The
  gap is uniform across shapes (85–89 %) and not bytes (we stream 7 % less
  than vLLM), so it is per-round kernel efficiency: at decode width our
  routed kernel leaves three of four warps idle. Measured and rejected (do
  not re-run): vLLM's MoE Marlin (faster below ~1,100 columns, slower above,
  one-way in-place adoption, 10× the quantisation error), a 3-stage cp.async
  pipeline, register-decoded fragments, the wide-plan threshold, the
  persistent-grid width, MTP (does not fit), bf16 KV, chunk width. Next: an
  NVFP4 expert artifact (`RedHatAI/Qwen3.6-35B-A3B-NVFP4`) and a
  row-parallel decode-width routed kernel (~2,340 by the round model).
- **27B, remaining.** 114 µs per column is 474 TFLOP/s against the 651–827
  the GEMMs measure; ~18 % of a captured window is gaps between kernels and
  the GDN chunked scan moves 145 MB per layer at 641 GB/s. Fusing the layer
  loop and widening that scan is the next 27B work.
- **fp8 KV** costs a few percent of decode and doubles the cache (27B:
  46k → 92k tokens at 48 lanes). Rewrite checkpoints off by default freed
  3.4 GB on the 27B (206,976 KV tokens at 48 lanes) and made 64+ lanes
  possible.
- **Correctness soak**: after the two mixed-round fixes (#71, #72) the 27B
  and 4B decode-heavy soaks are clean (1,060 and 4,097 requests, 0 corrupted);
  the 0.8B keeps a ~4-per-100,000 mixed-round corruption (open).

## Qwen3.8-Flash-Next — 111 GB MoE on one 5090 with CPU offload (2026-08-28)

512-expert MoE with hyper-connections and an n-gram memory; the experts
(W8 host bank, 154 GB pinned) do not fit the card. 512/128; 90 s at 16+
users, 60 s at one; GPU 1. llama.cpp / ik_llama.cpp serve the same GGUF with
experts on the CPU (`-ot exps=CPU`, 32 threads).

| config | users | decode tok/s | prefill tok/s | TTFT p50 |
|---|---:|---:|---:|---:|
| llama.cpp, 8× 5090 `--split-mode layer`, all resident | 1 / 16 / 32 / 64 | 39.3 / 39.1 / 28.8 / 24.7 | 157 / 156 / 115 / 99 | 0.95 s / 86 s / 132 s / 311 s (16 of 48 requests timed out at 16 users) |
| llama.cpp, 1× 5090, experts on CPU | 1 / 16 | 7.1 / 16.3 | 29 / 65 | 2.0 s / 29 s |
| ik_llama.cpp, 1× 5090, experts on CPU (AVX-512 iqk kernels) | 1 / 16 | 21.8 / 23.9 | 87 / 96 | 1.8 s / 30 s |
| surogate v0: experts zero-copy from pinned host | 1 / 16 | 5.2 / 7.6 | 21 / 30 | 1.8 s / 15.1 s |
| + expert slot cache (`--expert-slots 3000`, 14.6 GiB pool, bulk gather of misses) | 1 / 16 | 18.5 / 9.4 | 74 / 38 | 3.0 s / 17.9 s |
| + CPU expert split (`--cpu-moe-share 0.7`, host round overlapped, `numactl --interleave=all`) | 1 / 16 | 18.4 / **33.5** | 74 / 138 | 2.95 s / 12.3 s |
| same, `--cpu-moe-share auto` (measured host 199 GB/s vs PCIe 52 GB/s → 79 %) | 16 | 32.3 | 131 | 8.6 s |
| same, `--expert-slots 2000` (so 64 lanes fit) | 64 | 37.0 | 174 | 24.3 s |
| + prefill on the host (`--cpu-moe-prefill-share 0.7`, batched VNNI kernel) | 1 | 22.5 | 109 | **1.43 s** |
| same, prefill share 0.5 (explicit) | 16 | **37.9** | 172 | 7.3 s |
| **phase-2 defaults** (`--expert-slots 3000 --cpu-moe-share auto`: measured 80 % decode / 50 % prefill on the host) | 16 | 32.2 | 174 | 9.0 s |
| same, 1 user (prefill share 0.7) | 1 | 22.4 | 110 | **1.40 s** |

Prompt processing at one user (512 ÷ TTFT): full gather 174 t/s, prefill
split **358-366 t/s**; ik_llama.cpp ≈ 285. Both engines answer the probes
correctly (`Paris`, `2, 3, 5`).

External single-user references (llama.cpp PR #27742, other hardware and
quantisations; not run here):

| config | decode tok/s | prompt processing t/s |
|---|---:|---:|
| RTX 4090 + DDR4, UD-Q4_K_XL, `-cmoe -ub 4096`, 28k prompt | 20.8–22.5 | 356–384 |
| RTX 5090 + DDR5, UD-Q2_K_XL | 33–34 (26 @131k) | ~300 |
| RTX 5090, UD-Q3_K_XL, 18 layers' experts resident, KV q8_0, 4–5k prompts | 22.1–22.8 | ~765 |

Reading: single-user decode is bytes per expert over the host path (our W8
bank is ~1.5× the bytes of Q4_K_XL), so 18–22 is where a W8 bank lands — a
Q4-class host bank is the single-user decode lever. The prefill split now
matches the 4090 reference at 512-token prompts; the 765 figure is a
4–5k-prompt regime (4× the per-expert reuse) that needs Flash-Next's sparse
attention indexer beyond 2,051 tokens, not yet implemented. The 16-user
number on this shape is prefill-dominated, so the prefill split is the lever
at every concurrency: at 16 users (prefill share 0.5) it halves TTFT and adds
7 % decode; the share is lower than at one user (0.7) because at concurrency
the host must stay off the critical path of the mixed rounds.

## Open items

- **35B-A3B**: NVFP4 expert artifact; row-parallel decode-width routed kernel.
- **27B**: layer-loop fusion and a wider GDN chunked scan (the 114 µs column).
- **Flash-Next**: prefill split at 16–64 users; host tile kernel (interleaved
  rows); sparse indexer for >2k context; Q4 host bank; then pipeline/expert
  parallelism across cards (phases 3–4 in `design/INFERENCE.md`).
- **0.8B**: ~4-per-100,000 mixed-round corruption.
- Record the card with every number; re-measure single-user cells on the
  same card as the 100-user rows.
