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

Engine configuration (rows dated 2026-08-27 pm use `--max-num-seqs 64` and
`--max-num-batched-tokens 4096` on the 27B/35B, see the chunk-width and lane
paragraphs): originally `--max-num-seqs 64` at 0.8B/4B and `48` at 27B (each
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

### Why runs must be paired, and what varies (2026-08-27)

The engine is **not** the source of the run-to-run spread — it repeats to
about 1 % on a given card. The same 4B config on all eight cards at once,
twice, measured (tok/s):

| pass | gpu0 | gpu1 | gpu2 | gpu3 | gpu4 | gpu5 | gpu6 | gpu7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2,325 | 3,068 | 2,094 | 3,044 | 2,446 | 3,178 | 3,109 | 3,093 |
| 2 | 2,316 | 3,055 | 2,086 | 3,042 | 2,448 | 3,146 | 3,030 | 3,071 |

Each card reproduces itself within 0.1–2.5 %, but gpu0/2/4 sit 25–33 % below
gpu1/3/5/6/7 — persistently, under identical config at the same moment. That
split follows neither PCIe width (gpu0/1/4/6 are x16, the rest x8) nor NUMA
node (gpu0–3 hang off node 0, gpu4–7 off node 1), and the host is far from
saturated (load average 5–7 of 64 cores, 38 of 503 GB, 466 W total across
eight cards at 55 °C with no throttle flags). Whether it is intrinsic to
those cards or an artifact of eight simultaneous launches is under test.

The consequence for this board: **a number is only meaningful next to its
pair on the same card in the same batch.** Cross-batch comparisons of a lone
figure — which is how the morning's 4B row was read — can be off by a third
for reasons that have nothing to do with the code.

## Board of record (2026-08-28)

Every engine on its own card, all running at once, 100 users, 90 s,
512-token prompts, 128 output tokens, fp8 KV. The 27B row is the mean of two
passes with the cards rotated between them.

| model | weights | engine decode tok/s | vLLM decode tok/s | ratio | engine TTFT p50 | vLLM TTFT p50 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | from GGUF Q4_K_M | **10,095** | 6,694 | **+51 %** | **45 ms** | 658 ms |
| Qwen3.5-4B | NVFP4 3.56 GiB | **4,942** | 4,246 | **+16 %** | **45 ms** | 239 ms |
| Qwen3.8-27B | NVFP4 all, 128 lanes | **1,330** | 1,039 | **+28 %** | **170 ms** | 8.26 s |
| Qwen3.6-35B-A3B | Q4/Q5/Q6 MoE | 1,942 | **2,290** | **−15 %** | **253 ms** | 1.26 s |

Three of four beaten on throughput, all four on TTFT by 9–48x. The 27B row is
the all-NVFP4 artifact (#87): the round-cost section below rules out batching,
lanes, speculation and the chunk ladder by measurement, and the weight format
is what moved it. The 35B's remaining gap is that same lever, untaken.

## Verdict — paired, after the lane cap and multi-prompt rounds (2026-08-27 late)

Two changes land together here: the eight cards were found to be running at
different clock limits (three had been capped for heat) and were reset to
parity, and the engine's concurrency cap moved from 64 to 128 lanes with the
mixed round able to prefill several prompts at once (PATCHES.md #80). With
100 users and only 64 lanes, a third of the load waited for a lane — that
queue was the engine's TTFT.

| model | lanes | engine decode tok/s | vLLM decode tok/s | engine TTFT p50 | vLLM TTFT p50 |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 128 | **9,758** | 6,694 | **45 ms** | 658 ms |
| Qwen3.5-4B | 128 | 3,208 | **4,238** | **71 ms** | 243 ms |

The 0.8B now leads by **46 %** on throughput and 14× on TTFT. The 4B leads
3.4× on TTFT but still trails 24 % on decode throughput, and that gap is
weight bandwidth, not scheduling: our 4B artifact is 5.13 GiB of W8G32
against vLLM's ~2.2 GiB NVFP4 export, and the kernel profile puts the Marlin
W8 GEMMs at 52 % of decode time. An NVFP4 4B artifact is the lever there.

## 4B on NVFP4 weights (2026-08-27 late, PATCHES.md #83/#84)

The 4B artifact was 5.26 GiB of W8G32 against vLLM's NVFP4 export, and the
kernel profile put the Marlin W8 GEMMs at 52 % of decode time. It now
converts from `AxionML/Qwen3.5-4B-NVFP4` to a native NVFP4 artifact of
3.56 GiB. Getting there needed the fused ops to accept shapes outside the
27B's registered geometry (#84): the in-house W4A4 ladders stay
registered-only and every other shape runs on cuBLASLt, which is generic in
n and k.

Three engines, one card each, at the same time, 100 users, 90 s, 512/128,
128 lanes, chunk 2,048:

| engine | weights | decode tok/s | prefill tok/s | TTFT p50 | TTFT p95 | reqs ok/err |
|---|---|---:|---:|---:|---:|---:|
| **surogate serve** | NVFP4 3.56 GiB | **4,942** | **19,767** | **45 ms** | **51 ms** | 3,520/0 |
| surogate serve | W8G32 5.26 GiB | 3,218 | 12,870 | 70 ms | 76 ms | 2,300/0 |
| vLLM | NVFP4 | 4,246 | 16,984 | 239 ms | 304 ms | 3,000/0 |

**+16.4 % on decode and on prefill, at 5.3× better TTFT.** The weight
format was worth +54 % over our own W8 artifact — bigger than the whole gap
to vLLM, which is what the 52 % Marlin share predicted.

Two conversion traps cost more time than the kernels did, and both were
found by decoding the written artifact and comparing it against the
checkpoint rather than by looking at output text:

  - ModelOpt writes `weight_scale_2` / `input_scale` as **multipliers**
    (`amax/(448*6)`); the runtime's fields are **divisors** (the
    compressed-tensors convention the 27B recipe reads, block scale =
    `divisor * max_abs / 6`, undone by `alpha = 1/(input_div * weight_div)`).
    Passed through unchanged, the model emitted fluent noise.
  - the convolution ships channel-major `(8192,1,4)` and the artifact stores
    it tap-major `(4,8192)`. A reshape silently reinterprets those bytes.

Both artifacts decode bit-exact against the source now. The lesson for the
next target: verify a converted artifact numerically against its checkpoint
before reading anything into what the served model says.

The 27B does not fit 128 lanes on 32 GB — it refuses at startup, asking for
12.9 GiB of runtime reservation beyond its headroom — so it stays at 64 and
is measured separately below.

Lane count for the same 4B config, one card, nothing else running:

| lanes | decode tok/s | TTFT p50 |
|---:|---:|---:|
| 64 | 3,195 | 1,728 ms |
| 100 | 3,262 | 69 ms |
| 128 | 3,269 | 69 ms |

## Verdict — paired, two passes (2026-08-27 evening, 64 lanes)

The measurement that decides the goal: vLLM and the engine on **one card,
back to back, twice**, with all three models running as one batch so every
pair sees the same background. 100 users, 90 s, 512-token prompts, 128
output tokens. The two passes agree to within 0.5 % on every cell, so these
supersede any conflicting row below.

| model | card | engine decode tok/s | vLLM decode tok/s | ratio | engine TTFT p50 | vLLM TTFT p50 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | GPU1 | **8,916 / 8,922** | 6,632 / 6,665 | **+34 %** | **523 ms** | 712 ms |
| Qwen3.5-4B | GPU3 | 3,061 / 3,059 | **4,215 / 4,247** | **−28 %** | 1,515 ms | **250 ms** |
| Qwen3.8-27B | GPU5 | 940 / 943 | **1,051 / 1,044** | **−10 %** | **4,863 ms** | 8,119 ms |

So: the 0.8B is beaten comfortably, the 27B is close on decode while leading
on TTFT, and **the 4B is behind** — the morning's +11 % row was not
reproducible. The 4B's TTFT is the striking part: 1.5 s against vLLM's
0.25 s, six times worse, while its steady state holds 64 lanes and 13,400
prefill tok/s inside the run. That points at admission and prefill
scheduling for short prompts rather than at kernels, and is where the 4B
work should start.

A caveat that is not yet resolved: runs of the *same* 4B config in different
batches have measured 2,326 and 3,061 tok/s, each internally flat for its
whole 90 s and with byte-identical startup logs. Within a batch the engine
repeats to 0.5 %; across batches it does not, and neither the cards nor the
host explains it (see the method note above). An 8-card identical-config
test is running to separate card identity from launch-to-launch behaviour.

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
| Qwen3.8-27B | GPU3 | surogate serve | 833 | 3,332 | 7.87 s | 655/0 |
| | GPU3 | **vLLM** | **1,040** | **4,158** | 8.2 s | 814/0 |
| Qwen3.8-27B, decode-heavy 128/512 | GPU5 | **surogate serve** @48 | **1,633** | 408 | **14.4 s** | 388/0 |
| | GPU5 | **surogate serve** @64 | **1,884** | 471 | 14.2 s | 534/0 |
| | GPU5 | vLLM | 1,438 | 360 | 21.9 s | 356/0 |
| Qwen3.8-27B @64 lanes | GPU6 | surogate serve | 897 | 3,586 | **5.1 s** | 687/0 |
| | GPU6 | **vLLM** | **1,051** | **4,203** | 8.2 s | 836/0 |
| Qwen3.8-27B @64 lanes, prefill-heavy 2048/16 | GPU6 | surogate serve | 49 | 6,280 | 30.0 s | 369/0 |
| | GPU6 | **vLLM** | 92 | **11,822** | **14.9 s** | 607/0 |
| Qwen3.8-27B @64 lanes, chunk 4,096, prefill-heavy | GPU6 | surogate serve | 52 | 6,678 | 28.4 s | 388/0 |
| | GPU6 | **vLLM** | 92 | **11,818** | **14.9 s** | 607/0 |
| Qwen3.8-27B @64 lanes, chunk 4,096, prefill-heavy, #78 | GPU6 | surogate serve | 57 | 7,339 | 25.8 s | 418/0 |
| | GPU6 | **vLLM** | 92 | **11,818** | **14.9 s** | 607/0 |
| Qwen3.8-27B @64 lanes, chunk 4,096, #78 (balanced) | GPU6 | surogate serve | 919 | 3,677 | **5.0 s** | 703/0 |
| | GPU6 | **vLLM** | **1,062** | **4,247** | 8.1 s | 836/0 |
| Qwen3.8-27B @64 lanes, chunk 4,096 (balanced) | GPU6 | surogate serve | 913 | 3,652 | **5.0 s** | 699/0 |
| | GPU6 | **vLLM** | **1,062** | **4,247** | 8.1 s | 836/0 |
| Qwen3.5-4B @64 lanes, chunk 2,048 | GPU2 | surogate serve | 2,073 | 8,292 | 2.2 s | 1,520/0 |
| | GPU2 | **vLLM** | **3,641** | **14,564** | **0.20 s** | 2,600/0 |
| Qwen3.5-4B @64 lanes, chunk 2,048, prefill-heavy | GPU2 | surogate serve | 131 | 16,788 | 11.6 s | 833/0 |
| | GPU2 | **vLLM** | 273 | **34,989** | **5.0 s** | 1,623/0 |
| Qwen3.5-0.8B @64 lanes, chunk 2,048 | GPU0 | **surogate serve** | **6,478** | **25,910** | 0.72 s | 4,623/0 |
| | GPU0 | vLLM | 6,174 | 24,700 | **0.51 s** | 4,398/0 |
| Qwen3.5-0.8B @64 lanes, chunk 2,048, prefill-heavy | GPU0 | **surogate serve** | **502** | **64,208** | **3.0 s** | 2,917/0 |
| | GPU0 | vLLM | 356 | 45,571 | 3.8 s | 2,086/0 |

† engine prefill tok/s and TTFT p50 are taken from the server's own interval
and per-request logs of the same runs (the pairing script summarised only
decode); vLLM's are the load generator's client-side figures.

Engine: **+35%** at 0.8B, **+11%** at 4B, **−21%** at 27B on decode; ahead on
prefill at 0.8B and 4B. vLLM's own numbers moved against the 08-26 board too (4B
3,390 → 3,926, 27B 688 → 1,041), so neither side of that board was measured
under today's conditions; the 08-26 100-user rows are kept below only as
history.

## Prefill — 100 users, same card, same day (2026-08-27)

Prefill-heavy shape: 2048-token prompts, 16 tokens out, 100 closed-loop
users, 90 s, vLLM then the engine on the card shown. This is the shape that
isolates prompt processing; the balanced table above shows prefill only as a
by-product of a decode-limited workload.

| model | card | engine | prefill tok/s | decode tok/s | TTFT p50 | TTFT p95 | reqs ok/err |
|---|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | GPU4 | **surogate serve** | **81,376** | 636 | **2.39 s** | 2.39 s | 3,672/0 |
| | GPU4 | vLLM | 45,106 | 352 | 3.88 s | 3.95 s | 2,065/0 |
| Qwen3.5-4B | GPU5 | **surogate serve** | **40,677** | 318 | 4.77 s | 4.78 s | 1,883/0 |
| | GPU5 | vLLM | 34,964 | 273 | 5.00 s | 5.09 s | 1,621/0 |
| Qwen3.8-27B | GPU3 | surogate serve | 6,135 | 48 | 30.0 s | 30.3 s | 360/0 |
| | GPU3 | **vLLM** | **12,084** | 94 | **14.5 s** | 14.7 s | 617/0 |

Engine prefill: **+80%** at 0.8B, **+16%** at 4B, **−51%** at 27B. The 27B's
balanced-shape deficit (−21% on decode above) is this: the engine prefills
the 27B at half vLLM's rate, so with four prompt tokens per generated token
the prefill duty cycle starves decode. Prefill is the 27B lever, not lanes
or KV.

## Where the 27B prefill time goes (2026-08-27)

The 27B is the only NVFP4-native artifact and the only model behind vLLM.
Its W4A4 GEMM launchers gated the TMA schedule on `tokens >= 1024 &&
tokens % 256 == 0`; a mixed serving round (prefill chunk + decode batch) is
never block-aligned, so every prefill GEMM ran on the mma ladder. PATCHES.md
#75 splits the launch (TMA over the 256-aligned prefix, mma over the tail).
Same card, one run per cell, 48 lanes, fp8 KV auto:

| shape | old path (TMA off) | split, floor 1024 | split, floor 256 |
|---|---:|---:|---:|
| prefill-heavy 2048/16, GPU6 (prompt tok/s) | 5,867 | 6,088 | 6,244 |
| balanced 512/128, GPU7 (decode / prompt tok/s) | 826 / 3,303 | 822 / 3,289 | 841 / 3,362 |

Only +4–6 % on the prefill-heavy shape, because the two in-house schedules
are close. The kernel-level picture (GPU2, `ninfer_gdn_input_proj_bench` and
the `ninfer_linear_nvfp4_cublaslt_test` timings, TFLOP/s):

| GEMM | tokens | in-house W4A4 | cuBLASLt block-scaled FP4 |
|---|---:|---:|---:|
| GDN in_proj 16384×5120 | 1,024 (TMA) | 560 | 763 |
| GDN in_proj 16384×5120 | 1,077 (mma) | 545 | 756 |
| GDN in_proj 16384×5120 | 2,048 (TMA) | 611 | 827 |
| MLP gate-up 34816×5120 | 1,024 | 575 | 819 |
| residual 5120×6144 | 300 | 215 | 400 |
| GDN in_proj 16384×5120 | 64 | 179 (60 µs) | 260 (41 µs) |
| GDN in_proj 16384×5120 | 16 | 76 (35.5 µs) | 35 (76 µs) |

cuBLASLt (`CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`, CUDA 13.1) consumes
the artifact's weight codes and 128×4-tiled scales in place and is
bit-exact against the in-house kernels on identical quantized inputs
(`ninfer_linear_nvfp4_cublaslt_test`); it is 1.4–1.9× faster from 64 tokens
up and slower below (the in-house small-T kernels run at the weight-bandwidth
limit). PATCHES.md #76 routes every W4A4 GEMM from 64 tokens up through cuBLASLt.
Same card, route off → on, one run each: prefill-heavy 6,139 → 6,528 prompt
tok/s (GPU3, +6.3 %); balanced 672 → 717 decode / 2,687 → 2,869 prompt tok/s
(GPU4, +6.7 %). Consistent, but a 1,024-token chunk costs ~150 ms end to end
and its GEMMs were only ~55 ms of it, so the GEMM family was never the 2×.
Op-level, the route's gain depends on the card: on GPU7 (fast) the in-house
GDN projection already runs at 904 TFLOP/s against cuBLASLt's 1,006 (+11 %),
on GPU2 (slow) it is 545 against 751 (+38 %). The eager family timer at one
user (GPU5, no host contention) puts the layer loop at 126.5 ms per
1,073-token chunk with the route (137.7 without): attention 14.6 %, MLP of
the full-attention layers 10.6 %, GDN 43.0 % (input projection 48.7 %, conv
4.2 %, chunked scan 24.1 %, norm + output projection 23.0 %), MLP of the GDN
layers 31.8 %. Summing the isolated op benches at the same shape gives only
~90 ms, so ~35 ms per chunk are per-op overheads inside the layer loop that
graph replay may or may not remove — the next measurement. Isolated costs:
the prefill attention kernel is 311 µs per layer at 1,073 tokens (488 µs on
a 2,146-token context), 91 TFLOP/s — ~5× off an FA3-class kernel but only
5–8 ms of the chunk; the chunked GDN path (`ninfer_gated_delta_net_bench`,
48 value heads, 1,024 tokens) is 287 µs per layer — state passing 47 %,
WY/WU preparation 30 %, output 23 % — moving 145 MB of intermediates per
layer at 641 GB/s. Lanes: with #75/#76 in, 64 lanes beat 48 on the balanced
shape (679 → 726 decode tok/s on GPU0, TTFT p50 9.5 → 6.2 s; prefill-heavy
flat), so the 27B configuration moves to 64 lanes once re-paired.
Prefill chunk width (`--max-num-batched-tokens`, default 2,048) on the same
card (GPU4, prefill-heavy, 48 lanes): 1,024 → 5,377, 2,048 → 5,765, 4,096 →
6,131 (+6.3 % over the default), 8,192 flat at 6,117; TTFT unchanged. The loadgen's "2,048-token" prompts are 2,146 tokens
with the chat template, so 1,024 needs three rounds, 2,048 two and 4,096
one — about 7 % per round removed, which is the per-round fixed cost
(decode-lane work and bookkeeping) that multi-prompt prefill rounds would
amortise the same way. The same shape on the other models, same card each,
1,024 → 2,048: 0.8B 58,042 → 64,150 (+10.5 %, GPU0), 4B 15,489 → 16,773
(+8.3 %, GPU2), 35B-A3B 12,458 → 13,653 (+9.6 %, GPU1); the 35B at 4,096
reaches 15,276 on GPU7. The 4B rows above contradict the earlier pair (4,359 vs 3,926 on GPU5): here
the engine's 4B throughput roughly halves while vLLM's barely moves. Both
rows are same-card pairs, so the difference is not the hardware — the GPU2
pair ran while seven other engines were driving the host, the GPU5 pair did
not. A model this small needs many rounds per second and its per-round host
work does not overlap, so host contention costs it far more than it costs
vLLM; the controlled measurement is below. Kernel by kernel under that load
(Nsight Compute, GPU2, 100 users balanced, ~6 rounds): the Marlin W8 GEMMs
52 % — the 4B artifact carries 8-bit weights where vLLM's is NVFP4, twice
the bytes per decode step — the decode attention kernel 15.6 % (190 µs per
layer for 64 lanes), the GDN recurrent step 10.5 %, top-k sampling 2.4 %. An
NVFP4 4B artifact is the structural answer; the decode attention kernel the
second.

Where a 27B prefill chunk goes, kernel by kernel (Nsight Compute, one user,
eager, 3,000 kernels ≈ 2.7 chunks of 2,146-token prompts, GPU5): the
in-house FP8 W8A8 GEMM (`fp8_mma_kernel`: GDN input projection, attention
QKV, both output projections — the 27B keeps those weights FP8-row) **52 %**;
the cuBLASLt NVFP4 GEMMs of the MLPs 25.6 %; the prefill attention kernel
4.0 %; the chunked GDN path 7.9 % (state passing 3.9, WY/WU preparation 2.4,
output 1.6); SiLU·mul 2.5 %; the NVFP4 quantizer 1.5 %; conv 1.3 %; norms
~2 %. cuBLASLt's FP8 GEMM runs the same shapes 1.7–2× faster (651–750
TFLOP/s at T ≥ 1,073 against the in-house ~330). PATCHES.md #78 routes the
FP8 families to it from 128 tokens up; same card, route off → on, 64 lanes,
chunk 4,096: prefill-heavy 6,619 → 7,339 prompt tok/s (+10.9 %, GPU6),
balanced 949 → 960 decode tok/s (GPU5, noise — decode batches stay below the
threshold). The exact fp32 staging of the finish pass eats part of the
kernel gain (~670 MB per GDN projection at T = 4,096); a bf16 staging is the
follow-up.

Per-window laps (`SUROGATE_SERVE_PREFILL_TIMING=1`, one user, 2,146-token
prompts, same card) show where a 27B prefill window actually goes: with
graphs, 319 of 320 windows replay a captured body, 116.1 ms of the 116.8 ms
stream time is inside the replay, and the host wall per window is 116.8 ms —
**no host gap at all**, so the engine is not host-bound and graph capture is
in play. Without graphs the same window costs 120.7 ms (eager layer loop
120.3). The kernel sum from Nsight over the same shape is ~95 ms, so ~20 ms
per window (≈18 %) sits between kernels inside the graph.

That reframes the 27B gap. Single-stream, the two engines are close: TTFT for
a 2,146-token prompt is 239 ms here against vLLM's 213 ms (1.12×). Under 100
users the gap is 1.60× (7,372 against 11,818 prompt tok/s). So roughly
three-quarters of the deficit is not kernel speed but concurrency handling:
vLLM batches several prompts into one prefill step, while the executor holds
a single `prefill_lane_` and runs one prompt's chunk per round. **Multi-prompt
prefill rounds are the next 27B lever**, ahead of any further GEMM work.

CUDA graphs are not a lever on this shape: same card (GPU7), 64 lanes,
chunk 4,096, 100 users, `--enforce-eager` 6,882 against graph replay 6,821
prompt tok/s. Lanes on the 27B: 64 beats 48 on every shape now
(balanced 897 vs 833-class, decode-heavy 1,884 vs 1,633). Single-user prefill
(2,048 tokens): engine 264 ms eager / 257 ms graph (GPU5) against vLLM
213 ms (GPU3), so the single-stream gap is ~1.3× and the rest of the 2×
lives in the 100-user mixed-round regime.

## 27B on all-NVFP4 weights (2026-08-28, PATCHES.md #87)

The round-cost section below ruled out batching, lanes, speculation and the
chunk ladder by measurement, and ended on "what is left is the 114 us column".
It was the weight format, exactly as at the 4B.

Our 27B artifact kept the attention and GDN projections FP8-row only because
`unsloth/Qwen3.8-27B-NVFP4` exports them that way, and Nsight put those FP8
GEMMs at 52 % of a prefill chunk running at 651-750 TFLOP/s where the NVFP4
ones reach 827. **The checkpoint vLLM is served from in every paired row on
this board — `sakamakismile/Qwen3.8-27B-MTP-NVFP4` — quantises every language
linear**, ignore list the vision tower alone. Same 64 layers, same 5120 hidden,
same 48 value heads, and it carries the bf16 embedding, lm_head, MTP block and
vision tower too, so one checkpoint supplies the whole artifact.

One thing had to be solved first. The GDN input projection fuses `in_proj_qkv`
and `in_proj_z` into a 16384-row object and an NVFP4 object carries **one**
weight divisor, but this export gives the two sources different weight global
scales (layer 0: 6176 against 11264; layer 1: 8064 against 10112). Restating
one half onto the other's divisor measured **~2 % mean relative error with
93 % of its values moving** — a second quantisation of every GDN gate weight,
which is not a trade to make for throughput. So the halves stay apart and the
projection runs as two GEMMs, which is what the cuBLASLt route does with the
fused weight anyway. Attention (q|k|v) and MLP (gate|up) share their scales and
fuse cleanly.

Weights 18.98 -> 15.16 GiB. KV then holds 1,471 pages instead of 931 and 99
sequences run instead of 84. Two passes, cards rotated, everything at once,
100 users, 512/128, 90 s:

| | pass 1 | pass 2 | mean | prefill tok/s | TTFT p50 |
|---|---:|---:|---:|---:|---:|
| **all-NVFP4, 128 lanes** | 1,357 | 1,302 | **1,330** | **5,319** | **170 ms** |
| all-NVFP4, 96 lanes | 1,317 | 1,350 | 1,334 | 5,333 | 416 ms |
| mixed NVFP4/FP8 (before) | 966 | 985 | 976 | 3,901 | 1,916 ms |
| vLLM | 1,034 | 1,044 | 1,039 | 4,156 | 8,260 ms |

**+37 % over our own mixed artifact and +28 % over vLLM, at 48x its TTFT.**
128 lanes is the configuration: decode ties 96 lanes and TTFT is 2.4x better.
Every object decodes bit-exact against the checkpoint, and the served model
answers as it did before.

The lesson is now the same at two scales: **on this hardware the weight format
is worth more than every scheduling lever put together.** The 4B gained 54 %
from it, the 27B 37 %, and in both cases the levers the round-cost model
ranked above it moved nothing.

## The 35B's round, and everything that did not move it (2026-08-28)

The 35B is the one model still behind. Its round-cost model, measured the same
way as the 27B's:

| round | ms | columns |
|---|---:|---:|
| mixed (one prompt + the decode batch) | 62.0 | 640 + 95 |
| decode | 26.0 | 95 |

**20.9 ms fixed plus 56 us per column** unbatched — and unlike the 27B, the
column cost is *not* flat. The routed experts are streamed once per round no
matter how wide it is, so per-token cost falls steeply with width
(`ninfer_sparse_moe_bench`, q4-q5, cold, one layer):

| columns | 640 | 1,152 | 2,176 | 2,688 | 3,712 |
|---|---:|---:|---:|---:|---:|
| us/token | 1.26 | 0.91 | 0.67 | 0.64 | 0.64 |

That is why **batched prefill** (#88) pays here and is a no-op on a dense
model: four prompts in one mixed round cost about what two cost separately.
Two passes, cards rotated, 100 users, 512/128, 90 s:

| | pass 1 | pass 2 | mean | TTFT p50 |
|---|---:|---:|---:|---:|
| engine, prefill batch 4 | 1,960 | 1,924 | **1,942** | 253 ms |
| engine, unbatched | 1,756 | 1,768 | 1,762 | 130 ms |
| **vLLM** | 2,278 | 2,303 | **2,290** | 1,256 ms |

With the lane cap lifted first (1,595 -> 1,762) that is **+22 % on this model
this session**, at 85 % of vLLM and 5x its TTFT.

Everything else measured and rejected, so the next reader does not re-run it:

  - **vLLM's MoE Marlin** (#89, ported in full, kept behind
    `SUROGATE_SERVE_MOE_MARLIN`). It is **faster where it matters and slower
    where it does not**, per MoE layer (q4-q5, cold):

    | columns | 99 | 128 | 192 | 320 | 704 | 1,408 | 2,688 |
    |---|---:|---:|---:|---:|---:|---:|---:|
    | ours (us) | 582 | 631 | 690 | 735 | 848 | 1,182 | **1,735** |
    | Marlin (us) | **471** | **502** | **559** | **616** | **784** | **1,165** | 1,812 |

    Marlin wins by 16-20 % up to a few hundred columns and loses by 4.5 % at
    2,688. The reason is where each finds its parallelism: at 99 columns an
    expert holds about four rows, and our kernel splits a job's 32 columns
    across four warps, so one warp works and three idle — the decode-width
    profile shows it plainly (sm 37 %, dram 40 %, **warps_active 24 %**, none
    of them the limit). Marlin parallelises over the expert's 1,024 output
    rows instead and keeps every warp busy. Above the crossover ours wins
    because it computes gate and up in registers and writes `silu(gate)*up`
    directly, where Marlin needs a second fold pass.

    **Decode rounds are 15.8 of every 19.6 rounds**, so this is the dominant
    regime — but the crossover cannot be built as things stand. The Marlin B
    tiles are byte-identical in size to our Q4 codes, so a weight is adopted
    *in place* (40 layers of a second plane would be 10.7 GB, which the card
    does not have), and adoption is one-way: our kernels cannot read Marlin
    tiles. Routing only the narrow rounds through it therefore corrupts every
    wide round, and the decode and small-T MoE families — which graph capture
    exercises at T = 1..16 — would need porting too. With all of that done the
    round model puts the result at **~2,200 against vLLM's 2,290**, still short,
    and Marlin's rel_l2 is 1.56 % where ours is 0.16 % (gate and up round
    through bf16 before the fold, and our FP16 scales round to BF16 because
    Marlin ties the scale type to the compute type).
  - **A deeper cp.async pipeline** in the routed kernels: 3 stages instead of 2
    is 590/848/2,139 us at 99/704/2,688 columns against 571/848/1,735 — shared
    memory is what caps occupancy at 3 blocks/SM, and a third stage spends more
    than it hides.
  - **The MoE wide-plan threshold** (`SUROGATE_SERVE_MOE_WIDE_MIN`): the
    crossover is already right. Narrow wins below ~1,100 columns, wide above,
    and a batched round is above it. Sweeping it moved 1,792-1,823, noise.
  - **The persistent grid** (`SUROGATE_SERVE_MOE_BLOCKS_PER_SM`): saturates at
    the 3 blocks/SM it already uses (553 GB/s; 8 blocks gives 561, 2 gives 505).
  - **MTP speculation**: needs 16.4 GB of runtime reservation at 128 lanes
    against 6.8 GB free after weights, and the MTP block adds 2.2 GB of its own
    (it carries a full expert set). It does not fit at any lane count worth
    running.
  - **Decoding the A fragments straight into registers**, dropping the 8 KiB
    decoded-weight plane so the block fits 3 -> 4 per SM. Occupancy is what the
    profile says the kernel wants, and the fragment math is right (parity
    passes), but it is slower everywhere: 592/934/2,204 us at 99/704/2,688
    columns against 571/848/1,735. `ldmatrix` earns its 8 KiB — sharing the
    plane means one wide shared load per fragment and one decode for all eight
    warps, where the register form makes every warp redo the decode over
    byte-wide, bank-conflicted reads.
  - **The MoE family crossover.** `sparse_moe` picks the decode family below 47
    tokens and the prefill family above, and both halves of that are right:
    the decode family costs 2,246 us at 99 columns against the prefill family's
    584, and at the crossover itself they meet (473 us at T=46 decode, 489 at
    T=64 prefill). A serving round at 100 users carries 98 decode columns, so
    the board already runs the right one.
  - **bf16 KV** (1,774 against 1,798) and **chunk width** 2,048/4,096/8,192
    (1,765/1,798/1,778): flat.

The gap is the same on every shape, and it is **not weight bytes**. Two
passes, cards rotated, 100 users, 90 s:

| shape | engine | vLLM | ratio |
|---|---:|---:|---:|
| 512/128 (prefill-weighted) | 1,942 | 2,290 | 85 % |
| 256/256 | 2,897 | 3,406 | 85 % |
| 128/512 (decode-heavy) | 3,339 | 3,740 | 89 % |

A uniform 11-15 %, so it is per-round efficiency rather than anything about
how the workload is shaped. And the bytes run the other way: **our weights are
19.59 GiB where vLLM's are 21.03 GiB** (its own loader reports it). We stream
7 % less and are 13 % slower, so at equal kernel quality this model should be
ours by about 7 %. Everything left is kernel efficiency.

What is left is the two things the round model names. The fixed term is 41 % of
the second and is the expert stream — 18.6 GB per round at ~890 GB/s, half of
what the card can do — and the routed kernels are latency-bound rather than
bandwidth- or compute-bound (sm__throughput 60 %, dram 23 %, warps_active
44 %, occupancy capped by shared memory). The other 55 % is the column cost of
those same kernels. Both are the same kernel-design problem, and vLLM's own
kernel is not the answer to it at these widths. There is no byte lever here — we already read less than vLLM does. The
concrete opening is the decode-width warp waste: at 99 columns our kernel
leaves three of four warps idle, Marlin's row-parallel decomposition is 19 %
faster there, and a decomposition that splits row-blocks across warps (four
busy warps, weights decoded in registers, activations shared) should beat both.
The round model says that kernel at ~400 us per layer, together with prefill
batch 6, lands about 2,340 — past vLLM. That kernel does not exist yet.

## What a 27B round costs, and what that rules out (2026-08-27 late)

`SUROGATE_SERVE_ROUND_TIMING=1` (wall clock around each segment, no extra
syncs — unlike `PREFILL_TIMING`, which slows the engine 2x and is useless
for throughput attribution) at the tuned config, 96 lanes, chunk 4,096,
100 users, 512/128:

| round | ms | rounds per 60 s | columns |
|---|---:|---:|---:|
| mixed (one prompt's chunk + the decode batch) | 111.5 | 450 | 640 + 84 |
| decode | 38.4 | 257 | 84 |

Two points, one straight line: a round costs **28.8 ms fixed plus 114 us per
column**. It reproduces the measurement (7.5 mixed + 4.3 decode rounds per
second = 990 tok/s against 970 measured) and it holds three times further
out — the prefill-heavy shape's 2,146-column rounds predict 7,664 prompt
tok/s against 7,339 measured. So the second is spent:

| | ms/s | share |
|---|---:|---:|
| prefill columns (5 per generated token at 512/128) | 494 | 49 % |
| the fixed cost, once per round | 334 | 33 % |
| decode columns | 111 | 11 % |

Written out, `throughput = 1 / (6c + F/B)` with c the column cost, F the
fixed cost and B the decode batch. That model rules out three things we
tried, and the measurements agree with it in every case:

  - **Batching several prompts into one mixed round** (the machinery from
    #80). Algebraically a no-op: `n_m * 640ck` with `n_m = D/(128k)` is
    independent of k. Fewer mixed rounds force exactly as many more decode
    rounds, and every round pays F.
  - **More lanes.** B is `min(lanes, KV pages / 11)`, and pages fall by 44
    for every lane added (each lane reserves a ~99 MB GDN state slot), so
    96 lanes admit 84 sequences and 72 lanes admit 72. Sweeping 72→104
    lanes and 2,048/4,096 chunks moved decode between 944 and 994 — the
    engine's own run-to-run spread.
  - **MTP speculation** (#85 lifts the eight-row replay cap). It works and
    it is not marginal: 76.8 % of drafts accepted, 1.77 tokens per round,
    +33/48/55 % at eight lanes for one/two/three draft tokens. But it
    reserves a second state slot per lane — 171 MB per lane against 99 — so
    the 27B fits 48 lanes with it instead of 96, and 48 lanes + d=1 gives
    885 against the 96-lane control's 996.
  - **A finer prefill ladder.** A 552-token prompt runs a 640-column graph
    under the 128-rounded ladder: 88 pad columns, 7.5 % of the second, and
    at 114 us each they are not free. Rounding to 64 measured +2.8 % once
    and −1.2 % when the cards were rotated (996/958 against 969/990), so it
    is noise, and the 4B — five times as many rounds, so five times the
    capture cost for the extra families — lost 3.6 %. Reverted.

Final 27B board, two passes with the cards rotated between them, everything
running at once:

| | pass 1 | pass 2 | mean | TTFT p50 |
|---|---:|---:|---:|---:|
| surogate serve, 96 lanes | 969 | 990 | 980 | 1.9 s |
| **vLLM** | 1,034 | 1,043 | **1,039** | 8.3 s |

**94 % of vLLM on decode, 4.4x better on TTFT.** What is left is not
scheduling and not batching: it is the 114 us column. That is 474 TFLOP/s
against the 651–827 the GEMMs themselves measure, and the per-window laps
already located the difference — about 18 % of a captured window is gaps
between kernels, and the GDN chunked scan moves its 145 MB per layer at
641 GB/s. Fusing the layer loop and widening that scan is the next 27B
work; nothing above it is a config away.

## Qwen3.8-Flash-Next — the llama.cpp baseline (2026-08-28)

The next target (`design/serve-engine-flash-next.md`): 111 GB, 512-expert MoE
with hyper-connections and an n-gram memory, which no engine target serves
yet. llama.cpp upstream master (added `qwen4exp` this week; CUDA build at
`study/llama.cpp-master/build`) is the parity oracle and the bar. 512/128,
90 s at 16-32 users, 60 s at one:

| config | users | decode tok/s | prefill tok/s | TTFT p50 |
|---|---:|---:|---:|---:|
| 8× 5090, `--split-mode layer`, all resident | 1 | 39.3 | 157 | 0.95 s |
| | 32 | 28.8 | 115 | 132 s |
| 1× 5090, experts + PLE table on CPU (`-ot exps=CPU`), 32 threads | 1 | 7.1 | 29 | 2.0 s |
| | 16 | 16.3 | 65 | 29 s |
| **surogate serve v0**, 1× 5090, experts + PLE table zero-copy from pinned host (no cache, no CPU compute) | 1 | 5.2 | 21 | 1.79 s |
| | 16 | 7.6 | 30 | 15.1 s |
| ik_llama.cpp 7cff686d (AVX-512 iqk kernels, fused MoE), 1× 5090, `-ot exps=CPU`, 32 threads — loadgen timeouts counted as errors, rerun pending | 1 | 21.8 | 87 | 1.8 s |
| | 16 | 23.9 | 96 | 30 s |
| **surogate serve + expert slot cache** (`--expert-slots 3000`, 14.6 GiB device pool, bulk host→device gather of misses, no CPU compute) | 1 | 18.5 | 74 | 3.0 s |
| | 16 | (running) | | |

Both configurations answer the probes correctly (`'Paris'`, `'2, 3, and 5'`).
The engine row (2026-08-28, parity with llama.cpp verified stage by stage, see
`design/INFERENCE.md`) is the zero-copy v0: the MoE kernels dereference the
pinned host bank row by row over PCIe, which `nsys` shows is 95 % of decode at
4.7–12 GB/s effective. The expert slot cache (bulk gather at ~50 GB/s, hits free)
is the first phase-2 step; the CPU expert compute split is the second.
The 8-GPU number is the interesting one: eight cards holding every weight
resident produce *less* aggregate throughput at 32 users than at one, because
layer split without micro-batch pipelining serialises the cards and the
scheduler starves prefill (132 s TTFT). That is the shape the pipeline-parallel
design in phase 3 is built to beat by an order of magnitude; the single-GPU
CPU-MoE row is the phase-2 bar.

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

## Qwen3.6-35B-A3B (MoE)

First served on 2026-08-27 (PATCHES.md #74). The artifact is the committed
mixed Q4/Q5/Q6 routed-expert recipe, 22.4 GB, converted without the DFlash
drafter; it fits a 5090 with 8.4 GB left after weights, so no expert
offloading is involved. 100 users, 90 s, GPU7, fp8 KV, 64 lanes:

| engine | card | decode tok/s | prefill tok/s | TTFT p50 | reqs ok/err |
|---|---|---:|---:|---:|---:|
| surogate serve (512/128) | GPU1 | 1,510 | 6,041 | 3.0 s | 1,123/0 |
| **vLLM** NVFP4 (512/128) | GPU1 | **2,252** | **9,009** | **2.9 s** | 1,631/0 |
| surogate serve (2048/16) | GPU1 | 107 | 13,720 | 14.1 s | 698/0 |
| **vLLM** NVFP4 (2048/16) | GPU1 | 176 | **22,569** | **7.7 s** | 1,076/0 |
| surogate serve (512/128), chunk 4,096 | GPU3 | 1,509 | 6,036 | 3.0 s | 1,122/0 |
| **vLLM** NVFP4 (512/128) | GPU3 | **2,260** | **9,042** | **1.3 s** | 1,641/0 |
| surogate serve (2048/16), chunk 4,096 | GPU3 | 116 | 14,884 | 12.8 s | 748/0 |
| **vLLM** NVFP4 (2048/16) | GPU3 | 172 | **22,050** | **7.9 s** | 1,052/0 |

The engine is at 67 % of vLLM on both shapes (prefill-heavy 61 % before
`--max-num-batched-tokens 4096`). (Earlier GPU7 rows, different card:
1,596 / 6,384 and 13,531.)

**The 64-lane figure above was the engine's old ceiling, not the 35B's**
(2026-08-28). #79 moved the cap to 128 and nothing about the 35B stops it
taking them — the artifact leaves 8.4 GB after weights, and KV then admits 99
sequences. One card each, everything at once, 512/128, 100 users, 90 s:

| lanes | chunk | decode tok/s | prefill tok/s | KV pages | running | TTFT p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 4,096 | 1,595 | 6,380 | 6,724 | 64 | 3,456 ms |
| 96 | 4,096 | 1,746 | 6,983 | 4,428 | 96 | 320 ms |
| 128 | 4,096 | **1,781** | **7,126** | 2,133 | 99 | **129 ms** |
| 128 | 2,048 | 1,763 | 7,053 | 2,530 | 99 | 129 ms |

**+12 % decode and 27x better TTFT for a flag.** Paired against vLLM at that
setting, two passes with the cards rotated:

| | pass 1 | pass 2 | mean | TTFT p50 |
|---|---:|---:|---:|---:|
| surogate serve, 128 lanes | 1,787 | 1,780 | 1,784 | **129 ms** |
| **vLLM** | 2,280 | 2,234 | **2,257** | 1,259 ms |

So the 35B is at **79 % of vLLM** on decode, up from 67 %, and leads TTFT by
**9.8x**. What is left is the weight format, as at the 4B: our artifact is
21.3 GB of mixed Q4/Q5/Q6 routed experts where vLLM reads an NVFP4 export.
`RedHatAI/Qwen3.6-35B-A3B-NVFP4` is on disk and quantises the attention,
output and expert matrices (only `linear_attn` stays BF16), so the 4B's
+54 %-from-format result is the precedent to chase here.

vLLM's rows above run `RedHatAI/Qwen3.6-35B-A3B-NVFP4` from a local copy whose
`config.json` ignore list gained `re:.*linear_attn\\.in_proj_.*` — the export
keeps `linear_attn` in bf16 but names the modules `in_proj_qkv`/`in_proj_z`
while vLLM checks `in_proj_q/k/v/z`, so unpatched it fails with "different
quantization schemes". Flags: `--quantization compressed-tensors
--language-model-only --kv-cache-dtype fp8 --gpu-memory-utilization 0.92`.
Before that workaround, vLLM was blocked on an export it could load: the
`unsloth` NVFP4 export quantizes `linear_attn.in_proj_*` to FP8 and the fused
`in_proj_qkvz` loader dies on the split `weight_scale`; the `RedHatAI` export
keeps them bf16 but names them `in_proj_qkv`/`in_proj_z` in its ignore list
while vLLM checks `in_proj_q/k/v/z` ("different quantization schemes"). A
local copy with `re:.*linear_attn\\.in_proj_.*` added to the ignore list is
the workaround under test. The export behind the 08-26 row was a local
directory that no longer exists.

For reference, the 2026-08-26 GPU2 rows: vLLM NVFP4 1,565 tok/s (TTFT 2.2 s,
1,172/0), llama-server GGUF Q4_K_M 213 (238/111); single-user vLLM 208 ms
TTFT @1.9k / 163 tok/s, llama-server 608 ms / 159.

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

## Rewrite checkpoints off by default (PATCHES.md #73)

The GDN state pool used to hold two slots per lane: the live state and a
rewrite checkpoint for resuming an edited last turn. On the 27B a slot is
72 MiB, so the checkpoints held 3.4 GB at 48 lanes and pushed 64 lanes off a
cliff (8,576 KV tokens). They are now off by default (`--rewrite-checkpoints`
opts in; an edited turn re-prefills its prefix instead):

| 27B, fp8 KV, `--kv-capacity auto` | KV tokens | decode tok/s | TTFT p50 |
|---|---:|---:|---:|
| 48 lanes, checkpoints on (GPU7) | 92,096 | 824 | 8.0 s |
| 48 lanes, checkpoints off (GPU6) | 206,976 | 822 | 8.0 s |
| **64 lanes, checkpoints off (GPU7)** | 161,792 | **859** | **5.4 s** |

64 lanes is now the better 27B default. The gain is small because the 27B is
prefill-bound (prefill table above), not lane-bound. The 0.8B/4B board rows
are unchanged within noise on the new default (7,721 / 4,368 on the paired
cards), and the decode-heavy soaks on it are clean (27B 820 and 4B 3,846
requests, 0 errors, 0 fatals).

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

- **27B, 100 users: −20 % decode, −49 % prefill** against vLLM on the same
  card. The GEMM family is the lever now measured: cuBLASLt's block-scaled
  FP4 matmul is 1.4–1.9× the in-house W4A4 kernels from 64 tokens up,
  bit-exact, in place (PATCHES.md #76 routes prefill widths to it); the GDN
  recurrent scan (sequential, 23 % of the GDN family) is the next one.
  Lanes above 48 lose because per-lane GDN state squeezes the cache.
- **0.8B rare corruption**, ~4 per 100,000 requests (PATCHES.md, "Open").
- **NVFP4 MoE conversion** for the 35B-A3B class (vLLM's 1,565 tok/s is the bar).
- **Single-source NVFP4 ingest**: the converter still needs the bf16 base
  alongside the quantized export.
- **Record the card with every number**, and re-measure single-user cells
  on the same card as the 100-user rows.
