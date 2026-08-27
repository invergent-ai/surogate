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

### Why the card matters (2026-08-27)

All eight 5090s run under a 400 W software power cap (hardware maximum
575–600 W) and settle at very different SM clocks under load — measured
mid-run: GPU2 1,462 MHz, GPU0 1,612, GPU4 1,710, GPU3 1,950, GPU5 2,167,
GPU6 2,625. That 1.8× clock range is the whole "card spread": the same
cuBLASLt GEMM runs 763 TFLOP/s on GPU2 and 1,379 on GPU7. Only same-card
pairs are comparable; the kernel table below was taken on GPU2, the slowest
card, so its absolute numbers are a floor and its ratios are what matters.

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
reaches 15,276 on GPU7. Lanes on the 27B: 64 beats 48 on every shape now
(balanced 897 vs 833-class, decode-heavy 1,884 vs 1,633). Single-user prefill
(2,048 tokens): engine 264 ms eager / 257 ms graph (GPU5) against vLLM
213 ms (GPU3), so the single-stream gap is ~1.3× and the rest of the 2×
lives in the 100-user mixed-round regime.

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

The engine is at 67 % of vLLM on the balanced shape and 61 % on prefill-heavy
— the same gap as the 27B, and the 35B has no NVFP4 weights, so the gap is not
the GEMM family. (Earlier GPU7 rows, different card: 1,596 / 6,384 and 13,531.)

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
