# Serving engine — backlog

Open problems with their evidence, so the next person starts where the last one
stopped. Board of record and per-model attribution live in
`surogate/serve/BENCHMARKS.md`; patch history in `csrc/src/serve/PATCHES.md`.

## B1. Qwen3.6-35B-A3B (Q4/Q5/Q6 MoE) is at 85 % of vLLM — parked 2026-08-28

**Where it stands.** 1,942 decode tok/s against vLLM's 2,290 (100 users, 512/128,
90 s, two passes with cards rotated), TTFT 253 ms against 1,256 ms. The gap is
uniform across shapes (85 % at 256/256, 89 % at 128/512), so it is per-round
efficiency, not workload shape.

**What it is not.** Not weight bytes: our artifact loads 19.59 GiB where vLLM's
own loader reports 21.03 GiB — we stream 7 % less and run 13 % slower. Not
scheduling: the round-cost model (20.9 ms fixed + 56 us/column) was validated
and every scheduling lever measured. Not the MoE family crossover (both halves
verified at the boundary).

**What it is.** Kernel efficiency of the routed-expert GEMM family
(`ops/sparse_moe/prefill/`). At decode width (~99 columns) an expert holds ~4
rows and the kernel splits a job's 32 columns across 4 warps, so one works and
three idle: sm 37 %, dram 40 %, warps_active 24 %, none saturated. The weight
stream runs at 865 GB/s of 1,792 with no read amplification. Occupancy is
capped at 3 blocks/SM by ~33 KiB of shared memory.

**Shipped from this effort (+22 % on the model):** lane cap 64→128 (#79), and
batched prefill for MoE targets (`SUROGATE_SERVE_PREFILL_BATCH`, #88) — pays
because the per-token expert cost falls 1.26→0.64 us from 640→2,688 columns;
a no-op on dense models by the same arithmetic.

**Measured and rejected (do not re-run):**

| lever | result |
|---|---|
| vLLM MoE Marlin, full port (#89, behind `SUROGATE_SERVE_MOE_MARLIN`) | 16-20 % faster ≤320 columns, 4.5 % slower at 2,688; adoption is in-place and one-way (a second plane is 10.7 GB), so it cannot own only the narrow rounds; rel_l2 1.56 % vs our 0.16 % |
| register-decoded A fragments (drop the 8 KiB plane, 3→4 blocks/SM) | 27 % slower — `ldmatrix` earns its plane |
| 3-stage cp.async pipeline | slower everywhere, shared memory bound |
| MoE wide-plan threshold, persistent grid | already optimal / saturated |
| MTP speculation | needs 16.4 GB at 128 lanes vs 6.8 free (MTP block carries an expert set) |
| bf16 KV, chunk width, compacting `Sr` | flat / 16x sector amplification |

**The scoped fix.** A row-block-per-warp decomposition of the routed gate/up
and down kernels: each warp owns a row-block of the expert (four busy warps at
any width), weights decoded in registers, activations shared, ≤25 KiB shared so
4 blocks/SM fit. Marlin's row-parallel decomposition proves ~19 % is there at
decode width; the round model needs ~1.3x across all widths for parity and
~1.5x to win clearly. `kExpertBK=64` is one Q4 group and is baked into the
staging, swizzle and scale indexing, so this is a rewrite of the family, not a
constant change. Budget it as a kernel project with the bench
(`ninfer_sparse_moe_bench --sweep 64:2688`) and parity test
(`ninfer_sparse_moe_test`) as the harness.

## B2 — Flash-Next phase-1 exclusions (2026-08-28)

Recorded while onboarding Qwen3.8-Flash-Next (`targets/qwen4exp/`); none blocks the
first serve, each is a known limit of phase 1.

- **Context above 2051 tokens.** Dense attention is exact only below the QSA indexer's
  budget (`indexer_top_k + block - 1`); the indexer weights are in the artifact (validated,
  not resident). Serving longer contexts needs the indexer op and its selected-token mask in
  `gqa_attention`.
- **int8 KV for the 24q2 attention geometry.** The int8 decode kernel tiles at most three
  query-row tiles (48 rows); the group of twelve exceeds that from width 5. fp8 (the server
  default) and bf16 KV use the BF16 kernel and are served. Lane steps of more than 64 query
  rows (width ≥ 6 at group 12) are refused; only speculative widths reach them.
- **Speculative decoding (MTP/DFlash) and vision** are refused at `plan_load`; the GGUF
  carries no MTP block.
- **Expert access is zero-copy from pinned host memory (v0).** Every routed expert read
  crosses PCIe; phase 2 adds the device slot cache, CPU expert compute, and the
  bandwidth-matched split (FreeToken hybrid). The pinned bank fills at ~1.3 GB/s at load
  (~2 min for 154 GB); `Reader::read_direct` in large chunks would cut that.
- **BF16 dense GEMMs run through cuBLASLt** (hc down/up/inject, PLE key/value, GDN a_b);
  the hand-tuned BF16 family knows only the 27B MTP shapes. The hc weights read 1.26 GB
  per decode round at batch 1 (BF16); W8 storage would halve that.
- **Launcher GGUF path** fetches the frontend from the Hub (`Qwen/Qwen3.8-Flash-Next`);
  offline conversions need `--frontend` pointed at a local snapshot.
