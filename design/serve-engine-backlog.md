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

**The scoped fix was built and is worth nothing — 2026-08-30.** A row-parallel
gate/up kernel (`<4, 32>`: warp w owns row tile w and walks every column tile,
the SwiGLU pair exchanged through the dead `Bs` buffer) passes the fp64 oracle
including T=47, which routes through it, and `nsys` confirms it is the kernel
that runs. It is **flat at every width**:

| T | row-parallel | column-parallel |
|---:|---:|---:|
| 64 | 442.4 us | 440.4 us |
| 100 | 532.5 us | 530.4 us |
| 128 | 571.4 us | 573.5 us |
| 256 | 659.5 us | 661.5 us |
| 512 | 702.5 us | 702.5 us |

Idle warps were a symptom, not the cause. Two more occupancy levers move it
just as little: `SUROGATE_SERVE_MOE_BLOCKS_PER_SM` 3 → 12 (530-553 us, no
trend) and raising the kernel's `__launch_bounds__` residency hint from 3 to 8
(532 us). The kernel was reverted; only the finding is kept.

**What the arithmetic says instead.** At T=100 the round is 532 us and
`sparse_moe_prefill_q4_gate_up_kernel` is 63 % of it (345 us, `nsys`); the
routed down kernel is another 20 %. In those 345 us gate/up streams the 188
touched experts' weights — 188 x 1024 rows x 2048 values at 4.25 bpw = 209 MB —
which is **606 GB/s, 34 % of the 1,792 peak**, against 3.4 GFLOP of math, about
5 % of the card's bf16 rate. The kernel is not short of warps or blocks; it is
short of memory throughput on a stream that no occupancy knob accelerates.

The leading hypothesis is the read *shape*, not the byte count: for a fixed
k-group consecutive rows sit `GroupsPerRow * 32` = 1,024 bytes apart, so a block
issues 128 scattered 32-byte reads per k-step and never gets a contiguous run.
That is a weight-layout question — k-group-major within a row block — which is
the rewrite `kExpertBK=64` was always going to force. It is **not** confirmed:
`ncu` needs GPU counter permissions this host does not grant
(`ERR_NVGPUCTRPERM`), so the next person should start by getting counters and
measuring sector efficiency before writing any kernel. The bench
(`ninfer_sparse_moe_bench`, repaired 2026-08-30 — it had not compiled since the
geometry argument landed) and `ninfer_sparse_moe_test` are the harness.

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
