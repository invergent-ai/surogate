# Inference engine — progress log

Running record of the serving-engine work so it can be picked up cold. Newest entries at the
bottom of each section; the plan itself is `design/serve-engine-plan.md`, the Flash-Next design
`design/serve-engine-flash-next.md`, parked items `design/serve-engine-backlog.md`, the board of
record `surogate/serve/BENCHMARKS.md`.

## Objective (2026-08-28)

Serve `models/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1..4}-of-00004.gguf` (111 GB, GGUF arch
`qwen4exp`) on one RTX 5090 with CPU offloading (FreeToken/llama.cpp style) and on 8 × 5090
with pipeline and expert parallelism (vLLM style, no P2P/NVLink on consumer cards). The core
stays model- and SM-agnostic: every new architecture is a target on top of shared contracts.

Phase 1 = onboard the model on the qwen3_6 family runtime with simple expert streaming;
phase 2 = FreeToken hybrid (GPU slot cache + CPU expert compute + bandwidth-matched split);
phase 3 = PP across 8 GPUs with the offload inside each stage; phase 4 (EP) rejected by analysis for PCIe-only hosts — see Decisions.

## Status

| step | state | evidence |
|---|---|---|
| Model contract (hc, PLE, GDN sigmoid gate, MoE, indexer) | done | `design/serve-engine-flash-next.md` §5; llama.cpp `study/llama.cpp-master/src/models/qwen4exp.cpp` is the oracle |
| GGUF-native converter `tools/convert/qwen4exp/` | done | commit 81d818d9; W8 experts, un-tiled GDN V heads, q/k norms stored HF-style (γ−1) |
| Artifact `/home/densemax2/work/models/ninfer/qwen3_8_flash_next.ninfer` | done, verified | 163.1 GB in 1046 s; scratchpad `verify_flash_artifact.py`: W8 repacks and the 28.8 GB PLE table bit-exact vs the GGUF, requantised gate/up ≤ 5.4e-3 rel-L2, VERIFY_DONE bad=0 |
| llama.cpp baseline | done | BENCHMARKS.md: 8×5090 39.3 tok/s @1 / 28.8 @32; 1×5090 CPU-MoE 7.1 / 16.3 |
| Sigmoid-gated RMSNorm, W8 dispatch arms (13312/16384 × 2560, 2560 × 6144) | done | commit d2174926 |
| Sparse-MoE geometry from the weights (stage A) | done | commit 13a08634 |
| Sparse-MoE kernels instantiated per geometry, 512/10/640/2560 compiled (stage B) | done, 35B probe clean | see 2026-08-28 entry below |
| BF16 cuBLASLt GEMM route (`ops/linear/bf16/bf16_cublaslt.*`) | built (commit 81257e66) | for hc down/up/inject, PLE key/value, GDN a_b at 96×2560 |
| Hyper-connection op (`api/ops/hyper_connection.h`, `ops/hyper_connection/`) | built (commit 81257e66) | mix / combine / broadcast_streams |
| PLE op (`api/ops/ngram_ple.h`, `ops/ngram_ple/`) | built (commit 81257e66) | device-side hash, IQ4_NL row gather from pinned host, group norms, gate, dilated conv with per-slot state |
| qwen3_6 family: residual-width trait, embed/finish/norm hooks (F1), layer-prologue hook with per-column segment facts (F2) | done (commits 5b1b6efd, b4e13eb6), 35B unchanged | `runtime/residual_policy.h`, `runtime/prologue_columns.h`; staging in every forward entry |
| PLE state pool `core/ngram_ple_state.*` (per-slot conv history [9,10240] + token history [2]) | wired (F3, in 9a374d71) | `DecoderState::{copy,reset}_state_slot` forward to both pools at the slot-lifecycle sites; `ExecutionCore::ple` reaches every TextContext |
| Target `targets/qwen4exp/` (package, bindings, host bank, variant, registry) | serves with parity (2026-08-28) | model_id `qwen3.8-flash-next`, weights_id `w8-hc-v1`, target key `qwen4exp`; experts and the PLE table live in pinned, device-mapped host memory and the kernels read them zero-copy (v0) |
| Expert access v0: zero-copy reads from the pinned host bank (`impl/load/host_bank.*`) | written with the target | no staging copies at all in v0; the device slot cache and CPU expert compute come in phase 2 |
| Parity vs llama.cpp | done 2026-08-28 | token-0 stages within BF16 noise of the CPU reference, `l_last-0/1/2` match, answers `Paris` / 2,3,5 / ocean; defects were the SiLU gate (4aaa07fc) and the RMSNorm kernels' gate load (see log) |
| Phase 2: expert slot cache (`--expert-slots N`) | done 2026-08-28 | users=1 18.5 tok/s loadgen / 31.7 pure decode (v0 5.2), 63 % hits at one user, pool created before the KV plan; 16 users 9.4 (pool thrashes) |
| Phase 2: CPU expert split (`--cpu-moe-share F|auto`, prefill share, batched VNNI host kernel) | **done 2026-08-28** — defaults: 1 user TTFT 1.40 s (366 t/s prompt processing) / 22.4 tok/s; 16 users 32.2 tok/s (37.9 at explicit shares); correct under load, 0 fatals; 35B board unchanged (1,769) | rows in BENCHMARKS.md; the remaining phase-2 levers (Q4 host bank, vectorised tile repack, sparse indexer for >2k) are listed under Open items |
| First throughput row (zero-copy experts v0, board shape 512/128) | done 2026-08-28 | users=1: 5.2 decode / 21 prefill tok/s, TTFT 1.79 s; users=16: 7.6 / 30, TTFT 15.1 s; (128/128: 4.9 @1, 8.3 @16, 26.6 @32). llama.cpp 1×5090 CPU-MoE: 7.1 / 29 @1, 16.3 / 65 @16 |

## Next: phase 2 (single-GPU offload) — plan as of 2026-08-28

Where v0 stands: the MoE kernels read expert rows straight out of the pinned host bank over
PCIe (zero-copy inside the kernel), 4.9 tok/s at one user. The bandwidth bound for that path
is ~20 tok/s (10 experts × 4.9 MB × 48 layers = 2.4 GB/token at ~50 GB/s), so the kernel's
2.5 KB-row access pattern is latency-bound over PCIe, not bandwidth-bound — profile first
(`nsys` on the CLI decode) to confirm before changing anything.

Design (model-agnostic, keyed by `SparseMoeGeometry`; no kernel changes):

1. **Expert slot pool** on the device: `slots × (gate_up rows + down rows)` in the kernels'
   own W8 row-split layout (planes are row-major, so one expert is one contiguous slice per
   plane: codes + scales = two banks per matrix), so a slot is addressed exactly like an
   expert (`row_base = slot * 2 * intermediate`); `SparseMoeWeights.routed_*` point at the
   pool. Indirection: an optional per-layer device table `slot_of_expert[experts]` on
   `SparseMoeWeights`, consulted at the five row-base sites (decode d3/d4, small-T, prefill
   ×3) — ids stay expert ids everywhere (the prefill histogram/scan is per expert), a null
   table means resident experts. One-line change per site, no schedule changes.
   Split points (read 2026-08-28): decode d2 (top-k ids) → d3; small-T s2 → s3; prefill
   `select_count` (ids final) → `scan`/gather/expert kernels — resolve runs once per layer on
   the ids buffer each path already produces, so the three schedules keep their kernels.
2. **Resolve + gather op** between routing and the expert kernels: the wrapper splits into
   `sparse_moe_route` (d1/d2 → expert ids, α, shared scale), `expert_slot_resolve` (device
   table `[layers × experts] → slot`, LRU timestamps, active-slot protection, miss list with
   a device count word), `expert_slot_gather` (FreeToken's `fast_index_copy_multi` pattern:
   one launch copies every bank of the missing experts from pinned host pointers, row count
   read from the device word, `L1::no_allocate` loads) and the unchanged expert kernels fed
   with *slot* ids. Everything reads device words, so the whole round captures into the
   decode graph.
3. **Sizing**: MoE-first against a KV floor (`--kv-capacity`), flat id `layer*experts+expert`;
   `decode_routing_stats`-style oracle hit rate from observed routing before any policy
   beyond LRU.
4. **CPU expert compute + bandwidth split** (D3 in `serve-engine-flash-next.md`): only after
   1–3 are measured; the split fraction is computed on the device from a measured profile.
   CPU kernels: reuse `study/ik_llama.cpp` (owner's pointer, 2026-08-28) — `ggml/src/iqk/`
   `iqk_mul_mat_moe` / `iqk_moe_fused_up_gate` (AVX-512 `HAVE_FANCY_SIMD` path, needs the
   five AVX-512 macros per `docs/build.md`), and its `mul_mat_id` scheduling (rows grouped per
   expert, expert chunking over an atomic counter). Re-measure the CPU-MoE bar with that fork
   (`-fmoe`, no `-rtr` in hybrid mode) before claiming the ≥ 3× exit.
   Layout note for that step: our host bank is W8G32 *planar* (codes plane + fp16 scales
   plane per matrix, which is what the GPU gather wants); ik's kernels consume ggml block
   types (`Q8_0` = interleaved `{fp16 d; int8 qs[32]}`, numerically the same format). Either
   (a) keep a second, interleaved copy of the experts for the CPU (host RAM: 2 × 154 GB is
   tight on 503 GB), (b) gather from an interleaved bank and de-interleave in the gather
   kernel (byte-granular copies, PCIe is still the bound), or (c) write a planar-W8 AVX-512
   GEMV/GEMM of our own (decode is DRAM-bound so a simple VNNI kernel suffices; prefill is
   where ik's tiled GEMM would pay). Decide after the slot-cache numbers.
   Decision (2026-08-28, after the slot-cache numbers): (c) — an AVX-512 kernel over our own
   planar W8G32 rows, with ik's `iqk_gemm_legacy_quants.cpp` (Q8_0 path) as the reference
   for the inner loop. Reasons: decode is DRAM-bound (a whole layer's misses are ~2 GB/token
   at 160–320 GB/s), so a straightforward int8×bf16 FMA loop with one pinned thread per core
   per NUMA node reaches the bound; prefill is gather-bound per layer today (all ~512 used
   experts of a layer, 2.6 GB, land in the pool once per layer per chunk), so the CPU share
   helps prefill through the same split; and one host copy of the experts keeps the pinned
   footprint at 154 GB. The interleaved second copy (a) and the de-interleaving gather (b) are
   fallbacks if the planar kernel cannot reach DRAM speed.

35B regression watch (2026-08-28): `probe_35b.sh` on GPU 2 (both under host contention and
with an idle host) reports users=32 decode 1,137 / prefill 4,547 tok/s with correct answers,
against the board's 1,761 / 1,927 measured on GPU 1 earlier this session. The two GPU-2 runs
printed identical figures, so it is either a GPU-2 vs GPU-1 difference or a regression from
the day's family changes; a GPU-1 rerun with the current binary is queued behind the
Flash-Next measurements and decides it. The ik_llama.cpp build (7cff686d, CUDA + AVX-512)
succeeded; its server has fused MoE on by default (`-no-fmoe` disables; `-fmoe` is not a flag)
and the baseline run is queued.
- Slot cache, first end-to-end attempt (2026-08-28): full build OK, `ninfer_expert_slot_cache_test`
  **passes** (hits keep slots, misses gathered bit-exact through all four planes, eviction
  across layers, active-round protection). The CLI run itself tripped on my option choice
  (`--kv-capacity 4096` exceeds the CLI's usable range for `--max-context 2048`), so the
  gated throughput probes did not run; rebuilt with the pool-before-KV-plan change and
  re-queued with `--kv-capacity auto` behind the ik_llama.cpp baseline and the GPU-1 35B
  probe (both need an uncontended host / GPU 1).
- **ik_llama.cpp CPU-MoE baseline (2026-08-28, AVX-512 build, fused MoE default, `-ot exps=CPU`,
  32 threads, 512/128):** users=1 decode 21.8 tok/s, prefill 87 tok/s, TTFT 1.8 s; users=16
  decode 23.9, prefill 96, TTFT 30 s — 3× upstream llama.cpp's `--cpu-moe` (7.1 / 16.3) and
  the new phase-2 bar. Caveat: the loadgen counted errors (10/21 at users=1, 4/27 at 16) that
  the server log does not show — client timeouts (its default is short for 12-85 s requests);
  rerun with a long timeout queued. Its answers came with `<think>` blocks (the fork ignores
  `chat_template_kwargs.enable_thinking`), which does not change the throughput shape.
- **35B regression confirmed on GPU 1 (2026-08-28):** current binary, `probe_35b.sh`, users=32:
  decode 1,152 / prefill 4,610 tok/s (answers correct) vs 1,761 / 1,927 measured with the same
  script earlier today after the MoE-geometry refactor. GPU 2 gives the same 1,137, so it is
  the code, not the card. Bisect in flight: worktree `surogate-bisect` at 14ec176a (RMSNorm
  fix, before the slot-table/hook commits) and the pre-today worktree `surogate-pretoday`
  (f7145031), both probed on GPU 2 once the host is idle. Suspects, in order: the
  `slot_of_expert` parameter/indirection in the MoE kernels (register pressure / occupancy
  in the Q4/Q5 35B kernels, not the null check itself), the round-hook plumbing, the family
  hook no-ops. Rule: the 35B board rows must not move for the Flash-Next work.
- Slot cache end-to-end, second attempt (2026-08-28): with `--kv-capacity auto` the CLI ran
  and answered "The capital of France is **Paris**." — but at 5.37 tok/s, and the log had no
  "slot cache enabled" line: the new `--expert-slots` plumbing always calls
  `configure_expert_slots(0)` from the CLI default, and the lookup treated a configured 0 as
  "set" and skipped the env fallback, so the run measured v0 again. Fixed (a configured 0
  falls through to the env knob); the validation + probes rerun with `--expert-slots 3000`,
  then the ik baseline rerun and the 35B bisect probes follow on the same queue.
- **Expert slot cache, first numbers (2026-08-28, `--expert-slots 3000` = 14.6 GiB pool, KV auto,
  512/128):** users=1 decode **18.5 tok/s** (v0 5.2 → 3.6×), prefill 74 tok/s (v0 21 → 3.5×),
  TTFT 3.0 s, answers coherent on all four prompts (Paris; 2, 3, 5; ocean; Rayleigh
  scattering). That is within 15 % of ik_llama.cpp's 21.8 with only the bulk gather —
  no CPU expert compute, no hit-rate tuning, no prefill streaming yet. 16-user probe running.
  Reading of the round at users=1 (~54 ms/token): with 3,000 of 24,576 experts resident the
  hit rate is low, so ~400 misses × 5.2 MB ≈ 2 GB/token at ~50 GB/s ≈ 40 ms is the floor of
  this design — the CPU compute split (host DRAM 160–320 GB/s) is the next multiplier, then
  pool sizing against the KV floor.
- users=16 with the slot cache (512/128): decode 9.4 tok/s aggregate (v0 7.6), prefill 38,
  TTFT 17.9 s. Reading: at 16 lanes a decode round touches ~150 distinct experts per layer and
  a 512-token prefill chunk touches nearly all 512, so 3,000 slots (≈62 per layer) thrash and
  the round is one big PCIe gather (~0.8 GB per decode layer, 2.6 GB per prefill layer). The
  pool is a single-user / low-concurrency accelerator on this card; concurrency needs the CPU
  compute split (host DRAM 160–320 GB/s vs PCIe ~50) — implementation started:
  `api/ops/cpu_expert_compute.h` (planar-W8 gated FFN on the host, AVX-512 int8×int8 group
  dots with runtime detection and a scalar fallback, pinned worker pool, unit test against a
  double reference). The GPU-round integration (miss split, activation hand-off, host-function
  handshake, kernels skipping CPU-assigned pairs) is the next step.
- Follow-ups landed (2026-08-28, later): ik_llama.cpp rerun agrees (21.6 @1 / 24.0 @16, errors
  persist and are not server-side); the 35B bisect at 14ec176a gives 1,136 — the regression
  predates the slot-table/hook commits; the pre-today binary (f7145031) cannot load the current
  35B artifact (`dflash/feature_projection` missing — today's converter changes), so the next
  bisect points are 81257e66 (before the family residual hooks) and b4e13eb6 (after them,
  before the qwen4exp target), building now. The CPU expert-compute unit test passes on the
  AVX-512 path (rel-L2 ~1e-7 vs the double reference); the GPU-round integration is written:
  the resolve kernel takes a `cpu_share` and emits (token, expert, weight) jobs for the misses it
  keeps off the pool, the decode/small-T kernels contribute nothing for paths mapped to -1, and
  `qwen4exp`'s hook stages the activations + jobs to pinned memory, runs the pool from a
  `cudaLaunchHostFunc` node and adds the FP32 partial back (`SUROGATE_SERVE_CPU_MOE_SHARE=<f>`,
  decode/small-T rounds ≤ 64 tokens; prefill keeps the full gather). Compiling.
- ik_llama.cpp's loadgen errors are `ServerDisconnectedError` (its server drops some streamed
  connections; nothing in its log) — the ok requests carry the throughput figure. Options
  `--expert-slots N` / `--cpu-moe-share F` (server + CLI) now cover both phase-2 knobs.
- The first hit-rate diagnostic aborted: the readout did a synchronous copy inside the round
  hook while the decode graph was being captured. Hooks run once at capture (replays never
  call them), so the readout now skips capturing streams and is an eager-mode diagnostic
  (`--no-cuda-graph`); the CPU round itself is captured properly (memcpy + host-function +
  kernel nodes) and replays do call the host function.
- CPU split v2 (design, to do after v1 is measured): v1 serialises the host round on the
  main stream (D2H → host function → partial add → expert kernels), so the CPU and the PCIe
  gather never overlap. v2 moves the host round to a side stream forked after resolve
  (event), writes the partial into its own FP32 plane instead of the MoE output, and joins
  before `combine_into`, which then adds both planes (one extra read in the combine kernel).
  Expected at users=1 with share 0.7: CPU 1.4 GB at ~150 GB/s ≈ 9 ms overlapped with PCIe
  0.6 GB at 50 GB/s ≈ 12 ms → ~12 ms/token instead of ~21 ms serial (the pool-only path is
  ~40 ms). The share should track the measured CPU GB/s (the micro-benchmark) vs PCIe.
- Slot sweep (users=1, 512/128): 1,500 slots → 18.2 tok/s, 3,000 → 18.5. The pool size barely
  matters at one user, i.e. the hit rate is low either way and the gain over v0 came from the
  *bulk gather* (coalesced 16-byte copies at ~50 GB/s vs the kernels' 5–12 GB/s row reads),
  not from caching. Consequences: the pool can stay small (a staging buffer of a few layers'
  worth of experts) and give the memory back to KV; the CPU split and stream overlap are the
  levers; the eager hit-rate readout will quantify this. 4,500 slots (21.9 GiB) is refused by
  the planner on the 32 GB card ("automatic KV headroom requires 1 GiB, 0 available after
  weights") — the pool-before-KV ordering works as intended.
- Queue stall (2026-08-28, ~35 min lost): the bisect chain's process guard used `pgrep -f` with a
  pattern that also occurred elsewhere in its own command line, matched itself and spun; every
  chain gated on it (CPU-split run, hit-rate readout, CPU benchmark) waited until the owner
  pointed out the idle GPUs. Chains now gate on marker lines in output files and guard with
  `pgrep -x surogate-engine`; the bisect probes and the GPU-1 sequence were relaunched.
- 35B bisect (GPU 2, users=32): 81257e66 (before the family residual hooks) 1,134 and b4e13eb6
  (after them) 1,119 tok/s — the hooks are not it either; the slowdown predates 81257e66, which
  leaves stage B (823e240b, MoE kernels instantiated per geometry with derived constants) as
  the prime suspect. Stage A (13a08634) is building for the decisive probe; if it measures
  ~1,760 the fix is to compare stage B's derived constants (D1 warps, adaptive block counts,
  small-T router partitions / S2 batch limits, prefill persistent blocks, score-row padding)
  with the previous hand-tuned values for the 256/8/512/2048 geometry.
- **Correction:** the 1,761 / 1,927 figures came from `probe_35b_full.sh` (100 users, 128
  seqs, 4,096 batched tokens — the board's configuration), while every run today used
  `probe_35b.sh` (32 users, 32 seqs). All bisect points agree at ~1,130 tok/s *for 32 users*,
  so no regression has been demonstrated; the comparison was apples to oranges. The board
  configuration is being rerun with the current binary on GPU 2 (expect ~1,760 unbatched);
  the stage-A 32-user datapoint still lands as a sanity check. Lesson recorded: compare
  against the board only with the board's script.
- **Resolved:** the board configuration (`probe_35b_full.sh`, 100 users, 128 seqs, 4,096
  batched tokens) with the current binary gives decode **1,770 tok/s**, prefill 7,082, TTFT
  129 ms, 866/866 ok (board: 1,762 / —) — the 35B rows are intact after all of today's family
  and kernel changes. The 32-user datapoints (~1,130 at every commit) are simply the 32-user
  throughput. Bisect worktrees removed.
- **CPU split v1, first run (2026-08-28):** the CLI failed with `view element count mismatch`
  — in the decode loop the hook receives one token's ids but not which column of the block's
  output it is, so the Variant's thread-local output view had the wrong shape. Fix: the hook
  now carries the round's own input/output views (`resolve(ctx, ids, alpha, x, destination,
  stream)`; one column in the decode loop, the slice otherwise) and the Variant uses them.
- **Hit-rate readout (eager, 3,000 slots, single stream, 48 tokens):** sampled miss share
  36.5 % — i.e. **~63 % of routed experts hit the pool** at one user, far more than the sweep
  suggested; and the CLI's pure decode rate with the cache is **31.7 tok/s** (the loadgen's
  18.5 divides output tokens by wall time including the 512-token prefill; ik's per-stream p50
  is 35.4). So single-stream we are within ~10 % of ik with the pool alone.
- **CPU kernel micro-benchmark (synthetic 512-expert bank, real geometry):** 32 threads
  154 GB/s best (140 mean), 16 threads 178 (167), 8 threads 99 — one socket's DRAM speed;
  the 32-thread case loses to cross-NUMA traffic (the bank lives on one node) → NUMA-split
  banks next. Single-token rounds (10 jobs) only 10 GB/s: per job the kernel took ~6 ms, far
  below DRAM speed per core — the scalar fp16 scale conversion and per-row reductions
  dominated; the kernel now converts 16 scales at a time (F16C) and processes gate/up and
  down rows in pairs; re-measuring.
  Re-measured (while the 35B 100-user probe was loading the host, so aggregate figures are
  contaminated): one thread 405 µs per 5.2 MB expert = **13 GB/s per core** (was ~6 ms per
  job, i.e. the tightening is ~15×); 32 threads 139 GB/s, 16 threads 91 GB/s (contended);
  single-token rounds still ~6 ms because the 10 jobs leave 22 pinned threads idle and the
  wake-up latency under a loaded host dominates — an idle-host rerun is queued after the
  split probes. The pool needs job splitting (rows of one expert across threads) for
  single-user rounds and a spin-then-wait wake to cut the latency.
- NUMA plan for the CPU split (this box: 2 × EPYC 9124, 16 cores each, 2 nodes, ~160–190 GB/s
  per node): the pinned host bank is allocated by one thread, so its pages sit on one node and
  the 32-thread pool pays remote bandwidth. Two steps: (1) measure with the process under
  `numactl --interleave=all` (pages spread over both nodes; queued in the probe chain);
  (2) if it pays, allocate the bank per node (each layer's expert planes split by expert
  range across the two nodes, `mbind` before the first touch) and run one pool per node whose
  threads take the jobs whose expert lives on their node — the design's "one pool per NUMA
  node over that node's half of the expert bank".
- **CPU split v1 parity (2026-08-28):** with the hook carrying the round's views the CLI answers
  "The capital of France is **Paris**." with 50 % of every round's misses computed on the host —
  the whole host path (jobs from the resolve kernel, D2H staging, host-function round, FP32
  partial add) is numerically right. Speed as predicted for v1: 8.9 tok/s decode / 11.5 prefill
  (pool-only: 31.7), because the old pool served one expert per thread (10 jobs → 22 idle
  cores) and woke through a condition variable. Pool rewritten: a round runs as three
  row-chunked phases (quantise each token once; gate/up rows of every job in 8 chunks; down
  rows in 8 chunks with atomic adds into the token column) so 10 jobs occupy all cores, and
  workers spin ~20k pauses before sleeping. Building + measuring (unit test, idle-host bench,
  parity, probes at 0.5/0.7 share, 1 and 16 users, NUMA-interleaved run).
- Results (idle host): unit test OK; bench 1 thread 13.5 GB/s, 16 threads 157, **32 threads
  209 GB/s** (195 mean; was 154), and the 10-job single-user shape **177 GB/s best / 142 mean at
  0.37 ms per round** (was ~6 ms). CLI with `--cpu-moe-share 0.5`: still "Paris", decode
  **27.4 tok/s** (v1 8.9; pool-only 31.7). Reading: at one user a round misses only ~3–4
  experts (63 % hits), so the host takes ~2 experts (≈70 µs of work) but pays the fixed
  per-layer cost of four D2H copies, a host-function node and the partial-add kernel
  (~0.1 ms × 48 layers ≈ 5 ms/token) — the split cannot pay at one user until v2 overlaps it;
  at 16 users (~150 misses/round) it should. Probes running.
- **v2 overlap implemented (2026-08-28):** the host round now forks onto a side stream after
  resolve/gather (fork event → D2H staging → host-function round → join event) and the MLP
  block's `combine_into` joins it and adds the FP32 partial as an extra term of the
  hyper-connection combine (`hyper_connection_combine(block_output, extra, inject, residual)`),
  so the host computes while the GPU runs the pooled experts. Buffer reuse across layers is
  ordered by the single side stream and the per-block join. v2 build + parity + probes
  (0.5/0.7 share at 1 and 16 users, NUMA-interleaved 16-user run) queued behind the v1 probes.
- Prefill stays on the gather for now: a 512-token prompt routes ~10 tokens to each of the ~512
  experts a layer touches, i.e. ~49 M MACs per expert — at the host kernel's ~2.6 GMAC/s per
  core (it is a DRAM-shaped GEMV, not a GEMM) that is ~19 ms per expert-job, so 50 % of a
  layer on 32 cores would take ~150 ms against the 52 ms gather. Prefill on the CPU needs a
  tiled int8 GEMM (ik's `iqk_mul_mat_moe` shape) and is a separate step.
- **Hang in the v1 probe (2026-08-28, 15:00):** the one-user run stalled on its first
  512-token request for 20 min with the GPU idle and 32 host threads spinning; one pool
  worker was asleep. Cause: the pool published a round (`generation++` + `notify_all`) without
  holding the mutex, so a worker that had just failed its wait predicate but not yet blocked
  slept through the notification, and the coordinator — running inside a CUDA host-function
  callback — spun on the barrier forever, freezing the stream. Fix: publish under the mutex
  and re-notify periodically while waiting; the unit test now runs 3,000 tiny rounds on a
  32-thread pool under a watchdog. Every step of the measurement chain carries a timeout.
- The session's process crashed at ~15:05 while the relaunch waited on the hung server (the
  server needed SIGKILL; SIGTERM was ignored by the spinning callback thread). Resumed at 15:06:
  host idle, GPUs free, the pool fix and stress test were on disk but uncommitted; the chain
  (build → unit+stress test → v2 parity → probes 0.5/0.7 share at 1 and 16 users +
  NUMA-interleaved) was relaunched and the commit is gated on the stress test passing.
- Stress test passed (3,000 rounds, 32 threads, watchdog quiet); pool fix committed (597d01c2).
- v2 CLI parity holds ("Paris"), decode 27.7 tok/s at one user (v1 27.4): as predicted, at one
  user the split is bound by the per-layer host round-trip, overlap or not. The five server
  probes all refused to start: the host-round nodes (four memcpys, a host function and two
  events per layer) grow the captured decode graphs to 21.4 MiB per lane, over the 20 MiB
  Flash-Next allowance (16 lanes → 341.8 MB vs 335.5 MB). Allowance raised to 24 MiB;
  rebuild + the five probes rerunning.
- Policy + trimming for the split (committed, not yet in the probe binary): the split applies
  only to rounds of ≥ `SUROGATE_SERVE_CPU_MOE_MIN_TOKENS` columns (default 4) — at one user a
  round misses ~3–4 experts and the host round-trip (~0.1 ms × 48 layers) costs more than
  the ~2 experts it saves — and the job list is staged with one D2H copy (contiguous device
  block mirrored by a pinned block carved the same way) instead of four.
- **v2 probes (2026-08-28, 512/128, 3,000 slots):** users=1 share 0.5 → 17.1 tok/s (pool-only
  18.5: the split costs at one user, hence the min-tokens policy); **users=16 share 0.5 under
  `numactl --interleave=all` → decode 23.0 tok/s, prefill 92, TTFT 17.1 s** — 2.4× the
  pool-only 9.4, 3× v0's 7.6, level with ik_llama.cpp's 24.0 @16, before NUMA-aware banks and
  share tuning. Three configurations failed to start with `cudaMalloc(pool)` because my chain
  launched the next server before the killed one's GPU memory was released — the rerun waits
  for GPU 1 to be free between servers.
- NUMA placement, refined: the host bank's pinned pages are placed by the driver at
  `cudaHostAlloc` time (one allocation per matrix), not by the copy threads' first touch, so
  node-affine expert halves need a two-allocation bank (experts [0, 256) on node 0, the rest on
  node 1, allocated from threads bound to each node) plus a range-aware gather source and a
  pool whose worker groups are pinned per node and take jobs by expert range. `numactl
  --interleave=all` gives every round both nodes' bandwidth with half the traffic remote; the
  policy rerun measures interleaved vs not at 16 users, and that delta decides whether the
  two-allocation bank is worth its complexity.
- **Policy rerun (2026-08-28 15:49, min-tokens 4, single job copy, 512/128, 3,000 slots):**
  users=1 share 0.5 → 18.4 tok/s (= pool-only 18.5: the split stays out of one-user rounds);
  users=16 share 0.5 → 20.7 plain / 21.7 interleaved (NUMA interleave ≈ +5 %, so the
  two-allocation bank is not a priority); **users=16 share 0.7 → decode 32.9 tok/s, prefill
  132, TTFT 16.3 s** — 3.5× pool-only, 4.3× v0, 37 % above ik_llama.cpp's 24.0. Reading: at
  share 0.5 the GPU gather (~75 experts ≈ 0.4 GB at ~50 GB/s ≈ 8 ms/layer-round) still
  bounds the round while the host does its ~75 experts in ~2 ms; at 0.7 the balance shifts
  toward the host's ~180 GB/s. A share sweep (0.8, 0.9) and 32/64-user probes are queued;
  the bandwidth-matched split of the design (share from measured host vs PCIe rates) is the
  follow-up.
- Bandwidth-matched share (design D4), planned: at startup, with the pool and the split on,
  time a gather of N experts over PCIe and a host round of N experts, and set
  `share = host_rate / (host_rate + pcie_rate)` (clamped to [0.3, 0.9]) unless
  `--cpu-moe-share` is given; log both rates. On this box that formula predicts ~0.78
  (host ~180 GB/s vs PCIe ~50). `--cpu-moe-min-tokens N` (server + CLI) now sets the split's
  minimum round width (env `SUROGATE_SERVE_CPU_MOE_MIN_TOKENS`, default 4).
5. **Prefill**: selective streaming of used experts per layer with whole-layer double
   buffering on a side stream.

Exit for phase 2: the single-GPU board row at 1, 16 and 100 users, ≥ 3× llama.cpp
`--cpu-moe` (7.1 / 16.3 tok/s on this box).

## Phase 3: pipeline parallelism across the 8 cards — design (2026-08-28)

Goal: serve Flash-Next (and any family model) split by layer ranges over N GPUs, with several
micro-batches in flight so every card is busy, the phase-2 offload (slot pool + CPU split)
running inside each stage for what its layers' experts do not fit, and no P2P/NVLink
assumed. Bar: llama.cpp `--split-mode layer` on 8× 5090 — 39.3 tok/s at one user, 28.8 at
32 (its cards run one at a time); target: ~40+ per stream at one user and an order of
magnitude more aggregate at 64 users.

Hardware facts that shape it: GPUs 0-3 sit on NUMA node 0 (CPUs 0-15), 4-7 on node 1
(16-31); every GPU pair is PCIe-only (`nvidia-smi topo`: NODE within a socket, SYS across);
x16 on GPUs 0/1/4/6, x8 on 2/3/5/7; ~50 GB/s host↔device per x16 card. Stage order 0→7
therefore crosses sockets once (3→4), and a hop's pinned staging buffer lives on the socket
of its two GPUs (`numa_alloc_onnode` + `cudaHostRegister`).

What crosses a stage boundary: the family residual after layer `last-1` — for qwen4exp the
four hyper-connection streams, `[4, hidden, T]` BF16 = 20 KB per column (11.8 MB at a
576-column mixed round), plus the round's column metadata (positions, lane ids, segment
facts) which every stage needs to build the same round. Transport is device→pinned→device
(one `cudaMemcpyAsync` each side, events for ordering): ~0.25 ms per hop per direction at
T = 576, negligible against a ~100 ms stage round.

Structure:

1. **Layer-range programs.** `create_program` takes a layer range `[first, last)`: a stage
   with `first == 0` runs embedding (+ PLE at layer 1) and exports the residual after its
   last layer; a middle stage imports the residual and exports; the last stage imports,
   runs its layers, final norm, lm_head and sampling. Each program owns the per-layer state
   of its own layers only (KV pages of its attention layers, GDN slots of its GDN layers,
   PLE state on stage 0), so state memory also splits N ways. Weights: each stage
   materialises only its layers (the registry's per-layer bindings; the host bank stays one
   process-wide object, and each stage's slot pool caches its own layers' experts).
2. **One process, N device contexts.** Stages are threads in one process, each bound to a
   device (its own streams, graphs, workspace, pool, cache): the 154 GB host bank and the
   CPU pool are shared, requests and lanes live in one scheduler. The current-device
   globals (`expert_slot_cache_for_current_device`) already key by device.
3. **Micro-batch pipelining.** Lanes are partitioned into G groups (G = N by default); a
   round is built per group and flows stage 0 → N-1; the executor keeps one round per stage
   in flight, so stage s works on group g while stage s+1 works on group g-1. Decode: group
   g's next token is available when its round leaves stage N-1 and feeds stage 0's next
   round for g — a lane sees one token per G rounds of its group, i.e. per-stream latency ≈
   one full-model round as today, and aggregate ≈ N× a single card. A prompt's prefill
   chunks belong to its lane's group (mixed rounds as now). Graph capture per stage keys on
   the same (columns, band) ladder; capture happens per device.
4. **Per-stage offload.** Each stage decides its residency from its own free memory after
   its KV/state plan: the pool holds what fits of its 6 layers' experts (at W8, ~26 GB per
   stage does not fit, so the pool + CPU split run as in phase 2; at a Q4 bank everything is
   resident). The CPU split's host pool serves all stages: its rounds are sequential per
   layer anyway, and the stages' host rounds interleave; if the host becomes the limit,
   the per-socket split (16 threads per socket, stages 0-3 on node 0) is the lever measured
   in the lanes experiment.
5. **Sampling and streaming** stay where they are (last stage produces logits; the executor
   samples, streams, and hands tokens to stage 0 for the group's next round).

Steps, each with its test:
- A. Layer-range program on one device: run [0, 24) and [24, 48) back to back with the
  boundary export/import and compare token-0 residual and sampled tokens with the single
  program (parity tooling from phase 1; must be bit-exact given identical kernels).
- B. Two devices, one micro-batch: stage 0 on GPU 2, stage 1 on GPU 3 (same socket), then
  GPU 3 → GPU 4 (cross socket); parity again; measure hop cost.
- C. G micro-batches in flight on N devices (2, 4, 8): correctness under load (the
  under-load coherence probes, 0 fatals at 64 users), then throughput scaling at 1 / 16 /
  64 users; compare to the phase-2 single-card rows and llama.cpp's 8-card rows.
- D. Offload inside stages: W8 bank with pools per stage (the memory that does not fit),
  auto share per stage; then the board rows.

Step C3 — steady-state pipeline (design, 2026-08-28 22:05). The C1/C2 driver runs every
executor round as a closed software pipeline (fill, G steps, drain), so a token costs
(G+N−1) stage-round fixed costs instead of one; for a model that fits one card this is
slower than the card. Steady state means a group re-enters stage 0 as soon as its previous
tokens are committed, while the other groups sit at other stages, so the fixed cost is paid
once per group-round and the stages stay busy. That needs the executor to reason per group:

- Lanes are partitioned into G groups (lane % G, G = stages by default, width-limited as
  now). Each group has its own membership, its own staged prefills (a prefill lane belongs to
  its lane's group; a group's round is mixed when it has staged prefills), and at most one
  round in flight.
- The pipeline program exposes `launch_group(g, prefill_lanes, lanes, budgets)` (stage 0
  launch, records the group's in-flight state), `tick()` (advances every in-flight group by
  one stage where the next stage is free: consume stage s — blocking only on the group that
  has been in flight longest — park/copy the boundary, launch stage s+1; returns the groups
  whose last stage completed with their assembled results), and the existing per-lane
  bookkeeping calls (resolve, tokens propagation) applied per finished group.
- The executor's worker loop becomes: admit/top-up as now; for every group without a round
  in flight and with decode-ready lanes (or staged prefills), build its membership and
  launch; `tick()`; for each finished group, run the existing `process_decode_round` /
  prefill resolution on that group's membership and results. A group that has nothing to run
  idles. With N stages and G ≥ N groups the pipeline stays full; with one lane it degrades
  to lockstep (unchanged latency).
- Non-pipeline instances keep the single-round loop (the group machinery compiles in only
  for the pipeline instance type).
- Expected: a token then costs ~one stage round per group, and eight stages pipelined give
  ~N× the per-stage rate at high concurrency; the per-stage fixed cost (~30 ms today, far
  above the stage's GPU work) remains the multiplier on everything and is decomposed by the
  timestamped trace.

Non-goals for phase 3: tensor parallelism (needs all-reduce per layer — the EP argument
applies), expert parallelism (rejected below), P2P copies (not available; the host-staged
copy is the design, and it is also what a multi-host version would use).

## Decisions that shape the code

- **Multi-GPU = pipeline parallelism with the phase-2 offload inside each stage; expert
  parallelism is rejected for PCIe-only hosts (2026-08-28).** Both splits cut per-card expert
  memory 8-way, so the choice is the communication pattern. PP moves one 5 KB activation per
  token per stage boundary (7 hops per round, host-staged, no P2P needed) and scales with
  micro-batches in flight. EP needs an all-to-all dispatch and combine in every layer — 96
  host-staged exchanges with a barrier per round: ~20 ms of exchange floor at decode, and at
  prefill width (T × top-10 × 5 KB each way per layer) more PCIe time than expert compute.
  EP pays only where an exchange costs microseconds (NVLink); revisit on a P2P-capable box.
  PP × EP keeps the all-to-all for a marginal single-user gain and is not worth it. Phase 4
  therefore becomes a measurement only if such hardware appears; phase 3 is the work.

- Experts stay W8G32 in the artifact: regrouping K-quants into the kernels' group-64 formats
  measures 11–12 % rel-L2 error; W8 0.55 %; Q8_0/Q4_0/Q5_0/IQ4_NL repack bit-exactly.
- Hyper-connections live inside the target Variant. The family's `Variant` already owns every
  residual read/write (attention/GDN projections, `gdn_norm_control_projection`,
  `post_mixer`) and the arena scopes per mixer/MLP, so the family only needs a residual-width
  trait (`TextConfig::residual`, default `hidden`) at the residual planes plus hooks with
  bit-identical defaults for the five existing targets: attention-side norm, post-mixer norm,
  embed→residual, final residual→hidden, per-layer prologue (PLE).
- Flash-Next projections compose plain W8 `ops::linear` + `extract_bf16_columns` +
  `causal_conv1d_silu_snapshot` instead of extending the geometry-specialised fused wrappers
  (`attn_input_proj`, `gdn_input_proj`, `linear_add`, the BF16 GDN gating family).
  `gdn_input_projection_record` is speculative-only (no MTP/DFlash for this model) and is
  refused.
- BF16 dense problems (hc, PLE, a_b) run through cuBLASLt with cached plans; prewarm and
  prepare every shape before stream capture.
- Norm convention: GGUF gammas are folded (1+w). Only the attention q/k norms are stored as
  HF-style w because the family applies them with `unit_offset=true`; everything else is
  consumed with the folded gamma (`unit_offset=false`, or the op's own FP32 gamma).
- Sparse MoE: kernel bodies are `*_body.inc` files included once per geometry namespace
  (`geometry_qwen36`, `geometry_flash_next`); the wrapper derives the geometry from the router
  and shared-down shapes plus `SparseMoeWeights::experts_per_token` and refuses anything
  unregistered. Q5/Q6 routed-down paths need intermediate 512 (eight 64-wide groups per row)
  and throw for other geometries; Flash-Next uses W8+W8.
- PLE hash runs on device (ids are device tensors inside captured graphs); per-column metadata
  (segment begin, state slot, segment-last flag) is staged like the family's other round
  inputs. The table is read zero-copy from pinned host memory (52 GB/s measured; 1.4 KB per
  token).
- Expert bank and PLE table are bound ValidateOnly and read with `Reader::read_direct` into
  pinned host memory at load: the `Reader` dies after construction.

## How to run / verify

- Serve on one 5090 with the phase-2 offload (the configuration behind the board rows):
  `numactl --interleave=all surogate-engine <artifact>.ninfer --max-num-seqs 16 --kv-capacity auto
  --max-model-len 2048 --expert-slots 3000 --cpu-moe-share auto` — the pool takes 14.6 GiB
  (use `--expert-slots 2000` for 32-64 lanes so the graphs and KV fit), the decode share is
  measured at startup (host vs PCIe rates, ~0.8 here), the prefill share defaults to 0.5
  (`--cpu-moe-prefill-share 0.7` is better for a single user; 0 turns it off), the host
  pool takes one thread per physical core within the allowed cpuset
  (`SUROGATE_SERVE_CPU_MOE_THREADS` overrides). Diagnostics: `SUROGATE_SERVE_EXPERT_STATS=<rounds>`
  (eager only), `SUROGATE_SERVE_ROUND_TIMING=1`, `SUROGATE_CPU_EXPERT_NO_VNNI=1`,
  `SUROGATE_CPU_EXPERT_TILE=1`.
- Probe at concurrency: scratchpad `probe_slots.sh` (env USERS/SEQS/DUR/PTOK/MTOK/GPU/KV/EXTRA/
  NUMA/MAXLEN/PORT/OUT/SRVLOG) runs the coherence prompts before and *during* the load;
  `lane.sh` runs a list of probes node-bound on one GPU so two lanes share the host.
- Convert: `python -m surogate.serve.tools.convert.qwen4exp.convert --gguf <shard1> --frontend
  models/Qwen3.8-Flash-Next-frontend --out <path>.ninfer --device cuda` (GPU 7 was used).
- Verify an artifact against the GGUF: scratchpad `verify_flash_artifact.py` (decode objects,
  compare with gguf-py dequantisation using the converter's algebra; PLE table byte-compare).
- Build the engine: `cmake --build csrc/build-serve --parallel 32 --target surogate-engine`
  (never while an engine process is live: mmap SIGBUS).
- 35B regression probe: scratchpad `probe_35b.sh` (coherence + loadgen, GPU 1, port 8898).
- llama.cpp oracle: `study/llama.cpp-master/build/bin/llama-server` (readiness = `/health`).

Parity tooling (2026-08-28): `surogate/serve/tools/parity/qwen4exp/` (README inside) — engine
stage dumps via `SUROGATE_SERVE_DUMP_RESIDUAL`, CPU re-derivation of token 0 from the GGUF,
per-head full-vector comparison; llama.cpp's `llama-eval-callback` is the oracle.

## Log

### 2026-08-28

- Converter finished and verified; q/k norms rewritten in place to the HF convention
  (`patch_qk_norms.py`, 24 objects, max error 0) and the convention baked into `convert.py`.
- Disk: purged pip/uv/vLLM/cpptools caches and the launcher's artifact cache, removed the
  superseded HF checkpoints (27B-FP8, both 35B NVFP4 mirrors, five third-party 4B quants);
  242 GB free. Candidates left for the owner: HF `Qwen3.6-35B-A3B` BF16 (67 GB, only needed
  to reconvert the parked 35B), `Qwen3.5-9B` (19 GB), `ro-models/.local/artifacts/ro2/train/tmp_*`
  (≈86 GB of `tmp_` training outputs), `surogate-ro-model/.git` (28 GB), four sibling `.venv`s
  (43 GB), `actions-runner/_work` (13 GB), `LTX-2` weights (52 GB), modelscope datasets (13 GB).
- Sparse MoE stage A (geometry from weights) and stage B (per-geometry instantiation with the
  literal fixes: router dot loop, 9-warp block sizes, `expert * 2*intermediate` rows, top-k
  template, small-T S1 warps and S2 shared-memory batching, select/scan/gather/reduce launch
  shapes, W8 scale staging at non-multiple-of-8 group counts, SM count from the device) both
  build; 35B probe on the new kernels: coherent, 0 fatals, 1,147 tok/s at 32 users /
  `--max-num-seqs 32` (board row is 100 users / 128 lanes / chunk 4096: 1,942).
- Written but unbuilt: BF16 cuBLASLt route, hyper-connection op.
- 35B probes on the geometry-instantiated kernels, all coherent with 0 fatals: 1,147 tok/s
  (32 users, 32 lanes), 846 (100 users, 8 lanes, queue-bound), 1,752 (100 users, 128 lanes,
  chunk 4096, 60 s; the board row is 1,942 at 90 s) — a 90 s run is in flight to close the
  comparison. The kernel arithmetic for the Qwen3.6 geometry is unchanged by the
  instantiation; only launch shapes were generalised.
- Written, unbuilt: `ops/ngram_ple/ngram_ple.cu` (+ API), `core/ngram_ple_state.*`; all new
  sources are listed in `csrc/CMakeLists.txt`. Next: build them, then the family runtime
  residual-width trait and hooks, then the `qwen4exp` target.
- 35B at the board configuration (100 users, 128 lanes, chunk 4096, 90 s, fp8 KV is the
  server default) on the refactored kernels and the F1 family hooks: **1,761 tok/s unbatched,
  matching the board's own unbatched row (1,762)**; the board's 1,942 is the
  `SUROGATE_SERVE_PREFILL_BATCH=4` configuration (a run with it is in flight). No regression.
- Family stage F1 (residual trait + four hooks) builds and is committed (5b1b6efd); the
  existing targets take the default branch of every hook. Next: the per-layer prologue hook
  with PLE column metadata (F2), state-pool wiring (F3), then the `qwen4exp` target.
- Board-configuration probe with `SUROGATE_SERVE_PREFILL_BATCH=4` on the F1 build: 1,927 tok/s
  (board mean 1,942 from passes of 1,960/1,924) — the refactors are throughput-neutral.
- F2 (layer prologue + column staging) committed (b4e13eb6). The `qwen4exp` target is written:
  `api/targets/qwen4exp/package.h`, `targets/qwen4exp/impl/{config.h,package.cpp,variant.{h,cpp},
  load/{bindings.{h,cpp},host_bank.{h,cpp}}}`, registry entries, CMake. Design points: the
  Variant mixes the streams in the norm hooks and keeps the inject gates in a thread-local
  handle for the output projections; projections are plain W8 `linear` + column extraction
  (+ `causal_conv1d_silu_snapshot` for decode GDN); the MoE writes into a zeroed plane and is
  combined into the streams; the GDN control projection is a cuBLASLt BF16 GEMM + `gdn_gating`
  (already 48-head); MTP/DFlash/vision are refused at plan_load.
- The target builds and links (engine executor variant extended for it). llama.cpp's greedy
  answers for the parity prompts (8-GPU layer split, `enable_thinking=false`, 40 tokens) are in
  the scratchpad `bench/oracle_flash.txt`: "Paris"; "Three prime numbers are **2**, **3**,
  and **5**."; "The vast ocean stretches endlessly, its deep blue waters hiding countless
  mysteries beneath the rolling waves."; sky: "Sunlight enters Earth's atmosphere and scatters
  in all directions, with shorter blue wavelengths scattering more strongly ...".
- F3: `DecoderStateSpec::ple` → `plan_ngram_ple_state_pool`, `DecoderState::ple`, slot copy/reset
  helpers used at the six `program_impl.h` sites and the rewrite-checkpoint copy in the text
  context; `ExecutionCore::ple` set at its six initialisers; `configure_text_card` and the
  decode/dflash/mtp cards call `set_ple_state`. Variant supplies `ple_state_spec(slot_count)`.
- Target + F3 committed (9a374d71). First serve attempt of the 163 GB artifact on GPU 1
  (`probe_flash.sh`: `--max-num-seqs 4 --max-model-len 2048`, three greedy prompts, then a
  1-user loadgen) is running; the server option for the context is `--max-model-len`.
- First load attempt crawled at ~67 MB/s: the artifact mapping serves random object reads
  without readahead, so copying 154 GB out of it faulted one page at a time. `HostBank` now
  asks for `MADV_SEQUENTIAL` + `MADV_WILLNEED` over each object before its 16-thread copy;
  the serve was relaunched with that build (the stalled process had to be killed).
- Launcher: `gguf_target_key` maps arch `qwen4exp` → target `qwen4exp`; `ingest.py` fetches the
  model's own frontend from the Hub and runs the GGUF-native converter (95e5a9dc). The cuBLASLt
  BF16 route is prewarmed in `Package::create_program` before the program captures its graphs
  (uncommitted until the next build). The pinned host bank now fills at ~1.3 GB/s (154 GB in
  ~2 min) with the readahead fix.
- First serve attempt: the model loads in ~3 min (5.5 GiB device weights in 2 s, then the
  154 GB pinned bank), then program construction stopped at `gqa_attention: invalid KV cache
  head geometry` — the attention family registered 24q4 (27B), 16q2, 8q2, 16q4 but not the
  Flash-Next pair 24 query / 2 KV heads (group of twelve). Registered `Gqa256_24q2`
  (`GqaGeometry<256, 24, 2, 1>`): the wrapper pairs 24 with 2 when the cache says so, the
  decode/prefill launchers pick it by the cache's KV head count, int8 KV is refused for it
  (its kernels tile at most three query-row tiles), and a lane step of more than 64 query rows
  (width ≥ 6 with group 12) is refused explicitly instead of silently skipped by the BF16
  kernel. Build in flight.
- The server no longer appends the CLI usage after a load/serve failure (dc2996bc).
- With the 24q2 attention geometry the program got further and failed on workspace
  accounting: `workspace arena exhausted ... short by 10,518,528 bytes` = the embedding plane
  `embed_residual` kept alive for the whole forward (2560×2048×2) plus the inject-gate plane
  (4×2048×4) that outlived its scoped reservation. Fix: the embedding is transient (scoped,
  consumed by the broadcast) and the inject gates live in a 1 MB device buffer created in
  `create_program` (`Variant::prewarm_device_scratch`) instead of the arena. Build + probe
  chained in the background.
- With the workspace fix the first forward ran and faulted with `cudaErrorMisalignedAddress`;
  `CUDA_LAUNCH_BLOCKING=1` attributed it to the prefill W8 routed-down MoE kernel: it stages
  eight FP16 scales with one 16-byte `cp.async`, and a 640-wide expert has 20 groups = a
  40-byte scale row, so odd rows sit on an 8-byte boundary (the 35B's 32-byte rows never
  did). Scale rows that are not 16-byte aligned are now staged in 8-byte pieces (both W8
  prefill kernels; the 35B geometry keeps its 16-byte path). Build + launch-blocking probe
  chained in the background.
- **First end-to-end serve of Qwen3.8-Flash-Next (2026-08-28):** loads in 126 s, no faults,
  generates tokens — but the text is garbage ("Ken Ken Ken…"), so the forward has a semantic
  bug. Parity harness: `SUROGATE_SERVE_DUMP_RESIDUAL=<dir>` (Variant, debug-only) dumps the
  residual streams before every layer and the final mixed hidden of the first forwards
  (`compare_dumps.py` prints per-layer stats); llama.cpp's `llama-eval-callback` (built in
  `study/llama.cpp-master/build/bin`) prints every tensor's sum for the same templated prompt
  (`<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`,
  17 tokens, identical ids from the HF tokenizer and `llama-tokenize`). Comparison in
  progress: engine dump before layer L ↔ llama.cpp `l_last-(L-1)`, layer 0 ↔ `hc_init`.
- Parity localisation (2026-08-28): `hc_init` matches llama.cpp exactly (token 0 stream 0
  `[-0.0054, 0.0036, 0.0000]`, sums 45.28 vs 45.08), the residual after layer 0 already differs
  (engine sum 59.47 vs `l_last-0` 54.39; token-0 stream-0 `[-0.0045, 0.0060, 0.0075]` vs
  `[-0.0062, 0.0043, -0.0023]`) and layer 1 diverges massively (74.06 vs 32.14). Token 0 is a
  clean single-token unit test (no cross-token dependence), so the Variant now dumps the
  block intermediates too (`f<N>_L<L>_{mixer,mlp}_{mixed,inject,blockout,combined}.bin`,
  first two forwards, layers 0-1). llama.cpp reference for token 0 of layer 0: attention-side
  `hc_mixed-0` `[-0.2837, 0.4208, 0.0000 … 0.5738, -1.5387, -1.9012]` (sum 1895.9), attention
  `hc_inject-0` logits `[-18.04, -18.65, -11.47, -17.88]`, `linear_attn_out-0`
  `[-0.0390, 0.0326, -0.1039 … -0.0458, 0.0117, 0.2767]`, MLP-side `hc_mixed-0`
  `[-0.2214, 0.3401, -0.6444 … -0.0265, -0.8906, 0.7217]`, MLP `hc_inject-0`
  `[-41.57, -19.83, -6.72, -22.50]`, `ffn_out-0` `[0.0526, -0.0239, 0.0497 … -0.0338, 0.0537, 0.0255]`
  (from `bench/oracle_eval.txt` in the scratchpad; llama.cpp's mix algebra is in
  `study/llama.cpp-master/src/models/qwen4exp.cpp` `build_hc_mix`: per-stream RMS, folded
  gamma over all streams, `lo = silu(down·xn / hc)`, `gate = sigmoid(up·lo)`,
  `mixed = mean_s(xn ⊙ gate)`, `inject = w_inject·xn`, combine `x += out ⊗ 2σ(inject/hc)`).
- Per-block dumps (2026-08-28): the engine's layer-0 attention-side mix and inject gates match
  llama.cpp exactly (`mixed` `[-0.2812, 0.4219, 0 … 0.574, -1.539, -1.906]`, gates
  `[0.0218, 0.0187, 0.1077, 0.0226]` = `2σ(logit/4)` of llama's logits) and the combine is
  self-consistent (`-0.0054 + 0.0218·out`), so the hyper-connection op is right; a NumPy
  re-derivation from the GGUF (`hc_reference.py` in the scratchpad) reproduces llama.cpp's
  values too. **The GDN block output is wrong**: token 0 `[0.039, 0.106, 0.346 … 0.301]` vs
  `linear_attn_out-0` `[-0.039, 0.033, -0.104 … 0.277]`. First confirmed cause: the family's
  three `gated_rmsnorm` call sites always applied the SiLU output gate — Flash-Next's GDN gates
  with the sigmoid (`output_gate_type: sigmoid`, llama.cpp `build_norm_gated`). The gate is now a
  `requires`-probed Variant policy (`Variant::gdn_output_gate`, family default SiLU,
  `residual_policy.h::gdn_output_gate<Variant>()`); rebuild + dump rerun in flight to see whether
  the block output now matches or whether the 3:1 value/key head pairing (converter un-tiles
  llama.cpp's tiled V order to HF grouped order; kernel pairs value head h with key head
  ⌊h/3⌋) hides a second bug.
- With the sigmoid gate the GDN block output changed but still differs (token 0
  `[-0.117, -0.019, -0.158 … 0.231]` vs `[-0.039, 0.033, -0.104 … 0.277]`), so a second defect
  sits in the GDN chain. A NumPy re-derivation of the **whole layer 0 for token 0** from the
  GGUF now reproduces llama.cpp stage by stage (`gdn_reference.py`, `mlp_reference.py` in the
  scratchpad; to be folded into a repo parity tool): qkv/z projection, conv+SiLU, α/β/g
  gating, delta rule with llama.cpp's *tiled* pairing (value head h ↔ key head h mod 16 —
  the converter's un-tiling to HF grouped order, key head ⌊h/3⌋, is the equivalent
  convention the kernel uses), sigmoid-gated RMSNorm, out-projection
  (`[-0.039, 0.034, -0.104 … 0.277]`), combine, MLP-side mix + inject, softmax router
  top-10 `[136, 317, 77, 257, 175, 414, 2, 373, 242, 368]` (weights 0.246 … 0.067),
  routed experts (`ffn_moe_out` `[0.065, -0.032, 0.032]`), sigmoid-gated shared expert,
  `ffn_out` `[0.0545, -0.0226, 0.0504]`, layer output stream 0 `[-0.0062, 0.0044, -0.0023]`.
  The Variant dumps every GDN stage (`gdn_fused`, `gdn_ab`, `gdn_g`, `gdn_beta`, `gdn_final`,
  `gdn_out`) so the first mismatching stage can be read off directly.
- Conv-layout detour (2026-08-28, resolved as a non-bug): the stage dumps match the CPU
  reference through the fused qkv/z projection and the α/β/g gating and diverge at the
  gated-norm output, whose token-0 value is ∝ the direction of the convolved value channels,
  so the convolution weight layout was suspected: the op header documents the weight as
  `[C,4]` while the bindings declare ne `{4, C}`. The kernel is the authority: it reads
  `weight[tap*C + c]` (tap-major, channel fastest) and the 35B recipe writes
  `Transpose(Reshape(conv, (C,4)), (1,0))` — exactly what the Flash-Next converter already
  wrote. A channel-major patch made the output *worse* and was reverted (artifact re-patched
  in place from the GGUF, bit-exact; `verify_flash_artifact.py` now checks the conv object).
  Still open: something between the conv and the gated norm (or the norm itself); next is a
  full-vector, per-head comparison of `gdn_final` and `gdn_fused` (`compare_stage.py`) rather
  than first/last-3 spot checks.
- Full-vector comparison (2026-08-28): with the CPU reference corrected (its "fused" vector had
  used the conv output for the V rows), the engine's fused qkv|z projection matches over all
  16,384 rows (rel-L2 0.002; the artifact's V rows decode bit-exact against the un-tiled
  GGUF — the verifier only samples the first/last 512 rows, `check_rows.py` covers the
  middle). The gated-norm output mismatch has per-*head* structure (cos 0.4–0.9 per head,
  per-dim ratios inconsistent across heads), so the delta-net output `o` itself is off; for
  token 0 that can only come from a non-zero initial state (conv or recurrent slot) or the
  conv/delta ops seeing a different input than the dumped one. A generic family probe
  (`debug_probe<Variant>()`, no-op unless the Variant defines it) now dumps the conv state
  and recurrent state *before* layer 0's ops plus the conv output and `o`. 35B sanity on
  GPU 2 with the refactored family: "Paris" / 2,3,5 / ocean sentence — correct (its
  throughput was measured while GPU 1 ran the Flash-Next CLI, so it is not a board number).
- **Root cause of the remaining layer-0 mismatch (2026-08-28): the RMSNorm kernels load `z`
  only for the SiLU epilogue.** The probes showed the conv output and the delta-net output
  `o` matching the reference per head (cos 1.000, conv state all zero), leaving the gated
  norm. Reading `ops/kernel/rmsnorm.cuh`: every kernel variant guards the gate load with
  `if constexpr (Epilogue == RmsEpilogue::Gated)`, so the new `GatedSigmoid` epilogue saw
  `z = 0` and multiplied by `sigmoid(0) = 0.5`. Check: `0.5 / σ(z)` for head 0 dims 0–2 is
  `[0.555, 0.72, 0.93]`, exactly the observed engine/reference ratios `[0.552, 0.718, 0.925]`.
  Fix: `kRmsEpilogueReadsGate<Epilogue>` (Gated or GatedSigmoid) gates the load in all six
  sites; `test_gated_rmsnorm` now covers the sigmoid gate (incl. the eps-dominated tiny-input
  regime, the unaligned path and the d=1024/4096 kernel paths).
- **Parity reached (2026-08-28, commit below).** With the kernel fix every probed stage of layer
  0 and the entry of layer 1 sits within BF16 noise of the CPU reference for token 0
  (rel-L2: fused projection 0.002, gated norm 0.005, GDN out 0.004, MLP-side mix 0.006, MoE+shared
  out 0.003, layer-1 input 0.003, PLE out 0.003, layer-1 mix 0.005); the residual after layers
  0/1/2 matches llama.cpp's `l_last-0/1/2` (token 0 stream 0: `[-0.0062, 0.0044, -0.0023]`,
  `[-0.0027, -0.0105, -0.0028]`, `[-0.0074, -0.0127, -0.0019]`). Greedy CLI:
  "The capital of France is **Paris**."; served answers: `Paris`; "The first three prime
  numbers are **2**, **3**, and **5**." (llama.cpp: "Three prime numbers are **2**, **3**, and
  **5**."); "The vast ocean stretches endlessly …" (same opening as llama.cpp, then diverges).
  Remaining differences are quantisation-level (W8-requantised experts vs Q4_K, BF16 GEMMs),
  not semantic. Verification pattern worth keeping: token 0 of a fresh sequence is a
  state-free unit test of every block; per-head full-vector comparison beats first/last-3
  spot checks (the "V rows wrong" scare was the reference's own slip, the real defect was
  invisible to spot checks); when a stage matches on `o` but not on the norm output, read the
  kernel — the bug was a `constexpr` guard, not numerics.
- Throughput probing (2026-08-28): the first 32-sequence server refused to start —
  `CUDA Graph preparation consumed 502,936,672 bytes, exceeding the planned allowance of
  402,653,184` — the family reserves 12 MiB of graph memory per decode lane and Flash-Next's
  decode graphs (48 layers, four-stream residual, PLE nodes) measure 15.7 MiB per lane. The
  allowance is now a `requires`-probed Variant policy
  (`ordinary_graph_allowance_per_lane_bytes<Variant>()`, family default 12 MiB, Flash-Next
  20 MiB). 16-user probe (old binary, 128-token prompts / 128 new): decode 8.3 tok/s
  aggregate, prefill 8 tok/s, TTFT 10.1 s — prefill streams every used expert per chunk over
  zero-copy, so it is the worst part of v0 (llama.cpp CPU-MoE: 16.3 decode / 65 prefill at 16
  users, 512/128). 32-user probe, `nsys` decode profile and the board-shape 512/128 probes
  queued behind it.
- During the 32-user run (rebuilt binary) GPU 1 sits at 100 % utilisation with the host
  97 % idle: v0 is GPU-bound on PCIe-latency-bound zero-copy reads inside the MoE kernels,
  not host-bound — the slot pool + bulk gather is the right first move (bulk 16-byte
  coalesced row copies reach 52 GB/s; the kernels' scattered 2.5 KB rows do not). The
  coherence answers on this binary match llama.cpp's verbatim for the primes prompt.
- 32 users (rebuilt binary, 128/128): decode 26.6 tok/s aggregate, prefill 27 tok/s, TTFT
  14.3 s, 32/32 ok. v0 scales with concurrency because the touched expert set per round is
  shared across users (PCIe reads amortise), which is also why the slot pool + CPU compute
  design targets the round, not the token. Board-shape (512/128) probes at 1 and 16 users and
  the ik_llama.cpp CPU-MoE baseline are queued.
- Memory-planner integration for the slot pool (design note): `resolve_kv_capacity` hands the
  KV curve `available_runtime_bytes` after weights; the expert pool takes its share *before*
  the curve (MoE-first against a KV floor, `serve-engine-flash-next.md` D3), as a new field
  on `KvCapacityPolicy` (`expert_pool_bytes`, default from a fraction of free memory), so the
  existing explicit/automatic modes keep their meaning.
- **nsys profile of single-stream decode (2026-08-28, 17-token prompt + 24 new tokens):** the
  zero-copy MoE kernels are ~95 % of GPU time — per layer per token d4 (down) 3.50 ms + d3
  (gate/up) 2.67 ms = 6.2 ms, ×48 layers ≈ 0.3 s/token; the prefill W8 gate/up 24 ms + down
  10 ms per layer for 17 tokens. Effective PCIe rates: d3 32.8 MB in 2.67 ms = 12 GB/s, d4
  16.4 MB (640-byte rows) in 3.5 ms = 4.7 GB/s, against 52 GB/s for bulk 16-byte-coalesced
  row copies. Everything else (hc mix/combine, cuBLASLt, W8 GEMMs) is ~2 %. So the slot
  pool + bulk miss gather is worth ~6× on decode even at a 0 % hit rate, and the CPU compute
  split comes on top. Implementation started: `slot_of_expert` threaded through the
  decode/small-T kernels (null = identity), wrapper accepts a slot pool; prefill kernels next.
- Slot cache code landed unexercised (2026-08-28): `slot_of_expert` threaded through all nine
  row-base sites (decode d3/d4, small-T, prefill q4/w8 gate-up and qx/w8 down; the Marlin
  gate/up route refuses a pool), `api/ops/expert_slot_cache.h` + `ops/expert_slot_cache/`
  (pool over W8 row-split planes, directory with flat ids, one-block resolve with a clock-hand
  LRU and active-round protection, four-bank 16-byte gather with a device row count) — all
  compiled object-only (no link while GPU 1 measures). Next: a round hook in `sparse_moe()`
  (called with the final ids after decode d2 / small-T s2 / prefill select_count) so the
  Variant can resolve + gather before the expert kernels, then the target wiring
  (pool sized from free device memory, per-layer host banks), then measure.
- Board-shape probe (512-token prompts / 128 new, zero-copy v0): users=1 decode 5.2 tok/s,
  prefill 21 tok/s, TTFT 1.79 s (llama.cpp CPU-MoE: 7.1 / 29 / 2.0 s); users=16 decode 7.6 tok/s
  aggregate, prefill 30 tok/s, TTFT 15.1 s (llama.cpp CPU-MoE: 16.3 / 65 / 29 s). v0 is below
  the llama.cpp CPU-MoE bar at both points; the slot cache is the first move.
- Round hook + target wiring written (2026-08-28, compiling object-only): `SparseMoeRoundHook`
  (`sparse_moe(..., hook)`) is called with the round's final ids after decode d2 / small-T s2 /
  prefill select_count of each token slice; `qwen4exp`'s `post_mixer` builds the pooled
  weights for the layer (`expert_slot_weights`) and its hook resolves + gathers; the cache is
  per device, created in `prewarm_device_scratch` when `SUROGATE_SERVE_EXPERT_SLOTS=<slots>`
  is set (env knob for the first measurements; the memory-planner integration and a CLI
  option follow). Each layer's `SparseMoePayload` now carries its layer index. First test:
  `SUROGATE_SERVE_EXPERT_SLOTS=3000 --kv-capacity 4096` on GPU 1 (`probe_slots.sh`), answer
  parity first, then users=1/16 throughput.
- Invariant checked for the pool (2026-08-28): every kernel that reads routed weights through
  `slot_of_expert` does so only for experts that appear in the round's ids (decode/small-T:
  the ids themselves; prefill q4/qx: route jobs exist only for routed experts; prefill W8:
  the grid covers all experts but returns before any weight access when the expert has no
  assignments in the slice), and the hook receives exactly the ids the offsets were built
  from, so an unmapped table entry (-1) is never dereferenced.
- `--expert-slots N` (server and CLI, `EngineOptions::expert_slots`) replaces the env knob:
  `qwen4exp`'s `make_sequence_planner` records it and `prewarm_device_scratch` allocates the
  pool; `SUROGATE_SERVE_EXPERT_SLOTS` stays as the fallback. The pool is created in
  `make_sequence_planner`, i.e. before the engine measures free device memory for
  `--kv-capacity auto` (registry.cpp resolves the KV curve against `current_free_device_bytes()`
  after materialisation), so KV and pool no longer overcommit; the early preflight check does
  not see it (it is computed from planned weight bytes), which only weakens that early check.
- Policy chain complete (2026-08-28): share 0.7 at 16 users with `numactl --interleave=all`
  gives **33.5 tok/s decode / 138 prefill / TTFT 12.3 s** (32.9/132/16.3 s without interleave;
  the bank's +5 % NUMA gain shows up as +2 % end to end). Share 0.5 stays at 20.7/21.7. The
  two-allocation NUMA bank stays deprioritised.
- 32/64-user probes did not start (2026-08-28): 64 seqs — after the 14.6 GiB pool the engine's
  minimum runtime reservation (5.98 GB) does not fit (486 MB free); 32 seqs — graph preparation
  consumed 1004 MB against the planned 32 × 24 MiB. The wider decode lanes cost more per lane
  than the 16-lane measurement (31.4 vs 21.4 MiB), so `Variant::ordinary_graph_allowance_per_lane_bytes`
  is 48 MiB (commit 4b2baf9d); the 64-user probe reruns with `--expert-slots 2000` (9.7 GiB pool).
- `--cpu-moe-share auto` implemented (commit 4b2baf9d, object-compiled, not yet built into the
  binary while the sweep runs): `Variant::prepare_expert_split(model)`, called by
  `create_program` after `prewarm_device_scratch` and before the graphs are captured, times a
  gather of 64 experts of the first MoE layer into the pool (cudaEvents, 4 repeats) and a
  host round of the same 64 experts over 8 tokens (pool.run, 4 repeats) and sets
  `share = host_rate / (host_rate + pcie_rate)` clamped to [0.3, 0.9]; both rates are logged.
  The directory is reset afterwards. `EngineOptions::cpu_moe_share = -1` (server and CLI
  parse `auto`; `SUROGATE_SERVE_CPU_MOE_SHARE=auto`) requests it; an explicit share wins.
- External references (owner-supplied, 2026-08-28; not run here; in BENCHMARKS.md): llama.cpp
  PR #27742 — 4090 + DDR4, UD-Q4_K_XL, `-cmoe -b 4096 -ub 4096`, 28k prompt: 20.8-22.5 decode,
  356-384 prefill; 5090 + 64 GB DDR5, UD-Q2_K_XL: 33-34 decode at 32k, 26 at 131k, ~300
  prefill; 5090 + 64 GB, UD-Q3_K_XL, 18 layers' experts resident, KV q8_0, 128k: 22.1-22.8
  decode, ~765 prefill on 4-5k-token prompts. All single user. Reading: single-user decode is
  bytes-per-expert over the host path (Q2_K_XL ≈ half of Q4_K_XL ≈ 0.6× the W8 bank), so 18.4
  at one user is where a W8 bank lands; a Q4-class host bank (host int4 kernel + 4-bit gather)
  is the single-user lever and is not on the plan yet. The **prefill gap (74 vs 300-770)** is
  the real one: llama.cpp prefills the experts on the CPU with a batched int8 GEMM at
  4096-token micro-batches (compute-bound, ~80 tokens per expert per layer), while this engine
  gathers every touched expert per 512-token prompt (≈211 GB over PCIe ≈ 4 s). Two levers,
  both phase-2: (a) the tiled host int8 GEMM for prefill rounds (the deferred item), and (b)
  wider prefill rounds across users so the per-round gather is amortised (the gather is a
  fixed cost once most experts are touched: ~120 t/s ceiling at 512 columns, ~975 at 4096).
  None of the references report multi-user throughput; 33.5 aggregate at 16 users has no
  comparable there. Long context (their 80k-250k) is untested here (probes run at 2048).
- Share sweep at 16 users (2026-08-28, no interleave): 0.7 → 32.9 / 132, 0.8 → 33.9 / 143,
  0.9 → 33.0 / 152 (decode / prefill tok/s). Flat from 0.7 up: the split is saturated — with
  most misses on the host the round is bounded by the host round's own latency (the GPU side
  finishes first and waits at the join), and pushing the last misses off PCIe buys nothing.
  Consequence for the auto share: the optimum is a plateau, so the bandwidth-matched estimate
  (host 209 GB/s vs PCIe ~50 → 0.8) lands on it; no need for a finer policy. The next step
  up at 16 users is the round cost itself (host round latency: fewer, wider jobs per thread,
  prefetch of the next layer's jobs — or the Q4 host bank that halves the bytes).
- Prefill on the host, written (2026-08-28, unit-tested, unmeasured): the pool groups a
  round's jobs by expert (counting sort) and runs four phases — quantise x per token; gate/up
  row chunks per *expert group* with a 2-rows × 2-tokens AVX-512 inner kernel (the weight
  loads and the fp16 scale conversions are shared by the two tokens, so a row read from DRAM
  once serves every token routed to the expert); quantise h per job; down row chunks per
  expert group with atomic adds. Decode rounds degenerate to the old row-chunked GEMV.
  Engine side: `--cpu-moe-prefill-share F` (`SUROGATE_SERVE_CPU_MOE_PREFILL_SHARE`) gives
  rounds wider than 64 columns their own share, the host staging is sized to
  `prefill_chunk` columns (x 10 MB, out 20 MB, 20,480 jobs at 2048), and `share_for(tokens)`
  picks the share per round. Expectation from the arithmetic: ~10 tokens per expert on a
  512-token prompt lifts the host from DRAM-bound (13 GB/s/core) toward the int16-madd
  compute bound (~3×), and the gather shrinks by the prefill share; the prefill share
  optimum is bandwidth-matched like decode's (PCIe ≈ 4.2 s per 512-token prompt vs host
  ≈ 2-2.5 s → ~0.65).
- Auto-share probe at 16 users came back at 10.0 tok/s (the pool-only number): the split did
  not engage and the server log was overwritten by the next probe. `prepare_expert_split`
  now logs why it did not measure; the rerun keeps its log.
- VOID results (2026-08-28 16:15): the auto-share probe (10.0 @16) and the 32/64-user probes
  (graph allowance errors at 32 × 24 MiB / 64 × 24 MiB) ran a stale binary — plain `ninja`
  in `csrc/build-serve` builds `all`, which does not include `surogate-engine`; the binary was
  last linked at 15:31 with the min-tokens change (so the 33.5 / 33.9 rows are valid; the
  auto-share and allowance changes were never in a running binary). Chains now build the
  serve targets explicitly and print the binary's link time.
- Batched host kernel measured (2026-08-28, `ninfer_cpu_expert_compute_bench`, 32 threads):
  decode shape (160 jobs over 512 experts) 236 GB/s mean (was 209); prefill shape at ~10
  tokens per expert (5120 jobs) **574 GB/s-equivalent, 2.4×**; at ~40 tokens per expert
  (20,480 jobs) **900, 3.8×** — the host is compute-bound there (~0.9 TMAC/s with the
  int16-madd inner loop). Auto share works on the relinked binary: host 199 GB/s vs PCIe
  gather 52 GB/s → 79 %, 32.3 tok/s @16 (plateau; TTFT 8.6 s, 29 requests vs 21 at 0.7
  fixed). The 32-user probe at 3,000 slots fails on the runtime reservation (4.0 GB needed
  after the pool, 486 MB free) — it reruns at 2,000 slots like the 64-user one.
- Prefill metric correction: the board's "prefill tok/s" is loadgen's prompt tokens ÷ run
  wall time (decode included), not a prompt-processing rate; the comparable single-user
  number is 512 ÷ TTFT ≈ 174 t/s (ik_llama.cpp ≈ 285), so the gap to the external
  references is 2-4.5×, not 4-10×. Ceiling of the bandwidth-matched prefill split at 512-token
  prompts: ≈ 340 t/s with the int16 inner loop, ≈ 450-500 with VNNI; 765 (5090, Q3, 18
  layers resident, 4-5k prompts) is a longer-prompt regime (4× the per-expert reuse) and is
  to be measured at that prompt length.
- VNNI inner loop written (2026-08-28, unbuilt): `vpdpbusd` u8×s8 with the weights as the
  unsigned operand (sign-bit flip) and the token's per-group 128·Σq compensation subtracted
  in int32 before the float scale, so the numerics equal the int16 path; takes the batched
  2×2 calls when k % 64 == 0; `SUROGATE_CPU_EXPERT_NO_VNNI=1` vetoes it (A/B in the bench).
  This EPYC 9124 pair has avx512_vnni and avx512_bf16.
- 64 users (2026-08-28, share 0.7, interleave, 2,000 slots so the 64-lane graph allowance and
  the KV plan fit beside the pool): **86.5 tok/s aggregate decode, prefill 346 (loadgen),
  TTFT 18.6 s, 68 requests, 0 errors** — 2.6× the 16-user number. The host round's cost is
  per distinct expert, not per token, so wider rounds amortise it; the plateau seen in the
  16-user share sweep was the per-round bound, and concurrency is the way past it on this
  path. 32 users at 3,000 slots fails on the runtime reservation; it reruns at 2,000.
- Prefill split defect (2026-08-28): the first prefill-share probes answered `!!!!` (NaN) with
  TTFT 0.2 s — VOID rows. Cause: the four prefill W8/Qx kernels looked up `slot_of_expert`
  but did not skip an unmapped expert (slot -1 = computed on the host), so they read a
  negative row base; the decode kernels had the `block >= 0` skip, the prefill ones did not.
  Fix (this commit): whole-CTA `continue`/`return` on `block < 0` in all four, and
  `grouped_io` zeroed between the gate/up and the down launches when a slot table is in use
  (the down kernels write per-assignment rows that the reduce sums, so a skipped expert's
  rows must be zero; 26 MB at 5,120 assignments). Small-T rounds (s2→s3) still need the
  same audit before the split is allowed there beyond 64 columns.
- Small-T audit (2026-08-28): small-T's s3/s4 call the decode launchers
  (`sparse_moe_decode_launch_d3_small_t` / `d4_small_t`) whose kernels carry the `block >= 0`
  guard, so the prefill kernels were the only unguarded schedule; all three now skip
  host-routed experts.
- VNNI A/B (2026-08-28, bench, 32 threads, mean over rounds): decode shape 236 vs 240 GB/s
  (bandwidth-bound, no change as expected); 10 tokens/expert **872 vs 771 GB/s-eq (+13 %)**;
  40 tokens/expert **1,083 vs 897 (+21 %)**. Far below the 1.5-1.8× the instruction count
  suggests — the batched kernel is bound by something other than the dot (the per-pair
  permutes and the compensation, the atomic adds of the down phase, the phase barriers, or
  the per-job quantisation); a `perf` profile of the bench is the next step there.
- Prefill split, corrected binary (2026-08-28): share 0.5 at one user answers correctly and
  **TTFT 1.865 s (from 2.95 s) → 275 t/s prompt processing (from 174), 1.58×**; loadgen
  decode 22.4 (18.4) because the prompt phase takes less of the wall. Share 0.7, 16 users, 32
  users at 2,000 slots and the 4k-prompt pair follow in the same chain.
- Prefill split, share 0.7 at one user (2026-08-28): **TTFT 1.43 s → 358 t/s prompt processing
  (2.06× over the full gather's 174)**, answers correct; level with the 4090/Q4_K_XL llama.cpp
  reference at this prompt length, as the arithmetic predicted (~340).
- 16 users with the prefill split collapsed (2 ok / 2,576 errors): a garbage token in a
  *mixed* round (`kind=mixed batch=2 prefill_lane=3 lanes=16`) killed the worker loop. Root
  cause: the host-round staging was per hook call, but prefill and mixed rounds reach the hook
  in several slices — the second slice's host round zeroed the first's partial in the shared
  `out_host` and its job-list copy could race the next resolve. Fix (this commit): the split
  decision is per round (`begin_round` in post_mixer), each slice is staged at its column
  offset in a round-wide `x_host`/`out_host` (sized prefill chunk + 256 lanes), one host
  function per slice (ring of contexts), the main stream waits for the slice's copies before
  the next resolve, and the combine joins the last slice.
- 32 users at 2,000 slots, no prefill split: 30.9 tok/s (16 users: 33.5; 64 users: 86.5). On
  the 512/128 shape the 16-32-user numbers are prefill-dominated: 21-34 prompts × ~1.5-3 s of
  prefill fill most of the 90 s, and decode lanes only ride along in mixed rounds. 64 users
  gain because more lanes ride each prefill and because rounds of 47+ columns use the
  prefill (GEMM-shaped) kernels instead of small-T. So the prefill split is the lever for the
  board numbers at every concurrency, not just TTFT.
- 4k-token prompts are refused: `max_context exceeds the variant native context capacity` —
  `dense_exact_context = 2051` (indexer_top_k + indexer_block − 1); beyond it the attention
  layers need Flash-Next's sparse indexer, which the engine does not implement yet. The
  long-prompt comparison with the external references (their 4-28k prompts) waits on that;
  it is the next model-contract item after the prefill path.
- Host kernel profile (`perf`, 5,120-job shape): ~80 % of round time inside
  `dot_two_rows_two_tokens_vnni` at ~7.8 MAC/cycle/core, 8× under `vpdpbusd` peak — the
  per-group fp16 scale costs a cvt + fmadd + scale permute per 64 MACs per (row, token). The
  way past it is ik_llama.cpp's interleaved-rows layout (16 rows per zmm lane so one
  `dpbusd` advances 16 rows and the scale applies per 16 rows per group), which needs an
  on-the-fly repack of each expert chunk into a thread-local tile (repack ≈ one GEMV pass,
  then ~3× per token; pays off from ~4 tokens per expert). Queued after the mixed-round fix.
- Multi-slice staging, measured (2026-08-28, prefill share 0.7, decode share 0.7, interleave):
  1 user TTFT 1.44 s (as before, 23.9 decode); **16 users: 0 errors, TTFT 12.3 → 5.9 s,
  completions 21 → 39, prefill 138 → 177 loadgen, decode 33.5 → 29.5** — total tokens per
  second 171 → 206 (+20 %) although the decode column reads lower: the host is now the
  critical path of the mixed rounds (0.7 of ~450 prefill misses per layer on the host vs 0.3
  over PCIe), so the decode lanes riding along wait for it. A lower prefill share at high
  concurrency (0.4-0.5) should rebalance; sweep queued. 32 users at 2,000 slots: 23.2 /
  154, 44 completions, TTFT 20 s (30.9 / 136 / 34 without the prefill split).
- 64 users with the prefill split corrupted again (decode round of 40 lanes, after mixed
  rounds): the slice contexts came from a 32-entry ring, but a hook runs at *graph capture*
  and its context pointer is baked into the host-function node — 48 layers × slices × the
  graph ladder overrun the ring, so a replay read another slice's offset. Fix (this commit):
  one address-stable context per distinct (layer, offset, tokens), deduplicated in a deque
  and shared by every graph/eager round with that slice shape (identical triples have
  identical semantics). The earlier per-layer design was accidentally safe: its only
  replay-time field was `round_tokens`, and a too-large value merely computed unused columns.
- Tile (interleaved-rows) kernel measured (2026-08-28): a wash — best-of-round 880 vs 954
  GB/s-eq at ~10 tokens per expert and 1,302 vs 1,218 at ~40 (means too noisy to read). The
  repack gathers 4 bytes per row with scalar loads (128 per 64-byte vector), which costs
  about as much as the reuse saves at these group sizes. Opt-in now
  (`SUROGATE_CPU_EXPERT_TILE=1`); a vectorised repack (transpose via `vpunpck`/`vpermt2d`
  from 16 row loads) is the follow-up if the host kernel becomes the limit again.
- Stable contexts + prefill split, measured (2026-08-28): 1 user, prefill 0.7 → TTFT 1.33 s;
  **16 users, prefill 0.5 → 36.0 tok/s decode, 159 prefill, TTFT 5.8 s, 34 completions** —
  the best 16-user point (33.5 / 12.3 s without the prefill split; 29.5 / 5.9 s at prefill
  0.7): at concurrency the lower prefill share keeps the host off the critical path of the
  mixed rounds. 64 users (corruption check), 0.8 @1 and 0.7 @16 follow.
- 64 users with the prefill split (2026-08-28): **clean** — 0 fatals, 0 errors, 66
  completions — so the stable contexts hold. But 32.7 decode / 164 prefill / TTFT 28.9 s
  against 86.5 / 346 / 18.6 s without the prefill share, and the cause is in the slices fix,
  not the split: the staging copies stayed on the side stream and the main stream waited on
  a `copied_event` recorded after them — in stream order that event sits behind the previous
  layer's host function, so every layer's GPU work waited for the previous layer's host round
  and the v2 overlap was gone (the 16-user 36.0 was reached despite it). Fix (this commit):
  the two copies run on the main stream (stream order alone protects the job list from the
  next resolve), only the host function is forked. Rerun at 1 / 16 / 64 users follows.
- Copies on the main stream, measured (2026-08-28, binary 17:40): 1 user prefill 0.7 → TTFT
  1.41 s; **16 users prefill 0.5 → 37.9 tok/s decode, 172 prefill, TTFT 7.3 s** (best 16-user
  decode); **64 users prefill 0.5 → 33.4 / 157 / 27.5 s, unchanged** — so the serialisation
  was not what costs the 64-user run (86.5 / 346 / 18.6 s without the prefill share). The
  arithmetic says the split should shorten a 576-column mixed round (gather 3.6 s → 1.8 s
  with the host taking the other half in ~1.5 s at the bench rate), so either the host round
  runs far slower in situ than in the bench at that width, or the decode rounds changed too.
  Measuring instead of guessing: a same-binary pair at 64 users, prefill share 0 vs 0.5, with
  `SUROGATE_SERVE_ROUND_TIMING=1` (per-round-kind wall clock) is queued.
- 64 users, prefill share 0.7, copies-on-main-stream binary: corrupted again (fatal in a
  mixed round; 0.5 was clean twice). Share-dependent → a race that widens with the host
  round's length: the pinned job-list mirror was one buffer, and with the copies on the main
  stream slice k+1's copy can land while slice k's host function still reads its jobs (the
  side-stream sequencing that used to prevent it was what serialised host and GPU). Fix
  (this commit): one pinned mirror per slice ordinal within the round (8), owned by the
  slice context (keyed by layer, offset, tokens, ordinal); x_host/out_host were already
  per-offset and the next round starts only after the combine joined the last slice.
- **VOID: the 86.5 tok/s at 64 users** (2026-08-28, found by the round-timing control): the same
  configuration on the current binary gives 37.0 / 174 / TTFT 24.3 s with the same number of
  completions (67 vs 68). The 86.5 run used the 16:15 binary, before the prefill-kernel skip
  fix — in 47-64-column rounds the prefill kernels read garbage for host-routed experts, and
  garbage outputs run to `max_tokens` (128) where correct answers stop early (~50 tokens), so
  the decode count was inflated ~2×. The 1-user probes could not catch it (T = 1 rounds run
  the decode kernels, which had the guard). Consequences: the true concurrency curve without
  the prefill split is flat — 33.5 @16, 30.9 @32, 37.0 @64 — prefill-dominated everywhere,
  and the "47+ columns are superlinear" reading is withdrawn. The 64-user "regressions" of
  the slices/copies fixes were comparisons against this void number; those fixes stand on
  their own evidence (0 fatals, correct answers). Board corrected. Lesson: a throughput row
  needs its completions × tokens sanity-checked, and correctness probes must run at the
  concurrency being measured.
- Round-timing pair at 64 users (2026-08-28, binary 17:40, 32 threads): prefill share 0 →
  37.0 / 174 / TTFT 24.3 s; prefill share 0.5 → 33.3 / 156 / 25.8 s. The timer reports every
  round as "decode" at this concurrency (a prompt's chunk rides inside the lane rounds), so
  it does not separate the prefill cost; rounds run ~0.7-1 s. Reading: the prefill split is
  a clear win at 16 users (37.9 vs 33.5, TTFT halved) and slightly negative at 64, where the
  host is already the round's critical path with the decode share; the default stays 0.5
  and the 64-user trio (0.7 / 0.5 / 0) on the per-slice-mirror binary follows in lane A.
- Parallel lanes (2026-08-28, two node-bound instances at once — GPU 2 + NUMA node 0, GPU 3 +
  node 1, 16 host threads each, cpuset-aware pinning; the binary with per-slice job mirrors):
  **64 users at prefill 0.7: 0 fatals, 0 errors** — the share-dependent corruption is closed.
  Numbers are about half the single-instance ones, as each lane has half the cores and the
  two share DRAM (16.6 / 19.2 at 64 users for 0.7 / 0.5; 18.9 / 23.6 at 16 users for 0.5 /
  0.3; 12.0 and TTFT 2.4 s at one user), so lanes are for A/B sweeps and correctness at
  concurrency, not board rows. Two instances is the host's limit: each pins its own 154 GB
  bank (331 GB used with two). Reading: the prefill-share optimum tracks host strength (0.3
  on 16 threads, 0.5 on 32, 0.7 for a single user on 32), so with `--cpu-moe-share auto` the
  prefill share now defaults to the measured decode share − 0.3 in [0.2, 0.7] unless given.
- **Phase 2 closed (2026-08-28, binary 18:38, commit 9cc4cdd2 + docs):** 35B regression at
  the board config 1,769 decode / 7,076 prefill / TTFT 129 ms (board: 1,762 unbatched).
  Flash-Next on the defaults (`--expert-slots 3000 --cpu-moe-share auto`, interleave): 1 user
  (prefill 0.7) 22.4 tok/s, TTFT 1.40 s (366 t/s prompt processing); 16 users 32.2 tok/s, 174
  prefill, TTFT 9.0 s, 37 completions, under-load answers correct, 0 fatals (auto measured
  host 184-208 GB/s vs PCIe 52 → 78-80 % decode share, 50 % prefill). Run-to-run spread at 16
  users is ~±8 % (32.2 / 36.0 / 37.9 for near-identical configs). Left for later inside
  phase 2's scope: Q4 host bank (single-user decode), vectorised tile repack, the sparse
  indexer beyond 2,051 tokens. Phase 3 starts.
- Phase-3 bar completed (2026-08-28): llama.cpp 8× 5090 layer split at 16 users 39.1 tok/s,
  TTFT 86 s, 16 of 48 requests timed out; at 64 users 24.7, TTFT 311 s. Its cards run one at a
  time, so aggregate throughput does not rise with users and the queue explodes.
- Phase 3 step A infrastructure written and compiling (2026-08-28): `StageSpan` on the family
  `TextContext` (layer loop `[first, last)`, residual import instead of embed, residual export
  instead of finish/head, in all six forward entries and both graph bodies; head epilogues
  (scatter/sample/egress) skipped on stages without the head; chained decode rounds refused
  for stages), carried by `ExecutionCore` and set by every card configuration site;
  `EngineOptions::pipeline_stage_first/last`, `pipeline_import_pinned`,
  `pipeline_boundary_columns` flow through the sequence plan into the program, which
  allocates its pinned export buffer before graph capture and exposes it
  (`Program::stage_export_buffer`). The host bank (`HostBank::shared`, keyed by object
  names and sizes) and the CPU pool are now process-wide, so N stage instances pin the
  experts once. Next: `--devices`, N instances in the engine, and the `PipelineProgram`
  wrapper that drives the stages in lockstep behind the unchanged executor.
- Phase 3 v1 written (2026-08-28, commits b0727deb..70d6d67a): `--devices A,B,...` builds one
  stage instance per device (`construct_pipeline_target`: even layer split, one artifact
  reader, the host bank and CPU pool shared, later stages pinned to stage 0's resolved KV
  capacity, each stage's pinned export buffer handed to the next as its import);
  `PipelineProgram` presents the family `Program` interface to the unchanged executor and
  replays every call on every stage in order (device selected per call), sampling on the
  last stage; `PipelineRequestMemory` activates every stage's transient region;
  `PreparedPrompt::clone` gives each stage its own prompt; stages skip the chained decode
  family and run single rounds; head-less stages' placeholder ledger tokens are overwritten
  with the last stage's samples after every round (`Program::replace_pending_tokens`).
  Lockstep: no overlap yet (step C). First test running: greedy parity of a 2-stage pipeline
  on GPUs 2+3 against one device, CPU split off so the gather is deterministic.
- First 2-stage run (2026-08-28, GPUs 2+3, layers [0,24) / [24,48), CPU split off): the stage
  programs construct, capture their graphs, and the served answers are correct ('Paris',
  the primes identical to one device; the ocean/sky sentences equal a previous single-device
  run's — the single-device path itself varies run to run, control queued). Two defects on
  the way: kernel attributes (`cudaFuncSetAttribute` for dynamic shared memory) were set once
  per process under function-local statics, so the second device's launches failed with
  `cudaErrorInvalidValue` — now once per (device, kernel) via
  `ops/kernel/func_attribute.cuh` (15 sites); and the driver's token propagation refused
  mixed-round results, which carry one token per row and no counts. Lockstep cost at one
  user: 4.3 tok/s against 11.5 on one device (each stage pays the full per-round fixed cost
  and the stages run one after the other) — step C's overlap is what pipelining is for.
- Step B evidence and step C1 written (2026-08-28): 2 stages serve correctly at 1 and 4 users
  (4 users, split on: 32.7 tok/s, 16 completions, 0 fatals, correct answers under load);
  lockstep at one user 10.2 vs 11.5 tok/s on one device. Text parity is inconclusive by
  construction — two identical single-device runs already differ (the prefill MoE kernels
  assign work with atomics, so the reduce order and the bf16 roundings vary); the numeric
  token-0 dump comparison (final residual of forwards 0/1, one device vs two stages) is the
  parity test and is running. Step C1 (commit 6a64db16): `PipelineProgram::decode_batch`
  partitions the round into `groups` micro-batches (lane % groups, default = stages,
  `SUROGATE_SERVE_PIPELINE_GROUPS`), and runs them as a software pipeline over the stages
  with the program's launch/consume seam: at step t stage s consumes the group it launched at
  t−1 (parking its export in a per-(boundary, group) host slot; the last stage's tokens go
  to the assembled result) and launches group t−s after copying the previous stage's parked
  export into its own import buffer — consuming stage s before launching stage s+1 orders the
  data while the other stages keep running. Stages now own their import buffers. Mixed
  rounds stay lockstep per round (C2 splits them). Test running: one device vs lockstep vs
  pipelined at 8 users on the decode-heavy 128/512 shape (GPUs 2+3).
- **Step B parity: bit-exact** (2026-08-28). Token-0 dumps of the prompt forward ("The capital
  of France is", 17 columns, CPU split off) from one device and from the 2-stage pipeline
  (GPUs 2+3, boundary after layer 24) are identical at every layer boundary — all 48
  `f1_layer*.bin` at rel-L2 0, including layer 24 (the imported residual) — and at
  `f1_final`. The residual crosses the pinned boundary exactly. (The CLI run with dumps then
  aborted at the first decode step with an illegal address surfacing in the dump helper;
  the serving path decodes correctly with graphs, so this is being localised with
  CUDA_LAUNCH_BLOCKING on the eager path.)
- Two more per-process statics found by the eager 2-stage run (2026-08-28): the Marlin
  scratch (`g_scratch`, one buffer on the first device → illegal address on the second) is
  now per device; and `launch_ordinary_round`/`consume_ordinary_round` were private for the
  other variants (the C1 build had failed silently behind a stale binary; the chain now
  prints the binary's link time and the build exit). Step C2 (commit dd6956a8):
  `advance_prefill_mixed` is split at its synchronize into `launch_mixed_round` (staging,
  forward, sample, egress copy; the eager fallback still synchronises inside) and
  `consume_mixed_round` (synchronize + bookkeeping) with the in-flight record kept on the
  program; the driver's `run_grouped_round` pipelines every executor round: the prefill
  lanes all ride with the first non-empty decode group as the mixed round (so the executor
  sees exactly one mixed result), the other groups run decode-only rounds, and the software
  pipeline over stages is the same as C1's. Measurements queued: one device vs lockstep
  (groups=1) vs pipelined at 8 users on the decode-heavy 128/512 shape, C1 then C2.
- Eager 2-stage path verified after the Marlin fix (2026-08-28): the traced CLI run on GPUs
  4+5 generates through both stages (deferred start, one prefill chunk per stage, decode
  rounds stage 0 → 1), exit 0. C1 at 8 users on 128/512 (GPUs 2+3): one device 36.9 tok/s;
  **two stages in lockstep 72.5 tok/s** — 2× before any overlap, because each stage's pool
  caches only its own 24 layers' experts (3,000 slots per stage = twice the resident set),
  so misses and host rounds halve per stage. The pipelined number follows.
- First pipelined (groups=2) run hung under load (2026-08-28): server up, probes correct, no
  completions. With two groups in flight the two stages' CPU host rounds run at the same time
  on CUDA driver threads, and the shared `CpuExpertPool::run` is not re-entrant (its
  generation/phase protocol assumes one caller) — the workers waited on a phase that never
  came. Fix (commit 80bc1b9c): `run` takes a mutex, so the stages' host rounds alternate on the
  same cores; per-socket pools per stage group remain the option if the host becomes the
  limit. Other shared host state audited: the bank is read-only, the caches and slice
  contexts are per device, the thread-locals live on the driver thread only.
- 2 stages at one user, split on (2026-08-28): 13.1 tok/s, TTFT 4.3 s (one card: 22.4 / 1.4 s).
  With one lane there is one group, so the stages run in lockstep and every token pays two
  rounds' fixed costs plus two serialised host rounds; the prefill likewise runs stage after
  stage. Per-stream latency is not what the pipeline buys — aggregate throughput is — and
  the single-stream case will need the two stages' host rounds to run on separate sockets
  (per-socket pools) to get back to parity with one card.
- **8 stages construct and answer** (2026-08-28, first try, pre-C2 binary): `--devices
  0,…,7` builds eight stage instances of six layers each on all cards; each stage measures
  its own split (x16 cards: host ~195 GB/s vs PCIe 52 → 79 %; the x8 cards 2/3/5/7: PCIe 26
  → 88 %), and the served answers are correct ('Paris', the primes, the ocean sentence).
  The 1/16/64-user runs on the C2 binary follow the C2 comparison on GPUs 2+3.
- Per-socket host pools (2026-08-28, unbuilt into a binary yet): with stages on both sockets
  the shared pool serialised every stage's host round on the same 32 cores. The pipeline
  constructor now sets `EngineOptions::cpu_moe_pool_per_socket`, and each stage's cache
  takes the pool of its GPU's NUMA node (PCI `numa_node` → the node's physical cores from
  `/sys/devices/system/node/nodeN/cpulist`, first half on SMT-2), so stages 0-3 and 4-7 run
  their host rounds concurrently on their own cores and memory (16 threads each; the
  single-card server keeps the one 32-thread pool). `SUROGATE_SERVE_CPU_MOE_POOL_SHARED=1`
  restores the shared pool for A/B.
- Goal widened by the owner (2026-08-28 21:05): phase 3 = Flash-Next, Qwen3.8-27B and
  Qwen3.6-35B-A3B served on all 8 GPUs with pipeline parallelism. The stage machinery is the
  family's (`StageSpan` in `TextContext`, the driver templated on the stage instance), so
  wiring the two other targets is the registry's constructor templated over (Target, Loaded,
  Instance, layers) plus their executor variants (commit after fe9af1be). v1 caveat for the
  dense/resident targets: every stage materialises the whole model and plans all layers'
  state (the 27B is 15 GB per card, the 35B 21 GB), so memory does not shrink with the stage
  count yet — layer-subset materialisation is the follow-up; the throughput win comes from
  the stages running in parallel on different micro-batches. Queued after the Flash-Next
  chains: 27B and 35B as 8-stage pipelines at one user and the 100-user board shape.
- C2 measured (2026-08-28, GPUs 2+3, 8 users, 128/512, split on): one card 42.2, two stages
  lockstep 68.9, two stages pipelined (2 groups) **68.3** — no gain from the overlap, no hang,
  answers correct. Reading: with the decode share at ~0.8 the host round is each stage's
  critical path, and the stages' host rounds serialise on the shared pool (both cards are on
  socket 0, so per-socket pools cannot separate them either); the GPU sides overlap but the
  host path does not. The 2× over one card is the residency effect. Two A/Bs isolate the
  overlap and are queued after the 8-card runs: split off (GPU + PCIe only), and a
  cross-socket pair (GPUs 3+4) with per-socket pools.
- **First 8-card and 4-card Flash-Next numbers** (2026-08-28, C2 binary, shared host pool,
  512/128): 8 stages — 1 user 4.3 tok/s / TTFT 14.3 s; 16 users 32.8 / 14.1 s (32 completions,
  admission timeouts); 64 users did not fit (runtime reservation 7.6 GB vs 4.9 GB free beside
  the 14.6 GiB pool and 64-lane graphs). **4 stages on GPUs 4-7, 16 users: 79.7 tok/s, TTFT
  1.3 s, 65 completions, 0 errors — 2.4× one card.** Reading: with 8 groups at 16 users a
  round carries 2 lanes, so every step pays a full round's fixed costs eight times over, and
  every stage's host round queues on the single shared pool; 4 stages put 4 lanes per round
  on one socket. Fixes queued: groups follow the round's width (a minimum lane count per
  group, `SUROGATE_SERVE_PIPELINE_MIN_LANES`, default 4), 2,000 slots at 64 lanes, and the
  per-socket pools (building now).
- Per-socket pools measured (2026-08-28): 8 stages at 16 users 31.8 tok/s, TTFT 18.8 s; at one
  user 3.5 tok/s — no change, so the shared pool was not the limiter. The reading that fits
  all the 8-stage numbers: **the per-round fixed cost.** At one user a token is eight
  sequential stage rounds of ~36 ms although each stage runs a sixth of the layers (a single
  card's whole round is ~45 ms), and the mixed (prefill) rounds cross the eight stages one
  after another, so TTFT doubles and admission queues. Two responses, committed: a stage
  whose pool holds ≥ 90 % of its experts (8 stages: 3,000 of 3,072) switches the CPU split
  off — the host round trip per layer buys nothing there; and the pipeline trace now carries
  timestamps so the next pipelined run shows whether stage rounds actually overlap. The
  remaining levers are prefill batching (several prompts per mixed round: the executor's
  `mixed_prefill_batch_target`) and cutting the round's fixed cost itself.
- **27B and 35B-A3B serve on 8 stages** (2026-08-28, pp9 binary): 27B — 1 user 19.6 tok/s,
  TTFT 0.43 s; 100 users 213 tok/s, prefill 852, TTFT 0.75 s, 207 completions, some admission
  timeouts. 35B-A3B — 1 user 50.1, TTFT 0.35 s; 100 users 574, prefill 2,295, TTFT 0.34 s,
  469 completions. Correct answers, 0 fatals, 17.8 GB per card for the 27B. Against one card
  (1,330 / 1,942) the eight-card pipeline is 6× slower, and the arithmetic explains it
  exactly: `run_grouped_round` fills and drains the pipeline inside every executor round —
  G+N−1 = 15 steps for 8 groups over 8 stages (efficiency 8/15) — and each step costs one
  stage round's fixed part (~30 ms), so a token costs ~15 × 30 ms ≈ 0.5 s for all lanes: 100
  lanes / 0.5 s ≈ 200 tok/s, as measured. One card pays the fixed cost once per token
  (~40 ms for 100 lanes). Therefore: (1) the pipeline must run steady-state across executor
  rounds — each group is its own round that re-enters stage 0 as soon as its tokens are
  committed, which is the executor change deferred in step C: G independent round contexts
  instead of one membership; (2) the ~30-36 ms stage-round fixed cost (a 6-layer stage at
  one lane) is far above its GPU work and is the other half — the timestamped trace run is
  queued to split it between device time (consume blocking) and host gaps.
- Step C3 written (2026-08-28, commit above): the driver keeps one flight per group (decode,
  mixed, or a lone prefill run stage by stage synchronously), `tick()` consumes only the
  stage round that has been running longest, parks its export, and launches every parked
  group whose next stage is free (oldest first); the executor, for a pipeline program, runs
  `pipelined_iteration`: continuous CPU-only admission, then a launch for every idle group
  with work (its own membership by lane % groups, its own staged prefills), then one tick,
  then the usual per-round processing for each finished group. The closed-pipeline
  `decode_batch`/`advance_prefill_mixed` stay for the non-pipelined executor path. Test chain
  queued behind the trace run: 2 stages split-off vs the closed pipeline, then 8 stages at
  16/64/1 users, then the 27B and 35B at 100 users.
- Width-following groups + residency policy measured (2026-08-28, 8 stages, closed pipeline):
  **16 users 57.8 tok/s, TTFT 0.6 s, 48 completions** (31.8 / 18.8 s before). Rounds of 4
  lanes instead of 2 halve the fixed-cost multiplier and the stages skip the host round
  trips at 98 % residency. Still under the 4-stage 79.7 (fixed cost × 8 stages, fill/drain);
  the steady-state pipeline (C3) is what removes the fill/drain.
- Overlap A/Bs read (2026-08-28, closed pipeline, 8 users, 128/512): split off on GPUs 2+3 —
  lockstep 44.5 vs pipelined 44.8; cross-socket GPUs 3+4 with per-socket pools — 75.9 vs
  70.0. Neither is a driver defect: GPUs 2 and 3 are the x8 cards (26 GB/s each, evidently
  on shared lanes), so with the split off both stages' expert gathers contend on one link and
  cannot overlap; on the 3+4 pair the x8 stage is the bottleneck, and with two groups
  lockstep R3+R4 equals the pipeline's 1.5·R3. The pipeline pays where stage rounds are
  balanced and their bottleneck resources are independent — the 8-stage, high-residency
  case, which is C3's target. 8 stages at 64 lanes failed on stage 7 with
  `cudaGraphExecUpdate` result 5 (parameters changed) while instantiating its graph ladder;
  `DecodeGraphExecutable::update` now re-instantiates the executable when the driver
  refuses an in-place update (logged once), so the profile still runs as a graph.
- Trace and prefill batching (2026-08-28): the timestamped trace of the "pipelined, split
  off" 2-stage run shows a single group per round — 8 lanes over a 4-lane minimum give two
  groups only while all eight are active, and as requests finish (7, 6, 5 … lanes) the
  round is one group, i.e. lockstep — so the overlap A/Bs measured nothing about overlap.
  The trace also sizes a stage round: at one lane a 24-layer stage takes 14-18 ms with the
  gather over an x8 link and ~60-80 ms at 5-7 lanes; the host gap between one stage's
  consume and the next launch is ~5 ms (14 ms when the executor processes tokens). So the
  "fixed cost" is mostly miss gathers plus host-function latency per layer, which the
  98 %-resident 8-stage configuration does not pay. Prefill batch 4 on 8 stages: 16 users
  58.2 (TTFT 2.6 s — batching waits), **64 users 59.6 tok/s, 65 completions, TTFT 18 s** (it
  ran; the graph-update failure did not recur). Throughput flat from 16 to 64 users is the
  fill/drain signature; C3 is building.
- C3's first binary hung on the first request (2026-08-29): a lone-prefill flight runs its
  stages synchronously inside `launch_group_prefill`, so it was finished before any
  `tick()`, and `begin_flight` dropped the finished-group list — the executor never resolved
  the prefill, the lane never became decode-ready, and the loop spun on it (the server's
  warm-up then aborted). Fix (ed8f9033): launches park finished groups in
  `pending_finished_`, which the next `tick()` returns; the executor ticks whenever
  something is in flight or pending. The C3 chain is rerun with a 2-stage CLI generation
  check first.
- Pipelined-loop hardening (2026-08-29, 7eca72fe+): a lane whose group is mid-pipeline is not
  aborted under its in-flight round; the cancellation snapshot masks in-flight lanes and is
  retaken every iteration, so the cancel lands at the group's boundary.
- **C3 runs** (2026-08-29 01:50): the 2-stage CLI check generates correctly ('Paris', exit 0)
  and its trace shows the steady state — a group's next round launches on stage 0 in the same
  millisecond its previous round finished on the last stage, with no drain in between; at
  one lane a 24-layer stage round is 15-30 ms with the gather over the x8 link. The C3 chain
  now measures the 2-stage A/B at 8 users (2-lane groups), 8 stages at 16/64/1 users, and the
  27B and 35B at 100 users.
- Second C3 stall (2026-08-29 02:00): 8 users admitted, all six remaining sat in "prefilling"
  with the worker idle — the pipelined loop only launched a lone prefill for staged lanes
  that passed `mixed_round_supported`, so when none did (no decode lanes yet), nothing was
  ever launched. The single-round loop advances any staged prefill. Fix (e9bdc3b5): a lone
  prefill takes the group's first staged lane; mixed support gates only the mixed round.
- **C3 measured, first point** (2026-08-29 02:09): 2 stages on GPUs 2+3, split off, 8 users,
  128/512: **53.7 tok/s, 15 completions** against 44.5-44.8 for the closed pipeline / lockstep
  and 42.2 for one card — +21 % with the stages' gathers sharing one x8 link. The 8-stage
  points (16/64/1 users) and the 27B/35B at 100 users follow in the same chain.
- Third C3 fatal (2026-08-29 02:15, 8 stages, 16 users): "prefill lane set overflow" — the
  pipelined loop's own admission loop admitted into every free lane, but admission stages
  the prompt into the mixed prefill set, which holds 8. `top_up_prefill_lanes` already admits
  exactly while the set has room; the pipelined loop now uses it alone.
- **C3 on 8 stages traced** (2026-08-29 02:45, 16 users): decode is fine — one-column stage
  rounds of 3-10 ms cycling through all eight stages — but the lone prefill steps cost
  **1.5-2 s per stage** (96 steps, 12 prompts × 8 stages), executed synchronously on the
  executor thread, so decode starves and TTFT reaches 116 s (3.5 tok/s). The closed
  pipeline's mixed rounds prefilled the same 512-token chunk on a stage in ~75 ms, so the
  lone `advance_prefill_lane` path is ~20× slower than the mixed path on the same stage.
  Two threads of work: an A/B (prefill share 0; split off) to see which component the lone
  step spends its time in, and the structural fix — run lone prefills as asynchronous
  mixed rounds with zero decode lanes so the pipeline never blocks on them.
- Structural fix for the lone prefill (2026-08-29, commit 996041b8): a mixed round may now
  have zero decode lanes — it is then a batched prefill step (the decode head, sampling and
  egress are skipped at batch 0) — and the pipelined loop launches staged prompts as such
  rounds, so a prompt's chunk is an asynchronous flight through the stages like any decode
  group instead of a synchronous stage-by-stage step on the executor thread. The
  synchronous lone-prefill flight remains only for prompts the mixed path cannot advance.
- Lone-prefill A/B (a) (2026-08-29 02:53, 8 stages, 16 users): with `--cpu-moe-prefill-share 0`
  the lone step is still 1.8 s median (88 steps) while decode stage rounds take 6.2 ms — the
  host prefill split is not the cause. (b) split off, then the built-in prefill timer
  (`SUROGATE_SERVE_PREFILL_TIMING`) follow; C3v5 (zero-lane mixed rounds) is queued behind them.
- Boundary copy cost (2026-08-29, commit 6041be16): the driver copied the whole 47 MB boundary
  buffer twice per hop (park, import) regardless of the round's width — ~9 ms of host memcpy
  per stage transition, a large share of a 6 ms decode stage round. Decode flights now carry
  exactly their lanes' columns (one column per lane); mixed and prefill rounds keep the full
  copy because their graphs pad to buckets.
- Lone-prefill A/B (b) (2026-08-29 03:00): split fully off → still 1.75 s median per stage step
  (decode stage rounds 6.2 ms). The lone `advance_prefill_lane` path is slow on a stage on
  its own, independent of the expert path; and in C3 almost every prompt takes it, because
  a group prefills alone whenever *its own* group has no decode lanes — with eight groups
  and few decoding lanes that is nearly always (12 of 12 prompts in the trace). The mixed
  path prefilled the same chunk on a stage in ~75 ms in the closed pipeline, which is why
  C3v5's zero-lane mixed rounds are the fix; the prefill-timing run decomposes the lone
  path meanwhile.
- The prefill-timing run could not start (2026-08-29 03:04): `SUROGATE_SERVE_PREFILL_TIMING`
  lands in a `cudaErrorInvalidResourceHandle` at the third stage's warm-up — the window-lap
  events it uses belong to the first device. Debug-only path; left as is (multi-device
  unsafe, noted). The lone-path decomposition therefore waits; C3v5 is building.
- Zero-lane mixed rounds hit `slice range out of bounds` in the family's mixed body
  (`Tensor::slice` refuses length 0 and `mixed_chunk_multi` rejects batch ≤ 0), so the
  executor's launch of them is behind `SUROGATE_SERVE_PIPELINE_ZERO_LANE_MIXED` until the
  body accepts batch 0 (10b69c98). Meanwhile the lone path's cost is measured directly on a
  2-stage server with the capture log and the trace (one user, several ~512-token prompts).
- **Prefill on a stage is gather-bound in both paths** (2026-08-29 03:15). The lone-path
  experiment (2 stages, 1 user, capture log): all 32 captures happen at warm-up, none per
  request; the steps are replays of 1.1-3.5 s (bimodal) on the x8 pair. The closed
  pipeline's own trace has its mixed rounds at 1.3-2.4 s per stage as well. A 512-token
  prompt scans every expert of the stage (24 layers × ~440 ≈ 10k on 2 stages, 3,072 on 8) and
  a pool smaller than the scan misses nearly all of it (LRU on a cyclic scan) — 24 layers ×
  440 × 5 MB ≈ 50-90 GB per prefill step at 26-52 GB/s. So on 8 stages the pool is 72 slots
  short of fully resident: **3,072 slots ≈ 15 GiB fit**, after which a stage never gathers.
  Test running: 8 stages at 3,072 slots, 16 users (traced), 64 users (KV bounded), 1 user.
  For fewer stages the pool cannot hold the scan, and the lever there is a scan-resistant
  replacement (keep the resident set, gather scan misses through slots the scan itself just
  filled) or the CPU prefill split.
- Batch-0 mixed rounds (2026-08-29, commit ab89f445): the family's mixed bodies accept a
  decode batch of zero (the decode-row blocks — rope, attention, GDN conv and recurrent
  snapshots — are guarded, `Tensor::slice` allows an empty range, the head/sampling were
  already guarded), so a staged prompt's chunk runs as an asynchronous mixed flight through
  the stages; the pipelined loop uses that by default (`SUROGATE_SERVE_PIPELINE_LONE_PREFILL`
  forces the old synchronous step). Queued: rebuild, CLI check, then 8 stages at 3,072 slots
  (16 traced / 64 / 1), the 2-stage point and the 27B/35B at 100 users.
- Full residency does not fix the lone prefill step (2026-08-29): 8 stages with 3,072 slots
  (every expert of a stage resident, CPU split off by the policy) still show a 2.0 s median per
  lone prefill stage-step at 16 users (3.8 tok/s; the first prompt's steps 0.9-1.3 s per stage
  cold). So the cost is a per-step constant of the synchronous path, not the gather and not
  the layer count. Decode rounds on the same stages take ~7 ms each. Next measurement: the
  asynchronous batch-0 mixed flights (C3v6 chain) — if a stage's mixed step is as slow the
  constant is in the chunk forward itself and gets decomposed with per-device timing.
- Batch-0 mixed flights work (2026-08-29, commits ab89f445..9fd7d5cb): 2 stages (GPUs 2+3,
  split off) at 8 users: **56.1 tok/s** (C3 with synchronous lone prefill 53.7; lockstep 44.5),
  answers correct at load. TTFT p50 17.3 s — the prompt still costs ~0.8 s per 24-layer stage
  even for a 15-token prompt (CLI trace: mixed flight 810 ms on stage 0, 740 ms on stage 1),
  so a stage's prefill step is dominated by something that scales with the graph bucket, not
  the prompt. Per-device prefill timing (the family timer was a process-wide static with
  events on device 0) is being built in `csrc/build-serve-b` to decompose it.
- **8 stages, fully resident, asynchronous prefill (2026-08-29 04:10)**: Flash-Next on 8×5090,
  3,072 slots per stage (every expert of the stage resident, CPU split off), batch-0 mixed
  flights for the staged prompts: **383.9 tok/s decode at 16 users, TTFT p50 615 ms,
  1,549 prompt tok/s, 278 requests ok / 0 errors in 90 s, answers correct under load** — the
  synchronous lone step gave 3.8 tok/s on the same configuration; one card serves 32.2 at 16
  users. The 2-stage prompt cost (0.8 s per 24-layer stage) is the expert gather over the x8
  link of GPUs 2/3 (~150 paths × 24 layers × 5 MB ≈ 18 GB at 26 GB/s), which full residency
  removes at 8 stages.
- Same configuration, 1 user: **57.9 tok/s, TTFT 237 ms** (one card with the CPU split: 22.4 /
  1.40 s). 64 users with a 16,384-token KV did not fit beside the 14.9 GiB pool (needs 7.7 GB,
  4.5 GB free) — rerun queued with the auto KV size, plus 32 users.

