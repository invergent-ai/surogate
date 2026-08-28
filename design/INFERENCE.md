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
phase 3 = PP across 8 GPUs; phase 4 = EP measured against PP.

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
5. **Prefill**: selective streaming of used experts per layer with whole-layer double
   buffering on a side stream.

Exit for phase 2: the single-GPU board row at 1, 16 and 100 users, ≥ 3× llama.cpp
`--cpu-moe` (7.1 / 16.3 tok/s on this box).

## Decisions that shape the code

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
