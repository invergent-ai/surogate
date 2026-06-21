# Dispatch Pipeline Parallelism — Architecture & Feasibility Design

**Date:** 2026-06-20
**Status:** Design (architecture & feasibility) — pre-implementation
**Scope:** v1 = single-node, BF16 full-FT + LoRA, async 1-step-stale optimizer

---

## 1. Overview, goals, non-goals

### What it is
A new **opt-in** single-node multi-GPU training mode, `parallelism: dispatch_pp`, that treats the
node's GPUs as a **stateless round-robin compute pool**. Model work-weights live in pinned CPU RAM;
the model's transformer blocks are grouped into **stages** (with *asymmetric* forward/backward
partitioning); each stage is uploaded to whichever GPU is next, computed, and freed. Per-GPU peak
memory is bounded by ≈ 2× the largest stage — **independent of model size and GPU count** — and
inter-GPU traffic is ≈ zero (all rendezvous through CPU). Unlike `cpu_training`, which streams every
layer through every data-parallel GPU replica, dispatch-PP schedules each stage as one model-parallel
work item across the local GPU pool.

This is **model-parallel**, complementing surogate's current **data-parallel DDP**. It is *additive*:
it does not replace or modify the default path.

> Origin: the technique is "Dispatch Pipeline Parallelism" from RoundPipe (paper arXiv:2604.27085,
> code is LGPL-3.0). This design is a **clean-room reimplementation of the algorithm and runtime
> concepts** in surogate's Apache-2.0 C++/Python stack — no RoundPipe code is copied. A detailed
> engineering trace of the source technique lives in `study/RoundPipe/DISPATCH_PP_REFERENCE.md`.

### Why surogate needs it (the gap it fills)
surogate already has **CPU-RAM centric training** (`cpu_training`) built around `DslWeightManager`:
block work-weights live in pinned CPU memory, are gathered into GPU work buffers during forward and
backward, gradients can stream back to CPU per layer, and the CPU optimizer updates master params.
That path solves the basic "work weights do not fit resident in VRAM" problem for BF16 full-FT and
LoRA.

The remaining gap is **model-parallel scheduling on PCIe-only multi-GPU boxes**. `cpu_training` is
still data-parallel: every GPU streams every layer for its own batch slice, full fine-tune gradients
are all-reduced per layer, and each GPU executes the whole model graph. ZeRO-3 weight streaming also
reconstructs layers through cross-GPU all-gather. Both are useful, but both keep multi-GPU execution
tied to replica-style collectives.

Dispatch-PP reuses the existing CPU-resident weight/gradient machinery but changes ownership and
scheduling: a stage is uploaded to **one** idle GPU, run once, and handed off through CPU boundary
activations/grads. It targets the regime where resident DDP/ZeRO OOMs, `cpu_training` fits but spends
work on replicated full-graph execution or per-layer reduction, and PCIe/NVLink topology makes
collective-heavy approaches unattractive. It composes with — and should be benchmarked against — the
existing `cpu_training` and ZeRO/offload paths rather than pretending they do not exist.

### v1 goals
- Train models larger than resident DDP can hold on PCIe / consumer multi-GPU (no NVLink needed).
- Provide a model-parallel alternative to `cpu_training`'s replica-per-GPU execution.
- BF16 full fine-tune + LoRA.
- Async, 1-step-stale overlapped optimizer (a **core** component, not a later add-on).
- Reuse surogate's existing `DslWeightManager` pinned-buffer / double-buffer H2D/D2H, recompute, CPU
  gradient streaming, and DSL block machinery.

### Non-goals (explicitly deferred; interfaces reserved)
- FP8 / NVFP4 stage streaming (quant-state lifetime across CPU↔GPU stage boundaries).
- 2D parallelism (dispatch-PP × cross-node DDP).
- MoE expert-parallel interplay with stage streaming.
- A synchronous / bitwise-deterministic optimizer option.

### Success criteria
- **Correctness parity**: gradients match a single-GPU BF16 reference within tolerance, with the
  1-step staleness *isolated* in a controlled test (overlap disabled) so correctness and staleness are
  never conflated.
- **Capability**: train a model that OOMs under DDP on the same hardware.
- **Baseline comparison**: include `cpu_training` as the capacity/performance baseline; dispatch-PP
  must show the same memory invariant and a clear throughput win on at least one PCIe multi-GPU case
  before being treated as the preferred mode.
- **Memory invariant**: per-GPU peak memory ≈ 2× largest stage and roughly flat as GPU count scales.

### Coexistence & invariants (hard requirements)
- Default path (`parallelism` unset / `ddp`) is **byte-for-byte unchanged**.
- All changes are **additive** — new files/classes, or new entry points beside existing ones. No
  destructive edits to `execute_forward`/`execute_backward`, the NCCL all-reduce path, or the offload
  flags.
- v1 uses **no NCCL** — single-node coordination is through CPU. The deferred 2D phase is where NCCL
  DDP layers across nodes on top of dispatch-PP.
- Dispatch-PP is mutually exclusive with ZeRO sharding / EP in v1 (validated at config time).

---

## 2. Architecture

Follows surogate's existing split: **Python orchestrates, C++ runs the hot path.** Five components —
three new, two extensions.

```
 Python (orchestration, no hot path)                C++ engine (csrc/)
┌─────────────────────────────────┐      ┌──────────────────────────────────────┐
│ 1. DispatchPlanner  [NEW]        │      │ 3. DispatchScheduler  [NEW]          │
│    - per-block fwd/bwd timings   │ plan │    - GPU worker pool (reuse thread-   │
│    - asymmetric partition +      │─────▶│      per-GPU from MultiGPUPyTrainer)  │
│      cost search → StagePlan     │(data)│    - round-robin dispatch + backpress.│
│                                  │      │    - async optimizer thread (1-stale) │
│ 2. dispatch_pp train mode  [NEW] │      ├──────────────────────────────────────┤
│    - config plumbing, loop glue  │◀────▶│ 4. GraphExecutor sub-range exec [EXT]│
│      (surogate/train/...)        │ step │    - run blocks [i..j] with externally│
└─────────────────────────────────┘      │      supplied stage weights           │
                                          ├──────────────────────────────────────┤
                                          │ 5. Stage streaming [EXT of CPU-train]│
                                          │    - lift per-layer gather/release to │
                                          │      scheduler-owned stage ranges     │
                                          └──────────────────────────────────────┘
```

**1. DispatchPlanner (Python, new)** — `surogate/train/dispatch_pp/planner.py`. Pure logic, zero
hot-path. Consumes per-block fwd/bwd time + size estimates, measured PCIe bandwidth, and free VRAM;
runs the asymmetric-partition cost search; emits a `StagePlan`. Unit-testable without GPUs.

**2. dispatch_pp train mode (Python, new)** — a `parallelism: dispatch_pp` branch in the trainer that
builds the plan, hands it to the C++ scheduler once, then drives the existing step/update loop. Config
plumbing in `sft_config.py`.

**3. DispatchScheduler (C++, new)** — the runtime core. Reuses `MultiGPUPyTrainer`'s thread-per-GPU
pool, but the threads form a **stateless stage pool**: a stage is dispatched to the next idle GPU
(round-robin + `is_idle` backpressure), which uploads that stage's weights, runs the sub-range, ships
activations/grads to CPU, frees the weights. Owns the async optimizer thread and the CPU-resident
master params.

**4. GraphExecutor sub-range execution (C++, extend)** — today `execute_forward`/`execute_backward`
run the whole compiled DSL graph. Add entry points to run a **contiguous block sub-range** `[i..j]`
with weights supplied externally (the streamed stage copy) and activations taken/returned at the stage
boundary (CPU rendezvous). Deepest change; the main feasibility risk (Phase 0 gate).

**5. Stage streaming (C++, extend `DslWeightManager` / CPU-training primitives)** — reuse the mature
pinned-host + double-buffered H2D/D2H machinery that `cpu_training` already uses for work weights and
streaming grads. The new capability is not "stream work weights" in general; it is **stage-range
ownership**: gather/release an arbitrary contiguous block range for whichever GPU the scheduler chose,
with buffer lifetime managed across the stage pool instead of inside one replica's layer loop.

### Data flow per step
Python builds the plan once → C++ scheduler fans stages round-robin across the GPU pool → each stage
streams weights up through the extended `DslWeightManager` protocol, computes its sub-range, streams
activations/grads down via CPU → the main thread barriers on stage completion → the async optimizer
thread applies the (stale) update on CPU master params. **No NCCL on the v1 single-node path.**

### Boundaries / testability
The planner is pure data-in/data-out (test in isolation); `StagePlan` is the one cross-language
contract (a serializable struct); the scheduler depends only on the plan + the sub-range executor;
stage streaming depends only on the existing CPU-training weight/grad streaming primitives.

---

## 3. DispatchPlanner — partitioning algorithm

Pure Python, no hot path, fully unit-testable without GPUs. Clean-room reimplementation of the
bubble-avoidance logic, adapted to surogate's block model.

### Inputs
- `fwd_time[i]`, `bwd_time[i]` — measured wall-time per block (backward incl. recompute). surogate
  already collects per-layer timing; for the first few iterations fall back to a size-proportional
  estimate, then recalibrate from real measurements.
- `weight_bytes[i]` — block work-weight size (bf16).
- `act_bytes[i]` — activation/recompute working set for block *i* at the current `seq_len × micro_batch`.
- `pcie_bw`, `free_vram` — measured at startup (feeds the cost model and the memory ceiling).

### Output — a `StagePlan`
- `fwd_stages`: ordered contiguous block-ranges (ascending).
- `bwd_stages`: ordered contiguous block-ranges (descending), built **independently** from `bwd_time`
  → different cut points = asymmetric partitioning (backward ≈ 3× forward incl. recompute).
- `fused_tail`: trailing range run as one fwd+loss+bwd job (saves a CPU round-trip at the fwd/bwd
  boundary).
- `per_stage`: `{ weight_bytes, est_time, needs_grad, numa_node }` (`numa_node` = preferred NUMA
  placement of this stage's CPU-resident weights on multi-socket hosts; `null` on single-socket).

### Algorithm
1. **Stage packing under two ceilings.** Greedily accumulate consecutive blocks into a stage until
   either:
   - **time**: `stage_time + next_time > max_stage_workload`, or
   - **memory**: `stage_bytes + next_bytes > vram_budget / 2`. Forward stages use
     `stage_bytes = weight_bytes + act_bytes`; backward and fused-tail stages additionally hold the
     **gradient buffer** for *trainable* blocks (`+ weight_bytes` when `needs_grad`, zero for frozen
     LoRA base) — mirroring RoundPipe's param-only forward vs param+grad backward footprint, so LoRA's
     frozen base packs more blocks per stage. Deriving the grad term from `weight_bytes` is *exact* for
     v1 (full-FT grad is bf16 = `weight_bytes`; LoRA base is `needs_grad=false`), so no separate
     `grad_bytes` input is needed until a future mode has trainable blocks whose grad ≠ weight. The `/2`
     reserves headroom for **double-buffering**
     (prefetch stage *k+1* while stage *k* computes), reusing surogate's existing double-buffer pattern.

   The forward partition packs the prefix `[0, fused_lo)` by `fwd_time` (and forward memory); the
   backward partition packs the **same** prefix independently by `bwd_time` (≈3× forward incl.
   recompute) and backward memory → different cut points = asymmetric partitioning.
2. **Fused tail, sized by backward cost.** The trailing range runs fwd+loss+bwd as a single job, so its
   binding constraint is the *backward* budget, not the forward one. Reverse-pack from the last block:
   absorb trailing blocks while `tail_bwd_time + next_bwd_time ≤ max_stage_workload` **and**
   `tail_bytes + next_bytes ≤ vram_budget / 2`; the last block is always included (a stage is never
   empty). The first absorbed block is `fused_lo`, which bounds the forward/backward prefixes in step 1.
   Sizing the tail by `fwd_time` instead would over-pack it ~3× and make it the makespan straggler — the
   tail **must** be budgeted by `bwd_time`.
3. **Choose `max_stage_workload` by 1-D search.** Candidate budgets = every prefix-sum of consecutive
   block times landing in `[max_single_block_time, max_single_block_time × upper_threshold]`
   (`upper_threshold ≈ 1.1`; a stage can never be smaller than the slowest single block). For each
   candidate, pack fwd + fused-tail + bwd stages and count total stages.
4. **Cost model** `cost = max(total_stages, min_stages) × max_stage_workload` (proxy for pipeline
   makespan; `min_stages` defaults to the local GPU count so the floor ensures enough stages to fill
   every GPU, and raising it forces more, smaller stages to relieve memory pressure). Keep the
   min-cost plan.
5. **Validate** the plan covers every block exactly once, contiguously, fwd ascending / bwd descending.

### surogate-specific adaptations
- **Block granularity = DSL blocks**, not arbitrary layers — consume the IR's `num_hidden_layers`
  boundaries and `BlockSchema`. Embedding and LM-head are profiled as their own blocks and partitioned
  like any other; the LM-head naturally anchors the **fused tail** (the loss is computed there), and a
  block whose footprint exceeds the per-stage ceiling is isolated into its own stage by the
  single-block rule — there is no separate "force each into its own stage" pass.
- **Memory ceiling fed by real `free_vram`** (surogate already queries `GPUUtilInfo.mem_free`).
- **PCIe-aware operating-envelope check**: using the transfer model (§6) and measured
  `pcie_bw`, compute the **token threshold** to stay compute-bound and **emit a warning** during plan
  build if `tokens_per_step` (batch × seq_len × grad_accum) is below it — telling the user when the
  batch is too small for dispatch-PP to be efficient, rather than silently running transfer-bound.
- **LoRA mode**: base-weight stages are upload-only (frozen, no grad download); only the tail/adapter
  path carries grads — base stages marked `needs_grad=false` so streaming skips the grad D2H entirely.

### Cross-language contract — `StagePlan`
A plain serializable struct (block-range lists + flags + per-stage byte/time estimates) handed to the
C++ scheduler **once** per training run (re-planned only if shapes change). No hot-path Python.

### MoE caveat (deferred)
v1 treats blocks as dense for partitioning. MoE expert-parallel interplay (experts already sharded via
EP) is out of scope and rejected at config time.

---

## 4. DispatchScheduler — C++ runtime core + correctness

The hot path. Reuses `MultiGPUPyTrainer`'s thread-per-GPU pool but flips ownership: threads are a
**stateless stage pool**, not replica owners.

### Worker pool & dispatch
- One controller thread per GPU, each owning 4 CUDA streams with an explicit **priority split**
  (RoundPipe §4): the **critical-path** streams `act_up` (activation H2D) and `act_down` (activation
  D2H) gate compute and must not be blocked; the **non-critical** streams `weight_up` (work-weight
  H2D) and `grad_down` (gradient D2H) are **packed into the idle windows** left by activation transfers
  so they never head-of-line-block the critical path. Mirrors surogate's existing CPU-training streams;
  the new piece is that the scheduler owns the stage-range copy/lifetime instead of one replica's layer
  loop, and prioritizes activation transfers over weight/grad transfers. Weight/grad tensors are
  assigned to transfer windows by longest-processing-time-first bin-packing (large tensors, e.g. the
  LM head, pre-split) — surogate's `chunk_layer_params` analog.
- A stage is dispatched to the **next GPU in round-robin**; the launch side blocks on that GPU's
  `is_idle` slot → natural backpressure (stage *k* waits for GPU *k mod N* to free from stage *k−N*).
  No stage↔GPU affinity.
- Per stage on the assigned GPU: stream weights up → run sub-range on `compute` → stream activations
  (fwd) or grads (bwd) down → **free the weight copy**. Peak resident ≈ 2× largest stage.
- **NUMA-aware allocation & affinity (multi-socket hosts).** On dual-socket servers (e.g. 2× EPYC
  Genoa, the primary target rig) host RAM and PCIe roots are split across NUMA nodes. Three
  requirements: (a) each GPU's pinned staging buffers are allocated on the NUMA node local to that
  GPU's PCIe root (cross-NUMA H2D roughly halves effective bandwidth); (b) the CPU-resident block
  weights are pinned with a known NUMA placement; (c) round-robin dispatch is **NUMA-biased** — prefer
  assigning a stage to an idle GPU whose local node holds that stage's weights, falling back to any
  idle GPU only under backpressure. This is the one place pure round-robin needs a tweak; without it,
  a 4×5090 / dual-EPYC box runs stage uploads at ~half bandwidth. The planner exposes per-stage weight
  NUMA placement in the `StagePlan` so the scheduler can make this choice cheaply.

### Why surogate does NOT need RoundPipe's backward "tag" hack
RoundPipe's rotating per-GPU `tag` tensor + custom `autograd.Function`s exist **only** to drive
*PyTorch's* eager autograd from worker threads in the right cross-microbatch order. surogate has its
**own C++ AOT-compiled backward graph** — backward block order is already explicit in the IR. The
scheduler therefore **encodes stage order directly**: backward stages dispatched in reverse block order
with explicit dependency edges. This replaces ~200 lines of autograd-tag machinery with plain scheduler
edges, and is the single biggest simplification vs the source technique.

### Correctness mechanisms (kept)
Two layers, both already idioms in surogate's offload / CPU-training code:
1. **Stage dependency edges** (thread-level): backward stage *g* consumes stage *g+1*'s grad-output;
   the scheduler gates dispatch on completion (counting-semaphore / latch per stage). Enforces the
   data-dependency chain regardless of which GPU each stage landed on.
2. **CUDA events** (stream-level, under the above): the producing `down` stream records an event; the
   consuming `act_up`/`weight_up` stream waits on it before the H2D — so a stage never reads
   activations/grads whose D2H has not completed.
3. **Cross-stream buffer lifetime**: a freed stage-weight/activation buffer must not be reclaimed until
   every stream that touched it has synced. Reuse surogate's allocator discipline (equivalent to
   RoundPipe's `InterStreamMemManager`). A correctness must-have — getting it wrong frees GPU buffers
   under in-flight kernels.

### Async optimizer (1-step-stale) — core
- A dedicated optimizer thread drains a queue of per-stage update closures, applying grads →
  **CPU-resident master params** (reusing the `offload_master`/`offload_grads` + CPU-optimizer path
  that `cpu_training` already has). FP32 master + bf16 work weights, as today.
- Ordering per `step()`: enqueue, in order, *sync-back of last step's updated weights* → *move this
  step's CPU grads into the optimizer* → *the optimizer update* → *signal done*; then return
  immediately. Iteration N+1 forwards on weights from N−1's optimizer → **one step stale**, accepted as
  the v1 default.
- Fences must be **per-layer (not per-model)**. RoundPipe (§4.3.2) reports that whole-model sync stalls
  block 0 of the next iteration behind the deepest layer's optimizer update; binding the
  `param_copied` / `grad_copied` latches per layer lets the optimizer worker release early layers
  immediately while deeper layers still sync — worth **2.6–14 s/iter** (growing with trainable-param
  count). surogate already has per-layer offload fences, so this is granularity reuse, not new
  machinery. The paper's consistency protocol keeps three copies — GPU-transient, CPU-master,
  FP32-optimizer — under five ordering constraints (no torn weights, no incomplete/!consistent grads,
  optimizer-semantics); v1 maps these onto surogate's existing fence + master-param scheme.
- The design states plainly: **v1 trades bitwise determinism for the overlap.** A synchronous fallback
  is deferred; the design notes exactly where the barrier would go so it is a small later addition.

### Failure handling
A stage that throws aborts the iteration with the offending block id + GPU + pass (fwd/bwd/fused),
matching surogate's existing per-thread exception reporting. No partial-grad commit: the optimizer
thread consumes grads only once all backward stages for the step have signalled.

---

## 5. Config & API surface

Minimal, additive. Everything defaults to today's behavior.

### User-facing config (`SFTConfig` / YAML)
```yaml
parallelism: dispatch_pp        # NEW. unset/"ddp" → today's path, untouched
dispatch_pp:                    # NEW sub-block, only read when parallelism == dispatch_pp
  min_stages: <int>             # default = num_local_gpus
  upper_threshold: 1.1          # stage-balance slack (planner)
  vram_budget_gb: <float|auto>  # default auto = measured free_vram × 0.9
  recompute_grain: stage|layer  # default stage
  # async optimizer is ON (v1 core); no flag in v1
```
- Reuses existing knobs as-is: `gradient_accumulation_steps`, `sequence_len`,
  `per_device_train_batch_size`, `lora`, `master_dtype`. No new optimizer config — dispatch-PP drives
  the **existing** `OptimizerConfig` (AdamW / 8-bit / NorMuon) on the CPU-master path.
- **Validation rules** (shape/static checks in `SFTConfig.__post_init__`, beside the existing offload
  validations):
  - `dispatch_pp` ⟂ `zero_level>1`, `shard_weights`, `shard_gradients` — mutually exclusive in v1
    (error with a clear message).
  - `dispatch_pp` + `ep_size>1` (MoE) → **rejected in v1** (deferred).
  - `dispatch_pp` + `use_cuda_graphs` → disabled with a warning (same reason `offload_master` already
    disables graphs: cross-stream weight prefetch is incompatible with capture).
  - `dispatch_pp` + FP8/NVFP4 recipe → **rejected in v1** (BF16/LoRA only), with a "deferred" message.
- **Runtime warnings** (during trainer startup / plan build): emit the **token-threshold warning** (§3)
  after measuring PCIe bandwidth and free VRAM. This cannot be a pure config-time check because the
  planner needs live hardware measurements.

### Cross-language contract (`StagePlan`)
```
StagePlan {
  fwd_stages:  [[block_lo, block_hi], ...]      # contiguous, ascending
  bwd_stages:  [[block_lo, block_hi], ...]      # contiguous, descending
  fused_tail:  [block_lo, block_hi]
  per_stage:   { weight_bytes, est_time, needs_grad, numa_node }   # needs_grad=false for frozen LoRA base
                                                                  # numa_node = preferred socket, null if single-socket
}
```
Passed Python→C++ **once** per run; only `step()`/`update()` cross per iteration (existing API).

### C++ engine additions (`_surogate.pyi`)
- `SurogateTrainer(..., parallelism="dispatch_pp", stage_plan=StagePlan)` — new optional ctor args;
  absent → existing data-parallel trainer, unchanged.
- `GraphExecutor`: new **sub-range entry points** (`execute_forward_range(lo, hi, weights, act_in)`,
  `execute_backward_range(...)`) alongside the existing whole-graph methods — additive.
- New `RuntimeOptions` fields: `DispatchPP=false`, `DispatchRecomputeGrain`. Existing offload fields
  reused unchanged.

### What does NOT change
`step()` / `update_with_config()` / `DataLoader` / checkpoint format / `OptimizerConfig` / all existing
offload + ZeRO + EP flags. **Checkpoint interop**: dispatch-PP stores the same CPU-master params, so a
model trained with dispatch-PP loads identically under DDP and vice-versa.

---

## 6. Performance model & operating envelope

### Why large models are the *good* case
Per stage, per optimizer step:
- **Transfer** ≈ `stage_params × 6 bytes` for full-FT bf16 — params are uploaded **twice** per step
  (once for the forward pass, once re-uploaded for the backward pass because the GPU is stateless and
  recompute needs them again) plus grads downloaded once: `2×2 (up) + 1×2 (grad down) = 6`. The fused
  tail stage uploads once. LoRA base stages are upload-only (4 bytes, no grad download).
- **Compute** ≈ `stage_params × tokens × ~8 FLOPs` (fwd + bwd + recompute).

Params are uploaded once per pass and **reused across all microbatches in that pass** (upload gated on
first microbatch, free on last). Taking the ratio, `stage_params` **cancels** — so transfer-boundedness
depends on **tokens per optimizer step** and the **PCIe-bandwidth-to-FLOPS ratio**, *not* on model size:

```
compute / transfer  ≈  (8/6) × tokens × PCIe_BW / GPU_FLOPS
```

Compute-bound (transfer hidden) requires roughly:

```
tokens_per_step  ≳  (6/8) × GPU_FLOPS / PCIe_BW
                 ≈  3000 tokens  on a 4090 (~80 TFLOPS sustained bf16, ~20 GB/s PCIe Gen4)
                 ≈  ~1.5 sequences of 2048
```

Any realistic batch clears this; **bigger models have more compute per token, hiding transfer better,
not worse.** (RoundPipe's reported 8×4090 numbers show a 32B model trained at multi-thousand tok/s —
near A800-NVLink class — precisely because at that size the pass is compute-bound and weight streaming
is fully buried.)

**Paper-backed roofline (RoundPipe §3.3 / Appendix C).** The authors' operational-intensity analysis
concludes PCIe transfer "can be entirely overlapped by computation" once the **microbatch count**
per step is large enough: **B ≥ 8 for dense models, B ≥ 80 for MoE**. This is the authoritative,
unit-clean form of the envelope (microbatches, not tokens) and the planner should warn against it
directly. Two consequences: (1) the token-threshold above is a useful hardware-derived cross-check but
the **B ≥ 8 dense rule is the primary guidance**; (2) the **B ≥ 80 MoE** requirement is an additional,
independent reason MoE is deferred — naive stage streaming would need very large microbatch counts to
amortize, on top of the all-experts-streamed waste.

**Bubble model (RoundPipe).** Total GPU-time = `(M·S + N·(N−1))·t` for `M` microbatches, `S` stages,
`N` GPUs, per-stage time `t`; bubble ratio = `N·(N−1) / (M·S + N·(N−1))`. Because dispatch-PP's stage
count `S = S_fwd + S_bwd` is ~4/3× a looped schedule's (forward ~3× cheaper than backward, so it packs
more stages), the bubble ratio is smaller than classic PP at the same `M`, `N` — and shrinks further
with more microbatches. This is the formal statement of "more, smaller, equal-time stages → fuller
pipeline."

**Planner cost is cheap (paper-confirmed).** The partitioning is `O(L³)` (O(L²) candidate budgets ×
O(L) greedy pack); RoundPipe measures 2.9 ms for small models and 1.47 s for 235B (94 layers). This
confirms the planner belongs in Python with zero hot-path concern.

### Where it genuinely is slow (documented limits)
1. **Too few microbatches** (B < 8 dense / B < 80 MoE) → transfer not hidden, transfer-bound.
   Mitigation: lean on `gradient_accumulation_steps`/microbatching so each uploaded stage is reused
   many times; the planner warns when below the B ≥ 8 dense roofline.
2. **Weak PCIe** (Gen3, x8 risers, no P2P) → raises the token threshold. The planner's cost model and
   warning use measured bandwidth.
3. **vs. a model that fits in VRAM** — DDP is faster (no streaming). Dispatch-PP is for the
   doesn't-fit-resident regime. The serious baselines are `cpu_training` (fits via per-layer streaming
   but keeps data-parallel replicas), ZeRO-3/offload (all-gather bound on PCIe — RoundPipe reports
   ~70% of ZeRO-Infinity's time is communication, vs dispatch-PP's point-to-point host transfers +
   near-zero-bubble round-robin yielding a measured 1.48–2.16× throughput edge on 4090), and "cannot
   run" for default resident DDP.
4. **Long `seq_len`** — boundary activations grow with sequence length and can dominate transfer.
   Mitigation: `recompute_grain` + the planner accounting `act_bytes` in the memory ceiling.
5. **Pipeline fill/drain + first-iteration** before the timing-based plan calibrates — small fixed
   overheads.

---

## 7. Phasing, testing & risks

### Phased build path
Front-loads the biggest unknown so we fail fast if the engine cannot do sub-range execution.

- **Phase 0 — Feasibility spike (GATE).** Prove `GraphExecutor` can run a contiguous block sub-range
  `[i..j]` with externally-supplied weights and CPU-boundary activations, on **one GPU**, matching
  whole-graph fwd/bwd numerically. No scheduler, no streaming overlap. **If this cannot be made to work
  cleanly, the approach is reconsidered** — everything else depends on it.
- **Phase 1 — Single-GPU stage-range streaming. DONE (2026-06-21) — gate PASS.** The existing
  `DslWeightManager` per-block streaming path is reused unchanged: `offload_master` places block
  masters in pinned CPU and the executor's `handle_layer_start`/`handle_layer_end` gather/release each
  block's GPU work copy through a 2-slot double buffer (the dispatch-PP executor already drives those
  handlers). Streamed forward hidden states and per-block grad norms are **bit-identical** to a
  resident run on Qwen3-0.6B/4-layer, single RTX 5090
  (`tests/train/dispatch_pp/test_phase1_streaming.py`). The single-GPU memory invariant holds by
  construction (≤ ~2 blocks of work-weights resident, regardless of layer count). The quantitative
  peak-memory *scaling* assertion (≈ 2× largest stage, flat as N grows) is a Phase-2 concern — it needs
  the multi-GPU pool and GPU-resident-weight-bytes introspection. Stage-range (multi-block) gather is
  also Phase 2 (single-GPU streams per block, which is the stage unit there). See
  `2026-06-21-dispatch-pp-phase1-streaming.md`.
- **Phase 2 — Planner + multi-GPU pool.** Add the DispatchPlanner and round-robin dispatch across N
  local GPUs with the semaphore+event correctness model. Target: per-GPU memory ≈ flat as N grows;
  throughput scales ~linearly.
  - *Planner→model integration landed (2026-06-21):* `surogate/train/dispatch_pp/profile.py` derives
    real `BlockProfile`s from a checkpoint (per-block work-weight bytes read from the safetensors
    header, activation working-set from model dims + runtime shape, size-proportional fwd/bwd times),
    and `plan_for_model(...)` produces a NUMA-placed `StagePlan` with operating-envelope warnings.
    Validated end-to-end against Qwen3-0.6B (28 blocks): coverage, descending backward, NUMA
    round-robin, microbatch roofline warning (`tests/train/dispatch_pp/test_profile.py`).
    `resolve_vram_budget_bytes()` implements the `vram_budget_gb` / auto=free_vram×0.9 resolution.
    Remaining Phase-2 work is the C++ multi-GPU stateless pool (round-robin/NUMA dispatch, stage
    dependency edges + CUDA-event handoff, stage-range gather, GPU-resident-weight-bytes introspection,
    and the quantitative memory-scaling test).
  - *Multi-GPU round-robin forward dispatch landed (2026-06-21) — PASS.* The C++ runtime dispatches
    contiguous block stages round-robin across the stateless GPU pool (stage i -> GPU i % ngpu) via
    `run_work`, handing the full block boundary GPU -> host -> GPU with **no NCCL**
    (`MultiGPUPyTrainer::dispatch_pp_debug_forward_hidden_multigpu`). The fused-residual block carries
    two tensors across a boundary — `blocks[hi].res_att` (residual after attention) and
    `blocks[hi].mlp_down` (`x`, the previous block's MLP output, folded in by block hi+1's first op).
    Both are read **by name** on the sending GPU (kept live by `set_debug_preserve_layer`, which keeps
    the stage's last block's stack past its layer-end) and **bound by name** into the exact graph tids
    on the receiving GPU (allocate + `bind_tensor` via `set_debug_inject_named`), after which
    `debug_restore_stage_base` drops the preserved allocations so a reused GPU starts clean. Final
    hidden matches single-GPU within bf16 tolerance for 2 stages and for the round-robin-wrap case
    (more stages than GPUs), on Qwen3-0.6B/4-layer across 2 RTX 5090s
    (`tests/train/dispatch_pp/test_phase2_multigpu.py`: runs-end-to-end + 2-stage parity + wrap parity,
    all passing).
  - *Multi-GPU round-robin backward dispatch landed (2026-06-21) — PASS.* Stages run in reverse forward
    order (the loss-owning stage first) via `MultiGPUPyTrainer::dispatch_pp_debug_grad_norms_multigpu`.
    Each stage runs one GPU on the full batch: a forced-eager whole forward provides activations, then a
    bounded backward selects the stage's ops **by their owning block layer** [lo..hi]
    (`set_debug_backward_layer_range`) — robust to boundary view ops (e.g.
    `d_blocks[L].mlp_down -> .mlp_down_flat`) whose op index falls between `layer_end[L+1]` and
    `layer_start[L]`; by layer they belong to block L, so an op-index range would drop them into an
    inter-stage gap. The incoming boundary gradients (`d_blocks[hi].res_att` / `.mlp_down`, produced by
    the higher stage) are bound by name into the backward graph (`apply_debug_named_inject`), and the
    outgoing boundary (`d_blocks[lo-1].*`) is read by name and handed to the next lower stage's GPU
    through host memory. Running one GPU at a time required skipping the DP collectives that would
    deadlock waiting for idle GPUs: the per-layer grad all-reduce (`CompiledExecutor` skip flag), the
    wrapper-level `reduce_loss` + `reduce_all_async` (`GraphExecutor` skip flag), and
    `reduce_loss_on_completion` on the request. Per-block weight-grad L2 norms match whole-graph backward
    within bf16 tolerance for 2 stages and round-robin wrap
    (`tests/train/dispatch_pp/test_phase2_multigpu.py`: 3 backward tests passing).
    Still ahead: stage dependency edges + CUDA-event handoff for compute/transfer overlap, NUMA-biased
    dispatch, stage-range (multi-block) gather, and the quantitative memory-scaling test.
- **Phase 3 — Async 1-step-stale optimizer.** Overlap the optimizer thread on the CPU-master path;
  verify the controlled staleness test. Sequenced last because it is the hardest to debug — Phases 1–2
  run with a synchronous optimizer internally as scaffolding (sync is not a shipped v1 option).
- **Deferred (interfaces reserved):** FP8/NVFP4 stage streaming, 2D (× cross-node DDP), MoE/EP
  interplay, synchronous-optimizer user option.

### Testing strategy
- **Planner — pure unit tests (no GPU):** known per-block time/size arrays → assert exact stage cuts;
  asymmetry (fwd stages fatter than bwd); memory ceiling respected; cost-search picks min-cost;
  token-threshold warning fires below the boundary; LoRA marks base stages `needs_grad=false`.
- **Sub-range executor (Phase 0):** numerical parity of fwd outputs and bwd grads vs whole-graph, per
  block-range, tight tolerance.
- **Correctness parity:** dispatch-PP gradients vs single-GPU BF16 reference. The 1-step staleness is
  isolated by a dedicated test that **disables overlap** (synchronous optimizer) to assert
  bitwise-equivalent grads; a **separate** test asserts the stale path converges on a small real run —
  so staleness and correctness are never conflated.
- **Memory invariant:** assert per-GPU peak ≈ 2× largest stage and roughly flat across GPU counts.
- **`cpu_training` comparison:** for the same model, shape, and hardware, record memory and tok/s for
  `cpu_training` and dispatch-PP. Dispatch-PP does not need to beat every case, but v1 needs at least
  one PCIe multi-GPU case where the model-parallel stage pool is a clear win.
- **Scaling smoke test:** a model that OOMs under DDP trains under dispatch-PP on the same box; record
  tok/s vs GPU count.
- **Coexistence regression:** the existing DDP/offload/ZeRO/EP suite must pass unchanged with
  `parallelism` unset, and the existing `cpu_training` suite/coverage must keep passing unchanged —
  proves additivity.

### Risks (ranked)
1. **Sub-range execution in `GraphExecutor`** — highest. The engine was built to run whole graphs; if
   block boundaries are not cleanly separable (fused residual/norm across boundaries, etc.) Phase 0
   stalls. *Mitigation:* Phase 0 is the gate; scope to dense decoder blocks first.
2. **Cross-stream buffer lifetime** — a wrong free corrupts silently. *Mitigation:* reuse surogate's
   allocator discipline; buffer-lifetime assertions in debug builds.
3. **Async-optimizer races** on shared CPU master/grad buffers. *Mitigation:* per-stage
   `param_copied`/`grad_copied` fences; Phase 3 isolated and last.
4. **Transfer-bound on weak PCIe / small batch** — performance, not correctness. *Mitigation:* the
   planner's token-threshold warning makes it visible up front.
5. **No meaningful win over `cpu_training`** — correctness may be fine while throughput is not worth
   the extra runtime complexity. *Mitigation:* make `cpu_training` a first-class benchmark and require
   a clear PCIe multi-GPU win before recommending dispatch-PP.
6. **Activation transfer at long `seq_len`** — boundary activations grow with sequence length.
   *Mitigation:* `recompute_grain` + `act_bytes` in the memory ceiling; documented envelope limit.
7. **NUMA-blind allocation on multi-socket hosts** — performance, not correctness. On the primary
   target rig (2× EPYC Genoa, 4× 5090) NUMA-blind pinned buffers and naive round-robin run stage
   uploads cross-socket at ~half PCIe bandwidth, silently halving throughput. *Mitigation:* NUMA-local
   pinned staging + NUMA-biased dispatch (see §4 Worker pool); validate with a per-GPU achieved-H2D-
   bandwidth assertion in the Phase 2 scaling test.

### Open questions (flag, do not block)
- ~~Does the IR cleanly expose contiguous block sub-ranges with separable weights, or is a graph-compiler
  change needed for Phase 0?~~ **RESOLVED — Phase-0 gate PASS (2026-06-21).** `CompiledGraph` exposes
  per-block op ranges (`layer_start_indices` / `layer_end_indices`); `annotate_layer_boundaries` opens/
  closes a block only on non-gradient refs so no op straddles a boundary — the only cross-block
  dependency is the residual hidden state. A debug-only bounded op-range path in the compiled executor
  (guarded, default-off) runs blocks `[0..k]` then `[k+1..last]` as two segments on one shared executor
  state with the boundary residual round-tripped through host memory, matching the whole-graph final
  hidden state (rtol 1e-2); the bounded forced-eager backward matches whole-graph per-block grad norms
  (rtol 2e-2). Validated on Qwen3-0.6B/4-layer, single RTX 5090
  (`tests/train/dispatch_pp/test_phase0_subrange.py`). No graph-compiler change was needed. Caveat:
  running the block ranges as two *separate* backward invocations that share accumulated-gradient state
  across a CPU boundary needs multi-stage grad-accumulation ownership the executor does not expose — that
  is the DispatchScheduler / async-optimizer plans' first task, not a blocker for the gate.
- Optimal `recompute_grain` default (stage vs layer) under streaming — measure in Phase 1.
- NUMA placement policy for CPU-resident weights on multi-socket hosts: split the model across both
  nodes (balances bandwidth, but cross-node stages pay a penalty) vs replicate hot stages vs
  first-touch — measure on the dual-EPYC target in Phase 2.

---

## References
- `study/RoundPipe/DISPATCH_PP_REFERENCE.md` — engineering trace of the source technique.
- RoundPipe paper: arXiv:2604.27085 — https://arxiv.org/html/2604.27085v1 (technique origin; code is
  LGPL-3.0, not used). Key sections drawn on: §3.3 + Appendix C (roofline, B≥8 dense / B≥80 MoE),
  §4 (asymmetric partition, O(L³) planner, bubble formula), §4.3.2 (per-layer async sync, 3-copy
  consistency protocol), Appendix B.1 (per-layer activation formula, recompute vs reload 2.37–5.75×).
- surogate CPU-training / offloading: `docs/reference/config.md`, `docs/guides/offloading.md`,
  `surogate/core/config/sft_config.py` (`cpu_training`, `offload_*`), and
  `csrc/src/runtime/dsl/dsl_weight_manager.{h,cpp}`.
- surogate engine map: `csrc/src/binding/py_train.h` (MultiGPUPyTrainer),
  `csrc/src/runtime/training/model.h` (IModel), `csrc/src/runtime/executor/graph_executor.h`,
  `csrc/src/runtime/training/runtime_options.h`.
