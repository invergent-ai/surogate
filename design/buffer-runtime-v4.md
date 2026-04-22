# Plan: Phase Tree + Typed Regions (revised)

Replace the hand-authored `TensorSlot` / `SimplifiedLayerActivations` / `shared_tag` machinery with a structured phase-tree IR and a small vocabulary of typed memory regions. Layouts are computed at compile time by bump allocation + trivial within-frame coloring. **Allocator-induced aliasing bugs become structurally impossible** (intentional view/name aliases remain explicit and opt-in). Non-transformer architectures fit without runtime changes.

Supersedes schema-per-block and SSA+coloring plans. This revision incorporates code-review findings on backward cross-layer carry, CUDA-graph split-capture, Phase 3 scope, persistence inventory, and role unification.

---

## Status (keep updated)

Last refresh: 2026-04-22. Grep for `Status (keep updated)` to find and update this section after any milestone lands.

Legend: ✅ shipped • 🟡 in progress • ⬜ not started • ❌ abandoned

```
Phase 0 — Audits                                                        ✅ done
Phase 1 — Phase tree IR + region derivation + role unification          ✅ done
Phase 2 — Compile-time layout + within-frame coloring                   ✅ done
Phase 3 — Runtime-architecture migration + benchmark gate               ✅ done (arena consumption shipped; benchmark gate run at 3976cdb)
Phase 4 — Delete the legacy machinery (see design/buffer-runtime-v4-phase4-plan.md) 🟡
├── M4a-e: arena routing default-on (Persistent + Accumulator + LoRA)   ✅ done (56904e8 closes out)
├── M5: tid-baked dispatch (design/tid-baked-dispatch.md)               🟡 HERE
│   ├── M5.0  bind_from_region framework                                ✅ b4c34e2
│   ├── M5.α  globals bind-on-entry                                     ✅ 3309879
│   ├── M5.β  mSaved pre-bind at backward entry                         ✅ 80f0bf5
│   ├── M5.γ  FwdStack tid-baked dispatch                               🟡
│   │   ├── prereq: full FwdStack arena coverage                        ✅ 620f958
│   │   ├── session 1: FwdStack fast path in resolve_tensor             ✅ ee0a7ad
│   │   ├── session 2: consolidate fast paths, drop stray debug         ✅ 559e5e6
│   │   ├── session 3 / Session A: slot_to_tid LUT + helpers            ✅ 50daf70
│   │   ├── Session C design memo                                       ✅ 331f1fa (design/simplified-acts-deletion.md)
│   │   ├── Session C step 1: delete dead layer-end clears              ✅ 72e8f4a
│   │   ├── Session C step 2: delete dead persist bitmap                ✅ 0a28133
│   │   ├── Session C step 3 (Option C): block_activation_ptr → tid     ✅ 9ccc784
│   │   ├── Session D prep: excluded-slot mTensors binding              ✅ 9a0e1f9
│   │   ├── replay-path fix: Mapped-slot rejection, drop replay gate    ✅ 99368a5
│   │   ├── Session D: reorder set_active_executor + fwd-graph setter   ✅ ca48fbc
│   │   └── Session D proper: delete SimplifiedLayerActivations         ⬜ blocked — producer-based partition binds correctly by topology but regresses Q3.5 (norm 8.04→2.15) and GPT-OSS (norm 2.73→180). FwdStack arena data is overwritten across layers; debugging why legacy fallback is bit-correct while direct tid binding isn't requires per-tid shape/ptr divergence instrumentation. Postmortem: design/simplified-acts-deletion.md "Session D proper — attempted".
│   ├── M5.δ  views + gradient leftovers                                ⬜ not started
│   └── M5.ε  cleanup sweep                                             ⬜ not started
└── M6: re-run benchmark gate (3 models, memory ±2% + throughput)       ⬜ not started — blocked on M5 completion
Phase 5+                                                                 ⬜ not planned
```

**Current position (2026-04-22):** Phase 4 M5.γ. The cache-divergence bug class that motivated the memo is structurally closed by Option C (9ccc784) and the replay-path fix (99368a5). Session D (ca48fbc) shipped the two narrow wins toward simplified_acts deletion — reordering set_active_executor to after populate so the tid-first path is coherent at all times within execute_*, plus a forward-graph setter for future cross-graph work. Session D proper (the actual struct deletion) was attempted and abandoned after three partition strategies failed — producer-based topology is correct but binding cross-graph tids to `fwd_stack_ptr + offset` regresses on Q3.5 and GPT-OSS even though it's bit-identical to the legacy fallback on Q3. Postmortem captured in `design/simplified-acts-deletion.md`. The current dual-dispatch is bit-identical everywhere, well-understood, and carries no outstanding risk — further deletion is cosmetic and a future project, not a Phase 4 blocker.

**Next step:** run **M6** (the Phase 4 benchmark gate — 3 models, see [buffer-runtime-v4-benchmark.md](buffer-runtime-v4-benchmark.md) thresholds) to close Phase 4 at a coherent milestone. Session D proper's abandonment means further M5.γ deletion is parked; M6 validates the current state as correctness-and-performance-complete.

**Kill criteria status:** none hit. Phase 1's role unification (`MatmulRole` typed ID) composes with all outcomes and is the fallback safety net.

---

## Core idea

Training is not a general SSA graph — it's a structured, repeating tree-shaped schedule:

```
TrainStep          = MicrobatchLoop → OptStep
Microbatch         = FwdBlockSeq → BwdBlockSeq
FwdBlockSeq        = [FwdBlock_0, FwdBlock_1, ..., FwdBlock_N]
BwdBlockSeq        = [BwdBlock_N, ..., BwdBlock_0]
FwdBlock[i]        = (optional WeightGather[i]) → CoreFwd[i]
BwdBlock[i]        = CoreBwd[i] → (optional GradScatter[i])
CoreFwd / CoreBwd  = sub-phase tree for that block type (attn / mlp / moe / ssm / ...)
```

Tensor lifetimes fall into a small, fixed vocabulary — not an arbitrary lattice. Declare the structure; don't infer it.

---

## Region vocabulary (eight kinds)

| Region               | Lifetime                                               | Discipline                          | Typical tensors                                |
|----------------------|--------------------------------------------------------|-------------------------------------|------------------------------------------------|
| `FwdStack`           | nested phase scope within forward                      | bump + within-frame coloring        | QKV out, MLP up, swiglu, attention out         |
| `BwdStack`           | nested phase scope within backward                     | bump + within-frame coloring        | grad activations, dW partials                  |
| `SaveForBwd[i]`      | fwd block i exit → bwd block i entry                   | indexed heap slot                   | residual save, normed input                    |
| `BwdCrossLayer[j→k]` | bwd block j produce → bwd block k last use (j > k)     | indexed heap slot                   | d_router_logits for MoE aux loss, d_residuals  |
| `GatheredWeight[i]`  | WeightGather[i] → last use in BwdBlock[i]              | indexed heap slot                   | ZeRO-3 all-gathered weight shards              |
| `Accumulator`        | first microbatch write → optimizer step                | persistent within step              | grad accumulators                              |
| `Persistent`         | training-wide                                          | fixed address                       | FP8 amax history, optimizer state, master wts  |
| `Recomputed[i]`      | reuses FwdStack arena during BwdBlock[i] replay        | bump in re-entered FwdStack arena   | dropped activations replayed during bwd        |

**`BwdCrossLayer` is required, not a corner case.** Today's executor at [compiled_ops_execute.cpp:2337-2429](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L2337-L2429) explicitly detects tensors produced during `BwdBlock[j]` whose `last_use > layer_end[j]`, `cudaMalloc`s them, copies them off the stack, and routes future reads to the persistent copy. Examples include `d_blocks[N].router_logits` for MoE auxiliary loss. This is an *indexed heap* lifetime (same pattern as `SaveForBwd`), not representable as `BwdStack` or `Accumulator`. The new region makes it a first-class concept instead of a runtime fixup.

**Key invariants.**

- **Stacks nest.** Inner-phase pop does not free outer-phase allocations. The residual stream is allocated in the enclosing `FwdBlockSeq` scope and survives individual block pops. Across-block sharing is automatic (peak = max over blocks, not sum).
- **Bump is compile-time only.** The compiler walks the phase tree and bakes every tensor's byte offset. Runtime has no allocator.
- **Within-frame coloring is trivial.** Each frame has ~10-30 tensors; a linear-scan 1D-interval pass gives optimal packing. No global optimization problem.
- **Recompute is structural.** During `BwdBlock[i]` replay, re-enter the `FwdStack` arena (now empty), rerun `CoreFwd[i]` into a fresh frame, consume its outputs, pop. No SSA renaming, no autodiff rule rewrite.

---

## CUDA graph capture (honest)

Stable compile-time buffer addresses remove **allocator-induced** capture instability — no per-step allocation decisions can invalidate a captured graph. That is the win.

**Graph breaks caused by dynamic or capture-unsafe ops are orthogonal and remain.** Today [graph_executor.cpp:1007-1016](../csrc/src/runtime/executor/graph_executor.cpp#L1007-L1016) forces split-mode capture when any of:

- Document masking is active (dynamic `cu_seqlens`)
- Capture-unsafe ops are present (JIT kernel loading, MoE/EP per-step host-side bookkeeping)
- Tiled MLP is enabled (per-tile host loop)

In split-mode, [graph_compiler.cpp:1580-1619](../csrc/src/runtime/dsl/graph_compiler.cpp#L1580-L1619) produces per-layer `(segment, eager|graph)` lists; the executor captures each graph segment and runs eager segments between them. The phase-tree design **composes with** this — every `(FwdBlock[i], CoreFwd[i])` phase may contain nested graph/eager segments just like today. What changes: segment boundaries no longer need to care about allocator state (stable addresses), so graph segments are simpler to capture, and segment-boundary bookkeeping reduces.

Revised claim: **stable addresses remove allocator-induced instability for capture-safe segments; the split-capture machinery for dynamic/capture-unsafe ops stays and integrates cleanly.**

---

## Executor model (revised)

Phase tree exists only at compile time. Compile flattens to a linear instruction stream *with segment annotations*:

```
[PhaseEnter FwdBlockSeq, arena=FwdStack]
  [PhaseEnter FwdBlock_0]
    [Segment graph    : ops 0..5]   ; captured
    [Segment eager    : ops 5..6]   ; capture-unsafe op
    [Segment graph    : ops 6..12]
    [SaveCheckpoint list=...]       ; explicit save points
    [PruneByLastUse]                ; explicit pruning
  [PhaseExit FwdBlock_0]            ; pops inner frame
  ...
[PhaseExit FwdBlockSeq]
[PhaseEnter BwdBlockSeq, arena=BwdStack]
  [PhaseEnter BwdBlock_N]
    [Segment eager    : recompute FwdBlock_N into FwdStack]  ; only if recomputed
    [Segment graph    : bwd ops]
    [PersistCrossLayer list=d_router_logits, ...]            ; explicit
    [StackRestore]
  [PhaseExit BwdBlock_N]
  ...
```

Every thing today's executor does *implicitly* at `layer_start`/`layer_end` boundaries ([compiled_ops_execute.cpp:1115-1149](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L1115), [:2337-2429](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L2337-L2429)) becomes an **explicit instruction** in the flat stream: stack checkpoint, stack restore, save-list preparation, cross-layer persist, last-use pruning, recompute entry. The phase tree is the compile-time structure; the instruction stream is the runtime representation. This is a runtime-architecture migration — see Phase 3 below.

---

## Region derivation (with compat layer)

Most of today's `ActivationLayoutIR` fields imply a region:

- `save_for_backward=true` → `SaveForBwd[i]`
- `scope="block"` + `save_for_backward=false` → `FwdStack` in forward, `BwdStack` in backward
- `scope="global"` + `memory_hint="persistent"` → `Persistent`
- Gradient accumulator tensors (marked in IR) → `Accumulator`
- Dropped-and-recomputed (from recompute plan) → `Recomputed[i]`
- ZeRO-3 gathered shards (emitted by weight manager) → `GatheredWeight[i]`
- Detected `last_use > current layer_end` on backward stack → `BwdCrossLayer[j→k]`

**But** region derivation has real dependencies on today's runtime flags:

- Hybrid per-layer dims ([buffer_plan.h:43](../csrc/src/runtime/dsl/buffer_plan.h#L43))
- `kv_source_layers` (cross-layer KV sharing)
- `use_qk_norm`
- Recompute mode
- LoRA-only mode
- Slot availability booleans in `BufferPlan`
- Slot registry aliases ([tensor_slot_registry.cpp:18-174](../csrc/src/runtime/dsl/tensor_slot_registry.cpp#L18-L174))

**DSL author churn is low, not zero.** Rollout needs a **compatibility layer**: a pass that reads today's slot-registry entries (including aliases and conditional availability) and emits the phase-tree region assignments, so existing models migrate without per-model DSL edits. Once the compat layer is in, new architectures can annotate regions directly if defaults don't fit.

---

## Activation offload (honest integration)

Today's residual/activation offload is a residual-specific subsystem:

- Per-layer pinned host buffers and rotating device buffers ([residual_manager.h:58](../csrc/src/runtime/core/residual_manager.h#L58))
- Explicit `fetch_residual` / `put_residual` APIs ([residual_manager.h:116](../csrc/src/runtime/core/residual_manager.h#L116), [:139](../csrc/src/runtime/core/residual_manager.h#L139))
- Fused-op callsites that know about offload directly ([compiled_ops.cpp:585](../csrc/src/runtime/dsl/compiled_ops.cpp#L585), [fused_residual_rmsnorm.cpp:331](../csrc/src/runtime/ops/fused_residual_rmsnorm.cpp#L331))

Mapping to regions: `SaveForBwd[i]` splits into `SaveForBwd_cpu[i]` (pinned host, full window) + `SaveForBwd_gpu[i]` (bracketed device window, stream-aware prefetch). The `ResidualManager` becomes the **allocator for these sub-regions** — its per-layer host buffers and rotating device buffers become the region's backing store. Fused-op callsites continue to use the same fetch/put APIs, now driven by the region's prefetch sub-phase rather than ad-hoc scheduling.

**This is not a trivial relabel.** It's an integration between the new region model and an existing offload subsystem, and it's where most of the design risk in Phase 3 lives. Plan item: read all `fetch_residual`/`put_residual` callsites, enumerate the scheduling they depend on, and prove the phase-tree prefetch sub-phase subsumes them before Phase 3 commits.

---

## Phase 0 — Audits (5-8 days, no code change)

Expanded from v3. Confirm the design holds before committing.

**0.1 Persistence inventory** (not just Qwen3.5). Today has at least five distinct persistence escapes:

- Generic save-list persistence at layer boundaries ([compiled_ops_execute.cpp:636](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L636))
- Qwen3.5 forward-replay exception ([compiled_ops_execute.cpp:743](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L743))
- MoE-specific saves ([compiled_ops_execute.cpp:1317](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L1317))
- Capture-time saved-buffer preparation ([compiled_ops_save.cpp:58](../csrc/src/runtime/executor/compiled_ops_save.cpp#L58))
- Backward cross-layer persistence ([compiled_ops_execute.cpp:2337-2429](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L2337-L2429))

For each: document the lifetime pattern, which region it maps to, and what runtime machinery (prefetch, persist, copy, cudaMalloc+free) the new executor must reproduce.

**0.2 Pipeline/microbatch interleaving.** Does the trainer run 1F1B? If yes, concurrent phase instances each with independent arenas. Adds ~1-2 weeks to Phase 2.

**0.3 Activation offload deep-read.** Trace every `fetch_residual` / `put_residual` callsite. Enumerate all schedule dependencies. Confirm phase-tree prefetch sub-phases subsume them.

**0.4 CUDA graph capture interaction.** Enumerate all conditions that force split-mode today ([graph_executor.cpp:1007-1016](../csrc/src/runtime/executor/graph_executor.cpp#L1007-L1016)). Confirm the phase-tree + flat-instruction-stream model composes with them.

**0.5 Role namespace audit.** `MatmulOp` drives recipe dispatch ([matmul_context.h:27](../csrc/src/runtime/core/matmul_context.h#L27)), FP8 buffer routing, delayed-scaling indices ([graph_executor_utils.cpp:512](../csrc/src/runtime/executor/graph_executor_utils.cpp#L512)), weight-cache selection ([graph_executor_weight_cache.cpp:38](../csrc/src/runtime/executor/graph_executor_weight_cache.cpp#L38)), and matmul dispatch itself ([matmul.cpp:134](../csrc/src/runtime/ops/matmul.cpp#L134)). There's also string-derived role inference from weight names. Design a **typed role ID** (`enum class MatmulRole`) that's used on all hot paths, with string names only at the debug/checkpoint/schema boundary.

**Exit.** One-pager memo per audit. If any find a blocker, recalibrate before Phase 1.

---

## Phase 1 — Phase tree IR + region derivation + role unification (3-4 weeks)

1. **Add phase-tree types to the IR.** `Phase` node with kind ∈ {TrainStep, MicrobatchLoop, FwdBlockSeq, BwdBlockSeq, FwdBlock, BwdBlock, Custom, PrefetchSubPhase}. Ops live at leaves.
2. **Graph compiler emits the tree.** [graph_compiler.cpp](../csrc/src/runtime/dsl/graph_compiler.cpp) wraps block boundaries in phase nodes. `layer_start`/`layer_end` index flags become structural.
3. **Region deriver + compat layer.** Two passes: first, emit regions from today's slot-registry (including aliases and conditional availability); second, derive `BwdCrossLayer[j→k]` from last-use analysis on the backward ops. Validation: every tensor has exactly one region; region usage respects lifetime invariants.
4. **Typed role ID.** Introduce `enum class MatmulRole` with `string_name()` for debug. All hot paths (recipe dispatch, FP8 routing, weight cache, matmul dispatch, LoRA injection) take the enum. String names survive at debug, checkpoint, and IR-schema boundaries only.
5. **Shadow compile.** Produce the phase tree + region assignment in parallel with today's allocator. Log assignment sizes. No runtime change.

**Exit.** All training configs compile bit-exact. Shadow region assignment produced for all onboarded models. Role ID is the runtime-internal type; `MatmulOp` enum alias kept for backward compat, deleted in Phase 4.

---

## Phase 2 — Compile-time layout + within-frame coloring (1-2 weeks)

1. **Phase-tree walker** that computes each tensor's `(buffer_id, offset, bytes)` by bump within its region, honoring nested-phase pop semantics.
2. **Within-frame coloring pass.** Per stack frame, 1D-interval coloring reduces peak to within-frame optimal. Runs per-frame in microseconds.
3. **Alignment constraints.** Every tensor carries a `min_alignment` attribute (NCCL staging, FP8 block boundaries, capture-safe alignment). Bump honors alignment.
4. **Segment annotation.** Layout pass emits segment boundaries (graph/eager) alongside offsets, reusing [graph_compiler.cpp:1580-1619](../csrc/src/runtime/dsl/graph_compiler.cpp#L1580-L1619)'s logic.
5. **Determinism check.** `MPI_Allreduce`-hash the layout at init. Assert per-rank-role equality.

**Exit.** Layout produced in shadow mode. Peak-memory report per model vs today's hand-tuned.

---

## Phase 3 — Runtime-architecture migration (3-5 weeks, not "a core loop rewrite")

**This is not a ~500 LoC change.** Today's executor's layer boundaries drive at least seven cross-cutting subsystems that the new flat-instruction-stream must subsume, each confirmed in code:

- Split-graph segment dispatch ([compiled_ops_execute.cpp:1123-1149](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L1123-L1149))
- Save-buffer snapshots and capture-time preparation ([compiled_ops_execute.cpp:1310](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L1310), [compiled_ops_save.cpp:58](../csrc/src/runtime/executor/compiled_ops_save.cpp#L58))
- MoE-specific saves ([compiled_ops_execute.cpp:1317](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L1317), [:2141](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L2141))
- Stack checkpoint/restore at layer boundaries ([compiled_ops_execute.cpp:1115-1120](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L1115), [:2205](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L2205))
- Last-use pruning ([compiled_ops_execute.cpp:2339](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L2339))
- Backward cross-layer persistence via cudaMalloc ([compiled_ops_execute.cpp:2337-2429](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L2337-L2429))
- Recompute entry/exit

Each becomes an explicit instruction in the flat stream (see Executor model above). Ordering and scope:

1. **Ship the instruction-stream interpreter** with no behavior change: today's executor continues to drive the system, but emits equivalent instructions into a shadow stream as it goes. Validate instruction stream against actual execution.
2. **Flip each subsystem one at a time.** Split-segment dispatch first (lowest risk), then save-list, then stack checkpoint/restore, then BwdCrossLayer persist, then pruning, then MoE saves, then recompute. Each flip is gated by a flag.
3. **Capture invariants test.** After each flip, run full Qwen3-MoE / GPT-OSS / Nemotron-H CI under both `mGraphsEnabled=true` and `mGraphsEnabled=false` with doc masking / tiled MLP on and off. Any capture-mode change fails the flip.
4. **Tensor-id → buffer-offset baking.** Once all subsystems are on the instruction stream, replace runtime tensor lookups with baked offsets. Ops read `(buffer_id, offset)` from the instruction operand table.
5. **Delete the old executor path.** Gate flips to 100%; old implementation goes behind a compile-time flag for one release cycle, then deletes.

**→ Benchmark gate (hard).** After tensor-id baking. Measure on Qwen3-MoE and GPT-OSS, 4-8 GPU:

- Peak GPU memory
- cudaGraph capture-replay throughput (in split-capture mode — this is the realistic case)
- Step throughput (capture + eager paths, both)
- Recompute correctness vs. no-recompute baseline (diff tolerances)

Decision matrix:

| Peak memory          | Capture throughput    | Step throughput        | Action                                    |
|----------------------|-----------------------|------------------------|-------------------------------------------|
| < 2% regression      | < 1% regression       | < 1% regression        | Proceed to Phase 4                        |
| 2-5% regression      | any                   | any                    | Investigate within-frame coloring tuning  |
| > 5% regression      | —                     | —                      | **Stop.** Falls back to schema-per-block. |
| —                    | > 3% regression       | —                      | **Stop.** Capture interaction broken.     |
| —                    | —                     | > 2% regression        | **Stop.** Executor rewrite regressed.     |

---

## Phase 4 — Delete the legacy machinery (1-2 weeks)

With `phase_tree` mode proven, cut:

- `TensorSlot::Block*` / `MoE*` / `SSM*` enumerators (keep only truly-global: `TokenIDs`, `Targets`, `Losses`, `DLoss`, `FreqCis`, `Parameter`)
- `SimplifiedLayerActivations` struct ([run_state_types.h:23-67](../csrc/src/runtime/core/run_state_types.h#L23-L67))
- `shared_tag()` and the per-layer allocator loop ([dsl_run_state.cpp:685-813](../csrc/src/runtime/dsl/dsl_run_state.cpp#L685-L813))
- `share_ln1`/... booleans in [buffer_plan.h](../csrc/src/runtime/dsl/buffer_plan.h)
- `MatmulOp` enum alias (kept for back-compat in Phase 1)
- `builtin_slot_from_name` string table
- `layer_start`/`layer_end` index flags on ops (now structural)
- String-match dispatch branches in [compiled_ops_execute.cpp:180-224](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L180-L224) and [:707-740](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L707-L740)
- Ad-hoc backward cross-layer cudaMalloc path ([:2337-2429](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L2337-L2429))

Add the debuggability surface:

- `tensor_id → human_name` reverse map (survives dead-op elimination)
- `name → (region, buffer_id, offset)` gdb pretty-printer
- Error-message rewriter

---

## Intentional aliases (narrowed claim)

The new design makes **allocator-induced aliasing** — two tensors landing on overlapping bytes because of planner bugs — structurally impossible within regions. It does **not** eliminate intentional aliases:

- Name aliases (e.g., `x0`/`encoded`) declared in the slot registry at [tensor_slot_registry.cpp:20](../csrc/src/runtime/dsl/tensor_slot_registry.cpp#L20).
- View aliases (e.g., `moe_out` as a view of `mlp_down`) at [dsl_run_state.cpp:800](../csrc/src/runtime/dsl/dsl_run_state.cpp#L800).

These are opt-in equivalences, not allocator accidents. The region model carries them as explicit `alias_of` declarations in the IR, validated at compile time. The bug class that disappears: "two unrelated tensors ended up at the same address because `shared_tag` coincidence."

---

## Debug-mode eager oracle

Alternate compile path: every tensor is `Persistent`-region, every allocation is malloc+free. Same phase tree, same dispatch, no reuse. Diffing a training step in `phase_tree` vs `eager` catches buffer-overlap corruption within regions.

**Blind spots:**
- Does not catch stream-sync bugs (eager runs one stream).
- Does not catch ZeRO-3 streaming-specific bugs.
- Does not catch capture-specific bugs (split-capture-mode).
- Does not catch perf-only regressions.
- Does not catch intentional-alias misuse (aliases are explicit in both modes).

---

## Integration landmines (v2, honest)

**Backward cross-layer carry is a real region, not an edge case.** MoE aux-loss gradients and any bwd tensor with `last_use > current layer_end` live here. Today handled by ad-hoc cudaMalloc+copy; in the new design it's `BwdCrossLayer[j→k]`, indexed-heap same as `SaveForBwd`.

**Activation offload integration is Phase 3's biggest risk.** Today's `ResidualManager` has per-layer host buffers, rotating device buffers, explicit fetch/put, and fused-op callsites that know about it. The region model subsumes it but not by relabeling — by giving the manager a new role as the region's allocator, and by adding prefetch sub-phases to the phase tree. Prove this out in Phase 0.3 before Phase 2.

**CUDA graph split-capture stays.** Doc masking, capture-unsafe ops, tiled MLP all force segment-level capture today. The phase tree composes with this: segments live inside phases. What improves: stable compile-time addresses remove capture-invalidation-due-to-allocator. Graph-breaks-due-to-dynamic-ops are orthogonal.

**Role unification uses a typed ID.** `enum class MatmulRole` on hot paths (recipe dispatch, FP8 routing, weight cache, LoRA). Strings only at debug/checkpoint/schema boundaries. Compile-time validation that all LoRA roles exist in the FP8 role table if FP8 is active.

**MoE grouped-GEMM scratch.** Sub-phase of CoreFwd for MoE blocks. Scratch is bump-allocated in the MoE sub-frame, sized to `total_tokens × top_k × hidden`, popped at sub-phase exit. Matches today's layout.

**Pipeline parallelism.** If Phase 0.2 confirms 1F1B, each in-flight microbatch needs an independent phase-instance with its own arena. Peak = sum of concurrent instances' peaks. Adds runtime machinery; if no interleaving, skip.

**ZeRO-3.** `WeightGather[i]` sub-phase emits `GatheredWeight[i]` with explicit lifetime ending at `CoreBwd[i]` last use. `GradScatter[i]` consumes the dW accumulator. Phase trees are rank-identical in topology; region sizes identical across ranks.

**Determinism.** Phase tree rank-identical. Region sizes rank-identical. Buffer offsets rank-identical. `MPI_Allreduce`-hash the layout at init. Rank-divergent graph topology (if it exists per audit 0.2) weakens to per-role equality.

**Qwen3.5 LN1 persist.** Today's forward-replay exception ([compiled_ops_execute.cpp:743](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L743)) becomes `SaveForBwd[i]` with a recomputed-exception flag. No special case in the runtime.

**FP8 scaling state.** `Persistent` region, never reused.

**Within-frame coloring vs bump.** Pure bump is coarser than hand-tuned; within-frame coloring is optimal within frame; nested-stack semantics give optimal cross-block sharing. Expected peak vs hand-tuned: ±2% on transformer-dense, likely better on MoE.

**Phase tree doesn't fit X.** Static schedule with known shape covers all current Surogate architectures. Dynamic control flow does not fit. Not in roadmap; flag if it becomes one.

**Recompute in autodiff.** Replay re-enters the FwdStack arena and produces tensors at the same logical names. Autodiff rules unchanged. This is a genuine simplification vs SSA+coloring.

**Intentional aliases.** Explicit `alias_of` declarations in IR, validated at compile time. Allocator cannot silently create them; author must opt in.

**Checkpoint compatibility.** Checkpoints are role-name-level, not tensor-id-level. Phase 1's typed role ID keeps string names at the checkpoint boundary. No migration shim needed — verify in Phase 0.

---

## What dies / what stays

**Dies.** `TensorSlot::Block*`/`MoE*`/`SSM*` enumerators; `SimplifiedLayerActivations`; `shared_tag()`; the per-layer allocator loop; `share_ln1`/... booleans in `BufferPlan`; `MatmulOp` enum; `builtin_slot_from_name`; `layer_start`/`layer_end` flags; string-match dispatch branches; ad-hoc backward cross-layer cudaMalloc path.

**Stays.** `TensorAllocator` (owns the persistent slab). `CompiledGraph` (gains phase tree + instruction stream). `ResidualManager` (repurposed as region allocator for `SaveForBwd_cpu/gpu`). Weight manager. `DslParamStore` / `DslGradStore`. Recompute planner (now outputs "replay phase list"). LoRA injection (keys on `MatmulRole` enum). Split-capture segment machinery (composes with phase tree).

---

## Scope & timeline (revised)

- **Phase 0 — Audits:** 5-8 days
- **Phase 1 — Phase tree IR + region derivation + role unification:** 3-4 weeks
- **Phase 2 — Compile-time layout + coloring:** 1-2 weeks
- **Phase 3 — Runtime-architecture migration + benchmark gate:** 3-5 weeks
- **Phase 4 — Cleanup + debuggability:** 1-2 weeks

**Realistic total: 8-12 engineer-weeks** (was 6-9 before accepting Phase 3's true scope). Long tail of peak-memory tuning, capture regression chasing, activation-offload integration surprises likely adds 2 weeks. **Budget 14 weeks; cut scope if running long.**

---

## Kill criteria

Fall back to schema-per-block (and keep Phase 1's role unification as a standalone win) if any of:

1. Phase 3 peak-memory regression > 5% on Qwen3-MoE or GPT-OSS, OR
2. Phase 3 capture-replay throughput regression > 3%, OR
3. Phase 3 step throughput regression > 2%, OR
4. Phase 0 audits surface a cross-block sharing pattern that breaks nested-stack semantics, OR
5. Phase 0.2 finds pipeline interleaving requires concurrent-phase machinery that doubles Phase 2 scope, OR
6. Phase 0.3 finds activation-offload coupling that requires rewriting `ResidualManager` rather than repurposing it (Phase 3 grows by >2 weeks), OR
7. Phase 3 subsystem-by-subsystem flip reveals that any single subsystem (split-capture, save-buffer prep, recompute entry) doesn't cleanly map to an instruction-stream primitive.

Phase 1's role unification composes with any path forward.

---

## Comparison

| Concern                                  | Schema-per-block | SSA+coloring                     | Phase Tree (revised)             |
|------------------------------------------|------------------|----------------------------------|----------------------------------|
| Solves transformer-opinionation          | Yes              | Yes                              | Yes                              |
| Solves allocator-induced aliasing        | No               | Interval assertion               | Stack discipline                 |
| Handles backward cross-layer carry       | Unchanged        | Needs SSA + coloring             | `BwdCrossLayer[j→k]` region      |
| MoE grouped-GEMM                         | Yes              | Awkward                          | Natural (sub-phase)              |
| CUDA split-capture interaction           | Preserved        | Real risk                        | Composes; addresses stable       |
| Recompute interaction                    | Unchanged        | Autodiff rule audit              | Structural; no audit             |
| Pipeline parallelism                     | Unchanged        | SSA name discipline              | Phase subtree instances          |
| Peak memory vs hand-tuned                | Matches          | Coloring-dependent               | Within ±2% (within-frame opt)    |
| Executor change                          | None             | Small                            | Runtime-arch migration (3-5 wks) |
| DSL author burden                        | Region tags      | Low                              | Low (with compat layer)          |
| Activation offload integration           | Unchanged        | Unchanged                        | ResidualManager repurposed       |
| Realistic timeline                       | 1.5 weeks        | 8-12 weeks                       | 8-12 weeks                       |
| Primary risk                             | Paths forked     | Coloring quality + capture       | Phase 3 subsystem flip surprises |

Phase Tree and SSA+coloring have converged on timeline after honest scoping. Phase Tree still has the structural recompute/autodiff simplification, but loses the "easier executor rewrite" claim — the migration is real in both. Phase Tree wins on: no autodiff rule audit, cleaner MoE, explicit BwdCrossLayer region, natural composition with split-capture. SSA+coloring wins on: incremental rollout (can flip one scope at a time more easily), no new IR node types.

---

## Recommended first step

**Prototype one Llama forward-and-backward block** end-to-end in a branch: phase-tree IR, region derivation, layout, instruction stream, executor interpretation. Two weeks of work, not two days (earlier estimate was wrong — the executor piece is the hard part). If the prototype runs Llama 1 layer bit-identical and CUDA-graph-captures correctly, commit to Phase 0. If it surfaces a region that wasn't in the eight-region vocabulary or a subsystem that doesn't map to an instruction-stream primitive, fall back before committing weeks.
