# Tid-Baked Operand Dispatch

**Date:** 2026-04-21
**Status:** Design
**Unblocks:** Phase 4 M5.1 (delete `TensorSlot::Block*/MoE*`), M5.2 (delete
`SimplifiedLayerActivations`), M5.6 (delete `builtin_slot_from_name`), M5.7
(delete `layer_start`/`layer_end` flags on ops).

## Goal

Replace the name/slot-keyed runtime dispatch in `resolve_tensor()` with
a direct `mTensors[tid]` read that returns a fully-bound Tensor for
every op input and output. Once every operand goes through a tid-baked
fast path, the legacy dispatch chain — `mNamedTensors` hashmap,
`block_activation_ptr(slot)` struct-field indexing, `builtin_slot_from_name`
string table, `TensorSlot::Block*/MoE*` enumerators, `SimplifiedLayerActivations`
struct, and the associated `layer_start`/`layer_end` per-op flags — is
dead weight and can be deleted wholesale.

## Current state (evidence from code)

- [`compiled_ops.h:584`](../csrc/src/runtime/executor/compiled_ops.h#L584):
  `mTensors` is a `std::vector<Tensor>` indexed by `tensor_id`. Sized to
  `fwd_graph.num_tensors` per replay ([`compiled_ops_execute.cpp:179`](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L179)).
  Entries start empty; populated lazily via `store_tensor()` or
  eagerly via `bind_tensor()` / `persist_saved_layer_tensors()`.
- Two fast paths exist in `resolve_tensor()`
  ([`compiled_ops_save.cpp:954-1081`](../csrc/src/runtime/executor/compiled_ops_save.cpp#L954)):
  - **SaveForBwd shortcut** (lines 990–1003): `tid ≥ 0` + region
    `SaveForBwd` + cached `.Data` → direct `mTensors[tid]` return.
  - **Stack-reduction bypass** (lines 975–988): RSTDs / LSE / LN / AttOut /
    HOut slots skip `mNamedTensors` and reach `block_activation_ptr(slot)`.
- Everything else (~85–90% of operand accesses) falls through to the
  `mNamedTensors` string hashmap or `block_activation_ptr(slot)` struct
  field dispatch via `SimplifiedLayerActivations`.
- `TensorRef.slot` is populated at compile time for block/global/saved/
  param refs. Gradient and cross-layer paths can leave `slot = Mapped`
  ([`graph_compiler.cpp:692, 741`](../csrc/src/runtime/dsl/graph_compiler.cpp#L692)),
  forcing the slow path.

## Target end-state

Every `TensorRef ref` resolves in three steps:

1. `Tensor& t = mTensors[ref.tensor_id];`
2. If `t.Data == nullptr`, bind via region-specific fetch:
   `t = bind_from_region(ref.tensor_id, meta, ref.shape, ref.dtype);`
3. Return `t`.

No string map, no slot switch, no struct field indirection. The
region-specific binders are small (one per `RegionKind`) and all reach
an arena (`Persistent`, `Accumulator`, `FwdStack`, `BwdStack`,
`SaveForBwd`, `BwdCrossLayer`, `MoeSaved`) or a well-known
non-arena source (globals like Targets, Losses).

## Obstacles & concrete resolutions

Each obstacle gets a **mechanism** (how it's resolved) and a **scope**
(what needs to change in the code).

### O1. Saved tensors (`saved.X` references)

**Today.** `mSaved` (`std::unordered_map<std::string, Tensor>`) is the
sole source. `resolve_tensor` falls through to `mSaved[name]` for any
`TensorRef` whose slot is `Saved` or whose name hits the `saved.` prefix.

**Mechanism.** `Saved` refs already carry `tensor_id`. The saved slot's
tid can be the key instead of the name. `store_saved(tid, tensor)` and
`get_saved(tid)` replace the map.

**Scope.** Rewire `mSaved` from `unordered_map<string, Tensor>` to
`std::vector<Tensor>` sized by backward-graph tid count. `save_list`
becomes `vector<int> save_tids`. Callers in
[`compiled_ops_save.cpp:persist_saved_layer_tensors`](../csrc/src/runtime/executor/compiled_ops_save.cpp)
and `persist_saved_source_now` touch the tid directly. Name-keyed debug
(`--dump-saved`) is the only reason the string is worth keeping — stays
in `tensor_id_to_name[tid]`.

**Risk.** Medium. The save-list-to-tid mapping must be stable across
forward and backward compilations (finalize_save_for_bwd already walks
tids).

### O2. View/alias refs (`_flat`, `_biased`, transposed)

**Today.** A view of a base tensor is created at runtime via
`view_for_shape(base, shape, name)` — the view's `.Data` is the base's
`.Data`, `.Sizes` is from the ref. `mTensors[tid]` for the flat tid
points at a shape-adjusted Tensor object.

**Mechanism.** Baked at compile time as a **parent_tid + view_shape**
pair on the flat `TensorMeta`. At first access, the bind pass
materializes `Tensor{.Data = parent.Data, .Sizes = meta.view_shape}`.
The runtime shape (B, T) enters via symbolic resolution using the
current (B, T) pinned on `CompiledGraph`.

**Scope.** Add `parent_tid` + `view_shape_template` fields to
`TensorMeta` (both optional). Extend `bind_from_region` to materialize
views from the parent's cached Tensor. The per-ref `TensorRef.shape`
becomes redundant once (B, T) is known; keep it for one release as a
cross-check.

**Risk.** Low-medium. Shape inference is already centralized in
`op_shape_signatures.cpp`; the view-shape attached to `TensorMeta` just
moves that data earlier.

### O3. Cross-layer references (`blocks[L-1].X` from `blocks[L]`'s backward)

**Today.** A backward op in layer L may read a forward activation from
layer L-1. The forward and backward graphs are **separate**
`CompiledGraph` instances; the forward tid for `blocks[L-1].x` has no
counterpart in the backward graph's tid space. Today's resolution goes
through `mNamedTensors` (populated by the forward pass).

**Mechanism.** Every cross-graph edge a backward ref makes is already
known at compile time — it's a `Saved` ref (covered by O1) or a
recomputed-and-consumed tid (covered by O5). The backward graph
annotates each such ref with an explicit `source: {graph, tid}` pair.
At runtime, the backward binder dereferences into the forward graph's
`mTensors` via that annotation.

**Scope.** Add `cross_graph_source` (`std::optional<{GraphId, int tid}>`)
to backward `TensorRef`s. Compile-time validation: every backward
`TensorRef` has exactly one of (own-tid, save-source, recompute-source,
cross-graph-source). The executor holds a pointer to the forward graph's
`mTensors` for the duration of backward.

**Risk.** Medium. The forward/backward graph separation is a long-
standing boundary; this is the first runtime linkage between them.
Thread-safety on multi-step runs is trivial (forward is complete before
backward starts), but capture/replay interactions need a look.

### O4. Non-param globals (Targets, Losses, DLoss, FreqCis, TokenIDs)

**Today.** Resolved via slot enum dispatch —
[`compiled_ops_save.cpp:1057–1062, 1092–1096`](../csrc/src/runtime/executor/compiled_ops_save.cpp#L1057)
switch on `TensorSlot::Targets` → `mRunState.Targets`, etc.

**Mechanism.** These DO have tids but currently aren't bound in
`mTensors`. Bind them eagerly at `execute_forward` / `execute_backward`
entry: `mTensors[TOKENIDS_TID] = mRunState.Inputs;` etc.

**Scope.** Small. 5-7 globals each get a binding at entry of the
relevant execution function. `TensorSlot::TokenIDs`, `Targets`,
`Losses`, `DLoss`, `FreqCis`, `PositionIDs`, `Encoded`, `LNFinal`,
`LNFinalRSTD`, `FinalResidual` — all stay as enum values (they're
not `Block*` so don't affect M5.1), but the slot switch dispatch in
`resolve_tensor` can go away.

**Risk.** Low.

### O5. Recomputed activations

**Today.** Forward-replay during backward rebuilds activations into
`acts[slot]` of `simplified_acts(L)`. The backward graph's tid for a
forward op is present but not bound to the arena directly — it's
bound to whatever `acts[slot].Data` holds.

**Mechanism.** Replay writes to arena offsets (already the case after
Phase 4 M3 routing). Instead of rebinding via `acts[slot].Data`, the
backward binder for `RegionKind::FwdStack` tids reads
`arenas.fwd_stack_ptr + meta.offset` directly. The `acts[slot]` slot
routing becomes unnecessary.

**Scope.** Medium. Per-layer replay (`replay_layer_forward`) already
emits FwdStack writes at baked offsets. The missing piece is the
**backward** side: teach the backward binder to read FwdStack tids at
`fwd_stack_ptr + meta.offset` instead of going through `simplified_acts`.
This is the core of M5.2 (struct deletion).

**Risk.** Medium. The slot-alias collapse (shipped at `770492a`)
guarantees a single offset per slot per layer, so there's no
ambiguity — but the runtime frame discipline (per-layer replay
windows) has to still hold.

### O6. Gradient tensors (`d_X`)

**Today.** Three flavors with three dispatch mechanisms:
- **ParamGrad** (`d_param`): routed to `DslGradStore` via
  `base_param_from_grad_kind()`
  ([`compiled_ops_save.cpp:1018`](../csrc/src/runtime/executor/compiled_ops_save.cpp#L1018))
  using the baked `base_param_tid` on `TensorMeta`. Already tid-driven.
- **ActivationGrad** (`d_activation`): `Mapped` slot fallback; resolved
  via `simplified_grads` slot dispatch or stack alloc.
- **AccumTemp** (`_from_N`, `_accum_N`): `Mapped` slot; pre-allocated
  accumulators, not inheriting base's tid.

**Mechanism.** `ActivationGrad` and `AccumTemp` both have a
`base_producer_tid` or `base_grad_tid` already on `TensorMeta`. They're
stored either in the BwdStack arena (ActivationGrad) or in a dedicated
accumulator location (AccumTemp). Bind them via region-specific fetch
just like forward activations.

**Scope.** Extend the binder to handle `BwdStack` and `Accumulator`
regions. Accumulator is the last remaining shadow arena from Phase 4
and would need to be flipped default-on for this path to work — i.e.,
**M5.1/M5.2 require `SUROGATE_USE_PHASE_ACCUMULATOR` to ship first**.

**Risk.** Medium. The Accumulator flip is its own multi-session
milestone (M4e, not yet scheduled).

## Milestone plan

The obstacles decompose into five sequential milestones. Each lands
with validation before the next starts.

### M5.0 — Pre-requisites (independent prep)

1. **Ship the Accumulator arena** (M4e): `SUROGATE_USE_PHASE_ACCUMULATOR=1`
   default-on, `DslGradStore` rebinds into it (mirrors M4a/c). This
   unblocks O6.
2. **Write a tid-binding framework**: `bind_from_region(tid, meta)` that
   centralizes the fetch-by-region logic currently scattered across
   `resolve_tensor()`, `block_activation_ptr`, `block_gradient_ptr`,
   `global_activation_ptr`. Initially called only from `resolve_tensor`
   — same behavior, cleaner seams.

### M5.α — Globals + SaveForBwd bind-on-entry (O1, O4)

Eagerly bind all global + SaveForBwd tids into `mTensors` at execute
entry. `resolve_tensor` for these tids becomes a one-line read. The
fallback switch for globals goes away.

**Deletes:** (nothing yet — but shrinks the fallback path)
**Validates:** bit-identical on 3 models.

### M5.β — Cross-graph source annotation (O3)

Compile-time: every backward `TensorRef` gets a `source` annotation.
Backward executor grants read access to forward `mTensors` via a
pointer on `CompiledExecutor`. Fallback to `mNamedTensors` kept but
a `SUROGATE_CHECK_CROSS_GRAPH_SOURCE` env flag asserts every backward
ref is resolved via the annotation.

**Deletes:** (nothing yet)
**Validates:** 3 models with the check enabled → no assertions fire.

### M5.γ — FwdStack/BwdStack binder (O2, O5)

Binder for `RegionKind::FwdStack` and `BwdStack` reads from
`arenas.fwd_stack_ptr + meta.offset` / `arenas.bwd_stack_ptr + meta.offset`.
`acts[slot].Data` is no longer the authoritative source; ops bypass
`simplified_acts` entirely for block-scope activations.

**Deletes:** `SimplifiedLayerActivations` struct, `block_activation_ptr`,
`block_gradient_ptr`, `TensorSlot::Block*/MoE*/DBlock*` enumerators,
`builtin_slot_from_name` for block/MoE slots (globals remain),
`layer_start`/`layer_end` flags on ops (phase-tree instructions are the
source of truth).

This is the **big delete milestone**. 30-50 files touched.

**Validates:** 3 models + full benchmark gate (memory + throughput ±2%).

### M5.δ — Views + gradient leftovers (O2, O6)

Migrate view/alias tids (O2: `parent_tid` + `view_shape_template` on
`TensorMeta`) and any remaining gradient paths (O6: the slot dispatch
for `ActivationGrad` / `AccumTemp`).

**Deletes:** `resolve_slot_with_flat`, `resolve_block_slot` helpers,
remaining callers of `builtin_slot_from_name` (global-only usage stays
for debug dumps and one registry-init path).

**Validates:** 3 models + re-run benchmark.

### M5.ε — Cleanup sweep

Delete the now-dead name/slot dispatch code: `mNamedTensors` hashmap,
`try_get_tensor_fuzzy`, `resolve_slot_with_flat`, `resolve_block_slot`,
the name-keyed slot registry (`kSlotMappings` table). Everything in
`tensor_slot_dispatch.{h,cpp}` likely goes.

## Estimated effort

| Milestone | Sessions | Lines touched (est.) | Risk |
|-----------|:--------:|---------------------:|------|
| M5.0      |     1    |  ~150                | Low  |
| M5.α      |     1    |  ~80                 | Low  |
| M5.β      |     2    |  ~400                | Medium |
| M5.γ      |    3-4   |  ~1500–2500          | High |
| M5.δ      |     1    |  ~200                | Medium |
| M5.ε      |     1    |  ~500 (deletion)     | Low  |

**Total: 9-11 sessions of work.** M5.γ alone is the deepest because
it touches every block-scope op access.

## Open questions

1. **Parent-view lifetime.** When a view's parent tid lives in FwdStack
   (per-frame arena), does the view's runtime Tensor need invalidation
   at frame end? Likely yes — same rule as the parent.
2. **Capture-replay interaction.** CUDA graph capture bakes pointers.
   All tid-driven reads resolve to arena offsets (stable) — safe by
   construction. But the "bind on first access" pattern for temps
   still cudaMalloc's; M5 doesn't change that.
3. **Multi-graph tid registry for MoE save path.** MoE saves are keyed
   by name (`mMoeSavedBuffers`) and the arena is cross-step monotonic
   with a grow-on-exhaust cudaMalloc fallback (see M5.8 memo). O1's
   "tid-keyed saved" design collides here — the MoE path would need
   its own tid allocation for saves. Likely defer: MoE saves can stay
   name-keyed behind a specialized fast-path, at the cost of keeping
   a couple of `builtin_slot_from_name` calls in the MoE-only code.
4. **Cross-graph tid remapping across recompile.** When `(B, T)`
   changes and graphs recompile, forward tid N may no longer mean the
   same tensor. The backward graph's `cross_graph_source` annotation
   must be rebuilt each compile. Easy to forget — validate with
   `SUROGATE_CHECK_CROSS_GRAPH_SOURCE`.

## Non-goals

- **No new arenas.** Phase 4's 8-region arena layout
  (Persistent / Accumulator / FwdStack / BwdStack / SaveForBwd /
  BwdCrossLayer / MoeSaved / UnifiedStack) is the final set.
- **No Python-side changes.** The DSL already produces tids; this is
  a pure C++ dispatch-layer refactor.
- **No numerical behavior changes.** Every milestone validates
  bit-identical (within FP noise) on 3 models.

## Decision needed before starting

M5.0's "ship Accumulator arena" is the gating pre-requisite for the
whole chain. Without it, O6 stays partially slot-dispatched and we
can't cleanly delete `TensorSlot::BlockD*` (the gradient enumerators).

**Question for review:** land Accumulator arena (M4e) first, or accept
that M5 retains gradient enum values until then?
