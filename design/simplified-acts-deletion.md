# Deleting `SimplifiedLayerActivations`

**Date:** 2026-04-22
**Status:** Design
**Parent:** [`tid-baked-dispatch.md`](tid-baked-dispatch.md) — Session C of the big-delete milestone
**Predecessor commits:** `50daf70` (Session A: slot_to_tid), `72e8f4a` (Session C step 1: dead clears), `0a28133` (Session C step 2: dead persist bitmap)

## Goal

Remove the `SimplifiedLayerActivations` struct (and its `simplified_acts[L]` per-layer vector on `DslRunState`) so `mTensors[tid]` on `CompiledExecutor` becomes the **single source of truth** for every block-scope activation.

Once this lands, these can also go:
- `block_activation_ptr()` helper and every direct caller (~30 sites).
- `builtin_slot_from_name()` for block-scope slots (callers would go through the tid path).
- `TensorSlot::Block*/MoE*` enumerators that only exist to index into `simplified_acts`.
- `persist_across_layer_end` bitmap on `SimplifiedLayerGradients` once the same treatment is applied to `simplified_grads` (parallel milestone).

## Current state

After Session A + C:

- `simplified_acts[L][SLOT]` owns a `Tensor` per (layer, slot) pair.
  - Allocated in `dsl_run_state.cpp:720-817` via mAllocator (persistent) or `Tensor::from_pointer(nullptr, ...)` (stack-backed placeholder).
  - `consume_fwdstack_arena` (`graph_executor.cpp`) unconditionally overrides
    `acts[slot].Data = fwd_stack_ptr + meta.offset` for every allowlisted FwdStack slot.
  - 14+ excluded slots remain allocator-owned (`BlockResidualAtt`, `BlockRouterLogits`, MoE buffers, `BlockQKVRoPE` fallback, `BlockResidualFFN` managed-residual stream).
- `mTensors[tid]` on `CompiledExecutor` is a `std::vector<Tensor>` indexed by compile-time tid.
  - `mTensors.assign(num_tensors, Tensor{})` at each `execute_forward` / `execute_backward` / `replay_layer_forward` entry.
  - `populate_fwd_stack_bindings` pre-binds every FwdStack tid (except excluded slots) to `fwd_stack_ptr + meta.offset` right after the assign.
  - `resolve_tensor` has a fast path that returns `&mTensors[tid]` for FwdStack-region tids with non-empty `ref.shape`.

**For arena-backed slots, both caches point at the same bytes.** For excluded (non-arena) slots, `acts[slot].Data` is live and `mTensors[tid].Data` is null — the fast path falls through to `block_activation_ptr`.

## The cache-divergence problem (why Session B hung)

During Session B I attempted to migrate every `block_activation_ptr` caller to `block_slot_tensor` (the tid-first shim introduced in Session A). The mass migration hung Qwen3 indefinitely and was reverted.

Root cause: **mutation asymmetry.** A few call sites treat the returned `Tensor*` as a *mutable handle into the canonical store*:

```cpp
// ops/matmul_swiglu.cpp:120 — "rebind the slot to point at the live buffer"
if (Tensor* mlp_up = block_activation_ptr(mRunState, layer_idx, TensorSlot::BlockMLPUp)) {
    mlp_up->Data = up_out.Data;
}

// ops/matmul.cpp:261 — similar forward-hook rebind
// compiled_ops_execute.cpp:clear_large_bwd_grad_stack_slots — t->Data = nullptr at layer_end
```

With `block_activation_ptr`, every caller reads and writes the same `acts[slot]` — one store, one invariant. With `block_slot_tensor`, a mutation might land in `mTensors[tid]` while a *different* caller still reads `acts[slot]`, or vice versa. The two caches silently drift until a downstream kernel reads a stale or null pointer and stalls.

The provably-dead path we removed in Session C (the `clear_rstd_*` / `clear_ffn_temp_*` helpers) is exactly this shape — they were writing nullptr to `acts[slot]` while future reads would have gone through `mTensors[tid]`. Fortunately those clears were already no-ops under the unconditional arena override, so deleting them closed that specific divergence lane.

The remaining mutation sites (rebinds, MoE compact-info wires, view re-propagations) still assume `acts[slot]` is the canonical store. A naive migration breaks that invariant.

## Resolution options

Four candidate paths, ordered by increasing invasiveness. Each ends at the same destination (`mTensors[tid]` is authoritative) but takes a different route through the refactor.

### Option A — Dual-cache sync at every mutation

Keep both caches. Every site that mutates `acts[slot].Data` also updates `mTensors[tid]` (via `slot_to_tid`), and vice versa. A small helper `set_slot_data(layer_idx, slot, data)` encapsulates the two writes.

**Pros.** Smallest code change. No structural rewrite. Can migrate readers incrementally (each one that switches to `block_slot_tensor` is safe because both caches are always in sync).

**Cons.** Invariant is not enforced by types — a future mutation that forgets to call the helper silently re-introduces divergence. Each read pays two lookups (or the reader accepts non-canonical data from whichever cache the caller last wrote). This is the pattern I'd bet a 6-month maintenance window regresses.

**Effort.** ~2 sessions: identify and migrate mutation sites (~5-10 of them), audit every `.Data = X` on a Tensor pointer returned from `block_activation_ptr`.

### Option B — `acts[slot]` becomes a pointer into `mTensors`

Change `SimplifiedLayerActivations::slots` from `std::array<Tensor, kSize>` to `std::array<Tensor*, kSize>`. Each entry points directly at `mTensors[slot_to_tid(L, slot)]`. Mutations through `acts[slot]->Data` reach the same Tensor as `mTensors[tid].Data`.

**Pros.** Single source of truth, enforced by structure. Mutation sites work unchanged because the pointer indirection makes them transparent. No caller migration needed.

**Cons.**
- **Ordering problem.** `SimplifiedLayerActivations` is allocated at `DslRunState` init (model load). `mTensors` is resized per `execute_*` entry. When does the pointer get wired? Naive answer: rewire at every `execute_forward`/`_backward` entry, before any op runs. But `mTensors.assign(num_tensors, Tensor{})` invalidates the pointees on each entry — we need to re-establish both slot→tid AND `mTensors[tid]` state before anything reads `acts[slot]`.
- **Lifetime.** `mTensors` is a `std::vector`; reallocation (if `num_tensors` grows across compiles) invalidates the pointers. Must either (a) size `mTensors` to an upper bound once, or (b) re-wire pointers on every compile.
- **Null pointers.** `acts[slot]` entries where no tid exists (slots not used by the current model) would have to default to a sentinel (static empty Tensor or nullptr with a null check).

**Effort.** ~2-3 sessions: restructure `SimplifiedLayerActivations`, wire pointers at `execute_*` entry, audit null-safety on every `acts[slot]` dereference (~40 sites).

### Option C — `block_activation_ptr` delegates to `mTensors[tid]`

Change `block_activation_ptr` itself to route through the tid cache internally:

```cpp
Tensor* block_activation_ptr(DslRunState& rs, int layer_idx, TensorSlot slot) {
    // NEW: when an executor is active, mTensors[tid] is the truth.
    if (auto* exec = rs.active_executor()) {
        if (auto* graph = exec->current_graph()) {
            const int tid = graph->slot_to_tid(layer_idx, slot);
            if (tid >= 0 && exec->has_tensor(tid)) {
                return &exec->tensor(tid);
            }
        }
    }
    // Legacy fallback (no active executor, or non-arena slot without a pre-bound tid).
    ...existing simplified_acts logic...
}
```

**Pros.** Every caller benefits automatically. Mutations through the returned pointer land in the authoritative store. Incremental — the fallback keeps working while migration happens.

**Cons.**
- `DslRunState` needs an "active executor" handle. `CompiledExecutor` would set/clear a back-reference at execute entry/exit. Circular dependency risk — manageable with a forward declaration.
- Excluded slots (`BlockResidualAtt` et al.) still hit the fallback. Their mutations go to `acts[slot]` and `mTensors[tid]` stays untouched. This is **fine** as long as `slot_to_tid` returns -1 for them OR populate_fwd_stack_bindings stays exclusionary — the tid path never fires for those slots, so divergence doesn't arise. Already guarded by the skip list in populate.
- The tid path returning `&exec->tensor(tid)` exposes `mTensors` through a non-private accessor. Acceptable: `mTensors` is already read by op dispatchers via `resolve_tensor`.

**Effort.** ~1-2 sessions: add `DslRunState::active_executor`, wire set/clear at `execute_forward`/`_backward` entries, re-validate bit-identity. No caller migration needed.

### Option D — Structural deletion (the end state)

The final step: delete `SimplifiedLayerActivations` entirely. Every caller already reads `mTensors[tid]` (via Option C's rerouting). The allocations in `dsl_run_state.cpp:720-817` move to a dedicated per-layer block-state struct (or merge into a global tid-indexed allocator table) that `populate_fwd_stack_bindings` and its siblings populate.

**Pros.** Real deletion. `block_activation_ptr` → `mTensors[slot_to_tid]`. `builtin_slot_from_name` → `slot_to_tid`. `TensorSlot::Block*` enumerators stay (they index the per-layer LUT) but `simplified_acts` storage is gone.

**Cons.** Replaces the allocator-backed path for excluded slots (BlockResidualAtt, MoE, BlockResidualFFN). Each needs its allocation migrated — either into the fwd_stack arena (increases arena sizing), into a dedicated persistent buffer list, or folded into the existing residual-stream manager.

**Effort.** ~3-4 sessions: migrate excluded-slot allocations, delete `SimplifiedLayerActivations`, delete `block_activation_ptr` + `builtin_slot_from_name` block branches, validate at each step.

## Recommended sequence

**C → D.** Option C first because it's the smallest change that resolves the divergence without touching every caller, and it unblocks D. A + B both create ongoing maintenance hazards (A) or significant single-PR complexity (B).

```
Session 1 (Option C):
  - DslRunState::active_executor setter/getter, wired at execute_forward/backward
    entry and cleared at exit.
  - block_activation_ptr delegates to exec->mTensors[slot_to_tid] when active
    and the tid has a valid binding; falls through to simplified_acts otherwise.
  - Validate bit-identical on Q3/GPT-OSS/Q3.5. No caller migration.
  - Commit.

Session 2 (Option D prep — migrate excluded slots):
  - BlockResidualAtt: currently persistent cudaMalloc via mAllocator. Move to
    a dedicated per-layer persistent buffer list on DslRunState; wire
    populate_*_bindings to reflect it into mTensors[tid].
  - Similar for BlockRouterLogits/Probs/RoutingWeights/RoutingIndices,
    BlockPermutedInput/ScatterIndices, BlockExpertGateUp/Act/Down,
    BlockMoeOut (view). Each migrates to a dedicated storage and a
    populate step.
  - BlockResidualFFN stays on the managed-residual-stream path — populate
    step refreshes mTensors[tid] from rs.get_residual(L) at execute entry.
  - BlockQKVRoPE in-place fallback: populate step writes
    mTensors[tid_of_qkv_rope] = mTensors[tid_of_qkv] when qkv_rope isn't
    separately allocated.
  - Validate bit-identical after each slot migrates. ~5-10 commits in
    this session.

Session 3 (Option D — delete):
  - simplified_acts now has no unique state. All consumers read from
    mTensors. Delete the struct, the per-layer vector, and every
    reference.
  - block_activation_ptr reduces to mTensors[slot_to_tid]. Rename to
    block_slot_tensor (Session A's shim already has this name and
    semantics).
  - builtin_slot_from_name for block slots becomes unused. Delete.
  - TensorSlot::Block*/MoE* enumerators: keep as indices into the LUT
    (slot_to_tid rows), but the enum itself is no longer load-bearing
    outside this one use. Re-evaluate in a cleanup pass.
  - Validate 3-model + full benchmark gate (±2% memory / throughput).
  - Commit as one landmark PR.
```

## Risks

1. **Invalidation on recompile.** When (B, T) changes and graphs recompile, tid namespace resets. `mTensors.resize(new_num_tensors)` invalidates the old bindings. Any callback that cached a tid pointer across recompile crashes. Mitigation: recompile always happens before the next `execute_forward`, which re-populates. Audit: ensure no cross-compile tid holders.
2. **Replay mode.** `replay_layer_forward` swaps `mTensors` out, runs forward ops for a single layer, swaps back. During replay, `mCurrentGraph` and `mTensors` temporarily point at the forward graph's state — any `block_activation_ptr` caller during that window must see the forward arena. Option C's `active_executor` handle plus the swap/restore around replay takes care of this, but must be validated.
3. **Backward consuming forward activations.** Backward reads forward activations via save list (SaveForBwd) or recompute (replay). Both paths already use `mTensors[tid]` via `persist_saved_layer_tensors` / `populate_fwd_stack_bindings`. Session C's changes didn't touch that — known good.
4. **Excluded slots drift.** Even after Option D, if a new slot is added and someone forgets to add its allocation to a `populate_*_bindings` step, the tid path returns a null Tensor and kernels fault. Mitigation: a compile-time validator that every FwdStack tid with a block-scope slot has a binding source.
5. **Test coverage gap.** The 3-config smoke (Q3 / GPT-OSS / Q3.5) catches most paths but misses: Nemotron-H Mamba blocks, Qwen3-VL vision adapters, multi-GPU ZeRO-3, FP4 Quartet recipe. Sessions 2/3 should add at least one config per architecture family before landing.

## Non-goals

- **No new arenas.** Phase 4's 8-region layout is the final set.
- **No removal of `TensorSlot` enum.** Enumerators remain as row indices into `slot_tid_by_layer`. Deleting the enum itself is out of scope — it would touch hundreds of files and provides no runtime benefit.
- **No changes to the forward/backward graph separation.** `CompiledGraph` pairs stay as-is.
- **No changes to `SimplifiedLayerGradients`.** The BwdStack-side counterpart is a parallel migration tracked separately in `tid-baked-dispatch.md` (pending session).

## Open questions for review

1. **Option A vs C.** Am I underweighting Option A's simplicity? The dual-cache-sync helper would be maybe 30 lines and every mutation site becomes a 1-line change. The "future maintainer might forget" argument is real but not unique to this code.
2. **Excluded-slot landing site.** Should MoE slots move into the FwdStack arena (growing its size) or stay on mAllocator with a new population path? Arena growth has implications for cross-layer reuse and peak memory.
3. **Regression gate.** Are the 3 smoke configs enough, or do we need the full benchmark suite to land Session 3?

## Session D proper — attempted 2026-04-22, abandoned

Tried three partition strategies to bind cross-graph forward activations into `mTensors[tid]` at backward entry. All regressed. Captured here as evidence for future attempts.

**Why Option D's obvious plan is harder than it looks.**

1. **Region-only filter (`bwd_meta.region == Unknown`)** matches **0 tids**. bwd's `classify_tensors` + `derive_regions` + `finalize_save_for_bwd` promotion assigns non-Unknown regions (BwdStack, Persistent via global bindings, SaveForBwd via promotion) across the full tid range. The cross-graph set isn't identifiable by region alone.

2. **Producer-based filter (`fwd.produces(tid) && !bwd.produces(tid)`)** is semantically correct and matches the expected tid set — 672 tids on Qwen3, 1110 on Qwen3.5. But binding `mTensors[tid] = fwd_stack_ptr + meta.offset` regresses backward:
   - Qwen3 stays bit-identical (loss unchanged).
   - **Qwen3.5 norm 8.04 → 2.15** (gradient norm collapses).
   - **GPT-OSS norm 2.73 → 180** (gradient norm explodes).

3. **Narrowed to `kFwdStackConsumeSlots` allowlist**: same two regressions, slightly different magnitudes.

**Root cause (hypothesis).** FwdStack offsets are reused across layers via coloring. At any instant, `fwd_stack_ptr + offset` contains data from whichever layer wrote there most recently. During forward, layer N overwrites layer N-1's offset. Post-forward, the arena holds layer N-1's data (or whatever ran last). Backward runs in reverse: bwd(N-1) reads correctly, but bwd(N-2) needs arena data that was overwritten — which `replay_layer_forward(N-2)` is supposed to regenerate before bwd(N-2)'s ops run.

The legacy `block_activation_ptr` fallback works because it reads `simplified_acts[slot].Data`, which is the arena pointer — same location. The DATA at that location is what changes. If backward's replay happened for layer N-2, arena has layer N-2's data; both legacy and my tid path read it correctly.

My cross-graph populate binds `mTensors[tid].Data = arena+offset` once at backward entry. The pointer is stable; the data is whatever replay writes before bwd(L) runs. This should be equivalent to the legacy path... but the regressions prove it isn't.

**What I didn't resolve:**
- Why Q3 is bit-identical but Q3.5 and GPT-OSS regress with the same binding logic.
- Whether the issue is a shape mismatch (consumer expects view shape, producer ref has canonical shape) that the legacy fallback handles via an intermediate path but my direct binding doesn't.
- Whether the binding interacts badly with `replay_layer_forward`'s own `mTensors` swap/restore (my bound pointer persists across the swap).

**Recommendation for the next attempt:** don't just bind in backward — also bind at `replay_layer_forward` entry for the fwd_graph's own tids (need to check if already done), and add a debug assertion that every cross-graph read through the tid path matches the legacy fallback's returned pointer and shape bit-for-bit. If they diverge, we've identified the specific slot/op that needs different handling.

**Status:** `SimplifiedLayerActivations` deletion remains blocked. The existing dual-dispatch (Option C) is bit-identical on all tested configs and closed the original design risk (cache divergence from mutations). Further deletion is cosmetic and costs more sessions to get right than it saves.

## Session D proper — second attempt 2026-04-22 (post-M5.ζ), abandoned

Re-attacked after M5.ζ (`531cda3`) landed. Called `populate_fwd_stack_bindings(*mForwardGraph)` at `execute_backward` entry so forward-produced activation tids get pre-bound into bwd's `mTensors`. Same Q3 ✓ / Q3.5 ✗ / GPT-OSS partial pattern as before.

### New findings

1. **Every canonical tid matches the legacy fallback.** Added instrumentation that compared `mTensors[tid]` (tid-path) vs `block_activation_ptr(L, slot)` (legacy-path) for every block-scope FwdStack tid. Ran on Q3.5 — divergence counts came back `ptr=0 shape=0 dtype=0 only_legacy=0 only_tid=0`. Every pointer, shape, and dtype matched. So the regression is NOT a simple per-tid mismatch.

2. **Bisect by `out_ref->slot` narrows to Mapped-slot tids.** With `SUROGATE_XBIND_SKIP=0..26` (skip every named block slot), Q3.5 norm 2.1506 (still regressed). With `SUROGATE_XBIND_SKIP=0..63` (covers Mapped=64ish), Q3.5 norm 8.0431 (baseline). So the regression is triggered by populate binding tids whose producer's `TensorRef::slot` was left at its default `TensorSlot::Mapped` — not declared as a specific block slot in the DSL.

3. **mSaved-clobber hypothesis was wrong.** Added a skip for `mTensors[tid].Data != nullptr` in populate (to preserve `mSaved` pre-binds for force-persist tids like LoRA hooks). Skip fired ~54 times at bwd entry on Q3.5, but regression persisted.

4. **Shape-match check at the `resolve_tensor` fast path didn't help either.** The fast path at `compiled_ops_save.cpp:1256` only fires for FwdStack when `!ref.shape.empty()`; adding `cached.shape == ref.shape` as an extra gate (fall through to slot-dispatch view path when they mismatch) didn't change Q3.5 norm. So if shape was the issue, it wasn't reaching the fast path.

### What this tells us

The divergence isn't in what each `resolve_tensor` call returns — the tid-path and legacy-path return the same pointer/shape/dtype for every tid we can name. The regression must come from **a downstream consumer that reads state via a path other than `resolve_tensor`** — likely one of:

- A direct `mTensors[tid]` access (bypassing `resolve_tensor`) that sees the populate-bound Tensor instead of the null / uninitialized state it depended on.
- A `bind_tensor` or mutation through `block_activation_ptr` that updates `simplified_acts[L][slot]` but not the populate-bound `mTensors[tid]`, creating asymmetric caches mid-backward.
- An op that reads the Tensor struct by value (not just `.Data`) and picks up stale Rank/Sizes from the populate (forward's canonical shape) vs what backward's own flow expects.

Mapped-slot tids triggering the regression suggests path (3) — they're intermediates (views, concat outputs, etc.) whose backward usage may assume a different implicit shape than the forward producer declared.

### Recommendation for the third attempt

- **Instrument every `mTensors[tid]` read on the backward hot path**, not just `resolve_tensor` entries. `grep -rn "mTensors\[" csrc/src/runtime/` shows ~15 direct indexers; audit whether any of them expect `.Rank == 0` (no populate) and break when populate pre-fills.
- **Confirm whether Mapped-slot tids need a separate handling rule.** They escape the slot-based populate path's special cases (BlockResidualAtt, MoE, etc.) but still get the generic `t.Data = fwd_stack_ptr + meta.offset` treatment. Maybe the right answer is to NOT populate Mapped-slot tids and let them continue resolving via the slow path.
- **Consider whether replay_layer_forward's `mTensors` swap/restore interacts badly.** At backward entry my populate fills `mTensors`; replay swaps mTensors out for a fresh one, runs forward, swaps back. The swapped-back `mTensors` has the populate values — but any op during replay that wrote to `mTensors[tid]` (via tid cache) wouldn't reach the outer mTensors. Verify with a pointer snapshot at each boundary.

**Status:** Still blocked. Three attempts so far; each narrowed the problem but none closed it. The current dual-dispatch stays. Session D proper remains cosmetic cleanup, not a correctness issue.

## Session D proper — fourth attempt 2026-04-22 evening, abandoned

Final (for now) deep dive. Narrowed the Q3.5 regression to a **shape mismatch in Q3.5-specific attention views**, specifically `view_backward` resolving via `shape_like` against a populate-bound forward-activation tid.

### What was tried

- Added `skipped_already_bound` check in `populate_fwd_stack_bindings` (preserves mSaved pre-bind). Did NOT fix Q3.5 norm.
- Added `xbind-diff` instrumentation that snapshots `mTensors[tid].Data` before/after populate and logs per-tid changes. Produced a large list of Mapped-slot Q3.5 intermediates (lin_x_flat, split_N, transpose_N, lin_query/key/value, ones_1, ln1_weight_eff, ...).
- Tried skipping populate for tids whose forward producer is `Ones`/`Zeros` (stack-backed via temp_alloc + store_tensor). No change in Q3.5 norm.
- Tried skipping populate for a broader set of metadata ops (View/Transpose/Split/Narrow/Concat/Mul/Scale/MaskScatter/DeepstackInject/MRoPE/RoPE). Crashed Q3 with `view_backward: shape nelem mismatch` — those tids' bindings ARE needed for Q3.

### What the crashes revealed

Skipping Mapped-slot tids in populate (any variant) crashes Q3.5 backward with:

```
view_backward: shape nelem mismatch op=view_backward_20
  input=d_blocks[23].att_4d  in_shape=[2,4096,1024]   in_nelem=8388608   (Hkv * D = 8 * 128 = 1024)
  output=d_blocks[23].att    target_shape=[2,4096,8,256]  target_nelem=16777216  (Hq * D = 16 * 128 = 2048 when flat)
```

Q3.5 has `Hq=16, Hkv=8` — so `d_att_4d` vs `d_att` live at different head counts. `view_backward` resolves the target shape via `shape_like` → reads `mTensors[att_4d_tid]`. With populate bound → 4D shape (16M elem) returned. Without → falls through to the `wants_flat` infer path, which returns a 3D shape different from populate's.

The two paths produce DIFFERENT `target_shape` values, and the legacy (infer) path happens to match what the backward operand expects. The populate path returns the forward-canonical shape, which doesn't.

### Actual root cause

**The `view_backward` shape resolution mechanism depends on the runtime state of `mTensors[shape_like_tid]`.** Under the current design it's intentional that the tid is null / has a stale shape when the fallback infer path needs to fire. Populate filling in the tid breaks that fallback assumption.

This is a genuine design tension, not a bug in populate. Fixing it requires either:
1. Making `view_backward` ignore populate-bound shape_like entries (e.g., by checking an "is_recomputed" flag on the tid) and always fall through to infer.
2. Tracking a separate "fwd_shape" on the TensorMeta that populate can reflect without trampling `mTensors[tid].Rank/Sizes` — so `view_backward`'s shape_like reads the infer-compatible shape while the hot-path dispatch uses the populate's Data.
3. Restructuring `view_backward` so it doesn't depend on `shape_like` resolving differently in fwd vs bwd.

None of these are small changes.

### Current status

- Q3/Q3.5/GPT-OSS backward correctness intact on current HEAD (no populate at bwd entry).
- Session D proper is structurally blocked by the `view_backward` shape_like resolution mechanism.
- `SimplifiedLayerActivations` deletion still hinges on landing per-tid dispatch without disrupting `view_backward` — the root is now understood (`shape_like` reads populate-bound shape instead of inferring).
- The dual-dispatch shipped via Option C remains correct on all tested configs and is the end-of-line for this branch.

**Recommendation for a future attempt:** tackle `view_backward` FIRST — make its shape resolution self-contained (don't consult `mTensors[shape_like_tid]` for shape), then retry populate. Without that fix, populate at bwd entry is fundamentally incompatible with Q3.5's attention view chain.

## Session D proper — UNBLOCKED 2026-04-22 (commit `ab463bf`)

After four abandoned populate-based attempts, the working approach is snapshot/restore rather than re-deriving bindings at bwd entry.

### Mechanism

1. `CompiledExecutor` gets two new member vectors (`compiled_ops.h`):
   - `std::vector<Tensor> mForwardTensorsSnapshot;`
   - `std::unordered_map<std::string, Tensor> mForwardNamedTensorsSnapshot;`
2. At the end of `execute_forward` (after `set_active_executor(nullptr)`), snapshot the full `mTensors` / `mNamedTensors` state.
3. At the top of `execute_backward` — after `set_active_executor(this)`, **before** the `mSaved` pre-bind loop — restore snapshot entries filtered to tids where:
   - `tensor_meta[i].region == RegionKind::FwdStack`, AND
   - `snapshot[i].Data` lies in `[fwd_stack_ptr, fwd_stack_ptr + fwd_stack_bytes)`.

The arena-range guard is the key. It avoids `Stack.owns()` false positives that sank the BwdStack migration earlier, and it preserves whatever mutations forward made to the Tensor struct (shape, dtype, in-place rebinds via `mlp_up->Data = up_out.Data`, etc.) — so `view_backward`'s `shape_like` resolution reads forward's *actual* post-execution shape rather than a fresh populate that would desynchronize from the shape_like fallback path.

### Validation

| Config          | Before / baseline | After  | Status |
|-----------------|-------------------|--------|--------|
| Qwen3           | norm 3.4389       | 3.4390 | ✓      |
| Qwen3.5         | norm 8.0438       | 8.0439 | ✓      |
| GPT-OSS         | norm 2.7561       | 2.7282 | ✓      |
| Q3 no-recompute | — (peak 30004 MiB) | 444 ms / 36911 tps | ✓ |

All three failure modes from prior attempts (Q3.5 norm collapse, GPT-OSS norm explosion, `view_backward` shape mismatch crash) are resolved by the arena-range filter.

### Why this works where populate didn't

- Populate re-derives the Tensor from `meta.offset` and canonical `ref.shape`. That's a *synthetic* binding; consumers that read the Tensor struct by value pick up forward-canonical shape, which is wrong for bwd flows that expect an infer-compatible shape (Q3.5 `att_4d` vs `att` at Hq≠Hkv).
- Snapshot/restore replays exactly the bindings forward ended with — including every `bind_tensor`, in-place mutation, and view re-propagation. No shape re-derivation, no `shape_like` desync.
- The arena-range guard prevents restoring entries whose `Data` was a transient BwdStack or Persistent slot that doesn't survive into backward; only true FwdStack live-outs come back.

### Open items

- The `mForwardNamedTensorsSnapshot` map is populated but the current restore loop only restores tid entries; extend to named-tensor restoration if downstream migrations need it.

## Deletion landed 2026-04-22

Staged across five commits on top of the Session D proper unblock:

| Commit    | Change                                                                                  | LOC   |
|-----------|-----------------------------------------------------------------------------------------|-------|
| `19662ef` | `executor_tid_slot` falls back to `mForwardGraph->slot_to_tid` when bwd has no mapping  | +10   |
| `03c56b8` | `block_activation_ptr` drops the `simplified_acts` fallback branch                       | −57/+11 |
| `a29dbd1` | `populate_fwd_stack_bindings` stops reading `simplified_acts`                            | −60/+14 |
| `3190125` | Delete `consume_fwdstack_arena`, `simplified_acts()` accessor, `mSimplifiedActivations` | −296/+2 |
| `f677dc8` | Delete `SimplifiedLayerActivations` struct + `block_slot_tensor` shim                   | −57/+3  |
| `b2b3bef` | Extend populate + snapshot/restore to `SaveForBwd` region (no-recompute)                 | +44/−17 |

Total: −480 lines net.

### Key insights

1. **Cross-graph tid lookup gap.** The backward graph's `slot→tid` map only has slots declared as output by bwd ops (the `d_*` gradients). Forward-only slots like `res_att`, `ln2`, `ln1_rstd` — which backward READS — were never in the bwd map, so `slot_to_tid` returned −1 and every such read fell back to `simplified_acts`. Since fwd/bwd share the tid namespace, consulting `mForwardGraph->slot_to_tid` when bwd's lookup misses recovers the correct tid and the mTensors fast path takes over.
2. **simplified_acts was already redundant.** `consume_fwdstack_arena` wrote `fwd_stack_ptr + meta.offset` to `simplified_acts[L][slot].Data` for every arena-backed slot — the exact same pointer `populate_fwd_stack_bindings` writes into `mTensors[tid]`. The struct duplicated arena-backed bindings that mTensors already held.
3. **Hot-path fallback fires were all compile-time or return-nullptr anyway.** The audit counted 15–21 fallback fires per session; every one of them fired at setup time, and after the cross-graph fix all but 6 disappeared. The remaining 6 (Parameter, Mapped, D* gradient slots, Saved, BlockHOut L=last) either return `nullptr` from the fallback (not block activations) or are null-guarded by the single caller.
4. **SaveForBwd parity.** Under no-recompute, `finalize_save_for_bwd` promotes fwd→bwd-crossing block activations to `SaveForBwd`. The old `consume_fwdstack_arena` routed both regions; the new `populate_fwd_stack_bindings` and snapshot/restore had to cover both arenas too.

### Final state

- `mTensors[tid]` is the sole source of truth for block activations in both forward and backward.
- `block_activation_ptr` is 10 lines: tid-first lookup + `BlockResidualFFN → get_residual` + `BlockQKVRoPE → BlockQKV` fallback.
- `SimplifiedLayerActivations`, the `simplified_acts(L)` accessor, `mSimplifiedActivations` storage, `allocate_simplified_activations`, `consume_fwdstack_arena`, `kFwdStackConsumeSlots`, and `block_slot_tensor` no longer exist.
- Validation: Q3 / Q3.5 / GPT-OSS all within BF16 noise of baseline; Q3 no-recompute at 37.2k tps (base 36.9k).

## M5.δ — gradient-side deletion (partial, 2026-04-22)

Same scope for `SimplifiedLayerGradients` / `simplified_grads` / `block_gradient_ptr` / `consume_bwdstack_arena` / `allocate_simplified_gradients` / `refresh_simplified_gradients_base`. Started this session; partial progress.

### Landed (commit `218d170`)

**Dead `persist_across_layer_end` bitmap removed.** Grep-confirmed zero writes in the tree; the single read (`clear_large_bwd_grad_stack_slots`'s `if (bitmap[s]) return;` guard) always evaluated false and never short-circuited. Removing the field and the dead guard is a behavior-preserving cleanup.

### Blocked

Two attempted follow-up changes both regressed:

1. **Tid-first routing in `block_gradient_ptr`** (analog of `block_activation_ptr`'s Option C path). Step 0 norm collapsed to `nan`. Root cause: during backward, some ops mutate `simplified_grads[slot].Data` via the returned pointer (e.g., `matmul_swiglu.cpp:460` — `d_ln2->Data = d_inp_ptr->Data`) while `mTensors[tid]` is still empty. Tid-first returns `mTensors[tid]` for later readers, diverging from the mutated `simplified_grads[slot]`. Reverted.
2. **`populate_bwd_stack_bindings`** (analog of `populate_fwd_stack_bindings`) seeding `mTensors[tid].Data = bwd_stack_ptr + meta.offset` at backward entry. Step 0 norm regressed `3.4389 → 0.7786` — the exact value documented in the original design doc's post-failure notes. Root cause: pre-binding BwdStack tids with arena pointers bypasses the `Stack.owns(t.Data)` check at the cross-layer persist (`bwd_layer_end_cleanup`), so tids whose Data got rebound to Stack buffers via the mutation pattern never get persisted into `BwdCrossLayer` arena. Reverted.

### Why gradients are structurally harder than activations

The gradient-side machinery has three entangled subsystems the activation side didn't:

- **Mutation-to-alias pattern.** `d_ln2->Data = d_inp_ptr->Data` and similar rebind an already-arena-backed slot to point at a Stack-backed op output. The `simplified_grads[slot]` storage holds the rebind; `mTensors[tid]` doesn't see it unless we mirror the write there.
- **Temp-acquire on null-Data.** `block_gradient_ptr(...)->Data == nullptr` triggers `temp_acquire` (Stack allocation). Pre-populating `mTensors[tid]` with an arena pointer short-circuits this, changing whether the op writes to Stack or arena.
- **Cross-layer persist via `Stack.owns`.** `bwd_layer_end_cleanup` scans `mTensors` for Stack-resident Data whose `last_use > idx` and copies them into `BwdCrossLayer` arena. Any pre-population that bypasses this detection (arena pointers aren't Stack-owned) skips the persist — correct for the 8 allowlist slots, incorrect for other BwdStack tids that do cross-layer.

### What a clean M5.δ needs

The gradient deletion cannot just copy the activation playbook. It needs either:

1. **Mutation-aware populate.** Pre-populate `mTensors[tid]` identically to `simplified_grads[slot]` AND mirror every mutation through both until simplified_grads is deleted. Essentially a dual-cache sync layer — the Option A pattern from Session B that was abandoned on the activation side.
2. **Delete the mutation sites first.** Find the N mutation sites (matmul_swiglu, fused_residual_rmsnorm, compiled_ops_execute) and convert them to use `mTensors[tid]` directly (via `executor_tid_slot`), bypassing `block_gradient_ptr`. Once no mutations touch `simplified_grads`, tid-first routing becomes safe.
3. **Defer tid migration until after temp_acquire + cross-layer persist are restructured.** The current Stack+persist machinery predates the arena system; it could arguably be recast to work directly against `mTensors[tid]` and the tracking tables at `bwd_layer_end_cleanup`, at which point `simplified_grads` becomes redundant naturally.

Option 2 is the minimum-viable path — same shape as the activation deletion sequence (`03c56b8` and after). It's probably a 1–2 session effort with focused investigation.
