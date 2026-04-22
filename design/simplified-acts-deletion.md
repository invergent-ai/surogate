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
