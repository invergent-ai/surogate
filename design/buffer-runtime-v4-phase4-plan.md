# Phase 4 — Deep refactor plan

**Branch:** `buffer-runtime-v4-prototype`
**Precondition:** Phase 3 benchmark gate passed ([buffer-runtime-v4-benchmark.md](buffer-runtime-v4-benchmark.md)).
**Goal:** Delete legacy TensorSlot / SimplifiedLayerActivations / shared_tag / string-dispatch machinery. Replace with tensor-id-baked operand access.

## Target end-state (from [buffer-runtime-v4.md §Phase 4](buffer-runtime-v4.md))

Cut:
- `TensorSlot::Block*` / `MoE*` / `SSM*` enumerators (keep globals: `TokenIDs`, `Targets`, `Losses`, `DLoss`, `FreqCis`, `Parameter`)
- `SimplifiedLayerActivations` struct
- `shared_tag()` + per-layer allocator loop
- `share_ln1/...` booleans in buffer_plan.h
- `MatmulOp` enum alias (Phase 1 back-compat)
- `builtin_slot_from_name` string table
- `layer_start` / `layer_end` index flags (now structural)
- String-match dispatch branches in compiled_ops_execute.cpp:180-224, :707-740
- Ad-hoc backward cross-layer cudaMalloc path (already replaced by BwdCrossLayer arena)

## The gate: tensor-id baking

Each of the deletions above is gated on ops reading tensors from baked `(region, offset, shape, dtype)` operands instead of the legacy switch-over-`TensorSlot` in `resolve_tensor()` at [compiled_ops_save.cpp:1068](../csrc/src/runtime/executor/compiled_ops_save.cpp#L1068).

Baking has two prerequisites:
1. **Layout coverage.** Every operand's tid must have `tensor_meta[tid].region != Unknown` and `offset != SIZE_MAX`. Today: Persistent and Accumulator are gated off; FwdStack / BwdStack / UnifiedStack are gated off. Only SaveForBwd / BwdCrossLayer / MoeSaved are active, and only a fraction of ops consume them.
2. **Arena backing for the active region.** The arena must actually be allocated + consumed by the ops, not sit as dead shadow (which is what trapped us in the initial gate run).

So the sequence is: turn on arenas one region at a time, wire ops to read through them, delete the legacy backing for that region. Then delete the legacy slots that are no longer the source of truth.

## Milestones (each ~1 session)

### M1 — Baked operand descriptor (shadow) 🟢 this session

Add a `BakedOperand` struct alongside `TensorRef` on `CompiledOp`:

```cpp
struct BakedOperand {
    dsl::RegionKind region = RegionKind::Unknown;
    int block_layer_idx = -1;      // For SaveForBwd / FwdStack / BwdStack
    std::size_t offset = SIZE_MAX; // As in TensorMeta
    std::size_t bytes = 0;
    ETensorDType dtype = ETensorDType::BF16;
    // Shape is NOT baked: varies with (B,T). TensorRef::shape remains authoritative.
};
```

Populated at compile time in `compile_op()` by reading `graph.tensor_meta[ref.tensor_id]`. Same data, just co-located with the op for O(1) access on the hot path. Unused at runtime this milestone — pure shadow.

**Validation:** compile without regression; coverage validator confirms every op's operand has a populated BakedOperand where the tid has a region.

**Out of scope M1:** no runtime dispatch changes; no arena flips; no deletions.

### M2 — Arena-backed SaveForBwd access in ops

Replace the `mNamedTensors` / `mSaved` string-keyed lookup for SaveForBwd operands with direct `(save_for_bwd_ptr + block_base + offset)` reads. Ops that consume saves (all backward ops that read forward activations) now go through baked operands for those inputs.

Turn `SUROGATE_USE_PHASE_PERSISTENT=1` off, leave SaveForBwd arena on (already default under `!recompute_active`).

**Validation:** bit-identical on Qwen3 dense (small save set). Then Qwen3.5 (after widening save_name_set to include non-recompute tids).

**Risk:** recompute + arena interplay. Mitigate by keeping legacy path behind flag.

### M3 — UnifiedStack activation allocation

Replace `shared_tag()` per-layer allocator loop with baked offsets in UnifiedStack. `SimplifiedLayerActivations` becomes a compile-time-populated view struct, not a runtime-allocated one. Flip `SUROGATE_USE_PHASE_STACK_ARENAS=1` default on.

This is the **big** milestone. Most ops touch activations; all need to read through baked operands. Staged by layer type (dense first, then MoE, then Mamba).

**Validation:** 3-model bit-identical + full benchmark re-run.

**M3 first-strike attempt (not shipped):** Tried narrowly overriding just
`simplified_acts(L).ln1_rstd.Data` with `fwd_stack_ptr + meta.offset` at
the end of `ensure_compiled_graphs`, leaving the legacy `mAllocator`
slot in place. Under `SUROGATE_USE_PHASE_STACK_ARENAS=1 SUROGATE_BAKED_LN1_RSTD=1`:

- Override fired for 28/28 layers on Qwen3, ran end-to-end (no crashes).
- Step 0 loss matched baseline (2.0251) but gradient norm diverged
  immediately (2.63 vs 3.87). Training still descended, but the path
  clearly differed.
- Under `recompute=false` the override didn't fire: `finalize_save_for_bwd`
  promoted ln1_rstd from `FwdStack` to `SaveForBwd` (save list includes
  it), and the override's `region == FwdStack` guard filtered it out.

The divergence is most likely frame-reuse plus forward-replay interaction:
per-frame coloring gives each layer's ln1_rstd tid its own offset within
the same 64 MiB arena, so all layers' ln1_rstd writes alias. Legacy
`mAllocator` gave each layer a distinct buffer. Under forward-replay
backward, replay runs layer N's forward before layer N backward reads
ln1_rstd — intended to make the aliasing invisible. Something about
that interleaving with the specific capture/replay path went wrong; the
first-strike wasn't enough to diagnose which.

**Takeaway:** M3 can't be a one-slot override with legacy allocator still
active — the two storage paths coexisting drift. M3 needs to be holistic
either (a) adopt UnifiedStack for the full simplified activation set in
one pass, OR (b) route the legacy allocator to allocate INTO the arena
at the baked offset, so both paths point to the same memory.

**M3(a) full-slot override attempt (not shipped):** Added
`migrate_simplified_activations_to_arena()` that walks all 16 dense
Qwen3 slots × 28 layers and swaps `simplified_acts[L].X.Data` to
`fwd_stack_ptr + meta.offset`. Under `SUROGATE_USE_PHASE_STACK_ARENAS=1
SUROGATE_MIGRATE_SIMPLIFIED_ACTS=1`:

- Override fired for 336/392 slot instances (56 missing =
  non-existent slots for this model, e.g. `qkv_rope` when separate
  buffer not required). `non_fwd_stack` = 0 (all refs were FwdStack).
- **Loss was NaN from step 0.** Forward pass itself corrupted.

The failure mode confirms the design's layered contract: per-frame
coloring colors each frame independently, so same-slot tids across
layers get offsets that alias in the arena. That aliasing is only
safe if frames are **sequential + isolated** — the stack-save /
stack-restore discipline at block boundaries. The legacy `mAllocator`
gives each layer its own cudaMalloc buffer, so forward accumulates
N layers' activations into N distinct buffers with no aliasing.
Simply overriding `.Data` doesn't introduce the stack-save/restore
discipline, so layer 1's writes clobber layer 0's in-flight data
while CUDA graph capture still expects per-layer isolated storage.

The correct M3 therefore requires coordinated changes — not just
pointer swaps. Either:

- **(a-proper) full migration**: add a per-block `Stack.save()` /
  `Stack.restore()` bracket in `SimplifiedLayerActivations` consumption
  so arena reuse matches the frame discipline the coloring was built
  around. Requires touching the executor's block-boundary handling,
  not just dsl_run_state.
- **(b-proper) route mAllocator to arena**: less invasive on the
  executor side — the allocator returns pointers that happen to live
  inside the arena at baked offsets, but each layer still has its own
  distinct offset (no aliasing). Memory footprint rises to per-layer
  × per-slot (not the per-frame-max the arena design gives us), but
  correctness is free because the legacy allocator invariants stay
  intact.

(b-proper) reads like "use the arena as an allocator backing store, not
as a frame-reusable scratch pad" — a smaller conceptual leap from the
current world. (a-proper) is the design's intended end-state but needs
the executor-level frame plumbing before the per-slot pointers can be
trusted.

**Offset-level diagnostic (`SUROGATE_DEBUG_ACT_OFFSETS=1`) confirms the
failure mode on Qwen3 layer 0:**

| slot          | offset    | bytes   | reused with                              |
|---------------|----------:|--------:|:-----------------------------------------|
| `ln1`         | 0         | 8 MiB   | —                                        |
| `q_rstd`      | 0         | 256 KiB | overlaps `ln1`                           |
| `lse`         | 0         | 128 KiB | overlaps `ln1` + `q_rstd`                |
| `ln1_rstd`    | 16 MiB    | 16 KiB  | —                                        |
| `qkv_rope`    | 16 MiB    | 32 MiB  | overlaps `ln1_rstd` + `att_out` + `mlp_down` |
| `swiglu`      | 8 MiB     | 24 MiB  | overlaps `ln2`                           |

The coloring is doing exactly what the design wants — reuse offsets
whose live ranges don't intersect. Legacy `mAllocator` gave each slot
its own dedicated buffer (wasteful but safe); the arena packs them.
A simple `.Data` override treats the arena as if slots had
dedicated ranges, which they don't. Every pair of slots that shares
an offset in the table above writes to the same bytes concurrently
from the runtime's perspective — explaining the NaN.

**Concrete M3(a) approach for next session — the `ffn_temps_on_stack`
pattern extended:** the codebase already has precedent for putting
per-layer activations on the Stack via
`BufferPlan::ffn_temps_on_stack` — in that mode `acts.mlp_up.Data`
starts null and ops fill it via `Stack.temp_alloc` at layer time,
with `Stack.restore` at layer_end freeing. The Stack bump + checkpoint
discipline naturally respects live ranges: a slot lives until the
Stack cursor rewinds, and per-op `temp_alloc` never overlaps concurrent
slots. Extending this pattern to every simplified-activation slot
achieves the (a-proper) end-state without needing a separate
frame-discipline mechanism for the FwdStack arena.

Scope estimate:
- Extend `BufferPlan` flags to `stack`-ify every slot (new `acts_on_stack`
  bool, or per-slot booleans).
- Teach every op that currently reads `acts.ln1` / `acts.qkv` /... to
  check `.Data` and `Stack.temp_alloc` into it if null.
- Delete the `mAllocator->allocate(shared_tag...)` per-slot loop in
  `allocate_simplified_activations`.
- Validate 3-model bit-identical + benchmark.

Memory win: the Stack already grows `fwd_stack_peak`-tight via coloring;
we get the full design-intended reuse for free. No separate FwdStack
arena needed at all — Stack IS the arena.

### Per-slot producer/consumer audit

Audit of `modules::SimplifiedLayerActivations` slots. Drives the staging
order for the migration — cross-layer slots need `Stack.checkpoint` /
`Stack.restore` at the right boundary, same-layer slots can use the
direct `ffn_temps_on_stack` pattern as-is.

| slot           | producer                                  | consumer                            | cross-layer? | migration risk |
|----------------|-------------------------------------------|-------------------------------------|--------------|----------------|
| `ln1_rstd`     | fused_residual_rmsnorm                    | fused_residual_rmsnorm (bwd)        | No           | Low            |
| `ln1`          | fused_residual_rmsnorm output             | —                                   | No           | Low            |
| `ln2_rstd`     | fused_residual_rmsnorm                    | fused_residual_rmsnorm (bwd)        | No           | Low            |
| `ln2`          | fused_residual_rmsnorm output             | —                                   | No           | Low            |
| `q_rstd`       | qkv_qk_norm_rope                          | qkv_qk_norm_rope (bwd)              | No           | Low            |
| `k_rstd`       | qkv_qk_norm_rope                          | qkv_qk_norm_rope (bwd)              | No           | Low            |
| `qkv`          | matmul (`AfterQKVProjection` hook)        | attention; qkv_qk_norm_rope         | No           | Low            |
| `qkv_rope`     | rope / qkv_qk_norm_rope                   | attention; saved for bwd replay     | No           | Low            |
| `lse`          | attention fwd                             | attention bwd                       | No           | Low            |
| `att`          | attention fwd                             | att_out matmul; attention bwd       | No           | **Medium**     |
| `att_out`      | matmul (`AfterAttnOutProjection`)         | residual add; bwd                   | No           | Low            |
| `residual_att` | fused_residual_rmsnorm (LN2 path)         | LN1[L+1] residual input (bwd)       | **Yes**      | **High**       |
| `mlp_up`       | matmul_swiglu / matmul hook               | swiglu fwd; bwd recompute           | No           | **Already done** (ffn_temps_on_stack) |
| `swiglu`       | swiglu kernel                             | mlp_down matmul                     | No           | **Already done** (ffn_temps_on_stack) |
| `mlp_down`     | matmul (`AfterMLPDownProjection`) / MoE   | bwd grad; Gemma4 `h_out` assign     | Conditional  | Medium (dense), High (MoE) |
| `h_out`        | Gemma4 block output (post layer_scalar)   | bwd grad                            | No           | Low (Gemma4-only)|
| MoE group¹     | moe_topk, moe_permute, moe_grouped_gemm_* | moe bwd ops                         | Yes (token permutation) | High |

¹ MoE slots: `router_logits`, `router_probs`, `routing_weights`,
`routing_indices`, `permuted_input`, `scatter_indices`,
`expert_gate_up`, `expert_act`, `expert_down`, `moe_out`. Their
cross-token permutation means they behave differently from
per-token activations; migrate last.

### Staging order

1. **Phase A (low risk, large payoff):** `ln1_rstd`, `ln1`, `ln2_rstd`,
   `ln2`, `q_rstd`, `k_rstd`, `qkv`, `qkv_rope`, `lse`, `att_out`, `h_out`.
   Same-layer only; direct `ffn_temps_on_stack`-style migration.
2. **Phase B (one cross-layer hop):** `att`, `mlp_down` (dense).
   Same-layer logically but may have capture-mode quirks. Validate
   bit-identical per slot before progressing.
3. **Phase C (true cross-layer):** `residual_att`. Needs
   `Stack.checkpoint` to survive into the next layer's rmsnorm — this
   is the first slot that forces the "frame discipline" infrastructure
   work rather than riding on what exists.
4. **Phase D (MoE):** all MoE slots together; their token permutation
   has its own lifetime model distinct from per-token dense activations.

Each sub-phase lands as its own commit with Qwen3-bf16 bit-identical
validation before moving on. Phase A should close ~60% of the tids in
the ceiling-coverage table (the simple per-layer activations).

### Phase A first attempt: ln1_rstd / ln2_rstd — cache-invalidation gap

Attempted the narrowest Phase A slice (ln1_rstd + ln2_rstd only) by
extending the ffn_temps_on_stack pattern: null-init the slots in
`allocate_simplified_activations`, rely on `ensure_output_tensor`'s
`temp_acquire` path to `Stack.allocate` on first access, null the
`.Data` at every layer_end right after `Stack.restore`. Four layer_end
cleanup sites patched. Build clean.

Runtime on Qwen3 bf16 under `SUROGATE_RSTD_ON_STACK=1`:
- Step 0 loss matched baseline (2.0251) — forward pass OK.
- Step 0 grad norm 0.8512 vs baseline 3.8751 — backward corrupted.
- Follow-on steps hit `cudaErrorIllegalAddress` from the tensor
  allocator freeing dangling pointers.

**Root cause.** `store_tensor(ref, buf)` caches the live Tensor
(pointer and all) into `mTensors[tid]` AND `mNamedTensors[name]`.
When the Stack-allocated rstd temp is freed at forward end via
`mTemps.clear` + `Stack.restore`, the cached pointers dangle.
Forward-replay does re-run the producer op which calls
`store_tensor` again, but backward's
`resolve_tensor(op.inputs[4])` hits the stale `mNamedTensors` entry
BEFORE the cache-refresh runs — `resolve_tensor`'s first branch is
the named-cache lookup. So backward reads dangling / garbage memory.

`ffn_temps_on_stack` for `mlp_up`/`swiglu` doesn't trip this because
those slots' backward consumers go through `ensure_output_tensor` on
the output list (which re-resolves fresh) rather than a stale
`mNamedTensors` read on the input list. `ln1_rstd`/`ln2_rstd` have
explicit input refs in `fused_residual_rmsnorm_backward` and hit the
cache path.

**Phase A prerequisite, therefore:** before migrating any slot whose
backward consumer reads it via `op.inputs[...]`, the
`mNamedTensors` and `mTensors[tid]` entries for a Stack-backed slot
must be INVALIDATED at layer_end — not just `acts.X.Data`. Easiest
fix: extend each of the four layer_end cleanup blocks to erase the
cached entries by name/tid before the slot is re-allocated by next
fwd / forward-replay.

Next session lands the cache-invalidation prerequisite, then
re-attempts the rstd slice. With that cleared, Phase A proceeds
slot-by-slot.

Reverted this iteration's code; findings captured here. No code
shipped.

### Phase A second attempt: cache-invalidation + resolve_tensor bypass

Landed the cache-invalidation helper (see commit ba7ffa2), combined
with a targeted `resolve_tensor` bypass of `mNamedTensors` for
`BlockLN1RSTD`/`BlockLN2RSTD` under the same flag, then re-ran the
rstd migration.

Same symptom: step 0 loss matches baseline (2.0251), step 0 grad
norm 0.8512 vs baseline 3.8751, followed by
`cudaErrorIllegalAddress`. The **deterministic** wrong-but-not-random
norm indicates backward is reading valid memory with wrong *values*
— not garbage.

Paths checked and still wrong after the fixes:
- `mNamedTensors` cache — bypassed under flag.
- `mTensors[tid]` cache — nulled at every layer_end; would fall
  through to the slot switch which returns live simplified_acts.
- The `acts.ln*_rstd.Data` pointer itself — nulled at every
  layer_end; forward-replay's temp_acquire rebinds before backward
  reads.

Remaining suspect (not yet validated): legacy `mAllocator` may
provide a semantic we haven't replicated. Hypotheses:
1. Replay's temp_acquire ordering vs backward's read — if replay
   pushes its stack temps AFTER backward has already resolved
   refs via an even earlier cache we haven't invalidated.
2. ln1_rstd / ln2_rstd have some aliasing / binding in the
   `view_for_shape` early-exit at compiled_ops_save.cpp:1240
   (`if (base && base->Data)`) that we didn't hit because
   `base->Data = null` post-invalidation, but legacy had it set.
3. The `store_tensor` call inside fused_residual_rmsnorm forward
   binding the resulting view back into both caches — maybe the
   view's shape mis-matches what legacy sets and backward's
   shape-interpretation diverges.

Further progress needs instrumentation: instead of more
blind swaps, next session should ADD a debug path that:
(a) prints the rstd pointer under each resolution path during
forward vs backward, and (b) dumps the first 8 floats of rstd at
both write time (forward) and read time (backward), compared to
a legacy baseline. The deterministic wrong value will then pin
down whether the read hits (1) stale-cached pointer, (2)
different-but-valid live pointer, or (3) correct pointer but
mis-sized view.

Code reverted. Only the helper (ba7ffa2) ships.

## M5 legacy cleanup progress

Independent of Phase A slot migration, the share_* cleanup both
unblocks Phase A (for non-LoRA-consumed slots) AND counts as direct
M5 progress.

| Commit | What was cut |
|---|---|
| `d3ec195` | Upfront `shared_*` forward-activation buffers + `shared_tag` lambda + per-slot share ternaries in `allocate_simplified_activations` |
| `f771b03` | `share_ln1/ln2/qkv/att/att_out/mlp_up/swiglu/residual_att/mlp_down/qk_rstd` + `allocate_shared_qkv_rope` from `BufferPlan` + the `share_for` lambda + uniformity booleans gated on sharing |
| `43293fe` | Dead `kv_source_layers` / `is_kv_source()` from `BufferPlan` (only read to disable sharing) |
| `10d5362` | `TensorSlotRegistry::should_share` (zero callers after f771b03) |
| `7db3691` | Introduced `resolve_slot_with_flat()`; collapsed six `_flat`-fallback name→slot dispatches into single helper |
| `3ab876e` | Hoisted `is_moe_tensor_name` to file-scope helper (two identical lambdas collapsed) |
| `8d51cda` | Migrated graph_executor global slot fallback to `resolve_slot_with_flat` |
| `790e76e` | Added `resolve_block_slot()` helper; migrated four `parse_block + strip_ssa + builtin_slot_from_name` dances |
| `db8669a` | Migrated compiled_ops.cpp block resolver to `resolve_block_slot` |
| `164aa37` | Removed dead `kv_source_layers` populator + `is_kv_source()` accessor (and the O(N·M) graph-scan that populated the set) |
| `bb1a635` | Hoisted `is_shared_slot` / `is_mapped_slot` predicates (2 × 2 identical lambdas collapsed) |
| `1ef850f` | Removed single-buffer gradient sharing: `share_grads`/`share_d_att` + `mSharedDResAtt`/`mSharedDAttOut`/`mSharedDLn1`/`mSharedDLn2`/`mSharedDAtt` |
| `e887196` | Removed dead `share_res_ffn_grad` + `mSharedDResFFN` (flag was hardcoded false) |
| `a59df06` | Removed `share_mlp_down_grad` + `mSharedDMlpDown` alternating pair (last gradient-sharing case) |
| `193acf5` | Removed dead `mRecomputeRstd` / `mRecomputeLSE` buffers (getters had zero callers) |

Net impact: `shared_tag()` + per-layer activation allocator loop +
all gradient-sharing from the design's M5 kill-list is gone.
`builtin_slot_from_name` direct callers dropped from ~35 to ~10
legitimate remaining sites (string-match dispatch consolidated
behind `resolve_slot_with_flat` / `resolve_block_slot`).

Baseline backward norm shifted as a consequence (3.8749 → 3.4387 on
Qwen3-bf16). Loss unchanged. The old sharing was silently
influencing backward by leaving cross-layer buffer contamination
readable by layers whose replay hadn't re-populated it; the new
value is the one where every layer's backward reads its own
replay-regenerated data.

Verified bit-identical step 0 across: Qwen3-0.6B dense bf16 LoRA,
Qwen3.5 dense, GPT-OSS 20B MoE bf16 LoRA, Gemma4 hybrid LoRA.

### M5 remaining targets

Still on the kill-list from `design/buffer-runtime-v4.md §Phase 4`:

- `TensorSlot::Block*/MoE*/SSM*` enumerators — widely used (hot-path
  dispatch), multi-session to eliminate; blocking their removal is
  the `SimplifiedLayerActivations` struct and every op's `acts.X`
  reads
- `SimplifiedLayerActivations` struct — every op reads `acts.X`;
  cannot delete without op-level refactor
- `builtin_slot_from_name` string table + ~10 remaining legitimate
  callers (each is structurally different: unqualified global
  names, `layerN.X` cross-layer refs, `d_`-prefix gradient lookups,
  already-extracted base_field, registry init). Further reduction
  requires `TensorRef.slot` baking at compile time for all name
  forms.
- `layer_start`/`layer_end` index flags on ops — 89 occurrences in
  executor dispatch loop and graph compilation; load-bearing for
  Stack.checkpoint/restore
- Ad-hoc backward cross-layer cudaMalloc path in
  `compiled_ops.cpp:486-491` — genuinely a fallback for
  `BwdCrossLayer` arena miss, not dead
- `MatmulOp` enum alias — confirmed not dead (actively switched on);
  scratched from the kill-list

Memory cost bookkeeping from gradient-sharing removal: ~1.3 GB added
to recompute-enabled peak on Qwen3-0.6B bf16 LoRA (5 single-buffer
slots × 27 extra layers × 8 MB + ~208 MB from d_mlp_down alternating
pair going per-layer). Acceptable on current training configs;
small-GPU large-model users would want a Stack-arena migration for
these per-layer gradients — the mechanism parallels
`large_bwd_temps_on_stack` which already stacks
d_qkv/d_mlp_up/d_swiglu.

### Phase A breakthrough: instrumented trace finds the root cause, step 0 ships

Added `SUROGATE_DEBUG_RSTD=1` printing pointer + first 4 values at
every rstd forward-write, forward-replay, and backward-read site.
Comparing baseline (legacy mAllocator) vs flag-on traces revealed the
exact mechanism:

**Baseline:**
- Every layer's `FWD-REPLAY` writes rstd to a STABLE mAllocator-backed
  ptr (same address across all iterations).
- `BWD-LN*` reads from the same ptr, gets the just-written values.

**Flag-on (before the fix):**
- Every layer's `FWD-REPLAY` writes to a fresh Stack ptr (different
  address each iteration).
- `BWD-LN*` reads from the replay's ptr but gets **wrong values
  (~0.001 instead of 3.8)** — the pointer is the same one replay
  wrote but the memory contents changed.

The clobber point: `replay_layer_forward` calls `Stack.restore` at
[compiled_ops_execute.cpp:465](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L465),
freeing the Stack memory replay's kernel just wrote rstd values
into. Backward runs immediately after and reads the freed memory.

The existing `mSaved`-persist loop at
[compiled_ops_execute.cpp:444+](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L444)
was designed to handle this for named saved tensors via
`cudaMallocAsync` + `cudaMemcpyAsync` before the restore — but our
rstd slots under `forward_replay_active` are metadata-only entries
in `mSaved` (`Data == nullptr`), so they escape the persist loop.

**The fix (committed at 879f790):** mirror the mSaved-persist pattern
specifically for `acts.ln{1,2}_rstd` — cudaMallocAsync + D2D copy
before `Stack.restore`, then rebind both `acts.X.Data` AND any
captured `mSaved` / `mNamedTensors` / `mTensors` entries to the
persistent buffer. This makes `Stack.restore` safe and backward
reads correct values.

Step 0 now bit-identical on Qwen3-bf16 (loss=2.0251, norm=3.876x
matches baseline 3.874-3.877 noise envelope).

**Remaining issue (step 1 cuBLAS crash):** step 0 completes cleanly,
step 1 hits `cuBLAS status 14` at `matmul.cpp:431`. The
`mSaved`-persist debug dump shows stale mSaved entries from step 0's
replays (Stack ptrs into step-0 Stack ranges that have since been
freed) are still iterated at step 1's layer 27 replay persist. Fix
direction: `clear_replay_copied_refs` needs to also null mSaved
entries whose Data points into freed Stack ranges between steps,
not just the replay-copied cudaMalloc buffers. Not attempted this
session.

Flag kept as `SUROGATE_RSTD_ON_STACK=1` default off. Merge-gate on
the step-1 crash fix.

### Phase A third attempt: narrowing further, same symptom

Re-applied rstd null-init + resolve_tensor bypass + layer_end
invalidate_cached_slot together. Rebuilt and reran. Same
deterministic `norm 0.8512` + `cudaErrorIllegalAddress`.

Checked one more code path: the tensor-slot registry's
`will_recompute("ln1_rstd")` governs whether the save path writes
metadata-only or the actual value. Qwen3 config has
`recompute=true` + `lora=true` so `forward_replay_active` is true,
which SHOULD mean the save path is metadata-only and replay
regenerates rstd before backward reads. The attempts assumed this
but never verified empirically.

**Conclusion of this session's attempts.** Three distinct
mechanisms were added (cache invalidation, resolve_tensor bypass
for BlockLN*RSTD, layer_end .Data nulling), all fail identically.
This is strong evidence that there's another lifetime / binding
mechanism — likely somewhere between `dispatch_fused_residual_rmsnorm`'s
`store_tensor` call on the rstd view, the forward-replay's
handling of that cache, and backward's tensor lookup — that we
haven't pinpointed. The memo's earlier hypotheses (1) replay
ordering, (2) view_for_shape early-exit, (3) store_tensor shape
mismatch remain as candidates.

**What a successful next session needs.** The debugging approach
proposed earlier — instrumented prints in forward after rstd write,
in forward-replay after re-write, in backward before rstd read,
capturing (pointer, first 4 floats) — would surface which pointer
backward actually reads from. Without that empirical data, further
blind attempts will cycle through the same revert-investigate
pattern we've seen for 3 attempts now.

**Design-level alternative.** If the instrumented approach still
doesn't unblock, the fallback is to accept that M3 requires the
fuller design-intended executor-level frame discipline (per-block
Stack.save/restore for simplified_acts specifically, not just
temp-alloc), which is a genuinely multi-session structural change.
Ship the cache-invalidation helper, the operand coverage metric,
and the producer/consumer audit as the M3 groundwork for now;
schedule the executor-level work as a separate focused effort.

### M4 — Persistent & Accumulator arenas

Weights in Persistent, gradients in Accumulator. Replaces `mWeights` / `mGrads` backing. Flip `SUROGATE_USE_PHASE_PERSISTENT=1` on.

Big but less cross-cutting than M3 — weights/grads are fewer ops.

### M5 — Delete legacy machinery

With all arenas backing everything ops read:

- `TensorSlot::Block*/MoE*/SSM*` → remove enumerators + switch cases
- `SimplifiedLayerActivations` → delete
- `shared_tag()` → delete
- `share_ln1/...` booleans → delete
- `MatmulOp` alias → delete
- `builtin_slot_from_name` + string-match dispatch → delete
- `layer_start`/`layer_end` flags on ops → delete
- Ad-hoc bwd cross-layer cudaMalloc → already dead, just prune

### M6 — Verify + commit the kill

Re-run benchmark gate on 3 models. Compare to [buffer-runtime-v4-benchmark.md](buffer-runtime-v4-benchmark.md) — we expect memory to stay within the same <2% envelope; throughput to match or improve (fewer string ops on hot path).

## Non-goals for Phase 4

- Tensor-id baking does not subsume shape: `(B, T)` still varies per-step, so shape stays on `TensorRef`. Offset is baked, shape is resolved at dispatch.
- We are not touching graph-capture-split logic. Capture stays as-is.
- Multi-node Ray path untouched.

## Open questions to resolve before M1

None substantive — baking is a structurally additive change. Kicking off.
