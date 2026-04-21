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
| `d8ef2e7` | Added between-step Stack-owned mSaved scrub (defensive; closes latent stale-ptr bug class) |
| `9656ce3` | Dropped stale recomputation comment on mFP8ScalingState |
| `70243c6` | Removed dead `TensorSlot::Temporary` enum value + its 6 dead checks across buffer_plan, graph_compiler, compiled_ops_save, fused_residual_rmsnorm |
| `6530ace` | Consolidated `resolve_tensor`'s 40-case slot switch to delegate Block*/MoE*/BlockD*/global cases through `block_activation_ptr` / `block_gradient_ptr` / `global_activation_ptr` helpers. Struct-field indirection now referenced in exactly one place (`tensor_slot_dispatch.cpp`), groundwork for future SimplifiedLayerActivations replacement |
| `939a6f3` + `d0bfc0b` | Migrated 17 direct ops-side `simplified_acts(L).X` / `simplified_grads(L).X` accesses to `block_activation_ptr` / `block_gradient_ptr` helpers across fused_residual_rmsnorm, matmul, matmul_swiglu, add, moe_unpermute |
| `05c9b8c` | Migrated graph_executor + graph_executor_tensors to dispatch helpers. The four string-match dispatch functions in graph_executor_tensors (resolve_block_activation_tensor, block_activation_base_ptr, resolve_recomputed internals, resolve_block_gradient_tensor) each had 15-17 if-branches mapping field names to struct fields; replaced with single `builtin_slot_from_name → block_activation_ptr` calls. Net −90 lines |
| `2969a2e` | Migrated compiled_ops_execute.cpp (14 sites). Introduced file-scope `clear_rstd_stack_slots` / `clear_ffn_temp_stack_slots` / `clear_large_bwd_grad_stack_slots` helpers; the 9-slot rstd + 2-slot ffn + 3-slot bwd clear patterns each appeared 4× verbatim in the dispatch loop. Net −12 lines plus massive deduplication |
| `0f58c25` | Migrated compiled_ops_save.cpp. Collapsed the third clone of the 35-case Block*/MoE*/BlockD*/global dispatch switch — now delegated through the helpers. Net −21 lines |

**Executor migration complete.** SimplifiedLayerActivations / SimplifiedLayerGradients struct fields are now accessed through dispatch helpers from every caller except dsl_run_state.cpp (where the storage lives). Struct storage can now be swapped without touching any consumer — the precondition for `TensorSlot::Block*` enum removal.

### Layer-boundary bookkeeping consolidation

Separate from struct-field access cleanup: the layer-boundary
`Stack.checkpoint` / `Stack.restore` / stack-slot clear sequence was
duplicated across multiple dispatch paths. Consolidated:

- **Forward** (`on_fwd_layer_start` / `on_fwd_layer_end` helpers in
  execute_forward): PhaseEnter stream-driven path + flat-ops-loop path
  (start+end) now route through two lambdas. Tiled-MLP path stays
  inline (deliberately simpler: no MoE save, no stack-slot clears —
  those conditions never fire on tiled-MLP configs).
- **Backward** (`bwd_layer_end_cleanup` helper): legacy
  non-stream-driven backward path's ~146-line inline block collapsed
  into a single call. Added `capturing` parameter so the helper
  handles both the stream-driven (never capturing by construction)
  and legacy (may be capturing) paths. Capture-unsafe work
  (BwdCrossLayer persist, handle_layer_end, grad reduce/offload)
  is skipped when `capturing=true`; the capture-safe tail
  (Stack.restore, slot clears) always runs.

Net effect: layer-boundary bookkeeping has a single source of truth
per direction. Adding a new stack-backed slot category (or changing
the order of cleanups) is a one-place change.

Dispatch consolidation commits:
- `f6dc9a7` Forward: `on_fwd_layer_start` / `on_fwd_layer_end`
  extracted; 3 call sites migrated (~50 lines removed)
- `5b70b77` Backward: legacy-path inline block replaced with
  `bwd_layer_end_cleanup(L, idx, capturing)` call (~137 lines removed)

### Dead struct-field removal (Mamba / swiglu_scale)

The Mamba / SSM fields on `SimplifiedLayerActivations`
(`mamba_gate` / `mamba_conv_in` / `mamba_u` / `mamba_delta` /
`mamba_B` / `mamba_C` / `mamba_scan_out` / `mamba_gated` /
`mamba_normed` / `mamba_rstd` / `mamba_x`) and the matching
`d_mamba_*` fields on `SimplifiedLayerGradients` plus `swiglu_scale`
were declared in anticipation of per-field access patterns that
never materialized. Mamba ops route per-layer tensors through
`resolve_tensor` on `blocks[N].mamba_*` / `d_blocks[N].mamba_*`
names — they never touch these struct members.

Commit:
- `e8fe8c8` Delete 11+1 activation fields, 8 gradient fields, and 8
  `.Data = .Data` no-op lines from `reset_simplified_gradients`.
  Struct-field footprint drops to exactly the slots that the
  member-pointer tables in `tensor_slot_dispatch.cpp` cover.

**The struct is now a minimal, closed set of fields that every
caller reaches through the dispatch helpers.** Full storage
migration (named fields → `std::array<Tensor, N>` indexed by
`TensorSlot`) is a local change to `dsl_run_state.cpp`'s
allocation loop plus the member-pointer tables; no consumer code
needs to change.

### Storage migration (0493595)

Replaced the ~25 named Tensor fields on each struct with a
`std::array<Tensor, kSize>` indexed by `dsl::TensorSlot` (kSize =
`TensorSlot::Mapped + 1`), plus `operator[](TensorSlot)` for
ergonomic access. The struct's layout is now driven entirely by
the `TensorSlot` enum — they are the same data.

- `tensor_slot_dispatch.cpp`: member-pointer tables replaced with
  `is_block_activation_slot` / `is_block_gradient_slot` predicates
  + direct `acts[slot]` / `grads[slot]` lookups.
- `dsl_run_state.cpp` allocation: helper lambdas collapse the
  repeated if/else stack_or_alloc ternaries; `acts.field = ...`
  sites become `acts[TensorSlot::BlockField] = ...`.
- `dsl_run_state.cpp` reset: 10 per-field `.Data = .Data` lines
  collapsed into a loop over a slot-enum list.

Adding a new block slot is now a one-line enum addition + one-line
allocation site. No struct field, no member-pointer table entry,
no switch case. Storage footprint grows ~120 KB per model (Tensor{}
padding for unused enum indices on 28 layers × 2 structs);
negligible vs total memory.

Verified bit-identical on all four architectures (Qwen3 / Qwen3.5 /
Gemma4 hybrid / GPT-OSS MoE).

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

Still on the kill-list from `design/buffer-runtime-v4.md §Phase 4`
— each is blocked on architectural changes larger than this
session's scope:

- `TensorSlot::Block*/MoE*` enumerators — after `6530ace` the
  struct-field indirection is consolidated to one place (the
  `block_activation_ptr` / `block_gradient_ptr` switches), but the
  enumerators themselves are still load-bearing for that dispatch.
  Removing them requires replacing the switch with an indexed
  lookup (e.g., `std::array<Tensor, kNumBlockSlots>` per layer) AND
  updating the 17 direct `acts.X` ops-side accesses that bypass the
  helpers.
- `SimplifiedLayerActivations` struct — also holds Mamba/SSM
  fields that have no TensorSlot enum equivalent (mamba_gate,
  mamba_u, etc.). Removal is not 1:1 with the Block* enum — those
  Mamba ops need a parallel dispatch mechanism first.
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

### Step-1 crash under `SUROGATE_RSTD_ON_STACK=1`

With `CUDA_LAUNCH_BLOCKING=1` the crash pinpoints to
`cudaGraphLaunch` at
[graph_executor_utils.h:144](../csrc/src/runtime/executor/graph_executor_utils.h#L144).
Root cause is the persist_stack_slot path calling
`cudaMallocAsync` inside what becomes a captured CUDA graph:
captured graphs bake the pointer at capture time, but the
allocation is re-done on each replay — so later captured nodes
read a stale pointer. This is architectural: the rstd-on-stack
flag cannot ship until `persist_stack_slot` switches to a
pre-allocated scratch arena (analogous to how `BwdCrossLayer`
went from cudaMalloc fallback to arena-first). Flag stays
default-off; step 0 bit-identical under it but step 1+
train_step_graphed replay breaks.

**Fix shipped** (`4da0ae2`): 256 MiB replay-persist arena
replaces `cudaMallocAsync` in both persist loops. Base pointer
is stable across captures/replays; bump offset resets at each
replay entry (via `clear_replay_copied_refs`) and at each
`execute_forward` start. Eliminates the capture-time
cudaMallocAsync pointer-baking issue. Step 0 under the flag
stays bit-identical on all 4 architectures.

**Instrumentation clears the apparent step 1+ drift**: multi-run
comparison shows that the apparent drift was within baseline
self-variance. With CUDA graphs disabled,
`SUROGATE_RSTD_ON_STACK=1` runs cleanly through 5 steps with loss
values indistinguishable from the default path (flag-on step 1
1.6168 ∈ default range 1.6162–1.6170; flag-on step 2 1.5356–1.5361
∈ default range 1.5356–1.5373). No correctness bug in the
forward-replay path.

**Remaining step-1 crash under CUDA graphs**: with
`use_cuda_graphs: true` (default), `SUROGATE_RSTD_ON_STACK=1` faults
at step 1 replay (`cudaEventSynchronize` at
graph_executor.cpp:147). The arena fix addressed one class of
captured-graph pointer issues (cudaMallocAsync baking); the
remaining failure is in a different class — a Stack pointer
captured during step 0's backward graph that doesn't point at live
memory at step 1 replay. Likely needs explicit Stack-pointer
stabilization logic for the captured backward (Stack checkpoint/
restore semantics vs captured-graph pointer lifetime). Flag stays
default-off.

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

### Phase A shipped: shrink-stack invalidates cached Stack pointers

The residual step-1 crash under CUDA graphs had a straightforward
root cause: `DslRunState::shrink_stack_to_high_water_mark()` runs once
after warmup (step 0) and calls `resize_stack_to()`, which `free()`s
the old Stack buffer and allocates a new one at a (usually different)
base pointer. Any `simplified_acts[].Data` still pointing into the
old Stack range is now dangling — and when the new cudaMalloc reuses
overlapping memory, step 1 replay reads garbage or faults.

The classic `SUROGATE_DISABLE_SPLIT_SEG=1` run pinned the fault at
`rmsnorm.cu:452` with `cudaErrorIllegalAddress` on step 1, and a
one-line `SUROGATE_SKIP_STACK_SHRINK=1` probe confirmed the trigger —
disabling the shrink made flag-on bit-identical-within-noise through
5 steps.

**Fix shipped** ([48111be](../csrc/src/runtime/dsl/dsl_run_state.cpp)):
`resize_stack_to()` now snapshots the old Stack `[base, base+capacity)`
before freeing, walks every layer's `simplified_acts` / `simplified_grads`
slot array, and nulls any `.Data` that falls in the old range. A
`SUROGATE_DEBUG_STACK_RESIZE=1` env prints per-slot scrubs for
diagnosis. On Qwen3-0.6B the scrub catches 54 slots (27 layers × the
stack-resident `BlockLN1` + `BlockLN1RSTD` pair) that
`clear_rstd_stack_slots` fails to null at backward layer_end — a
latent bug that the scrub masks, worth tracking down in M5 but
immaterial for Phase A correctness.

**Flag removed** ([dc712ab](../csrc/src/runtime/executor/compiled_ops_execute.cpp)):
with the scrub in place, `SUROGATE_RSTD_ON_STACK` is unconditionally
on. The env var, the static `rstd_on_stack_enabled()` guard, and
every `if (rstd_on_stack_enabled())` gate are gone. `persist_stack_slot`
in `replay_layer_forward` fires for every layer; `clear_rstd_stack_slots`
runs at every fwd/bwd layer_end; the `bypass_named_for_rstd` path in
`resolve_tensor` is unconditional for the 9 stack-resident slots.

**Validation** (all fresh output dirs, bf16 LoRA recompute):
- **Qwen3-0.6B** (dense, 5 steps): loss envelope 1.6161–1.6176 at
  step 1 vs pre-change 1.6162–1.6173 — overlapping, no drift.
- **Qwen3.5-0.8B** (dense, 3 steps): 1.9893 → 1.6714 → 1.3969, no
  faults, no illegal access warnings.
- **GPT-OSS-20B** (MoE + MXFP4 QLoRA, 3 steps): loss 1.7916 → 2.1277
  → 1.9984 with aux-loss + imbalance metrics tracking cleanly.
- **Gemma4** remains a separate WIP blocker (pre-existing matmul rank
  mismatch; not introduced by this change).

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

### M4 shipped so far

- **UnifiedStack arena default-on** (`30cd8c3`, prereq `1430d22`):
  Split the `SUROGATE_USE_PHASE_STACK_ARENAS` gate so UnifiedStack (the
  Stack backing DslRunState actually consumes) is unconditional, while
  FwdStack / BwdStack shadows stay behind the env flag. Adopted-arena
  Stack is fully shrinkable after warmup (`shrink_stack_to_high_water_mark`
  taught to use `Stack.capacity()` and branch resize on
  `mOwnedExternalStack`). Memory-neutral on Qwen3-0.6B (328 MiB),
  Qwen3.5-0.8B (187 MiB), GPT-OSS-20B (29.8 GiB peak, identical to
  pre-flip). Losses bit-identical within noise.

### M4a shipped (Persistent arena for base weights — post-compile rebind)

`DslParamStore::rebind_to_persistent_arena` walks every locally-allocated
entry (skipping QLoRA-external and `managed_by_weight_manager`), looks up
its tid in the compiled forward graph, and for every tid whose
`RegionKind::Persistent` offset resolves inside the arena:
`cudaMemcpyAsync` the current bytes to `persistent_ptr + meta.offset`,
`mAllocator->free` the original buffer, and rebind
`entry.tensor.Data = arena_ptr`. `entry.tensor.Stats` and `Device` are
preserved across the rebind.

Call site: `GraphExecutor::compile_graphs`, immediately after
`consume_fwdstack_arena()`. Runs on the `mRunState.MainStream` so the
copy is ordered with the rest of the executor's work.

Arena sizing gate is split in two:

- `compute_arena_sizes` now reads `SUROGATE_USE_PHASE_PERSISTENT=1` and
  `SUROGATE_USE_PHASE_ACCUMULATOR=1` independently — M4a only wants the
  weight path, not gradients.
- `GraphExecutor::compile_graphs` clamps
  `mPhaseArenas.persistent_bytes` to
  `DslParamStore::rebindable_persistent_bytes(*mCompiledForward)`, which
  returns 0 as soon as **any** entry is external or weight-manager-
  managed. Layout's bump allocator assigns offsets to every ForwardParam
  tid, so the locally-allocated set's offsets are interspersed with
  QLoRA-external ones; sizing to just the locals' high-water mark would
  leave the external ranges unbacked. Whole-or-nothing keeps M4a sound;
  M4c/d widen it.

Validation (parallel GPU 1 / 2 / 3):

| Model     | Flag off loss  | Flag on loss   | Flag on grad norm | Arena bytes | Rebound |
|-----------|---------------:|---------------:|------------------:|------------:|--------:|
| Qwen3 0.6B    | 2.0251 | 2.0251 | 3.4382 (base 3.4390 ∈ noise) | 1.46 GiB | 227 |
| Qwen3.5 0.8B  | 1.9893 | 1.9893 | 4.1787                        | 1.88 GiB | 261 |
| GPT-OSS 20B   | 1.7873 | 1.7873 | 2.7290 **exact**              | 0        | 0 (gate) |

GPT-OSS's skip is by design: `is_qlora_param_name` marks every MoE
expert weight `external`, the gate returns 0, and arena allocation is
suppressed — no 40 GiB cudaMalloc attempted.

Qwen3.5 reports 36 `skipped_size_mismatch`. These are ForwardParam tids
whose `tensor_meta[tid].bytes == 0` (never referenced by any op in the
active graph — a pre-existing quirk of the hybrid architecture's
per-layer parameter inclusion). They stay in the allocator unchanged;
`skipped_size_mismatch` being non-zero is not a correctness issue but
is worth auditing in M5 cleanup.

Commits: implementation in `dsl_param_store.{h,cpp}` +
`graph_executor.cpp` call site + `graph_compiler.cpp` env-gate split.
Gated behind `SUROGATE_USE_PHASE_PERSISTENT=1`, default off.

### M4b shipped (LoRA adapter weights into Persistent arena)

LoRA adapters don't appear as `ForwardParam` tids in the compiled graph
— they're injected at runtime via forward hooks
(`AfterQKVProjection`/`AfterAttnOutProjection`/`AfterMLPUpProjection`/
`AfterMLPDownProjection`) and live entirely in
`ModularLoRAWeightsManager::mMaster` / `mWork`. The rebind therefore
can't reuse the compile-time `tensor_meta` offsets — instead, M4b
reserves a contiguous **slab at the tail of the Persistent arena**
sized from the LoRA manager's own inventory.

New methods on `ModularLoRAWeightsManager`:
- `total_persistent_bytes()` walks every leaf tensor (master + work,
  across blocks / attention / MLP / MoE-shared / MoE-experts /
  MoE-grouped / router) and sums byte sizes.
- `rebind_to_persistent_arena(arena_base, max_bytes, stream)` iterates
  the same tree, bump-allocates each tensor inside the slab,
  `cudaMemcpyAsync`'s live bytes, `mAllocator->free`'s the original,
  and rebinds `.Data`. Preserves `Stats`/`Device` as M4a does.

Call-site changes:
- `dsl_model_execution.cpp`: `set_lora_state` is now called **before**
  `ensure_graphs_compiled` with the weights manager (grads/run-state
  still pass nullptr; re-set fully after compile). This lets
  `compile_graphs` size the slab from `mLoRAWeights->total_persistent_bytes()`.
- `graph_executor.cpp::compile_graphs`: after M4a's clamp to
  `base_persistent_bytes`, appends `lora_slab_bytes` to
  `mPhaseArenas.persistent_bytes`. After base-weight rebind,
  calls `mLoRAWeights->rebind_to_persistent_arena(persistent_ptr +
  base_persistent_bytes, lora_slab_bytes, stream)`.

Works for all four cases:
1. Base local + LoRA (Qwen3 / Qwen3.5 bf16 LoRA): arena = base + LoRA.
2. Base external (QLoRA) + LoRA (GPT-OSS MXFP4): arena = LoRA only;
   base stays in allocator (as M4a already did).
3. Base local, no LoRA: arena = base (M4a behavior unchanged).
4. Base external, no LoRA: arena = 0 (unchanged).

Validation (parallel GPU 1 / 2 / 3, flag-on):

| Model     | Base rebound | LoRA rebound | Base bytes | LoRA bytes | Loss     | Notes               |
|-----------|-------------:|-------------:|-----------:|-----------:|---------:|---------------------|
| Qwen3 0.6B    |  227 |  784 | 1.46 GiB | 57.8 MiB  | 2.0251 step 0, 1.3377 step 4 | 5 steps clean |
| Qwen3.5 0.8B  |  261 |  384 | 1.88 GiB | 36.6 MiB  | 1.9893                       | step 0 match  |
| GPT-OSS 20B   |    0 |  576 | 0        | 705 MiB   | 1.7873, norm 2.7290 (exact)  | QLoRA skip OK |

Qwen3 flag-off baseline step-0 norm 3.4395; flag-on 3.4383 (Δ = 0.03%,
parallel-run noise envelope).

### M4c shipped (DslWeightManager master/work into Persistent arena)

`DslWeightManager::allocate_weights` creates per-weight master + work
tensors via `mAllocator->allocate`. Out of its five paths — streaming,
offload_master, cpu_training, separate-work (dtype mismatch),
shared-alias — M4c rebinds **every device-resident pair**, skipping
offloaded (pinned CPU) masters and streaming work (`.Data == nullptr`
prefetch slots). The fall-through handles dtype-mismatch (fp32-master /
bf16-work mixed-precision training) directly; streaming-sharded masters
still rebind (they're device-local even when work buffers aren't).

New methods on `DslWeightManager`:
- `total_persistent_bytes()` — sums bytes of every device-resident
  tensor across `mWeights` (master + work), deduping any aliased
  pointers so we don't double-count the non-`separate_work` case
  (shouldn't happen — that path uses DslParamStore — but defensive).
- `rebind_to_persistent_arena(arena_base, max_bytes, stream)` — walks
  `mWeights` in deterministic `mParamOrder`, bump-allocates each
  eligible tensor, `cudaMemcpyAsync`'s current bytes to the slab,
  `mAllocator->free`'s the original, rebinds `.Data`. Aliased
  buffers are detected by pointer equality and reuse the same slab
  slot (single free). Preserves `Stats` + `Device`.

Arena layout extended to three slabs:

1. `[0, base_persistent_bytes)` — M4a base weights (layout offsets).
2. `[base_persistent_bytes, + wm_slab_bytes)` — M4c DslWeightManager
   pairs (bump offsets inside the slab).
3. `[..+lora_slab_bytes)` — M4b LoRA adapters.

Wiring: `dsl_model_execution.cpp` sets the weight manager on the
executor **before** `ensure_graphs_compiled`, so `compile_graphs` can
compute the slab size upfront. Removed the redundant late
`set_weight_manager` call (idempotent; propagation to
`mCompiledExecutor` happens inside `init_compiled_execution`).

Validation (fp32-master / bf16-work qwen3 LoRA, 5 steps, parallel GPU):

| Step | Base loss | Flag loss | Base norm | Flag norm |
|-----:|----------:|----------:|----------:|----------:|
|  0   | 2.0251    | 2.0251    | 3.4393    | 3.4387    |
|  1   | 1.6158    | 1.6162    | 2.3241    | 2.3263    |
|  2   | 1.5360    | 1.5355    | 1.9830    | 1.9392    |
|  3   | 1.3975    | 1.3974    | 1.6312    | 1.6128    |
|  4   | 1.3373    | 1.3371    | 1.4493    | 1.4362    |

Loss deltas ≤ 0.03% through step 4. 227 WM-master (fp32) + 227
WM-work (bf16) rebound into a 4.31 GiB slab; 784 LoRA tensors into
57.8 MiB. `DslParamStore` correctly reports `skipped_managed=227`
(no locally-allocated base params; all are deferred to the weight
manager). Flag stays default off.

Regressions checked: Qwen3 / Qwen3.5 / GPT-OSS flag-on (no weight
manager configs) continue to produce identical rebind counts and
losses as before M4c.

### M4c2 shipped (streaming prefetch buffers)

`total_persistent_bytes()` and `rebind_to_persistent_arena()` now also
walk `mPrefetchBuffers[kNumPrefetchBuffers]`. Each slot's per-base-name
Tensor entries are passed through the existing `rebind_one` helper, so
the pointer-dedup map correctly collapses the many-to-one alias pattern
(all layers sharing a given base-name point at a single slot buffer
per prefetch slot). The first occurrence of each unique buffer does
the memcpy+free+rebind; later aliased entries simply repoint `.Data`.

Pinned-CPU masters (`offload_master=true` block weights) remain out of
the arena by design — arena storage is device-only, and moving pinned
storage to UVA would defeat the offload savings the flag was set to
obtain. They continue to hit the `skipped_offloaded` counter.

Validation: single-GPU qwen3 + `offload_master: true` + LoRA, 5 steps
flag-on. 3 WM-master (non-block embeddings / final_norm / lm_head) +
3 WM-work + 448 prefetch-entry rebinds (224 per slot × 2 slots,
collapsing to a handful of unique per-base-name buffers) + 784 LoRA
into a 1.31 GiB slab. 224 block masters skipped as offloaded; 224
streaming work entries skipped (filled at gather time into the
now-arena-backed prefetch slots). Loss trajectory 2.0251 → 1.3369,
norms within the baseline run-to-run envelope (baseline step 2 norm
varies 1.83–2.06 across three back-to-back runs; flag-on clusters
1.83–1.86). Flag stays default off.

### M4d shipped (quantized weights into a self-managed arena)

QLoRA base weights (FP8 / FP4 / BnB / MXFP4) live in
`GenericWeightManager::mWeights[name].quantized` as device-resident
`data` / `scales` / `meta` / `meta2` tensors, allocated via the shared
`TensorAllocator` during `import_and_quantize`. M4a's `compile_graphs`
hook couldn't reach them because `set_qlora_provider` runs **after**
the first `allocate_run_state` (which is what triggers `compile_graphs`);
by the time the executor sized the Persistent arena, the provider was
still null.

Rather than plumb a post-compile rebind hook back into the executor,
M4d gives `GenericWeightManager` its own dedicated arena:

- `total_persistent_bytes()` / `rebind_to_persistent_arena()` walk
  every device-resident owned tensor (`quantized.{data,scales,meta,meta2}`,
  `dequant_buffer`, `full_precision`), skipping host-backed entries
  (`quantized_has_host_storage`) and deduping by pointer.
- `consume_self_arena(stream)` — reads `SUROGATE_USE_PHASE_PERSISTENT=1`,
  checks free GPU memory against `needed + 1 GiB safety margin`, and
  (if it fits) `cudaMalloc`s a dedicated slab, then runs the rebind.
  Skipped with a log when the 2× transient peak would OOM.
- Stored in `mSelfArena` / `mSelfArenaBytes`; freed in the destructor
  alongside the transpose-temp buffer.

Wired from `DslModel::import_weights` immediately after
`mQLoRAProvider->import_and_quantize` completes. Exposed on
`QLoRAWeightProvider` as `virtual void consume_self_arena(stream)`;
`GenericQLoRAProvider` forwards to the weight manager.

Validation (flag-on):
- Qwen3 QLoRA-BnB (0.6B, NF4): 448 quantized + 115 full-precision
  tensors rebound → 810 MiB self-arena; 784 LoRA → 57.8 MiB Persistent
  slab. 5-step losses 2.2451 → 1.5164 (baseline 2.2451 → 1.5169;
  Δ ≤ 0.3%).
- Qwen3 QLoRA-FP4 (0.6B, E2M1 2D-block): 336 quantized + 115 full-
  precision → 830 MiB; 784 LoRA. Step-0 loss 2.1914 matches expected.
- GPT-OSS MXFP4 (20B): `consume_self_arena` requests 13 GiB, free is
  9 GiB (LoRA + other arenas already allocated) — auto-skipped.
  Training proceeds on the allocator-backed storage; loss 1.7873,
  norm 2.7290 bit-identical to baseline.

Non-QLoRA regressions (Qwen3 / Qwen3.5 / fp32-master mixed-precision
Qwen3 / offload_master Qwen3) unchanged — same rebind counts and
losses as before M4d.

Flag stays default off.

### M4 remaining (multi-session)
- **Accumulator arena for grads.** `DslGradStore` has a similar
  allocate-per-grad pattern; ZeRO-2 sharding + per-layer reduce add
  complexity. Gated separately now under
  `SUROGATE_USE_PHASE_ACCUMULATOR=1` (shadow; no op consumes it yet).
- **FwdStack / BwdStack shadows.** Blocked on M3 per-frame coloring
  work — a naive flip aliases each layer's same-slot tid into one arena
  offset and corrupts forward accumulation (see M3 first-strike memo).

### Direction reset: commit to full Phase 1–3 of design/buffer-runtime-v4.md

The M5 cleanup track (kill TensorSlot/shared_tag/builtin_slot_from_name)
is *gated on* the phase-tree runtime being in place — per the design,
legacy machinery can only be retired once ops read from baked
`(region, offset)` operands instead of named slot lookups. Previous
sessions treated the arena scaffolding as dead-when-unused and
trimmed it (FwdStack/BwdStack purge, commits `96d7f18` / `fdd5a5a` /
`a2038b6`); those were reverted in `156c1c8` / `766329e` / `bf8954f`
because the scaffolding is load-bearing for the design, not dead code.

Current state vs design phases:
- **Phase 1 (IR + regions, role unification):** IR + region derivation +
  layout + instruction stream all shadow-live. `MatmulRole` typed enum
  not started (still `modules::MatmulOp` on hot paths).
- **Phase 2 (compile-time layout + coloring):** bump + per-frame
  coloring shadow-live for all five populated regions. Alignment
  constraints not modeled. Determinism hash not implemented.
- **Phase 3 (runtime migration):** stream interpreter default-on.
  Handles `PhaseEnter` / `PhaseExit` / `SegmentDispatch`; `PruneByLastUse`
  is pass-through, `RecomputeBlock` is forward-no-op. SaveForBwd tid
  baking shipped (`66a69c0`). UnifiedStack adopted as Stack backing
  (`30cd8c3`). Ops still largely use legacy `resolve_tensor` named
  lookups for FwdStack/BwdStack/Accumulator operands.
- **Consumers of arenas:** SaveForBwd, BwdCrossLayer, MoeSaved,
  UnifiedStack all live. Persistent / Accumulator allocated under env
  gate, no op reads them. FwdStack / BwdStack shadow.

Concrete remaining commits toward Phase 3 completion:

1. ✅ Layout determinism hash — `eea6650` (single-rank invariant;
   MPI_Allreduce wiring deferred to distributed testing).
2. ✅ Frame-coloring liveness validator — `cf944d5`
   (`SUROGATE_CHECK_FRAME_COLORING=1` → pairwise overlap check).
3. ✅ Runtime op-io-aliasing validator — `b0eed6c`
   (`SUROGATE_CHECK_OP_IO_ALIASING=1` → per-op input/output overlap
   check at dispatch; `=abort` for hard fail).
4. ✅ Slot-alias offset-split validator — `b5d1a9e` (compile-time check
   that aliased tids share offsets; 280 / 114 / 168 splits on Qwen3 /
   Qwen3.5 / GPT-OSS under the pre-fix coloring).
5. ✅ Coloring collapse for slot-aliased tids — `770492a` (groups
   `{blocks[L].swiglu, blocks[L].swiglu_flat}` etc. into one coloring
   unit; all three architectures now report 0 splits / 0 violations).
6. `MatmulRole` typed enum (Phase 1 step 4) — either rename+expand
   `modules::MatmulOp` or introduce a parallel enum and migrate callers
   one hot path at a time. **Not started.** Cosmetic / cleanup-class;
   defer to M5 sweep.
7. `PruneByLastUse` real dispatch — move per-op `prune_by_last_use` out
   of the legacy backward loop into the instruction stream. **Attempted
   and reverted in compiled_ops_execute.cpp:2666 (no-op handler +
   inline per-op prune):** batching to the instruction caused 3×+ Stack
   bloat on Prologue-heavy graphs (MoE, hybrid Mamba) because hundreds
   of Prologue ops accumulate before any gets freed. Inline prune stays;
   the instruction is kept in the stream as a shape marker for
   validation only.
8. `RecomputeBlock` real dispatch — consume the instruction in backward
   instead of the current `mRecomputeFn` on op.layer_start.
   **Partially shipped** (compiled_ops_execute.cpp:2597–2603): the
   instruction's explicit handler calls `mRecomputeFn` when the block
   index changes. Per-op fallback triggers at lines 2622–2632 /
   2741–2748 / 2782–2796 / 2808–2827 are retained as safety nets for
   graphs where ops with `layer_start < 0` land outside a BwdBlock
   phase-tree bucket (GPT-OSS LM-head / MoE EP Prologue). Removal of
   those fallbacks is gated on explicit backfill of the instruction
   stream for those paths — left as a follow-up.
9. ✅ **Actual FwdStack/BwdStack arena consumption — shipped default-on**
   (`cb3f2da`). Single arena per region, peak = max over layers, not
   sum. Route every stack-backed `simplified_acts[L][SLOT].Data` to
   `fwd_stack_ptr + meta.offset` and set `persist_across_layer_end`
   so layer-end clears preserve the arena-baked pointer. FwdStack for
   qwen3: 200MB (vs 5.8GB under the earlier per-layer-sectioning
   attempt). GPT-OSS: 124MB (vs 2.9GB). Step-0 grad norms match
   baseline on qwen3 / qwen3.5 / gpt-oss within FP noise.

   **Root cause (validated by 1080 `[op-io-alias] RAW ...` messages
   on the broken code, silent on the fix):**
   `replay_layer_forward`'s op loop used `idx <= end` but
   `annotate_layer_boundaries` defines `layer_end_indices[L]` as the
   START of layer L+1 (half-open). Under qwen3's deferred-residual
   pattern, layer L+1's first forward op is a `fused_residual_rmsnorm`
   computing L+1's LN1 from L's residual+mlp_down — so layer L's
   replay was also executing L+1's LN1 op, and under the plan's
   shared-arena routing that write lands in the same byte range as
   layer L's BlockLN1 and clobbered L's replayed activations before
   L's backward ops read them. Per-layer sectioning had masked this
   by giving every layer its own arena slice (num_layers× memory).
   Fix: half-open `idx < end` in all three op-range loops.

   **Also extended `SUROGATE_CHECK_OP_IO_ALIASING`** with a read-
   after-write provenance validator: tracks the `(slot, layer_idx)`
   of the last write to each arena byte-range; a later op reading
   the same range with a different `(slot, layer_idx)` is flagged
   with reader + writer identities and op indices. The within-op
   aliasing and RAW checks share one env flag. Slot-alias false
   positives (e.g., `blocks[0].x_flat` is a different tid but same
   slot as `blocks[0].ln1`) handled by resolving the ref's effective
   slot from its name — including for `Saved` refs backward uses.
   Backward-chain gradient slots (`BlockD*`) excluded by slot filter
   because they intentionally share buffers across backward stages.

   **Also added `SUROGATE_CHECK_REPLAY_SCOPE`**: replay_layer_forward(L)
   must not dispatch ops whose `layer_start != L`. Directly catches
   the loop-bound class of bug and would have short-circuited the
   multi-session investigation.

### Rule (going forward)

If a feature passes validation, make it default and remove the env
gate — features-behind-gates that validate clean are legacy-in-waiting.
Conversely, features that fail validation: revert, document the gap,
ship only the infra pieces that are independently correct.
10. ✅ Persistent arena routing for every weight owner —
    `DslParamStore::rebind_to_persistent_arena` for locally-allocated
    base weights (clamp via `rebindable_persistent_bytes` auto-skips
    QLoRA / weight-manager configs);
    `ModularLoRAWeightsManager::rebind_to_persistent_arena` reserves a
    bump slab for every LoRA leaf (master + work, attention / MLP /
    MoE-experts / MoE-grouped / router);
    `DslWeightManager::rebind_to_persistent_arena` covers device-
    resident master/work pairs plus `mPrefetchBuffers[slot]` (aliased
    entries across layers share a slot via pointer-dedup, offloaded
    pinned-CPU masters stay outside — arena is device-only); and
    `GenericWeightManager::consume_self_arena` gives the QLoRA provider
    its own dedicated arena (quantized storage + dequant buffers +
    full-precision), called post-`import_and_quantize` because the
    provider isn't wired until then. **All default-on** after the
    `SUROGATE_USE_PHASE_PERSISTENT` gate was removed — the auto-skip
    logic protects the one failure mode (gpt-oss-20B-class configs
    where 2× the quantized bytes exceed free GPU memory). Bit-identical
    on Qwen3 (227 base + 784 LoRA), Qwen3.5 (261 base + 384 LoRA),
    GPT-OSS (0 base + 576 LoRA + QLoRA-self-arena auto-skipped),
    fp32-master mixed-precision Qwen3 (0 base + 227 WM-master + 227
    WM-work + 784 LoRA), offload_master Qwen3 (3 non-block WM-master
    + 3 WM-work + 448 prefetch + 784 LoRA), and Qwen3 QLoRA-BnB/FP4
    (448/336 quant + 115 full-precision + 784 LoRA into
    ~810-830 MiB self-arenas).
11. Accumulator arena for grads — same shape as (10), with ZeRO-2
    complications. **Not started.** Env-gate now split:
    `SUROGATE_USE_PHASE_ACCUMULATOR=1` (shadow; no op consumes it yet).
12. ✅ Benchmark gate against `buffer-runtime-v4-benchmark.md` re-run
    (2026-04-21) with arena consumption default-on: step time neutral
    (−0.3% on qwen3, within noise on qwen3.5 / gpt-oss), peak memory
    unchanged versus pre-arena-consumption (verified via `git stash`
    + rerun on qwen3: both configurations 11,796 MiB). Raw numbers
    appended to the benchmark doc.
13. Phase 4 (cleanup / M5 kill list) — unblocked by (9) + (12).
    **Ready to start.** Each kill item is a multi-commit sweep; see
    M5 below for targets.

### M5 — Delete legacy machinery

With all arenas backing everything ops read:

- `TensorSlot::Block*/MoE*/SSM*` → remove enumerators + switch cases
- `SimplifiedLayerActivations` → delete
- `shared_tag()` → delete
- `share_ln1/...` booleans → delete
- `MatmulOp` alias → delete
- `builtin_slot_from_name` + string-match dispatch → delete
- `layer_start`/`layer_end` flags on ops → delete
- ✅ Ad-hoc bwd cross-layer cudaMalloc fallback pruned.
  `CompiledExecutor::allocate_bwd_cross_layer` now throws on
  exhaustion/unbound instead of falling back to per-step cudaMalloc;
  `BwdXLayerAlloc::arena_backed` field + `mPersistedBackwardTensors`
  vector + three cleanup loops removed. The 64 MiB arena (sized in
  graph_executor.cpp) comfortably fits our test configs — if a
  larger MoE config exhausts it, the throw message names the
  current size so the caller can bump. Validated bit-identical on
  Qwen3 / Qwen3.5 / GPT-OSS (the only one that exercises bwd cross-
  layer persist, via MoE aux-loss).

### M6 — Verify + commit the kill

Re-run benchmark gate on 3 models. Compare to [buffer-runtime-v4-benchmark.md](buffer-runtime-v4-benchmark.md) — we expect memory to stay within the same <2% envelope; throughput to match or improve (fewer string ops on hot path).

## Non-goals for Phase 4

- Tensor-id baking does not subsume shape: `(B, T)` still varies per-step, so shape stays on `TensorRef`. Offset is baked, shape is resolved at dispatch.
- We are not touching graph-capture-split logic. Capture stays as-is.
- Multi-node Ray path untouched.

## Open questions to resolve before M1

None substantive — baking is a structurally additive change. Kicking off.
