# Dispatch-PP — recompute:false (cyclic activation sectioning)

## STATUS 2 (per-stage reset attempt — recompute:false on the 27B NOT viable)

Made the saved-tensor cache pool-recycle (SavedTensorCache::reset_to_pool) and skip the
whole-graph prepare_saved_buffers_for_capture on the dispatch path (force_linear) — together
these let the 27B (gated-delta) **fit** with recompute:false (no OOM). But two findings killed
it:
1. **Correctness:** the per-stage cache reset recycles buffers that `mSaved` still references,
   corrupting the gated-delta backward — it regressed BOTH recompute modes on the 27B (loss
   2.7 -> 6+). A correct reset needs the backward to release its `mSaved` refs per stage first.
2. **Benefit is marginal:** on the transfer-bound 27B, recompute:false's compute saving is
   masked by weight streaming (~4.8s/step either way; the ~1.5x estimate was wrong — dispatch
   re-forwards once per stage regardless of recompute). The ~25% win only shows on the
   compute-bound 0.8B.

So the per-stage reset is left **unwired** (documented in `reset_saved_cache`); the 27B example
stays recompute:true. Kept from this attempt (behavior-preserving, verified): the make_persistent_tensor
-> SavedTensorCache::acquire migration (cleaner, pool-ready) and the dispatch prep skip.

Verified final: 27B recompute:true loss 2.7 (no regression); 0.8B recompute on/off both parity
(false ~20% faster). recompute:false on a DENSE dispatch model works; on gated-delta it OOMs
(no reset) and the reset that would fix it isn't gated-delta-safe yet.

## STATUS (implementation revealed broader scope)

recompute:false sizes **multiple, non-uniform** per-layer arenas whole-model — not just
FwdStack. Found so far:
1. **FwdStack** (`color_frames`, per-layer sectioning) — FIXED: cyclic `L % stage_blocks`.
2. **SaveForBwd** (`save_for_bwd_block_bases`, cumulative) — FIXED: cyclic block bases.
3. **moe_saved** (256 MB monotonic-bump arena for MoE expert states,
   `graph_executor.cpp:999` + `compiled_ops.cpp:585`) — NOT FIXED. Different mechanism
   (cross-step bump + cudaMalloc fallback); under recompute:false all layers' expert states
   are saved → overflow → per-state fallback cudaMalloc → GPU OOM. Needs per-stage reset +
   stage-sized arena. **The 27B is MoE, so this blocks it.**
4. Possibly more (audit needed).

**Verified:** 0.8B (dense) dispatch recompute:false — exact loss parity + ~30% faster step
(103 vs 137 ms). FwdStack + SaveForBwd cyclic fixes are correct (gated by env; non-dispatch
byte-identical; planner now emits aligned uniform stages, same stages as before for divisible
cases). The **27B (MoE) still OOMs on moe_saved** — fixes #1/#2 land it past the 19.3 GB and
16.4 GB arenas but not the 256 MB MoE bump.

**Conclusion:** bigger than a single pass estimated — it's a per-arena audit, each arena a
different mechanism. fixes #1/#2 are a correct, gated foundation; #3+ remain.

**Unblocked by the SavedTensorCache refactor (committed separately):** the moe_saved arena
(#3) is actually the general persistent saved-tensor cache (gated-delta states + rope +
SaveForBwd fallback), now owned by `dsl::SavedTensorCache`. The per-stage reset the dispatch
fix needs is now a one-liner — `mSavedCache.free_all()` at the end of each forward/backward
stage job — and it is arena-aware by construction (no scattered cudaFree to get wrong). It
also needs a sized arena (so the freed-and-reacquired states are arena-backed, not cudaMalloc
churn): `compute_arena_sizes` moe_saved_bytes for the dispatch path. That is the remaining
work for #3; #1/#2 + the refactor are the foundation it builds on.

---


**Goal:** let `recompute: false` work on the 27B under dispatch-PP. The backward currently
re-forwards every stage (recompute), ~doubling forward compute: measured step = 96% fwd+bwd,
and fwd+bwd ≈ `F + 2F` (fwd + re-forward + bwd). Removing the re-forward → ~`2F` → **~1.5×
step speedup** (~4800 → ~3200 ms on the 27B). This is the only large training-speed lever left.

## Root cause (mapped)

`recompute:false` ⇒ `fwd_per_layer_sections = !recompute_enabled = true`
([graph_compiler.cpp:6730](csrc/src/runtime/dsl/graph_compiler.cpp#L6730)). In `color_frames`
([:3362](csrc/src/runtime/dsl/graph_compiler.cpp#L3362)), per-layer sectioning gives each layer
its own `[L*stride, (L+1)*stride)` slice so every saved activation has a distinct address (the
whole-model backward reads them all). ⇒ FwdStack arena = `num_layers × stride` = 19.3 GB at
seq 1024 → `cudaMalloc(fwd_stack)` OOMs. Addresses resolve as `fwd_stack_ptr + meta.offset`
([compiled_ops_save.cpp:1062](csrc/src/runtime/executor/compiled_ops_save.cpp#L1062)).

But dispatch runs the backward **one stage at a time** (each GPU re-forwards a stage, saves,
backwards, resets — stages on a GPU are sequential; concurrent stages are on *different* GPUs
with *separate* arenas). So only one stage's activations are ever live on a GPU → we need
`stage_blocks × stride`, not `num_layers × stride`.

## Fix: cyclic sectioning (no base-pointer shift)

Color dispatch layers cyclically: `base = (L % stage_blocks) * stride` instead of `L * stride`.
Arena = `stage_blocks × stride` (e.g. 4 × ~300 MB = 1.2 GB). A stage's layers map to distinct
sections `[0, stage_size)`; layers `stage_blocks` apart share a section (safe — never live
together). No runtime base shift (that would break the `[fwd_lo, fwd_hi]` bounds checks); the
offsets are already in `[0, stage_blocks*stride)`.

**Invariant this depends on:** stages must be **aligned to `stage_blocks`** (`lo = k*stage_blocks`)
so `L % stage_blocks == L - lo` within a stage (else the modulo wraps and two layers collide).
⇒ the planner must emit aligned uniform stages, not the current even split.

## Changes

1. **Planner** (`trainer.py:_dispatch_pp_plan`): aligned uniform stages of `stage_blocks`
   (`[0..sb-1], [sb..2sb-1], …`, last = remainder), `nst = ceil(n_layers/sb)`, still ≥ gpus.
2. **Coloring** (`graph_compiler.cpp`): thread `dispatch_sections` (= stage_blocks, from env
   `SUROGATE_DISPATCH_STAGE_BLOCKS`) into `compute_layout`→`color_frames`; when `>0` and
   `section_per_layer`, use `(L % dispatch_sections)` for the section base. Gated by the env,
   which only the dispatch trainer sets → non-dispatch paths are byte-for-byte unchanged.
3. Keep `recompute:false` allowed for dispatch (already un-forced in sft_config); flip the
   example to `recompute: false`.

## Test matrix (must all pass before commit)

- **0.8B dispatch, recompute:false** — loss parity vs recompute:true; faster step.
- **0.8B dispatch, recompute:true** — unchanged (cyclic only triggers when sections-per-layer,
  i.e. recompute:false; verify no regression).
- **27B dispatch, recompute:false** — fits (no fwd_stack OOM), loss sane, ~1.5× step.
- **Non-dispatch recompute:false** (small model, env unset) — byte-identical coloring (no
  regression) — the key safety check, since this touches shared core code.

## Risk

Core graph-compiler coloring is shared by every training mode. Mitigations: env-gated (dispatch
only), the non-dispatch test proves no regression, and the cyclic map degenerates to the current
`L*stride` when `dispatch_sections == 0`.
