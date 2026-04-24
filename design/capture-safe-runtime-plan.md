# Capture-safe runtime under sample packing (Option C)

## Motivation

Profiling on Gemma4-E2B LoRA training (bs=2, gas=4, seq_len=2048) shows:
- **Step time**: 3046 ms/step, SOL 27%, 5.4 k TPS
- **GPU kernel time**: 0.83 s/step (27% of wall time)
- **Kernel launches**: 54 k `cuda(Kernel|Launch)Kernel` calls per step + only ~93 `cudaGraphLaunch`
- **Root cause**: **~2.2 s/step CPU dispatch overhead**, not kernel quality

Unsloth at comparable configuration reaches 13.3 k TPS (67% SOL, ~1.23 s/step). The 2.5× gap
is dispatch-bound, not compute-bound.

### Why full-step CUDA graphs aren't captured today

`graph_executor.cpp:1250–1254`:
```cpp
const bool doc_masking_active = (mCuSeqlensGpu != nullptr);
const bool needs_split = doc_masking_active || has_capture_unsafe_ops || has_tiled_mlp;
const bool use_graphs = mGraphsEnabled && !in_capture && !needs_split;
```

`sample_packing: true` in the training config → `mCuSeqlensGpu` becomes non-null → we drop
to split-attention mode (capture only sub-segments per layer, leaving most ops eager).
cu_seqlens content varies per step, so it can't simply be baked into a captured graph.

### Scope of this plan

Make the runtime capture-safe under sample packing + document masking, so full-step graph
capture lights up for any dense model trained with `sample_packing: true`
(Llama, Mistral, Qwen2.5, Gemma4, …).

**Not in scope** (separate efforts, though the general-infrastructure fixes here make them
cheaper later):
- Qwen3.5 JIT kernel loading during step (`ChunkGatedDeltaRule`, `Qwen3_5Decay`)
- MoE routing / grouped-GEMM per-step host bookkeeping
- EP (expert-parallel) host-side split/reorder

## Success criteria

| Metric | Baseline | Target |
|---|---|---|
| Gemma4-E2B TPS (bs=2, gas=4) | 5.4 k | ≥ 9 k |
| Loss vs non-captured baseline (50 steps) | — | abs diff < 1e-3 |
| `cudaGraphLaunch`/step | ~93 | ≤ 3 (one per fwd, bwd, update) |
| `cudaLaunchKernel`/step | ~54 k | < 200 |
| Regression on MoE / Qwen3.5 / GPT-OSS | none | none |

## Phases

### Phase 0 — Discovery (1 d)

Build a capture-safety instrumentation wrapper. Goal: enumerate **every** call site that
becomes unsafe once `doc_masking_active` no longer forces split mode. No code fixes this
phase — only cataloging.

**Deliverables**:
- Wrapper around `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemcpyAsync`,
  `cudaStreamSynchronize`, `cudaMallocAsync`, `cudaFreeAsync` that logs the call when
  `cudaStreamIsCapturing` reports capturing. Log: caller file:line, C++ demangled call stack
  (via libdw), bytes, stream.
- Run force-capture (env-var bypass of `doc_masking_active`) on Gemma4-E2B. Capture full
  list of violations.
- Write a small table of `(symbol, frequency, fix strategy)` to drive Phases 1–4.

**Exit gate**: Complete, de-duped list of capture-unsafe call sites.

---

### Phase 1 — Arena preallocation sweep (2 d)

Hoist every `cudaMalloc`-during-capture site from Phase 0 outside capture. Known items:
- `CompiledExecutor::ensure_replay_persist_arena()` — 256 MiB lazy alloc
  ([compiled_ops.cpp:485–497](../csrc/src/runtime/executor/compiled_ops.cpp))
- Already handled: FP8/FP4 weight caches (primed pre-capture in `graph_executor.cpp:1276`)
- Already handled: saved-buffer preallocation (in `prepare_saved_buffers_for_capture`)

Phase 0 output adds to this list. Likely candidates: MoE selective-expert buffers (not in
scope for this plan's exit but fix while we're here if trivial), doc-masking
GPU cu_seqlens buffer (currently reallocated inside `set_doc_masking` at
`graph_executor.cpp:2230–2234` if size changes).

**Deliverables**:
- Every pre-capture `cudaMalloc` wired from `GraphExecutor::execute_forward`/`execute_backward`
  when `capturing` is true, before `cudaStreamBeginCapture`.
- Public helper methods (`prepare_*_for_capture`) on `CompiledExecutor` for each.

**Exit gate**: Force-capture no longer produces `cudaErrorStreamCaptureUnsupported`.

---

### Phase 2 — Recompute / replay-path capture safety (2–3 d)

Under force-capture we hit:
```
replay_layer_forward layer=34 op=31 (type=narrow): CompiledExecutor: tensor not found: scale_8
```

`replay_layer_forward` is the recompute path (gradient checkpointing). Under capture it
fails to resolve `scale_8` (likely a LoRA scaling slice tensor that lives in `FwdStack`
and isn't re-bound on replay).

**Investigation**:
- What is `scale_8`? Grep DSL graph; inspect `mTensors` / `mNamedTensors` at the failure
  point.
- Is the root cause: (a) tensor binding is evicted between forward and replay, (b) the
  replay path resolves via a different code path that doesn't honour the same bindings, or
  (c) a save-for-backward tensor is released before replay? Phase 0 traces should narrow
  this.

**Deliverables**:
- Replay path resolves all tensors the original forward resolves, when running under
  capture. Specifically: LoRA intermediates, forward-phase scaling tensors, any
  `saved_ref`-bound activations.
- Unit-test via `test-jit-kernel`-style suite: capture a replay segment, replay it, check
  identical kernel args across replays.

**Exit gate**: Force-capture completes one full forward + backward on a single micro-step
without tensor-resolution errors.

---

### Phase 3 — Updatable cu_seqlens memcpy node (2–3 d)

Today, `GraphExecutor::set_doc_masking` runs `cudaMemcpyAsync(mCuSeqlensGpu, host, …)` each
step before the captured graph would launch. Under full-step capture, this memcpy becomes
part of the captured sequence with the host pointer baked in — on replay, it reads stale
host memory.

**New design**:
- cu_seqlens host buffer: pinned memory, stable lifetime across steps (reallocated only
  on size change).
- Pre-capture: `cudaMemcpyAsync` is captured into the graph as a memcpy node.
- Per-step before `cudaGraphLaunch`: update the captured memcpy node's `src` pointer via
  `cudaGraphExecMemcpyNodeSetParams`. If the cu_seqlens *count* changes (rare — only when
  sequence packing yields a different number of documents), re-instantiate the graph
  (acceptable because count changes are infrequent; cache a small pool if measurements show
  frequent change).

**Instrumentation to decide pool size** (add in Phase 0 or early Phase 3): log cu_seqlens
count across a full epoch. If < 10 distinct values, a simple LRU pool suffices.

**Deliverables**:
- Pinned host cu_seqlens buffer in `GraphExecutor`.
- Captured memcpy node recorded at capture time; its `cudaGraphNode_t` retained on the
  executor.
- Per-step `cudaGraphExecMemcpyNodeSetParams` call in the replay path.
- Graph-exec pool keyed by cu_seqlens count.

**Exit gate**: Captured graph replays correctly with varying cu_seqlens content per step;
loss matches non-captured baseline over 10 steps.

---

### Phase 4 — num_docs / max_seqlen via device-pointer indirection (2–3 d)

Flash-varlen takes `num_docs` and `max_seqlen` as scalar kernel args today. Captured into
a graph, these become literals — wrong on replay when per-step packing produces different
values.

**Spike first** (2 h, *before committing the rest of Phase 4*): read
`csrc/src/runtime/attention/backend_flash_varlen.cpp` + the flash-varlen kernel signature.
If the kernel already accepts these as pointers → trivial. If not, decide between:
- **Option P4a**: thin adapter kernel that reads device-resident metadata and does an
  on-device dispatch into the right flash-varlen variant. Cost: ~1 d kernel work.
- **Option P4b**: carry a small `int32_t[2]` device buffer alongside cu_seqlens; pass its
  pointer to the kernel; kernel loads `num_docs`, `max_seqlen` at launch time.
  Cost: small flash-varlen kernel patch.

**Deliverables**:
- Device buffer containing `{num_docs, max_seqlen, total_q}` populated via the same pinned-
  host + captured-memcpy pattern as cu_seqlens.
- Flash-varlen entry point that reads these from device pointer, not scalar.
- Backward attention path updated identically.

**Exit gate**: Training run where per-step max_seqlen varies (confirmed via log) produces
correct loss.

---

### Phase 5 — Gate removal + correctness testing (2 d)

Remove `doc_masking_active` from `needs_split` in
`graph_executor.cpp::execute_forward`/`execute_backward`. Remove the env-var escape hatch
from earlier.

**Test matrix**:
| Config | bs | gas | seq_len | Check |
|---|---|---|---|---|
| Gemma4-E2B | 2 | 4 | 2048 | loss diff < 1e-3 (50 steps) |
| Gemma4-E2B | 1 | 8 | 2048 | loss diff < 1e-3 |
| Gemma4-E2B | 4 | 2 | 2048 | loss diff < 1e-3 |
| Gemma4-E2B | 2 | 4 | 4096 | loss diff < 1e-3 |

Use `surogate debug diff` as the gold oracle (compares against HF reference, already wired
for Gemma4).

**Exit gate**: All rows in matrix pass.

---

### Phase 6 — Generalization + regression (1–2 d)

Validate the fix across model families:
| Model | sample_packing | Expect |
|---|---|---|
| Llama-3 small | true | TPS uplift + loss correct |
| Mistral small | true | TPS uplift + loss correct |
| Qwen2.5 1.5B | true | TPS uplift + loss correct |
| Any MoE config | either | **no change** — still in split mode |
| Qwen3.5 | either | **no change** — still in split mode (JIT kernels) |
| Any dense, packing=false | false | **no change** — already captured |

**Exit gate**: Matrix passes; no regression on configs that were already fast.

---

### Phase 7 — Docs + cleanup (0.5 d)

- Top-of-file comment in `graph_executor.cpp` documenting the capture-safety contract:
  what a captured graph may not do, which metadata must go through device pointers, how
  to add new ops safely.
- Add a debug-build assertion around `cudaMalloc`/`cudaMemcpy` that fires if called while
  stream is capturing (guardrail against regressions).
- Remove `SUROGATE_FORCE_FULL_GRAPH_CAPTURE` env-var if it was left in tree.

**Exit gate**: clean diff, all tests green.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Flash-varlen can't accept device-pointer metadata | Med | +3–5 d | Phase 4 spike first; fall back to adapter kernel if needed |
| cu_seqlens count changes frequently → graph-exec pool thrash | Low–Med | Med | Phase 0 instrumentation measures distinct-count frequency; decide pool size empirically |
| Unknown capture-unsafe ops surface mid-implementation | Med | Med | Phase 0 explicitly catalogs; treat Phase 0 output as the authoritative list |
| LoRA backward has additional replay invariants besides `scale_8` | Med | Low–Med | Phase 2 sweeps the full replay path, not just the reported symptom |
| Correctness silently diverges (loss drifts over long runs) | Low | High | Phase 5 tests 50 steps, not just 5; automated loss-diff gate |

## Pivot points

- **After Phase 0**: if the violations list is >10 sites with complex fixes, reconsider
  scope — might need a bigger rewrite than one sprint.
- **After Phase 3**: if cu_seqlens update via `cudaGraphExecMemcpyNodeSetParams` needs
  per-step re-instantiation and the cost eats most of the wins → stop at "best effort"
  (typically 50–70% of theoretical win) rather than press on.
- **After Phase 5**: if loss diff > 1e-3 is persistent → stop, root-cause before shipping.

## Verification harness (build once, use throughout)

- **Perf rig**: `gemma4-e2b-lora-bf16.yaml` + `max_steps: 5` + `nsys profile --delay 20 --duration 10`.
  Script computes step time + `cudaGraphLaunch` count + `cudaLaunchKernel` count.
- **Correctness rig**: `surogate debug diff` against HF Gemma4 reference over 50 steps.
- **Matrix rig**: one short run per model in Phase 6 matrix.

Automate all three so each phase ends with `make capture-plan-gate` (or similar) that runs
the rigs and prints pass/fail.

## Timeline estimate

- **Best case**: 10–12 d
- **Expected**: 12–16 d
- **Worst case** (Phase 4 adapter kernel + correctness debug): 18–22 d

## Open questions to resolve in Phase 0

- What exactly is `scale_8` in the `replay_layer_forward` failure? (likely a LoRA scaling
  tensor in a forward activation slot)
- Does flash-varlen in this codebase accept `cu_seqlens` by device pointer already? (read
  the kernel signature in `csrc/src/runtime/attention/backend_flash_varlen.cpp`)
- How many distinct cu_seqlens counts per epoch on a typical dataset?
- Are there `cudaMallocAsync` sites that get captured but bind stale addresses on replay?
  (comment at `compiled_ops.cpp:596` hints yes)

## Phase 0 — findings log

### V1: `cudaMalloc` in `ensure_replay_persist_arena` (Phase 1, fixable)

- Site: [compiled_ops.cpp:485–497](../csrc/src/runtime/executor/compiled_ops.cpp)
- Trigger: lazy 256 MiB `cudaMalloc` for replay-persist arena, called from
  [compiled_ops_execute.cpp:661](../csrc/src/runtime/executor/compiled_ops_execute.cpp)
  inside `execute_forward`, which itself runs inside capture when full-step graph
  capture is on.
- Fix applied (as Phase-0 workaround, also needed for Phase 1): added public
  `prepare_replay_persist_arena_for_capture()` and call it from
  `GraphExecutor::execute_forward` / `execute_backward` in the `capturing` pre-capture
  branch. Confirmed: no more `cudaErrorStreamCaptureUnsupported` from this site.

### V2: `scale_8` not found during `replay_layer_forward` (Phase 2 proper)

- Symptom: `RuntimeError: replay_layer_forward layer=34 op=31 (type=narrow): CompiledExecutor: tensor not found: scale_8`
- Root cause: `scale_8` is the underlying name (`g.scale(...)` auto-named) of
  `per_layer_inputs` — produced **once** by the model-level PLI phase in
  [models/gemma4.py:435](../surogate/dsl/models/gemma4.py#L435), consumed by **every
  layer** via a compiler-synthesized `narrow` op
  ([py_compiler.py:1079](../surogate/dsl/py_compiler.py#L1079)) that slices a
  per-layer chunk out of the model-level `[B, T, n_layers, PLI_D]` tensor.
- `replay_layer_forward` binds only layer-scope tensors + a narrow set of
  model-scope inputs (token_ids, position_ids, x0, embed outputs, zeros, rope freqs,
  mSaved, saved_named_tensors). `scale_8`/`per_layer_inputs` is in none of these
  buckets — it's a model-scope activation that isn't saved for backward.
- Non-capture mode "works" for this because `mTensors`/`mNamedTensors` isn't cleared
  between forward and backward, so `scale_8` survives into replay via the
  `saved_named_tensors` snapshot path
  ([compiled_ops_execute.cpp:267](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L267)).
- Under force-capture, the flow is: forward captures → stream capture ends → forward
  stack arena rolls back → backward capture begins → `replay_layer_forward` starts
  from a fresh `mTensors` → `scale_8` isn't in `saved_named_tensors` for reasons
  worth investigating (likely stack rollback invalidates the data pointer even if
  the map entry survives).
- **Fix approach (Phase 2)**: pick one of
  (a) Add `per_layer_inputs` (and any other model-scope cross-layer activations)
      to the save list so `mSaved` holds them persistently, and let
      `replay_layer_forward`'s mSaved fallback bind them;
  (b) Extend `replay_layer_forward`'s external-input resolution chain with a
      "model-scope activation" lookup (new slot category) that survives stack rollback;
  (c) Tag such tensors as "persistent forward activations" at compile time so the
      layout allocator places them outside the FwdStack arena that gets rolled back.
- Open concerns: are there *other* model-scope tensors besides `per_layer_inputs`
  that would trip the same issue? Likely candidates: embedding output, final_norm
  intermediates. Phase 2 investigation should enumerate.

### V3 (blocked by memory): forward graph OOM with `recompute: false`

- Workaround attempted to bypass V2 by disabling recompute. Result: forward stack
  arena = 7.4 GiB (bs=1) / 14.8 GiB (bs=2), OOM at capture time.
- **Implication**: Phase 2 must be completed before Phase-0 discovery can continue
  past V2 — there's no cheap workaround. This is expected for E2B at seq_len=2048
  on a 32 GB GPU.
- **Silver lining**: full-step graph capture with recompute off would need even
  more memory budget than capture-on — so recompute must work under capture.

### Flash-varlen signature spike (Phase 4 preview)

- Flash-varlen takes `max_seqlen` as a scalar in
  [flash_attn_varlen.cpp:28](../csrc/src/runtime/attention/flash_attn_varlen.cpp#L28),
  fed into `params.seqlen_q`, `params.seqlen_q_rounded` etc.
  ([lines 65–70](../csrc/src/runtime/attention/flash_attn_varlen.cpp#L65-L70)).
- These go into **kernel params** (baked into kernel-arg buffer on launch), not
  launch-grid dims — confirmed by reading `set_common_params`. Launch-grid
  computation happens in flash-attn internals via template dispatch on HeadDim and
  IsCausal; the actual grid shape depends on `params.seqlen_q_rounded`,
  `params.h`, `params.b` (num_docs).
- Under capture, those kernel-arg values are baked. On replay with a smaller
  `max_seqlen`/`num_docs`, the kernel does extra work past the real sequence end
  (over-reading cu_seqlens zeros or stepping past `total_q`). Correctness
  depends on whether the kernel bounds checks against cu_seqlens (it should, but
  needs verification).
- **Phase 4 path of least resistance**: capture with `max_seqlen` pinned to the
  config's `sequence_len` (2048) and `num_docs` pinned to the worst-case packing.
  Kernel does extra iterations for under-full packings but produces correct
  results because cu_seqlens bounds the real work. Modest wasted cycles in
  exchange for full-step graph capture.
- **Phase 4 aggressive path**: device-pointer `(max_seqlen, num_docs, total_q)`
  with per-step memcpy + kernel patch to read from device. Requires editing
  flash-attn internal kernels. Estimated +2–3 d beyond the baseline path.

## Phase 0 → Phase 1 handoff

Arena preallocation (V1 fix) ships as Phase 1. V2 blocks further discovery —
proceed directly to Phase 2 implementation. Phase 4 decision (pad-to-seq_len vs
device-pointer) to be made after Phase 3 is in-flight when we can measure
overhead of the pad-to-seq_len approach.

## Phase 2 attempt 1 — findings (reverted, not shipped)

Attempted the "auto-detect + save via existing persistent-buffer path"
approach. The detection pass works correctly (identified `scale_8` /
`per_layer_inputs` as the only cross-layer global in Gemma4-E2B). The
runtime plumbing to persist it into `mSaved` via `force_persist_name` got
past V2 under force-capture (scale_8 resolves correctly during replay, the
persistent buffer pointer is valid). **But even in normal non-force-capture
mode, the combined change broke forward correctness — step 0 loss jumped
from 3.63 to 18.21 and diverged to NaN by step 3.**

**What was changed (all reverted)**:
- `TensorMeta::cross_layer_global` field + detection in
  `promote_cross_layer_fwd_reads`: find non-blocks tensors, produced by an
  op, consumed by ops in ≥2 layers.
- `CompiledGraph::cross_layer_global_names()` helper.
- `GraphExecutor::compile_graphs` unioning those names into `mSaveList`
  and the `save_name_arg` passed to `finalize_save_for_bwd`.
- `save_tensors` + `prepare_saved_buffers_for_capture`: treat
  `cross_layer_global` tensors as `force_persist_name` → preallocate
  buffer, memcpy each step.
- `save_tensors` skip-if-already-saved exemption for cross-layer globals
  (so the persistent buffer refreshes each step).

**What broke (not fully root-caused before reverting)**:
- With all gates properly narrowed (no region change, `promote()` helper
  still rejects non-block tensors, no other surgery), forward-at-step-0
  already showed loss 18 vs. clean-baseline 3.63. The correctness damage
  starts at the very first forward — not a backward-path issue.
- Hypothesis (unverified): adding a tensor name to `mSaveList` interacts
  with an earlier bookkeeping step that affects forward execution — e.g.,
  a layout or allocator decision that changes when a name becomes part of
  the save set. The `fwd_per_layer_sections`/`retain_through_forward`
  mechanism, or some arena-coloring pass driven by the compiled-forward
  save list, seems the likely suspect.

**Lessons for Phase 2 attempt 2**:
1. The existing `force_persist_name` / `mMoeSavedBuffers` path is not
   safe to feed with arbitrary new tensor names — it couples to compiler
   assumptions about which tensors are in the save list.
2. A safer design for cross-layer globals may need a **separate mechanism**
   instead of reusing `mSaved` + `mSaveList`:
   - Dedicated "model-scope persistent activation" slot category
     (compiler + allocator aware, outside the FwdStack arena from the
     start so no stack rollback concern). Similar to option (b) from the
     original Phase 2 design — heavier surgery but decouples from the
     save-list plumbing.
   - Or: bind the cross-layer-global directly during `replay_layer_forward`
     from a known-good source (the compiler can emit a side table
     `cross_layer_global_tids` that the replay path consults), without
     routing through `mSaved`.
3. Before ANY new code, build a diff-regression harness that runs a short
   training with `surogate debug diff` and flags >1e-3 loss divergence on
   the first few steps. Would have caught this within minutes rather than
   after the whole pipeline was touched.

**Recommended next attempt**: option (b) from Phase 2 design — new region
category. Larger surgery, but isolates cross-layer globals from the
existing save-list machinery that has baked-in assumptions we didn't fully
reverse-engineer.

## Phase 2 attempt 2 — shipped

Took a minimal-surgery approach based on what the runtime already had and
what attempt 1 taught us.

### The actual root cause (attempt 1 got close; attempt 2 nailed it)

Under force-capture, `replay_layer_forward` threw `tensor not found: scale_8`
during the backward pass of the last block. Three independent runtime facts
combine into the bug:

1. `scale_8` (Gemma4 `per_layer_inputs`) is a **model-scope tensor consumed
   by every layer** via compiler-synthesized `narrow_pli_L` ops
   ([py_compiler.py:1079](../surogate/dsl/py_compiler.py#L1079)).
2. The compiler leaves many forward op output `TensorRef::name`s empty
   (they resolve by slot/tid alone), so `store_tensor` never adds an
   `mNamedTensors["scale_8"]` entry. `saved_named_tensors` — the pre-replay
   snapshot — is missing it.
3. `prune_stack_tensors` at layer-end evicts any `mTensors[tid]` whose
   `Data` satisfies `Stack.owns(p) && !Stack.is_live(p)`. Arena-placed
   tensors trigger this (the unified_stack rebase put all arenas inside
   the Stack's backing memory, but `is_live` only tracks mAlloc records).
   So by layer 1's end, `scale_8`'s `mTensors[22].Data` is already nulled.

`replay_layer_forward`'s fallback chain was name-based only — it never
looked up by tid — so the tensor became unreachable.

### The fix — three narrow changes

1. **Compile-time flag** ([graph_compiler.cpp](../csrc/src/runtime/dsl/graph_compiler.cpp)):
   Pass 2 added to `promote_cross_layer_fwd_reads` detects tensors with
   `block_layer_idx < 0` that are produced by an op in the graph AND
   consumed by ops in ≥2 distinct layers. Flagged as `cross_layer_global`
   on `TensorMeta`. Gemma4-E2B reports 1 such tensor: `scale_8`.

2. **Runtime mask extension** ([compiled_ops_execute.cpp:727](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L727)):
   `mSaveMask[tid]` is additionally set for every `cross_layer_global`
   tid. `prune_stack_tensors` already skips save-mask entries, so this
   alone keeps the tensor alive across layer boundaries during forward.
   No `mSaveList` / `save_tensors` coupling (which was the source of
   attempt-1's stale-data regression).

3. **Tid-based fallback + snapshot carry-over**
   ([compiled_ops_execute.cpp](../csrc/src/runtime/executor/compiled_ops_execute.cpp)):
   - `execute_backward` now carries the forward snapshot across for
     tensors with `cross_layer_global=true && Stack.owns(data)`, not just
     FwdStack/SaveForBwd arena ranges. (Their data lives in stack temps,
     not the persistent arenas — but the pointer is stable because the
     Stack isn't rolled back between fwd and bwd.)
   - `replay_layer_forward` gains a tid-indexed fallback at the bottom of
     its external-input resolver: if `saved_tensors[inp.tensor_id].Data`
     is non-null, bind it. Handles both named and unnamed cross-layer
     globals transparently.

### Validation

- `loss_diff_harness.sh` (3-step run + baseline-band check) passes in
  normal mode: step 0/1/2 losses 3.63/3.76/3.57 as before.
- bs=1 gas=8 regression check: 3.57/3.57 — normal operation.
- Force-capture harness (`SUROGATE_FORCE_FULL_GRAPH_CAPTURE=1`): **V2
  (scale_8 not found) is cleared.** Next violation surfaces — V3 —
  documented below.

### V3 surfaces next

Under force-capture with attempt-2 applied, training now crashes with
`cudaErrorIllegalAddress` in `rmsnorm_backward_kernel10` at
[csrc/src/kernels/rmsnorm.cu:974](../csrc/src/kernels/rmsnorm.cu#L974).
This is a separate class of failure from V2:

- V2 was about a **model-scope tensor** (`scale_8`) that should survive
  but was being evicted → resolver couldn't find it.
- V3 is about a **block-scope tensor** (the RMSNorm input — likely
  `blocks[L].ln1` or equivalent) whose pointer has been invalidated by
  the time backward reads it. The resolver probably *finds* the tensor,
  but the pointer refers to memory that's been reused/rolled back.

Likely root cause: the `mForwardTensorsSnapshot` restore in
`execute_backward` restricts to FwdStack and SaveForBwd arenas. Block
activations in neither region (temp-alloc'd from Stack, not arena-placed)
don't get restored. They need the same fix that `cross_layer_global`
tensors got — but scoped to block activations.

## Phase 2 attempt 3 — V3 investigation (did not ship)

Pinpointed V3: `pli_proj_rn_flat` (PLI RMSNorm's `x` input, a model-scope
tensor in the Gemma4 PLI phase) has `Data=(nil)` when the backward
`rmsnorm_backward_kernel10` runs. Verified by instrumenting both
`dispatch_rmsnorm_backward` and `dispatch_fused_residual_rmsnorm_backward`
— the last successful call went to block 0's RMSNorm with valid pointers,
then `pli_proj_rn_flat` showed `x.Data=(nil) rstd.Data=(nil)`.

Tried: broaden the `cross_layer_global` classification to flag every
model-scope forward-produced tensor consumed anywhere (drops V2's
"≥2 consumer layers" constraint → 55 flagged tids, including the PLI
narrow outputs `pli_narrow_layer0..34`). Runtime harness green in
normal mode, but force-capture hit **a new error**:
`CompiledExecutor: cannot save tensor pli_narrow_layer21` at save-list
iteration in `save_tensors`. Traced to `mTensors[tid].Data == nullptr`
at save time — the pruner's `mSaveMask` gate was correctly true, yet
`mTensors[1516].Data` was never populated. Diag loop showed the narrow
op never wrote to that tid — even though the op declares it as output
and `dispatch_narrow` calls `store_tensor(op.outputs[0], out)`.

Two possibilities remain (not yet resolved):
- The narrow op's output `TensorRef::tensor_id` is `-1` in the compiled
  form (`store_tensor` skips `mTensors` when tid < 0, only updates
  `mNamedTensors`). The compile-time flag sees these names but runtime
  doesn't route their storage through mTensors[tid].
- Normal mode succeeds at this same save because a different resolver
  path fires. Need to compare save-list iteration behaviour between
  modes.

Reverted attempt 3 without shipping. Committed attempt 2 (`ef8595c`)
stands as the clean progress point — V2 cleared, harness green.

### V3 deep-dive (2026-04-24 session)

Traced V3 across four distinct clear-sites for `mTensors[tid]` that
collectively evict model-scope forward tensors before the refresh
`save_tensors` call reads them:

1. **Forward pruner** (`prune_stack_tensors` at fwd layer-end,
   [compiled_ops_execute.cpp:745](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L745)):
   clears entries whose Data satisfies `Stack.owns && !is_live` — arena-
   placed tensors inside the unified_stack buffer trip this. Already
   gated by `mSaveMask`, which attempt 2 extends via `cross_layer_global`.
2. **execute_backward mTensors.assign** (`compiled_ops_execute.cpp:1810`):
   unconditionally clears `mTensors` at backward entry. Only FwdStack +
   SaveForBwd arena tids are restored from `mForwardTensorsSnapshot`. CLG
   tids also need restore (attempt 2 added this).
3. **Backward pruner** (`prune_stack_tensors` in execute_backward,
   [compiled_ops_execute.cpp:1941](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L1941)):
   similar `Stack.owns && !is_live` check but *no* mSaveMask gate.
   Clears restored CLG pointers.
4. **prune_by_last_use** (compiled_ops_execute.cpp:2381): clears tids
   whose last-use index equals the current bwd op. No save/CLG gate.

Attempt 3 fix: widen all four sites to honor `cross_layer_global`, and
extend the flag set via `graph_executor.cpp` post-finalize loop over
`mSaveList` — because under recompute, `save_names` passed into
`finalize_save_for_bwd` is empty (by design), so the compile-time pass
doesn't see the real save list. The graph_executor loop sees it.

This works through V2 and V3 for `pli_proj_rn_flat`, but surfaces a
**V4**: segfault in `dispatch_narrow` during `replay_layer_forward`
(called during backward recompute). Cause: preserving `mTensors[tid]`
for pli_narrow_layer* keeps a stale pointer that points into stack
memory that's been reused by subsequent temp_allocs. The narrow op then
`cudaMemcpy2DAsync`'s from stale memory and crashes.

The real insight: our preservation only blocks pruner CLEARS, not the
underlying staleness. Stack memory gets reused even when we keep
pointers alive. To ship V3 correctly, model-scope forward tensors read
by backward either need:

- **(a)** a persistent arena that isn't overwritten between forward
  temp_allocs and backward reads (similar to SaveForBwd but indexed by
  something other than block_layer_idx), **or**
- **(b)** an explicit memcpy into a persistent buffer at the point where
  forward last writes them (same as `save_tensors` force_persist_name
  does for block saves — but that path is already buggy for narrow-op
  outputs where the compiler strips `TensorRef::name`), **or**
- **(c)** re-compute them via replay_layer_forward but with a tid-aware
  resolver that correctly traces back to their model-scope producers.

Each of these is a 3–5 day implementation with significant cross-cutting
changes. Reverted attempt 3 working tree; committed ef8595c stands.

### Session bottom line

Phase 2 attempt 3 made the fix architecturally clearer but exposed that
V3+ requires a structural change to how model-scope forward tensors
survive fwd→bwd under capture. Whack-a-mole pruner-skipping creates
use-after-free risks when stack memory is reused.

For the next session, recommended approach is **option (a)**: add a new
region `ModelScopePersistent` with its own arena. Forward ops producing
model-scope tensors consumed by backward get tids in this region, with
stable offsets. No pruning, no temp_alloc overlap, no snapshot-carry
gymnastics. The allocator + compiler changes are non-trivial but the
mechanics are clean.

### Harness-script hygiene

Committed in the session: `scripts/loss_diff_harness.sh` now uses a
dynamic `/tmp/harness_metrics_$$.jsonl` path for `SUROGATE_METRICS_PATH`
to avoid PermissionError when prior runs locked the default path.

## Phase 2 attempt 4 — ModelScopePersistent region + PersistentActivation arena binding

### Surgery (this session)

**New `RegionKind::ModelScopePersistent`** (fully wired — arena field on
`PhaseArenas`, peak on `CompiledGraph`, allocate/release, layout bucket +
bump, coverage validator, debug dump, `resolve_tid_in_arena` case). Kept
even though the final fix doesn't use a separate arena — the plumbing
is clean infrastructure for future refinement if intermediate model-
scope tensors ever need isolation from I/O buffers.

**Actual fix** (pivoted mid-session):
1. `finalize_save_for_bwd` flags model-scope fwd tensors (`block_layer_idx<0`,
   `produced_in_fwd`, `consumed_in_bwd`) that are already in
   `PersistentActivation` as `cross_layer_global`.
2. `GraphExecutor::compile_graphs` post-finalize extends this to tids in
   `mSaveList` (covers tensors the DSL saves even when backward doesn't
   directly consume them — e.g., `pli_narrow_layer*` compiler-synthesized
   outputs).
3. `populate_fwd_stack_bindings` gains a new branch: for
   `PersistentActivation` tids with `cross_layer_global`, bind
   `mTensors[tid].Data = persistent_activation_ptr + meta.offset`.

This routes the intermediate model-scope tensors through the
`PersistentActivation` arena that was already allocated but whose runtime
binding only covered named buffers (`x0`/`xF`/`ln_final`/…) via
`rebind_non_block_to_persistent_arena`. Now intermediate tids like
`pli_proj_rn_flat`, `scale_8`, and `pli_narrow_layer*` get stable arena
pointers that don't get overwritten by later `temp_allocs`.

### Validation

- Normal mode harness: green (loss 3.6317 / 3.7640 / 3.5727 unchanged).
- Force-capture (`SUROGATE_FORCE_FULL_GRAPH_CAPTURE=1`): V3 (pli_proj_rn_flat
  rmsnorm_backward IMA) is **cleared**. Training no longer crashes in
  the PLI-RMSNorm backward path.

### V4 surfaces

Next failure under force-capture: `dispatch_mul_backward: unsupported
broadcast pattern. a.name=blocks[34].pli_gate_act a.shape=[4096,256]
b.name=blocks[34].pli_flat b.shape=[2,2048,1536]`.

`pli_flat` has shape `[2,2048,1536]` (the block-input residual shape,
`[B,T,C]`) when it should be `[B*T, PLI_D] = [4096,256]`. Under normal
mode this works — under force-capture, the block-scoped `pli_flat` tid
gets the wrong Sizes in its `Tensor` struct by the time `mul_backward`
reads them.

Hypothesis: `pli_flat` is a compiler-alias view of `per_layer_input`.
Its TID binding under capture is reading from a different producer's
TensorRef shape (maybe the upstream `h_flat` or residual view) due to
tid aliasing. Unrelated to the PersistentActivation binding we added —
it's a block-scope tid and doesn't go through our new branch.

This is V4, not V3. Deferred — the harness guardrail catches it within
2 minutes of any change.

### Scope-level observation after four attempts

Each V reveals a distinct forward-to-backward capture invariant:
- V1: arena malloc during capture (fixed P0/P1)
- V2: cross-layer globals missing from replay resolver (fixed P2 attempt 2)
- V3: model-scope intermediate forward activations not arena-bound (fixed P2 attempt 4)
- V4: block-scope view-aliased tensor shape corruption under capture (open)

Expect V5+ to follow similar patterns. The harness + incremental-fix
strategy has proven much more reliable than batch-changing and crashing
into walls. Each attempt ships ~200 LoC and clears one class of
invariant.
