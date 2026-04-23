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
