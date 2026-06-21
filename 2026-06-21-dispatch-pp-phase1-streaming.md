# Dispatch-PP — Phase 1: Single-GPU Stage-Range Weight Streaming

> Follows `2026-06-20-dispatch-pp-planner-and-phase0.md` (Phase-0 gate: PASS) and
> `2026-06-20-dispatch-pp-design.md` §7 (Phase 1). Implement task-by-task with TDD.

**Goal:** Run transformer blocks with their work-weights **streamed per block from pinned CPU
memory** (not resident in VRAM) on a single GPU — gathered before a block computes, freed after —
so per-GPU peak memory is bounded by the largest stage's footprint rather than the whole model,
while producing results identical to a resident run. This is the design's "can we stream weights"
gate that Phase 0 deliberately isolated out (Phase 0 proved sub-range execution with *resident*
weights; Phase 1 adds streaming).

**Key finding (drives this plan):** the per-block weight-streaming stack already exists and is
reusable unchanged for single GPU. `RuntimeOptions.offload_master` (or `cpu_training`) makes
`DslWeightManager::needs_block_gather()` return true; the compiled executor's
`handle_layer_start`/`handle_layer_end` then call `gather_block` / `release_block` (double-buffered,
pinned-CPU master → GPU work copy) around every block — and the dispatch-PP debug executor already
calls those handlers. An empirical check (Qwen3-0.6B/4-layer, single RTX 5090) shows streamed
(`offload_master=true`) forward hidden states and per-block grad norms are **bit-identical** to a
resident run through the dispatch-PP path (fwd max-abs-diff 0.0, grad max-rel-diff 0.0).

So Phase 1 is primarily about **locking that behavior behind a regression test** and exposing a
**memory-introspection** hook so the bounded-footprint invariant is asserted, not just assumed.
Multi-block *stage-range* gather (one slot per stage of >1 block) and the multi-GPU pool are Phase 2.

## Implementation Progress

Updated before each commit. Status: ☐ not started · ◐ in progress · ☑ done.

- ☑ Task 1 — Streamed-vs-resident correctness parity (forward hidden + grad norms), single GPU
- ◐ Task 2 — Expose per-run peak GPU-allocator bytes to Python (memory introspection)
- ☐ Task 3 — Memory-invariant assertion: streamed peak ≪ resident-weights footprint
- ☐ Task 4 — Record Phase-1 verdict in the design spec §7

---

## Task 1: Streamed-vs-resident correctness parity (single GPU)

**Files:**
- Test: `tests/train/dispatch_pp/test_phase1_streaming.py`

The gate: a model trained with weights streamed from pinned CPU must produce the same forward
hidden state and the same per-block weight-grad norms as the resident-weights run, on one GPU.
Reuses the Phase-0 debug entry points (`dispatch_pp_debug_forward_hidden`,
`dispatch_pp_debug_grad_norms_whole`) — they run through the executor's per-block gather/release,
so enabling `offload_master` exercises the streaming path with no new C++.

- [ ] **Step 1:** Build two trainers from the same weights — one resident (`offload_master=False`),
  one streamed (`offload_master=True`) — and assert forward hidden + grad norms match within tight
  tolerance.
- [ ] **Step 2:** Run `pytest tests/train/dispatch_pp/test_phase1_streaming.py -v` (GPU-gated, skips
  without CUDA / cached Qwen3).
- [ ] **Step 3:** Commit.

---

## Task 2: Expose per-run peak GPU-allocator bytes to Python

**Files:**
- Modify: `csrc/src/binding/py_train.h`, `py_train.cpp`, `binding.cpp`

`DeviceMemoryStack::max_utilization()` / `get_allocation_stats()` exist (csrc/src/utilities/stack.h)
but the device weight-buffer footprint also lives in `DslWeightManager`/`DslParamStore`. Add a
debug-only `MultiGPUPyTrainer::gpu_weight_bytes_resident(int gpu)` (sum of bytes of GPU-resident
work buffers, i.e. excludes pinned-CPU masters) and `stack_peak_bytes(int gpu)`. These let a test
quantify "how much weight memory is on the GPU at once" under streaming vs resident.

- [ ] TDD as above; expose via nanobind; document in `_surogate.pyi`.

---

## Task 3: Memory-invariant assertion

**Files:**
- Test: `tests/train/dispatch_pp/test_phase1_streaming.py`

Assert the streamed run keeps **far less** weight memory resident than the resident run (≈ the
double-buffered largest-block footprint, independent of layer count), using the Task-2 hook. On the
4-layer tiny model the bound is `~2 blocks` vs `4 blocks + embed + head`; the assertion is
`streamed_resident_bytes < resident_resident_bytes` with margin, and scales with layer count.

---

## Task 4: Record Phase-1 verdict in the design spec

Update `2026-06-20-dispatch-pp-design.md` §7 (Phase 1 row / open questions) with the result:
single-GPU weight streaming reuses the existing `DslWeightManager` path unchanged, produces
bit-identical results, and bounds resident weight memory — gate PASS. Note that stage-range
(multi-block) gather and the multi-GPU pool are Phase 2.

---

## Out of scope (Phase 2+)

- **Stage-range gather** (`gather_block_range(i..j)` into one slot; ≥3 prefetch slots for pipeline
  overlap; pre-sized buffers for max stage width) — needed only when a stage holds >1 block.
- **Multi-GPU stateless pool** + round-robin/NUMA dispatch + the planner driving the plan (spec §4).
- **Async 1-step-stale optimizer** + cross-call gradient-accumulation handoff (the Phase-0 backward
  caveat) (spec §4, §7 Phase 3).
