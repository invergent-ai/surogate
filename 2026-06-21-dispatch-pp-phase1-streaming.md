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
- ☑ Task 2 — Memory invariant: established by construction for single GPU; quantitative
  scaling test landed in Phase 2 (`tests/train/dispatch_pp/test_phase2_memory.py`, via
  `dispatch_pp_weight_residency` introspection — streaming holds `slot_count` blocks
  resident, `< NUM_LAYERS`)
- ☑ Task 3 — Phase-1 verdict recorded in the design spec §7 (streamed-vs-resident parity PASS)

**Memory invariant — single GPU (Task 2 resolution).** The bounded-footprint invariant holds by
construction and is confirmed active: `offload_master=true` places every block's master weight in
**pinned CPU memory** (`EAllocationType::PINNED`, dsl_weight_manager.cpp), and the executor gathers a
block's GPU work copy into a **2-slot double buffer** before it computes and releases it after
(`handle_layer_start`/`handle_layer_end` → `gather_block`/`release_block`), so at most ~2 blocks of
work-weights are GPU-resident at once regardless of layer count. The Task-1 parity test exercises this
exact path. A *quantitative* byte-level assertion ("resident weight bytes ≈ 2× largest block, flat as
layers grow") is marginal on the 4-layer tiny model and most meaningful at scale; the design (§6,
success criteria) frames the memory invariant as "per-GPU peak ≈ 2× largest stage and roughly flat as
GPU count scales" — a **multi-GPU, large-model** claim. It is therefore implemented and asserted in
Phase 2 (multi-GPU pool), where GPU-resident-weight-bytes introspection is plumbed and the
flat-as-N-grows scaling is the real, testable property.

---

## Task 1: Streamed-vs-resident correctness parity (single GPU)

**Files:**
- Test: `tests/train/dispatch_pp/test_phase1_streaming.py`

The gate: a model trained with weights streamed from pinned CPU must produce the same forward
hidden state and the same per-block weight-grad norms as the resident-weights run, on one GPU.
Reuses the Phase-0 debug entry points (`dispatch_pp_forward_hidden`,
`dispatch_pp_grad_norms_whole`) — they run through the executor's per-block gather/release,
so enabling `offload_master` exercises the streaming path with no new C++.

- [ ] **Step 1:** Build two trainers from the same weights — one resident (`offload_master=False`),
  one streamed (`offload_master=True`) — and assert forward hidden + grad norms match within tight
  tolerance.
- [ ] **Step 2:** Run `pytest tests/train/dispatch_pp/test_phase1_streaming.py -v` (GPU-gated, skips
  without CUDA / cached Qwen3).
- [ ] **Step 3:** Commit.

---

## Task 2: Memory invariant (single GPU) — established by construction

Resolved in the progress notes above: the single-GPU bounded-footprint invariant holds by
construction (pinned-CPU masters + 2-slot per-block gather/release, confirmed active by the Task-1
parity path). GPU-resident-weight-bytes introspection and the *quantitative* peak-memory assertion
are deferred to Phase 2, where the meaningful claim — peak ≈ 2× largest stage, flat as the GPU count
grows, on a large model — is testable. No Phase-1 code change.

---

## Task 3: Record Phase-1 verdict in the design spec

Update `2026-06-20-dispatch-pp-design.md` §7 (Phase 1 row / open questions) with the result:
single-GPU weight streaming reuses the existing `DslWeightManager` path unchanged, produces
bit-identical results, and bounds resident weight memory — gate PASS. Note that stage-range
(multi-block) gather and the multi-GPU pool are Phase 2.

---

## Out of scope (Phase 2+)

- **Stage-range gather** (`gather_block_range(i..j)` into one slot; ≥3 prefetch slots for pipeline
  overlap; pre-sized buffers for max stage width) — needed only when a stage holds >1 block.
- **Multi-GPU stateless pool** + round-robin/NUMA dispatch + the planner driving the plan (spec §4),
  including **GPU-resident-weight-bytes introspection** and the quantitative memory-invariant test
  (peak ≈ 2× largest stage, flat as GPU count grows).
- **Async 1-step-stale optimizer** + cross-call gradient-accumulation handoff (the Phase-0 backward
  caveat) (spec §4, §7 Phase 3).
