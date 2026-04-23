# Buffer Runtime v4 — Prototype Go/No-Go Memo (M6)

**Date:** 2026-04-20
**Branch:** `buffer-runtime-v4-prototype`
**Commits:** M1 `d76c18c`, M2 `8af5156`, M3 `54267e7`, M4 `eff1260`
**Scope:** M1–M4 completed shadow-mode. M5 (interpreter + bit-identical check) deferred pending this decision.

---

## Decision gate (from buffer-runtime-v4.md:363–365)

> If the prototype runs Llama 1 layer bit-identical and CUDA-graph-captures correctly, commit to Phase 0. If it surfaces a region that wasn't in the eight-region vocabulary or a subsystem that doesn't map to an instruction-stream primitive, fall back before committing weeks.

M5 was the "runs bit-identical" half. It was deferred because the structural half (M1–M4) is the cheap and high-signal half, and it either blocks the whole design or clears the path. It cleared the path.

---

## What M1–M4 proved

| Milestone | Deliverable | Validation signal |
|---|---|---|
| **M1** Phase tree | Root → FwdBlockSeq/BwdBlockSeq → FwdBlock/BwdBlock | 4-layer Qwen3: complete op coverage, non-overlapping, backward in reverse order |
| **M2** Regions | TensorKind + block_layer_idx → 8-region vocabulary | Fwd: FwdStack=107 Acc=35 Pers=52; Bwd: BwdStack=196 Acc=35 Pers=44; **zero Unknown** |
| **M3** Layout peaks | Per-frame max-clique-sum (optimal coloring bound) | FwdStack **416KB vs 1.38MB naive (3.3×)**, BwdStack 512KB vs 2.03MB naive (4.0×); matches analytical estimate (~13 live × 32KB) |
| **M4** Instruction stream | 4 primitives (PhaseEnter/Exit, SegmentDispatch, PruneByLastUse) | Structurally sound stream; matched enter/exit nesting; covers every op range |

The four assumptions from the design that most concerned the review all held for the dense-transformer case:
- Phase-tree shape fits the IR without Python-side changes.
- The 8-region vocabulary covers every tid that today's `classify_tensors` produces.
- Interval-graph coloring reaches the theoretical optimum that today's hand-tuned `shared_tag` allocator approximates.
- 4 instruction primitives (plus SaveCheckpoint for cross-boundary, added later) subsume the block-local work today's executor does implicitly.

No kill-criterion fired.

---

## What M1–M4 did NOT prove

1. **Bit-identical runtime.** No interpreter. Instruction stream is structurally sound but has not executed.
2. **CUDA-graph capture.** Zero capture test. Stream assumes one synthetic graph-captured segment per block; real split-attention subdivision ([graph_compiler.cpp:1580–1619](../csrc/src/runtime/dsl/graph_compiler.cpp#L1580-L1619)) not integrated.
3. **SaveForBwd detection.** M2 classifies cross-graph forward-activations-referenced-by-backward as Scratch → BwdStack (pessimistic and *structurally wrong* for backward correctness once offsets are baked). See risk #1 below.
4. **BwdCrossLayer, GatheredWeight, Recomputed** — declared regions, zero tids assigned for dense Qwen3. Exist only on paper.
5. **Hybrid architectures (Nemotron-H, Mamba)** — untested.
6. **MoE grouped-GEMM sub-phases** — untested (Qwen3 dense).
7. **Offload (ResidualManager) repurposing** — `offload_residual=False` in the smoke test. This is the design's highest-risk integration ([buffer-runtime-v4.md:123–132](buffer-runtime-v4.md)).

---

## Risks surfaced during prototyping

### 1. Cross-graph classification is structurally lossy (requires fix before M5)

In the backward compile, forward activations read by backward ops have no producer in the backward graph. `classify_tensors` final pass ([graph_compiler.cpp:3029–3033](../csrc/src/runtime/dsl/graph_compiler.cpp#L3029-L3033)) coerces them `Unknown → Scratch`. M2 then maps Scratch → BwdStack (pessimistic). These tensors are actually **SaveForBwd** — persistent slots live from fwd-block exit to bwd-block entry.

Today's runtime handles this via a save-list built at graph-compile time and persisted at `layer_end` ([compiled_ops_execute.cpp:636](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L636)). The prototype's M2 ignores that list; the numbers look right because nothing reads them yet.

**Blocker for M5:** any interpreter that bakes offsets from M2's region assignment would silently corrupt backward. Fix: integrate save-list detection into `derive_regions()` by inspecting last-use of forward tids against the backward graph's op indices, or emit a shared save-list from Python IR. Roughly +2–3 days of work before M5 can even start.

### 2. `annotate_layer_boundaries` is string-parsed

Layer boundaries are reconstructed from tensor-name patterns like `blocks[N].field` ([graph_compiler.cpp:1500–1566](../csrc/src/runtime/dsl/graph_compiler.cpp#L1500-L1566)). The phase tree the prototype builds is structurally elegant but sits atop the same fragile foundation the design claims to replace. Emitting phase structure from Python (`py_compiler.py:825 _inline_stacked_blocks`) is the long-term fix, scoped in Phase 1 of the main plan, deliberately not in the prototype.

### 3. Per-direction phase trees vs TrainStep-level design

`GraphCompiler::compile()` runs per-direction, producing two disjoint phase trees (forward-rooted and backward-rooted). The design shows `TrainStep → MicrobatchLoop → FwdBlockSeq → BwdBlockSeq` as one tree. Bridging is a trainer-layer concern. Not blocking; flagging that Phase 2's layout pass must decide whether FwdStack and BwdStack share bytes across the step boundary (design assumes distinct arenas — this is correct but worth writing down).

---

## Kill-criteria status (from buffer-runtime-v4.md:325–336)

| # | Criterion | M1–M4 signal | Verdict |
|---|---|---|---|
| 1 | Phase 3 peak memory > 5% | Coloring peak ≈ hand-tuned | ✅ no trigger |
| 2 | Capture throughput > 3% | — | ⚠️ needs M5 + capture test |
| 3 | Step throughput > 2% | — | ⚠️ needs M5 + live run |
| 4 | Cross-block sharing breaks nested stack | Blocks clean, no cross-block reads except SaveForBwd (risk #1) | ✅ no trigger |
| 5 | 1F1B doubles Phase 2 | — | ⚠️ needs Phase 0.2 audit |
| 6 | Offload coupling > 2 extra weeks | — | ⚠️ needs Phase 0.3 audit |
| 7 | Subsystem resists instruction-stream primitive | 4 primitives cover Llama block-local | ⚠️ save-list + recompute + cross-layer untested |

No criterion is a stop. Three need M5-or-audit work. Four criteria are pending but untriggered.

---

## Recommendation

**Commit to Phase 0 audits next, not M5.**

M5 would cost 4–7 days (plus the +2–3 for the SaveForBwd fix in risk #1) and would prove *one* thing: dense Llama 1 block runs bit-identical + captures. It would *not* prove anything about the MoE/SSM/hybrid/offload/cross-layer risks (#5–#7 in kill criteria). The design's own Phase 0 checklist ([buffer-runtime-v4.md:136–158](buffer-runtime-v4.md)) addresses those risks directly in 5–8 days with no code.

The better sequencing:

1. **Phase 0 audits (5–8 days, no code)** — produce one-pager memos per the design's 0.1–0.5 list. If any audit fires a kill criterion, stop before investing M5.
2. **If audits are clean: M5 with realistic scope (4–7 days + 2–3 for save-list)** — build the interpreter against a prototype branch, prove bit-identical on 1 Llama layer, check capture safety.
3. **Only then commit to the full Phase 1–4 timeline.**

The prototype through M4 is cheap to keep around: it's shadow-only (zero runtime change), gated on three env vars, ~550 LoC concentrated in one file. It provides live validation that the structural assumptions still hold as the codebase evolves. Leave it in place; don't revert.

## Explicit go/no-go from this memo

- **GO**: commit to Phase 0 audits. M1–M4 shadow code stays on the prototype branch.
- **NO-GO** for M5-in-this-prototype: cost/coverage ratio is poor; do it as part of Phase 1 with the full tooling (role unification, compat layer, CI harness) in place.
- **FALLBACK** to schema-per-block: not triggered; keep on the table if Phase 0.3 (offload) or a Phase-1 M5 run surfaces a real blocker.

---

## Appendix: prototype deltas

| Commit | LoC | Files | Env flag |
|---|---|---|---|
| d76c18c (M1) | +185 | graph_compiler.{h,cpp} + design doc | `SUROGATE_DEBUG_PHASE_TREE=1` |
| 8af5156 (M2) | +99 | graph_compiler.{h,cpp} | `SUROGATE_DEBUG_REGIONS=1` |
| 54267e7 (M3) | +154 | graph_compiler.{h,cpp} | `SUROGATE_DEBUG_LAYOUT=1` |
| eff1260 (M4) | +125 | graph_compiler.{h,cpp} | `SUROGATE_DEBUG_INSTR_STREAM=1` |
| **Total** | **~560 LoC** | **2 C++ files** | 4 env flags |

Zero modifications to any other file. Zero runtime behavior change when flags unset.
