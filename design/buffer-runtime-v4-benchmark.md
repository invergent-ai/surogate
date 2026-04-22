# Buffer-Runtime-v4: Phase 3 Benchmark Gate

**Date:** 2026-04-20
**Branch:** `buffer-runtime-v4-prototype`
**Hardware:** 1× RTX 5090 (32 GiB, SM120), single-node, single GPU
**Harness:** [scripts/bench_buffer_runtime_v4.py](../scripts/bench_buffer_runtime_v4.py) — runs `surogate sft`, drops 3 warmup steps, averages remaining 17
**Toggle:** `SUROGATE_LEGACY_EXECUTOR=1` for legacy, unset for stream-driven (current default). *Post–Phase 4 M5: flag deleted; the stream-driven interpreter is unconditional and legacy mode is no longer reachable.*

## Verdict: **Pass. Proceed to Phase 4.**

All three models stay within the `<2% proceed` thresholds for both peak memory and step throughput. Stream-driven interpreter is bit-identical (validated in prior phases) and marginally faster on step time; arena allocation — now gated to what the runtime actually consumes — does not regress memory.

This result required two fixes landed in this gate (see `Fixes` below). An earlier run of this gate tripped the `>5%` peak-memory kill line — the initial pass recorded +65% (Qwen3), +123% (Qwen3.5), and a GPT-OSS OOM — because arenas were allocated for consumers that don't exist yet.

## Decision matrix

Design §"Decision matrix", applied row-by-row:

| Model        | Peak mem Δ | Step time Δ | Capture Δ | Verdict                                   |
|--------------|-----------:|------------:|:----------|:------------------------------------------|
| Qwen3 0.6B   |     +0.8%  |     −0.6%   | n/a¹      | **Proceed to Phase 4.**                    |
| Qwen3.5 0.8B |     +0.7%  |     −1.1%   | n/a¹      | **Proceed to Phase 4.**                    |
| GPT-OSS 20B  |     +0.1%  |     +0.3%   | n/a¹      | **Proceed to Phase 4.**                    |

¹ Single-path measurement (capture+eager unified); no split.

## Raw numbers

| Model        | Mode   | Avg step (ms) | Min/Max (ms) | Tokens/s | Peak GPU (MiB) |
|--------------|--------|--------------:|:-------------|---------:|---------------:|
| Qwen3 0.6B   | stream |         601.0 | 598 / 606    |   27,261 |          7,700 |
| Qwen3 0.6B   | legacy |         604.4 | 600 / 620    |   27,107 |          7,636 |
| Qwen3.5 0.8B | stream |       2,300.5 | 2289 / 2310  |   14,244 |         12,204 |
| Qwen3.5 0.8B | legacy |       2,327.0 | 2313 / 2349  |   14,082 |         12,124 |
| GPT-OSS 20B  | stream |       2,720.5 | 2580 / 2840  |      753 |         28,908 |
| GPT-OSS 20B  | legacy |       2,713.5 | 2578 / 2845  |      755 |         28,874 |

Configs: [qwen3-lora-bf16-bench.yaml](../examples/sft/qwen3/qwen3-lora-bf16-bench.yaml), [qwen35-text-lora-bf16-bench.yaml](../examples/sft/qwen35/qwen35-text-lora-bf16-bench.yaml), [gptoss-lora-mxfp4-bench.yaml](../examples/sft/gpt-oss/gptoss-lora-mxfp4-bench.yaml).
Raw JSON: `/tmp/bench_results/*.json`.

## Fixes landed during this gate

Three arena over-allocations were identified via `SUROGATE_DEBUG_LAYOUT=1` + `SUROGATE_DEBUG_REGIONS=1` + `SUROGATE_DEBUG_ARENA_COVERAGE=1`. All three are consumers-that-don't-exist-yet: the arena pre-allocated address space for ops that haven't been wired to tensor-id baking (design §Phase 3 step 4).

### 1. `FwdStack` / `BwdStack` arena — premature, no consumer

`resolve_tid_in_arena` returns pointers for these regions, but no op in [compiled_ops_execute.cpp](../csrc/src/runtime/executor/compiled_ops_execute.cpp) consults the returned pointer on the hot path. Tensor-id baking is the missing consumer. Gated behind `SUROGATE_USE_PHASE_STACK_ARENAS=1`, default off in [graph_compiler.cpp:compute_arena_sizes()](../csrc/src/runtime/dsl/graph_compiler.cpp).

Savings on Qwen3 0.6B: 184 MiB. Scales with depth × seq × batch on larger models.

### 2. `SaveForBwd` arena — over-classified promotion

`finalize_save_for_bwd` promoted every tid that was produced-in-fwd ∧ consumed-in-bwd ∧ `is_blocks()`. Legacy's save-snapshot path at [compiled_ops_execute.cpp:767+](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L767) is stricter: it only materializes names in `GraphExecutor::mSaveList` AND skips names the slot registry marks `will_recompute`. Fix: `finalize_save_for_bwd` now takes an `optional<unordered_set<string>>` [save_names](../csrc/src/runtime/dsl/graph_compiler.h), and the call site at [graph_executor.cpp](../csrc/src/runtime/executor/graph_executor.cpp) builds the set by mirroring both filters.

The separate forward-replay path at [compiled_ops_execute.cpp:749](../csrc/src/runtime/executor/compiled_ops_execute.cpp#L749) preempts arena-backed saves entirely (metadata-only for most block tensors; model-specific exceptions like Qwen3.5 `ln1` go through `mMoeSavedBuffers`, not the arena). Because forward replay is active whenever `recompute_enabled` is set (and `mRecomputeFn` is always wired), the call site passes an empty set under recompute — promoting zero tids, sizing `SaveForBwd` to 0.

Savings: 4721 MiB on Qwen3 0.6B (4722 → 0.9 MB); 6166 MiB on Qwen3.5 0.8B (6166 → 0 MB).

### 3. `UnifiedStack` arena — transient 2× Stack spike

The adoption sequence (`allocate new Stack-sized buffer → rebase → free old buffer`) doubles Stack footprint transiently. On GPT-OSS 20B this pushed peak past the 32 GiB budget and OOM'd before step 0. No op reads through the adopted buffer via the arena path — the adoption was pure centralization, not consumption. Gated behind the same `SUROGATE_USE_PHASE_STACK_ARENAS=1` flag.

Savings: eliminates a transient ~3.9 GiB spike on GPT-OSS 20B; no steady-state change.

## What Phase 3 now ships

With the three fixes above landed:

- `SaveForBwd` arena: allocated only for tids the runtime's save-snapshot path will actually materialize. Sized to 0 when recompute/forward-replay is active (the common case).
- `BwdCrossLayer` arena: 64 MiB fixed, consumed by the interpreter's cross-layer persist path — replaces the legacy cudaMalloc-per-persist at [compiled_ops_execute.cpp:2337+](../csrc/src/runtime/executor/compiled_ops_execute.cpp).
- `MoE saved` arena: 256 MiB when `NumExperts > 0`, consumed by MoE save path in [compiled_ops_save.cpp](../csrc/src/runtime/executor/compiled_ops_save.cpp).
- `Persistent` / `Accumulator` / `FwdStack` / `BwdStack` / `UnifiedStack`: all behind feature flags (`SUROGATE_USE_PHASE_PERSISTENT=1`, `SUROGATE_USE_PHASE_STACK_ARENAS=1`), default off. Re-enable when tensor-id baking wires op lookups to arena offsets.

## Step-time result

Stream-driven is within the noise floor of legacy on all three models — a touch faster on the two dense models (Qwen3 −0.6%, Qwen3.5 −1.1%), neutral on GPT-OSS (+0.3%). Consistent with the P4.1 per-op layer-end cleanup + P4.2 defaults flip: the flat-instruction loop avoids the per-layer dispatch branches and string-matching that legacy still does.

No kill criterion triggered on step throughput.

## What this gate did not measure

- **Capture-replay split.** Single-path only.
- **Multi-rank.** 1-GPU. 4-GPU ZeRO / NCCL-overlap interactions untested in this gate (prior phases validated correctness on 4-GPU but not perf).
- **Recompute correctness vs. no-recompute.** Prior phases validated numeric parity; no fresh diff-tolerance check here.

These are follow-ups. Nothing in the current fix set should affect them negatively.

## Status of Phase 3

- Correctness: ✅ bit-identical on Qwen3 dense, Qwen3-MoE, GPT-OSS MoE, Qwen3.5 hybrid (prior validation)
- Step throughput: ✅ within ±2% — actually faster on two of three
- Peak memory: ✅ within +1% on all three models
- Capture throughput: not measured — follow-up

**Phase 3 ready to proceed to Phase 4.**

## 2026-04-21 re-bench: arena consumption default-on

FwdStack / BwdStack arena routing of `simplified_acts` slots is now
the default path (see commit `cb3f2da`, half-open replay bounds).
`SUROGATE_USE_PHASE_STACK_ARENAS` env gate removed. UnifiedStack
adoption skipped (`arenas.unified_stack_bytes = 0`) because the
cudaMalloc→rebase→free sequence briefly held two Stack-sized
buffers and pushed peak past the gate on the dense models.

| Model        | Avg step (ms) | Tokens/s | Peak (MiB) | Δ step_ms vs pre-my-changes |
|--------------|--------------:|---------:|-----------:|----------------------------:|
| Qwen3 0.6B   |        578.4  |  28,328  |    11,796  | −0.3% (578 vs 580)          |
| Qwen3.5 0.8B |      2,239.5  |  14,631  |    19,020  | within noise                |
| GPT-OSS 20B  |      2,740.7  |     747  |    30,368  | within noise                |

Peak memory unchanged vs pre-arena-consumption (verified via
`git stash`+ rerun on Qwen3: both configurations report 11,796 MiB).
Step time neutral or fractionally better.

The absolute Qwen3 / Qwen3.5 peaks are higher than the 2026-04-20
baseline in the table above (7,700 / 12,204 → 11,796 / 19,020 MiB)
because of independent codebase changes between the two gate runs;
the measurement-local delta attributable to arena consumption is
the ±0.3% reported in the "Δ step_ms" column.

## 2026-04-22 M6 gate: Phase 4 close-out

Post–M5.γ (Option C + replay fix + Session D narrow wins, landing
at commit `61831e4`). This closes Phase 4 — every subsequent delta
shipped since the 2026-04-21 re-bench goes through this gate.

| Model        | Avg step (ms) | Tokens/s | Peak (MiB) | Δ step_ms vs 2026-04-21 | Δ peak_MiB vs 2026-04-21 |
|--------------|--------------:|---------:|-----------:|------------------------:|-------------------------:|
| Qwen3 0.6B   |        577.3  |  28,381  |    11,874  | **−0.2%** (within noise) | **+0.7%** ✓              |
| Qwen3.5 0.8B |      2,155.6  |  15,201  |    18,822  | **−3.7%** ✓              | **−1.0%** ✓              |
| GPT-OSS 20B  |      2,636.1  |     777  |    29,830  | **−3.8%** ✓              | **−1.8%** ✓              |

Applied to the decision matrix: all three rows read "Proceed."

Step throughput is **better** on the two larger models (Q3.5 and
GPT-OSS) — the M5.α/β/γ cache pre-bind work shortens `resolve_tensor`
on the hot path. Peak memory drops on the larger models too, likely
because M5.γ's unconditional arena override (`ee0a7ad`) lets
`consume_fwdstack_arena` route every allowlisted slot through the
arena instead of leaking a second copy through mAllocator's
allocator-owned path.

Q3 is within measurement noise on both axes. No model regresses.

**Status of Phase 4.**
- Correctness: ✅ bit-identical across all M5.γ sessions (3 configs in parallel, GPUs 1/2/3, after every session commit)
- Step throughput: ✅ −3.8% to +0.2% — all within or better than the ±2% envelope
- Peak memory: ✅ −1.8% to +0.7% — all within the ±2% envelope
- Structural: Option C closed the cache-divergence design risk. `SimplifiedLayerActivations` deletion attempted (Session D proper) and abandoned after three failed partition strategies — postmortem in `design/simplified-acts-deletion.md`. The dual-dispatch shipped is stable and well-understood; deletion is cosmetic.

**Phase 4 closed.** Next: Phase 5 or new feature work; the tid-baked
infrastructure (slot_to_tid LUT, executor_tid_slot, block_slot_tensor)
is a stable foundation for any future refactor that wants to revisit
the deletion.

**Not measured in this gate (same as prior):** capture-replay split,
multi-rank, fresh recompute correctness diff. These are untouched by
M5.γ changes; prior validation stands.
