# M5.d — Flag-gated Interpreter Scoping

**Status:** not implemented; scoping only.
**Blocker:** this is the 4–7 day executor-rewiring work the memo (M6) flagged as the wrong next step. Written as a checklist so a future session can pick it up with full context.

---

## What's done (M5.a–c)

| Milestone | Commit | Proves |
|---|---|---|
| M5.a — cross-graph SaveForBwd fixup | `e0d5fc9` | 59 tids correctly promoted on 4-layer Qwen3 |
| M5.b — offset baking | (pending) | Every live tid gets `TensorMeta::offset` from greedy coloring |
| M5.c — instruction stream validator | (pending) | Nesting balanced, ops 100% covered, prune ranges disjoint |

All three remain shadow-mode. Nothing consumes offsets or the instruction stream at runtime.

---

## What M5.d would require

A flag-gated mode where `CompiledExecutor::execute_forward` / `execute_backward` ignores today's `layer_start_indices`/`layer_end_indices` machinery and instead drives execution from `CompiledGraph::instruction_stream` + `TensorMeta::offset`. Tensor addresses come from arena-base + offset instead of `TensorAllocator::allocate()` pointers.

Concretely:

### 1. Arena allocator (replaces `TensorAllocator` for these regions)

One flat device buffer per region-family:

| Arena | Size | Behavior |
|---|---|---|
| PersistentArena | `persistent_bytes` | bump-only; never freed until step end |
| AccumulatorArena | `accumulator_bytes` | bump-only |
| FwdStackArena | `fwd_stack_peak` | stack; `PhaseEnter FwdBlock[i]` sets base, `PhaseExit` restores |
| BwdStackArena | `bwd_stack_peak` | stack; same pattern |
| SaveForBwdArena | `save_for_bwd_bytes` | per-block slots; bumped per `SaveForBwd[i]` |

Each `TensorMeta::offset` + arena base gives the final pointer. CUDA allocator APIs are untouched — just these 5 buffers allocated once at training start.

### 2. `PhaseInterpreter` (new class, ~400 LoC)

```cpp
class PhaseInterpreter {
  void run(const CompiledGraph& graph, CompiledExecutor& exec);
  // Walks graph.instruction_stream:
  //   PhaseEnter FwdBlock[i] -> fwd_stack_base += 0 (new frame at same base; old contents below stay)
  //   PhaseExit FwdBlock[i]  -> (no-op; next frame reuses)
  //   SegmentDispatch        -> for op in [start, end): exec.dispatch_op(op)
  //   PruneByLastUse         -> release refcounts (no free needed; arenas manage lifetime)
};
```

The tricky parts:

- **Stack frame bases**: today's `mStack.checkpoint()` / `mStack.restore()` lives on a separate allocator. New design: FwdBlock[i] reuses FwdStackArena[0..frame_peak]. Inner phases (if any) would add nested frames (TODO: does Llama need nesting beyond FwdBlock? — no, it doesn't; MoE does, later).
- **Tensor resolution on the hot path**: today `exec.mTensors[tid]` is a `Tensor` with a pointer set by `TensorAllocator`. Interpreter needs to override this per-region:
  ```cpp
  Tensor resolve(int tid) {
    const auto& meta = graph.tensor_meta[tid];
    switch (meta.region) {
      case Persistent:   return {persistent_base + meta.offset, ...};
      case Accumulator:  return {accumulator_base + meta.offset, ...};
      case FwdStack:     return {fwd_stack_base + meta.offset, ...};
      case BwdStack:     return {bwd_stack_base + meta.offset, ...};
      case SaveForBwd:   return {save_for_bwd_base[block] + meta.offset, ...};
    }
  }
  ```
  This means rewriting how `mTensors[tid]` is populated. Every op's dispatch function reads `mTensors[tid]` — they shouldn't care which allocator filled it.

### 3. State integration landmines

Today's `CompiledExecutor` holds a lot of state that the interpreter must not double-drive. Checklist of "who owns what":

- `mSlotRegistry` — stays; interpreter consults it for roles.
- `mNamedTensors` — stays for now (legacy lookup path); interpreter populates new tids into `mTensors` directly.
- `mTemps` — today used for op-internal scratch. Need to decide: does the interpreter's arena model cover these, or keep `mTemps` as a side allocator?
- `mStack` — conflicts with FwdStackArena. Resolution: in interpreter mode, skip stack ops; runtime only hits them in today's path.
- `mPersistedBackwardTensors` — the ad-hoc cross-layer `cudaMalloc` path. Not needed for Llama (no MoE aux-loss); skip for prototype.
- `ResidualManager` — compiled with `offload_residual=False` for the prototype; don't touch.
- Weight cache / LoRA hooks — untouched; interpreter just passes through.
- Forward-replay hooks for Qwen3.5 LN1 exception — skip for Llama.

### 4. Flag and fallback

```cpp
// RuntimeOptions
bool use_phase_interpreter = false;  // default off

// graph_executor.cpp
if (mOptions.use_phase_interpreter) {
    mPhaseInterpreter.run(*mCompiledForward, *mCompiledExecutor);
} else {
    mCompiledExecutor->execute_forward(...);  // today's path
}
```

Flag `SUROGATE_USE_PHASE_INTERPRETER=1` can gate from env.

### 5. Bit-identical validation

Run the same step twice — once with flag off, once with flag on. Dump activations per op index (use existing `SUROGATE_DEBUG_DUMP_TENSORS` machinery). Diff bytes. Tolerance: exact equality under BF16 (same kernels, same inputs, same order → same outputs).

Pass criterion: zero diffs for 1 layer Llama forward + backward.

### 6. CUDA-graph capture test

Enable `use_cuda_graphs=True`. Interpreter should be capture-safe because addresses are stable (one allocation per arena, never freed mid-step). If not:

- Split-attention subdivision: M4's single-SegmentDispatch-per-block doesn't subdivide around FlashAttention. For capture-unsafe ops, need to break into eager/graph segments per today's [graph_compiler.cpp:1580–1619](../csrc/src/runtime/dsl/graph_compiler.cpp#L1580-L1619) logic. Scope: +1–2 days.
- Dynamic shapes (doc masking): same as today, not new.

---

## Estimate

| Step | Time | Notes |
|---|---|---|
| Arena allocator | 0.5 d | 5 cudaMallocs + offset math |
| PhaseInterpreter skeleton | 1–2 d | walk stream, dispatch ops, stack-base mgmt |
| Tensor resolution override | 1–2 d | the hard part — every op's `mTensors[tid]` path |
| State integration (temps, stack, named) | 1–2 d | debug-heavy; subtle races |
| Flag-gate + parallel path | 0.5 d | switch in execute_forward/backward |
| Bit-identical check, 1-layer Llama | 1 d | dump + diff + triage |
| Capture-safety test | 1 d | + segment subdivision if needed |
| **Total** | **6–9 days** | |

## What it would prove

1. The 4 instruction primitives (plus SaveForBwd's implicit slot-based allocation) are **executionally complete** for block-local dense transformer ops.
2. Stable compile-time offsets plug cleanly into CUDA graph capture (the design's primary win).

## What it would NOT prove

- MoE, SSM, hybrid architectures (MoE adds grouped-GEMM sub-phases; SSM has non-block-scoped lifetimes)
- Cross-layer persistence (BwdCrossLayer; MoE aux-loss)
- ZeRO-3 (GatheredWeight region)
- ResidualManager offload integration (design's biggest Phase 3 risk per Phase 0.3)
- Multi-GPU determinism (rank-identical layout hash)

Each is its own separate work item in Phase 3 of the main plan.

---

## Recommendation (restates M6)

Do **not** execute M5.d as a continuation of this prototype. The evidence from M1–M5.a–c is already sufficient to commit to Phase 0 audits. M5.d is correctly scoped as "early work item in Phase 1," gated on Phase 0 audits not firing a kill criterion.

If a future session DOES want to tackle M5.d (e.g., to de-risk a specific concern), the checklist above is the starting point. Budget 6–9 days. Prototype-branch remains usable.
