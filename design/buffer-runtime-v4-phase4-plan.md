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
