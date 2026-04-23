# No-recompute NaN Investigation

**Status:** FIXED as of commit `531cda3` (2026-04-22). See `design/buffer-runtime-v4.md` M5.ζ for shipped details.

**Date:** 2026-04-22
**Branch:** `buffer-runtime-v4-prototype`
**Config:** `examples/sft/qwen3/qwen3-lora-bf16-bench-norecompute.yaml`

## Symptom

`recompute: false` (i.e. save all forward activations instead of replaying) produces `loss=nan, norm=0.0000` from step 0 onward. Pre-existing bug — present on the oldest commit reachable on this branch (`83b49d3`, before this phase's allocator cleanups). The old manifestation was a CUDA illegal memory access at `flash_attn_varlen.cpp:308`; somewhere between `83b49d3` and `8ffc9c5` the error was silenced and became a silent NaN.

## Root causes identified (partial fix attempted, reverted)

Two independent issues, both rooted in the FwdStack arena's coloring design assuming replay:

### 1. `FwdStack` arena is shared across layers

[`compute_layout`](../csrc/src/runtime/dsl/graph_compiler.cpp) calls `color_frames(fwd_frame, …, /*section_per_layer=*/false)`. All layers' FwdStack tids share the same byte offsets. Under recompute this is fine — replay regenerates each layer's activations just-in-time into the shared bytes. Under no-recompute, layer N's forward overwrites layer N-1's activations before backward can read them. Fixable by passing `section_per_layer=!recompute_enabled` through `compute_layout`; costs `num_layers×peak_per_frame` memory (Q3 96 MiB → 2.6 GiB).

### 2. `save_name_set` filters out `will_recompute` entries

[`graph_executor.cpp`](../csrc/src/runtime/executor/graph_executor.cpp) builds `save_name_set` from `mSaveList` and drops entries where `registry.will_recompute(name, lora_only)==true`. This is correct under recompute (replay regenerates those tids) but wrong under no-recompute — those very tids need explicit `SaveForBwd` promotion because there's no replay path. Fixable by passing `std::nullopt` (no filter) when `!recompute_active`.

### 3 (blocker). Runtime resolve bypasses per-tid region

Even after fixes 1+2, `loss` drops from NaN to ~16 but doesn't converge (baseline Q3 step 0 is `loss=2.02`). `op-io-alias` output shows within-layer coloring still overlapping `blocks[L].x_flat` with `blocks[L].qkv_flat` on the same bytes — i.e. some tids remain in FwdStack region and share frames despite the "promote all" policy.

The deeper issue: `resolve_tensor` for block-scope tensors goes through `block_activation_ptr(rs, layer_idx, slot) → simplified_acts[L][slot].Data`. This is a **per-slot** dispatch; `simplified_acts[L][slot].Data` is populated by `consume_fwdstack_arena` using `slot_to_tid(L, slot)` — which returns a single canonical tid per (layer, slot). SSA aliases that share a slot but have distinct tids (e.g., `blocks[L].x_flat` with its own tid, aliasing `blocks[L].ln1`) are NOT handled by the slot dispatch; they resolve to whichever pointer `simplified_acts[L][slot].Data` holds. If the canonical tid was promoted to `SaveForBwd` but the alias tid's own `meta.region` is still `FwdStack`, the runtime still reads/writes via the slot dispatch — which points to the FwdStack arena (or vice versa).

Making this work end-to-end requires extending the runtime resolve path to consult `meta.region` per-tid rather than per-slot — essentially the same architectural expansion the `SimplifiedLayerActivations` deletion design memo (`design/simplified-acts-deletion.md`) is blocked on.

## Current state

- Branch restored to pre-investigation state (commit `391781a`).
- No partial fix landed — each isolated fix changes peak memory (FwdStack +2.6 GiB + SaveForBwd +4.6 GiB on Q3) and still fails correctness without the runtime-resolve overhaul.
- Pre-existing NaN remains for `recompute: false`. Workaround: keep `recompute: true` (the default everywhere except this specific bench config).

## Recommendation

Tie the fix to Session D proper (delete `SimplifiedLayerActivations`) or the M5.δ views/gradient leftovers work. Both independently need per-tid region-aware dispatch; fixing no-recompute is a natural side-effect of that.

## Files touched during investigation (reverted)

- `csrc/src/runtime/dsl/graph_compiler.{h,cpp}` — thread `fwd_per_layer_sections` flag
- `csrc/src/runtime/executor/graph_executor.cpp` — pass `nullopt` under no-recompute
