# Refactor Completion Plan

This document tracks the remaining implementation work needed to move from the completed tracked phases in `REFACTOR.md` to the fully realized runtime architecture. It intentionally excludes the open roadmap-question bucket from `REFACTOR.md`.

## Status

Tracked phases 0-5 are complete for the scoped, guarded implementation. The remaining work is mostly replacing guarded/scaffold/diagnostic paths with authoritative runtime paths, then validating the end state.

## Track 1 - Finish Native FP8 MoE WGrad

- [ ] Replace the current per-expert FP8 matmul loop in `moe_grouped_gemm_weight_grad_fp8` with a true grouped FP8 TN implementation.
- [ ] Use cuBLASLt grouped matmul where shape/alignment support is available.
- [ ] Keep `SUROGATE_FP8_MOE_WGRAD=1` as the rollout gate until parity and perf are proven.
- [ ] Preserve BF16 fallback for unsupported shapes, missing cache/state, and cuBLASLt rejection.
- [ ] Add architecture-aware implementation notes for Ada, Hopper, and Blackwell based on the inspected CUTLASS grouped/FP8 examples.

Acceptance:

- [ ] Per-expert FP8 wgrad max-abs delta is within `5e-2` versus BF16 reference.
- [ ] 5-step FP8-forward/FP8-dgrad/FP8-wgrad loss is within 5% of FP8-forward/BF16-wgrad baseline.
- [ ] 2-GPU DP smoke passes with FP8 wgrad enabled.
- [ ] nsys confirms the intended FP8 wgrad path and no redundant expert weight requantization in backward.

## Track 2 - Promote Hook Dispatch From Opt-In To Runtime Path

- [ ] Move CPU-stream prefetch from imperative gather/wait paths into registered `before_consume` schema hooks.
- [ ] Move streaming grad offload after all-reduce into registered `after_all_reduce` hooks.
- [ ] Move EP all-to-all observation and follow-up actions into registered `after_all_to_all` hooks.
- [ ] Move LoRA after-produce actions for dense, router, shared expert, and grouped expert projections into registered schema hooks.
- [ ] Keep the current imperative paths as fallback until hook parity is green under CI.
- [ ] Flip default dispatch after parity, then remove fallback-only code that no longer has a caller.

Acceptance:

- [ ] `SUROGATE_ENABLE_SCHEMA_HOOK_DISPATCH=1` and default dispatch produce identical 5-step losses and gradient norms on the real-model acceptance queue.
- [ ] Hook registry reports full target coverage for CPU streaming, DP all-reduce, EP all-to-all, reduce-scatter, and LoRA after-produce paths.
- [ ] No runtime path depends on ad hoc hook-slot names when a structural `(schema_id, slot)` target exists.

## Track 3 - Make Schema-Driven Allocation Authoritative

- [ ] Replace enum-driven `BufferPlan` allocation decisions with schema-derived per-layer/per-slot allocation plans.
- [ ] Use schema shape, lifetime, residency, distribution, and save-for-backward metadata as the primary allocator inputs.
- [ ] Keep the current TensorSlot/legacy enum plan as a parity comparator during rollout.
- [ ] Remove or narrow legacy safety slacks after schema allocation parity is validated.
- [ ] Preserve capture safety for CUDA graph paths and arena fallback diagnostics.

Acceptance:

- [ ] `SUROGATE_BLOCK_SCHEMA_PLAN_ASSERT=1` is green across the real-model queue.
- [ ] Schema allocation bytes are authoritative in runtime debug and regression artifacts.
- [ ] Peak CUDA memory does not regress; Gemma4 and MoE acceptance rows show the expected allocation savings.
- [ ] Save-for-backward and frame-local arena summaries match schema lifetime declarations.

## Track 4 - Remove Descriptor Capability Fallbacks - COMPLETE

- [x] Audited quantized recipe ops for explicit `OpCapabilities`, `MatmulCapabilities`, `MoECapabilities`, storage compatibility, communication profile, and grouped semantics.
- [x] Converted recipe capability fallback logging into descriptor-deny diagnostics for supported quantized recipes.
- [x] Removed legacy allow-paths now that dense, Mamba out projection, and grouped MoE ops are annotated.
- [x] Kept explicit unsupported statuses for combinations that are intentionally unavailable.

Acceptance:

- [x] `SUROGATE_RECIPE_CAPABILITY_LOG=1` reports no legacy allow-paths for the real-model acceptance queue.
- [x] North-star coverage rows derive FP8/FP4 support from descriptor results, not fallback assumptions.
- [x] Unsupported FP4/MoE combinations fail or skip with explicit descriptor reasons.

## Track 5 - Activate Fusion Registry Rewrites

- [ ] Add a non-mutating graph rewrite preview that lists exact fusion candidates and replacement descriptors.
- [ ] Implement the first descriptor-driven rewrite for a low-risk dense fusion family.
- [ ] Extend rewrite support to MoE and Mamba candidates only after dense parity is green.
- [ ] Add rollback/disable flags per fusion family for safe rollout.

Acceptance:

- [ ] Fusion preview is deterministic and appears in regression artifacts.
- [ ] At least one dense fusion rewrite is enabled by default with numerical parity and no perf regression.
- [ ] MoE fusion candidates remain inert until explicit parity tests exist.

## Track 6 - Remove Remaining Name-Only Gradient Fallback - COMPLETE

- [x] Replaced executor split-planner use of `base_param_from_grad_heuristic` with an exact precomputed parameter-gradient output map derived from `DslGradStore::param_names()`.
- [x] Kept the old `d_<base>` inference only as a private graph-compiler debug cross-check under tensor-kind diagnostics.
- [x] Deleted the public executor heuristic utility once no runtime caller remained.

Acceptance:

- [x] Runtime gradient routing never infers parameter gradients from `d_<base>` names alone.
- [x] Tensor-kind debug checks still catch accumulator and activation-gradient misclassification.
- [x] Existing CUDA graph split behavior is unchanged by using exact grad-store names for the split-planner tail.

## Track 7 - Final Validation And Cleanup

- [ ] Run required local gates: `make wheel-dev`, `make test-unit`, `make test-integration`, and no-GPU regression tests excluding `tests/test_distributed.py`.
- [ ] Run real-model 5-step acceptance queue:
  - [ ] `examples/sft/qwen3/qwen3-lora-fp8.yaml`
  - [ ] `examples/sft/qwen35/qwen35-text-lora-fp8.yaml`
  - [ ] `examples/sft/gemma4/gemma4-e2b-lora-fp8.yaml`
  - [ ] `examples/sft/gpt-oss/gptoss-lora-mxfp4.yaml`
  - [ ] `examples/sft/qwen36moe/qwen36moe-lora-fp8.yaml`
  - [ ] `examples/sft/qwen3moe/qwen3moe-nvfp4.yaml`
- [ ] Run selected 2-GPU DP and EP smoke cases with the new authoritative paths enabled.
- [ ] Refresh regression artifacts and locked baselines only after the new paths are stable.
- [ ] Update `REFACTOR.md` to mark the completion tracks closed and point to this document for final evidence.

Acceptance:

- [ ] No required path depends on scaffold-only, inert-only, or diagnostic-only implementation for the refactor goals.
- [ ] All remaining fallback paths are either explicit unsupported-mode fallbacks or documented operational safety fallbacks.
- [ ] `git status --short` is clean after final commits.

Order:

Track 6 - remove name-only grad fallback.
Track 4 - remove descriptor capability fallbacks.
Track 1 - finish FP8 MoE wgrad properly.
Track 2 - promote hook dispatch.
Track 3 - schema-driven allocation authoritative.
Track 5 - activate fusion rewrites.
Track 7 - final validation.
